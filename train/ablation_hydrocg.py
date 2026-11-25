
import pickle, torch, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train_hydrocg import ChaohuHydroDataset, make_loaders, CFG
from model_hydrocg import MetHydroCG, HydroConvectiveCausalGraph, TemporalEncoder
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')


with open('dataset_v2.pkl', 'rb') as f:
    dataset = pickle.load(f)
train_loader, val_loader, test_loader = make_loaders(dataset, CFG)
wq_scaler = dataset['wq_scaler']
wq_cols = dataset['wq_cols']
station_coords = torch.tensor(dataset['station_coords'], dtype=torch.float32)
N = len(dataset['stations'])
wq_dim = len(wq_cols)
met_dps = dataset['met'].shape[-1]


def evaluate_model(model, loader):
    model.eval()
    all_med, all_tgt = [], []
    with torch.no_grad():
        for batch in loader:
            wq_=batch['wq'].to(device);mt_=batch['met'].to(device);ws_=batch['wind_sin'].to(device);wc_=batch['wind_cos'].to(device);wsp_=batch['wind_speed'].to(device);mfc_=batch['met_fc'].to(device);preds,_,_=model(wq_,mt_,ws_,wc_,wsp_,mfc_)
            all_med.append(preds[1].cpu().numpy())
            all_tgt.append(batch['target'].cpu().numpy())
    med = np.concatenate(all_med, axis=0); tgt = np.concatenate(all_tgt, axis=0)
    def inv(arr):
        flat = arr.reshape(-1, N*wq_dim)
        return wq_scaler.inverse_transform(flat).reshape(-1, N, wq_dim)
    true_all = inv(tgt); med_all = inv(med)
    do_idx = CFG['do_idx']
    do_mae = mean_absolute_error(true_all[:,:,do_idx].ravel(), med_all[:,:,do_idx].ravel())
    do_rmse = np.sqrt(mean_squared_error(true_all[:,:,do_idx].ravel(), med_all[:,:,do_idx].ravel()))
    do_mape = np.mean(np.abs((true_all[:,:,do_idx]-med_all[:,:,do_idx])/(np.abs(true_all[:,:,do_idx])+1e-8)))*100
    ov_mae = mean_absolute_error(true_all.ravel(), med_all.ravel())
    return do_mae, do_rmse, do_mape, ov_mae

# ── 消融变体 ──
class MetHydroCG_NoHydro(MetHydroCG):

    def forward(self, wq, met, wind_sin, wind_cos, wind_speed, met_fc=None):
        B, T, N, Dd = wq.shape
        met_global = met.reshape(B, T, -1).unsqueeze(2).expand(-1,-1,N,-1)
        x = torch.cat([wq, met_global], dim=-1)
        x_flat = x.permute(0,2,1,3).reshape(B*N,T,-1)
        h = self.encoder(x_flat).reshape(B,N,-1)
        # 用静态图替代平流核
        W_last = torch.eye(N, device=wq.device).unsqueeze(0).expand(B,-1,-1)
        wq_flat = wq.permute(0,2,1,3).reshape(B,N,T*Dd)
        h_ar = self.ar_bridge(wq_flat)
        h_spatial = F.relu(torch.bmm(W_last, wq[:, -1, :, :]))
        h_spatial = self.gcn(h_spatial)
        met_last = met_global[:, -1, :, :].mean(dim=1)
        g = self.regime_gate(met_last)
        out = (1 - g) * h_ar + g * h_spatial
        feat = torch.cat([out, h], dim=-1)
        return [head(feat) for head in self.q_heads], g, W_last

class MetHydroCG_NoAR(MetHydroCG):

    def forward(self, wq, met, wind_sin, wind_cos, wind_speed, met_fc=None):
        B, T, N, Dd = wq.shape
        met_global = met.reshape(B, T, -1).unsqueeze(2).expand(-1,-1,N,-1)
        x = torch.cat([wq, met_global], dim=-1)
        x_flat = x.permute(0,2,1,3).reshape(B*N,T,-1)
        h = self.encoder(x_flat).reshape(B,N,-1)
        W_hydro = self.hydro_kernel(wind_speed, wind_sin, wind_cos)
        W_last = W_hydro[:, -1, :, :]
        h_spatial = F.relu(torch.bmm(W_last, wq[:, -1, :, :]))
        h_spatial = self.gcn(h_spatial)
        met_last = met_global[:, -1, :, :].mean(dim=1)
        g = self.regime_gate(met_last)
        out = h_spatial  # 无 AR
        feat = torch.cat([out, h], dim=-1)
        return [head(feat) for head in self.q_heads], g, W_last

class MetHydroCG_NoGate(MetHydroCG):

    def forward(self, wq, met, wind_sin, wind_cos, wind_speed, met_fc=None):
        B, T, N, Dd = wq.shape
        met_global = met.reshape(B, T, -1).unsqueeze(2).expand(-1,-1,N,-1)
        x = torch.cat([wq, met_global], dim=-1)
        x_flat = x.permute(0,2,1,3).reshape(B*N,T,-1)
        h = self.encoder(x_flat).reshape(B,N,-1)
        W_hydro = self.hydro_kernel(wind_speed, wind_sin, wind_cos)
        W_last = W_hydro[:, -1, :, :]
        wq_flat = wq.permute(0,2,1,3).reshape(B,N,T*Dd)
        h_ar = self.ar_bridge(wq_flat)
        h_spatial = F.relu(torch.bmm(W_last, wq[:, -1, :, :]))
        h_spatial = self.gcn(h_spatial)
        g = torch.full((B,1,1), 0.5, device=wq.device)
        out = (1 - g) * h_ar + g * h_spatial
        feat = torch.cat([out, h], dim=-1)
        return [head(feat) for head in self.q_heads], g.unsqueeze(0).unsqueeze(0), W_last

def run_one(name, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'], eta_min=1e-5)
    best_val = float('inf')
    best_state = None
    for epoch in range(1, CFG['epochs']+1):
        model.train()
        for batch in train_loader:
            wq_=batch['wq'].to(device);mt_=batch['met'].to(device);ws_=batch['wind_sin'].to(device);wc_=batch['wind_cos'].to(device);wsp_=batch['wind_speed'].to(device);mfc_=batch['met_fc'].to(device);preds,_,_=model(wq_,mt_,ws_,wc_,wsp_,mfc_)
            loss, _ = model.compute_loss(preds, batch['target'].to(device))
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            do_mae, _, _, ov_mae = evaluate_model(model, val_loader)
            if ov_mae < best_val:
                best_val = ov_mae; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            print(f'  [{name}] epoch {epoch}  val DO_MAE={do_mae:.4f}  val_Overall={ov_mae:.4f}')
    model.load_state_dict(best_state)
    do_mae, do_rmse, do_mape, ov_mae = evaluate_model(model, test_loader)
    print(f'  [{name}] TEST DO_MAE={do_mae:.4f}  Overall={ov_mae:.4f}')
    return do_mae, do_rmse, do_mape, ov_mae


common_kwargs = dict(n_nodes=N, wq_dim=wq_dim, met_dim_per_station=met_dps, n_met_stations=3, hidden_dim=CFG['hidden_dim'], seq_len=CFG['seq_len'], station_coords=station_coords)
experiments = {
    'A1: w/o HydroKernel': MetHydroCG_NoHydro(**common_kwargs).to(device),
    'A2: w/o AR-Bridge': MetHydroCG_NoAR(**common_kwargs).to(device),
    'A3: w/o RegimeGate': MetHydroCG_NoGate(**common_kwargs).to(device),
}

results = {}
for name, model in experiments.items():
    print(f'\n===== {name} =====')
    do_mae, do_rmse, do_mape, ov_mae = run_one(name, model)
    results[name] = {'DO_MAE': round(do_mae,4), 'DO_RMSE': round(do_rmse,4), 'DO_MAPE%': round(do_mape,2), 'Overall_MAE': round(ov_mae,4)}

df = pd.DataFrame(results).T
df.to_csv('ablation_hydrocg_results.csv')
print('\n\n消融实验汇总：')
print(df.to_string())
