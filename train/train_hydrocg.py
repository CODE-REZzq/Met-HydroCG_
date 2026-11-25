

import argparse, os, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_preprocessing_v2 import load_and_preprocess
from model_hydrocg import MetHydroCG


CFG = dict(
    seq_len         = 30,
    pred_len        = 1,
    batch_size      = 16,
    epochs          = 50,
    lr              = 1e-3,
    hidden_dim      = 64,
    train_ratio     = 0.7,
    val_ratio       = 0.15,
    data_dir = '.',
    save_path       = "met_hydrocg.pt",
    do_idx          = 2,
    wind_threshold  = 5.0,
)


# ── Dataset ────────────────────────────────────────────────────────────────
class ChaohuHydroDataset(Dataset):
    def __init__(self, wq, met, wind_sincos, wind_speed, seq_len, pred_len):
        self.wq          = torch.tensor(wq, dtype=torch.float32)
        self.met         = torch.tensor(met, dtype=torch.float32)
        self.wind_sincos = torch.tensor(wind_sincos, dtype=torch.float32)
        self.wind_speed  = torch.tensor(wind_speed, dtype=torch.float32)
        self.seq_len     = seq_len
        self.pred_len    = pred_len
        self.T           = wq.shape[0]

    def __len__(self):
        return self.T - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s = idx
        e = idx + self.seq_len
        t = e + self.pred_len - 1
        return {
            "wq":           self.wq[s:e],
            "met":          self.met[s:e],
            "wind_sin":     self.wind_sincos[s:e, 0],
            "wind_cos":     self.wind_sincos[s:e, 1],
            "wind_speed":   self.wind_speed[s:e],
            "target":       self.wq[t],
            "met_fc":       self.met[t].reshape(-1),
            "wind_speed_t": self.wind_speed[t],
        }


def make_loaders(dataset, cfg):
    T = len(dataset["wq"])
    n_train = int(T * cfg["train_ratio"])
    n_val   = int(T * cfg["val_ratio"])

    def make_ds(start, end):
        return ChaohuHydroDataset(
            dataset["wq"][start:end],
            dataset["met"][start:end],
            dataset["wind_sincos"][start:end],
            dataset["wind_speed_raw"][start:end],
            cfg["seq_len"], cfg["pred_len"],
        )

    train_ds = make_ds(0, n_train)
    val_ds   = make_ds(n_train, n_train + n_val)
    test_ds  = make_ds(n_train + n_val, T)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False)
    return train_loader, val_loader, test_loader


# ── 评估函数 ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, cfg, wq_scaler, wq_cols=None,
             subset_name="full"):
    model.eval()
    all_preds = {q: [] for q in (0.1, 0.5, 0.9)}
    all_targets = []
    all_gates = []
    all_ws = []  # wind speeds for stratification

    with torch.no_grad():
        for batch in loader:
            wq_       = batch["wq"].to(device)
            met_      = batch["met"].to(device)
            wind_sin  = batch["wind_sin"].to(device)
            wind_cos  = batch["wind_cos"].to(device)
            wind_spd  = batch["wind_speed"].to(device)
            target    = batch["target"].to(device)
            met_fc    = batch["met_fc"].to(device)

            preds, g, W_last = model(wq_, met_, wind_sin, wind_cos, wind_spd, met_fc)

            for i, q in enumerate((0.1, 0.5, 0.9)):
                all_preds[q].append(preds[i].cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_gates.append(g.cpu().numpy())
            all_ws.append(batch["wind_speed_t"].cpu().numpy())

    targets = np.concatenate(all_targets, axis=0)
    preds_q = {q: np.concatenate(all_preds[q], axis=0) for q in (0.1, 0.5, 0.9)}
    gates   = np.concatenate(all_gates, axis=0).mean()
    ws_arr  = np.concatenate(all_ws, axis=0)


    N, D = targets.shape[1], targets.shape[2]
    if wq_cols is None:
        wq_cols = [f"Var{i}" for i in range(D)]

    def inv_all(arr):
        flat = arr.reshape(-1, N * D)
        inv  = wq_scaler.inverse_transform(flat)
        return inv.reshape(-1, N, D)

    true_all = inv_all(targets)
    med_all  = inv_all(preds_q[0.5])
    low_all  = inv_all(preds_q[0.1])
    high_all = inv_all(preds_q[0.9])


    if np.isnan(med_all).any():
        med_all  = np.nan_to_num(med_all, nan=0.0)
        low_all  = np.nan_to_num(low_all, nan=0.0)
        high_all = np.nan_to_num(high_all, nan=0.0)

    #
    overall_mae  = mean_absolute_error(true_all.ravel(), med_all.ravel())
    overall_rmse = np.sqrt(mean_squared_error(true_all.ravel(), med_all.ravel()))
    overall_mape = np.mean(np.abs((true_all - med_all) / (true_all + 1e-8))) * 100


    per_var = {}
    for d in range(D):
        t_d = true_all[:, :, d].ravel()
        m_d = med_all[:, :, d].ravel()
        per_var[wq_cols[d]] = {
            "mae":  mean_absolute_error(t_d, m_d),
            "rmse": np.sqrt(mean_squared_error(t_d, m_d)),
            "mape": np.mean(np.abs((t_d - m_d) / (t_d + 1e-8))) * 100,
        }

    do_name = wq_cols[cfg["do_idx"]]
    do_m = per_var[do_name]

    print(f"  [{subset_name}] {do_name} MAE={do_m['mae']:.4f}  "
          f"RMSE={do_m['rmse']:.4f}  MAPE={do_m['mape']:.2f}%  "
          f"总体MAE={overall_mae:.4f}  Gate={gates:.3f}")

    return {
        "overall_mae": overall_mae, "overall_rmse": overall_rmse,
        "overall_mape": overall_mape,
        "do_mae": do_m["mae"], "do_rmse": do_m["rmse"], "do_mape": do_m["mape"],
        "per_var": per_var, "gate_mean": gates,
        "do_true": true_all[:, :, cfg["do_idx"]],
        "do_med":  med_all[:, :, cfg["do_idx"]],
        "do_low":  low_all[:, :, cfg["do_idx"]],
        "do_high": high_all[:, :, cfg["do_idx"]],
        "ws_arr":  ws_arr,
    }


def evaluate_stratified(model, loader, device, cfg, wq_scaler, wq_cols, ws_arr):

    thresh = cfg["wind_threshold"]


    pass  # see train() for stratified evaluation


#
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    pkl = os.path.join(cfg["data_dir"], "dataset_v2.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            dataset = pickle.load(f)
        print("加载已有数据集")
    else:
        dataset = load_and_preprocess(cfg["data_dir"])

    train_loader, val_loader, test_loader = make_loaders(dataset, cfg)
    station_coords = torch.tensor(dataset["station_coords"], dtype=torch.float32)
    wq_cols = dataset["wq_cols"]
    wq_scaler = dataset["wq_scaler"]


    model = MetHydroCG(
        n_nodes=len(dataset["stations"]),
        wq_dim=len(wq_cols),
        met_dim_per_station=dataset["met"].shape[-1],
        n_met_stations=3,
        hidden_dim=cfg["hidden_dim"],
        seq_len=cfg["seq_len"],
        station_coords=station_coords,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-5)

    best_val = float("inf")
    best_state = None
    train_losses = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            wq_       = batch["wq"].to(device)
            met_      = batch["met"].to(device)
            wind_sin  = batch["wind_sin"].to(device)
            wind_cos  = batch["wind_cos"].to(device)
            wind_spd  = batch["wind_speed"].to(device)
            target    = batch["target"].to(device)
            met_fc    = batch["met_fc"].to(device)

            preds, _, _ = model(wq_, met_, wind_sin, wind_cos, wind_spd, met_fc)
            loss, info  = model.compute_loss(preds, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"\nEpoch {epoch}/{cfg['epochs']}  loss={avg_loss:.4f}")
            res = evaluate(model, val_loader, device, cfg, wq_scaler, wq_cols, "val")
            if res["overall_mae"] < best_val:
                best_val = res["overall_mae"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, cfg["save_path"])
                print(f"  ✓ 保存最优模型 (val={best_val:.4f})")


    print("\n── 测试集全量评估 ──")
    model.load_state_dict(torch.load(cfg["save_path"], map_location=device))
    test_res = evaluate(model, test_loader, device, cfg, wq_scaler, wq_cols, "test")


    print("\n── 风速分层解耦验证 ──")
    thresh = cfg["wind_threshold"]

    model.eval()
    samples = []
    with torch.no_grad():
        for batch in test_loader:
            wq_       = batch["wq"].to(device)
            met_      = batch["met"].to(device)
            wind_sin  = batch["wind_sin"].to(device)
            wind_cos  = batch["wind_cos"].to(device)
            wind_spd  = batch["wind_speed"].to(device)
            target    = batch["target"].to(device)
            met_fc    = batch["met_fc"].to(device)

            preds, _, _ = model(wq_, met_, wind_sin, wind_cos, wind_spd, met_fc)
            tgt = target.cpu().numpy()
            med = preds[1].cpu().numpy()  # 0.5 quantile
            ws_t = batch["wind_speed_t"].cpu().numpy()

            B = tgt.shape[0]
            for b in range(B):
                samples.append({
                    "ws": ws_t[b],
                    "tgt_do": tgt[b, :, cfg["do_idx"]],
                    "pred_do": med[b, :, cfg["do_idx"]],
                })


    normal_ws = [s for s in samples if s["ws"] < thresh]
    extreme_ws = [s for s in samples if s["ws"] >= thresh]

    def compute_do_mae(group):
        if len(group) == 0:
            return 0, 0, 0, 0
        tgt = np.concatenate([s["tgt_do"] for s in group])
        pred = np.concatenate([s["pred_do"] for s in group])
        # 反归一化
        N, D = test_res["do_true"].shape[1], len(wq_cols)
        # ...简化: 直接用归一化后值算相对 MAE
        mae = mean_absolute_error(tgt, pred)
        rmse = np.sqrt(mean_squared_error(tgt, pred))
        mape = np.mean(np.abs((tgt - pred) / (np.abs(tgt) + 1e-8))) * 100
        return mae, rmse, mape, len(group)

    n_mae, n_rmse, n_mape, n_cnt = compute_do_mae(normal_ws)
    e_mae, e_rmse, e_mape, e_cnt = compute_do_mae(extreme_ws)

    print(f"\n  平水期 (V<{thresh}m/s, n={n_cnt}): DO MAE={n_mae:.4f}  RMSE={n_rmse:.4f}  MAPE={n_mape:.2f}%")
    print(f"  极端期 (V>={thresh}m/s, n={e_cnt}): DO MAE={e_mae:.4f}  RMSE={e_rmse:.4f}  MAPE={e_mape:.2f}%")
    if n_mae > 0 and e_mae > 0:
        print(f"  极端/平水 MAE 比: {e_mae/n_mae:.2f}x  "
              f"({'✓ 物理鲁棒' if e_mae/n_mae < 1.5 else '⚠ 极端期退化'} )")


    plot_results(train_losses, test_res, cfg, n_mae, e_mae, n_cnt, e_cnt)
    return model, test_res


def plot_results(train_losses, test_res, cfg, n_mae, e_mae, n_cnt, e_cnt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(train_losses)
    axes[0].set_title("Training Loss (Q-Loss only)")
    axes[0].set_xlabel("Epoch")

    ax = axes[1]
    n_show = min(100, len(test_res["do_true"]))
    ax.plot(test_res["do_true"][:n_show, 0], 'k-', label="True DO", linewidth=1)
    ax.plot(test_res["do_med"][:n_show, 0], 'b-', label="Pred (median)", linewidth=0.8)
    ax.fill_between(range(n_show),
                    test_res["do_low"][:n_show, 0],
                    test_res["do_high"][:n_show, 0],
                    alpha=0.2, color='blue')
    ax.set_title(f"DO Prediction (Station 0)\nDO MAE={test_res['do_mae']:.3f}")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.bar(["Normal", "Extreme"], [n_mae, e_mae],
           color=["steelblue", "tomato"], alpha=0.7)
    ax.set_title(f"DO MAE by Wind Speed\n(V<{cfg['wind_threshold']} vs >={cfg['wind_threshold']} m/s)")
    for i, (v, cnt) in enumerate([(n_mae, n_cnt), (e_mae, e_cnt)]):
        ax.text(i, v + 0.001, f"{v:.4f}\n(n={cnt})", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("results_hydrocg.png", dpi=150)
    print("可视化已保存至 results_hydrocg.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="仅评估")
    args = parser.parse_args()

    if args.eval:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(CFG["data_dir"], "dataset_v2.pkl"), "rb") as f:
            dataset = pickle.load(f)
        _, _, test_loader = make_loaders(dataset, CFG)
        station_coords = torch.tensor(dataset["station_coords"], dtype=torch.float32)
        model = MetHydroCG(
            n_nodes=len(dataset["stations"]),
            wq_dim=len(dataset["wq_cols"]),
            met_dim_per_station=dataset["met"].shape[-1],
            n_met_stations=3,
            hidden_dim=CFG["hidden_dim"],
            seq_len=CFG["seq_len"],
            station_coords=station_coords,
        ).to(device)
        model.load_state_dict(torch.load(CFG["save_path"], map_location=device))
        print("── 测试集评估 ──")
        evaluate(model, test_loader, device, CFG, dataset["wq_scaler"],
                 dataset["wq_cols"], "test")
    else:
        train(CFG)
