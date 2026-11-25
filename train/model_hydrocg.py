

import torch
import torch.nn as nn
import torch.nn.functional as F



class HydroConvectiveCausalGraph(nn.Module):

    def __init__(self, station_coords: torch.Tensor, alpha_wind=0.05):
        super().__init__()

        self.register_buffer("coords", station_coords)
        self.N = station_coords.shape[0]

        self.register_buffer("static_dist", torch.cdist(self.coords, self.coords, p=2))

        self.alpha_wind = nn.Parameter(torch.tensor(alpha_wind))
        self.sigma      = nn.Parameter(torch.tensor(0.5))

    def forward(self, wind_speed: torch.Tensor, wind_sin: torch.Tensor,
                wind_cos: torch.Tensor) -> torch.Tensor:

        B, T = wind_speed.shape


        u = -wind_speed * wind_sin   # (B, T)
        v = -wind_speed * wind_cos   # (B, T)
        U = torch.stack([u, v], dim=-1).unsqueeze(2)  # (B, T, 1, 2)


        R = self.coords.unsqueeze(0) - self.coords.unsqueeze(1)  # (N, N, 2)
        R = R.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)  # (B, T, N, N, 2)
        U_exp = U.unsqueeze(3).expand(-1, -1, self.N, self.N, -1)  # (B, T, N, N, 2)


        advection_drift = torch.sum((R - self.alpha_wind * U_exp) ** 2, dim=-1)
        # (B, T, N, N)


        sigma_pos = self.sigma.abs() + 1e-6
        W_hydro = torch.exp(-advection_drift / (2 * sigma_pos ** 2))


        W_hydro = F.normalize(W_hydro, p=1, dim=-1)


        eye = torch.eye(self.N, device=W_hydro.device).unsqueeze(0).unsqueeze(0)
        W_hydro = W_hydro * 0.9 + eye * 0.1

        return W_hydro


class HydroRegimeGating(nn.Module):


    def __init__(self, met_dim: int, hidden=16):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(met_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, met_current: torch.Tensor) -> torch.Tensor:

        g = self.gate_mlp(met_current)  # (B, 1)
        return g.unsqueeze(-1)           # (B, 1, 1)



class TemporalEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers,
                          batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out, _ = self.gru(x)               # (B*N, T, H*2)
        h_last = out[:, -1, :]             # (B*N, H*2)
        h = self.proj(h_last)               # (B*N, H)
        return self.norm(h)



class MetHydroCG(nn.Module):
    def __init__(self,
                 n_nodes: int = 12,
                 wq_dim: int = 5,
                 met_dim_per_station: int = 7,
                 n_met_stations: int = 3,
                 hidden_dim: int = 64,
                 seq_len: int = 30,
                 station_coords: torch.Tensor = None):
        super().__init__()
        self.n_nodes = n_nodes
        self.wq_dim  = wq_dim
        self.hidden  = hidden_dim
        self.seq_len = seq_len

        in_dim   = wq_dim + met_dim_per_station * n_met_stations  # 5 + 21 = 26
        met_dim  = met_dim_per_station * n_met_stations            # 21


        self.encoder = TemporalEncoder(in_dim, hidden_dim)


        if station_coords is None:
            station_coords = torch.randn(n_nodes, 2)
        self.hydro_kernel = HydroConvectiveCausalGraph(station_coords)

        self.regime_gate = HydroRegimeGating(met_dim)


        self.ar_bridge = nn.Linear(seq_len * wq_dim, wq_dim)


        self.gcn = nn.Sequential(
            nn.Linear(wq_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, wq_dim),
        )

        self.quantiles = (0.1, 0.5, 0.9)
        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(wq_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, wq_dim),
            ) for _ in range(3)
        ])

    def forward(self, wq: torch.Tensor, met: torch.Tensor,
                wind_sin: torch.Tensor, wind_cos: torch.Tensor,
                wind_speed: torch.Tensor, met_fc: torch.Tensor = None):

        B, T, N, D = wq.shape
        met_dim_total = met.shape[-1] * met.shape[2]


        met_global = met.reshape(B, T, -1)                     # (B, T, met_total)
        met_node   = met_global.unsqueeze(2).expand(-1, -1, N, -1)
        x = torch.cat([wq, met_node], dim=-1)                  # (B, T, N, in_dim)


        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # (B*N, T, in_dim)
        h = self.encoder(x_flat)                                # (B*N, H)
        h = h.reshape(B, N, -1)                                 # (B, N, H)


        W_hydro = self.hydro_kernel(wind_speed, wind_sin, wind_cos)  # (B, T, N, N)
        W_last  = W_hydro[:, -1, :, :]                                 # (B, N, N)


        wq_flat = wq.permute(0, 2, 1, 3).reshape(B, N, T * D)  # (B, N, T*D)
        h_ar = self.ar_bridge(wq_flat)                           # (B, N, wq_dim)


        h_spatial = F.relu(torch.bmm(W_last, wq[:, -1, :, :]))   # (B, N, wq_dim)
        h_spatial = self.gcn(h_spatial)                            # (B, N, wq_dim)


        met_last = met_global[:, -1, :]                            # (B, met_total)
        g = self.regime_gate(met_last)                             # (B, 1, 1)
        out = (1 - g) * h_ar + g * h_spatial                       # (B, N, wq_dim)


        feat = torch.cat([out, h], dim=-1)                         # (B, N, wq_dim+H)
        preds = [head(feat) for head in self.q_heads]

        return preds, g, W_last

    def compute_loss(self, preds, targets):

        q_loss = torch.tensor(0.0, device=targets.device)
        for q_val, pred in zip(self.quantiles, preds):
            err = targets - pred
            q_loss += torch.where(err >= 0,
                                  q_val * err,
                                  (q_val - 1) * err).mean()
        return q_loss, {"q_loss": q_loss.item()}
