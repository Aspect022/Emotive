"""
benchmark/models/arch_d.py
Architecture D — CogProfile-Net v6: ConvLSTM Topomap + EEGNet + Engagement Index.
Inspired by: mEBAL2 ConvLSTM (Paper 3), sensors-24 mid-fusion CNN (Paper 7).
Fastest arch (~2h per fold).  Strong sanity-check baseline.
"""
import torch
import torch.nn as nn
from benchmark.config import (
    N_CH, WIN, EMB_DIM, N_CLUSTERS, DROPOUT, N_BANDS, TOPO_GRID, TOPO_T_STEPS
)
from benchmark.models.shared import EDLHead, DECLayer


class ConvLSTMCell(nn.Module):
    def __init__(self, in_c: int, hid_c: int, k: int = 3):
        super().__init__()
        p = k // 2
        def _conv(i, o): return nn.Conv2d(i, o, k, padding=p)
        self.Wi = _conv(in_c,  hid_c); self.Ui = _conv(hid_c, hid_c)
        self.Wf = _conv(in_c,  hid_c); self.Uf = _conv(hid_c, hid_c)
        self.Wo = _conv(in_c,  hid_c); self.Uo = _conv(hid_c, hid_c)
        self.Wc = _conv(in_c,  hid_c); self.Uc = _conv(hid_c, hid_c)
        for m in [self.Ui, self.Uf, self.Uo, self.Uc]:
            m.bias = None

    def forward(self, x, h, c):
        i = torch.sigmoid(self.Wi(x) + self.Ui(h))
        f = torch.sigmoid(self.Wf(x) + self.Uf(h))
        o = torch.sigmoid(self.Wo(x) + self.Uo(h))
        g = torch.tanh(   self.Wc(x) + self.Uc(h))
        c = f * c + i * g
        return o * torch.tanh(c), c


class EEGNet(nn.Module):
    def __init__(self, n_ch=N_CH, n_time=WIN, emb=EMB_DIM, drop=DROPOUT):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 32, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(32), nn.ELU())
        self.spatial = nn.Sequential(
            nn.Conv2d(32, 32, (n_ch, 1), groups=32, bias=False),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.sep = nn.Sequential(
            nn.Conv2d(32, 64, (1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(64), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            d = torch.zeros(1, 1, n_ch, n_time)
            flat = self.sep(self.spatial(self.temporal(d))).flatten(1).shape[1]
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(flat, emb), nn.LayerNorm(emb))

    def forward(self, x):
        return self.proj(self.sep(self.spatial(self.temporal(x.unsqueeze(1)))))


class ArchD(nn.Module):
    """
    Branch A: ConvLSTM over topomap time sequence -> z_A (128)
    Branch B: EEGNet on raw EEG -> z_B (128)
    Branch C: Engagement index MLP -> z_C (32)
    Fusion:   concat + MLP
    """
    def __init__(self):
        super().__init__()
        # ConvLSTM on (T, 5, 8, 8)
        self.clstm = ConvLSTMCell(N_BANDS, 32)
        self.pool_topo = nn.Sequential(
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(32 * 16, EMB_DIM), nn.LayerNorm(EMB_DIM))

        # EEGNet
        self.eegnet = EEGNet()

        # Engagement index
        self.ei_mlp = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32))

        # Fusion
        self.fuse = nn.Sequential(
            nn.Linear(EMB_DIM * 2 + 32, EMB_DIM), nn.ReLU(), nn.LayerNorm(EMB_DIM))
        self.dec  = DECLayer()
        self.edl  = EDLHead()

    def forward(self, x_topo, x_raw, x_ei):
        B, T, C, H, W = x_topo.shape
        h = torch.zeros(B, 32, H, W, device=x_topo.device)
        c = torch.zeros(B, 32, H, W, device=x_topo.device)
        for t in range(T):
            h, c = self.clstm(x_topo[:, t], h, c)
        za = self.pool_topo(h)
        zb = self.eegnet(x_raw)
        zc = self.ei_mlp(x_ei)
        z  = self.fuse(torch.cat([za, zb, zc], 1))
        return {"z": z, "q": self.dec(z), **self.edl(z)}
