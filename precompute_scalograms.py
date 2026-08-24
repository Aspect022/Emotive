import sys, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DL = ROOT / "data" / "processed" / "dl_dataset"
OUT_DIR = ROOT / "data" / "processed" / "scalograms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SFREQ = 128.0
WIN = 128
STEP = 32
N_FREQ = 64
FREQ_MIN = 1.0
FREQ_MAX = 45.0
PAD = 32
T_PAD = WIN + 2 * PAD
W0 = 5.0

def make_subwindows(X_raw, y, pids):
    wins, labs, tids, sids = [], [], [], []
    for i, (trial, lab, pid) in enumerate(zip(X_raw, y, pids)):
        T = trial.shape[1]
        for s in range(0, T - WIN + 1, STEP):
            wins.append(trial[:, s: s + WIN])
            labs.append(lab)
            tids.append(i)
            sids.append(pid)
    return (
        np.stack(wins, axis=0).astype(np.float32),
        np.array(labs, dtype=np.int64),
        np.array(tids, dtype=np.int64),
        np.array(sids),
    )

def build_morlet_filter_bank():
    freqs = np.logspace(np.log10(FREQ_MIN), np.log10(FREQ_MAX), N_FREQ)
    filters = []
    for f in freqs:
        s = W0 / (2.0 * np.pi * f)
        M = min(int(6.0 * s * SFREQ) | 1, T_PAD)
        if M < 3:
            M = 3
        t = (np.arange(M) - (M - 1) / 2.0) / SFREQ
        t_scaled = t / s
        psi = (np.pi ** -0.25) * np.exp(1j * W0 * t_scaled) * np.exp(-0.5 * (t_scaled ** 2)) / np.sqrt(s)
        if len(psi) < T_PAD:
            pl = (T_PAD - len(psi)) // 2
            pr = T_PAD - len(psi) - pl
            psi = np.pad(psi, (pl, pr))
        else:
            st = (len(psi) - T_PAD) // 2
            psi = psi[st: st + T_PAD]
        filters.append(psi)
    filters_np = np.array(filters)
    filters_t = torch.tensor(filters_np, dtype=torch.complex64)
    return torch.fft.fft(filters_t, dim=-1)  # (N_FREQ, T_PAD)

def main():
    print("=" * 60)
    print("CogProfile-Net v2: High-Speed PyTorch Scalogram Precomputation")
    print("=" * 60)
    
    print("\n[1/4] Loading raw trials...")
    X_raw = np.load(DL / "X_raw.npy")
    y = np.load(DL / "y_task.npy")
    pids = np.load(DL / "participant_ids.npy", allow_pickle=True)
    print(f"      Loaded X_raw={X_raw.shape}, y={y.shape}, subjects={len(np.unique(pids))}")
    
    print("\n[2/4] Slicing into 1s sliding sub-windows (75% overlap)...")
    X_win, y_win, t_ids, s_ids = make_subwindows(X_raw, y, pids)
    N = len(X_win)
    print(f"      Total sub-windows: {N:,} (14 channels x 128 samples)")
    
    print("\n[3/4] Building Morlet CWT filter bank...")
    filters_fft = build_morlet_filter_bank()  # (64, 192)
    print(f"      Filter bank FFT shape: {filters_fft.shape}")
    
    out_path = OUT_DIR / "scalograms.npy"
    print(f"\n[4/4] Generating & saving {N:,} scalograms to memory-mapped {out_path}...")
    scal_mmap = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.float16, shape=(N, 14, N_FREQ, WIN)
    )
    
    BATCH = 256
    t0 = time.time()
    for i in range(0, N, BATCH):
        batch_np = X_win[i: i + BATCH]
        b_len = len(batch_np)
        batch_t = torch.tensor(batch_np, dtype=torch.float32)
        
        # Reflect pad along temporal dimension
        padded = F.pad(batch_t, (PAD, PAD), mode="reflect")  # (B, 14, 192)
        sig_fft = torch.fft.fft(padded, dim=-1)             # (B, 14, 192)
        
        # Convolve via FFT multiply
        # (B, 14, 1, 192) * (1, 1, 64, 192).conj() -> (B, 14, 64, 192)
        prod = sig_fft.unsqueeze(2) * filters_fft.unsqueeze(0).unsqueeze(0).conj()
        conv = torch.fft.ifft(prod, dim=-1)
        
        # Crop central window & compute power
        power = conv[:, :, :, PAD: PAD + WIN].abs().pow(2)  # (B, 14, 64, 128)
        power = torch.log1p(power)
        
        # Channel-frequency-wise min-max normalisation over time
        mn = power.amin(dim=-1, keepdim=True)
        mx = power.amax(dim=-1, keepdim=True)
        power = (power - mn) / torch.clamp(mx - mn, min=1e-8)
        
        scal_mmap[i: i + b_len] = power.to(torch.float16).cpu().numpy()
        
        if (i // BATCH) % 15 == 0 or (i + BATCH >= N):
            pct = 100.0 * (i + b_len) / N
            elapsed = time.time() - t0
            eta = (elapsed / max(i + b_len, 1)) * (N - (i + b_len))
            print(f"      [{pct:5.1f}%] {i + b_len:6,d}/{N:,d} windows processed | Elapsed: {elapsed:4.1f}s | ETA: {eta:4.1f}s")
            
    scal_mmap.flush()
    print(f"\n[Saved] Scalograms: {out_path} ({out_path.stat().st_size / (1024**3):.2f} GB)")
    
    # Save accompanying arrays
    np.save(str(OUT_DIR / "windows_raw.npy"), X_win)
    np.save(str(OUT_DIR / "labels.npy"), y_win)
    np.save(str(OUT_DIR / "trial_ids.npy"), t_ids)
    np.save(str(OUT_DIR / "subject_ids.npy"), s_ids)
    print(f"[Saved] Raw windows & labels in {OUT_DIR}")
    print(f"\n[COMPLETE] Total precomputation time: {(time.time() - t0):.1f} seconds ({(time.time() - t0)/60:.2f} min).")

if __name__ == "__main__":
    main()
