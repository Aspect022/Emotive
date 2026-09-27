"""
benchmark/config.py  — All constants and CLI args.  Edit here only.
"""
import argparse

def get_args():
    p = argparse.ArgumentParser(description="CogProfile-Net benchmark")
    p.add_argument("--data_dir",  default=".",           help="X_raw.npy, y_task.npy, participant_ids.npy")
    p.add_argument("--cache_dir", default="./eeg_cache")
    p.add_argument("--out_dir",   default="./bench_out")
    p.add_argument("--archs",     nargs="+", default=["A","B","C","D"])
    p.add_argument("--folds",     type=int,   default=5)
    p.add_argument("--epochs",    type=int,   default=60)
    p.add_argument("--batch",     type=int,   default=64)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--workers",   type=int,   default=4)
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()

SFREQ      = 128
N_CH       = 14
WIN        = 128
STEP       = 32
N_FREQ     = 64
PAD        = 32
WAVELET    = "cmor1.5-1.0"
N_CLASSES  = 5

EEG_CH = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
TASK_NAMES = ["Arithmetic","Pattern","Reading","Attention","Spatial"]

BAND_HZ  = [(0.5,4.),(4.,8.),(8.,13.),(13.,30.),(30.,45.)]
N_BANDS  = len(BAND_HZ)

EMOTIV_TOPO = {
    "AF3":(2,0),"AF4":(5,0),"F7":(1,1),"F3":(3,1),"F4":(4,1),"F8":(6,1),
    "FC5":(2,2),"FC6":(5,2),"T7":(0,3),"T8":(7,3),
    "P7":(1,5),"P8":(6,5),"O1":(3,7),"O2":(4,7),
}

EMB_DIM       = 128
N_CLUSTERS    = 8
DROPOUT       = 0.4
GAT_HEADS     = 4
TOPO_GRID     = 8
TOPO_T_STEPS  = 16
DIM_NONLIN    = 56
DIM_NODE      = 13
DIM_RIEM      = 5 * (N_CH * (N_CH + 1) // 2)   # 525
