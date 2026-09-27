#!/usr/bin/env python3
"""
start.py — Single-command launcher for CogProfile-Net benchmark.

Usage:
    python start.py
    python start.py --archs A B --epochs 30
    python start.py --data_dir /data/eeg --folds 5

What this does (fully automated):
  1. Creates a Python venv at ./venv  (skips if already exists)
  2. Detects your CUDA version and installs the matching PyTorch wheel
  3. Installs all other requirements from requirements.txt
  4. Creates logs/<timestamp>.log
  5. Runs the benchmark — every line printed to terminal AND written to the log
"""

import os
import re
import sys
import subprocess
import platform
import shutil
import threading
import argparse
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.resolve()
VENV_DIR  = ROOT / "venv"
LOGS_DIR  = ROOT / "logs"
REQ_FILE  = ROOT / "requirements.txt"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE  = LOGS_DIR / f"bench_{TIMESTAMP}.log"

IS_WIN = platform.system() == "Windows"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if IS_WIN else "bin/python")
VENV_PIP    = VENV_DIR / ("Scripts/pip.exe"    if IS_WIN else "bin/pip")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _banner(msg: str):
    w = 65
    print(f"\n{'='*w}")
    print(f"  {msg}")
    print(f"{'='*w}")


def _run(cmd, **kwargs):
    """Run a command, raise on failure."""
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(f"\n[ERROR] Command failed: {' '.join(str(c) for c in cmd)}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Create venv
# ─────────────────────────────────────────────────────────────────────────────
def ensure_venv():
    if VENV_PYTHON.exists():
        print(f"  [venv] Found existing venv at {VENV_DIR}")
        return
    _banner(f"Creating virtual environment at {VENV_DIR}")
    _run([sys.executable, "-m", "venv", str(VENV_DIR)])
    print(f"  [venv] Created successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Detect CUDA and install PyTorch
# ─────────────────────────────────────────────────────────────────────────────
def _detect_cuda_tag() -> str:
    """
    Returns the PyTorch CUDA index tag, e.g. 'cu128', 'cu121', or 'cpu'.
    Tries nvidia-smi first, then nvcc, then falls back to cpu.
    """
    # Try nvidia-smi
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.check_output(
                [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, text=True).strip().split("\n")[0]
            # Driver version >= 570 → CUDA 12.8 capable (RTX 50 / Blackwell)
            # Driver version >= 550 → CUDA 12.4
            # Driver version >= 528 → CUDA 12.1
            major = int(out.split(".")[0])
            if major >= 570:  return "cu128"
            if major >= 550:  return "cu124"
            if major >= 528:  return "cu121"
            return "cu118"
        except Exception:
            pass

    # Try nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.check_output(
                [nvcc, "--version"], stderr=subprocess.DEVNULL, text=True)
            m = re.search(r"release (\d+)\.(\d+)", out)
            if m:
                major, minor = int(m.group(1)), int(m.group(2))
                if (major, minor) >= (12, 8): return "cu128"
                if (major, minor) >= (12, 4): return "cu124"
                if (major, minor) >= (12, 1): return "cu121"
                if (major, minor) >= (11, 8): return "cu118"
        except Exception:
            pass

    print("  [cuda] No GPU detected — installing CPU-only PyTorch.")
    return "cpu"


# PyTorch index URLs per CUDA tag
TORCH_INDEX = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cpu":   "https://download.pytorch.org/whl/cpu",
}


def install_packages():
    # Check if torch already installed in venv
    check = subprocess.run(
        [str(VENV_PYTHON), "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True)
    if check.returncode == 0:
        print(f"  [deps] PyTorch {check.stdout.strip()} already installed — skipping.")
    else:
        cuda_tag = _detect_cuda_tag()
        idx_url  = TORCH_INDEX[cuda_tag]
        _banner(f"Installing PyTorch  [{cuda_tag}]  from {idx_url}")
        _run([str(VENV_PIP), "install", "--upgrade", "pip"], capture_output=True)
        _run([str(VENV_PIP), "install",
              "torch", "torchvision",
              "--index-url", idx_url])
        print(f"  [deps] PyTorch installed ({cuda_tag}).")

    # Install the rest of requirements.txt
    _banner("Installing requirements.txt")
    _run([str(VENV_PIP), "install", "-r", str(REQ_FILE)])
    print("  [deps] All packages installed.")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Tee runner: stream subprocess output to terminal + log file
# ─────────────────────────────────────────────────────────────────────────────
def _stream_tee(proc, log_fh):
    """
    Reads stdout from proc line-by-line.
    Writes each line to sys.stdout (terminal) AND log_fh simultaneously.
    Blocks until the process ends.
    """
    def _pump(stream, label):
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fh.write(line)
            log_fh.flush()

    t_out = threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, "stdout"), daemon=True)
    t_out.start(); t_err.start()
    proc.wait()
    t_out.join(); t_err.join()


def run_benchmark(extra_args):
    LOGS_DIR.mkdir(exist_ok=True)
    _banner(f"Starting benchmark  →  log: logs/{LOG_FILE.name}")

    cmd = [str(VENV_PYTHON), "-m", "benchmark.run"] + extra_args

    header = (
        f"{'='*65}\n"
        f"  CogProfile-Net Benchmark Log\n"
        f"  Started : {datetime.now().isoformat()}\n"
        f"  Command : {' '.join(cmd)}\n"
        f"  Machine : {platform.node()}  ({platform.system()} {platform.release()})\n"
        f"{'='*65}\n\n"
    )

    with open(LOG_FILE, "w", encoding="utf-8") as log_fh:
        log_fh.write(header)
        sys.stdout.write(header)
        sys.stdout.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _stream_tee(proc, log_fh)

        footer = (
            f"\n{'='*65}\n"
            f"  Finished : {datetime.now().isoformat()}\n"
            f"  Exit code: {proc.returncode}\n"
            f"{'='*65}\n"
        )
        log_fh.write(footer)
        sys.stdout.write(footer)

    if proc.returncode != 0:
        sys.exit(f"\n[ERROR] Benchmark exited with code {proc.returncode}. "
                 f"See {LOG_FILE} for details.")

    print(f"\n  Log saved: {LOG_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Collect any extra args to pass through to benchmark/run.py
    # Strip '--' separator if user provides it
    extra = sys.argv[1:]

    _banner("CogProfile-Net  —  Environment Bootstrap")
    print(f"  Root    : {ROOT}")
    print(f"  Venv    : {VENV_DIR}")
    print(f"  Log     : {LOG_FILE}")
    print(f"  Python  : {sys.version.split()[0]}")

    ensure_venv()
    install_packages()
    run_benchmark(extra)


if __name__ == "__main__":
    main()
