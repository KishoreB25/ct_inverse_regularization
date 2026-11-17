"""
scripts/09_runtime_analysis.py
Benchmarks computation times across reconstruction algorithms:
FBP, Tikhonov Regularization, Landweber Iteration, and NSIT.
"""

import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solvers.operators import RadonOperator
from solvers.tikhonov import tikhonov_reconstruct
from solvers.landweber import landweber_reconstruct
from solvers.nsit import nsit_reconstruct

PHANTOM_PATH = PROJECT_ROOT / "data" / "phantoms" / "phantom.npy"
SINO_PATH = PROJECT_ROOT / "data" / "projections" / "sinogram_noisy.npy"
VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_runtimes():
    phantom = np.load(PHANTOM_PATH)
    sinogram = np.load(SINO_PATH)
    clean_sino_path = PROJECT_ROOT / "data" / "projections" / "sinogram.npy"

    delta = None
    if clean_sino_path.exists():
        clean_sino = np.load(clean_sino_path)
        delta = float(np.linalg.norm(sinogram - clean_sino))

    operator = RadonOperator(image_size=phantom.shape[0], num_angles=sinogram.shape[1])
    runtimes = {}

    # 1. FBP
    print("[*] Timing FBP baseline...")
    t0 = time.time()
    _ = operator.fbp(sinogram)
    runtimes["FBP"] = time.time() - t0

    # 2. Tikhonov
    print("[*] Timing Tikhonov (CG)...")
    t0 = time.time()
    _ = tikhonov_reconstruct(sinogram, operator, lam=10.0)
    runtimes["Tikhonov (CG)"] = time.time() - t0

    # 3. Landweber
    print("[*] Timing Landweber (50 iterations)...")
    t0 = time.time()
    _ = landweber_reconstruct(sinogram, operator, max_iter=50)
    runtimes["Landweber (50 iters)"] = time.time() - t0

    # 4. NSIT
    print("[*] Timing NSIT (Morozov stopped)...")
    t0 = time.time()
    _ = nsit_reconstruct(sinogram, operator, lambda0=10.0, q=0.7, max_iter=15, delta=delta)
    runtimes["NSIT (Morozov)"] = time.time() - t0

    print("\n--- Runtime Benchmark Results ---")
    for method, dur in runtimes.items():
        print(f"{method:<22}: {dur:.4f} seconds")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    methods = list(runtimes.keys())
    times = list(runtimes.values())
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]

    bars = ax.barh(methods, times, color=colors, height=0.55)
    ax.set_xlabel("Computation Time (seconds)", fontsize=11)
    ax.set_title("Reconstruction Algorithm Runtime Benchmark", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.05 * max(times), bar.get_y() + bar.get_height() / 2, f"{w:.2f}s", va="center", fontsize=10)

    ax.set_xlim(0, max(times) * 1.2)
    plt.tight_layout()
    out_path = VIS_DIR / "09_runtime_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved runtime benchmark plot to: {out_path}")

    return runtimes


if __name__ == "__main__":
    benchmark_runtimes()
