"""
scripts/08_log_metrics_during_iterations.py
Logs and plots Mean Squared Error (MSE) and residual convergence
across iterations for Landweber and NSIT algorithms.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solvers.operators import RadonOperator
from solvers.landweber import landweber_reconstruct
from solvers.nsit import nsit_reconstruct

PHANTOM_PATH = PROJECT_ROOT / "data" / "phantoms" / "phantom.npy"
SINO_PATH = PROJECT_ROOT / "data" / "projections" / "sinogram_noisy.npy"
VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


def log_iteration_convergence(landweber_iters: int = 50, nsit_iters: int = 15):
    phantom = np.load(PHANTOM_PATH)
    sinogram = np.load(SINO_PATH)
    clean_sino_path = PROJECT_ROOT / "data" / "projections" / "sinogram.npy"

    delta = None
    if clean_sino_path.exists():
        clean_sino = np.load(clean_sino_path)
        delta = float(np.linalg.norm(sinogram - clean_sino))

    operator = RadonOperator(image_size=phantom.shape[0], num_angles=sinogram.shape[1])

    print("[*] Tracking Landweber iteration MSE...")
    lw_result = landweber_reconstruct(
        sinogram=sinogram,
        operator=operator,
        max_iter=landweber_iters,
        ground_truth=phantom,
    )
    lw_mse = lw_result["mse_history"]

    print("[*] Tracking NSIT iteration MSE...")
    nsit_result = nsit_reconstruct(
        sinogram=sinogram,
        operator=operator,
        lambda0=10.0,
        q=0.7,
        max_iter=nsit_iters,
        delta=delta,
        ground_truth=phantom,
    )
    nsit_mse = nsit_result["mse_history"]

    # Plot convergence curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(range(1, len(lw_mse) + 1), lw_mse, "o-", color="tab:blue", label=f"Landweber (Final: {lw_mse[-1]:.4f})", markersize=3)
    ax1.set_title("Landweber MSE Convergence", fontsize=11)
    ax1.set_xlabel("Iteration", fontsize=10)
    ax1.set_ylabel("MSE vs Ground Truth", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(range(1, len(nsit_mse) + 1), nsit_mse, "s-", color="tab:red", label=f"NSIT (Final: {nsit_mse[-1]:.4f})", markersize=5)
    ax2.set_title("NSIT MSE Convergence", fontsize=11)
    ax2.set_xlabel("Iteration (Morozov Stopped)", fontsize=10)
    ax2.set_ylabel("MSE vs Ground Truth", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.suptitle("Iterative Regularization Convergence Analysis", fontsize=13)
    plt.tight_layout()
    out_path = VIS_DIR / "08_mse_convergence.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved convergence comparison plot to: {out_path}")


if __name__ == "__main__":
    log_iteration_convergence()
