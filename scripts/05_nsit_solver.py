"""
scripts/05_nsit_solver.py
Solves CT reconstruction using Non-Stationary Iterated Tikhonov (NSIT):
(A^T A + lambda_k * I) x_{k+1} = A^T b + lambda_k * x_k
lambda_k = lambda_0 * q^k
Equipped with Morozov's Discrepancy Principle stopping criterion:
Stop when ||A(x_k) - b||_2 <= delta (noise level).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solvers.operators import RadonOperator
from solvers.nsit import nsit_reconstruct
from evaluation.metrics import compute_metrics

RECON_DIR = PROJECT_ROOT / "outputs" / "reconstructions"
VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations"
RECON_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


def run_nsit_solver(lambda0: float = 10.0, q: float = 0.7, max_iter: int = 15):
    phantom_path = PROJECT_ROOT / "data" / "phantoms" / "phantom.npy"
    clean_sino_path = PROJECT_ROOT / "data" / "projections" / "sinogram.npy"
    noisy_sino_path = PROJECT_ROOT / "data" / "projections" / "sinogram_noisy.npy"

    if not phantom_path.exists() or not noisy_sino_path.exists():
        raise FileNotFoundError(
            "Required data files not found. Run scripts 01 and 02 first."
        )

    phantom = np.load(phantom_path)
    noisy_sino = np.load(noisy_sino_path)
    image_size = phantom.shape[0]
    num_angles = noisy_sino.shape[1]

    print(f"[*] Initializing Radon operator ({image_size}x{image_size}, {num_angles} angles)...")
    operator = RadonOperator(image_size=image_size, num_angles=num_angles)

    # Estimate noise norm delta for Morozov stopping criterion
    if clean_sino_path.exists():
        clean_sino = np.load(clean_sino_path)
        delta = float(np.linalg.norm(noisy_sino - clean_sino))
        print(f"[*] True noise norm delta = {delta:.2f} (used for Morozov discrepancy principle)")
    else:
        delta = None
        print("[!] Clean sinogram not found; Morozov criterion disabled (running fixed iters).")

    print(f"[*] Running NSIT solver (lambda0={lambda0}, q={q}, max_iter={max_iter})...")
    result = nsit_reconstruct(
        sinogram=noisy_sino,
        operator=operator,
        lambda0=lambda0,
        q=q,
        max_iter=max_iter,
        delta=delta,
        ground_truth=phantom,
    )

    recon = result["reconstruction"]
    runtime = result["runtime_sec"]
    residuals = result["residual_history"]
    mse_hist = result["mse_history"]
    stopped_morozov = result["stopped_by_morozov"]
    iters_run = result["iterations_run"]

    metrics = compute_metrics(phantom, recon)
    print(
        f"    [NSIT] Completed {iters_run} iters in {runtime:.2f}s "
        f"(Morozov stopping: {stopped_morozov}) | "
        f"MSE: {metrics['mse']:.4f} | PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}"
    )

    # Save outputs
    np.save(RECON_DIR / "nsit.npy", recon)
    plt.imsave(RECON_DIR / "nsit.png", recon, cmap="gray")
    print(f"[+] Saved NSIT reconstruction to: {RECON_DIR / 'nsit.png'}")

    # Visualization
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].imshow(phantom, cmap="gray")
    axs[0].set_title("Ground Truth")
    axs[0].axis("off")

    axs[1].imshow(recon, cmap="gray")
    axs[1].set_title(f"NSIT Recon ({iters_run} iters, PSNR: {metrics['psnr']:.1f} dB)")
    axs[1].axis("off")

    axs[2].plot(range(1, len(residuals) + 1), residuals, "o-", color="tab:red", label="Residual ||A(x_k) - b||")
    if delta is not None:
        axs[2].axhline(y=delta, color="gray", linestyle="--", label=f"Noise delta = {delta:.1f}")
    axs[2].set_title("Residual vs Iteration (Morozov Rule)")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Residual Norm")
    axs[2].legend()
    axs[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    vis_path = VIS_DIR / "nsit_visuals.png"
    plt.savefig(vis_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved NSIT visualization plot to: {vis_path}")

    return recon


if __name__ == "__main__":
    run_nsit_solver()
