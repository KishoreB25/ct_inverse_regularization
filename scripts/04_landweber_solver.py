"""
scripts/04_landweber_solver.py
Solves CT reconstruction using Landweber Iterative Regularization:
x_{k+1} = x_k + omega * A^T (b - A(x_k))
Iteration count serves as the regularization parameter (stopping early to avoid noise amplification).
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
from evaluation.metrics import compute_metrics

RECON_DIR = PROJECT_ROOT / "outputs" / "reconstructions"
VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations"
RECON_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


def run_landweber_solver(iterations: int = 50, omega: float = None):
    phantom_path = PROJECT_ROOT / "data" / "phantoms" / "phantom.npy"
    sino_path = PROJECT_ROOT / "data" / "projections" / "sinogram_noisy.npy"

    if not phantom_path.exists() or not sino_path.exists():
        raise FileNotFoundError(
            "Required data files not found. Run scripts 01 and 02 first."
        )

    phantom = np.load(phantom_path)
    noisy_sino = np.load(sino_path)
    image_size = phantom.shape[0]
    num_angles = noisy_sino.shape[1]

    print(f"[*] Initializing Radon operator ({image_size}x{image_size}, {num_angles} angles)...")
    operator = RadonOperator(image_size=image_size, num_angles=num_angles)

    print(f"[*] Running Landweber solver ({iterations} iterations)...")
    result = landweber_reconstruct(
        sinogram=noisy_sino,
        operator=operator,
        max_iter=iterations,
        omega=omega,
        ground_truth=phantom,
    )

    recon = result["reconstruction"]
    mse_history = result["mse_history"]
    runtime = result["runtime_sec"]
    used_omega = result["omega"]

    metrics = compute_metrics(phantom, recon)
    print(
        f"    [Landweber] Time: {runtime:.2f}s | Step omega: {used_omega:.5f} | "
        f"MSE: {metrics['mse']:.4f} | PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}"
    )

    # Save outputs
    np.save(RECON_DIR / "landweber.npy", recon)
    plt.imsave(RECON_DIR / "landweber.png", recon, cmap="gray")
    np.save(PROJECT_ROOT / "outputs" / "mse_landweber.npy", mse_history)
    print(f"[+] Saved reconstruction to: {RECON_DIR / 'landweber.png'}")

    # Visualization
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    axs[0].imshow(phantom, cmap="gray")
    axs[0].set_title("Ground Truth")
    axs[0].axis("off")

    axs[1].imshow(recon, cmap="gray")
    axs[1].set_title(f"Landweber (50 iters, PSNR: {metrics['psnr']:.1f} dB)")
    axs[1].axis("off")

    axs[2].plot(range(1, len(mse_history) + 1), mse_history, color="tab:blue", lw=2)
    axs[2].set_title("MSE vs Iteration")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("Mean Squared Error")
    axs[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    vis_path = VIS_DIR / "landweber_visuals.png"
    plt.savefig(vis_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved visualization plot to: {vis_path}")

    return recon


if __name__ == "__main__":
    run_landweber_solver()
