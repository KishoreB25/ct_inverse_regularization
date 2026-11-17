"""
scripts/03_tikhonov_solver.py
Solves CT reconstruction using Tikhonov Regularization:
min_x ||A x - b||_2^2 + lambda * ||x||_2^2
Equivalently solves the regularized normal equations:
(A^T A + lambda * I) x = A^T b
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solvers.operators import RadonOperator
from solvers.tikhonov import tikhonov_reconstruct
from evaluation.metrics import compute_metrics

RECON_DIR = PROJECT_ROOT / "outputs" / "reconstructions"
VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations"
RECON_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


def run_tikhonov_solver(lam: float = 10.0):
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

    print(f"[*] Running FBP baseline...")
    fbp_recon = operator.fbp(noisy_sino)
    fbp_metrics = compute_metrics(phantom, fbp_recon)
    print(f"    [FBP] MSE: {fbp_metrics['mse']:.4f} | PSNR: {fbp_metrics['psnr']:.2f} dB | SSIM: {fbp_metrics['ssim']:.4f}")

    print(f"[*] Solving Tikhonov regularized system (lambda={lam})...")
    result = tikhonov_reconstruct(noisy_sino, operator, lam=lam)
    tik_recon = result["reconstruction"]
    tik_metrics = compute_metrics(phantom, tik_recon)
    print(f"    [Tikhonov] MSE: {tik_metrics['mse']:.4f} | PSNR: {tik_metrics['psnr']:.2f} dB | SSIM: {tik_metrics['ssim']:.4f} | Runtime: {result['runtime_sec']:.2f}s")

    # Save arrays and images
    np.save(RECON_DIR / "tikhonov.npy", tik_recon)
    plt.imsave(RECON_DIR / "tikhonov.png", tik_recon, cmap="gray")
    np.save(RECON_DIR / "fbp.npy", fbp_recon)
    plt.imsave(RECON_DIR / "fbp.png", fbp_recon, cmap="gray")
    print(f"[+] Saved Tikhonov reconstruction to: {RECON_DIR / 'tikhonov.png'}")

    # Visual comparison figure
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(phantom, cmap="gray")
    axs[0].set_title("Ground Truth")
    axs[0].axis("off")

    axs[1].imshow(fbp_recon, cmap="gray")
    axs[1].set_title(f"FBP Baseline (PSNR: {fbp_metrics['psnr']:.1f} dB)")
    axs[1].axis("off")

    axs[2].imshow(tik_recon, cmap="gray")
    axs[2].set_title(f"Tikhonov (λ={lam}, PSNR: {tik_metrics['psnr']:.1f} dB)")
    axs[2].axis("off")

    plt.tight_layout()
    vis_path = VIS_DIR / "03_tikhonov_reconstruction.png"
    plt.savefig(vis_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved comparison plot to: {vis_path}")

    return tik_recon


if __name__ == "__main__":
    run_tikhonov_solver()
