"""
scripts/02_add_noise_and_save.py
Simulates realistic measurement noise on the CT projection data.
Adds Gaussian noise to model low-dose acquisition / ill-posed measurement.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SINO_DIR = PROJECT_ROOT / "data" / "projections"


def add_noise(noise_level: float = 0.05, seed: int = 42):
    sino_path = SINO_DIR / "sinogram.npy"
    if not sino_path.exists():
        raise FileNotFoundError(
            f"Sinogram not found at {sino_path}. Run scripts/01_generate_phantom_and_sinogram.py first."
        )

    sinogram = np.load(sino_path)
    print(f"[*] Loaded sinogram with shape {sinogram.shape}.")

    # Gaussian noise scaled by 5% of maximum projection value
    noise_std = noise_level * np.max(sinogram)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, sinogram.shape)
    noisy_sino = sinogram + noise

    # Clip to non-negative (physical transmission attenuation cannot be arbitrarily negative)
    noisy_sino = np.clip(noisy_sino, 0.0, None)

    out_npy = SINO_DIR / "sinogram_noisy.npy"
    np.save(out_npy, noisy_sino)
    print(f"[+] Saved noisy sinogram to: {out_npy}")

    # Save preview image
    out_png = SINO_DIR / "sinogram_noisy.png"
    plt.imsave(out_png, noisy_sino, cmap="gray")

    # Generate comparison plot
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(sinogram, cmap="gray", aspect="auto")
    axs[0].set_title("Original Clean Sinogram")
    axs[0].axis("off")

    axs[1].imshow(noisy_sino, cmap="gray", aspect="auto")
    axs[1].set_title(f"Noisy Sinogram ({int(noise_level * 100)}% Gaussian Noise)")
    axs[1].axis("off")

    plt.tight_layout()
    preview_path = SINO_DIR / "preview_noisy_sinogram.png"
    plt.savefig(preview_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved comparison preview to: {preview_path}")

    return noisy_sino


if __name__ == "__main__":
    add_noise()
