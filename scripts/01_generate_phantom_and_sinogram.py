"""
scripts/01_generate_phantom_and_sinogram.py
Generates the Shepp-Logan ground truth phantom and computes its clean Radon sinogram.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize

# Path configuration relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PHANTOM_DIR = PROJECT_ROOT / "data" / "phantoms"
SINO_DIR = PROJECT_ROOT / "data" / "projections"

PHANTOM_DIR.mkdir(parents=True, exist_ok=True)
SINO_DIR.mkdir(parents=True, exist_ok=True)


def generate_phantom_and_sinogram(image_size: int = 128, num_angles: int = 60):
    print(f"[*] Generating Shepp-Logan phantom ({image_size}x{image_size})...")
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (image_size, image_size), anti_aliasing=True)
    phantom = phantom / np.max(phantom)  # Normalize to [0, 1]

    print(f"[*] Computing Radon sinogram with {num_angles} projection angles...")
    theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)
    sinogram = radon(phantom, theta=theta, circle=True)

    # Save arrays
    phantom_npy = PHANTOM_DIR / "phantom.npy"
    sino_npy = SINO_DIR / "sinogram.npy"
    np.save(phantom_npy, phantom)
    np.save(sino_npy, sinogram)
    print(f"[+] Saved phantom to: {phantom_npy}")
    print(f"[+] Saved sinogram to: {sino_npy}")

    # Save preview images
    phantom_png = PHANTOM_DIR / "phantom.png"
    sino_png = SINO_DIR / "sinogram.png"
    plt.imsave(phantom_png, phantom, cmap="gray")
    plt.imsave(sino_png, sinogram, cmap="gray")

    # Generate combined preview figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(phantom, cmap="gray")
    axs[0].set_title("Shepp-Logan Ground Truth")
    axs[0].axis("off")

    axs[1].imshow(sinogram, cmap="gray", aspect="auto")
    axs[1].set_title(f"Clean Sinogram ({num_angles} angles)")
    axs[1].axis("off")

    plt.tight_layout()
    preview_path = SINO_DIR / "preview_phantom_and_sinogram.png"
    plt.savefig(preview_path, dpi=150)
    plt.close(fig)
    print(f"[+] Saved preview figure to: {preview_path}")

    return phantom, sinogram


if __name__ == "__main__":
    generate_phantom_and_sinogram()
