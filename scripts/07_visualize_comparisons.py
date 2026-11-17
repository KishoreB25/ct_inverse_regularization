"""
scripts/07_visualize_comparisons.py
Visual comparison suite for CT reconstructions:
1. Reconstructions alongside absolute error maps.
2. SSIM spatial error maps.
3. Center-row line intensity profile comparisons.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from skimage.metrics import structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PHANTOM_PATH = PROJECT_ROOT / "data" / "phantoms" / "phantom.npy"
RECON_DIR = PROJECT_ROOT / "outputs" / "reconstructions"
VIS_DIR = PROJECT_ROOT / "outputs" / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


def generate_visualizations():
    if not PHANTOM_PATH.exists():
        raise FileNotFoundError(f"Ground truth not found at {PHANTOM_PATH}")

    phantom = np.load(PHANTOM_PATH)
    candidate_methods = ["fbp", "tikhonov", "landweber", "nsit"]
    methods = []
    images = {"Ground Truth": phantom}
    errors = {}
    ssims = {}
    line_profiles = {}

    for method in candidate_methods:
        npy_path = RECON_DIR / f"{method}.npy"
        png_path = RECON_DIR / f"{method}.png"

        if npy_path.exists():
            img = np.load(npy_path)
        elif png_path.exists():
            img = imread(png_path)
            if img.ndim == 3:
                img = img[..., :3].mean(axis=-1)
            img = img.astype(np.float64)
        else:
            continue

        label = method.upper() if method in ["fbp", "nsit"] else method.title()
        methods.append(label)
        images[label] = img
        errors[label] = np.abs(phantom - img)

        _, ssim_map = structural_similarity(phantom, img, data_range=1.0, full=True)
        ssims[label] = ssim_map

        # Center line profile
        center_row = phantom.shape[0] // 2
        line_profiles[label] = img[center_row, :]

    if not methods:
        print("[!] No reconstruction results found to visualize.")
        return

    # --- 1. Reconstructions + Error Maps ---
    num_cols = len(images)
    fig, axs = plt.subplots(2, num_cols, figsize=(4 * num_cols, 7))

    for idx, (name, img) in enumerate(images.items()):
        axs[0, idx].imshow(img, cmap="gray", vmin=0, vmax=1)
        axs[0, idx].set_title(name, fontsize=12, fontweight="bold")
        axs[0, idx].axis("off")

        if name != "Ground Truth":
            im_err = axs[1, idx].imshow(errors[name], cmap="hot", vmin=0, vmax=0.5)
            axs[1, idx].set_title(f"|Error|: {name}", fontsize=11)
            axs[1, idx].axis("off")
            plt.colorbar(im_err, ax=axs[1, idx], fraction=0.046, pad=0.04)
        else:
            axs[1, idx].axis("off")
            axs[1, idx].text(0.2, 0.5, "Reference\n(No Error)", fontsize=11, style="italic")

    plt.suptitle("CT Inverse Regularization: Reconstructions & Error Maps", fontsize=14, y=0.98)
    plt.tight_layout()
    out1 = VIS_DIR / "07_comparisons_and_errors.png"
    plt.savefig(out1, dpi=200)
    plt.close(fig)
    print(f"[+] Saved comparison panel to: {out1}")

    # --- 2. SSIM Maps ---
    fig2, axs2 = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.5))
    if len(methods) == 1:
        axs2 = [axs2]

    for idx, m in enumerate(methods):
        im_ssim = axs2[idx].imshow(ssims[m], cmap="viridis", vmin=0, vmax=1)
        axs2[idx].set_title(f"SSIM Map: {m}", fontsize=11)
        axs2[idx].axis("off")
        plt.colorbar(im_ssim, ax=axs2[idx], fraction=0.046, pad=0.04)

    plt.suptitle("Structural Similarity Index (SSIM) Maps", fontsize=13)
    plt.tight_layout()
    out2 = VIS_DIR / "07_ssim_maps.png"
    plt.savefig(out2, dpi=200)
    plt.close(fig2)
    print(f"[+] Saved SSIM maps to: {out2}")

    # --- 3. Center Row Line Profiles ---
    center_row = phantom.shape[0] // 2
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    ax3.plot(phantom[center_row, :], label="Ground Truth", color="black", linewidth=2.5, linestyle="--")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, m in enumerate(methods):
        ax3.plot(line_profiles[m], label=m, color=colors[idx % len(colors)], linewidth=1.5, alpha=0.85)

    ax3.set_title(f"Center Row (y = {center_row}) Intensity Line Profile", fontsize=12)
    ax3.set_xlabel("Pixel Index (x)", fontsize=11)
    ax3.set_ylabel("Attenuation Value", fontsize=11)
    ax3.legend(loc="upper right", frameon=True)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out3 = VIS_DIR / "07_line_profiles.png"
    plt.savefig(out3, dpi=200)
    plt.close(fig3)
    print(f"[+] Saved line profiles to: {out3}")


if __name__ == "__main__":
    generate_visualizations()
