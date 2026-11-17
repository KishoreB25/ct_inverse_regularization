"""
scripts/01_load_lodopab_data.py
Optional loader for the LoDoPaB-CT dataset using DiVAL.
Demonstrates loading clinical low-dose CT projections and ground truth slices.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_lodopab_sample():
    try:
        from dival import get_standard_dataset
    except ImportError:
        print("[!] 'dival' is not installed or available.")
        print("    To use LoDoPaB real CT data, install dival (pip install dival) and configure dataset path.")
        return

    print("[*] Attempting to load LoDoPaB dataset (validation split)...")
    try:
        lodopab = get_standard_dataset("lodopab", impl="astra_cpu")
    except Exception as e:
        print(f"[!] Could not initialize LoDoPaB dataset: {e}")
        print("    Ensure dataset files are downloaded or configure ~/.dival/config.json.")
        return

    print(f"[+] Loaded LoDoPaB Dataset. Validation samples: {len(lodopab.validation)}")
    sample = lodopab.validation[0]
    sinogram = sample["observation"]
    ground_truth = sample["ground_truth"]

    sino_path = OUTPUT_DIR / "sample_lodopab_sinogram.png"
    gt_path = OUTPUT_DIR / "sample_lodopab_ground_truth.png"
    plt.imsave(sino_path, sinogram, cmap="gray")
    plt.imsave(gt_path, ground_truth, cmap="gray")
    print(f"[+] Saved sample sinogram to: {sino_path}")
    print(f"[+] Saved sample ground truth to: {gt_path}")


if __name__ == "__main__":
    load_lodopab_sample()
