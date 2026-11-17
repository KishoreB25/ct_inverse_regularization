"""
scripts/06_evaluate_metrics.py
Evaluates quantitative reconstruction quality metrics (MSE, PSNR, SSIM, Rel. Error)
across all computed regularization methods against the ground truth phantom.
"""

import sys
from pathlib import Path
import numpy as np
from matplotlib.image import imread

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_metrics, format_metrics_table, save_metrics_report

PHANTOM_PATH = PROJECT_ROOT / "data" / "phantoms" / "phantom.npy"
RECON_DIR = PROJECT_ROOT / "outputs" / "reconstructions"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_all_methods():
    if not PHANTOM_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth phantom not found at {PHANTOM_PATH}. Run script 01 first."
        )

    phantom = np.load(PHANTOM_PATH)
    methods = ["fbp", "tikhonov", "landweber", "nsit"]
    results = {}

    print("\n" + "=" * 55)
    print(" CT INVERSE REGULARIZATION EVALUATION REPORT")
    print("=" * 55)

    for method in methods:
        npy_path = RECON_DIR / f"{method}.npy"
        png_path = RECON_DIR / f"{method}.png"

        if npy_path.exists():
            recon = np.load(npy_path)
        elif png_path.exists():
            img = imread(png_path)
            if img.ndim == 3:
                img = img[..., :3].mean(axis=-1)
            recon = img.astype(np.float64)
        else:
            print(f"[!] Reconstruction for '{method}' not found. Skipping.")
            continue

        metrics = compute_metrics(phantom, recon, data_range=1.0)
        results[method] = metrics

    if results:
        table_str = format_metrics_table(results)
        print("\n" + table_str + "\n")

        # Save metrics
        json_path = METRICS_DIR / "evaluation_metrics.json"
        save_metrics_report(results, str(json_path))
        print(f"[+] Saved evaluation metrics to: {json_path}")
    else:
        print("[!] No reconstructions found in outputs/reconstructions/.")

    return results


if __name__ == "__main__":
    evaluate_all_methods()
