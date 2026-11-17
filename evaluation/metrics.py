"""
evaluation/metrics.py
Quantitative evaluation metrics and formatting for CT reconstructions.
"""

from typing import Dict, Any, Union
import json
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def compute_metrics(
    ground_truth: np.ndarray,
    reconstruction: np.ndarray,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """
    Computes standard quantitative metrics between ground truth and reconstruction.

    Parameters:
        ground_truth: True 2D phantom array
        reconstruction: Reconstructed 2D image array
        data_range: Dynamic range of input images (default: 1.0)

    Returns:
        Dict with keys: 'mse', 'psnr', 'ssim', 'rel_error', 'mae'
    """
    gt = np.asarray(ground_truth, dtype=np.float64)
    rec = np.asarray(reconstruction, dtype=np.float64)

    # Handle shape mismatch if any
    if gt.shape != rec.shape:
        raise ValueError(f"Shape mismatch: ground truth {gt.shape} vs reconstruction {rec.shape}")

    # If reconstruction is 3D/RGBA, convert to grayscale 2D
    if rec.ndim == 3:
        rec = rec[..., :3].mean(axis=-1)

    mse = float(mean_squared_error(gt, rec))
    psnr = float(peak_signal_noise_ratio(gt, rec, data_range=data_range))
    ssim = float(structural_similarity(gt, rec, data_range=data_range))
    mae = float(np.mean(np.abs(gt - rec)))

    gt_norm = np.linalg.norm(gt)
    rel_error = float(np.linalg.norm(gt - rec) / gt_norm) if gt_norm > 0 else 0.0

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "mae": mae,
        "rel_error": rel_error,
    }


def format_metrics_table(results: Dict[str, Dict[str, Any]]) -> str:
    """
    Formats comparison metrics as a clear ASCII table.
    """
    header = f"{'Method':<16} | {'MSE':>10} | {'PSNR (dB)':>10} | {'SSIM':>8} | {'Rel Err':>10} | {'Runtime (s)':>11}"
    separator = "-" * len(header)
    rows = [header, separator]

    for method, metrics in results.items():
        mse = metrics.get("mse", float("nan"))
        psnr = metrics.get("psnr", float("nan"))
        ssim = metrics.get("ssim", float("nan"))
        rel_err = metrics.get("rel_error", float("nan"))
        runtime = metrics.get("runtime_sec", float("nan"))

        row = (
            f"{method:<16} | {mse:>10.6f} | {psnr:>10.2f} | {ssim:>8.4f} | "
            f"{rel_err:>10.4f} | {runtime:>11.3f}"
        )
        rows.append(row)

    return "\n".join(rows)


def save_metrics_report(results: Dict[str, Dict[str, Any]], filepath: str) -> None:
    """
    Saves metrics results to a JSON file.
    """
    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            m: float(val) if isinstance(val, (int, float, np.number)) else val
            for m, val in v.items()
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=4)

