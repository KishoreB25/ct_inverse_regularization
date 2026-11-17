"""
solvers/landweber.py
Landweber Iterative Regularization:
x_{k+1} = x_k + omega * A^T (b - A(x_k))
Iteration count serves as the regularization parameter (early stopping / semi-convergence).
"""

from typing import Dict, Any, Optional
import time
import numpy as np
from .operators import RadonOperator


def landweber_reconstruct(
    sinogram: np.ndarray,
    operator: RadonOperator,
    max_iter: int = 50,
    omega: Optional[float] = None,
    ground_truth: Optional[np.ndarray] = None,
    non_negative: bool = True,
) -> Dict[str, Any]:
    """
    Reconstructs an image using Landweber iteration.

    Parameters:
        sinogram: (num_detectors, num_angles) 2D projection data
        operator: RadonOperator instance
        max_iter: Total number of iterations
        omega: Relaxation parameter. If None, set to 1.0 / ||A||^2 for guaranteed stability.
        ground_truth: Optional true phantom for per-iteration MSE tracking
        non_negative: Whether to enforce non-negativity constraint (x >= 0)

    Returns:
        dict containing:
            'reconstruction': 2D reconstructed image (N, N)
            'runtime_sec': Wall-clock runtime in seconds
            'omega': Relaxation parameter used
            'mse_history': List of per-iteration MSE values (if ground_truth provided)
            'residual_history': List of per-iteration residual norms ||A(x_k) - b||_2
    """
    if omega is None:
        spectral_norm = operator.estimate_spectral_norm()
        omega = 1.0 / spectral_norm  # Stable step size within (0, 2/||A||^2)

    start_time = time.time()
    x = np.zeros((operator.image_size, operator.image_size), dtype=np.float64)

    mse_history = []
    residual_history = []

    for k in range(max_iter):
        Ax = operator.forward(x)
        residual = sinogram - Ax
        residual_norm = float(np.linalg.norm(residual))
        residual_history.append(residual_norm)

        if ground_truth is not None:
            mse = float(np.mean((ground_truth - x) ** 2))
            mse_history.append(mse)

        # Adjoint update
        backproject = operator.adjoint(residual)
        x = x + omega * backproject

        if non_negative:
            x = np.clip(x, 0.0, None)

    runtime = time.time() - start_time

    return {
        "reconstruction": x,
        "runtime_sec": runtime,
        "omega": omega,
        "iterations": max_iter,
        "mse_history": mse_history,
        "residual_history": residual_history,
        "method": "Landweber",
    }

