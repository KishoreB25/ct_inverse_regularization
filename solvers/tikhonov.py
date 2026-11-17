"""
solvers/tikhonov.py
Standard Tikhonov Regularization for CT reconstruction:
Solves (A^T A + lambda * I) x = A^T b
"""

from typing import Dict, Any, Optional
import time
import numpy as np
from scipy.sparse.linalg import cg
from skimage.transform import iradon_sart
from .operators import RadonOperator


def tikhonov_reconstruct(
    sinogram: np.ndarray,
    operator: RadonOperator,
    lam: float = 10.0,
    rtol: float = 1e-4,
    maxiter: int = 50,
    method: str = "cg",
) -> Dict[str, Any]:
    """
    Reconstructs an image using Tikhonov regularization.

    Parameters:
        sinogram: (num_detectors, num_angles) 2D projection data
        operator: RadonOperator instance
        lam: Regularization parameter lambda > 0
        rtol: Relative tolerance for CG solver
        maxiter: Maximum iterations for CG solver
        method: 'cg' (canonical normal equation) or 'sart' (SART algebraic approx)

    Returns:
        dict containing:
            'reconstruction': 2D reconstructed image (N, N)
            'runtime_sec': Wall-clock runtime in seconds
            'lambda': Regularization parameter used
            'method': Method name
    """
    start_time = time.time()

    if method == "sart":
        recon = iradon_sart(sinogram, theta=operator.theta)
    else:
        # Canonical Tikhonov normal equation: (A^T A + lam * I) x = A^T b
        M = operator.normal_linear_operator(lam=lam)
        AT_b = operator.adjoint(sinogram).ravel()

        x0 = np.zeros(operator.image_size * operator.image_size, dtype=np.float64)
        x_flat, info = cg(M, AT_b, x0=x0, rtol=rtol, maxiter=maxiter)
        recon = x_flat.reshape((operator.image_size, operator.image_size))

    # Clip negative values if physical absorption constraint is applied
    recon = np.clip(recon, 0.0, None)
    runtime = time.time() - start_time

    return {
        "reconstruction": recon,
        "runtime_sec": runtime,
        "lambda": lam,
        "method": f"Tikhonov ({method.upper()})",
    }

