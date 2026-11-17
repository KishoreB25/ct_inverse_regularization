"""
solvers/nsit.py
Non-Stationary Iterated Tikhonov (NSIT) Regularization:
Solves (A^T A + lambda_k * I) x_{k+1} = A^T b + lambda_k * x_k
where lambda_k = lambda_0 * q^k (0 < q < 1)
Equipped with Morozov's Discrepancy Principle stopping criterion:
Stops when ||A(x_k) - b||_2 <= tau * delta
"""

from typing import Dict, Any, Optional
import time
import numpy as np
from scipy.sparse.linalg import cg
from .operators import RadonOperator


def nsit_reconstruct(
    sinogram: np.ndarray,
    operator: RadonOperator,
    lambda0: float = 10.0,
    q: float = 0.7,
    max_iter: int = 15,
    delta: Optional[float] = None,
    tau: float = 1.0,
    cg_rtol: float = 1e-4,
    cg_maxiter: int = 40,
    ground_truth: Optional[np.ndarray] = None,
    non_negative: bool = True,
) -> Dict[str, Any]:
    """
    Reconstructs an image using Non-Stationary Iterated Tikhonov (NSIT).

    Parameters:
        sinogram: (num_detectors, num_angles) 2D projection data
        operator: RadonOperator instance
        lambda0: Initial regularization parameter
        q: Decay rate (0 < q < 1)
        max_iter: Maximum NSIT outer iterations
        delta: Known/estimated noise level ||noise||_2 for Morozov stopping criterion
        tau: Discrepancy threshold multiplier (tau >= 1.0)
        cg_rtol: Relative tolerance for inner CG solver
        cg_maxiter: Max iterations for inner CG solver
        ground_truth: Optional true phantom for per-iteration MSE tracking
        non_negative: Whether to enforce non-negativity constraint (x >= 0)

    Returns:
        dict containing:
            'reconstruction': 2D reconstructed image (N, N)
            'runtime_sec': Wall-clock runtime in seconds
            'lambda0': Initial lambda
            'q': Decay factor
            'iterations_run': Number of iterations completed
            'stopped_by_morozov': Boolean
            'lambdas': List of lambda values per iteration
            'mse_history': List of per-iteration MSE values (if ground_truth provided)
            'residual_history': List of per-iteration residual norms ||A(x_k) - b||_2
    """
    start_time = time.time()
    n = operator.image_size * operator.image_size
    b_flat = sinogram.ravel()
    AT_b = operator.adjoint(sinogram).ravel()

    x_k = np.zeros(n, dtype=np.float64)

    mse_history = []
    residual_history = []
    lambdas = []
    stopped_by_morozov = False

    for k in range(max_iter):
        lam = float(lambda0 * (q ** k))
        lambdas.append(lam)

        # Build normal operator for (A^T A + lam * I)
        M = operator.normal_linear_operator(lam=lam)
        rhs = AT_b + lam * x_k

        # Inner Conjugate Gradient solve
        x_next, _ = cg(M, rhs, x0=x_k, rtol=cg_rtol, maxiter=cg_maxiter)

        if non_negative:
            x_next = np.clip(x_next, 0.0, None)

        x_k = x_next

        # Compute residual norm
        Ax_k = operator.forward(x_k.reshape((operator.image_size, operator.image_size)))
        res_norm = float(np.linalg.norm(Ax_k - sinogram))
        residual_history.append(res_norm)

        if ground_truth is not None:
            recon_2d = x_k.reshape((operator.image_size, operator.image_size))
            mse = float(np.mean((ground_truth - recon_2d) ** 2))
            mse_history.append(mse)

        # Morozov's Discrepancy Principle stopping criterion
        if delta is not None and res_norm <= tau * delta:
            stopped_by_morozov = True
            break

    recon = x_k.reshape((operator.image_size, operator.image_size))
    runtime = time.time() - start_time

    return {
        "reconstruction": recon,
        "runtime_sec": runtime,
        "lambda0": lambda0,
        "q": q,
        "iterations_run": len(residual_history),
        "stopped_by_morozov": stopped_by_morozov,
        "lambdas": lambdas,
        "mse_history": mse_history,
        "residual_history": residual_history,
        "method": "NSIT",
    }

