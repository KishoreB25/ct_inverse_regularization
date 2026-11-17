"""
solvers/operators.py
Forward Radon projection and adjoint backprojection operators for 2D CT.
"""

from typing import Optional, Tuple
import numpy as np
from skimage.transform import radon, iradon
from scipy.sparse.linalg import LinearOperator


class RadonOperator:
    """
    Encapsulates the discrete 2D Radon forward projection (A)
    and unweighted backprojection adjoint (A^T).
    """

    def __init__(self, image_size: int = 128, theta: Optional[np.ndarray] = None, num_angles: int = 60):
        self.image_size = int(image_size)
        if theta is None:
            self.theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)
        else:
            self.theta = np.asarray(theta, dtype=np.float64)
        self.num_angles = len(self.theta)
        self._spectral_norm: Optional[float] = None

    @property
    def sino_shape(self) -> Tuple[int, int]:
        # Radon detector dimension in scikit-image matches image dimension (diagonal padding in circle=True)
        # Using a dummy 1-pixel slice to get detector count reliably
        dummy = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        sino_dummy = radon(dummy, theta=self.theta, circle=True)
        return sino_dummy.shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward projection: b = A(x).
        Maps 2D image (N, N) -> sinogram (num_detectors, num_angles).
        """
        x_2d = x.reshape((self.image_size, self.image_size))
        return radon(x_2d, theta=self.theta, circle=True)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """
        Unfiltered backprojection: x = A^T(y).
        Adjoint operator to the forward Radon projection.
        Maps sinogram (num_detectors, num_angles) -> 2D image (N, N).
        """
        return iradon(
            y,
            theta=self.theta,
            circle=True,
            filter_name=None,
            output_size=self.image_size,
        )

    def fbp(self, y: np.ndarray, filter_name: str = 'ramp') -> np.ndarray:
        """
        Standard Filtered Backprojection (FBP) baseline reconstruction.
        """
        return iradon(
            y,
            theta=self.theta,
            circle=True,
            filter_name=filter_name,
            output_size=self.image_size,
        )

    def estimate_spectral_norm(self, num_iters: int = 15) -> float:
        """
        Estimates ||A||^2 = lambda_max(A^T A) via power iteration.
        Crucial for choosing a provably stable Landweber relaxation step size:
        omega in (0, 2 / ||A||^2).
        """
        if self._spectral_norm is not None:
            return self._spectral_norm

        rng = np.random.default_rng(42)
        v = rng.standard_normal((self.image_size, self.image_size))
        v /= np.linalg.norm(v)

        for _ in range(num_iters):
            Av = self.forward(v)
            AT_Av = self.adjoint(Av)
            norm = np.linalg.norm(AT_Av)
            if norm == 0:
                break
            v = AT_Av / norm

        self._spectral_norm = float(norm)
        return self._spectral_norm

    def normal_linear_operator(self, lam: float = 0.0) -> LinearOperator:
        """
        Returns a SciPy LinearOperator representing the normal system:
        M(v) = (A^T A + lam * I) v
        Operating on flattened image vectors of length (N * N).
        """
        n = self.image_size * self.image_size
        sino_shape = self.sino_shape

        def matvec(v_flat: np.ndarray) -> np.ndarray:
            x_2d = v_flat.reshape((self.image_size, self.image_size))
            Ax = radon(x_2d, theta=self.theta, circle=True)
            ATAx = iradon(
                Ax,
                theta=self.theta,
                circle=True,
                filter_name=None,
                output_size=self.image_size,
            )
            result = ATAx.ravel()
            if lam > 0.0:
                result = result + lam * v_flat
            return result

        return LinearOperator(shape=(n, n), matvec=matvec, dtype=np.float64)

