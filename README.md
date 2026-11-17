# CT Inverse Problem Regularization

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Scikit--Image%20%7C%20SciPy-Scientific%20Computing-green.svg)](https://scikit-image.org/)

A modular Python framework for solving the ill-posed **X-ray Computed Tomography (CT) inverse problem** using classical and advanced regularization techniques: **Tikhonov Regularization**, **Landweber Iteration**, and **Non-Stationary Iterated Tikhonov (NSIT)** with **Morozov's Discrepancy Principle**.

---

## 📌 Purpose & Background

In 2D X-ray Computed Tomography, the forward imaging model measures line integrals of the object's attenuation coefficient $x$ along projection rays at multiple angles $\theta$, modeled mathematically by the **Radon Transform** $A$:

$$b = A x + \epsilon$$

where:
- $x \in L^2(\Omega)$ represents the cross-sectional 2D image (attenuation values).
- $b$ is the measured projection sinogram.
- $\epsilon$ is measurement noise (e.g., electronic and Poisson/Gaussian photon noise).

### The Ill-Posed Nature of Inverse CT
According to **Hadamard's criteria**, an inverse problem is ill-posed if a solution does not exist, is not unique, or is not stable under perturbations. The continuous Radon transform has an unbounded inverse: high-frequency components of noise $\epsilon$ are exponentially or polynomially amplified when computing naive inversions (such as unfiltered backprojection or unregularized pseudo-inverse). In sparse-angle or low-dose imaging, standard **Filtered Backprojection (FBP)** produces prominent streak artifacts and noise amplification.

Regularization methods introduce prior physical constraints or penalty terms to stabilize the inversion and recover sharp, artifact-free reconstructions.

---

## 🔬 Methods Implemented

### 1. Filtered Backprojection (FBP) — Baseline
- Standard analytic inversion using the **Ram-Lak (ramp) filter** in the frequency domain.
- Serves as the baseline comparison. Highly sensitive to noise and angular sparsity.

### 2. Tikhonov Regularization (Ridge / $L_2$ Penalization)
- Minimizes the penalized least-squares objective:
  $$\min_{x} \|A x - b\|_2^2 + \lambda \|x\|_2^2$$
- Corresponds to solving the regularized normal equations:
  $$(A^T A + \lambda I) x = A^T b$$
- Solved efficiently via the **Conjugate Gradient (CG)** method on the symmetric positive-definite operator $(A^T A + \lambda I)$.

### 3. Landweber Iteration (Iterative Regularization)
- Formulates reconstruction as fixed-step gradient descent on the least-squares residual:
  $$x_{k+1} = x_k + \omega A^T (b - A x_k)$$
- **Guaranteed Convergence**: The relaxation parameter $\omega$ is chosen strictly within the spectral radius bound:
  $$0 < \omega < \frac{2}{\|A\|^2}$$
  The spectral norm $\|A\|^2 = \lambda_{\max}(A^T A)$ is estimated automatically via power iteration.
- **Regularization by Early Stopping**: The iteration count $k$ acts as the inverse regularization parameter (preventing the semi-convergence phenomenon where late iterations amplify noise).

### 4. Non-Stationary Iterated Tikhonov (NSIT)
- A multi-step regularized solver that updates the reconstruction with geometrically decreasing regularization parameter $\lambda_k$:
  $$(A^T A + \lambda_k I) x_{k+1} = A^T b + \lambda_k x_k, \quad \lambda_k = \lambda_0 \cdot q^k \quad (0 < q < 1)$$
- **A Posteriori Stopping Rule (Morozov's Discrepancy Principle)**:
  Terminates automatically as soon as the residual matches the estimated noise level $\delta$:
  $$\|A x_k - b\|_2 \le \tau \delta$$
  Guarantees optimal convergence rates without overfitting to noise.

---

## 📁 Repository Structure

```
ct_inverse_regularization/
│
├── data/                               # Input phantoms and sinogram data
│   ├── phantoms/                       # Shepp-Logan ground truth (phantom.npy, phantom.png)
│   └── projections/                    # Projections (sinogram.npy, sinogram_noisy.npy)
│
├── solvers/                            # Modular, reusable solver package
│   ├── __init__.py                     # Package interface
│   ├── operators.py                    # Radon forward, adjoint, FBP, and power iteration
│   ├── tikhonov.py                     # CG-based Tikhonov solver
│   ├── landweber.py                    # Landweber iteration with spectral stability
│   └── nsit.py                         # NSIT solver with Morozov stopping criterion
│
├── evaluation/                         # Modular evaluation package
│   ├── __init__.py                     # Package interface
│   └── metrics.py                      # MSE, PSNR, SSIM, Relative Error, and tables
│
├── scripts/                            # Standalone execution scripts (runnable from anywhere)
│   ├── 01_generate_phantom_and_sinogram.py
│   ├── 01_load_lodopab_data.py         # Optional real CT clinical loader (DiVAL)
│   ├── 02_add_noise_and_save.py        # Noise injection (5% Gaussian)
│   ├── 03_tikhonov_solver.py           # Tikhonov reconstruction & FBP baseline
│   ├── 04_landweber_solver.py          # Landweber reconstruction
│   ├── 05_nsit_solver.py               # NSIT reconstruction
│   ├── 06_evaluate_metrics.py          # Metrics summary report generator
│   ├── 07_visualize_comparisons.py     # Comparison panels, error & SSIM maps
│   ├── 08_log_metrics_during_iterations.py # Convergence curves
│   └── 09_runtime_analysis.py          # Execution time benchmarks
│
├── notebooks/                          # Jupyter Notebooks
│   └── 01_data_preparation.ipynb      # Interactive data exploration & visualization
│
├── outputs/                            # Generated results (reconstructions & plots)
│   ├── reconstructions/                # Saved .npy arrays and .png images
│   ├── visualizations/                 # Error maps, SSIM maps, line profiles, MSE curves
│   └── metrics/                        # JSON evaluation metrics report
│
├── run_pipeline.py                     # Master runner (executes complete workflow end-to-end)
├── test_astra.py                       # Environment & hardware acceleration diagnostic
├── requirements.txt                    # Project dependencies
├── LICENSE                             # MIT License
└── README.md                           # Documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python **3.9**, **3.10**, **3.11**, or **3.12**
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/KishoreB25/ct_inverse_regularization.git
cd ct_inverse_regularization
```

### 2. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
Run the diagnostic script to ensure all scientific computing modules are correctly installed:
```bash
python test_astra.py
```
> **Note on ASTRA-Toolbox**: The entire pipeline runs out-of-the-box on native CPU using `scikit-image` and `scipy`. Installing ASTRA-Toolbox is completely optional (for GPU acceleration via `conda install -c astra-toolbox astra-toolbox`).

---

## 🚀 How to Run

### Option A: One-Click End-to-End Pipeline
To run all stages (data generation, noise injection, solvers, evaluation, error mapping, and benchmarks) in one command:
```bash
python run_pipeline.py
```

### Option B: Step-by-Step Execution
All scripts in `scripts/` can be run independently from either the root repository directory or the `scripts/` folder:

```bash
# 1. Generate Shepp-Logan phantom and Radon sinogram (60 angles)
python scripts/01_generate_phantom_and_sinogram.py

# 2. Add 5% Gaussian noise to simulate low-dose measurement
python scripts/02_add_noise_and_save.py

# 3. Reconstruct using FBP baseline & Tikhonov Regularization
python scripts/03_tikhonov_solver.py

# 4. Reconstruct using Landweber Iteration (50 iterations)
python scripts/04_landweber_solver.py

# 5. Reconstruct using NSIT with Morozov's Discrepancy Principle
python scripts/05_nsit_solver.py

# 6. Evaluate quantitative metrics (MSE, PSNR, SSIM, Relative Error)
python scripts/06_evaluate_metrics.py

# 7. Generate visual comparisons, error maps, SSIM maps, and line profiles
python scripts/07_visualize_comparisons.py

# 8. Plot convergence curves (MSE vs iteration)
python scripts/08_log_metrics_during_iterations.py

# 9. Run execution time benchmarks
python scripts/09_runtime_analysis.py
```

### Option C: Using the Solvers as a Python Package
You can import the solvers directly into your own scripts or research notebooks:
```python
import numpy as np
from solvers import RadonOperator, tikhonov_reconstruct, landweber_reconstruct, nsit_reconstruct
from evaluation import compute_metrics

# 1. Setup operator
operator = RadonOperator(image_size=128, num_angles=60)

# 2. Load noisy sinogram
sinogram_noisy = np.load("data/projections/sinogram_noisy.npy")
phantom_true = np.load("data/phantoms/phantom.npy")

# 3. Solve with Landweber
lw_res = landweber_reconstruct(sinogram_noisy, operator, max_iter=50)
recon_lw = lw_res["reconstruction"]

# 4. Evaluate
metrics = compute_metrics(phantom_true, recon_lw)
print(f"PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")
```

---

## 📊 Experimental Results & Benchmarks

Simulated on a $128 \times 128$ Shepp-Logan phantom sampled at 60 sparse projection angles with **5% Gaussian noise**:

### Quantitative Metrics Summary

| Method | Mean Squared Error (MSE) ↓ | Peak SNR (PSNR) ↑ | SSIM ↑ | Relative $L_2$ Error ↓ | Convergence / Stopping Rule |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **FBP (Baseline)** | 0.0145 | 18.40 dB | 0.3537 | 0.5158 | Direct Filtered Backprojection |
| **Tikhonov (CG)** | 0.0138 | 18.60 dB | 0.4815 | 0.5041 | $\lambda = 10.0$ |
| **Landweber** | **0.0057** | **22.41 dB** | **0.5371** | **0.3252** | 50 iterations ($\omega = 0.00576$) |
| **NSIT (Morozov)** | **0.0057** | **22.40 dB** | **0.4522** | **0.3253** | **3 iterations** (Morozov stopped) |

### Key Observations
1. **Iterative Solvers Outperform Analytic FBP**: Both Landweber and NSIT achieve a **+4.0 dB PSNR gain** and a **60% reduction in MSE** compared to standard Filtered Backprojection.
2. **Speed & Efficiency of NSIT**: While Landweber requires 50 iterations to reach peak quality (~21.5s on CPU), **NSIT achieves identical MSE and PSNR in just 3 outer iterations (~3.7s on CPU)** by decaying the regularization parameter geometrically and terminating automatically via Morozov's Discrepancy Principle.
3. **Artifact Suppression**: Tikhonov and Landweber suppress high-frequency streak noise effectively, producing smooth homogeneous regions while preserving inner structural contours.

---

## 🖼️ Visualizations & Artifacts

All visual outputs are automatically saved to `outputs/visualizations/`:
- `07_comparisons_and_errors.png`: Side-by-side reconstruction images paired with absolute spatial error maps ($|x^* - x|$).
- `07_ssim_maps.png`: Spatial structural similarity distribution highlighting structural fidelity.
- `07_line_profiles.png`: Horizontal cross-sectional profile through the phantom center row ($y = 64$).
- `08_mse_convergence.png`: Per-iteration MSE reduction curves for Landweber and NSIT.
- `09_runtime_comparison.png`: Horizontal bar chart benchmarking computational runtimes.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
