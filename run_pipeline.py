"""
run_pipeline.py
Master runner script that executes the complete CT Inverse Regularization pipeline:
1. Synthetic Phantom & Sinogram generation
2. Realistic Gaussian noise addition
3. Tikhonov Regularization (Conjugate Gradient)
4. Landweber Iterative Regularization (stable relaxation)
5. Non-Stationary Iterated Tikhonov (NSIT) with Morozov's Discrepancy Principle
6. Quantitative Evaluation (MSE, PSNR, SSIM, Rel. Error)
7. Visual Comparison Suite & Error Maps
8. Iteration Convergence Analysis
9. Runtime Benchmarking
"""

import sys
import time
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Dynamically import numbered script modules
s01 = importlib.import_module("scripts.01_generate_phantom_and_sinogram")
s02 = importlib.import_module("scripts.02_add_noise_and_save")
s03 = importlib.import_module("scripts.03_tikhonov_solver")
s04 = importlib.import_module("scripts.04_landweber_solver")
s05 = importlib.import_module("scripts.05_nsit_solver")
s06 = importlib.import_module("scripts.06_evaluate_metrics")
s07 = importlib.import_module("scripts.07_visualize_comparisons")
s08 = importlib.import_module("scripts.08_log_metrics_during_iterations")
s09 = importlib.import_module("scripts.09_runtime_analysis")


def main():
    start_total = time.time()
    print("=" * 60)
    print("      CT INVERSE REGULARIZATION PIPELINE")
    print("=" * 60)

    print("\n>>> Step 1: Generating Ground Truth Phantom & Sinogram")
    s01.generate_phantom_and_sinogram(image_size=128, num_angles=60)

    print("\n>>> Step 2: Adding 5% Gaussian Noise to Sinogram")
    s02.add_noise(noise_level=0.05, seed=42)

    print("\n>>> Step 3: Running Tikhonov Regularization Solver")
    s03.run_tikhonov_solver(lam=10.0)

    print("\n>>> Step 4: Running Landweber Iterative Solver (50 iters)")
    s04.run_landweber_solver(iterations=50)

    print("\n>>> Step 5: Running NSIT Solver with Morozov Stopping")
    s05.run_nsit_solver(lambda0=10.0, q=0.7, max_iter=15)

    print("\n>>> Step 6: Computing Quantitative Evaluation Metrics")
    s06.evaluate_all_methods()

    print("\n>>> Step 7: Generating Visual Comparisons, Error Maps & Line Profiles")
    s07.generate_visualizations()

    print("\n>>> Step 8: Logging Iterative Convergence (MSE Curves)")
    s08.log_iteration_convergence()

    print("\n>>> Step 9: Running Runtime Benchmarks")
    s09.benchmark_runtimes()

    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    print(f" [SUCCESS] Complete pipeline finished in {total_time:.2f} seconds.")
    print(f" All reconstructions saved in : {PROJECT_ROOT / 'outputs' / 'reconstructions'}")
    print(f" All visualizations saved in  : {PROJECT_ROOT / 'outputs' / 'visualizations'}")
    print(f" Metrics report saved in      : {PROJECT_ROOT / 'outputs' / 'metrics'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
