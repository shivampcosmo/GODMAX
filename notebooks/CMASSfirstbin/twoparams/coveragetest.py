import os
import pickle as pk
import torch
import numpy as np
from pathlib import Path
import sys

# --- Path Setup for ltu-ili ---
sys.path.append('/work/hdd/bdne/aacharya2/ltu-ili')
from ili.validation.metrics import PosteriorCoverage

# =============================================================================
# 1. SETUP
# =============================================================================
labels = [r"\theta_{ej,0}", r"\nu_{\theta,M}", r"\nu_{\theta,z}", r"\mu_{\beta}"]
tracers = ['g2y', 'g2tau', 'g2kappa']
NUM_QUICK_SAMPLES = 10  # Number of test points to evaluate
RANDOM_SEED = 42        # For reproducible quick tests

# Load the parameters
theta_all = np.load('theta.npy')

print(f"Total dataset: {len(theta_all)} samples. Performing quick test on {NUM_QUICK_SAMPLES} samples.")

# =============================================================================
# 2. VALIDATION LOOP
# =============================================================================
for name in tracers:
    print(f"\n--- Quick Validation: {name} ---")
    
    pkl_path = f'ili_posterior_{name}.pkl'
    x_path = f'x_{name}.npy'
    out_dir = Path(f'./validation_quick_{name}')
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (os.path.exists(pkl_path) and os.path.exists(x_path)):
        print(f"Skipping {name}: Files not found.")
        continue

    # Load trained posterior and data
    with open(pkl_path, 'rb') as f:
        posterior = pk.load(f)
    x_all = np.load(x_path)

    # --- Subsampling for Speed ---
    np.random.seed(RANDOM_SEED)
    indices = np.random.choice(len(theta_all), NUM_QUICK_SAMPLES, replace=False)
    x_test = x_all[indices]
    theta_test = theta_all[indices]

    # Initialize PosteriorCoverage
    # num_samples refers to posterior samples per test point
    metric = PosteriorCoverage(
        num_samples=1000, # Reduced from 2000 for extra speed in quick test
        labels=labels,
        out_dir=out_dir,
        plot_list=["histogram", "coverage", "tarp"],
        save_samples=False
    )

    # Run the validation
    # With 10 points, this should finish in a few minutes per tracer
    metric(posterior=posterior, x=x_test, theta=theta_test)

print(f"\nQuick test complete. Results in ./validation_[tracer]/")
