"""
Example script showing how to use optimized zeta calculation for blackjax sampling.

This demonstrates the memory-efficient configuration for running MCMC chains.
"""

import sys
import os
import pathlib

# Setup paths (adjust to your environment)
curr_path = pathlib.Path().absolute()
abs_path_src = os.path.abspath(curr_path / "../src/")
sys.path.append(abs_path_src)

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Memory optimization settings
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

from get_radial_profiles import Profiles
import yaml


# ========== CONFIGURATION ==========

# Example parameter dictionaries (adjust to your needs)
sim_params_dict = {
    'cosmo': {
        'flat': True, 'H0': 67.2, 'Om0': 0.31, 'Ob0': 0.049,
        'sigma8': 0.81, 'ns': 0.95, 'w0': -1.0
    },
    'theta_ej_0': 4.0,
    'theta_co_0': 0.1,
    # ... add your other sim params
}

halo_params_dict = {
    'rmin': 5e-3,
    'rmax': 3,
    'nr': 100,  # Reduced from default 150 for memory
    'zmin': 1e-3,
    'zmax': 1.5,
    'nz': 20,   # Reduced from default 32 for memory
    'lg10_Mmin': 12,
    'lg10_Mmax': 15.0,
    'nM': 50,   # Reduced from default 64 for memory
    'ellmin': 10,
    'ellmax': 8000,
    'nell': 100,
    # ... add your other halo params
}

# ===== OPTIMIZATION FLAGS (KEY PART!) =====
analysis_dict = {
    'model_galaxies': True,
    'model_tSZ': True,
    'model_matter': 'DMB',
    'verbose_time': True,  # Enable timing to see performance

    # Memory optimization flags:
    'use_scan_zeta': True,   # Use scan instead of full vmap (4x less memory)
    'n_zeta_points': 16,     # Use 16 points instead of 32 (2x speedup)

    # Optional: use symbolic regression for even more speed
    # 'symbolic_hmf': True,
    # 'symbolic_pk': True,
}

other_params_dict = {
    # Add your other params
}


# ========== CREATE PROFILES ==========

print("Creating Profiles object with optimized settings...")
print(f"  use_scan_zeta: {analysis_dict['use_scan_zeta']}")
print(f"  n_zeta_points: {analysis_dict['n_zeta_points']}")
print(f"  Grid resolution: nr={halo_params_dict['nr']}, nz={halo_params_dict['nz']}, nM={halo_params_dict['nM']}")
print()

profiles = Profiles(
    sim_params_dict,
    halo_params_dict,
    analysis_dict,
    other_params_dict
)

print("Profiles created successfully!")
print(f"zeta_mat shape: {profiles.zeta_mat.shape}")
print()


# ========== VALIDATION (Optional) ==========

# Check that zeta values are reasonable
print("Zeta matrix statistics:")
print(f"  Min: {jnp.min(profiles.zeta_mat):.4f}")
print(f"  Max: {jnp.max(profiles.zeta_mat):.4f}")
print(f"  Mean: {jnp.mean(profiles.zeta_mat):.4f}")
print(f"  Std: {jnp.std(profiles.zeta_mat):.4f}")
print()


# ========== COMPARISON (Optional) ==========

# Compare with non-optimized version to check accuracy
if False:  # Set to True to run comparison
    print("Running comparison with non-optimized version...")

    analysis_dict_orig = analysis_dict.copy()
    analysis_dict_orig['use_scan_zeta'] = False
    analysis_dict_orig['n_zeta_points'] = 32

    profiles_orig = Profiles(
        sim_params_dict,
        halo_params_dict,
        analysis_dict_orig,
        other_params_dict
    )

    zeta_diff = jnp.abs(profiles.zeta_mat - profiles_orig.zeta_mat)
    print(f"Difference statistics:")
    print(f"  Max difference: {jnp.max(zeta_diff):.6f}")
    print(f"  Mean difference: {jnp.mean(zeta_diff):.6f}")
    print(f"  Median difference: {jnp.median(zeta_diff):.6f}")
    print()


# ========== USE WITH BLACKJAX ==========

# Now you can use profiles in your likelihood function
def log_likelihood(params):
    """Example likelihood function for MCMC sampling."""
    # Update parameters
    # profiles_updated = Profiles(...updated params...)

    # Compute observable (e.g., power spectrum)
    # Cl = get_Cl(profiles_updated, ...)

    # Compare with data
    # chi2 = jnp.sum((Cl - data)**2 / error**2)

    return -0.5 * chi2  # Replace with your actual calculation


# Example blackjax usage (pseudo-code)
"""
import blackjax

# Define sampler
warmup = blackjax.window_adaptation(
    blackjax.nuts,
    log_likelihood,
    num_steps=500,
)

# Run warmup
initial_state = ...
(state, parameters), _ = warmup.run(initial_state)

# Sample
kernel = blackjax.nuts(log_likelihood, **parameters)
states = blackjax.sample(kernel, state, num_samples=1000)
"""

print("Script completed successfully!")
print()
print("Memory optimization tips:")
print("  1. If still OOM: reduce nr, nz, or nM further")
print("  2. For faster: set n_zeta_points=8 (trade accuracy for speed)")
print("  3. For debugging: set verbose_time=True to see timing")
print("  4. For production: consider symbolic_hmf=True, symbolic_pk=True")
