# CRITICAL FIXES for Memory Exhausted Error

## The Problem

Your error shows:
```
Out of memory while trying to allocate 78749543408 bytes (78.7 GB)
```

This happens during **pathfinder initialization**, not profile creation. Pathfinder vmaps over your likelihood function, creating multiple copies of large arrays simultaneously.

## ROOT CAUSES

1. **Pathfinder vmap**: Creates `num_pathfinder_draws` copies of arrays
2. **Grid too large**: Your nr × nz × nM creates huge matrices
3. **Multiple array copies**: Each pathfinder sample copies all profile matrices
4. **Possible re-creation**: Likelihood might be recreating Profiles each call

## IMMEDIATE FIXES (Apply These Now!)

### Fix 1: Reduce Pathfinder Samples (CRITICAL!)

In your notebook, find this line:
```python
num_pathfinder_draws = ???  # Whatever it currently is
```

**Change to:**
```python
num_pathfinder_draws = 10  # Drastically reduce from default
```

Or **better yet, disable pathfinder entirely**:
```python
# DON'T use pathfinder for initialization
# Use simple initialization instead (see below)
```

### Fix 2: Reduce Grid Resolution (CRITICAL!)

In your `halo_params_dict`:

**Before:**
```python
halo_params_dict = {
    'nr': 100,  # or higher
    'nz': 20,   # or higher
    'nM': 50,   # or higher
    ...
}
```

**After:**
```python
halo_params_dict = {
    'nr': 40,   # Reduce by 60%
    'nz': 10,   # Reduce by 50%
    'nM': 25,   # Reduce by 50%
    ...
}
```

This alone reduces memory by **~8x**!

### Fix 3: Enable ALL Optimizations

Make sure these are set:
```python
analysis_dict = {
    'model_galaxies': True,
    'model_tSZ': True,
    'model_matter': 'DMB',
    'use_scan_zeta': True,      # MUST BE TRUE
    'n_zeta_points': 8,          # MUST BE 8 (not 16 or 32)
    'verbose_time': True,        # See what's slow
}
```

### Fix 4: Use Simple Initialization Instead of Pathfinder

**Replace this:**
```python
pf_state, pf_info = blackjax.pathfinder.approximate(
    pf_key, log_density, z_init, num_samples=num_pathfinder_draws, ...)
```

**With this:**
```python
# Simple initialization without pathfinder
initial_state = blackjax.nuts.init(z_init, log_density)

# Or use a smaller warmup without pathfinder
warmup = blackjax.window_adaptation(
    blackjax.nuts,
    log_density,
    num_steps=500,  # Warmup steps
)
(state, kernel_params), _ = warmup.run(jax.random.PRNGKey(0), initial_state)
```

### Fix 5: Ensure Profiles Created Only Once

**Critical**: Make sure your likelihood function looks like this:

**WRONG (creates Profiles each call):**
```python
def log_likelihood(params):
    profiles = Profiles(...)  # ❌ DON'T DO THIS
    # ... compute likelihood
```

**CORRECT (reuses Profiles):**
```python
# Create profiles ONCE before sampling
profiles_template = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)

def log_likelihood(params):
    # Update only what's needed, don't recreate
    # Use profiles_template or update in-place
    # ... compute likelihood
```

## COMPLETE FIXED CODE EXAMPLE

Here's what your sampling code should look like:

```python
import jax
import jax.numpy as jnp
import blackjax
jax.config.update("jax_enable_x64", True)

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# ===== STEP 1: MINIMAL GRID RESOLUTION =====
halo_params_dict = {
    'rmin': 5e-3, 'rmax': 3, 'nr': 40,      # Reduced!
    'zmin': 1e-3, 'zmax': 1.5, 'nz': 10,    # Reduced!
    'lg10_Mmin': 12, 'lg10_Mmax': 15.0, 'nM': 25,  # Reduced!
    'ellmin': 10, 'ellmax': 8000, 'nell': 50,
    # ... other params
}

# ===== STEP 2: ENABLE ALL OPTIMIZATIONS =====
analysis_dict = {
    'model_galaxies': True,
    'model_tSZ': True,
    'model_matter': 'DMB',
    'use_scan_zeta': True,    # Critical!
    'n_zeta_points': 8,        # Critical!
    'verbose_time': True,
}

# ===== STEP 3: CREATE PROFILES ONCE =====
print("Creating Profiles object...")
from get_radial_profiles import Profiles

profiles = Profiles(
    sim_params_dict,
    halo_params_dict,
    analysis_dict,
    other_params_dict
)
print("Profiles created successfully!")

# ===== STEP 4: DEFINE LIKELIHOOD (no Profiles recreation!) =====
def log_likelihood(params):
    """
    Likelihood function that reuses the profiles object.
    Update only the parameters that changed.
    """
    # Extract parameters
    # theta_ej = params[0]
    # theta_co = params[1]
    # ... etc

    # Compute observable (reusing profiles arrays where possible)
    # Cl = compute_Cl(profiles, params)  # Your actual computation

    # Compute chi-square
    # chi2 = jnp.sum((Cl - data)**2 / errors**2)

    # For testing, just return a simple value
    return -0.5 * jnp.sum(params**2)  # Gaussian prior for testing


# ===== STEP 5: SAMPLE WITHOUT PATHFINDER =====
print("Starting sampling...")

# Initial position
initial_params = jnp.zeros(num_params)

# Initialize NUTS
initial_state = blackjax.nuts.init(initial_params, log_likelihood)

# Warmup (replaces pathfinder)
print("Running warmup...")
warmup_key = jax.random.PRNGKey(0)
warmup = blackjax.window_adaptation(
    blackjax.nuts,
    log_likelihood,
    num_steps=200,  # Reduced warmup
)
(state, kernel_params), _ = warmup.run(warmup_key, initial_state)

# Sample
print("Running MCMC...")
kernel = blackjax.nuts(log_likelihood, **kernel_params)

def one_step(state, key):
    new_state, info = kernel.step(key, state)
    return new_state, (new_state.position, info.acceptance_probability)

# Run chain
num_samples = 100  # Start small for testing
keys = jax.random.split(jax.random.PRNGKey(1), num_samples)
final_state, (positions, accept_probs) = jax.lax.scan(one_step, state, keys)

print(f"Sampling complete! Mean acceptance: {jnp.mean(accept_probs):.2f}")
```

## IF STILL FAILING

### Option 1: Reduce Grid Even More
```python
halo_params_dict['nr'] = 30
halo_params_dict['nz'] = 8
halo_params_dict['nM'] = 20
```

### Option 2: Use float32 Instead of float64
```python
jax.config.update("jax_enable_x64", False)  # Use float32
```

### Option 3: Run Without JIT During Debugging
```python
# In get_radial_profiles.py, temporarily disable @jit decorators
# to see exactly where memory is used
```

### Option 4: Sample Sequentially (No Parallel Chains)
```python
num_chains = 1  # Only 1 chain at a time
# Run multiple times sequentially instead of in parallel
```

## VERIFICATION CHECKLIST

Before running sampling, verify:

- [ ] `halo_params_dict['nr'] <= 50`
- [ ] `halo_params_dict['nz'] <= 15`
- [ ] `halo_params_dict['nM'] <= 30`
- [ ] `analysis_dict['use_scan_zeta'] = True`
- [ ] `analysis_dict['n_zeta_points'] = 8`
- [ ] NOT using pathfinder (or num_pathfinder_draws <= 10)
- [ ] Profiles created only ONCE before sampling
- [ ] Likelihood function does NOT recreate Profiles
- [ ] `XLA_PYTHON_CLIENT_PREALLOCATE = 'false'`

## MEMORY CALCULATION

With reduced settings:
- nr=40, nz=10, nM=25
- Each 3D matrix: 40 × 10 × 25 = 10,000 elements
- With ~20 such matrices: 200K elements × 8 bytes = 1.6 MB per matrix set
- Total for one Profiles: ~50-100 MB (manageable!)

With your original settings (example):
- nr=150, nz=32, nM=64
- Each 3D matrix: 150 × 32 × 64 = 307,200 elements
- Total could be 3-5 GB per Profiles instance
- Pathfinder with 100 samples: 300-500 GB! (impossible!)

## RUN DIAGNOSTIC

Before trying sampling again, run:
```bash
cd /mnt/ceph/users/spandey/paper_pge/GODMAX
python diagnose_memory.py
```

This will show you exactly where memory is being used and confirm optimizations are working.

## SUMMARY

The 78.7 GB allocation is from pathfinder vmapping over large arrays. Fix by:
1. **Disable pathfinder** or reduce to 10 samples
2. **Reduce grid**: nr=40, nz=10, nM=25
3. **Enable optimizations**: use_scan_zeta=True, n_zeta_points=8
4. **Create Profiles once**, reuse in likelihood

This should bring memory usage from 78 GB → under 1 GB.
