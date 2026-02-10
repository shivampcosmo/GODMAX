"""
Diagnostic script to identify memory bottlenecks in your sampling setup.
Run this to see where memory is being consumed.
"""

import sys
import os
import pathlib

curr_path = pathlib.Path().absolute()
abs_path_src = os.path.abspath(curr_path / "../../src/")
sys.path.append(str(abs_path_src))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import tracemalloc
import gc


def check_array_sizes(obj, name="profiles", max_depth=1, current_depth=0):
    """Recursively check sizes of JAX arrays in an object."""
    if current_depth > max_depth:
        return []

    results = []

    if isinstance(obj, jnp.ndarray):
        size_mb = obj.nbytes / 1024**2
        results.append((name, obj.shape, size_mb))
    elif hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            if isinstance(attr_value, jnp.ndarray):
                size_mb = attr_value.nbytes / 1024**2
                results.append((f"{name}.{attr_name}", attr_value.shape, size_mb))

    return results


def diagnose_profiles():
    """Check memory usage of Profiles object."""
    from get_radial_profiles import Profiles

    print("="*60)
    print("DIAGNOSING PROFILES MEMORY USAGE")
    print("="*60)

    # Minimal test params
    sim_params_dict = {
        'cosmo': {'flat': True, 'H0': 67.2, 'Om0': 0.31, 'Ob0': 0.049,
                  'sigma8': 0.81, 'ns': 0.95, 'w0': -1.0},
        'theta_ej_0': 4.0, 'theta_co_0': 0.1, 'mu_beta': 0.21,
        'log10_Mstar0': 13.0, 'log10_Mc0': 14.83,
    }

    halo_params_dict = {
        'rmin': 5e-3, 'rmax': 3, 'nr': 100,
        'zmin': 1e-3, 'zmax': 1.5, 'nz': 20,
        'lg10_Mmin': 12, 'lg10_Mmax': 15.0, 'nM': 50,
        'ellmin': 10, 'ellmax': 8000, 'nell': 50,
        'kmin': 1e-4, 'kmax': 150, 'nk': 48,
    }

    # Check WITHOUT optimization
    print("\n1. Testing WITHOUT optimization flags...")
    analysis_dict_1 = {
        'model_galaxies': True,
        'model_tSZ': True,
        'model_matter': 'DMB',
        'verbose_time': True,
        'use_scan_zeta': False,
        'n_zeta_points': 8,
    }

    tracemalloc.start()
    gc.collect()

    try:
        profiles_1 = Profiles(sim_params_dict, halo_params_dict, analysis_dict_1, {})
        _ = profiles_1.zeta_mat.block_until_ready()

        current, peak = tracemalloc.get_traced_memory()
        print(f"✓ Success! Peak memory: {peak/1024**2:.1f} MB")

        # Check array sizes
        print("\nLargest arrays in Profiles object:")
        arrays = check_array_sizes(profiles_1)
        arrays.sort(key=lambda x: x[2], reverse=True)
        for name, shape, size_mb in arrays[:10]:
            print(f"  {name:30s} {str(shape):30s} {size_mb:8.1f} MB")

        total_mb = sum(x[2] for x in arrays)
        print(f"\nTotal array memory: {total_mb:.1f} MB")

    except Exception as e:
        print(f"✗ Failed: {e}")

    tracemalloc.stop()
    gc.collect()

    # Check WITH optimization
    print("\n" + "="*60)
    print("2. Testing WITH optimization flags...")
    analysis_dict_2 = {
        'model_galaxies': True,
        'model_tSZ': True,
        'model_matter': 'DMB',
        'verbose_time': True,
        'use_scan_zeta': True,
        'n_zeta_points': 8,
    }

    tracemalloc.start()
    gc.collect()

    try:
        profiles_2 = Profiles(sim_params_dict, halo_params_dict, analysis_dict_2, {})
        _ = profiles_2.zeta_mat.block_until_ready()

        current, peak = tracemalloc.get_traced_memory()
        print(f"✓ Success! Peak memory: {peak/1024**2:.1f} MB")

        # Check if flags were actually set
        print(f"\nOptimization flags check:")
        print(f"  use_scan_zeta: {getattr(profiles_2, 'use_scan_zeta', 'NOT SET')}")
        print(f"  n_zeta_points: {getattr(profiles_2, 'n_zeta_points', 'NOT SET')}")

        # Check array sizes
        print("\nLargest arrays in Profiles object:")
        arrays = check_array_sizes(profiles_2)
        arrays.sort(key=lambda x: x[2], reverse=True)
        for name, shape, size_mb in arrays[:10]:
            print(f"  {name:30s} {str(shape):30s} {size_mb:8.1f} MB")

        total_mb = sum(x[2] for x in arrays)
        print(f"\nTotal array memory: {total_mb:.1f} MB")

    except Exception as e:
        print(f"✗ Failed: {e}")

    tracemalloc.stop()

    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


def diagnose_vmap_issue():
    """Simulate what happens during pathfinder vmap."""
    print("\n" + "="*60)
    print("DIAGNOSING VMAP/PATHFINDER ISSUE")
    print("="*60)

    print("\nThe error shows pathfinder is trying to allocate 78.7 GB!")
    print("This happens when vmapping over likelihood function.")
    print("\nProblem: Pathfinder creates multiple samples and vmaps over them,")
    print("causing multiple copies of large arrays to be created.")

    print("\n" + "="*60)
    print("CRITICAL FIXES NEEDED:")
    print("="*60)
    print("""
1. REDUCE PATHFINDER SAMPLES (most important!)
   num_pathfinder_draws = 10  # Instead of 100+

2. REDUCE GRID RESOLUTION
   halo_params_dict['nr'] = 50   # Instead of 100+
   halo_params_dict['nz'] = 10   # Instead of 20+
   halo_params_dict['nM'] = 30   # Instead of 50+

3. USE OPTIMIZATION FLAGS
   analysis_dict['use_scan_zeta'] = True
   analysis_dict['n_zeta_points'] = 8

4. DISABLE PATHFINDER (if still fails)
   Use simple NUTS initialization instead

5. ENSURE LIKELIHOOD DOESN'T RECREATE PROFILES
   Profiles should be created ONCE, then reused
    """)


if __name__ == "__main__":
    diagnose_profiles()
    diagnose_vmap_issue()
