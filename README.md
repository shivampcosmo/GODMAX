# GODMAX

**G**as Therm**OD**ynamics and **M**atter Distribution using J**AX**

GODMAX is a JAX-based implementation of the halo model for cosmological calculations, designed for high-performance GPU computing with automatic differentiation and gradient-based sampling.

## Overview

GODMAX provides a fully differentiable framework for computing:
- **Halo model calculations** - One of the first JAX implementations of the halo model
- **Gas thermodynamics** - Including thermal Sunyaev-Zel'dovich (tSZ) and kinetic SZ (kSZ) effects
- **Matter power spectra** - 3D power spectra P(k,z) for various matter components
- **Angular power spectra** - 2D angular power spectra C(ℓ) for cosmological observables
- **Radial profiles** - Gas pressure, density, and other physical profiles in halos
- **Covariance matrices** - For multi-probe cosmological analyses
- **Pasting maps** - JAX-based pasting a lightcone halo catalog with multi-probe observables (lensing, tSZ, kSZ, galaxies)

### Key Features

- **JAX-native implementation** - Ready to run on GPUs with automatic parallelization
- **Fully differentiable** - All calculations preserve gradients for optimization and inference
- **HMC sampling** - Integration with NumPyro for Hamiltonian Monte Carlo sampling
- **JAX mcfit** - Differentiable implementation of mcfit for Hankel transforms, allowing gradients to flow through integral transforms
- **Modular design** - Extensible class-based architecture for easy customization

## Installation

### Prerequisites

- Python 3.8+
- JAX with GPU support (recommended)
- NumPyro for sampling

### Dependencies

The main dependencies include:
- `jax` and `jax-cosmo` for cosmological calculations
- `numpyro` for Bayesian sampling
- `astropy` for astronomical constants and units
- `interpax` for JAX-compatible interpolation
- Standard scientific Python stack (numpy, scipy)

## Project Structure

```
GODMAX/
├── src/                        # Source code
│   ├── base_class.py          # Base class with common functionality
│   ├── get_radial_profiles.py # Compute gas and matter profiles
│   ├── get_Pkzs.py            # 3D power spectra calculations
│   ├── get_Cls.py             # Angular power spectra
│   ├── get_covs.py            # Covariance matrix calculations
│   ├── get_sim_maps.py        # Simulate cosmological maps
│   ├── matter_pk_symbolic.py  # Matter power spectrum functions
│   ├── hmf_symbolic.py        # Halo mass function
│   ├── helpers/               # Utility functions
│   │   ├── constants.py       # Physical constants
│   │   ├── jax_cosmo_power.py # Power spectrum utilities
│   │   └── twobessel.py       # Two-Bessel integral transforms
│   └── mcfitjax/              # JAX implementation of mcfit
│       ├── mcfit_jax.py       # Main mcfit functionality
│       ├── transforms.py      # Hankel transforms
│       ├── kernels.py         # Transform kernels
│       └── cosmology_jax.py   # Cosmology-specific transforms
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── ACTxDES/               # ACT x DES (tSZ-lensing) project related notebooks
│   └── Pge/                   # kSZ-galaxy cross-correlation related
├── run_scripts/               # Scripts for running analyses
├── param_files/               # Parameter configuration files
│   └── params_default.yaml    # Default parameter settings
├── data/                      # Data files (not tracked)
└── measurements/              # Output measurements
```

## Usage

### Basic Example

```python
import yaml
import pathlib
curr_path = pathlib.Path().absolute()
abs_path_data = os.path.abspath(curr_path / "../../data/") 
abs_path_src = os.path.abspath(curr_path / "../../src/") 
abs_path_results = os.path.abspath(curr_path / "../../results/") 
abs_path_params = os.path.abspath(curr_path / "../../param_files/") 
from base_class import base_class
from get_radial_profiles import Profiles
from get_Pkzs import get_Pkz
from get_Cls import get_Cl
from deepmerge import always_merger

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def generate_dicts(data):
    sim_params_dict = data.get('sim_params', {})
    halo_params_dict = data.get('halo_params', {})
    analysis_dict = data.get('analysis', {})
    other_params_dict = data.get('other_params', {})
    return sim_params_dict, halo_params_dict, analysis_dict, other_params_dict

default_data = read_yaml(abs_path_params + '/params_default.yaml')
new_data = read_yaml(abs_path_params + '/Pge/params.yaml')
merged_data = always_merger.merge(default_data, new_data)
sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(merged_data)

base_test = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
profiles_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, base_class_obj=base_test)
Pkz_test = get_Pkz(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Profiles_obj=profiles_test)
Cl_test = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Pkz_obj=Pkz_test)
```

### Configuration

Parameters are organized into four categories:

1. **sim_params_dict**: Cosmological and baryonic physics parameters
   - Cosmological parameters (H0, Om0, sigma8, etc.)
   - Gas profile parameters (BCMP model)
   - Stellar mass and galaxy occupation parameters
   - Non-thermal pressure parameters

2. **halo_params_dict**: Numerical resolution and accuracy settings
   - Mass and redshift grid specifications
   - Integration parameters

3. **analysis_dict**: Analysis settings and model choices
   - Which observables to compute (tSZ, kSZ, galaxy clustering, etc.)
   - Accuracy parameters

4. **other_params_dict**: Additional systematic parameters
   - Intrinsic alignment parameters
   - Photo-z bias parameters

See [params_default.yaml](param_files/params_default.yaml) for a complete parameter reference.

### GPU Acceleration

GODMAX is designed to run on GPUs. To enable GPU support:

```python
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'

import jax
jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_enable_x64", True)

import numpyro
numpyro.set_platform("gpu")
numpyro.enable_x64()
```

### Sampling with NumPyro

GODMAX integrates with NumPyro for Hamiltonian Monte Carlo sampling:

```python
import numpyro
from numpyro.infer import MCMC, NUTS

# Define your likelihood model using GODMAX calculations
def model(data):
    # ... define priors and likelihood using GODMAX
    pass

# Run HMC sampling
nuts_kernel = NUTS(model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
mcmc.run(rng_key, data)
```

See the [run_scripts](run_scripts/) directory for complete sampling examples.

## Physics Models

GODMAX implements several physical models:

- **BCMP gas profile** - Pandey+24/Scheider+19 model for gas pressure and density
- **B12 profile** - Battaglia et al. 2012 gas profile
- **OWLS profile** - OverWhelmingly Large Simulations calibrated profiles
- **NFW profile** - Navarro-Frenk-White dark matter profile with optional truncation
- **Stellar mass-halo mass relation** - Flexible SHMR modeling
- **Galaxy occupation** - HOD-style galaxy distribution in halos

## Applications

GODMAX has been used for:

- Multi-probe cosmological analyses (3x2pt, tSZ, kSZ)
- Cross-correlation studies (ACT x DES)
- Parameter inference with gradient-based samplers
- Fast map simulations on spherical geometries
- Covariance matrix predictions

## Performance

- Runs efficiently on GPUs (tested on NVIDIA A100, V100)
- Automatic parallelization via JAX's vmap
- JIT compilation for optimal performance
- Gradient calculations via automatic differentiation

## Contributing

This is a research code. For questions or collaboration, please open an issue.

## License

See the repository for license information.

## Citation

If you use GODMAX in your research, please cite the relevant papers (https://arxiv.org/abs/2401.18072, https://arxiv.org/abs/2506.07432).

## Acknowledgments

GODMAX builds on:
- [JAX-COSMO](https://github.com/DifferentiableUniverseInitiative/jax_cosmo) and [Colossus](https://bdiemer.bitbucket.io/colossus/) for cosmological background calculations
- [NumPyro](https://github.com/pyro-ppl/numpyro) for probabilistic programming
- [mcfit](https://github.com/eelregit/mcfit) for the original mcfit implementation
