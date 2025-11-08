"""
GODMAX: Gas thermODynamics and Matter distribution using jAX

This package provides tools for analyzing shear2pt and shear-y correlations
using JAX for GPU acceleration and differentiable likelihoods.
"""

__version__ = "0.1.0"

# Import main modules for easier access
from . import constants
from . import background

# Import main classes
try:
    from .get_power_spectra_jit import get_power_BCMP
    from .setup_power_spectra_jit import setup_power_BCMP
    from .get_BCMP_profile_jit import get_BCMP_profile
    from .get_corr_func_jit import get_corr_func
except ImportError:
    # Some dependencies might not be installed yet
    pass

__all__ = [
    "constants",
    "background",
    "get_power_BCMP",
    "setup_power_BCMP",
    "get_BCMP_profile",
    "get_corr_func",
]
