"""GODMAX — Gas thermODynamics and Matter distribution using jAX.

A differentiable JAX halo model: radial profiles -> P(k, z) -> C(l) -> covariances.

The core inference chain is exposed at the top level::

    from godmax import Profiles, get_Pkz, get_Cl, get_cov, get_xi

which mirrors the class inheritance order
``base_class -> Profiles -> get_Pkz -> get_Cl -> get_cov`` (with ``get_xi`` on top of
``get_Cl``).
"""

from godmax.get_radial_profiles import Profiles
from godmax.get_Pkzs import get_Pkz
from godmax.get_Cls import get_Cl
from godmax.get_covs import get_cov
from godmax.get_Xis import get_xi

__version__ = "0.1.0"

__all__ = ["Profiles", "get_Pkz", "get_Cl", "get_cov", "get_xi", "__version__"]
