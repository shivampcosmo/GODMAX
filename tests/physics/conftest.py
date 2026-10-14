"""Shared configuration for the physics consistency suite.

Every model is built from `param_files/params_default.yaml` with a reduced but
representative grid (64 radial points to 16 Mpc/h, as in the tSZxEuclid Pge set-up),
one tomographic bin and a short ell range. Builds are cached for the whole session,
so each configuration is constructed once however many tests use it.

Run:
    uv run --no-sync --with pytest python -m pytest -q tests/physics
    uv run --no-sync --with pytest python -m pytest -q tests/physics -m "not slow"   # skip gradient tests
"""
import contextlib
import copy
import io
import pathlib
import warnings

import jax
import numpy as np
import yaml

try:                     # pytest is optional so that notebooks can reuse build() and base_dicts()
    import pytest
except ImportError:      # pragma: no cover
    pytest = None

jax.config.update("jax_enable_x64", True)

import godmax.get_radial_profiles as grp  # noqa: E402
from godmax import get_Cl  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
PARAMS = REPO / "param_files" / "params_default.yaml"

# Grid used by the suite. Tolerances in the tests assume these values.
GRID = dict(nr=64, rmin=0.005, rmax=16.0, nz=4, zmin=0.01, zmax=1.6, nM=12,
            lg10_Mmin=11.5, lg10_Mmax=15.5, nk=48, kmin=1e-4, kmax=150.0,
            ellmin=10.0, ellmax=3000.0, nell=12)

# Configurations: name -> analysis overrides
CONFIGS = {
    "on":           dict(model_matter="DMB", backreaction=True, baryonification_tSZ=True),
    "off":          dict(model_matter="halofit", backreaction=False, baryonification_tSZ=False),
    "no_backreact": dict(model_matter="DMB", backreaction=False, baryonification_tSZ=True),
    "tsz_off":      dict(model_matter="DMB", backreaction=True, baryonification_tSZ=False),
}


def _numeric(x):
    """YAML 1.1 reads bare '5e-4' as a string: convert numeric-looking strings."""
    if isinstance(x, dict):
        return {k: _numeric(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_numeric(v) for v in x]
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return x
    return x


def base_dicts():
    with open(PARAMS) as f:
        p = _numeric(yaml.safe_load(f))
    sim, halo, ana, other = (copy.deepcopy(p[k]) for k in ("sim_params", "halo_params", "analysis", "other_params"))
    halo.update(GRID)
    nb = ana.get("nbar_gal_comoving_val", 5e-4)
    ana["nbar_gal_comoving_val"] = float(nb[0] if isinstance(nb, list) else nb)
    for key in ("yy_noise_ell_fname", "yy_total_ell_fname"):
        ana.pop(key, None)
    zs = np.linspace(0.01, 1.5, 128)
    ana["nz_source_info_dict"] = {"nbins": 1, "z_array_source": zs, "nz0": ((zs > 0.2) & (zs < 1.0)).astype(float)}
    zl = np.linspace(0.0, 2.0, 401)
    ana["nz_lens_info_dict"] = {"nbins_lens": 1, "z_array_lens": zl, "nz0": ((zl > 0.2) & (zl < 0.8)).astype(float),
                                "z_edges_bins_lens": [[0.2, 0.8]]}
    ana["nbar_lens_bins"] = [0.3]
    ana["is_cmb_lensing"] = False
    return sim, halo, ana, other


_CACHE = {}


def build(config="on", ana_over=None, halo_over=None, sim_over=None, rho_clm=None, cls=get_Cl):
    """Build (and cache) a model. rho_clm: optional replacement for Profiles.get_rho_clm."""
    key = (config, repr(sorted((ana_over or {}).items())), repr(sorted((halo_over or {}).items())),
           repr(sorted((sim_over or {}).items())), getattr(rho_clm, "__name__", None), cls.__name__)
    if key in _CACHE:
        return _CACHE[key]
    sim, halo, ana, other = base_dicts()
    ana.update(CONFIGS[config]); ana.update(ana_over or {}); halo.update(halo_over or {}); sim.update(sim_over or {})
    previous = grp.Profiles.get_rho_clm
    try:
        if rho_clm is not None:
            grp.Profiles.get_rho_clm = rho_clm
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            obj = cls(sim, halo, ana, other)
    finally:
        grp.Profiles.get_rho_clm = previous
    _CACHE[key] = obj
    return obj


if pytest is not None:
    @pytest.fixture(scope="session")
    def model():
        """Factory fixture: model(config, ...) -> cached model object."""
        return build

    def pytest_configure(config):
        config.addinivalue_line("markers", "slow: gradient tests (several minutes)")
