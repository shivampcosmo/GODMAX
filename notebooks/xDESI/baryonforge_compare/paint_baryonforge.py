#!/usr/bin/env python
"""Paint native BaryonForge y and halo-only CMB-kappa maps on the shared catalog."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from common import (
    MAP_PRODUCT_SCHEMA,
    WORKSPACE_ROOT,
    assert_map_contract_unchanged,
    baryonforge_profile_kwargs,
    canonical_json,
    catalog_cosmology,
    current_map_contract,
    load_config_and_freeze_map_contract,
    load_yaml,
    resolve_path,
    validate_parameter_crosswalk,
)


BARYONFORGE_ROOT = WORKSPACE_ROOT / "BaryonForge"
NATIVE_BARYONFORGE_PROJECTION_METHOD = "native"
MATCHED_BARYONFORGE_PROJECTION_METHOD = "nonsingular_gauss_legendre"


def _scientific_imports():
    # The comparison is intentionally runnable from the two adjacent source
    # checkouts; installing an editable BaryonForge wheel is not required.
    if str(BARYONFORGE_ROOT) not in sys.path:
        sys.path.insert(0, str(BARYONFORGE_ROOT))
    import BaryonForge as bfg
    import healpy as hp
    import pyccl as ccl

    return bfg, hp, ccl


def build_ccl_cosmology(cosmo_dict: Mapping[str, float]):
    _, _, ccl = _scientific_imports()
    return ccl.Cosmology(
        Omega_c=float(cosmo_dict["Omega_m"]) - float(cosmo_dict["Omega_b"]),
        Omega_b=float(cosmo_dict["Omega_b"]),
        h=float(cosmo_dict["h"]),
        sigma8=float(cosmo_dict["sigma8"]),
        n_s=float(cosmo_dict["n_s"]),
        w0=float(cosmo_dict["w0"]),
        wa=float(cosmo_dict.get("wa", 0.0)),
        matter_power_spectrum="linear",
    )


def make_projection_factor_wrapper(ccl):
    class ProjectionFactorWrapper(ccl.halos.profiles.HaloProfile):
        """Multiply only the projected view of a profile by ``a**exponent``."""

        def __init__(self, profile, exponent: float):
            self.profile = profile
            self.exponent = float(exponent)
            self.p_keys = []
            super().__init__(mass_def=profile.mass_def)
            if hasattr(profile, "precision_fftlog"):
                self.update_precision_fftlog(**profile.precision_fftlog.to_dict())

        def _real(self, cosmo, r, M, a):
            return self.profile.real(cosmo, r, M, a)

        def _projected(self, cosmo, r, M, a):
            return self.profile.projected(cosmo, r, M, a) * float(a) ** self.exponent

    return ProjectionFactorWrapper


def make_matched_los_projection_wrapper(ccl):
    class MatchedLOSProjectionWrapper(ccl.halos.profiles.HaloProfile):
        """Project a 3D profile through a fixed, finite comoving LOS interval.

        The substitution ``l = R sinh(t)`` removes the Abel-kernel singularity
        at the projected radius.  Gauss--Legendre integration is then performed
        on ``0 <= t <= asinh(l_max / R)``.  The wrapped profile's ``real`` view
        is delegated unchanged.
        """

        def __init__(
            self,
            profile,
            *,
            los_max_comoving_mpc: float,
            num_points: int,
            projected_scale_factor_power: float,
        ):
            los_max = float(los_max_comoving_mpc)
            points = int(num_points)
            if not np.isfinite(los_max) or los_max <= 0.0:
                raise ValueError(
                    "los_max_comoving_mpc must be a finite positive number, "
                    f"got {los_max_comoving_mpc!r}."
                )
            if points < 2:
                raise ValueError(f"num_points must be at least 2, got {num_points!r}.")

            self.profile = profile
            self.los_max_comoving_mpc = los_max
            self.num_points = points
            self.projected_scale_factor_power = float(
                projected_scale_factor_power
            )
            self.integration_method = MATCHED_BARYONFORGE_PROJECTION_METHOD
            self.p_keys = list(getattr(profile, "p_keys", []))
            self._quadrature_nodes, self._quadrature_weights = (
                np.polynomial.legendre.leggauss(points)
            )
            super().__init__(mass_def=profile.mass_def)

        def _real(self, cosmo, r, M, a):
            return self.profile.real(cosmo, r, M, a)

        def _projected(self, cosmo, r, M, a):
            r_use = np.atleast_1d(np.asarray(r, dtype=np.float64))
            mass_use = np.atleast_1d(np.asarray(M, dtype=np.float64))
            if np.any(~np.isfinite(r_use)) or np.any(r_use <= 0.0):
                raise ValueError(
                    "Matched LOS projection requires finite, positive comoving radii."
                )
            if np.any(~np.isfinite(mass_use)) or np.any(mass_use <= 0.0):
                raise ValueError(
                    "Matched LOS projection requires finite, positive halo masses."
                )
            scale_factor = float(a)
            if not np.isfinite(scale_factor) or scale_factor <= 0.0:
                raise ValueError(
                    f"Matched LOS projection requires a finite positive scale factor, got {a!r}."
                )

            t_max = np.arcsinh(self.los_max_comoving_mpc / r_use)
            t = (
                0.5
                * t_max[:, None]
                * (self._quadrature_nodes[None, :] + 1.0)
            )
            cosh_t = np.cosh(t)
            radius_3d = r_use[:, None] * cosh_t
            sampled = np.asarray(
                self.profile.real(
                    cosmo,
                    radius_3d.reshape(-1),
                    mass_use,
                    scale_factor,
                ),
                dtype=np.float64,
            )
            expected_shape = (mass_use.size, radius_3d.size)
            if sampled.shape != expected_shape:
                raise ValueError(
                    "Wrapped profile returned an incompatible real-space shape: "
                    f"expected {expected_shape}, got {sampled.shape}."
                )
            sampled = sampled.reshape(
                mass_use.size, r_use.size, self.num_points
            )

            # The leading factor of two for +/- l cancels the one-half in the
            # affine map from the Gauss--Legendre interval [-1, 1] to [0, t_max].
            projected = np.sum(
                sampled
                * (r_use[:, None] * cosh_t)[None, :, :]
                * (t_max[:, None] * self._quadrature_weights[None, :])[None, :, :],
                axis=-1,
            )
            projected *= scale_factor ** self.projected_scale_factor_power

            if np.ndim(r) == 0:
                projected = np.squeeze(projected, axis=-1)
            if np.ndim(M) == 0:
                projected = np.squeeze(projected, axis=0)
            return projected

    return MatchedLOSProjectionWrapper


def apply_projection_adapter(
    profile,
    *,
    ccl,
    adapter: Mapping[str, Any],
    projected_scale_factor_power: float,
):
    """Apply the comparison-only LOS projector when explicitly requested."""

    method = str(
        adapter.get(
            "projected_profile_integration_method",
            NATIVE_BARYONFORGE_PROJECTION_METHOD,
        )
    )
    if method == NATIVE_BARYONFORGE_PROJECTION_METHOD:
        return profile
    if method != MATCHED_BARYONFORGE_PROJECTION_METHOD:
        raise ValueError(
            "Unsupported adapter.projected_profile_integration_method="
            f"{method!r}; expected {NATIVE_BARYONFORGE_PROJECTION_METHOD!r} or "
            f"{MATCHED_BARYONFORGE_PROJECTION_METHOD!r}."
        )

    MatchedLOSProjectionWrapper = make_matched_los_projection_wrapper(ccl)
    return MatchedLOSProjectionWrapper(
        profile,
        los_max_comoving_mpc=float(
            adapter["projected_profile_los_max_comoving_Mpc"]
        ),
        num_points=int(adapter["projected_profile_num_points"]),
        projected_scale_factor_power=float(projected_scale_factor_power),
    )


def make_matched_tsz_class(bfg):
    class MatchedThermalSZ(bfg.Profiles.Thermodynamic.ThermalSZ):
        """ThermalSZ with GODMAX's electron factor and PyCCL-compatible shapes."""

        def __init__(self, *args, electron_pressure_factor: float, **kwargs):
            self._matched_electron_pressure_factor = float(electron_pressure_factor)
            super().__init__(*args, **kwargs)

        def Pgas_to_Pe(self, cosmo, r, M, a):
            return self._matched_electron_pressure_factor

        def _real(self, cosmo, r, M, a):
            # BaryonForge ThermalSZ currently retains both axes even for
            # scalar inputs.  Its real-space projector assumes the standard
            # PyCCL shape convention and otherwise adds a spurious axis.
            profile = np.asarray(super()._real(cosmo, r, M, a))
            if np.ndim(r) == 0:
                profile = np.squeeze(profile, axis=-1)
            if np.ndim(M) == 0:
                profile = np.squeeze(profile, axis=0)
            return profile

    return MatchedThermalSZ


def make_cmb_convergence_class(ccl):
    class CMBConvergence(ccl.halos.profiles.HaloProfile):
        def __init__(self, surface_density, source_redshift: float):
            self.surface_density = surface_density
            self.a_source = 1.0 / (1.0 + float(source_redshift))
            self.p_keys = []
            super().__init__(mass_def=surface_density.mass_def)

        def _real(self, cosmo, r, M, a):
            return np.zeros((np.atleast_1d(M).size, np.atleast_1d(r).size))

        def _projected(self, cosmo, r, M, a):
            sigma_comoving = self.surface_density.projected(cosmo, r, M, a)
            sigma_critical = cosmo.sigma_critical(a_lens=a, a_source=self.a_source)
            return sigma_comoving / (float(a) ** 2 * sigma_critical)

    return CMBConvergence


def _one_halo_components(bfg, kwargs: Mapping[str, Any]):
    profile_kwargs = dict(kwargs)
    gas = bfg.Profiles.Schneider19.Gas(**profile_kwargs)
    stars = bfg.Profiles.Schneider19.Stars(**profile_kwargs)
    dark_matter = bfg.Profiles.Schneider19.DarkMatter(**profile_kwargs)
    collisionless = bfg.Profiles.Schneider19.CollisionlessMatter(
        gas=gas,
        stars=stars,
        darkmatter=dark_matter,
        **profile_kwargs,
    )
    # This intentionally omits TwoHalo and DarkMatterBaryon's global M_DMO/M_DMB
    # renormalization, neither of which is present in GODMAX rho_dmb.
    return gas, stars, collisionless, gas + stars + collisionless


def build_direct_models(params: Mapping[str, Any], cosmo):
    bfg, _, ccl = _scientific_imports()
    # PyCCL exposes the common 200c definition as a singleton instance.
    mass_def = ccl.halos.massdef.MassDef200c
    kwargs = baryonforge_profile_kwargs(params)
    kwargs.update(
        {
            "mass_def": mass_def,
            "c_M_relation": ccl.halos.concentration.ConcentrationDuffy08,
        }
    )

    # Pressure mutates cutoffs recursively, so its components must not be reused
    # by the independently painted convergence profile.
    gas_hse, _, _, matter_hse = _one_halo_components(bfg, kwargs)
    gas_kappa, stars_kappa, collisionless_kappa, matter_kappa = _one_halo_components(bfg, kwargs)

    pressure_total = bfg.Profiles.Thermodynamic.Pressure(
        gas=gas_hse,
        darkmatterbaryon=matter_hse,
        **kwargs,
    )
    alpha_nt = float(params["profile_parameters"]["alpha_nt"])
    if alpha_nt > 0.0:
        nt_kwargs = dict(kwargs)
        for key in ("alpha_nt", "nu_nt", "gamma_nt"):
            nt_kwargs.pop(key, None)
        nonthermal = bfg.Profiles.Thermodynamic.NonThermalFrac(
            alpha_nt=alpha_nt,
            nu_nt=float(params["profile_parameters"]["nu_nt"]),
            gamma_nt=float(params["profile_parameters"]["gamma_nt"]),
            **nt_kwargs,
        )
        pressure_thermal = pressure_total * (1.0 - nonthermal)
    else:
        pressure_thermal = pressure_total

    MatchedThermalSZ = make_matched_tsz_class(bfg)
    y_native = MatchedThermalSZ(
        pressure=pressure_thermal,
        electron_pressure_factor=float(params["adapter"]["electron_pressure_factor"]),
        **kwargs,
    )
    y_physical = bfg.Profiles.misc.ComovingToPhysical(y_native, factor=-3)
    ProjectionFactorWrapper = make_projection_factor_wrapper(ccl)
    if bool(params["adapter"]["remove_thermal_sz_extra_projected_a"]):
        y_direct = ProjectionFactorWrapper(y_physical, exponent=-1.0)
    else:
        y_direct = y_physical

    # The matched comparison uses one explicit comoving LOS definition in
    # both codes.  Matter is a comoving surface-density projection, whereas
    # Compton-y integrates physical pressure along physical path length and
    # therefore carries one additional factor of ``a``.
    matter_kappa = apply_projection_adapter(
        matter_kappa,
        ccl=ccl,
        adapter=params["adapter"],
        projected_scale_factor_power=0.0,
    )
    y_direct = apply_projection_adapter(
        y_direct,
        ccl=ccl,
        adapter=params["adapter"],
        projected_scale_factor_power=1.0,
    )

    CMBConvergence = make_cmb_convergence_class(ccl)
    kappa_direct = CMBConvergence(
        matter_kappa,
        source_redshift=float(params["adapter"]["cmb_source_redshift"]),
    )
    return {
        "mass_def": mass_def,
        "gas_direct": gas_kappa,
        "stars_direct": stars_kappa,
        "collisionless_direct": collisionless_kappa,
        "matter_direct": matter_kappa,
        "y_direct": y_direct,
        "kappa_direct": kappa_direct,
    }


def tabulate_projected_model(model, cosmo, params: Mapping[str, Any], *, smoke: bool, verbose: bool):
    bfg, _, ccl = _scientific_imports()
    tab = params["tabulation"]
    dims = dict(tab.get("smoke", {})) if smoke else {}
    table = bfg.utils.TabulatedProfile(model, cosmo)
    table.setup_interpolator(
        z_min=float(tab["z_min"]),
        z_max=float(tab["z_max"]),
        N_samples_z=int(dims.get("n_z", tab["n_z"])),
        z_linear_sampling=True,
        M_min=10.0 ** float(tab["log10_M_min_Msun"]),
        M_max=10.0 ** float(tab["log10_M_max_Msun"]),
        N_samples_Mass=int(dims.get("n_M", tab["n_M"])),
        R_min=float(tab["R_min_comoving_Mpc"]),
        R_max=float(tab["R_max_comoving_Mpc"]),
        N_samples_R=int(dims.get("n_R", tab["n_R"])),
        verbose=bool(verbose),
    )
    ProjectionFactorWrapper = make_projection_factor_wrapper(ccl)
    # BaryonForge TabulatedProfile stores projected training values multiplied by a.
    restored = ProjectionFactorWrapper(table, exponent=-1.0)
    return restored, table


def validate_table_nodes(direct, restored, table, cosmo) -> dict:
    z_grid = np.exp(np.asarray(table.raw_input_z_range)) - 1.0
    mass_grid = np.exp(np.asarray(table.raw_input_M_range))
    radius_grid = np.exp(np.asarray(table.raw_input_r_range))
    indices = (
        (0, 0, 0),
        (len(z_grid) // 2, len(mass_grid) // 2, len(radius_grid) // 2),
        (len(z_grid) - 1, len(mass_grid) - 1, len(radius_grid) - 1),
    )
    errors = []
    for iz, im, ir in indices:
        z = float(z_grid[iz])
        a = 1.0 / (1.0 + z)
        mass = float(mass_grid[im])
        radius = float(radius_grid[ir])
        # The BaryonForge real-space projector mishandles a scalar-radius,
        # scalar-mass pair while the painter always supplies a radius array.
        # Exercise the painter's actual call shape for this node check.
        radius_query = np.asarray([radius], dtype=np.float64)
        # The real-space projection quadrature changes its integration grid
        # with the full radius array supplied by the caller.  The training
        # value is therefore the only unambiguous node reference: it is the
        # direct profile evaluated on the exact table grid, with the table's
        # built-in factor of ``a`` undone.
        expected = float(np.asarray(table.raw_input_2D)[iz, im, ir] / a)
        actual = float(np.ravel(restored.projected(cosmo, radius_query, mass, a))[0])
        scale = max(abs(expected), 1.0e-300)
        errors.append(
            {
                "z": z,
                "mass_Msun": mass,
                "radius_comoving_Mpc": radius,
                "direct_training_value": expected,
                "tabulated": actual,
                "relative_error": abs(actual - expected) / scale,
            }
        )
    maximum = max(item["relative_error"] for item in errors)
    return {"ok": maximum < 1.0e-8, "max_relative_error": maximum, "nodes": errors}


def _heldout_cell_midpoints(grid: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Choose three table-cell midpoints spanning an in-domain interval."""

    targets = np.linspace(float(lower), float(upper), 5, dtype=np.float64)[1:-1]
    values = []
    for target in targets:
        index = int(np.clip(np.searchsorted(grid, target) - 1, 0, len(grid) - 2))
        values.append(0.5 * (float(grid[index]) + float(grid[index + 1])))
    return np.unique(np.asarray(values, dtype=np.float64))


def validate_table_interpolation(
    direct,
    restored,
    table,
    cosmo,
    domain_report: Mapping[str, Any],
    tolerance: float,
) -> dict:
    """Compare held-out interpolation points to direct fixed-grid projections."""

    raw_z = np.asarray(table.raw_input_z_range, dtype=np.float64)
    raw_m = np.asarray(table.raw_input_M_range, dtype=np.float64)
    raw_r = np.asarray(table.raw_input_r_range, dtype=np.float64)
    bounds = domain_report["catalog_query_bounds"]
    query_log1pz = _heldout_cell_midpoints(
        raw_z, math.log1p(bounds["z"][0]), math.log1p(bounds["z"][1])
    )
    query_logm = _heldout_cell_midpoints(
        raw_m,
        math.log(10.0) * bounds["log10_mass_Msun"][0],
        math.log(10.0) * bounds["log10_mass_Msun"][1],
    )
    query_logr = _heldout_cell_midpoints(
        raw_r,
        math.log(bounds["queried_comoving_radius_Mpc"][0]),
        math.log(bounds["queried_comoving_radius_Mpc"][1]),
    )
    query_m = np.exp(query_logm)
    query_r = np.exp(query_logr)
    evaluation_m = np.unique(np.concatenate([np.exp(raw_m), query_m]))
    evaluation_r = np.unique(np.concatenate([np.exp(raw_r), query_r]))
    mass_indices = np.searchsorted(evaluation_m, query_m)
    radius_indices = np.searchsorted(evaluation_r, query_r)

    samples = []
    relative_errors = []
    for log1pz in query_log1pz:
        redshift = math.exp(float(log1pz)) - 1.0
        a = 1.0 / (1.0 + redshift)
        direct_grid = np.asarray(
            direct.projected(cosmo, evaluation_r, evaluation_m, a), dtype=np.float64
        )
        expected = direct_grid[np.ix_(mass_indices, radius_indices)]
        actual = np.asarray(
            restored.projected(cosmo, query_r, query_m, a), dtype=np.float64
        )
        if actual.shape != expected.shape:
            raise ValueError(
                f"Held-out table shape mismatch: direct={expected.shape}, table={actual.shape}."
            )
        amplitude_floor = max(float(np.nanmax(np.abs(expected))) * 1.0e-10, 1.0e-300)
        valid = (
            np.isfinite(expected)
            & np.isfinite(actual)
            & (np.abs(expected) > amplitude_floor)
        )
        if not np.any(valid):
            raise ValueError("No finite, non-negligible held-out table validation points.")
        relative = np.abs(actual[valid] / expected[valid] - 1.0)
        relative_errors.extend(relative.tolist())
        for index in np.argwhere(valid):
            im, ir = (int(index[0]), int(index[1]))
            samples.append(
                {
                    "z": redshift,
                    "mass_Msun": float(query_m[im]),
                    "radius_comoving_Mpc": float(query_r[ir]),
                    "direct_fixed_grid": float(expected[im, ir]),
                    "tabulated": float(actual[im, ir]),
                    "relative_error": float(abs(actual[im, ir] / expected[im, ir] - 1.0)),
                }
            )
    maximum = float(np.max(relative_errors))
    return {
        "ok": maximum <= float(tolerance),
        "tolerance": float(tolerance),
        "max_relative_error": maximum,
        "median_relative_error": float(np.median(relative_errors)),
        "n_valid": int(len(relative_errors)),
        "samples": samples,
    }


def _load_catalog(path: Path, max_halos: int | None) -> tuple[dict, dict]:
    with h5py.File(path, "r") as handle:
        n_total = int(handle["z"].shape[0])
        stop = n_total if max_halos is None else min(n_total, int(max_halos))
        catalog = {
            name: np.asarray(handle[name][:stop])
            for name in (
                "ra_deg",
                "dec_deg",
                "z",
                "M200c_hMsun",
                "R200c_hMpc",
                "DA_hMpc",
                "source_row",
            )
        }
        attrs = {str(key): value for key, value in handle.attrs.items()}
    attrs["n_catalog_total"] = n_total
    attrs["n_catalog_used"] = stop
    return catalog, attrs


def native_painter_geometry(
    redshift: np.ndarray,
    mass_physical_msun: np.ndarray,
    *,
    cosmo,
    ccl,
    n_jobs: int,
    seed: int,
    max_paint: float,
) -> dict[str, Any]:
    """Reproduce the radius and distance geometry used by PaintProfilesShell.

    BaryonForge recomputes physical 200c radii with PyCCL.  Its HEALPix runner
    also approximates PyCCL's angular-diameter distance with a 1000-point cubic
    spline built independently inside every SplitJoinParallel worker.  The
    worker partition is therefore part of the native painting geometry.
    """

    from scipy import interpolate

    redshift = np.asarray(redshift, dtype=np.float64)
    mass_physical_msun = np.asarray(mass_physical_msun, dtype=np.float64)
    if redshift.ndim != 1 or mass_physical_msun.shape != redshift.shape:
        raise ValueError("BaryonForge geometry requires matching one-dimensional z and mass arrays.")
    if redshift.size == 0:
        raise ValueError("BaryonForge geometry requires at least one halo.")
    if int(n_jobs) < 1 or int(n_jobs) > redshift.size:
        raise ValueError(f"n_jobs must satisfy 1 <= n_jobs <= {redshift.size}, got {n_jobs}.")

    scale_factor = 1.0 / (1.0 + redshift)
    radius_physical_mpc = np.asarray(
        ccl.halos.massdef.MassDef200c.get_radius(
            cosmo,
            mass_physical_msun,
            scale_factor,
        ),
        dtype=np.float64,
    )
    distance_physical_mpc = np.empty_like(redshift)

    if int(n_jobs) == 1:
        partitions = [np.arange(redshift.size, dtype=np.int64)]
    else:
        shuffled = np.random.default_rng(int(seed)).choice(
            redshift.size,
            size=redshift.size,
            replace=False,
        )
        per_split = int(np.ceil(redshift.size / int(n_jobs)))
        partitions = [
            shuffled[index * per_split : (index + 1) * per_split]
            for index in range(int(n_jobs))
        ]
        if any(indices.size == 0 for indices in partitions):
            raise ValueError(
                "BaryonForge SplitJoinParallel would create an empty worker catalog for "
                f"halo_count={redshift.size}, n_jobs={n_jobs}. Use fewer workers."
            )

    partition_z_max = []
    for indices in partitions:
        worker_z = redshift[indices]
        worker_z_max = float(np.max(worker_z))
        distance_z = np.linspace(0.0, worker_z_max + 0.1, 1000)
        distance_grid = ccl.angular_diameter_distance(
            cosmo,
            1.0 / (1.0 + distance_z),
        )
        distance_spline = interpolate.CubicSpline(distance_z, distance_grid)
        distance_physical_mpc[indices] = distance_spline(worker_z)
        partition_z_max.append(worker_z_max)

    support_rad = float(max_paint) * radius_physical_mpc / distance_physical_mpc
    return {
        "radius_physical_mpc": radius_physical_mpc,
        "distance_physical_mpc": distance_physical_mpc,
        "support_rad": support_rad,
        "partition_z_max": partition_z_max,
        "n_jobs": int(n_jobs),
        "seed": int(seed),
    }


def validate_table_domain(
    path: Path,
    params: Mapping[str, Any],
    *,
    nside: int,
    max_paint: float,
    h: float,
    hp,
    cosmo,
    ccl,
    n_jobs: int,
    seed: int,
) -> dict:
    """Prove that every native painter query lies inside the BFG table."""

    with h5py.File(path, "r") as handle:
        ra = np.asarray(handle["ra_deg"][:], dtype=np.float64)
        dec = np.asarray(handle["dec_deg"][:], dtype=np.float64)
        redshift = np.asarray(handle["z"][:], dtype=np.float64)
        mass_physical = np.asarray(handle["M200c_hMsun"][:], dtype=np.float64) / float(h)
        radius_hmpc = np.asarray(handle["R200c_hMpc"][:], dtype=np.float64)
        distance_hmpc = np.asarray(handle["DA_hMpc"][:], dtype=np.float64)

    native_geometry = native_painter_geometry(
        redshift,
        mass_physical,
        cosmo=cosmo,
        ccl=ccl,
        n_jobs=int(n_jobs),
        seed=int(seed),
        max_paint=float(max_paint),
    )
    native_radius_mpc = native_geometry["radius_physical_mpc"]
    native_distance_mpc = native_geometry["distance_physical_mpc"]

    nearest = hp.ang2pix(int(nside), ra, dec, lonlat=True)
    halo_vectors = np.asarray(hp.ang2vec(ra, dec, lonlat=True), dtype=np.float64)
    pixel_vectors = np.stack(hp.pix2vec(int(nside), nearest), axis=1)
    chord_angle = np.sqrt(np.sum((halo_vectors - pixel_vectors) ** 2, axis=1))
    scale = 1.0 + redshift
    nearest_comoving_mpc = native_distance_mpc * chord_angle * scale
    maximum_comoving_mpc = float(max_paint) * native_radius_mpc * scale

    catalog_support_rad = float(max_paint) * radius_hmpc / distance_hmpc
    native_support_rad = np.asarray(native_geometry["support_rad"], dtype=np.float64)
    radius_ratio = native_radius_mpc * float(h) / radius_hmpc
    distance_ratio = native_distance_mpc * float(h) / distance_hmpc
    support_ratio = native_support_rad / catalog_support_rad

    tab = params["tabulation"]
    bounds = {
        "z": (float(np.min(redshift)), float(np.max(redshift))),
        "log10_mass_Msun": (
            float(np.log10(np.min(mass_physical))),
            float(np.log10(np.max(mass_physical))),
        ),
        "queried_comoving_radius_Mpc": (
            float(np.min(nearest_comoving_mpc)),
            float(np.max(maximum_comoving_mpc)),
        ),
    }
    table_bounds = {
        "z": (float(tab["z_min"]), float(tab["z_max"])),
        "log10_mass_Msun": (
            float(tab["log10_M_min_Msun"]),
            float(tab["log10_M_max_Msun"]),
        ),
        "queried_comoving_radius_Mpc": (
            float(tab["R_min_comoving_Mpc"]),
            float(tab["R_max_comoving_Mpc"]),
        ),
    }
    checks = {
        key: table_bounds[key][0] <= value[0] and value[1] <= table_bounds[key][1]
        for key, value in bounds.items()
    }
    report = {
        "ok": all(checks.values()),
        "n_halos": int(redshift.size),
        "catalog_query_bounds": bounds,
        "table_bounds": table_bounds,
        "checks": checks,
        "native_geometry": {
            "description": (
                "PyCCL MassDef200c radius and the native runner's 1000-point "
                "angular-diameter-distance CubicSpline in each SplitJoin worker"
            ),
            "n_jobs": int(native_geometry["n_jobs"]),
            "seed": int(native_geometry["seed"]),
            "partition_z_max": native_geometry["partition_z_max"],
            "radius_baryonforge_times_h_over_catalog": (
                float(np.min(radius_ratio)),
                float(np.max(radius_ratio)),
            ),
            "distance_baryonforge_times_h_over_catalog": (
                float(np.min(distance_ratio)),
                float(np.max(distance_ratio)),
            ),
            "support_baryonforge_over_catalog": (
                float(np.min(support_ratio)),
                float(np.max(support_ratio)),
            ),
            "catalog_support_angle_deg": (
                math.degrees(float(np.min(catalog_support_rad))),
                math.degrees(float(np.max(catalog_support_rad))),
            ),
            "baryonforge_support_angle_deg": (
                math.degrees(float(np.min(native_support_rad))),
                math.degrees(float(np.max(native_support_rad))),
            ),
        },
    }
    if not report["ok"]:
        raise ValueError(f"BaryonForge table does not enclose the painter domain: {report}")
    return report


def paint(
    config: Mapping[str, Any],
    *,
    nside: int,
    output: Path,
    smoke_table: bool,
    max_halos: int | None,
    n_jobs: int,
    overwrite: bool,
    verbose: bool,
    frozen_contract: Mapping[str, Any] | None = None,
) -> dict:
    frozen_contract = dict(
        frozen_contract
        if frozen_contract is not None
        else current_map_contract(config)
    )
    assert_map_contract_unchanged(
        frozen_contract,
        current_map_contract(config),
        context="BaryonForge pre-input-load validation",
    )
    crosswalk = validate_parameter_crosswalk(config)
    if not crosswalk["ok"]:
        raise ValueError(f"Parameter crosswalk failed: {crosswalk['failed']}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite explicitly.")
    catalog_path = resolve_path(config["catalog"]["output_h5"], config.get("_config_path"))
    params_path = resolve_path(config["baryonforge"]["params"], config.get("_config_path"))
    params = load_yaml(params_path)
    catalog, attrs = _load_catalog(catalog_path, max_halos)
    assert_map_contract_unchanged(
        frozen_contract,
        current_map_contract(config),
        context="BaryonForge post-input-load validation",
    )
    if len(catalog["z"]) == 0:
        raise ValueError("The BaryonForge painter requires at least one halo.")
    if int(n_jobs) < 1 or int(n_jobs) > len(catalog["z"]):
        raise ValueError(
            f"n_jobs must satisfy 1 <= n_jobs <= halo_count={len(catalog['z'])}, got {n_jobs}."
        )
    if max_halos is None:
        configured_n_jobs = int(config["baryonforge"]["n_jobs"])
        if smoke_table:
            raise ValueError(
                "A full-catalog product cannot use --smoke-table; reserve the coarse "
                "table for bounded --max-halos functional checks."
            )
        if int(n_jobs) != configured_n_jobs:
            raise ValueError(
                "Full production must use the configured SplitJoin partition because "
                "each worker builds its own distance spline: "
                f"n_jobs={n_jobs}, configured={configured_n_jobs}."
            )
    bfg, hp, ccl = _scientific_imports()
    cosmo_values = catalog_cosmology(attrs)
    cosmo = build_ccl_cosmology(cosmo_values)
    h = float(cosmo_values["h"])
    seed = int(config["pasting"]["random_seed"])
    max_paint = float(config["pasting"]["max_paint_R200c_factor"])
    table_domain = validate_table_domain(
        catalog_path,
        params,
        nside=int(nside),
        max_paint=max_paint,
        h=h,
        hp=hp,
        cosmo=cosmo,
        ccl=ccl,
        n_jobs=int(n_jobs),
        seed=seed,
    )
    native_geometry = native_painter_geometry(
        np.asarray(catalog["z"], dtype=np.float64),
        np.asarray(catalog["M200c_hMsun"], dtype=np.float64) / h,
        cosmo=cosmo,
        ccl=ccl,
        n_jobs=int(n_jobs),
        seed=seed,
        max_paint=max_paint,
    )
    support_rad = np.asarray(native_geometry["support_rad"], dtype=np.float64)
    catalog_support_rad = (
        max_paint
        * np.asarray(catalog["R200c_hMpc"], dtype=np.float64)
        / np.asarray(catalog["DA_hMpc"], dtype=np.float64)
    )
    max_pixel_radius = float(hp.max_pixrad(int(nside)))
    full_native_support_deg = table_domain["native_geometry"][
        "baryonforge_support_angle_deg"
    ]
    if math.radians(float(full_native_support_deg[0])) < max_pixel_radius:
        raise ValueError(
            "BaryonForge's native query_disc can return no pixel when the paint support "
            f"is smaller than the HEALPix covering radius: full-catalog min native support="
            f"{float(full_native_support_deg[0]):.6f} deg, "
            f"max_pixrad={math.degrees(max_pixel_radius):.6f} deg at NSIDE={nside}. "
            "Use NSIDE=1024 for the pinned comparison catalog."
        )

    model_start = time.perf_counter()
    direct = build_direct_models(params, cosmo)
    y_model, y_table = tabulate_projected_model(
        direct["y_direct"], cosmo, params, smoke=smoke_table, verbose=verbose
    )
    kappa_surface_model, matter_table = tabulate_projected_model(
        direct["matter_direct"], cosmo, params, smoke=smoke_table, verbose=verbose
    )
    _, _, ccl = _scientific_imports()
    CMBConvergence = make_cmb_convergence_class(ccl)
    kappa_model = CMBConvergence(
        kappa_surface_model,
        source_redshift=float(params["adapter"]["cmb_source_redshift"]),
    )
    node_checks = {
        "y": validate_table_nodes(direct["y_direct"], y_model, y_table, cosmo),
        "matter": validate_table_nodes(direct["matter_direct"], kappa_surface_model, matter_table, cosmo),
    }
    if not all(result["ok"] for result in node_checks.values()):
        raise RuntimeError(f"Tabulation node validation failed: {node_checks}")
    if smoke_table:
        interpolation_checks = {
            "status": "skipped_for_coarse_functional_smoke_table",
            "ok": None,
            "production_tolerance": float(params["tabulation"]["validation_max_relative_error"]),
        }
    else:
        tolerance = float(params["tabulation"]["validation_max_relative_error"])
        interpolation_checks = {
            "y": validate_table_interpolation(
                direct["y_direct"], y_model, y_table, cosmo, table_domain, tolerance
            ),
            "matter": validate_table_interpolation(
                direct["matter_direct"],
                kappa_surface_model,
                matter_table,
                cosmo,
                table_domain,
                tolerance,
            ),
        }
        if not all(result["ok"] for result in interpolation_checks.values()):
            raise RuntimeError(
                f"Held-out tabulation interpolation validation failed: {interpolation_checks}"
            )
    model_time = time.perf_counter() - model_start

    cosmo_dict = {
        "Omega_m": cosmo_values["Omega_m"],
        "Omega_b": cosmo_values["Omega_b"],
        "h": cosmo_values["h"],
        "sigma8": cosmo_values["sigma8"],
        "n_s": cosmo_values["n_s"],
        "w0": cosmo_values["w0"],
        "wa": 0.0,
    }
    halos = bfg.utils.HaloLightConeCatalog(
        ra=np.asarray(catalog["ra_deg"], dtype=np.float64),
        dec=np.asarray(catalog["dec_deg"], dtype=np.float64),
        M=np.asarray(catalog["M200c_hMsun"], dtype=np.float64) / h,
        z=np.asarray(catalog["z"], dtype=np.float64),
        cosmo=dict(cosmo_dict),
        source_row=np.asarray(catalog["source_row"], dtype=np.float64),
    )
    shell = bfg.utils.LightconeShell(
        map=np.zeros(hp.nside2npix(int(nside)), dtype=np.float64),
        cosmo=dict(cosmo_dict),
    )
    runner_kwargs = {
        "HaloLightConeCatalog": halos,
        "LightconeShell": shell,
        "epsilon_max": float(config["pasting"]["max_paint_R200c_factor"]),
        "mass_def": direct["mass_def"],
        "include_pixel_size": False,
        "verbose": bool(verbose),
    }
    def run_model(model):
        runner = bfg.Runners.PaintProfilesShell(model=model, **runner_kwargs)
        if int(n_jobs) == 1:
            return runner.process()
        return bfg.utils.SplitJoinParallel(
            runner,
            njobs=int(n_jobs),
            seed=int(config["pasting"]["random_seed"]),
        ).process()

    paint_start = time.perf_counter()
    ymap = run_model(y_model)
    y_time = time.perf_counter() - paint_start
    paint_start = time.perf_counter()
    kappamap = run_model(kappa_model)
    kappa_time = time.perf_counter() - paint_start

    assert_map_contract_unchanged(
        frozen_contract,
        current_map_contract(config),
        context="BaryonForge post-paint validation",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    complete_catalog_paint = bool(
        max_halos is None
        and not smoke_table
        and int(n_jobs) == int(config["baryonforge"]["n_jobs"])
        and int(len(catalog["z"])) == int(attrs["selection_rows"])
    )
    provenance = {
        **frozen_contract,
        "schema": MAP_PRODUCT_SCHEMA,
        "backend": "baryonforge",
        "nside": int(nside),
        "ordering": "RING",
        "catalog_path": frozen_contract["catalog_path"],
        "catalog_sha256": frozen_contract["catalog_sha256"],
        "params_path": frozen_contract["baryonforge_params_path"],
        "params_sha256": frozen_contract["baryonforge_params_sha256"],
        "mass_predicate": str(config["catalog"]["predicate"]),
        "selection_predicate": str(config["catalog"]["predicate"]),
        "halo_count": int(len(catalog["z"])),
        "n_halos_painted": int(len(catalog["z"])),
        "complete_catalog_paint": complete_catalog_paint,
        "halo_only": True,
        "z_min": float(np.min(catalog["z"])),
        "z_max": float(np.max(catalog["z"])),
        "max_paint_R200c_factor": max_paint,
        "smooth_profiles": False,
        "unit_boundary": {
            "catalog_mass": "M200c_hMsun in Msun/h",
            "catalog_radius": "R200c_hMpc is physical Mpc/h",
            "catalog_distance": "DA_hMpc is physical angular-diameter distance in Mpc/h",
            "painter_mass": "M200c_hMsun / h in physical Msun",
            "painter_geometry": "native PyCCL R200c and D_A in physical Mpc",
            "projected_profile_radius": "physical transverse separation / a in comoving Mpc",
            "map_ymap": "dimensionless Compton-y",
            "map_kappa_cmb": "dimensionless halo-only CMB convergence",
        },
        "node_checks": node_checks,
        "interpolation_checks": interpolation_checks,
        "model_setup_seconds": model_time,
        "y_paint_seconds": y_time,
        "kappa_paint_seconds": kappa_time,
        "smoke_table": bool(smoke_table),
        "max_halos": max_halos,
        "n_jobs": int(n_jobs),
        "baryonforge_splitjoin_n_jobs": int(n_jobs),
        "cosmology": cosmo_values,
        "minimum_support_angle_deg": math.degrees(float(np.min(support_rad))),
        "maximum_support_angle_deg": math.degrees(float(np.max(support_rad))),
        "catalog_minimum_support_angle_deg": math.degrees(
            float(np.min(catalog_support_rad))
        ),
        "catalog_maximum_support_angle_deg": math.degrees(
            float(np.max(catalog_support_rad))
        ),
        "support_baryonforge_over_catalog": (
            float(np.min(support_rad / catalog_support_rad)),
            float(np.max(support_rad / catalog_support_rad)),
        ),
        "native_geometry_partition_z_max": native_geometry["partition_z_max"],
        "healpix_max_pixel_radius_deg": math.degrees(max_pixel_radius),
        "table_domain": table_domain,
    }
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = provenance["schema"]
            handle.attrs["backend"] = "baryonforge"
            handle.attrs["nside"] = int(nside)
            handle.attrs["ordering"] = "RING"
            handle.attrs["catalog_sha256"] = provenance["catalog_sha256"]
            handle.attrs["catalog_path"] = provenance["catalog_path"]
            handle.attrs["params_sha256"] = provenance["params_sha256"]
            handle.attrs["comparison_config_sha256"] = provenance[
                "comparison_config_sha256"
            ]
            handle.attrs["source_manifest_sha256"] = provenance[
                "source_manifest_sha256"
            ]
            handle.attrs["effective_godmax_config_sha256"] = provenance[
                "effective_godmax_config_sha256"
            ]
            handle.attrs["selection_predicate"] = provenance["selection_predicate"]
            handle.attrs["halo_count"] = provenance["halo_count"]
            handle.attrs["halo_only"] = True
            handle.attrs["z_min"] = provenance["z_min"]
            handle.attrs["z_max"] = provenance["z_max"]
            handle.attrs["h"] = float(cosmo_values["h"])
            handle.attrs["H0"] = float(cosmo_values["H0"])
            handle.attrs["Omega_M"] = float(cosmo_values["Omega_m"])
            handle.attrs["Omega_b"] = float(cosmo_values["Omega_b"])
            handle.attrs["max_paint_R200c_factor"] = provenance["max_paint_R200c_factor"]
            handle.attrs["smooth_profiles"] = False
            handle.attrs["complete_catalog_paint"] = provenance[
                "complete_catalog_paint"
            ]
            handle.attrs["n_halos_painted"] = provenance["n_halos_painted"]
            handle.attrs["noise_policy"] = provenance["noise_policy"]
            handle.attrs["provisional_status"] = provenance[
                "provisional_status"
            ]
            maps = handle.create_group("maps")
            maps.create_dataset("map_ymap", data=np.asarray(ymap, dtype=np.float32), compression="lzf")
            maps.create_dataset("map_kappa_cmb", data=np.asarray(kappamap, dtype=np.float32), compression="lzf")
            group = handle.create_group("provenance")
            group.attrs["json"] = canonical_json(provenance)
            group.create_dataset("source_row", data=np.asarray(catalog["source_row"], dtype=np.int64), compression="lzf")
        assert_map_contract_unchanged(
            frozen_contract,
            current_map_contract(config),
            context="BaryonForge pre-publication validation",
        )
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    provenance["output_h5"] = str(output)
    return provenance


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--nside", type=int)
    parser.add_argument("--output")
    parser.add_argument("--smoke-table", action="store_true")
    parser.add_argument("--max-halos", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, frozen_contract = load_config_and_freeze_map_contract(args.config)
    nside = int(args.nside or config["pasting"]["nside"])
    output = resolve_path(args.output or config["baryonforge"]["output_h5"], config["_config_path"])
    report = paint(
        config,
        nside=nside,
        output=output,
        smoke_table=bool(args.smoke_table),
        max_halos=args.max_halos,
        n_jobs=int(args.n_jobs or config["baryonforge"].get("n_jobs", 1)),
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
        frozen_contract=frozen_contract,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
