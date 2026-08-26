"""Utilities for comparing HMC and SBI on analytical GODMAX Cls.

The setup uses one Backlight/DESI-like lens bin and a fixed Gaussian covariance
from a versioned fiducial theory product.  Callers can select either the full
GODMAX spectra or the map-matched resolved spectra.
"""

from __future__ import annotations

import copy
import json
import pathlib
import pickle
import sys
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import scipy.linalg

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.scipy.integrate as jsi
import jax.scipy.linalg as jsl

from gaussian_covariance import TARGET_SPECTRA
from fiducial_theory_datavector import (
    DEFAULT_PAINT_R200C_FACTOR,
    build_and_save_fiducial,
    build_notebook_matched_config,
    compute_ne0_cm3,
    ensure_repo_paths,
    load_validation_product,
)


THIS_DIR = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "outputs"
THEORY_SBI_DIR = OUTPUT_DIR / "theory_sbi"
DEFAULT_FIDUCIAL_PATH = THEORY_SBI_DIR / "fiducial_thetaej2_nuejm_minus0p1.npz"


@dataclass(frozen=True)
class ParameterSpec:
    """A sampled model parameter."""

    name: str
    label: str
    fiducial: float
    prior_min: float
    prior_max: float
    target: str = "sim"


@dataclass(frozen=True)
class DataSelection:
    """Selected probe/ell subset of a saved datavector product."""

    probes: tuple[str, ...]
    ell_min: float | None
    ell_max: float | None
    indices: np.ndarray
    ell_indices: np.ndarray
    native_ell_indices: np.ndarray
    native_nell: int
    labels: tuple[str, ...]


def default_parameter_specs() -> list[ParameterSpec]:
    """Return the starter baryonic parameter setup requested for this test."""

    return [
        ParameterSpec(
            name="theta_ej_0",
            label=r"\theta_{\rm ej,0}",
            fiducial=2.0,
            prior_min=0.5,
            prior_max=4.0,
            target="sim",
        ),
        ParameterSpec(
            name="nu_theta_ej_M",
            label=r"\nu^M_{\theta_{\rm ej}}",
            fiducial=-0.1,
            prior_min=-1.0,
            prior_max=0.0,
            target="sim",
        ),
    ]


def parse_param_specs(values: Sequence[str] | None) -> list[ParameterSpec]:
    """Parse CLI parameter specs.

    Format per entry:
    ``name:fiducial:prior_min:prior_max[:label[:target]]``.
    """

    if not values:
        return default_parameter_specs()

    specs: list[ParameterSpec] = []
    for item in values:
        parts = item.split(":")
        if len(parts) < 4:
            raise ValueError(
                "Parameter specs must be name:fiducial:prior_min:prior_max[:label[:target]]"
            )
        name = parts[0]
        fiducial = float(parts[1])
        prior_min = float(parts[2])
        prior_max = float(parts[3])
        label = parts[4] if len(parts) >= 5 and parts[4] else name
        target = parts[5] if len(parts) >= 6 and parts[5] else "sim"
        specs.append(ParameterSpec(name, label, fiducial, prior_min, prior_max, target))
    return specs


def parse_probe_list(probes: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize a probe list and validate it against the fiducial datavector."""

    if isinstance(probes, str):
        out = tuple(p.strip() for p in probes.split(",") if p.strip())
    else:
        out = tuple(str(p).strip() for p in probes if str(p).strip())
    unknown = sorted(set(out).difference(TARGET_SPECTRA))
    if unknown:
        raise ValueError(f"Unknown probes {unknown}; allowed probes are {TARGET_SPECTRA}")
    if not out:
        raise ValueError("At least one probe must be selected")
    return out


def fiducial_override_dict(param_specs: Sequence[ParameterSpec]) -> Dict[str, float]:
    """Return simulation-parameter overrides from fiducial values."""

    return {
        spec.name: float(spec.fiducial)
        for spec in param_specs
        if spec.target in ("sim", "cosmo")
    }


def ensure_default_fiducial_product(
    output_path: pathlib.Path | str = DEFAULT_FIDUCIAL_PATH,
    param_specs: Sequence[ParameterSpec] | None = None,
    force: bool = False,
    theory_mode: str = "map_matched_resolved",
) -> pathlib.Path:
    """Create the versioned fiducial theory/covariance product if needed."""

    output_path = pathlib.Path(output_path)
    param_specs = list(param_specs or default_parameter_specs())
    if output_path.exists() and not force:
        return output_path

    sim_overrides = {
        spec.name: float(spec.fiducial)
        for spec in param_specs
        if spec.target == "sim"
    }
    other_overrides = {
        spec.name: float(spec.fiducial)
        for spec in param_specs
        if spec.target == "other"
    }
    cosmo_overrides = {
        f"cosmo.{spec.name}": float(spec.fiducial)
        for spec in param_specs
        if spec.target == "cosmo"
    }
    sim_overrides.update(cosmo_overrides)

    build_and_save_fiducial(
        output_path=output_path,
        theory_mode=theory_mode,
        paint_r200c_factor=DEFAULT_PAINT_R200C_FACTOR,
        sim_param_overrides=sim_overrides,
        other_param_overrides=other_overrides,
    )
    return output_path


def build_data_selection(
    product: Mapping[str, object],
    probes: str | Sequence[str] = TARGET_SPECTRA,
    ell_min: float | None = None,
    ell_max: float | None = None,
) -> DataSelection:
    """Return datavector indices for a chosen probe and ell subset."""

    selected_probes = parse_probe_list(probes)
    ell = np.asarray(product["ell"], dtype=float)
    spectra_order = tuple(product["spectra_order"])
    nell = len(ell)
    ell_mask = np.ones(nell, dtype=bool)
    if ell_min is not None:
        ell_mask &= ell >= float(ell_min)
    if ell_max is not None:
        ell_mask &= ell <= float(ell_max)
    ell_indices = np.flatnonzero(ell_mask)
    if len(ell_indices) == 0:
        raise ValueError("The requested ell cuts remove every bin")

    ell_selection = product.get("metadata", {}).get("ell_selection", {})
    native_grid_indices = np.asarray(
        ell_selection.get("selected_native_ell_indices", np.arange(nell)),
        dtype=int,
    )
    if native_grid_indices.shape != (nell,):
        raise ValueError(
            "Saved native ell-index map is inconsistent with the product ell grid"
        )
    native_nell = int(ell_selection.get("original_nell", nell))
    if np.any(native_grid_indices < 0) or np.any(native_grid_indices >= native_nell):
        raise ValueError("Saved native ell indices lie outside the declared native grid")

    indices: list[int] = []
    labels: list[str] = []
    for probe in selected_probes:
        if probe not in spectra_order:
            raise ValueError(f"Probe {probe} is not in saved spectra_order={spectra_order}")
        block = spectra_order.index(probe)
        for iell in ell_indices:
            indices.append(block * nell + int(iell))
            labels.append(f"{probe}:ell={ell[iell]:.6g}")

    return DataSelection(
        probes=selected_probes,
        ell_min=None if ell_min is None else float(ell_min),
        ell_max=None if ell_max is None else float(ell_max),
        indices=np.asarray(indices, dtype=int),
        ell_indices=ell_indices.astype(int),
        native_ell_indices=native_grid_indices[ell_indices],
        native_nell=native_nell,
        labels=tuple(labels),
    )


def selected_product_arrays(
    product_path: pathlib.Path | str,
    probes: str | Sequence[str] = TARGET_SPECTRA,
    ell_min: float | None = None,
    ell_max: float | None = None,
) -> Dict[str, object]:
    """Load the selected observed vector and fixed covariance."""

    product = load_validation_product(product_path)
    selection = build_data_selection(product, probes=probes, ell_min=ell_min, ell_max=ell_max)
    idx = selection.indices
    data_vector = np.asarray(product["data_vector"], dtype=float)[idx]
    cov = np.asarray(product["cov"], dtype=float)[np.ix_(idx, idx)]
    cov = 0.5 * (cov + cov.T)
    chol, jitter = stable_cholesky(cov)
    precision = scipy.linalg.cho_solve((chol, True), np.eye(len(data_vector)))
    return {
        "product": product,
        "selection": selection,
        "data_vector": data_vector,
        "cov": cov,
        "chol": chol,
        "precision": precision,
        "jitter": jitter,
    }


def stable_cholesky(cov: np.ndarray, jitter_fraction: float = 1.0e-10) -> tuple[np.ndarray, float]:
    """Return a lower factor using correlation-scaled linear algebra."""

    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    if not np.all(np.isfinite(cov)) or np.any(diag <= 0.0):
        raise ValueError("Covariance must be finite with a strictly positive diagonal")
    scale = np.sqrt(diag)
    corr = cov / np.outer(scale, scale)
    corr = 0.5 * (corr + corr.T)
    try:
        return scale[:, None] * np.linalg.cholesky(corr), 0.0
    except np.linalg.LinAlgError:
        eig_min = float(np.min(np.linalg.eigvalsh(corr)))
        jitter = max(float(jitter_fraction), -eig_min + float(jitter_fraction))
        corr = corr + np.eye(corr.shape[0]) * jitter
        return scale[:, None] * np.linalg.cholesky(corr), jitter


def _apply_theta_to_dicts(
    sim_params_dict: dict,
    other_params_dict: dict,
    theta: jnp.ndarray,
    param_specs: Sequence[ParameterSpec],
) -> None:
    """Apply sampled values to GODMAX parameter dictionaries in place."""

    for ip, spec in enumerate(param_specs):
        value = theta[ip]
        if spec.target == "sim":
            sim_params_dict[spec.name] = value
        elif spec.target == "cosmo":
            sim_params_dict["cosmo"][spec.name] = value
        elif spec.target == "other":
            other_params_dict[spec.name] = value
        else:
            raise ValueError(f"Unknown parameter target {spec.target!r} for {spec.name}")


def _interp_transform_to_kpk_jax(k_src: jnp.ndarray, uk_src: jnp.ndarray,
                                 k_dst: jnp.ndarray) -> jnp.ndarray:
    """Interpolate profile transforms from FFTLog k to the halo-model k grid."""

    log_k_src = jnp.log(k_src)
    log_k_dst = jnp.log(k_dst)
    uk_flat = jnp.reshape(jnp.transpose(uk_src, (1, 2, 0)), (-1, uk_src.shape[0]))

    def interp_one(vals):
        return jnp.exp(
            jnp.interp(log_k_dst, log_k_src, jnp.log(jnp.clip(vals, 1.0e-300, jnp.inf)))
        )

    interp_flat = jax.vmap(interp_one)(uk_flat)
    return jnp.transpose(
        jnp.reshape(interp_flat, (uk_src.shape[1], uk_src.shape[2], k_dst.shape[0])),
        (2, 0, 1),
    )


def _truncated_profile_transforms_jax(pkz, paint_r200c_factor: float) -> Dict[str, jnp.ndarray]:
    """JAX version of the map-support truncation used in the saved theory product."""

    ensure_repo_paths()
    from mcfitjax.cosmology_jax import xi2P

    r_comoving = pkz.r_array[:, None, None]
    r200c_comoving = pkz.r200c_mat[None, :, :]
    mask = r_comoving <= float(paint_r200c_factor) * r200c_comoving
    xi2p = xi2P(pkz.r_array, nx=pkz.nr, lowring=True)

    k_src, uk_y_src = xi2p(pkz.y3d_mat * mask, axis=0, extrap=False)
    _, uk_ne_src = xi2p(pkz.ne_mat * mask, axis=0, extrap=False)
    _, uk_dmb_src = xi2p(
        pkz.rho_dmb_mat * mask / pkz.Mtot_mat[None, :, :],
        axis=0,
        extrap=False,
    )

    return {
        "uk_y": _interp_transform_to_kpk_jax(k_src, uk_y_src, pkz.kPk_array),
        "uk_ne": _interp_transform_to_kpk_jax(k_src, uk_ne_src, pkz.kPk_array),
        "uk_dmb": _interp_transform_to_kpk_jax(k_src, uk_dmb_src, pkz.kPk_array),
    }


def _trapz_lnm_jax(values: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    return jsi.trapezoid(values, x=jnp.log(mass), axis=-1)


def _interp_extrap_1d(x: jnp.ndarray, xp: jnp.ndarray, fp: jnp.ndarray) -> jnp.ndarray:
    """Linear 1D interpolation with linear extrapolation at both ends."""

    x = jnp.asarray(x)
    idx = jnp.searchsorted(xp, x, side="right") - 1
    idx = jnp.clip(idx, 0, xp.shape[0] - 2)
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    slope = (y1 - y0) / jnp.clip(x1 - x0, 1.0e-300, jnp.inf)
    return y0 + slope * (x - x0)


def _project_power_to_cl_jax(
    cls,
    power_kz: jnp.ndarray,
    prefactor1: jnp.ndarray,
    prefactor2: jnp.ndarray,
) -> jnp.ndarray:
    """Project custom P(k,z) to Cl using the saved map-matched convention."""

    ell = cls.ell_array
    z_grid = cls.z_array
    z_for = cls.z_array_for_Cls
    chi = cls.chi_array_for_Cls
    dchi_dz = cls.dchi_dz_array_for_Cls
    logk = jnp.log(cls.kPk_array)
    log_power = jnp.log(jnp.clip(power_kz, 1.0e-300, jnp.inf))
    log_power_zfor_by_k = jax.vmap(
        lambda row: _interp_extrap_1d(z_for, z_grid, row)
    )(log_power)

    def one_ell(ell_val):
        k_for = (ell_val + 0.5) / jnp.clip(chi, 1.0)
        log_pk_for = jax.vmap(
            lambda logk_val, col: _interp_extrap_1d(logk_val, logk, col),
            in_axes=(0, 1),
        )(jnp.log(k_for), log_power_zfor_by_k)
        integrand = (
            prefactor1
            * prefactor2
            * chi ** 2
            * dchi_dz
            * jnp.exp(log_pk_for)
        )
        return jsi.trapezoid(integrand, x=z_for)

    return jax.vmap(one_ell)(ell)


def _build_map_matched_signal_cls_jax(
    context: Mapping[str, object],
    paint_r200c_factor: float = DEFAULT_PAINT_R200C_FACTOR,
) -> Dict[str, jnp.ndarray]:
    """Return JAX map-matched signal spectra for HMC/SBI."""

    pkz = context["pkz"]
    cls = context["cls"]
    tr = _truncated_profile_transforms_jax(pkz, paint_r200c_factor)

    hmf_3 = pkz.hmf_Mz_mat[None, :, :]
    bias_3 = pkz.bias_Mz_mat[None, :, :]
    mtot_3 = pkz.Mtot_mat[None, :, :]
    mass = pkz.M_array
    plin = pkz.plin_kz_mat
    rhom0 = pkz.rhom_0

    ukg = pkz.ukg_cross
    ukg_auto = pkz.ukg_auto_sqr
    y_field = tr["uk_y"]
    e_field = tr["uk_ne"]
    m_field = mtot_3 * tr["uk_dmb"] / rhom0

    pgg_1h = _trapz_lnm_jax(ukg_auto * hmf_3, mass)
    pgy_1h = _trapz_lnm_jax(ukg * y_field * hmf_3, mass)
    pge_1h = _trapz_lnm_jax(ukg * e_field * hmf_3, mass)
    pgm_1h = _trapz_lnm_jax(ukg * m_field * hmf_3, mass)

    bg = pkz.bg_kz_mat
    by = _trapz_lnm_jax(y_field * hmf_3 * bias_3, mass)
    be = _trapz_lnm_jax(e_field * hmf_3 * bias_3, mass)
    bm = _trapz_lnm_jax(m_field * hmf_3 * bias_3, mass)

    pgg_2h = bg * bg * plin
    pgy_2h = bg * by * plin
    pge_2h = bg * be * plin
    pgm_2h = bg * bm * plin

    pyy = _trapz_lnm_jax(y_field * y_field * hmf_3, mass) + by * by * plin
    pye = _trapz_lnm_jax(y_field * e_field * hmf_3, mass) + by * be * plin
    pee = _trapz_lnm_jax(e_field * e_field * hmf_3, mass) + be * be * plin
    pym = _trapz_lnm_jax(y_field * m_field * hmf_3, mass) + by * bm * plin
    pem = _trapz_lnm_jax(e_field * m_field * hmf_3, mass) + be * bm * plin
    pmm = _trapz_lnm_jax(m_field * m_field * hmf_3, mass) + bm * bm * plin

    chi = cls.chi_array_for_Cls
    dchi = cls.dchi_dz_array_for_Cls
    z_for = cls.z_array_for_Cls
    pref_g = cls.Wg_mat[0] / (dchi * chi ** 2)
    pref_y = cls.Wy_array / chi ** 2
    pref_tau_abs = cls.const_coeff_tau * (1.0 + z_for) ** 2 / chi ** 2
    pref_kappa = (
        (1.0 + cls.mult_shear_bias_array[0])
        * jnp.squeeze(cls.Wk_mat[0])
        / chi ** 2
    )
    beam = jnp.exp(-0.5 * cls.ell_array * (cls.ell_array + 1.0) * cls.sig_beam ** 2)

    gg = _project_power_to_cl_jax(cls, pgg_1h + pgg_2h, pref_g, pref_g)
    gy = _project_power_to_cl_jax(cls, pgy_1h + pgy_2h, pref_g, pref_y) * beam
    gtau = _project_power_to_cl_jax(cls, pge_1h + pge_2h, pref_g, pref_tau_abs) * beam
    gkappa = _project_power_to_cl_jax(cls, pgm_1h + pgm_2h, pref_g, pref_kappa) * beam

    return {
        "gg": gg,
        "gy": gy,
        "gtau": gtau,
        "gkappa": gkappa,
        "yy": _project_power_to_cl_jax(cls, pyy, pref_y, pref_y) * beam ** 2,
        "ytau": _project_power_to_cl_jax(cls, pye, pref_y, pref_tau_abs) * beam ** 2,
        "tautau": _project_power_to_cl_jax(cls, pee, pref_tau_abs, pref_tau_abs) * beam ** 2,
        "ykappa": _project_power_to_cl_jax(cls, pym, pref_y, pref_kappa) * beam ** 2,
        "taukappa": _project_power_to_cl_jax(cls, pem, pref_tau_abs, pref_kappa) * beam ** 2,
        "kappakappa": _project_power_to_cl_jax(cls, pmm, pref_kappa, pref_kappa) * beam ** 2,
    }


def _build_full_target_signal_cls_jax(
    cls,
    tau_physical_correction: float,
) -> Dict[str, jnp.ndarray]:
    """Return the four target spectra from the existing full-theory branch."""

    tau_correction = jnp.asarray(tau_physical_correction)
    beam = jnp.asarray(cls.Bl_mat[:, 0])
    return {
        "gg": jnp.asarray(cls.Cl_gal_gal_tot_mat[:, 0, 0]),
        "gy": jnp.asarray(cls.Cl_gal_y_tot_mat[:, 0]) / jnp.clip(
            beam, jnp.finfo(beam.dtype).tiny
        ),
        "gtau": jnp.asarray(cls.Cl_gal_tau_tot_mat[:, 0]) * tau_correction,
        "gkappa": jnp.asarray(cls.Cl_gal_kappa_tot_mat[:, 0, 0]),
    }


def make_theory_vector_function(
    param_specs: Sequence[ParameterSpec],
    selection: DataSelection,
    gal_zmin: float = 0.4,
    gal_zmax: float = 0.6,
    nbar_comoving: float = 1.0e-4,
    hod_mass_cut: float = 1.0e13,
    kappa_source: str = "cmb",
    paint_r200c_factor: float = DEFAULT_PAINT_R200C_FACTOR,
    jit_compile: bool = True,
    fixed_param_specs: Sequence[ParameterSpec] | None = None,
    theory_mode: str = "map_matched_resolved",
    remove_galaxy_baryon_suppression: bool = True,
):
    """Create a JAX callable returning the selected theory datavector.

    ``param_specs`` defines the entries supplied in the callable's ``theta``
    argument.  ``fixed_param_specs`` applies additional parameter values at
    their ``fiducial`` settings before the sampled entries are applied.  This
    is needed when sampling a subset of the parameters used to create a saved
    fiducial product; otherwise omitted parameters would silently fall back to
    the defaults in the base GODMAX configuration.
    """

    fixed_param_specs = tuple(fixed_param_specs or ())
    if theory_mode not in ("full", "map_matched_resolved"):
        raise ValueError("theory_mode must be 'full' or 'map_matched_resolved'")
    sampled_names = {spec.name for spec in param_specs}
    fixed_names = {spec.name for spec in fixed_param_specs}
    overlap = sampled_names.intersection(fixed_names)
    if overlap:
        raise ValueError(
            "Parameters cannot be both sampled and fixed: "
            f"{sorted(overlap)}"
        )

    ensure_repo_paths()
    from base_class import base_class
    from get_radial_profiles import Profiles
    from get_Pkzs import get_Pkz
    from get_Cls import get_Cl

    base_config = build_notebook_matched_config(
        gal_zmin=gal_zmin,
        gal_zmax=gal_zmax,
        nbar_comoving=nbar_comoving,
        kappa_source=kappa_source,
    )
    (
        base_sim_params_dict,
        base_halo_params_dict,
        base_analysis_dict,
        base_other_params_dict,
        _cosmo_jax,
        _zarray_lens,
        _nz_lens,
        _gal_zrange,
    ) = base_config
    fixed_theta = jnp.asarray(fiducial_theta(fixed_param_specs))
    tau_physical_correction = compute_ne0_cm3(base_sim_params_dict["cosmo"]) * (
        1.0 + 0.5 * (gal_zmin + gal_zmax)
    ) ** 3
    target_spectra = tuple(TARGET_SPECTRA)
    direct_indices = []
    for probe in selection.probes:
        block = target_spectra.index(probe)
        direct_indices.extend(
            block * selection.native_nell + int(iell)
            for iell in selection.native_ell_indices
        )
    selected_indices = jnp.asarray(direct_indices, dtype=jnp.int32)

    def vector_fn(theta):
        theta = jnp.asarray(theta)
        sim_params_dict = copy.deepcopy(base_sim_params_dict)
        halo_params_dict = copy.deepcopy(base_halo_params_dict)
        analysis_dict = copy.deepcopy(base_analysis_dict)
        other_params_dict = copy.deepcopy(base_other_params_dict)
        _apply_theta_to_dicts(
            sim_params_dict,
            other_params_dict,
            fixed_theta,
            fixed_param_specs,
        )
        _apply_theta_to_dicts(sim_params_dict, other_params_dict, theta, param_specs)

        base_obj = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
        profiles = Profiles(
            sim_params_dict,
            halo_params_dict,
            analysis_dict,
            other_params_dict,
            base_class_obj=base_obj,
        )
        mass_mask = jnp.where(profiles.M_array > hod_mass_cut, 1.0, 0.0)
        mass_mask_2d = jnp.tile(mass_mask, (halo_params_dict["nz"], 1))
        profiles.Ncen_mat = profiles.Ncen_mat * mass_mask_2d
        profiles.Nsat_mat = profiles.Nsat_mat * mass_mask_2d

        pkz = get_Pkz(
            sim_params_dict,
            halo_params_dict,
            analysis_dict,
            other_params_dict,
            Profiles_obj=profiles,
        )
        if remove_galaxy_baryon_suppression:
            pkz.Pgg_tot_mat = pkz.Pgg_1h_kz_mat + pkz.Pgg_2h_kz_mat
            pkz.Pgm_tot_mat = pkz.Pgm_1h_kz_mat + pkz.Pgm_2h_kz_mat
            pkz.Pgm_nfw_tot_mat = pkz.Pgm_nfw_1h_kz_mat + pkz.Pgm_nfw_2h_kz_mat
        cls = get_Cl(
            sim_params_dict,
            halo_params_dict,
            analysis_dict,
            other_params_dict,
            Pkz_obj=pkz,
        )
        context = {"pkz": pkz, "cls": cls}
        if theory_mode == "full":
            cls_signal = _build_full_target_signal_cls_jax(
                cls,
                tau_physical_correction=tau_physical_correction,
            )
        else:
            cls_signal = _build_map_matched_signal_cls_jax(
                context,
                paint_r200c_factor=paint_r200c_factor,
            )
        full_vector = jnp.concatenate([cls_signal[spec] for spec in target_spectra])
        return full_vector[selected_indices]

    return jax.jit(vector_fn) if jit_compile else vector_fn


def make_offset_corrected_theory_vector_function(
    param_specs: Sequence[ParameterSpec],
    selection: DataSelection,
    fiducial_vector: np.ndarray,
    jit_compile: bool = True,
    **kwargs,
):
    """Return a theory callable with a constant fiducial numerical offset.

    The direct JAX projection is used for derivatives and parameter response.
    The constant offset enforces exact agreement with the saved fiducial product
    at ``theta_fiducial``; it is a numerical projection alignment, not a
    physical calibration.
    """

    raw_fn = make_theory_vector_function(
        param_specs,
        selection,
        jit_compile=jit_compile,
        **kwargs,
    )
    theta0 = jnp.asarray(fiducial_theta(param_specs))
    offset = jnp.asarray(fiducial_vector) - raw_fn(theta0)

    def vector_fn(theta):
        return raw_fn(theta) + offset

    return jax.jit(vector_fn) if jit_compile else vector_fn


def make_linearized_theory_vector_function(
    param_specs: Sequence[ParameterSpec],
    selection: DataSelection,
    fiducial_vector: np.ndarray,
    jit_compile: bool = True,
    **kwargs,
):
    """Return a fast fiducial-centered linear response model.

    The Jacobian is computed from the offset-corrected direct analytical
    evaluator.  This backend is useful for running HMC/SBI convergence tests
    when the exact map-matched evaluator is too expensive for thousands of
    likelihood calls.
    """

    direct_fn = make_offset_corrected_theory_vector_function(
        param_specs,
        selection,
        fiducial_vector=fiducial_vector,
        jit_compile=jit_compile,
        **kwargs,
    )
    theta0 = jnp.asarray(fiducial_theta(param_specs))
    mu0 = jnp.asarray(fiducial_vector)
    jac = jax.jacfwd(direct_fn)(theta0)

    def vector_fn(theta):
        return mu0 + jac @ (jnp.asarray(theta) - theta0)

    return (jax.jit(vector_fn) if jit_compile else vector_fn), {
        "theta0": np.asarray(theta0, dtype=float),
        "mu0": np.asarray(mu0, dtype=float),
        "jacobian": np.asarray(jac, dtype=float),
    }


def make_inference_theory_vector_function(
    param_specs: Sequence[ParameterSpec],
    selection: DataSelection,
    fiducial_vector: np.ndarray,
    backend: str = "linearized",
    fiducial_offset: bool = True,
    jit_compile: bool = True,
    **kwargs,
):
    """Create the theory vector callable used by HMC/SBI runners."""

    if backend == "linearized":
        if not fiducial_offset:
            raise ValueError("The linearized backend requires fiducial_offset=True")
        return make_linearized_theory_vector_function(
            param_specs,
            selection,
            fiducial_vector=fiducial_vector,
            jit_compile=jit_compile,
            **kwargs,
        )
    if backend == "direct":
        if fiducial_offset:
            fn = make_offset_corrected_theory_vector_function(
                param_specs,
                selection,
                fiducial_vector=fiducial_vector,
                jit_compile=jit_compile,
                **kwargs,
            )
            return fn, {}
        fn = make_theory_vector_function(
            param_specs,
            selection,
            jit_compile=jit_compile,
            **kwargs,
        )
        return fn, {}
    raise ValueError("backend must be 'linearized' or 'direct'")


def fiducial_theta(param_specs: Sequence[ParameterSpec]) -> np.ndarray:
    return np.asarray([spec.fiducial for spec in param_specs], dtype=float)


def prior_bounds(param_specs: Sequence[ParameterSpec]) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray([spec.prior_min for spec in param_specs], dtype=float)
    high = np.asarray([spec.prior_max for spec in param_specs], dtype=float)
    return low, high


def validate_theory_vector(
    vector_fn,
    selected: Mapping[str, object],
    param_specs: Sequence[ParameterSpec],
) -> Dict[str, float | bool]:
    """Compare the JAX evaluator to the saved fiducial vector."""

    theta0 = jnp.asarray(fiducial_theta(param_specs))
    pred = np.asarray(vector_fn(theta0), dtype=float)
    truth = np.asarray(selected["data_vector"], dtype=float)
    abs_diff = np.abs(pred - truth)
    rel_diff = abs_diff / np.clip(np.abs(truth), 1.0e-300, np.inf)

    def scalar_loglike(theta):
        mu = vector_fn(theta)
        resid = jnp.asarray(selected["data_vector"]) - mu
        y = jsl.solve_triangular(jnp.asarray(selected["chol"]), resid, lower=True)
        return -0.5 * jnp.dot(y, y)

    grad = np.asarray(jax.grad(scalar_loglike)(theta0), dtype=float)
    return {
        "finite_prediction": bool(np.all(np.isfinite(pred))),
        "max_abs_diff": float(np.max(abs_diff)),
        "max_rel_diff": float(np.max(rel_diff)),
        "median_rel_diff": float(np.median(rel_diff)),
        "finite_gradient": bool(np.all(np.isfinite(grad))),
        "gradient_norm": float(np.linalg.norm(grad)),
    }


def metadata_json(
    param_specs: Sequence[ParameterSpec],
    selection: DataSelection,
    extra: Mapping[str, object] | None = None,
    fixed_param_specs: Sequence[ParameterSpec] | None = None,
) -> str:
    payload = {
        "parameter_specs": [asdict(spec) for spec in param_specs],
        "fixed_parameter_specs": [
            asdict(spec) for spec in (fixed_param_specs or ())
        ],
        "selection": {
            "probes": list(selection.probes),
            "ell_min": selection.ell_min,
            "ell_max": selection.ell_max,
            "ndata": int(len(selection.indices)),
        },
    }
    payload.update(dict(extra or {}))
    return json.dumps(payload, indent=2, sort_keys=True)


def save_pickle(path: pathlib.Path | str, obj) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: pathlib.Path | str):
    with pathlib.Path(path).open("rb") as f:
        return pickle.load(f)
