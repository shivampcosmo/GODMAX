"""Stage-31 NumPyro HMC fit helpers for the xDESI GODMAX comparison."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import yaml
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import initialize_model

try:
    from . import godmax_multiprobe_theory_utils as gmt
except ImportError:  # pragma: no cover - used when running this file as a script.
    import godmax_multiprobe_theory_utils as gmt


DEFAULT_STAGE31_CONFIG = "param_files/xDESI/params_multiprobe_fast1024_hmc_stage31.yaml"
PARAMETER_COUNT_STAGE31 = 31
HALO_MASS_FLOOR_TOL = 5.0e-7
FIXED_ZERO_HOD_ARRAYS = (
    "gamma_a_fshmr_array",
    "beta_a_fshmr_array",
    "delta_a_fshmr_array",
)


def log_status(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    base_name: str
    target: str
    fiducial: float
    prior_kind: str = "uniform"
    prior_min: float = -math.inf
    prior_max: float = math.inf
    prior_mean: Optional[float] = None
    prior_sigma: Optional[float] = None
    array_key: Optional[str] = None
    array_index: Optional[int] = None


@dataclass(frozen=True)
class SpectrumSpec:
    name: str
    family: str
    theory_key: str
    fields: Tuple[str, str]
    pz_bin: Optional[int]
    window: jnp.ndarray
    transfer: jnp.ndarray
    scalar_factor: float
    ksz_amp: float
    source_band_count: int
    selected_band_indices: Tuple[int, ...]
    ell_band: Tuple[float, ...]


@dataclass(frozen=True)
class LikelihoodData:
    names: Tuple[str, ...]
    families: Tuple[str, ...]
    labels: Tuple[str, ...]
    theory_keys: Tuple[str, ...]
    ell_band: jnp.ndarray
    data_vector: jnp.ndarray
    covariance: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    spectrum_specs: Tuple[SpectrumSpec, ...]
    whitener: jnp.ndarray
    corr_eigenvalues: np.ndarray
    kept_modes: np.ndarray
    eigenvalue_threshold: float

    @property
    def rank(self) -> int:
        return int(np.sum(self.kept_modes))


@dataclass(frozen=True)
class FitContext:
    config: Mapping[str, object]
    stage_config: Mapping[str, object]
    prior_config: Mapping[str, object]
    parameter_specs: Tuple[ParameterSpec, ...]
    likelihood: LikelihoodData


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_prior_range(value: object, name: str) -> Tuple[float, float]:
    if isinstance(value, str):
        parts = value.split()
    else:
        parts = list(value) if isinstance(value, Sequence) else []
    if len(parts) != 2:
        raise ValueError(f"Prior for {name!r} must contain exactly two numbers.")
    lo, hi = float(parts[0]), float(parts[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
        raise ValueError(f"Invalid prior range for {name!r}: {value!r}.")
    return lo, hi


def parse_prior_gaussian(value: object, name: str) -> Tuple[float, float]:
    if isinstance(value, Mapping):
        if "mu" in value and "sigma" in value:
            mu, sigma = float(value["mu"]), float(value["sigma"])
        elif "mean" in value and "sigma" in value:
            mu, sigma = float(value["mean"]), float(value["sigma"])
        else:
            raise ValueError(f"Gaussian prior for {name!r} must contain mu/sigma.")
    elif isinstance(value, str):
        parts = value.split()
        if len(parts) != 2:
            raise ValueError(f"Gaussian prior for {name!r} must contain exactly two numbers.")
        mu, sigma = float(parts[0]), float(parts[1])
    else:
        parts = list(value) if isinstance(value, Sequence) else []
        if len(parts) != 2:
            raise ValueError(f"Gaussian prior for {name!r} must contain exactly two numbers.")
        mu, sigma = float(parts[0]), float(parts[1])
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"Invalid Gaussian prior for {name!r}: {value!r}.")
    return mu, sigma


def load_stage31_config(config_path: str | Path = DEFAULT_STAGE31_CONFIG) -> dict:
    root = gmt.repo_root()
    path = gmt.resolve_repo_path(config_path, root)
    raw = read_yaml(path)
    raw["config_path"] = path
    raw["comparison_config"] = gmt.resolve_repo_path(raw["comparison_config"], root)
    raw["prior_file"] = gmt.resolve_repo_path(raw["prior_file"], root)
    raw["output_dir"] = gmt.resolve_repo_path(raw["output_dir"], root)
    return raw


def load_materialized_comparison(stage_config: Mapping[str, object]) -> dict:
    cfg = gmt.load_comparison_config(stage_config["comparison_config"])
    if bool(stage_config.get("force_simple_1h2h", False)):
        cfg = enforce_simple_1h2h_transitions(cfg)
    cfg = gmt.materialize_nz_inputs(cfg)
    cfg = gmt.compute_desi_nbar_comoving(cfg)
    return cfg


def enforce_simple_1h2h_transitions(config: Mapping[str, object]) -> dict:
    """Force every configurable tracer pair onto plain P_1h + P_2h."""

    cfg = copy.deepcopy(dict(config))
    params = cfg.setdefault("params", {})
    analysis = params.setdefault("analysis", {})
    other = params.setdefault("other_params", {})
    transition_keys = (
        "tSZ_transition_model",
        "gg_transition_model",
        "galaxy_matter_transition_model",
        "gm_transition_model",
        "galaxy_electron_transition_model",
        "ge_transition_model",
    )
    alpha_keys = ("alpha_ky", "alpha_gy", "alpha_gg", "alpha_gm", "alpha_ge")
    for key in transition_keys:
        analysis[key] = "poweradd"
    for key in alpha_keys:
        other[key] = 1.0
    cfg.setdefault("metadata", {})["simple_1h2h_enforced"] = {
        "transition_model": "poweradd",
        "alpha": 1.0,
        "transition_keys": list(transition_keys),
        "alpha_keys": list(alpha_keys),
    }
    return cfg


def build_parameter_specs(config: Mapping[str, object], prior_config: Mapping[str, object]) -> Tuple[ParameterSpec, ...]:
    sim_params = config["params"]["sim_params"]
    other_params = config["params"].setdefault("other_params", {})
    prior_uniform = prior_config.get("prior_uniform", {})
    prior_gaussian = prior_config.get("prior_gaussian", {})
    vary = prior_config["vary"]

    specs: List[ParameterSpec] = []
    for name in vary["baryon_scalars"]:
        lo, hi = parse_prior_range(prior_uniform[name], name)
        fid = float(sim_params[name])
        specs.append(
            ParameterSpec(
                name=name,
                base_name=name,
                target="sim_scalar",
                fiducial=fid,
                prior_kind="uniform",
                prior_min=lo,
                prior_max=hi,
            )
        )

    for name in vary.get("other_scalars", []):
        if name in prior_gaussian:
            mu, sigma = parse_prior_gaussian(prior_gaussian[name], name)
            specs.append(
                ParameterSpec(
                    name=name,
                    base_name=name,
                    target="other_scalar",
                    fiducial=float(other_params.get(name, mu)),
                    prior_kind="normal",
                    prior_mean=mu,
                    prior_sigma=sigma,
                )
            )
        else:
            lo, hi = parse_prior_range(prior_uniform[name], name)
            specs.append(
                ParameterSpec(
                    name=name,
                    base_name=name,
                    target="other_scalar",
                    fiducial=float(other_params[name]),
                    prior_kind="uniform",
                    prior_min=lo,
                    prior_max=hi,
                )
            )

    for base_name in vary["hod_arrays"]:
        array_key = f"{base_name}_array"
        values = list(sim_params[array_key])
        lo, hi = parse_prior_range(prior_uniform[base_name], base_name)
        for pz_bin in vary["hod_indices"]:
            idx = int(pz_bin)
            specs.append(
                ParameterSpec(
                    name=f"{base_name}_pz{idx}",
                    base_name=base_name,
                    target="hod_array",
                    fiducial=float(values[idx]),
                    prior_kind="uniform",
                    prior_min=lo,
                    prior_max=hi,
                    array_key=array_key,
                    array_index=idx,
                )
            )

    for bin_index in vary.get("des_source_photoz_bins", []):
        idx = int(bin_index)
        name = f"Delta_z_bias_bin{idx}"
        mu, sigma = parse_prior_gaussian(prior_gaussian[name], name)
        values = list(other_params.get("Delta_z_bias_array", []))
        fid = float(values[idx - 1]) if len(values) >= idx else mu
        specs.append(
            ParameterSpec(
                name=name,
                base_name="Delta_z_bias",
                target="other_array",
                fiducial=fid,
                prior_kind="normal",
                prior_mean=mu,
                prior_sigma=sigma,
                array_key="Delta_z_bias_array",
                array_index=idx - 1,
            )
        )

    for bin_index in vary.get("des_shear_m_bins", []):
        idx = int(bin_index)
        name = f"mult_shear_bias_bin{idx}"
        mu, sigma = parse_prior_gaussian(prior_gaussian[name], name)
        values = list(other_params.get("mult_shear_bias_array", []))
        fid = float(values[idx - 1]) if len(values) >= idx else mu
        specs.append(
            ParameterSpec(
                name=name,
                base_name="mult_shear_bias",
                target="other_array",
                fiducial=fid,
                prior_kind="normal",
                prior_mean=mu,
                prior_sigma=sigma,
                array_key="mult_shear_bias_array",
                array_index=idx - 1,
            )
        )
    return tuple(specs)


def validate_parameter_specs(
    config: Mapping[str, object],
    specs: Sequence[ParameterSpec],
    prior_config: Optional[Mapping[str, object]] = None,
) -> dict:
    sim_params = config["params"]["sim_params"]
    errors: List[str] = []
    names = [spec.name for spec in specs]
    expected_count = None
    if prior_config is not None and prior_config.get("expected_parameter_count") is not None:
        expected_count = int(prior_config["expected_parameter_count"])
    elif prior_config is not None:
        expected_count = PARAMETER_COUNT_STAGE31
    if expected_count is not None and len(names) != expected_count:
        errors.append(f"Expected {expected_count} varied parameters, found {len(names)}.")
    if len(names) != len(set(names)):
        errors.append("Parameter names are not unique.")
    for spec in specs:
        if spec.prior_kind == "uniform" and not spec.prior_min <= spec.fiducial <= spec.prior_max:
            errors.append(
                f"Fiducial {spec.name}={spec.fiducial:g} is outside "
                f"[{spec.prior_min:g}, {spec.prior_max:g}]."
            )
        if spec.prior_kind == "normal":
            if spec.prior_mean is None or spec.prior_sigma is None or spec.prior_sigma <= 0.0:
                errors.append(f"Gaussian prior for {spec.name} is missing a finite positive sigma.")
        if spec.target == "other_array" and (spec.array_key is None or spec.array_index is None):
            errors.append(f"Malformed other_params array spec for {spec.name}.")
        if spec.target == "hod_array" and spec.array_index == 0:
            errors.append(f"{spec.name} varies HOD array entry 0, which must remain fixed.")
    varied_bases = {spec.base_name for spec in specs}
    if "cosmo" in varied_bases:
        errors.append("Cosmology must not be varied.")
    for zero_key in FIXED_ZERO_HOD_ARRAYS:
        values = np.asarray(sim_params.get(zero_key, []), dtype=np.float64)
        if values.size and np.all(values == 0.0) and zero_key[:-6] in varied_bases:
            errors.append(f"Fixed-zero HOD array {zero_key} is marked varied.")
    if errors:
        raise ValueError("Invalid stage-31 parameter registry:\n" + "\n".join(f"- {err}" for err in errors))
    return {
        "n_parameters": len(specs),
        "parameter_names": names,
        "hod_entry0_fixed": True,
        "fixed_zero_hod_arrays": list(FIXED_ZERO_HOD_ARRAYS),
    }


def validate_halo_mass_floor(stage_config: Mapping[str, object], config: Mapping[str, object]) -> dict:
    """Enforce the Backlight-compatible hard M200c lower integration limit."""

    required = stage_config.get("minimum_halo_log10_m200c_hmsun")
    halo_params = config.get("params", {}).get("halo_params", {})
    actual = halo_params.get("lg10_Mmin")
    if required is None:
        return {
            "required": None,
            "actual": float(actual) if actual is not None else None,
            "enforced": False,
        }
    if actual is None:
        raise ValueError(
            "Stage config requires minimum_halo_log10_m200c_hmsun="
            f"{float(required):.6f}, but merged halo_params.lg10_Mmin is missing."
        )
    actual_float = float(actual)
    required_float = float(required)
    if not np.isfinite(actual_float):
        raise ValueError(f"halo_params.lg10_Mmin is not finite: {actual!r}.")
    if abs(actual_float - required_float) > HALO_MASS_FLOOR_TOL:
        raise ValueError(
            "Backlight-compatible Stage-31 fits require the GODMAX halo mass grid "
            f"to start at log10(M200c/[Msun/h])={required_float:.6f}; "
            f"merged halo_params.lg10_Mmin={actual_float:.6f}."
        )
    return {
        "required": required_float,
        "actual": actual_float,
        "enforced": True,
        "mass_definition": "M200c",
        "mass_units": "Msun/h",
    }


def _value_for_bin(values: Mapping[int, float], bin_index: int) -> float:
    for key in (bin_index, str(bin_index), f"bin{bin_index}", f"s{bin_index}", f"tomo{bin_index}"):
        if key in values:
            return float(values[key])
    return 0.0


def _shear_m_factor(fields: Sequence[str], shear_m_bias: Mapping[int, float]) -> float:
    factor = 1.0
    for field in fields:
        if len(field) > 1 and field[0] == "s" and field[1:].isdigit():
            factor *= 1.0 + _value_for_bin(shear_m_bias, int(field[1:]))
    return float(factor)


def _shear_sign_factor(fields: Sequence[str], field_meta: Mapping[str, object]) -> float:
    factor = 1.0
    for field in fields:
        meta_outer = field_meta.get(field, {})
        if not isinstance(meta_outer, Mapping) or str(meta_outer.get("kind", "")) != "des_shear":
            continue
        meta = meta_outer.get("metadata", {})
        if not isinstance(meta, Mapping):
            meta = {}
        factor *= -float(meta.get("shear_e_to_kappa_sign", 1.0))
    return float(factor)


def _positive_float_mapping(value: object, label: str) -> Dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping, got {type(value).__name__}.")
    out: Dict[str, float] = {}
    for key, raw in value.items():
        if raw is None:
            continue
        ell_value = float(raw)
        if not np.isfinite(ell_value) or ell_value <= 0.0:
            raise ValueError(f"{label}[{key!r}] must be a positive finite ell value, got {raw!r}.")
        out[str(key)] = ell_value
    return out


def _ell_min_for_spectrum(
    name: str,
    family: str,
    theory_key: str,
    cut_config: Mapping[str, object],
) -> Optional[float]:
    if not cut_config:
        return None
    spectrum_map = _positive_float_mapping(cut_config.get("spectrum_ell_min"), "likelihood_cuts.spectrum_ell_min")
    for key in (name, theory_key):
        if key in spectrum_map:
            return spectrum_map[key]
    family_map = _positive_float_mapping(cut_config.get("family_ell_min"), "likelihood_cuts.family_ell_min")
    if family in family_map:
        return family_map[family]
    if cut_config.get("default_ell_min") is None:
        return None
    ell_min = float(cut_config["default_ell_min"])
    if not np.isfinite(ell_min) or ell_min <= 0.0:
        raise ValueError(f"likelihood_cuts.default_ell_min must be positive finite, got {ell_min!r}.")
    return ell_min


def _ell_max_for_spectrum(
    name: str,
    family: str,
    theory_key: str,
    cut_config: Mapping[str, object],
) -> Optional[float]:
    if not cut_config:
        return None
    spectrum_map = _positive_float_mapping(cut_config.get("spectrum_ell_max"), "likelihood_cuts.spectrum_ell_max")
    for key in (name, theory_key):
        if key in spectrum_map:
            return spectrum_map[key]
    family_map = _positive_float_mapping(cut_config.get("family_ell_max"), "likelihood_cuts.family_ell_max")
    if family in family_map:
        return family_map[family]
    if cut_config.get("default_ell_max") is None:
        return None
    ell_max = float(cut_config["default_ell_max"])
    if not np.isfinite(ell_max) or ell_max <= 0.0:
        raise ValueError(f"likelihood_cuts.default_ell_max must be positive finite, got {ell_max!r}.")
    return ell_max


def _selected_band_indices(
    ell: np.ndarray,
    ell_left: Optional[np.ndarray],
    ell_right: Optional[np.ndarray],
    ell_min: Optional[float],
    ell_max: Optional[float],
    selection: str,
    name: str,
) -> np.ndarray:
    n_band = int(np.asarray(ell).size)
    if ell_min is None and ell_max is None:
        return np.arange(n_band, dtype=int)
    selection = selection.lower()
    if selection in {"center", "centre", "effective"}:
        basis = np.asarray(ell, dtype=np.float64)
    elif selection in {"left", "lower", "ell_left"}:
        if ell_left is None:
            raise ValueError(f"likelihood_cuts.band_selection={selection!r} needs ell_left for {name}.")
        basis = np.asarray(ell_left, dtype=np.float64)
    elif selection in {"right", "upper", "ell_right"}:
        if ell_right is None:
            raise ValueError(f"likelihood_cuts.band_selection={selection!r} needs ell_right for {name}.")
        basis = np.asarray(ell_right, dtype=np.float64)
    else:
        raise ValueError(
            "likelihood_cuts.band_selection must be one of center, left, or right; "
            f"got {selection!r}."
        )
    if basis.shape != (n_band,):
        raise ValueError(f"Band selection basis for {name} has shape {basis.shape}, expected {(n_band,)}.")
    keep = np.ones(n_band, dtype=bool)
    if ell_min is not None:
        keep &= basis >= float(ell_min)
    if ell_max is not None:
        keep &= basis <= float(ell_max)
    selected = np.flatnonzero(keep).astype(int)
    if selected.size == 0:
        lo = "-inf" if ell_min is None else f"{ell_min:g}"
        hi = "inf" if ell_max is None else f"{ell_max:g}"
        raise ValueError(f"ell range [{lo}, {hi}] selects zero bandpowers for {name}.")
    return selected


def prepare_likelihood_data(config: Mapping[str, object], stage_config: Mapping[str, object]) -> LikelihoodData:
    gmt.ensure_godmax_import_paths(Path(config["repo_root"]))
    from multiprobe_namaster import (
        _load_default_transfers,
        ksz_velocity_amplitudes_from_field_metadata,
    )

    measurement_path = Path(config["paths"]["measurement_h5"])
    raw_wrapper = config["raw"].get("theory_to_data_vector", {})
    threshold = float(stage_config.get("covariance", {}).get("eigenvalue_threshold", 1.0e-8))
    cut_config = stage_config.get("likelihood_cuts", {})
    if cut_config is None:
        cut_config = {}
    if not isinstance(cut_config, Mapping):
        raise ValueError(f"likelihood_cuts must be a mapping, got {type(cut_config).__name__}.")
    band_selection = str(cut_config.get("band_selection", "center"))
    if bool(stage_config.get("sample_des_shear_m_bias_in_model", False)):
        shear_m_bias = {}
    else:
        shear_m_bias = config["metadata"]["shear_m_bias_means"]
    spectrum_specs: List[SpectrumSpec] = []

    with h5py.File(measurement_path, "r") as h5:
        measurement_config = json.loads(h5.attrs["config_json"])
        lmax = int(measurement_config["lmax"])
        field_meta = json.loads(h5["fields"].attrs["metadata_json"])
        transfers = _load_default_transfers(
            h5,
            lmax,
            include_pixel_windows=bool(raw_wrapper.get("include_default_pixel_windows", True)),
            include_act_beams=bool(raw_wrapper.get("include_default_act_beams", True)),
        )
        ksz_amps = ksz_velocity_amplitudes_from_field_metadata(
            field_meta,
            sigma_true_over_c=None,
            velocity_correlation=float(raw_wrapper.get("ksz_velocity_correlation", 0.3)),
        )
        full_ell = np.asarray(h5["joint/ell"][:], dtype=np.float64)
        full_data_vector = np.asarray(h5["joint/data_vector"][:], dtype=np.float64)
        full_covariance = np.asarray(h5["joint/cov"][:], dtype=np.float64)
        full_starts = np.asarray(h5["joint/slice_start"][:], dtype=int)
        full_stops = np.asarray(h5["joint/slice_stop"][:], dtype=int)
        default_ell_left = np.asarray(h5["ell_left"][:], dtype=np.float64) if "ell_left" in h5 else None
        default_ell_right = np.asarray(h5["ell_right"][:], dtype=np.float64) if "ell_right" in h5 else None
        names = tuple(x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["joint/spectrum_names"][:])
        labels: List[str] = []
        families: List[str] = []
        theory_keys: List[str] = []
        selected_global_indices: List[int] = []
        selected_ell_chunks: List[np.ndarray] = []
        starts: List[int] = []
        stops: List[int] = []
        cursor = 0
        for i_name, name in enumerate(names):
            group = h5[f"spectra/{name}"]
            fields = tuple(json.loads(group.attrs["fields"]))
            family = str(group.attrs["family"])
            theory_key = str(group.attrs["theory_key"])
            metadata = json.loads(group.attrs["metadata_json"])
            ell = np.asarray(group["ell"][:] if "ell" in group else full_ell, dtype=np.float64)
            ell_left = np.asarray(group["ell_left"][:], dtype=np.float64) if "ell_left" in group else default_ell_left
            ell_right = np.asarray(group["ell_right"][:], dtype=np.float64) if "ell_right" in group else default_ell_right
            source_start = int(full_starts[i_name])
            source_stop = int(full_stops[i_name])
            source_band_count = source_stop - source_start
            if ell.shape != (source_band_count,):
                raise ValueError(f"{name} ell has shape {ell.shape}, expected {(source_band_count,)}.")
            ell_min = _ell_min_for_spectrum(name, family, theory_key, cut_config)
            ell_max = _ell_max_for_spectrum(name, family, theory_key, cut_config)
            selected = _selected_band_indices(ell, ell_left, ell_right, ell_min, ell_max, band_selection, name)
            if np.any(selected < 0) or np.any(selected >= source_band_count):
                raise ValueError(f"Selected band indices for {name} are outside 0..{source_band_count - 1}.")
            scalar = _shear_m_factor(fields, shear_m_bias)
            if bool(raw_wrapper.get("theory_shear_e_is_positive_kappa", True)):
                scalar *= _shear_sign_factor(fields, field_meta)
            transfer = transfers.get(fields[0], np.ones(lmax + 1)) * transfers.get(fields[1], np.ones(lmax + 1))
            pz_bin = int(metadata["desi_pz"]) if family == "desi_pi_act_T" else None
            window = np.asarray(group["bandpower_window_selected"][:], dtype=np.float64)
            if window.shape[0] != source_band_count:
                raise ValueError(
                    f"{name} bandpower_window_selected has {window.shape[0]} rows, "
                    f"expected {source_band_count}."
                )
            starts.append(cursor)
            cursor += int(selected.size)
            stops.append(cursor)
            selected_global_indices.extend((source_start + selected).astype(int).tolist())
            selected_ell = ell[selected]
            selected_ell_chunks.append(selected_ell)
            spectrum_specs.append(
                SpectrumSpec(
                    name=name,
                    family=family,
                    theory_key=theory_key,
                    fields=(fields[0], fields[1]),
                    pz_bin=pz_bin,
                    window=jnp.asarray(window[selected], dtype=jnp.float64),
                    transfer=jnp.asarray(transfer, dtype=jnp.float64),
                    scalar_factor=float(scalar),
                    ksz_amp=float(ksz_amps[pz_bin]) if pz_bin is not None else 0.0,
                    source_band_count=source_band_count,
                    selected_band_indices=tuple(int(x) for x in selected),
                    ell_band=tuple(float(x) for x in selected_ell),
                )
            )
            labels.append(str(group.attrs.get("label", name)))
            families.append(family)
            theory_keys.append(theory_key)
        keep = np.asarray(selected_global_indices, dtype=int)
        ell_band = np.concatenate(selected_ell_chunks).astype(np.float64)
        data_vector = full_data_vector[keep]
        covariance = full_covariance[np.ix_(keep, keep)]

    sigma = np.sqrt(np.diag(covariance))
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("Measurement covariance has invalid diagonal entries.")
    corr = covariance / np.outer(sigma, sigma)
    corr = 0.5 * (corr + corr.T)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    kept = eigenvalues > threshold
    if not np.any(kept):
        raise ValueError(f"No covariance modes retained at eigenvalue threshold {threshold:g}.")
    whitener = (eigenvectors[:, kept].T / np.sqrt(eigenvalues[kept])[:, None]) / sigma[None, :]

    return LikelihoodData(
        names=names,
        families=tuple(families),
        labels=tuple(labels),
        theory_keys=tuple(theory_keys),
        ell_band=jnp.asarray(ell_band, dtype=jnp.float64),
        data_vector=jnp.asarray(data_vector, dtype=jnp.float64),
        covariance=np.asarray(covariance, dtype=np.float64),
        starts=np.asarray(starts, dtype=int),
        stops=np.asarray(stops, dtype=int),
        spectrum_specs=tuple(spectrum_specs),
        whitener=jnp.asarray(whitener, dtype=jnp.float64),
        corr_eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
        kept_modes=np.asarray(kept, dtype=bool),
        eigenvalue_threshold=threshold,
    )


def prepare_fit_context(config_path: str | Path = DEFAULT_STAGE31_CONFIG) -> FitContext:
    stage_config = load_stage31_config(config_path)
    prior_config = read_yaml(stage_config["prior_file"])
    config = load_materialized_comparison(stage_config)
    validate_halo_mass_floor(stage_config, config)
    specs = build_parameter_specs(config, prior_config)
    validate_parameter_specs(config, specs, prior_config)
    likelihood = prepare_likelihood_data(config, stage_config)
    return FitContext(
        config=config,
        stage_config=stage_config,
        prior_config=prior_config,
        parameter_specs=specs,
        likelihood=likelihood,
    )


def pack_fiducial_sample(specs: Sequence[ParameterSpec]) -> Dict[str, float]:
    return {spec.name: float(spec.fiducial) for spec in specs}


def pack_sample_from_params_file(context: FitContext, params_path: str | Path) -> Dict[str, float]:
    """Extract sampled Stage-31 parameter values from a saved params YAML."""

    raw = read_yaml(params_path)
    params = raw.get("params", raw)
    if "sim_params" not in params:
        raise KeyError(f"{params_path} does not contain a sim_params block.")
    sim_params = params["sim_params"]
    other_params = params.get("other_params", {})
    out: Dict[str, float] = {}
    for spec in context.parameter_specs:
        if spec.target == "sim_scalar":
            if spec.base_name not in sim_params:
                raise KeyError(f"{params_path} missing sim_params.{spec.base_name}.")
            value = sim_params[spec.base_name]
        elif spec.target == "other_scalar":
            if spec.base_name not in other_params:
                raise KeyError(f"{params_path} missing other_params.{spec.base_name}.")
            value = other_params[spec.base_name]
        elif spec.target == "hod_array":
            if spec.array_key is None or spec.array_index is None:
                raise ValueError(f"Malformed HOD spec: {spec}")
            if spec.array_key not in sim_params:
                raise KeyError(f"{params_path} missing sim_params.{spec.array_key}.")
            value = sim_params[spec.array_key][int(spec.array_index)]
        elif spec.target == "other_array":
            if spec.array_key is None or spec.array_index is None:
                raise ValueError(f"Malformed other_params array spec: {spec}")
            if spec.array_key not in other_params:
                raise KeyError(f"{params_path} missing other_params.{spec.array_key}.")
            value = other_params[spec.array_key][int(spec.array_index)]
        else:
            raise ValueError(f"Unknown parameter target {spec.target!r}.")
        value = float(value)
        if spec.prior_kind == "uniform" and not spec.prior_min <= value <= spec.prior_max:
            raise ValueError(
                f"Initial value for {spec.name}={value:g} is outside "
                f"[{spec.prior_min:g}, {spec.prior_max:g}]."
            )
        out[spec.name] = value
    return out


def pack_fiducial_vector(specs: Sequence[ParameterSpec]) -> jnp.ndarray:
    return jnp.asarray([spec.fiducial for spec in specs], dtype=jnp.float64)


def unpack_parameter_vector(specs: Sequence[ParameterSpec], vector: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    vector = jnp.asarray(vector, dtype=jnp.float64)
    return {spec.name: vector[i] for i, spec in enumerate(specs)}


def apply_sample_to_config(config: Mapping[str, object], specs: Sequence[ParameterSpec], sample_values: Mapping[str, object]) -> dict:
    out = copy.deepcopy(dict(config))
    sim_params = out["params"]["sim_params"]
    other_params = out["params"].setdefault("other_params", {})
    sampled_shear_m = False
    sampled_des_y3 = False
    for spec in specs:
        value = sample_values[spec.name]
        if spec.target == "sim_scalar":
            sim_params[spec.base_name] = value
        elif spec.target == "other_scalar":
            other_params[spec.base_name] = value
        elif spec.target == "hod_array":
            if spec.array_key is None or spec.array_index is None:
                raise ValueError(f"Malformed HOD spec: {spec}")
            arr = jnp.asarray(sim_params[spec.array_key], dtype=jnp.float64)
            sim_params[spec.array_key] = arr.at[int(spec.array_index)].set(value)
        elif spec.target == "other_array":
            if spec.array_key is None or spec.array_index is None:
                raise ValueError(f"Malformed other_params array spec: {spec}")
            arr = jnp.asarray(other_params.get(spec.array_key, []), dtype=jnp.float64)
            min_size = int(spec.array_index) + 1
            if arr.size < min_size:
                arr = jnp.pad(arr, (0, min_size - arr.size))
            other_params[spec.array_key] = arr.at[int(spec.array_index)].set(value)
            if spec.array_key == "mult_shear_bias_array":
                sampled_shear_m = True
                sampled_des_y3 = True
            elif spec.array_key == "Delta_z_bias_array":
                sampled_des_y3 = True
        else:
            raise ValueError(f"Unknown parameter target {spec.target!r}.")
    if sampled_des_y3:
        other_params["sampled_des_y3_nuisance"] = True
    if sampled_shear_m:
        other_params["sampled_des_shear_m_bias_in_model"] = True
        out.setdefault("metadata", {})["sampled_des_shear_m_bias_in_model"] = True
    return out


def _params_for_one_model(config: Mapping[str, object], *, is_cmb_lensing: bool) -> Tuple[dict, dict, dict, dict]:
    params = copy.deepcopy(config["params"])
    lmax = int(config["metadata"]["lmax"])
    params["halo_params"]["ell_array"] = jnp.arange(2, lmax + 1, dtype=jnp.float64)
    params["analysis"]["is_cmb_lensing"] = bool(is_cmb_lensing)
    params["analysis"]["symbolic_pk"] = False
    params["analysis"]["symbolic_hmf"] = False
    if is_cmb_lensing:
        z_source = params["analysis"]["nz_source_info_dict"]["z_array_source"]
        params["analysis"]["nz_source_info_dict"] = {
            "nbins": 1,
            "z_array_source": z_source,
            "nz0": np.ones(len(z_source), dtype=np.float64).tolist(),
        }
        params["other_params"]["Delta_z_bias_array"] = [0.0]
        params["other_params"]["mult_shear_bias_array"] = [0.0]
    return params["sim_params"], params["halo_params"], params["analysis"], params["other_params"]


def build_one_model_from_config(config: Mapping[str, object], *, is_cmb_lensing: bool):
    gmt.ensure_godmax_import_paths(Path(config["repo_root"]))
    from base_class import base_class
    from get_Cls import get_Cl
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles

    sim_params, halo_params, analysis, other_params = _params_for_one_model(config, is_cmb_lensing=is_cmb_lensing)
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = Profiles(sim_params, halo_params, analysis, other_params, base_class_obj=base)
    pkz = get_Pkz(sim_params, halo_params, analysis, other_params, Profiles_obj=profiles)
    return get_Cl(sim_params, halo_params, analysis, other_params, Pkz_obj=pkz)


def build_models_from_sample(context: FitContext, sample_values: Mapping[str, object]):
    sampled_config = apply_sample_to_config(context.config, context.parameter_specs, sample_values)
    return gmt.build_godmax_models(sampled_config)


def _trapz_jax(values: jnp.ndarray, x: jnp.ndarray, axis: int) -> jnp.ndarray:
    if hasattr(jnp, "trapezoid"):
        return jnp.trapezoid(values, x=x, axis=axis)
    return jnp.trapezoid(values, x=x, axis=axis)


def ne0_cm3_jax(cosmo_params: Mapping[str, float], helium_mass_fraction: float = 0.24) -> jnp.ndarray:
    from astropy import constants as const

    h = float(cosmo_params["H0"]) / 100.0
    rho_crit_0 = 1.878e-29 * h**2
    ne0 = rho_crit_0 * float(cosmo_params["Ob0"]) * (1.0 - helium_mass_fraction / 2.0) / const.m_p.to("g").value
    return jnp.asarray(ne0, dtype=jnp.float64)


def corrected_gal_tau_cls_zdependent_jax(cls_obj) -> jnp.ndarray:
    z = jnp.asarray(cls_obj.z_array_for_Cls, dtype=jnp.float64)
    chi = jnp.asarray(cls_obj.chi_array_for_Cls, dtype=jnp.float64)
    dchi_dz = jnp.asarray(cls_obj.dchi_dz_array_for_Cls, dtype=jnp.float64)
    wg = jnp.asarray(cls_obj.Wg_mat, dtype=jnp.float64)
    wtau = jnp.asarray(cls_obj.Wtau_array, dtype=jnp.float64)
    pge = jnp.asarray(cls_obj.cached_power_spectra[2, 4], dtype=jnp.float64)
    wtau_corrected = wtau * ne0_cm3_jax(cls_obj.cosmo_params) * (1.0 + z) ** 3
    prefac_tau = wtau_corrected / chi**2
    prefac_g = wg / (dchi_dz[None, :] * chi[None, :] ** 2)
    common = pge * prefac_tau[None, :] * chi[None, :] ** 2 * dchi_dz[None, :]
    integrand = common[:, None, :] * prefac_g[None, :, :]
    return _trapz_jax(integrand, z, axis=2)


def extract_theory_cls_jax_from_models(models: Mapping[str, object]) -> Dict[str, jnp.ndarray]:
    theory: Dict[str, jnp.ndarray] = {}
    cls_wl = models["wl"]
    for i in range(4):
        for j in range(i, 4):
            theory[f"des_shear_EE_tomo{i + 1}_tomo{j + 1}"] = cls_wl.Cl_kappa_kappa_tot_mat[:, i, j]
    for i in range(4):
        theory[f"act_y_des_shear_E_tomo{i + 1}"] = cls_wl.Cl_kappa_y_tot_mat[:, i]

    for pz_bin in range(1, 5):
        pz_wl = models["gal_wl_by_pz"][pz_bin]
        pz_cmb = models["gal_cmb_by_pz"][pz_bin]
        theory[f"desi_g_auto_pz{pz_bin}"] = pz_wl.Cl_gal_gal_tot_mat[:, 0, 0]
        theory[f"desi_g_act_y_pz{pz_bin}"] = pz_wl.Cl_gal_y_tot_mat[:, 0]
        theory[f"desi_g_act_kappa_pz{pz_bin}"] = pz_cmb.Cl_gal_kappa_tot_mat[:, 0, 0]
        for tomo in range(1, 5):
            theory[f"desi_g_des_shear_E_pz{pz_bin}_tomo{tomo}"] = pz_wl.Cl_gal_kappa_tot_mat[:, 0, tomo - 1]
        theory[f"desi_g_tau_pz{pz_bin}"] = corrected_gal_tau_cls_zdependent_jax(pz_wl)[:, 0]
    return theory


def extract_theory_cls_jax(cls_wl, cls_cmb=None) -> Dict[str, jnp.ndarray]:
    if isinstance(cls_wl, Mapping) and "gal_wl_by_pz" in cls_wl:
        return extract_theory_cls_jax_from_models(cls_wl)
    theory: Dict[str, jnp.ndarray] = {}
    for i in range(4):
        for j in range(i, 4):
            theory[f"des_shear_EE_tomo{i + 1}_tomo{j + 1}"] = cls_wl.Cl_kappa_kappa_tot_mat[:, i, j]
    for i in range(4):
        theory[f"act_y_des_shear_E_tomo{i + 1}"] = cls_wl.Cl_kappa_y_tot_mat[:, i]
        theory[f"desi_g_auto_pz{i + 1}"] = cls_wl.Cl_gal_gal_tot_mat[:, i, i]
        theory[f"desi_g_act_y_pz{i + 1}"] = cls_wl.Cl_gal_y_tot_mat[:, i]
        theory[f"desi_g_act_kappa_pz{i + 1}"] = cls_cmb.Cl_gal_kappa_tot_mat[:, i, 0]
    for i in range(4):
        for j in range(4):
            theory[f"desi_g_des_shear_E_pz{i + 1}_tomo{j + 1}"] = cls_wl.Cl_gal_kappa_tot_mat[:, i, j]
    tau = corrected_gal_tau_cls_zdependent_jax(cls_wl)
    for i in range(4):
        theory[f"desi_g_tau_pz{i + 1}"] = tau[:, i]
    return theory


def ell2_to_full_lmax(cl_ell2: jnp.ndarray) -> jnp.ndarray:
    cl_ell2 = jnp.asarray(cl_ell2, dtype=jnp.float64)
    return jnp.concatenate([cl_ell2[:1], cl_ell2[:1], cl_ell2])


def theory_data_vector_jax(likelihood: LikelihoodData, theory_cls: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    gmt.ensure_godmax_import_paths()
    from multiprobe_namaster import TCMB_UK

    out = []
    for spec in likelihood.spectrum_specs:
        if spec.family == "desi_pi_act_T":
            cl = -float(TCMB_UK) * float(spec.ksz_amp) * theory_cls[spec.theory_key]
        elif spec.name in theory_cls:
            cl = theory_cls[spec.name]
        else:
            cl = theory_cls[spec.theory_key]
        full = ell2_to_full_lmax(cl * float(spec.scalar_factor))
        out.append(spec.window @ (full[: spec.window.shape[1]] * spec.transfer[: spec.window.shape[1]]))
    return jnp.concatenate(out)


def evaluate_sample_theory_vector(context: FitContext, sample_values: Mapping[str, object]) -> jnp.ndarray:
    models = build_models_from_sample(context, sample_values)
    theory_cls = extract_theory_cls_jax_from_models(models)
    return theory_data_vector_jax(context.likelihood, theory_cls)


def whitened_chi2(likelihood: LikelihoodData, theory_vector: jnp.ndarray) -> jnp.ndarray:
    residual = likelihood.data_vector - theory_vector
    white = likelihood.whitener @ residual
    return jnp.sum(white**2)


def parameter_vector_chi2(context: FitContext, vector: jnp.ndarray) -> jnp.ndarray:
    sample_values = unpack_parameter_vector(context.parameter_specs, vector)
    theory_vector = evaluate_sample_theory_vector(context, sample_values)
    return whitened_chi2(context.likelihood, theory_vector)


def _finite_array_summary(arr: object, prefix: str = "") -> dict:
    arr_np = np.asarray(arr)
    flat = arr_np.reshape(-1)
    finite = np.isfinite(flat)
    out = {
        f"{prefix}size": int(flat.size),
        f"{prefix}finite": bool(np.all(finite)),
        f"{prefix}nan_count": int(np.sum(np.isnan(flat))),
        f"{prefix}inf_count": int(np.sum(np.isinf(flat))),
    }
    if np.any(finite):
        out[f"{prefix}finite_abs_max"] = float(np.max(np.abs(flat[finite])))
    return out


def physical_gradient_diagnostics(context: FitContext, vector: Optional[jnp.ndarray] = None) -> dict:
    x0 = pack_fiducial_vector(context.parameter_specs) if vector is None else jnp.asarray(vector, dtype=jnp.float64)
    out = {
        "mode": "physical_parameter_vector",
        "n_parameters": len(context.parameter_specs),
        "parameter_names": [spec.name for spec in context.parameter_specs],
    }
    try:
        value = parameter_vector_chi2(context, x0)
        out["value_only"] = float(np.asarray(value))
        out["value_only_finite"] = bool(np.isfinite(np.asarray(value)))
    except Exception as exc:  # pragma: no cover - diagnostic path.
        out["value_only_error_type"] = type(exc).__name__
        out["value_only_error"] = str(exc)
        return out

    try:
        value, grad = jax.value_and_grad(lambda vec: parameter_vector_chi2(context, vec))(x0)
        grad_np = np.asarray(grad)
        out["value_and_grad_ok"] = True
        out["value_and_grad_value"] = float(np.asarray(value))
        out.update(_finite_array_summary(grad_np, prefix="grad_"))
        bad = np.where(~np.isfinite(grad_np))[0]
        out["bad_gradients"] = [
            {
                "i": int(i),
                "name": context.parameter_specs[int(i)].name,
                "grad": float(grad_np[int(i)]) if np.isfinite(grad_np[int(i)]) else str(grad_np[int(i)]),
            }
            for i in bad
        ]
    except Exception as exc:  # pragma: no cover - diagnostic path.
        out["value_and_grad_ok"] = False
        out["value_and_grad_error_type"] = type(exc).__name__
        out["value_and_grad_error"] = str(exc)
    return out


def physical_jvp_diagnostics(
    context: FitContext,
    parameter_indices: Optional[Sequence[int]] = None,
    vector: Optional[jnp.ndarray] = None,
) -> List[dict]:
    x0 = pack_fiducial_vector(context.parameter_specs) if vector is None else jnp.asarray(vector, dtype=jnp.float64)
    indices = list(range(len(context.parameter_specs))) if parameter_indices is None else [int(i) for i in parameter_indices]
    rows = []
    for i in indices:
        spec = context.parameter_specs[i]
        tangent = jnp.zeros_like(x0).at[i].set(1.0)
        t0 = time.time()
        row = {"i": i, "name": spec.name}
        try:
            value, deriv = jax.jvp(lambda vec: parameter_vector_chi2(context, vec), (x0,), (tangent,))
            value_np = np.asarray(value)
            deriv_np = np.asarray(deriv)
            row.update(
                ok=True,
                chi2=float(value_np),
                chi2_finite=bool(np.isfinite(value_np)),
                dchi2_dparam=float(deriv_np) if np.isfinite(deriv_np) else str(deriv_np),
                deriv_finite=bool(np.isfinite(deriv_np)),
                seconds=time.time() - t0,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path.
            row.update(ok=False, error_type=type(exc).__name__, error=str(exc), seconds=time.time() - t0)
        rows.append(row)
    return rows


def finite_difference_diagnostics(
    context: FitContext,
    parameter_indices: Optional[Sequence[int]] = None,
    vector: Optional[jnp.ndarray] = None,
    rel_step: float = 1.0e-4,
    abs_step: float = 1.0e-5,
) -> List[dict]:
    x0 = np.asarray(pack_fiducial_vector(context.parameter_specs) if vector is None else vector, dtype=np.float64)
    indices = list(range(len(context.parameter_specs))) if parameter_indices is None else [int(i) for i in parameter_indices]
    rows = []
    for i in indices:
        spec = context.parameter_specs[i]
        step = max(float(abs_step), float(rel_step) * max(1.0, abs(float(x0[i]))))
        lower_room = float(x0[i] - spec.prior_min)
        upper_room = float(spec.prior_max - x0[i])
        if lower_room <= 0.0 or upper_room <= 0.0:
            rows.append({"i": i, "name": spec.name, "ok": False, "error": "fiducial outside prior"})
            continue
        step = min(step, 0.25 * lower_room, 0.25 * upper_room)
        row = {"i": i, "name": spec.name, "step": step}
        t0 = time.time()
        try:
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += step
            x_minus[i] -= step
            f_plus = float(np.asarray(parameter_vector_chi2(context, jnp.asarray(x_plus, dtype=jnp.float64))))
            f_minus = float(np.asarray(parameter_vector_chi2(context, jnp.asarray(x_minus, dtype=jnp.float64))))
            deriv = (f_plus - f_minus) / (2.0 * step)
            row.update(
                ok=True,
                f_plus=f_plus,
                f_minus=f_minus,
                fd_dchi2_dparam=deriv,
                finite=bool(np.isfinite(f_plus) and np.isfinite(f_minus) and np.isfinite(deriv)),
                seconds=time.time() - t0,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path.
            row.update(ok=False, error_type=type(exc).__name__, error=str(exc), seconds=time.time() - t0)
        rows.append(row)
    return rows


def numpyro_model(context: FitContext) -> None:
    sample_values = {}
    for spec in context.parameter_specs:
        if spec.prior_kind == "normal":
            if spec.prior_mean is None or spec.prior_sigma is None:
                raise ValueError(f"Gaussian prior for {spec.name} is missing mean/sigma.")
            prior = dist.Normal(float(spec.prior_mean), float(spec.prior_sigma))
        elif spec.prior_kind == "uniform":
            prior = dist.Uniform(float(spec.prior_min), float(spec.prior_max))
        else:
            raise ValueError(f"Unknown prior kind {spec.prior_kind!r} for {spec.name}.")
        sample_values[spec.name] = numpyro.sample(spec.name, prior)
    theory_vector = evaluate_sample_theory_vector(context, sample_values)
    chi2 = whitened_chi2(context.likelihood, theory_vector)
    numpyro.deterministic("chi2", chi2)
    numpyro.factor("xdesi_loglike", -0.5 * chi2)


def static_summary(context: FitContext) -> dict:
    validation = validate_parameter_specs(context.config, context.parameter_specs, context.prior_config)
    halo_mass_floor = validate_halo_mass_floor(context.stage_config, context.config)
    likelihood = context.likelihood
    analysis = context.config["params"].get("analysis", {})
    other = context.config["params"].get("other_params", {})
    return {
        "stage": context.stage_config["stage"],
        "n_parameters": validation["n_parameters"],
        "n_spectra": len(likelihood.names),
        "data_vector_size": int(likelihood.data_vector.size),
        "covariance_rank": likelihood.rank,
        "covariance_size": int(likelihood.data_vector.size),
        "dropped_covariance_modes": int(likelihood.data_vector.size - likelihood.rank),
        "corr_eigenvalue_threshold": likelihood.eigenvalue_threshold,
        "min_corr_eigenvalue": float(np.min(likelihood.corr_eigenvalues)),
        "max_corr_eigenvalue": float(np.max(likelihood.corr_eigenvalues)),
        "parameter_names": validation["parameter_names"],
        "halo_lg10_Mmin": halo_mass_floor["actual"],
        "minimum_halo_log10_m200c_hmsun": halo_mass_floor["required"],
        "halo_mass_floor_enforced": halo_mass_floor["enforced"],
        "desi_lens_redshift_kind": context.config.get("metadata", {}).get("lens_redshift_kind", ""),
        "desi_lens_nz_provenance": context.config.get("metadata", {}).get("lens_nz_provenance", {}),
        "galaxy_modeling": "single_godmax_object_per_photometric_pz_bin",
        "likelihood_cuts": context.stage_config.get("likelihood_cuts", {}),
        "bands_per_spectrum": {
            spec.name: {
                "n_selected": int(len(spec.selected_band_indices)),
                "source_band_count": int(spec.source_band_count),
                "selected_band_indices": list(spec.selected_band_indices),
                "ell_min": float(np.min(spec.ell_band)),
                "ell_max": float(np.max(spec.ell_band)),
            }
            for spec in likelihood.spectrum_specs
        },
        "simple_1h2h_enforced": context.config.get("metadata", {}).get("simple_1h2h_enforced", None),
        "transition_models": {
            "matter_transition_model": analysis.get(
                "matter_transition_model",
                analysis.get("mm_transition_model", "response"),
            ),
            "tSZ_transition_model": analysis.get("tSZ_transition_model", "poweradd"),
            "gg_transition_model": analysis.get("gg_transition_model", "poweradd"),
            "galaxy_matter_transition_model": analysis.get(
                "galaxy_matter_transition_model",
                analysis.get("gm_transition_model", "poweradd"),
            ),
            "galaxy_electron_transition_model": analysis.get(
                "galaxy_electron_transition_model",
                analysis.get("ge_transition_model", "poweradd"),
            ),
        },
        "transition_alphas": {
            "alpha_ky": float(other.get("alpha_ky", 1.0)),
            "alpha_gy": float(other.get("alpha_gy", 1.0)),
            "alpha_gg": float(other.get("alpha_gg", 1.0)),
            "alpha_gm": float(other.get("alpha_gm", 1.0)),
            "alpha_ge": float(other.get("alpha_ge", 1.0)),
        },
    }


def compare_fiducial_windowing(context: FitContext) -> dict:
    fid = pack_fiducial_sample(context.parameter_specs)
    models = build_models_from_sample(context, fid)
    theory_cls_jax = extract_theory_cls_jax_from_models(models)
    jax_vector = np.asarray(theory_data_vector_jax(context.likelihood, theory_cls_jax))
    theory_cls_np = gmt.extract_theory_cls_from_models(models, context.config["metadata"])
    ell_theory = gmt.model_ell_array(models)
    wrapper_vector, wrapper_names = gmt.theory_data_vector(
        context.config,
        theory_cls_np,
        ell_theory,
    )
    if wrapper_vector.shape != jax_vector.shape:
        measurement = gmt.load_measurement_data(context.config["paths"]["measurement_h5"])
        chunks = []
        for spec in context.likelihood.spectrum_specs:
            index = measurement.names.index(spec.name)
            start = int(measurement.starts[index])
            selected = np.asarray(spec.selected_band_indices, dtype=int)
            chunks.append(wrapper_vector[start + selected])
        wrapper_vector = np.concatenate(chunks)
    delta = jax_vector - wrapper_vector
    return {
        "jax_vector": jax_vector,
        "wrapper_vector": wrapper_vector,
        "wrapper_names": wrapper_names,
        "max_abs_delta": float(np.max(np.abs(delta))),
        "max_rel_delta": float(np.max(np.abs(delta) / np.maximum(np.abs(wrapper_vector), 1.0e-300))),
        "allclose": bool(np.allclose(jax_vector, wrapper_vector, rtol=1.0e-10, atol=1.0e-18)),
        "ksz_median_raw_by_pz": [
            float(np.nanmedian(jax_vector[int(context.likelihood.starts[i]) : int(context.likelihood.stops[i])]))
            for i, name in enumerate(context.likelihood.names)
            if name.startswith("desi_pi_act_T")
        ],
    }


def sampler_settings(stage_config: Mapping[str, object], *, smoke: bool, overrides: Optional[Mapping[str, object]] = None) -> dict:
    settings = dict(stage_config["smoke_sampler" if smoke else "sampler"])
    for key, value in (overrides or {}).items():
        if value is not None:
            settings[key] = value
    return settings


def configure_numpyro_platform(platform: Optional[str]) -> dict:
    if platform:
        numpyro.set_platform(platform)
    devices = jax.devices()
    device_platforms = sorted({str(device.platform) for device in devices})
    backend = jax.default_backend()
    if platform == "gpu" and backend not in {"gpu", "cuda"} and "gpu" not in device_platforms:
        raise RuntimeError(
            "Requested GPU execution, but JAX is not using a GPU backend. "
            f"jax.default_backend()={backend!r}, devices={[str(device) for device in devices]!r}. "
            "Restart the notebook kernel and run the environment cell before importing JAX, "
            "or launch the CLI with JAX_PLATFORMS=cuda set before Python starts."
        )
    return {
        "requested_platform": platform or "default",
        "jax_default_backend": backend,
        "jax_device_count": len(devices),
        "jax_device_platforms": device_platforms,
        "jax_devices": [str(device) for device in devices],
    }


def gpu_sanity_check(matrix_size: int = 4096, *, require_gpu: bool = True) -> dict:
    """Run one synchronized JAX operation and report where it executed."""

    devices = jax.devices()
    backend = jax.default_backend()
    device_platforms = sorted({str(device.platform) for device in devices})
    if require_gpu and backend not in {"gpu", "cuda"} and "gpu" not in device_platforms:
        raise RuntimeError(
            "GPU sanity check requested, but JAX is not on a GPU backend. "
            f"jax.default_backend()={backend!r}, devices={[str(device) for device in devices]!r}."
        )

    n = int(matrix_size)
    t0 = time.time()
    x = jnp.ones((n, n), dtype=jnp.float64)
    y = (x @ x).sum()
    y.block_until_ready()
    elapsed = time.time() - t0
    return {
        "jax_default_backend": backend,
        "jax_device_platforms": device_platforms,
        "jax_devices": [str(device) for device in devices],
        "matrix_size": n,
        "result": float(np.asarray(y)),
        "seconds": elapsed,
    }


def initialization_diagnostics(
    context: FitContext,
    settings: Mapping[str, object],
    init_values: Optional[Mapping[str, float]] = None,
) -> dict:
    provided_init = init_values is not None
    init_values = dict(init_values) if provided_init else pack_fiducial_sample(context.parameter_specs)
    forward_mode = bool(settings.get("forward_mode_differentiation", False))
    out = {
        "seed": int(settings.get("seed", 42)),
        "forward_mode_differentiation": forward_mode,
        "initial_point": "provided" if provided_init else "fiducial",
        "checks": [],
    }
    for validate_grad in (False, True):
        check = {"validate_grad": validate_grad}
        try:
            info = initialize_model(
                jax.random.PRNGKey(int(settings.get("seed", 42))),
                lambda: numpyro_model(context),
                init_strategy=init_to_value(values=init_values),
                forward_mode_differentiation=forward_mode,
                validate_grad=validate_grad,
            )
            potential_energy = np.asarray(info.param_info.potential_energy)
            check["ok"] = True
            check["potential_energy"] = float(potential_energy)
            check["potential_energy_finite"] = bool(np.all(np.isfinite(potential_energy)))
            if info.param_info.z_grad is not None:
                grad_flat, _ = ravel_pytree(info.param_info.z_grad)
                grad_np = np.asarray(grad_flat)
                check["gradient_size"] = int(grad_np.size)
                check["gradient_finite"] = bool(np.all(np.isfinite(grad_np)))
                check["gradient_nan_count"] = int(np.sum(np.isnan(grad_np)))
                check["gradient_inf_count"] = int(np.sum(np.isinf(grad_np)))
                finite = grad_np[np.isfinite(grad_np)]
                if finite.size:
                    check["gradient_abs_max"] = float(np.max(np.abs(finite)))
        except Exception as exc:  # pragma: no cover - diagnostic path.
            check["ok"] = False
            check["error_type"] = type(exc).__name__
            check["error"] = str(exc)
        out["checks"].append(check)
    return out


def run_hmc(
    context: FitContext,
    *,
    smoke: bool = False,
    overrides: Optional[Mapping[str, object]] = None,
    init_values: Optional[Mapping[str, float]] = None,
    checkpoint_samples_every: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
) -> MCMC:
    settings = sampler_settings(context.stage_config, smoke=smoke, overrides=overrides)
    checkpoint_every = int(checkpoint_samples_every or 0)
    if checkpoint_every > 0 and not smoke:
        return run_hmc_checkpointed(
            context,
            settings=settings,
            init_values=init_values,
            checkpoint_samples_every=checkpoint_every,
            output_dir=output_dir,
        )
    return run_hmc_single(context, settings=settings, init_values=init_values)


def _build_nuts_kernel(
    context: FitContext,
    settings: Mapping[str, object],
    init_values: Mapping[str, float],
) -> NUTS:
    kwargs = {
        "init_strategy": init_to_value(values=init_values),
        "dense_mass": bool(settings.get("dense_mass", True)),
        "max_tree_depth": int(settings.get("max_tree_depth", 8)),
        "forward_mode_differentiation": bool(settings.get("forward_mode_differentiation", False)),
    }
    if settings.get("target_accept_prob") is not None:
        kwargs["target_accept_prob"] = float(settings["target_accept_prob"])
    return NUTS(lambda: numpyro_model(context), **kwargs)


def _target_accept_label(settings: Mapping[str, object]) -> str:
    value = settings.get("target_accept_prob")
    return "numpyro_default" if value is None else f"{float(value)}"


def _build_mcmc(
    kernel: NUTS,
    settings: Mapping[str, object],
    *,
    num_warmup: int,
    num_samples: int,
) -> MCMC:
    return MCMC(
        kernel,
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(settings["num_chains"]),
        chain_method=str(settings.get("chain_method", "vectorized")),
        progress_bar=bool(settings.get("progress_bar", True)),
        jit_model_args=bool(settings.get("jit_model_args", True)),
    )


def run_hmc_single(
    context: FitContext,
    *,
    settings: Mapping[str, object],
    init_values: Optional[Mapping[str, float]] = None,
) -> MCMC:
    num_chains = int(settings["num_chains"])
    numpyro.set_host_device_count(max(1, num_chains))
    init_values = dict(init_values) if init_values is not None else pack_fiducial_sample(context.parameter_specs)
    log_status(
        "[hmc] configuring NUTS "
        f"num_chains={num_chains} chain_method={settings.get('chain_method', 'vectorized')} "
        f"num_warmup={int(settings['num_warmup'])} num_samples={int(settings['num_samples'])} "
        f"max_tree_depth={int(settings.get('max_tree_depth', 8))} "
        f"target_accept_prob={_target_accept_label(settings)} "
        f"dense_mass={bool(settings.get('dense_mass', True))} "
        f"progress_bar={bool(settings.get('progress_bar', True))}"
    )
    kernel = _build_nuts_kernel(context, settings, init_values)
    mcmc = _build_mcmc(
        kernel,
        settings,
        num_warmup=int(settings["num_warmup"]),
        num_samples=int(settings["num_samples"]),
    )
    log_status("[hmc] mcmc.run begin")
    mcmc.run(
        jax.random.PRNGKey(int(settings.get("seed", 42))),
        extra_fields=("potential_energy", "diverging", "accept_prob", "num_steps"),
    )
    log_status("[hmc] mcmc.run done")
    return mcmc


class _ArrayMCMCResult:
    def __init__(self, samples: Mapping[str, np.ndarray], extra_fields: Mapping[str, np.ndarray]):
        self._samples = {str(key): np.asarray(value) for key, value in samples.items()}
        self._extra_fields = {str(key): np.asarray(value) for key, value in extra_fields.items()}

    def get_samples(self, group_by_chain: bool = False) -> Dict[str, np.ndarray]:
        if group_by_chain:
            raise NotImplementedError("Checkpointed array result stores flattened chains only.")
        return dict(self._samples)

    def get_extra_fields(self, group_by_chain: bool = False) -> Dict[str, np.ndarray]:
        if group_by_chain:
            raise NotImplementedError("Checkpointed array result stores flattened chains only.")
        return dict(self._extra_fields)


def _flatten_extra_fields(extra_fields: Mapping[str, object]) -> Dict[str, np.ndarray]:
    out = {}
    for key, value in extra_fields.items():
        arr = np.asarray(value)
        out[key] = arr.reshape((-1,) + arr.shape[2:]) if arr.ndim >= 2 else arr.reshape(-1)
    return out


def _concat_chain_dict(chunks: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not chunks:
        return {}
    keys = sorted(chunks[0])
    return {key: np.concatenate([np.asarray(chunk[key]) for chunk in chunks], axis=0) for key in keys}


def _save_npz_atomic(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(tmp_path, path)


def _write_checkpoint_outputs(
    context: FitContext,
    *,
    samples: Mapping[str, np.ndarray],
    extra: Mapping[str, np.ndarray],
    output_dir: Path,
    suffix: str,
    chunk_index: int,
    draws_per_worker: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if "chi2" in samples:
        chi2 = np.asarray(samples["chi2"], dtype=np.float64)
        best_idx = int(np.nanargmin(chi2))
        best_chi2 = float(chi2[best_idx])
        best_sample = {
            spec.name: float(np.asarray(samples[spec.name])[best_idx])
            for spec in context.parameter_specs
            if spec.name in samples
        }
    else:
        best_idx = -1
        best_chi2 = float("nan")
        best_sample = {}

    payload = {f"sample__{key}": np.asarray(value) for key, value in samples.items()}
    payload.update({f"extra__{key}": np.asarray(value) for key, value in extra.items()})
    payload["parameter_names"] = np.asarray([spec.name for spec in context.parameter_specs])
    payload["metadata_json"] = np.asarray(
        json.dumps(
            {
                "checkpoint": True,
                "chunk_index": int(chunk_index),
                "draws_per_worker": int(draws_per_worker),
                "n_flat_samples": int(np.asarray(next(iter(samples.values()))).shape[0]) if samples else 0,
                "best_sample_index": best_idx,
                "best_whitened_chi2": best_chi2,
                "static_summary": static_summary(context),
                "parameter_specs": parameter_specs_jsonable(context.parameter_specs),
            }
        )
    )
    checkpoint_path = output_dir / f"chain_{suffix}_checkpoint_{draws_per_worker:06d}.npz"
    latest_path = output_dir / f"chain_{suffix}_checkpoint_latest.npz"
    _save_npz_atomic(checkpoint_path, **payload)
    _save_npz_atomic(latest_path, **payload)

    summary_path = output_dir / f"checkpoint_summary_{suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            gmt.to_jsonable(
                {
                    "checkpoint_path": checkpoint_path,
                    "latest_path": latest_path,
                    "chunk_index": int(chunk_index),
                    "draws_per_worker": int(draws_per_worker),
                    "n_flat_samples": int(np.asarray(next(iter(samples.values()))).shape[0]) if samples else 0,
                    "best_sample_index": best_idx,
                    "best_whitened_chi2": best_chi2,
                    "best_sample": best_sample,
                }
            ),
            handle,
            indent=2,
        )

    if best_sample:
        best_config = apply_sample_to_config(context.config, context.parameter_specs, best_sample)
        best_params_path = output_dir / f"bestfit_params_{suffix}_checkpoint_latest.yaml"
        with open(best_params_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(gmt.to_jsonable(best_config["params"]), handle, sort_keys=False)
    log_status(
        f"[hmc] checkpoint chunk={chunk_index} draws_per_worker={draws_per_worker} "
        f"best_chi2={best_chi2:.8e} path={latest_path}"
    )


def run_hmc_checkpointed(
    context: FitContext,
    *,
    settings: Mapping[str, object],
    init_values: Optional[Mapping[str, float]],
    checkpoint_samples_every: int,
    output_dir: Optional[str | Path] = None,
) -> _ArrayMCMCResult:
    total_samples = int(settings["num_samples"])
    chunk_size = int(checkpoint_samples_every)
    if chunk_size <= 0:
        raise ValueError("checkpoint_samples_every must be positive.")
    num_chains = int(settings["num_chains"])
    numpyro.set_host_device_count(max(1, num_chains))
    init_values = dict(init_values) if init_values is not None else pack_fiducial_sample(context.parameter_specs)
    output = Path(output_dir) if output_dir is not None else Path(context.stage_config["output_dir"])
    suffix = "stage31"
    log_status(
        "[hmc] configuring checkpointed NUTS "
        f"num_chains={num_chains} chain_method={settings.get('chain_method', 'vectorized')} "
        f"num_warmup={int(settings['num_warmup'])} num_samples={total_samples} "
        f"checkpoint_samples_every={chunk_size} "
        f"max_tree_depth={int(settings.get('max_tree_depth', 8))} "
        f"target_accept_prob={_target_accept_label(settings)} "
        f"dense_mass={bool(settings.get('dense_mass', True))} "
        f"progress_bar={bool(settings.get('progress_bar', True))}"
    )
    kernel = _build_nuts_kernel(context, settings, init_values)
    mcmc = _build_mcmc(
        kernel,
        settings,
        num_warmup=int(settings["num_warmup"]),
        num_samples=min(chunk_size, total_samples),
    )
    rng_keys = jax.random.split(
        jax.random.PRNGKey(int(settings.get("seed", 42))),
        int(math.ceil(total_samples / chunk_size)) + 1,
    )
    extra_fields = ("potential_energy", "diverging", "accept_prob", "num_steps")
    log_status("[hmc] checkpointed warmup begin")
    mcmc.warmup(rng_keys[0], extra_fields=extra_fields)
    log_status("[hmc] checkpointed warmup done")

    sample_chunks: List[Dict[str, np.ndarray]] = []
    extra_chunks: List[Dict[str, np.ndarray]] = []
    draws_done = 0
    chunk_index = 0
    while draws_done < total_samples:
        chunk_index += 1
        current = min(chunk_size, total_samples - draws_done)
        mcmc.num_samples = int(current)
        log_status(
            f"[hmc] checkpointed sample chunk {chunk_index} begin "
            f"draws={draws_done}:{draws_done + current}"
        )
        mcmc.run(rng_keys[chunk_index], extra_fields=extra_fields)
        sample_chunks.append({key: np.asarray(value) for key, value in mcmc.get_samples(group_by_chain=False).items()})
        extra_chunks.append(_flatten_extra_fields(mcmc.get_extra_fields(group_by_chain=False)))
        draws_done += current
        samples = _concat_chain_dict(sample_chunks)
        extra = _concat_chain_dict(extra_chunks)
        _write_checkpoint_outputs(
            context,
            samples=samples,
            extra=extra,
            output_dir=output,
            suffix=suffix,
            chunk_index=chunk_index,
            draws_per_worker=draws_done,
        )
        mcmc._warmup_state = mcmc._last_state
    log_status("[hmc] checkpointed sampling done")
    return _ArrayMCMCResult(_concat_chain_dict(sample_chunks), _concat_chain_dict(extra_chunks))


def best_sample_from_mcmc(mcmc: MCMC, specs: Sequence[ParameterSpec]) -> Tuple[Dict[str, float], int, float]:
    samples = {key: np.asarray(value) for key, value in mcmc.get_samples(group_by_chain=False).items()}
    if "chi2" not in samples:
        raise KeyError("MCMC samples do not contain deterministic 'chi2'.")
    chi2 = np.asarray(samples["chi2"], dtype=np.float64)
    idx = int(np.nanargmin(chi2))
    best = {spec.name: float(samples[spec.name][idx]) for spec in specs}
    return best, idx, float(chi2[idx])


def measurement_from_likelihood(context: FitContext, likelihood: LikelihoodData) -> gmt.MeasurementData:
    source = gmt.load_measurement_data(context.config["paths"]["measurement_h5"])
    names = list(likelihood.names)
    ell_left = None
    ell_right = None
    if source.ell_left is not None and source.ell_right is not None:
        left_chunks = []
        right_chunks = []
        for spec in likelihood.spectrum_specs:
            selected = np.asarray(spec.selected_band_indices, dtype=int)
            left_chunks.append(np.asarray(source.ell_left, dtype=np.float64)[selected])
            right_chunks.append(np.asarray(source.ell_right, dtype=np.float64)[selected])
        ell_left = np.concatenate(left_chunks) if left_chunks else None
        ell_right = np.concatenate(right_chunks) if right_chunks else None
    return gmt.MeasurementData(
        path=source.path,
        names=names,
        ell=np.asarray(likelihood.ell_band),
        data_vector=np.asarray(likelihood.data_vector),
        covariance=np.asarray(likelihood.covariance),
        starts=np.asarray(likelihood.starts, dtype=int),
        stops=np.asarray(likelihood.stops, dtype=int),
        families={name: family for name, family in zip(names, likelihood.families)},
        labels={name: label for name, label in zip(names, likelihood.labels)},
        theory_keys={name: theory_key for name, theory_key in zip(names, likelihood.theory_keys)},
        ell_left=ell_left,
        ell_right=ell_right,
    )


def measurement_for_plots(context: FitContext) -> gmt.MeasurementData:
    return measurement_from_likelihood(context, context.likelihood)


def full_likelihood_for_plots(context: FitContext) -> LikelihoodData:
    stage_config = copy.deepcopy(dict(context.stage_config))
    stage_config["likelihood_cuts"] = {}
    return prepare_likelihood_data(context.config, stage_config)


def likelihood_active_band_indices(context: FitContext) -> Dict[str, Tuple[int, ...]]:
    return {spec.name: tuple(int(i) for i in spec.selected_band_indices) for spec in context.likelihood.spectrum_specs}


def parameter_specs_jsonable(specs: Sequence[ParameterSpec]) -> List[dict]:
    return [
        {
            "name": spec.name,
            "base_name": spec.base_name,
            "target": spec.target,
            "prior_kind": spec.prior_kind,
            "prior_min": spec.prior_min if np.isfinite(spec.prior_min) else None,
            "prior_max": spec.prior_max if np.isfinite(spec.prior_max) else None,
            "prior_mean": spec.prior_mean,
            "prior_sigma": spec.prior_sigma,
            "fiducial": spec.fiducial,
            "array_key": spec.array_key,
            "array_index": spec.array_index,
        }
        for spec in specs
    ]


def save_fit_outputs(
    context: FitContext,
    mcmc: MCMC,
    *,
    smoke: bool = False,
    output_dir: Optional[str | Path] = None,
) -> dict:
    log_status("[hmc] saving outputs begin")
    output = Path(output_dir) if output_dir is not None else Path(context.stage_config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    suffix = "smoke_stage31" if smoke else "stage31"
    samples = {key: np.asarray(value) for key, value in mcmc.get_samples(group_by_chain=False).items()}
    extra = _flatten_extra_fields(mcmc.get_extra_fields())
    best_sample, best_idx, best_chi2 = best_sample_from_mcmc(mcmc, context.parameter_specs)
    chi2_dof = int(context.likelihood.rank)
    reduced_chi2 = float(best_chi2) / max(float(chi2_dof), 1.0)
    models = build_models_from_sample(context, best_sample)
    theory_cls = extract_theory_cls_jax_from_models(models)
    best_theory = np.asarray(theory_data_vector_jax(context.likelihood, theory_cls))
    measurement = measurement_for_plots(context)
    stats = gmt.comparison_statistics(measurement, best_theory)
    full_likelihood = full_likelihood_for_plots(context)
    full_measurement = measurement_from_likelihood(context, full_likelihood)
    full_best_theory = np.asarray(theory_data_vector_jax(full_likelihood, theory_cls))

    chain_path = output / f"chain_{suffix}.npz"
    payload = {f"sample__{key}": value for key, value in samples.items()}
    payload.update({f"extra__{key}": value for key, value in extra.items()})
    payload["parameter_names"] = np.asarray([spec.name for spec in context.parameter_specs])
    payload["metadata_json"] = np.asarray(
        json.dumps(
            {
                "smoke": smoke,
                "best_sample_index": best_idx,
                "best_whitened_chi2": best_chi2,
                "static_summary": static_summary(context),
                "parameter_specs": parameter_specs_jsonable(context.parameter_specs),
            }
        )
    )
    np.savez_compressed(chain_path, **payload)

    best_config = apply_sample_to_config(context.config, context.parameter_specs, best_sample)
    best_params_path = output / f"bestfit_params_{suffix}.yaml"
    with open(best_params_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(gmt.to_jsonable(best_config["params"]), handle, sort_keys=False)

    theory_path = output / f"bestfit_theory_data_vector_{suffix}.npz"
    np.savez_compressed(
        theory_path,
        ell_band=np.asarray(measurement.ell),
        data_vector=np.asarray(measurement.data_vector),
        theory_vector=best_theory,
        covariance=np.asarray(measurement.covariance),
        spectrum_names=np.asarray(measurement.names),
        best_sample_json=np.asarray(json.dumps(best_sample)),
        best_whitened_chi2=np.asarray(best_chi2),
    )

    full_theory_path = output / f"bestfit_full_theory_data_vector_{suffix}.npz"
    np.savez_compressed(
        full_theory_path,
        ell_band=np.asarray(full_measurement.ell),
        data_vector=np.asarray(full_measurement.data_vector),
        theory_vector=full_best_theory,
        covariance=np.asarray(full_measurement.covariance),
        spectrum_names=np.asarray(full_measurement.names),
        best_sample_json=np.asarray(json.dumps(best_sample)),
        best_whitened_chi2=np.asarray(best_chi2),
        likelihood_bestfit_theory_vector=np.asarray(str(theory_path)),
    )

    pdf_path = output / f"posterior_predictive_comparison_{suffix}.pdf"
    plot_paths = gmt.plot_family_comparisons(measurement, best_theory, output, pdf_path=pdf_path)
    dell_pdf_path = output / f"posterior_predictive_dell_comparison_{suffix}.pdf"
    dell_plot_paths = gmt.plot_family_dell_comparisons(
        measurement,
        best_theory,
        output,
        pdf_path=dell_pdf_path,
        filename_prefix=f"posterior_predictive_dell_{suffix}",
        total_reduced_chi2=reduced_chi2,
        chi2_dof=chi2_dof,
    )
    full_dell_pdf_path = output / f"posterior_predictive_full_dell_comparison_{suffix}.pdf"
    full_dell_plot_paths = gmt.plot_family_dell_comparisons(
        full_measurement,
        full_best_theory,
        output,
        pdf_path=full_dell_pdf_path,
        filename_prefix=f"posterior_predictive_full_dell_{suffix}",
        active_band_indices=likelihood_active_band_indices(context),
        total_reduced_chi2=reduced_chi2,
        chi2_dof=chi2_dof,
    )

    summary_path = output / f"fit_summary_{suffix}.json"
    summary = {
        "smoke": smoke,
        "chain_path": chain_path,
        "bestfit_params_path": best_params_path,
        "bestfit_theory_vector_path": theory_path,
        "bestfit_full_theory_vector_path": full_theory_path,
        "posterior_predictive_pdf": pdf_path,
        "posterior_predictive_dell_pdf": dell_pdf_path,
        "posterior_predictive_full_dell_pdf": full_dell_pdf_path,
        "plot_paths": plot_paths,
        "dell_plot_paths": dell_plot_paths,
        "full_dell_plot_paths": full_dell_plot_paths,
        "best_sample_index": best_idx,
        "best_sample": best_sample,
        "best_whitened_chi2": best_chi2,
        "pseudo_inverse_stats": stats,
        "static_summary": static_summary(context),
        "parameter_specs": parameter_specs_jsonable(context.parameter_specs),
        "priors": context.prior_config,
        "fixed_cosmology": context.config["params"]["sim_params"]["cosmo"],
        "fixed_zero_hod_arrays": list(FIXED_ZERO_HOD_ARRAYS),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(gmt.to_jsonable(summary), handle, indent=2)

    saved = {
        "chain": chain_path,
        "bestfit_params": best_params_path,
        "bestfit_theory_vector": theory_path,
        "bestfit_full_theory_vector": full_theory_path,
        "summary": summary_path,
        "pdf": pdf_path,
        "dell_pdf": dell_pdf_path,
        "full_dell_pdf": full_dell_pdf_path,
        "plots": plot_paths,
        "dell_plots": dell_plot_paths,
        "full_dell_plots": full_dell_plot_paths,
    }
    log_status(f"[hmc] saving outputs done best_chi2={best_chi2:.8e}")
    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_STAGE31_CONFIG)
    parser.add_argument("--smoke", action="store_true", help="Use the smoke sampler settings.")
    parser.add_argument("--validate-only", action="store_true", help="Prepare and validate static likelihood pieces only.")
    parser.add_argument("--compare-fiducial", action="store_true", help="Compare JAX-native fiducial windowing to theory_to_data_vector.")
    parser.add_argument("--debug-init", action="store_true", help="Check fiducial model initialization and gradients before sampling.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-warmup", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument("--chain-method", choices=["parallel", "sequential", "vectorized"], default=None)
    parser.add_argument("--max-tree-depth", type=int, default=None)
    parser.add_argument("--target-accept-prob", type=float, default=None)
    parser.add_argument(
        "--checkpoint-samples-every",
        type=int,
        default=None,
        help="If positive, run post-warmup sampling in chunks and save cumulative worker checkpoints every N samples.",
    )
    parser.add_argument("--init-params", default=None, help="Saved params YAML to use as the NUTS initial point.")
    parser.add_argument("--platform", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--gpu-sanity-check", action="store_true", help="Run one synchronized JAX matmul before setup.")
    parser.add_argument("--gpu-sanity-matrix-size", type=int, default=4096)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    t0 = time.time()
    log_status("[hmc] configure runtime begin")
    runtime = configure_numpyro_platform(args.platform)
    print(json.dumps(gmt.to_jsonable({"runtime": runtime}), indent=2), flush=True)
    if args.gpu_sanity_check:
        log_status("[hmc] gpu sanity check begin")
        check = gpu_sanity_check(args.gpu_sanity_matrix_size, require_gpu=args.platform == "gpu")
        print(json.dumps(gmt.to_jsonable({"gpu_sanity_check": check}), indent=2), flush=True)
    log_status("[hmc] prepare_fit_context begin")
    context = prepare_fit_context(args.config)
    log_status("[hmc] prepare_fit_context done")
    print(json.dumps(gmt.to_jsonable(static_summary(context)), indent=2), flush=True)
    overrides = {
        "seed": args.seed,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": args.num_chains,
        "chain_method": args.chain_method,
        "max_tree_depth": args.max_tree_depth,
        "target_accept_prob": args.target_accept_prob,
        "progress_bar": False if args.no_progress else None,
    }
    if args.compare_fiducial:
        log_status("[hmc] compare_fiducial begin")
        comparison = compare_fiducial_windowing(context)
        printable = {key: value for key, value in comparison.items() if key not in {"jax_vector", "wrapper_vector"}}
        print(json.dumps(gmt.to_jsonable(printable), indent=2), flush=True)
    init_values = pack_sample_from_params_file(context, args.init_params) if args.init_params else None
    if args.init_params:
        print(json.dumps(gmt.to_jsonable({"init_params": args.init_params}), indent=2), flush=True)
    if args.debug_init:
        log_status("[hmc] initialization diagnostics begin")
        settings = sampler_settings(context.stage_config, smoke=args.smoke, overrides=overrides)
        print(
            json.dumps(
                gmt.to_jsonable({"initialization_diagnostics": initialization_diagnostics(context, settings, init_values=init_values)}),
                indent=2,
            ),
            flush=True,
        )
    if args.validate_only:
        log_status(f"[hmc] validated stage-31 setup in {time.time() - t0:.1f} s")
        return 0
    mcmc = run_hmc(
        context,
        smoke=args.smoke,
        overrides=overrides,
        init_values=init_values,
        checkpoint_samples_every=args.checkpoint_samples_every,
        output_dir=args.output_dir,
    )
    saved = save_fit_outputs(context, mcmc, smoke=args.smoke, output_dir=args.output_dir)
    print(json.dumps(gmt.to_jsonable(saved), indent=2), flush=True)
    log_status(f"[hmc] completed stage-31 HMC run in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
