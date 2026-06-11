#!/usr/bin/env python
"""Focused kSZ harmonic-space diagnostics for the xDESI survey maps.

This module intentionally measures only the DESI velocity-momentum x ACT
temperature spectra and related nulls.  It avoids the full 46-spectrum
multi-probe covariance so kSZ estimator/debug plots can be regenerated
quickly from the cached map product.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
from scipy import stats

from multiprobe_namaster import (
    FieldMap,
    MeasurementConfig,
    NmtProbeField,
    SpectrumSpec,
    SCHEMA_MAPS,
    build_nmt_fields,
    compute_covariance_block,
    h5_attrs_to_jsonable,
    ksz_velocity_amplitudes_from_field_metadata,
    make_bins,
    measure_spectrum,
    utc_now,
)


DEFAULT_MAPS_PATH = (
    "data/xDESI/processed/multiprobe_namaster/lowres/"
    "xdesi_multiprobe_maps_nside1024_lmax2048.h5"
)
DEFAULT_OUTPUT = (
    "data/xDESI/processed/multiprobe_namaster/diagnostics/"
    "ksz_lowres_diagnostic.json"
)
DEFAULT_PLOT = (
    "data/xDESI/processed/multiprobe_namaster/diagnostics/"
    "ksz_lowres_diagnostic.png"
)


@dataclass
class KszDiagnosticConfig:
    maps_path: str = DEFAULT_MAPS_PATH
    lmax: int = 2048
    ell_min: int = 8
    n_bins: int = 32
    nside: int = 1024
    n_iter: int = 0
    n_iter_mask: int = 0
    covariance_l_toeplitz: int = -1
    covariance_l_exact: int = -1
    covariance_dl_band: int = -1
    covariance_workspace_cache_size: int = 0
    covariance_input_mode: str = "decoupled_total_bandpowers_unbinned"
    covariance_input_smooth_bandpowers: bool = True
    covariance_input_smooth_window: int = 5
    covariance_zero_parity_odd_inputs: bool = True
    include_shuffle_null: bool = True
    include_galaxy_temperature: bool = True
    subtract_joint_mean_for_ksz: bool = False
    snr_rcond: float = 1.0e-10

    def measurement_config(self) -> MeasurementConfig:
        return MeasurementConfig(
            stage="ksz_diagnostic",
            nside=int(self.nside),
            lmax=int(self.lmax),
            ell_min=int(self.ell_min),
            n_bins=int(self.n_bins),
            n_iter=int(self.n_iter),
            n_iter_mask=int(self.n_iter_mask),
            covariance_l_toeplitz=int(self.covariance_l_toeplitz),
            covariance_l_exact=int(self.covariance_l_exact),
            covariance_dl_band=int(self.covariance_dl_band),
            covariance_workspace_cache_size=int(self.covariance_workspace_cache_size),
            covariance_input_mode=str(self.covariance_input_mode),
            covariance_input_smooth_bandpowers=bool(self.covariance_input_smooth_bandpowers),
            covariance_input_smooth_window=int(self.covariance_input_smooth_window),
            covariance_zero_parity_odd_inputs=bool(self.covariance_zero_parity_odd_inputs),
            compute_covariance=True,
            include_ksz_velocity_shuffle=bool(self.include_shuffle_null),
        )


def _json_dumps(obj: object) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def _read_metadata_json(h5: h5py.File) -> Dict[str, object]:
    raw = h5.attrs.get("metadata_json", "{}")
    return json.loads(raw)


def load_selected_map_fields(
    path: str | Path,
    field_names: Sequence[str],
) -> Tuple[Dict[str, FieldMap], Dict[str, object]]:
    """Load only the requested fields and their referenced masks from a map product."""

    path = Path(path)
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema") != SCHEMA_MAPS:
            raise ValueError(f"{path} is not a {SCHEMA_MAPS} product.")
        metadata = _read_metadata_json(h5)
        masks: Dict[str, np.ndarray] = {}
        fields: Dict[str, FieldMap] = {}
        for name in field_names:
            if f"fields/{name}" not in h5:
                raise KeyError(f"Missing field {name!r} in {path}.")
            g = h5[f"fields/{name}"]
            maps = [g[f"map{i}"][:] for i in range(len([k for k in g if k.startswith("map")]))]
            mask_name = str(g.attrs["mask_ref"])
            if mask_name not in masks:
                masks[mask_name] = h5[f"masks/{mask_name}"][:]
            catalog: Dict[str, np.ndarray] = {}
            if "catalog" in g:
                cg = g["catalog"]
                catalog = {key: np.asarray(cg[key][:], dtype=np.float64) for key in cg}
            fields[name] = FieldMap(
                name=str(g.attrs["name"]),
                label=str(g.attrs["label"]),
                kind=str(g.attrs["kind"]),
                spin=int(g.attrs["spin"]),
                maps=maps,
                mask=masks[mask_name],
                mask_name=mask_name,
                metadata=json.loads(g.attrs["metadata_json"]),
                catalog=catalog,
            )
    return fields, metadata


def ksz_specs(prefix: str = "desi_pi_act_T", pi_prefix: str = "pi") -> List[SpectrumSpec]:
    specs: List[SpectrumSpec] = []
    for pz_bin in range(1, 5):
        specs.append(
            SpectrumSpec(
                name=f"{prefix}_pz{pz_bin}",
                family=prefix,
                fields=(f"{pi_prefix}{pz_bin}", "T"),
                component=0,
                label=f"DESI {pi_prefix} pz {pz_bin} x ACT T",
                theory_key=f"desi_pi_act_T_pz{pz_bin}",
                metadata={"desi_pz": pz_bin},
            )
        )
    return specs


def galaxy_temperature_specs() -> List[SpectrumSpec]:
    specs: List[SpectrumSpec] = []
    for pz_bin in range(1, 5):
        specs.append(
            SpectrumSpec(
                name=f"desi_g_act_T_pz{pz_bin}",
                family="desi_g_act_T_leakage",
                fields=(f"g{pz_bin}", "T"),
                component=0,
                label=f"DESI g pz {pz_bin} x ACT T",
                theory_key="diagnostic",
                metadata={"desi_pz": pz_bin},
            )
        )
    return specs


def required_fields(config: KszDiagnosticConfig) -> List[str]:
    fields = ["T"]
    fields.extend([f"pi{i}" for i in range(1, 5)])
    if config.include_shuffle_null:
        fields.extend([f"pi_shuf{i}" for i in range(1, 5)])
    if config.include_galaxy_temperature:
        fields.extend([f"g{i}" for i in range(1, 5)])
    return fields


def _weighted_mean(values: np.ndarray, mask: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(mask) & (mask > 0)
    if not np.any(good):
        return np.nan
    return float(np.sum(values[good] * mask[good]) / np.sum(mask[good]))


def _weighted_rms(values: np.ndarray, mask: np.ndarray) -> float:
    good = np.isfinite(values) & np.isfinite(mask) & (mask > 0)
    if not np.any(good):
        return np.nan
    mean = np.sum(values[good] * mask[good]) / np.sum(mask[good])
    return float(np.sqrt(np.sum(mask[good] * np.square(values[good] - mean)) / np.sum(mask[good])))


def _quantiles(values: np.ndarray, mask: np.ndarray, qs: Sequence[float] = (0.001, 0.01, 0.5, 0.99, 0.999)) -> List[float]:
    good = np.isfinite(values) & np.isfinite(mask) & (mask > 0)
    if not np.any(good):
        return [np.nan for _ in qs]
    return [float(x) for x in np.quantile(values[good], qs)]


def mask_overlap_summary(fields: Mapping[str, FieldMap]) -> Dict[str, object]:
    pi = fields["pi1"]
    temp = fields["T"]
    product = pi.mask * temp.mask
    return {
        "desi_mask_name": pi.mask_name,
        "temperature_mask_name": temp.mask_name,
        "desi_fsky_mean_mask": float(np.mean(pi.mask)),
        "temperature_fsky_mean_mask": float(np.mean(temp.mask)),
        "joint_fsky_mean_mask_product": float(np.mean(product)),
        "joint_binary_fsky": float(np.mean((pi.mask > 0) & (temp.mask > 0))),
        "joint_mask_sum": float(np.sum(product, dtype=np.float64)),
    }


def map_ingredient_summary(fields: Mapping[str, FieldMap]) -> Dict[str, object]:
    out: Dict[str, object] = {"overlap": mask_overlap_summary(fields), "temperature": {}, "pi": {}, "galaxy": {}}
    temp = fields["T"]
    t = temp.maps[0]
    joint_mask = fields["pi1"].mask * temp.mask
    out["temperature"] = {
        "units": temp.metadata.get("units", ""),
        "masked_mean_T_own_mask": _weighted_mean(t, temp.mask),
        "masked_rms_T_own_mask": _weighted_rms(t, temp.mask),
        "masked_mean_T_joint_mask": _weighted_mean(t, joint_mask),
        "masked_rms_T_joint_mask": _weighted_rms(t, joint_mask),
        "quantiles_T_joint_mask": _quantiles(t, joint_mask),
    }
    for pz_bin in range(1, 5):
        pi = fields[f"pi{pz_bin}"]
        pi_map = pi.maps[0]
        pi_joint_mask = pi.mask * temp.mask
        meta = dict(pi.metadata)
        out["pi"][f"pz{pz_bin}"] = {
            "masked_mean_pi_own_mask": _weighted_mean(pi_map, pi.mask),
            "masked_rms_pi_own_mask": _weighted_rms(pi_map, pi.mask),
            "masked_mean_pi_joint_mask": _weighted_mean(pi_map, pi_joint_mask),
            "masked_rms_pi_joint_mask": _weighted_rms(pi_map, pi_joint_mask),
            "quantiles_pi_joint_mask": _quantiles(pi_map, pi_joint_mask),
            "n_gal": meta.get("n_gal"),
            "mean_z": meta.get("mean_z"),
            "n_eff_per_sr": meta.get("n_eff_per_sr"),
            "rms_rec_vr_over_c_weighted": meta.get("rms_rec_vr_over_c_weighted", meta.get("rms_rec_vr_over_c")),
            "sigma_rec_vr_over_c_weighted": meta.get("sigma_rec_vr_over_c_weighted", meta.get("sigma_rec_vr_over_c")),
            "sigma_true_gas_over_c": meta.get("sigma_true_gas_over_c"),
            "mean_vr_over_c_weighted": meta.get("mean_vr_over_c_weighted", meta.get("mean_vr_over_c")),
            "alpha_galaxy_to_random": meta.get("alpha_galaxy_to_random"),
        }
        if pi.catalog:
            cat_w = np.asarray(pi.catalog.get("weight", []), dtype=np.float64)
            cat_v = np.asarray(pi.catalog.get("field", []), dtype=np.float64)
            good = np.isfinite(cat_w) & np.isfinite(cat_v) & (cat_w > 0)
            if np.any(good):
                mean_v = float(np.sum(cat_w[good] * cat_v[good]) / np.sum(cat_w[good]))
                rms_v = float(np.sqrt(np.sum(cat_w[good] * cat_v[good] ** 2) / np.sum(cat_w[good])))
            else:
                mean_v = np.nan
                rms_v = np.nan
            out["pi"][f"pz{pz_bin}"].update(
                {
                    "catalog_momentum_available": True,
                    "catalog_n_sources": int(cat_v.size),
                    "catalog_weighted_mean_field": mean_v,
                    "catalog_weighted_rms_field": rms_v,
                    "namaster_field_class": meta.get("namaster_field_class", ""),
                }
            )
        else:
            out["pi"][f"pz{pz_bin}"]["catalog_momentum_available"] = False
        if f"g{pz_bin}" in fields:
            g = fields[f"g{pz_bin}"]
            g_map = g.maps[0]
            out["galaxy"][f"pz{pz_bin}"] = {
                "masked_mean_delta_own_mask": _weighted_mean(g_map, g.mask),
                "masked_rms_delta_own_mask": _weighted_rms(g_map, g.mask),
                "masked_mean_delta_joint_mask": _weighted_mean(g_map, pi_joint_mask),
                "masked_rms_delta_joint_mask": _weighted_rms(g_map, pi_joint_mask),
            }
    try:
        # ksz_velocity_amplitudes_from_field_metadata expects the saved
        # measurement-product field-metadata shape:
        # {field_name: {"metadata": field_metadata}}.
        out["default_ksz_A_v_by_pz"] = ksz_velocity_amplitudes_from_field_metadata(
            {name: {"metadata": field.metadata} for name, field in fields.items()}
        )
    except Exception as exc:
        out["default_ksz_A_v_error"] = str(exc)
    return out


def subtract_ksz_joint_mask_means(fields: MutableMapping[str, FieldMap]) -> Dict[str, object]:
    """Subtract means over the DESI x ACT temperature joint mask for kSZ fields."""

    if "T" not in fields or "pi1" not in fields:
        raise KeyError("Need T and pi1 fields to define the kSZ joint mask.")
    joint_mask = fields["T"].mask * fields["pi1"].mask
    changes: Dict[str, object] = {
        "joint_mask_sum": float(np.sum(joint_mask, dtype=np.float64)),
        "fields": {},
    }
    t_mean = _weighted_mean(fields["T"].maps[0], joint_mask)
    fields["T"].maps[0][:] = fields["T"].maps[0] - np.float32(t_mean)
    fields["T"].maps[0][fields["T"].mask <= 0] = 0.0
    fields["T"].metadata = {
        **fields["T"].metadata,
        "ksz_joint_mean_subtracted": t_mean,
        "ksz_joint_mean_subtraction_note": "Subtracted weighted mean over DESI random mask x ACT T mask for kSZ diagnostic.",
    }
    changes["fields"]["T"] = {"subtracted_mean": t_mean}
    for prefix in ("pi", "pi_shuf"):
        for pz_bin in range(1, 5):
            name = f"{prefix}{pz_bin}"
            if name not in fields:
                continue
            mean = _weighted_mean(fields[name].maps[0], joint_mask)
            fields[name].maps[0][:] = fields[name].maps[0] - np.float32(mean)
            fields[name].maps[0][fields[name].mask <= 0] = 0.0
            if fields[name].catalog and "field" in fields[name].catalog:
                fields[name].catalog["field"] = np.asarray(fields[name].catalog["field"], dtype=np.float64) - mean
            fields[name].metadata = {
                **fields[name].metadata,
                "ksz_joint_mean_subtracted": mean,
                "ksz_joint_mean_subtraction_note": "Subtracted weighted mean over DESI random mask x ACT T mask for kSZ diagnostic.",
            }
            changes["fields"][name] = {"subtracted_mean": mean}
    return changes


def measure_specs(
    specs: Sequence[SpectrumSpec],
    fields: Mapping[str, NmtProbeField],
    bins: nmt.NmtBin,
    config: MeasurementConfig,
) -> Dict[str, Dict[str, object]]:
    workspace_cache: Dict[Tuple[str, str], nmt.NmtWorkspace] = {}
    out: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        out[spec.name] = measure_spectrum(spec, fields, bins, workspace_cache, config)
    return out


def joint_covariance(
    specs: Sequence[SpectrumSpec],
    fields: Mapping[str, NmtProbeField],
    bins: nmt.NmtBin,
    config: MeasurementConfig,
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    workspace_cache: Dict[Tuple[str, str], nmt.NmtWorkspace] = {}
    cov_workspace_cache: Dict[Tuple[str, str, str, str], nmt.NmtCovarianceWorkspace] = {}
    input_cl_cache: Dict[Tuple[str, ...], np.ndarray] = {}
    n_per = bins.get_n_bands()
    n_data = n_per * len(specs)
    cov = np.zeros((n_data, n_data), dtype=np.float64)
    slices: Dict[str, Tuple[int, int]] = {}
    for i, spec_i in enumerate(specs):
        slices[spec_i.name] = (i * n_per, (i + 1) * n_per)
        for j, spec_j in enumerate(specs[i:], start=i):
            block = compute_covariance_block(
                spec_i,
                spec_j,
                fields,
                bins,
                workspace_cache,
                cov_workspace_cache,
                input_cl_cache,
                config,
            )
            si = slice(i * n_per, (i + 1) * n_per)
            sj = slice(j * n_per, (j + 1) * n_per)
            cov[si, sj] = block
            if i != j:
                cov[sj, si] = block.T
    return cov, slices


def attach_covariance_errors(
    spectra: MutableMapping[str, MutableMapping[str, object]],
    cov: np.ndarray,
    slices: Mapping[str, Tuple[int, int]],
) -> None:
    for name, (start, stop) in slices.items():
        if name not in spectra:
            continue
        block = cov[start:stop, start:stop]
        spectra[name]["err"] = np.sqrt(np.clip(np.diag(block), 0.0, np.inf))


def concatenate_data_vector(
    spectra: Mapping[str, Mapping[str, object]],
    specs: Sequence[SpectrumSpec],
) -> np.ndarray:
    return np.concatenate([np.asarray(spectra[spec.name]["cl"], dtype=np.float64) for spec in specs])


def covariance_subselect(
    cov: np.ndarray,
    specs: Sequence[SpectrumSpec],
    ell: np.ndarray,
    ell_min: Optional[float] = None,
    ell_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_per = len(ell)
    band = np.ones(n_per, dtype=bool)
    if ell_min is not None:
        band &= ell >= float(ell_min)
    if ell_max is not None:
        band &= ell <= float(ell_max)
    idx = np.concatenate([i * n_per + np.where(band)[0] for i in range(len(specs))])
    return cov[np.ix_(idx, idx)], idx


def null_test_from_covariance(
    data: np.ndarray,
    cov: np.ndarray,
    *,
    rcond: float = 1.0e-10,
) -> Dict[str, object]:
    data = np.asarray(data, dtype=np.float64)
    cov = 0.5 * (np.asarray(cov, dtype=np.float64) + np.asarray(cov, dtype=np.float64).T)
    good = np.isfinite(data) & np.isfinite(np.diag(cov)) & (np.diag(cov) > 0)
    if not np.all(good):
        data = data[good]
        cov = cov[np.ix_(good, good)]
    diag = np.diag(cov)
    diag_chi2 = float(np.sum(np.square(data) / diag)) if data.size else np.nan
    diag_pte = float(stats.chi2.sf(diag_chi2, data.size)) if data.size and np.isfinite(diag_chi2) else np.nan
    try:
        evals, evecs = np.linalg.eigh(cov)
        max_eval = float(np.max(evals)) if evals.size else np.nan
        threshold = max(float(rcond) * max(max_eval, 0.0), 0.0)
        keep = evals > threshold
        coeff = evecs[:, keep].T @ data if np.any(keep) else np.asarray([], dtype=np.float64)
        chi2 = float(np.sum(np.square(coeff) / evals[keep])) if coeff.size else np.nan
        dof = int(np.count_nonzero(keep))
        pte = float(stats.chi2.sf(chi2, dof)) if dof > 0 and np.isfinite(chi2) else np.nan
        return {
            "n_data": int(data.size),
            "n_kept_modes": dof,
            "n_dropped_modes": int(data.size - dof),
            "n_negative_eig": int(np.count_nonzero(evals < -threshold)),
            "eig_min": float(evals[0]) if evals.size else np.nan,
            "eig_max": max_eval,
            "eig_threshold": threshold,
            "diag_chi2": diag_chi2,
            "diag_sqrt_chi2": float(np.sqrt(max(diag_chi2, 0.0))) if np.isfinite(diag_chi2) else np.nan,
            "diag_dof": int(data.size),
            "diag_pte": diag_pte,
            "full_cov_chi2": chi2,
            "full_cov_sqrt_chi2": float(np.sqrt(max(chi2, 0.0))) if np.isfinite(chi2) else np.nan,
            "full_cov_dof": dof,
            "full_cov_pte": pte,
            "interpretation": (
                "This is a zero-signal chi-square/null test. sqrt(chi2) is not a "
                "kSZ detection SNR for a many-bin vector. A detection SNR needs a "
                "model/template vector and should be computed as a fitted amplitude."
            ),
        }
    except np.linalg.LinAlgError as exc:
        return {
            "n_data": int(data.size),
            "diag_chi2": diag_chi2,
            "diag_sqrt_chi2": float(np.sqrt(max(diag_chi2, 0.0))) if np.isfinite(diag_chi2) else np.nan,
            "diag_dof": int(data.size),
            "diag_pte": diag_pte,
            "full_cov_error": str(exc),
        }


def template_amplitude_fit(
    data: np.ndarray,
    cov: np.ndarray,
    template: np.ndarray,
    *,
    rcond: float = 1.0e-10,
) -> Dict[str, object]:
    """Fit a single template amplitude A with data = A * template + noise."""

    data = np.asarray(data, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    cov = 0.5 * (np.asarray(cov, dtype=np.float64) + np.asarray(cov, dtype=np.float64).T)
    good = (
        np.isfinite(data)
        & np.isfinite(template)
        & np.isfinite(np.diag(cov))
        & (np.diag(cov) > 0)
    )
    data = data[good]
    template = template[good]
    cov = cov[np.ix_(good, good)]
    evals, evecs = np.linalg.eigh(cov)
    threshold = max(float(rcond) * max(float(np.max(evals)), 0.0), 0.0)
    keep = evals > threshold
    if not np.any(keep):
        return {"amplitude": np.nan, "amplitude_sigma": np.nan, "amplitude_snr": np.nan, "n_kept_modes": 0}
    d = evecs[:, keep].T @ data
    t = evecs[:, keep].T @ template
    inv_var = 1.0 / evals[keep]
    fisher = float(np.sum(t * t * inv_var))
    numer = float(np.sum(t * d * inv_var))
    amp = numer / fisher if fisher > 0 else np.nan
    sigma = 1.0 / np.sqrt(fisher) if fisher > 0 else np.nan
    return {
        "amplitude": amp,
        "amplitude_sigma": sigma,
        "amplitude_snr": float(amp / sigma) if sigma > 0 else np.nan,
        "n_kept_modes": int(np.count_nonzero(keep)),
        "note": "Only valid if the supplied template shape is physically meaningful.",
    }


def summarize_null_tests(
    spectra: Mapping[str, Mapping[str, object]],
    specs: Sequence[SpectrumSpec],
    cov: np.ndarray,
    rcond: float = 1.0e-10,
) -> Dict[str, object]:
    ell = np.asarray(next(iter(spectra.values()))["ell"], dtype=np.float64)
    data = concatenate_data_vector(spectra, specs)
    out = {"all_bands": null_test_from_covariance(data, cov, rcond=rcond)}
    for cut in (300.0, 500.0, 800.0, 1000.0):
        subcov, idx = covariance_subselect(cov, specs, ell, ell_min=cut)
        out[f"ell_ge_{int(cut)}"] = null_test_from_covariance(data[idx], subcov, rcond=rcond)
    per_bin: Dict[str, object] = {}
    n_per = len(ell)
    for i, spec in enumerate(specs):
        sl = slice(i * n_per, (i + 1) * n_per)
        per_bin[spec.name] = null_test_from_covariance(data[sl], cov[sl, sl], rcond=rcond)
    out["per_bin"] = per_bin
    return out


def _serializable_spectrum(spec: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in spec.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif key == "metadata":
            out[key] = value
        elif isinstance(value, tuple):
            out[key] = list(value)
        elif value is None:
            out[key] = None
        else:
            out[key] = value
    return out


def run_ksz_diagnostic(config: KszDiagnosticConfig) -> Dict[str, object]:
    field_maps, map_metadata = load_selected_map_fields(config.maps_path, required_fields(config))
    joint_mean_subtraction: Dict[str, object] = {}
    if config.subtract_joint_mean_for_ksz:
        joint_mean_subtraction = subtract_ksz_joint_mask_means(field_maps)
    mconfig = config.measurement_config()
    bins = make_bins(mconfig)
    nmt_fields = build_nmt_fields(field_maps, mconfig)
    main_specs = ksz_specs()
    shuffle_specs = ksz_specs(prefix="shuffle_pi_act_T", pi_prefix="pi_shuf") if config.include_shuffle_null else []
    gt_specs = galaxy_temperature_specs() if config.include_galaxy_temperature else []

    main_spectra = measure_specs(main_specs, nmt_fields, bins, mconfig)
    shuffle_spectra = measure_specs(shuffle_specs, nmt_fields, bins, mconfig) if shuffle_specs else {}
    gt_spectra = measure_specs(gt_specs, nmt_fields, bins, mconfig) if gt_specs else {}

    cov, slices = joint_covariance(main_specs, nmt_fields, bins, mconfig)
    attach_covariance_errors(main_spectra, cov, slices)
    null_tests = summarize_null_tests(main_spectra, main_specs, cov, config.snr_rcond)
    shuffle_null_tests: Dict[str, object] = {}
    if shuffle_specs:
        shuffle_cov, shuffle_slices = joint_covariance(shuffle_specs, nmt_fields, bins, mconfig)
        attach_covariance_errors(shuffle_spectra, shuffle_cov, shuffle_slices)
        shuffle_null_tests = summarize_null_tests(shuffle_spectra, shuffle_specs, shuffle_cov, config.snr_rcond)

    return {
        "created_utc": utc_now(),
        "config": asdict(config),
        "map_metadata_config": map_metadata.get("config", {}),
        "joint_mean_subtraction": joint_mean_subtraction,
        "ingredient_summary": map_ingredient_summary(field_maps),
        "ell": bins.get_effective_ells().tolist(),
        "spectra": {name: _serializable_spectrum(spec) for name, spec in main_spectra.items()},
        "shuffle_spectra": {name: _serializable_spectrum(spec) for name, spec in shuffle_spectra.items()},
        "galaxy_temperature_spectra": {name: _serializable_spectrum(spec) for name, spec in gt_spectra.items()},
        "covariance": cov.tolist(),
        "covariance_slices": {name: list(sl) for name, sl in slices.items()},
        "null_tests": null_tests,
        "shuffle_null_tests": shuffle_null_tests,
        "notes": {
            "measured_quantity": "C_ell^{pi,T_uK}; pi is dimensionless weighted reconstructed velocity over c.",
            "expected_ksz_sign": "For positive C_ell^{g,tau}, the simple model predicts C_ell^{pi,T} = -T_CMB_uK * A_v * C_ell^{g,tau}. Diagnostic plots therefore show both raw ell*C_ell and -ell*C_ell.",
            "snr_interpretation": "The diagnostic reports chi-square/PTE null tests. Do not interpret sqrt(chi2) as a kSZ detection SNR. A template-amplitude SNR requires a theory vector.",
            "lowres_caveat": "This low-res product stops at ell=2048. The kSZ reference analyses use much higher ell ranges, so low-res SNR is expected to be modest.",
        },
    }


def write_result(result: Mapping[str, object], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(_json_dumps(result))
    tmp.replace(out)
    return out


def _dynamic_symmetric_ylim(
    values: Iterable[np.ndarray],
    errors: Iterable[np.ndarray] = (),
    floor: float = 1.0e-4,
    percentile: float = 98.0,
) -> Tuple[float, float]:
    parts = [np.ravel(np.asarray(v, dtype=np.float64)) for v in values]
    parts.extend([np.ravel(np.asarray(e, dtype=np.float64)) for e in errors])
    data = np.concatenate([p[np.isfinite(p)] for p in parts if p.size]) if parts else np.asarray([])
    if data.size == 0:
        return -1.0, 1.0
    lim = float(np.nanpercentile(np.abs(data), float(percentile)))
    lim = max(lim * 1.25, floor)
    return -lim, lim


def plot_ksz_diagnostic(result: Mapping[str, object], output: Optional[str | Path] = None):
    ell = np.asarray(result["ell"], dtype=np.float64)
    spectra = result["spectra"]
    shuffle = result.get("shuffle_spectra", {})
    pz_names = [f"desi_pi_act_T_pz{i}" for i in range(1, 5)]
    shuf_names = [f"shuffle_pi_act_T_pz{i}" for i in range(1, 5)]

    y_main = [ell * np.asarray(spectra[name]["cl"], dtype=np.float64) for name in pz_names]
    e_main = [ell * np.asarray(spectra[name].get("err", np.zeros_like(ell)), dtype=np.float64) for name in pz_names]
    colors = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True, sharey=False)
    high_ell = ell >= 1000.0
    if not np.any(high_ell):
        high_ell = np.ones_like(ell, dtype=bool)
    for i, ax in enumerate(axes.flat, start=1):
        name = f"desi_pi_act_T_pz{i}"
        cl = np.asarray(spectra[name]["cl"], dtype=np.float64)
        err = np.asarray(spectra[name].get("err", np.zeros_like(ell)), dtype=np.float64)
        y = ell * cl
        yerr = ell * err
        ax.errorbar(ell, y, yerr=yerr, fmt="o", ms=4, capsize=2, color=colors[i - 1], label="pi x T")
        ylim_values = [y[high_ell], y[high_ell] + yerr[high_ell], y[high_ell] - yerr[high_ell]]
        shuf_name = f"shuffle_pi_act_T_pz{i}"
        if shuf_name in shuffle:
            shuf_cl = np.asarray(shuffle[shuf_name]["cl"], dtype=np.float64)
            y_shuf = ell * shuf_cl
            ylim_values.append(y_shuf[high_ell])
            ax.plot(ell, y_shuf, color="0.35", lw=1.2, alpha=0.8, label="velocity shuffle")
        ax.axhline(0.0, color="0.2", lw=0.8)
        ax.axvspan(1000.0, 2048.0, color="0.92", zorder=-10)
        ax.set_title(f"DESI pz {i}")
        lo, hi = _dynamic_symmetric_ylim(ylim_values, floor=5.0e-8, percentile=99.0)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel(r"Multipole $\ell$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\ell C_\ell^{\pi T}$ [$\mu$K]")
    fig.suptitle("kSZ estimator diagnostic: raw DESI pi x ACT T", y=0.995)
    fig.tight_layout()
    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=170)
    return fig, axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-path", default=DEFAULT_MAPS_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-output", default=DEFAULT_PLOT)
    parser.add_argument("--lmax", type=int, default=2048)
    parser.add_argument("--ell-min", type=int, default=8)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--covariance-input-smooth-window", type=int, default=5)
    parser.add_argument("--no-covariance-input-smoothing", action="store_true")
    parser.add_argument("--keep-covariance-parity-odd-inputs", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--no-gT", action="store_true")
    parser.add_argument("--subtract-joint-mean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = KszDiagnosticConfig(
        maps_path=args.maps_path,
        lmax=args.lmax,
        ell_min=args.ell_min,
        n_bins=args.n_bins,
        covariance_input_smooth_bandpowers=not args.no_covariance_input_smoothing,
        covariance_input_smooth_window=args.covariance_input_smooth_window,
        covariance_zero_parity_odd_inputs=not args.keep_covariance_parity_odd_inputs,
        include_shuffle_null=not args.no_shuffle,
        include_galaxy_temperature=not args.no_gT,
        subtract_joint_mean_for_ksz=bool(args.subtract_joint_mean),
    )
    print(f"[{utc_now()}] Running kSZ diagnostic from {config.maps_path}", flush=True)
    result = run_ksz_diagnostic(config)
    out = write_result(result, args.output)
    print(f"[{utc_now()}] Wrote {out.resolve()}", flush=True)
    if args.plot_output:
        plot_ksz_diagnostic(result, args.plot_output)
        print(f"[{utc_now()}] Wrote {Path(args.plot_output).resolve()}", flush=True)
    print("[NULL_TESTS]", _json_dumps(result["null_tests"]), flush=True)


if __name__ == "__main__":
    main()
