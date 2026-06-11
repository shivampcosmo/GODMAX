"""High-ell kSZ-only NaMaster measurement.

This script measures the DESI DR9 weighted velocity-momentum catalog field
cross-correlated with the ACT DR6 CMB temperature map. It intentionally avoids
building the full multi-probe map product at nside=4096; the momentum field is
constructed with NaMaster's catalog estimator, following the kSZ tutorial.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import healpy as hp
import matplotlib
import numpy as np
import pymaster as nmt

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiprobe_namaster import (
    DESI_DR9_SELECTION_DATASET,
    DESI_DR9_WEIGHT_DATASET,
    FieldMap,
    MeasurementConfig,
    NmtProbeField,
    SpectrumSpec,
    SurveyBundle,
    _clean_map,
    _clean_mask,
    _load_healpix_random_count_map,
    _select_covariance_component_block,
    _subtract_masked_mean,
    covariance_diagnostics,
    h5_attrs_to_jsonable,
    read_enmap_from_h5,
)
from pixell import reproject

try:
    from scipy.stats import chi2 as scipy_chi2
except Exception:  # pragma: no cover - diagnostic convenience only
    scipy_chi2 = None


SCHEMA = "xdesi_ksz_high_ell_measurement_v1"
DEFAULT_OUTPUT = Path("data/xDESI/processed/multiprobe_namaster/ksz_high_ell/ksz_high_ell_nside4096_lmax8192.h5")
DEFAULT_PLOT = Path("data/xDESI/processed/multiprobe_namaster/diagnostics/ksz_high_ell_Dl_lmax8192.png")
DEFAULT_SUMMARY = Path("data/xDESI/processed/multiprobe_namaster/diagnostics/ksz_high_ell_Dl_lmax8192_summary.json")


@dataclass
class HighEllKszConfig:
    survey_root: str = "data/xDESI/survey_data"
    output: str = str(DEFAULT_OUTPUT)
    plot_output: str = str(DEFAULT_PLOT)
    summary_output: str = str(DEFAULT_SUMMARY)
    workspace_cache_dir: Optional[str] = None
    nside: int = 4096
    lmax: int = 8192
    ell_min: int = 300
    n_bins: int = 17
    pz_bins: Tuple[int, ...] = (1, 2, 3, 4)
    act_downgrade: int = 1
    subtract_act_masked_mean: bool = True
    subtract_velocity_weighted_mean: bool = False
    velocity_sign: float = 1.0
    velocity_clip_sigma: Optional[float] = None
    shuffle_velocity_seed: Optional[int] = None
    plot_sign: float = -1.0
    fit_ell_min: float = 1000.0
    fit_ell_max: float = 7000.0
    plot_ell_min: float = 1000.0
    plot_ell_max: float = 8000.0
    covariance_l_toeplitz: int = -1
    covariance_l_exact: int = -1
    covariance_dl_band: int = -1
    masked_on_input_temperature: bool = False


def make_paper_like_log_edges(ell_min: int, ell_max: int, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return right-exclusive integer edges for paper-like high-ell kSZ bins.

    The 2604.19744 paper describes 17 theory/multipole bins over roughly
    ell=300..10000 and fits 1000..7000. For our lmax=8192 diagnostic, we keep
    17 logarithmically spaced bins over ell_min..ell_max.
    """

    if ell_min < 2:
        raise ValueError("ell_min must be >= 2 for logarithmic kSZ binning.")
    if ell_max <= ell_min:
        raise ValueError("ell_max must exceed ell_min.")
    raw = np.exp(np.linspace(np.log(float(ell_min)), np.log(float(ell_max) + 1.0), int(n_bins) + 1))
    edges = np.rint(raw).astype(np.int64)
    edges[0] = int(ell_min)
    edges[-1] = int(ell_max) + 1
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    edges[-1] = int(ell_max) + 1
    return edges[:-1].astype(np.int32), edges[1:].astype(np.int32)


def dl_factor(ell: np.ndarray) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64)
    return ell * (ell + 1.0) / (2.0 * np.pi)


def safe_snr(data: np.ndarray, cov: np.ndarray, rcond: float = 1.0e-10) -> Dict[str, object]:
    data = np.asarray(data, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    good = np.isfinite(data) & np.all(np.isfinite(cov), axis=0) & np.all(np.isfinite(cov), axis=1)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != data.size:
        return {"snr": float("nan"), "chi2": float("nan"), "n": 0, "reason": "shape_mismatch"}
    diag = np.diag(cov)
    good &= np.isfinite(diag) & (diag > 0)
    if not np.any(good):
        return {"snr": float("nan"), "chi2": float("nan"), "n": 0, "reason": "no_positive_diagonal"}
    idx = np.flatnonzero(good)
    subcov = cov[np.ix_(idx, idx)]
    subdata = data[idx]
    try:
        inv = np.linalg.pinv(subcov, rcond=rcond, hermitian=True)
    except TypeError:
        inv = np.linalg.pinv(subcov, rcond=rcond)
    chi2 = float(subdata @ inv @ subdata)
    out = {"snr": float(np.sqrt(max(chi2, 0.0))), "chi2": chi2, "n": int(idx.size)}
    if scipy_chi2 is not None:
        out["zero_signal_pte"] = float(scipy_chi2.sf(chi2, int(idx.size)))
    return out


def _covariance_workspace_from_fields(
    f_a1: nmt.NmtField,
    f_a2: nmt.NmtField,
    f_b1: nmt.NmtField,
    f_b2: nmt.NmtField,
    config: HighEllKszConfig,
) -> nmt.NmtCovarianceWorkspace:
    kwargs = {
        "l_toeplitz": int(config.covariance_l_toeplitz),
        "l_exact": int(config.covariance_l_exact),
        "dl_band": int(config.covariance_dl_band),
    }
    try:
        return nmt.NmtCovarianceWorkspace.from_fields(f_a1, f_a2, f_b1, f_b2, all_spins=True, **kwargs)
    except TypeError:
        return nmt.NmtCovarianceWorkspace.from_fields(f_a1, f_a2, f_b1, f_b2, spin0_only=False, **kwargs)


def _workspace_from_fields(
    f_a: nmt.NmtField,
    f_b: nmt.NmtField,
    bins: nmt.NmtBin,
    config: HighEllKszConfig,
) -> nmt.NmtWorkspace:
    return nmt.NmtWorkspace.from_fields(
        f_a,
        f_b,
        bins,
        l_toeplitz=int(config.covariance_l_toeplitz),
        l_exact=int(config.covariance_l_exact),
        dl_band=int(config.covariance_dl_band),
    )


def load_desi_mask(bundle: SurveyBundle, config: HighEllKszConfig) -> Tuple[np.ndarray, Dict[str, object]]:
    counts = _load_healpix_random_count_map(bundle.desi_random_count_maps, config.nside)
    valid = counts > 0
    if not np.any(valid):
        raise ValueError("DESI random-count map has no valid pixels.")
    mean_count = float(np.mean(counts[valid], dtype=np.float64))
    mask = np.zeros_like(counts, dtype=np.float32)
    mask[valid] = counts[valid] / mean_count
    meta = {
        "source": str(bundle.desi_random_count_maps),
        "dataset": f"nside{int(config.nside)}/random_count",
        "nside": int(config.nside),
        "mean_random_count_valid": mean_count,
        "n_valid_pixels": int(np.count_nonzero(valid)),
        "fsky_weighted_mean": float(np.mean(mask, dtype=np.float64)),
        "fsky_squared_mean": float(np.mean(mask.astype(np.float64) ** 2)),
        "caveat": "Uses the transferred one-realization DR9 random-count nside4096 mask directly.",
    }
    del counts, valid
    gc.collect()
    return mask, meta


def load_act_temperature_healpix(bundle: SurveyBundle, config: HighEllKszConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    mask_em = read_enmap_from_h5(bundle.act_cmb, "maps/analysis_mask", "mask_wcs_header", config.act_downgrade)
    mask_hp = reproject.map2healpix(mask_em, nside=int(config.nside), lmax=int(config.lmax))
    del mask_em
    gc.collect()

    map_em = read_enmap_from_h5(bundle.act_cmb, "maps/cmb_temperature", "map_wcs_header", config.act_downgrade)
    t_hp = reproject.map2healpix(map_em, nside=int(config.nside), lmax=int(config.lmax))
    del map_em
    gc.collect()

    mask_hp = _clean_mask(mask_hp).astype(np.float32, copy=False)
    t_hp = _clean_map(t_hp).astype(np.float32, copy=False)
    mean_before = float(np.sum(t_hp.astype(np.float64) * mask_hp.astype(np.float64)) / np.sum(mask_hp, dtype=np.float64))
    if config.subtract_act_masked_mean:
        t_hp = _subtract_masked_mean(t_hp, mask_hp).astype(np.float32, copy=False)
    else:
        t_hp[mask_hp <= 0] = 0.0
    mean_after = float(np.sum(t_hp.astype(np.float64) * mask_hp.astype(np.float64)) / np.sum(mask_hp, dtype=np.float64))

    with h5py.File(bundle.act_cmb, "r") as h5:
        attrs = h5_attrs_to_jsonable(h5["geometry"].attrs)
    meta = {
        "source": str(bundle.act_cmb),
        "map_dataset": "maps/cmb_temperature",
        "mask_dataset": "maps/analysis_mask",
        "units": "uK_CMB_likely",
        "nside": int(config.nside),
        "lmax": int(config.lmax),
        "act_downgrade": int(config.act_downgrade),
        "subtract_masked_mean": bool(config.subtract_act_masked_mean),
        "masked_mean_before": mean_before,
        "masked_mean_after": mean_after,
        "fsky_weighted_mean": float(np.mean(mask_hp, dtype=np.float64)),
        "fsky_squared_mean": float(np.mean(mask_hp.astype(np.float64) ** 2)),
        "geometry_attrs": attrs,
    }
    return t_hp, mask_hp, meta


def load_desi_catalog_by_pz(
    bundle: SurveyBundle,
    desi_mask: np.ndarray,
    config: HighEllKszConfig,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, object]]:
    npix = hp.nside2npix(int(config.nside))
    if desi_mask.shape != (npix,):
        raise ValueError(f"DESI mask has shape {desi_mask.shape}; expected {(npix,)}.")

    out: Dict[int, Dict[str, np.ndarray]] = {}
    summary: Dict[str, object] = {
        "source": str(bundle.desi_catalog),
        "selection_dataset": DESI_DR9_SELECTION_DATASET,
        "weight_dataset": DESI_DR9_WEIGHT_DATASET,
        "nside": int(config.nside),
        "subtract_velocity_weighted_mean": bool(config.subtract_velocity_weighted_mean),
        "velocity_sign": float(config.velocity_sign),
        "velocity_clip_sigma": None if config.velocity_clip_sigma is None else float(config.velocity_clip_sigma),
        "shuffle_velocity_seed": None if config.shuffle_velocity_seed is None else int(config.shuffle_velocity_seed),
        "bins": {},
    }
    with h5py.File(bundle.desi_catalog, "r") as h5:
        summary["file_attrs"] = h5_attrs_to_jsonable(h5.attrs)
        cat = h5["catalog"]
        ra_all = np.asarray(cat["ra_deg"][:], dtype=np.float64)
        dec_all = np.asarray(cat["dec_deg"][:], dtype=np.float64)
        z_all = np.asarray(cat["z"][:], dtype=np.float64)
        vr_all = np.asarray(cat["vr_over_c"][:], dtype=np.float64)
        pz_all = np.asarray(cat["pz_bin"][:], dtype=np.int16)
        valid_for_cl_all = np.asarray(cat["valid_for_cl"][:], dtype=bool)
        weight_all = np.asarray(cat["weight_imaging_mean1"][:], dtype=np.float64)

    base = (
        valid_for_cl_all
        & np.isfinite(ra_all)
        & np.isfinite(dec_all)
        & np.isfinite(z_all)
        & np.isfinite(vr_all)
        & np.isfinite(weight_all)
        & (weight_all > 0.0)
    )
    pix_all = hp.ang2pix(int(config.nside), ra_all[base], dec_all[base], lonlat=True)
    base_indices = np.flatnonzero(base)
    in_mask_base = desi_mask[pix_all] > 0.0
    selected_indices_in_mask = base_indices[in_mask_base]
    del pix_all, base_indices, in_mask_base, base
    gc.collect()

    for pz_bin in config.pz_bins:
        idx = selected_indices_in_mask[pz_all[selected_indices_in_mask] == int(pz_bin)]
        if idx.size == 0:
            raise ValueError(f"No valid DESI rows for pz_bin={pz_bin}.")
        ra = np.asarray(ra_all[idx], dtype=np.float64)
        dec = np.asarray(dec_all[idx], dtype=np.float64)
        weights = np.asarray(weight_all[idx], dtype=np.float64)
        vr = np.asarray(vr_all[idx], dtype=np.float64)
        weighted_mean_vr = float(np.sum(weights * vr) / np.sum(weights))
        weighted_rms_vr = float(np.sqrt(np.sum(weights * vr**2) / np.sum(weights)))
        weighted_std_vr = float(np.sqrt(np.sum(weights * (vr - weighted_mean_vr) ** 2) / np.sum(weights)))
        n_before_velocity_clip = int(vr.size)
        if config.velocity_clip_sigma is not None:
            clip_sigma = float(config.velocity_clip_sigma)
            if not np.isfinite(clip_sigma) or clip_sigma <= 0.0:
                raise ValueError("velocity_clip_sigma must be a positive finite number.")
            keep = np.abs(vr - weighted_mean_vr) <= clip_sigma * weighted_std_vr
            ra = ra[keep]
            dec = dec[keep]
            weights = weights[keep]
            idx = idx[keep]
            vr = vr[keep]
            if vr.size == 0:
                raise ValueError(f"Velocity clipping removed all DESI rows for pz_bin={pz_bin}.")
            weighted_mean_vr = float(np.sum(weights * vr) / np.sum(weights))
            weighted_rms_vr = float(np.sqrt(np.sum(weights * vr**2) / np.sum(weights)))
            weighted_std_vr = float(np.sqrt(np.sum(weights * (vr - weighted_mean_vr) ** 2) / np.sum(weights)))
        if config.shuffle_velocity_seed is not None:
            rng = np.random.default_rng(int(config.shuffle_velocity_seed) + int(pz_bin) * 1009)
            vr = np.asarray(rng.permutation(vr), dtype=np.float64)
        if config.subtract_velocity_weighted_mean:
            vr = vr - weighted_mean_vr
        vr = float(config.velocity_sign) * vr
        out[int(pz_bin)] = {
            "ra_deg": ra,
            "dec_deg": dec,
            "weight": weights,
            "field": vr,
        }
        summary["bins"][f"pz{int(pz_bin)}"] = {
            "n_objects": int(idx.size),
            "n_before_velocity_clip": n_before_velocity_clip,
            "n_removed_velocity_clip": int(n_before_velocity_clip - idx.size),
            "sum_weight": float(np.sum(weights)),
            "sum_weight_sq": float(np.sum(weights**2)),
            "z_weighted_mean": float(np.sum(weights * z_all[idx]) / np.sum(weights)),
            "z_min": float(np.min(z_all[idx])),
            "z_max": float(np.max(z_all[idx])),
            "vr_over_c_weighted_mean_before_optional_subtraction": weighted_mean_vr,
            "vr_over_c_weighted_std": weighted_std_vr,
            "vr_over_c_weighted_rms": weighted_rms_vr,
            "velocity_sign_applied": float(config.velocity_sign),
            "velocity_shuffled": config.shuffle_velocity_seed is not None,
        }

    del ra_all, dec_all, z_all, vr_all, pz_all, valid_for_cl_all, weight_all, selected_indices_in_mask
    gc.collect()
    return out, summary


def make_catalog_fields(
    catalogs: Mapping[int, Mapping[str, np.ndarray]],
    desi_mask: np.ndarray,
    config: HighEllKszConfig,
) -> Tuple[Dict[int, nmt.NmtField], nmt.NmtField]:
    f_gmask = nmt.NmtField(
        desi_mask,
        None,
        spin=0,
        lmax=int(config.lmax),
        lmax_mask=int(config.lmax),
        n_iter=0,
        n_iter_mask=0,
        lite=True,
    )
    fields: Dict[int, nmt.NmtField] = {}
    for pz_bin, cat in catalogs.items():
        fields[int(pz_bin)] = nmt.NmtFieldCatalogMomentum(
            np.asarray([cat["ra_deg"], cat["dec_deg"]], dtype=np.float64),
            np.asarray(cat["weight"], dtype=np.float64),
            np.asarray(cat["field"], dtype=np.float64),
            None,
            None,
            lmax=int(config.lmax),
            lmax_mask=int(config.lmax),
            spin=0,
            field_is_weighted=False,
            lonlat=True,
            mask=np.asarray(desi_mask, dtype=np.float64),
            n_iter_mask=0,
        )
    return fields, f_gmask


def make_temperature_field(t_hp: np.ndarray, t_mask: np.ndarray, config: HighEllKszConfig) -> nmt.NmtField:
    return nmt.NmtField(
        t_mask,
        [t_hp],
        spin=0,
        lmax=int(config.lmax),
        lmax_mask=int(config.lmax),
        n_iter=0,
        n_iter_mask=0,
        lite=True,
        masked_on_input=bool(config.masked_on_input_temperature),
    )


def catalog_nf(field: nmt.NmtField, lmax: int) -> np.ndarray:
    nf = getattr(field, "Nf", None)
    if nf is None:
        return np.zeros((1, int(lmax) + 1), dtype=np.float64)
    arr = np.asarray(nf, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((1, int(lmax) + 1), float(arr), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < int(lmax) + 1:
        padded = np.zeros((arr.shape[0], int(lmax) + 1), dtype=np.float64)
        padded[:, : arr.shape[-1]] = arr
        arr = padded
    return arr[:, : int(lmax) + 1]


def workspace_cache_dir(config: HighEllKszConfig) -> Path:
    if config.workspace_cache_dir:
        return Path(config.workspace_cache_dir)
    return Path(config.output).parent / "workspaces"


def workspace_cache_paths(config: HighEllKszConfig) -> Tuple[Path, Path]:
    root = workspace_cache_dir(config)
    tag = (
        f"nside{int(config.nside)}_lmax{int(config.lmax)}_"
        f"ell{int(config.ell_min)}_nbin{int(config.n_bins)}_"
        f"exact_ltplz{int(config.covariance_l_toeplitz)}_"
        f"lex{int(config.covariance_l_exact)}_dl{int(config.covariance_dl_band)}"
    )
    return root / f"ksz_pT_workspace_{tag}.fits", root / f"ksz_cov_workspace_{tag}.fits"


def load_or_build_workspace(
    cache_path: Path,
    builder,
    label: str,
):
    if cache_path.exists():
        print(f"[ksz-high] Loading cached exact {label}: {cache_path}", flush=True)
        if label == "pi x T workspace":
            return nmt.NmtWorkspace.from_file(str(cache_path))
        return nmt.NmtCovarianceWorkspace.from_file(str(cache_path))
    obj = builder()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    obj.write_to(str(cache_path))
    print(f"[ksz-high] Wrote exact {label}: {cache_path}", flush=True)
    return obj


def compute_measurement(config: HighEllKszConfig) -> Dict[str, object]:
    bundle = SurveyBundle.from_root(config.survey_root)
    left, right = make_paper_like_log_edges(config.ell_min, config.lmax, config.n_bins)
    bins = nmt.NmtBin.from_edges(left, right)
    ell = np.asarray(bins.get_effective_ells(), dtype=np.float64)

    print(f"[ksz-high] Loading DESI nside={config.nside} random mask", flush=True)
    desi_mask, desi_mask_meta = load_desi_mask(bundle, config)
    print(f"[ksz-high] Loading ACT T map and analysis mask at nside={config.nside}", flush=True)
    t_hp, t_mask, t_meta = load_act_temperature_healpix(bundle, config)
    print("[ksz-high] Loading DESI DR9 weighted catalog rows", flush=True)
    catalogs, catalog_meta = load_desi_catalog_by_pz(bundle, desi_mask, config)

    fsky_lrg = float(np.mean(desi_mask.astype(np.float64) ** 2))
    fsky_t = float(np.mean(t_mask.astype(np.float64) ** 2))
    fsky_comb = float(np.mean(desi_mask.astype(np.float64) * t_mask.astype(np.float64)))
    if min(fsky_lrg, fsky_t, fsky_comb) <= 0.0:
        raise ValueError(f"Invalid fsky values: {fsky_lrg}, {fsky_t}, {fsky_comb}")

    print("[ksz-high] Building NaMaster catalog-momentum and ACT T fields", flush=True)
    pi_fields, f_gmask = make_catalog_fields(catalogs, desi_mask, config)
    f_tmap = make_temperature_field(t_hp, t_mask, config)

    pz_bins = tuple(int(p) for p in config.pz_bins)
    workspace_path, cov_workspace_path = workspace_cache_paths(config)
    print("[ksz-high] Computing/loading one shared pi x T workspace", flush=True)
    wsp_pT = load_or_build_workspace(
        workspace_path,
        lambda: _workspace_from_fields(pi_fields[pz_bins[0]], f_tmap, bins, config),
        "pi x T workspace",
    )
    print("[ksz-high] Computing/loading one shared kSZ covariance workspace", flush=True)
    cw = load_or_build_workspace(
        cov_workspace_path,
        lambda: _covariance_workspace_from_fields(f_gmask, f_tmap, f_gmask, f_tmap, config),
        "kSZ covariance workspace",
    )

    print("[ksz-high] Computing pseudo spectra and decoupled C_ell", flush=True)
    pcl_TT = nmt.compute_coupled_cell(f_tmap, f_tmap)
    pcls_pT: Dict[int, np.ndarray] = {}
    pcls_Tp: Dict[int, np.ndarray] = {}
    spectra: Dict[str, Dict[str, object]] = {}
    for pz_bin in pz_bins:
        name = f"desi_pi_act_T_pz{pz_bin}"
        pcl = nmt.compute_coupled_cell(pi_fields[pz_bin], f_tmap)
        pcls_pT[pz_bin] = pcl
        pcls_Tp[pz_bin] = nmt.compute_coupled_cell(f_tmap, pi_fields[pz_bin])
        cl = np.asarray(wsp_pT.decouple_cell(pcl)[0], dtype=np.float64)
        factor = dl_factor(ell)
        dl_raw = factor * cl
        dl_plot = float(config.plot_sign) * dl_raw
        spectra[name] = {
            "name": name,
            "pz_bin": int(pz_bin),
            "ell": ell,
            "cl": cl,
            "dl": dl_raw,
            "dl_raw_piT": dl_raw,
            "dl_paper_ksz": -dl_raw,
            "dl_plot": dl_plot,
            "pcl": np.asarray(pcl[0], dtype=np.float64),
        }

    print("[ksz-high] Computing Gaussian covariance blocks", flush=True)
    n_bands = int(bins.get_n_bands())
    n_spec = len(pz_bins)
    cov = np.zeros((n_spec * n_bands, n_spec * n_bands), dtype=np.float64)
    covariance_blocks: Dict[Tuple[str, str], np.ndarray] = {}
    input_cls: Dict[str, np.ndarray] = {
        "T__x__T": np.asarray(pcl_TT[0] / fsky_t, dtype=np.float64),
    }
    for i, pz_i in enumerate(pz_bins):
        name_i = f"desi_pi_act_T_pz{pz_i}"
        input_cls[f"pi{pz_i}__x__T"] = np.asarray(pcls_pT[pz_i][0] / fsky_comb, dtype=np.float64)
        input_cls[f"T__x__pi{pz_i}"] = np.asarray(pcls_Tp[pz_i][0] / fsky_comb, dtype=np.float64)
        for j, pz_j in enumerate(pz_bins):
            if j < i:
                continue
            name_j = f"desi_pi_act_T_pz{pz_j}"
            pcl_pp = nmt.compute_coupled_cell(pi_fields[pz_i], pi_fields[pz_j])
            if pz_i == pz_j:
                pcl_pp = np.asarray(pcl_pp, dtype=np.float64) + catalog_nf(pi_fields[pz_i], config.lmax)
            input_cls[f"pi{pz_i}__x__pi{pz_j}"] = np.asarray(pcl_pp[0] / fsky_lrg, dtype=np.float64)
            block_raw = nmt.gaussian_covariance(
                cw,
                0,
                0,
                0,
                0,
                pcl_pp / fsky_lrg,
                pcls_pT[pz_i] / fsky_comb,
                pcls_Tp[pz_j] / fsky_comb,
                pcl_TT / fsky_t,
                wsp_pT,
                wb=wsp_pT,
                coupled=False,
            )
            block = _select_covariance_component_block(block_raw, n_bands, 1, 1, 0, 0)
            if i == j:
                block = 0.5 * (block + block.T)
            covariance_blocks[(name_i, name_j)] = np.asarray(block, dtype=np.float64)
            row = slice(i * n_bands, (i + 1) * n_bands)
            col = slice(j * n_bands, (j + 1) * n_bands)
            cov[row, col] = block
            if i != j:
                cov[col, row] = block.T

    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    err_cl_by_spec: Dict[str, np.ndarray] = {}
    data_cl = []
    data_dl = []
    for i, pz_bin in enumerate(pz_bins):
        name = f"desi_pi_act_T_pz{pz_bin}"
        sl = slice(i * n_bands, (i + 1) * n_bands)
        err_cl = np.sqrt(np.where(np.diag(cov[sl, sl]) > 0, np.diag(cov[sl, sl]), np.nan))
        err_dl = dl_factor(ell) * err_cl
        spectra[name]["err_cl"] = err_cl
        spectra[name]["err_dl"] = err_dl
        spectra[name]["err_dl_raw_piT"] = err_dl
        spectra[name]["err_dl_paper_ksz"] = err_dl
        spectra[name]["err_dl_plot"] = abs(float(config.plot_sign)) * err_dl
        err_cl_by_spec[name] = err_cl
        data_cl.append(np.asarray(spectra[name]["cl"], dtype=np.float64))
        data_dl.append(np.asarray(spectra[name]["dl"], dtype=np.float64))

    data_cl_vec = np.concatenate(data_cl)
    data_dl_vec = np.concatenate(data_dl)
    fit_band = (ell >= float(config.fit_ell_min)) & (ell <= float(config.fit_ell_max))
    high_plot_band = (ell >= float(config.plot_ell_min)) & (ell <= float(config.plot_ell_max))
    fit_idx = np.concatenate([np.flatnonzero(fit_band) + i * n_bands for i in range(n_spec)])
    high_idx = np.concatenate([np.flatnonzero(high_plot_band) + i * n_bands for i in range(n_spec)])
    per_bin_snr: Dict[str, object] = {}
    for i, pz_bin in enumerate(pz_bins):
        name = f"desi_pi_act_T_pz{pz_bin}"
        sl = slice(i * n_bands, (i + 1) * n_bands)
        fit_local = np.flatnonzero(fit_band)
        high_local = np.flatnonzero(high_plot_band)
        per_bin_snr[name] = {
            "all_bands": safe_snr(data_cl_vec[sl], cov[sl, sl]),
            "fit_range": safe_snr(data_cl_vec[sl][fit_local], cov[sl, sl][np.ix_(fit_local, fit_local)]),
            "plot_range": safe_snr(data_cl_vec[sl][high_local], cov[sl, sl][np.ix_(high_local, high_local)]),
        }

    result = {
        "config": asdict(config),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": SCHEMA,
        "namaster_version": getattr(nmt, "__version__", "unknown"),
        "bundle_root": str(bundle.root),
        "input_files": {
            "desi_catalog": str(bundle.desi_catalog),
            "desi_random_count_maps": str(bundle.desi_random_count_maps),
            "act_cmb": str(bundle.act_cmb),
        },
        "ell": ell,
        "ell_left": left,
        "ell_right": right,
        "binning": {
            "type": "paper_like_log17",
            "paper_reference": "notebooks/xDESI/papers/ksz/2604.19744v1.pdf",
            "note": "17 logarithmic bins over ell_min..lmax; paper fits 1000<=ell<=7000.",
        },
        "sign_conventions": {
            "raw_measured_quantity": "C_ell^{pi,T_uK}; pi uses +vr_over_c from the supplied DESI catalog.",
            "paper_theory_relation": "C_ell^{pi,T} = - r sigma_true sigma_rec C_ell^{tau,g}.",
            "paper_plot_quantity": "D_ell^kSZ = -ell(ell+1) C_ell^{pi,T} / (2*pi).",
            "plot_sign_applied_to_raw_Dell": float(config.plot_sign),
        },
        "workspace_cache": {
            "pi_x_T_workspace": str(workspace_path),
            "covariance_workspace": str(cov_workspace_path),
            "exact": int(config.covariance_l_toeplitz) <= 0,
        },
        "fsky": {
            "desi_random_mask_squared_mean": fsky_lrg,
            "act_temperature_mask_squared_mean": fsky_t,
            "desi_x_act_mask_mean_product": fsky_comb,
        },
        "desi_mask": desi_mask_meta,
        "act_temperature": t_meta,
        "desi_catalog": catalog_meta,
        "spectra": spectra,
        "covariance": cov,
        "covariance_diagnostics": covariance_diagnostics(cov, compute_eig=True),
        "data_vector_cl": data_cl_vec,
        "data_vector_dl": data_dl_vec,
        "snr": {
            "all_bands": safe_snr(data_cl_vec, cov),
            "fit_range": safe_snr(data_cl_vec[fit_idx], cov[np.ix_(fit_idx, fit_idx)]),
            "plot_range": safe_snr(data_cl_vec[high_idx], cov[np.ix_(high_idx, high_idx)]),
            "per_bin": per_bin_snr,
            "fit_ell_min": float(config.fit_ell_min),
            "fit_ell_max": float(config.fit_ell_max),
            "plot_ell_min": float(config.plot_ell_min),
            "plot_ell_max": float(config.plot_ell_max),
        },
        "input_cls_for_covariance": input_cls,
        "covariance_blocks": covariance_blocks,
    }
    return result


def write_result(result: Mapping[str, object], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    ell = np.asarray(result["ell"], dtype=np.float64)
    cov = np.asarray(result["covariance"], dtype=np.float64)
    diag = np.diag(cov)
    denom = np.sqrt(np.outer(np.where(diag > 0, diag, np.nan), np.where(diag > 0, diag, np.nan)))
    corr = cov / denom
    with h5py.File(tmp, "w") as h5:
        h5.attrs["schema"] = SCHEMA
        h5.attrs["metadata_json"] = json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                not in {
                    "ell",
                    "ell_left",
                    "ell_right",
                    "spectra",
                    "covariance",
                    "data_vector_cl",
                    "data_vector_dl",
                    "input_cls_for_covariance",
                    "covariance_blocks",
                }
            },
            indent=2,
            default=_json_default,
        )
        h5.create_dataset("ell", data=ell)
        h5.create_dataset("ell_left", data=np.asarray(result["ell_left"], dtype=np.int32))
        h5.create_dataset("ell_right", data=np.asarray(result["ell_right"], dtype=np.int32))
        h5.create_dataset("data_vector_cl", data=np.asarray(result["data_vector_cl"], dtype=np.float64))
        h5.create_dataset("data_vector_dl", data=np.asarray(result["data_vector_dl"], dtype=np.float64))
        h5.create_dataset("covariance", data=cov)
        h5.create_dataset("correlation", data=corr)
        sg = h5.create_group("spectra")
        for name, spec in result["spectra"].items():
            g = sg.create_group(name)
            g.attrs["pz_bin"] = int(spec["pz_bin"])
            for dataset in (
                "cl",
                "dl",
                "dl_raw_piT",
                "dl_paper_ksz",
                "dl_plot",
                "err_cl",
                "err_dl",
                "err_dl_raw_piT",
                "err_dl_paper_ksz",
                "err_dl_plot",
                "pcl",
            ):
                g.create_dataset(dataset, data=np.asarray(spec[dataset], dtype=np.float64))
        ig = h5.create_group("input_cls_for_covariance")
        ig.attrs["mode"] = "ksz_tutorial_pseudo_cl_over_fsky"
        for name, values in result["input_cls_for_covariance"].items():
            ig.create_dataset(name, data=np.asarray(values, dtype=np.float64))
        bg = h5.create_group("covariance_blocks")
        for (name_i, name_j), block in result["covariance_blocks"].items():
            bg.create_dataset(f"{name_i}__x__{name_j}", data=np.asarray(block, dtype=np.float64))
    os.replace(tmp, path)
    return path


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_summary(result: Mapping[str, object], output: str | Path, product: Path, plot: Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_utc": result["created_utc"],
        "product": str(product),
        "plot": str(plot),
        "namaster_version": result["namaster_version"],
        "config": result["config"],
        "binning": result["binning"],
        "fsky": result["fsky"],
        "snr": result["snr"],
        "covariance_diagnostics": result["covariance_diagnostics"],
    }
    path.write_text(json.dumps(summary, indent=2, default=_json_default))
    return path


def plot_result(result: Mapping[str, object], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    ell = np.asarray(result["ell"], dtype=np.float64)
    spectra = result["spectra"]
    snr = result["snr"]["per_bin"]
    plot_min = float(result["config"]["plot_ell_min"])
    plot_max = float(result["config"]["plot_ell_max"])
    fit_min = float(result["config"]["fit_ell_min"])
    fit_max = float(result["config"]["fit_ell_max"])
    plot_sign = float(result["config"].get("plot_sign", -1.0))
    if plot_sign == -1.0:
        ylabel = r"$D_\ell^{\rm kSZ}=-\ell(\ell+1)C_\ell^{\pi T}/2\pi$ [$\mu$K]"
        title_prefix = r"paper-sign $D_\ell^{\rm kSZ}$"
    elif plot_sign == 1.0:
        ylabel = r"raw $D_\ell^{\pi T}=\ell(\ell+1)C_\ell^{\pi T}/2\pi$ [$\mu$K]"
        title_prefix = r"raw $D_\ell^{\pi T}$"
    else:
        ylabel = rf"{plot_sign:g} $\times\,\ell(\ell+1)C_\ell^{{\pi T}}/2\pi$ [$\mu$K]"
        title_prefix = rf"plot sign {plot_sign:g}"

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    axes = axes.ravel()
    for ax, name in zip(axes, spectra):
        spec = spectra[name]
        y = np.asarray(spec.get("dl_plot", spec["dl"]), dtype=np.float64)
        err = np.asarray(spec.get("err_dl_plot", spec["err_dl"]), dtype=np.float64)
        ax.axhline(0.0, color="0.25", lw=0.9)
        ax.axvspan(0.0, fit_min, color="#d95f5f", alpha=0.10, lw=0)
        ax.axvspan(fit_max, float(result["config"]["lmax"]), color="#d95f5f", alpha=0.10, lw=0)
        ax.errorbar(
            ell,
            y,
            yerr=err,
            fmt="o",
            ms=4.0,
            lw=1.1,
            capsize=2.5,
            color="#b2182b",
            ecolor="#6b6b6b",
            label=rf"pz {spec['pz_bin']}",
        )
        focus = (ell >= plot_min) & (ell <= plot_max) & np.isfinite(y) & np.isfinite(err)
        if np.any(focus):
            yy = np.concatenate([y[focus] - err[focus], y[focus] + err[focus], y[focus]])
            finite = yy[np.isfinite(yy)]
            if finite.size:
                ymin = float(np.min(finite))
                ymax = float(np.max(finite))
                pad = 0.12 * max(ymax - ymin, np.max(np.abs(finite)), 1.0e-20)
                ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlim(max(0.0, plot_min - 200.0), min(float(result["config"]["lmax"]), plot_max + 200.0))
        fit_snr = snr[name]["fit_range"]["snr"]
        all_snr = snr[name]["all_bands"]["snr"]
        ax.text(
            0.03,
            0.94,
            rf"pz {spec['pz_bin']}  S/N$_{{1000-7000}}$={fit_snr:.2f}  all={all_snr:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(alpha=0.18, lw=0.6)
    for ax in axes[::2]:
        ax.set_ylabel(ylabel)
    for ax in axes[-2:]:
        ax.set_xlabel(r"Multipole $\ell$")
    joint = result["snr"]["fit_range"]["snr"]
    fig.suptitle(
        rf"DESI DR9 weighted catalog momentum $\times$ ACT DR6 T: {title_prefix}, "
        rf"$\ell_\max={int(result['config']['lmax'])}$, joint S/N$_{{1000-7000}}$={joint:.2f}",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> HighEllKszConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", default=HighEllKszConfig.survey_root)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--plot-output", default=str(DEFAULT_PLOT))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--workspace-cache-dir", default=None)
    parser.add_argument("--nside", type=int, default=4096)
    parser.add_argument("--lmax", type=int, default=8192)
    parser.add_argument("--ell-min", type=int, default=300)
    parser.add_argument("--n-bins", type=int, default=17)
    parser.add_argument("--pz-bins", default="1,2,3,4")
    parser.add_argument("--act-downgrade", type=int, default=1)
    parser.add_argument("--no-act-mean-subtraction", action="store_true")
    parser.add_argument("--subtract-velocity-weighted-mean", action="store_true")
    parser.add_argument("--velocity-sign", type=float, default=1.0)
    parser.add_argument("--velocity-clip-sigma", type=float, default=None)
    parser.add_argument("--shuffle-velocity-seed", type=int, default=None)
    parser.add_argument("--plot-sign", type=float, default=-1.0)
    parser.add_argument("--fit-ell-min", type=float, default=1000.0)
    parser.add_argument("--fit-ell-max", type=float, default=7000.0)
    parser.add_argument("--plot-ell-min", type=float, default=1000.0)
    parser.add_argument("--plot-ell-max", type=float, default=8000.0)
    parser.add_argument("--covariance-l-toeplitz", type=int, default=-1)
    parser.add_argument("--covariance-l-exact", type=int, default=-1)
    parser.add_argument("--covariance-dl-band", type=int, default=-1)
    parser.add_argument("--masked-on-input-temperature", action="store_true")
    args = parser.parse_args(argv)
    pz_bins = tuple(int(x) for x in str(args.pz_bins).split(",") if str(x).strip())
    return HighEllKszConfig(
        survey_root=str(args.survey_root),
        output=str(args.output),
        plot_output=str(args.plot_output),
        summary_output=str(args.summary_output),
        workspace_cache_dir=None if args.workspace_cache_dir is None else str(args.workspace_cache_dir),
        nside=int(args.nside),
        lmax=int(args.lmax),
        ell_min=int(args.ell_min),
        n_bins=int(args.n_bins),
        pz_bins=pz_bins,
        act_downgrade=int(args.act_downgrade),
        subtract_act_masked_mean=not bool(args.no_act_mean_subtraction),
        subtract_velocity_weighted_mean=bool(args.subtract_velocity_weighted_mean),
        velocity_sign=float(args.velocity_sign),
        velocity_clip_sigma=None if args.velocity_clip_sigma is None else float(args.velocity_clip_sigma),
        shuffle_velocity_seed=None if args.shuffle_velocity_seed is None else int(args.shuffle_velocity_seed),
        plot_sign=float(args.plot_sign),
        fit_ell_min=float(args.fit_ell_min),
        fit_ell_max=float(args.fit_ell_max),
        plot_ell_min=float(args.plot_ell_min),
        plot_ell_max=float(args.plot_ell_max),
        covariance_l_toeplitz=int(args.covariance_l_toeplitz),
        covariance_l_exact=int(args.covariance_l_exact),
        covariance_dl_band=int(args.covariance_dl_band),
        masked_on_input_temperature=bool(args.masked_on_input_temperature),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    result = compute_measurement(config)
    product = write_result(result, config.output)
    plot = plot_result(result, config.plot_output)
    summary = write_summary(result, config.summary_output, product, plot)
    print(f"[ksz-high] Wrote {product}", flush=True)
    print(f"[ksz-high] Wrote {plot}", flush=True)
    print(f"[ksz-high] Wrote {summary}", flush=True)
    print(json.dumps(result["snr"], indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
