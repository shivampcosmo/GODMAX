"""xDESI helpers for Abacus halo map pasting and quick validation."""

from __future__ import annotations

import copy
import gc
import importlib
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count, get_context
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import healpy as hp
import numpy as np
import yaml
from astropy.io import fits
from astropy import constants as astro_const
from scipy.interpolate import interp1d
try:
    from numba import get_num_threads, njit, prange, set_num_threads
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - only used on environments without numba
    _NUMBA_AVAILABLE = False
    prange = range
    get_num_threads = None
    set_num_threads = None

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def _decorator(func):
            return func

        return _decorator

from abacus_lightcone_catalog import ensure_under_xdesi, load_config, xdesi_dir


GODMAX_ROOT = xdesi_dir().parents[1]
SRC_DIR = GODMAX_ROOT / "src"
PASTING_DIR = GODMAX_ROOT / "notebooks" / "pasting"
SURVEY_MEASURE_DIR = GODMAX_ROOT / "notebooks" / "xDESI" / "survey_measure"
for _path in (SRC_DIR, PASTING_DIR, SURVEY_MEASURE_DIR, GODMAX_ROOT / "data", GODMAX_ROOT / "param_files"):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))


MAP_DATASETS = (
    "map_rhom_dmb",
    "map_rhom_dmo",
    "map_rhom",
    "map_ymap",
    "map_ksz",
    "map_tau",
    "map_kappa_cmb",
    "map_kappa_wl",
    "map_kappa_wl_tomo2",
    "map_kappa_wl_tomo3",
    "map_kappa_wl_tomo4",
)

_PIXEL_STATE: Dict[str, object] = {}
_PIXEL_BACKEND_HEALPY = "healpy"
_PIXEL_BACKEND_HEALPY_BUFF = "healpy_buff"
_PIXEL_BACKEND_HEALPY_RING = "healpy_ring"
_PIXEL_BACKEND_HEALPY_STENCIL = "healpy_stencil"
_PIXEL_BACKENDS = {
    _PIXEL_BACKEND_HEALPY,
    _PIXEL_BACKEND_HEALPY_BUFF,
    _PIXEL_BACKEND_HEALPY_RING,
    _PIXEL_BACKEND_HEALPY_STENCIL,
}
_RING_GEOM_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
_EFFECTIVE_GRID_SIGNIFICANT_DIGITS = 13


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _canonicalize_effective_grid(values: np.ndarray) -> np.ndarray:
    """Suppress the observed CPU-math-library drift in analytic setup grids.

    ``np.exp`` and ``np.geomspace`` may differ by a few final float64 bits on
    otherwise identical nodes.  Those differences do not resolve a grid cell,
    but hashing the raw arrays makes the map contract architecture-dependent.
    Quantizing only these analytically generated setup grids to 13 significant
    decimal digits keeps their relative perturbation below 5e-13 while making
    both the values used by GODMAX and their provenance byte-stable on the
    tested Rome and H100-host architectures.  A cross-architecture manifest
    comparison remains the falsifier for any new platform.
    """

    array = np.asarray(values, dtype=np.float64)
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("Effective GODMAX setup grids must be finite.")
    canonical = np.fromiter(
        (
            float(format(float(value), f".{_EFFECTIVE_GRID_SIGNIFICANT_DIGITS}g"))
            for value in array.ravel()
        ),
        dtype=np.float64,
        count=array.size,
    )
    return canonical.reshape(array.shape)


def effective_grid_canonicalization_contract() -> Dict[str, object]:
    """Describe the deliberately narrow generated-grid canonicalization."""

    return {
        "method": "decimal_significant_digits",
        "significant_digits": _EFFECTIVE_GRID_SIGNIFICANT_DIGITS,
        "affected_effective_config_paths": [
            "analysis.k_array_survey",
            "analysis.l_array_survey",
            "analysis.dl_array_survey",
            "halo_params.ell_array",
        ],
        "applies_to_catalog_or_profile_values": False,
    }


def _effective_survey_grids(h0: float, nside: int) -> Tuple[np.ndarray, ...]:
    """Construct the final canonicalized k, ell, and delta-ell grids."""

    hubble = float(h0)
    if not np.isfinite(hubble) or hubble <= 0.0:
        raise ValueError(f"H0 must be finite and positive, got {h0!r}.")
    resolution = int(nside)
    if resolution <= 0:
        raise ValueError(f"NSIDE must be positive, got {nside!r}.")

    k_survey = _canonicalize_effective_grid(
        np.geomspace(3.0e-1, 10.0, 10) / (hubble / 100.0)
    )
    lmin, lmax, dl_log = 20.0, 3.0 * resolution, 0.08
    ell_edges = np.exp(
        np.arange(np.log(lmin), np.log(max(lmin + 1.0, lmax)), dl_log)
    )
    ell = _canonicalize_effective_grid(
        0.5 * (ell_edges[1:] + ell_edges[:-1])
    )
    delta_ell = _canonicalize_effective_grid(ell_edges[1:] - ell_edges[:-1])
    return k_survey, ell, delta_ell


def auto_cpu_workers() -> int:
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(key)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return max(1, cpu_count())


def read_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: dict, override: Mapping[str, object]) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def generate_dicts(data: Mapping[str, object]):
    return (
        copy.deepcopy(data.get("sim_params", {})),
        copy.deepcopy(data.get("halo_params", {})),
        copy.deepcopy(data.get("analysis", {})),
        copy.deepcopy(data.get("other_params", {})),
    )


def load_halo_catalog(catalog_path: Path | str, indices: Optional[np.ndarray] = None) -> Tuple[dict, dict]:
    catalog_path = Path(catalog_path)
    with h5py.File(catalog_path, "r") as handle:
        attrs = dict(handle.attrs)
        fields = {}
        names = [
            "ra_deg",
            "dec_deg",
            "z",
            "M200c_hMsun",
            "log10M200c_hMsun",
            "vlos_kms",
            "R200c_hMpc",
            "DA_hMpc",
        ]
        if indices is None:
            for name in names:
                fields[name] = handle[name][:]
        else:
            idx = np.asarray(indices, dtype=np.int64)
            idx.sort()
            for name in names:
                fields[name] = handle[name][idx]
    return fields, attrs


def halo_catalog_size(catalog_path: Path | str) -> Tuple[int, dict]:
    catalog_path = Path(catalog_path)
    with h5py.File(catalog_path, "r") as handle:
        attrs = dict(handle.attrs)
        return int(handle["z"].shape[0]), attrs


def load_halo_catalog_slice(catalog_path: Path | str, start: int, stop: int) -> Tuple[dict, dict]:
    catalog_path = Path(catalog_path)
    start = max(0, int(start))
    stop = max(start, int(stop))
    with h5py.File(catalog_path, "r") as handle:
        attrs = dict(handle.attrs)
        n_total = int(handle["z"].shape[0])
        stop = min(stop, n_total)
        fields = {}
        names = [
            "ra_deg",
            "dec_deg",
            "z",
            "M200c_hMsun",
            "log10M200c_hMsun",
            "vlos_kms",
            "R200c_hMpc",
            "DA_hMpc",
        ]
        for name in names:
            fields[name] = handle[name][start:stop]
    return fields, attrs


def load_halo_catalog_ranges(catalog_path: Path | str, ranges: Sequence[Tuple[int, int]]) -> Tuple[dict, dict]:
    catalog_path = Path(catalog_path)
    clean_ranges = [(max(0, int(start)), max(0, int(stop))) for start, stop in ranges if int(stop) > int(start)]
    with h5py.File(catalog_path, "r") as handle:
        attrs = dict(handle.attrs)
        n_total = int(handle["z"].shape[0])
        names = [
            "ra_deg",
            "dec_deg",
            "z",
            "M200c_hMsun",
            "log10M200c_hMsun",
            "vlos_kms",
            "R200c_hMpc",
            "DA_hMpc",
        ]
        fields = {}
        for name in names:
            pieces = [handle[name][start:min(stop, n_total)] for start, stop in clean_ranges if start < n_total]
            if pieces:
                fields[name] = np.concatenate(pieces)
            else:
                fields[name] = handle[name][0:0]
    return fields, attrs


def catalog_path(config: Mapping[str, object], catalog_key: str) -> Path:
    spec = config["catalogs"][catalog_key]
    catalog_subdir = str(config["project"].get("catalog_subdir", "abacus_halos"))
    path = Path(config["project"]["output_root"]).expanduser().resolve() / catalog_subdir / spec["output_name"]
    ensure_under_xdesi(path)
    return path


def map_run_dir(config: Mapping[str, object], run_name: Optional[str] = None) -> Path:
    name = run_name or str(config["pasting"].get("run_name", "abacus_xdesi"))
    map_subdir = str(config["project"].get("map_subdir", "abacus_maps"))
    path = Path(config["project"]["output_root"]).expanduser().resolve() / map_subdir / name
    ensure_under_xdesi(path)
    return path


def partial_map_path(config: Mapping[str, object], catalog_key: str, nside: int, split_index: int, num_splits: int) -> Path:
    return map_run_dir(config) / f"abacus_pasted_maps_{catalog_key}_nside{nside}_split{split_index:03d}of{num_splits:03d}.h5"


def final_map_path(config: Mapping[str, object], catalog_key: str, nside: int) -> Path:
    return map_run_dir(config) / f"abacus_pasted_maps_{catalog_key}_nside{nside}.h5"


def load_source_nz(config: Mapping[str, object]) -> dict:
    gcfg = config["godmax"]
    with fits.open(gcfg["source_nz_fits"]) as hdul:
        data = hdul[gcfg["source_nz_hdu"]].data
        z = np.asarray(data[gcfg["source_nz_z_column"]], dtype=np.float64)
        nz = np.asarray(data[gcfg["source_nz_bin_column"]], dtype=np.float64)
    nz = np.maximum(nz, float(gcfg.get("source_nz_floor", 0.0)))
    return {"z_array_source": z, "nbins": 1, "nz0": nz}


def load_xdesi_fit_lens_info(config: Mapping[str, object]) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Reproduce the xDESI Abacus lens n(z)/nbar setup used by test_fit_abacus."""

    fit_path = Path(config["godmax"]["xdesi_fit_summary"])
    with open(fit_path, "rb") as handle:
        fit = pickle.load(handle)

    zarray_lens = np.linspace(0.001, 1.6, 200)
    zvals = fit["zvals"]
    nz_lens = {}
    nbar_interp = interp1d(
        fit["zcens_comoving"],
        fit["nbar_comoving"],
        fill_value=1.0e-8,
        bounds_error=False,
    )
    nbar_array = nbar_interp(zarray_lens)

    for jz, z_group in enumerate(zvals):
        key = f"z{z_group[0]:.3f}_{z_group[-1]:.3f}"
        z_file = fit["nz_gal_all"]["z_array"]
        nz_file = fit["nz_gal_all"][key]
        interp = interp1d(z_file, nz_file, fill_value=0.0, bounds_error=False)
        hist_z = interp(zarray_lens)
        norm = np.trapezoid(hist_z, zarray_lens) if hasattr(np, "trapezoid") else np.trapz(hist_z, zarray_lens)
        nz_lens[jz] = hist_z / norm if norm > 0 else hist_z

    nz_info = {"z_array_lens": zarray_lens, "nbins_lens": len(zvals)}
    z_edges = []
    for jz, z_group in enumerate(zvals):
        nz = np.maximum(nz_lens[jz], 1.0e-3)
        nz_info[f"nz{jz}"] = nz
        support = zarray_lens[np.where(nz > 0.7)[0]]
        if len(support) > 1:
            z_edges.append([support[0], support[-1]])
        else:
            z_edges.append([float(z_group[0]), float(z_group[-1])])
    nz_info["z_edges_bins_lens"] = np.asarray(z_edges)
    return nz_info, zarray_lens, nbar_array


def _catalog_cosmology(attrs: Mapping[str, object]) -> dict:
    return {
        "flat": True,
        "H0": float(attrs.get("H0", 67.11)),
        "Om0": float(attrs.get("Omega_M", 0.3175)),
        "Ob0": float(attrs.get("Omega_b", 0.049)),
        "sigma8": float(attrs.get("sigma8", 0.834)),
        "ns": float(attrs.get("ns", 0.9624)),
        "w0": float(attrs.get("w0", -1.0)),
    }


def prepare_godmax_config(
    config: Mapping[str, object],
    catalog_attrs: Optional[Mapping[str, object]] = None,
    *,
    is_cmb_lensing: bool = False,
    z_max: Optional[float] = None,
    log10_mass_min: Optional[float] = None,
):
    if "comparison_config" in config.get("godmax", {}):
        return prepare_stage31_godmax_config(
            config,
            catalog_attrs,
            is_cmb_lensing=is_cmb_lensing,
            z_max=z_max,
            log10_mass_min=log10_mass_min,
        )

    default_data = read_yaml(config["godmax"]["default_params"])
    xdesi_data = read_yaml(config["godmax"]["xdesi_params"])
    merged = deep_update(default_data, xdesi_data)
    sim_params, halo_params, analysis, other_params = generate_dicts(merged)
    import jax.numpy as jnp

    if catalog_attrs and bool(config["godmax"].get("override_cosmology_from_catalog", True)):
        sim_params["cosmo"] = _catalog_cosmology(catalog_attrs)

    nz_lens_info, zarray_lens, nbar_array = load_xdesi_fit_lens_info(config)
    analysis["nz_lens_info_dict"] = nz_lens_info
    analysis["nbar_gal_comoving_zarray"] = zarray_lens
    analysis["nbar_gal_comoving_val"] = nbar_array

    source_nz = load_source_nz(config)
    analysis["nz_source_info_dict"] = source_nz
    analysis["is_cmb_lensing"] = bool(is_cmb_lensing)
    other_params["Delta_z_bias_array"] = np.zeros(source_nz["nbins"])
    other_params["mult_shear_bias_array"] = np.zeros(source_nz["nbins"])

    cp = sim_params["cosmo"]
    k_survey, ell, dell = _effective_survey_grids(
        cp["H0"], int(config["pasting"].get("nside", 1024))
    )
    analysis["k_array_survey"] = jnp.array(k_survey)
    halo_params["ell_array"] = jnp.array(ell)
    analysis["l_array_survey"] = jnp.array(ell)
    analysis["dl_array_survey"] = jnp.array(dell)

    if z_max is not None:
        zmax = max(float(z_max), 0.05)
        halo_params["zmin"] = min(float(halo_params.get("zmin", 0.01)), 0.001)
        halo_params["zmax"] = max(zmax, float(halo_params.get("zmin", 0.01)) + 0.05)
        halo_params["nz"] = max(int(halo_params.get("nz", 64)), 48)
        analysis["zmin_for_Cls"] = 0.001
        analysis["zmax_for_Cls"] = halo_params["zmax"]
        analysis["nz_for_Cls"] = max(int(analysis.get("nz_for_Cls", 128)), 128)

    if log10_mass_min is not None:
        halo_params["lg10_Mmin"] = min(float(halo_params.get("lg10_Mmin", 10.5)), float(log10_mass_min) - 0.25)
    halo_params["lg10_Mmax"] = max(float(halo_params.get("lg10_Mmax", 16.0)), 16.0)
    halo_params["rmin"] = min(float(halo_params.get("rmin", 0.005)), 0.003)
    halo_params["rmax"] = max(float(halo_params.get("rmax", 12.0)), 12.0)
    halo_params["nr"] = max(int(halo_params.get("nr", 24)), 48)

    analysis["symbolic_pk"] = False
    analysis["symbolic_hmf"] = False
    return sim_params, halo_params, analysis, other_params


def prepare_stage31_godmax_config(
    config: Mapping[str, object],
    catalog_attrs: Optional[Mapping[str, object]] = None,
    *,
    is_cmb_lensing: bool = False,
    z_max: Optional[float] = None,
    log10_mass_min: Optional[float] = None,
):
    import godmax_multiprobe_theory_utils as gmt

    gcfg = config["godmax"]
    cfg = gmt.load_comparison_config(gcfg["comparison_config"])
    if "bestfit_params" in gcfg:
        cfg["params"] = gmt.deep_update(cfg["params"], read_yaml(gcfg["bestfit_params"]))
    cfg = gmt.materialize_nz_inputs(cfg)
    cfg = gmt.compute_desi_nbar_comoving(cfg)
    cfg["metadata"]["lmax"] = min(
        int(config.get("pasting", {}).get("lmax", 1024)),
        3 * int(config.get("pasting", {}).get("nside", 1024)) - 1,
    )
    pz_bin = int(config.get("pasting", {}).get("pz_bin", 1))
    pz_cfg = gmt.config_for_single_desi_pz(cfg, pz_bin)
    sim_params, halo_params, analysis, other_params = gmt._params_for_model(
        pz_cfg,
        is_cmb_lensing=bool(is_cmb_lensing),
    )

    if catalog_attrs and bool(gcfg.get("override_cosmology_from_catalog", True)):
        sim_params["cosmo"] = _catalog_cosmology(catalog_attrs)

    if z_max is not None:
        zmax = max(float(z_max), 0.05)
        halo_params["zmin"] = min(float(halo_params.get("zmin", 0.01)), 0.001)
        halo_params["zmax"] = max(zmax, float(halo_params.get("zmin", 0.01)) + 0.05)
        halo_params["nz"] = max(int(halo_params.get("nz", 64)), 48)
        analysis["zmin_for_Cls"] = 0.001
        analysis["zmax_for_Cls"] = halo_params["zmax"]
        analysis["nz_for_Cls"] = max(int(analysis.get("nz_for_Cls", 128)), 128)

    if log10_mass_min is not None:
        analytic_floor = float(gcfg.get("analytic_hod_log10_m_floor", 10.5))
        halo_params["lg10_Mmin"] = min(
            float(halo_params.get("lg10_Mmin", analytic_floor)),
            analytic_floor,
            float(log10_mass_min) - 0.25,
        )
    halo_params["lg10_Mmax"] = max(float(halo_params.get("lg10_Mmax", 16.0)), 16.0)
    halo_params["rmin"] = min(float(halo_params.get("rmin", 0.005)), 0.003)
    halo_params["rmax"] = max(float(halo_params.get("rmax", 12.0)), 12.0)
    halo_params["nr"] = max(int(halo_params.get("nr", 24)), 48)
    analysis["symbolic_pk"] = False
    analysis["symbolic_hmf"] = False
    analysis["single_photometric_pz_bin"] = pz_bin
    return sim_params, halo_params, analysis, other_params


def _normalize_pixel_backend(pixel_backend: str) -> str:
    backend = str(pixel_backend or _PIXEL_BACKEND_HEALPY).replace("-", "_").lower()
    if backend not in _PIXEL_BACKENDS:
        raise ValueError(f"Unsupported pixel_backend={pixel_backend!r}; expected one of {sorted(_PIXEL_BACKENDS)}.")
    return backend


def _unit_vec_lonlat_deg(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra_rad = math.radians(float(ra_deg))
    dec_rad = math.radians(float(dec_deg))
    cos_dec = math.cos(dec_rad)
    return np.asarray(
        [cos_dec * math.cos(ra_rad), cos_dec * math.sin(ra_rad), math.sin(dec_rad)],
        dtype=np.float64,
    )


def _estimate_query_disc_pixels(nside: int, radius_rad: float, safety_factor: float) -> int:
    radius = float(np.clip(radius_rad, 0.0, math.pi))
    estimate = 6.0 * float(nside) * float(nside) * (1.0 - math.cos(radius))
    return max(16, int(math.ceil(float(safety_factor) * estimate)) + 64)


def _query_disc_pixels(state: Mapping[str, object], jhalo: int, angle: float) -> Tuple[np.ndarray, int]:
    vec = _unit_vec_lonlat_deg(state["ra"][jhalo], state["dec"][jhalo])
    if state.get("pixel_backend") != _PIXEL_BACKEND_HEALPY_BUFF:
        pix = hp.query_disc(state["nside"], vec, angle, inclusive=state["inclusive"])
        return np.asarray(pix, dtype=state["pixel_dtype"]), 0

    nside = int(state["nside"])
    safety = float(state.get("query_disc_buffer_safety_factor", 2.0))
    min_len = _estimate_query_disc_pixels(nside, float(angle), safety)
    buff = state.get("query_disc_buffer")
    grows = 0
    if buff is None or len(buff) < min_len:
        state["query_disc_buffer"] = np.empty(min_len, dtype=np.int64)
        grows += 1

    while True:
        try:
            view = hp.query_disc(
                nside,
                vec,
                angle,
                inclusive=state["inclusive"],
                buff=state["query_disc_buffer"],
            )
            return np.asarray(view, dtype=state["pixel_dtype"]).copy(), grows
        except ValueError as exc:
            if "Buffer too small" not in str(exc):
                raise
            old_len = len(state["query_disc_buffer"])
            state["query_disc_buffer"] = np.empty(max(old_len * 2, min_len * 2), dtype=np.int64)
            grows += 1
            if grows > 4:
                raise


def _precompute_pixel_grouping(nearby_pix_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pix = np.asarray(nearby_pix_all)
    if pix.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.asarray([0], dtype=np.int64),
        )
    sort_idx = np.argsort(pix)
    sorted_pix = pix[sort_idx]
    pix_unique = np.unique(sorted_pix)
    change_points = np.diff(sorted_pix, prepend=sorted_pix[0] - 1, append=sorted_pix[-1] + 1) != 0
    boundaries = np.where(change_points)[0].astype(np.int64)
    return pix_unique.astype(np.int64), sort_idx.astype(np.int64), boundaries


def _init_pixel_worker(
    ra,
    dec,
    r200c,
    da,
    max_paint,
    nside,
    pixel_dtype,
    inclusive,
    single_pixel_angle_factor=0.0,
    pixel_backend: str = _PIXEL_BACKEND_HEALPY,
    query_disc_buffer_safety_factor: float = 2.0,
):
    global _PIXEL_STATE
    _PIXEL_STATE = {
        "ra": ra,
        "dec": dec,
        "r200c": r200c,
        "da": da,
        "max_paint": float(max_paint),
        "nside": int(nside),
        "pixel_dtype": pixel_dtype,
        "inclusive": bool(inclusive),
        "single_pixel_angle_rad": float(single_pixel_angle_factor) * float(hp.nside2resol(int(nside))),
        "pixel_backend": _normalize_pixel_backend(pixel_backend),
        "query_disc_buffer_safety_factor": max(1.0, float(query_disc_buffer_safety_factor)),
        "query_disc_buffer": None,
    }


def _process_halo_pixel_index(jhalo: int):
    state = _PIXEL_STATE
    ra = state["ra"]
    dec = state["dec"]
    r200c = state["r200c"]
    da = state["da"]
    angle = state["max_paint"] * float(r200c[jhalo]) / max(float(da[jhalo]), 1.0e-8)
    used_shortcut = bool(angle <= float(state.get("single_pixel_angle_rad", 0.0)))
    if used_shortcut:
        nearby_pix = np.asarray([hp.ang2pix(state["nside"], ra[jhalo], dec[jhalo], lonlat=True)], dtype=state["pixel_dtype"])
        buffer_grows = 0
    else:
        nearby_pix, buffer_grows = _query_disc_pixels(state, jhalo, angle)
        if len(nearby_pix) == 0:
            nearby_pix = np.asarray([hp.ang2pix(state["nside"], ra[jhalo], dec[jhalo], lonlat=True)], dtype=state["pixel_dtype"])

    nearby_ra, nearby_dec = hp.pix2ang(state["nside"], nearby_pix, lonlat=True)
    ra1, dec1 = np.radians(float(ra[jhalo])), np.radians(float(dec[jhalo]))
    ra2, dec2 = np.radians(nearby_ra), np.radians(nearby_dec)
    a = np.sin((dec1 - dec2) / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra1 - ra2) / 2.0) ** 2
    theta = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    distances = (float(da[jhalo]) * theta).astype(np.float32)
    return nearby_pix, distances, int(jhalo), len(nearby_pix), used_shortcut, int(buffer_grows)


def _warm_pixel_worker(jhalo: int):
    state = _PIXEL_STATE
    # Force each forked worker to fault in the shared catalog arrays before
    # chunk 0, then exercise the same pixel-query path used by real chunks.
    checksum = 0.0
    for key in ("ra", "dec", "r200c", "da"):
        checksum += float(np.sum(state[key], dtype=np.float64))
    pix, _dist, _halo, npix, used_shortcut, buffer_grows = _process_halo_pixel_index(int(jhalo))
    return {
        "pid": int(os.getpid()),
        "jhalo": int(jhalo),
        "npix": int(npix),
        "used_shortcut": bool(used_shortcut),
        "buffer_grows": int(buffer_grows),
        "checksum": float(checksum),
        "first_pix": int(pix[0]) if len(pix) else -1,
    }


def _angular_distances_hMpc(ra1_deg, dec1_deg, ra2_deg, dec2_deg, da_hMpc):
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = np.radians(ra2_deg)
    dec2 = np.radians(dec2_deg)
    a = np.sin((dec1 - dec2) / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra1 - ra2) / 2.0) ** 2
    theta = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return (np.asarray(da_hMpc, dtype=np.float32) * theta).astype(np.float32)


def _angular_distances_rad(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = np.radians(ra2_deg)
    dec2 = np.radians(dec2_deg)
    a = np.sin((dec1 - dec2) / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra1 - ra2) / 2.0) ** 2
    return 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@njit(cache=False)
def _ring_first_idx_leq_desc(values, threshold):
    lo = 0
    hi = len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] > threshold:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=False)
def _ring_last_idx_geq_desc(values, threshold):
    lo = 0
    hi = len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] >= threshold:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@njit(cache=False)
def _ring_segment_count(phi_lo, phi_hi, phi0, dphi, nr, x, y, z, zr, sr, cosr):
    two_pi = 2.0 * math.pi
    eps = 1.0e-12
    count = 0
    for copy_idx in range(2):
        shift = two_pi * copy_idx
        j0 = int(math.ceil((phi_lo + shift - phi0) / dphi - eps))
        j1 = int(math.floor((phi_hi + shift - phi0) / dphi + eps))
        if j1 < 0 or j0 >= nr:
            continue
        if j0 < 0:
            j0 = 0
        if j1 >= nr:
            j1 = nr - 1
        for j in range(j0, j1 + 1):
            phi = phi0 + j * dphi
            if phi >= two_pi:
                phi -= two_pi
            dot = x * sr * math.cos(phi) + y * sr * math.sin(phi) + z * zr
            if dot >= cosr - 1.0e-13:
                count += 1
    return count


@njit(cache=False)
def _ring_segment_fill(
    pix_out,
    dist_out,
    halo_out,
    offset,
    phi_lo,
    phi_hi,
    phi0,
    dphi,
    nr,
    startpix,
    x,
    y,
    z,
    zr,
    sr,
    cosr,
    da,
    jhalo,
):
    two_pi = 2.0 * math.pi
    eps = 1.0e-12
    written = 0
    for copy_idx in range(2):
        shift = two_pi * copy_idx
        j0 = int(math.ceil((phi_lo + shift - phi0) / dphi - eps))
        j1 = int(math.floor((phi_hi + shift - phi0) / dphi + eps))
        if j1 < 0 or j0 >= nr:
            continue
        if j0 < 0:
            j0 = 0
        if j1 >= nr:
            j1 = nr - 1
        for j in range(j0, j1 + 1):
            phi = phi0 + j * dphi
            if phi >= two_pi:
                phi -= two_pi
            dot = x * sr * math.cos(phi) + y * sr * math.sin(phi) + z * zr
            if dot >= cosr - 1.0e-13:
                dot_clip = dot
                if dot_clip > 1.0:
                    dot_clip = 1.0
                elif dot_clip < -1.0:
                    dot_clip = -1.0
                pix_out[offset + written] = startpix + j
                dist_out[offset + written] = da * math.acos(dot_clip)
                halo_out[offset + written] = jhalo
                written += 1
    return written


@njit(cache=False)
def _ring_count_one(
    ra_rad,
    dec_rad,
    radius,
    ring_startpix,
    ring_npix,
    ring_z,
    ring_sin,
    ring_phi0,
):
    two_pi = 2.0 * math.pi
    if radius < 0.0:
        radius = 0.0
    if radius > math.pi:
        radius = math.pi
    z = math.sin(dec_rad)
    cos_dec = math.cos(dec_rad)
    x = cos_dec * math.cos(ra_rad)
    y = cos_dec * math.sin(ra_rad)
    theta0 = math.acos(min(1.0, max(-1.0, z)))
    theta_min = theta0 - radius
    if theta_min < 0.0:
        theta_min = 0.0
    theta_max = theta0 + radius
    if theta_max > math.pi:
        theta_max = math.pi
    z_high = math.cos(theta_min)
    z_low = math.cos(theta_max)
    first = _ring_first_idx_leq_desc(ring_z, z_high)
    last = _ring_last_idx_geq_desc(ring_z, z_low)
    if first < 0:
        first = 0
    if last >= len(ring_z):
        last = len(ring_z) - 1
    if first > last:
        return 0

    cosr = math.cos(radius)
    phi_c = ra_rad % two_pi
    total = 0
    for iring in range(first, last + 1):
        nr = int(ring_npix[iring])
        zr = ring_z[iring]
        sr = ring_sin[iring]
        amp = cos_dec * sr
        base = z * zr
        dphi = two_pi / nr
        phi0 = ring_phi0[iring]
        if amp <= 1.0e-15:
            if base >= cosr - 1.0e-13:
                for j in range(nr):
                    phi = phi0 + j * dphi
                    if phi >= two_pi:
                        phi -= two_pi
                    dot = x * sr * math.cos(phi) + y * sr * math.sin(phi) + z * zr
                    if dot >= cosr - 1.0e-13:
                        total += 1
            continue
        arg = (cosr - base) / amp
        if arg > 1.0:
            continue
        if arg <= -1.0:
            for j in range(nr):
                phi = phi0 + j * dphi
                if phi >= two_pi:
                    phi -= two_pi
                dot = x * sr * math.cos(phi) + y * sr * math.sin(phi) + z * zr
                if dot >= cosr - 1.0e-13:
                    total += 1
            continue
        delta = math.acos(arg)
        lo = phi_c - delta
        hi = phi_c + delta
        if lo < 0.0:
            total += _ring_segment_count(lo + two_pi, two_pi, phi0, dphi, nr, x, y, z, zr, sr, cosr)
            total += _ring_segment_count(0.0, hi, phi0, dphi, nr, x, y, z, zr, sr, cosr)
        elif hi >= two_pi:
            total += _ring_segment_count(lo, two_pi, phi0, dphi, nr, x, y, z, zr, sr, cosr)
            total += _ring_segment_count(0.0, hi - two_pi, phi0, dphi, nr, x, y, z, zr, sr, cosr)
        else:
            total += _ring_segment_count(lo, hi, phi0, dphi, nr, x, y, z, zr, sr, cosr)
    return total


@njit(cache=False)
def _ring_fill_one(
    pix_out,
    dist_out,
    halo_out,
    offset,
    ra_rad,
    dec_rad,
    radius,
    da,
    jhalo,
    ring_startpix,
    ring_npix,
    ring_z,
    ring_sin,
    ring_phi0,
):
    two_pi = 2.0 * math.pi
    if radius < 0.0:
        radius = 0.0
    if radius > math.pi:
        radius = math.pi
    z = math.sin(dec_rad)
    cos_dec = math.cos(dec_rad)
    x = cos_dec * math.cos(ra_rad)
    y = cos_dec * math.sin(ra_rad)
    theta0 = math.acos(min(1.0, max(-1.0, z)))
    theta_min = theta0 - radius
    if theta_min < 0.0:
        theta_min = 0.0
    theta_max = theta0 + radius
    if theta_max > math.pi:
        theta_max = math.pi
    z_high = math.cos(theta_min)
    z_low = math.cos(theta_max)
    first = _ring_first_idx_leq_desc(ring_z, z_high)
    last = _ring_last_idx_geq_desc(ring_z, z_low)
    if first < 0:
        first = 0
    if last >= len(ring_z):
        last = len(ring_z) - 1
    if first > last:
        return 0

    cosr = math.cos(radius)
    phi_c = ra_rad % two_pi
    written = 0
    for iring in range(first, last + 1):
        nr = int(ring_npix[iring])
        startpix = int(ring_startpix[iring])
        zr = ring_z[iring]
        sr = ring_sin[iring]
        amp = cos_dec * sr
        base = z * zr
        dphi = two_pi / nr
        phi0 = ring_phi0[iring]
        if amp <= 1.0e-15:
            if base >= cosr - 1.0e-13:
                for j in range(nr):
                    phi = phi0 + j * dphi
                    if phi >= two_pi:
                        phi -= two_pi
                    dot = x * sr * math.cos(phi) + y * sr * math.sin(phi) + z * zr
                    if dot >= cosr - 1.0e-13:
                        dot_clip = min(1.0, max(-1.0, dot))
                        pix_out[offset + written] = startpix + j
                        dist_out[offset + written] = da * math.acos(dot_clip)
                        halo_out[offset + written] = jhalo
                        written += 1
            continue
        arg = (cosr - base) / amp
        if arg > 1.0:
            continue
        if arg <= -1.0:
            for j in range(nr):
                phi = phi0 + j * dphi
                if phi >= two_pi:
                    phi -= two_pi
                dot = x * sr * math.cos(phi) + y * sr * math.sin(phi) + z * zr
                if dot >= cosr - 1.0e-13:
                    dot_clip = min(1.0, max(-1.0, dot))
                    pix_out[offset + written] = startpix + j
                    dist_out[offset + written] = da * math.acos(dot_clip)
                    halo_out[offset + written] = jhalo
                    written += 1
            continue
        delta = math.acos(arg)
        lo = phi_c - delta
        hi = phi_c + delta
        if lo < 0.0:
            written += _ring_segment_fill(
                pix_out, dist_out, halo_out, offset + written,
                lo + two_pi, two_pi, phi0, dphi, nr, startpix,
                x, y, z, zr, sr, cosr, da, jhalo,
            )
            written += _ring_segment_fill(
                pix_out, dist_out, halo_out, offset + written,
                0.0, hi, phi0, dphi, nr, startpix,
                x, y, z, zr, sr, cosr, da, jhalo,
            )
        elif hi >= two_pi:
            written += _ring_segment_fill(
                pix_out, dist_out, halo_out, offset + written,
                lo, two_pi, phi0, dphi, nr, startpix,
                x, y, z, zr, sr, cosr, da, jhalo,
            )
            written += _ring_segment_fill(
                pix_out, dist_out, halo_out, offset + written,
                0.0, hi - two_pi, phi0, dphi, nr, startpix,
                x, y, z, zr, sr, cosr, da, jhalo,
            )
        else:
            written += _ring_segment_fill(
                pix_out, dist_out, halo_out, offset + written,
                lo, hi, phi0, dphi, nr, startpix,
                x, y, z, zr, sr, cosr, da, jhalo,
            )
    return written


@njit(cache=False, parallel=True)
def _ring_query_batch_numba(
    query_ids,
    ra_deg,
    dec_deg,
    da_hmpc,
    angles,
    fallback_pix,
    fallback_dist,
    ring_startpix,
    ring_npix,
    ring_z,
    ring_sin,
    ring_phi0,
):
    deg2rad = math.pi / 180.0
    n = len(query_ids)
    counts = np.empty(n, dtype=np.int64)
    for i in prange(n):
        jhalo = int(query_ids[i])
        count = _ring_count_one(
            float(ra_deg[jhalo]) * deg2rad,
            float(dec_deg[jhalo]) * deg2rad,
            float(angles[jhalo]),
            ring_startpix,
            ring_npix,
            ring_z,
            ring_sin,
            ring_phi0,
        )
        if count <= 0:
            count = 1
        counts[i] = count

    starts = np.empty(n, dtype=np.int64)
    total = 0
    for i in range(n):
        starts[i] = total
        total += counts[i]

    pix_out = np.empty(total, dtype=np.int64)
    dist_out = np.empty(total, dtype=np.float32)
    halo_out = np.empty(total, dtype=np.int64)
    for i in prange(n):
        jhalo = int(query_ids[i])
        offset = starts[i]
        written = _ring_fill_one(
            pix_out,
            dist_out,
            halo_out,
            offset,
            float(ra_deg[jhalo]) * deg2rad,
            float(dec_deg[jhalo]) * deg2rad,
            float(angles[jhalo]),
            float(da_hmpc[jhalo]),
            jhalo,
            ring_startpix,
            ring_npix,
            ring_z,
            ring_sin,
            ring_phi0,
        )
        if written <= 0:
            pix_out[offset] = int(fallback_pix[i])
            dist_out[offset] = float(fallback_dist[i])
            halo_out[offset] = jhalo
    return pix_out, dist_out, halo_out, counts


def _ring_geometry(nside: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nside = int(nside)
    cached = _RING_GEOM_CACHE.get(nside)
    if cached is not None:
        return cached
    rings = np.arange(1, 4 * nside, dtype=np.int64)
    startpix, ringpix, costheta, sintheta, _shifted = hp.ringinfo(nside, rings)
    _theta0, phi0 = hp.pix2ang(nside, np.asarray(startpix, dtype=np.int64), nest=False)
    geom = (
        np.asarray(startpix, dtype=np.int64),
        np.asarray(ringpix, dtype=np.int64),
        np.asarray(costheta, dtype=np.float64),
        np.asarray(sintheta, dtype=np.float64),
        np.asarray(phi0, dtype=np.float64),
    )
    _RING_GEOM_CACHE[nside] = geom
    return geom


def _build_ring_pixel_results(
    halo_ids: np.ndarray,
    *,
    ra: np.ndarray,
    dec: np.ndarray,
    da: np.ndarray,
    angles: np.ndarray,
    nside: int,
    pixel_dtype,
) -> List[Tuple[np.ndarray, np.ndarray, int, int, bool, int]]:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("pixel_backend='healpy_ring' requires numba, but numba is not importable.")
    halo_ids = np.asarray(halo_ids, dtype=np.int64)
    if halo_ids.size == 0:
        return []
    fallback_pix = hp.ang2pix(nside, ra[halo_ids], dec[halo_ids], lonlat=True).astype(np.int64, copy=False)
    fallback_ra, fallback_dec = hp.pix2ang(nside, fallback_pix, lonlat=True)
    fallback_dist = _angular_distances_hMpc(ra[halo_ids], dec[halo_ids], fallback_ra, fallback_dec, da[halo_ids])
    ring_startpix, ring_npix, ring_z, ring_sin, ring_phi0 = _ring_geometry(nside)
    pix_flat, dist_flat, halo_flat, counts = _ring_query_batch_numba(
        halo_ids,
        np.asarray(ra, dtype=np.float64),
        np.asarray(dec, dtype=np.float64),
        np.asarray(da, dtype=np.float64),
        np.asarray(angles, dtype=np.float64),
        fallback_pix,
        np.asarray(fallback_dist, dtype=np.float32),
        ring_startpix,
        ring_npix,
        ring_z,
        ring_sin,
        ring_phi0,
    )
    pix_flat = np.asarray(pix_flat, dtype=pixel_dtype)
    dist_flat = np.asarray(dist_flat, dtype=np.float32)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    ends = np.cumsum(counts).astype(np.int64)
    results: List[Tuple[np.ndarray, np.ndarray, int, int, bool, int]] = []
    for local, jhalo in enumerate(halo_ids):
        start = int(starts[local])
        end = int(ends[local])
        # If numerical edge cases changed the number written, fall back to the
        # halo_flat boundaries rather than assuming count/fill are identical.
        if end > len(pix_flat) or start >= len(pix_flat) or (end > start and int(halo_flat[start]) != int(jhalo)):
            matches = np.where(np.asarray(halo_flat) == int(jhalo))[0]
            if len(matches):
                start = int(matches[0])
                end = int(matches[-1]) + 1
        pix = pix_flat[start:end]
        dist = dist_flat[start:end]
        results.append((pix, dist, int(jhalo), int(len(pix)), False, 0))
    return results


def _build_stencil_pixel_results(
    halo_ids: np.ndarray,
    *,
    ra: np.ndarray,
    dec: np.ndarray,
    da: np.ndarray,
    angles: np.ndarray,
    nside: int,
    pixel_dtype,
) -> List[Tuple[np.ndarray, np.ndarray, int, int, bool, int]]:
    """Build small-radius pixel candidates from center+nearest-neighbor stencils.

    This path is exact only when the one-ring stencil contains all pixel centers
    inside the disc. Callers keep a conservative angular-radius threshold and
    should validate it against the plain ``healpy.query_disc`` backend.
    """

    halo_ids = np.asarray(halo_ids, dtype=np.int64)
    if halo_ids.size == 0:
        return []

    center_pix = hp.ang2pix(nside, ra[halo_ids], dec[halo_ids], lonlat=True).astype(np.int64, copy=False)
    neighbors = np.asarray(hp.get_all_neighbours(nside, center_pix, nest=False), dtype=np.int64)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    candidates = np.concatenate([center_pix[None, :], neighbors], axis=0).T

    flat_candidates = candidates.reshape(-1)
    flat_local = np.repeat(np.arange(halo_ids.size, dtype=np.int64), candidates.shape[1])
    valid = flat_candidates >= 0
    flat_candidates = flat_candidates[valid]
    flat_local = flat_local[valid]
    if flat_candidates.size == 0:
        flat_candidates = center_pix
        flat_local = np.arange(halo_ids.size, dtype=np.int64)

    # Drop duplicate candidate pixels per halo, mostly for polar/edge cases.
    order = np.lexsort((flat_candidates, flat_local))
    flat_candidates = flat_candidates[order]
    flat_local = flat_local[order]
    unique = np.ones(flat_candidates.size, dtype=bool)
    unique[1:] = (flat_candidates[1:] != flat_candidates[:-1]) | (flat_local[1:] != flat_local[:-1])
    flat_candidates = flat_candidates[unique]
    flat_local = flat_local[unique]

    cand_ra, cand_dec = hp.pix2ang(nside, flat_candidates, lonlat=True)
    jhalo = halo_ids[flat_local]
    theta = _angular_distances_rad(ra[jhalo], dec[jhalo], cand_ra, cand_dec)
    keep = theta <= (angles[jhalo] + 1.0e-12)
    kept_pix = flat_candidates[keep].astype(pixel_dtype, copy=False)
    kept_local = flat_local[keep]
    kept_dist = (da[halo_ids[kept_local]] * theta[keep]).astype(np.float32)

    counts = np.bincount(kept_local, minlength=halo_ids.size).astype(np.int64)
    missing_local = np.where(counts == 0)[0]
    if missing_local.size:
        fallback_pix = center_pix[missing_local].astype(pixel_dtype, copy=False)
        fallback_ra, fallback_dec = hp.pix2ang(nside, fallback_pix, lonlat=True)
        fallback_halo = halo_ids[missing_local]
        fallback_dist = _angular_distances_hMpc(
            ra[fallback_halo],
            dec[fallback_halo],
            fallback_ra,
            fallback_dec,
            da[fallback_halo],
        )
        kept_pix = np.concatenate([kept_pix, fallback_pix.astype(pixel_dtype, copy=False)])
        kept_dist = np.concatenate([kept_dist, fallback_dist.astype(np.float32, copy=False)])
        kept_local = np.concatenate([kept_local, missing_local.astype(np.int64, copy=False)])
        counts[missing_local] = 1

    order = np.argsort(kept_local, kind="stable")
    kept_local = kept_local[order]
    kept_pix = kept_pix[order]
    kept_dist = kept_dist[order]

    results: List[Tuple[np.ndarray, np.ndarray, int, int, bool, int]] = []
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ends = np.cumsum(counts)
    for local, jhalo_value in enumerate(halo_ids):
        start = int(starts[local])
        end = int(ends[local])
        pix = kept_pix[start:end].astype(pixel_dtype, copy=False)
        dist = kept_dist[start:end].astype(np.float32, copy=False)
        results.append((pix, dist, int(jhalo_value), int(len(pix)), False, 0))
    return results


def _build_pixel_batch(
    start: int,
    end: int,
    *,
    ra: np.ndarray,
    dec: np.ndarray,
    r200c: np.ndarray,
    da: np.ndarray,
    mass: np.ndarray,
    redshift: np.ndarray,
    vlos: np.ndarray,
    max_paint: float,
    nside: int,
    pixel_dtype,
    single_pixel_angle_rad: float,
    pool=None,
    pool_chunksize: int = 1,
    pixel_backend: str = _PIXEL_BACKEND_HEALPY,
    stencil_pixel_angle_rad: float = 0.0,
    include_legacy_pixel_arrays: bool = False,
) -> Optional[dict]:
    halo_ids = np.arange(int(start), int(end), dtype=np.int64)
    if len(halo_ids) == 0:
        return None
    angles = float(max_paint) * r200c[halo_ids] / np.maximum(da[halo_ids], 1.0e-8)
    shortcut_mask = angles <= float(single_pixel_angle_rad)
    stencil_mask = np.zeros(len(halo_ids), dtype=bool)
    if pixel_backend == _PIXEL_BACKEND_HEALPY_STENCIL and float(stencil_pixel_angle_rad) > float(single_pixel_angle_rad):
        stencil_mask = (~shortcut_mask) & (angles <= float(stencil_pixel_angle_rad))
    query_ids = halo_ids[(~shortcut_mask) & (~stencil_mask)]
    stencil_ids = halo_ids[stencil_mask]

    stencil_results = _build_stencil_pixel_results(
        stencil_ids,
        ra=ra,
        dec=dec,
        da=da,
        angles=float(max_paint) * r200c / np.maximum(da, 1.0e-8),
        nside=nside,
        pixel_dtype=pixel_dtype,
    )
    ring_results = []
    query_results = []
    n_query_disc_buffer_grows = 0
    if pixel_backend == _PIXEL_BACKEND_HEALPY_RING and len(query_ids):
        ring_results = _build_ring_pixel_results(
            query_ids,
            ra=ra,
            dec=dec,
            da=da,
            angles=float(max_paint) * r200c / np.maximum(da, 1.0e-8),
            nside=nside,
            pixel_dtype=pixel_dtype,
        )
    elif len(query_ids):
        if pool is None:
            query_results = [_process_halo_pixel_index(int(jhalo)) for jhalo in query_ids]
        else:
            query_results = pool.map(_process_halo_pixel_index, [int(jhalo) for jhalo in query_ids], chunksize=max(1, int(pool_chunksize)))
        n_query_disc_buffer_grows = int(sum(int(res[5]) for res in query_results if len(res) > 5))
    pixel_results = stencil_results + ring_results + query_results
    lengths = np.ones(len(halo_ids), dtype=np.int64)
    for res in pixel_results:
        lengths[int(res[2]) - int(start)] = int(res[3])
    total = int(lengths.sum())
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int32)
    ends = np.cumsum(lengths).astype(np.int32)
    pix = np.empty(total, dtype=pixel_dtype)
    dist = np.empty(total, dtype=np.float32)
    halo_indices = np.empty(total, dtype=np.int64)

    shortcut_ids = halo_ids[shortcut_mask]
    if len(shortcut_ids):
        shortcut_local = shortcut_ids - int(start)
        shortcut_offsets = starts[shortcut_local]
        shortcut_pix = hp.ang2pix(nside, ra[shortcut_ids], dec[shortcut_ids], lonlat=True).astype(pixel_dtype)
        shortcut_ra, shortcut_dec = hp.pix2ang(nside, shortcut_pix, lonlat=True)
        shortcut_dist = _angular_distances_hMpc(
            ra[shortcut_ids],
            dec[shortcut_ids],
            shortcut_ra,
            shortcut_dec,
            da[shortcut_ids],
        )
        pix[shortcut_offsets] = shortcut_pix
        dist[shortcut_offsets] = np.maximum(shortcut_dist, 1.0e-7)
        halo_indices[shortcut_offsets] = shortcut_ids

    for res in pixel_results:
        local = int(res[2]) - int(start)
        out_start = int(starts[local])
        out_end = int(ends[local])
        pix[out_start:out_end] = res[0]
        dist[out_start:out_end] = np.maximum(res[1], 1.0e-7)
        halo_indices[out_start:out_end] = int(res[2])

    out = {
        "nearby_pix_all": pix,
        "distances": dist,
        "logM": np.log(mass[halo_indices]).astype(np.float32),
        "z": redshift[halo_indices].astype(np.float32),
        "vlos": vlos[halo_indices].astype(np.float32),
        "n_halos": int(len(halo_ids)),
        "n_single_pixel_shortcut": int(len(shortcut_ids)),
        "n_stencil": int(len(stencil_ids)),
        "n_ring": int(len(query_ids) if pixel_backend == _PIXEL_BACKEND_HEALPY_RING else 0),
        "n_query_disc": int(0 if pixel_backend == _PIXEL_BACKEND_HEALPY_RING else len(query_ids)),
        "n_query_disc_buffer_grows": int(n_query_disc_buffer_grows),
    }
    if include_legacy_pixel_arrays:
        out.update(
            {
                "start_ind": starts,
                "end_ind": ends,
                "ang_distance_all": da[halo_ids].astype(np.float32),
                "rp_max_all": (float(max_paint) * r200c[halo_ids]).astype(np.float32),
            }
        )
    return out


def _concatenate_pixel_results(
    results,
    mass,
    redshift,
    vlos,
    da,
    r200c,
    max_paint,
    pixel_dtype,
    include_legacy_pixel_arrays: bool = False,
):
    results = [res for res in results if res is not None]
    if not results:
        return None
    lengths = np.asarray([res[3] for res in results], dtype=np.int64)
    total = int(lengths.sum())
    pix = np.empty(total, dtype=pixel_dtype)
    dist = np.empty(total, dtype=np.float32)
    halo_indices = np.empty(total, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int32)
    ends = np.cumsum(lengths).astype(np.int32)
    for start, end, res in zip(starts, ends, results):
        pix[start:end] = res[0]
        dist[start:end] = np.maximum(res[1], 1.0e-7)
        halo_indices[start:end] = res[2]

    orig_halo_idx = np.asarray([res[2] for res in results], dtype=np.int64)
    n_single_pixel_shortcut = int(sum(1 for res in results if len(res) > 4 and bool(res[4])))
    out = {
        "nearby_pix_all": pix,
        "distances": dist,
        "logM": np.log(mass[halo_indices]).astype(np.float32),
        "z": redshift[halo_indices].astype(np.float32),
        "vlos": vlos[halo_indices].astype(np.float32),
        "n_halos": int(len(orig_halo_idx)),
        "n_single_pixel_shortcut": n_single_pixel_shortcut,
        "n_query_disc": int(len(results) - n_single_pixel_shortcut),
        "n_query_disc_buffer_grows": int(sum(int(res[5]) for res in results if len(res) > 5)),
    }
    if include_legacy_pixel_arrays:
        out.update(
            {
                "start_ind": starts,
                "end_ind": ends,
                "ang_distance_all": da[orig_halo_idx].astype(np.float32),
                "rp_max_all": (max_paint * r200c[orig_halo_idx]).astype(np.float32),
            }
        )
    return out


def build_pixel_work_package(
    catalog: Mapping[str, np.ndarray],
    nside: int,
    max_paint: float,
    batch_size: int,
    workers: Optional[int] = None,
    *,
    start_method: Optional[str] = None,
    pool=None,
    pool_chunksize: Optional[int] = None,
    single_pixel_angle_factor: float = 0.0,
    index_start: int = 0,
    index_end: Optional[int] = None,
    verbose: bool = True,
    log_batches: bool = False,
    pixel_backend: str = _PIXEL_BACKEND_HEALPY,
    query_disc_buffer_safety_factor: float = 2.0,
    stencil_pixel_angle_factor: float = 1.0,
    include_legacy_pixel_arrays: bool = False,
    precompute_pixel_groups: bool = True,
    pixel_gc_collect_every_n_batches: int = 0,
):
    pixel_backend = _normalize_pixel_backend(pixel_backend)
    pixel_dtype = np.int32 if nside <= 8192 else np.int64
    ra = np.asarray(catalog["ra_deg"], dtype=np.float32)
    dec = np.asarray(catalog["dec_deg"], dtype=np.float32)
    r200c = np.asarray(catalog["R200c_hMpc"], dtype=np.float32)
    da = np.asarray(catalog["DA_hMpc"], dtype=np.float32)
    mass = np.asarray(catalog["M200c_hMsun"], dtype=np.float64)
    redshift = np.asarray(catalog["z"], dtype=np.float32)
    vlos = np.asarray(catalog["vlos_kms"], dtype=np.float32)
    n_halos_total = len(ra)
    index_start = max(0, int(index_start))
    index_end = n_halos_total if index_end is None else min(int(index_end), n_halos_total)
    n_halos = max(0, index_end - index_start)
    if n_halos == 0:
        return None
    batch_size = n_halos if int(batch_size) <= 0 else int(batch_size)
    pixel_gc_collect_every_n_batches = int(pixel_gc_collect_every_n_batches)
    workers = auto_cpu_workers() if workers is None or int(workers) <= 0 else int(workers)
    if pixel_backend == _PIXEL_BACKEND_HEALPY_RING:
        if _NUMBA_AVAILABLE and set_num_threads is not None:
            try:
                set_num_threads(max(1, int(workers)))
            except ValueError:
                set_num_threads(max(1, min(int(workers), int(get_num_threads()))))
        # The ring backend handles query halos in a compiled batch kernel; a
        # Python multiprocessing pool would only add startup/IPC overhead.
        workers = 1
    start_method = start_method or os.environ.get("ABACUS_PASTE_PIXEL_START_METHOD", "forkserver")
    single_pixel_angle_factor = max(0.0, float(single_pixel_angle_factor))
    single_pixel_angle_rad = single_pixel_angle_factor * float(hp.nside2resol(int(nside)))
    stencil_pixel_angle_factor = max(0.0, float(stencil_pixel_angle_factor))
    stencil_pixel_angle_rad = stencil_pixel_angle_factor * float(hp.nside2resol(int(nside)))
    if pool_chunksize is None or int(pool_chunksize) <= 0:
        pool_chunksize = max(1, int(math.ceil(n_halos / max(1, int(workers) * 8))))
    else:
        pool_chunksize = int(pool_chunksize)
    all_batches = []
    n_pixel_batches = 0

    def _maybe_collect_pixel_gc() -> None:
        nonlocal n_pixel_batches
        n_pixel_batches += 1
        if pixel_gc_collect_every_n_batches > 0 and n_pixel_batches % pixel_gc_collect_every_n_batches == 0:
            gc.collect()

    _log(
        f"[paste:cpu] pixel-neighbor build start halos={n_halos:,} nside={nside} "
        f"batch_size={batch_size:,} workers={workers} start_method={start_method} "
        f"pool_chunksize={pool_chunksize:,} single_pixel_angle_factor={single_pixel_angle_factor:.3g} "
        f"stencil_pixel_angle_factor={stencil_pixel_angle_factor:.3g} pixel_backend={pixel_backend} "
        f"gc_collect_every_n_batches={pixel_gc_collect_every_n_batches}",
        verbose,
    )
    if int(workers) <= 1:
        _init_pixel_worker(
            ra,
            dec,
            r200c,
            da,
            max_paint,
            nside,
            pixel_dtype,
            False,
            single_pixel_angle_factor,
            pixel_backend,
            query_disc_buffer_safety_factor,
        )
        for start in range(index_start, index_end, batch_size):
            end = min(start + batch_size, index_end)
            t_batch = time.perf_counter()
            batch = _build_pixel_batch(
                start,
                end,
                ra=ra,
                dec=dec,
                r200c=r200c,
                da=da,
                mass=mass,
                redshift=redshift,
                vlos=vlos,
                max_paint=max_paint,
                nside=nside,
                pixel_dtype=pixel_dtype,
                single_pixel_angle_rad=single_pixel_angle_rad,
                pool=None,
                pool_chunksize=pool_chunksize,
                pixel_backend=pixel_backend,
                stencil_pixel_angle_rad=stencil_pixel_angle_rad,
                include_legacy_pixel_arrays=include_legacy_pixel_arrays,
            )
            if batch is not None:
                all_batches.append(batch)
            _log(
                f"[paste:cpu] pixel batch halos {start - index_start:,}:{end - index_start:,} "
                f"pairs={len(batch['nearby_pix_all']) if batch else 0:,} "
                f"time={time.perf_counter() - t_batch:.1f}s",
                verbose and log_batches,
            )
            _maybe_collect_pixel_gc()
    elif pool is not None:
        _log("[paste:cpu] using persistent multiprocessing Pool for healpy.query_disc.", verbose)
        for start in range(index_start, index_end, batch_size):
            end = min(start + batch_size, index_end)
            t_batch = time.perf_counter()
            batch = _build_pixel_batch(
                start,
                end,
                ra=ra,
                dec=dec,
                r200c=r200c,
                da=da,
                mass=mass,
                redshift=redshift,
                vlos=vlos,
                max_paint=max_paint,
                nside=nside,
                pixel_dtype=pixel_dtype,
                single_pixel_angle_rad=single_pixel_angle_rad,
                pool=pool,
                pool_chunksize=pool_chunksize,
                pixel_backend=pixel_backend,
                stencil_pixel_angle_rad=stencil_pixel_angle_rad,
                include_legacy_pixel_arrays=include_legacy_pixel_arrays,
            )
            if batch is not None:
                all_batches.append(batch)
            _log(
                f"[paste:cpu] pixel batch halos {start - index_start:,}:{end - index_start:,} "
                f"pairs={len(batch['nearby_pix_all']) if batch else 0:,} "
                f"time={time.perf_counter() - t_batch:.1f}s",
                verbose and log_batches,
            )
            _maybe_collect_pixel_gc()
    else:
        _log(
            "[paste:cpu] using multiprocessing Pool for healpy.query_disc. "
            "Use start_method=forkserver or spawn after JAX has initialized.",
            verbose,
        )
        ctx = get_context(str(start_method))
        with ctx.Pool(
            processes=int(workers),
            initializer=_init_pixel_worker,
            initargs=(
                ra,
                dec,
                r200c,
                da,
                max_paint,
                nside,
                pixel_dtype,
                False,
                single_pixel_angle_factor,
                pixel_backend,
                query_disc_buffer_safety_factor,
            ),
        ) as pool:
            for start in range(index_start, index_end, batch_size):
                end = min(start + batch_size, index_end)
                t_batch = time.perf_counter()
                batch = _build_pixel_batch(
                    start,
                    end,
                    ra=ra,
                    dec=dec,
                    r200c=r200c,
                    da=da,
                    mass=mass,
                    redshift=redshift,
                    vlos=vlos,
                    max_paint=max_paint,
                    nside=nside,
                    pixel_dtype=pixel_dtype,
                    single_pixel_angle_rad=single_pixel_angle_rad,
                    pool=pool,
                    pool_chunksize=pool_chunksize,
                    pixel_backend=pixel_backend,
                    stencil_pixel_angle_rad=stencil_pixel_angle_rad,
                    include_legacy_pixel_arrays=include_legacy_pixel_arrays,
                )
                if batch is not None:
                    all_batches.append(batch)
                _log(
                    f"[paste:cpu] pixel batch halos {start - index_start:,}:{end - index_start:,} "
                    f"pairs={len(batch['nearby_pix_all']) if batch else 0:,} "
                    f"time={time.perf_counter() - t_batch:.1f}s",
                    verbose and log_batches,
                )
                _maybe_collect_pixel_gc()
    if not all_batches:
        return None
    total_pix = sum(len(batch["nearby_pix_all"]) for batch in all_batches)
    total_halos = sum(int(batch.get("n_halos", 0)) for batch in all_batches)
    total_single_pixel_shortcut = sum(int(batch.get("n_single_pixel_shortcut", 0)) for batch in all_batches)
    total_stencil = sum(int(batch.get("n_stencil", 0)) for batch in all_batches)
    total_ring = sum(int(batch.get("n_ring", 0)) for batch in all_batches)
    total_query_disc = sum(int(batch.get("n_query_disc", 0)) for batch in all_batches)
    total_query_disc_buffer_grows = sum(int(batch.get("n_query_disc_buffer_grows", 0)) for batch in all_batches)
    out = {
        "nearby_pix_all": np.empty(total_pix, dtype=pixel_dtype),
        "distances": np.empty(total_pix, dtype=np.float32),
        "logM": np.empty(total_pix, dtype=np.float32),
        "z": np.empty(total_pix, dtype=np.float32),
        "vlos": np.empty(total_pix, dtype=np.float32),
        "n_halos": int(total_halos),
        "n_single_pixel_shortcut": int(total_single_pixel_shortcut),
        "n_stencil": int(total_stencil),
        "n_ring": int(total_ring),
        "n_query_disc": int(total_query_disc),
        "n_query_disc_buffer_grows": int(total_query_disc_buffer_grows),
        "pixel_backend": str(pixel_backend),
        "query_disc_buffer_safety_factor": float(query_disc_buffer_safety_factor),
        "stencil_pixel_angle_factor": float(stencil_pixel_angle_factor),
        "include_legacy_pixel_arrays": bool(include_legacy_pixel_arrays),
    }
    if include_legacy_pixel_arrays:
        out.update(
            {
                "start_ind": np.empty(total_halos, dtype=np.int32),
                "end_ind": np.empty(total_halos, dtype=np.int32),
                "ang_distance_all": np.empty(total_halos, dtype=np.float32),
                "rp_max_all": np.empty(total_halos, dtype=np.float32),
            }
        )
    po = 0
    ho = 0
    for batch in all_batches:
        npix = len(batch["nearby_pix_all"])
        nh = int(batch.get("n_halos", 0))
        out["nearby_pix_all"][po : po + npix] = batch["nearby_pix_all"]
        out["distances"][po : po + npix] = batch["distances"]
        out["logM"][po : po + npix] = batch["logM"]
        out["z"][po : po + npix] = batch["z"]
        out["vlos"][po : po + npix] = batch["vlos"]
        if include_legacy_pixel_arrays:
            out["start_ind"][ho : ho + nh] = batch["start_ind"] + po
            out["end_ind"][ho : ho + nh] = batch["end_ind"] + po
            out["ang_distance_all"][ho : ho + nh] = batch["ang_distance_all"]
            out["rp_max_all"][ho : ho + nh] = batch["rp_max_all"]
        po += npix
        ho += nh
    batch = None
    del all_batches
    if pixel_gc_collect_every_n_batches > 0:
        gc.collect()
    if precompute_pixel_groups:
        pix_unique, sort_idx, boundaries = _precompute_pixel_grouping(out["nearby_pix_all"])
        out["pix_unique"] = pix_unique
        out["sort_idx"] = sort_idx
        out["boundaries"] = boundaries
    _log(
        f"[paste:cpu] pixel-neighbor build done halos={total_halos:,} pairs={total_pix:,} "
        f"stencil={total_stencil:,} ring={total_ring:,} "
        f"query_disc={total_query_disc:,} buffer_grows={total_query_disc_buffer_grows:,}",
        verbose,
    )
    return out


def create_pixel_worker_pool(
    catalog: Mapping[str, np.ndarray],
    nside: int,
    max_paint: float,
    workers: int,
    start_method: str,
    single_pixel_angle_factor: float = 0.0,
    pixel_backend: str = _PIXEL_BACKEND_HEALPY,
    query_disc_buffer_safety_factor: float = 2.0,
    verbose: bool = True,
):
    pixel_backend = _normalize_pixel_backend(pixel_backend)
    workers = auto_cpu_workers() if int(workers) <= 0 else int(workers)
    if workers <= 1:
        return None
    pixel_dtype = np.int32 if nside <= 8192 else np.int64
    ra = np.asarray(catalog["ra_deg"], dtype=np.float32)
    dec = np.asarray(catalog["dec_deg"], dtype=np.float32)
    r200c = np.asarray(catalog["R200c_hMpc"], dtype=np.float32)
    da = np.asarray(catalog["DA_hMpc"], dtype=np.float32)
    _log(
        f"[paste:cpu] initialize persistent pixel Pool workers={workers} "
        f"start_method={start_method} halos={len(ra):,} "
        f"single_pixel_angle_factor={float(single_pixel_angle_factor):.3g} "
        f"pixel_backend={pixel_backend}",
        verbose,
    )
    t0 = time.perf_counter()
    ctx = get_context(str(start_method))
    pool = ctx.Pool(
        processes=int(workers),
        initializer=_init_pixel_worker,
        initargs=(
            ra,
            dec,
            r200c,
            da,
            max_paint,
            nside,
            pixel_dtype,
            False,
            float(single_pixel_angle_factor),
            pixel_backend,
            query_disc_buffer_safety_factor,
        ),
    )
    _log(f"[paste:cpu] persistent pixel Pool initialized time={time.perf_counter() - t0:.1f}s", verbose)
    return pool


def _choose_pixel_pool_warmup_halo(
    r200c: np.ndarray,
    da: np.ndarray,
    max_paint: float,
    *,
    chunk_size: int = 1_000_000,
) -> int:
    n_halos = int(len(r200c))
    if n_halos <= 0:
        return 0
    best_index = 0
    best_angle = -np.inf
    for start in range(0, n_halos, int(chunk_size)):
        stop = min(start + int(chunk_size), n_halos)
        angle = float(max_paint) * np.asarray(r200c[start:stop], dtype=np.float64) / np.maximum(
            np.asarray(da[start:stop], dtype=np.float64),
            1.0e-8,
        )
        angle = np.nan_to_num(angle, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        local = int(np.argmax(angle))
        value = float(angle[local])
        if value > best_angle:
            best_angle = value
            best_index = start + local
    return int(best_index)


def warm_pixel_worker_pool(
    pool,
    catalog: Mapping[str, np.ndarray],
    max_paint: float,
    workers: int,
    *,
    verbose: bool = True,
) -> Dict[str, object]:
    if pool is None or int(workers) <= 0:
        return {
            "enabled": False,
            "time_s": 0.0,
            "n_worker_pids": 0,
            "n_tasks": 0,
            "warmup_halo_index": None,
        }
    r200c = np.asarray(catalog["R200c_hMpc"], dtype=np.float32)
    da = np.asarray(catalog["DA_hMpc"], dtype=np.float32)
    warmup_halo_index = _choose_pixel_pool_warmup_halo(r200c, da, float(max_paint))
    n_tasks = max(1, int(workers))
    _log(
        f"[paste:cpu] warm persistent pixel Pool start tasks={n_tasks} "
        f"warmup_halo_index={warmup_halo_index}",
        verbose,
    )
    t0 = time.perf_counter()
    results = pool.map(_warm_pixel_worker, [int(warmup_halo_index)] * n_tasks, chunksize=1)
    elapsed = time.perf_counter() - t0
    pids = sorted({int(result["pid"]) for result in results})
    n_query_disc = sum(1 for result in results if not bool(result["used_shortcut"]))
    n_buffer_grows = sum(int(result.get("buffer_grows", 0)) for result in results)
    npix_values = [int(result["npix"]) for result in results]
    _log(
        f"[paste:cpu] warm persistent pixel Pool done time={elapsed:.1f}s "
        f"worker_pids={len(pids)}/{int(workers)} query_disc_tasks={n_query_disc} "
        f"npix_min={min(npix_values) if npix_values else 0} npix_max={max(npix_values) if npix_values else 0} "
        f"buffer_grows={n_buffer_grows}",
        verbose,
    )
    return {
        "enabled": True,
        "time_s": float(elapsed),
        "n_worker_pids": int(len(pids)),
        "n_tasks": int(n_tasks),
        "warmup_halo_index": int(warmup_halo_index),
        "query_disc_tasks": int(n_query_disc),
        "buffer_grows": int(n_buffer_grows),
        "npix_min": int(min(npix_values)) if npix_values else 0,
        "npix_max": int(max(npix_values)) if npix_values else 0,
        "pids": [int(pid) for pid in pids],
    }


def split_indices_by_cost(catalog: Mapping[str, np.ndarray], nside: int, max_paint: float, num_splits: int) -> List[np.ndarray]:
    angle = max_paint * np.asarray(catalog["R200c_hMpc"], dtype=np.float64) / np.maximum(catalog["DA_hMpc"], 1.0e-8)
    pix_area = hp.nside2pixarea(nside)
    cost = np.maximum(1.0, math.pi * angle**2 / pix_area)
    order = np.argsort(cost)[::-1]
    loads = np.zeros(num_splits, dtype=np.float64)
    buckets: List[List[int]] = [[] for _ in range(num_splits)]
    for idx in order:
        target = int(np.argmin(loads))
        buckets[target].append(int(idx))
        loads[target] += cost[idx]
    return [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]


def contiguous_split_bounds(n_total: int, split_index: int, num_splits: int) -> Tuple[int, int]:
    n_total = int(n_total)
    split_index = int(split_index)
    num_splits = int(num_splits)
    if num_splits <= 0:
        raise ValueError(f"num_splits must be positive, got {num_splits}.")
    if split_index < 0 or split_index >= num_splits:
        raise ValueError(f"split_index must be in [0, {num_splits}), got {split_index}.")
    start = (n_total * split_index) // num_splits
    stop = (n_total * (split_index + 1)) // num_splits
    return int(start), int(stop)


def block_striped_split_ranges(n_total: int, split_index: int, num_splits: int, block_size: int) -> List[Tuple[int, int]]:
    n_total = int(n_total)
    split_index = int(split_index)
    num_splits = int(num_splits)
    block_size = max(1, int(block_size))
    if num_splits <= 0:
        raise ValueError(f"num_splits must be positive, got {num_splits}.")
    if split_index < 0 or split_index >= num_splits:
        raise ValueError(f"split_index must be in [0, {num_splits}), got {split_index}.")
    ranges: List[Tuple[int, int]] = []
    block_id = 0
    for start in range(0, n_total, block_size):
        stop = min(start + block_size, n_total)
        if block_id % num_splits == split_index:
            ranges.append((int(start), int(stop)))
        block_id += 1
    return ranges


def _chunk_size_for_nside(config: Mapping[str, object], nside: int) -> int:
    mapping = config["pasting"].get("chunk_halos_by_nside", {})
    if nside in mapping:
        return int(mapping[nside])
    if str(nside) in mapping:
        return int(mapping[str(nside)])
    return 50000


def _timing_path_for_map(path: Path | str) -> Path:
    path = Path(path)
    return path.with_suffix(path.suffix + ".timing.json")


def _write_json_atomic(path: Path | str, payload: Mapping[str, object]) -> None:
    path = Path(path)
    ensure_under_xdesi(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _empty_maps(nside: int) -> Dict[str, np.ndarray]:
    npix = 12 * int(nside) ** 2
    return {name: np.zeros(npix, dtype=np.float32) for name in MAP_DATASETS}


def _add_if_present(maps: Dict[str, np.ndarray], key: str, obj, attr: str) -> None:
    if hasattr(obj, attr):
        value = getattr(obj, attr)
        if isinstance(value, tuple) and len(value) == 2:
            pix, vals = value
            pix = np.asarray(pix, dtype=np.int64)
            vals = np.asarray(np.nan_to_num(vals), dtype=np.float32)
            maps[key][pix] += vals
        else:
            maps[key] += np.asarray(np.nan_to_num(value), dtype=np.float32)


def wl_source_bins_from_config(config: Mapping[str, object]) -> List[int]:
    raw = config.get("pasting", {}).get("source_bins_for_galaxy_cross", [1])
    if isinstance(raw, (int, np.integer)):
        values = [int(raw)]
    else:
        values = [int(value) for value in raw]
    bins = sorted({value for value in values if 1 <= value <= 4})
    return bins or [1]


def configure_jax_runtime_for_pasting(config: Mapping[str, object], verbose: bool = True) -> None:
    pasting = config.get("pasting", {})
    jax_cfg = pasting.get("jax", {}) if isinstance(pasting.get("jax", {}), Mapping) else {}
    preallocate = bool(jax_cfg.get("preallocate", pasting.get("jax_preallocate", True)))
    mem_fraction = float(jax_cfg.get("memory_fraction", pasting.get("jax_memory_fraction", 0.95)))
    platform = os.environ.get("PASTE_JAX_PLATFORMS", jax_cfg.get("platforms", pasting.get("jax_platforms", None)))
    cache_dir = os.environ.get(
        "PASTE_JAX_COMPILATION_CACHE_DIR",
        jax_cfg.get("compilation_cache_dir", pasting.get("jax_compilation_cache_dir", None)),
    )
    cache_min_compile_time = jax_cfg.get(
        "persistent_cache_min_compile_time_secs",
        pasting.get("jax_persistent_cache_min_compile_time_secs", None),
    )
    cache_min_entry_size = jax_cfg.get(
        "persistent_cache_min_entry_size_bytes",
        pasting.get("jax_persistent_cache_min_entry_size_bytes", None),
    )
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if preallocate else "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    if platform is not None:
        platform_str = str(platform).strip()
        if platform_str and platform_str.lower() not in {"auto", "none", "unset"}:
            os.environ["JAX_PLATFORMS"] = platform_str
        else:
            os.environ.pop("JAX_PLATFORMS", None)
    if cache_dir is not None:
        cache_dir_str = str(cache_dir).strip()
        if cache_dir_str and cache_dir_str.lower() not in {"auto", "none", "unset", "false"}:
            cache_path = Path(cache_dir_str).expanduser()
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
                os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_path)
            except Exception as exc:
                _log(f"[paste:jax] could not create JAX compilation cache dir {cache_path}: {exc}", verbose)
        else:
            os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
    if cache_min_compile_time is not None:
        os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = str(cache_min_compile_time)
    if cache_min_entry_size is not None:
        os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = str(cache_min_entry_size)
    _log(
        "[paste:jax] runtime env "
        f"XLA_PYTHON_CLIENT_PREALLOCATE={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')} "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')} "
        f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '<unset>')} "
        f"JAX_COMPILATION_CACHE_DIR={os.environ.get('JAX_COMPILATION_CACHE_DIR', '<unset>')}",
        verbose,
    )


def run_paste_split(
    config_path: Path | str,
    catalog_key: str,
    split_index: int,
    num_splits: int,
    nside: int,
    overwrite: bool = False,
    verbose: Optional[bool] = None,
    pixel_workers: Optional[int] = None,
    pixel_start_method: Optional[str] = None,
    pixel_backend: Optional[str] = None,
    query_disc_buffer_safety_factor: Optional[float] = None,
    profiles_class_path: Optional[str] = None,
) -> Path:
    wall0 = time.perf_counter()
    config = load_config(config_path)
    if verbose is None:
        verbose = bool(config.get("pasting", {}).get("verbose", True))
    configure_jax_runtime_for_pasting(config, verbose)
    _log(f"[paste] load config={config_path}", verbose)
    cat_path = catalog_path(config, catalog_key)
    max_paint = float(config["pasting"]["max_paint_R200c_factor"])
    split_strategy = str(config.get("pasting", {}).get("split_strategy", "contiguous")).strip().lower()
    t_select = time.perf_counter()
    split_idx = None
    split_start = None
    split_stop = None
    split_ranges: List[Tuple[int, int]] = []
    split_block_size = None
    if split_strategy in {"contiguous", "slice", "hdf5_slice"}:
        n_total_catalog, attrs = halo_catalog_size(cat_path)
        split_start, split_stop = contiguous_split_bounds(n_total_catalog, split_index, num_splits)
        split_ranges = [(split_start, split_stop)]
        _log(
            f"[paste:io] load contiguous catalog slice: {cat_path} "
            f"split={split_index}/{num_splits} rows={split_start:,}:{split_stop:,} "
            f"n_total={n_total_catalog:,}",
            verbose,
        )
        catalog, attrs = load_halo_catalog_slice(cat_path, split_start, split_stop)
        split_load_note = "contiguous HDF5 slice; avoids per-rank full-catalog load"
    elif split_strategy in {"block_striped", "striped", "block-stripe", "block_stripe"}:
        n_total_catalog, attrs = halo_catalog_size(cat_path)
        split_block_size = int(config.get("pasting", {}).get("split_block_halos", 250000))
        split_ranges = block_striped_split_ranges(n_total_catalog, split_index, num_splits, split_block_size)
        n_rows = sum(stop - start for start, stop in split_ranges)
        preview = split_ranges[:3]
        _log(
            f"[paste:io] load block-striped catalog slices: {cat_path} "
            f"split={split_index}/{num_splits} ranges={len(split_ranges):,} rows={n_rows:,} "
            f"block_size={split_block_size:,} first_ranges={preview}",
            verbose,
        )
        catalog, attrs = load_halo_catalog_ranges(cat_path, split_ranges)
        split_load_note = "block-striped HDF5 slices; balanced redshift mix without per-rank full-catalog load"
    elif split_strategy in {"cost", "cost_balance", "balanced"}:
        _log(f"[paste:io] load full catalog for cost-balanced split planning: {cat_path}", verbose)
        full_catalog, attrs = load_halo_catalog(cat_path)
        n_total_catalog = len(full_catalog["z"])
        _log(f"[paste:cpu] split catalog by cost n_halos={n_total_catalog:,} num_splits={num_splits}", verbose)
        splits = split_indices_by_cost(full_catalog, nside, max_paint, num_splits)
        split_idx = splits[int(split_index)]
        _log(f"[paste:cpu] materialize selected split in memory n_split={len(split_idx):,}", verbose)
        catalog = {key: np.asarray(value)[split_idx] for key, value in full_catalog.items()}
        del full_catalog
        split_load_note = "cost-balanced split; requires full-catalog load on each rank"
    else:
        raise ValueError(f"Unknown pasting.split_strategy={split_strategy!r}. Use 'contiguous' or 'cost'.")
    split_materialize_time_s = time.perf_counter() - t_select
    _log(f"[paste:io] selected split materialized time={split_materialize_time_s:.1f}s", verbose)
    t_post_split_gc = time.perf_counter()
    gc.collect()
    post_split_gc_time_s = time.perf_counter() - t_post_split_gc
    _log(f"[paste] catalog={catalog_key} split={split_index}/{num_splits} halos={len(catalog['z']):,}", verbose)

    chunk_size = _chunk_size_for_nside(config, nside)
    pixel_batch_size = int(config["pasting"].get("pixel_batch_size", 2000))
    pixel_gc_collect_every_n_batches = int(config["pasting"].get("pixel_gc_collect_every_n_batches", 0))
    pixel_pool_chunksize = int(config["pasting"].get("pixel_pool_chunksize", 0))
    single_pixel_angle_factor = float(config["pasting"].get("single_pixel_angle_factor", 0.0))
    stencil_pixel_angle_factor = float(config["pasting"].get("stencil_pixel_angle_factor", 1.0))
    if pixel_workers is None:
        pixel_workers = int(config["pasting"].get("pixel_workers", 0))
    if pixel_start_method is None:
        pixel_start_method = str(config["pasting"].get("pixel_start_method", "fork"))
    persistent_pixel_pool = bool(config["pasting"].get("persistent_pixel_pool", True))
    pixel_pool_warmup = bool(config["pasting"].get("pixel_pool_warmup", True))
    pixel_log_batches = bool(config["pasting"].get("pixel_log_batches", False))
    pixel_backend = _normalize_pixel_backend(
        pixel_backend if pixel_backend is not None else config["pasting"].get("pixel_backend", _PIXEL_BACKEND_HEALPY)
    )
    query_disc_buffer_safety_factor = float(
        query_disc_buffer_safety_factor
        if query_disc_buffer_safety_factor is not None
        else config["pasting"].get("query_disc_buffer_safety_factor", 2.0)
    )
    include_legacy_pixel_arrays = bool(config["pasting"].get("include_legacy_pixel_arrays", False))
    jax_clear_caches_every = int(config["pasting"].get("jax_clear_caches_every_n_chunks", 1))
    pixel_prefetch_next_chunk = bool(config["pasting"].get("pixel_prefetch_next_chunk", False))
    _log(
        f"[paste] chunk_size={chunk_size:,} pixel_batch_size={pixel_batch_size:,} "
        f"pixel_gc_collect_every_n_batches={pixel_gc_collect_every_n_batches} "
        f"pixel_pool_chunksize={pixel_pool_chunksize:,} "
        f"single_pixel_angle_factor={single_pixel_angle_factor:.3g} "
        f"stencil_pixel_angle_factor={stencil_pixel_angle_factor:.3g} "
        f"pixel_workers={pixel_workers} pixel_start_method={pixel_start_method} "
        f"persistent_pixel_pool={persistent_pixel_pool} pixel_pool_warmup={pixel_pool_warmup} "
        f"pixel_log_batches={pixel_log_batches} "
        f"pixel_backend={pixel_backend} "
        f"jax_clear_caches_every_n_chunks={jax_clear_caches_every} "
        f"pixel_prefetch_next_chunk={pixel_prefetch_next_chunk}",
        verbose,
    )
    pixel_pool = None
    pixel_pool_setup_time_s = 0.0
    pixel_pool_create_time_s = 0.0
    pixel_pool_warmup_result: Dict[str, object] = {
        "enabled": False,
        "time_s": 0.0,
        "n_worker_pids": 0,
        "n_tasks": 0,
        "warmup_halo_index": None,
    }
    if persistent_pixel_pool and int(pixel_workers) != 1 and pixel_backend != _PIXEL_BACKEND_HEALPY_RING:
        t_pixel_pool_setup = time.perf_counter()
        t_pixel_pool_create = time.perf_counter()
        pixel_pool = create_pixel_worker_pool(
            catalog,
            nside,
            max_paint,
            int(pixel_workers),
            pixel_start_method,
            single_pixel_angle_factor,
            pixel_backend,
            query_disc_buffer_safety_factor,
            verbose,
        )
        pixel_pool_create_time_s = time.perf_counter() - t_pixel_pool_create
        if pixel_pool_warmup:
            pixel_pool_warmup_result = warm_pixel_worker_pool(
                pixel_pool,
                catalog,
                max_paint,
                int(pixel_workers),
                verbose=verbose,
            )
        pixel_pool_setup_time_s = time.perf_counter() - t_pixel_pool_setup

    _log("[paste:jax] import JAX/GODMAX modules", verbose)
    t_jax_import = time.perf_counter()
    import jax
    import jax.numpy as jnp
    from base_class import base_class
    from get_radial_profiles import Profiles as NativeProfiles
    from get_sim_maps import setup_sim_map, get_sim_map
    profiles_class = NativeProfiles
    if profiles_class_path is not None:
        module_name, separator, class_name = str(profiles_class_path).rpartition(".")
        if not separator or not module_name or not class_name:
            raise ValueError(
                "profiles_class_path must be a fully qualified module.Class name; "
                f"got {profiles_class_path!r}."
            )
        profiles_class = getattr(importlib.import_module(module_name), class_name)
        if not isinstance(profiles_class, type) or not issubclass(
            profiles_class, NativeProfiles
        ):
            raise TypeError(
                f"{profiles_class_path} is not a get_radial_profiles.Profiles subclass."
            )
    profiles_class_fqname = (
        f"{profiles_class.__module__}.{profiles_class.__qualname__}"
    )
    jax_module_import_time_s = time.perf_counter() - t_jax_import
    jax_cache_config_time_s = 0.0
    jax_cache_config_error = None
    jax_cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if jax_cache_dir:
        t_jax_cache_config = time.perf_counter()
        try:
            jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
        except Exception as exc:
            jax_cache_config_error = str(exc)
            _log(f"[paste:jax] could not update jax_compilation_cache_dir={jax_cache_dir}: {exc}", verbose)
        jax_cache_config_time_s = time.perf_counter() - t_jax_cache_config

    _log(f"[paste:jax] devices={jax.devices()}", verbose)

    _log("[paste:godmax] prepare Stage-31/GODMAX parameter dictionaries", verbose)
    t_prepare_godmax = time.perf_counter()
    sim_params, halo_params, analysis, other_params = prepare_godmax_config(
        config,
        attrs,
        is_cmb_lensing=False,
        z_max=float(attrs.get("z_max", np.max(catalog["z"]) if len(catalog["z"]) else 0.5)),
        log10_mass_min=float(attrs.get("log10_m_min_hmsun", np.min(catalog["log10M200c_hMsun"]))),
    )
    godmax_prepare_config_time_s = time.perf_counter() - t_prepare_godmax
    _log("[paste:godmax] instantiate base_class", verbose)
    t_base_class = time.perf_counter()
    base = base_class(sim_params, halo_params, analysis, other_params)
    base_class_time_s = time.perf_counter() - t_base_class
    _log("[paste:godmax] instantiate Profiles", verbose)
    t_profiles = time.perf_counter()
    profiles = profiles_class(
        sim_params,
        halo_params,
        analysis,
        other_params,
        base_class_obj=base,
    )
    profiles_time_s = time.perf_counter() - t_profiles

    get_kappa_wl = bool(config["pasting"].get("get_kappa_wl", True))
    wl_source_bins = wl_source_bins_from_config(config) if get_kappa_wl else []
    store_projected_matter_maps = bool(config["pasting"].get("store_projected_matter_maps", True))
    get_kappa_cmb = bool(config["pasting"].get("get_kappa_cmb", True))
    use_multi_kappa_maps = bool(config["pasting"].get("use_multi_kappa_maps", False))
    multi_kappa_source_bins = [int(source_bin) - 1 for source_bin in wl_source_bins] if use_multi_kappa_maps else []
    _log(f"[paste:godmax] DES WL source bins requested={wl_source_bins}", verbose)
    setup_params = {
        "nside": int(nside),
        "smooth_profiles": bool(config["pasting"].get("smooth_profiles", True)),
        "profile_timing": bool(config["pasting"].get("profile_timing", False)),
        "use_fused_profile_maps": bool(config["pasting"].get("use_fused_profile_maps", True)),
        "return_sparse_maps": bool(config["pasting"].get("return_sparse_maps", True)),
        "store_projected_matter_maps": store_projected_matter_maps,
        "galaxy_population_chunk_size": int(config["pasting"].get("galaxy_population_chunk_size", 20000)),
        "galaxy_max_gals_round_to": int(config["pasting"].get("galaxy_max_gals_round_to", 16)),
        "galaxy_population_group_by_max_gals": bool(config["pasting"].get("galaxy_population_group_by_max_gals", False)),
        "galaxy_population_backend": str(config["pasting"].get("galaxy_population_backend", "padded_precomputed")),
        "galaxy_compact_max_satellite_groups": int(config["pasting"].get("galaxy_compact_max_satellite_groups", 32)),
        "get_galmap": bool(config["pasting"].get("get_galmap", True)),
        "get_ymap": bool(config["pasting"].get("get_ymap", True)),
        "get_kSZmap": bool(config["pasting"].get("get_kszmap", True)),
        "get_taumap": bool(config["pasting"].get("get_taumap", True)),
        "get_kappamap": bool((not use_multi_kappa_maps) and get_kappa_wl and 1 in wl_source_bins),
        "get_multi_kappamap": bool(use_multi_kappa_maps and (multi_kappa_source_bins or get_kappa_cmb)),
        "multi_kappa_source_bins": multi_kappa_source_bins,
        "multi_kappa_include_cmb": bool(use_multi_kappa_maps and get_kappa_cmb),
        "get_baryonifiedmap": bool(config["pasting"].get("get_baryonifiedmap", store_projected_matter_maps)),
        "kappa_source_bin": 0,
    }
    _log(f"[paste:jax] setup_sim_map start params={json.dumps(setup_params, sort_keys=True)}", verbose)
    t_setup_main = time.perf_counter()
    setup = setup_sim_map(sim_params, halo_params, analysis, other_params, setup_params, Profiles_obj=profiles)
    setup_sim_map_main_time_s = time.perf_counter() - t_setup_main
    setup_sim_map_main_profile_s = copy.deepcopy(getattr(setup, "timing_results", {}))
    _log("[paste:jax] setup_sim_map done", verbose)
    extra_wl_setups = {}
    setup_sim_map_wl_time_s: Dict[str, float] = {}
    setup_sim_map_wl_profile_s: Dict[str, object] = {}
    for source_bin in ([] if use_multi_kappa_maps else [value for value in wl_source_bins if value != 1]):
        wl_setup_params = dict(setup_params)
        wl_setup_params.update(
            {
                "get_galmap": False,
                "get_ymap": False,
                "get_kSZmap": False,
                "get_taumap": False,
                "get_kappamap": True,
                "get_baryonifiedmap": False,
                "kappa_source_bin": int(source_bin) - 1,
            }
        )
        _log(f"[paste:jax] setup_sim_map start DES WL source bin {source_bin}", verbose)
        t_setup_wl = time.perf_counter()
        wl_setup = setup_sim_map(sim_params, halo_params, analysis, other_params, wl_setup_params, Profiles_obj=setup)
        setup_sim_map_wl_time_s[str(source_bin)] = float(time.perf_counter() - t_setup_wl)
        setup_sim_map_wl_profile_s[str(source_bin)] = copy.deepcopy(getattr(wl_setup, "timing_results", {}))
        extra_wl_setups[int(source_bin)] = (wl_setup_params, wl_setup)
        _log(f"[paste:jax] setup_sim_map done DES WL source bin {source_bin}", verbose)
    cmb_setup = None
    cmb_setup_params = None
    setup_sim_map_cmb_time_s = None
    setup_sim_map_cmb_profile_s: Dict[str, object] = {}
    if get_kappa_cmb and not use_multi_kappa_maps:
        cmb_profiles = copy.copy(setup)
        cmb_profiles.is_cmb_lensing = True
        cmb_setup_params = dict(setup_params)
        cmb_setup_params.update(
            {
                "get_galmap": False,
                "get_ymap": False,
                "get_kSZmap": False,
                "get_taumap": False,
                "get_kappamap": True,
                "get_baryonifiedmap": False,
            }
        )
        _log("[paste:jax] setup_sim_map start: CMB kappa with CMB lensing kernel", verbose)
        t_setup_cmb = time.perf_counter()
        cmb_setup = setup_sim_map(sim_params, halo_params, analysis, other_params, cmb_setup_params, Profiles_obj=cmb_profiles)
        setup_sim_map_cmb_time_s = float(time.perf_counter() - t_setup_cmb)
        setup_sim_map_cmb_profile_s = copy.deepcopy(getattr(cmb_setup, "timing_results", {}))
        _log("[paste:jax] setup_sim_map done: CMB kappa", verbose)

    t_maps_allocate = time.perf_counter()
    maps = _empty_maps(nside)
    if not store_projected_matter_maps:
        for name in ("map_rhom_dmb", "map_rhom_dmo", "map_rhom"):
            maps.pop(name, None)
    maps_allocate_time_s = time.perf_counter() - t_maps_allocate
    galaxies = []
    n_halos = len(catalog["z"])
    pre_chunk_setup_time_s = time.perf_counter() - (t_select + split_materialize_time_s)
    legacy_split_materialize_plus_setup_time_s = time.perf_counter() - t_select
    timing: Dict[str, object] = {
        "config_path": str(config_path),
        "catalog_key": str(catalog_key),
        "split_index": int(split_index),
        "num_splits": int(num_splits),
        "nside": int(nside),
        "n_total_catalog_halos": int(n_total_catalog),
        "n_halos": int(n_halos),
        "split_strategy": str(split_strategy),
        "split_load_note": str(split_load_note),
        "split_start": None if split_start is None else int(split_start),
        "split_stop": None if split_stop is None else int(split_stop),
        "split_block_halos": None if split_block_size is None else int(split_block_size),
        "split_n_ranges": int(len(split_ranges)),
        "split_ranges_head": [(int(start), int(stop)) for start, stop in split_ranges[:8]],
        "split_materialize_time_s": float(split_materialize_time_s),
        "hdf5_read_time_s": float(split_materialize_time_s),
        "post_split_gc_time_s": float(post_split_gc_time_s),
        "legacy_split_materialize_plus_setup_time_s": float(legacy_split_materialize_plus_setup_time_s),
        "pre_chunk_setup_time_s": float(pre_chunk_setup_time_s),
        "pixel_pool_create_time_s": float(pixel_pool_create_time_s),
        "pixel_pool_setup_time_s": float(pixel_pool_setup_time_s),
        "jax_module_import_time_s": float(jax_module_import_time_s),
        "jax_compilation_cache_dir": str(jax_cache_dir) if jax_cache_dir else "",
        "jax_cache_config_time_s": float(jax_cache_config_time_s),
        "jax_cache_config_error": "" if jax_cache_config_error is None else str(jax_cache_config_error),
        "godmax_prepare_config_time_s": float(godmax_prepare_config_time_s),
        "base_class_time_s": float(base_class_time_s),
        "profiles_time_s": float(profiles_time_s),
        "profiles_class_fqname": profiles_class_fqname,
        "setup_sim_map_main_time_s": float(setup_sim_map_main_time_s),
        "setup_sim_map_main_profile_s": setup_sim_map_main_profile_s,
        "setup_sim_map_wl_time_s": setup_sim_map_wl_time_s,
        "setup_sim_map_wl_profile_s": setup_sim_map_wl_profile_s,
        "setup_sim_map_cmb_time_s": None if setup_sim_map_cmb_time_s is None else float(setup_sim_map_cmb_time_s),
        "setup_sim_map_cmb_profile_s": setup_sim_map_cmb_profile_s,
        "maps_allocate_time_s": float(maps_allocate_time_s),
        "chunk_size": int(chunk_size),
        "pixel_batch_size": int(pixel_batch_size),
        "pixel_gc_collect_every_n_batches": int(pixel_gc_collect_every_n_batches),
        "pixel_pool_chunksize": int(pixel_pool_chunksize),
        "single_pixel_angle_factor": float(single_pixel_angle_factor),
        "stencil_pixel_angle_factor": float(stencil_pixel_angle_factor),
        "pixel_workers": int(pixel_workers),
        "pixel_start_method": str(pixel_start_method),
        "pixel_backend": str(pixel_backend),
        "query_disc_buffer_safety_factor": float(query_disc_buffer_safety_factor),
        "include_legacy_pixel_arrays": bool(include_legacy_pixel_arrays),
        "persistent_pixel_pool": bool(persistent_pixel_pool),
        "pixel_pool_warmup": bool(pixel_pool_warmup),
        "pixel_pool_warmup_result": pixel_pool_warmup_result,
        "pixel_log_batches": bool(pixel_log_batches),
        "jax_clear_caches_every_n_chunks": int(jax_clear_caches_every),
        "pixel_prefetch_next_chunk": bool(pixel_prefetch_next_chunk),
        "max_paint_R200c_factor": float(max_paint),
        "store_projected_matter_maps": bool(store_projected_matter_maps),
        "use_multi_kappa_maps": bool(use_multi_kappa_maps),
        "multi_kappa_source_bins": [int(value) for value in multi_kappa_source_bins],
        "multi_kappa_include_cmb": bool(use_multi_kappa_maps and get_kappa_cmb),
        "chunks": [],
    }
    def _build_pixels_for_chunk(chunk_id: int, start: int, end: int):
        _log(f"[paste] chunk {chunk_id + 1}: halos {start:,}:{end:,}", verbose)
        chunk_for_pixels = catalog if pixel_pool is not None else {key: value[start:end] for key, value in catalog.items()}
        t0 = time.perf_counter()
        pixels_out = build_pixel_work_package(
            chunk_for_pixels,
            nside,
            max_paint,
            pixel_batch_size,
            workers=pixel_workers,
            start_method=pixel_start_method,
            pool=pixel_pool,
            pool_chunksize=pixel_pool_chunksize,
            single_pixel_angle_factor=single_pixel_angle_factor,
            index_start=start if pixel_pool is not None else 0,
            index_end=end if pixel_pool is not None else None,
            verbose=verbose,
            log_batches=pixel_log_batches,
            pixel_backend=pixel_backend,
            query_disc_buffer_safety_factor=query_disc_buffer_safety_factor,
            stencil_pixel_angle_factor=stencil_pixel_angle_factor,
            include_legacy_pixel_arrays=include_legacy_pixel_arrays,
            precompute_pixel_groups=True,
            pixel_gc_collect_every_n_batches=pixel_gc_collect_every_n_batches,
        )
        pixel_time_out = time.perf_counter() - t0
        n_pairs_out = len(pixels_out["nearby_pix_all"]) if pixels_out else 0
        n_single_pixel_shortcut_out = int(pixels_out.get("n_single_pixel_shortcut", 0)) if pixels_out else 0
        n_ring_out = int(pixels_out.get("n_ring", 0)) if pixels_out else 0
        n_query_disc_out = int(pixels_out.get("n_query_disc", 0)) if pixels_out else 0
        _log(
            f"[paste:cpu] pixel work pairs={n_pairs_out:,} "
            f"single_pixel_shortcut={n_single_pixel_shortcut_out:,} "
            f"ring={n_ring_out:,} query_disc={n_query_disc_out:,} time={pixel_time_out:.1f}s",
            verbose,
        )
        return pixels_out, pixel_time_out, n_pairs_out, n_single_pixel_shortcut_out

    chunk_ranges = [
        (chunk_id, start, min(start + chunk_size, n_halos))
        for chunk_id, start in enumerate(range(0, n_halos, chunk_size))
    ]
    prefetch_enabled = bool(pixel_prefetch_next_chunk and pixel_pool is not None and len(chunk_ranges) > 1)
    pixel_executor = ThreadPoolExecutor(max_workers=1) if prefetch_enabled else None
    try:
        pixel_future = None
        if pixel_executor is not None and chunk_ranges:
            first_chunk_id, first_start, first_end = chunk_ranges[0]
            pixel_future = pixel_executor.submit(_build_pixels_for_chunk, first_chunk_id, first_start, first_end)
        for range_pos, (chunk_id, start, end) in enumerate(chunk_ranges):
            chunk = {key: value[start:end] for key, value in catalog.items()}
            if pixel_future is not None:
                pixels, pixel_time, n_pairs, n_single_pixel_shortcut = pixel_future.result()
                pixel_future = None
            else:
                pixels, pixel_time, n_pairs, n_single_pixel_shortcut = _build_pixels_for_chunk(chunk_id, start, end)
            if pixel_executor is not None and range_pos + 1 < len(chunk_ranges):
                next_chunk_id, next_start, next_end = chunk_ranges[range_pos + 1]
                pixel_future = pixel_executor.submit(_build_pixels_for_chunk, next_chunk_id, next_start, next_end)
            if pixels is None:
                timing["chunks"].append(
                    {
                        "chunk_id": int(chunk_id),
                        "halo_start": int(start),
                        "halo_end": int(end),
                        "n_halos": int(end - start),
                        "n_pairs": 0,
                        "n_ring": 0,
                        "n_query_disc": 0,
                        "n_query_disc_buffer_grows": 0,
                        "pixel_backend": str(pixel_backend),
                        "pixel_time_s": float(pixel_time),
                        "gpu_main_time_s": None,
                        "gpu_wl_extra_time_s": {},
                        "gpu_cmb_time_s": None,
                        "total_elapsed_s": float(time.perf_counter() - wall0),
                    }
                )
                continue
            n_query_disc = int(pixels.get("n_query_disc", 0))
            n_ring = int(pixels.get("n_ring", 0))
            n_query_disc_buffer_grows = int(pixels.get("n_query_disc_buffer_grows", 0))

            _log("[paste:gpu] transfer chunk arrays to JAX device", verbose)
            t_transfer = time.perf_counter()
            mock_params = dict(setup_params)
            pix_prop_all = np.column_stack(
                (
                    np.log(pixels["distances"]),
                    pixels["z"],
                    pixels["logM"],
                    pixels["vlos"],
                )
            ).astype(np.float32, copy=False)
            mock_params.update(
                {
                    "halo_z": jnp.array(chunk["z"], dtype=jnp.float32),
                    "halo_ra": jnp.array(chunk["ra_deg"], dtype=jnp.float32),
                    "halo_dec": jnp.array(chunk["dec_deg"], dtype=jnp.float32),
                    "halo_M": jnp.array(chunk["M200c_hMsun"], dtype=jnp.float64),
                    "halo_DA": jnp.array(chunk["DA_hMpc"], dtype=jnp.float32),
                    "halo_vlos": jnp.array(chunk["vlos_kms"], dtype=jnp.float32),
                    "nearby_pix_all": pixels["nearby_pix_all"],
                    "pix_unique": pixels.get("pix_unique"),
                    "sort_idx": pixels.get("sort_idx"),
                    "boundaries": pixels.get("boundaries"),
                    "pix_prop_all": jnp.array(pix_prop_all, dtype=jnp.float32),
                    "random_seed": int(config["pasting"].get("random_seed", 42)) + int(split_index) * 100000 + chunk_id,
                }
            )
            transfer_time = time.perf_counter() - t_transfer
            _log("[paste:gpu] get_sim_map start: galaxy/tSZ/kSZ/tau/WL-kappa/baryonified maps", verbose)
            t_gpu = time.perf_counter()
            mock_map = get_sim_map(sim_params, halo_params, analysis, other_params, mock_params, Profiles_obj=setup)
            gpu_main_time = time.perf_counter() - t_gpu
            gpu_main_profile = copy.deepcopy(getattr(mock_map, "timing_results", {}))
            galaxy_population_diagnostics = copy.deepcopy(getattr(mock_map, "galaxy_population_diagnostics", {}))
            _log(f"[paste:gpu] get_sim_map done time={gpu_main_time:.1f}s", verbose)
            if store_projected_matter_maps:
                _add_if_present(maps, "map_rhom_dmb", mock_map, "rhommap_final")
                _add_if_present(maps, "map_rhom", mock_map, "rhommap_final")
                _add_if_present(maps, "map_rhom_dmo", mock_map, "rhom_dmo_map_final")
            _add_if_present(maps, "map_ymap", mock_map, "ymap_final")
            _add_if_present(maps, "map_ksz", mock_map, "kszmap_final")
            _add_if_present(maps, "map_tau", mock_map, "taumap_final")
            _add_if_present(maps, "map_kappa_wl", mock_map, "kappamap_final")
            if use_multi_kappa_maps and hasattr(mock_map, "multi_kappamaps_final"):
                multi_maps = getattr(mock_map, "multi_kappamaps_final")
                for source_bin in wl_source_bins:
                    label = f"source_bin_{int(source_bin) - 1}"
                    dataset = "map_kappa_wl" if int(source_bin) == 1 else f"map_kappa_wl_tomo{int(source_bin)}"
                    if label in multi_maps:
                        value = multi_maps[label]
                        if isinstance(value, tuple) and len(value) == 2:
                            pix, vals = value
                            maps[dataset][np.asarray(pix, dtype=np.int64)] += np.asarray(np.nan_to_num(vals), dtype=np.float32)
                        else:
                            maps[dataset] += np.asarray(np.nan_to_num(value), dtype=np.float32)
                if get_kappa_cmb and "cmb" in multi_maps:
                    value = multi_maps["cmb"]
                    if isinstance(value, tuple) and len(value) == 2:
                        pix, vals = value
                        maps["map_kappa_cmb"][np.asarray(pix, dtype=np.int64)] += np.asarray(np.nan_to_num(vals), dtype=np.float32)
                    else:
                        maps["map_kappa_cmb"] += np.asarray(np.nan_to_num(value), dtype=np.float32)
            if hasattr(mock_map, "final_galaxy_catalog"):
                galaxies.append(np.asarray(mock_map.final_galaxy_catalog, dtype=np.float32))

            wl_times: Dict[str, float] = {}
            wl_profiles: Dict[str, object] = {}
            for source_bin, (wl_setup_params, wl_setup) in extra_wl_setups.items():
                wl_params = dict(mock_params)
                wl_params.update(wl_setup_params)
                _log(f"[paste:gpu] get_sim_map start: DES WL kappa source bin {source_bin}", verbose)
                t_wl = time.perf_counter()
                wl_map = get_sim_map(sim_params, halo_params, analysis, other_params, wl_params, Profiles_obj=wl_setup)
                wl_time = time.perf_counter() - t_wl
                wl_times[str(source_bin)] = float(wl_time)
                wl_profiles[str(source_bin)] = copy.deepcopy(getattr(wl_map, "timing_results", {}))
                _log(f"[paste:gpu] DES WL kappa source bin {source_bin} done time={wl_time:.1f}s", verbose)
                _add_if_present(maps, f"map_kappa_wl_tomo{source_bin}", wl_map, "kappamap_final")
                del wl_map, wl_params

            cmb_time = None
            cmb_profile = {}
            if cmb_setup is not None and cmb_setup_params is not None:
                _log("[paste:gpu] get_sim_map start: CMB kappa", verbose)
                cmb_params = dict(mock_params)
                cmb_params.update(cmb_setup_params)
                t_cmb = time.perf_counter()
                cmb_map = get_sim_map(sim_params, halo_params, analysis, other_params, cmb_params, Profiles_obj=cmb_setup)
                cmb_time = time.perf_counter() - t_cmb
                cmb_profile = copy.deepcopy(getattr(cmb_map, "timing_results", {}))
                _log(f"[paste:gpu] CMB kappa done time={cmb_time:.1f}s", verbose)
                _add_if_present(maps, "map_kappa_cmb", cmb_map, "kappamap_final")
                del cmb_map, cmb_params

            del mock_map, mock_params, pixels
            if jax_clear_caches_every > 0 and (chunk_id + 1) % jax_clear_caches_every == 0:
                jax.clear_caches()
            gc.collect()
            elapsed = time.perf_counter() - wall0
            timing["chunks"].append(
                {
                    "chunk_id": int(chunk_id),
                    "halo_start": int(start),
                    "halo_end": int(end),
                    "n_halos": int(end - start),
                    "n_pairs": int(n_pairs),
                    "n_single_pixel_shortcut": int(n_single_pixel_shortcut),
                    "n_ring": int(n_ring),
                    "n_query_disc": int(n_query_disc),
                    "n_query_disc_buffer_grows": int(n_query_disc_buffer_grows),
                    "pixel_backend": str(pixel_backend),
                    "pairs_per_halo": float(n_pairs / max(1, end - start)),
                    "pixel_time_s": float(pixel_time),
                    "transfer_time_s": float(transfer_time),
                    "gpu_main_time_s": float(gpu_main_time),
                    "gpu_main_profile_s": gpu_main_profile,
                    "galaxy_population_diagnostics": galaxy_population_diagnostics,
                    "gpu_wl_extra_time_s": wl_times,
                    "gpu_wl_extra_profile_s": wl_profiles,
                    "gpu_cmb_time_s": None if cmb_time is None else float(cmb_time),
                    "gpu_cmb_profile_s": cmb_profile,
                    "total_elapsed_s": float(elapsed),
                }
            )
            _log(f"[paste] chunk {chunk_id + 1} done elapsed_total={elapsed:.1f}s", verbose)
    finally:
        if pixel_executor is not None:
            pixel_executor.shutdown(wait=True)
        if pixel_pool is not None:
            _log("[paste:cpu] close persistent pixel Pool", verbose)
            pixel_pool.close()
            pixel_pool.join()

    galaxy_catalog = np.concatenate(galaxies, axis=0) if galaxies else np.empty((0, 7), dtype=np.float32)
    out_path = partial_map_path(config, catalog_key, nside, split_index, num_splits)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it.")
    t_write = time.perf_counter()
    write_maps_h5(
        out_path,
        maps,
        galaxy_catalog,
        {
            "catalog_key": catalog_key,
            "catalog_path": str(cat_path),
            "split_index": int(split_index),
            "num_splits": int(num_splits),
            "nside": int(nside),
            "n_input_halos": int(n_total_catalog),
            "n_split_halos": int(n_halos),
            "split_strategy": str(split_strategy),
            "split_start": -1 if split_start is None else int(split_start),
            "split_stop": -1 if split_stop is None else int(split_stop),
            "split_block_halos": -1 if split_block_size is None else int(split_block_size),
            "split_n_ranges": int(len(split_ranges)),
            "split_load_note": str(split_load_note),
            "max_paint_R200c_factor": max_paint,
            "smooth_profiles": bool(config["pasting"].get("smooth_profiles", True)),
            "profiles_class_fqname": profiles_class_fqname,
            "wl_source_bins_json": json.dumps(wl_source_bins),
            "wl_source_bin_datasets_json": json.dumps(
                {str(source_bin): ("map_kappa_wl" if int(source_bin) == 1 else f"map_kappa_wl_tomo{int(source_bin)}") for source_bin in wl_source_bins}
            ),
            "source_nz": str(
                config["godmax"].get(
                    "source_nz_fits",
                    config["godmax"].get("map_h5", config["godmax"].get("comparison_config", "")),
                )
            ),
            "galaxy_catalog_columns_json": json.dumps(
                [
                    "ra_deg",
                    "dec_deg",
                    "z",
                    "host_M200c_hMsun",
                    "is_central",
                    "valid",
                    "host_vlos_kms",
                ]
            ),
        },
    )
    timing["write_h5_time_s"] = float(time.perf_counter() - t_write)
    timing["total_time_s"] = float(time.perf_counter() - wall0)
    timing["output_h5"] = str(out_path)
    timing["n_galaxies"] = int(len(galaxy_catalog))
    timing_path = _timing_path_for_map(out_path)
    _write_json_atomic(timing_path, timing)
    _log(f"[paste] wrote {out_path} total_time={timing['total_time_s']:.1f}s timing={timing_path}", verbose)
    return out_path


def write_maps_h5(path: Path | str, maps: Mapping[str, np.ndarray], galaxies: np.ndarray, attrs: Mapping[str, object]) -> None:
    path = Path(path)
    ensure_under_xdesi(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as handle:
        group = handle.create_group("maps")
        for key, value in maps.items():
            group.create_dataset(key, data=np.asarray(value, dtype=np.float32), compression="lzf")
        handle.create_dataset("galaxies", data=np.asarray(galaxies, dtype=np.float32), compression="lzf")
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                handle.attrs[key] = json.dumps(value)
            else:
                handle.attrs[key] = value
        handle.attrs["n_galaxies"] = int(len(galaxies))
    os.replace(tmp_path, path)


def load_maps_h5(path: Path | str) -> Tuple[dict, np.ndarray, dict]:
    with h5py.File(path, "r") as handle:
        maps = {key: handle["maps"][key][:] for key in handle["maps"]}
        galaxies = handle["galaxies"][:]
        attrs = dict(handle.attrs)
    return maps, galaxies, attrs


def combine_partial_maps(config_path: Path | str, catalog_key: str, num_splits: int, nside: int, overwrite: bool = False) -> Path:
    config = load_config(config_path)
    paths = [partial_map_path(config, catalog_key, nside, split, num_splits) for split in range(num_splits)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing partial map files:\n" + "\n".join(missing))
    maps_sum = None
    galaxies = []
    attrs = {}
    for path in paths:
        maps, gals, part_attrs = load_maps_h5(path)
        if maps_sum is None:
            maps_sum = {key: np.zeros_like(value, dtype=np.float32) for key, value in maps.items()}
        for key, value in maps.items():
            maps_sum[key] += value.astype(np.float32)
        if len(gals):
            galaxies.append(gals.astype(np.float32))
        attrs = dict(part_attrs)
    final_gals = np.concatenate(galaxies, axis=0) if galaxies else np.empty((0, 7), dtype=np.float32)
    out_path = final_map_path(config, catalog_key, nside)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it.")
    attrs.update({"combined_from_num_splits": int(num_splits), "split_files_json": json.dumps([str(p) for p in paths])})
    write_maps_h5(out_path, maps_sum, final_gals, attrs)
    print(f"[combine] wrote {out_path}")
    return out_path


def make_galaxy_overdensity_map(galaxies: np.ndarray, nside: int, z_range: Optional[Tuple[float, float]] = None):
    if galaxies.size == 0:
        npix = 12 * nside**2
        return np.zeros(npix, dtype=np.float32), np.nan, 0
    valid = galaxies[:, 5] > 0.5
    if z_range is not None:
        valid &= (galaxies[:, 2] >= z_range[0]) & (galaxies[:, 2] < z_range[1])
    gals = galaxies[valid]
    npix = 12 * nside**2
    gmap = np.zeros(npix, dtype=np.float32)
    if len(gals):
        pix = hp.ang2pix(nside, gals[:, 0], gals[:, 1], lonlat=True)
        np.add.at(gmap, pix, 1.0)
    mean = float(np.mean(gmap))
    delta = gmap / mean - 1.0 if mean > 0 else gmap
    return delta.astype(np.float32), mean, int(len(gals))


def measure_basic_cls(maps: Mapping[str, np.ndarray], galaxies: np.ndarray, nside: int, z_range: Optional[Tuple[float, float]] = None):
    lmax = 3 * nside - 1
    delta_g, mean_g, n_gal = make_galaxy_overdensity_map(galaxies, nside, z_range)
    pixwin = hp.pixwin(nside, lmax=lmax)
    pixwin2 = np.maximum(pixwin**2, 1.0e-30)
    out = {"ell": np.arange(lmax + 1), "n_gal": n_gal, "mean_g_per_pix": mean_g}
    shot = 4.0 * math.pi / n_gal if n_gal > 0 else 0.0
    gg_raw = hp.anafast(delta_g, lmax=lmax)
    gg_with_shot = gg_raw / pixwin2
    out["gg_with_shot"] = gg_with_shot
    out["gg_without_shot"] = (gg_raw - shot) / pixwin2
    out["gg"] = out["gg_without_shot"]
    for field, map_key in (
        ("gy", "map_ymap"),
        ("gksz", "map_ksz"),
        ("gtau", "map_tau"),
        ("gkappa_cmb", "map_kappa_cmb"),
        ("gkappa_wl", "map_kappa_wl"),
    ):
        out[field] = hp.anafast(delta_g, maps[map_key], lmax=lmax) / pixwin2
    out["shot_gg"] = shot
    out["shot_gg_deconvolved"] = shot / pixwin2
    out["pixwin2"] = pixwin2
    return out


def apply_hod_mass_cut(profiles, log10_mass_min: float) -> None:
    import jax.numpy as jnp

    mask = jnp.asarray(jnp.log10(profiles.M_array) >= float(log10_mass_min))
    profiles.Ncen_mat = profiles.Ncen_mat * mask[None, :]
    profiles.Nsat_mat = profiles.Nsat_mat * mask[None, :]


def _validate_nonnegative_pge(pkz) -> None:
    pge = np.asarray(pkz.Pge_tot_mat)
    if not np.all(np.isfinite(pge)):
        raise ValueError("Pge_tot_mat contains non-finite values before Cl projection.")
    min_pge = float(np.min(pge))
    if min_pge < -1.0e-12:
        nneg = int(np.count_nonzero(pge < 0.0))
        raise ValueError(f"Pge_tot_mat is negative before Cl projection: min={min_pge:.6e}, nneg={nneg}.")


def build_theory_cls(
    config_path: Path | str,
    catalog_key: str,
    *,
    is_cmb_lensing: bool,
    log10_mass_min: float,
    z_max: float,
    include_ia: bool = False,
    gg_transition_model: Optional[str] = None,
):
    from base_class import base_class
    from get_radial_profiles import Profiles
    from get_Pkzs import get_Pkz
    from get_Cls import get_Cl

    config = load_config(config_path)
    _, attrs = load_halo_catalog(catalog_path(config, catalog_key), indices=np.asarray([], dtype=np.int64))
    sim_params, halo_params, analysis, other_params = prepare_godmax_config(
        config,
        attrs,
        is_cmb_lensing=is_cmb_lensing,
        z_max=z_max,
        log10_mass_min=log10_mass_min,
    )
    if gg_transition_model is not None:
        analysis["gg_transition_model"] = str(gg_transition_model)
    if not include_ia:
        other_params["A_IA"] = 0.0
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = Profiles(sim_params, halo_params, analysis, other_params, base_class_obj=base)
    apply_hod_mass_cut(profiles, log10_mass_min)
    pkz = get_Pkz(sim_params, halo_params, analysis, other_params, Profiles_obj=profiles)
    _validate_nonnegative_pge(pkz)
    cls = get_Cl(sim_params, halo_params, analysis, other_params, Pkz_obj=pkz)
    return cls


def tau_theory_conversion(cosmo_params: Mapping[str, float], z_mean: float) -> float:
    h = float(cosmo_params["H0"]) / 100.0
    ob0 = float(cosmo_params["Ob0"])
    ne0_cm3 = (1.878e-29 * h**2) * ob0 * (1.0 - 0.24 / 2.0) / astro_const.m_p.to("g").value
    return float(ne0_cm3 * (1.0 + z_mean) ** 3)
