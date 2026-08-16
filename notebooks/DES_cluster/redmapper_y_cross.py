"""Validated TreeCorr redMaPPer x Compton-y simulation/data diagnostics.

The estimator is the compensated scalar profile DK-RK used by the source
ACTxDES notebook.  This module adds deterministic random selection, explicit
footprints, CAR solid-angle weights, bounded random-catalog I/O, and complete
provenance.  It deliberately does not fit or rescale the two profiles.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import treecorr
import yaml
from astropy.io import fits
from pixell import enmap
from scipy.stats import chi2 as chi2_distribution


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG_PATH = HERE / "params_redmapper_y_cross.yaml"
RESULT_SCHEMA = "godmax_des_cluster_redmapper_y_cross_v1"


def sha256_file(path: str | Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def row_index_sha256(indices: np.ndarray) -> str:
    rows = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(rows.tobytes()).hexdigest()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_config(cfg)
    cfg["_config_path"] = str(config_path)
    cfg["_config_sha256"] = sha256_file(config_path)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    if cfg.get("schema") != "godmax_des_cluster_redmapper_y_cross_config_v1":
        raise ValueError("Unexpected redMaPPer-y configuration schema.")
    selection = cfg["selection"]
    if selection["richness_operator"] != ">" or list(selection["redshift_operators"]) != [
        ">",
        "<",
    ]:
        raise ValueError("Cuts must remain strict: lambda>min and zmin<z<zmax.")
    if float(selection["richness_min"]) != 20.0:
        raise ValueError("This comparison is locked to the source notebook's lambda > 20 cut.")
    redshift_min = float(selection["redshift_min"])
    redshift_max = float(selection["redshift_max"])
    if not (
        np.isfinite(redshift_min)
        and np.isfinite(redshift_max)
        and 0.0 <= redshift_min < redshift_max
    ):
        raise ValueError("Redshift bounds must be finite, nonnegative, and strictly ordered.")
    randoms = cfg["randoms"]
    if list(map(int, randoms["ratios"])) != [5, 10, 20]:
        raise ValueError("Random convergence ratios must remain [5,10,20].")
    if int(randoms["convergence_reference_ratio"]) != 20:
        raise ValueError("Random convergence reference must be 20x.")
    tc = cfg["treecorr"]
    expected = {
        "nbins": 20,
        "min_sep_arcmin": 2.5,
        "max_sep_arcmin": 250.0,
        "bin_type": "Log",
        "bin_slop": 0.0,
        "metric": "Euclidean",
    }
    for key, value in expected.items():
        if tc[key] != value:
            raise ValueError(f"TreeCorr contract drift for {key}: {tc[key]!r} != {value!r}.")
    if int(tc["npatch"]) <= int(tc["nbins"]):
        raise ValueError("Jackknife patch count must exceed the data-vector length.")
    if float(cfg["simulation"]["beam_fwhm_arcmin"]) != 1.6:
        raise ValueError("Simulation beam convention must remain 1.6 arcmin.")
    science_factor = int(cfg["data"]["ymap_downsample_factor"])
    control_factor = int(cfg["data"]["ymap_control_downsample_factor"])
    if {science_factor, control_factor} != {1, 2}:
        raise ValueError("Data pixelization comparison must use factors 1 and 2 exactly once each.")
    validation = cfg["validation"]
    if not isinstance(validation.get("require_data_resolution_pass", True), bool):
        raise ValueError("require_data_resolution_pass must be an explicit boolean.")
    expected_validation = {
        "data_resolution_theta_min_arcmin": 5.0,
        "data_resolution_median_shift_sigma_max": 0.10,
        "data_resolution_maximum_shift_sigma_max": 0.50,
    }
    for key, value in expected_validation.items():
        if float(validation[key]) != value:
            raise ValueError(f"Preregistered validation threshold drift for {key}.")
    for section, keys in {
        "simulation": ("cluster_fits", "random_hdf5", "ymap_hdf5"),
        "data": ("cluster_fits", "random_fits", "ymap_fits", "mask_fits"),
    }.items():
        for key in keys:
            if not Path(cfg[section][key]).is_file():
                raise FileNotFoundError(cfg[section][key])


def _strict_selection(richness: np.ndarray, redshift: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    selection = cfg["selection"]
    return (
        (np.asarray(richness) > float(selection["richness_min"]))
        & (np.asarray(redshift) > float(selection["redshift_min"]))
        & (np.asarray(redshift) < float(selection["redshift_max"]))
    )


def selection_label(cfg: Mapping[str, Any]) -> str:
    selection = cfg["selection"]
    return (
        f"lambda > {float(selection['richness_min']):g} and "
        f"{float(selection['redshift_min']):g} < z < "
        f"{float(selection['redshift_max']):g} (strict; z column)"
    )


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Stable vectorized SplitMix64 keys for deterministic nested samples."""
    x = np.asarray(values, dtype=np.uint64) ^ np.uint64(seed)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def deterministic_nested_sample(
    source_indices: np.ndarray,
    target: int,
    seed: int,
) -> np.ndarray:
    source_indices = np.asarray(source_indices, dtype=np.int64)
    if target <= 0 or len(source_indices) < int(target):
        raise ValueError(f"Need {target} random candidates, found {len(source_indices)}.")
    keys = _splitmix64(source_indices, seed)
    take = np.argpartition(keys, int(target) - 1)[: int(target)]
    order = np.lexsort((source_indices[take], keys[take]))
    return source_indices[take[order]]


def _treecorr_kwargs(cfg: Mapping[str, Any], *, patched: bool) -> dict[str, Any]:
    tc = cfg["treecorr"]
    result = {
        "nbins": int(tc["nbins"]),
        "min_sep": float(tc["min_sep_arcmin"]),
        "max_sep": float(tc["max_sep_arcmin"]),
        "sep_units": "arcmin",
        "bin_type": str(tc["bin_type"]),
        "bin_slop": float(tc["bin_slop"]),
        "metric": str(tc["metric"]),
        "num_threads": int(tc["num_threads"]),
        "verbose": 0,
    }
    if patched:
        result["var_method"] = str(tc["var_method"])
    return result


def _hash_inputs(paths: Mapping[str, str | Path], workers: int) -> dict[str, str]:
    items = [(name, Path(path).resolve()) for name, path in paths.items()]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        hashes = list(pool.map(lambda item: sha256_file(item[1]), items))
    return {name: digest for (name, _), digest in zip(items, hashes)}


def _input_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    return {
        "simulation_cluster": Path(cfg["simulation"]["cluster_fits"]),
        "simulation_random": Path(cfg["simulation"]["random_hdf5"]),
        "simulation_ymap": Path(cfg["simulation"]["ymap_hdf5"]),
        "simulation_ymap_marker": Path(str(cfg["simulation"]["ymap_hdf5"]) + ".validated.json"),
        "data_cluster": Path(cfg["data"]["cluster_fits"]),
        "data_random": Path(cfg["data"]["random_fits"]),
        "data_ymap": Path(cfg["data"]["ymap_fits"]),
        "data_mask": Path(cfg["data"]["mask_fits"]),
    }


def _mask_points(ra_deg: np.ndarray, dec_deg: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    pixels = hp.ang2pix(hp.get_nside(mask), ra_deg, dec_deg, lonlat=True, nest=False)
    return np.asarray(mask[pixels] > float(threshold), dtype=bool)


def load_cluster_selection(
    fits_path: str | Path,
    cfg: Mapping[str, Any],
    expected_rows: int,
    expected_digest: str,
    *,
    mask: np.ndarray | None = None,
    mask_threshold: float = 0.9,
    octant_footprint: np.ndarray | None = None,
) -> dict[str, Any]:
    with fits.open(fits_path, memmap=True) as handle:
        data = handle[1].data
        richness = np.asarray(data[str(cfg["selection"]["richness_column"])]).reshape(-1)
        redshift = np.asarray(data[str(cfg["selection"]["redshift_column"])]).reshape(-1)
        keep = _strict_selection(richness, redshift, cfg)
        source_rows = np.flatnonzero(keep).astype(np.int64)
        if len(source_rows) != int(expected_rows):
            raise ValueError(f"Pre-mask cluster count drift: {len(source_rows)} != {expected_rows}.")
        digest = row_index_sha256(source_rows)
        if digest != str(expected_digest):
            raise ValueError(f"Pre-mask cluster row digest drift: {digest}.")
        ra = np.asarray(data["ra"][keep], dtype=np.float64)
        dec = np.asarray(data["dec"][keep], dtype=np.float64)
        z = np.asarray(redshift[keep], dtype=np.float64)
        lam = np.asarray(richness[keep], dtype=np.float64)

    spatial_keep = np.ones(len(ra), dtype=bool)
    if mask is not None:
        spatial_keep &= _mask_points(ra, dec, mask, mask_threshold)
    if octant_footprint is not None:
        pixels = hp.ang2pix(hp.get_nside(octant_footprint), ra, dec, lonlat=True, nest=False)
        spatial_keep &= np.asarray(octant_footprint[pixels], dtype=bool)
    post_rows = source_rows[spatial_keep]
    if len(post_rows) == 0:
        raise ValueError("No clusters remain after applying the y-map footprint.")
    return {
        "ra_deg": ra[spatial_keep],
        "dec_deg": dec[spatial_keep],
        "redshift": z[spatial_keep],
        "richness": lam[spatial_keep],
        "source_rows": post_rows,
        "pre_mask_rows": int(len(source_rows)),
        "pre_mask_row_index_sha256": digest,
        "selected_rows": int(len(post_rows)),
        "selected_row_index_sha256": row_index_sha256(post_rows),
    }


def load_data_random_sample(
    cfg: Mapping[str, Any],
    mask: np.ndarray,
    target: int,
) -> dict[str, Any]:
    path = cfg["data"]["random_fits"]
    columns = cfg["data"]["random_columns"]
    selection = cfg["selection"]
    with fits.open(path, memmap=True) as handle:
        data = handle[1].data
        z = np.asarray(data[str(columns["redshift"])]).reshape(-1)
        radial = (z > float(selection["redshift_min"])) & (z < float(selection["redshift_max"]))
        radial_rows = np.flatnonzero(radial).astype(np.int64)
        ra = np.asarray(data[str(columns["ra"])][radial], dtype=np.float64)
        dec = np.asarray(data[str(columns["dec"])][radial], dtype=np.float64)
        footprint_keep = _mask_points(ra, dec, mask, float(cfg["data"]["mask_threshold"]))
        eligible_rows = radial_rows[footprint_keep]
        selected_rows = deterministic_nested_sample(
            eligible_rows, int(target), int(cfg["randoms"]["hash_seed_data"])
        )
        selected_data = data[selected_rows]
        selected_ra = np.asarray(selected_data[str(columns["ra"])], dtype=np.float64)
        selected_dec = np.asarray(selected_data[str(columns["dec"])], dtype=np.float64)
    if not np.all(_mask_points(selected_ra, selected_dec, mask, cfg["data"]["mask_threshold"])):
        raise ValueError("Selected data randoms escaped the ACT x DES footprint.")
    return {
        "ra_deg": selected_ra,
        "dec_deg": selected_dec,
        "source_rows": selected_rows,
        "source_row_index_sha256": row_index_sha256(selected_rows),
        "eligible_rows": int(len(eligible_rows)),
        "sampling": "stable SplitMix64 smallest keys from all z/mask-eligible FITS rows",
        "weights": "unit (matches source notebook; FITS WEIGHT intentionally unused)",
    }


def _comoving_radius_bounds(cfg: Mapping[str, Any]) -> tuple[float, float]:
    import tsz_pasting as tp

    z = np.asarray(
        [cfg["selection"]["redshift_min"], cfg["selection"]["redshift_max"]],
        dtype=np.float64,
    )
    radius = tp.comoving_distance_hmpc(z, cfg["simulation"]["cosmology"])
    return float(radius[0]), float(radius[1])


def load_simulation_random_sample(cfg: Mapping[str, Any], target: int) -> dict[str, Any]:
    path = cfg["simulation"]["random_hdf5"]
    dataset_name = str(cfg["simulation"]["random_dataset"])
    columns = cfg["simulation"]["random_columns"]
    rmin, rmax = _comoving_radius_bounds(cfg)
    candidate_rows: list[np.ndarray] = []
    candidate_ra: list[np.ndarray] = []
    candidate_dec: list[np.ndarray] = []
    with h5py.File(path, "r") as handle:
        source = handle[dataset_name]
        chunk_rows = int(source.chunks[0]) if source.chunks else 100_000
        n_chunks = int(np.ceil(len(source) / chunk_rows))
        requested = int(cfg["simulation"]["random_sampled_chunks"])
        chunk_ids = np.unique(np.linspace(0, n_chunks - 1, requested, dtype=np.int64))
        for chunk_id in chunk_ids:
            start = int(chunk_id) * chunk_rows
            stop = min(start + chunk_rows, len(source))
            rows = source[start:stop]
            radius = np.asarray(rows[str(columns["radius"])], dtype=np.float64)
            keep = (radius > rmin) & (radius < rmax)
            if not np.any(keep):
                continue
            candidate_rows.append(np.arange(start, stop, dtype=np.int64)[keep])
            candidate_ra.append(np.asarray(rows[str(columns["ra"])][keep], dtype=np.float64))
            candidate_dec.append(np.asarray(rows[str(columns["dec"])][keep], dtype=np.float64))
    global_rows = np.concatenate(candidate_rows)
    ra_all = np.concatenate(candidate_ra)
    dec_all = np.concatenate(candidate_dec)
    if np.any((ra_all < 0.0) | (ra_all > 90.0) | (dec_all < 0.0) | (dec_all > 90.0)):
        raise ValueError("Simulation random candidates are not in the first octant.")
    selected_rows = deterministic_nested_sample(
        global_rows, int(target), int(cfg["randoms"]["hash_seed_simulation"])
    )
    order = np.argsort(global_rows)
    positions = np.searchsorted(global_rows[order], selected_rows)
    source_positions = order[positions]
    if not np.array_equal(global_rows[source_positions], selected_rows):
        raise RuntimeError("Could not map selected simulation random rows to coordinates.")
    ra = ra_all[source_positions]
    dec = dec_all[source_positions]
    return {
        "ra_deg": ra,
        "dec_deg": dec,
        "source_rows": selected_rows,
        "source_row_index_sha256": row_index_sha256(selected_rows),
        "candidate_rows": int(len(global_rows)),
        "sampled_chunks": int(len(chunk_ids)),
        "source_chunk_rows": int(chunk_rows),
        "radius_min_hmpc": rmin,
        "radius_max_hmpc": rmax,
        "sampling": "evenly stratified HDF5 chunks then stable SplitMix64 smallest keys",
        "weights": "unit; random HDF5 provides no weights",
    }


def _first_octant_footprint(nside: int, fact: int) -> tuple[np.ndarray, np.ndarray]:
    pixels = hp.query_polygon(
        int(nside),
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        inclusive=True,
        fact=int(fact),
        nest=False,
    )
    footprint = np.zeros(hp.nside2npix(int(nside)), dtype=bool)
    footprint[pixels] = True
    return footprint, pixels


def load_simulation_y_pixels(cfg: Mapping[str, Any]) -> dict[str, Any]:
    import plot_tsz_halo_correlations as map_validation

    ymap, metadata = map_validation.load_validated_ymap(cfg["simulation"]["ymap_hdf5"])
    nside = int(metadata["nside"])
    footprint, pixels = _first_octant_footprint(
        nside, int(cfg["simulation"]["footprint_query_fact"])
    )
    theta, phi = hp.pix2ang(nside, pixels, nest=False)
    smoothed = hp.smoothing(
        ymap,
        fwhm=np.deg2rad(float(cfg["simulation"]["beam_fwhm_arcmin"]) / 60.0),
        pol=False,
        iter=int(cfg["simulation"]["beam_smoothing_iter"]),
        lmax=int(cfg["simulation"]["beam_lmax"]),
        use_weights=True,
        use_pixel_weights=False,
        nest=False,
    )
    if not np.all(np.isfinite(smoothed)):
        raise ValueError("Beam-smoothed simulation y map is nonfinite.")
    result = {
        "ra_deg": np.rad2deg(phi),
        "dec_deg": 90.0 - np.rad2deg(theta),
        "k": np.asarray(smoothed[pixels], dtype=np.float64),
        "k_unsmoothed": np.asarray(ymap[pixels], dtype=np.float64),
        "weights": np.ones(len(pixels), dtype=np.float64),
        "footprint": footprint,
        "pixels": pixels,
        "metadata": {
            **metadata,
            "footprint_pixels": int(len(pixels)),
            "footprint_fsky": float(len(pixels) / len(ymap)),
            "beam_smoothed_min": float(np.min(smoothed[pixels])),
            "beam_smoothed_max": float(np.max(smoothed[pixels])),
            "beam_fwhm_arcmin": float(cfg["simulation"]["beam_fwhm_arcmin"]),
            "beam_smoothing_iter": int(cfg["simulation"]["beam_smoothing_iter"]),
            "beam_lmax": int(cfg["simulation"]["beam_lmax"]),
        },
    }
    del ymap, smoothed
    return result


def load_data_y_native(cfg: Mapping[str, Any]) -> tuple[Any, np.ndarray]:
    ymap = enmap.read_map(cfg["data"]["ymap_fits"])
    mask = np.asarray(hp.read_map(cfg["data"]["mask_fits"], field=0, memmap=True), dtype=np.float32)
    if hp.get_nside(mask) != int(cfg["data"]["mask_nside"]):
        raise ValueError("ACT x DES mask NSIDE differs from the configuration.")
    if not np.all(np.isfinite(ymap)) or not np.all(np.isfinite(mask)):
        raise ValueError("ACT y map or mask is nonfinite.")
    return ymap, mask


def extract_data_y_pixels(
    native_ymap: Any,
    mask: np.ndarray,
    cfg: Mapping[str, Any],
    factor: int,
    *,
    row_block: int = 128,
) -> dict[str, Any]:
    downgraded = enmap.downgrade(native_ymap, int(factor))
    pixel_area_rows = np.asarray(
        enmap.pixsizemap(downgraded.shape, downgraded.wcs, broadcastable=True),
        dtype=np.float64,
    )[:, 0]
    positive_area = pixel_area_rows[pixel_area_rows > 0.0]
    area_scale = float(np.median(positive_area))
    ra_parts: list[np.ndarray] = []
    dec_parts: list[np.ndarray] = []
    k_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for start in range(0, downgraded.shape[-2], int(row_block)):
        stop = min(start + int(row_block), downgraded.shape[-2])
        block = downgraded[start:stop, :]
        positions = block.posmap()
        dec = np.rad2deg(np.asarray(positions[0])).reshape(-1)
        ra = np.mod(np.rad2deg(np.asarray(positions[1])).reshape(-1), 360.0)
        values = np.asarray(block, dtype=np.float64).reshape(-1)
        pixels = hp.ang2pix(hp.get_nside(mask), ra, dec, lonlat=True, nest=False)
        keep = (mask[pixels] > float(cfg["data"]["mask_threshold"])) & np.isfinite(values)
        if not np.any(keep):
            continue
        row_weights = np.broadcast_to(
            pixel_area_rows[start:stop, None] / area_scale,
            block.shape,
        ).reshape(-1)
        ra_parts.append(np.asarray(ra[keep], dtype=np.float64))
        dec_parts.append(np.asarray(dec[keep], dtype=np.float64))
        k_parts.append(np.asarray(values[keep], dtype=np.float64))
        weight_parts.append(np.asarray(row_weights[keep], dtype=np.float64))
    result = {
        "ra_deg": np.concatenate(ra_parts),
        "dec_deg": np.concatenate(dec_parts),
        "k": np.concatenate(k_parts),
        "weights": np.concatenate(weight_parts),
        "factor": int(factor),
        "pixel_arcmin": float(cfg["data"]["ymap_native_pixel_arcmin"]) * int(factor),
        "native_shape": tuple(map(int, native_ymap.shape[-2:])),
        "downgraded_shape": tuple(map(int, downgraded.shape[-2:])),
        "weight_policy": "exact pixell CAR solid angle normalized by median retained row area",
    }
    if np.any(~np.isfinite(result["weights"])) or np.any(result["weights"] <= 0.0):
        raise ValueError("Data y-map solid-angle weights are invalid.")
    del downgraded
    return result


def _catalog(
    ra: np.ndarray,
    dec: np.ndarray,
    *,
    patch_centers: np.ndarray | None = None,
    npatch: int | None = None,
    patch_seed: int | None = None,
    k: np.ndarray | None = None,
    w: np.ndarray | None = None,
) -> treecorr.Catalog:
    kwargs: dict[str, Any] = {
        "ra": np.asarray(ra, dtype=np.float64),
        "dec": np.asarray(dec, dtype=np.float64),
        "ra_units": "deg",
        "dec_units": "deg",
    }
    if patch_centers is not None:
        kwargs["patch_centers"] = patch_centers
    elif npatch is not None:
        kwargs["npatch"] = int(npatch)
        kwargs["rng"] = np.random.RandomState(int(patch_seed))
    if k is not None:
        kwargs["k"] = np.asarray(k, dtype=np.float64)
    if w is not None:
        kwargs["w"] = np.asarray(w, dtype=np.float64)
    return treecorr.Catalog(**kwargs)


def _process_nk(cat_n: treecorr.Catalog, cat_k: treecorr.Catalog, cfg: Mapping[str, Any], *, patched: bool) -> treecorr.NKCorrelation:
    corr = treecorr.NKCorrelation(**_treecorr_kwargs(cfg, patched=patched))
    corr.process(cat_n, cat_k, num_threads=int(cfg["treecorr"]["num_threads"]))
    return corr


def _null_chi2(xi: np.ndarray, covariance: np.ndarray, rcond: float) -> dict[str, float | int]:
    covariance = np.asarray(covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    threshold = float(np.max(eigenvalues)) * float(rcond)
    keep = eigenvalues > threshold
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        raise ValueError("Random-position null covariance has zero retained rank.")
    projected = eigenvectors[:, keep].T @ np.asarray(xi, dtype=np.float64)
    chi2_value = float(np.sum(projected**2 / eigenvalues[keep]))
    return {
        "chi2": chi2_value,
        "rank": rank,
        "pte": float(chi2_distribution.sf(chi2_value, rank)),
        "eigenvalue_threshold": threshold,
    }


def _random_prefix(random_sample: Mapping[str, Any], count: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(random_sample["ra_deg"][:count], dtype=np.float64),
        np.asarray(random_sample["dec_deg"][:count], dtype=np.float64),
    )


def measure_patched_profile(
    clusters: Mapping[str, Any],
    random_sample: Mapping[str, Any],
    y_pixels: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    patch_seed: int,
) -> dict[str, Any]:
    ncluster = int(clusters["selected_rows"])
    ratios = [int(value) for value in cfg["randoms"]["ratios"]]
    reference_ratio = int(cfg["randoms"]["convergence_reference_ratio"])
    max_count = reference_ratio * ncluster
    if len(random_sample["ra_deg"]) != max_count:
        raise ValueError("Random sample length does not equal reference_ratio*ncluster.")

    patch_seed_catalog = _catalog(
        random_sample["ra_deg"],
        random_sample["dec_deg"],
        npatch=int(cfg["treecorr"]["npatch"]),
        patch_seed=int(patch_seed),
    )
    patch_centers = np.asarray(patch_seed_catalog.patch_centers, dtype=np.float64)
    cat_k = _catalog(
        y_pixels["ra_deg"],
        y_pixels["dec_deg"],
        k=y_pixels["k"],
        w=y_pixels.get("weights"),
        patch_centers=patch_centers,
    )
    cat_clusters = _catalog(
        clusters["ra_deg"], clusters["dec_deg"], patch_centers=patch_centers
    )
    cluster_y = _process_nk(cat_clusters, cat_k, cfg, patched=True)

    random_correlations: dict[int, treecorr.NKCorrelation] = {}
    random_profiles: dict[int, np.ndarray] = {}
    for ratio in ratios:
        count = ratio * ncluster
        ra, dec = _random_prefix(random_sample, count)
        cat_random = _catalog(ra, dec, patch_centers=patch_centers)
        corr = _process_nk(cat_random, cat_k, cfg, patched=True)
        random_correlations[ratio] = corr
        random_profiles[ratio] = np.asarray(cluster_y.raw_xi - corr.raw_xi, dtype=np.float64)

    reference = random_correlations[reference_ratio]
    cluster_y.calculateXi(rk=reference)
    covariance_reference = np.asarray(cluster_y.cov, dtype=np.float64)
    sigma_reference = np.sqrt(np.clip(np.diag(covariance_reference), 0.0, None))
    if np.any(sigma_reference <= 0.0) or np.any(~np.isfinite(sigma_reference)):
        raise ValueError("Reference jackknife errors are nonpositive or nonfinite.")
    convergence: dict[int, dict[str, float | bool]] = {}
    chosen_ratio = reference_ratio
    for ratio in ratios:
        normalized = np.abs(random_profiles[ratio] - random_profiles[reference_ratio]) / sigma_reference
        median = float(np.median(normalized))
        maximum = float(np.max(normalized))
        passed = (
            median <= float(cfg["randoms"]["median_shift_sigma_max"])
            and maximum <= float(cfg["randoms"]["maximum_shift_sigma_max"])
        )
        convergence[ratio] = {"median_shift_sigma": median, "max_shift_sigma": maximum, "pass": passed}
        if passed and chosen_ratio == reference_ratio:
            chosen_ratio = ratio

    chosen_random = random_correlations[chosen_ratio]
    cluster_y.calculateXi(rk=chosen_random)
    covariance = np.asarray(cluster_y.cov, dtype=np.float64)
    xi = np.asarray(cluster_y.xi, dtype=np.float64)

    half_count = int(cfg["randoms"]["null_half_ratio"]) * ncluster
    ra_a, dec_a = _random_prefix(random_sample, half_count)
    ra_b = np.asarray(random_sample["ra_deg"][half_count : 2 * half_count], dtype=np.float64)
    dec_b = np.asarray(random_sample["dec_deg"][half_count : 2 * half_count], dtype=np.float64)
    random_a = _process_nk(_catalog(ra_a, dec_a, patch_centers=patch_centers), cat_k, cfg, patched=True)
    random_b = _process_nk(_catalog(ra_b, dec_b, patch_centers=patch_centers), cat_k, cfg, patched=True)
    random_a.calculateXi(rk=random_b)
    null_covariance = np.asarray(random_a.cov, dtype=np.float64)
    null_xi = np.asarray(random_a.xi, dtype=np.float64)
    null_test = _null_chi2(
        null_xi,
        null_covariance,
        float(cfg["treecorr"]["null_covariance_rcond"]),
    )

    patch_counts = np.bincount(
        np.asarray(cat_clusters.patch, dtype=np.int64), minlength=int(cfg["treecorr"]["npatch"])
    )
    return {
        "theta_arcmin": np.exp(np.asarray(cluster_y.meanlogr, dtype=np.float64)),
        "xi": xi,
        "covariance": covariance,
        "error": np.sqrt(np.clip(np.diag(covariance), 0.0, None)),
        "raw_cluster_y": np.asarray(cluster_y.raw_xi, dtype=np.float64),
        "chosen_random_y": np.asarray(chosen_random.raw_xi, dtype=np.float64),
        "random_profiles": {ratio: profile for ratio, profile in random_profiles.items()},
        "random_convergence": convergence,
        "chosen_random_ratio": int(chosen_ratio),
        "null_xi": null_xi,
        "null_covariance": null_covariance,
        "null_test": null_test,
        "patch_centers": patch_centers,
        "cluster_patch_count_min": int(np.min(patch_counts)),
        "cluster_patch_count_max": int(np.max(patch_counts)),
        "cluster_patch_count_zero": int(np.count_nonzero(patch_counts == 0)),
    }


def measure_unpatched_profile(
    clusters: Mapping[str, Any],
    random_sample: Mapping[str, Any],
    y_pixels: Mapping[str, Any],
    cfg: Mapping[str, Any],
    random_ratio: int,
) -> dict[str, np.ndarray]:
    nrandom = int(random_ratio) * int(clusters["selected_rows"])
    cat_k = _catalog(
        y_pixels["ra_deg"], y_pixels["dec_deg"], k=y_pixels["k"], w=y_pixels.get("weights")
    )
    cat_c = _catalog(clusters["ra_deg"], clusters["dec_deg"])
    ra, dec = _random_prefix(random_sample, nrandom)
    cat_r = _catalog(ra, dec)
    cluster_y = _process_nk(cat_c, cat_k, cfg, patched=False)
    random_y = _process_nk(cat_r, cat_k, cfg, patched=False)
    return {
        "theta_arcmin": np.exp(np.asarray(cluster_y.meanlogr, dtype=np.float64)),
        "xi": np.asarray(cluster_y.raw_xi - random_y.raw_xi, dtype=np.float64),
    }


def _resolution_control(
    science: Mapping[str, Any], control: Mapping[str, Any], cfg: Mapping[str, Any]
) -> dict[str, float | bool]:
    use = np.asarray(science["theta_arcmin"]) >= float(
        cfg["validation"]["data_resolution_theta_min_arcmin"]
    )
    normalized = np.abs(np.asarray(control["xi"]) - np.asarray(science["xi"])) / np.asarray(
        science["error"]
    )
    median = float(np.median(normalized[use]))
    maximum = float(np.max(normalized[use]))
    passed = (
        median <= float(cfg["validation"]["data_resolution_median_shift_sigma_max"])
        and maximum <= float(cfg["validation"]["data_resolution_maximum_shift_sigma_max"])
    )
    return {"median_shift_sigma": median, "max_shift_sigma": maximum, "pass": passed}


def _write_group(group: h5py.Group, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif isinstance(value, (str, bytes, int, float, bool, np.generic)):
            group.attrs[key] = value


def save_result(
    path: str | Path,
    cfg: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    simulation: Mapping[str, Any],
    data: Mapping[str, Any],
    simulation_clusters: Mapping[str, Any],
    data_clusters: Mapping[str, Any],
    simulation_randoms: Mapping[str, Any],
    data_randoms: Mapping[str, Any],
    data_resolution: Mapping[str, Any],
    simulation_unsmoothed: Mapping[str, Any],
) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    try:
        with h5py.File(staging, "w") as handle:
            handle.attrs["schema"] = RESULT_SCHEMA
            handle.attrs["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            handle.attrs["config_path"] = str(cfg["_config_path"])
            handle.attrs["config_sha256"] = str(cfg["_config_sha256"])
            handle.attrs["helper_sha256"] = sha256_file(__file__)
            handle.attrs["treecorr_version"] = treecorr.__version__
            handle.attrs["input_sha256_json"] = json.dumps(dict(input_hashes), sort_keys=True)
            handle.attrs["selection"] = selection_label(cfg)
            handle.attrs["estimator"] = "TreeCorr NK: <y>_cluster - <y>_random"
            handle.attrs["act_units_status"] = str(cfg["data"]["ymap_units_status"])
            handle.attrs["act_beam_status"] = str(cfg["data"]["beam_status"])
            handle.attrs["simulation_beam_status"] = str(cfg["simulation"]["beam_status"])
            handle.attrs["noise_policy_json"] = json.dumps(
                {
                    "simulation": cfg["simulation"]["noise_policy"],
                    "data": cfg["data"]["noise_policy"],
                },
                sort_keys=True,
            )
            handle.attrs["cluster_weight_policy"] = str(
                cfg["selection"]["cluster_weight_policy"]
            )
            handle.attrs["data_transfer_status"] = str(cfg["data"]["transfer_status"])
            handle.attrs["simulation_mass_amplitude_status"] = str(
                cfg["simulation"]["mass_amplitude_status"]
            )
            resolution_pass = bool(data_resolution["metrics"]["pass"])
            handle.attrs["data_resolution_required"] = bool(
                cfg["validation"].get("require_data_resolution_pass", True)
            )
            handle.attrs["validation_status"] = (
                "run_local_checks_passed" if resolution_pass else "refuted_data_resolution"
            )
            handle.attrs["interpretation"] = (
                "one-realization diagnostic; no fitted amplitude, chi-square comparison, or detection claim"
            )
            handle.attrs["config_yaml"] = yaml.safe_dump(
                {key: value for key, value in cfg.items() if not key.startswith("_")}, sort_keys=True
            )
            for label, result, clusters, randoms in (
                ("simulation", simulation, simulation_clusters, simulation_randoms),
                ("data", data, data_clusters, data_randoms),
            ):
                group = handle.create_group(label)
                for key in (
                    "theta_arcmin",
                    "xi",
                    "covariance",
                    "error",
                    "raw_cluster_y",
                    "chosen_random_y",
                    "null_xi",
                    "null_covariance",
                    "patch_centers",
                ):
                    group.create_dataset(key, data=result[key])
                group.attrs["chosen_random_ratio"] = int(result["chosen_random_ratio"])
                group.attrs["random_convergence_json"] = json.dumps(
                    result["random_convergence"], sort_keys=True
                )
                group.attrs["null_test_json"] = json.dumps(result["null_test"], sort_keys=True)
                group.attrs["cluster_patch_count_min"] = int(result["cluster_patch_count_min"])
                group.attrs["cluster_patch_count_max"] = int(result["cluster_patch_count_max"])
                group.attrs["cluster_patch_count_zero"] = int(result["cluster_patch_count_zero"])
                group.attrs["y_pixel_metadata_json"] = json.dumps(
                    result["y_pixel_metadata"], sort_keys=True
                )
                group.attrs["cluster_metadata_json"] = json.dumps(
                    {key: value for key, value in clusters.items() if not isinstance(value, np.ndarray)},
                    sort_keys=True,
                )
                group.create_dataset("cluster_redshift", data=clusters["redshift"])
                group.create_dataset("cluster_richness", data=clusters["richness"])
                group.attrs["random_metadata_json"] = json.dumps(
                    {key: value for key, value in randoms.items() if not isinstance(value, np.ndarray)},
                    sort_keys=True,
                )
                convergence_group = group.create_group("random_ratio_profiles")
                for ratio, profile in result["random_profiles"].items():
                    convergence_group.create_dataset(str(ratio), data=profile)
            controls = handle.create_group("controls")
            controls.create_dataset("data_resolution_control_theta_arcmin", data=data_resolution["profile"]["theta_arcmin"])
            controls.create_dataset("data_resolution_control_xi", data=data_resolution["profile"]["xi"])
            controls.attrs["data_resolution_json"] = json.dumps(data_resolution["metrics"], sort_keys=True)
            controls.create_dataset(
                "simulation_unsmoothed_theta_arcmin", data=simulation_unsmoothed["theta_arcmin"]
            )
            controls.create_dataset("simulation_unsmoothed_xi", data=simulation_unsmoothed["xi"])
            handle.flush()
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            staging.unlink()
        raise
    return output


def plot_comparison(analysis: Mapping[str, Any]) -> plt.Figure:
    sim = analysis["simulation"]
    data = analysis["data"]
    sim_clusters = analysis["simulation_clusters"]
    data_clusters = analysis["data_clusters"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    axis = axes[0]
    axis.errorbar(
        data["theta_arcmin"],
        data["xi"],
        yerr=data["error"],
        fmt="o",
        ms=4,
        capsize=2,
        label=f"ACT x DES data (N={data_clusters['selected_rows']:,})",
    )
    axis.errorbar(
        np.asarray(sim["theta_arcmin"]) * 1.025,
        sim["xi"],
        yerr=sim["error"],
        fmt="s",
        ms=4,
        capsize=2,
        label=f"Simulation, 1.6' beam convention (N={sim_clusters['selected_rows']:,})",
    )
    all_values = np.concatenate([np.asarray(data["xi"]), np.asarray(sim["xi"])])
    nonzero = np.abs(all_values[all_values != 0.0])
    linthresh = max(float(np.percentile(nonzero, 15)) if len(nonzero) else 1.0e-10, 1.0e-14)
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=linthresh, linscale=0.7)
    axis.axhline(0.0, color="0.4", lw=0.8)
    axis.set_xlabel(r"Angular separation $\theta$ [arcmin]")
    axis.set_ylabel(r"Compensated $\xi_{c y}=\langle y\rangle_c-\langle y\rangle_r$")
    selection = analysis["config"]["selection"]
    zmin = float(selection["redshift_min"])
    zmax = float(selection["redshift_max"])
    axis.set_title(
        rf"Same strict cuts: $\lambda>{float(selection['richness_min']):g}$, "
        rf"${zmin:g}<z<{zmax:g}$"
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=9)

    bins = np.linspace(zmin, zmax, 16)
    axes[1].hist(
        data_clusters["redshift"], bins=bins, density=True, histtype="step", lw=2, label="Data"
    )
    axes[1].hist(
        sim_clusters["redshift"],
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        label="Simulation",
    )
    axes[1].set_xlabel("Catalog z used by source notebook")
    axes[1].set_ylabel("Normalized cluster density")
    axes[1].set_title("Selected redshift distributions are not matched")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle(
        "TreeCorr redMaPPer x y diagnostic; ACT units/beam metadata absent, simulation mass amplitude provisional",
        fontsize=11,
    )
    resolution = analysis.get("data_resolution", {}).get("metrics", {})
    if resolution and not bool(resolution.get("pass", False)):
        fig.text(
            0.5,
            0.005,
            (
                "CONDITIONAL DIAGNOSTIC: data pixel-resolution control failed "
                f"(max shift {float(resolution['max_shift_sigma']):.3f} sigma)"
            ),
            ha="center",
            va="bottom",
            color="crimson",
            fontsize=10,
            weight="bold",
        )
    return fig


def plot_diagnostics(analysis: Mapping[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for column, label in enumerate(("data", "simulation")):
        result = analysis[label]
        axis = axes[0, column]
        for ratio, profile in sorted(result["random_profiles"].items()):
            axis.plot(result["theta_arcmin"], profile, marker="o", ms=3, label=f"{ratio}x randoms")
        axis.set_xscale("log")
        axis.set_yscale("symlog", linthresh=1.0e-9, linscale=0.7)
        axis.axhline(0.0, color="0.4", lw=0.8)
        axis.set_title(f"{label.capitalize()} random-density convergence")
        axis.set_xlabel(r"$\theta$ [arcmin]")
        axis.set_ylabel(r"$\xi_{cy}$")
        axis.legend(fontsize=8)
        axis.grid(True, which="both", alpha=0.25)

        null_axis = axes[1, column]
        null_error = np.sqrt(np.clip(np.diag(result["null_covariance"]), 0.0, None))
        null_axis.errorbar(result["theta_arcmin"], result["null_xi"], yerr=null_error, fmt="o", ms=3)
        null_axis.set_xscale("log")
        null_axis.axhline(0.0, color="0.4", lw=0.8)
        null_axis.set_title(
            f"RandomA-RandomB null: PTE={result['null_test']['pte']:.3f}, rank={result['null_test']['rank']}"
        )
        null_axis.set_xlabel(r"$\theta$ [arcmin]")
        null_axis.set_ylabel(r"Null $\xi_y$")
        null_axis.grid(True, which="both", alpha=0.25)
    return fig


def run_analysis(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    cfg = load_config(config_path)
    input_paths = _input_paths(cfg)
    input_hashes_before = _hash_inputs(
        input_paths, int(cfg["validation"]["input_hash_workers"])
    )

    simulation_y = load_simulation_y_pixels(cfg)
    simulation_clusters = load_cluster_selection(
        cfg["simulation"]["cluster_fits"],
        cfg,
        int(cfg["simulation"]["expected_pre_mask_rows"]),
        str(cfg["simulation"]["expected_pre_mask_row_index_sha256"]),
        octant_footprint=simulation_y["footprint"],
    )
    simulation_randoms = load_simulation_random_sample(
        cfg,
        int(cfg["randoms"]["convergence_reference_ratio"])
        * int(simulation_clusters["selected_rows"]),
    )
    simulation = measure_patched_profile(
        simulation_clusters,
        simulation_randoms,
        simulation_y,
        cfg,
        patch_seed=int(cfg["treecorr"]["patch_seed_simulation"]),
    )
    simulation["y_pixel_metadata"] = {
        "count": int(len(simulation_y["k"])),
        "nside": int(simulation_y["metadata"]["nside"]),
        "ordering": "RING",
        "weight_policy": "equal HEALPix pixel area",
        "footprint": "inclusive first octant",
        "footprint_fsky": float(simulation_y["metadata"]["footprint_fsky"]),
        "beam_fwhm_arcmin": float(simulation_y["metadata"]["beam_fwhm_arcmin"]),
        "beam_smoothing_iter": int(simulation_y["metadata"]["beam_smoothing_iter"]),
        "beam_lmax": int(simulation_y["metadata"]["beam_lmax"]),
        "beam_smoothed_min": float(simulation_y["metadata"]["beam_smoothed_min"]),
        "beam_smoothed_max": float(simulation_y["metadata"]["beam_smoothed_max"]),
    }
    simulation_unsmoothed = measure_unpatched_profile(
        simulation_clusters,
        simulation_randoms,
        {**simulation_y, "k": simulation_y["k_unsmoothed"]},
        cfg,
        int(simulation["chosen_random_ratio"]),
    )
    del simulation_y

    native_data_y, data_mask = load_data_y_native(cfg)
    data_clusters = load_cluster_selection(
        cfg["data"]["cluster_fits"],
        cfg,
        int(cfg["data"]["expected_pre_mask_rows"]),
        str(cfg["data"]["expected_pre_mask_row_index_sha256"]),
        mask=data_mask,
        mask_threshold=float(cfg["data"]["mask_threshold"]),
    )
    data_randoms = load_data_random_sample(
        cfg,
        data_mask,
        int(cfg["randoms"]["convergence_reference_ratio"]) * int(data_clusters["selected_rows"]),
    )
    data_y = extract_data_y_pixels(
        native_data_y, data_mask, cfg, int(cfg["data"]["ymap_downsample_factor"])
    )
    data = measure_patched_profile(
        data_clusters,
        data_randoms,
        data_y,
        cfg,
        patch_seed=int(cfg["treecorr"]["patch_seed_data"]),
    )
    data["y_pixel_metadata"] = {
        "count": int(len(data_y["k"])),
        "native_shape": list(data_y["native_shape"]),
        "downgraded_shape": list(data_y["downgraded_shape"]),
        "downsample_factor": int(data_y["factor"]),
        "pixel_arcmin": float(data_y["pixel_arcmin"]),
        "weight_policy": str(data_y["weight_policy"]),
        "mask_threshold": float(cfg["data"]["mask_threshold"]),
    }
    del data_y
    data_y_control = extract_data_y_pixels(
        native_data_y,
        data_mask,
        cfg,
        int(cfg["data"]["ymap_control_downsample_factor"]),
    )
    data_control_profile = measure_unpatched_profile(
        data_clusters,
        data_randoms,
        data_y_control,
        cfg,
        int(data["chosen_random_ratio"]),
    )
    data_resolution_metrics = _resolution_control(data, data_control_profile, cfg)
    data_resolution_metrics.update(
        {
            "science_downsample_factor": int(cfg["data"]["ymap_downsample_factor"]),
            "control_downsample_factor": int(cfg["data"]["ymap_control_downsample_factor"]),
        }
    )
    del data_y_control, native_data_y, data_mask

    central_bins = int(cfg["validation"]["central_positive_bins"])
    for label, result in (("simulation", simulation), ("data", data)):
        if float(np.mean(result["xi"][:central_bins])) <= 0.0:
            raise ValueError(f"{label} central compensated profile is not positive.")
        if float(result["null_test"]["pte"]) <= float(cfg["randoms"]["null_pte_min"]):
            raise ValueError(f"{label} random-position null fails: {result['null_test']}.")
    if (
        not bool(data_resolution_metrics["pass"])
        and bool(cfg["validation"].get("require_data_resolution_pass", True))
    ):
        raise ValueError(f"Data science-versus-control resolution check fails: {data_resolution_metrics}.")

    input_hashes_after = _hash_inputs(
        input_paths, int(cfg["validation"]["input_hash_workers"])
    )
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("An input file changed during the read-only analysis.")

    output_dir = Path(cfg["output"]["directory"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = save_result(
        output_dir / str(cfg["output"]["result_hdf5"]),
        cfg,
        input_hashes_before,
        simulation,
        data,
        simulation_clusters,
        data_clusters,
        simulation_randoms,
        data_randoms,
        {"profile": data_control_profile, "metrics": data_resolution_metrics},
        simulation_unsmoothed,
    )
    analysis = {
        "config": cfg,
        "simulation": simulation,
        "data": data,
        "simulation_clusters": simulation_clusters,
        "data_clusters": data_clusters,
        "simulation_randoms": simulation_randoms,
        "data_randoms": data_randoms,
        "simulation_unsmoothed": simulation_unsmoothed,
        "data_resolution": {"profile": data_control_profile, "metrics": data_resolution_metrics},
        "input_hashes": input_hashes_before,
        "result_path": str(result_path),
    }
    comparison = plot_comparison(analysis)
    comparison_path = output_dir / str(cfg["output"]["comparison_png"])
    comparison.savefig(comparison_path, dpi=170, bbox_inches="tight")
    diagnostics = plot_diagnostics(analysis)
    diagnostics_path = output_dir / str(cfg["output"]["diagnostics_png"])
    diagnostics.savefig(diagnostics_path, dpi=170, bbox_inches="tight")
    analysis["comparison_figure"] = comparison
    analysis["diagnostics_figure"] = diagnostics
    analysis["comparison_path"] = str(comparison_path)
    analysis["diagnostics_path"] = str(diagnostics_path)
    analysis["summary"] = {
        "result_hdf5": str(result_path),
        "comparison_png": str(comparison_path),
        "diagnostics_png": str(diagnostics_path),
        "simulation_clusters": int(simulation_clusters["selected_rows"]),
        "data_clusters": int(data_clusters["selected_rows"]),
        "simulation_random_ratio": int(simulation["chosen_random_ratio"]),
        "data_random_ratio": int(data["chosen_random_ratio"]),
        "simulation_null_pte": float(simulation["null_test"]["pte"]),
        "data_null_pte": float(data["null_test"]["pte"]),
        "data_resolution": dict(data_resolution_metrics),
        "validation_status": (
            "run_local_checks_passed"
            if bool(data_resolution_metrics["pass"])
            else "refuted_data_resolution"
        ),
        "simulation_central_mean": float(np.mean(simulation["xi"][:central_bins])),
        "data_central_mean": float(np.mean(data["xi"][:central_bins])),
        "input_hashes_unchanged": input_hashes_before == input_hashes_after,
        "treecorr_version": treecorr.__version__,
        "claim_boundary": (
            "diagnostic only; ACT BUNIT/beam metadata absent; no fitted amplitude or goodness claim"
        ),
    }
    return analysis
