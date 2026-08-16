#!/usr/bin/env python
"""Prepare DES Y3 tomographic shear HEALPix maps for NaMaster.

The input pickle was produced by ``ACTxDES_measurements.ipynb`` from the
DES Y3 metacalibration catalog.  It stores, for each of the four tomographic
bins, calibrated shear components, sky positions, and metacalibration weights.

This script converts that catalog product into portable HEALPix HDF5 maps:

* weighted mean gamma1 and gamma2 maps at nside=1024, 2048, and/or 4096;
* a NaMaster-convention gamma2 map with the standard sign flip;
* weighted count masks following the DES Y3 harmonic-space paper;
* binary masks, counts, weight-squared maps, and shape-noise ingredients;
* HEALPix pixel windows and sqrt-spaced bandpower suggestions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/act_desi_ksz_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/act_desi_ksz_xdgcache")

import dill
import h5py
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NERSC_PROJECT_DIR = Path("/global/cfs/cdirs/lsst/www/shivamp/DESI")
DEFAULT_OUTDIR = PACKAGE_ROOT
DEFAULT_PICKLE = Path("/global/cfs/cdirs/des/data_actxdes/des_data/cat_DES_shearcat_all_dump_nzfix_Feb25.pk")
SOURCE_NOTEBOOK = NERSC_PROJECT_DIR / "ACTxDES_measurements.ipynb"
SHEAR_PAPER_PDF = NERSC_PROJECT_DIR / "2203.07128v1.pdf"
DESY3_HARMONIC_ARXIV = "https://arxiv.org/abs/2203.07128"
DESY3_HARMONIC_MNRAS = "https://academic.oup.com/mnras/article/515/2/1942/6625643"
NAMASTER_FIELD_DOCS = "https://namaster.readthedocs.io/en/latest/api/pymaster.field.html"
DES_SHEAR_DIR = Path("data/des_y3_shear_maps")
QUICKLOOK_DIR = Path("quicklook_figures")
SHEAR_DOC_REL = Path("docs/DES_Y3_SHEAR_MAPS.md")
SHEAR_NOTEBOOK_REL = Path("notebooks/prepare_des_y3_shear_maps.ipynb")
DEFAULT_NSIDES = (1024, 2048, 4096)
HIGH_ELL_TARGET = 8192
MAP_CHUNK_PIXELS = 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def ensure_dirs(outdir: Path) -> None:
    for subdir in (DES_SHEAR_DIR, QUICKLOOK_DIR, Path("docs"), Path("notebooks"), Path("scripts")):
        (outdir / subdir).mkdir(parents=True, exist_ok=True)


def package_relative(path: Path, outdir: Path) -> str:
    try:
        return str(path.resolve().relative_to(outdir.resolve()))
    except ValueError:
        return str(path)


def h5_1d_kwargs(dtype: np.dtype | str, size: int | None = None) -> dict:
    kwargs = {
        "dtype": dtype,
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
    }
    if size is None:
        kwargs["chunks"] = True
    else:
        kwargs["chunks"] = (min(int(size), MAP_CHUNK_PIXELS),)
    return kwargs


def h5_uncompressed_1d_kwargs(dtype: np.dtype | str, size: int | None = None) -> dict:
    if size is None:
        return {"dtype": dtype, "chunks": True}
    return {
        "dtype": dtype,
        "chunks": (min(int(size), MAP_CHUNK_PIXELS),),
    }


def write_map(group: h5py.Group, name: str, data: np.ndarray, dtype: str, units: str, description: str) -> h5py.Dataset:
    arr = np.asarray(data, dtype=dtype)
    dset = group.create_dataset(name, data=arr, **h5_1d_kwargs(arr.dtype, arr.size))
    dset.attrs["units"] = units
    dset.attrs["description"] = description
    dset.attrs["ordering"] = "HEALPix RING"
    return dset


def write_uncompressed_map(
    group: h5py.Group,
    name: str,
    data: np.ndarray,
    dtype: str,
    units: str,
    description: str,
) -> h5py.Dataset:
    arr = np.asarray(data, dtype=dtype)
    dset = group.create_dataset(name, data=arr, **h5_uncompressed_1d_kwargs(arr.dtype, arr.size))
    dset.attrs["units"] = units
    dset.attrs["description"] = description
    dset.attrs["ordering"] = "HEALPix RING"
    return dset


def write_1d(group: h5py.Group, name: str, data: np.ndarray, dtype: str, description: str) -> h5py.Dataset:
    arr = np.asarray(data, dtype=dtype)
    dset = group.create_dataset(name, data=arr, **h5_1d_kwargs(arr.dtype, arr.size))
    dset.attrs["description"] = description
    return dset


def set_common_attrs(h5: h5py.File, outdir: Path, source_pickle: Path, nside: int) -> None:
    h5.attrs["product_type"] = "DES Y3 metacalibration tomographic shear HEALPix maps"
    h5.attrs["created_utc"] = utc_now()
    h5.attrs["created_by"] = Path(__file__).name
    h5.attrs["hostname"] = socket.gethostname()
    h5.attrs["python"] = sys.version
    h5.attrs["platform"] = platform.platform()
    h5.attrs["transfer_package_root"] = "."
    h5.attrs["path_convention"] = "Package-internal paths are relative to the transfer package root containing README.md."
    h5.attrs["output_directory"] = "."
    h5.attrs["nersc_source_pickle"] = str(source_pickle)
    h5.attrs["nersc_source_notebook"] = str(SOURCE_NOTEBOOK)
    h5.attrs["nersc_reference_paper_pdf"] = str(SHEAR_PAPER_PDF)
    h5.attrs["nersc_source_path_note"] = "NERSC source paths are provenance only and are not required after transferring this package."
    h5.attrs["reference_paper_arxiv"] = DESY3_HARMONIC_ARXIV
    h5.attrs["reference_paper_mnras"] = DESY3_HARMONIC_MNRAS
    h5.attrs["namaster_field_docs"] = NAMASTER_FIELD_DOCS
    h5.attrs["nside"] = int(nside)
    h5.attrs["npix"] = int(hp.nside2npix(nside))
    h5.attrs["ordering"] = "RING"
    h5.attrs["spin"] = 2
    h5.attrs["map_definition"] = "gamma_p = sum_i w_i gamma_i / sum_i w_i for galaxies i in HEALPix pixel p"
    h5.attrs["mask_definition"] = "Weighted count masks are sum_i w_i per pixel; normalized masks divide this by the mean over observed pixels."
    h5.attrs["namaster_gamma_maps"] = "Use [gamma1, gamma2_namaster] for NmtField spin=2. gamma2_namaster = -gamma2_catalog."
    h5.attrs["purification_recommendation"] = "DES Y3 harmonic-space paper did not apply E/B purification; purify_e=False and purify_b=False are the matching defaults."
    h5.attrs["pixel_window_note"] = "Use the stored HEALPix polarization pixel window when comparing pixelized shear spectra with theory."
    h5.attrs["small_scale_note"] = (
        "The nside=2048 and nside=4096 products are intended for high-ell tests. They contain many one-galaxy pixels, "
        "so shear auto spectra are shape-noise dominated at small scales and require careful noise subtraction, "
        "mask/window treatment, and scale cuts."
    )


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"n_finite": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan}
    vals = values[finite]
    return {
        "n_finite": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
    }


def sqrt_bandpower_edges(ell_min: int, ell_max: int, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.rint(np.linspace(np.sqrt(ell_min), np.sqrt(ell_max), n_bins + 1) ** 2).astype(np.int32)
    raw[0] = ell_min
    raw[-1] = ell_max + 1
    edges = [int(raw[0])]
    for val in raw[1:]:
        val = int(val)
        if val <= edges[-1]:
            val = edges[-1] + 1
        edges.append(val)
    edges[-1] = ell_max + 1
    edges_arr = np.asarray(edges, dtype=np.int32)
    if np.any(np.diff(edges_arr) <= 0):
        raise ValueError(f"Bandpower edges are not strictly increasing: {edges_arr}")
    return edges_arr[:-1], edges_arr[1:]


def write_bandpower_edges(group: h5py.Group, ell_left: np.ndarray, ell_right: np.ndarray, description: str) -> None:
    group.attrs["description"] = description
    group.attrs["n_bins"] = int(len(ell_left))
    group.attrs["ell_min"] = int(ell_left[0])
    group.attrs["ell_max_inclusive"] = int(ell_right[-1] - 1)
    group.attrs["right_edges_are_exclusive"] = True
    write_1d(group, "ell_left", ell_left, "i4", "Left bin edges.")
    write_1d(group, "ell_right", ell_right, "i4", "Right bin edges, exclusive.")
    ell_eff = 0.5 * (ell_left.astype(np.float64) + (ell_right - 1).astype(np.float64))
    write_1d(group, "ell_center_simple", ell_eff, "f8", "Simple midpoint of each integer-ell band.")


def save_mollview(
    data: np.ndarray,
    observed: np.ndarray,
    path: Path,
    title: str,
    unit: str,
    cmap: str,
    symmetric: bool = False,
) -> None:
    finite = observed & np.isfinite(data)
    if np.any(finite):
        vals = data[finite]
        if symmetric:
            scale = float(np.nanpercentile(np.abs(vals), 98.0))
            vmin, vmax = -scale, scale
        else:
            vmin, vmax = np.nanpercentile(vals, [2.0, 98.0])
            vmin = float(vmin)
            vmax = float(vmax)
    else:
        vmin, vmax = 0.0, 1.0

    shown = np.full(data.shape, hp.UNSEEN, dtype=np.float32)
    shown[finite] = np.asarray(data[finite], dtype=np.float32)
    hp.mollview(shown, title=title, unit=unit, cmap=cmap, min=vmin, max=vmax)
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close("all")


def write_pixel_windows(h5: h5py.File, nside: int) -> None:
    lmax = 3 * nside - 1
    pw_temperature, pw_polarization = hp.pixwin(nside, pol=True, lmax=lmax)
    group = h5.create_group("pixel_window")
    group.attrs["lmax"] = int(lmax)
    group.attrs["description"] = (
        "HEALPix pixel window functions from healpy.pixwin(nside, pol=True). "
        "Use the polarization window for spin-2 shear maps."
    )
    write_1d(group, "ell", np.arange(lmax + 1, dtype=np.int32), "i4", "Multipole ell.")
    write_1d(group, "temperature", pw_temperature, "f8", "Spin-0 HEALPix pixel window.")
    write_1d(group, "polarization", pw_polarization, "f8", "Polarization/spin-2 HEALPix pixel window.")


def write_bandpower_suggestion(h5: h5py.File, nside: int) -> None:
    default_ell_max = 2048 if nside <= 1024 else min(HIGH_ELL_TARGET, 3 * nside - 1)
    default_n_bins = 32 if default_ell_max <= 2048 else 64
    ell_left, ell_right = sqrt_bandpower_edges(8, default_ell_max, default_n_bins)
    group = h5.create_group("bandpowers")
    write_bandpower_edges(
        group,
        ell_left,
        ell_right,
        (
            "Default equal-weight, sqrt-spaced bins for this map resolution. "
            "The nside=1024 file uses the DES-Y3 fiducial ell=8..2048 setup; "
            "higher-resolution files use high-ell suggestions up to min(8192, 3*nside-1) for small-scale tests."
        ),
    )

    alternates = group.create_group("alternates")
    ell_left_2048, ell_right_2048 = sqrt_bandpower_edges(8, 2048, 32)
    write_bandpower_edges(
        alternates.create_group("des_y3_fiducial_ell8_2048"),
        ell_left_2048,
        ell_right_2048,
        "DES Y3 fiducial 32-bin equal-weight sqrt-spaced binning from ell=8 to ell=2048.",
    )
    if nside >= 4096:
        ell_left_high, ell_right_high = sqrt_bandpower_edges(8, min(HIGH_ELL_TARGET, 3 * nside - 1), 64)
        write_bandpower_edges(
            alternates.create_group("high_ell_ell8_8192"),
            ell_left_high,
            ell_right_high,
            "High-ell 64-bin equal-weight sqrt-spaced suggestion for nside=4096 small-scale tests.",
        )


def process_one_tomo(
    tomo_key: int,
    arrays: list,
    nside: int,
    maps_group: h5py.Group,
    hist_group: h5py.Group,
    figdir: Path,
    skip_quicklooks: bool,
) -> dict[str, float | int | str]:
    if len(arrays) != 6:
        raise ValueError(f"Tomographic bin {tomo_key} has {len(arrays)} entries; expected 6.")

    g1 = np.asarray(arrays[0], dtype=np.float32)
    g2 = np.asarray(arrays[1], dtype=np.float32)
    ra = np.asarray(arrays[2], dtype=np.float64)
    dec = np.asarray(arrays[3], dtype=np.float64)
    weight = np.asarray(arrays[5], dtype=np.float32)

    n_input = int(ra.size)
    valid = (
        np.isfinite(g1)
        & np.isfinite(g2)
        & np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(weight)
        & (weight > 0.0)
        & (dec >= -90.0)
        & (dec <= 90.0)
    )
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise ValueError(f"Tomographic bin {tomo_key} has no valid sources.")
    if n_valid != n_input:
        log(f"  tomo {tomo_key}: dropping {n_input - n_valid:,} invalid/non-positive-weight sources")

    g1 = g1[valid]
    g2 = g2[valid]
    ra = np.mod(ra[valid], 360.0)
    dec = dec[valid]
    weight = weight[valid]

    npix = hp.nside2npix(nside)
    pixel_area_sr = hp.nside2pixarea(nside)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    del theta, phi
    gc.collect()

    name = f"tomo{tomo_key}"
    one_based = tomo_key + 1
    group = maps_group.create_group(name)
    hgroup = hist_group.create_group(name)

    count = np.bincount(pix, minlength=npix).astype(np.uint32)
    observed = count > 0
    n_observed = int(np.count_nonzero(observed))
    mean_count_observed = float(np.sum(count, dtype=np.float64) / n_observed)
    area_observed_deg2 = float(n_observed * hp.nside2pixarea(nside, degrees=True))
    count_edges = np.arange(0, int(np.max(count[observed])) + 2, dtype=np.int32)
    count_hist, _ = np.histogram(count[observed], bins=count_edges)
    write_1d(hgroup, "count_per_observed_pixel_hist", count_hist, "i8", "Histogram of source counts in observed pixels.")
    write_1d(hgroup, "count_per_observed_pixel_edges", count_edges, "i4", "Edges for count_per_observed_pixel_hist.")
    write_map(group, "count", count, "u4", "galaxies", "Number of valid DES shear sources per pixel.")
    del count, count_edges, count_hist
    gc.collect()

    sum_w = np.bincount(pix, weights=weight, minlength=npix)
    sum_w_total = float(np.sum(sum_w))
    mean_weight_observed = float(sum_w_total / n_observed)
    write_map(group, "mask_weight_raw", sum_w, "f4", "sum of DES metacalibration weights", "Weighted count mask sum_i w_i per pixel.")
    mask_weight = np.zeros(npix, dtype=np.float32)
    mask_weight[observed] = (sum_w[observed] / mean_weight_observed).astype(np.float32)
    write_map(
        group,
        "mask_weight",
        mask_weight,
        "f4",
        "dimensionless",
        "Weighted count mask normalized by its mean over observed pixels.",
    )
    if not skip_quicklooks:
        save_mollview(
            mask_weight,
            observed,
            figdir / f"des_y3_shear_tomo{one_based}_mask_weight_nside{nside}.png",
            f"DES Y3 tomo {one_based} weighted mask, nside={nside}",
            "normalized sum weights",
            "viridis",
            symmetric=False,
        )
    del mask_weight
    gc.collect()

    mask_binary = observed.astype(np.uint8)
    write_map(group, "mask_binary", mask_binary, "u1", "dimensionless", "Binary observed-pixel mask, 1 where sum_i w_i > 0.")
    del mask_binary
    gc.collect()

    sum_w2 = np.bincount(pix, weights=weight * weight, minlength=npix)
    sum_w2_total = float(np.sum(sum_w2))
    n_eff_global = float(sum_w_total * sum_w_total / sum_w2_total)
    n_eff_per_arcmin2 = float(n_eff_global / (area_observed_deg2 * 3600.0))
    write_map(group, "sum_weight_sq", sum_w2, "f4", "weight^2", "Sum_i w_i^2 per pixel.")
    del sum_w2
    gc.collect()

    gamma_edges = np.linspace(-0.5, 0.5, 201)
    sum_w_g1 = np.bincount(pix, weights=weight * g1, minlength=npix)
    mean_g1_weighted = float(np.sum(sum_w_g1) / sum_w_total)
    gamma1 = np.zeros(npix, dtype=np.float32)
    gamma1[observed] = (sum_w_g1[observed] / sum_w[observed]).astype(np.float32)
    gamma1_summary = json.dumps(finite_summary(gamma1[observed]))
    gamma1_hist, _ = np.histogram(gamma1[observed], bins=gamma_edges)
    write_map(group, "gamma1", gamma1, "f4", "dimensionless", "Weighted mean calibrated DES gamma1/e1 per pixel, catalog convention.")
    write_1d(hgroup, "gamma_hist_edges", gamma_edges, "f4", "Edges for gamma1/gamma2 observed-pixel histograms.")
    write_1d(hgroup, "gamma1_hist", gamma1_hist, "i8", "Histogram of observed-pixel gamma1 values.")
    if not skip_quicklooks:
        save_mollview(
            gamma1,
            observed,
            figdir / f"des_y3_shear_tomo{one_based}_gamma1_nside{nside}.png",
            f"DES Y3 tomo {one_based} gamma1, nside={nside}",
            "gamma1",
            "coolwarm",
            symmetric=True,
        )
    del sum_w_g1, gamma1, gamma1_hist
    gc.collect()

    sum_w_g2 = np.bincount(pix, weights=weight * g2, minlength=npix)
    mean_g2_weighted = float(np.sum(sum_w_g2) / sum_w_total)
    gamma2 = np.zeros(npix, dtype=np.float32)
    gamma2[observed] = (sum_w_g2[observed] / sum_w[observed]).astype(np.float32)
    gamma2_summary = json.dumps(finite_summary(gamma2[observed]))
    gamma2_hist, _ = np.histogram(gamma2[observed], bins=gamma_edges)
    write_map(group, "gamma2", gamma2, "f4", "dimensionless", "Weighted mean calibrated DES gamma2/e2 per pixel, catalog convention.")
    write_map(
        group,
        "gamma2_namaster",
        -gamma2,
        "f4",
        "dimensionless",
        "Sign-flipped gamma2 for NaMaster/HEALPix spin convention: gamma2_namaster = -gamma2.",
    )
    write_1d(hgroup, "gamma2_hist", gamma2_hist, "i8", "Histogram of observed-pixel gamma2 values.")
    if not skip_quicklooks:
        save_mollview(
            gamma2,
            observed,
            figdir / f"des_y3_shear_tomo{one_based}_gamma2_nside{nside}.png",
            f"DES Y3 tomo {one_based} gamma2, nside={nside}",
            "gamma2",
            "coolwarm",
            symmetric=True,
        )
    del sum_w_g2, gamma2, gamma2_hist, gamma_edges
    gc.collect()

    sum_w2_e2_over2 = np.bincount(pix, weights=0.5 * weight * weight * (g1 * g1 + g2 * g2), minlength=npix)
    noise_pseudo_raw = float(pixel_area_sr * np.sum(sum_w2_e2_over2, dtype=np.float64) / npix)
    noise_pseudo_weight_norm = float(noise_pseudo_raw / (mean_weight_observed * mean_weight_observed))
    noise_binary_sum = np.sum(sum_w2_e2_over2[observed] / (sum_w[observed] * sum_w[observed]), dtype=np.float64)
    noise_pseudo_binary = float(pixel_area_sr * noise_binary_sum / npix)
    write_map(
        group,
        "sum_w2_e2_over2",
        sum_w2_e2_over2,
        "f4",
        "weight^2",
        "Per-pixel sum_i w_i^2 * (gamma1_i^2 + gamma2_i^2) / 2 for DES-Y3-style shape-noise estimates.",
    )
    del sum_w2_e2_over2
    gc.collect()

    group.attrs["des_tomographic_bin_zero_based"] = int(tomo_key)
    group.attrs["des_tomographic_bin_one_based"] = int(one_based)
    group.attrs["source_pickle_layout"] = "[gamma1_calibrated, gamma2_calibrated, ra_deg, dec_deg, placeholder, weight]"
    group.attrs["n_input_sources"] = n_input
    group.attrs["n_valid_sources"] = n_valid
    group.attrs["n_observed_pixels"] = n_observed
    group.attrs["area_observed_deg2_binary"] = area_observed_deg2
    group.attrs["mean_count_per_observed_pixel"] = mean_count_observed
    group.attrs["mean_weight_per_observed_pixel"] = mean_weight_observed
    group.attrs["weighted_mean_gamma1_catalog"] = mean_g1_weighted
    group.attrs["weighted_mean_gamma2_catalog"] = mean_g2_weighted
    group.attrs["n_eff_global_weights"] = n_eff_global
    group.attrs["n_eff_per_arcmin2_binary_area"] = n_eff_per_arcmin2
    group.attrs["shape_noise_pseudo_cl_raw_weight_mask"] = noise_pseudo_raw
    group.attrs["shape_noise_pseudo_cl_normalized_weight_mask"] = noise_pseudo_weight_norm
    group.attrs["shape_noise_pseudo_cl_binary_mask"] = noise_pseudo_binary
    group.attrs["shape_noise_note"] = (
        "For shear auto spectra, subtract a coupled/pseudo noise spectrum before decoupling. "
        "If using mask_weight_raw, put shape_noise_pseudo_cl_raw_weight_mask in EE and BB and zero in EB/BE. "
        "If using mask_weight, use shape_noise_pseudo_cl_normalized_weight_mask. "
        "Cross spectra do not receive this additive shape-noise bias."
    )

    summary = {
        "tomo_zero_based": int(tomo_key),
        "tomo_one_based": int(one_based),
        "n_input_sources": n_input,
        "n_valid_sources": n_valid,
        "n_observed_pixels": n_observed,
        "area_observed_deg2_binary": area_observed_deg2,
        "mean_count_per_observed_pixel": mean_count_observed,
        "mean_weight_per_observed_pixel": mean_weight_observed,
        "weighted_mean_gamma1_catalog": mean_g1_weighted,
        "weighted_mean_gamma2_catalog": mean_g2_weighted,
        "gamma1_map_summary": gamma1_summary,
        "gamma2_map_summary": gamma2_summary,
        "n_eff_global_weights": n_eff_global,
        "n_eff_per_arcmin2_binary_area": n_eff_per_arcmin2,
        "shape_noise_pseudo_cl_raw_weight_mask": noise_pseudo_raw,
        "shape_noise_pseudo_cl_normalized_weight_mask": noise_pseudo_weight_norm,
        "shape_noise_pseudo_cl_binary_mask": noise_pseudo_binary,
    }

    del g1, g2, ra, dec, weight, pix
    del observed, sum_w
    gc.collect()

    return summary


def plot_summary_counts(summaries: list[dict[str, float | int | str]], outdir: Path, nside: int) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    tomo = [int(s["tomo_one_based"]) for s in summaries]
    counts = [float(s["mean_count_per_observed_pixel"]) for s in summaries]
    ax.plot(tomo, counts, marker="o", lw=1.8)
    ax.set_xticks(tomo)
    ax.set_xlabel("DES Y3 tomographic bin")
    ax.set_ylabel("Mean sources / observed HEALPix pixel")
    ax.set_title(f"DES Y3 shear sampling at nside={nside}")
    fig.tight_layout()
    fig.savefig(outdir / QUICKLOOK_DIR / f"des_y3_shear_mean_counts_nside{nside}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    n_eff = [float(s["n_eff_per_arcmin2_binary_area"]) for s in summaries]
    ax.plot(tomo, n_eff, marker="o", lw=1.8, color="black")
    ax.set_xticks(tomo)
    ax.set_xlabel("DES Y3 tomographic bin")
    ax.set_ylabel("n_eff [arcmin^-2]")
    ax.set_title("Effective weighted source density")
    fig.tight_layout()
    fig.savefig(outdir / QUICKLOOK_DIR / f"des_y3_shear_neff_nside{nside}.png", dpi=180)
    plt.close(fig)


def nside_from_product_path(path: Path) -> int | None:
    marker = "nside"
    if marker not in path.stem:
        return None
    try:
        return int(path.stem.split(marker)[-1])
    except ValueError:
        return None


def collect_shear_products(outdir: Path, new_paths: list[Path]) -> dict[int, Path]:
    products: dict[int, Path] = {}
    for path in (outdir / DES_SHEAR_DIR).glob("des_y3_metacal_shear_maps_nside*.h5"):
        nside = nside_from_product_path(path)
        if nside is not None:
            products[nside] = path.resolve()
    for path in new_paths:
        nside = nside_from_product_path(path)
        if nside is not None:
            products[nside] = path.resolve()
    return dict(sorted(products.items()))


def update_manifest(outdir: Path, shear_paths: list[Path]) -> None:
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"outdir": str(outdir)}

    required = list(manifest.get("required_later_analysis_packages", []))
    for package in ("dill", "healpy", "pymaster/NaMaster"):
        if package not in required:
            required.append(package)

    products = collect_shear_products(outdir, shear_paths)
    products_json = {f"nside{nside}": package_relative(path, outdir) for nside, path in products.items()}

    manifest["updated_utc"] = utc_now()
    manifest["transfer_root"] = "."
    manifest["path_convention"] = "All package-internal file paths are relative to the transfer package root containing README.md."
    manifest.setdefault("products", {})
    manifest["products"]["des_y3_shear_maps"] = products_json
    manifest["des_shear_products"] = products_json
    if 1024 in products:
        manifest["des_shear_product"] = package_relative(products[1024], outdir)
    elif products:
        manifest["des_shear_product"] = package_relative(next(iter(products.values())), outdir)
    if 2048 in products:
        manifest["des_shear_midres_product"] = package_relative(products[2048], outdir)
    if 4096 in products:
        manifest["des_shear_highres_product"] = package_relative(products[4096], outdir)
    manifest["des_shear_documentation"] = str(SHEAR_DOC_REL)
    manifest["des_shear_notebook"] = str(SHEAR_NOTEBOOK_REL)
    manifest.setdefault("nersc_source_provenance", {})
    manifest["nersc_source_provenance"]["des_y3_shear_pickle"] = str(DEFAULT_PICKLE)
    manifest["nersc_source_provenance"]["des_y3_shear_processing_notebook"] = str(SOURCE_NOTEBOOK)
    manifest["nersc_source_provenance"]["des_y3_harmonic_space_reference_pdf"] = str(SHEAR_PAPER_PDF)
    manifest["quicklook_figures"] = sorted(package_relative(path, outdir) for path in (outdir / QUICKLOOK_DIR).glob("*.png"))
    manifest["required_later_analysis_packages"] = required
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"Updated manifest {manifest_path}")


def write_shear_maps(outdir: Path, source_pickle: Path, nside: int, force: bool, skip_quicklooks: bool) -> Path:
    final_path = outdir / DES_SHEAR_DIR / f"des_y3_metacal_shear_maps_nside{nside}.h5"
    if final_path.exists() and not force:
        log(f"Skipping existing DES shear product {final_path}")
        return final_path

    tmp_path = final_path.with_name(final_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    log(f"Loading DES Y3 processed shear pickle: {source_pickle}")
    with source_pickle.open("rb") as f:
        cat_des = dill.load(f)
    if sorted(cat_des.keys()) != [0, 1, 2, 3]:
        raise ValueError(f"Expected tomographic keys [0,1,2,3], found {sorted(cat_des.keys())}")

    log(f"Writing DES Y3 shear HEALPix maps at nside={nside}: {final_path}")
    summaries: list[dict[str, float | int | str]] = []
    with h5py.File(tmp_path, "w", track_order=True) as h5:
        set_common_attrs(h5, outdir, source_pickle, nside)
        write_pixel_windows(h5, nside)
        write_bandpower_suggestion(h5, nside)

        maps_group = h5.create_group("maps")
        hist_group = h5.create_group("histograms")
        summary_group = h5.create_group("summary")
        figdir = outdir / QUICKLOOK_DIR

        for tomo_key in [0, 1, 2, 3]:
            log(f"  processing DES tomo bin {tomo_key + 1}")
            summary = process_one_tomo(
                tomo_key,
                cat_des[tomo_key],
                nside,
                maps_group,
                hist_group,
                figdir,
                skip_quicklooks,
            )
            summaries.append(summary)

        summary_group.attrs["tomographic_summary_json"] = json.dumps(summaries, indent=2)
        summary_table_dtype = np.dtype(
            [
                ("tomo", "i4"),
                ("n_sources", "i8"),
                ("n_pixels", "i8"),
                ("area_deg2", "f8"),
                ("mean_count", "f8"),
                ("mean_weight", "f8"),
                ("n_eff_arcmin2", "f8"),
                ("noise_raw_weight_mask", "f8"),
                ("noise_normalized_weight_mask", "f8"),
                ("noise_binary_mask", "f8"),
            ]
        )
        table = np.zeros(len(summaries), dtype=summary_table_dtype)
        for i, item in enumerate(summaries):
            table[i] = (
                int(item["tomo_one_based"]),
                int(item["n_valid_sources"]),
                int(item["n_observed_pixels"]),
                float(item["area_observed_deg2_binary"]),
                float(item["mean_count_per_observed_pixel"]),
                float(item["mean_weight_per_observed_pixel"]),
                float(item["n_eff_per_arcmin2_binary_area"]),
                float(item["shape_noise_pseudo_cl_raw_weight_mask"]),
                float(item["shape_noise_pseudo_cl_normalized_weight_mask"]),
                float(item["shape_noise_pseudo_cl_binary_mask"]),
            )
        summary_group.create_dataset("tomographic_summary_table", data=table)

    os.replace(tmp_path, final_path)

    if not skip_quicklooks:
        plot_summary_counts(summaries, outdir, nside)

    del cat_des
    gc.collect()
    log(f"Finished DES Y3 shear product: {final_path}")
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Transfer-product output directory.")
    parser.add_argument("--source-pickle", type=Path, default=DEFAULT_PICKLE, help="Processed DES Y3 shear pickle.")
    parser.add_argument(
        "--nside",
        type=int,
        nargs="+",
        default=list(DEFAULT_NSIDES),
        help="One or more HEALPix nsides for shear maps. Default: 1024 2048 4096.",
    )
    parser.add_argument(
        "--quicklook-max-nside",
        type=int,
        default=1024,
        help="Only make full Mollweide quicklooks up to this nside.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing DES shear HDF5 product.")
    parser.add_argument("--skip-quicklooks", action="store_true", help="Skip PNG quicklook generation.")
    parser.add_argument("--no-manifest-update", action="store_true", help="Do not update the transfer manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    source_pickle = args.source_pickle.resolve()
    nsides = [int(nside) for nside in args.nside]
    for nside in nsides:
        if not hp.isnsideok(nside):
            raise ValueError(f"Invalid HEALPix nside: {nside}")
    ensure_dirs(outdir)
    shear_paths = []
    for nside in nsides:
        skip_quicklooks = args.skip_quicklooks or nside > args.quicklook_max_nside
        if skip_quicklooks and not args.skip_quicklooks:
            log(f"Skipping full Mollweide quicklooks for nside={nside}; use --quicklook-max-nside {nside} to enable.")
        shear_paths.append(write_shear_maps(outdir, source_pickle, nside, args.force, skip_quicklooks))
    if not args.no_manifest_update:
        update_manifest(outdir, shear_paths)
    log("DES Y3 shear map preparation complete.")


if __name__ == "__main__":
    main()
