#!/usr/bin/env python
"""Prepare portable HDF5 inputs for DESI x ACT kSZ angular spectra.

This script keeps the original ASCII and FITS products untouched and writes a
compact transfer directory containing:

* one HDF5 catalog per DESI photo-z bin with the columns needed for kSZ work;
* one combined catalog HDF5 with all bins concatenated;
* one ACT HDF5 containing the CMB map, mask, WCS metadata, and quicklooks;
* PNG quicklook figures for redshift distributions and sky coverage.

The DESI velocity convention follows the input instructions:
    vr_over_c = input_column_15 / 3e5
No sign flip is applied here.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/act_desi_ksz_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/act_desi_ksz_xdgcache")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pixell import enmap


C_KM_S = 3.0e5
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NERSC_PROJECT_DIR = Path("/global/cfs/cdirs/lsst/www/shivamp/DESI")
DEFAULT_OUTDIR = PACKAGE_ROOT
PAPER_PDF = NERSC_PROJECT_DIR / "2604.19744v1.pdf"
DESI_CATALOG_DIR = Path("data/desi_dr10_extended_velocity_catalogs")
DESI_RANDOMS_DIR = Path("data/desi_dr10_imaging_randoms")
DESI_ABACUS_CALIBRATION_DIR = Path("data/desi_abacus_velocity_calibration")
DESI_HEALPIX_DIR = Path("data/desi_dr10_healpix_quicklooks")
ACT_CMB_DIR = Path("data/act_dr6_cmb_temperature")
ACT_TSZ_DIR = Path("data/act_dr6_tsz_compton_y")
ACT_LENSING_DIR = Path("data/act_dr6_lensing_kappa")
QUICKLOOK_DIR = Path("quicklook_figures")
DR10_RANDOMS_SOURCE_DIR = Path("/global/cfs/cdirs/cosmo/data/legacysurvey/dr10/randoms")
DR10_RANDOMS_SOURCE_PATH = DR10_RANDOMS_SOURCE_DIR / "randoms-1-0.fits"
DR10_RANDOMS_SHA256_SOURCE_PATH = DR10_RANDOMS_SOURCE_DIR / "legacysurvey_dr10_randoms.sha256sum"
DR10_RANDOMS_1_0_SHA256 = "ebe86f81db7eecdd1c01b451e29c4a6d79e054d055776328d1f38a52daa0d008"
ABACUS_SIGMA_TRUE_SOURCE_PATH = Path(
    "/global/cfs/cdirs/desi/users/boryanah/kSZ_recon/for_fiona/"
    "Extended_LRG_zerr0.0_AbacusSummit_huge_c000_ph201.npz"
)
ABACUS_SIGMA_TRUE_PRODUCT = (
    "sigma_true_gas_abacus_extended_lrg_zerr0p0_ph201_photometric_bins.json"
)
ACT_MAP_PATH = Path("/pscratch/sd/b/boryanah/ACTxDESI/ACT/hilc_fullRes_TT_17000.fits")
ACT_MASK_PATH = Path(
    "/pscratch/sd/b/boryanah/ACTxDESI/ACT/"
    "wide_mask_GAL070_apod_1.50_deg_wExtended_srcfree_Will.fits"
)
TSZ_DIR = Path("/global/cfs/cdirs/cmb/data/act_dr6/dr6.02/nilc/published")
TSZ_MAP_PATH = TSZ_DIR / "act-planck_dr6.02_nilc_ComptonY_deproj_cib_cibdBeta_1.7_10.7.fits"
TSZ_FOOTPRINT_MASK_PATH = TSZ_DIR / "ilc_footprint_mask.fits"
TSZ_INPAINT_MASK_PATH = TSZ_DIR / "ilc_inpaint_mask.fits"
TSZ_SUBTRACTED_MASK_PATH = TSZ_DIR / "ilc_subtracted_mask.fits"
TSZ_MANIFEST_PATH = TSZ_DIR / "manifest.json"

LENSING_NOTEBOOK = NERSC_PROJECT_DIR / "ACT_DR6_lensing_CIB_correlation.ipynb"
LENSING_BASELINE_DIR = Path("/pscratch/sd/b/boryanah/ACTxDESI/ACT/maps/baseline")
LENSING_KAPPA_MAP_PATH = LENSING_BASELINE_DIR / "kappa_alm_data_act_dr6_lensing_v1_baseline_masked.fits"
LENSING_KAPPA_ALM_PATH = LENSING_BASELINE_DIR / "kappa_alm_data_act_dr6_lensing_v1_baseline.fits"
LENSING_HEALPIX_MASK_PATH = LENSING_BASELINE_DIR / "mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits"
LENSING_NOISE_PATH = LENSING_BASELINE_DIR / "N_L_kk_act_dr6_lensing_v1_baseline.txt"
LENSING_FILTER_PATH = LENSING_BASELINE_DIR / "kappa_filter_act_dr6_lensing_v1_baseline.txt"
LENSING_DOWNLOAD_URLS = {
    "kappa_alm": "https://phy-act1.princeton.edu/public/data/dr6_lensing_v1/maps/baseline/kappa_alm_data_act_dr6_lensing_v1_baseline.fits",
    "healpix_mask": "https://phy-act1.princeton.edu/public/data/dr6_lensing_v1/maps/baseline/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits",
    "noise": "https://phy-act1.princeton.edu/public/data/dr6_lensing_v1/maps/baseline/N_L_kk_act_dr6_lensing_v1_baseline.txt",
    "clkk": "https://phy-act1.princeton.edu/public/data/dr6_lensing_v1/misc/clkk.txt",
}


@dataclass(frozen=True)
class CatalogSpec:
    label: str
    pz_bin: int
    path: Path


CATALOG_SPECS = (
    CatalogSpec(
        "pz1",
        1,
        Path(
            "/pscratch/sd/b/boryanah/ACTxDESI/DESI/DESI_pz1/"
            "extended_catalog_dr10_allfoot_perbin_sigmaz0.0500.txt"
        ),
    ),
    CatalogSpec(
        "pz2",
        2,
        Path(
            "/pscratch/sd/b/boryanah/ACTxDESI/DESI/DESI_pz2/"
            "extended_catalog_dr10_allfoot_perbin_sigmaz0.0500.txt"
        ),
    ),
    CatalogSpec(
        "pz3",
        3,
        Path(
            "/pscratch/sd/b/boryanah/ACTxDESI/DESI/DESI_pz3/"
            "extended_catalog_dr10_allfoot_perbin_sigmaz0.0500.txt"
        ),
    ),
    CatalogSpec(
        "pz4",
        4,
        Path(
            "/pscratch/sd/b/boryanah/ACTxDESI/DESI/DESI_pz4/"
            "extended_catalog_dr10_allfoot_perbin_sigmaz0.0500.txt"
        ),
    ),
)


COLUMN_SCHEMA = {
    "ra_deg": {"source_column": 0, "dtype": "float32", "units": "deg"},
    "dec_deg": {"source_column": 1, "dtype": "float32", "units": "deg"},
    "z": {"source_column": 2, "dtype": "float32", "units": "dimensionless"},
    "v_los_km_s": {"source_column": 15, "dtype": "float32", "units": "km s^-1"},
    "vr_over_c": {
        "source_column": 15,
        "dtype": "float32",
        "units": "dimensionless",
        "formula": "input_column_15 / 3.0e5",
    },
    "mass_msun": {"source_column": 18, "dtype": "float64", "units": "Msun"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def ensure_dirs(outdir: Path) -> None:
    for subdir in (
        DESI_CATALOG_DIR,
        DESI_RANDOMS_DIR,
        DESI_ABACUS_CALIBRATION_DIR,
        DESI_HEALPIX_DIR,
        ACT_CMB_DIR,
        ACT_TSZ_DIR,
        ACT_LENSING_DIR,
        QUICKLOOK_DIR,
        Path("scripts"),
        Path("notebooks"),
        Path("docs"),
    ):
        (outdir / subdir).mkdir(parents=True, exist_ok=True)


def package_relative(path: Path, outdir: Path) -> str:
    try:
        return str(path.resolve().relative_to(outdir.resolve()))
    except ValueError:
        return str(path)


def atomic_h5_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def finish_atomic_write(tmp_path: Path, final_path: Path) -> None:
    os.replace(tmp_path, final_path)


def h5_1d_kwargs(dtype: np.dtype | str) -> dict:
    return {
        "dtype": dtype,
        "chunks": True,
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
    }


def write_1d(group: h5py.Group, name: str, data: np.ndarray, units: str, description: str) -> None:
    dset = group.create_dataset(name, data=data, **h5_1d_kwargs(data.dtype))
    dset.attrs["units"] = units
    dset.attrs["description"] = description
    if name in COLUMN_SCHEMA:
        dset.attrs["source_column"] = COLUMN_SCHEMA[name]["source_column"]


def set_common_attrs(h5: h5py.File, product_type: str, outdir: Path) -> None:
    h5.attrs["product_type"] = product_type
    h5.attrs["created_utc"] = utc_now()
    h5.attrs["created_by"] = Path(__file__).name
    h5.attrs["hostname"] = socket.gethostname()
    h5.attrs["python"] = sys.version
    h5.attrs["platform"] = platform.platform()
    h5.attrs["transfer_package_root"] = "."
    h5.attrs["path_convention"] = "Package-internal paths are relative to the transfer package root containing README.md."
    h5.attrs["output_directory"] = "."
    h5.attrs["nersc_reference_paper_pdf"] = str(PAPER_PDF)
    h5.attrs["nersc_source_path_note"] = "NERSC source paths are provenance only and are not required after transferring this package."


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return {
            "n_finite": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
        }
    vals = values[finite]
    return {
        "n_finite": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
    }


def save_histogram(group: h5py.Group, prefix: str, counts: np.ndarray, edges: np.ndarray) -> None:
    group.create_dataset(f"{prefix}_counts", data=counts.astype(np.int64), **h5_1d_kwargs("i8"))
    edge_ds = group.create_dataset(f"{prefix}_edges", data=edges.astype(np.float64), **h5_1d_kwargs("f8"))
    edge_ds.attrs["description"] = f"Bin edges for {prefix}_counts."


def process_catalog(
    spec: CatalogSpec,
    outdir: Path,
    z_edges: np.ndarray,
    logmass_edges: np.ndarray,
    vr_edges: np.ndarray,
    force: bool,
) -> Path:
    final_path = outdir / DESI_CATALOG_DIR / f"desi_dr10_extended_{spec.label}_compact.h5"
    if final_path.exists() and not force:
        log(f"Skipping existing catalog product {final_path}")
        return final_path

    tmp_path = atomic_h5_path(final_path)
    if tmp_path.exists():
        tmp_path.unlink()

    log(f"Loading full ASCII catalog for {spec.label}: {spec.path}")
    cat = np.loadtxt(spec.path)
    if cat.ndim != 2 or cat.shape[1] <= 18:
        raise ValueError(f"{spec.path} has unexpected shape {cat.shape}; need at least 19 columns.")

    ra = cat[:, 0].astype(np.float32, copy=True)
    dec = cat[:, 1].astype(np.float32, copy=True)
    z = cat[:, 2].astype(np.float32, copy=True)
    v_los = cat[:, 15].astype(np.float32, copy=True)
    vr_over_c = (cat[:, 15] / C_KM_S).astype(np.float32)
    mass = cat[:, 18].astype(np.float64, copy=True)
    n_rows, n_cols = cat.shape
    del cat
    gc.collect()

    finite_z = np.isfinite(z)
    finite_mass = np.isfinite(mass) & (mass > 0.0)
    finite_vr = np.isfinite(vr_over_c)
    z_counts, _ = np.histogram(z[finite_z], bins=z_edges)
    logmass_counts, _ = np.histogram(np.log10(mass[finite_mass]), bins=logmass_edges)
    vr_counts, _ = np.histogram(vr_over_c[finite_vr], bins=vr_edges)

    log(f"Writing {final_path}")
    with h5py.File(tmp_path, "w", track_order=True) as h5:
        set_common_attrs(h5, "DESI DR10 Extended compact velocity catalog", outdir)
        h5.attrs["nersc_source_ascii_path"] = str(spec.path)
        h5.attrs["photo_z_bin_label"] = spec.label
        h5.attrs["photo_z_bin_number"] = spec.pz_bin
        h5.attrs["source_n_rows"] = int(n_rows)
        h5.attrs["source_n_columns"] = int(n_cols)
        h5.attrs["column_schema_json"] = json.dumps(COLUMN_SCHEMA, indent=2)
        h5.attrs["velocity_convention"] = "vr_over_c = input column 15 / 3.0e5; no sign flip applied"
        h5.attrs["c_km_s_used"] = C_KM_S

        g = h5.create_group("catalog")
        write_1d(g, "ra_deg", ra, "deg", "Right ascension.")
        write_1d(g, "dec_deg", dec, "deg", "Declination.")
        write_1d(g, "z", z, "dimensionless", "Photometric redshift from the source catalog.")
        write_1d(g, "v_los_km_s", v_los, "km s^-1", "Line-of-sight reconstructed velocity.")
        write_1d(g, "vr_over_c", vr_over_c, "dimensionless", "Line-of-sight velocity divided by c.")
        write_1d(g, "mass_msun", mass, "Msun", "Stellar mass estimate from source column 18.")

        hist = h5.create_group("histograms")
        save_histogram(hist, "z", z_counts, z_edges)
        save_histogram(hist, "log10_mass_msun", logmass_counts, logmass_edges)
        save_histogram(hist, "vr_over_c", vr_counts, vr_edges)

        stats = h5.create_group("summary")
        stats.attrs["n_objects"] = int(n_rows)
        stats.attrs["vr_rms"] = float(np.std(vr_over_c))
        stats.attrs["z_summary_json"] = json.dumps(finite_summary(z))
        stats.attrs["vr_over_c_summary_json"] = json.dumps(finite_summary(vr_over_c))
        stats.attrs["v_los_km_s_summary_json"] = json.dumps(finite_summary(v_los))
        stats.attrs["mass_msun_summary_json"] = json.dumps(finite_summary(mass))

    finish_atomic_write(tmp_path, final_path)
    log(f"Finished {spec.label}: n={n_rows:,}, vr_rms={np.std(vr_over_c):.6e}")
    return final_path


def combine_catalogs(outdir: Path, catalog_paths: Iterable[Path], force: bool) -> Path:
    final_path = outdir / DESI_CATALOG_DIR / "desi_dr10_extended_all_pz_compact.h5"
    if final_path.exists() and not force:
        log(f"Skipping existing combined catalog product {final_path}")
        return final_path

    tmp_path = atomic_h5_path(final_path)
    if tmp_path.exists():
        tmp_path.unlink()

    catalog_paths = list(catalog_paths)
    sizes = []
    for path in catalog_paths:
        with h5py.File(path, "r") as h5:
            sizes.append(int(h5.attrs["source_n_rows"]))
    total = int(np.sum(sizes))
    starts = np.cumsum([0] + sizes[:-1]).astype(np.int64)
    stops = np.cumsum(sizes).astype(np.int64)

    log(f"Writing combined catalog with {total:,} objects")
    with h5py.File(tmp_path, "w", track_order=True) as h5:
        set_common_attrs(h5, "Combined DESI DR10 Extended compact velocity catalog", outdir)
        h5.attrs["source_catalog_hdf5_json"] = json.dumps([package_relative(path, outdir) for path in catalog_paths], indent=2)
        h5.attrs["column_schema_json"] = json.dumps(COLUMN_SCHEMA, indent=2)
        h5.attrs["velocity_convention"] = "vr_over_c = input column 15 / 3.0e5; no sign flip applied"
        h5.attrs["n_objects"] = total

        g = h5.create_group("catalog")
        datasets = {
            "ra_deg": g.create_dataset("ra_deg", shape=(total,), **h5_1d_kwargs("f4")),
            "dec_deg": g.create_dataset("dec_deg", shape=(total,), **h5_1d_kwargs("f4")),
            "z": g.create_dataset("z", shape=(total,), **h5_1d_kwargs("f4")),
            "v_los_km_s": g.create_dataset("v_los_km_s", shape=(total,), **h5_1d_kwargs("f4")),
            "vr_over_c": g.create_dataset("vr_over_c", shape=(total,), **h5_1d_kwargs("f4")),
            "mass_msun": g.create_dataset("mass_msun", shape=(total,), **h5_1d_kwargs("f8")),
            "pz_bin": g.create_dataset("pz_bin", shape=(total,), **h5_1d_kwargs("u1")),
        }
        datasets["ra_deg"].attrs["units"] = "deg"
        datasets["dec_deg"].attrs["units"] = "deg"
        datasets["z"].attrs["units"] = "dimensionless"
        datasets["v_los_km_s"].attrs["units"] = "km s^-1"
        datasets["vr_over_c"].attrs["units"] = "dimensionless"
        datasets["mass_msun"].attrs["units"] = "Msun"
        datasets["pz_bin"].attrs["description"] = "Original DESI_pz bin number, 1-4."

        slices = h5.create_group("photo_z_bin_slices")
        slices.create_dataset("label", data=np.array([p.stem.encode("utf8") for p in catalog_paths]))
        slices.create_dataset("pz_bin", data=np.array([1, 2, 3, 4], dtype=np.uint8))
        slices.create_dataset("start", data=starts)
        slices.create_dataset("stop", data=stops)
        slices.create_dataset("n_objects", data=np.array(sizes, dtype=np.int64))

        for path, start, stop, pz_bin in zip(catalog_paths, starts, stops, [1, 2, 3, 4]):
            log(f"  adding {path.name}: rows {start}:{stop}")
            with h5py.File(path, "r") as src:
                sg = src["catalog"]
                for name in ("ra_deg", "dec_deg", "z", "v_los_km_s", "vr_over_c", "mass_msun"):
                    datasets[name][start:stop] = sg[name][:]
                datasets["pz_bin"][start:stop] = np.uint8(pz_bin)

        hist = h5.create_group("histograms")
        z_edges = np.linspace(0.0, 2.0, 201)
        z_by_pz = np.zeros((4, len(z_edges) - 1), dtype=np.int64)
        z_total = np.zeros(len(z_edges) - 1, dtype=np.int64)
        vr_edges = np.linspace(-0.01, 0.01, 201)
        vr_total = np.zeros(len(vr_edges) - 1, dtype=np.int64)
        for i, (start, stop) in enumerate(zip(starts, stops)):
            z = datasets["z"][start:stop]
            vr = datasets["vr_over_c"][start:stop]
            z_by_pz[i], _ = np.histogram(z[np.isfinite(z)], bins=z_edges)
            z_total += z_by_pz[i]
            finite_vr = np.isfinite(vr)
            vr_total += np.histogram(vr[finite_vr], bins=vr_edges)[0]
        hist.create_dataset("z_by_pz_counts", data=z_by_pz, **h5_1d_kwargs("i8"))
        save_histogram(hist, "z_all", z_total, z_edges)
        save_histogram(hist, "vr_over_c_all", vr_total, vr_edges)

    finish_atomic_write(tmp_path, final_path)
    log(f"Finished combined catalog: {final_path}")
    return final_path


def plot_catalog_quicklooks(combined_path: Path, outdir: Path) -> None:
    figdir = outdir / QUICKLOOK_DIR
    with h5py.File(combined_path, "r") as h5:
        z = h5["catalog/z"][:]
        ra = h5["catalog/ra_deg"][:]
        dec = h5["catalog/dec_deg"][:]
        vr = h5["catalog/vr_over_c"][:]
        pz = h5["catalog/pz_bin"][:]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 2.0, 121)
    for pz_bin in (1, 2, 3, 4):
        sel = pz == pz_bin
        ax.hist(z[sel], bins=bins, histtype="step", lw=1.8, label=f"pz{pz_bin}")
    ax.set_xlabel("z")
    ax.set_ylabel("N")
    ax.set_title("DESI DR10 Extended redshift distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "desi_nz_by_pz.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vr[np.isfinite(vr)], bins=120, histtype="step", color="black")
    ax.set_xlabel("v_los / c")
    ax.set_ylabel("N")
    ax.set_title("Reconstructed line-of-sight velocities")
    fig.tight_layout()
    fig.savefig(figdir / "desi_vr_over_c_hist.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    finite = np.isfinite(ra) & np.isfinite(dec)
    h = ax.hist2d(ra[finite], dec[finite], bins=(720, 240), cmap="magma", cmin=1)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title("DESI catalog sky density")
    fig.colorbar(h[3], ax=ax, label="objects / bin")
    fig.tight_layout()
    fig.savefig(figdir / "desi_ra_dec_density.png", dpi=180)
    plt.close(fig)


def write_healpix_quicklooks(combined_path: Path, outdir: Path, nside: int, force: bool) -> Path:
    final_path = outdir / DESI_HEALPIX_DIR / f"desi_healpix_nside{nside}_quicklook.h5"
    if final_path.exists() and not force:
        log(f"Skipping existing HEALPix quicklook product {final_path}")
        return final_path

    tmp_path = atomic_h5_path(final_path)
    if tmp_path.exists():
        tmp_path.unlink()

    import healpy as hp

    log(f"Building DESI HEALPix quicklook maps at nside={nside}")
    with h5py.File(combined_path, "r") as h5:
        ra = h5["catalog/ra_deg"][:]
        dec = h5["catalog/dec_deg"][:]
        vr = h5["catalog/vr_over_c"][:]
        pz = h5["catalog/pz_bin"][:]

    npix = hp.nside2npix(nside)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix = hp.ang2pix(nside, theta, phi)

    counts_all = np.bincount(pix, minlength=npix).astype(np.float32)
    vsum_all = np.bincount(pix, weights=vr, minlength=npix).astype(np.float32)
    counts_by_pz = np.zeros((4, npix), dtype=np.float32)
    vsum_by_pz = np.zeros((4, npix), dtype=np.float32)
    for i, pz_bin in enumerate((1, 2, 3, 4)):
        sel = pz == pz_bin
        counts_by_pz[i] = np.bincount(pix[sel], minlength=npix).astype(np.float32)
        vsum_by_pz[i] = np.bincount(pix[sel], weights=vr[sel], minlength=npix).astype(np.float32)

    observed = counts_all > 0
    mean_count = float(np.mean(counts_all[observed]))
    delta_g = np.full(npix, np.nan, dtype=np.float32)
    velocity_weighted = np.full(npix, np.nan, dtype=np.float32)
    delta_g[observed] = counts_all[observed] / mean_count - 1.0
    velocity_weighted[observed] = vsum_all[observed] / mean_count
    velocity_weighted[observed] -= np.mean(velocity_weighted[observed])

    log(f"Writing HEALPix quicklook product {final_path}")
    with h5py.File(tmp_path, "w", track_order=True) as h5:
        set_common_attrs(h5, "DESI HEALPix quicklook maps", outdir)
        h5.attrs["source_combined_catalog"] = package_relative(combined_path, outdir)
        h5.attrs["nside"] = int(nside)
        h5.attrs["ordering"] = "RING"
        h5.attrs["velocity_convention"] = "vr_over_c = input column 15 / 3.0e5; no sign flip applied"
        h5.attrs["normalization_note"] = (
            "velocity_weighted_all = sum(vr_over_c) / mean_count over observed pixels, "
            "then mean-subtracted over observed pixels."
        )

        g = h5.create_group("maps")
        g.create_dataset("counts_all", data=counts_all, **h5_1d_kwargs("f4"))
        g.create_dataset("vsum_vr_over_c_all", data=vsum_all, **h5_1d_kwargs("f4"))
        g.create_dataset("counts_by_pz", data=counts_by_pz, compression="gzip", compression_opts=4, shuffle=True)
        g.create_dataset("vsum_vr_over_c_by_pz", data=vsum_by_pz, compression="gzip", compression_opts=4, shuffle=True)
        g.create_dataset("delta_g_all", data=delta_g, **h5_1d_kwargs("f4"))
        g.create_dataset("velocity_weighted_all", data=velocity_weighted, **h5_1d_kwargs("f4"))
        g.create_dataset("observed_mask_all", data=observed.astype(np.uint8), **h5_1d_kwargs("u1"))

    finish_atomic_write(tmp_path, final_path)

    figdir = outdir / QUICKLOOK_DIR
    hp.mollview(
        np.where(observed, delta_g, hp.UNSEEN),
        title=f"DESI count overdensity, nside={nside}",
        unit="delta_g",
        min=-1,
        max=5,
    )
    plt.savefig(figdir / f"desi_healpix_delta_g_nside{nside}.png", dpi=170, bbox_inches="tight")
    plt.close("all")

    hp.mollview(
        np.where(observed, velocity_weighted, hp.UNSEEN),
        title=f"DESI velocity-weighted tracer, nside={nside}",
        unit="sum(v/c) / mean count",
    )
    plt.savefig(figdir / f"desi_healpix_velocity_weighted_nside{nside}.png", dpi=170, bbox_inches="tight")
    plt.close("all")

    log(f"Finished HEALPix quicklooks: {final_path}")
    return final_path


def fits_header_strings(path: Path) -> list[str]:
    headers: list[str] = []
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul:
            headers.append(hdu.header.tostring(sep="\n", endcard=True, padding=False))
    return headers


def write_2d_dataset_from_enmap(
    group: h5py.Group,
    name: str,
    arr: np.ndarray,
    dtype: str,
    units: str,
    description: str,
    compression: str | None,
    compression_opts: int | None,
    block_rows: int,
) -> h5py.Dataset:
    kwargs = {"shape": arr.shape, "dtype": dtype, "chunks": (min(block_rows, arr.shape[0]), min(2048, arr.shape[1]))}
    if compression is not None:
        kwargs["compression"] = compression
        kwargs["shuffle"] = True
        if compression_opts is not None:
            kwargs["compression_opts"] = compression_opts
    dset = group.create_dataset(name, **kwargs)
    dset.attrs["units"] = units
    dset.attrs["description"] = description
    dset.attrs["axis_order"] = "pixell/enmap native (y, x) = (declination-like pixel, right-ascension-like pixel)"
    dset.attrs["stored_dtype"] = dtype
    dset.attrs["source_dtype"] = str(arr.dtype)

    for y0 in range(0, arr.shape[0], block_rows):
        y1 = min(y0 + block_rows, arr.shape[0])
        dset[y0:y1, :] = np.asarray(arr[y0:y1, :], dtype=dtype)
    return dset


def save_array_quicklook(arr: np.ndarray, png_path: Path, title: str, cmap: str, stride: int = 32) -> np.ndarray:
    sample = np.asarray(arr[::stride, ::stride], dtype=np.float32)
    finite = np.isfinite(sample)
    if np.any(finite):
        vmin, vmax = np.nanpercentile(sample[finite], [2, 98])
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(12, 3.8))
    im = ax.imshow(sample, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(f"x pixel / {stride}")
    ax.set_ylabel(f"y pixel / {stride}")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    return sample


def save_act_hdf5(outdir: Path, force: bool, block_rows: int) -> Path:
    final_path = outdir / ACT_CMB_DIR / "act_dr6_hilc_fullres_tt_17000_mask_transfer.h5"
    if final_path.exists() and not force:
        log(f"Skipping existing ACT product {final_path}")
        return final_path

    tmp_path = atomic_h5_path(final_path)
    if tmp_path.exists():
        tmp_path.unlink()

    log("Reading ACT map and mask geometries")
    map_shape, map_wcs = enmap.read_map_geometry(str(ACT_MAP_PATH))
    mask_shape, mask_wcs = enmap.read_map_geometry(str(ACT_MASK_PATH))
    if tuple(map_shape) != tuple(mask_shape):
        raise ValueError(f"Map shape {map_shape} and mask shape {mask_shape} differ.")
    if map_wcs.to_header_string() != mask_wcs.to_header_string():
        log("WARNING: map and mask WCS headers differ; both will be saved.")

    with h5py.File(tmp_path, "w", track_order=True) as h5:
        set_common_attrs(h5, "ACT DR6 CMB temperature map and analysis mask", outdir)
        h5.attrs["nersc_act_map_source_fits"] = str(ACT_MAP_PATH)
        h5.attrs["nersc_act_mask_source_fits"] = str(ACT_MASK_PATH)
        h5.attrs["map_shape"] = tuple(int(x) for x in map_shape)
        h5.attrs["storage_note"] = "CMB and mask are stored as float32 for portability and transfer size."

        geom = h5.create_group("geometry")
        geom.attrs["map_wcs_header"] = map_wcs.to_header_string()
        geom.attrs["mask_wcs_header"] = mask_wcs.to_header_string()
        geom.attrs["shape_yx"] = tuple(int(x) for x in map_shape)
        geom.attrs["projection"] = "pixell/enmap CAR WCS"
        geom.attrs["pixel_size_deg_from_wcs_cdelt"] = tuple(float(x) for x in map_wcs.wcs.cdelt)
        geom.create_dataset("map_fits_headers", data=np.array(fits_header_strings(ACT_MAP_PATH), dtype=h5py.string_dtype()))
        geom.create_dataset("mask_fits_headers", data=np.array(fits_header_strings(ACT_MASK_PATH), dtype=h5py.string_dtype()))

        maps = h5.create_group("maps")
        quick = h5.create_group("quicklook")

        log(f"Reading CMB map with pixell.enmap.read_fits: {ACT_MAP_PATH}")
        cmb = enmap.read_fits(str(ACT_MAP_PATH))
        cmb_quick = save_array_quicklook(
            cmb,
            outdir / QUICKLOOK_DIR / "act_cmb_map_stride32.png",
            "ACT DR6 HILC TT 17000 quicklook",
            "coolwarm",
            stride=32,
        )
        quick.create_dataset("cmb_stride32", data=cmb_quick, compression="gzip", compression_opts=4, shuffle=True)
        write_2d_dataset_from_enmap(
            maps,
            "cmb_temperature",
            cmb,
            "f4",
            "unknown; likely uK_CMB for ACT temperature products",
            "ACT DR6 HILC full-resolution TT map, source-free product requested by user.",
            compression=None,
            compression_opts=None,
            block_rows=block_rows,
        )
        del cmb, cmb_quick
        gc.collect()

        log(f"Reading ACT mask with pixell.enmap.read_fits: {ACT_MASK_PATH}")
        mask = enmap.read_fits(str(ACT_MASK_PATH))
        mask_quick = save_array_quicklook(
            mask,
            outdir / QUICKLOOK_DIR / "act_mask_stride32.png",
            "ACT wide mask quicklook",
            "viridis",
            stride=32,
        )
        quick.create_dataset("mask_stride32", data=mask_quick, compression="gzip", compression_opts=4, shuffle=True)
        write_2d_dataset_from_enmap(
            maps,
            "analysis_mask",
            mask,
            "f4",
            "dimensionless",
            "ACT wide GAL070 apodized extended source-free mask requested by user.",
            compression="gzip",
            compression_opts=4,
            block_rows=block_rows,
        )
        del mask, mask_quick
        gc.collect()

    finish_atomic_write(tmp_path, final_path)
    log(f"Finished ACT HDF5: {final_path}")
    return final_path


def source_manifest_entry(manifest_path: Path, filename: str) -> dict:
    if not manifest_path.exists():
        return {}
    entries = json.loads(manifest_path.read_text())
    for entry in entries:
        if entry.get("filename") == filename:
            return entry
    return {}


def write_text_dataset(group: h5py.Group, name: str, lines: np.ndarray, description: str) -> None:
    dset = group.create_dataset(name, data=lines.astype(np.float64), compression="gzip", compression_opts=4, shuffle=True)
    dset.attrs["description"] = description


def save_tsz_hdf5(outdir: Path, force: bool, block_rows: int) -> Path:
    final_path = outdir / ACT_TSZ_DIR / "act_dr6_nilc_compton_y_deproj_cib_cibdbeta_1p7_10p7_transfer.h5"
    if final_path.exists() and not force:
        log(f"Skipping existing tSZ product {final_path}")
        return final_path

    tmp_path = atomic_h5_path(final_path)
    if tmp_path.exists():
        tmp_path.unlink()

    mask_paths = {
        "footprint_mask": TSZ_FOOTPRINT_MASK_PATH,
        "inpaint_mask": TSZ_INPAINT_MASK_PATH,
        "subtracted_mask": TSZ_SUBTRACTED_MASK_PATH,
    }

    log("Reading tSZ map and NILC mask geometries")
    map_shape, map_wcs = enmap.read_map_geometry(str(TSZ_MAP_PATH))
    mask_geometries = {name: enmap.read_map_geometry(str(path)) for name, path in mask_paths.items()}
    for name, (shape, wcs) in mask_geometries.items():
        if tuple(shape) != tuple(map_shape):
            raise ValueError(f"tSZ map shape {map_shape} and {name} shape {shape} differ.")
        if wcs.to_header_string() != map_wcs.to_header_string():
            log(f"WARNING: tSZ map WCS and {name} WCS differ; both headers will be saved.")

    with h5py.File(tmp_path, "w", track_order=True) as h5:
        set_common_attrs(h5, "ACT DR6 NILC Compton-y map and masks", outdir)
        h5.attrs["nersc_tsz_map_source_fits"] = str(TSZ_MAP_PATH)
        h5.attrs["nersc_footprint_mask_source_fits"] = str(TSZ_FOOTPRINT_MASK_PATH)
        h5.attrs["nersc_inpaint_mask_source_fits"] = str(TSZ_INPAINT_MASK_PATH)
        h5.attrs["nersc_subtracted_mask_source_fits"] = str(TSZ_SUBTRACTED_MASK_PATH)
        h5.attrs["nersc_nilc_manifest_source"] = str(TSZ_MANIFEST_PATH)
        h5.attrs["map_shape"] = tuple(int(x) for x in map_shape)
        h5.attrs["recommended_default_mask"] = "masks/footprint_mask"
        h5.attrs["storage_note"] = "Compton-y and masks are stored as float32 in pixell/enmap native CAR order."
        h5.attrs["source_manifest_entries_json"] = json.dumps(
            {
                TSZ_MAP_PATH.name: source_manifest_entry(TSZ_MANIFEST_PATH, TSZ_MAP_PATH.name),
                TSZ_FOOTPRINT_MASK_PATH.name: source_manifest_entry(TSZ_MANIFEST_PATH, TSZ_FOOTPRINT_MASK_PATH.name),
                TSZ_INPAINT_MASK_PATH.name: source_manifest_entry(TSZ_MANIFEST_PATH, TSZ_INPAINT_MASK_PATH.name),
                TSZ_SUBTRACTED_MASK_PATH.name: source_manifest_entry(TSZ_MANIFEST_PATH, TSZ_SUBTRACTED_MASK_PATH.name),
            },
            indent=2,
        )

        geom = h5.create_group("geometry")
        geom.attrs["map_wcs_header"] = map_wcs.to_header_string()
        geom.attrs["shape_yx"] = tuple(int(x) for x in map_shape)
        geom.attrs["projection"] = "pixell/enmap CAR WCS"
        geom.attrs["pixel_size_deg_from_wcs_cdelt"] = tuple(float(x) for x in map_wcs.wcs.cdelt)
        geom.create_dataset("map_fits_headers", data=np.array(fits_header_strings(TSZ_MAP_PATH), dtype=h5py.string_dtype()))
        for name, path in mask_paths.items():
            shape, wcs = mask_geometries[name]
            geom.attrs[f"{name}_wcs_header"] = wcs.to_header_string()
            geom.create_dataset(
                f"{name}_fits_headers",
                data=np.array(fits_header_strings(path), dtype=h5py.string_dtype()),
            )

        maps = h5.create_group("maps")
        masks = h5.create_group("masks")
        quick = h5.create_group("quicklook")

        log(f"Reading tSZ Compton-y map with pixell.enmap.read_fits: {TSZ_MAP_PATH}")
        ymap = enmap.read_fits(str(TSZ_MAP_PATH))
        y_quick = save_array_quicklook(
            ymap,
            outdir / QUICKLOOK_DIR / "act_tsz_compton_y_stride32.png",
            "ACT DR6 NILC Compton-y quicklook",
            "coolwarm",
            stride=32,
        )
        quick.create_dataset("compton_y_stride32", data=y_quick, compression="gzip", compression_opts=4, shuffle=True)
        write_2d_dataset_from_enmap(
            maps,
            "compton_y",
            ymap,
            "f4",
            "dimensionless Compton-y",
            "ACT-Planck DR6.02 NILC Compton-y map with CIB and CIB-dBeta deprojection, beta=1.7, f=10.7.",
            compression=None,
            compression_opts=None,
            block_rows=block_rows,
        )
        del ymap, y_quick
        gc.collect()

        for name, path in mask_paths.items():
            log(f"Reading tSZ mask {name}: {path}")
            mask = enmap.read_fits(str(path))
            mask_quick = save_array_quicklook(
                mask,
                outdir / QUICKLOOK_DIR / f"act_tsz_{name}_stride32.png",
                f"ACT DR6 NILC {name.replace('_', ' ')} quicklook",
                "viridis",
                stride=32,
            )
            quick.create_dataset(f"{name}_stride32", data=mask_quick, compression="gzip", compression_opts=4, shuffle=True)
            write_2d_dataset_from_enmap(
                masks,
                name,
                mask,
                "f4",
                "dimensionless",
                f"ACT DR6 NILC {name.replace('_', ' ')} from the published NILC directory.",
                compression="gzip",
                compression_opts=4,
                block_rows=block_rows,
            )
            del mask, mask_quick
            gc.collect()

    finish_atomic_write(tmp_path, final_path)
    log(f"Finished tSZ HDF5: {final_path}")
    return final_path


def save_lensing_hdf5(outdir: Path, force: bool, block_rows: int) -> Path:
    final_path = outdir / ACT_LENSING_DIR / "act_dr6_lensing_v1_baseline_kappa_transfer.h5"
    if final_path.exists() and not force:
        log(f"Skipping existing lensing product {final_path}")
        return final_path

    tmp_path = atomic_h5_path(final_path)
    if tmp_path.exists():
        tmp_path.unlink()

    log("Reading ACT DR6 lensing kappa map geometry")
    kappa_shape, kappa_wcs = enmap.read_map_geometry(str(LENSING_KAPPA_MAP_PATH))

    with h5py.File(tmp_path, "w", track_order=True) as h5:
        set_common_attrs(h5, "ACT DR6 lensing baseline kappa map and mask", outdir)
        h5.attrs["nersc_lensing_notebook_reference"] = str(LENSING_NOTEBOOK)
        h5.attrs["nersc_kappa_map_source_fits"] = str(LENSING_KAPPA_MAP_PATH)
        h5.attrs["nersc_kappa_alm_source_fits"] = str(LENSING_KAPPA_ALM_PATH)
        h5.attrs["nersc_healpix_mask_source_fits"] = str(LENSING_HEALPIX_MASK_PATH)
        h5.attrs["nersc_noise_curve_source"] = str(LENSING_NOISE_PATH)
        h5.attrs["nersc_filter_curve_source"] = str(LENSING_FILTER_PATH)
        h5.attrs["map_shape"] = tuple(int(x) for x in kappa_shape)
        h5.attrs["storage_note"] = "Kappa map and converted CAR mask are stored as float32."
        h5.attrs["download_urls_json"] = json.dumps(LENSING_DOWNLOAD_URLS, indent=2)
        h5.attrs["notebook_filter_formula"] = (
            "Notebook demonstration reads kappa alms, uses clkk and N_L, forms "
            "filter = nan_to_num(2*clkk/(clkk+noise_kappa)/ells**2), sets ell<100 to 0, "
            "then calls pixell.curvedsky.alm2map on an enmap with the lensing mask geometry."
        )
        h5.attrs["primary_map_note"] = (
            "Primary map comes from the precomputed ACTxDESI baseline_masked CAR FITS product "
            "with the same 1 arcmin CAR geometry used in the notebook."
        )

        geom = h5.create_group("geometry")
        geom.attrs["map_wcs_header"] = kappa_wcs.to_header_string()
        geom.attrs["shape_yx"] = tuple(int(x) for x in kappa_shape)
        geom.attrs["projection"] = "pixell/enmap CAR WCS"
        geom.attrs["pixel_size_deg_from_wcs_cdelt"] = tuple(float(x) for x in kappa_wcs.wcs.cdelt)
        geom.create_dataset("kappa_map_fits_headers", data=np.array(fits_header_strings(LENSING_KAPPA_MAP_PATH), dtype=h5py.string_dtype()))
        geom.create_dataset("healpix_mask_fits_headers", data=np.array(fits_header_strings(LENSING_HEALPIX_MASK_PATH), dtype=h5py.string_dtype()))
        geom.create_dataset("kappa_alm_fits_headers", data=np.array(fits_header_strings(LENSING_KAPPA_ALM_PATH), dtype=h5py.string_dtype()))

        curves = h5.create_group("curves")
        noise = np.loadtxt(LENSING_NOISE_PATH)
        filt = np.loadtxt(LENSING_FILTER_PATH)
        write_text_dataset(curves, "N_L_kk_baseline", noise, "ACT DR6 baseline lensing noise curve; columns are ell and N_L.")
        write_text_dataset(curves, "kappa_filter_baseline", filt, "Precomputed ACTxDESI baseline kappa filter file; columns are ell and filter.")

        maps = h5.create_group("maps")
        masks = h5.create_group("masks")
        quick = h5.create_group("quicklook")

        log(f"Reading ACT lensing kappa CAR map: {LENSING_KAPPA_MAP_PATH}")
        kappa = enmap.read_fits(str(LENSING_KAPPA_MAP_PATH))
        kappa_quick = save_array_quicklook(
            kappa,
            outdir / QUICKLOOK_DIR / "act_lensing_kappa_stride16.png",
            "ACT DR6 lensing baseline kappa quicklook",
            "coolwarm",
            stride=16,
        )
        quick.create_dataset("kappa_stride16", data=kappa_quick, compression="gzip", compression_opts=4, shuffle=True)
        write_2d_dataset_from_enmap(
            maps,
            "kappa",
            kappa,
            "f4",
            "dimensionless",
            "ACT DR6 lensing v1 baseline kappa map in the notebook CAR geometry.",
            compression=None,
            compression_opts=None,
            block_rows=block_rows,
        )
        del kappa, kappa_quick
        gc.collect()

        log(f"Reading and reprojecting ACT lensing HEALPix mask: {LENSING_HEALPIX_MASK_PATH}")
        import healpy as hp
        from pixell import reproject

        mask_hp = hp.read_map(str(LENSING_HEALPIX_MASK_PATH), verbose=False)
        nside = hp.npix2nside(mask_hp.size)
        mask_car = reproject.healpix2map(mask_hp, kappa_shape, kappa_wcs)
        del mask_hp
        gc.collect()
        mask_car = np.clip(mask_car, 0.0, 1.0)
        mask_car = enmap.enmap(np.asarray(mask_car, dtype=np.float32), kappa_wcs)
        mask_quick = save_array_quicklook(
            mask_car,
            outdir / QUICKLOOK_DIR / "act_lensing_mask_stride16.png",
            "ACT DR6 lensing baseline mask quicklook",
            "viridis",
            stride=16,
        )
        quick.create_dataset("mask_stride16", data=mask_quick, compression="gzip", compression_opts=4, shuffle=True)
        mask_dset = write_2d_dataset_from_enmap(
            masks,
            "lensing_mask_apodized",
            mask_car,
            "f4",
            "dimensionless",
            "ACT DR6 lensing v1 baseline HEALPix mask reprojected to the kappa CAR geometry.",
            compression="gzip",
            compression_opts=4,
            block_rows=block_rows,
        )
        mask_dset.attrs["source_healpix_nside"] = int(nside)
        mask_dset.attrs["source_healpix_ordering"] = "RING"
        binary = enmap.enmap((np.asarray(mask_car) >= 0.9).astype(np.uint8), kappa_wcs)
        bin_dset = write_2d_dataset_from_enmap(
            masks,
            "lensing_mask_binary_ge_0p9",
            binary,
            "u1",
            "dimensionless",
            "Convenience binary mask derived from lensing_mask_apodized >= 0.9.",
            compression="gzip",
            compression_opts=4,
            block_rows=block_rows,
        )
        bin_dset.attrs["threshold"] = 0.9
        del mask_car, mask_quick, binary
        gc.collect()

    finish_atomic_write(tmp_path, final_path)
    log(f"Finished lensing HDF5: {final_path}")
    return final_path


def write_manifest(
    outdir: Path,
    catalog_paths: list[Path],
    combined_path: Path | None,
    act_path: Path | None,
    healpix_path: Path | None,
    tsz_path: Path | None,
    lensing_path: Path | None,
) -> None:
    figures = sorted(package_relative(path, outdir) for path in (outdir / QUICKLOOK_DIR).glob("*.png"))
    random_path = outdir / DESI_RANDOMS_DIR / DR10_RANDOMS_SOURCE_PATH.name
    random_sha_path = outdir / DESI_RANDOMS_DIR / DR10_RANDOMS_SHA256_SOURCE_PATH.name
    sigma_true_path = outdir / DESI_ABACUS_CALIBRATION_DIR / ABACUS_SIGMA_TRUE_PRODUCT
    manifest = {
        "created_utc": utc_now(),
        "transfer_root": ".",
        "path_convention": "All package-internal file paths are relative to the transfer package root containing README.md.",
        "products": {
            "desi_dr10_extended_velocity_catalogs": {
                "per_photoz_bin": [package_relative(path, outdir) for path in catalog_paths],
                "combined": package_relative(combined_path, outdir) if combined_path else None,
            },
            "desi_dr10_imaging_randoms": {
                "single_file": package_relative(random_path, outdir) if random_path.exists() else None,
                "sha256sum_file": package_relative(random_sha_path, outdir) if random_sha_path.exists() else None,
                "sha256": DR10_RANDOMS_1_0_SHA256,
                "recommended_use": "Build DESI galaxy angular selection masks by pixelizing RA/DEC after applying the same imaging cuts as the galaxy sample.",
                "one_file_caveat": "One DR10 random realization is appropriate for transfer-size-conscious nside=1024 masks and tests. For raw nside=4096 masks, smooth/apodize this mask or transfer more randoms-1-* files because one realization is sparse per high-resolution pixel.",
            },
            "desi_abacus_velocity_calibration": {
                "sigma_true_gas_photometric_bins": package_relative(sigma_true_path, outdir) if sigma_true_path.exists() else None,
                "recommended_use": "Use sigma_true_gas_km_s if the theory code handles the division by c; use sigma_true_gas_over_c_3e5 if the code expects a dimensionless velocity factor.",
            },
            "desi_dr10_healpix_quicklooks": package_relative(healpix_path, outdir) if healpix_path else None,
            "act_dr6_cmb_temperature": package_relative(act_path, outdir) if act_path else None,
            "act_dr6_tsz_compton_y": package_relative(tsz_path, outdir) if tsz_path else None,
            "act_dr6_lensing_kappa": package_relative(lensing_path, outdir) if lensing_path else None,
        },
        "quicklook_figures": figures,
        "scripts": {
            "prepare_act_desi_ksz_hdf5": "scripts/prepare_act_desi_ksz_hdf5.py",
            "prepare_des_y3_shear_maps": "scripts/prepare_des_y3_shear_maps.py",
        },
        "notebooks": {
            "prepare_act_desi_ksz_hdf5": "notebooks/prepare_act_desi_ksz_hdf5.ipynb",
            "prepare_des_y3_shear_maps": "notebooks/prepare_des_y3_shear_maps.ipynb",
        },
        "documentation": {
            "readme": "README.md",
            "des_y3_shear_maps": "docs/DES_Y3_SHEAR_MAPS.md",
            "desi_dr10_randoms": "docs/DESI_DR10_RANDOMS.md",
            "desi_abacus_sigma_true_gas": "docs/DESI_ABACUS_SIGMA_TRUE_GAS.md",
        },
        "nersc_source_provenance": {
            "note": "These absolute paths describe where the products were built on the NERSC filesystem. They are provenance only and are not needed after transferring this package.",
            "desi_dr10_extended_velocity_catalogs": [str(spec.path) for spec in CATALOG_SPECS],
            "desi_dr10_imaging_randoms": {
                "copied_random": str(DR10_RANDOMS_SOURCE_PATH),
                "full_random_family": str(DR10_RANDOMS_SOURCE_DIR / "randoms-1-{0..19}.fits"),
                "sha256sum_file": str(DR10_RANDOMS_SHA256_SOURCE_PATH),
            },
            "desi_abacus_velocity_calibration": str(ABACUS_SIGMA_TRUE_SOURCE_PATH),
            "act_dr6_cmb_temperature_map": str(ACT_MAP_PATH),
            "act_dr6_cmb_temperature_mask": str(ACT_MASK_PATH),
            "act_dr6_tsz_compton_y_map": str(TSZ_MAP_PATH),
            "act_dr6_tsz_masks": {
                "footprint": str(TSZ_FOOTPRINT_MASK_PATH),
                "inpaint": str(TSZ_INPAINT_MASK_PATH),
                "subtracted": str(TSZ_SUBTRACTED_MASK_PATH),
            },
            "act_dr6_lensing": {
                "kappa_map": str(LENSING_KAPPA_MAP_PATH),
                "kappa_alm": str(LENSING_KAPPA_ALM_PATH),
                "healpix_mask": str(LENSING_HEALPIX_MASK_PATH),
                "noise_curve": str(LENSING_NOISE_PATH),
                "filter_curve": str(LENSING_FILTER_PATH),
                "processing_notebook": str(LENSING_NOTEBOOK),
            },
            "ksz_reference_paper_pdf": str(PAPER_PDF),
        },
        "velocity_convention": "vr_over_c = source column 15 / 3.0e5; no sign flip applied",
        "required_later_analysis_packages": ["numpy", "h5py", "astropy", "pixell", "healpy", "pymaster/NaMaster"],
    }
    path = outdir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"Wrote manifest {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Transfer-product output directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing HDF5 products.")
    parser.add_argument("--skip-catalogs", action="store_true", help="Do not process DESI ASCII catalogs.")
    parser.add_argument("--skip-act", action="store_true", help="Do not process the ACT map/mask FITS files.")
    parser.add_argument("--skip-tsz", action="store_true", help="Do not process the ACT DR6 NILC Compton-y map/masks.")
    parser.add_argument("--skip-lensing", action="store_true", help="Do not process the ACT DR6 lensing kappa map/mask.")
    parser.add_argument("--block-rows", type=int, default=128, help="Rows per block when writing large 2D maps.")
    parser.add_argument("--healpix-nside", type=int, default=512, help="NSIDE for galaxy HEALPix quicklook maps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    ensure_dirs(outdir)

    z_edges = np.linspace(0.0, 2.0, 201)
    logmass_edges = np.linspace(8.0, 13.5, 221)
    vr_edges = np.linspace(-0.01, 0.01, 201)

    catalog_paths: list[Path] = []
    combined_path: Path | None = None
    act_path: Path | None = None
    healpix_path: Path | None = None
    tsz_path: Path | None = None
    lensing_path: Path | None = None

    if not args.skip_catalogs:
        for spec in CATALOG_SPECS:
            catalog_paths.append(process_catalog(spec, outdir, z_edges, logmass_edges, vr_edges, args.force))
        combined_path = combine_catalogs(outdir, catalog_paths, args.force)
        plot_catalog_quicklooks(combined_path, outdir)
        healpix_path = write_healpix_quicklooks(combined_path, outdir, args.healpix_nside, args.force)

    if not args.skip_act:
        act_path = save_act_hdf5(outdir, args.force, args.block_rows)

    if not args.skip_tsz:
        tsz_path = save_tsz_hdf5(outdir, args.force, args.block_rows)

    if not args.skip_lensing:
        lensing_path = save_lensing_hdf5(outdir, args.force, args.block_rows)

    write_manifest(outdir, catalog_paths, combined_path, act_path, healpix_path, tsz_path, lensing_path)
    log("All requested products are complete.")


if __name__ == "__main__":
    main()
