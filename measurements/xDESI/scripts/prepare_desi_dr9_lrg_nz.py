#!/usr/bin/env python
"""Prepare theory-ready DESI DR9 Extended LRG redshift distributions.

This script reconciles three ingredients:

* the public Zhou et al. DR9 Extended-LRG spectroscopic-calibrated dN/dz table;
* the exact DR9 Extended velocity catalogs used in this transfer package;
* the extra kSZ-style photo-z uncertainty cut, Z_PHOT_STD <= 0.05*(1+zphot),
  that defines the ``sigmaz0.0500`` velocity products.

The official redshift-distribution table is the safest calibration anchor.  The
``zphot_std0p05_spec_ratio_corrected`` product keeps that official calibration
but applies a smooth ratio measured from the public spectroscopic subset before
renormalizing to the exact catalog surface density implied by the public quality
cuts plus the photo-z uncertainty cut.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_ROOT = Path("/global/cfs/cdirs/lsst/www/shivamp/desi/lrg_xcorr_2023/v1")
PUBLIC_CATALOG = PUBLIC_ROOT / "catalogs/dr9_extended_lrg_pzbins.fits"
PUBLIC_PZ = PUBLIC_ROOT / "catalogs/more/dr9_extended_lrg_pz.fits"
PUBLIC_DNDZ = PUBLIC_ROOT / "redshift_dist/extended_lrg_pz_dndz_iron_v0.4_dz_0.02.txt"
STARDENS = PUBLIC_ROOT / "misc/pixweight-dr7.1-0.22.0_stardens_64_ring.fits"

TRANSFER_CATALOG_DIR = PACKAGE_ROOT / "data/desi_dr9_extended_velocity_catalogs"
OUT_DIR = PACKAGE_ROOT / "data/desi_dr9_redshift_distributions"
DOC_PATH = PACKAGE_ROOT / "docs/DESI_DR9_EXTENDED_LRG_NZ.md"
FIG_DIR = PACKAGE_ROOT / "quicklook_figures"

OUT_H5 = OUT_DIR / "desi_dr9_extended_lrg_sigmaz0p05_true_nz.h5"
OUT_CSV = OUT_DIR / "desi_dr9_extended_lrg_sigmaz0p05_true_nz.csv"
OUT_SUMMARY = OUT_DIR / "desi_dr9_extended_lrg_sigmaz0p05_true_nz_summary.json"
OUT_PLOT = FIG_DIR / "desi_dr9_extended_lrg_sigmaz0p05_true_nz.png"

MIN_LRG_NOBS = 2
MAX_EBV = 0.15
MAX_STARDENS = 2500.0
STARDENS_NSIDE = 64
ZSTD_FACTOR = 0.05


PAPER_KSZ_COUNTS = {
    "dr9_extended_fiducial_outlier_and_photoz_correction_act_overlap": {
        "pz1": 954820,
        "pz2": 1628650,
        "pz3": 2125787,
        "pz4": 1996525,
        "all": 6697792,
    },
    "dr9_extended_photoz_correction_only_act_overlap": {
        "pz1": 963631,
        "pz2": 1658313,
        "pz3": 2174053,
        "pz4": 2054075,
        "all": 6850072,
    },
}

KSZ_FIDUCIAL_LABEL = "dr9_extended_fiducial_outlier_and_photoz_correction_act_overlap"
KSZ_PHOTOZ_ONLY_LABEL = "dr9_extended_photoz_correction_only_act_overlap"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def package_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PACKAGE_ROOT.resolve()))
    except ValueError:
        return str(path)


def load_bad_stardens_pixels() -> np.ndarray:
    with fits.open(STARDENS, memmap=True) as hdul:
        tab = hdul[1].data
        bad = tab["HPXPIXEL"][np.asarray(tab["STARDENS"]) >= MAX_STARDENS]
    return np.asarray(np.sort(bad), dtype=np.int64)


def quality_mask(tab, bad_stardens_pixels: np.ndarray) -> np.ndarray:
    ra = np.asarray(tab["RA"])
    dec = np.asarray(tab["DEC"])
    pix = hp.ang2pix(STARDENS_NSIDE, ra, dec, lonlat=True, nest=False)
    good = ~((dec < -10.5) & (ra > 120.0) & (ra < 260.0))
    good &= (
        (np.asarray(tab["PIXEL_NOBS_G"]) >= MIN_LRG_NOBS)
        & (np.asarray(tab["PIXEL_NOBS_R"]) >= MIN_LRG_NOBS)
        & (np.asarray(tab["PIXEL_NOBS_Z"]) >= MIN_LRG_NOBS)
    )
    good &= np.asarray(tab["lrg_mask"]) == 0
    good &= np.asarray(tab["EBV"]) < MAX_EBV
    good &= ~np.isin(pix, bad_stardens_pixels)
    return good


def unit_integral_density(n_per_deg2_bin: np.ndarray, dz: np.ndarray) -> np.ndarray:
    total = np.sum(n_per_deg2_bin)
    density = n_per_deg2_bin / dz
    if total > 0:
        density = density / total
    return density


def stats_for_distribution(zmid: np.ndarray, n_per_deg2_bin: np.ndarray) -> dict:
    total = float(np.sum(n_per_deg2_bin))
    if total <= 0:
        return {"surface_density_per_deg2": 0.0, "mean_z": None, "sigma_z": None}
    mean = float(np.sum(zmid * n_per_deg2_bin) / total)
    sigma = float(np.sqrt(np.sum((zmid - mean) ** 2 * n_per_deg2_bin) / total))
    return {"surface_density_per_deg2": total, "mean_z": mean, "sigma_z": sigma}


def stats_for_count_distribution(zmid: np.ndarray, n_per_redshift_bin: np.ndarray) -> dict:
    total = float(np.sum(n_per_redshift_bin))
    if total <= 0:
        return {"object_count": 0.0, "mean_z": None, "sigma_z": None}
    mean = float(np.sum(zmid * n_per_redshift_bin) / total)
    sigma = float(np.sqrt(np.sum((zmid - mean) ** 2 * n_per_redshift_bin) / total))
    return {"object_count": total, "mean_z": mean, "sigma_z": sigma}


def read_transfer_counts() -> dict:
    out = {}
    for b in range(1, 5):
        path = TRANSFER_CATALOG_DIR / f"desi_dr9_extended_pz{b}_compact_with_weights.h5"
        with h5py.File(path, "r") as h5:
            g = h5["catalog"]
            valid = g["valid_for_cl"][:]
            photsys = g["photsys"][:]
            weight = g["weight_imaging_mean1"][:]
            out[f"pz{b}"] = {
                "transfer_hdf5": package_relative(path),
                "n_rows": int(g["z"].shape[0]),
                "n_valid_for_cl": int(np.count_nonzero(valid)),
                "sum_weight_imaging_mean1_valid_for_cl": float(np.sum(weight[valid])),
                "n_valid_photsys_north": int(np.count_nonzero(valid & (photsys == b"N"))),
                "n_valid_photsys_south": int(np.count_nonzero(valid & (photsys == b"S"))),
            }
    out["all"] = {
        "n_rows": int(sum(out[f"pz{b}"]["n_rows"] for b in range(1, 5))),
        "n_valid_for_cl": int(sum(out[f"pz{b}"]["n_valid_for_cl"] for b in range(1, 5))),
    }
    return out


def read_photoz_histograms(edges: np.ndarray) -> dict:
    out = {}
    for b in range(1, 5):
        path = TRANSFER_CATALOG_DIR / f"desi_dr9_extended_pz{b}_compact_with_weights.h5"
        with h5py.File(path, "r") as h5:
            g = h5["catalog"]
            valid = g["valid_for_cl"][:]
            z = g["z"][:]
            weight = g["weight_imaging_mean1"][:]
        counts, _ = np.histogram(z[valid], bins=edges)
        weighted, _ = np.histogram(z[valid], bins=edges, weights=weight[valid])
        out[f"pz{b}"] = {"counts": counts, "weighted_counts": weighted}
    return out


def build_ksz_count_scaled_products(products: dict) -> dict:
    """Scale the calibrated true-z shape to kSZ-paper final sample counts."""
    base = products["zphot_std0p05_spec_ratio_corrected"]
    count_products: dict[str, dict] = {}
    for sample_label, counts in PAPER_KSZ_COUNTS.items():
        count_products[sample_label] = {}
        for b in range(1, 5):
            key = f"pz{b}"
            shape = np.asarray(base[key], dtype=np.float64)
            if np.sum(shape) <= 0:
                count_products[sample_label][key] = np.zeros_like(shape)
            else:
                count_products[sample_label][key] = counts[key] * shape / np.sum(shape)
    return count_products


def make_products() -> tuple[dict, dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    official = np.genfromtxt(PUBLIC_DNDZ, names=True)
    zmin = official["zmin"].astype(np.float64)
    zmax = official["zmax"].astype(np.float64)
    zmid = 0.5 * (zmin + zmax)
    dz = zmax - zmin
    edges = np.r_[zmin, zmax[-1]]

    transfer_counts = read_transfer_counts()
    photoz_hists = read_photoz_histograms(edges)

    bad_stardens_pixels = load_bad_stardens_pixels()
    public_counts: dict[str, dict] = {}
    products: dict[str, dict] = {
        "published_full_extended_lrg": {},
        "catalog_rescaled_same_shape": {},
        "zphot_std0p05_spec_ratio_corrected": {},
        "photoz_histogram_valid_for_cl_diagnostic": {},
    }

    with fits.open(PUBLIC_CATALOG, memmap=True) as cat_hdul, fits.open(PUBLIC_PZ, memmap=True) as pz_hdul:
        tab = cat_hdul[1].data
        pztab = pz_hdul[1].data
        pz_bin = np.asarray(tab["pz_bin"])
        zphot = np.asarray(tab["Z_PHOT_MEDIAN"])
        zstd = np.asarray(pztab["Z_PHOT_STD"])
        zspec = np.asarray(pztab["Z_SPEC"])
        quality = quality_mask(tab, bad_stardens_pixels)
        zstd_cut = zstd <= ZSTD_FACTOR * (1.0 + zphot)
        spec_good = zspec > 0.0

        for b in range(1, 5):
            key = f"pz{b}"
            public_bin = pz_bin == b
            public_quality = public_bin & quality
            public_zstd_quality = public_quality & zstd_cut
            spec_quality = public_quality & spec_good
            spec_zstd_quality = public_zstd_quality & spec_good

            full_hist, _ = np.histogram(zspec[spec_quality], bins=edges)
            cut_hist, _ = np.histogram(zspec[spec_zstd_quality], bins=edges)

            official_n = official[f"bin_{b}_combined"].astype(np.float64)
            exact_fraction = (
                np.count_nonzero(public_zstd_quality) / np.count_nonzero(public_quality)
            )
            exact_nbar = float(np.sum(official_n) * exact_fraction)

            global_spec_fraction = (
                np.count_nonzero(spec_zstd_quality) / np.count_nonzero(spec_quality)
                if np.count_nonzero(spec_quality)
                else exact_fraction
            )
            # Ratio regularization keeps empty noisy tails from dominating while
            # still letting the spectroscopic subset inform the shape change.
            alpha = 10.0
            ratio = (cut_hist + alpha * global_spec_fraction) / (full_hist + alpha)
            ratio = gaussian_filter1d(ratio.astype(np.float64), sigma=1.0, mode="nearest")
            ratio = np.clip(ratio, 0.0, 1.25)
            corrected = official_n * ratio
            if np.sum(corrected) > 0:
                corrected *= exact_nbar / np.sum(corrected)

            rescaled = official_n * exact_fraction

            products["published_full_extended_lrg"][key] = official_n
            products["catalog_rescaled_same_shape"][key] = rescaled
            products["zphot_std0p05_spec_ratio_corrected"][key] = corrected
            products["photoz_histogram_valid_for_cl_diagnostic"][key] = photoz_hists[key][
                "weighted_counts"
            ].astype(np.float64)

            public_counts[key] = {
                "public_pz_bin_count": int(np.count_nonzero(public_bin)),
                "public_quality_count": int(np.count_nonzero(public_quality)),
                "public_quality_and_zphot_std0p05_count": int(
                    np.count_nonzero(public_zstd_quality)
                ),
                "public_quality_and_zphot_std0p05_fraction": float(exact_fraction),
                "spec_quality_count": int(np.count_nonzero(spec_quality)),
                "spec_quality_and_zphot_std0p05_count": int(
                    np.count_nonzero(spec_zstd_quality)
                ),
                "spec_quality_and_zphot_std0p05_fraction": float(global_spec_fraction),
                "transfer_valid_for_cl_count": transfer_counts[key]["n_valid_for_cl"],
                "matches_transfer_valid_for_cl": bool(
                    abs(
                        np.count_nonzero(public_zstd_quality)
                        - transfer_counts[key]["n_valid_for_cl"]
                    )
                    <= 2
                ),
            }

    ksz_count_products = build_ksz_count_scaled_products(products)

    summary: dict[str, object] = {
        "created_utc": utc_now(),
        "created_by": Path(__file__).name,
        "public_calibration_source": str(PUBLIC_DNDZ),
        "public_catalog_source": str(PUBLIC_CATALOG),
        "public_photoz_source": str(PUBLIC_PZ),
        "transfer_catalog_dir": package_relative(TRANSFER_CATALOG_DIR),
        "recommended_theory_group": (
            f"ksz_paper_scaled_counts/{KSZ_FIDUCIAL_LABEL}"
        ),
        "recommended_shape_group": "zphot_std0p05_spec_ratio_corrected",
        "recommended_normalized_kernel_dataset": "nz_unit_integral",
        "recommended_ksz_count_dataset": "n_per_redshift_bin_count",
        "units_note": (
            "n_per_deg2_bin is bin-integrated surface density in galaxies deg^-2 "
            "per 0.02-wide redshift bin. dndz_per_deg2 = n_per_deg2_bin/dz. "
            "n_per_redshift_bin_count in ksz_paper_scaled_counts is bin-integrated "
            "object count for the kSZ-paper sample. dndz_count = count/dz. "
            "nz_unit_integral integrates to 1 over dz and is usually the theory-code input."
        ),
        "public_counts": public_counts,
        "transfer_counts": transfer_counts,
        "paper_ksz_counts": PAPER_KSZ_COUNTS,
        "distribution_stats": {},
        "ksz_count_scaled_stats": {},
    }

    with h5py.File(OUT_H5, "w") as h5:
        h5.attrs["product_type"] = "DESI DR9 Extended LRG true n(z) products for theory"
        h5.attrs["created_utc"] = summary["created_utc"]
        h5.attrs["created_by"] = Path(__file__).name
        h5.attrs["hostname"] = socket.gethostname()
        h5.attrs["python"] = sys.version
        h5.attrs["platform"] = platform.platform()
        h5.attrs["path_convention"] = (
            "Package-internal paths are relative to the transfer package root containing README.md."
        )
        h5.attrs["nersc_source_path_note"] = (
            "Absolute paths stored here are NERSC provenance only and are not required after transfer."
        )
        h5.attrs["recommended_theory_group"] = summary["recommended_theory_group"]
        h5.attrs["recommended_shape_group"] = summary["recommended_shape_group"]
        h5.attrs["recommended_normalized_kernel_dataset"] = summary[
            "recommended_normalized_kernel_dataset"
        ]
        h5.attrs["recommended_ksz_count_dataset"] = summary["recommended_ksz_count_dataset"]
        h5.attrs["units_note"] = summary["units_note"]
        h5.attrs["public_calibration_source_nersc"] = str(PUBLIC_DNDZ)
        h5.attrs["public_catalog_source_nersc"] = str(PUBLIC_CATALOG)
        h5.attrs["public_photoz_source_nersc"] = str(PUBLIC_PZ)

        bins = h5.create_group("redshift_bins")
        bins.create_dataset("zmin", data=zmin)
        bins.create_dataset("zmax", data=zmax)
        bins.create_dataset("zmid", data=zmid)
        bins.create_dataset("dz", data=dz)

        for method, by_bin in products.items():
            g = h5.create_group(method)
            if method == "published_full_extended_lrg":
                g.attrs["description"] = (
                    "Official Zhou et al. calibrated spectroscopic N(z) for the full "
                    "DR9 Extended LRG tomographic sample."
                )
            elif method == "catalog_rescaled_same_shape":
                g.attrs["description"] = (
                    "Official shape rescaled by the public quality+Z_PHOT_STD cut fraction. "
                    "Use for sensitivity tests if you want the official shape unchanged."
                )
            elif method == "zphot_std0p05_spec_ratio_corrected":
                g.attrs["description"] = (
                    "Recommended exact-catalog estimate: official calibrated N(z) multiplied "
                    "by a smooth spectroscopic-subset ratio for the Z_PHOT_STD <= 0.05*(1+zphot) "
                    "cut, then renormalized to the exact catalog surface density."
                )
            else:
                g.attrs["description"] = (
                    "Diagnostic photometric-redshift histogram from the transfer HDF5 catalog. "
                    "Do not use this as true redshift N(z)."
                )

            for key, nbin in by_bin.items():
                bg = g.create_group(key)
                bg.create_dataset("n_per_deg2_bin", data=nbin)
                bg.create_dataset("dndz_per_deg2", data=nbin / dz)
                bg.create_dataset("nz_unit_integral", data=unit_integral_density(nbin, dz))
                stats = stats_for_distribution(zmid, nbin)
                for attr_key, attr_val in stats.items():
                    if attr_val is not None:
                        bg.attrs[attr_key] = attr_val
                summary["distribution_stats"].setdefault(method, {})[key] = stats

        ksz_group = h5.create_group("ksz_paper_scaled_counts")
        ksz_group.attrs["description"] = (
            "Calibrated true-redshift shapes scaled to the DR9 Extended kSZ-paper "
            "ACT-overlap sample counts. These groups are the recommended abundance "
            "normalization when the same kSZ-cleaned sample is used for harmonic "
            "auto/cross spectra."
        )
        ksz_group.attrs["shape_source_group"] = "zphot_std0p05_spec_ratio_corrected"
        for sample_label, by_bin in ksz_count_products.items():
            sg = ksz_group.create_group(sample_label)
            sg.attrs["paper_count_all"] = PAPER_KSZ_COUNTS[sample_label]["all"]
            sg.attrs["counts_source"] = (
                "2407.07152v2 Table I / Appendix G DR9 Extended sample counts. "
                "The fiducial label uses both outlier and photo-z correction; "
                "the photoz-only label omits velocity-outlier cleaning."
            )
            for b in range(1, 5):
                key = f"pz{b}"
                count_bin = by_bin[key]
                bg = sg.create_group(key)
                bg.create_dataset("n_per_redshift_bin_count", data=count_bin)
                bg.create_dataset("dndz_count", data=count_bin / dz)
                bg.create_dataset("nz_unit_integral", data=unit_integral_density(count_bin, dz))
                bg.attrs["paper_object_count"] = PAPER_KSZ_COUNTS[sample_label][key]
                bg.attrs["allfoot_valid_for_cl_count"] = transfer_counts[key]["n_valid_for_cl"]
                bg.attrs["count_scale_vs_allfoot_valid_for_cl"] = (
                    PAPER_KSZ_COUNTS[sample_label][key]
                    / transfer_counts[key]["n_valid_for_cl"]
                )
                stats = stats_for_count_distribution(zmid, count_bin)
                for attr_key, attr_val in stats.items():
                    if attr_val is not None:
                        bg.attrs[attr_key] = attr_val
                summary["ksz_count_scaled_stats"].setdefault(sample_label, {})[key] = stats

        meta = h5.create_group("metadata")
        meta.attrs["summary_json"] = json.dumps(summary, indent=2)

    write_csv(products, ksz_count_products, zmin, zmax, zmid, dz)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    make_plot(products, zmid)
    write_doc(summary)
    update_manifest()
    return summary, products


def write_csv(products: dict, ksz_count_products: dict, zmin, zmax, zmid, dz) -> None:
    fieldnames = ["zmin", "zmax", "zmid", "dz"]
    for method in products:
        for b in range(1, 5):
            prefix = f"{method}_pz{b}"
            fieldnames.extend(
                [
                    f"{prefix}_n_per_deg2_bin",
                    f"{prefix}_dndz_per_deg2",
                    f"{prefix}_nz_unit_integral",
                ]
            )
    for sample_label in ksz_count_products:
        for b in range(1, 5):
            prefix = f"ksz_paper_scaled_counts_{sample_label}_pz{b}"
            fieldnames.extend(
                [
                    f"{prefix}_n_per_redshift_bin_count",
                    f"{prefix}_dndz_count",
                    f"{prefix}_nz_unit_integral",
                ]
            )

    with OUT_CSV.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(zmid)):
            row = {
                "zmin": zmin[i],
                "zmax": zmax[i],
                "zmid": zmid[i],
                "dz": dz[i],
            }
            for method, by_bin in products.items():
                for b in range(1, 5):
                    vals = by_bin[f"pz{b}"]
                    prefix = f"{method}_pz{b}"
                    row[f"{prefix}_n_per_deg2_bin"] = vals[i]
                    row[f"{prefix}_dndz_per_deg2"] = vals[i] / dz[i]
                    row[f"{prefix}_nz_unit_integral"] = unit_integral_density(vals, dz)[i]
            for sample_label, by_bin in ksz_count_products.items():
                for b in range(1, 5):
                    vals = by_bin[f"pz{b}"]
                    prefix = f"ksz_paper_scaled_counts_{sample_label}_pz{b}"
                    row[f"{prefix}_n_per_redshift_bin_count"] = vals[i]
                    row[f"{prefix}_dndz_count"] = vals[i] / dz[i]
                    row[f"{prefix}_nz_unit_integral"] = unit_integral_density(vals, dz)[i]
            writer.writerow(row)


def make_plot(products: dict, zmid: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    for ax, b in zip(axes.ravel(), range(1, 5)):
        key = f"pz{b}"
        ax.plot(
            zmid,
            products["published_full_extended_lrg"][key],
            color="0.35",
            lw=1.8,
            label="published full",
        )
        ax.plot(
            zmid,
            products["catalog_rescaled_same_shape"][key],
            color="#4C78A8",
            lw=1.5,
            ls="--",
            label="rescaled same shape",
        )
        ax.plot(
            zmid,
            products["zphot_std0p05_spec_ratio_corrected"][key],
            color="#F58518",
            lw=1.8,
            label="recommended",
        )
        ax.set_title(f"DR9 Extended pz{b}")
        ax.set_ylabel(r"$N(z)$ [deg$^{-2}$ per bin]")
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("true redshift z")
    axes[1, 1].set_xlabel("true redshift z")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=180)
    plt.close(fig)


def write_doc(summary: dict) -> None:
    stats = summary["distribution_stats"]["zphot_std0p05_spec_ratio_corrected"]
    ksz_stats = summary["ksz_count_scaled_stats"][KSZ_FIDUCIAL_LABEL]
    rows = []
    ksz_rows = []
    for b in range(1, 5):
        key = f"pz{b}"
        s = stats[key]
        c = summary["public_counts"][key]
        rows.append(
            f"| {b} | {c['transfer_valid_for_cl_count']:,} | "
            f"{s['surface_density_per_deg2']:.3f} | {s['mean_z']:.4f} | {s['sigma_z']:.4f} |"
        )
        ks = ksz_stats[key]
        ksz_rows.append(
            f"| {b} | {int(round(ks['object_count'])):,} | "
            f"{ks['mean_z']:.4f} | {ks['sigma_z']:.4f} | "
            f"{summary['paper_ksz_counts'][KSZ_FIDUCIAL_LABEL][key] / c['transfer_valid_for_cl_count']:.4f} |"
        )
    text = f"""# DESI DR9 Extended LRG n(z)

This note documents the redshift distributions prepared for the exact DR9
Extended LRG catalog used in this transfer package and the final kSZ-cleaned
DR9 Extended sample counts from the real-space kSZ paper.

## Recommendation

Use:

```text
data/desi_dr9_redshift_distributions/desi_dr9_extended_lrg_sigmaz0p05_true_nz.h5
```

and the HDF5 group:

```text
ksz_paper_scaled_counts/{KSZ_FIDUCIAL_LABEL}/pz{{1,2,3,4}}/nz_unit_integral
```

as the default theory-code normalized `n(z)` for the final DR9 Extended
kSZ-cleaned sample.  This dataset is a probability density in redshift; it is
normalized so that `sum(nz_unit_integral * dz) = 1`.

If the theory code wants the final kSZ sample abundance rather than a
unit-normalized kernel, use:

```text
ksz_paper_scaled_counts/{KSZ_FIDUCIAL_LABEL}/pz{{1,2,3,4}}/n_per_redshift_bin_count
```

This sums to the DR9 Extended fiducial kSZ-paper counts in the ACT overlap.
The corresponding per-unit-redshift count distribution is:

```text
ksz_paper_scaled_counts/{KSZ_FIDUCIAL_LABEL}/pz{{1,2,3,4}}/dndz_count
```

The shape source for this final kSZ group is
`zphot_std0p05_spec_ratio_corrected`, i.e. the public spectroscopic-calibrated
Extended-LRG `N(z)` corrected for the `Z_PHOT_STD <= 0.05*(1+Z_PHOT_MEDIAN)`
cut.  The additional velocity-outlier and ACT-overlap selection is represented
as a count normalization from the kSZ paper, because the reduced stacking
products do not retain object IDs or spectroscopic redshift calibrators for an
object-level remeasurement of the shape.

## Why Not Use The Catalog Photo-z Histogram?

The catalog `catalog/z` values are `Z_PHOT_MEDIAN`, i.e. the photometric
redshift used to assign the tomographic bin.  The original sample paper
explicitly recommends using DESI spectroscopic redshift distributions for
tomographic analyses, rather than individual photo-z values.  Therefore the
photo-z histogram is saved only as a diagnostic.

## Exact Sample Match

The transfer catalog rows used for galaxy auto/cross spectra are selected by:

```text
DR9 Extended LRG tomographic bin
DR9 LRG quality footprint cuts
Z_PHOT_STD <= 0.05 * (1 + Z_PHOT_MEDIAN)
finite positive public imaging weight
```

Applying those cuts to the public DR9 Extended LRG catalog reproduces the
`valid_for_cl` HDF5 counts to integer precision:

| pz bin | valid_for_cl objects | N(z) surface density [deg^-2] | mean true z | sigma true z |
|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Final kSZ-Cleaned Count Scaling

The real-space kSZ paper reports smaller DR9 Extended counts because those are
for the overlap with ACT and, for the fiducial sample, with both photo-z
correction and velocity-outlier cleaning.  For consistency with a single HOD
fit to the kSZ-cleaned sample, the `ksz_paper_scaled_counts` group rescales the
calibrated shape to those paper counts:

| pz bin | fiducial kSZ-cleaned objects | mean true z | sigma true z | count/all-footprint scale |
|---:|---:|---:|---:|---:|
{chr(10).join(ksz_rows)}

## HDF5 Groups

- `published_full_extended_lrg`: official Zhou et al. calibrated
  spectroscopic `N(z)` for the full DR9 Extended LRG tomographic sample.
- `catalog_rescaled_same_shape`: same official shape, rescaled by the exact
  `Z_PHOT_STD` cut fraction in the public catalog.
- `zphot_std0p05_spec_ratio_corrected`: recommended exact-catalog estimate.
  It anchors to the official calibrated `N(z)`, applies a smooth ratio measured
  from the public spectroscopic subset for the `Z_PHOT_STD` cut, then
  renormalizes to the exact catalog surface density.
- `ksz_paper_scaled_counts/{KSZ_FIDUCIAL_LABEL}`: default final kSZ-cleaned
  theory product.  It uses the same calibrated shape and scales it to the
  DR9 Extended fiducial counts from the kSZ paper.
- `ksz_paper_scaled_counts/{KSZ_PHOTOZ_ONLY_LABEL}`: same idea, but scaled to
  the Appendix G "photo-z correction only" counts for sensitivity checks.
- `photoz_histogram_valid_for_cl_diagnostic`: weighted histogram of
  `Z_PHOT_MEDIAN` from the transfer HDF5 catalogs.  This is diagnostic only.

## Provenance

NERSC-only source paths:

```text
{PUBLIC_DNDZ}
{PUBLIC_CATALOG}
{PUBLIC_PZ}
{STARDENS}
```

Package-relative products:

```text
{package_relative(OUT_H5)}
{package_relative(OUT_CSV)}
{package_relative(OUT_SUMMARY)}
{package_relative(OUT_PLOT)}
```
"""
    DOC_PATH.write_text(text)


def update_manifest() -> None:
    manifest_path = PACKAGE_ROOT / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {}
    manifest.setdefault("products", {})
    manifest["products"]["desi_dr9_redshift_distributions"] = {
        "extended_lrg_sigmaz0p05_true_nz_hdf5": package_relative(OUT_H5),
        "extended_lrg_sigmaz0p05_true_nz_csv": package_relative(OUT_CSV),
        "extended_lrg_sigmaz0p05_true_nz_summary": package_relative(OUT_SUMMARY),
        "recommended_group": f"ksz_paper_scaled_counts/{KSZ_FIDUCIAL_LABEL}",
        "recommended_dataset": "nz_unit_integral",
        "recommended_count_dataset": "n_per_redshift_bin_count",
        "shape_source_group": "zphot_std0p05_spec_ratio_corrected",
    }
    manifest.setdefault("documentation", {})
    manifest["documentation"]["desi_dr9_extended_lrg_nz"] = package_relative(DOC_PATH)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PACKAGE_ROOT, help="Transfer package root.")
    args = parser.parse_args()
    if args.root.resolve() != PACKAGE_ROOT.resolve():
        raise ValueError("This script currently expects to live inside the transfer package.")

    required = [PUBLIC_CATALOG, PUBLIC_PZ, PUBLIC_DNDZ, STARDENS]
    required.extend(
        TRANSFER_CATALOG_DIR / f"desi_dr9_extended_pz{b}_compact_with_weights.h5"
        for b in range(1, 5)
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    summary, _ = make_products()
    print(f"Wrote {OUT_H5}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {DOC_PATH}")
    print("Recommended group:", summary["recommended_theory_group"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
