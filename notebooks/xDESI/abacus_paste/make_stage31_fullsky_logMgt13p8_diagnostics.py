#!/usr/bin/env python
"""Measure and visualize the Stage-31 high-mass full-sky paste products.

This script is intentionally diagnostic/lightweight: it reads the four
combined HDF5 map products, builds galaxy overdensity maps from the pasted
catalogs, measures full-sky healpy bandpowers to lmax=1024, and writes PNG
preview panels that a notebook can display without recomputing spectra.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/godmax-matplotlib-cache")

import h5py
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = REPO_ROOT / "data/xDESI/processed/abacus_backlight/stage31_fullsky_logMgt13p8"
MAP_ROOT = RUN_ROOT / "maps"
MEAS_ROOT = RUN_ROOT / "measurements"
PLOT_ROOT = RUN_ROOT / "plots"

NSIDE = 1024
L_MAX = 1024
ELL_MIN = 8
N_BINS = 20
FULL_SKY_AREA_DEG2 = 4.0 * math.pi * (180.0 / math.pi) ** 2

PZ_PRODUCTS = {
    "pz1": {
        "label": "pz1",
        "z_range": (0.3, 0.62),
        "path": MAP_ROOT
        / "stage31_fullsky_pz1_logMgt13p8"
        / "abacus_pasted_maps_pz1fullsky_z0p3_0p62_logMgt13p8_nside1024.h5",
    },
    "pz2": {
        "label": "pz2",
        "z_range": (0.43110627652982897, 0.8035931265106552),
        "path": MAP_ROOT
        / "stage31_fullsky_pz2_logMgt13p8"
        / "abacus_pasted_maps_pz2fullsky_z0p431_0p804_logMgt13p8_nside1024.h5",
    },
    "pz3": {
        "label": "pz3",
        "z_range": (0.63, 0.98),
        "path": MAP_ROOT
        / "stage31_fullsky_pz3_logMgt13p8"
        / "abacus_pasted_maps_pz3fullsky_z0p63_0p98_logMgt13p8_nside1024.h5",
    },
    "pz4": {
        "label": "pz4",
        "z_range": (0.7131674616590881, 1.1898555882069786),
        "path": MAP_ROOT
        / "stage31_fullsky_pz4_logMgt13p8"
        / "abacus_pasted_maps_pz4fullsky_z0p713_1p19_logMgt13p8_nside1024.h5",
    },
}

FIELD_SPECS = {
    "map_ymap": {"short": "y", "title": "tSZ y", "kind": "positive", "cmap": "magma"},
    "map_tau": {"short": "tau", "title": "tau", "kind": "positive", "cmap": "viridis"},
    "map_ksz": {"short": "ksz", "title": "kSZ", "kind": "signed", "cmap": "coolwarm"},
    "map_kappa_cmb": {"short": "kappa_cmb", "title": "CMB kappa", "kind": "signed", "cmap": "coolwarm"},
    "map_kappa_wl": {"short": "kappa_wl1", "title": "WL kappa s1", "kind": "signed", "cmap": "coolwarm"},
    "map_kappa_wl_tomo2": {"short": "kappa_wl2", "title": "WL kappa s2", "kind": "signed", "cmap": "coolwarm"},
    "map_kappa_wl_tomo3": {"short": "kappa_wl3", "title": "WL kappa s3", "kind": "signed", "cmap": "coolwarm"},
    "map_kappa_wl_tomo4": {"short": "kappa_wl4", "title": "WL kappa s4", "kind": "signed", "cmap": "coolwarm"},
}

PANEL_FIELDS = ("delta_g", "map_ymap", "map_tau", "map_ksz", "map_kappa_cmb", "map_kappa_wl")


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def map_stats(values: np.ndarray) -> Dict[str, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return {key: float("nan") for key in ("min", "p01", "p50", "p99", "max", "mean", "std", "nonzero_fraction")}
    return {
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1.0)),
        "p50": float(np.percentile(finite, 50.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "nonzero_fraction": float(np.count_nonzero(finite) / finite.size),
    }


def display_range(values: np.ndarray, kind: str) -> Tuple[float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return -1.0, 1.0
    nonzero = finite[finite != 0.0]
    sample = nonzero if nonzero.size > 100 else finite
    if kind == "positive":
        hi = float(np.percentile(sample, 99.5))
        if not np.isfinite(hi) or hi <= 0.0:
            hi = float(np.max(sample)) if sample.size else 1.0
        return 0.0, hi if hi > 0.0 else 1.0
    scale = float(np.percentile(np.abs(sample), 99.5))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(np.abs(sample))) if sample.size else 1.0
    return -scale, scale if scale > 0.0 else 1.0


def galaxy_maps(galaxies: np.ndarray, nside: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    npix = hp.nside2npix(nside)
    valid_flag = galaxies[:, 5] > 0.5 if galaxies.size else np.zeros(0, dtype=bool)
    finite_coords = np.isfinite(galaxies[:, 0]) & np.isfinite(galaxies[:, 1]) if galaxies.size else valid_flag
    physical_coords = (
        finite_coords
        & (galaxies[:, 0] >= 0.0)
        & (galaxies[:, 0] < 360.0)
        & (galaxies[:, 1] >= -90.0)
        & (galaxies[:, 1] <= 90.0)
    ) if galaxies.size else valid_flag
    valid = valid_flag & physical_coords
    gals = galaxies[valid]
    counts = np.zeros(npix, dtype=np.float32)
    if len(gals):
        theta = np.deg2rad(90.0 - np.asarray(gals[:, 1], dtype=np.float64))
        phi = np.deg2rad(np.asarray(gals[:, 0], dtype=np.float64) % 360.0)
        pix = hp.ang2pix(nside, theta, phi)
        np.add.at(counts, pix, 1.0)
    mean = float(np.mean(counts))
    delta = counts / mean - 1.0 if mean > 0.0 else counts.copy()
    stats = {
        "n_galaxies_valid": int(len(gals)),
        "n_galaxies_valid_flag": int(np.count_nonzero(valid_flag)),
        "n_galaxies_excluded_bad_coordinates": int(np.count_nonzero(valid_flag & ~physical_coords)),
        "n_galaxies_total_rows": int(len(galaxies)),
        "mean_per_pixel": mean,
        "surface_density_per_deg2": float(len(gals) / FULL_SKY_AREA_DEG2),
        "shot_noise_4pi_over_ngal": float(4.0 * math.pi / len(gals)) if len(gals) else float("nan"),
    }
    return counts, delta.astype(np.float32), stats


def bin_edges(ell_min: int, ell_max: int, n_bins: int) -> List[Tuple[int, int]]:
    raw = np.linspace(ell_min, ell_max + 1, n_bins + 1)
    edges = np.unique(np.round(raw).astype(int))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            out.append((int(lo), int(hi)))
    return out


def binned_rows(
    pz: str,
    spectrum: str,
    field: str,
    ell: np.ndarray,
    cl: np.ndarray,
    edges: Iterable[Tuple[int, int]],
) -> List[Dict[str, float | str | int]]:
    rows = []
    dell = ell * (ell + 1.0) * cl / (2.0 * math.pi)
    weights_all = 2.0 * ell + 1.0
    for lo, hi in edges:
        mask = (ell >= lo) & (ell < hi) & np.isfinite(cl)
        if not np.any(mask):
            continue
        weights = weights_all[mask]
        cl_band = float(np.average(cl[mask], weights=weights))
        dell_band = float(np.average(dell[mask], weights=weights))
        rows.append(
            {
                "pz_bin": pz,
                "spectrum": spectrum,
                "field": field,
                "ell_min": int(lo),
                "ell_max_exclusive": int(hi),
                "ell_eff": float(np.average(ell[mask], weights=weights)),
                "cl": cl_band,
                "dell": dell_band,
            }
        )
    return rows


def binned_corr_rows(
    pz: str,
    field: str,
    ell: np.ndarray,
    cl_gx: np.ndarray,
    cl_gg: np.ndarray,
    cl_xx: np.ndarray,
    edges: Iterable[Tuple[int, int]],
) -> List[Dict[str, float | str | int]]:
    rows = []
    weights_all = 2.0 * ell + 1.0
    for lo, hi in edges:
        mask = (ell >= lo) & (ell < hi) & np.isfinite(cl_gx) & np.isfinite(cl_gg) & np.isfinite(cl_xx)
        if not np.any(mask):
            continue
        weights = weights_all[mask]
        gx = float(np.average(cl_gx[mask], weights=weights))
        gg = float(np.average(cl_gg[mask], weights=weights))
        xx = float(np.average(cl_xx[mask], weights=weights))
        denom = math.sqrt(max(gg, 0.0) * max(xx, 0.0))
        rows.append(
            {
                "pz_bin": pz,
                "spectrum": "r_gx",
                "field": field,
                "ell_min": int(lo),
                "ell_max_exclusive": int(hi),
                "ell_eff": float(np.average(ell[mask], weights=weights)),
                "cl": float(gx / denom) if denom > 0.0 else float("nan"),
                "dell": float("nan"),
            }
        )
    return rows


def pixel_pearson(delta_g: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(delta_g, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return float("nan")
    x = x[mask] - float(np.mean(x[mask]))
    y = y[mask] - float(np.mean(y[mask]))
    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    return float(np.dot(x, y) / denom) if denom > 0.0 else float("nan")


def overview_png(pz: str, panel_maps: Mapping[str, np.ndarray], out_path: Path, plot_nside: int = 256) -> None:
    fig = plt.figure(figsize=(15, 7.8))
    for idx, field in enumerate(PANEL_FIELDS, start=1):
        values = np.asarray(panel_maps[field], dtype=np.float32)
        if hp.get_nside(values) != plot_nside:
            values = hp.ud_grade(values, plot_nside, power=0).astype(np.float32)
        if field == "delta_g":
            lo, hi = display_range(values, "signed")
            title = f"{pz} galaxy overdensity"
            cmap = "RdBu_r"
            unit = "delta_g"
        else:
            spec = FIELD_SPECS[field]
            lo, hi = display_range(values, spec["kind"])
            title = f"{pz} {spec['title']}"
            cmap = spec["cmap"]
            unit = spec["short"]
        hp.mollview(
            values,
            fig=fig.number,
            sub=(2, 3, idx),
            title=title,
            min=lo,
            max=hi,
            cmap=cmap,
            unit=unit,
            cbar=True,
            notext=True,
        )
    fig.suptitle(f"Stage-31 full-sky high-mass paste ({pz}, nside {NSIDE}, log10M >= 13.8)", y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def spectra_overview_png(bandpower_csv: Path, out_path: Path) -> None:
    rows = []
    with bandpower_csv.open() as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    plot_fields = [
        ("g_map_ymap", "map_ymap", "g x y"),
        ("g_map_tau", "map_tau", "g x tau"),
        ("g_map_kappa_cmb", "map_kappa_cmb", "g x CMB kappa"),
        ("gg_shot_sub", "galaxies", "g x g shot-sub"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (spectrum, field, title) in zip(axes.ravel(), plot_fields):
        for pz in PZ_PRODUCTS:
            sub = [
                row
                for row in rows
                if row["pz_bin"] == pz and row["spectrum"] == spectrum and row["field"] == field
            ]
            if not sub:
                continue
            ell = np.array([float(row["ell_eff"]) for row in sub])
            dell = np.array([float(row["dell"]) for row in sub])
            ax.plot(ell, dell, marker="o", ms=3, lw=1.2, label=pz)
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel(r"$D_\ell$")
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\ell$")
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("Full-sky high-mass pasted-map diagnostic bandpowers")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    MEAS_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    out_npz = MEAS_ROOT / "stage31_fullsky_logMgt13p8_raw_cls_lmax1024.npz"
    out_csv = MEAS_ROOT / "stage31_fullsky_logMgt13p8_bandpowers_lmax1024_nbin20.csv"
    out_json = MEAS_ROOT / "stage31_fullsky_logMgt13p8_diagnostic_summary.json"

    ell = np.arange(L_MAX + 1, dtype=np.float64)
    # Keep this diagnostic self-contained. Some healpy installs try to fetch
    # pixel-window FITS files over the network; raw spectra are enough for the
    # relative checks and preview plots produced here.
    pixwin2 = np.ones(L_MAX + 1, dtype=np.float64)
    edges = bin_edges(ELL_MIN, L_MAX, N_BINS)
    arrays: Dict[str, np.ndarray] = {"ell": ell.astype(np.int32), "pixwin2": pixwin2.astype(np.float64)}
    band_rows: List[Dict[str, float | str | int]] = []
    summary = {
        "run_root": str(RUN_ROOT),
        "nside": NSIDE,
        "lmax": L_MAX,
        "ell_min": ELL_MIN,
        "n_bins": N_BINS,
        "mass_cut_log10M_hMsun": 13.8,
        "full_sky_area_deg2": FULL_SKY_AREA_DEG2,
        "pixel_window_deconvolved": False,
        "spectrum_note": "Raw healpy.anafast spectra; no pixel-window deconvolution applied.",
        "products": {},
        "outputs": {
            "raw_cls_npz": str(out_npz),
            "bandpowers_csv": str(out_csv),
            "summary_json": str(out_json),
            "plots": [],
        },
    }

    for pz, meta in PZ_PRODUCTS.items():
        path = Path(meta["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"[diagnostics] {pz}: reading {path}")
        with h5py.File(path, "r") as handle:
            galaxies = np.asarray(handle["galaxies"], dtype=np.float32)
            map_keys = sorted(handle["maps"].keys())
            attrs = {key: _json_safe(value) for key, value in handle.attrs.items()}

            _, delta_g, gal_stats = galaxy_maps(galaxies, NSIDE)
            cl_gg_raw = hp.anafast(delta_g, lmax=L_MAX, iter=0)
            shot = float(gal_stats["shot_noise_4pi_over_ngal"])
            cl_gg_deconv = cl_gg_raw / pixwin2
            cl_gg_shot_sub = (cl_gg_raw - shot) / pixwin2

            arrays[f"{pz}_gg_with_shot"] = cl_gg_deconv.astype(np.float64)
            arrays[f"{pz}_gg_shot_sub"] = cl_gg_shot_sub.astype(np.float64)
            band_rows.extend(binned_rows(pz, "gg_with_shot", "galaxies", ell, cl_gg_deconv, edges))
            band_rows.extend(binned_rows(pz, "gg_shot_sub", "galaxies", ell, cl_gg_shot_sub, edges))

            product_summary = {
                "map_h5": str(path),
                "h5_attrs": attrs,
                "available_maps": map_keys,
                "z_range": list(meta["z_range"]),
                "galaxies": gal_stats,
                "map_stats": {"delta_g": map_stats(delta_g)},
                "pixel_pearson_with_delta_g": {},
                "plots": {},
            }

            panel_maps = {"delta_g": delta_g}
            for map_key, spec in FIELD_SPECS.items():
                if map_key not in handle["maps"]:
                    continue
                values = np.nan_to_num(np.asarray(handle["maps"][map_key], dtype=np.float32))
                product_summary["map_stats"][map_key] = map_stats(values)
                product_summary["pixel_pearson_with_delta_g"][map_key] = pixel_pearson(delta_g, values)

                cl_gx = hp.anafast(delta_g, values, lmax=L_MAX, iter=0) / pixwin2
                cl_xx = hp.anafast(values, lmax=L_MAX, iter=0) / pixwin2
                arrays[f"{pz}_g_{map_key}"] = cl_gx.astype(np.float64)
                arrays[f"{pz}_auto_{map_key}"] = cl_xx.astype(np.float64)
                band_rows.extend(binned_rows(pz, f"g_{map_key}", map_key, ell, cl_gx, edges))
                band_rows.extend(binned_rows(pz, f"auto_{map_key}", map_key, ell, cl_xx, edges))
                band_rows.extend(binned_corr_rows(pz, map_key, ell, cl_gx, cl_gg_deconv, cl_xx, edges))

                if map_key in PANEL_FIELDS:
                    panel_maps[map_key] = values
                del values

            png = PLOT_ROOT / f"stage31_fullsky_logMgt13p8_{pz}_map_overview.png"
            overview_png(pz, panel_maps, png)
            product_summary["plots"]["map_overview_png"] = str(png)
            summary["outputs"]["plots"].append(str(png))
            summary["products"][pz] = product_summary
            del delta_g, galaxies, panel_maps

    fieldnames = ["pz_bin", "spectrum", "field", "ell_min", "ell_max_exclusive", "ell_eff", "cl", "dell"]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(band_rows)
    np.savez_compressed(out_npz, **arrays)

    spectra_png = PLOT_ROOT / "stage31_fullsky_logMgt13p8_bandpowers_overview.png"
    spectra_overview_png(out_csv, spectra_png)
    summary["outputs"]["plots"].append(str(spectra_png))

    with out_json.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=_json_safe)

    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
