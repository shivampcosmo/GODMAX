#!/usr/bin/env python
"""Measure DES Y3 tomo-4 shear Cls and a paper-style decoupled covariance.

This standalone script starts from the transfer-package DES Y3 shear map HDF5,
measures the tomo-4 x tomo-4 shear auto spectrum with NaMaster, builds a
decoupled Gaussian bandpower covariance, and writes the paper-style comparison
plot used in the tomo-4 covariance diagnostic.

No shear maps are modified.  The script only reads the HDF5 map product and
writes diagnostic outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/act_desi_ksz_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/act_desi_ksz_xdgcache")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover - fallback for minimal environments
    gaussian_filter1d = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional paper-panel overlay
    Image = None


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHEAR_H5 = PACKAGE_ROOT / "data/des_y3_shear_maps/des_y3_metacal_shear_maps_nside1024.h5"
DEFAULT_OUTDIR = PACKAGE_ROOT / "diagnostics/des_y3_shear_tomo4_covariance"
DEFAULT_PAPER_PANEL = Path("/global/cfs/cdirs/lsst/www/shivamp/DESI/shear_4_4_paper_DES.png")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def package_relative(path: Path, root: Path = PACKAGE_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def make_nmt_bins(
    ell_left: np.ndarray,
    ell_right: np.ndarray,
    nside: int,
    pixwin_pol: np.ndarray,
    apply_pixel_window: bool = True,
) -> nmt.NmtBin:
    """Create a NaMaster binning object with optional shear pixel-window correction."""

    ell_left = np.asarray(ell_left, dtype=np.int32)
    ell_right = np.asarray(ell_right, dtype=np.int32)
    lmax = int(ell_right[-1] - 1)

    f_ell = None
    if apply_pixel_window:
        f_ell = np.ones(lmax + 1, dtype=np.float64)
        usable = np.asarray(pixwin_pol[: lmax + 1], dtype=np.float64)
        good = usable > 0
        f_ell[good] = 1.0 / usable[good] ** 2
        f_ell[~good] = 0.0

    # Newer NaMaster accepts f_ell in from_edges.  The NERSC environment used
    # here accepts f_ell through the direct constructor only.
    try:
        if f_ell is None:
            return nmt.NmtBin.from_edges(ell_left, ell_right, is_Dell=False)
        return nmt.NmtBin.from_edges(ell_left, ell_right, is_Dell=False, f_ell=f_ell)
    except TypeError:
        ells = np.arange(lmax + 1, dtype=np.int32)
        bpws = -np.ones(lmax + 1, dtype=np.int32)
        weights = np.zeros(lmax + 1, dtype=np.float64)
        for ib, (lo, hi) in enumerate(zip(ell_left, ell_right)):
            lo = max(int(lo), 0)
            hi = min(int(hi), lmax + 1)
            bpws[lo:hi] = ib
            weights[lo:hi] = 1.0
        return nmt.NmtBin(
            nside=nside,
            ells=ells,
            bpws=bpws,
            weights=weights,
            lmax=lmax,
            is_Dell=False,
            f_ell=f_ell,
        )


def make_spin2_field(mask: np.ndarray, gamma1: np.ndarray, gamma2_namaster: np.ndarray, lmax: int) -> nmt.NmtField:
    """Build the DES shear spin-2 field in NaMaster convention."""

    kwargs = {
        "spin": 2,
        "purify_e": False,
        "purify_b": False,
        "n_iter": 0,
        "lmax_sht": int(lmax),
        "lite": True,
    }
    try:
        return nmt.NmtField(mask, [gamma1, gamma2_namaster], **kwargs)
    except TypeError:
        kwargs.pop("lite", None)
        return nmt.NmtField(mask, [gamma1, gamma2_namaster], **kwargs)


def make_noise_template(noise_level: float, lmax: int) -> np.ndarray:
    """Flat spin-2 shape-noise pseudo-Cl template in EE and BB."""

    noise = np.zeros((4, int(lmax) + 1), dtype=np.float64)
    noise[0, :] = float(noise_level)
    noise[3, :] = float(noise_level)
    return noise


def decouple_with_noise(workspace: nmt.NmtWorkspace, coupled_cell: np.ndarray, noise_template: np.ndarray) -> np.ndarray:
    """Decouple pseudo-Cl and subtract the pseudo-noise template."""

    try:
        return workspace.decouple_cell(coupled_cell, cl_noise=noise_template)
    except TypeError:
        return workspace.decouple_cell(coupled_cell - noise_template)


def smooth_positive_bandpowers(values: np.ndarray, floor: float = 1.0e-20) -> np.ndarray:
    """Smooth positive bandpowers for theory-like covariance input."""

    values = np.maximum(np.asarray(values, dtype=np.float64), floor)
    if gaussian_filter1d is None or values.size < 5:
        return values
    return np.exp(gaussian_filter1d(np.log(values), sigma=1.0, mode="nearest"))


def bandpowers_to_full_ell(
    cl_bpw: np.ndarray,
    ell_left: np.ndarray,
    ell_right: np.ndarray,
    lmax: int,
) -> np.ndarray:
    """Expand constant-in-bandpower spectra to full-ell arrays for covariance."""

    cl_bpw = np.asarray(cl_bpw, dtype=np.float64)
    full = np.zeros((cl_bpw.shape[0], int(lmax) + 1), dtype=np.float64)
    for ib, (lo, hi) in enumerate(zip(ell_left, ell_right)):
        lo = max(int(lo), 0)
        hi = min(int(hi), int(lmax) + 1)
        full[:, lo:hi] = cl_bpw[:, ib][:, None]
    if int(ell_left[0]) > 0:
        full[:, : int(ell_left[0])] = cl_bpw[:, 0][:, None]
    return full


def sanitize_total_cls(cl_full: np.ndarray, zero_cross: bool = True) -> np.ndarray:
    """Ensure EE/BB total covariance inputs are positive and finite."""

    out = np.nan_to_num(np.asarray(cl_full, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0).copy()
    positive = out[[0, 3], :]
    positive = positive[positive > 0]
    floor = float(np.min(positive) * 1.0e-6) if positive.size else 1.0e-20
    out[0, :] = np.maximum(out[0, :], floor)
    out[3, :] = np.maximum(out[3, :], floor)
    if zero_cross:
        out[1, :] = 0.0
        out[2, :] = 0.0
    return out


def load_tomo_maps(shear_h5: Path, tomo_index: int, mask_name: str) -> dict:
    """Load the map arrays and metadata needed for the tomo auto spectrum."""

    group_name = f"maps/tomo{tomo_index}"
    with h5py.File(shear_h5, "r") as h5:
        if group_name not in h5:
            raise KeyError(f"{group_name} not found in {shear_h5}")
        group = h5[group_name]
        if mask_name not in group:
            raise KeyError(f"{mask_name} not found in {group_name}")
        nside = int(h5.attrs["nside"])
        ell_left = h5["bandpowers/ell_left"][:].astype(np.int32)
        ell_right = h5["bandpowers/ell_right"][:].astype(np.int32)
        pixwin_pol = h5["pixel_window/polarization"][:]
        noise_attr = {
            "mask_weight_raw": "shape_noise_pseudo_cl_raw_weight_mask",
            "mask_weight": "shape_noise_pseudo_cl_normalized_weight_mask",
            "mask_binary": "shape_noise_pseudo_cl_binary_mask",
        }[mask_name]
        return {
            "nside": nside,
            "mask": group[mask_name][:].astype(np.float64),
            "gamma1": group["gamma1"][:].astype(np.float64),
            "gamma2_namaster": group["gamma2_namaster"][:].astype(np.float64),
            "ell_left": ell_left,
            "ell_right": ell_right,
            "pixwin_pol": pixwin_pol,
            "noise_level": float(group.attrs[noise_attr]),
            "noise_attr": noise_attr,
            "n_valid_sources": int(group.attrs["n_valid_sources"]),
            "area_observed_deg2_binary": float(group.attrs["area_observed_deg2_binary"]),
            "n_eff_per_arcmin2_binary_area": float(group.attrs["n_eff_per_arcmin2_binary_area"]),
        }


def measure_cls_and_covariance(
    maps: dict,
    apply_pixel_window: bool = True,
    covariance_input: str = "smoothed_decoupled_total",
) -> dict:
    """Measure tomo auto Cls and a decoupled NaMaster Gaussian covariance."""

    ell_left = maps["ell_left"]
    ell_right = maps["ell_right"]
    nside = int(maps["nside"])
    lmax = int(ell_right[-1] - 1)
    bins = make_nmt_bins(ell_left, ell_right, nside, maps["pixwin_pol"], apply_pixel_window)
    ell_eff = bins.get_effective_ells()

    field = make_spin2_field(maps["mask"], maps["gamma1"], maps["gamma2_namaster"], lmax)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, bins, n_iter=0, lmax_mask=lmax)

    coupled_cell = nmt.compute_coupled_cell(field, field)
    noise = make_noise_template(maps["noise_level"], lmax)
    cl_signal = decouple_with_noise(workspace, coupled_cell, noise)
    noise_decoupled = workspace.decouple_cell(noise)
    cl_total_bpw = cl_signal + noise_decoupled

    if covariance_input == "measured_decoupled_total":
        covariance_bpw = cl_total_bpw.copy()
        covariance_bpw[1, :] = 0.0
        covariance_bpw[2, :] = 0.0
    elif covariance_input == "smoothed_decoupled_total":
        covariance_bpw = np.zeros_like(cl_total_bpw)
        covariance_bpw[0, :] = smooth_positive_bandpowers(cl_total_bpw[0, :])
        covariance_bpw[3, :] = smooth_positive_bandpowers(np.maximum(cl_total_bpw[3, :], noise_decoupled[3, :]))
    else:
        raise ValueError("covariance_input must be 'smoothed_decoupled_total' or 'measured_decoupled_total'")

    covariance_full = sanitize_total_cls(bandpowers_to_full_ell(covariance_bpw, ell_left, ell_right, lmax))

    covariance_workspace = nmt.NmtCovarianceWorkspace()
    covariance_workspace.compute_coupling_coefficients(field, field, field, field, lmax=lmax, n_iter=0)
    covariance = nmt.gaussian_covariance(
        covariance_workspace,
        2,
        2,
        2,
        2,
        covariance_full,
        covariance_full,
        covariance_full,
        covariance_full,
        workspace,
        workspace,
        coupled=False,
    )

    nbpw = len(ell_eff)
    expected_shape = (4 * nbpw, 4 * nbpw)
    if covariance.shape != expected_shape:
        raise RuntimeError(
            f"Expected decoupled bandpower covariance shape {expected_shape}, got {covariance.shape}. "
            "Do not use coupled=True for this paper-style plot."
        )
    ee_covariance = covariance.reshape(nbpw, 4, nbpw, 4)[:, 0, :, 0]
    ee_error = np.sqrt(np.maximum(np.diag(ee_covariance), 0.0))

    return {
        "ell_eff": ell_eff,
        "ell_left": ell_left,
        "ell_right": ell_right,
        "cl_signal": cl_signal,
        "noise_decoupled": noise_decoupled,
        "cl_total_bpw": cl_total_bpw,
        "covariance_input_bpw": covariance_bpw,
        "covariance_input_full": covariance_full,
        "covariance": covariance,
        "ee_covariance": ee_covariance,
        "ee_error": ee_error,
        "covariance_shape": covariance.shape,
        "lmax": lmax,
    }


def plot_paper_style(
    result: dict,
    out_png: Path,
    paper_panel: Path | None = None,
    covariance_label: str = "smoothed_decoupled_total",
) -> None:
    """Write the paper-style tomo-4 x tomo-4 comparison plot."""

    ell_eff = result["ell_eff"]
    cl_signal = result["cl_signal"]
    ee_error = result["ee_error"]
    x = np.sqrt(ell_eff)
    y_ee = ell_eff * cl_signal[0] * 1.0e7
    yerr_ee = ell_eff * ee_error * 1.0e7
    y_bb = ell_eff * cl_signal[3] * 1.0e7

    include_panel = paper_panel is not None and paper_panel.exists() and Image is not None
    if include_panel:
        fig = plt.figure(figsize=(12.2, 4.5))
        grid = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.18)
        ax = fig.add_subplot(grid[0, 0])
        ax_img = fig.add_subplot(grid[0, 1])
    else:
        fig, ax = plt.subplots(figsize=(7.1, 4.5))
        ax_img = None

    ax.errorbar(
        x,
        y_ee,
        yerr=yerr_ee,
        fmt="o",
        ms=4.2,
        color="#1266c3",
        ecolor="#1266c3",
        capsize=0,
        lw=1.4,
        label="DES Y3 tomo 4 x 4",
    )
    ax.plot(x, y_bb, color="0.55", lw=1.0, alpha=0.85, label="BB noise-subtracted")
    ax.axhline(0, color="0.2", lw=0.8, ls="--")

    # Visual scale-cut bands to mimic the DES Y3 Fig. 4 style.  These bands are
    # for plot comparison only; they are not used in the covariance calculation.
    for lo, hi, alpha in [(200, 300, 0.10), (400, 2048, 0.18)]:
        ax.axvspan(np.sqrt(lo), np.sqrt(hi), color="0.5", alpha=alpha, zorder=0)

    xticks = np.asarray([0, 100, 400, 900, 1600], dtype=float)
    ax.set_xticks(np.sqrt(xticks))
    ax.set_xticklabels([str(int(t)) for t in xticks])
    ax.set_xlim(0, np.sqrt(2048) * 1.02)
    ax.set_ylim(-1.2, 8.6)
    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$\ell C_\ell^{EE}\ (10^{-7})$")
    ax.set_title("This measurement: corrected decoupled covariance")
    ax.text(0.03, 0.95, "4,4", transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.text(
        0.03,
        0.84,
        f"covariance: {covariance_label.replace('_', ' ')}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax.legend(loc="lower left", fontsize=8, frameon=True)

    if include_panel and ax_img is not None:
        img = Image.open(paper_panel)
        width, height = img.size
        crop = img.crop((int(0.765 * width), 0, width, height))
        ax_img.imshow(crop)
        ax_img.set_title("Paper Fig. 4, 4x4 panel")
        ax_img.axis("off")

    fig.suptitle("DES Y3 tomo 4 shear auto spectrum: paper-style error-bar check", y=1.02, fontsize=13)
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shear-h5", type=Path, default=DEFAULT_SHEAR_H5, help="DES Y3 shear HDF5 map product.")
    parser.add_argument("--tomo-index", type=int, default=3, help="Zero-based tomographic index. 3 is DES source bin 4.")
    parser.add_argument(
        "--mask-name",
        choices=("mask_weight_raw", "mask_weight", "mask_binary"),
        default="mask_weight_raw",
        help="Mask dataset to use. The matching shape-noise attribute is selected automatically.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument(
        "--output-prefix",
        default="tomo4_cls_paper_style_corrected_covariance",
        help="Basename for PNG/NPZ/JSON outputs.",
    )
    parser.add_argument(
        "--covariance-input",
        choices=("smoothed_decoupled_total", "measured_decoupled_total"),
        default="smoothed_decoupled_total",
        help="Full-ell total Cl model to pass to gaussian_covariance(..., coupled=False).",
    )
    parser.add_argument(
        "--paper-panel",
        type=Path,
        default=DEFAULT_PAPER_PANEL,
        help="Optional paper Fig. 4 crop. If missing, the plot is made without the reference panel.",
    )
    parser.add_argument("--no-paper-panel", action="store_true", help="Do not include the paper-panel image.")
    parser.add_argument("--no-pixel-window", action="store_true", help="Disable the stored HEALPix polarization pixel-window correction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_png = args.outdir / f"{args.output_prefix}.png"
    out_npz = args.outdir / f"{args.output_prefix}_arrays.npz"
    out_json = args.outdir / f"{args.output_prefix}_summary.json"

    print(f"[{utc_now()}] Loading DES shear maps from {args.shear_h5}", flush=True)
    maps = load_tomo_maps(args.shear_h5, args.tomo_index, args.mask_name)

    print(f"[{utc_now()}] Measuring Cls and decoupled covariance", flush=True)
    result = measure_cls_and_covariance(
        maps,
        apply_pixel_window=not args.no_pixel_window,
        covariance_input=args.covariance_input,
    )

    paper_panel = None if args.no_paper_panel else args.paper_panel
    print(f"[{utc_now()}] Writing plot to {out_png}", flush=True)
    plot_paper_style(result, out_png, paper_panel=paper_panel, covariance_label=args.covariance_input)

    np.savez_compressed(
        out_npz,
        ell_eff=result["ell_eff"],
        ell_left=result["ell_left"],
        ell_right=result["ell_right"],
        cl_signal=result["cl_signal"],
        noise_decoupled=result["noise_decoupled"],
        cl_total_bpw=result["cl_total_bpw"],
        covariance_input_bpw=result["covariance_input_bpw"],
        ee_covariance=result["ee_covariance"],
        ee_error=result["ee_error"],
        ell_cl_ee_1e7=result["ell_eff"] * result["cl_signal"][0] * 1.0e7,
        ell_error_ee_1e7=result["ell_eff"] * result["ee_error"] * 1.0e7,
        ell_cl_bb_1e7=result["ell_eff"] * result["cl_signal"][3] * 1.0e7,
    )

    summary = {
        "created_utc": utc_now(),
        "created_by": Path(__file__).name,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "shear_h5": package_relative(args.shear_h5),
        "tomo_index_zero_based": int(args.tomo_index),
        "tomo_label_one_based": int(args.tomo_index + 1),
        "mask_name": args.mask_name,
        "noise_attribute": maps["noise_attr"],
        "noise_level": maps["noise_level"],
        "n_valid_sources": maps["n_valid_sources"],
        "area_observed_deg2_binary": maps["area_observed_deg2_binary"],
        "n_eff_per_arcmin2_binary_area": maps["n_eff_per_arcmin2_binary_area"],
        "nside": maps["nside"],
        "lmax": result["lmax"],
        "n_bandpowers": int(len(result["ell_eff"])),
        "pixel_window_correction": not args.no_pixel_window,
        "covariance_input": args.covariance_input,
        "covariance_mode": "gaussian_covariance(..., coupled=False)",
        "covariance_shape": list(result["covariance_shape"]),
        "outputs": {
            "plot_png": package_relative(out_png),
            "arrays_npz": package_relative(out_npz),
            "summary_json": package_relative(out_json),
        },
        "paper_panel": str(args.paper_panel) if paper_panel is not None else None,
        "first_five_ell_eff": result["ell_eff"][:5].tolist(),
        "first_five_ell_cl_ee_1e7": (result["ell_eff"][:5] * result["cl_signal"][0, :5] * 1.0e7).tolist(),
        "first_five_ell_error_ee_1e7": (result["ell_eff"][:5] * result["ee_error"][:5] * 1.0e7).tolist(),
        "map_change_note": "No shear-map datasets are modified by this script.",
    }
    write_json(out_json, summary)
    print(f"[{utc_now()}] Wrote arrays to {out_npz}", flush=True)
    print(f"[{utc_now()}] Wrote summary to {out_json}", flush=True)
    print(json.dumps(summary["first_five_ell_error_ee_1e7"], indent=2), flush=True)


if __name__ == "__main__":
    main()
