#!/usr/bin/env python
"""Plot production BaryonForge--GODMAX maps and common-mask C_ell products.

The plotting stage is deliberately provenance-bound: the comparison YAML and
both map files must match the hashes frozen into ``common_mask_statistics.h5``.
All 13 measured spectra are displayed, and map residuals always mean
BaryonForge minus GODMAX.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import healpy as hp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, SymLogNorm  # noqa: E402

from common import load_config, read_map_file, sha256_file  # noqa: E402


SCHEMA = "baryonforge_godmax_production_plots_v1"
STATISTICS_SCHEMA = "baryonforge_godmax_common_mask_statistics_v1"
SPECTRUM_NAMES = (
    "godmax_yy",
    "godmax_kk",
    "godmax_yk",
    "baryonforge_yy",
    "baryonforge_kk",
    "baryonforge_yk",
    "cross_backend_yy",
    "cross_backend_kk",
    "godmax_y_baryonforge_k",
    "baryonforge_y_godmax_k",
    "residual_yy",
    "residual_kk",
    "residual_yk",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _atomic_save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    overwrite: bool,
) -> list[Path]:
    outputs = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    collisions = [path for path in outputs if path.exists()]
    if collisions and not overwrite:
        raise FileExistsError(f"Refusing to overwrite plot(s): {collisions}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for output in outputs:
        temporary = output.with_name(f".{output.stem}.tmp.{os.getpid()}{output.suffix}")
        kwargs = {"dpi": 190} if output.suffix == ".png" else {}
        figure.savefig(temporary, bbox_inches="tight", **kwargs)
        os.replace(temporary, output)
        written.append(output)
    plt.close(figure)
    return written


def _load_statistics(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
]:
    with h5py.File(path, "r") as handle:
        schema = str(handle.attrs.get("schema", ""))
        if schema != STATISTICS_SCHEMA:
            raise ValueError(f"{path} has schema {schema!r}, expected {STATISTICS_SCHEMA!r}.")
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        ell = np.asarray(handle["ell"][:], dtype=np.float64)
        ell_left = np.asarray(handle["ell_left"][:], dtype=np.float64)
        ell_right = np.asarray(handle["ell_right"][:], dtype=np.float64)
        binary_pixels = np.asarray(handle["mask/binary_pixel_index"][:], dtype=np.int64)
        available = tuple(handle["spectra"].keys())
        if set(available) != set(SPECTRUM_NAMES):
            raise ValueError(
                f"Statistics spectrum set differs from the required 13-spectrum contract: {available}"
            )
        spectra = {
            name: np.asarray(handle["spectra"][name]["cl"][:], dtype=np.float64)
            for name in SPECTRUM_NAMES
        }
        diagnostics = {
            name: {
                key: np.asarray(dataset[:])
                for key, dataset in group.items()
            }
            for name, group in handle["diagnostics"].items()
        }
    if ell.ndim != 1 or ell.size == 0 or np.any(~np.isfinite(ell)) or np.any(ell <= 0.0):
        raise ValueError("Effective multipoles must be a finite, positive, non-empty vector.")
    for name, values in spectra.items():
        if values.shape != ell.shape or np.any(~np.isfinite(values)):
            raise ValueError(f"Spectrum {name} has invalid shape or non-finite values.")
    if ell_left.shape != ell.shape or ell_right.shape != ell.shape:
        raise ValueError("Bandpower edge arrays must have the same shape as the effective multipoles.")
    if np.any(ell_left > ell) or np.any(ell_right < ell):
        raise ValueError("Effective multipoles must lie inside their recorded bandpower edges.")
    if binary_pixels.ndim != 1 or binary_pixels.size == 0:
        raise ValueError("Statistics product has an invalid or empty binary-cap pixel list.")
    if np.any(np.diff(binary_pixels) <= 0):
        raise ValueError("Binary-cap pixel indices must be strictly increasing and unique.")
    return ell, ell_left, ell_right, binary_pixels, spectra, diagnostics, metadata


def _validate_inputs(
    config_path: Path,
    statistics_path: Path,
    godmax_path: Path,
    baryonforge_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    inputs = {
        "config": config_path,
        "statistics": statistics_path,
        "godmax_map": godmax_path,
        "baryonforge_map": baryonforge_path,
        "plot_driver": Path(__file__).resolve(),
    }
    frozen = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for name, path in inputs.items()
    }
    expected = {
        "config": str(metadata["config_sha256"]),
        "godmax_map": str(metadata["godmax_map_sha256"]),
        "baryonforge_map": str(metadata["baryonforge_map_sha256"]),
    }
    mismatches = {
        name: {"actual": frozen[name]["sha256"], "expected": digest}
        for name, digest in expected.items()
        if frozen[name]["sha256"] != digest
    }
    if mismatches:
        raise RuntimeError(f"Plot inputs do not match the statistics provenance: {mismatches}")
    return frozen


def _set_signed_scale(axis: plt.Axes, arrays: Sequence[np.ndarray]) -> None:
    finite = np.concatenate([np.asarray(array)[np.isfinite(array)] for array in arrays])
    if finite.size and np.all(finite > 0.0):
        axis.set_yscale("log")
        return
    absolute = np.abs(finite)
    nonzero = absolute[absolute > 0.0]
    if nonzero.size:
        axis.set_yscale("symlog", linthresh=max(float(np.min(nonzero)) * 0.5, np.finfo(float).tiny))


def plot_all_cls(
    ell: np.ndarray,
    ell_left: np.ndarray,
    ell_right: np.ndarray,
    spectra: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    overwrite: bool,
) -> list[Path]:
    groups = (
        (
            r"$C_\ell^{yy}$",
            (
                ("godmax_yy", "GODMAX auto", "o"),
                ("baryonforge_yy", "BaryonForge auto", "s"),
                ("cross_backend_yy", "GODMAX x BaryonForge", "D"),
                ("residual_yy", "residual auto", "x"),
            ),
        ),
        (
            r"$C_\ell^{\kappa\kappa}$",
            (
                ("godmax_kk", "GODMAX auto", "o"),
                ("baryonforge_kk", "BaryonForge auto", "s"),
                ("cross_backend_kk", "GODMAX x BaryonForge", "D"),
                ("residual_kk", "residual auto", "x"),
            ),
        ),
        (
            r"$C_\ell^{y\kappa}$",
            (
                ("godmax_yk", "GODMAX y x kappa", "o"),
                ("baryonforge_yk", "BaryonForge y x kappa", "s"),
                ("godmax_y_baryonforge_k", "GODMAX y x BF kappa", "^"),
                ("baryonforge_y_godmax_k", "BF y x GODMAX kappa", "v"),
                ("residual_yk", "residual y x kappa", "x"),
            ),
        ),
    )
    colors = ("#234f81", "#e1812c", "#3a923a", "#8b5fbf", "#c03d3e")
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.1))
    for axis, (ylabel, specs) in zip(axes, groups):
        arrays = []
        for color, (name, label, marker) in zip(colors, specs):
            values = np.asarray(spectra[name])
            arrays.append(values)
            axis.errorbar(
                ell,
                values,
                xerr=np.vstack((ell - ell_left, ell_right - ell)),
                color=color,
                marker=marker,
                ms=4.2,
                lw=1.5,
                capsize=0,
                label=label,
            )
        axis.set_xscale("log")
        _set_signed_scale(axis, arrays)
        axis.grid(alpha=0.22, which="both")
        axis.set_xlabel(r"effective multipole $\ell$")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
    figure.suptitle(
        "All 13 common-mask C_ell bandpowers; horizontal bars show bins, no covariance assigned\n"
        "residual = BaryonForge - GODMAX; HEALPix pixel window is not deconvolved"
    )
    figure.tight_layout()
    return _atomic_save_figure(figure, output_dir, "01_all_cls", overwrite=overwrite)


def _valid_values(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    mask = np.isfinite(values)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    return np.where(mask, values, np.nan)


def plot_cl_diagnostics(
    ell: np.ndarray,
    spectra: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
    *,
    overwrite: bool,
) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.6), sharex=True)
    colors = {"y": "#8c2981", "kappa": "#287c8e"}
    labels = {"y": "y", "kappa": r"$\kappa_{\rm CMB}$"}
    for field in ("y", "kappa"):
        group = diagnostics[field]
        valid = group["valid_amplitude_and_coherence"]
        amplitude = _valid_values(group["amplitude_sqrt_auto_ratio"], valid)
        coherence = _valid_values(group["coherence"], valid)
        residual = _valid_values(
            group["residual_fraction_of_godmax_auto"],
            group["valid_gain_and_residual"],
        )
        axes[0, 0].plot(
            ell,
            100.0 * (amplitude - 1.0),
            marker="o",
            color=colors[field],
            label=labels[field],
        )
        axes[0, 1].plot(
            ell,
            np.maximum(1.0 - coherence, np.finfo(float).tiny),
            marker="o",
            color=colors[field],
            label=labels[field],
        )
        axes[1, 0].plot(ell, residual, marker="o", color=colors[field], label=labels[field])

    yk_valid = diagnostics["yk"]["valid"]
    yk_ratio = _valid_values(diagnostics["yk"]["baryonforge_over_godmax"], yk_valid)
    gm_yk = np.asarray(spectra["godmax_yk"])
    cross_specs = (
        ("BaryonForge y x kappa", yk_ratio, "#e1812c", "s"),
        (
            "GODMAX y x BF kappa",
            np.divide(
                spectra["godmax_y_baryonforge_k"],
                gm_yk,
                out=np.full_like(gm_yk, np.nan),
                where=gm_yk != 0.0,
            ),
            "#3a923a",
            "^",
        ),
        (
            "BF y x GODMAX kappa",
            np.divide(
                spectra["baryonforge_y_godmax_k"],
                gm_yk,
                out=np.full_like(gm_yk, np.nan),
                where=gm_yk != 0.0,
            ),
            "#8b5fbf",
            "v",
        ),
    )
    for label, ratio, color, marker in cross_specs:
        axes[1, 1].plot(ell, 100.0 * (ratio - 1.0), color=color, marker=marker, label=label)

    axes[0, 0].axhline(0.0, color="0.25", lw=0.9)
    axes[0, 0].set_ylabel(r"$100[\sqrt{C_\ell^{\rm BF}/C_\ell^{\rm GM}}-1]$ [%]")
    axes[0, 0].set_title("Auto-spectrum amplitude difference")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel(r"$1-r_\ell^{\rm BF,GM}$")
    axes[0, 1].set_title("Backend decorrelation")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel(r"$C_\ell^{\rm residual}/C_\ell^{\rm GM}$")
    axes[1, 0].set_title("Residual auto-power fraction")
    axes[1, 1].axhline(0.0, color="0.25", lw=0.9)
    axes[1, 1].set_ylabel("cross-spectrum difference [%]")
    axes[1, 1].set_title(r"Relative to GODMAX $C_\ell^{y\kappa}$")
    for axis in axes.flat:
        axis.set_xscale("log")
        axis.grid(alpha=0.22, which="both")
        axis.legend(fontsize=8)
    for axis in axes[1, :]:
        axis.set_xlabel(r"effective multipole $\ell$")
    figure.suptitle("BaryonForge--GODMAX bandpower diagnostics")
    figure.tight_layout()
    return _atomic_save_figure(figure, output_dir, "02_cl_diagnostics", overwrite=overwrite)


def _project_map(
    values: np.ndarray,
    *,
    nside: int,
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
    xsize: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    margin_deg = max(0.25, min(1.0, 0.04 * float(radius_deg)))
    # Gnomonic coordinates are tan(theta), so the required plane width is not
    # simply twice the cap radius (a material distinction for the 2400-deg2 cap).
    plane_half_width_rad = math.tan(math.radians(float(radius_deg) + margin_deg))
    reso_arcmin = 2.0 * math.degrees(plane_half_width_rad) * 60.0 / int(xsize)
    projector = hp.projector.GnomonicProj(
        rot=(float(center_ra_deg), float(center_dec_deg), 0.0),
        xsize=int(xsize),
        ysize=int(xsize),
        reso=reso_arcmin,
    )
    projected = np.asarray(
        projector.projmap(
            np.asarray(values),
            vec2pix_func=lambda x, y, z: hp.vec2pix(int(nside), x, y, z),
        ),
        dtype=np.float64,
    )
    extent_rad = projector.get_extent()
    extent_deg = tuple(float(math.degrees(value)) for value in extent_rad)
    return projected, extent_deg


def _positive_limits(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    positive = np.concatenate((left[valid & (left > 0.0)], right[valid & (right > 0.0)]))
    if positive.size == 0:
        raise ValueError("Cannot make logarithmic map panels because no positive cap pixels exist.")
    vmin = float(np.percentile(positive, 1.0))
    vmax = float(np.percentile(positive, 99.85))
    vmax = max(vmax, float(np.max(positive)) * 1.0e-4)
    if not (math.isfinite(vmin) and math.isfinite(vmax) and 0.0 < vmin < vmax):
        vmin = max(float(np.min(positive)), np.finfo(float).tiny)
        vmax = float(np.max(positive))
    if not vmin < vmax:
        vmax = np.nextafter(vmin, math.inf)
    return vmin, vmax


def plot_map_triplet(
    config: Mapping[str, Any],
    godmax: np.ndarray,
    baryonforge: np.ndarray,
    *,
    field_label: str,
    title: str,
    stem: str,
    output_dir: Path,
    overwrite: bool,
    xsize: int,
    binary_pixels: np.ndarray,
) -> list[Path]:
    sky = config["sky_patch"]
    nside = hp.npix2nside(np.asarray(godmax).size)
    if np.asarray(baryonforge).size != hp.nside2npix(nside):
        raise ValueError("GODMAX and BaryonForge map lengths differ.")
    projection_kwargs = {
        "nside": nside,
        "center_ra_deg": float(sky["center_ra_deg"]),
        "center_dec_deg": float(sky["center_dec_deg"]),
        "radius_deg": float(sky["radius_deg"]),
        "xsize": int(xsize),
    }
    gm_projected, extent = _project_map(godmax, **projection_kwargs)
    bf_projected, _ = _project_map(baryonforge, **projection_kwargs)
    binary = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    binary_pixels = np.asarray(binary_pixels, dtype=np.int64)
    if np.any(binary_pixels < 0) or np.any(binary_pixels >= binary.size):
        raise ValueError("Statistics binary-mask pixel indices fall outside the map geometry.")
    binary[binary_pixels] = 1.0
    mask_projected, _ = _project_map(binary, **projection_kwargs)
    valid = np.isfinite(mask_projected) & (mask_projected > 0.5)
    valid &= np.isfinite(gm_projected) & np.isfinite(bf_projected)
    if not np.any(valid):
        raise ValueError("Projected cap contains no valid pixels.")
    residual = bf_projected - gm_projected
    vmin, vmax = _positive_limits(gm_projected, bf_projected, valid)
    residual_values = np.abs(residual[valid])
    residual_scale = float(np.percentile(residual_values, 99.85))
    residual_scale = max(residual_scale, float(np.max(residual_values)) * 1.0e-4)
    residual_scale = max(residual_scale, np.finfo(float).tiny)
    nonzero = residual_values[residual_values > 0.0]
    linthresh = max(
        residual_scale * 1.0e-3,
        float(np.percentile(nonzero, 2.0)) if nonzero.size else np.finfo(float).tiny,
    )

    figure, axes = plt.subplots(1, 3, figsize=(17.4, 5.25))
    for axis, image, panel_title in zip(
        axes[:2],
        (gm_projected, bf_projected),
        ("GODMAX", "BaryonForge"),
    ):
        plotted = axis.imshow(
            np.where(valid & (image > 0.0), image, np.nan),
            origin="lower",
            extent=extent,
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation="nearest",
        )
        axis.set_title(panel_title)
        figure.colorbar(plotted, ax=axis, fraction=0.042, pad=0.025, label=field_label)
    plotted = axes[2].imshow(
        np.where(valid, residual, np.nan),
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=SymLogNorm(
            linthresh=linthresh,
            vmin=-residual_scale,
            vmax=residual_scale,
        ),
        interpolation="nearest",
    )
    axes[2].set_title("BaryonForge - GODMAX")
    figure.colorbar(
        plotted,
        ax=axes[2],
        fraction=0.042,
        pad=0.025,
        label=f"residual {field_label}",
    )
    for index, axis in enumerate(axes):
        axis.set_xlabel("gnomonic x [deg]")
        if index == 0:
            axis.set_ylabel("gnomonic y [deg]")
    figure.suptitle(
        f"{title}: common {float(sky['area_deg2']):g} deg2 cap, raw unsmoothed halo-only maps\n"
        "inner binary mask shown; spectra instead use weighted-mean subtraction and C2 apodization"
    )
    figure.text(
        0.5,
        0.018,
        "provisional mass proxy; deterministic noiseless maps",
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.91))
    return _atomic_save_figure(figure, output_dir, stem, overwrite=overwrite)


def _write_manifest(
    path: Path,
    *,
    overwrite: bool,
    config: Mapping[str, Any],
    statistics_metadata: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
    figures: Sequence[Path],
) -> dict[str, Any]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite manifest {path}.")
    manifest = {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "residual_definition": "BaryonForge minus GODMAX",
        "spectrum_quantity": "C_ell bandpowers",
        "spectrum_names": list(SPECTRUM_NAMES),
        "nside": int(statistics_metadata["nside"]),
        "sky_patch": _jsonable(config["sky_patch"]),
        "input_products": _jsonable(inputs),
        "figures": [
            {
                "path": str(figure),
                "sha256": sha256_file(figure),
                "size_bytes": int(figure.stat().st_size),
            }
            for figure in figures
        ],
        "plot_notes": [
            "All 13 spectra stored by measure_statistics.py appear in 01_all_cls.",
            "Map panels use identical logarithmic color limits for the two backends.",
            "Residual map panels use a symmetric logarithmic norm and the BF-minus-GODMAX convention.",
            "Only the inner cap is displayed; catalog edge-buffer pixels are excluded from the visualization.",
            "No covariance is attached because both maps are deterministic transforms of one halo catalog.",
        ],
    }
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    statistics_path = Path(args.statistics).expanduser().resolve()
    godmax_path = Path(args.godmax_maps).expanduser().resolve()
    baryonforge_path = Path(args.baryonforge_maps).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config = load_config(config_path)
    (
        ell,
        ell_left,
        ell_right,
        binary_pixels,
        spectra,
        diagnostics,
        statistics_metadata,
    ) = _load_statistics(statistics_path)
    inputs = _validate_inputs(
        config_path,
        statistics_path,
        godmax_path,
        baryonforge_path,
        statistics_metadata,
    )

    godmax_maps, godmax_attrs = read_map_file(godmax_path)
    baryonforge_maps, baryonforge_attrs = read_map_file(baryonforge_path)
    for backend, attrs in (("GODMAX", godmax_attrs), ("BaryonForge", baryonforge_attrs)):
        if str(attrs.get("ordering", "")).upper() != "RING":
            raise ValueError(f"{backend} map ordering must be RING.")
        if int(attrs.get("nside", -1)) != int(statistics_metadata["nside"]):
            raise ValueError(f"{backend} map NSIDE differs from the statistics product.")

    expected_outputs = [
        output_dir / f"{stem}.{extension}"
        for stem in ("01_all_cls", "02_cl_diagnostics", "03_tsz_maps", "04_cmb_lensing_maps")
        for extension in ("png", "pdf")
    ] + [output_dir / "plot_manifest.json"]
    collisions = [path for path in expected_outputs if path.exists()]
    if collisions and not bool(args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing plot products: {collisions}")

    figure_paths: list[Path] = []
    figure_paths.extend(
        plot_all_cls(
            ell,
            ell_left,
            ell_right,
            spectra,
            output_dir,
            overwrite=bool(args.overwrite),
        )
    )
    figure_paths.extend(
        plot_cl_diagnostics(
            ell,
            spectra,
            diagnostics,
            output_dir,
            overwrite=bool(args.overwrite),
        )
    )
    figure_paths.extend(
        plot_map_triplet(
            config,
            godmax_maps["map_ymap"],
            baryonforge_maps["map_ymap"],
            field_label="dimensionless Compton-y",
            title="tSZ Compton-y",
            stem="03_tsz_maps",
            output_dir=output_dir,
            overwrite=bool(args.overwrite),
            xsize=int(args.map_xsize),
            binary_pixels=binary_pixels,
        )
    )
    figure_paths.extend(
        plot_map_triplet(
            config,
            godmax_maps["map_kappa_cmb"],
            baryonforge_maps["map_kappa_cmb"],
            field_label=r"dimensionless $\kappa_{\rm CMB}$",
            title="CMB lensing convergence",
            stem="04_cmb_lensing_maps",
            output_dir=output_dir,
            overwrite=bool(args.overwrite),
            xsize=int(args.map_xsize),
            binary_pixels=binary_pixels,
        )
    )

    # Fail if an input changed during plotting, before publishing the manifest.
    for name in ("config", "statistics", "godmax_map", "baryonforge_map", "plot_driver"):
        current = sha256_file(inputs[name]["path"])
        if current != inputs[name]["sha256"]:
            raise RuntimeError(f"Input {name} changed during plotting.")
    manifest_path = output_dir / "plot_manifest.json"
    manifest = _write_manifest(
        manifest_path,
        overwrite=bool(args.overwrite),
        config=config,
        statistics_metadata=statistics_metadata,
        inputs=inputs,
        figures=figure_paths,
    )
    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "n_spectra": len(SPECTRUM_NAMES),
        "n_figures": len(figure_paths),
        "figure_sha256": {Path(item["path"]).name: item["sha256"] for item in manifest["figures"]},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Matched comparison YAML.")
    parser.add_argument("--statistics", required=True, help="Common-mask statistics HDF5.")
    parser.add_argument("--godmax-maps", required=True, help="GODMAX native map HDF5.")
    parser.add_argument("--baryonforge-maps", required=True, help="BaryonForge native map HDF5.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG/PDF figures and manifest.")
    parser.add_argument("--map-xsize", type=int, default=800, help="Square gnomonic map size in pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Atomically replace existing plot products.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.map_xsize) < 128:
        raise ValueError("--map-xsize must be at least 128.")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
