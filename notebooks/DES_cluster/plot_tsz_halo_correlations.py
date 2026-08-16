"""Validated Healpy diagnostics for the DES-cluster halo-only tSZ map.

This module intentionally produces raw scalar pseudo-spectra with one explicit
common first-octant mask.  It does not apply a beam, HEALPix pixel-window
correction, shot-noise subtraction, mode-coupling correction, f_sky rescaling,
or survey bandpower window.  The outputs are diagnostics of the pasted map,
not the xDESI NaMaster data vector.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MAP_PATH = (
    REPO_ROOT
    / "data"
    / "DES_cluster"
    / "tsz_maps"
    / "c000_ph000_Mgt1e13_zle0p85_nside2048_halosall_30b918e83963.h5"
)
DEFAULT_PARAMS_PATH = HERE / "params_tsz_zmax0p85.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "DES_cluster" / "tsz_analysis"

MAP_DATASET = "maps/map_ymap"
MAP_SCHEMA = "godmax_des_cluster_tsz_map_v1"
MARKER_SCHEMA = "godmax_des_cluster_tsz_validation_v1"
ANALYSIS_SCHEMA = "godmax_des_cluster_tsz_healpy_diagnostics_v1"
EXPECTED_MAP_SHA256 = "36a158b87220abb5ea2fcf483e3d77e7dcc5c635dbf5ecbf6a9eb71abe25ca99"
EXPECTED_SELECTED_SHA256 = "3d702ceed8fb737ffcd31670792ae650fa11ddcdc0507389bac8e9627fab57b4"
EXPECTED_SELECTED_ROWS = 1_299_336

RAW_SPECTRUM_POLICY = {
    "estimator": "healpy scalar map2alm + alm2cl",
    "sky": "first-octant lightcone footprint",
    "mask": "common sharp inclusive spherical-triangle mask from healpy.query_polygon",
    "mask_mean_subtraction": "within-footprint mean removed; zero outside",
    "beam_correction": "none",
    "pixel_window_correction": "none",
    "shot_noise_subtraction": "none",
    "mode_coupling_correction": "none",
    "f_sky_correction": "none",
    "survey_bandpower_window": "none",
    "monopole": "removed from y and halo overdensity before map2alm",
}


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA256 for ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def load_validated_ymap(
    map_path: str | Path = DEFAULT_MAP_PATH,
    *,
    verify_sha256: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the validated RING Compton-y map and fail closed on provenance."""
    path = Path(map_path).resolve()
    marker_path = path.with_name(path.name + ".validated.json")
    if not path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(f"Expected map and validation marker: {path}, {marker_path}")

    marker_sha256 = sha256_file(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema") != MARKER_SCHEMA:
        raise ValueError(f"Unexpected validation-marker schema: {marker.get('schema')}")
    for key in ("complete_selected_catalog_painted", "finite", "nonnegative"):
        if not bool(marker.get(key)):
            raise ValueError(f"Validation marker does not attest {key}=true.")
    actual_sha256 = sha256_file(path) if verify_sha256 else str(marker["output_sha256"])
    if actual_sha256 != str(marker["output_sha256"]):
        raise ValueError("HDF5 SHA256 does not match the validation marker.")
    if path == DEFAULT_MAP_PATH.resolve() and actual_sha256 != EXPECTED_MAP_SHA256:
        raise ValueError(f"Production map SHA256 drift: {actual_sha256}")

    with h5py.File(path, "r") as handle:
        if list(handle.keys()) != ["maps"] or list(handle["maps"].keys()) != ["map_ymap"]:
            raise ValueError("Expected exactly maps/map_ymap in the production HDF5.")
        attrs = {str(key): _plain(value) for key, value in handle.attrs.items()}
        if attrs.get("schema") != MAP_SCHEMA:
            raise ValueError(f"Unexpected map schema: {attrs.get('schema')}")
        if attrs.get("ordering") != "RING":
            raise ValueError("Only a RING-ordered map is supported.")
        if attrs.get("map_units") != "dimensionless Compton-y":
            raise ValueError(f"Unexpected map units: {attrs.get('map_units')}")
        if not bool(attrs.get("complete_selected_catalog_painted")):
            raise ValueError("The product does not attest to painting the complete selection.")
        if attrs.get("selected_row_index_sha256") != attrs.get("painted_row_index_sha256"):
            raise ValueError("Selected and painted halo-row digests differ.")
        for key in ("selected_row_index_sha256", "painted_row_index_sha256"):
            if str(marker.get(key)) != str(attrs.get(key)):
                raise ValueError(f"Validation-marker {key} differs from the map HDF5.")
        if int(marker.get("selected_rows_available", -1)) != int(
            attrs.get("selected_rows_available", -2)
        ):
            raise ValueError("Validation-marker selected count differs from the map HDF5.")
        dataset = handle[MAP_DATASET]
        if dataset.dtype != np.dtype("float32"):
            raise ValueError(f"Expected float32 Compton-y map, got {dataset.dtype}.")
        ymap = np.asarray(dataset[:], dtype=np.float32)

    nside = hp.npix2nside(len(ymap))
    if int(attrs["nside"]) != nside:
        raise ValueError(f"nside metadata mismatch: {attrs['nside']} != {nside}.")
    if int(marker.get("npix", -1)) != len(ymap):
        raise ValueError("Validation-marker npix differs from the map HDF5.")
    if not np.all(np.isfinite(ymap)) or np.any(ymap < 0.0):
        raise ValueError("Compton-y map is nonfinite or negative.")
    if np.count_nonzero(ymap) == 0:
        raise ValueError("Validated positive-amplitude map is identically zero.")

    metadata = {
        **attrs,
        "map_path": str(path),
        "marker_path": str(marker_path),
        "marker_sha256": marker_sha256,
        "map_sha256": actual_sha256,
        "marker_schema": marker.get("schema"),
        "npix": int(len(ymap)),
        "nside": int(nside),
        "map_min": float(np.min(ymap)),
        "map_max": float(np.max(ymap)),
        "map_mean": float(np.mean(ymap, dtype=np.float64)),
        "map_sum": float(np.sum(ymap, dtype=np.float64)),
        "map_nonzero": int(np.count_nonzero(ymap)),
    }
    return ymap, metadata


def _selected_mask(mass: np.ndarray, redshift: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    selection = cfg["catalog"]["selection"]
    if selection.get("operator") != ">" or selection.get("redshift_max_operator") != "<=":
        raise ValueError("This diagnostic requires mass > threshold and inclusive z <= zmax.")
    keep = np.asarray(mass) > float(selection["mass_min_hmsun"])
    redshift_max = selection.get("redshift_max")
    if redshift_max is not None:
        keep &= np.asarray(redshift) <= float(redshift_max)
    return keep


def build_selected_halo_overdensity(
    params_path: str | Path,
    nside: int,
    *,
    ymap: np.ndarray | None = None,
    expected_map_metadata: Mapping[str, Any] | None = None,
    chunk_size: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Stream the exact pasted selection into a RING halo-overdensity map."""
    import tsz_pasting as tp

    cfg = tp.load_params(params_path)
    preflight = tp.preflight_catalog(cfg)
    catalog_path = Path(cfg["catalog"]["path"]).resolve()
    catalog_sha256 = sha256_file(catalog_path)
    fields = cfg["catalog"]["fields"]
    field_names = (
        str(fields["mass"]),
        str(fields["x"]),
        str(fields["y"]),
        str(fields["z_position"]),
        str(fields["redshift"]),
    )
    observer = np.asarray(cfg["catalog"]["observer_xyz_hmpc"], dtype=np.float64)
    npix = hp.nside2npix(int(nside))
    counts = np.zeros(npix, dtype=np.int32)
    index_hasher = hashlib.sha256()
    selected_rows = 0

    with h5py.File(catalog_path, "r") as handle:
        source = handle[str(cfg["catalog"]["dataset"])]
        for start in range(0, len(source), int(chunk_size)):
            stop = min(start + int(chunk_size), len(source))
            rows = source.fields(field_names)[start:stop]
            mass = np.asarray(rows[field_names[0]], dtype=np.float64)
            redshift = np.asarray(rows[field_names[4]], dtype=np.float64)
            keep = _selected_mask(mass, redshift, cfg)
            if not np.any(keep):
                continue
            global_rows = np.arange(start, stop, dtype=np.int64)[keep].astype("<i8", copy=False)
            index_hasher.update(global_rows.tobytes())
            xyz = np.column_stack(
                [rows[field_names[1]][keep], rows[field_names[2]][keep], rows[field_names[3]][keep]]
            ).astype(np.float64, copy=False)
            relative = xyz - observer[None, :]
            radius = np.linalg.norm(relative, axis=1)
            if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
                raise ValueError("Invalid observer-relative halo position.")
            theta = np.arccos(np.clip(relative[:, 2] / radius, -1.0, 1.0))
            phi = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
            pixels = hp.ang2pix(int(nside), theta, phi, nest=False)
            np.add.at(counts, pixels, 1)
            selected_rows += int(np.count_nonzero(keep))

    selected_sha256 = index_hasher.hexdigest()
    if selected_rows != int(preflight["selected_rows"]):
        raise ValueError(f"Halo-map count differs from preflight: {selected_rows}.")
    if selected_sha256 != str(preflight["selected_row_index_sha256"]):
        raise ValueError("Halo-map ordered-row digest differs from preflight.")
    if int(np.sum(counts, dtype=np.int64)) != selected_rows:
        raise ValueError("Halo count-map sum differs from the streamed selection.")

    if expected_map_metadata is not None:
        if selected_rows != int(expected_map_metadata["selected_rows_available"]):
            raise ValueError("Halo count differs from the pasted-map selection.")
        if selected_sha256 != str(expected_map_metadata["selected_row_index_sha256"]):
            raise ValueError("Halo ordered-row digest differs from the pasted map.")
        if str(Path(cfg["catalog"]["path"]).resolve()) != str(
            Path(expected_map_metadata["source_catalog"]).resolve()
        ):
            raise ValueError("Catalog path differs from the pasted-map provenance.")
        if catalog_sha256 != str(expected_map_metadata["catalog_sha256"]):
            raise ValueError("Catalog SHA256 differs from the pasted-map provenance.")
        if tp._configuration_hash(cfg) != str(expected_map_metadata["config_sha256"]):
            raise ValueError("Merged params differ from the pasted-map configuration hash.")

    footprint_pixels = hp.query_polygon(
        int(nside),
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        inclusive=True,
        fact=4,
        nest=False,
    )
    footprint = np.zeros(npix, dtype=bool)
    footprint[footprint_pixels] = True
    halos_outside_footprint = int(np.sum(counts[~footprint], dtype=np.int64))
    if halos_outside_footprint != 0:
        raise ValueError(f"Inclusive octant footprint misses {halos_outside_footprint} halos.")

    mean_count = selected_rows / float(np.count_nonzero(footprint))
    delta_h = np.zeros(npix, dtype=np.float32)
    delta_h[footprint] = counts[footprint].astype(np.float32) / np.float32(mean_count) - 1.0
    delta_h[footprint] -= np.float32(np.mean(delta_h[footprint], dtype=np.float64))
    delta_mean = float(np.mean(delta_h[footprint], dtype=np.float64))

    halo_weighted_mean_y = None
    all_sky_mean_y = None
    footprint_mean_y = None
    y_sum_inside_footprint = None
    y_sum_outside_footprint = None
    y_fraction_outside_footprint = None
    if ymap is not None:
        if len(ymap) != npix:
            raise ValueError("y and halo maps have different pixel counts.")
        weighted_sum = 0.0
        for start in range(0, npix, 4_000_000):
            stop = min(start + 4_000_000, npix)
            weighted_sum += float(
                np.sum(
                    np.asarray(ymap[start:stop], dtype=np.float64)
                    * np.asarray(counts[start:stop], dtype=np.float64),
                    dtype=np.float64,
                )
            )
        halo_weighted_mean_y = weighted_sum / selected_rows
        all_sky_mean_y = float(np.mean(ymap, dtype=np.float64))
        y_sum_inside_footprint = float(np.sum(ymap[footprint], dtype=np.float64))
        y_sum_outside_footprint = float(np.sum(ymap[~footprint], dtype=np.float64))
        footprint_mean_y = float(np.mean(ymap[footprint], dtype=np.float64))
        y_fraction_outside_footprint = y_sum_outside_footprint / (
            y_sum_inside_footprint + y_sum_outside_footprint
        )

    metadata = {
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_sha256,
        "catalog_dataset": str(cfg["catalog"]["dataset"]),
        "params_path": str(Path(params_path).resolve()),
        "config_sha256": tp._configuration_hash(cfg),
        "selection_predicate": (
            f"{fields['mass']} > {float(cfg['catalog']['selection']['mass_min_hmsun']):.17g} "
            f"Msun/h and {fields['redshift']} <= "
            f"{float(cfg['catalog']['selection']['redshift_max']):.12g}"
        ),
        "selected_rows": int(selected_rows),
        "selected_row_index_sha256": selected_sha256,
        "count_map_sum": int(np.sum(counts, dtype=np.int64)),
        "occupied_pixels": int(np.count_nonzero(counts)),
        "footprint_pixels": int(np.count_nonzero(footprint)),
        "footprint_fsky": float(np.mean(footprint, dtype=np.float64)),
        "footprint_definition": (
            "healpy.query_polygon([+x,+y,+z], inclusive=True, fact=4, RING)"
        ),
        "halos_outside_footprint": halos_outside_footprint,
        "mean_count_per_pixel": float(mean_count),
        "halo_overdensity_mean_within_footprint": delta_mean,
        "halo_weighted_mean_y": halo_weighted_mean_y,
        "all_sky_mean_y": all_sky_mean_y,
        "footprint_mean_y": footprint_mean_y,
        "y_sum_inside_footprint": y_sum_inside_footprint,
        "y_sum_outside_footprint": y_sum_outside_footprint,
        "y_fraction_outside_footprint": y_fraction_outside_footprint,
        "halo_y_mean_enhancement_over_footprint": (
            None
            if halo_weighted_mean_y is None or footprint_mean_y in (None, 0.0)
            else float(halo_weighted_mean_y / footprint_mean_y)
        ),
        "nside": int(nside),
        "ordering": "RING",
        "observer_xyz_hmpc": observer.tolist(),
    }
    del counts
    return delta_h, footprint, metadata


def weighted_log_bins(
    ell: np.ndarray,
    values: np.ndarray,
    *,
    ell_min: int = 2,
    ell_max: int | None = None,
    n_bins: int = 28,
) -> dict[str, np.ndarray]:
    """Bin an ell spectrum with full-sky mode-count weights ``2 ell + 1``."""
    ell = np.asarray(ell, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if ell.shape != values.shape:
        raise ValueError("ell and values must have identical shapes.")
    if ell_max is None:
        ell_max = int(np.max(ell))
    if int(ell_min) < 1 or int(ell_max) <= int(ell_min) or int(n_bins) < 2:
        raise ValueError("Invalid logarithmic bin specification.")
    edges = np.unique(
        np.rint(np.geomspace(int(ell_min), int(ell_max) + 1, int(n_bins) + 1)).astype(int)
    )
    centers: list[float] = []
    binned: list[float] = []
    left: list[int] = []
    right: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = (ell >= lo) & (ell < hi) & np.isfinite(values)
        if not np.any(use):
            continue
        weights = 2.0 * ell[use].astype(np.float64) + 1.0
        centers.append(float(np.sum(weights * ell[use]) / np.sum(weights)))
        binned.append(float(np.sum(weights * values[use]) / np.sum(weights)))
        left.append(int(lo))
        right.append(int(hi))
    return {
        "ell_eff": np.asarray(centers, dtype=np.float64),
        "value": np.asarray(binned, dtype=np.float64),
        "ell_left": np.asarray(left, dtype=np.int32),
        "ell_right_exclusive": np.asarray(right, dtype=np.int32),
    }


def compute_masked_pseudo_spectra(
    ymap: np.ndarray,
    delta_h: np.ndarray,
    footprint: np.ndarray,
    *,
    lmax: int = 4096,
    iter_count: int = 0,
    n_bins: int = 28,
) -> dict[str, Any]:
    """Compute raw first-octant yy and halo-overdensity x y pseudo-spectra."""
    if ymap.shape != delta_h.shape:
        raise ValueError("y and halo-overdensity maps must have the same shape.")
    if footprint.shape != ymap.shape or footprint.dtype != np.dtype(bool):
        raise ValueError("Footprint must be a boolean map matching the fields.")
    if not np.any(footprint) or np.all(footprint):
        raise ValueError("Expected a nonempty partial-sky footprint.")
    nside = hp.npix2nside(len(ymap))
    if int(lmax) > 3 * nside - 1:
        raise ValueError(f"lmax={lmax} exceeds the HEALPix limit for nside={nside}.")
    if not np.all(np.isfinite(delta_h)):
        raise ValueError("Halo-overdensity map is nonfinite.")

    y_centered = np.zeros_like(ymap, dtype=np.float32)
    y_centered[footprint] = np.asarray(ymap[footprint], dtype=np.float32)
    y_centered[footprint] -= np.float32(np.mean(y_centered[footprint], dtype=np.float64))
    h_centered = np.asarray(delta_h, dtype=np.float32).copy()
    if np.any(h_centered[~footprint] != 0.0):
        raise ValueError("Halo field must be exactly zero outside the common footprint.")
    h_centered[footprint] -= np.float32(np.mean(h_centered[footprint], dtype=np.float64))
    masked_y_mean_after_centering = float(np.mean(y_centered[footprint], dtype=np.float64))
    masked_halo_mean_after_centering = float(
        np.mean(h_centered[footprint], dtype=np.float64)
    )

    alm_y = hp.map2alm(
        y_centered,
        lmax=int(lmax),
        iter=int(iter_count),
        pol=False,
        use_weights=True,
        use_pixel_weights=False,
    )
    del y_centered
    alm_h = hp.map2alm(
        h_centered,
        lmax=int(lmax),
        iter=int(iter_count),
        pol=False,
        use_weights=True,
        use_pixel_weights=False,
    )
    del h_centered

    cl_yy = np.asarray(hp.alm2cl(alm_y, lmax=int(lmax)), dtype=np.float64)
    cl_hy = np.asarray(hp.alm2cl(alm_h, alm_y, lmax=int(lmax)), dtype=np.float64)
    del alm_h, alm_y
    ell = np.arange(len(cl_yy), dtype=np.int32)
    if not np.all(np.isfinite(cl_yy)) or not np.all(np.isfinite(cl_hy)):
        raise ValueError("Healpy returned nonfinite spectra.")
    auto_tolerance = 64.0 * np.finfo(np.float64).eps * max(float(np.max(cl_yy)), 1.0e-300)
    if float(np.min(cl_yy)) < -auto_tolerance:
        raise ValueError(f"yy auto spectrum is negative beyond roundoff: {np.min(cl_yy)}.")

    prefactor = ell.astype(np.float64) * (ell.astype(np.float64) + 1.0) / (2.0 * np.pi)
    dl_yy = prefactor * cl_yy
    dl_hy = prefactor * cl_hy
    binned_yy = weighted_log_bins(ell, dl_yy, ell_min=2, ell_max=lmax, n_bins=n_bins)
    binned_hy = weighted_log_bins(ell, dl_hy, ell_min=2, ell_max=lmax, n_bins=n_bins)
    diagnostic_range = (binned_hy["ell_eff"] >= 20.0) & (binned_hy["ell_eff"] <= 3000.0)
    positive_fraction = (
        float(np.mean(binned_hy["value"][diagnostic_range] > 0.0))
        if np.any(diagnostic_range)
        else None
    )

    return {
        "ell": ell,
        "cl_yy": cl_yy,
        "cl_hy": cl_hy,
        "dl_yy": dl_yy,
        "dl_hy": dl_hy,
        "binned_yy": binned_yy,
        "binned_hy": binned_hy,
        "lmax": int(lmax),
        "map2alm_iter": int(iter_count),
        "n_bins": int(n_bins),
        "hy_positive_bin_fraction_ell20_3000": positive_fraction,
        "yy_min_cl": float(np.min(cl_yy)),
        "masked_y_mean_after_centering": masked_y_mean_after_centering,
        "masked_halo_mean_after_centering": masked_halo_mean_after_centering,
        "policy": dict(RAW_SPECTRUM_POLICY),
    }


def plot_log_ymap(ymap: np.ndarray, metadata: Mapping[str, Any]) -> plt.Figure:
    """Return a Healpy Mollweide plot of log10 Compton-y."""
    positive = np.asarray(ymap[ymap > 0.0], dtype=np.float32)
    if len(positive) == 0:
        raise ValueError("Cannot log-plot an identically zero y map.")
    floor = max(float(np.percentile(positive, 0.5)), float(metadata["map_max"]) * 1.0e-8)
    log_map = np.log10(np.maximum(ymap, np.float32(floor)))
    fig = plt.figure(figsize=(12, 7))
    hp.mollview(
        log_map,
        fig=fig.number,
        nest=False,
        title=(
            "GODMAX halo-only Compton-y: "
            r"$M_{\rm interp}>10^{13}\,M_\odot/h$, $z\leq0.85$"
        ),
        unit=r"$\log_{10} y$ (display floor only)",
        cmap="inferno",
        min=float(np.log10(floor)),
        max=float(np.log10(metadata["map_max"])),
        xsize=1400,
    )
    hp.graticule(dpar=30, dmer=45, alpha=0.25)
    fig.text(
        0.5,
        0.035,
        "RING, NSIDE=2048; no beam, smoothing, diffuse gas, noise, or analytic two-halo term",
        ha="center",
        fontsize=10,
    )
    del positive, log_map
    return fig


def plot_binned_spectra(spectra: Mapping[str, Any]) -> plt.Figure:
    """Return yy and halo-overdensity x y diagnostic bandpower plots."""
    yy = spectra["binned_yy"]
    hy = spectra["binned_hy"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    yy_positive = yy["value"] > 0.0
    axes[0].loglog(yy["ell_eff"][yy_positive], yy["value"][yy_positive], "o-", ms=4)
    axes[0].set_xlabel(r"Multipole $\ell$")
    axes[0].set_ylabel(r"$\ell(\ell+1)\widetilde C_\ell^{yy}/(2\pi)$")
    axes[0].set_title(r"Raw masked $y$-$y$ pseudo-spectrum (dimensionless $y^2$)")

    axes[1].plot(hy["ell_eff"], hy["value"], "o-", ms=4)
    axes[1].set_xscale("log")
    if np.any(hy["value"] <= 0.0):
        nonzero = np.abs(hy["value"][hy["value"] != 0.0])
        linthresh = max(
            float(np.percentile(nonzero, 20)) if len(nonzero) else 1.0e-20,
            1.0e-30,
        )
        axes[1].set_yscale("symlog", linthresh=linthresh, linscale=0.7)
        axes[1].set_ylim(-1.25 * linthresh, 1.5 * float(np.max(hy["value"])))
        axes[1].axhline(0.0, color="0.4", lw=0.8)
    else:
        axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Multipole $\ell$")
    axes[1].set_ylabel(r"$\ell(\ell+1)\widetilde C_\ell^{\delta_h y}/(2\pi)$")
    axes[1].set_title(r"Raw masked pasted-halo $\times y$ (dimensionless $y$)")

    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    fig.suptitle(
        (
            "Healpy raw masked pseudo-spectra: NSIDE=2048, lmax=4096, RING, iter=0; "
            "sharp octant mask, ring weights"
        ),
        fontsize=11,
    )
    return fig


def save_analysis_product(
    output_path: str | Path,
    map_metadata: Mapping[str, Any],
    halo_metadata: Mapping[str, Any],
    spectra: Mapping[str, Any],
    *,
    helper_path: str | Path = __file__,
) -> Path:
    """Atomically save compact unbinned/binned spectra and provenance."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    try:
        with h5py.File(staging, "w") as handle:
            handle.attrs["schema"] = ANALYSIS_SCHEMA
            handle.attrs["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            handle.attrs["input_map"] = str(map_metadata["map_path"])
            handle.attrs["input_map_sha256"] = str(map_metadata["map_sha256"])
            handle.attrs["input_marker"] = str(map_metadata["marker_path"])
            handle.attrs["input_marker_sha256"] = str(map_metadata["marker_sha256"])
            handle.attrs["input_marker_schema"] = str(map_metadata["marker_schema"])
            handle.attrs["input_catalog"] = str(halo_metadata["catalog_path"])
            handle.attrs["input_catalog_sha256"] = str(halo_metadata["catalog_sha256"])
            handle.attrs["input_config_sha256"] = str(halo_metadata["config_sha256"])
            handle.attrs["helper_sha256"] = sha256_file(helper_path)
            handle.attrs["selection_predicate"] = str(halo_metadata["selection_predicate"])
            handle.attrs["selected_rows"] = int(halo_metadata["selected_rows"])
            handle.attrs["selected_row_index_sha256"] = str(
                halo_metadata["selected_row_index_sha256"]
            )
            handle.attrs["nside"] = int(map_metadata["nside"])
            handle.attrs["ordering"] = "RING"
            handle.attrs["lmax"] = int(spectra["lmax"])
            handle.attrs["map2alm_iter"] = int(spectra["map2alm_iter"])
            handle.attrs["masked_y_mean_after_centering"] = float(
                spectra["masked_y_mean_after_centering"]
            )
            handle.attrs["masked_halo_mean_after_centering"] = float(
                spectra["masked_halo_mean_after_centering"]
            )
            handle.attrs["spectrum_policy_json"] = json.dumps(spectra["policy"], sort_keys=True)
            handle.attrs["footprint_definition"] = str(halo_metadata["footprint_definition"])
            handle.attrs["footprint_pixels"] = int(halo_metadata["footprint_pixels"])
            handle.attrs["footprint_fsky"] = float(halo_metadata["footprint_fsky"])
            handle.attrs["mass_definition_is_provisional"] = True
            handle.attrs["mass_assumption"] = str(map_metadata["mass_assumption"])
            handle.attrs["halo_metadata_json"] = json.dumps(dict(halo_metadata), sort_keys=True)
            raw = handle.create_group("raw")
            raw.create_dataset("ell", data=spectra["ell"])
            raw.create_dataset("cl_yy", data=spectra["cl_yy"])
            raw.create_dataset("cl_halo_y", data=spectra["cl_hy"])
            raw.create_dataset("dl_yy", data=spectra["dl_yy"])
            raw.create_dataset("dl_halo_y", data=spectra["dl_hy"])
            binned = handle.create_group("binned")
            for name, source in (("yy", spectra["binned_yy"]), ("halo_y", spectra["binned_hy"])):
                group = binned.create_group(name)
                for key, value in source.items():
                    group.create_dataset(key, data=value)
            handle.flush()
        os.replace(staging, path)
    except Exception:
        if staging.exists():
            staging.unlink()
        raise
    return path


def run_diagnostics(
    map_path: str | Path = DEFAULT_MAP_PATH,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    lmax: int = 4096,
    iter_count: int = 0,
    n_bins: int = 28,
) -> dict[str, Any]:
    """Run the validated map, halo-map, spectra, and compact-product workflow."""
    map_path = Path(map_path).resolve()
    input_sha256_before = sha256_file(map_path)
    ymap, map_metadata = load_validated_ymap(map_path)
    delta_h, footprint, halo_metadata = build_selected_halo_overdensity(
        params_path,
        int(map_metadata["nside"]),
        ymap=ymap,
        expected_map_metadata=map_metadata,
    )
    spectra = compute_masked_pseudo_spectra(
        ymap,
        delta_h,
        footprint,
        lmax=lmax,
        iter_count=iter_count,
        n_bins=n_bins,
    )
    if float(halo_metadata["halo_weighted_mean_y"]) <= float(halo_metadata["footprint_mean_y"]):
        raise ValueError(
            "Count-weighted y at halo pixels does not exceed the first-octant footprint mean."
        )
    if spectra["hy_positive_bin_fraction_ell20_3000"] is None:
        raise ValueError("No halo-y diagnostic bins fall within 20 <= ell <= 3000.")
    if float(spectra["hy_positive_bin_fraction_ell20_3000"]) < 0.8:
        raise ValueError("Fewer than 80% of halo-y diagnostic bins are positive.")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tsz_halo_correlations_nside{map_metadata['nside']}_lmax{lmax}"
    spectra_path = save_analysis_product(
        output_dir / f"{stem}.h5", map_metadata, halo_metadata, spectra
    )
    input_sha256_after = sha256_file(map_path)
    if input_sha256_before != input_sha256_after:
        raise RuntimeError("Input HDF5 changed during the read-only analysis.")

    summary = {
        "map_path": str(map_path),
        "map_sha256_before": input_sha256_before,
        "map_sha256_after": input_sha256_after,
        "map_sha_unchanged": input_sha256_before == input_sha256_after,
        "marker_sha256": str(map_metadata["marker_sha256"]),
        "spectra_path": str(spectra_path),
        "nside": int(map_metadata["nside"]),
        "npix": int(map_metadata["npix"]),
        "lmax": int(lmax),
        "map2alm_iter": int(iter_count),
        "selected_rows": int(halo_metadata["selected_rows"]),
        "selected_row_index_sha256": str(halo_metadata["selected_row_index_sha256"]),
        "count_map_sum": int(halo_metadata["count_map_sum"]),
        "occupied_halo_pixels": int(halo_metadata["occupied_pixels"]),
        "halo_overdensity_mean_within_footprint": float(
            halo_metadata["halo_overdensity_mean_within_footprint"]
        ),
        "footprint_pixels": int(halo_metadata["footprint_pixels"]),
        "footprint_fsky": float(halo_metadata["footprint_fsky"]),
        "halos_outside_footprint": int(halo_metadata["halos_outside_footprint"]),
        "y_fraction_outside_footprint": float(halo_metadata["y_fraction_outside_footprint"]),
        "all_sky_mean_y": float(halo_metadata["all_sky_mean_y"]),
        "footprint_mean_y": float(halo_metadata["footprint_mean_y"]),
        "halo_weighted_mean_y": float(halo_metadata["halo_weighted_mean_y"]),
        "halo_y_mean_enhancement_over_footprint": float(
            halo_metadata["halo_y_mean_enhancement_over_footprint"]
        ),
        "masked_y_mean_after_centering": float(spectra["masked_y_mean_after_centering"]),
        "masked_halo_mean_after_centering": float(
            spectra["masked_halo_mean_after_centering"]
        ),
        "hy_positive_bin_fraction_ell20_3000": float(
            spectra["hy_positive_bin_fraction_ell20_3000"]
        ),
        "yy_min_cl": float(spectra["yy_min_cl"]),
        "raw_spectrum_policy": dict(RAW_SPECTRUM_POLICY),
        "provisional_mass_assumption": str(map_metadata["mass_assumption"]),
    }
    return {
        "ymap": ymap,
        "delta_h": delta_h,
        "footprint": footprint,
        "map_metadata": map_metadata,
        "halo_metadata": halo_metadata,
        "spectra": spectra,
        "summary": summary,
    }
