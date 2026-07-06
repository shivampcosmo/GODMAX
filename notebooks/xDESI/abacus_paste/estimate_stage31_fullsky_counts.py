"""Estimate full-sky Abacus halo counts for Stage-31 tomographic bins.

This is intentionally lighter than catalog preprocessing: it reads only halo
particle counts and interpolated comoving distance from the ASDF lightcone,
then counts halos by true-redshift bin and mass threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import asdf
import h5py
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
XDESI_DIR = THIS_DIR.parent
REPO_ROOT = XDESI_DIR.parents[1]
if str(XDESI_DIR) not in sys.path:
    sys.path.insert(0, str(XDESI_DIR))

from abacus_lightcone_catalog import (  # noqa: E402
    FIELD_ALIASES,
    _get_first,
    list_snapshot_files,
    make_chi_to_z_interpolator,
)


DEFAULT_INPUT_ROOT = Path("/mnt/ceph/users/backlight/AbacusBacklight_base_c9999_ph9999/lightcone_halos")
DEFAULT_MAP_H5 = (
    REPO_ROOT
    / "data/xDESI/processed/multiprobe_namaster_true_nz/fast1024/"
    / "xdesi_multiprobe_maps_nside1024_lmax1024_nbin10_linear.h5"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/xDESI/processed/abacus_backlight/stage31_fullsky_mass_scan/"
    / "stage31_fullsky_count_scan.json"
)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _quantile_from_density(z: np.ndarray, y: np.ndarray, fraction: float) -> float:
    seg = 0.5 * (y[:-1] + y[1:]) * np.diff(z)
    total = float(seg.sum())
    if total <= 0.0:
        return float("nan")
    cdf = np.concatenate([[0.0], np.cumsum(seg)])
    return float(np.interp(float(fraction) * total, cdf, z))


def _retained_fraction(z: np.ndarray, y: np.ndarray, z_min: float, z_max: float) -> float:
    total = _trapz(y, z)
    if total <= 0.0:
        return 0.0
    zz = np.r_[z_min, z[(z > z_min) & (z < z_max)], z_max]
    yy = np.interp(zz, z, y, left=0.0, right=0.0)
    return _trapz(yy, zz) / total


def load_desi_nz_metadata(map_h5: Path) -> dict:
    with h5py.File(map_h5, "r") as handle:
        z = np.asarray(handle["nz/desi/z_mid"][:], dtype=np.float64)
        nz = np.asarray(handle["nz/desi/nz_dndz_by_pz"][:], dtype=np.float64)
        surface = np.asarray(handle["nz/desi/nz_surface_density_per_deg2_by_pz"][:], dtype=np.float64)
        mean_z = np.asarray(handle["nz/desi/nz_mean_true_z_by_pz"][:], dtype=np.float64)
        sigma_z = np.asarray(handle["nz/desi/nz_sigma_true_z_by_pz"][:], dtype=np.float64)

    out = {}
    for idx in range(nz.shape[0]):
        y = nz[idx]
        central98 = (_quantile_from_density(z, y, 0.01), _quantile_from_density(z, y, 0.99))
        central99 = (_quantile_from_density(z, y, 0.005), _quantile_from_density(z, y, 0.995))
        out[f"pz{idx + 1}"] = {
            "surface_density_per_deg2": float(surface[idx]),
            "mean_true_z": float(mean_z[idx]),
            "sigma_true_z": float(sigma_z[idx]),
            "central98_z_min": float(central98[0]),
            "central98_z_max": float(central98[1]),
            "central98_retained_fraction": float(_retained_fraction(z, y, *central98)),
            "central99_z_min": float(central99[0]),
            "central99_z_max": float(central99[1]),
            "central99_retained_fraction": float(_retained_fraction(z, y, *central99)),
        }
    return out


def default_pz_bins(nz_meta: Mapping[str, Mapping[str, float]], retained: str) -> dict:
    if retained not in {"central98", "central99"}:
        raise ValueError("retained must be central98 or central99")
    bins = {}
    for pz in range(1, 5):
        meta = nz_meta[f"pz{pz}"]
        z_min = float(meta[f"{retained}_z_min"])
        z_max = float(meta[f"{retained}_z_max"])
        bins[f"pz{pz}"] = {
            "z_min": z_min,
            "z_max": z_max,
            "retained_fraction": float(meta[f"{retained}_retained_fraction"]),
            "retained_definition": retained,
        }

    # Preserve the already validated catalog cuts for pz1 and pz3.
    if retained == "central99":
        bins["pz1"].update({"z_min": 0.30, "z_max": 0.62})
        bins["pz1"]["retained_fraction"] = float(_retained_from_meta_cut(nz_meta, "pz1", 0.30, 0.62))
        bins["pz3"].update({"z_min": 0.63, "z_max": 0.98})
        bins["pz3"]["retained_fraction"] = float(_retained_from_meta_cut(nz_meta, "pz3", 0.63, 0.98))
    return bins


def _retained_from_meta_cut(nz_meta: Mapping[str, Mapping[str, float]], pz_key: str, z_min: float, z_max: float) -> float:
    # The exact n(z) arrays are not kept in nz_meta; reopen the default HDF5.
    with h5py.File(DEFAULT_MAP_H5, "r") as handle:
        z = np.asarray(handle["nz/desi/z_mid"][:], dtype=np.float64)
        pz = int(pz_key.replace("pz", "")) - 1
        y = np.asarray(handle["nz/desi/nz_dndz_by_pz"][pz, :], dtype=np.float64)
    return _retained_fraction(z, y, z_min, z_max)


def parse_thresholds(text: str) -> np.ndarray:
    values = [float(item) for item in text.replace(" ", "").split(",") if item]
    if not values:
        raise ValueError("No thresholds supplied.")
    return np.asarray(sorted(set(values)), dtype=np.float64)


def scan_counts(
    input_root: Path,
    pz_bins: Mapping[str, Mapping[str, float]],
    thresholds: Sequence[float],
    redshift_dir_padding: float,
    max_files: int | None = None,
) -> dict:
    thresholds = np.asarray(thresholds, dtype=np.float64)
    mass_thresholds = np.power(10.0, thresholds)
    max_z = max(float(item["z_max"]) for item in pz_bins.values()) + float(redshift_dir_padding)
    files = list_snapshot_files(input_root, max_z)
    if max_files is not None:
        files = files[: int(max_files)]
    counts = {key: {f"logMgt{thr:.3f}": 0 for thr in thresholds} for key in pz_bins}
    files_out = []
    t0 = time.perf_counter()
    for file_index, (z_dir, path) in enumerate(files):
        file_t0 = time.perf_counter()
        with asdf.open(path, lazy_load=True) as af:
            header = af["header"]
            halo_lc = af["halo_lightcone"]
            n_arr, _ = _get_first(halo_lc, FIELD_ALIASES["n_interp"])
            chi_arr, _ = _get_first(halo_lc, FIELD_ALIASES["chi"])
            n_interp = np.asarray(n_arr[:], dtype=np.float32)
            mass = n_interp.astype(np.float64) * float(header["ParticleMassHMsun"])
            keep = mass > float(mass_thresholds.min())
            n_prefilter = int(np.count_nonzero(keep))
            if n_prefilter:
                chi_to_z = make_chi_to_z_interpolator(header, max_z + 0.2)
                z = chi_to_z(np.asarray(chi_arr[keep], dtype=np.float32))
                mass_keep = mass[keep]
                for pz_key, bin_cfg in pz_bins.items():
                    z_mask = (z >= float(bin_cfg["z_min"])) & (z < float(bin_cfg["z_max"])) & np.isfinite(z)
                    if not np.any(z_mask):
                        continue
                    mass_z = mass_keep[z_mask]
                    for thr, mass_thr in zip(thresholds, mass_thresholds):
                        counts[pz_key][f"logMgt{thr:.3f}"] += int(np.count_nonzero(mass_z > mass_thr))
        files_out.append(
            {
                "file_index": int(file_index),
                "z_dir": float(z_dir),
                "path": str(path),
                "n_prefilter_min_threshold": n_prefilter,
                "elapsed_s": float(time.perf_counter() - file_t0),
            }
        )
        print(
            f"[count-scan] {file_index + 1}/{len(files)} zdir={z_dir:.3f} "
            f"prefilter={n_prefilter:,} elapsed={files_out[-1]['elapsed_s']:.1f}s",
            flush=True,
        )
    return {
        "input_root": str(input_root),
        "redshift_dir_padding": float(redshift_dir_padding),
        "thresholds_log10_hmsun": [float(x) for x in thresholds],
        "pz_bins": {key: dict(value) for key, value in pz_bins.items()},
        "counts": counts,
        "files": files_out,
        "total_elapsed_s": float(time.perf_counter() - t0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--map-h5", type=Path, default=DEFAULT_MAP_H5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thresholds", default="12.5,12.75,13.0,13.1,13.2,13.3,13.4,13.5,13.6,13.7,13.8,14.0")
    parser.add_argument("--retained", choices=("central98", "central99"), default="central99")
    parser.add_argument("--redshift-dir-padding", type=float, default=0.08)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    nz_meta = load_desi_nz_metadata(args.map_h5)
    pz_bins = default_pz_bins(nz_meta, args.retained)
    result = scan_counts(
        args.input_root.expanduser().resolve(),
        pz_bins,
        parse_thresholds(args.thresholds),
        redshift_dir_padding=float(args.redshift_dir_padding),
        max_files=args.max_files,
    )
    result["desi_nz_metadata"] = nz_meta
    result["map_h5"] = str(args.map_h5.expanduser().resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[count-scan] wrote {args.output}")


if __name__ == "__main__":
    main()
