"""Reproduce the bounded DES-cluster tSZ validation ledger.

This script is intentionally separate from production.  It never submits a
cluster job and never paints the full catalog.  Use ``--stage all`` for the
complete bounded suite or select one stage while developing.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-des-cluster")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tsz_pasting as tp


PARAMS = HERE / "params_tsz.yaml"


def _release_jax() -> None:
    gc.collect()
    if "jax" in sys.modules:
        import jax

        jax.clear_caches()


def _runtime() -> dict:
    return {
        "halo_chunk_size": 64,
        "pixel_batch_size": 64,
        "pair_batch_size": 65536,
        "pixel_workers": 1,
        "verbose": False,
        "jax_platforms": "cpu",
    }


def preflight() -> dict:
    cfg = tp.load_params(PARAMS)
    report = tp.preflight_catalog(cfg)
    keys = (
        "source_rows",
        "selected_rows",
        "all_source_rows_pass_cut",
        "mass_min_hmsun",
        "mass_max_hmsun",
        "z_min",
        "z_max",
        "max_distance_redshift_relative_error",
    )
    return {key: report[key] for key in keys}


def reference() -> dict:
    result = tp.validate_pair_kernel_against_reference(
        PARAMS,
        overrides={
            "map": {"nside": 1024},
            "runtime": {
                "halo_chunk_size": 8,
                "pixel_batch_size": 8,
                "pair_batch_size": 64,
                "pixel_workers": 1,
                "verbose": False,
                "jax_platforms": "cpu",
            },
        },
        max_halos=8,
        alternate_pair_batch_size=97,
    )
    if not result["passed"]:
        raise AssertionError(result)
    return result


def radial_domain() -> dict:
    """Count central clamps and prove that five-R support stays below rp_max."""

    import healpy as hp

    cfg = tp.load_params(
        PARAMS,
        {"runtime": {"jax_platforms": "cpu", "pixel_workers": 1, "verbose": False}},
    )
    setup = tp.build_profile_setup(cfg)
    rp_min = float(np.asarray(setup.rp_array)[0])
    rp_max = float(np.asarray(setup.rp_array)[-1])
    native_rp_min = float(setup.native_projected_rp_min_hmpc)
    del setup
    _release_jax()

    nside = int(cfg["map"]["nside"])
    below_extended = 0
    below_native = 0
    halos = 0
    nearest_min = np.inf
    nearest_max = 0.0
    support_max = 0.0
    for chunk in tp._iter_selected_chunks(cfg, None):
        pixels = hp.ang2pix(nside, chunk["ra_deg"], chunk["dec_deg"], lonlat=True)
        pixel_ra, pixel_dec = hp.pix2ang(nside, pixels, lonlat=True)
        ra1 = np.radians(chunk["ra_deg"])
        dec1 = np.radians(chunk["dec_deg"])
        ra2 = np.radians(pixel_ra)
        dec2 = np.radians(pixel_dec)
        cosine = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
        nearest = chunk["DA_hMpc"] * np.arccos(np.clip(cosine, -1.0, 1.0))
        below_extended += int(np.count_nonzero(nearest < rp_min))
        below_native += int(np.count_nonzero(nearest < native_rp_min))
        halos += len(nearest)
        nearest_min = min(nearest_min, float(np.min(nearest)))
        nearest_max = max(nearest_max, float(np.max(nearest)))
        support_max = max(
            support_max,
            float(np.max(float(cfg["map"]["max_paint_R200c_factor"]) * chunk["R200c_hMpc"])),
        )
    result = {
        "nside": nside,
        "n_halos": int(halos),
        "projected_radius_min_hmpc": rp_min,
        "native_projected_radius_min_hmpc": native_rp_min,
        "projected_radius_max_hmpc": rp_max,
        "nearest_center_distance_min_hmpc": nearest_min,
        "nearest_center_distance_max_hmpc": nearest_max,
        "n_halos_below_extended_grid": int(below_extended),
        "n_halos_below_native_grid": int(below_native),
        "max_five_R200c_hmpc": support_max,
        "upper_grid_encloses_five_R_support": bool(support_max <= rp_max),
    }
    if (
        halos != 3_001_721
        or below_extended != 0
        or not result["upper_grid_encloses_five_R_support"]
    ):
        raise AssertionError(result)
    return result


def central_grid_convergence() -> dict:
    """Compare 64/96-node projected central grids on all native-grid misses."""

    import healpy as hp

    cfg = tp.load_params(
        PARAMS,
        {"runtime": {"jax_platforms": "cpu", "pixel_workers": 1, "verbose": False}},
    )
    default_setup = tp.build_profile_setup(cfg)
    native_rp_min = float(default_setup.native_projected_rp_min_hmpc)
    row_indices = []
    distances = []
    redshifts = []
    masses = []
    row_offset = 0
    nside = int(cfg["map"]["nside"])
    for chunk in tp._iter_selected_chunks(cfg, None):
        pixels = hp.ang2pix(nside, chunk["ra_deg"], chunk["dec_deg"], lonlat=True)
        pixel_ra, pixel_dec = hp.pix2ang(nside, pixels, lonlat=True)
        ra1 = np.radians(chunk["ra_deg"])
        dec1 = np.radians(chunk["dec_deg"])
        ra2 = np.radians(pixel_ra)
        dec2 = np.radians(pixel_dec)
        cosine = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
        nearest = chunk["DA_hMpc"] * np.arccos(np.clip(cosine, -1.0, 1.0))
        affected = np.flatnonzero(nearest < native_rp_min)
        if len(affected):
            row_indices.extend((row_offset + affected).tolist())
            distances.extend(nearest[affected].tolist())
            redshifts.extend(chunk["z"][affected].tolist())
            masses.extend(chunk["M200c_hMsun"][affected].tolist())
        row_offset += len(nearest)

    work = {
        "distances": np.asarray(distances, dtype=np.float32),
        "z": np.asarray(redshifts, dtype=np.float32),
        "logM": np.log(np.asarray(masses, dtype=np.float64)).astype(np.float32),
    }
    default_evaluator = tp.make_pair_evaluator(default_setup, 1024)
    default_y = tp.evaluate_pairs_fixed(default_evaluator, work, 1024)
    del default_evaluator, default_setup
    _release_jax()

    dense_cfg = tp.load_params(
        PARAMS,
        {
            "profiles": {
                "projected_radius_num_central_points": cfg["validation"][
                    "dense_projected_radius_num_central_points"
                ],
                "overrides": {
                    "analysis": cfg["validation"]["dense_grid_overrides"]["analysis"]
                },
            },
            "runtime": {"jax_platforms": "cpu", "pixel_workers": 1, "verbose": False},
        },
    )
    dense_setup = tp.build_profile_setup(dense_cfg)
    dense_evaluator = tp.make_pair_evaluator(dense_setup, 1024)
    dense_y = tp.evaluate_pairs_fixed(dense_evaluator, work, 1024)
    relative = np.abs(dense_y - default_y) / np.maximum(np.abs(dense_y), 1.0e-30)
    closest = int(np.argmin(work["distances"]))
    result = {
        "n_affected_halos": int(len(work["distances"])),
        "default_central_points": int(cfg["profiles"]["projected_radius_num_central_points"]),
        "dense_central_points": int(dense_cfg["profiles"]["projected_radius_num_central_points"]),
        "default_los_points": int(
            cfg["profiles"]["overrides"]["analysis"]["num_points_projected_profile"]
        ),
        "dense_los_points": int(
            dense_cfg["profiles"]["overrides"]["analysis"]["num_points_projected_profile"]
        ),
        "max_relative_difference": float(np.max(relative, initial=0.0)),
        "p50_relative_difference": float(np.median(relative)),
        "p99_relative_difference": float(np.quantile(relative, 0.99)),
        "closest_row": int(row_indices[closest]),
        "closest_distance_hmpc": float(work["distances"][closest]),
        "closest_dense_default_ratio": float(dense_y[closest] / default_y[closest]),
    }
    if len(work["distances"]) != 467 or result["max_relative_difference"] >= 0.005:
        raise AssertionError(result)
    return result


def controls(work_dir: Path) -> dict:
    base = {
        "map": {"nside": 8},
        "runtime": {**_runtime(), "halo_chunk_size": 1, "pixel_batch_size": 1, "pair_batch_size": 64},
        "output": {"directory": str(work_dir), "compression": None},
    }
    zero_halo = tp.run_tsz_paste(
        PARAMS,
        overrides={**base, "output": {**base["output"], "run_name": "zero_halo"}},
        max_halos=0,
        overwrite=True,
    )
    zero_amp = tp.run_tsz_paste(
        PARAMS,
        overrides={
            **base,
            "map": {"nside": 8, "pressure_amplitude": 0.0},
            "output": {**base["output"], "run_name": "zero_amp"},
        },
        max_halos=64,
        overwrite=True,
    )
    nulls = {}
    for label, product in (("zero_halo", zero_halo), ("zero_amplitude", zero_amp)):
        ymap, attrs = tp.load_tsz_map(product["path"])
        with h5py.File(product["path"], "r") as handle:
            structure = {key: list(handle[key].keys()) for key in handle.keys()}
        nulls[label] = {
            "nonzero": int(np.count_nonzero(ymap)),
            "painted": int(attrs["n_halos_painted"]),
            "structure": structure,
        }
        if nulls[label]["nonzero"] != 0 or structure != {"maps": ["map_ymap"]}:
            raise AssertionError(nulls[label])

    amplitude_maps = []
    for amplitude in (1.0, 2.0):
        product = tp.run_tsz_paste(
            PARAMS,
            overrides={
                **base,
                "map": {"nside": 8, "pressure_amplitude": amplitude},
                "output": {**base["output"], "run_name": f"amplitude_{amplitude:g}"},
            },
            max_halos=1,
            overwrite=True,
        )
        amplitude_maps.append(tp.load_tsz_map(product["path"])[0])
        _release_jax()
    amplitude = {
        "nonzero": int(np.count_nonzero(amplitude_maps[0])),
        "bitwise_2x": bool(np.array_equal(amplitude_maps[1], 2.0 * amplitude_maps[0])),
        "sum_ratio": float(
            amplitude_maps[1].sum(dtype=np.float64) / amplitude_maps[0].sum(dtype=np.float64)
        ),
    }
    if not amplitude["bitwise_2x"]:
        raise AssertionError(amplitude)
    return {"nulls": nulls, "amplitude": amplitude}


def _run_sample(
    work_dir: Path,
    run_name: str,
    sample_rows: np.ndarray,
    *,
    nside: int,
    extra: dict | None = None,
) -> np.ndarray:
    overrides = {
        "map": {"nside": nside},
        "runtime": _runtime(),
        "output": {"directory": str(work_dir), "compression": None, "run_name": run_name},
    }
    if extra:
        overrides = tp._deep_update(overrides, extra)
    product = tp.run_tsz_paste(
        PARAMS,
        overrides=overrides,
        row_indices=sample_rows,
        overwrite=True,
    )
    ymap = tp.load_tsz_map(product["path"])[0]
    _release_jax()
    return ymap


def convergence(work_dir: Path) -> dict:
    import healpy as hp

    cfg = tp.load_params(PARAMS)
    sample_rows = tp.stratified_row_indices(cfg, 64)
    with h5py.File(cfg["catalog"]["path"], "r") as handle:
        sample = handle[cfg["catalog"]["dataset"]].fields(("M_interp", "redshift_interp"))[
            sample_rows
        ]
    sample_domain = {
        "n": int(len(sample_rows)),
        "first": int(sample_rows[0]),
        "last": int(sample_rows[-1]),
        "mass_min_hmsun": float(np.min(sample["M_interp"])),
        "mass_max_hmsun": float(np.max(sample["M_interp"])),
        "z_min": float(np.min(sample["redshift_interp"])),
        "z_max": float(np.max(sample["redshift_interp"])),
    }

    default_1024 = _run_sample(work_dir, "grid_default", sample_rows, nside=1024)
    dense_1024 = _run_sample(
        work_dir,
        "grid_dense",
        sample_rows,
        nside=1024,
        extra={
            "profiles": {
                "overrides": cfg["validation"]["dense_grid_overrides"],
                "projected_radius_num_central_points": cfg["validation"][
                    "dense_projected_radius_num_central_points"
                ],
            }
        },
    )
    active = (default_1024 != 0.0) | (dense_1024 != 0.0)
    relative = np.abs(dense_1024[active] - default_1024[active]) / np.maximum(
        np.maximum(np.abs(dense_1024[active]), np.abs(default_1024[active])), 1.0e-20
    )
    grid = {
        "active_pixels": int(np.count_nonzero(active)),
        "sum_ratio_dense_default": float(
            dense_1024.sum(dtype=np.float64) / default_1024.sum(dtype=np.float64)
        ),
        "max_relative_active": float(np.max(relative)),
        "p50_relative_active": float(np.median(relative)),
        "p99_relative_active": float(np.quantile(relative, 0.99)),
        "max_absolute": float(np.max(np.abs(dense_1024 - default_1024))),
    }

    default_2048 = _run_sample(work_dir, "resolution_2048", sample_rows, nside=2048)
    integrated_1024 = default_1024.sum(dtype=np.float64) * hp.nside2pixarea(1024)
    integrated_2048 = default_2048.sum(dtype=np.float64) * hp.nside2pixarea(2048)
    resolution = {
        "integrated_Y_1024_sr": float(integrated_1024),
        "integrated_Y_2048_sr": float(integrated_2048),
        "ratio_2048_1024": float(integrated_2048 / integrated_1024),
        "relative_difference": float(abs(integrated_2048 - integrated_1024) / integrated_2048),
    }

    one_row = np.asarray([sample_rows[int(np.argmax(sample["M_interp"]))]], dtype=np.int64)
    support5 = _run_sample(work_dir, "support5", one_row, nside=1024)
    support6 = _run_sample(
        work_dir,
        "support6",
        one_row,
        nside=1024,
        extra={"map": {"max_paint_R200c_factor": 6.0}},
    )
    inner = support5 != 0.0
    support = {
        "row": int(one_row[0]),
        "mass_hmsun": float(np.max(sample["M_interp"])),
        "active_5R": int(np.count_nonzero(support5)),
        "active_6R": int(np.count_nonzero(support6)),
        "inner_bitwise_unchanged": bool(np.array_equal(support5[inner], support6[inner])),
        "outer_added": int(np.count_nonzero((support6 != 0.0) & ~inner)),
    }
    if not support["inner_bitwise_unchanged"] or support["outer_added"] <= 0:
        raise AssertionError(support)
    return {"sample": sample_domain, "grid": grid, "resolution": resolution, "support": support}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "reference",
            "radial",
            "central",
            "controls",
            "convergence",
            "all",
        ),
        default="preflight",
    )
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/des_cluster_tsz_validation"))
    args = parser.parse_args()
    result = {}
    if args.stage in ("preflight", "all"):
        result["preflight"] = preflight()
    if args.stage in ("reference", "all"):
        result["reference"] = reference()
        _release_jax()
    if args.stage in ("radial", "all"):
        result["radial"] = radial_domain()
    if args.stage in ("central", "all"):
        result["central"] = central_grid_convergence()
    if args.stage in ("controls", "all"):
        result["controls"] = controls(args.work_dir / "controls")
    if args.stage in ("convergence", "all"):
        result["convergence"] = convergence(args.work_dir / "convergence")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
