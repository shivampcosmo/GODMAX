"""Sample full-sky high-mass halos and benchmark pixel-neighbor construction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping

import asdf
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
XDESI_DIR = THIS_DIR.parent
REPO_ROOT = XDESI_DIR.parents[1]
for _path in (XDESI_DIR,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from abacus_lightcone_catalog import (  # noqa: E402
    FIELD_ALIASES,
    _get_first,
    list_snapshot_files,
    make_chi_to_z_interpolator,
    position_to_radec,
    r200c_hmpc,
)
from abacus_pasting_helpers import build_pixel_work_package  # noqa: E402


DEFAULT_INPUT_ROOT = Path("/mnt/ceph/users/backlight/AbacusBacklight_base_c9999_ph9999/lightcone_halos")
DEFAULT_COUNT_SCAN = (
    REPO_ROOT
    / "data/xDESI/processed/abacus_backlight/stage31_fullsky_mass_scan/"
    / "stage31_fullsky_count_scan_central99_20260608.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/xDESI/processed/abacus_backlight/stage31_fullsky_mass_scan/"
    / "stage31_fullsky_pixel_sample_benchmark_20260608.json"
)


def parse_thresholds(text: str) -> list[float]:
    return [float(item) for item in text.replace(" ", "").split(",") if item]


def sample_halos_for_threshold(
    *,
    input_root: Path,
    pz_bins: Mapping[str, Mapping[str, float]],
    threshold: float,
    counts_for_threshold: Mapping[str, int],
    sample_per_bin: int,
    redshift_dir_padding: float,
    seed: int,
) -> dict[str, dict[str, list[np.ndarray]]]:
    rng = np.random.default_rng(int(seed) + int(round(1000 * threshold)))
    max_z = max(float(item["z_max"]) for item in pz_bins.values()) + float(redshift_dir_padding)
    files = list_snapshot_files(input_root, max_z)
    samples = {
        pz_key: {
            "ra_deg": [],
            "dec_deg": [],
            "z": [],
            "M200c_hMsun": [],
            "R200c_hMpc": [],
            "DA_hMpc": [],
            "vlos_kms": [],
        }
        for pz_key in pz_bins
    }
    target_prob = {
        pz_key: min(1.0, float(sample_per_bin) / max(1.0, float(counts_for_threshold[pz_key])))
        for pz_key in pz_bins
    }
    mass_threshold = float(10.0**threshold)
    for file_index, (z_dir, path) in enumerate(files):
        t0 = time.perf_counter()
        with asdf.open(path, lazy_load=True) as af:
            header = af["header"]
            halo_lc = af["halo_lightcone"]
            n_arr, _ = _get_first(halo_lc, FIELD_ALIASES["n_interp"])
            chi_arr, _ = _get_first(halo_lc, FIELD_ALIASES["chi"])
            n_interp = np.asarray(n_arr[:], dtype=np.float32)
            mass = n_interp.astype(np.float64) * float(header["ParticleMassHMsun"])
            keep_mass = mass > mass_threshold
            if not np.any(keep_mass):
                continue
            base_idx = np.flatnonzero(keep_mass)
            chi = np.asarray(chi_arr[base_idx], dtype=np.float32)
            chi_to_z = make_chi_to_z_interpolator(header, max_z + 0.2)
            z = chi_to_z(chi)
            mass_keep = mass[base_idx]
            pos_arr, _ = _get_first(halo_lc, FIELD_ALIASES["position"])
            for pz_key, bin_cfg in pz_bins.items():
                mask = (z >= float(bin_cfg["z_min"])) & (z < float(bin_cfg["z_max"])) & np.isfinite(z)
                if not np.any(mask):
                    continue
                idx = base_idx[mask]
                if len(idx) > 0 and target_prob[pz_key] < 1.0:
                    choose = rng.random(len(idx)) < target_prob[pz_key]
                    idx = idx[choose]
                    local_z = z[mask][choose]
                    local_mass = mass_keep[mask][choose]
                    local_chi = chi[mask][choose]
                else:
                    local_z = z[mask]
                    local_mass = mass_keep[mask]
                    local_chi = chi[mask]
                if len(idx) == 0:
                    continue
                pos = np.asarray(pos_arr[idx], dtype=np.float32)
                ra, dec, _ = position_to_radec(pos)
                bucket = samples[pz_key]
                bucket["ra_deg"].append(ra)
                bucket["dec_deg"].append(dec)
                bucket["z"].append(local_z.astype(np.float32))
                bucket["M200c_hMsun"].append(local_mass.astype(np.float64))
                bucket["R200c_hMpc"].append(r200c_hmpc(local_mass, local_z, header).astype(np.float32))
                bucket["DA_hMpc"].append((local_chi / (1.0 + local_z)).astype(np.float32))
                bucket["vlos_kms"].append(np.zeros(len(idx), dtype=np.float32))
        print(
            f"[pixel-sample] threshold={threshold:.2f} file {file_index + 1}/{len(files)} "
            f"zdir={z_dir:.3f} elapsed={time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    return samples


def concatenate_sample(raw: Mapping[str, list[np.ndarray]], sample_per_bin: int, seed: int) -> dict[str, np.ndarray]:
    out = {}
    for key, arrays in raw.items():
        if arrays:
            out[key] = np.concatenate(arrays)
        else:
            out[key] = np.empty(0, dtype=np.float32)
    n = len(out["z"])
    if n > sample_per_bin:
        rng = np.random.default_rng(int(seed) + n)
        idx = rng.choice(n, size=int(sample_per_bin), replace=False)
        out = {key: value[idx] for key, value in out.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--count-scan", type=Path, default=DEFAULT_COUNT_SCAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thresholds", default="12.75,13.0")
    parser.add_argument("--sample-per-bin", type=int, default=100000)
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pixel-batch-size", type=int, default=1000000)
    parser.add_argument("--pixel-gc-collect-every-n-batches", type=int, default=0)
    parser.add_argument("--pixel-pool-chunksize", type=int, default=32)
    parser.add_argument("--single-pixel-angle-factor", type=float, default=0.5)
    parser.add_argument("--max-paint-r200c-factor", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    count_scan = json.loads(args.count_scan.read_text())
    pz_bins = count_scan["pz_bins"]
    results = {
        "count_scan": str(args.count_scan.expanduser().resolve()),
        "thresholds_log10_hmsun": parse_thresholds(args.thresholds),
        "sample_per_bin": int(args.sample_per_bin),
        "nside": int(args.nside),
        "workers": int(args.workers),
        "pixel_batch_size": int(args.pixel_batch_size),
        "pixel_gc_collect_every_n_batches": int(args.pixel_gc_collect_every_n_batches),
        "pixel_pool_chunksize": int(args.pixel_pool_chunksize),
        "single_pixel_angle_factor": float(args.single_pixel_angle_factor),
        "max_paint_R200c_factor": float(args.max_paint_r200c_factor),
        "benchmarks": {},
    }
    for threshold in results["thresholds_log10_hmsun"]:
        count_key = f"logMgt{threshold:.3f}"
        counts_for_threshold = {pz: int(count_scan["counts"][pz][count_key]) for pz in pz_bins}
        raw_samples = sample_halos_for_threshold(
            input_root=args.input_root.expanduser().resolve(),
            pz_bins=pz_bins,
            threshold=float(threshold),
            counts_for_threshold=counts_for_threshold,
            sample_per_bin=int(args.sample_per_bin),
            redshift_dir_padding=float(count_scan["redshift_dir_padding"]),
            seed=int(args.seed),
        )
        threshold_result = {}
        for pz_key in sorted(pz_bins):
            sample = concatenate_sample(raw_samples[pz_key], int(args.sample_per_bin), int(args.seed))
            t0 = time.perf_counter()
            pixels = build_pixel_work_package(
                sample,
                int(args.nside),
                float(args.max_paint_r200c_factor),
                int(args.pixel_batch_size),
                workers=int(args.workers),
                start_method="fork",
                pool_chunksize=int(args.pixel_pool_chunksize),
                single_pixel_angle_factor=float(args.single_pixel_angle_factor),
                pixel_backend="healpy",
                include_legacy_pixel_arrays=False,
                precompute_pixel_groups=True,
                pixel_gc_collect_every_n_batches=int(args.pixel_gc_collect_every_n_batches),
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            n_halos = int(len(sample["z"]))
            n_pairs = int(len(pixels["nearby_pix_all"])) if pixels is not None else 0
            threshold_result[pz_key] = {
                "fullsky_count": int(counts_for_threshold[pz_key]),
                "sample_halos": n_halos,
                "pixel_pairs": n_pairs,
                "pairs_per_halo": float(n_pairs / max(1, n_halos)),
                "single_pixel_shortcuts": int(pixels.get("n_single_pixel_shortcut", 0)) if pixels else 0,
                "query_disc": int(pixels.get("n_query_disc", 0)) if pixels else 0,
                "pixel_time_s": float(elapsed),
                "sample_halos_per_s": float(n_halos / elapsed) if elapsed > 0 else None,
            }
            print(
                f"[pixel-sample] threshold={threshold:.2f} {pz_key} "
                f"halos={n_halos:,} pairs_per_halo={threshold_result[pz_key]['pairs_per_halo']:.3f} "
                f"time={elapsed:.2f}s",
                flush=True,
            )
        results["benchmarks"][f"logMgt{threshold:.3f}"] = threshold_result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[pixel-sample] wrote {args.output}")


if __name__ == "__main__":
    main()
