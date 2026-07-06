#!/usr/bin/env python
"""Summarize Abacus paste benchmark JSONs and extrapolate full-run timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _load_pixel_rows(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            rows.append({"path": str(path), **row})
    return pd.DataFrame(rows)


def _load_gpu_rows(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            flat = {
                "path": str(path),
                "nside": payload.get("nside"),
                "n_halos": payload.get("n_halos"),
                "n_pairs": payload.get("n_pairs"),
                "pixel_time_s": payload.get("pixel_time_s"),
                "fused": row.get("fused"),
                "runtime_s": row.get("runtime_s"),
            }
            timing = row.get("timing_results", {}) or {}
            flat["include_galaxies"] = "galaxy_population" in timing
            for key, value in timing.items():
                flat[f"profile_{key}_s"] = value
            rows.append(flat)
    return pd.DataFrame(rows)


def _best_pixel_model(pixel: pd.DataFrame, nside: int) -> dict:
    sub = pixel[pixel["nside"] == int(nside)].copy()
    if sub.empty:
        return {}
    sub = sub.sort_values("halos_per_s", ascending=False)
    row = sub.iloc[0].to_dict()
    return {
        "source": row.get("path"),
        "halos_per_s": float(row["halos_per_s"]),
        "pairs_per_halo": float(row["pairs_per_halo"]),
        "workers": int(row.get("workers", -1)),
        "single_pixel_angle_factor": float(row.get("single_pixel_angle_factor", np.nan)),
    }


def _gpu_pair_model(gpu: pd.DataFrame, nside: int) -> dict:
    sub = gpu[(gpu["nside"] == int(nside)) & (gpu["fused"] == True)].copy()  # noqa: E712
    if sub.empty:
        sub = gpu[gpu["nside"] == int(nside)].copy()
    if "include_galaxies" in sub.columns and bool(sub["include_galaxies"].any()):
        sub = sub[sub["include_galaxies"] == True].copy()  # noqa: E712
    if sub.empty:
        return {}
    sub = sub.dropna(subset=["n_pairs", "runtime_s"])
    if sub.empty:
        return {}
    pairs = sub["n_pairs"].to_numpy(dtype=float)
    runtime = sub["runtime_s"].to_numpy(dtype=float)
    if len(sub) >= 2 and np.ptp(pairs) > 0:
        slope, intercept = np.polyfit(pairs, runtime, deg=1)
    else:
        slope = float(runtime[-1] / max(pairs[-1], 1.0))
        intercept = 0.0
    return {
        "source_rows": int(len(sub)),
        "pair_slope_s_per_pair": float(max(slope, 0.0)),
        "intercept_s": float(max(intercept, 0.0)),
        "pairs_per_s_per_gpu": float(1.0 / max(slope, 1.0e-12)),
        "max_sample_pairs": int(np.max(pairs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/xDESI/processed/abacus_backlight")
    parser.add_argument("--run-glob", default="stage31_pz*")
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--n-halos", type=float, required=True, help="Target total halo count for extrapolation.")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    pixel_paths = sorted(root.glob(f"{args.run_glob}/measurements/pixel_work*.json"))
    gpu_paths = sorted(root.glob(f"{args.run_glob}/measurements/gpu_chunk*.json"))
    pixel = _load_pixel_rows(pixel_paths)
    gpu = _load_gpu_rows(gpu_paths)
    pixel_model = _best_pixel_model(pixel, args.nside) if not pixel.empty else {}
    gpu_model = _gpu_pair_model(gpu, args.nside) if not gpu.empty else {}

    total_halos = float(args.n_halos)
    pairs_per_halo = pixel_model.get("pairs_per_halo", np.nan)
    total_pairs = total_halos * pairs_per_halo if np.isfinite(pairs_per_halo) else np.nan
    pixel_time_s = total_halos / (pixel_model["halos_per_s"] * max(1, args.num_gpus)) if pixel_model else np.nan
    if gpu_model and np.isfinite(total_pairs):
        gpu_time_s = (gpu_model["intercept_s"] * args.num_gpus + gpu_model["pair_slope_s_per_pair"] * total_pairs) / max(1, args.num_gpus)
    else:
        gpu_time_s = np.nan
    estimate = {
        "nside": int(args.nside),
        "target_n_halos": int(total_halos),
        "num_gpus": int(args.num_gpus),
        "pixel_model": pixel_model,
        "gpu_model": gpu_model,
        "estimated_total_pairs": None if not np.isfinite(total_pairs) else int(total_pairs),
        "estimated_pixel_time_s_per_node": None if not np.isfinite(pixel_time_s) else float(pixel_time_s),
        "estimated_gpu_time_s_per_node": None if not np.isfinite(gpu_time_s) else float(gpu_time_s),
        "estimated_runtime_s_lower_bound": None
        if not (np.isfinite(pixel_time_s) and np.isfinite(gpu_time_s))
        else float(max(pixel_time_s, gpu_time_s)),
        "notes": [
            "Pixel time uses best observed per-rank benchmark row multiplied by num_gpus concurrent ranks.",
            "GPU time uses fused benchmark pair-slope and assumes perfect distribution over GPUs.",
            "This is an extrapolation; full paste-split timing JSONs are stronger evidence.",
        ],
    }
    text = json.dumps(estimate, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
