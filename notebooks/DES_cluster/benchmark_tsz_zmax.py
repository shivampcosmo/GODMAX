#!/usr/bin/env python
"""Deterministic CPU scaling sample for the z-limited NSIDE-2048 tSZ run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-des-cluster")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tsz_pasting as tp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default=str(HERE / "params_tsz_zmax0p85.yaml"))
    parser.add_argument("--sample-halos", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--work-dir", default="/tmp/des_cluster_zmax_benchmark_repro")
    args = parser.parse_args()

    cfg = tp.load_params(args.params, {"runtime": {"jax_platforms": "cpu"}})
    mass_name, _, _, _, redshift_name = tp._field_names(cfg)
    with h5py.File(cfg["catalog"]["path"], "r") as handle:
        records = handle[cfg["catalog"]["dataset"]].fields((mass_name, redshift_name))[:]
    valid = np.flatnonzero(
        tp._selection_mask(records[mass_name], records[redshift_name], cfg)
    )
    if args.sample_halos <= 0 or args.sample_halos > len(valid):
        raise ValueError("sample-halos must be positive and no larger than the selected catalog.")
    rows = np.sort(
        np.random.default_rng(args.seed).choice(valid, size=args.sample_halos, replace=False)
    )
    overrides = {
        "runtime": {
            "jax_platforms": "cpu",
            "halo_chunk_size": 10_000,
            "pixel_batch_size": 2_000,
            "pair_batch_size": 65_536,
            "pixel_workers": 8,
            "verbose": False,
        },
        "output": {
            "directory": str(Path(args.work_dir).resolve()),
            "run_name": f"zmax0p85_bench{args.sample_halos}",
            "compression": None,
        },
    }
    result = tp.run_tsz_paste(
        args.params,
        overrides=overrides,
        row_indices=rows,
        overwrite=True,
    )
    print(
        json.dumps(
            {
                "sample_halos": args.sample_halos,
                "seed": args.seed,
                "selected_catalog_rows": int(len(valid)),
                "sample_first_row": int(rows[0]),
                "sample_last_row": int(rows[-1]),
                "output": str(result["path"]),
                **result["diagnostics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
