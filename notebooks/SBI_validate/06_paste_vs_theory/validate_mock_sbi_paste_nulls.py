#!/usr/bin/env python3
"""Compare two paste products dataset-by-dataset and require bitwise equality.

This is the null control for every change made to the paste path.  Three uses:

* **Null A (reproduction).** Paste with the gas override set to the frozen values
  and require the y/tau/kappa maps to be bitwise identical to the archived split.
  Proves the override machinery is inert when it should be.
* **Null B (galaxy skip).** Paste with ``get_galmap: false`` and require the same.
  This is what licenses skipping 66% of the chunk loop; if it fails, the campaign
  falls back to painting the galaxy map at 2.3x the cost.
* **Null C (split invariance).** Combine at a different ``num_splits`` and require
  agreement to float round-off, guarding the "chunk loop repaints the full
  catalog" failure mode, which scales map amplitudes by the chunk count.

Bitwise is the right bar for A and B: the same inputs through the same code must
produce the same float32 bits.  Anything less would hide a real perturbation.
"""

from __future__ import annotations

# --- keep imports working from a theme subfolder: common/ holds the
# --- modules shared by more than one stage.
import pathlib as _pl, sys as _sys
_ROOT = _pl.Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "common"):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import argparse
import json
import os
import pathlib
import sys

import h5py
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import mock_sbi_common as msc

REQUIRED_MAPS = ("map_ymap", "map_tau", "map_kappa_cmb")


def read_maps(path: pathlib.Path) -> tuple[dict[str, np.ndarray], dict]:
    with h5py.File(path, "r") as handle:
        maps = {key: np.asarray(handle["maps"][key]) for key in handle["maps"]}
        attrs = {k: handle.attrs[k] for k in handle.attrs}
    return maps, attrs


def compare(a_path: pathlib.Path, b_path: pathlib.Path, *, tolerance: float) -> dict:
    a_maps, a_attrs = read_maps(a_path)
    b_maps, b_attrs = read_maps(b_path)
    shared = sorted(set(a_maps) & set(b_maps))
    per_dataset = {}
    for key in shared:
        left, right = a_maps[key], b_maps[key]
        same_bits = left.dtype == right.dtype and left.shape == right.shape and \
            np.array_equal(left.view(np.uint8), right.view(np.uint8))
        diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
        scale = np.maximum(np.abs(left.astype(np.float64)), np.abs(right.astype(np.float64)))
        nonzero = scale > 0
        per_dataset[key] = {
            "bitwise_identical": bool(same_bits),
            "dtype": [str(left.dtype), str(right.dtype)],
            "max_abs_difference": float(diff.max()) if diff.size else 0.0,
            "max_relative_difference": float(np.max(diff[nonzero] / scale[nonzero])) if np.any(nonzero) else 0.0,
        }
    return {
        "a": str(a_path), "b": str(b_path),
        "a_sha256": msc.sha256_file(a_path), "b_sha256": msc.sha256_file(b_path),
        "a_map_datasets": sorted(a_maps), "b_map_datasets": sorted(b_maps),
        "only_in_a": sorted(set(a_maps) - set(b_maps)),
        "only_in_b": sorted(set(b_maps) - set(a_maps)),
        "shared_datasets": shared,
        "per_dataset": per_dataset,
        "a_n_galaxies": int(a_attrs.get("n_galaxies", -1)),
        "b_n_galaxies": int(b_attrs.get("n_galaxies", -1)),
        "tolerance": tolerance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=pathlib.Path, required=True)
    parser.add_argument("--b", type=pathlib.Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--require", choices=("bitwise", "roundoff"), default="bitwise")
    parser.add_argument("--tolerance", type=float, default=1.0e-6,
                        help="relative tolerance for --require roundoff")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    report = compare(args.a, args.b, tolerance=args.tolerance)
    report["label"] = args.label
    report["requirement"] = args.require

    missing = [key for key in REQUIRED_MAPS if key not in report["shared_datasets"]]
    if missing:
        report["status"] = "FAIL"
        report["failure"] = f"required map dataset(s) absent from one side: {missing}"
    elif args.require == "bitwise":
        bad = [k for k in REQUIRED_MAPS if not report["per_dataset"][k]["bitwise_identical"]]
        report["status"] = "PASS" if not bad else "FAIL"
        if bad:
            report["failure"] = f"not bitwise identical: {bad}"
    else:
        bad = [k for k in REQUIRED_MAPS
               if report["per_dataset"][k]["max_relative_difference"] > args.tolerance]
        report["status"] = "PASS" if not bad else "FAIL"
        if bad:
            report["failure"] = f"exceeds relative tolerance {args.tolerance}: {bad}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)

    print(f"[{args.label}] {report['status']}  requirement={args.require}")
    print(f"  a: {args.a.name}  datasets {report['a_map_datasets']}  n_galaxies {report['a_n_galaxies']}")
    print(f"  b: {args.b.name}  datasets {report['b_map_datasets']}  n_galaxies {report['b_n_galaxies']}")
    if report["only_in_a"] or report["only_in_b"]:
        print(f"  only in a: {report['only_in_a']}   only in b: {report['only_in_b']}")
    for key in REQUIRED_MAPS:
        stats = report["per_dataset"].get(key)
        if stats is None:
            print(f"  {key:16s} ABSENT")
            continue
        print(f"  {key:16s} bitwise={stats['bitwise_identical']!s:5s} "
              f"max|abs| {stats['max_abs_difference']:.3e}  "
              f"max|rel| {stats['max_relative_difference']:.3e}")
    if report["status"] != "PASS":
        print(f"  FAILURE: {report['failure']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
