#!/usr/bin/env python3
"""Measure combined pasted maps into 42-vectors through the frozen estimator.

One paste -> one deterministic 42-vector.  The estimator is the contract's: the
regenerated float64 mask, the saved hash-checked NaMaster workspace, the frozen
galaxy alm, the 14 inference bands.  Nothing here rebuilds a workspace or a mask.

Two guards that exist because the failure is silent:

* **Injectivity.** Two design points receiving one map, or one map being reused
  for two parameter points, produces a perfectly smooth emulator of the wrong
  function.  So the map sha is required to be unique per distinct parameter point
  and identical only where the parameters are identical.
* **Non-degeneracy against the reference.** Every non-anchor point must differ
  from the frozen reference paste by more than float noise; if a config's override
  silently failed to apply, the vector would come back equal to the reference and
  nothing else would complain.
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
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import yaml

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import mock_sbi_common as msc

# A genuine parameter change must move the vector by far more than float32 map
# storage noise (~1e-7 relative).  This is a "did the override apply at all" guard.
MIN_RELATIVE_DIFFERENCE_FROM_REFERENCE = 1.0e-4


def combined_map_path(config_path: pathlib.Path) -> pathlib.Path:
    with pathlib.Path(config_path).open() as handle:
        config = yaml.safe_load(handle)
    paste = config["pasting"]
    project = config["project"]
    catalog_key = config["resolved_theory"]["catalog_key"]
    root = pathlib.Path(project["output_root"]) / project["map_subdir"] / paste["run_name"]
    return root / f"abacus_pasted_maps_{catalog_key}_nside{int(paste['nside'])}.h5"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, required=True,
                        help="paste_plan.json written by mock_sbi_design.py")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--allow-missing", action="store_true",
                        help="Measure whatever is present and report the rest, instead of failing")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    output = args.output or args.plan.with_name("responses.npz")

    print(f"[1/3] loading the frozen estimator ...", flush=True)
    ctx = msc.load_estimator_context()
    reference = np.load(msc.REPO_ROOT / "data/SBI_validate/mock_sbi/reference_paste_vector.npz")
    mu_reference = np.asarray(reference["mu_paste_reference"], dtype=np.float64)

    print(f"[2/3] measuring {plan['count']} pasted points ...", flush=True)
    records, vectors, missing = [], [], []
    by_map_sha = defaultdict(list)
    for entry in plan["entries"]:
        config_path = msc.REPO_ROOT / entry["config_path"] if not pathlib.Path(entry["config_path"]).is_absolute() \
            else pathlib.Path(entry["config_path"])
        cached = entry.get("cached_map")
        if cached:
            map_path = pathlib.Path(cached)
            if not map_path.is_absolute():
                map_path = msc.REPO_ROOT / map_path
            if msc.sha256_file(map_path) != entry["cached_map_sha256"]:
                raise RuntimeError(f"Cached map {map_path} changed since the plan was written")
        else:
            map_path = combined_map_path(config_path)
        if not map_path.is_file():
            missing.append({"run_name": entry["run_name"], "expected_map": str(map_path)})
            continue
        if not cached and msc.sha256_file(config_path) != entry["config_sha256"]:
            raise RuntimeError(f"{config_path} changed since the plan was written")
        t0 = time.time()
        vector = msc.measure_paste_file(map_path, ctx)
        elapsed = time.time() - t0
        map_sha = msc.sha256_file(map_path)
        by_map_sha[map_sha].append(entry["theta_sha256"])
        relative = float(np.max(np.abs(vector - mu_reference)
                               / np.maximum(np.abs(mu_reference), 1e-300)))
        record = {
            "index": entry["index"], "run_name": entry["run_name"],
            "theta": entry["theta"], "theta_sha256": entry["theta_sha256"],
            "component": entry["component"], "log_q": entry["log_q"],
            "sampling_role": entry["sampling_role"],
            "importance_eligible": entry["importance_eligible"],
            "map_path": str(map_path), "map_sha256": map_sha,
            "vector_sha256": msc.sha256_array(vector),
            "max_relative_difference_from_reference": relative,
            "chi2_against_observation": None,
            "measure_seconds": elapsed,
        }
        records.append(record)
        vectors.append(vector)
        print(f"      {entry['index']:4d} {entry['run_name']}  reldiff-vs-ref {relative:.3e}  "
              f"({elapsed:.1f}s)", flush=True)

    if missing and not args.allow_missing:
        raise SystemExit(f"{len(missing)} paste(s) are missing their combined map; "
                         f"first: {missing[0]}")

    if not records:
        raise SystemExit("No pasted maps found to measure")
    vectors = np.asarray(vectors, dtype=np.float64)
    observation, _ = msc.load_inference_observation()
    for record, vector in zip(records, vectors):
        record["chi2_against_observation"] = ctx.chi2(observation - vector)

    print("[3/3] guards", flush=True)
    collisions = {sha: sorted(set(v)) for sha, v in by_map_sha.items() if len(set(v)) > 1}
    if collisions:
        raise RuntimeError(f"One map is shared by different parameter points: {collisions}")
    duplicate_vectors = {}
    seen = {}
    for record in records:
        seen.setdefault(record["vector_sha256"], []).append(record["theta_sha256"])
    for sha, thetas in seen.items():
        if len(set(thetas)) > 1:
            duplicate_vectors[sha] = sorted(set(thetas))
    if duplicate_vectors:
        raise RuntimeError(f"Distinct parameter points produced identical vectors: {duplicate_vectors}")
    degenerate = [r["run_name"] for r in records
                  if r["sampling_role"] != "forced_or_diagnostic"
                  and r["max_relative_difference_from_reference"] < MIN_RELATIVE_DIFFERENCE_FROM_REFERENCE]
    if degenerate:
        raise RuntimeError(
            f"These points are indistinguishable from the reference paste, which is what a "
            f"silently-unapplied gas override looks like: {degenerate}"
        )
    print(f"      injectivity ok ({len(records)} maps, {len(by_map_sha)} distinct)")
    print(f"      all non-anchor points differ from the reference by "
          f">{MIN_RELATIVE_DIFFERENCE_FROM_REFERENCE:.0e} relative")

    payload = {
        "schema_version": "godmax.mock_sbi.responses.v1",
        "plan": str(args.plan), "plan_sha256": msc.sha256_file(args.plan),
        "round": plan["round"], "n_measured": len(records), "missing": missing,
        "vector_order": msc.VECTOR_ORDER,
        "estimator": {
            "noise_contract_sha256": ctx.contract_sha256,
            "workspace_sha256": ctx.workspace_sha256,
            "mask_array_sha256": ctx.mask_sha256,
            "fixed_galaxy_alm_sha256": ctx.galaxy_alm_sha256,
            "mask_metadata": ctx.mask_metadata,
        },
        "reference_paste_sha256": msc.sha256_array(mu_reference),
        "records": records,
    }
    # `entry["index"]` is the point's id in the FULL 128-point design, not its position in
    # this plan.  Indexing plan["entries"] with it is only correct when the plan is the
    # complete, contiguous design; on any SUBSET -- the partial-round trial, or a plan with
    # points excluded -- it silently pairs each row with a DIFFERENT point's u, and only
    # raises when an id happens to exceed the subset length.  Look the entry up by id.
    entry_by_index = {}
    for entry in plan["entries"]:
        if entry["index"] in entry_by_index:
            raise RuntimeError(f"plan has two entries with index {entry['index']}")
        entry_by_index[entry["index"]] = entry
    missing_ids = sorted({r["index"] for r in records} - set(entry_by_index))
    if missing_ids:
        raise RuntimeError(f"measured points absent from the plan: {missing_ids}")

    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, vectors=vectors,
                            theta=np.asarray([[r["theta"][n] for n in
                                               ("theta_ej_0", "alpha_nt", "mu_beta",
                                                "theta_co_0", "nu_theta_ej_M")]
                                              for r in records], dtype=np.float64),
                            u=np.asarray([entry_by_index[r["index"]]["u"]
                                          if entry_by_index[r["index"]]["u"] is not None
                                          else [np.nan] * 5 for r in records], dtype=np.float64),
                            log_q=np.asarray([np.nan if r["log_q"] is None else r["log_q"]
                                              for r in records], dtype=np.float64),
                            importance_eligible=np.asarray([r["importance_eligible"] for r in records]),
                            manifest_json=json.dumps(payload, sort_keys=True))
    os.replace(tmp, output)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {output}  ({len(records)} measured, {len(missing)} missing)")
    chi2 = np.asarray([r["chi2_against_observation"] for r in records])
    print(f"  chi2 vs the observation across the design: min {chi2.min():.1f} "
          f"median {np.median(chi2):.1f} max {chi2.max():.3g}   (42 bands, 0 free parameters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
