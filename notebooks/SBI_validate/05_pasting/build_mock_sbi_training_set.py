#!/usr/bin/env python3
"""Turn measured pasted responses into the NPE training set, and optionally write the
noise-augmented maps.

The identity this rests on
--------------------------
``x(theta, seed) = mu_paste(theta) + nu(seed)`` holds to 8.4e-15, because the galaxy
field is frozen across the five sampled gas parameters, the pseudo-Cl estimator is
bilinear in the two alms, and ``decouple_cell`` is linear.  So one expensive paste
yields as many training rows as wanted, and the augmented rows are computed by adding
a pre-measured noise vector rather than by re-measuring a map.

That is NOT the forbidden ``L @ epsilon`` augmentation.  Every ``nu`` in the bank came
from ``synalm -> alm2map -> mask -> map2alm -> alm2cl -> decouple_cell`` on the real
mask, so it carries the mask coupling and the estimator exactly.  The measured
consequence is that the bank's whitened covariance is 0.9266 ``C`` rather than ``C``:
the frozen covariance is the ensemble one and includes the y/tau/kappa SIGNAL legs'
own sample variance, while the mock holds the signal field fixed.  Both objects are
right; they condition on different things.  The campaign keeps ``C`` in every
likelihood so all three methods carry the identical conservatism, and the ~3.7% width
consequence is reported rather than tuned away.

Maps
----
``--write-maps K`` writes the first K augmented realizations per point as float32
``signal + noise`` HEALPix maps for y, tau and CMB kappa, unmasked, at the paste
nside.  They are written from the SAME seeds as the training rows, so a map and its
row are the same realization.  float32 costs about 1e-7 relative on a re-measured
bandpower, which is recorded as a diagnostic; the training rows themselves stay
float64 and are never read back from the float32 product.

The galaxy map is not rewritten: it is frozen and identical at every design point, so
it is referenced by hash instead of copied 512 x 5 times.
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
import hashlib
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import h5py
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import mock_sbi_common as msc

BANK_PATH = msc.REPO_ROOT / "data/SBI_validate/mock_sbi/noise_bank_training.npz"
# Namespace for the point->bank assignment.  Disjoint from the bank's own seed
# namespace and from the observation's, so no training row can ever reuse the
# observation's noise draw.
ASSIGNMENT_NAMESPACE = (20260825, 205)


def assign_bank_indices(round_label: str, n_points: int, n_replicas: int,
                        bank_size: int) -> np.ndarray:
    """Deterministic (point, replica) -> bank index, unique within a point.

    Unique WITHIN a point matters: two replicas of the same paste sharing one noise
    vector would be duplicate training rows wearing different labels.  Across points
    reuse is unavoidable once ``n_points * n_replicas`` exceeds the bank, and is
    harmless -- it correlates rows, which costs effective sample size, and does not
    bias ``p(x | theta)`` because every row is still a valid draw at its own theta.
    """

    if n_replicas > bank_size:
        raise ValueError(f"{n_replicas} replicas requested from a bank of {bank_size}")
    # Python's str.hash is salted per process (PYTHONHASHSEED), so it would give a
    # different assignment on every run and silently break reproducibility.  A digest
    # of the label is stable across processes, machines and interpreter versions.
    label_word = int.from_bytes(
        hashlib.sha256(str(round_label).encode("utf-8")).digest()[:4], "big")
    entropy = tuple(ASSIGNMENT_NAMESPACE) + (label_word,)
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return np.stack([rng.choice(bank_size, size=n_replicas, replace=False)
                     for _ in range(n_points)])


def write_augmented_maps(signal_maps, noise_maps, path: pathlib.Path, *, attrs: dict,
                         compression: str | None) -> int:
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as handle:
        handle.attrs["schema_version"] = "godmax.mock_sbi.augmented_maps.v1"
        for key, value in attrs.items():
            handle.attrs[key] = value if isinstance(value, (int, float, str)) \
                else json.dumps(value, sort_keys=True)
        for field, dataset in msc.MAP_DATASETS.items():
            total = (np.asarray(signal_maps[field], dtype=np.float64)
                     + np.asarray(noise_maps[field], dtype=np.float64))
            handle.create_dataset(dataset, data=total.astype(np.float32),
                                  compression=compression)
    os.replace(tmp, path)
    return path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--responses", type=pathlib.Path, nargs="+", required=True,
                        help="responses.npz from measure_mock_sbi_pastes.py, one per round")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--replicas", type=int, default=64,
                        help="noise draws per pasted point used for TRAINING")
    parser.add_argument("--write-maps", type=int, default=0,
                        help="also write this many augmented map sets per point (float32)")
    parser.add_argument("--maps-dir", type=pathlib.Path,
                        default=msc.REPO_ROOT / "data/SBI_validate/mock_sbi/augmented_maps")
    parser.add_argument("--map-compression", default=None,
                        choices=(None, "lzf", "gzip"))
    parser.add_argument("--verify-maps", type=int, default=1,
                        help="re-measure this many written map sets and report the float32 cost")
    args = parser.parse_args()
    if args.write_maps > args.replicas:
        raise SystemExit("--write-maps cannot exceed --replicas: a written map must "
                         "correspond to a training row, so its seed must be one of them")
    started = time.time()

    print("[1/4] loading the frozen estimator and the noise bank ...", flush=True)
    context = msc.load_estimator_context()
    bank_payload = np.load(BANK_PATH)
    bank_vectors = np.asarray(bank_payload["vectors"], dtype=np.float64)
    bank_seeds = np.asarray(bank_payload["seeds"], dtype=np.uint32)
    seed_order = [str(v) for v in bank_payload["seed_field_order"]]
    if tuple(seed_order) != tuple(msc.NOISE_FIELDS):
        raise RuntimeError(f"Bank seed field order {seed_order} != {msc.NOISE_FIELDS}")
    reserved = msc.reserved_observation_seeds()
    if reserved.intersection(bank_seeds.ravel().tolist()):
        raise RuntimeError("The bank contains an observation seed; a training row would "
                           "then carry the observation's own noise")
    print(f"      bank {bank_vectors.shape[0]} draws; whitened chi2/dim "
          f"{float(np.mean(np.sum(np.linalg.solve(context.cholesky, bank_vectors.T).T ** 2, axis=1)) / 42):.4f}")

    print(f"[2/4] pooling {len(args.responses)} response file(s) ...", flush=True)
    theta, u, log_q, eligible, mu, provenance = [], [], [], [], [], []
    for path in args.responses:
        payload = np.load(path, allow_pickle=True)
        manifest = json.loads(str(payload["manifest_json"]))
        if manifest["estimator"]["noise_contract_sha256"] != context.contract_sha256:
            raise RuntimeError(f"{path} was measured against a different noise contract")
        if manifest["estimator"]["workspace_sha256"] != context.workspace_sha256:
            raise RuntimeError(f"{path} was measured with a different NaMaster workspace")
        n = int(payload["vectors"].shape[0])
        theta.append(np.asarray(payload["theta"], dtype=np.float64))
        u.append(np.asarray(payload["u"], dtype=np.float64))
        log_q.append(np.asarray(payload["log_q"], dtype=np.float64))
        eligible.append(np.asarray(payload["importance_eligible"], dtype=bool))
        mu.append(np.asarray(payload["vectors"], dtype=np.float64))
        provenance.extend([{"responses": str(path), "round": manifest["round"],
                            "record": record} for record in manifest["records"]])
        print(f"      {path.name}: round {manifest['round']}, {n} points")
    theta = np.concatenate(theta); u = np.concatenate(u); log_q = np.concatenate(log_q)
    eligible = np.concatenate(eligible); mu = np.concatenate(mu)
    n_points = mu.shape[0]

    # Injectivity across ROUNDS as well as within one: the per-round script cannot see
    # that round 2 re-drew a point round 1 already pasted.
    vector_shas = [msc.sha256_array(row) for row in mu]
    if len(set(vector_shas)) != n_points:
        duplicates = {s for s in vector_shas if vector_shas.count(s) > 1}
        raise RuntimeError(f"{len(duplicates)} response vector(s) appear at more than one "
                           f"design point across the pooled rounds")

    print(f"[3/4] augmenting {n_points} points x {args.replicas} noise draws ...", flush=True)
    label = "+".join(str(p.parent.name) for p in args.responses)
    indices = assign_bank_indices(label, n_points, args.replicas, bank_vectors.shape[0])
    x_rows = mu[:, None, :] + bank_vectors[indices]                  # (n_points, M, 42)
    rows_theta = np.repeat(theta, args.replicas, axis=0)
    rows_u = np.repeat(u, args.replicas, axis=0)
    rows_log_q = np.repeat(log_q, args.replicas)
    rows_eligible = np.repeat(eligible, args.replicas)
    rows_point = np.repeat(np.arange(n_points), args.replicas)
    rows_x = x_rows.reshape(n_points * args.replicas, msc.VECTOR_SIZE)
    rows_seeds = bank_seeds[indices].reshape(n_points * args.replicas, 3)
    unique_rows = len({msc.sha256_array(row) for row in rows_x})
    print(f"      {rows_x.shape[0]} training rows, {unique_rows} distinct; "
          f"bank reuse factor {n_points * args.replicas / bank_vectors.shape[0]:.2f}x")
    if unique_rows != rows_x.shape[0]:
        raise RuntimeError("two training rows are identical; a point/replica pair was "
                           "assigned the same noise vector twice")

    map_records = []
    if args.write_maps:
        import healpy as hp
        args.maps_dir.mkdir(parents=True, exist_ok=True)
        print(f"[3b] writing {args.write_maps} float32 augmented map set(s) per point ...",
              flush=True)
        for point in range(n_points):
            record = provenance[point]["record"]
            source = pathlib.Path(record["map_path"])
            signal_maps = msc.read_paste_maps(source)
            point_dir = args.maps_dir / record["run_name"]
            point_dir.mkdir(parents=True, exist_ok=True)
            for replica in range(args.write_maps):
                bank_index = int(indices[point, replica])
                seeds = {name: int(bank_seeds[bank_index, position])
                         for position, name in enumerate(msc.NOISE_FIELDS)}
                noise_maps = {
                    name: hp.alm2map(msc.synalm_seeded(context.noise_cls[name], seeds[name]),
                                     nside=msc.NSIDE, lmax=msc.LMAX)
                    for name in msc.NOISE_FIELDS}
                out = point_dir / f"augmented_nside{msc.NSIDE}_replica{replica:03d}.h5"
                size = write_augmented_maps(
                    signal_maps, noise_maps, out,
                    attrs={"run_name": record["run_name"],
                           "theta_sha256": record["theta_sha256"],
                           "theta_json": json.dumps(record["theta"], sort_keys=True),
                           "replica": replica, "bank_index": bank_index,
                           "noise_field_seeds_json": json.dumps(seeds, sort_keys=True),
                           "signal_map_path": str(source),
                           "signal_map_sha256": record["map_sha256"],
                           "fixed_galaxy_map_note":
                               "the galaxy field is frozen across the five sampled gas "
                               "parameters and is NOT duplicated here -- and it is NOT in "
                               "signal_map_path either, which carries only y/tau/kappa "
                               "because the paste ran with get_galmap: false. Use "
                               "fixed_masked_alm/g in fixed_galaxy_alm_path (already "
                               "multiplied by the mask, lmax=2048), or rebuild the field "
                               "from the frozen galaxy catalog in "
                               "frozen_galaxy_catalog_path (dataset 'galaxies', columns "
                               "in its galaxy_catalog_columns_json attr).",
                           "fixed_galaxy_alm_path": str(msc.NOISE_CONTRACT_PATH),
                           "fixed_galaxy_alm_sha256": context.galaxy_alm_sha256,
                           "frozen_galaxy_catalog_path": str(msc.FROZEN_MAP_PATH),
                           "nside": msc.NSIDE, "lmax": msc.LMAX, "dtype": "float32",
                           "vector_sha256": msc.sha256_array(x_rows[point, replica])},
                    compression=args.map_compression)
                map_records.append({"run_name": record["run_name"], "replica": replica,
                                    "path": str(out), "bytes": int(size),
                                    "bank_index": bank_index, "seeds": seeds})
            if point == 0:
                print(f"      {args.write_maps} set(s) for point 0: "
                      f"{sum(r['bytes'] for r in map_records) / 2**30:.3f} GiB; "
                      f"projected total "
                      f"{sum(r['bytes'] for r in map_records) * n_points / 2**30:.1f} GiB",
                      flush=True)

    verification = []
    if args.write_maps and args.verify_maps:
        print(f"[3c] re-measuring {args.verify_maps} written map set(s) ...", flush=True)
        for record in map_records[:args.verify_maps]:
            with h5py.File(record["path"], "r") as handle:
                maps = {name: np.asarray(handle[dataset], dtype=np.float64)
                        for name, dataset in msc.MAP_DATASETS.items()}
            measured = msc.measure_map_vector(maps, context)
            point = [r["run_name"] for r in
                     [p["record"] for p in provenance]].index(record["run_name"])
            expected = x_rows[point, record["replica"]]
            relative = float(np.max(np.abs(measured / expected - 1.0)))
            verification.append({
                "path": record["path"], "max_relative": relative,
                "whitened_chi2": context.chi2(measured - expected)})
            print(f"      {pathlib.Path(record['path']).name}: float32 storage costs "
                  f"{relative:.3e} relative, whitened chi2 "
                  f"{verification[-1]['whitened_chi2']:.3e}")

    print("[4/4] writing the training set ...", flush=True)
    manifest = {
        "schema_version": "godmax.mock_sbi.training_set.v1",
        "responses": [str(p) for p in args.responses],
        "responses_sha256": [msc.sha256_file(p) for p in args.responses],
        "n_points": int(n_points), "replicas": int(args.replicas),
        "n_rows": int(rows_x.shape[0]),
        "bank": {"path": str(BANK_PATH), "sha256": msc.sha256_file(BANK_PATH),
                 "size": int(bank_vectors.shape[0]),
                 "reuse_factor": float(n_points * args.replicas / bank_vectors.shape[0])},
        "assignment_namespace": list(ASSIGNMENT_NAMESPACE),
        "estimator": {"noise_contract_sha256": context.contract_sha256,
                      "workspace_sha256": context.workspace_sha256,
                      "mask_array_sha256": context.mask_sha256,
                      "fixed_galaxy_alm_sha256": context.galaxy_alm_sha256},
        "vector_order": msc.VECTOR_ORDER,
        "maps_written": len(map_records),
        "maps_total_bytes": int(sum(r["bytes"] for r in map_records)),
        "map_records": map_records,
        "map_verification": verification,
        "provenance": provenance,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, x=rows_x, theta=rows_theta, u=rows_u,
                            log_q=rows_log_q, importance_eligible=rows_eligible,
                            point_index=rows_point, noise_seeds=rows_seeds,
                            mu=mu, point_theta=theta, point_u=u, point_log_q=log_q,
                            point_importance_eligible=eligible,
                            bank_indices=indices,
                            manifest_json=json.dumps(manifest, sort_keys=True))
    os.replace(tmp, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.output}: {rows_x.shape[0]} rows from {n_points} pasted points")
    if map_records:
        print(f"      {len(map_records)} augmented map sets, "
              f"{manifest['maps_total_bytes'] / 2**30:.1f} GiB under {args.maps_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
