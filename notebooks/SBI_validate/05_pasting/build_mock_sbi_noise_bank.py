#!/usr/bin/env python3
"""Build the theta-independent noise bank for mock SBI, and calibrate it.

Because the galaxy leg is frozen, ``x(theta, seed) = mu_paste(theta) + nu(seed)``
holds to machine precision with ``nu`` independent of ``theta`` (verified by
``validate_mock_sbi_foundations.py``).  So the noise only has to be measured
once, and every pasted point can be augmented with the whole bank for free.

Each ``nu`` is a field-level harmonic draw pushed through the contract's exact
estimator: synalm on the frozen noise curves, alm2map, mask-weighted centring,
map2alm, cross with the fixed galaxy alm, ``workspace.decouple_cell``.  Drawing
``L @ epsilon`` in bandpower space instead is forbidden for this experiment --
it bypasses the mask coupling and cannot be validated against map products.

Calibration: with N draws the empirical covariance of nu should match the frozen
contract covariance.  The whitened spectrum of ``C^-1/2 Cov(nu) C^-1/2`` should
concentrate at 1 with width ~sqrt(2/N).  A systematic departure means the noise
model and the covariance disagree, which would invalidate the theory runs too.
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
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

NAMESPACES = {"training": "NAMESPACE_TRAINING", "holdout": "NAMESPACE_HOLDOUT"}

_CONTEXT = None
_SEEDS = None


def _initializer(threads: int) -> None:
    global _CONTEXT
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import mock_sbi_common as msc
    _CONTEXT = msc.load_estimator_context()


def _draw(index: int) -> tuple[int, np.ndarray]:
    import mock_sbi_common as msc
    return index, msc.noise_vector(_SEEDS[index], _CONTEXT)


def _set_seeds(seeds) -> None:
    global _SEEDS
    _SEEDS = seeds


def calibration(vectors: np.ndarray, cholesky: np.ndarray) -> dict:
    """Whitened calibration of the bank against the frozen covariance."""

    n, dim = vectors.shape
    whitened = np.linalg.solve(cholesky, vectors.T).T          # (n, 42)
    chi2 = np.einsum("ij,ij->i", whitened, whitened)
    empirical = np.cov(whitened, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(0.5 * (empirical + empirical.T))
    mean_norm = float(np.mean(chi2))
    return {
        "n_draws": int(n),
        "dimension": int(dim),
        "expected_relative_precision": float(np.sqrt(2.0 / n)),
        "mean_whitened_chi2": mean_norm,
        "mean_whitened_chi2_over_dim": mean_norm / dim,
        "mean_whitened_chi2_standard_error": float(np.std(chi2, ddof=1) / np.sqrt(n)),
        "whitened_eigenvalue_min": float(eigenvalues.min()),
        "whitened_eigenvalue_max": float(eigenvalues.max()),
        "whitened_eigenvalue_median": float(np.median(eigenvalues)),
        "whitened_mean_abs_offset": float(np.max(np.abs(np.mean(whitened, axis=0)))),
        "per_band_sd_ratio_median": float(np.median(np.std(whitened, axis=0, ddof=1))),
        "per_band_sd_ratio_min": float(np.min(np.std(whitened, axis=0, ddof=1))),
        "per_band_sd_ratio_max": float(np.max(np.std(whitened, axis=0, ddof=1))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--namespace", choices=sorted(NAMESPACES), default="training")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    import mock_sbi_common as msc

    output = args.output or (msc.REPO_ROOT / f"data/SBI_validate/mock_sbi/noise_bank_{args.namespace}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    namespace = getattr(msc, NAMESPACES[args.namespace])
    seeds = msc.noise_bank_seeds(args.count, namespace)
    print(f"namespace {args.namespace} entropy {namespace}  count {args.count}", flush=True)
    print(f"seed collision checks passed against {len(msc.reserved_observation_seeds())} "
          f"frozen observation seeds", flush=True)

    started = time.time()
    vectors = np.empty((args.count, msc.VECTOR_SIZE), dtype=np.float64)
    context = mp.get_context("fork")
    _set_seeds(seeds)
    with context.Pool(processes=args.workers, initializer=_initializer,
                      initargs=(args.threads_per_worker,)) as pool:
        done = 0
        for index, vector in pool.imap_unordered(_draw, range(args.count), chunksize=4):
            vectors[index] = vector
            done += 1
            if done % 128 == 0 or done == args.count:
                rate = done / (time.time() - started)
                remaining = (args.count - done) / max(rate, 1e-9)
                print(f"  {done}/{args.count}  {rate:.2f} draws/s  eta {remaining/60:.1f} min",
                      flush=True)
    elapsed = time.time() - started

    if not np.all(np.isfinite(vectors)):
        raise RuntimeError("Noise bank contains non-finite entries")
    ctx = msc.load_estimator_context()
    report = calibration(vectors, ctx.cholesky)
    report.update({
        "namespace": args.namespace,
        "namespace_entropy": list(namespace),
        "elapsed_seconds": elapsed,
        "draws_per_second": args.count / elapsed,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "noise_contract_sha256": ctx.contract_sha256,
        "workspace_sha256": ctx.workspace_sha256,
        "mask_array_sha256": ctx.mask_sha256,
        "vectors_sha256": msc.sha256_array(vectors),
        "vector_order": msc.VECTOR_ORDER,
    })

    seed_array = np.asarray([[s[name] for name in msc.NOISE_FIELDS] for s in seeds], dtype=np.uint32)
    # np.savez_compressed appends ".npz" unless the *name* already ends in it, so
    # write through an explicit handle to keep the atomic-rename temp name exact.
    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, vectors=vectors, seeds=seed_array,
                            seed_field_order=np.asarray(msc.NOISE_FIELDS, dtype="U8"),
                            report_json=json.dumps(report, sort_keys=True))
    os.replace(tmp, output)
    (output.with_suffix(".report.json")).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\nwrote {output}  ({elapsed/60:.1f} min, {args.count/elapsed:.2f} draws/s)")
    print(f"  mean whitened chi2 / dim   {report['mean_whitened_chi2_over_dim']:.4f} "
          f"(+/- {report['mean_whitened_chi2_standard_error']/report['dimension']:.4f})")
    print(f"  whitened eigenvalues       {report['whitened_eigenvalue_min']:.3f} .. "
          f"{report['whitened_eigenvalue_max']:.3f} (median {report['whitened_eigenvalue_median']:.3f})")
    print(f"  expected precision         {report['expected_relative_precision']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
