#!/usr/bin/env python3
"""Mock-SBI posterior by NLE on measured pasted bandpowers.

This is the production counterpart of the Stage-0 oracle's winning arm. The oracle
established, on a paste-anchored stand-in simulator with an exactly-known posterior, that
NLE + score compression + 64 free noise draws per pasted point, pooled over four network
seeds, recovers the exact posterior at 512 points to 0.050 sigma drift and 6.6% width --
inside the pre-registered gate, where NPE and NRE both failed. This script runs that same
configuration on the real thing.

Why NLE needs no proposal correction, which is the whole reason it is used here
------------------------------------------------------------------------------
The design points come from a mixture centred on the theory-SBI posterior, NOT from the
box prior. For NPE that matters enormously: it estimates ``p(theta|x)`` from training
pairs whose theta came from the design ``q``, so the learned density is proportional to
``p(x|theta) q(theta)`` and must be reweighted by ``p0/q`` -- a correction measured to
collapse (Pareto k up to 1.35, invalid in 2 of 4 replicates at 512 points).

NLE estimates ``p(x|theta)``. The design distribution controls only *where the surrogate
is accurate*, never what is being estimated. The prior is applied analytically at sampling
time, by running MCMC against ``p0(u) * p_learned(x_obs|u)`` with ``p0 = N(0, I)`` in
probit coordinates. So ``log_q`` is carried in the training set for provenance and is
deliberately NOT used here. There is nothing to correct.

Partial designs are valid
-------------------------
The design was drawn IID from a fixed mixture, so any subset chosen by a mechanism
independent of theta is still an IID sample from that mixture -- a smaller design, not a
biased one. Paste cost was verified theta-independent (identical pixel-neighbour pair
counts across all concurrent tasks at the same chunk), so "whichever points finished
first" is such a mechanism. A trial run on partial results is therefore statistically
clean; it is only noisier, and the oracle's ladder quantifies how much.
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

import numpy as np
import torch
from torch.distributions import Independent, Normal

THIS_DIR = pathlib.Path(__file__).resolve().parent
for _p in (THIS_DIR, THIS_DIR.parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mock_sbi_common as msc

DATA = msc.REPO_ROOT / "data/SBI_validate/mock_sbi"
REFERENCE_POINT = DATA / "oracle_sc_reference_point.npz"
MOCK_OBSERVATION = msc.REPO_ROOT / "data/SBI_validate/three_probe_inference/observation_mock.h5"
PARAMETERS = ("theta_ej_0", "alpha_nt", "mu_beta", "theta_co_0", "nu_theta_ej_M")
PRIOR_LOW = np.array([0.5, 0.0, 0.005, 0.001, -1.0])
PRIOR_HIGH = np.array([8.0, 0.5, 1.5, 0.5, 1.0])
# Identical to the validated oracle arm.
TRAINING = dict(training_batch_size=512, learning_rate=5e-4, validation_fraction=0.10,
                stop_after_epochs=25, max_num_epochs=500)
MCMC = dict(method="slice_np_vectorized", num_chains=20, warmup_steps=250, thin=1)
NAMESPACE = (20260825, 1301)


def seeded(*spawn) -> int:
    return int(np.random.SeedSequence(tuple(NAMESPACE) + tuple(int(v) for v in spawn))
               .generate_state(1, dtype=np.uint32)[0])


def theta_from_u(u: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * ndtr(np.asarray(u, dtype=np.float64))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--training-set", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, default=4,
                        help="network replicates to pool. Four is what the oracle "
                             "validated; a single seed is not quotable (run-to-run "
                             "scatter measured at 0.078 vs 0.522 sigma).")
    parser.add_argument("--compression", choices=("score", "raw"), default="score")
    parser.add_argument("--posterior-samples", type=int, default=20000)
    parser.add_argument("--tag", default="trial")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("[1/5] loading the frozen estimator inputs ...", flush=True)
    import h5py
    with h5py.File(MOCK_OBSERVATION, "r") as handle:
        observation = np.asarray(handle["data_vector"], dtype=np.float64)
        cholesky = np.asarray(handle["cholesky"], dtype=np.float64)
        obs_kind = str(handle.attrs["observation_kind"])
    print(f"      observation {MOCK_OBSERVATION.name}: {obs_kind}")
    print(f"      sha256 {msc.sha256_array(observation)[:16]}")

    cache = np.load(REFERENCE_POINT, allow_pickle=True)
    operator = np.asarray(cache["operator"], dtype=np.float64)
    reference_prediction = np.asarray(cache["reference_prediction"], dtype=np.float64)
    print(f"      score operator from {REFERENCE_POINT.name} "
          f"(u_map {np.round(np.asarray(cache['u_map']), 3)})")

    def summarise(vectors: np.ndarray) -> np.ndarray:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        whitened = np.linalg.solve(cholesky, (vectors - reference_prediction[None, :]).T)
        return whitened.T if args.compression == "raw" else (operator @ whitened).T

    print("[2/5] loading the training set ...", flush=True)
    payload = np.load(args.training_set, allow_pickle=True)
    manifest = json.loads(str(payload["manifest_json"]))
    u_rows = np.asarray(payload["u"], dtype=np.float64)
    x_rows = np.asarray(payload["x"], dtype=np.float64)
    finite = np.all(np.isfinite(u_rows), axis=1) & np.all(np.isfinite(x_rows), axis=1)
    if not np.all(finite):
        # A forced anchor has no probit coordinate; it is a diagnostic execution, not an
        # IID design draw, and including it would misrepresent the design.
        print(f"      dropping {int((~finite).sum())} row(s) with no probit coordinate")
    u_rows, x_rows = u_rows[finite], x_rows[finite]
    n_points = int(manifest["n_points"])
    print(f"      {x_rows.shape[0]} rows from {n_points} pasted points "
          f"x {manifest['replicas']} noise draws")
    print(f"      bank reuse factor {manifest['bank']['reuse_factor']:.2f}x")
    if manifest["estimator"]["workspace_sha256"] != json.loads(
            str(payload["manifest_json"]))["estimator"]["workspace_sha256"]:
        raise RuntimeError("training set estimator identity is inconsistent")

    summaries = summarise(x_rows)
    observed = summarise(observation[None, :])[0]
    print(f"      summary dimension {observed.size}; |s_obs| {np.linalg.norm(observed):.4f}")

    print(f"[3/5] training {args.seeds} NLE replicate(s) ...", flush=True)
    from sbi.inference import SNLE
    prior = Independent(Normal(torch.zeros(5), torch.ones(5)), 1)
    observed_torch = torch.as_tensor(observed, dtype=torch.float32)
    per_seed, records = [], []
    for seed in range(args.seeds):
        t0 = time.time()
        inference = SNLE(prior=prior, density_estimator="nsf", device="cpu",
                         show_progress_bars=False)
        # One round: every pasted point is appended together. No `proposal` argument
        # exists for SNLE -- the prior is applied analytically when sampling below.
        inference.append_simulations(torch.as_tensor(u_rows, dtype=torch.float32),
                                     torch.as_tensor(summaries, dtype=torch.float32))
        torch.manual_seed(seeded(seed, 1))
        estimator = inference.train(**TRAINING, show_train_summary=False)
        posterior = inference.build_posterior(
            estimator, prior=prior, sample_with="mcmc", mcmc_method=MCMC["method"],
            mcmc_parameters=dict(num_chains=MCMC["num_chains"],
                                 warmup_steps=MCMC["warmup_steps"], thin=MCMC["thin"]))
        draws = np.asarray(posterior.sample((args.posterior_samples,), x=observed_torch,
                                            show_progress_bars=False), dtype=np.float64)
        per_seed.append(draws)
        records.append({"seed": seed, "seconds": time.time() - t0,
                        "mean_u": draws.mean(axis=0).tolist(),
                        "sd_u": draws.std(axis=0, ddof=1).tolist()})
        print(f"      seed {seed}: mean {np.round(draws.mean(axis=0), 3)}  "
              f"sd {np.round(draws.std(axis=0, ddof=1), 3)}  ({time.time()-t0:.0f}s)",
              flush=True)

    print("[4/5] pooling the replicates ...", flush=True)
    pooled_u = np.concatenate(per_seed, axis=0)
    pooled_theta = theta_from_u(pooled_u)
    spread = np.max(np.abs(np.asarray([r["mean_u"] for r in records])
                           - pooled_u.mean(axis=0)[None, :])
                    / pooled_u.std(axis=0, ddof=1)[None, :]) if args.seeds > 1 else float("nan")
    print(f"      pooled mean_u {np.round(pooled_u.mean(axis=0), 4)}")
    print(f"      pooled sd_u   {np.round(pooled_u.std(axis=0, ddof=1), 4)}")
    print(f"      largest single-seed departure from the pooled mean: {spread:.3f} sigma")
    print("      (that spread IS the ensemble's reason to exist; a single seed is not "
          "quotable)")

    print("[5/5] writing ...", flush=True)
    out = args.output_dir / f"mock_snle_{args.tag}_posterior.npz"
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, u=pooled_u, theta=pooled_theta,
                            **{f"u_seed_{r['seed']}": s for r, s in zip(records, per_seed)},
                            manifest_json=json.dumps({
                                "schema_version": "godmax.mock_sbi.snle_posterior.v1",
                                "tag": args.tag, "estimator": "SNLE (NLE, one round)",
                                "proposal_correction_required": False,
                                "compression": args.compression,
                                "seeds": args.seeds, "per_seed": records,
                                "n_pasted_points": n_points,
                                "n_training_rows": int(x_rows.shape[0]),
                                "training_set": str(args.training_set),
                                "training_set_sha256": msc.sha256_file(args.training_set),
                                "observation": str(MOCK_OBSERVATION),
                                "observation_kind": obs_kind,
                                "observation_sha256": msc.sha256_array(observation),
                                "reference_point": str(REFERENCE_POINT),
                                "max_seed_departure_sigma": float(spread),
                                "training": TRAINING, "mcmc": MCMC,
                                "namespace": list(NAMESPACE),
                                "elapsed_seconds": time.time() - started,
                                "parameter_names": list(PARAMETERS),
                            }, sort_keys=True))
    os.replace(tmp, out)
    print(f"\nwrote {out}")
    print(f"  {n_points} pasted points, {x_rows.shape[0]} training rows, "
          f"{args.seeds} pooled seeds, {(time.time()-started)/60:.1f} min")
    print("\nposterior in physical units (pooled):")
    for i, name in enumerate(PARAMETERS):
        q = np.percentile(pooled_theta[:, i], [5, 50, 95])
        print(f"   {name:16s} {q[1]:9.4f}   90% CI [{q[0]:9.4f}, {q[2]:9.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
