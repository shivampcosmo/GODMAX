#!/usr/bin/env python3
"""Preconditioned four-chain NUTS for the frozen three-probe contract (v2).

Changes relative to the depth-6 runner, each aimed at a measured failure:

0. **Checkpointing.**  Warm-up state is pickled once, then sampling proceeds in
   chunks of ``--checkpoint-interval`` draws per chain.  Each chunk writes the
   cumulative samples, the sampler state, and a ``progress.json`` carrying the
   running divergence count, tree-depth saturation, acceptance and a projected
   time to completion, so a long run can be inspected while it is still running
   and resumed with ``--resume`` if it is pre-empted or hits the wall clock.

1. **Probit coordinates.**  The parameters are sampled as an unbounded standard
   normal ``u`` and mapped to the box by ``theta = low + (high-low) Phi(u)``,
   which reproduces the uniform box prior exactly.  ``mu_beta`` and
   ``theta_co_0`` are prior-dominated and ``theta_ej_0`` reaches within 0.13% of
   its lower bound, so the previous ``dist.Uniform`` parameterisation put heavy
   mass where the sigmoid transform stretches to infinity.  The SBI runner uses
   the identical coordinates, so the two posteriors are directly comparable
   without a change of variables.

2. **Laplace preconditioning.**  ``inverse_mass_matrix`` is initialised from the
   inverse Hessian at the pinned MAP and the chains start at the MAP.  The
   posterior has ``corr(theta_ej_0, nu_theta_ej_M) = +0.955`` and
   ``corr(theta_ej_0, alpha_nt) = -0.703``; that ridge is what forced 22.883% of
   depth-6 transitions to saturate.

3. **Tree depth 7 and 1500 warm-up steps**, replacing the hard-coded
   ``depth == 6`` identity check that pinned a configuration already measured to
   saturate.  Depth is capped at 7 by ``MAX_ALLOWED_TREE_DEPTH``; the point of
   the Laplace metric in item 2 is that a well-conditioned posterior does not
   need deep trees, so if depth 7 still saturates, the preconditioner is what
   needs work and the run is rejected rather than the gate relaxed.

4. **Converged forward grid** ``(256, 48, 48, 2049)``.  The previous
   ``(64, 48, 22, 64)`` grid fails its own 0.5% non-regression gate at 1.0e-2
   median in every probe and biases the whitened chi-square by 13 to 20 units.

5. **Replayable chi-square.**  A CPU replay of the depth-6 artifact disagreed
   with its own stored ``chi2`` deterministic by about 88 units at every one of
   its 12,000 samples, with byte-identical sources and hash-verified inputs; the
   only remaining difference was the execution backend.  This runner therefore
   records the full ``src/`` source manifest, the backend, and an explicit
   post-hoc re-evaluation of the potential at a fixed subset of its own draws,
   and it checks the forward against the reference artifact's parity vectors.
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
import pickle
import time

# Stamped at import so the wall-clock budget counts JAX import and forward-model
# construction, which are several minutes and would otherwise be invisible to it.
_PROCESS_STARTED = time.time()

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

from three_probe_agreement_common import (
    GRID, PARAMETER_NAMES, PARITY_RELATIVE_TOLERANCE, REFERENCE_POINT_PATH,
    atomic_json, atomic_npz, backend_manifest, build_problem,
    credible_interval_summary, environment_manifest, numerical_source_manifest,
    sha256_file, theta_from_probit,
)

GATE_FAILURE_EXIT_CODE = 3
MAX_ALLOWED_TREE_DEPTH = 7
def _atomic_pickle(path: pathlib.Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _checkpoint_stem(draws: int) -> str:
    return f"checkpoint_{draws:06d}"


def _newest_checkpoint(output_dir: pathlib.Path) -> int | None:
    """Highest draw count with a complete (npz + state + ready) checkpoint."""

    best = None
    for ready in sorted(output_dir.glob("checkpoint_*.ready.json")):
        stem = ready.name[: -len(".ready.json")]
        if not ((output_dir / f"{stem}.npz").is_file() and (output_dir / f"{stem}.state.pkl").is_file()):
            continue
        draws = int(json.loads(ready.read_text())["draws_per_chain"])
        best = draws if best is None else max(best, draws)
    return best


CONVERGENCE_GATE = dict(max_rhat=1.01, min_ess=400.0, max_divergences=0, max_saturation=0.01)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-wall-seconds", type=float, default=None,
                        help="Stop sampling cleanly once the next checkpoint chunk "
                             "would cross this many seconds from process start, and "
                             "write the final artifact with the draws actually "
                             "collected. Makes a wall-clock budget a guarantee "
                             "instead of a rate extrapolation.")
    parser.add_argument("--contract", type=pathlib.Path, default=None,
                        help="Registered inference contract supplying the "
                             "observation. Default: the production pasted-map "
                             "contract. The loader admits only registered "
                             "contracts, so this selects between audited inputs.")
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--reference-point", type=pathlib.Path, default=REFERENCE_POINT_PATH)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--max-tree-depth", type=int, default=7)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=260822901)
    parser.add_argument("--init-jitter", type=float, default=0.5)
    parser.add_argument("--replay-draws", type=int, default=64)
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Draws per chain between checkpoints.  Each checkpoint holds the "
                             "cumulative samples, so contours can be made from any of them "
                             "while the run is still going.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the newest complete checkpoint in --output-dir.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-nonportable-forward", action="store_true",
                        help="Sample anyway on a forward model that fails CPU/GPU "
                             "parity; the artifact is labelled accordingly.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Accept a non-converged reference grid and label the artifact SMOKE.")
    args = parser.parse_args()

    if args.max_tree_depth > MAX_ALLOWED_TREE_DEPTH:
        raise ValueError(
            f"max_tree_depth {args.max_tree_depth} exceeds the allowed maximum "
            f"{MAX_ALLOWED_TREE_DEPTH} for this campaign")

    started = time.time()
    environment = environment_manifest()
    sources = numerical_source_manifest()
    backend = backend_manifest()

    reference = json.loads(args.reference_point.read_text())
    if reference["schema"] != "godmax.sbi.three_probe_reference_point.v2":
        raise RuntimeError("Unexpected reference-point schema")
    # Three states: True (stage 1 verified CPU/GPU parity), False (stage 1
    # measured a disagreement), None (never checked).  Only True samples freely;
    # None is treated as unverified rather than as permission.
    portable = reference.get("backend_portable")
    if portable is False and not args.allow_nonportable_forward:
        raise RuntimeError(
            "The pinned reference point records that the forward model is NOT "
            "backend-portable. Refusing to sample: a posterior from a forward "
            "model that disagrees with itself across backends is not a result. "
            "Re-run stage 1 after fixing the compiler toolchain, or pass "
            "--allow-nonportable-forward for an explicitly-labelled diagnostic.")
    if portable is None and not (args.smoke_test or args.allow_nonportable_forward):
        raise RuntimeError(
            "The pinned reference point carries no backend-parity verdict. "
            "Run stage 1 (which stamps it) before sampling, or pass "
            "--smoke-test / --allow-nonportable-forward deliberately.")
    grid = tuple(reference["grid"])
    if grid != GRID and not args.smoke_test:
        raise RuntimeError(f"Reference point grid {grid} is not the converged grid {GRID}")
    reference_sha256 = sha256_file(args.reference_point)

    problem = build_problem(grid, jit_compile=True, contract_path=args.contract)
    if reference["contract_sha256"] != problem.contract.contract_sha256:
        raise RuntimeError("Reference point was built against a different inference contract")

    # Backend parity: is this machine computing the same forward as the machine
    # that produced the reference artifact?  Recorded either way, never silently.
    parity_points = np.asarray(reference["parity"]["probit_points"], dtype=np.float64)
    parity_expected = np.asarray(reference["parity"]["vectors"], dtype=np.float64)
    parity_here = np.stack([np.asarray(problem.predict_u(jnp.asarray(p)), dtype=np.float64)
                            for p in parity_points])
    parity_relative = np.abs(parity_here / parity_expected - 1.0)
    parity = dict(
        reference_backend=reference["backend"]["default_backend"],
        reference_device_kind=reference["backend"]["device_kind"],
        this_backend=backend["default_backend"],
        this_device_kind=backend["device_kind"],
        max_relative_difference=float(parity_relative.max()),
        median_relative_difference=float(np.median(parity_relative)),
        tolerance=PARITY_RELATIVE_TOLERANCE,
        passed=bool(parity_relative.max() <= PARITY_RELATIVE_TOLERANCE),
        chi2_reference=reference["parity"]["chi2"],
        chi2_here=[float(problem.chi2_u(jnp.asarray(p))) for p in parity_points],
    )
    print("backend parity:", json.dumps(parity, sort_keys=True), flush=True)

    u_map = np.asarray(reference["u_map"], dtype=np.float64)
    laplace_covariance = np.asarray(reference["laplace_covariance"], dtype=np.float64)
    laplace_covariance = 0.5 * (laplace_covariance + laplace_covariance.T)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight = dict(
        schema="godmax.sbi.three_probe_hmc_v2_preflight.v1",
        status="PASS",
        grid=list(grid),
        grid_is_converged=bool(grid == GRID),
        smoke_test=bool(args.smoke_test),
        allow_nonportable_forward=bool(args.allow_nonportable_forward),
        backend_portable=portable,
        contract_sha256=problem.contract.contract_sha256,
        reference_point_sha256=reference_sha256,
        runner_source_sha256=sha256_file(pathlib.Path(__file__)),
        numerical_sources=sources,
        environment=environment,
        backend=backend,
        backend_parity=parity,
        configuration=dict(warmup=args.warmup, samples=args.samples, chains=args.chains,
                           max_tree_depth=args.max_tree_depth, target_accept=args.target_accept,
                           seed=args.seed, init_jitter=args.init_jitter,
                           dense_mass=True, chain_method="vectorized",
                           parameterization="exact_box_to_standard_normal_probit",
                           preconditioner="laplace_inverse_hessian_at_pinned_map"),
        chi2_at_map=reference["chi2_at_map"],
    )
    atomic_json(args.output_dir / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return

    potential_u = problem.potential_u

    def probabilistic_model() -> None:
        u = numpyro.sample("u", dist.Normal(jnp.zeros(5), jnp.ones(5)).to_event(1))
        # The standard-normal prior is declared above, so only the likelihood
        # goes in the factor; potential_u already contains 0.5|u|^2.
        numpyro.factor("three_probe_loglike", -(potential_u(u) - 0.5 * jnp.dot(u, u)))
        theta = problem.theta_of_u(u)
        for index, name in enumerate(PARAMETER_NAMES):
            numpyro.deterministic(name, theta[index])

    # Chains start at the MAP, dispersed by a fraction of the Laplace scale, so
    # r_hat still measures mixing between genuinely separated starting points.
    key_init, key_run = jax.random.split(jax.random.PRNGKey(args.seed))
    chol = np.linalg.cholesky(laplace_covariance)
    dispersion = np.asarray(jax.random.normal(key_init, (args.chains, 5)), dtype=np.float64)
    init_u = u_map[None, :] + args.init_jitter * dispersion @ chol.T

    kernel = NUTS(
        probabilistic_model,
        dense_mass=True,
        target_accept_prob=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        inverse_mass_matrix=jnp.asarray(laplace_covariance),
        adapt_mass_matrix=True,
        adapt_step_size=True,
    )
    extra_fields = ("potential_energy", "diverging", "accept_prob", "num_steps")
    # Clamp before validating, so a short run (a smoke test) does not have to
    # restate the interval just to stay under the sample count.
    interval = min(max(int(args.checkpoint_interval), 1), int(args.samples))
    process_started = _PROCESS_STARTED
    # Enough time to write hmc_samples.npz, run the chi2 replay and emit the gate.
    # Scaled so a small budget (tests, smoke runs) is not consumed entirely by the
    # reserve, while a production budget still keeps the full 10 minutes.
    finalisation_reserve_seconds = (
        600.0 if args.max_wall_seconds is None
        else min(600.0, 0.15 * float(args.max_wall_seconds)))
    if args.samples % interval != 0:
        raise ValueError(f"samples ({args.samples}) must be a multiple of "
                         f"checkpoint-interval ({interval}); pass "
                         f"--checkpoint-interval to a divisor of --samples")
    sampler = MCMC(kernel, num_warmup=args.warmup, num_samples=interval,
                   num_chains=args.chains, chain_method="vectorized", progress_bar=True)

    sample_parts: dict[str, list[np.ndarray]] = {}
    extra_parts: dict[str, list[np.ndarray]] = {}
    completed = 0
    resumed_from = None
    warmup_seconds = 0.0

    resume_draws = _newest_checkpoint(args.output_dir) if args.resume else None
    if resume_draws:
        stem = _checkpoint_stem(resume_draws)
        with np.load(args.output_dir / f"{stem}.npz", allow_pickle=False) as payload:
            for key in payload.files:
                if key.startswith("sample_"):
                    sample_parts[key[len("sample_"):]] = [payload[key]]
                elif key.startswith("extra_"):
                    extra_parts[key[len("extra_"):]] = [payload[key]]
        with (args.output_dir / f"{stem}.state.pkl").open("rb") as handle:
            state = pickle.load(handle)
        meta = json.loads((args.output_dir / f"{stem}.ready.json").read_text())
        # A resumed chain must be continuing the same target.  Stage 1 rebuilds
        # the reference point on every run, so a stale checkpoint in a fixed
        # output directory is a realistic way to silently mix two posteriors.
        for field, current in (("reference_point_sha256", reference_sha256),
                               ("contract_sha256", problem.contract.contract_sha256),
                               ("numerical_sources_aggregate", sources["aggregate_sha256"])):
            if field in meta and meta[field] != current:
                raise RuntimeError(
                    f"Refusing to resume: checkpoint {stem} has {field}="
                    f"{meta[field]} but this run has {current}. Move or delete "
                    f"{args.output_dir} to start a fresh chain.")
        if meta.get("grid", list(grid)) != list(grid) or \
                meta.get("max_tree_depth", args.max_tree_depth) != args.max_tree_depth:
            raise RuntimeError(
                f"Refusing to resume: checkpoint {stem} used grid {meta.get('grid')} "
                f"and depth {meta.get('max_tree_depth')}, this run uses {list(grid)} "
                f"and depth {args.max_tree_depth}")
        completed = resume_draws
        resumed_from = stem
        warmup_seconds = float(meta["warmup_seconds"])
        print(f"resuming from {stem}: {completed}/{args.samples} draws per chain", flush=True)
    else:
        warmup_started = time.time()
        sampler.warmup(key_run, init_params={"u": jnp.asarray(init_u)},
                       extra_fields=extra_fields, collect_warmup=False)
        warmup_seconds = time.time() - warmup_started
        state = sampler.post_warmup_state
        _atomic_pickle(args.output_dir / "warmup_state.pkl", jax.device_get(state))
        atomic_json(args.output_dir / "warmup_complete.json", dict(
            warmup_steps=args.warmup, warmup_seconds=warmup_seconds,
            state_sha256=sha256_file(args.output_dir / "warmup_state.pkl")))
        print(f"warm-up complete in {warmup_seconds:.1f} s", flush=True)

    sampling_started = time.time()
    stopped_early = None
    while completed < args.samples:
        # A wall-clock budget must never be enforced by SLURM killing the process:
        # that loses the final artifact and the gate verdict, leaving only
        # checkpoints. Stop one chunk short instead and finalise honestly.
        if args.max_wall_seconds is not None and completed > (resume_draws or 0):
            elapsed = time.time() - process_started
            per_draw = (time.time() - sampling_started) / max(completed - (resume_draws or 0), 1)
            projected = elapsed + per_draw * interval + finalisation_reserve_seconds
            if projected > args.max_wall_seconds:
                stopped_early = dict(
                    reason="max_wall_seconds", limit_seconds=float(args.max_wall_seconds),
                    elapsed_seconds=float(elapsed),
                    projected_next_chunk_seconds=float(projected),
                    draws_per_chain_collected=int(completed),
                    draws_per_chain_requested=int(args.samples),
                    seconds_per_draw=float(per_draw),
                    finalisation_reserve_seconds=float(finalisation_reserve_seconds))
                print(f"[stop] wall budget {args.max_wall_seconds:.0f} s would be "
                      f"exceeded by the next {interval}-draw chunk "
                      f"(elapsed {elapsed:.0f} s, {per_draw:.2f} s/draw). "
                      f"Finalising with {completed}/{args.samples} draws per chain.",
                      flush=True)
                break
        sampler.post_warmup_state = state
        sampler.run(state.rng_key, extra_fields=extra_fields)
        state = sampler.last_state
        chunk_samples = sampler.get_samples(group_by_chain=True)
        chunk_extras = sampler.get_extra_fields(group_by_chain=True)
        for key, value in chunk_samples.items():
            sample_parts.setdefault(key, []).append(np.asarray(value))
        for key, value in chunk_extras.items():
            extra_parts.setdefault(key, []).append(np.asarray(value))
        completed += interval

        cumulative_samples = {k: np.concatenate(v, axis=1) for k, v in sample_parts.items()}
        cumulative_extras = {k: np.concatenate(v, axis=1) for k, v in extra_parts.items()}
        sample_parts = {k: [v] for k, v in cumulative_samples.items()}
        extra_parts = {k: [v] for k, v in cumulative_extras.items()}

        stem = _checkpoint_stem(completed)
        running_divergences = int(cumulative_extras["diverging"].sum())
        running_steps = cumulative_extras["num_steps"]
        running_saturation = float(np.mean(running_steps >= (2 ** args.max_tree_depth - 1)))
        atomic_npz(args.output_dir / f"{stem}.npz",
                   **{f"sample_{k}": v for k, v in cumulative_samples.items()},
                   **{f"extra_{k}": v for k, v in cumulative_extras.items()})
        _atomic_pickle(args.output_dir / f"{stem}.state.pkl", jax.device_get(state))
        # Live status, rewritten every checkpoint, so the run can be inspected
        # while it is still going rather than only after it finishes.
        status = dict(
            draws_per_chain=completed, target_draws_per_chain=args.samples,
            fraction_complete=completed / args.samples,
            chains=args.chains, warmup_seconds=warmup_seconds,
            sampling_seconds=time.time() - sampling_started,
            seconds_per_draw=(time.time() - sampling_started) / max(completed - (resume_draws or 0), 1),
            projected_remaining_seconds=((time.time() - sampling_started)
                                         / max(completed - (resume_draws or 0), 1)
                                         * (args.samples - completed)),
            running_divergences=running_divergences,
            running_tree_depth_saturation=running_saturation,
            running_mean_accept_prob=float(cumulative_extras["accept_prob"].mean()),
            running_mean_num_steps=float(running_steps.mean()),
            resumed_from=resumed_from,
            reference_point_sha256=reference_sha256,
            contract_sha256=problem.contract.contract_sha256,
            numerical_sources_aggregate=sources["aggregate_sha256"],
            grid=list(grid),
            max_tree_depth=args.max_tree_depth,
            npz_sha256=sha256_file(args.output_dir / f"{stem}.npz"),
            state_sha256=sha256_file(args.output_dir / f"{stem}.state.pkl"),
        )
        # Health warning at every checkpoint.  The smoke run showed acceptance
        # 0.204 against a target of 0.9 at depth 4 with 25 warm-up steps, where
        # step-size adaptation has barely started.  If the production run still
        # looks like that after 1500 warm-up steps, the Laplace metric is not
        # working and the remaining wall clock is wasted -- so say so early and
        # loudly rather than only in the final gate.
        health = []
        if status["running_mean_accept_prob"] < 0.6:
            health.append(f"acceptance {status['running_mean_accept_prob']:.3f} far below "
                          f"target {args.target_accept}")
        if running_saturation > 0.05:
            health.append(f"tree-depth saturation {running_saturation:.2%} above 5%")
        if running_divergences > 0.01 * completed * args.chains:
            health.append(f"{running_divergences} divergences in "
                          f"{completed * args.chains} transitions")
        status["health_warnings"] = health
        atomic_json(args.output_dir / f"{stem}.ready.json", status)
        atomic_json(args.output_dir / "progress.json", status)
        if health:
            print("[WARNING] preconditioning looks ineffective: " + "; ".join(health)
                  + " -- this run will fail its gate; consider cancelling.", flush=True)
        print(f"[checkpoint] {completed}/{args.samples} draws/chain  "
              f"divergences {running_divergences}  saturation {running_saturation:.3%}  "
              f"accept {status['running_mean_accept_prob']:.3f}  "
              f"eta {status['projected_remaining_seconds'] / 3600.0:.2f} h", flush=True)

    wall_seconds = time.time() - started
    samples = {k: np.concatenate(v, axis=1) for k, v in sample_parts.items()}
    extras = {k: np.concatenate(v, axis=1) for k, v in extra_parts.items()}
    u_samples = np.asarray(samples["u"], dtype=np.float64)
    # After an early stop the chain is legitimately shorter than requested; the
    # invariant is that it matches what the loop reported, not the request.
    expected_draws = completed
    if u_samples.shape[:2] != (args.chains, expected_draws):
        raise RuntimeError(f"Assembled sample shape {u_samples.shape} does not match "
                           f"({args.chains}, {expected_draws}, 5)")
    theta_samples = {name: np.asarray(samples[name], dtype=np.float64) for name in PARAMETER_NAMES}

    diagnostics = summary(theta_samples | {"u": np.asarray(samples["u"])}, group_by_chain=True)
    per_parameter = {
        name: {key: float(value) for key, value in values.items() if np.ndim(value) == 0}
        for name, values in diagnostics.items() if name in PARAMETER_NAMES
    }
    max_rhat = max(value["r_hat"] for value in per_parameter.values())
    min_ess = min(value["n_eff"] for value in per_parameter.values())
    divergences = int(np.asarray(extras["diverging"]).sum())
    num_steps = np.asarray(extras["num_steps"])
    saturation = float(np.mean(num_steps >= (2 ** args.max_tree_depth - 1)))

    # Replayable chi-square: recompute the potential from scratch at a fixed
    # subset of the recorded draws.  This is the check the depth-6 artifact
    # failed by 88 units, and it is now part of the artifact itself.
    flat_u = u_samples.reshape(-1, 5)
    stride = max(flat_u.shape[0] // max(args.replay_draws, 1), 1)
    replay_index = np.arange(0, flat_u.shape[0], stride)[: args.replay_draws]
    replay_chi2 = np.asarray([float(problem.chi2_u(jnp.asarray(flat_u[i]))) for i in replay_index])
    in_chain_potential = np.asarray(extras["potential_energy"]).reshape(-1)[replay_index]
    replay_potential = 0.5 * replay_chi2 + 0.5 * np.sum(flat_u[replay_index] ** 2, axis=1)
    potential_offset = replay_potential - in_chain_potential
    replay = dict(
        n_draws=int(replay_index.size),
        chi2_min=float(replay_chi2.min()), chi2_median=float(np.median(replay_chi2)),
        chi2_q95=float(np.percentile(replay_chi2, 95.0)),
        potential_offset_median=float(np.median(potential_offset)),
        potential_offset_max_abs_deviation=float(np.max(np.abs(potential_offset - np.median(potential_offset)))),
        # NumPyro's recorded potential energy carries an additive normalisation
        # constant, so only the *spread* of the offset is meaningful.
        self_consistent=bool(np.max(np.abs(potential_offset - np.median(potential_offset))) < 1.0e-6),
    )

    gate_items = dict(
        max_rhat=max_rhat <= CONVERGENCE_GATE["max_rhat"],
        min_ess=min_ess >= CONVERGENCE_GATE["min_ess"],
        divergences=divergences <= CONVERGENCE_GATE["max_divergences"],
        tree_depth_saturation=saturation <= CONVERGENCE_GATE["max_saturation"],
        chi2_replayable=replay["self_consistent"],
    )
    gate = all(gate_items.values())

    atomic_npz(
        args.output_dir / "hmc_samples.npz",
        u=u_samples,
        **{f"sample_{name}": value for name, value in theta_samples.items()},
        **{f"extra_{name}": np.asarray(value) for name, value in extras.items()},
        replay_index=replay_index, replay_chi2=replay_chi2,
    )
    payload = dict(
        schema="godmax.sbi.three_probe_hmc_v2.v1",
        status=("SMOKE_" if args.smoke_test else "") + ("PASS" if gate else "COMPLETED_REJECTED"),
        gate=gate_items,
        gate_thresholds=CONVERGENCE_GATE,
        configuration=preflight["configuration"] | {"output_dir": str(args.output_dir.resolve())},
        grid=list(grid),
        grid_is_converged=bool(grid == GRID),
        smoke_test=bool(args.smoke_test),
        allow_nonportable_forward=bool(args.allow_nonportable_forward),
        backend_portable=portable,
        contract_sha256=problem.contract.contract_sha256,
        reference_point_sha256=reference_sha256,
        runner_source_sha256=sha256_file(pathlib.Path(__file__)),
        numerical_sources=sources,
        environment=environment,
        backend=backend,
        backend_parity=parity,
        wall_seconds=wall_seconds,
        checkpoint_interval=interval,
        resumed_from=resumed_from,
        max_rhat=max_rhat, min_ess=min_ess, divergences=divergences,
        tree_depth_saturation_fraction=saturation,
        mean_num_steps=float(num_steps.mean()), max_num_steps=int(num_steps.max()),
        mean_accept_prob=float(np.asarray(extras["accept_prob"]).mean()),
        per_parameter=per_parameter,
        theta_summary=credible_interval_summary(
            np.stack([theta_samples[name].reshape(-1) for name in PARAMETER_NAMES], axis=1),
            PARAMETER_NAMES),
        u_summary=credible_interval_summary(flat_u, tuple(f"u_{n}" for n in PARAMETER_NAMES)),
        posterior_correlation=np.corrcoef(
            np.stack([theta_samples[name].reshape(-1) for name in PARAMETER_NAMES])).tolist(),
        chi2_replay=replay,
        chi2_reference=dict(retained_rank=42, n_varied=5, expected=37, expected_scatter=8.6),
    )
    payload["stopped_early"] = stopped_early
    payload["draws_per_chain_collected"] = int(completed)
    if stopped_early is not None:
        payload["gate"]["reached_requested_draws"] = False
        payload["status"] = "COMPLETED_REJECTED"
    atomic_json(args.output_dir / "hmc_diagnostics.json", payload)
    atomic_json(args.output_dir / "convergence_gate.json",
                {key: payload[key] for key in ("status", "gate", "max_rhat", "min_ess",
                                               "divergences", "tree_depth_saturation_fraction",
                                               "backend_parity", "chi2_replay")})
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("numerical_sources", "posterior_correlation")}, sort_keys=True))
    print(f"\nABSOLUTE FIT: replayed whitened chi2 median {replay['chi2_median']:.2f} "
          f"against the nominal reference 42-5 = 37 +- 8.6.")
    if not gate:
        print(f"HMC completed but FAILED the gate: "
              f"{[k for k, v in gate_items.items() if not v]}", flush=True)
        raise SystemExit(GATE_FAILURE_EXIT_CODE)


if __name__ == "__main__":
    main()
