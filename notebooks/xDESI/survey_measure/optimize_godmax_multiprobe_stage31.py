#!/usr/bin/env python
"""Bounded MAP optimizer for the xDESI GODMAX Stage-31 likelihood."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


def _preconfigure_jax_from_argv(argv: Sequence[str]) -> None:
    platform = None
    for i, arg in enumerate(argv):
        if arg == "--platform" and i + 1 < len(argv):
            platform = argv[i + 1]
            break
        if arg.startswith("--platform="):
            platform = arg.split("=", 1)[1]
            break
    if platform == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    elif platform == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "True")


_preconfigure_jax_from_argv(sys.argv[1:])

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import godmax_multiprobe_hmc_stage31 as hmc31
import godmax_multiprobe_theory_utils as gmt


def log_status(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=hmc31.DEFAULT_STAGE31_CONFIG)
    parser.add_argument("--platform", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--gpu-sanity-check", action="store_true")
    parser.add_argument("--gpu-sanity-matrix-size", type=int, default=2048)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix", default="stage31_map")
    parser.add_argument("--validate-only", action="store_true", help="Prepare context and starts, then exit before optimization.")
    parser.add_argument("--method", choices=["adam", "lbfgsb", "adam-lbfgsb"], default="adam-lbfgsb")
    parser.add_argument("--num-starts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--init-params", default=None)
    parser.add_argument("--no-init-start", action="store_true")
    parser.add_argument("--no-fiducial-start", action="store_true")
    parser.add_argument("--random-mode", choices=["prior", "around-init", "around-fiducial"], default="around-init")
    parser.add_argument("--start-jitter", type=float, default=0.08)
    parser.add_argument("--adam-steps", type=int, default=60)
    parser.add_argument("--adam-lr", type=float, default=2.0e-3)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--adam-grad-clip", type=float, default=1.0e6)
    parser.add_argument("--lbfgs-maxiter", type=int, default=80)
    parser.add_argument("--lbfgs-maxfun", type=int, default=120)
    parser.add_argument("--lbfgs-ftol", type=float, default=1.0e-7)
    parser.add_argument("--lbfgs-gtol", type=float, default=1.0e-5)
    parser.add_argument("--lbfgs-maxls", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--eval-log-every",
        type=int,
        default=1,
        help="Print objective value/gradient begin/end messages every N evaluations; 0 disables.",
    )
    parser.add_argument("--combine-worker-dir", default=None)
    parser.add_argument("--worker-pattern", default="worker_*/map_optimization_summary.json")
    return parser


def bounds_arrays(specs: Sequence[hmc31.ParameterSpec]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray([spec.prior_min for spec in specs], dtype=np.float64)
    upper = np.asarray([spec.prior_max for spec in specs], dtype=np.float64)
    return lower, upper, upper - lower


def sample_to_x(specs: Sequence[hmc31.ParameterSpec], sample: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(sample[spec.name]) for spec in specs], dtype=np.float64)


def x_to_sample(specs: Sequence[hmc31.ParameterSpec], x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {spec.name: float(x[i]) for i, spec in enumerate(specs)}


def x_to_u(x: np.ndarray, lower: np.ndarray, span: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(x, dtype=np.float64) - lower) / span, 0.0, 1.0)


def u_to_x(u: np.ndarray, lower: np.ndarray, span: np.ndarray) -> np.ndarray:
    return lower + np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0) * span


class Objective:
    def __init__(self, context: hmc31.FitContext, *, eval_log_every: int = 1):
        self.context = context
        self.lower, self.upper, self.span = bounds_arrays(context.parameter_specs)
        self.lower_j = jnp.asarray(self.lower, dtype=jnp.float64)
        self.span_j = jnp.asarray(self.span, dtype=jnp.float64)
        self.n_eval = 0
        self.eval_log_every = int(eval_log_every)

        def chi2_u(u):
            u = jnp.clip(jnp.asarray(u, dtype=jnp.float64), 0.0, 1.0)
            x = self.lower_j + u * self.span_j
            return hmc31.parameter_vector_chi2(context, x)

        self._value_and_grad = jax.value_and_grad(chi2_u)

    def value_and_grad_np(self, u_np: np.ndarray) -> Tuple[float, np.ndarray]:
        self.n_eval += 1
        log_this_eval = self.eval_log_every > 0 and (self.n_eval == 1 or self.n_eval % self.eval_log_every == 0)
        t0 = time.time()
        if log_this_eval:
            log_status(f"[objective] eval={self.n_eval} begin")
        value, grad = self._value_and_grad(jnp.asarray(u_np, dtype=jnp.float64))
        value_np = float(np.asarray(value))
        grad_np = np.asarray(grad, dtype=np.float64)
        if not np.isfinite(value_np) or not np.all(np.isfinite(grad_np)):
            bad = int(np.sum(~np.isfinite(grad_np)))
            log_status(f"[objective] non-finite value/grad at eval {self.n_eval}: value={value_np}, bad_grad={bad}")
            value_np = 1.0e300
            grad_np = np.zeros_like(np.asarray(u_np, dtype=np.float64))
        if log_this_eval:
            grad_norm = float(np.linalg.norm(grad_np))
            log_status(
                f"[objective] eval={self.n_eval} done chi2={value_np:.8e} "
                f"grad_norm={grad_norm:.4e} elapsed_s={time.time() - t0:.1f}"
            )
        return value_np, grad_np

    def value_np(self, u_np: np.ndarray) -> float:
        value, _ = self.value_and_grad_np(u_np)
        return value


def make_starts(
    context: hmc31.FitContext,
    *,
    num_starts: int,
    seed: int,
    init_params: Optional[str],
    include_init: bool,
    include_fiducial: bool,
    random_mode: str,
    jitter: float,
) -> List[dict]:
    lower, _, span = bounds_arrays(context.parameter_specs)
    rng = np.random.default_rng(int(seed))
    fid_sample = hmc31.pack_fiducial_sample(context.parameter_specs)
    fid_u = x_to_u(sample_to_x(context.parameter_specs, fid_sample), lower, span)
    init_sample = hmc31.pack_sample_from_params_file(context, init_params) if init_params else None
    init_u = x_to_u(sample_to_x(context.parameter_specs, init_sample), lower, span) if init_sample else None

    starts: List[dict] = []
    if include_init and init_u is not None and len(starts) < num_starts:
        starts.append({"name": "init_params", "u": init_u})
    if include_fiducial and len(starts) < num_starts:
        starts.append({"name": "fiducial", "u": fid_u})

    if random_mode == "around-init" and init_u is not None:
        center = init_u
    elif random_mode in {"around-init", "around-fiducial"}:
        center = fid_u
    else:
        center = None

    while len(starts) < num_starts:
        if center is None:
            u = rng.uniform(0.0, 1.0, size=len(context.parameter_specs))
            mode = "prior"
        else:
            u = np.clip(center + rng.normal(0.0, float(jitter), size=len(context.parameter_specs)), 0.0, 1.0)
            mode = random_mode
        starts.append({"name": f"random_{mode}_{len(starts)}", "u": u})
    return starts


def run_adam(
    objective: Objective,
    u0: np.ndarray,
    *,
    steps: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    grad_clip: float,
    log_every: int,
    label: str,
) -> dict:
    u = np.clip(np.asarray(u0, dtype=np.float64), 0.0, 1.0)
    m = np.zeros_like(u)
    v = np.zeros_like(u)
    best_u = u.copy()
    best_value = np.inf
    history = []
    t0 = time.time()
    for step in range(1, int(steps) + 1):
        u_eval = u.copy()
        value, grad = objective.value_and_grad_np(u)
        grad_norm = float(np.linalg.norm(grad))
        if np.isfinite(grad_norm) and grad_clip > 0.0 and grad_norm > grad_clip:
            grad = grad * (grad_clip / grad_norm)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        mhat = m / (1.0 - beta1**step)
        vhat = v / (1.0 - beta2**step)
        u = np.clip(u - lr * mhat / (np.sqrt(vhat) + eps), 0.0, 1.0)
        if value < best_value:
            best_value = value
            best_u = u_eval
        if step == 1 or step == steps or (log_every > 0 and step % log_every == 0):
            row = {"step": step, "chi2": value, "grad_norm": grad_norm, "elapsed_s": time.time() - t0}
            history.append(row)
            log_status(
                f"[adam:{label}] step={step}/{steps} chi2={value:.8e} "
                f"grad_norm={grad_norm:.4e} elapsed_s={time.time() - t0:.1f}"
            )
    return {
        "method": "adam",
        "label": label,
        "success": True,
        "message": "completed",
        "chi2": float(best_value),
        "u": best_u,
        "history": history,
        "n_eval": objective.n_eval,
        "elapsed_s": time.time() - t0,
    }


def run_lbfgsb(
    objective: Objective,
    u0: np.ndarray,
    *,
    maxiter: int,
    maxfun: int,
    ftol: float,
    gtol: float,
    maxls: int,
    label: str,
) -> dict:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - environment failure.
        raise RuntimeError("scipy is required for L-BFGS-B optimization.") from exc

    t0 = time.time()
    eval0 = objective.n_eval

    def fun_and_jac(u):
        value, grad = objective.value_and_grad_np(u)
        log_status(f"[lbfgsb:{label}] eval={objective.n_eval} chi2={value:.8e} elapsed_s={time.time() - t0:.1f}")
        return value, grad

    result = minimize(
        fun_and_jac,
        np.clip(np.asarray(u0, dtype=np.float64), 0.0, 1.0),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 1.0)] * len(u0),
        options={
            "maxiter": int(maxiter),
            "maxfun": int(maxfun),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "maxls": int(maxls),
        },
    )
    return {
        "method": "lbfgsb",
        "label": label,
        "success": bool(result.success),
        "message": str(result.message),
        "chi2": float(result.fun),
        "u": np.asarray(result.x, dtype=np.float64),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "n_eval": int(objective.n_eval - eval0),
        "elapsed_s": time.time() - t0,
    }


def save_map_outputs(
    context: hmc31.FitContext,
    *,
    output_dir: Path,
    suffix: str,
    best_sample: Mapping[str, float],
    best_chi2: float,
    trials: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_config = hmc31.apply_sample_to_config(context.config, context.parameter_specs, best_sample)
    best_params_path = output_dir / f"map_bestfit_params_{suffix}.yaml"
    with open(best_params_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(gmt.to_jsonable(best_config["params"]), handle, sort_keys=False)

    best_theory = np.asarray(hmc31.evaluate_sample_theory_vector(context, best_sample))
    measurement = hmc31.measurement_for_plots(context)
    stats = gmt.comparison_statistics(measurement, best_theory)
    measurement_identity = gmt.measurement_identity_sha256(measurement)
    likelihood_identity = hmc31.likelihood_identity(context.likelihood)
    comparison_config_identity = gmt.comparison_config_identity_sha256(context.config)
    theory_response_identity = gmt.theory_response_identity_sha256(context.config)
    parameter_names = [spec.name for spec in context.parameter_specs]
    parameter_contract_identity = hmc31.parameter_contract_identity_sha256(
        context.parameter_specs
    )
    vector_cache_fields = gmt.theory_vector_cache_fields(
        best_theory,
        measurement_identity,
        {
            "product_kind": "stage31_optimized_bestfit_active",
            "chain_contract_version": hmc31.STAGE31_CHAIN_CONTRACT_VERSION,
            "likelihood_identity_sha256": likelihood_identity,
            "comparison_config_identity_sha256": comparison_config_identity,
            "theory_response_identity_sha256": theory_response_identity,
            "parameter_names": parameter_names,
            "parameter_contract_identity_sha256": parameter_contract_identity,
            "best_sample": dict(best_sample),
            "best_whitened_chi2": float(best_chi2),
        },
    )

    theory_path = output_dir / f"map_bestfit_theory_data_vector_{suffix}.npz"
    np.savez_compressed(
        theory_path,
        ell_band=np.asarray(measurement.ell),
        data_vector=np.asarray(measurement.data_vector),
        theory_vector=best_theory,
        covariance=np.asarray(measurement.covariance),
        spectrum_names=np.asarray(measurement.names),
        slice_start=np.asarray(measurement.starts, dtype=np.int64),
        slice_stop=np.asarray(measurement.stops, dtype=np.int64),
        measurement_identity_sha256=np.asarray(measurement_identity),
        likelihood_identity_sha256=np.asarray(likelihood_identity),
        chain_contract_version=np.asarray(hmc31.STAGE31_CHAIN_CONTRACT_VERSION),
        theory_response_identity_sha256=np.asarray(theory_response_identity),
        parameter_names=np.asarray(parameter_names),
        parameter_contract_identity_sha256=np.asarray(parameter_contract_identity),
        best_sample_json=np.asarray(json.dumps(best_sample)),
        best_whitened_chi2=np.asarray(best_chi2),
        **vector_cache_fields,
    )

    pdf_path = output_dir / f"map_bestfit_comparison_{suffix}.pdf"
    plot_paths = gmt.plot_family_comparisons(measurement, best_theory, output_dir, pdf_path=pdf_path)

    summary_path = output_dir / "map_optimization_summary.json"
    summary = {
        "suffix": suffix,
        "best_whitened_chi2": best_chi2,
        "best_sample": dict(best_sample),
        "bestfit_params_path": best_params_path,
        "bestfit_theory_vector_path": theory_path,
        "posterior_predictive_pdf": pdf_path,
        "plot_paths": plot_paths,
        "pseudo_inverse_stats": stats,
        "trials": trials,
        "metadata": metadata,
        "static_summary": hmc31.static_summary(context),
        "parameter_specs": hmc31.parameter_specs_jsonable(context.parameter_specs),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(gmt.to_jsonable(summary), handle, indent=2)

    trials_path = output_dir / f"map_trials_{suffix}.npz"
    np.savez_compressed(
        trials_path,
        trial_chi2=np.asarray([trial["chi2"] for trial in trials], dtype=np.float64),
        trial_method=np.asarray([str(trial["method"]) for trial in trials]),
        trial_label=np.asarray([str(trial["label"]) for trial in trials]),
        trial_success=np.asarray([bool(trial.get("success", False)) for trial in trials]),
    )

    return {
        "summary": summary_path,
        "trials": trials_path,
        "bestfit_params": best_params_path,
        "bestfit_theory_vector": theory_path,
        "pdf": pdf_path,
        "plots": plot_paths,
        "best_whitened_chi2": best_chi2,
    }


def run_optimizer(args: argparse.Namespace) -> dict:
    log_status("[map] configure runtime begin")
    runtime = hmc31.configure_numpyro_platform(args.platform)
    print(json.dumps(gmt.to_jsonable({"runtime": runtime}), indent=2), flush=True)
    if args.gpu_sanity_check:
        log_status("[map] gpu sanity check begin")
        check = hmc31.gpu_sanity_check(args.gpu_sanity_matrix_size, require_gpu=args.platform == "gpu")
        print(json.dumps(gmt.to_jsonable({"gpu_sanity_check": check}), indent=2), flush=True)

    log_status("[map] prepare_fit_context begin")
    context = hmc31.prepare_fit_context(args.config)
    log_status(f"[map] prepare_fit_context done n_parameters={len(context.parameter_specs)}")
    starts = make_starts(
        context,
        num_starts=args.num_starts,
        seed=args.seed,
        init_params=args.init_params,
        include_init=not args.no_init_start,
        include_fiducial=not args.no_fiducial_start,
        random_mode=args.random_mode,
        jitter=args.start_jitter,
    )
    log_status(f"[map] starts prepared: {[str(start['name']) for start in starts]}")
    if args.validate_only:
        return {
            "validated": True,
            "runtime": runtime,
            "n_parameters": len(context.parameter_specs),
            "n_starts": len(starts),
            "start_names": [str(start["name"]) for start in starts],
            "static_summary": hmc31.static_summary(context),
        }
    log_status("[map] objective init begin")
    objective = Objective(context, eval_log_every=args.eval_log_every)
    log_status("[map] objective init done")
    lower, _, span = bounds_arrays(context.parameter_specs)
    trials = []
    t0 = time.time()
    for i, start in enumerate(starts):
        label = f"{i:03d}_{start['name']}"
        u_start = np.asarray(start["u"], dtype=np.float64)
        log_status(f"[start:{label}] begin")
        current_u = u_start
        if args.method in {"adam", "adam-lbfgsb"}:
            adam = run_adam(
                objective,
                current_u,
                steps=args.adam_steps,
                lr=args.adam_lr,
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                eps=args.adam_eps,
                grad_clip=args.adam_grad_clip,
                log_every=args.log_every,
                label=label,
            )
            current_u = np.asarray(adam["u"], dtype=np.float64)
            trials.append(adam)
        if args.method in {"lbfgsb", "adam-lbfgsb"}:
            lbfgs = run_lbfgsb(
                objective,
                current_u,
                maxiter=args.lbfgs_maxiter,
                maxfun=args.lbfgs_maxfun,
                ftol=args.lbfgs_ftol,
                gtol=args.lbfgs_gtol,
                maxls=args.lbfgs_maxls,
                label=label,
            )
            trials.append(lbfgs)
        log_status(f"[start:{label}] done current_best_chi2={min(float(trial['chi2']) for trial in trials):.8e}")

    if not trials:
        raise RuntimeError("No optimizer trials were run.")
    best_trial = min(trials, key=lambda trial: float(trial["chi2"]))
    best_x = u_to_x(np.asarray(best_trial["u"], dtype=np.float64), lower, span)
    best_sample = x_to_sample(context.parameter_specs, best_x)
    metadata = {
        "runtime": runtime,
        "method": args.method,
        "num_starts": args.num_starts,
        "seed": args.seed,
        "init_params": args.init_params,
        "random_mode": args.random_mode,
        "start_jitter": args.start_jitter,
        "objective_evaluations": objective.n_eval,
        "elapsed_s": time.time() - t0,
    }
    log_status(f"[map] optimization done best_chi2={float(best_trial['chi2']):.8e}; saving outputs")
    return save_map_outputs(
        context,
        output_dir=Path(args.output_dir),
        suffix=args.suffix,
        best_sample=best_sample,
        best_chi2=float(best_trial["chi2"]),
        trials=trials,
        metadata=metadata,
    )


def combine_workers(args: argparse.Namespace) -> dict:
    log_status("[combine] configure runtime begin")
    runtime = hmc31.configure_numpyro_platform(args.platform)
    print(json.dumps(gmt.to_jsonable({"runtime": runtime}), indent=2), flush=True)
    if args.gpu_sanity_check:
        log_status("[combine] gpu sanity check begin")
        check = hmc31.gpu_sanity_check(args.gpu_sanity_matrix_size, require_gpu=args.platform == "gpu")
        print(json.dumps(gmt.to_jsonable({"gpu_sanity_check": check}), indent=2), flush=True)

    log_status("[combine] prepare_fit_context begin")
    context = hmc31.prepare_fit_context(args.config)
    log_status("[combine] prepare_fit_context done")
    worker_dir = Path(args.combine_worker_dir)
    summaries = sorted(worker_dir.glob(args.worker_pattern))
    if not summaries:
        raise FileNotFoundError(f"No worker summaries found under {worker_dir} with pattern {args.worker_pattern!r}.")
    log_status(f"[combine] found {len(summaries)} worker summaries")

    worker_payloads = []
    for path in summaries:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_summary_path"] = str(path)
        worker_payloads.append(payload)
    best_payload = min(worker_payloads, key=lambda payload: float(payload["best_whitened_chi2"]))
    output_dir = Path(args.output_dir)
    result = save_map_outputs(
        context,
        output_dir=output_dir,
        suffix=args.suffix,
        best_sample=best_payload["best_sample"],
        best_chi2=float(best_payload["best_whitened_chi2"]),
        trials=[
            {
                "method": "worker_best",
                "label": Path(payload["_summary_path"]).parent.name,
                "success": True,
                "message": payload["_summary_path"],
                "chi2": float(payload["best_whitened_chi2"]),
                "u": np.zeros(len(context.parameter_specs), dtype=np.float64),
            }
            for payload in worker_payloads
        ],
        metadata={
            "runtime": runtime,
            "worker_summaries": summaries,
            "best_worker_summary": best_payload["_summary_path"],
        },
    )

    best_worker_dir = Path(best_payload["_summary_path"]).parent
    copied_summary = output_dir / "best_worker_summary.json"
    shutil.copy2(best_payload["_summary_path"], copied_summary)
    result["best_worker_summary"] = copied_summary
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.combine_worker_dir:
        result = combine_workers(args)
    else:
        result = run_optimizer(args)
    print(json.dumps(gmt.to_jsonable(result), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
