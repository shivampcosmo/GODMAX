#!/usr/bin/env python3
"""Build the pinned reference point shared by the three-probe HMC and SBI runs.

Produces, on the converged forward grid and in standard-normal probit
coordinates:

* ``u_map``     -- the maximum a posteriori point, from a multi-start L-BFGS
                   whose starting points are fixed constants (no oracle, no HMC);
* ``laplace_covariance`` -- the inverse Hessian of the negative log posterior at
                   the MAP, used as the NUTS ``inverse_mass_matrix`` and as the
                   scale of the SBI defensive proposal;
* ``score_operator``     -- the exact 5-dimensional normalized score summary;
* ``parity_vectors``     -- forward predictions at three fixed probit points, so
                   that any later run on a different backend can be checked
                   against the arithmetic that produced this artifact.

Both runners load this file and verify its hash.  Computing the MAP once, here,
is deliberate: it guarantees that the HMC preconditioner and the SBI compression
point are the same numbers, so "the two methods addressed the same problem" is
checkable rather than asserted.
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
import pathlib
import time

import numpy as np
from scipy.optimize import minimize

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from three_probe_agreement_common import (
    GRID, PARITY_PROBIT_POINTS, PARAMETER_NAMES, REFERENCE_POINT_PATH,
    atomic_json, backend_manifest, build_problem, environment_manifest,
    numerical_source_manifest, score_operator, sha256_array, theta_from_probit,
)

# Fixed, oracle-free multi-start set in probit coordinates.
START_POINTS = (
    (0.0, 0.0, 0.0, 0.0, 0.0),
    (-1.0, 1.0, 0.0, -1.0, -1.0),
    (1.0, -1.0, 1.0, 1.0, 1.0),
    (-2.0, 0.5, -0.5, 0.0, -2.0),
    (-1.5, 1.0, 0.5, -0.5, -0.25),
    (0.5, -0.5, -1.0, 0.5, 0.5),
    (-2.5, 1.5, 1.0, -1.0, -0.5),
    (2.0, -1.5, 0.0, 1.0, 1.5),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=pathlib.Path, default=None,
                        help="Registered inference contract supplying the "
                             "observation. Default: the production pasted-map "
                             "contract. The loader admits only registered "
                             "contracts, so this selects between audited inputs.")
    parser.add_argument("--output", type=pathlib.Path, default=REFERENCE_POINT_PATH)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--hessian-step", type=float, default=1.0e-3)
    parser.add_argument("--eigenvalue-floor", type=float, default=1.0e-6)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0e-2,
                        help="Reject the MAP if |grad| exceeds this.")
    parser.add_argument("--min-agreeing-starts", type=int, default=2,
                        help="How many independent starts must reach the best basin.")
    parser.add_argument("--basin-potential-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--allow-unconverged-map", action="store_true")
    parser.add_argument("--grid-override", type=int, nargs=4, default=None,
                        help="Cheap grid for smoke tests only; recorded in the artifact.")
    args = parser.parse_args()

    started = time.time()
    environment = environment_manifest()
    sources = numerical_source_manifest()
    backend = backend_manifest()
    grid = tuple(args.grid_override) if args.grid_override else GRID
    problem = build_problem(grid, contract_path=args.contract)
    value_and_grad = jax.jit(jax.value_and_grad(problem.potential_u))

    def objective(u: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(jnp.asarray(u, dtype=jnp.float64))
        return float(value), np.asarray(gradient, dtype=np.float64)

    attempts = []
    for index, start in enumerate(START_POINTS):
        result = minimize(objective, np.asarray(start, dtype=np.float64), jac=True,
                          method="L-BFGS-B", options=dict(maxiter=args.maxiter, ftol=1e-14, gtol=1e-10))
        attempts.append(dict(start=list(start), u=np.asarray(result.x, dtype=np.float64),
                             potential=float(result.fun), n_eval=int(result.nfev),
                             success=bool(result.success), message=str(result.message)))
        print(f"start {index}: potential {result.fun:.6f} after {result.nfev} evaluations "
              f"({'ok' if result.success else result.message})", flush=True)

    best = min(attempts, key=lambda item: item["potential"])
    u_map = best["u"]
    spread = float(np.max([np.linalg.norm(item["u"] - u_map) for item in attempts]))
    potential_spread = float(np.max([item["potential"] for item in attempts]) - best["potential"])

    # Central finite differences of the *analytic* gradient.  A full
    # jax.hessian through this forward (interpax cubic splines, the piecewise
    # j0_safe branches, the float32 painter round-trips) returned non-finite
    # second derivatives, so the second derivative is taken numerically from
    # first derivatives that are known to be finite.  Ten gradient evaluations.
    step = args.hessian_step
    gradient_norm = float(np.linalg.norm(objective(u_map)[1]))

    # Gate the selected MAP.  A reference point that is not actually a minimum
    # poisons everything downstream: it is the HMC preconditioner and start, and
    # the point the SBI score compression linearises about.  The previous version
    # had no such gate, so a short or stalled optimisation would have propagated
    # silently.
    agreeing = [item for item in attempts
                if item["potential"] - best["potential"] <= args.basin_potential_tolerance]
    basin_spread = float(np.max([np.linalg.norm(item["u"] - u_map) for item in agreeing]))
    convergence = dict(
        gradient_norm=gradient_norm,
        max_gradient_norm=float(args.max_gradient_norm),
        gradient_ok=bool(gradient_norm <= args.max_gradient_norm),
        n_starts=len(attempts),
        n_agreeing_starts=len(agreeing),
        min_agreeing_starts=int(args.min_agreeing_starts),
        agreement_ok=bool(len(agreeing) >= args.min_agreeing_starts),
        basin_u_spread=basin_spread,
        multistart_u_spread=spread,
        multistart_potential_spread=potential_spread,
        best_potential=best["potential"],
        runner_up_potential=float(sorted(item["potential"] for item in attempts)[1]),
    )
    print("\n=== MAP convergence ===")
    for key, value in convergence.items():
        print(f"   {key:28s} {value}")
    if not args.allow_unconverged_map:
        if not convergence["gradient_ok"]:
            raise RuntimeError(
                f"MAP is not converged: |grad| = {gradient_norm:.3e} exceeds "
                f"{args.max_gradient_norm:.1e}. Raise --maxiter, or pass "
                f"--allow-unconverged-map for a deliberately labelled diagnostic.")
        if not convergence["agreement_ok"]:
            raise RuntimeError(
                f"Only {len(agreeing)} of {len(attempts)} starts reached the best "
                f"basin (need {args.min_agreeing_starts}); potentials "
                f"{sorted(round(i['potential'], 4) for i in attempts)}. The forward "
                f"model has multiple basins and a single lucky start is not a "
                f"reference point.")

    hessian = np.zeros((5, 5), dtype=np.float64)
    for axis in range(5):
        offset = np.zeros(5); offset[axis] = step
        _, gradient_plus = objective(u_map + offset)
        _, gradient_minus = objective(u_map - offset)
        hessian[:, axis] = (gradient_plus - gradient_minus) / (2.0 * step)
    hessian = 0.5 * (hessian + hessian.T)
    if not np.all(np.isfinite(hessian)):
        raise RuntimeError("Finite-difference Hessian at the MAP is not finite")
    eigenvalues = np.linalg.eigvalsh(hessian)
    # The Hessian is only a preconditioner and a proposal scale, never part of
    # the target, so a floored eigenvalue cannot bias a posterior.  It is
    # recorded explicitly rather than applied silently.
    eigenvalue_floor = float(args.eigenvalue_floor)
    repaired = bool(np.min(eigenvalues) < eigenvalue_floor)
    if repaired:
        values, vectors = np.linalg.eigh(hessian)
        hessian = vectors @ np.diag(np.maximum(values, eigenvalue_floor)) @ vectors.T
        eigenvalues = np.linalg.eigvalsh(hessian)
    laplace_covariance = np.linalg.inv(hessian)

    scores = score_operator(problem, u_map)
    chi2_map = float(problem.chi2_u(jnp.asarray(u_map)))
    theta_map = theta_from_probit(u_map, problem.low, problem.high)[0]

    parity_points = np.asarray(PARITY_PROBIT_POINTS, dtype=np.float64)
    parity_vectors = np.stack([np.asarray(problem.predict_u(jnp.asarray(point)), dtype=np.float64)
                               for point in parity_points])
    parity_chi2 = [float(problem.chi2_u(jnp.asarray(point))) for point in parity_points]

    payload = dict(
        schema="godmax.sbi.three_probe_reference_point.v2",
        grid=list(grid),
        grid_is_converged=bool(grid == GRID),
        parameter_names=list(PARAMETER_NAMES),
        contract_sha256=problem.contract.contract_sha256,
        environment=environment,
        numerical_sources=sources,
        backend=backend,
        wall_seconds=time.time() - started,
        optimization=dict(
            starts=[dict(start=item["start"], potential=item["potential"], n_eval=item["n_eval"],
                         success=item["success"], message=item["message"],
                         u=item["u"].tolist()) for item in attempts],
            best_potential=best["potential"],
            multistart_u_spread=spread,
            multistart_potential_spread=potential_spread,
            gradient_norm_at_map=gradient_norm,
            convergence=convergence,
        ),
        u_map=u_map.tolist(),
        theta_map=theta_map.tolist(),
        chi2_at_map=chi2_map,
        chi2_reference=dict(retained_rank=42, n_varied=5, expected=37, expected_scatter=8.6),
        hessian=hessian.tolist(),
        hessian_eigenvalues=eigenvalues.tolist(),
        hessian_finite_difference_step=step,
        hessian_eigenvalue_floor_applied=repaired,
        hessian_eigenvalue_floor=eigenvalue_floor,
        laplace_covariance=laplace_covariance.tolist(),
        laplace_sigma=np.sqrt(np.diag(laplace_covariance)).tolist(),
        laplace_correlation=(laplace_covariance / np.outer(
            np.sqrt(np.diag(laplace_covariance)), np.sqrt(np.diag(laplace_covariance)))).tolist(),
        score_operator=scores["operator"].tolist(),
        score_reference_prediction=scores["reference_prediction"].tolist(),
        score_gram=scores["gram"].tolist(),
        score_gram_eigenvalues=scores["gram_eigenvalues"].tolist(),
        fisher_condition_number=scores["fisher_condition_number"],
        score_normalization="posterior_metric_fisher_plus_prior",
        # Set to False by the stage-1 parity step when CPU and GPU disagree.
        backend_portable=None,
        score_noise_covariance=scores["noise_covariance"].tolist(),
        score_noise_covariance_eigenvalues=np.linalg.eigvalsh(scores["noise_covariance"]).tolist(),
        observed_score=(scores["operator"] @ np.linalg.solve(
            problem.cholesky, problem.observation - scores["reference_prediction"])).tolist(),
        parity=dict(
            probit_points=parity_points.tolist(),
            vectors_sha256=sha256_array(parity_vectors),
            vectors=parity_vectors.tolist(),
            chi2=parity_chi2,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print("\n=== reference point ===")
    print("theta_map          :", dict(zip(PARAMETER_NAMES, np.round(theta_map, 5))))
    print("u_map              :", np.round(u_map, 5))
    print("chi2 at MAP        :", round(chi2_map, 3), "  (nominal reference 42-5 = 37 +- 8.6)")
    print("multistart spread  : |u| %.3e   potential %.3e" % (spread, potential_spread))
    print("gradient norm      : %.3e" % gradient_norm)
    print("laplace sigma (u)  :", np.round(np.sqrt(np.diag(laplace_covariance)), 4))
    print("Fisher cond number : %.4g" % scores["fisher_condition_number"])
    observed_score = np.asarray(payload["observed_score"])
    print("observed score     :", np.round(observed_score, 4),
          " |s| = %.4f  (a typical draw has |s| ~ %.4f)"
          % (np.linalg.norm(observed_score),
             np.sqrt(np.trace(scores["noise_covariance"]))))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
