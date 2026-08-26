#!/usr/bin/env python3
"""One-point paste/theory transfer and the frozen round-1 design guide q0.

Two products, both cheap, both required before any production paste is submitted.

**The transfer.**  ``r_hat = mu_paste(theta_ref) / mu_theory(theta_ref)``, a
42-vector measured from the single paste that already exists.  ``theta_ref`` is
read from ``params_default.yaml`` -- the paster's own configuration, which is what
actually produced the frozen map.  ``mu_theory`` is the production JAX forward
model at the identical grid the theory runs used.

**The guide.**  ``mu_tilde(theta) = r_hat * mu_theory(theta)`` is a zeroth-order
paste emulator: it removes the per-band multiplicative offset (median -15% in gy)
that makes the raw theory model misfit the pasted observation by chi2 ~161.  Its
Laplace approximation in probit coordinates is the round-1 design guide.

Why this matters: the *raw* theory posterior on the pasted observation puts the
mock's own generating point at the 99.98th percentile, so drawing a 200-300 point
paste budget from it would spend almost all of it where the paste model does not
fit.  This script measures whether the transfer correction repairs that, and
refuses to emit a guide if it does not.

A Gaussian guide is sufficient by design: it is mixed with a broadened component
and a prior component before any point is drawn, and round 2 re-centres on the
mock posterior.  It is a proposal, never a posterior.
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

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

THIS_DIR = pathlib.Path(__file__).resolve().parent
for _p in (THIS_DIR, THIS_DIR.parents[2], THIS_DIR.parents[2] / "src",
           THIS_DIR.parents[2] / "notebooks" / "xDESI"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mock_sbi_common as msc
from three_probe_agreement_common import (
    GRID, backend_manifest, build_problem, numerical_source_manifest, probit_from_theta,
    sha256_array, theta_from_probit,
)
from three_probe_fast_paste import SAMPLED_GAS_PARAMETERS

PARAMETER_NAMES = tuple(SAMPLED_GAS_PARAMETERS)

# The transfer is a ratio, so a theory band that is near zero or of the opposite
# sign to the paste would make it meaningless.  Refuse rather than emit garbage.
MIN_ABS_THEORY_BAND_SNR = 1.0
# The guide is only useful if it actually brackets where the paste model can fit.
# The raw theory posterior fails this at 24.5; a guide worse than the 99th
# percentile of a 5-dof chi-square would not be a usable design centre.
MAX_GENERATING_POINT_MAHALANOBIS2 = float(chi2_dist.ppf(0.99, 5))


def reference_theta() -> tuple[np.ndarray, dict]:
    """The gas parameters the frozen paste was actually produced with."""

    params_path = msc.REPO_ROOT / "param_files/params_default.yaml"
    with params_path.open() as handle:
        params = yaml.safe_load(handle)
    sim = params["sim_params"]
    theta = np.asarray([float(sim[name]) for name in PARAMETER_NAMES], dtype=np.float64)
    return theta, {
        "source": "param_files/params_default.yaml sim_params",
        "source_sha256": msc.sha256_file(params_path),
        "names": list(PARAMETER_NAMES),
        "values": theta.tolist(),
        "note": "the paster's own configuration; this is the point the frozen map was painted at",
    }


def laplace(potential, u_start: np.ndarray, *, step: float = 1.0e-3) -> dict:
    """MAP plus finite-difference Hessian of a scalar potential in probit space."""

    value_and_grad = jax.jit(jax.value_and_grad(potential))

    def objective(u):
        value, grad = value_and_grad(jnp.asarray(u, dtype=jnp.float64))
        return float(value), np.asarray(grad, dtype=np.float64)

    result = minimize(objective, np.asarray(u_start, dtype=np.float64), jac=True,
                      method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-10})
    u_map = np.asarray(result.x, dtype=np.float64)
    dim = u_map.size
    hessian = np.empty((dim, dim), dtype=np.float64)
    for i in range(dim):
        plus, minus = u_map.copy(), u_map.copy()
        plus[i] += step
        minus[i] -= step
        _, gp = objective(plus)
        _, gm = objective(minus)
        hessian[i] = (gp - gm) / (2.0 * step)
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.any(eigenvalues <= 0.0):
        raise RuntimeError(f"Laplace Hessian is not positive definite: eigenvalues {eigenvalues}")
    covariance = np.linalg.inv(hessian)
    return {
        "u_map": u_map,
        "hessian": hessian,
        "hessian_eigenvalues": eigenvalues,
        "covariance": covariance,
        "sigma": np.sqrt(np.diag(covariance)),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "gradient_norm_at_map": float(np.linalg.norm(objective(u_map)[1])),
    }


def mahalanobis2(point: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    delta = np.asarray(point, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return float(delta @ np.linalg.solve(covariance, delta))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=msc.REPO_ROOT / "data/SBI_validate/mock_sbi")
    parser.add_argument("--reference-vector", type=pathlib.Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    reference_path = args.reference_vector or (args.output_dir / "reference_paste_vector.npz")
    stored = np.load(reference_path)
    mu_paste = np.asarray(stored["mu_paste_reference"], dtype=np.float64)
    observation_stored = np.asarray(stored["observation"], dtype=np.float64)

    theta_ref, theta_provenance = reference_theta()
    print(f"[1/5] reference paste point (from params_default.yaml)")
    for name, value in zip(PARAMETER_NAMES, theta_ref):
        print(f"      {name:16s} {value:.6g}")

    print("[2/5] building the production theory forward model ...", flush=True)
    problem = build_problem(contract_path=msc.INFERENCE_CONTRACT_PATH)
    if not np.array_equal(problem.observation, observation_stored):
    # probit_from_theta / theta_from_probit are batch helpers returning (n, 5).
        raise RuntimeError("Contract observation differs from the archived reference observation")
    u_ref = probit_from_theta(theta_ref, problem.low, problem.high)[0]
    mu_theory = np.asarray(problem.predict_u(jnp.asarray(u_ref)), dtype=np.float64)

    print("[3/5] transfer diagnostics")
    band_sd = np.sqrt(np.diag(problem.contract.covariance))
    theory_snr = np.abs(mu_theory) / band_sd
    if np.any(theory_snr < MIN_ABS_THEORY_BAND_SNR):
        bad = np.where(theory_snr < MIN_ABS_THEORY_BAND_SNR)[0]
        raise RuntimeError(f"Theory bands {bad.tolist()} are below |SNR|=1; a ratio transfer is unsafe")
    if np.any(np.sign(mu_theory) != np.sign(mu_paste)):
        bad = np.where(np.sign(mu_theory) != np.sign(mu_paste))[0]
        raise RuntimeError(f"Paste and theory differ in sign at bands {bad.tolist()}")
    r_hat = mu_paste / mu_theory

    def chi2_of(vector: np.ndarray) -> float:
        w = np.linalg.solve(problem.cholesky, observation_stored - vector)
        return float(w @ w)

    rows = []
    for index, spectrum in enumerate(msc.SPECTRA):
        sl = slice(index * msc.N_BAND, (index + 1) * msc.N_BAND)
        rows.append((spectrum, float(np.median(r_hat[sl])), float(r_hat[sl].min()),
                     float(r_hat[sl].max()), float(np.median(theory_snr[sl]))))
    print(f"      {'probe':10s} {'median r':>9s} {'min r':>8s} {'max r':>8s} {'median |SNR|':>13s}")
    for spectrum, median, low, high, snr in rows:
        print(f"      {spectrum:10s} {median:9.4f} {low:8.4f} {high:8.4f} {snr:13.1f}")
    chi2_raw = chi2_of(mu_theory)
    chi2_corrected = chi2_of(r_hat * mu_theory)
    chi2_paste = chi2_of(mu_paste)
    print(f"      chi2(obs, raw theory at ref)        = {chi2_raw:9.2f}")
    print(f"      chi2(obs, transfer-corrected theory)= {chi2_corrected:9.2f}   (= chi2 of the paste, by construction)")
    print(f"      chi2(obs, pasted signal at ref)     = {chi2_paste:9.2f}   for 42 bands, 0 free parameters")

    print("[4/5] Laplace guides ...", flush=True)
    factor = jnp.asarray(problem.cholesky, dtype=jnp.float64)
    observed = jnp.asarray(observation_stored, dtype=jnp.float64)
    r_hat_j = jnp.asarray(r_hat, dtype=jnp.float64)

    def potential_corrected(u):
        u = jnp.asarray(u, dtype=jnp.float64)
        residual = observed - r_hat_j * problem.predict_u(u)
        w = jax.scipy.linalg.solve_triangular(factor, residual, lower=True)
        return 0.5 * jnp.dot(w, w) + 0.5 * jnp.dot(u, u)

    guide = laplace(potential_corrected, np.zeros(5))
    raw = laplace(problem.potential_u, np.zeros(5))

    theta_guide = theta_from_probit(guide["u_map"], problem.low, problem.high)[0]
    print(f"      corrected MAP  chi2 = {float(2.0*potential_corrected(jnp.asarray(guide['u_map']))) - float(guide['u_map'] @ guide['u_map']):.2f}")
    print(f"      {'param':16s} {'theta MAP':>11s} {'u MAP':>9s} {'u sigma':>9s}")
    for i, name in enumerate(PARAMETER_NAMES):
        print(f"      {name:16s} {theta_guide[i]:11.5f} {guide['u_map'][i]:9.4f} {guide['sigma'][i]:9.4f}")

    print("[5/5] does the guide bracket the generating point?")
    d2_guide = mahalanobis2(u_ref, guide["u_map"], guide["covariance"])
    d2_raw = mahalanobis2(u_ref, raw["u_map"], raw["covariance"])
    pct_guide = 100.0 * chi2_dist.cdf(d2_guide, 5)
    pct_raw = 100.0 * chi2_dist.cdf(d2_raw, 5)
    print(f"      raw theory guide      : Mahalanobis^2 = {d2_raw:7.2f}  -> generating point at the {pct_raw:7.3f} percentile")
    print(f"      transfer-corrected    : Mahalanobis^2 = {d2_guide:7.2f}  -> generating point at the {pct_guide:7.3f} percentile")
    print(f"      gate: Mahalanobis^2 <= {MAX_GENERATING_POINT_MAHALANOBIS2:.2f} (99th pct of 5 dof)")

    passed = d2_guide <= MAX_GENERATING_POINT_MAHALANOBIS2
    payload = {
        "status": "PASS" if passed else "FAIL",
        "gate": {
            "generating_point_mahalanobis2": d2_guide,
            "generating_point_percentile": pct_guide,
            "threshold": MAX_GENERATING_POINT_MAHALANOBIS2,
            "passed": bool(passed),
            "raw_theory_mahalanobis2": d2_raw,
            "raw_theory_percentile": pct_raw,
        },
        "transfer": {
            "r_hat": r_hat.tolist(),
            "r_hat_sha256": sha256_array(r_hat),
            "per_probe": [
                {"spectrum": s, "median": m, "min": lo, "max": hi, "median_abs_snr": snr}
                for s, m, lo, hi, snr in rows
            ],
            "chi2_raw_theory_at_reference": chi2_raw,
            "chi2_transfer_corrected_at_reference": chi2_corrected,
            "chi2_pasted_signal_at_reference": chi2_paste,
            "chi2_reference": {"retained_rank": 42, "n_varied": 5, "expected": 37,
                               "expected_scatter": float(np.sqrt(2.0 * 37))},
        },
        "guide": {
            "coordinates": "standard-normal probit u; theta = low + (high-low) * Phi(u)",
            "u_map": guide["u_map"].tolist(),
            "theta_map": theta_guide.tolist(),
            "covariance": guide["covariance"].tolist(),
            "sigma": guide["sigma"].tolist(),
            "hessian_eigenvalues": guide["hessian_eigenvalues"].tolist(),
            "optimizer_success": guide["optimizer_success"],
            "gradient_norm_at_map": guide["gradient_norm_at_map"],
        },
        "raw_theory_guide": {
            "u_map": raw["u_map"].tolist(),
            "theta_map": theta_from_probit(raw["u_map"], problem.low, problem.high)[0].tolist(),
            "covariance": raw["covariance"].tolist(),
            "sigma": raw["sigma"].tolist(),
        },
        "reference_point": theta_provenance,
        "identity": {
            "grid": list(GRID),
            "contract_sha256": problem.contract.contract_sha256,
            "mu_paste_sha256": sha256_array(mu_paste),
            "mu_theory_sha256": sha256_array(mu_theory),
            "observation_sha256": sha256_array(observation_stored),
            "numerical_sources": numerical_source_manifest(),
            "backend": backend_manifest(),
        },
        "elapsed_seconds": time.time() - started,
    }
    out = args.output_dir / "transfer_and_guide.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    np.savez(args.output_dir / "transfer_and_guide.npz",
             r_hat=r_hat, mu_theory_reference=mu_theory, mu_paste_reference=mu_paste,
             u_reference=u_ref, guide_u_map=guide["u_map"], guide_covariance=guide["covariance"],
             raw_u_map=raw["u_map"], raw_covariance=raw["covariance"])
    print(f"\nstatus {payload['status']}   wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
