"""Run score-compressed Gaussian/Student-t likelihood tests for analytical Cls.

This is the low-simulation alternative to neural SNPE for the analytical
validation problem.  The full datavector is compressed to one score summary per
sampled parameter.  Simulations are then used only to estimate the covariance of
that compressed summary; the posterior is evaluated on a dense two-dimensional
grid with the same rectangular priors used by the HMC/SBI runs.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from theory_sbi_utils import (
    DEFAULT_FIDUCIAL_PATH,
    THEORY_SBI_DIR,
    ensure_default_fiducial_product,
    fiducial_theta,
    make_inference_theory_vector_function,
    parse_param_specs,
    parse_probe_list,
    prior_bounds,
    selected_product_arrays,
    validate_theory_vector,
)


def _stable_inverse_cov(cov: np.ndarray) -> tuple[np.ndarray, float]:
    cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    jitter = 0.0
    try:
        return np.linalg.inv(cov), jitter
    except np.linalg.LinAlgError:
        eig_min = float(np.min(np.linalg.eigvalsh(cov)))
        floor = 1.0e-10 * max(float(np.median(np.diag(cov))), 1.0e-300)
        jitter = max(floor, -eig_min + floor)
        return np.linalg.inv(cov + np.eye(cov.shape[0]) * jitter), jitter


def _grid_posterior(
    prior_min: np.ndarray,
    prior_max: np.ndarray,
    theta_fiducial: np.ndarray,
    fisher_transform: np.ndarray,
    x_obs_u: np.ndarray,
    cov_u: np.ndarray,
    likelihood: str,
    nsim: int,
    ngrid: int,
) -> dict[str, np.ndarray | float]:
    theta0_grid = np.linspace(prior_min[0], prior_max[0], ngrid)
    theta1_grid = np.linspace(prior_min[1], prior_max[1], ngrid)
    grid0, grid1 = np.meshgrid(theta0_grid, theta1_grid, indexing="ij")
    theta_stack = np.stack([grid0, grid1], axis=-1)
    model_u = (theta_stack - theta_fiducial[None, None, :]) @ fisher_transform
    resid = x_obs_u[None, None, :] - model_u
    inv_cov_u, jitter = _stable_inverse_cov(cov_u)
    chi2 = np.einsum("...i,ij,...j->...", resid, inv_cov_u, resid)
    sign, logdet = np.linalg.slogdet(cov_u + np.eye(cov_u.shape[0]) * jitter)
    if sign <= 0:
        raise RuntimeError("Compressed covariance is not positive definite")
    if likelihood == "gaussian":
        loglike = -0.5 * (chi2 + logdet)
    elif likelihood == "student_t":
        if nsim <= 1:
            raise ValueError("Student-t likelihood requires nsim > 1")
        loglike = -0.5 * logdet - 0.5 * nsim * np.log1p(chi2 / (nsim - 1.0))
    else:
        raise ValueError("likelihood must be 'gaussian' or 'student_t'")
    logw = loglike - np.max(loglike)
    weight = np.exp(logw)
    norm = np.sum(weight)
    mean = np.array([
        np.sum(weight * grid0) / norm,
        np.sum(weight * grid1) / norm,
    ])
    var0 = np.sum(weight * (grid0 - mean[0]) ** 2) / norm
    var1 = np.sum(weight * (grid1 - mean[1]) ** 2) / norm
    cov01 = np.sum(weight * (grid0 - mean[0]) * (grid1 - mean[1])) / norm
    cov = np.array([[var0, cov01], [cov01, var1]])
    return {
        "theta0_grid": theta0_grid,
        "theta1_grid": theta1_grid,
        "log_posterior_grid": logw - np.log(norm),
        "posterior_weight_grid": weight / norm,
        "mean": mean,
        "cov": cov,
        "jitter": jitter,
    }


def _sample_grid(
    theta0_grid: np.ndarray,
    theta1_grid: np.ndarray,
    weight_grid: np.ndarray,
    nsamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    flat_weight = np.ravel(weight_grid)
    flat_weight = flat_weight / np.sum(flat_weight)
    choice = rng.choice(flat_weight.size, size=nsamples, replace=True, p=flat_weight)
    i0, i1 = np.unravel_index(choice, weight_grid.shape)
    return np.column_stack([theta0_grid[i0], theta1_grid[i1]])


def run_compressed_likelihood(
    fiducial_path: pathlib.Path,
    output_dir: pathlib.Path,
    probes: tuple[str, ...],
    param_specs,
    ell_min: float | None,
    ell_max: float | None,
    nsim: int,
    likelihood: str,
    ngrid: int,
    posterior_samples: int,
    seed: int,
    theory_backend: str,
    jit_compile: bool,
    fiducial_offset: bool,
    use_true_covariance: bool,
) -> dict[str, pathlib.Path]:
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = selected_product_arrays(fiducial_path, probes=probes, ell_min=ell_min, ell_max=ell_max)
    selection = selected["selection"]
    vector_fn, theory_info = make_inference_theory_vector_function(
        param_specs,
        selection,
        fiducial_vector=selected["data_vector"],
        backend=theory_backend,
        fiducial_offset=fiducial_offset,
        jit_compile=jit_compile,
    )
    validation = validate_theory_vector(vector_fn, selected, param_specs)
    theta0 = fiducial_theta(param_specs)
    prior_min, prior_max = prior_bounds(param_specs)
    chol = np.asarray(selected["chol"], dtype=float)
    if "jacobian" in theory_info:
        jac = np.asarray(theory_info["jacobian"], dtype=float)
    else:
        jac = np.asarray(jax.jacfwd(vector_fn)(jnp.asarray(theta0)), dtype=float)
    jac_white = np.linalg.solve(chol, jac)
    fisher = jac_white.T @ jac_white
    fisher = 0.5 * (fisher + fisher.T)
    fisher_inv = np.linalg.pinv(fisher)
    compression_matrix = fisher_inv @ jac_white.T
    fisher_transform = np.linalg.cholesky(fisher)

    rng = np.random.default_rng(seed)
    if use_true_covariance:
        summary_theta_cov = fisher_inv
        summary_theta_samples = rng.multivariate_normal(theta0, summary_theta_cov, size=nsim)
    else:
        eps = rng.normal(size=(nsim, len(selected["data_vector"])))
        summary_theta_samples = theta0[None, :] + eps @ compression_matrix.T
        summary_theta_cov = np.cov(summary_theta_samples.T, ddof=1)
    summary_u_samples = (summary_theta_samples - theta0[None, :]) @ fisher_transform
    summary_u_cov = np.cov(summary_u_samples.T, ddof=1)

    x_obs_u = np.zeros(len(theta0), dtype=float)
    grid = _grid_posterior(
        prior_min=prior_min,
        prior_max=prior_max,
        theta_fiducial=theta0,
        fisher_transform=fisher_transform,
        x_obs_u=x_obs_u,
        cov_u=summary_u_cov,
        likelihood=likelihood,
        nsim=nsim,
        ngrid=ngrid,
    )
    exact_grid = _grid_posterior(
        prior_min=prior_min,
        prior_max=prior_max,
        theta_fiducial=theta0,
        fisher_transform=fisher_transform,
        x_obs_u=x_obs_u,
        cov_u=np.eye(len(theta0)),
        likelihood="gaussian",
        nsim=nsim,
        ngrid=ngrid,
    )
    samples = _sample_grid(
        np.asarray(grid["theta0_grid"]),
        np.asarray(grid["theta1_grid"]),
        np.asarray(grid["posterior_weight_grid"]),
        posterior_samples,
        rng,
    )
    samples_exact = _sample_grid(
        np.asarray(exact_grid["theta0_grid"]),
        np.asarray(exact_grid["theta1_grid"]),
        np.asarray(exact_grid["posterior_weight_grid"]),
        posterior_samples,
        rng,
    )

    samples_path = output_dir / "compressed_likelihood_samples.npz"
    np.savez_compressed(
        samples_path,
        samples=samples,
        samples_exact=samples_exact,
        theta_fiducial=theta0,
        prior_min=prior_min,
        prior_max=prior_max,
        data_vector=np.asarray(selected["data_vector"], dtype=float),
        cov=np.asarray(selected["cov"], dtype=float),
        selection_indices=np.asarray(selection.indices),
        selection_ell_indices=np.asarray(selection.ell_indices),
        fisher=fisher,
        fisher_inv=fisher_inv,
        fisher_transform=fisher_transform,
        compression_matrix=compression_matrix,
        summary_theta_samples=summary_theta_samples,
        summary_theta_cov=summary_theta_cov,
        summary_u_samples=summary_u_samples,
        summary_u_cov=summary_u_cov,
        theta0_grid=np.asarray(grid["theta0_grid"]),
        theta1_grid=np.asarray(grid["theta1_grid"]),
        posterior_weight_grid=np.asarray(grid["posterior_weight_grid"]),
        log_posterior_grid=np.asarray(grid["log_posterior_grid"]),
        exact_posterior_weight_grid=np.asarray(exact_grid["posterior_weight_grid"]),
        posterior_mean=np.asarray(grid["mean"]),
        posterior_cov=np.asarray(grid["cov"]),
        exact_posterior_mean=np.asarray(exact_grid["mean"]),
        exact_posterior_cov=np.asarray(exact_grid["cov"]),
    )
    diagnostics = {
        "runtime_sec": time.time() - t0,
        "fiducial_path": str(fiducial_path),
        "samples_path": str(samples_path),
        "probes": list(probes),
        "ell_min": ell_min,
        "ell_max": ell_max,
        "nsim": nsim,
        "likelihood": likelihood,
        "ngrid": ngrid,
        "posterior_samples": posterior_samples,
        "seed": seed,
        "theory_backend": theory_backend,
        "fiducial_offset_correction": fiducial_offset,
        "use_true_covariance": use_true_covariance,
        "validation": validation,
        "posterior_mean": np.asarray(grid["mean"]).tolist(),
        "posterior_std": np.sqrt(np.diag(np.asarray(grid["cov"]))).tolist(),
        "exact_posterior_mean": np.asarray(exact_grid["mean"]).tolist(),
        "exact_posterior_std": np.sqrt(np.diag(np.asarray(exact_grid["cov"]))).tolist(),
        "summary_u_cov": summary_u_cov.tolist(),
        "summary_theta_cov": summary_theta_cov.tolist(),
    }
    diagnostics_path = output_dir / "compressed_likelihood_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    return {"samples_path": samples_path, "diagnostics_path": diagnostics_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiducial-path", default=str(DEFAULT_FIDUCIAL_PATH))
    parser.add_argument("--output-dir", default=str(THEORY_SBI_DIR / "compressed_likelihood"))
    parser.add_argument("--probes", default="gg,gy,gtau,gkappa")
    parser.add_argument("--ell-min", type=float, default=None)
    parser.add_argument("--ell-max", type=float, default=None)
    parser.add_argument("--param-spec", action="append", default=[])
    parser.add_argument("--force-fiducial", action="store_true")
    parser.add_argument("--nsim", type=int, default=256)
    parser.add_argument("--likelihood", choices=("gaussian", "student_t"), default="student_t")
    parser.add_argument("--ngrid", type=int, default=900)
    parser.add_argument("--posterior-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--theory-backend", choices=("linearized", "direct"), default="linearized")
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--no-fiducial-offset", action="store_true")
    parser.add_argument("--use-true-covariance", action="store_true",
                        help="Use the analytical score-summary covariance instead of estimating it from simulations.")
    args = parser.parse_args()

    param_specs = parse_param_specs(args.param_spec)
    fiducial_path = ensure_default_fiducial_product(
        args.fiducial_path,
        param_specs=param_specs,
        force=args.force_fiducial,
    )
    result = run_compressed_likelihood(
        fiducial_path=pathlib.Path(fiducial_path),
        output_dir=pathlib.Path(args.output_dir),
        probes=parse_probe_list(args.probes),
        param_specs=param_specs,
        ell_min=args.ell_min,
        ell_max=args.ell_max,
        nsim=args.nsim,
        likelihood=args.likelihood,
        ngrid=args.ngrid,
        posterior_samples=args.posterior_samples,
        seed=args.seed,
        theory_backend=args.theory_backend,
        jit_compile=not args.no_jit,
        fiducial_offset=not args.no_fiducial_offset,
        use_true_covariance=args.use_true_covariance,
    )
    print(f"Saved compressed-likelihood samples to {result['samples_path']}")
    print(f"Saved compressed-likelihood diagnostics to {result['diagnostics_path']}")


if __name__ == "__main__":
    main()
