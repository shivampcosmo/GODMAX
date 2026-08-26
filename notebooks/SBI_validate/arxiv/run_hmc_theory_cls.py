"""Run NumPyro NUTS for the analytical Cl SBI-validation likelihood."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Sequence

import numpy as np

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

numpyro.enable_x64()

from theory_sbi_utils import (
    DEFAULT_FIDUCIAL_PATH,
    ParameterSpec,
    THEORY_SBI_DIR,
    ensure_default_fiducial_product,
    fiducial_theta,
    make_inference_theory_vector_function,
    metadata_json,
    parse_param_specs,
    parse_probe_list,
    prior_bounds,
    selected_product_arrays,
    validate_theory_vector,
)


def run_hmc(
    fiducial_path: pathlib.Path,
    output_dir: pathlib.Path,
    probes: tuple[str, ...],
    param_specs,
    ell_min: float | None,
    ell_max: float | None,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    max_tree_depth: int,
    dense_mass: bool,
    target_accept_prob: float,
    seed: int,
    chain_method: str,
    jit_compile: bool,
    fiducial_offset: bool,
    theory_backend: str,
    fixed_param_specs: Sequence[ParameterSpec] | None = None,
    theory_mode: str = "map_matched_resolved",
) -> dict:
    """Run and save a fixed-covariance Gaussian NUTS chain."""

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
        fixed_param_specs=fixed_param_specs,
        theory_mode=theory_mode,
    )
    validation = validate_theory_vector(vector_fn, selected, param_specs)

    obs = jnp.asarray(selected["data_vector"])
    chol = jnp.asarray(selected["chol"])
    low, high = prior_bounds(param_specs)
    low_j = jnp.asarray(low)
    high_j = jnp.asarray(high)
    theta0 = fiducial_theta(param_specs)
    init_values = {spec.name: float(spec.fiducial) for spec in param_specs}

    def model():
        values = []
        for ip, spec in enumerate(param_specs):
            values.append(
                numpyro.sample(
                    spec.name,
                    dist.Uniform(low_j[ip], high_j[ip]),
                )
            )
        theta = jnp.stack(values)
        mu = vector_fn(theta)
        resid = obs - mu
        white = jsl.solve_triangular(chol, resid, lower=True)
        numpyro.factor("fixed_cov_gaussian_loglike", -0.5 * jnp.dot(white, white))

    numpyro.set_host_device_count(max(num_chains, 1))
    kernel = NUTS(
        model,
        dense_mass=dense_mass,
        init_strategy=init_to_value(values=init_values),
        max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=True,
    )

    t0 = time.time()
    mcmc.run(
        jax.random.PRNGKey(seed),
        extra_fields=("potential_energy", "diverging", "accept_prob", "num_steps"),
    )
    runtime_sec = time.time() - t0

    samples_chain = mcmc.get_samples(group_by_chain=True)
    samples_flat = mcmc.get_samples(group_by_chain=False)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    divergences = np.asarray(extra["diverging"], dtype=bool)
    accept_prob = np.asarray(extra["accept_prob"], dtype=float)
    num_steps = np.asarray(extra["num_steps"], dtype=int)
    tree_depth = np.floor(np.log2(np.maximum(num_steps, 1))).astype(int) + 1
    tree_depth_saturated = tree_depth >= int(max_tree_depth)
    final_step_size = np.asarray(mcmc.last_state.adapt_state.step_size, dtype=float)
    min_potential_energy_per_chain = np.min(
        np.asarray(extra["potential_energy"], dtype=float),
        axis=-1,
    )

    sampler_diagnostics = {
        "divergence_count": int(np.sum(divergences)),
        "mean_accept_prob": float(np.mean(accept_prob)),
        "divergence_count_per_chain": np.sum(divergences, axis=-1).astype(int).tolist(),
        "tree_depth_saturation_count": int(np.sum(tree_depth_saturated)),
        "tree_depth_saturation_fraction": float(np.mean(tree_depth_saturated)),
        "tree_depth_saturation_fraction_per_chain": np.mean(
            tree_depth_saturated,
            axis=-1,
        ).tolist(),
        "max_observed_tree_depth": int(np.max(tree_depth)),
        "final_step_size_per_chain": np.atleast_1d(final_step_size).tolist(),
        "min_potential_energy_per_chain": np.atleast_1d(
            min_potential_energy_per_chain
        ).tolist(),
    }

    np_payload = {
        "theta_fiducial": theta0,
        "prior_min": low,
        "prior_max": high,
        "data_vector": np.asarray(selected["data_vector"]),
        "cov": np.asarray(selected["cov"]),
        "chol": np.asarray(selected["chol"]),
        "selection_indices": np.asarray(selection.indices),
        "selection_ell_indices": np.asarray(selection.ell_indices),
        "metadata_json": np.asarray(
            metadata_json(
                param_specs,
                selection,
                {
                    "fiducial_path": str(fiducial_path),
                    "runtime_sec": runtime_sec,
                    "num_warmup": num_warmup,
                    "num_samples": num_samples,
                    "num_chains": num_chains,
                    "max_tree_depth": max_tree_depth,
                    "dense_mass": dense_mass,
                    "target_accept_prob": target_accept_prob,
                    "chain_method": chain_method,
                    "initialization_strategy": "init_to_value_all_chains_at_fiducial",
                    "seed": seed,
                    "fiducial_offset_correction": fiducial_offset,
                    "theory_backend": theory_backend,
                    "theory_mode": theory_mode,
                    "validation": validation,
                    "sampler_diagnostics": sampler_diagnostics,
                },
                fixed_param_specs=fixed_param_specs,
            )
        ),
    }
    for spec in param_specs:
        np_payload[f"samples_{spec.name}"] = np.asarray(samples_flat[spec.name])
        np_payload[f"samples_chain_{spec.name}"] = np.asarray(samples_chain[spec.name])
    for key, value in extra.items():
        np_payload[f"extra_{key}"] = np.asarray(value)
    for key, value in theory_info.items():
        np_payload[f"theory_{key}"] = np.asarray(value)

    samples_path = output_dir / "hmc_samples.npz"
    np.savez_compressed(samples_path, **np_payload)

    diagnostics = {
        "runtime_sec": runtime_sec,
        "validation": validation,
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "max_tree_depth": max_tree_depth,
        "dense_mass": dense_mass,
        "target_accept_prob": target_accept_prob,
        "chain_method": chain_method,
        "initialization_strategy": "init_to_value_all_chains_at_fiducial",
        "seed": seed,
        "fiducial_offset_correction": fiducial_offset,
        "theory_backend": theory_backend,
        "theory_mode": theory_mode,
        "fiducial_path": str(fiducial_path),
        **sampler_diagnostics,
        "fixed_parameter_specs": [
            {
                "name": spec.name,
                "label": spec.label,
                "fiducial": spec.fiducial,
                "prior_min": spec.prior_min,
                "prior_max": spec.prior_max,
                "target": spec.target,
            }
            for spec in (fixed_param_specs or ())
        ],
    }
    try:
        import arviz as az

        # Build diagnostics from the already saved chain arrays.  ``az.from_numpyro``
        # treats ``numpyro.factor`` as an observed site and replays this expensive
        # full-theory likelihood at every posterior draw to populate log_likelihood,
        # which is unnecessary for R-hat/ESS diagnostics.
        idata = az.from_dict(
            posterior={
                spec.name: np.asarray(samples_chain[spec.name]) for spec in param_specs
            }
        )
        summary = az.summary(
            idata,
            var_names=[spec.name for spec in param_specs],
            round_to="none",
        )
        diagnostics["arviz_summary"] = json.loads(summary.to_json())
        diagnostics["max_rhat"] = float(summary["r_hat"].max())
        diagnostics["max_rhat_parameter"] = str(summary["r_hat"].idxmax())
        diagnostics["min_ess_bulk"] = float(summary["ess_bulk"].min())
        diagnostics["min_ess_bulk_parameter"] = str(summary["ess_bulk"].idxmin())
        diagnostics["min_ess_tail"] = float(summary["ess_tail"].min())
        diagnostics["min_ess_tail_parameter"] = str(summary["ess_tail"].idxmin())
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        diagnostics["arviz_error"] = repr(exc)

    with (output_dir / "hmc_diagnostics.json").open("w") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)

    return {
        "samples_path": samples_path,
        "diagnostics_path": output_dir / "hmc_diagnostics.json",
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiducial-path", default=str(DEFAULT_FIDUCIAL_PATH))
    parser.add_argument("--output-dir", default=str(THEORY_SBI_DIR / "joint_gg_gy_gtau_gkappa"))
    parser.add_argument("--probes", default="gg,gy,gtau,gkappa")
    parser.add_argument("--ell-min", type=float, default=None)
    parser.add_argument("--ell-max", type=float, default=None)
    parser.add_argument("--param-spec", action="append", default=[])
    parser.add_argument("--force-fiducial", action="store_true")
    parser.add_argument("--num-warmup", type=int, default=2000)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--max-tree-depth", type=int, default=6)
    parser.add_argument("--dense-mass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument("--chain-method", choices=("parallel", "sequential", "vectorized"),
                        default="vectorized")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--no-fiducial-offset", action="store_true",
                        help="Disable the constant numerical offset that aligns the JAX evaluator to the saved fiducial product.")
    parser.add_argument("--theory-backend", choices=("linearized", "direct"),
                        default="linearized")
    parser.add_argument("--theory-mode", choices=("full", "map_matched_resolved"),
                        default="map_matched_resolved")
    args = parser.parse_args()

    param_specs = parse_param_specs(args.param_spec)
    fiducial_path = ensure_default_fiducial_product(
        args.fiducial_path,
        param_specs=param_specs,
        force=args.force_fiducial,
        theory_mode=args.theory_mode,
    )
    result = run_hmc(
        fiducial_path=pathlib.Path(fiducial_path),
        output_dir=pathlib.Path(args.output_dir),
        probes=parse_probe_list(args.probes),
        param_specs=param_specs,
        ell_min=args.ell_min,
        ell_max=args.ell_max,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        max_tree_depth=args.max_tree_depth,
        dense_mass=args.dense_mass,
        target_accept_prob=args.target_accept_prob,
        seed=args.seed,
        chain_method=args.chain_method,
        jit_compile=not args.no_jit,
        fiducial_offset=not args.no_fiducial_offset,
        theory_backend=args.theory_backend,
        theory_mode=args.theory_mode,
    )
    print(f"Saved HMC samples to {result['samples_path']}")
    print(f"Saved HMC diagnostics to {result['diagnostics_path']}")


if __name__ == "__main__":
    main()
