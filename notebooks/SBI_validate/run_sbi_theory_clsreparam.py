"""Run active-learning SNPE for the analytical Cl SBI-validation likelihood."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from torch.distributions import Distribution, constraints

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax

from theory_sbi_utils import (
    DEFAULT_FIDUCIAL_PATH,
    THEORY_SBI_DIR,
    ensure_default_fiducial_product,
    fiducial_theta,
    make_inference_theory_vector_function,
    metadata_json,
    parse_param_specs,
    parse_probe_list,
    prior_bounds,
    save_pickle,
    selected_product_arrays,
    validate_theory_vector,
    phi_parameter_specs,
    phi_theta_transform,
)


class FisherWhitenedBoxUniform(Distribution):
    """BoxUniform prior transformed by u = L.T(theta - theta_fid).

    Rows are represented as ``u = (theta - theta_fid) @ L`` where
    ``fisher = L @ L.T``.  The support is the original rectangular parameter
    box mapped into this linear basis.
    """

    arg_constraints = {}
    support = constraints.real_vector
    has_rsample = False

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        theta_fiducial: np.ndarray,
        transform: np.ndarray,
        device: str = "cpu",
    ) -> None:
        event_shape = torch.Size([len(low)])
        super().__init__(batch_shape=torch.Size(), event_shape=event_shape, validate_args=False)
        self.low = torch.as_tensor(low, dtype=torch.float32, device=device)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=device)
        self.theta_fiducial = torch.as_tensor(theta_fiducial, dtype=torch.float32, device=device)
        self.transform = torch.as_tensor(transform, dtype=torch.float32, device=device)
        self.inv_transform = torch.linalg.inv(self.transform)
        log_volume_theta = torch.sum(torch.log(self.high - self.low))
        log_abs_det = torch.linalg.slogdet(self.transform)[1]
        self._log_prob_inside = -log_volume_theta - log_abs_det

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self.event_shape
        theta = self.low + torch.rand(
            shape,
            dtype=self.low.dtype,
            device=self.low.device,
        ) * (self.high - self.low)
        return (theta - self.theta_fiducial) @ self.transform

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        theta = self.theta_fiducial + value @ self.inv_transform
        inside = torch.logical_and(theta >= self.low, theta <= self.high).all(dim=-1)
        logp = torch.full(
            value.shape[:-1],
            float(self._log_prob_inside.detach().cpu()),
            dtype=value.dtype,
            device=value.device,
        )
        return torch.where(inside, logp, torch.full_like(logp, -torch.inf))


FisherWhitenedBoxUniform.__module__ = "run_sbi_theory_cls"
sys.modules.setdefault("run_sbi_theory_cls", sys.modules[__name__])


def _parse_rounds(text: str) -> list[int]:
    rounds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not rounds:
        raise ValueError("At least one SBI round must be requested")
    return rounds


def run_sbi(
    fiducial_path: pathlib.Path,
    output_dir: pathlib.Path,
    probes: tuple[str, ...],
    param_specs,
    ell_min: float | None,
    ell_max: float | None,
    simulations_per_round: list[int],
    posterior_samples: int,
    seed: int,
    hidden_features: int,
    num_transforms: int,
    density_estimator_model: str,
    num_components: int,
    num_bins: int,
    training_batch_size: int,
    max_num_epochs: int,
    jit_compile: bool,
    fiducial_offset: bool,
    theory_backend: str,
    summary_compression: str,
    device: str,
    discard_prior_samples: bool,
    retrain_from_scratch: bool,
    force_first_round_loss: bool,
    num_atoms: int,
    validation_fraction: float,
    learning_rate: float,
    parameter_transform: str,
    theory_param_specs=None,
    theta_transform=None,
) -> dict:
    """Run sequential SNPE using noisy theory-Cl summaries.

    ``param_specs`` defines the prior actually sampled by SNPE (the
    "inference basis", e.g. (theta_ej_0, phi)). ``theory_param_specs`` /
    ``theta_transform`` exist because the GODMAX simulator underlying
    ``vector_fn`` reads named physical keys (e.g. "theta_ej_0",
    "nu_theta_ej_M") out of its parameter dict -- it has no notion of any
    other sampled quantity, e.g. "phi". If ``param_specs`` samples a basis
    that isn't literally the physical simulator parameters, ``theta``
    must be converted to the physical basis via ``theta_transform`` before
    it is handed to ``vector_fn``/used to build a Jacobian; otherwise the
    sampled value is written directly into the physical slot as if it
    were already physical, which is either an error or silently zeroes
    out that parameter's effect on the model.

    ``theta_transform`` must be written with jnp ops and must act on the
    *last* axis of its input, so it is correct both for a single vector
    ``theta`` of shape ``(n_params,)`` (used by HMC / jax.grad) and for a
    batch ``theta`` of shape ``(n_samples, n_params)`` (used by the SBI
    batch-simulation path below), e.g. via ``theta[..., i]`` indexing.

    This is a deterministic change of variables performed purely for the
    simulator call; it does not alter the prior, the SNPE training
    targets, or any saved theta values, which remain in the
    ``param_specs`` (inference) basis throughout.
    """

    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    selected = selected_product_arrays(fiducial_path, probes=probes, ell_min=ell_min, ell_max=ell_max)
    selection = selected["selection"]

    theory_specs = theory_param_specs if theory_param_specs is not None else param_specs

    vector_fn_theory, theory_info = make_inference_theory_vector_function(
        theory_specs,
        selection,
        fiducial_vector=selected["data_vector"],
        backend=theory_backend,
        fiducial_offset=fiducial_offset,
        jit_compile=jit_compile,
    )
    # Pure physics sanity check against the saved fiducial product: this must
    # always be done in the physical basis, independent of theta_transform.
    validation = validate_theory_vector(vector_fn_theory, selected, theory_specs)

    if theta_transform is not None:
        def vector_fn(theta):
            return vector_fn_theory(theta_transform(theta))
    else:
        vector_fn = vector_fn_theory

    low, high = prior_bounds(param_specs)
    theta_prior = BoxUniform(
        low=torch.as_tensor(low, dtype=torch.float32, device=torch_device),
        high=torch.as_tensor(high, dtype=torch.float32, device=torch_device),
    )
    obs_np = np.asarray(selected["data_vector"], dtype=float)
    chol_np = np.asarray(selected["chol"], dtype=float)
    theta0_np = fiducial_theta(param_specs)  # inference-basis fiducial (e.g. phi)
    theta0_physical_np = fiducial_theta(theory_specs)  # physical-basis fiducial (e.g. nu)
    compression_info = {"summary_compression": summary_compression}

    # theory_info["jacobian"] (only present for backend="linearized") is
    # d(vector_fn_theory)/d(theta_physical) at theta0_physical. It is exact
    # and reusable for the fast batch-simulation shortcut below regardless
    # of theta_transform, AS LONG AS theta is first converted to the
    # physical basis -- the linearized model is affine in theta_physical by
    # construction, so no additional approximation is introduced by a
    # nonlinear theta_transform.
    jac_for_batch = np.asarray(theory_info["jacobian"], dtype=float) if "jacobian" in theory_info else None

    if summary_compression == "none":
        x_obs_np = np.zeros(len(obs_np), dtype=float)
        compress = lambda x_white: x_white
    elif summary_compression == "score":
        # For score compression we need the Jacobian *in the inference
        # basis* (d(vector_fn)/d(theta_sampled) at theta0_np), since the
        # compression map is centered at, and expressed in units of,
        # theta0_np. When theta_transform is identity this equals
        # theory_info["jacobian"] exactly, so the shortcut is safe only in
        # that case; otherwise it must be recomputed via the chain rule
        # through the composed vector_fn.
        if "jacobian" in theory_info and theta_transform is None:
            jac = np.asarray(theory_info["jacobian"], dtype=float)
        else:
            jac = np.asarray(jax.jacfwd(vector_fn)(jnp.asarray(theta0_np)), dtype=float)
        jac_white = np.linalg.solve(chol_np, jac)
        fisher = jac_white.T @ jac_white
        fisher_inv = np.linalg.pinv(fisher)
        compression_matrix = fisher_inv @ jac_white.T
        x_obs_np = theta0_np.copy()

        def compress(x_white):
            return theta0_np + compression_matrix @ x_white

        compression_info.update({
            "fisher": fisher,
            "fisher_inv": fisher_inv,
            "compression_matrix": compression_matrix,
        })
    else:
        raise ValueError("summary_compression must be 'score' or 'none'")

    transform_info = {
        "parameter_transform": parameter_transform,
        "parameter_transform_matrix": None,
        "parameter_transform_inverse": None,
    }
    if parameter_transform == "none":
        prior = theta_prior

        def inference_to_theta_np(samples: np.ndarray) -> np.ndarray:
            return np.asarray(samples, dtype=float)

        def theta_to_inference_np(samples: np.ndarray) -> np.ndarray:
            return np.asarray(samples, dtype=float)

        def transform_summary_np(summary_theta_units: np.ndarray) -> np.ndarray:
            return np.asarray(summary_theta_units, dtype=float)

    elif parameter_transform == "fisher":
        if summary_compression != "score":
            raise ValueError("parameter_transform='fisher' requires summary_compression='score'")
        fisher = np.asarray(compression_info["fisher"], dtype=float)
        fisher = 0.5 * (fisher + fisher.T)
        transform = np.linalg.cholesky(fisher)
        inv_transform = np.linalg.inv(transform)
        prior = FisherWhitenedBoxUniform(
            low=low,
            high=high,
            theta_fiducial=theta0_np,
            transform=transform,
            device=device,
        )
        x_obs_np = np.zeros_like(theta0_np)

        def inference_to_theta_np(samples: np.ndarray) -> np.ndarray:
            return theta0_np[None, :] + np.asarray(samples, dtype=float) @ inv_transform

        def theta_to_inference_np(samples: np.ndarray) -> np.ndarray:
            return (np.asarray(samples, dtype=float) - theta0_np[None, :]) @ transform

        def transform_summary_np(summary_theta_units: np.ndarray) -> np.ndarray:
            return (np.asarray(summary_theta_units, dtype=float) - theta0_np[None, :]) @ transform

        transform_info.update({
            "parameter_transform_matrix": transform,
            "parameter_transform_inverse": inv_transform,
        })
    else:
        raise ValueError("parameter_transform must be 'none' or 'fisher'")

    x_obs = torch.as_tensor(x_obs_np, dtype=torch.float32, device=torch_device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    def simulate_whitened(theta_t: torch.Tensor, round_seed: int) -> torch.Tensor:
        theta_np = inference_to_theta_np(theta_t.detach().cpu().numpy())
        eps_rng = np.random.default_rng(round_seed)
        if jac_for_batch is not None:
            if theta_transform is not None:
                # Convert the whole batch from the inference/sampled basis
                # (e.g. phi) to the physical basis (e.g. nu) in one call --
                # exact, not an approximation, since the linearized model is
                # affine in the physical parameters by construction.
                theta_physical_np = np.asarray(
                    theta_transform(jnp.asarray(theta_np)), dtype=float
                )
            else:
                theta_physical_np = theta_np
            mu_minus_obs = (theta_physical_np - theta0_physical_np) @ jac_for_batch.T
            mean_white = np.linalg.solve(chol_np, mu_minus_obs.T).T
            x_white = mean_white + eps_rng.normal(size=mean_white.shape)
            if summary_compression == "score":
                x_theta_units = theta0_np[None, :] + x_white @ compression_matrix.T
                x_out = transform_summary_np(x_theta_units)
            else:
                x_out = x_white
            return torch.as_tensor(x_out, dtype=torch.float32, device=torch_device)

        x_rows = []
        for row in theta_np:
            mu = np.asarray(vector_fn(jnp.asarray(row, dtype=jnp.float64)), dtype=float)
            mean_white = np.linalg.solve(chol_np, mu - obs_np)
            x_rows.append(compress(mean_white + eps_rng.normal(size=mean_white.shape)))
        return torch.as_tensor(
            transform_summary_np(np.asarray(x_rows)),
            dtype=torch.float32,
            device=torch_device,
        )

    density_estimator = posterior_nn(
        model=density_estimator_model,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_components=num_components,
        num_bins=num_bins,
    )
    inference = SNPE(
        prior=prior,
        density_estimator=density_estimator,
        device=device,
        show_progress_bars=True,
    )
    proposal = prior
    posterior = None
    round_paths: list[str] = []
    t0 = time.time()

    def sample_from_proposal_inside_prior(proposal, nsim: int) -> torch.Tensor:
        """Sample from a proposal and reject points outside the transformed prior."""

        chunks = []
        n_kept = 0
        n_attempted = 0
        max_attempted = max(100 * nsim, nsim + 10000)
        while n_kept < nsim and n_attempted < max_attempted:
            ndraw = min(max(2 * (nsim - n_kept), 256), max_attempted - n_attempted)
            draw = proposal.sample((ndraw,), x=x_obs, show_progress_bars=True)
            n_attempted += int(draw.shape[0])
            finite = torch.isfinite(prior.log_prob(draw.detach()))
            if finite.any():
                accepted = draw[finite]
                chunks.append(accepted)
                n_kept += int(accepted.shape[0])
        if n_kept < nsim:
            raise RuntimeError(
                f"Only drew {n_kept} proposal samples inside the prior after {n_attempted} attempts"
            )
        return torch.cat(chunks, dim=0)[:nsim]

    def sample_final_inside_prior(posterior, nsim: int) -> tuple[torch.Tensor, int]:
        """Draw final posterior samples and keep only physical-prior samples."""

        chunks = []
        n_kept = 0
        n_attempted = 0
        max_attempted = max(50 * nsim, nsim + 10000)
        while n_kept < nsim and n_attempted < max_attempted:
            ndraw = min(max(2 * (nsim - n_kept), 1024), max_attempted - n_attempted)
            draw = posterior.sample((ndraw,), x=x_obs, show_progress_bars=True)
            n_attempted += int(draw.shape[0])
            finite = torch.isfinite(prior.log_prob(draw.detach()))
            if finite.any():
                accepted = draw[finite]
                chunks.append(accepted)
                n_kept += int(accepted.shape[0])
        if n_kept < nsim:
            raise RuntimeError(
                f"Only drew {n_kept} posterior samples inside the prior after {n_attempted} attempts"
            )
        return torch.cat(chunks, dim=0)[:nsim], n_attempted

    for iround, nsim in enumerate(simulations_per_round):
        if iround == 0:
            theta = prior.sample((nsim,))
            proposal_for_append = None
        else:
            theta = sample_from_proposal_inside_prior(posterior, nsim)
            proposal_for_append = proposal

        x = simulate_whitened(theta, round_seed=seed + 1000 * (iround + 1))
        theta_physical = inference_to_theta_np(theta.detach().cpu().numpy())
        npz_path = output_dir / f"sbi_round{iround}_simulations.npz"
        np.savez_compressed(
            npz_path,
            theta=theta_physical,
            theta_inference=theta.detach().cpu().numpy(),
            x=x.detach().cpu().numpy(),
            round=np.asarray(iround),
            nsim=np.asarray(nsim),
        )
        round_paths.append(str(npz_path))

        density_estimator = inference.append_simulations(
            theta,
            x,
            proposal=proposal_for_append,
        ).train(
            num_atoms=num_atoms,
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            max_num_epochs=max_num_epochs,
            discard_prior_samples=discard_prior_samples,
            retrain_from_scratch=retrain_from_scratch,
            force_first_round_loss=force_first_round_loss,
            show_train_summary=True,
        )
        posterior = inference.build_posterior(density_estimator)
        posterior.set_default_x(x_obs)
        proposal = posterior
        save_pickle(output_dir / f"sbi_posterior_round{iround}.pkl", posterior)

    post_samples_inference_t, posterior_sample_attempts = sample_final_inside_prior(
        posterior,
        posterior_samples,
    )
    post_samples_inference = post_samples_inference_t.detach().cpu().numpy()
    post_samples = inference_to_theta_np(post_samples_inference)
    samples_path = output_dir / "sbi_posterior_samples.npz"
    np.savez_compressed(
        samples_path,
        samples=post_samples,
        samples_inference=post_samples_inference,
        theta_fiducial=fiducial_theta(param_specs),
        theta_fiducial_inference=theta_to_inference_np(theta0_np[None, :])[0],
        prior_min=low,
        prior_max=high,
        data_vector=obs_np,
        cov=np.asarray(selected["cov"]),
        chol=chol_np,
        selection_indices=np.asarray(selection.indices),
        selection_ell_indices=np.asarray(selection.ell_indices),
        metadata_json=np.asarray(
            metadata_json(
                param_specs,
                selection,
                {
                    "fiducial_path": str(fiducial_path),
                    "runtime_sec": time.time() - t0,
                    "simulations_per_round": simulations_per_round,
                    "posterior_samples": posterior_samples,
                    "hidden_features": hidden_features,
                    "num_transforms": num_transforms,
                    "training_batch_size": training_batch_size,
                    "max_num_epochs": max_num_epochs,
                    "validation": validation,
                    "round_paths": round_paths,
                    "x_is_centered_cholesky_whitened": True,
                    "fiducial_offset_correction": fiducial_offset,
                    "theory_backend": theory_backend,
                    "summary_compression": summary_compression,
                    "device": device,
                    "density_estimator": density_estimator_model,
                    "discard_prior_samples": discard_prior_samples,
                    "retrain_from_scratch": retrain_from_scratch,
                    "force_first_round_loss": force_first_round_loss,
                    "num_atoms": num_atoms,
                    "validation_fraction": validation_fraction,
                    "learning_rate": learning_rate,
                    "parameter_transform": parameter_transform,
                    "posterior_sample_attempts": posterior_sample_attempts,
                    "num_components": num_components,
                    "num_bins": num_bins,
                    "reparameterized": theta_transform is not None,
                    "theory_param_names": [spec.name for spec in theory_specs],
                },
            )
        ),
    )
    for key, value in theory_info.items():
        with open(output_dir / f"theory_{key}.npy", "wb") as f:
            np.save(f, np.asarray(value))
    for key, value in compression_info.items():
        if key == "summary_compression":
            continue
        with open(output_dir / f"sbi_{key}.npy", "wb") as f:
            np.save(f, np.asarray(value))
    for key, value in transform_info.items():
        if value is None or key == "parameter_transform":
            continue
        with open(output_dir / f"sbi_{key}.npy", "wb") as f:
            np.save(f, np.asarray(value))

    diagnostics = {
        "runtime_sec": time.time() - t0,
        "validation": validation,
        "fiducial_path": str(fiducial_path),
        "simulations_per_round": simulations_per_round,
        "posterior_samples": posterior_samples,
        "posterior_sample_attempts": posterior_sample_attempts,
        "round_paths": round_paths,
        "samples_path": str(samples_path),
        "fiducial_offset_correction": fiducial_offset,
        "theory_backend": theory_backend,
        "summary_compression": summary_compression,
        "device": device,
        "density_estimator": density_estimator_model,
        "discard_prior_samples": discard_prior_samples,
        "retrain_from_scratch": retrain_from_scratch,
        "force_first_round_loss": force_first_round_loss,
        "num_atoms": num_atoms,
        "validation_fraction": validation_fraction,
        "learning_rate": learning_rate,
        "parameter_transform": parameter_transform,
        "hidden_features": hidden_features,
        "num_transforms": num_transforms,
        "num_components": num_components,
        "num_bins": num_bins,
        "training_batch_size": training_batch_size,
        "max_num_epochs": max_num_epochs,
        "reparameterized": theta_transform is not None,
    }
    with (output_dir / "sbi_diagnostics.json").open("w") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)

    return {
        "samples_path": samples_path,
        "diagnostics_path": output_dir / "sbi_diagnostics.json",
        "round_paths": round_paths,
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
    parser.add_argument("--simulations-per-round", default="512,512,1024")
    parser.add_argument("--posterior-samples", type=int, default=20000)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--num-transforms", type=int, default=5)
    parser.add_argument("--density-estimator-model", choices=("nsf", "maf", "mdn", "made"),
                        default="nsf")
    parser.add_argument("--num-components", type=int, default=10,
                        help="Number of mixture components for MDN density estimators.")
    parser.add_argument("--num-bins", type=int, default=10,
                        help="Number of spline bins for NSF density estimators.")
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--max-num-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-jit", action="store_true")
    parser.add_argument("--no-fiducial-offset", action="store_true",
                        help="Disable the constant numerical offset that aligns the JAX evaluator to the saved fiducial product.")
    parser.add_argument("--theory-backend", choices=("linearized", "direct"),
                        default="linearized")
    parser.add_argument("--summary-compression", choices=("score", "none"),
                        default="score")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--discard-prior-samples", action="store_true",
                        help="In later rounds, train SNPE without round-0 prior simulations.")
    parser.add_argument("--retrain-from-scratch", action="store_true",
                        help="Retrain the density estimator from scratch in each round.")
    parser.add_argument("--force-first-round-loss", action="store_true",
                        help="Use first-round loss even after proposal rounds.")
    parser.add_argument("--num-atoms", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--parameter-transform", choices=("none", "fisher"), default="none",
                        help="Train SNPE in raw parameters or in a Fisher-whitened linear parameter basis.")
    parser.add_argument("--reparameterize-phi", action="store_true",
    help="Sample (theta_ej_0, phi) with a uniform prior instead of "
         "(theta_ej_0, nu_theta_ej_M); phi is converted back to nu "
         "before being handed to the simulator.",)
    args = parser.parse_args()

    base_param_specs = parse_param_specs(args.param_spec)  # physical: theta_ej_0, nu_theta_ej_M

    # Fiducial product MUST be built from physical specs, before any phi swap.
    fiducial_path = ensure_default_fiducial_product(args.fiducial_path,
    param_specs=base_param_specs,force=args.force_fiducial,)

    param_specs = base_param_specs
    theory_param_specs = None
    theta_transform = None
    if args.reparameterize_phi:
        theory_param_specs = base_param_specs
        param_specs = phi_parameter_specs(base_param_specs)
        theta_transform = phi_theta_transform
    
    result = run_sbi(
        fiducial_path=pathlib.Path(fiducial_path),
        output_dir=pathlib.Path(args.output_dir),
        probes=parse_probe_list(args.probes),
        param_specs=param_specs,
        theory_param_specs=theory_param_specs,
        theta_transform=theta_transform,
        ell_min=args.ell_min,
        ell_max=args.ell_max,
        simulations_per_round=_parse_rounds(args.simulations_per_round),
        posterior_samples=args.posterior_samples,
        seed=args.seed,
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms,
        density_estimator_model=args.density_estimator_model,
        num_components=args.num_components,
        num_bins=args.num_bins,
        training_batch_size=args.training_batch_size,
        max_num_epochs=args.max_num_epochs,
        jit_compile=not args.no_jit,
        fiducial_offset=not args.no_fiducial_offset,
        theory_backend=args.theory_backend,
        summary_compression=args.summary_compression,
        device=args.device,
        discard_prior_samples=args.discard_prior_samples,
        retrain_from_scratch=args.retrain_from_scratch,
        force_first_round_loss=args.force_first_round_loss,
        num_atoms=args.num_atoms,
        validation_fraction=args.validation_fraction,
        learning_rate=args.learning_rate,
        parameter_transform=args.parameter_transform,
    )
    print(f"Saved SBI posterior samples to {result['samples_path']}")
    print(f"Saved SBI diagnostics to {result['diagnostics_path']}")


if __name__ == "__main__":
    main()
