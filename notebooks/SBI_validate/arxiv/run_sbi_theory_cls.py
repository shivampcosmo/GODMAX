"""Run active-learning SNPE for the analytical Cl SBI-validation likelihood."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Sequence

import numpy as np
import torch
from scipy.special import ndtr, ndtri
from torch.distributions import Distribution, Independent, Normal, constraints

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax

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
    save_pickle,
    selected_product_arrays,
    validate_theory_vector,
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


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: pathlib.Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
    fixed_param_specs: Sequence[ParameterSpec] | None = None,
    theory_mode: str = "map_matched_resolved",
    simulation_batch_size: int = 1,
    posterior_samples_per_round: int = 0,
    round_contract: dict | None = None,
    validate_theory: bool = True,
    validation_override: dict | None = None,
) -> dict:
    """Run sequential SNPE using noisy whitened theory vectors or summaries."""

    from sbi.inference import SNPE
    from sbi.utils import BoxUniform, posterior_nn

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

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
    if validate_theory:
        validation = validate_theory_vector(vector_fn, selected, param_specs)
    elif validation_override is not None:
        validation = dict(validation_override)
    else:
        raise ValueError("Skipping theory validation requires validation_override")

    low, high = prior_bounds(param_specs)
    theta_prior = BoxUniform(
        low=torch.as_tensor(low, dtype=torch.float32, device=torch_device),
        high=torch.as_tensor(high, dtype=torch.float32, device=torch_device),
        device=device,
    )
    obs_np = np.asarray(selected["data_vector"], dtype=float)
    chol_np = np.asarray(selected["chol"], dtype=float)
    theta0_np = fiducial_theta(param_specs)
    compression_info = {"summary_compression": summary_compression}
    jac_for_batch = np.asarray(theory_info["jacobian"], dtype=float) if "jacobian" in theory_info else None
    theory_evaluation_mode = (
        "linearized_jacobian_batch"
        if jac_for_batch is not None
        else "direct_per_simulation"
    )
    if summary_compression == "none":
        x_obs_np = np.zeros(len(obs_np), dtype=float)
        compress = lambda x_white: x_white
    elif summary_compression == "score":
        if "jacobian" in theory_info:
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

    elif parameter_transform == "probit":
        if summary_compression != "none":
            raise ValueError("parameter_transform='probit' requires summary_compression='none'")
        prior = Independent(
            Normal(
                torch.zeros(len(low), dtype=torch.float32, device=torch_device),
                torch.ones(len(low), dtype=torch.float32, device=torch_device),
            ),
            1,
        )

        def inference_to_theta_np(samples: np.ndarray) -> np.ndarray:
            quantiles = ndtr(np.asarray(samples, dtype=float))
            return low[None, :] + (high - low)[None, :] * quantiles

        def theta_to_inference_np(samples: np.ndarray) -> np.ndarray:
            quantiles = (
                (np.asarray(samples, dtype=float) - low[None, :])
                / (high - low)[None, :]
            )
            if np.any((quantiles <= 0.0) | (quantiles >= 1.0)):
                raise ValueError("Probit transform requires parameters strictly inside the prior")
            return ndtri(quantiles)

        def transform_summary_np(summary: np.ndarray) -> np.ndarray:
            return np.asarray(summary, dtype=float)

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
        raise ValueError("parameter_transform must be 'none', 'probit', or 'fisher'")

    x_obs = torch.as_tensor(x_obs_np, dtype=torch.float32, device=torch_device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    if simulation_batch_size < 1:
        raise ValueError("simulation_batch_size must be positive")
    if posterior_samples_per_round < 0:
        raise ValueError("posterior_samples_per_round cannot be negative")
    batched_vector_fn = None
    if jac_for_batch is None and simulation_batch_size > 1:
        batched_vector_fn = jax.jit(jax.vmap(vector_fn))

    def simulate_whitened(
        theta_t: torch.Tensor, round_seed: int
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        theta_np = inference_to_theta_np(theta_t.detach().cpu().numpy())
        eps_rng = np.random.default_rng(round_seed)
        if jac_for_batch is not None:
            mu_minus_obs = (theta_np - theta0_np) @ jac_for_batch.T
            mean_white = np.linalg.solve(chol_np, mu_minus_obs.T).T
            x_white = mean_white + eps_rng.normal(size=mean_white.shape)
            if summary_compression == "score":
                x_theta_units = theta0_np[None, :] + x_white @ compression_matrix.T
                x_out = transform_summary_np(x_theta_units)
            else:
                x_out = x_white
            return (
                torch.as_tensor(x_out, dtype=torch.float32, device=torch_device),
                mean_white,
                x_white - mean_white,
            )

        if batched_vector_fn is None:
            prediction_chunks = (
                np.asarray(
                    vector_fn(jnp.asarray(row, dtype=jnp.float64)), dtype=float
                )[None, :]
                for row in theta_np
            )
        else:
            prediction_chunks = (
                np.asarray(
                    batched_vector_fn(
                        jnp.asarray(
                            theta_np[start : start + simulation_batch_size],
                            dtype=jnp.float64,
                        )
                    ),
                    dtype=float,
                )
                for start in range(0, len(theta_np), simulation_batch_size)
            )
        predictions = np.concatenate(list(prediction_chunks), axis=0)
        mean_white = np.linalg.solve(chol_np, (predictions - obs_np).T).T
        noise_all = eps_rng.normal(size=mean_white.shape)
        x_white = mean_white + noise_all
        if summary_compression == "none":
            x_rows = x_white
        else:
            x_rows = np.asarray([compress(row) for row in x_white])
        return (
            torch.as_tensor(
                transform_summary_np(np.asarray(x_rows)),
                dtype=torch.float32,
                device=torch_device,
            ),
            mean_white,
            noise_all,
        )

    density_estimator = posterior_nn(
        model=density_estimator_model,
        z_score_theta="independent",
        z_score_x="independent",
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
    round_simulation_runtime_sec: list[float] = []
    round_training_summaries: list[dict[str, object]] = []
    round_posterior_paths: list[str] = []
    round_ready_paths: list[str] = []
    last_round_posterior_samples: np.ndarray | None = None
    last_round_posterior_samples_inference: np.ndarray | None = None
    last_round_posterior_attempts: int | None = None
    previous_round_samples_sha256: str | None = None
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

        round_seed = seed + 1000 * (iround + 1)
        simulation_t0 = time.time()
        x, x_mean_white, epsilon_white = simulate_whitened(
            theta, round_seed=round_seed
        )
        round_simulation_runtime_sec.append(time.time() - simulation_t0)
        theta_physical = inference_to_theta_np(theta.detach().cpu().numpy())
        x_np = x.detach().cpu().numpy()
        if (
            not np.all(np.isfinite(theta_physical))
            or not np.all((theta_physical >= low[None, :]) & (theta_physical <= high[None, :]))
            or not np.all(np.isfinite(x_np))
            or not np.all(np.isfinite(x_mean_white))
            or not np.all(np.isfinite(epsilon_white))
        ):
            raise RuntimeError(f"Round {iround + 1} produced non-finite/out-of-prior simulations")
        if summary_compression == "none" and not np.array_equal(
            x_np, np.asarray(x_mean_white + epsilon_white, dtype=np.float32)
        ):
            raise RuntimeError(f"Round {iround + 1} whitened simulation reconstruction failed")
        npz_path = output_dir / f"sbi_round{iround}_simulations.npz"
        _atomic_npz(
            npz_path,
            {
                "theta": theta_physical,
                "theta_inference": theta.detach().cpu().numpy(),
                "x": x_np,
                "x_mean_white": x_mean_white,
                "epsilon_white": epsilon_white,
                "round": np.asarray(iround),
                "round_seed": np.asarray(round_seed),
                "nsim": np.asarray(nsim),
            },
        )
        round_paths.append(str(npz_path))

        training_log_start = len(inference._summary["training_log_probs"])
        validation_log_start = len(inference._summary["validation_log_probs"])
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
        round_training_summaries.append({
            "round": iround,
            "epochs_trained": int(inference._summary["epochs_trained"][-1]),
            "best_validation_log_prob": float(
                inference._summary["best_validation_log_prob"][-1]
            ),
            "training_log_probs": [
                float(value)
                for value in inference._summary["training_log_probs"][training_log_start:]
            ],
            "validation_log_probs": [
                float(value)
                for value in inference._summary["validation_log_probs"][validation_log_start:]
            ],
            "epoch_durations_sec": [
                float(value)
                for value in inference._summary["epoch_durations_sec"][validation_log_start:]
            ],
            "reached_requested_max_num_epochs": bool(
                int(inference._summary["epochs_trained"][-1]) >= max_num_epochs
            ),
        })
        current_training = round_training_summaries[-1]
        training_values = np.asarray(current_training["training_log_probs"], dtype=float)
        validation_values = np.asarray(
            current_training["validation_log_probs"], dtype=float
        )
        if (
            training_values.size == 0
            or validation_values.size == 0
            or not np.all(np.isfinite(training_values))
            or not np.all(np.isfinite(validation_values))
            or not np.isfinite(current_training["best_validation_log_prob"])
        ):
            raise RuntimeError(f"Round {iround + 1} training diagnostics are non-finite")
        if current_training["reached_requested_max_num_epochs"]:
            raise RuntimeError(
                f"Round {iround + 1} reached max_num_epochs without an early stop"
            )
        posterior = inference.build_posterior(density_estimator)
        posterior.set_default_x(x_obs)
        proposal = posterior
        save_pickle(output_dir / f"sbi_posterior_round{iround}.pkl", posterior)

        if posterior_samples_per_round:
            round_samples_t, round_attempts = sample_final_inside_prior(
                posterior, posterior_samples_per_round
            )
            round_samples_inference = round_samples_t.detach().cpu().numpy()
            round_samples = inference_to_theta_np(round_samples_inference)
            if (
                round_samples.shape != (posterior_samples_per_round, len(param_specs))
                or not np.all(np.isfinite(round_samples))
                or not np.all(
                    (round_samples >= low[None, :])
                    & (round_samples <= high[None, :])
                )
            ):
                raise RuntimeError(
                    f"Round {iround + 1} posterior samples are invalid/out of prior"
                )
            round_metadata = {
                "schema": "godmax.sbi.round.v1",
                "round_index": iround,
                "round_number": iround + 1,
                "simulations_this_round": nsim,
                "simulations_cumulative": int(
                    sum(simulations_per_round[: iround + 1])
                ),
                "simulations_per_round": list(simulations_per_round),
                "posterior_samples": posterior_samples_per_round,
                "posterior_sample_attempts": round_attempts,
                "seed": seed,
                "round_seed": round_seed,
                "theory_backend": theory_backend,
                "theory_mode": theory_mode,
                "theory_evaluation_mode": theory_evaluation_mode,
                "fiducial_offset_correction": fiducial_offset,
                "summary_compression": summary_compression,
                "summary_dimension": int(np.size(x_obs_np)),
                "parameter_transform": parameter_transform,
                "internal_z_score_theta": "independent",
                "internal_z_score_x": "independent",
                "density_estimator": density_estimator_model,
                "hidden_features": hidden_features,
                "num_transforms": num_transforms,
                "num_components": num_components,
                "num_bins": num_bins,
                "training_batch_size": training_batch_size,
                "max_num_epochs": max_num_epochs,
                "simulation_batch_size": simulation_batch_size,
                "training_summary": round_training_summaries[-1],
                "contract": dict(round_contract or {}),
            }
            round_sample_path = (
                output_dir / f"sbi_posterior_samples_round{iround + 1}.npz"
            )
            _atomic_npz(
                round_sample_path,
                {
                    "samples": round_samples,
                    "samples_inference": round_samples_inference,
                    "theta_fiducial": theta0_np,
                    "prior_min": low,
                    "prior_max": high,
                    "data_vector": obs_np,
                    "cov": np.asarray(selected["cov"]),
                    "chol": chol_np,
                    "selection_indices": np.asarray(selection.indices),
                    "selection_ell_indices": np.asarray(selection.ell_indices),
                    "metadata_json": np.asarray(
                        json.dumps(round_metadata, sort_keys=True)
                    ),
                },
            )
            round_diagnostics_path = (
                output_dir / f"sbi_round{iround + 1}_diagnostics.json"
            )
            _atomic_json(round_diagnostics_path, round_metadata)
            samples_sha256 = _file_sha256(round_sample_path)
            posterior_path = output_dir / f"sbi_posterior_round{iround}.pkl"
            ready_payload = {
                "schema": "godmax.sbi.round.ready.v1",
                "round_index": iround,
                "round_number": iround + 1,
                "samples_path": str(round_sample_path),
                "samples_sha256": samples_sha256,
                "simulation_path": str(npz_path),
                "simulation_sha256": _file_sha256(npz_path),
                "posterior_path": str(posterior_path),
                "posterior_sha256": _file_sha256(posterior_path),
                "diagnostics_path": str(round_diagnostics_path),
                "diagnostics_sha256": _file_sha256(round_diagnostics_path),
                "previous_round_samples_sha256": previous_round_samples_sha256,
                "contract": dict(round_contract or {}),
            }
            ready_path = output_dir / f"sbi_round{iround + 1}.ready.json"
            _atomic_json(ready_path, ready_payload)
            round_posterior_paths.append(str(round_sample_path))
            round_ready_paths.append(str(ready_path))
            previous_round_samples_sha256 = samples_sha256
            last_round_posterior_samples = round_samples
            last_round_posterior_samples_inference = round_samples_inference
            last_round_posterior_attempts = round_attempts

    if (
        posterior_samples_per_round == posterior_samples
        and last_round_posterior_samples is not None
    ):
        post_samples = last_round_posterior_samples
        post_samples_inference = last_round_posterior_samples_inference
        posterior_sample_attempts = int(last_round_posterior_attempts)
    else:
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
                    "round_simulation_runtime_sec": round_simulation_runtime_sec,
                    "round_training_summaries": round_training_summaries,
                    "round_posterior_paths": round_posterior_paths,
                    "round_ready_paths": round_ready_paths,
                    "seed": seed,
                    "x_is_centered_cholesky_whitened": True,
                    "fiducial_offset_correction": fiducial_offset,
                    "theory_backend": theory_backend,
                    "theory_mode": theory_mode,
                    "theory_evaluation_mode": theory_evaluation_mode,
                    "summary_compression": summary_compression,
                    "summary_dimension": int(np.size(x_obs_np)),
                    "device": device,
                    "density_estimator": density_estimator_model,
                    "internal_z_score_theta": "independent",
                    "internal_z_score_x": "independent",
                    "discard_prior_samples": discard_prior_samples,
                    "retrain_from_scratch": retrain_from_scratch,
                    "force_first_round_loss": force_first_round_loss,
                    "num_atoms": num_atoms,
                    "validation_fraction": validation_fraction,
                    "learning_rate": learning_rate,
                    "parameter_transform": parameter_transform,
                    "posterior_sample_attempts": posterior_sample_attempts,
                    "posterior_sample_attempts_interpretation": (
                        "wrapper candidate draws after DirectPosterior sampling; not the "
                        "density-network rejection-attempt count"
                    ),
                    "num_components": num_components,
                    "num_bins": num_bins,
                    "simulation_batch_size": simulation_batch_size,
                    "posterior_samples_per_round": posterior_samples_per_round,
                    "round_contract": dict(round_contract or {}),
                },
                fixed_param_specs=fixed_param_specs,
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
        "round_simulation_runtime_sec": round_simulation_runtime_sec,
        "round_training_summaries": round_training_summaries,
        "round_posterior_paths": round_posterior_paths,
        "round_ready_paths": round_ready_paths,
        "seed": seed,
        "samples_path": str(samples_path),
        "fiducial_offset_correction": fiducial_offset,
        "theory_backend": theory_backend,
        "theory_mode": theory_mode,
        "theory_evaluation_mode": theory_evaluation_mode,
        "summary_compression": summary_compression,
        "summary_dimension": int(np.size(x_obs_np)),
        "device": device,
        "density_estimator": density_estimator_model,
        "internal_z_score_theta": "independent",
        "internal_z_score_x": "independent",
        "discard_prior_samples": discard_prior_samples,
        "retrain_from_scratch": retrain_from_scratch,
        "force_first_round_loss": force_first_round_loss,
        "num_atoms": num_atoms,
        "validation_fraction": validation_fraction,
        "learning_rate": learning_rate,
        "parameter_transform": parameter_transform,
        "posterior_sample_attempts_interpretation": (
            "wrapper candidate draws after DirectPosterior sampling; not the "
            "density-network rejection-attempt count"
        ),
        "hidden_features": hidden_features,
        "num_transforms": num_transforms,
        "num_components": num_components,
        "num_bins": num_bins,
        "training_batch_size": training_batch_size,
        "max_num_epochs": max_num_epochs,
        "simulation_batch_size": simulation_batch_size,
        "posterior_samples_per_round": posterior_samples_per_round,
        "round_contract": dict(round_contract or {}),
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
    parser.add_argument("--theory-mode", choices=("full", "map_matched_resolved"),
                        default="map_matched_resolved")
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
    parser.add_argument(
        "--parameter-transform",
        choices=("none", "probit", "fisher"),
        default="none",
        help=(
            "Train SNPE in raw parameters, an exact box-prior-to-Normal probit basis, "
            "or a Fisher-whitened linear basis."
        ),
    )
    args = parser.parse_args()

    param_specs = parse_param_specs(args.param_spec)
    fiducial_path = ensure_default_fiducial_product(
        args.fiducial_path,
        param_specs=param_specs,
        force=args.force_fiducial,
        theory_mode=args.theory_mode,
    )
    result = run_sbi(
        fiducial_path=pathlib.Path(fiducial_path),
        output_dir=pathlib.Path(args.output_dir),
        probes=parse_probe_list(args.probes),
        param_specs=param_specs,
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
        theory_mode=args.theory_mode,
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
