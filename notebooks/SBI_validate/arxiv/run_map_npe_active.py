"""Run active NPE for pasted-map summaries with multi-anchor compression."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import subprocess
import sys
import time

import numpy as np
import torch
from torch.distributions import Distribution, MultivariateNormal, constraints

from theory_sbi_utils import DEFAULT_FIDUCIAL_PATH, default_parameter_specs, parse_probe_list, prior_bounds
from map_npe_utils import (
    build_multi_anchor_compressor,
    compress_datavectors,
    guide_posterior_from_observation,
    save_compressor,
    save_json,
    split_rounds,
)


class MixtureBoxGaussian(Distribution):
    """Mixture of a BoxUniform prior and a Gaussian restricted to the prior box."""

    arg_constraints = {}
    support = constraints.real_vector
    has_rsample = False

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray,
        prior_weight: float,
        component_weights: np.ndarray | None = None,
        device: str = "cpu",
    ) -> None:
        event_shape = torch.Size([len(low)])
        super().__init__(batch_shape=torch.Size(), event_shape=event_shape, validate_args=False)
        self.low = torch.as_tensor(low, dtype=torch.float32, device=device)
        self.high = torch.as_tensor(high, dtype=torch.float32, device=device)
        mean_arr = np.atleast_2d(np.asarray(mean, dtype=float))
        if mean_arr.shape[1] != len(low):
            raise ValueError("mean must have shape (ndim,) or (ncomp, ndim)")
        cov_input = np.asarray(cov, dtype=float)
        if cov_input.ndim == 2:
            cov_one = 0.5 * (cov_input + cov_input.T)
            cov_arr = np.repeat(cov_one[None, :, :], mean_arr.shape[0], axis=0)
        elif cov_input.ndim == 3:
            cov_arr = cov_input
            if cov_arr.shape[0] != mean_arr.shape[0]:
                raise ValueError("cov and mean must have the same number of components")
        else:
            raise ValueError("cov must have shape (ndim, ndim) or (ncomp, ndim, ndim)")
        cov_fixed = []
        for cov_i in cov_arr:
            cov_i = 0.5 * (cov_i + cov_i.T)
            eig = np.linalg.eigvalsh(cov_i)
            if np.min(eig) <= 0:
                cov_i = cov_i + np.eye(cov_i.shape[0]) * (abs(float(np.min(eig))) + 1.0e-8)
            cov_fixed.append(cov_i)
        cov_arr = np.asarray(cov_fixed, dtype=float)
        if component_weights is None:
            weights = np.ones(mean_arr.shape[0], dtype=float) / float(mean_arr.shape[0])
        else:
            weights = np.asarray(component_weights, dtype=float)
            weights = np.clip(weights, 0.0, np.inf)
            if weights.shape != (mean_arr.shape[0],) or np.sum(weights) <= 0.0:
                raise ValueError("component_weights must be positive with one entry per component")
            weights = weights / np.sum(weights)
        self.loc = torch.as_tensor(mean_arr, dtype=torch.float32, device=device)
        self.component_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.covariance_matrix = torch.as_tensor(cov_arr, dtype=torch.float32, device=device)
        self.gaussian = MultivariateNormal(self.loc, covariance_matrix=self.covariance_matrix)
        self.prior_weight = float(np.clip(prior_weight, 0.0, 1.0))
        self._log_prior = -torch.sum(torch.log(self.high - self.low))

    @property
    def mean(self) -> torch.Tensor:
        uniform_mean = 0.5 * (self.low + self.high)
        gaussian_mean = torch.sum(self.component_weights[:, None] * self.loc, dim=0)
        if self.prior_weight >= 1.0:
            return uniform_mean
        if self.prior_weight <= 0.0:
            return gaussian_mean
        return self.prior_weight * uniform_mean + (1.0 - self.prior_weight) * gaussian_mean

    @property
    def variance(self) -> torch.Tensor:
        uniform_mean = 0.5 * (self.low + self.high)
        uniform_var = (self.high - self.low) ** 2 / 12.0
        gaussian_mean = torch.sum(self.component_weights[:, None] * self.loc, dim=0)
        gaussian_var = torch.sum(
            self.component_weights[:, None]
            * (
                torch.diagonal(self.covariance_matrix, dim1=-2, dim2=-1)
                + (self.loc - gaussian_mean[None, :]) ** 2
            ),
            dim=0,
        )
        if self.prior_weight >= 1.0:
            return uniform_var
        if self.prior_weight <= 0.0:
            return gaussian_var
        mix_mean = self.mean
        return (
            self.prior_weight * (uniform_var + (uniform_mean - mix_mean) ** 2)
            + (1.0 - self.prior_weight) * (gaussian_var + (gaussian_mean - mix_mean) ** 2)
        )

    @property
    def stddev(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp(self.variance, min=1.0e-30))

    def _inside(self, value: torch.Tensor) -> torch.Tensor:
        return torch.logical_and(value >= self.low, value <= self.high).all(dim=-1)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        nsamp = int(np.prod(sample_shape)) if sample_shape else 1
        chunks = []
        kept = 0
        max_attempted = max(100 * nsamp, nsamp + 10000)
        attempted = 0
        while kept < nsamp and attempted < max_attempted:
            nleft = nsamp - kept
            ndraw = min(max(2 * nleft, 256), max_attempted - attempted)
            use_prior = torch.rand(ndraw, device=self.low.device) < self.prior_weight
            prior_draw = self.low + torch.rand(ndraw, len(self.low), device=self.low.device) * (self.high - self.low)
            comp = torch.multinomial(self.component_weights, ndraw, replacement=True)
            gauss_all = self.gaussian.sample((ndraw,))
            gauss_draw = gauss_all[torch.arange(ndraw, device=self.low.device), comp]
            draw = torch.where(use_prior[:, None], prior_draw, gauss_draw)
            inside = self._inside(draw)
            if inside.any():
                accepted = draw[inside]
                chunks.append(accepted)
                kept += int(accepted.shape[0])
            attempted += ndraw
        if kept < nsamp:
            raise RuntimeError(f"Only sampled {kept}/{nsamp} points inside the prior")
        out = torch.cat(chunks, dim=0)[:nsamp]
        return out.reshape(sample_shape + self.event_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        inside = self._inside(value)
        log_prior = self._log_prior.expand(value.shape[:-1])
        flat = value.reshape((-1, value.shape[-1]))
        log_gauss_components = (
            self.gaussian.log_prob(flat[:, None, :])
            + torch.log(torch.clamp(self.component_weights, min=1.0e-30))[None, :]
        )
        log_gauss = torch.logsumexp(log_gauss_components, dim=-1).reshape(value.shape[:-1])
        if self.prior_weight <= 0:
            log_mix = log_gauss
        elif self.prior_weight >= 1:
            log_mix = log_prior
        else:
            terms = torch.stack([
                torch.log(torch.as_tensor(self.prior_weight, dtype=value.dtype, device=value.device)) + log_prior,
                torch.log(torch.as_tensor(1.0 - self.prior_weight, dtype=value.dtype, device=value.device)) + log_gauss,
            ])
            log_mix = torch.logsumexp(terms, dim=0)
        return torch.where(inside, log_mix, torch.full_like(log_mix, -torch.inf))


def _parse_float_list(text: str, expected: int) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} comma-separated values, got {values}")
    return values


def _inside_prior(samples: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.all((samples >= low[None, :]) & (samples <= high[None, :]), axis=1)


def sample_proposal_np(proposal: Distribution, nsim: int) -> np.ndarray:
    with torch.no_grad():
        return proposal.sample((nsim,)).detach().cpu().numpy()


def _kmeans_labels(samples: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, dtype=float)
    n_components = int(min(max(1, n_components), len(samples)))
    rng = np.random.default_rng(seed)
    centers = samples[rng.choice(len(samples), size=n_components, replace=False)].copy()
    labels = np.zeros(len(samples), dtype=int)
    for _ in range(30):
        dist2 = np.sum((samples[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels_new = np.argmin(dist2, axis=1)
        if np.array_equal(labels_new, labels):
            break
        labels = labels_new
        for icomp in range(n_components):
            member = samples[labels == icomp]
            if len(member):
                centers[icomp] = member.mean(axis=0)
            else:
                centers[icomp] = samples[rng.integers(0, len(samples))]
    return centers, labels


def fit_temperature_mixture_proposal(
    samples: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    prior_weight: float,
    temperature: float,
    n_components: int,
    seed: int,
    device: str,
) -> MixtureBoxGaussian:
    """Fit a small broadened Gaussian mixture to posterior samples."""

    samples = np.asarray(samples, dtype=float)
    samples = samples[_inside_prior(samples, low, high)]
    if len(samples) < 4:
        mean = 0.5 * (low + high)
        cov = np.diag((high - low) ** 2)
        return MixtureBoxGaussian(low, high, mean, cov, prior_weight=1.0, device=device)
    centers, labels = _kmeans_labels(samples, n_components=n_components, seed=seed)
    ndim = samples.shape[1]
    global_cov = np.cov(samples.T) if len(samples) > ndim else np.diag((high - low) ** 2 / 12.0)
    global_cov = np.atleast_2d(global_cov)
    floor = np.diag(np.maximum((high - low) ** 2 * 1.0e-5, 1.0e-8))
    covs = []
    weights = []
    for icomp in range(len(centers)):
        member = samples[labels == icomp]
        weights.append(max(len(member), 1))
        if len(member) > ndim:
            cov_i = np.cov(member.T)
        else:
            cov_i = global_cov
        cov_i = np.atleast_2d(cov_i) * float(temperature) ** 2 + floor
        covs.append(cov_i)
    return MixtureBoxGaussian(
        low,
        high,
        np.clip(centers, low[None, :], high[None, :]),
        np.asarray(covs),
        prior_weight=prior_weight,
        component_weights=np.asarray(weights, dtype=float),
        device=device,
    )


def load_observation(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"]))
    return (
        np.asarray(data["data_vector"], dtype=float),
        np.asarray(data["ell"], dtype=float),
        np.asarray(data["delta_ell"], dtype=float),
        metadata,
    )


def combine_round_shards(output_dir: pathlib.Path, round_index: int, num_gpus: int) -> dict[str, np.ndarray]:
    shard_dir = output_dir / "shards" / f"round{round_index:02d}"
    shards = []
    for rank in range(num_gpus):
        path = shard_dir / f"shard_rank{rank:02d}_of{num_gpus:02d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing worker shard {path}")
        shards.append(np.load(path, allow_pickle=True))
    theta = np.vstack([s["theta"] for s in shards if len(s["theta"])])
    sim_id = np.concatenate([s["sim_id"] for s in shards if len(s["sim_id"])])
    data_vector = np.vstack([s["data_vector"] for s in shards if len(s["data_vector"])])
    order = np.argsort(sim_id)
    return {
        "theta": theta[order],
        "sim_id": sim_id[order],
        "data_vector": data_vector[order],
        "round_index": np.full(len(order), round_index, dtype=int),
    }


def launch_round_workers(
    output_dir: pathlib.Path,
    theta_table: pathlib.Path,
    round_index: int,
    num_gpus: int,
    nside: int,
    base_seed: int,
    probes: str,
    theory_path: pathlib.Path,
    add_survey_noise: bool,
    save_map_products: bool,
) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for rank in range(num_gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        env.pop("JAX_PLATFORMS", None)
        cmd = [
            sys.executable,
            "-u",
            "notebooks/SBI_validate/run_map_npe_round_worker.py",
            "--theta-table",
            str(theta_table),
            "--output-dir",
            str(output_dir),
            "--round-index",
            str(round_index),
            "--rank",
            str(rank),
            "--world-size",
            str(num_gpus),
            "--theory-path",
            str(theory_path),
            "--nside",
            str(nside),
            "--base-seed",
            str(base_seed),
            "--probes",
            probes,
        ]
        cmd.append("--add-survey-noise" if add_survey_noise else "--no-add-survey-noise")
        if save_map_products:
            cmd.append("--save-map-products")
        stdout = (log_dir / f"round{round_index:02d}_rank{rank:02d}.out").open("w")
        stderr = (log_dir / f"round{round_index:02d}_rank{rank:02d}.err").open("w")
        procs.append((subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env), stdout, stderr))
    failures = []
    for proc, stdout, stderr in procs:
        code = proc.wait()
        stdout.close()
        stderr.close()
        if code != 0:
            failures.append(code)
    if failures:
        raise RuntimeError(f"{len(failures)} worker process(es) failed in round {round_index}")


def train_snpe_ensemble(
    theta_rounds: list[np.ndarray],
    x_rounds: list[np.ndarray],
    proposal_rounds: list[Distribution],
    prior: Distribution,
    x_obs: np.ndarray,
    output_dir: pathlib.Path,
    ensemble_size: int,
    posterior_samples: int,
    density_estimator: str,
    num_components: int,
    hidden_features: int,
    max_num_epochs: int,
    training_batch_size: int,
    seed: int,
    device: str,
    label: str,
) -> tuple[np.ndarray, list[object]]:
    from sbi.inference import SNPE
    from sbi.utils import posterior_nn

    samples_all = []
    posteriors = []
    x_obs_t = torch.as_tensor(x_obs, dtype=torch.float32, device=device)
    nsamp_each = int(np.ceil(posterior_samples / ensemble_size))
    for iens in range(ensemble_size):
        torch.manual_seed(seed + 7919 * iens)
        density = posterior_nn(
            model=density_estimator,
            hidden_features=hidden_features,
            num_components=num_components,
        )
        inference = SNPE(prior=prior, density_estimator=density, device=device, show_progress_bars=True)
        for theta_np, x_np, proposal in zip(theta_rounds, x_rounds, proposal_rounds):
            inference.append_simulations(
                torch.as_tensor(theta_np, dtype=torch.float32, device=device),
                torch.as_tensor(x_np, dtype=torch.float32, device=device),
                proposal=proposal,
            )
        estimator = inference.train(
            num_atoms=min(20, max(2, sum(len(t) for t in theta_rounds) // 4)),
            training_batch_size=training_batch_size,
            max_num_epochs=max_num_epochs,
            validation_fraction=0.15,
            learning_rate=5.0e-4,
            retrain_from_scratch=True,
            discard_prior_samples=False,
            show_train_summary=True,
        )
        posterior = inference.build_posterior(estimator)
        posterior.set_default_x(x_obs_t)
        draw = posterior.sample((max(2 * nsamp_each, 2048),), x=x_obs_t, show_progress_bars=True)
        draw_np = draw.detach().cpu().numpy()
        low_np = prior.low.detach().cpu().numpy()
        high_np = prior.high.detach().cpu().numpy()
        draw_np = draw_np[_inside_prior(draw_np, low_np, high_np)]
        if len(draw_np) < nsamp_each:
            extra = posterior.sample((10 * nsamp_each,), x=x_obs_t, show_progress_bars=True).detach().cpu().numpy()
            extra = extra[_inside_prior(extra, low_np, high_np)]
            draw_np = np.vstack([draw_np, extra])
        samples_all.append(draw_np[:nsamp_each])
        posteriors.append(posterior)
        with open(output_dir / f"{label}_posterior_ensemble{iens}.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
    return np.vstack(samples_all)[:posterior_samples], posteriors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--observation", type=pathlib.Path, required=True)
    parser.add_argument("--theory-path", type=pathlib.Path, default=DEFAULT_FIDUCIAL_PATH)
    parser.add_argument("--nsim-total", type=int, default=256)
    parser.add_argument("--rounds", default="64,64,64,64")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--base-seed", type=int, default=20260527)
    parser.add_argument("--probes", default="gg,gy,gtau,gkappa")
    parser.add_argument("--proposal-broadening-round0", type=float, default=4.0)
    parser.add_argument("--proposal-temperature", type=float, default=1.5)
    parser.add_argument("--proposal-components", type=int, default=5)
    parser.add_argument("--prior-mixture-rounds", default="0.30,0.20,0.10,0.05")
    parser.add_argument("--density-estimator", default="mdn", choices=("mdn", "nsf", "maf"))
    parser.add_argument("--num-components", type=int, default=5)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--posterior-samples", type=int, default=50000)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--max-num-epochs", type=int, default=250)
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--guide-ngrid", type=int, default=240)
    parser.add_argument("--guide-backend", choices=("linearized", "direct"), default="linearized")
    parser.add_argument("--compression-backend", choices=("linearized", "direct"), default="direct")
    parser.add_argument("--add-survey-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-map-products", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rounds = split_rounds(args.rounds)
    if sum(rounds) != args.nsim_total:
        raise ValueError(f"rounds sum to {sum(rounds)} but nsim-total={args.nsim_total}")
    prior_mixture = _parse_float_list(args.prior_mixture_rounds, len(rounds))
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    param_specs = default_parameter_specs()
    probes = parse_probe_list(args.probes)
    low, high = prior_bounds(param_specs)
    prior = MixtureBoxGaussian(low, high, 0.5 * (low + high), np.diag((high - low) ** 2), 1.0, device=args.device)
    obs_vector, obs_ell, obs_delta_ell, obs_meta = load_observation(args.observation)
    if not np.all(np.isfinite(obs_vector)):
        bad = int(np.size(obs_vector) - np.count_nonzero(np.isfinite(obs_vector)))
        raise ValueError(f"Observation contains {bad} non-finite datavector entries")
    ell_min = float(np.min(obs_ell))
    ell_max = float(np.max(obs_ell))
    validation_truth = np.asarray(
        obs_meta.get("theta_truth", [np.nan] * len(param_specs)),
        dtype=float,
    )
    guide, _ = guide_posterior_from_observation(
        obs_vector,
        args.theory_path,
        probes,
        param_specs,
        ell_min=ell_min,
        ell_max=ell_max,
        backend=args.guide_backend,
        ngrid=args.guide_ngrid,
        broadening=1.0,
    )
    compressor = build_multi_anchor_compressor(
        guide.anchors,
        args.theory_path,
        probes,
        param_specs,
        ell_min=ell_min,
        ell_max=ell_max,
        backend=args.compression_backend,
    )
    save_compressor(args.output_dir / "multi_anchor_compressor.npz", compressor)
    x_obs_raw = compress_datavectors(obs_vector[None, :], compressor)[0]

    theta_rounds: list[np.ndarray] = []
    x_rounds_raw: list[np.ndarray] = []
    proposal_rounds: list[Distribution] = []
    all_round_data = []
    next_proposal = MixtureBoxGaussian(
        low,
        high,
        guide.guide_mean,
        guide.guide_cov * args.proposal_broadening_round0**2,
        prior_weight=prior_mixture[0],
        device=args.device,
    )

    sim_id_offset = 0
    for iround, nsim in enumerate(rounds):
        proposal = next_proposal
        theta = sample_proposal_np(proposal, nsim)
        sim_ids = np.arange(sim_id_offset, sim_id_offset + nsim, dtype=int)
        sim_id_offset += nsim
        theta_table = args.output_dir / f"theta_round{iround:02d}.npz"
        np.savez_compressed(theta_table, theta=theta, sim_id=sim_ids, round_index=np.asarray(iround))
        proposal_rounds.append(proposal)
        launch_round_workers(
            args.output_dir,
            theta_table,
            iround,
            args.num_gpus,
            args.nside,
            args.base_seed,
            args.probes,
            args.theory_path,
            args.add_survey_noise,
            args.save_map_products,
        )
        round_data = combine_round_shards(args.output_dir, iround, args.num_gpus)
        if len(round_data["sim_id"]) != nsim:
            raise RuntimeError(
                f"Round {iround} produced {len(round_data['sim_id'])}/{nsim} valid simulations"
            )
        round_features = compress_datavectors(round_data["data_vector"], compressor)
        theta_rounds.append(round_data["theta"])
        x_rounds_raw.append(round_features)
        all_round_data.append(round_data)

        if iround + 1 < len(rounds):
            x_accum = np.vstack(x_rounds_raw)
            x_mean = x_accum.mean(axis=0)
            x_std = np.maximum(x_accum.std(axis=0), 1.0e-12)
            x_rounds_std = [(x - x_mean[None, :]) / x_std[None, :] for x in x_rounds_raw]
            x_obs_std = (x_obs_raw - x_mean) / x_std

            proposal_samples, _ = train_snpe_ensemble(
                theta_rounds,
                x_rounds_std,
                proposal_rounds,
                prior,
                x_obs_std,
                args.output_dir,
                ensemble_size=1,
                posterior_samples=20000,
                density_estimator=args.density_estimator,
                num_components=args.num_components,
                hidden_features=args.hidden_features,
                max_num_epochs=args.max_num_epochs,
                training_batch_size=args.training_batch_size,
                seed=args.base_seed + 1000 * (iround + 1),
                device=args.device,
                label=f"proposal_round{iround:02d}",
            )
            next_proposal = fit_temperature_mixture_proposal(
                proposal_samples,
                low,
                high,
                prior_weight=prior_mixture[iround + 1],
                temperature=args.proposal_temperature,
                n_components=args.proposal_components,
                seed=args.base_seed + 2000 * (iround + 1),
                device=args.device,
            )

    x_accum = np.vstack(x_rounds_raw)
    x_mean = x_accum.mean(axis=0)
    x_std = np.maximum(x_accum.std(axis=0), 1.0e-12)
    x_rounds_std = [(x - x_mean[None, :]) / x_std[None, :] for x in x_rounds_raw]
    x_obs_std = (x_obs_raw - x_mean) / x_std
    final_samples, _ = train_snpe_ensemble(
        theta_rounds,
        x_rounds_std,
        proposal_rounds,
        prior,
        x_obs_std,
        args.output_dir,
        ensemble_size=args.ensemble_size,
        posterior_samples=args.posterior_samples,
        density_estimator=args.density_estimator,
        num_components=args.num_components,
        hidden_features=args.hidden_features,
        max_num_epochs=args.max_num_epochs,
        training_batch_size=args.training_batch_size,
        seed=args.base_seed + 99999,
        device=args.device,
        label="final",
    )

    all_theta = np.vstack([d["theta"] for d in all_round_data])
    all_data = np.vstack([d["data_vector"] for d in all_round_data])
    all_features = np.vstack(x_rounds_raw)
    all_round_idx = np.concatenate([d["round_index"] for d in all_round_data])
    all_sim_id = np.concatenate([d["sim_id"] for d in all_round_data])
    np.savez_compressed(
        args.output_dir / "map_npe_simulations.npz",
        theta=all_theta,
        data_vector=all_data,
        features_raw=all_features,
        features_mean=x_mean,
        features_std=x_std,
        x_obs_raw=x_obs_raw,
        x_obs_std=x_obs_std,
        ell=obs_ell,
        delta_ell=obs_delta_ell,
        round_index=all_round_idx,
        sim_id=all_sim_id,
    )
    np.savez_compressed(
        args.output_dir / "map_npe_posterior_samples.npz",
        samples=final_samples,
        theta_validation_truth=validation_truth,
        prior_min=low,
        prior_max=high,
        x_obs_raw=x_obs_raw,
        x_obs_std=x_obs_std,
        ell=obs_ell,
        delta_ell=obs_delta_ell,
    )
    diagnostics = {
        "runtime_sec": time.time() - t0,
        "observation": str(args.observation),
        "observation_metadata": obs_meta,
        "theory_path": str(args.theory_path),
        "nsim_total": int(args.nsim_total),
        "n_valid_simulations": int(len(all_sim_id)),
        "rounds": rounds,
        "num_gpus": int(args.num_gpus),
        "nside": int(args.nside),
        "ell_min": ell_min,
        "ell_max": ell_max,
        "nell": int(len(obs_ell)),
        "probes": list(probes),
        "proposal_broadening_round0": float(args.proposal_broadening_round0),
        "proposal_temperature": float(args.proposal_temperature),
        "proposal_components": int(args.proposal_components),
        "prior_mixture_rounds": prior_mixture,
        "density_estimator": args.density_estimator,
        "num_components": int(args.num_components),
        "ensemble_size": int(args.ensemble_size),
        "posterior_samples": int(args.posterior_samples),
        "guide_backend": args.guide_backend,
        "compression_backend": args.compression_backend,
        "guide_mean": guide.guide_mean.tolist(),
        "guide_cov": guide.guide_cov.tolist(),
        "anchors": guide.anchors.tolist(),
        "final_sample_mean": final_samples.mean(axis=0).tolist(),
        "final_sample_std": final_samples.std(axis=0).tolist(),
    }
    save_json(args.output_dir / "map_npe_diagnostics.json", diagnostics)
    print(f"Saved final posterior samples to {args.output_dir / 'map_npe_posterior_samples.npz'}")
    print(json.dumps({k: diagnostics[k] for k in ("nsim_total", "rounds", "final_sample_mean", "final_sample_std")}, indent=2))


if __name__ == "__main__":
    main()
