#!/usr/bin/env python
"""Bounded MAP optimization for the Stage-31 GODMAX xDESI fit."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import optax
import yaml
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from scipy.optimize import minimize

import godmax_multiprobe_hmc_stage31 as hmc31
import godmax_multiprobe_theory_utils as gmt


DEFAULT_RUN_DIR = (
    "notebooks/xDESI/survey_measure/outputs/"
    "godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu/"
    "stage31_hmc_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_"
    "60param_13log_apo1degC2_pairmean_warm100_2000x16_checkpoint25_v1"
)
DEFAULT_SUFFIX = "stage31_map_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_13log_apo1degC2_pairmean"


def log_status(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_pair(value: object, *, option: str) -> Optional[Tuple[float, float]]:
    if value is None or str(value).strip() == "":
        return None
    parts = str(value).replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"{option} must contain two values, got {value!r}.")
    lo, hi = float(parts[0]), float(parts[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
        raise ValueError(f"{option} must be finite and increasing, got {value!r}.")
    return (lo, hi)


def parse_int_list(value: object, *, option: str) -> List[int]:
    if value is None or str(value).strip() == "":
        return []
    out = []
    for part in str(value).replace(",", " ").split():
        ivalue = int(part)
        if ivalue <= 0:
            raise ValueError(f"{option} entries must be positive integers, got {part!r}.")
        out.append(ivalue)
    return out


def latest_hmc_bestfit(run_dir: str | Path) -> Path:
    checkpoint_dir = latest_hmc_checkpoint_dir(run_dir)
    matches = sorted(checkpoint_dir.glob("bestfit_params_*.yaml"))
    if not matches:
        raise FileNotFoundError(f"No bestfit_params_*.yaml files found in {checkpoint_dir}")
    return matches[-1]


def latest_hmc_checkpoint_dir(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    checkpoint_root = root / "combined" / "checkpoints"
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_root}")
    candidates: List[Tuple[int, Path]] = []
    for checkpoint_dir in checkpoint_root.glob("checkpoint_*"):
        if not checkpoint_dir.is_dir():
            continue
        try:
            checkpoint = int(checkpoint_dir.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        candidates.append((checkpoint, checkpoint_dir))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_* directories found below {checkpoint_root}")
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def latest_hmc_chain_path(run_dir: str | Path) -> Path:
    checkpoint_dir = latest_hmc_checkpoint_dir(run_dir)
    matches = sorted(checkpoint_dir.glob("chain_*.npz"))
    if not matches:
        raise FileNotFoundError(f"No chain_*.npz file found in {checkpoint_dir}")
    return matches[-1]


def vector_from_sample(context: hmc31.FitContext, sample: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(sample[spec.name]) for spec in context.parameter_specs], dtype=np.float64)


def top_hmc_vectors(context: hmc31.FitContext, chain_path: str | Path, top_k: int) -> List[np.ndarray]:
    if top_k <= 0:
        return []
    path = Path(chain_path)
    if not path.exists():
        return []
    data = np.load(path, allow_pickle=True)
    if "sample__chi2" not in data:
        return []
    chi2 = np.asarray(data["sample__chi2"], dtype=np.float64).reshape(-1)
    order = np.argsort(chi2)
    vectors: List[np.ndarray] = []
    seen = set()
    for idx in order:
        if not np.isfinite(chi2[int(idx)]):
            continue
        values = []
        missing = False
        for spec in context.parameter_specs:
            key = f"sample__{spec.name}"
            if key not in data:
                missing = True
                break
            values.append(float(np.asarray(data[key]).reshape(-1)[int(idx)]))
        if missing:
            return []
        vector = np.asarray(values, dtype=np.float64)
        fingerprint = tuple(np.round(vector, decimals=12).tolist())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        vectors.append(vector)
        if len(vectors) >= int(top_k):
            break
    return vectors


def load_restart_candidate_qs(
    *,
    path: str | Path,
    context: hmc31.FitContext,
    transform: Mapping[str, np.ndarray],
    q0: np.ndarray,
    top_k: int,
) -> List[np.ndarray]:
    if top_k <= 0 or not path:
        return []
    data = np.load(path, allow_pickle=True)
    if "q_candidates" not in data or "parameter_names" not in data:
        raise KeyError(f"{path} must contain q_candidates and parameter_names arrays.")
    q_old = np.asarray(data["q_candidates"], dtype=np.float64)
    names_old = [str(name) for name in np.asarray(data["parameter_names"]).tolist()]
    if q_old.ndim != 2 or q_old.shape[1] != len(names_old):
        raise ValueError(
            f"Malformed restart candidates in {path}: q_candidates shape={q_old.shape}, "
            f"n_parameter_names={len(names_old)}."
        )
    if "map_objective" in data:
        order = np.argsort(np.asarray(data["map_objective"], dtype=np.float64).reshape(-1))
        q_old = q_old[order]

    current_index = {spec.name: i for i, spec in enumerate(context.parameter_specs)}
    lower = np.asarray(transform["lower"], dtype=np.float64)
    upper = np.asarray(transform["upper"], dtype=np.float64)
    out: List[np.ndarray] = []
    seen = set()
    for row in q_old:
        q = np.asarray(q0, dtype=np.float64).copy()
        matched = 0
        for old_i, name in enumerate(names_old):
            new_i = current_index.get(name)
            if new_i is None:
                continue
            q[new_i] = row[old_i]
            matched += 1
        if matched == 0:
            continue
        q = np.clip(q, lower, upper)
        fingerprint = tuple(np.round(q, decimals=12).tolist())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(q)
        if len(out) >= int(top_k):
            break
    return out


def build_normalized_transform(
    specs: Sequence[hmc31.ParameterSpec],
    *,
    uniform_eps: float,
    normal_bound_sigma: float,
) -> dict:
    n = len(specs)
    is_uniform = np.asarray([spec.prior_kind == "uniform" for spec in specs], dtype=bool)
    is_normal = np.asarray([spec.prior_kind == "normal" for spec in specs], dtype=bool)
    loc = np.zeros(n, dtype=np.float64)
    scale = np.ones(n, dtype=np.float64)
    lower = np.full(n, -np.inf, dtype=np.float64)
    upper = np.full(n, np.inf, dtype=np.float64)
    for i, spec in enumerate(specs):
        if spec.prior_kind == "uniform":
            loc[i] = float(spec.prior_min)
            scale[i] = float(spec.prior_max) - float(spec.prior_min)
            lower[i] = float(uniform_eps)
            upper[i] = 1.0 - float(uniform_eps)
        elif spec.prior_kind == "normal":
            if spec.prior_mean is None or spec.prior_sigma is None:
                raise ValueError(f"Gaussian prior for {spec.name} is missing mean/sigma.")
            loc[i] = float(spec.prior_mean)
            scale[i] = float(spec.prior_sigma)
            if normal_bound_sigma > 0.0:
                lower[i] = -float(normal_bound_sigma)
                upper[i] = float(normal_bound_sigma)
        else:
            raise ValueError(f"Unknown prior kind {spec.prior_kind!r} for {spec.name}.")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        bad = [specs[int(i)].name for i in np.where((~np.isfinite(scale)) | (scale <= 0.0))[0]]
        raise ValueError(f"Invalid prior scale for parameters: {bad}")
    return {
        "is_uniform": is_uniform,
        "is_normal": is_normal,
        "loc": loc,
        "scale": scale,
        "lower": lower,
        "upper": upper,
    }


def physical_to_normalized(x: np.ndarray, transform: Mapping[str, np.ndarray]) -> np.ndarray:
    q = (np.asarray(x, dtype=np.float64) - transform["loc"]) / transform["scale"]
    return np.clip(q, transform["lower"], transform["upper"])


def normalized_to_physical_jax(q: jnp.ndarray, transform: Mapping[str, np.ndarray]) -> jnp.ndarray:
    loc = jnp.asarray(transform["loc"], dtype=jnp.float64)
    scale = jnp.asarray(transform["scale"], dtype=jnp.float64)
    lower = jnp.asarray(transform["lower"], dtype=jnp.float64)
    upper = jnp.asarray(transform["upper"], dtype=jnp.float64)
    q_clipped = jnp.clip(jnp.asarray(q, dtype=jnp.float64), lower, upper)
    return loc + scale * q_clipped


def normalized_to_physical_np(q: np.ndarray, transform: Mapping[str, np.ndarray]) -> np.ndarray:
    q_clipped = np.clip(np.asarray(q, dtype=np.float64), transform["lower"], transform["upper"])
    return np.asarray(transform["loc"], dtype=np.float64) + np.asarray(transform["scale"], dtype=np.float64) * q_clipped


def sample_from_vector(context: hmc31.FitContext, vector: np.ndarray | jnp.ndarray) -> Dict[str, float]:
    vec = np.asarray(vector, dtype=np.float64)
    return {spec.name: float(vec[i]) for i, spec in enumerate(context.parameter_specs)}


def objective_parts_from_physical(context: hmc31.FitContext, vector: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    chi2 = hmc31.parameter_vector_chi2(context, vector)
    penalty = []
    for i, spec in enumerate(context.parameter_specs):
        if spec.prior_kind == "normal":
            if spec.prior_mean is None or spec.prior_sigma is None:
                raise ValueError(f"Gaussian prior for {spec.name} is missing mean/sigma.")
            penalty.append(((vector[i] - float(spec.prior_mean)) / float(spec.prior_sigma)) ** 2)
    prior_penalty = jnp.sum(jnp.asarray(penalty, dtype=jnp.float64)) if penalty else jnp.asarray(0.0, dtype=jnp.float64)
    return chi2 + prior_penalty, chi2, prior_penalty


def make_objective(context: hmc31.FitContext, transform: Mapping[str, np.ndarray]):
    is_normal = jnp.asarray(transform["is_normal"], dtype=bool)

    def objective(q: jnp.ndarray) -> jnp.ndarray:
        physical = normalized_to_physical_jax(q, transform)
        chi2 = hmc31.parameter_vector_chi2(context, physical)
        prior_penalty = jnp.sum(jnp.where(is_normal, q**2, 0.0))
        return chi2 + prior_penalty

    return objective


def evaluate_q(context: hmc31.FitContext, transform: Mapping[str, np.ndarray], q: np.ndarray) -> dict:
    physical = normalized_to_physical_jax(jnp.asarray(q, dtype=jnp.float64), transform)
    map_objective, chi2, prior_penalty = objective_parts_from_physical(context, physical)
    chi2 = float(np.asarray(chi2))
    prior_penalty = float(np.asarray(prior_penalty))
    map_objective = float(np.asarray(map_objective))
    n_modes = int(context.likelihood.rank)
    n_params = len(context.parameter_specs)
    dof = max(n_modes - n_params, 1)
    return {
        "map_objective": map_objective,
        "chi2": chi2,
        "prior_penalty": prior_penalty,
        "reduced_chi2": chi2 / float(dof),
        "chi2_per_mode": chi2 / max(float(n_modes), 1.0),
        "chi2_dof": dof,
        "chi2_n_modes": n_modes,
        "n_fit_parameters": n_params,
        "physical": np.asarray(physical, dtype=np.float64),
    }


class BestTracker:
    def __init__(
        self,
        *,
        context: hmc31.FitContext,
        transform: Mapping[str, np.ndarray],
        output_dir: Path,
        suffix: str,
    ) -> None:
        self.context = context
        self.transform = transform
        self.output_dir = output_dir
        self.suffix = suffix
        self.trace_path = output_dir / f"map_trace_{suffix}.jsonl"
        self.best: Optional[dict] = None
        self.n_eval = 0
        self.last_q: Optional[np.ndarray] = None
        self.last_value: Optional[float] = None
        self.last_grad_norm: Optional[float] = None

    def record_eval(self, q: np.ndarray, value: float, grad: np.ndarray) -> None:
        self.n_eval += 1
        self.last_q = np.asarray(q, dtype=np.float64).copy()
        self.last_value = float(value)
        self.last_grad_norm = float(np.linalg.norm(np.asarray(grad, dtype=np.float64)))

    def record_population_eval(self, q_pop: np.ndarray, values: np.ndarray, grads: np.ndarray) -> int:
        values_np = np.asarray(values, dtype=np.float64).reshape(-1)
        grads_np = np.asarray(grads, dtype=np.float64)
        q_np = np.asarray(q_pop, dtype=np.float64)
        self.n_eval += int(values_np.size)
        idx = int(np.nanargmin(values_np))
        self.last_q = q_np[idx].copy()
        self.last_value = float(values_np[idx])
        self.last_grad_norm = float(np.linalg.norm(grads_np[idx]))
        return idx

    def maybe_update(self, q: np.ndarray, *, track: int, stage: str, iteration: int, force: bool = False) -> dict:
        stats = evaluate_q(self.context, self.transform, q)
        improved = self.best is None or stats["map_objective"] < self.best["map_objective"]
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "track": int(track),
            "stage": stage,
            "iteration": int(iteration),
            "n_eval": int(self.n_eval),
            "improved": bool(improved),
            "map_objective": stats["map_objective"],
            "chi2": stats["chi2"],
            "prior_penalty": stats["prior_penalty"],
            "reduced_chi2": stats["reduced_chi2"],
            "grad_norm": self.last_grad_norm,
        }
        with open(self.trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(gmt.to_jsonable(record)) + "\n")
        if improved or force:
            if improved:
                self.best = {
                    **stats,
                    "q": np.asarray(q, dtype=np.float64).copy(),
                    "track": int(track),
                    "stage": stage,
                    "iteration": int(iteration),
                    "n_eval": int(self.n_eval),
                }
                self.write_best_snapshot()
            log_status(
                f"[map] track={track} {stage} iter={iteration} chi2={stats['chi2']:.8e} "
                f"prior={stats['prior_penalty']:.5g} map={stats['map_objective']:.8e} "
                f"red_chi2={stats['reduced_chi2']:.6g} improved={improved}"
            )
        return stats

    def maybe_update_fast(
        self,
        q: np.ndarray,
        map_objective: float,
        *,
        track: int,
        stage: str,
        iteration: int,
        force: bool = False,
    ) -> None:
        q_np = np.asarray(q, dtype=np.float64)
        map_value = float(map_objective)
        improved = self.best is None or map_value < float(self.best["map_objective"])
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "track": int(track),
            "stage": stage,
            "iteration": int(iteration),
            "n_eval": int(self.n_eval),
            "improved": bool(improved),
            "map_objective": map_value,
            "chi2": None,
            "prior_penalty": None,
            "reduced_chi2": None,
            "grad_norm": self.last_grad_norm,
        }
        with open(self.trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(gmt.to_jsonable(record)) + "\n")
        if improved or force:
            if improved:
                self.best = {
                    "map_objective": map_value,
                    "chi2": float("nan"),
                    "prior_penalty": float("nan"),
                    "reduced_chi2": float("nan"),
                    "chi2_per_mode": float("nan"),
                    "chi2_dof": max(int(self.context.likelihood.rank) - len(self.context.parameter_specs), 1),
                    "chi2_n_modes": int(self.context.likelihood.rank),
                    "n_fit_parameters": len(self.context.parameter_specs),
                    "physical": normalized_to_physical_np(q_np, self.transform),
                    "q": q_np.copy(),
                    "track": int(track),
                    "stage": stage,
                    "iteration": int(iteration),
                    "n_eval": int(self.n_eval),
                }
                self.write_best_snapshot()
            log_status(
                f"[map] track={track} {stage} iter={iteration} map={map_value:.8e} "
                f"grad_norm={self.last_grad_norm} improved={improved}"
            )

    def write_best_snapshot(self) -> None:
        if self.best is None:
            return
        sample = sample_from_vector(self.context, self.best["physical"])
        config = hmc31.apply_sample_to_config(self.context.config, self.context.parameter_specs, sample)
        path = self.output_dir / f"bestfit_params_{self.suffix}_latest.yaml"
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(gmt.to_jsonable(config["params"]), handle, sort_keys=False)


def initialize_population(
    *,
    q0: np.ndarray,
    restart_qs: Sequence[np.ndarray] = (),
    hmc_vectors: Sequence[np.ndarray],
    transform: Mapping[str, np.ndarray],
    population_size: int,
    perturb_scale: float,
    seed: int,
) -> np.ndarray:
    lower = np.asarray(transform["lower"], dtype=np.float64)
    upper = np.asarray(transform["upper"], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    starts: List[np.ndarray] = []
    for q in restart_qs:
        q_arr = np.clip(np.asarray(q, dtype=np.float64), lower, upper)
        if q_arr.shape != q0.shape:
            continue
        if not any(np.allclose(q_arr, existing, rtol=0.0, atol=1.0e-10) for existing in starts):
            starts.append(q_arr)
        if len(starts) >= int(population_size):
            break
    q0_clipped = np.clip(np.asarray(q0, dtype=np.float64), lower, upper)
    if len(starts) < int(population_size) and not any(
        np.allclose(q0_clipped, existing, rtol=0.0, atol=1.0e-10) for existing in starts
    ):
        starts.append(q0_clipped)
    for vector in hmc_vectors:
        q = physical_to_normalized(np.asarray(vector, dtype=np.float64), transform)
        if not any(np.allclose(q, existing, rtol=0.0, atol=1.0e-10) for existing in starts):
            starts.append(q)
        if len(starts) >= int(population_size):
            break
    span = np.where(np.isfinite(upper - lower), upper - lower, 1.0)
    while len(starts) < int(population_size):
        perturb = rng.normal(0.0, float(perturb_scale), size=q0.shape) * np.minimum(span, 1.0)
        starts.append(np.clip(q0 + perturb, lower, upper))
    return np.asarray(starts[: int(population_size)], dtype=np.float64)


def make_population_value_and_grad(objective, max_batch_size: int):
    full_value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(objective)))
    if int(max_batch_size) <= 0:
        return full_value_and_grad

    batch_size = int(max_batch_size)

    def value_and_grad_chunked(q_pop: jnp.ndarray):
        q_pop = jnp.asarray(q_pop, dtype=jnp.float64)
        values = []
        grads = []
        for start in range(0, int(q_pop.shape[0]), batch_size):
            v_chunk, g_chunk = full_value_and_grad(q_pop[start : start + batch_size])
            values.append(v_chunk)
            grads.append(g_chunk)
        return jnp.concatenate(values, axis=0), jnp.concatenate(grads, axis=0)

    return value_and_grad_chunked


def make_adam_optimizer(
    *,
    lr: float,
    steps: int,
    schedule: str,
    min_lr_fraction: float,
    warmup_steps: int,
):
    schedule_name = str(schedule).strip().lower().replace("-", "_")
    base_lr = float(lr)
    min_fraction = float(np.clip(min_lr_fraction, 0.0, 1.0))
    n_steps = max(int(steps), 1)
    if schedule_name in ("constant", "none"):
        return optax.adam(base_lr)
    if schedule_name not in ("cosine", "warmup_cosine", "linear"):
        raise ValueError(
            "--population-lr-schedule/--adam-lr-schedule must be one of "
            "constant, cosine, warmup_cosine, or linear."
        )
    if schedule_name == "warmup_cosine" and int(warmup_steps) <= 0:
        warmup = min(max(n_steps // 20, 1), 100)
    else:
        warmup = max(int(warmup_steps), 0)
    warmup = min(warmup, n_steps)
    decay_steps = max(n_steps - warmup, 1)

    def lr_schedule(count):
        step = jnp.asarray(count, dtype=jnp.float64)
        progress_all = jnp.clip(step / float(n_steps), 0.0, 1.0)
        if schedule_name == "linear":
            factor = 1.0 - (1.0 - min_fraction) * progress_all
            return base_lr * factor

        progress_decay = jnp.clip((step - float(warmup)) / float(decay_steps), 0.0, 1.0)
        cosine_factor = min_fraction + (1.0 - min_fraction) * 0.5 * (
            1.0 + jnp.cos(jnp.pi * progress_decay)
        )
        decay_lr = base_lr * cosine_factor
        if warmup <= 0:
            return decay_lr
        warmup_progress = jnp.clip(step / float(warmup), 0.0, 1.0)
        warmup_lr = base_lr * (min_fraction + (1.0 - min_fraction) * warmup_progress)
        return jnp.where(step < warmup, warmup_lr, decay_lr)

    return optax.adam(lr_schedule)


def run_population_adam(
    *,
    q_pop0: np.ndarray,
    batched_value_and_grad,
    bounds: Tuple[np.ndarray, np.ndarray],
    tracker: BestTracker,
    steps: int,
    lr: float,
    lr_schedule: str,
    lr_min_fraction: float,
    lr_warmup_steps: int,
    log_every: int,
) -> Tuple[np.ndarray, np.ndarray]:
    q_pop = jnp.asarray(q_pop0, dtype=jnp.float64)
    lower, upper = bounds
    lower_j = jnp.asarray(lower, dtype=jnp.float64)
    upper_j = jnp.asarray(upper, dtype=jnp.float64)
    values0, grads0 = batched_value_and_grad(q_pop)
    values0_np = np.asarray(values0, dtype=np.float64)
    best_idx = tracker.record_population_eval(np.asarray(q_pop), values0_np, np.asarray(grads0))
    tracker.maybe_update_fast(
        np.asarray(q_pop)[best_idx],
        float(values0_np[best_idx]),
        track=best_idx,
        stage="population_start",
        iteration=0,
    )
    if steps <= 0:
        return np.asarray(q_pop, dtype=np.float64), values0_np

    opt = make_adam_optimizer(
        lr=float(lr),
        steps=int(steps),
        schedule=str(lr_schedule),
        min_lr_fraction=float(lr_min_fraction),
        warmup_steps=int(lr_warmup_steps),
    )
    opt_state = opt.init(q_pop)
    values = values0
    grads = grads0
    for step in range(1, int(steps) + 1):
        if step > 1:
            values, grads = batched_value_and_grad(q_pop)
        values_np = np.asarray(values, dtype=np.float64)
        best_idx = tracker.record_population_eval(np.asarray(q_pop), values_np, np.asarray(grads))
        if step == 1 or step == steps or (log_every > 0 and step % log_every == 0):
            tracker.maybe_update_fast(
                np.asarray(q_pop)[best_idx],
                float(values_np[best_idx]),
                track=best_idx,
                stage="population_adam",
                iteration=step - 1,
            )
        updates, opt_state = opt.update(grads, opt_state, q_pop)
        q_pop = optax.apply_updates(q_pop, updates)
        q_pop = jnp.clip(q_pop, lower_j, upper_j)
    final_values, final_grads = batched_value_and_grad(q_pop)
    final_values_np = np.asarray(final_values, dtype=np.float64)
    best_idx = tracker.record_population_eval(np.asarray(q_pop), final_values_np, np.asarray(final_grads))
    tracker.maybe_update_fast(
        np.asarray(q_pop)[best_idx],
        float(final_values_np[best_idx]),
        track=best_idx,
        stage="population_final",
        iteration=int(steps),
        force=True,
    )
    return np.asarray(q_pop, dtype=np.float64), final_values_np


def run_population_benchmark(
    *,
    context: hmc31.FitContext,
    transform: Mapping[str, np.ndarray],
    q0: np.ndarray,
    hmc_vectors: Sequence[np.ndarray],
    output_dir: Path,
    suffix: str,
    sizes: Sequence[int],
    seed: int,
    perturb_scale: float,
    repeats: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    objective = make_objective(context, transform)
    batched_value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(objective)))
    jsonl_path = output_dir / f"population_benchmark_{suffix}.jsonl"
    summary_path = output_dir / f"population_benchmark_{suffix}.json"
    rows = []
    best_success = None
    log_status(f"[map] population benchmark sizes={list(sizes)} repeats={int(repeats)}")
    for size in sizes:
        row = {
            "population_size": int(size),
            "status": "started",
            "time_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(gmt.to_jsonable(row)) + "\n")
        try:
            q_candidates = initialize_population(
                q0=q0,
                restart_qs=(),
                hmc_vectors=hmc_vectors,
                transform=transform,
                population_size=int(size),
                perturb_scale=float(perturb_scale),
                seed=int(seed),
            )
            elapsed = []
            best_values = []
            for repeat in range(max(1, int(repeats))):
                t0 = time.time()
                values, grads = batched_value_and_grad(jnp.asarray(q_candidates, dtype=jnp.float64))
                values.block_until_ready()
                grads.block_until_ready()
                elapsed.append(time.time() - t0)
                values_np = np.asarray(values, dtype=np.float64)
                grads_np = np.asarray(grads, dtype=np.float64)
                best_values.append(float(np.nanmin(values_np)))
                finite_values = bool(np.all(np.isfinite(values_np)))
                finite_grads = bool(np.all(np.isfinite(grads_np)))
                if not finite_values or not finite_grads:
                    raise FloatingPointError(
                        f"non-finite benchmark output for population_size={size}: "
                        f"finite_values={finite_values} finite_grads={finite_grads}"
                    )
            row.update(
                status="success",
                seconds_first=float(elapsed[0]),
                seconds_repeats=[float(x) for x in elapsed],
                best_map_objective=float(best_values[-1]),
                min_seconds=float(np.min(elapsed)),
                max_seconds=float(np.max(elapsed)),
                median_seconds=float(np.median(elapsed)),
                n_parameters=len(context.parameter_specs),
                n_population=int(size),
            )
            best_success = int(size)
            log_status(
                f"[map] benchmark population_size={size} success "
                f"first_seconds={elapsed[0]:.3f} best_map={best_values[-1]:.8e}"
            )
        except BaseException as exc:
            row.update(
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                time_failed=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            rows.append(row)
            with open(jsonl_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(gmt.to_jsonable(row)) + "\n")
            log_status(f"[map] benchmark population_size={size} failed: {type(exc).__name__}: {exc}")
            break
        rows.append(row)
        with open(jsonl_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(gmt.to_jsonable(row)) + "\n")

    summary = {
        "sizes": list(int(x) for x in sizes),
        "best_success_population_size": best_success,
        "rows": rows,
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(gmt.to_jsonable(summary), handle, indent=2)
    log_status(f"[map] population benchmark done best_success_population_size={best_success}")
    return summary


def run_adam(
    *,
    q0: np.ndarray,
    value_and_grad,
    bounds: Tuple[np.ndarray, np.ndarray],
    tracker: BestTracker,
    track: int,
    steps: int,
    lr: float,
    lr_schedule: str,
    lr_min_fraction: float,
    lr_warmup_steps: int,
    log_every: int,
) -> np.ndarray:
    if steps <= 0:
        return np.asarray(q0, dtype=np.float64)
    lower, upper = bounds
    opt = make_adam_optimizer(
        lr=float(lr),
        steps=int(steps),
        schedule=str(lr_schedule),
        min_lr_fraction=float(lr_min_fraction),
        warmup_steps=int(lr_warmup_steps),
    )
    q = jnp.asarray(q0, dtype=jnp.float64)
    opt_state = opt.init(q)
    best_q = np.asarray(q0, dtype=np.float64)
    best_value = math.inf
    for step in range(1, int(steps) + 1):
        value, grad = value_and_grad(q)
        value_np = float(np.asarray(value))
        grad_np = np.asarray(grad, dtype=np.float64)
        q_np = np.asarray(q, dtype=np.float64)
        tracker.record_eval(q_np, value_np, grad_np)
        if value_np < best_value:
            best_value = value_np
            best_q = q_np.copy()
        if step == 1 or step == steps or (log_every > 0 and step % log_every == 0):
            tracker.maybe_update_fast(q_np, value_np, track=track, stage="adam", iteration=step)
        updates, opt_state = opt.update(grad, opt_state, q)
        q = optax.apply_updates(q, updates)
        q = jnp.clip(q, jnp.asarray(lower, dtype=jnp.float64), jnp.asarray(upper, dtype=jnp.float64))
    return best_q


def run_lbfgsb(
    *,
    q0: np.ndarray,
    value_and_grad,
    bounds: Tuple[np.ndarray, np.ndarray],
    tracker: BestTracker,
    track: int,
    maxiter: int,
    maxfun: int,
    ftol: float,
    gtol: float,
    maxls: int,
    log_every: int,
) -> dict:
    lower, upper = bounds
    scipy_bounds = list(zip(lower.tolist(), upper.tolist()))
    callback_iter = {"i": 0}

    def fun(q_np: np.ndarray):
        q = jnp.asarray(q_np, dtype=jnp.float64)
        value, grad = value_and_grad(q)
        value_np = float(np.asarray(value))
        grad_np = np.asarray(grad, dtype=np.float64)
        if not np.isfinite(value_np) or not np.all(np.isfinite(grad_np)):
            value_np = 1.0e100
            grad_np = np.zeros_like(q_np, dtype=np.float64)
        tracker.record_eval(q_np, value_np, grad_np)
        return value_np, grad_np

    def callback(q_np: np.ndarray) -> None:
        callback_iter["i"] += 1
        iteration = callback_iter["i"]
        if iteration == 1 or iteration % max(1, int(log_every)) == 0:
            map_value = tracker.last_value if tracker.last_value is not None else float("inf")
            tracker.maybe_update_fast(
                np.asarray(q_np, dtype=np.float64),
                float(map_value),
                track=track,
                stage="lbfgsb",
                iteration=iteration,
            )

    result = minimize(
        fun,
        np.asarray(q0, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=scipy_bounds,
        callback=callback,
        options={
            "maxiter": int(maxiter),
            "maxfun": int(maxfun),
            "ftol": float(ftol),
            "gtol": float(gtol),
            "maxls": int(maxls),
            "disp": False,
        },
    )
    tracker.maybe_update_fast(
        np.asarray(result.x, dtype=np.float64),
        float(result.fun),
        track=track,
        stage="lbfgsb_final",
        iteration=callback_iter["i"],
        force=True,
    )
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "fun": float(result.fun),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "x": np.asarray(result.x, dtype=np.float64),
    }


def write_outputs(
    *,
    context: hmc31.FitContext,
    transform: Mapping[str, np.ndarray],
    best: Mapping[str, object],
    output_dir: Path,
    suffix: str,
    start_info: Mapping[str, object],
    optimizer_results: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
) -> dict:
    sample = sample_from_vector(context, best["physical"])
    best_config = hmc31.apply_sample_to_config(context.config, context.parameter_specs, sample)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_params_path = output_dir / f"bestfit_params_{suffix}.yaml"
    with open(best_params_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(gmt.to_jsonable(best_config["params"]), handle, sort_keys=False)

    models = hmc31.build_models_from_sample(context, sample)
    theory_cls = hmc31.extract_theory_cls_jax_from_models(models)
    active_theory = np.asarray(hmc31.theory_data_vector_jax(context.likelihood, theory_cls))
    active_measurement = hmc31.measurement_for_plots(context)
    full_likelihood = hmc31.full_likelihood_for_plots(context)
    full_measurement = hmc31.measurement_from_likelihood(context, full_likelihood)
    full_theory = np.asarray(hmc31.theory_data_vector_jax(full_likelihood, theory_cls))

    chi2 = float(np.asarray(hmc31.whitened_chi2(context.likelihood, jnp.asarray(active_theory, dtype=jnp.float64))))
    n_modes = int(context.likelihood.rank)
    n_params = len(context.parameter_specs)
    chi2_dof = max(n_modes - n_params, 1)
    reduced_chi2 = chi2 / float(chi2_dof)
    q_best = np.asarray(best["q"], dtype=np.float64)
    prior_penalty = float(np.sum(np.where(np.asarray(transform["is_normal"], dtype=bool), q_best**2, 0.0)))
    map_objective = chi2 + prior_penalty

    active_vector_path = output_dir / f"bestfit_theory_data_vector_{suffix}.npz"
    np.savez_compressed(
        active_vector_path,
        ell_band=np.asarray(active_measurement.ell),
        data_vector=np.asarray(active_measurement.data_vector),
        theory_vector=active_theory,
        covariance=np.asarray(active_measurement.covariance),
        spectrum_names=np.asarray(active_measurement.names),
        best_sample_json=np.asarray(json.dumps(sample)),
        best_whitened_chi2=np.asarray(chi2),
        best_prior_penalty=np.asarray(prior_penalty),
        best_map_objective=np.asarray(map_objective),
    )

    full_vector_path = output_dir / f"bestfit_full_theory_data_vector_{suffix}.npz"
    np.savez_compressed(
        full_vector_path,
        ell_band=np.asarray(full_measurement.ell),
        data_vector=np.asarray(full_measurement.data_vector),
        theory_vector=full_theory,
        covariance=np.asarray(full_measurement.covariance),
        spectrum_names=np.asarray(full_measurement.names),
        best_sample_json=np.asarray(json.dumps(sample)),
        best_whitened_chi2=np.asarray(chi2),
        best_prior_penalty=np.asarray(prior_penalty),
        best_map_objective=np.asarray(map_objective),
        likelihood_bestfit_theory_vector=np.asarray(str(active_vector_path)),
    )

    plot_ell_max = None if args.plot_ell_max is not None and args.plot_ell_max <= 0.0 else float(args.plot_ell_max)
    plot_xlim = parse_pair(args.plot_xlim, option="--plot-xlim")
    residual_ylim = parse_pair(args.residual_ylim, option="--residual-ylim")

    dell_pdf = output_dir / f"map_bestfit_full_dell_comparison_{suffix}.pdf"
    dell_paths = gmt.plot_family_dell_comparisons(
        full_measurement,
        full_theory,
        output_dir,
        pdf_path=dell_pdf,
        filename_prefix=f"map_bestfit_full_dell_{suffix}",
        ell_max=plot_ell_max,
        ksz_ylim=parse_pair(args.ksz_ylim, option="--ksz-ylim"),
        ksz_scale=float(args.ksz_scale),
        active_band_indices=hmc31.likelihood_active_band_indices(context),
        total_reduced_chi2=reduced_chi2,
        chi2_dof=chi2_dof,
        xscale=str(args.plot_xscale),
        xlim=plot_xlim,
    )
    residual_pdf = output_dir / f"map_bestfit_full_dell_residuals_{suffix}.pdf"
    residual_paths = gmt.plot_family_dell_residual_comparisons(
        full_measurement,
        full_theory,
        output_dir,
        pdf_path=residual_pdf,
        filename_prefix=f"map_bestfit_full_dell_residuals_{suffix}",
        ell_max=plot_ell_max,
        ksz_scale=1.0,
        active_band_indices=hmc31.likelihood_active_band_indices(context),
        total_reduced_chi2=reduced_chi2,
        chi2_dof=chi2_dof,
        xscale=str(args.plot_xscale),
        xlim=plot_xlim,
        ylim=residual_ylim,
    )

    bounds_table = []
    for i, spec in enumerate(context.parameter_specs):
        bounds_table.append(
            {
                "name": spec.name,
                "prior_kind": spec.prior_kind,
                "q_best": float(np.asarray(best["q"])[i]),
                "value_best": float(np.asarray(best["physical"])[i]),
                "q_lower": float(transform["lower"][i]) if np.isfinite(transform["lower"][i]) else None,
                "q_upper": float(transform["upper"][i]) if np.isfinite(transform["upper"][i]) else None,
                "prior_min": spec.prior_min if np.isfinite(spec.prior_min) else None,
                "prior_max": spec.prior_max if np.isfinite(spec.prior_max) else None,
                "prior_mean": spec.prior_mean,
                "prior_sigma": spec.prior_sigma,
            }
        )

    summary = {
        "config": str(args.config),
        "start_params": str(start_info["params_path"]),
        "output_dir": str(output_dir),
        "suffix": suffix,
        "optimizer": {
            "method": "vectorized population Adam plus normalized bounded L-BFGS-B polish",
            "population_size": int(args.population_size),
            "population_steps": int(args.population_steps),
            "population_lr": float(args.population_lr),
            "population_lr_schedule": str(args.population_lr_schedule),
            "population_lr_min_fraction": float(args.population_lr_min_fraction),
            "population_lr_warmup_steps": int(args.population_lr_warmup_steps),
            "population_eval_batch_size": int(args.population_eval_batch_size),
            "restart_candidates": str(args.restart_candidates) if args.restart_candidates else None,
            "restart_top_k": int(args.restart_top_k),
            "hmc_top_k": int(args.hmc_top_k),
            "polish_top_k": int(args.polish_top_k),
            "adam_steps": int(args.adam_steps),
            "adam_lr": float(args.adam_lr),
            "adam_lr_schedule": str(args.adam_lr_schedule),
            "adam_lr_min_fraction": float(args.adam_lr_min_fraction),
            "adam_lr_warmup_steps": int(args.adam_lr_warmup_steps),
            "n_restarts": int(args.n_restarts),
            "perturb_scale": float(args.perturb_scale),
            "normal_bound_sigma": float(args.normal_bound_sigma),
            "uniform_eps": float(args.uniform_eps),
            "lbfgsb_maxiter": int(args.lbfgsb_maxiter),
            "lbfgsb_maxfun": int(args.lbfgsb_maxfun),
            "lbfgsb_ftol": float(args.lbfgsb_ftol),
            "lbfgsb_gtol": float(args.lbfgsb_gtol),
            "lbfgsb_maxls": int(args.lbfgsb_maxls),
        },
        "start": gmt.to_jsonable(start_info),
        "best": {
            "track": int(best["track"]),
            "stage": str(best["stage"]),
            "iteration": int(best["iteration"]),
            "n_eval": int(best["n_eval"]),
            "best_whitened_chi2": chi2,
            "best_prior_penalty": prior_penalty,
            "best_map_objective": map_objective,
            "best_reduced_chi2": reduced_chi2,
            "best_chi2_dof": chi2_dof,
            "best_chi2_per_mode": chi2 / max(float(n_modes), 1.0),
            "chi2_n_modes": n_modes,
            "n_fit_parameters": n_params,
        },
        "optimizer_results": gmt.to_jsonable(list(optimizer_results)),
        "paths": {
            "bestfit_params": str(best_params_path),
            "active_theory_vector": str(active_vector_path),
            "full_theory_vector": str(full_vector_path),
            "full_dell_pdf": str(dell_pdf),
            "full_residual_pdf": str(residual_pdf),
            "trace_jsonl": str(output_dir / f"map_trace_{suffix}.jsonl"),
        },
        "plot_paths": {
            "dell": [str(path) for path in dell_paths],
            "residuals": [str(path) for path in residual_paths],
        },
        "static_summary": hmc31.static_summary(context),
        "parameter_specs": hmc31.parameter_specs_jsonable(context.parameter_specs),
        "normalized_bounds": bounds_table,
    }
    summary_path = output_dir / f"map_summary_{suffix}.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(gmt.to_jsonable(summary), handle, indent=2)
    summary["paths"]["summary"] = str(summary_path)
    log_status(f"[map] saved outputs summary={summary_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=hmc31.DEFAULT_STAGE31_CONFIG)
    parser.add_argument("--params", default=None, help="Starting params YAML. Defaults to latest bestfit below --hmc-run-dir.")
    parser.add_argument("--hmc-run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--population-steps", type=int, default=40)
    parser.add_argument("--population-lr", type=float, default=1.0e-3)
    parser.add_argument("--population-lr-schedule", default="constant", choices=("constant", "cosine", "warmup_cosine", "linear"))
    parser.add_argument("--population-lr-min-fraction", type=float, default=0.1)
    parser.add_argument("--population-lr-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--population-eval-batch-size",
        type=int,
        default=0,
        help="Evaluate population gradients in chunks of this size. 0 uses one full population batch.",
    )
    parser.add_argument("--restart-candidates", default=None, help="NPZ with q_candidates and parameter_names.")
    parser.add_argument("--restart-top-k", type=int, default=0)
    parser.add_argument("--hmc-top-k", type=int, default=8)
    parser.add_argument("--polish-top-k", type=int, default=2)
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--perturb-scale", type=float, default=0.03)
    parser.add_argument("--adam-steps", type=int, default=0)
    parser.add_argument("--adam-lr", type=float, default=2.0e-3)
    parser.add_argument("--adam-lr-schedule", default="constant", choices=("constant", "cosine", "warmup_cosine", "linear"))
    parser.add_argument("--adam-lr-min-fraction", type=float, default=0.1)
    parser.add_argument("--adam-lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lbfgsb-maxiter", type=int, default=120)
    parser.add_argument("--lbfgsb-maxfun", type=int, default=180)
    parser.add_argument("--lbfgsb-ftol", type=float, default=1.0e-7)
    parser.add_argument("--lbfgsb-gtol", type=float, default=1.0e-4)
    parser.add_argument("--lbfgsb-maxls", type=int, default=20)
    parser.add_argument("--normal-bound-sigma", type=float, default=8.0)
    parser.add_argument("--uniform-eps", type=float, default=1.0e-6)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--plot-ell-max", type=float, default=3000.0)
    parser.add_argument("--plot-xscale", default="log", choices=("linear", "log", "symlog"))
    parser.add_argument("--plot-xlim", default="100,3000")
    parser.add_argument("--residual-ylim", default=None)
    parser.add_argument("--ksz-scale", type=float, default=1.0e3)
    parser.add_argument("--ksz-ylim", default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-population-sizes", default="1,2,4,8")
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_status(f"[map] preparing context config={args.config}")
    context = hmc31.prepare_fit_context(args.config)
    params_path = Path(args.params) if args.params else latest_hmc_bestfit(args.hmc_run_dir)
    log_status(f"[map] starting params={params_path}")
    start_sample = hmc31.pack_sample_from_params_file(context, params_path)
    start_vector = vector_from_sample(context, start_sample)
    transform = build_normalized_transform(
        context.parameter_specs,
        uniform_eps=float(args.uniform_eps),
        normal_bound_sigma=float(args.normal_bound_sigma),
    )
    q0 = physical_to_normalized(start_vector, transform)

    hmc_chain_path = None
    hmc_vectors: List[np.ndarray] = []
    restart_qs: List[np.ndarray] = []
    if args.restart_candidates:
        restart_qs = load_restart_candidate_qs(
            path=args.restart_candidates,
            context=context,
            transform=transform,
            q0=q0,
            top_k=int(args.restart_top_k),
        )
        log_status(f"[map] loaded {len(restart_qs)} restart candidates from {args.restart_candidates}")
    try:
        hmc_chain_path = latest_hmc_chain_path(args.hmc_run_dir)
        hmc_vectors = top_hmc_vectors(context, hmc_chain_path, int(args.hmc_top_k))
        log_status(f"[map] loaded {len(hmc_vectors)} HMC population starts from {hmc_chain_path}")
    except Exception as exc:
        log_status(f"[map] HMC population starts unavailable: {type(exc).__name__}: {exc}")

    if args.benchmark_only:
        run_population_benchmark(
            context=context,
            transform=transform,
            q0=q0,
            hmc_vectors=hmc_vectors,
            output_dir=output_dir,
            suffix=args.suffix,
            sizes=parse_int_list(args.benchmark_population_sizes, option="--benchmark-population-sizes"),
            seed=int(args.seed),
            perturb_scale=float(args.perturb_scale),
            repeats=int(args.benchmark_repeats),
        )
        return 0

    start_stats = evaluate_q(context, transform, q0)
    start_info = {
        "params_path": str(params_path),
        "map_objective": start_stats["map_objective"],
        "chi2": start_stats["chi2"],
        "prior_penalty": start_stats["prior_penalty"],
        "reduced_chi2": start_stats["reduced_chi2"],
        "chi2_dof": start_stats["chi2_dof"],
        "chi2_n_modes": start_stats["chi2_n_modes"],
        "n_fit_parameters": start_stats["n_fit_parameters"],
        "q_clipped_from_start": bool(np.any(np.abs(q0 - ((start_vector - transform["loc"]) / transform["scale"])) > 0.0)),
    }
    log_status(
        f"[map] start chi2={start_info['chi2']:.8e} prior={start_info['prior_penalty']:.5g} "
        f"map={start_info['map_objective']:.8e} red_chi2={start_info['reduced_chi2']:.6g}"
    )
    if args.validate_only:
        summary_path = output_dir / f"map_validate_{args.suffix}.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(gmt.to_jsonable({"start": start_info, "static_summary": hmc31.static_summary(context)}), handle, indent=2)
        log_status(f"[map] validate-only wrote {summary_path}")
        return 0
    start_info["hmc_chain_path"] = str(hmc_chain_path) if hmc_chain_path is not None else None
    start_info["n_hmc_population_starts"] = int(len(hmc_vectors))
    start_info["restart_candidates_path"] = str(args.restart_candidates) if args.restart_candidates else None
    start_info["n_restart_candidate_starts"] = int(len(restart_qs))

    objective = make_objective(context, transform)
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    batched_value_and_grad = make_population_value_and_grad(objective, int(args.population_eval_batch_size))
    bounds = (np.asarray(transform["lower"], dtype=np.float64), np.asarray(transform["upper"], dtype=np.float64))
    tracker = BestTracker(context=context, transform=transform, output_dir=output_dir, suffix=args.suffix)
    tracker.n_eval = 1
    tracker.best = {
        **start_stats,
        "q": q0.copy(),
        "track": 0,
        "stage": "start",
        "iteration": 0,
        "n_eval": tracker.n_eval,
    }
    tracker.write_best_snapshot()
    with open(tracker.trace_path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                gmt.to_jsonable(
                    {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "track": 0,
                        "stage": "start",
                        "iteration": 0,
                        "n_eval": tracker.n_eval,
                        "improved": True,
                        "map_objective": start_stats["map_objective"],
                        "chi2": start_stats["chi2"],
                        "prior_penalty": start_stats["prior_penalty"],
                        "reduced_chi2": start_stats["reduced_chi2"],
                        "grad_norm": None,
                    }
                )
            )
            + "\n"
        )

    rng = np.random.default_rng(int(args.seed))
    optimizer_results = []

    q_candidates = initialize_population(
        q0=q0,
        restart_qs=restart_qs,
        hmc_vectors=hmc_vectors,
        transform=transform,
        population_size=max(1, int(args.population_size)),
        perturb_scale=float(args.perturb_scale),
        seed=int(args.seed),
    )
    if int(args.population_size) > 1 or int(args.population_steps) > 0:
        log_status(
            f"[map] population begin size={q_candidates.shape[0]} steps={int(args.population_steps)} "
            f"lr={float(args.population_lr):g} schedule={args.population_lr_schedule}"
        )
        q_candidates, candidate_values = run_population_adam(
            q_pop0=q_candidates,
            batched_value_and_grad=batched_value_and_grad,
            bounds=bounds,
            tracker=tracker,
            steps=int(args.population_steps),
            lr=float(args.population_lr),
            lr_schedule=str(args.population_lr_schedule),
            lr_min_fraction=float(args.population_lr_min_fraction),
            lr_warmup_steps=int(args.population_lr_warmup_steps),
            log_every=int(args.log_every),
        )
        candidate_values = np.asarray(candidate_values, dtype=np.float64)
        if tracker.best is not None and "q" in tracker.best:
            tracked_best_q = np.asarray(tracker.best["q"], dtype=np.float64)
            tracked_best_value = float(tracker.best["map_objective"])
            already_present = any(
                np.allclose(tracked_best_q, row, rtol=0.0, atol=1.0e-10) for row in np.asarray(q_candidates)
            )
            if np.isfinite(tracked_best_value) and not already_present:
                q_candidates = np.vstack([np.asarray(q_candidates, dtype=np.float64), tracked_best_q[None, :]])
                candidate_values = np.concatenate([candidate_values, np.asarray([tracked_best_value])])
                log_status("[map] added tracked best Adam point to polish candidate pool")
        order = np.argsort(candidate_values)
        q_candidates = q_candidates[order]
        candidate_values = candidate_values[order]
        population_candidates_path = output_dir / f"population_candidates_{args.suffix}.npz"
        np.savez_compressed(
            population_candidates_path,
            q_candidates=q_candidates,
            map_objective=candidate_values,
            parameter_names=np.asarray([spec.name for spec in context.parameter_specs]),
        )
        optimizer_results.append(
            {
                "stage": "population_adam",
                "population_size": int(args.population_size),
                "polish_candidate_pool_size": int(q_candidates.shape[0]),
                "steps": int(args.population_steps),
                "best_final_map_objective": float(candidate_values[0]),
                "worst_final_map_objective": float(candidate_values[-1]),
                "population_candidates": str(population_candidates_path),
            }
        )

    polish_starts = [q_candidates[i].copy() for i in range(min(int(args.polish_top_k), q_candidates.shape[0]))]
    while len(polish_starts) < int(args.n_restarts):
        span = np.where(np.isfinite(bounds[1] - bounds[0]), bounds[1] - bounds[0], 1.0)
        perturb = rng.normal(0.0, float(args.perturb_scale), size=q0.shape) * np.minimum(span, 1.0)
        polish_starts.append(np.clip(q0 + perturb, bounds[0], bounds[1]))

    for track, q_start in enumerate(polish_starts):
        log_status(f"[map] track={track} begin")
        q_after_adam = run_adam(
            q0=q_start,
            value_and_grad=value_and_grad,
            bounds=bounds,
            tracker=tracker,
            track=track,
            steps=int(args.adam_steps),
            lr=float(args.adam_lr),
            lr_schedule=str(args.adam_lr_schedule),
            lr_min_fraction=float(args.adam_lr_min_fraction),
            lr_warmup_steps=int(args.adam_lr_warmup_steps),
            log_every=int(args.log_every),
        )
        result = run_lbfgsb(
            q0=q_after_adam,
            value_and_grad=value_and_grad,
            bounds=bounds,
            tracker=tracker,
            track=track,
            maxiter=int(args.lbfgsb_maxiter),
            maxfun=int(args.lbfgsb_maxfun),
            ftol=float(args.lbfgsb_ftol),
            gtol=float(args.lbfgsb_gtol),
            maxls=int(args.lbfgsb_maxls),
            log_every=int(args.log_every),
        )
        optimizer_results.append({k: v for k, v in result.items() if k != "x"})
        log_status(
            f"[map] track={track} done success={result['success']} status={result['status']} "
            f"nit={result['nit']} nfev={result['nfev']} fun={result['fun']:.8e}"
        )

    if tracker.best is None:
        raise RuntimeError("Optimizer did not record any finite point.")
    summary = write_outputs(
        context=context,
        transform=transform,
        best=tracker.best,
        output_dir=output_dir,
        suffix=args.suffix,
        start_info=start_info,
        optimizer_results=optimizer_results,
        args=args,
    )
    log_status(
        f"[map] complete best_chi2={summary['best']['best_whitened_chi2']:.8e} "
        f"best_map={summary['best']['best_map_objective']:.8e} "
        f"best_red_chi2={summary['best']['best_reduced_chi2']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
