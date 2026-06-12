#!/usr/bin/env python
"""
run_hmc_vs_sbi_2pt.py

HMC (NumPyro NUTS) vs SBI (LtU-ILI NPE+MDN) comparison on 2pt Cl statistics.

HMC:
  - Forward model : analytical theory Cls via theory_sbi_utils / run_hmc_theory_cls
  - Covariance    : theory Gaussian covariance from the fiducial .npz product
  - Observation   : x_obs.npy sliced to the relevant 2pt probe indices

SBI:
  - Loads existing NPE+MDN posteriors from the main SBI pipeline where available,
    or trains new ones from x_train_full_noisy.npy / theta_train_full.npy
  - One posterior per probe: gy, gtau, gkappa, all_2pt
  - Normalisation: per-feature z-score (matching main SBI pipeline)
  - Observation   : x_obs.npy sliced + normalised with saved scalers

Output: one GetDist triangle plot per probe saved to WORK_DIR/
  hmc_vs_sbi_gy.pdf  hmc_vs_sbi_gtau.pdf  hmc_vs_sbi_gkappa.pdf  hmc_vs_sbi_all_2pt.pdf

Run from:
  /work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/sbi_Cls/
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle as pk
import sys
import threading
import time
import traceback
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# =============================================================================
# PATHS
# =============================================================================

WORK_DIR     = str(pathlib.Path(__file__).parent.resolve())
SBI_VALIDATE = '/work/hdd/bdne/aacharya2/GODMAX/notebooks/SBI_validate'
LTU_ILI_PATH = '/work/hdd/bdne/aacharya2/ltu-ili'

for _p in [SBI_VALIDATE, LTU_ILI_PATH, WORK_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from theory_sbi_utils import (
    DEFAULT_FIDUCIAL_PATH,
    default_parameter_specs,
    ensure_default_fiducial_product,
    fiducial_theta,
    make_inference_theory_vector_function,
    parse_probe_list,
    prior_bounds,
    selected_product_arrays,
    stable_cholesky,
    validate_theory_vector,
)
from run_hmc_theory_cls import run_hmc

from ili.dataloaders import StaticNumpyLoader
from ili.inference   import InferenceRunner
from ili.utils       import load_nde_sbi
from sbi.utils       import BoxUniform

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

# =============================================================================
# CONSTANTS  (matching main SBI pipeline exactly)
# =============================================================================

ADD_SURVEY_NOISE = True
_CACHE_SUFFIX    = '_noisy' if ADD_SURVEY_NOISE else ''

PRIOR_LOW    = [1.0, -0.3]
PRIOR_HIGH   = [6.0,  0.0]
PARAM_NAMES  = ['theta_ej_0', 'nu_theta_ej_M']
PARAM_LABELS = [r'$$\theta_{\rm ej,0}$$', r'$$\nu_{\theta_{\rm ej}}^{M}$$']
FIDUCIAL     = [2.0, -0.1]

LMIN       = 100
LMAX       = 1500
N_ELL_BINS = 20

# x_obs.npy ordering: [g2y | g2tau | g2kappa | gy | gtau | gkappa]
CL_SPECS_FULL  = ['g2y', 'g2tau', 'g2kappa', 'gy', 'gtau', 'gkappa']
ALL_2PT_PROBES = ['gy', 'gtau', 'gkappa']
INDIVIDUAL_STATS = {'gy', 'gtau', 'gkappa'}

VAL_FRACTION = 0.10

EQUAL_ARCH = {
    'hidden_features': 64,
    'num_components':  5,
    'learning_rate':   5e-4,
    'batch_size':      64,
    'max_num_epochs':  200,
    'repeats':         6,
}

HMC_NUM_WARMUP     = 3000
HMC_NUM_SAMPLES    = 2000
HMC_NUM_CHAINS     = 4
HMC_MAX_TREE_DEPTH = 10
HMC_DENSE_MASS     = True
HMC_TARGET_ACCEPT  = 0.9
HMC_CHAIN_METHOD   = 'vectorized'
HMC_SEED           = 42

SBI_N_SAMPLES = 4000

PROBE_TO_THEORY = {
    'gy':      ('gy',),
    'gtau':    ('gtau',),
    'gkappa':  ('gkappa',),
    'all_2pt': ('gy', 'gtau', 'gkappa'),
}

# =============================================================================
# ELL BINNING  (matching main pipeline exactly)
# =============================================================================

def make_ell_bins(lmin: int = LMIN, lmax: int = LMAX, n_bins: int = N_ELL_BINS):
    edges   = np.unique(np.logspace(np.log10(lmin), np.log10(lmax), n_bins + 1).astype(int))
    centres = 0.5 * (edges[:-1] + edges[1:]).astype(float)
    return edges, centres


ELL_EDGES, ELL_CENTRES = make_ell_bins()
N_ELL_BINS_ACTUAL = len(ELL_EDGES) - 1

# Slice map: label → list of column indices in the full x vector
_SLICE = {
    label: list(range(i * N_ELL_BINS_ACTUAL, (i + 1) * N_ELL_BINS_ACTUAL))
    for i, label in enumerate(CL_SPECS_FULL)
}

STAT_MAP_2PT = {
    'gy':      _SLICE['gy'],
    'gtau':    _SLICE['gtau'],
    'gkappa':  _SLICE['gkappa'],
    'all_2pt': _SLICE['gy'] + _SLICE['gtau'] + _SLICE['gkappa'],
}

# =============================================================================
# UTILITIES
# =============================================================================

def _fpath(work_dir: str, fname: str) -> str:
    return os.path.join(work_dir, fname)


def _clip_to_prior(samples: np.ndarray) -> np.ndarray:
    return np.clip(
        samples,
        a_min=np.array(PRIOR_LOW,  dtype=np.float32),
        a_max=np.array(PRIOR_HIGH, dtype=np.float32),
    )


def _print_sample_stats(tag: str, samples: np.ndarray) -> None:
    print(f'  [{tag}] theta_ej_0  = {samples[:, 0].mean():.3f} +/- {samples[:, 0].std():.3f}')
    print(f'  [{tag}] nu_theta_ej = {samples[:, 1].mean():.3f} +/- {samples[:, 1].std():.3f}')


def _slice_xobs(x_obs_full: np.ndarray, probes: tuple) -> np.ndarray:
    """Extract elements of x_obs_full for the given probes (in probe order)."""
    idx = []
    for probe in probes:
        idx.extend(_SLICE[probe])
    return x_obs_full[np.array(idx)]


def _normalise(x_obs_full: np.ndarray, x_obs_idx: list,
               x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    xo = x_obs_full[x_obs_idx].astype(np.float32)
    return (xo - x_mean) / np.where(x_std < 1e-10, 1.0, x_std)


def _make_blocks(n_features: int) -> list[list[int]]:
    return [
        list(range(i, min(i + N_ELL_BINS_ACTUAL, n_features)))
        for i in range(0, n_features, N_ELL_BINS_ACTUAL)
    ]


# =============================================================================
# SAMPLING UTILITIES
# =============================================================================

def _sample_member_thread(
    member, x_t: torch.Tensor, n_samples: int,
    result: list, exception: list,
) -> None:
    try:
        try:
            x_t = x_t.to(next(member.parameters()).device)
        except Exception:
            pass
        s = member.sample((n_samples,), x=x_t, show_progress_bars=False)
        result[0] = s.detach().cpu().numpy()
    except Exception as e:
        exception[0] = e


def _run_in_thread(fn_args: tuple, timeout: float) -> tuple[Optional[np.ndarray], Optional[Exception]]:
    """Run _sample_member_thread in a daemon thread with a timeout."""
    result, exception = [None], [None]
    t = threading.Thread(
        target=_sample_member_thread, args=(*fn_args, result, exception), daemon=True
    )
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None, TimeoutError(f'Thread timed out after {timeout}s.')
    return result[0], exception[0]


def sample_ensemble_direct(
    posterior,
    x_obs_norm: np.ndarray,
    n_samples: int = 500,
    timeout_per_member: float = 120.0,
) -> Optional[np.ndarray]:
    """
    Draw samples from a (possibly ensemble) posterior.
    Returns an array of shape (n_samples, n_params) or None on failure.
    """
    x_t = torch.from_numpy(np.asarray(x_obs_norm)).float().reshape(1, -1)
    members = getattr(posterior, 'posteriors', None)

    # ── single posterior ──────────────────────────────────────────────────────
    if members is None:
        arr, exc = _run_in_thread((posterior, x_t, n_samples), timeout_per_member)
        if exc is not None:
            print(f'    [WARN] Posterior sampling failed: {exc}')
            return None
        if arr is None:
            print(f'    [WARN] Posterior timed out after {timeout_per_member}s.')
        return arr

    # ── ensemble posterior ────────────────────────────────────────────────────
    per_member = max(1, n_samples // len(members))
    collected  = []

    for i, member in enumerate(members):
        arr, exc = _run_in_thread((member, x_t, per_member), timeout_per_member)
        if exc is not None:
            print(f'    [SKIP] Member {i} failed: {exc}')
        elif arr is None:
            print(f'    [SKIP] Member {i} timed out.')
        else:
            collected.append(arr)

    if not collected:
        print(f'    [WARN] All {len(members)} members failed or timed out.')
        return None

    n_ok = len(collected)
    if n_ok < len(members):
        print(f'    [INFO] {n_ok}/{len(members)} members contributed samples.')

    combined = np.concatenate(collected, axis=0)
    np.random.shuffle(combined)
    if len(combined) >= n_samples:
        return combined[:n_samples]
    return combined[np.random.choice(len(combined), size=n_samples, replace=True)]


# =============================================================================
# HMC
# =============================================================================

def run_hmc_probe(
    probe_name:    str,
    x_obs_full:    np.ndarray,
    fiducial_path: pathlib.Path,
    param_specs,
    output_dir:    pathlib.Path,
    probes_arg:    tuple,
) -> np.ndarray:
    """
    Run NUTS for one probe.

    Covariance is sourced from the fiducial .npz; the observation is the
    matching slice of x_obs.npy.  Returns flat samples of shape
    (num_chains * num_samples, n_params).
    """
    out_path = output_dir / f'hmc_samples_{probe_name}.npz'
    if out_path.exists():
        print(f'  [HMC/{probe_name}] Loading cached samples from {out_path}')
        d = np.load(out_path)
        return np.column_stack([d[f'samples_{n}'] for n in PARAM_NAMES])

    print(f'\n[HMC/{probe_name}] Setting up...')

    selected   = selected_product_arrays(fiducial_path, probes=probes_arg, ell_min=None, ell_max=None)
    x_obs_sel  = _slice_xobs(x_obs_full, probes_arg)

    vector_fn, _ = make_inference_theory_vector_function(
        param_specs,
        selected['selection'],
        fiducial_vector=x_obs_sel,
        backend='linearized',
        fiducial_offset=True,
        jit_compile=True,
    )

    validate_theory_vector(
        vector_fn,
        {**selected, 'data_vector': x_obs_sel, 'chol': selected['chol']},
        param_specs,
    )

    obs      = jnp.asarray(x_obs_sel,       dtype=jnp.float64)
    chol     = jnp.asarray(selected['chol'], dtype=jnp.float64)
    low_j    = jnp.asarray(PRIOR_LOW,        dtype=jnp.float64)
    high_j   = jnp.asarray(PRIOR_HIGH,       dtype=jnp.float64)
    init_val = {spec.name: float(spec.fiducial) for spec in param_specs}

    def model():
        values = [
            numpyro.sample(spec.name, dist.Uniform(low_j[ip], high_j[ip]))
            for ip, spec in enumerate(param_specs)
        ]
        mu    = vector_fn(jnp.stack(values))
        resid = obs - mu
        white = jsl.solve_triangular(chol, resid, lower=True)
        numpyro.factor('loglike', -0.5 * jnp.dot(white, white))

    numpyro.set_host_device_count(max(HMC_NUM_CHAINS, 1))
    kernel = NUTS(
        model,
        dense_mass=HMC_DENSE_MASS,
        init_strategy=init_to_value(values=init_val),
        max_tree_depth=HMC_MAX_TREE_DEPTH,
        target_accept_prob=HMC_TARGET_ACCEPT,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=HMC_NUM_WARMUP,
        num_samples=HMC_NUM_SAMPLES,
        num_chains=HMC_NUM_CHAINS,
        chain_method=HMC_CHAIN_METHOD,
        progress_bar=True,
    )

    t0 = time.time()
    mcmc.run(
        jax.random.PRNGKey(HMC_SEED),
        extra_fields=('potential_energy', 'diverging', 'accept_prob', 'num_steps'),
    )
    runtime = time.time() - t0
    print(f'  [HMC/{probe_name}] Done in {runtime:.1f}s')

    samples_flat  = mcmc.get_samples(group_by_chain=False)
    samples_chain = mcmc.get_samples(group_by_chain=True)
    extra         = mcmc.get_extra_fields(group_by_chain=True)

    payload = {
        'x_obs_sel': np.asarray(x_obs_sel),
        'cov':       np.asarray(selected['cov']),
        'chol':      np.asarray(selected['chol']),
    }
    for spec in param_specs:
        payload[f'samples_{spec.name}']       = np.asarray(samples_flat[spec.name])
        payload[f'samples_chain_{spec.name}'] = np.asarray(samples_chain[spec.name])
    for k, v in extra.items():
        payload[f'extra_{k}'] = np.asarray(v)
    np.savez_compressed(out_path, **payload)

    try:
        import arviz as az
        idata   = az.from_numpyro(mcmc)
        summary = az.summary(idata, var_names=PARAM_NAMES)
        diag = {
            'probe':         probe_name,
            'runtime_sec':   runtime,
            'max_rhat':      float(summary['r_hat'].max()),
            'min_ess_bulk':  float(summary['ess_bulk'].min()),
            'arviz_summary': json.loads(summary.to_json()),
        }
        with (output_dir / f'hmc_diagnostics_{probe_name}.json').open('w') as f:
            json.dump(diag, f, indent=2, sort_keys=True)
        print(f'  [HMC/{probe_name}] r_hat_max={diag["max_rhat"]:.3f}  '
              f'ess_bulk_min={diag["min_ess_bulk"]:.0f}')
    except Exception as exc:
        print(f'  [HMC/{probe_name}] ArviZ diagnostics failed: {exc}')

    return np.column_stack([np.asarray(samples_flat[n]) for n in PARAM_NAMES])


# =============================================================================
# SBI — normalisation helpers
# =============================================================================

def _compute_normalisation(
    xt_full: np.ndarray,
    xo: np.ndarray,
    is_individual: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (xt_norm, xo_norm, x_mean, x_std).
    Individual probes: single global z-score.
    Multi-probe:       per-ell-block z-score.
    """
    n_stats = xt_full.shape[1]

    if is_individual:
        x_mean = np.mean(xt_full, axis=0)
        x_std  = np.std(xt_full,  axis=0)
        x_std[x_std < 1e-10] = 1.0
        xt_norm = (xt_full - x_mean) / x_std
        xo_norm = (xo      - x_mean) / x_std
        return xt_norm, xo_norm, x_mean, x_std

    blocks  = _make_blocks(n_stats)
    xt_norm = np.empty_like(xt_full)
    xo_norm = np.empty(n_stats, dtype=np.float32)
    x_mean  = np.empty(n_stats, dtype=np.float32)
    x_std   = np.empty(n_stats, dtype=np.float32)

    for blk in blocks:
        blk = np.asarray(blk)
        m = np.mean(xt_full[:, blk], axis=0)
        s = np.std(xt_full[:,  blk], axis=0)
        s[s < 1e-10] = 1.0
        xt_norm[:, blk] = (xt_full[:, blk] - m) / s
        xo_norm[blk]    = (xo[blk]          - m) / s
        x_mean[blk]     = m
        x_std[blk]      = s

    return xt_norm, xo_norm, x_mean, x_std


def _load_scalers_and_normalise(
    work_dir:    str,
    prefix:      str,
    probe_name:  str,
    x_obs_full:  np.ndarray,
    x_obs_idx:   list,
) -> Optional[np.ndarray]:
    """
    Load <prefix><probe_name>_{mean,std}.npy and return normalised x_obs,
    or None if either file is missing.
    """
    mean_path = _fpath(work_dir, f'{prefix}{probe_name}_mean.npy')
    std_path  = _fpath(work_dir, f'{prefix}{probe_name}_std.npy')
    if not (os.path.exists(mean_path) and os.path.exists(std_path)):
        return None
    x_mean  = np.load(mean_path)
    x_std   = np.load(std_path)
    xo_norm = _normalise(x_obs_full, x_obs_idx, x_mean, x_std)
    np.save(_fpath(work_dir, f'xobs_2pt_{probe_name}.npy'), xo_norm)
    return xo_norm


# =============================================================================
# SBI — training
# =============================================================================

def train_sbi_probe(
    name:        str,
    x_obs_idx:   list,
    x_train:     np.ndarray,
    theta_train: np.ndarray,
    x_obs_full:  np.ndarray,
    work_dir:    str,
    device:      str,
) -> tuple[str, bool, str]:
    """
    Train one NPE+MDN posterior (matching main pipeline architecture).
    Saves scalers and normalised arrays with the '2pt_' prefix so they do not
    clash with the main pipeline's files.

    Returns (name, success, message).
    """
    torch.set_num_threads(1)

    try:
        n_stats = len(x_obs_idx)
        n_train = len(theta_train)

        xt_full = x_train[:, x_obs_idx].astype(np.float32)
        xo      = x_obs_full[x_obs_idx].astype(np.float32)

        xt_norm, xo_norm, x_mean, x_std = _compute_normalisation(
            xt_full, xo, is_individual=(name in INDIVIDUAL_STATS)
        )

        frac_below = float((xo < xt_full.min(axis=0)).mean())
        frac_above = float((xo > xt_full.max(axis=0)).mean())
        print(
            f'{name:12s}  n_stats={n_stats}  n_train={n_train}  '
            f'frac_below={frac_below:.2f}  frac_above={frac_above:.2f}',
            flush=True,
        )

        np.save(_fpath(work_dir, f'scaler_2pt_{name}_mean.npy'),  x_mean)
        np.save(_fpath(work_dir, f'scaler_2pt_{name}_std.npy'),   x_std)
        np.save(_fpath(work_dir, f'x_2pt_{name}.npy'),            xt_norm)
        np.save(_fpath(work_dir, f'xobs_2pt_{name}.npy'),         xo_norm)
        np.save(_fpath(work_dir, f'theta_train_2pt_{name}.npy'),  theta_train)

        loader = StaticNumpyLoader(
            in_dir=work_dir,
            x_file=f'x_2pt_{name}.npy',
            theta_file=f'theta_train_2pt_{name}.npy',
            xobs_file=f'xobs_2pt_{name}.npy',
        )

        nets = load_nde_sbi(
            engine='NPE', model='mdn',
            repeats=EQUAL_ARCH['repeats'],
            hidden_features=EQUAL_ARCH['hidden_features'],
            num_components=EQUAL_ARCH['num_components'],
        )

        runner = InferenceRunner.load(
            backend='sbi', engine='NPE',
            prior=BoxUniform(
                low =torch.tensor(PRIOR_LOW,  dtype=torch.float32, device=device),
                high=torch.tensor(PRIOR_HIGH, dtype=torch.float32, device=device),
            ),
            nets=nets,
            out_dir=pathlib.Path(_fpath(work_dir, f'sbi_logs_2pt_{name}')),
            device=device,
            train_args={
                'training_batch_size': EQUAL_ARCH['batch_size'],
                'learning_rate':       EQUAL_ARCH['learning_rate'],
                'max_num_epochs':      EQUAL_ARCH['max_num_epochs'],
                'stop_after_epochs':   50,
                'clip_max_norm':       5.0,
                'validation_fraction': VAL_FRACTION,
            },
        )
        posterior, _ = runner(loader)

        with open(_fpath(work_dir, f'ili_posterior_2pt_{name}.pkl'), 'wb') as f:
            pk.dump(posterior, f)

        msg = (
            f'[{name}] DONE  n_stats={n_stats}  n_train={n_train}  '
            f'hfs={EQUAL_ARCH["hidden_features"]}  '
            f'num_components={EQUAL_ARCH["num_components"]}  '
            f'repeats={EQUAL_ARCH["repeats"]}'
        )
        return name, True, msg

    except Exception:
        return name, False, f'[{name}] FAILED:\n{traceback.format_exc()}'


# =============================================================================
# SBI — load or train, then sample
# =============================================================================

def run_sbi_probe(
    probe_name:   str,
    x_obs_full:   np.ndarray,
    x_train:      np.ndarray,
    theta_train:  np.ndarray,
    work_dir:     str,
    device:       str,
    force_retrain: bool = False,
) -> Optional[np.ndarray]:
    """
    Load or train one SBI posterior, then draw samples.

    Priority:
      1. Main pipeline posterior  (ili_posterior_{name}.pkl  + scaler_{name}_*.npy)
      2. This script's own posterior  (ili_posterior_2pt_{name}.pkl)
      3. Train a new posterior from scratch

    Returns samples of shape (SBI_N_SAMPLES, n_params) or None on failure.
    """
    x_obs_idx = STAT_MAP_2PT[probe_name]
    tag       = f'SBI/{probe_name}'

    def _sample_from_pkl(pkl_path: str, xo_norm: np.ndarray) -> Optional[np.ndarray]:
        with open(pkl_path, 'rb') as f:
            posterior = pk.load(f)
        print(f'  [{tag}] Sampling {SBI_N_SAMPLES} samples...')
        samples = sample_ensemble_direct(posterior, xo_norm, n_samples=SBI_N_SAMPLES)
        if samples is None:
            return None
        samples = _clip_to_prior(samples)
        _print_sample_stats(tag, samples)
        return samples

    # ── 1. Main pipeline posterior ────────────────────────────────────────────
    main_pkl = _fpath(work_dir, f'ili_posterior_{probe_name}.pkl')
    if os.path.exists(main_pkl) and not force_retrain:
        print(f'\n[{tag}] Using main pipeline posterior: {main_pkl}')
        xo_norm = _load_scalers_and_normalise(work_dir, 'scaler_', probe_name, x_obs_full, x_obs_idx)
        if xo_norm is None:
            print(f'  [{tag}] Main pipeline scalers missing — falling through.')
        else:
            samples = _sample_from_pkl(main_pkl, xo_norm)
            if samples is not None:
                return samples
            print(f'  [{tag}] Sampling from main posterior failed — falling through.')

    # ── 2. This script's own cached posterior ─────────────────────────────────
    own_pkl = _fpath(work_dir, f'ili_posterior_2pt_{probe_name}.pkl')
    if os.path.exists(own_pkl) and not force_retrain:
        print(f'\n[{tag}] Loading cached 2pt posterior: {own_pkl}')
        xo_norm = _load_scalers_and_normalise(work_dir, 'scaler_2pt_', probe_name, x_obs_full, x_obs_idx)
        if xo_norm is None:
            print(f'  [{tag}] 2pt scalers missing, recomputing from training data...')
            xt = x_train[:, x_obs_idx].astype(np.float32)
            xo = x_obs_full[x_obs_idx].astype(np.float32)
            _, xo_norm, x_mean, x_std = _compute_normalisation(
                xt, xo, is_individual=(probe_name in INDIVIDUAL_STATS)
            )
            np.save(_fpath(work_dir, f'scaler_2pt_{probe_name}_mean.npy'), x_mean)
            np.save(_fpath(work_dir, f'scaler_2pt_{probe_name}_std.npy'),  x_std)
            np.save(_fpath(work_dir, f'xobs_2pt_{probe_name}.npy'),        xo_norm)

        samples = _sample_from_pkl(own_pkl, xo_norm)
        if samples is not None:
            return samples
        print(f'  [{tag}] Sampling from cached 2pt posterior failed — retraining.')
        # ── 3. Train from scratch ─────────────────────────────────────────────────
    print(f'\n[{tag}] Training new posterior...')
    name, success, msg = train_sbi_probe(
        probe_name, x_obs_idx, x_train, theta_train, x_obs_full, work_dir, device,
    )
    print(f'  {msg}')
    if not success:
        return None

    xo_norm = _load_scalers_and_normalise(
        work_dir, 'scaler_2pt_', probe_name, x_obs_full, x_obs_idx,
    )
    if xo_norm is None:
        print(f'  [{tag}] Scalers still missing after training — aborting.')
        return None

    samples = _sample_from_pkl(own_pkl, xo_norm)
    if samples is None:
        print(f'  [{tag}] Sampling failed after training.')
    return samples


# =============================================================================
# PLOTTING
# =============================================================================

def _make_fallback_plot(
    probe_name:  str,
    hmc_samples: np.ndarray,
    sbi_samples: np.ndarray,
    output_dir:  str,
) -> None:
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for pi, (ax, pname, plabel) in enumerate(zip(axes, PARAM_NAMES, PARAM_LABELS)):
        lo, hi = PRIOR_LOW[pi], PRIOR_HIGH[pi]
        xs = np.linspace(lo, hi, 400)
        for samples, color, label in [
            (hmc_samples, '#1f77b4', 'HMC / NUTS'),
            (sbi_samples, '#d62728', 'SBI / NPE+MDN'),
        ]:
            kde = gaussian_kde(samples[:, pi])
            ys  = kde(xs)
            ax.plot(xs, ys / ys.max(), color=color, lw=1.8, label=label)
        ax.axvline(FIDUCIAL[pi], color='black', lw=1.2, ls='--', label='fiducial')
        ax.set_xlabel(plabel, fontsize=12)
        ax.set_ylabel('Normalised P', fontsize=11)
        ax.set_xlim(lo, hi)
        ax.legend(fontsize=9)
        ax.set_title(f'{probe_name} — {pname}')

    fig.suptitle(f'HMC vs SBI: {probe_name}', fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'hmc_vs_sbi_{probe_name}.pdf')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  [plot] Saved {out_path}')


def make_triangle_plot(
    probe_name:  str,
    hmc_samples: np.ndarray,
    sbi_samples: np.ndarray,
    output_dir:  str,
) -> None:
    try:
        from getdist import MCSamples, plots
    except ImportError:
        print('  [plot] getdist not available — falling back to matplotlib.')
        _make_fallback_plot(probe_name, hmc_samples, sbi_samples, output_dir)
        return

    names  = PARAM_NAMES
    labels = [r'\theta_{\rm ej,0}', r'\nu_{\theta_{\rm ej}}^{M}']
    gd_settings = {'smooth_scale_1D': 0.35, 'smooth_scale_2D': 0.35}

    hmc_gd = MCSamples(
        samples=hmc_samples, names=names, labels=labels,
        label='HMC / NUTS', settings=gd_settings,
    )
    sbi_gd = MCSamples(
        samples=sbi_samples, names=names, labels=labels,
        label='SBI / NPE+MDN', settings=gd_settings,
    )

    g = plots.get_subplot_plotter(width_inch=6.2)
    g.settings.legend_fontsize = 9
    g.settings.axes_labelsize  = 10
    g.triangle_plot(
        [hmc_gd, sbi_gd],
        params=names,
        filled=True,
        legend_labels=['HMC / NUTS', 'SBI / NPE+MDN'],
        contour_colors=['#1f77b4', '#d62728'],
        markers={n: v for n, v in zip(names, FIDUCIAL)},
        marker_args={'color': 'black', 'lw': 1.2, 'ls': '--'},
    )
    out_path = os.path.join(output_dir, f'hmc_vs_sbi_{probe_name}.pdf')
    g.export(out_path)
    print(f'  [plot] Saved {out_path}')


# =============================================================================
# MAIN
# =============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='HMC vs SBI comparison on 2pt Cl statistics.')
    parser.add_argument('--force-retrain',   action='store_true',
                        help='Retrain SBI posteriors even if .pkl files exist.')
    parser.add_argument('--force-fiducial',  action='store_true',
                        help='Regenerate the fiducial .npz theory product.')
    parser.add_argument('--skip-hmc',        action='store_true')
    parser.add_argument('--skip-sbi',        action='store_true')
    parser.add_argument('--probes',          default='gy,gtau,gkappa,all_2pt',
                        help='Comma-separated probe names to compare.')
    parser.add_argument('--sbi-n-samples',   type=int, default=SBI_N_SAMPLES)
    parser.add_argument('--hmc-num-warmup',  type=int, default=HMC_NUM_WARMUP)
    parser.add_argument('--hmc-num-samples', type=int, default=HMC_NUM_SAMPLES)
    parser.add_argument('--hmc-num-chains',  type=int, default=HMC_NUM_CHAINS)
    parser.add_argument('--output-dir',      default=WORK_DIR)
    return parser.parse_args()


def _load_hmc_cache(probe_name: str, output_dir: str) -> Optional[np.ndarray]:
    cached = pathlib.Path(output_dir) / f'hmc_{probe_name}' / f'hmc_samples_{probe_name}.npz'
    if cached.exists():
        d = np.load(cached)
        samples = np.column_stack([d[f'samples_{n}'] for n in PARAM_NAMES])
        print(f'  [HMC/{probe_name}] Loaded {len(samples)} cached samples.')
        return samples
    print(f'  [HMC/{probe_name}] --skip-hmc set and no cache found.')
    return None


def _load_sbi_skip(
    probe_name: str,
    x_obs_full: np.ndarray,
) -> Optional[np.ndarray]:
    """
    When --skip-sbi is set, attempt to load samples from any available cached
    posterior (main pipeline first, then 2pt-specific).
    """
    for pkl_name, scaler_prefix in [
        (f'ili_posterior_{probe_name}.pkl',     'scaler_'),
        (f'ili_posterior_2pt_{probe_name}.pkl', 'scaler_2pt_'),
    ]:
        pkl_path = _fpath(WORK_DIR, pkl_name)
        if not os.path.exists(pkl_path):
            continue

        xo_norm = _load_scalers_and_normalise(
            WORK_DIR, scaler_prefix, probe_name,
            x_obs_full, STAT_MAP_2PT[probe_name],
        )
        if xo_norm is None:
            continue

        with open(pkl_path, 'rb') as f:
            posterior = pk.load(f)
        samples = sample_ensemble_direct(posterior, xo_norm, n_samples=SBI_N_SAMPLES)
        if samples is not None:
            samples = _clip_to_prior(samples)
            print(f'  [SBI/{probe_name}] Loaded from {pkl_name}, '
                  f'{len(samples)} samples.')
            return samples

    print(f'  [SBI/{probe_name}] --skip-sbi set and no usable cached posterior found.')
    return None


def _save_samples(
    probe_name:       str,
    hmc_samples_dict: dict,
    sbi_samples_dict: dict,
    output_dir:       str,
) -> None:
    if probe_name in hmc_samples_dict:
        s = hmc_samples_dict[probe_name]
        path = os.path.join(output_dir, f'hmc_samples_{probe_name}.npz')
        np.savez_compressed(path, theta_ej_0=s[:, 0], nu_theta_ej_M=s[:, 1])
        print(f'  [save/{probe_name}] Written: {path}')

    if probe_name in sbi_samples_dict:
        path = os.path.join(output_dir, f'sbi_samples_{probe_name}.npy')
        np.save(path, sbi_samples_dict[probe_name])
        print(f'  [save/{probe_name}] Written: {path}')


def _print_summary(probe_list: list, hmc_dict: dict, sbi_dict: dict) -> None:
    print(f'\n{"="*60}')
    print('  SUMMARY')
    print(f'{"="*60}')
    for probe_name in probe_list:
        for label, d in [('HMC', hmc_dict), ('SBI', sbi_dict)]:
            if probe_name in d:
                s = d[probe_name]
                status = (
                    f'OK  ({len(s)} samples  '
                    f'theta_ej_0={s[:,0].mean():.3f}+/-{s[:,0].std():.3f}  '
                    f'nu={s[:,1].mean():.3f}+/-{s[:,1].std():.3f})'
                )
            else:
                status = 'MISSING'
            print(f'  {probe_name:10s}  {label}: {status}')


if __name__ == '__main__':
    args = _parse_args()

    # Override globals from CLI flags
    HMC_NUM_WARMUP  = args.hmc_num_warmup
    HMC_NUM_SAMPLES = args.hmc_num_samples
    HMC_NUM_CHAINS  = args.hmc_num_chains
    SBI_N_SAMPLES   = args.sbi_n_samples

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    probe_list = [p.strip() for p in args.probes.split(',') if p.strip()]
    for p in probe_list:
        if p not in STAT_MAP_2PT:
            raise ValueError(f'Unknown probe "{p}". Choose from {list(STAT_MAP_2PT)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device      : {device}')
    print(f'Output dir  : {output_dir}')
    print(f'Probes      : {probe_list}')
    print(f'N_ELL_BINS  : {N_ELL_BINS_ACTUAL}')
    print(f'N_SUMMARY   : {N_ELL_BINS_ACTUAL * len(CL_SPECS_FULL)}')

    # ── 1. Load x_obs.npy ────────────────────────────────────────────────────
    x_obs_path = _fpath(WORK_DIR, 'x_obs.npy')
    if not os.path.exists(x_obs_path):
        raise FileNotFoundError(
            f'x_obs.npy not found at {x_obs_path}. '
            'Run the main SBI pipeline first.')
    x_obs_full = np.load(x_obs_path)
    print(f'\nx_obs shape : {x_obs_full.shape}  '
          f'(expect {N_ELL_BINS_ACTUAL * len(CL_SPECS_FULL)})')
    for label in CL_SPECS_FULL:
        sl = _SLICE[label]
        print(f'  {label:8s}: mean={x_obs_full[sl].mean():.4e}  '
              f'range=[{x_obs_full[sl].min():.4e}, {x_obs_full[sl].max():.4e}]')

    # ── 2. Load training data ─────────────────────────────────────────────────
    x_train_path     = _fpath(WORK_DIR, f'x_train_full{_CACHE_SUFFIX}.npy')
    theta_train_path = _fpath(WORK_DIR, 'theta_train_full.npy')
    if not os.path.exists(x_train_path) or not os.path.exists(theta_train_path):
        raise FileNotFoundError(
            f'Training data not found at {WORK_DIR}. '
            'Run the main SBI pipeline first.')
    x_train     = np.load(x_train_path)
    theta_train = np.load(theta_train_path)
    print(f'\nTraining data: x_train={x_train.shape}  theta_train={theta_train.shape}')

    # ── 3. Fiducial theory product (for HMC covariance) ──────────────────────
    param_specs   = default_parameter_specs()
    fiducial_path = ensure_default_fiducial_product(
        DEFAULT_FIDUCIAL_PATH,
        param_specs=param_specs,
        force=args.force_fiducial,
    )
    print(f'\nFiducial product: {fiducial_path}')

    # ── 4. Per-probe HMC + SBI ────────────────────────────────────────────────
    hmc_samples_dict: dict[str, np.ndarray] = {}
    sbi_samples_dict: dict[str, np.ndarray] = {}

    for probe_name in probe_list:
        print(f'\n{"="*60}')
        print(f'  PROBE: {probe_name}')
        print(f'{"="*60}')

        # ── HMC ──────────────────────────────────────────────────────────────
        if not args.skip_hmc:
            hmc_out_dir = pathlib.Path(output_dir) / f'hmc_{probe_name}'
            hmc_out_dir.mkdir(parents=True, exist_ok=True)
            hmc_samples = run_hmc_probe(
                probe_name    = probe_name,
                x_obs_full    = x_obs_full,
                fiducial_path = fiducial_path,
                param_specs   = param_specs,
                output_dir    = hmc_out_dir,
                probes_arg    = PROBE_TO_THEORY[probe_name],
            )
            hmc_samples_dict[probe_name] = hmc_samples
            print(f'  [HMC/{probe_name}] {len(hmc_samples)} samples  '
                  f'theta_ej_0={hmc_samples[:,0].mean():.3f}'
                  f'+/-{hmc_samples[:,0].std():.3f}  '
                  f'nu={hmc_samples[:,1].mean():.3f}'
                  f'+/-{hmc_samples[:,1].std():.3f}')
        else:
            hmc_samples = _load_hmc_cache(probe_name, output_dir)
            if hmc_samples is not None:
                hmc_samples_dict[probe_name] = hmc_samples

        # ── SBI ──────────────────────────────────────────────────────────────
        if not args.skip_sbi:
            sbi_samples = run_sbi_probe(
                probe_name    = probe_name,
                x_obs_full    = x_obs_full,
                x_train       = x_train,
                theta_train   = theta_train,
                work_dir      = WORK_DIR,
                device        = device,
                force_retrain = args.force_retrain,
            )
            if sbi_samples is not None:
                sbi_samples_dict[probe_name] = sbi_samples
        else:
            sbi_samples = _load_sbi_skip(probe_name, x_obs_full)
            if sbi_samples is not None:
                sbi_samples_dict[probe_name] = sbi_samples

        # ── Save flat samples ─────────────────────────────────────────────────
        _save_samples(probe_name, hmc_samples_dict, sbi_samples_dict, output_dir)

        # ── Plot ──────────────────────────────────────────────────────────────
        if probe_name in hmc_samples_dict and probe_name in sbi_samples_dict:
            make_triangle_plot(
                probe_name  = probe_name,
                hmc_samples = hmc_samples_dict[probe_name],
                sbi_samples = sbi_samples_dict[probe_name],
                output_dir  = output_dir,
            )
        else:
            missing = (
                (['HMC'] if probe_name not in hmc_samples_dict else []) +
                (['SBI'] if probe_name not in sbi_samples_dict else [])
            )
            print(f'  [plot/{probe_name}] Skipping — missing: {missing}')

    # ── 5. Summary ────────────────────────────────────────────────────────────
    _print_summary(probe_list, hmc_samples_dict, sbi_samples_dict)
    print('\nAll done.')
