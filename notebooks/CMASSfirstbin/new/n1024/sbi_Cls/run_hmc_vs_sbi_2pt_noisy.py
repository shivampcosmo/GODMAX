#!/usr/bin/env python
"""
run_hmc_vs_sbi_2pt_noisy.py

HMC (NumPyro NUTS) vs SBI (LtU-ILI NPE+MDN) comparison on 2pt Cl statistics.

HMC:
  - Forward model : analytical theory Cls via theory_sbi_utils / run_hmc_theory_cls
  - Covariance    : theory Gaussian covariance from the fiducial .npz product
  - Observation   : x_obs.npy sliced to the relevant 2pt probe indices

SBI:
  - Trains NPE+MDN on simulation Cls (x_train_full.npy / theta_train_full.npy)
  - One posterior per probe: gy, gtau, gkappa, all_2pt
  - Normalisation: per-feature z-score (matching main SBI pipeline)
  - Observation   : x_obs.npy sliced + normalised with saved scalers

Output: one GetDist triangle plot per probe saved to WORK_DIR/
  hmc_vs_sbi_gy.pdf, hmc_vs_sbi_gtau.pdf, hmc_vs_sbi_gkappa.pdf, hmc_vs_sbi_all_2pt.pdf

Run from:
  /work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/sbi_Cls/
"""

from __future__ import annotations

import os
import sys
import json
import time
import pickle as pk
import pathlib
import threading
import multiprocessing as mp

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# PATHS
# =============================================================================

WORK_DIR     = str(pathlib.Path(__file__).parent.resolve())
SBI_VALIDATE = '/work/hdd/bdne/aacharya2/GODMAX/notebooks/SBI_validate'
LTU_ILI_PATH = '/work/hdd/bdne/aacharya2/ltu-ili'

for p in [SBI_VALIDATE, LTU_ILI_PATH, WORK_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── imports from SBI_validate ─────────────────────────────────────────────────
from theory_sbi_utils import (
    DEFAULT_FIDUCIAL_PATH,
    THEORY_SBI_DIR,
    default_parameter_specs,
    ensure_default_fiducial_product,
    make_inference_theory_vector_function,
    parse_probe_list,
    selected_product_arrays,
    validate_theory_vector,
    fiducial_theta,
    prior_bounds,
    stable_cholesky,
)
from run_hmc_theory_cls import run_hmc

# ── LtU-ILI ──────────────────────────────────────────────────────────────────
from ili.dataloaders import StaticNumpyLoader
from ili.inference   import InferenceRunner
from ili.utils       import load_nde_sbi

# ── JAX / NumPyro ─────────────────────────────────────────────────────────────
from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from sbi.utils import BoxUniform

# =============================================================================
# CONSTANTS  (matching main SBI pipeline exactly)
# =============================================================================
PRIOR_LOW    = [1.0, -0.3]
PRIOR_HIGH   = [6.0,  0.0]
PARAM_NAMES  = ['theta_ej_0', 'nu_theta_ej_M']
PARAM_LABELS = [r'$\theta_{\rm ej,0}$', r'$\nu_{\theta_{\rm ej}}^{M}$']
FIDUCIAL     = [2.0, -0.1]

LMIN       = 100
LMAX       = 1500
N_ELL_BINS = 20

# x_obs.npy ordering: [g2y | g2tau | g2kappa | gy | gtau | gkappa]
# each block is N_ELL_BINS_ACTUAL wide.
CL_SPECS_FULL = [
    'g2y', 'g2tau', 'g2kappa', 'gy', 'gtau', 'gkappa',
]

VAL_FRACTION     = 0.10
INDIVIDUAL_STATS = {'gy', 'gtau', 'gkappa'}

EQUAL_ARCH = {
    'hidden_features': 64,
    'num_transforms':  6,
    'learning_rate':   1e-4,
    'batch_size':      64,
    'max_num_epochs':  500,
    'repeats':         6,
}

# HMC settings
HMC_NUM_WARMUP     = 3000
HMC_NUM_SAMPLES    = 2000
HMC_NUM_CHAINS     = 4
HMC_MAX_TREE_DEPTH = 10
HMC_DENSE_MASS     = True
HMC_TARGET_ACCEPT  = 0.9
HMC_CHAIN_METHOD   = 'vectorized'
HMC_SEED           = 42

# SBI settings
SBI_N_SAMPLES = 4000

# =============================================================================
# ELL BINNING  (matching main SBI pipeline exactly)
# =============================================================================

def make_ell_bins(lmin=LMIN, lmax=LMAX, n_bins=N_ELL_BINS):
    edges = np.unique(
        np.logspace(np.log10(lmin), np.log10(lmax), n_bins + 1).astype(int)
    )
    centres = 0.5 * (edges[:-1] + edges[1:]).astype(float)
    return edges, centres

ELL_EDGES, ELL_CENTRES = make_ell_bins()
N_ELL_BINS_ACTUAL = len(ELL_EDGES) - 1

# ── STAT_MAP: indices into x_obs.npy ─────────────────────────────────────────
_s = {
    label: list(range(i * N_ELL_BINS_ACTUAL, (i + 1) * N_ELL_BINS_ACTUAL))
    for i, label in enumerate(CL_SPECS_FULL)
}

STAT_MAP_2PT = {
    'gy':      _s['gy'],
    'gtau':    _s['gtau'],
    'gkappa':  _s['gkappa'],
    'all_2pt': _s['gy'] + _s['gtau'] + _s['gkappa'],
}

# Ordered list of 2pt probe names that make up all_2pt (for covariance probes arg)
ALL_2PT_PROBES = ['gy', 'gtau', 'gkappa']

# =============================================================================
# SAMPLING UTILITY
# =============================================================================

def _sample_member_thread(member, x_t, n_samples, result, exception):
    try:
        s = member.sample((n_samples,), x=x_t, show_progress_bars=False)
        result[0] = s.detach().cpu().numpy()
    except Exception as e:
        exception[0] = e


def sample_ensemble_direct(posterior, x_obs_norm, n_samples=500,
                            timeout_per_member=120):
    x_t = torch.from_numpy(np.asarray(x_obs_norm)).float().reshape(1, -1)

    try:
        members = posterior.posteriors
    except AttributeError:
        result, exception = [None], [None]
        t = threading.Thread(
            target=_sample_member_thread,
            args=(posterior, x_t, n_samples, result, exception),
            daemon=True,)
        t.start()
        t.join(timeout=timeout_per_member)
        if t.is_alive():
            print(f'    [WARN] Single posterior timed out after {timeout_per_member}s.')
            return None
        if exception[0] is not None:
            print(f'    [WARN] Single posterior failed: {exception[0]}')
            return None
        return result[0]

    n_members  = len(members)
    per_member = max(1, n_samples // n_members)
    collected  = []

    for i, member in enumerate(members):
        result, exception = [None], [None]
        t = threading.Thread(
            target=_sample_member_thread,
            args=(member, x_t, per_member, result, exception),
            daemon=True,
        )
        t.start()
        t.join(timeout=timeout_per_member)
        if t.is_alive():
            print(f'    [SKIP] Member {i} timed out.')
        elif exception[0] is not None:
            print(f'    [SKIP] Member {i} failed: {exception[0]}')
        else:
            collected.append(result[0])

    if not collected:
        print(f'    [WARN] All {n_members} members failed or timed out.')
        return None

    n_ok = len(collected)
    if n_ok < n_members:
        print(f'    [INFO] {n_ok}/{n_members} members contributed samples.')

    combined = np.concatenate(collected, axis=0)
    np.random.shuffle(combined)
    if len(combined) >= n_samples:
        return combined[:n_samples]
    idx = np.random.choice(len(combined), size=n_samples, replace=True)
    return combined[idx]

# =============================================================================
# HMC
# =============================================================================

def run_hmc_probe(
    probe_name: str,
    x_obs_full: np.ndarray,
    fiducial_path: pathlib.Path,
    param_specs,
    output_dir: pathlib.Path,
    probes_arg: tuple,
) -> np.ndarray:

    out_path = output_dir / f'hmc_samples_{probe_name}.npz'
    if out_path.exists():
        print(f'  [HMC/{probe_name}] Loading cached samples from {out_path}')
        d = np.load(out_path)
        return np.column_stack([d[f'samples_{n}'] for n in PARAM_NAMES])

    print(f'\n[HMC/{probe_name}] Setting up...')

    # Load covariance + selection from fiducial product
    selected = selected_product_arrays(
        fiducial_path, probes=probes_arg, ell_min=None, ell_max=None,
    )
    selection = selected['selection']

    # Slice x_obs FIRST so it can be passed as fiducial_vector
    x_obs_sel = _slice_xobs_for_probes(x_obs_full, probes_arg)

    vector_fn, theory_info = make_inference_theory_vector_function(
        param_specs,
        selection,
        fiducial_vector=x_obs_sel,
        backend='linearized',
        fiducial_offset=True,
        jit_compile=True,
    )

    jac = theory_info["jacobian"]  # shape (n_data, n_params)
    for ip, spec in enumerate(param_specs):
        col = jac[:, ip]
        print(f'  [JAC/{probe_name}] {spec.name}: '
              f'max={np.abs(col).max():.4e}  '
              f'mean={np.abs(col).mean():.4e}  '
              f'norm={np.linalg.norm(col):.4e}')

    # Validate using x_obs_sel as the data vector
    selected_for_validation = dict(selected)
    selected_for_validation['data_vector'] = x_obs_sel
    diag = validate_theory_vector(vector_fn, selected_for_validation, param_specs)
    print(f'  [HMC/{probe_name}] gradient_norm={diag["gradient_norm"]:.4e}  '
          f'max_rel_diff={diag["max_rel_diff"]:.4e}  '
          f'finite_gradient={diag["finite_gradient"]}')

    obs  = jnp.asarray(x_obs_sel,       dtype=jnp.float64)
    chol = jnp.asarray(selected['chol'], dtype=jnp.float64)

    low_j  = jnp.asarray([PRIOR_LOW[0],  PRIOR_LOW[1]],  dtype=jnp.float64)
    high_j = jnp.asarray([PRIOR_HIGH[0], PRIOR_HIGH[1]], dtype=jnp.float64)
    init_values = {spec.name: float(spec.fiducial) for spec in param_specs}

    def model():
        values = []
        for ip, spec in enumerate(param_specs):
            values.append(
                numpyro.sample(spec.name, dist.Uniform(low_j[ip], high_j[ip]))
            )
        theta = jnp.stack(values)
        mu    = vector_fn(theta)
        resid = obs - mu
        white = jsl.solve_triangular(chol, resid, lower=True)
        numpyro.factor('loglike', -0.5 * jnp.dot(white, white))

    numpyro.set_host_device_count(max(HMC_NUM_CHAINS, 1))
    kernel = NUTS(
        model,
        dense_mass=HMC_DENSE_MASS,
        init_strategy=init_to_value(values=init_values),
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
        'cov':  np.asarray(selected['cov']),
        'chol': np.asarray(selected['chol']),
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
            'probe':        probe_name,
            'runtime_sec':  runtime,
            'max_rhat':     float(summary['r_hat'].max()),
            'min_ess_bulk': float(summary['ess_bulk'].min()),
            'arviz_summary': json.loads(summary.to_json()),
        }
        with (output_dir / f'hmc_diagnostics_{probe_name}.json').open('w') as f:
            json.dump(diag, f, indent=2, sort_keys=True)
        print(f'  [HMC/{probe_name}] r_hat_max={diag["max_rhat"]:.3f}  '
              f'ess_bulk_min={diag["min_ess_bulk"]:.0f}')
    except Exception as exc:
        print(f'  [HMC/{probe_name}] ArviZ diagnostics failed: {exc}')

    return np.column_stack([np.asarray(samples_flat[n]) for n in PARAM_NAMES])


def _slice_xobs_for_probes(x_obs_full: np.ndarray,
                            probes_arg: tuple) -> np.ndarray:
    """
    Extract the elements of x_obs_full (shape N_SUMMARY,) that correspond
    to the requested probes, in the same order that selected_product_arrays
    returns them.

    x_obs.npy ordering : [g2y | g2tau | g2kappa | gy | gtau | gkappa]
    theory product ordering (probes_arg) : e.g. ('gy',) or ('gy','gtau','gkappa')

    Both use N_ELL_BINS_ACTUAL bins per spectrum, so we can directly index
    via STAT_MAP_2PT / _s.
    """
    idx = []
    for probe in probes_arg:
        idx.extend(_s[probe])
    return x_obs_full[np.array(idx)]


# =============================================================================
# SBI: train one posterior per probe on simulation Cls
# =============================================================================

def make_blocks(n_features):
    return [
        list(range(i, min(i + N_ELL_BINS_ACTUAL, n_features)))
        for i in range(0, n_features, N_ELL_BINS_ACTUAL)
    ]


def train_sbi_probe(args):
    """Worker function: train one NPE+NSF posterior. Runs in a subprocess."""
    torch.set_num_threads(1)
    (name, x_obs_idx, x_train, theta_train, work_dir, device) = args

    sys.path.insert(0, LTU_ILI_PATH)
    from ili.dataloaders import StaticNumpyLoader
    from ili.inference   import InferenceRunner
    from ili.utils       import load_nde_sbi
    from sbi.utils       import BoxUniform

    def fpath(fname):
        return os.path.join(work_dir, fname)

    try:
        n_stats = len(x_obs_idx)
        n_train = len(theta_train)

        xt_full = x_train[:, x_obs_idx].astype(np.float32)
        xo      = np.load(fpath('x_obs.npy'))[x_obs_idx].astype(np.float32)

        is_individual = name in INDIVIDUAL_STATS
        blocks = None if is_individual else make_blocks(n_stats)

        if blocks is None:
            x_mean = np.mean(xt_full, axis=0)
            x_std  = np.std(xt_full,  axis=0)
            x_std[x_std < 1e-10] = 1.0
            xt_norm = (xt_full - x_mean) / x_std
            xo_norm = (xo      - x_mean) / x_std
        else:
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

        frac_below = float((xo < xt_full.min(axis=0)).mean())
        frac_above = float((xo > xt_full.max(axis=0)).mean())
        print(f'{name:12s}  n_stats={n_stats}  n_train={n_train}  '
              f'frac_below={frac_below:.2f}  frac_above={frac_above:.2f}',
              flush=True)

        np.save(fpath(f'scaler_{name}_mean.npy'), x_mean)
        np.save(fpath(f'scaler_{name}_std.npy'),  x_std)
        np.save(fpath(f'x_{name}.npy'),           xt_norm)
        np.save(fpath(f'xobs_{name}.npy'),        xo_norm)
        np.save(fpath(f'theta_train_{name}.npy'), theta_train)

        loader = StaticNumpyLoader(
            in_dir=work_dir,
            x_file=f'x_{name}.npy',
            theta_file=f'theta_train_{name}.npy',
            xobs_file=f'xobs_{name}.npy',
        )

        hfs        = EQUAL_ARCH['hidden_features']
        nts        = EQUAL_ARCH['num_transforms']
        batch_size = EQUAL_ARCH['batch_size']
        lr         = EQUAL_ARCH['learning_rate']
        max_epochs = EQUAL_ARCH['max_num_epochs']
        repeats    = EQUAL_ARCH['repeats']

        train_args = {
            'training_batch_size': batch_size,
            'learning_rate':       lr,
            'max_num_epochs':      max_epochs,
            'stop_after_epochs':   100,
            'clip_max_norm':       5.0,
            'validation_fraction': VAL_FRACTION,
        }

        nets = load_nde_sbi(
            engine='NPE', model='nsf',
            repeats=repeats,
            hidden_features=hfs,
            num_transforms=nts,
        )

        runner = InferenceRunner.load(
            backend='sbi', engine='NPE',
            prior=BoxUniform(
                low =torch.tensor(PRIOR_LOW,  dtype=torch.float32, device=device),
                high=torch.tensor(PRIOR_HIGH, dtype=torch.float32, device=device),
            ),
            nets=nets,
            out_dir=pathlib.Path(fpath(f'sbi_logs_{name}')),
            device=device,
            train_args=train_args,
        )
        posterior, _ = runner(loader)

        with open(fpath(f'ili_posterior_{name}.pkl'), 'wb') as f:
            pk.dump(posterior, f)

        return name, True, (f'[{name}] DONE  n_stats={n_stats}  n_train={n_train}  '
                            f'hfs={hfs}  nts={nts}  repeats={repeats}')

    except Exception:
        import traceback
        return name, False, f'[{name}] FAILED:\n{traceback.format_exc()}'


def run_sbi_probe(
    probe_name: str,
    x_obs_full: np.ndarray,
    x_train: np.ndarray,
    theta_train: np.ndarray,
    work_dir: str,
    device: str,
    force_retrain: bool = False,
) -> np.ndarray | None:
    posterior_path = os.path.join(work_dir, f'ili_posterior_{probe_name}.pkl')

    if not os.path.exists(posterior_path) or force_retrain:
        print(f'\n[SBI/{probe_name}] Training posterior...')
        x_obs_idx = STAT_MAP_2PT[probe_name]
        name, success, msg = train_sbi_probe((
            probe_name, x_obs_idx, x_train, theta_train, work_dir, device,
        ))
        print(f'  {msg}')
        if not success:
            return None
    else:
        scaler_mean_path = os.path.join(work_dir, f'scaler_{probe_name}_mean.npy')
        scaler_std_path  = os.path.join(work_dir, f'scaler_{probe_name}_std.npy')
        xobs_norm_path   = os.path.join(work_dir, f'xobs_{probe_name}.npy')

        if (os.path.exists(scaler_mean_path)
                and os.path.exists(scaler_std_path)
                and not os.path.exists(xobs_norm_path)):
            x_mean = np.load(scaler_mean_path)
            x_std  = np.load(scaler_std_path)
            xo     = x_obs_full[STAT_MAP_2PT[probe_name]].astype(np.float32)
            np.save(xobs_norm_path, (xo - x_mean) / x_std)

        print(f'\n[SBI/{probe_name}] Loading cached posterior from {posterior_path}')

    with open(posterior_path, 'rb') as f:
        posterior = pk.load(f)

    xobs_norm_path = os.path.join(work_dir, f'xobs_{probe_name}.npy')
    if not os.path.exists(xobs_norm_path):
        print(f'  [SBI/{probe_name}] xobs_{probe_name}.npy not found — '
              f'run training first.')
        return None
    xo_norm = np.load(xobs_norm_path)

    print(f'  [SBI/{probe_name}] Sampling {SBI_N_SAMPLES} posterior samples...')
    samples = sample_ensemble_direct(posterior, xo_norm, n_samples=SBI_N_SAMPLES)
    if samples is None:
        print(f'  [SBI/{probe_name}] Sampling failed.')
        return None

    samples = np.clip(
        samples,
        a_min=np.array(PRIOR_LOW,  dtype=np.float32),
        a_max=np.array(PRIOR_HIGH, dtype=np.float32),
    )
    print(f'  [SBI/{probe_name}] theta_ej_0  = '
          f'{samples[:,0].mean():.3f} +/- {samples[:,0].std():.3f}')
    print(f'  [SBI/{probe_name}] nu_theta_ej = '
          f'{samples[:,1].mean():.3f} +/- {samples[:,1].std():.3f}')
    return samples


# =============================================================================
# PLOTTING
# =============================================================================

def make_triangle_plot(
    probe_name: str,
    hmc_samples: np.ndarray,
    sbi_samples: np.ndarray,
    output_dir: str,
):
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
    g.settings.legend_fontsize  = 9
    g.settings.axes_labelsize   = 10
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


def _make_fallback_plot(
    probe_name: str,
    hmc_samples: np.ndarray,
    sbi_samples: np.ndarray,
    output_dir: str,
):
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for pi, (ax, pname, plabel) in enumerate(
            zip(axes, PARAM_NAMES, PARAM_LABELS)):
        lo, hi = PRIOR_LOW[pi], PRIOR_HIGH[pi]
        xs = np.linspace(lo, hi, 400)
        for samples, color, label in [
            (hmc_samples, '#1f77b4', 'HMC / NUTS'),
            (sbi_samples, '#d62728', 'SBI / NPE+NSF'),
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


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='HMC vs SBI comparison on 2pt Cl statistics.')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Retrain SBI posteriors even if .pkl files exist.')
    parser.add_argument('--force-fiducial', action='store_true',
                        help='Regenerate the fiducial .npz theory product.')
    parser.add_argument('--skip-hmc', action='store_true')
    parser.add_argument('--skip-sbi', action='store_true')
    parser.add_argument('--probes', default='gy,gtau,gkappa,all_2pt',
                        help='Comma-separated probe names to compare.')
    parser.add_argument('--sbi-n-samples', type=int, default=SBI_N_SAMPLES)
    parser.add_argument('--hmc-num-warmup',   type=int, default=HMC_NUM_WARMUP)
    parser.add_argument('--hmc-num-samples',  type=int, default=HMC_NUM_SAMPLES)
    parser.add_argument('--hmc-num-chains',   type=int, default=HMC_NUM_CHAINS)
    parser.add_argument('--output-dir', default=WORK_DIR)
    args = parser.parse_args()

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
    x_obs_path = os.path.join(WORK_DIR, 'x_obs.npy')
    if not os.path.exists(x_obs_path):
        raise FileNotFoundError(
            f'x_obs.npy not found at {x_obs_path}. '
            'Run the main SBI pipeline first to generate it.')
    x_obs_full = np.load(x_obs_path)
    print(f'\nx_obs shape : {x_obs_full.shape}  '
          f'(expect {N_ELL_BINS_ACTUAL * len(CL_SPECS_FULL)})')
    for label in CL_SPECS_FULL:
        sl = _s[label]
        print(f'  {label:8s}: mean={x_obs_full[sl].mean():.4e}  '
              f'range=[{x_obs_full[sl].min():.4e}, {x_obs_full[sl].max():.4e}]')

    # ── 2. Load training data ─────────────────────────────────────────────────
    x_train_path     = os.path.join(WORK_DIR, 'x_train_full_noisy.npy')
    theta_train_path = os.path.join(WORK_DIR, 'theta_train_full.npy')
    if not os.path.exists(x_train_path) or not os.path.exists(theta_train_path):
        raise FileNotFoundError(
            f'Training data not found at {WORK_DIR}. '
            'Run the main SBI pipeline first.')
    x_train     = np.load(x_train_path)
    theta_train = np.load(theta_train_path)
    print(f'\nTraining data: x_train={x_train.shape}  '
          f'theta_train={theta_train.shape}')

    # ── 3. Fiducial theory product (for HMC covariance) ──────────────────────
    param_specs   = default_parameter_specs()
    fiducial_path = ensure_default_fiducial_product(DEFAULT_FIDUCIAL_PATH,
        param_specs=param_specs, force=args.force_fiducial,)
    print(f'\nFiducial product: {fiducial_path}')

    # ── 4. Per-probe HMC + SBI ────────────────────────────────────────────────
    hmc_samples_dict = {}
    sbi_samples_dict = {}

    PROBE_TO_THEORY = {
        'gy':      ('gy',),
        'gtau':    ('gtau',),
        'gkappa':  ('gkappa',),
        'all_2pt': ('gy', 'gtau', 'gkappa'),
    }

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
            cached = pathlib.Path(output_dir) / f'hmc_{probe_name}' \
                     / f'hmc_samples_{probe_name}.npz'
            if cached.exists():
                d = np.load(cached)
                hmc_samples_dict[probe_name] = np.column_stack(
                    [d[f'samples_{n}'] for n in PARAM_NAMES])
                print(f'  [HMC/{probe_name}] Loaded {len(hmc_samples_dict[probe_name])} '
                      f'cached samples.')
            else:
                print(f'  [HMC/{probe_name}] --skip-hmc set and no cache found.')

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
            posterior_path = os.path.join(WORK_DIR, f'ili_posterior_{probe_name}.pkl')
            xobs_norm_path = os.path.join(WORK_DIR, f'xobs_{probe_name}.npy')
            if os.path.exists(posterior_path) and os.path.exists(xobs_norm_path):
                with open(posterior_path, 'rb') as f:
                    posterior = pk.load(f)
                xo_norm = np.load(xobs_norm_path)
                sbi_samples = sample_ensemble_direct(
                    posterior, xo_norm, n_samples=SBI_N_SAMPLES)
                if sbi_samples is not None:
                    sbi_samples = np.clip(
                        sbi_samples,
                        a_min=np.array(PRIOR_LOW),
                        a_max=np.array(PRIOR_HIGH),
                    )
                    sbi_samples_dict[probe_name] = sbi_samples
                    print(f'  [SBI/{probe_name}] Loaded cached posterior, '
                          f'{len(sbi_samples)} samples.')
            else:
                print(f'  [SBI/{probe_name}] --skip-sbi set and no cache found.')

        # ── Save ─────────────────────────────────────────────────────────────
        if probe_name in hmc_samples_dict:
            save_path = os.path.join(output_dir, f'hmc_samples_{probe_name}.npz')
            np.savez_compressed(save_path,
                theta_ej_0    = hmc_samples_dict[probe_name][:, 0],
                nu_theta_ej_M = hmc_samples_dict[probe_name][:, 1],
            )
            print(f'  [save/{probe_name}] Written: {save_path}')

        if probe_name in sbi_samples_dict:
            save_path = os.path.join(output_dir, f'sbi_samples_{probe_name}.npy')
            np.save(save_path, sbi_samples_dict[probe_name])
            print(f'  [save/{probe_name}] Written: {save_path}')

        # ── Plot ─────────────────────────────────────────────────────────────
        if probe_name in hmc_samples_dict and probe_name in sbi_samples_dict:
            make_triangle_plot(
                probe_name  = probe_name,
                hmc_samples = hmc_samples_dict[probe_name],
                sbi_samples = sbi_samples_dict[probe_name],
                output_dir  = output_dir,
            )
        else:
            missing = []
            if probe_name not in hmc_samples_dict:
                missing.append('HMC')
            if probe_name not in sbi_samples_dict:
                missing.append('SBI')
            print(f'  [plot/{probe_name}] Skipping — missing: {missing}')

    # ── 5. Summary ───────────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print('  SUMMARY')
    print(f'{"="*60}')
    for probe_name in probe_list:
        hmc_ok = probe_name in hmc_samples_dict
        sbi_ok = probe_name in sbi_samples_dict
        print(f'  {probe_name:10s}  HMC={"OK" if hmc_ok else "MISSING":7s}  '
              f'SBI={"OK" if sbi_ok else "MISSING"}')

    print('\nAll done.')
