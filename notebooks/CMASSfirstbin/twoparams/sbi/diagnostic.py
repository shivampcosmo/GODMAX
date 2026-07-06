#!/usr/bin/env python3
"""
Posterior collapse diagnostic for GODMAX HOD SBI.
Tests all trained posteriors for:
  1. Input sensitivity (does the posterior change with different inputs?)
  2. Prior coverage (is the posterior just the prior?)
  3. Stat map index correctness (are the right statistics being used?)
  4. Per-member ensemble diversity (are members agreeing or all collapsed?)

Run this after training, before drawing the next proposal round.
"""

import os
import sys
import torch
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon

sys.path.append('/work/hdd/bdne/aacharya2/ltu-ili')

# =============================================================================
# CONFIGURATION — match your main script exactly
# =============================================================================
PRIOR_LOW   = np.array([1.0,  0.01])
PRIOR_HIGH  = np.array([6.0,  1.5])
PARAM_NAMES = ['theta_ej_0', 'mu_beta']
LABELS      = [r'$\theta_{ej,0}$', r'$\mu_{\beta}$']
N_DIAG_SAMPLES = 1000   # posterior samples per test point
N_PRIOR_SAMPLES = 1000  # samples from prior for comparison

STAT_MAP =  {
    # --- Single-tracer 3pt cross-moments ---
    'g2y':         [0,  9,  18, 27, 36],
    'g2tau':       [1,  10, 19, 28, 37],
    'g2kappa':     [2,  11, 20, 29, 38],

    # --- Single-tracer 2pt cross-moments ---
    'gy':          [3,  12, 21, 30, 39],
    'gtau':        [4,  13, 22, 31, 40],
    'gkappa':      [5,  14, 23, 32, 41],

    # --- Auto-moments <X²> at each scale (NEW) ---
    # Key diagnostic for mu_beta: profile shape information without galaxy weighting
    'yy':          [6,  15, 24, 33, 42],
    'tt':          [7,  16, 25, 34, 43],
    'kk':          [8,  17, 26, 35, 44],

    # --- Full joint (all 45 statistics) ---
    'JOINT':       list(range(45)),

    # --- Per-tracer totals: 3pt + 2pt + auto ---
    'y_total':     [0,  3,  6,  9,  12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
    'tau_total':   [1,  4,  7,  10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43],
    'kappa_total': [2,  5,  8,  11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44],

    # --- Category totals ---
    'all_3pt':     [0,  1,  2,  9,  10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38],
    'all_2pt':     [3,  4,  5,  12, 13, 14, 21, 22, 23, 30, 31, 32, 39, 40, 41],
    'all_auto':    [6,  7,  8,  15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44],
}

DIAG_DIR = Path('./diagnostics')
DIAG_DIR.mkdir(exist_ok=True)

# =============================================================================
# LOAD SHARED DATA
# =============================================================================
print("Loading data...")
x_obs     = np.load('x_obs.npy').astype(np.float32)
x_train   = np.load('x_train_full.npy').astype(np.float32)   # full pooled array
theta_train = np.load('theta.npy').astype(np.float32)

# Reconstruct train/val split using same seed as main script
np.random.seed(42)
all_indices = np.arange(len(theta_train))
np.random.shuffle(all_indices)
n_val     = max(5, int(len(theta_train) * 0.20))
val_idx   = all_indices[:n_val]
train_idx = all_indices[n_val:]

print(f"Total samples: {len(theta_train)}, Train: {len(train_idx)}, Val: {n_val}")

# =============================================================================
# HELPER: Sample from a single ensemble member directly
# =============================================================================
import threading

def _sample_member_thread(member, x_t, n, result, exception):
    """Thread target — writes output into shared lists."""
    try:
        s = member.sample((n,), x=x_t, show_progress_bars=False)
        result[0] = s.detach().cpu().numpy()
    except Exception as e:
        exception[0] = e


def sample_member(member, x_norm_1d, n=N_DIAG_SAMPLES, timeout_seconds=45):
    """
    Draw n samples from a single posterior member with a hard timeout.

    Replaces the warnings.filterwarnings approach which could not intercept
    sbi's rejection sampling message because sbi issues it via
    logging.warning() not warnings.warn().

    Returns np.ndarray of shape (n, n_params) or None on timeout/failure.
    """
    x_t       = torch.from_numpy(x_norm_1d).float().reshape(1, -1)
    result    = [None]
    exception = [None]

    t = threading.Thread(
        target=_sample_member_thread,
        args=(member, x_t, n, result, exception),
        daemon=True
    )
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        return None

    if exception[0] is not None:
        return None

    return result[0]

def sample_posterior(posterior, x_norm_1d, n=N_DIAG_SAMPLES):
    """
    Sample from all ensemble members independently and pool.
    Returns np.ndarray of shape (n, n_params) or None.
    """
    try:
        members = posterior.posteriors
    except AttributeError:
        return sample_member(posterior, x_norm_1d, n)

    per_member = max(1, n // len(members))
    collected  = []

    for m in members:
        s = sample_member(m, x_norm_1d, per_member)
        if s is not None:
            collected.append(s)

    if not collected:
        return None

    combined = np.concatenate(collected, axis=0)
    np.random.shuffle(combined)

    # Guarantee exactly n samples regardless of how many members succeeded
    if len(combined) >= n:
        return combined[:n]
    else:
        idx = np.random.choice(len(combined), size=n, replace=True)
        return combined[idx]
# =============================================================================
# HELPER: Jensen-Shannon divergence between two 1D sample sets
# Used to measure how much the posterior changes between inputs
# JS divergence = 0 → identical distributions
# JS divergence = 1 → maximally different (log2 base)
# =============================================================================
def js_divergence_1d(s1, s2, n_bins=30, lo=None, hi=None):
    lo = min(s1.min(), s2.min()) if lo is None else lo
    hi = max(s1.max(), s2.max()) if hi is None else hi
    bins  = np.linspace(lo, hi, n_bins + 1)
    h1, _ = np.histogram(s1, bins=bins, density=True)
    h2, _ = np.histogram(s2, bins=bins, density=True)
    h1 = h1 + 1e-10  # avoid log(0)
    h2 = h2 + 1e-10
    h1 /= h1.sum()
    h2 /= h2.sum()
    return jensenshannon(h1, h2)


# =============================================================================
# DIAGNOSTIC 1: Input Sensitivity Test
# If the posterior is identical for very different inputs, the flow has collapsed
# =============================================================================
def test_input_sensitivity(name, posterior, x_mean, x_std, idx):
    """
    Sample the posterior at four very different inputs and measure
    how much the output distribution changes across them.

    Returns:
        dict with sensitivity scores per parameter
    """
    print(f"\n  [Sensitivity] {name}")

    # Define test inputs:
    # 1. Normalised reference observation
    xo_n   = ((x_obs[idx] - x_mean) / x_std).astype(np.float32)

    # 2. A real training point near the edge of parameter space
    #    (find the training sample with largest theta_ej_0)
    max_idx = np.argmax(theta_train[train_idx, 0])
    x_edge1 = x_train[train_idx][max_idx, idx]
    x_edge1_n = ((x_edge1 - x_mean) / x_std).astype(np.float32)

    # 3. A real training point near the opposite edge (smallest theta_ej_0)
    min_idx = np.argmin(theta_train[train_idx, 0])
    x_edge2 = x_train[train_idx][min_idx, idx]
    x_edge2_n = ((x_edge2 - x_mean) / x_std).astype(np.float32)

    # 4. Pure Gaussian noise — should give maximally uninformative
    #    posterior if the flow is input-sensitive
    x_noise_n = np.random.randn(len(idx)).astype(np.float32)

    test_inputs = {
        'reference':  xo_n,
        'edge_high':  x_edge1_n,
        'edge_low':   x_edge2_n,
        'pure_noise': x_noise_n,
    }

    samples_per_input = {}
    stats_per_input   = {}

    for label, x_t in test_inputs.items():
        s = sample_posterior(posterior, x_t)
        if s is None:
            print(f"    {label:12s}: FAILED (all members degenerate)")
            continue
        # Clip to prior
        s = np.clip(s, PRIOR_LOW, PRIOR_HIGH)
        samples_per_input[label] = s
        stats_per_input[label] = {
            'mean': s.mean(axis=0),
            'std':  s.std(axis=0),
        }
        print(f"    {label:12s}: "
              f"theta_ej_0={s[:,0].mean():.3f}±{s[:,0].std():.3f}  "
              f"mu_beta={s[:,1].mean():.3f}±{s[:,1].std():.3f}")

    if len(samples_per_input) < 2:
        print(f"    Insufficient valid samples for sensitivity analysis.")
        return None

    # Compute JS divergence between reference and each other input
    # A well-trained flow should show HIGH divergence (different inputs → different posteriors)
    # A collapsed flow will show LOW divergence (all inputs → same posterior)
    keys = list(samples_per_input.keys())
    ref_key = 'reference' if 'reference' in keys else keys[0]
    ref_s   = samples_per_input[ref_key]

    js_scores = {}
    for label, s in samples_per_input.items():
        if label == ref_key:
            continue
        js_p0 = js_divergence_1d(
            ref_s[:, 0], s[:, 0], lo=PRIOR_LOW[0], hi=PRIOR_HIGH[0]
        )
        js_p1 = js_divergence_1d(
            ref_s[:, 1], s[:, 1], lo=PRIOR_LOW[1], hi=PRIOR_HIGH[1]
        )
        js_scores[label] = (js_p0, js_p1)
        print(f"    JS({ref_key} vs {label:12s}): "
              f"theta_ej_0={js_p0:.4f}  mu_beta={js_p1:.4f}")

    # VERDICT
    # Noise input should give a JS divergence >> 0 if the flow is input-sensitive
    noise_js = js_scores.get('pure_noise', (0, 0))
    COLLAPSE_THRESHOLD = 0.05   # JS < 0.05 for noise input → likely collapsed
    is_collapsed = all(js < COLLAPSE_THRESHOLD for js in noise_js)

    if is_collapsed:
        print(f"    ⚠️  COLLAPSE DETECTED: "
              f"JS(ref vs noise) = {noise_js} < {COLLAPSE_THRESHOLD}")
        print(f"       The flow is ignoring its input and outputting a fixed distribution.")
    else:
        print(f"    ✓  Flow appears input-sensitive.")

    return {
        'samples':    samples_per_input,
        'stats':      stats_per_input,
        'js_scores':  js_scores,
        'collapsed':  is_collapsed,
    }


# =============================================================================
# DIAGNOSTIC 2: Prior Comparison Test
# Compare posterior width to prior width — collapsed posteriors look like
# smooth Gaussians while priors are flat, but the widths can be similar
# =============================================================================
def test_prior_comparison(name, posterior, x_mean, x_std, idx):
    """
    Compare the posterior at the reference observation to the prior.
    A well-trained posterior should be substantially narrower than the prior.
    """
    print(f"\n  [Prior comparison] {name}")

    # Prior samples (uniform)
    prior_samples = np.column_stack([
        np.random.uniform(PRIOR_LOW[i], PRIOR_HIGH[i], N_PRIOR_SAMPLES)
        for i in range(len(PRIOR_LOW))
    ])
    prior_widths = prior_samples.std(axis=0)

    # Posterior samples at reference
    xo_n = ((x_obs[idx] - x_mean) / x_std).astype(np.float32)
    post_s = sample_posterior(posterior, xo_n)

    if post_s is None:
        print(f"    Sampling failed.")
        return None

    post_s = np.clip(post_s, PRIOR_LOW, PRIOR_HIGH)
    post_widths = post_s.std(axis=0)

    compression = prior_widths / (post_widths + 1e-8)

    for i, pname in enumerate(PARAM_NAMES):
        print(f"    {pname}: prior_std={prior_widths[i]:.4f}, "
              f"post_std={post_widths[i]:.4f}, "
              f"compression={compression[i]:.2f}x")
        if compression[i] < 1.5:
            print(f"      ⚠️  Low compression — posterior barely tighter than prior")
        elif compression[i] > 50:
            print(f"      ⚠️  Extreme compression — may indicate overconfident collapse")
        else:
            print(f"      ✓  Reasonable compression")

    return {
        'prior_widths': prior_widths,
        'post_widths':  post_widths,
        'compression':  compression,
        'post_samples': post_s,
    }


# =============================================================================
# DIAGNOSTIC 3: Per-member Ensemble Diversity
# If all members output the same distribution, the ensemble has degenerated
# =============================================================================
def test_ensemble_diversity(name, posterior, x_mean, x_std, idx):
    """
    Sample from each ensemble member independently and compare their outputs.
    Healthy ensembles show some spread between members.
    Collapsed ensembles have all members outputting the same distribution.
    """
    print(f"\n  [Ensemble diversity] {name}")

    try:
        members = posterior.posteriors
    except AttributeError:
        print(f"    Not an ensemble posterior — skipping.")
        return None

    xo_n = ((x_obs[idx] - x_mean) / x_std).astype(np.float32)

    member_means = []
    member_stds  = []

    for i, m in enumerate(members):
        s = sample_member(m, xo_n, n=500)
        if s is None:
            print(f"    Member {i}: FAILED")
            continue
        s = np.clip(s, PRIOR_LOW, PRIOR_HIGH)
        member_means.append(s.mean(axis=0))
        member_stds.append(s.std(axis=0))
        print(f"    Member {i}: "
              f"theta_ej_0={s[:,0].mean():.3f}±{s[:,0].std():.3f}  "
              f"mu_beta={s[:,1].mean():.3f}±{s[:,1].std():.3f}")

    if len(member_means) < 2:
        return None

    member_means = np.array(member_means)
    member_stds  = np.array(member_stds)

    # Spread of means across members — low spread → collapse
    mean_spread = member_means.std(axis=0)
    std_spread  = member_stds.std(axis=0)

    print(f"    Spread of member means:  "
          f"theta_ej_0={mean_spread[0]:.4f}  mu_beta={mean_spread[1]:.4f}")
    print(f"    Spread of member stds:   "
          f"theta_ej_0={std_spread[0]:.4f}   mu_beta={std_spread[1]:.4f}")

    DIVERSITY_THRESHOLD = 0.05
    if all(mean_spread < DIVERSITY_THRESHOLD):
        print(f"    ⚠️  Low ensemble diversity — all members agree suspiciously well.")
    else:
        print(f"    ✓  Ensemble shows healthy member diversity.")

    return {
        'member_means': member_means,
        'member_stds':  member_stds,
        'mean_spread':  mean_spread,
        'std_spread':   std_spread,
    }


# =============================================================================
# DIAGNOSTIC 4: Stat Map Index Verification
# Print the actual summary statistic values being passed to each posterior
# =============================================================================
def test_statmap_indices(name, idx, x_mean, x_std):
    """
    Print the actual unnormalised and normalised values at the reference
    observation for the indices used by this statistic combination.
    Helps catch index scrambling from the loop-merge bug.
    """
    print(f"\n  [Stat map check] {name} — indices: {idx}")

    raw_vals  = x_obs[idx]
    norm_vals = (raw_vals - x_mean) / (x_std + 1e-8)

    # Expected layout for the single-loop interleaved format
    # Each group of 6 is one scale: [3pt_y, 3pt_tau, 3pt_k, 2pt_y, 2pt_tau, 2pt_k]
    # All values at the reference should be non-zero and finite
    n_zero  = np.sum(np.abs(raw_vals) < 1e-12)
    n_inf   = np.sum(~np.isfinite(raw_vals))
    n_stats = len(idx)

    print(f"    n_stats={n_stats}, n_zero={n_zero}, n_nonfinite={n_inf}")
    print(f"    Raw    range: [{raw_vals.min():.4e}, {raw_vals.max():.4e}]")
    print(f"    Normed range: [{norm_vals.min():.4f}, {norm_vals.max():.4f}]")

    if n_zero > n_stats // 2:
        print(f"    ⚠️  More than half the statistics are zero — possible index error")
    if np.abs(norm_vals).max() > 10:
        print(f"    ⚠️  Normalised values exceed ±10 — possible scaler mismatch")
    if n_inf > 0:
        print(f"    ⚠️  Non-finite values in normalised statistics")

    # Sign check: 3pt stats <g²X> should be positive (they are squared * tracer)
    # 2pt stats <gX> can be positive or negative
    # For kappa stats specifically, sign can vary by scale
    print(f"    Raw values per index:")
    for i, (raw, norm) in enumerate(zip(raw_vals, norm_vals)):
        print(f"      idx[{i}]={idx[i]:3d}: raw={raw:.4e}  normed={norm:.4f}")

    return {'raw': raw_vals, 'normed': norm_vals}


# =============================================================================
# DIAGNOSTIC 5: Full Visualisation Plot
# One figure per statistic with all diagnostic information
# =============================================================================
def plot_diagnostics(name, sensitivity_result, prior_result, diversity_result, idx):
    """
    Produce a single summary figure for a given statistic with:
    - Posterior at different inputs (sensitivity test)
    - Comparison to prior width
    - Per-member spread
    """
    if sensitivity_result is None or prior_result is None:
        return

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Posterior Collapse Diagnostics: {name}  '
                 f'(n_stats={len(idx)})', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    samples_dict = sensitivity_result['samples']
    prior_s      = np.column_stack([
        np.random.uniform(PRIOR_LOW[i], PRIOR_HIGH[i], N_PRIOR_SAMPLES)
        for i in range(len(PRIOR_LOW))
    ])

    colors = {
        'reference':  'steelblue',
        'edge_high':  'tomato',
        'edge_low':   'seagreen',
        'pure_noise': 'orange',
        'prior':      'grey',
    }

    for p, pname in enumerate(PARAM_NAMES):
        lo, hi = PRIOR_LOW[p], PRIOR_HIGH[p]
        bins   = np.linspace(lo, hi, 40)

        # Row 0: marginal histograms per input
        ax = fig.add_subplot(gs[0, p * 2: p * 2 + 2])
        ax.hist(prior_s[:, p], bins=bins, density=True,
                histtype='step', color=colors['prior'],
                linewidth=1.5, linestyle='--', label='Prior', alpha=0.7)

        for label, s in samples_dict.items():
            ax.hist(s[:, p], bins=bins, density=True,
                    histtype='step', color=colors.get(label, 'k'),
                    linewidth=2, label=label, alpha=0.9)

        ax.axvline(x_obs[0] if p == 0 else x_obs[1],
                   color='k', linestyle=':', alpha=0.5, label='Truth (approx)')
        ax.set_xlabel(LABELS[p])
        ax.set_ylabel('Density')
        ax.set_title(f'{LABELS[p]}: Sensitivity Test')
        ax.legend(fontsize=7)
        ax.set_xlim(lo, hi)

    # Row 1 left: compression bar chart
    ax_comp = fig.add_subplot(gs[1, 0:2])
    compression = prior_result['compression']
    bars = ax_comp.bar(PARAM_NAMES, compression, color=['steelblue', 'tomato'], alpha=0.8)
    ax_comp.axhline(1.0, color='k',  ls='--', label='No compression (= prior)')
    ax_comp.axhline(1.5, color='orange', ls='--', alpha=0.7, label='Collapse threshold (1.5x)')
    ax_comp.set_ylabel('Prior std / Posterior std')
    ax_comp.set_title('Posterior Compression vs Prior')
    ax_comp.legend(fontsize=8)
    for bar, val in zip(bars, compression):
        ax_comp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.1f}x', ha='center', va='bottom', fontsize=10)

    # Row 1 right: JS divergence heatmap
    ax_js = fig.add_subplot(gs[1, 2:4])
    js_data = sensitivity_result['js_scores']
    labels_js = list(js_data.keys())
    js_matrix = np.array([[js_data[l][p] for l in labels_js]
                           for p in range(len(PARAM_NAMES))])

    im = ax_js.imshow(js_matrix, aspect='auto', cmap='viridis',
                      vmin=0, vmax=0.5)
    ax_js.set_xticks(range(len(labels_js)))
    ax_js.set_xticklabels(labels_js, rotation=30, ha='right', fontsize=8)
    ax_js.set_yticks(range(len(PARAM_NAMES)))
    ax_js.set_yticklabels(LABELS, fontsize=8)
    ax_js.set_title('JS Divergence from Reference\n(higher = more input-sensitive)')
    plt.colorbar(im, ax=ax_js)

    # Annotate cells
    for i in range(len(PARAM_NAMES)):
        for j in range(len(labels_js)):
            val = js_matrix[i, j]
            ax_js.text(j, i, f'{val:.3f}', ha='center', va='center',
                       color='white' if val < 0.25 else 'black', fontsize=9)

    collapsed = sensitivity_result.get('collapsed', False)
    status    = '⚠️ COLLAPSED' if collapsed else '✓ HEALTHY'
    fig.text(0.5, 0.02, f'Overall Status: {status}',
             ha='center', fontsize=13,
             color='red' if collapsed else 'green',
             fontweight='bold')

    outpath = DIAG_DIR / f'collapse_diagnostic_{name}.jpg'
    plt.savefig(outpath, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {outpath}")


# =============================================================================
# MAIN DIAGNOSTIC LOOP
# =============================================================================
summary = {}

for name, idx in STAT_MAP.items():
    print(f"\n{'='*65}")
    print(f"  DIAGNOSING: {name}  ({len(idx)} statistics)")
    print(f"{'='*65}")

    # Load posterior
    posterior_path = f'ili_posterior_{name}.pkl'
    if not os.path.exists(posterior_path):
        print(f"  [SKIP] No saved posterior found.")
        continue

    with open(posterior_path, 'rb') as f:
        posterior = pk.load(f)

    # Load scalers
    scaler_mean_path = f'scaler_{name}_mean.npy'
    scaler_std_path  = f'scaler_{name}_std.npy'
    if os.path.exists(scaler_mean_path) and os.path.exists(scaler_std_path):
        x_mean = np.load(scaler_mean_path)
        x_std  = np.load(scaler_std_path)
    else:
        # Recompute from training split
        xt_tr  = x_train[train_idx][:, idx]
        x_mean = xt_tr.mean(axis=0)
        x_std  = xt_tr.std(axis=0) + 1e-8
        print(f"  [INFO] Scaler recomputed from training split.")

    # Run all diagnostics
    idx_result         = test_statmap_indices(name, idx, x_mean, x_std)
    sensitivity_result = test_input_sensitivity(name, posterior, x_mean, x_std, idx)
    prior_result       = test_prior_comparison(name, posterior, x_mean, x_std, idx)
    diversity_result   = test_ensemble_diversity(name, posterior, x_mean, x_std, idx)

    # Visualise
    plot_diagnostics(name, sensitivity_result, prior_result, diversity_result, idx)

    # Collect summary
    collapsed   = (sensitivity_result or {}).get('collapsed', None)
    compression = (prior_result or {}).get('compression', np.array([np.nan, np.nan]))
    js_noise    = (sensitivity_result or {}).get('js_scores', {}).get('pure_noise', (np.nan, np.nan))

    summary[name] = {
        'collapsed':   collapsed,
        'compression': compression,
        'js_noise':    js_noise,
    }

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print(f"\n\n{'='*75}")
print(f"  COLLAPSE DIAGNOSTIC SUMMARY")
print(f"{'='*75}")
print(f"  {'Statistic':<16} {'Collapsed?':<12} "
      f"{'Compression_p0':>16} {'Compression_p1':>16} "
      f"{'JS_noise_p0':>13} {'JS_noise_p1':>13}")
print(f"  {'-'*74}")

for name, res in summary.items():
    status = '⚠️ YES' if res['collapsed'] else ('✓ No' if res['collapsed'] is False else '?')
    c      = res['compression']
    js     = res['js_noise']
    print(f"  {name:<16} {status:<12} "
          f"{c[0]:>16.2f} {c[1]:>16.2f} "
          f"{js[0]:>13.4f} {js[1]:>13.4f}")

print(f"\n  Compression < 1.5x  so  posterior barely tighter than prior (likely collapsed)")
print(f"  JS(noise) < 0.05, i.e.,  flow ignoring input (likely collapsed)")
print(f"\n  Diagnostic plots saved to: {DIAG_DIR}/")
