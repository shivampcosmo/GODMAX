import os
import sys
import pickle as pk
import torch
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import logging
from pathlib import Path

# Silence GetDist logging
logging.getLogger('getdist').setLevel(logging.ERROR)

import matplotlib as mpl
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 7
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 7
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

sys.path.append('/work/hdd/bdne/aacharya2/ltu-ili')
from ili.validation.metrics import PlotSinglePosterior

logging.getLogger('getdist').setLevel(logging.ERROR)
from getdist.plots import GetDistPlotSettings
# Create a settings object first
plot_settings = GetDistPlotSettings()
plot_settings.axes_fontsize = 20
plot_settings.lab_fontsize = 22
plot_settings.legend_fontsize = 20
plot_settings.title_limit_fontsize = 18

# =============================================================================
# 0. SANITY CHECK
# =============================================================================
try:
    theta_check = np.load('theta.npy')
    x_check     = np.load('x_JOINT.npy')
    print("--- SBI Dataset Verification ---")
    print(f"Parameters shape: {theta_check.shape}")
    print(f"Statistics shape: {x_check.shape}")
    print(f"Loaded {theta_check.shape[0]} samples.")
    print("--------------------------------")
except FileNotFoundError as e:
    print(f"Initialization Error: {e}")

# =============================================================================
# 1. SETUP
# =============================================================================
def sample_ensemble_direct(posterior, x_obs_norm, n_samples=5000):
    """
    Sample from each member of an EnsemblePosterior individually and
    concatenate. Avoids the rejection-sampling hang that
    posterior.sample((N,), x=...) triggers when called on the full ensemble.

    Each member gets n_samples // n_members draws so the total is ~n_samples.
    The remainder is drawn from the first member to hit exactly n_samples.
    """
    x_tensor = torch.from_numpy(np.array(x_obs_norm)).float().reshape(1, -1)
    members  = getattr(posterior, 'posteriors', [posterior])
    n_mem    = len(members)
    base     = n_samples // n_mem
    extra    = n_samples - base * n_mem   # assigned to member 0

    all_samples = []
    for i, member in enumerate(members):
        n_draw = base + (extra if i == 0 else 0)
        try:
            s = member.sample((n_draw,), x=x_tensor)
            all_samples.append(s.detach().cpu().numpy())
        except Exception as e:
            print(f'  [WARN] Member {i} sampling failed: {e}')

    if not all_samples:
        return None
    return np.vstack(all_samples)

# PlotSinglePosterior uses $...$ format
labels      = [r'$\theta_{ej,0}$', r'${\nu_{\theta_{ej}}}^{M}$']
param_names = ['p1', 'p2']
truth_values = np.array([2.0, -0.1])

comparison_colors = ['#1f77b4', '#d62728', '#2ca02c']

stat_map_keys = [
    'g2y', 'g2tau', 'g2kappa',
    'gy',  'gtau',  'gkappa',
    'JOINT', 'y_total', 'tau_total', 'kappa_total',
    'all_3pt', 'all_2pt',
]

tracer_groups = [
    {
        'title': 'y_correlations',
        'pairs': [('g2y', '3-point'), ('gy', '2-point'), ('y_total', 'all_y')],
        'legend_labels': [r'$g^2 y$', r'$gy$', r'$g^2 y + gy$'],
    },
    {
        'title': 'tau_correlations',
        'pairs': [('g2tau', '3-point'), ('gtau', '2-point'), ('tau_total', 'all_tau')],
        'legend_labels': [r'$g^2 \tau$', r'$g\tau$', r'$g^2 \tau + g\tau$'],
    },
    {
        'title': 'kappa_correlations',
        'pairs': [('g2kappa', '3-point'), ('gkappa', '2-point'), ('kappa_total', 'all_kappa')],
        'legend_labels': [r'$g^2 \kappa$', r'$g\kappa$', r'$g^2 \kappa + g\kappa$'],
    },
]

os.makedirs('plotcontours2params', exist_ok=True)

# =============================================================================
# 2. INDIVIDUAL POSTERIOR PLOTS — PlotSinglePosterior
# =============================================================================
# sample_method='direct' internally calls each ensemble member's .sample()
# separately, which is equivalent to sample_ensemble_direct and avoids the
# rejection-sampling hang that posterior.sample() on EnsemblePosterior triggers.

for name in stat_map_keys:
    pkl_path  = f'ili_posterior_{name}.pkl'
    xobs_path = f'xobs_{name}.npy'

    if not (os.path.exists(pkl_path) and os.path.exists(xobs_path)):
        print(f'[SKIP] Missing files for {name}')
        continue

    with open(pkl_path, 'rb') as f:
        posterior = pk.load(f)

    x_obs_norm = np.load(xobs_path)
    out_dir    = Path(f'plotcontours2params/{name}')
    out_dir.mkdir(exist_ok=True)

    metric = PlotSinglePosterior(
        num_samples=5000,
        sample_method='direct',   # avoids rejection-sampling hang
        labels=labels,
        out_dir=out_dir,
    )

    try:
        fig = metric(
            posterior=posterior,
            x_obs=x_obs_norm,
            theta_fid=truth_values,  # plotted as fiducial marker automatically
            plot_kws=dict(fill=True),
        )
        print(f'[DONE] {name} --> {out_dir}')
    except Exception as e:
        print(f'[WARN] PlotSinglePosterior failed for {name}: {e}')

    plt.close('all')

# =============================================================================
# 3. GROUP COMPARISON PLOTS: GetDist with corrected sampling
# =============================================================================
# ltu-ili has no multi-posterior overlay API, so GetDist is still used here.
# The critical fix vs the old code: sample_ensemble_direct instead of
# posterior.sample() which hangs on EnsemblePosterior.
labels      = [r'\theta_{ej,0}', r'{\nu_{\theta_{ej}}}^{M}']
def get_mcsamples(name, legend_label, n_samples=5000):
    """
    Sample from an ensemble posterior using sample_ensemble_direct
    (per-member direct sampling with timeout), then return MCSamples.
    """
    pkl_path  = f'ili_posterior_{name}.pkl'
    xobs_path = f'xobs_{name}.npy'

    if not (os.path.exists(pkl_path) and os.path.exists(xobs_path)):
        print(f'[SKIP] Missing files for {name}')
        return None

    with open(pkl_path, 'rb') as f:
        posterior = pk.load(f)

    x_obs_norm = np.load(xobs_path)
    samples    = sample_ensemble_direct(posterior, x_obs_norm, n_samples=n_samples)

    if samples is None:
        print(f'[WARN] Sampling failed for {name}')
        return None

    return MCSamples(
        samples=samples,
        names=param_names,
        labels=labels,
        label=legend_label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )


for group in tracer_groups:
    print(f"\nGenerating comparison plot for {group['title']}...")
    mcsamples_list = []

    for tracer_id, legend_tag in group['pairs']:
        smp = get_mcsamples(tracer_id, f'{tracer_id} ({legend_tag})')
        if smp is not None:
            mcsamples_list.append(smp)

    if len(mcsamples_list) < 2:
        print(f'[SKIP] Fewer than 2 valid posteriors for {group["title"]}')
        continue

    n = len(mcsamples_list)
    g = plots.get_subplot_plotter(width_inch=8)
    g.triangle_plot(
        mcsamples_list,
        filled=True,
        colors=comparison_colors[:n],
        legend_labels=group['legend_labels'][:n],
        contour_args=[{'alpha': 0.6}] * n,
        line_args=[{'lw': 2.5, 'color': c} for c in comparison_colors[:n]],
    )

    # Truth lines — same as before
    for i in range(len(param_names)):
        g.subplots[i, i].axvline(truth_values[i], color='black', ls=':', lw=2, zorder=10)
        for j in range(i):
            g.subplots[i, j].axvline(truth_values[j], color='black', ls=':', lw=2, zorder=10)
            g.subplots[i, j].axhline(truth_values[i], color='black', ls=':', lw=2, zorder=10)

    output_fn = f'plotcontours2params/contour_comparison_{group["title"]}.png'
    g.export(output_fn)
    print(f'[DONE] Saved: {output_fn}')

plt.close('all')
os._exit(0)
