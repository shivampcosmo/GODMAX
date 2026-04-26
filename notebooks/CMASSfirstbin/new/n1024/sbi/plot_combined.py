import os
import pickle as pk
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from getdist import plots, MCSamples
from getdist.plots import GetDistPlotSettings
import logging

logging.getLogger('getdist').setLevel(logging.ERROR)

mpl.rcParams['xtick.major.size']  = 10
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size']  = 7
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size']  = 10
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size']  = 7
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.direction']   = 'in'
mpl.rcParams['ytick.direction']   = 'in'
mpl.rcParams['font.size']         = 20
mpl.rcParams['axes.labelsize']    = 22
mpl.rcParams['legend.fontsize']   = 20
mpl.rcParams['xtick.labelsize']   = 16
mpl.rcParams['ytick.labelsize']   = 16

plot_settings = GetDistPlotSettings()
plot_settings.axes_fontsize        = 20
plot_settings.lab_fontsize         = 22
plot_settings.legend_fontsize      = 20
plot_settings.title_limit_fontsize = 18

# =============================================================================
# 1. SETUP
# =============================================================================
labels       = [r'\theta_{ej,0}', r'{\nu_{\theta_{ej}}}^{M}']
names        = ['p1', 'p2']
truth_values = [2.0, -0.1]

# Axis limits as a list parallel to `names` — avoids any key-mismatch with
# GetDist's param_limits dict (which expects MCSamples name strings, here p1/p2)
PARAM_LIMITS = [
    (1.0,  6.0),   # p1 : theta_ej_0
    (-0.3, 0.0),   # p2 : nu_theta_ej_M
]

os.makedirs('plotcontours2params', exist_ok=True)

import io

class _CPUUnpickler(pk.Unpickler):
    """Unpickler that remaps all torch tensors to CPU regardless of where
    they were saved."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(
                io.BytesIO(b), map_location='cpu', weights_only=False
            )
        return super().find_class(module, name)

def load_posterior(path):
    with open(path, 'rb') as f:
        return _CPUUnpickler(f).load()

# =============================================================================
# 2. SAMPLING HELPER
# =============================================================================

def sample_ensemble_direct(posterior, x_obs_norm, n_samples=5000):
    x_tensor = torch.from_numpy(np.array(x_obs_norm)).float().reshape(1, -1)
    members  = getattr(posterior, 'posteriors', [posterior])
    n_mem    = len(members)
    base     = n_samples // n_mem
    extra    = n_samples - base * n_mem

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


def get_raw_samples(name, n_samples=5000):
    pkl_path  = f'ili_posterior_{name}.pkl'
    xobs_path = f'xobs_{name}.npy'

    if not (os.path.exists(pkl_path) and os.path.exists(xobs_path)):
        print(f'[SKIP] Missing files for {name}')
        return None

    posterior  = load_posterior(pkl_path)
    x_obs_norm = np.load(xobs_path)
    samples    = sample_ensemble_direct(posterior, x_obs_norm, n_samples=n_samples)

    if samples is None:
        print(f'[WARN] All members failed for {name}')
    return samples


def make_mcs(samples, label):
    return MCSamples(
        samples=samples,
        names=names,
        labels=labels,
        label=label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )


def make_triangle(mcs_list, colors, legend_labels, output_path,
                  param_limits=PARAM_LIMITS, truth_values=truth_values):
    """
    Build and save a GetDist triangle plot.

    Axis limits are applied manually after triangle_plot() returns because
    GetDist's param_limits argument is unreliable across versions and is
    overridden by the data range during internal rendering.

    truth_values and param_limits are both ordered parallel to `names`.
    """
    n = len(mcs_list)
    g = plots.get_subplot_plotter(width_inch=8, settings=plot_settings)
    g.triangle_plot(
        mcs_list,
        filled=True,
        colors=colors[:n],
        legend_labels=legend_labels[:n],
        contour_args=[{'alpha': 0.6}] * n,
        line_args=[{'lw': 2.5, 'color': c} for c in colors[:n]],
    )

    n_params = len(names)
    for i in range(n_params):
        lo_i, hi_i = param_limits[i]

        # Diagonal: 1D marginal — set xlim and truth vertical line
        g.subplots[i, i].set_xlim(lo_i, hi_i)
        g.subplots[i, i].axvline(
            truth_values[i], color='black', ls='--', lw=2, zorder=10)

        for j in range(i):
            lo_j, hi_j = param_limits[j]

            # Off-diagonal: 2D joint — set both axes and both truth lines
            g.subplots[i, j].set_xlim(lo_j, hi_j)
            g.subplots[i, j].set_ylim(lo_i, hi_i)
            g.subplots[i, j].axvline(
                truth_values[j], color='black', ls='--', lw=2, zorder=10)
            g.subplots[i, j].axhline(
                truth_values[i], color='black', ls='--', lw=2, zorder=10)

    g.export(output_path)
    print(f'[DONE] Saved: {output_path}')


# =============================================================================
# 3. PLOT 1: CATEGORY COMPARISON  (all_2pt / all_3pt / JOINT)
# =============================================================================
print('\nGenerating Plot 1: Information Category Comparison...')

s_2pt   = get_raw_samples('all_2pt')
s_3pt   = get_raw_samples('all_3pt')
s_joint = get_raw_samples('JOINT')

mcs_cat = [m for m in [
    make_mcs(s_2pt,   '2pt')     if s_2pt   is not None else None,
    make_mcs(s_3pt,   '3pt')     if s_3pt   is not None else None,
    make_mcs(s_joint, '2pt+3pt') if s_joint is not None else None,
] if m is not None]

if len(mcs_cat) >= 2:
    make_triangle(
        mcs_list      = mcs_cat,
        colors        = ['#d62728', '#1f77b4', '#2ca02c'],
        legend_labels = ['2pt', '3pt', '2pt+3pt'],
        output_path   = 'plotcontours2params/category_comparison.png',
    )
else:
    print('[SKIP] Plot 1: fewer than 2 valid posteriors.')

# =============================================================================
# 4. PLOT 2: TRACER-WISE COMPARISON  (y_total / tau_total / kappa_total)
# =============================================================================
print('\nGenerating Plot 2: Tracer-wise (2pt+3pt) Comparison...')

s_y     = get_raw_samples('y_total')
s_tau   = get_raw_samples('tau_total')
s_kappa = get_raw_samples('kappa_total')

mcs_tracers = [m for m in [
    make_mcs(s_y,     r'$y$ (2pt+3pt)')      if s_y     is not None else None,
    make_mcs(s_tau,   r'$\tau$ (2pt+3pt)')   if s_tau   is not None else None,
    make_mcs(s_kappa, r'$\kappa$ (2pt+3pt)') if s_kappa is not None else None,
] if m is not None]

if len(mcs_tracers) >= 2:
    make_triangle(
        mcs_list      = mcs_tracers,
        colors        = ['#d62728', '#1f77b4', '#2ca02c'],
        legend_labels = [r'$y$ (2pt+3pt)', r'$\tau$ (2pt+3pt)', r'$\kappa$ (2pt+3pt)'],
        output_path   = 'plotcontours2params/tracer_comparison.png',
    )
else:
    print('[SKIP] Plot 2: fewer than 2 valid posteriors.')

plt.close('all')
os._exit(0)
