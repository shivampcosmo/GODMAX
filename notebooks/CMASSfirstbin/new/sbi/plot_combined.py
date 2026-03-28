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

plot_settings = GetDistPlotSettings()
plot_settings.axes_fontsize        = 20
plot_settings.lab_fontsize         = 22
plot_settings.legend_fontsize      = 20
plot_settings.title_limit_fontsize = 18

# =============================================================================
# 1. SETUP
# =============================================================================
labels      = [r'\theta_{ej,0}', r'{\nu_{\theta_{ej}}}^{M}']
names       = ['p1', 'p2']
truth_values = [2.0, -0.1]

os.makedirs('plotcontours2params', exist_ok=True)

# =============================================================================
# 2. SAMPLING HELPER
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


def get_raw_samples(name, n_samples=5000):
    """
    Load posterior and xobs for a given statistic name and return
    a (n_samples, n_params) numpy array via sample_ensemble_direct.
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
        print(f'[WARN] All members failed for {name}')
    return samples


def make_mcs(samples, label):
    """Wrap a numpy sample array in an MCSamples object."""
    return MCSamples(
        samples=samples,
        names=names,
        labels=labels,
        label=label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )


def add_truth_lines(g, truth_values, names):
    """Draw dashed truth lines on all 1D and 2D subplots."""
    for i in range(len(names)):
        g.subplots[i, i].axvline(
            truth_values[i], color='black', ls='--', lw=2, zorder=10
        )
        for j in range(i):
            g.subplots[i, j].axvline(
                truth_values[j], color='black', ls='--', lw=2, zorder=10
            )
            g.subplots[i, j].axhline(
                truth_values[i], color='black', ls='--', lw=2, zorder=10
            )


def make_triangle(mcs_list, colors, legend_labels, output_path):
    """Build and save a GetDist triangle plot."""
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
    add_truth_lines(g, truth_values, names)
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
    make_mcs(s_2pt,   '2pt')      if s_2pt   is not None else None,
    make_mcs(s_3pt,   '3pt')      if s_3pt   is not None else None,
    make_mcs(s_joint, '2pt+3pt')  if s_joint is not None else None,
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
    make_mcs(s_y,     r'$y$ (2pt+3pt)')       if s_y     is not None else None,
    make_mcs(s_tau,   r'$\tau$ (2pt+3pt)')    if s_tau   is not None else None,
    make_mcs(s_kappa, r'$\kappa$ (2pt+3pt)')  if s_kappa is not None else None,
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
