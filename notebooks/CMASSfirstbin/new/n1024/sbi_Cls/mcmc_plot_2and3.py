import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
from getdist import plots, loadMCSamples
from getdist.plots import GetDistPlotSettings

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
# CONFIG
# =============================================================================
WORK_DIR   = os.path.dirname(os.path.abspath(__file__))
CHAINS_DIR = os.path.join(WORK_DIR, 'chains')

# Cobaya uses these exact strings as parameter names
param_names  = ['theta_ej_0', 'nu_theta_ej_M']
labels_gd    = [r'\theta_{ej,0}', r'{\nu_{\theta_{ej}}}^{M}']
truth_values = np.array([2.0, -0.1])
PARAM_LIMITS = {
    'theta_ej_0':    (1.0,  6.0),
    'nu_theta_ej_M': (-0.3, 0.0),
}

COMPARISON_COLORS = ['#1f77b4', '#d62728', '#2ca02c']

tracer_groups = [
    {
        'title': 'y_correlations',
        'pairs': [('g2y', '3-point'), ('gy', '2-point'), ('y_total', 'all_y')],
        'legend_labels': [r'$g^2 y$', r'$gy$', r'$g^2 y + gy$'],
    },
    {
        'title': 'tau_correlations',
        'pairs': [ ('gtau', '2-point')],
        'legend_labels': [ r'$g\tau$'],
    },
    {
        'title': 'kappa_correlations',
        'pairs': [('g2kappa', '3-point'), ('gkappa', '2-point'), ('kappa_total', 'all_kappa')],
        'legend_labels': [r'$g^2 \kappa$', r'$g\kappa$', r'$g^2 \kappa + g\kappa$'],
    },
]

os.makedirs('plotcontours_MCMC', exist_ok=True)

# =============================================================================
# LOAD CHAINS
# =============================================================================
def get_mcsamples(stat_name, legend_label):
    chain_prefix = os.path.join(CHAINS_DIR, stat_name, 'mcmc')
    # Cobaya writes mcmc.1.txt, mcmc.2.txt, ... check at least one exists
    if not any(os.path.exists(f'{chain_prefix}.{i}.txt') for i in range(1, 10)):
        print(f'[SKIP] No chain files found for {stat_name} at {chain_prefix}')
        return None
    try:
        gd = loadMCSamples(
            chain_prefix,
            settings={'ignore_rows': 0.3, 'smooth_scale_2D': 0.7,
                      'boundary_correction_order': 1},
        )
        # Override labels to match the other plotting scripts
        for i, (name, label) in enumerate(zip(param_names, labels_gd)):
            p = gd.getParamNames().parWithName(name)
            if p is not None:
                p.label = label
        gd.label = legend_label
        return gd
    except Exception as e:
        print(f'[WARN] Failed to load chains for {stat_name}: {e}')
        return None

# =============================================================================
# COMPARISON PLOTS
# =============================================================================
for group in tracer_groups:
    print(f"\nGenerating comparison plot for {group['title']}...")
    mcsamples_list = []

    for stat_name, legend_tag in group['pairs']:
        smp = get_mcsamples(stat_name, f'{stat_name} ({legend_tag})')
        if smp is not None:
            mcsamples_list.append(smp)

    if len(mcsamples_list) < 2:
        print(f"[SKIP] Fewer than 2 valid posteriors for {group['title']}")
        continue

    n = len(mcsamples_list)
    g = plots.get_subplot_plotter(width_inch=8, settings=plot_settings)
    g.triangle_plot(
        mcsamples_list,
        params=param_names,
        filled=True,
        colors=COMPARISON_COLORS[:n],
        legend_labels=group['legend_labels'][:n],
        contour_args=[{'alpha': 0.6}] * n,
        line_args=[{'lw': 2.5, 'color': c} for c in COMPARISON_COLORS[:n]],
    )

    for i in range(len(param_names)):
        lo_i, hi_i = PARAM_LIMITS[param_names[i]]
        g.subplots[i, i].set_xlim(lo_i, hi_i)
        g.subplots[i, i].axvline(truth_values[i], color='black', ls='--', lw=2, zorder=10)
        for j in range(i):
            lo_j, hi_j = PARAM_LIMITS[param_names[j]]
            g.subplots[i, j].set_xlim(lo_j, hi_j)
            g.subplots[i, j].set_ylim(lo_i, hi_i)
            g.subplots[i, j].axvline(truth_values[j], color='black', ls='--', lw=2, zorder=10)
            g.subplots[i, j].axhline(truth_values[i], color='black', ls='--', lw=2, zorder=10)

    output_fn = f"plotcontours_MCMC/contour_comparison_{group['title']}.png"
    g.export(output_fn)
    print(f'[DONE] Saved: {output_fn}')

plt.close('all')
print('\nDone.')
os._exit(0)
