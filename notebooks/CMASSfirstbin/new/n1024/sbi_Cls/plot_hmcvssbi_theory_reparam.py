import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from getdist import plots, MCSamples
from getdist.plots import GetDistPlotSettings
import logging
from pathlib import Path

logging.getLogger('getdist').setLevel(logging.ERROR)

# =============================================================================
# PLOT STYLE
# =============================================================================
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
mpl.rcParams['legend.fontsize']   = 18
mpl.rcParams['xtick.labelsize']   = 16
mpl.rcParams['ytick.labelsize']   = 16

plot_settings = GetDistPlotSettings()
plot_settings.axes_fontsize        = 20
plot_settings.lab_fontsize         = 22
plot_settings.legend_fontsize      = 18
plot_settings.title_limit_fontsize = 18

# =============================================================================
# PARAMETERS
# =============================================================================
_phi_fid = 2.0 * np.cbrt(-0.1)       # ~ -0.9283
_phi_min = 6.0 * np.cbrt(-0.3)       # ~ -4.016
_phi_max = 0.0
_t_min, _t_max   = 1.0, 6.0
_nu_min, _nu_max = -0.3, 0.0

PARAM_NAMES     = ['theta_ej_0', 'phi']
PARAM_LABELS_GD = [r'\theta_{\rm ej,0}', r'\theta_{\rm ej,0}\,(\nu_{\theta_{\rm ej}}^{M})^{1/3}']
TRUTH_VALUES    = np.array([2.0, _phi_fid])
PARAM_LIMITS    = [(_t_min, _t_max), (_phi_min, _phi_max)]

HMC_COLOR  = '#1f77b4'   # blue
SBI_COLOR  = '#d62728'   # red
HMC_1D_LW  = 3.0
SBI_1D_LW  = 3.0
HMC_ALPHA  = 0.4
SBI_ALPHA  = 0.6

PROBES = ['gy', 'gtau', 'gkappa', 'all_2pt']

THEORY_SBI_BASE = Path('outputs') / 'theory_sbi'

OUTPUT_DIR = 'plotcontours_reparam'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# SUPPORT FILTER
# =============================================================================
def _in_support(t, phi, prior_min, prior_max):
    return (
        (t   >= prior_min[0]) & (t   <= prior_max[0]) &
        (phi >= prior_min[1]) & (phi <= prior_max[1])
    )
# =============================================================================
# LOADERS
# =============================================================================
def load_hmc_mcsamples(probe, label='HMC / NUTS (theory)'):
    path = THEORY_SBI_BASE / f'{probe}_linearized_reparam' / 'hmc_samples.npz'
    if not path.exists():
        print(f'[SKIP] Theory HMC not found: {path}')
        return None

    data = np.load(path, allow_pickle=True)
    try:
        t   = np.asarray(data['samples_theta_ej_0']).ravel()
        phi = np.asarray(data['samples_phi']).ravel()
    except KeyError as e:
        print(f'[ERROR] Missing key in {path}: {e}')
        print(f'        Available keys: {list(data.keys())}')
        return None

    prior_min = np.asarray(data['prior_min'])
    prior_max = np.asarray(data['prior_max'])
    samples = np.column_stack([t, phi])
    mask    = _in_support(t, phi, prior_min, prior_max)
    samples = samples[mask]
    print(f'  [{probe}] HMC: {samples.shape[0]} samples (of {len(t)}) inside prior box  '
          f'theta_ej_0={samples[:,0].mean():.3f}+/-{samples[:,0].std():.3f}  '
          f'phi={samples[:,1].mean():.3f}+/-{samples[:,1].std():.3f}')

    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_LABELS_GD,
        label=label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )


def load_sbi_mcsamples(probe, label='SBI / NPE+MDN (theory)'):
    path = THEORY_SBI_BASE / f'{probe}_linearized_fisher_mdn5_reparam' / 'sbi_posterior_samples.npz'
    if not path.exists():
        print(f'[SKIP] Theory SBI not found: {path}')
        return None

    data = np.load(path, allow_pickle=True)
    try:
        samples_all = np.asarray(data['samples'], dtype=float)  # (theta_ej_0, phi), inference basis
    except KeyError as e:
        print(f'[ERROR] Missing key in {path}: {e}')
        print(f'        Available keys: {list(data.keys())}')
        return None

    prior_min = np.asarray(data['prior_min'])
    prior_max = np.asarray(data['prior_max'])
    t   = samples_all[:, 0]
    phi = samples_all[:, 1]
    mask    = _in_support(t, phi, prior_min, prior_max)
    samples = samples_all[mask]
    print(f'  [{probe}] SBI: {samples.shape[0]} samples (of {len(t)}) inside prior box  '
          f'theta_ej_0={samples[:,0].mean():.3f}+/-{samples[:,0].std():.3f}  '
          f'phi={samples[:,1].mean():.3f}+/-{samples[:,1].std():.3f}')

    return MCSamples(
        samples=samples,
        names=PARAM_NAMES,
        labels=PARAM_LABELS_GD,
        label=label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1},
    )

# =============================================================================
# MAIN PLOTTING LOOP
# =============================================================================
for probe in PROBES:
    print(f'\n── Probe: {probe} ──')

    hmc_smp = load_hmc_mcsamples(probe)
    sbi_smp = load_sbi_mcsamples(probe)

    entries = [
        (hmc_smp, HMC_COLOR, HMC_ALPHA, HMC_1D_LW, 'HMC / NUTS'),
        (sbi_smp, SBI_COLOR, SBI_ALPHA, SBI_1D_LW, 'SBI / NPE+MDN'),
    ]
    present = [(s, c, a, lw, lbl) for s, c, a, lw, lbl in entries if s is not None]

    if not present:
        print(f'[SKIP] No data available for {probe}')
        continue

    mcsamples_list = [s   for s, c, a, lw, lbl in present]
    colors         = [c   for s, c, a, lw, lbl in present]
    alphas         = [a   for s, c, a, lw, lbl in present]
    linewidths     = [lw  for s, c, a, lw, lbl in present]
    legend_labels  = [lbl for s, c, a, lw, lbl in present]

    g = plots.get_subplot_plotter(width_inch=8, settings=plot_settings)
    g.triangle_plot(
        mcsamples_list,
        filled=True,
        colors=colors,
        legend_labels=legend_labels,
        contour_args=[{'alpha': a} for a in alphas],
        line_args=[{'lw': lw, 'color': c} for lw, c in zip(linewidths, colors)],
    )

    # ── Axis limits and truth lines ───────────────────────────────────────────
    for i, (lo_i, hi_i) in enumerate(PARAM_LIMITS):
        ax_diag = g.subplots[i, i]
        ax_diag.set_xlim(lo_i, hi_i)
        ax_diag.axvline(TRUTH_VALUES[i], color='black', ls='--', lw=2, zorder=10)

        for j in range(i):
            lo_j, hi_j = PARAM_LIMITS[j]
            ax_off = g.subplots[i, j]
            ax_off.set_xlim(lo_j, hi_j)
            ax_off.set_ylim(lo_i, hi_i)
            ax_off.axvline(TRUTH_VALUES[j], color='black', ls='--', lw=2, zorder=10)
            ax_off.axhline(TRUTH_VALUES[i], color='black', ls='--', lw=2, zorder=10)

    out_path = os.path.join(OUTPUT_DIR, f'reparam_{probe}.png')
    g.export(out_path, dpi=200)
    print(f'[DONE] Saved: {out_path}')
    plt.close('all')

print('\nAll plots complete.')
os._exit(0)
