import os
import pickle as pk
import torch
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import logging

logging.getLogger('getdist').setLevel(logging.ERROR)

# =============================================================================
# 1. SETUP
# =============================================================================
labels      = [r"\theta_{ej,0}", r"\mu_{\beta}"]
names       = ["p1", "p2"]
truth_values = [2.0, 0.6]

def get_raw_samples(name):
    pkl_path  = f'ili_posterior_{name}.pkl'
    xobs_path = f'xobs_{name}.npy'
    if not (os.path.exists(pkl_path) and os.path.exists(xobs_path)):
        print(f"Warning: Missing files for {name}. Skipping...")
        return None
    with open(pkl_path, 'rb') as f:
        posterior = pk.load(f)
    xo = torch.from_numpy(np.load(xobs_path)).float().reshape(1, -1)
    return posterior.sample((30000,), x=xo).detach().cpu().numpy().squeeze()

os.makedirs("plotcontours2params", exist_ok=True)

# =============================================================================
# 2. AUTO-MOMENT COMPARISON: yy, tt, kk, all_auto
# =============================================================================
print("\nGenerating Plot: Auto-moment Comparison (yy, tt, kk, all_auto)...")

samples_yy       = get_raw_samples('yy')
samples_tt       = get_raw_samples('tt')
samples_kk       = get_raw_samples('kk')
samples_all_auto = get_raw_samples('all_auto')

# Check all loaded successfully
auto_samples = {
    'yy':       samples_yy,
    'tt':       samples_tt,
    'kk':       samples_kk,
    'all_auto': samples_all_auto,
}
for name, s in auto_samples.items():
    if s is None:
        raise RuntimeError(
            f"Could not load samples for '{name}'. "
            f"Check that ili_posterior_{name}.pkl and xobs_{name}.npy exist."
        )

#   yy  --> red   (matches y_total in tracer plot)
#   tt  --> blue  (matches tau_total in tracer plot)
#   kk  --> green (matches kappa_total in tracer plot)
# all_auto --> purple: high contrast to all three, not used elsewhere
auto_colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']

mcs_auto = [
    MCSamples(
        samples=samples_yy,
        names=names, labels=labels,
        label=r'$\langle y^2 \rangle$',
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1}
    ),
    MCSamples(
        samples=samples_tt,
        names=names, labels=labels,
        label=r'$\langle \tau^2 \rangle$',
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1}
    ),
    MCSamples(
        samples=samples_kk,
        names=names, labels=labels,
        label=r'$\langle \kappa^2 \rangle$',
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1}
    ),
    MCSamples(
        samples=samples_all_auto,
        names=names, labels=labels,
        label=r'$\langle y^2 \rangle + \langle \tau^2 \rangle + \langle \kappa^2 \rangle$',
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1}
    ),
]

g = plots.get_subplot_plotter(width_inch=10)
g.triangle_plot(
    mcs_auto,
    filled=True,
    colors=auto_colors,
    legend_labels=[
        r'$y^2$',
        r'$\tau^2$',
        r'$\kappa^2$',
        r'$y^2 + \tau^2 + \kappa^2$',
    ],
    contour_args=[{'alpha': 0.6} for _ in auto_colors],
    line_args=[{'lw': 2.5, 'color': c} for c in auto_colors],
)

# Truth lines
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

g.export('plotcontours2params/auto_comparison.png')
print("  Saved to plotcontours2params/auto_comparison.png")

plt.close('all')
os._exit(0)
