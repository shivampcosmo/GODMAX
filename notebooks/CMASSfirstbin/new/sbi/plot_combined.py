import os
import pickle as pk
import torch
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import logging

# Silence GetDist logging
logging.getLogger('getdist').setLevel(logging.ERROR)

# =============================================================================
# 1. SETUP (2-PARAMETER CASE)
# =============================================================================
labels = [r"\theta_{ej,0}", r"{\nu_{\theta_{ej}}}^{M}"]
names = ["p1", "p2"]
truth_values = [2.0, -0.1]

def get_raw_samples(name):
    pkl_path = f'ili_posterior_{name}.pkl'
    xobs_path = f'xobs_{name}.npy'
    if not (os.path.exists(pkl_path) and os.path.exists(xobs_path)):
        print(f"Warning: Missing files for {name}. Skipping...")
        return None
    with open(pkl_path, 'rb') as f:
        posterior = pk.load(f)
    xo = torch.from_numpy(np.load(xobs_path)).float().reshape(1, -1)
    # 30k samples for smooth contours
    return posterior.sample((30000,), x=xo).detach().cpu().numpy().squeeze()

os.makedirs("plotcontours2params", exist_ok=True)

# =============================================================================
# 2. PLOT 1: CATEGORY COMPARISON (Dedicated All 2pt vs All 3pt vs JOINT)
# =============================================================================
print("\nGenerating Plot 1: Information Category Comparison (Meta-Joints)...")

# Use dedicated posterior samples instead of vstack
samples_2pt = get_raw_samples('all_2pt')
samples_3pt = get_raw_samples('all_3pt')
samples_joint = get_raw_samples('JOINT')

mcs_cat = [
    MCSamples(samples=samples_2pt, names=names, labels=labels, label=r'2pt',
              settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1}),
    MCSamples(samples=samples_3pt, names=names, labels=labels, label=r'3pt',
              settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1}),
    MCSamples(samples=samples_joint, names=names, labels=labels, label='2pt+3pt',
              settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1})
]

cat_colors = ['#d62728', '#1f77b4', '#2ca02c']#, '#9467bd'] # Red, Blue, Green, Purple

g1 = plots.get_subplot_plotter(width_inch=10)
g1.triangle_plot(
    mcs_cat,
    filled=True,
    colors=cat_colors,
    legend_labels=['2pt', '3pt', '2pt+3pt'],
    contour_args=[{'alpha': 0.6} for _ in cat_colors],
    line_args=[{'lw': 2.5, 'color': c} for c in cat_colors]
)

# Truth lines for Plot 1
for i in range(len(names)):
    g1.subplots[i, i].axvline(truth_values[i], color='black', ls='--', lw=2, zorder=10)
    for j in range(i):
        g1.subplots[i, j].axvline(truth_values[j], color='black', ls='--', lw=2, zorder=10)
        g1.subplots[i, j].axhline(truth_values[i], color='black', ls='--', lw=2, zorder=10)

g1.export('plotcontours2params/category_comparison.png')

# =============================================================================
# 3. PLOT 2: TRACER-WISE COMBINATION (Dedicated total info per tracer)
# =============================================================================
print("\nGenerating Plot 2: Tracer-wise (auto+2pt+3pt) Comparison (Meta-Joints)...")

# Use dedicated total info posterior samples instead of vstack
samples_y_total = get_raw_samples('y_total')
samples_tau_total = get_raw_samples('tau_total')
samples_kappa_total = get_raw_samples('kappa_total')

mcs_tracers = [
    MCSamples(samples=samples_y_total, names=names, labels=labels, label='y (2pt+3pt)',
              settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1}),
    MCSamples(samples=samples_tau_total, names=names, labels=labels, label='tau (2pt+3pt)',
              settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1}),
    MCSamples(samples=samples_kappa_total, names=names, labels=labels, label='k (2pt+3pt)',
              settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1})
]

tracer_colors = ['#d62728', '#1f77b4', '#2ca02c'] # Red, Blue, Green

g2 = plots.get_subplot_plotter(width_inch=10)
g2.triangle_plot(
    mcs_tracers,
    filled=True,
    colors=tracer_colors,
    legend_labels=[r'$y$ (2pt+3pt)', r'$\tau$ (2pt+3pt)', r'$\kappa$ (2pt+3pt)'],
    contour_args=[{'alpha': 0.6} for _ in tracer_colors],
    line_args=[{'lw': 2.5, 'color': c} for c in tracer_colors]
)

# Truth lines for Plot 2
for i in range(len(names)):
    g2.subplots[i, i].axvline(truth_values[i], color='black', ls='--', lw=2, zorder=10)
    for j in range(i):
        g2.subplots[i, j].axvline(truth_values[j], color='black', ls='--', lw=2, zorder=10)
        g2.subplots[i, j].axhline(truth_values[i], color='black', ls='--', lw=2, zorder=10)

g2.export('plotcontours2params/tracer_comparison.png')

plt.close('all')
os._exit(0)
