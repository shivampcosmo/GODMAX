import os
import pickle as pk
import torch
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import logging

# Silence GetDist logging
logging.getLogger('getdist').setLevel(logging.ERROR)


theta_check = np.load('theta.npy')
x_check = np.load('x_JOINT.npy')

# =============================================================================
# 0. SANITY CHECK 
# =============================================================================
print("--- SBI Dataset Verification ---")
print(f"Parameters shape: {theta_check.shape} (Samples, Params)")
print(f"Statistics shape: {x_check.shape} (Samples, Stats)")
print(f"Successfully loaded {theta_check.shape[0]} simulation samples.")
print("--------------------------------")
# =============================================================================
# 1. SETUP
# =============================================================================
# LaTeX labels for the plot axes
labels = [r"\theta_{ej,0}", r"\nu_{\theta,M}", r"\nu_{\theta,z}", r"\mu_{\beta}"]
names = ["p1", "p2", "p3", "p4"]
truth_values = [2.0, -0.1, 0.0, 0.6]

# Standardized high-contrast colors for all plots
# Blue = 3-point (Higher Order), Red = 2-point (Standard)
comparison_colors = ['#1f77b4', '#d62728'] 

tracer_groups = [
    {
        'title': 'y_correlations', 
        'pairs': [('g2y', '3-point'), ('gy', '2-point')], 
        'legend_labels': [r'$g^2 y$', r'$gy$']
    },
    {
        'title': 'tau_correlations', 
        'pairs': [('g2tau', '3-point'), ('gtau', '2-point')], 
        'legend_labels': [r'$g^2 \tau$', r'$g\tau$']
    },
    {
        'title': 'kappa_correlations', 
        'pairs': [('g2kappa', '3-point'), ('gkappa', '2-point')], 
        'legend_labels': [r'$g^2 \kappa$', r'$g\kappa$']
    }
]

def get_mcsamples(name, legend_label):
    pkl_path = f'ili_posterior_{name}.pkl'
    xobs_path = f'xobs_{name}.npy'

    if not (os.path.exists(pkl_path) and os.path.exists(xobs_path)):
        print(f"Warning: Missing files for {name}. Skipping...")
        return None

    with open(pkl_path, 'rb') as f:
        posterior = pk.load(f)

    xo_val = np.load(xobs_path)
    xo = torch.from_numpy(xo_val).float().reshape(1, -1)

    # Generate samples using Direct sampling for speed and high fidelity
    raw_samples = posterior.sample((30000,), x=xo).detach().cpu().numpy().squeeze()

    return MCSamples(
        samples=raw_samples,
        names=names,
        labels=labels,
        label=legend_label,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1}
    )

# =============================================================================
# 2. PLOTTING LOOP
# =============================================================================
for group in tracer_groups:
    print(f"\nGenerating standardized Blue/Red plot for {group['title']}...")
    mcsamples_list = []
    
    for tracer_id, legend_tag in group['pairs']:
        smp = get_mcsamples(tracer_id, f"{tracer_id} ({legend_tag})")
        if smp:
            mcsamples_list.append(smp)

    if len(mcsamples_list) < 2:
        continue

    # Triangle Plot Setup
    g = plots.get_subplot_plotter(width_inch=10)
    
    g.triangle_plot(
        mcsamples_list,
        filled=True,
        colors=comparison_colors,
        legend_labels=group['legend_labels'],
        contour_args=[{'alpha': 0.6}, {'alpha': 0.6}],
        # Solid Blue line for 3-point, Dashed Red line for 2-point
        line_args=[
            {'lw': 2.5, 'ls': '-', 'color': comparison_colors[0]}, 
            {'lw': 2.5, 'ls': '--', 'color': comparison_colors[1]}
        ]
    )

    # ADD TRUTH LINES (Dotted Red)
    for i in range(len(names)):
        g.subplots[i, i].axvline(truth_values[i], color='red', ls=':', lw=1.5, zorder=10)
        for j in range(i):
            g.subplots[i, j].axvline(truth_values[j], color='red', ls=':', lw=1.2, zorder=10)
            g.subplots[i, j].axhline(truth_values[i], color='red', ls=':', lw=1.2, zorder=10)

    output_fn = f"plotcontours/contour_comparison_{group['title']}.png" #
    g.export(output_fn)
    print(f"Saved: {output_fn}")

# Clean exit
plt.close('all')
os._exit(0)
