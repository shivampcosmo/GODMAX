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
labels = [r"\theta_{ej,0}", r"\mu_{\beta}"]
names = ["p1", "p2"]
truth_values = [2.0, 0.6]

def get_raw_samples(name):
    pkl_path = f'ili_posterior_{name}.pkl'
    xobs_path = f'xobs_{name}.npy'
    
    if not (os.path.exists(pkl_path) and os.path.exists(xobs_path)):
        print(f"Warning: Missing files for {name}. Skipping...")
        return None
        
    with open(pkl_path, 'rb') as f:
        posterior = pk.load(f)
    xo = torch.from_numpy(np.load(xobs_path)).float().reshape(1, -1)
    
    print(f"Sampling {name} posterior directly (no MCMC)...")
    
    # Standard fast sampling! 30k samples for super smooth contours.
    samples = posterior.sample((30000,), x=xo, show_progress_bars=True)
    return samples.detach().cpu().numpy().squeeze()

out_dir = "plotcontours_cls"
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# 2. PLOT: gy and gkappa
# =============================================================================
print("\nGenerating Plot: gy and gkappa...")

# Define only the well-matched probes
probes = ['gy', 'gkappa']
colors = ['#ff7f0e', '#2ca02c'] # Orange, Green
legend_labels = [r'$C_\ell^{gy}$', r'$C_\ell^{g\kappa}$']

mcs_list = []
valid_colors = []
valid_labels = []

# Extract samples and build MCSamples objects dynamically
for probe, color, leg_label in zip(probes, colors, legend_labels):
    samples = get_raw_samples(probe)
    
    if samples is not None:
        mcs_list.append(
            MCSamples(samples=samples, names=names, labels=labels, label=leg_label,
                      settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1})
        )
        valid_colors.append(color)
        valid_labels.append(leg_label)

# Generate the GetDist plot
if mcs_list:
    print("\nGenerating final plot image...")
    g = plots.get_subplot_plotter(width_inch=10)
    
    g.triangle_plot(
        mcs_list,
        filled=True,
        colors=valid_colors,
        legend_labels=valid_labels,
        contour_args=[{'alpha': 0.6} for _ in valid_colors],
        line_args=[{'lw': 2.5, 'color': c} for c in valid_colors]
    )

    # Add Truth lines
    for i in range(len(names)):
        g.subplots[i, i].axvline(truth_values[i], color='black', ls='--', lw=2, zorder=10)
        for j in range(i):
            g.subplots[i, j].axvline(truth_values[j], color='black', ls='--', lw=2, zorder=10)
            g.subplots[i, j].axhline(truth_values[i], color='black', ls='--', lw=2, zorder=10)

    save_path = os.path.join(out_dir, 'cls_gy_gkappa_only.png')
    g.export(save_path)
    print(f"\nPlot successfully saved to: {save_path}")
else:
    print("\nNo valid samples found to plot. Check your file paths.")

plt.close('all')
os._exit(0)
