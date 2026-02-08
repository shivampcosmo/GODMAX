import os
import pickle as pk
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import logging

# Silence the harmless GetDist cleanup error
logging.getLogger('getdist').setLevel(logging.ERROR)

# =============================================================================
# 1. SETUP
# =============================================================================
# LaTeX labels for the plot axes
labels = [r"\theta_{ej,0}", r"\mu_{\beta}"]
names = ["p1", "p2"] # GetDist internal identifiers
truth_values = [2.0, 0.6]

# The order here must match the custom_colors list below
tracers = ['JOINT']
custom_colors = ['#2ca02c']#Green

mcsamples_list = []

# =============================================================================
# 2. LOAD & SAMPLE USING TRACER-SPECIFIC XOBS
# =============================================================================
for name in tracers:
    pkl_path = f'ili_posterior_{name}.pkl'
    xobs_path = f'xobs_{name}.npy'
    
    if not (os.path.exists(pkl_path) and os.path.exists(xobs_path)):
        print(os.path.exists(pkl_path))
        print(f"Warning: Missing files for {name}. Skipping...")
        continue

    # Load posterior and the specific x_obs used for this tracer
    with open(pkl_path, 'rb') as f:
        posterior = pk.load(f)
    
    # Load the specific observation vector for this tracer
    xo_val = np.load(xobs_path)
    xo = torch.from_numpy(xo_val).float().reshape(1, -1)
    
    print(f"Sampling {name} using observation from {xobs_path}...")
    
    # Generate 30k samples for high-quality, smooth contours
    raw_samples = posterior.sample((30000,), x=xo).detach().cpu().numpy().squeeze()
    
    # Convert to GetDist MCSamples
    # smooth_scale_2D=0.7 removes splotches for publication-quality lines
    smp = MCSamples(
        samples=raw_samples, 
        names=names, 
        labels=labels, 
        label=name,
        settings={'smooth_scale_2D': 0.7, 'boundary_correction_order': 1}
    )
    mcsamples_list.append(smp)

# =============================================================================
# 3. TRIANGLE PLOT
# =============================================================================
g = plots.get_subplot_plotter(width_inch=10)

g.triangle_plot(
    mcsamples_list, 
    filled=True, 
    colors=custom_colors,
    legend_labels=['joint constraint'],
    contour_args=[{'alpha': 0.6} for _ in custom_colors],
    line_args=[{'lw': 2.5, 'color': c} for c in custom_colors]
)

# ADD TRUTH LINES (Red dashed)
for i in range(len(names)):
    # 1D histograms (Diagonal)
    g.subplots[i, i].axvline(truth_values[i], color='black', ls='--', lw=2, zorder=10)
    for j in range(i):
        # 2D contours (Off-diagonal)
        g.subplots[i, j].axvline(truth_values[j], color='black', ls='--', lw=2, zorder=10)
        g.subplots[i, j].axhline(truth_values[i], color='black', ls='--', lw=2, zorder=10)

plt.savefig('plotcontours2params/joint_2params.png', bbox_inches='tight', dpi=300)
print("Plot successfully saved: joint_2params.png")

# Force exit to prevent the GetDist __del__ NoneType error
plt.close('all')
os._exit(0)
