import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. LOAD DATA
# =============================================================================
# Assuming these were saved by your 2-parameter SBI script
theta_train = np.load('theta.npy')
SCALES = [4.0, 8.0, 16.0, 32.0, 64.0]
TRACERS = ['g2y', 'g2tau', 'g2kappa']

# =============================================================================
# 2. PLOTTING
# =============================================================================
for name in TRACERS:
    x_train_tracer = np.load(f'x_{name}.npy')
    x_obs_tracer = np.load(f'xobs_{name}.npy')
    
    # Calculate Ratios for all samples (N_samples, 5_scales)
    ratios = x_train_tracer / x_obs_tracer
    
    # Calculate Distance from Reference (MSE of the ratio vs 1.0)
    # We want (ratio - 1)^2 to be as small as possible
    distances = np.mean((ratios - 1.0)**2, axis=1)
    best_idx = np.argmin(distances)
    
    # Pick 9 random indices excluding the best one
    all_indices = np.arange(len(ratios))
    other_indices = np.delete(all_indices, best_idx)
    random_indices = np.random.choice(other_indices, 10, replace=False)
    
    plt.figure(figsize=(10, 7))
    
    # Plot the 9 random samples first (thin lines)
    for idx in random_indices:
        te0, ntM, ntz, mb = theta_train[idx]
        plt.plot(SCALES, ratios[idx], color='green', alpha=0.5, lw=5, 
                 label=None) # No label for gray lines to keep legend clean

    # Plot the BEST match (thick blue line)
    te0_best, ntM_best, ntz_best, mb_best = theta_train[best_idx]
    print("for ",name," best fit params are ", f"{te0_best:.2f}, {ntM_best:.2f}, {ntz_best:.2f}, {mb_best:.2f}")
    plt.plot(
    SCALES,
    ratios[best_idx],
    color='blue',
    marker='o',
    lw=3,
    markersize=8,
    label=(
        rf'BEST MATCH: '
        rf'$\theta_{{ej,0}}={te0_best:.2f}, '
        rf'\nu_{{{{\theta_{{ej}}}}^M}}={ntM_best:.2f}, '
        rf'\nu_{{{{\theta_{{ej}}}}^z}}={ntz_best:.2f}, '
        rf'\mu_\beta={mb_best:.2f}$'
        )
    )
    # Add a dummy line for the "Other Samples" to the legend
    plt.plot([], [], color='green', alpha=0.5, lw=5, label='10 Random Samples')

    # Formatting
    plt.axhline(1.0, color='red', linestyle='--', lw=2, label='Reference (Truth)')
    plt.xlabel(r'$\theta_{\rm smooth}$ [arcmin]', fontsize=14)
    plt.ylabel(rf'$\langle gg {name[2:]} \rangle / \langle gg {name[2:]} \rangle_{{\rm ref}}$', fontsize=14)
    plt.xscale('log')
    plt.xticks(SCALES, [str(s) for s in SCALES])
    
    # Center the plot around the reference
    plt.ylim(0.2, 2.0) 
    
    plt.title(f'Sample Ratios for {name}', fontsize=16)
    plt.legend(fontsize=10, loc='best', frameon=True)
    plt.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'moments_ratio_{name}.png', dpi=300)
    plt.show()

print("Plots generated. Look for 'moments_ratio_...png' files.")
