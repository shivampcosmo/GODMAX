import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# =============================================================================
# GLOBAL STYLE
# =============================================================================
FONTSIZE       = 18
LABELSIZE      = 16
TICK_MAJOR_S   = 8
TICK_MAJOR_W   = 1.5
TICK_MINOR_S   = 5
TICK_MINOR_W   = 1.0
TICK_DIRECTION = 'in'

mpl.rcParams.update({
    'font.size':             FONTSIZE,
    'axes.labelsize':        FONTSIZE,
    'axes.titlesize':        FONTSIZE,
    'xtick.labelsize':       LABELSIZE,
    'ytick.labelsize':       LABELSIZE,
    'legend.fontsize':       13,
    'xtick.major.size':      TICK_MAJOR_S,
    'xtick.major.width':     TICK_MAJOR_W,
    'xtick.minor.size':      TICK_MINOR_S,
    'xtick.minor.width':     TICK_MINOR_W,
    'ytick.major.size':      TICK_MAJOR_S,
    'ytick.major.width':     TICK_MAJOR_W,
    'ytick.minor.size':      TICK_MINOR_S,
    'ytick.minor.width':     TICK_MINOR_W,
    'xtick.direction':       TICK_DIRECTION,
    'ytick.direction':       TICK_DIRECTION,
    'xtick.top':             True,
    'ytick.right':           True,
    'font.family':           'serif',
})

# =============================================================================
# 1. LOAD DATA
# =============================================================================
theta_train = np.load('theta.npy')   # shape (N, 2)
x_obs       = np.load('x_obs.npy')  # shape (30,) -- the full unnormalised reference vector

SCALES  = [4.0, 8.0, 16.0, 32.0, 64.0]
TRACERS = ['g2y', 'g2tau', 'g2kappa', 'gy', 'gtau', 'gkappa'] 

# Nicer LaTeX labels for axis titles
TRACER_LABELS = {
    'g2y':     r'$\langle g^2 y \rangle$',
    'g2tau':   r'$\langle g^2 \tau \rangle$',
    'g2kappa': r'$\langle g^2 \kappa \rangle$',
    'gy':      r'$\langle g y \rangle$',
    'gtau':    r'$\langle g \tau \rangle$',
    'gkappa':  r'$\langle g \kappa \rangle$',
}

# Indices of each statistic in the full 30-element vector (mirrors stat_map)
STAT_IDX = {
    'g2y':     [0,  6,  12, 18, 24],
    'g2tau':   [1,  7,  13, 19, 25],
    'g2kappa': [2,  8,  14, 20, 26],
    'gy':      [3,  9,  15, 21, 27],
    'gtau':    [4,  10, 16, 22, 28],
    'gkappa':  [5,  11, 17, 23, 29],
}

N_RANDOM   = 500
REF_PARAMS = (2.0, -0.1)   # truth values used in the reference run

os.makedirs('plotmoments', exist_ok=True)
np.random.seed(42)

# =============================================================================
# 2. PLOTTING LOOP
# =============================================================================
for name in TRACERS:
    idx = STAT_IDX[name]

    # Load the *un-normalised* training vectors (saved before normalisation)
    # x_{name}.npy was saved AFTER normalisation in the training script, so
    # we reconstruct from x_train_full and the saved scaler instead.
    x_mean = np.load(f'scaler_{name}_mean.npy')   # (n_stats,)
    x_std  = np.load(f'scaler_{name}_std.npy')    # (n_stats,)

    x_train_norm = np.load(f'x_{name}.npy')       # normalised, shape (N_train, n_stats)
    x_train_raw  = x_train_norm * x_std + x_mean  # back to physical units, (N_train, n_stats)

    x_obs_slice  = x_obs[idx]                     # reference vector, (5,)

    # Find the best-matching training sample (smallest MSE to reference)
    distances = np.mean((x_train_raw - x_obs_slice[None, :]) ** 2, axis=1)
    best_idx  = int(np.argmin(distances))

    # Draw N_RANDOM other samples
    other_idx    = np.delete(np.arange(len(x_train_raw)), best_idx)
    random_idx   = np.random.choice(other_idx, N_RANDOM, replace=False)

    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    # --- 50 random samples ---
    for i, ridx in enumerate(random_idx):
        ax.plot(
            SCALES, x_train_raw[ridx],
            color='steelblue', alpha=0.35, lw=1.8,
            label='Random samples' if i == 0 else None
        )

    # --- Reference observation ---
    ax.plot(
        SCALES, x_obs_slice,
        color='red', ls='--', lw=2.5, marker='s', markersize=7,
        label=(rf'Reference  '
               rf'($\theta_{{ej,0}}={REF_PARAMS[0]}$, '
               rf'$\nu_{{\theta_{{ej}}}}^M={REF_PARAMS[1]}$)')
    )

    print('For ', name, " the min and max values are",np.min(x_obs_slice),np.max(x_obs_slice))
    '''
    # --- Best-matching training sample ---
    te0_best, ntM_best = theta_train[best_idx]
    print(f"{name}: best-fit params  theta_ej_0={te0_best:.3f},  "
          f"nu_theta_ej_M={ntM_best:.3f}")

    ax.plot(SCALES, x_train_raw[best_idx],
        color='darkorange', lw=2.5, marker='o', markersize=8, zorder=5,
        label=(rf'Best match  '
               rf'($\theta_{{ej,0}}={te0_best:.2f}$, '
               rf'$\nu_{{\theta_{{ej}}}}^M={ntM_best:.2f}$)'))
    '''

    # --- Formatting ---
    ax.set_xscale('log')
    ax.set_xticks(SCALES)
    ax.set_xticklabels([str(int(s)) for s in SCALES])

    ax.set_xlabel(r'$\theta_{\rm smooth}\ [\rm arcmin]$')
    ax.set_ylabel(TRACER_LABELS[name])
    ax.set_title(f'Data vectors : {TRACER_LABELS[name]}')

    ax.legend(loc='best', frameon=True, framealpha=0.85)
    ax.grid(True, alpha=0.2, which='both')

    plt.tight_layout()
    out_path = f'plotmoments/moments_vector_{name}.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

print("\nAll done.")
