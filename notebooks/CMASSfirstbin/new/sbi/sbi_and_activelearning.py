import os
import sys
import torch
import numpy as np
import healpy as hp
import pickle as pk
import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
#import warnings

# --- Path Setup for ltu-ili ---
sys.path.append('/work/hdd/bdne/aacharya2/ltu-ili')
from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.utils import load_nde_sbi, Uniform

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
# =============================================================================
# 0. VALIDATION CODE (Without using PosteriorCoverage as it can crash)
# =============================================================================
import threading
def _sample_member_thread(member, x_t, n_samples, result, exception):
    """Thread target: writes output into shared lists."""
    try:
        s = member.sample((n_samples,), x=x_t, show_progress_bars=False)
        result[0] = s.detach().cpu().numpy()
    except Exception as e:
        exception[0] = e


def sample_ensemble_direct(posterior, x_obs_norm, n_samples=500,
                            timeout_per_member=45):
    """
    Sample directly from each ensemble member and pool results.
    This bypasses EnsemblePosterior's importance-weighted resampling
    which is what triggers the rejection sampling hang.

    Uses a per-member thread timeout to handle the case where sbi's
    rejection sampling hangs indefinitely. The hang is announced via
    logging.warning() inside sbi, not warnings.warn(), so
    warnings.filterwarnings() cannot intercept it. The timeout is the
    only reliable mechanism.

    Args:
        posterior:           trained ltu-ili/sbi posterior object
        x_obs_norm:          normalised observation vector, shape (n_stats,)
        n_samples:           total posterior samples to return
        timeout_per_member:  seconds before abandoning a hung member

    Returns:
        np.ndarray of shape (n_samples, n_params), or None if all fail
    """
    x_t = torch.from_numpy(x_obs_norm).float().reshape(1, -1)

    try:
        members = posterior.posteriors
    except AttributeError:
        # Not an ensemble. Single posterior, sample with timeout
        result    = [None]
        exception = [None]
        t = threading.Thread(
            target=_sample_member_thread,
            args=(posterior, x_t, n_samples, result, exception),
            daemon=True
        )
        t.start()
        t.join(timeout=timeout_per_member)
        if t.is_alive():
            print(f"    [WARN] Single posterior timed out after {timeout_per_member}s.")
            return None
        if exception[0] is not None:
            print(f"    [WARN] Single posterior failed: {exception[0]}")
            return None
        return result[0]

    n_members  = len(members)
    per_member = max(1, n_samples // n_members)
    collected  = []

    for i, member in enumerate(members):
        result    = [None]
        exception = [None]
        t = threading.Thread(
            target=_sample_member_thread,
            args=(member, x_t, per_member, result, exception),
            daemon=True
        )
        t.start()
        t.join(timeout=timeout_per_member)

        if t.is_alive():
            print(f"    [SKIP] Member {i} timed out after {timeout_per_member}s "
                  f"    rejection sampling hung.")
        elif exception[0] is not None:
            print(f"    [SKIP] Member {i} failed: {exception[0]}")
        else:
            collected.append(result[0])

    if not collected:
        print(f"    [WARN] All {n_members} ensemble members failed or timed out.")
        return None

    n_ok = len(collected)
    if n_ok < n_members:
        print(f"    [INFO] {n_ok}/{n_members} members contributed samples.")

    combined = np.concatenate(collected, axis=0)
    np.random.shuffle(combined)

    # Guarantee exactly n_samples regardless of how many members succeeded.
    # If fewer members contributed than expected, upsample with replacement
    # rather than returning a short array that breaks np.array() stacking
    # in run_validation_direct.
    if len(combined) >= n_samples:
        return combined[:n_samples]
    else:
        print(f"    [INFO] Only {len(combined)} raw samples collected — "
              f"upsampling to {n_samples} with replacement.")
        idx = np.random.choice(len(combined), size=n_samples, replace=True)
        return combined[idx]

def run_validation_direct(name, posterior, xt_val, theta_val, labels, val_dir,
                           prior_low, prior_high):
    """
    Full validation using direct per-member sampling.
    Computes rank histogram, coverage, and TARP manually so
    we never call PosteriorCoverage._sample_dataset which hangs.

    Args:
        name:        statistic name string
        posterior:   trained posterior
        xt_val:      normalised validation summaries, shape (n_val, n_stats)
        theta_val:   true parameters, shape (n_val, n_params)
        labels:      list of parameter label strings
        val_dir:     Path to output directory
        prior_low:   list of prior lower bounds
        prior_high:  list of prior upper bounds
    """
    val_dir = Path(val_dir)
    val_dir.mkdir(exist_ok=True)

    n_val, n_params = theta_val.shape
    if n_val < 5:
        print(f"  [SKIP] Only {n_val} val points for {name}.")
        return

    print(f"  Validating {name} on {n_val} points using direct member sampling...")

    n_post      = 1000 if n_val >= 30 else 500
    all_samples = []
    valid_idx   = []

    for i in range(n_val):
        s = sample_ensemble_direct(posterior, xt_val[i], n_samples=n_post)
        if s is None:
            print(f"    [WARN] All ensemble members failed for val point {i}, skipping.")
            continue
        s = np.clip(s, a_min=prior_low, a_max=prior_high)
        all_samples.append(s)
        valid_idx.append(i)

    if len(valid_idx) < 3:
        print(f"  [SKIP] Fewer than 3 valid validation points for {name}, skipping plots.")
        return

    all_samples  = np.array(all_samples)       # (n_valid, n_post, n_params)
    theta_subset = theta_val[valid_idx]         # (n_valid, n_params)
    n_valid      = len(valid_idx)

    print(f"  Collected samples for {n_valid}/{n_val} validation points.")

    # --- Rank histogram ---
    ranks = (all_samples < theta_subset[:, None, :]).sum(axis=1)  # (n_valid, n_params)

    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
    if n_params == 1:
        axes = [axes]
    navg = n_valid / 10
    for p, (ax, label) in enumerate(zip(axes, labels)):
        ax.hist(ranks[:, p], bins=10, range=(0, n_post))
        ax.axhline(navg,             color='k', ls='-',  label='Expected')
        ax.axhline(navg - navg**0.5, color='k', ls='--', alpha=0.5)
        ax.axhline(navg + navg**0.5, color='k', ls='--', alpha=0.5)
        ax.set_title(label)
        ax.set_xlabel('Rank')
        ax.grid(True)
    axes[0].set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(val_dir / 'ranks_histogram.jpg', dpi=150)
    plt.close()

    # --- Coverage plot ---
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
    if n_params == 1:
        axes = [axes]
    cdf = np.linspace(0, 1, n_valid)
    for p, (ax, label) in enumerate(zip(axes, labels)):
        xr = np.sort(ranks[:, p] / n_post)
        ax.plot(cdf, cdf, 'k--', label='Ideal')
        ax.plot(xr,  cdf, lw=2,  label='Posterior')
        ax.set(aspect='equal', adjustable='box')
        ax.set_title(label)
        ax.set_xlabel('Predicted Percentile')
        ax.grid(True)
    axes[0].set_ylabel('Empirical Percentile')
    plt.tight_layout()
    plt.savefig(val_dir / 'plot_coverage.jpg', dpi=150)
    plt.close()

    # --- TARP ---
    # tarp defaults to n_sims // 10 bins. Need at least n_valid >= 50
    # to get 5+ meaningful bins.
    TARP_MIN_POINTS = 50

    if n_valid >= TARP_MIN_POINTS:
        try:
            import tarp

            # Explicitly set num_alpha_bins: never rely on n_sims // 10 default
            n_alpha_bins = int(np.clip(n_valid / 3, 10, 50))

            # tarp expects (n_samples, n_sims, n_params)
            # all_samples is (n_valid, n_post, n_params) --> (n_post, n_valid, n_params)
            samples_for_tarp = all_samples.transpose(1, 0, 2)

            print(f"  TARP shapes: samples={samples_for_tarp.shape}, "
                  f"theta={theta_subset.shape}, n_alpha_bins={n_alpha_bins}")

            ecp, alpha = tarp.get_tarp_coverage(
                samples_for_tarp, theta_subset,
                references='random', metric='euclidean',
                norm=True, bootstrap=True,
                num_bootstrap=100, num_alpha_bins=n_alpha_bins,
            )

            print(f"  TARP ecp shape: {ecp.shape}, alpha shape: {alpha.shape}")
            print(f"  ecp mean range: [{np.mean(ecp, axis=0).min():.3f}, "
                  f"{np.mean(ecp, axis=0).max():.3f}]")

            ecp_mean = np.mean(ecp, axis=0)
            ecp_std  = np.std(ecp,  axis=0)

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot([0, 1], [0, 1], 'k--', label='Ideal')
            ax.plot(alpha, ecp_mean, color='b', lw=2, label='TARP')
            ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std,
                            alpha=0.3, color='b', label=r'$\pm 1\sigma$')
            ax.fill_between(alpha, ecp_mean - 2*ecp_std, ecp_mean + 2*ecp_std,
                            alpha=0.15, color='b', label=r'$\pm 2\sigma$')
            ax.set_xlabel('Credibility Level')
            ax.set_ylabel('Expected Coverage')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.legend()
            ax.set_title(f'{name}  (n_val={n_valid})')
            plt.tight_layout()
            plt.savefig(val_dir / 'plot_TARP.jpg', dpi=150)
            plt.close()
            print(f"  TARP plot saved → {val_dir / 'plot_TARP.jpg'}")

        except Exception as e:
            print(f"  [WARN] TARP failed for {name}: {e}")

    else:
        print(f"  [SKIP] TARP needs >={TARP_MIN_POINTS} valid points, have {n_valid}. "
              f"Need {TARP_MIN_POINTS - n_valid} more.")

    print(f"  Validation plots in {val_dir}")


# =============================================================================
# 1. DATA AGGREGATION (2-Parameter Version)
# =============================================================================
BASE_DIR = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/new'
CSV_FILES = [
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/lhs_samples.csv',
#    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/round2_samples.csv',
#    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/round3_samples.csv',
]
NEXT_ROUND = int(len(CSV_FILES) + 1)
print(f"The next round of samples to be run will be round {NEXT_ROUND}")

NSIDE  = 512
SCALES = [4.0, 8.0, 16.0, 32.0, 64.0]

CACHE_DIR = 'sample_vector_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_moments(path):
    """
    Extract summary statistics from a single combined map pickle file.

    Vector layout: 6 statistics per scale. 5 scales = 30 elements total.
    Within each scale block of 6:
      idx 0: <ggy>   (3pt cross, y tracer)
      idx 1: <ggtau>   (3pt cross, tau tracer)
      idx 2: <ggkappa>   (3pt cross, kappa tracer)
      idx 3: <gy>    (2pt cross, y tracer)
      idx 4: <gtau>    (2pt cross, tau tracer)
      idx 5: <gkappa>    (2pt cross, kappa tracer)

    Scale blocks:
      Scale 0 (4  arcmin): indices  0-5
      Scale 1 (8  arcmin): indices  6-11
      Scale 2 (16 arcmin): indices 12-17
      Scale 3 (32 arcmin): indices 18-23
      Scale 4 (64 arcmin): indices 24-29
    """
    pattern = os.path.join(path, "**", f"allmaps_sim_B12_nside{NSIDE}.pkl")
    files   = glob.glob(pattern, recursive=True)

    if len(files) == 0:
        return None

    ymap, gmap, kmap, tmap = [
        np.zeros(12 * NSIDE**2, dtype=np.float32) for _ in range(4)
    ]
    LMAX = 3 * NSIDE - 1

    for f in files:
        with open(f, 'rb') as h:
            data = pk.load(h)
            ymap += np.nan_to_num(data.get('map_ymap', 0))
            kmap += np.nan_to_num(data.get('map_kappa', 0))
            tmap += np.nan_to_num(data.get('map_tau', 0))
            mock_gals_dict = data.get('mock_gals_all', {})
            for chunk_idx in mock_gals_dict:
                gal_data = mock_gals_dict[chunk_idx]
                if gal_data.size > 0:
                    ra_gal  = gal_data[:, 0] % 360.0
                    dec_gal = np.clip(gal_data[:, 1], -90.0, 90.0)
                    n_bad   = np.sum(~np.isfinite(ra_gal) | ~np.isfinite(dec_gal))
                    if n_bad > 0:
                        mask    = np.isfinite(ra_gal) & np.isfinite(dec_gal)
                        ra_gal  = ra_gal[mask]
                        dec_gal = dec_gal[mask]
                    if len(ra_gal) > 0:
                        pix   = hp.ang2pix(NSIDE, ra_gal, dec_gal, lonlat=True)
                        gmap += np.bincount(pix, minlength=12 * NSIDE**2)

    # Build footprint mask from galaxy map --> pixels where the lightcone
    # has coverage. Use the unsmoothed gmap before delta_g conversion.
    footprint = gmap > 0   # True for pixels inside the lightcone

    dg = (gmap / np.mean(gmap[footprint]) - 1.0) if np.mean(gmap[footprint]) > 0 \
         else np.zeros_like(gmap)

    vec = []
    for th in SCALES:
        fwhm = np.radians(th / 60.)
        gs   = hp.smoothing(dg,   fwhm=fwhm, lmax=LMAX, verbose=False)
        ys   = hp.smoothing(ymap, fwhm=fwhm, lmax=LMAX, verbose=False)
        ts   = hp.smoothing(tmap, fwhm=fwhm, lmax=LMAX, verbose=False)
        ks   = hp.smoothing(kmap, fwhm=fwhm, lmax=LMAX, verbose=False)

        # Restrict mean to footprint pixels only
        vec.extend([np.mean((gs**2 * ys)[footprint]),
                    np.mean((gs**2 * ts)[footprint]),
                    np.mean((gs**2 * ks)[footprint])])
        vec.extend([np.mean((gs * ys)[footprint]),
                    np.mean((gs * ts)[footprint]),
                    np.mean((gs * ks)[footprint])])

    return np.array(vec, dtype=np.float32)   # length 30

print("Extracting reference run...")
x_obs = extract_moments(os.path.join(BASE_DIR, 'reference_run')).astype(np.float32)
np.save('x_obs.npy', x_obs)
print(f"  x_obs shape: {x_obs.shape}")   # should be (30,)

theta_train, x_train = [], []

for csv in CSV_FILES:
    df = pd.read_csv(csv)

    # Folder offset: each round's sample_id starts from 0 in its own CSV,
    # but the on-disk folders are numbered globally.
    if   'lhs'    in csv: offset = 0
    elif 'round2' in csv: offset = 500

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Loading {os.path.basename(csv)}"):
        sid        = int(row['sample_id']) + offset
        cache_file = os.path.join(CACHE_DIR, f"x_sample_{sid}.npy")

        if os.path.exists(cache_file):
            v = np.load(cache_file)
        else:
            s_path = os.path.join(BASE_DIR, f"sample_{sid}")
            v      = extract_moments(s_path)
            if v is not None:
                np.save(cache_file, v)

        if v is not None:
            theta_train.append([row['theta_ej_0'], row['nu_theta_ej_M']])
            x_train.append(v)

x_train     = np.array(x_train).astype(np.float32)
theta_train = np.array(theta_train).astype(np.float32)

np.save('x_train_full.npy', x_train)
np.save('theta.npy', theta_train)

print(f"Loaded {len(theta_train)} simulations. "
      f"x_train shape: {x_train.shape}, theta_train shape: {theta_train.shape}")


# =============================================================================
# 2. LTU-ILI TRAINING LOOP
# =============================================================================
stat_map = {
    # --- Individual 3pt cross-moments ---
    'g2y':         [0,  6,  12, 18, 24],
    'g2tau':       [1,  7,  13, 19, 25],
    'g2kappa':     [2,  8,  14, 20, 26],

    # --- Individual 2pt cross-moments ---
    'gy':          [3,  9,  15, 21, 27],
    'gtau':        [4,  10, 16, 22, 28],
    'gkappa':      [5,  11, 17, 23, 29],

    # --- Full joint ---
    'JOINT':       list(range(30)),

    # --- Per-tracer totals (3pt + 2pt) ---
    'y_total':     [0,  3,  6,  9,  12, 15, 18, 21, 24, 27],
    'tau_total':   [1,  4,  7,  10, 13, 16, 19, 22, 25, 28],
    'kappa_total': [2,  5,  8,  11, 14, 17, 20, 23, 26, 29],

    # --- Category totals ---
    'all_3pt':     [0,  1,  2,  6,  7,  8,  12, 13, 14, 18, 19, 20, 24, 25, 26],
    'all_2pt':     [3,  4,  5,  9,  10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29],
}

np.random.seed(42)
all_indices = np.arange(len(theta_train))
np.random.shuffle(all_indices)

n_val           = max(5, int(len(theta_train) * 0.20))
val_idx         = all_indices[:n_val]
train_idx       = all_indices[n_val:]
theta_val       = theta_train[val_idx]
theta_train_set = theta_train[train_idx]

print(f"Train/val split: {len(train_idx)} train, {n_val} val")

VALIDATE_DURING_LOOP = {'JOINT'}
labels = [r'$\theta_{ej,0}$', r'${\nu_{\theta_{ej}}}^{M}$']

for name, idx in stat_map.items():
    n_stats = len(idx)
    n_train = len(theta_train_set)
    ratio   = n_train / n_stats

    print(f"\n--- Training NPE: {name}  ({n_stats} statistics, ratio={ratio:.1f}) ---")

    xt_full  = x_train[:, idx]
    xt_train = xt_full[train_idx]
    xt_val_s = xt_full[val_idx]
    xo       = x_obs[idx]

    x_mean   = np.mean(xt_train, axis=0)
    x_std    = np.std(xt_train,  axis=0)
    xt_train = (xt_train - x_mean) / (x_std + 1e-8)
    xt_val_s = (xt_val_s - x_mean) / (x_std + 1e-8)
    xo       = (xo       - x_mean) / (x_std + 1e-8)

    np.save(f'scaler_{name}_mean.npy', x_mean)
    np.save(f'scaler_{name}_std.npy',  x_std)
    np.save(f'x_{name}.npy',           xt_train)
    np.save(f'xobs_{name}.npy',        xo)
    np.save(f'theta_train_{name}.npy', theta_train_set)

    loader = StaticNumpyLoader(
        in_dir='./',
        x_file=f'x_{name}.npy',
        theta_file=f'theta_train_{name}.npy',
        xobs_file=f'xobs_{name}.npy'
    )

    # --- Adaptive ensemble size ---
    # Ensemble uncertainty scales as 1/sqrt(n_members). At high ratio the
    # flow is easy to learn and fewer members suffice. JOINT has the lowest
    # ratio and benefits most from a full ensemble.
    if n_stats <= 5:
        repeats = 3
    elif n_stats <= 10:
        repeats = 4
    elif n_stats <= 15:
        repeats = 6
    else:
        repeats = 8

    # --- Adaptive network size ---
    # At high ratio, small networks converge faster and overfit less.
    # Scale width and depth with the number of statistics.
    if n_stats <= 5:
        hfs = 16
        nts = 2
    elif n_stats <= 15:
        hfs = 32
        nts = 3
    else:
        hfs = 64
        nts = 4

    # --- NSF threshold ---
    if n_stats <= 5:
        nsf_threshold = 8
    elif n_stats <= 15:
        nsf_threshold = 12
    elif n_stats <= 30:
        nsf_threshold = 20
    else:
        nsf_threshold = 999

    # --- Adaptive batch size ---
    # sbi default of 50 is too small for n_train ~ 400.
    # Target ~12% of the effective training data (after sbi's internal
    # validation_fraction=0.1 split), capped at 256.
    n_train_effective = int(n_train * 0.9)
    batch_size = int(np.clip(n_train_effective // 8, 32, 256))

    # --- Training arguments ---
    # stop_after_epochs=40 gives more patience than the sbi default of 20,
    # which is important for JOINT where the loss surface is harder.
    # z_score_x='none' because xt_train is already manually normalized above;
    # passing it through sbi's internal z-scorer again is redundant and wastes
    # one affine layer. z_score_theta='independent' is kept so the prior is
    # properly handled internally.
    train_args = {
        'training_batch_size': batch_size,
        'learning_rate':       5e-4,
        'max_num_epochs':      500,
        'stop_after_epochs':   40,
        'clip_max_norm':       5.0,
        'validation_fraction': 0.1,
    }

    print(f"  [TRAIN] device={device}, batch_size={batch_size}, "
          f"stop_after={train_args['stop_after_epochs']}, "
          f"n_train_effective={n_train_effective}")

    if ratio < nsf_threshold:
        print(f"  [ARCH] MAF-only  "
              f"(ratio={ratio:.1f} < {nsf_threshold}, "
              f"repeats={repeats}, hfs={hfs}, nts={nts})")
        nets = load_nde_sbi(
            engine='NPE', model='maf',
            repeats=repeats, hidden_features=hfs, num_transforms=nts,
            z_score_x='none',
            z_score_theta='independent',
        )
    else:
        n_nsf = max(1, repeats // 3)
        n_maf = repeats - n_nsf
        print(f"  [ARCH] NSF+MAF   "
              f"(ratio={ratio:.1f} >= {nsf_threshold}, "
              f"repeats={repeats} [{n_nsf} NSF + {n_maf} MAF], "
              f"hfs={hfs}, nts={nts})")
        nets = (
            load_nde_sbi(
                engine='NPE', model='nsf', repeats=n_nsf,
                hidden_features=hfs, num_transforms=nts,
                z_score_x='none', z_score_theta='independent',
            )
            + load_nde_sbi(
                engine='NPE', model='maf', repeats=n_maf,
                hidden_features=hfs, num_transforms=nts,
                z_score_x='none', z_score_theta='independent',
            )
        )

    runner = InferenceRunner.load(
        backend='sbi', engine='NPE',
        prior=Uniform(low=[1.0, -0.3], high=[6.0, 0.0]),
        nets=nets,
        out_dir=Path(f'./sbi_logs_{name}'),
        device=device,
        train_args=train_args,
    )
    posterior, _ = runner(loader)

    with open(f'ili_posterior_{name}.pkl', 'wb') as f:
        pk.dump(posterior, f)

    if name not in VALIDATE_DURING_LOOP:
        print(f"  [SKIP] Validation deferred for {name}. "
              f"Run post-hoc after all rounds.")
        continue
    '''
    # --- JOINT validation only during active learning ---
    print(f"  Validating {name} on {n_val} hold-out points...")
    run_validation_direct(
        name       = name,
        posterior  = posterior,
        xt_val     = xt_val_s,
        theta_val  = theta_val,
        labels     = labels,
        val_dir    = Path(f'./validation_{name}'),
        prior_low  = [1.0,  0.01],
        prior_high = [6.0,  1.5],)
    #'''


# =============================================================================
# 3. NEXT ROUND PROPOSAL (Next 200 Samples)
# =============================================================================
with open('ili_posterior_gtau.pkl', 'rb') as f:
    #joint_posterior = pk.load(f)
    gtau_posterior = pk.load(f)

# Reuse the normalised xobs saved during training
#xo_joint_norm = np.load('xobs_JOINT.npy')
xo_gtau_norm = np.load('xobs_gtau.npy')

# Use per-member direct sampling to avoid EnsemblePosterior rejection hang
print("\nGenerating next round proposals via direct ensemble sampling...")
#next_theta = sample_ensemble_direct(joint_posterior, xo_joint_norm, n_samples=200)
next_theta = sample_ensemble_direct(gtau_posterior, xo_gtau_norm, n_samples=200)

if next_theta is None:
    raise RuntimeError(
        "Failed to generate proposal samples. All ensemble members degenerate. "
        "Check the JOINT posterior training logs.")

# Clip to prior bounds, as flow tails can occasionally escape
next_theta = np.clip(next_theta, a_min=[1.0, -0.3], a_max=[6.0, 0.0])

out_csv = f'/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/round{NEXT_ROUND}_samples.csv'
pd.DataFrame(next_theta, columns=['theta_ej_0', 'nu_theta_ej_M']).to_csv(out_csv, index_label='sample_id')
print(f"Generated {out_csv} with {len(next_theta)} proposals.")
print(f"  theta_ej_0: mean={next_theta[:,0].mean():.3f}, "
      f"std={next_theta[:,0].std():.3f}, "
      f"range=[{next_theta[:,0].min():.3f}, {next_theta[:,0].max():.3f}]")
print(f"  nu_theta_ej_M:    mean={next_theta[:,1].mean():.3f}, "
      f"std={next_theta[:,1].std():.3f}, "
      f"range=[{next_theta[:,1].min():.3f}, {next_theta[:,1].max():.3f}]")


# =============================================================================
# 4. PER-STATISTIC POST-HOC VALIDATION
# =============================================================================
FULL_VALIDATE_THRESHOLD = 100
'''
if len(theta_train) >= FULL_VALIDATE_THRESHOLD:
    print("\nRunning post-hoc validation for all statistics...")

    for name in stat_map:
        if name == 'JOINT':
            continue   # already validated during the active learning loop

        posterior_path = f'ili_posterior_{name}.pkl'
        if not os.path.exists(posterior_path):
            print(f"  [SKIP] No saved posterior for {name}.")
            continue

        with open(posterior_path, 'rb') as f:
            post = pk.load(f)

        idx = stat_map[name]

        scaler_mean_path = f'scaler_{name}_mean.npy'
        scaler_std_path  = f'scaler_{name}_std.npy'

        if os.path.exists(scaler_mean_path) and os.path.exists(scaler_std_path):
            x_mean = np.load(scaler_mean_path)
            x_std  = np.load(scaler_std_path)
        else:
            xt_tr  = x_train[train_idx][:, idx]
            x_mean = xt_tr.mean(axis=0)
            x_std  = xt_tr.std(axis=0) + 1e-8
            print(f"  [INFO] Recomputed scaler for {name} from training split.")

        xt_v = ((x_train[val_idx][:, idx] - x_mean) / x_std).astype(np.float32)
        th_v = theta_train[val_idx]

        run_validation_direct(
            name       = name,
            posterior  = post,
            xt_val     = xt_v,
            theta_val  = th_v,
            labels     = labels,
            val_dir    = Path(f'./validation_{name}'),
            prior_low  = [1.0,  0.01],
            prior_high = [6.0,  1.5],
        )

else:
    remaining = FULL_VALIDATE_THRESHOLD - len(theta_train)
    print(f"\n[SKIP] Post-hoc validation deferred. Need {remaining} more samples "
          f"({len(theta_train)}/{FULL_VALIDATE_THRESHOLD} total).")
#'''
print("\nAll done. Generate plots and run the next round of samples!")
