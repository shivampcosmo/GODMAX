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
import warnings
# --- Path Setup for ltu-ili ---
sys.path.append('/work/hdd/bdne/aacharya2/ltu-ili')
from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.utils import load_nde_sbi, Uniform
#from ili.validation.metrics import PosteriorCoverage

import warnings

# =============================================================================
# 0. VALIDATION CODE (Without using PosteriorCoverage as it can crash)
# =============================================================================

def sample_ensemble_direct(posterior, x_obs_norm, n_samples=500):
    """
    Sample directly from each ensemble member and pool results.
    This bypasses EnsemblePosterior's importance-weighted resampling
    which is what triggers the rejection sampling hang.

    Args:
        posterior:      trained ltu-ili/sbi posterior object
        x_obs_norm:     normalised observation vector, shape (n_stats,)
        n_samples:      total posterior samples to return

    Returns:
        np.ndarray of shape (n_samples, n_params), or None if all members fail
    """
    x_t = torch.from_numpy(x_obs_norm).float().reshape(1, -1)

    # Access individual member posteriors inside the ensemble
    try:
        members = posterior.posteriors   # EnsemblePosterior attribute
    except AttributeError:
        # Not an ensemble — single posterior, just sample directly
        try:
            s = posterior.sample((n_samples,), x=x_t, show_progress_bars=False)
            return s.detach().cpu().numpy()
        except Exception as e:
            print(f"    [WARN] Single posterior sampling failed: {e}")
            return None

    n_members       = len(members)
    per_member      = max(1, n_samples // n_members)
    collected       = []

    for i, member in enumerate(members):
        # Suppress the sbi rejection warning per member —
        # if a member is degenerate we just skip it rather than hanging
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'error',
                message='.*proposal samples are.*accepted.*',
                category=UserWarning,
            )
            try:
                s = member.sample(
                    (per_member,),
                    x=x_t,
                    show_progress_bars=False
                )
                collected.append(s.detach().cpu().numpy())
            except UserWarning:
                print(f"    [SKIP] Ensemble member {i} degenerate — dropping from sample.")
            except Exception as e:
                print(f"    [SKIP] Ensemble member {i} failed: {e}")

    if not collected:
        return None

    combined = np.concatenate(collected, axis=0)
    # Shuffle so samples are not blocked by member
    np.random.shuffle(combined)
    return combined[:n_samples]


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

    # Determine how many posterior samples per validation point
    n_post = 1000 if n_val >= 30 else 500

    all_samples = []   # will be (n_val, n_post, n_params)
    valid_idx   = []

    for i in range(n_val):
        s = sample_ensemble_direct(posterior, xt_val[i], n_samples=n_post)
        if s is None:
            print(f"    [WARN] All ensemble members failed for val point {i}, skipping.")
            continue

        # Clip to prior support — flow tails can escape bounds
        s = np.clip(s, a_min=prior_low, a_max=prior_high)
        all_samples.append(s)
        valid_idx.append(i)

    if len(valid_idx) < 3:
        print(f"  [SKIP] Fewer than 3 valid validation points for {name}, skipping plots.")
        return

    all_samples  = np.array(all_samples)              # (n_valid, n_post, n_params)
    theta_subset = theta_val[valid_idx]                # (n_valid, n_params)
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
        ax.axhline(navg,               color='k',  ls='-',  label='Expected')
        ax.axhline(navg - navg**0.5,   color='k',  ls='--', alpha=0.5)
        ax.axhline(navg + navg**0.5,   color='k',  ls='--', alpha=0.5)
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
        ax.plot(cdf, cdf,  'k--', label='Ideal')
        ax.plot(xr,  cdf,  lw=2,  label='Posterior')
        ax.set(aspect='equal', adjustable='box')
        ax.set_title(label)
        ax.set_xlabel('Predicted Percentile')
        ax.grid(True)
    axes[0].set_ylabel('Empirical Percentile')
    plt.tight_layout()
    plt.savefig(val_dir / 'plot_coverage.jpg', dpi=150)
    plt.close()

    # --- TARP ---
    # Need enough valid points for num_alpha_bins to be meaningful.
    # tarp defaults to n_sims // 10 bins. So we need at least n_valid >= 50
    # to get 5+ bins. Raise threshold accordingly.
    TARP_MIN_POINTS = 50

    if n_valid >= TARP_MIN_POINTS:
        try:
            import tarp

            # Explicitly set num_alpha_bins — never rely on the n_sims // 10 default
            # Use ~n_valid / 3 bins, capped between 10 and 50
            n_alpha_bins = int(np.clip(n_valid / 3, 10, 50))

            # tarp expects (n_samples, n_sims, n_params)
            # all_samples is (n_valid, n_post, n_params) → transpose to (n_post, n_valid, n_params)
            samples_for_tarp = all_samples.transpose(1, 0, 2)

            print(f"  TARP shapes: samples={samples_for_tarp.shape}, "
                  f"theta={theta_subset.shape}, n_alpha_bins={n_alpha_bins}")

            ecp, alpha = tarp.get_tarp_coverage(samples_for_tarp, theta_subset,
            references='random',metric='euclidean',norm=True,bootstrap=True,
            num_bootstrap=100,num_alpha_bins=n_alpha_bins,)

            # ecp shape with bootstrap=True: (num_bootstrap, num_alpha_bins)
            print(f"  TARP ecp shape: {ecp.shape}, alpha shape: {alpha.shape}")
            print(f"  ecp mean range: [{np.mean(ecp, axis=0).min():.3f}, "
                  f"{np.mean(ecp, axis=0).max():.3f}]")

            ecp_mean = np.mean(ecp, axis=0)
            ecp_std  = np.std(ecp,  axis=0)

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.plot([0, 1], [0, 1], 'k--', label='Ideal')
            ax.plot(alpha, ecp_mean, color='b', lw=2, label='TARP')
            ax.fill_between(alpha,ecp_mean - ecp_std,ecp_mean + ecp_std,
                        alpha=0.3, color='b', label=r'$\pm 1\sigma$')
            ax.fill_between(alpha,ecp_mean - 2 * ecp_std,ecp_mean + 2 * ecp_std,
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
BASE_DIR = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin'
CSV_FILES = [
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/wl/lhs_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/wl/round2_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/wl/round3_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/wl/round4_samples.csv',
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/wl/round5_samples.csv',
#    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/wl/round6_samples.csv'
    ]
NEXT_ROUND = int(len(CSV_FILES)+1)
print(f"The next round of samples to be run will be round {NEXT_ROUND}")
NSIDE = 512
SCALES = [4.0, 8.0, 16.0, 32.0, 64.0]
CACHE_DIR = 'sample_vector_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_moments(path):
    # Standard glob pattern from 4-param script
    pattern = os.path.join(path, "**", f"allmaps_nside{NSIDE}_z*_split*.pkl")
    files = glob.glob(pattern, recursive=True)
    if len(files) < 4: return None
    
    ymap, gmap, kmap, tmap = [np.zeros(12*NSIDE**2, dtype=np.float32) for _ in range(4)]
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
                    ra_gal  = gal_data[:, 0]
                    dec_gal = gal_data[:, 1]
                    # Satellite galaxies near poles can have Dec slightly outside [-90, 90]
                    # due to HOD offset placement not clipping coordinates after get_sim_map.
                    # Wrap RA to [0, 360] and clip Dec to valid range before pixelisation.
                    ra_gal  = ra_gal % 360.0
                    dec_gal = np.clip(dec_gal, -90.0, 90.0)

                    n_bad = np.sum(~np.isfinite(ra_gal) | ~np.isfinite(dec_gal))
                    if n_bad > 0:
                        print(f"  [WARN] Dropping {n_bad} galaxies with non-finite coordinates in chunk {chunk_idx}")
                        mask   = np.isfinite(ra_gal) & np.isfinite(dec_gal)
                        ra_gal, dec_gal = ra_gal[mask], dec_gal[mask]

                    if len(ra_gal) > 0:
                        pix   = hp.ang2pix(NSIDE, ra_gal, dec_gal, lonlat=True)
                        gmap += np.bincount(pix, minlength=12 * NSIDE**2)
    
    dg = (gmap / np.mean(gmap) - 1.0) if np.mean(gmap) > 0 else np.zeros_like(gmap)
    vec = []
    for th in SCALES:
        f = np.radians(th/60.)
        # Smoothing with EXPLICIT LMAX for consistency
        gs = hp.smoothing(dg, fwhm=f, lmax=LMAX, verbose=False)
        ys = hp.smoothing(ymap, fwhm=f, lmax=LMAX, verbose=False)
        ts = hp.smoothing(tmap, fwhm=f, lmax=LMAX, verbose=False)
        ks = hp.smoothing(kmap, fwhm=f, lmax=LMAX, verbose=False)

        # 3-point moments <ggTracer> (indices 0-14 in output vector)
        vec.extend([np.mean(gs**2 * ys), np.mean(gs**2 * ts), np.mean(gs**2 * ks)])
        # 2-point moments <gTracer> (indices 15-29 in output vector)
        vec.extend([np.mean(gs * ys), np.mean(gs * ts), np.mean(gs * ks)])
    return np.array(vec)
print("Extracting reference run...")
# Reference is in original location
x_obs = extract_moments(os.path.join(BASE_DIR, 'reference_run')).astype(np.float32)
np.save('x_obs.npy', x_obs)

theta_train, x_train = [], []
for csv in CSV_FILES:
    df = pd.read_csv(csv)
    
    # Determine folder offset based on the round
    if "lhs" in csv: offset = 0
    elif "round2" in csv: offset = 30
    elif "round3" in csv: offset = 60
    elif "round4" in csv: offset = 90
    elif "round5" in csv: offset = 120
    elif "round6" in csv: offset = 150

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {os.path.basename(csv)}"):
        sid = int(row['sample_id']) + offset
        cache_file = os.path.join(CACHE_DIR, f"x_sample_{sid}.npy")

        if os.path.exists(cache_file):
            v = np.load(cache_file)
        else:
            # Note the nesting
            s_path = os.path.join(BASE_DIR, "twoparams", "wl", f"sample_{sid}")
            v = extract_moments(s_path)
            if v is not None: np.save(cache_file, v)

        if v is not None:
            # ONLY 2 PARAMETERS: theta_ej_0 and mu_beta
            theta_train.append([row['theta_ej_0'], row['mu_beta']])
            x_train.append(v)

x_train = np.array(x_train).astype(np.float32)
theta_train = np.array(theta_train).astype(np.float32)
np.save('x_train_full.npy', x_train)   # full pooled unnormalised summaries
np.save('theta.npy', theta_train)

# =============================================================================
# 2. LTU-ILI TRAINING LOOP
# =============================================================================
# stat_map for single-loop interleaved layout
# Layout per scale: [3pt_y, 3pt_tau, 3pt_k, 2pt_y, 2pt_tau, 2pt_k]
stat_map = {
    'g2y':         [0,  6,  12, 18, 24],
    'g2tau':       [1,  7,  13, 19, 25],
    'g2kappa':     [2,  8,  14, 20, 26],
    'gy':          [3,  9,  15, 21, 27],
    'gtau':        [4,  10, 16, 22, 28],
    'gkappa':      [5,  11, 17, 23, 29],
    'JOINT':       list(range(30)),
    # Tracer totals (3pt + 2pt for each tracer)
    'y_total':     [0,  3,  6,  9,  12, 15, 18, 21, 24, 27],
    'tau_total':   [1,  4,  7,  10, 13, 16, 19, 22, 25, 28],
    'kappa_total': [2,  5,  8,  11, 14, 17, 20, 23, 26, 29],
    # Category totals
    'all_3pt':     [0,  1,  2,  6,  7,  8,  12, 13, 14, 18, 19, 20, 24, 25, 26],
    'all_2pt':     [3,  4,  5,  9,  10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29],
}

np.random.seed(42)
all_indices = np.arange(len(theta_train))
np.random.shuffle(all_indices)

n_val     = max(5, int(len(theta_train) * 0.20))
val_idx   = all_indices[:n_val]
train_idx = all_indices[n_val:]

theta_val = theta_train[val_idx]
theta_train_set = theta_train[train_idx]

# Statistics to validate DURING the active learning loop
# Only JOINT matters for proposal generation
VALIDATE_DURING_LOOP = {'JOINT'}
# Statistics to validate AFTER all rounds are done
# (run these separately once you have 100+ samples)
VALIDATE_POST_HOC = set(stat_map.keys()) - VALIDATE_DURING_LOOP

for name, idx in stat_map.items():
    print(f"\n--- Training NPE: {name} ---")

    # Split the statistics
    xt_full = x_train[:, idx]
    xt_train = xt_full[train_idx]
    #xt_train = xt_train + 0.01 * np.random.randn(*xt_train.shape).astype(np.float32)
    xt_val = xt_full[val_idx]
    xo = x_obs[idx]

    #standardize the stats
    x_mean = np.mean(xt_train, axis=0)
    x_std  = np.std(xt_train, axis=0)
    xt_train = (xt_train - x_mean) / (x_std + 1e-8)
    xt_val   = (xt_val   - x_mean) / (x_std + 1e-8)
    xo       = (xo       - x_mean) / (x_std + 1e-8)

    np.save(f'scaler_{name}_mean.npy',x_mean)
    np.save(f'scaler_{name}_std.npy',x_std)
    np.save(f'x_{name}.npy', xt_train)
    np.save(f'xobs_{name}.npy', xo)
    np.save(f'theta_train_{name}.npy', theta_train_set)

    loader = StaticNumpyLoader(
        in_dir='./',
        x_file=f'x_{name}.npy',
        theta_file=f'theta_train_{name}.npy',
        xobs_file=f'xobs_{name}.npy'
    )

    n_stats    = len(idx)
    n_train    = len(theta_train_set)
    ratio      = n_train / n_stats   # samples per statistic dimension

    hfs = 64 if n_stats >= 10 else 32
    nts = 4  if n_stats >= 10 else 3

    if ratio < 5:
        # Too few training samples per statistic for NSF LU decomposition to be stable
        # MAF only, so no LU inversion, numerically robust at low sample counts
        print(f"  [ARCH] MAF-only (ratio={ratio:.1f} samples/stat < 5)")
        nets = load_nde_sbi(engine='NPE', model='maf', repeats=8,
            hidden_features=hfs, num_transforms=nts)
    else:
        # Enough samples for NSF + MAF ensemble
        print(f"  [ARCH] NSF+MAF (ratio={ratio:.1f} samples/stat >= 5)")
        nets = (load_nde_sbi(engine='NPE', model='nsf', repeats=2,
                     hidden_features=hfs, num_transforms=nts) \
                + load_nde_sbi(engine='NPE', model='maf', repeats=6,
                     hidden_features=hfs, num_transforms=nts))

    runner = InferenceRunner.load(backend='sbi', engine='NPE',
        prior=Uniform(low=[1.0, 0.01], high=[6.0, 1.5]),
        nets=nets, out_dir=Path(f'./sbi_logs_{name}'))
    posterior, _ = runner(loader)

    with open(f'ili_posterior_{name}.pkl', 'wb') as f:
        pk.dump(posterior, f)

    # Only validate the statistics that matter right now
    if name not in VALIDATE_DURING_LOOP:
        print(f"  [SKIP] Validation deferred for {name}. Run post-hoc after all rounds.")
        continue

    # --- JOINT VALIDATION ONLY ---
    labels  = [r'$\theta_{ej,0}$', r'$\mu_{\beta}$']
    print(f"  Validating {name} on {n_val} hold-out points...")

    run_validation_direct(
    name       = name,
    posterior  = posterior,
    xt_val     = xt_val,       # already normalised
    theta_val  = theta_val,
    labels     = labels,
    val_dir    = Path(f'./validation_{name}'),
    prior_low  = [1.0,  0.01],
    prior_high = [6.0,  1.5],)
# =============================================================================
# 3. NEXT ROUND PROPOSAL (Next 30 Samples)
# =============================================================================

with open('ili_posterior_JOINT.pkl', 'rb') as f:
    joint_posterior = pk.load(f)

#reuse the normalised xobs saved during training:
xo_joint_norm = np.load('xobs_JOINT.npy')   # already standardised
xo_tensor     = torch.from_numpy(xo_joint_norm).float().reshape(1, -1)
next_theta    = joint_posterior.sample((30,), x=xo_tensor).detach().cpu().numpy()

# Clip to prior bounds as flow tails occasionally escape
next_theta = np.clip(next_theta, a_min=[1.0, 0.01], a_max=[6.0, 1.5])

pd.DataFrame(next_theta,
             columns=['theta_ej_0', 'mu_beta']
            ).to_csv(f'round{NEXT_ROUND}_samples.csv', index_label='sample_id')

print(f"\nFinished. Generated round{NEXT_ROUND}_samples.csv for the 2-parameter project.")

# =============================================================================
# 4. PER STATISTIC VALIDATION
# =============================================================================
# Run this once you have 100+ total samples (~20+ val points)
# At that point TARP becomes meaningful
FULL_VALIDATE_THRESHOLD = 100  # total samples with  80 in train, 20 in val

if len(theta_train) >= FULL_VALIDATE_THRESHOLD:
    print("\nRunning post-hoc validation for all statistics...")

    for name in stat_map:
        if name == 'JOINT':
            continue  # already validated during the active learning loop

        posterior_path = f'ili_posterior_{name}.pkl'
        if not os.path.exists(posterior_path):
            print(f"  [SKIP] No saved posterior for {name}, skipping.")
            continue

        with open(posterior_path, 'rb') as f:
            post = pk.load(f)

        idx = stat_map[name]

        # Load scalers if saved during training, otherwise recompute from
        # the training split — must use train set only, never val set
        scaler_mean_path = f'scaler_{name}_mean.npy'
        scaler_std_path  = f'scaler_{name}_std.npy'

        if os.path.exists(scaler_mean_path) and os.path.exists(scaler_std_path):
            x_mean = np.load(scaler_mean_path)
            x_std  = np.load(scaler_std_path)
        else:
            # Recompute from training split
            # x_train and train_idx are defined above
            xt_tr  = x_train[train_idx][:, idx]
            x_mean = xt_tr.mean(axis=0)
            x_std  = xt_tr.std(axis=0) + 1e-8
            print(f"  [INFO] Recomputed scaler for {name} from training split.")

        # Normalise validation set with training scaler
        # x_train and val_idx are defined above
        xt_v = ((x_train[val_idx][:, idx] - x_mean) / x_std).astype(np.float32)
        th_v = theta_train[val_idx]

        # TARP needs 20+ points to be meaningful
        n_val    = len(th_v)
        plots    = ['histogram', 'coverage']
        if n_val >= 20:
            plots.append('tarp')

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

print("All done, generate plots and run the next round of samples!")
