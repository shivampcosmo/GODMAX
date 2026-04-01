import os
import sys
import threading
import multiprocessing as mp
import torch
import numpy as np
import healpy as hp
import pickle as pk
import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import random
# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================
LTU_ILI_PATH = '/work/hdd/bdne/aacharya2/ltu-ili'
sys.path.append(LTU_ILI_PATH)

from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.utils import load_nde_sbi, Uniform

BASE_DIR  = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/new'
WORK_DIR  = str(Path(__file__).parent.resolve())   # absolute path to script dir
CACHE_DIR = os.path.join(WORK_DIR, 'sample_vector_cache')

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(42)
# Each entry: (csv_path, global_id_offset)
CSV_FILES = [
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/lhs_samples.csv', 0),
#   ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/round2_samples.csv', 500),
#   ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/round3_samples.csv', 700),
]
NEXT_ROUND = len(CSV_FILES) + 1

NSIDE        = 512
SCALES       = [4.0, 8.0, 16.0, 32.0, 64.0]   # arcmin
PRIOR_LOW    = [1.0, -0.3]
PRIOR_HIGH   = [6.0,  0.0]
PARAM_LABELS = [r'$$\theta_{ej,0}$$', r'$${\nu_{\theta_{ej}}}^{M}$$']
PARAM_NAMES  = ['theta_ej_0', 'nu_theta_ej_M']
PROPOSAL_STAT           = 'gtau'
FULL_VALIDATE_THRESHOLD = 100
VALIDATION_SEED         = 99
VAL_FRACTION            = 0.15

# Vector layout: 6 statistics x 5 angular scales = 30 elements.
# Within each scale block: [0]<ggy> [1]<ggtau> [2]<ggkappa> [3]<gy> [4]<gtau> [5]<gkappa>
STAT_MAP = {
    # Individual 3pt cross-moments
    'g2y':         [0,  6,  12, 18, 24],
    'g2tau':       [1,  7,  13, 19, 25],
    'g2kappa':     [2,  8,  14, 20, 26],
    # Individual 2pt cross-moments
    'gy':          [3,  9,  15, 21, 27],
    'gtau':        [4,  10, 16, 22, 28],
    'gkappa':      [5,  11, 17, 23, 29],
    # Full joint
    'JOINT':       list(range(30)),
    # Per-tracer totals (3pt + 2pt)
    'y_total':     [0,  3,  6,  9,  12, 15, 18, 21, 24, 27],
    'tau_total':   [1,  4,  7,  10, 13, 16, 19, 22, 25, 28],
    'kappa_total': [2,  5,  8,  11, 14, 17, 20, 23, 26, 29],
    # Category totals
    'all_3pt':     [0,  1,  2,  6,  7,  8,  12, 13, 14, 18, 19, 20, 24, 25, 26],
    'all_2pt':     [3,  4,  5,  9,  10, 11, 15, 16, 17, 21, 22, 23, 27, 28, 29],
}
N_STATISTICS = len(STAT_MAP)   # 12 --> matches --cpus-per-task in the job script


# =============================================================================
# SAMPLING UTILITIES
# =============================================================================

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
    Bypasses EnsemblePosterior's importance-weighted resampling which
    triggers the rejection-sampling hang. Uses per-member thread timeouts
    since sbi announces hangs via logging.warning(), not warnings.warn().

    Args:
        posterior:           trained ltu-ili/sbi posterior object
        x_obs_norm:          normalised observation vector, shape (n_stats,)
        n_samples:           total posterior samples to return
        timeout_per_member:  seconds before abandoning a hung member

    Returns:
        np.ndarray of shape (n_samples, n_params), or None if all members fail
    """
    x_t = torch.from_numpy(np.asarray(x_obs_norm)).float().reshape(1, -1)

    try:
        members = posterior.posteriors
    except AttributeError:
        result, exception = [None], [None]
        t = threading.Thread(
            target=_sample_member_thread,
            args=(posterior, x_t, n_samples, result, exception),
            daemon=True,
        )
        t.start()
        t.join(timeout=timeout_per_member)
        if t.is_alive():
            print(f'    [WARN] Single posterior timed out after {timeout_per_member}s.')
            return None
        if exception[0] is not None:
            print(f'    [WARN] Single posterior failed: {exception[0]}')
            return None
        return result[0]

    n_members  = len(members)
    per_member = max(1, n_samples // n_members)
    collected  = []

    for i, member in enumerate(members):
        result, exception = [None], [None]
        t = threading.Thread(
            target=_sample_member_thread,
            args=(member, x_t, per_member, result, exception),
            daemon=True,
        )
        t.start()
        t.join(timeout=timeout_per_member)
        if t.is_alive():
            print(f'    [SKIP] Member {i} timed out (rejection sampling hung).')
        elif exception[0] is not None:
            print(f'    [SKIP] Member {i} failed: {exception[0]}')
        else:
            collected.append(result[0])

    if not collected:
        print(f'    [WARN] All {n_members} ensemble members failed or timed out.')
        return None

    n_ok = len(collected)
    if n_ok < n_members:
        print(f'    [INFO] {n_ok}/{n_members} members contributed samples.')

    combined = np.concatenate(collected, axis=0)
    np.random.shuffle(combined)
    if len(combined) >= n_samples:
        return combined[:n_samples]
    else:
        print(f'    [INFO] Only {len(combined)} raw samples --> '
              f'upsampling to {n_samples} with replacement.')
        idx = np.random.choice(len(combined), size=n_samples, replace=True)
        return combined[idx]


def run_validation_direct(name, posterior, xt_val, theta_val, labels, val_dir,
                           prior_low, prior_high, n_post=1000):
    """
    Posterior calibration diagnostics via direct per-member sampling.
    Produces rank histogram, coverage plot, and TARP.

    PlotSinglePosterior from ltu-ili is NOT used here: it calls
    posterior.sample() on the EnsemblePosterior directly, which triggers
    the same rejection-sampling hang as PosteriorCoverage.

    Args:
        name:       statistic name string
        posterior:  trained posterior
        xt_val:     normalised validation summaries, shape (n_val, n_stats)
        theta_val:  true parameters, shape (n_val, n_params)
        labels:     parameter label strings
        val_dir:    Path to output directory
        prior_low:  prior lower bounds
        prior_high: prior upper bounds
        n_post:     posterior samples per validation point
    """
    val_dir = Path(val_dir)
    val_dir.mkdir(exist_ok=True)

    n_val, n_params = theta_val.shape
    if n_val < 5:
        print(f'  [SKIP] Only {n_val} val points. Need >=5.')
        return

    print(f'  Validating {name} on {n_val} points...')
    all_samples, valid_idx = [], []
    for i in range(n_val):
        s = sample_ensemble_direct(posterior, xt_val[i], n_samples=n_post)
        if s is None:
            print(f'    [WARN] All members failed for val point {i}, skipping.')
            continue
        all_samples.append(np.clip(s, prior_low, prior_high))
        valid_idx.append(i)

    if len(valid_idx) < 3:
        print(f'  [SKIP] Fewer than 3 successful val points for {name}.')
        return

    all_samples  = np.array(all_samples)    # (n_valid, n_post, n_params)
    theta_subset = theta_val[valid_idx]      # (n_valid, n_params)
    n_valid      = len(valid_idx)
    print(f'  Collected samples for {n_valid}/{n_val} validation points.')

    # Rank statistic: uniform ranks → calibrated posterior
    ranks = (all_samples < theta_subset[:, None, :]).sum(axis=1)

    # --- Rank histogram ---
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 4))
    if n_params == 1:
        axes = [axes]
    navg = n_valid / 10
    for p, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.hist(ranks[:, p], bins=10, range=(0, n_post))
        ax.axhline(navg,             color='k', ls='-',  label='Expected')
        ax.axhline(navg - navg**0.5, color='k', ls='--', alpha=0.5)
        ax.axhline(navg + navg**0.5, color='k', ls='--', alpha=0.5)
        ax.set_title(lbl)
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
    for p, (ax, lbl) in enumerate(zip(axes, labels)):
        xr = np.sort(ranks[:, p] / n_post)
        ax.plot(cdf, cdf, 'k--', label='Ideal')
        ax.plot(xr,  cdf, lw=2,  label='Posterior')
        ax.set(aspect='equal', adjustable='box')
        ax.set_title(lbl)
        ax.set_xlabel('Predicted Percentile')
        ax.grid(True)
    axes[0].set_ylabel('Empirical Percentile')
    plt.tight_layout()
    plt.savefig(val_dir / 'plot_coverage.jpg', dpi=150)
    plt.close()

    # --- TARP ---
    TARP_MIN = 50
    if n_valid >= TARP_MIN:
        try:
            import tarp
            n_alpha_bins     = int(np.clip(n_valid / 3, 10, 50))
            samples_for_tarp = all_samples.transpose(1, 0, 2)
            print(f'  TARP: samples={samples_for_tarp.shape}, '
                  f'theta={theta_subset.shape}, n_alpha_bins={n_alpha_bins}')
            ecp, alpha = tarp.get_tarp_coverage(
                samples_for_tarp, theta_subset,
                references='random', metric='euclidean',
                norm=True, bootstrap=True,
                num_bootstrap=100, num_alpha_bins=n_alpha_bins,
            )
            ecp_mean = np.mean(ecp, axis=0)
            ecp_std  = np.std(ecp,  axis=0)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], 'k--', label='Ideal')
            ax.plot(alpha, ecp_mean, color='b', lw=2, label='TARP')
            ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std,
                            alpha=0.3, color='b', label=r'$\pm 1\sigma$')
            ax.fill_between(alpha, ecp_mean - 2*ecp_std, ecp_mean + 2*ecp_std,
                            alpha=0.15, color='b', label=r'$\pm 2\sigma$')
            ax.set(xlabel='Credibility Level', ylabel='Expected Coverage',
                   xlim=(0, 1), ylim=(0, 1), aspect='equal',
                   title=f'{name}  (n_val={n_valid})')
            ax.legend()
            plt.tight_layout()
            plt.savefig(val_dir / 'plot_TARP.jpg', dpi=150)
            plt.close()
            print(f'  TARP saved --> {val_dir / "plot_TARP.jpg"}')
        except Exception as e:
            print(f'  [WARN] TARP failed: {e}')
    else:
        print(f'  [SKIP] TARP needs >={TARP_MIN} valid points, have {n_valid}. '
              f'Need {TARP_MIN - n_valid} more.')

    print(f'  Validation plots saved in {val_dir}')


# =============================================================================
# SUMMARY STATISTIC EXTRACTION
# =============================================================================

def extract_moments(path):
    """
    Extract summary statistics from all pkl files under path.

    Returns a float32 array of length 30, or None if no pkl files found.
    Means are restricted to the galaxy footprint mask to match what an
    observer would measure over the survey footprint.
    """
    pattern = os.path.join(path, '**', f'allmaps_sim_B12_nside{NSIDE}.pkl')
    files   = glob.glob(pattern, recursive=True)
    if not files:
        return None

    LMAX = 3 * NSIDE - 1
    ymap = np.zeros(12 * NSIDE**2, dtype=np.float32)
    gmap = np.zeros(12 * NSIDE**2, dtype=np.float32)
    kmap = np.zeros(12 * NSIDE**2, dtype=np.float32)
    tmap = np.zeros(12 * NSIDE**2, dtype=np.float32)

    for fpath in files:
        with open(fpath, 'rb') as h:
            data = pk.load(h)
        ymap += np.nan_to_num(data.get('map_ymap',  0))
        kmap += np.nan_to_num(data.get('map_kappa', 0))
        tmap += np.nan_to_num(data.get('map_tau',   0))
        for chunk_idx, gal_data in data.get('mock_gals_all', {}).items():
            if gal_data.size == 0:
                continue
            ra_gal  = gal_data[:, 0] % 360.0
            dec_gal = np.clip(gal_data[:, 1], -90.0, 90.0)
            mask    = np.isfinite(ra_gal) & np.isfinite(dec_gal)
            if mask.any():
                pix   = hp.ang2pix(NSIDE, ra_gal[mask], dec_gal[mask], lonlat=True)
                gmap += np.bincount(pix, minlength=12 * NSIDE**2)

    footprint  = gmap > 0
    mean_g_pix = np.mean(gmap[footprint])
    dg = (gmap / mean_g_pix - 1.0) if mean_g_pix > 0 else np.zeros_like(gmap)

    vec = []
    for th in SCALES:
        fwhm = np.radians(th / 60.0)
        gs   = hp.smoothing(dg,   fwhm=fwhm, lmax=LMAX, verbose=False)
        ys   = hp.smoothing(ymap, fwhm=fwhm, lmax=LMAX, verbose=False)
        ts   = hp.smoothing(tmap, fwhm=fwhm, lmax=LMAX, verbose=False)
        ks   = hp.smoothing(kmap, fwhm=fwhm, lmax=LMAX, verbose=False)
        fp   = footprint
        vec.extend([
            float(np.mean((gs**2 * ys)[fp])),
            float(np.mean((gs**2 * ts)[fp])),
            float(np.mean((gs**2 * ks)[fp])),
            float(np.mean((gs * ys)[fp])),
            float(np.mean((gs * ts)[fp])),
            float(np.mean((gs * ks)[fp])),
        ])

    return np.array(vec, dtype=np.float32)


# =============================================================================
# TRAINING WORKER — runs in its own process via multiprocessing.Pool
# =============================================================================

def train_one_statistic(args):
    """
    Train a single NPE for one entry in STAT_MAP.
    Designed to run in a subprocess via multiprocessing.Pool so that all
    statistics are trained in parallel, one process per CPU.

    Each subprocess imports ltu-ili independently, which is safe because
    ltu-ili has no shared mutable global state between statistics.

    Args:
        args: tuple of (name, idx, x_train, theta_train, x_obs, work_dir, device)

    Returns:
        (name, success: bool, message: str)
    """
    torch.set_num_threads(1)  # prevent OpenMP contention across parallel workers
    name, idx, x_train, theta_train, x_obs, work_dir, device = args

    # Re-add path in subprocess --> sys.path is not inherited via pickle
    sys.path.append(LTU_ILI_PATH)
    from ili.dataloaders import StaticNumpyLoader
    from ili.inference import InferenceRunner
    from ili.utils import load_nde_sbi, Uniform

    # All file I/O uses absolute paths so subprocess working directory does
    # not matter
    def fpath(fname):
        return os.path.join(work_dir, fname)

    try:
        n_stats = len(idx)
        n_train = len(theta_train)
        ratio   = n_train / n_stats

        xt_full = x_train[:, idx]
        xo      = x_obs[idx]

        x_mean  = np.mean(xt_full, axis=0)
        x_std   = np.std(xt_full,  axis=0)
        xt_norm = ((xt_full - x_mean) / (x_std + 1e-8)).astype(np.float32)
        xo_norm = ((xo      - x_mean) / (x_std + 1e-8)).astype(np.float32)

        np.save(fpath(f'scaler_{name}_mean.npy'), x_mean)
        np.save(fpath(f'scaler_{name}_std.npy'),  x_std)
        np.save(fpath(f'x_{name}.npy'),           xt_norm)
        np.save(fpath(f'xobs_{name}.npy'),        xo_norm)
        np.save(fpath(f'theta_train_{name}.npy'), theta_train)
        
        print(f'{name:12s}  mean_std={x_std.mean():.4e}  '
          f'min_std={x_std.min():.4e}  max_std={x_std.max():.4e}')
        frac_below = (xo < xt_full.min(axis=0)).mean()
        frac_above = (xo > xt_full.max(axis=0)).mean()
        print(f'{name:12s}  frac_below_sim_range={frac_below:.2f}  '
          f'frac_above_sim_range={frac_above:.2f}')

        loader = StaticNumpyLoader(
            in_dir=work_dir,
            x_file=f'x_{name}.npy',
            theta_file=f'theta_train_{name}.npy',
            xobs_file=f'xobs_{name}.npy',
        )

        # --- Adaptive ensemble and network size ---
        if n_stats <= 5:
            hfs, nts = 16, 3     # was 16, 2
            repeats  = 5         # was 3
        elif n_stats <= 15:
            hfs, nts = 32, 3    # was 32, 3
            repeats  = 6         # was 4
        else:
            hfs, nts = 32, 4    # was 64, 4
            repeats  = 6

        # --- Architecture selection ---
        if n_stats <= 5:
            nsf_threshold = 8
        elif n_stats <= 15:
            nsf_threshold = 12
        elif n_stats <= 30:
            nsf_threshold = 20
        else:
            nsf_threshold = 999

        # --- Adaptive batch size ---
        n_train_eff = int(n_train * 0.85)
        batch_size  = int(np.clip(n_train_eff // 8, 32, 256))

        train_args = {
            'training_batch_size': batch_size,
            'learning_rate':       5e-4,
            'max_num_epochs':      500,
            'stop_after_epochs':   50,
            'clip_max_norm':       5.0,
            'validation_fraction': 0.15,
        }
        '''
        if ratio < nsf_threshold:
            nets = load_nde_sbi(
                engine='NPE', model='maf',
                repeats=repeats, hidden_features=hfs, num_transforms=nts,
            )
            arch_str = f'MAF (ratio={ratio:.1f} < {nsf_threshold})'
        else:
            n_nsf = max(1, repeats // 3)
            n_maf = repeats - n_nsf
            nets  = (
                load_nde_sbi(engine='NPE', model='nsf', repeats=n_nsf,
                             hidden_features=hfs, num_transforms=nts)
                + load_nde_sbi(engine='NPE', model='maf', repeats=n_maf,
                               hidden_features=hfs, num_transforms=nts)
            )
            arch_str = (f'NSF+MAF (ratio={ratio:.1f} >= {nsf_threshold}, '
                        f'{n_nsf} NSF + {n_maf} MAF)')
        '''
        if n_stats <= 20:
            nets = load_nde_sbi(engine='NPE', model='nsf',repeats=repeats,
                                hidden_features=hfs, num_transforms=nts)
            arch_str = (f'NSF pure (n_stats={n_stats}, ratio={ratio:.1f}, '
                        f'repeats={repeats}, hfs={hfs}, nts={nts})')
        else:
            #n_nsf = max(2, repeats // 2)    # majority NSF, not minority
            #n_maf = repeats - n_nsf
            n_maf = 2
            n_nsf = repeats - n_maf
            nets  = (load_nde_sbi(engine='NPE', model='nsf', repeats=n_nsf,
                             hidden_features=hfs, num_transforms=nts)
                + load_nde_sbi(engine='NPE', model='maf', repeats=n_maf,
                               hidden_features=hfs, num_transforms=nts))
            arch_str = (f'NSF+MAF (ratio={ratio:.1f}, '
                        f'{n_nsf} NSF + {n_maf} MAF, '
                        f'hfs={hfs}, nts={nts})')

        runner = InferenceRunner.load(
            backend='sbi', engine='NPE',
            prior=Uniform(low=PRIOR_LOW, high=PRIOR_HIGH),
            nets=nets,
            out_dir=Path(fpath(f'sbi_logs_{name}')),
            device=device,
            train_args=train_args,
        )
        posterior, _ = runner(loader)

        with open(fpath(f'ili_posterior_{name}.pkl'), 'wb') as f:
            pk.dump(posterior, f)

        msg = (f'[{name}] DONE --> {n_stats} stats, {n_train} sims, '
               f'{arch_str}, repeats={repeats}, hfs={hfs}, nts={nts}, '
               f'batch={batch_size}')
        return name, True, msg

    except Exception as e:
        import traceback
        return name, False, f'[{name}] FAILED: {traceback.format_exc()}'


# =============================================================================
# DATA LOADING
# =============================================================================

if __name__ == '__main__':
    # Guard required for multiprocessing on some platforms.
    # All top-level executable code lives inside this block.

    os.makedirs(CACHE_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'The next round of samples to be run will be round {NEXT_ROUND}')

    # --- x_obs ---
    print('Extracting reference run (x_obs)...')
    x_obs = extract_moments(os.path.join(BASE_DIR, 'reference_run'))
    if x_obs is None:
        raise RuntimeError('No pkl files found in reference_run directory.')
    np.save(os.path.join(WORK_DIR, 'x_obs.npy'), x_obs)
    print(f'  x_obs shape: {x_obs.shape}')

    # --- Training data ---
    theta_list, x_list = [], []
    for csv_path, offset in CSV_FILES:
        df = pd.read_csv(csv_path)
        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f'Loading {os.path.basename(csv_path)}'):
            sid        = int(row['sample_id']) + offset
            cache_file = os.path.join(CACHE_DIR, f'x_sample_{sid}.npy')
            if os.path.exists(cache_file):
                v = np.load(cache_file)
            else:
                v = extract_moments(os.path.join(BASE_DIR, f'sample_{sid}'))
                if v is not None:
                    np.save(cache_file, v)
            if v is not None:
                theta_list.append([row['theta_ej_0'], row['nu_theta_ej_M']])
                x_list.append(v)

    x_train     = np.array(x_list,     dtype=np.float32)
    theta_train = np.array(theta_list, dtype=np.float32)
    np.save(os.path.join(WORK_DIR, 'x_train_full.npy'),    x_train)
    np.save(os.path.join(WORK_DIR, 'theta_train_full.npy'), theta_train)
    print(f'Loaded {len(theta_train)} simulations. '
          f'x_train: {x_train.shape}, theta_train: {theta_train.shape}')

    # =========================================================================
    # PARALLEL TRAINING
    # =========================================================================
    # One process per statistic. n_cpus is capped at the number of statistics
    # so we never spawn idle processes. SLURM --cpus-per-task controls how
    # many processes actually run simultaneously.
    n_cpus = min(N_STATISTICS, int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    print(f'\nTraining {N_STATISTICS} posteriors in parallel '
          f'({n_cpus} processes)...')

    worker_args = [
        (name, idx, x_train, theta_train, x_obs, WORK_DIR, device)
        for name, idx in STAT_MAP.items()
    ]

    # mp.get_context('spawn') is used rather than the default 'fork' because:
    # 1. 'fork' + PyTorch = undefined behaviour (PyTorch uses internal threads)
    # 2. 'spawn' starts a clean interpreter so each worker imports its own
    #    copy of torch/ltu-ili with no shared state
    ctx  = mp.get_context('spawn')
    with ctx.Pool(processes=n_cpus) as pool:
        results = pool.map(train_one_statistic, worker_args)

    print('\n--- Training Summary ---')
    all_ok = True
    for name, success, msg in results:
        status = 'OK  ' if success else 'FAIL'
        print(f'  [{status}] {msg}')
        if not success:
            all_ok = False

    if not all_ok:
        raise RuntimeError('One or more statistics failed to train. '
                           'See messages above.')

    # =========================================================================
    # NEXT-ROUND ACTIVE LEARNING PROPOSAL
    # =========================================================================
    print(f'\nGenerating round {NEXT_ROUND} proposals from '
          f'{PROPOSAL_STAT} posterior...')

    proposal_pkl  = os.path.join(WORK_DIR, f'ili_posterior_{PROPOSAL_STAT}.pkl')
    proposal_xobs = os.path.join(WORK_DIR, f'xobs_{PROPOSAL_STAT}.npy')

    if not os.path.exists(proposal_pkl):
        raise FileNotFoundError(
            f'Proposal posterior not found: {proposal_pkl}.')

    with open(proposal_pkl, 'rb') as f:
        proposal_posterior = pk.load(f)
    xo_proposal_norm = np.load(proposal_xobs)

    next_theta = sample_ensemble_direct(
        proposal_posterior, xo_proposal_norm, n_samples=200)
    if next_theta is None:
        raise RuntimeError(
            f'Failed to sample from {PROPOSAL_STAT} posterior. '
            'Check training logs for degenerate ensemble members.')

    next_theta = np.clip(next_theta, a_min=PRIOR_LOW, a_max=PRIOR_HIGH)

    out_csv = (f'/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/'
               f'round{NEXT_ROUND}_samples.csv')
    pd.DataFrame(next_theta, columns=PARAM_NAMES).to_csv(
        out_csv, index_label='sample_id')
    print(f'Saved {len(next_theta)} proposals to {out_csv}')
    for i, pname in enumerate(PARAM_NAMES):
        col = next_theta[:, i]
        print(f'  {pname}: mean={col.mean():.3f}, std={col.std():.3f}, '
              f'range=[{col.min():.3f}, {col.max():.3f}]')

    # =========================================================================
    # POST-HOC VALIDATION
    # =========================================================================
    # Uncomment once you have >= FULL_VALIDATE_THRESHOLD simulations.
    # Uses a separate held-out set drawn with VALIDATION_SEED so it is
    # reproducible and independent of the training seed.

    '''
    if len(theta_train) >= FULL_VALIDATE_THRESHOLD:
        print('\nRunning post-hoc validation for all statistics...')
        rng     = np.random.default_rng(VALIDATION_SEED)
        n_val   = max(10, int(len(theta_train) * VAL_FRACTION))
        val_idx = rng.choice(len(theta_train), size=n_val, replace=False)
        theta_val = theta_train[val_idx]

        for name, idx in STAT_MAP.items():
            posterior_path = os.path.join(WORK_DIR, f'ili_posterior_{name}.pkl')
            if not os.path.exists(posterior_path):
                print(f'  [SKIP] No saved posterior for {name}.')
                continue
            with open(posterior_path, 'rb') as f:
                post = pk.load(f)
            x_mean = np.load(os.path.join(WORK_DIR, f'scaler_{name}_mean.npy'))
            x_std  = np.load(os.path.join(WORK_DIR, f'scaler_{name}_std.npy'))
            xt_val = ((x_train[val_idx][:, idx] - x_mean) / (x_std + 1e-8)
                      ).astype(np.float32)
            run_validation_direct(
                name       = name,
                posterior  = post,
                xt_val     = xt_val,
                theta_val  = theta_val,
                labels     = PARAM_LABELS,
                val_dir    = Path(os.path.join(WORK_DIR, f'validation_{name}')),
                prior_low  = PRIOR_LOW,
                prior_high = PRIOR_HIGH,
                n_post     = 1000,
            )
    else:
        remaining = FULL_VALIDATE_THRESHOLD - len(theta_train)
        print(f'\n[SKIP] Post-hoc validation deferred — need {remaining} more sims '
              f'({len(theta_train)}/{FULL_VALIDATE_THRESHOLD}).')
    #'''

    print('\nAll done. Generate contour plots and run the next round of samples!')
