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
from ili.validation import ValidationRunner, PosteriorCoverage
from ili.utils import load_nde_sbi

BASE_DIR  = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/new/n1024'
WORK_DIR  = str(Path(__file__).parent.resolve())
CACHE_DIR = os.path.join(WORK_DIR, 'sample_vector_cache_cls')
VALIDATION_CSV = '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/validation_samples.csv'
VAL_CACHE_DIR  = os.path.join(WORK_DIR, 'validation_vector_cache_cls')

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(420)

CSV_FILES = [
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/lhs_samples.csv', 0),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round2_samples.csv', 500),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round3_samples.csv', 700),
    ]
NEXT_ROUND = len(CSV_FILES) + 1

NSIDE      = 1024
LMIN       = 100
LMAX       = 1500
N_ELL_BINS = 20
GAL_ZMIN   = 0.3
GAL_ZMAX   = 0.5

PRIOR_LOW    = [1.0, -0.3]
PRIOR_HIGH   = [6.0,  0.0]
PARAM_LABELS = [r'$$\theta_{ej,0}$$', r'$${\nu_{\theta_{ej}}}^{M}$$']
PARAM_NAMES  = ['theta_ej_0', 'nu_theta_ej_M']
PROPOSAL_STAT           = 'gy'
FULL_VALIDATE_THRESHOLD = 100
VAL_FRACTION = 0.15

# ── Ell binning ───────────────────────────────────────────────────────────────
def make_ell_bins(lmin=LMIN, lmax=LMAX, n_bins=N_ELL_BINS):
    edges   = np.logspace(np.log10(lmin), np.log10(lmax), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return edges.astype(int), centres

ELL_EDGES, ELL_CENTRES = make_ell_bins()

def bin_cl(cl_full):
    ell    = np.arange(len(cl_full))
    bp     = np.zeros(N_ELL_BINS)
    for i in range(N_ELL_BINS):
        lo, hi = ELL_EDGES[i], ELL_EDGES[i + 1]
        mask   = (ell >= lo) & (ell < hi)
        if mask.any():
            w     = 2 * ell[mask] + 1
            bp[i] = np.sum(w * cl_full[mask]) / np.sum(w)
    return bp

# ── Cl specs: (label, field_a, field_b) ──────────────────────────────────────
# field codes: 'gal_sq' = delta_gal^2 - mean, 'gal' = delta_gal,
#              'ymap', 'tau', 'kappa'
CL_SPECS = [
    ('g2y',     'gal_sq', 'ymap'),
    ('g2tau',   'gal_sq', 'tau'),
    ('g2kappa', 'gal_sq', 'kappa'),
    ('gy',      'gal',    'ymap'),
    ('gtau',    'gal',    'tau'),
    ('gkappa',  'gal',    'kappa'),
]
N_SPECS   = len(CL_SPECS)            # 6
N_SUMMARY = N_SPECS * N_ELL_BINS     # 6 * 20 = 120

# ── STAT_MAP: contiguous blocks of N_ELL_BINS per spectrum ───────────────────
_s = {label: list(range(i * N_ELL_BINS, (i + 1) * N_ELL_BINS))
      for i, (label, _, _) in enumerate(CL_SPECS)}

STAT_MAP = {
    'g2y':         _s['g2y'],
    'g2tau':       _s['g2tau'],
    'g2kappa':     _s['g2kappa'],
    'gy':          _s['gy'],
    'gtau':        _s['gtau'],
    'gkappa':      _s['gkappa'],
    'y_total':     _s['g2y']    + _s['gy'],
    'tau_total':   _s['g2tau']  + _s['gtau'],
    'kappa_total': _s['g2kappa']+ _s['gkappa'],
    'all_3pt':     _s['g2y']    + _s['g2tau']  + _s['g2kappa'],
    'all_2pt':     _s['gy']     + _s['gtau']   + _s['gkappa'],
    'JOINT':       list(range(N_SUMMARY)),
}
N_STATISTICS = len(STAT_MAP)

# Set FORCE_EQUAL_ARCH = True to give every statistic an identical network so
# that performance differences cannot be attributed to capacity differences
# (as happened with Optuna giving tau_total 3 transforms vs kappa_total 5).
# Set to False to use Optuna hyperparameters (or adaptive defaults) instead.
FORCE_EQUAL_ARCH = True
EQUAL_ARCH = {
    'hidden_features': 32,
    'num_transforms':  5,
    'learning_rate':   2e-4,
    'batch_size':      32,
    'max_num_epochs':  500,
    'repeats':         6,
}

ILIAS_BASE = '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/sbi/ilias_results'
OPTUNA_STUDY_DIRS = {name: os.path.join(ILIAS_BASE, name)
    for name in STAT_MAP}

def load_optuna_hyperparams(model_dir, study_name='study'):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    db_path = os.path.join(model_dir, 'optuna_study.db')
    if not os.path.exists(db_path):
        return None
    storage = f"sqlite:///{db_path}"
    study   = optuna.load_study(storage=storage, study_name=study_name)
    best    = study.best_trial
    mcfg    = best.user_attrs['mcfg']
    print(f'  [{os.path.basename(model_dir)}] '
          f'Best trial #{best.number}  score={best.value:.4f}  '
          f'hfs={mcfg["hidden_features"]}  nts={mcfg["num_transforms"]}  '
          f'lr={mcfg["learning_rate"]:.2e}  '
          f'batch={mcfg["batch_size"]}  '
          f'epochs={mcfg["max_epochs"]}')
    return {
        'hidden_features': mcfg['hidden_features'],
        'num_transforms':  mcfg['num_transforms'],
        'learning_rate':   mcfg['learning_rate'],
        'batch_size':      mcfg['batch_size'],
        'max_num_epochs':  mcfg['max_epochs'],
    }

INDIVIDUAL_STATS = {'gy', 'gtau', 'gkappa', 'g2y', 'g2tau', 'g2kappa'}
BLOCK_SIZE = N_ELL_BINS   # 20 ell bins per block

def make_blocks(n_features):
    """Split local indices into consecutive blocks of BLOCK_SIZE."""
    return [list(range(i, i + BLOCK_SIZE))
            for i in range(0, n_features, BLOCK_SIZE)]


# =============================================================================
# SAMPLING UTILITIES
# =============================================================================

def _sample_member_thread(member, x_t, n_samples, result, exception):
    try:
        s = member.sample((n_samples,), x=x_t, show_progress_bars=False)
        result[0] = s.detach().cpu().numpy()
    except Exception as e:
        exception[0] = e


def sample_ensemble_direct(posterior, x_obs_norm, n_samples=500,
                            timeout_per_member=45):
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
            print(f'    [SKIP] Member {i} timed out.')
        elif exception[0] is not None:
            print(f'    [SKIP] Member {i} failed: {exception[0]}')
        else:
            collected.append(result[0])

    if not collected:
        print(f'    [WARN] All {n_members} members failed or timed out.')
        return None

    n_ok = len(collected)
    if n_ok < n_members:
        print(f'    [INFO] {n_ok}/{n_members} members contributed samples.')

    combined = np.concatenate(collected, axis=0)
    np.random.shuffle(combined)
    if len(combined) >= n_samples:
        return combined[:n_samples]
    idx = np.random.choice(len(combined), size=n_samples, replace=True)
    return combined[idx]


# =============================================================================
# SUMMARY STATISTIC EXTRACTION
# =============================================================================

def extract_Cls(path):
    """
    Extract Cl summary statistics from all pkl files under path.

    Returns a float32 array of length N_SUMMARY (120), or None if no pkl
    files found. The vector is ordered as:
        [g2y bins 0..19 | g2tau bins 0..19 | g2kappa bins 0..19 |
         gy  bins 0..19 | gtau  bins 0..19 | gkappa  bins 0..19]
    """
    pattern = os.path.join(path, '**', f'allmaps_sim_B12_nside{NSIDE}.pkl')
    files   = glob.glob(pattern, recursive=True)
    if not files:
        return None

    ymap = np.zeros(12 * NSIDE**2, dtype=np.float64)
    gmap = np.zeros(12 * NSIDE**2, dtype=np.float64)
    kmap = np.zeros(12 * NSIDE**2, dtype=np.float64)
    tmap = np.zeros(12 * NSIDE**2, dtype=np.float64)

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

    mean_g = np.mean(gmap)
    if mean_g <= 0:
        return None
    delta_gal = gmap / mean_g - 1.0

    # Squared galaxy field (mean-subtracted so it is zero-mean)
    delta_sq  = delta_gal ** 2
    delta_sq -= np.mean(delta_sq)

    # Pixel window correction for cross-spectra (applied once)
    pixwin = hp.pixwin(NSIDE, lmax=LMAX)
    pw     = np.ones(LMAX + 1)
    pw[:len(pixwin)] = pixwin
    pw = np.where(pw > 0, pw, 1.0)

    field_map = {
        'gal':    delta_gal,
        'gal_sq': delta_sq,
        'ymap':   np.nan_to_num(ymap),
        'tau':    np.nan_to_num(tmap),
        'kappa':  np.nan_to_num(kmap),
    }

    vec = []
    for label, field_a, field_b in CL_SPECS:
        map_a = field_map[field_a]
        map_b = field_map[field_b]
        cl_full = hp.anafast(map_a, map_b, lmax=LMAX)
        cl_full = cl_full / pw[:len(cl_full)]   # pixel window correction
        vec.append(bin_cl(cl_full))

    return np.concatenate(vec).astype(np.float32)


# =============================================================================
# TRAINING WORKER — runs in its own process via multiprocessing.Pool
# =============================================================================

def train_one_statistic(args):
    torch.set_num_threads(1)
    name, idx, x_train, theta_train, x_obs, work_dir, device, blocks, opt_hps = args

    sys.path.append(LTU_ILI_PATH)
    from ili.dataloaders import StaticNumpyLoader
    from ili.inference import InferenceRunner
    from ili.utils import load_nde_sbi
    from sbi.utils import BoxUniform

    def fpath(fname):
        return os.path.join(work_dir, fname)

    try:
        n_stats = len(idx)
        n_train = len(theta_train)
        ratio   = n_train / n_stats

        xt_full = x_train[:, idx].astype(np.float32)
        xo      = x_obs[idx].astype(np.float32)

        if blocks is None:
            x_mean = np.mean(xt_full, axis=0)
            x_std  = np.std( xt_full, axis=0)
            x_std[x_std < 1e-10] = 1.0

            xt_norm = xt_full / x_mean
            xo_norm = xo      / x_mean

            print(f'{name:12s}  [per-feature z-score]  '
                  f'raw_mean={np.abs(x_mean).mean():.4e}  '
                  f'raw_std={x_std.mean():.4e}', flush=True)

        else:
            xt_norm = np.empty_like(xt_full)
            xo_norm = np.empty(n_stats, dtype=np.float32)
            x_mean  = np.empty(n_stats, dtype=np.float32)
            x_std   = np.empty(n_stats, dtype=np.float32)

            for blk in blocks:
                blk = np.asarray(blk)
                m = np.mean(xt_full[:, blk], axis=0)
                s = np.std( xt_full[:, blk], axis=0)
                s[s < 1e-10] = 1.0
                xt_norm[:, blk] = xt_full[:, blk] / m
                xo_norm[blk]    = xo[blk]          / m
                x_mean[blk]     = m
                x_std[blk]      = s

            print(f'{name:12s}  [per-block normalisation, {len(blocks)} blocks]  '
                  f'mean_std={x_std.mean():.4e}  '
                  f'min_std={x_std.min():.4e}  max_std={x_std.max():.4e}', flush=True)

        frac_below = (xo < xt_full.min(axis=0)).mean()
        frac_above = (xo > xt_full.max(axis=0)).mean()
        print(f'{name:12s}  frac_below_sim_range={frac_below:.2f}  '
              f'frac_above_sim_range={frac_above:.2f}', flush=True)

        np.save(fpath(f'scaler_{name}_mean.npy'), x_mean)
        np.save(fpath(f'scaler_{name}_std.npy'),  x_std)
        np.save(fpath(f'x_{name}.npy'),           xt_norm)
        np.save(fpath(f'xobs_{name}.npy'),        xo_norm)
        np.save(fpath(f'theta_train_{name}.npy'), theta_train)

        loader = StaticNumpyLoader(
            in_dir=work_dir,
            x_file=f'x_{name}.npy',
            theta_file=f'theta_train_{name}.npy',
            xobs_file=f'xobs_{name}.npy',
        )

        if FORCE_EQUAL_ARCH:
            hfs        = EQUAL_ARCH['hidden_features']
            nts        = EQUAL_ARCH['num_transforms']
            batch_size = EQUAL_ARCH['batch_size']
            lr         = EQUAL_ARCH['learning_rate']
            max_epochs = EQUAL_ARCH['max_num_epochs']
            repeats    = EQUAL_ARCH['repeats']
            print(f'{name:12s}  [forced equal arch: '
                  f'hfs={hfs}, nts={nts}, lr={lr:.2e}, '
                  f'batch={batch_size}, epochs={max_epochs}]')
        elif opt_hps is not None:
            hfs        = opt_hps['hidden_features']
            nts        = opt_hps['num_transforms']
            batch_size = opt_hps['batch_size']
            lr         = opt_hps['learning_rate']
            max_epochs = opt_hps['max_num_epochs']
            repeats    = 6
        else:
            if n_stats <= 5:
                hfs, nts, repeats = 16, 3, 5
            elif n_stats <= 15:
                hfs, nts, repeats = 32, 3, 6
            else:
                hfs, nts, repeats = 32, 4, 6
            n_train_eff = int(n_train * 0.85)
            batch_size  = int(np.clip(n_train_eff // 8, 32, 256))
            lr          = 5e-4
            max_epochs  = 500

        train_args = {
            'training_batch_size': batch_size,
            'learning_rate':       lr,
            'max_num_epochs':      max_epochs,
            'stop_after_epochs':   50,
            'clip_max_norm':       5.0,
            'validation_fraction': VAL_FRACTION,
        }

        nets = load_nde_sbi(
            engine='NPE', model='nsf',
            repeats=repeats, hidden_features=hfs, num_transforms=nts,
        )
        arch_str = (f'NSF  n_stats={n_stats}  ratio={ratio:.1f}  '
                    f'repeats={repeats}  hfs={hfs}  nts={nts}')

        runner = InferenceRunner.load(
            backend='sbi', engine='NPE',
            prior=BoxUniform(
                low =torch.tensor(PRIOR_LOW,  dtype=torch.float32, device=device),
                high=torch.tensor(PRIOR_HIGH, dtype=torch.float32, device=device),
            ),
            nets=nets,
            out_dir=Path(fpath(f'sbi_logs_{name}')),
            device=device,
            train_args=train_args,
        )
        posterior, _ = runner(loader)

        with open(fpath(f'ili_posterior_{name}.pkl'), 'wb') as f:
            pk.dump(posterior, f)

        msg = (f'[{name}] DONE --> {n_stats} stats, {n_train} sims, '
               f'{arch_str}, batch={batch_size}')
        return name, True, msg

    except Exception as e:
        import traceback
        return name, False, f'[{name}] FAILED: {traceback.format_exc()}'


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    os.makedirs(CACHE_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'The next round of samples to be run will be round {NEXT_ROUND}')
    print(f'FORCE_EQUAL_ARCH = {FORCE_EQUAL_ARCH}')
    if FORCE_EQUAL_ARCH:
        print(f'  Architecture: {EQUAL_ARCH}')
    print(f'Cl settings: LMIN={LMIN}  LMAX={LMAX}  N_ELL_BINS={N_ELL_BINS}  '
          f'N_SUMMARY={N_SUMMARY}')
    print(f'ELL_CENTRES: {np.round(ELL_CENTRES).astype(int).tolist()}')

    # --- x_obs ---
    print('Extracting reference run (x_obs)...')
    x_obs = extract_Cls(os.path.join(BASE_DIR, 'reference_run'))
    if x_obs is None:
        raise RuntimeError('No pkl files found in reference_run directory.')
    print(f'  x_obs shape: {x_obs.shape}  (expect {N_SUMMARY})')
    for i, (label, _, _) in enumerate(CL_SPECS):
        sl = slice(i * N_ELL_BINS, (i + 1) * N_ELL_BINS)
        print(f'    {label:8s}: mean={x_obs[sl].mean():.4e}  '
              f'range=[{x_obs[sl].min():.4e}, {x_obs[sl].max():.4e}]')
    np.save(os.path.join(WORK_DIR, 'x_obs.npy'), x_obs)

    # --- Training data ---
    x_train_path     = os.path.join(WORK_DIR, 'x_train_full.npy')
    theta_train_path = os.path.join(WORK_DIR, 'theta_train_full.npy')

    if os.path.exists(x_train_path) and os.path.exists(theta_train_path):
        print('\nLoading cached Cl training arrays...')
        x_train     = np.load(x_train_path)
        theta_train = np.load(theta_train_path)
    else:
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
                    v = extract_Cls(os.path.join(BASE_DIR, f'sample_{sid}'))
                    if v is not None:
                        np.save(cache_file, v)
                if v is not None:
                    theta_list.append([row['theta_ej_0'], row['nu_theta_ej_M']])
                    x_list.append(v)

        x_train     = np.array(x_list,     dtype=np.float32)
        theta_train = np.array(theta_list, dtype=np.float32)
        np.save(x_train_path,     x_train)
        np.save(theta_train_path, theta_train)

    print(f'Loaded {len(theta_train)} simulations. '
          f'x_train: {x_train.shape}, theta_train: {theta_train.shape}')

    # =========================================================================
    # PARALLEL TRAINING
    # =========================================================================
    n_cpus = min(N_STATISTICS, int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count())))
    print(f'\nTraining {N_STATISTICS} posteriors in parallel '
          f'({n_cpus} processes, each on CPU)...')

    if not FORCE_EQUAL_ARCH:
        print('\nLoading per-statistic Optuna hyperparameters...')

    worker_args = []
    for name, idx in STAT_MAP.items():
        blocks  = None if name in INDIVIDUAL_STATS else make_blocks(len(idx))
        opt_hps = None
        if not FORCE_EQUAL_ARCH:
            opt_hps = load_optuna_hyperparams(OPTUNA_STUDY_DIRS.get(name, ''))
            if opt_hps is None:
                print(f'  [{name}] No Optuna study found, using adaptive defaults.')
        worker_args.append(
            (name, idx, x_train, theta_train, x_obs, WORK_DIR,
             'cpu', blocks, opt_hps))

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_cpus) as pool:
        results = pool.map(train_one_statistic, worker_args)

    print('\n--- Training Summary ---')
    for name, success, msg in results:
        status = 'OK  ' if success else 'FAIL'
        print(f'  [{status}] {msg}')

    # =========================================================================
    # NEXT-ROUND ACTIVE LEARNING PROPOSAL
    # =========================================================================
    print(f'\nGenerating round {NEXT_ROUND} proposals from '
          f'{PROPOSAL_STAT} posterior...')

    proposal_pkl  = os.path.join(WORK_DIR, f'ili_posterior_{PROPOSAL_STAT}.pkl')
    proposal_xobs = os.path.join(WORK_DIR, f'xobs_{PROPOSAL_STAT}.npy')

    if not os.path.exists(proposal_pkl):
        raise FileNotFoundError(f'Proposal posterior not found: {proposal_pkl}.')

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

    out_csv = (f'/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/'
               f'round{NEXT_ROUND}_samples.csv')
    pd.DataFrame(next_theta, columns=PARAM_NAMES).to_csv(
        out_csv, index_label='sample_id')
    print(f'Saved {len(next_theta)} proposals to {out_csv}')
    for i, pname in enumerate(PARAM_NAMES):
        col = next_theta[:, i]
        print(f'  {pname}: mean={col.mean():.3f}, std={col.std():.3f}, '
              f'range=[{col.min():.3f}, {col.max():.3f}]')

    print('\nAll done. Generate contour plots and run the next round of samples!')
