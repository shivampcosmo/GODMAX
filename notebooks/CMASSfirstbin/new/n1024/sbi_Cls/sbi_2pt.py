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
from scipy.interpolate import interp1d
# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================
LTU_ILI_PATH = '/work/hdd/bdne/aacharya2/ltu-ili'
sys.path.append(LTU_ILI_PATH)

PASTING_DIR = '/work/hdd/bdne/aacharya2/GODMAX/notebooks/pasting'
if PASTING_DIR not in sys.path:
    sys.path.append(PASTING_DIR)

from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.validation import ValidationRunner
from ili.utils import load_nde_sbi

import pathlib
curr_path = pathlib.Path().absolute()
project_base = curr_path.parents[4]
abs_path_data = project_base / "data"
abs_path_src = project_base / "src"
abs_path_results = project_base / "results"
abs_path_params = project_base / "param_files"

for path in [curr_path, abs_path_data, abs_path_src, abs_path_results, abs_path_params]:
    sys.path.append(str(path))

from paste_backlight_utils import (
    get_project_paths, build_config, make_galaxy_map,
    compute_shot_noise_Cl, compute_Cl_ratio_in_bands,
    compute_Cl_gg_1h_2h, compute_hod_shot_noise_Cl,)

from base_class import base_class
from get_radial_profiles import Profiles
from get_Pkzs import get_Pkz
from get_Cls import get_Cl
BASE_DIR  = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/new/n1024'
WORK_DIR  = str(Path(__file__).parent.resolve())
CACHE_DIR = os.path.join(WORK_DIR, 'sample_vector_cache_cls')

CSV_FILES = [
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/lhs_samples.csv', 0),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round2_samples.csv', 500),
]

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(420)

# =============================================================================
# Cl SETTINGS
# =============================================================================
NSIDE       = 1024
LMIN        = 100
LMAX        = 1500
N_ELL_BINS  = 20
F_SKY       = 1.0
GAL_ZMIN = 0.3
GAL_ZMAX = 0.5

CL_SPECS = [
    ('gg',     'gal', 'gal',   'auto'),
    ('gy',     'gal', 'ymap',  'cross'),
    ('gtau',   'gal', 'tau',   'cross'),
    ('gkappa', 'gal', 'kappa', 'cross'),
]
MAP_KEYS = {
    'gal':   None,
    'ymap':  'map_ymap',
    'tau':   'map_tau',
    'kappa': 'map_kappa',
}
N_SPECS   = len(CL_SPECS)
N_SUMMARY = N_SPECS * N_ELL_BINS

# =============================================================================
# PRIOR AND NETWORK
# =============================================================================
PRIOR_LOW    = [1.0, -0.3]
PRIOR_HIGH   = [6.0,  0.0]
PARAM_LABELS = [r'$\theta_{ej,0}$', r'$\nu_{\theta_{ej}}^{M}$']
PARAM_NAMES  = ['theta_ej_0', 'nu_theta_ej_M']
VAL_FRACTION = 0.15

EQUAL_ARCH = {
    'hidden_features': 64,
    'num_transforms':  5,
    'learning_rate':   2e-4,
    'batch_size':      32,
    'max_num_epochs':  500,
    'repeats':         6,
}

STAT_MAP = {
    'gg':     list(range(0 * N_ELL_BINS, 1 * N_ELL_BINS)),
    'gy':     list(range(1 * N_ELL_BINS, 2 * N_ELL_BINS)),
    'gtau':   list(range(2 * N_ELL_BINS, 3 * N_ELL_BINS)),
    'gkappa': list(range(3 * N_ELL_BINS, 4 * N_ELL_BINS)),
    'JOINT':  list(range(N_SUMMARY)),
}
N_STATISTICS = len(STAT_MAP)

# =============================================================================
# ELL BINNING
# =============================================================================

def make_ell_bins(lmin=LMIN, lmax=LMAX, n_bins=N_ELL_BINS):
    edges   = np.logspace(np.log10(lmin), np.log10(lmax), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return edges.astype(int), centres

ELL_EDGES, ELL_CENTRES = make_ell_bins()

def bin_cl(cl_full, ell_edges=ELL_EDGES):
    ell    = np.arange(len(cl_full))
    n_bins = len(ell_edges) - 1
    bp     = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = ell_edges[i], ell_edges[i + 1]
        mask   = (ell >= lo) & (ell < hi)
        if mask.any():
            w     = 2 * ell[mask] + 1
            bp[i] = np.sum(w * cl_full[mask]) / np.sum(w)
    return bp

# =============================================================================
# THEORY Cl INITIALISATION
# FIX 1: halo_params_dict['zmin'] = 0.2 so the galaxy window [0.3, 0.5]
#         sits inside the halo z-grid, preventing 1/nbar^2 blowup in
#         Cl_gal_gal_tot_mat.
# =============================================================================

def init_theory_cls():

    paths = get_project_paths()
    (sim_params_dict, halo_params_dict, analysis_dict,
     other_params_dict, cosmo_jax, zarray_lens, nz_lens, gal_zrange) = build_config(
        paths['params'], paths['data'],
        nbar_comoving=1e-4, gal_zmin=0.3, gal_zmax=0.5)

    # FIX 1: push grid boundary below gal_zmin=0.3
    halo_params_dict['zmin'] = 0.2

    base_obj     = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
    profiles_obj = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict,
                            base_class_obj=base_obj)

    import jax.numpy as jnp
    M_halo_MIN = 10**12.75   # no-op since M_array starts at 10^13; keeps all bins
    mask_1d    = jnp.where(profiles_obj.M_array > M_halo_MIN, 1.0, 0.0)
    mask_2d    = jnp.tile(mask_1d, (halo_params_dict['nz'], 1))
    profiles_obj.Ncen_mat = profiles_obj.Ncen_mat * mask_2d
    profiles_obj.Nsat_mat = profiles_obj.Nsat_mat * mask_2d

    Pkz = get_Pkz(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict,
                  Profiles_obj=profiles_obj)
    Cls = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict,
                 Pkz_obj=Pkz)
    return Cls

# =============================================================================
# THEORY Cl COVARIANCE  (Gaussian approximation)
# FIX 2: read all four spectra directly from Cls_test attributes using
#         Cl_gal_gal_tot_mat[:, 0, 0] — safe now that zmin=0.2 is set.
# =============================================================================

def compute_gaussian_cov(Cls, f_sky=F_SKY):
    ell_th = np.array(Cls.ell_array).ravel()

    Cl_th = {
        'gg':     np.abs(np.array(Cls.Cl_gal_gal_tot_mat[:, 0, 0])),
        'gy':     np.abs(np.array(Cls.Cl_gal_y_tot_mat[:, 0])),
        'gtau':   np.abs(np.array(Cls.Cl_gal_tau_tot_mat[:, 0])),
        'gkappa': np.abs(np.array(Cls.Cl_gal_kappa_tot_mat[:, 0]).ravel()),
    }

    for k, v in Cl_th.items():
        print(f'  theory {k}: shape={v.shape}  '
              f'range=[{v.min():.3e}, {v.max():.3e}]')

    delta_ell = np.diff(ELL_EDGES).astype(float)
    n_modes   = (2.0 * ELL_CENTRES + 1.0) * f_sky * delta_ell

    cov_diag = []
    for label, _, _, _ in CL_SPECS:
        cl_fn  = interp1d(ell_th, Cl_th[label],
                          bounds_error=False,
                          fill_value=(Cl_th[label][0], Cl_th[label][-1]))
        cl_bin = np.clip(cl_fn(ELL_CENTRES), 1e-40, None)

        cl_gg_fn  = interp1d(ell_th, Cl_th['gg'],
                             bounds_error=False,
                             fill_value=(Cl_th['gg'][0], Cl_th['gg'][-1]))
        cl_gg_bin = np.clip(cl_gg_fn(ELL_CENTRES), 1e-40, None)

        if label == 'gg':
            var = 2.0 * cl_bin**2 / n_modes
        else:
            var = (cl_gg_bin * cl_bin + cl_bin**2) / n_modes

        print(f'  [{label:8s}] var range = [{var.min():.3e}, {var.max():.3e}]')
        cov_diag.append(var)

    cov_diag = np.concatenate(cov_diag)
    inv_cov  = 1.0 / np.clip(cov_diag, 1e-40, None)
    return cov_diag, inv_cov

# =============================================================================
# SUMMARY STATISTIC EXTRACTION  (unchanged)
# =============================================================================

def extract_Cls(path, shot_noise_correct=True):
    pattern = os.path.join(path, '**', f'allmaps_sim_B12_nside{NSIDE}.pkl')
    files   = glob.glob(pattern, recursive=True)
    if not files:
        return None

    # Load maps exactly as the comparison script does
    mock_gals = None
    ymap  = np.zeros(12 * NSIDE**2, dtype=np.float64)
    kmap  = np.zeros(12 * NSIDE**2, dtype=np.float64)
    tmap  = np.zeros(12 * NSIDE**2, dtype=np.float64)

    for fpath in files:
        with open(fpath, 'rb') as h:
            data = pk.load(h)
        ymap += np.nan_to_num(data.get('map_ymap',  0))
        kmap += np.nan_to_num(data.get('map_kappa', 0))
        tmap += np.nan_to_num(data.get('map_tau',   0))
        if mock_gals is None:
            mock_gals = np.array(data['mock_gals_all'][0])

    if mock_gals is None or len(mock_gals) == 0:
        return None

    # Exactly as comparison script
    map_gal   = make_galaxy_map(mock_gals, NSIDE, zmin=GAL_ZMIN, zmax=GAL_ZMAX)
    delta_gal = map_gal / np.mean(map_gal)

    Cl_shot, n_gal, nbar_sr = compute_shot_noise_Cl(mock_gals, NSIDE, GAL_ZMIN, GAL_ZMAX)

    # Pixel window — used only for cross-spectra (once), not for gg
    pixwin = hp.pixwin(NSIDE, lmax=LMAX)
    pw     = np.ones(LMAX + 1)
    pw[:len(pixwin)] = pixwin
    pw = np.where(pw > 0, pw, 1.0)

    vec = []
    for label, key_a, key_b, spec_type in CL_SPECS:
        if key_a == 'gal':
            map_a = delta_gal
        else:
            map_a = np.nan_to_num(
                ymap if key_a == 'ymap' else
                tmap if key_a == 'tau'  else kmap)

        if key_b == 'gal':
            map_b = delta_gal
        else:
            map_b = np.nan_to_num(
                ymap if key_b == 'ymap' else
                tmap if key_b == 'tau'  else kmap)

        if spec_type == 'auto' and key_a == 'gal':
            cl_full = hp.anafast(delta_gal, lmax=LMAX)
            if shot_noise_correct:
                cl_full = cl_full - Cl_shot

        else:
            # Cross: anafast(delta_gal, map_x), divide by pixwin once
            cl_full = hp.anafast(map_a, map_b, lmax=LMAX)
            cl_full = cl_full / pw[:len(cl_full)]

        bp = bin_cl(cl_full, ELL_EDGES)
        vec.append(bp)

    return np.concatenate(vec).astype(np.float32)
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
# TRAINING WORKER
# =============================================================================

def train_one_statistic(args):
    torch.set_num_threads(1)
    (name, idx, x_train, theta_train, x_obs,
     cov_diag, work_dir, device) = args

    sys.path.append(LTU_ILI_PATH)
    from ili.dataloaders import StaticNumpyLoader
    from ili.inference   import InferenceRunner
    from ili.utils       import load_nde_sbi
    from sbi.utils       import BoxUniform

    def fpath(fname):
        return os.path.join(work_dir, fname)

    try:
        n_stats = len(idx)
        n_train = len(theta_train)

        xt_full = x_train[:, idx].astype(np.float32)
        xo      = x_obs[idx].astype(np.float32)

        x_mean = np.mean(xt_full, axis=0)
        x_mean[np.abs(x_mean) < 1e-30] = 1.0

        xt_norm = xt_full / x_mean
        xo_norm = xo      / x_mean

        if cov_diag is not None:
            cov_sel    = cov_diag[idx]
            cov_weight = 1.0 / np.sqrt(np.clip(cov_sel, 1e-40, None))
            cov_weight = cov_weight / cov_weight.mean()
            xt_norm    = xt_norm * cov_weight
            xo_norm    = xo_norm * cov_weight
        else:
            cov_weight = np.ones(n_stats, dtype=np.float32)

        print(f'{name:12s}  n_stats={n_stats}  n_train={n_train}  '
              f'mean_xmean={np.abs(x_mean).mean():.4e}', flush=True)

        frac_below = (xo < xt_full.min(axis=0)).mean()
        frac_above = (xo > xt_full.max(axis=0)).mean()
        print(f'{name:12s}  frac_below={frac_below:.2f}  '
              f'frac_above={frac_above:.2f}', flush=True)

        np.save(fpath(f'scaler_cls_{name}_mean.npy'),      x_mean)
        np.save(fpath(f'scaler_cls_{name}_covweight.npy'), cov_weight)
        np.save(fpath(f'x_cls_{name}.npy'),                xt_norm)
        np.save(fpath(f'xobs_cls_{name}.npy'),             xo_norm)
        np.save(fpath(f'theta_train_cls_{name}.npy'),      theta_train)

        loader = StaticNumpyLoader(
            in_dir=work_dir,
            x_file=f'x_cls_{name}.npy',
            theta_file=f'theta_train_cls_{name}.npy',
            xobs_file=f'xobs_cls_{name}.npy',
        )

        hfs        = EQUAL_ARCH['hidden_features']
        nts        = EQUAL_ARCH['num_transforms']
        batch_size = EQUAL_ARCH['batch_size']
        lr         = EQUAL_ARCH['learning_rate']
        max_epochs = EQUAL_ARCH['max_num_epochs']
        repeats    = EQUAL_ARCH['repeats']

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
            repeats=repeats,
            hidden_features=hfs,
            num_transforms=nts,
        )

        runner = InferenceRunner.load(
            backend='sbi', engine='NPE',
            prior=BoxUniform(
                low =torch.tensor(PRIOR_LOW,  dtype=torch.float32, device=device),
                high=torch.tensor(PRIOR_HIGH, dtype=torch.float32, device=device),
            ),
            nets=nets,
            out_dir=Path(fpath(f'sbi_logs_cls_{name}')),
            device=device,
            train_args=train_args,
        )
        posterior, _ = runner(loader)

        with open(fpath(f'ili_posterior_cls_{name}.pkl'), 'wb') as f:
            pk.dump(posterior, f)

        msg = (f'[{name}] DONE  n_stats={n_stats}  n_train={n_train}  '
               f'hfs={hfs}  nts={nts}  repeats={repeats}')
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
    print(f'Cl settings: LMIN={LMIN}  LMAX={LMAX}  N_ELL_BINS={N_ELL_BINS}  '
          f'N_SUMMARY={N_SUMMARY}  F_SKY={F_SKY}')
    print(f'ELL_CENTRES: {np.round(ELL_CENTRES).astype(int).tolist()}')

    # ------------------------------------------------------------------ #
    # 1.  x_obs                                                           #
    # ------------------------------------------------------------------ #
    x_obs_path = os.path.join(WORK_DIR, 'x_obs_cls.npy')
    if os.path.exists(x_obs_path):
        print('\nLoading cached x_obs_cls...')
        x_obs = np.load(x_obs_path)
    else:
        print('\nExtracting Cl from reference run...')
        x_obs = extract_Cls(os.path.join(BASE_DIR, 'reference_run'))
        if x_obs is None:
            raise RuntimeError('No pkl files found in reference_run.')
        np.save(x_obs_path, x_obs)
    print(f'  x_obs shape: {x_obs.shape}  (expect {N_SUMMARY})')
    print(f'  x_obs per-spec means:')
    print(f"\n  gg Cl_raw range  : {x_obs[:N_ELL_BINS].min():.4e}  {x_obs[:N_ELL_BINS].max():.4e}")
    # If all negative, shot noise is dominating -- check c_shot_gg in extract_Cls
    for i, (label, _, _, _) in enumerate(CL_SPECS):
        sl = slice(i * N_ELL_BINS, (i + 1) * N_ELL_BINS)
        print(f'    {label:8s}: {x_obs[sl].mean():.4e}  '
              f'range=[{x_obs[sl].min():.4e}, {x_obs[sl].max():.4e}]')

    # ------------------------------------------------------------------ #
    # 2.  Theory Cls and Gaussian covariance                              #
    # ------------------------------------------------------------------ #
    cov_path = os.path.join(WORK_DIR, 'gaussian_cov_diag_cls.npy')
    if os.path.exists(cov_path):
        cov_diag = np.load(cov_path)
        if not np.all(np.isfinite(cov_diag)):
            print('Cached cov_diag is invalid — recomputing.')
            os.remove(cov_path)
            cov_diag = None
        else:
            print(f'\nLoading cached Gaussian covariance diagonal...')
    else:
        cov_diag = None

    if cov_diag is None:
        print('\nComputing theory Cls and Gaussian covariance...')
        Cls_fid     = init_theory_cls()
        cov_diag, _ = compute_gaussian_cov(Cls_fid, f_sky=F_SKY)
        np.save(cov_path, cov_diag)
        print(f'  Saved to {cov_path}')

    print(f'  cov_diag shape: {cov_diag.shape}')
    print(f'  noise std per spec:')
    for i, (label, _, _, _) in enumerate(CL_SPECS):
        sl = slice(i * N_ELL_BINS, (i + 1) * N_ELL_BINS)
        print(f'    {label:8s}: sqrt(cov) range = '
              f'[{np.sqrt(cov_diag[sl]).min():.4e}, '
              f'{np.sqrt(cov_diag[sl]).max():.4e}]')

    # ------------------------------------------------------------------ #
    # 3.  Training data                                                   #
    # ------------------------------------------------------------------ #
    x_train_path     = os.path.join(WORK_DIR, 'x_train_cls.npy')
    theta_train_path = os.path.join(WORK_DIR, 'theta_train_cls.npy')

    if os.path.exists(x_train_path) and os.path.exists(theta_train_path):
        print('\nLoading cached Cl training arrays...')
        x_train     = np.load(x_train_path)
        theta_train = np.load(theta_train_path)
    else:
        print('\nExtracting Cl training data from simulations...')
        theta_list, x_list = [], []
        for csv_path, offset in CSV_FILES:
            df = pd.read_csv(csv_path)
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc=f'Loading {os.path.basename(csv_path)}'):
                sid        = int(row['sample_id']) + offset
                cache_file = os.path.join(CACHE_DIR, f'x_cls_sample_{sid}.npy')
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

    print(f'  Loaded {len(theta_train)} simulations.')
    print(f'  x_train: {x_train.shape},  theta_train: {theta_train.shape}')

    print('\n  x_obs vs simulation range per spectrum:')
    for i, (label, _, _, _) in enumerate(CL_SPECS):
        sl         = slice(i * N_ELL_BINS, (i + 1) * N_ELL_BINS)
        xo_sl      = x_obs[sl]
        xt_sl      = x_train[:, sl]
        frac_below = (xo_sl < xt_sl.min(axis=0)).mean()
        frac_above = (xo_sl > xt_sl.max(axis=0)).mean()
        print(f'    {label:8s}: below={frac_below:.1%}  above={frac_above:.1%}')

    # ------------------------------------------------------------------ #
    # 4.  Parallel training                                               #
    # ------------------------------------------------------------------ #
    n_cpus = min(N_STATISTICS,
                 int(os.environ.get('SLURM_CPUS_PER_TASK',
                                    mp.cpu_count())))
    print(f'\nTraining {N_STATISTICS} posteriors in parallel '
          f'({n_cpus} processes)...')

    worker_args = [
        (name, idx, x_train, theta_train, x_obs,
         cov_diag, WORK_DIR, 'cpu')
        for name, idx in STAT_MAP.items()
    ]

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_cpus) as pool:
        results = pool.map(train_one_statistic, worker_args)

    print('\n--- Training Summary ---')
    for name, success, msg in results:
        status = 'OK  ' if success else 'FAIL'
        print(f'  [{status}] {msg}')

    # ------------------------------------------------------------------ #
    # 5.  Sample posteriors and make plots                                #
    # ------------------------------------------------------------------ #
    print('\nSampling posteriors...')

    fig_corner, axes_corner = plt.subplots(
        len(STAT_MAP), 2,
        figsize=(10, 4 * len(STAT_MAP)))
    if len(STAT_MAP) == 1:
        axes_corner = axes_corner[np.newaxis, :]

    summary_rows = []

    for stat_idx, (name, idx) in enumerate(STAT_MAP.items()):
        posterior_path = os.path.join(WORK_DIR, f'ili_posterior_cls_{name}.pkl')
        if not os.path.exists(posterior_path):
            print(f'  [{name}] No posterior file found, skipping.')
            continue

        with open(posterior_path, 'rb') as f:
            posterior = pk.load(f)

        x_mean     = np.load(os.path.join(WORK_DIR, f'scaler_cls_{name}_mean.npy'))
        cov_weight = np.load(os.path.join(WORK_DIR, f'scaler_cls_{name}_covweight.npy'))

        xo_norm = (x_obs[idx] / x_mean) * cov_weight

        samples = sample_ensemble_direct(posterior, xo_norm, n_samples=2000)
        if samples is None:
            print(f'  [{name}] Sampling failed, skipping.')
            continue

        mean_theta = samples[:, 0].mean()
        std_theta  = samples[:, 0].std()
        mean_nu    = samples[:, 1].mean()
        std_nu     = samples[:, 1].std()

        summary_rows.append({
            'statistic':          name,
            'theta_ej_0_mean':    mean_theta,
            'theta_ej_0_std':     std_theta,
            'nu_theta_ej_M_mean': mean_nu,
            'nu_theta_ej_M_std':  std_nu,
        })

        print(f'  [{name}]  '
              f'theta_ej_0 = {mean_theta:.4f} +/- {std_theta:.4f}  '
              f'nu_theta_ej_M = {mean_nu:.4f} +/- {std_nu:.4f}')

        ax0, ax1 = axes_corner[stat_idx]
        ax0.hist(samples[:, 0], bins=40, color='steelblue',
                 density=True, alpha=0.7)
        ax0.set_xlabel(PARAM_LABELS[0], fontsize=11)
        ax0.set_ylabel('P', fontsize=11)
        ax0.set_title(f'{name} - {PARAM_NAMES[0]}')
        ax0.set_xlim(PRIOR_LOW[0], PRIOR_HIGH[0])

        ax1.hist(samples[:, 1], bins=40, color='darkorange',
                 density=True, alpha=0.7)
        ax1.set_xlabel(PARAM_LABELS[1], fontsize=11)
        ax1.set_ylabel('P', fontsize=11)
        ax1.set_title(f'{name} - {PARAM_NAMES[1]}')
        ax1.set_xlim(PRIOR_LOW[1], PRIOR_HIGH[1])

    fig_corner.suptitle('SBI posteriors - Cl bandpowers', fontsize=14)
    plt.tight_layout()
    corner_path = os.path.join(WORK_DIR, 'posteriors_cls_all_stats.png')
    plt.savefig(corner_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'\nPosterior panel plot saved to {corner_path}')

    # ------------------------------------------------------------------ #
    # 6.  Overlay plot                                                    #
    # ------------------------------------------------------------------ #
    fig_ov, axes_ov = plt.subplots(1, 2, figsize=(14, 5))
    cmap   = plt.get_cmap('tab10')
    colors = [cmap(i / max(len(STAT_MAP) - 1, 1))
              for i in range(len(STAT_MAP))]

    for stat_idx, (name, idx) in enumerate(STAT_MAP.items()):
        posterior_path = os.path.join(WORK_DIR, f'ili_posterior_cls_{name}.pkl')
        if not os.path.exists(posterior_path):
            continue

        with open(posterior_path, 'rb') as f:
            posterior = pk.load(f)

        x_mean     = np.load(os.path.join(WORK_DIR, f'scaler_cls_{name}_mean.npy'))
        cov_weight = np.load(os.path.join(WORK_DIR, f'scaler_cls_{name}_covweight.npy'))
        xo_norm    = (x_obs[idx] / x_mean) * cov_weight

        samples = sample_ensemble_direct(posterior, xo_norm, n_samples=2000)
        if samples is None:
            continue

        color = colors[stat_idx]
        for pi, ax in enumerate(axes_ov):
            lo, hi = PRIOR_LOW[pi], PRIOR_HIGH[pi]
            xs     = np.linspace(lo, hi, 300)
            from scipy.stats import gaussian_kde
            kde     = gaussian_kde(samples[:, pi])
            ys      = kde(xs)
            ys_norm = ys / (ys.max() + 1e-30)
            ax.plot(xs, ys_norm, color=color, lw=1.8, label=name)
            ax.set_xlabel(PARAM_LABELS[pi], fontsize=13)
            ax.set_ylabel('Normalised P',   fontsize=12)
            ax.set_xlim(lo, hi)

    for ax in axes_ov:
        ax.legend(fontsize=8, ncol=2)
    fig_ov.suptitle('SBI Cl posteriors - all statistics', fontsize=14)
    plt.tight_layout()
    ov_path = os.path.join(WORK_DIR, 'posteriors_cls_overlay.png')
    plt.savefig(ov_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Overlay plot saved to {ov_path}')

    # ------------------------------------------------------------------ #
    # 7.  Summary CSV                                                     #
    # ------------------------------------------------------------------ #
    if summary_rows:
        summary_df   = pd.DataFrame(summary_rows)
        summary_path = os.path.join(WORK_DIR, 'posterior_summary_cls.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f'\nSummary CSV saved to {summary_path}')
        print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 8.  Observed Cl diagnostic plot                                     #
    # ------------------------------------------------------------------ #
    fig_cl, axes_cl = plt.subplots(
        1, N_SPECS, figsize=(5 * N_SPECS, 4), squeeze=False)

    for i, (label, _, _, _) in enumerate(CL_SPECS):
        ax = axes_cl[0, i]
        sl = slice(i * N_ELL_BINS, (i + 1) * N_ELL_BINS)
        xo = x_obs[sl]
        xt = x_train[:, sl]
        cv = np.sqrt(cov_diag[sl])

        for j in range(min(50, len(xt))):
            ax.plot(ELL_CENTRES, np.abs(xt[j]),
                    color='grey', alpha=0.15, lw=0.7)

        ax.errorbar(ELL_CENTRES, np.abs(xo), yerr=cv,
                    fmt='o', color='steelblue', ms=4,
                    label='x_obs +/- Gaussian sigma')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$',                    fontsize=12)
        ax.set_ylabel(rf'$C_\ell^{{{label}}}$',    fontsize=12)
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig_cl.suptitle('Observed Cl vs training scatter', fontsize=13)
    plt.tight_layout()
    cl_diag_path = os.path.join(WORK_DIR, 'cl_diagnostic.png')
    plt.savefig(cl_diag_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Cl diagnostic plot saved to {cl_diag_path}')

    print('\nAll done.')
