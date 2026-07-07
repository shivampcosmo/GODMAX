import os
import sys
import threading
import torch
import numpy as np
import healpy as hp
import pickle as pk
import glob
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import random
from scipy.interpolate import interp1d as _interp1d

# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================
LTU_ILI_PATH = '/work/hdd/bdne/aacharya2/ltu-ili'
sys.path.append(LTU_ILI_PATH)

from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.validation import ValidationRunner, PosteriorCoverage
from ili.utils import load_nde_sbi

BASE_DIR      = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/new/n1024'
WORK_DIR      = str(Path(__file__).parent.resolve())
OUTPUTS_DIR   = str(Path(WORK_DIR).parent / 'outputs')
NOISE_PATH    = os.path.join(OUTPUTS_DIR, 'sbi_noise_spectra.npz')

ADD_SURVEY_NOISE = True

_CACHE_SUFFIX = '_noisy' if ADD_SURVEY_NOISE else ''
CACHE_DIR     = os.path.join(WORK_DIR, f'sample_vector_cache_cls{_CACHE_SUFFIX}')
VAL_CACHE_DIR = os.path.join(WORK_DIR, f'validation_vector_cache_cls{_CACHE_SUFFIX}')

VALIDATION_CSV = '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/validation_samples.csv'


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(111)

# Third element: True = prior round, False = proposal round.
CSV_FILES = [
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/lhs_samples.csv',    0,    True),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round2_samples.csv', 500,  False),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round3_samples.csv', 700,  False),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round4_samples.csv', 900,  False),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round5_samples.csv', 1100, False),
]
NEXT_ROUND = len(CSV_FILES) + 1

NSIDE      = 1024
LMIN       = 100
LMAX       = 1500
N_ELL_BINS = 20

PRIOR_LOW    = [1.0, -0.3]
PRIOR_HIGH   = [6.0,  0.0]
PARAM_LABELS = [r'$$\theta_{ej,0}$$', r'$${\nu_{\theta_{ej}}}^{M}$$']
PARAM_NAMES  = ['theta_ej_0', 'nu_theta_ej_M']
PROPOSAL_STAT = 'JOINT'
VAL_FRACTION  = 0.10

PCA_VARIANCE_THRESHOLD = 0.99

# =============================================================================
# ELL BINNING
# =============================================================================

def make_ell_bins(lmin=LMIN, lmax=LMAX, n_bins=N_ELL_BINS):
    edges = np.unique(
        np.logspace(np.log10(lmin), np.log10(lmax), n_bins + 1).astype(int)
    )
    if len(edges) != n_bins + 1:
        print(f'[WARN] make_ell_bins: requested {n_bins+1} edges, '
              f'got {len(edges)} unique after int cast. '
              f'Effective bins: {len(edges)-1}')
    centres = 0.5 * (edges[:-1] + edges[1:]).astype(float)
    return edges, centres

ELL_EDGES, ELL_CENTRES = make_ell_bins()
N_ELL_BINS_ACTUAL = len(ELL_EDGES) - 1

_ELL_FULL  = np.arange(LMAX + 1)
_SCALE     = _ELL_FULL * (_ELL_FULL + 1) / (2.0 * np.pi)
_W         = 2 * _ELL_FULL + 1
_BIN_MASKS = [
    (_ELL_FULL >= ELL_EDGES[i]) & (_ELL_FULL < ELL_EDGES[i + 1])
    for i in range(N_ELL_BINS_ACTUAL)
]
_BIN_WSUM = np.array([
    float(_W[m].sum()) if _W[m].sum() > 0 else 1.0
    for m in _BIN_MASKS
])

def bin_cl(cl_full):
    cl_scaled = cl_full[:LMAX + 1] * _SCALE
    return np.array([
        np.dot(_W[m], cl_scaled[m]) / _BIN_WSUM[i]
        if _BIN_MASKS[i].any() else 0.0
        for i, m in enumerate(_BIN_MASKS)
    ], dtype=np.float32)

# =============================================================================
# CL SPECS AND STAT MAP
# =============================================================================

CL_SPECS = [
    ('g2y',     'gal_sq', 'ymap'),
    ('g2tau',   'gal_sq', 'tau'),
    ('g2kappa', 'gal_sq', 'kappa'),
    ('gy',      'gal',    'ymap'),
    ('gtau',    'gal',    'tau'),
    ('gkappa',  'gal',    'kappa'),
]
N_SPECS   = len(CL_SPECS)
N_SUMMARY = N_SPECS * N_ELL_BINS_ACTUAL

_s = {label: list(range(i * N_ELL_BINS_ACTUAL, (i + 1) * N_ELL_BINS_ACTUAL))
      for i, (label, _, _) in enumerate(CL_SPECS)}

STAT_MAP = {
    'g2y':         _s['g2y'],
    'g2tau':       _s['g2tau'],
    'g2kappa':     _s['g2kappa'],
    'gy':          _s['gy'],
    'gtau':        _s['gtau'],
    'gkappa':      _s['gkappa'],
    'y_total':     _s['g2y']     + _s['gy'],
    'tau_total':   _s['g2tau']   + _s['gtau'],
    'kappa_total': _s['g2kappa'] + _s['gkappa'],
    'all_3pt':     _s['g2y']     + _s['g2tau']  + _s['g2kappa'],
    'all_2pt':     _s['gy']      + _s['gtau']   + _s['gkappa'],
    'JOINT':       list(range(N_SUMMARY)),
}
N_STATISTICS = len(STAT_MAP)

INDIVIDUAL_STATS = {'gy', 'gtau', 'gkappa', 'g2y', 'g2tau', 'g2kappa'}
BLOCK_SIZE = N_ELL_BINS_ACTUAL

def make_blocks(n_features):
    return [list(range(i, min(i + BLOCK_SIZE, n_features)))
            for i in range(0, n_features, BLOCK_SIZE)]

# =============================================================================
# ARCHITECTURE
# =============================================================================

FORCE_EQUAL_ARCH = False
EQUAL_ARCH = {
    'hidden_features': 64,
    'num_components':  5,
    'learning_rate':   5e-4,
    'batch_size':      64,
    'max_num_epochs':  200,
    'repeats':         6,
}

ILIAS_BASE = '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/sbi_Cls/ilias_results'
OPTUNA_STUDY_DIRS = {name: os.path.join(ILIAS_BASE, name) for name in STAT_MAP}

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
          f'hfs={mcfg["hidden_features"]}  num_components={mcfg["num_components"]}  '
          f'lr={mcfg["learning_rate"]:.2e}  '
          f'batch={mcfg["batch_size"]}  epochs={mcfg["max_epochs"]}')
    return {
        'hidden_features':  mcfg['hidden_features'],
        'num_components':   mcfg['num_components'],
        'learning_rate':    mcfg['learning_rate'],
        'batch_size':       mcfg['batch_size'],
        'max_num_epochs':   mcfg['max_epochs'],
    }

# =============================================================================
# NOISE UTILITIES
# =============================================================================

_NOISE_PKG_CACHE = None

def _get_noise_pkg():
    global _NOISE_PKG_CACHE
    if _NOISE_PKG_CACHE is not None:
        return _NOISE_PKG_CACHE
    if not os.path.exists(NOISE_PATH):
        raise FileNotFoundError(
            f'Noise package not found: {NOISE_PATH}\n'
            'Run:  python gen_fid_noise_spectra.py'
        )
    data    = np.load(NOISE_PATH)
    ell_th  = data['ell'].astype(float)
    ell_int = np.arange(LMAX + 1, dtype=float)
    pkg = {}
    for field in ('nl_yy', 'nl_tautau', 'nl_kk'):
        vals     = data[field].astype(float)
        interped = _interp1d(
            ell_th, vals,
            bounds_error=False,
            fill_value=(vals[0], vals[-1]),
        )(ell_int)
        interped     = np.clip(interped, 0.0, np.inf)
        interped[:2] = 0.0
        pkg[field]   = interped
    _NOISE_PKG_CACHE = pkg
    print(f'[noise] Loaded noise spectra from {NOISE_PATH}')
    print(f'  nl_yy     range (ell>={LMIN}): '
          f'[{pkg["nl_yy"][LMIN:].min():.3e}, {pkg["nl_yy"][LMIN:].max():.3e}]')
    print(f'  nl_tautau range (ell>={LMIN}): '
          f'[{pkg["nl_tautau"][LMIN:].min():.3e}, {pkg["nl_tautau"][LMIN:].max():.3e}]')
    print(f'  nl_kk     range (ell>={LMIN}): '
          f'[{pkg["nl_kk"][LMIN:].min():.3e}, {pkg["nl_kk"][LMIN:].max():.3e}]')
    return _NOISE_PKG_CACHE


def _path_to_seed(path: str) -> int:
    import hashlib
    return int(hashlib.md5(path.encode()).hexdigest(), 16) % (2**31 - 1)


def _add_noise_to_maps(ymap, tmap, kmap, path: str) -> tuple:
    pkg  = _get_noise_pkg()
    seed = _path_to_seed(path)
    rng  = np.random.default_rng(seed)
    def _draw(nl_key):
        s = int(rng.integers(0, 2**31 - 1))
        np.random.seed(s)
        return hp.synfast(pkg[nl_key], nside=NSIDE, lmax=LMAX,
                          new=True, verbose=False).astype(np.float64)
    return ymap + _draw('nl_yy'), tmap + _draw('nl_tautau'), kmap + _draw('nl_kk')

# =============================================================================
# SAMPLING UTILITIES
# =============================================================================

def _sample_member_thread(member, x_t, n_samples, result, exception):
    try:
        # Move x_t to the same device as the member posterior
        x_t = torch.as_tensor(x_t, dtype=torch.float32).to(member._device)
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

def extract_Cls(path, add_noise=ADD_SURVEY_NOISE):
    pattern = os.path.join(path, '**', f'allmaps_sim_B12_nside{NSIDE}.pkl')
    files   = glob.glob(pattern, recursive=True)
    if not files:
        return None

    ymap = np.zeros(12 * NSIDE**2, dtype=np.float64)
    gmap = np.zeros(12 * NSIDE**2, dtype=np.float32)
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
                gmap += np.bincount(pix, minlength=12 * NSIDE**2).astype(np.float32)

    mean_g = float(np.mean(gmap))
    if mean_g <= 0:
        return None

    npix_filled = int(np.sum(gmap > 0))
    fsky        = npix_filled / (12 * NSIDE**2)

    if add_noise:
        ymap, tmap, kmap = _add_noise_to_maps(ymap, tmap, kmap, path)

    delta_gal  = gmap.astype(np.float64) / mean_g - 1.0
    delta_sq   = delta_gal ** 2
    delta_sq  -= np.mean(delta_sq)

    pixwin = hp.pixwin(NSIDE)
    pw1    = pixwin[:LMAX + 1].copy()
    pw1    = np.where(pw1 > 0, pw1, 1.0)
    pw2    = pw1 ** 2

    field_map = {
        'gal':    (delta_gal,           pw1),
        'gal_sq': (delta_sq,            pw2),
        'ymap':   (np.nan_to_num(ymap), None),
        'tau':    (np.nan_to_num(tmap), None),
        'kappa':  (np.nan_to_num(kmap), None),
    }

    vec = []
    for label, field_a, field_b in CL_SPECS:
        map_a, pw_a = field_map[field_a]
        map_b, pw_b = field_map[field_b]

        cl_full = hp.anafast(map_a, map_b, lmax=LMAX)
        assert len(cl_full) == LMAX + 1, \
            f'anafast returned {len(cl_full)} elements, expected {LMAX+1}'

        if pw_a is not None:
            cl_full = cl_full / pw_a
        if pw_b is not None:
            cl_full = cl_full / pw_b

        bp = bin_cl(cl_full)

        if label.startswith('g2'):
            neg_frac = float((bp < 0).mean())
            if neg_frac > 0.3:
                print(f'  [WARN] {label}: {neg_frac:.0%} of bins negative '
                      f'(fsky={fsky:.3f}) — may indicate low S/N')
        vec.append(bp)

    return np.concatenate(vec).astype(np.float32)

# =============================================================================
# PCA HELPER
# =============================================================================

def fit_pca(x_norm, variance_threshold=PCA_VARIANCE_THRESHOLD):
    from sklearn.decomposition import PCA
    n_samples, n_features = x_norm.shape
    max_comp = min(n_samples - 1, n_features)
    pca = PCA(n_components=max_comp, svd_solver='full')
    pca.fit(x_norm)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_comp = max(1, min(n_comp, max_comp))
    print(f'  [PCA] kept {n_comp}/{max_comp} components '
          f'({cumvar[n_comp-1]*100:.1f}% variance)', flush=True)
    return pca, n_comp

# =============================================================================
# SEQUENTIAL MULTIROUND TRAINING  — one statistic, via ltu-ili SBIRunner
# =============================================================================

def train_one_statistic_gpu(name, idx, x_rounds, theta_rounds, x_obs,
                             work_dir, device, blocks, opt_hps):
    """
    Train one MDN ensemble posterior using ltu-ili's SBIRunner._train_round()
    called once per round, accumulating data and updating the proposal after
    each round — mirroring how the active learning rounds were actually generated.

    Round 1 (prior round):   proposal = BoxUniform prior
    Round R (R > 1):         proposal = EnsemblePosterior from rounds 1..R-1
    """
    import joblib
    from sbi.utils import BoxUniform
    from ili.inference.runner_sbi import SBIRunner

    sbi_device = 'cuda' if device.startswith('cuda') else 'cpu'

    def fpath(fname):
        return os.path.join(work_dir, fname)

    try:
        n_stats  = len(idx)
        n_rounds = len(x_rounds)
        n_train  = sum(len(t) for t in theta_rounds)

        # ── z-score using round-1 (prior) data only ───────────────────────────
        x_r1   = x_rounds[0][:, idx].astype(np.float32)
        xo_raw = x_obs[idx].astype(np.float32)

        if blocks is None:
            x_mean = np.mean(x_r1, axis=0)
            x_std  = np.std( x_r1, axis=0)
            x_std[x_std < 1e-10] = 1.0
        else:
            x_mean = np.empty(n_stats, dtype=np.float32)
            x_std  = np.empty(n_stats, dtype=np.float32)
            for blk in blocks:
                blk = np.asarray(blk)
                m = np.mean(x_r1[:, blk], axis=0)
                s = np.std( x_r1[:, blk], axis=0)
                s[s < 1e-10] = 1.0
                x_mean[blk] = m
                x_std[blk]  = s

        def normalise(x_slice):
            if blocks is None:
                return ((x_slice - x_mean) / x_std).astype(np.float32)
            out = np.empty_like(x_slice)
            for blk in blocks:
                blk = np.asarray(blk)
                out[:, blk] = (x_slice[:, blk] - x_mean[blk]) / x_std[blk]
            return out.astype(np.float32)

        np.save(fpath(f'scaler_{name}_mean.npy'), x_mean)
        np.save(fpath(f'scaler_{name}_std.npy'),  x_std)

        # ── PCA fit on round-1 normalised data only ───────────────────────────
        x_r1_norm = normalise(x_r1)
        pca, n_comp_var = fit_pca(x_r1_norm)

        # signal-based component selection using nu parameter
        x_r1_pca  = pca.transform(x_r1_norm)
        r2_nu     = np.array([
            np.corrcoef(x_r1_pca[:, i], theta_rounds[0][:, 1])[0, 1] ** 2
            for i in range(pca.n_components_)
        ])
        cumr2_nu  = np.cumsum(r2_nu)
        n_comp_nu = int(np.searchsorted(cumr2_nu, 0.80 * cumr2_nu[-1]) + 1)
        n_comp    = max(n_comp_var, n_comp_nu)
        print(f'  [{name}] PCA: variance-based={n_comp_var}  '
              f'nu-signal-based={n_comp_nu}  final={n_comp}', flush=True)

        joblib.dump(pca, fpath(f'pca_{name}.pkl'))
        np.save(fpath(f'pca_{name}_n_comp.npy'), np.array(n_comp))

        def compress(x_full_slice):
            """Normalise then PCA-compress a raw Cl block (n, N_SUMMARY)."""
            return pca.transform(
                normalise(x_full_slice[:, idx])
            )[:, :n_comp].astype(np.float32)

        # compressed observed data vector
        xo_norm = normalise(xo_raw.reshape(1, -1))
        xo_comp = pca.transform(xo_norm)[:, :n_comp][0].astype(np.float32)
        np.save(fpath(f'xobs_{name}.npy'), xo_comp)

        xo_tensor = torch.tensor(xo_comp, dtype=torch.float32).to(sbi_device)

        # ── Architecture ──────────────────────────────────────────────────────
        if FORCE_EQUAL_ARCH:
            hfs            = EQUAL_ARCH['hidden_features']
            num_components = EQUAL_ARCH['num_components']
            batch_size     = EQUAL_ARCH['batch_size']
            lr             = EQUAL_ARCH['learning_rate']
            max_epochs     = EQUAL_ARCH['max_num_epochs']
            repeats        = EQUAL_ARCH['repeats']
        elif opt_hps is not None:
            hfs            = opt_hps['hidden_features']
            num_components = opt_hps['num_components']
            batch_size     = opt_hps['batch_size']
            lr             = opt_hps['learning_rate']
            max_epochs     = opt_hps['max_num_epochs']
            repeats        = 6
        else:
            hfs            = 64
            num_components = EQUAL_ARCH['num_components']
            n_train_eff    = int(n_train * (1.0 - VAL_FRACTION))
            batch_size     = int(np.clip(n_train_eff // 8, 32, 256))
            lr             = 5e-4
            max_epochs     = 500
            repeats        = 6

        print(f'  [{name}] hfs={hfs}  num_components={num_components}  '
              f'lr={lr:.2e}  batch={batch_size}  epochs={max_epochs}  '
              f'repeats={repeats}  n_rounds={n_rounds}  '
              f'n_pca={n_comp}', flush=True)

        train_args = dict(
            training_batch_size=batch_size,
            learning_rate=lr,
            max_num_epochs=max_epochs,
            stop_after_epochs=50,
            clip_max_norm=5.0,
            validation_fraction=VAL_FRACTION,
        )

        # ── Prior ─────────────────────────────────────────────────────────────
        prior = BoxUniform(
            low =torch.tensor(PRIOR_LOW,  dtype=torch.float32).to(sbi_device),
            high=torch.tensor(PRIOR_HIGH, dtype=torch.float32).to(sbi_device),
        )

        # ── Build one SBIRunner per ensemble member, all sharing same nets ────
        # load_nde_sbi returns a list of nets; one per repeat
        nets = load_nde_sbi(
            engine='NPE', model='mdn',
            repeats=repeats,
            hidden_features=hfs,
            num_components=num_components,
        )

        # SBIRunner wraps the ensemble and exposes _train_round()
        runner = InferenceRunner.load(
            backend='sbi',
            engine='NPE',
            prior=prior,
            nets=nets,
            out_dir=Path(fpath(f'sbi_logs_{name}')),
            device=sbi_device,
            train_args=train_args,
        )
        # _setup_engine() creates one sbi SNPE inference object per net
        models = [runner._setup_engine(net) for net in runner.nets]

        # ── Multiround loop ───────────────────────────────────────────────────
        # proposal starts as prior; after each round it becomes the posterior
        # trained on all data seen so far — matching how rounds were generated.
        proposal        = prior
        posterior_ensemble = None

        for r_idx, (csv_path, offset, is_prior_round) in enumerate(CSV_FILES):
            # compress this round's data
            x_comp_r  = compress(x_rounds[r_idx])           # (n_r, n_comp)
            theta_r   = theta_rounds[r_idx].astype(np.float32)  # (n_r, N_PARAMS)

            x_tensor     = torch.tensor(x_comp_r, dtype=torch.float32).to(sbi_device)
            theta_tensor = torch.tensor(theta_r,  dtype=torch.float32).to(sbi_device)

            # append_simulations is called inside _train_round with the
            # correct proposal: prior for round 1, posterior for rounds 2+
            round_proposal = prior if is_prior_round else proposal

            print(f'  [{name}] round={r_idx+1}/{n_rounds}  '
                  f'n={len(theta_r)}  '
                  f'{"(prior)" if is_prior_round else "(proposal)"}',
                  flush=True)

            # _train_round accumulates data internally across calls via sbi's
            # append_simulations, then trains all ensemble members together
            posterior_ensemble, summaries = runner._train_round(
                models=models,
                x=x_tensor,
                theta=theta_tensor,
                proposal=round_proposal,
            )

            # set observed data on ensemble and use as proposal for next round
            proposal = posterior_ensemble.set_default_x(xo_tensor)

            print(f'  [{name}] round={r_idx+1} complete  '
                  f'val_loss={summaries[0]["best_validation_loss"][-1]:.4f}',
                  flush=True)

        # ── Save final posterior ───────────────────────────────────────────────
        with open(fpath(f'ili_posterior_{name}.pkl'), 'wb') as f:
            pk.dump(posterior_ensemble, f)

        msg = (f'[{name}] DONE  n_pca={n_comp}  n_train={n_train}  '
               f'n_rounds={n_rounds}  hfs={hfs}  '
               f'repeats={repeats}  device={sbi_device}')
        return name, True, msg

    except Exception:
        import traceback
        return name, False, f'[{name}] FAILED: {traceback.format_exc()}'


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-reload', action='store_true',
                        help='Ignore cached arrays and rebuild from CSVs.')
    args, _ = parser.parse_known_args()

    os.makedirs(CACHE_DIR,     exist_ok=True)
    os.makedirs(VAL_CACHE_DIR, exist_ok=True)

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    sbi_device = 'cuda' if device.startswith('cuda') else 'cpu'
    print(f'Device: {device}')
    print(f'Next round:            {NEXT_ROUND}')
    print(f'ADD_SURVEY_NOISE     = {ADD_SURVEY_NOISE}')
    print(f'FORCE_EQUAL_ARCH     = {FORCE_EQUAL_ARCH}')
    print(f'PCA_VARIANCE_THRESH  = {PCA_VARIANCE_THRESHOLD}')
    print(f'Cl settings: LMIN={LMIN}  LMAX={LMAX}  '
          f'N_ELL_BINS={N_ELL_BINS_ACTUAL}  N_SUMMARY={N_SUMMARY}')
    print(f'ELL_CENTRES: {np.round(ELL_CENTRES).astype(int).tolist()}')

    if ADD_SURVEY_NOISE:
        _get_noise_pkg()

    # ── x_obs ─────────────────────────────────────────────────────────────────
    print('\nExtracting reference run (x_obs)...')
    x_obs = extract_Cls(os.path.join(BASE_DIR, 'reference_run'))
    if x_obs is None:
        raise RuntimeError('No pkl files found in reference_run directory.')
    print(f'  x_obs shape: {x_obs.shape}  (expect {N_SUMMARY})')
    for i, (label, _, _) in enumerate(CL_SPECS):
        sl = slice(i * N_ELL_BINS_ACTUAL, (i + 1) * N_ELL_BINS_ACTUAL)
        print(f'    {label:8s}: mean={x_obs[sl].mean():.4e}  '
              f'range=[{x_obs[sl].min():.4e}, {x_obs[sl].max():.4e}]')
    np.save(os.path.join(WORK_DIR, 'x_obs.npy'), x_obs)

    # ── Training data — per-round ──────────────────────────────────────────────
    x_rounds_path     = os.path.join(WORK_DIR, f'x_rounds{_CACHE_SUFFIX}.npy')
    theta_rounds_path = os.path.join(WORK_DIR, 'theta_rounds.npy')

    need_rebuild = (
        args.force_reload
        or not os.path.exists(x_rounds_path)
        or not os.path.exists(theta_rounds_path)
    )

    if not need_rebuild:
        print('\nLoading cached per-round Cl arrays...')
        x_rounds     = list(np.load(x_rounds_path,     allow_pickle=True))
        theta_rounds = list(np.load(theta_rounds_path, allow_pickle=True))
    else:
        x_rounds, theta_rounds = [], []

        for csv_path, offset, is_prior_round in CSV_FILES:
            df = pd.read_csv(csv_path)
            x_r, theta_r = [], []

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
                    x_r.append(v)
                    theta_r.append([row['theta_ej_0'], row['nu_theta_ej_M']])

            if x_r:
                x_rounds.append(np.array(x_r,     dtype=np.float32))
                theta_rounds.append(np.array(theta_r, dtype=np.float32))
                label = 'prior' if is_prior_round else 'proposal'
                print(f'  Round {len(x_rounds)} ({label}): {len(x_r)} sims')

        np.save(x_rounds_path,     np.array(x_rounds,     dtype=object))
        np.save(theta_rounds_path, np.array(theta_rounds, dtype=object))

    total_sims = sum(len(t) for t in theta_rounds)
    print(f'Loaded {total_sims} simulations across {len(x_rounds)} rounds.')
    for r_idx, (xr, tr) in enumerate(zip(x_rounds, theta_rounds)):
        label = 'prior' if CSV_FILES[r_idx][2] else 'proposal'
        print(f'  Round {r_idx+1} ({label}): {len(tr)} sims')

    # ── Per-spectrum Fisher correlation diagnostic ─────────────────────────────
    x_train_all = np.concatenate(x_rounds,     axis=0)
    theta_all   = np.concatenate(theta_rounds, axis=0)
    print('\nPer-spectrum Fisher correlations with parameters:')
    for i, (label, _, _) in enumerate(CL_SPECS):
        sl       = slice(i * N_ELL_BINS_ACTUAL, (i + 1) * N_ELL_BINS_ACTUAL)
        cl_block = x_train_all[:, sl]
        for p, pname in enumerate(PARAM_NAMES):
            r = np.array([
                np.corrcoef(cl_block[:, j], theta_all[:, p])[0, 1]
                for j in range(N_ELL_BINS_ACTUAL)
            ])
            print(f'  {label:8s}  {pname}: '
                  f'max|r|={np.abs(r).max():.3f}  '
                  f'mean|r|={np.abs(r).mean():.3f}  '
                  f'best_ell={int(ELL_CENTRES[np.abs(r).argmax()])}')

    # ── Sequential GPU SBI training ────────────────────────────────────────────
    print(f'\nTraining {N_STATISTICS} posteriors sequentially on {device}...')

    results = []
    for name, idx in STAT_MAP.items():
        print(f'\n=== SBI training: {name} ===', flush=True)
        blocks  = None if name in INDIVIDUAL_STATS else make_blocks(len(idx))
        opt_hps = None
        if not FORCE_EQUAL_ARCH:
            opt_hps = load_optuna_hyperparams(OPTUNA_STUDY_DIRS.get(name, ''))
            if opt_hps is None:
                print(f'  [{name}] No Optuna study found, using adaptive defaults.')

        result = train_one_statistic_gpu(
            name, idx, x_rounds, theta_rounds, x_obs,
            WORK_DIR, device, blocks, opt_hps)
        results.append(result)

        if device == 'cuda':
            torch.cuda.empty_cache()

    print('\n--- Training Summary ---')
    for name, success, msg in results:
        status = 'OK  ' if success else 'FAIL'
        print(f'  [{status}] {msg}')

    # ═══════════════════════════════════════════════════════════════════════
# Validation  (runs entirely on CPU to avoid device-mismatch issues)
# ═══════════════════════════════════════════════════════════════════════
if not os.path.exists(VALIDATION_CSV):
    print(f'\n[Validation] {VALIDATION_CSV} not found — skipping.')
else:
    print('\nLoading held-out validation set...')
    os.makedirs(VAL_CACHE_DIR, exist_ok=True)

    val_df    = pd.read_csv(VALIDATION_CSV)
    theta_val = val_df[PARAM_NAMES].values.astype(np.float32)
    x_val_list, missing_val = [], []

    for i, row in tqdm(val_df.iterrows(), total=len(val_df),
                       desc='Extracting validation Cls'):
        sid        = int(row['sample_id'])
        cache_file = os.path.join(VAL_CACHE_DIR, f'x_val_{sid}.npy')
        if os.path.exists(cache_file):
            v = np.load(cache_file)
        else:
            v = extract_Cls(os.path.join(BASE_DIR, f'validation_{sid}'))
            if v is not None:
                np.save(cache_file, v)
        if v is not None:
            x_val_list.append(v)
        else:
            missing_val.append(sid)
            print(f'  [WARN] No data for validation_{sid}, skipping.')

    if missing_val:
        keep      = [i for i, row in val_df.iterrows()
                     if int(row['sample_id']) not in missing_val]
        theta_val = theta_val[keep]

    if len(x_val_list) < 10:
        print('[SKIP] Fewer than 10 valid validation points, skipping.')
    else:
        import joblib
        x_val_full = np.array(x_val_list, dtype=np.float32)
        print(f'  Validation set: {len(theta_val)} points  '
              f'x_val: {x_val_full.shape}')

        np.save(os.path.join(WORK_DIR, 'x_val_full.npy'),     x_val_full)
        np.save(os.path.join(WORK_DIR, 'theta_val_full.npy'), theta_val)

        val_ok, val_failed = [], []

        for name, idx in STAT_MAP.items():
            posterior_path = os.path.join(WORK_DIR, f'ili_posterior_{name}.pkl')
            if not os.path.exists(posterior_path):
                print(f'  [SKIP] No saved posterior for {name}.')
                continue

            print(f'\n  === Validating {name} ===')

            # ── 1. Load posterior and move entirely to CPU ──────────────
            with open(posterior_path, 'rb') as f:
                post = pk.load(f)

            # Move every sub-posterior's network to CPU so prior bounds,
            # theta, and network weights all live on the same device.
            members = post.posteriors if hasattr(post, 'posteriors') else [post]
            for member in members:
                for attr in ('posterior_estimator', '_neural_net',
                             '_posterior_estimator', '_likelihood_estimator',
                             '_ratio_estimator'):
                    net = getattr(member, attr, None)
                    if net is not None:
                        net.to('cpu')
                # Move the prior's parameter tensors to CPU if possible
                prior = getattr(member, '_prior', None)
                if prior is not None:
                    for buf_attr in ('low', 'high', 'loc', 'scale'):
                        buf = getattr(prior, buf_attr, None)
                        if isinstance(buf, torch.Tensor):
                            setattr(prior, buf_attr, buf.to('cpu'))

            # ── 2. Preprocessing: z-score + PCA ────────────────────────
            x_mean = np.load(os.path.join(WORK_DIR, f'scaler_{name}_mean.npy'))
            x_std  = np.load(os.path.join(WORK_DIR, f'scaler_{name}_std.npy'))

            x_val_slice = x_val_full[:, idx].astype(np.float32)
            xt_val_norm = ((x_val_slice - x_mean) / x_std).astype(np.float32)

            pca_path   = os.path.join(WORK_DIR, f'pca_{name}.pkl')
            ncomp_path = os.path.join(WORK_DIR, f'pca_{name}_n_comp.npy')
            if os.path.exists(pca_path) and os.path.exists(ncomp_path):
                pca    = joblib.load(pca_path)
                n_comp = int(np.load(ncomp_path))
                xt_val = pca.transform(xt_val_norm)[:, :n_comp].astype(np.float32)
                print(f'  [{name}] Applied PCA: '
                      f'{xt_val_norm.shape[1]} -> {n_comp}')
            else:
                print(f'  [WARN] No PCA found for {name}, using z-scored data.')
                xt_val = xt_val_norm

            # ── 3. Write data for StaticNumpyLoader ────────────────────
            val_dir = Path(WORK_DIR) / f'validation_{name}'
            val_dir.mkdir(exist_ok=True, parents=True)
            np.save(val_dir / 'x_val.npy',     xt_val)
            np.save(val_dir / 'theta_val.npy', theta_val)

            loader = StaticNumpyLoader(
                in_dir=str(val_dir),
                x_file='x_val.npy',
                theta_file='theta_val.npy',
            )

            # ── 4. ltu-ili ValidationRunner + PosteriorCoverage ────────
            try:
                metrics = {
                    'coverage': PosteriorCoverage(
                        num_samples=200,
                        sample_method='emcee',
                        sample_params={
                            'num_chains': 8,
                            'thin':       2,
                            'burn_in':    50,
                        },
                        labels=PARAM_LABELS,
                        out_dir=str(val_dir),
                        plot_list=['coverage', 'histogram',
                                   'tarp', 'predictions'],
                    )
                }
                val_runner = ValidationRunner(
                    posterior=post,          # plain post — now on CPU
                    metrics=metrics,
                    out_dir=str(val_dir),
                )
                val_runner(loader)
                print(f'  [{name}] Plots saved to {val_dir}')
                val_ok.append(name)
            except Exception:
                import traceback
                print(f'  [FAIL] {name}: {traceback.format_exc()}')
                val_failed.append(name)

        print('\n--- Validation Summary ---')
        print(f'  OK:     {val_ok}')
        if val_failed:
            print(f'  Failed: {val_failed}')
    # ── Active learning proposal ───────────────────────────────────────────────
    print(f'\nGenerating round {NEXT_ROUND} proposals from '
          f'{PROPOSAL_STAT} posterior...')

    proposal_pkl  = os.path.join(WORK_DIR, f'ili_posterior_{PROPOSAL_STAT}.pkl')
    proposal_xobs = os.path.join(WORK_DIR, f'xobs_{PROPOSAL_STAT}.npy')

    if not os.path.exists(proposal_pkl):
        raise FileNotFoundError(
            f'Proposal posterior not found: {proposal_pkl}. '
            f'Check that {PROPOSAL_STAT} trained successfully.')

    with open(proposal_pkl, 'rb') as f:
        proposal_posterior = pk.load(f)
    xo_proposal = np.load(proposal_xobs)

    next_theta = sample_ensemble_direct(
        proposal_posterior, xo_proposal, n_samples=200)
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
        print(f'  {pname}: mean={col.mean():.3f}  std={col.std():.3f}  '
              f'range=[{col.min():.3f}, {col.max():.3f}]')

    print('\nAll done. Generate contour plots and run the next round of samples!')
