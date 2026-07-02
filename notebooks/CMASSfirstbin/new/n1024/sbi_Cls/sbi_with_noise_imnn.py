import os
import sys
import threading
import torch
import torch.nn as nn
import torch.optim as optim
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

CSV_FILES = [
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/lhs_samples.csv',    0),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round2_samples.csv', 500),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round3_samples.csv', 700),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round4_samples.csv', 900),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round5_samples.csv',1100),
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
N_PARAMS     = len(PARAM_NAMES)
PROPOSAL_STAT = 'JOINT'
VAL_FRACTION  = 0.10

# =============================================================================
# IMNN SETTINGS
# =============================================================================
IMNN_N_SUMMARIES = 6
IMNN_HIDDEN      = [128, 128, 64]
IMNN_EPOCHS      = 300
IMNN_LR          = 1e-3
IMNN_BATCH_FRAC  = 0.25
IMNN_PATIENCE    = 30
IMNN_LAMBDA      = 10.0

# =============================================================================
# SBI / MDN ARCHITECTURE
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

OPTUNA_STUDY_DIRS = {name: os.path.join(ILIAS_BASE, name) for name in STAT_MAP}

INDIVIDUAL_STATS = {'gy', 'gtau', 'gkappa', 'g2y', 'g2tau', 'g2kappa'}
BLOCK_SIZE = N_ELL_BINS_ACTUAL

def make_blocks(n_features):
    return [list(range(i, min(i + BLOCK_SIZE, n_features)))
            for i in range(0, n_features, BLOCK_SIZE)]

# =============================================================================
# OPTUNA HYPERPARAMETER LOADER
# =============================================================================

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
            args=(member, x_t, per_member, result, exception),daemon=True,
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
# Z-SCORE NORMALISATION
# =============================================================================

def zscore_stat(x_full, x_obs_slice, blocks):
    """Z-score normalise. Returns (xt_norm, xo_norm, mean, std)."""
    n_stats = x_full.shape[1]
    if blocks is None:
        x_mean = np.mean(x_full, axis=0)
        x_std  = np.std( x_full, axis=0)
        x_std[x_std < 1e-10] = 1.0
        xt_norm = (x_full      - x_mean) / x_std
        xo_norm = (x_obs_slice - x_mean) / x_std
    else:
        xt_norm = np.empty_like(x_full)
        xo_norm = np.empty(n_stats, dtype=np.float32)
        x_mean  = np.empty(n_stats, dtype=np.float32)
        x_std   = np.empty(n_stats, dtype=np.float32)
        for blk in blocks:
            blk = np.asarray(blk)
            m   = np.mean(x_full[:, blk], axis=0)
            s   = np.std( x_full[:, blk], axis=0)
            s[s < 1e-10] = 1.0
            xt_norm[:, blk] = (x_full[:, blk]  - m) / s
            xo_norm[blk]    = (x_obs_slice[blk] - m) / s
            x_mean[blk]     = m
            x_std[blk]      = s
    return xt_norm.astype(np.float32), xo_norm.astype(np.float32), x_mean, x_std

# =============================================================================
# IMNN
# =============================================================================

class IMNNCompressor(nn.Module):
    """MLP compressing data vectors to Fisher-information-maximising summaries."""
    def __init__(self, n_input: int, n_summaries: int, hidden: list):
        super().__init__()
        layers = []
        in_dim = n_input
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_summaries))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _imnn_loss(t_fiducial, t_plus, t_minus, delta_theta, lam=IMNN_LAMBDA):
    """
    loss = -log|F| + lam * ||C - I||_F^2
    t_fiducial : (n, n_s)
    t_plus     : (n_params, n_s)  mean summaries for above-median theta
    t_minus    : (n_params, n_s)  mean summaries for below-median theta
    delta_theta: (n_params,)
    """
    n, n_s = t_fiducial.shape

    # ── Covariance ────────────────────────────────────────────────────────────
    t_mean = t_fiducial.mean(dim=0, keepdim=True)
    diff   = t_fiducial - t_mean
    C      = (diff.T @ diff) / (n - 1)

    # Stronger diagonal regularisation to avoid singular C early in training
    C_reg  = C + 1e-3 * torch.eye(n_s, device=C.device)

    # Use pseudoinverse — safe even if C is singular
    C_inv  = torch.linalg.pinv(C_reg)

    # ── Fisher matrix ─────────────────────────────────────────────────────────
    dmu_dtheta = (t_plus - t_minus) / (2.0 * delta_theta.unsqueeze(-1))   # (n_params, n_s)
    F          = dmu_dtheta @ C_inv @ dmu_dtheta.T                         # (n_params, n_params)

    # Regularise F too to avoid log of zero/negative determinant
    F_reg          = F + 1e-6 * torch.eye(F.shape[0], device=F.device)
    sign, logdetF  = torch.linalg.slogdet(F_reg)

    # Only use logdetF if the matrix is positive definite
    if sign > 0:
        loss_F = -logdetF
    else:
        # Fall back to pure regularisation loss until C stabilises
        loss_F = torch.tensor(0.0, device=C.device, requires_grad=False)

    # ── Covariance regularisation: push C towards identity ────────────────────
    loss_reg = torch.sum((C - torch.eye(n_s, device=C.device)) ** 2)

    total_loss = loss_F + lam * loss_reg
    return total_loss, -logdetF.item() if sign > 0 else float('nan'), loss_reg.item()

def fit_imnn(x_norm, theta_train, n_summaries, hidden=None,
             epochs=IMNN_EPOCHS, lr=IMNN_LR, batch_frac=IMNN_BATCH_FRAC,
             patience=IMNN_PATIENCE, lam=IMNN_LAMBDA,
             device='cpu', name='') -> IMNNCompressor:
    if hidden is None:
        hidden = IMNN_HIDDEN

    n_input = x_norm.shape[1]
    net     = IMNNCompressor(n_input, n_summaries, hidden).to(device)
    # ── Weight init: small last-layer weights help prevent early collapse ──────
    with torch.no_grad():
        last_layer = list(net.net.children())[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.normal_(last_layer.weight, std=0.01)
            nn.init.zeros_(last_layer.bias)
    opt     = optim.Adam(net.parameters(), lr=lr)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=patience // 2)

    x_t     = torch.tensor(x_norm,      dtype=torch.float32, device=device)
    theta_t = torch.tensor(theta_train, dtype=torch.float32, device=device)

    prior_range = torch.tensor(
        [h - l for h, l in zip(PRIOR_HIGH, PRIOR_LOW)],
        dtype=torch.float32, device=device)
    delta_theta = 0.01 * prior_range

    n_train    = len(x_t)
    batch_size = max(32, int(n_train * batch_frac))
    n_params   = theta_t.shape[1]

    best_loss  = np.inf
    best_state = None
    no_improve = 0

    print(f'  [{name}] IMNN: {n_input} -> {n_summaries} summaries  '
          f'hidden={hidden}  n_train={n_train}  '
          f'batch={batch_size}  epochs={epochs}  device={device}', flush=True)

    for epoch in range(epochs):
        net.train()
        perm       = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_train, batch_size):
            idx_b   = perm[start:start + batch_size]
            xb      = x_t[idx_b]
            theta_b = theta_t[idx_b]

            # Need enough samples to estimate covariance reliably
            if len(idx_b) < n_summaries + 2:
                continue

            t_fid = net(xb)

            t_plus_list, t_minus_list = [], []
            for p in range(n_params):
                med        = theta_b[:, p].median()
                above_mask = theta_b[:, p] >= med
                below_mask = ~above_mask
                if above_mask.sum() < 2 or below_mask.sum() < 2:
                    t_plus_list.append(t_fid.mean(dim=0))
                    t_minus_list.append(t_fid.mean(dim=0))
                else:
                    t_plus_list.append(net(xb[above_mask]).mean(dim=0))
                    t_minus_list.append(net(xb[below_mask]).mean(dim=0))

            t_plus  = torch.stack(t_plus_list,  dim=0)
            t_minus = torch.stack(t_minus_list, dim=0)

            loss, logdetF, reg = _imnn_loss(t_fid, t_plus, t_minus,
                                             delta_theta, lam=lam)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        epoch_loss /= max(n_batches, 1)
        sched.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 50 == 0:
            print(f'  [{name}] epoch {epoch+1:4d}/{epochs}  '
                  f'loss={epoch_loss:.4f}  best={best_loss:.4f}  '
                  f'logdetF={logdetF:.3f}  reg={reg:.4f}', flush=True)

        if no_improve >= patience:
            print(f'  [{name}] Early stop at epoch {epoch+1}', flush=True)
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    print(f'  [{name}] IMNN done. Best loss={best_loss:.4f}', flush=True)
    return net


def apply_imnn(net, x_norm, device='cpu'):
    net.eval()
    with torch.no_grad():
        t = net(torch.tensor(x_norm, dtype=torch.float32, device=device))
    return t.cpu().numpy()


def load_imnn(name, work_dir, device='cpu'):
    arch        = np.load(os.path.join(work_dir, f'imnn_{name}_arch.npy'),
                          allow_pickle=True)
    n_input     = int(arch[0])
    n_summaries = int(arch[1])
    hidden      = [int(h) for h in arch[2:]]
    net         = IMNNCompressor(n_input, n_summaries, hidden).to(device)
    net.load_state_dict(
        torch.load(os.path.join(work_dir, f'imnn_{name}_weights.pt'),
                   map_location=device))
    net.eval()
    return net

# =============================================================================
# SBI TRAINING  (single statistic, fully on GPU)
# =============================================================================

def train_one_statistic_gpu(name, xt_comp, theta_train, xo_comp,
                             work_dir, device, opt_hps):
    """Train one MDN posterior entirely on GPU."""
    from ili.dataloaders import StaticNumpyLoader
    from ili.inference import InferenceRunner
    from ili.utils import load_nde_sbi
    from sbi.utils import BoxUniform

    def fpath(fname):
        return os.path.join(work_dir, fname)

    try:
        n_summaries = xt_comp.shape[1]
        n_train     = len(theta_train)

        np.save(fpath(f'x_{name}.npy'),           xt_comp)
        np.save(fpath(f'xobs_{name}.npy'),        xo_comp)
        np.save(fpath(f'theta_train_{name}.npy'), theta_train)

        loader = StaticNumpyLoader(
            in_dir=work_dir,
            x_file=f'x_{name}.npy',
            theta_file=f'theta_train_{name}.npy',
            xobs_file=f'xobs_{name}.npy',
        )

        # ── MDN architecture ──────────────────────────────────────────────────
        if FORCE_EQUAL_ARCH:
            hfs            = EQUAL_ARCH['hidden_features']
            num_components = EQUAL_ARCH['num_components']
            batch_size     = EQUAL_ARCH['batch_size']
            lr             = EQUAL_ARCH['learning_rate']
            max_epochs     = EQUAL_ARCH['max_num_epochs']
            repeats        = EQUAL_ARCH['repeats']
            print(f'{name:12s}  [forced arch] hfs={hfs}  '
                  f'num_components={num_components}  '
                  f'lr={lr:.2e}  batch={batch_size}  epochs={max_epochs}',
                  flush=True)
        elif opt_hps is not None:
            hfs            = opt_hps['hidden_features']
            num_components = opt_hps['num_components']
            batch_size     = opt_hps['batch_size']
            lr             = opt_hps['learning_rate']
            max_epochs     = opt_hps['max_num_epochs']
            repeats        = 6
            print(f'{name:12s}  [optuna arch] hfs={hfs}  '
                  f'num_components={num_components}  '
                  f'lr={lr:.2e}  batch={batch_size}  epochs={max_epochs}',
                  flush=True)
        else:
            hfs            = 64
            num_components = EQUAL_ARCH['num_components']
            n_train_eff    = int(n_train * (1.0 - VAL_FRACTION))
            batch_size     = int(np.clip(n_train_eff // 8, 32, 256))
            lr             = 5e-4
            max_epochs     = 500
            repeats        = 6
            print(f'{name:12s}  [adaptive arch] hfs={hfs}  '
                  f'num_components={num_components}  '
                  f'lr={lr:.2e}  batch={batch_size}  epochs={max_epochs}',
                  flush=True)

        train_args = {
            'training_batch_size': batch_size,
            'learning_rate':       lr,
            'max_num_epochs':      max_epochs,
            'stop_after_epochs':   50,
            'clip_max_norm':       5.0,
            'validation_fraction': VAL_FRACTION,
        }

        nets = load_nde_sbi(engine='NPE', model='mdn',
                            repeats=repeats,
                            hidden_features=hfs,
                            num_components=num_components)

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

        msg = (f'[{name}] DONE  n_summaries={n_summaries}  '
               f'n_train={n_train}  hfs={hfs}  '
               f'repeats={repeats}  batch={batch_size}  device={device}')
        return name, True, msg

    except Exception as e:
        import traceback
        return name, False, f'[{name}] FAILED: {traceback.format_exc()}'

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-reload', action='store_true',
                        help='Ignore cached x_train/theta_train and rebuild from CSVs.')
    args, _ = parser.parse_known_args()

    os.makedirs(CACHE_DIR,     exist_ok=True)
    os.makedirs(VAL_CACHE_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Next round:       {NEXT_ROUND}')
    print(f'ADD_SURVEY_NOISE  = {ADD_SURVEY_NOISE}')
    print(f'FORCE_EQUAL_ARCH  = {FORCE_EQUAL_ARCH}')
    print(f'IMNN_N_SUMMARIES  = {IMNN_N_SUMMARIES}')
    print(f'IMNN_HIDDEN       = {IMNN_HIDDEN}')
    print(f'IMNN_EPOCHS       = {IMNN_EPOCHS}')
    print(f'IMNN_LAMBDA       = {IMNN_LAMBDA}')
    print(f'Cl settings: LMIN={LMIN}  LMAX={LMAX}  '
          f'N_ELL_BINS={N_ELL_BINS_ACTUAL}  N_SUMMARY={N_SUMMARY}')
    print(f'ELL_CENTRES: {np.round(ELL_CENTRES).astype(int).tolist()}')
    print(f'OUTPUTS_DIR: {OUTPUTS_DIR}')
    print(f'NOISE_PATH:  {NOISE_PATH}')

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

    # ── Training data ──────────────────────────────────────────────────────────
    x_train_path     = os.path.join(WORK_DIR, f'x_train_full{_CACHE_SUFFIX}.npy')
    theta_train_path = os.path.join(WORK_DIR, 'theta_train_full.npy')

    if not args.force_reload \
            and os.path.exists(x_train_path) \
            and os.path.exists(theta_train_path):
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
          f'x_train: {x_train.shape}  theta_train: {theta_train.shape}')

    # ── Per-spectrum Fisher correlation diagnostic ─────────────────────────────
    print('\nPer-spectrum Fisher correlations with parameters:')
    for i, (label, _, _) in enumerate(CL_SPECS):
        sl       = slice(i * N_ELL_BINS_ACTUAL, (i + 1) * N_ELL_BINS_ACTUAL)
        cl_block = x_train[:, sl]
        for p, pname in enumerate(PARAM_NAMES):
            r = np.array([
                np.corrcoef(cl_block[:, j], theta_train[:, p])[0, 1]
                for j in range(N_ELL_BINS_ACTUAL)
            ])
            print(f'  {label:8s}  {pname}: '
                  f'max|r|={np.abs(r).max():.3f}  '
                  f'mean|r|={np.abs(r).mean():.3f}  '
                  f'best_ell={int(ELL_CENTRES[np.abs(r).argmax()])}')

    # ── IMNN pre-compression on GPU ────────────────────────────────────────────
    print('\nPre-computing IMNN compressions on GPU...')
    imnn_compressed = {}   # name -> (xt_comp, xo_comp)

    for name, idx in STAT_MAP.items():
        xt_full = x_train[:, idx].astype(np.float32)
        xo      = x_obs[idx].astype(np.float32)
        n_stats = len(idx)

        blocks = None if name in INDIVIDUAL_STATS else make_blocks(n_stats)
        xt_norm, xo_norm, x_mean, x_std = zscore_stat(xt_full, xo, blocks)

        np.save(os.path.join(WORK_DIR, f'scaler_{name}_mean.npy'), x_mean)
        np.save(os.path.join(WORK_DIR, f'scaler_{name}_std.npy'),  x_std)

        if name.startswith('g2'):
            neg_frac = float((xt_full < 0).mean())
            print(f'  [{name}] neg_fraction={neg_frac:.3f}', flush=True)

        frac_below = float((xo < xt_full.min(axis=0)).mean())
        frac_above = float((xo > xt_full.max(axis=0)).mean())
        print(f'  [{name}] frac_below={frac_below:.2f}  '
              f'frac_above={frac_above:.2f}', flush=True)

        n_summaries = IMNN_N_SUMMARIES if n_stats > N_ELL_BINS_ACTUAL \
                      else max(N_PARAMS, 3)

        imnn_net = fit_imnn(
            xt_norm, theta_train,
            n_summaries=n_summaries,
            hidden=IMNN_HIDDEN,
            epochs=IMNN_EPOCHS,
            lr=IMNN_LR,
            batch_frac=IMNN_BATCH_FRAC,
            patience=IMNN_PATIENCE,
            lam=IMNN_LAMBDA,
            device=device,
            name=name,
        )

        xt_comp = apply_imnn(imnn_net, xt_norm, device=device).astype(np.float32)
        xo_comp = apply_imnn(imnn_net, xo_norm.reshape(1, -1),
                             device=device)[0].astype(np.float32)

        # R² diagnostic
        for p, pname in enumerate(PARAM_NAMES):
            r2 = np.array([
                np.corrcoef(xt_comp[:, s], theta_train[:, p])[0, 1] ** 2
                for s in range(n_summaries)
            ])
            print(f'  [{name}] IMNN R² with {pname}: '
                  f'{np.round(r2, 3).tolist()}  max={r2.max():.3f}', flush=True)

        # Save
        torch.save(imnn_net.state_dict(),
                   os.path.join(WORK_DIR, f'imnn_{name}_weights.pt'))
        np.save(os.path.join(WORK_DIR, f'imnn_{name}_arch.npy'),
                np.array([n_stats, n_summaries] + IMNN_HIDDEN, dtype=object))
        np.save(os.path.join(WORK_DIR, f'imnn_{name}_n_summaries.npy'),
                np.array(n_summaries))

        imnn_compressed[name] = (xt_comp, xo_comp)
        print(f'  [{name}] {n_stats} -> {n_summaries} summaries saved.', flush=True)

    # ── Sequential GPU SBI training ────────────────────────────────────────────
    print(f'\nTraining {N_STATISTICS} posteriors sequentially on {device}...')

    results = []
    for name, idx in STAT_MAP.items():
        print(f'\n=== SBI training: {name} ===', flush=True)
        xt_comp, xo_comp = imnn_compressed[name]
        opt_hps = None
        if not FORCE_EQUAL_ARCH:
            opt_hps = load_optuna_hyperparams(OPTUNA_STUDY_DIRS.get(name, ''))
            if opt_hps is None:
                print(f'  [{name}] No Optuna study found, using adaptive defaults.')

        result = train_one_statistic_gpu(
            name, xt_comp, theta_train, xo_comp, WORK_DIR, device, opt_hps)
        results.append(result)

        # Free GPU cache between runs
        if device == 'cuda':
            torch.cuda.empty_cache()

    print('\n--- Training Summary ---')
    for name, success, msg in results:
        status = 'OK  ' if success else 'FAIL'
        print(f'  [{status}] {msg}')

    # ── Validation ────────────────────────────────────────────────────────────
    if not os.path.exists(VALIDATION_CSV):
        print(f'\n[Validation] {VALIDATION_CSV} not found — skipping.')
    else:
        print('\nLoading held-out validation set...')
        os.makedirs(VAL_CACHE_DIR, exist_ok=True)

        val_df      = pd.read_csv(VALIDATION_CSV)
        theta_val   = val_df[PARAM_NAMES].values.astype(np.float32)
        x_val_list  = []
        missing_val = []

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
                print(f'  [WARN] No data found for validation_{sid}, skipping.')

        if missing_val:
            keep      = [i for i, row in val_df.iterrows()
                         if int(row['sample_id']) not in missing_val]
            theta_val = theta_val[keep]

        if len(x_val_list) < 10:
            print('[SKIP] Fewer than 10 valid validation points, skipping.')
        else:
            x_val_full = np.array(x_val_list, dtype=np.float32)
            print(f'  Validation set: {len(theta_val)} points  '
                  f'x_val: {x_val_full.shape}  theta_val: {theta_val.shape}')

            np.save(os.path.join(WORK_DIR, 'x_val_full.npy'),     x_val_full)
            np.save(os.path.join(WORK_DIR, 'theta_val_full.npy'), theta_val)
            print('  Saved x_val_full.npy and theta_val_full.npy')

            val_ok, val_failed = [], []

            for name, idx in STAT_MAP.items():
                posterior_path = os.path.join(WORK_DIR, f'ili_posterior_{name}.pkl')
                if not os.path.exists(posterior_path):
                    print(f'  [SKIP] No saved posterior for {name}.')
                    continue

                print(f'\n  === Validating {name} ===')

                # ── Load posterior ─────────────────────────────────────────────
                with open(posterior_path, 'rb') as f:
                    post = pk.load(f)

                # ── Move entire posterior to CPU for emcee compatibility ────────
                # emcee always runs on CPU; the posterior was trained on GPU so
                # all tensors (network weights + prior bounds) must be moved.
                def _to_cpu(p):
                    try:
                        p._neural_net = p._neural_net.to('cpu')
                    except AttributeError:
                        pass
                    for attr in ('_prior', 'prior'):
                        try:
                            prior = getattr(p, attr)
                            if hasattr(prior, 'low'):
                                prior.low  = prior.low.to('cpu')
                                prior.high = prior.high.to('cpu')
                        except AttributeError:
                            pass
                    try:
                        prior = p.potential_fn.prior
                        if hasattr(prior, 'low'):
                            prior.low  = prior.low.to('cpu')
                            prior.high = prior.high.to('cpu')
                    except AttributeError:
                        pass
                    return p

                try:
                    # EnsemblePosterior: move each member
                    for i, member in enumerate(post.posteriors):
                        post.posteriors[i] = _to_cpu(member)
                    # Also move ensemble-level prior if present
                    for attr in ('_prior', 'prior'):
                        try:
                            prior = getattr(post, attr)
                            if hasattr(prior, 'low'):
                                prior.low  = prior.low.to('cpu')
                                prior.high = prior.high.to('cpu')
                        except AttributeError:
                            pass
                except AttributeError:
                    # Single posterior
                    post = _to_cpu(post)

                print(f'  [{name}] Posterior moved to CPU for emcee sampling.')

                # ── z-score validation data using saved scalers ────────────────
                x_mean = np.load(os.path.join(WORK_DIR, f'scaler_{name}_mean.npy'))
                x_std  = np.load(os.path.join(WORK_DIR, f'scaler_{name}_std.npy'))
                xt_val_norm = ((x_val_full[:, idx] - x_mean)
                               / (x_std + 1e-8)).astype(np.float32)

                # ── Apply saved IMNN (on CPU for consistency) ──────────────────
                imnn_weights = os.path.join(WORK_DIR, f'imnn_{name}_weights.pt')
                imnn_arch    = os.path.join(WORK_DIR, f'imnn_{name}_arch.npy')
                if os.path.exists(imnn_weights) and os.path.exists(imnn_arch):
                    imnn_net = load_imnn(name, WORK_DIR, device='cpu')
                    xt_val   = apply_imnn(imnn_net, xt_val_norm,
                                          device='cpu').astype(np.float32)
                    n_s      = int(np.load(os.path.join(
                                   WORK_DIR, f'imnn_{name}_n_summaries.npy')))
                    print(f'  [{name}] Applied IMNN: '
                          f'{xt_val_norm.shape[1]} -> {n_s} summaries')
                else:
                    print(f'  [WARN] No IMNN found for {name}, using z-scored data.')
                    xt_val = xt_val_norm

                # ── Save compressed validation data ────────────────────────────
                val_dir = Path(WORK_DIR) / f'validation_{name}'
                val_dir.mkdir(exist_ok=True, parents=True)
                np.save(val_dir / 'x_val.npy',     xt_val)
                np.save(val_dir / 'theta_val.npy', theta_val)

                loader = StaticNumpyLoader(
                    in_dir=str(val_dir),
                    x_file='x_val.npy',
                    theta_file='theta_val.npy',
                )

                # ── Run validation metrics ─────────────────────────────────────
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
                            out_dir=val_dir,
                            plot_list=['coverage', 'histogram',
                                       'tarp', 'predictions'],
                        )
                    }
                    val_runner = ValidationRunner(
                        posterior=post,
                        metrics=metrics,
                        out_dir=val_dir,
                    )
                    val_runner(loader)
                    print(f'  [{name}] Plots saved to {val_dir}')
                    val_ok.append(name)
                except Exception:
                    import traceback
                    print(f'  [FAIL] {name}: {traceback.format_exc()}')
                    val_failed.append(name)

                # Free GPU memory between runs
                if device == 'cuda':
                    torch.cuda.empty_cache()

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
