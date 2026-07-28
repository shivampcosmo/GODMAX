import os
import sys
import glob
import pickle as pk
import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

import getdist.plots as gdplt
from cobaya.likelihood import Likelihood
from cobaya import run as cobaya_run

# =============================================================================
# PATHS AND CONSTANTS
# =============================================================================

BASE_DIR  = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/new/n1024'
WORK_DIR  = str(Path(__file__).parent.resolve())
CACHE_DIR = os.path.join(WORK_DIR, 'sample_vector_cache_cls')

CSV_FILES = [
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/lhs_samples.csv', 0),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round2_samples.csv', 500),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round3_samples.csv', 700),
]

NSIDE      = 1024
LMIN       = 100
LMAX       = 1500
N_ELL_BINS = 20

PRIOR_LOW    = [1.0, -0.3]
PRIOR_HIGH   = [6.0,  0.0]
PARAM_LABELS = [r'$$\theta_{ej,0}$$', r'$${\nu_{\theta_{ej}}}^{M}$$']
PARAM_NAMES  = ['theta_ej_0', 'nu_theta_ej_M']

RUN_STATS     = None   # set to e.g. ['gkappa', 'g2kappa'] to run a subset
GP_N_RESTARTS = 5
CHAINS_DIR    = os.path.join(WORK_DIR, 'chains_cls')
MCMC_SAMPLES  = 5_000
RMINUS1_STOP  = 0.02

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
              f'Effective bins: {len(edges) - 1}')
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
    ('g2y',     'gal_sq',    'ymap'),
    ('g2tau',   'gal_sq',    'tau'),
    ('g2kappa', 'gal_sq',    'kappa'),
    ('k2g',     'kappa_sq',  'gal'),
    ('gky',     'gkap_prod', 'ymap'),
    ('gy',      'gal',       'ymap'),
    ('gtau',    'gal',       'tau'),
    ('gkappa',  'gal',       'kappa'),
]
N_SPECS   = len(CL_SPECS)
N_SUMMARY = N_SPECS * N_ELL_BINS_ACTUAL

_s = {label: list(range(i * N_ELL_BINS_ACTUAL, (i + 1) * N_ELL_BINS_ACTUAL))
      for i, (label, _, _) in enumerate(CL_SPECS)}

STAT_MAP = {
    'g2y':          _s['g2y'],
    'g2tau':        _s['g2tau'],
    'g2kappa':      _s['g2kappa'],
    'k2g':          _s['k2g'],
    'gky':          _s['gky'],
    'gy':           _s['gy'],
    'gtau':         _s['gtau'],
    'gkappa':       _s['gkappa'],
    'y_total':      _s['g2y']    + _s['gy'],
    'tau_total':    _s['g2tau']  + _s['gtau'],
    'kappa_total':  _s['g2kappa']+ _s['gkappa'],
    'g2kappa_k2g':  _s['g2kappa']+ _s['k2g'],
    'kappa_full':   _s['g2kappa']+ _s['k2g']  + _s['gkappa'],
    'kappa_break':  _s['g2kappa']+ _s['k2g']  + _s['gky'],
    'k2g_gkappa':   _s['k2g']   + _s['gkappa'],
    'gky_gy':       _s['gky']   + _s['gy'],
    'all_3pt':      _s['g2y']   + _s['g2tau'] + _s['g2kappa']
                                + _s['k2g']   + _s['gky'],
    'all_2pt':      _s['gy']    + _s['gtau']  + _s['gkappa'],
    'JOINT':        list(range(N_SUMMARY)),
}

# =============================================================================
# SUMMARY STATISTIC EXTRACTION
# =============================================================================

def extract_Cls(path):
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
                pix  = hp.ang2pix(NSIDE, ra_gal[mask], dec_gal[mask], lonlat=True)
                gmap += np.bincount(pix, minlength=12 * NSIDE**2).astype(np.float32)

    mean_g = float(np.mean(gmap))
    if mean_g <= 0:
        return None

    delta_gal = gmap.astype(np.float64) / mean_g - 1.0
    delta_sq  = delta_gal ** 2
    delta_sq -= np.mean(delta_sq)

    kmap_clean  = np.nan_to_num(kmap)
    delta_kappa = kmap_clean - float(np.mean(kmap_clean))
    kappa_sq    = delta_kappa ** 2
    kappa_sq   -= np.mean(kappa_sq)
    gkap_prod   = delta_gal * delta_kappa
    gkap_prod  -= np.mean(gkap_prod)

    pixwin = hp.pixwin(NSIDE)
    pw1    = pixwin[:LMAX + 1].copy()
    pw1    = np.where(pw1 > 0, pw1, 1.0)
    pw2    = pw1 ** 2

    field_map = {
        'gal':       (delta_gal,           pw1),
        'gal_sq':    (delta_sq,            pw2),
        'kappa_sq':  (kappa_sq,            None),
        'gkap_prod': (gkap_prod,           pw1),
        'ymap':      (np.nan_to_num(ymap), None),
        'tau':       (np.nan_to_num(tmap), None),
        'kappa':     (kmap_clean,          None),
    }

    vec = []
    for label, field_a, field_b in CL_SPECS:
        map_a, pw_a = field_map[field_a]
        map_b, pw_b = field_map[field_b]
        cl_full = hp.anafast(map_a, map_b, lmax=LMAX)
        if pw_a is not None:
            cl_full = cl_full / pw_a
        if pw_b is not None:
            cl_full = cl_full / pw_b
        vec.append(bin_cl(cl_full))

    return np.concatenate(vec).astype(np.float32)

# =============================================================================
# GP EMULATOR
# =============================================================================

class GPEmulator:
    def __init__(self, n_restarts: int = 5):
        self.n_restarts    = n_restarts
        self.gps_          = []
        self.theta_scaler_ = StandardScaler()
        self.x_scaler_     = StandardScaler()
        self.noise_var_    = None

    def fit(self, theta: np.ndarray, x: np.ndarray):
        theta_s = self.theta_scaler_.fit_transform(theta.astype(np.float64))
        x_s     = self.x_scaler_.fit_transform(x.astype(np.float64))

        n_params   = theta.shape[1]
        n_features = x.shape[1]
        self.gps_       = []
        self.noise_var_ = np.zeros(n_features)

        kernel_base = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(n_params),
                  length_scale_bounds=(1e-2, 10.0))
            + WhiteKernel(noise_level=1e-3,
                          noise_level_bounds=(1e-8, 1.0))
        )

        print(f'  Fitting {n_features} GPs on {theta.shape[0]} simulations...')
        for i in tqdm(range(n_features), desc='  GP fitting'):
            gp = GaussianProcessRegressor(
                kernel=kernel_base.clone_with_theta(kernel_base.theta),
                n_restarts_optimizer=self.n_restarts,
                normalize_y=False,
                alpha=0.0,
            )
            gp.fit(theta_s, x_s[:, i])

            white_noise_scaled = gp.kernel_.k2.noise_level
            x_std_i            = self.x_scaler_.scale_[i]
            LOO_BOUND = 1e-7
            if white_noise_scaled < LOO_BOUND:
                y_pred_loo     = gp.predict(theta_s)
                loo_var_scaled = np.var(x_s[:, i] - y_pred_loo)
                noise_scaled = max(loo_var_scaled, LOO_BOUND)
            else:
                noise_scaled = white_noise_scaled

            self.noise_var_[i] = noise_scaled * x_std_i**2
            self.gps_.append(gp)

        return self

    def predict(self, theta: np.ndarray, return_std: bool = False):
        theta   = np.atleast_2d(theta).astype(np.float64)
        theta_s = self.theta_scaler_.transform(theta)

        n_features = len(self.gps_)
        n_query    = theta.shape[0]
        mean_s     = np.zeros((n_query, n_features))
        std_s      = np.zeros((n_query, n_features))

        for i, gp in enumerate(self.gps_):
            if return_std:
                m, s        = gp.predict(theta_s, return_std=True)
                std_s[:, i] = s
            else:
                m = gp.predict(theta_s)
            mean_s[:, i] = m

        mean_raw = self.x_scaler_.inverse_transform(mean_s)

        if return_std:
            std_raw = std_s * self.x_scaler_.scale_[np.newaxis, :]
            return mean_raw, std_raw

        return mean_raw

    def save(self, path: str):
        with open(path, 'wb') as f:
            pk.dump(self, f)

    @staticmethod
    def load(path: str) -> 'GPEmulator':
        with open(path, 'rb') as f:
            return pk.load(f)


# =============================================================================
# COBAYA LIKELIHOOD
# =============================================================================

class GPEmulatorLikelihood(Likelihood):
    """
    Gaussian log-likelihood wrapping a GP emulator for Cl band-powers.

    The diagonal covariance uses the GP WhiteKernel noise estimate per bin.
    For Cl statistics this is a reasonable approximation — the full
    Knox covariance would be the ideal replacement if available.
    """

    emulator_path : str  = ''
    x_obs_path    : str  = ''
    feature_idx   : list = []

    def initialize(self):
        self.emulator_     = GPEmulator.load(self.emulator_path)
        x_obs_full         = np.load(self.x_obs_path).astype(np.float64)
        self.x_obs_        = x_obs_full[self.feature_idx]
        noise_var          = self.emulator_.noise_var_
        self.inv_cov_diag_ = 1.0 / np.clip(noise_var, 1e-40, None)

        print(f'[GPEmulatorLikelihood] {len(self.feature_idx)} Cl bins loaded.')
        print(f'  x_obs  range: [{self.x_obs_.min():.4e}, {self.x_obs_.max():.4e}]')
        print(f'  noise  range: [{noise_var.min():.4e}, {noise_var.max():.4e}]')

    def get_requirements(self):
        return {}

    def logp(self, **params_values):
        theta    = np.array([[params_values['theta_ej_0'],
                              params_values['nu_theta_ej_M']]], dtype=np.float64)
        x_pred   = self.emulator_.predict(theta)[0]
        residual = self.x_obs_ - x_pred
        return float(-0.5 * np.sum(residual**2 * self.inv_cov_diag_))


# =============================================================================
# HELPERS
# =============================================================================

def fit_or_load_emulator(stat_name, feature_idx, theta_train, x_train,
                         emulator_path):
    if os.path.exists(emulator_path):
        print(f'  [{stat_name}] Loading cached GP emulator from {emulator_path}')
        return GPEmulator.load(emulator_path)

    print(f'  [{stat_name}] Fitting GP emulator on '
          f'{len(feature_idx)} Cl bins...')
    x_train_stat = x_train[:, feature_idx].astype(np.float64)
    emulator     = GPEmulator(n_restarts=GP_N_RESTARTS)
    emulator.fit(theta_train.astype(np.float64), x_train_stat)
    emulator.save(emulator_path)
    print(f'  [{stat_name}] Emulator saved to {emulator_path}')
    return emulator


def make_cobaya_info(stat_name, emulator_path, x_obs_path, feature_idx):
    prop_theta    = (PRIOR_HIGH[0] - PRIOR_LOW[0]) * 0.1
    prop_nu       = (PRIOR_HIGH[1] - PRIOR_LOW[1]) * 0.1
    ref_theta     = 0.5 * (PRIOR_LOW[0] + PRIOR_HIGH[0])
    ref_nu        = 0.5 * (PRIOR_LOW[1] + PRIOR_HIGH[1])
    output_prefix = os.path.join(CHAINS_DIR, stat_name, 'mcmc')

    return {
        'likelihood': {
            'gp_emulator': {
                'external':      GPEmulatorLikelihood,
                'input_params':  PARAM_NAMES,
                'emulator_path': emulator_path,
                'x_obs_path':    x_obs_path,
                'feature_idx':   list(feature_idx),
            }
        },
        'params': {
            'theta_ej_0': {
                'prior':    {'min': PRIOR_LOW[0],  'max': PRIOR_HIGH[0]},
                'ref':      ref_theta,
                'proposal': prop_theta,
                'latex':    r'\theta_{ej,0}',
            },
            'nu_theta_ej_M': {
                'prior':    {'min': PRIOR_LOW[1],  'max': PRIOR_HIGH[1]},
                'ref':      ref_nu,
                'proposal': prop_nu,
                'latex':    r'\nu_{\theta_{ej}}^{M}',
            },
        },
        'sampler': {
            'mcmc': {
                'Rminus1_stop':               RMINUS1_STOP,
                'Rminus1_cl_stop':            0.2,
                'learn_proposal':             True,
                'learn_proposal_Rminus1_max': 0.5,
                'burn_in':                    200,
                'max_samples':                MCMC_SAMPLES,
                'max_tries':                  10_000,
            }
        },
        'output': output_prefix,
        'force':  True,
    }


def plot_validation(stat_name, emulator, theta_train, x_train_stat, feature_idx):
    """Plot GP predicted vs simulated for each Cl bin."""
    x_pred = emulator.predict(theta_train.astype(np.float64))
    n_feat = len(feature_idx)
    ncols  = min(n_feat, N_ELL_BINS_ACTUAL)
    nrows  = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for j in range(n_feat):
        ax     = axes_flat[j]
        y_sim  = x_train_stat[:, j]
        y_pred = x_pred[:, j]
        lo     = min(y_sim.min(), y_pred.min())
        hi     = max(y_sim.max(), y_pred.max())
        ax.scatter(y_sim, y_pred, s=6, alpha=0.4, color='steelblue')
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        # Label by which spectrum and which ell bin
        spec_idx = j // N_ELL_BINS_ACTUAL
        bin_idx  = j  % N_ELL_BINS_ACTUAL
        spec_name = CL_SPECS[feature_idx[j] // N_ELL_BINS_ACTUAL][0] \
            if feature_idx[j] // N_ELL_BINS_ACTUAL < len(CL_SPECS) else '?'
        ax.set_title(f'{spec_name}  ℓ≈{int(ELL_CENTRES[bin_idx])}',
                     fontsize=7)
        ax.set_xlabel('Sim',  fontsize=7)
        ax.set_ylabel('GP',   fontsize=7)
        ax.tick_params(labelsize=6)

    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f'GP emulator validation — {stat_name}', fontsize=11)
    plt.tight_layout()
    out = os.path.join(WORK_DIR, f'gp_validation_{stat_name}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [{stat_name}] Validation plot saved to {out}')


def plot_cl_obs_vs_pred(stat_name, emulator, x_obs, feature_idx, theta_ref=None):
    """
    Plot the observed Cl band-powers vs GP prediction at a reference point.
    Useful for checking the emulator is centred on the data.
    """
    if theta_ref is None:
        theta_ref = np.array([[0.5 * (PRIOR_LOW[0] + PRIOR_HIGH[0]),
                               0.5 * (PRIOR_LOW[1] + PRIOR_HIGH[1])]])
    x_pred = emulator.predict(theta_ref)[0]
    x_obs_stat = x_obs[feature_idx]

    n_feat   = len(feature_idx)
    n_specs  = n_feat // N_ELL_BINS_ACTUAL
    fig, axes = plt.subplots(1, n_specs,
                             figsize=(4 * n_specs, 4), squeeze=False)
    for s in range(n_specs):
        sl       = slice(s * N_ELL_BINS_ACTUAL, (s + 1) * N_ELL_BINS_ACTUAL)
        ax       = axes[0, s]
        spec_idx = feature_idx[s * N_ELL_BINS_ACTUAL] // N_ELL_BINS_ACTUAL
        label    = CL_SPECS[spec_idx][0] if spec_idx < len(CL_SPECS) else '?'
        ax.plot(ELL_CENTRES, x_obs_stat[sl],  'k-o',  ms=4, label='Observed')
        ax.plot(ELL_CENTRES, x_pred[sl],       'r--s', ms=4, label='GP (prior centre)')
        ax.set_xscale('log')
        ax.set_xlabel(r'$$\ell$$', fontsize=11)
        ax.set_ylabel(r'$$\ell(\ell+1)C_\ell/2\pi$$', fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle(f'Observed vs GP prediction — {stat_name}', fontsize=12)
    plt.tight_layout()
    out = os.path.join(WORK_DIR, f'cl_obs_vs_pred_{stat_name}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [{stat_name}] Cl obs vs pred plot saved to {out}')


def plot_triangle(stat_name, gd_sample):
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(
        gd_sample,
        PARAM_NAMES,
        filled=True,
        param_limits={
            'theta_ej_0':    (PRIOR_LOW[0], PRIOR_HIGH[0]),
            'nu_theta_ej_M': (PRIOR_LOW[1], PRIOR_HIGH[1]),
        },
    )
    out = os.path.join(WORK_DIR, f'posterior_triangle_{stat_name}.png')
    gdplot.export(out)
    print(f'  [{stat_name}] Triangle plot saved to {out}')


def plot_marginals(stat_name, gd_sample):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, param, label, lo, hi in zip(
            axes, PARAM_NAMES, PARAM_LABELS, PRIOR_LOW, PRIOR_HIGH):
        xs      = np.linspace(lo, hi, 400)
        p1d     = gd_sample.get1DDensity(param)
        ys      = p1d.Prob(xs)
        ys_norm = ys / (ys.max() + 1e-30)
        map_val = xs[np.argmax(ys)]
        ax.plot(xs, ys_norm, color='steelblue', lw=2)
        ax.fill_between(xs, ys_norm, alpha=0.25, color='steelblue')
        ax.axvline(map_val, color='steelblue', ls='--', lw=1.2,
                   label=f'MAP = {map_val:.3f}')
        ax.set_xlabel(label,          fontsize=13)
        ax.set_ylabel('Normalised P', fontsize=12)
        ax.set_title(f'{stat_name} — {param}')
        ax.legend(fontsize=10)
    fig.suptitle(f'1-D marginals [{stat_name}]', fontsize=14)
    plt.tight_layout()
    out = os.path.join(WORK_DIR, f'posterior_marginals_{stat_name}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [{stat_name}] Marginals plot saved to {out}')


def plot_all_posteriors(results: dict):
    stat_names = list(results.keys())
    n_stats    = len(stat_names)
    if n_stats == 0:
        return

    cmap   = plt.get_cmap('tab20')
    colors = [cmap(i / max(n_stats - 1, 1)) for i in range(n_stats)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, param, label, lo, hi in zip(
            axes, PARAM_NAMES, PARAM_LABELS, PRIOR_LOW, PRIOR_HIGH):
        xs = np.linspace(lo, hi, 400)
        for color, stat_name in zip(colors, stat_names):
            gd = results[stat_name]['gd_sample']
            try:
                p1d     = gd.get1DDensity(param)
                ys      = p1d.Prob(xs)
                ys_norm = ys / (ys.max() + 1e-30)
                ax.plot(xs, ys_norm, label=stat_name, color=color, lw=1.8)
            except Exception as e:
                print(f'  [WARN] Could not plot {stat_name}/{param}: {e}')
        ax.set_xlabel(label,          fontsize=13)
        ax.set_ylabel('Normalised P', fontsize=12)
        ax.set_title(param)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle('MCMC posteriors (Cls) — all statistics', fontsize=14)
    plt.tight_layout()
    out = os.path.join(WORK_DIR, 'posteriors_all_stats_cls.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'\nOverlay plot saved to {out}')


def plot_summary_table(results: dict):
    stat_names = list(results.keys())
    n_stats    = len(stat_names)
    if n_stats == 0:
        return

    x_pos = np.arange(n_stats)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, n_stats * 0.9), 5))

    for ax, param, label, lo, hi in zip(
            axes, PARAM_NAMES, PARAM_LABELS, PRIOR_LOW, PRIOR_HIGH):
        means = [results[s]['means'][PARAM_NAMES.index(param)] for s in stat_names]
        stds  = [results[s]['stds'][PARAM_NAMES.index(param)]  for s in stat_names]
        ax.bar(x_pos, means, yerr=stds, capsize=4,
               color='steelblue', alpha=0.7, ecolor='black')
        ax.axhline(lo, ls='--', lw=0.8, color='grey')
        ax.axhline(hi, ls='--', lw=0.8, color='grey')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stat_names, rotation=40, ha='right', fontsize=9)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'Posterior mean ± std  [{param}]')
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

    fig.suptitle('MCMC posterior summary (Cls) — all statistics', fontsize=13)
    plt.tight_layout()
    out = os.path.join(WORK_DIR, 'summary_table_all_stats_cls.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Summary table saved to {out}')


def plot_cl_spectra_training(x_train, theta_train, stat_name, feature_idx):
    """
    Plot all training Cl band-powers coloured by parameter value.
    Useful for visually checking sensitivity of each spectrum to parameters.
    """
    n_feat  = len(feature_idx)
    n_specs = max(1, n_feat // N_ELL_BINS_ACTUAL)
    fig, axes = plt.subplots(2, n_specs,
                             figsize=(4 * n_specs, 7), squeeze=False)

    for pidx, pname in enumerate(PARAM_NAMES):
        pvals  = theta_train[:, pidx]
        vmin, vmax = pvals.min(), pvals.max()
        cmap   = plt.get_cmap('viridis')

        for s in range(n_specs):
            ax  = axes[pidx, s]
            sl  = slice(s * N_ELL_BINS_ACTUAL, (s + 1) * N_ELL_BINS_ACTUAL)
            fidx_sl = feature_idx[sl] if hasattr(feature_idx[0], '__len__') \
                      else feature_idx[s * N_ELL_BINS_ACTUAL:
                                       (s + 1) * N_ELL_BINS_ACTUAL]
            spec_global = fidx_sl[0] // N_ELL_BINS_ACTUAL \
                if fidx_sl[0] // N_ELL_BINS_ACTUAL < len(CL_SPECS) else 0
            spec_label  = CL_SPECS[spec_global][0]

            for i in range(len(theta_train)):
                color = cmap((pvals[i] - vmin) / (vmax - vmin + 1e-30))
                ax.plot(ELL_CENTRES,
                        x_train[i, fidx_sl],
                        color=color, alpha=0.3, lw=0.7)

            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=pname)
            ax.set_xscale('log')
            ax.set_xlabel(r'$$\ell$$',                        fontsize=10)
            ax.set_ylabel(r'$$\ell(\ell+1)C_\ell/2\pi$$',    fontsize=9)
            ax.set_title(f'{spec_label} — coloured by {pname}', fontsize=9)

    fig.suptitle(f'Training Cls coloured by parameter — {stat_name}',
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(WORK_DIR, f'cl_training_{stat_name}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [{stat_name}] Training Cl plot saved to {out}')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    os.makedirs(CACHE_DIR,  exist_ok=True)
    os.makedirs(CHAINS_DIR, exist_ok=True)

    print(f'Cl settings: LMIN={LMIN}  LMAX={LMAX}  '
          f'N_ELL_BINS={N_ELL_BINS_ACTUAL}  N_SUMMARY={N_SUMMARY}')
    print(f'ELL_CENTRES: {np.round(ELL_CENTRES).astype(int).tolist()}')
    print(f'CL_SPECS:    {[s[0] for s in CL_SPECS]}')

    # ── x_obs ─────────────────────────────────────────────────────────────────
    x_obs_path = os.path.join(WORK_DIR, 'x_obs_cls.npy')
    if os.path.exists(x_obs_path):
        print('\nLoading cached x_obs (Cls)...')
        x_obs = np.load(x_obs_path)
    else:
        print('\nExtracting x_obs from reference run...')
        x_obs = extract_Cls(os.path.join(BASE_DIR, 'reference_run'))
        if x_obs is None:
            raise RuntimeError('No pkl files found in reference_run.')
        np.save(x_obs_path, x_obs)

    print(f'  x_obs shape: {x_obs.shape}  (expect {N_SUMMARY})')
    for i, (label, _, _) in enumerate(CL_SPECS):
        sl = slice(i * N_ELL_BINS_ACTUAL, (i + 1) * N_ELL_BINS_ACTUAL)
        print(f'    {label:8s}: mean={x_obs[sl].mean():.4e}  '
              f'range=[{x_obs[sl].min():.4e}, {x_obs[sl].max():.4e}]')

    # ── Training data ─────────────────────────────────────────────────────────
    x_train_path     = os.path.join(WORK_DIR, 'x_train_full_cls.npy')
    theta_train_path = os.path.join(WORK_DIR, 'theta_train_full_cls.npy')

    if os.path.exists(x_train_path) and os.path.exists(theta_train_path):
        print('\nLoading cached Cl training arrays...')
        x_train     = np.load(x_train_path)
        theta_train = np.load(theta_train_path)
        if x_train.shape[1] != N_SUMMARY:
            print(f'  [WARN] Cached x_train has {x_train.shape[1]} features, '
                  f'expected {N_SUMMARY}. Recomputing...')
            os.remove(x_train_path)
            os.remove(theta_train_path)
            x_train = None
    else:
        x_train = None

    if x_train is None:
        theta_list, x_list = [], []
        for csv_path, offset in CSV_FILES:
            df = pd.read_csv(csv_path)
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc=f'Loading {os.path.basename(csv_path)}'):
                sid        = int(row['sample_id']) + offset
                cache_file = os.path.join(CACHE_DIR, f'x_sample_{sid}.npy')
                if os.path.exists(cache_file):
                    v = np.load(cache_file)
                    if v.shape[0] != N_SUMMARY:
                        os.remove(cache_file)
                        v = None
                else:
                    v = None
                if v is None:
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

    print(f'Training data: {theta_train.shape[0]} sims, '
          f'x={x_train.shape}, theta={theta_train.shape}')

    # ── Fisher correlation diagnostic ─────────────────────────────────────────
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

    # ── Which statistics to run ───────────────────────────────────────────────
    stats_to_run = RUN_STATS if RUN_STATS is not None else list(STAT_MAP.keys())
    print(f'\nWill run MCMC for {len(stats_to_run)} statistics: {stats_to_run}')

    # ── Main loop ─────────────────────────────────────────────────────────────
    results = {}

    for stat_name in stats_to_run:
        print(f'\n{"=" * 60}')
        print(f'  Statistic: {stat_name}')
        print(f'{"=" * 60}')

        feature_idx   = STAT_MAP[stat_name]
        emulator_path = os.path.join(WORK_DIR,
                                     f'gp_emulator_cls_{stat_name}.pkl')
        x_train_stat  = x_train[:, feature_idx].astype(np.float64)
        chain_dir     = os.path.join(CHAINS_DIR, stat_name)
        os.makedirs(chain_dir, exist_ok=True)

        # Fit or load emulator
        emulator = fit_or_load_emulator(
            stat_name, feature_idx, theta_train, x_train, emulator_path)

        print(f'  [{stat_name}] Noise std per Cl bin '
              f'(min/mean/max): '
              f'{np.sqrt(emulator.noise_var_).min():.3e} / '
              f'{np.sqrt(emulator.noise_var_).mean():.3e} / '
              f'{np.sqrt(emulator.noise_var_).max():.3e}')

        # Validation plots
        plot_validation(stat_name, emulator, theta_train,
                        x_train_stat, feature_idx)
        plot_cl_obs_vs_pred(stat_name, emulator, x_obs, feature_idx)
        plot_cl_spectra_training(x_train, theta_train, stat_name, feature_idx)

        # Run MCMC
        info = make_cobaya_info(stat_name, emulator_path,
                                x_obs_path, feature_idx)
        print(f'  [{stat_name}] Starting Cobaya MCMC...')
        try:
            updated_info, sampler = cobaya_run(info)
        except Exception as e:
            import traceback
            print(f'  [{stat_name}] MCMC FAILED: {traceback.format_exc()}')
            continue

        # Collect results
        try:
            gd_sample = sampler.products(
                to_getdist=True, skip_samples=0.3)['sample']

            means  = [float(gd_sample.mean(p))             for p in PARAM_NAMES]
            stdevs = [float(np.sqrt(gd_sample.var(p)))     for p in PARAM_NAMES]

            results[stat_name] = {
                'means':     means,
                'stds':      stdevs,
                'gd_sample': gd_sample,
            }

            print(f'\n  [{stat_name}] Posterior summary:')
            for pname, m, s in zip(PARAM_NAMES, means, stdevs):
                print(f'    {pname:35s}  mean = {m:.4f}  std = {s:.4f}')

            plot_triangle(stat_name, gd_sample)
            plot_marginals(stat_name, gd_sample)

        except Exception:
            import traceback
            print(f'  [{stat_name}] Failed to collect results: '
                  f'{traceback.format_exc()}')
            continue

    # ── Cross-statistic comparison plots ──────────────────────────────────────
    print(f'\n{"=" * 60}')
    print('  Generating cross-statistic comparison plots...')
    print(f'{"=" * 60}')

    if len(results) == 0:
        print('No successful MCMC runs to compare.')
    else:
        plot_all_posteriors(results)
        plot_summary_table(results)

        # ── Print final summary table to stdout ───────────────────────────────
        col_w = 10
        header = f'{"statistic":>15s}'
        for pname in PARAM_NAMES:
            header += f'  {"mean":>{col_w}s}  {"std":>{col_w}s}'
        header += f'   params'
        print('\n' + '=' * len(header))
        print('  MCMC POSTERIOR SUMMARY (Cls)')
        print('=' * len(header))
        subheader = f'{"":>15s}'
        for pname in PARAM_NAMES:
            subheader += f'  {pname[:col_w]:>{col_w}s}  {"":>{col_w}s}'
        print(subheader)
        print('-' * len(header))

        for stat_name, res in results.items():
            row = f'{stat_name:>15s}'
            for m, s in zip(res['means'], res['stds']):
                row += f'  {m:{col_w}.4f}  {s:{col_w}.4f}'
            row += f'   n_bins={len(STAT_MAP[stat_name])}'
            print(row)

        print('=' * len(header))

        # ── Overlay: MCMC (Cls) vs SBI posteriors if available ────────────────
        sbi_results = {}
        for stat_name in results.keys():
            sbi_posterior_path = os.path.join(
                os.path.dirname(WORK_DIR),
                'sbi_Cls',
                f'ili_posterior_{stat_name}.pkl',
            )
            xobs_norm_path = os.path.join(
                os.path.dirname(WORK_DIR),
                'sbi_Cls',
                f'xobs_{stat_name}.npy',
            )
            if os.path.exists(sbi_posterior_path) and \
               os.path.exists(xobs_norm_path):
                try:
                    import torch
                    import threading

                    with open(sbi_posterior_path, 'rb') as f:
                        sbi_post = pk.load(f)

                    xo_norm = np.load(xobs_norm_path)
                    x_t     = torch.from_numpy(xo_norm).float().reshape(1, -1)

                    # Try to sample from the SBI posterior
                    try:
                        members = sbi_post.posteriors
                    except AttributeError:
                        members = [sbi_post]

                    collected = []
                    per_member = max(1, 2000 // len(members))
                    for member in members:
                        result_s, exc = [None], [None]
                        def _sample(m=member):
                            try:
                                s = m.sample((per_member,), x=x_t,
                                             show_progress_bars=False)
                                result_s[0] = s.detach().cpu().numpy()
                            except Exception as e:
                                exc[0] = e
                        t = threading.Thread(target=_sample, daemon=True)
                        t.start()
                        t.join(timeout=60)
                        if result_s[0] is not None:
                            collected.append(result_s[0])

                    if collected:
                        sbi_samples = np.concatenate(collected, axis=0)
                        sbi_results[stat_name] = sbi_samples
                        print(f'  [SBI overlay] Loaded {len(sbi_samples)} '
                              f'samples for {stat_name}')
                except Exception as e:
                    print(f'  [SBI overlay] Could not load {stat_name}: {e}')

        if sbi_results:
            import getdist
            from getdist import MCSamples

            cmap   = plt.get_cmap('tab10')
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            for ax, param, label, lo, hi in zip(
                    axes, PARAM_NAMES, PARAM_LABELS, PRIOR_LOW, PRIOR_HIGH):
                xs     = np.linspace(lo, hi, 400)
                pidx   = PARAM_NAMES.index(param)

                for i, stat_name in enumerate(results.keys()):
                    color_mcmc = cmap(i / max(len(results) - 1, 1))
                    color_sbi  = cmap(i / max(len(results) - 1, 1))

                    # MCMC curve
                    try:
                        gd  = results[stat_name]['gd_sample']
                        p1d = gd.get1DDensity(param)
                        ys  = p1d.Prob(xs)
                        ys /= (ys.max() + 1e-30)
                        ax.plot(xs, ys,
                                color=color_mcmc, lw=2.0, ls='-',
                                label=f'{stat_name} MCMC')
                    except Exception as e:
                        print(f'  [WARN] MCMC curve {stat_name}/{param}: {e}')

                    # SBI curve
                    if stat_name in sbi_results:
                        try:
                            sbi_s  = sbi_results[stat_name]
                            gd_sbi = MCSamples(
                                samples=sbi_s,
                                names=PARAM_NAMES,
                                labels=[l.replace('$', '')
                                        for l in PARAM_LABELS],
                            )
                            p1d_sbi = gd_sbi.get1DDensity(param)
                            ys_sbi  = p1d_sbi.Prob(xs)
                            ys_sbi /= (ys_sbi.max() + 1e-30)
                            ax.plot(xs, ys_sbi,
                                    color=color_sbi, lw=1.5, ls='--',
                                    label=f'{stat_name} SBI')
                        except Exception as e:
                            print(f'  [WARN] SBI curve {stat_name}/{param}: {e}')

                ax.set_xlabel(label,          fontsize=13)
                ax.set_ylabel('Normalised P', fontsize=12)
                ax.set_title(param)
                ax.legend(fontsize=6, ncol=2)

            fig.suptitle('MCMC (solid) vs SBI (dashed) — Cl posteriors',
                         fontsize=13)
            plt.tight_layout()
            out = os.path.join(WORK_DIR, 'mcmc_vs_sbi_cls.png')
            plt.savefig(out, dpi=130, bbox_inches='tight')
            plt.close()
            print(f'\nMCMC vs SBI overlay saved to {out}')

        # ── Per-spectrum Cl power spectrum plot for reference run ─────────────
        fig, axes = plt.subplots(2, len(CL_SPECS) // 2 + len(CL_SPECS) % 2,
                                 figsize=(4 * (len(CL_SPECS) // 2 + 1), 8),
                                 squeeze=False)
        axes_flat = axes.flatten()
        for i, (label, _, _) in enumerate(CL_SPECS):
            sl  = slice(i * N_ELL_BINS_ACTUAL, (i + 1) * N_ELL_BINS_ACTUAL)
            ax  = axes_flat[i]
            ax.plot(ELL_CENTRES, x_obs[sl], 'k-o', ms=4, lw=1.5)
            ax.axhline(0, color='grey', lw=0.8, ls='--')
            ax.set_xscale('log')
            ax.set_xlabel(r'$\ell$',                     fontsize=10)
            ax.set_ylabel(r'$\ell(\ell+1)C_\ell/2\pi$', fontsize=9)
            ax.set_title(label,                          fontsize=11)

        for j in range(len(CL_SPECS), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle('Reference run Cl band-powers', fontsize=13)
        plt.tight_layout()
        out = os.path.join(WORK_DIR, 'reference_cls.png')
        plt.savefig(out, dpi=130, bbox_inches='tight')
        plt.close()
        print(f'Reference Cl plot saved to {out}')

    print('\nAll done.')
