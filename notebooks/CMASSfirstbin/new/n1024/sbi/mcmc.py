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
CACHE_DIR = os.path.join(WORK_DIR, 'sample_vector_cache')

CSV_FILES = [
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/lhs_samples.csv', 0),
    ('/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/round2_samples.csv', 500),
]

NSIDE        = 1024
SCALES       = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
PRIOR_LOW    = [1.0, -0.3]
PRIOR_HIGH   = [6.0,  0.0]
PARAM_LABELS = [r'$$\theta_{ej,0}$$', r'$$\nu_{\theta_{ej}}^{M}$$']
PARAM_NAMES  = ['theta_ej_0', 'nu_theta_ej_M']

STAT_MAP = {
#    'g2y':         [0,  6, 12, 18, 24, 30],
    'g2tau':       [1,  7, 13, 19, 25, 31],
#    'g2kappa':     [2,  8, 14, 20, 26, 32],
#    'gy':          [3,  9, 15, 21, 27, 33],
#    'gtau':        [4, 10, 16, 22, 28, 34],
#    'gkappa':      [5, 11, 17, 23, 29, 35],
#    'y_total':     [0,  6, 12, 18, 24, 30,
#                    3,  9, 15, 21, 27, 33],
    'tau_total':   [1,  7, 13, 19, 25, 31,
                    4, 10, 16, 22, 28, 34],
#    'kappa_total': [2,  8, 14, 20, 26, 32,
#                    5, 11, 17, 23, 29, 35],
    'all_3pt':     [0,  6, 12, 18, 24, 30,
                    1,  7, 13, 19, 25, 31,
                    2,  8, 14, 20, 26, 32],
    'all_2pt':     [3,  9, 15, 21, 27, 33,
                    4, 10, 16, 22, 28, 34,
                    5, 11, 17, 23, 29, 35],
    'JOINT':       list(range(36)),
}

# Set to a list like ['gtau', 'gkappa'] to run only a subset, or None for all
RUN_STATS     = None

GP_N_RESTARTS = 5
CHAINS_DIR    = os.path.join(WORK_DIR, 'chains')
MCMC_SAMPLES  = 5_000
RMINUS1_STOP  = 0.02

# =============================================================================
# SUMMARY STATISTIC EXTRACTION
# =============================================================================

def extract_moments(path):
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
                pix  = hp.ang2pix(NSIDE, ra_gal[mask], dec_gal[mask], lonlat=True)
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
# GP EMULATOR
# =============================================================================

class GPEmulator:
    """
    One GP per output feature. Both theta (inputs) and x (outputs) are
    z-scored before fitting so the kernel operates in O(1) space regardless
    of the raw amplitudes (~1e-5 for tau/y statistics).

    The WhiteKernel fitted amplitude gives a per-feature noise variance used
    as the diagonal of the observational covariance in the likelihood. When
    the WhiteKernel hits its lower bound the noise estimate is unreliable, so
    we fall back to the LOO residual variance as a floor.
    """

    def __init__(self, n_restarts: int = 5):
        self.n_restarts    = n_restarts
        self.gps_          = []
        self.theta_scaler_ = StandardScaler()
        self.x_scaler_     = StandardScaler()
        self.noise_var_    = None   # shape (n_features,), raw units

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
                noise_scaled   = max(loo_var_scaled, LOO_BOUND)
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
    Gaussian log-likelihood wrapping a GP emulator.

    Class attributes (overridable via the Cobaya info dict):
        emulator_path : str   path to pickled GPEmulator
        x_obs_path    : str   path to .npy observed summary vector (length 36)
        feature_idx   : list  indices into the full 36-vector to use
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

        print(f'[GPEmulatorLikelihood] {len(self.feature_idx)} features loaded.')
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

def fit_or_load_emulator(stat_name, feature_idx, theta_train, x_train, emulator_path):
    if os.path.exists(emulator_path):
        print(f'  [{stat_name}] Loading cached GP emulator from {emulator_path}')
        return GPEmulator.load(emulator_path)

    print(f'  [{stat_name}] Fitting GP emulator...')
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
                'input_params':  PARAM_NAMES,       # <-- fixes the routing error
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
    x_pred = emulator.predict(theta_train.astype(np.float64))
    n_feat = len(feature_idx)
    ncols  = min(n_feat, 6)
    nrows  = (n_feat + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for j in range(n_feat):
        ax     = axes_flat[j]
        y_sim  = x_train_stat[:, j]
        y_pred = x_pred[:, j]
        lo     = min(y_sim.min(), y_pred.min())
        hi     = max(y_sim.max(), y_pred.max())
        ax.scatter(y_sim, y_pred, s=8, alpha=0.5, color='steelblue')
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('Simulated',    fontsize=9)
        ax.set_ylabel('GP predicted', fontsize=9)
        ax.set_title(f'feat {j}  (global idx {feature_idx[j]})', fontsize=9)

    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f'GP emulator validation — {stat_name}', fontsize=12)
    plt.tight_layout()
    out = os.path.join(WORK_DIR, f'gp_validation_{stat_name}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  [{stat_name}] Validation plot saved to {out}')


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
        xs  = np.linspace(lo, hi, 400)
        # param is e.g. 'theta_ej_0' — a valid GetDist parameter name
        p1d = gd_sample.get1DDensity(param)
        ys  = p1d.Prob(xs)
        ys_norm = ys / (ys.max() + 1e-30)
        map_val = xs[np.argmax(ys)]
        ax.plot(xs, ys_norm, color='steelblue', lw=2)
        ax.fill_between(xs, ys_norm, alpha=0.25, color='steelblue')
        ax.axvline(map_val, color='steelblue', ls='--', lw=1.2,
                   label=f'MAP = {map_val:.3f}')
        # label (the LaTeX string) is only used for the axis label, never
        # passed to GetDist — this is safe
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
                p1d = gd.get1DDensity(param)
                ys  = p1d.Prob(xs)
                ys_norm = ys / (ys.max() + 1e-30)
                ax.plot(xs, ys_norm, label=stat_name, color=color, lw=1.8)
            except Exception as e:
                print(f'  [WARN] Could not plot {stat_name}/{param}: {e}')
        ax.set_xlabel(label,          fontsize=13)
        ax.set_ylabel('Normalised P', fontsize=12)
        ax.set_title(param)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle('MCMC posteriors — all statistics', fontsize=14)
    plt.tight_layout()
    out = os.path.join(WORK_DIR, 'posteriors_all_stats.png')
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
        pidx  = PARAM_NAMES.index(param)
        means = [results[s]['means'][pidx] for s in stat_names]
        stds  = [results[s]['stds'][pidx]  for s in stat_names]
        ax.bar(x_pos, means, yerr=stds, capsize=4,
               color='steelblue', alpha=0.7, ecolor='black')
        ax.axhline(lo, ls='--', lw=0.8, color='grey')
        ax.axhline(hi, ls='--', lw=0.8, color='grey')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stat_names, rotation=40, ha='right', fontsize=9)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'Posterior mean ± std  [{param}]')
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))

    fig.suptitle('MCMC posterior summary — all statistics', fontsize=13)
    plt.tight_layout()
    out = os.path.join(WORK_DIR, 'summary_table_all_stats.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Summary table saved to {out}')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    os.makedirs(CACHE_DIR,  exist_ok=True)
    os.makedirs(CHAINS_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1.  x_obs                                                           #
    # ------------------------------------------------------------------ #
    x_obs_path = os.path.join(WORK_DIR, 'x_obs.npy')
    if os.path.exists(x_obs_path):
        print('Loading cached x_obs...')
        x_obs = np.load(x_obs_path)
    else:
        print('Extracting x_obs from reference run...')
        x_obs = extract_moments(os.path.join(BASE_DIR, 'reference_run'))
        if x_obs is None:
            raise RuntimeError('No pkl files found in reference_run.')
        np.save(x_obs_path, x_obs)
    print(f'  x_obs shape: {x_obs.shape}')

    # ------------------------------------------------------------------ #
    # 2.  Training data                                                   #
    # ------------------------------------------------------------------ #
    x_train_path     = os.path.join(WORK_DIR, 'x_train_full.npy')
    theta_train_path = os.path.join(WORK_DIR, 'theta_train_full.npy')

    if os.path.exists(x_train_path) and os.path.exists(theta_train_path):
        print('Loading cached training arrays...')
        x_train     = np.load(x_train_path)
        theta_train = np.load(theta_train_path)
    else:
        print('Extracting simulation summaries...')
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
        np.save(x_train_path,     x_train)
        np.save(theta_train_path, theta_train)

    print(f'Training data: {theta_train.shape[0]} sims, '
          f'x={x_train.shape}, theta={theta_train.shape}')

    # ------------------------------------------------------------------ #
    # 3.  Which statistics to run                                         #
    # ------------------------------------------------------------------ #
    stats_to_run = RUN_STATS if RUN_STATS is not None else list(STAT_MAP.keys())
    print(f'\nWill run MCMC for {len(stats_to_run)} statistics: {stats_to_run}')

    # ------------------------------------------------------------------ #
    # 4.  Main loop                                                       #
    # ------------------------------------------------------------------ #
    results = {}

    for stat_name in stats_to_run:
        print(f'\n{"=" * 60}')
        print(f'  Statistic: {stat_name}')
        print(f'{"=" * 60}')

        feature_idx   = STAT_MAP[stat_name]
        emulator_path = os.path.join(WORK_DIR, f'gp_emulator_{stat_name}.pkl')
        x_train_stat  = x_train[:, feature_idx].astype(np.float64)
        chain_dir     = os.path.join(CHAINS_DIR, stat_name)
        os.makedirs(chain_dir, exist_ok=True)

        # 4a. Fit or load emulator
        emulator = fit_or_load_emulator(
            stat_name, feature_idx, theta_train, x_train, emulator_path)

        print(f'  [{stat_name}] Noise std per feature:')
        for i, nv in enumerate(emulator.noise_var_):
            print(f'    feat {i:2d}: noise_std = {np.sqrt(nv):.4e}')

        # 4b. Emulator validation plot
        plot_validation(stat_name, emulator, theta_train, x_train_stat, feature_idx)

        # 4c. Build Cobaya info and run MCMC
        info = make_cobaya_info(stat_name, emulator_path, x_obs_path, feature_idx)
        print(f'  [{stat_name}] Starting Cobaya MCMC...')
        try:
            updated_info, sampler = cobaya_run(info)
        except Exception as e:
            print(f'  [{stat_name}] MCMC FAILED: {e}')
            continue

        # 4d. Collect results
        try:
            gd_sample = sampler.products(to_getdist=True, skip_samples=0.3)['sample']

            # Fetch by name to guarantee correct parameter ordering
            means  = [float(gd_sample.mean(p))                    for p in PARAM_NAMES]
            stdevs = [float(np.sqrt(gd_sample.var(p)))            for p in PARAM_NAMES]

            results[stat_name] = {
                'means':     means,
                'stds':      stdevs,
                'gd_sample': gd_sample,
            }

            print(f'\n  [{stat_name}] Posterior summary:')
            for pname, plabel, m, s in zip(PARAM_NAMES, PARAM_LABELS, means, stdevs):
                print(f'    {pname:35s}  mean = {m:.4f}  std = {s:.4f}')

            # 4e. Per-statistic plots
            plot_triangle(stat_name, gd_sample)
            plot_marginals(stat_name, gd_sample)

        except Exception as e:
            import traceback
            print(f'  [{stat_name}] Failed to collect/plot results: {traceback.format_exc()}')
            continue
    # ------------------------------------------------------------------ #
    # 5.  Cross-statistic comparison plots                                #
    # ------------------------------------------------------------------ #
    print(f'\n{"=" * 60}')
    print(f'  Generating cross-statistic comparison plots...')
    print(f'{"=" * 60}')

    if len(results) == 0:
        print('No successful MCMC runs to compare.')
    else:
        plot_all_posteriors(results)
        plot_summary_table(results)

        # Print final summary table to stdout
        print(f'\n  [{stat_name}] Posterior summary:')
        for pname, plabel, m, s in zip(PARAM_NAMES, PARAM_LABELS, means, stdevs):
            # pname for machine readability, plabel only for display
            print(f'    {pname:35s}  mean = {m:.4f}  std = {s:.4f}')
        print(header)
        print('-' * len(header))
        for stat_name, res in results.items():
            row = f'{stat_name:>15s}'
            for m, s in zip(res['means'], res['stds']):
                row += f'  {m:8.4f}  {s:7.4f}'
            print(row)

    print('\nAll done.')
