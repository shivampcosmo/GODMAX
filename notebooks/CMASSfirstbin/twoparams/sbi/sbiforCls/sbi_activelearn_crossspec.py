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
import yaml
import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d

# --- Path Setup for ltu-ili & GODMAX ---
sys.path.append('/work/hdd/bdne/aacharya2/ltu-ili')
from ili.dataloaders import StaticNumpyLoader
from ili.inference import InferenceRunner
from ili.utils import load_nde_sbi, Uniform
from ili.validation.metrics import PosteriorCoverage

curr_path = Path().absolute()
project_base = curr_path.parents[4] # Adjust this if running from a different relative directory
abs_path_data = os.path.abspath(project_base / "data")
abs_path_src = os.path.abspath(project_base / "src")
abs_path_params = os.path.abspath(project_base / "param_files")
sys.path.insert(0, abs_path_src)

from base_class import base_class
from get_radial_profiles import Profiles
from get_Pkzs import get_Pkz
from get_Cls import get_Cl
from jax_cosmo import Cosmology
from jax_cosmo.background import radial_comoving_distance

# =============================================================================
# 1. CONFIGURATION & BINNING
# =============================================================================
BASE_DIR = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin'
CSV_FILES = [
    '/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/twoparams/wl/lhs_samples.csv'
]
CACHE_DIR = 'sample_cls_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

NSIDE = 512
LMAX = 3 * NSIDE - 1
L_RANGE = np.arange(2, LMAX + 1)
PIXWIN = hp.pixwin(NSIDE, lmax=LMAX)[2:]

# Define 8 log-spaced bins between l=100 and l=1000
L_MIN, L_MAX_BIN = 100, 1000
NUM_BINS = 8
ELL_EDGES = np.logspace(np.log10(L_MIN), np.log10(L_MAX_BIN), NUM_BINS + 1)

def bin_spectrum(cl_array, l_array):
    """Averages the power spectrum within the defined ell bins."""
    binned_cl = np.zeros(NUM_BINS)
    for i in range(NUM_BINS):
        mask = (l_array >= ELL_EDGES[i]) & (l_array < ELL_EDGES[i+1])
        binned_cl[i] = np.mean(cl_array[mask])
    return binned_cl

# =============================================================================
# 2. TRUE OBSERVATION (ANALYTIC THEORY)
# =============================================================================
def get_theory_observation():
    """Runs the GODMAX pipeline to generate the analytic true observation."""
    print("Generating Analytic True Observation (x_obs)...")
    default_data = yaml.safe_load(open(abs_path_params + '/params_anshuman.yaml'))
    sim_params_dict = default_data.get('sim_params', {})
    halo_params_dict = default_data.get('halo_params', {})
    analysis_dict = default_data.get('analysis', {})
    other_params_dict = default_data.get('other_params', {})

    cosmo_params_dict = {
        'w0': -1.0, 'flat': True, 'H0': 67.11, 'Om0': 0.3175,
        'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624
    }
    sim_params_dict['cosmo'] = cosmo_params_dict
    h = cosmo_params_dict['H0'] / 100.

    cosmo_jax = Cosmology(
        Omega_c=cosmo_params_dict['Om0'] - cosmo_params_dict['Ob0'],
        Omega_b=cosmo_params_dict['Ob0'], h=h, sigma8=cosmo_params_dict['sigma8'],
        n_s=cosmo_params_dict['ns'], Omega_k=0., w0=cosmo_params_dict['w0'], wa=0.
    )

    # Setup redshift integration and galaxy number density
    Z_MIN, Z_MAX = 0.001, 2.1
    zarray_lens = np.linspace(Z_MIN, Z_MAX, 31)
    zmin_gal, zmax_gal = 0.3, 0.5
    zmin_max_edges = np.linspace(zmin_gal, zmax_gal + 0.001, 21)
    zcen = 0.5 * (zmin_max_edges[1:] + zmin_max_edges[:-1])

    nz_f = (4.0 / 3.0) * jnp.pi * (
        radial_comoving_distance(cosmo_jax, jnp.atleast_1d(1.0 / (1.0 + zmin_max_edges[1:])))**3
        - radial_comoving_distance(cosmo_jax, jnp.atleast_1d(1.0 / (1.0 + zmin_max_edges[:-1])))**3
    )
    nz_f = np.array(nz_f)
    nz_f[np.where((zcen < zmin_gal) | (zcen > zmax_gal))[0]] = 0.0
    nz_f_norm = nz_f / np.trapezoid(nz_f, zcen)
    hist_z = interp1d(zcen, nz_f_norm, fill_value=0.0, bounds_error=False)(zarray_lens)

    # Reference N-body density
    fsky = 1.0 # Assume full sky for volume calculation initially
    chi_min = float(radial_comoving_distance(cosmo_jax, jnp.atleast_1d(1.0 / (1.0 + zmin_gal)))[0])
    chi_max = float(radial_comoving_distance(cosmo_jax, jnp.atleast_1d(1.0 / (1.0 + zmax_gal)))[0])
    V_comoving = (4.0 / 3.0) * np.pi * (chi_max**3 - chi_min**3) * fsky
    nbar_sim = 1928248 / V_comoving # Hardcoding reference galaxy count for nbar

    analysis_dict['nbar_gal_comoving_zarray'] = zarray_lens
    analysis_dict['nbar_gal_comoving_val'] = np.full_like(zarray_lens, nbar_sim)
    analysis_dict['nz_lens_info_dict'] = {'z_array_lens': zarray_lens, 'nbins_lens': 1, 'nz0': hist_z}
    analysis_dict['is_cmb_lensing'] = True
    analysis_dict['nz_source_info_dict'] = {'z_array_source': jnp.ones(1), 'nbins': 1, 'nz0': jnp.ones(1)}
    other_params_dict['Delta_z_bias_array'] = jnp.zeros(1)
    other_params_dict['mult_shear_bias_array'] = jnp.zeros(1)

    ks = np.geomspace(1e-2, 50, 80)
    analysis_dict['k_array_survey'] = jnp.array(ks)
    
    lmin_th, lmax_th, dl_log_array = 80.0, 8800.0, 0.23025851
    l_array_all = np.exp(np.arange(np.log(lmin_th), np.log(lmax_th), dl_log_array))
    l_array_survey = (l_array_all[1:] + l_array_all[:-1]) / 2.
    halo_params_dict['ell_array'] = jnp.array(l_array_survey)
    analysis_dict['l_array_survey'] = jnp.array(l_array_survey)
    analysis_dict['dl_array_survey'] = jnp.array(l_array_all[1:] - l_array_all[:-1])
    analysis_dict['symbolic_pk'] = True
    analysis_dict['symbolic_hmf'] = True

    halo_params_dict.update({
        'rmin': 0.005, 'rmax': 10.0, 'nr': 48,
        'zmin': Z_MIN, 'zmax': Z_MAX, 'nz': 31,
        'lg10_Mmin': 11.75, 'lg10_Mmax': 16.0, 'nM': 32
    })
    
    analysis_dict['beam_fwhm_arcmin'] = 1e-5 # Force unsmoothed output for correct asymmetric matching

    base_test = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
    Prof_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, base_class_obj=base_test)
    
    hard_mask_2d = jnp.tile(jnp.where(Prof_test.M_array >= 10**12.75, 1.0, 0.0), (halo_params_dict['nz'], 1))
    Prof_test.Ncen_mat = jnp.stack([Prof_test.get_Ncen(jz, jnp.arange(halo_params_dict['nM'])) for jz in range(halo_params_dict['nz'])]) * hard_mask_2d
    Prof_test.Nsat_mat = jnp.stack([Prof_test.get_Nsat(jz, jnp.arange(halo_params_dict['nM'])) for jz in range(halo_params_dict['nz'])]) * hard_mask_2d

    pkz_test = get_Pkz(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Profiles_obj=Prof_test)
    Cls_test = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Pkz_obj=pkz_test)

    # Unit Corrections
    from astropy import constants as const
    import astropy.units as u
    ne0_cm3 = (1.878e-29 * h**2) * cosmo_params_dict['Ob0'] * (1 - 0.24 / 2) / const.m_p.to(u.g).value
    z_mean_shell = 0.4
    
    ell_theory = np.array(Cls_test.ell_array)
    Cl_gg_th = np.array(Cls_test.Cl_gal_gal_tot_mat[:, 0, 0])
    Cl_gy_th = np.array(Cls_test.Cl_gal_y_tot_mat[:, 0])
    Cl_gtau_th = np.array(Cls_test.Cl_gal_tau_tot_mat[:, 0]) * ne0_cm3 * (1.0 + z_mean_shell)**3
    Cl_gkappa_th = np.array(Cls_test.Cl_gal_kappa_tot_mat[:, 0, 0]) * (h**2)

    # Apply Simulation Beam
    sigma_sim = hp.nside2resol(NSIDE) / np.sqrt(8. * np.log(2.))
    beam_ell = np.exp(-0.5 * L_RANGE * (L_RANGE + 1) * sigma_sim**2)

    th_vec = []
    for th_arr, apply_beam in [(Cl_gg_th, False), (Cl_gy_th, True), (Cl_gkappa_th, True), (Cl_gtau_th, True)]:
        th_interp = interp1d(ell_theory, th_arr, bounds_error=False, fill_value=0.0)(L_RANGE)
        if apply_beam: th_interp *= beam_ell
        th_vec.extend(bin_spectrum(th_interp, L_RANGE))
        
    return np.array(th_vec, dtype=np.float32)

if not os.path.exists('x_obs_cls.npy'):
    x_obs = get_theory_observation()
    np.save('x_obs_cls.npy', x_obs)
else:
    x_obs = np.load('x_obs_cls.npy')
    
# =============================================================================
# 3. SIMULATION EXTRACTION
# =============================================================================
def extract_binned_cls(path):
    """Loads N-body maps, calculates Cls, and bins them."""
    pattern = os.path.join(path, "allmaps_nside512_z*_split*.pkl")
    files = glob.glob(pattern)
    if len(files) == 0: return None
    
    npix = 12 * NSIDE**2
    maps = {k: np.zeros(npix, dtype=np.float32) for k in ['kappa', 'ymap', 'tau', 'gal']}
    total_gals = 0.0

    for f in files:
        with open(f, 'rb') as h:
            data = pk.load(h)
            maps['ymap'] += np.nan_to_num(data.get('map_ymap', 0))
            maps['kappa'] += np.nan_to_num(data.get('map_kappa', 0))
            maps['tau'] += np.nan_to_num(data.get('map_tau', 0))
            mock_gals_dict = data.get('mock_gals_all', {})
            for chunk in mock_gals_dict.values():
                if chunk is not None and chunk.size > 0:
                    ra, dec = chunk[:, 0], chunk[:, 1]
                    valid = ~(np.isnan(ra) | np.isnan(dec))
                    ra, dec = np.mod(ra[valid], 360.0), np.clip(dec[valid], -90.0, 90.0)
                    total_gals += len(ra)
                    if len(ra) > 0:
                        pix = hp.ang2pix(NSIDE, ra, dec, lonlat=True)
                        maps['gal'] += np.bincount(pix, minlength=npix)

    mask = (maps['kappa'] != 0.0)
    fsky = np.sum(mask) / npix if np.sum(mask) > 0 else 1e-6

    if total_gals > 0:
        mean_gal = np.sum(maps['gal'][mask]) / np.sum(mask)
        maps['gal'] = np.where(mask, (maps['gal'] / mean_gal) - 1.0, 0.0)
        shot_noise = (4 * np.pi * fsky) / total_gals
    else:
        shot_noise = 0.0

    vec = []
    # Order must match Theory: gg, gy, gk, gtau
    for map_key, do_shot_noise in [('gal', True), ('ymap', False), ('kappa', False), ('tau', False)]:
        cl_raw = hp.anafast(maps['gal'], maps[map_key], lmax=LMAX)[2:] / fsky
        if do_shot_noise: cl_raw -= shot_noise
        cl_corr = cl_raw / (PIXWIN**2)
        vec.extend(bin_spectrum(cl_corr, L_RANGE))
        
    return np.array(vec, dtype=np.float32)

# =============================================================================
# 4. LOAD TRAINING DATA
# =============================================================================
theta_train, x_train = [], []
for csv in CSV_FILES:
    df = pd.read_csv(csv)
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading samples"):
        sid = int(row['sample_id'])
        cache_file = os.path.join(CACHE_DIR, f"x_cls_sample_{sid}.npy")

        if os.path.exists(cache_file):
            v = np.load(cache_file)
        else:
            s_path = os.path.join(BASE_DIR, "twoparams", "wl", f"sample_{sid}")
            v = extract_binned_cls(s_path)
            if v is not None: np.save(cache_file, v)

        if v is not None:
            theta_train.append([row['theta_ej_0'], row['mu_beta']])
            x_train.append(v)

x_train = np.array(x_train).astype(np.float32)
theta_train = np.array(theta_train).astype(np.float32)

np.save('theta.npy', theta_train)

# =============================================================================
# 5. LTU-ILI TRAINING LOOP
# =============================================================================
# Vector size is 32 (8 bins * 4 probes)
stat_map = {
    'gg': list(range(0, 8)),
    'gy': list(range(8, 16)),
    'gkappa': list(range(16, 24)),
    'gtau': list(range(24, 32)),
    'JOINT': list(range(32)),
}

np.random.seed(42)
all_indices = np.arange(len(theta_train))
np.random.shuffle(all_indices)

val_idx = all_indices[:2] # Small val set given 30 samples total
train_idx = all_indices

theta_val = theta_train[val_idx]
theta_train_set = theta_train[train_idx]

#standardize the statistics
x_mean = np.mean(x_train, axis=0)
x_std = np.std(x_train, axis=0)
x_train = (x_train - x_mean) / (x_std + 1e-6)
x_obs = (x_obs - x_mean) / (x_std + 1e-6)

for name, idx in stat_map.items():
    print(f"\n--- Training NPE: {name} ---")

    xt_full = x_train[:, idx]
    xt_train = xt_full[train_idx]
    xt_val = xt_full[val_idx]
    xo = x_obs[idx]

    np.save(f'x_{name}.npy', xt_train)
    np.save(f'xobs_{name}.npy', xo)
    np.save(f'theta_train_{name}.npy', theta_train_set)

    loader = StaticNumpyLoader(
        in_dir='./',
        x_file=f'x_{name}.npy',
        theta_file=f'theta_train_{name}.npy',
        xobs_file=f'xobs_{name}.npy'
    )

    nets = load_nde_sbi(engine='NPE', model='nsf', repeats=2, hidden_features=32, num_transforms=3) + \
           load_nde_sbi(engine='NPE', model='maf', repeats=6, hidden_features=32, num_transforms=3)

    runner = InferenceRunner.load(
        backend='sbi', engine='NPE',
        prior=Uniform(low=[1.0, 0.01], high=[6.0, 1.5]),
        nets=nets, out_dir=Path(f'./sbi_logs_{name}')
    )

    posterior, _ = runner(loader)

    with open(f'ili_posterior_{name}.pkl', 'wb') as f:
        pk.dump(posterior, f)

# =============================================================================
# 6. ROUND 2 PROPOSAL
# =============================================================================
with open('ili_posterior_JOINT.pkl', 'rb') as f:
    joint_posterior = pk.load(f)

xo_tensor = torch.from_numpy(x_obs).float().reshape(1, -1)
next_theta = joint_posterior.sample((20,), x=xo_tensor).detach().cpu().numpy()

pd.DataFrame(next_theta, columns=['theta_ej_0', 'mu_beta']).to_csv('round2_samples.csv', index_label='sample_id')
print("\nFinished. Generated round2_samples.csv")
