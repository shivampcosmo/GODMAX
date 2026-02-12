import os
import sys
import pathlib
import pickle as pk
import yaml
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax_cosmo.background import radial_comoving_distance
from jax_cosmo import Cosmology
from scipy.interpolate import interp1d
from tqdm import tqdm
from deepmerge import always_merger

# --- PATH LOGIC ---
curr_path = pathlib.Path().absolute()
project_base = curr_path.parents[2] 
abs_path_data = os.path.abspath(project_base / "data")
abs_path_src = os.path.abspath(project_base / "src")
abs_path_params = os.path.abspath(project_base / "param_files")
sys.path.insert(0, abs_path_src)

from base_class import base_class
from get_radial_profiles import Profiles
from get_Pkzs import get_Pkz
from get_Cls import get_Cl

nside = 512
lmax = 3*nside - 1

# --- PARAMETER MERGE ---
default_data = yaml.safe_load(open(abs_path_params + '/params_default.yaml'))
new_data = yaml.safe_load(open(abs_path_params + '/xCMASS/params_fit_test.yaml'))
merged_data = always_merger.merge(default_data, new_data)
sim_p, halo_p, anal_p, other_p = merged_data['sim_params'], merged_data['halo_params'], merged_data['analysis'], merged_data['other_params']

# Use explicit cosmology from your run script
cosmo_dict = {'w0': -1.0, 'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624}
sim_p['cosmo'] = cosmo_dict
sim_p['init_power'] = True

# --- THEORY ALIGNMENT (z=0.3 to 0.5) ---
z_array = np.linspace(0.001, 1.6, 200)
hist_z = np.zeros_like(z_array)
hist_z[(z_array >= 0.3) & (z_array <= 0.5)] = 1.0
hist_z /= np.trapz(hist_z, z_array)

df_pk = pk.load(open(f"{abs_path_data}/CMASS/measure_cmass_planck.pk", 'rb'))
leff = df_pk['leff']

anal_p.update({
    'nz_lens_info_dict': {'z_array_lens': z_array, 'nbins_lens': 1, 'nz0': np.maximum(hist_z, 1e-10)},
    'is_cmb_lensing': True, 'l_array_survey': jnp.array(leff), 'symbolic_pk': True
})

# --- COMPUTE THEORY ---
base_test = base_class(sim_p, halo_p, anal_p, other_p)
pkz_test = get_Pkz(sim_p, halo_p, anal_p, other_p, Profiles_obj=Profiles(sim_p, halo_p, anal_p, other_p, base_class_obj=base_test))
Cls_test = get_Cl(sim_p, halo_p, anal_p, other_p, Pkz_obj=pkz_test)

# --- AGGREGATION ---
data_dir = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/reference_run'
pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
npix = 12 * nside**2
maps = {k: np.zeros(npix) for k in ['kappa', 'ymap', 'tau', 'gal']}

for fname in tqdm(pkl_files, desc="Aggregating Sim"):
    with open(os.path.join(data_dir, fname), 'rb') as f:
        res = pk.load(f)
    # Summing across all snapshots to build the shell signal
    maps['kappa'] += np.nan_to_num(res.get('map_kappa', 0))
    maps['ymap']  += np.nan_to_num(res.get('map_ymap', 0))
    maps['tau']   += np.nan_to_num(res.get('map_tau', 0))
    for chunk in res['mock_gals_all'].values():
        if chunk is not None:
            g_pix = hp.ang2pix(nside, np.array(chunk[:,0]), np.array(chunk[:,1]), lonlat=True)
            maps['gal'] += np.bincount(g_pix, minlength=npix)

# --- MASKED NORMALIZATION & SHOT NOISE ---
mask = (maps['gal'] > 0)
fsky = np.sum(mask) / npix
n_gal_total = np.sum(maps['gal'])
mean_gal_masked = n_gal_total / np.sum(mask)

# Convert to overdensity delta_g = (n/nbar) - 1
maps['gal'] = np.where(mask, (maps['gal'] / mean_gal_masked) - 1.0, 0.0)

# Theoretical shot noise floor corrected for fsky
shot_noise = (4 * np.pi * fsky) / n_gal_total

os.makedirs('plots_tvss', exist_ok=True)
with open('plots_tvss/validation_stats.txt', 'w') as f:
    f.write(f"fsky: {fsky}\nshot_noise: {shot_noise}\ntotal_gals: {n_gal_total}\n")

# --- PLOTTING & CHI2 ---
pixwin = hp.pixwin(nside, lmax=lmax)
stats_cycle = [
    ('gg', Cls_test.Cl_gal_gal_tot_mat[:, 0, 0], maps['gal'], maps['gal']),
    ('gy', Cls_test.Cl_gal_y_tot_mat[:, 0],     maps['gal'], maps['ymap']),
    ('gtau', Cls_test.Cl_gal_tau_tot_mat[:, 0], maps['gal'], maps['tau']),
    ('gkappa', Cls_test.Cl_gal_kappa_tot_mat[:, 0, 0], maps['gal'], maps['kappa'])
]

chi2_results = {}
for label, theory, m1, m2 in stats_cycle:
    # 1. Compute Raw Cl and correct for fsky
    cl_sim_raw = hp.anafast(m1, m2, lmax=lmax)[2:lmax+1]
    cl_sim = cl_sim_raw / (pixwin[2:lmax+1]**2) / fsky
    
    # 2. Flatten gg slope using shot noise subtraction
    if label == 'gg':
        cl_sim = cl_sim - shot_noise
    
    # 3. Interpolate Theory to Sim grid
    th_interp = interp1d(leff, theory[:len(leff)], bounds_error=False, fill_value="extrapolate")(np.arange(2, lmax+1))
    
    # 4. Calculate Reduced Chi-Squared (metric for goodness of fit)
    variance = (0.1 * th_interp)**2 # Metric: check consistency within 10% tolerance
    chi2 = np.sum((cl_sim - th_interp)**2 / variance) / (lmax-1)
    chi2_results[label] = chi2

    plt.figure(figsize=(10, 8))
    plt.loglog(np.arange(2, lmax+1), np.abs(cl_sim), 'o', markersize=2, alpha=0.3, label=f'Sim (Redchi2={chi2:.2e})')
    plt.loglog(leff, theory[:len(leff)], color='blue', lw=2, label='Theory (Aligned z=0.3-0.5)')
    plt.title(f"Comparison {label}"); plt.xlabel(r'$\ell$'); plt.ylabel(r'$C_\ell$')
    plt.xlim(100, 3e3); plt.grid(True, alpha=0.2); plt.legend()
    plt.savefig(f'plots_tvss/thvssim_{label}.png', dpi=300); plt.close()

print("\n--- Final Chi2 Results ---")
for k, v in chi2_results.items(): print(f"{k}: {v:.4e}")
