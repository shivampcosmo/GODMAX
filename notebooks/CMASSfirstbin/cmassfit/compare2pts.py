import os
import sys
import pathlib
import pickle as pk
import yaml
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.interpolate import interp1d
from tqdm import tqdm

# =============================================================================
# 1. SETUP & PATHS
# =============================================================================
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
lmax = 3 * nside - 1
l_range = np.arange(2, lmax + 1)

# =============================================================================
# 2. PARAMETER LOADING & STABILITY
# =============================================================================
default_data = yaml.safe_load(open(abs_path_params + '/params_default.yaml'))
sim_p, halo_p, anal_p, other_p = default_data['sim_params'], default_data['halo_params'], default_data['analysis'], default_data['other_params']

# Match Simulation Cosmology
sim_p['cosmo'] = {'w0': -1.0, 'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624}
sim_p['init_power'] = True

# USE STABLE HOD (TEST2)
sim_p.update({
    'log10M1_fshmr': 12.804, 'log10M1_a_fshmr': -0.475, 'log10Mstar0_fshmr': 10.72,
    'log10Mstar0_a_fshmr': 0.55, 'beta_fshmr': 0.4, 'beta_a_fshmr': 0.2,
    'delta_fshmr': 0.4419, 'delta_a_fshmr': -0.49, 'gamma_fshmr': 0.3442,
    'gamma_a_fshmr': -0.436, 'siglogMstar_Ncen': 0.78853, 'alphasat_Nsat': 1.8972
})

anal_p.update({
    'backreaction': False,           # Disable to stop numerical ringing
    'calc_nfw_only': False,          # Enable galaxies
    'num_points_trapz_int': 64,
    'num_points_gal_cal': 64
})

# Remove solver target to prevent NaNs
if 'nbar_gal_comoving_val' in anal_p:
    del anal_p['nbar_gal_comoving_val'] 

halo_p.update({
    'lg10_Mmin': 11.0, 'lg10_Mmax': 15.5,
    'nM': 32, 'nz': 32, 'rmax': 16.0,
    'conc_model': 'Diemer15',        # Stable model
    'zmax': 0.8                      
})

# Redshift Distribution
z_array = np.linspace(0.001, 1.6, 200)
hist_z = np.zeros_like(z_array)
mask_z = (z_array >= 0.3) & (z_array <= 0.5)
hist_z[mask_z] = 1.0
hist_z /= np.trapz(hist_z, z_array)

anal_p.update({
    'nz_lens_info_dict': {'z_array_lens': z_array, 'nbins_lens': 1, 'nz0': np.maximum(hist_z, 1e-10)},
    'is_cmb_lensing': True, 'l_array_survey': jnp.array(l_range), 'symbolic_pk': True
})

# =============================================================================
# 3. COMPUTE THEORY
# =============================================================================
print("Computing Theory...")
h = sim_p['cosmo']['H0'] / 100.0
ne0_cm3 = (sim_p['cosmo']['Ob0'] * 1.8784e-29 * h**2) / (1.14 * 1.6726e-24)

base_test = base_class(sim_p, halo_p, anal_p, other_p)
Prof_test = Profiles(sim_p, halo_p, anal_p, other_p, base_class_obj=base_test)

# Clean inputs to prevent spike propagation
Prof_test.Ncen_mat = jnp.nan_to_num(Prof_test.Ncen_mat)
Prof_test.Nsat_mat = jnp.nan_to_num(Prof_test.Nsat_mat)

pkz_test = get_Pkz(sim_p, halo_p, anal_p, other_p, Profiles_obj=Prof_test)
Cls_test = get_Cl(sim_p, halo_p, anal_p, other_p, Pkz_obj=pkz_test)

# =============================================================================
# 4. AGGREGATION
# =============================================================================
data_dir = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/reference_run'
pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
npix = 12 * nside**2
maps = {k: np.zeros(npix) for k in ['kappa', 'ymap', 'tau', 'gal']}

for fname in tqdm(pkl_files, desc="Aggregating"):
    try:
        with open(os.path.join(data_dir, fname), 'rb') as f: res = pk.load(f)
        for k in ['kappa', 'ymap', 'tau']: maps[k] += np.nan_to_num(res.get(f'map_{k}', 0))
        for chunk in res['mock_gals_all'].values():
            if chunk is not None:
                g_pix = hp.ang2pix(nside, np.array(chunk[:,0]), np.array(chunk[:,1]), lonlat=True)
                maps['gal'] += np.bincount(g_pix, minlength=npix)
    except Exception: pass

mask = (maps['gal'] > 0)
fsky = np.sum(mask) / npix
n_gal = np.sum(maps['gal'])
maps['gal'] = np.where(mask, (maps['gal'] / (n_gal / np.sum(mask))) - 1.0, 0.0)
shot_noise = (4 * np.pi * fsky) / n_gal
pixwin = hp.pixwin(nside, lmax=lmax)

# =============================================================================
# 5. PLOTTING & CHI2
# =============================================================================
os.makedirs('plots_tvss', exist_ok=True)
stats = [
    ('gg', Cls_test.Cl_gal_gal_tot_mat[:,0,0], 'gal', r'C_\ell^{\mathrm{gg}}'), 
    ('gy', Cls_test.Cl_gal_y_tot_mat[:,0], 'ymap', r'C_\ell^{\mathrm{gy}}'), 
    ('gtau', Cls_test.Cl_gal_tau_tot_mat[:,0], 'tau', r'C_\ell^{\mathrm{g\tau}}'), 
    ('gkappa', Cls_test.Cl_gal_kappa_tot_mat[:,0,0], 'kappa', r'C_\ell^{\mathrm{g\kappa}}')
]

for label, th_full, map_key, y_label in stats:
    cl_sim = hp.anafast(maps['gal'], maps[map_key], lmax=lmax)[2:] / (pixwin[2:]**2) / fsky
    if label == 'gg': cl_sim -= 0#shot_noise
    
    th = interp1d(Cls_test.ell_array, th_full, bounds_error=False, fill_value="extrapolate")(l_range)
    if label == 'gtau': th *= ne0_cm3
    
    # Chi2 Calculation
    valid = (l_range > 100) & (l_range < 2000)
    variance = (0.1 * th[valid])**2 + 1e-30
    chi2 = np.mean((cl_sim[valid] - th[valid])**2 / variance)
    
    plt.figure(figsize=(10, 8))
    plt.plot(l_range, np.abs(cl_sim), color='black', lw=2, alpha=0.7, label=f'Sim (RedChi2={chi2:.2e})')
    plt.plot(l_range, th, 'r-', lw=2.5, label='Theory')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\ell$', fontsize=20)
    plt.ylabel(f'${y_label}$', fontsize=20)
    plt.title(f'Comparison: {label}', fontsize=22)
    plt.legend(fontsize=20)
    plt.xlim(100, 3e3)
    plt.grid(True, which="both", alpha=0.2)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    
    plt.savefig(f'plots_tvss/thvssim_{label}.png', dpi=300)
    plt.close()

print("\n--- Final Chi2 Results ---")
for label, _, _, _ in stats:
    print(f"{label}: {chi2:.4e}")
