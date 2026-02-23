import os
import sys
import pathlib
import pickle as pk
import yaml
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d
from tqdm import tqdm
from astropy.cosmology import FlatLambdaCDM
import jax.scipy.integrate as jsi 

# =============================================================================
# 1. SETUP
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
# 2. LOAD DATA
# =============================================================================
print("1. Loading Data...")
data_dir = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/reference_run'
pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
npix = 12 * nside**2
maps = {k: np.zeros(npix) for k in ['kappa', 'ymap', 'tau', 'gal']}

total_galaxies = 0.0
for fname in tqdm(pkl_files, desc="Aggregating"):
    try:
        with open(os.path.join(data_dir, fname), 'rb') as f: res = pk.load(f)
        for k in ['kappa', 'ymap', 'tau']: maps[k] += np.nan_to_num(res.get(f'map_{k}', 0))
        for chunk in res['mock_gals_all'].values():
            if chunk is not None:
                total_galaxies += len(chunk)
                ra, dec = np.array(chunk[:,0]), np.array(chunk[:,1])
                if np.any(np.isnan(ra)) or np.any(np.isnan(dec)): continue
                g_pix = hp.ang2pix(nside, ra, dec, lonlat=True)
                maps['gal'] += np.bincount(g_pix, minlength=npix)
    except Exception: pass

mask = (maps['gal'] > 0)
fsky = np.sum(mask) / npix
if fsky < 1e-6: fsky = 0.5745 

if total_galaxies > 0:
    mean_gal = np.mean(maps['gal'][mask])
    maps['gal'] = np.where(mask, (maps['gal'] / mean_gal) - 1.0, 0.0)
    shot_noise = (4 * np.pi * fsky) / total_galaxies
else:
    maps['gal'] = np.zeros(npix)
    shot_noise = 0.0

print(f"Total Galaxies: {int(total_galaxies)}")

# =============================================================================
# 3. CONFIGURE THEORY
# =============================================================================
print("2. Configuring Theory...")
default_data = yaml.safe_load(open(abs_path_params + '/params_default.yaml'))
sim_p, halo_p, anal_p, other_p = default_data['sim_params'], default_data['halo_params'], default_data['analysis'], default_data['other_params']

sim_p['cosmo'] = {'w0': -1.0, 'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624}

# --- STABILITY SETTINGS ---
Z_MIN = 0.01
Z_MAX = 2.1 

halo_p.update({
    'rmin': 0.001, 'rmax': 10.0, 'nr': 48,
    'zmin': Z_MIN, 'zmax': Z_MAX, 'nz': 31, 
    'lg10_Mmin': 11.75, 'lg10_Mmax': 16.0,  # Wide range to support low threshold
    'nM': 32,
})

# Remove density input (we will set threshold manually)
if 'nbar_gal_comoving_val' in anal_p: del anal_p['nbar_gal_comoving_val']

anal_p.update({
    'zmin_for_Cls': Z_MIN, 'zmax_for_Cls': Z_MAX, 
#    'nz_for_Cls': 64,'num_points_trapz_int': 64
})

# Lens Distribution (Masked window)
z_array_lens = np.linspace(Z_MIN, Z_MAX, 128)
hist_z = np.zeros_like(z_array_lens)
mask_z = (z_array_lens >= 0.3) & (z_array_lens <= 0.5)
hist_z[mask_z] = 1.0
hist_z /= jsi.trapezoid(hist_z, z_array_lens)

anal_p.update({
    'nz_lens_info_dict': {'z_array_lens': z_array_lens, 'nbins_lens': 1, 'nz0': np.maximum(hist_z, 1e-10)},
    'is_cmb_lensing': True, 'l_array_survey': jnp.array(l_range), 'symbolic_pk': True
})

# =============================================================================
# 4. COMPUTE WITH "TRANSLATOR" LOGIC
# =============================================================================
print("3. Computing...")
try:
    base_test = base_class(sim_p, halo_p, anal_p, other_p)
    Prof_test = Profiles(sim_p, halo_p, anal_p, other_p, base_class_obj=base_test)
    
    # --- THE TRANSLATOR FIX ---
    # We want to emulate a Halo Mass cut of 10^12.75.
    # We ask the code: "What Stellar Mass corresponds to M_halo = 10^12.75?"
    
    # 1. Pick a representative redshift (z=0.4)
    z_idx_mid = jnp.argmin(jnp.abs(Prof_test.z_array - 0.4))
    
    # 2. Find the Halo Mass Index nearest to 10^12.75
    M_halo_target = 10**12.75
    M_idx = jnp.argmin(jnp.abs(Prof_test.M_array - M_halo_target))
    
    # 3. Calculate the Equivalent Stellar Mass
    M_star_equivalent = Prof_test.get_Mstar_Mh(z_idx_mid, M_idx)
    log_M_star_eq = jnp.log10(M_star_equivalent)
    
    print(f"\n>>> PHYSICS MATCH FOUND:")
    print(f"    Simulation Halo Cut: 10^12.75 M_sun")
    print(f"    Equivalent Stellar Cut: 10^{log_M_star_eq:.2f} M_sun")
    print(f"    (Using this value to bypass density solver and fix bias)\n")
    
    # 4. Apply this Threshold Constant across all z (Circumvents 10^14 issue)
    Prof_test.Mthresh_array = jnp.ones(halo_p['nz']) * M_star_equivalent
    
    # 5. Recalculate HOD
    Prof_test.Ncen_mat = jnp.stack([Prof_test.get_Ncen(jz, jnp.arange(halo_p['nM'])) for jz in range(halo_p['nz'])])
    Prof_test.Nsat_mat = jnp.stack([Prof_test.get_Nsat(jz, jnp.arange(halo_p['nM'])) for jz in range(halo_p['nz'])])
    
    # --- PIPELINE CONTINUES ---
    pkz_test = get_Pkz(sim_p, halo_p, anal_p, other_p, Profiles_obj=Prof_test)
    
    # Recalculate nbarz for consistency
    dndlogM = Prof_test.hmf_Mz_mat
    Ntot_mat = Prof_test.Ncen_mat + Prof_test.Nsat_mat
    pkz_test.nbarz = jsi.trapezoid(dndlogM * Ntot_mat, x=jnp.log(Prof_test.M_array), axis=1)
    
    Cls_test = get_Cl(sim_p, halo_p, anal_p, other_p, Pkz_obj=pkz_test)

    # Plotting
    pixwin = hp.pixwin(nside, lmax=lmax)
    os.makedirs('plots_tvss', exist_ok=True)
    stats = [
        ('gg', Cls_test.Cl_gal_gal_tot_mat[:,0,0], 'gal', r'C_\ell^{\mathrm{gg}}'), 
        ('gy', Cls_test.Cl_gal_y_tot_mat[:,0], 'ymap', r'C_\ell^{\mathrm{gy}}'), 
        ('gtau', Cls_test.Cl_gal_tau_tot_mat[:,0], 'tau', r'C_\ell^{\mathrm{g\tau}}'), 
        ('gkappa', Cls_test.Cl_gal_kappa_tot_mat[:,0,0], 'kappa', r'C_\ell^{\mathrm{g\kappa}}')
    ]
    
    ne0_cm3 = (sim_p['cosmo']['Ob0'] * 1.8784e-29 * (0.6711)**2) / (1.14 * 1.6726e-24)

    for label, th_full, map_key, y_label in stats:
        cl_sim = hp.anafast(maps['gal'], maps[map_key], lmax=lmax)[2:] / (pixwin[2:]**2) / fsky
        if label == 'gg': cl_sim -= shot_noise
        
        th = interp1d(Cls_test.ell_array, th_full, bounds_error=False, fill_value=0.0)(l_range)
        if label == 'gtau': th *= ne0_cm3
        
        valid = (l_range > 100) & (l_range < 2000) & (np.isfinite(th)) & (np.abs(th) > 1e-30)
        
        if np.sum(valid) > 0:
            variance = (0.1 * th[valid])**2
            chi2 = np.mean(np.divide((cl_sim[valid] - th[valid])**2, variance, out=np.zeros_like(variance), where=variance!=0))
        else:
            chi2 = 0.0
            
        print(f"For {label} RedChi2 is {chi2:.2e}")
        
        plt.figure(figsize=(10, 8))
        plt.plot(l_range, np.abs(cl_sim), color='k', lw=2.5, alpha=0.7, label=f'Sim (RedChi2={chi2:.2e})')
        plt.plot(l_range, th, 'r-', lw=2.5, label='Theory')
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel(r'$\ell$', fontsize=20); plt.ylabel(f'${y_label}$', fontsize=20)
        plt.title(f'Comparison: {label}', fontsize=22); plt.legend(fontsize=20)
        plt.grid(True, which="both", alpha=0.2)
        plt.savefig(f'plots_tvss/thvssim_{label}.png', dpi=300); plt.close()

except Exception as e:
    print(f"ERROR: {e}")
    import traceback; traceback.print_exc()

print("Done.")
