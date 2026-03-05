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
from jax_cosmo import Cosmology
from jax_cosmo.background import radial_comoving_distance
import jax_cosmo.background as bkgrd

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

# =============================================================================
# 2. LOAD DATA
# =============================================================================
print("1. Loading Data...")
data_dir = '/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/reference_run'
pkl_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

npix = 12 * nside**2
maps = {k: np.zeros(npix) for k in ['kappa', 'ymap', 'tau', 'gal']}
total_galaxies_sim = 0.0

for fname in tqdm(pkl_files, desc="Aggregating"):
    file_path = os.path.join(data_dir, fname)
    with open(file_path, 'rb') as f:
        res = pk.load(f)

    for k in ['kappa', 'ymap', 'tau']:
        if f'map_{k}' in res:
            maps[k] += np.nan_to_num(res[f'map_{k}'])

    mock_gals = res.get('mock_gals_all', {})
    for chunk in mock_gals.values():
        if chunk is not None and len(chunk) > 0:
            ra, dec = np.array(chunk[:, 0]), np.array(chunk[:, 1])
            valid_mask = ~(np.isnan(ra) | np.isnan(dec))
            ra, dec = ra[valid_mask], dec[valid_mask]

            if len(ra) == 0:
                continue
            total_galaxies_sim += len(ra)

            ra = np.mod(ra, 360.0)
            dec = np.clip(dec, -90.0, 90.0)
            g_pix = hp.ang2pix(nside, ra, dec, lonlat=True)
            maps['gal'] += np.bincount(g_pix, minlength=npix)

mask = (maps['kappa'] != 0.0)
fsky = np.sum(mask) / npix
if fsky < 1e-6:
    fsky = 1.0

if total_galaxies_sim > 0:
    mean_gal = np.sum(maps['gal'][mask]) / np.sum(mask)
    maps['gal'] = np.where(mask, (maps['gal'] / mean_gal) - 1.0, 0.0)
    shot_noise = (4 * np.pi * fsky) / total_galaxies_sim
else:
    maps['gal'] = np.zeros(npix)
    shot_noise = 0.0

print(f"Total Galaxies Loaded: {int(total_galaxies_sim)}")
print(f"fsky: {fsky:.4f}")
print(f"Shot noise: {shot_noise:.4e}")

# =============================================================================
# 3. CONFIGURE THEORY
# =============================================================================
print("2. Configuring Theory...")

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
    Omega_b=cosmo_params_dict['Ob0'],
    h=h,
    sigma8=cosmo_params_dict['sigma8'],
    n_s=cosmo_params_dict['ns'],
    Omega_k=0.,
    w0=cosmo_params_dict['w0'],
    wa=0.
)

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
indsel = np.where((zcen < zmin_gal) | (zcen > zmax_gal))[0]
nz_f[indsel] = 0.0
nz_f_norm = nz_f / np.trapezoid(nz_f, zcen)
nz_f_norm_interp = interp1d(zcen, nz_f_norm, fill_value=0.0, bounds_error=False)
hist_z = nz_f_norm_interp(zarray_lens)

chi_min = float(radial_comoving_distance(cosmo_jax, jnp.atleast_1d(1.0 / (1.0 + zmin_gal)))[0])
chi_max = float(radial_comoving_distance(cosmo_jax, jnp.atleast_1d(1.0 / (1.0 + zmax_gal)))[0])
V_comoving = (4.0 / 3.0) * np.pi * (chi_max**3 - chi_min**3) * fsky
nbar_sim = total_galaxies_sim / V_comoving
print(f"   -> nbar from sim: {nbar_sim:.4e} (Mpc/h)^-3")

nz_comoving = np.full_like(zarray_lens, nbar_sim)
analysis_dict['nbar_gal_comoving_zarray'] = zarray_lens
analysis_dict['nbar_gal_comoving_val'] = nz_comoving

nz_lens_info_dict = {}
nz_lens_info_dict['z_array_lens'] = zarray_lens
nz_lens_info_dict['nbins_lens'] = 1
nz_lens_info_dict['nz0'] = hist_z
analysis_dict['nz_lens_info_dict'] = nz_lens_info_dict

analysis_dict['is_cmb_lensing'] = True
nz_source_info_dict = {}
nz_source_info_dict['z_array_source'] = jnp.ones(1)
nz_source_info_dict['nbins'] = 1
nz_source_info_dict['nz0'] = jnp.ones(1)
analysis_dict['nz_source_info_dict'] = nz_source_info_dict
other_params_dict['Delta_z_bias_array'] = jnp.zeros(1)
other_params_dict['mult_shear_bias_array'] = jnp.zeros(1)

lmin_th, lmax_th, dl_log_array = 80.0, 8800.0, 0.23025851
l_array_all = np.exp(np.arange(np.log(lmin_th), np.log(lmax_th), dl_log_array))
dl_array = l_array_all[1:] - l_array_all[:-1]
l_array_survey = (l_array_all[1:] + l_array_all[:-1]) / 2.
halo_params_dict['ell_array'] = jnp.array(l_array_survey)
analysis_dict['l_array_survey'] = jnp.array(l_array_survey)
analysis_dict['dl_array_survey'] = jnp.array(dl_array)

analysis_dict['symbolic_pk'] = True
analysis_dict['symbolic_hmf'] = True
ks = np.geomspace(1e-3, 100, 200)
analysis_dict['k_array_survey'] = jnp.array(ks)

halo_params_dict.update({
    'rmin': 0.005, 'rmax': 10.0, 'nr': 48,
    'zmin': Z_MIN, 'zmax': Z_MAX, 'nz': 31,
    'lg10_Mmin': 12.75, 'lg10_Mmax': 16.0, 'nM': 32
})

# =============================================================================
# FORCE UN-SMOOTHED THEORY FOR PROPER ASYMMETRIC BEAM MATCHING
# =============================================================================
analysis_dict['beam_fwhm_arcmin'] = 1e-5

# =============================================================================
# 4. COMPUTE THEORY
# =============================================================================
print("3. Computing Theory...")
try:
    base_test = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
    Prof_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, base_class_obj=base_test)
    '''
    M_halo_cut = 10**12.75
    mass_grid = Prof_test.M_array
    hard_mask = jnp.where(mass_grid >= M_halo_cut, 1.0, 0.0)
    hard_mask_2d = jnp.tile(hard_mask, (halo_params_dict['nz'], 1))

    Ncen_standard = jnp.stack([Prof_test.get_Ncen(jz, jnp.arange(halo_params_dict['nM']))
                                for jz in range(halo_params_dict['nz'])])
    Nsat_standard = jnp.stack([Prof_test.get_Nsat(jz, jnp.arange(halo_params_dict['nM']))
                                for jz in range(halo_params_dict['nz'])])

    Prof_test.Ncen_mat = Ncen_standard * hard_mask_2d
    Prof_test.Nsat_mat = Nsat_standard * hard_mask_2d
    '''
    pkz_test = get_Pkz(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Profiles_obj=Prof_test)

    from astropy import constants as const
    import astropy.units as u
    rho_crit_0 = 1.878e-29 * h**2
    mp_val = const.m_p.to(u.g).value
    Ob0 = cosmo_params_dict['Ob0']
    Y_He = 0.24
    ne0_cm3 = rho_crit_0 * Ob0 * (1 - Y_He / 2) / mp_val

    Cls_test = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Pkz_obj=pkz_test)
    
    # =============================================================================
    # 5. DIAGNOSTICS & PLOTTING
    # =============================================================================
    print("4. Plotting...")
    ell_theory = np.array(Cls_test.ell_array)
    l_range_int = np.arange(2, lmax + 1)
    pixwin = hp.pixwin(nside, lmax=lmax)
    os.makedirs('plots_tvss', exist_ok=True)

    # Apply the (1+z)^3 volume expansion to gtau directly
    z_mean_shell = 0.4
    Cl_gtau_corrected = np.array(Cls_test.Cl_gal_tau_tot_mat[:, 0]) * ne0_cm3 * (1.0 + z_mean_shell)**3

    stats = [
        ('gg',     np.array(Cls_test.Cl_gal_gal_tot_mat[:, 0, 0]), r'C_\ell^{\mathrm{gg}}', 'gal'),
        ('gy',     np.array(Cls_test.Cl_gal_y_tot_mat[:, 0]),      r'C_\ell^{\mathrm{gy}}', 'ymap'),
        ('gtau',   Cl_gtau_corrected,                              r'C_\ell^{\mathrm{g\tau}}', 'tau'),
        ('gkappa', np.array(Cls_test.Cl_gal_kappa_tot_mat[:, 0, 0]), r'C_\ell^{\mathrm{g\kappa}}', 'kappa'),
    ]

    # Calculate exact Simulation Gaussian beam for nside=512
    sigma_sim = hp.nside2resol(nside) / np.sqrt(8. * np.log(2.))
    beam_ell = np.exp(-0.5 * l_range_int * (l_range_int + 1) * sigma_sim**2)

    for label, th_arr, y_label, map_key in stats:
        cl_sim_raw = hp.anafast(maps['gal'], maps[map_key], lmax=lmax)[2:] / fsky
        
        # We process theory directly on l_range_int
        th_interp = interp1d(ell_theory, th_arr, bounds_error=False, fill_value=0.0)
        th = th_interp(l_range_int)

        if label == 'gg':
            cl_sim_raw -= shot_noise
            # NO BEAM APPLIED: Galaxies were merely binned, not smoothed.
            
        elif label == 'gkappa':
            th *= (h**2)      #kappa from theory was dimensionless, but this sorts it out
            th *= beam_ell    # Apply beam: kappa map was smoothed in sim
            
        elif label in ['gy', 'gtau']:
            th *= beam_ell    # Apply beam: gas maps were smoothed in sim

        cl_sim = cl_sim_raw / (pixwin[2:]**2)

        ell_check = 200
        idx_sim = np.argmin(np.abs(l_range_int - ell_check))
        idx_th = np.argmin(np.abs(l_range_int - ell_check)) 
        
        if np.abs(th[idx_th]) > 1e-30 and np.abs(cl_sim[idx_sim]) > 1e-30:
            amp_ratio = cl_sim[idx_sim] / th[idx_th]
        else:
            amp_ratio = np.nan

        valid = ((l_range_int > 100) & (l_range_int < 500)
                 & np.isfinite(th) & (np.abs(th) > 1e-30)
                 & np.isfinite(cl_sim) & (np.abs(cl_sim) > 1e-30))
                 
        if np.sum(valid) > 0:
            ratio = cl_sim[valid] / th[valid]
            chi2 = np.mean((ratio - 1.0)**2) / (0.1**2)
        else:
            chi2 = 0.0

        print(f"  {label}: RedChi2(100-500) = {chi2:.2e}, sim/theory at ell={ell_check}: {amp_ratio:.3f}")

        plt.figure(figsize=(9, 7))
        plt.plot(l_range_int, np.abs(cl_sim), color='k', lw=2.5, alpha=0.7, label='Sim')
        if np.any(np.isfinite(th) & (th != 0)):
            plt.plot(l_range_int, np.abs(th), 'r-', lw=2.5, label='Theory')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\ell$', fontsize=20)
        plt.ylabel(f'${y_label}$', fontsize=20)
        plt.title(f'{label}: sim/th@ell{ell_check}={amp_ratio:.3f}', fontsize=20)
        plt.legend(fontsize=18)
        plt.grid(True, which="both", alpha=0.2)
        plt.xlim(100, 1e3)
        plt.savefig(f'plots_tvss/thvssim_{label}.png', dpi=300)
        plt.close()

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
