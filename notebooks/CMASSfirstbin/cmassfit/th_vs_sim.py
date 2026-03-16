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

# =============================================================================
# 1. SETUP
# =============================================================================
curr_path = pathlib.Path().absolute()
project_base = curr_path.parents[2]
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

# fsky from kappa mask — kappa is zero only in unobserved pixels
mask = (maps['kappa'] != 0.0)
fsky = np.sum(mask) / npix
if fsky < 1e-6:
    fsky = 1.0

if total_galaxies_sim > 0:
    # Convert raw galaxy counts to overdensity delta_g = n/nbar - 1
    mean_gal = np.sum(maps['gal'][mask]) / np.sum(mask)
    maps['gal'] = np.where(mask, (maps['gal'] / mean_gal) - 1.0, 0.0)
    # Poisson shot noise: N_shot = 1/nbar [sr] = 4pi*fsky / N_gal
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
sim_params_dict   = default_data.get('sim_params', {})
halo_params_dict  = default_data.get('halo_params', {})
analysis_dict     = default_data.get('analysis', {})
other_params_dict = default_data.get('other_params', {})

# Use the sim cosmology, not the yaml default
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
zarray_lens = np.linspace(Z_MIN, Z_MAX, 255)

# Build n(z) from comoving shell volumes in the galaxy redshift bin [0.3, 0.5]
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
# Normalise so get_Cls interprets it as a probability distribution
hist_z = hist_z / hist_z.sum()
print("check sum of hist_z: ", np.sum(hist_z))

# Compute nbar_sim from total galaxy count and comoving shell volume
chi_min = float(radial_comoving_distance(cosmo_jax, jnp.atleast_1d(1.0 / (1.0 + zmin_gal)))[0])
chi_max = float(radial_comoving_distance(cosmo_jax, jnp.atleast_1d(1.0 / (1.0 + zmax_gal)))[0])
V_comoving = (4.0 / 3.0) * np.pi * (chi_max**3 - chi_min**3) * fsky
nbar_sim = total_galaxies_sim / V_comoving
print(f"   -> nbar from sim: {nbar_sim:.4e} (Mpc/h)^-3")

# Pass nbar_sim to get_Mthresh so the HOD threshold is solved to match the sim
nz_comoving = np.full_like(zarray_lens, nbar_sim)
analysis_dict['nbar_gal_comoving_zarray'] = zarray_lens
analysis_dict['nbar_gal_comoving_val'] = nz_comoving

# Set lens n(z) for Cl projection
nz_lens_info_dict = {
    'z_array_lens': zarray_lens,
    'nbins_lens': 1,
    'nz0': hist_z
}
analysis_dict['nz_lens_info_dict'] = nz_lens_info_dict

# CMB lensing source — dummy n(z), projection handled internally
analysis_dict['is_cmb_lensing'] = True
analysis_dict['nz_source_info_dict'] = {
    'z_array_source': jnp.ones(1),
    'nbins': 1,
    'nz0': jnp.ones(1)
}
other_params_dict['Delta_z_bias_array']   = jnp.zeros(1)
other_params_dict['mult_shear_bias_array'] = jnp.zeros(1)

# Log-spaced ell array for theory Cl computation
lmin_th, lmax_th, dl_log_array = 80.0, 8800.0, 0.23025851
l_array_all  = np.exp(np.arange(np.log(lmin_th), np.log(lmax_th), dl_log_array))
dl_array     = l_array_all[1:] - l_array_all[:-1]
l_array_survey = (l_array_all[1:] + l_array_all[:-1]) / 2.
halo_params_dict['ell_array']          = jnp.array(l_array_survey)
analysis_dict['l_array_survey']        = jnp.array(l_array_survey)
analysis_dict['dl_array_survey']       = jnp.array(dl_array)

analysis_dict['symbolic_pk']  = True
analysis_dict['symbolic_hmf'] = True
analysis_dict['k_array_survey'] = jnp.array(np.geomspace(1e-3, 100, 200))

# Match the sim halo physics grid exactly:
# - lg10_Mmin=11.75 gives a wide enough mass grid for HOD to solve Mthresh
# - rmax=10 Mpc/h covers the gas profile out to 3*R200c for massive halos
halo_params_dict.update({
    'rmin': 0.005, 'rmax': 10.0, 'nr': 48,
    'zmin': Z_MIN, 'zmax': Z_MAX, 'nz': 31,
    'lg10_Mmin': 11.75, 'lg10_Mmax': 16.0, 'nM': 32
})

# Disable theory beam smoothing. The sim maps are not pre-smoothed by a theory beam;
# beam effects from pixelization are handled separately on the sim side
analysis_dict['beam_fwhm_arcmin'] = 1e-5

# =============================================================================
# 4. COMPUTE THEORY
# =============================================================================
print("3. Computing Theory...")
try:
    base_test = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
    Prof_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, base_class_obj=base_test)

    # -------------------------------------------------------------------------
    # Apply the same halo mass cut as the sim: open_data uses M200c > 10^12.75
    # Zero out HOD occupation for halos below this mass so theory and sim
    # integrate over the same halo population
    # -------------------------------------------------------------------------
    M_halo_cut = 10**12.75
    hard_mask = jnp.where(Prof_test.M_array > M_halo_cut, 1.0, 0.0)
    hard_mask_2d = jnp.tile(hard_mask, (halo_params_dict['nz'], 1))
    Ncen_standard = jnp.stack([Prof_test.get_Ncen(jz, jnp.arange(halo_params_dict['nM']))
                                for jz in range(halo_params_dict['nz'])])
    Nsat_standard = jnp.stack([Prof_test.get_Nsat(jz, jnp.arange(halo_params_dict['nM']))
                                for jz in range(halo_params_dict['nz'])])
    Prof_test.Ncen_mat = Ncen_standard * hard_mask_2d
    Prof_test.Nsat_mat = Nsat_standard * hard_mask_2d

    # -------------------------------------------------------------------------
    # Truncate gas profiles at 3*R200c to match the sim painter, which uses
    # max_paint_R200c_factor=3.0 in process_halos_in_batches. Gas beyond this
    # radius is not painted in the sim, so we zero it in theory too.
    # -------------------------------------------------------------------------
    r_3d     = Prof_test.r_array[:, None, None]
    r200c_3d = Prof_test.r200c_mat[None, :, :]
    gas_mask = jnp.where(r_3d <= 3.0 * r200c_3d, 1.0, 0.0)
    for attr in ['rho_gas_mat', 'rho_gas_mat_physical',
                 'ne_mat', 'ne_mat_physical', 'Pe_mat_physical', 'y3d_mat']:
        if hasattr(Prof_test, attr):
            setattr(Prof_test, attr, getattr(Prof_test, attr) * gas_mask)

    pkz_test = get_Pkz(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Profiles_obj=Prof_test)

    # -------------------------------------------------------------------------
    # Remove Pmm_sup from galaxy power spectra.
    # Pmm_sup = P_halofit / P_halomodel_NFW corrects the matter field for
    # baryonic suppression and nonlinear effects beyond the NFW profile.
    # Galaxy positions are set by the HOD on halo centres and NFW satellite
    # profiles — they are not suppressed by baryonic gas physics. Applying
    # Pmm_sup to Pgg and Pgm incorrectly suppresses galaxy power by ~16% at
    # k~0.3 h/Mpc, inflating sim/theory by ~1.19 for Pgm and ~1.42 for Pgg.
    # Note: Pgy_tot_mat already has no Pmm_sup in the poweradd model.
    # -------------------------------------------------------------------------
    pkz_test.Pgg_tot_mat     = pkz_test.Pgg_1h_kz_mat + pkz_test.Pgg_2h_kz_mat
    pkz_test.Pgm_tot_mat     = pkz_test.Pgm_1h_kz_mat + pkz_test.Pgm_2h_kz_mat
    pkz_test.Pgm_nfw_tot_mat = pkz_test.Pgm_nfw_1h_kz_mat + pkz_test.Pgm_nfw_2h_kz_mat

    # tau Cl from theory is dimensionless (integrated optical depth profile);
    # multiply by ne0 and (1+z)^3 to convert to physical electron column units
    # matching the sim's tau map which is painted in physical coordinates
    from astropy import constants as const
    import astropy.units as u
    rho_crit_0 = 1.878e-29 * h**2          # g/cm^3
    mp_val     = const.m_p.to(u.g).value
    Y_He       = 0.24
    ne0_cm3    = rho_crit_0 * cosmo_params_dict['Ob0'] * (1 - Y_He / 2) / mp_val

    Cls_test = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Pkz_obj=pkz_test)

    # =============================================================================
    # 5. DIAGNOSTICS & PLOTTING
    # =============================================================================
    print("4. Plotting...")
    ell_theory  = np.array(Cls_test.ell_array)
    l_range_int = np.arange(2, lmax + 1)
    os.makedirs('plots_tvss', exist_ok=True)

    z_mean_shell = 0.4
    Cl_gtau_corrected = (np.array(Cls_test.Cl_gal_tau_tot_mat[:, 0])
                         * ne0_cm3 * (1.0 + z_mean_shell)**3)

    stats = [
        ('gg',     np.array(Cls_test.Cl_gal_gal_tot_mat[:, 0, 0]),   r'C_\ell^{\mathrm{gg}}',      'gal'),
        ('gy',     np.array(Cls_test.Cl_gal_y_tot_mat[:, 0]),         r'C_\ell^{\mathrm{gy}}',      'ymap'),
        ('gtau',   Cl_gtau_corrected,                                  r'C_\ell^{\mathrm{g\tau}}',   'tau'),
        ('gkappa', np.array(Cls_test.Cl_gal_kappa_tot_mat[:, 0, 0]),  r'C_\ell^{\mathrm{g\kappa}}', 'kappa'),
    ]

    # Pixel window function of nside=512 HEALPix map expressed as a Gaussian beam.
    # Used only to damp the *theory* for gas and kappa cross-spectra, because those
    # sim maps are painted with smooth profiles and then read off at pixel centres —
    # effectively convolved with the pixel beam. The galaxy map is a point-source
    # count map (np.bincount), not convolved, so no beam is applied to gg.
    sigma_sim = hp.nside2resol(nside) / np.sqrt(8. * np.log(2.))
    beam_ell  = np.exp(-0.5 * l_range_int * (l_range_int + 1) * sigma_sim**2)

    for label, th_arr, y_label, map_key in stats:
        # Raw pseudo-Cl divided by fsky to correct for partial sky
        cl_sim_raw = hp.anafast(maps['gal'], maps[map_key], lmax=lmax)[2:] / fsky

        th_interp = interp1d(ell_theory, th_arr, bounds_error=False, fill_value=0.0)
        th = th_interp(l_range_int)

        if label == 'gg':
            # Subtract Poisson shot noise from the auto-spectrum
            cl_sim_raw -= shot_noise
            # No pixel window correction: galaxy map is a binned count map,
            # not convolved with the pixel beam, so pixwin does not apply
            cl_sim = cl_sim_raw

        elif label == 'gkappa':
            # kappa map units: theory gives convergence in h-free units,
            # sim stores it in (Mpc/h)^-2 — multiply by h^2 to match
            th *= h**2
            # kappa map is painted with smooth profiles -> apply pixel beam to theory
            th *= beam_ell
            # Galaxy map has no pixel beam; kappa map does -> single pixwin factor
            cl_sim = cl_sim_raw

        elif label in ['gy', 'gtau']:
            # Gas maps (y, tau) are painted with smooth profiles -> apply pixel beam
            th *= beam_ell
            cl_sim = cl_sim_raw

        ell_check = 200
        idx = np.argmin(np.abs(l_range_int - ell_check))
        amp_ratio = (cl_sim[idx] / th[idx]
                     if np.abs(th[idx]) > 1e-30 and np.abs(cl_sim[idx]) > 1e-30
                     else np.nan)

        valid = ((l_range_int > 100) & (l_range_int < 500)
                 & np.isfinite(th) & (np.abs(th) > 1e-30)
                 & np.isfinite(cl_sim) & (np.abs(cl_sim) > 1e-30))
        chi2 = (np.mean((cl_sim[valid] / th[valid] - 1.0)**2) / 0.1**2
                if np.sum(valid) > 0 else 0.0)

        print(f"  {label}: RedChi2(100-500) = {chi2:.2e}, sim/theory at ell={ell_check}: {amp_ratio:.3f}")

        plt.figure(figsize=(9, 7))
        plt.plot(l_range_int, np.abs(cl_sim), color='k', lw=2.5, alpha=0.7, label='Sim')
        if np.any(np.isfinite(th) & (th != 0)):
            plt.plot(l_range_int, np.abs(th), 'r-', lw=2.5, label='Theory')
        plt.xscale('log'); plt.yscale('log')
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
