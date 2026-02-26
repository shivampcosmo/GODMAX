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
    fsky = 0.5745

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

# KEY FIX: zmax must cover z_array_for_Cls to prevent extrapolation disaster
Z_MIN, Z_MAX = 0.005, 2.0  # Extended to cover Cls integration range
zarray_lens = np.linspace(0.001, 0.8, 40)  # n(z) only defined to z=0.8

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
print(np.trapezoid(hist_z, zarray_lens), "CHECKKKKKKKK")
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

ks = np.geomspace(1e-2, 50, 80)
analysis_dict['k_array_survey'] = jnp.array(ks / h)

lmin_th, lmax_th, dl_log_array = 30.0, 8800.0, 0.1
l_array_all = np.exp(np.arange(np.log(lmin_th), np.log(lmax_th), dl_log_array))
dl_array = l_array_all[1:] - l_array_all[:-1]
l_array_survey = (l_array_all[1:] + l_array_all[:-1]) / 2.
halo_params_dict['ell_array'] = jnp.array(l_array_survey)
analysis_dict['l_array_survey'] = jnp.array(l_array_survey)
analysis_dict['dl_array_survey'] = jnp.array(dl_array)

analysis_dict['symbolic_pk'] = True
analysis_dict['symbolic_hmf'] = True
ks = np.geomspace(1e-2, 50, 80)  # h/Mpc
analysis_dict['k_array_survey'] = jnp.array(ks)  # pass as h/Mpc directly

# KEY FIX: extend halo model z range to cover the Cls integration range
halo_params_dict.update({
    'rmin': 0.005, 'rmax': 10.0, 'nr': 48,
    'zmin': Z_MIN, 'zmax': Z_MAX, 'nz': 51,  # more z points to cover 0.005-2.0
    'lg10_Mmin': 11.75, 'lg10_Mmax': 16.0, 'nM': 32
})

# =============================================================================
# 4. COMPUTE THEORY
# =============================================================================
print("3. Computing Theory...")
try:
    base_test = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
    Prof_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, base_class_obj=base_test)
    print("chi_CMB is", bkgrd.radial_comoving_distance(Prof_test.cosmo_jax, 1.0 / (1.0 + 1089.0)).item())
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

    print(f"   -> Hard mass cut at M_halo >= {M_halo_cut:.2e}")
    print(f"   -> z_array range: [{float(Prof_test.z_array[0]):.4f}, {float(Prof_test.z_array[-1]):.4f}], nz={len(Prof_test.z_array)}")

    pkz_test = get_Pkz(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Profiles_obj=Prof_test)

    # ne0 for tau correction
    from astropy import constants as const
    import astropy.units as u
    rho_crit_0 = 1.878e-29 * h**2
    mp = const.m_p.to(u.g).value
    Ob0 = cosmo_params_dict['Ob0']
    Y_He = 0.24
    ne0_cm3 = rho_crit_0 * Ob0 * (1 - Y_He / 2) / mp
    print(f"   -> ne0_cm3: {ne0_cm3:.4e}")

    Cls_test = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Pkz_obj=pkz_test)
    k_test = 0.1
    Pk_test = jnp.exp(jnp.interp(jnp.log(k_test), jnp.log(Cls_test.kPk_array), jnp.log(Cls_test.plin_kz_mat[:, 0])))
    print(f"P_lin(k=0.1, z={Cls_test.z_array[0]:.3f}) = {Pk_test:.1f}")
    # Check that cached power spectra are sane now
    cached = np.array(Cls_test.cached_power_spectra)
    z_for_cls = np.array(Cls_test.z_array_for_Cls)
    print(f"   -> z_array_for_Cls range: [{z_for_cls.min():.4f}, {z_for_cls.max():.4f}]")
    print(f"   -> cached[2,0] (Pgm) max: {np.nanmax(cached[2,0]):.3e}")
    print(f"   -> cached[2,2] (Pgg) max: {np.nanmax(cached[2,2]):.3e}")
    print(f"   -> cached[0,0] (Pmm) max: {np.nanmax(cached[0,0]):.3e}")

    # =============================================================================
    # 5. THEORY vs SIM LENSING KERNEL DIAGNOSTIC
    # =============================================================================
    print("\n=== LENSING KERNEL DIAGNOSTIC ===")

    # --- Theory side ---
    # From get_cmb_lensing_kernel: uses (100/c)^2, distances in Mpc/h
    c_km_s = const.c.value * 1e-3  # speed of light in km/s
    H0 = cosmo_params_dict['H0']
    Om0 = cosmo_params_dict['Om0']

    chi_CMB_theory = float(bkgrd.radial_comoving_distance(Cls_test.cosmo_jax, 1.0 / (1.0 + 1089.0))[0])
    constant_factor_cmb_theory = 3.0 * (100.)**2 * Om0 / (2.0 * c_km_s**2)

    # Theory arrays used in Limber integral
    z_cls = np.array(Cls_test.z_array_for_Cls)
    chi_cls = np.array(Cls_test.chi_array_for_Cls)
    dchi_dz_cls = np.array(Cls_test.dchi_dz_array_for_Cls)

    # Theory Wk(z) — the full kernel as computed in get_cmb_lensing_kernel
    Wk_theory = constant_factor_cmb_theory * np.clip(chi_CMB_theory - chi_cls, 0, None) / chi_CMB_theory * (1.0 + z_cls) * chi_cls
    # Theory effective weight per dz in the Limber integrand: Wk / chi^2 * dchi/dz
    # (this is the prefac that multiplies P(k,z) * chi^2 * dchi/dz in get_Cl_tot,
    #  but after cancellation with chi^2, the effective weight per dz is Wk/chi^2 * dchi/dz)
    theory_effective_weight_per_dz = Wk_theory / chi_cls**2 * dchi_dz_cls

    print(f"   Theory chi_CMB: {chi_CMB_theory:.4f} Mpc/h")
    print(f"   Theory constant_factor: {constant_factor_cmb_theory:.6e}")
    print(f"   Theory Wk_mat[0] from object: {np.array(Cls_test.Wk_mat[0])[:5]}")
    print(f"   Theory Wk recomputed:         {Wk_theory[:5]}")

    # --- Simulation side ---
    # Your simulation snapshot at z=0.484
    z_sim = 0.484
    a_sim = 1.0 / (1.0 + z_sim)
    chi_sim = float(bkgrd.radial_comoving_distance(cosmo_jax, jnp.atleast_1d(a_sim))[0])
    chi_CMB_sim = float(bkgrd.radial_comoving_distance(cosmo_jax, jnp.atleast_1d(1.0 / (1.0 + 1089.0)))[0])

    # =============================================================================
    # 6. DIAGNOSTICS & PLOTTING
    # =============================================================================
    ell_theory = np.array(Cls_test.ell_array)
    Cl_gtau_corrected = np.array(Cls_test.Cl_gal_tau_tot_mat[:, 0]) * ne0_cm3

    print("\n=== Cl DIAGNOSTICS ===")
    for name, arr in [
        ("Cl_gg", np.array(Cls_test.Cl_gal_gal_tot_mat[:, 0, 0])),
        ("Cl_gy", np.array(Cls_test.Cl_gal_y_tot_mat[:, 0])),
        ("Cl_gtau", Cl_gtau_corrected),
        ("Cl_gkappa", np.array(Cls_test.Cl_gal_kappa_tot_mat[:, 0, 0])),
    ]:
        finite_nonzero = np.isfinite(arr) & (arr != 0)
        if np.any(finite_nonzero):
            print(f"{name}: range=[{np.nanmin(arr[finite_nonzero]):.3e}, {np.nanmax(arr[finite_nonzero]):.3e}]")
    print("=== END ===\n")

    print("4. Plotting...")
    l_range_int = np.arange(2, lmax + 1)
    pixwin = hp.pixwin(nside, lmax=lmax)
    os.makedirs('plots_tvss', exist_ok=True)
    # =============================================================================
    # KAPPA CALIBRATION: Compare kappa-kappa auto to isolate kappa normalization
    # =============================================================================
    # Theory kappa-kappa
    Cl_kk_theory = np.array(Cls_test.Cl_kappa_kappa_tot_mat[:, 0, 0])

    # Sim kappa-kappa  
    cl_kk_sim_raw = hp.anafast(maps['kappa'], lmax=lmax)[2:] / fsky
    cl_kk_sim = cl_kk_sim_raw / (pixwin[2:]**2)
    # Compare at ell=200
    ell_check = 200
    idx_sim = np.argmin(np.abs(l_range_int - ell_check))
    idx_th = np.argmin(np.abs(ell_theory - ell_check))
    kk_ratio = cl_kk_sim[idx_sim] / Cl_kk_theory[idx_th]
    print(f"   -> kappa-kappa sim/theory at ell={ell_check}: {kk_ratio:.4f}")
    print(f"   -> sqrt(kk_ratio) = {np.sqrt(kk_ratio):.4f}  (this is the kappa normalization factor)")
    print(f"   -> Expected from h factors: h^2 = {h**2:.4f}")
    # If kk_ratio ~ h^2, then the sim's kappa is off by a factor of h
    # For gkappa: sim/theory should be ~ sqrt(kk_ratio) * (galaxy_bias_correction)

    stats = [
        ('gg',     np.array(Cls_test.Cl_gal_gal_tot_mat[:, 0, 0]),
         r'C_\ell^{\mathrm{gg}}', 'gal'),
        ('gy',     np.array(Cls_test.Cl_gal_y_tot_mat[:, 0]),
         r'C_\ell^{\mathrm{gy}}', 'ymap'),
        ('gtau',   Cl_gtau_corrected,
         r'C_\ell^{\mathrm{g\tau}}', 'tau'),
        ('gkappa', np.array(Cls_test.Cl_gal_kappa_tot_mat[:, 0, 0]),
         r'C_\ell^{\mathrm{g\kappa}}', 'kappa'),
    ]

    for label, th_arr, y_label, map_key in stats:
        cl_sim_raw = hp.anafast(maps['gal'], maps[map_key], lmax=lmax)[2:] / fsky
        if label == 'gg':
            cl_sim_raw -= shot_noise
            th_arr /= (H0/100)**4
        cl_sim = cl_sim_raw / (pixwin[2:]**2)

        th_interp = interp1d(ell_theory, th_arr, bounds_error=False, fill_value=0.0)
        th = th_interp(l_range_int)
        
        ell_check = 200
        idx_sim = np.argmin(np.abs(l_range_int - ell_check))
        idx_th = np.argmin(np.abs(ell_theory - ell_check))
        if np.abs(th_arr[idx_th]) > 1e-30 and np.abs(cl_sim[idx_sim]) > 1e-30:
            amp_ratio = cl_sim[idx_sim] / th_arr[idx_th]
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
        if np.any(np.isfinite(th_arr) & (th_arr != 0)):
            plt.plot(ell_theory, np.abs(th_arr), 'r-', lw=2.5, label='Theory')
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
