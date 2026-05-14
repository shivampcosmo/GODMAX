# =============================================================================
# COMPARE REFERENCE RUN SIMULATED Cls vs THEORY
# =============================================================================
import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pickle as pk
import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d

plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 16, 'legend.fontsize': 14})

# --- Paths ---
pasting_dir = "/work/hdd/bdne/aacharya2/GODMAX/notebooks/pasting"
if pasting_dir not in sys.path:
    sys.path.append(pasting_dir)

curr_path = pathlib.Path().absolute()
project_base = curr_path.parents[3]
for p in [curr_path, project_base / "src"]:
    sys.path.append(str(p))

from paste_backlight_utils import (
    get_project_paths, build_config, make_galaxy_map,
    compute_shot_noise_Cl, compute_Cl_ratio_in_bands,
    compute_Cl_gg_1h_2h, compute_hod_shot_noise_Cl,)

from base_class import base_class
from get_radial_profiles import Profiles
from get_Pkzs import get_Pkz
from get_Cls import get_Cl

paths = get_project_paths()

# =============================================================================
# 1. CONFIG: match simulation script exactly
# =============================================================================
nside         = 1024
gal_zmin      = 0.3
gal_zmax      = 0.5
nbar_comoving = 1e-4

(sim_params_dict, halo_params_dict, analysis_dict,
 other_params_dict, cosmo_jax, zarray_lens, nz_lens, gal_zrange) = build_config(
    paths["params"], paths["data"],
    nbar_comoving=nbar_comoving, gal_zmin=gal_zmin, gal_zmax=gal_zmax)

gal_zmin, gal_zmax = gal_zrange
print(f"Galaxy z-range: [{gal_zmin}, {gal_zmax}]")

# =============================================================================
# 2. THEORY PIPELINE: matches simulation script exactly
# =============================================================================
base_test     = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
profiles_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict,
                         base_class_obj=base_test)
Pkz_test = get_Pkz(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict,
                   Profiles_obj=profiles_test)
Cls_test = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict,
                  Pkz_obj=Pkz_test)

# =============================================================================
# 3. LOAD REFERENCE RUN MAPS
# =============================================================================
sdir           = f"/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/new/n1024/reference_run"
save_map_fname = f"{sdir}/allmaps_sim_B12_nside{nside}.pkl"
outputfolder   = f"/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/Clplots"

print(f"Loading: {save_map_fname}")
saved_data = pk.load(open(save_map_fname, "rb"))
print(f"Available keys: {list(saved_data.keys())}")

mock_gals = np.array(saved_data["mock_gals_all"][0])
map_tSZ   = saved_data["map_ymap"]
map_gal   = make_galaxy_map(mock_gals, nside, zmin=gal_zmin, zmax=gal_zmax)

# =============================================================================
# 4. GALAXY OVERDENSITY & SHOT NOISE
# =============================================================================
delta_gal    = map_gal / np.mean(map_gal)

Cl_shot, n_gal, nbar_sr = compute_shot_noise_Cl(mock_gals, nside, gal_zmin, gal_zmax)
Cl_shot_hod  = compute_hod_shot_noise_Cl(Cls_test)

print(f"n_gal:         {n_gal:.0f}")
print(f"nbar [sr^-1]:  {nbar_sr:.4e}")
print(f"Cl_shot (map): {Cl_shot:.4e}")
print(f"Cl_shot (HOD): {float(Cl_shot_hod):.4e}")

# =============================================================================
# 5. MEASURE Cl^{gg}
# =============================================================================
Cl_gg_raw     = hp.anafast(delta_gal, lmax=2*nside)
ell           = np.arange(len(Cl_gg_raw))
Cl_gg_no_shot = Cl_gg_raw - Cl_shot

ell_th    = np.array(Cls_test.ell_array)
Cl_th     = np.array(Cls_test.Cl_gal_gal_tot_mat[:, 0, 0])
cl_decomp = compute_Cl_gg_1h_2h(Cls_test)

# =============================================================================
# 6. PLOT Cl^{gg}: 3-panel
# =============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))

ax1.plot(ell,    Cl_gg_raw,                        'gray', alpha=0.4, lw=0.5, label='Measured raw')
ax1.plot(ell,    np.maximum(Cl_gg_no_shot, 1e-8),  'k-',  lw=1.2,           label='Meas $-$ shot')
ax1.plot(ell_th, Cl_th,                            'b-',  lw=2,             label='Theory total')
ax1.plot(cl_decomp['ell'], cl_decomp['Cl_2h'],     'b--', lw=1.2, alpha=0.7, label='Theory 2-halo')
ax1.plot(cl_decomp['ell'], cl_decomp['Cl_1h'],     'b:',  lw=1.2, alpha=0.7, label='Theory 1-halo')
ax1.axhline(Cl_shot, color='r', ls=':', lw=1, label=f'Shot noise = {Cl_shot:.2e}')
ax1.set(xscale='log', yscale='log', xlabel=r'$\ell$', ylabel=r'$C_\ell^{\mathrm{gg}}$',
        xlim=(80, 1500), ylim=(1e-6, 1e-2))
ax1.legend(fontsize=12)
ax1.set_title(r'$C_\ell^{gg}$: measured vs theory')

Cl_th_at_ell = interp1d(ell_th, Cl_th, bounds_error=False, fill_value=np.nan)(ell)
ratio        = np.where(Cl_th_at_ell > 0, Cl_gg_no_shot / Cl_th_at_ell, np.nan)
mask_ratio   = (ell >= 80) & (ell <= 1200) & (Cl_gg_no_shot > 0)
ax2.plot(ell[mask_ratio], ratio[mask_ratio], 'k-', lw=0.8)
ax2.axhline(1.0, color='b', ls='-', lw=1)
ax2.axhspan(0.9, 1.1, alpha=0.1, color='blue')
ax2.set(xscale='log', xlabel=r'$\ell$', ylabel=r'$C_\ell^{\mathrm{meas}} / C_\ell^{\mathrm{th}}$',
        xlim=(80, 1200), ylim=(0.0, 3.0))
ax2.set_title('Ratio (shot-subtracted / theory)')

ax3.plot(ell,    Cl_gg_raw,             'gray', alpha=0.5, lw=0.8, label='Raw measured')
ax3.plot(ell_th, Cl_th + Cl_shot,       'r-',  lw=2,              label='Theory + shot noise (map)')
ax3.plot(ell_th, Cl_th + Cl_shot_hod,  'r--', lw=1.5,            label='Theory + shot noise (HOD)')
ax3.set(xscale='log', yscale='log', xlabel=r'$\ell$', ylabel=r'$C_\ell^{\mathrm{gg}}$',
        xlim=(80, 1500), ylim=(1e-6, 1e-2))
ax3.legend(fontsize=12)
ax3.set_title('Shot noise comparison')

plt.tight_layout()
plt.savefig(f"{outputfolder}/Cl_gg_sim_vs_theory_nside{nside}.png", dpi=150, bbox_inches='tight')
plt.show()

band_ratios_gg = compute_Cl_ratio_in_bands(
    ell, np.maximum(Cl_gg_no_shot, 1e-20), ell_th, Cl_th)

# =============================================================================
# 7. MEASURE Cl^{gy}: 2-panel
# =============================================================================
Cl_gy_raw    = hp.anafast(delta_gal, map_tSZ, lmax=2*nside)
pixwin       = hp.pixwin(nside)[:len(Cl_gy_raw)]
Cl_gy_deconv = Cl_gy_raw / np.where(pixwin > 0, pixwin, 1.0)

ell_th_gy = np.array(Cls_test.ell_array)
Cl_th_gy  = np.array(Cls_test.Cl_gal_y_tot_mat[:, 0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(ell,       Cl_gy_deconv, 'k-', lw=0.8, label='Measured (deconvolved)')
ax1.plot(ell_th_gy, Cl_th_gy,    'b-', lw=2,   label='Theory')
ax1.set(xscale='log', yscale='log', xlabel=r'$\ell$', ylabel=r'$C_\ell^{\mathrm{gy}}$',
        xlim=(100, 1000), ylim=(1e-14, 1e-10))
ax1.legend(fontsize=12)
ax1.set_title(r'$C_\ell^{gy}$: measured vs theory')

Cl_th_gy_at_ell = interp1d(ell_th_gy, Cl_th_gy, bounds_error=False, fill_value=np.nan)(ell)
mask_gy  = (ell >= 80) & (ell <= 1000) & (Cl_gy_deconv > 0) & np.isfinite(Cl_th_gy_at_ell) & (Cl_th_gy_at_ell > 0)
ratio_gy = Cl_gy_deconv / np.where(Cl_th_gy_at_ell > 0, Cl_th_gy_at_ell, np.nan)
ax2.plot(ell[mask_gy], ratio_gy[mask_gy], 'k-', lw=0.8)
ax2.axhline(1.0, color='b', ls='-', lw=1)
ax2.axhspan(0.9, 1.1, alpha=0.1, color='blue')
ax2.set(xscale='log', xlabel=r'$\ell$', ylabel=r'$C_\ell^{\mathrm{meas}} / C_\ell^{\mathrm{th}}$',
        xlim=(100, 1000), ylim=(0, 3))
ax2.set_title('Ratio (gy)')

plt.tight_layout()
plt.savefig(f"{outputfolder}/Cl_gy_sim_vs_theory_nside{nside}.png", dpi=150, bbox_inches='tight')
plt.show()

_ = compute_Cl_ratio_in_bands(ell, np.maximum(Cl_gy_deconv, 1e-20), ell_th_gy, Cl_th_gy)
