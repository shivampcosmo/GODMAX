# =============================================================================
# Q: Do theory and HOD data vectors agree at the fiducial point?
# =============================================================================
import os, sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pickle as pk
import jax.numpy as jnp
from scipy.interpolate import interp1d

pasting_dir = "/work/hdd/bdne/aacharya2/GODMAX/notebooks/pasting"
if pasting_dir not in sys.path:
    sys.path.append(pasting_dir)

curr_path    = pathlib.Path().absolute()
project_base = curr_path.parents[4]
for p in [curr_path, project_base / "src"]:
    sys.path.append(str(p))

from paste_backlight_utils import get_project_paths, build_config, make_galaxy_map, compute_shot_noise_Cl
from base_class          import base_class
from get_radial_profiles import Profiles
from get_Pkzs            import get_Pkz
from get_Cls             import get_Cl

paths = get_project_paths()

# =============================================================================
# 1. THEORY PIPELINE
# =============================================================================
nside = 1024
gal_zmin, gal_zmax, nbar_comoving = 0.3, 0.5, 1e-4

(sim_params_dict, halo_params_dict, analysis_dict,
 other_params_dict, cosmo_jax, zarray_lens, nz_lens, gal_zrange) = build_config(
    paths["params"], paths["data"],
    nbar_comoving=nbar_comoving, gal_zmin=gal_zmin, gal_zmax=gal_zmax)

gal_zmin, gal_zmax   = gal_zrange
halo_params_dict['zmin'] = 0.2

base_test     = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
profiles_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict,
                         base_class_obj=base_test)

M_halo_MIN        = 10**12.75
halo_selM_mask    = jnp.where(profiles_test.M_array > M_halo_MIN, 1.0, 0.0)
halo_selM_mask_2d = jnp.tile(halo_selM_mask, (halo_params_dict['nz'], 1))
profiles_test.Ncen_mat = profiles_test.Ncen_mat * halo_selM_mask_2d
profiles_test.Nsat_mat = profiles_test.Nsat_mat * halo_selM_mask_2d

Pkz_test = get_Pkz(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Profiles_obj=profiles_test)
Cls_test = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Pkz_obj=Pkz_test)

ell_th       = np.array(Cls_test.ell_array)
Cl_th_gg     = np.array(Cls_test.Cl_gal_gal_tot_mat[:, 0, 0])
Cl_th_gy     = np.array(Cls_test.Cl_gal_y_tot_mat[:, 0])
Cl_th_gtau   = np.array(Cls_test.Cl_gal_tau_tot_mat[:, 0])
Cl_th_gkappa = np.array(Cls_test.Cl_gal_kappa_tot_mat[:, 0, 0])

# =============================================================================
# 2. HOD SIMULATION Cls  (reference run)
# =============================================================================
sdir         = "/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/new/n1024/reference_run"
saved_data   = pk.load(open(f"{sdir}/allmaps_sim_B12_nside{nside}.pkl", "rb"))

mock_gals = np.array(saved_data["mock_gals_all"][0])
map_tSZ   = np.array(saved_data["map_ymap"])
map_tSZ   = map_tSZ - np.mean(map_tSZ)
map_tau   = np.array(saved_data["map_tau"])
map_tau   = map_tau - np.mean(map_tau)
map_kappa = np.array(saved_data["map_kappa"])
map_kappa = map_kappa - np.mean(map_kappa)
map_gal   = make_galaxy_map(mock_gals, nside, zmin=gal_zmin, zmax=gal_zmax)
delta_gal = map_gal / np.mean(map_gal) - 1.0

Cl_shot, n_gal, nbar_sr = compute_shot_noise_Cl(mock_gals, nside, gal_zmin, gal_zmax)
pixwin = hp.pixwin(nside, lmax=2 * nside)

ell               = np.arange(2 * nside + 1)
Cl_gg_sim         = hp.anafast(delta_gal,          lmax=2*nside) - Cl_shot
Cl_gy_sim         = hp.anafast(delta_gal, map_tSZ,   lmax=2*nside) / np.where(pixwin>0, pixwin, 1.)
Cl_gtau_sim       = hp.anafast(delta_gal, map_tau,   lmax=2*nside) / np.where(pixwin>0, pixwin, 1.)
Cl_gkappa_sim     = hp.anafast(delta_gal, map_kappa, lmax=2*nside) / np.where(pixwin>0, pixwin, 1.)

# =============================================================================
# 3. RATIO: HOD sim / theory  at each theory ell
# =============================================================================
def ratio_at_theory_ell(ell_sim, cl_sim, ell_th, cl_th):
    cl_sim_interp = interp1d(ell_sim, cl_sim, bounds_error=False, fill_value=np.nan)(ell_th)
    return cl_sim_interp / np.where(np.abs(cl_th) > 0, cl_th, np.nan)

ratio_gg     = ratio_at_theory_ell(ell, Cl_gg_sim,     ell_th, Cl_th_gg)
ratio_gy     = ratio_at_theory_ell(ell, Cl_gy_sim,     ell_th, Cl_th_gy)
ratio_gtau   = ratio_at_theory_ell(ell, Cl_gtau_sim,   ell_th, Cl_th_gtau)
ratio_gkappa = ratio_at_theory_ell(ell, Cl_gkappa_sim, ell_th, Cl_th_gkappa)

print(f"\n{'ell':>8}  {'gg':>8}  {'gy':>8}  {'gtau':>8}  {'gkappa':>8}")
print("-" * 50)
for i, el in enumerate(ell_th):
    print(f"{el:8.1f}  {ratio_gg[i]:8.3f}  {ratio_gy[i]:8.3f}  "
          f"{ratio_gtau[i]:8.3f}  {ratio_gkappa[i]:8.3f}")

print("Theory Cl_gy   :", Cl_th_gy[:3])
print("Theory Cl_gtau :", Cl_th_gtau[:3])
print("Theory Cl_gg   :", Cl_th_gg[:3])
print("Theory Cl_gkappa:", Cl_th_gkappa[:3])

print("\nSim Cl_gy   (raw, first 3 nonzero ells):", Cl_gy_sim[100:103])
print("Sim Cl_gtau (raw, first 3 nonzero ells):", Cl_gtau_sim[100:103])
print("Sim Cl_gg   (raw, first 3 nonzero ells):", Cl_gg_sim[100:103])
print("Sim Cl_gkappa (raw, first 3 nonzero ells):", Cl_gkappa_sim[100:103])

print("\nRatio Cl_th_gtau / Cl_th_gy :", (Cl_th_gtau / Cl_th_gy)[:5])
print("Ratio Cl_sim_gtau / Cl_sim_gy:", (Cl_gtau_sim[100:105] / Cl_gy_sim[100:105]))
# =============================================================================
# 4. ONE FIGURE: 4 ratio panels
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)

for ax, ratio, label in zip(axes,
    [ratio_gg, ratio_gy, ratio_gtau, ratio_gkappa],
    [r'$C_\ell^{gg}$', r'$C_\ell^{gy}$', r'$C_\ell^{g\tau}$', r'$C_\ell^{g\kappa}$']):
    ax.plot(ell_th, ratio, 'ko-', ms=5, lw=1)
    ax.axhline(1.0, color='b', lw=1)
    ax.axhspan(0.9, 1.1, alpha=0.1, color='blue', label=r'$\pm10\%$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylim(0, 3)
    ax.set_title(label)
    ax.legend(fontsize=11)

axes[0].set_ylabel('HOD sim / Theory')
fig.suptitle('Do HOD and theory data vectors agree at fiducial?', fontsize=13)
plt.tight_layout()
outputfolder = "/work/hdd/bdne/aacharya2/GODMAX/notebooks/CMASSfirstbin/new/n1024/Clplots"
os.makedirs(outputfolder, exist_ok=True)
plt.savefig(f"{outputfolder}/fiducial_agreement_sim_vs_theory.png", dpi=150, bbox_inches='tight')
plt.show()
