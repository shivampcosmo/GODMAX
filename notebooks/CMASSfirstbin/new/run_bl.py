# =============================================================================
# 1. IMPORT MODULES
# =============================================================================
import os
import sys
import ctypes
import glob

# --- THE cuSPARSE & cuDNN CONDA FIX ---
conda_lib_dir = os.path.join(sys.prefix, "lib")
def force_load_lib(name_pattern):
    libs = glob.glob(os.path.join(conda_lib_dir, name_pattern))
    if libs:
        libs.sort()
        try:
            ctypes.CDLL(libs[-1], mode=ctypes.RTLD_GLOBAL)
            print(f"Successfully force-loaded {os.path.basename(libs[-1])}")
        except Exception as e:
            print(f"Warning: Failed to load {name_pattern}: {e}")

force_load_lib("libcusparse.so*")
force_load_lib("libcudnn.so*")

import pathlib
import re
import gc
import yaml
import warnings
import time
from multiprocessing import Pool, cpu_count
import ast
from functools import partial

warnings.filterwarnings("ignore")
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
from jax.lib import xla_bridge
import jax.numpy as jnp
import jax_cosmo.background as bkgrd

platform = xla_bridge.get_backend().platform
jax.config.update('jax_platform_name', platform)
jax.config.update("jax_enable_x64", True)
print(f"JAX backend: {platform}, Total devices: {jax.device_count()}, Local devices: {jax.local_device_count()}")

import numpyro
numpyro.set_platform("gpu")
numpyro.enable_x64()

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import healpy as hp
from tqdm import tqdm
import pickle as pk
import h5py as h5
import argparse

# --- Argument Parsing ---
parser = argparse.ArgumentParser()
parser.add_argument("--nside", type=int, default=512)
parser.add_argument("--jdevice", type=int, default=0)
parser.add_argument("--ndevices", type=int, default=1)
parser.add_argument("--is_reference", action="store_true", help="Use default YAML params")
parser.add_argument("--is_validation",  action="store_true", help="using LH set for validation")
parser.add_argument("--theta_ej_0", type=float)
parser.add_argument("--nu_theta_ej_M", type=float)
parser.add_argument("--nu_theta_ej_z", type=float)
parser.add_argument("--mu_beta", type=float)
parser.add_argument("--sample_id", type=int)
args = parser.parse_args()

plt.rcParams['text.usetex'] = True

curr_path = pathlib.Path().absolute()
project_base = curr_path.parents[2]
abs_path_data = project_base / "data"
abs_path_src = project_base / "src"
abs_path_results = project_base / "results"
abs_path_params = project_base / "param_files"

for path in [curr_path, abs_path_data, abs_path_src, abs_path_results, abs_path_params]:
    sys.path.append(str(path))

from get_radial_profiles import Profiles
from get_sim_maps import setup_sim_map, get_sim_map
import helpers.constants as constants

# =============================================================================
# 2. NOTEBOOK UTILS & CONFIGURATION
# =============================================================================
pasting_dir = "/work/hdd/bdne/aacharya2/GODMAX/notebooks/pasting"
if pasting_dir not in sys.path:
    sys.path.append(pasting_dir)
# Import the notebook's utility wrappers
from paste_backlight_utils import (
    get_project_paths, build_config, load_halo_catalog,
    generate_maps, make_galaxy_map, compute_ngal_vs_z,
    stack_snapshot_maps, check_nz_nbar_consistency, nbar_to_nz_lens,
    update_nz_from_mock_catalog, measure_hod_from_catalog,
    compute_shot_noise_Cl, compute_Cl_ratio_in_bands,
    compare_sim_vs_theory_hmf, compute_Cl_gg_1h_2h, 
    compute_hod_shot_noise_Cl, print_diagnostic_summary,compute_kappa_map
)

paths = get_project_paths()
nside_map = args.nside

# Apply your custom redshift bounds
gal_zmin = 0.3
gal_zmax = 0.5
nbar_comoving = 1e-4

(sim_params_dict, halo_params_dict, analysis_dict,
 other_params_dict, cosmo_jax, zarray_lens, nz_lens, gal_zrange) = build_config(
    paths["params"], paths["data"], nbar_comoving=nbar_comoving, gal_zmin=gal_zmin, gal_zmax=gal_zmax)

gal_zmin, gal_zmax = gal_zrange
print(f"Galaxy z-range: [{gal_zmin}, {gal_zmax}]")

# -----------------------------------------------------------------
# APPLY CSV OVERRIDES OR REFERENCE DEFAULTS
# -----------------------------------------------------------------
if args.is_reference:
    print("Running REFERENCE simulation with default YAML parameters...")
    save_folder = "reference_run2"
elif args.is_validation:
    print("Running the validation set with Latin Hypercube generated values")
    update_dict = {}
    if args.theta_ej_0 is not None: update_dict['theta_ej_0'] = args.theta_ej_0
    if args.nu_theta_ej_M is not None: update_dict['nu_theta_ej_M'] = args.nu_theta_ej_M
    if args.nu_theta_ej_z is not None: update_dict['nu_theta_ej_z'] = args.nu_theta_ej_z
    if args.mu_beta is not None: update_dict['mu_beta'] = args.mu_beta

    sim_params_dict.update(update_dict)
    save_folder = f"validation_{args.sample_id}"
else:
    print(args.nu_theta_ej_M)
    update_dict = {}
    if args.theta_ej_0 is not None: update_dict['theta_ej_0'] = args.theta_ej_0
    if args.nu_theta_ej_M is not None: update_dict['nu_theta_ej_M'] = args.nu_theta_ej_M
    if args.nu_theta_ej_z is not None: update_dict['nu_theta_ej_z'] = args.nu_theta_ej_z
    if args.mu_beta is not None: update_dict['mu_beta'] = args.mu_beta
    
    sim_params_dict.update(update_dict)
    save_folder = f"sample_{args.sample_id}"

# =============================================================================
# 3. INITIALIZE MODEL PIPELINE & HOD MASKING
# =============================================================================
from base_class import base_class
from get_radial_profiles import Profiles
from get_Pkzs import get_Pkz
from get_Cls import get_Cl

base_test = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
profiles_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, base_class_obj=base_test)

# Apply your custom 10^12.75 Mass Cut
M_halo_MIN = 10**12.75
halo_selM_mask = jnp.where(profiles_test.M_array > M_halo_MIN, 1.0, 0.0)
halo_selM_mask_2d = jnp.tile(halo_selM_mask, (halo_params_dict['nz'], 1))
halo_sel_mask_2d = halo_selM_mask_2d

profiles_test.Ncen_mat = profiles_test.Ncen_mat * halo_sel_mask_2d
profiles_test.Nsat_mat = profiles_test.Nsat_mat * halo_sel_mask_2d

Pkz_test = get_Pkz(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Profiles_obj=profiles_test)
Cls_test = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Pkz_obj=Pkz_test)

# =============================================================================
# 4. SETUP MAP GENERATOR
# =============================================================================
PROFILE_TIMING = True
halo_params_dict_copy = halo_params_dict.copy()
halo_params_dict_copy.update({
    'rmin': 0.005, 'rmax': 10.0, 'nr': 48,
    'zmin': 0.005, 'zmax': 0.8, 'nz': 52,
    'lg10_Mmin': 11.75, 'lg10_Mmax': 16.0, 'nM': 42, # Base grid wider than mask
})

mock_params_dict_setup = {
    "nside": args.nside,"get_baryonifiedmap": True,
    "get_ymap": True, "get_kSZmap": True, "get_taumap": True,
    "get_kappamap": True, "get_galmap": True, "smooth_profiles": True,
}

Prof_test = setup_sim_map(
    sim_params_dict, halo_params_dict_copy, analysis_dict,
    other_params_dict, mock_params_dict_setup, Profiles_obj=profiles_test)

# =============================================================================
# 5. LOAD HALO CATALOG & GENERATE MAPS
# =============================================================================
sdir = f"/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/new/{save_folder}"
os.makedirs(sdir, exist_ok=True)
save_map_fname = f"{sdir}/allmaps_sim_B12_nside{args.nside}.pkl"

if os.path.exists(save_map_fname):
    print(f"File {save_map_fname} already exists. Skipping.")
    sys.exit(0)

# Note: The notebook utils expect a single, pre-compiled HDF5 catalog 
# rather than looping over PKDGRAV snap_num folders.
catalog_path = f"{paths['data']}/backlight/halo_catalog_Mlim_1e12.75_zlim_0.3_0.5.h5"

if not os.path.exists(catalog_path):
    print(f"\nERROR: Expected aggregated catalog at {catalog_path} not found.")
    print("Because you removed the snap_num loop, you must pre-aggregate the catalog.")
    sys.exit(1)

ra_all, dec_all, z_all, M200c_all, vlos_all = load_halo_catalog(catalog_path)

print(len(ra_all))
print(f"RA range before adjustment: [{ra_all.min()}, {ra_all.max()}]")

# Notebook standard coordinate clipping
indsel = np.where((ra_all > 2e-5) & (ra_all < 360-2e-5) & (dec_all > -90+2e-5) & (dec_all < 90-2e-5))
ra_all = ra_all[indsel]
dec_all = dec_all[indsel]
z_all = z_all[indsel]
M200c_all = M200c_all[indsel]
vlos_all = vlos_all[indsel]

print(f"RA range after clipping: [{ra_all.min()}, {ra_all.max()}]")
print(len(ra_all))
print(f"Number of halos: {len(M200c_all)}")
print(f"log10(M200c) --> min: {np.log10(M200c_all).min():.2f}, mean: {np.log10(M200c_all).mean():.2f}, max: {np.log10(M200c_all).max():.2f}")
print(f"Mean z: {z_all.mean():.2f}")

# The notebook wraps the entire painting loop in these calls
# Step 1: generate halo-painted maps — save_path=None until kappa is appended
saved_data = generate_maps(ra_all, dec_all, z_all, M200c_all, vlos_all,
    Prof_test, mock_params_dict_setup, args.nside,
    sim_params_dict, halo_params_dict_copy, analysis_dict, other_params_dict,
    save_path=None,profile_timing=PROFILE_TIMING,)

# Step 2: compute kappa from N-body sheets + baryonic halo correction
rho_m     = Prof_test.get_rho_m(0.0)
part_mass = float((rho_m * 1000.0**3) / 2048.0**3)
chi_CMB = float(bkgrd.radial_comoving_distance(Prof_test.cosmo_jax, 1.0 / (1.0 + 1089.0))[0])
sim_file_path = '/work/hdd/bdne/spandey3/backlight/fiducial/100'
zlist         = np.loadtxt(f'{sim_file_path}/zlist.txt')
snap_num_all  = zlist[:, 0].astype(int)
zval_all      = zlist[:, 1]

map_kappa = compute_kappa_map(
    sim_file_path     = sim_file_path,
    snap_num_all      = snap_num_all,
    zval_all          = zval_all,
    zmin              = gal_zmin,
    zmax              = gal_zmax,
    Prof_test         = Prof_test,
    nside             = args.nside,
    part_mass         = part_mass,
    chi_CMB           = chi_CMB,
    cosmo_params_dict = {'Om0': 0.3175},
    map_rhom_dmb      = saved_data['map_rhom_dmb'],
    unbound_cache_path = f'{sdir}/map_kappa_unbound.npy',
    weight_cache_path  = f'{sdir}/weight_eff_kappa.npy',)

# Step 3: insert kappa and write the final pkl
saved_data['map_kappa'] = map_kappa
with open(save_map_fname, 'wb') as f:
    pk.dump(saved_data, f)
print(f"Saved to {save_map_fname} with keys: {list(saved_data.keys())}")

print("\nMap generation successfully completed.")
