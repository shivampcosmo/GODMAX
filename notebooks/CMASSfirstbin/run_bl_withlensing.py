# =============================================================================
# 1. IMPORTS AND SETUP (WITH CONDA cuSPARSE & cuDNN ENFORCEMENT)
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
parser.add_argument("--nside", type=int)
parser.add_argument("--jdevice", type=int)
parser.add_argument("--ndevices", type=int)
parser.add_argument("--is_reference", action="store_true", help="Use default YAML params")
parser.add_argument("--theta_ej_0", type=float)
parser.add_argument("--nu_theta_ej_M", type=float)
parser.add_argument("--nu_theta_ej_z", type=float)
parser.add_argument("--mu_beta", type=float)
parser.add_argument("--sample_id", type=int)
args = parser.parse_args()

plt.rcParams['text.usetex'] = True

curr_path = pathlib.Path().absolute()
project_base = curr_path.parents[1]
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
# 2. HELPER FUNCTIONS
# =============================================================================
def split_array_uniform(arr, num_parts):
    return np.array_split(arr, num_parts)

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_dicts(data):
    return data.get('sim_params', {}), data.get('halo_params', {}), \
           data.get('analysis', {}), data.get('other_params', {})

def open_data(file):
    ldir = f'/work/hdd/bdne/spandey3/backlight/fiducial/100/halos/{snap_num_current}/'
    df = h5.File(ldir+file, 'r')
    M200c = df['M200c'][()]
    X, Y, Z = df['X'][()], df['Y'][()], df['Z'][()]
    VX, VY, VZ = df['VX'][()], df['VY'][()], df['VZ'][()]
    if (M200c.shape) is not None:
        indsel = np.where(M200c>1e13)[0]
        X_val, Y_val, Z_val = X[indsel], Y[indsel], Z[indsel]
        M200c_val, VX_val, VY_val, VZ_val = M200c[indsel], VX[indsel], VY[indsel], VZ[indsel]
        Vlos = (VX_val*X_val + VY_val*Y_val + VZ_val*Z_val)/np.sqrt(X_val**2 + Y_val**2 + Z_val**2)
    else:
        X_val, Y_val, Z_val, M200c_val, VX_val, VY_val, VZ_val, Vlos = [np.array([]) for _ in range(8)]
    df.close()
    return (X_val, Y_val, Z_val, Vlos, M200c_val)

def concatenate_data(results):
    lengths = np.array([len(result[0]) for result in results])
    total_length = lengths.sum()
    X_all, Y_all, Z_all = np.empty(total_length), np.empty(total_length), np.empty(total_length)
    Vlos_all, M200_all = np.empty(total_length), np.empty(total_length)
    end_ind_all = np.cumsum(lengths)
    start_ind_all = np.roll(end_ind_all, 1); start_ind_all[0] = 0
    for i, (start, end, result) in enumerate(zip(start_ind_all, end_ind_all, results)):
        X_all[start:end], Y_all[start:end], Z_all[start:end] = result[0], result[1], result[2]
        Vlos_all[start:end], M200_all[start:end] = result[3], result[4]
    return X_all, Y_all, Z_all, Vlos_all, M200_all

def process_halo(args):
    jhalo, ra_all_np, dec_all_np, halo_cat_R200c_np, halo_cat_DV_np, max_paint_R200c_factor, nside_local, pixel_dtype = args
    vec = hp.ang2vec(ra_all_np[jhalo], dec_all_np[jhalo], lonlat=True)
    nearby_angle = max_paint_R200c_factor * halo_cat_R200c_np[jhalo] / halo_cat_DV_np[jhalo]
    nearby_pix = hp.query_disc(nside_local, vec, nearby_angle)
    if len(nearby_pix) == 0: return None
    nearby_pix = np.asarray(nearby_pix, dtype=pixel_dtype)
    nearby_ra, nearby_dec = hp.pix2ang(nside_local, nearby_pix, lonlat=True)
    ra1, dec1 = np.radians(ra_all_np[jhalo]), np.radians(dec_all_np[jhalo])
    ra2, dec2 = np.radians(nearby_ra), np.radians(nearby_dec)
    a = np.sin((dec1 - dec2)/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra1 - ra2)/2)**2
    theta = 2 * np.arcsin(np.sqrt(a))
    distances = (halo_cat_DV_np[jhalo] * theta).astype(np.float32)
    return (nearby_pix, distances, jhalo, len(nearby_pix))

def concatenate_batch_results(results, M_all_chunk, z_all_chunk, vlos_all_chunk, halo_cat_DV_np, halo_cat_R200c_np, max_paint_R200c_factor, pixel_dtype):
    if not results: return None
    lengths = np.array([res[3] for res in results], dtype=np.int32)
    total_length = lengths.sum()
    nearby_pix_all, distances_pix_all, halo_indices = np.empty(total_length, dtype=pixel_dtype), np.empty(total_length, dtype=np.float32), np.empty(total_length, dtype=np.int32)
    end_indices = np.cumsum(lengths)
    start_indices = np.concatenate([[0], end_indices[:-1]])
    for i, (start, end, res) in enumerate(zip(start_indices, end_indices, results)):
        nearby_pix_all[start:end], distances_pix_all[start:end], halo_indices[start:end] = res[0], res[1], res[2]
    original_halo_indices = np.array([res[2] for res in results], dtype=np.int32)
    logM_ind_all, z_ind_all, vlos_ind_all = np.log(M_all_chunk[halo_indices]).astype(np.float32), z_all_chunk[halo_indices].astype(np.float32), vlos_all_chunk[halo_indices].astype(np.float32)
    ang_distance_all, rp_max_all = halo_cat_DV_np[original_halo_indices].astype(np.float32), (max_paint_R200c_factor * halo_cat_R200c_np[original_halo_indices]).astype(np.float32)
    return (nearby_pix_all, distances_pix_all, start_indices, end_indices, logM_ind_all, z_ind_all, vlos_ind_all, ang_distance_all, rp_max_all)

def final_concatenate_batches(all_results, pixel_dtype):
    total_pix, total_halos = sum(len(res[0]) for res in all_results), sum(len(res[2]) for res in all_results)
    final_nearby_pix, final_distances, final_logM, final_z, final_vlos = np.empty(total_pix, dtype=pixel_dtype), np.empty(total_pix, dtype=np.float32), np.empty(total_pix, dtype=np.float32), np.empty(total_pix, dtype=np.float32), np.empty(total_pix, dtype=np.float32)
    final_start_ind, final_end_ind, final_ang_dist, final_rp_max = np.empty(total_halos, dtype=np.int32), np.empty(total_halos, dtype=np.int32), np.empty(total_halos, dtype=np.float32), np.empty(total_halos, dtype=np.float32)
    pix_offset, halo_offset = 0, 0
    for result in all_results:
        n_pix_batch, n_halo_batch = len(result[0]), len(result[2])
        nearby_pix_b, dist_b, start_b, end_b, logM_b, z_b, vlos_b, ang_dist_b, rp_max_b = result
        final_nearby_pix[pix_offset : pix_offset + n_pix_batch], final_distances[pix_offset : pix_offset + n_pix_batch] = nearby_pix_b, dist_b
        final_logM[pix_offset : pix_offset + n_pix_batch], final_z[pix_offset : pix_offset + n_pix_batch], final_vlos[pix_offset : pix_offset + n_pix_batch] = logM_b, z_b, vlos_b
        final_start_ind[halo_offset : halo_offset + n_halo_batch], final_end_ind[halo_offset : halo_offset + n_halo_batch] = start_b + pix_offset, end_b + pix_offset
        final_ang_dist[halo_offset : halo_offset + n_halo_batch], final_rp_max[halo_offset : halo_offset + n_halo_batch] = ang_dist_b, rp_max_b
        pix_offset += n_pix_batch; halo_offset += n_halo_batch
    return (final_nearby_pix, final_distances, final_start_ind, final_end_ind, final_logM, final_z, final_vlos, final_ang_dist, final_rp_max)

def process_halos_in_batches(M_all_chunk, ra_all_chunk, dec_all_chunk, z_all_chunk, vlos_all_chunk, halo_cat_R200c, halo_cat_DA, max_paint_R200c_factor, nside, batch_size=1000):
    pixel_dtype = np.int32 if nside <= 8192 else np.int64
    ra_all_np, dec_all_np = np.clip(np.array(ra_all_chunk, dtype=np.float32), 0.01, 359.99), np.clip(np.array(dec_all_chunk, dtype=np.float32), -89.99, 89.99)
    halo_cat_R200c_np, halo_cat_DV_np = np.array(halo_cat_R200c, dtype=np.float32), np.array(halo_cat_DA, dtype=np.float32)
    n_halos, all_results = len(z_all_chunk), []
    with Pool(cpu_count()) as pool:
        for batch_start in range(0, n_halos, batch_size):
            batch_end = min(batch_start + batch_size, n_halos)
            batch_args = [(jhalo, ra_all_np, dec_all_np, halo_cat_R200c_np, halo_cat_DV_np, max_paint_R200c_factor, nside, pixel_dtype) for jhalo in range(batch_start, batch_end)]
            batch_results = pool.map(process_halo, batch_args)
            batch_results = [r for r in batch_results if r is not None]
            if batch_results:
                batch_data = concatenate_batch_results(batch_results, M_all_chunk, z_all_chunk, vlos_all_chunk, halo_cat_DV_np, halo_cat_R200c_np, max_paint_R200c_factor, pixel_dtype)
                if batch_data is not None: all_results.append(batch_data)
            gc.collect()
    return final_concatenate_batches(all_results, pixel_dtype) if all_results else None

# --- Timing Flag ---
PROFILE_TIMING = True
if PROFILE_TIMING: script_start_time = time.perf_counter()

# =============================================================================
# 3. INITIALIZATION & SETUP
# =============================================================================
nside, jdevice, Ndevices = args.nside, args.jdevice, args.ndevices
if PROFILE_TIMING: setup_start_time = time.perf_counter()

yaml_file_path = f'{str(abs_path_params)}/params_default.yaml'
data = read_yaml(yaml_file_path)
sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(data)

if args.is_reference:
    save_folder = "reference_run"
else:
    sim_params_dict.update({'theta_ej_0': args.theta_ej_0, 'nu_theta_ej_M': args.nu_theta_ej_M, 'nu_theta_ej_z': args.nu_theta_ej_z, 'mu_beta': args.mu_beta})
    save_folder = f"sample_{args.sample_id}"

halo_params_dict.update({'rmin': 0.00075, 'rmax': 10.0, 'nr': 48, 'zmin': 0.005, 'zmax': 2.1, 'nz':31, 'lg10_Mmin': 10.75, 'lg10_Mmax': 16.0, 'nM': 40})
cosmo_params_dict = {'w0': -1.0, 'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624}

Prof_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
mock_params_setup = {'nside': nside, 'get_ymap': True, 'get_taumap': True, 'get_kappamap': True, 'get_baryonifiedmap': True, 'get_galmap': True, 'smooth_profiles': True}
Prof_test = setup_sim_map(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, mock_params_setup, Profiles_obj=Prof_test)

chi_CMB = bkgrd.radial_comoving_distance(Prof_test.cosmo_jax, 1.0 / (1.0 + 1089.0)).item()
c_light = 299792.458
rho_m = Prof_test.get_rho_m(0.0)
part_mass = float((rho_m * (1000)**3)/2048**3)

if PROFILE_TIMING: setup_end_time = time.perf_counter(); print(f"[PROFILE] Setup: {setup_end_time - setup_start_time:.2f}s")

sim_file_path = '/work/hdd/bdne/spandey3/backlight/fiducial/100'
zlist = np.loadtxt(f'{sim_file_path}/zlist.txt')
snap_num_all, zval_all = zlist[:,0].astype(int), zlist[:,1]
zmax_maps, zmin_maps = 0.5, 0.3
snaps_in_shell = snap_num_all[(zval_all < zmax_maps) & (zval_all > zmin_maps)]

# =============================================================================
# 4. REDSHIFT LOOP
# =============================================================================
for snap_num in snaps_in_shell:
    if PROFILE_TIMING: z_slice_start_time = time.perf_counter()
    snap_num_current = snap_num
    zval = zval_all[snap_num_all == snap_num][0]
    save_map_fname = f'/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/{save_folder}/allmaps_nside{args.nside}_z{zval:.3f}_split{args.jdevice}.pkl'

    if not os.path.exists(save_map_fname):
        if PROFILE_TIMING: print(f"\nProcessing z = {zval:.3f}")
        chi_v = bkgrd.radial_comoving_distance(Prof_test.cosmo_jax, 1.0 / (1.0 + zval)).item()
        
        # Physics Correction: Calculate Snapshot thickness for volume integration 
        dz_snapshot = (zmax_maps - zmin_maps) / len(snaps_in_shell)
        H_z = cosmo_params_dict['H0'] * Prof_test.get_Ez(zval)
        dchi_snapshot = (c_light / H_z) * dz_snapshot
        
        # Matter Density Normalization
        vol_shell_pix = (4.0/3.0) * np.pi * ((chi_v + dchi_snapshot/2)**3 - (chi_v - dchi_snapshot/2)**3) / (12 * nside**2)
        mean_mass_per_pix = rho_m * vol_shell_pix

        map_tot_orig = hp.read_map(f'{sim_file_path}/compressed_massMaps/massSheet_tot_z_{int(snap_num)}.fits.gz', verbose=False)
        map_tot_orig_ds = hp.ud_grade(np.array(map_tot_orig * part_mass), nside, power=-2)
        # Convert raw mass to dimensionless overdensity delta = (rho - rho_bar)/rho_bar
        delta_sheet = (map_tot_orig_ds / mean_mass_per_pix) - 1.0

        if PROFILE_TIMING: load_start = time.perf_counter()
        files_all = os.listdir(f'{sim_file_path}/halos/{snap_num}/')
        with Pool(cpu_count()) as pool: results = pool.map(open_data, files_all)
        X_all, Y_all, Z_all, vlos_all, M200c_all = concatenate_data(results)

        ra_all, dec_all = hp.vec2ang(np.array([X_all, Y_all, Z_all]).T, lonlat=True)
        indsel = np.where((M200c_all < 6e15) & (M200c_all > (10**12.0)))[0]
        split_idx = np.array_split(indsel[np.flip(np.argsort(M200c_all[indsel]))], Ndevices)[jdevice]
        ra_s, dec_s, M_s, vlos_s, z_s = ra_all[split_idx], dec_all[split_idx], M200c_all[split_idx], vlos_all[split_idx], zval * np.ones_like(ra_all[split_idx])

        map_rhom_dmb, map_rhom_dmo = [np.zeros(12 * nside**2, dtype=np.float32) for _ in range(2)]
        map_ymap, map_tau, map_kappa_halo = [np.zeros(12 * nside**2, dtype=np.float32) for _ in range(3)]
        mock_gals_all = {}

        for i in tqdm(range(int(np.ceil(len(M_s)/1e5))), desc=f"Painting"):
            sl = slice(i*100000, (i+1)*100000)
            Mc, rac, decc, vc = M_s[sl], ra_s[sl], dec_s[sl], vlos_s[sl]
            zc = z_s[sl]
            rho_c = constants.RHO_CRIT_0_KPC3 * Prof_test.get_Ez(zc[0])**2 * 1e9
            R200c = (Mc * 3.0 / (4.0 * jnp.pi * 200 * rho_c))**(1.0/3.0)
            DA = bkgrd.angular_diameter_distance(Prof_test.cosmo_jax, 1./(1.+zc))

            res = process_halos_in_batches(Mc, rac, decc, zc, vc, R200c, DA, 3.0, nside)
            if res:
                m_p = {**mock_params_setup, 'halo_z': jnp.array(zc), 'halo_ra': jnp.array(rac), 'halo_dec': jnp.array(decc), 'halo_M': jnp.array(Mc), 'halo_vlos': jnp.array(vc), 'nearby_pix_all': jnp.array(res[0]), 'start_ind': jnp.array(res[2]), 'end_ind': jnp.array(res[3]), 'pix_prop_all': (jnp.array([np.log(res[1]), res[5], res[4], res[6]]).T), 'ang_distance_all': jnp.array(res[7]), 'rp_max_all': jnp.array(res[8])}
                mock_map = get_sim_map(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, m_p, Profiles_obj=Prof_test)

                map_rhom_dmb += np.nan_to_num(mock_map.rhommap_final)
                map_rhom_dmo += np.nan_to_num(mock_map.rhom_dmo_map_final)
                # Volume integration: Weight LOS quantities by shell thickness
                map_ymap += np.nan_to_num(mock_map.ymap_final) * dchi_snapshot
                map_tau += np.nan_to_num(mock_map.taumap_final) * dchi_snapshot
                # Overdensity fix: (rho_halo - rho_bar)/rho_bar
                map_kappa_halo += (np.nan_to_num(mock_map.rhommap_final) - np.nan_to_num(mock_map.rhom_dmo_map_final)) / mean_mass_per_pix
                mock_gals_all[i] = mock_map.final_galaxy_catalog
                jax.clear_caches(); gc.collect()

        weight_k = (1.5 * (cosmo_params_dict['H0']/c_light)**2 * cosmo_params_dict['Om0']) * ((chi_CMB - chi_v)/chi_CMB) * (1.0 + zval) * chi_v * dchi_snapshot
        map_kappa = (delta_sheet + map_kappa_halo) * weight_k

        if PROFILE_TIMING:
            print(f"[PROFILE] Snapshot Verification: Mean Kappa: {np.mean(map_kappa):.2e} | Mean Y: {np.mean(map_ymap):.2e}")
            save_start = time.perf_counter()

        with open(save_map_fname, 'wb') as f:
            pk.dump({'mock_gals_all': mock_gals_all, 'map_ymap': map_ymap, 'map_tau': map_tau, 'map_kappa': map_kappa, 
                     'map_rhom_dmb': map_rhom_dmb, 'map_rhom_dmo': map_rhom_dmo}, f)

        if PROFILE_TIMING: print(f"[PROFILE] Save & total: {time.perf_counter() - z_slice_start_time:.2f}s")
