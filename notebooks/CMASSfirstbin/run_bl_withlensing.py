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
def split_array_power_ratio(arr, num_parts, power_val=0.5):
    """Splits an array into parts with sizes proportional to j^power_val."""
    n_total = arr.shape[0]
    j_values = np.arange(1, num_parts + 1)
    weights = j_values**power_val
    sum_of_weights = np.sum(weights)
    ideal_sizes = (n_total / sum_of_weights) * weights
    
    int_sizes = np.floor(ideal_sizes).astype(int)
    remainder = n_total - np.sum(int_sizes)
    
    # Distribute remainder based on fractional parts
    fractional_parts = ideal_sizes - int_sizes
    indices_to_increment = np.argsort(fractional_parts)[-remainder:]
    int_sizes[indices_to_increment] += 1
    
    split_indices = np.cumsum(int_sizes)[:-1]
    return np.split(arr, split_indices)

def split_array_uniform(arr, num_parts):
    """Splits an array into parts with nearly equal numbers of halos."""
    # np.array_split handles cases where total halos aren't perfectly divisible
    return np.array_split(arr, num_parts)

def read_yaml(file_path):
    """Reads a YAML file and returns its content."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_dicts(data):
    """Extracts parameter dictionaries from the main data dictionary."""
    sim_params_dict = data.get('sim_params', {})
    halo_params_dict = data.get('halo_params', {})
    analysis_dict = data.get('analysis', {})
    other_params_dict = data.get('other_params', {})
    return sim_params_dict, halo_params_dict, analysis_dict, other_params_dict

# --- Functions for Parallel Halo Processing ---

def open_data(file):
    df = h5.File(ldir+file, 'r')
    M200c = df['M200c'][()]
    X, Y, Z = df['X'][()], df['Y'][()], df['Z'][()]
    VX, VY, VZ = df['VX'][()], df['VY'][()], df['VZ'][()]
    if (M200c.shape) is not None:
        indsel = np.where(M200c>1e13)[0]
        X_val = X[indsel]
        Y_val = Y[indsel]
        Z_val = Z[indsel]
        M200c_val = M200c[indsel]
        VX_val = VX[indsel]
        VY_val = VY[indsel]
        VZ_val = VZ[indsel]
        Vlos = (VX_val*X_val + VY_val*Y_val + VZ_val*Z_val)/np.sqrt(X_val**2 + Y_val**2 + Z_val**2)
    else:
        X_val = np.array([])
        Y_val = np.array([])
        Z_val = np.array([])
        M200c_val = np.array([])
        VX_val = np.array([])
        VY_val = np.array([])
        VZ_val = np.array([])
        Vlos = np.array([])
    df.close()
    return (X_val, Y_val, Z_val, Vlos, M200c_val)

def concatenate_data(results):
    # Pre-compute lengths for each part
    lengths = np.array([len(result[0]) for result in results])
    total_length = lengths.sum()

    # Pre-allocate arrays
    X_all = np.empty(total_length)
    Y_all = np.empty(total_length)
    Z_all = np.empty(total_length)
    Vlos_all = np.empty(total_length)
    M200_all = np.empty(total_length)

    # Calculate start and end indices
    end_ind_all = np.cumsum(lengths)
    start_ind_all = np.roll(end_ind_all, 1)
    start_ind_all[0] = 0


    # Use array slicing for assignment
    for i, (start, end, result) in enumerate(zip(start_ind_all, end_ind_all, results)):
        X_all[start:end] = result[0]
        Y_all[start:end] = result[1]
        Z_all[start:end] = result[2]
        Vlos_all[start:end] = result[3]
        M200_all[start:end] = result[4]

    return X_all, Y_all, Z_all, Vlos_all, M200_all

def process_halo(args):
    """Finds pixels near a single halo."""
    jhalo, ra_all_np, dec_all_np, halo_cat_R200c_np, halo_cat_DV_np, max_paint_R200c_factor, nside_local, pixel_dtype = args
    
    vec = hp.ang2vec(ra_all_np[jhalo], dec_all_np[jhalo], lonlat=True)
    # Angle subtended by the painting radius
    nearby_angle = max_paint_R200c_factor * halo_cat_R200c_np[jhalo] / halo_cat_DV_np[jhalo]
    nearby_pix = hp.query_disc(nside_local, vec, nearby_angle)
    
    if len(nearby_pix) == 0:
        return None
    
    nearby_pix = np.asarray(nearby_pix, dtype=pixel_dtype)
    nearby_ra, nearby_dec = hp.pix2ang(nside_local, nearby_pix, lonlat=True)
    
    # Haversine formula to calculate angular separation
    ra1, dec1 = np.radians(ra_all_np[jhalo]), np.radians(dec_all_np[jhalo])
    ra2, dec2 = np.radians(nearby_ra), np.radians(nearby_dec)
    
    a = np.sin((dec1 - dec2)/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra1 - ra2)/2)**2
    theta = 2 * np.arcsin(np.sqrt(a))
    
    # Projected distance in Mpc
    distances = (halo_cat_DV_np[jhalo] * theta).astype(np.float32)
    
    return (nearby_pix, distances, jhalo, len(nearby_pix))

def concatenate_batch_results(results, M_all_chunk, z_all_chunk, vlos_all_chunk, halo_cat_DV_np, halo_cat_R200c_np, max_paint_R200c_factor, pixel_dtype):
    """Concatenates results from a single batch of parallel processing."""
    if not results:
        return None
        
    lengths = np.array([res[3] for res in results], dtype=np.int32)
    total_length = lengths.sum()
    
    # Pre-allocate arrays for efficiency
    nearby_pix_all = np.empty(total_length, dtype=pixel_dtype)
    distances_pix_all = np.empty(total_length, dtype=np.float32)
    halo_indices = np.empty(total_length, dtype=np.int32)
    
    end_indices = np.cumsum(lengths)
    start_indices = np.concatenate([[0], end_indices[:-1]])
    
    for i, (start, end, res) in enumerate(zip(start_indices, end_indices, results)):
        nearby_pix_all[start:end] = res[0]
        distances_pix_all[start:end] = res[1]
        halo_indices[start:end] = res[2]
    
    original_halo_indices = np.array([res[2] for res in results], dtype=np.int32)
    
    # Map pixel-level properties back from halo-level properties
    logM_ind_all = np.log(M_all_chunk[halo_indices]).astype(np.float32)
    z_ind_all = z_all_chunk[halo_indices].astype(np.float32)
    vlos_ind_all = vlos_all_chunk[halo_indices].astype(np.float32)
    
    # Map halo-level properties
    ang_distance_all = halo_cat_DV_np[original_halo_indices].astype(np.float32)
    rp_max_all = (max_paint_R200c_factor * halo_cat_R200c_np[original_halo_indices]).astype(np.float32)
    
    return (nearby_pix_all, distances_pix_all, start_indices, end_indices, 
            logM_ind_all, z_ind_all, vlos_ind_all, ang_distance_all, rp_max_all)

def final_concatenate_batches(all_results, pixel_dtype):
    """Final concatenation of results from all batches."""
    total_pix = sum(len(res[0]) for res in all_results)
    total_halos = sum(len(res[2]) for res in all_results)
    
    # Pre-allocate final arrays
    final_nearby_pix = np.empty(total_pix, dtype=pixel_dtype)
    final_distances = np.empty(total_pix, dtype=np.float32)
    final_logM = np.empty(total_pix, dtype=np.float32)
    final_z = np.empty(total_pix, dtype=np.float32)
    final_vlos = np.empty(total_pix, dtype=np.float32)
    final_start_ind = np.empty(total_halos, dtype=np.int32)
    final_end_ind = np.empty(total_halos, dtype=np.int32)
    final_ang_dist = np.empty(total_halos, dtype=np.float32)
    final_rp_max = np.empty(total_halos, dtype=np.float32)
    
    pix_offset, halo_offset = 0, 0
    for result in all_results:
        n_pix_batch = len(result[0])
        n_halo_batch = len(result[2])
        
        # Unpack batch results
        nearby_pix_b, dist_b, start_b, end_b, logM_b, z_b, vlos_b, ang_dist_b, rp_max_b = result
        
        # Fill pixel-level arrays
        final_nearby_pix[pix_offset : pix_offset + n_pix_batch] = nearby_pix_b
        final_distances[pix_offset : pix_offset + n_pix_batch] = dist_b
        final_logM[pix_offset : pix_offset + n_pix_batch] = logM_b
        final_z[pix_offset : pix_offset + n_pix_batch] = z_b
        final_vlos[pix_offset : pix_offset + n_pix_batch] = vlos_b
        
        # Fill halo-level arrays, adjusting indices with the offset
        final_start_ind[halo_offset : halo_offset + n_halo_batch] = start_b + pix_offset
        final_end_ind[halo_offset : halo_offset + n_halo_batch] = end_b + pix_offset
        final_ang_dist[halo_offset : halo_offset + n_halo_batch] = ang_dist_b
        final_rp_max[halo_offset : halo_offset + n_halo_batch] = rp_max_b
        
        pix_offset += n_pix_batch
        halo_offset += n_halo_batch
        
    return (final_nearby_pix, final_distances, final_start_ind, final_end_ind,
            final_logM, final_z, final_vlos, final_ang_dist, final_rp_max)

def process_halos_in_batches(M_all_chunk, ra_all_chunk, dec_all_chunk, z_all_chunk, vlos_all_chunk, 
                             halo_cat_R200c, halo_cat_DA, max_paint_R200c_factor, nside, batch_size=1000):
    """Manages the parallel processing of halos in manageable batches."""
    pixel_dtype = np.int32 if nside <= 8192 else np.int64
    
    ra_all_np = np.clip(np.array(ra_all_chunk, dtype=np.float32), 0.01, 359.99)
    dec_all_np = np.clip(np.array(dec_all_chunk, dtype=np.float32), -89.99, 89.99)
    halo_cat_R200c_np = np.array(halo_cat_R200c, dtype=np.float32)
    halo_cat_DV_np = np.array(halo_cat_DA, dtype=np.float32)
    
    n_halos = len(z_all_chunk)
    all_results = []
    
    with Pool(cpu_count()) as pool:
        for batch_start in range(0, n_halos, batch_size):
            batch_end = min(batch_start + batch_size, n_halos)
            print(f"Processing halo batch {batch_start//batch_size + 1}/{(n_halos - 1)//batch_size + 1}...")
            
            # Create arguments for each halo in the batch
            batch_args = [(jhalo, ra_all_np, dec_all_np, halo_cat_R200c_np, halo_cat_DV_np, 
                           max_paint_R200c_factor, nside, pixel_dtype) for jhalo in range(batch_start, batch_end)]
            
            # Map processing to the pool
            batch_results = pool.map(process_halo, batch_args)
            batch_results = [r for r in batch_results if r is not None]
            
            if batch_results:
                # Concatenate results from the current batch
                batch_data = concatenate_batch_results(batch_results, M_all_chunk, z_all_chunk, vlos_all_chunk, 
                                                       halo_cat_DV_np, halo_cat_R200c_np, max_paint_R200c_factor, pixel_dtype)
                if batch_data is not None:
                    all_results.append(batch_data)
            
            del batch_results, batch_args
            gc.collect()
    
    if not all_results:
        return None
        
    return final_concatenate_batches(all_results, pixel_dtype)


# --- Timing Flag ---
# PROFILE_TIMING = True
PROFILE_TIMING = False

if PROFILE_TIMING:
    script_start_time = time.perf_counter()
    
# --- Analysis Parameters ---
nside = args.nside
jdevice = args.jdevice
Ndevices = args.ndevices

print(f"nside: {nside}")
print(f"jdevice: {jdevice}")
print(f"Ndevices: {Ndevices}")
print(f"Sample ID: {args.sample_id}")
print(f"Params: {args.theta_ej_0}, {args.nu_theta_ej_M}, {args.nu_theta_ej_z}, {args.mu_beta}")

# --- Load Configuration and Initialize Models ---
if PROFILE_TIMING:
    setup_start_time = time.perf_counter()

# =============================================================================
# INITIALIZATION & SETUP
# =============================================================================
yaml_file_path = f'{str(abs_path_params)}/params_default.yaml'
data = read_yaml(yaml_file_path)
sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(data)

if args.is_reference:
    print("Running REFERENCE simulation with default YAML parameters...")
    save_folder = "reference_run"
else:
    sim_params_dict.update({
        'theta_ej_0': args.theta_ej_0,
        'nu_theta_ej_M': args.nu_theta_ej_M,
        'nu_theta_ej_z': args.nu_theta_ej_z,
        'mu_beta': args.mu_beta
    })
    save_folder = f"sample_{args.sample_id}"

# Override halo parameters for this analysis
halo_params_dict.update({
    'rmin': 0.00075, 'rmax': 10.0, 'nr': 48,
    'zmin': 0.005, 'zmax': 2.1, 'nz':31,
    'lg10_Mmin': 10.75, 'lg10_Mmax': 16.0, 'nM': 40
})

# cosmology of the backlight simulations
cosmo_params_dict = {'w0': -1.0, 'flat': True, 'H0': 67.11, 'Om0': 0.3175,
                        'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624}

# Get the radial profiles
Prof_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)

# Setup simulated map generation.
mock_params_dict_setup = {}
mock_params_dict_setup['nside'] = nside
mock_params_dict_setup['get_ymap'] = True
mock_params_dict_setup['get_kSZmap'] = True
mock_params_dict_setup['get_taumap'] = True
mock_params_dict_setup['get_kappamap'] = True
mock_params_dict_setup['get_baryonifiedmap'] = True
mock_params_dict_setup['get_galmap'] = True
mock_params_dict_setup['smooth_profiles'] = True
Prof_test = setup_sim_map(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, mock_params_dict_setup, Profiles_obj=Prof_test)

chi2z_interp = interpolate.interp1d(np.log(Prof_test.chi_array), Prof_test.z_array, fill_value="extrapolate")

rho_m = Prof_test.get_rho_m(0.0)
Mtot = rho_m * (1000)**3
part_mass = float(Mtot/2048**3)

if PROFILE_TIMING:
    setup_end_time = time.perf_counter()
    print(f"\n[PROFILE] Initial setup time: {setup_end_time - setup_start_time:.2f} seconds")

# --- Identify Redshift Slices to Process ---
sim_file_path = '/work/hdd/bdne/spandey3/backlight/fiducial/100'
zlist = np.loadtxt(f'{sim_file_path}/zlist.txt')
snap_num_all = zlist[:,0].astype(int)
zval_all = zlist[:,1]

# Sort snapshots to isolate differential shells
sorted_indices = np.argsort(zval_all)
sorted_snaps = snap_num_all[sorted_indices]
sorted_zvals = zval_all[sorted_indices]

# --- Loop Over Each Redshift Slice ---
zmax_maps = 0.5
zmin_maps = 0.3
c_light = 299792.458
chi_CMB = bkgrd.radial_comoving_distance(Prof_test.cosmo_jax, 1.0 / (1.0 + 1089.0)).item()

snaps_in_shell = snap_num_all[(zval_all < zmax_maps) & (zval_all > zmin_maps)]

for snap_num in snap_num_all[1:-1]:
    if PROFILE_TIMING:
        z_slice_start_time = time.perf_counter()

    ind_snapnum = np.where(snap_num_all == snap_num)[0]
    zval = zval_all[ind_snapnum][0]
    if zval < zmax_maps and zval > zmin_maps:
        sdir = f'/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/{save_folder}'
        os.makedirs(sdir, exist_ok=True)
        save_map_fname = f'{sdir}/allmaps_nside{args.nside}_z{zval:.3f}_split{args.jdevice}.pkl'

        if not os.path.exists(save_map_fname):
            print(f"\nProcessing redshift slice: z = {zval:.3f}, snap_num = {snap_num}")
            
            # --- PHYSICS CONSTANTS FOR THIS SHELL ---
            chi_v = bkgrd.radial_comoving_distance(Prof_test.cosmo_jax, 1.0 / (1.0 + zval)).item()
            dz_snapshot = (zmax_maps - zmin_maps) / len(snaps_in_shell)
            H_z = cosmo_params_dict['H0'] * Prof_test.get_Ez(zval)
            dchi_snapshot = (c_light / H_z) * dz_snapshot
            
            vol_shell_pix = (4.0/3.0) * np.pi * ((chi_v + dchi_snapshot/2)**3 - (chi_v - dchi_snapshot/2)**3) / (12 * nside**2)
            mean_mass_per_pix = rho_m * vol_shell_pix

            # --- DIFFERENTIAL SHELL CALCULATION ---
            map_tot_orig = hp.read_map(f'{sim_file_path}/compressed_massMaps/massSheet_tot_z_{int(snap_num)}.fits.gz', verbose=False)

            curr_idx = np.where(sorted_snaps == snap_num)[0][0]
            if curr_idx > 0:
                prev_snap = sorted_snaps[curr_idx - 1]
                map_prev = hp.read_map(f'{sim_file_path}/compressed_massMaps/massSheet_tot_z_{int(prev_snap)}.fits.gz', verbose=False)
                map_shell = map_tot_orig - map_prev
            else:
                map_shell = map_tot_orig

            map_tot_orig_ds = hp.ud_grade(np.array(map_shell * part_mass), nside, power=-2)
            delta_sheet = (map_tot_orig_ds / mean_mass_per_pix) - 1.0

            # --- LOAD HALO DATA ---
            if PROFILE_TIMING:
                load_start_time = time.perf_counter()

            ldir = f'{sim_file_path}/halos/{snap_num}/'
            files_all = os.listdir(ldir)
            with Pool(cpu_count()) as pool:
                results = pool.map(open_data, files_all)
            X_all, Y_all, Z_all, vlos_all, M200c_all = concatenate_data(results)

            if PROFILE_TIMING:
                load_end_time = time.perf_counter()
                print(f"[PROFILE] Data loading for z={zval:.3f}: {load_end_time - load_start_time:.2f} seconds")

            # RA, DEC, and v_los selections
            if PROFILE_TIMING:
                selection_start_time = time.perf_counter()

            ra_all, dec_all = hp.vec2ang(np.array([X_all, Y_all, Z_all]).T, lonlat=True)
            z_all = zval * np.ones_like(ra_all)
            ra_all, dec_all = np.clip(ra_all, 0, 360), np.clip(dec_all, -90, 90)

            indsel = np.where((M200c_all < 6e15) & (M200c_all > (10**12.0)))[0]
            argsort = np.flip(np.argsort(M200c_all[indsel]))
            Nsel = (len(argsort) // Ndevices) * Ndevices
            argsort = argsort[:Nsel]
            argsort_split = split_array_uniform(argsort, Ndevices)
            argsort_here = argsort_split[jdevice]
            final_indices = indsel[argsort_here]

            ra_all, dec_all, z_all, M200c_all, vlos_all = (
                ra_all[final_indices], dec_all[final_indices], z_all[final_indices],
                M200c_all[final_indices], vlos_all[final_indices]
            )

            if PROFILE_TIMING:
                selection_end_time = time.perf_counter()
                print(f"[PROFILE] Halo selection & processing for z={zval:.3f}: {selection_end_time - selection_start_time:.2f} seconds")

            print(f"Number of halos for this device: {len(M200c_all)}")
            print(f"Mean log(M200c): {np.mean(np.log10(M200c_all)):.2f}, Mean z: {np.mean(z_all):.2f}")

            # --- CHUNKING AND MAP GENERATION ---
            nh_max_map = {8192: 4e3, 4096: 5e4, 2048: 5e5, 1024: 1e7}
            nh_max = nh_max_map.get(nside, 1e5)
            nsel = len(M200c_all)
            num_chunks = int(np.ceil(nsel / nh_max)) if nsel > nh_max else 1

            # Initialize empty maps for this redshift slice
            map_rhom_dmb = np.zeros(12 * nside**2, dtype=np.float32)
            map_rhom_dmo = np.zeros(12 * nside**2, dtype=np.float32)
            map_kappa = np.zeros(12 * nside**2, dtype=np.float32)
            map_ymap = np.zeros(12 * nside**2, dtype=np.float32)
            map_ksz = np.zeros(12 * nside**2, dtype=np.float32)
            map_tau = np.zeros(12 * nside**2, dtype=np.float32)
            mock_gals_all = {}

            for i in tqdm(range(num_chunks), desc=f"Painting maps for z={zval:.3f}"):
                start, end = int(i * nh_max), int((i + 1) * nh_max)
                M_chunk, ra_chunk, dec_chunk, z_chunk, vlos_chunk = (
                    M200c_all[start:end], ra_all[start:end], dec_all[start:end],
                    z_all[start:end], vlos_all[start:end]
                )

                scale_fac = 1. / (1. + z_chunk)
                rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(Prof_test.cosmo_jax, scale_fac) * 1e9
                rho_treshold = 200 * rho_c_z
                R200c = (M_chunk * 3.0 / (4.0 * jnp.pi * rho_treshold))**(1.0 / 3.0)
                DA = bkgrd.angular_diameter_distance(Prof_test.cosmo_jax, scale_fac)

                if PROFILE_TIMING:
                    pixel_finding_start_time = time.perf_counter()

                max_paint_R200c_factor = 3.0
                batch_size = int(nh_max // 2) if nh_max > 2 else 1
                result = process_halos_in_batches(
                    M_chunk, ra_chunk, dec_chunk, z_chunk, vlos_chunk,
                    R200c, DA, max_paint_R200c_factor, nside, batch_size
                )

                if PROFILE_TIMING:
                    pixel_finding_end_time = time.perf_counter()
                    print(f"\n[PROFILE] Chunk {i+1}/{num_chunks} - Pixel finding: {pixel_finding_end_time - pixel_finding_start_time:.2f} seconds")

                if result:
                    mock_params_dict = {
                        **mock_params_dict_setup,
                        'halo_z': jnp.array(z_chunk, dtype=jnp.float32),
                        'halo_ra': jnp.array(ra_chunk, dtype=jnp.float32),
                        'halo_dec': jnp.array(dec_chunk, dtype=jnp.float32),
                        'halo_M': jnp.array(M_chunk, dtype=jnp.float32),
                        'halo_vlos': jnp.array(vlos_chunk, dtype=jnp.float32),
                        'nearby_pix_all': jnp.array(result[0]),
                        'start_ind': jnp.array(result[2], dtype=jnp.int32),
                        'end_ind': jnp.array(result[3], dtype=jnp.int32),
                        'pix_prop_all': (jnp.array([np.log(result[1]), result[5], result[4], result[6]]).T).astype(jnp.float32),
                        'ang_distance_all': jnp.array(result[7]),
                        'rp_max_all': jnp.array(result[8]),
                        'profile_timing': PROFILE_TIMING
                    }

                    if PROFILE_TIMING:
                        map_gen_start_time = time.perf_counter()

                    mock_map = get_sim_map(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, mock_params_dict, Profiles_obj=Prof_test)

                    # Accumulate baryonic profiles
                    map_rhom_dmb += np.nan_to_num(mock_map.rhommap_final)
                    map_rhom_dmo += np.nan_to_num(mock_map.rhom_dmo_map_final)
                    
                    # PHYSICS CORRECTION: Y, Tau, kSZ are projected - sum them directly.
                    map_ymap += np.nan_to_num(mock_map.ymap_final)
                    map_ksz += np.nan_to_num(mock_map.kszmap_final)
                    map_tau += np.nan_to_num(mock_map.taumap_final)
                    
                    mock_gals_all[i] = mock_map.final_galaxy_catalog
                    
                    # --- KAPPA CONSTRUCTION (Volumetric density requires dchi) ---
                    # Calculate halo overdensity: (rho_halo - rho_dmo) / rho_mean
                    map_diff = (np.nan_to_num(mock_map.rhommap_final) - np.nan_to_num(mock_map.rhom_dmo_map_final)) / mean_mass_per_pix

                    if i == 0:
                        # Convert JAX array to NumPy for Healpy compatibility
                        delta_sheet_np = np.array(delta_sheet)
                        map_shell_smooth = hp.smoothing(delta_sheet_np, sigma=mock_map.sigma_val, verbose=False)
                        map_kappa += map_shell_smooth + map_diff
                    else:
                        map_kappa += map_diff

                    jax.clear_caches(); gc.collect()

            # Apply lensing weight kernel to dimensionless overdensity map
            weight_k = (1.5 * (cosmo_params_dict['H0']/c_light)**2 * cosmo_params_dict['Om0']) * ((chi_CMB - chi_v)/chi_CMB) * (1.0 + zval) * chi_v * dchi_snapshot
            map_kappa *= weight_k

            # --- SAVE DATA FOR SLICE ---
            saved_data = {
                'mock_gals_all': mock_gals_all,
                'map_rhom_dmb': map_rhom_dmb,
                'map_ymap': map_ymap,
                'map_ksz': map_ksz,
                'map_tau': map_tau,
                'map_kappa': map_kappa,
                'map_rhom_dmo': map_rhom_dmo,
                'map_gy': map_ymap,
                'map_gtau': map_tau,
                'map_gkappa': map_kappa
            }

            with open(save_map_fname, 'wb') as f:
                pk.dump(saved_data, f)
            
            if PROFILE_TIMING:
                save_end_time = time.perf_counter()
                print(f"[PROFILE] Saving results for z={zval:.3f}: {save_end_time - z_slice_start_time:.2f} seconds")

            if PROFILE_TIMING:
                z_slice_end_time = time.perf_counter()
                print(f"[PROFILE] TOTAL TIME for z-slice {zval:.3f}: {z_slice_end_time - z_slice_start_time:.2f} seconds")

            del saved_data, mock_gals_all
            gc.collect()
        else:
            print(f"File {save_map_fname} already exists. Skipping redshift slice z={zval:.3f}.")

if PROFILE_TIMING:
    script_end_time = time.perf_counter()
    print(f"\n[PROFILE] TOTAL SCRIPT EXECUTION TIME: {script_end_time - script_start_time:.2f} seconds")
