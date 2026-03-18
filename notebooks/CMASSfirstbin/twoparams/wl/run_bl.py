# =============================================================================
# 1. IMPORTS AND SETUP
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
parser.add_argument("--theta_ej_0", type=float)
parser.add_argument("--nu_theta_ej_M", type=float)
parser.add_argument("--nu_theta_ej_z", type=float)
parser.add_argument("--mu_beta", type=float)
parser.add_argument("--sample_id", type=int)
args = parser.parse_args()

plt.rcParams['text.usetex'] = True

curr_path = pathlib.Path().absolute()
project_base = curr_path.parents[3]
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
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def generate_dicts(data):
    sim_params_dict = data.get('sim_params', {})
    halo_params_dict = data.get('halo_params', {})
    analysis_dict = data.get('analysis', {})
    other_params_dict = data.get('other_params', {})
    return sim_params_dict, halo_params_dict, analysis_dict, other_params_dict

def open_data(file, Mlim=1e12):
    # Relies on global ldir being set in the loop
    df = h5.File(ldir+file, 'r')
    M200c = df['M200c'][()]
    X, Y, Z = df['X'][()], df['Y'][()], df['Z'][()]
    VX, VY, VZ = df['VX'][()], df['VY'][()], df['VZ'][()]

    # Check for non-empty dataset
    if (M200c.shape) is not None:
        indsel = np.where(M200c > Mlim)[0]
        X_val = X[indsel]
        Y_val = Y[indsel]
        Z_val = Z[indsel]
        M200c_val = M200c[indsel]
        VX_val = VX[indsel]
        VY_val = VY[indsel]
        VZ_val = VZ[indsel]
        Vlos = (VX_val*X_val + VY_val*Y_val + VZ_val*Z_val)/np.sqrt(X_val**2 + Y_val**2 + Z_val**2)
    else:
        X_val = np.array([]); Y_val = np.array([]); Z_val = np.array([]); M200c_val = np.array([])
        VX_val = np.array([]); VY_val = np.array([]); VZ_val = np.array([]); Vlos = np.array([])
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

def process_halo(args_in):
    jhalo, ra_all_np, dec_all_np, halo_cat_R200c_np, halo_cat_DV_np, max_paint_R200c_factor, nside_local, pixel_dtype = args_in
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
    nearby_pix_all = np.empty(total_length, dtype=pixel_dtype)
    distances_pix_all = np.empty(total_length, dtype=np.float32)
    halo_indices = np.empty(total_length, dtype=np.int32)
    end_indices = np.cumsum(lengths); start_indices = np.concatenate([[0], end_indices[:-1]])
    for i, (start, end, res) in enumerate(zip(start_indices, end_indices, results)):
        nearby_pix_all[start:end], distances_pix_all[start:end], halo_indices[start:end] = res[0], res[1], res[2]
    original_halo_indices = np.array([res[2] for res in results], dtype=np.int32)
    logM_ind_all = np.log(M_all_chunk[halo_indices]).astype(np.float32)
    z_ind_all = z_all_chunk[halo_indices].astype(np.float32)
    vlos_ind_all = vlos_all_chunk[halo_indices].astype(np.float32)
    ang_distance_all = halo_cat_DV_np[original_halo_indices].astype(np.float32)
    rp_max_all = (max_paint_R200c_factor * halo_cat_R200c_np[original_halo_indices]).astype(np.float32)
    return (nearby_pix_all, distances_pix_all, start_indices, end_indices, logM_ind_all, z_ind_all, vlos_ind_all, ang_distance_all, rp_max_all)

def final_concatenate_batches(all_results, pixel_dtype):
    total_pix = sum(len(res[0]) for res in all_results); total_halos = sum(len(res[2]) for res in all_results)
    final_nearby_pix, final_distances = np.empty(total_pix, dtype=pixel_dtype), np.empty(total_pix, dtype=np.float32)
    final_logM, final_z, final_vlos = np.empty(total_pix, dtype=np.float32), np.empty(total_pix, dtype=np.float32), np.empty(total_pix, dtype=np.float32)
    final_start_ind, final_end_ind = np.empty(total_halos, dtype=np.int32), np.empty(total_halos, dtype=np.int32)
    final_ang_dist, final_rp_max = np.empty(total_halos, dtype=np.float32), np.empty(total_halos, dtype=np.float32)
    pix_offset, halo_offset = 0, 0
    for result in all_results:
        n_pix_batch, n_halo_batch = len(result[0]), len(result[2])
        nearby_pix_b, dist_b, start_b, end_b, logM_b, z_b, vlos_b, ang_dist_b, rp_max_b = result
        final_nearby_pix[pix_offset : pix_offset + n_pix_batch] = nearby_pix_b
        final_distances[pix_offset : pix_offset + n_pix_batch] = dist_b
        final_logM[pix_offset : pix_offset + n_pix_batch] = logM_b
        final_z[pix_offset : pix_offset + n_pix_batch] = z_b
        final_vlos[pix_offset : pix_offset + n_pix_batch] = vlos_b
        final_start_ind[halo_offset : halo_offset + n_halo_batch] = start_b + pix_offset
        final_end_ind[halo_offset : halo_offset + n_halo_batch] = end_b + pix_offset
        final_ang_dist[halo_offset : halo_offset + n_halo_batch] = ang_dist_b
        final_rp_max[halo_offset : halo_offset + n_halo_batch] = rp_max_b
        pix_offset += n_pix_batch; halo_offset += n_halo_batch
    return (final_nearby_pix, final_distances, final_start_ind, final_end_ind, final_logM, final_z, final_vlos, final_ang_dist, final_rp_max)

def process_halos_in_batches(M_all_chunk, ra_all_chunk, dec_all_chunk, z_all_chunk, vlos_all_chunk, halo_cat_R200c, halo_cat_DA, max_paint_R200c_factor, nside, batch_size=1000):
    pixel_dtype = np.int32 if nside <= 8192 else np.int64
    ra_all_np, dec_all_np = np.clip(np.array(ra_all_chunk, dtype=np.float32), 0.01, 359.99), np.clip(np.array(dec_all_chunk, dtype=np.float32), -89.99, 89.99)
    halo_cat_R200c_np, halo_cat_DV_np = np.array(halo_cat_R200c, dtype=np.float32), np.array(halo_cat_DA, dtype=np.float32)
    n_halos = len(z_all_chunk); all_results = []
    with Pool(cpu_count()) as pool:
        for batch_start in range(0, n_halos, batch_size):
            batch_end = min(batch_start + batch_size, n_halos)
            batch_args = [(jhalo, ra_all_np, dec_all_np, halo_cat_R200c_np, halo_cat_DV_np, max_paint_R200c_factor, nside, pixel_dtype) for jhalo in range(batch_start, batch_end)]
            batch_results = pool.map(process_halo, batch_args); batch_results = [r for r in batch_results if r is not None]
            if batch_results:
                batch_data = concatenate_batch_results(batch_results, M_all_chunk, z_all_chunk, vlos_all_chunk, halo_cat_DV_np, halo_cat_R200c_np, max_paint_R200c_factor, pixel_dtype)
                if batch_data is not None: all_results.append(batch_data)
            del batch_results, batch_args; gc.collect()
    return final_concatenate_batches(all_results, pixel_dtype) if all_results else None

# =============================================================================
# 3. INITIALIZATION & SETUP
# =============================================================================
PROFILE_TIMING = True
if PROFILE_TIMING:
    script_start_time = time.perf_counter()

nside, jdevice, Ndevices = args.nside, args.jdevice, args.ndevices
yaml_file_path = f'{str(abs_path_params)}/params_anshuman.yaml'
data = read_yaml(yaml_file_path)
sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(data)

# --- HYBRID PARAMETER LOGIC ---
if args.is_reference:
    print("\n[INFO] Running REFERENCE simulation with default YAML parameters...")
    save_folder = "reference_run"
else:
    print(f"\n[INFO] Running VARIATION simulation for Sample ID: {args.sample_id}")
    # Update only parameters that are explicitly passed via CLI
    if args.theta_ej_0 is not None: sim_params_dict['theta_ej_0'] = args.theta_ej_0
    if args.mu_beta is not None: sim_params_dict['mu_beta'] = args.mu_beta
    if args.nu_theta_ej_M is not None: sim_params_dict['nu_theta_ej_M'] = args.nu_theta_ej_M
    if args.nu_theta_ej_z is not None: sim_params_dict['nu_theta_ej_z'] = args.nu_theta_ej_z
    save_folder = f"sample_{args.sample_id}"

# --- PARAMETER SANITY CHECK ---
print("="*30)
print("FINAL PHYSICS PARAMETERS:")
print(f"  theta_ej_0:    {sim_params_dict.get('theta_ej_0')}")
print(f"  mu_beta:       {sim_params_dict.get('mu_beta')}")
print(f"  nu_theta_ej_M: {sim_params_dict.get('nu_theta_ej_M')}")
print(f"  nu_theta_ej_z: {sim_params_dict.get('nu_theta_ej_z')}")
print("="*30)

halo_params_dict.update({
    'rmin': 0.001, 'rmax': 10.0, 'nr': 48,
    'zmin': 0.001, 'zmax': 2.1, 'nz': 31,
    'lg10_Mmin': 11.75, 'lg10_Mmax': 16.0, 'nM': 32
})
LOAD_MASS_CUT = 10**12.75

cosmo_params_dict = {'w0': -1.0, 'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624}

if PROFILE_TIMING: setup_start_time = time.perf_counter()

Prof_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
mock_params_dict_setup = {'nside': nside, 'get_ymap': True, 'get_kSZmap': True, 'get_taumap': True, 'get_kappamap': True, 'get_baryonifiedmap': True, 'get_galmap': True, 'smooth_profiles': True}
Prof_test = setup_sim_map(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, mock_params_dict_setup, Profiles_obj=Prof_test)

rho_m = Prof_test.get_rho_m(0.0); part_mass = float((rho_m * (1000)**3)/2048**3)
chi_CMB = bkgrd.radial_comoving_distance(Prof_test.cosmo_jax, 1.0 / (1.0 + 1089.0)).item()
c_light = 299792.458

if PROFILE_TIMING:
    setup_end_time = time.perf_counter()
    print(f"\n[PROFILE] Initial setup time: {setup_end_time - setup_start_time:.2f} seconds")

sim_file_path = '/work/hdd/bdne/spandey3/backlight/fiducial/100'
zlist = np.loadtxt(f'{sim_file_path}/zlist.txt')
snap_num_all, zval_all = zlist[:,0].astype(int), zlist[:,1]
sorted_indices = np.argsort(zval_all)
sorted_snaps, sorted_zvals = snap_num_all[sorted_indices], zval_all[sorted_indices]

# =============================================================================
# 4. REDSHIFT LOOP
# =============================================================================
zmax_maps, zmin_maps = 0.5, 0.3
snaps_in_shell = snap_num_all[(zval_all < zmax_maps) & (zval_all > zmin_maps)]

for snap_num in snaps_in_shell:
    if PROFILE_TIMING: z_slice_start_time = time.perf_counter()
    ind_snapnum = np.where(snap_num_all == snap_num)[0]
    zval = zval_all[ind_snapnum][0]

    sdir = f'/work/hdd/bdne/aacharya2/GODMAX/results/backlight_pkdgrav/CMASSfirstbin/twoparams/wl/{save_folder}'
    os.makedirs(sdir, exist_ok=True)
    save_map_fname = f'{sdir}/allmaps_nside{args.nside}_z{zval:.3f}_split{args.jdevice}.pkl'

    if not os.path.exists(save_map_fname):
        print(f"\nProcessing redshift slice: z = {zval:.3f}, snap_num = {snap_num}")
        chi_v = bkgrd.radial_comoving_distance(Prof_test.cosmo_jax, 1.0 / (1.0 + zval)).item()
        dz_snapshot = (zmax_maps - zmin_maps) / len(snaps_in_shell)
        dchi_snapshot = (c_light / (cosmo_params_dict['H0'] * Prof_test.get_Ez(zval))) * dz_snapshot
        mean_mass_per_pix = rho_m * (4.0/3.0) * np.pi * ((chi_v + dchi_snapshot/2)**3 - (chi_v - dchi_snapshot/2)**3) / (12 * nside**2)

        map_tot_orig = hp.read_map(f'{sim_file_path}/compressed_massMaps/massSheet_tot_z_{int(snap_num)}.fits.gz', verbose=False)
        curr_idx = np.where(sorted_snaps == snap_num)[0][0]
        map_shell = map_tot_orig - hp.read_map(f'{sim_file_path}/compressed_massMaps/massSheet_tot_z_{int(sorted_snaps[curr_idx-1])}.fits.gz', verbose=False) if curr_idx > 0 else map_tot_orig
        delta_sheet = (hp.ud_grade(np.array(map_shell * part_mass), nside, power=-2) / mean_mass_per_pix) - 1.0

        if PROFILE_TIMING: load_start_time = time.perf_counter()
        global ldir
        ldir = f'{sim_file_path}/halos/{snap_num}/'
        files_all = os.listdir(ldir)

        with Pool(cpu_count()) as pool:
            results = pool.map(partial(open_data, Mlim=LOAD_MASS_CUT), files_all)
        X_all, Y_all, Z_all, vlos_all, M200c_all = concatenate_data(results)

        if PROFILE_TIMING: print(f"[PROFILE] Data loading for z={zval:.3f}: {time.perf_counter() - load_start_time:.2f} seconds")

        ra_all, dec_all = hp.vec2ang(np.array([X_all, Y_all, Z_all]).T, lonlat=True)
        z_all = zval * np.ones_like(ra_all)
        ra_all, dec_all = np.clip(ra_all, 0, 360), np.clip(dec_all, -90, 90)

        indsel = np.where((M200c_all < 6e15) & (M200c_all > (10**12.0)))[0]
        argsort = np.flip(np.argsort(M200c_all[indsel]))
        Nsel = (len(argsort) // Ndevices) * Ndevices
        argsort = argsort[:Nsel]
        argsort_split = np.array_split(argsort, Ndevices)
        argsort_here = argsort_split[jdevice]
        final_indices = indsel[argsort_here]

        ra_all, dec_all, z_all, M200c_all, vlos_all = ra_all[final_indices], dec_all[final_indices], z_all[final_indices], M200c_all[final_indices], vlos_all[final_indices]

        nh_max = {8192: 4e3, 4096: 5e4, 2048: 5e5, 1024: 1e7, 512: 5e7}.get(nside, 1e5)
        num_chunks = int(np.ceil(len(M200c_all) / nh_max))
        map_rhom_dmb = np.zeros(12 * nside**2, dtype=np.float32); map_rhom_dmo = np.zeros(12 * nside**2, dtype=np.float32)
        map_kappa = np.zeros(12 * nside**2, dtype=np.float32); map_ymap = np.zeros(12 * nside**2, dtype=np.float32)
        map_ksz = np.zeros(12 * nside**2, dtype=np.float32); map_tau = np.zeros(12 * nside**2, dtype=np.float32)
        mock_gals_all = {}

        for i in tqdm(range(num_chunks), desc=f"Painting maps for z={zval:.3f}"):
            start, end = int(i * nh_max), int((i + 1) * nh_max)
            Mc, rac, decc, zc, vc = M200c_all[start:end], ra_all[start:end], dec_all[start:end], z_all[start:end], vlos_all[start:end]
            scale_fac = 1. / (1. + zc)
            rho_treshold = 200 * constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(Prof_test.cosmo_jax, scale_fac) * 1e9
            R200c, DA = (Mc * 3.0 / (4.0 * jnp.pi * rho_treshold))**(1.0/3.0), bkgrd.angular_diameter_distance(Prof_test.cosmo_jax, scale_fac)
            result = process_halos_in_batches(Mc, rac, decc, zc, vc, R200c, DA, 3.0, nside)
            if result:
                m_p = {**mock_params_dict_setup, 'halo_z': jnp.array(zc), 'halo_ra': jnp.array(rac), 'halo_dec': jnp.array(decc), 'halo_M': jnp.array(Mc), 'halo_vlos': jnp.array(vc), 'nearby_pix_all': jnp.array(result[0]), 'start_ind': jnp.array(result[2]), 'end_ind': jnp.array(result[3]), 'pix_prop_all': (jnp.array([np.log(result[1]), result[5], result[4], result[6]]).T).astype(jnp.float32), 'ang_distance_all': jnp.array(result[7]), 'rp_max_all': jnp.array(result[8]), 'profile_timing': False}
                mock_map = get_sim_map(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, m_p, Profiles_obj=Prof_test)
                map_ymap += np.nan_to_num(mock_map.ymap_final); map_tau += np.nan_to_num(mock_map.taumap_final); map_ksz += np.nan_to_num(mock_map.kszmap_final)
                map_rhom_dmb += np.nan_to_num(mock_map.rhommap_final); map_rhom_dmo += np.nan_to_num(mock_map.rhom_dmo_map_final); mock_gals_all[i] = mock_map.final_galaxy_catalog
                diff = (np.nan_to_num(mock_map.rhommap_final) - np.nan_to_num(mock_map.rhom_dmo_map_final)) / mean_mass_per_pix
                map_kappa += (hp.smoothing(np.array(delta_sheet), sigma=mock_map.sigma_val, verbose=False) + diff) if i == 0 else diff
                jax.clear_caches(); gc.collect()

        map_kappa *= (1.5 * (cosmo_params_dict['H0']/c_light)**2 * cosmo_params_dict['Om0']) * ((chi_CMB - chi_v)/chi_CMB) * (1.0 + zval) * chi_v * dchi_snapshot
        saved_data = {'mock_gals_all': mock_gals_all, 'map_rhom_dmb': map_rhom_dmb, 'map_ymap': map_ymap, 'map_ksz': map_ksz, 'map_tau': map_tau, 'map_kappa': map_kappa, 'map_rhom_dmo': map_rhom_dmo, 'map_gy': map_ymap, 'map_gtau': map_tau, 'map_gkappa': map_kappa}
        with open(save_map_fname, 'wb') as f: pk.dump(saved_data, f)
        if PROFILE_TIMING: print(f"[PROFILE] TOTAL TIME for z-slice {zval:.3f}: {time.perf_counter() - z_slice_start_time:.2f} seconds")
        del saved_data, mock_gals_all; gc.collect()
    else:
        print(f"File {save_map_fname} already exists. Skipping.")

if PROFILE_TIMING: print(f"\n[PROFILE] TOTAL SCRIPT EXECUTION TIME: {time.perf_counter() - script_start_time:.2f} seconds")
