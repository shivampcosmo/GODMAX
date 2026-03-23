import sys, os
# os.environ['PATH']='/projects/bdne/spandey3/tex/texlive/bin/x86_64-linux:'+ os.environ['PATH']
# os.environ['PYTHONPATH']='/projects/bdne/spandey3/tex/texlive/bin/x86_64-linux:'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.98'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax_cosmo.background as bkgrd
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from jax.lib import xla_bridge
platform = xla_bridge.get_backend().platform
import jax
print(jax.local_device_count(), jax.device_count())
jax.config.update('jax_platform_name', platform)
jax.config.update("jax_enable_x64", True)
import jax
# Change the current working directory to the desired path
# os.chdir('/mnt/home/spandey/ceph/GODMAX/src/')
import matplotlib
import matplotlib.pyplot as pl
# set latex to false:
pl.rcParams['text.usetex'] = False
import pathlib
curr_path = pathlib.Path().absolute()
abs_path_data = os.path.abspath(curr_path / "../../data/") 
abs_path_src = os.path.abspath(curr_path / "../../src/") 
abs_path_results = os.path.abspath(curr_path / "../../results/") 
sys.path.append((curr_path))
sys.path.append((abs_path_data))
sys.path.append((abs_path_results))
sys.path.append(abs_path_src)
import numpyro
numpyro.set_platform("gpu")
numpyro.enable_x64()
from jax import config
config.update("jax_enable_x64", True)
import scipy.interpolate as interp
import pickle as pk
import numpy as np
import jax.numpy as jnp
import colossus 
from jax import vmap, grad, pmap
import matplotlib.pyplot as pl
from get_sim_maps import setup_sim_map, get_sim_map
import helpers.constants as constants
from tqdm import tqdm
# from time import time
import time
pl.rc('text', usetex=True)
import gc
# Palatino
# pl.rc('font', family='DejaVu Sans')
import ast
import yaml

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
from halo_processing_utils import process_halo

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

def open_data(file, Mlim=1e8):
    df = h5.File(ldir+file, 'r')
    M200c = df['M200c'][()]
    X, Y, Z = df['X'][()], df['Y'][()], df['Z'][()]
    VX, VY, VZ = df['VX'][()], df['VY'][()], df['VZ'][()]
    if (M200c.shape) is not None:
        indsel = np.where(M200c>Mlim)[0]
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

def process_halos_in_batches(M_all_chunk, ra_all_chunk, dec_all_chunk, z_all_chunk, vlos_all_chunk, halo_cat_R200c, halo_cat_DA, max_paint_R200c_factor, nside, batch_size=1000):
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



nside = int(ast.literal_eval(sys.argv[1]))
print('nside: ', nside)
jdevice = int(ast.literal_eval(sys.argv[2]))
print('jdevice: ', jdevice)
Ndevices = int(ast.literal_eval(sys.argv[3]))
print('Ndevices: ', Ndevices)

PROFILE_TIMING = True

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def generate_dicts(data):
    sim_params_dict = data.get('sim_params', {})
    halo_params_dict = data.get('halo_params', {})
    analysis_dict = data.get('analysis', {})
    other_params_dict = data.get('other_params', {})
    return sim_params_dict, halo_params_dict, analysis_dict, other_params_dict

yaml_file_path = '/mnt/ceph/users/spandey/ltu-godmax/GODMAX/param_files/params_default.yaml'
data = read_yaml(yaml_file_path)
sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(data)

halo_params_dict['rmin'] = 0.0001
halo_params_dict['rmax'] = 8.0
halo_params_dict['nr'] = 86
halo_params_dict['zmax'] = 2.75
halo_params_dict['nz'] = 87
halo_params_dict['lg10_Mmin'] = 13.0
halo_params_dict['lg10_Mmax'] = 16.0
halo_params_dict['nM'] = 78

from get_B12_profile import Battaglia_12_16
import helpers.constants as constants
cosmo_params_dict = {'w0':-1.0 ,'flat': True, 'H0': 69.0, 'Om0': 0.31, 'Ob0': 0.049, 'sigma8':0.81 ,'ns': 0.965}


# density_params_test = {
#     'rho0': {
#         'A0': 5e3,
#         'alpha_m': 0.4,
#         'alpha_z': -0.8
#         },
#     'alpha': {
#         'A0': 0.88,
#         'alpha_m': -0.03,
#         'alpha_z': 0.19
#         },
#     'beta': {
#         'A0': 4.83,
#         'alpha_m': 0.25,
#         'alpha_z': -0.5
#         }
#     }

# B12_test = Battaglia_12_16(sim_params_dict={'cosmo':cosmo_params_dict, 'init_power':False}, halo_params_dict=halo_params_dict, density_params_def=density_params_test)
profiles_test = Battaglia_12_16(sim_params_dict={'cosmo':cosmo_params_dict, 'init_power':True}, halo_params_dict=halo_params_dict)

mock_params_dict_setup = {}
mock_params_dict_setup['nside'] = nside
mock_params_dict_setup['get_ymap'] = True
mock_params_dict_setup['get_kSZmap'] = True
mock_params_dict_setup['get_taumap'] = True
mock_params_dict_setup['get_kappamap'] = False
mock_params_dict_setup['get_galmap'] = False
mock_params_dict_setup['smooth_profiles'] = True
B12_test = setup_sim_map(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, mock_params_dict_setup, Profiles_obj=profiles_test)


from astropy.io import fits
# df = fits.open('/mnt/home/spandey/ceph/websky/halo_input_wM200c_zmax2p5_minM200m_noh_5e13.fits')
df = fits.open('/mnt/ceph/users/spandey/websky/halo_input_wM200c_zmax2p5_minM200m_4e13_new.fits')
# df = fits.open('/mnt/home/spandey/ceph/websky/halo_input_wM200c_zmax2p5_minM200c_9e13_cfix7.fits')
# df[1].header
ra_all = df[1].data['ra']
dec_all = df[1].data['dec']
z_all = df[1].data['z']
M200c_all = df[1].data['M200c_wh']
M200m_all = df[1].data['M200m_wh']
vlos_all = df[1].data['vrad']
indsel = np.where((z_all>0.195) & (M200m_all<1e15))[0]
ra_all = ra_all[indsel]
dec_all = dec_all[indsel]
z_all = z_all[indsel]
M200c_all = M200c_all[indsel]
M200m_all = M200m_all[indsel]
vlos_all = vlos_all[indsel]
print('total number of halos: ', len(M200c_all))

argsort = np.flip(np.argsort(M200c_all))
Nsel = (len(argsort)//Ndevices)*Ndevices
argsort = argsort[:Nsel]
Ngal_per_device = Nsel//Ndevices
argsort_here = argsort[jdevice*Ngal_per_device:(jdevice+1)*Ngal_per_device]
ra_all = ra_all[argsort_here]
dec_all = dec_all[argsort_here]
z_all = z_all[argsort_here]
M200c_all = M200c_all[argsort_here]
M200m_all = M200m_all[argsort_here]
vlos_all = vlos_all[argsort_here]
print('number of halos: ', len(M200c_all), ', mean log(M200c)', np.mean(np.log10(M200c_all)), ', mean z', np.mean(z_all))

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")



import pickle as pk
import jax
import scipy.interpolate as interp
import healpy as hp
import numpy as np
from multiprocessing import Pool, cpu_count
from astropy.io import fits
# import constants
import jax_cosmo.background as bkgrd
# from get_sim_maps import get_sim_map
import h5py as h5


jax.clear_caches()
# nside = 4096
# nside = 8192
sdir = '/mnt/ceph/users/spandey/ltu-godmax/GODMAX/results/mock_gen/maps_websky/'
# save_kszmap_fname = sdir + f'tSZ_sim_B12_testv3_nside_{nside}.pkl'
# save_map_fname = sdir + f'tau_sim_B16_testv11_nside_{nside}_split_{jdevice}_{Ndevices}.pkl'
save_map_fname = sdir + f'RUN2_v3_tau_sim_B16_nside_{nside}_split_{jdevice}_{Ndevices}.pkl'
# if not os.path.exists(save_kszmap_fname):
halo_ra, halo_dec = ra_all, dec_all
halo_z = z_all
halo_m = M200c_all

print('number of halos: ', len(halo_m))
M_all = halo_m
ra_all = halo_ra
dec_all = halo_dec
z_all = halo_z
# vlos_all = np.zeros_like(z_all)
nsel = len(M_all)

# # nh_max = 4e4
# # nh_max = 4e3
# if nside == 8192:
#     nh_max = 4e3
# elif nside == 4096:
#     nh_max = 5e4
# elif nside == 2048:
#     nh_max = 8e5
# elif nside == 1024:
#     nh_max = 1e7
# else:
#     print('nside not supported')
# # nh_max = 1e5
# # nh_max = 2e6
# if nsel > nh_max:
#     num_chunks = int(np.ceil(nsel / nh_max))
# else:
#     num_chunks = 1


# --- Chunking and Map Generation ---
nh_max_map = {8192: 4e3, 4096: 1e4, 2048: 8e5, 1024: 1e7, 512: 5e7}
nh_max = nh_max_map.get(nside, 1e5) # Default if nside not in map
nsel = len(M200c_all)
num_chunks = int(np.ceil(nsel / nh_max)) if nsel > nh_max else 1

map_rhom, map_ymap, map_ksz, map_tau = (np.zeros(12 * nside**2, dtype=np.float32) for _ in range(4))
mock_gals_all = {}

for i in tqdm(range(num_chunks)):
    start, end = int(i * nh_max), int((i + 1) * nh_max)
    # start, end = 0, len(ra_all)
    M_chunk, ra_chunk, dec_chunk, z_chunk, vlos_chunk = (
        M200c_all[start:end], ra_all[start:end], dec_all[start:end],
        z_all[start:end], vlos_all[start:end]
    )
    
    scale_fac = 1. / (1. + z_chunk)
    rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(B12_test.cosmo_jax, scale_fac) * 1e9
    rho_treshold = 200 * rho_c_z
    R200c = (M_chunk * 3.0 / (4.0 * jnp.pi * rho_treshold))**(1.0 / 3.0)
    DA = bkgrd.angular_diameter_distance(B12_test.cosmo_jax, scale_fac)
    
    if PROFILE_TIMING:
        pixel_finding_start_time = time.perf_counter()
    
    max_paint_R200c_factor = 3.0
    # batch_size = int(nh_max // 2) if nh_max > 2 else 1
    batch_size = len(ra_all)
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
            'halo_M': jnp.array(M_chunk, dtype=jnp.float64),
            'halo_vlos': jnp.array(vlos_chunk, dtype=jnp.float32),
            'nearby_pix_all': jnp.array(result[0]),
            'start_ind': jnp.array(result[2], dtype=jnp.int32),
            'end_ind': jnp.array(result[3], dtype=jnp.int32),
            'pix_prop_all': (jnp.array([np.log(result[1]), result[5], result[4], result[6]]).T).astype(jnp.float32),
            # 'pix_prop_all': (jnp.array([np.log(result[1] * (1 + result[5])), result[5], result[4], result[6]]).T).astype(jnp.float32),            
            'ang_distance_all': jnp.array(result[7]),
            'rp_max_all': jnp.array(result[8]),
            'profile_timing': PROFILE_TIMING
        }
        
        if PROFILE_TIMING:
            map_gen_start_time = time.perf_counter()
        
        mock_map = get_sim_map(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, mock_params_dict, Profiles_obj=B12_test)
        
        # map_rhom += np.nan_to_num(mock_map.rhommap_final)
        map_ymap += np.nan_to_num(mock_map.ymap_final)
        map_ksz += np.nan_to_num(mock_map.kszmap_final)
        map_tau += np.nan_to_num(mock_map.taumap_final)
        # mock_gals_all[i] = mock_map.final_galaxy_catalog
        # print(mock_gals_all[i].shape)

        if PROFILE_TIMING:
            map_gen_end_time = time.perf_counter()
            print(f"[PROFILE] Chunk {i+1}/{num_chunks} - JAX map generation: {map_gen_end_time - map_gen_start_time:.2f} seconds")
        
        del mock_map, result, mock_params_dict
        jax.clear_caches()
        gc.collect()

# Save results for the current redshift slice
if PROFILE_TIMING:
    save_start_time = time.perf_counter()
    

saved_data = {'map_ymap': map_ymap, 'map_ksz': map_ksz, 'map_tau': map_tau}

with open(save_map_fname, 'wb') as f:
    pk.dump(saved_data, f)

del saved_data, mock_gals_all
gc.collect()


# else:
#     saved = pk.load(open(save_ymap_fname, 'rb'))
#     ymap_test = saved['ymap_test']

