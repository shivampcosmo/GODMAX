%load_ext autoreload
%autoreload 2

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
pl.rc('text', usetex=True)
import gc
# Palatino
# pl.rc('font', family='DejaVu Sans')
import ast
import yaml

nside = int(ast.literal_eval(sys.argv[1]))
print('nside: ', nside)
jdevice = int(ast.literal_eval(sys.argv[2]))
print('jdevice: ', jdevice)
Ndevices = int(ast.literal_eval(sys.argv[3]))
print('Ndevices: ', Ndevices)
# nside = 2048
# jdevice = 0
# Ndevices = 

def split_array_power_ratio(arr, num_parts, power_val=0.5):
    n_total = arr.shape[0]
    j_values = np.arange(1, num_parts + 1)
    sqrt_j = j_values**power_val
    sum_of_sqrts = np.sum(sqrt_j)
    ideal_s1 = n_total / sum_of_sqrts
    ideal_sizes = ideal_s1 * sqrt_j
    int_sizes = np.floor(ideal_sizes).astype(int)
    remainder = n_total - np.sum(int_sizes)
    fractional_parts = ideal_sizes - int_sizes
    indices_to_increment = np.argsort(fractional_parts)[-remainder:]
    int_sizes[indices_to_increment] += 1
    split_indices = np.cumsum(int_sizes)[:-1]
    return np.split(arr, split_indices)

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

yaml_file_path = '/mnt/home/spandey/ceph/GODMAX/param_files/params_default.yaml'
data = read_yaml(yaml_file_path)
sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(data)

halo_params_dict['rmin'] = 0.0001
halo_params_dict['rmax'] = 10.0
halo_params_dict['nr'] = 126
halo_params_dict['zmin'] = 0.01
halo_params_dict['zmax'] = 4.1
halo_params_dict['nz'] = 127
halo_params_dict['lg10_Mmin'] = 12.0
halo_params_dict['lg10_Mmax'] = 16.0
halo_params_dict['nM'] = 128

from get_B12_profile import Battaglia_12_16
import helpers.constants as constants
# cosmo_params_dict = {'w0':-1.0 ,'flat': True, 'H0': 69.0, 'Om0': 0.31, 'Ob0': 0.049, 'sigma8':0.81 ,'ns': 0.965}
cosmo_params_dict = {'w0':-1.0 ,'flat': True, 'H0': 67.74, 'Om0': 0.3089, 'Ob0': 0.0486, 'sigma8':0.8159 ,'ns': 0.9667}
# B12_test = Battaglia_12_16({'cosmo':cosmo_params_dict, 'init_power':False}, halo_params_dict)


B12_test = Battaglia_12_16(sim_params_dict={'cosmo':cosmo_params_dict, 'init_power':True}, halo_params_dict=halo_params_dict)

from astropy.io import fits
import h5py as h5
import healpy as hp

fname = '/mnt/ceph/users/abayer/fastpm/halfdome/stampede2_3750Mpch_6144cube/final_res/halos/lightcone_100.hdf5'
with h5.File(fname, 'r') as f:
    # print(f.keys())
    M200c_all = f['halo_mass_m200c'][:]
    z_all = f['redshift'][:]
    pos = f['Position'][:]
    v_all = f['Velocity'][:]

ra_all, dec_all = hp.vec2ang(pos, lonlat=True)

# get the line of sight velocity:
vlos_all = np.sum(v_all * hp.ang2vec(ra_all, dec_all, lonlat=True), axis=1)

zmax = 1.5
indsel = np.where((z_all>0.05) & (z_all<zmax) & (M200c_all<4e15) & (M200c_all>(10**13.0)))[0]
ra_all = ra_all[indsel]
dec_all = dec_all[indsel]
z_all = z_all[indsel]
M200c_all = M200c_all[indsel]
# M200m_all = M200m_all[indsel]
vlos_all = vlos_all[indsel]
print('total number of halos: ', len(M200c_all))

argsort = np.flip(np.argsort(M200c_all))
Nsel = (len(argsort)//Ndevices)*Ndevices
argsort = argsort[:Nsel]
# Ngal_per_device = Nsel//Ndevices
# argsort_here = argsort[jdevice*Ngal_per_device:(jdevice+1)*Ngal_per_device]
argsort_split = split_array_power_ratio(argsort, Ndevices)
argsort_here = argsort_split[jdevice]

ra_all = ra_all[argsort_here]
dec_all = dec_all[argsort_here]
z_all = z_all[argsort_here]
M200c_all = M200c_all[argsort_here]
vlos_all = vlos_all[argsort_here]

print('number of halos: ', len(M200c_all), ', mean log(M200c)', np.mean(np.log10(M200c_all)), ', mean z', np.mean(z_all))
print('log10Mmin: ', np.min(np.log10(M200c_all)), ', log10Mmax: ', np.max(np.log10(M200c_all)))
print('zmin: ', np.min(z_all), ', zmax: ', np.max(z_all))
print('ra min: ', np.min(ra_all), ', ra max: ', np.max(ra_all))
print('dec min: ', np.min(dec_all), ', dec max: ', np.max(dec_all))

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
import jax_cosmo.background as bkgrd
from get_sim_maps import get_sim_map
import h5py as h5

jax.clear_caches()
# nside = 4096
# nside = 8192
sdir = '/mnt/home/spandey/ceph/GODMAX/notebooks/all_arxiv/mock_gen/maps_halfdome/'
save_map_fname = sdir + f'tSZ_sim_B12_testv12_nside_{nside}_split_{jdevice}_{Ndevices}_zmax_{zmax}.pkl'
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

if nside == 8192:
    nh_max = 4e3
elif nside == 4096:
    nh_max = 5e4
elif nside == 2048:
    if jdevice == 0:
        nh_max = 5e5
    else:
        nh_max = 8e5
elif nside == 1024:
    nh_max = 2e5
else:
    print('nside not supported')
if nsel > nh_max:
    num_chunks = int(np.ceil(nsel / nh_max))
else:
    num_chunks = 1

map_test = np.zeros(12*nside**2, dtype=np.float32)
from tqdm import tqdm

def process_halo(args):
    jhalo, ra_all_np, dec_all_np, z_all_chunk, M_all_chunk, vlos_all_chunk, halo_cat_R200c_np, halo_cat_DV_np, max_paint_R200c_factor, nside_local, pixel_dtype = args
    
    vec = hp.ang2vec(ra_all_np[jhalo], dec_all_np[jhalo], lonlat=True)
    nearby_angle = max_paint_R200c_factor * halo_cat_R200c_np[jhalo] / halo_cat_DV_np[jhalo]
    nearby_pix = hp.query_disc(nside_local, vec, nearby_angle)
    
    if len(nearby_pix) == 0:
        return None  # Handle empty results
    
    nearby_pix = np.asarray(nearby_pix, dtype=pixel_dtype)
    nearby_ra, nearby_dec = hp.pix2ang(nside_local, nearby_pix, lonlat=True)
    
    ra1, dec1 = np.radians(ra_all_np[jhalo]), np.radians(dec_all_np[jhalo])
    ra2, dec2 = np.radians(nearby_ra), np.radians(nearby_dec)
    
    dra = ra1 - ra2
    ddec = dec1 - dec2
    a = np.sin(ddec/2)**2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra/2)**2
    theta = 2 * np.arcsin(np.sqrt(a))
    
    distances = (halo_cat_DV_np[jhalo] * theta).astype(np.float32)
    
    return (nearby_pix, distances, jhalo, len(nearby_pix))

def concatenate_results(results, M_all_chunk, z_all_chunk, vlos_all_chunk, halo_cat_DV_np, halo_cat_R200c_np, max_paint_R200c_factor, pixel_dtype):
    if not results:
        return None
        
    lengths = np.array([result[3] for result in results], dtype=np.int32)
    total_length = lengths.sum()
    
    nearby_pix_all = np.empty(total_length, dtype=pixel_dtype)
    distances_pix_all = np.empty(total_length, dtype=np.float32)
    halo_indices = np.empty(total_length, dtype=np.int32)
    
    end_ind_all = np.cumsum(lengths)
    start_ind_all = np.concatenate([[0], end_ind_all[:-1]])
    
    for i, (start, end, result) in enumerate(zip(start_ind_all, end_ind_all, results)):
        nearby_pix_all[start:end] = result[0]
        distances_pix_all[start:end] = result[1]
        halo_indices[start:end] = result[2]
    
    halo_start_indices = np.array([result[2] for result in results], dtype=np.int32)
    logM_ind_all = np.log(M_all_chunk[halo_indices], dtype=np.float32)
    z_ind_all = z_all_chunk[halo_indices].astype(np.float32)
    vlos_ind_all = vlos_all_chunk[halo_indices].astype(np.float32)
    ang_distance_all = halo_cat_DV_np[halo_start_indices].astype(np.float32)
    rp_max_all = (max_paint_R200c_factor * halo_cat_R200c_np[halo_start_indices]).astype(np.float32)
    
    return (nearby_pix_all, distances_pix_all, start_ind_all, end_ind_all, 
            logM_ind_all, z_ind_all, vlos_ind_all, ang_distance_all, rp_max_all)

def process_halos_in_batches(M_all_chunk, ra_all_chunk, dec_all_chunk, z_all_chunk, vlos_all_chunk, 
                           halo_cat_R200c, halo_cat_DA, max_paint_R200c_factor, nside, batch_size=1000):
    pixel_dtype = np.int32 if nside <= 8192 else np.int64
    
    ra_all_np = np.clip(np.array(ra_all_chunk, dtype=np.float32), 0.01, 359.99)
    dec_all_np = np.clip(np.array(dec_all_chunk, dtype=np.float32), -89.99, 89.99)
    z_all_np = np.array(z_all_chunk, dtype=np.float32)
    halo_cat_R200c_np = np.array(halo_cat_R200c, dtype=np.float32)
    halo_cat_DV_np = np.array(halo_cat_DA, dtype=np.float32)
    halo_vlos_np = np.array(vlos_all_chunk, dtype=np.float32)
    M_all_np = np.array(M_all_chunk, dtype=np.float32)
    
    n_halos = len(z_all_chunk)
    all_results = []
    
    for batch_start in range(0, n_halos, batch_size):
        batch_end = min(batch_start + batch_size, n_halos)
        print(f"Processing batch {batch_start//batch_size + 1}/{(n_halos-1)//batch_size + 1}")
        
        batch_args = []
        for jhalo in range(batch_start, batch_end):
            args = (jhalo, ra_all_np, dec_all_np, z_all_np, M_all_np, halo_vlos_np, 
                   halo_cat_R200c_np, halo_cat_DV_np, max_paint_R200c_factor, nside, pixel_dtype)
            batch_args.append(args)
        
        with Pool(cpu_count()) as pool:
            batch_results = pool.map(process_halo, batch_args)
        
        batch_results = [r for r in batch_results if r is not None]
        
        if batch_results:
            batch_M = M_all_np[batch_start:batch_end]
            batch_z = z_all_np[batch_start:batch_end] 
            batch_vlos = halo_vlos_np[batch_start:batch_end]
            batch_DV = halo_cat_DV_np[batch_start:batch_end]
            batch_R200c = halo_cat_R200c_np[batch_start:batch_end]
            
            batch_results_adjusted = []
            for result in batch_results:
                pix, dist, halo_idx, n_pix = result
                adjusted_idx = halo_idx  # Keep original index
                batch_results_adjusted.append((pix, dist, adjusted_idx, n_pix))
            
            batch_data = concatenate_results(batch_results_adjusted, M_all_np, z_all_np, halo_vlos_np, 
                                           halo_cat_DV_np, halo_cat_R200c_np, max_paint_R200c_factor, pixel_dtype)
            
            if batch_data is not None:
                all_results.append(batch_data)
        
        del batch_results, batch_args
        gc.collect()
    
    if not all_results:
        return None
        
    return final_concatenate_batches(all_results, pixel_dtype)

def final_concatenate_batches(all_results, pixel_dtype):
    """Final concatenation of batch results"""
    total_pix = sum(len(result[0]) for result in all_results)
    total_halos = sum(len(result[2]) for result in all_results)
    
    final_nearby_pix = np.empty(total_pix, dtype=pixel_dtype)
    final_distances = np.empty(total_pix, dtype=np.float32)
    final_logM = np.empty(total_pix, dtype=np.float32)
    final_z = np.empty(total_pix, dtype=np.float32)
    final_vlos = np.empty(total_pix, dtype=np.float32)
    final_start_ind = np.empty(total_halos, dtype=np.int32)
    final_end_ind = np.empty(total_halos, dtype=np.int32)
    final_ang_dist = np.empty(total_halos, dtype=np.float32)
    final_rp_max = np.empty(total_halos, dtype=np.float32)
    
    pix_offset = 0
    halo_offset = 0
    
    for result in all_results:
        nearby_pix_all, distances_pix_all, start_ind_all, end_ind_all, logM_ind_all, z_ind_all, vlos_ind_all, ang_distance_all, rp_max_all = result
        
        n_pix_batch = len(nearby_pix_all)
        n_halo_batch = len(start_ind_all)
        
        final_nearby_pix[pix_offset:pix_offset + n_pix_batch] = nearby_pix_all
        final_distances[pix_offset:pix_offset + n_pix_batch] = distances_pix_all
        final_logM[pix_offset:pix_offset + n_pix_batch] = logM_ind_all
        final_z[pix_offset:pix_offset + n_pix_batch] = z_ind_all
        final_vlos[pix_offset:pix_offset + n_pix_batch] = vlos_ind_all
        
        final_start_ind[halo_offset:halo_offset + n_halo_batch] = start_ind_all + pix_offset
        final_end_ind[halo_offset:halo_offset + n_halo_batch] = end_ind_all + pix_offset
        final_ang_dist[halo_offset:halo_offset + n_halo_batch] = ang_distance_all
        final_rp_max[halo_offset:halo_offset + n_halo_batch] = rp_max_all
        
        pix_offset += n_pix_batch
        halo_offset += n_halo_batch
    
    return (final_nearby_pix, final_distances, final_start_ind, final_end_ind,
            final_logM, final_z, final_vlos, final_ang_dist, final_rp_max)

for i in tqdm(range(num_chunks)):
    if i == num_chunks - 1:
        M_all_chunk = M_all[int(i*nh_max):]
        ra_all_chunk = ra_all[int(i*nh_max):]
        dec_all_chunk = dec_all[int(i*nh_max):]
        z_all_chunk = z_all[int(i*nh_max):]
        vlos_all_chunk = vlos_all[int(i*nh_max):]
    else:
        M_all_chunk = M_all[int(i*nh_max):int((i+1)*nh_max)]
        ra_all_chunk = ra_all[int(i*nh_max):int((i+1)*nh_max)]
        dec_all_chunk = dec_all[int(i*nh_max):int((i+1)*nh_max)]
        z_all_chunk = z_all[int(i*nh_max):int((i+1)*nh_max)]
        vlos_all_chunk = vlos_all[int(i*nh_max):int((i+1)*nh_max)]

    mock_params_dict = {}
    mock_params_dict['nside'] = nside
    mock_params_dict['get_ymap'] = True
    mock_params_dict['smooth_profiles'] = True

    halo_cat_scale_fac = 1./(1. + z_all_chunk)
    halo_cat_rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(B12_test.cosmo_jax, halo_cat_scale_fac) * 1e9
    mdef_delta = 200
    halo_cat_rho_treshold = mdef_delta * halo_cat_rho_c_z
    halo_cat_R200c = (M_all_chunk * 3.0 / 4.0 / jnp.pi / halo_cat_rho_treshold)**(1.0 / 3.0)
    halo_cat_DA = bkgrd.angular_diameter_distance(B12_test.cosmo_jax, halo_cat_scale_fac)
    max_paint_R200c_factor = 3.

    batch_size = int(nh_max//2)
    # print('processing batches')
    result = process_halos_in_batches(
        M_all_chunk, ra_all_chunk, dec_all_chunk, z_all_chunk, vlos_all_chunk,
        halo_cat_R200c, halo_cat_DA, max_paint_R200c_factor, nside, batch_size
    )
    # print('finished processing batches')

    if result is not None:
        nearby_pix_all, distances_pix_all, start_ind_all, end_ind_all, logM_ind_all, z_ind_all, vlos_ind_all, ang_distance_all, rp_max_all = result

        mock_params_dict['halo_z'] = jnp.array(z_all_chunk, dtype=jnp.float32)
        mock_params_dict['halo_ra'] = jnp.array(ra_all_chunk, dtype=jnp.float32)
        mock_params_dict['halo_dec'] = jnp.array(dec_all_chunk, dtype=jnp.float32)
        mock_params_dict['halo_M'] = jnp.array(M_all_chunk, dtype=jnp.float32)
        mock_params_dict['halo_vlos'] = jnp.array(vlos_all_chunk, dtype=jnp.float32)

        mock_params_dict['nearby_pix_all'] = jnp.array(nearby_pix_all)
        mock_params_dict['pix_prop_all'] = jnp.array([np.log(distances_pix_all), z_ind_all, logM_ind_all, vlos_ind_all]).T
        mock_params_dict['ang_distance_all'] = jnp.array(ang_distance_all)
        mock_params_dict['rp_max_all'] = jnp.array(rp_max_all)
        mock_params_dict['start_ind'] = jnp.array(start_ind_all, dtype=jnp.int32)
        mock_params_dict['end_ind'] = jnp.array(end_ind_all, dtype=jnp.int32)

        # print('generating mock map')
        mock_map_test = get_sim_map(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, mock_params_dict, Profiles_obj=B12_test)
        map_test += np.array(np.nan_to_num(mock_map_test.ymap_final), dtype=np.float32)
        # print('finished generating mock map')

    del result, mock_params_dict
    if 'mock_map_test' in locals():
        del mock_map_test
    gc.collect()

# Save results
saved = {'map_test': map_test}
pk.dump(saved, open(save_map_fname, 'wb'))


