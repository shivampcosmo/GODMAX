#!/usr/bin/env python
"""
Distributed tSZ map generation using JAX distributed across multiple nodes and GPUs.
This code automatically handles node/GPU distribution using JAX's distributed module.
"""

import sys
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# ==============================================================================
# JAX Multi-Node/Multi-GPU Initialization - MUST BE FIRST
# ==============================================================================
import jax
import jax.distributed
jax.distributed.initialize()

from jax.lib import xla_bridge
platform = xla_bridge.get_backend().platform
print(f"JAX process {jax.process_index()}/{jax.process_count()} initialized on platform: {platform}")
print(f"Local device count: {jax.local_device_count()}, Total device count: {jax.device_count()}")

jax.config.update('jax_platform_name', platform)
jax.config.update("jax_enable_x64", True)

import numpyro
numpyro.set_platform("gpu")
numpyro.enable_x64()

import numpy as np
import jax.numpy as jnp
from jax import pmap, jit
import healpy as hp
import yaml
import ast
from tqdm import tqdm
import pickle as pk
import h5py as h5
from multiprocessing import Pool, cpu_count
import warnings
import gc
import pathlib
from functools import partial
from typing import Dict, Tuple, Any

# Suppress warnings
warnings.filterwarnings("ignore")

# Import custom modules
import pathlib
curr_path = pathlib.Path().absolute()
abs_path_data = os.path.abspath(curr_path / "../../data/") 
abs_path_src = os.path.abspath(curr_path / "../../src/") 
abs_path_results = os.path.abspath(curr_path / "../../results/") 
sys.path.append((curr_path))
sys.path.append((abs_path_data))
sys.path.append((abs_path_results))
sys.path.append(abs_path_src)
import jax_cosmo.background as bkgrd
from get_B12_profile import Battaglia_12_16
import helpers.constants as constants
from get_sim_maps import get_sim_map

# ==============================================================================
# Configuration Functions
# ==============================================================================

def read_yaml(file_path: str) -> dict:
    """Read YAML configuration file."""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def generate_dicts(data: dict) -> Tuple[dict, dict, dict, dict]:
    """Generate parameter dictionaries from YAML data."""
    sim_params_dict = data.get('sim_params', {})
    halo_params_dict = data.get('halo_params', {})
    analysis_dict = data.get('analysis', {})
    other_params_dict = data.get('other_params', {})
    return sim_params_dict, halo_params_dict, analysis_dict, other_params_dict

def setup_paths() -> Tuple[pathlib.Path, ...]:
    """Setup and return necessary paths."""
    curr_path = pathlib.Path().absolute()
    abs_path_data = os.path.abspath(curr_path / "../../data/")
    abs_path_src = os.path.abspath(curr_path / "../../src/")
    abs_path_results = os.path.abspath(curr_path / "../../results/")
    
    # Add to path
    for path in [str(curr_path), abs_path_data, abs_path_results, abs_path_src]:
        if path not in sys.path:
            sys.path.append(path)
    
    return curr_path, abs_path_data, abs_path_src, abs_path_results

# ==============================================================================
# Data Loading and Distribution Functions
# ==============================================================================

def load_and_filter_catalog(fname: str, zmin: float = 0.05, zmax: float = 1.5, 
                           Mmin: float = 1e13, Mmax: float = 4e15) -> Tuple[np.ndarray, ...]:
    """Load and filter halo catalog."""
    with h5.File(fname, 'r') as f:
        M200c_all = f['halo_mass_m200c'][:]
        z_all = f['redshift'][:]
        pos = f['Position'][:]
        v_all = f['Velocity'][:]
    
    # Convert positions to RA/Dec
    ra_all, dec_all = hp.vec2ang(pos, lonlat=True)
    
    # Calculate line-of-sight velocities
    vlos_all = np.sum(v_all * hp.ang2vec(ra_all, dec_all, lonlat=True), axis=1)
    
    # Filter halos
    mask = (z_all > zmin) & (z_all < zmax) & (M200c_all > Mmin) & (M200c_all < Mmax)
    
    return (ra_all[mask], dec_all[mask], z_all[mask], 
            M200c_all[mask], vlos_all[mask])

def distribute_halos_across_nodes(ra_all: np.ndarray, dec_all: np.ndarray, 
                                 z_all: np.ndarray, M200c_all: np.ndarray, 
                                 vlos_all: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, ...]:
    """Distribute halos across nodes, sorted by mass."""
    # Sort by mass (descending)
    argsort = np.flip(np.argsort(M200c_all))
    
    # Apply sorting
    ra_all = ra_all[argsort]
    dec_all = dec_all[argsort]
    z_all = z_all[argsort]
    M200c_all = M200c_all[argsort]
    vlos_all = vlos_all[argsort]
    
    # Ensure divisibility by number of nodes
    n_halos = len(M200c_all)
    n_halos_trimmed = (n_halos // num_nodes) * num_nodes
    
    if n_halos_trimmed < n_halos:
        print(f"Trimming {n_halos - n_halos_trimmed} halos for even distribution")
        ra_all = ra_all[:n_halos_trimmed]
        dec_all = dec_all[:n_halos_trimmed]
        z_all = z_all[:n_halos_trimmed]
        M200c_all = M200c_all[:n_halos_trimmed]
        vlos_all = vlos_all[:n_halos_trimmed]
    
    # Split data for all nodes
    ra_splits = np.array_split(ra_all, num_nodes)
    dec_splits = np.array_split(dec_all, num_nodes)
    z_splits = np.array_split(z_all, num_nodes)
    m_splits = np.array_split(M200c_all, num_nodes)
    vlos_splits = np.array_split(vlos_all, num_nodes)
    
    return list(zip(ra_splits, dec_splits, z_splits, m_splits, vlos_splits))

# ==============================================================================
# CPU Pre-processing Functions
# ==============================================================================

def process_halo_cpu(args: Tuple) -> Tuple[np.ndarray, ...]:
    """Process a single halo on CPU to find nearby pixels."""
    (jhalo, ra_arr, dec_arr, r200c_arr, da_arr, m_arr, 
     z_arr, vlos_arr, nside, max_paint_factor) = args
    
    # Find nearby pixels
    vec = hp.ang2vec(ra_arr[jhalo], dec_arr[jhalo], lonlat=True)
    nearby_angle = max_paint_factor * r200c_arr[jhalo] / da_arr[jhalo]
    nearby_pix = hp.query_disc(nside, vec, nearby_angle)
    nearby_ra, nearby_dec = hp.pix2ang(nside, nearby_pix, lonlat=True)
    
    # Calculate physical distances using Haversine formula
    def haversine(theta):
        return np.sin(theta / 2.) ** 2.
    
    ra1, dec1 = np.deg2rad(ra_arr[jhalo]), np.deg2rad(dec_arr[jhalo])
    ra2, dec2 = np.deg2rad(nearby_ra), np.deg2rad(nearby_dec)
    theta = 2. * np.arcsin(np.sqrt(
        haversine(dec1 - dec2) + np.cos(dec1) * np.cos(dec2) * haversine(ra1 - ra2)
    ))
    physical_distances = da_arr[jhalo] * theta
    
    num_pix = len(nearby_pix)
    return (
        np.array(nearby_pix, dtype=np.int32),
        np.array(physical_distances, dtype=np.float64),
        np.full(num_pix, np.log(m_arr[jhalo]), dtype=np.float64),
        np.full(num_pix, z_arr[jhalo], dtype=np.float64),
        np.full(num_pix, vlos_arr[jhalo], dtype=np.float64),
        np.full(num_pix, da_arr[jhalo], dtype=np.float64),
        np.full(num_pix, max_paint_factor * r200c_arr[jhalo], dtype=np.float64)
    )

def concatenate_halo_results(results: list) -> Tuple[np.ndarray, ...]:
    """Concatenate results from parallel halo processing."""
    if not results:
        return (np.array([]), np.array([]), np.array([]), np.array([]), 
                np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    
    lengths = np.array([len(r[0]) for r in results])
    total_length = lengths.sum()
    
    if total_length == 0:
        return (np.array([]), np.array([]), np.array([]), np.array([]), 
                np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    
    # Calculate indices
    end_ind_all = np.cumsum(lengths)
    start_ind_all = np.zeros_like(end_ind_all)
    start_ind_all[1:] = end_ind_all[:-1]
    
    # Pre-allocate arrays
    nearby_pix_all = np.empty(total_length, dtype=np.int32)
    distances_pix_all = np.empty(total_length, dtype=np.float64)
    logM_ind_all = np.empty(total_length, dtype=np.float64)
    z_ind_all = np.empty(total_length, dtype=np.float64)
    vlos_ind_all = np.empty(total_length, dtype=np.float64)
    ang_distance_all = np.empty(total_length, dtype=np.float64)
    rp_max_all = np.empty(total_length, dtype=np.float64)
    
    # Fill arrays
    for i, res in enumerate(results):
        start, end = start_ind_all[i], end_ind_all[i]
        nearby_pix_all[start:end] = res[0]
        distances_pix_all[start:end] = res[1]
        logM_ind_all[start:end] = res[2]
        z_ind_all[start:end] = res[3]
        vlos_ind_all[start:end] = res[4]
        ang_distance_all[start:end] = res[5]
        rp_max_all[start:end] = res[6]
    
    return (nearby_pix_all, distances_pix_all, start_ind_all, end_ind_all, 
            logM_ind_all, z_ind_all, vlos_ind_all, ang_distance_all, rp_max_all)

def prepare_chunk_for_gpu(M_chunk: np.ndarray, ra_chunk: np.ndarray, 
                          dec_chunk: np.ndarray, z_chunk: np.ndarray, 
                          vlos_chunk: np.ndarray, B12_test: Any, 
                          nside: int, n_cpus: int = None) -> Dict:
    """Prepare a chunk of halos for GPU processing."""
    if len(M_chunk) == 0:
        return None
    
    # Calculate halo properties
    scale_fac = 1. / (1. + z_chunk)
    rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(B12_test.cosmo_jax, scale_fac) * 1e9
    rho_threshold = 200 * rho_c_z
    R200c = (M_chunk * 3.0 / (4.0 * np.pi * rho_threshold))**(1.0 / 3.0)
    DA = bkgrd.angular_diameter_distance(B12_test.cosmo_jax, scale_fac)
    max_paint_R200c_factor = 3.0
    
    # Clip coordinates to valid ranges
    ra_chunk_clipped = np.clip(ra_chunk, 0.01, 359.99)
    dec_chunk_clipped = np.clip(dec_chunk, -89.99, 89.99)
    
    # Prepare arguments for multiprocessing
    pool_args = [
        (j, ra_chunk_clipped, dec_chunk_clipped, np.array(R200c), 
         np.array(DA), M_chunk, z_chunk, vlos_chunk, nside, max_paint_R200c_factor)
        for j in range(len(z_chunk))
    ]
    
    # Process halos in parallel on CPU
    if n_cpus is None:
        n_cpus = cpu_count()
    
    with Pool(processes=n_cpus) as pool:
        results = pool.map(process_halo_cpu, pool_args)
    
    # Concatenate results
    (nearby_pix_all, distances_pix_all, start_ind_all, end_ind_all, 
     logM_ind_all, z_ind_all, vlos_ind_all, ang_distance_all, rp_max_all) = concatenate_halo_results(results)
    
    if len(nearby_pix_all) == 0:
        return None
    
    return {
        'halo_z': jnp.array(z_chunk),
        'halo_ra': jnp.array(ra_chunk_clipped),
        'halo_dec': jnp.array(dec_chunk_clipped),
        'halo_M': jnp.array(M_chunk),
        'halo_vlos': jnp.array(vlos_chunk),
        'nearby_pix_all': jnp.array(nearby_pix_all),
        'pix_prop_all': jnp.array([
            np.log(distances_pix_all), z_ind_all, 
            logM_ind_all, vlos_ind_all
        ]).T,
        'start_ind': jnp.int32(start_ind_all),
        'end_ind': jnp.int32(end_ind_all),
        'ang_distance_all': jnp.array(ang_distance_all)[start_ind_all],
        'rp_max_all': jnp.array(rp_max_all)[start_ind_all],
        'nside': nside,
        'get_ymap': True,
        'smooth_profiles': True
    }

def get_chunk_size(nside: int) -> int:
    """Get optimal chunk size based on nside."""
    chunk_sizes = {
        512: 2e6,
        1024: 1e6,
        2048: 5e5,
        4096: 5e4,
        8192: 4e3
    }
    return int(chunk_sizes.get(nside, 1e5))

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    # Parse command line arguments
    if len(sys.argv) != 2:
        if jax.process_index() == 0:
            print(f"Usage: python {sys.argv[0]} <nside>")
            print("Example: python tsz_map_distributed.py 2048")
        sys.exit(1)
    
    nside = int(ast.literal_eval(sys.argv[1]))
    
    # Get node rank and world size from JAX distributed
    node_rank = jax.process_index()
    num_nodes = jax.process_count()
    n_gpus_local = jax.local_device_count()
    
    if node_rank == 0:
        print(f"="*60)
        print(f"Starting distributed tSZ map generation")
        print(f"NSIDE: {nside}")
        print(f"Total nodes: {num_nodes}")
        print(f"GPUs per node: {n_gpus_local}")
        print(f"Total GPUs: {jax.device_count()}")
        print(f"="*60)
    
    # Setup paths
    setup_paths()
    
    # Load configuration
    yaml_file_path = '/mnt/home/spandey/ceph/GODMAX/param_files/params_default.yaml'
    data = read_yaml(yaml_file_path)
    sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(data)
    
    # Update halo parameters
    halo_params_dict.update({
        'rmin': 0.0001, 'rmax': 10.0, 'nr': 126,
        'zmin': 0.01, 'zmax': 4.1, 'nz': 127,
        'lg10_Mmin': 12.0, 'lg10_Mmax': 16.0, 'nM': 128
    })
    
    # Setup cosmology
    cosmo_params_dict = {
        'w0': -1.0, 'flat': True, 'H0': 67.74,
        'Om0': 0.3089, 'Ob0': 0.0486,
        'sigma8': 0.8159, 'ns': 0.9667
    }
    
    # Initialize Battaglia profile
    B12_test = Battaglia_12_16(
        sim_params_dict={'cosmo': cosmo_params_dict, 'init_power': True},
        halo_params_dict=halo_params_dict
    )
    
    # Load and distribute data
    zmax = 1.5
    # if node_rank == 0:
    print("Loading halo catalog...")
    fname = '/mnt/ceph/users/abayer/fastpm/halfdome/stampede2_3750Mpch_6144cube/final_res/halos/lightcone_100.hdf5'
    ra_all, dec_all, z_all, M200c_all, vlos_all = load_and_filter_catalog(
        fname, zmin=0.05, zmax=zmax, Mmin=1e13, Mmax=4e15
    )
    
    print(f"Total halos after filtering: {len(M200c_all):,}")
    
    # Distribute halos across nodes
    data_splits = distribute_halos_across_nodes(
        ra_all, dec_all, z_all, M200c_all, vlos_all, num_nodes
    )
    # else:
    #     data_splits = None
    
    # Broadcast data from rank 0 to all other ranks
    # data_splits = jax.experimental.multihost_utils.broadcast_one_to_all(data_splits)
    
    ra_node, dec_node, z_node, M200c_node, vlos_node = data_splits[node_rank]
    
    print(f"Node {node_rank}: Processing {len(M200c_node):,} halos")
    print(f"Node {node_rank}: log10(M) range: [{np.min(np.log10(M200c_node)):.2f}, "
          f"{np.max(np.log10(M200c_node)):.2f}]")
    print(f"Node {node_rank}: z range: [{np.min(z_node):.2f}, {np.max(z_node):.2f}]")
    
    # Initialize output map
    map_node = np.zeros(12 * nside**2, dtype=np.float32)
    
    # Determine chunk size
    chunk_size = get_chunk_size(nside)
    num_chunks = int(np.ceil(len(M200c_node) / chunk_size))
    
    print(f"Node {node_rank}: Processing {num_chunks} chunks of size {chunk_size:.0f}")
    
    # Define pmapped function for multi-GPU execution
    @partial(pmap, in_axes=(None, None, None, None, 
                           {'start_ind': 0, 'end_ind': 0, 
                            'ang_distance_all': 0, 'rp_max_all': 0,
                            'halo_z': 0, 'halo_ra': 0, 'halo_dec': 0,
                            'halo_M': 0, 'halo_vlos': 0,
                            'nearby_pix_all': None, 'pix_prop_all': None,
                            'nside': None, 'get_ymap': None, 'smooth_profiles': None},
                           None),
            static_broadcasted_argnums=(0, 1, 2, 3, 5))
    def get_sim_map_pmap(sim_params, halo_params, analysis, other_params, mock_params, profile):
        return get_sim_map(sim_params, halo_params, analysis, other_params, mock_params, profile)
    
    # Process chunks
    for chunk_idx in tqdm(range(num_chunks), desc=f"Node {node_rank}", disable=(node_rank != 0)):
        # Get chunk slice
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(M200c_node))
        
        M_chunk = M200c_node[start_idx:end_idx]
        ra_chunk = ra_node[start_idx:end_idx]
        dec_chunk = dec_node[start_idx:end_idx]
        z_chunk = z_node[start_idx:end_idx]
        vlos_chunk = vlos_node[start_idx:end_idx]
        
        if len(M_chunk) == 0:
            continue
        
        # Prepare chunk for GPU processing
        mock_params_dict = prepare_chunk_for_gpu(
            M_chunk, ra_chunk, dec_chunk, z_chunk, vlos_chunk,
            B12_test, nside
        )
        
        if mock_params_dict is None:
            continue
        
        # Check if we can use multiple GPUs
        num_halos_chunk = len(mock_params_dict['halo_z'])
        
        if n_gpus_local > 1 and num_halos_chunk >= n_gpus_local:
            # Ensure divisibility by number of GPUs
            num_halos_pmap = (num_halos_chunk // n_gpus_local) * n_gpus_local
            
            # Prepare data for pmap
            mock_params_pmap = {}
            for key, val in mock_params_dict.items():
                if key in ['start_ind', 'end_ind', 'ang_distance_all', 'rp_max_all',
                          'halo_z', 'halo_ra', 'halo_dec', 'halo_M', 'halo_vlos']:
                    # Reshape for pmap: (n_gpus, halos_per_gpu)
                    mock_params_pmap[key] = val[:num_halos_pmap].reshape(n_gpus_local, -1)
                else:
                    # Broadcast to all GPUs
                    mock_params_pmap[key] = val
            
            # Execute on multiple GPUs
            mock_maps_gpus = get_sim_map_pmap(
                sim_params_dict, halo_params_dict, analysis_dict,
                other_params_dict, mock_params_pmap, B12_test
            )
            
            # Sum maps from all GPUs
            chunk_map = jnp.sum(mock_maps_gpus.ymap_final, axis=0)
        else:
            # Single GPU execution
            mock_map = get_sim_map(
                sim_params_dict, halo_params_dict, analysis_dict,
                other_params_dict, mock_params_dict, B12_test
            )
            chunk_map = mock_map.ymap_final
        
        # Add to node's map
        map_node += np.array(np.nan_to_num(chunk_map), dtype=np.float32)
        
        # Periodic cleanup
        if chunk_idx % 5 == 0:
            jax.clear_caches()
            gc.collect()
    
    # Save node's map
    sdir = '/mnt/home/spandey/ceph/GODMAX/notebooks/all_arxiv/mock_gen/maps_halfdome/'
    os.makedirs(sdir, exist_ok=True)
    save_map_fname = sdir + f'tSZ_sim_B12_node_{node_rank}_{num_nodes}_nside_{nside}_zmax_{zmax}.pkl'
    
    print(f"Node {node_rank}: Saving map to {save_map_fname}")
    saved = {
        'map_node': map_node,
        'node_rank': node_rank,
        'num_nodes': num_nodes,
        'n_halos_processed': len(M200c_node),
        'nside': nside,
        'zmax': zmax
    }
    
    with open(save_map_fname, 'wb') as f:
        pk.dump(saved, f)
    
    # Print summary
    print(f"Node {node_rank} Summary:")
    print(f"  Processed {len(M200c_node):,} halos")
    print(f"  Map mean: {np.mean(map_node):.2e}, std: {np.std(map_node):.2e}")
    print(f"  Non-zero pixels: {np.sum(map_node != 0):,}")
    
    # Wait for all nodes to complete
    jax.experimental.multihost_utils.sync_global_devices("complete")
    
    if node_rank == 0:
        print(f"="*60)
        print("All nodes completed successfully!")
        print(f"="*60)

if __name__ == "__main__":
    main()