import sys, os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
# ==============================================================================
# IMPORTANT: Import and set multiprocessing start method early
import multiprocessing as mp
# ==============================================================================
from jax.lib import xla_bridge
import jax.numpy as jnp
from jax import pmap
import numpy as np
import numpyro
import warnings
import yaml
import ast
from tqdm import tqdm
import pickle as pk
import h5py as h5
from multiprocessing import Pool, cpu_count
import gc
import pathlib
from mpi4py import MPI
import healpy as hp

# ==============================================================================
# === WORKER FUNCTION ==========================================================
# ==============================================================================
# This function MUST be defined at the top level of the module so it can be
# pickled and sent to the 'spawned' child processes.
def process_halo_sub_chunk(args):
    # Imports are needed here because 'spawn' creates a fresh process
    import numpy as np
    import healpy as hp

    start_idx, end_idx, ra_chunk, dec_chunk, M_chunk, z_chunk, vlos_chunk, R200c, DA, nside, max_paint_factor = args
    list_nearby_pix, list_distances_pix = [], []
    list_logM_ind, list_z_ind, list_vlos_ind = [], [], []
    list_ang_distance, list_rp_max = [], []
    list_start_ind, list_end_ind = [], []
    current_index = 0

    for jhalo in range(start_idx, end_idx):
        vec = hp.ang2vec(ra_chunk[jhalo], dec_chunk[jhalo], lonlat=True)
        nearby_angle = max_paint_factor * R200c[jhalo] / DA[jhalo]
        nearby_pix = hp.query_disc(nside, vec, nearby_angle, inclusive=True)
        if len(nearby_pix) == 0: continue
        nearby_ra, nearby_dec = hp.pix2ang(nside, nearby_pix, lonlat=True)

        def hav(theta): return np.sin(theta / 2.) ** 2
        ra1, dec1 = np.deg2rad(ra_chunk[jhalo]), np.deg2rad(dec_chunk[jhalo])
        ra2, dec2 = np.deg2rad(nearby_ra), np.deg2rad(nearby_dec)
        theta = 2. * np.arcsin(np.sqrt(hav(dec1 - dec2) + np.cos(dec1) * np.cos(dec2) * hav(ra1 - ra2)))
        physical_distances = DA[jhalo] * theta
        num_pix = len(nearby_pix)
        
        list_nearby_pix.append(nearby_pix)
        list_distances_pix.append(physical_distances)
        list_logM_ind.append(np.repeat(np.log(M_chunk[jhalo]), num_pix))
        list_z_ind.append(np.repeat(z_chunk[jhalo], num_pix))
        list_vlos_ind.append(np.repeat(vlos_chunk[jhalo], num_pix))
        list_ang_distance.append(DA[jhalo])
        list_rp_max.append(max_paint_factor * R200c[jhalo])
        list_start_ind.append(current_index)
        current_index += num_pix
        list_end_ind.append(current_index)

    if not list_nearby_pix: return None

    return (
        np.concatenate(list_nearby_pix, dtype=np.int32),
        np.concatenate(list_distances_pix, dtype=np.float32),
        np.concatenate(list_logM_ind, dtype=np.float32),
        np.concatenate(list_z_ind, dtype=np.float32),
        np.concatenate(list_vlos_ind, dtype=np.float32),
        np.array(list_ang_distance, dtype=np.float32),
        np.array(list_rp_max, dtype=np.float32),
        np.array(list_start_ind, dtype=np.int32),
        np.array(list_end_ind, dtype=np.int32)
    )

# ==============================================================================
# === MAIN APPLICATION LOGIC ===================================================
# ==============================================================================
def main():
    platform = xla_bridge.get_backend().platform
    print(f"JAX process {jax.process_index()}/{jax.process_count()} initialized on platform: {platform}")
    print(f"Local device count: {jax.local_device_count()}, Total device count: {jax.device_count()}")

    jax.config.update('jax_platform_name', platform)
    jax.config.update("jax_enable_x64", True)

    numpyro.set_platform("gpu")
    numpyro.enable_x64()
    warnings.filterwarnings("ignore")

    curr_path = pathlib.Path().absolute()
    abs_path_src = os.path.abspath(curr_path / "../../src/") 
    sys.path.append(str(abs_path_src))
    from get_B12_profile import Battaglia_12_16
    import jax_cosmo.background as bkgrd
    import helpers.constants as constants
    from get_sim_maps import get_sim_map

    def read_yaml(file_path):
        with open(file_path, 'r') as file: return yaml.safe_load(file)

    def generate_dicts(data):
        return (data.get('sim_params', {}), data.get('halo_params', {}),
                data.get('analysis', {}), data.get('other_params', {}))

    if len(sys.argv) != 2:
        if jax.process_index() == 0: print(f"Usage: python {sys.argv[0]} <nside>")
        sys.exit(1)

    nside = int(ast.literal_eval(sys.argv[1]))
    node_rank = jax.process_index()
    num_nodes = jax.process_count()
    print(f'Node {node_rank}/{num_nodes} | NSIDE: {nside}')

    yaml_file_path = '/mnt/home/spandey/ceph/GODMAX/param_files/params_default.yaml'
    data = read_yaml(yaml_file_path)
    sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(data)
    halo_params_dict.update({'rmin': 0.0001, 'rmax': 10.0, 'nr': 126, 'zmin': 0.01, 'zmax': 4.1, 'nz': 127, 'lg10_Mmin': 12.0, 'lg10_Mmax': 16.0, 'nM': 128})
    cosmo_params_dict = {'w0':-1.0 ,'flat': True, 'H0': 67.74, 'Om0': 0.3089, 'Ob0': 0.0486, 'sigma8':0.8159 ,'ns': 0.9667}
    B12_test = Battaglia_12_16(sim_params_dict={'cosmo':cosmo_params_dict, 'init_power':False}, halo_params_dict=halo_params_dict)

    COMM = MPI.COMM_WORLD
    node_rank_mpi, num_nodes_mpi = COMM.Get_rank(), COMM.Get_size()
    all_nodes_data_np = None
    if node_rank_mpi == 0:
        print(f"Node 0: Loading and preparing halo catalog for all {num_nodes_mpi} nodes...")
        with h5.File('/mnt/ceph/users/abayer/fastpm/halfdome/stampede2_3750Mpch_6144cube/final_res/halos/lightcone_100.hdf5', 'r') as f:
            M200c_all, z_all, pos, v_all = f['halo_mass_m200c'][:], f['redshift'][:], f['Position'][:], f['Velocity'][:]
        ra_all, dec_all = hp.vec2ang(pos, lonlat=True)
        vlos_all = np.sum(v_all * hp.ang2vec(ra_all, dec_all, lonlat=True), axis=1)
        indsel = np.where((z_all > 0.05) & (z_all < 1.5) & (M200c_all < 4e15) & (M200c_all > (10**13.0)))[0]
        ra_all, dec_all, z_all, M200c_all, vlos_all = ra_all[indsel], dec_all[indsel], z_all[indsel], M200c_all[indsel], vlos_all[indsel]
        argsort = np.flip(np.argsort(M200c_all))
        Nsel = (len(argsort) // num_nodes_mpi) * num_nodes_mpi
        argsort = argsort[:Nsel]
        ra_all, dec_all, z_all, M200c_all, vlos_all = ra_all[argsort], dec_all[argsort], z_all[argsort], M200c_all[argsort], vlos_all[argsort]
        all_nodes_data_np = list(zip(np.array_split(ra_all, num_nodes_mpi), np.array_split(dec_all, num_nodes_mpi), np.array_split(z_all, num_nodes_mpi), np.array_split(M200c_all, num_nodes_mpi), np.array_split(vlos_all, num_nodes_mpi)))

    print(f"Node {node_rank_mpi}: Waiting to receive broadcasted halo data...")
    all_nodes_data = COMM.bcast(all_nodes_data_np, root=0)
    print(f"Node {node_rank_mpi}: Data received successfully.")
    ra_all, dec_all, z_all, M200c_all, vlos_all = all_nodes_data[node_rank_mpi]
    print(f'Node {node_rank}: Num halos = {len(M200c_all):,}')

    sdir = '/mnt/home/spandey/ceph/GODMAX/notebooks/all_arxiv/mock_gen/maps_halfdome/'
    save_map_fname = sdir + f'tSZ_sim_B12_testv11_nside_{nside}_split_{node_rank}_{num_nodes}_zmax_1.5.pkl'
    nh_max = 50000 if nside == 4096 else 100000
    num_chunks = int(np.ceil(len(M200c_all) / nh_max))
    map_final_node = np.zeros(12 * nside**2, dtype=np.float32)

    for i in tqdm(range(num_chunks), desc=f"Node {node_rank} Chunks"):
        start, end = int(i * nh_max), int(min((i + 1) * nh_max, len(M200c_all)))
        M_chunk_main, ra_chunk_main, dec_chunk_main, z_chunk_main, vlos_chunk_main = \
            M200c_all[start:end], ra_all[start:end], dec_all[start:end], z_all[start:end], vlos_all[start:end]
        if len(M_chunk_main) == 0: continue

        scale_fac = 1. / (1. + z_chunk_main)
        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(B12_test.cosmo_jax, scale_fac) * 1e9
        R200c_main = (M_chunk_main * 3.0 / (4.0 * np.pi * (200 * rho_c_z)))**(1.0 / 3.0)
        DA_main = bkgrd.angular_diameter_distance(B12_test.cosmo_jax, scale_fac)
        ra_chunk_clipped, dec_chunk_clipped = np.clip(ra_chunk_main, 0.01, 359.99), np.clip(dec_chunk_main, -89.99, 89.99)
        n_workers = cpu_count()
        sub_chunk_indices = np.linspace(0, len(z_chunk_main), n_workers + 1, dtype=int)
        pool_args = [(sub_chunk_indices[k], sub_chunk_indices[k+1], ra_chunk_clipped, dec_chunk_clipped, M_chunk_main, z_chunk_main, vlos_chunk_main, R200c_main, DA_main, nside, 3.0) for k in range(n_workers)]

        with Pool(processes=n_workers) as pool: results = pool.map(process_halo_sub_chunk, pool_args)
        
        results = [res for res in results if res is not None]
        if not results: continue
        
        parts = list(zip(*results))
        total_pix_offset = 0
        corrected_start_ind_parts, corrected_end_ind_parts = [], []
        for j in range(len(parts[7])):
            start, end = parts[7][j], parts[8][j]
            corrected_start_ind_parts.append(start + total_pix_offset)
            corrected_end_ind_parts.append(end + total_pix_offset)
            if len(end) > 0: total_pix_offset = end[-1] + total_pix_offset

        nearby_pix_all, distances_pix_all, logM_ind_all, z_ind_all, vlos_ind_all, ang_distance_all, rp_max_all, start_ind_all, end_ind_all = \
            [np.concatenate(p) for p in parts[:7]] + [np.concatenate(corrected_start_ind_parts), np.concatenate(corrected_end_ind_parts)]
        del results, parts, corrected_start_ind_parts, corrected_end_ind_parts; gc.collect()

        N_gpus = jax.local_device_count()
        num_halos_pmap = (len(ang_distance_all) // N_gpus) * N_gpus
        if num_halos_pmap == 0: continue
            
        start_ind_gpus, end_ind_gpus, ang_dist_gpus, rp_max_gpus = \
            [jnp.array(arr[:num_halos_pmap]).reshape(N_gpus, -1) for arr in [start_ind_all, end_ind_all, ang_distance_all, rp_max_all]]
        pix_prop_all_jnp = jnp.stack([jnp.log(distances_pix_all), z_ind_all, logM_ind_all, vlos_ind_all], axis=1)

        pmap_mock_params_dict = {'start_ind': start_ind_gpus, 'end_ind': end_ind_gpus, 'ang_distance_all': ang_dist_gpus, 'rp_max_all': rp_max_gpus,
                                 'nearby_pix_all': jnp.array(nearby_pix_all), 'pix_prop_all': pix_prop_all_jnp, 'nside': nside, 'get_ymap': True}
        del nearby_pix_all, distances_pix_all, logM_ind_all, z_ind_all, vlos_ind_all, pix_prop_all_jnp, start_ind_all, end_ind_all, ang_distance_all, rp_max_all; gc.collect()

        pmapped_get_sim_map = pmap(get_sim_map, in_axes=(None, None, None, None, {'start_ind': 0, 'end_ind': 0, 'ang_distance_all': 0, 'rp_max_all': 0, 'nearby_pix_all': None, 'pix_prop_all': None, 'nside': None, 'get_ymap': None}, None), static_broadcasted_argnums=(0, 1, 2, 3, 5))
        mock_maps_gpus = pmapped_get_sim_map(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, pmap_mock_params_dict, B12_test)
        chunk_map = jnp.sum(mock_maps_gpus.ymap_final, axis=0)
        map_final_node += np.array(np.nan_to_num(chunk_map), dtype=np.float32)
        jax.clear_caches()

    print(f"Node {node_rank}: Saving map to {save_map_fname}")
    with open(save_map_fname, 'wb') as f: pk.dump({'map_test': map_final_node}, f)
    print(f"Node {node_rank}: Processing complete.")

# ==============================================================================
# === SCRIPT ENTRYPOINT ========================================================
# ==============================================================================
if __name__ == "__main__":
    # This block is executed only in the main process, not in spawned children.
    
    # 1. Set the start method to 'spawn' for CUDA and JAX distributed safety.
    #    This must be done before any other CUDA/JAX calls.
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method could not be set (likely already set).")

    # 2. Initialize the JAX distributed system ONLY in the main processes.
    import jax.distributed
    jax.distributed.initialize()

    # 3. Run the main application logic.
    main()