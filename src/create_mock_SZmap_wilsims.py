import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_ENABLE_X64"] = "True"
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']=".97"
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']="False"
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95'
os.chdir('/mnt/home/spandey/ceph/GODMAX/src/')
# from jax.config import config
from jax import config
# config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import colossus 
from jax import vmap, grad
#import pyccl as ccl
import pickle as pk
import scipy.interpolate as interp
import warnings

# Suppress RuntimeWarnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_ENABLE_X64"] = "True"
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.95'


# Change the current working directory to the desired path
os.chdir('/mnt/home/spandey/ceph/GODMAX/src/')
# from jax.config import config
from jax import config
# config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
import colossus 
from jax import vmap, grad
%matplotlib inline
import matplotlib.pyplot as pl
pl.rc('text', usetex=True)
# Palatino
pl.rc('font', family='DejaVu Sans')
#import pyccl as ccl
import pickle as pk
import jax
import scipy.interpolate as interp
import healpy as hp
import numpy as np
from multiprocessing import Pool, cpu_count
from astropy.io import fits
import constants
import jax_cosmo.background as bkgrd
from get_sim_on_halos_NO_CONC_jit import get_mock_map
import h5py as h5
%load_ext autoreload
%autoreload 2

import warnings

# Suppress RuntimeWarnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

cosmo_params_dict = {'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.0463, 'sigma8': 0.834, 'ns': 0.9624, 'w0':-1.0}
sim_params_dict = {}
sim_params_dict['nfw_trunc'] = True
sim_params_dict['gamma_rhogas'] = 5.0
sim_params_dict['delta_rhogas'] = 9.0
# sim_params_dict['theta_co'] = 0.01
# sim_params_dict['theta_ej'] = 1.0

sim_params_dict['theta_co_0'] = 0.01
sim_params_dict['log10_Mstar0_theta_co'] = 15.0
sim_params_dict['nu_theta_co_M'] = 0.0
sim_params_dict['nu_theta_co_z'] = 0.0

sim_params_dict['theta_ej_0'] = 4.0
sim_params_dict['log10_Mstar0_theta_ej'] = 15.0
sim_params_dict['nu_theta_ej_M'] = 0.0
sim_params_dict['nu_theta_ej_z'] = 0.5

sim_params_dict['log10_Mc0'] = 15.1
sim_params_dict['log10_Mstar0'] = 14.0
sim_params_dict['mu_beta'] = 0.21
sim_params_dict['nu_z'] = -5.0
sim_params_dict['nu_M'] = 0.0

sim_params_dict['eta_star'] = 0.3
sim_params_dict['eta_cga'] = 0.6


sim_params_dict['neg_bhse_plus_1'] = 0.833
sim_params_dict['A_starcga'] = 0.09
sim_params_dict['log10_M1_starcga'] = 11.4
sim_params_dict['epsilon_rt'] = 4.0


sim_params_dict['a_zeta'] = 0.3
sim_params_dict['n_zeta'] = 2
sim_params_dict['alpha_nt'] = 0.18
sim_params_dict['beta_nt'] = 0.5
sim_params_dict['n_nt'] = 0.3
sim_params_dict['cosmo'] = cosmo_params_dict


halo_params_dict = {}
halo_params_dict['rmin'], halo_params_dict['rmax'], halo_params_dict['nr'] = 1e-3, 18, 96
# halo_params_dict['zmin'], halo_params_dict['zmax'], halo_params_dict['nz'] = 0.001, 0.002, 2
halo_params_dict['zmin'], halo_params_dict['zmax'], halo_params_dict['nz'] = 0.001, 3.0, 16
# halo_params_dict['z_array'] = np.array([1e-3, 0.5, 1.0])
# halo_params_dict['z_array'] = np.array([1e-3])
# halo_params_dict['z_array'] = np.array([0.5])
# halo_params_dict['z_array'] = np.array([1.0])
# halo_params_dict['nz'] = len(halo_params_dict['z_array'])
halo_params_dict['lg10_Mmin'], halo_params_dict['lg10_Mmax'], halo_params_dict['nM'] = 12.0, 15.0, 48
halo_params_dict['cmin'], halo_params_dict['cmax'], halo_params_dict['nc'] = 3, 7, 24
halo_params_dict['ellmin'], halo_params_dict['ellmax'], halo_params_dict['nell'] = 8, 2**15, 64
# try:
halo_params_dict['sig_logc_z_array'] = np.ones(halo_params_dict['nz']) * 0.01
halo_params_dict['mdef'] = '200c'
halo_params_dict['hmf_model'] = 'T10'
halo_params_dict['conc_model'] = 'Duffy08'
halo_params_dict['do_corr_2h_mm'] = True

# halo_params_dict['do_corr_2h_mm'] = False

df = fits.open('/mnt/home/spandey/ceph/GODMAX/data/sim_3x2pt_simulated_DV_PKproject_values_bestfit_maglim_3x2LCDM_final.fits') 
z_array = df['nz_source'].data['Z_MID']
nz_info_dict = {}
nz_info_dict['z_array'] = z_array
nz_info_dict['nbins'] = 4
nz_info_dict['nz0'] = np.maximum(df['nz_source'].data['BIN1'], 1e-4)
nz_info_dict['nz1'] = np.maximum(df['nz_source'].data['BIN2'], 1e-4)
nz_info_dict['nz2'] = np.maximum(df['nz_source'].data['BIN3'], 1e-4)
nz_info_dict['nz3'] = np.maximum(df['nz_source'].data['BIN4'], 1e-4)
analysis_dict = {}
analysis_dict['nz_info_dict'] = nz_info_dict
analysis_dict['do_sheary'] = True
analysis_dict['do_shear2pt'] = True
analysis_dict['do_yy'] = False

df_data = fits.open('/mnt/home/spandey/ceph/GODMAX/data/DES_ACT_full_data_theorycov_2.5.fits')
theta_data = df_data['compton_shear'].data['ANG'][0:20]

analysis_dict['ellmin_transf'], analysis_dict['ellmax_transf'], analysis_dict['nell_transf'] = 8, 2**15, 16384
analysis_dict['angles_data_array'] = jnp.array(theta_data)
analysis_dict['beam_fwhm_arcmin'] = 1.6
analysis_dict['want_like_diff'] = False
analysis_dict['calc_nfw_only'] = True


from get_BCMP_profile_NO_CONC_jit import BCM_18_wP_NO_CONC
BCMP_test = BCM_18_wP_NO_CONC(sim_params_dict, halo_params_dict)


# for snap_num in snap_num_all[1:]:
z2chi_file = '/mnt/ceph/users/wcoulton/nbodysims/products/fiducial/100/z2chi_interp.txt'
z2chi = np.loadtxt(z2chi_file)
# z2chi.shape

zlist_file = '/mnt/ceph/users/wcoulton/nbodysims/products/fiducial/100/zlist.txt'
zlist = np.loadtxt(zlist_file)
snap_num_all = zlist[:,0].astype(int)
zval_all = zlist[:,1]

zmax_ymap = 3.0
sim_number = 100
# snap_num = 90
print(snap_num_all[1:-1])
for snap_num in snap_num_all[1:-1]:
    ind_snapnum = np.where(snap_num_all == snap_num)[0]
    zval = zval_all[ind_snapnum]
    if zval < zmax_ymap:
        sdir = '/mnt/home/spandey/ceph/GODMAX/notebooks/mock_gen/ymap_will_sims/'
        save_ymap_fname = sdir + f'ymap_sim_{sim_number}_test_snap_{snap_num}.pkl'
        if not os.path.exists(save_ymap_fname):
            
            print(snap_num, zval)
            ldir = f'/mnt/home/wcoulton/ceph/nbodysims/products/fiducial/{sim_number}/halos/{snap_num}/'
            # get all the files in this directory:
            files_all = os.listdir(ldir)
            def open_data(file):
                df = h5.File(ldir+file, 'r')
                M200c = df['M200c'][()]
                X, Y, Z = df['X'][()], df['Y'][()], df['Z'][()]
                if (M200c.shape) is not None:
                    indsel = np.where(M200c>1e12)[0]
                    X_val = X[indsel]
                    Y_val = Y[indsel]
                    Z_val = Z[indsel]
                    M200c_val = M200c[indsel]
                else:
                    X_val = np.array([])
                    Y_val = np.array([])
                    Z_val = np.array([])
                    M200c_val = np.array([])
                df.close()
                return (X_val, Y_val, Z_val, M200c_val)

            def concatenate_data(results):
                # Pre-compute lengths for each part
                lengths = np.array([len(result[0]) for result in results])
                total_length = lengths.sum()

                # Pre-allocate arrays
                X_all = np.empty(total_length)
                Y_all = np.empty(total_length)
                Z_all = np.empty(total_length)
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
                    M200_all[start:end] = result[3]

                return X_all, Y_all, Z_all, M200_all

            with Pool(cpu_count()) as pool:
                results = pool.map(open_data, files_all)


            X_all, Y_all, Z_all, M200c_all = concatenate_data(results)

            halo_ra, halo_dec = hp.vec2ang(np.array([X_all, Y_all, Z_all]).T, lonlat=True)
            halo_z = zval * np.ones_like(halo_ra)
            halo_m = M200c_all
            print('number of halos: ', len(halo_m))
            M_all = halo_m
            ra_all = halo_ra
            dec_all = halo_dec
            z_all = halo_z
            vlos_all = np.zeros_like(z_all)
            nsel = len(M_all)

            nh_max = 2e6
            if nsel > nh_max:
                num_chunks = int(np.ceil(nsel / nh_max))
            else:
                num_chunks = 1

            nside = 1024
            ymap_test = np.zeros(12*nside**2)
            for i in range(num_chunks):
                print(f'chunk {i+1}/{num_chunks}')
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
                mock_params_dict['halo_z'] = jnp.array(z_all_chunk)
                mock_params_dict['halo_ra'] = jnp.array(ra_all_chunk)
                mock_params_dict['halo_dec'] = jnp.array(dec_all_chunk)
                mock_params_dict['halo_M'] = jnp.array(M_all_chunk)
                mock_params_dict['halo_vlos'] = jnp.array(vlos_all_chunk)
                mock_params_dict['nside'] = nside

                halo_cat_scale_fac = 1./(1. + z_all_chunk)

                halo_cat_rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(BCMP_test.cosmo_jax,halo_cat_scale_fac) * 1e9
                mdef_delta=200
                halo_cat_rho_treshold = mdef_delta * halo_cat_rho_c_z


                halo_cat_R200c = (M_all_chunk * 3.0 / 4.0 / jnp.pi / halo_cat_rho_treshold)**(1.0 / 3.0)
                halo_cat_DA = bkgrd.angular_diameter_distance(BCMP_test.cosmo_jax,halo_cat_scale_fac)
                max_paint_R200c_factor = 3.


                ra_all_np = np.clip(np.array(ra_all_chunk), 0., 360.)
                dec_all_np = np.clip(np.array(dec_all_chunk), -90., 90.)
                z_all_np = np.array(z_all_chunk)
                halo_cat_R200c_np = np.array(halo_cat_R200c)
                halo_cat_DV_np = np.array(halo_cat_DA)
                def process_halo(jhalo):
                    vec = hp.ang2vec(ra_all_np[jhalo], dec_all_np[jhalo], lonlat=True)

                    nearby_angle = max_paint_R200c_factor * halo_cat_R200c_np[jhalo] / halo_cat_DV_np[jhalo]
                    nearby_pix = hp.query_disc(mock_params_dict['nside'], vec, nearby_angle)

                    nearby_ra, nearby_dec = hp.pix2ang(mock_params_dict['nside'], nearby_pix, lonlat=True)

                    def hav(theta):
                        return np.sin(theta / 2.) ** 2.

                    ra1, dec1 = ra_all_np[jhalo] * np.pi / 180., dec_all_np[jhalo] * np.pi / 180.
                    ra2, dec2 = nearby_ra * np.pi / 180., nearby_dec * np.pi / 180.
                    theta = 2. * np.arcsin(np.sqrt(hav(dec1 - dec2) + np.cos(dec1) * np.cos(dec2) * hav(ra1 - ra2)))

                    physical_distances_jhalo = halo_cat_DV_np[jhalo] * theta

                    nearby_pix_array = np.array(nearby_pix)
                    distances_pix_array = np.array(physical_distances_jhalo)
                    logM_ind_array = np.ones(len(nearby_pix)) * np.log(M_all_chunk[jhalo])
                    z_ind_array = np.ones(len(nearby_pix)) * z_all_chunk[jhalo]

                    return (nearby_pix_array, distances_pix_array, logM_ind_array, z_ind_array, len(nearby_pix))

                def concatenate_results(results):
                    # Pre-compute lengths for each part
                    lengths = np.array([len(result[0]) for result in results])
                    total_length = lengths.sum()

                    # Pre-allocate arrays
                    nearby_pix_all = np.empty(total_length, dtype=int)
                    distances_pix_all = np.empty(total_length)
                    logM_ind_all = np.empty(total_length)
                    z_ind_all = np.empty(total_length)

                    # Calculate start and end indices
                    end_ind_all = np.cumsum(lengths)
                    start_ind_all = np.roll(end_ind_all, 1)
                    start_ind_all[0] = 0

                    # Use array slicing for assignment
                    for i, (start, end, result) in enumerate(zip(start_ind_all, end_ind_all, results)):
                        nearby_pix_all[start:end] = result[0]
                        distances_pix_all[start:end] = result[1]
                        logM_ind_all[start:end] = result[2]
                        z_ind_all[start:end] = result[3]

                    return nearby_pix_all, distances_pix_all, start_ind_all, end_ind_all, logM_ind_all, z_ind_all


                # if __name__ == "__main__":
                with Pool(cpu_count()) as pool:
                    results = pool.map(process_halo, range(len(z_all_chunk)))


                nearby_pix_all, distances_pix_all, start_ind_all, end_ind_all, logM_ind_all, z_ind_all = concatenate_results(results)

                mock_params_dict['nearby_pix_all'] = jnp.array(nearby_pix_all)
                mock_params_dict['pix_prop_all'] = jnp.array([np.log(distances_pix_all), z_ind_all, logM_ind_all]).T
                mock_params_dict['start_ind'] = jnp.int32(jnp.array(start_ind_all))
                mock_params_dict['end_ind'] = jnp.int32(jnp.array(end_ind_all))

                mock_map_test = get_mock_map(sim_params_dict, halo_params_dict, mock_params_dict, BCMP_obj=BCMP_test)

                # find non-finite values and set them to zero
                ymap_test += jnp.nan_to_num(mock_map_test.ymap_final)
                jax.clear_caches()
                


            saved = {}
            saved['ymap_test'] = ymap_test
            pk.dump(saved,open(save_ymap_fname, 'wb'))


