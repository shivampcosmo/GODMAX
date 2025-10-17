import pickle as pk
import numpy as np
import os
import sys, os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from jax.lib import xla_bridge
platform = xla_bridge.get_backend().platform
import jax
import jax.numpy as jnp
from jax import vmap, grad, pmap
print(jax.local_device_count(), jax.device_count())
jax.config.update('jax_platform_name', platform)
jax.config.update("jax_enable_x64", True)

import pathlib
curr_path = pathlib.Path().absolute()
abs_path_data = os.path.abspath(curr_path / "../../data/") 
abs_path_src = os.path.abspath(curr_path / "../../src/") 
abs_path_results = os.path.abspath(curr_path / "../../results/") 
abs_path_params = os.path.abspath(curr_path / "../../param_files/") 
sys.path.append((curr_path))
sys.path.append((abs_path_data))
sys.path.append((abs_path_results))
sys.path.append(abs_path_src)

from jax import config
import scipy.interpolate as interp
import pickle as pk
import numpy as np
import colossus 
import configobj

from base_class import base_class
from get_radial_profiles import Profiles
from get_Pkzs import get_Pkz
from get_Cls import get_Cl
from get_Xis import get_xi
from get_covs import get_cov
import warnings
warnings.filterwarnings('ignore')

import yaml
from deepmerge import always_merger
from jax_cosmo import Cosmology
from jax_cosmo.background import angular_diameter_distance, radial_comoving_distance
from astropy import constants as const
import jax_cosmo.background as bkgrd

import ast
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Training script with key-value arguments')
    
    # Define arguments with their default values and types
    parser.add_argument('--probes', type=str, default="ky,kk,gg,gy,gk,ge", 
                        help='Probes to forecast')
    parser.add_argument('--lmax', type=int, default=8000,
                        help='Maximum ell value for the probes')
    parser.add_argument('--num_warmup', type=int, default=6000,
                        help='Number of warmup iterations for MCMC')
    parser.add_argument('--num_samples', type=int, default=6000,
                        help='Number of samples for MCMC')
    parser.add_argument('--num_chains', type=int, default=24,
                        help='Number of chains for MCMC')
    parser.add_argument('--max_tree_depth', type=int, default=4,
                        help='Maximum tree depth for NUTS sampler')
    parser.add_argument('--bao_prior', type=bool, default=False,
                        help='Use BAO prior')
    parser.add_argument('--init_strategy', type=str, default="median",
                        help='Initialization strategy for sampler')    
    args = parser.parse_args()
    return args




args = parse_args()
probes_forecast = args.probes
probes_forecast = probes_forecast.split(',')
probes_forecast_all_str = '_'.join(probes_forecast)
sc_val = args.lmax
num_warmup = args.num_warmup
num_samples = args.num_samples
num_chains= args.num_chains
n_parallel = 4
max_tree_depth = args.max_tree_depth
wbao_prior = args.bao_prior
init_strategy = args.init_strategy

PROBE_CONFIGS = {
     "ky_kk_gg_gy_gk_ge":{
        # "probe":,
        "output_key": "FoM_all_probes_all",
        "acorr_logic": lambda lmax: {},  # No acorr params for this probe
        "file_logic": lambda lmax: (6000, 9000) if lmax == 8000 else (6000, 6000),
    },
    "kk_gg_gk_ge":{
        # "probe": 
        "output_key": "FoM_all_notSZ",
        # Pass acorr params ONLY when lmax is 8000
        "acorr_logic": lambda lmax: {"acorr_min": 0.05, "acorr_max": 0.1} if lmax == 8000 else {},
        "file_logic": lambda lmax: (6000, 9000) if lmax == 8000 else (4000, 3000),
    },
    "kk_gg_gk":{
        # "probe": "kk_gg_gk",
        "output_key": "FoM_all_notSZ_nokSZ",
        # Pass acorr params ONLY when lmax is less than 8000
        "acorr_logic": lambda lmax: {"acorr_min": 0.05, "acorr_max": 0.1} if lmax < 8000 else {},
        "file_logic": lambda lmax: (4000, 3000),
    },
    "gg_gk_ge":{
        # "probe": "gg_gk_ge",
        "output_key": "FoM_all_noshear",
        # Pass acorr params for all lmax values
        "acorr_logic": lambda lmax: {"acorr_min": 0.06, "acorr_max": 0.1},
        "file_logic": lambda lmax: (6000, 6000) if lmax > 1000 else (4000, 3000),
    },
    "gg_gk":{
        # "probe": "gg_gk",
        "output_key": "FoM_all_noshear_nokSZ",
        # Pass acorr params for all lmax values
        "acorr_logic": lambda lmax: {"acorr_min": 0.06, "acorr_max": 0.1},
        "file_logic": lambda lmax: (4000, 3000),
    },
}

dict_probe = PROBE_CONFIGS[probes_forecast_all_str]
acorr_params = dict_probe["acorr_logic"](sc_val)

def get_samps(fname, nchains = 96, acorr_max = 0.075, acorr_min = 0.0075):
    df = pk.load(open(fname,'rb'))
    sig8 = df['sigma8']
    
    sig8_rs = sig8.reshape(nchains,-1)
    # print(sig8_rs.shape)
    from numpyro.diagnostics import autocorrelation, autocovariance
    
    acorr = autocorrelation(sig8_rs, axis=1)
    ind_del = []
    for ji in range(acorr.shape[0]):
        if (np.std(acorr[ji,100:]) > acorr_max) or (np.std(acorr[ji,100:]) < acorr_min):
            ind_del.append(ji)    
    print(len(ind_del))
    samps = []
    keys = []
    for key in df:
        if ('base' not in key) and ('decentered' not in key):
            if key not in ['RUN_SETTINGS', 'diverging', 'potential_energy']:
            # if key not in ['prior_min', 'prior_max', 'fiducial_sims_params', 'fiducial_other_params', 'fiducial_halo_params', 'fiducial_analysis_params', 'diverging', 'potential_energy']:
                if ('Delta_z_bias_array' in key) or ('mult_shear_bias_array' in key):
                    for jb in range(4):
                        samps.append(df[key][:, jb])
                        keys.append(key + '_' + str(jb))
                        # print(df[key][:, jb].shape)
                # print(samps[0].shape)
                # print(df[key].shape)
                else:
                    samps.append(df[key])
                    keys.append(key)
    
    samps = np.array(samps).T
    ind_sigma8 = keys.index('sigma8')
    # ind_thetaejM = keys.index('nu_theta_ej_M')
    ind_Om = keys.index('Om0')
    samp_S8 = samps[:,ind_sigma8] * (samps[:,ind_Om]/0.3)**0.5
    
    samps = np.concatenate([samps, samp_S8[:,None]], axis=1)
    keys.append('S8')
    
    import getdist
    from getdist import plots, MCSamples
    
    names = keys    
    ind_frac_rm = 0.5
    # nchains = 80
    samps_sel = samps.reshape(nchains,-1, samps.shape[-1])
    nsamp_per_chain = samps_sel.shape[1]
    nrm = int(ind_frac_rm * nsamp_per_chain)
    samps_sel = samps_sel[:, nrm:, :]
    
    if len(ind_del) > 0:
        samps_sel = np.delete(samps_sel, ind_del, axis=0)
    else:
        samps_sel = samps_sel
    
    samps = samps_sel.reshape(-1, samps_sel.shape[-1])
    print(samps.shape, samps_sel.shape)
    return samps, keys



def get_dicts_default():
    
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
    
    default_data = read_yaml(abs_path_params + '/params_default.yaml')
    # sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(default_data)
    new_data = read_yaml(abs_path_params + '/Pge/params.yaml')
    # new_data = read_yaml(abs_path_params + '/DESxACT/params_v0.yaml')
    merged_data = always_merger.merge(default_data, new_data)
    
    sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(merged_data)
    
    # analysis_dict['beam_fwhm_arcmin'] = 1.4
    df_nz_comoving = np.loadtxt(abs_path_data + '/pge/comoving_nz_bgs_lrg_DESIY3_key_zall.txt')
    zarray_comoving = df_nz_comoving[:,0]
    nz_comoving_orig = df_nz_comoving[:,1]
    indsel = np.where(zarray_comoving < 0.4)[0]
    nz_comoving = np.zeros_like(zarray_comoving)
    nz_comoving[indsel] = nz_comoving_orig[indsel]*1.0
    indsel = np.where(zarray_comoving > 0.4)[0]
    nz_comoving[indsel] = nz_comoving_orig[indsel]*1.0
    analysis_dict['nbar_gal_comoving_zarray'] = zarray_comoving
    analysis_dict['nbar_gal_comoving_val'] = nz_comoving
    
    
    
    from scipy.interpolate import interp1d
    # ks = np.geomspace(5e-2,50,15) # wavenumbers
    ks = np.geomspace(3e-1,10,10) # wavenumbers
    zedges = np.array([0.1, 0.4, 0.6, 0.8,1.1])
    zarray_lens = np.linspace(0.001, 1.6, 100)
    nbins_lens = len(zedges) - 1
    
    cosmo_params = sim_params_dict.get('cosmo')
    cosmo_jax = Cosmology(
                Omega_c=cosmo_params['Om0'] - cosmo_params['Ob0'],
                Omega_b=cosmo_params['Ob0'],
                h=cosmo_params['H0'] / 100.,
                sigma8=cosmo_params['sigma8'],
                n_s=cosmo_params['ns'],
                Omega_k=0.,
                w0=cosmo_params['w0'],
                wa=0.
                )
    scale_fac_a_array = 1.0 / (1.0 + zarray_lens)
    chi_array = radial_comoving_distance(cosmo_jax, scale_fac_a_array)
    dchi_dz_array = (const.c.value * 1e-3) / bkgrd.H(cosmo_jax, scale_fac_a_array)
    nz_comoving_interp = interp1d(zarray_comoving, nz_comoving, fill_value=1e-20, bounds_error=False)
    nz_comoving_zarray = nz_comoving_interp(zarray_lens)
    nz_zarray = nz_comoving_zarray * (chi_array**2 * dchi_dz_array)
    nz_zarray = nz_zarray/(np.trapz(nz_zarray, zarray_lens))
    nz_lens = {}
    for jb in range(nbins_lens):
        nz_jb = np.zeros_like(nz_zarray)
        indsel = np.where((zarray_lens > zedges[jb]) & (zarray_lens < zedges[jb+1]))[0]
        nz_jb[indsel] = nz_zarray[indsel]
        norm_val = np.trapz(nz_jb, zarray_lens)
        nz_jb = nz_jb/norm_val
        nz_lens[jb] = nz_jb
    
    nz_info_dict = {}
    nz_info_dict['z_array_lens'] = zarray_lens
    nz_info_dict['nbins_lens'] = nbins_lens
    for ji in range(nz_info_dict['nbins_lens']):
        nz_info_dict['nz'+str(ji)] = np.maximum(nz_lens[ji], 1e-3)
    analysis_dict['nz_lens_info_dict'] = nz_info_dict
    
    
    from astropy.io import fits
    df = fits.open(os.path.abspath(abs_path_data + '/forecast/lsst_simulate_Y1.fits'))
    z_array = df['nz_source'].data['Z_MID']
    nz_info_dict = {}
    nz_info_dict['z_array_source'] = z_array
    nz_info_dict['nbins'] = 5
    for ji in range(nz_info_dict['nbins']):
        nz_info_dict['nz'+str(ji)] = np.maximum(df['nz_source'].data['BIN'+str(ji+1)], 1e-4)
    analysis_dict['nz_source_info_dict'] = nz_info_dict
    other_params_dict['Delta_z_bias_array'] = np.zeros(analysis_dict['nz_source_info_dict']['nbins'])
    other_params_dict['mult_shear_bias_array'] = np.zeros(analysis_dict['nz_source_info_dict']['nbins'])
    
    
    analysis_dict['angles_data_array'] = df['xip'].data['ANG'][0:20]
    
    
    analysis_dict['k_array_survey'] = jnp.array(ks / (sim_params_dict['cosmo']['H0']/100.))
    
    lmin, lmax, dl_log_array = 80.0, 8800.0, 0.23025851
    l_array_all = np.exp(np.arange(np.log(lmin), np.log(lmax), dl_log_array))
    dl_array = l_array_all[1:] - l_array_all[:-1]
    l_array_survey = (l_array_all[1:] + l_array_all[:-1]) / 2.
    halo_params_dict['ell_array'] = jnp.array(l_array_survey)
    analysis_dict['l_array_survey'] = jnp.array(l_array_survey)
    analysis_dict['dl_array_survey'] = jnp.array(dl_array)
    analysis_dict['yy_noise_ell_fname'] = os.path.abspath(abs_path_data + '/pge/Noise_fid_yy_beamed_1p4arcmin.txt')
    
    Ngals_bins = []
    Vol_Gpc3 = []
    z_cens = []
    P0 = 9e3
    fsky_DESIxDESI = analysis_dict['fsky_gg']
    for jb in range(nbins_lens):
        nz_jb_only = np.copy(nz_comoving_zarray)
        indsel = np.where((zarray_lens < zedges[jb]) | (zarray_lens > zedges[jb+1]))[0]
        nz_jb_only[indsel] = 0.0
        nz_integrate = np.trapz(nz_jb_only*(chi_array**2)*dchi_dz_array, zarray_lens)
    
        ntot_jb = nz_integrate*4*np.pi*fsky_DESIxDESI
        Ngals_bins.append(ntot_jb)
    
        zmean = np.trapz(zarray_lens*nz_jb_only*(chi_array**2)*dchi_dz_array, zarray_lens)/nz_integrate
        z_cens.append(zmean)
    
        fsky_nz = np.ones_like(zarray_lens)
        fsky_nz[indsel] = 0.0
        fsky_nz_desi = fsky_nz * (nz_jb_only*P0/(1 + nz_jb_only*P0))**2
        h3 = (sim_params_dict['cosmo']['H0'] / 100.0)**3
        Vol_Gpc3_comoving = 4*np.pi*fsky_DESIxDESI*np.trapz(fsky_nz * (chi_array**2)*dchi_dz_array, zarray_lens)/(h3 * 1e9)   
        Vol_Gpc3_comoving_desi = 4*np.pi*fsky_DESIxDESI*np.trapz(fsky_nz_desi * (chi_array**2)*dchi_dz_array, zarray_lens)/(h3 * 1e9)   
    
        Vol_Gpc3.append(Vol_Gpc3_comoving_desi)
        # print(zedges[jb], zedges[jb+1], 'Total galaxies = {:.2f}'.format(ntot_jb), ' Vol:', Vol_Gpc3_comoving_desi)
        # print('Bin {}: Total galaxies = {:.2f}'.format(jb, ntot_jb))
        
    
    Ngals_bins, Vol_Gpc3, z_cens = np.array(Ngals_bins), np.array(Vol_Gpc3), np.array(z_cens)
        
        # Ngals_bins = 3.33*np.array([506905, 771875, 859824])
    analysis_dict['nbar_lens_bins'] = Ngals_bins/(analysis_dict['fsky_gg']*41253*(60**2))
    
    analysis_dict['symbolic_pk'] = True
    analysis_dict['symbolic_hmf'] = True

    return sim_params_dict, halo_params_dict, analysis_dict, other_params_dict

# init_strategy = 

save_chain_dir = abs_path_results + '/pge/chains_july_v5/'
if wbao_prior:
    savefname_out = save_chain_dir + f'{probes_forecast_all_str}/mcmc_v5_{probes_forecast_all_str}_scval_{sc_val}_samples_{num_samples}_warmup_{num_warmup}_num_chains_{num_chains*n_parallel}_treedepth_{max_tree_depth}_wbaoprior_{wbao_prior}.pkl'
else:
    savefname_out = save_chain_dir + f'{probes_forecast_all_str}/mcmc_v5_{probes_forecast_all_str}_scval_{sc_val}_samples_{num_samples}_warmup_{num_warmup}_num_chains_{num_chains*n_parallel}_treedepth_{max_tree_depth}.pkl'


samps, keys = get_samps(savefname_out, nchains=num_chains, **acorr_params)

nsamp_tot = samps.shape[0]
# indsel = np.sort(np.random.randint(0, nsamp_tot, 64))
# indsel = np.sort(np.random.randint(0, nsamp_tot, 8))
indsel = np.sort(np.random.randint(0, nsamp_tot, 512))


from tqdm import tqdm

# ind_post = keys.index('potential_energy')
# post = -1.* samps[:,ind_post]

# map_ind = np.where(post == post.max())[0][0]

# Y_model_all_samp = []
Pmm_ratio_all_samp = []

for map_ind in tqdm(indsel):
    saved_bestfit = {}
    for jk, key in enumerate(keys):
        # print(key, samps[map_ind, jk])
        saved_bestfit[key] = samps[map_ind, jk]
    
    # potential_energy -4916.635080802822
    import yaml
    import pathlib
    curr_path = pathlib.Path().absolute()
    abs_path_data = os.path.abspath(curr_path / "../../data/") 
    abs_path_src = os.path.abspath(curr_path / "../../src/") 
    abs_path_results = os.path.abspath(curr_path / "../../results/") 
    abs_path_params = os.path.abspath(curr_path / "../../param_files/") 
    

    sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = get_dicts_default()
    
    
    for key in saved_bestfit.keys():
        if key in list(sim_params_dict.keys()):
            sim_params_dict[key] = float(saved_bestfit[key])
        if key in list(sim_params_dict['cosmo'].keys()):
            sim_params_dict['cosmo'][key] = float(saved_bestfit[key])
            if key == 'h':
                sim_params_dict['cosmo']['H0'] = 100*saved_bestfit[key]
        if key in list(other_params_dict.keys()):
            other_params_dict[key] = float(saved_bestfit[key])
    for jb in range(analysis_dict['nz_source_info_dict']['nbins'] - 1):
        other_params_dict['Delta_z_bias_array'][jb] = saved_bestfit[f'Delta_z_bias_array_{jb}']
    for jb in range(analysis_dict['nz_source_info_dict']['nbins'] - 1):
        other_params_dict['mult_shear_bias_array'][jb] = saved_bestfit[f'mult_shear_bias_array_{jb}']

    # halo_params_dict['rmax'] = 12.0
    # halo_params_dict['nr'] = 128
    # halo_params_dict['nk'] = 96
    base_test = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
    profiles_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, base_class_obj=base_test)
    Pkz_test = get_Pkz(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Profiles_obj=profiles_test)
    ratio = Pkz_test.Pmm_dmb_tot_mat[:,0]/Pkz_test.Pmm_nfw_tot_mat[:,0]    
    # Pmm_ratio_all_samp.append(savitzky_golay(ratio, 19, 3))
    # print(ratio)
    Pmm_ratio_all_samp.append(ratio)


Pmm_ratio_all_samp = np.array(Pmm_ratio_all_samp)
# Pmm_ratio_all_samp_mean = np.mean(Pmm_ratio_all_samp, axis=0)
# Pmm_ratio_all_samp_std = np.std(Pmm_ratio_all_samp, axis=0)
Pmin = np.percentile(Pmm_ratio_all_samp, 16, axis=0)
Pmax = np.percentile(Pmm_ratio_all_samp, 84, axis=0)

