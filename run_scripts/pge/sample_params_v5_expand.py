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
jax.config.update("jax_debug_nans", True)
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

import numpyro
numpyro.set_platform("gpu")
numpyro.enable_x64()
numpyro.set_host_device_count(jax.device_count())
from numpyro.handlers import seed, trace, condition
from numpyro.infer.reparam import LocScaleReparam, TransformReparam
from numpyro.infer import HMC, HMCECS, MCMC, NUTS, SA, SVI, Trace_ELBO, init_to_value
from numpyro.distributions.transforms import AffineTransform
import numpyro.distributions as dist

from jax import config
import scipy.interpolate as interp
import pickle as pk
import numpy as np
import colossus 
import configobj
import copy
import yaml
from deepmerge import always_merger
import ast 

from base_class import base_class
from get_radial_profiles import Profiles
from get_Pkzs import get_Pkz
from get_Cls import get_Cl
from get_Xis import get_xi
from get_covs import get_cov
import jax_cosmo.background as bkgrd
from jax_cosmo import Cosmology
from jax_cosmo.background import angular_diameter_distance, radial_comoving_distance
from astropy import constants as const

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
sc_val = args.lmax
num_warmup = args.num_warmup
num_samples = args.num_samples
num_chains= args.num_chains
max_tree_depth = args.max_tree_depth
wbao_prior = args.bao_prior
init_strategy = args.init_strategy
run_this_script = True

n_parallel = jax.local_device_count()

print(f'Running with probes: {probes_forecast}, sc_val: {sc_val}, num_warmup: {num_warmup}, num_samples: {num_samples}, num_chains: {num_chains}, max_tree_depth: {max_tree_depth}, wbao_prior: {wbao_prior}, init_strategy: {init_strategy}')

# print(sys.argv)
# probes_forecast = list(sys.argv[1])
# probes_forecast = sys.argv[1].split(',')
# sc_val = int(sys.argv[2])


# num_warmup = 4000
# num_samples = 10000

# num_warmup = 3000
# num_samples = 4000

# num_warmup = 6000
# num_samples = 6000

# num_chains= 24
# max_tree_depth = 4


save_chain_dir = abs_path_results + '/pge/chains_july_v5/'
probes_forecast_all_str = '_'.join(probes_forecast)

# check if the directory exists and if not then create it:
if not os.path.exists(save_chain_dir + f'{probes_forecast_all_str}/'):
    os.makedirs(save_chain_dir + f'{probes_forecast_all_str}/')


savefname_out = save_chain_dir + f'{probes_forecast_all_str}/mcmc_v5_{probes_forecast_all_str}_scval_{sc_val}_samples_{num_samples}_warmup_{num_warmup}_num_chains_{num_chains*n_parallel}_treedepth_{max_tree_depth}_wbaoprior_{wbao_prior}.pkl'
# if wbao_prior:
#     savefname_out = save_chain_dir + f'{probes_forecast_all_str}/mcmc_v5_{probes_forecast_all_str}_scval_{sc_val}_samples_{num_samples}_warmup_{num_warmup}_num_chains_{num_chains*n_parallel}_treedepth_{max_tree_depth}_wbaoprior_{wbao_prior}.pkl'
# else:
#     savefname_out = save_chain_dir + f'{probes_forecast_all_str}/mcmc_v5_{probes_forecast_all_str}_scval_{sc_val}_samples_{num_samples}_warmup_{num_warmup}_num_chains_{num_chains*n_parallel}_treedepth_{max_tree_depth}.pkl'

print(probes_forecast, sc_val, savefname_out)
run_this_script = True
# # check if the file exists:
# if os.path.exists(savefname_out):
#     df = pk.load(open(savefname_out, 'rb'))
#     print(df.keys())
#     run_this_script = False
#     print('File already exists, skipping the run.')

if run_this_script:
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
    nz_info_dict_s = {}
    nz_info_dict_s['z_array_source'] = z_array
    nz_info_dict_s['nbins'] = 5
    for ji in range(nz_info_dict_s['nbins']):
        nz_info_dict_s['nz'+str(ji)] = np.maximum(df['nz_source'].data['BIN'+str(ji+1)], 1e-4)
    analysis_dict['nz_source_info_dict'] = nz_info_dict_s


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
        print(zedges[jb], zedges[jb+1], 'Total galaxies = {:.2f}'.format(ntot_jb), ' Vol:', Vol_Gpc3_comoving_desi)
        # print('Bin {}: Total galaxies = {:.2f}'.format(jb, ntot_jb))
        

    Ngals_bins, Vol_Gpc3, z_cens = np.array(Ngals_bins), np.array(Vol_Gpc3), np.array(z_cens)
        
        # Ngals_bins = 3.33*np.array([506905, 771875, 859824])
    analysis_dict['nbar_lens_bins'] = Ngals_bins/(analysis_dict['fsky_gg']*41253*(60**2))

    analysis_dict['symbolic_pk'] = True
    analysis_dict['symbolic_hmf'] = True



    import pickle as pk

    # df = pk.load(open(abs_path_data + '/pge/DV_fid_Cl_Pk_cov_mthresh_hod_v3.pk', 'rb'))
    df = pk.load(open(abs_path_data + '/pge/DV_fid_Cl_Pk_cov_mthresh_hod_v5_symb_dataconsist.pk', 'rb'))
    probes_all = df['probes']
    ell_edges_left = df['edges_left']
    ell_edges_right = df['edges_right']
    ell_array = df['ell_array']
    k_array = df['k_array']

    bin_comb_all_wprobe = df['bin_comb_all_wprobe']

    cov_total_orig = df['cov_mat']
    Cl_total = df['Cl_Pk']

    cov_total_orig_scaled = df['cov_mat_scaled']
    Cl_total_scaled = df['dv_scaled']
    scale_vec = jnp.array(df['scale_vec'])



    import copy
    from tqdm import tqdm
    # from get_power_spectra import get_power_BCMP
    get_power_BCMP_test = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)

    nell = len(ell_array)
    nk = len(analysis_dict['k_array_survey'])
    Cl_all = jnp.zeros((cov_total_orig.shape[0]))
    for jp1 in range(len(bin_comb_all_wprobe)):
        binwprobe1 = bin_comb_all_wprobe[jp1]
        probe1 = binwprobe1[0]
        bin_comb1 = binwprobe1[1]
        b1, b2 = bin_comb1[0], bin_comb1[1]
        if probe1 == 'ky':
            Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_y_tot_mat[:,b1-1])
        if probe1 == 'kk':
            Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_kappa_tot_mat[:,b1-1,b2-1])
        if probe1 == 'gg':
            Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_gal_tot_mat[:,b1-1,b2-1])
        if probe1 == 'gk':
            Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_kappa_tot_mat[:,b1-1,b2-1])
        if probe1 == 'gy':
            Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_y_tot_mat[:,b1-1])

    edge_left_pge = ell_edges_left[-1]
    Cl_all = Cl_all.at[edge_left_pge:edge_left_pge+nk].set(get_power_BCMP_test.Pge_tot_mat[0,:])
    Cl_all = Cl_all.at[edge_left_pge+nk:edge_left_pge+2*nk].set(get_power_BCMP_test.Pge_tot_mat[1,:])
    Cl_all = Cl_all.at[edge_left_pge+2*nk:edge_left_pge+3*nk].set(get_power_BCMP_test.Pge_tot_mat[2,:])
    Cl_all = Cl_all.at[edge_left_pge+3*nk:edge_left_pge+4*nk].set(get_power_BCMP_test.Pge_tot_mat[3,:])

    Cl_all = Cl_all * scale_vec



    ell_min_max_dict = {'ky':[0, 11000], 'kk':[0, 11000], 'gg':[0, 11000], 'gk':[0, 11000], 'ge':[0, 11000]}

    ell_sel_min = []
    ell_sel_max = []
    for jp in range(len(probes_forecast)):
        indp = probes_all.index(probes_forecast[jp])
        # print(indp)
        ell_sel_min.append(ell_edges_left[indp])
        ell_sel_max.append(ell_edges_right[indp])
        print(probes_forecast[jp], ell_sel_min[jp], ell_sel_max[jp])

    ell_sel_min = np.array(ell_sel_min)
    ell_sel_max = np.array(ell_sel_max)
    argsort_min = np.argsort(ell_sel_min)
    probes_forecast = [probes_forecast[jp] for jp in argsort_min]
    ell_sel_min = ell_sel_min[argsort_min]
    ell_sel_max = ell_sel_max[argsort_min]


    import yaml

    fname = abs_path_data + f'/pge/scale_cuts_all_probes_ellmax_{sc_val}.yaml'
    with open(fname, 'r') as stream:
        sc_all = yaml.load(stream, Loader=yaml.SafeLoader)

    ell_sel_all = []
    # ell_count = 0
    for jp in range(len(probes_forecast)):
        probe = probes_forecast[jp]
        sc_all_jp = sc_all[probe]
        ell_min_jp = ell_sel_min[jp]
        if probe != 'ge':
            for jsc in range(len(sc_all_jp)):
                key = list(sc_all_jp.keys())[jsc]
                scmin_jp_jsc = sc_all_jp[key][0]
                scmax_jp_jsc = sc_all_jp[key][1]
                indsel = np.where((ell_array >= scmin_jp_jsc) & (ell_array <= scmax_jp_jsc))[0]
                if len(indsel) > 0:
                    ell_sel_all.append(np.arange(ell_min_jp + indsel[0], ell_min_jp + indsel[-1]+1))
                    ell_min_jp += nell           
        else:
            for jsc in range(len(sc_all_jp)):
                key = list(sc_all_jp.keys())[jsc]
                scmin_jp_jsc = sc_all_jp[key][0]
                scmax_jp_jsc = sc_all_jp[key][1]
                indsel = np.where((k_array >= scmin_jp_jsc) & (k_array <= scmax_jp_jsc))[0]
                ell_sel_all.append(np.arange(ell_min_jp + indsel[0], ell_min_jp + indsel[-1]+1))
                ell_min_jp += nk
        ell_count = len(np.concatenate(ell_sel_all))
        print(probe, ell_count)
            # ell_max_jp = ell_sel_max[jp]
        # ell_sel_all.append(np.arange(ell_sel_min[jp], ell_sel_max[jp]))
    ell_sel_all = np.concatenate(ell_sel_all)

    data_vec = Cl_total_scaled[ell_sel_all]
    cov_total = cov_total_orig_scaled[ell_sel_all,:][:, ell_sel_all]
    P_total = jnp.linalg.inv(cov_total)

    if wbao_prior:
        with open(abs_path_params + '/Pge/priors_bao.yaml', 'r') as file:
            data = yaml.safe_load(file)
    else:
        with open(abs_path_params + '/Pge/priors.yaml', 'r') as file:
            data = yaml.safe_load(file)
    prior_limits = {key: tuple(map(float, value.split())) for key, value in data['prior_uniform'].items()}
    prior_gaussian = {key: tuple(map(float, value.split())) for key, value in data['prior_gaussian'].items()}

    prior_min_all_dict, prior_max_all_dict = {}, {}
    for key in prior_limits.keys():
        prior_min_all_dict[key] = prior_limits[key][0]
        prior_max_all_dict[key] = prior_limits[key][1]

    prior_mu_all_dict, prior_sig_all_dict = {}, {}
    for key in prior_gaussian.keys():
        prior_mu_all_dict[key] = prior_gaussian[key][0]
        prior_sig_all_dict[key] = prior_gaussian[key][1]
    prior_delta_z_mu_all = jnp.array([prior_mu_all_dict['Delta_z_bias_bin1'], prior_mu_all_dict['Delta_z_bias_bin2'], prior_mu_all_dict['Delta_z_bias_bin3'], prior_mu_all_dict['Delta_z_bias_bin4'], prior_mu_all_dict['Delta_z_bias_bin5']])
    prior_delta_z_sig_all = jnp.array([prior_sig_all_dict['Delta_z_bias_bin1'], prior_sig_all_dict['Delta_z_bias_bin2'], prior_sig_all_dict['Delta_z_bias_bin3'], prior_sig_all_dict['Delta_z_bias_bin4'], prior_sig_all_dict['Delta_z_bias_bin5']])
    prior_mult_shear_mu_all = jnp.array([prior_mu_all_dict['mult_shear_bias_bin1'], prior_mu_all_dict['mult_shear_bias_bin2'], prior_mu_all_dict['mult_shear_bias_bin3'], prior_mu_all_dict['mult_shear_bias_bin4'], prior_mu_all_dict['mult_shear_bias_bin5']])
    prior_mult_shear_sig_all = jnp.array([prior_sig_all_dict['mult_shear_bias_bin1'], prior_sig_all_dict['mult_shear_bias_bin2'], prior_sig_all_dict['mult_shear_bias_bin3'], prior_sig_all_dict['mult_shear_bias_bin4'], prior_sig_all_dict['mult_shear_bias_bin5']])


    cosmo_params_vary_names = ['Om0', 'sigma8', 'Ob0', 'h', 'ns']
    sims_params_vary_names_baryons = ['theta_ej_0', 'theta_co_0','nu_theta_ej_z','nu_theta_ej_M', 'delta_rhogas', 'mu_beta', 'alpha_nt']
    sims_params_vary_names_gals = ['log10M1_fshmr', 'log10M1_a_fshmr','gamma_fshmr','gamma_a_fshmr',  'delta_fshmr', 'delta_a_fshmr', 'siglogMstar_Ncen', 'alphasat_Nsat']
    sims_params_vary_names = sims_params_vary_names_baryons + sims_params_vary_names_gals
    IA_params_vary_names = ['A_IA', 'eta_IA']
    mult_shear_vary_names = ['mult_shear_bias_bin1', 'mult_shear_bias_bin2', 'mult_shear_bias_bin3', 'mult_shear_bias_bin4', 'mult_shear_bias_bin5'] 
    Delta_shear_vary_names = ['Delta_z_bias_bin1', 'Delta_z_bias_bin2', 'Delta_z_bias_bin3', 'Delta_z_bias_bin4', 'Delta_z_bias_bin5']
    analysis_vary_names = []






    from numpyro.distributions.transforms import AffineTransform
    import numpyro.distributions as dist
    import numpyro
    def Uniform(name, min_value, max_value):
        """ Creates a Uniform distribution in target range from a base
        distribution between [-3, 3]
        """
        s = (max_value - min_value) / 6.
        return numpyro.sample(
                name,
                dist.TransformedDistribution(
                    dist.Uniform(-3., 3.),
                    AffineTransform(min_value + 3.*s, s),
                ),
            )


    import jax 
    from tqdm import tqdm
    import copy
    import numpyro
    import copy


    def model():
        sim_params_dict_vary = copy.deepcopy(sim_params_dict)
        other_params_dict_vary = copy.deepcopy(other_params_dict)
        analysis_dict_vary = copy.deepcopy(analysis_dict)
        if len(cosmo_params_vary_names) > 0:
            for jp in range(len(cosmo_params_vary_names)):
                if wbao_prior and cosmo_params_vary_names[jp] == 'Om0':
                    prior_mu = prior_mu_all_dict['Om0']
                    prior_sig = prior_sig_all_dict['Om0']
                    sim_params_dict_vary['cosmo']['Om0'] = numpyro.sample('Om0', dist.Normal(prior_mu, prior_sig))
                else:
                    if cosmo_params_vary_names[jp] == 'h':
                        fac = 100.
                        cosmo_name = 'H0'
                    else:
                        fac = 1.
                        cosmo_name = cosmo_params_vary_names[jp]
                    prior_min_jp = prior_min_all_dict[cosmo_params_vary_names[jp]]
                    prior_max_jp = prior_max_all_dict[cosmo_params_vary_names[jp]]
                    sim_params_dict_vary['cosmo'][cosmo_name] = fac * Uniform(cosmo_params_vary_names[jp], prior_min_jp, prior_max_jp)

        if len(sims_params_vary_names) > 0:
            for jp in range(len(sims_params_vary_names)):
                prior_min_jp = prior_min_all_dict[sims_params_vary_names[jp]]
                prior_max_jp = prior_max_all_dict[sims_params_vary_names[jp]]
                sim_params_dict_vary[sims_params_vary_names[jp]] = Uniform(sims_params_vary_names[jp], prior_min_jp, prior_max_jp)
        
        if len(IA_params_vary_names) > 0:
            for jp in range(len(IA_params_vary_names)):
                prior_min_jp = prior_min_all_dict[IA_params_vary_names[jp]]
                prior_max_jp = prior_max_all_dict[IA_params_vary_names[jp]]
                other_params_dict_vary[IA_params_vary_names[jp]] = Uniform(IA_params_vary_names[jp], prior_min_jp, prior_max_jp)
            
        if len(prior_delta_z_mu_all) > 0:
            Delta_z_bias_array = numpyro.sample('Delta_z_bias_array', dist.Normal(prior_delta_z_mu_all, prior_delta_z_sig_all)) 
            other_params_dict_vary['Delta_z_bias_array'] = Delta_z_bias_array
        
        if len(prior_mult_shear_mu_all) > 0:
            mult_shear_bias_array = numpyro.sample('mult_shear_bias_array', dist.Normal(prior_mult_shear_mu_all, prior_mult_shear_sig_all))
            other_params_dict_vary['mult_shear_bias_array'] = mult_shear_bias_array
        
        if len(analysis_vary_names) > 0:
            for jp in range(len(analysis_vary_names)):
                prior_min_jp = prior_min_all_dict[analysis_vary_names[jp]]
                prior_max_jp = prior_max_all_dict[analysis_vary_names[jp]]
                analysis_dict_vary[analysis_vary_names[jp]] = Uniform(analysis_vary_names[jp], prior_min_jp, prior_max_jp)
        
        get_power_BCMP_test = get_Cl(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict_vary)

        nell = len(ell_array)
        nk = len(analysis_dict['k_array_survey'])
        Cl_all = jnp.zeros((cov_total_orig.shape[0]))
        for jp1 in range(len(bin_comb_all_wprobe)):
            binwprobe1 = bin_comb_all_wprobe[jp1]
            probe1 = binwprobe1[0]
            bin_comb1 = binwprobe1[1]
            b1, b2 = bin_comb1[0], bin_comb1[1]
            if probe1 == 'ky':
                Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_y_tot_mat[:,b1-1])
            if probe1 == 'kk':
                Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_kappa_tot_mat[:,b1-1,b2-1])
            if probe1 == 'gg':
                Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_gal_tot_mat[:,b1-1,b2-1])
            if probe1 == 'gk':
                Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_kappa_tot_mat[:,b1-1,b2-1])
            if probe1 == 'gy':
                Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_y_tot_mat[:,b1-1])

        edge_left_pge = ell_edges_left[-1]
        Cl_all = Cl_all.at[edge_left_pge:edge_left_pge+nk].set(get_power_BCMP_test.Pge_tot_mat[0,:])
        Cl_all = Cl_all.at[edge_left_pge+nk:edge_left_pge+2*nk].set(get_power_BCMP_test.Pge_tot_mat[1,:])
        Cl_all = Cl_all.at[edge_left_pge+2*nk:edge_left_pge+3*nk].set(get_power_BCMP_test.Pge_tot_mat[2,:])
        Cl_all = Cl_all * scale_vec
        mu = Cl_all[ell_sel_all]
        return numpyro.sample('cl', dist.MultivariateNormal(mu, 
                                                                # precision_matrix=P_total,
                                                                covariance_matrix=cov_total
                                                                ))




    from numpyro.handlers import seed, trace, condition
    # Now we condition the model on obervations
    observed_model = condition(model, {'cl': data_vec})

    import numpyro
    from numpyro.infer.reparam import LocScaleReparam, TransformReparam

    def config(x):
        if type(x['fn']) is dist.TransformedDistribution:
            return TransformReparam()
        elif type(x['fn']) is dist.Normal and ('decentered' not in x['name']):
            return LocScaleReparam(centered=0)
        else:
            return None

    observed_model_reparam = numpyro.handlers.reparam(observed_model, config=config)

    init_sample_dict = {}
    for key in sim_params_dict.keys():
        if key == 'cosmo':
            for subkey in sim_params_dict[key].keys():
                if subkey == 'H0':
                    init_sample_dict['h'] = sim_params_dict[key][subkey] / 100.0
                else:
                    init_sample_dict[subkey] = sim_params_dict[key][subkey]
        else:
            init_sample_dict[key] = sim_params_dict[key]
    for key in other_params_dict.keys():
        init_sample_dict[key] = other_params_dict[key]
    
    init_sample_dict['Delta_z_bias_array'] = jnp.array(prior_mult_shear_mu_all)
    init_sample_dict['mult_shear_bias_array'] = jnp.array(prior_mult_shear_mu_all)
    
    def get_jittered_init_values(rng_key, init_params, num_chains, jitter_std=0.01):
        jittered_params = {}
        for key, val in init_params.items():
            # jax.random.normal needs a shape, so we get it from the value
            shape = (num_chains,) + jnp.shape(val)
            noise = jax.random.normal(rng_key, shape=shape) * jitter_std
            jittered_params[key] = val + noise
        return jittered_params
    

    def do_mcmc(rng_key, n_vectorized=num_chains):
        # nuts_kernel = NUTS(model)
        
        if init_strategy == "median":
            nuts_kernel = numpyro.infer.NUTS(observed_model_reparam,
                                        # step_size=3e-1, 
                                        init_strategy=numpyro.infer.init_to_median,
                                        dense_mass=True,
                                        max_tree_depth=max_tree_depth,
                                        # max_tree_depth=5,                                     
                                        adapt_mass_matrix=True, 
                                        adapt_step_size=True
                                        )
        if init_strategy == "value":
            init_key, mcmc_key = jax.random.split(rng_key)
            init_vals_jittered = get_jittered_init_values(init_key, init_sample_dict, n_vectorized)
            nuts_kernel = numpyro.infer.NUTS(observed_model_reparam,
                                        # step_size=3e-1, 
                                        # init_strategy=numpyro.infer.init_to_value(values=init_sample_dict),
                                        init_strategy=numpyro.infer.init_to_value(values=init_vals_jittered),
                                        dense_mass=True,
                                        max_tree_depth=max_tree_depth,
                                        # max_tree_depth=5,                                     
                                        adapt_mass_matrix=True, 
                                        adapt_step_size=True
                                        )

        
        mcmc = numpyro.infer.MCMC(nuts_kernel, 
                                num_warmup=num_warmup, 
                                num_samples=num_samples,
                                num_chains=n_vectorized,
                                chain_method='vectorized',
                                progress_bar=False,
                                jit_model_args=True)

        mcmc.run(
            rng_key,
            extra_fields=("potential_energy", "diverging", "accept_prob"),
        )
        num_divergences = jnp.sum(mcmc.get_extra_fields()["diverging"])
        print(f"Device finished with {num_divergences} divergences.")

        return {**mcmc.get_samples(), **mcmc.get_extra_fields()}


    rng_keys = jax.random.split(jax.random.PRNGKey(42), n_parallel)
    traces = pmap(do_mcmc)(rng_keys)

    # concatenate traces along pmap'ed axis
    trace = {k: np.concatenate(v) for k, v in traces.items()}
    trace['RUN_SETTINGS'] = {}
    trace['RUN_SETTINGS']['prior_min'] = prior_min_all_dict
    trace['RUN_SETTINGS']['prior_max'] = prior_max_all_dict
    trace['RUN_SETTINGS']['fiducial_sims_params'] = sim_params_dict
    trace['RUN_SETTINGS']['fiducial_other_params'] = other_params_dict
    trace['RUN_SETTINGS']['fiducial_halo_params'] = halo_params_dict
    trace['RUN_SETTINGS']['fiducial_analysis_params'] = analysis_dict
    import dill as dill

    print(save_chain_dir)
    # dill.dump(trace, open(save_chain_dir + f'mcmc_v1_test.pkl', 'wb'))
    dill.dump(trace, open(savefname_out, 'wb'))