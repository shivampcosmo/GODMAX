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

import numpyro
numpyro.set_platform("gpu")
numpyro.enable_x64()
numpyro.set_host_device_count(jax.device_count())
from numpyro.handlers import seed, trace, condition
from numpyro.infer.reparam import LocScaleReparam, TransformReparam
from numpyro.infer import HMC, HMCECS, MCMC, NUTS, SA, SVI, Trace_ELBO, init_to_value
from numpyro.distributions.transforms import AffineTransform
import numpyro.distributions as dist

import jax 
from tqdm import tqdm
import copy


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

import ast
print(sys.argv)
# probes_forecast = list(sys.argv[1])
probes_forecast = sys.argv[1].split(',')
sc_val = int(sys.argv[2])
probes_forecast_all_str = '_'.join(probes_forecast)
run_this_script = True

save_chain_dir = abs_path_results + f'/pge/chains_Apr_Fisher/{probes_forecast_all_str}/'

# check if the directory exists, if not create it:
if not os.path.exists(save_chain_dir):
    os.makedirs(save_chain_dir)
    print(f"Directory {save_chain_dir} created")
else:
    print(f"Directory {save_chain_dir} already exists")

savefname_out = save_chain_dir + f'mcmc_v2_{probes_forecast_all_str}_scval_{sc_val}_Fisher.pkl'

print(probes_forecast, sc_val, savefname_out)

# check if the file exists:
if os.path.exists(savefname_out):
    df = pk.load(open(savefname_out, 'rb'))
    print(df.keys())
    run_this_script = False
    print('File already exists, skipping the run.')



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
    # new_data = read_yaml(abs_path_params + '/DESxACT/params_v2.yaml')
    # new_data = read_yaml(abs_path_params + '/DESxACT/params_v0.yaml')
    # merged_data = always_merger.merge(default_data, new_data)

    sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(default_data)

    analysis_dict['beam_fwhm_arcmin'] = 1.4

    from scipy.interpolate import interp1d
    ks = np.geomspace(5e-2,50,15) # wavenumbers
    zedges = np.array([0.4, 0.6, 0.8,1.1])
    nz_lrg_all = np.loadtxt(abs_path_data + '/pge/desi_lrg_nz.txt')
    zv, nzv = nz_lrg_all[:,0], nz_lrg_all[:,1]

    nbins_lens = len(zedges) - 1
    zarray_lens = np.linspace(0.001, 1.6, 100)
    nz_interp = interp1d(zv, nzv, fill_value=1e-10, bounds_error=False)
    nz_zarray = nz_interp(zarray_lens)
    nz_lens = {}
    for jb in range(nbins_lens):
        nz_jb = np.zeros_like(nz_zarray)
        indsel = np.where((zarray_lens > zedges[jb]) & (zarray_lens < zedges[jb+1]))[0]
        nz_jb[indsel] = nz_zarray[indsel]
        norm_val = np.trapezoid(nz_jb, zarray_lens)
        nz_jb = nz_jb/norm_val
        nz_lens[jb] = nz_jb

    nz_info_dict = {}
    nz_info_dict['z_array_lens'] = zarray_lens
    nz_info_dict['nbins_lens'] = 3
    for ji in range(nz_info_dict['nbins_lens']):
        nz_info_dict['nz'+str(ji)] = np.maximum(nz_lens[ji], 1e-4)
    analysis_dict['nz_lens_info_dict'] = nz_info_dict

    df_nz_comoving = np.loadtxt(abs_path_data + '/pge/desi_lrg_comoving_density.txt')
    zarray_comoving = df_nz_comoving[:,0]
    nz_comoving = df_nz_comoving[:,1]
    analysis_dict['nbar_gal_comoving_zarray'] = zarray_comoving
    analysis_dict['nbar_gal_comoving_val'] = nz_comoving

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

    # lmin, lmax, dl_log_array = 10.0, 11000.0, 0.23025851
    lmin, lmax, dl_log_array = 80.0, 8800.0, 0.23025851
    l_array_all = np.exp(np.arange(np.log(lmin), np.log(lmax), dl_log_array))
    dl_array = l_array_all[1:] - l_array_all[:-1]
    l_array_survey = (l_array_all[1:] + l_array_all[:-1]) / 2.
    halo_params_dict['ell_array'] = jnp.array(l_array_survey)
    analysis_dict['l_array_survey'] = jnp.array(l_array_survey)
    analysis_dict['dl_array_survey'] = jnp.array(dl_array)
    analysis_dict['yy_noise_ell_fname'] = os.path.abspath(abs_path_data + '/pge/Noise_fid_yy_beamed_1p4arcmin.txt')
    Ngals_bins = 3.33*np.array([506905, 771875, 859824])
    analysis_dict['nbar_lens_bins'] = Ngals_bins/(analysis_dict['fsky_gg']*41253*(60**2))

    analysis_dict['symbolic_pk'] = True
    analysis_dict['symbolic_hmf'] = True

    import pickle as pk

    # df = pk.load(open(abs_path_data + '/pge/DV_fid_Cl_Pk_cov_mthresh_hod_v3.pk', 'rb'))
    df = pk.load(open(abs_path_data + '/pge/DV_fid_Cl_Pk_cov_mthresh_hod_v4_symb.pk', 'rb'))
    probes_all = df['probes']
    ell_edges_left = df['edges_left']
    ell_edges_right = df['edges_right']
    ell_array = df['ell_array']
    k_array = df['k_array']
    cov_total_orig = df['cov_mat']
    Cl_total = df['Cl_Pk']
    bin_comb_all_wprobe = df['bin_comb_all_wprobe']

    @jax.jit
    def call_Cl_from_calc(get_power_BCMP_test):
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
        return Cl_all



    import copy
    from tqdm import tqdm
    # from get_power_spectra import get_power_BCMP
    get_power_BCMP_test = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
    Cl_all = call_Cl_from_calc(get_power_BCMP_test)

    # nell = len(ell_array)
    # nk = len(analysis_dict['k_array_survey'])
    # Cl_all = jnp.zeros((cov_total_orig.shape[0]))
    # for jp1 in range(len(bin_comb_all_wprobe)):
    #     binwprobe1 = bin_comb_all_wprobe[jp1]
    #     probe1 = binwprobe1[0]
    #     bin_comb1 = binwprobe1[1]
    #     b1, b2 = bin_comb1[0], bin_comb1[1]
    #     if probe1 == 'ky':
    #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_y_tot_mat[:,b1-1])
    #     if probe1 == 'kk':
    #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_kappa_tot_mat[:,b1-1,b2-1])
    #     if probe1 == 'gg':
    #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_gal_tot_mat[:,b1-1,b2-1])
    #     if probe1 == 'gk':
    #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_kappa_tot_mat[:,b1-1,b2-1])
    #     if probe1 == 'gy':
    #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_y_tot_mat[:,b1-1])

    # edge_left_pge = ell_edges_left[-1]
    # Cl_all = Cl_all.at[edge_left_pge:edge_left_pge+nk].set(get_power_BCMP_test.Pge_tot_mat[0,:])
    # Cl_all = Cl_all.at[edge_left_pge+nk:edge_left_pge+2*nk].set(get_power_BCMP_test.Pge_tot_mat[1,:])
    # Cl_all = Cl_all.at[edge_left_pge+2*nk:edge_left_pge+3*nk].set(get_power_BCMP_test.Pge_tot_mat[2,:])


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
            # ell_max_jp = ell_sel_max[jp]
        # ell_sel_all.append(np.arange(ell_sel_min[jp], ell_sel_max[jp]))
    ell_sel_all = np.concatenate(ell_sel_all)

    ell_sel_all = []
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
            # ell_max_jp = ell_sel_max[jp]
        # ell_sel_all.append(np.arange(ell_sel_min[jp], ell_sel_max[jp]))
    ell_sel_all = np.concatenate(ell_sel_all)


    data_vec = Cl_all[ell_sel_all]
    cov_total = cov_total_orig[ell_sel_all,:][:, ell_sel_all]
    P_total = jnp.linalg.inv(cov_total)
    cov_forecast = cov_total


    do_cosmo = True
    do_sim = True
    do_IA = True
    do_multz = True
    do_dz = True



    if do_cosmo:
        cosmo_params_vary_names_all = ['Om0', 'sigma8', 'Ob0', 'h', 'ns']
        # cosmo_params_vary_names_all = ['Om0', 'sigma8']

        cosmo_params_fid_all = np.zeros(len(cosmo_params_vary_names_all))
        for jp in range(len(cosmo_params_vary_names_all)):
            cosmo_params_fid_all[jp] = sim_params_dict['cosmo'][cosmo_params_vary_names_all[jp]]

        dmu_cosmo_all = np.zeros((len(ell_sel_all), len(cosmo_params_vary_names_all)))
        for jp in tqdm(range(len(cosmo_params_vary_names_all))):
            cosmo_params_vary_names = [cosmo_params_vary_names_all[jp]]
            params_fid = jnp.array([sim_params_dict['cosmo'][cosmo_params_vary_names_all[jp]]])

            print(cosmo_params_vary_names, params_fid)

            def get_mean_cosmo(p):
                sim_params_dict_vary = copy.deepcopy(sim_params_dict)
                other_params_dict_vary = copy.deepcopy(other_params_dict)

                for jp in range(len(cosmo_params_vary_names)):
                    sim_params_dict_vary['cosmo'][cosmo_params_vary_names[jp]] = p[jp]
                
                get_power_BCMP_test = get_power_BCMP(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict_vary, verbose_time=False)
                Cl_all = call_Cl_from_calc(get_power_BCMP_test)
                # nell = len(ell_array)
                # nk = len(analysis_dict['k_array_survey'])
                # Cl_all = jnp.zeros((cov_total.shape[0]))
                # for jp1 in range(len(bin_comb_all_wprobe)):
                #     binwprobe1 = bin_comb_all_wprobe[jp1]
                #     probe1 = binwprobe1[0]
                #     bin_comb1 = binwprobe1[1]
                #     b1, b2 = bin_comb1[0], bin_comb1[1]
                #     if probe1 == 'ky':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_y_tot_mat[b1-1,:])
                #     if probe1 == 'kk':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_kappa_tot_mat[b1-1,b2-1,:])
                #     if probe1 == 'gg':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_gal_tot_mat[b1-1,b2-1,:])
                #     if probe1 == 'gk':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_kappa_tot_mat[b1-1,b2-1,:])
                #     if probe1 == 'gy':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_y_tot_mat[b1-1,:])

                # edge_left_pge = ell_edges_left[-1]
                # Cl_all = Cl_all.at[edge_left_pge:edge_left_pge+nk].set(get_power_BCMP_test.Pge_tot_mat[0,:])
                # Cl_all = Cl_all.at[edge_left_pge+nk:edge_left_pge+2*nk].set(get_power_BCMP_test.Pge_tot_mat[1,:])
                # Cl_all = Cl_all.at[edge_left_pge+2*nk:edge_left_pge+3*nk].set(get_power_BCMP_test.Pge_tot_mat[2,:])

                return Cl_all[ell_sel_all]

            jac_mean = jax.jit(jax.jacfwd(get_mean_cosmo))
            dmu_cosmo = jac_mean(params_fid)
            print(np.all(np.isfinite(dmu_cosmo)))
            dmu_cosmo_all[:, jp] = dmu_cosmo[:,0]


        P = jnp.linalg.inv(cov_forecast)

        F_cosmo = jnp.matmul(dmu_cosmo_all.T, jnp.matmul(P, dmu_cosmo_all))


    if do_sim:
        import jax 
        from tqdm import tqdm
        # sims_params_vary_names_all = ['theta_ej_0', 'mu_beta', 'delta_rhogas', 'nu_theta_ej_M', 'nu_theta_ej_z', 'beta_fshmr', 'gamma_fshmr',  'betacut_Nsat', 'siglogMstar_Ncen', 'alphasat_Nsat']
        sims_params_vary_names_baryons = ['theta_ej_0','nu_theta_ej_z','nu_theta_ej_M', 'mu_beta', 'alpha_nt']
        sims_params_vary_names_gals = ['log10M1_fshmr', 'log10M1_a_fshmr', 'beta_fshmr', 'beta_a_fshmr', 'delta_fshmr', 'delta_a_fshmr', 'siglogMstar_Ncen', 'alphasat_Nsat']
        sims_params_vary_names_all = sims_params_vary_names_baryons + sims_params_vary_names_gals

        sims_params_fid_all = np.zeros(len(sims_params_vary_names_all))
        for jp in range(len(sims_params_vary_names_all)):
            sims_params_fid_all[jp] = sim_params_dict[sims_params_vary_names_all[jp]]

        dmu_sims_all = np.zeros((len(ell_sel_all), len(sims_params_vary_names_all)))
        for jp in tqdm(range(len(sims_params_vary_names_all))):
            sim_param_vary_names = [sims_params_vary_names_all[jp]]
            params_fid = jnp.array([sim_params_dict[sims_params_vary_names_all[jp]]])
            print(sim_param_vary_names, params_fid)
            
            @jax.jit
            def get_mean_sims(p):
                sim_params_dict_vary = copy.deepcopy(sim_params_dict)
                other_params_dict_vary = copy.deepcopy(other_params_dict)

                for jp in range(len(sim_param_vary_names)):
                    sim_params_dict_vary[sim_param_vary_names[jp]] = p[jp]
                get_power_BCMP_test = get_power_BCMP(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict_vary, verbose_time=False)
                Cl_all = call_Cl_from_calc(get_power_BCMP_test)
                # nell = len(ell_array)
                # nk = len(analysis_dict['k_array_survey'])
                # Cl_all = jnp.zeros((cov_total.shape[0]))
                # for jp1 in range(len(bin_comb_all_wprobe)):
                #     binwprobe1 = bin_comb_all_wprobe[jp1]
                #     probe1 = binwprobe1[0]
                #     bin_comb1 = binwprobe1[1]
                #     b1, b2 = bin_comb1[0], bin_comb1[1]
                #     if probe1 == 'ky':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_y_tot_mat[b1-1,:])
                #     if probe1 == 'kk':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_kappa_tot_mat[b1-1,b2-1,:])
                #     if probe1 == 'gg':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_gal_tot_mat[b1-1,b2-1,:])
                #     if probe1 == 'gk':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_kappa_tot_mat[b1-1,b2-1,:])
                #     if probe1 == 'gy':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_y_tot_mat[b1-1,:])

                # edge_left_pge = ell_edges_left[-1]
                # Cl_all = Cl_all.at[edge_left_pge:edge_left_pge+nk].set(get_power_BCMP_test.Pge_tot_mat[0,:])
                # Cl_all = Cl_all.at[edge_left_pge+nk:edge_left_pge+2*nk].set(get_power_BCMP_test.Pge_tot_mat[1,:])
                # Cl_all = Cl_all.at[edge_left_pge+2*nk:edge_left_pge+3*nk].set(get_power_BCMP_test.Pge_tot_mat[2,:])

                return Cl_all[ell_sel_all]

            jac_mean = jax.jit(jax.jacfwd(get_mean_sims))
            dmu_sim = jac_mean(params_fid)
            print(np.all(np.isfinite(dmu_sim)))
            dmu_sims_all[:,jp] = dmu_sim[:,0]



        P = jnp.linalg.inv(cov_forecast)

        F_sims = jnp.matmul(dmu_sims_all.T, jnp.matmul(P, dmu_sims_all))


    if do_IA:

        import jax 
        from tqdm import tqdm


        IA_params_vary_names_all = ['A_IA', 'eta_IA']
        # IA_params_vary_names_all = ['A_IA', 'eta_IA']

        IA_params_fid_all = np.zeros(len(IA_params_vary_names_all))
        for jp in range(len(IA_params_vary_names_all)):
            IA_params_fid_all[jp] = other_params_dict[IA_params_vary_names_all[jp]]

        dmu_IA_all = np.zeros((len(ell_sel_all), len(IA_params_vary_names_all)))
        for jp in tqdm(range(len(IA_params_vary_names_all))):
            IA_param_vary_names = [IA_params_vary_names_all[jp]]
            params_fid = jnp.array([other_params_dict[IA_params_vary_names_all[jp]]])

            @jax.jit
            def get_mean_IA(p):
                sim_params_dict_vary = copy.deepcopy(sim_params_dict)
                other_params_dict_vary = copy.deepcopy(other_params_dict)

                for jp in range(len(IA_param_vary_names)):
                    other_params_dict_vary[IA_param_vary_names[jp]] = p[jp]
                
                get_power_BCMP_test = get_power_BCMP(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict_vary, verbose_time=False)
                Cl_all = call_Cl_from_calc(get_power_BCMP_test)
                # nell = len(ell_array)
                # nk = len(analysis_dict['k_array_survey'])
                # Cl_all = jnp.zeros((cov_total.shape[0]))
                # for jp1 in range(len(bin_comb_all_wprobe)):
                #     binwprobe1 = bin_comb_all_wprobe[jp1]
                #     probe1 = binwprobe1[0]
                #     bin_comb1 = binwprobe1[1]
                #     b1, b2 = bin_comb1[0], bin_comb1[1]
                #     if probe1 == 'ky':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_y_tot_mat[b1-1,:])
                #     if probe1 == 'kk':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_kappa_tot_mat[b1-1,b2-1,:])
                #     if probe1 == 'gg':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_gal_tot_mat[b1-1,b2-1,:])
                #     if probe1 == 'gk':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_kappa_tot_mat[b1-1,b2-1,:])
                #     if probe1 == 'gy':
                #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_y_tot_mat[b1-1,:])

                # edge_left_pge = ell_edges_left[-1]
                # Cl_all = Cl_all.at[edge_left_pge:edge_left_pge+nk].set(get_power_BCMP_test.Pge_tot_mat[0,:])
                # Cl_all = Cl_all.at[edge_left_pge+nk:edge_left_pge+2*nk].set(get_power_BCMP_test.Pge_tot_mat[1,:])
                # Cl_all = Cl_all.at[edge_left_pge+2*nk:edge_left_pge+3*nk].set(get_power_BCMP_test.Pge_tot_mat[2,:])

                return Cl_all[ell_sel_all]

            jac_mean = jax.jit(jax.jacfwd(get_mean_IA))
            dmu_IA = jac_mean(params_fid)
            dmu_IA_all[:, jp] = dmu_IA[:, 0]

        P = jnp.linalg.inv(cov_forecast)
        F_IA = jnp.matmul(dmu_IA_all.T, jnp.matmul(P, dmu_IA_all))


    if do_multz:

        import jax 

        @jax.jit
        def get_mean_mult_shear(p):
            sim_params_dict_vary = copy.deepcopy(sim_params_dict)
            other_params_dict_vary = copy.deepcopy(other_params_dict)

            other_params_dict_vary['mult_shear_bias_array'] = p

            get_power_BCMP_test = get_Cl(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict_vary, verbose_time=False)
            Cl_all = call_Cl_from_calc(get_power_BCMP_test)
            # nell = len(ell_array)
            # nk = len(analysis_dict['k_array_survey'])
            # Cl_all = jnp.zeros((cov_total.shape[0]))
            # for jp1 in range(len(bin_comb_all_wprobe)):
            #     binwprobe1 = bin_comb_all_wprobe[jp1]
            #     probe1 = binwprobe1[0]
            #     bin_comb1 = binwprobe1[1]
            #     b1, b2 = bin_comb1[0], bin_comb1[1]
            #     if probe1 == 'ky':
            #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_y_tot_mat[b1-1,:])
            #     if probe1 == 'kk':
            #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_kappa_tot_mat[b1-1,b2-1,:])
            #     if probe1 == 'gg':
            #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_gal_tot_mat[b1-1,b2-1,:])
            #     if probe1 == 'gk':
            #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_kappa_tot_mat[b1-1,b2-1,:])
            #     if probe1 == 'gy':
            #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_y_tot_mat[b1-1,:])

            # edge_left_pge = ell_edges_left[-1]
            # Cl_all = Cl_all.at[edge_left_pge:edge_left_pge+nk].set(get_power_BCMP_test.Pge_tot_mat[0,:])
            # Cl_all = Cl_all.at[edge_left_pge+nk:edge_left_pge+2*nk].set(get_power_BCMP_test.Pge_tot_mat[1,:])
            # Cl_all = Cl_all.at[edge_left_pge+2*nk:edge_left_pge+3*nk].set(get_power_BCMP_test.Pge_tot_mat[2,:])

            # return Cl_all[ell_sel_all]

        mult_shear_vary_names = ['mult_shear_bias_bin1', 'mult_shear_bias_bin2', 'mult_shear_bias_bin3', 'mult_shear_bias_bin4', 'mult_shear_bias_bin5'] 
        params_fid = jnp.zeros(len(mult_shear_vary_names))
        jac_mean = jax.jit(jax.jacfwd(get_mean_mult_shear))
        dmu_mult_shear_all = jac_mean(params_fid)

        P = jnp.linalg.inv(cov_forecast)

        F_mz = jnp.matmul(dmu_mult_shear_all.T, jnp.matmul(P, dmu_mult_shear_all))

        # import numpy as np
        d = np.ones(F_mz.shape[0]) *  1/(3e-3)**2

        F_mz = F_mz + np.diag(d)



    if do_dz:

        import jax 

        @jax.jit
        def get_mean_bias_Deltaz(p):
            sim_params_dict_vary = copy.deepcopy(sim_params_dict)
            other_params_dict_vary = copy.deepcopy(other_params_dict)

            other_params_dict_vary['Delta_z_bias_array'] = p

            get_power_BCMP_test = get_power_BCMP(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict_vary, verbose_time=False)
            Cl_all = call_Cl_from_calc(get_power_BCMP_test)
            # nell = len(ell_array)
            # nk = len(analysis_dict['k_array_survey'])
            # Cl_all = jnp.zeros((cov_total.shape[0]))
            # for jp1 in range(len(bin_comb_all_wprobe)):
            #     binwprobe1 = bin_comb_all_wprobe[jp1]
            #     probe1 = binwprobe1[0]
            #     bin_comb1 = binwprobe1[1]
            #     b1, b2 = bin_comb1[0], bin_comb1[1]
            #     if probe1 == 'ky':
            #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_y_tot_mat[b1-1,:])
            #     if probe1 == 'kk':
            #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_kappa_kappa_tot_mat[b1-1,b2-1,:])
            #     if probe1 == 'gg':
            #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_gal_tot_mat[b1-1,b2-1,:])
            #     if probe1 == 'gk':
            #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_kappa_tot_mat[b1-1,b2-1,:])
            #     if probe1 == 'gy':
            #         Cl_all = Cl_all.at[jp1*nell:(jp1+1)*nell].set(get_power_BCMP_test.Cl_gal_y_tot_mat[b1-1,:])

            # edge_left_pge = ell_edges_left[-1]
            # Cl_all = Cl_all.at[edge_left_pge:edge_left_pge+nk].set(get_power_BCMP_test.Pge_tot_mat[0,:])
            # Cl_all = Cl_all.at[edge_left_pge+nk:edge_left_pge+2*nk].set(get_power_BCMP_test.Pge_tot_mat[1,:])
            # Cl_all = Cl_all.at[edge_left_pge+2*nk:edge_left_pge+3*nk].set(get_power_BCMP_test.Pge_tot_mat[2,:])

            return Cl_all[ell_sel_all]

        Delta_shear_vary_names = ['Delta_z_bias_bin1', 'Delta_z_bias_bin2', 'Delta_z_bias_bin3', 'Delta_z_bias_bin4', 'Delta_z_bias_bin5']
        params_fid = jnp.zeros(len(Delta_shear_vary_names))
        jac_mean = jax.jit(jax.jacfwd(get_mean_bias_Deltaz))
        dmu_bias_Deltaz_all = jac_mean(params_fid)


        P = jnp.linalg.inv(cov_forecast)

        F_dz = jnp.matmul(dmu_bias_Deltaz_all.T, jnp.matmul(P, dmu_bias_Deltaz_all))

        # import numpy as np
        d = np.ones(F_dz.shape[0]) *  1/(1e-3)**2

        F_dz = F_dz + np.diag(d)


    dmu_all = np.concatenate([dmu_cosmo_all, dmu_sims_all, dmu_IA_all, dmu_mult_shear_all, dmu_bias_Deltaz_all], axis=1)
    param_vary_name_all = cosmo_params_vary_names_all + sims_params_vary_names_all + IA_params_vary_names_all + mult_shear_vary_names + Delta_shear_vary_names
    P = jnp.linalg.inv(cov_forecast)

    F_all = jnp.matmul(dmu_all.T, jnp.matmul(P, dmu_all))

    import numpy as np
    d = np.zeros(F_all.shape[0])
    nbins = analysis_dict['nz_source_info_dict']['nbins']
    d[F_all.shape[0]-nbins:] = 1/(1e-3)**2
    d[F_all.shape[0]-2*nbins:F_all.shape[0]-nbins] = 1/(3e-3)**2

    F_all = F_all + np.diag(d)

    saved_dict_final_F = {}
    saved_dict_final_F['F_all'] = F_all
    saved_dict_final_F['F_cosmo'] = F_cosmo
    saved_dict_final_F['F_sims'] = F_sims
    saved_dict_final_F['F_IA'] = F_IA
    saved_dict_final_F['F_mz'] = F_mz
    saved_dict_final_F['F_dz'] = F_dz

    saved_dict_final_F['dmu_cosmo_all'] = dmu_cosmo_all
    saved_dict_final_F['dmu_sims_all'] = dmu_sims_all
    saved_dict_final_F['dmu_IA_all'] = dmu_IA_all
    saved_dict_final_F['dmu_mult_shear_all'] = dmu_mult_shear_all
    saved_dict_final_F['dmu_bias_Deltaz_all'] = dmu_bias_Deltaz_all


    saved_dict_final_F['param_vary_name_all'] = param_vary_name_all
    saved_dict_final_F['cosmo_params_vary_names_all'] = cosmo_params_vary_names_all
    saved_dict_final_F['sims_params_vary_names_all'] = sims_params_vary_names_all
    saved_dict_final_F['IA_params_vary_names_all'] = IA_params_vary_names_all
    saved_dict_final_F['mult_shear_vary_names'] = mult_shear_vary_names
    saved_dict_final_F['Delta_shear_vary_names'] = Delta_shear_vary_names

    saved_dict_final_F['sim_params_dict'] = sim_params_dict
    saved_dict_final_F['halo_params_dict'] = halo_params_dict
    saved_dict_final_F['analysis_dict'] = analysis_dict
    saved_dict_final_F['other_params_dict'] = other_params_dict

    dill.dump(saved_dict_final_F, open(savefname_out, 'wb'))