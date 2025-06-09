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


deproj = sys.argv[1]
probe = sys.argv[2]
model_matter = sys.argv[3]
use_gty_scale_cuts = True
use_xipm_Y3_scale_cuts = bool(ast.literal_eval(sys.argv[4]))


print(deproj, probe, model_matter, use_xipm_Y3_scale_cuts)

try:
    smooth_ym_model = sys.argv[5]
except:
    smooth_ym_model = 'poweradd'



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
new_data = read_yaml(abs_path_params + '/DESxACT/params_v3.yaml')
merged_data = always_merger.merge(default_data, new_data)

sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(merged_data)

from astropy.io import fits
df_cs = fits.open(os.path.abspath(abs_path_data + '/DESxACT/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_newbins.fits'))
z_array = df_cs['nz_source'].data['Z_MID']
nz_info_dict = {}
nz_info_dict['z_array_source'] = z_array
nz_info_dict['nbins'] = 4
for ji in range(nz_info_dict['nbins']):
    nz_info_dict['nz'+str(ji)] = np.maximum(df_cs['nz_source'].data['BIN'+str(ji+1)], 1e-4)
analysis_dict['nz_source_info_dict'] = nz_info_dict
other_params_dict['Delta_z_bias_array'] = np.zeros(analysis_dict['nz_source_info_dict']['nbins'])
other_params_dict['mult_shear_bias_array'] = np.zeros(analysis_dict['nz_source_info_dict']['nbins'])

analysis_dict['angles_data_array'] = df_cs['xip'].data['ANG'][0:20]

lmin, lmax, dl_log_array = 1.0, 81000.0, 0.23025851/3
l_array_all = np.exp(np.arange(np.log(lmin), np.log(lmax), dl_log_array))
dl_array = l_array_all[1:] - l_array_all[:-1]
l_array_survey = (l_array_all[1:] + l_array_all[:-1]) / 2.
halo_params_dict['ell_array'] = jnp.array(l_array_survey)
analysis_dict['l_array_survey'] = jnp.array(l_array_survey)
analysis_dict['dl_array_survey'] = jnp.array(dl_array)
analysis_dict['tSZ_transition_model'] = smooth_ym_model
analysis_dict['model_matter'] = model_matter

deproj_to_true_y_file = {
    'None': 'ilc_SZ_yy',
    'cib_1p0': 'ilc_SZ_deproj_cib_1.0_10.7_yy',
    'cib_1p2': 'ilc_SZ_deproj_cib_1.2_10.7_yy',
    'cib_1p4': 'ilc_SZ_deproj_cib_1.4_10.7_yy',
    'cib_1p6': 'ilc_SZ_deproj_cib_1.6_10.7_yy',
    'cib_1p7': 'ilc_SZ_deproj_cib_1.7_10.7_yy',
    'cib_1p8': 'ilc_SZ_deproj_cib_1.8_10.7_yy',
    'cib_2p0': 'ilc_SZ_deproj_cib_2.0_10.7_yy',
    'cib_1p0_dBeta': 'ilc_SZ_deproj_cib_cibdBeta_1.0_10.7_yy',
    'cib_1p2_dBeta': 'ilc_SZ_deproj_cib_cibdBeta_1.2_10.7_yy',
    'cib_1p4_dBeta': 'ilc_SZ_deproj_cib_cibdBeta_1.4_10.7_yy',
    'cib_1p6_dBeta': 'ilc_SZ_deproj_cib_cibdBeta_1.6_10.7_yy',
    'cib_1p7_dBeta': 'ilc_SZ_deproj_cib_cibdBeta_1.7_10.7_yy',
    'cib_1p8_dBeta': 'ilc_SZ_deproj_cib_cibdBeta_1.8_10.7_yy',
    'cib_2p0_dBeta': 'ilc_SZ_deproj_cib_cibdBeta_2.0_10.7_yy',
}

save_DV_dir = os.path.abspath(abs_path_data + '/DESxACT/DV_v3/')
df_measure = pk.load(open(f'{save_DV_dir}/DESxACT_gty_xip_xim_DV_{deproj_to_true_y_file[deproj]}.pk', 'rb'))
cov_total = df_measure['cov_total']
xi_all = df_measure['xi_all']
theta_all = df_measure['theta_all']
cov_total = jnp.array(cov_total)
data_vec = jnp.array(xi_all)


cov_total = jnp.array(cov_total)
data_vec = jnp.array(xi_all)


if probe == 'xip_xim':
    cov_total = cov_total[80:, 80:]
    data_vec = data_vec[80:]
elif probe == 'gty':
    cov_total = cov_total[:80, :80]
    data_vec = data_vec[:80]

fname = abs_path_data + '/DESxACT/scale_cuts_xipm_y3_gty_fid.ini'
config = configobj.ConfigObj(fname)
sc_xipm = config['shear-shear']
sc_gty = config['shear-tsz']

bin1_vals =  df_cs['xip'].data['BIN1'][::20]
bin2_vals =  df_cs['xip'].data['BIN2'][::20]
biny_vals = np.array([1,2,3,4])
ang_vals = df_cs['xip'].data['ANG'][0:20]

def process_indices(num, bin1_vals, bin2_vals, angle_ranges, offset, probe, angle_type, do_scalecuts):
    indices, indrm = [], []
    for js in range(num):
        binv, thetav = divmod(js, 20)
        bin1, bin2 = (bin1_vals[binv] - 1, bin2_vals[binv] - 1) if angle_type != 'gty' else (None, None)
        if do_scalecuts:
            sc_min, sc_max = map(float, angle_ranges[binv].split())
        else:
            sc_min, sc_max = 1e-3, 999.0
        if sc_min < ang_vals[thetav] < sc_max:
            indices.append([int(thetav), int(bin1), int(bin2)] if bin1 is not None else [int(thetav), int(binv)])
        elif probe in [angle_type, 'all']:
            indrm.append(int(offset + js))
    return jnp.array(indices, dtype=jnp.int32), jnp.array(indrm, dtype=jnp.int32)

# Process gty indices
sc_ranges_gty = [sc_gty[f'angle_range_gty_{biny}_0'] for biny in biny_vals]
index_gty, indrm_gty = process_indices(80, None, None, sc_ranges_gty, 0, probe, 'gty', use_gty_scale_cuts)
len_ind_gty = len(index_gty)
# Process xip indices
sc_ranges_xip = [sc_xipm[f'angle_range_xip_{bin1}_{bin2}'] for bin1, bin2 in zip(bin1_vals, bin2_vals)]
offset_xip = 80 if probe in ['gty', 'all'] else 0
index_xip, indrm_xip = process_indices(200, bin1_vals, bin2_vals, sc_ranges_xip, offset_xip, probe, 'xip_xim', use_xipm_Y3_scale_cuts)
len_ind_xip = len(index_xip)
# Process xim indices
sc_ranges_xim = [sc_xipm[f'angle_range_xim_{bin1}_{bin2}'] for bin1, bin2 in zip(bin1_vals, bin2_vals)]
offset_xim = 280 if probe in ['gty', 'all'] else 200
index_xim, indrm_xim = process_indices(200, bin1_vals, bin2_vals, sc_ranges_xim, offset_xim, probe, 'xip_xim', use_xipm_Y3_scale_cuts)
len_ind_xim = len(index_xim)

# Combine results
indrm = jnp.concatenate([indrm_gty, indrm_xip, indrm_xim], dtype=jnp.int32)
print('removing indices: ', indrm)

if len(indrm) > 0:
    data_vec = jnp.delete(data_vec, indrm)
    cov_total = jnp.delete(cov_total, indrm, axis=0)
    cov_total = jnp.delete(cov_total, indrm, axis=1)
P_total = jnp.linalg.inv(cov_total)


with open(abs_path_params + '/DESxACT/priors_v3_wider.yaml', 'r') as file:
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
prior_delta_z_mu_all = jnp.array([prior_mu_all_dict['Delta_z_bias_bin1'], prior_mu_all_dict['Delta_z_bias_bin2'], prior_mu_all_dict['Delta_z_bias_bin3'], prior_mu_all_dict['Delta_z_bias_bin4']])
prior_delta_z_sig_all = jnp.array([prior_sig_all_dict['Delta_z_bias_bin1'], prior_sig_all_dict['Delta_z_bias_bin2'], prior_sig_all_dict['Delta_z_bias_bin3'], prior_sig_all_dict['Delta_z_bias_bin4']])
prior_mult_shear_mu_all = jnp.array([prior_mu_all_dict['mult_shear_bias_bin1'], prior_mu_all_dict['mult_shear_bias_bin2'], prior_mu_all_dict['mult_shear_bias_bin3'], prior_mu_all_dict['mult_shear_bias_bin4']])
prior_mult_shear_sig_all = jnp.array([prior_sig_all_dict['mult_shear_bias_bin1'], prior_sig_all_dict['mult_shear_bias_bin2'], prior_sig_all_dict['mult_shear_bias_bin3'], prior_sig_all_dict['mult_shear_bias_bin4']])


cosmo_params_vary_names = ['Om0', 'sigma8', 'Ob0', 'h', 'ns']
# sims_params_vary_names = ['theta_ej_0', 'nu_theta_ej_z', 'nu_theta_ej_M', 'delta_rhogas', 'mu_beta', 'log10_Mc0', 'alpha_nt']
# sims_params_vary_names = ['theta_ej_0','theta_co_z', 'nu_theta_ej_M', 'nu_theta_ej_z', 'delta_rhogas', 'mu_beta','alpha_nt']
# sims_params_vary_names = ['theta_ej_0','theta_co_0', 'nu_theta_ej_M', 'nu_theta_ej_z', 'nu_theta_co_z', 'delta_rhogas', 'mu_beta','alpha_nt']
# sims_params_vary_names = ['theta_ej_0','nu_theta_ej_z', 'mu_beta','alpha_nt']
sims_params_vary_names = ['theta_ej_0','nu_theta_ej_z','nu_theta_ej_M', 'mu_beta', 'alpha_nt']
# sims_params_vary_names = ['theta_ej_0','theta_co_0', 'nu_theta_ej_M', 'nu_theta_ej_z', 'nu_theta_co_M', 'nu_theta_co_z','alpha_nt']
other_params_vary_names = ['alpha_ky', 'A_IA', 'eta_IA']
mult_shear_vary_names = ['mult_shear_bias_bin1', 'mult_shear_bias_bin2', 'mult_shear_bias_bin3', 'mult_shear_bias_bin4'] 
Delta_shear_vary_names = ['Delta_z_bias_bin1', 'Delta_z_bias_bin2', 'Delta_z_bias_bin3', 'Delta_z_bias_bin4']


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

def config(x):
    if type(x['fn']) is dist.TransformedDistribution:
        return TransformReparam()
    elif type(x['fn']) is dist.Normal and ('decentered' not in x['name']):
        return LocScaleReparam(centered=0)
    else:
        return None

def model():
    sim_params_dict_vary = copy.deepcopy(sim_params_dict)
    other_params_dict_vary = copy.deepcopy(other_params_dict)

    if len(cosmo_params_vary_names) > 0:
        for jp in range(len(cosmo_params_vary_names)):
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
    
    if len(other_params_vary_names) > 0:
        for jp in range(len(other_params_vary_names)):
            prior_min_jp = prior_min_all_dict[other_params_vary_names[jp]]
            prior_max_jp = prior_max_all_dict[other_params_vary_names[jp]]
            other_params_dict_vary[other_params_vary_names[jp]] = Uniform(other_params_vary_names[jp], prior_min_jp, prior_max_jp)

    if len(prior_delta_z_mu_all) > 0:
        Delta_z_bias_array = numpyro.sample('Delta_z_bias_array', dist.Normal(prior_delta_z_mu_all, prior_delta_z_sig_all)) 
        other_params_dict_vary['Delta_z_bias_array'] = Delta_z_bias_array
    
    if len(prior_mult_shear_mu_all) > 0:
        mult_shear_bias_array = numpyro.sample('mult_shear_bias_array', dist.Normal(prior_mult_shear_mu_all, prior_mult_shear_sig_all))
        other_params_dict_vary['mult_shear_bias_array'] = mult_shear_bias_array
        
    get_corrfunc_BCMP_test = get_xi(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict_vary)

    def get_gty_from_index(index):
        index_val = index_gty[index]
        return get_corrfunc_BCMP_test.gty_out_mat[index_val[0], index_val[1]]

    def get_xip_from_index(index):
        index_val = index_xip[index]
        return get_corrfunc_BCMP_test.xip_out_mat[index_val[0], index_val[1], index_val[2]]

    def get_xim_from_index(index):
        index_val = index_xim[index]
        return get_corrfunc_BCMP_test.xim_out_mat[index_val[0], index_val[1], index_val[2]]

    gty_val = vmap(get_gty_from_index)(jnp.arange(len_ind_gty))
    xip_val = vmap(get_xip_from_index)(jnp.arange(len_ind_xip))
    xim_val = vmap(get_xim_from_index)(jnp.arange(len_ind_xim))

    if probe == 'xip_xim':
        mu = jnp.concatenate([xip_val, xim_val])
    elif probe == 'gty':
        mu = gty_val
    else:
        mu = jnp.concatenate([gty_val, xip_val, xim_val])
    return numpyro.sample('cl', dist.MultivariateNormal(mu, 
                                                        # precision_matrix=P_total,
                                                        covariance_matrix=cov_total))



observed_model = condition(model, {'cl': data_vec})
observed_model_reparam = numpyro.handlers.reparam(observed_model, config=config)


# num_warmup = 3500
# num_samples = 4500
# num_chains= 16
# max_tree_depth = 4


num_warmup = 8000
num_samples = 8000
num_chains= 16
max_tree_depth = 4

# num_warmup = 6000
# num_samples = 2500
# num_chains= 20


def do_mcmc(rng_key, n_vectorized=num_chains):
    # nuts_kernel = NUTS(model)
    nuts_kernel = numpyro.infer.NUTS(observed_model_reparam,
                                step_size=3e-1, 
                                init_strategy=numpyro.infer.init_to_median,
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
        extra_fields=("potential_energy",),
    )
    return {**mcmc.get_samples(), **mcmc.get_extra_fields()}

n_parallel = jax.local_device_count()
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
trace['RUN_SETTINGS']['index_gty'] = index_gty
trace['RUN_SETTINGS']['index_xip'] = index_xip
trace['RUN_SETTINGS']['index_xim'] = index_xim
trace['RUN_SETTINGS']['indrm'] = indrm
import dill as dill
save_chain_dir = abs_path_results + '/DESxACT/chains_Feb/'
print(save_chain_dir)
dill.dump(trace, open(save_chain_dir + f'mcmc_v10_widemuej_nzfix_probe_{probe}_modelmatter_{model_matter}_deproj_{deproj}_samples_{num_samples}_warmup_{num_warmup}_num_chains_{num_chains*n_parallel}_treedepth_{max_tree_depth}_gtysc_{use_gty_scale_cuts}_Y3xipmsc_{use_xipm_Y3_scale_cuts}.pkl', 'wb'))

