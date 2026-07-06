import sys, os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.9'
import jax_cosmo.background as bkgrd
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from jax.lib import xla_bridge
# platform = xla_bridge.get_backend().platform
import jax
from jaxopt import ScipyBoundedMinimize
from jaxopt import LBFGSB
from jax import value_and_grad

print(jax.local_device_count(), jax.device_count())
# jax.config.update('jax_platform_name', platform)
jax.config.update("jax_enable_x64", True)
import jax
import matplotlib.pyplot as pl
# set latex to false:
pl.rcParams['text.usetex'] = True
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


from base_class import base_class
from get_radial_profiles import Profiles
from get_Pkzs import get_Pkz
from get_Cls import get_Cl
from get_Xis import get_xi
from get_covs import get_cov
import matplotlib.pyplot as pl
from jax import config
config.update("jax_enable_x64", True)
import scipy.interpolate as interp
import pickle as pk
import numpy as np
import copy
import jax.numpy as jnp
import colossus 
from jax import vmap, grad, pmap
import matplotlib.pyplot as pl
import jax_cosmo.background as bkgrd
import time
from jax_cosmo import Cosmology
from jax_cosmo.background import angular_diameter_distance, radial_comoving_distance
from astropy import constants as const
pl.rc('text', usetex=True)
import getdist
from getdist import plots, MCSamples

from tqdm import tqdm
import copy
import numpyro
numpyro.set_platform("gpu")
numpyro.enable_x64()
numpyro.set_host_device_count(jax.device_count())
from numpyro.handlers import seed, trace, condition
# Now we condition the model on obervations


import numpyro
from numpyro.infer.reparam import LocScaleReparam, TransformReparam

def config(x):
    if type(x['fn']) is dist.TransformedDistribution:
        return TransformReparam()
    elif type(x['fn']) is dist.Normal and ('decentered' not in x['name']):
        return LocScaleReparam(centered=0)
    else:
        return None


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

from numpyro.infer import HMC, HMCECS, MCMC, NUTS, SA, SVI, Trace_ELBO, init_to_value


ell_min = int(sys.argv[1])
ell_max = int(sys.argv[2])
num_warmup = int(sys.argv[3])
num_samples = int(sys.argv[4])
prior_version_to_run = int(sys.argv[5])
# Parse bin indices from sys.argv[6], e.g., '[1,2,3]' or '[2]'
bins_to_fit_str = sys.argv[6] if len(sys.argv) > 6 else '[1,2,3,4]'
delta_ell_bins = int(sys.argv[7]) if len(sys.argv) > 7 else 100
# Convert string to list of integers (1-indexed from user, convert to 0-indexed)
bins_to_fit = [int(b)-1 for b in bins_to_fit_str.strip('[]').split(',')]
print(f"Fitting bins (0-indexed): {bins_to_fit}")
import yaml
# from deepmerge import always_merger
from deepmerge import Merger
my_merger = Merger(
    [
        (list, ["override"]),  # Override lists instead of appending
        (dict, ["merge"])      # Still merge dictionaries
    ],
    ["override"],  # Default fallback strategy
    ["override"]
)

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
if len(bins_to_fit) == 1:
    new_data = read_yaml(abs_path_params + f'/xDESI/fit_only_bin{bins_to_fit[0]+1}/params_fit.yaml')
else:
    new_data = read_yaml(abs_path_params + '/xDESI/params_fit_abacus.yaml')
merged_data = my_merger.merge(default_data, new_data)

sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(merged_data)


from scipy.interpolate import interp1d
ks = np.geomspace(3e-1,10,10) # wavenumbers
zarray_lens = np.linspace(0.001, 1.6, 200)


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


# df_sims = pk.load(open(os.path.abspath(abs_path_data + '/xDESI/abacus_LRG_nz_Clgg_v4_combined_16sims.pkl'), 'rb'))
df_sims = pk.load(open(os.path.abspath(abs_path_data + f'/xDESI/abacus_LRG_nz_Clgg_v5_deltaell_{delta_ell_bins}_combined_16sims.pkl'), 'rb'))


zvals = df_sims['zvals']
nbins_lens = len(zvals)
delta_zarray = zarray_lens[1] - zarray_lens[0]
zarray_lens_edges = np.concatenate(([zarray_lens[0] - delta_zarray/2], 0.5*(zarray_lens[1:] + zarray_lens[:-1]), [zarray_lens[-1] + delta_zarray/2]))
nz_lens = {}

Ngals_bins = []
for jz, zval_group in enumerate(zvals):    
    Ngals_jz = df_sims['Ngal_all'][f'z{zval_group[0]:.3f}_{zval_group[-1]:.3f}']
    Ngals_bins.append(Ngals_jz)

    zcen_nz_file = df_sims['nz_gal_all']['z_array']
    nz_gal_file = df_sims['nz_gal_all'][f'z{zval_group[0]:.3f}_{zval_group[-1]:.3f}']
    nz_gal_interp = interp1d(zcen_nz_file, nz_gal_file, fill_value=0.0, bounds_error=False)

    hist_z = nz_gal_interp(zarray_lens)
    nz_lens[jz] = hist_z/( np.trapz(hist_z, zarray_lens) )

    nz_comoving = np.zeros_like(zarray_lens)
    indsel = np.where(hist_z > 1e-5)[0]

nbar_comoving_file = df_sims['nbar_comoving']
zcens_comoving_file = df_sims['zcens_comoving']

nbar_interp = interp1d(zcens_comoving_file, nbar_comoving_file, fill_value=1e-8, bounds_error=False)
nbar_array = nbar_interp(zarray_lens)

analysis_dict['nbar_gal_comoving_zarray'] = zarray_lens
analysis_dict['nbar_gal_comoving_val'] = nbar_array


nz_info_dict = {}
nz_info_dict['z_array_lens'] = zarray_lens
nz_info_dict['nbins_lens'] = nbins_lens
for ji in range(nz_info_dict['nbins_lens']):
    nz_info_dict['nz'+str(ji)] = np.maximum(nz_lens[ji], 1e-3)
z_edges_lens = []
for ji in range(nz_info_dict['nbins_lens']):
    nz_ji = nz_info_dict['nz'+str(ji)]
    zvals_ji = zarray_lens[np.where(nz_ji > 7e-1)[0]]
    z_edges_lens.append(np.array([zvals_ji[0], zvals_ji[-1]]))
z_edges_lens = np.array(z_edges_lens)
nz_info_dict['z_edges_bins_lens'] = z_edges_lens
analysis_dict['nz_lens_info_dict'] = nz_info_dict


from astropy.io import fits
df = fits.open(os.path.abspath(abs_path_data + '/forecast/lsst_simulate_Y1.fits'))
z_array = df['nz_source'].data['Z_MID']
nz_info_dict = {}
nz_info_dict['z_array_source'] = z_array
nz_info_dict['nbins'] = 1
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

analysis_dict['nbar_lens_bins'] = np.array(Ngals_bins)/(analysis_dict['fsky_gg']*41253*(60**2))

analysis_dict['symbolic_pk'] = True
analysis_dict['symbolic_hmf'] = True


base_test = base_class(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
profiles_test = Profiles(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, base_class_obj=base_test)
Pkz_test = get_Pkz(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Profiles_obj=profiles_test)
Cls_test = get_Cl(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Pkz_obj=Pkz_test)


# df_sims = pk.load(open(os.path.abspath(abs_path_data + '/xDESI/abacus_LRG_nz_Clgg_v4_combined_16sims.pkl'), 'rb'))
Cls_sims = df_sims['Cl_gg_all']
len_tot_ell = len(Cls_sims['l_array'])


l_array_sims = jnp.array(Cls_sims['l_array'])
indsel_ell = jnp.where((l_array_sims >= ell_min) & (l_array_sims <= ell_max))[0]
l_array_sims = l_array_sims[indsel_ell]

# Map bin indices to redshift keys
bin_to_zkey = {
    0: 'z0.300_0.400',
    1: 'z0.450_0.575',
    2: 'z0.650_0.800',
    3: 'z0.875_1.100'
}

# Load data only for selected bins
Cl_gg_sims_dict = {}
cov_gaussian_dict = {}
Cls_std_sims = df_sims['Cl_gg_all_std']

for bin_idx in bins_to_fit:
    zkey = bin_to_zkey[bin_idx]
    Cl_gg_sims_dict[bin_idx] = jnp.array(Cls_sims[zkey][indsel_ell])
    cov_gaussian_dict[bin_idx] = jnp.array(Cls_std_sims[zkey][indsel_ell])**2

print(l_array_sims[0], l_array_sims[-1])

# Concatenate data for selected bins only
data_vec_list = [Cl_gg_sims_dict[bin_idx] for bin_idx in bins_to_fit]
data_vec = jnp.concatenate(data_vec_list, axis=0)

# Concatenate covariances for selected bins only
cov_list = [cov_gaussian_dict[bin_idx] for bin_idx in bins_to_fit]
sig_total_combined = jnp.sqrt(jnp.concatenate(cov_list, axis=0))
cov_total = jnp.diag(sig_total_combined**2)

if len(bins_to_fit) == 1:
    with open(abs_path_params + f'/xDESI/fit_only_bin{bins_to_fit[0]+1}/priors_fit_v{prior_version_to_run}.yaml', 'r') as file:
        data = yaml.safe_load(file)
else:
    with open(abs_path_params + f'/xDESI/priors_fit_v{prior_version_to_run}.yaml', 'r') as file:
        data = yaml.safe_load(file)    
prior_limits = {key: tuple(map(float, value.split())) for key, value in data['prior_uniform'].items()}

prior_min_all_dict, prior_max_all_dict = {}, {}
for key in prior_limits.keys():
    prior_min_all_dict[key] = prior_limits[key][0]
    prior_max_all_dict[key] = prior_limits[key][1]


def model():
    sim_params_dict_vary = copy.deepcopy(sim_params_dict)
    other_params_dict_vary = copy.deepcopy(other_params_dict)

    # Automatically detect which array parameters to vary based on priors file
    # Extract base parameter names from prior keys (e.g., 'log10M1_fshmr_bin1' -> 'log10M1_fshmr')
    array_params_to_vary = set()
    for prior_key in prior_min_all_dict.keys():
        if '_bin' in prior_key:
            # Extract the base name (everything before '_bin')
            param_base_name = prior_key.rsplit('_bin', 1)[0]
            array_params_to_vary.add(param_base_name)
        else:
            param_base_name = prior_key
            array_params_to_vary.add(param_base_name)

    
    # Process each array parameter found in priors
    for param_base_name in array_params_to_vary:
        array_name = f'{param_base_name}_array'
        
        # Check if this array exists in sim_params_dict
        is_hod_param_array = False
        if (array_name not in sim_params_dict):
            if param_base_name not in other_params_dict:
                continue
            else:
                array_name = param_base_name
        else:
            is_hod_param_array = True
        
        if is_hod_param_array:
            # Start with the default array
            param_array = jnp.array(sim_params_dict[array_name])
            
            # Vary indices 1, 2, 3, 4 (keep index 0 fixed)
            for bin_idx in range(1, nbins_lens+1):  # bins 1, 2, 3, 4
                prior_key = f'{param_base_name}_bin{bin_idx}'
                
                # Check if priors exist for this parameter and bin
                if prior_key in prior_min_all_dict and prior_key in prior_max_all_dict:
                    prior_min = prior_min_all_dict[prior_key]
                    prior_max = prior_max_all_dict[prior_key]
                    
                    # Sample the parameter value
                    param_val = Uniform(prior_key, prior_min, prior_max)
                    
                    # Update the array at this index
                    param_array = param_array.at[bin_idx].set(param_val)
            
            # Update sim_params_dict_vary with the modified array
            sim_params_dict_vary[array_name] = param_array
        else:
            # Single parameter in other_params_dict
            prior_key = param_base_name
            
            # Check if priors exist for this parameter
            if prior_key in prior_min_all_dict and prior_key in prior_max_all_dict:
                prior_min = prior_min_all_dict[prior_key]
                prior_max = prior_max_all_dict[prior_key]
                
                # Sample the parameter value
                param_val = Uniform(prior_key, prior_min, prior_max)
                
                # Update other_params_dict_vary with the sampled value
                other_params_dict_vary[param_base_name] = param_val
    
    # Build the model as before
    base_test = base_class(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict_vary)
    profiles_test = Profiles(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict_vary, base_class_obj=base_test)
    Pkz_test = get_Pkz(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict_vary, Profiles_obj=profiles_test)
    get_power_BCMP_test = get_Cl(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict_vary, Pkz_obj=Pkz_test)

    nell = len(l_array_sims)
    mu = jnp.zeros(len(data_vec))
    # Loop only over selected bins
    for idx, jp1 in enumerate(bins_to_fit):
        theory_val_jp1 = get_power_BCMP_test.Cl_gal_gal_tot_mat[:,jp1,jp1]
        theory_val = jnp.exp(jnp.interp(jnp.log(l_array_sims), jnp.log(get_power_BCMP_test.ell_array), jnp.log(jnp.nan_to_num(theory_val_jp1) + 1e-20)))
        mu = mu.at[idx*nell:(idx+1)*nell].set(theory_val)

    return numpyro.sample('cl', dist.MultivariateNormal(mu, 
                                                            covariance_matrix=cov_total
                                                            ))


observed_model = condition(model, {'cl': data_vec})
observed_model_reparam = numpyro.handlers.reparam(observed_model, config=config)

# num_warmup = 1600
# num_samples = 3600
num_chains = 24
max_tree_depth = 4


def do_mcmc(rng_key, n_vectorized=num_chains):
    nuts_kernel = numpyro.infer.NUTS(observed_model_reparam,
                                step_size=3e-1, 
                                init_strategy=numpyro.infer.init_to_median,
                                dense_mass=True,
                                max_tree_depth=max_tree_depth,
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



with open(abs_path_params + '/xDESI/params_latex.yaml', 'r') as file:
    data = yaml.safe_load(file)
latex_vars = {key: tuple(map(str, value.split())) for key, value in data['latex_names'].items()}


samps_all_dict = {k: np.concatenate(v) for k, v in traces.items()}

import pickle as pk

samps = []
keys = []
for key in samps_all_dict:
    if ('base' not in key) and ('decentered' not in key) and ('potential_energy' not in key) and ('diverging' not in key):
        # print(key, samps_all_dict[key].shape, len(samps_all_dict[key].shape))
        if len(samps) == 0:
            samps = samps_all_dict[key][:, None]
        else:
            if len(samps_all_dict[key].shape) > 1:
                samps = np.concatenate((samps, samps_all_dict[key]), axis=1)
            else:
                samps = np.concatenate((samps, samps_all_dict[key][:, None]), axis=1)
        if 'array' in key:
            for jb in range(samps_all_dict[key].shape[1]):
                keys.append(key + str(jb))
        else:
            keys.append(key)
samps = np.array(samps)
names = keys
samples = MCSamples(samples=samps,names = names, labels=[latex_vars[param][0] for param in names])
params = samples.getParamNames().getRunningNames()

print(samps.shape, len(names))

chi2 = samps_all_dict['potential_energy']
names_array = list(prior_min_all_dict.keys())
xmin = []
indmin = np.argmin(chi2)

param_vary_names_final = []
for name in names_array:
    indsel = names.index(name)
    # xmin.append(np.median(samps[:, indsel]))
    xmin.append((samps[indmin, indsel]))
    param_vary_names_final.append(name)
    print(name, samps[indmin, indsel])

params_vary_names = list(prior_min_all_dict.keys())
lower_bounds_dict = prior_min_all_dict
upper_bounds_dict = prior_max_all_dict

def get_value(x, return_model=False):

    sim_params_dict_vary = copy.deepcopy(sim_params_dict)
    
    # for jp in range(len(sims_params_vary_names)):
    array_params_to_vary = set()
    for prior_key in prior_min_all_dict.keys():
        if '_bin' in prior_key:
            # Extract the base name (everything before '_bin')
            param_base_name = prior_key.rsplit('_bin', 1)[0]
            array_params_to_vary.add(param_base_name)
    
    # Process each array parameter found in priors
    for param_base_name in array_params_to_vary:
        array_name = f'{param_base_name}_array'
        
        # Check if this array exists in sim_params_dict
        if array_name not in sim_params_dict:
            continue
            
        # Start with the default array
        param_array = jnp.array(sim_params_dict[array_name])
        
        # Vary indices 1, 2, 3, 4 (keep index 0 fixed)
        for bin_idx in range(1, nbins_lens+1):  # bins 1, 2, 3, 4
            prior_key = f'{param_base_name}_bin{bin_idx}'
            
            # Check if priors exist for this parameter and bin
            if prior_key in prior_min_all_dict and prior_key in prior_max_all_dict:
                indsel = params_vary_names.index(prior_key)
                param_val = x[indsel]
                
                # Update the array at this index
                param_array = param_array.at[bin_idx].set(param_val)
        
        # Update sim_params_dict_vary with the modified array
        sim_params_dict_vary[array_name] = param_array
        
    base_test = base_class(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict)
    profiles_test = Profiles(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict, base_class_obj=base_test)
    Pkz_test = get_Pkz(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict, Profiles_obj=profiles_test)
    get_power_BCMP_test = get_Cl(sim_params_dict_vary, halo_params_dict, analysis_dict, other_params_dict, Pkz_obj=Pkz_test)


    nell = len(l_array_sims)
    theory_combined = jnp.zeros(len(data_vec))
    # Loop only over selected bins
    for idx, jp1 in enumerate(bins_to_fit):
        theory_val_jp1 = get_power_BCMP_test.Cl_gal_gal_tot_mat[:,jp1,jp1]
        theory_val = jnp.exp(jnp.interp(jnp.log(l_array_sims), jnp.log(get_power_BCMP_test.ell_array), jnp.log(theory_val_jp1)))
        theory_combined = theory_combined.at[idx*nell:(idx+1)*nell].set(theory_val)
    chi2 = jnp.sqrt(jnp.sum((theory_combined - data_vec)**2 / sig_total_combined**2))
    if return_model:
        return chi2, get_power_BCMP_test
    else:
        return chi2

chi2_good, model_good = get_value(xmin, return_model=True)
print(chi2_good)


# Build plot data for selected bins only
plot_data = {
    'l_array_sims': l_array_sims,
    'bins_to_fit': bins_to_fit
}

# Add data and model predictions for each selected bin
for bin_idx in bins_to_fit:
    plot_data[f'Cl_gg_sims{bin_idx}'] = Cl_gg_sims_dict[bin_idx]
    plot_data[f'cov_gaussian_{bin_idx}'] = cov_gaussian_dict[bin_idx]
    logClgg_sims_interp = jnp.exp(jnp.interp(jnp.log(l_array_sims), jnp.log(Cls_test.ell_array), jnp.log(model_good.Cl_gal_gal_tot_mat[:,bin_idx,bin_idx])))
    plot_data[f'logClgg_sims_interp{bin_idx}'] = logClgg_sims_interp


saved = {'samps': samps, 'names': names, 'labels': [latex_vars[param][0] for param in names], 'chi2_good': chi2_good, 'param_vary_names_final': param_vary_names_final, 'xmin': xmin, 'lower_bounds_dict': lower_bounds_dict, 'upper_bounds_dict': upper_bounds_dict, 'plot_data': plot_data, 'bins_to_fit': bins_to_fit}

# Create filename with bin information
bins_str = '_'.join([str(b+1) for b in bins_to_fit])  # Convert back to 1-indexed for filename

pk.dump(saved, open(abs_path_results + f'/xDESI/mcmc_fit_abacus_v5_16sims_deltaell_{delta_ell_bins}_{num_samples}samples_ellrange_{ell_min}_{ell_max}_bins_{bins_str}_nparams_{len(param_vary_names_final)}_v{prior_version_to_run}.pkl', 'wb'))


