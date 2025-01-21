import jax.numpy as jnp
from jax import vmap
import numpy as np
import astropy.units as u
from astropy import constants as const
import jax.scipy.integrate as jsi
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.keV / u.cm**3)).value
mp = (1.6726219e-27*u.kg).to(u.Msun).value
mue = 1.14
Mpc_to_cm = 3.086e24
G_new_rhom = const.G.to(u.Mpc**3 / ((u.s**2) * u.M_sun))
import jax_cosmo.background as bkgrd
import time
import helpers.constants as constants
from jax_cosmo import Cosmology
from helpers.jax_cosmo_power import linear_matter_power
from jax_cosmo.background import angular_diameter_distance, radial_comoving_distance

class EmptyCallable:
    def __call__(self, *args, **kwargs):
        return 0

def get_vmapped_func(func, num_args):
    """
    Get the vmap function for the given function with the number of arguments.
    """
    if num_args == 2:
        func = vmap(func, (0, None))
        func = vmap(func, (None, 0))
        return func
    if num_args == 3:
        func = vmap(func, (0, None, None))
        func = vmap(func, (None, 0, None))
        func = vmap(func, (None, None, 0))
        return func
    if num_args == 4:
        func = vmap(func, (0, None, None, None))
        func = vmap(func, (None, 0, None, None))
        func = vmap(func, (None, None, 0, None))
        func = vmap(func, (None, None, None, 0))
        return func

def get_vmapped_func_warg(func, num_args1, num_args2):
    """
    Get the vmap function for the given function with the number of arguments.
    """
    if (num_args1 == 2) and (num_args2 == 3):
        return vmap(vmap(func, in_axes=(0, None, None)), in_axes=(None, 0, None))
    if (num_args1 == 2) and (num_args2 == 4):
        return vmap(vmap(func, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None))
    if (num_args1 == 3) and (num_args2 == 4):
        func = vmap(func, (0, None, None, None))
        func = vmap(func, (None, 0, None, None))
        func = vmap(func, (None, None, 0, None))
        return func
    if (num_args1 == 2) and (num_args2 == 5):
        func = vmap(func, (0, None, None, None, None))
        func = vmap(func, (None, 0, None, None, None))
        return func
    if (num_args1 == 3) and (num_args2 == 5):
        func = vmap(func, (0, None, None, None, None))
        func = vmap(func, (None, 0, None, None, None))
        func = vmap(func, (None, None, 0, None, None))
        return func


class base_class:
    """"
    Class to calculate the BCMP profile as described in BCM 2018 (Schneider et al.) by augmenting it to have pressure term
    """

    def __init__(
            self,
            sim_params_dict: dict= None,
            halo_params_dict: dict = None,
            analysis_dict: dict = None,     
            other_params_dict: dict = None
        ):
        """
        Initialize the class with the simulation and halo parameters.

        Args:
            sim_params_dict (dict): Dictionary containing the simulation parameters. The dictionary should contain all the relevant cosmological and BCMP parameters. See the example notebook () for more details.
            halo_params_dict (dict): Dictionary containing the halo model calculation parameters. This basically controls the resolution of the calculation. See the example notebook () for more details.
            analysis_dict (dict): Dictionary containing the analysis parameters. See the example notebook () for more details.
            num_points_trapz_int (int): Number of points to use in the trapezoidal integration.
            verbose_time (bool): If True, print the time taken to calculate the BCMP profile.
        Returns:
            None
        """

        self.ENABLE_TIMING = analysis_dict.get('verbose_time', False)
        
        self.read_all_input(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
        if self.init_power:
            self.get_power_spectra_cosmo()

    def timing_decorator(func):
        """Decorator to time a function if the instance or class enables timing."""
        def wrapper(self, *args, **kwargs):
            # Check if timing is enabled (instance or class-level flag)
            if getattr(self, "ENABLE_TIMING", False):
                start_time = time.perf_counter()
                result = func(self, *args, **kwargs)
                end_time = time.perf_counter()
                print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds")
                return result
            else:
                return func(self, *args, **kwargs)
        return wrapper

    def logspace_trapezoidal_integral(self, f, logx, jz=None, jM=None, axis_tup=(0, None, None, None)):
        x = jnp.exp(logx)
        fx = (vmap(f, axis_tup)(jnp.arange(len(logx)), jz, jM, x))*(4*jnp.pi*x**2) * x
        integral_value = jsi.trapezoid(fx, x=logx)
        return integral_value


    @timing_decorator
    def read_all_input(self, sim_params_dict, halo_params_dict, analysis_dict=None, other_params_dict=None):
        """
        Read all the input parameters from the input dictionaries and store them in the class.
        """

        # Initialize the jax_cosmo cosmology
        cosmo_params = sim_params_dict.get('cosmo', {'flat': True, 'H0': 67.2, 'Om0': 0.31, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95, 'w0':-1.0})
        self.cosmo_params = cosmo_params
        self.cosmo_jax = Cosmology(
            Omega_c=cosmo_params['Om0'] - cosmo_params['Ob0'],
            Omega_b=cosmo_params['Ob0'],
            h=cosmo_params['H0'] / 100.,
            sigma8=cosmo_params['sigma8'],
            n_s=cosmo_params['ns'],
            Omega_k=0.,
            w0=cosmo_params['w0'],
            wa=0.
            )
        self.h = cosmo_params['H0'] / 100.
        self.Om0 = cosmo_params['Om0']
        self.H0 = 100. * (u.km / (u.s * u.Mpc))
        self.rho_m_bar = self.cosmo_params['Om0'] * ((3 * (self.H0**2) / (8 * jnp.pi * G_new_rhom)).to(u.M_sun / (u.Mpc**3))).value
        self.init_power = sim_params_dict.get('init_power', True)
        # Get all the relevant parameters from the input dictionaries. First start with the simulation parameters


        self.nfw_trunc = sim_params_dict.get('nfw_trunc',True)
        self.epsilon_rt=sim_params_dict.get('epsilon_rt',4.0)

        
        # theta_ejection is one of the main parameters of the model, controls the radii to which the gas is ejected due to feedback. Get it evolution parameters with mass, redshift and concentration.
        self.theta_ej_0=sim_params_dict.get('theta_ej_0',4.0)
        self.log10_Mstar0_theta_ej=sim_params_dict.get('log10_Mstar0_theta_ej',14.0)
        self.nu_theta_ej_M=sim_params_dict.get('nu_theta_ej_M',0.0)
        self.nu_theta_ej_z=sim_params_dict.get('nu_theta_ej_z',0.0)
        self.nu_theta_ej_c=sim_params_dict.get('nu_theta_ej_c',0.0)        

        # This is the core radius of the gas profile.
        self.theta_co_0=sim_params_dict.get('theta_co_0',0.1)
        self.log10_Mstar0_theta_co=sim_params_dict.get('log10_Mstar0_theta_co',14.0)
        self.nu_theta_co_M=sim_params_dict.get('nu_theta_co_M',0.0)
        self.nu_theta_co_z=sim_params_dict.get('nu_theta_co_z',0.0)                
        self.nu_theta_co_c=sim_params_dict.get('nu_theta_co_c',0.0)                        
        

        # Parameters controlling beta, the outer slope of the gas profile. This one is also pretty important.
        self.mu_beta=sim_params_dict.get('mu_beta',0.21)
        log10_Mstar0 = sim_params_dict.get('log10_Mstar0',13.0)
        self.Mstar0 = 10**log10_Mstar0        
        log10_Mc0 = sim_params_dict.get('log10_Mc0',14.83)
        self.Mc0 = 10**log10_Mc0
        self.nu_z = sim_params_dict.get('nu_z',0.0)
        self.nu_M = sim_params_dict.get('nu_M',0.0)

        # Other parameters controlling the slope of the gas profile.
        self.gamma_rhogas = sim_params_dict.get('gamma_rhogas', 2.)
        self.delta_rhogas = sim_params_dict.get('delta_rhogas', 7.)


        # Non-thermal pressure support parameters:
        self.a_zeta=sim_params_dict.get('a_zeta',0.3)
        self.n_zeta=sim_params_dict.get('n_zeta',2.0)
        self.alpha_nt = sim_params_dict.get('alpha_nt',0.18)
        self.beta_nt = sim_params_dict.get('beta_nt',0.5)
        self.n_nt = sim_params_dict.get('n_nt',0.3)


        # Stellar profile parameters. Basically these are the parameters for the HOD of galaxies as well. See Leauthaud et al. 2011 for more details.
        # The parameters control the amplitude and redshift evolution of the stellar profile.
        self.log10M1_fshmr = sim_params_dict.get('log10M1_fshmr', 12.35)
        self.log10M1_a_fshmr = sim_params_dict.get('log10M1_a_fshmr', 0.28)        
        self.log10Mstar0_fshmr = sim_params_dict.get('log10Mstar0_fshmr', 10.72)
        self.log10Mstar0_a_fshmr = sim_params_dict.get('log10Mstar0_a_fshmr', 0.55)        
        self.beta_fshmr = sim_params_dict.get('beta_fshmr', 0.44)        
        self.beta_a_fshmr = sim_params_dict.get('beta_a_fshmr', 0.18)        
        self.delta_fshmr = sim_params_dict.get('delta_fshmr', 0.57)        
        self.delta_a_fshmr = sim_params_dict.get('delta_a_fshmr', 0.17)                
        self.gamma_fshmr = sim_params_dict.get('gamma_fshmr', 1.56)        
        self.gamma_a_fshmr = sim_params_dict.get('gamma_a_fshmr', 2.51)                
        self.siglogMstar_Ncen = sim_params_dict.get('siglogMstar_Ncen', 0.25)                
        self.alphasat_Nsat = sim_params_dict.get('alphasat_Nsat', 1.0)                
        self.Bcut_Nsat = sim_params_dict.get('Bcut_Nsat', 1.69)                        
        self.Bsat_Nsat = sim_params_dict.get('Bsat_Nsat', 9.01)                                
        self.betacut_Nsat = sim_params_dict.get('betacut_Nsat', 0.6)                                
        self.betasat_Nsat = sim_params_dict.get('betasat_Nsat', 0.74)    

        # In case don't want to model proper galaxies, then can just set simple stellar profile parameters:
        self.eta_star=sim_params_dict.get('eta_star',0.3)
        self.eta_cga=sim_params_dict.get('eta_cga',0.6)
        self.A_starcga=sim_params_dict.get('A_starcga',0.09)
        log10_M1_starcga=sim_params_dict.get('log10_M1_starcga',11.4)
        self.M1_starcga=10**log10_M1_starcga


        # Now get the halo parameters:

        # First the radial grid for the halo profiles.
        rmin, rmax, nr = halo_params_dict.get('rmin', 5e-3), halo_params_dict.get('rmax',3), halo_params_dict.get('nr', 63)
        self.r_array = jnp.logspace(jnp.log10(rmin), jnp.log10(rmax), nr)

        # Now the redshift array. Note that either can provide a redshift array or the min, max and number of redshifts.
        zmin, zmax, nz = halo_params_dict.get('zmin', 1e-3), halo_params_dict.get('zmax',1.5), halo_params_dict.get('nz',32)
        if 'z_array' in halo_params_dict.keys():
            self.z_array = jnp.array(halo_params_dict['z_array'])
            nz = len(self.z_array)
        else:
            self.z_array = jnp.linspace(zmin, zmax, nz)
        self.scale_fac_a_array = 1./(1. + self.z_array)

        kmin, kmax, nk = halo_params_dict.get('kmin', 1E-4), halo_params_dict.get('kmax', 150.0), halo_params_dict.get('nk', 48)
        self.kPk_array = jnp.logspace(jnp.log10(kmin), jnp.log10(kmax), nk)


        # Now the mass array:
        lg10_Mmin, lg10_Mmax, nM = halo_params_dict.get('lg10_Mmin', 12), halo_params_dict.get('lg10_Mmax', 15.0), halo_params_dict.get('nM', 32)
        self.M_array = jnp.logspace(lg10_Mmin, lg10_Mmax, nM)
        self.mdef_Delta = halo_params_dict.get('mdef_Delta', 200)
        self.nM, self.nz, self.nr, self.nk = nM, nz, nr, nk


        self.conc_model = halo_params_dict.get('conc_model','Duffy08')
        self.hmf_model = halo_params_dict.get('hmf_model', 'tinker08')

        self.ell_array = halo_params_dict.get('ell_array',None)
        if self.ell_array is None:
            ellmin, ellmax, nell = halo_params_dict['ellmin'], halo_params_dict['ellmax'], halo_params_dict['nell']
            self.ell_array = jnp.logspace(jnp.log10(ellmin), jnp.log10(ellmax), nell)        
        else:            
            ellmin, ellmax, nell = jnp.min(self.ell_array), jnp.max(self.ell_array), len(self.ell_array)
        self.nell = len(self.ell_array)
        self.do_corr_2h_mm = halo_params_dict.get('do_corr_2h_mm',True)


        self.model_galaxies = analysis_dict.get('model_galaxies',True)
        self.model_tSZ = analysis_dict.get('model_tSZ',True)
        # Weather to model the matter with full baryonic effects or just with halofit, for shear-2pt chains
        self.model_matter = analysis_dict.get('model_matter','DMB')

        self.angles_data_array = jnp.array(analysis_dict.get('angles_data_array', jnp.logspace(jnp.log10(2.5), jnp.log10(250), 20)))
        self.nt_out = len(self.angles_data_array)

        # Comoving number density of the galaxies. This can be an array of redshifts and values. Basically M_star_threshold (mininum stellar mass) is obtained from this.
        self.nbar_gal_comoving_z_array = analysis_dict.get('nbar_gal_comoving_zarray', jnp.linspace(0.01, 2.0, 64))                            
        self.nbar_gal_comoving_val_array = analysis_dict.get('nbar_gal_comoving_val', jnp.zeros(64))                            
        # Get the comoving number density of galaxies at these redshifts.
        try:
            self.nbar_gal_comoving_array = jnp.interp(self.z_array, self.nbar_gal_comoving_z_array, self.nbar_gal_comoving_val_array)
        except:
            self.nbar_gal_comoving_array = jnp.zeros_like(self.z_array)


        # Controls the accuracy of the trapezoidal integration.
        self.num_points_trapz_int = analysis_dict.get('num_points_trapz_int', 64)
        self.num_points_gal_cal = analysis_dict.get('num_points_trapz_int', 32)
        
        self.calc_nfw_only = analysis_dict.get('calc_nfw_only', True)
        self.beam_fwhm_arcmin = analysis_dict.get('beam_fwhm_arcmin', 1.4)
        self.sig_beam = self.beam_fwhm_arcmin * (1. / 60.) * (jnp.pi / 180.) * (1. / jnp.sqrt(8. * jnp.log(2)))

        self.k_array_survey = analysis_dict.get('k_array_survey', self.kPk_array)
        self.zmin_for_Cls = analysis_dict.get('zmin_for_Cls', 0.01)
        self.zmax_for_Cls = analysis_dict.get('zmax_for_Cls', 1.6)
        self.nz_for_Cls = analysis_dict.get('nz_for_Cls', 128)
        self.z_array_for_Cls = jnp.linspace(self.zmin_for_Cls, self.zmax_for_Cls, self.nz_for_Cls)
        self.scale_fac_a_array_for_Cls = 1./(1. + self.z_array_for_Cls)

        nz_info_dict = analysis_dict.get('nz_source_info_dict', None)
        try:
            self.nbins = nz_info_dict['nbins']
            self.z_array_nz = jnp.array(nz_info_dict['z_array_source'])
            pzs_inp_mat = np.zeros((self.nbins, len(self.z_array_nz)))
            for jb in range(self.nbins):
                pzs_inp_mat[jb, :] = nz_info_dict['nz' + str(jb)]
            self.pzs_inp_mat_inp = jnp.array(pzs_inp_mat)
        except:
            self.nbins = 1
            self.z_array_nz = jnp.linspace(0.01, 1.5, 128)
            self.pzs_inp_mat_inp = jnp.array([jnp.ones_like(self.z_array_nz)])

        nz_info_dict = analysis_dict.get('nz_lens_info_dict', None)
        try:
            self.nbins_lens = nz_info_dict['nbins_lens']
            self.z_array_nz_lens = jnp.array(nz_info_dict['z_array_lens'])
            self.zmax_lens = self.z_array_nz_lens[-1]
            pzs_inp_mat = np.zeros((self.nbins_lens, len(self.z_array_nz_lens)))
            for jb in range(self.nbins_lens):
                pzs_inp_mat[jb, :] = nz_info_dict['nz' + str(jb)]
            self.pzs_inp_mat_inp_lens = jnp.array(pzs_inp_mat)
        except:
            self.nbins_lens = 1
            self.z_array_nz_lens = jnp.linspace(0.01, 1.5, 128)
            self.pzs_inp_mat_inp_lens = jnp.array([jnp.ones_like(self.z_array_nz_lens)])


        self.zmax = self.z_array_nz[-1]
        self.A_IA = other_params_dict.get('A_IA', 0.1)
        self.eta_IA = other_params_dict.get('eta_IA', 0.1)
        self.z0_IA = other_params_dict.get('z0_IA', 0.62)
        self.C1_bar = other_params_dict.get('C1_rhocrit', 0.0134)
        self.C1_rho_m_bar = self.C1_bar * self.cosmo_params['Om0']
        self.Delta_z_bias_array = jnp.array(other_params_dict.get('Delta_z_bias_array', [0.0]))
        self.mult_shear_bias_array = jnp.array(other_params_dict.get('mult_shear_bias_array', [0.0]))

        self.tSZ_transition_model = analysis_dict.get('tSZ_transition_model', 'poweradd')
        self.alpha_ky = other_params_dict.get('alpha_ky', 1.0)
        self.alpha_gy = other_params_dict.get('alpha_gy', 1.0)

    @timing_decorator
    def get_power_spectra_cosmo(self):
        """
        Get the linear matter power spectra and comoving distances, growth etc using jax_cosmo at a defined k_array
        """
        self.chi_array = radial_comoving_distance(self.cosmo_jax, self.scale_fac_a_array)
        self.DA_array = angular_diameter_distance(self.cosmo_jax, self.scale_fac_a_array)
        self.growth_array = bkgrd.growth_factor(self.cosmo_jax, self.scale_fac_a_array)
        self.dchi_dz_array = (const.c.value * 1e-3) / bkgrd.H(self.cosmo_jax, self.scale_fac_a_array)
        self.plin_kz_mat = vmap(linear_matter_power,(None, None, 0))(self.cosmo_jax, self.kPk_array, self.scale_fac_a_array).T

        self.chi_array_for_Cls = jnp.exp(jnp.interp(self.z_array_for_Cls, self.z_array, jnp.log(self.chi_array)))
        self.dchi_dz_array_for_Cls = jnp.exp(jnp.interp(self.z_array_for_Cls, self.z_array, jnp.log(self.dchi_dz_array)))
        self.growth_array_for_Cls = jnp.exp(jnp.interp(self.z_array_for_Cls, self.z_array, jnp.log(self.growth_array)))
        self.rhom_0 = self.get_rho_m(0.0)

    def get_rho_m(self, z):
        return (constants.RHO_CRIT_0_KPC3 * self.Om0 * (1.0 + z)**3) * 1E9


    def get_Ez(self, z):
        zp1 = (1.0 + z)
        t = (self.Om0) * zp1**3 + (1 - self.Om0)
        E = jnp.sqrt(t)        
        return E

    def get_rho_c(self, z):
        return constants.RHO_CRIT_0_KPC3 * self.get_Ez(z)**2  * 1E9    
