import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
import helpers.constants as constants
import astropy.units as u
import jax
from astropy import constants as const
import jax.scipy.integrate as jsi
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.keV / u.cm**3)).value
mp = (1.6726219e-27*u.kg).to(u.Msun).value
mue = 1.14
Mpc_to_cm = 3.086e24
import helpers.constants as constants
import jax_cosmo.background as bkgrd
import time
from jax_cosmo.scipy.integrate import romb, simps
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from base_class import base_class, get_vmapped_func, get_vmapped_func_warg
import interpax

class Profiles(base_class):
    """"
    Class to calculate the BCMP profile as described in BCM 2018 (Schneider et al.) by augmenting it to have pressure term
    """

    def __init__(
            self,
            sim_params_dict: dict,
            halo_params_dict: dict,
            analysis_dict: dict = None,     
            other_params_dict: dict = None,
            base_class_obj = None
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
        if base_class_obj is None:
            super().__init__(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
        else:
            self.__dict__.update(base_class_obj.__dict__)

        self.setup_hmf()
        self.get_hmf()
        self.get_conc_Mz()
        self.setup_main_calc()
        self.get_DMO_profiles()
        self.run_stars_calc()
        self.run_gas_calc()
        self.run_clm_calc()
        self.run_cga_calc()

        self.run_dmb_calc()
        if self.model_tSZ:
            self.run_pressure_calc()


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


    @timing_decorator
    def setup_hmf(self):
        """
        Setup the the halo mass function calculation by calculating the sigma_Mz matrix, dlg_sigma_dlnM matrix
        """
        self.sigma_Mz_mat = get_vmapped_func(self.get_sigma_Mz, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.nu_Mz_mat = constants.DELTA_COLLAPSE / self.sigma_Mz_mat
        grad_lgsigma = grad(self.get_lgsigma_z, argnums=1)
        self.dlgsig_dlnM_mat = get_vmapped_func(grad_lgsigma, 2)(jnp.arange(self.nz), jnp.log(self.M_array)).T

    @timing_decorator
    def get_hmf(self):
        """
        Get the halo mass function by using the sigma_Mz matrix and the dlgsig_dlnM matrix.
        """
        rhom_z_array = (constants.RHO_CRIT_0_KPC3 * self.cosmo_params['Om0'] * jnp.ones_like(self.z_array)) * 1E9
        rhom_z_mat = jnp.repeat(rhom_z_array[None, :], self.nM, axis=0)
        M_z_mat = jnp.repeat(self.M_array[:, None], self.nz, axis=1)
        if self.hmf_model == 'T08':
            self.fsigma_Mz_mat = get_vmapped_func(self.get_fsigma_Mz_T08, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        elif self.hmf_model == 'T10':
            self.fsigma_Mz_mat = get_vmapped_func(self.get_fsigma_Mz_T10, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        else:
            raise ValueError("HMF model not recognized")
        self.hmf_Mz_mat = -1 * self.fsigma_Mz_mat * (rhom_z_mat/M_z_mat).T * self.dlgsig_dlnM_mat

    @timing_decorator
    def get_conc_Mz(self):
        """
        Get the concentration-mass relation matrix
        """
        if self.conc_model == 'Prada12':
            self.conc_Mz_mat = get_vmapped_func(self.get_conc_Mz_Prada12, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        if self.conc_model == 'Duffy08':
            self.conc_Mz_mat = get_vmapped_func(self.get_conc_Mz_Duffy08, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        if self.conc_model == 'Diemer15':
            self.conc_Mz_mat = get_vmapped_func(self.get_conc_Mz_Diemer15, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T

    @timing_decorator
    def setup_main_calc(self):
        """
        Setup the main calculation by calculating things like r200c matrix, Mc matrix, beta matrix etc.
        """
        self.r200c_mat = get_vmapped_func(self.get_M_to_R, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.rt_mat = self.r200c_mat * self.epsilon_rt
        self.Mc_mat = get_vmapped_func(self.get_Mc, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.beta_mat = get_vmapped_func(self.get_beta, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.theta_co = get_vmapped_func(self.get_theta_co, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.theta_ej = get_vmapped_func(self.get_theta_ej, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.r_co_mat = self.theta_co * self.r200c_mat
        self.r_ej_mat = self.theta_ej * self.r200c_mat

    @timing_decorator
    def get_DMO_profiles(self):
        self.rho_nfw_unnorm_mat = get_vmapped_func(self.get_rho_nfw_unnorm, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.rho_nfw_norm_mat = get_vmapped_func(self.get_nfw_norm, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.rho_nfw_mat = get_vmapped_func(self.get_rho_nfw_normed, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.Mtot_mat = get_vmapped_func(self.get_Mtot, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T

    @timing_decorator
    def run_stars_calc(self):
        """
        Run the Ncen/Nsat calculation to get the stellar/galaxy counts.
        """
        if self.model_galaxies:
            self.Mthresh_array = vmap(self.get_Mthresh)(jnp.arange(self.nz))
            self.Ncen_mat = get_vmapped_func(self.get_Ncen, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
            self.Nsat_mat = get_vmapped_func(self.get_Nsat, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
            self.fstar_cen_mat = get_vmapped_func(self.get_fstar_cen, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
            self.fstar_sat_mat = get_vmapped_func(self.get_fstar_sat, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
            self.fstar_tot_mat = self.fstar_cen_mat + self.fstar_sat_mat
        else:
            self.fcga_array = self.A_starcga * ((self.M1_starcga / self.M_array) ** self.eta_cga)
            self.fstar_array = self.A_starcga * ((self.M1_starcga / self.M_array) ** self.eta_star)
            self.fstar_tot_mat = self.fstar_array[None, :] * jnp.ones((self.nz, self.nM))
            self.fstar_cen_mat = self.fcga_array[None, :] * jnp.ones((self.nz, self.nM))
            self.fstar_sat_mat = self.fstar_tot_mat - self.fstar_cen_mat

    @timing_decorator
    def run_gas_calc(self):
        """
        Run the gas profile calculation.
        """
        self.fgas_mat = (self.cosmo_params['Ob0'] / self.cosmo_params['Om0']) - self.fstar_tot_mat
        self.fclm_mat = (1 - self.cosmo_params['Ob0'] / self.cosmo_params['Om0']) + self.fstar_sat_mat
        self.Rh_mat = 0.015 * self.r200c_mat
        self.rho_gas_norm_mat = get_vmapped_func(self.get_rho_gas_norm, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.rho_gas_mat = get_vmapped_func(self.get_rho_gas_normed, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.rho_gas_mat_physical = self.rho_gas_mat / (self.scale_fac_a_array[None, :, None] ** 3)
        self.Mgas_mat = get_vmapped_func(self.get_Mgas, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        h = self.cosmo_params['H0'] / 100.
        self.ne_mat_physical = self.rho_gas_mat_physical/(mue*mp*(Mpc_to_cm**3)/(h**2)) # in cm**-3
        self.ne_mat = self.rho_gas_mat/(mue*mp*(Mpc_to_cm**3)/(h**2)) # in cm**-3
        self.ne_mat_norm = self.Mgas_mat/(mue*mp*(Mpc_to_cm**3)/(h**2)) # in cm**-3


    @timing_decorator
    def run_clm_calc(self):
        """
        Run the collision less matter profile calculation.
        """
        self.zeta_mat = get_vmapped_func(self.get_zeta, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.Mclm_mat = get_vmapped_func(self.get_Mclm, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        # self.rho_clm_mat = get_vmapped_func(self.get_rho_clm, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.rho_clm_mat = get_vmapped_func(self.get_rho_clm, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T

    @timing_decorator
    def run_cga_calc(self):
        """
        Run the central galaxy profile calculation.
        """
        self.rho_cga_mat = get_vmapped_func(self.get_rho_cga, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T

    @timing_decorator
    def run_dmb_calc(self):
        """
        Get the final total matter profiles
        """
        # self.rho_dmb_mat = get_vmapped_func(self.get_rho_dmb, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.rho_dmb_mat = self.rho_cga_mat + self.rho_clm_mat + self.rho_gas_mat
        # self.rho_dmb_mat_physical = self.rho_dmb_mat / (self.scale_fac_a_array[None, :, None] ** 3)
        self.Mdmb_mat = get_vmapped_func(self.get_Mdmb, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T

    @timing_decorator
    def run_pressure_calc(self):
        """
        Get the pressure profiles
        """
        Ptot_mat = get_vmapped_func(self.get_Ptot, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        # this was pressure in the comoving coordinates. Convert to physical coordinates:
        Ptot_mat_physical = Ptot_mat / (self.scale_fac_a_array[None, :, None] ** 3)
        Pnt_fac = get_vmapped_func(self.get_Pnt_fac, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        Pnt_mat = Pnt_fac * Ptot_mat
        # Pnt_mat_physical = Pnt_fac * Ptot_mat_physical
        Pth_mat = Ptot_mat * jnp.maximum(0, 1 - Pnt_fac)
        Pth_mat_physical = Ptot_mat_physical * jnp.maximum(0, 1 - Pnt_fac)        
        # this was thermal pressure. Convert to electron pressure using Xh=0.76 and dividing by 2*(Xh + 1)/(5*Xh + 3) ~ 1.932
        self.Pe_mat_physical = Pth_mat_physical/1.932
        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        coeff = sigmat / (m_e * (c ** 2))
        oneMpc = (((10 ** 6)) * (u.pc).to(u.m)) * (u.m)
        const_coeff = (((coeff * oneMpc).to(((u.cm ** 3) / u.keV))).value)/(self.cosmo_params['H0']/100.)
        self.y3d_mat = const_coeff * self.Pe_mat_physical


    @partial(jit, static_argnums=(0,))
    def get_M_to_R(self, jz, jM, mdef_delta=200):
        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(self.cosmo_jax,self.scale_fac_a_array[jz]) * 1e9
        rho_treshold = mdef_delta * rho_c_z
        R = (self.M_array[jM] * 3.0 / 4.0 / jnp.pi / rho_treshold)**(1.0 / 3.0)
        # convert to comoving coordinates
        R *= (1 + self.z_array[jz])
        return R


    @partial(jit, static_argnums=(0,))        
    def get_lgsigma_z(self, jz, lgM, kmin=0.0001, kmax=1000.0):
        M = jnp.exp(lgM)
        R = (3.0 * M / 4.0 / jnp.pi / self.get_rho_m(0.0))**(1.0 / 3.0)
        def int_sigma(logk):
            k = jnp.exp(logk)
            x = k * R
            w = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / (x * x * x)
            pkz = jnp.exp(jnp.interp(logk, jnp.log(self.kPk_array), jnp.log(self.plin_kz_mat[:, jz])))
            return k * (k * w) ** 2 * pkz

        # y = romb(int_sigma, jnp.log10(kmin), jnp.log10(kmax), divmax=3)
        y = simps(int_sigma, jnp.log10(kmin), jnp.log10(kmax), N=64) 
        return jnp.log(jnp.sqrt(y / (2.0 * jnp.pi**2.0)))

        
    @partial(jit, static_argnums=(0,))        
    def get_sigma_Mz(self, jz, jM, kmin=0.0001, kmax=1000.0):
        R = (3.0 * self.M_array[jM] / 4.0 / jnp.pi / self.get_rho_m(0.0))**(1.0 / 3.0)

        def int_sigma(logk):
            k = jnp.exp(logk)
            x = k * R
            w = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / (x * x * x)
            pkz = jnp.exp(jnp.interp(logk, jnp.log(self.kPk_array), jnp.log(self.plin_kz_mat[:, jz])))
            return k * (k * w) ** 2 * pkz

        # y = romb(int_sigma, jnp.log10(kmin), jnp.log10(kmax), divmax=3)
        y = simps(int_sigma, jnp.log10(kmin), jnp.log10(kmax), N=64)        
        return jnp.sqrt(y / (2.0 * jnp.pi**2.0))


    @partial(jit, static_argnums=(0,))
    def get_fsigma_Mz_T08(self, jz, jM, mdef_delta=200):
        '''Tinker 2008 mass function'''
        sigma = self.sigma_Mz_mat[jz, jM]
        z = self.z_array[jz]
        rho_treshold = mdef_delta * self.get_rho_c(z)
        Delta_m = round(rho_treshold / self.get_rho_m(z))

        fit_Delta = jnp.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
        fit_A0 = jnp.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
        fit_a0 = jnp.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
        fit_b0 = jnp.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
        fit_c0 = jnp.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
            
        
        A0 = jnp.interp(Delta_m, fit_Delta, fit_A0)
        a0 = jnp.interp(Delta_m, fit_Delta, fit_a0)
        b0 = jnp.interp(Delta_m, fit_Delta, fit_b0)
        c0 = jnp.interp(Delta_m, fit_Delta, fit_c0)
        
        alpha = 10**(-(0.75 / jnp.log10(Delta_m / 75.0))**1.2)
        A = A0 * (1.0 + z)**-0.14
        a = a0 * (1.0 + z)**-0.06
        b = b0 * (1.0 + z)**-alpha
        c = c0
        f = A * ((sigma / b)**-a + 1.0) * jnp.exp(-c / sigma**2)
        
        return f

    @partial(jit, static_argnums=(0,))
    def get_fsigma_Mz_T10(self, jz, jM, mdef_delta=200):
        '''Tinker 2010 mass function. Thanks to chto for the code'''
        sigma = self.sigma_Mz_mat[jz, jM]
        delta_c = constants.DELTA_COLLAPSE
        nu = delta_c / sigma
        z = self.z_array[jz]
        rho_treshold = mdef_delta * self.get_rho_c(z)
        Delta_m = round(rho_treshold / self.get_rho_m(z))
        fit_Delta = jnp.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
        fit_alpha = jnp.array([0.368, 0.363, 0.385, 0.389, 0.393, 0.365, 0.379, 0.355, 0.327])
        fit_beta = jnp.array([0.589, 0.585, 0.544, 0.543, 0.564, 0.623, 0.637, 0.673, 0.702])
        fit_gamma =  jnp.array([0.864, 0.922, 0.987, 1.09, 1.20, 1.34, 1.50, 1.68, 1.81])
        fit_phi = jnp.array([-0.729, -0.789, -0.910, -1.05, -1.20, -1.26, -1.45, -1.50, -1.49])
        fit_eta = jnp.array([-0.243, -0.261, -0.261, -0.273, -0.278, -0.301, -0.301, -0.319, -0.336])
        alpha = jnp.interp(Delta_m, fit_Delta, fit_alpha)
        beta = jnp.interp(Delta_m, fit_Delta, fit_beta)
        gamma = jnp.interp(Delta_m, fit_Delta, fit_gamma)
        phi = jnp.interp(Delta_m, fit_Delta, fit_phi)
        eta = jnp.interp(Delta_m, fit_Delta, fit_eta)


        beta = beta*(1+z)**0.2
        phi = phi*(1+z)**(-0.08)
        eta = eta*(1+z)**0.27
        gamma = gamma*(1+z)**(-0.01)
        fnu= alpha*(1+(beta*nu)**(-2.0*phi))*nu**(2*eta)*jnp.exp(-gamma*nu**2/2)
        return nu*fnu        
    


    @partial(jit, static_argnums=(0,))
    def get_Mc(self, jz, jM):
        value = self.Mc0 * jnp.power(((self.M_array[jM])/self.Mstar0), self.nu_M) * jnp.power((1 + (self.z_array[jz])), self.nu_z)
        return value


    @partial(jit, static_argnums=(0,))
    def get_conc_Mz_Duffy08(self, jz, jM):
        '''Duffy 2008 concentration relation, for mdef = 200c'''
        M = self.M_array[jM]
        z = self.z_array[jz]
        A = 5.71
        B = -0.084
        C = -0.47
        c = A * (M / 2E12)**B * (1.0 + z)**C
        return c
    
    @partial(jit, static_argnums=(0,))
    def get_conc_Mz_Prada12(self, jz, jM):
        '''Prada 2012 concentration relation, for mdef = 200c'''
        nu = self.nu_Mz_mat[jz, jM]
        z = self.z_array[jz]
        def cmin(x):
            return 3.681 + (5.033 - 3.681) * (1.0 / jnp.pi * jnp.arctan(6.948 * (x - 0.424)) + 0.5)
        def smin(x):
            return 1.047 + (1.646 - 1.047) * (1.0 / jnp.pi * jnp.arctan(7.386 * (x - 0.526)) + 0.5)

        a = 1.0 / (1.0 + z)
        x = ((1 - self.Om0) / self.Om0) ** (1.0 / 3.0) * a

        B0 = cmin(x) / cmin(1.393)
        B1 = smin(x) / smin(1.393)
        temp_sig = 1.686 / nu
        temp_sigp = temp_sig * B1
        temp_C = 2.881 * ((temp_sigp / 1.257) ** 1.022 + 1) * jnp.exp(0.06 / temp_sigp ** 2)
        c200c = B0 * temp_C
        return c200c    
    
    @partial(jit, static_argnums=(0,))    
    def get_conc_Mz_Diemer15(self, jz, jM):
        nu = self.nu_Mz_mat[jz, jM]
        z = self.z_array[jz]
        M = self.M_array[jM]
        DIEMER15_KAPPA = 1.00
        # R = peaks.lagrangianR(M)
        rho_m = (constants.RHO_CRIT_0_KPC3 * self.Om0) * 1E9
        R = (3.0 * M / 4.0 / jnp.pi / rho_m )**(1.0 / 3.0)
        k_R = 2.0 * jnp.pi / R * DIEMER15_KAPPA        

        interp = InterpolatedUnivariateSpline(jnp.log10(self.kPk_array), jnp.log10(self.plin_kz_mat[:, 0]))
        n = interp.derivative(jnp.log10(k_R), n = 1)

        DIEMER15_MEDIAN_PHI_0 = 6.58
        DIEMER15_MEDIAN_PHI_1 = 1.27
        DIEMER15_MEDIAN_ETA_0 = 7.28
        DIEMER15_MEDIAN_ETA_1 = 1.56
        DIEMER15_MEDIAN_ALPHA = 1.08
        DIEMER15_MEDIAN_BETA  = 1.77

        floor = DIEMER15_MEDIAN_PHI_0 + n * DIEMER15_MEDIAN_PHI_1
        nu0 = DIEMER15_MEDIAN_ETA_0 + n * DIEMER15_MEDIAN_ETA_1
        alpha = DIEMER15_MEDIAN_ALPHA
        beta = DIEMER15_MEDIAN_BETA

        c = 0.5 * floor * ((nu0 / nu)**alpha + (nu / nu0)**beta)

        return c


    @partial(jit, static_argnums=(0,))
    def get_theta_ej(self, jz, jM):
        value = self.theta_ej_0 * jnp.power(((self.M_array[jM])/10**self.log10_Mstar0_theta_ej), self.nu_theta_ej_M) * jnp.power((1 + (self.z_array[jz])), self.nu_theta_ej_z)
        return value

    @partial(jit, static_argnums=(0,))
    def get_theta_co(self, jz, jM):
        value = self.theta_co_0 * jnp.power(((self.M_array[jM])/10**self.log10_Mstar0_theta_co), self.nu_theta_co_M) * jnp.power((1 + (self.z_array[jz])), self.nu_theta_co_z)
        return value                        

    @partial(jit, static_argnums=(0,))
    def get_beta(self, jz, jM):
        value = 3*jnp.power(self.M_array[jM]/self.Mc_mat[jz, jM],self.mu_beta)/(1 + jnp.power(self.M_array[jM]/self.Mc_mat[jz, jM],self.mu_beta))
        return value

    @partial(jit, static_argnums=(0,))
    def get_rho_nfw_unnorm(self, jr, jz, jM, r_array_here=None):
        '''This is the NFW profile (Eq.2.18)'''
        r200c = self.r200c_mat[jz, jM]
        rt = self.rt_mat[jz, jM]
        conc = self.conc_Mz_mat[jz, jM]
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        rs = r200c / conc
        x = r / rs
        y = r / rt
        if self.nfw_trunc:
            rho_nfw = (1 / (x * (1 + x)**2)) * (1 / (1 + y**2)**2)
        else:
            rho_nfw = 1 / (x * (1 + x)**2)
        return rho_nfw

    @partial(jit, static_argnums=(0,))
    def get_nfw_norm(self, jz, jM):
        '''This is the normalization of the NFW profile'''
        r200c = self.r200c_mat[jz, jM]
        M200c = self.M_array[jM]
        logx = jnp.linspace(jnp.log(0.01*r200c), jnp.log(r200c), self.num_points_trapz_int)
        int_unnorm_prof = self.logspace_trapezoidal_integral(self.get_rho_nfw_unnorm, logx, jz=jz, jM=jM, axis_tup=(0, None, None, None))
        rho_nfw_0 = M200c / int_unnorm_prof
        return rho_nfw_0

    @partial(jit, static_argnums=(0,))
    def get_rho_nfw_normed(self, jr, jz, jM, r_array_here=None):
        '''This is the NFW profile (Eq.2.18)'''
        # r200c = self.r200c_mat[jz, jM]
        # rt = self.rt_mat[jz, jM]
        # conc = self.conc_Mz_mat[jz, jM]
        # if r_array_here is None:
        #     r = self.r_array[jr]
        # else:
        #     r = r_array_here[jr]
        # rs = r200c / conc
        # x = r / rs
        # y = r / rt
        # if self.nfw_trunc:
        #     rho_nfw = (1 / (x * (1 + x)**2)) * (1 / (1 + y**2)**2)
        # else:
        #     rho_nfw = 1 / (x * (1 + x)**2)
        rho_nfw = self.get_rho_nfw_unnorm(jr, jz, jM, r_array_here=r_array_here)
        prefac = self.rho_nfw_norm_mat[jz, jM]
        return prefac * rho_nfw
    
    @partial(jit, static_argnums=(0,))
    def get_Mtot(self, jz, jM, rmax_r200c=16):
        '''This is the total mass of all matter '''
        r200c = self.r200c_mat[jz, jM]
        logx = jnp.linspace(jnp.log(0.01*r200c), jnp.log(rmax_r200c*r200c), self.num_points_trapz_int)
        Mtot = self.logspace_trapezoidal_integral(self.get_rho_nfw_normed, logx, jz=jz, jM=jM, axis_tup=(0, None, None, None))
        return Mtot


    @partial(jit, static_argnums=(0,))
    def get_Mnfw(self, jr, jz, jM, r_array_here=None):
        '''This is the mass of the NFW profile'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jz, jM])
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        Mnfw = self.logspace_trapezoidal_integral(self.get_rho_nfw_normed, logx, jz=jz, jM=jM, axis_tup=(0, None, None, None))
        return Mnfw

    @partial(jit, static_argnums=(0,))
    def get_Mh_Mstar(self, jz, jM, Mstar_array=None): 
        npoints = self.num_points_gal_cal
        aval = self.scale_fac_a_array[jz]
        log10M1 = self.log10M1_fshmr + self.log10M1_a_fshmr * (aval - 1)
        Mstar0 = 10**(self.log10Mstar0_fshmr + self.log10Mstar0_a_fshmr * (aval - 1))
        beta = self.beta_fshmr + self.beta_a_fshmr * (aval - 1)
        delta = self.delta_fshmr + self.delta_a_fshmr * (aval - 1)
        gamma = self.gamma_fshmr + self.gamma_a_fshmr * (aval - 1)

        if Mstar_array is None:
            Mstar_array = jnp.logspace(8, 14, npoints)
        log10Mh = log10M1 + beta * jnp.log10(Mstar_array / Mstar0) + ((Mstar_array/Mstar0)**delta)/(1 + (Mstar_array/Mstar0)**(-gamma)) - 0.5
        return 10**log10Mh


    @partial(jit, static_argnums=(0,))
    def get_Mstar_Mh(self, jz, jM): 
        npoints = self.num_points_gal_cal        
        Mval = self.M_array[jM]
        Mval_no_h = Mval/self.h
        Mstar_array = jnp.logspace(8, 14, npoints)
        log10Mh = jnp.log10(self.get_Mh_Mstar(jz, jM))
        log10Mstar_Mh = jnp.interp(jnp.log10(Mval_no_h),log10Mh,jnp.log10(Mstar_array))
        Mstar_Mh = 10**log10Mstar_Mh
        Mstar_wh = Mstar_Mh * self.h
        return Mstar_wh

    @partial(jit, static_argnums=(0,))
    def get_Ncen(self, jz, jM):
        log10mthresh = jnp.log10(self.Mthresh_array[jz])
        log10Mstar = jnp.log10(self.get_Mstar_Mh(jz, jM))
        num = log10mthresh - log10Mstar
        denom = jnp.sqrt(2) * self.siglogMstar_Ncen
        val = 0.5 * (1 - jax.lax.erf(num / denom))
        return val

    @partial(jit, static_argnums=(0,))
    def get_Nsat(self, jz, jM):
        log10mthresh = jnp.log10(self.Mthresh_array[jz])
        Mval = self.M_array[jM]
        Mh_Mthresh = self.get_Mh_Mstar(jz, jM, Mstar_array=10**log10mthresh)
        Msat = (1e12 * self.h) * self.Bsat_Nsat * (Mh_Mthresh / 1e12)**self.betasat_Nsat
        Mcut = (1e12 * self.h) * self.Bcut_Nsat * (Mh_Mthresh / 1e12)**self.betacut_Nsat
        Ncen = self.get_Ncen(jz, jM)
        val = Ncen * ((Mval / Msat)**self.alphasat_Nsat) * jnp.exp(-(Mcut / Mval))
        return val

    @partial(jit, static_argnums=(0,))
    def get_Mthresh(self, jz):
        npoints = self.num_points_gal_cal
        nbar_inp = self.nbar_gal_comoving_array[jz]

        def get_Ncen(jz, jM, log10mthresh):
            log10Mstar = jnp.log10(self.get_Mstar_Mh(jz, jM))
            num = log10mthresh - log10Mstar
            denom = jnp.sqrt(2) * self.siglogMstar_Ncen
            val = 0.5 * (1 - jax.lax.erf(num / denom))
            return val

        def get_Nsat(jz, jM, log10mthresh):
            Mval = self.M_array[jM]
            Mh_Mthresh = self.get_Mh_Mstar(jz, jM, Mstar_array=10**log10mthresh)
            Msat = (1e12 * self.h) * self.Bsat_Nsat * (Mh_Mthresh / 1e12)**self.betasat_Nsat
            Mcut = (1e12 * self.h) * self.Bcut_Nsat * (Mh_Mthresh / 1e12)**self.betacut_Nsat
            Ncen = get_Ncen(jz, jM, log10mthresh)
            val = Ncen * ((Mval / Msat)**self.alphasat_Nsat) * jnp.exp(-(Mcut / Mval))
            return val

        Mthresh_array = jnp.logspace(9, 13, npoints)
        Ncen_mat = get_vmapped_func(get_Ncen, 3)(jnp.array([jz]), jnp.arange(len(self.M_array)), jnp.log10(Mthresh_array)).T
        Nsat_mat = get_vmapped_func(get_Nsat, 3)(jnp.array([jz]), jnp.arange(len(self.M_array)), jnp.log10(Mthresh_array)).T
        Ntot_mat = (Ncen_mat + Nsat_mat)[0,...]
        dndlogM = self.hmf_Mz_mat[jz,:][:, None]
        nbar = jsi.trapezoid(dndlogM * Ntot_mat, x=jnp.log(self.M_array), axis=0)

        func = nbar_inp - nbar
        log10Mthresh = jnp.interp(0, func, jnp.log10(Mthresh_array))
        return 10**log10Mthresh

    @partial(jit, static_argnums=(0,))
    def get_fstar_cen(self, jz, jM):
        npoints = self.num_points_gal_cal        
        def get_Ncen(jz, jM, log10mthresh):
            log10Mstar = jnp.log10(self.get_Mstar_Mh(jz, jM))
            num = log10mthresh - log10Mstar
            denom = jnp.sqrt(2) * self.siglogMstar_Ncen
            val = 0.5 * (1 - jax.lax.erf(num / denom))
            return val

        log10mthresh = jnp.log10(self.Mthresh_array[jz])
        Mthresh_array = jnp.logspace(log10mthresh, 14, npoints)
        Ncen_mat = get_vmapped_func(get_Ncen, 3)(jnp.array([jz]), jnp.array([jM]), jnp.log10(Mthresh_array)).T

        val1 = (Ncen_mat[0,0,-1]*Mthresh_array[-1] - Ncen_mat[0,0,0]*Mthresh_array[0])
        val2 = jsi.trapezoid(Ncen_mat[0,0,:]*Mthresh_array, x=jnp.log10(Mthresh_array))
        Mstar_cen = val2 - val1
        Mtot = self.Mtot_mat[jz, jM]
        return Mstar_cen / Mtot

    @partial(jit, static_argnums=(0,))
    def get_fstar_sat(self, jz, jM):
        npoints = self.num_points_gal_cal        
        def get_Ncen(jz, jM, log10mthresh):
            log10Mstar = jnp.log10(self.get_Mstar_Mh(jz, jM))
            num = log10mthresh - log10Mstar
            denom = jnp.sqrt(2) * self.siglogMstar_Ncen
            val = 0.5 * (1 - jax.lax.erf(num / denom))
            return val

        def get_Nsat(jz, jM, log10mthresh):
            Mval = self.M_array[jM]
            Mh_Mthresh = self.get_Mh_Mstar(jz, jM, Mstar_array=10**log10mthresh)
            Msat = (1e12 * self.h) * self.Bsat_Nsat * (Mh_Mthresh / 1e12)**self.betasat_Nsat
            Mcut = (1e12 * self.h) * self.Bcut_Nsat * (Mh_Mthresh / 1e12)**self.betacut_Nsat
            Ncen = get_Ncen(jz, jM, log10mthresh)
            val = Ncen * ((Mval / Msat)**self.alphasat_Nsat) * jnp.exp(-(Mcut / Mval))
            return val

        log10mthresh = jnp.log10(self.Mthresh_array[jz])
        Mthresh_array = jnp.logspace(log10mthresh, 14, npoints)
        Nsat_mat = get_vmapped_func(get_Nsat, 3)(jnp.array([jz]), jnp.array([jM]), jnp.log10(Mthresh_array)).T

        val1 = (Nsat_mat[0,0,-1]*Mthresh_array[-1] - Nsat_mat[0,0,0]*Mthresh_array[0])
        val2 = jsi.trapezoid(Nsat_mat[0,0,:]*Mthresh_array, x=jnp.log10(Mthresh_array))
        Mstar_sat = val2 - val1
        Mtot = self.Mtot_mat[jz, jM]
        return Mstar_sat / Mtot

    @partial(jit, static_argnums=(0,))
    def get_rho_cga(self, jr, jz, jM, r_array_here=None):
        ''' This is central galaxy profile (Eq.2.10)'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        rho_cga = (self.fstar_cen_mat[jz, jM] * self.Mtot_mat[jz, jM]) / (4 * (jnp.pi**1.5) * self.Rh_mat[jz, jM] * r**2) * jnp.exp(-(0.5 * r / self.Rh_mat[jz, jM])**2)
        return rho_cga

    @partial(jit, static_argnums=(0,))
    def get_Mcga(self, jr, jz, jM, r_array_here=None):
        '''This is the mass of the central galaxy profile'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jz, jM])        
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        Mcga = self.logspace_trapezoidal_integral(self.get_rho_cga, logx, jz=jz, jM=jM, axis_tup=(0, None, None, None))
        return Mcga

    
    @partial(jit, static_argnums=(0,))
    def get_rho_gas_unnorm(self, jr, jz, jM, r_array_here=None):
        '''
        This is the gas profile (Eq.2.12)
        '''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        u = r / self.r_co_mat[jz, jM]
        v = r / self.r_ej_mat[jz, jM]

        # y = r / self.rt_mat[jz, jM]
        # fac = (1 / (1 + y**2)**2)

        rho_gas_unnorm = 1 / (jnp.power(1 + u, self.beta_mat[jz, jM]) * jnp.power(1 + jnp.power(v, self.gamma_rhogas), (self.delta_rhogas - self.beta_mat[jz, jM]) / self.gamma_rhogas))
        return rho_gas_unnorm    

    @partial(jit, static_argnums=(0,))
    def get_rho_gas_norm(self, jz, jM, rmax_r200c=12):
        '''This is the normalization of the gas profile'''
        r200c = self.r200c_mat[jz, jM]
        logx = jnp.linspace(jnp.log(0.01*r200c), jnp.log(rmax_r200c*r200c), self.num_points_trapz_int)
        # logx = jnp.linspace(jnp.log(0.01*r200c), jnp.log(self.r_array[-1]), self.num_points_trapz_int)        
        int_unnorm_prof = self.logspace_trapezoidal_integral(self.get_rho_gas_unnorm, logx, jz=jz, jM=jM, axis_tup=(0, None, None, None))
        rho_gas_norm = self.fgas_mat[jz, jM] * self.Mtot_mat[jz, jM] / int_unnorm_prof
        return rho_gas_norm


    @partial(jit, static_argnums=(0,))
    def get_rho_gas_normed(self, jr, jz, jM, r_array_here=None):
        '''This is the NFW profile (Eq.2.18)'''
        # if r_array_here is None:
        #     r = self.r_array[jr]
        # else:
        #     r = r_array_here[jr]

        # u = r / self.r_co_mat[jz, jM]
        # v = r / self.r_ej_mat[jz, jM]
        
        # # y = r / self.rt_mat[jz, jM]
        # # fac = (1 / (1 + y**2)**2)
        
        # rho_gas_unnorm = 1 / (jnp.power(1 + u, self.beta_mat[jz, jM]) * jnp.power(1 + jnp.power(v, self.gamma_rhogas), (self.delta_rhogas - self.beta_mat[jz, jM]) / self.gamma_rhogas))
        rho_gas_unnorm = self.get_rho_gas_unnorm(jr, jz, jM, r_array_here=r_array_here)
        prefac = self.rho_gas_norm_mat[jz, jM]
        return prefac * rho_gas_unnorm

    @partial(jit, static_argnums=(0,))
    def get_Mgas(self, jr, jz, jM, r_array_here=None):
        '''This is the mass of the gas profile'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jz, jM])        
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        Mgas = self.logspace_trapezoidal_integral(self.get_rho_gas_normed, logx, jz=jz, jM=jM, axis_tup=(0, None, None, None))
        return Mgas

    @partial(jit, static_argnums=(0,))
    def get_zeta(self, jr, jz, jM, r_array_here=None):
        '''This requires solving the equation iteratively. 
        The main equation is: (rf/ri - 1) - a*((Mi/Mf)**n - 1) = 0
        where, things to solve for is zeta = rf/ri
        Here, Mi = M_nfw(ri)
        and, Mf = fclm * M_nfw(ri) + M_cga(rf) + M_gas(rf)
        '''
        if r_array_here is None:
            ri = self.r_array[jr]
        else:
            ri = r_array_here[jr]
        Mi = self.get_Mnfw(jr, jz, jM, r_array_here=r_array_here)

        def zeta_equation(zeta):
            rf = zeta * ri
            Mf = self.fclm_mat[jz, jM] * Mi + self.get_Mcga(0, jz, jM, r_array_here=jnp.array([rf])) + self.get_Mgas(0, jz, jM, r_array_here=jnp.array([rf]))
            return ((rf / ri - 1) - self.a_zeta * ((Mi / Mf)**self.n_zeta - 1))
        zeta_array = jnp.linspace(0.5, 1.5, 32)
        value_out = vmap(zeta_equation)(zeta_array)
        zeta = jnp.interp(0.0, value_out, zeta_array)
        return zeta


    # @partial(jit, static_argnums=(0,))
    # def get_rho_clm(self, jr, jz, jM, r_array_here=None):
    #     if r_array_here is None:
    #         r_array_here = self.r_array
    #     zeta = (jnp.interp(jnp.log(r_array_here[jr]), jnp.log(self.r_array), self.zeta_mat[:,jz, jM]))
    #     if r_array_here is None:
    #         r_array_new = self.r_array/zeta        
    #     else:
    #         r_array_new = r_array_here/zeta
        
    #     rho_nfw = self.get_rho_nfw_normed(jr, jz, jM, r_array_new)
    #     rho_clm = (self.fclm_mat[jz, jM] / (zeta**3)) * rho_nfw
    #     return rho_clm

    @partial(jit, static_argnums=(0,))
    def get_rho_clm(self, jz, jM, r_array_here=None):
        if r_array_here is None:
            r_array_here = self.r_array
            Mclm_here = self.Mclm_mat[:, jz, jM]
        else:
            Mclm_here = jnp.exp(jnp.interp(jnp.log(r_array_here), jnp.log(self.r_array), jnp.log(self.Mclm_mat[:, jz, jM])))
        # zeta = (jnp.interp(jnp.log(r_array_here[jr]), jnp.log(self.r_array), self.zeta_mat[:,jz, jM]))
        # if r_array_here is None:
        #     r_array_new = self.r_array/zeta        
        # else:
        #     r_array_new = r_array_here/zeta
        
        lnMclm_interp = interpax.CubicSpline(jnp.log(r_array_here), jnp.log(Mclm_here + 1e-30), extrapolate=True, check=False)
        dlnMclm_dr = lnMclm_interp.derivative(nu=1)(jnp.log(r_array_here))
        dMclm_dr = dlnMclm_dr * Mclm_here / r_array_here
        rho_clm = dMclm_dr / (4*jnp.pi*r_array_here**2)      
        # rho_nfw = self.get_rho_nfw_normed(jr, jz, jM, r_array_new)
        # rho_clm = (self.fclm_mat[jz, jM] / (zeta**3)) * rho_nfw
        return jnp.clip(rho_clm, 0, 1e30)


    @partial(jit, static_argnums=(0,))
    def get_Mclm(self, jr, jz, jM, r_array_here=None):
        '''Collison less matter profile (includes dark matter and satellite galaxies)'''
        if r_array_here is None:
            r_array_here = self.r_array
            zeta = (jnp.interp(jnp.log(r_array_here[jr]), jnp.log(self.r_array), self.zeta_mat[:,jz, jM]))
        else:
            zeta = self.zeta_mat[:,jz, jM]
        if r_array_here is None:
            r_array_new = self.r_array/zeta        
        else:
            r_array_new = r_array_here/zeta

        M_clm = self.fclm_mat[jz, jM] * self.get_Mnfw(jr, jz, jM, r_array_new)
        return M_clm


    # @partial(jit, static_argnums=(0,))
    # def get_rho_dmb(self, jr, jz, jM, r_array_here=None):
    #     '''This is the total matter profile with all the components (Eq.2.2)'''    
    #     rho_dmb = self.get_rho_gas_normed(jr, jz, jM, r_array_here=r_array_here) + \
    #         self.get_rho_cga(jr, jz, jM, r_array_here=r_array_here) + self.get_rho_clm(jr, jz, jM, r_array_here=r_array_here)
    #     return rho_dmb

    @partial(jit, static_argnums=(0,))
    def get_rho_dmb(self, jr, jz, jM, r_array_here=None):
        '''This is the total matter profile with all the components (Eq.2.2)'''    
        rho_dmb = self.get_rho_gas_normed(jr, jz, jM, r_array_here=r_array_here) + \
            self.get_rho_cga(jr, jz, jM, r_array_here=r_array_here) + self.get_rho_clm(jz, jM, r_array_here=r_array_here)[jr]
        return rho_dmb

    @partial(jit, static_argnums=(0,))
    def get_Mdmb(self, jr, jz, jM, r_array_here=None):
        '''This is the mass inside some radius for the full dmb profile'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jz, jM])
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        Mdmb = self.logspace_trapezoidal_integral(self.get_rho_dmb, logx, jz=jz, jM=jM, axis_tup=(0, None, None, None))
        return Mdmb
    
    @partial(jit, static_argnums=(0,))
    def get_Mdmb_r200(self, jz, jM):
        '''This is the mass inside some radius for the full dmb profile'''
        r = self.r200c_mat[jz, jM]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jz, jM])        
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        Mdmb = self.logspace_trapezoidal_integral(self.get_rho_dmb, logx, jz=jz, jM=jM, axis_tup=(0, None, None, None))
        return Mdmb

    @partial(jit, static_argnums=(0,))
    def get_r500_z0_wMdmb(self, jz, jM):
        '''This is the mass inside some radius for the full dmb profile'''
        r = self.r200c_mat[jz, jM]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jz, jM])        
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        Mdmb = self.logspace_trapezoidal_integral(self.get_rho_dmb, logx, jz=jz, jM=jM, axis_tup=(0, None, None, None))
        return Mdmb


    @partial(jit, static_argnums=(0,))
    def get_Ptot(self, jr, jz, jM, r_array_here=None, rmax_r200c=6):
        '''This is the total pressure profile, assuming HSE'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        logx = jnp.linspace(jnp.log(r), jnp.log(rmax_r200c*self.r200c_mat[jz, jM]), self.num_points_trapz_int)
        x = jnp.exp(logx)
        fx1 = (vmap(self.get_rho_gas_normed, (0, None, None,None))(jnp.arange(len(logx)), jz, jM, x))
        fx2 = jnp.exp(jnp.interp(logx, jnp.log(self.r_array), jnp.log(self.Mdmb_mat[:,jz, jM])))
        fx = (fx1 * fx2 * G_new / x**2) * x
        Ptot = jsi.trapezoid(fx, x=logx)
        # there is a factor of h^2 as dP = -G * rho_g * M(<r)/r^2 dr ~ G * M^2/r^4 and both mass and r are in the units of little h
        Ptot = jnp.clip(Ptot, 1e-30) * (self.cosmo_params['H0'] / 100.)**2
        return Ptot
    
    @partial(jit, static_argnums=(0,))
    def get_fz_Pnt(self, jz, rmax_r200c=6):
        '''This is the evolution of non-thermal pressure with redshift'''
        fmax = (rmax_r200c)**(-1 * self.n_nt) / self.alpha_nt
        fz = jnp.minimum((1 + self.z_array[jz])**self.beta_nt, (fmax - 1) * jnp.tanh(self.beta_nt * self.z_array[jz]) + 1)
        return fz

    @partial(jit, static_argnums=(0,))
    def get_Pnt_fac(self, jr, jz, jM, r_array_here=None):
        '''This is the non-thermal pressure profile'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        Pnt_fac = self.alpha_nt * self.get_fz_Pnt(jz) * ((r / self.r200c_mat[jz, jM])**self.n_nt)
        return Pnt_fac

