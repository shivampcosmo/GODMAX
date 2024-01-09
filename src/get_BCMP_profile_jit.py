from jax.lib import xla_bridge
platform = xla_bridge.get_backend().platform
import jax
jax.config.update('jax_platform_name', platform)
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from jaxopt import Bisection
from functools import partial
import constants
import astropy.units as u
from astropy import constants as const
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.keV / u.cm**3)).value
mp = (1.6726219e-27*u.kg).to(u.Msun).value
mue = 1.14
Mpc_to_cm = 3.086e24
import jax_cosmo.background as bkgrd
import time
from jax_cosmo import Cosmology
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline


class BCM_18_wP:

    def __init__(
            self,
            sim_params_dict,
            halo_params_dict,
            num_points_trapz_int=64,
            verbose_time=False            
        ):

        if verbose_time:
            ti = time.time()
        cosmo_params = sim_params_dict['cosmo']
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

        self.nfw_trunc = sim_params_dict.get('nfw_trunc',True)

        
        self.theta_ej_0=sim_params_dict.get('theta_ej_0',4.0)
        self.log10_Mstar0_theta_ej=sim_params_dict.get('log10_Mstar0_theta_ej',14.0)
        self.nu_theta_ej_M=sim_params_dict.get('nu_theta_ej_M',0.0)
        self.nu_theta_ej_z=sim_params_dict.get('nu_theta_ej_z',0.0)
        self.nu_theta_ej_c=sim_params_dict.get('nu_theta_ej_c',0.0)        

        self.theta_co_0=sim_params_dict.get('theta_co_0',0.1)
        self.log10_Mstar0_theta_co=sim_params_dict.get('log10_Mstar0_theta_co',14.0)
        self.nu_theta_co_M=sim_params_dict.get('nu_theta_co_M',0.0)
        self.nu_theta_co_z=sim_params_dict.get('nu_theta_co_z',0.0)                
        self.nu_theta_co_c=sim_params_dict.get('nu_theta_co_c',0.0)                        

        self.neg_bhse_plus_1=sim_params_dict.get('neg_bhse_plus_1',0.833)
        self.mu_beta=sim_params_dict.get('mu_beta',0.21)
        self.eta_star=sim_params_dict.get('eta_star',0.3)
        self.eta_cga=sim_params_dict.get('eta_cga',0.6)
        self.A_starcga=sim_params_dict.get('A_starcga',0.09)
        log10_M1_starcga=sim_params_dict.get('log10_M1_starcga',11.4)
        self.M1_starcga=10**log10_M1_starcga
        self.epsilon_rt=sim_params_dict.get('epsilon_rt',4.0)
        log10_Mc0 = sim_params_dict.get('log10_Mc0',14.83)
        self.Mc0 = 10**log10_Mc0
        self.nu_z = sim_params_dict.get('nu_z',0.0)
        self.nu_M = sim_params_dict.get('nu_M',0.0)
        log10_Mstar0 = sim_params_dict.get('log10_Mstar0',13.0)
        self.Mstar0 = 10**log10_Mstar0
        self.a_zeta=sim_params_dict.get('a_zeta',0.3)
        self.n_zeta=sim_params_dict.get('n_zeta',2.0)
        self.alpha_nt = sim_params_dict.get('alpha_nt',0.18)
        self.beta_nt = sim_params_dict.get('beta_nt',0.5)
        self.n_nt = sim_params_dict.get('n_nt',0.3)
        self.gamma_rhogas = sim_params_dict.get('gamma_rhogas', 2.)
        self.delta_rhogas = sim_params_dict.get('delta_rhogas', 7.)

        self.num_points_trapz_int = num_points_trapz_int


        rmin, rmax, nr = halo_params_dict.get('rmin', 5e-3), halo_params_dict.get('rmax',3), halo_params_dict.get('nr', 63)
        zmin, zmax, nz = halo_params_dict.get('zmin', 1e-3), halo_params_dict.get('zmax',1.5), halo_params_dict.get('nz',32)
        lg10_Mmin, lg10_Mmax, nM = halo_params_dict.get('lg10_Mmin', 12), halo_params_dict.get('lg10_Mmax', 15.0), halo_params_dict.get('nM', 32)
        cmin, cmax, nc = halo_params_dict.get('cmin',2), halo_params_dict.get('cmax',8), halo_params_dict.get('nc',32)
        self.r_array = jnp.logspace(jnp.log10(rmin), jnp.log10(rmax), nr)
        if 'z_array' in halo_params_dict.keys():
            self.z_array = jnp.array(halo_params_dict['z_array'])
            nz = len(self.z_array)
        else:
            self.z_array = jnp.linspace(zmin, zmax, nz)
        self.scale_fac_a_array = 1./(1. + self.z_array)
        self.M_array = jnp.logspace(lg10_Mmin, lg10_Mmax, nM)
        self.conc_array = jnp.exp(jnp.linspace(jnp.log(cmin), jnp.log(cmax), nc))
        
        vmap_func1 = vmap(self.get_M_to_R, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.r200c_mat = vmap_func2(jnp.arange(nM), jnp.arange(nz)).T
        # add a concentration axis to r200c_mat as the first axis. i.e. repeat along the first axis:
        self.r200c_mat_repeat = jnp.repeat(self.r200c_mat[None, :, :], nc, axis=0)

        self.rt_mat = self.r200c_mat * self.epsilon_rt

        vmap_func1 = vmap(self.get_Mc, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Mc_mat = vmap_func2(jnp.arange(nM),jnp.arange(nz)).T

        vmap_func1 = vmap(self.get_beta, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.beta_mat = vmap_func2(jnp.arange(nM), jnp.arange(nz)).T


        vmap_func1 = vmap(self.get_theta_co, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.theta_co = vmap_func3(jnp.arange(nc), jnp.arange(nM),jnp.arange(nz)).T        

        vmap_func1 = vmap(self.get_theta_ej, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.theta_ej = vmap_func3(jnp.arange(nc), jnp.arange(nM),jnp.arange(nz)).T                

        self.r_co_mat = self.theta_co * self.r200c_mat_repeat
        self.r_ej_mat = self.theta_ej * self.r200c_mat_repeat
        self.fstar_array = self.A_starcga * ((self.M1_starcga / self.M_array) ** self.eta_star)
        self.fgas_array = (cosmo_params['Ob0'] / cosmo_params['Om0']) - self.fstar_array
        self.fcga_array = self.A_starcga * ((self.M1_starcga / self.M_array) ** self.eta_cga)
        self.fclm_array = (1 - cosmo_params['Ob0'] / cosmo_params['Om0']) + self.fstar_array - self.fcga_array
        self.Rh_mat = 0.015 * self.r200c_mat
        
        vmap_func1 = vmap(self.get_rho_nfw_unnorm, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.rho_nfw_unnorm_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        vmap_func1 = vmap(self.get_nfw_norm, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.rho_nfw_norm_mat = vmap_func3(jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        vmap_func1 = vmap(self.get_rho_nfw_normed, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.rho_nfw_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        vmap_func1 = vmap(self.get_Mtot, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.Mtot_mat = vmap_func3(jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        vmap_func1 = vmap(self.get_rho_gas_norm, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.rho_gas_norm_mat = vmap_func3(jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        vmap_func1 = vmap(self.get_rho_gas_normed, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.rho_gas_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        self.rho_gas_mat_physical = self.rho_gas_mat / (self.scale_fac_a_array[None, None, :, None] ** 3)

        vmap_func1 = vmap(self.get_zeta, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.zeta_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        vmap_func1 = vmap(self.get_Mclm, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.Mclm_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T        


        vmap_func1 = vmap(self.get_rho_clm, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.rho_clm_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T        

        vmap_func1 = vmap(self.get_rho_cga, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.rho_cga_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T        

        vmap_func1 = vmap(self.get_rho_dmb, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.rho_dmb_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        self.rho_dmb_mat_physical = self.rho_dmb_mat / (self.scale_fac_a_array[None, None, :, None] ** 3)

        vmap_func1 = vmap(self.get_Mdmb, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.Mdmb_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        vmap_func1 = vmap(self.get_Mdmb_r200, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.Mdmb_r200_mat = vmap_func3(jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T        

        vmap_func1 = vmap(self.get_Ptot, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.Ptot_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        # this was pressure in the comoving coordinates. Convert to physical coordinates:
        # This comes because dP/dr = -G * rho_gas * M(<r) / r**2
        # So it will simplify to P ~ rho_g/r and when we will convert rho_g and r to physical coordinates, we will get a**4 factor
        self.Ptot_mat_physical = self.Ptot_mat / (self.scale_fac_a_array[None, None, :, None] ** 4)

        vmap_func1 = vmap(self.get_Pnt_fac, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.Pnt_fac = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        self.Pnt_mat = self.Pnt_fac * self.Ptot_mat
        self.Pnt_mat_physical = self.Pnt_fac * self.Ptot_mat_physical
        self.Pth_mat = self.Ptot_mat * jnp.maximum(0, 1 - self.Pnt_fac)
        self.Pth_mat_physical = self.Ptot_mat_physical * jnp.maximum(0, 1 - self.Pnt_fac)
        
        # this was thermal pressure. Convert to electron pressure using Xh=0.76 and dividing by 2*(Xh + 1)/(5*Xh + 3) ~ 1.932
        self.Pe_mat_physical = self.Pth_mat_physical/1.932
        h = cosmo_params['H0'] / 100.
        self.ne_mat_physical = self.rho_gas_mat_physical/(mue*mp*(Mpc_to_cm**3)/(h**2)) # in cm**-3

        if verbose_time:
            tf = time.time()
            print('Time taken to calculate BCMP profile: ', tf-ti, ' seconds')

    def logspace_trapezoidal_integral(self, f, logx, jc=None, jz=None, jM=None, axis_tup=(0, None, None, None)):
        x = jnp.exp(logx)
        if jc is None:
            fx = (vmap(f, axis_tup)(jnp.arange(len(logx)), jz, jM, x))*(4*jnp.pi*x**2) * x
        else:
            fx = (vmap(f, axis_tup)(jnp.arange(len(logx)), jc, jz, jM, x))*(4*jnp.pi*x**2) * x
        integral_value = jnp.trapz(fx, x=logx)
        return integral_value

    @partial(jit, static_argnums=(0,))
    def get_M_to_R(self, jM, jz, mdef_delta=200):
        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(self.cosmo_jax,self.scale_fac_a_array[jz]) * 1e9
        rho_treshold = mdef_delta * rho_c_z
        R = (self.M_array[jM] * 3.0 / 4.0 / jnp.pi / rho_treshold)**(1.0 / 3.0)
        # convert to comoving coordinates
        R *= (1 + self.z_array[jz])
        return R

    @partial(jit, static_argnums=(0,))
    def get_Mc(self, jM, jz):
        value = self.Mc0 * jnp.power(((self.M_array[jM])/self.Mstar0), self.nu_M) * jnp.power((1 + (self.z_array[jz])), self.nu_z)
        return value

    # @partial(jit, static_argnums=(0,))
    # def get_theta_ej(self, jM, jz):
    #     value = self.theta_ej_0 * jnp.power(((self.M_array[jM])/10**self.log10_Mstar0_theta_ej), self.nu_theta_ej_M) * jnp.power((1 + (self.z_array[jz])), self.nu_theta_ej_z)
    #     return value

    # @partial(jit, static_argnums=(0,))
    # def get_theta_co(self, jM, jz):
    #     value = self.theta_co_0 * jnp.power(((self.M_array[jM])/10**self.log10_Mstar0_theta_co), self.nu_theta_co_M) * jnp.power((1 + (self.z_array[jz])), self.nu_theta_co_z)
    #     return value                        

    @partial(jit, static_argnums=(0,))
    def get_theta_ej(self, jc, jM, jz):
        value = self.theta_ej_0 * jnp.power(((self.M_array[jM])/10**self.log10_Mstar0_theta_ej), self.nu_theta_ej_M) * jnp.power((1 + (self.z_array[jz])), self.nu_theta_ej_z) * jnp.power(1/self.conc_array[jc], self.nu_theta_ej_c)
        return value

    @partial(jit, static_argnums=(0,))
    def get_theta_co(self, jc, jM, jz):
        value = self.theta_co_0 * jnp.power(((self.M_array[jM])/10**self.log10_Mstar0_theta_co), self.nu_theta_co_M) * jnp.power((1 + (self.z_array[jz])), self.nu_theta_co_z) * jnp.power(1/self.conc_array[jc], self.nu_theta_co_c)
        return value                        


    # @partial(jit, static_argnums=(0,))
    # def get_beta(self, jM, jz):
    #     value = 3 - jnp.power((self.Mc_mat[jM, jz] / (self.M_array[jM])), self.mu_beta)
    #     return value

    @partial(jit, static_argnums=(0,))
    def get_beta(self, jM, jz):
        value = 3*jnp.power(self.M_array[jM]/self.Mc_mat[jM, jz],self.mu_beta)/(1 + jnp.power(self.M_array[jM]/self.Mc_mat[jM, jz],self.mu_beta))
        return value


    @partial(jit, static_argnums=(0,))
    def get_rho_nfw_unnorm(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the NFW profile (Eq.2.18)'''
        r200c = self.r200c_mat[jM, jz]
        rt = self.rt_mat[jM, jz]
        conc = self.conc_array[jc]
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
    def get_nfw_norm(self, jc, jz, jM):
        '''This is the normalization of the NFW profile'''
        r200c = self.r200c_mat[jM, jz]
        M200c = self.M_array[jM]
        logx = jnp.linspace(jnp.log(0.01*r200c), jnp.log(r200c), self.num_points_trapz_int)
        int_unnorm_prof = self.logspace_trapezoidal_integral(self.get_rho_nfw_unnorm, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
        rho_nfw_0 = M200c / int_unnorm_prof
        return rho_nfw_0


    @partial(jit, static_argnums=(0,))
    def get_rho_nfw_normed(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the NFW profile (Eq.2.18)'''
        r200c = self.r200c_mat[jM, jz]
        rt = self.rt_mat[jM, jz]
        conc = self.conc_array[jc]
        prefac = self.rho_nfw_norm_mat[jc, jz, jM]
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
        return prefac * rho_nfw
    
    @partial(jit, static_argnums=(0,))
    def get_Mtot(self, jc, jz, jM, rmax_r200c=16):
        '''This is the total mass of all matter '''
        r200c = self.r200c_mat[jM, jz]
        logx = jnp.linspace(jnp.log(0.01*r200c), jnp.log(rmax_r200c*r200c), self.num_points_trapz_int)
        Mtot = self.logspace_trapezoidal_integral(self.get_rho_nfw_normed, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
        return Mtot


    @partial(jit, static_argnums=(0,))
    def get_Mnfw(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the mass of the NFW profile'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jM, jz])
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        Mnfw = self.logspace_trapezoidal_integral(self.get_rho_nfw_normed, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
        return Mnfw

    @partial(jit, static_argnums=(0,))
    def get_rho_cga(self, jr, jc, jz, jM, r_array_here=None):
        ''' This is central galaxy profile (Eq.2.10)'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        rho_cga = (self.fcga_array[jM] * self.Mtot_mat[jc, jz, jM]) / (4 * (jnp.pi**1.5) * self.Rh_mat[jM, jz] * r**2) * jnp.exp(-(0.5 * r / self.Rh_mat[jM, jz])**2)
        return rho_cga

    @partial(jit, static_argnums=(0,))
    def get_Mcga(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the mass of the central galaxy profile'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jM, jz])        
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        Mcga = self.logspace_trapezoidal_integral(self.get_rho_cga, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
        return Mcga


    # @partial(jit, static_argnums=(0,))
    # def get_rho_gas_unnorm(self, jr, jz, jM, r_array_here=None):
    #     '''
    #     This is the gas profile (Eq.2.12)
    #     '''
    #     if r_array_here is None:
    #         r = self.r_array[jr]
    #     else:
    #         r = r_array_here[jr]
    #     u = r / self.r_co_mat[jM, jz]
    #     v = r / self.r_ej_mat[jM, jz]
    #     rho_gas_unnorm = 1 / (jnp.power(1 + u, self.beta_mat[jM, jz]) * jnp.power(1 + v**2, (7 - self.beta_mat[jM, jz]) / 2.))
    #     return rho_gas_unnorm
    
    @partial(jit, static_argnums=(0,))
    def get_rho_gas_unnorm(self, jr, jc, jz, jM, r_array_here=None):
        '''
        This is the gas profile (Eq.2.12)
        '''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        u = r / self.r_co_mat[jc, jM, jz]
        v = r / self.r_ej_mat[jc, jM, jz]
        rho_gas_unnorm = 1 / (jnp.power(1 + u, self.beta_mat[jM, jz]) * jnp.power(1 + jnp.power(v, self.gamma_rhogas), (self.delta_rhogas - self.beta_mat[jM, jz]) / self.gamma_rhogas))
        return rho_gas_unnorm    

    @partial(jit, static_argnums=(0,))
    def get_rho_gas_norm(self, jc, jz, jM, rmax_r200c=16):
        '''This is the normalization of the gas profile'''
        r200c = self.r200c_mat[jM, jz]
        logx = jnp.linspace(jnp.log(0.01*r200c), jnp.log(rmax_r200c*r200c), self.num_points_trapz_int)
        # logx = jnp.linspace(jnp.log(0.01*r200c), jnp.log(self.r_array[-1]), self.num_points_trapz_int)        
        int_unnorm_prof = self.logspace_trapezoidal_integral(self.get_rho_gas_unnorm, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
        rho_gas_norm = self.fgas_array[jM] * self.Mtot_mat[jc, jz, jM] / int_unnorm_prof
        return rho_gas_norm


    @partial(jit, static_argnums=(0,))
    def get_rho_gas_normed(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the NFW profile (Eq.2.18)'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        # u = r / self.r_co_mat[jM, jz]
        # v = r / self.r_ej_mat[jM, jz]
        # rho_gas_unnorm = 1 / (jnp.power(1 + u, self.beta_mat[jM, jz]) * jnp.power(1 + v**2, (7 - self.beta_mat[jM, jz]) / 2.))
        u = r / self.r_co_mat[jc, jM, jz]
        v = r / self.r_ej_mat[jc, jM, jz]
        rho_gas_unnorm = 1 / (jnp.power(1 + u, self.beta_mat[jM, jz]) * jnp.power(1 + jnp.power(v, self.gamma_rhogas), (self.delta_rhogas - self.beta_mat[jM, jz]) / self.gamma_rhogas))
        prefac = self.rho_gas_norm_mat[jc, jz, jM]
        return prefac * rho_gas_unnorm

    @partial(jit, static_argnums=(0,))
    def get_Mgas(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the mass of the gas profile'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jM, jz])        
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        Mgas = self.logspace_trapezoidal_integral(self.get_rho_gas_normed, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
        return Mgas

    @partial(jit, static_argnums=(0,))
    def get_zeta(self, jr, jc, jz, jM, r_array_here=None):
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
        Mi = self.get_Mnfw(jr, jc, jz, jM, r_array_here=r_array_here)

        def zeta_equation(zeta):
            rf = zeta * ri
            Mf = self.fclm_array[jM] * Mi + self.get_Mcga(0, jc, jz, jM, r_array_here=jnp.array([rf])) + self.get_Mgas(0, jc, jz, jM, r_array_here=jnp.array([rf]))
            return ((rf / ri - 1) - self.a_zeta * ((Mi / Mf)**self.n_zeta - 1))
        zeta_array = jnp.linspace(0.5, 1.5, 32)
        value_out = vmap(zeta_equation)(zeta_array)
        zeta = jnp.interp(0.0, value_out, zeta_array)
        return zeta


    @partial(jit, static_argnums=(0,))
    def get_rho_clm(self, jr, jc, jz, jM, r_array_here=None):
        if r_array_here is None:
            r_array_here = self.r_array
        zeta = (jnp.interp(jnp.log(r_array_here[jr]), jnp.log(self.r_array), self.zeta_mat[:,jc, jz, jM]))
        if r_array_here is None:
            r_array_new = self.r_array/zeta        
        else:
            r_array_new = r_array_here/zeta
        
        rho_nfw = self.get_rho_nfw_normed(jr, jc, jz, jM, r_array_new)
        rho_clm = (self.fclm_array[jM] / (zeta**3)) * rho_nfw
        return rho_clm


    @partial(jit, static_argnums=(0,))
    def get_Mclm(self, jr, jc, jz, jM, r_array_here=None):
        if r_array_here is None:
            r_array_here = self.r_array
        zeta = (jnp.interp(jnp.log(r_array_here[jr]), jnp.log(self.r_array), self.zeta_mat[:,jc, jz, jM]))
        if r_array_here is None:
            r_array_new = self.r_array/zeta        
        else:
            r_array_new = r_array_here/zeta

        M_clm = self.fclm_array[jM] * self.get_Mnfw(jr, jc, jz, jM, r_array_new)
        return M_clm


    @partial(jit, static_argnums=(0,))
    def get_rho_dmb(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the total matter profile with all the components (Eq.2.2)'''    
        rho_dmb = self.get_rho_gas_normed(jr, jc, jz, jM, r_array_here=r_array_here) + \
            self.get_rho_cga(jr, jc, jz, jM, r_array_here=r_array_here) + self.get_rho_clm(jr, jc, jz, jM, r_array_here=r_array_here)
        return rho_dmb

    @partial(jit, static_argnums=(0,))
    def get_Mdmb(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the mass inside some radius for the full dmb profile'''
        # Mdmb = simps(lambda x: get_rho_dmb(x) * 4 * jnp.pi * x**2, 1e-3, r, int_simps_points)
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jM, jz])
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        Mdmb = self.logspace_trapezoidal_integral(self.get_rho_dmb, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
        return Mdmb
    
    @partial(jit, static_argnums=(0,))
    def get_Mdmb_r200(self, jc, jz, jM):
        '''This is the mass inside some radius for the full dmb profile'''
        r = self.r200c_mat[jM, jz]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jM, jz])        
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        Mdmb = self.logspace_trapezoidal_integral(self.get_rho_dmb, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
        return Mdmb


    @partial(jit, static_argnums=(0,))
    def get_Ptot(self, jr, jc, jz, jM, r_array_here=None, rmax_r200c=6):
        '''This is the total pressure profile, assuming HSE'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        logx = jnp.linspace(jnp.log(r), jnp.log(rmax_r200c*self.r200c_mat[jM, jz]), self.num_points_trapz_int)
        x = jnp.exp(logx)
        fx1 = (vmap(self.get_rho_gas_normed, (0, None, None, None,None))(jnp.arange(len(logx)), jc, jz, jM, x))
        fx2 = jnp.exp(jnp.interp(logx, jnp.log(self.r_array), jnp.log(self.Mdmb_mat[:,jc, jz, jM])))
        fx = (fx1 * fx2 * G_new / x**2) * x
        Ptot = jnp.trapz(fx, x=logx)
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
    def get_Pnt_fac(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the non-thermal pressure profile'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        Pnt_fac = self.alpha_nt * self.get_fz_Pnt(jz) * ((r / self.r200c_mat[jM, jz])**self.n_nt)
        return Pnt_fac

