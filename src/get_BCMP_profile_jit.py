# platform = 'cpu'
from jax.lib import xla_bridge
platform = xla_bridge.get_backend().platform
import jax
jax.config.update('jax_platform_name', platform)

import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from jaxopt import Bisection
from functools import partial

import astropy.units as u
from astropy import constants as const
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.eV / u.cm**3)).value
from colossus.halo import mass_so

from jax_cosmo import Cosmology
from colossus.cosmology import cosmology



class BCM_18_wP:

    def __init__(
            self,
            sim_params_dict,
            halo_params_dict,
            num_points_trapz_int=64
        ):
        cosmo_params = sim_params_dict['cosmo']
        self.cosmo_colossus = cosmology.setCosmology('myCosmo', **cosmo_params)
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

        self.nfw_trunc = sim_params_dict['nfw_trunc']
        self.theta_co=sim_params_dict['theta_co']
        self.theta_ej=sim_params_dict['theta_ej']
        self.neg_bhse_plus_1=sim_params_dict['neg_bhse_plus_1']
        self.mu_beta=sim_params_dict['mu_beta']
        self.eta_star=sim_params_dict['eta_star']
        self.eta_cga=sim_params_dict['eta_cga']
        self.A_starcga=sim_params_dict['A_starcga']
        self.M1_starcga=sim_params_dict['M1_starcga']
        self.epsilon_rt=sim_params_dict['epsilon_rt']
        self.Mc0 = sim_params_dict['Mc0']
        self.nu_z = sim_params_dict['nu_z']
        self.nu_M = sim_params_dict['nu_M']
        self.Mstar0 = sim_params_dict['Mstar0']
        self.a_zeta=sim_params_dict['a_zeta']
        self.n_zeta=sim_params_dict['n_zeta']
        self.alpha_nt = sim_params_dict['alpha_nt']
        self.beta_nt = sim_params_dict['beta_nt']
        self.n_nt = sim_params_dict['n_nt']

        self.num_points_trapz_int = num_points_trapz_int


        rmin, rmax, nr = halo_params_dict['rmin'], halo_params_dict['rmax'], halo_params_dict['nr']
        zmin, zmax, nz = halo_params_dict['zmin'], halo_params_dict['zmax'], halo_params_dict['nz']
        Mmin, Mmax, nM = halo_params_dict['Mmin'], halo_params_dict['Mmax'], halo_params_dict['nM']
        cmin, cmax, nc = halo_params_dict['cmin'], halo_params_dict['cmax'], halo_params_dict['nc']
        self.r_array = jnp.logspace(np.log10(rmin), np.log10(rmax), nr)
        self.z_array = jnp.linspace(zmin, zmax, nz)
        self.M200c_array = jnp.logspace(np.log10(Mmin), np.log10(Mmax), nM)
        self.conc_array = jnp.exp(jnp.linspace(jnp.log(cmin), jnp.log(cmax), nc))
        
        mdef = halo_params_dict['mdef']
        r200c_mat = np.zeros((len(self.M200c_array), len(self.z_array)))
        for jz in range(len(self.z_array)):
            r200c_mat[:, jz] = (mass_so.M_to_R(self.M200c_array, self.z_array[jz], mdef) / (1000.))

        self.r200c_mat = jnp.array(r200c_mat)


        self.rt_mat = self.r200c_mat * self.epsilon_rt
        Mc_mat = np.zeros((len(self.M200c_array), len(self.z_array)))
        beta_mat = np.zeros((len(self.M200c_array), len(self.z_array)))
        for jM in range(len(self.M200c_array)):
            Mc_mat[jM, :] = self.Mc0 * ((np.array(self.M200c_array)[jM]/self.Mstar0) ** self.nu_M) * ((1 + np.array(self.z_array)) ** self.nu_z)
            beta_mat[jM, :] = 3 - ((Mc_mat[jM, :] / np.array(self.M200c_array)[jM]) ** self.mu_beta)
        self.Mc_mat = jnp.array(Mc_mat)
        self.beta_mat = jnp.array(beta_mat)
        self.r_co_mat = self.theta_co * self.r200c_mat
        self.r_ej_mat = self.theta_ej * self.r200c_mat
        self.fstar_array = self.A_starcga * ((self.M1_starcga / self.M200c_array) ** self.eta_star)
        self.fgas_array = (cosmo_params['Ob0'] / cosmo_params['Om0']) - self.fstar_array
        self.fcga_array = self.A_starcga * ((self.M1_starcga / self.M200c_array) ** self.eta_cga)
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

        vmap_func1 = vmap(self.get_zeta, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.zeta_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        vmap_func1 = vmap(self.get_rho_dmb, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.rho_dmb_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        vmap_func1 = vmap(self.get_Mdmb, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.Mdmb_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        vmap_func1 = vmap(self.get_Ptot, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.Ptot_mat = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        vmap_func1 = vmap(self.get_Pnt_fac, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        Pnt_fac = vmap_func4(jnp.arange(nr), jnp.arange(nc), jnp.arange(nz), jnp.arange(nM)).T

        self.Pnt_mat = Pnt_fac * self.Ptot_mat
        self.Pth_mat = self.Ptot_mat * jnp.maximum(0, 1 - (self.Pnt_mat / self.Ptot_mat))

    def logspace_trapezoidal_integral(self, f, logx, jc=None, jz=None, jM=None, axis_tup=(0, None, None, None)):
        x = jnp.exp(logx)
        if jc is None:
            fx = (vmap(f, axis_tup)(jnp.arange(len(logx)), jz, jM, x))*(4*jnp.pi*x**2) * x
        else:
            fx = (vmap(f, axis_tup)(jnp.arange(len(logx)), jc, jz, jM, x))*(4*jnp.pi*x**2) * x
        integral_value = jnp.trapz(fx, x=logx)
        return integral_value



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
        M200c = self.M200c_array[jM]
        logx = jnp.linspace(jnp.log(0.01*r200c), jnp.log(r200c), self.num_points_trapz_int)
        # x = jnp.exp(logx)
        # fx = (vmap(self.get_rho_nfw_unnorm, (0, None, None, None,None))(jnp.arange(len(logx)), jc, jz, jM, x))*(4*jnp.pi*x**2) * x
        # int_unnorm_prof = jnp.trapz(fx, x=logx)
        int_unnorm_prof = self.logspace_trapezoidal_integral(self.get_rho_nfw_unnorm, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
        # print(int_unnorm_prof)
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
    def get_Mtot(self, jc, jz, jM, rmax_r200c=10):
        '''This is the total mass of all matter '''
        # Mtot = simps(
        #     lambda x: get_rho_nfw(x) * 4 * jnp.pi * x**2, 5e-4, rmax_r200c * r200c, int_simps_points
        #     )
        r200c = self.r200c_mat[jM, jz]
        logx = jnp.linspace(jnp.log(0.005*r200c), jnp.log(rmax_r200c*r200c), self.num_points_trapz_int)
        # x = jnp.exp(logx)
        # fx = (vmap(self.get_rho_nfw_normed, (0, None, None, None,None))(jnp.arange(len(logx)), jc, jz, jM, x))*(4*jnp.pi*x**2) * x
        # Mtot = jnp.trapz(fx, x=logx)
        Mtot = self.logspace_trapezoidal_integral(self.get_rho_nfw_normed, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
        return Mtot


    @partial(jit, static_argnums=(0,))
    def get_Mnfw(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the mass of the NFW profile'''
        # Mnfw = simps(lambda x: get_rho_nfw(x) * 4 * jnp.pi * x**2, 5e-4, r, int_simps_points)
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jM, jz])
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        # x = jnp.exp(logx)
        # fx = (vmap(self.get_rho_nfw_normed, (0, None, None, None,None))(jnp.arange(len(logx)), jc, jz, jM, x))*(4*jnp.pi*x**2) * x
        # Mnfw = jnp.trapz(fx, x=logx)
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
        # Mcga = simps(lambda x: get_rho_cga(x) * 4 * jnp.pi * x**2, 5e-4, r, int_simps_points)
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jM, jz])        
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        # x = jnp.exp(logx)
        # fx = (vmap(self.get_rho_cga, (0, None, None, None,None))(jnp.arange(len(logx)), jc, jz, jM, x))*(4*jnp.pi*x**2) * x
        # Mcga = jnp.trapz(fx, x=logx)
        Mcga = self.logspace_trapezoidal_integral(self.get_rho_cga, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
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
        u = r / self.r_co_mat[jM, jz]
        v = r / self.r_ej_mat[jM, jz]
        rho_gas_unnorm = 1 / (jnp.power(1 + u, self.beta_mat[jM, jz]) * jnp.power(1 + v**2, (7 - self.beta_mat[jM, jz]) / 2.))
        return rho_gas_unnorm

    @partial(jit, static_argnums=(0,))
    def get_rho_gas_norm(self, jc, jz, jM, rmax_r200c=20):
        '''This is the normalization of the gas profile'''
        # int_unnorm_prof = simps(
        #     lambda x: get_rho_gas_unnorm(x) * 4 * jnp.pi * x**2, 5e-4, rmax_r200c * r200c,
        #     int_simps_points
        #     )        
        r200c = self.r200c_mat[jM, jz]
        logx = jnp.linspace(jnp.log(0.01*r200c), jnp.log(rmax_r200c*r200c), self.num_points_trapz_int)
        # x = jnp.exp(logx)
        # fx = (vmap(self.get_rho_gas_unnorm, (0, None, None,None))(jnp.arange(len(logx)), jz, jM, x))*(4*jnp.pi*x**2) * x
        # int_unnorm_prof = jnp.trapz(fx, x=logx)
        int_unnorm_prof = self.logspace_trapezoidal_integral(self.get_rho_gas_unnorm, logx, jc=None, jz=jz, jM=jM, axis_tup=(0, None, None, None))
        rho_gas_norm = self.fgas_array[jM] * self.Mtot_mat[jc, jz, jM] / int_unnorm_prof
        return rho_gas_norm


    @partial(jit, static_argnums=(0,))
    def get_rho_gas_normed(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the NFW profile (Eq.2.18)'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        u = r / self.r_co_mat[jM, jz]
        v = r / self.r_ej_mat[jM, jz]
        rho_gas_unnorm = 1 / (jnp.power(1 + u, self.beta_mat[jM, jz]) * jnp.power(1 + v**2, (7 - self.beta_mat[jM, jz]) / 2.))
        prefac = self.rho_gas_norm_mat[jc, jz, jM]
        return prefac * rho_gas_unnorm

    @partial(jit, static_argnums=(0,))
    def get_Mgas(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the mass of the gas profile'''
        # Mgas = simps(lambda x: get_rho_gas(x) * 4 * jnp.pi * x**2, 5e-4, r, int_simps_points)
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        minr = jnp.minimum(5e-4, 0.005*self.r200c_mat[jM, jz])        
        logx = jnp.linspace(jnp.log(minr), jnp.log(r), self.num_points_trapz_int)
        # x = jnp.exp(logx)
        # fx = (vmap(self.get_rho_gas_normed, (0, None, None, None,None))(jnp.arange(len(logx)), jc, jz, jM, x))*(4*jnp.pi*x**2) * x
        # Mgas = jnp.trapz(fx, x=logx)
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
        bisec = Bisection(optimality_fun=zeta_equation, lower=0.01, upper=1.5, 
                    check_bracket=False, unroll=True)
        zeta = bisec.run().params
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
        # x = jnp.exp(logx)
        # fx = (vmap(self.get_rho_dmb, (0, None, None, None,None))(jnp.arange(len(logx)), jc, jz, jM, x))*(4*jnp.pi*x**2) * x
        # Mdmb = jnp.trapz(fx, x=logx)
        Mdmb = self.logspace_trapezoidal_integral(self.get_rho_dmb, logx, jc=jc, jz=jz, jM=jM, axis_tup=(0, None, None, None, None))
        return Mdmb

    @partial(jit, static_argnums=(0,))
    def get_Ptot(self, jr, jc, jz, jM, r_array_here=None):
        '''This is the total pressure profile, assuming HSE'''
        # Ptot = simps(
        #     lambda x: G_new * get_rho_gas(x) * get_Mdmb(x) / x**2, r, 6*r200c, int_simps_points
        #     )
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        logx = jnp.linspace(jnp.log(r), jnp.log(6*self.r200c_mat[jM, jz]), self.num_points_trapz_int)
        x = jnp.exp(logx)
        fx1 = (vmap(self.get_rho_gas_normed, (0, None, None, None,None))(jnp.arange(len(logx)), jc, jz, jM, x))
        # fx2 = (vmap(self.get_Mdmb, (0, None, None, None,None))(jnp.arange(len(logx)), jc, jz, jM, x))
        fx2 = jnp.interp(logx, jnp.log(self.r_array), self.Mdmb_mat[:,jc, jz, jM])
        fx = (fx1 * fx2 * G_new / x**2) * x
        Ptot = jnp.trapz(fx, x=logx)
        return Ptot
    
    @partial(jit, static_argnums=(0,))
    def get_fz_Pnt(self, jz):
        '''This is the evolution of non-thermal pressure with redshift'''
        fmax = 4**(-1 * self.n_nt) / self.alpha_nt
        fz = jnp.minimum((1 + self.z_array[jz])**self.beta_nt, (fmax - 1) * jnp.tanh(self.beta_nt * self.z_array[jz]) + 1)
        return fz

    @partial(jit, static_argnums=(0,))
    def get_Pnt_fac(self, jr, jc, jM, jz, r_array_here=None):
        '''This is the non-thermal pressure profile'''
        if r_array_here is None:
            r = self.r_array[jr]
        else:
            r = r_array_here[jr]
        Pnt_fac = self.alpha_nt * self.get_fz_Pnt(jz) * ((r / self.r200c_mat[jM, jz])**self.n_nt)
        return Pnt_fac

