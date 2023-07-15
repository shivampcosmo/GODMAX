import os
from get_BCMP_profile_jit import BCM_18_wP
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from jax import vmap, grad
from jax_cosmo import Cosmology
from functools import partial
from jax_cosmo.power import linear_matter_power
from jax_cosmo.background import angular_diameter_distance, radial_comoving_distance
import jax_cosmo.transfer as tklib
import astropy.units as u
from astropy import constants as const
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.eV / u.cm**3)).value
G_new_rhom = const.G.to(u.Mpc**3 / ((u.s**2) * u.M_sun))
import constants
from mcfit import xi2P

import jax_cosmo.background as bkgrd
# import jax_cosmo.constants as const
import jax_cosmo.transfer as tklib
from jax_cosmo.scipy.integrate import romb
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.scipy.interpolate import interp


class setup_power_BCMP:
    def __init__(
                self,
                sim_params_dict,
                halo_params_dict,
                num_points_trapz_int=64,
                BCMP_obj=None
            ):    
        
        self.cosmo_params = sim_params_dict['cosmo']

        self.cosmo_jax = Cosmology(
            Omega_c=self.cosmo_params['Om0'] - self.cosmo_params['Ob0'],
            Omega_b=self.cosmo_params['Ob0'],
            h=self.cosmo_params['H0'] / 100.,
            sigma8=self.cosmo_params['sigma8'],
            n_s=self.cosmo_params['ns'],
            Omega_k=0.,
            w0=self.cosmo_params['w0'],
            wa=0.
            )

        H0 = 100. * (u.km / (u.s * u.Mpc))
        self.rho_m_bar = ((self.cosmo_params['Om0'] * 3 * (H0**2) / (8 * jnp.pi * G_new_rhom)).to(u.M_sun / (u.Mpc**3))).value

        if BCMP_obj is None:
            BCMP_obj = BCM_18_wP(sim_params_dict, halo_params_dict, num_points_trapz_int=num_points_trapz_int)
        self.Mtot_mat = BCMP_obj.Mtot_mat
        Mtot_rep = jnp.repeat(self.Mtot_mat[None, :, :, :], len(BCMP_obj.r_array), axis=0)
        self.r_array = BCMP_obj.r_array
        self.M_array = BCMP_obj.M200c_array
        self.z_array = BCMP_obj.z_array
        self.scale_fac_a_array = 1./(1. + self.z_array)
        self.conc_array = BCMP_obj.conc_array
        self.nM, self.nz, self.nc = len(self.M_array), len(self.z_array), len(self.conc_array)
        self.r200c_mat = BCMP_obj.r200c_mat
        self.rho_dmb_mat = BCMP_obj.rho_dmb_mat
        self.rho_nfw_mat = BCMP_obj.rho_nfw_mat
        self.sig_logc_z_array = jnp.array(halo_params_dict['sig_logc_z_array'])
        self.beam_fwhm_arcmin = sim_params_dict['beam_fwhm_arcmin']        

        self.kPk_array = jnp.logspace(jnp.log10(1E-3), jnp.log10(100), 100)
        self.plin_kz_mat = vmap(linear_matter_power,(None, None, 0))(self.cosmo_jax, self.kPk_array, self.scale_fac_a_array).T


        self.chi_array = radial_comoving_distance(self.cosmo_jax, self.scale_fac_a_array)
        self.DA_array = angular_diameter_distance(self.cosmo_jax, self.scale_fac_a_array)
        self.growth_array = bkgrd.growth_factor(self.cosmo_jax, self.scale_fac_a_array)

        vmap_func1 = vmap(self.get_sigma_Mz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.sigma_Mz_mat = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T

        grad_lgsigma = grad(self.get_lgsigma_z, argnums=1)
        vmap_func1 = vmap(grad_lgsigma, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.dlgsig_dlnM_mat = vmap_func2(jnp.arange(self.nz), jnp.log(self.M_array)).T

        rhom_z_array = (constants.RHO_CRIT_0_KPC3 * self.cosmo_params['Om0'] * (1.0 + self.z_array)**3) * 1E9
        rhom_z_mat = jnp.repeat(rhom_z_array[None, :], self.nM, axis=0)
        M_z_mat = jnp.repeat(self.M_array[:, None], self.nz, axis=1)

        vmap_func1 = vmap(self.get_fsigma_Mz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.fsigma_Mz_mat = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T

        self.hmf_Mz_mat = -1 * self.fsigma_Mz_mat * (rhom_z_mat/M_z_mat).T * self.dlgsig_dlnM_mat

        vmap_func1 = vmap(self.get_bias_Mz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.bias_Mz_mat = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T

        vmap_func1 = vmap(self.get_conc_Mz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.conc_Mz_mat = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T

        
        

        self.k, self.uk_nfw = (xi2P(BCMP_obj.r_array)(BCMP_obj.rho_nfw_mat/Mtot_rep, axis=0))
        _, self.uk_dmb = (xi2P(BCMP_obj.r_array)(BCMP_obj.rho_dmb_mat/Mtot_rep, axis=0))
        self.k = jnp.array(self.k)
        self.uk_nfw = jnp.array(self.uk_nfw)
        self.uk_dmb = jnp.array(self.uk_dmb)

        vmap_func1 = vmap(self.get_uknfw_interp_Pk, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.uk_nfw_Pk = vmap_func3(jnp.arange(self.nc), jnp.arange(self.nz), jnp.arange(self.nM)).T

        vmap_func1 = vmap(self.get_ukdmb_interp_Pk, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.uk_dmb_Pk = vmap_func3(jnp.arange(self.nc), jnp.arange(self.nz), jnp.arange(self.nM)).T

        vmap_func1 = vmap(self.get_Pmm_dmb_1h, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pmm_dmb_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T

        vmap_func1 = vmap(self.get_Pmm_nfw_1h, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pmm_nfw_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T


        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        coeff = sigmat / (m_e * (c ** 2))
        oneMpc_h = (((10 ** 6) / self.cosmo_jax.h) * (u.pc).to(u.m)) * (u.m)
        const_coeff = ((coeff * oneMpc_h).to(((u.cm ** 3) / u.eV))).value
        Pe_conv_fac =  0.518
        self.y3d_mat = Pe_conv_fac * const_coeff * BCMP_obj.Pth_mat

        ellmin, ellmax, nell = halo_params_dict['ellmin'], halo_params_dict['ellmax'], halo_params_dict['nell']
        self.ell_array = jnp.logspace(jnp.log10(ellmin), jnp.log10(ellmax), nell)
        self.sig_beam = self.beam_fwhm_arcmin * (1. / 60.) * (jnp.pi / 180.) * (1. / jnp.sqrt(8. * jnp.log(2)))

        vmap_func1 = vmap(self.get_uyl, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.uyl_mat = vmap_func4(jnp.arange(nell), jnp.arange(self.nc), jnp.arange(self.nz), jnp.arange(self.nM)).T
        
        vmap_func1 = vmap(self.get_byl, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.byl_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T

        vmap_func1 = vmap(self.get_ukappal_dmb_prefac, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.ukappal_dmb_prefac_mat = vmap_func4(jnp.arange(nell), jnp.arange(self.nc), jnp.arange(self.nz), jnp.arange(self.nM)).T

        vmap_func1 = vmap(self.get_ukappal_nfw_prefac, (0, None, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0, None))
        vmap_func4 = vmap(vmap_func3, (None, None, None, 0))
        self.ukappal_nfw_prefac_mat = vmap_func4(jnp.arange(nell), jnp.arange(self.nc), jnp.arange(self.nz), jnp.arange(self.nM)).T

        vmap_func1 = vmap(self.get_Pklin_lz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pklin_lz_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T



    def get_rho_m(self, z):
        return (constants.RHO_CRIT_0_KPC3 * self.cosmo_params['Om0'] * (1.0 + z)**3) * 1E9

    def get_Ez(self, z):
        zp1 = (1.0 + z)
        t = (self.cosmo_params['Om0']) * zp1**3 + (1 - self.cosmo_params['Om0'])
        E = jnp.sqrt(t)        
        return E

    def get_rho_c(self, z):
        return constants.RHO_CRIT_0_KPC3 * self.get_Ez(z)**2  * 1E9      

    @partial(jit, static_argnums=(0,))        
    def get_lgsigma_z(self, jz, lgM, kmin=0.0001, kmax=1000.0):
        M = jnp.exp(lgM)
        R = (3.0 * M / 4.0 / np.pi / self.get_rho_m(0.0))**(1.0 / 3.0)
        def int_sigma(logk):
            k = jnp.exp(logk)
            x = k * R
            w = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / (x * x * x)
            pkz = jnp.exp(jnp.interp(logk, jnp.log(self.kPk_array), jnp.log(self.plin_kz_mat[:, jz])))
            return k * (k * w) ** 2 * pkz

        y = romb(int_sigma, jnp.log10(kmin), jnp.log10(kmax), divmax=7)
        return jnp.log(jnp.sqrt(y / (2.0 * jnp.pi**2.0)))

        
    @partial(jit, static_argnums=(0,))        
    def get_sigma_Mz(self, jz, jM, kmin=0.0001, kmax=1000.0):
        R = (3.0 * self.M_array[jM] / 4.0 / np.pi / self.get_rho_m(0.0))**(1.0 / 3.0)

        def int_sigma(logk):
            k = jnp.exp(logk)
            x = k * R
            w = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / (x * x * x)
            pkz = jnp.exp(jnp.interp(logk, jnp.log(self.kPk_array), jnp.log(self.plin_kz_mat[:, jz])))
            return k * (k * w) ** 2 * pkz

        y = romb(int_sigma, jnp.log10(kmin), jnp.log10(kmax), divmax=7)
        return jnp.sqrt(y / (2.0 * jnp.pi**2.0))


    @partial(jit, static_argnums=(0,))
    def get_fsigma_Mz(self, jz, jM, mdef_delta=200):
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
    def get_bias_Mz(self, jz, jM, mdef_delta=200):
        '''Tinker 2010 bias function'''
        sigma = self.sigma_Mz_mat[jz, jM]
        delta_c = constants.DELTA_COLLAPSE
        nu = delta_c / sigma

        z = self.z_array[jz]    
        rho_treshold = mdef_delta * self.get_rho_c(z)
        Delta = rho_treshold / self.get_rho_m(z)
        y = jnp.log10(Delta)

        A = 1.0 + 0.24 * y * jnp.exp(-1.0 * (4.0 / y)**4)
        a = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * jnp.exp(-1.0 * (4.0 / y)**4)
        c = 2.4
        
        bias = 1.0 - A * nu**a / (nu**a + constants.DELTA_COLLAPSE**a) + B * nu**b + C * nu**c
        return bias
    
    @partial(jit, static_argnums=(0,))
    def get_conc_Mz(self, jz, jM, mdef='200c'):
        '''Duffy 2008 concentration relation, for mdef = 200c'''
        M = self.M_array[jM]
        z = self.z_array[jz]
        A = 5.71
        B = -0.084
        C = -0.47

        c = A * (M / 2E12)**B * (1.0 + z)**C
        
        return c


    @partial(jit, static_argnums=(0,))
    def get_uyl(self, jl, jc, jz, jM, xmin=0.01, xmax=4, num_points_trapz_int=64):
        r200c = self.r200c_mat[jz, jM]
        # z = self.z_array[jz]
        # az = 1.0 / (1.0 + z)
        # Da_z = angular_diameter_distance(self.cosmo_jax, az)
        Da_z = jnp.clip(self.DA_array[jz], 1.0)
        l200c = Da_z/r200c
        prefac = r200c/l200c**2
        logx_array = jnp.linspace(jnp.log(xmin), jnp.log(xmax), num_points_trapz_int)
        x_array = jnp.exp(logx_array)

        y3d_min = jnp.min(jnp.absolute(self.y3d_mat[:,jc, jz, jM]))
        y3d_clipped = jnp.clip(self.y3d_mat[:,jc, jz, jM], y3d_min + 1e-25)
        # y3d_xarray = jnp.exp(jnp.interp(logx_array, jnp.log(self.r_array/r200c), jnp.log(self.y3d_mat[:,jc, jz, jM])))
        y3d_xarray = jnp.exp(jnp.interp(logx_array, jnp.log(self.r_array/r200c), jnp.log(y3d_clipped)))        
        ell = self.ell_array[jl]
        sin_fac = (jnp.sin(ell*x_array/l200c))/(ell*x_array/l200c)

        fx = y3d_xarray * sin_fac * (4*jnp.pi*x_array**2) * x_array
        uyl = prefac * jnp.trapz(fx, x=logx_array)
        Bl = jnp.exp(-1. * ell * (ell + 1) * (self.sig_beam ** 2) / 2.)
        return uyl * Bl


    @partial(jit, static_argnums=(0,))
    def get_byl(self, jl, jz):
        uyl_jl_jz = self.uyl_mat[jl, :, jz, :]
        cmean_jz = self.conc_Mz_mat[jz, :]
        logc_array = jnp.log(self.conc_array)
        sig_logc = self.sig_logc_z_array[jz]
        conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))

        fx = uyl_jl_jz.T * p_logc_Mz
        uyl_intc = jnp.trapz(fx, x=logc_array)

        dndlnM_z = self.hmf_Mz_mat[jz, :]
        bM_z = self.bias_Mz_mat[jz, :]
        fx = uyl_intc * dndlnM_z * bM_z
        byl = jnp.trapz(fx, x=jnp.log(self.M_array))
        return byl

    @partial(jit, static_argnums=(0,))
    def get_Pklin_lz(self, jl, jz):
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.plin_kz_mat[:,jz])))
        return Pkz_ell


    @partial(jit, static_argnums=(0,))
    def get_ukappal_dmb_prefac(self, jl, jc, jz, jM):
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        # uk_dmb_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(self.uk_dmb[:,jc, jz, jM])))
        uk_min = jnp.min(jnp.absolute(self.uk_dmb[:,jc, jz, jM]))
        uk_clipped = jnp.clip(self.uk_dmb[:,jc, jz, jM], uk_min + 1e-25) * self.M_array[jM]/self.rho_m_bar
        uk_dmb_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(uk_clipped)))        
        return uk_dmb_ell

    @partial(jit, static_argnums=(0,))
    def get_ukappal_nfw_prefac(self, jl, jc, jz, jM):
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        # uk_nfw_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(self.uk_nfw[:,jc, jz, jM])))
        uk_min = jnp.min(jnp.absolute(self.uk_nfw[:,jc, jz, jM]))
        uk_clipped = jnp.clip(self.uk_nfw[:,jc, jz, jM], uk_min + 1e-25) * self.M_array[jM]/self.rho_m_bar
        uk_nfw_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(uk_clipped)))        
        return uk_nfw_ell
    
    @partial(jit, static_argnums=(0,))
    def get_ukdmb_interp_Pk(self, jc, jz, jM):
        ukdmb_array_kPk = jnp.exp(jnp.interp(jnp.log(self.kPk_array), jnp.log(self.k), jnp.log(self.uk_dmb[:,jc, jz, jM])))
        return ukdmb_array_kPk

    @partial(jit, static_argnums=(0,))
    def get_uknfw_interp_Pk(self, jc, jz, jM):
        uknfw_array_kPk = jnp.exp(jnp.interp(jnp.log(self.kPk_array), jnp.log(self.k), jnp.log(self.uk_nfw[:,jc, jz, jM])))
        return uknfw_array_kPk


    @partial(jit, static_argnums=(0,))
    def get_Pmm_dmb_1h(self, jk, jz):
        cmean_jz = self.conc_Mz_mat[jz, :]
        logc_array = jnp.log(self.conc_array)
        sig_logc = self.sig_logc_z_array[jz]
        conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))
        
        fx = ((self.Mtot_mat[:, jz, :] *  self.uk_dmb_Pk[jk,:,jz,:])**2).T * p_logc_Mz
        ukz_intc = jnp.trapz(fx, x=logc_array)
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        rhom_z = self.get_rho_m(self.z_array[jz])
        fx = ukz_intc * dndlnM_z * ((1/rhom_z)**2)
        Pmm_1h = jnp.trapz(fx, x=jnp.log(self.M_array))
        return Pmm_1h

    @partial(jit, static_argnums=(0,))
    def get_Pmm_nfw_1h(self, jk, jz):
        cmean_jz = self.conc_Mz_mat[jz, :]
        logc_array = jnp.log(self.conc_array)
        sig_logc = self.sig_logc_z_array[jz]
        conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))
        
        fx = ((self.Mtot_mat[:, jz, :] *  self.uk_nfw_Pk[jk,:,jz,:])**2).T * p_logc_Mz
        ukz_intc = jnp.trapz(fx, x=logc_array)
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        rhom_z = self.get_rho_m(self.z_array[jz])
        fx = ukz_intc * dndlnM_z * ((1/rhom_z)**2)
        Pmm_1h = jnp.trapz(fx, x=jnp.log(self.M_array))
        return Pmm_1h


