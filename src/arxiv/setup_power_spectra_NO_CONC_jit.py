import os, sys
from get_BCMP_profile_jit import BCM_18_wP
from get_BCMP_profile_NO_CONC_jit import BCM_18_wP_NO_CONC
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from jax import vmap, grad
import math
from jax_cosmo import Cosmology
from functools import partial
# from jax_cosmo.power import linear_matter_power
from GODMAX.src.jax_cosmo_power import linear_matter_power, halofit_parameters, nonlinear_matter_power
from jax_cosmo.background import angular_diameter_distance, radial_comoving_distance
import jax_cosmo.transfer as tklib
import astropy.units as u
from astropy import constants as const
import jax.scipy.integrate as jsi
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.keV / u.cm**3)).value
G_new_rhom = const.G.to(u.Mpc**3 / ((u.s**2) * u.M_sun))
import constants
# from mcfit import xi2P
# sys.path.append('/mnt/home/spandey/ceph/GODMAX/src/mcfit_jax')
# from cosmology_jax import xi2P
from mcfitjax.cosmology_jax import xi2P
import time
import jax_cosmo.background as bkgrd
import jax_cosmo.transfer as tklib
from jax_cosmo.scipy.integrate import romb
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.scipy.interpolate import interp
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline

class setup_power_BCMP_NO_CONC:
    def __init__(
                self,
                sim_params_dict,
                halo_params_dict,
                analysis_dict,
                num_points_trapz_int=64,
                BCMP_obj=None,
                verbose_time=False
            ):    
        
        self.cosmo_params = sim_params_dict['cosmo']
        self.Om0 = self.cosmo_params['Om0']

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
        self.rho_m_bar = self.cosmo_params['Om0'] * ((3 * (H0**2) / (8 * jnp.pi * G_new_rhom)).to(u.M_sun / (u.Mpc**3))).value
        self.conc_dep_model = analysis_dict.get('conc_dep_model',False)
        if BCMP_obj is None:
            if self.conc_dep_model:
                BCMP_obj = BCM_18_wP(sim_params_dict, halo_params_dict, num_points_trapz_int=num_points_trapz_int, verbose_time=verbose_time, analysis_dict=analysis_dict)
            else:
                BCMP_obj = BCM_18_wP_NO_CONC(sim_params_dict, halo_params_dict, num_points_trapz_int=num_points_trapz_int, verbose_time=verbose_time, analysis_dict=analysis_dict)
           
        if verbose_time:
            ti = time.time()

        self.Mtot_mat = BCMP_obj.Mtot_mat
        Mtot_rep = jnp.repeat(self.Mtot_mat[None, :, :], len(BCMP_obj.r_array), axis=0)
        self.r_array = BCMP_obj.r_array
        self.M_array = BCMP_obj.M_array
        self.z_array = BCMP_obj.z_array
        self.scale_fac_a_array = 1./(1. + self.z_array)
        # self.conc_array = BCMP_obj.conc_array
        self.nr, self.nM, self.nz = len(self.r_array), len(self.M_array), len(self.z_array)
        self.ell_array = halo_params_dict.get('ell_array',None)
        if self.ell_array is None:
            ellmin, ellmax, nell = halo_params_dict['ellmin'], halo_params_dict['ellmax'], halo_params_dict['nell']
            self.ell_array = jnp.logspace(jnp.log10(ellmin), jnp.log10(ellmax), nell)        
        else:            
            ellmin, ellmax, nell = jnp.min(self.ell_array), jnp.max(self.ell_array), len(self.ell_array)
        self.nell = len(self.ell_array)
        self.r200c_mat = BCMP_obj.r200c_mat
        self.rho_dmb_mat = BCMP_obj.rho_dmb_mat
        self.rho_nfw_mat = BCMP_obj.rho_nfw_mat
        # self.sig_logc_z_array = jnp.array(halo_params_dict['sig_logc_z_array'])
        self.beam_fwhm_arcmin = analysis_dict['beam_fwhm_arcmin']      
        self.calc_nfw_only = analysis_dict['calc_nfw_only']  

        if verbose_time:
            ti_pk = time.time()
        self.kPk_array = jnp.logspace(jnp.log10(1E-3), jnp.log10(100), 64)
        self.plin_kz_mat = vmap(linear_matter_power,(None, None, 0))(self.cosmo_jax, self.kPk_array, self.scale_fac_a_array).T
        hfit_params = vmap(halofit_parameters,(None, 0))(self.cosmo_jax, self.scale_fac_a_array).T
        self.phfit_kz_mat = vmap(nonlinear_matter_power,(None, None, 0, None, None, None))(self.cosmo_jax, self.kPk_array, self.scale_fac_a_array, self.plin_kz_mat, hfit_params, self.scale_fac_a_array).T
        
        if verbose_time:
            tf_pk = time.time()
            print('Time taken to setup Pk: ', tf_pk - ti_pk)

        self.chi_array = radial_comoving_distance(self.cosmo_jax, self.scale_fac_a_array)
        self.DA_array = angular_diameter_distance(self.cosmo_jax, self.scale_fac_a_array)
        self.growth_array = bkgrd.growth_factor(self.cosmo_jax, self.scale_fac_a_array)


        if verbose_time:
            ti_hmf = time.time()
        vmap_func1 = vmap(self.get_sigma_Mz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.sigma_Mz_mat = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.nu_Mz_mat = constants.DELTA_COLLAPSE / self.sigma_Mz_mat

        grad_lgsigma = grad(self.get_lgsigma_z, argnums=1)
        vmap_func1 = vmap(grad_lgsigma, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.dlgsig_dlnM_mat = vmap_func2(jnp.arange(self.nz), jnp.log(self.M_array)).T

        # rhom_z_array = (constants.RHO_CRIT_0_KPC3 * self.cosmo_params['Om0'] * (1.0 + self.z_array)**3) * 1E9
        rhom_z_array = (constants.RHO_CRIT_0_KPC3 * self.cosmo_params['Om0'] * jnp.ones_like(self.z_array)) * 1E9
        rhom_z_mat = jnp.repeat(rhom_z_array[None, :], self.nM, axis=0)
        M_z_mat = jnp.repeat(self.M_array[:, None], self.nz, axis=1)
        
        hmf_model = halo_params_dict['hmf_model']
        if hmf_model == 'T08':
            vmap_func1 = vmap(self.get_fsigma_Mz_T08, (0, None))
        elif hmf_model == 'T10':
            vmap_func1 = vmap(self.get_fsigma_Mz_T10, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.fsigma_Mz_mat = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T

        self.hmf_Mz_mat = -1 * self.fsigma_Mz_mat * (rhom_z_mat/M_z_mat).T * self.dlgsig_dlnM_mat

        vmap_func1 = vmap(self.get_bias_Mz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.bias_Mz_mat = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T

        # conc_model = halo_params_dict.get('conc_model','Prada12')
        # if conc_model == 'Prada12':
        #     vmap_func1 = vmap(self.get_conc_Mz_Prada12, (0, None))
        # if conc_model == 'Duffy08':
        #     vmap_func1 = vmap(self.get_conc_Mz_Duffy08, (0, None))
        # if conc_model == 'Diemer15':
        #     vmap_func1 = vmap(self.get_conc_Mz_Diemer15, (0, None))    
        # vmap_func2 = vmap(vmap_func1, (None, 0))
        # self.conc_Mz_mat = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.conc_Mz_mat = BCMP_obj.conc_Mz_mat

        if verbose_time:
            tf_hmf = time.time()
            print('Time taken to setup HMF: ', tf_hmf - ti_hmf)


        if verbose_time:
            ti_uks = time.time()
        
        if self.calc_nfw_only:
            self.rho_nfw_normed_M = BCMP_obj.rho_nfw_mat/Mtot_rep
        self.rho_dmb_normed_M = BCMP_obj.rho_dmb_mat/Mtot_rep
        # if we want the likelihood to be differentiable, need to do the normal integrals. Else do it with mcfit
        # want_like_diff = analysis_dict['want_like_diff']
        self.k = jnp.array(self.kPk_array)
        # if want_like_diff:            
        #     if self.calc_nfw_only:
        #         self.uk_nfw = vmap(self.get_uknfw_from_rho)(jnp.arange(len(self.kPk_array)))
        #     self.uk_dmb = vmap(self.get_ukdmb_from_rho)(jnp.arange(len(self.kPk_array)))
        # else:
        if self.calc_nfw_only:
            # N = 2**(math.ceil(math.log2(halo_params_dict['nr'] * 2)))
            self.k_mcfit, self.uk_nfw = (xi2P(BCMP_obj.r_array, nx=halo_params_dict['nr'], lowring=True)(self.rho_nfw_normed_M, axis=0, extrap=False))
            self.uk_nfw_tointp = jnp.array(self.uk_nfw)
            vmap_func1 = vmap(self.get_uknfw_interp_Pk, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.uk_nfw = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T

        # N = 2**(math.ceil(math.log2(halo_params_dict['nr'] * 2)))
        self.k_mcfit, self.uk_dmb = (xi2P(BCMP_obj.r_array, nx=halo_params_dict['nr'],lowring=True)(self.rho_dmb_normed_M, axis=0, extrap=False))
        self.uk_dmb_tointp = jnp.array(self.uk_dmb)
        vmap_func1 = vmap(self.get_ukdmb_interp_Pk, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.uk_dmb = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T


        # correct the 2halo term for matter. e.g in Cacciato et al 2012, Schmidt 2016, Mead et al 2020:
        do_corr_2h_mm = halo_params_dict['do_corr_2h_mm']
        if do_corr_2h_mm:
            bm_largescales_2h = vmap(self.get_bm_largescales_2h)(jnp.arange((self.nz)))
            self.bm_largescales_2h_mat = jnp.tile(bm_largescales_2h, (len(self.kPk_array), 1))

            vmap_func1 = vmap(self.get_bm_dmb_2h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.bm_dmb_2h = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T

            self.bm_largescales_2h_mat_lt_Mmin = 1. - self.bm_largescales_2h_mat
            self.bm_dmb_kz_mat = self.bm_dmb_2h + self.bm_largescales_2h_mat_lt_Mmin

            if self.calc_nfw_only:
                vmap_func1 = vmap(self.get_bm_nfw_2h, (0, None))
                vmap_func2 = vmap(vmap_func1, (None, 0))
                self.bm_nfw_2h = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T
                self.bm_nfw_kz_mat = self.bm_nfw_2h + self.bm_largescales_2h_mat_lt_Mmin
        
        else:
            self.bm_dmb_kz_mat = jnp.ones((len(self.kPk_array), self.nz))
            if self.calc_nfw_only:
                self.bm_nfw_kz_mat = jnp.ones((len(self.kPk_array), self.nz))

        if self.calc_nfw_only:
            vmap_func1 = vmap(self.get_Pmm_dmb_1h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Pmm_dmb_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T

            vmap_func1 = vmap(self.get_Pmm_nfw_1h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Pmm_nfw_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T

            self.Pmm_dmb_tot_mat = self.Pmm_dmb_1h_mat + (self.bm_dmb_kz_mat)**2 * self.plin_kz_mat
            self.Pmm_nfw_tot_mat = self.Pmm_nfw_1h_mat + (self.bm_nfw_kz_mat)**2 * self.plin_kz_mat
            self.Pmm_sup_tot_mat = self.Pmm_dmb_tot_mat / self.Pmm_nfw_tot_mat


        if verbose_time:
            tf_uks = time.time()
            print('Time taken to setup uks and bks: ', tf_uks - ti_uks)                

        vmap_func1 = vmap(self.get_bkl_dmb, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.bkl_dmb_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T

        if self.calc_nfw_only:
            vmap_func1 = vmap(self.get_bkl_nfw, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.bkl_nfw_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T



        if verbose_time:
            ti_uls = time.time()

        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        coeff = sigmat / (m_e * (c ** 2))
        oneMpc = (((10 ** 6)) * (u.pc).to(u.m)) * (u.m)
        const_coeff = (((coeff * oneMpc).to(((u.cm ** 3) / u.keV))).value)/(self.cosmo_params['H0']/100.)

        # oneMpc_h = (((10 ** 6) / self.cosmo_jax.h) * (u.pc).to(u.m)) * (u.m)
        # const_coeff = ((coeff * oneMpc_h).to(((u.cm ** 3) / u.keV))).value
        # Pe_conv_fac =  0.518
        # vol_universe = (self.scale_fac_a_array)**3
        # self.vol_universe_mat = jnp.tile(vol_universe[None, None, :, None], (self.nr, self.nc, 1, self.nM))
        # self.vol_universe_mat = jnp.ones_like(BCMP_obj.Pth_mat)
        # self.y3d_mat = Pe_conv_fac * const_coeff * BCMP_obj.Pth_mat * self.vol_universe_mat
        self.y3d_mat = const_coeff * BCMP_obj.Pe_mat_physical


        self.sig_beam = self.beam_fwhm_arcmin * (1. / 60.) * (jnp.pi / 180.) * (1. / jnp.sqrt(8. * jnp.log(2)))

        vmap_func1 = vmap(self.get_Pklin_lz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pklin_lz_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T


        # if want_like_diff:
        #     vmap_func1 = vmap(self.get_uyl, (0, None, None, None))
        # else:
        #     vmap_func1 = vmap(self.get_uyl_mcfit, (0, None, None, None))
        vmap_func1 = vmap(self.get_uyl, (0, None, None))
        # vmap_func1 = vmap(self.get_uyl_mcfit, (0, None, None, None))        
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.uyl_mat = vmap_func3(jnp.arange(nell), jnp.arange(self.nz), jnp.arange(self.nM)).T
        
        vmap_func1 = vmap(self.get_byl, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.byl_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T

        vmap_func1 = vmap(self.get_ukappal_dmb_prefac, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.ukappal_dmb_prefac_mat = vmap_func3(jnp.arange(nell), jnp.arange(self.nz), jnp.arange(self.nM)).T

        if self.calc_nfw_only:
            vmap_func1 = vmap(self.get_ukappal_nfw_prefac, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.ukappal_nfw_prefac_mat = vmap_func3(jnp.arange(nell), jnp.arange(self.nz), jnp.arange(self.nM)).T


        if verbose_time:
            tf_uls = time.time()
            print('Time taken to setup uls and bls: ', tf_uls - ti_uls)

        if verbose_time:
            tf = time.time()
            print('Time taken to setup power spectra: ', tf - ti)



    def get_rho_m(self, z):
        return (constants.RHO_CRIT_0_KPC3 * self.Om0 * (1.0 + z)**3) * 1E9

    def get_Ez(self, z):
        zp1 = (1.0 + z)
        t = (self.Om0) * zp1**3 + (1 - self.Om0)
        E = jnp.sqrt(t)        
        return E

    def get_rho_c(self, z):
        return constants.RHO_CRIT_0_KPC3 * self.get_Ez(z)**2  * 1E9    

    @partial(jit, static_argnums=(0,))        
    def get_uknfw_from_rho(self, jk):
        k = self.kPk_array[jk]
        prefac = 4 * jnp.pi * (self.r_array**3) * (jnp.sin(k*self.r_array) / (k*self.r_array))
        prefac_repeat_shape = jnp.tile(prefac.reshape(self.nr,1,1,1), (1,self.nc,self.nz,self.nM))
        uk = jsi.trapezoid(prefac_repeat_shape * self.rho_nfw_normed_M, jnp.log(self.r_array), axis=0)
        return uk

    @partial(jit, static_argnums=(0,))        
    def get_ukdmb_from_rho(self, jk):
        k = self.kPk_array[jk]
        prefac = 4 * jnp.pi * (self.r_array**3) * (jnp.sin(k*self.r_array) / (k*self.r_array))
        prefac_repeat_shape = jnp.tile(prefac.reshape(self.nr,1,1,1), (1,self.nc,self.nz,self.nM))
        uk = jsi.trapezoid(prefac_repeat_shape * self.rho_dmb_normed_M, jnp.log(self.r_array), axis=0)
        return uk


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
    
    def get_conc_Mz_Diemer15(self, jz, jM):
        nu = self.nu_Mz_mat[jz, jM]
        z = self.z_array[jz]
        M = self.M_array[jM]
        DIEMER15_KAPPA = 1.00
        # R = peaks.lagrangianR(M)
        rho_m = (constants.RHO_CRIT_0_KPC3 * self.Om0) * 1E9
        R = (3.0 * M / 4.0 / np.pi / rho_m )**(1.0 / 3.0)
        k_R = 2.0 * np.pi / R * DIEMER15_KAPPA        

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


    # @partial(jit, static_argnums=(0,))
    # def get_uyl(self, jl, jc, jz, jM, xmin=0.01, xmax=4, num_points_trapz_int=64):
    #     r200c = self.r200c_mat[jz, jM]
    #     # z = self.z_array[jz]
    #     # az = 1.0 / (1.0 + z)
    #     # Da_z = angular_diameter_distance(self.cosmo_jax, az)
    #     Da_z = jnp.clip(self.DA_array[jz], 1.0)
    #     l200c = Da_z/r200c
    #     prefac = r200c/l200c**2
    #     logx_array = jnp.linspace(jnp.log(xmin), jnp.log(xmax), num_points_trapz_int)
    #     x_array = jnp.exp(logx_array)

    #     y3d_min = jnp.min(jnp.absolute(self.y3d_mat[:,jc, jz, jM]))
    #     y3d_clipped = jnp.clip(self.y3d_mat[:,jc, jz, jM], y3d_min + 1e-25)
    #     # y3d_xarray = jnp.exp(jnp.interp(logx_array, jnp.log(self.r_array/r200c), jnp.log(self.y3d_mat[:,jc, jz, jM])))
    #     y3d_xarray = jnp.exp(jnp.interp(logx_array, jnp.log(self.r_array/r200c), jnp.log(y3d_clipped)))        
    #     ell = self.ell_array[jl]
    #     sin_fac = (jnp.sin(ell*x_array/l200c))/(ell*x_array/l200c)

    #     fx = y3d_xarray * sin_fac * (4*jnp.pi*x_array**2) * x_array
    #     uyl = prefac * jsi.trapezoid(fx, x=logx_array)
    #     Bl = jnp.exp(-1. * ell * (ell + 1) * (self.sig_beam ** 2) / 2.)
    #     return uyl * Bl
    
    @partial(jit, static_argnums=(0,))
    def get_uyl(self, jl, jz, jM, xmin=0.001, xmax=10, num_points_trapz_int=4000):
        chiz = jnp.clip(self.chi_array[jz], 1.0)
        az = 1.0 / (1.0 + self.z_array[jz])
        prefac = az/(chiz**2)
        rmin = xmin * self.r200c_mat[jz, jM]
        rmax = xmax * self.r200c_mat[jz, jM]
        logr_array_int = jnp.linspace(jnp.log(rmin), jnp.log(rmax), num_points_trapz_int)
        r_array_int = jnp.exp(logr_array_int)

        y3d_min = jnp.min(jnp.absolute(self.y3d_mat[:,jz, jM]))
        y3d_clipped = jnp.clip(self.y3d_mat[:,jz, jM], y3d_min + 1e-30)
        y3d_rarray = jnp.exp(jnp.interp(logr_array_int, jnp.log(self.r_array), jnp.log(y3d_clipped)))        
        ell = self.ell_array[jl]
        sin_fac = (jnp.sin((ell + 0.5)*r_array_int/chiz))/(((ell + 0.5)*r_array_int/chiz))

        fx = y3d_rarray * sin_fac * (4*jnp.pi*r_array_int**2) * r_array_int
        uyl = prefac * jsi.trapezoid(fx, x=logr_array_int) 
        Bl = jnp.exp(-1. * ell * (ell + 1) * (self.sig_beam ** 2) / 2.)
        return uyl * Bl

    # @partial(jit, static_argnums=(0,))
    # def get_uyl_mcfit(self, jl, jc, jz, jM, xmin=0.01, xmax=3, num_points_trapz_int=128):
    #     chiz = jnp.clip(self.chi_array[jz], 1.0)
    #     az = 1.0 / (1.0 + self.z_array[jz])
    #     prefac = az/(chiz**2)
    #     rmin = xmin * self.r200c_mat[jz, jM]
    #     rmax = xmax * self.r200c_mat[jz, jM]        

    #     y3d_min = jnp.min(jnp.absolute(self.y3d_mat[:,jc, jz, jM]))
    #     y3d_clipped = jnp.clip(self.y3d_mat[:,jc, jz, jM], y3d_min + 1e-30)
    #     # logr_array_mcfit = jnp.linspace(jnp.log(jnp.min(self.r_array/chiz)), jnp.log(jnp.max(self.r_array/chiz)), num_points_trapz_int)
    #     logr_array_mcfit = logr_array_int = jnp.linspace(jnp.log(rmin), jnp.log(rmax), num_points_trapz_int)
    #     r_array_mcfit = jnp.exp(logr_array_mcfit)
    #     # y3d_array = jnp.exp(jnp.interp(jnp.log(r_array_mcfit), jnp.log(self.r_array/chiz), jnp.log(y3d_clipped)))
    #     y3d_rarray = jnp.exp(jnp.interp(logr_array_int, jnp.log(self.r_array), jnp.log(y3d_clipped)))        
    #     k_mcfit, uy_mcfit = (xi2P(jnp.array(r_array_mcfit), nx=num_points_trapz_int,lowring=True)(jnp.array(y3d_rarray),  extrap=False))
    #     uy_mcfit = jnp.array(uy_mcfit)
    #     ell = self.ell_array[jl]
    #     uyl = prefac *  (chiz**3) * jnp.exp(jnp.interp(jnp.log(ell), jnp.log(k_mcfit), jnp.log(uy_mcfit)))
    #     Bl = jnp.exp(-1. * ell * (ell + 1) * (self.sig_beam ** 2) / 2.) 
    #     return uyl * Bl     


    @partial(jit, static_argnums=(0,))
    def get_byl(self, jl, jz):
        # uyl_jl_jz = self.uyl_mat[jl, :, jz, :]
        # cmean_jz = self.conc_Mz_mat[jz, :]
        # logc_array = jnp.log(self.conc_array)
        # sig_logc = self.sig_logc_z_array[jz]
        # conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        # cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        # p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))
        # fx = uyl_jl_jz.T * p_logc_Mz
        # uyl_intc = jsi.trapezoid(fx, x=logc_array)
        uyl_intc = self.uyl_mat[jl, jz, :]     

        dndlnM_z = self.hmf_Mz_mat[jz, :]
        bM_z = self.bias_Mz_mat[jz, :]
        fx = uyl_intc * dndlnM_z * bM_z
        byl = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return byl

    @partial(jit, static_argnums=(0,))
    def get_Pklin_lz(self, jl, jz):
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.plin_kz_mat[:,jz])))
        return Pkz_ell


    @partial(jit, static_argnums=(0,))
    def get_ukappal_dmb_prefac(self, jl, jz, jM):
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        # uk_dmb_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(self.uk_dmb[:,jc, jz, jM])))
        uk_min = jnp.min(jnp.absolute(self.uk_dmb[:,jz, jM]))
        # uk_clipped = jnp.clip(self.uk_dmb[:,jc, jz, jM], uk_min + 1e-25) * self.M_array[jM]/self.rho_m_bar
        uk_clipped = jnp.clip(self.uk_dmb[:,jz, jM], uk_min + 1e-25) * self.Mtot_mat[jz, jM]/self.rho_m_bar        
        uk_dmb_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(uk_clipped)))        
        return uk_dmb_ell

    @partial(jit, static_argnums=(0,))
    def get_ukappal_nfw_prefac(self, jl, jz, jM):
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        # uk_nfw_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(self.uk_nfw[:,jc, jz, jM])))
        uk_min = jnp.min(jnp.absolute(self.uk_nfw[:,jz, jM]))
        # uk_clipped = jnp.clip(self.uk_nfw[:,jc, jz, jM], uk_min + 1e-25) * self.M_array[jM]/self.rho_m_bar
        uk_clipped = jnp.clip(self.uk_nfw[:,jz, jM], uk_min + 1e-25) * self.Mtot_mat[jz, jM]/self.rho_m_bar        
        uk_nfw_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(uk_clipped)))        
        return uk_nfw_ell
    
    @partial(jit, static_argnums=(0,))
    def get_ukdmb_interp_Pk(self, jz, jM):
        ukdmb_array_kPk = jnp.exp(jnp.interp(jnp.log(self.kPk_array), jnp.log(self.k_mcfit), jnp.log(self.uk_dmb_tointp[:,jz, jM])))
        return ukdmb_array_kPk

    @partial(jit, static_argnums=(0,))
    def get_uknfw_interp_Pk(self, jz, jM):
        uknfw_array_kPk = jnp.exp(jnp.interp(jnp.log(self.kPk_array), jnp.log(self.k_mcfit), jnp.log(self.uk_nfw_tointp[:,jz, jM])))
        return uknfw_array_kPk

    @partial(jit, static_argnums=(0,))
    def get_bm_dmb_2h(self, jk, jz):
        '''Function getting the 2halo effective bias of the matter fields'''
        # cmean_jz = self.conc_Mz_mat[jz, :]
        # logc_array = jnp.log(self.conc_array)
        # sig_logc = self.sig_logc_z_array[jz]
        # conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        # cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        # p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))
        
        # fx = ((self.Mtot_mat[:, jz, :] *  self.uk_dmb[jk,:,jz,:])).T * p_logc_Mz
        # ukz_intc = jsi.trapezoid(fx, x=logc_array)
        ukz_intc = self.Mtot_mat[jz, :] *  self.uk_dmb[jk,jz,:]
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        # rhom_z = self.get_rho_m(self.z_array[jz])
        rhom_z = self.get_rho_m(0.0)
        fx = ukz_intc * dndlnM_z * self.bias_Mz_mat[jz,:] * ((1/rhom_z))
        bmm_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return bmm_2h

    @partial(jit, static_argnums=(0,))
    def get_bm_largescales_2h(self, jz):
        '''Get the large scale limit of the above 2halo integral'''
        # cmean_jz = self.conc_Mz_mat[jz, :]
        # logc_array = jnp.log(self.conc_array)
        # sig_logc = self.sig_logc_z_array[jz]
        # conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        # cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        # p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))
        
        # fx = ((self.Mtot_mat[:, jz, :])).T * p_logc_Mz
        # ukz_intc = jsi.trapezoid(fx, x=logc_array)
        ukz_intc = self.Mtot_mat[jz, :]
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        rhom_z = self.get_rho_m(0.0) #want comoving density
        fx = ukz_intc * dndlnM_z * self.bias_Mz_mat[jz,:] * ((1/rhom_z))
        bmm_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return bmm_2h

    @partial(jit, static_argnums=(0,))
    def get_bm_nfw_2h(self, jk, jz):
        '''Function getting the 2halo effective bias of the matter fields'''
        # cmean_jz = self.conc_Mz_mat[jz, :]
        # logc_array = jnp.log(self.conc_array)
        # sig_logc = self.sig_logc_z_array[jz]
        # conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        # cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        # p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))
        
        # fx = ((self.Mtot_mat[:, jz, :] *  self.uk_nfw[jk,:,jz,:])).T * p_logc_Mz
        # ukz_intc = jsi.trapezoid(fx, x=logc_array)
        ukz_intc = self.Mtot_mat[jz, :] *  self.uk_nfw[jk,jz,:]
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        rhom_z = self.get_rho_m(0.0) #want comoving density
        fx = ukz_intc * dndlnM_z * self.bias_Mz_mat[jz,:] * ((1/rhom_z))
        bmm_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return bmm_2h


    @partial(jit, static_argnums=(0,))
    def get_bkl_dmb(self, jl, jz):
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        bkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(self.bm_dmb_kz_mat[:,jz])))
        return bkz_ell

    @partial(jit, static_argnums=(0,))
    def get_bkl_nfw(self, jl, jz):
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        bkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(self.bm_nfw_kz_mat[:,jz])))
        return bkz_ell


    @partial(jit, static_argnums=(0,))
    def get_Pmm_dmb_1h(self, jk, jz):
        # cmean_jz = self.conc_Mz_mat[jz, :]
        # logc_array = jnp.log(self.conc_array)
        # sig_logc = self.sig_logc_z_array[jz]
        # conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        # cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        # p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))
        # fx = ((self.Mtot_mat[:, jz, :] *  self.uk_dmb[jk,:,jz,:])**2).T * p_logc_Mz
        # ukz_intc = jsi.trapezoid(fx, x=logc_array)
        ukz_intc = (self.Mtot_mat[jz, :] *  self.uk_dmb[jk,jz,:])**2        
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        rhom_z = self.get_rho_m(0.0) #want comoving density
        fx = ukz_intc * dndlnM_z * ((1/rhom_z)**2)
        Pmm_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return Pmm_1h

    @partial(jit, static_argnums=(0,))
    def get_Pmm_nfw_1h(self, jk, jz):
        # cmean_jz = self.conc_Mz_mat[jz, :]
        # logc_array = jnp.log(self.conc_array)
        # sig_logc = self.sig_logc_z_array[jz]
        # conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        # cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        # p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))        
        # fx = ((self.Mtot_mat[:, jz, :] *  self.uk_nfw[jk,:,jz,:])**2).T * p_logc_Mz
        # ukz_intc = jsi.trapezoid(fx, x=logc_array)
        ukz_intc = (self.Mtot_mat[jz, :] *  self.uk_nfw[jk,jz,:])**2
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        rhom_z = self.get_rho_m(0.0) #want comoving density
        fx = ukz_intc * dndlnM_z * ((1/rhom_z)**2)
        Pmm_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return Pmm_1h


    @partial(jit, static_argnums=(0,))
    def get_dPmm_dmb_dlnM_1h(self, jk, jz):
        # cmean_jz = self.conc_Mz_mat[jz, :]
        # logc_array = jnp.log(self.conc_array)
        # sig_logc = self.sig_logc_z_array[jz]
        # conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        # cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        # p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))        
        # fx = ((self.Mtot_mat[:, jz, :] *  self.uk_dmb[jk,:,jz,:])**2).T * p_logc_Mz
        # ukz_intc = jsi.trapezoid(fx, x=logc_array)
        ukz_intc = (self.Mtot_mat[jz, :] *  self.uk_dmb[jk,jz,:])**2
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        rhom_z = self.get_rho_m(0.0) #want comoving density
        fx = ukz_intc * dndlnM_z * ((1/rhom_z)**2)
        # Pmm_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return fx

    @partial(jit, static_argnums=(0,))
    def get_dPmm_nfw_dlnM_1h(self, jk, jz):
        # cmean_jz = self.conc_Mz_mat[jz, :]
        # logc_array = jnp.log(self.conc_array)
        # sig_logc = self.sig_logc_z_array[jz]
        # conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        # cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        # p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))        
        # fx = ((self.Mtot_mat[:, jz, :] *  self.uk_nfw[jk,:,jz,:])**2).T * p_logc_Mz
        # ukz_intc = jsi.trapezoid(fx, x=logc_array)
        ukz_intc = (self.Mtot_mat[jz, :] *  self.uk_nfw[jk,jz,:])**2
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        rhom_z = self.get_rho_m(0.0) #want comoving density
        fx = ukz_intc * dndlnM_z * ((1/rhom_z)**2)
        # Pmm_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return fx

