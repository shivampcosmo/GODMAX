import os, sys
from get_radial_profiles import BCM_18_wP
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from jax import vmap, grad
import math
import jax
from jax_cosmo import Cosmology
from functools import partial
# from jax_cosmo.power import linear_matter_power
from helpers.jax_cosmo_power import linear_matter_power, halofit_parameters, nonlinear_matter_power
from jax_cosmo.background import angular_diameter_distance, radial_comoving_distance
import jax_cosmo.transfer as tklib
import astropy.units as u
from astropy import constants as const
import jax.scipy.integrate as jsi
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.keV / u.cm**3)).value
G_new_rhom = const.G.to(u.Mpc**3 / ((u.s**2) * u.M_sun))
import helpers.constants as constants
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
        func = vmap(func, (0, None, None))
        func = vmap(func, (None, 0, None))
        return func
    if (num_args == 3) and (num_args2 == 4):
        func = vmap(func, (0, None, None, None))
        func = vmap(func, (None, 0, None, None))
        func = vmap(func, (None, None, 0, None))
        return func

class setup_power_BCMP(BCM_18_wP):
    def __init__(
                self,
                sim_params_dict,
                halo_params_dict,
                analysis_dict,
                num_points_trapz_int=64,
                BCMP_obj=None,
                verbose_time=False
            ):    
        
        if BCMP_obj is None:
            super().__init__(sim_params_dict, halo_params_dict, num_points_trapz_int=num_points_trapz_int, verbose_time=verbose_time, analysis_dict=analysis_dict)
        else:
            self.__dict__.update(BCMP_obj.__dict__)


        self.bias_Mz_mat = get_vmapped_func(self.BCMP_obj.get_bias_Mz, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T

        # self.conc_Mz_mat = BCMP_obj.conc_Mz_mat

        # self.k = jnp.array(self.kPk_array)

        self.k_mcfit, self.uk_dmb = (xi2P(self.r_array, nx=halo_params_dict['nr'],lowring=True)(self.rho_dmb_mat / self.Mtot_mat[None, :, :], axis=0, extrap=False))
        self.uk_dmb_tointp = jnp.array(self.uk_dmb)
        # vmap_func1 = vmap(self.get_ukdmb_interp_Pk, (0, None))
        # vmap_func2 = vmap(vmap_func1, (None, 0))
        # self.uk_dmb = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.uk_dmb = get_vmapped_func_warg(self.BCMP_obj.get_uk_from_interp_Pk, 2, 3)(jnp.arange(self.nz), jnp.arange(self.nM), 'dmb').T

        self.k_mcfit, self.uk_clm = (xi2P(self.r_array, nx=halo_params_dict['nr'],lowring=True)(self.rho_clm_mat / self.Mclm_mat[-1, :, :][None, :, :], axis=0, extrap=False))
        self.uk_clm_tointp = jnp.array(self.uk_clm)
        # vmap_func1 = vmap(self.get_ukclm_interp_Pk, (0, None))
        # vmap_func2 = vmap(vmap_func1, (None, 0))
        # self.uk_clm = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.uk_clm = get_vmapped_func_warg(self.BCMP_obj.get_uk_from_interp_Pk, 2, 3)(jnp.arange(self.nz), jnp.arange(self.nM), 'clm').T


        self.nbarz = jsi.trapezoid(self.hmf_Mz_mat * (self.Ncen + self.Nsat), jnp.log(self.M_array), axis=-1)

        self.ukg_cross = (self.Ncen[None,:,:] + self.Nsat[None,:,:] * self.uk_clm)/self.nbarz[None,:,None]
        ukg_auto_arg = jnp.clip(jnp.nan_to_num(2 * self.Ncen[None,:,:] * self.Nsat[None,:,:] * self.uk_clm + (self.Nsat[None,:,:] * self.uk_clm)**2), 1e-10, 2e4)
        self.ukg_auto_sqr = (ukg_auto_arg)/(self.nbarz[None,:,None] ** 2)


        self.k_mcfit, self.uk_ne = (xi2P(BCMP_obj.r_array, nx=halo_params_dict['nr'],lowring=True)(BCMP_obj.ne_mat / BCMP_obj.ne_mat_norm[-1, :, :][None, :, :], axis=0, extrap=False))
        self.uk_ne_tointp = jnp.array(self.uk_ne)

        # vmap_func1 = vmap(self.get_ukne_interp_Pk, (0, None))
        # vmap_func2 = vmap(vmap_func1, (None, 0))
        # self.uk_ne = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.uk_ne = get_vmapped_func_warg(self.BCMP_obj.get_uk_from_interp_Pk, 2, 3)(jnp.arange(self.nz), jnp.arange(self.nM), 'ne').T
        # self.uk_ne = self.uk_ne/self.nebarz[None,:,None]


        # correct the 2halo term for matter. e.g in Cacciato et al 2012, Schmidt 2016, Mead et al 2020:
        if self.do_corr_2h_mm:
            bm_largescales_2h = vmap(self.get_bm_largescales_2h)(jnp.arange((self.nz)))
            self.bm_largescales_2h_mat = jnp.tile(bm_largescales_2h, (len(self.kPk_array), 1))

            # vmap_func1 = vmap(self.get_bm_dmb_2h, (0, None))
            # vmap_func2 = vmap(vmap_func1, (None, 0))
            # self.bm_dmb_2h = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T
            self.bm_dmb_2h = get_vmapped_func_warg(self.get_b_2h, 2, 3)(jnp.arange(self.nz), jnp.arange(self.nM), 'dmb').T
            self.bm_largescales_2h_mat_lt_Mmin = 1. - self.bm_largescales_2h_mat
            self.bm_dmb_kz_mat = self.bm_dmb_2h + self.bm_largescales_2h_mat_lt_Mmin

        else:
            self.bm_dmb_kz_mat = jnp.ones((len(self.kPk_array), self.nz))


        vmap_func1 = vmap(self.get_Pmm_dmb_1h, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pmm_dmb_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T
        self.Pmm_dmb_tot_mat = ((self.Pmm_dmb_1h_mat)**(self.alpha_1h2h_mm) + ((self.bm_dmb_kz_mat)**2 * self.plin_kz_mat)**(self.alpha_1h2h_mm))**(1/self.alpha_1h2h_mm)


        if (self.smooth_1h2h_mm_model == 'response') or (self.smooth_1h2h_ym_model == 'response'):

            hfit_params = vmap(halofit_parameters,(None, 0))(self.cosmo_jax, self.scale_fac_a_array).T
            self.phfit_kz_mat = vmap(nonlinear_matter_power,(None, None, 0, None, None, None))(self.cosmo_jax, self.kPk_array, self.scale_fac_a_array, self.plin_kz_mat, hfit_params, self.scale_fac_a_array).T

            self.rho_nfw_normed_M = BCMP_obj.rho_nfw_mat/Mtot_rep

            self.k_mcfit, self.uk_nfw = (xi2P(BCMP_obj.r_array, nx=halo_params_dict['nr'], lowring=True)(self.rho_nfw_normed_M, axis=0, extrap=False))
            self.uk_nfw_tointp = jnp.array(self.uk_nfw)
            vmap_func1 = vmap(self.get_uknfw_interp_Pk, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.uk_nfw = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T

            if do_corr_2h_mm:
                vmap_func1 = vmap(self.get_bm_nfw_2h, (0, None))
                vmap_func2 = vmap(vmap_func1, (None, 0))
                self.bm_nfw_2h = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T
                self.bm_nfw_kz_mat = self.bm_nfw_2h + self.bm_largescales_2h_mat_lt_Mmin            
            else:
                self.bm_nfw_kz_mat = jnp.ones((len(self.kPk_array), self.nz))


            vmap_func1 = vmap(self.get_Pmm_nfw_1h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Pmm_nfw_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T

            self.Pmm_nfw_tot_mat = self.Pmm_nfw_1h_mat + (self.bm_nfw_kz_mat)**2 * self.plin_kz_mat
            self.Pmm_sup_tot_mat = self.phfit_kz_mat / self.Pmm_nfw_tot_mat

        # if (self.smooth_1h2h_mm_model == 'response'):
            self.Pmm_tot_kz_mat = self.Pmm_dmb_tot_mat * self.Pmm_sup_tot_mat
        else:
            self.Pmm_tot_kz_mat = self.Pmm_dmb_tot_mat


        self.k_mcfit, self.uk_y = (xi2P(self.r_array, nx=halo_params_dict['nr'], lowring=True)(self.y3d_mat, axis=0, extrap=False))
        self.uk_y_tointp = jnp.array(self.uk_y)
        # vmap_func1 = vmap(self.get_uky_interp_Pk, (0, None))
        # vmap_func2 = vmap(vmap_func1, (None, 0))
        # self.uk_y = vmap_func2(jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.uk_y = get_vmapped_func_warg(self.BCMP_obj.get_uk_from_interp_Pk, 2, 3)(jnp.arange(self.nz), jnp.arange(self.nM), 'Pe').T

        # vmap_func1 = vmap(self.get_bk_y_2h, (0, None))
        # vmap_func2 = vmap(vmap_func1, (None, 0))
        # self.by_kz_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T
        self.by_kz_mat = get_vmapped_func_warg(self.BCMP_obj.get_b_2h, 2, 3)(jnp.arange(self.nz), jnp.arange(self.nM), 'Pe').T

        vmap_func1 = vmap(self.get_Pym_dmb_1h, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pym_dmb_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T

        self.Pym_dmb_tot_mat = ((self.Pym_dmb_1h_mat)**(self.alpha_1h2h_ym) + ((self.by_kz_mat) * (self.bm_dmb_kz_mat) * self.plin_kz_mat)**(self.alpha_1h2h_ym))**(1/self.alpha_1h2h_ym)

        if (self.smooth_1h2h_ym_model == 'response'):
            self.Pym_tot_kz_mat = self.Pym_dmb_tot_mat * self.Pmm_sup_tot_mat
        else:
            self.Pym_tot_kz_mat = self.Pym_dmb_tot_mat



        vmap_func1 = vmap(self.get_Pge_1h, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pge_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T

        vmap_func1 = vmap(self.get_bk_g_2h, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.bg_kz_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T

        vmap_func1 = vmap(self.get_bk_e_2h, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.be_kz_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T

        self.Pge_2h_mat = self.bg_kz_mat * self.be_kz_mat * self.plin_kz_mat

        self.Pge_tot_kz_mat = self.Pge_1h_mat + self.Pge_2h_mat
        if (self.smooth_1h2h_ge_model == 'response'):
            self.Pge_tot_kz_mat = self.Pge_tot_kz_mat * self.Pmm_sup_tot_mat



        self.sig_beam = self.beam_fwhm_arcmin * (1. / 60.) * (jnp.pi / 180.) * (1. / jnp.sqrt(8. * jnp.log(2)))

        # need to get Pgkappa and Pgy

        vmap_func1 = vmap(self.get_Pgm_1h, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pgm_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T
        self.Pgm_2h_mat = self.bg_kz_mat * (self.bm_dmb_kz_mat) * self.plin_kz_mat
        self.Pgm_tot_kz_mat = self.Pgm_1h_mat + self.Pgm_2h_mat
        if (self.smooth_1h2h_gm_model == 'response'):
            self.Pgm_tot_kz_mat = self.Pgm_tot_kz_mat * self.Pmm_sup_tot_mat

        
        if self.calc_nfw_only:
            vmap_func1 = vmap(self.get_Pgm_nfw_1h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Pgm_nfw_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T
            self.Pgm_nfw_2h_mat = self.bg_kz_mat * (self.bm_nfw_kz_mat) * self.plin_kz_mat
            self.Pgm_nfw_tot_kz_mat = self.Pgm_nfw_1h_mat + self.Pgm_nfw_2h_mat
            if (self.smooth_1h2h_gm_model == 'response'):
                self.Pgm_nfw_tot_kz_mat = self.Pgm_nfw_tot_kz_mat * self.Pmm_sup_tot_mat


        vmap_func1 = vmap(self.get_Pgy_1h, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pgy_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T
        self.Pgy_2h_mat = self.bg_kz_mat * (self.by_kz_mat) * self.plin_kz_mat
        self.Pgy_tot_kz_mat = self.Pgy_1h_mat + self.Pgy_2h_mat
        if (self.smooth_1h2h_gy_model == 'response'):
            self.Pgy_tot_kz_mat = self.Pgy_tot_kz_mat * self.Pmm_sup_tot_mat


        vmap_func1 = vmap(self.get_Pgg_1h, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pgg_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T
        self.Pgg_2h_mat = self.bg_kz_mat * (self.bg_kz_mat) * self.plin_kz_mat
        self.Pgg_tot_kz_mat = self.Pgg_1h_mat + self.Pgg_2h_mat
        if (self.smooth_1h2h_gg_model == 'response'):
            self.Pgg_tot_kz_mat = self.Pgg_tot_kz_mat * self.Pmm_sup_tot_mat


        # vmap_func1 = vmap(self.get_Pklin_lz, (0, None))
        # vmap_func2 = vmap(vmap_func1, (None, 0))
        # self.Pklin_lz_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T

        vmap_func1 = vmap(self.get_Pkmm_lz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pkmm_lz_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T

        vmap_func1 = vmap(self.get_Pkgm_lz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pkgm_lz_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T

        if self.calc_nfw_only:
            vmap_func1 = vmap(self.get_Pkmm_halofit_lz, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Pkmm_nfw_lz_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T

            vmap_func1 = vmap(self.get_Pkgm_nfw_lz, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Pkgm_nfw_lz_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T            

        vmap_func1 = vmap(self.get_Pkym_lz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pkym_lz_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T


        vmap_func1 = vmap(self.get_Pkgy_lz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pkgy_lz_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T


        vmap_func1 = vmap(self.get_Pkgg_lz, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pkgg_lz_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T

        self.get_cov = analysis_dict.get('get_cov',False)  
        if self.get_cov:
            vmap_func1 = vmap(self.get_uyl, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.uyl_mat = vmap_func3(jnp.arange(nell), jnp.arange(self.nz), jnp.arange(self.nM)).T

            vmap_func1 = vmap(self.get_ukappal_dmb_prefac, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.ukappal_dmb_prefac_mat = vmap_func3(jnp.arange(nell), jnp.arange(self.nz), jnp.arange(self.nM)).T

            vmap_func1 = vmap(self.get_ugl_cross, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.ugl_mat = vmap_func3(jnp.arange(nell), jnp.arange(self.nz), jnp.arange(self.nM)).T

            vmap_func1 = vmap(self.get_Pyy_1h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Pyy_1h_mat = vmap_func2(jnp.arange(len(self.kPk_array)), jnp.arange(self.nz)).T
            self.Pyy_2h_mat = self.by_kz_mat * (self.by_kz_mat) * self.plin_kz_mat
            self.Pyy_tot_kz_mat = self.Pyy_1h_mat + self.Pyy_2h_mat

            vmap_func1 = vmap(self.get_Pkyy_lz, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Pkyy_lz_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T

        # # if want_like_diff:
        # #     vmap_func1 = vmap(self.get_uyl, (0, None, None, None))
        # # else:
        # #     vmap_func1 = vmap(self.get_uyl_mcfit, (0, None, None, None))
        # vmap_func1 = vmap(self.get_uyl, (0, None, None))
        # # vmap_func1 = vmap(self.get_uyl_mcfit, (0, None, None, None))        
        # vmap_func2 = vmap(vmap_func1, (None, 0, None))
        # vmap_func3 = vmap(vmap_func2, (None, None, 0))
        # self.uyl_mat = vmap_func3(jnp.arange(nell), jnp.arange(self.nz), jnp.arange(self.nM)).T
        
        # vmap_func1 = vmap(self.get_byl, (0, None))
        # vmap_func2 = vmap(vmap_func1, (None, 0))
        # self.byl_mat = vmap_func2(jnp.arange(nell), jnp.arange(self.nz)).T

        # vmap_func1 = vmap(self.get_ukappal_dmb_prefac, (0, None, None))
        # vmap_func2 = vmap(vmap_func1, (None, 0, None))
        # vmap_func3 = vmap(vmap_func2, (None, None, 0))
        # self.ukappal_dmb_prefac_mat = vmap_func3(jnp.arange(nell), jnp.arange(self.nz), jnp.arange(self.nM)).T

        # if self.calc_nfw_only:
        #     vmap_func1 = vmap(self.get_ukappal_nfw_prefac, (0, None, None))
        #     vmap_func2 = vmap(vmap_func1, (None, 0, None))
        #     vmap_func3 = vmap(vmap_func2, (None, None, 0))
        #     self.ukappal_nfw_prefac_mat = vmap_func3(jnp.arange(nell), jnp.arange(self.nz), jnp.arange(self.nM)).T


        # if verbose_time:
        #     tf_uls = time.time()
        #     print('Time taken to setup uls and bls: ', tf_uls - ti_uls)

        # if verbose_time:
        #     tf = time.time()
        #     print('Time taken to setup power spectra: ', tf - ti)

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

    def get_rho_m(self, z):
        return (constants.RHO_CRIT_0_KPC3 * self.Om0 * (1.0 + z)**3) * 1E9

    def get_Ez(self, z):
        zp1 = (1.0 + z)
        t = (self.Om0) * zp1**3 + (1 - self.Om0)
        E = jnp.sqrt(t)        
        return E

    def get_rho_c(self, z):
        return constants.RHO_CRIT_0_KPC3 * self.get_Ez(z)**2  * 1E9    

    # @partial(jit, static_argnums=(0,))        
    # def get_uknfw_from_rho(self, jk):
    #     k = self.kPk_array[jk]
    #     prefac = 4 * jnp.pi * (self.r_array**3) * (jnp.sin(k*self.r_array) / (k*self.r_array))
    #     prefac_repeat_shape = jnp.tile(prefac.reshape(self.nr,1,1,1), (1,self.nc,self.nz,self.nM))
    #     uk = jsi.trapezoid(prefac_repeat_shape * self.rho_nfw_normed_M, jnp.log(self.r_array), axis=0)
    #     return uk

    # @partial(jit, static_argnums=(0,))        
    # def get_ukdmb_from_rho(self, jk):
    #     k = self.kPk_array[jk]
    #     prefac = 4 * jnp.pi * (self.r_array**3) * (jnp.sin(k*self.r_array) / (k*self.r_array))
    #     prefac_repeat_shape = jnp.tile(prefac.reshape(self.nr,1,1,1), (1,self.nc,self.nz,self.nM))
    #     uk = jsi.trapezoid(prefac_repeat_shape * self.rho_dmb_normed_M, jnp.log(self.r_array), axis=0)
    #     return uk

    # @partial(jit, static_argnums=(0,))        
    # def get_uky_from_P(self, jk):
    #     k = self.kPk_array[jk]
    #     prefac = 4 * jnp.pi * (self.r_array**3) * (jnp.sin(k*self.r_array) / (k*self.r_array))
    #     prefac_repeat_shape = jnp.tile(prefac.reshape(self.nr,1,1,1), (1,self.nc,self.nz,self.nM))
    #     uk = jsi.trapezoid(prefac_repeat_shape * self.y3d_mat, jnp.log(self.r_array), axis=0)
    #     return uk

    @partial(jit, static_argnums=(0,))        
    def get_uk_from_xi(self, jk, probe):
        k = self.kPk_array[jk]
        prefac = 4 * jnp.pi * (self.r_array**3) * (jnp.sin(k*self.r_array) / (k*self.r_array))
        prefac_repeat_shape = jnp.tile(prefac.reshape(self.nr,1,1,1), (1,self.nc,self.nz,self.nM))
        if probe == 'dmb':
            xi_mat = self.rho_dmb_normed_M
        elif probe == 'nfw':
            xi_mat = self.rho_nfw_normed_M
        elif probe == 'Pe':
            xi_mat = self.y3d_mat
        uk = jsi.trapezoid(prefac_repeat_shape * xi_mat, jnp.log(self.r_array), axis=0)
        return uk        
    
    # def get_Ncen_Nsat(self):
    #     if self.hod_type == 'Zheng05':
    #         Ncen = 0.5 * (1.0 + jnp.tanh((jnp.log10(self.M_array) - self.hod_params['logMmin']) / self.hod_params['sigma_logM']))
    #         Nsat = jnp.zeros_like(Ncen)
    #         # indsel = jnp.where(self.M_array > 10**self.hod_params['logM0'])
    #         value = Ncen * (jnp.abs(self.M_array - 10**self.hod_params['logM0']) / 10**self.hod_params['logM1'])**self.hod_params['alpha']
    #         Nsat = jnp.where(self.M_array > 10**self.hod_params['logM0'], value, 1e-30)            
    #         # Nsat_gt0 = Ncen[indsel] * ((self.M_array[indsel] - 10**self.hod_params['logM0']) / 10**self.hod_params['logM1'])**self.hod_params['alpha']
    #         # Nsat = Nsat.at[indsel].set(Nsat_gt0)

    #     return Ncen, Nsat
   
    
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
    def get_uk_from_interp_Pk(self, jz, jM, probe):
        if probe == 'dmb':
            uk_val = jnp.clip(self.uk_dmb_tointp[:,jz, jM], 1e-30, 1)
        elif probe == 'nfw':
            uk_val = jnp.clip(self.uk_nfw_tointp[:,jz, jM], 1e-30, 1)
        elif probe == 'clm':
            uk_val = jnp.clip(self.uk_clm_tointp[:,jz, jM], 1e-30, 1)
        elif probe == 'Pe':
            uk_val = self.uk_y_tointp[:,jz, jM]
        elif probe == 'ne':
            uk_val = self.uk_ne_tointp[:,jz, jM]
        else:
            raise ValueError('Probe not recognized')
        # ukdmb_val = jnp.clip(self.uk_dmb_tointp[:,jz, jM], 1e-30, 1)
        ukdmb_array_kPk = jnp.exp(jnp.interp(jnp.log(self.kPk_array), jnp.log(self.k_mcfit), jnp.log(uk_val)))
        return ukdmb_array_kPk


    # @partial(jit, static_argnums=(0,))
    # def get_ukdmb_interp_Pk(self, jz, jM):
    #     ukdmb_val = jnp.clip(self.uk_dmb_tointp[:,jz, jM], 1e-30, 1)
    #     ukdmb_array_kPk = jnp.exp(jnp.interp(jnp.log(self.kPk_array), jnp.log(self.k_mcfit), jnp.log(ukdmb_val)))
    #     return ukdmb_array_kPk

    # @partial(jit, static_argnums=(0,))
    # def get_ukclm_interp_Pk(self, jz, jM):
    #     ukclm_val = jnp.clip(self.uk_clm_tointp[:,jz, jM], 1e-30, 1)
    #     ukclm_array_kPk = jnp.exp(jnp.interp(jnp.log(self.kPk_array), jnp.log(self.k_mcfit), jnp.log(ukclm_val)))
    #     return ukclm_array_kPk

    # @partial(jit, static_argnums=(0,))
    # def get_uknfw_interp_Pk(self, jz, jM):
    #     uknfw_val = jnp.clip(self.uk_nfw_tointp[:,jz, jM], 1e-30, 1)
    #     uknfw_array_kPk = jnp.exp(jnp.interp(jnp.log(self.kPk_array), jnp.log(self.k_mcfit), jnp.log(uknfw_val)))
    #     return uknfw_array_kPk

    # @partial(jit, static_argnums=(0,))
    # def get_uky_interp_Pk(self, jz, jM):
    #     uky_array_kPk = jnp.exp(jnp.interp(jnp.log(self.kPk_array), jnp.log(self.k_mcfit), jnp.log(self.uk_y_tointp[:,jz, jM])))
    #     return uky_array_kPk

    # @partial(jit, static_argnums=(0,))
    # def get_ukne_interp_Pk(self, jz, jM):
    #     ukne_array_kPk = jnp.exp(jnp.interp(jnp.log(self.kPk_array), jnp.log(self.k_mcfit), jnp.log(self.uk_ne_tointp[:,jz, jM])))
    #     return ukne_array_kPk


    @partial(jit, static_argnums=(0,))
    def get_b_2h(self, jk, jz, probe):
        '''Function getting the 2halo effective bias of the matter fields'''
        if probe == 'dmb':
            rhom_z = self.get_rho_m(0.0)
            ukz_intc = self.Mtot_mat[jz, :] *  self.uk_dmb[jk,jz,:] * ((1/rhom_z))
        elif probe == 'nfw':
            rhom_z = self.get_rho_m(0.0)
            ukz_intc = self.Mtot_mat[jz, :] *  self.uk_nfw[jk,jz,:] * ((1/rhom_z))
        elif probe == 'Pe':
            ukz_intc = self.uk_y[jk,jz,:]
        elif probe == 'ne':
            rhom_z = self.get_rho_m(0.0)
            ukz_intc = self.Mtot_mat[jz, :] *  self.uk_ne[jk,jz,:] * ((1/rhom_z))
        elif probe == 'gal':
            ukz_intc = self.ukg_cross[jk,jz,:]
        else:
            raise ValueError('Probe not recognized')
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        fx = ukz_intc * dndlnM_z * self.bias_Mz_mat[jz,:]
        b_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return b_2h

    @partial(jit, static_argnums=(0,))
    def get_bm_largescales_2h(self, jz):
        '''Get the large scale limit of the above 2halo integral'''
        ukz_intc = self.Mtot_mat[jz, :]
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        rhom_z = self.get_rho_m(0.0) #want comoving density
        fx = ukz_intc * dndlnM_z * self.bias_Mz_mat[jz,:] * ((1/rhom_z))
        bmm_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return bmm_2h

    # @partial(jit, static_argnums=(0,))
    # def get_bm_dmb_2h(self, jk, jz):
    #     '''Function getting the 2halo effective bias of the matter fields'''
    #     ukz_intc = self.Mtot_mat[jz, :] *  self.uk_dmb[jk,jz,:]
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]     
    #     # rhom_z = self.get_rho_m(self.z_array[jz])
    #     rhom_z = self.get_rho_m(0.0)
    #     fx = ukz_intc * dndlnM_z * self.bias_Mz_mat[jz,:] * ((1/rhom_z))
    #     bmm_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return bmm_2h


    # @partial(jit, static_argnums=(0,))
    # def get_bm_nfw_2h(self, jk, jz):
    #     '''Function getting the 2halo effective bias of the matter fields'''
    #     ukz_intc = self.Mtot_mat[jz, :] *  self.uk_nfw[jk,jz,:]
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]     
    #     rhom_z = self.get_rho_m(0.0) #want comoving density
    #     fx = ukz_intc * dndlnM_z * self.bias_Mz_mat[jz,:] * ((1/rhom_z))
    #     bmm_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return bmm_2h

    # @partial(jit, static_argnums=(0,))
    # def get_bk_y_2h(self, jk, jz):
    #     '''Function getting the 2halo effective bias of the matter fields'''
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]     
    #     fx = self.uk_y[jk,jz,:] * dndlnM_z * self.bias_Mz_mat[jz,:]
    #     by_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return by_2h


    # @partial(jit, static_argnums=(0,))
    # def get_bk_g_2h(self, jk, jz):
    #     '''Function getting the 2halo effective bias of the matter fields'''
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]     
    #     fx = self.ukg_cross[jk,jz,:] * dndlnM_z * self.bias_Mz_mat[jz,:]
    #     bg_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return bg_2h

    # @partial(jit, static_argnums=(0,))
    # def get_bk_e_2h(self, jk, jz):
    #     '''Function getting the 2halo effective bias of the matter fields'''
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]     
    #     rhom_z = self.get_rho_m(0.0) #want comoving density        
    #     fx = self.Mtot_mat[jz, :] *  self.uk_ne[jk,jz,:] * dndlnM_z * self.bias_Mz_mat[jz,:] * ((1/rhom_z))
    #     be_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return be_2h

    @partial(jit, static_argnums=(0,))
    def get_P_1h(self, jk, jz, probe1, probe2):
        if 'dmb' in (probe1, probe2):
            rhom_0 = self.get_rho_m(0.0) #want comoving density
            if probe1 == 'dmb':
                ukz1 = (self.Mtot_mat[jz, :] *  self.uk_dmb[jk,jz,:]) / rhom_0
            else:
                ukz2 = (self.Mtot_mat[jz, :] *  self.uk_dmb[jk,jz,:]) / rhom_0
        elif 'nfw' in (probe1, probe2):
            rhom_0 = self.get_rho_m(0.0)
            if probe1 == 'nfw':
                ukz1 = (self.Mtot_mat[jz, :] *  self.uk_nfw[jk,jz,:]) / rhom_0
            else:
                ukz2 = (self.Mtot_mat[jz, :] *  self.uk_nfw[jk,jz,:]) / rhom_0
        elif 'Pe' in (probe1, probe2):
            if probe1 == 'Pe':
                ukz1 = self.uk_y[jk,jz,:]
            else:
                ukz2 = self.uk_y[jk,jz,:]
        elif 'ne' in (probe1, probe2):
            rhom_0 = self.get_rho_m(0.0)
            if probe1 == 'ne':
                ukz1 = (self.Mtot_mat[jz, :] *  self.uk_ne[jk,jz,:]) / rhom_0
            else:
                ukz2 = (self.Mtot_mat[jz, :] *  self.uk_ne[jk,jz,:]) / rhom_0
        elif 'gal' in (probe1, probe2):
            if probe1 == probe2 == 'gal':
                ukz_sqr = self.ukg_auto_sqr[jk,jz,:]
            elif probe1 == 'gal':
                ukz1 = self.ukg_cross[jk,jz,:]
            else:
                ukz2 = self.ukg_cross[jk,jz,:]
        else:
            raise ValueError('Probe not recognized')

        # probe_map = {
        #     'dmb': lambda: (self.Mtot_mat[jz, :] * self.uk_dmb[jk, jz, :]) / self.get_rho_m(0.0),
        #     'nfw': lambda: (self.Mtot_mat[jz, :] * self.uk_nfw[jk, jz, :]) / self.get_rho_m(0.0),
        #     'Pe': lambda: self.uk_y[jk, jz, :],
        #     'ne': lambda: (self.Mtot_mat[jz, :] * self.uk_ne[jk, jz, :]) / self.get_rho_m(0.0),
        #     'gal': lambda: self.ukg_cross[jk, jz, :] if probe1 != probe2 else self.ukg_auto_sqr[jk, jz, :]
        # }

        # if probe1 in probe_map:
        #     ukz1 = probe_map[probe1]()
        # if probe2 in probe_map:
        #     ukz2 = probe_map[probe2]()
        # if probe1 == probe2 == 'gal':
        #     ukz_sqr = self.ukg_auto_sqr[jk, jz, :]
        # else:
        #     ukz_sqr = ukz1 * ukz2

        if not (probe1 == probe2 == 'gal'):
            ukz_sqr = ukz1 * ukz2

        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        P_1h = jsi.trapezoid(ukz_sqr * dndlnM_z , x=jnp.log(self.M_array))
        return P_1h



    # @partial(jit, static_argnums=(0,))
    # def get_Pmm_dmb_1h(self, jk, jz):
    #     ukz_intc = (self.Mtot_mat[jz, :] *  self.uk_dmb[jk,jz,:])**2        
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]     
    #     rhom_z = self.get_rho_m(0.0) #want comoving density
    #     fx = ukz_intc * dndlnM_z * ((1/rhom_z)**2)
    #     Pmm_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return Pmm_1h

    # @partial(jit, static_argnums=(0,))
    # def get_Pmm_nfw_1h(self, jk, jz):
    #     ukz_intc = (self.Mtot_mat[jz, :] *  self.uk_nfw[jk,jz,:])**2
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]     
    #     rhom_z = self.get_rho_m(0.0) #want comoving density
    #     fx = ukz_intc * dndlnM_z * ((1/rhom_z)**2)
    #     Pmm_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return Pmm_1h

    # @partial(jit, static_argnums=(0,))
    # def get_Pym_dmb_1h(self, jk, jz):
    #     ukm = (self.Mtot_mat[jz, :] *  self.uk_dmb[jk,jz,:])
    #     uym = self.uk_y[jk,jz,:]
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]     
    #     rhom_z = self.get_rho_m(0.0) #want comoving density
    #     fx = ukm * uym * dndlnM_z * ((1/rhom_z))
    #     Pym_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return Pym_1h

    # @partial(jit, static_argnums=(0,))
    # def get_Pge_1h(self, jk, jz):
    #     ukg = self.ukg_cross[jk,jz,:]
    #     uke = self.Mtot_mat[jz, :] *  self.uk_ne[jk,jz,:]
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]   
    #     rhom_z = self.get_rho_m(0.0) #want comoving density  
    #     fx = ukg * uke * dndlnM_z * (1/rhom_z)
    #     Pge_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return Pge_1h

    # @partial(jit, static_argnums=(0,))
    # def get_Pgm_1h(self, jk, jz):
    #     ukg = self.ukg_cross[jk,jz,:]
    #     ukm = (self.Mtot_mat[jz, :] *  self.uk_dmb[jk,jz,:])
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]   
    #     rhom_z = self.get_rho_m(0.0) #want comoving density  
    #     fx = ukg * ukm * dndlnM_z * (1/rhom_z)
    #     Pgm_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return Pgm_1h

    # @partial(jit, static_argnums=(0,))
    # def get_Pgm_nfw_1h(self, jk, jz):
    #     ukg = self.ukg_cross[jk,jz,:]
    #     ukm = (self.Mtot_mat[jz, :] *  self.uk_nfw[jk,jz,:])
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]   
    #     rhom_z = self.get_rho_m(0.0) #want comoving density  
    #     fx = ukg * ukm * dndlnM_z * (1/rhom_z)
    #     Pgm_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return Pgm_1h    

    # @partial(jit, static_argnums=(0,))
    # def get_Pgy_1h(self, jk, jz):
    #     ukg = self.ukg_cross[jk,jz,:]
    #     uym = self.uk_y[jk,jz,:]
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]   
    #     fx = ukg * uym * dndlnM_z
    #     Pgy_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return Pgy_1h

    # @partial(jit, static_argnums=(0,))
    # def get_Pyy_1h(self, jk, jz):
    #     uym = self.uk_y[jk,jz,:]
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]   
    #     fx = uym * uym * dndlnM_z
    #     Pyy_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return Pyy_1h


    # @partial(jit, static_argnums=(0,))
    # def get_Pgg_1h(self, jk, jz):
    #     # ukg = self.ukg_auto[jk,jz,:]
    #     ukg_sqr = self.ukg_auto_sqr[jk,jz,:]
    #     dndlnM_z = self.hmf_Mz_mat[jz, :]   
    #     # fx = ukg * ukg * dndlnM_z
    #     fx = ukg_sqr * dndlnM_z
    #     Pgg_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return Pgg_1h

    @partial(jit, static_argnums=(0,))
    def get_P_lz(self, jl, jz, Pk_mat):
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.Pmm_tot_kz_mat[:,jz])))
        return Pkz_ell

    # @partial(jit, static_argnums=(0,))
    # def get_Pkmm_lz(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.Pmm_tot_kz_mat[:,jz])))
    #     return Pkz_ell
    
    # @partial(jit, static_argnums=(0,))
    # def get_Pkmm_halofit_lz(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.phfit_kz_mat[:,jz])))
    #     return Pkz_ell    

    # @partial(jit, static_argnums=(0,))
    # def get_Pkgm_lz(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.Pgm_tot_kz_mat[:,jz])))
    #     return Pkz_ell

    # @partial(jit, static_argnums=(0,))
    # def get_Pkgm_nfw_lz(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.Pgm_nfw_tot_kz_mat[:,jz])))
    #     return Pkz_ell


    # @partial(jit, static_argnums=(0,))
    # def get_Pkgg_lz(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.Pgg_tot_kz_mat[:,jz])))
    #     return Pkz_ell


    # @partial(jit, static_argnums=(0,))
    # def get_Pkym_lz(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     Bl = jnp.exp(-1. * ell * (ell + 1) * (self.sig_beam ** 2) / 2.)
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.Pym_tot_kz_mat[:,jz])))
    #     return Bl*Pkz_ell

    # @partial(jit, static_argnums=(0,))
    # def get_Pkyy_lz(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     Bl = jnp.exp(-1. * ell * (ell + 1) * (self.sig_beam ** 2) / 2.)
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.Pyy_tot_kz_mat[:,jz])))
    #     return (Bl**2)*Pkz_ell

    # @partial(jit, static_argnums=(0,))
    # def get_Pkgy_lz(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     Bl = jnp.exp(-1. * ell * (ell + 1) * (self.sig_beam ** 2) / 2.)
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.Pgy_tot_kz_mat[:,jz])))
    #     return Bl*Pkz_ell


    @partial(jit, static_argnums=(0,))
    def get_dPmm_dmb_dlnM_1h(self, jk, jz):
        ukz_intc = (self.Mtot_mat[jz, :] *  self.uk_dmb[jk,jz,:])**2
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        rhom_z = self.get_rho_m(0.0) #want comoving density
        fx = ukz_intc * dndlnM_z * ((1/rhom_z)**2)
        # Pmm_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return fx

    @partial(jit, static_argnums=(0,))
    def get_dPmm_nfw_dlnM_1h(self, jk, jz):
        ukz_intc = (self.Mtot_mat[jz, :] *  self.uk_nfw[jk,jz,:])**2
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        rhom_z = self.get_rho_m(0.0) #want comoving density
        fx = ukz_intc * dndlnM_z * ((1/rhom_z)**2)
        # Pmm_1h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return fx

    
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
    def get_ugl_cross(self, jl, jz, jM):
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        # uk_dmb_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(self.uk_dmb[:,jc, jz, jM])))
        uk_min = jnp.min(jnp.absolute(self.ukg_cross[:,jz, jM]))
        # uk_clipped = jnp.clip(self.uk_dmb[:,jc, jz, jM], uk_min + 1e-25) * self.M_array[jM]/self.rho_m_bar
        uk_clipped = jnp.clip(self.ukg_cross[:,jz, jM], uk_min + 1e-25)
        uk_dmb_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(uk_clipped)))        
        return uk_dmb_ell

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


    # # @partial(jit, static_argnums=(0,))
    # # def get_uyl_mcfit(self, jl, jc, jz, jM, xmin=0.01, xmax=3, num_points_trapz_int=128):
    # #     chiz = jnp.clip(self.chi_array[jz], 1.0)
    # #     az = 1.0 / (1.0 + self.z_array[jz])
    # #     prefac = az/(chiz**2)
    # #     rmin = xmin * self.r200c_mat[jz, jM]
    # #     rmax = xmax * self.r200c_mat[jz, jM]        

    # #     y3d_min = jnp.min(jnp.absolute(self.y3d_mat[:,jc, jz, jM]))
    # #     y3d_clipped = jnp.clip(self.y3d_mat[:,jc, jz, jM], y3d_min + 1e-30)
    # #     # logr_array_mcfit = jnp.linspace(jnp.log(jnp.min(self.r_array/chiz)), jnp.log(jnp.max(self.r_array/chiz)), num_points_trapz_int)
    # #     logr_array_mcfit = logr_array_int = jnp.linspace(jnp.log(rmin), jnp.log(rmax), num_points_trapz_int)
    # #     r_array_mcfit = jnp.exp(logr_array_mcfit)
    # #     # y3d_array = jnp.exp(jnp.interp(jnp.log(r_array_mcfit), jnp.log(self.r_array/chiz), jnp.log(y3d_clipped)))
    # #     y3d_rarray = jnp.exp(jnp.interp(logr_array_int, jnp.log(self.r_array), jnp.log(y3d_clipped)))        
    # #     k_mcfit, uy_mcfit = (xi2P(jnp.array(r_array_mcfit), nx=num_points_trapz_int,lowring=True)(jnp.array(y3d_rarray),  extrap=False))
    # #     uy_mcfit = jnp.array(uy_mcfit)
    # #     ell = self.ell_array[jl]
    # #     uyl = prefac *  (chiz**3) * jnp.exp(jnp.interp(jnp.log(ell), jnp.log(k_mcfit), jnp.log(uy_mcfit)))
    # #     Bl = jnp.exp(-1. * ell * (ell + 1) * (self.sig_beam ** 2) / 2.) 
    # #     return uyl * Bl     


    # @partial(jit, static_argnums=(0,))
    # def get_byl(self, jl, jz):
    #     # uyl_jl_jz = self.uyl_mat[jl, :, jz, :]
    #     # cmean_jz = self.conc_Mz_mat[jz, :]
    #     # logc_array = jnp.log(self.conc_array)
    #     # sig_logc = self.sig_logc_z_array[jz]
    #     # conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
    #     # cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
    #     # p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))
    #     # fx = uyl_jl_jz.T * p_logc_Mz
    #     # uyl_intc = jsi.trapezoid(fx, x=logc_array)
    #     uyl_intc = self.uyl_mat[jl, jz, :]     

    #     dndlnM_z = self.hmf_Mz_mat[jz, :]
    #     bM_z = self.bias_Mz_mat[jz, :]
    #     fx = uyl_intc * dndlnM_z * bM_z
    #     byl = jsi.trapezoid(fx, x=jnp.log(self.M_array))
    #     return byl



    # @partial(jit, static_argnums=(0,))
    # def get_ukappal_nfw_prefac(self, jl, jz, jM):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     # uk_nfw_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(self.uk_nfw[:,jc, jz, jM])))
    #     uk_min = jnp.min(jnp.absolute(self.uk_nfw[:,jz, jM]))
    #     # uk_clipped = jnp.clip(self.uk_nfw[:,jc, jz, jM], uk_min + 1e-25) * self.M_array[jM]/self.rho_m_bar
    #     uk_clipped = jnp.clip(self.uk_nfw[:,jz, jM], uk_min + 1e-25) * self.Mtot_mat[jz, jM]/self.rho_m_bar        
    #     uk_nfw_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(uk_clipped)))        
    #     return uk_nfw_ell


    # @partial(jit, static_argnums=(0,))
    # def get_bkl_dmb(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     bkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(self.bm_dmb_kz_mat[:,jz])))
    #     return bkz_ell

    # @partial(jit, static_argnums=(0,))
    # def get_bkl_nfw(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     bkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.k), jnp.log(self.bm_nfw_kz_mat[:,jz])))
    #     return bkz_ell

    # @partial(jit, static_argnums=(0,))
    # def get_Pklin_lz(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.plin_kz_mat[:,jz])))
    #     return Pkz_ell

    # @partial(jit, static_argnums=(0,))
    # def get_Pknl_lz(self, jl, jz):
    #     ell = self.ell_array[jl]
    #     chi_z = self.chi_array[jz]
    #     k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
    #     Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(self.pnl_kz_mat[:,jz])))
    #     return Pkz_ell