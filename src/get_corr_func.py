import os
from get_power_spectra import get_power_BCMP
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import jax.scipy.integrate as jsi
from jax_cosmo import Cosmology
from functools import partial
import astropy.units as u
from astropy import constants as const
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.eV / u.cm**3)).value
# from transforms import Hankel
# from mcfit import Hankel
from mcfitjax.transforms import Hankel
import time

import scipy.special
from jax import custom_jvp, pure_callback, vmap
import jax
# see https://github.com/google/jax/issues/11002

def generate_bessel(function):
    """function is Jv, Yv, Hv_1,Hv_2"""

    @custom_jvp
    def cv(v, x):
        return pure_callback(
            lambda vx: function(*vx),
            x,
            (v, x),
            vectorized=True,
        )

    @cv.defjvp
    def cv_jvp(primals, tangents):
        v, x = primals
        dv, dx = tangents
        primal_out = cv(v, x)

        # https://dlmf.nist.gov/10.6 formula 10.6.1
        tangents_out = jax.lax.cond(
            v == 0,
            lambda: -cv(v + 1, x),
            lambda: 0.5 * (cv(v - 1, x) - cv(v + 1, x)),
        )

        return primal_out, tangents_out * dx

    return cv
jv_jax = generate_bessel(scipy.special.jv)


class get_corrfunc_BCMP:
    def __init__(self,
                sim_params_dict,
                halo_params_dict,
                analysis_dict,
                other_params_dict=None,
                num_points_trapz_int=64,
                setup_power_BCMP_obj=None,
                get_power_BCMP_obj=None,
                verbose_time=False
                ):

        if verbose_time:
            t0 = time.time()

        self.cosmo_params = sim_params_dict['cosmo']
        self.conc_dep_model = analysis_dict.get('conc_dep_model',False)
        if verbose_time:
            ti = time.time()
        # if get_power_BCMP_obj is None:
            # if self.conc_dep_model:
            #     get_power_BCMP_obj = get_power_BCMP(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, num_points_trapz_int=num_points_trapz_int, setup_power_BCMP_obj=setup_power_BCMP_obj, verbose_time=verbose_time)
            # else:
        get_power_BCMP_obj = get_power_BCMP(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, num_points_trapz_int=num_points_trapz_int, setup_power_BCMP_obj=setup_power_BCMP_obj, verbose_time=verbose_time)
        if verbose_time:
            print('Time for setup_power_BCMP: ', time.time() - ti)
            ti = time.time()
        want_like_diff = analysis_dict['want_like_diff']
        self.angles_data_array = jnp.array(analysis_dict['angles_data_array'])
        self.nt_out = len(self.angles_data_array)
        self.calc_nfw_only = analysis_dict['calc_nfw_only']
        self.ell_array = get_power_BCMP_obj.ell_array
        self.log_ell_array = jnp.log(self.ell_array)
        self.nbins = get_power_BCMP_obj.nbins

        self.get_sep_1h2h = analysis_dict.get('get_sep_1h2h',False)
        
        if analysis_dict['do_sheary']:
            
            if verbose_time:
                ti = time.time()

            if self.get_sep_1h2h:
                self.Cl_kappa_y_1h = get_power_BCMP_obj.Cl_kappa_y_1h_mat
                self.Cl_kappa_y_2h = get_power_BCMP_obj.Cl_kappa_y_2h_mat                
                theta_out, xi_out = (Hankel(self.ell_array, nu=2, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_y_1h, axis=1, extrap=False))
                self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
                self.gty_1h_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))
                _, xi_out = (Hankel(self.ell_array, nu=2, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_y_2h, axis=1, extrap=False))
                self.gty_2h_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))
    
                # self.alpha_gty = analysis_dict.get('alpha_gty',1.0)
                # self.Cl_kappa_y_tot = (jnp.clip(self.Cl_kappa_y_1h, 0)**self.alpha_gty + jnp.clip(self.Cl_kappa_y_2h, 0)**self.alpha_gty)**(1/self.alpha_gty)
                self.Cl_kappa_y_tot = self.Cl_kappa_y_1h + self.Cl_kappa_y_2h
                _, xi_out = (Hankel(self.ell_array, nu=2, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_y_tot, axis=1, extrap=False))
                # self.gty_tot_mat = jnp.array((self.gty_1h_mat**self.alpha_gty + self.gty_2h_mat**self.alpha_gty)**(1/self.alpha_gty))
                self.gty_tot_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))

                vmap_func1 = vmap(self.interp_gty1h_theta, (0, None))
                vmap_func2 = vmap(vmap_func1, (None, 0))
                self.gty_1h_out_mat = vmap_func2(jnp.arange(self.nt_out), jnp.arange(self.nbins)).T

                vmap_func1 = vmap(self.interp_gty2h_theta, (0, None))
                vmap_func2 = vmap(vmap_func1, (None, 0))
                self.gty_2h_out_mat = vmap_func2(jnp.arange(self.nt_out), jnp.arange(self.nbins)).T

            else:
                self.Cl_kappa_y_tot = get_power_BCMP_obj.Cl_kappa_y_1h_mat + get_power_BCMP_obj.Cl_kappa_y_2h_mat
                theta_out, xi_out = (Hankel(self.ell_array, nu=2, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_y_tot, axis=1, extrap=False))
                self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
                self.gty_tot_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))



            vmap_func1 = vmap(self.interp_gty_theta, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.gty_out_mat = vmap_func2(jnp.arange(self.nt_out), jnp.arange(self.nbins)).T

        if verbose_time:
            print('Time for gty Hankel transform: ', time.time() - ti)
            ti = time.time()

        if analysis_dict['do_shear2pt']:
            # if self.get_sep_1h2h:
                # self.Cl_kappa_kappa_1h = get_power_BCMP_obj.Cl_kappa_kappa_1h_mat
                # self.Cl_kappa_kappa_2h = get_power_BCMP_obj.Cl_kappa_kappa_2h_mat
            self.Cl_kappa_kappa_tot = get_power_BCMP_obj.Cl_kappa_kappa_tot_mat


            # theta_out, xi_out = (Hankel(self.ell_array, nu=0, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_kappa_1h, axis=2, extrap=False))
            # self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
            # self.xip_1h_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))
            # _, xi_out = (Hankel(self.ell_array, nu=0, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_kappa_2h, axis=2, extrap=False))
            # self.xip_2h_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))

            # self.alpha_xip = analysis_dict.get('alpha_xip',1.0)         
            # self.Cl_kappa_kappa_tot = (jnp.clip(self.Cl_kappa_kappa_1h, 0)**self.alpha_xip + jnp.clip(self.Cl_kappa_kappa_2h, 0)**self.alpha_xip)**(1/self.alpha_xip)
            # self.Cl_kappa_kappa_tot = self.Cl_kappa_kappa_1h + self.Cl_kappa_kappa_2h
            theta_out, xi_out = (Hankel(self.ell_array, nu=0, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_kappa_tot, axis=2, extrap=False))
            self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
            self.xip_tot_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))   
                # self.xip_tot_mat = jnp.array((self.xip_1h_mat**self.alpha_xip + self.xip_2h_mat**self.alpha_xip)**(1/self.alpha_xip))

                # vmap_func1 = vmap(self.interp_xip1h_theta, (0, None))
                # vmap_func2 = vmap(vmap_func1, (None, 0))
                # self.xip_1h_out_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nbins)).T

                # vmap_func1 = vmap(self.interp_xip2h_theta, (0, None))
                # vmap_func2 = vmap(vmap_func1, (None, 0))
                # self.xip_2h_out_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nbins)).T
            
            # else:
            #     self.Cl_kappa_kappa_tot = get_power_BCMP_obj.Cl_kappa_kappa_tot_mat
            #     theta_out, xi_out = (Hankel(self.ell_array, nu=0, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_kappa_tot, axis=2, extrap=False))
            #     self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
            #     self.xip_tot_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))

            vmap_func1 = vmap(self.interp_xip_theta, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.xip_out_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nbins)).T


            if verbose_time:
                print('Time for xip Hankel transform: ', time.time() - ti)
                ti = time.time()

            # if self.get_sep_1h2h:
                # theta_out, xi_out = (Hankel(self.ell_array, nu=4, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_kappa_1h, axis=2, extrap=False))
                # self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
                # self.xim_1h_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))
                # _, xi_out = (Hankel(self.ell_array, nu=4, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_kappa_2h, axis=2, extrap=False))
                # self.xim_2h_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))

                # self.alpha_xim = analysis_dict.get('alpha_xim',1.0)                        
                # self.Cl_kappa_kappa_tot = (jnp.clip(self.Cl_kappa_kappa_1h, 0)**self.alpha_xim + jnp.clip(self.Cl_kappa_kappa_2h, 0)**self.alpha_xim)**(1/self.alpha_xim)
                # self.Cl_kappa_kappa_tot = self.Cl_kappa_kappa_1h + self.Cl_kappa_kappa_2h
            _, xi_out = (Hankel(self.ell_array, nu=4, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_kappa_tot, axis=2, extrap=False))
            self.xim_tot_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))
                # self.xim_tot_mat = jnp.array((self.xim_1h_mat**self.alpha_xim + self.xim_2h_mat**self.alpha_xim)**(1/self.alpha_xim))

                # vmap_func1 = vmap(self.interp_xim1h_theta, (0, None))
                # vmap_func2 = vmap(vmap_func1, (None, 0))
                # self.xim_1h_out_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nbins)).T                

                # vmap_func1 = vmap(self.interp_xim2h_theta, (0, None))
                # vmap_func2 = vmap(vmap_func1, (None, 0))
                # self.xim_2h_out_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nbins)).T                

            # else:
            #     self.Cl_kappa_kappa_tot = get_power_BCMP_obj.Cl_kappa_kappa_tot_mat
            #     theta_out, xi_out = (Hankel(self.ell_array, nu=4, q=1.0, nx=halo_params_dict['nell'], lowring=True)(self.Cl_kappa_kappa_tot, axis=2, extrap=False))
            #     self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
            #     self.xim_tot_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))

            vmap_func1 = vmap(self.interp_xim_theta, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.xim_out_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nbins)).T                

            if verbose_time:
                print('Time for xim Hankel transform: ', time.time() - ti)

    @partial(jit, static_argnums=(0,))
    def get_J0ltheta(self, jt):
        thetav = self.angles_data_array[jt] * (1/60.) * (jnp.pi/180.)
        ell_theta = self.ell_array_transf * thetav
        value = vmap(jv_jax, in_axes=(None, 0))(0, ell_theta)
        return value


    @partial(jit, static_argnums=(0,))
    def get_J2ltheta(self, jt):
        thetav = self.angles_data_array[jt] * (1/60.) * (jnp.pi/180.)
        ell_theta = self.ell_array_transf * thetav
        value = vmap(jv_jax, in_axes=(None, 0))(2, ell_theta)
        return value

    @partial(jit, static_argnums=(0,))
    def get_J4ltheta(self, jt):
        thetav = self.angles_data_array[jt] * (1/60.) * (jnp.pi/180.)
        ell_theta = self.ell_array_transf * thetav
        value = vmap(jv_jax, in_axes=(None, 0))(4, ell_theta)
        return value


    @partial(jit, static_argnums=(0,))
    def get_Hankel_gty_1h(self, jt):
        prefac = (self.ell_array_transf**2) * self.J2ltheta_mat[jt,:] * (1/(2*jnp.pi))
        prefac_tiled = jnp.tile(prefac, (self.nbins, 1))
        integrand = prefac_tiled * self.Cl_kappa_y_1h_ell_transf
        value = jsi.trapezoid(integrand, jnp.log(self.ell_array_transf), axis=-1)
        return value

    @partial(jit, static_argnums=(0,))
    def get_Hankel_gty_2h(self, jt):
        prefac = (self.ell_array_transf**2) * self.J2ltheta_mat[jt,:] * (1/(2*jnp.pi))
        prefac_tiled = jnp.tile(prefac, (self.nbins, 1))
        integrand = prefac_tiled * self.Cl_kappa_y_2h_ell_transf
        value = jsi.trapezoid(integrand, jnp.log(self.ell_array_transf), axis=-1)
        return value


    # @partial(jit, static_argnums=(0,))
    # def get_Hankel_xip_1h(self, jt):
    #     prefac = (self.ell_array_transf**2) * self.J0ltheta_mat[jt,:] * (1/(2*jnp.pi))
    #     prefac_tiled = jnp.tile(prefac, (self.nbins,self.nbins, 1))
    #     integrand = prefac_tiled * self.Cl_kappa_kappa_1h_ell_transf
    #     value = jsi.trapezoid(integrand, jnp.log(self.ell_array_transf), axis=-1)
    #     return value

    # @partial(jit, static_argnums=(0,))
    # def get_Hankel_xip_2h(self, jt):
    #     prefac = (self.ell_array_transf**2) * self.J0ltheta_mat[jt,:] * (1/(2*jnp.pi))
    #     prefac_tiled = jnp.tile(prefac, (self.nbins,self.nbins, 1))
    #     integrand = prefac_tiled * self.Cl_kappa_kappa_2h_ell_transf
    #     value = jsi.trapezoid(integrand, jnp.log(self.ell_array_transf), axis=-1)
    #     return value    

    # @partial(jit, static_argnums=(0,))
    # def get_Hankel_xim_1h(self, jt):
    #     prefac = (self.ell_array_transf**2) * self.J4ltheta_mat[jt,:] * (1/(2*jnp.pi))
    #     prefac_tiled = jnp.tile(prefac, (self.nbins,self.nbins, 1))
    #     integrand = prefac_tiled * self.Cl_kappa_kappa_1h_ell_transf
    #     value = jsi.trapezoid(integrand, jnp.log(self.ell_array_transf), axis=-1)
    #     return value

    # @partial(jit, static_argnums=(0,))
    # def get_Hankel_xim_2h(self, jt):
    #     prefac = (self.ell_array_transf**2) * self.J4ltheta_mat[jt,:] * (1/(2*jnp.pi))
    #     prefac_tiled = jnp.tile(prefac, (self.nbins,self.nbins, 1))
    #     integrand = prefac_tiled * self.Cl_kappa_kappa_2h_ell_transf
    #     value = jsi.trapezoid(integrand, jnp.log(self.ell_array_transf), axis=-1)
    #     return value    

    @partial(jit, static_argnums=(0,))
    def interp_gty1h_theta(self, jt, jb):
        gty_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array[jt]), jnp.log(self.theta_out_arcmin), jnp.log(self.gty_1h_mat[jb,:])))
        return gty_out

    @partial(jit, static_argnums=(0,))
    def interp_gty2h_theta(self, jt, jb):
        gty_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array[jt]), jnp.log(self.theta_out_arcmin), jnp.log(self.gty_2h_mat[jb,:])))
        return gty_out

    @partial(jit, static_argnums=(0,))
    def interp_gty_theta(self, jt, jb):
        gty_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array[jt]), jnp.log(self.theta_out_arcmin), jnp.log(self.gty_tot_mat[jb,:])))
        return gty_out

    # @partial(jit, static_argnums=(0,))
    # def interp_xip1h_theta(self, jb1, jb2):
    #     xip_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xip_1h_mat[jb1,jb2,:])))
    #     return xip_out

    # @partial(jit, static_argnums=(0,))
    # def interp_xip2h_theta(self, jb1, jb2):
    #     xip_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xip_2h_mat[jb1,jb2,:])))
    #     return xip_out

    @partial(jit, static_argnums=(0,))
    def interp_xip_theta(self, jb1, jb2):
        xip_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xip_tot_mat[jb1,jb2,:])))
        return xip_out

    # @partial(jit, static_argnums=(0,))
    # def interp_xim1h_theta(self, jb1, jb2):
    #     xim_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xim_1h_mat[jb1,jb2,:])))
    #     return xim_out

    # @partial(jit, static_argnums=(0,))
    # def interp_xim2h_theta(self, jb1, jb2):
    #     xim_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xim_2h_mat[jb1,jb2,:])))
    #     return xim_out

    @partial(jit, static_argnums=(0,))
    def interp_xim_theta(self, jb1, jb2):
        xim_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xim_tot_mat[jb1,jb2,:])))
        return xim_out
