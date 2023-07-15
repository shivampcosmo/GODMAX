import os
from get_power_spectra_jit import get_power_BCMP
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from jax_cosmo import Cosmology
from functools import partial
import astropy.units as u
from astropy import constants as const
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.eV / u.cm**3)).value
# from transforms import Hankel
from mcfit import Hankel
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
                num_points_trapz_int=64,
                setup_power_BCMP_obj=None,
                get_power_BCMP_obj=None,
                verbose_time=False
                ):

        if verbose_time:
            t0 = time.time()

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

        if verbose_time:
            ti = time.time()
        if get_power_BCMP_obj is None:
            get_power_BCMP_obj = get_power_BCMP(sim_params_dict, halo_params_dict, analysis_dict, num_points_trapz_int=num_points_trapz_int, setup_power_BCMP_obj=setup_power_BCMP_obj, verbose_time=verbose_time)
        if verbose_time:
            print('Time for setup_power_BCMP: ', time.time() - ti)
            ti = time.time()
            
        self.angles_data_array = jnp.array(analysis_dict['angles_data_array'])
        self.nt_out = len(self.angles_data_array)
        
        self.ell_array = get_power_BCMP_obj.ell_array
        self.nbins = get_power_BCMP_obj.nbins
        if analysis_dict['do_sheary']:
            self.Cl_kappa_y_1h = get_power_BCMP_obj.Cl_kappa_y_1h_mat
            self.Cl_kappa_y_2h = get_power_BCMP_obj.Cl_kappa_y_2h_mat

            if verbose_time:
                ti = time.time()

            self.gty_1h_out_mat = vmap(self.get_Hankel_gty_1h)(jnp.arange(len(self.angles_data_array)))
            self.gty_2h_out_mat = vmap(self.get_Hankel_gty_2h)(jnp.arange(len(self.angles_data_array)))            
            self.gty_out_mat = self.gty_1h_out_mat + self.gty_2h_out_mat


        if verbose_time:
            print('Time for gty Hankel transform: ', time.time() - ti)
            ti = time.time()

        if analysis_dict['do_shear2pt']:
            self.Cl_kappa_kappa_1h = get_power_BCMP_obj.Cl_kappa_kappa_1h_mat
            self.Cl_kappa_kappa_2h = get_power_BCMP_obj.Cl_kappa_kappa_2h_mat

            self.xip_1h_out_mat = vmap(self.get_Hankel_xip_1h)(jnp.arange(len(self.angles_data_array)))
            self.xip_2h_out_mat = vmap(self.get_Hankel_xip_2h)(jnp.arange(len(self.angles_data_array)))            
            self.xip_out_mat = self.xip_1h_out_mat + self.xip_2h_out_mat


            if verbose_time:
                print('Time for xip Hankel transform: ', time.time() - ti)
                ti = time.time()


            self.xim_1h_out_mat = vmap(self.get_Hankel_xim_1h)(jnp.arange(len(self.angles_data_array)))
            self.xim_2h_out_mat = vmap(self.get_Hankel_xim_2h)(jnp.arange(len(self.angles_data_array)))            
            self.xim_out_mat = self.xim_1h_out_mat + self.xim_2h_out_mat


            if verbose_time:
                print('Time for xim Hankel transform: ', time.time() - ti)

    @partial(jit, static_argnums=(0,))
    def get_Hankel_gty_1h(self, jt):
        thetav = self.angles_data_array[jt] * (1/60.) * (jnp.pi/180.)
        ell_theta = self.ell_array * thetav
        J2ltheta = y = vmap(jv_jax, in_axes=(None, 0))(2, ell_theta)
        prefac = (self.ell_array**2) * J2ltheta * (1/(2*jnp.pi))
        prefac_tiled = jnp.tile(prefac, (self.nbins, 1))
        integrand = prefac_tiled * self.Cl_kappa_y_1h
        gty_1h = jnp.trapz(integrand, jnp.log(self.ell_array), axis=-1)
        return gty_1h

    @partial(jit, static_argnums=(0,))
    def get_Hankel_gty_2h(self, jt):
        thetav = self.angles_data_array[jt] * (1/60.) * (jnp.pi/180.)
        ell_theta = self.ell_array * thetav
        J2ltheta = y = vmap(jv_jax, in_axes=(None, 0))(2, ell_theta)
        prefac = (self.ell_array**2) * J2ltheta * (1/(2*jnp.pi))
        prefac_tiled = jnp.tile(prefac, (self.nbins, 1))
        integrand = prefac_tiled * self.Cl_kappa_y_2h
        gty_1h = jnp.trapz(integrand, jnp.log(self.ell_array), axis=-1)
        return gty_1h    

    @partial(jit, static_argnums=(0,))
    def get_Hankel_xip_1h(self, jt):
        thetav = self.angles_data_array[jt] * (1/60.) * (jnp.pi/180.)
        ell_theta = self.ell_array * thetav
        J2ltheta = y = vmap(jv_jax, in_axes=(None, 0))(0, ell_theta)
        prefac = (self.ell_array**2) * J2ltheta * (1/(2*jnp.pi))
        prefac_tiled = jnp.tile(prefac, (self.nbins, 1))
        integrand = prefac_tiled * self.Cl_kappa_kappa_1h
        gty_1h = jnp.trapz(integrand, jnp.log(self.ell_array), axis=-1)
        return gty_1h

    @partial(jit, static_argnums=(0,))
    def get_Hankel_xip_2h(self, jt):
        thetav = self.angles_data_array[jt] * (1/60.) * (jnp.pi/180.)
        ell_theta = self.ell_array * thetav
        J2ltheta = y = vmap(jv_jax, in_axes=(None, 0))(0, ell_theta)
        prefac = (self.ell_array**2) * J2ltheta * (1/(2*jnp.pi))
        prefac_tiled = jnp.tile(prefac, (self.nbins, 1))
        integrand = prefac_tiled * self.Cl_kappa_kappa_2h
        gty_1h = jnp.trapz(integrand, jnp.log(self.ell_array), axis=-1)
        return gty_1h    

    @partial(jit, static_argnums=(0,))
    def get_Hankel_xim_1h(self, jt):
        thetav = self.angles_data_array[jt] * (1/60.) * (jnp.pi/180.)
        ell_theta = self.ell_array * thetav
        J2ltheta = y = vmap(jv_jax, in_axes=(None, 0))(4, ell_theta)
        prefac = (self.ell_array**2) * J2ltheta * (1/(2*jnp.pi))
        prefac_tiled = jnp.tile(prefac, (self.nbins, 1))
        integrand = prefac_tiled * self.Cl_kappa_kappa_1h
        gty_1h = jnp.trapz(integrand, jnp.log(self.ell_array), axis=-1)
        return gty_1h

    @partial(jit, static_argnums=(0,))
    def get_Hankel_xim_2h(self, jt):
        thetav = self.angles_data_array[jt] * (1/60.) * (jnp.pi/180.)
        ell_theta = self.ell_array * thetav
        J2ltheta = y = vmap(jv_jax, in_axes=(None, 0))(4, ell_theta)
        prefac = (self.ell_array**2) * J2ltheta * (1/(2*jnp.pi))
        prefac_tiled = jnp.tile(prefac, (self.nbins, 1))
        integrand = prefac_tiled * self.Cl_kappa_kappa_2h
        gty_1h = jnp.trapz(integrand, jnp.log(self.ell_array), axis=-1)
        return gty_1h    

    # @partial(jit, static_argnums=(0,))
    # def interp_gty_theta(self, jb, jt):
    #     gty_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array[jt]), jnp.log(self.theta_out_arcmin), jnp.log(self.gty_tot_mat[jb,:])))
    #     return gty_out

    # @partial(jit, static_argnums=(0,))
    # def interp_xip_theta(self, jb1, jb2):
    #     xip_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xip_tot_mat[jb1,jb2,:])))
    #     return xip_out

    # @partial(jit, static_argnums=(0,))
    # def interp_xim_theta(self, jb1, jb2):
    #     xim_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xim_tot_mat[jb1,jb2,:])))
    #     return xim_out
