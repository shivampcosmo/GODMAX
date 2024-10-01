import os
from get_power_spectra import get_power_BCMP
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
        # self.Cls_data_array = analysis_dict['Cls_data_array']
        # self.cov_data_mat = analysis_dict['cov_data_mat']

        self.ell_array = get_power_BCMP_obj.ell_array
        self.nbins = get_power_BCMP_obj.nbins
        # print(analysis_dict['corr_types'])
        # print(['shear_y'] in analysis_dict['corr_types'])
        if analysis_dict['do_sheary']:
            self.Cl_kappa_y_1h = get_power_BCMP_obj.Cl_kappa_y_1h_mat
            self.Cl_kappa_y_2h = get_power_BCMP_obj.Cl_kappa_y_2h_mat

            if verbose_time:
                ti = time.time()
            theta_out, xi_out = (Hankel(self.ell_array, nu=2, q=1.0)(self.Cl_kappa_y_1h, axis=1, extrap=True))
            self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
            self.gty_1h_mat = xi_out * (1 / (2 * jnp.pi))
            _, xi_out = (Hankel(self.ell_array, nu=2, q=1.0)(self.Cl_kappa_y_2h, axis=1, extrap=True))
            self.gty_2h_mat = xi_out * (1 / (2 * jnp.pi))

            self.gty_tot_mat = jnp.array(self.gty_1h_mat + self.gty_2h_mat)
            # print(self.gty_1h_mat)
            # vmap_func1 = vmap(self.interp_gty_theta, (0, None))
            vmap_func1 = vmap(self.interp_gty_theta, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.gty_out_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nt_out)).T

            # self.gty_out_mat = vmap(self.interp_gty_theta)(jnp.arange(self.nbins))
            # import pdb; pdb.set_trace() 


        if verbose_time:
            print('Time for gty Hankel transform: ', time.time() - ti)
            ti = time.time()

        if analysis_dict['do_shear2pt']:
            self.Cl_kappa_kappa_1h = get_power_BCMP_obj.Cl_kappa_kappa_1h_mat
            self.Cl_kappa_kappa_2h = get_power_BCMP_obj.Cl_kappa_kappa_2h_mat

            theta_out, xi_out = (Hankel(self.ell_array, nu=0, q=1.0)(self.Cl_kappa_kappa_1h, axis=2, extrap=True))
            self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
            self.xip_1h_mat = xi_out * (1 / (2 * jnp.pi))
            _, xi_out = (Hankel(self.ell_array, nu=0, q=1.0)(self.Cl_kappa_kappa_2h, axis=2, extrap=True))
            self.xip_2h_mat = xi_out * (1 / (2 * jnp.pi))
            self.xip_tot_mat = jnp.array(self.xip_1h_mat + self.xip_2h_mat)
            vmap_func1 = vmap(self.interp_xip_theta, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.xip_out_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nbins)).T

            if verbose_time:
                print('Time for xip Hankel transform: ', time.time() - ti)
                ti = time.time()


            theta_out, xi_out = (Hankel(self.ell_array, nu=4, q=1.0)(self.Cl_kappa_kappa_1h, axis=2, extrap=True))
            self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
            self.xim_1h_mat = xi_out * (1 / (2 * jnp.pi))
            _, xi_out = (Hankel(self.ell_array, nu=4, q=1.0)(self.Cl_kappa_kappa_2h, axis=2, extrap=True))
            self.xim_2h_mat = xi_out * (1 / (2 * jnp.pi))
            self.xim_tot_mat = jnp.array(self.xim_1h_mat + self.xim_2h_mat)
            vmap_func1 = vmap(self.interp_xim_theta, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.xim_out_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nbins)).T

            if verbose_time:
                print('Time for xim Hankel transform: ', time.time() - ti)

    @partial(jit, static_argnums=(0,))
    def interp_gty_theta(self, jb, jt):
        gty_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array[jt]), jnp.log(self.theta_out_arcmin), jnp.log(self.gty_tot_mat[jb,:])))
        return gty_out

    @partial(jit, static_argnums=(0,))
    def interp_xip_theta(self, jb1, jb2):
        xip_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xip_tot_mat[jb1,jb2,:])))
        return xip_out

    @partial(jit, static_argnums=(0,))
    def interp_xim_theta(self, jb1, jb2):
        xim_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xim_tot_mat[jb1,jb2,:])))
        return xim_out
