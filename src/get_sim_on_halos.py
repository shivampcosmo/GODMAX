import os, sys
from get_BCMP_profile import BCM_18_wP
import jax.numpy as jnp
from astropy.io import fits
import healpy as hp
import jax.scipy.integrate as jsi
import pdb
import pickle
import jax
from functools import partial
from astropy import constants as const
from scipy import interpolate as interp
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.keV / u.cm**3)).value
G_new_rhom = const.G.to(u.Mpc**3 / ((u.s**2) * u.M_sun))
import constants

mp = (1.6726219e-27*u.kg).to(u.Msun).value
mue = 1.14
Mpc_to_cm = 3.086e24
import jax_cosmo.background as bkgrd
from jax import lax
import sys
import time
from jax import grad, jit, vmap
import numpy as np
import math
from jax_cosmo import Cosmology
sys.path.append(('/mnt/home/spandey/ceph/interpax'))
import interpax
from tqdm import tqdm
import time


class get_mock_map:
    def __init__(
                self,
                sim_params_dict,
                halo_params_dict,
                mock_params_dict,                
                # analysis_dict,
                get_ymap=False,
                get_kSZmap=False,
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

        if BCMP_obj is None:
            BCMP_obj = BCM_18_wP(sim_params_dict, halo_params_dict, num_points_trapz_int=num_points_trapz_int, verbose_time=verbose_time)


        self.H0 = 100. * (u.km / (u.s * u.Mpc))
        self.rho_m_bar = self.cosmo_params['Om0'] * ((3 * (self.H0**2) / (8 * jnp.pi * G_new_rhom)).to(u.M_sun / (u.Mpc**3))).value

        # self.cmean_all_Mz = mock_params_dict['cmean_jM_jz']
        # self.c_array = BCMP_obj.conc_array
        self.M_array = BCMP_obj.M_array
        self.z_array = BCMP_obj.z_array
        self.scale_factor_array = 1. / (1 + self.z_array)
        H_array = self.H0 * bkgrd.H(self.cosmo_jax, self.scale_factor_array)        
        self.H_array_interp = interpax.Interpolator1D(self.z_array, H_array, extrap=True)
        self.r_array = BCMP_obj.r_array

        self.rp_array = BCMP_obj.r_array[2:-2]

        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        coeff = sigmat / (m_e * (c ** 2))
        oneMpc = (((10 ** 6)) * (u.pc).to(u.m)) * (u.m)


        if get_ymap:
            self.Pe_mat_physical = BCMP_obj.Pe_mat_physical
            self.const_coeff = (((coeff * oneMpc).to(((u.cm ** 3) / u.keV))).value)/(self.cosmo_params['H0']/100.)

            vmap_func1 = vmap(self.get_y2D_phyical_proj, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.y2D_mat_physical = vmap_func3(jnp.arange(len(self.rp_array)), jnp.arange(len(self.z_array)), jnp.arange(len(self.M_array))).T        

            self.log_y2D_interp = interpax.Interpolator3D(jnp.log(self.rp_array), self.z_array, jnp.log(self.M_array), jnp.log(self.y2D_mat_physical), extrap=True)

            self.nside_map = mock_params_dict['nside']
            self.y_sim = jnp.zeros(self.nside_map**2 * 12)

            self.nearby_pix_all = mock_params_dict['nearby_pix_all']
            self.pix_prop_all = mock_params_dict['pix_prop_all']

            self.yjpix_all = vmap(self.get_Pe_healpix)(jnp.arange(len(self.pix_prop_all)))

            pix_unique = np.unique(self.nearby_pix_all)
            sort_index_nearby_pix_all = np.argsort(self.nearby_pix_all)
            sorted_nearby_pix_all = self.nearby_pix_all[sort_index_nearby_pix_all]
            ypix_all_sorted = self.yjpix_all[sort_index_nearby_pix_all]
            change_points = np.diff(sorted_nearby_pix_all, prepend=sorted_nearby_pix_all[0]-1, append=sorted_nearby_pix_all[-1]+1) != 0 
            boundaries = np.where(change_points)[0]
            ypix_sum = np.add.reduceat(ypix_all_sorted, boundaries[:-1])

            ymap_final = np.zeros(12 * mock_params_dict['nside']**2)
            ymap_final[pix_unique] = ypix_sum
            self.ymap_final = ymap_final

        if get_kSZmap:
            self.ne_mat_physical = BCMP_obj.ne_mat_physical
            coeff_kSZ = sigmat/(c)
            self.const_coeff_kSZ =  (((coeff_kSZ * oneMpc).to(((u.cm ** 3) / (u.km/u.s)))).value)

            self.tauz_array = vmap(self.get_tau_z)(jnp.arange(len(self.z_array)))
            self.tau_interp = interpax.Interpolator1D(self.z_array, self.tauz_array, extrap=True)


            vmap_func1 = vmap(self.get_ne2D_phyical_proj, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.ne2D_mat_physical = vmap_func3(jnp.arange(len(self.rp_array)), jnp.arange(len(self.z_array)), jnp.arange(len(self.M_array))).T        

            self.log_ne2D_interp = interpax.Interpolator3D(jnp.log(self.rp_array), self.z_array, jnp.log(self.M_array), jnp.log(self.ne2D_mat_physical), extrap=True)

            self.nside_map = mock_params_dict['nside']
            self.ksz_sim = jnp.zeros(self.nside_map**2 * 12)

            self.nearby_pix_all = mock_params_dict['nearby_pix_all']
            self.pix_prop_all = mock_params_dict['pix_prop_all']

            self.kszjpix_all = vmap(self.get_kSZ_healpix)(jnp.arange(len(self.pix_prop_all)))

            pix_unique = np.unique(self.nearby_pix_all)
            sort_index_nearby_pix_all = np.argsort(self.nearby_pix_all)
            sorted_nearby_pix_all = self.nearby_pix_all[sort_index_nearby_pix_all]
            kszpix_all_sorted = self.kszjpix_all[sort_index_nearby_pix_all]
            change_points = np.diff(sorted_nearby_pix_all, prepend=sorted_nearby_pix_all[0]-1, append=sorted_nearby_pix_all[-1]+1) != 0 
            boundaries = np.where(change_points)[0]
            kszpix_sum = np.add.reduceat(kszpix_all_sorted, boundaries[:-1])

            kszmap_final = np.zeros(12 * mock_params_dict['nside']**2)
            kszmap_final[pix_unique] = kszpix_sum
            self.kszmap_final = kszmap_final

    @partial(jit, static_argnums=(0,))        
    def get_tau_z(self, jz):
        z = self.z_array[jz]
        z_array = jnp.linspace(0.000001, z, 100)
        # scale_factor_array = 1. / (1 + z_array)
        # H_array = self.H0 * bkgrd.H(self.cosmo_jax, scale_factor_array)
        H_array = self.H_array_interp(z_array)
        to_int = (1 + z_array)**2 / H_array
        tau_z = jsi.trapezoid(to_int, z_array)
        sigma_T = const.sigma_T.to(u.Mpc**2).value
        c = const.c.to(u.km/u.s).value
        ne_bar = self.cosmo_params['Ob0'] * RHO_CRIT_0_MPC3 / (mue * mp)
        val = sigma_T * ne_bar * c * tau_z
        return val

    @partial(jit, static_argnums=(0,))        
    def get_y2D_phyical_proj(self, jrp, jz, jM, num_trapz_points=32):
        rp = self.rp_array[jrp]
        zval = self.z_array[jz]
        r_array_here = jnp.exp(jnp.linspace(jnp.log(rp*1.01), jnp.log(jnp.max(self.r_array)), num_trapz_points))
        Pe_rarray_here = jnp.exp(jnp.interp(jnp.log(r_array_here), jnp.log(self.r_array/(1 + zval)), jnp.log(self.Pe_mat_physical[:,jz, jM])))
        num = r_array_here * Pe_rarray_here
        denom = jnp.sqrt(r_array_here ** 2 - rp ** 2)
        toint = num / denom
        val = 2. * jsi.trapezoid(toint * r_array_here, jnp.log(r_array_here))
        return self.const_coeff * val

    @partial(jit, static_argnums=(0,))        
    def get_ne2D_phyical_proj(self, jrp, jz, jM, num_trapz_points=32):
        rp = self.rp_array[jrp]
        zval = self.z_array[jz]
        r_array_here = jnp.exp(jnp.linspace(jnp.log(rp*1.01), jnp.log(jnp.max(self.r_array)), num_trapz_points))
        ne_rarray_here = jnp.exp(jnp.interp(jnp.log(r_array_here), jnp.log(self.r_array/(1 + zval)), jnp.log(self.ne_mat_physical[:,jz, jM])))
        num = r_array_here * ne_rarray_here
        denom = jnp.sqrt(r_array_here ** 2 - rp ** 2)
        toint = num / denom
        val = 2. * jsi.trapezoid(toint * r_array_here, jnp.log(r_array_here))
        return self.const_coeff_kSZ * val

    @partial(jit, static_argnums=(0,))
    def get_Pe_healpix(self, jpix):
        prop_jpix = self.pix_prop_all[jpix]
        y_jpix = jnp.exp(self.log_y2D_interp(prop_jpix[0], prop_jpix[1], prop_jpix[2]))        
        return y_jpix

    @partial(jit, static_argnums=(0,))
    def get_kSZ_healpix(self, jpix):
        prop_jpix = self.pix_prop_all[jpix]
        zval = prop_jpix[1]
        tau = self.tau_interp(zval)
        fac = jnp.exp(-tau)
        ksz_jpix = -1 * fac * prop_jpix[3] * jnp.exp(self.log_ne2D_interp(prop_jpix[0], prop_jpix[1], prop_jpix[2]))        
        return ksz_jpix

 