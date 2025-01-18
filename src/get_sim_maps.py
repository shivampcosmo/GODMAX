import os, sys
from base_class import get_vmapped_func, get_vmapped_func_warg
from get_radial_profiles import Profiles
import jax.numpy as jnp
from astropy.io import fits
import healpy as hp
import jax.scipy.integrate as jsi
from functools import partial
from astropy import constants as const
from scipy import interpolate as interp
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.keV / u.cm**3)).value
G_new_rhom = const.G.to(u.Mpc**3 / ((u.s**2) * u.M_sun))
import helpers.constants as constants
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
import interpax
from tqdm import tqdm
import time


class get_sim_map(Profiles):
    def __init__(
                self,
                sim_params_dict: dict,
                halo_params_dict: dict,
                analysis_dict: dict,     
                other_params_dict: dict,
                mock_params_dict: dict,                
                Profiles_obj=None,
            ):    
        if Profiles_obj is None:
            super().__init__(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
        else:
            self.__dict__.update(Profiles_obj.__dict__)


        H_array = self.H0 * bkgrd.H(self.cosmo_jax, self.scale_fac_a_array)        
        self.H_array_interp = interpax.Interpolator1D(self.z_array, H_array, extrap=True)

        self.rp_array = self.r_array[2:-2]

        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        coeff = sigmat / (m_e * (c ** 2))
        oneMpc = (((10 ** 6)) * (u.pc).to(u.m)) * (u.m)

        get_ymap = mock_params_dict.get('get_ymap', False)
        get_kSZmap = mock_params_dict.get('get_kSZmap', False)
        get_taumap = mock_params_dict.get('get_taumap', False)

        if get_ymap:
            self.Pe_mat_physical = self.Pe_mat_physical
            self.const_coeff = (((coeff * oneMpc).to(((u.cm ** 3) / u.keV))).value)/(self.cosmo_params['H0']/100.)

            self.y2D_mat_physical = get_vmapped_func(self.get_y2D_phyical_proj, 3)(jnp.arange(len(self.rp_array)), jnp.arange(len(self.z_array)), jnp.arange(len(self.M_array))).T
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

        if get_kSZmap or get_taumap:
            self.ne_mat_physical = self.ne_mat_physical

            self.tauz_array = vmap(self.get_tau_z)(jnp.arange(len(self.z_array)))
            self.tau_interp = interpax.Interpolator1D(self.z_array, self.tauz_array, extrap=True)

            self.ne2D_mat_physical = get_vmapped_func(self.get_ne2D_phyical_proj, 3)(jnp.arange(len(self.rp_array)), jnp.arange(len(self.z_array)), jnp.arange(len(self.M_array))).T
            self.log_ne2D_interp = interpax.Interpolator3D(jnp.log(self.rp_array), self.z_array, jnp.log(self.M_array), jnp.log(self.ne2D_mat_physical), extrap=True)

            self.nearby_pix_all = mock_params_dict['nearby_pix_all']
            self.pix_prop_all = mock_params_dict['pix_prop_all']
            pix_unique = np.unique(self.nearby_pix_all)
            sort_index_nearby_pix_all = np.argsort(self.nearby_pix_all)
            sorted_nearby_pix_all = self.nearby_pix_all[sort_index_nearby_pix_all]
            change_points = np.diff(sorted_nearby_pix_all, prepend=sorted_nearby_pix_all[0]-1, append=sorted_nearby_pix_all[-1]+1) != 0 
            boundaries = np.where(change_points)[0]

            self.nside_map = mock_params_dict['nside']

            if get_kSZmap:
                coeff_kSZ = sigmat/(c)
                self.const_coeff_kSZ =  (((coeff_kSZ * oneMpc).to(((u.cm ** 3) / (u.km/u.s)))).value)
                self.ksz_sim = jnp.zeros(self.nside_map**2 * 12)
                self.kszjpix_all = vmap(self.get_kSZ_healpix)(jnp.arange(len(self.pix_prop_all)))
                kszpix_all_sorted = self.kszjpix_all[sort_index_nearby_pix_all]
                kszpix_sum = np.add.reduceat(kszpix_all_sorted, boundaries[:-1])
                kszmap_final = np.zeros(12 * mock_params_dict['nside']**2)
                kszmap_final[pix_unique] = kszpix_sum
                self.kszmap_final = kszmap_final

            if get_taumap:
                coeff_tau = sigmat
                self.const_coeff_tau =  ((coeff_tau * oneMpc).to(u.cm ** 3)).value
                self.tau_sim = jnp.zeros(self.nside_map**2 * 12)
                self.taujpix_all = vmap(self.get_tau_healpix)(jnp.arange(len(self.pix_prop_all)))
                taupix_all_sorted = self.taujpix_all[sort_index_nearby_pix_all]
                taupix_sum = np.add.reduceat(taupix_all_sorted, boundaries[:-1])
                taumap_final = np.zeros(12 * mock_params_dict['nside']**2)
                taumap_final[pix_unique] = taupix_sum
                self.taumap_final = taumap_final

    @partial(jit, static_argnums=(0,))        
    def get_tau_z(self, jz):
        z = self.z_array[jz]
        z_array = jnp.linspace(0.000001, z, 100)
        H_array = self.H_array_interp(z_array)
        to_int = (1 + z_array)**2 / H_array
        tau_z = jsi.trapezoid(to_int, z_array)
        sigma_T = const.sigma_T.to(u.Mpc**2).value
        c = const.c.to(u.km/u.s).value
        ne_bar = self.cosmo_params['Ob0'] * RHO_CRIT_0_MPC3 / (mue * mp)
        val = sigma_T * ne_bar * c * tau_z
        return val

    @partial(jit, static_argnums=(0,))        
    def get_y2D_phyical_proj(self, jrp, jz, jM, num_trapz_points=40):
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
    def get_ne2D_phyical_proj(self, jrp, jz, jM, num_trapz_points=40):
        rp = self.rp_array[jrp]
        zval = self.z_array[jz]
        r_array_here = jnp.exp(jnp.linspace(jnp.log(rp*1.01), jnp.log(jnp.max(self.r_array)), num_trapz_points))
        ne_rarray_here = jnp.exp(jnp.interp(jnp.log(r_array_here), jnp.log(self.r_array/(1 + zval)), jnp.log(self.ne_mat_physical[:,jz, jM])))
        num = r_array_here * ne_rarray_here
        denom = jnp.sqrt(r_array_here ** 2 - rp ** 2)
        toint = num / denom
        val = 2. * jsi.trapezoid(toint * r_array_here, jnp.log(r_array_here))
        return val

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
        ksz_jpix = -1 * self.const_coeff_kSZ * fac * prop_jpix[3] * jnp.exp(self.log_ne2D_interp(prop_jpix[0], prop_jpix[1], prop_jpix[2]))        
        return ksz_jpix

    @partial(jit, static_argnums=(0,))
    def get_tau_healpix(self, jpix):
        prop_jpix = self.pix_prop_all[jpix]
        zval = prop_jpix[1]
        tau_jpix = self.const_coeff_tau * jnp.exp(self.log_ne2D_interp(prop_jpix[0], prop_jpix[1], prop_jpix[2]))        
        return tau_jpix
 