import os, sys
from base_class import get_vmapped_func, get_vmapped_func_warg
from get_radial_profiles import Profiles
import jax
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
from mcfitjax.transforms import Hankel


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
        # self.rp_array = self.r_array[:-3] 

        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        coeff = sigmat / (m_e * (c ** 2))
        oneMpc = (((10 ** 6)) * (u.pc).to(u.m)) * (u.m)

        get_ymap = mock_params_dict.get('get_ymap', False)
        get_kSZmap = mock_params_dict.get('get_kSZmap', False)
        get_taumap = mock_params_dict.get('get_taumap', False)

        # norm_profiles = mock_params_dict.get('norm_profiles', False)
        smooth_profiles = mock_params_dict.get('smooth_profiles', False)        
        self.start_ind_all = jnp.array(mock_params_dict['start_ind'])
        self.end_ind_all = jnp.array(mock_params_dict['end_ind'])
        self.ang_distance_all = jnp.array(mock_params_dict.get('ang_distance_all', [1e-3]))
        self.rp_max_all = jnp.array(mock_params_dict.get('rp_max_all', [1e-3]))

        self.nside_map = mock_params_dict['nside']
        theta_fwhm_arcmin = hp.nside2resol(self.nside_map, arcmin=True)
        theta_fwhm_rad = (theta_fwhm_arcmin / 60.) * (jnp.pi / 180.)
        self.sigma_val = theta_fwhm_rad / jnp.sqrt(8. * jnp.log(2.))


        self.nearby_pix_all = mock_params_dict['nearby_pix_all']
        self.pix_prop_all = mock_params_dict['pix_prop_all']

        pix_unique = np.unique(self.nearby_pix_all)
        sort_index_nearby_pix_all = np.argsort(self.nearby_pix_all)
        sorted_nearby_pix_all = self.nearby_pix_all[sort_index_nearby_pix_all]
        
        change_points = np.diff(sorted_nearby_pix_all, prepend=sorted_nearby_pix_all[0]-1, append=sorted_nearby_pix_all[-1]+1) != 0 
        boundaries = np.where(change_points)[0]
        


        if get_ymap:
            # self.Pe_mat_physical = self.Pe_mat_physical
            self.const_coeff = (((coeff * oneMpc).to(((u.cm ** 3) / u.keV))).value)/(self.cosmo_params['H0']/100.)

            self.y2D_mat_physical = get_vmapped_func(self.get_y2D_phyical_proj, 3)(jnp.arange(len(self.rp_array)), jnp.arange(len(self.z_array)), jnp.arange(len(self.M_array))).T
            if smooth_profiles:
                y2D_mat_smooth = get_vmapped_func(self.get_y2D_smoothed_prof, 2)(jnp.arange(len(self.z_array)), jnp.arange(len(self.M_array))).T
                self.y2D_mat_physical = y2D_mat_smooth

            self.log_y2D_interp = interpax.Interpolator3D(jnp.log(self.rp_array), self.z_array, jnp.log(self.M_array), jnp.log(self.y2D_mat_physical), extrap=[1e-20, 1e-20])

            self.yjpix_all = vmap(self.get_Pe_healpix)(jnp.arange(len(self.pix_prop_all)))
            self.yjpix_all_normed = self.yjpix_all
            # if norm_profiles:
                # self.max_slice_size = int(jnp.max(self.end_ind_all - self.start_ind_all))

                # self.yjpix_sum = vmap(self.get_sum_yjpix)(jnp.arange(len(self.start_ind_all)))
    
                # self.logM_halos_start = self.pix_prop_all[self.start_ind_all, 2]
                # self.z_halos_start = self.pix_prop_all[self.start_ind_all, 1]
                # self.yint_2D_all = vmap(self.get_y2D_integrate_rp)(jnp.arange(len(self.start_ind_all)))

                # self.yint_ratio = self.yint_2D_all / self.yjpix_sum

                # # repeat the yint_ratio for each pixel in the halo:
                # yint_ratio_all = jnp.repeat(self.yint_ratio, self.end_ind_all - self.start_ind_all)
                # self.yjpix_all_normed = self.yjpix_all * yint_ratio_all
            # else:
                # self.yjpix_all_normed = self.yjpix_all


            ypix_all_sorted = self.yjpix_all_normed[sort_index_nearby_pix_all]
            ypix_sum = np.add.reduceat(ypix_all_sorted, boundaries[:-1])
            ymap_final = np.zeros(12 * self.nside_map**2, dtype=np.float32)
            ymap_final[pix_unique] = ypix_sum.astype(np.float32)
            self.ymap_final = ymap_final

        if get_kSZmap or get_taumap:
            # self.ne_mat_physical = self.ne_mat_physical

            self.tauz_array = vmap(self.get_tau_z)(jnp.arange(len(self.z_array)))
            self.tau_interp = interpax.Interpolator1D(self.z_array, self.tauz_array, extrap=True)

            self.ne2D_mat_physical = get_vmapped_func(self.get_ne2D_phyical_proj, 3)(jnp.arange(len(self.rp_array)), jnp.arange(len(self.z_array)), jnp.arange(len(self.M_array))).T
            if smooth_profiles:
                ne2D_mat_smooth = get_vmapped_func(self.get_ne2D_smoothed_prof, 2)(jnp.arange(len(self.z_array)), jnp.arange(len(self.M_array))).T
                self.ne2D_mat_physical = ne2D_mat_smooth

            self.log_ne2D_interp = interpax.Interpolator3D(jnp.log(self.rp_array), self.z_array, jnp.log(self.M_array), jnp.log(self.ne2D_mat_physical), method='linear', extrap=True)

            if get_kSZmap:
                coeff_kSZ = sigmat/(c)
                self.const_coeff_kSZ =  (((coeff_kSZ * oneMpc).to(((u.cm ** 3) / (u.km/u.s)))).value)/(self.cosmo_params['H0']/100.)
                # self.ksz_sim = jnp.zeros(self.nside_map**2 * 12)
                kszjpix_all = vmap(self.get_kSZ_healpix)(jnp.arange(len(self.pix_prop_all)))
                kszpix_all_sorted = kszjpix_all[sort_index_nearby_pix_all]
                kszpix_sum = np.add.reduceat(kszpix_all_sorted, boundaries[:-1])
                kszmap_final = np.zeros(12 * mock_params_dict['nside']**2, dtype=np.float32)
                kszmap_final[pix_unique] = kszpix_sum.astype(np.float32)
                self.kszmap_final = kszmap_final

            if get_taumap:
                coeff_tau = sigmat
                self.const_coeff_tau =  (((coeff_tau * oneMpc).to(u.cm ** 3)).value)/(self.cosmo_params['H0']/100.)
                # self.tau_sim = jnp.zeros(self.nside_map**2 * 12)
                taujpix_all = vmap(self.get_tau_healpix)(jnp.arange(len(self.pix_prop_all)))
                taupix_all_sorted = taujpix_all[sort_index_nearby_pix_all]
                taupix_sum = np.add.reduceat(taupix_all_sorted, boundaries[:-1])
                taumap_final = np.zeros(12 * mock_params_dict['nside']**2, dtype=np.float32)
                taumap_final[pix_unique] = taupix_sum.astype(np.float32)
                self.taumap_final = taumap_final

    @partial(jit, static_argnums=(0,))        
    def get_tau_z(self, jz):
        z = self.z_array[jz]
        z_array = jnp.linspace(0.001, z, 100)
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
        # val = 2. * jsi.trapezoid(toint, (r_array_here))        
        return self.const_coeff * val

    @partial(jit, static_argnums=(0,))        
    def get_y2D_smoothed_prof(self, jz, jM):
        zval = self.z_array[jz]
        DA_val = self.DA_array[jz]
        theta_array = (self.rp_array / DA_val)  # in radians 
        ell_out, yell_out = (Hankel(theta_array, nu=0, q=1.0, nx=len(theta_array), lowring=True)(self.y2D_mat_physical[:,jz, jM], extrap=True))
        yell_out = jnp.array(yell_out * ((2 * jnp.pi)))   
        b_ell = jnp.exp(-0.5 * ell_out * (ell_out + 1.) * (self.sigma_val ** 2))

        theta_out, y_out_smooth = (Hankel(ell_out, nu=0, q=1.0, nx=len(ell_out), lowring=True)(jnp.clip(b_ell * yell_out, 1e-40, 1e10), extrap=True))
        y_out_smooth = jnp.array(y_out_smooth * (1./(2 * jnp.pi)))   
        y_out_smooth = jnp.clip(y_out_smooth, 1e-20, jnp.max(self.y2D_mat_physical[jz, jM]))

        return y_out_smooth        

    @partial(jit, static_argnums=(0,))        
    def get_y2D_integrate_rp(self, jhalo):
        rp_array_here = jnp.exp(jnp.linspace(jnp.log(self.rp_array[1]), jnp.log(0.95*self.rp_max_all[jhalo]), 16))
        # logM_here = self.logM_halos_start[jhalo]
        logM_here = self.pix_prop_all[self.start_ind_all, 2][jhalo]
        # z_here = self.z_halos_start[jhalo]
        z_here = self.pix_prop_all[self.start_ind_all, 1][jhalo]
        surface_area_here = 4 * jnp.pi * self.ang_distance_all[jhalo]**2
        y2D_here = jnp.exp(self.log_y2D_interp(jnp.log(rp_array_here), z_here, logM_here))
        val = 4 * jnp.pi * 2 * jnp.pi * jsi.trapezoid((rp_array_here**2) * y2D_here, jnp.log(rp_array_here))/surface_area_here
        return val

    @partial(jit, static_argnums=(0,))        
    def get_sum_yjpix(self, jhalo):
        start_ind = self.start_ind_all[jhalo]
        end_ind = self.end_ind_all[jhalo]
        actual_size = end_ind - start_ind
        
        sliced_array = jax.lax.dynamic_slice(
            self.yjpix_all, (start_ind,), (self.max_slice_size,)
        )

        mask = jnp.arange(self.max_slice_size) < actual_size
        area_pixel = 4 * jnp.pi / (12 * self.nside_map**2)
        return jnp.sum(sliced_array * mask) * area_pixel

    @partial(jit, static_argnums=(0,))
    def get_Pe_healpix(self, jpix):
        prop_jpix = self.pix_prop_all[jpix]
        y_jpix = jnp.exp(self.log_y2D_interp(prop_jpix[0], prop_jpix[1], prop_jpix[2]))        
        return y_jpix

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
        return val
    
    @partial(jit, static_argnums=(0,))        
    def get_ne2D_smoothed_prof(self, jz, jM):
        zval = self.z_array[jz]
        DA_val = self.DA_array[jz]
        theta_array = (self.rp_array / DA_val)  # in radians 
        ell_out, nell_out = (Hankel(theta_array, nu=0, q=1.0, nx=len(theta_array), lowring=True)(self.ne2D_mat_physical[:,jz, jM], extrap=True))
        nell_out = jnp.array(nell_out * ((2 * jnp.pi)))   
        b_ell = jnp.exp(-0.5 * ell_out * (ell_out + 1.) * (self.sigma_val ** 2))

        theta_out, n_out_smooth = (Hankel(ell_out, nu=0, q=1.0, nx=len(ell_out), lowring=True)(jnp.clip(b_ell * nell_out, 1e-40, 1e10), extrap=True))
        n_out_smooth = jnp.array(n_out_smooth * (1./(2 * jnp.pi)))   
        n_out_smooth = jnp.clip(n_out_smooth, 1e-20, jnp.max(self.ne2D_mat_physical[jz, jM]))

        return n_out_smooth 

    @partial(jit, static_argnums=(0,)) 
    def get_ne2D_integrate_rp(self, jhalo):
        rp_array_here = self.rp_array[1:-1]
        logM_here = self.logM_halos_start[jhalo]
        z_here = self.z_halos_start[jhalo]
        ne2D_here = jnp.exp(self.log_ne2D_interp(rp_array_here, z_here, logM_here))
        val = 2*jnp.pi * jsi.trapezoid((rp_array_here**2) * ne2D_here, jnp.log(rp_array_here))
        return val

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
        # zval = prop_jpix[1]
        tau_jpix = self.const_coeff_tau * jnp.exp(self.log_ne2D_interp(prop_jpix[0], prop_jpix[1], prop_jpix[2]))        
        return tau_jpix
 