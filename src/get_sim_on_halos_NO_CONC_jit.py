import os, sys
from get_BCMP_profile_jit import BCM_18_wP
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

# @jit
# def hav(theta):
#     return jnp.sin(theta/2.)**2.

# #assumes radians
# @jit
# def ang_sep(ra1, dec1, ra2, dec2):
#     #Haversine formula
#     theta = 2.*jnp.arcsin(jnp.sqrt(hav(dec1 - dec2) + jnp.cos(dec1)*jnp.cos(dec2)*hav(ra1-ra2)))
#     return theta

# @jit
# def eq2ang(ra,dec):
#     phi = ra*jnp.pi/180.
#     theta = (jnp.pi/2.) - dec*(jnp.pi/180.)
#     return theta, phi

# @jit
# def ang2eq(theta,phi):
#     ra = phi*180./jnp.pi
#     dec = 90. - theta*180./jnp.pi
#     return ra, dec

class get_mock_map:
    def __init__(
                self,
                sim_params_dict,
                halo_params_dict,
                mock_params_dict,                
                # analysis_dict,
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


        H0 = 100. * (u.km / (u.s * u.Mpc))
        self.rho_m_bar = self.cosmo_params['Om0'] * ((3 * (H0**2) / (8 * jnp.pi * G_new_rhom)).to(u.M_sun / (u.Mpc**3))).value

        # self.cmean_all_Mz = mock_params_dict['cmean_jM_jz']
        # self.c_array = BCMP_obj.conc_array
        self.M_array = BCMP_obj.M_array
        self.z_array = BCMP_obj.z_array
        self.r_array = BCMP_obj.r_array

        self.rp_array = BCMP_obj.r_array[2:-2]

        self.Pe_mat_physical = BCMP_obj.Pe_mat_physical
        self.rho_dmb_mat_physical = BCMP_obj.rho_dmb_mat_physical

        # vmap_func1 = vmap(self.get_conc_Mz_Duffy08, (0, None))
        # vmap_func2 = vmap(vmap_func1, (None, 0))
        # self.conc_Mz_mat = vmap_func2(jnp.arange(len(self.z_array)), jnp.arange(len(self.M_array))).T

        self.conc_Mz_mat = BCMP_obj.conc_Mz_mat

        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        coeff = sigmat / (m_e * (c ** 2))
        oneMpc = (((10 ** 6)) * (u.pc).to(u.m)) * (u.m)
        self.const_coeff = (((coeff * oneMpc).to(((u.cm ** 3) / u.keV))).value)/(self.cosmo_params['H0']/100.)

        # coeff_kSZ = sigmat/(c)
        # self.const_coeff_kSZ =  (((coeff_kSZ * oneMpc).to(((u.Mpc ** 2) / (u.km/u.s)))).value)

        vmap_func1 = vmap(self.get_y2D_phyical_proj, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.y2D_mat_physical = vmap_func3(jnp.arange(len(self.rp_array)), jnp.arange(len(self.z_array)), jnp.arange(len(self.M_array))).T        

        self.log_y2D_interp = interpax.Interpolator3D(jnp.log(self.rp_array), self.z_array, jnp.log(self.M_array), jnp.log(self.y2D_mat_physical), extrap=True)

        self.nside_map = mock_params_dict['nside']
        self.y_sim = jnp.zeros(self.nside_map**2 * 12)

        self.halo_cat_z = mock_params_dict['halo_z']
        self.halo_cat_M = mock_params_dict['halo_M']
        self.halo_cat_scale_fac = 1./(1. + self.halo_cat_z)
        self.halo_cat_ra = mock_params_dict['halo_ra']
        self.halo_cat_dec = mock_params_dict['halo_dec']
        self.halo_vlos = mock_params_dict['halo_vlos']

        self.nearby_pix_all = mock_params_dict['nearby_pix_all']
        # self.distance_nearby_all = mock_params_dict['distance_nearby_all']
        # self.log10M_ind_all = mock_params_dict['log10M_ind_all']
        # self.z_ind_all = mock_params_dict['z_ind_all']
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
        # self.start_ind_all = mock_params_dict['start_ind']
        # self.end_ind_all = mock_params_dict['end_ind']


        # # self.distance_nearby_all = mock_params_dict['distance_nearby_all']
        # # self.nearbypix_all = mock_params_dict['nearbypix_all']
        # halo_cat_rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(self.cosmo_jax,self.halo_cat_scale_fac) * 1e9
        # mdef_delta=200
        # halo_cat_rho_treshold = mdef_delta * halo_cat_rho_c_z
        # self.halo_cat_R200c = (self.halo_cat_M * 3.0 / 4.0 / jnp.pi / halo_cat_rho_treshold)**(1.0 / 3.0)
        # self.halo_cat_DA = bkgrd.angular_diameter_distance(self.cosmo_jax,self.halo_cat_scale_fac)
        # self.max_paint_R200c_factor = 3.
        # self.nearby_pix_all = self.get_nearby_pix()
        # self.distance_nearby_all = self.get_physical_distances_nearby_pix()

        # self.run = vmap(self.get_Pe_healpix_jit)(jnp.arange(len(self.halo_cat_z)))

        # for jhalo in tqdm(range(len(self.halo_cat_z))):
            # y_jhalo = self.get_Pe_healpix_jit(jhalo)
            # y_jhalo, nearby_jhalo = self.get_Pe_healpix(jhalo)
        #     self.y_sim = self.y_sim.at[nearby_jhalo].add(y_jhalo)
            # self.ysim = jax.ops.index_add(self.ysim, self.nearby_pix_all[jhalo], y_jhalo)

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
    def get_M_to_R(self, jhalo, mdef_delta=200):
        M_halo = self.halo_M[jhalo]
        z_halo = self.halo_z[jhalo]
        scale_fac_halo = 1. / (1. + z_halo)
        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(self.cosmo_jax,scale_fac_halo) * 1e9
        rho_treshold = mdef_delta * rho_c_z
        R = (M_halo * 3.0 / 4.0 / jnp.pi / rho_treshold)**(1.0 / 3.0)
        # keep physical coordinates
        # R *= (1 + self.z_array[jz])
        return R

    def get_nearby_pix(self):
        nearby_pix_all = {}
        for jhalo in range(len(self.halo_cat_z)):
            vec = hp.ang2vec(self.halo_cat_ra[jhalo], self.halo_cat_dec[jhalo], lonlat=True)
            nearby_angle = self.max_paint_R200c_factor*self.halo_cat_R200c[jhalo]/self.halo_cat_DA[jhalo]
            nearby_pix = hp.query_disc(self.nside_map, vec, nearby_angle)
            nearby_pix_all[jhalo] = jnp.array(nearby_pix)
        return nearby_pix_all

    def get_physical_distances_nearby_pix(self):
        physical_distances_all = {}
        for jhalo in range(len(self.halo_cat_z)):
            nearby_pix = self.nearby_pix_all[jhalo]
            nearby_ra, nearby_dec = hp.pix2ang(self.nside_map, nearby_pix, lonlat=True)
            angular_distances = ang_sep(self.halo_cat_ra[jhalo]*np.pi/180., self.halo_cat_dec[jhalo]*np.pi/180., nearby_ra*np.pi/180., nearby_dec*np.pi/180.)
            physical_distances = self.halo_cat_DA[jhalo]*angular_distances
            physical_distances_all[jhalo] = jnp.array(physical_distances)
        return physical_distances_all


    # @partial(jit, static_argnums=(0,))        
    # def hav(self, theta):
    #     return jnp.sin(theta/2.)**2.

    # @partial(jit, static_argnums=(0,))        
    # def ang_sep(self, ra1, dec1, ra2, dec2):
    #     #Haversine formula
    #     theta = 2.*jnp.arcsin(jnp.sqrt(self.hav(dec1 - dec2) + jnp.cos(dec1)*jnp.cos(dec2)*self.hav(ra1-ra2)))
    #     return theta

    @partial(jit, static_argnums=(0,))        
    def get_y2D_phyical_proj(self, jrp, jz, jM, num_trapz_points=32):
        # cval_jM_jz = self.conc_Mz_mat[jz, jM]
        # jc = jnp.argmin(jnp.abs(self.c_array - cval_jM_jz))
        rp = self.rp_array[jrp]
        r_array_here = jnp.exp(jnp.linspace(jnp.log(rp*1.01), jnp.log(jnp.max(self.r_array)), num_trapz_points))
        Pe_rarray_here = jnp.exp(jnp.interp(jnp.log(r_array_here), jnp.log(self.r_array), jnp.log(self.Pe_mat_physical[:,jz, jM])))
        num = r_array_here * Pe_rarray_here
        denom = jnp.sqrt(r_array_here ** 2 - rp ** 2)
        toint = num / denom
        val = 2. * jsi.trapezoid(toint * r_array_here, jnp.log(r_array_here))
        return self.const_coeff * val

    @partial(jit, static_argnums=(0,))        
    def get_rhodmb_phyical_proj(self, jrp, jz, jM, num_trapz_points=32):
        # cval_jM_jz = self.conc_Mz_mat[jz, jM]
        # jc = jnp.argmin(jnp.abs(self.c_array - cval_jM_jz))
        rp = self.rp_array[jrp]
        r_array_here = jnp.linspace(jnp.log(rp*1.01), jnp.log(jnp.max(self.r_array)), num_trapz_points)
        rhodmb_rarray_here = jnp.exp(jnp.interp(jnp.log(r_array_here), jnp.log(self.r_array), jnp.log(self.rho_dmb_mat_physical[:,jz, jM])))
        num = r_array_here * rhodmb_rarray_here
        denom = np.sqrt(r_array_here ** 2 - rp ** 2)
        toint = num / denom
        val = 2. * jsi.trapezoid(toint * r_array_here, jnp.log(r_array_here))
        return val


    # @partial(jit, static_argnums=(0,))
    # def get_Pe_healpix_jit(self, jhalo):
    #     # distance_nearby_jhalo = self.distance_nearby_all[jhalo]
    #     # nearbypix_jhalo = self.nearby_pix_all[jhalo]

    #     start_ind_jhalo = self.start_ind_all[jhalo].astype(int)
    #     end_ind_jhalo = self.end_ind_all[jhalo].astype(int)
    #     # distance_nearby_jhalo = jax.lax.dynamic_slice(self.distance_nearby_all, (start_ind_jhalo,), (end_ind_jhalo - start_ind_jhalo,))
    #     # nearbypix_jhalo = jax.lax.dynamic_slice(self.nearby_pix_all, start_ind_jhalo, end_ind_jhalo - start_ind_jhalo)

    #     distance_nearby_jhalo = self.distance_nearby_all.at[start_ind_jhalo:end_ind_jhalo].get()
    #     nearbypix_jhalo = self.nearby_pix_all.at[start_ind_jhalo:end_ind_jhalo].get()

    #     y_jhalo = jnp.exp(self.log_y2D_interp(jnp.log(distance_nearby_jhalo), self.halo_cat_z[jhalo], jnp.log(self.halo_cat_M[jhalo])))
        
    #     #Add contribution from cluster to ymap
    #     # self.y_sim[nearbypix_jhalo] += y_jhalo
    #     # self.y_sim = self.y_sim.at[nearbypix_jhalo].add(y_jhalo)
    #     # return y_jhalo, nearbypix_jhalo
    #     return y_jhalo

    # # @partial(jit, static_argnums=(0,))
    # def get_Pe_healpix_jit(self, jhalo):

    #     start_ind_jhalo = self.start_ind_all[jhalo].astype(int)
    #     end_ind_jhalo = self.end_ind_all[jhalo].astype(int)

    #     # distance_nearby_jhalo = lax.dynamic_slice(self.distance_nearby_all, (start_ind_jhalo,), (end_ind_jhalo - start_ind_jhalo,))
    #     # nearbypix_jhalo = lax.dynamic_slice(self.nearby_pix_all, (start_ind_jhalo,), (end_ind_jhalo - start_ind_jhalo,))

    #     # distance_nearby_jhalo = self.distance_nearby_all.at[start_ind_jhalo:end_ind_jhalo].get()
    #     # nearbypix_jhalo = self.nearby_pix_all.at[start_ind_jhalo:end_ind_jhalo].get()


    #     # distance_nearby_jhalo = lax.dynamic_slice_in_dim(self.distance_nearby_all, start_ind_jhalo, end_ind_jhalo - start_ind_jhalo)
    #     # nearbypix_jhalo = lax.dynamic_slice_in_dim(self.nearby_pix_all, start_ind_jhalo, end_ind_jhalo - start_ind_jhalo)


    #     y_jhalo = jnp.exp(self.log_y2D_interp(jnp.log(distance_nearby_jhalo), self.halo_cat_z[jhalo], jnp.log(self.halo_cat_M[jhalo])))
    #     # y_jhalo = []
    #     # for jp in range((end_ind_jhalo - start_ind_jhalo)):
    #         # distance_nearby_jhalo_jp = self.distance_nearby_all[start_ind_jhalo + jp]
    #         # y_jhalo.append(jnp.exp(self.log_y2D_interp(jnp.log(distance_nearby_jhalo_jp), self.halo_cat_z[jhalo], jnp.log(self.halo_cat_M[jhalo]))))
        
    #     return y_jhalo

    @partial(jit, static_argnums=(0,))
    def get_Pe_healpix(self, jpix):
        # distance_nearby_jhalo = self.distance_nearby_all.at[start_ind_jhalo:end_ind_jhalo].get()
        # nearbypix_jhalo = self.nearby_pix_all.at[start_ind_jhalo:end_ind_jhalo].get()
        prop_jpix = self.pix_prop_all[jpix]
        # y_jhalo = jnp.exp(self.log_y2D_interp(jnp.log(distance_nearby_jhalo), self.halo_cat_z[jhalo], jnp.log(self.halo_cat_M[jhalo])))
        y_jpix = jnp.exp(self.log_y2D_interp(prop_jpix[0], prop_jpix[1], prop_jpix[2]))        
        
        #Add contribution from cluster to ymap
        # self.y_sim[nearbypix_jhalo] += y_jhalo
        # self.y_sim = self.y_sim.at[nearbypix_jhalo].add(y_jhalo)
        return y_jpix
        # return y_jhalo   

    # # @partial(jit, static_argnums=(0,))
    # def get_Pe_healpix(self, jhalo):
    #     # distance_nearby_jhalo = self.distance_nearby_all[jhalo]
    #     # nearbypix_jhalo = self.nearby_pix_all[jhalo]

    #     ti = time.time()
    #     start_ind_jhalo = self.start_ind_all[jhalo].astype(int)
    #     end_ind_jhalo = self.end_ind_all[jhalo].astype(int)
    #     tf = time.time()
    #     print('getting indices took: ', tf-ti)
    #     # distance_nearby_jhalo = jax.lax.dynamic_slice(self.distance_nearby_all, (start_ind_jhalo,), (end_ind_jhalo - start_ind_jhalo,))
    #     # nearbypix_jhalo = jax.lax.dynamic_slice(self.nearby_pix_all, start_ind_jhalo, end_ind_jhalo - start_ind_jhalo)

    #     distance_nearby_jhalo = self.distance_nearby_all.at[start_ind_jhalo:end_ind_jhalo].get()
    #     nearbypix_jhalo = self.nearby_pix_all.at[start_ind_jhalo:end_ind_jhalo].get()

    #     y_jhalo = jnp.exp(self.log_y2D_interp(jnp.log(distance_nearby_jhalo), self.halo_cat_z[jhalo], jnp.log(self.halo_cat_M[jhalo])))
        
    #     #Add contribution from cluster to ymap
    #     # self.y_sim[nearbypix_jhalo] += y_jhalo
    #     # self.y_sim = self.y_sim.at[nearbypix_jhalo].add(y_jhalo)
    #     return y_jhalo, nearbypix_jhalo
    #     # return y_jhalo    
