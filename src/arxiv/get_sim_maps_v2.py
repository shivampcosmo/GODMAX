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
from jax.random import PRNGKey, split, poisson, uniform, normal
from functools import partial
from jax.scipy.ndimage import map_coordinates
key = PRNGKey(42)
key, subkey1, subkey2, subkey3, subkey4 = split(key, 5)

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
        get_kappamap = mock_params_dict.get('get_kappamap', False)   
        get_galmap = mock_params_dict.get('get_galmap', False)     

        smooth_profiles = mock_params_dict.get('smooth_profiles', False)        
        self.start_ind_all = jnp.array(mock_params_dict['start_ind'])
        self.end_ind_all = jnp.array(mock_params_dict['end_ind'])
        self.ang_distance_all = jnp.array(mock_params_dict.get('ang_distance_all', [1e-3]))
        self.rp_max_all = jnp.array(mock_params_dict.get('rp_max_all', [1e-3]))

        self.nside_map = mock_params_dict['nside']
        theta_fwhm_arcmin = hp.nside2resol(self.nside_map, arcmin=True)
        theta_fwhm_rad = (theta_fwhm_arcmin / 60.) * (jnp.pi / 180.)
        self.sigma_val = theta_fwhm_rad / jnp.sqrt(8. * jnp.log(2.))

        self.nearby_pix_all = jnp.array(mock_params_dict['nearby_pix_all'], dtype=jnp.int32)
        self.pix_prop_all = jnp.array(mock_params_dict['pix_prop_all'], dtype=jnp.float32)

        pix_unique = np.unique(self.nearby_pix_all)
        sort_index_nearby_pix_all = np.argsort(self.nearby_pix_all)
        sorted_nearby_pix_all = self.nearby_pix_all[sort_index_nearby_pix_all]
        
        change_points = np.diff(sorted_nearby_pix_all, prepend=sorted_nearby_pix_all[0]-1, 
                               append=sorted_nearby_pix_all[-1]+1) != 0 
        boundaries = np.where(change_points)[0]



        if get_ymap:
            self.const_coeff = (((coeff * oneMpc).to(((u.cm ** 3) / u.keV))).value)/(self.cosmo_params['H0']/100.)

            self.y2D_mat_physical = get_vmapped_func(self.get_y2D_phyical_proj, 3)(
                jnp.arange(len(self.rp_array)), 
                jnp.arange(len(self.z_array)), 
                jnp.arange(len(self.M_array))
            ).T.astype(jnp.float32)
            
            if smooth_profiles:
                y2D_mat_smooth = get_vmapped_func(self.get_y2D_smoothed_prof, 2)(
                    jnp.arange(len(self.z_array)), 
                    jnp.arange(len(self.M_array))
                ).T.astype(jnp.float32)
                del self.y2D_mat_physical  # Free old array
                self.y2D_mat_physical = y2D_mat_smooth

            self.log_y2D_interp = interpax.Interpolator3D(
                jnp.log(self.rp_array).astype(jnp.float32), 
                self.z_array.astype(jnp.float32), 
                jnp.log(self.M_array).astype(jnp.float32), 
                jnp.log(self.y2D_mat_physical), 
                extrap=[1e-20, 1e-20]
            )

            self.yjpix_all_normed = vmap(self.get_Pe_healpix)(jnp.arange(len(self.pix_prop_all)))

            ypix_all_sorted = np.asarray(self.yjpix_all_normed)[sort_index_nearby_pix_all]
            ypix_sum = np.add.reduceat(ypix_all_sorted, boundaries[:-1])
            ymap_final = np.zeros(12 * self.nside_map**2, dtype=np.float32)
            ymap_final[pix_unique] = ypix_sum.astype(np.float32)
            self.ymap_final = ymap_final

            del self.yjpix_all_normed, ypix_all_sorted

        if get_kSZmap or get_taumap:
            self.tauz_array = vmap(self.get_tau_z)(jnp.arange(len(self.z_array))).astype(jnp.float32)
            self.tau_interp = interpax.Interpolator1D(
                self.z_array.astype(jnp.float32), 
                self.tauz_array, 
                extrap=True
            )

            self.ne2D_mat_physical = get_vmapped_func(self.get_ne2D_phyical_proj, 3)(
                jnp.arange(len(self.rp_array)), 
                jnp.arange(len(self.z_array)), 
                jnp.arange(len(self.M_array))
            ).T.astype(jnp.float32)
            
            if smooth_profiles:
                ne2D_mat_smooth = get_vmapped_func(self.get_ne2D_smoothed_prof, 2)(
                    jnp.arange(len(self.z_array)), 
                    jnp.arange(len(self.M_array))
                ).T.astype(jnp.float32)
                del self.ne2D_mat_physical
                self.ne2D_mat_physical = ne2D_mat_smooth

            self.log_ne2D_interp = interpax.Interpolator3D(
                jnp.log(self.rp_array).astype(jnp.float32), 
                self.z_array.astype(jnp.float32), 
                jnp.log(self.M_array).astype(jnp.float32), 
                jnp.log(self.ne2D_mat_physical).astype(jnp.float32), 
                extrap=[1e-20, 1e-20]
                # method='linear', 
                # extrap=True
            )

            if get_kSZmap:
                coeff_kSZ = sigmat/(c)
                self.const_coeff_kSZ = (((coeff_kSZ * oneMpc).to(((u.cm ** 3) / (u.km/u.s)))).value)/(self.cosmo_params['H0']/100.)
                
                kszjpix_all = vmap(self.get_kSZ_healpix)(jnp.arange(len(self.pix_prop_all)))
                kszpix_all_sorted = np.asarray(kszjpix_all)[sort_index_nearby_pix_all]
                kszpix_sum = np.add.reduceat(kszpix_all_sorted, boundaries[:-1])
                kszmap_final = np.zeros(12 * mock_params_dict['nside']**2, dtype=np.float32)
                kszmap_final[pix_unique] = kszpix_sum.astype(np.float32)
                self.kszmap_final = kszmap_final
                del kszjpix_all, kszpix_all_sorted

            if get_taumap:
                coeff_tau = sigmat
                self.const_coeff_tau = (((coeff_tau * oneMpc).to(u.cm ** 3)).value)/(self.cosmo_params['H0']/100.)
                
                taujpix_all = vmap(self.get_tau_healpix)(jnp.arange(len(self.pix_prop_all)))
                taupix_all_sorted = np.asarray(taujpix_all)[sort_index_nearby_pix_all]
                taupix_sum = np.add.reduceat(taupix_all_sorted, boundaries[:-1])
                taumap_final = np.zeros(12 * mock_params_dict['nside']**2, dtype=np.float32)
                taumap_final[pix_unique] = taupix_sum.astype(np.float32)
                self.taumap_final = taumap_final
                del taujpix_all, taupix_all_sorted

        if get_kappamap:
            self.rho_dmb_mat_physical = (self.rho_dmb_mat / (self.scale_fac_a_array[None, :, None] ** 3)).astype(jnp.float32)

            self.rhom2D_mat_physical = get_vmapped_func(self.get_rhom2D_phyical_proj, 3)(
                jnp.arange(len(self.rp_array)), 
                jnp.arange(len(self.z_array)), 
                jnp.arange(len(self.M_array))
            ).T.astype(jnp.float32)
            
            if smooth_profiles:
                rhom2D_mat_smooth = get_vmapped_func(self.get_rhom2D_smoothed_prof, 2)(
                    jnp.arange(len(self.z_array)), 
                    jnp.arange(len(self.M_array))
                ).T.astype(jnp.float32)
                del self.rhom2D_mat_physical
                self.rhom2D_mat_physical = rhom2D_mat_smooth

            self.log_rhom2D_interp = interpax.Interpolator3D(
                jnp.log(self.rp_array).astype(jnp.float32), 
                self.z_array.astype(jnp.float32), 
                jnp.log(self.M_array).astype(jnp.float32), 
                jnp.log(self.rhom2D_mat_physical).astype(jnp.float32), 
                extrap=[1e-20, 1e-20]
                # method='linear', 
                # extrap=True
            )

            rhomjpix_all = vmap(self.get_rhom_healpix)(jnp.arange(len(self.pix_prop_all)))
            rhomjpix_all_sorted = np.asarray(rhomjpix_all)[sort_index_nearby_pix_all]
            rhomjpix_sum = np.add.reduceat(rhomjpix_all_sorted, boundaries[:-1])
            rhommap_final = np.zeros(12 * mock_params_dict['nside']**2, dtype=np.float32)
            rhommap_final[pix_unique] = rhomjpix_sum.astype(np.float32)
            self.rhommap_final = rhommap_final
            del rhomjpix_all, rhomjpix_all_sorted


        if get_galmap:
            self.mass_grid = self.M_array.astype(jnp.float32)
            self.z_grid = self.z_array.astype(jnp.float32)
            self.r_comoving_grid = (self.r_array * 1000).astype(jnp.float32)  # convert Mpc/h to kpc/h
            
            # Get the profile table
            mock_rho_table_comoving = self.rho_clm_mat
            self.mock_rho_table = mock_rho_table_comoving / (self.scale_fac_a_array[None, :, None] ** 3)
            
            # Create PPF interpolator
            self.ppf_table_3d = self.create_ppf_interpolator_3d(
                self.mock_rho_table,
                self.mass_grid,
                self.z_grid,
                self.r_comoving_grid
            )
            
            # HOD parameters
            mean_ncen_all, mean_nsat_all = self.Ncen_mat, self.Nsat_mat
            max_mean_nsat = jnp.max(mean_nsat_all)
            max_gals_per_halo = int(jnp.ceil(max_mean_nsat + 10 * jnp.sqrt(max_mean_nsat))) + 1
            
            # Get number of halos and create random key
            NUM_HALOS = len(mock_params_dict['halo_ra'])
            key = PRNGKey(mock_params_dict.get('random_seed', 42))
            
            # Create a wrapper function that doesn't include self in vmapped arguments
            def populate_wrapper(key, ra, dec, z, mass):
                return self.populate_one_halo(
                    key, ra, dec, z, mass, max_gals_per_halo,
                    self.ppf_table_3d["mass_grid"],
                    self.ppf_table_3d["z_grid"],
                    self.ppf_table_3d["r_comoving_grid"],
                    self.ppf_table_3d["cdf_table"]
                )
            
            # Vectorize the wrapper
            vectorized_populate = vmap(populate_wrapper)
            jitted_vectorized_populate = jit(vectorized_populate)

            print(f"\n--- Step 3: Populating {NUM_HALOS} halos with galaxies ---")
            keys = split(key, NUM_HALOS)

            padded_galaxy_catalog = jitted_vectorized_populate(
                keys,
                jnp.array(mock_params_dict['halo_ra'], dtype=jnp.float32),
                jnp.array(mock_params_dict['halo_dec'], dtype=jnp.float32),
                jnp.array(mock_params_dict['halo_z'], dtype=jnp.float32),
                jnp.array(mock_params_dict['halo_M'], dtype=jnp.float32)
            )
            padded_galaxy_catalog.block_until_ready()
            print("Population complete.")

            # --- 6. Post-processing ---
            print("\n--- Step 4: Filtering padded catalog ---")
            flat_padded_catalog = padded_galaxy_catalog.reshape(-1, 6)
            valid_mask = flat_padded_catalog[:, 5] > 0.5
            final_galaxy_catalog = flat_padded_catalog[valid_mask]
            self.final_galaxy_catalog = final_galaxy_catalog







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
        r_max = jnp.min(jnp.array([jnp.max(self.r_array), rp * 100.0]))  # Limit integration range
        r_array_here = jnp.exp(jnp.linspace(jnp.log(rp*1.01), jnp.log(r_max), num_trapz_points))
        
        Pe_rarray_here = jnp.exp(jnp.interp(
            jnp.log(r_array_here), 
            jnp.log(self.r_array/(1 + zval)), 
            jnp.log(self.Pe_mat_physical[:,jz, jM])
        ))
        num = r_array_here * Pe_rarray_here
        denom = jnp.sqrt(r_array_here ** 2 - rp ** 2)
        toint = num / denom
        val = 2. * jsi.trapezoid(toint * r_array_here, jnp.log(r_array_here))
        return self.const_coeff * val

    @partial(jit, static_argnums=(0,))        
    def get_y2D_smoothed_prof(self, jz, jM):
        zval = self.z_array[jz]
        DA_val = self.DA_array[jz]
        theta_array = (self.rp_array / DA_val)  # in radians 
        
        ell_out, yell_out = (Hankel(theta_array, nu=0, q=1.0, nx=len(theta_array), lowring=True)(
            self.y2D_mat_physical[:,jz, jM], extrap=True))
        yell_out = jnp.array(yell_out * ((2 * jnp.pi)))   
        b_ell = jnp.exp(-0.5 * ell_out * (ell_out + 1.) * (self.sigma_val ** 2))

        theta_out, y_out_smooth = (Hankel(ell_out, nu=0, q=1.0, nx=len(ell_out), lowring=True)(
            jnp.clip(b_ell * yell_out, 1e-40, 1e10), extrap=True))
        y_out_smooth = jnp.array(y_out_smooth * (1./(2 * jnp.pi)))   
        y_out_smooth = jnp.clip(y_out_smooth, 1e-20, jnp.max(self.y2D_mat_physical[jz, jM]))

        return y_out_smooth        

    @partial(jit, static_argnums=(0,))
    def get_Pe_healpix(self, jpix):
        prop_jpix = self.pix_prop_all[jpix]
        y_jpix = jnp.exp(self.log_y2D_interp(prop_jpix[0], prop_jpix[1], prop_jpix[2]))        
        return y_jpix

    @partial(jit, static_argnums=(0,))        
    def get_ne2D_phyical_proj(self, jrp, jz, jM, num_trapz_points=32):
        rp = self.rp_array[jrp]
        zval = self.z_array[jz]
        r_max = jnp.min(jnp.array([jnp.max(self.r_array), rp * 100.0]))
        r_array_here = jnp.exp(jnp.linspace(jnp.log(rp*1.01), jnp.log(r_max), num_trapz_points))
        
        ne_rarray_here = jnp.exp(jnp.interp(
            jnp.log(r_array_here), 
            jnp.log(self.r_array/(1 + zval)), 
            jnp.log(self.ne_mat_physical[:,jz, jM])
        ))
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
        ell_out, nell_out = (Hankel(theta_array, nu=0, q=1.0, nx=len(theta_array), lowring=True)(
            self.ne2D_mat_physical[:,jz, jM], extrap=True))
        nell_out = jnp.array(nell_out * ((2 * jnp.pi)))   
        b_ell = jnp.exp(-0.5 * ell_out * (ell_out + 1.) * (self.sigma_val ** 2))

        theta_out, n_out_smooth = (Hankel(ell_out, nu=0, q=1.0, nx=len(ell_out), lowring=True)(
            jnp.clip(b_ell * nell_out, 1e-40, 1e10), extrap=True))
        n_out_smooth = jnp.array(n_out_smooth * (1./(2 * jnp.pi)))   
        n_out_smooth = jnp.clip(n_out_smooth, 1e-20, jnp.max(self.ne2D_mat_physical[jz, jM]))

        return n_out_smooth 


    @partial(jit, static_argnums=(0,))        
    def get_rhom2D_phyical_proj(self, jrp, jz, jM, num_trapz_points=32):
        rp = self.rp_array[jrp]
        zval = self.z_array[jz]
        r_max = jnp.min(jnp.array([jnp.max(self.r_array), rp * 100.0]))
        r_array_here = jnp.exp(jnp.linspace(jnp.log(rp*1.01), jnp.log(r_max), num_trapz_points))
        
        ne_rarray_here = jnp.exp(jnp.interp(
            jnp.log(r_array_here), 
            jnp.log(self.r_array/(1 + zval)), 
            jnp.log(self.rho_dmb_mat_physical[:,jz, jM])
        ))
        num = r_array_here * ne_rarray_here
        denom = jnp.sqrt(r_array_here ** 2 - rp ** 2)
        toint = num / denom
        val = 2. * jsi.trapezoid(toint * r_array_here, jnp.log(r_array_here))
        return val
    
    @partial(jit, static_argnums=(0,))        
    def get_rhom2D_smoothed_prof(self, jz, jM):
        zval = self.z_array[jz]
        DA_val = self.DA_array[jz]
        theta_array = (self.rp_array / DA_val)  # in radians 
        ell_out, nell_out = (Hankel(theta_array, nu=0, q=1.0, nx=len(theta_array), lowring=True)(
            self.rhom2D_mat_physical[:,jz, jM], extrap=True))
        nell_out = jnp.array(nell_out * ((2 * jnp.pi)))   
        b_ell = jnp.exp(-0.5 * ell_out * (ell_out + 1.) * (self.sigma_val ** 2))

        theta_out, n_out_smooth = (Hankel(ell_out, nu=0, q=1.0, nx=len(ell_out), lowring=True)(
            jnp.clip(b_ell * nell_out, 1e-40, 1e10), extrap=True))
        n_out_smooth = jnp.array(n_out_smooth * (1./(2 * jnp.pi)))   
        n_out_smooth = jnp.clip(n_out_smooth, 1e-20, jnp.max(self.rhom2D_mat_physical[jz, jM]))

        return n_out_smooth 


    @partial(jit, static_argnums=(0,))
    def get_kSZ_healpix(self, jpix):
        prop_jpix = self.pix_prop_all[jpix]
        zval = prop_jpix[1]
        tau = self.tau_interp(zval)
        fac = jnp.exp(-tau)
        ksz_jpix = -1 * self.const_coeff_kSZ * fac * prop_jpix[3] * jnp.exp(
            self.log_ne2D_interp(prop_jpix[0], prop_jpix[1], prop_jpix[2]))        
        return ksz_jpix

    @partial(jit, static_argnums=(0,))
    def get_tau_healpix(self, jpix):
        prop_jpix = self.pix_prop_all[jpix]
        tau_jpix = self.const_coeff_tau * jnp.exp(
            self.log_ne2D_interp(prop_jpix[0], prop_jpix[1], prop_jpix[2]))        
        return tau_jpix

    @partial(jit, static_argnums=(0,))
    def get_rhom_healpix(self, jpix):
        prop_jpix = self.pix_prop_all[jpix].astype(jnp.float32)
        rhom_jpix = jnp.exp(
            self.log_rhom2D_interp(prop_jpix[0], prop_jpix[1], prop_jpix[2]))        
        return rhom_jpix

    # --- 2. Physics Models (HOD, R200c, etc.) ---
    @partial(jit, static_argnums=(0,))
    def get_hod_params(self, mass, z):
        """Get HOD parameters for M200c mass definition"""
        log_m = jnp.log10(mass)
        m_min = 12.0
        m1_prime = 13.5
        mean_ncen = 0.5 * (1 + jax.scipy.special.erf((log_m - m_min) / 0.2))
        mean_nsat = jnp.where(log_m > m_min, mean_ncen * ((mass / (10**m1_prime))**1.0), 0.0)
        return mean_ncen, mean_nsat

    @partial(jit, static_argnums=(0,))
    def mass_to_r200c(self, mass, z):
        """Convert M200c to R200c in comoving kpc/h"""
        H0 = 100.0  # km/s/Mpc in h units
        Om0 = 0.3
        OL0 = 0.7
        
        E_z = jnp.sqrt(Om0 * (1 + z)**3 + OL0)
        rho_crit_0 = 2.775e11 / 1e9  # Convert from M_sun/Mpc^3 to M_sun/kpc^3
        rho_crit_z = rho_crit_0 * E_z**2
        r200c_phys = (3 * mass / (4 * jnp.pi * 200 * rho_crit_z))**(1/3)
        r200c_comoving = r200c_phys * (1 + z)
        
        return r200c_comoving

    @partial(jit, static_argnums=(0,))
    def angular_diameter_distance(self, z):
        """Angular diameter distance in comoving Mpc/h"""
        H0 = 100.0  # km/s/Mpc in h units
        c = 299792.458  # km/s
        Om0 = 0.3
        
        def E(z_prime):
            return jnp.sqrt(Om0 * (1 + z_prime)**3 + (1 - Om0))
        
        z_arr = jnp.linspace(0, z, 100)
        integrand = 1.0 / E(z_arr)
        chi = jsi.trapezoid(integrand, z_arr)
        
        return (c / H0) * chi / (1 + z)

    @partial(jit, static_argnums=(0,))
    def cumulative_trapezoid_jax(self, y, x, axis=-1, initial=0.0):
        """Cumulative trapezoidal integration"""
        dx = jnp.diff(x)
        y_moved = jnp.moveaxis(y, axis, -1)
        y_left = y_moved[..., :-1]
        y_right = y_moved[..., 1:]
        areas = 0.5 * (y_left + y_right) * dx
        cum_areas = jnp.cumsum(areas, axis=-1)
        initial_shape = list(cum_areas.shape[:-1]) + [1]
        initial_array = jnp.full(initial_shape, initial)
        cum_integral = jnp.concatenate([initial_array, cum_areas], axis=-1)
        result = jnp.moveaxis(cum_integral, -1, axis)
        return result

    @partial(jit, static_argnums=(0,))
    def create_ppf_interpolator_3d(self, profile_table, mass_grid, z_grid, r_comoving_grid):
        """Create CDF table from profile in comoving coordinates"""
        # PDF is profile * r^2 (in spherical coordinates)
        pdf_table = profile_table * r_comoving_grid[:, None, None]**2
        pdf_table = jnp.swapaxes(pdf_table, 0, 2)  # Move radius axis to back for integration
        # print(pdf_table.shape, mass_grid.shape, z_grid.shape, r_comoving_grid.shape)
        # Integrate to get CDF
        # cdf_table_unnormalized = self.cumulative_trapezoid_jax(
        #     pdf_table, r_comoving_grid, axis=2, initial=0.0
        # )
        y = pdf_table
        x = r_comoving_grid
        axis = 2
        initial = 0.0
        dx = jnp.diff(x)
        y_moved = jnp.moveaxis(y, axis, -1)
        y_left = y_moved[..., :-1]
        y_right = y_moved[..., 1:]
        areas = 0.5 * (y_left + y_right) * dx
        cum_areas = jnp.cumsum(areas, axis=-1)
        initial_shape = list(cum_areas.shape[:-1]) + [1]
        initial_array = jnp.full(initial_shape, initial)
        cum_integral = jnp.concatenate([initial_array, cum_areas], axis=-1)
        cdf_table_unnormalized = jnp.moveaxis(cum_integral, -1, axis)



        norm = cdf_table_unnormalized[:, :, -1, jnp.newaxis]
        cdf_table = cdf_table_unnormalized / jnp.maximum(norm, 1e-9)
        
        return {
            "mass_grid": mass_grid,
            "z_grid": z_grid,
            "r_comoving_grid": r_comoving_grid,
            "cdf_table": cdf_table
        }

    @partial(jit, static_argnums=(0,2))
    def sample_radii_from_ppf_3d(self, key, num_samples, mass, z, mass_grid, z_grid, r_comoving_grid, cdf_table):
        """Sample comoving radii by interpolating the pre-computed 3D PPF table."""
        u = uniform(key, (num_samples,))
        
        log_mass_grid = jnp.log10(mass_grid)
        query_log_mass = jnp.log10(mass)
        
        mass_idx = jnp.clip(
            jnp.interp(query_log_mass, log_mass_grid, jnp.arange(len(log_mass_grid))), 
            0, len(log_mass_grid) - 1
        )
        z_idx = jnp.clip(
            jnp.interp(z, z_grid, jnp.arange(len(z_grid))), 
            0, len(z_grid) - 1
        )

        radius_indices = jnp.arange(len(r_comoving_grid))
        coords = jnp.stack([
            jnp.full_like(radius_indices, mass_idx, dtype=jnp.float32),
            jnp.full_like(radius_indices, z_idx, dtype=jnp.float32),
            radius_indices.astype(jnp.float32)
        ])
        
        cdf_for_halo = map_coordinates(cdf_table, coords, order=1)
        
        vmap_interp = vmap(jnp.interp, in_axes=(0, None, None))
        r_samples_comoving = vmap_interp(u, cdf_for_halo, r_comoving_grid)
        
        return r_samples_comoving

    @partial(jit, static_argnums=(0, 7, 8))
    def place_satellites(self, key, r_sats_comoving, halo_ra, halo_dec, halo_z, halo_mass, num_sats, do_rsd=False):
        """Place satellites given their comoving radii."""
        r_sats_physical = r_sats_comoving / (1 + halo_z)
        
        key_pos, key_vel = split(key)
        phi = uniform(key_pos, (num_sats,), minval=0, maxval=2*jnp.pi)
        cos_theta = uniform(split(key_pos, 2)[1], (num_sats,), minval=-1, maxval=1)
        sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta**2, 0.0))
        
        dx = r_sats_physical * sin_theta * jnp.cos(phi)
        dy = r_sats_physical * sin_theta * jnp.sin(phi)
        
        dist_a_kpc = self.angular_diameter_distance(halo_z) * 1000
        
        rad_to_deg = 180.0 / jnp.pi
        d_ra_rad = dx / dist_a_kpc
        d_dec_rad = dy / dist_a_kpc
        
        cos_dec = jnp.maximum(jnp.cos(jnp.deg2rad(halo_dec)), 1e-6)
        sat_ra = halo_ra + (d_ra_rad * rad_to_deg) / cos_dec    
        sat_dec = halo_dec + d_dec_rad * rad_to_deg
        sat_ra = jnp.mod(sat_ra, 360.0)
        
        if not do_rsd:
            sat_z = jnp.full((num_sats,), halo_z)
        else:
            sigma_v = 106.2 * (halo_mass / 1e12)**(1/3)
            peculiar_velocity = normal(key_vel, (num_sats,)) * sigma_v
            c_light = 299792.458
            sat_z = halo_z + (peculiar_velocity / c_light) * (1 + halo_z)
        
        return sat_ra, sat_dec, sat_z

    # @partial(jit, static_argnums=(0, 5))
    def populate_one_halo(self, key, halo_ra, halo_dec, halo_z, halo_mass, max_gals, 
                          mass_grid, z_grid, r_comoving_grid, cdf_table):
        """Populate one halo with galaxies"""
        key_hod, key_sat_pos = split(key)
        
        # Sample number of galaxies
        mean_ncen, mean_nsat = self.get_hod_params(halo_mass, halo_z)
        ncen = jnp.clip(poisson(key_hod, mean_ncen), 0, 1)
        nsat = poisson(split(key_hod, 2)[1], mean_nsat)
        nsat = jnp.minimum(nsat, max_gals - 1)
        
        # Initialize galaxy catalog
        pad_value = -1.0
        gal_catalog = jnp.full((max_gals, 6), pad_value, dtype=jnp.float32)
        gal_catalog = gal_catalog.at[:, 5].set(0.0)
        
        # Place central
        central_valid = jnp.where(ncen > 0, 1.0, 0.0)
        gal_catalog = gal_catalog.at[0].set(
            jnp.array([halo_ra, halo_dec, halo_z, halo_mass, 1.0, central_valid])
        )
        
        # Sample satellite positions in comoving coordinates
        r_sats_comoving = self.sample_radii_from_ppf_3d(
            key_sat_pos, max_gals - 1, halo_mass, halo_z, 
            mass_grid, z_grid, r_comoving_grid, cdf_table
        )
        
        # Place satellites
        sats_ra, sats_dec, sats_z = self.place_satellites(
            split(key_sat_pos, 2)[1], r_sats_comoving, 
            halo_ra, halo_dec, halo_z, halo_mass, max_gals - 1
        )
        
        # Update catalog
        sat_indices = jnp.arange(1, max_gals)
        sat_mask = (sat_indices <= nsat) & (ncen > 0)
        gal_catalog = gal_catalog.at[1:, 0].set(jnp.where(sat_mask, sats_ra, pad_value))
        gal_catalog = gal_catalog.at[1:, 1].set(jnp.where(sat_mask, sats_dec, pad_value))
        gal_catalog = gal_catalog.at[1:, 2].set(jnp.where(sat_mask, sats_z, pad_value))
        gal_catalog = gal_catalog.at[1:, 3].set(jnp.where(sat_mask, halo_mass, pad_value))
        gal_catalog = gal_catalog.at[1:, 4].set(0.0)  # satellites
        gal_catalog = gal_catalog.at[1:, 5].set(jnp.where(sat_mask, 1.0, 0.0))
        
        return gal_catalog