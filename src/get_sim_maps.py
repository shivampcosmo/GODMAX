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
from jax.random import PRNGKey, split, poisson, uniform, normal, bernoulli
from functools import partial
from jax.scipy.ndimage import map_coordinates
key = PRNGKey(42)
key, subkey1, subkey2, subkey3, subkey4 = split(key, 5)


class setup_sim_map(Profiles):
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
            
        # --- Timing Setup ---
        self.profile_timing = mock_params_dict.get('profile_timing', False)
        if self.profile_timing:
            self.timing_results = {}
            total_start_time = time.perf_counter()

        # Common setup
        if self.profile_timing: start_time = time.perf_counter()
        self._setup_common(mock_params_dict)
        if self.profile_timing: self.timing_results['setup_common'] = time.perf_counter() - start_time
        # Process requested maps
        self._process_maps(mock_params_dict)

        # --- Print Timing Report ---
        if self.profile_timing:
            total_end_time = time.perf_counter()
            self.timing_results['total_execution'] = total_end_time - total_start_time
            print("\n--- JAX Map Generation Timing Profile ---")
            for step, duration in self.timing_results.items():
                print(f"{step:<35}: {duration:.4f} seconds")
            print("---------------------------------------\n")

    def _setup_common(self, mock_params_dict):
        """Common setup for all map types"""
        # Interpolators and arrays
        H_array = self.H0 * bkgrd.H(self.cosmo_jax, self.scale_fac_a_array)        
        self.H_array_interp = interpax.Interpolator1D(
            self.z_array.astype(jnp.float32), 
            H_array.astype(jnp.float32), 
            extrap=True
        )
        
        # self.rp_array = self.r_array[2:-2].astype(jnp.float32)
        # For projection calculations, we want to ensure we have enough resolution at small radii, so we create a log-spaced rp array that covers the same range as self.r_array but with more points at small radii. This is in physical units, so we don't divide by (1+z) here.
        self.rp_array = jnp.logspace(jnp.log10(self.r_array[2]), jnp.log10(self.r_array[-2]), num=len(self.r_array)-3).astype(jnp.float32)
        
        # Map parameters
        self.nside_map = mock_params_dict['nside']
        theta_fwhm_arcmin = hp.nside2resol(self.nside_map, arcmin=True)
        theta_fwhm_rad = (theta_fwhm_arcmin / 60.) * (jnp.pi / 180.)
        self.sigma_val = theta_fwhm_rad / jnp.sqrt(8. * jnp.log(2.))
                
        # Constants
        self._setup_constants()
        
        # Other parameters
        self.smooth_profiles = mock_params_dict.get('smooth_profiles', False)

    def _setup_constants(self):
        """Setup physical constants"""
        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        oneMpc = (((10 ** 6)) * (u.pc).to(u.m)) * (u.m)
        
        # Compton-y constant
        coeff = sigmat / (m_e * (c ** 2))
        self.const_coeff = (((coeff * oneMpc).to(((u.cm ** 3) / u.keV))).value)/(self.cosmo_params['H0']/100.)
        
        # kSZ constant  
        coeff_kSZ = sigmat/c
        self.const_coeff_kSZ = (((coeff_kSZ * oneMpc).to(((u.cm ** 3) / (u.km/u.s)))).value)/(self.cosmo_params['H0']/100.)
        
        # Tau constant
        coeff_tau = sigmat
        self.const_coeff_tau = (((coeff_tau * oneMpc).to(u.cm ** 3)).value)/(self.cosmo_params['H0']/100.)


    def _process_maps(self, mock_params_dict):
        """Process all requested map types"""
        get_ymap = mock_params_dict.get('get_ymap', False)
        get_kSZmap = mock_params_dict.get('get_kSZmap', False)
        get_taumap = mock_params_dict.get('get_taumap', False)
        get_kappamap = mock_params_dict.get('get_kappamap', False)   
        get_baryonified_map = mock_params_dict.get('get_baryonifiedmap', False)
        get_galmap = mock_params_dict.get('get_galmap', False)
        
        # Setup tau interpolator if needed
        if get_kSZmap or get_taumap:
            if self.profile_timing: start_time = time.perf_counter()
            self._setup_tau_interpolator()
            if self.profile_timing: self.timing_results['setup_tau_interpolator'] = time.perf_counter() - start_time
        
        # Process y map
        if get_ymap:
            self._setup_ymap()
        
        # Process electron density maps
        if get_kSZmap or get_taumap:
            self._setup_ne_maps(get_kSZmap, get_taumap)
        
        # Process kappa map
        if get_kappamap:
            self._setup_kappamap()

            if get_baryonified_map:
                self._setup_DMOmap()
        
        if get_galmap:
            self._setup_galmap()
        
    def _setup_tau_interpolator(self):
        """Setup tau(z) interpolator"""
        tauz_array = vmap(self.get_tau_z)(jnp.arange(len(self.z_array))).astype(jnp.float32)
        self.tau_interp = interpax.Interpolator1D(
            self.z_array.astype(jnp.float32), 
            tauz_array, 
            extrap=True
        )

    def _setup_ymap(self):
        """Setup Compton-y map"""
        if self.profile_timing: start_time = time.perf_counter()
        self.y2D_mat_physical = self._compute_projections(
            self.Pe_mat_physical, 'y2D'
        ).astype(jnp.float32)
        if self.profile_timing: self.timing_results['ymap_projection_calculation'] = time.perf_counter() - start_time
        
        if self.smooth_profiles:
            if self.profile_timing: start_time = time.perf_counter()
            self.y2D_mat_physical = self._apply_smoothing_to_profile(
                self.y2D_mat_physical, 'y2D'
            ).astype(jnp.float32)
            if self.profile_timing: self.timing_results['ymap_profile_smoothing'] = time.perf_counter() - start_time
        
        if self.profile_timing: start_time = time.perf_counter()
        self.log_y2D_interp = interpax.Interpolator3D(
            jnp.log(self.rp_array), self.z_array.astype(jnp.float32), 
            jnp.log(self.M_array).astype(jnp.float32), jnp.nan_to_num(jnp.log(self.y2D_mat_physical), nan=-20, posinf=-20, neginf=-20), 
            extrap=[-20, -20]
        )
        if self.profile_timing: self.timing_results['ymap_interpolator_creation'] = time.perf_counter() - start_time
        
    def _setup_ne_maps(self, get_kSZmap, get_taumap):
        """Setup electron density based maps"""
        if self.profile_timing: start_time = time.perf_counter()
        self.ne2D_mat_physical = self._compute_projections(
            self.ne_mat_physical, 'ne2D'
        ).astype(jnp.float32)
        if self.profile_timing: self.timing_results['ne_map_projection_calculation'] = time.perf_counter() - start_time
        
        if self.smooth_profiles:
            if self.profile_timing: start_time = time.perf_counter()
            self.ne2D_mat_physical = self._apply_smoothing_to_profile(
                self.ne2D_mat_physical, 'ne2D'
            ).astype(jnp.float32)
            if self.profile_timing: self.timing_results['ne_map_profile_smoothing'] = time.perf_counter() - start_time
        
        if self.profile_timing: start_time = time.perf_counter()
        self.log_ne2D_interp = interpax.Interpolator3D(
            jnp.log(self.rp_array), self.z_array.astype(jnp.float32), 
            jnp.log(self.M_array).astype(jnp.float32), jnp.nan_to_num(jnp.log(self.ne2D_mat_physical), nan=-20, posinf=-20, neginf=-20), 
            extrap=[-20, -20]
        )
        if self.profile_timing: self.timing_results['ne_map_interpolator_creation'] = time.perf_counter() - start_time
        
    def _setup_kappamap(self):
        """Setup kappa (lensing) map"""
        self.rho_dmb_mat_physical = (self.rho_dmb_mat / (self.scale_fac_a_array[None, :, None] ** 3)).astype(jnp.float32)
                
        if self.profile_timing: start_time = time.perf_counter()
        self.rhom2D_mat_physical_orig = self._compute_projections(
            self.rho_dmb_mat_physical, 'rhom2D'
        ).astype(jnp.float32)
        if self.profile_timing: self.timing_results['kappa_map_projection_calculation'] = time.perf_counter() - start_time
        
        if self.smooth_profiles:
            if self.profile_timing: start_time = time.perf_counter()
            self.rhom2D_mat_physical = self._apply_smoothing_to_profile(
                self.rhom2D_mat_physical_orig, 'rhom2D'
            ).astype(jnp.float32)
            if self.profile_timing: self.timing_results['kappa_map_profile_smoothing'] = time.perf_counter() - start_time
        else:
            self.rhom2D_mat_physical = self.rhom2D_mat_physical_orig
        if self.profile_timing: start_time = time.perf_counter()
        self.log_rhom2D_interp = interpax.Interpolator3D(
            jnp.log(self.rp_array), self.z_array.astype(jnp.float32), 
            jnp.log(self.M_array).astype(jnp.float32), jnp.nan_to_num(jnp.log(self.rhom2D_mat_physical), nan=-20, posinf=-20, neginf=-20), 
            extrap=[-20, -20]
        )
        if self.profile_timing: self.timing_results['kappa_map_interpolator_creation'] = time.perf_counter() - start_time

    def _setup_DMOmap(self):
        """Setup kappa (lensing) map"""
        self.rho_dmo_mat_physical = (self.rho_nfw_mat / (self.scale_fac_a_array[None, :, None] ** 3)).astype(jnp.float32)
                
        if self.profile_timing: start_time = time.perf_counter()
        self.rhom_dmo_2D_mat_physical_orig = self._compute_projections(
            self.rho_dmo_mat_physical, 'rhom2D_dmo'
        ).astype(jnp.float32)
        if self.profile_timing: self.timing_results['kappa_map_projection_calculation'] = time.perf_counter() - start_time
        
        if self.smooth_profiles:
            if self.profile_timing: start_time = time.perf_counter()
            self.rhom_dmo_2D_mat_physical = self._apply_smoothing_to_profile(
                self.rhom_dmo_2D_mat_physical_orig, 'rhom2D_dmo'
            ).astype(jnp.float32)
            if self.profile_timing: self.timing_results['kappa_map_profile_smoothing'] = time.perf_counter() - start_time
        else:
            self.rhom_dmo_2D_mat_physical = self.rhom_dmo_2D_mat_physical_orig
        if self.profile_timing: start_time = time.perf_counter()
        self.log_rhom_dmo_2D_interp = interpax.Interpolator3D(
            jnp.log(self.rp_array), self.z_array.astype(jnp.float32), 
            jnp.log(self.M_array).astype(jnp.float32), jnp.nan_to_num(jnp.log(self.rhom_dmo_2D_mat_physical), nan=-20, posinf=-20, neginf=-20), 
            extrap=[-20, -20]
        )
        if self.profile_timing: self.timing_results['kappa_map_interpolator_creation'] = time.perf_counter() - start_time

    def _setup_galmap(self):
        """Setup Ncen, Nsat interpolator.

        Uses monotonic interpolation to prevent cubic-spline overshoot at
        the sharp nbar(z) boundary, which otherwise creates unphysical
        pile-ups in the galaxy redshift distribution.
        """
        self.logNcen_interp = interpax.Interpolator2D(
            self.z_array.astype(jnp.float32),
            jnp.log(self.M_array).astype(jnp.float32),
            jnp.log(self.Ncen_mat + 1e-20).astype(jnp.float32),
            method='monotonic',
            extrap=[-20, -20]
        )
        self.logNsat_interp = interpax.Interpolator2D(
            self.z_array.astype(jnp.float32),
            jnp.log(self.M_array).astype(jnp.float32),
            jnp.log(self.Nsat_mat + 1e-20).astype(jnp.float32),
            method='monotonic',
            extrap=[-20, -20]
        )

    def _compute_projections(self, mat_physical, profile_name):
        """Compute 2D projections for a given physical matrix"""
        if profile_name == 'y2D':
            return get_vmapped_func(self.get_y2D_physical_proj, 3)(
                jnp.arange(len(self.rp_array)), 
                jnp.arange(len(self.z_array)), 
                jnp.arange(len(self.M_array))
            ).T
        elif profile_name == 'ne2D':
            return get_vmapped_func(self.get_ne2D_physical_proj, 3)(
                jnp.arange(len(self.rp_array)), 
                jnp.arange(len(self.z_array)), 
                jnp.arange(len(self.M_array))
            ).T
        elif profile_name == 'rhom2D':
            return get_vmapped_func(self.get_rhom2D_physical_proj, 3)(
                jnp.arange(len(self.rp_array)), 
                jnp.arange(len(self.z_array)), 
                jnp.arange(len(self.M_array))
            ).T
        elif profile_name == 'rhom2D_dmo':
            return get_vmapped_func(self.get_rhom2D_dmo_physical_proj, 3)(
                jnp.arange(len(self.rp_array)), 
                jnp.arange(len(self.z_array)), 
                jnp.arange(len(self.M_array))
            ).T

    def _apply_smoothing_to_profile(self, proj_2d, profile_name):
        """Apply smoothing to 2D profile"""
        if profile_name == 'y2D':
            return get_vmapped_func(self.get_y2D_smoothed_prof, 2)(
                jnp.arange(len(self.z_array)), 
                jnp.arange(len(self.M_array))
            ).T
        elif profile_name == 'ne2D':
            return get_vmapped_func(self.get_ne2D_smoothed_prof, 2)(
                jnp.arange(len(self.z_array)), 
                jnp.arange(len(self.M_array))
            ).T
        elif profile_name == 'rhom2D':
            return get_vmapped_func(self.get_rhom2D_smoothed_prof, 2)(
                jnp.arange(len(self.z_array)), 
                jnp.arange(len(self.M_array))
            ).T
        elif profile_name == 'rhom2D_dmo':
            return get_vmapped_func(self.get_rhom2D_dmo_smoothed_prof, 2)(
                jnp.arange(len(self.z_array)), 
                jnp.arange(len(self.M_array))
            ).T
    
    @partial(jit, static_argnums=(0,))        
    def get_tau_z(self, jz):
        """Compute tau(z) for given redshift index"""
        z = self.z_array[jz]
        z_array = jnp.linspace(0.001, z, 100)
        H_array = self.H_array_interp(z_array)
        to_int = (1 + z_array)**2 / H_array
        tau_z = jsi.trapezoid(to_int, z_array)
        
        sigma_T = const.sigma_T.to(u.Mpc**2).value
        c = const.c.to(u.km/u.s).value
        ne_bar = self.cosmo_params['Ob0'] * RHO_CRIT_0_MPC3 / (mue * mp)
        return sigma_T * ne_bar * c * tau_z

    @partial(jit, static_argnums=(0, 6))
    def _generic_2D_projection(self, jrp, jz, jM, mat_physical, const_factor=1.0, num_trapz_points=32):
        """Generic 2D projection helper"""
        zval = self.z_array[jz]
        # rp = self.rp_array[jrp]/(1 + zval)
        rp = self.rp_array[jrp]
        r_max = jnp.minimum(jnp.max(self.r_array), rp * 100.0)
        r_array_here = jnp.exp(jnp.linspace(jnp.log(rp*1.01), jnp.log(r_max), num_trapz_points))
        
        quantity_rarray = jnp.exp(jnp.interp(
            jnp.log(r_array_here), 
            jnp.log(self.r_array/(1 + zval)), 
            jnp.log(mat_physical[:,jz, jM])
        ))
        
        integrand = r_array_here * quantity_rarray / jnp.sqrt(r_array_here**2 - rp**2)
        return const_factor * 2.0 * jsi.trapezoid(integrand * r_array_here, jnp.log(r_array_here))

    @partial(jit, static_argnums=(0, 4))        
    def get_y2D_physical_proj(self, jrp, jz, jM, num_trapz_points=32):
        """Compute y2D projection"""
        return self._generic_2D_projection(jrp, jz, jM, self.Pe_mat_physical, self.const_coeff, num_trapz_points)

    @partial(jit, static_argnums=(0, 4))        
    def get_ne2D_physical_proj(self, jrp, jz, jM, num_trapz_points=32):
        """Compute ne2D projection"""
        return self._generic_2D_projection(jrp, jz, jM, self.ne_mat_physical, 1.0, num_trapz_points)

    @partial(jit, static_argnums=(0, 4))        
    def get_rhom2D_physical_proj(self, jrp, jz, jM, num_trapz_points=32):
        """Compute rhom2D projection"""
        return self._generic_2D_projection(jrp, jz, jM, self.rho_dmb_mat_physical, 1.0, num_trapz_points)

    @partial(jit, static_argnums=(0, 4))        
    def get_rhom2D_dmo_physical_proj(self, jrp, jz, jM, num_trapz_points=32):
        """Compute rhom2D projection"""
        return self._generic_2D_projection(jrp, jz, jM, self.rho_dmo_mat_physical, 1.0, num_trapz_points)


    # @partial(jit, static_argnums=(0,))
    # def _generic_smoothing(self, jz, jM, proj_2d_mat):
    #     """Generic smoothing helper"""
    #     DA_val = self.DA_array[jz]
    #     theta_array = self.rp_array / DA_val
        
    #     ell_out, prof_ell = Hankel(theta_array, nu=0, q=1.0, nx=len(theta_array), lowring=True)(
    #         proj_2d_mat[:,jz, jM], extrap=True
    #     )
    #     prof_ell = prof_ell * (2 * jnp.pi)
        
    #     b_ell = jnp.exp(-0.5 * ell_out * (ell_out + 1.) * (self.sigma_val ** 2))
        
    #     theta_out, prof_smooth = Hankel(ell_out, nu=0, q=1.0, nx=len(ell_out), lowring=True)(
    #         jnp.clip(b_ell * prof_ell, 1e-40, 1e10), extrap=True
    #     )
    #     prof_smooth = prof_smooth / (2 * jnp.pi)
        
    #     return jnp.clip(prof_smooth, 1e-20, jnp.max(proj_2d_mat[:,jz, jM]))

    @partial(jit, static_argnums=(0,))
    def _generic_smoothing(self, jz, jM, proj_2d_mat):
        """Generic smoothing helper"""
        DA_val = self.DA_array[jz]
        theta_array = self.rp_array / DA_val
        # theta_array = self.rp_array / (DA_val * (1 + self.z_array[jz]))

        theta_array_here = jnp.logspace(-6, jnp.log10(jnp.pi/8), num=100)
        prof_here = jnp.exp(jnp.interp(
            jnp.log(theta_array_here), 
            jnp.log(theta_array), 
            jnp.log(proj_2d_mat[:,jz, jM])
        ))

        ell_out, prof_ell = Hankel(theta_array_here, nu=0, q=0.95, nx=len(theta_array_here), lowring=True)(
            prof_here, extrap=True
        )
        prof_ell = prof_ell * (2 * jnp.pi)
        
        b_ell = jnp.exp(-0.5 * ell_out * (ell_out + 1.) * (self.sigma_val ** 2))
        
        theta_out, prof_smooth = Hankel(ell_out, nu=0, q=0.95, nx=len(ell_out), lowring=True)(
            jnp.clip(b_ell * prof_ell, 1e-40, 1e40), extrap=True
        )
        prof_smooth = prof_smooth / (2 * jnp.pi)

        prof_smooth_interp = jnp.exp(jnp.interp(
            jnp.log(theta_array), 
            jnp.log(theta_out), 
            jnp.log(prof_smooth)
        ))

        return jnp.clip(prof_smooth_interp,  jnp.min(proj_2d_mat[:,jz, jM]), jnp.max(proj_2d_mat[:,jz, jM]))

    @partial(jit, static_argnums=(0,))        
    def get_y2D_smoothed_prof(self, jz, jM):
        """Apply smoothing to y2D profile"""
        return self._generic_smoothing(jz, jM, self.y2D_mat_physical)

    @partial(jit, static_argnums=(0,))        
    def get_ne2D_smoothed_prof(self, jz, jM):
        """Apply smoothing to ne2D profile"""
        return self._generic_smoothing(jz, jM, self.ne2D_mat_physical)

    @partial(jit, static_argnums=(0,))        
    def get_rhom2D_smoothed_prof(self, jz, jM):
        """Apply smoothing to rhom2D profile"""
        return self._generic_smoothing(jz, jM, self.rhom2D_mat_physical_orig)

    @partial(jit, static_argnums=(0,))
    def get_rhom2D_dmo_smoothed_prof(self, jz, jM):
        """Apply smoothing to rhom2D DMO profile"""
        return self._generic_smoothing(jz, jM, self.rhom_dmo_2D_mat_physical_orig)



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
            
        # --- Timing Setup ---
        self.profile_timing = mock_params_dict.get('profile_timing', False)
        if self.profile_timing:
            self.timing_results = {}
            total_start_time = time.perf_counter()

        # Common setup
        if self.profile_timing: start_time = time.perf_counter()
        self._setup_common(mock_params_dict)
        if self.profile_timing: self.timing_results['setup_common'] = time.perf_counter() - start_time
        
        # Process requested maps
        self._process_maps(mock_params_dict)

        # --- Print Timing Report ---
        if self.profile_timing:
            total_end_time = time.perf_counter()
            self.timing_results['total_execution'] = total_end_time - total_start_time
            print("\n--- JAX Map Generation Timing Profile ---")
            for step, duration in self.timing_results.items():
                print(f"{step:<35}: {duration:.4f} seconds")
            print("---------------------------------------\n")


    def _setup_common(self, mock_params_dict):
        """Common setup for all map types"""
        # Interpolators and arrays
        H_array = self.H0 * bkgrd.H(self.cosmo_jax, self.scale_fac_a_array)        
        self.H_array_interp = interpax.Interpolator1D(
            self.z_array.astype(jnp.float32), 
            H_array.astype(jnp.float32), 
            extrap=True
        )
        
        # self.rp_array = self.r_array[2:-2].astype(jnp.float32)
        
        # Map parameters
        self.nside_map = mock_params_dict['nside']
        theta_fwhm_arcmin = hp.nside2resol(self.nside_map, arcmin=True)
        theta_fwhm_rad = (theta_fwhm_arcmin / 60.) * (jnp.pi / 180.)
        self.sigma_val = theta_fwhm_rad / jnp.sqrt(8. * jnp.log(2.))
        self.pix_area = hp.nside2pixarea(self.nside_map, degrees=False)
        
        # Pixel arrays
        self.nearby_pix_all = jnp.array(mock_params_dict['nearby_pix_all'], dtype=jnp.int32)
        self.pix_prop_all = jnp.array(mock_params_dict['pix_prop_all'], dtype=jnp.float32)
        
        # Pre-compute sorting and grouping info
        self.pix_unique, self.sort_idx, self.boundaries = self._precompute_pixel_grouping()
        
        # Constants
        self._setup_constants()
        
        # Other parameters
        self.smooth_profiles = mock_params_dict.get('smooth_profiles', False)

    def _setup_constants(self):
        """Setup physical constants"""
        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        oneMpc = (((10 ** 6)) * (u.pc).to(u.m)) * (u.m)
        
        # Compton-y constant
        coeff = sigmat / (m_e * (c ** 2))
        self.const_coeff = (((coeff * oneMpc).to(((u.cm ** 3) / u.keV))).value)/(self.cosmo_params['H0']/100.)
        
        # kSZ constant  
        coeff_kSZ = sigmat/c
        self.const_coeff_kSZ = (((coeff_kSZ * oneMpc).to(((u.cm ** 3) / (u.km/u.s)))).value)/(self.cosmo_params['H0']/100.)
        
        # Tau constant
        coeff_tau = sigmat
        self.const_coeff_tau = (((coeff_tau * oneMpc).to(u.cm ** 3)).value)/(self.cosmo_params['H0']/100.)

    def _precompute_pixel_grouping(self):
        """Pre-compute pixel sorting and grouping"""
        pix_unique = np.unique(self.nearby_pix_all)
        sort_idx = np.argsort(self.nearby_pix_all)
        sorted_pix = self.nearby_pix_all[sort_idx]
        
        change_points = np.diff(sorted_pix, prepend=sorted_pix[0]-1, append=sorted_pix[-1]+1) != 0 
        boundaries = np.where(change_points)[0]
        
        return pix_unique, sort_idx, boundaries

    def _process_maps(self, mock_params_dict):
        """Process all requested map types"""
        get_ymap = mock_params_dict.get('get_ymap', False)
        get_kSZmap = mock_params_dict.get('get_kSZmap', False)
        get_taumap = mock_params_dict.get('get_taumap', False)
        get_kappamap = mock_params_dict.get('get_kappamap', False)   
        get_baryonified_map = mock_params_dict.get('get_baryonifiedmap', False)
        get_galmap = mock_params_dict.get('get_galmap', False)

        
        # Setup tau interpolator if needed
        if get_kSZmap or get_taumap:
            if self.profile_timing: start_time = time.perf_counter()
            self._setup_tau_interpolator()
            if self.profile_timing: self.timing_results['setup_tau_interpolator'] = time.perf_counter() - start_time
        
        # Process y map
        if get_ymap:
            self._get_ymap()
        
        # Process electron density maps
        if get_kSZmap or get_taumap:
            self._get_ne_maps(get_kSZmap, get_taumap)
        
        # Process kappa map
        if get_kappamap:
            self._get_kappamap()

        if get_baryonified_map:
            self._get_kappamap_dmo()
        
        # Process galaxy catalog
        if get_galmap:
            self._setup_galaxy_catalog(mock_params_dict)

    @partial(jit, static_argnums=(0,))        
    def get_tau_z(self, jz):
        """Compute tau(z) for given redshift index"""
        z = self.z_array[jz]
        z_array = jnp.linspace(0.001, z, 100)
        H_array = self.H_array_interp(z_array)
        to_int = (1 + z_array)**2 / H_array
        tau_z = jsi.trapezoid(to_int, z_array)
        
        sigma_T = const.sigma_T.to(u.Mpc**2).value
        c = const.c.to(u.km/u.s).value
        ne_bar = self.cosmo_params['Ob0'] * RHO_CRIT_0_MPC3 / (mue * mp)
        return sigma_T * ne_bar * c * tau_z

    def _setup_tau_interpolator(self):
        """Setup tau(z) interpolator"""
        tauz_array = vmap(self.get_tau_z)(jnp.arange(len(self.z_array))).astype(jnp.float32)
        self.tau_interp = interpax.Interpolator1D(
            self.z_array.astype(jnp.float32), 
            tauz_array, 
            extrap=True
        )

    def _chunked_vmap(self, func, n_total, chunk_size=20_000_000):
        """Apply a vmapped function in chunks to bound GPU memory usage."""
        result = np.empty(n_total, dtype=np.float32)
        jitted_func = jit(vmap(func))
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk_out = jitted_func(jnp.arange(start, end))
            result[start:end] = np.asarray(chunk_out)
            del chunk_out
        return result

    def _get_ymap(self):
        """Get Compton-y map"""
        if self.profile_timing: start_time = time.perf_counter()
        yjpix_all = self._chunked_vmap(self.get_y_healpix, len(self.pix_prop_all))
        self.ymap_final = self._assemble_map(yjpix_all)
        if self.profile_timing: self.timing_results['ymap_generation_and_assembly'] = time.perf_counter() - start_time

    def _get_ne_maps(self, get_kSZmap, get_taumap):
        """Get electron density maps"""
        if get_kSZmap:
            if self.profile_timing: start_time = time.perf_counter()
            kszjpix_all = self._chunked_vmap(self.get_kSZ_healpix, len(self.pix_prop_all))
            self.kszmap_final = self._assemble_map(kszjpix_all)
            if self.profile_timing: self.timing_results['ksz_map_generation_and_assembly'] = time.perf_counter() - start_time

        if get_taumap:
            if self.profile_timing: start_time = time.perf_counter()
            taujpix_all = self._chunked_vmap(self.get_tau_healpix, len(self.pix_prop_all))
            self.taumap_final = self._assemble_map(taujpix_all)
            if self.profile_timing: self.timing_results['tau_map_generation_and_assembly'] = time.perf_counter() - start_time

    def _get_kappamap(self):
        """Get kappa (lensing) map"""
        if self.profile_timing: start_time = time.perf_counter()
        rhomjpix_all = self._chunked_vmap(self.get_rhom_healpix, len(self.pix_prop_all))
        self.rhommap_final = self._assemble_map(rhomjpix_all)
        if self.profile_timing: self.timing_results['kappa_map_generation_and_assembly'] = time.perf_counter() - start_time

    def _get_kappamap_dmo(self):
        """Get kappa (lensing) map"""
        if self.profile_timing: start_time = time.perf_counter()
        rhom_dmo_jpix_all = self._chunked_vmap(self.get_rhom_dmo_healpix, len(self.pix_prop_all))
        self.rhom_dmo_map_final = self._assemble_map(rhom_dmo_jpix_all)
        if self.profile_timing: self.timing_results['kappa_dmo_map_generation_and_assembly'] = time.perf_counter() - start_time


    def _assemble_map(self, pixel_values):
        """Efficiently assemble final map from pixel values"""
        sorted_values = np.asarray(pixel_values)[self.sort_idx]
        summed_values = np.add.reduceat(sorted_values, self.boundaries[:-1])
        
        final_map = np.zeros(12 * self.nside_map**2, dtype=np.float32)
        final_map[self.pix_unique] = summed_values.astype(np.float32)
        return final_map

    def _setup_galaxy_catalog(self, mock_params_dict):
        """Setup galaxy catalog generation.

        Set mock_params_dict['use_poisson_centrals'] = True to use the old
        Poisson-clip central sampling for A/B comparison diagnostics.
        """
        self._use_poisson_centrals = mock_params_dict.get('use_poisson_centrals', False)
        if self.profile_timing: start_time = time.perf_counter()
        self.mass_grid = self.M_array.astype(jnp.float64)
        self.z_grid = self.z_array.astype(jnp.float32)
        self.r_comoving_grid = (self.r_array * 1000).astype(jnp.float32)
        
        mock_rho_table_comoving = self.rho_clm_mat
        mock_rho_table = mock_rho_table_comoving / (self.scale_fac_a_array[None, :, None] ** 3)
        
        self.ppf_table_3d = self.create_ppf_interpolator_3d(
            mock_rho_table, self.mass_grid, self.z_grid, self.r_comoving_grid
        )
        if self.profile_timing: self.timing_results['galaxy_ppf_interpolator_creation'] = time.perf_counter() - start_time
        
        if self.profile_timing: start_time = time.perf_counter()
        mean_ncen_all, mean_nsat_all = self.Ncen_mat, self.Nsat_mat
        max_mean_nsat = jnp.max(mean_nsat_all)
        max_gals_per_halo = int(jnp.ceil(max_mean_nsat + jnp.sqrt(max_mean_nsat))) + 2
        
        NUM_HALOS = len(mock_params_dict['halo_ra'])
        key = PRNGKey(mock_params_dict.get('random_seed', 42))
        
        def populate_wrapper(key, ra, dec, z, mass):
            return self.populate_one_halo(
                key, ra, dec, z, mass, max_gals_per_halo,
                self.ppf_table_3d["mass_grid"], self.ppf_table_3d["z_grid"],
                self.ppf_table_3d["r_comoving_grid"], self.ppf_table_3d["cdf_table"]
            )
        
        jitted_vectorized_populate = jit(vmap(populate_wrapper))
        keys = split(key, NUM_HALOS)
        
        padded_galaxy_catalog = jitted_vectorized_populate(
            keys,
            jnp.array(mock_params_dict['halo_ra'], dtype=jnp.float32),
            jnp.array(mock_params_dict['halo_dec'], dtype=jnp.float32),
            jnp.array(mock_params_dict['halo_z'], dtype=jnp.float32),
            jnp.array(mock_params_dict['halo_M'], dtype=jnp.float64)
        )
        padded_galaxy_catalog.block_until_ready()
        if self.profile_timing: self.timing_results['galaxy_population'] = time.perf_counter() - start_time
        
        if self.profile_timing: start_time = time.perf_counter()
        flat_padded_catalog = padded_galaxy_catalog.reshape(-1, 6)
        valid_mask = flat_padded_catalog[:, 5] > 0.5
        self.final_galaxy_catalog = flat_padded_catalog[valid_mask]
        if self.profile_timing: self.timing_results['galaxy_catalog_filtering'] = time.perf_counter() - start_time


    # ========== HEALPix pixel value methods ==========
    
    @partial(jit, static_argnums=(0,))
    def get_y_healpix(self, jpix):
        """Get y value for HEALPix pixel"""
        prop = self.pix_prop_all[jpix]
        return jnp.exp(self.log_y2D_interp(prop[0], prop[1], prop[2]))

    @partial(jit, static_argnums=(0,))
    def get_kSZ_healpix(self, jpix):
        """Get kSZ value for HEALPix pixel"""
        prop = self.pix_prop_all[jpix]
        tau = self.tau_interp(prop[1])
        fac = jnp.exp(-tau)
        return -self.const_coeff_kSZ * fac * prop[3] * jnp.exp(
            self.log_ne2D_interp(prop[0], prop[1], prop[2]))

    @partial(jit, static_argnums=(0,))
    def get_tau_healpix(self, jpix):
        """Get tau value for HEALPix pixel"""
        prop = self.pix_prop_all[jpix]
        return self.const_coeff_tau * jnp.exp(
            self.log_ne2D_interp(prop[0], prop[1], prop[2]))

    @partial(jit, static_argnums=(0,))
    def get_rhom_healpix(self, jpix):
        """Get rhom value for HEALPix pixel"""
        prop = self.pix_prop_all[jpix]
        DA_val = jnp.exp(jnp.interp(prop[1], self.z_array, jnp.log(self.DA_array)))
        pix_area_corr = self.pix_area * (DA_val**2)
        return pix_area_corr * jnp.exp(self.log_rhom2D_interp(prop[0], prop[1], prop[2]))

    @partial(jit, static_argnums=(0,))
    def get_rhom_dmo_healpix(self, jpix):
        """Get rhom value for HEALPix pixel"""
        prop = self.pix_prop_all[jpix]
        DA_val = jnp.exp(jnp.interp(prop[1], self.z_array, jnp.log(self.DA_array)))
        pix_area_corr = self.pix_area * (DA_val**2)
        return pix_area_corr * jnp.exp(self.log_rhom_dmo_2D_interp(prop[0], prop[1], prop[2]))


    # ========== Galaxy catalog methods ==========
    
    @partial(jit, static_argnums=(0,))
    def get_hod_params(self, mass, z):
        """Get HOD parameters for M200c mass definition"""
        mean_ncen = jnp.nan_to_num(jnp.exp(self.logNcen_interp(z, jnp.log(mass))))
        mean_nsat = jnp.nan_to_num(jnp.exp(self.logNsat_interp(z, jnp.log(mass))))
        # Clamp to physical range (Ncen ∈ [0,1] by construction of the erf HOD)
        mean_ncen = jnp.clip(mean_ncen, 0.0, 1.0)
        mean_nsat = jnp.clip(mean_nsat, 0.0)

        return mean_ncen, mean_nsat

    @partial(jit, static_argnums=(0,))
    def angular_diameter_distance(self, z):
        """Angular diameter distance in Mpc/h (physical, not comoving).

        Uses the simulation cosmology from self.cosmo_params.
        H0=100 gives distances in Mpc/h units by construction.
        """
        H0 = 100.0  # gives Mpc/h units
        c = 299792.458
        Om0 = self.cosmo_params['Om0']

        def E(z_prime):
            return jnp.sqrt(Om0 * (1 + z_prime)**3 + (1 - Om0))

        z_arr = jnp.linspace(0, z, 100)
        integrand = 1.0 / E(z_arr)
        chi = jsi.trapezoid(integrand, z_arr)

        return (c / H0) * chi / (1 + z)

    @partial(jit, static_argnums=(0,))
    def create_ppf_interpolator_3d(self, profile_table, mass_grid, z_grid, r_comoving_grid):
        """Create CDF table from profile in comoving coordinates"""
        # PDF is profile * r^2
        pdf_table = profile_table * r_comoving_grid[:, None, None]**2
        pdf_table = jnp.swapaxes(pdf_table, 0, 2)
        
        # Integrate to get CDF
        dx = jnp.diff(r_comoving_grid)
        y_moved = jnp.moveaxis(pdf_table, 2, -1)
        y_left = y_moved[..., :-1]
        y_right = y_moved[..., 1:]
        areas = 0.5 * (y_left + y_right) * dx
        cum_areas = jnp.cumsum(areas, axis=-1)
        initial_array = jnp.zeros(list(cum_areas.shape[:-1]) + [1])
        cdf_table_unnormalized = jnp.concatenate([initial_array, cum_areas], axis=-1)
        cdf_table_unnormalized = jnp.moveaxis(cdf_table_unnormalized, -1, 2)
        
        # Normalize
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

    def populate_one_halo(self, key, halo_ra, halo_dec, halo_z, halo_mass, max_gals,
                          mass_grid, z_grid, r_comoving_grid, cdf_table):
        """Populate one halo with galaxies.

        Central sampling uses Bernoulli(mean_ncen) to match the HOD theory
        convention where <Ncen(M)> = 0.5*(1 - erf(...)).
        The previous Poisson-clip approach gave <Ncen> = 1 - exp(-mean_ncen),
        which underestimates centrals by ~37% at mean_ncen ~ 1.

        Satellite sampling uses Poisson(mean_nsat) — unconditional on central,
        consistent with the 1-halo pair-counting formula
        2*Ncen*Nsat*uk + Nsat^2*uk^2 used in get_Pkzs.
        """
        key_hod, key_sat_pos = split(key)

        # Sample number of galaxies
        mean_ncen, mean_nsat = self.get_hod_params(halo_mass, halo_z)
        # Bernoulli draw for central (matches theory <Ncen> = p)
        # Set use_poisson_centrals=True in mock_params_dict for old (buggy) behavior
        ncen = jnp.where(
            getattr(self, '_use_poisson_centrals', False),
            jnp.clip(poisson(key_hod, mean_ncen), 0, 1),          # old: biased low
            bernoulli(key_hod, mean_ncen).astype(jnp.int32),       # new: correct
        )
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
        
        # Sample satellite positions
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
        # sat_mask = (sat_indices <= nsat) & (ncen > 0)
        sat_mask = (sat_indices <= nsat)
        gal_catalog = gal_catalog.at[1:, 0].set(jnp.where(sat_mask, sats_ra, pad_value))
        gal_catalog = gal_catalog.at[1:, 1].set(jnp.where(sat_mask, sats_dec, pad_value))
        gal_catalog = gal_catalog.at[1:, 2].set(jnp.where(sat_mask, sats_z, pad_value))
        gal_catalog = gal_catalog.at[1:, 3].set(jnp.where(sat_mask, halo_mass, pad_value))
        gal_catalog = gal_catalog.at[1:, 4].set(0.0)  # satellites
        gal_catalog = gal_catalog.at[1:, 5].set(jnp.where(sat_mask, 1.0, 0.0))
        
        return gal_catalog

