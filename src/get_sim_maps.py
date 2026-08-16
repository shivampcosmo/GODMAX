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


def _np_trapezoid(y, x=None, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


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
        self._profiles_obj_ref = Profiles_obj

        # The historical Abel projector mixes a physical transverse radius
        # with the comoving upper edge of ``r_array``.  Keep that path as the
        # compatibility default, but allow comparison/validation runs to opt
        # into a unit-consistent, nonsingular line-of-sight quadrature.
        self.projected_profile_integration_method = str(
            analysis_dict.get(
                'projected_profile_integration_method',
                'legacy_log_radius',
            )
        )
        supported_projection_methods = {
            'legacy_log_radius',
            'physical_table_cosh',
        }
        if self.projected_profile_integration_method not in supported_projection_methods:
            raise ValueError(
                'Unsupported projected_profile_integration_method='
                f'{self.projected_profile_integration_method!r}; expected one of '
                f'{sorted(supported_projection_methods)}.'
            )
        self.num_points_projected_profile = int(
            analysis_dict.get('num_points_projected_profile', 32)
        )
        if self.num_points_projected_profile < 2:
            raise ValueError(
                'num_points_projected_profile must be at least 2; got '
                f'{self.num_points_projected_profile}.'
            )
        los_max = analysis_dict.get(
            'projected_profile_los_max_comoving_mpc', None
        )
        self.projected_profile_los_max_comoving_mpc = (
            None if los_max is None else float(los_max)
        )
        if (
            self.projected_profile_los_max_comoving_mpc is not None
            and self.projected_profile_los_max_comoving_mpc <= 0.0
        ):
            raise ValueError(
                'projected_profile_los_max_comoving_mpc must be positive or '
                f'null; got {self.projected_profile_los_max_comoving_mpc}.'
            )
            
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
        self.use_fused_profile_maps = mock_params_dict.get('use_fused_profile_maps', True)
        self.return_sparse_maps = mock_params_dict.get('return_sparse_maps', False)
        self.store_projected_matter_maps = mock_params_dict.get('store_projected_matter_maps', True)

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
        get_multi_kappamap = mock_params_dict.get('get_multi_kappamap', False)
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
        if get_kappamap or get_multi_kappamap:
            self._setup_kappamap()

            if get_baryonified_map:
                self._setup_DMOmap()
        
        if get_galmap:
            self._setup_galmap()
        
    def _setup_tau_interpolator(self):
        """Setup tau(z) interpolator"""
        if hasattr(self, 'tau_interp'):
            return
        tauz_array = vmap(self.get_tau_z)(jnp.arange(len(self.z_array))).astype(jnp.float32)
        self.tau_interp = interpax.Interpolator1D(
            self.z_array.astype(jnp.float32), 
            tauz_array, 
            extrap=True
        )

    def _setup_ymap(self):
        """Setup Compton-y map"""
        if hasattr(self, 'log_y2D_interp'):
            return
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
        if hasattr(self, 'log_ne2D_interp'):
            return
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
        if hasattr(self, 'log_rhom2D_interp'):
            return
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
        if hasattr(self, 'log_rhom_dmo_2D_interp'):
            return
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
        """Setup Ncen, Nsat interpolators.

        N_cen is a sharp erf turn-on in [0, 1]; it is interpolated LINEARLY in
        its value (method='linear'). Interpolating log(N_cen) with the monotone
        scheme on the coarse mass grid systematically undershoots the central
        turn-on, suppressing the realized galaxy number density by ~13% (the
        centrals specifically by ~16%). Linear interpolation removes that
        undershoot and, being piecewise-linear, also avoids the cubic-spline
        overshoot at the sharp nbar(z) boundary that motivated the monotone
        scheme; the downstream clip to [0, 1] in get_hod_params bounds any
        extrapolation.

        N_sat is ~power-law in mass and is well represented by monotone
        interpolation of log(N_sat), which is retained.
        """
        self.Ncen_interp = interpax.Interpolator2D(
            self.z_array.astype(jnp.float32),
            jnp.log(self.M_array).astype(jnp.float32),
            self.Ncen_mat.astype(jnp.float32),
            method='linear',
            extrap=True,
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
    def _generic_2D_projection(self, jrp, jz, jM, mat_physical, const_factor=1.0, num_trapz_points=None):
        """Project a physical 3D table at a physical transverse radius.

        ``legacy_log_radius`` reproduces the historical implementation.
        ``physical_table_cosh`` uses a physical line-of-sight limit bounded by
        the comoving 3D table and the substitution ``r = rp cosh(t)``.  The
        latter removes the Abel singularity without dropping the interval
        immediately above ``rp`` and never extrapolates the terminal value.
        """
        zval = self.z_array[jz]
        rp = self.rp_array[jrp]
        n_points = (
            self.num_points_projected_profile
            if num_trapz_points is None
            else int(num_trapz_points)
        )

        if self.projected_profile_integration_method == 'physical_table_cosh':
            table_r_max = jnp.max(self.r_array) / (1.0 + zval)
            has_support = rp < table_r_max
            table_los_max = jnp.sqrt(
                jnp.maximum(table_r_max**2 - rp**2, 0.0)
            )
            los_cutoff = getattr(
                self, 'projected_profile_los_max_comoving_mpc', None
            )
            if los_cutoff is None:
                los_max = table_los_max
            else:
                requested_los_max = (
                    los_cutoff
                    * self.h
                    / (1.0 + zval)
                )
                los_max = jnp.minimum(requested_los_max, table_los_max)
            t_max = jnp.arcsinh(los_max / rp)
            nodes, weights = np.polynomial.legendre.leggauss(n_points)
            nodes = jnp.asarray(nodes)
            weights = jnp.asarray(weights)
            t_array = 0.5 * (nodes + 1.0) * t_max
            r_array_here = rp * jnp.cosh(t_array)
            quantity_rarray = jnp.exp(jnp.interp(
                jnp.log(r_array_here),
                jnp.log(self.r_array / (1.0 + zval)),
                jnp.log(mat_physical[:, jz, jM]),
            ))
            projected = t_max * jnp.sum(
                weights * r_array_here * quantity_rarray
            )
            return jnp.where(has_support, const_factor * projected, 0.0)

        # Historical compatibility path.  Here ``r_max`` is comoving while
        # ``rp`` and the interpolation coordinates are physical; interpolation
        # therefore clamps the endpoint above the physical table support.
        r_max = jnp.minimum(jnp.max(self.r_array), rp * 100.0)
        r_array_here = jnp.exp(jnp.linspace(jnp.log(rp*1.01), jnp.log(r_max), n_points))
        
        quantity_rarray = jnp.exp(jnp.interp(
            jnp.log(r_array_here), 
            jnp.log(self.r_array/(1 + zval)), 
            jnp.log(mat_physical[:,jz, jM])
        ))
        
        integrand = r_array_here * quantity_rarray / jnp.sqrt(r_array_here**2 - rp**2)
        return const_factor * 2.0 * jsi.trapezoid(integrand * r_array_here, jnp.log(r_array_here))

    @partial(jit, static_argnums=(0, 4))        
    def get_y2D_physical_proj(self, jrp, jz, jM, num_trapz_points=None):
        """Compute y2D projection"""
        return self._generic_2D_projection(jrp, jz, jM, self.Pe_mat_physical, self.const_coeff, num_trapz_points)

    @partial(jit, static_argnums=(0, 4))        
    def get_ne2D_physical_proj(self, jrp, jz, jM, num_trapz_points=None):
        """Compute ne2D projection"""
        return self._generic_2D_projection(jrp, jz, jM, self.ne_mat_physical, 1.0, num_trapz_points)

    @partial(jit, static_argnums=(0, 4))        
    def get_rhom2D_physical_proj(self, jrp, jz, jM, num_trapz_points=None):
        """Compute rhom2D projection"""
        return self._generic_2D_projection(jrp, jz, jM, self.rho_dmb_mat_physical, 1.0, num_trapz_points)

    @partial(jit, static_argnums=(0, 4))        
    def get_rhom2D_dmo_physical_proj(self, jrp, jz, jM, num_trapz_points=None):
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
        self._profiles_obj_ref = Profiles_obj
            
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
        
        # Pixel arrays. Pixel ids/grouping stay on host; only per-pixel profile
        # properties need to live on the JAX device.
        self.nearby_pix_all = np.asarray(mock_params_dict['nearby_pix_all'], dtype=np.int64)
        self.pix_prop_all = jnp.array(mock_params_dict['pix_prop_all'], dtype=jnp.float32)
        
        # Pre-compute sorting and grouping info, or accept CPU-side grouping from
        # the pixel builder to avoid device-host roundtrips here.
        if (
            mock_params_dict.get('pix_unique') is not None
            and mock_params_dict.get('sort_idx') is not None
            and mock_params_dict.get('boundaries') is not None
        ):
            self.pix_unique = np.asarray(mock_params_dict['pix_unique'], dtype=np.int64)
            self.sort_idx = np.asarray(mock_params_dict['sort_idx'], dtype=np.int64)
            self.boundaries = np.asarray(mock_params_dict['boundaries'], dtype=np.int64)
        else:
            self.pix_unique, self.sort_idx, self.boundaries = self._precompute_pixel_grouping()
        
        # Constants
        self._setup_constants()
        
        # Other parameters
        self.smooth_profiles = mock_params_dict.get('smooth_profiles', False)
        self.use_fused_profile_maps = mock_params_dict.get('use_fused_profile_maps', True)
        self.return_sparse_maps = mock_params_dict.get('return_sparse_maps', False)
        self.store_projected_matter_maps = mock_params_dict.get('store_projected_matter_maps', True)

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
        nearby_pix = np.asarray(self.nearby_pix_all)
        pix_unique = np.unique(nearby_pix)
        sort_idx = np.argsort(nearby_pix)
        sorted_pix = nearby_pix[sort_idx]
        
        change_points = np.diff(sorted_pix, prepend=sorted_pix[0]-1, append=sorted_pix[-1]+1) != 0 
        boundaries = np.where(change_points)[0]
        
        return pix_unique, sort_idx, boundaries

    def _process_maps(self, mock_params_dict):
        """Process all requested map types"""
        get_ymap = mock_params_dict.get('get_ymap', False)
        get_kSZmap = mock_params_dict.get('get_kSZmap', False)
        get_taumap = mock_params_dict.get('get_taumap', False)
        get_kappamap = mock_params_dict.get('get_kappamap', False)   
        get_multi_kappamap = mock_params_dict.get('get_multi_kappamap', False)
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
            self._setup_lensing_kernel_interpolator(mock_params_dict)
            self._get_kappamap()

        if get_multi_kappamap:
            self._setup_multi_lensing_kernel_interpolator(mock_params_dict)
            self._get_multi_kappamaps()

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

    def _setup_lensing_kernel_interpolator(self, mock_params_dict):
        """Set up W_kappa(z) for converting projected mass to convergence.

        The map-side conversion mirrors the analytic Cl convention:

            kappa(theta) = W_kappa(z) * a(z)^2 * Sigma_phys(theta) / rho_m_bar

        where ``W_kappa`` is the same CMB/source lensing kernel used in
        get_Cls and ``Sigma_phys`` is the projected physical matter density
        returned by ``log_rhom2D_interp``.
        """
        kappa_source_bin = int(mock_params_dict.get('kappa_source_bin', 0))
        Wk_array, source_label = self._compute_wkappa_array(
            kappa_source_bin=kappa_source_bin,
            is_cmb_lensing=bool(getattr(self, 'is_cmb_lensing', False)),
        )

        self.kappa_source_bin = kappa_source_bin
        self.kappa_source_label = source_label
        self.Wkappa_array_for_map = jnp.array(Wk_array, dtype=jnp.float32)
        self.Wkappa_interp = interpax.Interpolator1D(
            self.z_array.astype(jnp.float32),
            self.Wkappa_array_for_map,
            extrap=True,
        )

    def _compute_wkappa_array(self, kappa_source_bin=0, is_cmb_lensing=False):
        """Compute the map-side lensing kernel for one DES source bin or CMB."""
        z_grid = np.asarray(self.z_array, dtype=np.float64)
        chi_grid = np.asarray(self.chi_array, dtype=np.float64)
        constant_factor = (
            3.0 * (100.0 ** 2) * float(self.cosmo_jax.Omega_m)
            / (2.0 * ((const.c.value * 1e-3) ** 2))
        )

        if bool(is_cmb_lensing):
            chi_cmb = float(np.ravel(np.asarray(self.chi_CMB, dtype=np.float64))[0])
            radial_kernel = np.clip(chi_cmb - chi_grid, 0.0, None) / max(chi_cmb, 0.1)
            Wk_array = constant_factor * radial_kernel * (1.0 + z_grid) * chi_grid
            source_label = 'cmb'
        else:
            z_source = np.asarray(self.z_array_nz, dtype=np.float64)
            nz_source = np.asarray(self.pzs_inp_mat_inp[kappa_source_bin], dtype=np.float64)
            norm = _np_trapezoid(nz_source, z_source)
            if norm > 0:
                nz_source = nz_source / norm
            chi_source = np.asarray(
                bkgrd.radial_comoving_distance(
                    self.cosmo_jax,
                    jnp.array(1.0 / (1.0 + z_source), dtype=jnp.float32),
                ),
                dtype=np.float64,
            )
            Wk_array = np.zeros_like(z_grid)
            for iz, (z_lens, chi_lens) in enumerate(zip(z_grid, chi_grid)):
                geom = np.clip(chi_source - chi_lens, 0.0, None) / np.clip(chi_source, 0.1, None)
                radial_kernel = _np_trapezoid(nz_source * geom, z_source) * (1.0 + z_lens) * chi_lens
                Wk_array[iz] = constant_factor * radial_kernel
            source_label = f'source_bin_{kappa_source_bin}'
        return Wk_array, source_label

    def _setup_multi_lensing_kernel_interpolator(self, mock_params_dict):
        """Set up multiple W_kappa(z) kernels for one-pass kappa map generation."""
        labels = []
        kernels = []
        for source_bin in mock_params_dict.get('multi_kappa_source_bins', []):
            Wk_array, label = self._compute_wkappa_array(
                kappa_source_bin=int(source_bin),
                is_cmb_lensing=False,
            )
            labels.append(label)
            kernels.append(Wk_array)
        if bool(mock_params_dict.get('multi_kappa_include_cmb', False)):
            Wk_array, label = self._compute_wkappa_array(
                kappa_source_bin=0,
                is_cmb_lensing=True,
            )
            labels.append(label)
            kernels.append(Wk_array)
        if not kernels:
            self.multi_kappa_labels = []
            self.Wkappa_multi_array_for_map = jnp.empty((0, len(self.z_array)), dtype=jnp.float32)
            return
        self.multi_kappa_labels = labels
        self.Wkappa_multi_array_for_map = jnp.array(np.asarray(kernels, dtype=np.float32), dtype=jnp.float32)

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

    def _chunked_vmap_matrix(self, func, n_total, n_cols, chunk_size=20_000_000):
        """Apply a vmapped vector-valued function in chunks."""
        result = np.empty((n_total, n_cols), dtype=np.float32)
        jitted_func = jit(vmap(func))
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            chunk_out = jitted_func(jnp.arange(start, end))
            result[start:end, :] = np.asarray(chunk_out)
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
        if get_kSZmap and get_taumap and self.use_fused_profile_maps:
            if self.profile_timing: start_time = time.perf_counter()
            ne_maps_all = self._chunked_vmap_matrix(self.get_kSZ_tau_healpix, len(self.pix_prop_all), 2)
            self.kszmap_final = self._assemble_map(ne_maps_all[:, 0])
            self.taumap_final = self._assemble_map(ne_maps_all[:, 1])
            if self.profile_timing: self.timing_results['ksz_tau_map_generation_and_assembly'] = time.perf_counter() - start_time
            return

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
        if self.use_fused_profile_maps and self.store_projected_matter_maps:
            kappa_maps_all = self._chunked_vmap_matrix(self.get_rhom_kappa_healpix, len(self.pix_prop_all), 2)
            self.rhommap_final = self._assemble_map(kappa_maps_all[:, 0])
            self.kappamap_final = self._assemble_map(kappa_maps_all[:, 1])
        else:
            if self.store_projected_matter_maps:
                rhomjpix_all = self._chunked_vmap(self.get_rhom_healpix, len(self.pix_prop_all))
                self.rhommap_final = self._assemble_map(rhomjpix_all)
            kappajpix_all = self._chunked_vmap(self.get_kappa_healpix, len(self.pix_prop_all))
            self.kappamap_final = self._assemble_map(kappajpix_all)
        if self.profile_timing: self.timing_results['kappa_map_generation_and_assembly'] = time.perf_counter() - start_time

    def _get_multi_kappamaps(self):
        """Get multiple kappa maps with one density-profile interpolation pass."""
        if self.profile_timing: start_time = time.perf_counter()
        n_kernels = int(len(getattr(self, 'multi_kappa_labels', [])))
        if n_kernels <= 0:
            self.multi_kappamaps_final = {}
            return
        kappa_maps_all = self._chunked_vmap_matrix(self.get_multi_kappa_healpix, len(self.pix_prop_all), n_kernels)
        self.multi_kappamaps_final = {}
        for idx, label in enumerate(self.multi_kappa_labels):
            self.multi_kappamaps_final[str(label)] = self._assemble_map(kappa_maps_all[:, idx])
        if self.profile_timing: self.timing_results['multi_kappa_map_generation_and_assembly'] = time.perf_counter() - start_time

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
        if self.return_sparse_maps:
            return self.pix_unique.astype(np.int64), summed_values.astype(np.float32)
        
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
        
        self.ppf_table_3d, ppf_reused = self._get_or_create_galaxy_ppf_table_3d(
            mock_rho_table, self.mass_grid, self.z_grid, self.r_comoving_grid
        )
        if self.profile_timing:
            self.timing_results['galaxy_ppf_interpolator_creation'] = time.perf_counter() - start_time
            self.timing_results['galaxy_ppf_interpolator_reused'] = bool(ppf_reused)
        
        if self.profile_timing: start_time = time.perf_counter()
        NUM_HALOS = len(mock_params_dict['halo_ra'])
        key = PRNGKey(mock_params_dict.get('random_seed', 42))
        
        halo_vlos = mock_params_dict.get('halo_vlos', jnp.zeros(NUM_HALOS, dtype=jnp.float32))

        keys = split(key, NUM_HALOS)
        galaxy_chunk_size = int(mock_params_dict.get('galaxy_population_chunk_size', 20000))
        galaxy_chunk_size = max(1, galaxy_chunk_size)
        max_gals_round_to = int(mock_params_dict.get('galaxy_max_gals_round_to', 16))
        max_gals_round_to = max(1, max_gals_round_to)
        group_by_max_gals = bool(mock_params_dict.get('galaxy_population_group_by_max_gals', False))
        compact_max_satellite_groups = int(mock_params_dict.get('galaxy_compact_max_satellite_groups', 32))
        compact_max_satellite_groups = max(1, compact_max_satellite_groups)
        halo_ra_all = jnp.array(mock_params_dict['halo_ra'], dtype=jnp.float32)
        halo_dec_all = jnp.array(mock_params_dict['halo_dec'], dtype=jnp.float32)
        halo_z_all = jnp.array(mock_params_dict['halo_z'], dtype=jnp.float32)
        halo_M_all = jnp.array(mock_params_dict['halo_M'], dtype=jnp.float64)
        halo_vlos_all = jnp.array(halo_vlos, dtype=jnp.float32)
        if mock_params_dict.get('halo_DA') is not None:
            halo_da_all = jnp.array(mock_params_dict['halo_DA'], dtype=jnp.float32)
        else:
            halo_da_all = vmap(self.angular_diameter_distance)(halo_z_all).astype(jnp.float32)
        population_backend = str(mock_params_dict.get('galaxy_population_backend', 'padded_precomputed')).replace('-', '_')
        if population_backend in {'compact_exact', 'exact_compact'}:
            population_backend = 'compact'
        if population_backend not in {'compact', 'padded', 'padded_precomputed'}:
            raise ValueError(
                "galaxy_population_backend must be one of 'compact', 'padded', or "
                f"'padded_precomputed', got {population_backend!r}."
            )

        valid_catalog_chunks = []
        populate_func_cache = {}
        satellite_func_cache = {}
        diag = {
            "backend": population_backend,
            "use_poisson_centrals": bool(self._use_poisson_centrals),
            "n_halos": int(NUM_HALOS),
            "chunk_size": int(galaxy_chunk_size),
            "group_by_max_gals": bool(group_by_max_gals),
            "max_gals_round_to": int(max_gals_round_to),
            "compact_max_satellite_groups": int(compact_max_satellite_groups),
            "expected_ncen": 0.0,
            "expected_nsat": 0.0,
            "realized_ncen": 0,
            "realized_nsat": 0,
            "n_clipped_halos": 0,
            "n_clipped_sats": 0,
            "max_nsat_raw": 0,
            "max_nsat_clipped": 0,
            "max_gals_per_halo": 0,
            "n_satellite_groups": 0,
            "n_nonzero_satellite_halos": 0,
            "n_compact_chunks": 0,
            "n_compact_fallback_chunks": 0,
            "compact_fallback_reasons": [],
            "n_output_galaxies": 0,
        }
        for start in range(0, NUM_HALOS, galaxy_chunk_size):
            end = min(start + galaxy_chunk_size, NUM_HALOS)
            mean_ncen_chunk, mean_nsat_chunk = vmap(self.get_hod_params)(
                halo_M_all[start:end],
                halo_z_all[start:end],
            )
            diag["expected_ncen"] += float(np.sum(np.asarray(mean_ncen_chunk, dtype=np.float64)))
            diag["expected_nsat"] += float(np.sum(np.asarray(mean_nsat_chunk, dtype=np.float64)))
            if group_by_max_gals:
                mean_nsat_np = np.asarray(mean_nsat_chunk, dtype=np.float64)
                raw_max_gals_np = np.ceil(mean_nsat_np + np.sqrt(np.maximum(mean_nsat_np, 0.0))).astype(np.int64) + 2
                max_gals_np = (
                    np.ceil(np.maximum(2, raw_max_gals_np) / float(max_gals_round_to)).astype(np.int64)
                    * int(max_gals_round_to)
                )
                group_values = np.unique(max_gals_np)
            else:
                max_mean_nsat = float(jnp.max(mean_nsat_chunk))
                raw_max_gals = int(math.ceil(max_mean_nsat + math.sqrt(max(max_mean_nsat, 0.0)))) + 2
                group_values = np.array(
                    [int(math.ceil(max(2, raw_max_gals) / max_gals_round_to) * max_gals_round_to)],
                    dtype=np.int64,
                )
                max_gals_np = np.full(end - start, int(group_values[0]), dtype=np.int64)
            diag["max_gals_per_halo"] = max(diag["max_gals_per_halo"], int(np.max(max_gals_np)) if len(max_gals_np) else 0)

            if population_backend == 'compact':
                max_sats_chunk = jnp.asarray(max_gals_np - 1, dtype=jnp.int32)
                ncen_chunk, nsat_chunk, nsat_raw_chunk = self.sample_hod_counts_from_means(
                    keys[start:end],
                    mean_ncen_chunk,
                    mean_nsat_chunk,
                    max_sats_chunk,
                    bool(self._use_poisson_centrals),
                )
                ncen_np = np.asarray(ncen_chunk, dtype=np.int32)
                nsat_np = np.asarray(nsat_chunk, dtype=np.int32)
                nsat_raw_np = np.asarray(nsat_raw_chunk, dtype=np.int32)
                diag["realized_ncen"] += int(np.sum(ncen_np))
                diag["realized_nsat"] += int(np.sum(nsat_np))
                diag["n_clipped_halos"] += int(np.count_nonzero(nsat_raw_np > nsat_np))
                diag["n_clipped_sats"] += int(np.sum(nsat_raw_np - nsat_np))
                diag["max_nsat_raw"] = max(diag["max_nsat_raw"], int(np.max(nsat_raw_np)) if len(nsat_raw_np) else 0)
                diag["max_nsat_clipped"] = max(diag["max_nsat_clipped"], int(np.max(nsat_np)) if len(nsat_np) else 0)

                sat_counts = np.unique(nsat_np[nsat_np > 0])
                diag["n_satellite_groups"] += int(len(sat_counts))
                diag["n_nonzero_satellite_halos"] += int(np.count_nonzero(nsat_np > 0))
                new_sat_count_values = [int(value) for value in sat_counts if int(value) not in satellite_func_cache]
                total_satellite_kernels_after_chunk = len(satellite_func_cache) + len(new_sat_count_values)
                if len(sat_counts) > compact_max_satellite_groups or total_satellite_kernels_after_chunk > compact_max_satellite_groups:
                    diag["n_compact_fallback_chunks"] += 1
                    diag["compact_fallback_reasons"].append(
                        {
                            "halo_start": int(start),
                            "halo_end": int(end),
                            "n_satellite_groups": int(len(sat_counts)),
                            "n_new_satellite_groups": int(len(new_sat_count_values)),
                            "n_cached_satellite_groups": int(len(satellite_func_cache)),
                            "max_nsat_clipped": int(np.max(nsat_np)) if len(nsat_np) else 0,
                        }
                    )
                    for max_gals_per_halo_raw in group_values:
                        max_gals_per_halo = int(max_gals_per_halo_raw)
                        local_idx = np.where(max_gals_np == max_gals_per_halo)[0]
                        if local_idx.size == 0:
                            continue
                        halo_idx = jnp.asarray(start + local_idx, dtype=jnp.int32)
                        cache_key = ('compact_fallback_padded_precomputed', max_gals_per_halo)
                        if cache_key not in populate_func_cache:
                            def populate_wrapper(key, ra, dec, z, mass, vlos, da, mean_ncen, mean_nsat, _max_gals=max_gals_per_halo):
                                return self.populate_one_halo_precomputed_hod(
                                    key, ra, dec, z, mass, vlos, da, _max_gals, mean_ncen, mean_nsat,
                                    self.ppf_table_3d["mass_grid"], self.ppf_table_3d["z_grid"],
                                    self.ppf_table_3d["r_comoving_grid"], self.ppf_table_3d["cdf_table"]
                                )
                            populate_func_cache[cache_key] = jit(vmap(populate_wrapper))
                        padded_chunk = populate_func_cache[cache_key](
                            keys[halo_idx],
                            halo_ra_all[halo_idx],
                            halo_dec_all[halo_idx],
                            halo_z_all[halo_idx],
                            halo_M_all[halo_idx],
                            halo_vlos_all[halo_idx],
                            halo_da_all[halo_idx],
                            mean_ncen_chunk[jnp.asarray(local_idx, dtype=jnp.int32)],
                            mean_nsat_chunk[jnp.asarray(local_idx, dtype=jnp.int32)],
                        )
                        padded_chunk.block_until_ready()
                        flat_chunk = padded_chunk.reshape(-1, 7)
                        valid_chunk = flat_chunk[flat_chunk[:, 5] > 0.5]
                        valid_catalog_chunks.append(np.asarray(valid_chunk, dtype=np.float32))
                        del padded_chunk, flat_chunk, valid_chunk, halo_idx
                    continue

                diag["n_compact_chunks"] += 1
                central_idx = np.where(ncen_np > 0)[0]
                if central_idx.size:
                    central_rows = np.empty((central_idx.size, 7), dtype=np.float32)
                    central_rows[:, 0] = np.asarray(halo_ra_all[start:end], dtype=np.float32)[central_idx]
                    central_rows[:, 1] = np.asarray(halo_dec_all[start:end], dtype=np.float32)[central_idx]
                    central_rows[:, 2] = np.asarray(halo_z_all[start:end], dtype=np.float32)[central_idx]
                    central_rows[:, 3] = np.asarray(halo_M_all[start:end], dtype=np.float32)[central_idx]
                    central_rows[:, 4] = 1.0
                    central_rows[:, 5] = 1.0
                    central_rows[:, 6] = np.asarray(halo_vlos_all[start:end], dtype=np.float32)[central_idx]
                    valid_catalog_chunks.append(central_rows)
                for sat_count_raw in sat_counts:
                    sat_count = int(sat_count_raw)
                    local_idx = np.where(nsat_np == sat_count)[0]
                    if local_idx.size == 0:
                        continue
                    halo_idx = jnp.asarray(start + local_idx, dtype=jnp.int32)
                    if sat_count not in satellite_func_cache:
                        def satellite_wrapper(key, ra, dec, z, mass, vlos, da, _sat_count=sat_count):
                            return self.populate_satellites_fixed_count(
                                key, ra, dec, z, mass, vlos, da, _sat_count,
                                self.ppf_table_3d["mass_grid"], self.ppf_table_3d["z_grid"],
                                self.ppf_table_3d["r_comoving_grid"], self.ppf_table_3d["cdf_table"]
                            )
                        satellite_func_cache[sat_count] = jit(vmap(satellite_wrapper))
                    sat_chunk = satellite_func_cache[sat_count](
                        keys[halo_idx],
                        halo_ra_all[halo_idx],
                        halo_dec_all[halo_idx],
                        halo_z_all[halo_idx],
                        halo_M_all[halo_idx],
                        halo_vlos_all[halo_idx],
                        halo_da_all[halo_idx],
                    )
                    sat_chunk.block_until_ready()
                    valid_catalog_chunks.append(np.asarray(sat_chunk.reshape(-1, 7), dtype=np.float32))
                    del sat_chunk, halo_idx
                continue

            for max_gals_per_halo_raw in group_values:
                max_gals_per_halo = int(max_gals_per_halo_raw)
                local_idx = np.where(max_gals_np == max_gals_per_halo)[0]
                if local_idx.size == 0:
                    continue
                halo_idx = jnp.asarray(start + local_idx, dtype=jnp.int32)
                cache_key = (population_backend, max_gals_per_halo)
                if cache_key not in populate_func_cache:
                    if population_backend == 'padded_precomputed':
                        def populate_wrapper(key, ra, dec, z, mass, vlos, da, mean_ncen, mean_nsat, _max_gals=max_gals_per_halo):
                            return self.populate_one_halo_precomputed_hod(
                                key, ra, dec, z, mass, vlos, da, _max_gals, mean_ncen, mean_nsat,
                                self.ppf_table_3d["mass_grid"], self.ppf_table_3d["z_grid"],
                                self.ppf_table_3d["r_comoving_grid"], self.ppf_table_3d["cdf_table"]
                            )
                    else:
                        def populate_wrapper(key, ra, dec, z, mass, vlos, da, mean_ncen, mean_nsat, _max_gals=max_gals_per_halo):
                            return self.populate_one_halo(
                                key, ra, dec, z, mass, vlos, _max_gals,
                                self.ppf_table_3d["mass_grid"], self.ppf_table_3d["z_grid"],
                                self.ppf_table_3d["r_comoving_grid"], self.ppf_table_3d["cdf_table"]
                            )
                    populate_func_cache[cache_key] = jit(vmap(populate_wrapper))
                jitted_vectorized_populate = populate_func_cache[cache_key]
                padded_chunk = jitted_vectorized_populate(
                    keys[halo_idx],
                    halo_ra_all[halo_idx],
                    halo_dec_all[halo_idx],
                    halo_z_all[halo_idx],
                    halo_M_all[halo_idx],
                    halo_vlos_all[halo_idx],
                    halo_da_all[halo_idx],
                    mean_ncen_chunk[jnp.asarray(local_idx, dtype=jnp.int32)],
                    mean_nsat_chunk[jnp.asarray(local_idx, dtype=jnp.int32)],
                )
                padded_chunk.block_until_ready()
                flat_chunk = padded_chunk.reshape(-1, 7)
                valid_chunk = flat_chunk[flat_chunk[:, 5] > 0.5]
                valid_np = np.asarray(valid_chunk, dtype=np.float32)
                diag["realized_ncen"] += int(np.count_nonzero(valid_np[:, 4] > 0.5)) if len(valid_np) else 0
                diag["realized_nsat"] += int(np.count_nonzero((valid_np[:, 4] < 0.5) & (valid_np[:, 5] > 0.5))) if len(valid_np) else 0
                valid_catalog_chunks.append(valid_np)
                del padded_chunk, flat_chunk, valid_chunk, halo_idx
        if self.profile_timing: self.timing_results['galaxy_population'] = time.perf_counter() - start_time
        
        if self.profile_timing: start_time = time.perf_counter()
        if valid_catalog_chunks:
            self.final_galaxy_catalog = np.concatenate(valid_catalog_chunks, axis=0)
        else:
            self.final_galaxy_catalog = np.empty((0, 7), dtype=np.float32)
        diag["n_output_galaxies"] = int(len(self.final_galaxy_catalog))
        self.galaxy_population_diagnostics = diag
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
    def get_kSZ_tau_healpix(self, jpix):
        """Get kSZ and tau values with one electron-density interpolation."""
        prop = self.pix_prop_all[jpix]
        ne2d = jnp.exp(self.log_ne2D_interp(prop[0], prop[1], prop[2]))
        tau_z = self.tau_interp(prop[1])
        fac = jnp.exp(-tau_z)
        ksz = -self.const_coeff_kSZ * fac * prop[3] * ne2d
        tau = self.const_coeff_tau * ne2d
        return jnp.array([ksz, tau], dtype=jnp.float32)

    @partial(jit, static_argnums=(0,))
    def get_rhom_healpix(self, jpix):
        """Get rhom value for HEALPix pixel"""
        prop = self.pix_prop_all[jpix]
        DA_val = jnp.exp(jnp.interp(prop[1], self.z_array, jnp.log(self.DA_array)))
        pix_area_corr = self.pix_area * (DA_val**2)
        return pix_area_corr * jnp.exp(self.log_rhom2D_interp(prop[0], prop[1], prop[2]))

    @partial(jit, static_argnums=(0,))
    def get_kappa_healpix(self, jpix):
        """Get lensing-weighted convergence for a HEALPix pixel."""
        prop = self.pix_prop_all[jpix]
        z = prop[1]
        scale_factor = 1.0 / (1.0 + z)
        sigma_phys = jnp.exp(self.log_rhom2D_interp(prop[0], z, prop[2]))
        Wkappa = self.Wkappa_interp(z)
        return Wkappa * (scale_factor ** 2) * sigma_phys / self.rho_m_bar

    @partial(jit, static_argnums=(0,))
    def get_rhom_kappa_healpix(self, jpix):
        """Get projected matter and lensing-weighted kappa with one density interpolation."""
        prop = self.pix_prop_all[jpix]
        z = prop[1]
        sigma_phys = jnp.exp(self.log_rhom2D_interp(prop[0], z, prop[2]))
        DA_val = jnp.exp(jnp.interp(z, self.z_array, jnp.log(self.DA_array)))
        pix_area_corr = self.pix_area * (DA_val**2)
        rhom = pix_area_corr * sigma_phys
        scale_factor = 1.0 / (1.0 + z)
        Wkappa = self.Wkappa_interp(z)
        kappa = Wkappa * (scale_factor ** 2) * sigma_phys / self.rho_m_bar
        return jnp.array([rhom, kappa], dtype=jnp.float32)

    @partial(jit, static_argnums=(0,))
    def get_multi_kappa_healpix(self, jpix):
        """Get multiple lensing-weighted convergence values from one density interpolation."""
        prop = self.pix_prop_all[jpix]
        z = prop[1]
        scale_factor = 1.0 / (1.0 + z)
        sigma_phys = jnp.exp(self.log_rhom2D_interp(prop[0], z, prop[2]))
        wkappa = vmap(lambda kernel: jnp.interp(z, self.z_array, kernel))(self.Wkappa_multi_array_for_map)
        return wkappa * (scale_factor ** 2) * sigma_phys / self.rho_m_bar

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
        mean_ncen = jnp.nan_to_num(self.Ncen_interp(z, jnp.log(mass)))
        mean_nsat = jnp.nan_to_num(jnp.exp(self.logNsat_interp(z, jnp.log(mass))))
        # Clamp to physical range (Ncen ∈ [0,1] by construction of the erf HOD)
        mean_ncen = jnp.clip(mean_ncen, 0.0, 1.0)
        mean_nsat = jnp.clip(mean_nsat, 0.0)

        return mean_ncen, mean_nsat

    @partial(jit, static_argnums=(0, 5))
    def sample_hod_counts_from_means(self, keys, mean_ncen, mean_nsat, max_sats, use_poisson_centrals):
        """Sample central/satellite counts from precomputed HOD means."""

        def _one(key, ncen_mean, nsat_mean, max_sat):
            key_hod, _ = split(key)
            key_sat = split(key_hod, 2)[1]
            if use_poisson_centrals:
                ncen = jnp.clip(poisson(key_hod, ncen_mean), 0, 1).astype(jnp.int32)
            else:
                ncen = bernoulli(key_hod, ncen_mean).astype(jnp.int32)
            nsat_raw = poisson(key_sat, nsat_mean).astype(jnp.int32)
            nsat = jnp.minimum(nsat_raw, max_sat.astype(jnp.int32))
            return ncen, nsat, nsat_raw

        return vmap(_one)(keys, mean_ncen, mean_nsat, max_sats)

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

    def _get_or_create_galaxy_ppf_table_3d(self, profile_table, mass_grid, z_grid, r_comoving_grid):
        """Reuse the galaxy PPF table across chunks when the setup object is reused."""
        cache_key = (
            tuple(np.shape(profile_table)),
            tuple(np.shape(mass_grid)),
            tuple(np.shape(z_grid)),
            tuple(np.shape(r_comoving_grid)),
            str(getattr(profile_table, "dtype", "")),
        )
        cached = getattr(self, "_cached_galaxy_ppf_table_3d", None)
        if isinstance(cached, dict) and cached.get("cache_key") == cache_key:
            return cached["table"], True

        table = self.create_ppf_interpolator_3d(profile_table, mass_grid, z_grid, r_comoving_grid)
        cached = {"cache_key": cache_key, "table": table}
        self._cached_galaxy_ppf_table_3d = cached
        profiles_obj_ref = getattr(self, "_profiles_obj_ref", None)
        if profiles_obj_ref is not None:
            setattr(profiles_obj_ref, "_cached_galaxy_ppf_table_3d", cached)
        return table, False

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

    @partial(jit, static_argnums=(0, 8, 9))
    def place_satellites_with_da(self, key, r_sats_comoving, halo_ra, halo_dec, halo_z, halo_mass, halo_da_hMpc, num_sats, do_rsd=False):
        """Place satellites using a precomputed angular diameter distance in Mpc/h."""
        r_sats_physical = r_sats_comoving / (1 + halo_z)

        key_pos, key_vel = split(key)
        phi = uniform(key_pos, (num_sats,), minval=0, maxval=2*jnp.pi)
        cos_theta = uniform(split(key_pos, 2)[1], (num_sats,), minval=-1, maxval=1)
        sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta**2, 0.0))

        dx = r_sats_physical * sin_theta * jnp.cos(phi)
        dy = r_sats_physical * sin_theta * jnp.sin(phi)

        dist_a_kpc = jnp.maximum(halo_da_hMpc, 1.0e-8) * 1000.0

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

        return sat_ra.astype(jnp.float32), sat_dec.astype(jnp.float32), sat_z.astype(jnp.float32)

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
        
        return sat_ra.astype(jnp.float32), sat_dec.astype(jnp.float32), sat_z.astype(jnp.float32)

    @partial(jit, static_argnums=(0, 8))
    def populate_satellites_fixed_count(self, key, halo_ra, halo_dec, halo_z, halo_mass, halo_vlos, halo_da_hMpc, num_sats,
                                        mass_grid, z_grid, r_comoving_grid, cdf_table):
        """Populate only satellites for halos with the same realized satellite count."""
        _, key_sat_pos = split(key)
        r_sats_comoving = self.sample_radii_from_ppf_3d(
            key_sat_pos, num_sats, halo_mass, halo_z,
            mass_grid, z_grid, r_comoving_grid, cdf_table
        )
        sats_ra, sats_dec, sats_z = self.place_satellites_with_da(
            split(key_sat_pos, 2)[1], r_sats_comoving,
            halo_ra, halo_dec, halo_z, halo_mass, halo_da_hMpc, num_sats
        )
        return jnp.stack(
            [
                sats_ra,
                sats_dec,
                sats_z,
                jnp.full((num_sats,), halo_mass, dtype=jnp.float32),
                jnp.zeros((num_sats,), dtype=jnp.float32),
                jnp.ones((num_sats,), dtype=jnp.float32),
                jnp.full((num_sats,), halo_vlos, dtype=jnp.float32),
            ],
            axis=1,
        ).astype(jnp.float32)

    def populate_one_halo_precomputed_hod(self, key, halo_ra, halo_dec, halo_z, halo_mass, halo_vlos, halo_da_hMpc, max_gals,
                                          mean_ncen, mean_nsat, mass_grid, z_grid, r_comoving_grid, cdf_table):
        """Padded one-halo population using precomputed HOD means and catalog DA."""
        key_hod, key_sat_pos = split(key)

        ncen = jnp.where(
            getattr(self, '_use_poisson_centrals', False),
            jnp.clip(poisson(key_hod, mean_ncen), 0, 1),
            bernoulli(key_hod, mean_ncen).astype(jnp.int32),
        )
        nsat = poisson(split(key_hod, 2)[1], mean_nsat)
        nsat = jnp.minimum(nsat, max_gals - 1)

        pad_value = jnp.array(-1.0, dtype=jnp.float32)
        zero = jnp.array(0.0, dtype=jnp.float32)
        one = jnp.array(1.0, dtype=jnp.float32)
        halo_mass_f32 = halo_mass.astype(jnp.float32)
        gal_catalog = jnp.full((max_gals, 7), pad_value, dtype=jnp.float32)
        gal_catalog = gal_catalog.at[:, 5].set(zero)
        gal_catalog = gal_catalog.at[:, 6].set(zero)

        central_valid = jnp.where(ncen > 0, one, zero)
        gal_catalog = gal_catalog.at[0].set(
            jnp.array([halo_ra, halo_dec, halo_z, halo_mass_f32, one, central_valid, halo_vlos], dtype=jnp.float32)
        )

        r_sats_comoving = self.sample_radii_from_ppf_3d(
            key_sat_pos, max_gals - 1, halo_mass, halo_z,
            mass_grid, z_grid, r_comoving_grid, cdf_table
        )

        sats_ra, sats_dec, sats_z = self.place_satellites_with_da(
            split(key_sat_pos, 2)[1], r_sats_comoving,
            halo_ra, halo_dec, halo_z, halo_mass, halo_da_hMpc, max_gals - 1
        )

        sat_indices = jnp.arange(1, max_gals)
        sat_mask = (sat_indices <= nsat)
        gal_catalog = gal_catalog.at[1:, 0].set(jnp.where(sat_mask, sats_ra, pad_value))
        gal_catalog = gal_catalog.at[1:, 1].set(jnp.where(sat_mask, sats_dec, pad_value))
        gal_catalog = gal_catalog.at[1:, 2].set(jnp.where(sat_mask, sats_z, pad_value))
        gal_catalog = gal_catalog.at[1:, 3].set(jnp.where(sat_mask, halo_mass_f32, pad_value))
        gal_catalog = gal_catalog.at[1:, 4].set(zero)
        gal_catalog = gal_catalog.at[1:, 5].set(jnp.where(sat_mask, one, zero))
        gal_catalog = gal_catalog.at[1:, 6].set(jnp.where(sat_mask, halo_vlos, zero))

        return gal_catalog

    def populate_one_halo(self, key, halo_ra, halo_dec, halo_z, halo_mass, halo_vlos, max_gals,
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
        pad_value = jnp.array(-1.0, dtype=jnp.float32)
        zero = jnp.array(0.0, dtype=jnp.float32)
        one = jnp.array(1.0, dtype=jnp.float32)
        halo_mass_f32 = halo_mass.astype(jnp.float32)
        gal_catalog = jnp.full((max_gals, 7), pad_value, dtype=jnp.float32)
        gal_catalog = gal_catalog.at[:, 5].set(zero)
        gal_catalog = gal_catalog.at[:, 6].set(zero)
        
        # Place central
        central_valid = jnp.where(ncen > 0, one, zero)
        gal_catalog = gal_catalog.at[0].set(
            jnp.array([halo_ra, halo_dec, halo_z, halo_mass_f32, one, central_valid, halo_vlos], dtype=jnp.float32)
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
        gal_catalog = gal_catalog.at[1:, 3].set(jnp.where(sat_mask, halo_mass_f32, pad_value))
        gal_catalog = gal_catalog.at[1:, 4].set(zero)  # satellites
        gal_catalog = gal_catalog.at[1:, 5].set(jnp.where(sat_mask, one, zero))
        gal_catalog = gal_catalog.at[1:, 6].set(jnp.where(sat_mask, halo_vlos, zero))
        
        return gal_catalog
