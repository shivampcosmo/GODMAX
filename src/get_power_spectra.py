import os
from get_BCMP_profile_jit import BCM_18_wP
from setup_power_spectra import setup_power_BCMP
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from jax import vmap, grad
from jax_cosmo import Cosmology
from functools import partial
from jax_cosmo.power import sigmasqr
from jax_cosmo.background import angular_diameter_distance, radial_comoving_distance
import jax_cosmo.transfer as tklib
import astropy.units as u
from astropy import constants as const
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.eV / u.cm**3)).value
import constants
from mcfit import xi2P

import jax_cosmo.background as bkgrd
import jax_cosmo.constants as const
import jax_cosmo.transfer as tklib
from jax_cosmo.scipy.integrate import romb
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.scipy.interpolate import interp


class get_power_BCMP:
    def __init__(
                self,
                sim_params_dict,
                halo_params_dict,
                nz_info_dict,
                num_points_trapz_int=64
            ):    
        
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

        setup_power_BCMP_test = setup_power_BCMP(sim_params_dict, halo_params_dict, num_points_trapz_int=num_points_trapz_int)

        