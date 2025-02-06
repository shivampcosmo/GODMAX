import numpy as np
from base_class import base_class, get_vmapped_func, get_vmapped_func_warg
from functools import partial
from jax import grad, jit, vmap
import astropy.units as u
from astropy import constants as const
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.keV / u.cm**3)).value
mp = (1.6726219e-27*u.kg).to(u.Msun).value
mue = 1.14
Mpc_to_cm = 3.086e24
G_new_rhom = const.G.to(u.Mpc**3 / ((u.s**2) * u.M_sun))
import helpers.constants as constants
import jax_cosmo.background as bkgrd
from jax_cosmo import Cosmology
import jax.numpy as jnp

pressure_params_def = {
    'P0': {
        'A0': 18.1,
        'alpha_m': 0.154,
        'alpha_z': -0.758
        },
    'xc': {
        'A0': 0.497,
        'alpha_m': -0.00865,
        'alpha_z': 0.731
        },
    'beta': {
        'A0': 4.35,
        'alpha_m': 0.0393,
        'alpha_z': 0.415
        }
    }
density_params_def = {
    'rho0': {
        'A0': 4e3,
        'alpha_m': 0.29,
        'alpha_z': -0.66
        },
    'alpha': {
        'A0': 0.88,
        'alpha_m': -0.03,
        'alpha_z': 0.19
        },
    'beta': {
        'A0': 3.83,
        'alpha_m': 0.04,
        'alpha_z': -0.025
        }
    }


class Battaglia_12_16(base_class):

    def __init__(
            self,
            sim_params_dict=None,
            halo_params_dict=None,
            pressure_params_def=pressure_params_def,
            density_params_def=density_params_def,
        ):
        '''Note that here M is in Msun/h'''
        super().__init__(sim_params_dict, halo_params_dict, {}, {})
        self.pressure_params_def = pressure_params_def
        self.density_params_def = density_params_def
        self.r200c_mat = get_vmapped_func(self.get_M_to_R, 2)(jnp.arange(self.nM), jnp.arange(self.nz)).T
        self.fb = self.cosmo_params['Ob0'] / self.cosmo_params['Om0']
        self.rho_gas_mat_physical = get_vmapped_func(self.get_rho_gas, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.Pth_mat_physical = get_vmapped_func(self.get_Pth, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.Pe_mat_physical = self.Pth_mat_physical/1.932
        self.ne_mat_physical = self.rho_gas_mat_physical/(mue*mp*(Mpc_to_cm**3)) # in cm**-3

    @partial(jit, static_argnums=(0,))
    def get_M_to_R(self, jM, jz, mdef_delta=200):
        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(self.cosmo_jax,self.scale_fac_a_array[jz]) * 1e9
        rho_treshold = mdef_delta * rho_c_z
        R = (self.M_array[jM] * 3.0 / 4.0 / jnp.pi / rho_treshold)**(1.0 / 3.0)
        return R

    @partial(jit, static_argnums=(0,1))
    def get_params_density(self, key, jM, jz):
        params_dict = self.density_params_def
        Mval = self.M_array[jM] / self.h
        zval = self.z_array[jz]
        A0 = params_dict[key]['A0']
        alpha_m = params_dict[key]['alpha_m']
        alpha_z = params_dict[key]['alpha_z']
        A = A0 * (Mval / 1e14)**alpha_m * (1 + zval)**alpha_z
        return A

    @partial(jit, static_argnums=(0,1))
    def get_params_pressure(self, key, jM, jz):
        params_dict = self.pressure_params_def
        Mval = self.M_array[jM] / self.h
        zval = self.z_array[jz]
        A0 = params_dict[key]['A0']
        alpha_m = params_dict[key]['alpha_m']
        alpha_z = params_dict[key]['alpha_z']
        A = A0 * (Mval / 1e14)**alpha_m * (1 + zval)**alpha_z
        return A

    @partial(jit, static_argnums=(0))
    def get_rho_fit(self, jr, jz, jM):
        rho0_density = self.get_params_density('rho0', jM, jz)
        alpha_density = self.get_params_density('alpha', jM, jz)
        beta_density = self.get_params_density('beta', jM, jz)
        xc_density = 0.5
        gamma_density = -0.2

        a = self.scale_fac_a_array[jz]
        x = a * self.r_array[jr] / (self.r200c_mat[jM, jz])
        # rho_fit = rho0_density * ((x / xc_density)**gamma_density) * (
        #     1 + (x / xc_density)**alpha_density
        #     )**(-(beta_density - gamma_density) / alpha_density)
        rho_fit = rho0_density * ((x / xc_density)**gamma_density) * (
            1 + (x / xc_density)**alpha_density
            )**(-(beta_density + gamma_density) / alpha_density)
        return rho_fit

    @partial(jit, static_argnums=(0))
    def get_P_fit(self, jr, jz, jM):
        a = self.scale_fac_a_array[jz]
        x = a * self.r_array[jr] / (self.r200c_mat[jM, jz])
        P0_pressure = self.get_params_pressure('P0', jM, jz)
        xc_pressure = self.get_params_pressure('xc', jM, jz)
        beta_pressure = self.get_params_pressure('beta', jM, jz)
        alpha_pressure = 1.0
        gamma_pressure = -0.3

        P_fit = P0_pressure * (x / xc_pressure)**gamma_pressure * (1 + (x / xc_pressure)**alpha_pressure)**(-beta_pressure)
        return P_fit

    @partial(jit, static_argnums=(0))
    def get_rho_gas(self, jr, jz, jM):
        rho_fit = self.get_rho_fit(jr, jz, jM)
        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(self.cosmo_jax,self.scale_fac_a_array[jz]) * 1e9 * self.h**2
        return self.fb * rho_c_z * rho_fit

    @partial(jit, static_argnums=(0))
    def get_Pth(self, jr, jz, jM):
        P_fit = self.get_P_fit(jr, jz, jM)

        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(self.cosmo_jax,self.scale_fac_a_array[jz]) * 1e9 * self.h**2        
        coeff = (const.G * (const.M_sun**2) / ((1.0 * u.Mpc)**4)).to((u.keV / (u.cm**3))).value

        Mval = self.M_array[jM]
        rDelta = self.r200c_mat[jM, jz]
        P_Delta = coeff * Mval * self.mdef_Delta * rho_c_z * self.fb / (2. * rDelta)

        return P_Delta * P_fit