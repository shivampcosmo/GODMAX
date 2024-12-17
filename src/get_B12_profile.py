import numpy as np
from scipy.special import spence
from scipy.optimize import fmin, differential_evolution, minimize
from scipy.optimize import newton
import scipy as sp
from functools import partial
from jax import grad, jit, vmap
import scipy.interpolate as interpolate
from colossus.cosmology import cosmology
import astropy.units as u
from astropy import constants as const
from colossus.halo import mass_so
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.keV / u.cm**3)).value
mp = (1.6726219e-27*u.kg).to(u.Msun).value
mue = 1.14
Mpc_to_cm = 3.086e24
G_new_rhom = const.G.to(u.Mpc**3 / ((u.s**2) * u.M_sun))
import constants
import jax_cosmo.background as bkgrd
import time
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


class Battaglia_12_16:

    def __init__(
            self,
            halo_params_dict,
            cosmo_params=None,
            pressure_params_def=pressure_params_def,
            density_params_def=density_params_def,
            mdef_Delta=200
        ):
        '''Note that here M is in Msun/h'''
        # if cosmo is None:
            # cosmo = cosmology.setCosmology('planck18')
        self.cosmo_params = cosmo_params
        self.cosmo_jax = Cosmology(
            Omega_c=cosmo_params['Om0'] - cosmo_params['Ob0'],
            Omega_b=cosmo_params['Ob0'],
            h=cosmo_params['H0'] / 100.,
            sigma8=cosmo_params['sigma8'],
            n_s=cosmo_params['ns'],
            Omega_k=0.,
            w0=cosmo_params['w0'],
            wa=0.
            )


        rmin, rmax, nr = halo_params_dict.get('rmin', 5e-3), halo_params_dict.get('rmax',3), halo_params_dict.get('nr', 63)
        zmin, zmax, nz = halo_params_dict.get('zmin', 1e-3), halo_params_dict.get('zmax',1.5), halo_params_dict.get('nz',32)
        lg10_Mmin, lg10_Mmax, nM = halo_params_dict.get('lg10_Mmin', 12), halo_params_dict.get('lg10_Mmax', 15.0), halo_params_dict.get('nM', 32)
        # if self.conc_dep_model:
            # cmin, cmax, nc = halo_params_dict.get('cmin',2), halo_params_dict.get('cmax',8), halo_params_dict.get('nc',32)
            # self.conc_array = jnp.exp(jnp.linspace(jnp.log(cmin), jnp.log(cmax), nc))
        self.r_array = jnp.logspace(jnp.log10(rmin), jnp.log10(rmax), nr)
        if 'z_array' in halo_params_dict.keys():
            self.z_array = jnp.array(halo_params_dict['z_array'])
            nz = len(self.z_array)
        else:
            self.z_array = jnp.linspace(zmin, zmax, nz)
        self.scale_fac_a_array = 1./(1. + self.z_array)
        self.M_array = jnp.logspace(lg10_Mmin, lg10_Mmax, nM)
        self.nM, self.nz = nM, nz

        # self.cosmo = cosmo
        self.h = cosmo_params['H0'] / 100.
        # self.M = M / self.h
        # print('M = ', self.M)
        # self.z = z
        self.mdef_Delta = mdef_Delta
        # mdef = str(mdef_Delta) + 'c'
        self.pressure_params_def = pressure_params_def
        self.density_params_def = density_params_def
        # self.rDelta = mass_so.M_to_R(M, z, mdef) / (1000. * self.h)

        vmap_func1 = vmap(self.get_M_to_R, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.r200c_mat = vmap_func2(jnp.arange(nM), jnp.arange(nz)).T


        # self.rho0_density = self.get_params('rho0', density_params_def)
        # self.alpha_density = self.get_params('alpha', density_params_def)
        # self.beta_density = self.get_params('beta', density_params_def)
        # self.xc_density = 0.5
        # self.gamma_density = -0.2

        # self.P0_pressure = self.get_params('P0', pressure_params_def)
        # self.xc_pressure = self.get_params('xc', pressure_params_def)
        # self.beta_pressure = self.get_params('beta', pressure_params_def)
        # self.alpha_pressure = 1.0
        # self.gamma_pressure = -0.3

        # self.rho_crit_z = cosmo.rho_crit(z) * 1e9 * h**2
        # self.rho_crit_z = cosmo.rho_c(z) * 1e9 * self.h**2
        # self.rho_crit_z = cosmo.rho_c(z) * 1e9      
        self.fb = cosmo_params['Ob0'] / cosmo_params['Om0']

        vmap_func1 = vmap(self.get_rho_gas, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.rho_gas_mat_physical = vmap_func3(jnp.arange(nr), jnp.arange(nz), jnp.arange(nM)).T        

        vmap_func1 = vmap(self.get_Pth, (0, None, None))
        vmap_func2 = vmap(vmap_func1, (None, 0, None))
        vmap_func3 = vmap(vmap_func2, (None, None, 0))
        self.Pth_mat_physical = vmap_func3(jnp.arange(nr), jnp.arange(nz), jnp.arange(nM)).T        

        self.Pe_mat_physical = self.Pth_mat_physical/1.932
        h = cosmo_params['H0'] / 100.
        self.ne_mat_physical = self.rho_gas_mat_physical/(mue*mp*(Mpc_to_cm**3)) # in cm**-3


    @partial(jit, static_argnums=(0,))
    def get_M_to_R(self, jM, jz, mdef_delta=200):
        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(self.cosmo_jax,self.scale_fac_a_array[jz]) * 1e9
        rho_treshold = mdef_delta * rho_c_z
        R = (self.M_array[jM] * 3.0 / 4.0 / jnp.pi / rho_treshold)**(1.0 / 3.0)
        # convert to comoving coordinates
        # R *= (1 + self.z_array[jz])
        return R

    @partial(jit, static_argnums=(0,1,2))
    def get_params_density(self, key, jM, jz):
        params_dict = self.density_params_def
        Mval = self.M_array[jM] / self.h
        zval = self.z_array[jz]
        A0 = params_dict[key]['A0']
        alpha_m = params_dict[key]['alpha_m']
        alpha_z = params_dict[key]['alpha_z']
        A = A0 * (Mval / 1e14)**alpha_m * (1 + zval)**alpha_z
        return A

    @partial(jit, static_argnums=(0,1,2))
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
        rho_fit = rho0_density * ((x / xc_density)**gamma_density) * (
            1 + (x / xc_density)**alpha_density
            )**(-(beta_density - gamma_density) / alpha_density)
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
        return rho_c_z * rho_fit

    @partial(jit, static_argnums=(0))
    def get_Pth(self, jr, jz, jM):
        P_fit = self.get_P_fit(jr, jz, jM)

        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(self.cosmo_jax,self.scale_fac_a_array[jz]) * 1e9 * self.h**2        
        coeff = (const.G * (const.M_sun**2) / ((1.0 * u.Mpc)**4)).to((u.keV / (u.cm**3))).value

        Mval = self.M_array[jM]
        rDelta = self.r200c_mat[jM, jz]
        P_Delta = coeff * Mval * self.mdef_Delta * rho_c_z * self.fb / (2. * rDelta)

        return P_Delta * P_fit