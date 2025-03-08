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
from colossus.halo import concentration
from colossus.halo import mass_defs
from colossus.cosmology import cosmology
cosmology.setCosmology('WMAP7')

pressure_params_ref = {
    'P0': {
        'A0': 0.694,
        'alpha_m': 0.245,
        'alpha_z': 0.0
        },
    'c500': {
        'A0': 0.986,
        'alpha_m': 0.072,
        'alpha_z': 0.0
        },
    'beta': {
        'A0': 4.512,
        'alpha_m': 0.0,
        'alpha_z': 0.0
        },
    'alpha': {
        'A0':1.489,
        'alpha_m': 0.0,
        'alpha_z': 0.0        
        },
    'gamma': {
        'A0':1.174,
        'alpha_m': 0.0,
        'alpha_z': 0.0        
        }
    }

pressure_params_TAGN8 = {
    'P0': {
        'A0': 0.791,
        'alpha_m': 0.805,
        'alpha_z': 0.0
        },
    'c500': {
        'A0': 0.892,
        'alpha_m': 0.263,
        'alpha_z': 0.0
        },
    'beta': {
        'A0': 4.625,
        'alpha_m': 0.0,
        'alpha_z': 0.0
        },
    'alpha': {
        'A0':1.517,
        'alpha_m': 0.0,
        'alpha_z': 0.0        
        },
    'gamma': {
        'A0':0.814,
        'alpha_m': 0.0,
        'alpha_z': 0.0        
        }
    }


# pressure_params_TAGN8 = {
#     'P0': {
#         'A0': 0.581,
#         'alpha_m': 0.819,
#         'alpha_z': 0.0
#         },
#     'c500': {
#         'A0': 1.035,
#         'alpha_m': 0.273,
#         'alpha_z': 0.0
#         },
#     'beta': {
#         'A0': 3.835,
#         'alpha_m': 0.0,
#         'alpha_z': 0.0
#         },
#     'alpha': {
#         'A0':2.017,
#         'alpha_m': 0.0,
#         'alpha_z': 0.0        
#         },
#     'gamma': {
#         'A0':1.076,
#         'alpha_m': 0.0,
#         'alpha_z': 0.0        
#         }
#     }

# pressure_params_TAGN8p5 = {
#     'P0': {
#         'A0': 0.235,
#         'alpha_m': 0.864,
#         'alpha_z': 0.0
#         },
#     'c500': {
#         'A0': 0.597,
#         'alpha_m': 0.246,
#         'alpha_z': 0.0
#         },
#     'beta': {
#         'A0': 4.85,
#         'alpha_m': 0.0,
#         'alpha_z': 0.0
#         },
#     'alpha': {
#         'A0':1.572,
#         'alpha_m': 0.0,
#         'alpha_z': 0.0        
#         },
#     'gamma': {
#         'A0':0.92,
#         'alpha_m': 0.0,
#         'alpha_z': 0.0        
#         }
#     }

pressure_params_TAGN8p5 = {
    'P0': {
        'A0': 0.21,
        'alpha_m': 0.83,
        'alpha_z': 0.0
        },
    'c500': {
        'A0': 0.7,
        'alpha_m': 0.24,
        'alpha_z': 0.0
        },
    'beta': {
        'A0': 4.1,
        'alpha_m': 0.0,
        'alpha_z': 0.0
        },
    'alpha': {
        'A0':1.8,
        'alpha_m': 0.0,
        'alpha_z': 0.0        
        },
    'gamma': {
        'A0':1,
        'alpha_m': 0.0,
        'alpha_z': 0.0        
        }
    }

pressure_params_def = {'ref':pressure_params_ref, 'agn_8':pressure_params_TAGN8, 'agn_8p5':pressure_params_TAGN8p5}

class LeBrun15(base_class):

    def __init__(
            self,
            sim_params_dict=None,
            halo_params_dict=None,
            pressure_params_def=pressure_params_def
        ):
        '''Note that here M is in Msun/h'''
        super().__init__(sim_params_dict, halo_params_dict, {}, {})
        owls_type = sim_params_dict['owls_type']
        self.pressure_params_def = pressure_params_def[owls_type]
        self.r200c_mat = get_vmapped_func(self.get_M_to_R, 2)(jnp.arange(self.nM), jnp.arange(self.nz)).T
        M500c_mat = np.zeros((self.nM, self.nz))
        R500c_mat = np.zeros((self.nM, self.nz))
        for jz in range(len(self.z_array)):
            c200c = concentration.concentration(self.M_array, '200c', self.z_array[jz], model = 'bullock01')
            M500c, R500c, c500c = mass_defs.changeMassDefinition(self.M_array, c200c, self.z_array[jz], '200c', '500c')
            M500c_mat[:,jz] = M500c
            R500c_mat[:,jz] = R500c/1000.
        self.M500c_mat = jnp.array(M500c_mat)
        self.r500c_mat = jnp.array(R500c_mat)

        
        self.fb = self.cosmo_params['Ob0'] / self.cosmo_params['Om0']
        self.Pth_mat_physical = get_vmapped_func(self.get_Pth, 3)(jnp.arange(self.nr), jnp.arange(self.nz), jnp.arange(self.nM)).T
        self.Pe_mat_physical = self.Pth_mat_physical/1.932
        sigmat = const.sigma_T
        m_e = const.m_e
        c = const.c
        coeff = sigmat / (m_e * (c ** 2))
        oneMpc = (((10 ** 6)) * (u.pc).to(u.m)) * (u.m)
        const_coeff = (((coeff * oneMpc).to(((u.cm ** 3) / u.keV))).value)/(self.cosmo_params['H0']/100.)
        self.y3d_mat = const_coeff * self.Pe_mat_physical


    @partial(jit, static_argnums=(0,))
    def get_M_to_R(self, jM, jz, mdef_delta=200):
        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(self.cosmo_jax,self.scale_fac_a_array[jz]) * 1e9
        rho_treshold = mdef_delta * rho_c_z
        R = (self.M_array[jM] * 3.0 / 4.0 / jnp.pi / rho_treshold)**(1.0 / 3.0)
        return R

    @partial(jit, static_argnums=(0,1))
    def get_params_pressure(self, key, jM, jz):
        params_dict = self.pressure_params_def
        Mval = self.M500c_mat[jM,jz] / self.h
        zval = self.z_array[jz]
        A0 = params_dict[key]['A0']
        alpha_m = params_dict[key]['alpha_m']
        alpha_z = params_dict[key]['alpha_z']
        A = A0 * (Mval / 1e14)**alpha_m * (1 + zval)**alpha_z
        return A


    @partial(jit, static_argnums=(0))
    def get_P_fit(self, jr, jz, jM):
        a = self.scale_fac_a_array[jz]
        x = a * self.r_array[jr] / (self.r500c_mat[jM, jz])
        c500 = self.get_params_pressure('c500', jM, jz)
        P0 = self.get_params_pressure('P0', jM, jz)
        alpha = self.get_params_pressure('alpha', jM, jz)
        beta = self.get_params_pressure('beta', jM, jz)
        gamma = self.get_params_pressure('gamma', jM, jz)
        P_fit = P0/((c500 * x)**gamma * (1 + (c500*x)**alpha)**((beta - gamma)/(alpha)))
        return P_fit


    @partial(jit, static_argnums=(0))
    def get_Pth(self, jr, jz, jM):
        P_fit = self.get_P_fit(jr, jz, jM)

        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(self.cosmo_jax,self.scale_fac_a_array[jz]) * 1e9 * self.h**2        
        coeff = (const.G * (const.M_sun**2) / ((1.0 * u.Mpc)**4)).to((u.keV / (u.cm**3))).value

        Mval = self.M500c_mat[jM,jz]
        rDelta = self.r500c_mat[jM, jz]
        P_Delta = coeff * Mval * 500 * rho_c_z * self.fb / (2. * rDelta)

        return P_Delta * P_fit