import numpy as np
from scipy.special import spence
from scipy.optimize import fmin, differential_evolution, minimize
from scipy.optimize import newton
import scipy as sp
import scipy.interpolate as interpolate
from colossus.cosmology import cosmology
import astropy.units as u
from astropy import constants as const
from colossus.halo import mass_so



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
            M,
            z,
            cosmo=None,
            pressure_params_def=pressure_params_def,
            density_params_def=density_params_def,
            mdef_Delta=200
        ):
        '''Note that here M is in Msun/h'''
        if cosmo is None:
            cosmo = cosmology.setCosmology('planck18')
        self.cosmo = cosmo
        self.h = cosmo.H0 / 100.
        self.M = M / self.h
        # print('M = ', self.M)
        self.z = z
        self.mdef_Delta = mdef_Delta
        mdef = str(mdef_Delta) + 'c'
        self.pressure_params_def = pressure_params_def
        self.density_params_def = density_params_def
        self.rDelta = mass_so.M_to_R(M, z, mdef) / (1000. * self.h)
        self.rho0_density = self.get_params('rho0', density_params_def)
        self.alpha_density = self.get_params('alpha', density_params_def)
        self.beta_density = self.get_params('beta', density_params_def)
        self.xc_density = 0.5
        self.gamma_density = -0.2

        self.P0_pressure = self.get_params('P0', pressure_params_def)
        self.xc_pressure = self.get_params('xc', pressure_params_def)
        self.beta_pressure = self.get_params('beta', pressure_params_def)
        self.alpha_pressure = 1.0
        self.gamma_pressure = -0.3

        # self.rho_crit_z = cosmo.rho_crit(z) * 1e9 * h**2
        self.rho_crit_z = cosmo.rho_c(z) * 1e9 * self.h**2
        # self.rho_crit_z = cosmo.rho_c(z) * 1e9      
        self.fb = cosmo.Ob0 / cosmo.Om0

    def get_params(self, key, params_dict):
        A0 = params_dict[key]['A0']
        alpha_m = params_dict[key]['alpha_m']
        alpha_z = params_dict[key]['alpha_z']
        A = A0 * (self.M / 1e14)**alpha_m * (1 + self.z)**alpha_z
        return A

    def get_rho_fit(self, r):
        x = r / (self.rDelta * self.h)
        rho_fit = self.rho0_density * ((x / self.xc_density)**self.gamma_density) * (
            1 + (x / self.xc_density)**self.alpha_density
            )**(-(self.beta_density - self.gamma_density) / self.alpha_density)
        return rho_fit

    def get_P_fit(self, r):
        x = r / (self.rDelta * self.h)
        P_fit = self.P0_pressure * (x / self.xc_pressure
                          )**self.gamma_pressure * (1 +
                                                    (x / self.xc_pressure)**self.alpha_pressure)**(-self.beta_pressure)
        return P_fit

    def get_rho_gas(self, r):
        rho_fit = self.get_rho_fit(r)
        return self.rho_crit_z * rho_fit

    def get_Pth(self, r):
        P_fit = self.get_P_fit(r)
        coeff = (const.G * (const.M_sun**2) / ((1.0 * u.Mpc)**4)).to((u.eV / (u.cm**3))).value

        P_Delta = coeff * self.M * self.mdef_Delta * self.rho_crit_z * self.fb / (2. * self.rDelta)

        return P_Delta * P_fit