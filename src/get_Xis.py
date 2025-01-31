from get_Cls import get_Cl
from base_class import get_vmapped_func
import jax.numpy as jnp
from jax import jit
from functools import partial
from mcfitjax.transforms import Hankel

class get_xi(get_Cl):
    """
    Class that extends the get_Cl object to compute real-space correlation functions.

    This class uses Hankel transforms (from mcfitjax) to convert the angular power
    spectra computed in get_Cl to correlation functions in real space, such as
    shear-tSZ cross-correlations and shear auto-correlations. It also sets up
    interpolation functions to evaluate these correlation functions at specific
    angular scales.

    Attributes:
        theta_out_arcmin (jax.numpy.DeviceArray): The angular scales in arcminutes
            at which the correlation functions are evaluated.
        gty_out_mat (jax.numpy.DeviceArray): The interpolated shear-tSZ correlation
            function matrix at user-specified angles.
        xip_out_mat (jax.numpy.DeviceArray): The interpolated ξ+ correlation matrix at
            user-specified angles.
        xim_out_mat (jax.numpy.DeviceArray): The interpolated ξ− correlation matrix at
            user-specified angles.

    Methods:
        __init__(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, Cl_obj=None):
            Initialize by either calling the parent constructor (if Cl_obj is None) or
            copying attributes from the passed Cl_obj. Performs Hankel transforms
            to compute correlation functions from the angular power spectra, and
            sets up interpolation for these correlation functions at specific angles.
        
        interp_gty_theta(jt, jb):
            Interpolates the shear-tSZ correlation at a particular angle for a given bin.
        
        interp_xip_theta(jb1, jb2):
            Interpolates the shear-shear ξ+ correlation at specific angles for the
            given bins.
        
        interp_xim_theta(jb1, jb2):
            Interpolates the shear-shear ξ− correlation at specific angles for the
            given bins.
    """

    def __init__(
                self,
                sim_params_dict: dict,
                halo_params_dict: dict,
                analysis_dict: dict,     
                other_params_dict: dict,
                Cl_obj=None
                ):
        if Cl_obj is None:
            super().__init__(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
        else:
            self.__dict__.update(Cl_obj.__dict__)

        theta_out, xi_out = (Hankel(self.ell_array, nu=2, q=1.0, nx=self.nell, lowring=True)(self.Cl_kappa_y_tot_mat, axis=0, extrap=False))
        self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
        self.gty_tot_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))
        self.gty_out_mat = get_vmapped_func(self.interp_gty_theta, 2)(jnp.arange(self.nt_out), jnp.arange(self.nbins)).T

        theta_out, xi_out = (Hankel(self.ell_array, nu=0, q=1.0, nx=self.nell, lowring=True)(self.Cl_kappa_kappa_tot_mat, axis=0, extrap=False))
        self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
        self.xip_tot_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))   
        self.xip_out_mat = get_vmapped_func(self.interp_xip_theta, 2)(jnp.arange(self.nbins), jnp.arange(self.nbins)).T

        theta_out, xi_out = (Hankel(self.ell_array, nu=4, q=1.0, nx=self.nell, lowring=True)(self.Cl_kappa_kappa_tot_mat, axis=0, extrap=False))
        self.theta_out_arcmin = theta_out * (180. / jnp.pi) * 60.
        self.xim_tot_mat = jnp.array(xi_out * (1 / (2 * jnp.pi)))
        self.xim_out_mat = get_vmapped_func(self.interp_xim_theta, 2)(jnp.arange(self.nbins), jnp.arange(self.nbins)).T

    @partial(jit, static_argnums=(0,))
    def interp_gty_theta(self, jt, jb):
        gty_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array[jt]), jnp.log(self.theta_out_arcmin), jnp.log(self.gty_tot_mat[:, jb])))
        return gty_out

    @partial(jit, static_argnums=(0,))
    def interp_xip_theta(self, jb1, jb2):
        xip_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xip_tot_mat[:, jb1,jb2])))
        return xip_out

    @partial(jit, static_argnums=(0,))
    def interp_xim_theta(self, jb1, jb2):
        xim_out = jnp.exp(jnp.interp(jnp.log(self.angles_data_array), jnp.log(self.theta_out_arcmin), jnp.log(self.xim_tot_mat[:, jb1,jb2])))
        return xim_out
