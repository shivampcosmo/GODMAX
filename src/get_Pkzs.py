from get_radial_profiles import Profiles
from base_class import get_vmapped_func, get_vmapped_func_warg
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from helpers.jax_cosmo_power import halofit_parameters, nonlinear_matter_power
import jax.scipy.integrate as jsi
import helpers.constants as constants
from mcfitjax.cosmology_jax import xi2P
from matter_pk_symbolic import *


class get_Pkz(Profiles):
    """
    Compute the total power spectra for matter, tSZ, and galaxy contributions over k and z.
    Sets the class attributes (e.g., Pmm_tot_mat, Pym_tot_mat, Pgg_tot_mat) depending on
    the chosen models (matter, tSZ, galaxy). Includes 1-halo and 2-halo terms, and optionally
    applies 1-halo to 2-halo transition regime corrections.
    
    Returns:
        None
    """    
    def __init__(
                self,
                sim_params_dict: dict,
                halo_params_dict: dict,
                analysis_dict: dict,     
                other_params_dict: dict,
                Profiles_obj=None,
            ):   
        if Profiles_obj is None:
            super().__init__(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
        else:
            self.__dict__.update(Profiles_obj.__dict__)

        # Do the FFTlog transform of the real-space profiles. Normalize each
        # profile by the mass represented on this same radial grid so u(k->0)=1.
        xi2P_obj = (xi2P(self.r_array, nx=self.nr,lowring=True))
        mass_shell_prefac = 4.0 * jnp.pi * self.r_array[:, None, None]**2
        self.Mdmb_grid_mat = jsi.trapezoid(mass_shell_prefac * self.rho_dmb_mat, x=self.r_array, axis=0)
        self.Mnfw_grid_mat = jsi.trapezoid(mass_shell_prefac * self.rho_nfw_mat, x=self.r_array, axis=0)

        self.k_mcfit, uk_dmb = xi2P_obj(self.rho_dmb_mat / jnp.clip(self.Mdmb_grid_mat[None, :, :], 1e-30), axis=0, extrap=False)
        self.uk_dmb_tointp = jnp.array(uk_dmb)
        self.k_mcfit, uk_nfw = xi2P_obj(self.rho_nfw_mat / jnp.clip(self.Mnfw_grid_mat[None, :, :], 1e-30), axis=0, extrap=False)
        self.uk_nfw_tointp = jnp.array(uk_nfw)

        if self.model_galaxies:
            if self.clm_fourier_transform_method == 'legacy_fftlog':
                self.k_mcfit, uk_clm = xi2P_obj(self.rho_clm_mat / self.Mclm_mat[-1, :, :][None, :, :], axis=0, extrap=False)
                self.uk_clm_tointp = jnp.array(uk_clm)
            # Use the electron profile shape here; physical n_e units enter the tau map/projection separately.
            self.k_mcfit, uk_ne = xi2P_obj(self.ne_mat / self.ne_mat_norm[-1, :, :][None, :, :], axis=0, extrap=False)
            self.uk_ne_tointp = jnp.array(uk_ne)
        else: self.uk_clm_tointp, self.uk_ne_tointp = jnp.zeros((1,1,1)), jnp.zeros((1,1,1))
                        
        if self.model_tSZ:
            self.k_mcfit, uk_y = xi2P_obj(self.y3d_mat, axis=0, extrap=False)
            self.uk_y_tointp = jnp.array(uk_y)
        else: self.uk_y_tointp = jnp.zeros((1,1,1)) 

        if self.model_galaxies and self.clm_fourier_transform_method == 'direct_shell':
            k_clm = jnp.concatenate((self.kPk_array, self.k_mcfit))
            uk_clm = self.get_uk_clm_shell(k_clm)
            self.uk_clm = uk_clm[:self.nk]
            self.uk_clm_tointp = uk_clm[self.nk:]
                       

        # Get the Fourier profiles uk's in the interpolated k array:
        vmapped_func = get_vmapped_func_warg(self.get_uk_from_interp_Pk, 2, 3)
        self.uk_dmb = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 0).T
        self.uk_nfw = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 1).T
        if self.model_tSZ:
            self.uk_y = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 3).T
        # else: self.uk_y = jnp.zeros((self.nk, self.nz, self.nM))
        else: self.uk_y = jnp.zeros((1,1,1))
        if self.model_galaxies:
            if self.clm_fourier_transform_method == 'legacy_fftlog':
                self.uk_clm = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 2).T
            self.nbarz = jnp.maximum(jsi.trapezoid(self.hmf_Mz_mat * (self.Ncen_mat + self.Nsat_mat), jnp.log(self.M_array), axis=-1), 1e-10)
            self.ukg_cross = jnp.maximum((self.Ncen_mat[None,:,:] + self.Nsat_mat[None,:,:] * self.uk_clm)/self.nbarz[None,:,None], 1e-10)
            ukg_auto_arg = jnp.maximum(
                jnp.nan_to_num(
                    2 * self.Ncen_mat[None,:,:] * self.Nsat_mat[None,:,:] * self.uk_clm
                    + (self.Nsat_mat[None,:,:] * self.uk_clm)**2,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                1e-10,
            )
            self.ukg_auto_sqr = (ukg_auto_arg)/(self.nbarz[None,:,None] ** 2)
            self.uk_ne = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 4).T
        # else: self.uk_clm, self.ukg_cross, self.ukg_auto_sqr, self.uk_ne = jnp.zeros((self.nk, self.nz, self.nM)), jnp.zeros((self.nk, self.nz, self.nM)), jnp.zeros((self.nk, self.nz, self.nM)), jnp.zeros((self.nk, self.nz, self.nM))
        else: self.uk_clm, self.ukg_cross, self.ukg_auto_sqr, self.uk_ne = jnp.zeros((1,1,1)), jnp.zeros((1,1,1)), jnp.zeros((1,1,1)), jnp.zeros((1,1,1))

        # Get the halofit power spectra:
        if self.symbolic_pk:
            vmap_func = vmap(symbolic_pkhalofit,(None, None, None, None, None, None, None, None, 0))
            self.phfit_kz_mat = vmap_func(self.kPk_array, self.plin_kz_mat, self.Om0, self.cosmo_params['Ob0'], self.h, self.cosmo_params['ns'], self.cosmo_params['sigma8'], self.z_array, jnp.arange(self.nz)).T
        else:
            hfit_params = vmap(halofit_parameters,(None, 0))(self.cosmo_jax, self.scale_fac_a_array).T
            self.phfit_kz_mat = vmap(nonlinear_matter_power,(None, None, 0, None, None, None))(self.cosmo_jax, self.kPk_array, self.scale_fac_a_array, self.plin_kz_mat, hfit_params, self.scale_fac_a_array).T

        # Get the large-scale bias of the fields:
        self.bias_Mz_mat = get_vmapped_func(self.get_bias_Mz, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T

        vmapped_func = get_vmapped_func_warg(self.get_b_2h, 2, 3)
        if self.do_corr_2h_mm:
            bm_largescales_2h = vmap(self.get_bm_largescales_2h)(jnp.arange((self.nz)))
            bm_largescales_2h_mat = jnp.tile(bm_largescales_2h, ((self.nk), 1))

            self.bm_dmb_2h = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 0).T
            self.bm_largescales_2h_mat_lt_Mmin = 1. - bm_largescales_2h_mat
            self.bm_dmb_kz_mat = self.bm_dmb_2h + self.bm_largescales_2h_mat_lt_Mmin

            bm_nfw_2h = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 1).T
            self.bm_nfw_kz_mat = bm_nfw_2h + self.bm_largescales_2h_mat_lt_Mmin   
        else:
            self.bm_dmb_kz_mat = jnp.ones((len(self.nk), self.nz))
            self.bm_nfw_kz_mat = jnp.ones((len(self.nk), self.nz))

        if self.model_tSZ:
            self.by_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 3).T
        else: self.by_kz_mat = None
        if self.model_galaxies:
            self.bg_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 2).T
            self.be_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 4).T
            if self.do_corr_2h_mm:
                self.be_kz_mat = self.be_kz_mat + self.bm_largescales_2h_mat_lt_Mmin
        else: self.bg_kz_mat, self.be_kz_mat = None, None

        # Get the 2-halo power:
        self.Pmm_dmb_2h_kz_mat = self.bm_dmb_kz_mat * self.bm_dmb_kz_mat * self.plin_kz_mat
        self.Pmm_nfw_2h_kz_mat = self.bm_nfw_kz_mat * self.bm_nfw_kz_mat * self.plin_kz_mat
        if self.model_tSZ:
            self.Pym_2h_kz_mat = self.bm_dmb_kz_mat * self.by_kz_mat * self.plin_kz_mat
        if self.model_galaxies:
            self.Pge_2h_kz_mat = self.bg_kz_mat * self.be_kz_mat * self.plin_kz_mat
            self.Pgm_2h_kz_mat = self.bg_kz_mat * self.bm_dmb_kz_mat * self.plin_kz_mat
            self.Pgm_nfw_2h_kz_mat = self.bg_kz_mat * self.bm_nfw_kz_mat * self.plin_kz_mat
            self.Pgy_2h_kz_mat = self.by_kz_mat * self.bg_kz_mat * self.plin_kz_mat
            self.Pgg_2h_kz_mat = self.bg_kz_mat * self.bg_kz_mat * self.plin_kz_mat

        # Get the 1-halo power:
        vmapped_func = get_vmapped_func_warg(self.get_P_1h, 2, 4)
        self.Pmm_dmb_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 0, 0).T
        self.Pmm_nfw_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 1, 1).T

        if self.lowpass_Pmm1h_lowk:
            k_lowpass = self.kthresh_lowpass_Pmm1h_lowk
            self.lowpass_filter = 1 / (1 + (k_lowpass / self.kPk_array[:, None])**4)
            self.Pmm_nfw_1h_kz_mat = self.Pmm_nfw_1h_kz_mat * self.lowpass_filter
            self.Pmm_dmb_1h_kz_mat = self.Pmm_dmb_1h_kz_mat * self.lowpass_filter

        if self.model_tSZ:
            self.Pym_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 0, 3).T
        if self.model_galaxies:
            self.Pge_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 2, 4).T
            self.Pgm_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 0, 2).T
            self.Pgm_nfw_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 1, 2).T
            self.Pgy_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 3, 2).T
            self.Pgg_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 2, 2).T

        # Get the total power:
        self.Pmm_nfw_tot_mat = self.Pmm_nfw_1h_kz_mat + self.Pmm_nfw_2h_kz_mat
        self.Pmm_dmb_tot_mat = self.Pmm_dmb_1h_kz_mat + self.Pmm_dmb_2h_kz_mat        
        self.Pmm_sup_tot_mat = self.phfit_kz_mat / self.Pmm_nfw_tot_mat
        if self.model_matter == 'halofit':
            self.Pmm_tot_mat = self.phfit_kz_mat
        else:
            self.Pmm_tot_mat = (self.Pmm_dmb_tot_mat) * self.Pmm_sup_tot_mat
        if self.model_tSZ:
            self.Pym_tot_mat = self._combine_1h2h_poweradd(self.Pym_1h_kz_mat, self.Pym_2h_kz_mat, self.alpha_ky)
            if self.tSZ_transition_model == 'response':
                self.Pym_tot_mat = self.Pym_tot_mat * self.Pmm_sup_tot_mat
        if self.model_galaxies:
            self.Pge_tot_mat = self._combine_1h2h_poweradd(self.Pge_1h_kz_mat, self.Pge_2h_kz_mat, self.alpha_ge)
            if self.galaxy_electron_transition_model == 'response':
                self.Pge_tot_mat = self.Pge_tot_mat * self.Pmm_sup_tot_mat
            self.Pgm_tot_mat = self._combine_1h2h_poweradd(self.Pgm_1h_kz_mat, self.Pgm_2h_kz_mat, self.alpha_gm)
            if self.galaxy_matter_transition_model == 'response':
                self.Pgm_tot_mat = self.Pgm_tot_mat * self.Pmm_sup_tot_mat
            self.Pgm_nfw_tot_mat = self._combine_1h2h_poweradd(self.Pgm_nfw_1h_kz_mat, self.Pgm_nfw_2h_kz_mat, self.alpha_gm)
            if self.galaxy_matter_transition_model == 'response':
                self.Pgm_nfw_tot_mat = self.Pgm_nfw_tot_mat * self.Pmm_sup_tot_mat
            self.Pgy_tot_mat = self._combine_1h2h_poweradd(self.Pgy_1h_kz_mat, self.Pgy_2h_kz_mat, self.alpha_gy)
            if self.tSZ_transition_model == 'response':
                self.Pgy_tot_mat = self.Pgy_tot_mat * self.Pmm_sup_tot_mat
            self.Pgg_tot_mat = self._combine_1h2h_poweradd(self.Pgg_1h_kz_mat, self.Pgg_2h_kz_mat, self.alpha_gg)
            if self.gg_transition_model == 'response':
                self.Pgg_tot_mat = (self.Pgg_1h_kz_mat + self.Pgg_2h_kz_mat) * self.Pmm_sup_tot_mat



    @partial(jit, static_argnums=(0,))
    def get_uk_from_interp_Pk(self, jz, jM, probe):
        '''Compute uk values based on the probe and interpolate over kPk_array.'''
        
        # Helper function to select uk_val based on the probe
        def compute_uk_val(probe):
            conditions = [
                (probe == 0, jnp.clip(self.uk_dmb_tointp[:, jz, jM], 1e-30, 1)),
                (probe == 1, jnp.clip(self.uk_nfw_tointp[:, jz, jM], 1e-30, 1)),
                (probe == 2, jnp.clip(self.uk_clm_tointp[:, jz, jM], 1e-30, 1)),
                (probe == 3, self.uk_y_tointp[:, jz, jM]),
                (probe == 4, self.uk_ne_tointp[:, jz, jM]),
            ]
            
            # Default value if no condition matches
            uk_val = jnp.nan
            for condition, value in conditions:
                uk_val = jnp.where(condition, value, uk_val)
            return uk_val

        # Compute uk_val based on the probe
        uk_val = compute_uk_val(probe)

        # Perform interpolation in log space for stability
        return jnp.exp(
            jnp.interp(
                jnp.log(self.kPk_array), 
                jnp.log(self.k_mcfit), 
                jnp.log(jnp.clip(uk_val, 1e-30, jnp.inf))
            )
        )
    
    @partial(jit, static_argnums=(0,))
    def get_bias_Mz(self, jz, jM, mdef_delta=200):
        '''Tinker 2010 bias function'''
        sigma = self.sigma_Mz_mat[jz, jM]
        delta_c = constants.DELTA_COLLAPSE
        nu = delta_c / sigma

        z = self.z_array[jz]    
        rho_treshold = mdef_delta * self.get_rho_c(z)
        Delta = rho_treshold / self.get_rho_m(z)
        y = jnp.log10(Delta)

        A = 1.0 + 0.24 * y * jnp.exp(-1.0 * (4.0 / y)**4)
        a = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * jnp.exp(-1.0 * (4.0 / y)**4)
        c = 2.4
        
        bias = 1.0 - A * nu**a / (nu**a + constants.DELTA_COLLAPSE**a) + B * nu**b + C * nu**c
        return bias

    @partial(jit, static_argnums=(0,))
    def compute_ukz(self, jk, jz, probe):
        """
        Compute the field-specific uk values for a given k and redshift bin.

        Depending on the probe type:
            0 -> dark matter-baryon (dmb)
            1 -> NFW halo model
            2 -> galaxies
            3 -> Compton y-parameter
            4 -> electron number density

        Args:
            jk (int): Index for the wavenumber (k).
            jz (int): Index for the redshift bin (z).
            probe (int): Integer code selecting which field to compute uk for.

        Returns:
            ukz (jax.numpy.DeviceArray): The dimensionless clustering amplitude or weighting function
                for the specified probe, evaluated at the given k and z.
        """        
        conditions = [
            (probe == 0, (self.Mtot_mat[jz, :] * self.uk_dmb[jk, jz, :]) / self.rhom_0),
            (probe == 1, (self.Mtot_mat[jz, :] * self.uk_nfw[jk, jz, :]) / self.rhom_0),
            (probe == 2, self.ukg_cross[jk, jz, :]),
            (probe == 3, self.uk_y[jk, jz, :]),
            (probe == 4, (self.Mtot_mat[jz, :] * self.uk_ne[jk, jz, :]) / self.rhom_0),
        ]
        
        # Default value if no condition matches
        ukz = jnp.nan
        for condition, value in conditions:
            ukz = jnp.where(condition, value, ukz)
        return ukz

    
    @partial(jit, static_argnums=(0,))
    def get_b_2h(self, jk, jz, probe):
        '''Function getting the 2halo effective bias of the fields'''
        ukz = self.compute_ukz(jk, jz, probe)
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        fx = ukz * dndlnM_z * self.bias_Mz_mat[jz, :]
        b_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))        
        return b_2h

    
    @partial(jit, static_argnums=(0,))
    def get_bm_largescales_2h(self, jz):
        '''Get the large scale limit of the above 2halo integral'''
        ukz_intc = self.Mtot_mat[jz, :]
        dndlnM_z = self.hmf_Mz_mat[jz, :]     
        fx = ukz_intc * dndlnM_z * self.bias_Mz_mat[jz,:] * (1/self.rhom_0)
        bmm_2h = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        return bmm_2h

    @partial(jit, static_argnums=(0,))
    def get_P_1h(self, jk, jz, probe1, probe2):
        """
        Compute the 1-halo power spectrum for the specified probes.

        This function calculates ukz values for two probes, handles the special case
        when both probes are galaxies (auto-squared), and integrates over the halo mass
        function to obtain the 1-halo contribution.

        Args:
            jk (int): Index for the wavenumber (k).
            jz (int): Index for the redshift bin (z).
            probe1 (int): Integer code selecting the first field.
            probe2 (int): Integer code selecting the second field.

        Returns:
            jax.numpy.DeviceArray: The 1-halo power spectrum for the selected probes.
        """

        # Compute ukz1 and ukz2
        ukz1 = self.compute_ukz(jk, jz, probe1)
        ukz2 = self.compute_ukz(jk, jz, probe2)
        # Handle the special case for both probes being 2 (auto-squared case)
        ukz_sqr = jnp.where(
            jnp.logical_and(probe1 == 2, probe2 == 2), self.ukg_auto_sqr[jk, jz, :], ukz1 * ukz2
        )
        # Compute P_1h using trapezoid integration
        dndlnM_z = self.hmf_Mz_mat[jz, :]
        P_1h = jsi.trapezoid(ukz_sqr * dndlnM_z, x=jnp.log(self.M_array))
        return P_1h


    @staticmethod
    def _combine_1h2h_poweradd(p1, p2, alpha):
        alpha = float(alpha)
        if alpha == 1.0:
            return p1 + p2
        return (jnp.clip(p1, 1e-60, jnp.inf) ** alpha + jnp.clip(p2, 1e-60, jnp.inf) ** alpha) ** (1.0 / alpha)

    @staticmethod
    def _clm_tophat_window(x):
        """Spherical top-hat window with a stable small-x branch."""
        x2 = x * x
        small = jnp.abs(x) < 1e-2
        x_safe = jnp.where(small, 1.0, x)
        exact = 3.0 * (jnp.sin(x_safe) - x_safe * jnp.cos(x_safe)) / x_safe**3
        series = 1.0 - x2/10.0 + x2**2/280.0 - x2**3/15120.0
        return jnp.where(small, series, exact)

    @staticmethod
    def _clm_inner_window(x):
        """Window for the unresolved NFW-cusp cell, M(<r) proportional to r^2."""
        x2 = x * x
        small = jnp.abs(x) < 1e-2
        x_safe = jnp.where(small, 1.0, x)
        exact = 2.0 * (1.0 - jnp.cos(x_safe)) / x_safe**2
        series = 1.0 - x2/12.0 + x2**2/360.0 - x2**3/20160.0
        return jnp.where(small, series, exact)

    @classmethod
    def _clm_shell_window(cls, k_array, r_array):
        """Constant-density shell windows on a radial grid in Mpc/h."""
        k = jnp.atleast_1d(k_array)[:, None]
        rin = r_array[:-1]
        rout = r_array[1:]
        volume = rout**3 - rin**3

        exact = (
            rout[None, :]**3 * cls._clm_tophat_window(k * rout[None, :])
            - rin[None, :]**3 * cls._clm_tophat_window(k * rin[None, :])
        ) / volume[None, :]

        k2 = k * k
        series = (
            1.0
            - k2/10.0 * ((rout**5 - rin**5) / volume)[None, :]
            + k2**2/280.0 * ((rout**7 - rin**7) / volume)[None, :]
            - k2**3/15120.0 * ((rout**9 - rin**9) / volume)[None, :]
        )
        return jnp.where(jnp.abs(k * rout[None, :]) < 1e-2, series, exact)

    def get_uk_clm_shell(self, k_array, Mclm_mat=None):
        """Direct finite-volume CLM transform.

        ``k_array`` is in h/Mpc, ``r_array`` is in Mpc/h, and cumulative CLM
        masses are in Msun/h, so ``kr`` and the returned normalized window are
        dimensionless. Signed shell masses and signed high-k windows are kept.
        """
        enclosed_mass = self.Mclm_mat if Mclm_mat is None else Mclm_mat
        shell_mass = self.get_Mclm_shell_masses(enclosed_mass)
        k = jnp.atleast_1d(k_array)

        inner = self._clm_inner_window(k * self.r_array[0])[:, None]
        outer = self._clm_shell_window(k, self.r_array)
        window = jnp.concatenate((inner, outer), axis=1)

        endpoint = enclosed_mass[-1]
        endpoint_safe = jnp.where(endpoint > 0.0, endpoint, jnp.nan)
        return jnp.einsum('kr,rzm->kzm', window, shell_mass) / endpoint_safe[None, :, :]