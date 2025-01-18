from get_radial_profiles import Profiles
from base_class import get_vmapped_func, get_vmapped_func_warg
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from helpers.jax_cosmo_power import halofit_parameters, nonlinear_matter_power
import jax.scipy.integrate as jsi
import helpers.constants as constants
from mcfitjax.cosmology_jax import xi2P



class get_Pkz(Profiles):
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

        # Do the FFTlog transform of the real-space profiles:
        xi2P_obj = (xi2P(self.r_array, nx=self.nr,lowring=True))
        self.k_mcfit, uk_dmb = xi2P_obj(self.rho_dmb_mat / self.Mtot_mat[None, :, :], axis=0, extrap=False)
        self.uk_dmb_tointp = jnp.array(uk_dmb)
        self.k_mcfit, uk_nfw = xi2P_obj(self.rho_nfw_mat / self.Mtot_mat[None, :, :], axis=0, extrap=False)
        self.uk_nfw_tointp = jnp.array(uk_nfw)

        if self.model_galaxies:
            self.k_mcfit, uk_clm = xi2P_obj(self.rho_clm_mat / self.Mclm_mat[-1, :, :][None, :, :], axis=0, extrap=False)
            self.uk_clm_tointp = jnp.array(uk_clm)
            self.k_mcfit, uk_ne = xi2P_obj(self.ne_mat / self.ne_mat_norm[-1, :, :][None, :, :], axis=0, extrap=False)
            self.uk_ne_tointp = jnp.array(uk_ne)
        else: self.uk_clm_tointp, self.uk_ne_tointp = jnp.zeros((1,1,1)), jnp.zeros((1,1,1))
                        
        if self.model_tSZ:
            self.k_mcfit, uk_y = xi2P_obj(self.y3d_mat, axis=0, extrap=False)
            self.uk_y_tointp = jnp.array(uk_y)
        else: self.uk_y_tointp = jnp.zeros((1,1,1)) 
                       

        # Get the Fourier profiles uk's in the interpolated k array:
        vmapped_func = get_vmapped_func_warg(self.get_uk_from_interp_Pk, 2, 3)
        self.uk_dmb = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 0).T
        self.uk_nfw = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 1).T
        if self.model_tSZ:
            self.uk_y = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 3).T
        # else: self.uk_y = jnp.zeros((self.nk, self.nz, self.nM))
        else: self.uk_y = jnp.zeros((1,1,1))
        if self.model_galaxies:
            self.uk_clm = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 2).T
            self.nbarz = jsi.trapezoid(self.hmf_Mz_mat * (self.Ncen_mat + self.Nsat_mat), jnp.log(self.M_array), axis=-1)
            self.ukg_cross = (self.Ncen_mat[None,:,:] + self.Nsat_mat[None,:,:] * self.uk_clm)/self.nbarz[None,:,None]
            ukg_auto_arg = jnp.clip(jnp.nan_to_num(2 * self.Ncen_mat[None,:,:] * self.Nsat_mat[None,:,:] * self.uk_clm + (self.Nsat_mat[None,:,:] * self.uk_clm)**2), 1e-10, 2e4)
            self.ukg_auto_sqr = (ukg_auto_arg)/(self.nbarz[None,:,None] ** 2)
            self.uk_ne = vmapped_func(jnp.arange(self.nz), jnp.arange(self.nM), 4).T
        # else: self.uk_clm, self.ukg_cross, self.ukg_auto_sqr, self.uk_ne = jnp.zeros((self.nk, self.nz, self.nM)), jnp.zeros((self.nk, self.nz, self.nM)), jnp.zeros((self.nk, self.nz, self.nM)), jnp.zeros((self.nk, self.nz, self.nM))
        else: self.uk_clm, self.ukg_cross, self.ukg_auto_sqr, self.uk_ne = jnp.zeros((1,1,1)), jnp.zeros((1,1,1)), jnp.zeros((1,1,1)), jnp.zeros((1,1,1))

        # Get the halofit power spectra:
        hfit_params = vmap(halofit_parameters,(None, 0))(self.cosmo_jax, self.scale_fac_a_array).T
        self.phfit_kz_mat = vmap(nonlinear_matter_power,(None, None, 0, None, None, None))(self.cosmo_jax, self.kPk_array, self.scale_fac_a_array, self.plin_kz_mat, hfit_params, self.scale_fac_a_array).T

        # Get the large-scale bias of the fields:
        self.bias_Mz_mat = get_vmapped_func(self.get_bias_Mz, 2)(jnp.arange(self.nz), jnp.arange(self.nM)).T

        vmapped_func = get_vmapped_func_warg(self.get_b_2h, 2, 3)
        if self.do_corr_2h_mm:
            bm_largescales_2h = vmap(self.get_bm_largescales_2h)(jnp.arange((self.nz)))
            bm_largescales_2h_mat = jnp.tile(bm_largescales_2h, ((self.nk), 1))

            bm_dmb_2h = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 0).T
            bm_largescales_2h_mat_lt_Mmin = 1. - bm_largescales_2h_mat
            self.bm_dmb_kz_mat = bm_dmb_2h + bm_largescales_2h_mat_lt_Mmin

            bm_nfw_2h = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 1).T
            self.bm_nfw_kz_mat = bm_nfw_2h + bm_largescales_2h_mat_lt_Mmin   
        else:
            self.bm_dmb_kz_mat = jnp.ones((len(self.nk), self.nz))
            self.bm_nfw_kz_mat = jnp.ones((len(self.nk), self.nz))

        if self.model_tSZ:
            by_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 3).T
        else: by_kz_mat = None
        if self.model_galaxies:
            bg_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 2).T
            be_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 4).T
        else: bg_kz_mat, be_kz_mat = None, None

        # Get the 2-halo power:
        self.Pmm_dmb_2h_kz_mat = self.bm_dmb_kz_mat * self.bm_dmb_kz_mat * self.plin_kz_mat
        self.Pmm_nfw_2h_kz_mat = self.bm_nfw_kz_mat * self.bm_nfw_kz_mat * self.plin_kz_mat
        if self.model_tSZ:
            self.Pym_2h_kz_mat = self.bm_dmb_kz_mat * by_kz_mat * self.plin_kz_mat
        if self.model_galaxies:
            self.Pge_2h_kz_mat = bg_kz_mat * be_kz_mat * self.plin_kz_mat
            self.Pgm_2h_kz_mat = bg_kz_mat * self.bm_dmb_kz_mat * self.plin_kz_mat
            self.Pgm_nfw_2h_kz_mat = bg_kz_mat * self.bm_nfw_kz_mat * self.plin_kz_mat
            self.Pgy_2h_kz_mat = by_kz_mat * bg_kz_mat * self.plin_kz_mat
            self.Pgg_2h_kz_mat = bg_kz_mat * bg_kz_mat * self.plin_kz_mat

        # Get the 1-halo power:
        vmapped_func = get_vmapped_func_warg(self.get_P_1h, 2, 4)
        self.Pmm_dmb_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 0, 0).T
        self.Pmm_nfw_1h_kz_mat = vmapped_func(jnp.arange(self.nk), jnp.arange(self.nz), 1, 1).T
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
        self.Pmm_tot_mat = (self.Pmm_dmb_tot_mat) * self.Pmm_sup_tot_mat
        if self.model_tSZ:
            self.Pym_tot_mat = (self.Pym_1h_kz_mat + self.Pym_2h_kz_mat) * self.Pmm_sup_tot_mat
        if self.model_galaxies:
            self.Pge_tot_mat = (self.Pge_1h_kz_mat + self.Pge_2h_kz_mat) * self.Pmm_sup_tot_mat
            self.Pgm_tot_mat = (self.Pgm_1h_kz_mat + self.Pgm_2h_kz_mat) * self.Pmm_sup_tot_mat
            self.Pgm_nfw_tot_mat = (self.Pgm_nfw_1h_kz_mat + self.Pgm_nfw_2h_kz_mat) * self.Pmm_sup_tot_mat
            self.Pgy_tot_mat = (self.Pgy_1h_kz_mat + self.Pgy_2h_kz_mat) * self.Pmm_sup_tot_mat
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