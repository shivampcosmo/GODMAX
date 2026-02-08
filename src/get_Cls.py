from get_Pkzs import get_Pkz
from base_class import get_vmapped_func, get_vmapped_func_warg, EmptyCallable
import jax.numpy as jnp
from jax import jit, vmap
import jax.scipy.integrate as jsi
from functools import partial
from astropy import constants as const
import interpax
from jax_cosmo.scipy.integrate import simps
import time
from astropy import constants as const
import astropy.units as u


class get_Cl(get_Pkz):
    """
    Class that extends the get_Pkz object to compute angular power spectra C(ℓ).

    This class uses the 3D power spectra calculated in get_Pkz and transforms
    them to angular multipole space. It manages beam factors, optionally applies
    smoothing, and stores 2D interpolators for quick evaluation of the angular
    power at given ℓ and redshift.

    Attributes:
        sim_params_dict (dict): Dictionary containing the simulation parameters.
        halo_params_dict (dict): Dictionary containing the halo model parameters.
        analysis_dict (dict): Dictionary containing the analysis parameters.
        other_params_dict (dict): Dictionary containing other parameters.
        Pkz_obj (get_Pkz): If provided, the attributes from this object are
            copied into the get_Cl instance.
    """    
    def __init__(
                self,
                sim_params_dict: dict,
                halo_params_dict: dict,
                analysis_dict: dict,     
                other_params_dict: dict,
                Pkz_obj=None
            ):    
        if Pkz_obj is None:
            super().__init__(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict)
        else:
            self.__dict__.update(Pkz_obj.__dict__)

        # Convert the 3D power spectra multipole space Pkz_ell:
        vmapped_func = get_vmapped_func_warg(self.get_P_lz, 2, 3)
        self.Pkmm_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pmm_tot_mat).T
        self.Pkmm_nfw_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.phfit_kz_mat).T 
        if self.model_tSZ:       
            self.Pkym_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pym_tot_mat).T
            Bl_array = jnp.exp(-1. * self.ell_array * (self.ell_array + 1) * (self.sig_beam ** 2) / 2.)
            self.Bl_mat = Bl_array[:, None]
            self.Pkym_lz_mat = self.Pkym_lz_mat * self.Bl_mat
        if self.model_galaxies:
            self.Pkge_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pge_tot_mat).T
            self.Pkgm_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pgm_tot_mat).T
            self.Pkgm_nfw_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pgm_nfw_tot_mat).T
            self.Pkgy_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pgy_tot_mat).T
            self.Pkgy_lz_mat = self.Pkgy_lz_mat * self.Bl_mat
            self.Pkgg_lz_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nz), self.Pgg_tot_mat).T

        # Get the interpolators:
        self.cached_power_spectra = jnp.zeros((5, 5, self.nell, self.nz_for_Cls))
        log_ell = jnp.log(self.ell_array)
        self.logPkmmlz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkmm_lz_mat), extrap=True)        
        self.cached_power_spectra = self.cached_power_spectra.at[0,0].set(vmap(lambda l: jnp.exp(self.logPkmmlz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
        self.logPkmm_nfw_lz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkmm_nfw_lz_mat), extrap=True)        
        self.cached_power_spectra = self.cached_power_spectra.at[1,1].set(vmap(lambda l: jnp.exp(self.logPkmm_nfw_lz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
        if self.model_tSZ:
            self.logPkymlz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkym_lz_mat), extrap=True)                
            self.cached_power_spectra = self.cached_power_spectra.at[0,3].set(vmap(lambda l: jnp.exp(self.logPkymlz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
            self.cached_power_spectra = self.cached_power_spectra.at[3,0].set(self.cached_power_spectra[0,3])
        else: self.logPkymlz_2d_interp = EmptyCallable()
        if self.model_galaxies:
            self.logPkgmlz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkgm_lz_mat), extrap=True)        
            self.cached_power_spectra = self.cached_power_spectra.at[2,0].set(vmap(lambda l: jnp.exp(self.logPkgmlz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
            self.cached_power_spectra = self.cached_power_spectra.at[0,2].set(self.cached_power_spectra[2,0])
            self.logPkgglz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkgg_lz_mat), extrap=True)        
            self.cached_power_spectra = self.cached_power_spectra.at[2,2].set(vmap(lambda l: jnp.exp(self.logPkgglz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
            self.logPkgylz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkgy_lz_mat), extrap=True)         
            self.cached_power_spectra = self.cached_power_spectra.at[2,3].set(vmap(lambda l: jnp.exp(self.logPkgylz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
            self.cached_power_spectra = self.cached_power_spectra.at[3,2].set(self.cached_power_spectra[2,3])
            self.logPkgm_nfw_lz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkgm_nfw_lz_mat), extrap=True)        
            self.cached_power_spectra = self.cached_power_spectra.at[2,1].set(vmap(lambda l: jnp.exp(self.logPkgm_nfw_lz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
            self.cached_power_spectra = self.cached_power_spectra.at[1,2].set(self.cached_power_spectra[2,1])

            self.logPkge_lz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), self.z_array, jnp.log(self.Pkge_lz_mat), extrap=True)        
            self.cached_power_spectra = self.cached_power_spectra.at[2,4].set(vmap(lambda l: jnp.exp(self.logPkge_lz_2d_interp(l, self.z_array_for_Cls)))(log_ell))
            self.cached_power_spectra = self.cached_power_spectra.at[4,2].set(self.cached_power_spectra[2,4])

        else: self.logPkgmlz_2d_interp, self.logPkgglz_2d_interp, self.logPkgylz_2d_interp, self.logPkgm_nfw_lz_2d_interp, self.logPkge_lz_2d_interp = EmptyCallable(), EmptyCallable(), EmptyCallable(), EmptyCallable(), EmptyCallable()

        # Get the window functions for different probes:
        if self.is_cmb_lensing:
            self.Wk_mat = get_vmapped_func(self.get_cmb_lensing_kernel, 2)(jnp.arange(self.nbins), jnp.arange(self.nz_for_Cls)).T
        else:
            self.pzs_inp_mat = vmap(self.get_photoz_biased_nz)(jnp.arange(self.nbins))
            self.Wk_gravonly_mat = get_vmapped_func(self.get_weak_lensing_kernel, 2)(jnp.arange(self.nbins), jnp.arange(self.nz_for_Cls)).T
            self.nla_mat = get_vmapped_func(self.get_nla_kernel, 2)(jnp.arange(self.nbins), jnp.arange(self.nz_for_Cls)).T        
            self.Wk_mat = self.Wk_gravonly_mat + self.nla_mat
        self.Wy_array = (1.0 / (1.0 + self.z_array_for_Cls))
        oneMpc = (((10 ** 6)) * (u.pc).to(u.m)) * (u.m)
        self.const_coeff_tau = (((const.sigma_T * oneMpc).to(u.cm ** 3)).value)/(self.cosmo_params['H0']/100.)
        self.Wtau_array = self.const_coeff_tau * (1.0 / (1.0 + self.z_array_for_Cls))

        if self.model_galaxies:
            self.Wg_mat = vmap(self.get_nz_lens_interp)(jnp.arange(self.nbins_lens))
        else: self.Wg_mat = jnp.zeros((1,1))

        if self.ENABLE_TIMING:
            ti = time.time()
        # Get the Cls:
        # vmapped_func = get_vmapped_func_warg(self.get_Cl_tot, 3, 5)
        # self.Cl_kappa_kappa_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins), jnp.arange(self.nbins), 0, 0).T

        vmapped_func = get_vmapped_func_warg(self.get_Cl_tot, 2, 4)
        self.Cl_kappa_kappa_tot_mat = vmapped_func(jnp.arange(self.nbins), jnp.arange(self.nbins), 0, 0).T


        # self.Cl_kappa_kappa_nfw_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins), jnp.arange(self.nbins), 1, 1).T
        if self.ENABLE_TIMING:
            print("Time to compute the kappa kappa: ", time.time() - ti)
            ti = time.time()

        if self.model_galaxies:
            # self.Cl_gal_gal_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins_lens), jnp.arange(self.nbins_lens), 2, 2).T
            self.Cl_gal_gal_tot_mat = vmapped_func(jnp.arange(self.nbins_lens), jnp.arange(self.nbins_lens), 2, 2).T
            if self.ENABLE_TIMING:
                print("Time to compute the gal gal: ", time.time() - ti)
                ti = time.time()
            
            # self.Cl_gal_kappa_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins_lens), jnp.arange(self.nbins), 2, 0).T
            self.Cl_gal_kappa_tot_mat = vmapped_func(jnp.arange(self.nbins_lens), jnp.arange(self.nbins), 2, 0).T
            if self.ENABLE_TIMING:
                print("Time to compute the kappa gal: ", time.time() - ti)
                ti = time.time()
            # self.Cl_gal_kappa_nfw_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins_lens), jnp.arange(self.nbins), 2, 1).T
            # Get the Pge in the given k-array:
            self.Pge_zarray = vmap(self.get_Pge_interpz)(jnp.arange(self.nk))
            self.Pge_tot_mat = vmap(self.get_Pge_tot_ks)(jnp.arange(self.nbins_lens))

            

            if self.ENABLE_TIMING:
                print("Time to compute the Pge in the k-array: ", time.time() - ti)
                ti = time.time()
        if self.model_tSZ:
            # vmapped_func = get_vmapped_func_warg(self.get_Cl_tot, 2, 5)
            vmapped_func = get_vmapped_func_warg(self.get_Cl_tot, 1, 4)

            # self.Cl_kappa_y_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins), 0, 0, 3).T
            self.Cl_kappa_y_tot_mat = vmapped_func(jnp.arange(self.nbins), 0, 0, 3).T
            if self.ENABLE_TIMING:
                print("Time to compute the kappa y: ", time.time() - ti)
                ti = time.time()
            if self.model_galaxies:
                # self.Cl_gal_y_tot_mat = vmapped_func(jnp.arange(self.nell), jnp.arange(self.nbins_lens), 0, 2, 3).T        
                self.Cl_gal_y_tot_mat = vmapped_func(jnp.arange(self.nbins_lens), 0, 2, 3).T

                self.Cl_gal_tau_tot_mat = vmapped_func(jnp.arange(self.nbins_lens), 0, 2, 4).T

                if self.ENABLE_TIMING:
                    print("Time to compute the gal y: ", time.time() - ti)
        # if self.ENABLE_TIMING:
        #     print("Time to compute the angular power spectra: ", time.time() - ti)
        #     ti = time.time()

    @partial(jit, static_argnums=(0,))
    def get_P_lz(self, jl, jz, Pk_mat):
        """
        Compute the projected angular power spectrum for a given ℓ and redshift index.

        This method takes the 3D power spectrum (Pk_mat) at a given redshift bin jz,
        converts the target multipole ℓ into a corresponding wavenumber k, and then
        interpolates the 3D power spectrum at that k to get the angular power value.

        Args:
            jl (int): The index of the multipole ℓ in self.ell_array.
            jz (int): The index of the redshift in self.chi_array.
            Pk_mat (jax.numpy.DeviceArray): The 3D power spectrum array, with shape
                [n_k, n_z].

        Returns:
            jax.numpy.DeviceArray: The projected angular power spectrum at the
            specified multipole ℓ and redshift jz.
        """        
        ell = self.ell_array[jl]
        chi_z = self.chi_array[jz]
        k_ell = (ell + 0.5)/jnp.clip(chi_z, 1.0)
        Pkz_ell = jnp.exp(jnp.interp(jnp.log(k_ell), jnp.log(self.kPk_array), jnp.log(Pk_mat[:,jz])))
        return Pkz_ell

    @partial(jit, static_argnums=(0,))
    def get_P_lz_for_Cls(self, jl, jz, Pk_interp_obj):
        value = jnp.exp(Pk_interp_obj(jnp.log(self.ell_array[jl]), self.z_array_for_Cls[jz]))        
        return value 
    
    @partial(jit, static_argnums=(0,))
    def get_photoz_biased_nz(self, jb):
        """
        Returns a photo-z shift parameter biased n(z)
        """
        val_biased = jnp.interp(self.z_array_nz - self.Delta_z_bias_array[jb], self.z_array_nz, self.pzs_inp_mat_inp[jb, :])
        norm_val = jsi.trapezoid(val_biased, x=self.z_array_nz)
        return val_biased / norm_val

    @partial(jit, static_argnums=(0,))
    def get_nz_lens_interp(self, jb):
        nz_jb = self.pzs_inp_mat_inp_lens[jb,:]
        nz_interp = jnp.interp(self.z_array_for_Cls, self.z_array_nz_lens, nz_jb)
        norm_val = jsi.trapezoid(nz_interp, x=self.z_array_for_Cls)
        return nz_interp / norm_val

    @partial(jit, static_argnums=(0,))
    def get_cmb_lensing_kernel(self, jb, jz):
        """
        Returns a weak lensing kernel

        Note: this function handles differently nzs that correspond to extended redshift
        distribution, and delta functions.
        """
        z = self.z_array_for_Cls[jz]
        chi = self.chi_array_for_Cls[jz]

        radial_kernel_cmb = jnp.clip(self.chi_CMB - chi, 0) / jnp.clip(self.chi_CMB, 0.1)
        constant_factor_cmb = 3.0 * (100.)**2 * self.cosmo_jax.Omega_m / (2.0 * ((const.c.value * 1e-3)**2))
        return constant_factor_cmb * radial_kernel_cmb * (1.0 + z) * chi


    @partial(jit, static_argnums=(0,))
    def get_weak_lensing_kernel(self, jb, jz):
        """
        Returns a weak lensing kernel

        Note: this function handles differently nzs that correspond to extended redshift
        distribution, and delta functions.
        """
        z = self.z_array_for_Cls[jz]
        chi = self.chi_array_for_Cls[jz]

        @vmap
        def integrand(z_prime):
            chi_prime = jnp.exp(jnp.interp(z_prime, self.z_array, jnp.log(self.chi_array)))
            dndz = (jnp.interp(z_prime, self.z_array_nz, self.pzs_inp_mat[jb, :]))
            return dndz * jnp.clip(chi_prime - chi, 0) / jnp.clip(chi_prime, 0.1)

        radial_kernel = simps(integrand, z, self.zmax, 128) * (1.0 + z) * chi

        constant_factor = 3.0 * (100.)**2 * self.cosmo_jax.Omega_m / (2.0 * ((const.c.value * 1e-3)**2))
        return constant_factor * radial_kernel

    @partial(jit, static_argnums=(0,))
    def get_nla_kernel(self, jb, jz):
        """
        Computes the NLA IA kernel
        """
        z = self.z_array_for_Cls[jz]
        Dz = self.growth_array_for_Cls[jz]
        # Az_IA = -1. * self.A_IA * self.rho_m_bar * self.C1_bar * (1. / Dz) * ((1. + z) / (1. + self.z0_IA))**self.eta_IA
        Az_IA = -1. * self.A_IA * self.C1_rho_m_bar * (1. / Dz) * ((1. + z) / (1. + self.z0_IA))**self.eta_IA        
        # dchi_dz = (const.c.to(u.km / u.s)).value / (bkgrd.H(self.cosmo_jax, z2a(z)))
        dchi_dz = self.dchi_dz_array_for_Cls[jz]
        dndz = (jnp.interp(z, self.z_array_nz, self.pzs_inp_mat[jb, :]))
        return Az_IA * dndz / dchi_dz

    @partial(jit, static_argnums=(0,))
    def get_Cl_tot(self, jb1, jb2, probe1, probe2): 
        """
        Compute the total angular power spectrum C(ℓ) for given multipole index jl,
        lens/source bins jb1, jb2, and probe types probe1, probe2.

        This function determines prefactors based on the probe type, selects the
        appropriate 2D interpolator for the power spectrum, and applies the
        line-of-sight integration or weighting factors as needed.

        Args:
            jl (int): Index of the multipole ℓ in self.ell_array.
            jb1 (int): First index for lens/source bin.
            jb2 (int): Second index for lens/source bin.
            probe1 (int): Integer code selecting the first field (e.g., shear, galaxies, tSZ).
            probe2 (int): Integer code selecting the second field.

        Returns:
            jax.numpy.DeviceArray: The total angular power spectrum C(ℓ) for the
            specified multipole index, bins, and probes.
        """        

        # Handle the first probe condition
        @jit
        def compute_prefac(probe, jb):
            conditions = [
                (probe == 0, (1. + self.mult_shear_bias_array[jb]) * (self.Wk_mat[jb] / (self.chi_array_for_Cls**2))),
                (probe == 1, (1. + self.mult_shear_bias_array[jb]) * (self.Wk_mat[jb] / (self.chi_array_for_Cls**2))),
                (probe == 2, self.Wg_mat[jb] / (self.dchi_dz_array_for_Cls * self.chi_array_for_Cls**2)),
                (probe == 3, self.Wy_array / (self.chi_array_for_Cls**2)),
                (probe == 4, self.Wtau_array / (self.chi_array_for_Cls**2))
            ]
            
            # Default value if no condition matches
            prefac = jnp.nan
            for condition, value in conditions:
                prefac = jnp.where(condition, value, prefac)
            return prefac

        # Compute prefactors for probe1 and probe2
        prefac_for_uk1 = compute_prefac(probe1, jb1)
        prefac_for_uk2 = compute_prefac(probe2, jb2)        

        Pk = self.cached_power_spectra[probe1, probe2]
        fx = prefac_for_uk1 * prefac_for_uk2  * (self.chi_array_for_Cls ** 2) * self.dchi_dz_array_for_Cls * Pk
        return jsi.trapezoid(fx, x=self.z_array_for_Cls)     

    @partial(jit, static_argnums=(0,))
    def get_Pge_interpz(self, jk):
        return jnp.interp(self.z_array_for_Cls, self.z_array, self.Pge_tot_mat[jk,:])

    @partial(jit, static_argnums=(0,))
    def get_Pge_tot_ks(self, jb):
        """
        Get the galaxy-electron cross power spectrum for a given lens bin jb
        """
        fx_intz = jsi.trapezoid(self.Pge_zarray * self.Wg_mat[jb][None,:], x=self.z_array_for_Cls)
        return jnp.exp(jnp.interp(jnp.log(self.k_array_survey), jnp.log(self.kPk_array), jnp.log(fx_intz + 1e-40)))