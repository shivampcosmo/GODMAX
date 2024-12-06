from setup_power_spectra import setup_power_BCMP
# from setup_power_spectra_NO_CONC_jit import setup_power_BCMP_NO_CONC
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from jax import vmap
import jax.scipy.integrate as jsi
from jax_cosmo import Cosmology
from functools import partial
import astropy.units as u
from astropy import constants as const
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_Mpc_s_units = const.G.to(u.Mpc**3 / ((u.s**2) * u.M_sun))
import time
import interpax
import jax_cosmo.background as bkgrd
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.utils import z2a

class get_power_BCMP:
    def __init__(
                self,
                sim_params_dict,
                halo_params_dict,
                analysis_dict,
                other_params_dict,
                num_points_trapz_int=64,
                setup_power_BCMP_obj=None,
                verbose_time=False
            ):    

        if verbose_time:
            t0 = time.time()

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

        self.conc_dep_model = analysis_dict.get('conc_dep_model',False)
        if verbose_time:
            ti = time.time()
        if setup_power_BCMP_obj is None:
            setup_power_BCMP_obj = setup_power_BCMP(sim_params_dict, halo_params_dict, analysis_dict, num_points_trapz_int=num_points_trapz_int, verbose_time=verbose_time)

        if verbose_time:
            print('Time for setup_power_BCMP: ', time.time() - ti)
            ti = time.time()

        self.calc_nfw_only = analysis_dict['calc_nfw_only']
        self.r_array = setup_power_BCMP_obj.r_array
        self.M_array = setup_power_BCMP_obj.M_array
        z_array_orig = setup_power_BCMP_obj.z_array
        self.z_array_orig = z_array_orig
        scale_fac_a_array_orig = 1./(1. + z_array_orig)
        # self.conc_array = setup_power_BCMP_obj.conc_array
        self.nM, nz_orig = len(self.M_array), len(z_array_orig)
        chi_array = setup_power_BCMP_obj.chi_array
        dchi_dz_array = (const.c.value * 1e-3) / bkgrd.H(self.cosmo_jax, scale_fac_a_array_orig)
        growth_array = setup_power_BCMP_obj.growth_array

        self.ell_array = setup_power_BCMP_obj.ell_array

        self.logPkmmlz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), z_array_orig, jnp.log(setup_power_BCMP_obj.Pkmm_lz_mat), extrap=True)        
        self.logPkymlz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), z_array_orig, jnp.log(setup_power_BCMP_obj.Pkym_lz_mat), extrap=True)                
        self.logPkgglz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), z_array_orig, jnp.log(setup_power_BCMP_obj.Pkgg_lz_mat), extrap=True)        
        self.logPkgmlz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), z_array_orig, jnp.log(setup_power_BCMP_obj.Pkgm_lz_mat), extrap=True)        
        self.logPkgylz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), z_array_orig, jnp.log(setup_power_BCMP_obj.Pkgy_lz_mat), extrap=True)                

        self.nell = len(self.ell_array)

        self.k_array_survey = analysis_dict.get('k_array_survey', setup_power_BCMP_obj.k)
        self.zmin_pk = analysis_dict.get('zmin_pk', 0.01)
        self.zmax_pk = analysis_dict.get('zmax_pk', 1.6)
        self.nz = analysis_dict.get('nz_pk', 128)
        self.z_array = jnp.linspace(self.zmin_pk, self.zmax_pk, self.nz)
        self.scale_fac_a_array = 1./(1. + self.z_array)
        self.chi_array = jnp.exp(jnp.interp(self.z_array, z_array_orig, jnp.log(chi_array)))
        self.dchi_dz_array = jnp.exp(jnp.interp(self.z_array, z_array_orig, jnp.log(dchi_dz_array)))
        self.growth_array = jnp.exp(jnp.interp(self.z_array, z_array_orig, jnp.log(growth_array)))

        self.k_array = setup_power_BCMP_obj.k
        self.nk = len(self.k_array)
        self.Pge_orig = setup_power_BCMP_obj.Pge_tot_kz_mat
        self.Pge_zarray = vmap(self.get_Pge_interpz)(jnp.arange(self.nk))


        vmap_func1 = vmap(self.get_Pmm_interp, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pkmm_lz_mat = vmap_func2(jnp.arange(self.nell), jnp.arange(self.nz)).T

        vmap_func1 = vmap(self.get_Pym_interp, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pkym_lz_mat = vmap_func2(jnp.arange(self.nell), jnp.arange(self.nz)).T

        vmap_func1 = vmap(self.get_Pgg_interp, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pkgg_lz_mat = vmap_func2(jnp.arange(self.nell), jnp.arange(self.nz)).T

        vmap_func1 = vmap(self.get_Pgm_interp, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pkgm_lz_mat = vmap_func2(jnp.arange(self.nell), jnp.arange(self.nz)).T

        vmap_func1 = vmap(self.get_Pgy_interp, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Pkgy_lz_mat = vmap_func2(jnp.arange(self.nell), jnp.arange(self.nz)).T
        
        nz_info_dict = analysis_dict['nz_source_info_dict']
        self.nbins = nz_info_dict['nbins']
        self.z_array_nz = jnp.array(nz_info_dict['z_array_source'])
        self.zmax = self.z_array_nz[-1]
        pzs_inp_mat = np.zeros((self.nbins, len(self.z_array_nz)))
        for jb in range(self.nbins):
            pzs_inp_mat[jb, :] = nz_info_dict['nz' + str(jb)]
        self.pzs_inp_mat_inp = jnp.array(pzs_inp_mat)

        if other_params_dict is not None:
            self.A_IA = other_params_dict['A_IA']
            self.eta_IA = other_params_dict['eta_IA']
            self.z0_IA = other_params_dict['z0_IA']
            self.C1_bar = other_params_dict['C1_rhocrit']
            H0 = 100. * (u.km / (u.s * u.Mpc))
            # self.rho_m_bar = self.cosmo_params['Om0'] * ((3 * (H0**2) / (8 * np.pi * G_Mpc_s_units)).to(u.M_sun / (u.Mpc**3))).value
            self.C1_rho_m_bar = self.C1_bar * self.cosmo_params['Om0']
            self.Delta_z_bias_array = jnp.array(other_params_dict['Delta_z_bias_array'])
            self.mult_shear_bias_array = jnp.array(other_params_dict['mult_shear_bias_array'])
        else:
            self.A_IA = 0.0
            self.eta_IA = 1.0
            self.z0_IA = 1.0
            self.C1_bar = 1.0
            self.rho_m_bar = 1.0
            self.Delta_z_bias_array = jnp.zeros(self.nbins)
            self.mult_shear_bias_array = jnp.zeros(self.nbins)
        
        self.pzs_inp_mat = vmap(self.get_photoz_biased_nz)(jnp.arange(self.nbins))
            

        if verbose_time:
            ti = time.time()
        vmap_func1 = vmap(self.get_weak_lensing_kernel, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.Wk_gravonly_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nz)).T
        if verbose_time:
            print('Time for computing Wk_mat: ', time.time() - ti)
            ti = time.time()
        
        if verbose_time:
            ti = time.time()
        vmap_func1 = vmap(self.get_nla_kernel, (0, None))
        vmap_func2 = vmap(vmap_func1, (None, 0))
        self.nla_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nz)).T
        if verbose_time:
            print('Time for computing nla_mat: ', time.time() - ti)
            ti = time.time()
        
        self.Wk_mat = self.Wk_gravonly_mat + self.nla_mat

        self.Wy_array = (1.0 / (1.0 + self.z_array))


        nz_info_dict = analysis_dict['nz_lens_info_dict']
        self.nbins_lens = nz_info_dict['nbins_lens']
        self.z_array_nz_lens = jnp.array(nz_info_dict['z_array_lens'])
        self.zmax_lens = self.z_array_nz_lens[-1]
        pzs_inp_mat = np.zeros((self.nbins_lens, len(self.z_array_nz_lens)))
        for jb in range(self.nbins_lens):
            pzs_inp_mat[jb, :] = nz_info_dict['nz' + str(jb)]
        self.pzs_inp_mat_inp_lens = jnp.array(pzs_inp_mat)
        self.pzs_inp_mat_inp_lens = vmap(self.get_nz_lens_interp)(jnp.arange(self.nbins_lens))


        self.Wg_mat = self.pzs_inp_mat_inp_lens

        if analysis_dict['do_ky']:
            vmap_func1 = vmap(self.get_Cl_kappa_y_tot, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Cl_kappa_y_tot_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_y_mat: ', time.time() - ti)
                ti = time.time()

        if analysis_dict['do_kk']:             
            vmap_func1 = vmap(self.get_Cl_kappa_kappa_tot, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.Cl_kappa_kappa_tot_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_kappa_mat: ', time.time() - ti)
                ti = time.time()                

        if analysis_dict['do_gg']:             
            vmap_func1 = vmap(self.get_Cl_gal_gal_tot, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.Cl_gal_gal_tot_mat = vmap_func3(jnp.arange(self.nbins_lens), jnp.arange(self.nbins_lens), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_gg_mat: ', time.time() - ti)
                ti = time.time()   

        if analysis_dict['do_gk']:             
            vmap_func1 = vmap(self.get_Cl_gal_kappa_tot, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.Cl_gal_kappa_tot_mat = vmap_func3(jnp.arange(self.nbins_lens), jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_gk_mat: ', time.time() - ti)
                ti = time.time()  

        if analysis_dict['do_gy']:
            vmap_func1 = vmap(self.get_Cl_gal_y_tot, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Cl_gal_y_tot_mat = vmap_func2(jnp.arange(self.nbins_lens), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_gy_mat: ', time.time() - ti)
                ti = time.time()

        if analysis_dict['do_ge']:
            self.Pge_tot_array_orig = vmap(self.get_Pge_tot)(jnp.arange(self.nbins_lens))
            self.Pge_tot_mat = vmap(self.get_Pge_tot_ks)(jnp.arange(self.nbins_lens))

        self.get_cov = analysis_dict.get('get_cov',False)  
        if self.get_cov:
            self.ukappal_dmb_prefac_mat_tointp = setup_power_BCMP_obj.ukappal_dmb_prefac_mat
            vmap_func1 = vmap(self.get_ukl_interp, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.ukappal_dmb_prefac_mat = vmap_func2(jnp.arange(self.nell), jnp.arange(self.nM)).T
            self.ukappal_dmb_prefac_mat = jnp.moveaxis(self.ukappal_dmb_prefac_mat, 0, 1)
            self.ukappa_l_for_cov = vmap(self.get_ukappa_l_forcov)(jnp.arange(self.nbins))

            self.uyl_mat_tointp = setup_power_BCMP_obj.uyl_mat
            vmap_func1 = vmap(self.get_uyl_interp, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.uyl_mat = vmap_func2(jnp.arange(self.nell), jnp.arange(self.nM)).T
            self.uyl_mat = jnp.moveaxis(self.uyl_mat, 0, 1)
            self.uy_l_for_cov = self.get_uy_l_forcov()

            self.ugl_mat_tointp = setup_power_BCMP_obj.ugl_mat
            vmap_func1 = vmap(self.get_ugl_interp, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.ugl_mat = vmap_func2(jnp.arange(self.nell), jnp.arange(self.nM)).T
            self.ugl_mat = jnp.moveaxis(self.ugl_mat, 0, 1)
            self.ug_l_for_cov = vmap(self.get_ug_l_forcov)(jnp.arange(self.nbins))

            self.hmf_Mz_mat_orig = setup_power_BCMP_obj.hmf_Mz_mat
            self.hmf_Mz_mat = vmap(self.get_hmf_interp)(jnp.arange(self.nM)).T

            self.logPkyylz_2d_interp = interpax.Interpolator2D(jnp.log(self.ell_array), z_array_orig, jnp.log(setup_power_BCMP_obj.Pkyy_lz_mat), extrap=True)                
            vmap_func1 = vmap(self.get_Pyy_interp, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Pkyy_lz_mat = vmap_func2(jnp.arange(self.nell), jnp.arange(self.nz)).T

            vmap_func1 = vmap(self.get_Cl_y_y_tot, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Cl_y_y_tot_mat = vmap_func2(jnp.arange(self.nbins_lens), jnp.arange(self.nell)).T


    @partial(jit, static_argnums=(0,))
    def get_photoz_biased_nz(self, jb):
        """
        Returns a photo-z biased n(z)
        """
        val_biased = jnp.interp(self.z_array_nz - self.Delta_z_bias_array[jb], self.z_array_nz, self.pzs_inp_mat_inp[jb, :])
        norm_val = jsi.trapezoid(val_biased, x=self.z_array_nz)
        value = val_biased / norm_val
        return value


    @partial(jit, static_argnums=(0,))
    def get_nz_lens_interp(self, jb):
        nz_jb = self.pzs_inp_mat_inp_lens[jb,:]
        nz_interp = jnp.interp(self.z_array, self.z_array_nz_lens, nz_jb)
        norm_val = jsi.trapezoid(nz_interp, x=self.z_array)
        value = nz_interp / norm_val
        return value

    @partial(jit, static_argnums=(0,))
    def get_Pge_interpz(self, jk):
        pk_jb = jnp.interp(self.z_array, self.z_array_orig, self.Pge_orig[jk,:])
        return pk_jb


    @partial(jit, static_argnums=(0,))
    def get_weak_lensing_kernel(self, jb, jz):
        """
        Returns a weak lensing kernel

        Note: this function handles differently nzs that correspond to extended redshift
        distribution, and delta functions.
        """
        z = self.z_array[jz]
        chi = self.chi_array[jz]

        @vmap
        def integrand(z_prime):
            chi_prime = jnp.exp(jnp.interp(z_prime, self.z_array, jnp.log(self.chi_array)))
            dndz = (jnp.interp(z_prime, self.z_array_nz, self.pzs_inp_mat[jb, :]))
            return dndz * jnp.clip(chi_prime - chi, 0) / jnp.clip(chi_prime, 0.1)

        radial_kernel = simps(integrand, z, self.zmax, 128) * (1.0 + z) * chi

        H0 = 100.0
        c = const.c.value * 1e-3
        constant_factor = 3.0 * H0**2 * self.cosmo_jax.Omega_m / (2.0 * (c**2))
        return constant_factor * radial_kernel

    @partial(jit, static_argnums=(0,))
    def get_nla_kernel(self, jb, jz):
        """
        Computes the NLA IA kernel
        """
        z = self.z_array[jz]
        Dz = self.growth_array[jz]
        # Az_IA = -1. * self.A_IA * self.rho_m_bar * self.C1_bar * (1. / Dz) * ((1. + z) / (1. + self.z0_IA))**self.eta_IA
        Az_IA = -1. * self.A_IA * self.C1_rho_m_bar * (1. / Dz) * ((1. + z) / (1. + self.z0_IA))**self.eta_IA        
        # dchi_dz = (const.c.to(u.km / u.s)).value / (bkgrd.H(self.cosmo_jax, z2a(z)))
        dchi_dz = self.dchi_dz_array[jz]
        dndz = (jnp.interp(z, self.z_array_nz, self.pzs_inp_mat[jb, :]))
        value = Az_IA * dndz / dchi_dz
        return value     

    @partial(jit, static_argnums=(0,))
    def get_Pmm_interp(self, jl, jz):
        value = jnp.exp(self.logPkmmlz_2d_interp(jnp.log(self.ell_array[jl]), self.z_array[jz]))        
        return value  

    @partial(jit, static_argnums=(0,))
    def get_Pym_interp(self, jl, jz):
        value = jnp.exp(self.logPkymlz_2d_interp(jnp.log(self.ell_array[jl]), self.z_array[jz]))        
        return value  

    @partial(jit, static_argnums=(0,))
    def get_Pgg_interp(self, jl, jz):
        value = jnp.exp(self.logPkgglz_2d_interp(jnp.log(self.ell_array[jl]), self.z_array[jz]))        
        return value  

    @partial(jit, static_argnums=(0,))
    def get_Pgm_interp(self, jl, jz):
        value = jnp.exp(self.logPkgmlz_2d_interp(jnp.log(self.ell_array[jl]), self.z_array[jz]))        
        return value  

    @partial(jit, static_argnums=(0,))
    def get_Pgy_interp(self, jl, jz):
        value = jnp.exp(self.logPkgylz_2d_interp(jnp.log(self.ell_array[jl]), self.z_array[jz]))        
        return value  

    @partial(jit, static_argnums=(0,))
    def get_Pyy_interp(self, jl, jz):
        value = jnp.exp(self.logPkyylz_2d_interp(jnp.log(self.ell_array[jl]), self.z_array[jz]))        
        return value  


    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_y_tot(self, jb, jl):
        """
        Computes the 2-halo term of the cross-spectrum between the convergence of two bins (dmb only).
        """
        Wk_jb = self.Wk_mat[jb]
        prefac_for_uk = Wk_jb/(self.chi_array**2)
        Wy = self.Wy_array
        prefac_for_uy = Wy/(self.chi_array**2)
        
        fx = prefac_for_uk * prefac_for_uy  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pkym_lz_mat[jl]
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return (1. + self.mult_shear_bias_array[jb]) * fx_intz

    @partial(jit, static_argnums=(0,))
    def get_Cl_gal_y_tot(self, jb, jl):
        """
        Computes the 2-halo term of the cross-spectrum between the convergence of two bins (dmb only).
        """
        Wk_jb = self.Wg_mat[jb]
        prefac_for_uk = Wk_jb/(self.dchi_dz_array * self.chi_array**2)
        Wy = self.Wy_array
        prefac_for_uy = Wy/(self.chi_array**2)
        
        fx = prefac_for_uk * prefac_for_uy  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pkgy_lz_mat[jl]
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return fx_intz

    @partial(jit, static_argnums=(0,))
    def get_Cl_y_y_tot(self, jb, jl):
        """
        Computes the 2-halo term of the cross-spectrum between the convergence of two bins (dmb only).
        """
        Wy = self.Wy_array
        prefac_for_uy = Wy/(self.chi_array**2)
        
        fx = prefac_for_uy * prefac_for_uy  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pkyy_lz_mat[jl]
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return fx_intz


    @partial(jit, static_argnums=(0,))
    def get_Cl_gal_kappa_tot(self, jb1, jb2, jl):
        """
        Computes the 2-halo term of the cross-spectrum between the convergence of two bins (dmb only).
        """
        Wk_jb1 = self.Wg_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.dchi_dz_array * self.chi_array**2)
        Wk_jb2 = self.Wk_mat[jb2]
        prefac_for_uk2 = Wk_jb2/(self.chi_array**2)
        
        fx = prefac_for_uk1 * prefac_for_uk2  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pkgm_lz_mat[jl]
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return (1. + self.mult_shear_bias_array[jb2]) * fx_intz

    @partial(jit, static_argnums=(0,))
    def get_Cl_gal_gal_tot(self, jb1, jb2, jl):
        """
        Computes the 2-halo term of the cross-spectrum between the convergence of two bins (dmb only).
        """
        Wk_jb1 = self.Wg_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.dchi_dz_array * self.chi_array**2)
        Wk_jb2 = self.Wg_mat[jb2]
        prefac_for_uk2 = Wk_jb2/(self.dchi_dz_array * self.chi_array**2)
        
        fx = prefac_for_uk1 * prefac_for_uk2  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pkgg_lz_mat[jl]
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return fx_intz

    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_kappa_tot(self, jb1, jb2, jl):
        """
        Computes the 2-halo term of the cross-spectrum between the convergence of two bins (dmb only).
        """
        Wk_jb1 = self.Wk_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)
        Wk_jb2 = self.Wk_mat[jb2]
        prefac_for_uk2 = Wk_jb2/(self.chi_array**2)
        
        fx = prefac_for_uk1 * prefac_for_uk2  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pkmm_lz_mat[jl]
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return (1. + self.mult_shear_bias_array[jb1]) * (1. + self.mult_shear_bias_array[jb2]) * fx_intz

    @partial(jit, static_argnums=(0,))
    def get_Pge_tot(self, jb):
        """
        Computes the 2-halo term of the cross-spectrum between the convergence of two bins (dmb only).
        """
        Wk_jb = self.Wg_mat[jb][None,:]        
        fx = self.Pge_zarray * Wk_jb
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return fx_intz

    @partial(jit, static_argnums=(0,))
    def get_Pge_tot_ks(self, jb):
        """
        Computes the 2-halo term of the cross-spectrum between the convergence of two bins (dmb only).
        """
        value = jnp.exp(jnp.interp(jnp.log(self.k_array_survey), jnp.log(self.k_array), jnp.log(self.Pge_tot_array_orig[jb,:] + 1e-40)))
        return value


    @partial(jit, static_argnums=(0,))
    def get_ukl_interp(self, jl, jM):
        val = jnp.interp(self.z_array, self.z_array_orig, self.ukappal_dmb_prefac_mat_tointp[jl,:,jM])
        return val

    @partial(jit, static_argnums=(0,))
    def get_uyl_interp(self, jl, jM):
        val = jnp.interp(self.z_array, self.z_array_orig, self.uyl_mat_tointp[jl,:,jM])
        return val

    @partial(jit, static_argnums=(0,))
    def get_ugl_interp(self, jl, jM):
        val = jnp.interp(self.z_array, self.z_array_orig, self.ugl_mat_tointp[jl,:,jM])
        return val


    @partial(jit, static_argnums=(0,))
    def get_ukappa_l_forcov(self, jb):
        Wk_jb = self.Wk_mat[jb,:]
        prefac_for_uk = Wk_jb/(self.chi_array**2)
        prefac_for_uk_tile = jnp.tile(prefac_for_uk[None,:,None], (self.ukappal_dmb_prefac_mat.shape[0], 1, self.ukappal_dmb_prefac_mat.shape[2]))
        return prefac_for_uk_tile *  self.ukappal_dmb_prefac_mat
    
    @partial(jit, static_argnums=(0,))
    def get_uy_l_forcov(self):
        Wk_jb = self.Wy_array
        prefac_for_uk = Wk_jb/(self.chi_array**2)
        prefac_for_uk_tile = jnp.tile(prefac_for_uk[None,:,None], (self.uyl_mat.shape[0], 1, self.uyl_mat.shape[2]))
        return prefac_for_uk_tile *  self.uyl_mat

    @partial(jit, static_argnums=(0,))
    def get_ug_l_forcov(self, jb):
        Wk_jb = self.Wg_mat[jb]
        prefac_for_uk = Wk_jb/(self.dchi_dz_array * self.chi_array**2)
        prefac_for_uk_tile = jnp.tile(prefac_for_uk[None,:,None], (self.ugl_mat.shape[0], 1, self.ugl_mat.shape[2]))
        return prefac_for_uk_tile *  self.ugl_mat        
    
    @partial(jit, static_argnums=(0,))
    def get_hmf_interp(self, jM):
        val = jnp.interp(self.z_array, self.z_array_orig, self.hmf_Mz_mat_orig[:,jM])
        return val
