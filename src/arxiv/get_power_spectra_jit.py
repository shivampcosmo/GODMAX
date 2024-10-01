from setup_power_spectra_jit import setup_power_BCMP
from setup_power_spectra_NO_CONC_jit import setup_power_BCMP_NO_CONC
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
        self.z_array = setup_power_BCMP_obj.z_array
        self.scale_fac_a_array = 1./(1. + self.z_array)
        self.conc_array = setup_power_BCMP_obj.conc_array
        self.nM, self.nz, self.nc = len(self.M_array), len(self.z_array), len(self.conc_array)
        self.chi_array = setup_power_BCMP_obj.chi_array
        self.dchi_dz_array = (const.c.value * 1e-3) / bkgrd.H(self.cosmo_jax, self.scale_fac_a_array)
        self.hmf_Mz_mat = setup_power_BCMP_obj.hmf_Mz_mat
        self.uyl_mat = jnp.moveaxis(setup_power_BCMP_obj.uyl_mat, 1, 3)
        self.byl_mat = setup_power_BCMP_obj.byl_mat
        self.ukappal_dmb_prefac_mat = jnp.moveaxis(setup_power_BCMP_obj.ukappal_dmb_prefac_mat, 1, 3)
        self.bkl_dmb_mat = setup_power_BCMP_obj.bkl_dmb_mat        
        if self.calc_nfw_only:
            self.ukappal_nfw_prefac_mat = jnp.moveaxis(setup_power_BCMP_obj.ukappal_nfw_prefac_mat, 1, 3)        
            self.bkl_nfw_mat = setup_power_BCMP_obj.bkl_nfw_mat
            
        self.Pklin_lz_mat = setup_power_BCMP_obj.Pklin_lz_mat
        self.ell_array = setup_power_BCMP_obj.ell_array
        self.nell = len(self.ell_array)
        self.growth_array = setup_power_BCMP_obj.growth_array
        self.dchi_dz_array = (const.c.to(u.km / u.s)).value / (bkgrd.H(self.cosmo_jax, setup_power_BCMP_obj.scale_fac_a_array))

        
        nz_info_dict = analysis_dict['nz_info_dict']
        self.nbins = nz_info_dict['nbins']
        self.z_array_nz = jnp.array(nz_info_dict['z_array'])
        self.zmax = self.z_array_nz[-1]
        pzs_inp_mat = np.zeros((self.nbins, len(self.z_array_nz)))
        for jb in range(self.nbins):
            pzs_inp_mat[jb, :] = nz_info_dict['nz' + str(jb)]
        self.pzs_inp_mat_inp = jnp.array(pzs_inp_mat)

        if other_params_dict is not None:
            self.A_IA = other_params_dict['A_IA']
            self.eta_IA = other_params_dict['eta_IA']
            self.z0_IA = other_params_dict['z0_IA']
            self.C1_bar = other_params_dict['C1_bar']
            H0 = 100. * (u.km / (u.s * u.Mpc))
            self.rho_m_bar = self.cosmo_params['Om0'] * ((3 * (H0**2) / (8 * np.pi * G_Mpc_s_units)).to(u.M_sun / (u.Mpc**3))).value
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

        self.logc_array = jnp.log(self.conc_array)
        sig_logc = setup_power_BCMP_obj.sig_logc_z_array
        sig_logc_mat = jnp.tile(sig_logc[:, None, None], (1, self.nM, self.nc))
        conc_mat = jnp.tile(self.conc_array[None, None, :], (self.nz, self.nM, 1))
        cmean_mat = jnp.tile(setup_power_BCMP_obj.conc_Mz_mat[:,:,None], (1, 1, self.nc))
        self.p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_mat)/ sig_logc_mat)**2) * (1.0/(sig_logc_mat * jnp.sqrt(2*jnp.pi)))
        if verbose_time:
            print('Time for computing p_logc_Mz: ', time.time() - ti)
            ti = time.time()

        if analysis_dict['do_sheary']:
            vmap_func1 = vmap(self.get_Cl_kappa_y_1h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Cl_kappa_y_1h_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_y_1h_mat: ', time.time() - ti)
                ti = time.time()

            vmap_func1 = vmap(self.get_Cl_kappa_y_2h, (0, None))
            vmap_func2 = vmap(vmap_func1, (None, 0))
            self.Cl_kappa_y_2h_mat = vmap_func2(jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_y_2h_mat: ', time.time() - ti)
                ti = time.time()

        if analysis_dict.get('do_yy', False):
            self.Cl_y_y_1h_mat = vmap(self.get_Cl_y_y_1h)(jnp.arange(self.nell))
            if verbose_time:
                print('Time for computing Cl_y_y_1h_mat: ', time.time() - ti)
                ti = time.time()

            self.Cl_y_y_2h_mat = vmap(self.get_Cl_y_y_2h)(jnp.arange(self.nell))
            if verbose_time:
                print('Time for computing Cl_y_y_2h_mat: ', time.time() - ti)
                ti = time.time()

        if analysis_dict['do_shear2pt']:
            vmap_func1 = vmap(self.get_Cl_kappa_kappa_1h, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.Cl_kappa_kappa_1h_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_kappa_1h_mat: ', time.time() - ti)
                ti = time.time()

            vmap_func1 = vmap(self.get_Cl_kappa_kappa_2h, (0, None, None))
            vmap_func2 = vmap(vmap_func1, (None, 0, None))
            vmap_func3 = vmap(vmap_func2, (None, None, 0))
            self.Cl_kappa_kappa_2h_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
            if verbose_time:
                print('Time for computing Cl_kappa_kappa_2h_mat: ', time.time() - ti)
                # print('Total time for computing all Cls: ', time.time() - t0)
                ti = time.time()                

            if self.calc_nfw_only:
                vmap_func1 = vmap(self.get_Cl_kappa_kappa_nfw_1h, (0, None, None))
                vmap_func2 = vmap(vmap_func1, (None, 0, None))
                vmap_func3 = vmap(vmap_func2, (None, None, 0))
                self.Cl_kappa_kappa_nfw_1h_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
                if verbose_time:
                    print('Time for computing Cl_kappa_kappa_nfw_1h_mat: ', time.time() - ti)
                    # print('Total time for computing all Cls: ', time.time() - t0)                
                    ti = time.time()

                vmap_func1 = vmap(self.get_Cl_kappa_kappa_nfw_2h, (0, None, None))
                vmap_func2 = vmap(vmap_func1, (None, 0, None))
                vmap_func3 = vmap(vmap_func2, (None, None, 0))
                self.Cl_kappa_kappa_nfw_2h_mat = vmap_func3(jnp.arange(self.nbins), jnp.arange(self.nbins), jnp.arange(self.nell)).T
                if verbose_time:
                    print('Time for computing Cl_kappa_kappa_2h_mat: ', time.time() - ti)
                    print('Total time for computing all Cls: ', time.time() - t0)
                    # ti = time.time()                

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
            return dndz * jnp.clip(chi_prime - chi, 0) / jnp.clip(chi_prime, 1.0)

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
        Az_IA = -1. * self.A_IA * self.rho_m_bar * self.C1_bar * (1. / Dz) * ((1. + z) / (1. + self.z0_IA))**self.eta_IA
        # dchi_dz = (const.c.to(u.km / u.s)).value / (bkgrd.H(self.cosmo_jax, z2a(z)))
        dchi_dz = self.dchi_dz_array[jz]
        dndz = (jnp.interp(z, self.z_array_nz, self.pzs_inp_mat[jb, :]))
        value = Az_IA * dndz / dchi_dz
        return value        

    @partial(jit, static_argnums=(0,))
    def get_Cl_y_y_1h(self, jl):
        """
        Computes the 1-halo term of the auto-spectrum of the Compton-y map.
        """
        uyl_jl = self.uyl_mat[jl, ...]        
        fx = uyl_jl * uyl_jl * self.p_logc_Mz
        fx_intc = jsi.trapezoid(fx, x=self.logc_array)
        fx = fx_intc * self.hmf_Mz_mat
        fx_intM = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        fx = fx_intM * (self.chi_array ** 2) * self.dchi_dz_array
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return fx_intz
    
    @partial(jit, static_argnums=(0,))
    def get_Cl_y_y_2h(self, jl):
        """
        Computes the 2-halo term of the auto-spectrum of the Compton-y map.
        """
        byl_jl = self.byl_mat[jl]
        
        fx = byl_jl * byl_jl * (self.chi_array ** 2) * self.dchi_dz_array * self.Pklin_lz_mat[jl]
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return fx_intz


    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_y_1h(self, jb, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        Wk_jb = self.Wk_mat[jb,:]
        prefac_for_uk = Wk_jb/(self.chi_array**2)
        uyl_jl = self.uyl_mat[jl, ...]
        ukl_jl = self.ukappal_dmb_prefac_mat[jl, ...]
        
        fx = uyl_jl * ukl_jl * self.p_logc_Mz
        fx_intc = jsi.trapezoid(fx, x=self.logc_array)
        fx = fx_intc * self.hmf_Mz_mat
        fx_intM = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        fx = fx_intM * prefac_for_uk  * (self.chi_array ** 2) * self.dchi_dz_array
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return (1. + self.mult_shear_bias_array[jb]) * fx_intz

    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_y_2h(self, jb, jl):
        """
        Computes the 2-halo term of the cross-spectrum between the convergence and the
        Compton-y map.
        """
        Wk_jb = self.Wk_mat[jb]
        prefac_for_uk = Wk_jb/(self.chi_array**2)
        bkl_jl = self.bkl_dmb_mat[jl]
        byl_jl = self.byl_mat[jl]
        
        fx = byl_jl * bkl_jl * prefac_for_uk  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pklin_lz_mat[jl]
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return (1. + self.mult_shear_bias_array[jb]) * fx_intz


    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_kappa_1h(self, jb1, jb2, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence of two bins (dmb only).
        """
        Wk_jb1 = self.Wk_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)
        Wk_jb2 = self.Wk_mat[jb2]
        prefac_for_uk2 = Wk_jb2/(self.chi_array**2)

        ukl_jl = self.ukappal_dmb_prefac_mat[jl]       
        
        fx = ukl_jl * ukl_jl * self.p_logc_Mz
        fx_intc = jsi.trapezoid(fx, x=self.logc_array)
        fx = fx_intc * self.hmf_Mz_mat
        fx_intM = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        fx = fx_intM * prefac_for_uk1 * prefac_for_uk2 * (self.chi_array ** 2) * self.dchi_dz_array
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return (1. + self.mult_shear_bias_array[jb1]) * (1. + self.mult_shear_bias_array[jb2]) * fx_intz

    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_kappa_2h(self, jb1, jb2, jl):
        """
        Computes the 2-halo term of the cross-spectrum between the convergence of two bins (dmb only).
        """
        Wk_jb1 = self.Wk_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)
        Wk_jb2 = self.Wk_mat[jb2]
        prefac_for_uk2 = Wk_jb2/(self.chi_array**2)
        bkl_jl = self.bkl_dmb_mat[jl]
        
        fx = (bkl_jl**2) * prefac_for_uk1 * prefac_for_uk2  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pklin_lz_mat[jl]
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return (1. + self.mult_shear_bias_array[jb1]) * (1. + self.mult_shear_bias_array[jb2]) * fx_intz

    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_kappa_nfw_1h(self, jb1, jb2, jl):
        """
        Computes the 1-halo term of the cross-spectrum between the convergence of two bins (nfw only).
        """
        Wk_jb1 = self.Wk_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)
        Wk_jb2 = self.Wk_mat[jb2]
        prefac_for_uk2 = Wk_jb2/(self.chi_array**2)

        ukl_jl = self.ukappal_nfw_prefac_mat[jl]       
        
        fx = ukl_jl * ukl_jl * self.p_logc_Mz
        fx_intc = jsi.trapezoid(fx, x=self.logc_array)
        fx = fx_intc * self.hmf_Mz_mat
        fx_intM = jsi.trapezoid(fx, x=jnp.log(self.M_array))
        fx = fx_intM * prefac_for_uk1 * prefac_for_uk2 * (self.chi_array ** 2) * self.dchi_dz_array
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return (1. + self.mult_shear_bias_array[jb1]) * (1. + self.mult_shear_bias_array[jb2]) * fx_intz


    @partial(jit, static_argnums=(0,))
    def get_Cl_kappa_kappa_nfw_2h(self, jb1, jb2, jl):
        """
        Computes the 2-halo term of the cross-spectrum between the convergence of two bins (nfw only).
        """
        Wk_jb1 = self.Wk_mat[jb1]
        prefac_for_uk1 = Wk_jb1/(self.chi_array**2)
        Wk_jb2 = self.Wk_mat[jb2]
        prefac_for_uk2 = Wk_jb2/(self.chi_array**2)
        bkl_jl = self.bkl_nfw_mat[jl]
        
        fx = (bkl_jl**2) * prefac_for_uk1 * prefac_for_uk2  * (self.chi_array ** 2) * self.dchi_dz_array * self.Pklin_lz_mat[jl]
        fx_intz = jsi.trapezoid(fx, x=self.z_array)
        return (1. + self.mult_shear_bias_array[jb1]) * (1. + self.mult_shear_bias_array[jb2]) * fx_intz