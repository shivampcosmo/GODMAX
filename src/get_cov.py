import os
from get_power_spectra_jit import get_power_BCMP
from setup_power_spectra_jit import setup_power_BCMP
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import jax.scipy.integrate as jsi
from jax_cosmo import Cosmology
from functools import partial
import astropy.units as u
from astropy import constants as const
RHO_CRIT_0_MPC3 = 2.77536627245708E11
G_new = ((const.G * (u.M_sun / u.Mpc**3) * (u.M_sun) / (u.Mpc)).to(u.eV / u.cm**3)).value
# from transforms import Hankel
# from mcfit import Hankel
from mcfitjax.transforms import Hankel
import time
from twobessel import *
import interpax

class get_cov:
    def __init__(self,
                sim_params_dict,
                halo_params_dict,
                analysis_dict,
                other_params_dict=None,
                num_points_trapz_int=64,
                setup_power_BCMP_obj=None,
                get_power_BCMP_obj=None,
                verbose_time=False
                ):

        if verbose_time:
            t0 = time.time()

        self.cosmo_params = sim_params_dict['cosmo']

        if verbose_time:
            ti = time.time()
        if setup_power_BCMP_obj is None:
            setup_power_BCMP_obj = setup_power_BCMP(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, verbose_time=verbose_time)            
        if get_power_BCMP_obj is None:
            get_power_BCMP_obj = get_power_BCMP(sim_params_dict, halo_params_dict, analysis_dict, other_params_dict, num_points_trapz_int=num_points_trapz_int, setup_power_BCMP_obj=setup_power_BCMP_obj, verbose_time=verbose_time)
        if verbose_time:
            print('Time for setup_power_BCMP: ', time.time() - ti)
            ti = time.time()

        analysis_coords = analysis_dict['analysis_coords']
        l_array_survey = analysis_dict['l_array_survey']
        self.nell, self.nM, self.nc, self.nz = setup_power_BCMP_obj.nell, setup_power_BCMP_obj.nM, setup_power_BCMP_obj.nc, setup_power_BCMP_obj.nz

        self.fsky_dict = {
            'yy': analysis_dict['fsky_yy'],
            'yk': analysis_dict['fsky_ky'],
            'ky': analysis_dict['fsky_ky'],
            'kk': analysis_dict['fsky_kk'],
            }

        self.stats_analyze = ['ky', 'kk']
        stats_analyze_pairs = []
        stats_analyze_pairs_all = []
        index_params = range(len(self.stats_analyze))
        for j1 in index_params:
            for j2 in index_params:
                if j2 >= j1:
                    stats_analyze_pairs.append([self.stats_analyze[j1], self.stats_analyze[j2]])

                stats_analyze_pairs_all.append([self.stats_analyze[j1], self.stats_analyze[j2]])

        self.stats_analyze_pairs = stats_analyze_pairs
        self.stats_analyze_pairs_all = stats_analyze_pairs_all

        self.Cl_result_dict = {}
        log_Cl_yy_interp = interpax.interp1d(
            jnp.log(get_power_BCMP_obj.ell_array), jnp.log(jnp.abs(get_power_BCMP_obj.Cl_y_y_1h_mat + get_power_BCMP_obj.Cl_y_y_2h_mat) + 1e-25)
            )
        self.Cl_result_dict['yy']['0_0']['tot'] = jnp.exp(log_Cl_yy_interp(jnp.log(l_array_survey)))

        log_Cl_ky_interp = interpax.interp2d(
            jnp.arange(get_power_BCMP_obj.nbins), jnp.log(get_power_BCMP_obj.ell_array),
            jnp.log(jnp.abs(get_power_BCMP_obj.Cl_kappa_y_1h_mat + get_power_BCMP_obj.Cl_kappa_y_2h_mat) + 1e-25)
            )
        
        log_Cl_kk_interp = interpax.interp3d(
            jnp.arange(get_power_BCMP_obj.nbins), jnp.arange(get_power_BCMP_obj.nbins), jnp.log(get_power_BCMP_obj.ell_array),
            jnp.log(jnp.abs(get_power_BCMP_obj.Cl_kappa_kappa_1h_mat + get_power_BCMP_obj.Cl_kappa_kappa_2h_mat) + 1e-25)
            )

        for jb1 in range(get_power_BCMP_obj.nbins):
            self.Cl_result_dict['ky'][str(jb1+1) + '_' + str(0)]['tot'] = jnp.exp(log_Cl_ky_interp(jb1, jnp.log(l_array_survey)))

            for jb2 in range(get_power_BCMP_obj.nbins):
                self.Cl_result_dict['kk'][str(jb1+1) + '_' + str(jb2+1)]['tot'] = jnp.exp(log_Cl_kk_interp(jb1, jb2, jnp.log(l_array_survey)))

        # uyl --> (nl, nc, nz, nM)
        uyl_jl_jz = setup_power_BCMP_obj.uyl_mat
        cmean_jz = setup_power_BCMP_obj.conc_Mz_mat
        logc_array = jnp.log(self.conc_array)
        sig_logc = jnp.tile(setup_power_BCMP_obj.sig_logc_z_array, (self.nell, self.nc, 1, self.nM))
        conc_mat = jnp.tile(self.conc_array, (self.nM, 1))
        cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))

        fx = uyl_jl_jz.T * p_logc_Mz
        uyl_intc = jsi.trapezoid(fx, x=logc_array)


        if analysis_coords == 'real':
            if self.verbose:
                print('setting up realspace covariance')

            isodd = 0
            ell_temp = l_array_survey

            if np.mod(len(ell_temp), 2) > 0:
                isodd = 1
                ell = ell_temp[:-1]
            else:
                ell = ell_temp
            nl = len(ell)
            dlnk = np.log(ell[1] / ell[0])
            ell_mat = np.tile(ell.reshape(nl, 1), (1, nl))
            ell1_ell2 = ell_mat * ell_mat.T
            self.fftcovtot_dict = {}

        self.covG_dict = {}
        self.covNG_dict = {}
        self.covtot_dict = {}
        if analysis_coords == 'real':
            self.fftcovtot_dict = {}

        for j in range(len(self.stats_analyze_pairs)):
            stats_analyze_1, stats_analyze_2 = self.stats_analyze_pairs[j]
            if self.verbose:
                print('starting covariance of ' + str(stats_analyze_1) + ' and ' + str(stats_analyze_2))
            if stats_analyze_1 in self.Cl_result_dict.keys():
                stats_analyze_1_ordered = stats_analyze_1
            else:
                stats_analyze_1_ordered = list(stats_analyze_1)[1] + list(stats_analyze_1)[0]
            bin_combs_stat1 = self.Cl_result_dict[stats_analyze_1_ordered]['bin_combs']
            bins1_stat1 = []
            bins2_stat1 = []
            for jb in range(len(bin_combs_stat1)):
                bins1_stat1.append(bin_combs_stat1[jb][0])
                bins2_stat1.append(bin_combs_stat1[jb][1])

            if stats_analyze_2 in self.Cl_result_dict.keys():
                stats_analyze_2_ordered = stats_analyze_2
            else:
                stats_analyze_2_ordered = list(stats_analyze_2)[1] + list(stats_analyze_2)[0]
            bin_combs_stat2 = self.Cl_result_dict[stats_analyze_2_ordered]['bin_combs']
            bins1_stat2 = []
            bins2_stat2 = []
            for jb in range(len(bin_combs_stat2)):
                bins1_stat2.append(bin_combs_stat2[jb][0])
                bins2_stat2.append(bin_combs_stat2[jb][1])

            covG_stat12 = {}
            covNG_stat12 = {}
            covtot_stat12 = {}
            isgtykk, isgtygty, iskkkk, isgygy = False, False, False, False
            if analysis_coords == 'real':
                fftcovtot_stat12 = {}
                fftmcovtot_stat12 = {}
                fftpmcovtot_stat12 = {}
                if (stats_analyze_1_ordered == 'ky') and (stats_analyze_2_ordered == 'ky'):
                    gtfftcovtot_stat12 = {}
                    isgtygty = True
                # if (stats_analyze_1_ordered == 'gy') and (stats_analyze_2_ordered == 'gy'):
                #     isgygy = True
                if (stats_analyze_1_ordered == 'kk') and (stats_analyze_2_ordered == 'kk'):
                    iskkkk = True
                if ((stats_analyze_1_ordered == 'kk') and
                    (stats_analyze_2_ordered
                        == 'ky')) or ((stats_analyze_1_ordered == 'ky') and (stats_analyze_2_ordered == 'kk')):
                    kkgtfftcovtot_stat12 = {}
                    kkmgtfftcovtot_stat12 = {}
                    isgtykk = True
            bins_comb = []
            for jb1 in range(len(bins1_stat1)):
                for jb2 in range(len(bins1_stat2)):
                    if self.verbose:
                        print(
                            stats_analyze_1_ordered, stats_analyze_2_ordered, bins1_stat1[jb1], bins2_stat1[jb1],
                            bins1_stat2[jb2], bins2_stat2[jb2]
                            )

                    covG = self.get_cov_G(
                        bins1_stat1[jb1], bins2_stat1[jb1], bins1_stat2[jb2], bins2_stat2[jb2],
                        stats_analyze_1_ordered, stats_analyze_2_ordered, self.Cl_result_dict, self.fsky_dict
                        )

                    A, B = list(stats_analyze_1_ordered)
                    C, D = list(stats_analyze_2_ordered)

                    uAl_zM_dict = PrepDV_params['u' + A + 'l_zM_dict' + str(bins1_stat1[jb1])]
                    uBl_zM_dict = PrepDV_params['u' + B + 'l_zM_dict' + str(bins2_stat1[jb1])]
                    uCl_zM_dict = PrepDV_params['u' + C + 'l_zM_dict' + str(bins1_stat2[jb2])]
                    uDl_zM_dict = PrepDV_params['u' + D + 'l_zM_dict' + str(bins2_stat2[jb2])]

                    covNG = self.get_cov_NG(
                        PrepDV.l_array_survey, stats_analyze_1_ordered, stats_analyze_2_ordered,
                        PrepDV.PS.use_only_halos, self.fsky_dict, uAl_zM_dict, uBl_zM_dict, uCl_zM_dict, uDl_zM_dict,
                        beam_fwhm_arcmin[0]
                        )

                    covtot = covG + covNG
                    # covtot = covG
                    bin_key = 'bin_' + str(bins1_stat1[jb1]) + '_' + str(bins2_stat1[jb1]) + '_' + str(
                        bins1_stat2[jb2]
                        ) + '_' + str(bins2_stat2[jb2])
                    covG_stat12[bin_key] = covG
                    covNG_stat12[bin_key] = covNG
                    covtot_stat12[bin_key] = covtot
                    bins_comb.append([bins1_stat1[jb1], bins2_stat1[jb1], bins1_stat2[jb2], bins2_stat2[jb2]])
                    if analysis_coords == 'real':
                        if isodd:
                            covtot_rs = covtot[:-1, :][:, :-1]
                        else:
                            covtot_rs = covtot
                        newtwobessel = two_Bessel(
                            ell,
                            ell,
                            covtot_rs * (ell1_ell2**2) * (1. / (4 * np.pi**2)),
                            nu1=1.01,
                            nu2=1.01,
                            N_extrap_low=0,
                            N_extrap_high=0,
                            c_window_width=0.25,
                            N_pad=1000
                            )
                        t1, t2, cov_fft = newtwobessel.two_Bessel_binave(0, 0, dlnk, dlnk)
                        theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                        cov_tot_fft = cov_fft[:, :-1][:-1, :]
                        fftcovtot_stat12[bin_key] = cov_tot_fft

                        if iskkkk:
                            t1, t2, cov_fft = newtwobessel.two_Bessel_binave(4, 4, dlnk, dlnk)
                            theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                            cov_tot_fftm = cov_fft[:, :-1][:-1, :]
                            fftmcovtot_stat12[bin_key] = cov_tot_fftm

                            t1, t2, cov_fft = newtwobessel.two_Bessel_binave(4, 0, dlnk, dlnk)
                            theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                            cov_tot_fftm = cov_fft[:, :-1][:-1, :]
                            fftpmcovtot_stat12[bin_key] = cov_tot_fftm

                        if isgtygty:
                            t1, t2, covgt_fft = newtwobessel.two_Bessel_binave(2, 2, dlnk, dlnk)
                            gtfftcovtot_stat12[bin_key] = covgt_fft[:, :-1][:-1, :]
                            theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                            if 'theta' not in gtfftcovtot_stat12.keys():
                                gtfftcovtot_stat12['theta'] = theta_vals_arcmin_fft

                        if isgtykk:
                            t1, t2, covgt_fft = newtwobessel.two_Bessel_binave(2, 0, dlnk, dlnk)
                            kkgtfftcovtot_stat12[bin_key] = covgt_fft[:, :-1][:-1, :]
                            theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                            if 'theta' not in kkgtfftcovtot_stat12.keys():
                                kkgtfftcovtot_stat12['theta'] = theta_vals_arcmin_fft

                            t1, t2, covgt_fft = newtwobessel.two_Bessel_binave(2, 4, dlnk, dlnk)
                            kkmgtfftcovtot_stat12[bin_key] = covgt_fft[:, :-1][:-1, :]
                            theta_vals_arcmin_fft = (t1[:-1] + t1[1:]) / 2. / np.pi * 180 * 60
                            if 'theta' not in kkmgtfftcovtot_stat12.keys():
                                kkmgtfftcovtot_stat12['theta'] = theta_vals_arcmin_fft
                            
                        if 'theta' not in fftcovtot_stat12.keys():
                            fftcovtot_stat12['theta'] = theta_vals_arcmin_fft
                        if 'theta' not in fftmcovtot_stat12.keys():
                            fftmcovtot_stat12['theta'] = theta_vals_arcmin_fft
                        if 'theta' not in fftpmcovtot_stat12.keys():
                            fftpmcovtot_stat12['theta'] = theta_vals_arcmin_fft

            covG_stat12['bins_comb'] = bins_comb
            covNG_stat12['bins_comb'] = bins_comb
            covtot_stat12['bins_comb'] = bins_comb
            if analysis_coords == 'real':
                fftcovtot_stat12['bins_comb'] = bins_comb
                fftmcovtot_stat12['bins_comb'] = bins_comb
                fftpmcovtot_stat12['bins_comb'] = bins_comb
                if isgtygty:
                    gtfftcovtot_stat12['bins_comb'] = bins_comb
                    stat_analyze_key = 'gty_gty'
                    self.fftcovtot_dict[stat_analyze_key] = gtfftcovtot_stat12

                if isgtykk:
                    kkgtfftcovtot_stat12['bins_comb'] = bins_comb
                    if ((stats_analyze_1_ordered == 'kk') and (stats_analyze_2_ordered == 'ky')):
                        stat_analyze_key1 = 'kk_gty'
                        stat_analyze_key2 = 'kkm_gty'
                    else:
                        stat_analyze_key1 = 'gty_kk'
                        stat_analyze_key2 = 'gty_kkm'

                    self.fftcovtot_dict[stat_analyze_key1] = kkgtfftcovtot_stat12
                    kkmgtfftcovtot_stat12['bins_comb'] = bins_comb
                    self.fftcovtot_dict[stat_analyze_key2] = kkmgtfftcovtot_stat12

                self.fftcovtot_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = fftcovtot_stat12
                if iskkkk:
                    self.fftcovtot_dict['kkm_kkm'] = fftmcovtot_stat12
                    self.fftcovtot_dict['kk_kkm'] = fftpmcovtot_stat12

            self.covG_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covG_stat12
            self.covNG_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covNG_stat12
            self.covtot_dict[stats_analyze_1_ordered + '_' + stats_analyze_2_ordered] = covtot_stat12


    @partial(jit, static_argnums=(0,))
    def get_uyl_intc(self, jz):
        uyl_jl_jz = self.uyl_mat[:, :, jz, :]
        cmean_jz = self.conc_Mz_mat[jz, :]
        logc_array = jnp.log(self.conc_array)
        sig_logc = self.sig_logc_z_array[jz]
        conc_mat = jnp.tile(self.conc_array, (self.nell, self.nM, 1))
        cmean_jz_mat = jnp.tile(cmean_jz, (self.nc, 1)).T
        p_logc_Mz = jnp.exp(-0.5 * (jnp.log(conc_mat/cmean_jz_mat)/ sig_logc)**2) * (1.0/(sig_logc * jnp.sqrt(2*jnp.pi)))

        fx = uyl_jl_jz.T * p_logc_Mz
        uyl_intc = jsi.trapezoid(fx, x=logc_array)

        return uyl_intc


    def get_cov_G(
            self, bin1_stat1, bin2_stat1, bin1_stat2, bin2_stat2, stats_analyze_1, stats_analyze_2, Cl_result_dict,
            fsky_dict
        ):

        A, B = list(stats_analyze_1)
        C, D = list(stats_analyze_2)
        k_sum1 = (A == 'k') * (B == 'k')
        k_sum2 = (C == 'k') * (D == 'k')
        if k_sum1 == 1 and k_sum2 == 1:
            iskk = 1
        else:
            iskk = 0
        stats_pairs = [A + C, B + D, A + D, B + C]
        bin_pairs = [
            [bin1_stat1, bin1_stat2], [bin2_stat1, bin2_stat2], [bin1_stat1, bin2_stat2], [bin2_stat1, bin1_stat2]
            ]
        Cl_stats_dict = {}
        Nl_stats_dict = {}

        for j in range(len(stats_pairs)):
            stat = stats_pairs[j]
            bin_pair = bin_pairs[j]
            bin_key = 'bin_' + str(bin_pair[0]) + '_' + str(bin_pair[1])
            Atemp, Btemp = list(stat)
            if Atemp == Btemp:
                try:
                    Cl_temp = Cl_result_dict[stat][bin_key]['tot_plus_noise_ellsurvey']
                    if iskk:
                        Nl = Cl_result_dict[stat][bin_key]['tot_plus_noise_ellsurvey'] - Cl_result_dict[stat][bin_key][
                            'tot_ellsurvey']
                        Nl_stats_dict[j] = Nl
                    Cl_stats_dict[j] = Cl_result_dict[stat][bin_key]['tot_plus_noise_ellsurvey']
                except:
                    bin_key = 'bin_' + str(bin_pair[1]) + '_' + str(bin_pair[0])
                    if iskk:
                        Nl = Cl_result_dict[stat][bin_key]['tot_plus_noise_ellsurvey'] - Cl_result_dict[stat][bin_key][
                            'tot_ellsurvey']
                        Nl_stats_dict[j] = Nl
                    Cl_stats_dict[j] = Cl_result_dict[stat][bin_key]['tot_plus_noise_ellsurvey']
            else:
                try:
                    Cl_stats_dict[j] = Cl_result_dict[stat][bin_key]['tot_plus_noise_ellsurvey']
                except:
                    bin_key = 'bin_' + str(bin_pair[1]) + '_' + str(bin_pair[0])
                    Cl_stats_dict[j] = Cl_result_dict[Btemp + Atemp][bin_key]['tot_plus_noise_ellsurvey']

        fsky_j = np.sqrt(fsky_dict[A + B] * fsky_dict[C + D])

        to_mult = np.ones_like(Cl_result_dict['l_array_survey'])

        if iskk:
            #           if doing xi_plus or xi_minus, then add the shape noise due to BB correlations
            Nl_BB = (bin1_stat1 == bin1_stat2) * (bin2_stat1 == bin2_stat2) * Nl_stats_dict[0] * Nl_stats_dict[
                1] + (bin1_stat1 == bin2_stat2) * (bin2_stat1 == bin1_stat2) * Nl_stats_dict[2] * Nl_stats_dict[3]
            val_diag = (
                1. / (fsky_j * (2 * Cl_result_dict['l_array_survey'] + 1.) * Cl_result_dict['dl_array_survey'])
                ) * (Cl_stats_dict[0] * Cl_stats_dict[1] + Cl_stats_dict[2] * Cl_stats_dict[3] + Nl_BB) * (to_mult**2)
        else:
            val_diag = (
                1. / (fsky_j * (2 * Cl_result_dict['l_array_survey'] + 1.) * Cl_result_dict['dl_array_survey'])
                ) * (Cl_stats_dict[0] * Cl_stats_dict[1] + Cl_stats_dict[2] * Cl_stats_dict[3]) * (to_mult**2)
        return np.diag(val_diag)
    
    def get_cov_NG(
            self, l_array_survey, stats_analyze_1, stats_analyze_2, use_only_halos, fsky_dict, uAl_zM_dict, uBl_zM_dict,
            uCl_zM_dict, uDl_zM_dict, beam_fwhm_arcmin
        ):
        A, B = list(stats_analyze_1)
        C, D = list(stats_analyze_2)

        if use_only_halos and (A + B + '_' + C + D != 'yy_yy'):
            nl = len(l_array_survey)
            val_NG = np.zeros((nl, nl))
        else:
            T_l_ABCD = self.get_T_ABCD_NG(
                l_array_survey, A, B, C, D, uAl_zM_dict, uBl_zM_dict, uCl_zM_dict, uDl_zM_dict, beam_fwhm_arcmin
                )
            fsky_j = np.sqrt(fsky_dict[A + B] * fsky_dict[C + D])
            val_NG = (1. / (4. * np.pi * fsky_j)) * T_l_ABCD

        return val_NG    

    def get_T_ABCD_NG(
            self, l_array_all, A, B, C, D, uAl_zM_dict, uBl_zM_dict, uCl_zM_dict, uDl_zM_dict, beam_fwhm_arcmin
        ):
        nl = len(l_array_all)

        addb = self.add_beam_to_theory
        sig_beam = beam_fwhm_arcmin * (1. / 60.) * (np.pi / 180.) * (1. / np.sqrt(8. * np.log(2)))

        ul_A_mat, ul_B_mat, ul_C_mat, ul_D_mat = np.zeros((nl, self.PS_prepDV.nz, self.PS_prepDV.nm)), np.zeros(
            (nl, self.PS_prepDV.nz, self.PS_prepDV.nm)
            ), np.zeros((nl, self.PS_prepDV.nz, self.PS_prepDV.nm)), np.zeros(
                (nl, self.PS_prepDV.nz, self.PS_prepDV.nm)
                )
        for j in range(nl):
            Bl = np.exp(-1. * l_array_all[j] * (l_array_all[j] + 1) * (sig_beam**2) / 2.)
            isy = (A == 'y')
            ul_A_mat[j, :, :] = (uAl_zM_dict[round(l_array_all[j], 1)]) * (Bl**(isy * addb))
            isy = (B == 'y')
            ul_B_mat[j, :, :] = (uBl_zM_dict[round(l_array_all[j], 1)]) * (Bl**(isy * addb))
            isy = (C == 'y')
            ul_C_mat[j, :, :] = (uCl_zM_dict[round(l_array_all[j], 1)]) * (Bl**(isy * addb))
            isy = (D == 'y')
            ul_D_mat[j, :, :] = (uDl_zM_dict[round(l_array_all[j], 1)]) * (Bl**(isy * addb))

        uAl1_uBl1 = ul_A_mat * ul_B_mat
        uCl2_uDl2 = ul_C_mat * ul_D_mat
        uAl1_uBl1_mat = np.tile(uAl1_uBl1.reshape(1, nl, self.PS_prepDV.nz, self.PS_prepDV.nm), (nl, 1, 1, 1))
        uCl2_uDl2_mat = np.tile(uCl2_uDl2.reshape(nl, 1, self.PS_prepDV.nz, self.PS_prepDV.nm), (1, nl, 1, 1))
        dndm_array_mat = np.tile(
            self.PS_prepDV.dndm_array.reshape(1, 1, self.PS_prepDV.nz, self.PS_prepDV.nm), (nl, nl, 1, 1)
            )
        if 'g' in [A, B, C, D]:
            toint_M = (
                uAl1_uBl1_mat * uCl2_uDl2_mat
                ) * dndm_array_mat * self.PS_prepDV.M_mat_cond_inbin * self.PS_prepDV.z_mat_cond_inbin
        else:
            toint_M = (uAl1_uBl1_mat * uCl2_uDl2_mat) * dndm_array_mat
        val_z = sp.integrate.simps(toint_M, self.PS_prepDV.M_array)
        chi2_array_mat = np.tile((self.PS_prepDV.chi_array**2).reshape(1, 1, self.PS_prepDV.nz), (nl, nl, 1))
        dchi_dz_array_mat = np.tile(self.PS_prepDV.dchi_dz_array.reshape(1, 1, self.PS_prepDV.nz), (nl, nl, 1))
        toint_z = val_z * chi2_array_mat * dchi_dz_array_mat
        val = sp.integrate.simps(toint_z, self.PS_prepDV.z_array)

        if self.PS_prepDV.use_only_halos:
            if (A + B + C + D not in [''.join(elem) for elem in list(set(list(itertools.permutations('gyyy'))))
                                     ]) and (A + B + C + D != 'yyyy'):
                val = np.zeros(val.shape)

        return val    