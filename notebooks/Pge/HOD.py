import numpy as np
from scipy import interpolate, integrate
import scipy.special
import warnings
import sys, os
sys.path.append('/projects/bdne/spandey3/BaryonForge')

import BaryonForge as bfg
import BaryonForge.Profiles

import pyccl as ccl

class BaryonifiedHODFractions():
	def __init__(self, *, M_2_Mtot=None, Mstar_th=10**10.72, 
					M1_fshmr=10**12.55, log10M1_a_fshmr=0.38, 
					Mstar0_fshmr=10**10.72, log10Mstar0_a_fshmr=0.55,
					beta_fshmr=0.4, beta_a_fshmr=0.2, delta_shmr=0.37, delta_a_shmr=0.21, gamma_fshmr=1.79, gamma_a_fshmr=-0.72, 
					siglogMstar_Ncen=0.7, Bsat_Nsat=9.01, betasat_Nsat=0.3, Bcut_Nsat=1.69, betacut_Nsat=0.6, alphasat_Nsat=1.19,
					is_number_counts=True, SGA=None, CGA=None):

		# Central number count
		self.M_2_Mtot = M_2_Mtot
		self.Mstar_th = Mstar_th
		self.siglogMstar_Ncen = siglogMstar_Ncen

		# Stellar-to-halo mass relation (attributes now match __init__ names)
		self.M1_fshmr = M1_fshmr
		self.log10M1_a_fshmr = log10M1_a_fshmr
		self.Mstar0_fshmr = Mstar0_fshmr
		self.log10Mstar0_a_fshmr = log10Mstar0_a_fshmr

		self.beta_fshmr = beta_fshmr
		self.beta_a_fshmr = beta_a_fshmr
		self.delta_shmr = delta_shmr
		self.delta_a_shmr = delta_a_shmr
		self.gamma_fshmr = gamma_fshmr
		self.gamma_a_fshmr = gamma_a_fshmr

		# Satellite number count (attributes now match __init__ names)
		self.alphasat_Nsat = alphasat_Nsat
		self.Bsat_Nsat = Bsat_Nsat
		self.Bcut_Nsat = Bcut_Nsat
		self.betasat_Nsat = betasat_Nsat
		self.betacut_Nsat = betacut_Nsat

		self._Nc_cache = None


	def update_parameters(self,*, Mstar_th=None, M1_fshmr=None, Mstar0_fshmr=None,
							beta_fshmr=None, delta_shmr=None, gamma_fshmr=None,
							sigma_logMstar=None, Bsat_Nsat=None, betasat_Nsat=None,
							Bcut_Nsat=None, betacut_Nsat=None, alphasat_Nsat=None):
		# Update parameters (names matching __init__)
		if Mstar_th is not None:
			self.Mstar_th = Mstar_th
		if M1_fshmr is not None:
			self.M1_fshmr = M1_fshmr
		if Mstar0_fshmr is not None:
			self.Mstar0_fshmr = Mstar0_fshmr
		if beta_fshmr is not None:
			self.beta_fshmr = beta_fshmr
		if delta_shmr is not None:
			self.delta_shmr = delta_shmr
		if gamma_fshmr is not None:
			self.gamma_fshmr = gamma_fshmr
		if sigma_logMstar is not None:
			self.siglogMstar_Ncen = sigma_logMstar
		if Bsat_Nsat is not None:
			self.Bsat_Nsat = Bsat_Nsat
		if betasat_Nsat is not None:
			self.betasat_Nsat = betasat_Nsat
		if Bcut_Nsat is not None:
			self.Bcut_Nsat = Bcut_Nsat
		if betacut_Nsat is not None:
			self.betacut_Nsat = betacut_Nsat
		if alphasat_Nsat is not None:
			self.alphasat_Nsat = alphasat_Nsat

	def Mh_Mstar(self, Mstar, a):
		"""
		Returns halo mass given stellar mass using the inverse of the stellar-to-halo mass relation
		"""
		beta_SHMR = self.beta_fshmr  + self.beta_a_fshmr * (a-1)
		delta_SHMR = self.delta_shmr  + self.delta_a_shmr * (a-1)
		gamma_SHMR = self.gamma_fshmr  + self.gamma_a_fshmr * (a-1)

		M1 = 10**(np.log10(self.M1_fshmr) + self.log10M1_a_fshmr * (a-1))
		Mstar0 = 10**(np.log10(self.Mstar0_fshmr) + self.log10Mstar0_a_fshmr * (a-1))

		x = Mstar/Mstar0
		log10_inv_fSHMR = np.log10(M1) + beta_SHMR* np.log10(x) + (x)**delta_SHMR/(1 + x**-gamma_SHMR) - 0.5

		return 10**log10_inv_fSHMR

	def Mstar_Mh(self, M, a):
		"""
		Returns stellar mass given halo mass using the (numerically) inverted inverse stellar-to-halo mass relation.
		"""
		from scipy.optimize import minimize

		M = np.atleast_1d(M)
		result = np.zeros_like(M)

		def to_solve(logMstar, target_M):
			return np.sum((np.log10(self.Mh_Mstar(10**logMstar, a)) - np.log10(target_M))**2)

		for i, Mi in enumerate(M):
			sol = scipy.optimize.differential_evolution(to_solve, bounds=[(8, 20)], args=(Mi,))
			result[i] = 10**sol.x

		return result if result.size > 1 else result[0]


	def _Nc(self, M, a, Mstar_th=None, use_interp=True):
		"""
		Mean number of central galaxies
		<Ncen(M|Mstar_th)> = 0.5 * [1 - erf((log10(Mstar_th) - log10(fSHMR(M))) / (sqrt(2) * sigma_logMstar))]
		"""
		if Mstar_th is None:
			Mstar_th = self.Mstar_th

		if self._Nc_cache is not None and use_interp:
			return self._Nc_cache((M, Mstar_th))


		numerator = np.log10(Mstar_th) - np.log10(self.Mstar_Mh(M, a))
		denominator = np.sqrt(2) * self.siglogMstar_Ncen
		Nc = 0.5 * (1 - scipy.special.erf(numerator / denominator))

		return Nc

	def _Nc_interpolator(self, a):
		"""
		Interpolator for mean number of central galaxies,
		on a 2D grid of M and Mstar_th
		"""

		from scipy.interpolate import RegularGridInterpolator
		if self._Nc_cache is None:
			Mgrid = np.logspace(9, 16, 50)
			Mstar_th_grid = np.logspace(9, 16, 50)
			Mc, Mstc = np.meshgrid(Mgrid, Mstar_th_grid, indexing='ij')
			Nc_grid = self._Nc(Mc, a, Mstar_th=Mstc)

			self._Nc_cache = RegularGridInterpolator((Mgrid, Mstar_th_grid), Nc_grid, method='cubic',
													bounds_error=False, fill_value=0.0)


	def _Ns(self, M, a, Mstar_th=None):
		"""
		Mean number of satellite galaxies
		<Nsat(M|Mstar_th)> = <Ncen(M|Mstar_th)> * (M/Msat)^alpha_sat * exp(-Mcut/M)
		"""
		if Mstar_th is None:
			Mstar_th = self.Mstar_th

		Mhalo = self.Mh_Mstar(Mstar_th, a)

		Msat = self.Bsat_Nsat * (Mhalo/1e12)**self.betasat_Nsat * 1e12
		Mcut = self.Bcut_Nsat * (Mhalo/1e12)**self.betacut_Nsat * 1e12

		return self._Nc(M, a, Mstar_th) * (M/Msat)**self.alphasat_Nsat * np.exp(-Mcut/M)

	def _integrator(self, M, a, Ngal, Mstar_th, cosmo):
		'''Eq. 10'''

		M = np.atleast_1d(M)

		if self.M_2_Mtot is None:
			M_tot = M
		else:
			M_tot = self.M_2_Mtot(cosmo, M,  a)

		results = np.zeros_like(M, dtype=float)

		Mstar_min = Mstar_th
		Mstar_max = 1e14
		n_points = 32

		Mstars = np.logspace(np.log10(Mstar_min), np.log10(Mstar_max), num=n_points)


		for i, this_Mhalo in enumerate(M):
			# Discrete numerical integration (log-spaced) instead of quad
			integrand_vals = np.array([Ngal(this_Mhalo, a, Mstar_th=ms) for ms in Mstars])

			term1 =  (Ngal(this_Mhalo, a, Mstar_th=Mstar_max) * Mstar_max
								- Ngal(this_Mhalo, a, Mstar_th=Mstar_th) * Mstar_th)
			
			term2 = scipy.integrate.trapezoid(integrand_vals*Mstars, x=np.log10(Mstars))


			results[i] = (term2 - term1) / M_tot[i]
		return results


	def _fc(self, M, a, Mstar_th=None, **kwargs):
		if Mstar_th is None:
			Mstar_th = self.Mstar_th
		return self._integrator(M, a, self._Nc, Mstar_th, **kwargs)[:, None]


	def _fs(self, M, a, Mstar_th=None, **kwargs):
		if Mstar_th is None:
			Mstar_th = self.Mstar_th
		return self._integrator(M, a, self._Ns, Mstar_th, **kwargs)[:, None]


class BaryonifiedHOD(ccl.halos.profiles.HaloProfile):
	def __init__(self, *, mass_def, concentration, is_number_counts=True, 
			CGA=None, SGA=None, **kwargs):

		self.CGA = CGA if CGA is not None else Stars(**kwargs, r_min_int = 5e-3, r_max_int = 8, r_steps = 500)
		self.SGA = SGA if SGA is not None else bfg.Profiles.Schneider19.DarkMatter(**kwargs, r_min_int = 1e-3, r_max_int = 1e2, r_steps = 500)

		super().__init__(mass_def=mass_def, concentration=concentration,
					is_number_counts=is_number_counts)

	# def _real(self, cosmo, r, M, a):
	# 	"""
	# 	Interface with real-space
	# 	"""
	# 	return self._fftlog_wrap(cosmo, r, M, a, fourier_out=False)

	@property
	def fractions(self):
		return self.CGA.fractions

	def update_parameters(self, **kwargs):
		self.fractions.update_parameters(**kwargs)
		self.CGA = self.CGA(**kwargs)
		self.SGA = self.SGA(**kwargs)

	def _fourier(self, cosmo, k, M, a):
		"""
		Interface with Fourier-space
		"""
		M = np.atleast_1d(M)
		Ncen = np.atleast_1d(self.fractions._Nc(M, a, self.fractions.Mstar_th))
		Nsat = np.atleast_1d(self.fractions._Ns(M, a, self.fractions.Mstar_th))

		return Ncen[:, None] + Nsat[:, None] * self._usat_fourier(cosmo, k, M, a)
		# return Nsat[:, None] * self.satellites.fourier(cosmo, k, M, a)/self.M_2_Mtot(cosmo, M, a)[:, None]


	def _usat_fourier(self, cosmo, k, M, a):
		"""
		Interface with Fourier-space for satellites
		"""
		M = np.atleast_1d(M)
		fs = np.atleast_1d(self.fractions._fs(M, a, self.fractions.Mstar_th, cosmo=cosmo))

		# SGA.fourier should return a normalized profile i.e., u(k|M,a) should go to 1 on large scales	
		return  self.SGA.fourier(cosmo, k, M, a)/self.fractions.M_2_Mtot(cosmo, M, a)[:, None]#/fs

	# def _real(self, cosmo, r, M, a):
	# 	"""
	# 	Interface with real-space
	# 	"""
	# 	return self._real_cen(cosmo, r, M, a) + self._real_sat(cosmo, r, M, a)

	def _ucen_real(self, cosmo, r, M, a):
		"""
		Interface with real-space for centrals only
		"""
		M = np.atleast_1d(M)
		return self.CGA.real(cosmo, r, M, a)


	def _usat_real(self, cosmo, r, M, a):
		"""
		Interface with real-space for satellites only
		"""
		M = np.atleast_1d(M)
		fs = np.atleast_1d(self.fractions._fs(M, a, self.fractions.Mstar_th, cosmo=cosmo))

		u_sat = self.SGA.real(cosmo, r, M, a)*fs

		return u_sat


	def _fourier_variance(self, cosmo, k, M, a):
		# Fourier-space variance of the HOD profile
		M_use = np.atleast_1d(M)
		k_use = np.atleast_1d(k)

		Ncen = self.fractions._Nc(M_use, a)
		Nsat = self.fractions._Ns(M_use, a)
		# NFW profile
		uk = self._usat_fourier(cosmo, k_use, M_use, a)

		prof = 2 * Ncen[:, None] *  Nsat[:, None] * uk + (Nsat[:, None] * uk)**2

		if np.ndim(k) == 0:
			prof = np.squeeze(prof, axis=-1)
		if np.ndim(M) == 0:
			prof = np.squeeze(prof, axis=0)
		return prof

	def get_normalization(self, cosmo, a, hmc):
		"""Returns the normalization of this profile, which is the
		mean galaxy number density.

		Args:
			cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology
				object.
			a (:obj:`float`): scale factor.
			hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo
				model calculator object.

		Returns:
			:obj:`float`: normalization factor of this profile.
		"""
		def integ(M):
			Nc = self.fractions._Nc(M, a)
			Ns = self.fractions._Ns(M, a)

			return Nc + Ns
		return hmc.integrate_over_massfunc(integ, cosmo, a)



class Stars(bfg.Profiles.Schneider19.SchneiderProfiles, BaryonifiedHODFractions):
	"""Stolen from BaryonForge
	"""

	def __init__(self, fractions=None, **kwargs):
		self.fractions = fractions if fractions is not None else BaryonifiedHODFractions(**kwargs)
		super().__init__(**kwargs)
	
		#For some reason, we need to make this extreme in order
		#to prevent ringing in the profiles. Haven't figured out
		#why this is the case
		self.update_precision_fftlog(padding_lo_fftlog = 1e-5, padding_hi_fftlog = 1e5)
		self.epsilon_h = 0.015
		self.epsilon = 4

	def _real(self, cosmo, r, M, a):

		r_use = np.atleast_1d(r)
		M_use = np.atleast_1d(M)

		z = 1/a - 1

		R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

		# PRS: Following line is changed
		f_cga  = self.fractions._fc(M_use, a=a, cosmo=cosmo)
		R_h    = self.epsilon_h * R[:, None]

		r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
		DM    = bfg.Profiles.Schneider19.DarkMatter(**self.model_params); setattr(DM, 'cutoff', 1e3)
		rho   = DM.real(cosmo, r_integral, M_use, a)
		M_tot = np.trapezoid(4*np.pi*r_integral**2 * rho, r_integral, axis = -1)
		M_tot = np.atleast_1d(M_tot)[:, None]

		arg  = (r_use[None, :] - self.cutoff)
		arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
		kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff

		prof = f_cga * M_tot / (4*np.pi**(3/2)*R_h) * 1/r_use**2 * np.exp(-(r_use/2/R_h)**2) * kfac

		#Handle dimensions so input dimensions are mirrored in the output
		if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
		if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)
		return prof

# class SatelliteStars(bfg.Profiles.Schneider19.DarkMatter):


class BaryonifiedHOD2pt(ccl.halos.profiles_2pt.Profile2pt):
	""" This class implements the Fourier-space 1-halo 2-point
	correlator for the HOD profile.

	.. math::
	   \\langle n_g^2(k)|M,a\\rangle = \\bar{N}_c(M,a)
	   \\left[2f_c(a)\\bar{N}_s(M,a) u_{\\rm sat}(r|M,a)+
	   (\\bar{N}_s(M,a) u_{\\rm sat}(r|M,a))^2\\right],

	where all quantities are described in the documentation of
	:class:`~pyccl.halos.profiles.hod.HaloProfileHOD`.
	"""

	def fourier_2pt(self, cosmo, k, M, a, prof, *, prof2=None, diag=True):
		if prof2 is None:
			prof2 = prof

		# If the profiles are different assume disjoint tracers
		if prof != prof2:
			uk1 = prof.fourier(cosmo, k, M, a)
			uk2 = prof2.fourier(cosmo, k, M, a)


		# PRS: Need this for type checking
		HOD = BaryonifiedHOD
		if not (isinstance(prof, HOD) and isinstance(prof2, HOD)):
			raise TypeError("prof and prof2 must be BaryonifiedHOD")

		# TODO: This should be implemented in _fourier_variance
		if (diag is True) or (isinstance(k, float)):
			if prof == prof2:
				output = prof._fourier_variance(cosmo, k, M, a)
			else:
				output = uk1 * uk2 * (1 + self.r_corr)
		elif isinstance(M, float):
			if prof == prof2:
				uk1 = prof.fourier(cosmo, k, M, a)
				output = uk1[None, :] * uk1[:, None] * (1 + self.r_corr)
			else:
				output = uk1[None, :] * uk2[:, None] * (1 + self.r_corr)
		else:
			if prof == prof2:
				uk1 = prof.fourier(cosmo, k, M, a)
				output = uk1[:, None, :] * uk1[:, :, None] * (1 + self.r_corr)
			else:
				output = uk1[:, None, :] * uk2[:, :, None] * (1 + self.r_corr)

		return output


class CollisionlessMatter(bfg.Profiles.Schneider19.SchneiderProfiles):
	def __init__(self, gas = None, stars = None, darkmatter = None, max_iter = 10, reltol = 1e-2, r_min_int = 1e-8, r_max_int = 1e5, r_steps = 5000, **kwargs):

		self.Gas   = gas
		self.Stars = stars
		self.DarkMatter = darkmatter

		if self.Gas is None: self.Gas = bfg.Profiles.Schneider19.Gas(**kwargs)
		if self.Stars is None: self.Stars = bfg.Profiles.Schneider19.Stars(**kwargs)
		if self.DarkMatter is None: self.DarkMatter = bfg.Profiles.Schneider19.DarkMatter(**kwargs)

		#Stop any artificially cutoffs when doing the relaxation.
		#The profile will be cutoff at the very last step instead
		self.Gas.set_parameter('cutoff', 1000)
		self.Stars.set_parameter('cutoff', 1000)
		self.DarkMatter.set_parameter('cutoff', 1000)
			
		self.max_iter   = max_iter
		self.reltol     = reltol

		self.r_min_int  = r_min_int
		self.r_max_int  = r_max_int
		self.r_steps    = r_steps
		
		super().__init__(**kwargs, r_min_int = r_min_int, r_max_int = r_max_int, r_steps = r_steps)


	def _real(self, cosmo, r, M, a):
		r_use = np.atleast_1d(r)
		M_use = np.atleast_1d(M)

		if np.min(r) < self.r_min_int: 
			warnings.warn(f"Decrease integral lower limit, r_min_int ({self.r_min_int}) < minimum radius ({np.min(r)})", UserWarning)
		if np.max(r) > self.r_max_int: 
			warnings.warn(f"Increase integral upper limit, r_max_int ({self.r_max_int}) < maximum radius ({np.max(r)})", UserWarning)

		#Def radius sampling for doing iteration.
		#And don't check iteration near the boundaries, since we can have numerical errors
		#due to the finite width oof the profile during iteration.
		#Radius boundary is very large, I found that worked best without throwing edgecases
		#especially when doing FFTlog transforms
		r_integral = np.geomspace(self.r_min_int, self.r_max_int, self.r_steps)
		safe_range = (r_integral > 2 * np.min(r_integral) ) & (r_integral < 1/2 * np.max(r_integral) )
		
		z = 1/a - 1

		R = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc

		# eta_cga = self.eta + self.eta_delta
		# tau_cga = self.tau + self.tau_delta

		# PRS: Following line is changed
		f_sga  = self.Stars.fractions._fs(M_use, a=a, cosmo=cosmo)
		f_clm  = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m + f_sga

		rho_i      = self.DarkMatter.real(cosmo, r_integral, M_use, a)
		rho_cga    = self.Stars.real(cosmo, r_integral, M_use, a)
		rho_gas    = self.Gas.real(cosmo, r_integral, M_use, a)

		#Need to add the offset manually now since scipy deprecates initial != 0
		#Offset required so that the integrated array has the same size as the profile array
		dlnr  = np.log(r_integral[1]) - np.log(r_integral[0])
		dV    = 4 * np.pi * r_integral**3 * dlnr
		M_i   = integrate.cumulative_simpson(dV * rho_i  , axis = -1, initial = 0) + dV[0] * rho_i[:, [0]]
		M_cga = integrate.cumulative_simpson(dV * rho_cga, axis = -1, initial = 0) + dV[0] * rho_cga[:, [0]]
		M_gas = integrate.cumulative_simpson(dV * rho_gas, axis = -1, initial = 0) + dV[0] * rho_gas[:, [0]]

		#We intentionally set Extrapolate = True. This is to handle behavior at extreme small-scales (due to stellar profile)
		#and radius limits at largest scales. Using extrapolate=True does not introduce numerical artifacts into predictions
		ln_M_NFW = [interpolate.PchipInterpolator(np.log(r_integral), np.log(M_i[m_i]),   extrapolate = True) for m_i in range(M_i.shape[0])]
		ln_M_cga = [interpolate.PchipInterpolator(np.log(r_integral), np.log(M_cga[m_i]), extrapolate = True) for m_i in range(M_i.shape[0])]
		ln_M_gas = [interpolate.PchipInterpolator(np.log(r_integral), np.log(M_gas[m_i]), extrapolate = True) for m_i in range(M_i.shape[0])]

		del M_cga, M_gas, rho_i, rho_cga, rho_gas

		relaxation_fraction = np.ones_like(M_i)

		for m_i in range(M_i.shape[0]):
			
			counter  = 0
			max_rel_diff = np.inf #Initializing variable at infinity
			
			while max_rel_diff > self.reltol:

				with np.errstate(over = 'ignore'):
					r_f  = r_integral*relaxation_fraction[m_i]
					M_f  = f_clm[m_i]*M_i[m_i] + np.exp(ln_M_cga[m_i](np.log(r_f))) + np.exp(ln_M_gas[m_i](np.log(r_f)))

				relaxation_fraction_new = self.a*( (M_i[m_i]/M_f)**self.n - 1 ) + 1

				diff     = relaxation_fraction_new/relaxation_fraction[m_i] - 1
				abs_diff = np.abs(diff)
				
				max_rel_diff = np.max(abs_diff[safe_range])
				
				relaxation_fraction[m_i] = relaxation_fraction_new

				counter += 1

				#Though we do a while loop, we break it off after 10 tries
				#this seems to work well enough. The loop converges
				#after two or three iterations.
				if (counter >= self.max_iter) & (max_rel_diff > self.reltol): 
					
					med_rel_diff = np.max(abs_diff[safe_range])
					warn_text = ("Profile of halo index %d did not converge after %d tries. " % (m_i, counter) +
								 "Max_diff = %0.5f, Median_diff = %0.5f. Try increasing max_iter." % (max_rel_diff, med_rel_diff)
								)
					
					warnings.warn(warn_text, UserWarning)
					break

		ln_M_clm = np.vstack([np.log(f_clm[m_i]) + ln_M_NFW[m_i](np.log(r_integral/relaxation_fraction[m_i])) for m_i in range(M_i.shape[0])])
		ln_M_clm = interpolate.CubicSpline(np.log(r_integral), ln_M_clm, axis = -1, extrapolate = False)
		log_der  = ln_M_clm.derivative(nu = 1)(np.log(r_use))
		lin_der  = log_der * np.exp(ln_M_clm(np.log(r_use))) / r_use
		prof     = 1/(4*np.pi*r_use**2) * lin_der
		prof     = np.clip(prof, 0, None) #If prof < 0 due to interpolation errors, then force it to 0.
		
		arg  = (r_use[None, :] - self.cutoff)
		arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
		kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
		prof = np.where(np.isfinite(prof), prof, 0) * kfac

		#Handle dimensions so input dimensions are mirrored in the output
		if np.ndim(r) == 0:
			prof = np.squeeze(prof, axis=-1)
		if np.ndim(M) == 0:
			prof = np.squeeze(prof, axis=0)

		return prof


class SatelliteStars(CollisionlessMatter):
	"""
	Stolen from BaryonForge
	"""
	def __init__(self, **kwargs):
		super().__init__(**kwargs)


	def _real(self, cosmo, r, M, a):

		M_use = np.atleast_1d(M)
		
		f_sga  = self.Stars.fractions._fs(M_use, a=a, cosmo=cosmo)
		f_clm  = 1 - cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m + f_sga
		
		if np.ndim(M) == 0: 
			f_clm = np.squeeze(f_clm, axis = 0)
			f_sga = np.squeeze(f_sga, axis = 0)

		prof   = super()._real(cosmo, r, M, a) * (1/f_clm)
		return prof

class DarkMatter(bfg.Profiles.Schneider19.SchneiderProfiles):
    def _real(self, cosmo, r, M, a):

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        z = 1/a - 1

        if (self.cdelta is None) and (self.c_M_relation is None):
            c_M_relation = ccl.halos.concentration.ConcentrationDiemer15(mass_def = self.mass_def) #Use the diemer calibration
        elif self.c_M_relation is not None:
            c_M_relation = self.c_M_relation
        else:
            assert self.cdelta is not None, "Either provide cdelta or a c_M_relation input"
            c_M_relation = ccl.halos.concentration.ConcentrationConstant(self.cdelta, mass_def = self.mass_def)
            
        c   = c_M_relation(cosmo, M_use, a)
        c   = np.where(np.isfinite(c), c, 1) #Set default to r_s = R200c if c200c broken (normally for low mass obj in some cosmologies)
        R   = self.mass_def.get_radius(cosmo, M_use, a)/a #in comoving Mpc
        r_s = R/c
        r_t = R*self.epsilon
        
        r_s, r_t = r_s[:, None], r_t[:, None]

        
        #Get the normalization (rho_c) numerically
        #The analytic integral doesn't work since we have a truncation radii now.
        #We loop over every halo, instead of vectorizing, since the integral limits
        #now depend on the halo radius. 
        Normalization = np.zeros_like(M_use)
        for m_i in range(M_use.size):
            r_integral     = np.geomspace(self.r_min_int, R[m_i], self.r_steps)
            prof_integral  = 1/(r_integral/r_s[m_i] * (1 + r_integral/r_s[m_i])**2) * 1/(1 + (r_integral/r_t[m_i])**2)**2
            Normalization[m_i] = np.trapz(4*np.pi*r_integral**2 * prof_integral, r_integral)


        rho_c = M_use/Normalization
        rho_c = rho_c[:, None]

        arg  = (r_use[None, :] - self.cutoff)
        arg  = np.where(arg > 30, np.inf, arg) #This is to prevent an overflow in the exponential
        kfac = 1/( 1 + np.exp(2*arg) ) #Extra exponential cutoff
        prof = rho_c/(r_use/r_s * (1 + r_use/r_s)**2) * 1/(1 + (r_use/r_t)**2)**2 * kfac
        
        #Handle dimensions so input dimensions are mirrored in the output
        if np.ndim(r) == 0: prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0: prof = np.squeeze(prof, axis=0)


        return prof
if __name__ == "__main__":
	# Example usage
	cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96)
	mdef = ccl.halos.massdef.MassDef200c
	# The Tinker 2008 mass function
	hmf = ccl.halos.MassFuncTinker08(mass_def=mdef)
	bM = ccl.halos.HaloBiasTinker10(mass_def=mdef)
	concentration = ccl.halos.concentration.duffy08.ConcentrationDuffy08(mass_def=mdef)
	hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=bM, mass_def=mdef)

	hod = BaryonifiedHOD(mass_def=mdef, concentration=concentration)

	hod.real(cosmo, [1], [1e12], [1])
