###################################################################################################
#
# constants.py 	        (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

"""
Useful physical and astronomical constants.
"""

###################################################################################################
# PHYSICS CONSTANTS IN CGS
###################################################################################################

C = 2.99792458E10
"""The speed of light in cm/s (PDG 2018)."""

M_PROTON = 1.672621898e-24
"""The mass of a proton in gram (PDG 2018)."""

KB = 1.38064852E-16
"""The Boltzmann constant in erg/K (PDG 2018)."""

SIGMA_SB = 5.670367E-5
"""The Stefan-Boltzmann constant in erg/cm^2/s/K^4 (PDG 2018)."""

G_CGS = 6.67408E-8
"""The gravitational constant G in :math:`{\\rm cm}^3 / g / s^2` (PDG 2018)."""

EV = 1.6021766208E-12
"""The energy 1 eV in erg (PDG 2018)."""

H = 6.626070040E-27
"""The planck constant in erg s (PDG 2018)."""

RYDBERG = 13.605693009
"""The Rydberg constant in eV (PDG 2018)."""

###################################################################################################
# ASTRONOMY UNIT CONVERSIONS
###################################################################################################

AU = 1.495978707E13
"""Astronomical unit (au) in centimeters (PDG 2018)."""

PC  = 3.08567758149E18
"""A parsec in centimeters (PDG 2018)."""

KPC = 3.08567758149E21
"""A kiloparsec in centimeters (PDG 2018)."""

MPC = 3.08567758149E24
"""A megaparsec in centimeters (PDG 2018)."""

YEAR = 31556925.2
"""A year in seconds (PDG 2018)."""

GYR = 3.15569252E16
"""A gigayear in seconds (PDG 2018)."""

MSUN = 1.9884754153381438E33
"""A solar mass, :math:`M_{\odot}`, in grams (IAU 2015)."""

G = 4.300917270038E-6
"""The gravitational constant G in :math:`{\\rm kpc} \ {\\rm km}^2 / M_{\odot} / s^2`. This 
constant is computed from the cgs version but given with more significant digits to preserve
consistency with the :math:`{\\rm Mpc}` and :math:`M_{\odot}` units."""

###################################################################################################
# ASTRONOMY CONSTANTS
###################################################################################################

RHO_CRIT_0_KPC3 = 2.77536627245708E2
"""The critical density of the universe at z = 0 in units of :math:`M_{\odot} h^2 / {\\rm kpc}^3`."""

RHO_CRIT_0_MPC3 = 2.77536627245708E11
"""The critical density of the universe at z = 0 in units of :math:`M_{\odot} h^2 / {\\rm Mpc}^3`."""

DELTA_COLLAPSE = 1.68647
"""The linear overdensity threshold for halo collapse according to the spherical top-hat collapse 
model (`Gunn & Gott 1972 <http://adsabs.harvard.edu/abs/1972ApJ...176....1G>`__). This number 
corresponds to :math:`3/5 (3\pi/2)^{2/3}` and is modified very slightly in a non-EdS universe."""
