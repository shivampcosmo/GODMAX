from jax.numpy import arange, exp, log, ndim, pi, sqrt
import jax.numpy as jnp
from mcfitjax.loggamma_jax import loggamma as loggamma_orig
from jax import vmap
loggamma = vmap(loggamma_orig, in_axes=0)
from mcfitjax.loggamma_jax import cgamma as gamma


def _deriv(MK, deriv):
    """Real deriv is wrt :math:`t`, complex deriv is wrt :math:`\ln t`"""
    if deriv == 0:
        return MK

    if isinstance(deriv, complex):
        def MKderiv(z):
            return (-z) ** deriv.imag * MK(z)
        return MKderiv

    def MKderiv(z):
        poly = arange(deriv) + 1
        poly = poly - z if ndim(z) == 0 else poly - z.reshape(-1, 1)
        poly = poly.prod(axis=-1)
        return poly * MK(z - deriv)
    return MKderiv

def Mellin_BesselJ(nu, deriv=0):
    def MK(z):
        if jnp.shape(z) == ():
            return exp(log(2)*(z-1) + loggamma_orig(0.5*(nu+z)) - loggamma_orig(0.5*(2+nu-z)))
        else:
            return exp(log(2)*(z-1) + loggamma(0.5*(nu+z)) - loggamma(0.5*(2+nu-z)))
    return _deriv(MK, deriv)

def Mellin_SphericalBesselJ(nu, deriv=0):
    def MK(z):
        if jnp.shape(z) == ():
            return exp(log(2)*(z-1.5) + loggamma_orig((0.5*(nu+z))) - loggamma_orig((0.5*(3+nu-z))))
        else:
            return exp(log(2)*(z-1.5) + loggamma(jnp.asarray(0.5*(nu+z))) - loggamma(jnp.asarray(0.5*(3+nu-z))))    
    return _deriv(MK, deriv)

def Mellin_FourierSine(deriv=0):
    def MK(z):
        return exp(log(2)*(z-0.5) + loggamma(0.5*(1+z)) - loggamma(0.5*(2-z)))
    return _deriv(MK, deriv)

def Mellin_FourierCosine(deriv=0):
    def MK(z):
        return exp(log(2)*(z-0.5) + loggamma(0.5*z) - loggamma(0.5*(1-z)))
    return _deriv(MK, deriv)

def Mellin_Tophat(dim, deriv=0):
    def MK(z):
        return exp(log(2)*(z-1) + loggamma(1+0.5*dim) + loggamma(0.5*z) \
                - loggamma(0.5*(2+dim-z)))
    return _deriv(MK, deriv)

def Mellin_TophatSq(dim, deriv=0):
    if dim == 1:
        def MK(z):
            return -0.25*sqrt(pi) * exp(loggamma(0.5*(z-2)) - loggamma(0.5*(3-z)))
    elif dim == 3:
        def MK(z):
            return 2.25*sqrt(pi)*(z-2)/(z-6) \
                    * exp(loggamma(0.5*(z-4)) - loggamma(0.5*(5-z)))
    else:
        def MK(z):
            return exp(log(2)*(dim-1) + 2*loggamma(1+0.5*dim) \
                    + loggamma(0.5*(1+dim-z)) + loggamma(0.5*z) \
                    - loggamma(1+dim-0.5*z) - loggamma(0.5*(2+dim-z))) / sqrt(pi)
    return _deriv(MK, deriv)

def Mellin_Gauss(deriv=0):
    def MK(z):
        return 2**(0.5*z-1) * gamma(0.5*z)
    return _deriv(MK, deriv)

def Mellin_GaussSq(deriv=0):
    def MK(z):
        return 0.5 * gamma(0.5*z)
    return _deriv(MK, deriv)
