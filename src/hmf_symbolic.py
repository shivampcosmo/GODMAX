import jax.numpy as jnp
from jax import jit
from matter_pk_symbolic import symbolic_D, symbolic_As

@jit
def symbolic_lnsigma_corr(R, Omm, Omb, h, ns):
    """
    Compute log(sigma(R) / sqrt(1e9As) / D) using a symbolic approximation,
    where sigma(R) is the RMS linear fluctuation in spheres of radius R,
    As is the amplitude of the primordial power spectrum, and D is the 
    linear growth factor. This quantity is independent of redshift and As, and
    therefore redshift and sigma8 do not appear as arguments.
    
    Args:
        :R (float): The smoothing radius of the density field [Mpc / h]
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
            
    Returns:
        :ypred (float): The predicted quantity
    """

    # b = [0.496037204724, 1.173170991936, 0.235984, 3.68971367925806, 0.544524168693354, 
    #      0.0964844742384783, 7.67857, 0.034277, 0.978223, 2.081887, 0.415016867127108, 0.601191, 
    #      0.218860948982, 0.25609340484599996, 2.99433, 0.010821, 0.226344, 220.525447854224, 
    #      0.156336, 0.002714, 0.04580338310877485, 4.25944805847718, 0.270237, 0.003452, 
    #      0.0639725099618148, 0.034096, 0.000824]
    b = [0.496, 1.173, 0.236, 3.69, 0.5445, 0.09648, 7.679, 0.03428, 0.9782, 2.082, 0.415,
         0.6012, 0.2189, 0.2561, 2.994, 0.01082, 0.2263, 220.5, 0.1563, 0.002714, 0.0458,
         4.259, 0.2702, 0.003452, 0.06397, 0.0341, 0.000824]  # rounded to ~4 sig figs

    ypred = (b[0]*Omm + b[1]*ns + jnp.log(b[2]*h) - b[3]*jnp.sqrt(b[4]*Omm - Omb + b[5]*ns)
             + (b[6]*R)**(b[7]*h) * (jnp.log(b[8]*R) - b[9]*h) *
             (b[10]*jnp.sqrt(ns) - b[11]*ns + b[12]*Omb*h - b[13]*Omm*h)
             - (b[14]*Omm + (b[15]*R)**(b[16]*ns) + jnp.log(b[17]*Omm*h))
             * (b[18]*h + (b[19]*R)**(Omb/(b[20]*ns - b[21]*Omm) + b[22]) - (b[23]*R)**(b[24]*Omb/Omm + b[25]*Omm + b[26]*R))
             )

    return ypred

@jit
def symbolic_dlnsigmadR(R, Omm, Omb, h, ns):
    """
    Compute dlog(sigma(R)) / dR using a symbolic approximation,
    where sigma(R) is the RMS linear fluctuation in spheres of radius R.
    This quantity is independent of redshift and As, and
    therefore redshift and sigma8 do not appear as arguments.
    This function is the analytic derivative of symbolic_lnsigma_corr.
    
    Args:
        :R (float): The smoothing radius of the density field [Mpc / h]
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
            
    Returns:
        :ypred (float): The predicted quantity
    """
    
    b = [0.496, 1.173, 0.236, 3.69, 0.5445, 0.09648, 7.679, 0.03428, 0.9782, 2.082, 0.415,
         0.6012, 0.2189, 0.2561, 2.994, 0.01082, 0.2263, 220.5, 0.1563, 0.002714, 0.0458,
         4.259, 0.2702, 0.003452, 0.06397, 0.0341, 0.000824]  # rounded to ~4 sig figs

    ypred = (
        (b[10]*jnp.sqrt(ns) - b[11]*ns + b[12]*Omb*h - b[13]*Omm*h)
        * (b[7]*h*(jnp.log(b[8]*R) - b[9]*h) + 1) * (b[6]*R)**(b[7]*h)
        - b[16]*ns*(b[15]*R)**(b[16]*ns)
        * (b[18]*h + (b[19]*R)**(Omb/(b[20]*ns - b[21]*Omm) + b[22]) - (b[23]*R)**(b[24]*Omb/Omm + b[25]*Omm + b[26]*R))
        - (b[14]*Omm + (b[15]*R)**(b[16]*ns) + jnp.log(b[17]*Omm*h))
        * ((Omb/(b[20]*ns - b[21]*Omm) + b[22])*(b[19]*R)**(Omb/(b[20]*ns - b[21]*Omm) + b[22])
           - (b[23]*R)**(b[24]*Omb/Omm + b[25]*Omm + b[26]*R)*(b[26]*R*jnp.log(b[23]*R)+b[24]*Omb/Omm + b[25]*Omm + b[26]*R))
    )
    
    ypred /= R

    return ypred


@jit
def symbolic_sigma(R, Omm, Omb, h, ns, sigma8, z):
    """
    Compute the RMS linear fluctuation in spheres of radius R 
    using a symbolic approximation.
    
    Args:
        :R (float): The smoothing radius of the density field [Mpc / h]
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :z (float): Redshift to evaluate at
            
    Returns:
        :ypred (float): The predicted quantity
    """
    
    sigma = symbolic_lnsigma_corr(R, Omm, Omb, h, ns)
    D = symbolic_D(Omm, z)
    As = symbolic_As(Omm, Omb, h, ns, sigma8)
    sigma = jnp.exp(sigma) * jnp.sqrt(1e9 * As) * D
    
    return sigma