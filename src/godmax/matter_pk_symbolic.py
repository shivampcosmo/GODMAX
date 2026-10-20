import jax.numpy as jnp
from jax import jit

@jit
def symbolic_As(Omm, Omb, h, ns, sigma8):
    """
    Compute As from sigma8 and other cosmological parameters
    using a symbolic approximation

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
    Returns:
        :As (float): The amplitude of the primordial power spectrum
    """

    # b = [0.955337, 68.807808, 0.5159, 1.0, 1.204974, 0.757878, 2.141755, 3.037621,
    #      4.71485, 0.962402, 5.467293, 1.0, 1.18861011269465, 0.196998, 0.538841,
    #      46.237621, 0.917115, 13.52331, 35.327904, 1e-06, 1.0, 10.883345, 0.730041]
    b = [0.95534, 68.80781, 0.5159, 1.18861, 0.197, 0.53884, 0.01983, 0.76405, 0.29247, 10.88335, 
         0.73004, 1.20497, 0.75788, 2.14175, 3.03762, 4.71485, 5.46729, 0.9624]
    corr = (
        b[0]*ns*(b[1]*Omm)**(-b[2]*h)
        + (b[3]*jnp.sqrt(h) + b[4]*ns)*(-b[5]*ns + (Omb - b[6])/(b[7]*Omm - b[8]*Omb))
        + (b[9]*Omm)**(-b[10]*h)
        + (b[11]*Omm)**(b[12]*h)*(b[13]*Omb + (b[14]*h)**(-b[15]*Omm))/(b[16]*Omm + b[17]*Omb)
    )
    
    As = (sigma8 * jnp.exp(corr)) ** 2 * 1.e-9

    return As

@jit
def symbolic_D(Omm, z):
    """
    Compute the linear growth factor for a LCDM universe at a given
    redshift using a symbolic approximations

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :z (float): Redshift to evaluate at

    Returns:
        :D (float): The linear growth factor at redshift z

    """

    a = 1/(1+z)
    x = (Omm - 1) / Omm * a ** 3

    # Operon fit
    # b = [1.36854354591569, 0.854158584488509, 0.64694, 0.969454954982762]
    # b = [1.3685, 0.8542, 0.6469, 0.9695]  #  Rounded to 4 decimal places
    # f = b[0]/jnp.sqrt((b[1] - x)**b[2] + b[3])
    
    # Operon fit but ensure 2F1(1/3,1,11/6;0) = 1
    b = [0.8665, 0.6457, 0.9522]
    f = jnp.sqrt(b[2] + b[0]**b[1]) / jnp.sqrt((b[0] - x)**b[1] + b[2])
    
    D = a * f

    return D

@jit
def get_eisensteinhu_nw(Omm, Omb, h, ns, sigma8, z, k, TCMB=2.7255):
    """
    Compute the no-wiggles Eisenstein & Hu approximation
    to the linear P(k) at redshift z.

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :z (float): Redshift to evaluate at
        :k (jnp.ndarray): k values to evaluate P(k) at [h / Mpc]
        :TCMB (float, default=2.7255): z=0 CMB Temperature [K]

    Returns:
        :pk (jnp.ndarray): Approximate linear power spectrum at corresponding k values [(Mpc/h)^3]
    """

    # Cosmological parameters
    As = symbolic_As(Omm, Omb, h, ns, sigma8) * 1e9
    ombom0 = Omb / Omm
    om0h2 = Omm * h**2
    ombh2 = Omb * h**2
    theta2p7 = TCMB / 2.7

    # Compute scale factor s, alphaGamma, and effective shape Gamma
    s = 44.5 * jnp.log(9.83 / om0h2) / jnp.sqrt(1.0 + 10.0 * ombh2**0.75)
    alphaGamma = 1.0 - 0.328 * \
        jnp.log(431.0 * om0h2) * ombom0 + 0.38 * \
        jnp.log(22.3 * om0h2) * ombom0**2
    Gamma = Omm * h * (alphaGamma + (1.0 - alphaGamma) /
                       (1.0 + (0.43 * k * h * s)**4))

    # Compute q, C0, L0, and tk_eh
    q = k * theta2p7**2 / Gamma
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    L0 = jnp.log(2.0 * jnp.exp(1.0) + 1.8 * q)
    tk_eh = L0 / (L0 + C0 * q**2)

    kpivot = 0.05

    #  Linear growth factor
    D = symbolic_D(Omm, z)

    pk = (
        2 * jnp.pi ** 2 / k ** 3
        * (As * 1e-9) * (k * h / kpivot) ** (ns - 1)
        * (2 * k ** 2 * 2998**2 / 5 / Omm) ** 2
        * tk_eh ** 2
        * D ** 2
    )

    return pk


@jit
def symbolic_pklin(Omm, Omb, h, ns, sigma8, z, k):
    """
    Compute the symbolic approximation to the linear P(k) at redshift z.

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :z (float): Redshift to evaluate at
        :k (jnp.ndarray): k values to evaluate P(k) at [h / Mpc]

    Returns:
        :pk (jnp.ndarray): Approximate linear power spectrum at corresponding k values [(Mpc/h)^3]
    """

    c = [0.013924, 5.1771, 20.636, 0.49092, 3.4224, 0.35621, 0.016739, 0.18401, 0.58832, 5.108, 10.783,
         0.0043879, 0.11547, 1.3869, 2.9301, 32.014, 0.002192, 0.002926, 316.2, 1.2158, 0.095822,
         0.074921, 0.0067841, 0.0093912, 0.022678, 0.00093762, 0.64834, 0.006882, 0.6518, 0.11855,
         2.6863, 42.261, 121.6, 123.21, 14.51, 1378.1, 22.375, 1.0369, 75.439, 3.6282, 0.15279, 41.884,
         1052.3, 0.22687, 0.022003, 236.72, 0.16852, 0.55815, 0.040376, 0.4352, 0.011574, 0.043423, 0.001834,
         205.7, 0.26165, 0.0073544, 0.99135, 0.18444, 0.089257, 0.0074827, 7.7431e-05, 1.1788, 0.44205, 0.098723,
         0.01075, 0.007492, 0.82882, 1.1882, 383.17, 397.51, 4.4661, 0.81134, 0.0052959, 0.15939]

    # Compute raw SR fit
    F1 = (
        c[0]*(k + c[1]*(1-c[2]*k*Omm**(-c[3])) *
              (-Omm + (c[4] * Omb + c[5]*k)/jnp.sqrt(c[6] + (Omb + c[7]*k)**2)
               - c[8] * (c[9]*Omm)**(-c[10]*Omb)*k /
               jnp.sqrt((c[11]+h**2)*(c[12]*k**2 + Omm**2))
               + c[13]*(c[14]*Omm)**(-c[15]*Omb + c[16]*k)*k /
               jnp.sqrt(c[17] + (c[18]*Omm)**(c[19]*h)*(c[20]*Omm - k)**2)
               ) /
              (jnp.sqrt((c[21] + 1/(c[22] + (c[23] - c[24]*Omb - k)**2 /
                                    (c[25]+(k-c[26]*Omb)**2)))*(c[27] + (c[28]*Omb + c[29] - k)**2)))
              ))
    F2 = (1 / jnp.sqrt(1 +
                       ((c[30]*Omm)**(-c[31]*Omb)*(jnp.cos(c[32]*k) - jnp.cos(c[33]*k))/jnp.sqrt(1 + jnp.cos(c[34]*k)**2)
                        + c[35] * (c[36]*Omm)**(c[37]*h) * k**2 *
                        (c[38]*k)**(-c[39]*Omb) / jnp.sqrt(h**2 + c[40])
                        )**2))
    F3 = (
        (c[41]*(c[42]*Omm)**(c[43]*h)/jnp.sqrt(c[44] + h**2) -
         c[45]*((c[46]*Omm)**(c[47]*h*Omb - c[48]*h**2)))*k
        + c[49]/jnp.sqrt(c[50] + (c[51]*Omm - k)**2))
    F4 = (((c[52]*k*(c[53]*Omm)**(c[54]*h)/jnp.sqrt(c[55] + (c[56]*Omb - k)**2)/(Omb + c[57]*k))
           + (c[58]*k + c[59])/jnp.sqrt(c[60] + k**2*(c[61]*Omm)**(-c[62]*h)) - c[63] - c[64]/jnp.sqrt(k**2 + c[65]))
          / jnp.sqrt(1 + c[66]*(-Omm +
                                (c[67]*Omb + c[68]*Omm/Omb*(c[69]*Omm)**(-c[70]*h) + c[71]*k) /
                                jnp.sqrt(c[72] + (k - c[73])**2))**2))

    # Enforce low-k behaviour. F2->1 so we don't need a new variable for this
    F1_lowk = (c[0]*c[1]*(-Omm + c[4]*Omb/jnp.sqrt(c[6] + Omb**2)) /
               jnp.sqrt((c[21] + 1/(c[22] + (c[23]-c[24]*Omb)**2/(c[25]+(c[26]*Omb)**2)))*(c[27] + (c[28]*Omb + c[29])**2)))
    F3_lowk = c[49] / jnp.sqrt(c[50] + (c[51]*Omm)**2)
    F4_lowk = ((c[59]/jnp.sqrt(c[60]) - c[63] - c[64]/jnp.sqrt(c[65])) /
               jnp.sqrt(1 + c[66]*(-Omm + (c[67]*Omb + c[68]*Omm/Omb*(c[69]*Omm)**(-c[70]*h))/jnp.sqrt(c[72]+c[73]**2))**2))
    F5 = - F1_lowk * (jnp.cos(F3_lowk) + F4_lowk)

    # Combine fits
    logF = F1 * (F2 * jnp.cos(F3) + F4) + F5
    pk_nw = get_eisensteinhu_nw(Omm, Omb, h, ns, sigma8, z, k)
    pk = jnp.exp(logF) * pk_nw

    return pk


@jit
def symbolic_ksigma(Omm, Omb, h, ns, sigma8, z):
    """
    Symbolic approximation to the nonlinear scale used for halofit, ksigma.

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h

    Returns:
        :ksigma (float): The nonlinear scale for this cosmology
    """

    # b = [0.345776345776, 0.014767, 0.082499, 4.64239, 0.473773, 0.384702482445851, 2.00514720773505, 0.022065, 0.295824, 0.496218, 0.03355,
    #      0.646666, 1.1394, 8.497899, 4.570055, 0.644841, 0.102213, 0.378249, 1.23868, 18.267942, 0.104309, 0.343493, 0.217777, 0.164372,
    #      0.241262, 0.034817, 0.514162, 0.498318, 1.10023, 0.887106, 0.725573, 0.140535, 0.61402, 1.027476, 3.036082, 0.913197, 0.454534,
    #      0.244376, 113.504471, 97.353813, 0.153385, 0.963861, 1.309369, 1.615641, 0.370806]
    b = [0.3458, 0.01477, 0.0825, 4.642, 0.4738, 0.3847, 2.005, 0.02206, 0.2958, 0.4962, 0.03355, 0.6467, 1.139, 8.498,
         4.57, 0.6448, 0.1022, 0.3782, 1.239, 18.27, 0.1043, 0.3435, 0.2178, 0.1644, 0.2413, 0.03482, 0.5142, 0.4983,
         1.1, 0.8871, 0.7256, 0.1405, 0.614, 1.027, 3.036, 0.9132, 0.4545, 0.2444, 113.5, 97.35, 0.1534, 0.9639, 1.309, 1.616, 0.3708]

    F1 = -b[0]*sigma8 - (b[1]*h)**(b[2]*z*(b[3]*Omm) **
                                   (b[4]*sigma8 - b[5]*jnp.sqrt(z)))
    F2 = b[6]/h*(b[7]*z)**(b[8]*ns + b[9]*sigma8 - b[10]*z)
    F3 = (
        -b[11]*z*(-b[12]*Omb + (b[13]*Omm + jnp.log(b[14]*ns))
                  ** (-b[15]*z*(b[16]*z)**(-b[17]*h)))
        - (b[18]*sigma8*(b[19]*Omm)**(-b[20]*z) - (b[21]*ns)**((-b[22]*ns-b[23]*z)*(b[24]*h)**(b[25]*z))) *
        (-b[26]*h + b[27]*ns + (-b[28]*Omb + b[29]*Omm)
         ** (-b[30]*Omb - b[31]*sigma8))
    )
    F4 = (b[32]*Omm + b[33]*sigma8 - b[34] + (b[35]*ns)**(b[36]*ns + b[37]*z)
          + (-b[38]*Omb + b[39]*Omm)**(b[40]*sigma8) + (b[41]*ns)**(b[42]*sigma8*(b[43]*h)**(-b[44]*z)))

    ksigma = jnp.exp(F1 + F2 + F3/F4)

    return ksigma

@jit
def symbolic_neff(Omm, Omb, h, ns, sigma8, z):
    """
    Symbolic approximation to the effective slope used for halofit, neff.

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h

    Returns:
        :neff (float): The effective slope for this cosmology
    """

    # b = [3.3207732533252, 6.373804, 0.230426, 0.164228, 0.063953, 0.146115, 0.217148,
    #      0.883513440757977, 0.745719937073723, 0.0536565259052097, 0.267958952080351, 6.477813, 2.350206,
    #      1.387198, 0.612163, 0.878426, 0.646557, 512.827332, 0.089426]
    b = [3.3208, 6.3738, 0.2304, 0.1642, 0.064, 0.1461, 0.2171, 0.8835, 0.7457, 0.0537, 0.268,
         6.4778, 2.3502, 1.3872, 0.6122, 0.8784, 0.6466, 512.8273, 0.0894]

    neff = ((b[0]*jnp.sqrt(ns) - b[1])*(b[2]*ns + b[3]*sigma8 + b[4]*z - b[5])**(b[6]*sigma8)
            + (b[7]*jnp.sqrt(-Omb + b[8]*Omm) - b[9] /
               (h*(b[10]*jnp.sqrt(ns) + (b[11]*Omm)**(-b[12]*z - b[13]))**(b[14]*Omm)))
            * (b[15]*h + b[16]*sigma8 + (b[17]*z)**(-b[18]*z))
            )

    return neff

@jit
def symbolic_C(Omm, Omb, h, ns, sigma8, z):
    """
    Symbolic approximation to the effective curvature used for halofit, C.

    Args:
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h

    Returns:
        :C (float): The effective curvate for this cosmology
    """

    # b = [4.91691, 0.042616, 1461.121094, 2181.431396, 11.153474, 0.478397, 0.09069, 0.034299, 0.043169, 0.037205,
    #      0.091072, 0.151021, 0.04674, 0.048536, 0.054956, 0.036314, 0.18055, 0.170738, 0.231533, 0.407454,
    #      0.593005, 1.840155, 1.028081, 0.026453, 0.065075, 0.064769, 0.192002, 0.003867, 0.000556, 0.000851,
    #      0.000177, 0.03328, 0.041811, 0.060018]
    b = [4.917, 0.04262, 1461.0, 2181.0, 11.15, 0.4784, 0.09069, 0.0343, 0.04317, 0.0372, 0.09107, 0.151,
         0.04674, 0.04854, 0.05496, 0.03631, 0.1805, 0.1707, 0.2315, 0.4075, 0.593, 1.84, 1.028, 0.02645,
         0.06507, 0.06477, 0.192, 0.003867, 0.000556, 0.000851, 0.000177, 0.03328, 0.04181, 0.06002]

    C = (
        (b[0]*z)**(-b[1]*z) *
        (b[2]*Omm - b[3]*Omb + b[4]*z) **
        (b[5] + (b[6]*Omm)**(-b[7]*h+b[8]*ns-b[9]*sigma8) -
         (b[10]*z + b[11]*ns)**(-b[12]*Omm-b[13]*h-b[14]*sigma8-b[15]*z))
        * (-b[16]*h - b[17]*sigma8 + (b[18]*sigma8)**((b[19]*Omm-b[20]*Omb+b[21])**(-b[22]*h)*(b[23]*Omm + b[24]*h + b[25]*ns + b[26]*sigma8 - b[27]*z))
           - (-b[28]*Omm + b[29]*ns + b[30]*z)**(b[31]*h+b[32]*ns+b[33]*sigma8))
    )

    return C

@jit
def symbolic_pkhalofit(k, plin_mat, Omm, Omb, h, ns, sigma8, z_array, jz):
    """
    Given a linear power spectrum, compute the halofit approximation to
    the nonlinear powerspectrum. The halofit variables ksigma, neff and C
    are computed using their symbolic approximations.

    Args:
        :k (jnp.ndarray): k values to evaluate P(k) at [h / Mpc]
        :plin (jnp.ndarray): The linear matter power spectrum [(Mpc/h)^3]
        :Omm (float): The z=0 total matter density parameter, Omega_m
        :Omb (float): The z=0 baryonic density parameter, Omega_b
        :h (float): Hubble constant, H0, divided by 100 km/s/Mpc
        :ns (float): Spectral tilt of primordial power spectrum
        :sigma8 (float): Root-mean-square density fluctuation when the linearly
            evolved field is smoothed with a top-hat filter of radius 8 Mpc/h
        :z (float): The redshift to evaluate P(k) at

    Returns:
        :p_nl (jnp.ndarray): The predicted non-linear P(k) [(Mpc/h)^3]
    """
    z = z_array[jz]
    plin = plin_mat[:,jz]
    ksigma = symbolic_ksigma(Omm, Omb, h, ns, sigma8, z)
    neff = symbolic_neff(Omm, Omb, h, ns, sigma8, z)
    C = symbolic_C(Omm, Omb, h, ns, sigma8, z)

    pars = [1.5222, 2.8553, 2.3706, 0.9903, 0.2250, 0.6083,
            -0.5642, 0.5864, 0.5716, -1.5474, 0.3698, 2.0404, 0.8161, 0.5869,
            0.1971, -0.0843, 0.8460, 5.2105, 3.6902, -0.0307, -0.0585,
            0.0743, 6.0835, 1.3373, -0.1959, -5.5274, 2.0379, -0.7354, 0.3157,
            1.2490, 0.3980, -0.1682]

    y = k / ksigma
    a = 1 / (1 + z)

    # 1 halo term parameters
    an = (pars[0] + pars[1] * neff + pars[2] * neff ** 2 + pars[3] * neff ** 3
          + pars[4] * neff ** 4 - pars[5] * C)
    an = 10. ** an
    bn = pars[6] + pars[7] * neff + pars[8] * neff ** 2 + pars[9] * C
    bn = 10. ** bn
    cn = pars[10] + pars[11] * neff + pars[12] * neff ** 2 + pars[13] * C
    cn = 10. ** cn
    gamma = pars[14] + pars[15] * neff + pars[16] * C
    nu = 10. ** (pars[17] + pars[18] * neff)
    Omz = Omm / a ** 3 / (Omm / a ** 3 + 1. - Omm)
    f1 = Omz ** pars[19]
    f2 = Omz ** pars[20]
    f3 = Omz ** pars[21]

    # 2 halo term parameters
    alpha = jnp.abs(pars[22] + pars[23] * neff +
                    pars[24] * neff ** 2 + pars[25] * C)
    beta = (pars[26] + pars[27] * neff + pars[28] * neff ** 2
            + pars[29] * neff ** 3 + pars[30] * neff ** 4 + pars[31] * C)

    # Predict 1 halo term
    deltaH2 = an * y ** (3 * f1) / (1 + bn * y ** f2 +
                                    (cn * f3 * y) ** (3 - gamma))
    deltaH2 /= 1 + nu / y ** 2
    ph = deltaH2 * (2 * jnp.pi ** 2) / k ** 3

    # Predict 2 halo term
    deltaL2 = k ** 3 * plin / (2 * jnp.pi ** 2)
    pq = plin * (1 + deltaL2) ** beta / \
        (1 + alpha * deltaL2) * jnp.exp(- y/4 - y**2/8)

    # Total prediction
    p_nl = ph + pq

    return p_nl