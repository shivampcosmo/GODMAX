import jax
import jax.numpy as jnp

"""
Copied from Adam Coogan's implementation:
https://github.com/adam-coogan/jax/blob/gamma-complex/third_party/scipy_special_gamma/gamma.py
"""


SMALLX = 7
SMALLY = 7
TAYLOR_RADIUS = 0.2


@jax.jit
def loggamma_recurrence(z):
    """
    Backward recurrence relation.
    See Proposition 2.2 in [1] and the Julia implementation [2].
    """
    signflips = 0
    sb = 0
    nsb = 0
    shiftprod = z
    z = z + 1
    init_val = (signflips, sb, nsb, shiftprod, z)

    def cond_fun(val):
        z = val[4]
        return z.real <= SMALLX

    def body_fun(val):
        signflips, sb, nsb, shiftprod, z = val
        shiftprod = shiftprod * z
        nsb = jnp.signbit(shiftprod.imag).astype(int)
        signflips = signflips + jnp.logical_and(nsb != 0, sb == 0).astype(int)
        sb = nsb
        z = z + 1
        return (signflips, sb, nsb, shiftprod, z)

    signflips, _, _, shiftprod, z = jax.lax.while_loop(cond_fun, body_fun, init_val)

    return loggamma_stirling(z) - jnp.log(shiftprod) - signflips * 2 * jnp.pi * 1j


@jax.jit
def loggamma_stirling(z):
    """
    Stirling series for log-Gamma.
    The coefficients are B[2*n]/(2*n*(2*n - 1)) where B[2*n] is the
    (2*n)th Bernoulli number. See (1.1) in [1].
    """
    coeffs = jnp.array(
        [
            -2.955065359477124183e-2, 6.4102564102564102564e-3,
            -1.9175269175269175269e-3, 8.4175084175084175084e-4,
            -5.952380952380952381e-4, 7.9365079365079365079e-4,
            -2.7777777777777777778e-3, 8.3333333333333333333e-2,
        ]
    )
    rz = 1.0 / z
    rzz = rz / z
    return (
        (z - 0.5) * jnp.log(z)
        - z
        + jnp.log(2 * jnp.pi) / 2
        + rz * jnp.polyval(coeffs, rzz)
    )


@jax.jit
def loggamma_taylor(z):
    """
    Taylor series for log-Gamma around z = 1.
    It is
    loggamma(z + 1) = -gamma*z + zeta(2)*z**2/2 - zeta(3)*z**3/3 ...
    where gamma is the Euler-Mascheroni constant.
    """
    coeffs = jnp.array(
        [
            -4.3478266053040259361e-2, 4.5454556293204669442e-2,
            -4.7619070330142227991e-2, 5.000004769810169364e-2,
            -5.2631679379616660734e-2, 5.5555767627403611102e-2,
            -5.8823978658684582339e-2, 6.2500955141213040742e-2,
            -6.6668705882420468033e-2, 7.1432946295361336059e-2,
            -7.6932516411352191473e-2, 8.3353840546109004025e-2,
            -9.0954017145829042233e-2, 1.0009945751278180853e-1,
            -1.1133426586956469049e-1, 1.2550966952474304242e-1,
            -1.4404989676884611812e-1, 1.6955717699740818995e-1,
            -2.0738555102867398527e-1, 2.7058080842778454788e-1,
            -4.0068563438653142847e-1, 8.2246703342411321824e-1,
            -5.7721566490153286061e-1,
        ]
    )
    z = z - 1
    return z * jnp.polyval(coeffs, z)


@jax.jit
def loggamma(z):
    """
    Compute the principal branch of log-Gamma.

    The order of the last three branches in the conditional is switched relative
    to the scipy implementation to circumvent jax's lack of support for recursion.
    """
    z = z + 0.0j  # make complex

    cond_1 = jnp.isnan(z)
    cond_2 = jnp.logical_and(z.real <= 0, z == jnp.floor(z.real))
    cond_3 = jnp.logical_or(z.real > SMALLX, jnp.abs(z.imag) > SMALLY)
    cond_4 = jnp.abs(z - 1) <= TAYLOR_RADIUS
    cond_5 = jnp.abs(z - 2) <= TAYLOR_RADIUS
    cond_8 = z.real < 0.1
    cond_6a = z.imag >= 0
    cond_6 = jnp.logical_and(cond_6a, ~cond_8)  # z.real >= 0.1
    cond_7a = z.imag < 0
    cond_7 = jnp.logical_and(cond_7a, ~cond_8)  # z.real >= 0.1

    fn_nan = lambda _: jnp.nan + 1j * jnp.nan
    fn_loggamma_taylor_2 = lambda z: jnp.log(z - 1) + loggamma_taylor(z - 1)
    fn_loggamma_recurrence_conj = lambda z: loggamma_recurrence(
        z.conjugate()
    ).conjugate()

    # Recur into these branches if z < 0.1
    cond_1_1mz = jnp.isnan(1 - z)
    cond_2_1mz = jnp.logical_and((1 - z).real <= 0, 1 - z == jnp.floor(z.real))
    cond_3_1mz = jnp.logical_or((1 - z).real > SMALLX, jnp.abs((1 - z).imag) > SMALLY)
    cond_4_1mz = jnp.abs((1 - z) - 1) <= TAYLOR_RADIUS
    cond_5_1mz = jnp.abs((1 - z) - 2) <= TAYLOR_RADIUS
    cond_6a_1mz = (1 - z).imag >= 0

    fn_z_lt_0_1 = lambda z: jax.lax.cond(
        cond_1_1mz,
        fn_nan,
        lambda z: jax.lax.cond(
            cond_2_1mz,
            fn_nan,
            lambda z: jax.lax.cond(
                cond_3_1mz,
                loggamma_stirling,
                lambda z: jax.lax.cond(
                    cond_4_1mz,
                    loggamma_taylor,
                    lambda z: jax.lax.cond(
                        cond_5_1mz,
                        fn_loggamma_taylor_2,
                        lambda z: jax.lax.cond(
                            cond_6a_1mz,
                            loggamma_recurrence,
                            fn_loggamma_recurrence_conj,  # cond_7a holds
                            z,
                        ),
                        z,
                    ),
                    z,
                ),
                z,
            ),
            z,
        ),
        z,
    )

    def fn_recursive(z):
        # Reflection formula; see Proposition 3.1 in [1]
        tmp = jnp.copysign(2 * jnp.pi, z.imag) * jnp.floor(0.5 * z.real + 0.25)
        # TODO: check that sin(pi * x) is accurate enough
        return (
            jnp.log(jnp.pi)
            + 1j * tmp
            - jnp.log(jnp.sin(jnp.pi * z))
            - fn_z_lt_0_1(1 - z)
        )

    # General case
    return jax.lax.cond(
        cond_1,
        fn_nan,
        lambda z: jax.lax.cond(
            cond_2,
            fn_nan,
            lambda z: jax.lax.cond(
                cond_3,
                loggamma_stirling,
                lambda z: jax.lax.cond(
                    cond_4,
                    loggamma_taylor,
                    lambda z: jax.lax.cond(
                        cond_5,
                        fn_loggamma_taylor_2,
                        lambda z: jax.lax.cond(
                            cond_6,
                            loggamma_recurrence,
                            lambda z: jax.lax.cond(
                                cond_7,
                                fn_loggamma_recurrence_conj,
                                fn_recursive,  # cond_8 holds
                                z,
                            ),
                            z,
                        ),
                        z,
                    ),
                    z,
                ),
                z,
            ),
            z,
        ),
        z,
    )


@jax.jit
def cgamma(z):
    """
    Compute Gamma(z) using loggamma.
    """
    fn_nan = lambda _: jnp.nan + 1j * jnp.nan
    fn_gamma = lambda z: jnp.exp(loggamma(z))
    return jax.lax.cond(
        jnp.logical_and(z.real <= 0, z == jnp.floor(z.real)), fn_nan, fn_gamma, z
    )


@jax.jit
def crgamma(z):
    """
    Compute 1/Gamma(z) using loggamma.
    """
    fn_nan = lambda _: 0.0 + 0j
    fn_gamma = lambda z: jnp.exp(-loggamma(z))
    return jax.lax.cond(
        jnp.logical_and(z.real <= 0, z == jnp.floor(z.real)), fn_nan, fn_gamma, z
    )