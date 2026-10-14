"""Read-only consistency checks for GODMAX model objects.

These functions never modify a model. They compute diagnostics from the arrays a
`Profiles` / `get_Pkz` / `get_Cl` object already holds, and are shared by the pytest
suite (`tests/physics/`) and the pre-flight script (`scripts/preflight.py`).

Units follow GODMAX: r in comoving Mpc/h, k in h/Mpc, M in Msun/h.

Physics behind the checks
-------------------------
* Baryonification only redistributes mass inside a halo, so every mass-normalised
  Fourier profile obeys u(k -> 0) = 1 and baryonification ON/OFF must agree on large
  scales for matter, galaxies and lensing. tSZ follows pressure, not mass, and is
  exempt.
* The collisionless matter (CLM) is defined through its enclosed mass
  M_clm(<r) = f_clm M_nfw(r / zeta(r)). Any density built from it must integrate
  back to that mass; the shell transform conserves it exactly.
* The radial grid must contain the mass it is normalised to, otherwise
  normalising the profile hides missing (or extra) mass.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from godmax.base_class import get_vmapped_func


def _a(obj, name):
    return np.asarray(getattr(obj, name))


def _stats(x):
    x = np.asarray(x, dtype=float)
    return {"min": float(np.nanmin(x)), "median": float(np.nanmedian(x)), "max": float(np.nanmax(x))}


# --------------------------------------------------------------------------------------
# Collisionless-matter shells and mass closure
# --------------------------------------------------------------------------------------
def shell_masses(Mclm):
    """[M(<r0), diff(M)] along the radial (first) axis."""
    Mclm = np.asarray(Mclm)
    return np.concatenate([Mclm[:1], np.diff(Mclm, axis=0)], axis=0)


def shell_report(obj):
    """Negative CLM shell masses and telescoping error (shells must sum to every M(<r_i))."""
    M = _a(obj, "Mclm_mat")
    dm = shell_masses(M)
    neg = dm[1:] < 0
    return {
        "n_negative_shells": int(neg.sum()),
        "min_shell_over_Mend": float(np.min(dm[1:] / M[-1][None])),
        "telescoping_max_rel_err": float(np.max(np.abs(np.cumsum(dm, axis=0) / M - 1.0))),
    }


def rho_clm_closure(obj):
    """Trapezoid integral of rho_clm_mat over r_array divided by M_clm(r_max) - M_clm(r_0).

    This is what the DMB profile (u_dmb, P_mm) sees. Equal to 1 for a mass-conserving
    density; the residual mixes derivative error and trapezoid error on r_array."""
    r = _a(obj, "r_array")
    M = _a(obj, "Mclm_mat")
    integral = np.trapezoid(4 * np.pi * r[:, None, None] ** 2 * _a(obj, "rho_clm_mat"), r, axis=0)
    return integral / (M[-1] - M[0])


def coverage_report(obj):
    """How much of each component's target mass lies on the radial grid.

    clm_endpoint : M_clm(r_max) / (f_clm M_tot)
    gas_on_grid  : int rho_gas dV on r_array / (f_gas M_tot)
    dmb_on_grid  : int rho_dmb dV on r_array / M_tot
    All should be ~1. Values < 1 mean the grid truncates the halo; > 1 mean the profile
    extends beyond the radius it was normalised within (more mass on the grid than M_tot).
    """
    r = _a(obj, "r_array")
    pref = 4 * np.pi * r[:, None, None] ** 2
    Mtot = _a(obj, "Mtot_mat")
    out = {"clm_endpoint": _a(obj, "Mclm_mat")[-1] / (_a(obj, "fclm_mat") * Mtot),
           "dmb_on_grid": np.trapezoid(pref * _a(obj, "rho_dmb_mat"), r, axis=0) / Mtot}
    if hasattr(obj, "fgas_mat"):
        out["gas_on_grid"] = np.trapezoid(pref * _a(obj, "rho_gas_mat"), r, axis=0) / (_a(obj, "fgas_mat") * Mtot)
    return out


# --------------------------------------------------------------------------------------
# zeta (adiabatic relaxation) solve
# --------------------------------------------------------------------------------------
ZETA_BRACKET = (0.5, 1.5)          # trial range used by Profiles.get_zeta


def _zeta_equation(obj, jr, jz, jM, zeta):
    """(rf/ri - 1) - a_zeta [(Mi/Mf)^n_zeta - 1], exactly as in Profiles.get_zeta."""
    ri = obj.r_array[jr]
    Mi = obj.get_Mnfw(jr, jz, jM)
    rf = zeta * ri
    Mf = (obj.fclm_mat[jz, jM] * Mi + obj.get_Mcga(0, jz, jM, r_array_here=jnp.array([rf]))
          + obj.get_Mgas(0, jz, jM, r_array_here=jnp.array([rf])))
    return (rf / ri - 1) - obj.a_zeta * ((Mi / Mf) ** obj.n_zeta - 1)


def zeta_report(obj, h=1e-4):
    """Accuracy of the stored zeta_mat as a root of the relaxation equation.

    zeta_error ~ F(zeta) / F'(zeta) (one Newton step) estimates |zeta_code - zeta_root|.
    at_bracket_edge flags roots that hit the trial range, where get_zeta's linear
    interpolation silently clamps."""
    if not getattr(obj, "backreaction", True):
        return None
    zmat = obj.zeta_mat

    def per_node(jr, jz, jM):
        z0 = zmat[jr, jz, jM]
        F = _zeta_equation(obj, jr, jz, jM, z0)
        dF = (_zeta_equation(obj, jr, jz, jM, z0 + h) - _zeta_equation(obj, jr, jz, jM, z0 - h)) / (2 * h)
        return jnp.squeeze(F / dF)

    err = np.asarray(get_vmapped_func(per_node, 3)(jnp.arange(obj.nr), jnp.arange(obj.nz), jnp.arange(obj.nM)).T)
    zeta = np.asarray(zmat)
    edge = (np.abs(zeta - ZETA_BRACKET[0]) < 1e-6) | (np.abs(zeta - ZETA_BRACKET[1]) < 1e-6)
    return {"zeta_error_abs_max": float(np.max(np.abs(err))), "n_at_bracket_edge": int(edge.sum()),
            "zeta_range": (float(zeta.min()), float(zeta.max()))}


# --------------------------------------------------------------------------------------
# Fourier profiles
# --------------------------------------------------------------------------------------
def u_k0_report(obj):
    """Raw (unclipped) u(k) at the lowest FFTLog k for each mass-normalised profile, and the
    shell transform at exactly k = 0. All should be 1 (within the FFTLog low-k accuracy)."""
    out = {}
    for name in ("uk_dmb_tointp", "uk_nfw_tointp", "uk_clm_tointp", "uk_ne_tointp"):
        if hasattr(obj, name) and np.asarray(getattr(obj, name)).ndim == 3:
            out[name] = _stats(np.asarray(getattr(obj, name))[0])
    if hasattr(obj, "get_uk_clm_shell") and hasattr(obj, "Mclm_mat"):
        out["shell_u_at_k0_max_abs_err"] = float(np.max(np.abs(np.asarray(obj.get_uk_clm_shell(jnp.array([0.0]))) - 1.0)))
    return out


def finite_report(obj, names):
    """Names of arrays that contain NaN or inf."""
    bad = []
    for n in names:
        if hasattr(obj, n):
            x = np.asarray(getattr(obj, n))
            if not np.all(np.isfinite(x)):
                bad.append(n)
    return bad


# --------------------------------------------------------------------------------------
# Large-scale ON/OFF ratios
# --------------------------------------------------------------------------------------
def max_lowk_deviation(on, off, name, kmax=1e-2):
    """max over z and k <= kmax of |P_on / P_off - 1| for a (nk, nz) array."""
    k = _a(on, "kPk_array")
    sel = k <= kmax
    return float(np.max(np.abs(_a(on, name)[sel] / _a(off, name)[sel] - 1.0)))


def lowest_ell_deviation(on, off, name):
    """|C_on / C_off - 1| at the lowest ell, maximised over bin pairs."""
    a, b = _a(on, name), _a(off, name)
    return float(np.max(np.abs(a[0] / b[0] - 1.0)))


# --------------------------------------------------------------------------------------
# Exact CLM density (reference for the derivative used in get_rho_clm)
# --------------------------------------------------------------------------------------
@partial(jax.jit, static_argnums=(0,))
def exact_rho_clm(self, jz, jM, r_array_here=None):
    """Analytic d/dr of the model's own M_clm(<r) = f_clm M_nfw(r / zeta(r)), with zeta
    interpolated linearly in ln r as in Profiles.get_Mclm:
        rho_clm = f_clm rho_nfw(r/zeta) / zeta^3 * (1 - dln zeta / dln r).
    Same signature as Profiles.get_rho_clm, so it can be swapped in for a test build."""
    r = self.r_array if r_array_here is None else r_array_here
    lr_g, z_g = jnp.log(self.r_array), self.zeta_mat[:, jz, jM]
    lr = jnp.log(r)
    zeta = jnp.interp(lr, lr_g, z_g)
    dzeta_dlnr = jax.vmap(jax.grad(lambda x: jnp.interp(x, lr_g, z_g)))(lr)
    rho_nfw = jax.vmap(self.get_rho_nfw_normed, (0, None, None, None))(jnp.arange(r.shape[0]), jz, jM, r / zeta)
    return self.fclm_mat[jz, jM] * rho_nfw / zeta ** 3 * (1.0 - dzeta_dlnr / zeta)


# --------------------------------------------------------------------------------------
# Halofit non-linear scale: grid-free reference for its cosmology derivative
# --------------------------------------------------------------------------------------
def knl_reference(cosmo_fn, theta, a, n_simpson=4096):
    """k_nl and d k_nl / d theta from sigma^2(R) = 1 solved by bisection, differentiated with
    the implicit-function theorem. cosmo_fn(theta) -> jax_cosmo.Cosmology. Same integrand and
    transfer function as godmax.helpers.jax_cosmo_power.halofit_parameters, but no node grid."""
    import jax_cosmo.background as bkgrd
    import jax_cosmo.transfer as tklib
    from jax_cosmo.scipy.integrate import simps
    from godmax.helpers.jax_cosmo_power import linear_matter_power

    def sigma2(lnR, th):
        c = cosmo_fn(th)

        def integrand(lnk):
            k = jnp.exp(lnk)
            pk = linear_matter_power(c, k, transfer_fn=tklib.Eisenstein_Hu)
            g = bkgrd.growth_factor(c, jnp.atleast_1d(a))[0]
            return pk * k ** 3 * jnp.exp(-(k * jnp.exp(lnR)) ** 2) / (2 * jnp.pi ** 2) * g ** 2
        return simps(integrand, jnp.log(1e-4), jnp.log(1e4), n_simpson)

    s2 = jax.jit(sigma2)
    lo, hi = np.log(1e-4), np.log(1e1)
    f_lo = float(s2(lo, theta)) - 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = float(s2(mid, theta)) - 1.0
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    lnR = 0.5 * (lo + hi)
    dlnR = -np.asarray(jax.grad(sigma2, argnums=1)(lnR, theta)) / float(jax.grad(sigma2, argnums=0)(lnR, theta))
    knl = float(np.exp(-lnR))
    return knl, -knl * dlnR
