"""Projected-profile operator used by the three-probe paste validation.

The map painter evaluates a physical line-of-sight projection and only paints
pixels whose physical transverse distance is below ``R_paint``.  Consequently
its continuum Fourier kernel is a *cylindrically windowed projection*, not in
general the Fourier transform of a profile truncated at spherical radius
``R_paint``.  This module keeps those two operators explicit.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import interpax


def painter_rp_nodes(radius_comoving_hmpc: np.ndarray) -> np.ndarray:
    """Reproduce ``setup_sim_map._setup_common`` projected-radius nodes."""

    radius = np.asarray(radius_comoving_hmpc, dtype=np.float64)
    if radius.ndim != 1 or radius.size < 6 or np.any(np.diff(radius) <= 0.0):
        raise ValueError("radius must be a strictly increasing 1D grid with >=6 points")
    return np.logspace(np.log10(radius[2]), np.log10(radius[-2]), radius.size - 3)


def project_physical_profile_cosh(
    radius_comoving_hmpc: np.ndarray,
    profile_physical: np.ndarray,
    redshift: float,
    rp_physical_hmpc: np.ndarray,
    *,
    n_los: int,
) -> np.ndarray:
    """Match the painter's unit-consistent ``physical_table_cosh`` projector.

    The result is ``2 integral_0^L dl profile(sqrt(R^2+l^2))``.  Positive
    profiles are required because the production painter interpolates their
    logarithm.
    """

    radius = np.asarray(radius_comoving_hmpc, dtype=np.float64)
    profile = np.asarray(profile_physical, dtype=np.float64)
    rp = np.atleast_1d(np.asarray(rp_physical_hmpc, dtype=np.float64))
    if radius.ndim != 1 or profile.shape != radius.shape:
        raise ValueError("radius and profile must be finite 1D arrays of equal shape")
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(profile)):
        raise ValueError("projection inputs must be finite")
    if np.any(radius <= 0.0) or np.any(np.diff(radius) <= 0.0) or np.any(profile <= 0.0):
        raise ValueError("log-interpolated radius/profile inputs must be positive and ordered")
    if np.any(rp <= 0.0) or not np.isfinite(redshift) or redshift < 0.0 or n_los < 2:
        raise ValueError("invalid projected radius, redshift, or LOS resolution")

    r_phys = radius / (1.0 + float(redshift))
    nodes, weights = np.polynomial.legendre.leggauss(int(n_los))
    out = np.zeros_like(rp)
    supported = rp < r_phys[-1]
    if not np.any(supported):
        return out
    rp_s = rp[supported]
    los_max = np.sqrt(np.maximum(r_phys[-1] ** 2 - rp_s**2, 0.0))
    t_max = np.arcsinh(los_max / rp_s)
    t = 0.5 * (nodes[None, :] + 1.0) * t_max[:, None]
    r_eval = rp_s[:, None] * np.cosh(t)
    quantity = np.exp(np.interp(np.log(r_eval), np.log(r_phys), np.log(profile)))
    out[supported] = t_max * np.sum(weights[None, :] * r_eval * quantity, axis=1)
    return out


def project_physical_profile_legacy(
    radius_comoving_hmpc: np.ndarray,
    profile_physical: np.ndarray,
    redshift: float,
    rp_physical_hmpc: np.ndarray,
    *,
    n_los: int,
) -> np.ndarray:
    """Reproduce the painter's historical mixed-unit log-radius projector."""

    radius = np.asarray(radius_comoving_hmpc, dtype=np.float64)
    profile = np.asarray(profile_physical, dtype=np.float64)
    rp = np.atleast_1d(np.asarray(rp_physical_hmpc, dtype=np.float64))
    if profile.shape != radius.shape or np.any(profile <= 0.0) or n_los < 2:
        raise ValueError("legacy projection requires an aligned positive profile")
    out = np.empty_like(rp)
    physical_radius = radius / (1.0 + float(redshift))
    for index, transverse in enumerate(rp):
        radius_max = min(float(radius[-1]), 100.0 * float(transverse))
        sample = np.exp(np.linspace(np.log(1.01 * transverse), np.log(radius_max), int(n_los)))
        quantity = np.exp(np.interp(np.log(sample), np.log(physical_radius), np.log(profile)))
        integrand = sample * quantity / np.sqrt(sample**2 - transverse**2)
        out[index] = 2.0 * np.trapz(integrand * sample, x=np.log(sample))
    return out


def painter_log_interpolate(
    rp_nodes_physical_hmpc: np.ndarray,
    projected_nodes: np.ndarray,
    rp_eval_physical_hmpc: np.ndarray,
    *,
    outside_log_value: float = -20.0,
) -> np.ndarray:
    """Reproduce the painter's cubic log-profile radial interpolation."""

    nodes = np.asarray(rp_nodes_physical_hmpc, dtype=np.float64)
    values = np.asarray(projected_nodes, dtype=np.float64)
    rp = np.asarray(rp_eval_physical_hmpc, dtype=np.float64)
    if nodes.ndim != 1 or values.shape != nodes.shape or np.any(values < 0.0):
        raise ValueError("projected nodes must be a non-negative 1D table")
    safe_log = np.full(values.shape, float(outside_log_value), dtype=np.float64)
    positive = values > 0.0
    safe_log[positive] = np.log(values[positive])
    interpolator = interpax.Interpolator1D(
        jnp.asarray(np.log(nodes), dtype=jnp.float64),
        jnp.asarray(safe_log, dtype=jnp.float64),
        method="cubic",
        extrap=[float(outside_log_value), float(outside_log_value)],
    )
    return np.asarray(jnp.exp(interpolator(jnp.asarray(np.log(rp), dtype=jnp.float64))))


def projected_painter_transform(
    k_comoving_hmpc: np.ndarray,
    rp_physical_hmpc: np.ndarray,
    projected_profile: np.ndarray,
    redshift: float,
    paint_radius_physical_hmpc: float,
    *,
    physical_to_theory_volume_factor: float,
) -> np.ndarray:
    """Hankel transform the painted, projected profile through its aperture."""

    k = np.asarray(k_comoving_hmpc, dtype=np.float64)
    rp = np.asarray(rp_physical_hmpc, dtype=np.float64)
    sigma = np.asarray(projected_profile, dtype=np.float64)
    if rp.ndim != 1 or sigma.shape != rp.shape or np.any(np.diff(rp) <= 0.0):
        raise ValueError("rp/projected profile must be aligned on an increasing 1D grid")
    if np.any(k < 0.0) or np.any(~np.isfinite(sigma)):
        raise ValueError("k must be non-negative and the projected profile finite")
    keep = rp <= float(paint_radius_physical_hmpc)
    if np.count_nonzero(keep) < 2:
        raise ValueError("paint aperture contains fewer than two quadrature points")
    q_phys = k[:, None] * (1.0 + float(redshift))
    integrand = 2.0 * np.pi * rp[None, keep] * sigma[None, keep] * j0(q_phys * rp[None, keep])
    return float(physical_to_theory_volume_factor) * np.trapz(integrand, x=rp[keep], axis=1)


def spherical_support_transform(
    k_comoving_hmpc: np.ndarray,
    radius_comoving_hmpc: np.ndarray,
    profile_theory: np.ndarray,
    paint_radius_comoving_hmpc: float,
) -> np.ndarray:
    """Direct signed 3D transform of the spherical-support candidate."""

    k = np.asarray(k_comoving_hmpc, dtype=np.float64)
    radius = np.asarray(radius_comoving_hmpc, dtype=np.float64)
    profile = np.asarray(profile_theory, dtype=np.float64)
    keep = radius <= float(paint_radius_comoving_hmpc)
    if np.count_nonzero(keep) < 2:
        raise ValueError("spherical aperture contains fewer than two radial nodes")
    kr = k[:, None] * radius[None, keep]
    sinc = np.sinc(kr / np.pi)
    integrand = 4.0 * np.pi * radius[None, keep] ** 2 * profile[None, keep] * sinc
    return np.trapz(integrand, x=radius[keep], axis=1)


def symmetric_fractional_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Bounded signed difference, with exact zero for two zero entries."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    scale = np.abs(left) + np.abs(right)
    return np.divide(2.0 * (left - right), scale, out=np.zeros_like(scale), where=scale > 0.0)
