"""Pure-JAX primitives for the exact three-probe map-theory forward operator.

These functions deliberately contain no truth point, likelihood, or noise draw.
They reproduce the painter's physical-table-cosh projection, its radial Gaussian
smoothing, and the frozen estimator/window projection. The real-space smoothing
is embedded in the resulting tables, so the exact path applies no second Bell. The GODMAX profile bridge
is a separate gate: it must supply x64 theta-dependent profile tables.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import pathlib
import sys
from typing import Any, Callable

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import jax.scipy.integrate as jsi
import jax.scipy.special as jss
import h5py
import interpax
import numpy as np
import yaml


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "notebooks" / "xDESI", THIS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


PARAMETER_NAMES = ("theta_ej_0", "alpha_nt", "mu_beta", "theta_co_0", "nu_theta_ej_M")
CONFIG_PATH = THIS_DIR / "three_probe_mock_experiment.yaml"
MAP_PATH = REPO_ROOT / (
    "data/SBI_validate/three_probe_mock/maps/c0000_z0p3_0p5_mmin5e11_cosh32_fast/"
    "abacus_pasted_maps_c0000_z0p3_0p5_mmin5e11_nside1024.h5"
)


@dataclass(frozen=True)
class ThreeProbeForwardModel:
    vector_fn: Callable[[jnp.ndarray], jnp.ndarray]
    dense_fn: Callable[[jnp.ndarray], dict[str, jnp.ndarray]]
    metadata: dict[str, Any]


def j0_safe(argument: jnp.ndarray) -> jnp.ndarray:
    """Differentiable J0 stable at zero and at large Hankel arguments.

    The installed ``jax.scipy.special.bessel_jn`` becomes inaccurate for the
    large arguments reached by the painter aperture.  These standard rational
    and asymptotic approximants avoid that failure and remain traceable.
    """

    x = jnp.asarray(argument, dtype=jnp.float64)
    ax = jnp.abs(x)
    def polevl(value: jnp.ndarray, coefficients: tuple[float, ...]) -> jnp.ndarray:
        result = jnp.asarray(coefficients[0], dtype=value.dtype)
        for coefficient in coefficients[1:]:
            result = result * value + coefficient
        return result

    def p1evl(value: jnp.ndarray, coefficients: tuple[float, ...]) -> jnp.ndarray:
        result = value + coefficients[0]
        for coefficient in coefficients[1:]:
            result = result * value + coefficient
        return result

    z = ax * ax
    rp = (-4.79443220978201773821e9, 1.95617491946556577543e12,
          -2.49248344360967716204e14, 9.70862251047306323952e15)
    rq = (4.99563147152651017219e2, 1.73785401676374683123e5,
          4.84409658339962045305e7, 1.11855537045356834862e10,
          2.11277520115489217587e12, 3.10518229857422583814e14,
          3.18121955943204943306e16, 1.71086294081043136091e18)
    small_rational = (
        (z - 5.78318596294678452118) * (z - 30.4712623436620863991)
        * polevl(z, rp) / p1evl(z, rq)
    )
    small = jnp.where(ax < 1.0e-5, 1.0 - z / 4.0, small_rational)

    safe_ax = jnp.maximum(ax, 5.0)
    w = 5.0 / safe_ax
    q = 25.0 / (safe_ax * safe_ax)
    pp = (7.96936729297347051624e-4, 8.28352392107440799803e-2,
          1.23953371646414299388, 5.44725003058768775090,
          8.74716500199817011941, 5.30324038235394892183,
          0.999999999999999997821)
    pq = (9.24408810558863637013e-4, 8.56288474354474431428e-2,
          1.25352743901058953537, 5.47097740330417105182,
          8.76190883237069594232, 5.30605288235394617618,
          1.00000000000000000218)
    qp = (-1.13663838898469149931e-2, -1.28252718670509318512,
          -19.5539544257735972385, -93.2060152123768261369,
          -177.681167980488050595, -147.077505154951170175,
          -51.4105326766599330220, -6.05014350600728481186)
    qq = (64.3178256118178023184, 856.430025976980587198,
          3882.40183605401609683, 7240.46774195652478189,
          5930.72701187316984827, 2062.09331660327847417,
          242.005740240291393179)
    p = polevl(q, pp) / polevl(q, pq)
    q_value = polevl(q, qp) / p1evl(q, qq)
    phase = safe_ax - jnp.pi / 4.0
    large = (p * jnp.cos(phase) - w * q_value * jnp.sin(phase))
    large *= jnp.sqrt(2.0 / jnp.pi) / jnp.sqrt(safe_ax)
    return jnp.where(ax <= 5.0, small, large)


_F32_DROP = np.uint64(29)              # float64 keeps 52 mantissa bits, float32 keeps 23
_F32_ONE = jnp.uint64(1)
_F32_LOW = (_F32_ONE << _F32_DROP) - _F32_ONE
_F32_HALF = _F32_ONE << (_F32_DROP - np.uint64(1))
_F32_MIN_NORMAL = 2.0 ** -126
_F32_MAX = float(np.finfo(np.float32).max)
_F32_MASK = jnp.uint64(0xFFFFFFFFE0000000)   # clears the low 29 mantissa bits
_F32_DENORMAL_STEP = 2.0 ** -149              # spacing of the float32 denormal grid


@jax.custom_jvp
def quantize_to_float32(value: jnp.ndarray) -> jnp.ndarray:
    """Round to float32 precision without ever emitting a float32 convert.

    The immutable pasted map stored its painter tables as float32, so this
    narrowing is part of the physical operator, not a rounding detail.

    A bare ``x.astype(float32).astype(float64)`` does not survive compilation.
    Backend bisection job 6928490 showed XLA GPU eliminating the convert pair
    while XLA CPU performed it: on GPU the round-trip returned 1.0e-46
    unchanged, a value float32 cannot represent at any exponent, and 236 of 341
    probe values that CPU flushed to zero came back non-zero.  Job 6928544
    established that ``jax.lax.optimization_barrier`` between the converts does
    NOT prevent this -- the outputs were bit-identical to the unguarded version
    on both backends -- so the elision is not something a scheduling barrier
    reaches.  ``--xla_allow_excess_precision`` defaults to true, which is the
    likely licence for it.

    The consequence was severe rather than cosmetic: downstream,
    ``painter_log_interpolate_jax`` branches on ``values > 0.0`` and substitutes
    -20.0 in log space for a zero, so the disagreement was amplified through
    ``jnp.exp`` to factors up to 258, reaching the 42-vector as a max relative
    difference of 1.5e+01 and a whitened chi-square gap of 334 units.

    This implementation therefore rounds the mantissa by integer bit
    manipulation in float64 throughout, reproducing round-to-nearest-ties-to-even
    followed by float32 flush-to-zero below ``2**-126`` and overflow to infinity.
    It is bit-identical to a genuine ``float32`` cast across 77,011 test values
    spanning 1e-46 to 1e30, both signs, the denormal window, the min-normal
    boundary, exact ties, +-0, +-inf and NaN.  CPU is the side to match: it is
    the one that agrees with the non-JAX host GODMAX implementation to a median
    1.2e-05 in the frozen non-regression check.
    """

    x = jnp.asarray(value, dtype=jnp.float64)
    bits = jax.lax.bitcast_convert_type(x, jnp.uint64)
    lower = bits & _F32_LOW
    truncated = bits & _F32_MASK
    tie_to_even = ((truncated >> _F32_DROP) & _F32_ONE) == _F32_ONE
    round_up = (lower > _F32_HALF) | ((lower == _F32_HALF) & tie_to_even)
    rounded = jax.lax.bitcast_convert_type(
        truncated + jnp.where(round_up, _F32_ONE << _F32_DROP, jnp.uint64(0)), jnp.float64)
    # Below the smallest normal, float32 is a fixed-point grid of multiples of
    # 2**-149, and IEEE -- hence numpy -- ROUNDS onto that grid rather than
    # flushing to zero.  An earlier revision flushed, and disagreed with a
    # genuine numpy float32 cast in 11,323 of 102,012 probe values.  numpy is the
    # correct target: the painter wrote its float32 tables eagerly with numpy,
    # and the host validator casts eagerly too.
    denormal = jnp.round(x / _F32_DENORMAL_STEP) * _F32_DENORMAL_STEP
    rounded = jnp.where(jnp.abs(x) < _F32_MIN_NORMAL, denormal, rounded)
    rounded = jnp.where(jnp.abs(rounded) > _F32_MAX, jnp.sign(x) * jnp.inf, rounded)
    rounded = jnp.where(jnp.isnan(x), x, rounded)
    return jnp.where(jnp.isinf(x), x, rounded)


@quantize_to_float32.defjvp
def _quantize_to_float32_jvp(primals, tangents):
    """Exact quantised value, identity derivative.

    ``bitcast_convert_type`` has no meaningful JVP, so the primal alone gives
    EXACTLY ZERO gradient -- the failure INV-JAX-GRAD-FINITE-01 exists for, and
    what jobs 6928581 and 6928582 hit: every column of dmu/dtheta vanished, the
    score Gram matrix went singular, and L-BFGS stalled wherever it started.
    The original bare cast had a derivative of exactly 1.0, so the identity
    tangent restores the derivative semantics the forward model was validated
    with while keeping the corrected value.

    A ``stop_gradient`` straight-through, ``x + stop_gradient(q - x)``, would
    also give the right value and derivative, but it is an *algebraic identity*,
    and XLA has already demonstrated on this code that it will exploit those --
    it is what eliminated the original convert pair.  ``custom_jvp`` leaves no
    identity to fold.
    """

    (x,), (tangent,) = primals, tangents
    return quantize_to_float32(x), tangent

def assert_float32_quantization_effective() -> dict[str, object]:
    """Fail closed if the float32 narrowing is inactive, inexact, or gradient-dead.

    Called once when the forward model is built.  Cheap, and it converts three
    separate silent failures into immediate errors:

    * **inactive** -- XLA eliminating the narrowing, which is what made the GPU
      compute a different operator from the CPU (jobs 6928490, 6928544);
    * **inexact** -- disagreeing with a genuine numpy float32 cast, which an
      earlier flush-to-zero revision did in 11,323 of 102,012 probe values;
    * **gradient-dead** -- the right value with a zero derivative, which took out
      jobs 6928581 and 6928582 via a singular score Gram matrix.

    numpy's eager cast is the reference because the painter wrote its float32
    tables with numpy and the host validator casts eagerly too.
    """

    probe_np = np.array([1.0e-46, 1.0e-40, 1.0e-30, float(np.pi), 3.5e38, 0.0,
                         2.0 ** -149, 2.0 ** -126], dtype=np.float64)
    with np.errstate(over="ignore"):          # 3.5e38 overflows to inf by design
        expected = np.float64(np.float32(probe_np))
    got = np.asarray(jax.jit(quantize_to_float32)(jnp.asarray(probe_np)), dtype=np.float64)
    disagreement = ~((expected == got) | (np.isnan(expected) & np.isnan(got)))
    if np.any(disagreement):
        raise RuntimeError(
            f"float32 narrowing disagrees with a genuine numpy float32 cast at "
            f"{int(disagreement.sum())} of {probe_np.size} probes: "
            f"inputs {probe_np[disagreement].tolist()}, numpy "
            f"{expected[disagreement].tolist()}, got {got[disagreement].tolist()}.")
    # Quantisation must actually change something, or it has been optimised away.
    changed = int(np.sum(got != probe_np))
    if changed == 0:
        raise RuntimeError(
            "float32 narrowing is not taking effect: every probe came back "
            "unchanged, so the operator is not the painter-equivalent one. "
            "See jobs 6928490 and 6928544.")
    # A quantiser with the right value and no derivative is worse than none: it
    # reads as "this parameter is unconstrained by data".
    slopes = np.asarray(jax.vmap(jax.grad(quantize_to_float32))(
        jnp.asarray([0.5, 3.7, 1.0e-10, 1.0e10], dtype=jnp.float64)), dtype=np.float64)
    if not np.all(slopes == 1.0):
        raise RuntimeError(
            f"float32 quantiser has lost its gradient: d/dx = {slopes.tolist()}, "
            f"expected all 1.0. The original bare cast had gradient 1.0 and the "
            f"forward model was validated with that derivative. See jobs 6928581 "
            f"and 6928582.")
    return dict(probes=probe_np.tolist(), quantised=got.tolist(),
                n_changed_by_quantisation=changed, gradient=slopes.tolist())


def project_physical_profile_cosh_jax(
    radius_comoving_hmpc: jnp.ndarray,
    profile_physical_r: jnp.ndarray,
    redshift: jnp.ndarray,
    rp_physical_hmpc: jnp.ndarray,
    gauss_legendre_nodes: jnp.ndarray,
    gauss_legendre_weights: jnp.ndarray,
) -> jnp.ndarray:
    """Differentiable counterpart of the painter's physical-table-cosh integral."""

    radius = jnp.asarray(radius_comoving_hmpc, dtype=jnp.float64)
    profile = jnp.asarray(profile_physical_r, dtype=jnp.float64)
    rp = jnp.asarray(rp_physical_hmpc, dtype=jnp.float64)
    z = jnp.asarray(redshift, dtype=jnp.float64)
    nodes = jnp.asarray(gauss_legendre_nodes, dtype=jnp.float64)
    weights = jnp.asarray(gauss_legendre_weights, dtype=jnp.float64)
    physical_radius = radius / (1.0 + z)
    table_r_max = physical_radius[-1]
    has_support = rp < table_r_max
    los_max = jnp.sqrt(jnp.maximum(table_r_max**2 - rp**2, 0.0))
    t_max = jnp.arcsinh(los_max / rp)
    t = 0.5 * (nodes[None, :] + 1.0) * t_max[:, None]
    r_eval = rp[:, None] * jnp.cosh(t)
    values = jnp.exp(
        jnp.interp(jnp.log(r_eval), jnp.log(physical_radius), jnp.log(jnp.maximum(profile, jnp.finfo(profile.dtype).tiny)))
    )
    projected = t_max * jnp.sum(weights[None, :] * r_eval * values, axis=1)
    return jnp.where(has_support, projected, jnp.zeros_like(projected))


def smooth_radial_gaussian_jax(
    rp_physical_hmpc: jnp.ndarray,
    angular_diameter_distance_hmpc: jnp.ndarray,
    projected_profile: jnp.ndarray,
    sigma_rad: jnp.ndarray,
    gauss_legendre_nodes: jnp.ndarray,
    gauss_legendre_weights: jnp.ndarray,
    radial_sigma_cutoff: float = 10.0,
) -> jnp.ndarray:
    """Reproduce the painter's flux-renormalized circular Gaussian smoothing."""

    rp = jnp.asarray(rp_physical_hmpc, dtype=jnp.float64)
    source = jnp.asarray(projected_profile, dtype=jnp.float64)
    theta = rp / jnp.asarray(angular_diameter_distance_hmpc, dtype=jnp.float64)
    sigma = jnp.asarray(sigma_rad, dtype=jnp.float64)
    nodes = jnp.asarray(gauss_legendre_nodes, dtype=jnp.float64)
    weights = jnp.asarray(gauss_legendre_weights, dtype=jnp.float64)
    lower = jnp.maximum(0.0, theta - radial_sigma_cutoff * sigma)
    upper = jnp.minimum(theta[-1], theta + radial_sigma_cutoff * sigma)
    midpoint, half_width = 0.5 * (lower + upper), 0.5 * (upper - lower)
    theta_prime = midpoint[:, None] + half_width[:, None] * nodes[None, :]
    floor = jnp.finfo(source.dtype).tiny
    source_at_prime = jnp.exp(
        jnp.interp(jnp.log(jnp.maximum(theta_prime, theta[0])), jnp.log(theta), jnp.log(jnp.maximum(source, floor)))
    )
    argument = theta[:, None] * theta_prime / sigma**2
    radial_kernel = (
        theta_prime / sigma**2
        * jnp.exp(-0.5 * ((theta[:, None] - theta_prime) / sigma) ** 2)
        * jss.i0e(argument)
    )
    smoothed = half_width * jnp.sum(weights[None, :] * radial_kernel * source_at_prime, axis=-1)
    input_flux = jsi.trapezoid(theta * source, theta)
    output_flux = jsi.trapezoid(theta * smoothed, theta)
    return jnp.maximum(smoothed * input_flux / jnp.maximum(output_flux, floor), floor)


def painter_log_interpolate_jax(
    rp_nodes_physical_hmpc: jnp.ndarray,
    projected_nodes: jnp.ndarray,
    rp_eval_physical_hmpc: jnp.ndarray,
) -> jnp.ndarray:
    """JAX painter-equivalent cubic interpolation of a positive projected table."""

    nodes = jnp.asarray(rp_nodes_physical_hmpc, dtype=jnp.float64)
    values = jnp.asarray(projected_nodes, dtype=jnp.float64)
    rp_eval = jnp.asarray(rp_eval_physical_hmpc, dtype=jnp.float64)
    safe_log = jnp.where(values > 0.0, jnp.log(jnp.maximum(values, jnp.finfo(values.dtype).tiny)), -20.0)
    interpolator = interpax.Interpolator1D(
        jnp.log(nodes), safe_log, method="cubic", extrap=[-20.0, -20.0]
    )
    return jnp.exp(interpolator(jnp.log(rp_eval)))


def cylindrical_hankel_transform_jax(
    k_comoving_hmpc: jnp.ndarray,
    rp_physical_hmpc: jnp.ndarray,
    projected_profile: jnp.ndarray,
    redshift: jnp.ndarray,
    paint_radius_physical_hmpc: jnp.ndarray,
    physical_to_theory_volume_factor: jnp.ndarray,
) -> jnp.ndarray:
    """Transform a projected painter profile through its transverse aperture."""

    k = jnp.asarray(k_comoving_hmpc, dtype=jnp.float64)
    rp = jnp.asarray(rp_physical_hmpc, dtype=jnp.float64)
    sigma = jnp.asarray(projected_profile, dtype=jnp.float64)
    aperture = jnp.asarray(paint_radius_physical_hmpc, dtype=jnp.float64)
    integrand = 2.0 * jnp.pi * rp[None, :] * sigma[None, :] * j0_safe(
        k[:, None] * (1.0 + jnp.asarray(redshift, dtype=jnp.float64)) * rp[None, :]
    )
    # This is the host operator's discrete ``rp <= Rpaint`` support: retain
    # whole trapezoids only when both endpoints lie in the aperture.  Masking
    # the integrand itself would retain a spurious final half-trapezoid.
    interval = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * (rp[1:] - rp[:-1])[None, :]
    keep_interval = rp[1:] <= aperture
    integral = jnp.sum(interval * keep_interval[None, :], axis=-1)
    return jnp.asarray(physical_to_theory_volume_factor, dtype=jnp.float64) * integral


def apply_frozen_estimator_jax(
    dense_cls: dict[str, jnp.ndarray],
    pixel_window_g: jnp.ndarray,
    window: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the galaxy pixel window and NaMaster window to embedded-smoothed theory."""

    transfer_g = jnp.asarray(pixel_window_g, dtype=jnp.float64)
    saved_window = jnp.asarray(window, dtype=jnp.float64)
    bands = []
    for spectrum in ("gy", "gkappa", "gtau"):
        cls = jnp.asarray(dense_cls[spectrum], dtype=jnp.float64)
        bands.append(saved_window @ (transfer_g * cls))
    return jnp.concatenate(bands)


def _project_smooth_transform_field(
    radius: jnp.ndarray,
    rp_nodes: jnp.ndarray,
    redshift: jnp.ndarray,
    angular_diameter_distance: jnp.ndarray,
    r200c_zm: jnp.ndarray,
    profile_rzm: jnp.ndarray,
    k: jnp.ndarray,
    sigma_rad: jnp.ndarray,
    projection_nodes: jnp.ndarray,
    projection_weights: jnp.ndarray,
    smoothing_nodes: jnp.ndarray,
    smoothing_weights: jnp.ndarray,
    factor_z: jnp.ndarray,
    *,
    dense_radius_nodes: int,
) -> jnp.ndarray:
    """Project, quantize, smooth, interpolate, and transform one physical field."""

    profile_zmr = jnp.transpose(profile_rzm, (1, 2, 0))

    def one_z(profiles_mr: jnp.ndarray, z: jnp.ndarray, da: jnp.ndarray,
              r200_m: jnp.ndarray, factor: jnp.ndarray) -> jnp.ndarray:
        def one_mass(profile_r: jnp.ndarray, r200: jnp.ndarray) -> jnp.ndarray:
            projected = project_physical_profile_cosh_jax(
                radius, profile_r, z, rp_nodes, projection_nodes, projection_weights
            )
            # The immutable pasted map stored its painter tables as float32.
            # Reproduce that declared data-product boundary, then immediately
            # return to x64 for smoothing, integration, and gradients.
            projected = quantize_to_float32(projected)
            smoothed = quantize_to_float32(smooth_radial_gaussian_jax(
                rp_nodes, da, projected, sigma_rad, smoothing_nodes, smoothing_weights))
            aperture = 8.0 * r200 / (1.0 + z)
            rp_dense = jnp.geomspace(
                jnp.maximum(aperture * 1.0e-7, jnp.finfo(jnp.float64).tiny),
                aperture,
                int(dense_radius_nodes),
            )
            dense_profile = painter_log_interpolate_jax(rp_nodes, smoothed, rp_dense)
            return cylindrical_hankel_transform_jax(
                k, rp_dense, dense_profile, z, aperture, factor
            )

        return jax.vmap(one_mass)(profiles_mr, r200_m)

    transformed_zmk = jax.vmap(one_z)(
        profile_zmr, redshift, angular_diameter_distance, r200c_zm, factor_z
    )
    return jnp.transpose(transformed_zmk, (2, 0, 1))


def _interp_power_at_limber_k(
    power_kz: jnp.ndarray, k_grid: jnp.ndarray, k_limber_z: jnp.ndarray
) -> jnp.ndarray:
    logk = jnp.log(k_grid)
    return jax.vmap(
        lambda values, target: jnp.interp(jnp.log(target), logk, values),
        in_axes=(1, 0),
    )(power_kz, k_limber_z)


def project_resolved_power_to_cls_jax(
    ell: jnp.ndarray,
    k: jnp.ndarray,
    redshift: jnp.ndarray,
    chi: jnp.ndarray,
    powers: dict[str, jnp.ndarray],
    realized_nz: jnp.ndarray,
    cmb_efficiency: jnp.ndarray,
    tau_constant: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Pure-JAX Limber projection matching the host resolved-theory cross spectra."""

    def one_ell(ell_value: jnp.ndarray) -> jnp.ndarray:
        k_limber = (ell_value + 0.5) / chi
        pgy = _interp_power_at_limber_k(powers["Pgy_resolved"], k, k_limber)
        pge = _interp_power_at_limber_k(powers["Pge_resolved"], k, k_limber)
        pgm = _interp_power_at_limber_k(powers["Pgm_resolved"], k, k_limber)
        common = realized_nz / (chi * chi)
        gy = jsi.trapezoid(common * pgy / (1.0 + redshift), redshift)
        gkappa = jsi.trapezoid(common * pgm * cmb_efficiency, redshift)
        gtau = jsi.trapezoid(
            common * pge * tau_constant * (1.0 + redshift) ** 2, redshift
        )
        return jnp.stack((gy, gkappa, gtau))

    # Every ell is independent and the retained 2049x48 interpolation surface
    # is small on the target GPU.  ``lax.map`` serialized all 2049 Limber
    # projections inside every NUTS leapfrog step; vmap fuses them into one
    # batched kernel without changing the arithmetic for any ell.
    values = jax.vmap(one_ell)(ell)
    return {"gy": values[:, 0], "gkappa": values[:, 1], "gtau": values[:, 2]}


def make_three_probe_forward_model(
    contract_path: pathlib.Path,
    *,
    config_path: pathlib.Path = CONFIG_PATH,
    map_path: pathlib.Path = MAP_PATH,
    dense_radius_nodes: int = 256,
    profile_nr: int = 48,
    profile_nz: int = 48,
    limber_ell_nodes: int = 2049,
    jit_compile: bool = True,
) -> ThreeProbeForwardModel:
    """Build the exact truth-free theta-to-42-vector callable used by validation and HMC."""

    from astropy import constants as const
    import astropy.units as u
    from base_class import base_class
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles
    from three_probe_fast_paste import _catalog_attrs, prepare_fast_paste_godmax_config
    from three_probe_inference_contract import load_training_contract
    from three_probe_noiseless_theory import override_analysis_with_map_kernels
    from three_probe_resolved_theory import assemble_resolved_power

    contract = load_training_contract(contract_path)
    config_path = pathlib.Path(config_path).resolve()
    map_path = pathlib.Path(map_path).resolve()
    if map_path != MAP_PATH.resolve():
        raise ValueError("Forward model must use the frozen c0000 nside-1024 map")
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    catalog_path = pathlib.Path(config["resolved_theory"]["catalog_path"])
    sim, halo, analysis, other = prepare_fast_paste_godmax_config(
        config, _catalog_attrs(catalog_path), config_path=config_path
    )
    with h5py.File(map_path, "r") as handle:
        kernels = {name: np.asarray(handle[f"kernels/{name}"]) for name in handle["kernels"]}
        sigma_rad = float(handle["kernels"].attrs["profile_smoothing_sigma_rad"])
    override_analysis_with_map_kernels(analysis, kernels)
    halo["nr"] = int(profile_nr)
    halo["nz"] = int(profile_nz)
    if int(halo["nM"]) != 24 or int(halo["nk"]) != 48:
        raise ValueError("Forward model mass/k node counts must remain 24/48")
    if int(profile_nr) < 16 or int(profile_nz) < 16:
        raise ValueError("Profile radial/redshift grids require at least 16 nodes")
    if int(limber_ell_nodes) < 16 or int(limber_ell_nodes) > 2049:
        raise ValueError("Limber ell grid must contain between 16 and 2049 nodes")
    for name, value in contract.fixed_parameters.items():
        sim[name] = value
    if tuple(parameter["name"] for parameter in contract.sampled_parameters) != PARAMETER_NAMES:
        raise ValueError("Training-contract sampled parameter order mismatch")

    float32_quantization_probe = assert_float32_quantization_effective()

    projection_nodes_np, projection_weights_np = np.polynomial.legendre.leggauss(32)
    smoothing_nodes_np, smoothing_weights_np = np.polynomial.legendre.leggauss(64)
    projection_nodes = jnp.asarray(projection_nodes_np, dtype=jnp.float64)
    projection_weights = jnp.asarray(projection_weights_np, dtype=jnp.float64)
    smoothing_nodes = jnp.asarray(smoothing_nodes_np, dtype=jnp.float64)
    smoothing_weights = jnp.asarray(smoothing_weights_np, dtype=jnp.float64)
    sigma_j = jnp.asarray(sigma_rad, dtype=jnp.float64)
    ell = jnp.arange(2049, dtype=jnp.float64)
    if int(limber_ell_nodes) == 2049:
        ell_projection = ell
    else:
        ell_projection_np = np.unique(
            np.rint(np.geomspace(0.5, 2048.5, int(limber_ell_nodes)) - 0.5)
        ).astype(np.float64)
        ell_projection_np = np.unique(np.concatenate(([0.0], ell_projection_np, [2048.0])))
        ell_projection = jnp.asarray(ell_projection_np, dtype=jnp.float64)
    pixel_window = jnp.asarray(contract.pixel_window_g, dtype=jnp.float64)
    window = jnp.asarray(contract.window, dtype=jnp.float64)
    realized_z = jnp.asarray(kernels["realized_hod_galaxy_redshift"], dtype=jnp.float64)
    realized_nz_saved = jnp.asarray(kernels["realized_hod_galaxy_nz"], dtype=jnp.float64)
    halo_z_saved = jnp.asarray(kernels["halo_redshift"], dtype=jnp.float64)
    wkappa_saved = jnp.asarray(kernels["cmb_lensing_efficiency_Wkappa_hmpc"], dtype=jnp.float64)
    one_mpc = ((10**6) * u.pc.to(u.m)) * u.m
    h = float(sim["cosmo"]["H0"]) / 100.0
    y_constant = float(((const.sigma_T / (const.m_e * const.c**2) * one_mpc).to(u.cm**3 / u.keV)).value / h)
    tau_constant = jnp.asarray(float(((const.sigma_T * one_mpc).to(u.cm**3)).value / h), dtype=jnp.float64)

    def dense_fn(theta: jnp.ndarray) -> dict[str, jnp.ndarray]:
        theta = jnp.asarray(theta, dtype=jnp.float64)
        sim_here = copy.deepcopy(sim)
        other_here = copy.deepcopy(other)
        for index, name in enumerate(PARAMETER_NAMES):
            sim_here[name] = theta[index]
        base = base_class(sim_here, halo, analysis, other_here)
        profiles = Profiles(sim_here, halo, analysis, other_here, base_class_obj=base)
        pkz = get_Pkz(sim_here, halo, analysis, other_here, Profiles_obj=profiles)
        radius = jnp.asarray(profiles.r_array, dtype=jnp.float64)
        rp_nodes = jnp.geomspace(radius[2], radius[-2], radius.shape[0] - 3)
        z = jnp.asarray(pkz.z_array, dtype=jnp.float64)
        a = 1.0 / (1.0 + z)
        da = jnp.asarray(pkz.chi_array, dtype=jnp.float64) * a
        k = jnp.asarray(pkz.kPk_array, dtype=jnp.float64)
        common = dict(
            radius=radius, rp_nodes=rp_nodes, redshift=z,
            angular_diameter_distance=da, r200c_zm=jnp.asarray(profiles.r200c_mat),
            k=k, sigma_rad=sigma_j, projection_nodes=projection_nodes,
            projection_weights=projection_weights, smoothing_nodes=smoothing_nodes,
            smoothing_weights=smoothing_weights, dense_radius_nodes=dense_radius_nodes,
        )
        transforms = {
            "u_y_absolute": _project_smooth_transform_field(
                profile_rzm=jnp.asarray(profiles.Pe_mat_physical) * y_constant,
                factor_z=a**-3, **common
            ),
            "u_e_absolute": _project_smooth_transform_field(
                profile_rzm=jnp.asarray(profiles.ne_mat_physical),
                factor_z=jnp.ones_like(z), **common
            ),
            "u_m_over_rhom": _project_smooth_transform_field(
                profile_rzm=jnp.asarray(profiles.rho_dmb_mat) / a[None, :, None] ** 3,
                factor_z=jnp.ones_like(z) / jnp.asarray(profiles.rhom_0), **common
            ),
        }
        fields = {
            "g": jnp.asarray(pkz.ukg_cross), "y": transforms["u_y_absolute"],
            "e": transforms["u_e_absolute"], "m": transforms["u_m_over_rhom"],
        }
        powers = assemble_resolved_power(
            jnp.asarray(pkz.M_array), jnp.asarray(pkz.hmf_Mz_mat),
            jnp.asarray(pkz.bias_Mz_mat), jnp.asarray(pkz.plin_kz_mat),
            fields, jnp.asarray(pkz.ukg_auto_sqr),
        )
        realized_nz = jnp.interp(z, realized_z, realized_nz_saved)
        realized_nz = realized_nz / jsi.trapezoid(realized_nz, z)
        wkappa = jnp.interp(z, halo_z_saved, wkappa_saved)
        coarse_cls = project_resolved_power_to_cls_jax(
            ell_projection, k, z, jnp.asarray(pkz.chi_array), powers,
            realized_nz, wkappa, tau_constant,
        )
        if int(limber_ell_nodes) == 2049:
            return coarse_cls
        log_ell_projection = jnp.log(ell_projection + 0.5)
        log_ell = jnp.log(ell + 0.5)
        return {
            name: jnp.exp(
                jnp.interp(
                    log_ell,
                    log_ell_projection,
                    jnp.log(jnp.maximum(value, jnp.finfo(jnp.float64).tiny)),
                )
            )
            for name, value in coarse_cls.items()
        }

    def vector_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return apply_frozen_estimator_jax(dense_fn(theta), pixel_window, window)

    if jit_compile:
        dense_fn = jax.jit(dense_fn)
        vector_fn = jax.jit(vector_fn)
    return ThreeProbeForwardModel(
        vector_fn=vector_fn,
        dense_fn=dense_fn,
        metadata={
            "parameter_names": list(PARAMETER_NAMES),
            "grid": {"nr": int(profile_nr), "nM": 24, "nz": int(profile_nz), "nk": 48},
            "dense_radius_nodes": int(dense_radius_nodes), "n_los": 32,
            "limber_ell_nodes_requested": int(limber_ell_nodes),
            "limber_ell_nodes_actual": int(ell_projection.shape[0]),
            "smoothing_nodes": 64, "smoothing_embedded": True,
            "external_profile_bell_applied": False,
            "float32_quantization_probe": float32_quantization_probe,
            "backend": jax.default_backend(),
            "contract_sha256": contract.contract_sha256,
        },
    )
