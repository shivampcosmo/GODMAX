"""Common-support resolved halo-model power for the three-probe SBI experiment.

This module is deliberately an adapter, not a replacement for GODMAX.  GODMAX
constructs the HMF, halo bias, HOD and radial-profile transforms.  The adapter
then integrates those arrays directly on the catalog-matched grid so no cached
low-mass completion can leak into a resolved-only theory prediction.

Masses are numerical ``Msun/h`` and the mass integration measure is ``dlnM``.
The catalog particle-count proxy is provisionally identified with M200c for
this comparison; that is not a recovered spherical-overdensity mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.integrate as jsi
from jax import lax


FIELD_ORDER = ("g", "y", "e", "m")
PAIR_ORDER = tuple(
    (left, right)
    for index, left in enumerate(FIELD_ORDER)
    for right in FIELD_ORDER[index:]
)


@dataclass(frozen=True)
class ResolvedSupport:
    """Frozen numerical support shared by catalog rows and theory integrals."""

    mass_min_hmsun: float = 5.0e11
    mass_max_hmsun: float = 1.0e16
    z_min: float = 0.3
    z_max: float = 0.5
    mass_semantics: str = "provisional_particle_count_proxy_as_M200c"
    unresolved_completion: bool = False


def resolved_halo_overrides(
    support: ResolvedSupport,
    *,
    n_mass: int,
    n_redshift: int,
    n_k: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return constructor overrides that establish common support before GODMAX.

    Continuous quadrature includes the upper endpoint as a zero-measure boundary;
    this is equivalent to the catalog's strict upper predicate for integration.
    """

    if support.unresolved_completion:
        raise ValueError("Resolved theory requires unresolved_completion=false")
    if min(n_mass, n_redshift, n_k) < 2:
        raise ValueError("Resolved grids require at least two samples per axis")
    halo = {
        "lg10_Mmin": float(np.log10(support.mass_min_hmsun)),
        "lg10_Mmax": float(np.log10(support.mass_max_hmsun)),
        "nM": int(n_mass),
        "zmin": float(support.z_min),
        "zmax": float(support.z_max),
        "nz": int(n_redshift),
        "nk": int(n_k),
        "mdef_Delta": 200,
        "hmf_model": "T10",
    }
    analysis = {
        "symbolic_hmf": False,
        "symbolic_pk": False,
        "zmin_for_Cls": float(support.z_min),
        "zmax_for_Cls": float(support.z_max),
        "nz_for_Cls": int(n_redshift),
    }
    return halo, analysis


def validate_resolved_inputs(
    mass: object,
    redshift: object,
    k: object,
    hmf: object,
    halo_bias: object,
    linear_power: object,
    fields: Mapping[str, object],
    galaxy_auto_second_moment: object,
    support: ResolvedSupport,
) -> None:
    """Fail closed on support, shapes, finiteness and completion policy.

    This validation belongs outside a differentiated/JIT-compiled calculation.
    ``assemble_resolved_power`` below contains only JAX array operations.
    """

    if support.unresolved_completion:
        raise ValueError("Unresolved completion is forbidden for map-matched theory")
    if support.mass_semantics != "provisional_particle_count_proxy_as_M200c":
        raise ValueError("The provisional catalog mass semantics must be explicit")

    mass_np = np.asarray(mass, dtype=np.float64)
    z_np = np.asarray(redshift, dtype=np.float64)
    k_np = np.asarray(k, dtype=np.float64)
    if mass_np.ndim != 1 or z_np.ndim != 1 or k_np.ndim != 1:
        raise ValueError("Mass, redshift and k grids must be one-dimensional")
    if min(mass_np.size, z_np.size, k_np.size) < 2:
        raise ValueError("Resolved grids require at least two samples per axis")
    if not all(np.all(np.isfinite(array)) for array in (mass_np, z_np, k_np)):
        raise ValueError("Resolved grids contain non-finite values")
    if not all(np.all(np.diff(array) > 0.0) for array in (mass_np, z_np, k_np)):
        raise ValueError("Resolved grids must be strictly increasing")

    endpoint_rtol = 8.0 * np.finfo(np.float64).eps
    expected = (
        (mass_np[0], support.mass_min_hmsun, "mass minimum"),
        (mass_np[-1], support.mass_max_hmsun, "mass maximum"),
        (z_np[0], support.z_min, "redshift minimum"),
        (z_np[-1], support.z_max, "redshift maximum"),
    )
    for actual, target, label in expected:
        if not np.isclose(actual, target, rtol=endpoint_rtol, atol=0.0):
            raise ValueError(f"Resolved {label} differs from contract: {actual} != {target}")

    nz, nm, nk = z_np.size, mass_np.size, k_np.size
    expected_zm = (nz, nm)
    expected_kz = (nk, nz)
    expected_kzm = (nk, nz, nm)
    arrays = {
        "hmf": (hmf, expected_zm),
        "halo_bias": (halo_bias, expected_zm),
        "linear_power": (linear_power, expected_kz),
        "galaxy_auto_second_moment": (galaxy_auto_second_moment, expected_kzm),
    }
    if tuple(fields) != FIELD_ORDER:
        raise ValueError(f"Fields must be ordered exactly as {FIELD_ORDER}")
    arrays.update({f"field_{name}": (fields[name], expected_kzm) for name in FIELD_ORDER})
    for name, (value, shape) in arrays.items():
        value_np = np.asarray(value)
        if value_np.shape != shape:
            raise ValueError(f"{name} has shape {value_np.shape}; expected {shape}")
        if value_np.dtype != np.dtype(np.float64):
            raise ValueError(f"{name} must be float64, found {value_np.dtype}")
        if not np.all(np.isfinite(value_np)):
            raise ValueError(f"{name} contains non-finite values")


def _integrate_dlnm(integrand: jnp.ndarray, mass: jnp.ndarray) -> jnp.ndarray:
    return jsi.trapezoid(integrand, x=jnp.log(mass), axis=-1)


def assemble_resolved_power(
    mass: jnp.ndarray,
    hmf_zm: jnp.ndarray,
    halo_bias_zm: jnp.ndarray,
    linear_power_kz: jnp.ndarray,
    fields_kzm: Mapping[str, jnp.ndarray],
    galaxy_auto_second_moment_kzm: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Assemble raw resolved 1h/2h powers using one mass quadrature.

    All field transforms have shape ``(nk, nz, nM)``.  HMF and halo bias have
    shape ``(nz, nM)`` and linear power has shape ``(nk, nz)``.  The routine is
    pure JAX: callers run :func:`validate_resolved_inputs` once before tracing.
    """

    hmf_kzm = hmf_zm[None, :, :]
    bias_kzm = halo_bias_zm[None, :, :]

    effective_bias = {
        name: _integrate_dlnm(fields_kzm[name] * hmf_kzm * bias_kzm, mass)
        for name in FIELD_ORDER
    }
    result: dict[str, jnp.ndarray] = {
        f"b{name}_resolved": effective_bias[name] for name in FIELD_ORDER
    }

    for left, right in PAIR_ORDER:
        label = f"P{left}{right}"
        if left == right == "g":
            one_halo_integrand = galaxy_auto_second_moment_kzm * hmf_kzm
        else:
            one_halo_integrand = fields_kzm[left] * fields_kzm[right] * hmf_kzm
        one_halo = _integrate_dlnm(one_halo_integrand, mass)
        two_halo = effective_bias[left] * effective_bias[right] * linear_power_kz
        result[f"{label}_1h"] = one_halo
        result[f"{label}_2h"] = two_halo
        result[f"{label}_resolved"] = one_halo + two_halo
    return result


def fields_from_godmax(
    pkz: object,
    map_matched_transforms: Mapping[str, object],
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Create g/y/e/m field arrays without reading completed GODMAX biases.

    The electron input is the transform of absolute comoving electron number
    density, matching the tau painter.  Matter is the direct transform of the
    painted density divided by ``rho_m0``.  The
    caller must provide painter-support-matched y/electron/matter transforms;
    silently falling back to the untruncated cached profiles is forbidden.
    """

    required = ("u_y_absolute", "u_e_absolute", "u_m_over_rhom")
    missing = [name for name in required if name not in map_matched_transforms]
    if missing:
        raise ValueError(f"Missing map-matched profile transforms: {missing}")

    fields = {
        "g": jnp.asarray(pkz.ukg_cross),
        "y": jnp.asarray(map_matched_transforms["u_y_absolute"]),
        "e": jnp.asarray(map_matched_transforms["u_e_absolute"]),
        "m": jnp.asarray(map_matched_transforms["u_m_over_rhom"]),
    }
    return fields, jnp.asarray(pkz.ukg_auto_sqr)


def spherical_profile_transform(
    k_hmpc: jnp.ndarray,
    radius_mpch: jnp.ndarray,
    density_rzm: jnp.ndarray,
) -> jnp.ndarray:
    """Direct signed spherical transform on any requested k grid.

    Evaluating ``sin(kr)/(kr)`` directly avoids FFTLog's narrower implicit k
    support and therefore gives a genuine zero-mode limit rather than clamped
    endpoint values.  ``lax.map`` bounds peak memory while preserving gradients.
    """

    radius = jnp.asarray(radius_mpch)
    density = jnp.asarray(density_rzm)
    radial_weight = 4.0 * jnp.pi * radius[:, None, None] ** 2 * density

    def transform_one(k_value: jnp.ndarray) -> jnp.ndarray:
        kernel = jnp.sinc(k_value * radius / jnp.pi)[:, None, None]
        return jsi.trapezoid(radial_weight * kernel, x=radius, axis=0)

    return lax.map(transform_one, jnp.asarray(k_hmpc))


def map_matched_profile_transforms(
    pkz: object,
    *,
    paint_r200c_factor: float = 8.0,
) -> dict[str, jnp.ndarray]:
    """Differentiable transforms with the resolved painter's radial support.

    This is the spherical-support approximation currently pre-registered for
    the analytic null.  Its agreement with the projected map operator remains
    a later acceptance gate; no empirical normalization is applied here.
    """

    r = jnp.asarray(pkz.r_array)
    mask = r[:, None, None] <= (
        jnp.asarray(paint_r200c_factor) * jnp.asarray(pkz.r200c_mat)[None, :, :]
    )
    k_target = jnp.asarray(pkz.kPk_array)
    return {
        "u_y_absolute": spherical_profile_transform(
            k_target, r, jnp.asarray(pkz.y3d_mat) * mask
        ),
        "u_e_absolute": spherical_profile_transform(
            k_target, r, jnp.asarray(pkz.ne_mat) * mask
        ),
        "u_m_over_rhom": spherical_profile_transform(
            k_target, r, jnp.asarray(pkz.rho_dmb_mat) * mask / jnp.asarray(pkz.rhom_0)
        ),
    }


def assemble_from_godmax(
    pkz: object,
    map_matched_transforms: Mapping[str, object],
    support: ResolvedSupport,
) -> dict[str, jnp.ndarray]:
    """Host-only validated bridge to raw powers.

    Validate once before inference; a traced likelihood must call
    :func:`assemble_resolved_power` directly because this convenience bridge
    intentionally performs NumPy fail-closed checks.
    """

    fields, galaxy_auto = fields_from_godmax(pkz, map_matched_transforms)
    validate_resolved_inputs(
        pkz.M_array,
        pkz.z_array,
        pkz.kPk_array,
        pkz.hmf_Mz_mat,
        pkz.bias_Mz_mat,
        pkz.plin_kz_mat,
        fields,
        galaxy_auto,
        support,
    )
    return assemble_resolved_power(
        jnp.asarray(pkz.M_array),
        jnp.asarray(pkz.hmf_Mz_mat),
        jnp.asarray(pkz.bias_Mz_mat),
        jnp.asarray(pkz.plin_kz_mat),
        fields,
        galaxy_auto,
    )
