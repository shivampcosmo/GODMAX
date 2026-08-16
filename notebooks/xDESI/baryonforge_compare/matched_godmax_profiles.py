"""Comparison-only GODMAX profiles with an asymptotic mass boundary.

The Schneider baryonic-correction model normalizes the redistributed
components at infinity.  Native GODMAX approximates that boundary at
``8 R200c``.  This opt-in subclass keeps every native formula and the native
``0.01 R200c`` lower boundary, but evaluates the affected normalization and
hydrostatic chain to a wider configured upper boundary with a memory-neutral
Gauss--Legendre rule in log radius.  The global/native trapezoid count remains
64; the comparison rule also uses at most 64 simultaneous radial nodes.
The separate non-thermal-fraction cap remains on GODMAX's native convention;
its redshift evolution is disabled in the matched parameter crosswalk.

Nothing imports this class on the normal GODMAX path.  The matched comparison
selects it explicitly and records the concrete class in product provenance.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

from get_radial_profiles import G_new, Profiles


VARIANT = "asymptotic_total_mass_v1"
RMAX_ANALYSIS_KEY = "comparison_extended_profile_rmax_r200c"
METHOD_ANALYSIS_KEY = "comparison_extended_profile_integration_method"
POINTS_ANALYSIS_KEY = "comparison_extended_profile_num_points"
INTEGRATION_METHOD = "gauss_legendre_log"
NATIVE_CORE_POINTS = 64
MAX_COMPARISON_POINTS = 64


class AsymptoticNormalizationProfiles(Profiles):
    """Use a converged outer boundary without enlarging radial workspaces."""

    normalization_variant = VARIANT
    integration_rmin_r200c = 0.01

    def __init__(
        self,
        sim_params_dict: dict,
        halo_params_dict: dict,
        analysis_dict: Mapping[str, Any] | None = None,
        other_params_dict: dict | None = None,
        base_class_obj=None,
    ) -> None:
        if analysis_dict is None or RMAX_ANALYSIS_KEY not in analysis_dict:
            raise KeyError(
                f"{type(self).__name__} requires analysis.{RMAX_ANALYSIS_KEY}."
            )
        rmax = float(analysis_dict[RMAX_ANALYSIS_KEY])
        if not math.isfinite(rmax) or rmax <= 8.0:
            raise ValueError(
                f"analysis.{RMAX_ANALYSIS_KEY} must be finite and greater than "
                f"the native 8 R200c boundary; got {rmax!r}."
            )
        method = str(analysis_dict.get(METHOD_ANALYSIS_KEY, ""))
        if method != INTEGRATION_METHOD:
            raise ValueError(
                f"analysis.{METHOD_ANALYSIS_KEY} must be {INTEGRATION_METHOD!r}; "
                f"got {method!r}."
            )
        points = int(analysis_dict.get(POINTS_ANALYSIS_KEY, -1))
        if not 1 < points <= MAX_COMPARISON_POINTS:
            raise ValueError(
                f"analysis.{POINTS_ANALYSIS_KEY} must be in [2, "
                f"{MAX_COMPARISON_POINTS}]; got {points!r}."
            )
        core_points = int(analysis_dict.get("num_points_trapz_int", -1))
        if core_points != NATIVE_CORE_POINTS:
            raise ValueError(
                "The matched comparison keeps analysis.num_points_trapz_int at "
                f"the native {NATIVE_CORE_POINTS}; got {core_points!r}."
            )

        # Set this before Profiles.__init__: construction dispatches virtually
        # through every overridden method below.
        self.integration_rmax_r200c = rmax
        self.extended_profile_integration_method = method
        self.extended_profile_num_points = points
        nodes, weights = np.polynomial.legendre.leggauss(points)
        self.extended_profile_log_nodes = jnp.asarray(nodes, dtype=jnp.float64)
        self.extended_profile_log_weights = jnp.asarray(weights, dtype=jnp.float64)
        super().__init__(
            sim_params_dict,
            halo_params_dict,
            dict(analysis_dict),
            other_params_dict,
            base_class_obj=base_class_obj,
        )

    def _log_rule(self, lower_radius, upper_radius):
        """Map the fixed Legendre rule to a requested log-radius interval."""

        lower = jnp.log(lower_radius)
        upper = jnp.log(upper_radius)
        half_width = 0.5 * (upper - lower)
        midpoint = 0.5 * (upper + lower)
        log_radius = midpoint + half_width * self.extended_profile_log_nodes
        weights = half_width * self.extended_profile_log_weights
        return log_radius, weights

    def _mass_integral(self, density_function, lower_radius, upper_radius, jz, jM):
        """Integrate ``4 pi r^2 rho dr`` with the fixed log-radius rule."""

        log_radius, weights = self._log_rule(lower_radius, upper_radius)
        radius = jnp.exp(log_radius)
        density = vmap(density_function, (0, None, None, None))(
            jnp.arange(self.extended_profile_num_points), jz, jM, radius
        )
        return jnp.sum(weights * 4.0 * jnp.pi * radius**3 * density)

    @partial(jit, static_argnums=(0,))
    def get_nfw_norm(self, jz, jM):
        """Normalize NFW with the same 64-node high-order rule."""

        r200c = self.r200c_mat[jz, jM]
        integral = self._mass_integral(
            self.get_rho_nfw_unnorm,
            0.01 * r200c,
            r200c,
            jz,
            jM,
        )
        return self.M_array[jM] / integral

    @partial(jit, static_argnums=(0,))
    def get_Mtot(self, jz, jM, rmax_r200c=None):
        """Normalize the extended DMO mass on the comparison boundary."""

        r200c = self.r200c_mat[jz, jM]
        return self._mass_integral(
            self.get_rho_nfw_normed,
            0.01 * r200c,
            self.integration_rmax_r200c * r200c,
            jz,
            jM,
        )

    @partial(jit, static_argnums=(0,))
    def get_rho_gas_norm(self, jz, jM, rmax_r200c=None):
        """Normalize gas to ``f_gas M_tot`` on the same wide domain."""

        r200c = self.r200c_mat[jz, jM]
        integral = self._mass_integral(
            self.get_rho_gas_unnorm,
            0.01 * r200c,
            self.integration_rmax_r200c * r200c,
            jz,
            jM,
        )
        return self.fgas_mat[jz, jM] * self.Mtot_mat[jz, jM] / integral

    @partial(jit, static_argnums=(0,))
    def get_Mdmb(self, jr, jz, jM, r_array_here=None):
        """Build the HSE enclosed-mass table with the high-order rule."""

        if r_array_here is None:
            radius = self.r_array[jr]
        else:
            radius = r_array_here[jr]
        minimum_radius = jnp.minimum(5.0e-4, 0.005 * self.r200c_mat[jz, jM])
        return self._mass_integral(
            self.get_rho_dmb,
            minimum_radius,
            radius,
            jz,
            jM,
        )

    @partial(jit, static_argnums=(0,))
    def get_Ptot(self, jr, jz, jM, r_array_here=None, rmax_r200c=None):
        """Apply the HSE boundary condition at the same infinity proxy."""

        if r_array_here is None:
            radius = self.r_array[jr]
        else:
            radius = r_array_here[jr]
        log_radius, weights = self._log_rule(
            radius,
            self.integration_rmax_r200c * self.r200c_mat[jz, jM],
        )
        query_radius = jnp.exp(log_radius)
        gas_density = vmap(self.get_rho_gas_normed, (0, None, None, None))(
            jnp.arange(self.extended_profile_num_points),
            jz,
            jM,
            query_radius,
        )
        enclosed_mass = jnp.exp(
            jnp.interp(
                log_radius,
                jnp.log(self.r_array),
                jnp.log(self.Mdmb_mat[:, jz, jM]),
            )
        )
        pressure = jnp.sum(
            weights * gas_density * enclosed_mass * G_new / query_radius
        )
        h = self.cosmo_params["H0"] / 100.0
        return jnp.clip(pressure, 1.0e-30) * h**2
