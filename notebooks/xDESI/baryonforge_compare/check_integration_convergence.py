#!/usr/bin/env python
"""Check the GODMAX infinity proxy and quadrature on its full profile grid."""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (
    WORKSPACE_ROOT,
    canonical_json,
    comparison_source_manifest,
    load_config,
    profile_integration_contract,
    resolve_path,
    runtime_version_manifest,
    sha256_file,
    sha256_json,
    validate_parameter_crosswalk,
)
from compare_profiles import _godmax_profiles
from get_radial_profiles import G_new


CONVERGENCE_SCHEMA = "godmax_asymptotic_integration_convergence_v3"
LOG_TRAPEZOID = "uniform_log_trapezoid"
LOG_GAUSS_LEGENDRE = "gauss_legendre_log"


def _method_tag(method: str) -> str:
    if method == LOG_TRAPEZOID:
        return "trap"
    if method == LOG_GAUSS_LEGENDRE:
        return "gl"
    raise ValueError(f"Unsupported log-radius quadrature method {method!r}.")


def convergence_variants(
    config: Mapping[str, Any],
) -> tuple[dict[str, tuple[float, int, str]], str, str]:
    """Build the convergence scan from the frozen production/reference config."""

    contract = profile_integration_contract(config)["godmax"]
    production_rmax = float(contract["r_max_R200c"])
    production_points = int(contract["extended_num_points"])
    production_method = str(contract["extended_integration_method"])
    reference_rmax = float(
        config["validation"]["asymptotic_convergence_reference_rmax_R200c"]
    )
    reference_points = int(
        config["validation"]["asymptotic_convergence_reference_points"]
    )
    reference_method = str(
        config["validation"].get(
            "asymptotic_convergence_reference_method", LOG_GAUSS_LEGENDRE
        )
    )
    if reference_rmax <= production_rmax:
        raise ValueError(
            "The convergence reference boundary must exceed the production "
            f"boundary: reference={reference_rmax}, production={production_rmax}."
        )
    if reference_points <= production_points:
        raise ValueError(
            "The convergence reference quadrature must be finer than production: "
            f"reference={reference_points}, production={production_points}."
        )

    production_tag = _method_tag(production_method)
    reference_tag = _method_tag(reference_method)
    production = (
        f"production_{production_rmax:g}R_{production_tag}{production_points}"
    )
    reference = f"reference_{reference_rmax:g}R_{reference_tag}{reference_points}"
    variants = {
        "native_8R_trap64": (8.0, 64, LOG_TRAPEZOID),
        "failed_128R_trap64": (128.0, 64, LOG_TRAPEZOID),
        "old_128R_trap256": (128.0, 256, LOG_TRAPEZOID),
        production: (production_rmax, production_points, production_method),
        f"points_{production_rmax:g}R_{reference_tag}{reference_points}": (
            production_rmax,
            reference_points,
            reference_method,
        ),
        f"bound_{reference_rmax:g}R_{production_tag}{production_points}": (
            reference_rmax,
            production_points,
            production_method,
        ),
        reference: (reference_rmax, reference_points, reference_method),
    }
    return variants, production, reference


@lru_cache(maxsize=None)
def _legendre_rule(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(n_points))
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _log_rule(
    lower_log_radius: float | np.ndarray,
    upper_log_radius: float | np.ndarray,
    n_points: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return log-radius nodes and optional quadrature weights."""

    lower = np.asarray(lower_log_radius, dtype=np.float64)
    upper = np.asarray(upper_log_radius, dtype=np.float64)
    if method == LOG_TRAPEZOID:
        fraction = np.linspace(0.0, 1.0, int(n_points))
        return lower + fraction.reshape((-1,) + (1,) * lower.ndim) * (upper - lower), None
    if method == LOG_GAUSS_LEGENDRE:
        nodes, base_weights = _legendre_rule(int(n_points))
        half_width = 0.5 * (upper - lower)
        midpoint = 0.5 * (upper + lower)
        reshape = (-1,) + (1,) * lower.ndim
        return (
            midpoint + nodes.reshape(reshape) * half_width,
            base_weights.reshape(reshape) * half_width,
        )
    raise ValueError(f"Unsupported log-radius quadrature method {method!r}.")


def _integrate_log(
    values: np.ndarray,
    log_radius: np.ndarray,
    weights: np.ndarray | None,
) -> np.ndarray:
    if weights is None:
        return np.trapz(values, x=log_radius, axis=0)
    return np.sum(weights * values, axis=0)


def _log_mass_integral(
    shape: np.ndarray,
    radius: np.ndarray,
    log_radius: np.ndarray,
    weights: np.ndarray | None,
) -> np.ndarray:
    return _integrate_log(4.0 * np.pi * radius**3 * shape, log_radius, weights)


def independent_normalizers(
    profiles,
    *,
    rmax_r200c: float,
    n_points: int,
    method: str,
) -> dict[str, np.ndarray]:
    """Reproduce a configured log-radius normalizer chain with NumPy."""

    r200 = np.asarray(profiles.r200c_mat, dtype=np.float64)
    concentration = np.asarray(profiles.conc_Mz_mat, dtype=np.float64)
    truncation = np.asarray(profiles.rt_mat, dtype=np.float64)
    mass = np.asarray(profiles.M_array, dtype=np.float64)[None, :]

    inner_log_scaled, inner_weights = _log_rule(
        np.log(0.01), np.log(1.0), n_points, method
    )
    inner_scaled = np.exp(inner_log_scaled)[:, None, None]
    inner_radius = inner_scaled * r200[None, :, :]
    inner_x = inner_radius / (r200 / concentration)[None, :, :]
    inner_nfw = 1.0 / (inner_x * (1.0 + inner_x) ** 2)
    if bool(profiles.nfw_trunc):
        inner_nfw /= (1.0 + (inner_radius / truncation[None, :, :]) ** 2) ** 2
    inner_log_radius = np.log(inner_radius)
    inner_rule_weights = (
        None if inner_weights is None else inner_weights[:, None, None]
    )
    nfw_norm = mass / _log_mass_integral(
        inner_nfw, inner_radius, inner_log_radius, inner_rule_weights
    )

    log_scaled, scaled_weights = _log_rule(
        np.log(0.01), np.log(float(rmax_r200c)), n_points, method
    )
    scaled = np.exp(log_scaled)[:, None, None]
    radius = scaled * r200[None, :, :]
    log_radius = np.log(radius)
    rule_weights = None if scaled_weights is None else scaled_weights[:, None, None]
    x_nfw = radius / (r200 / concentration)[None, :, :]
    nfw = 1.0 / (x_nfw * (1.0 + x_nfw) ** 2)
    if bool(profiles.nfw_trunc):
        nfw /= (1.0 + (radius / truncation[None, :, :]) ** 2) ** 2
    total_mass = nfw_norm * _log_mass_integral(
        nfw, radius, log_radius, rule_weights
    )

    core = np.asarray(profiles.r_co_mat, dtype=np.float64)[None, :, :]
    ejection = np.asarray(profiles.r_ej_mat, dtype=np.float64)[None, :, :]
    beta = np.asarray(profiles.beta_mat, dtype=np.float64)[None, :, :]
    gas_shape = 1.0 / (
        (1.0 + radius / core) ** beta
        * (1.0 + (radius / ejection) ** float(profiles.gamma_rhogas))
        ** ((float(profiles.delta_rhogas) - beta) / float(profiles.gamma_rhogas))
    )
    if bool(profiles.model_galaxies):
        raise ValueError(
            "The independent convergence chain requires model_galaxies=false; "
            "the galaxy-HOD stellar fractions need the native JAX chain."
        )

    mass_1d = np.asarray(profiles.M_array, dtype=np.float64)
    shape = (int(profiles.nz), mass_1d.size)
    fstar_total = np.broadcast_to(
        float(profiles.A_starcga)
        * (float(profiles.M1_starcga) / mass_1d) ** float(profiles.eta_star),
        shape,
    ).copy()
    fstar_central = np.broadcast_to(
        float(profiles.A_starcga)
        * (float(profiles.M1_starcga) / mass_1d) ** float(profiles.eta_cga),
        shape,
    ).copy()
    fstar_satellite = fstar_total - fstar_central
    baryon_fraction = float(profiles.Ob0) / float(profiles.Om0)
    gas_fraction = baryon_fraction - fstar_total
    collisionless_fraction = 1.0 - baryon_fraction + fstar_satellite

    gas_integral = _log_mass_integral(
        gas_shape, radius, log_radius, rule_weights
    )
    gas_norm = gas_fraction * total_mass / gas_integral
    return {
        "nfw_norm": nfw_norm,
        "Mtot": total_mass,
        "fstar_total": fstar_total,
        "fstar_central": fstar_central,
        "fstar_satellite": fstar_satellite,
        "fgas": gas_fraction,
        "fclm": collisionless_fraction,
        "gas_shape_integral": gas_integral,
        "gas_norm": gas_norm,
    }


def _relative(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (
        np.asarray(candidate, dtype=np.float64)
        / np.asarray(reference, dtype=np.float64)
        - 1.0
    )


def pressure_convergence(
    profiles,
    variants: Mapping[str, tuple[float, int, str]],
    reference_name: str,
) -> dict[str, Any]:
    """Isolate the HSE upper-bound/quadrature error on nine grid neighborhoods."""

    mass_targets = np.asarray([1.0e13, 1.0e14, 1.0e15])
    redshift_targets = np.asarray([0.65, 0.80, 0.95])
    scaled_radius = np.asarray([0.02, 0.1, 0.5, 1.0, 5.0])
    mass_grid = np.asarray(profiles.M_array, dtype=np.float64)
    redshift_grid = np.asarray(profiles.z_array, dtype=np.float64)
    radius_grid = np.asarray(profiles.r_array, dtype=np.float64)
    log_radius_grid = np.log(radius_grid)
    r200_grid = np.asarray(profiles.r200c_mat, dtype=np.float64)
    gas_norm_grid = np.asarray(profiles.rho_gas_norm_mat, dtype=np.float64)
    core_grid = np.asarray(profiles.r_co_mat, dtype=np.float64)
    ejection_grid = np.asarray(profiles.r_ej_mat, dtype=np.float64)
    beta_grid = np.asarray(profiles.beta_mat, dtype=np.float64)
    mass_profile_grid = np.asarray(profiles.Mdmb_mat, dtype=np.float64)

    nodes = []
    values = {name: [] for name in variants}
    for redshift in redshift_targets:
        jz = int(np.argmin(np.abs(redshift_grid - redshift)))
        for mass in mass_targets:
            jm = int(np.argmin(np.abs(np.log(mass_grid / mass))))
            r200 = float(r200_grid[jz, jm])
            beta = float(beta_grid[jz, jm])
            core = float(core_grid[jz, jm])
            ejection = float(ejection_grid[jz, jm])
            gas_norm = float(gas_norm_grid[jz, jm])
            cumulative_mass = np.maximum(mass_profile_grid[:, jz, jm], 1.0e-300)
            nodes.append(
                {
                    "target_z": float(redshift),
                    "grid_z": float(redshift_grid[jz]),
                    "target_mass_hMsun": float(mass),
                    "grid_mass_hMsun": float(mass_grid[jm]),
                }
            )
            for variant, (upper, n_points, method) in variants.items():
                node_pressure = []
                for x_value in scaled_radius:
                    radius = float(x_value * r200)
                    log_query, weights = _log_rule(
                        np.log(radius),
                        np.log(float(upper) * r200),
                        int(n_points),
                        method,
                    )
                    query = np.exp(log_query)
                    gas_shape = 1.0 / (
                        (1.0 + query / core) ** beta
                        * (1.0 + (query / ejection) ** float(profiles.gamma_rhogas))
                        ** (
                            (float(profiles.delta_rhogas) - beta)
                            / float(profiles.gamma_rhogas)
                        )
                    )
                    enclosed_mass = np.exp(
                        np.interp(
                            log_query,
                            log_radius_grid,
                            np.log(cumulative_mass),
                        )
                    )
                    integrand = gas_norm * gas_shape * enclosed_mass / query
                    node_pressure.append(
                        _integrate_log(integrand, log_query, weights)
                    )
                values[variant].append(node_pressure)

    arrays = {name: np.asarray(value) for name, value in values.items()}
    reference = arrays[reference_name]
    errors = {name: _relative(value, reference) for name, value in arrays.items()}
    return {
        "nodes": nodes,
        "scaled_radius_R200c": scaled_radius,
        "values": arrays,
        "relative_errors": errors,
        "method": (
            "Configured log-radius quadrature with the production gas "
            "normalization and native log-interpolated Mdmb table held fixed; "
            "isolates upper-bound and quadrature effects"
        ),
    }


def _variant_component_densities(
    profiles,
    state: Mapping[str, np.ndarray],
    jz: int,
    jM: int,
    radius: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the current comparison's three matter components."""

    radius = np.asarray(radius, dtype=np.float64)
    r200 = float(np.asarray(profiles.r200c_mat)[jz, jM])
    concentration = float(np.asarray(profiles.conc_Mz_mat)[jz, jM])
    truncation = float(np.asarray(profiles.rt_mat)[jz, jM])
    x_nfw = radius / (r200 / concentration)
    nfw_shape = 1.0 / (x_nfw * (1.0 + x_nfw) ** 2)
    if bool(profiles.nfw_trunc):
        nfw_shape /= (1.0 + (radius / truncation) ** 2) ** 2

    core = float(np.asarray(profiles.r_co_mat)[jz, jM])
    ejection = float(np.asarray(profiles.r_ej_mat)[jz, jM])
    beta = float(np.asarray(profiles.beta_mat)[jz, jM])
    gas_shape = 1.0 / (
        (1.0 + radius / core) ** beta
        * (1.0 + (radius / ejection) ** float(profiles.gamma_rhogas))
        ** ((float(profiles.delta_rhogas) - beta) / float(profiles.gamma_rhogas))
    )
    gas = float(state["gas_norm"][jz, jM]) * gas_shape

    half_mass_radius = float(np.asarray(profiles.Rh_mat)[jz, jM])
    central_amplitude = (
        float(state["fstar_central"][jz, jM])
        * float(state["Mtot"][jz, jM])
        / (4.0 * np.pi**1.5 * half_mass_radius)
    )
    central = (
        central_amplitude
        * np.exp(-((0.5 * radius / half_mass_radius) ** 2))
        / radius**2
    )
    collisionless = (
        float(state["fclm"][jz, jM]) * float(state["nfw_norm"][jz, jM]) * nfw_shape
    )
    return {
        "gas": gas,
        "central": central,
        "collisionless": collisionless,
        "total": gas + central + collisionless,
    }


def _rebuild_mdmb_node(
    profiles,
    state: Mapping[str, np.ndarray],
    jz: int,
    jM: int,
    n_points: int,
    method: str,
) -> np.ndarray:
    """Mirror ``get_Mdmb`` at every radius in the GODMAX table."""

    r200 = float(np.asarray(profiles.r200c_mat)[jz, jM])
    minimum_radius = min(5.0e-4, 0.005 * r200)
    output = []
    for outer_radius in np.asarray(profiles.r_array, dtype=np.float64):
        log_radius, weights = _log_rule(
            np.log(minimum_radius),
            np.log(float(outer_radius)),
            int(n_points),
            method,
        )
        radius = np.exp(log_radius)
        density = _variant_component_densities(profiles, state, jz, jM, radius)["total"]
        output.append(
            _integrate_log(4.0 * np.pi * radius**3 * density, log_radius, weights)
        )
    values = np.asarray(output, dtype=np.float64)
    if not np.all(np.isfinite(values) & (values > 0.0)):
        raise ValueError("The rebuilt Mdmb table is not finite and strictly positive.")
    return values


def _node_indices(profiles) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    mass_targets = np.asarray([1.0e13, 1.0e14, 1.0e15])
    redshift_targets = np.asarray([0.65, 0.80, 0.95])
    mass_grid = np.asarray(profiles.M_array, dtype=np.float64)
    redshift_grid = np.asarray(profiles.z_array, dtype=np.float64)
    nodes: list[dict[str, Any]] = []
    indices: list[tuple[int, int]] = []
    for redshift in redshift_targets:
        jz = int(np.argmin(np.abs(redshift_grid - redshift)))
        for mass in mass_targets:
            jM = int(np.argmin(np.abs(np.log(mass_grid / mass))))
            indices.append((jz, jM))
            nodes.append(
                {
                    "target_z": float(redshift),
                    "grid_z": float(redshift_grid[jz]),
                    "grid_z_index": jz,
                    "target_mass_hMsun": float(mass),
                    "grid_mass_hMsun": float(mass_grid[jM]),
                    "grid_mass_index": jM,
                }
            )
    return nodes, indices


def full_chain_pressure_convergence(
    profiles,
    candidates: Mapping[str, Mapping[str, np.ndarray]],
    variants: Mapping[str, tuple[float, int, str]],
    production_name: str,
    reference_name: str,
) -> dict[str, Any]:
    """Independently rebuild each variant through Mdmb and HSE pressure."""

    if bool(profiles.model_galaxies):
        raise ValueError("Full-chain NumPy convergence requires model_galaxies=false.")
    if bool(profiles.backreaction):
        raise ValueError("Full-chain NumPy convergence requires backreaction=false.")

    import jax.numpy as jnp

    nodes, indices = _node_indices(profiles)
    scaled_radius = np.asarray([0.02, 0.1, 0.5, 1.0, 5.0])
    radius_grid = np.asarray(profiles.r_array, dtype=np.float64)
    log_radius_grid = np.log(radius_grid)
    r200_grid = np.asarray(profiles.r200c_mat, dtype=np.float64)
    h = float(profiles.cosmo_params["H0"]) / 100.0
    values: dict[str, np.ndarray] = {}
    mdmb_values: dict[str, np.ndarray] = {}
    interpolation_counts: dict[str, dict[str, int]] = {}
    production_jax_values: list[list[float]] = []

    for name, (upper, n_points, method) in variants.items():
        state = candidates[name]
        node_pressures: list[list[float]] = []
        node_masses: list[np.ndarray] = []
        counts = {"below_table": 0, "above_table": 0, "total": 0}
        for jz, jM in indices:
            r200 = float(r200_grid[jz, jM])
            mdmb = _rebuild_mdmb_node(
                profiles, state, jz, jM, n_points, method
            )
            node_masses.append(mdmb)
            pressure_at_radius: list[float] = []
            jax_pressure_at_radius: list[float] = []
            for scaled_value in scaled_radius:
                evaluation_radius = float(scaled_value * r200)
                log_query, weights = _log_rule(
                    np.log(evaluation_radius),
                    np.log(float(upper) * r200),
                    int(n_points),
                    method,
                )
                query = np.exp(log_query)
                gas = _variant_component_densities(profiles, state, jz, jM, query)[
                    "gas"
                ]
                enclosed_mass = np.exp(
                    np.interp(log_query, log_radius_grid, np.log(mdmb))
                )
                counts["below_table"] += int(
                    np.count_nonzero(log_query < log_radius_grid[0])
                )
                counts["above_table"] += int(
                    np.count_nonzero(log_query > log_radius_grid[-1])
                )
                counts["total"] += int(log_query.size)
                pressure = _integrate_log(
                    gas * enclosed_mass * float(G_new) / query,
                    log_query,
                    weights,
                )
                pressure_at_radius.append(max(float(pressure), 1.0e-30) * h**2)
                if name == production_name:
                    jax_pressure_at_radius.append(
                        float(
                            profiles.get_Ptot(
                                0,
                                jz,
                                jM,
                                r_array_here=jnp.asarray([evaluation_radius]),
                            )
                        )
                    )
            node_pressures.append(pressure_at_radius)
            if name == production_name:
                production_jax_values.append(jax_pressure_at_radius)
        values[name] = np.asarray(node_pressures, dtype=np.float64)
        mdmb_values[name] = np.asarray(node_masses, dtype=np.float64)
        interpolation_counts[name] = counts

    reference = values[reference_name]
    errors = {name: _relative(value, reference) for name, value in values.items()}
    production_mdmb = np.asarray(
        [np.asarray(profiles.Mdmb_mat)[:, jz, jM] for jz, jM in indices],
        dtype=np.float64,
    )
    production_reproduction = {
        "Mdmb": float(
            np.max(np.abs(_relative(mdmb_values[production_name], production_mdmb)))
        ),
        "Ptot": float(
            np.max(
                np.abs(
                    _relative(
                        values[production_name],
                        np.asarray(production_jax_values, dtype=np.float64),
                    )
                )
            )
        ),
    }
    return {
        "production": production_name,
        "reference": reference_name,
        "nodes": nodes,
        "scaled_radius_R200c": scaled_radius,
        "values": values,
        "mdmb_values": mdmb_values,
        "relative_errors": errors,
        "production_rebuild_max_abs_relative_error": production_reproduction,
        "mdmb_interpolation_counts": interpolation_counts,
        "rebuilt_fields": [
            "nfw_norm",
            "Mtot",
            "fstar_total",
            "fstar_central",
            "fstar_satellite",
            "fgas",
            "fclm",
            "gas_norm",
            "rho_gas",
            "rho_cga",
            "rho_clm",
            "Mdmb",
            "Ptot",
        ],
        "method": (
            "Independent NumPy affected-chain rebuild for every variant using "
            "the native GODMAX formulas and each configured quadrature: NFW/Mtot, simple-star "
            "fractions, gas normalization, gas/central/collisionless densities, "
            "the native radial Mdmb table, then HSE pressure with native clamped "
            "log-Mdmb interpolation"
        ),
    }


def _save_figure(
    normalizer_errors: Mapping[str, Mapping[str, float]],
    pressure: Mapping[str, Any],
    full_chain_pressure: Mapping[str, Any],
    variants: Mapping[str, tuple[float, int, str]],
    reference_name: str,
    tolerance: float,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [name for name in variants if name != reference_name]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        [normalizer_errors[name]["gas_norm"] for name in labels],
        width=width,
        label="gas normalization",
    )
    axes[0].bar(
        positions + width / 2,
        [normalizer_errors[name]["Mtot"] for name in labels],
        width=width,
        label="extended DMO mass",
    )
    axes[0].axhline(tolerance, color="black", linestyle="--", label="fixed gate")
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions, labels, rotation=25, ha="right")
    axes[0].set(
        ylabel=f"max absolute relative error vs {reference_name.removeprefix('reference_')}",
        title="Full 48 x 48 GODMAX grid",
    )
    axes[0].legend(fontsize=8)

    scaled_radius = full_chain_pressure["scaled_radius_R200c"]
    for name in labels:
        error = np.max(np.abs(full_chain_pressure["relative_errors"][name]), axis=0)
        axes[1].plot(scaled_radius, error, marker="o", label=name)
    conditional_name = str(full_chain_pressure["production"])
    conditional_error = np.max(
        np.abs(pressure["relative_errors"][conditional_name]), axis=0
    )
    axes[1].plot(
        scaled_radius,
        conditional_error,
        color="black",
        linestyle=":",
        marker="s",
        label=f"{conditional_name} held-fixed diagnostic",
    )
    axes[1].axhline(tolerance, color="black", linestyle="--", label="fixed gate")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel=r"radius [$R_{200c}$]",
        ylabel="max absolute pressure relative error",
        title="Nine mass-redshift neighborhoods",
    )
    axes[1].legend(fontsize=8)
    fig.suptitle("GODMAX asymptotic-boundary convergence")
    fig.tight_layout()

    outputs = []
    for extension, kwargs in (("png", {"dpi": 180}), ("pdf", {})):
        output = output_dir / f"integration_convergence.{extension}"
        temporary = (
            output_dir / f".integration_convergence.tmp.{os.getpid()}.{extension}"
        )
        fig.savefig(temporary, bbox_inches="tight", **kwargs)
        os.replace(temporary, output)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def run(config: Mapping[str, Any], output: Path, figure_dir: Path) -> dict[str, Any]:
    crosswalk = validate_parameter_crosswalk(config)
    if not crosswalk["ok"]:
        raise ValueError(f"Parameter crosswalk failed: {crosswalk['failed']}")
    variants, production_name, reference_name = convergence_variants(config)
    profiles, _, context = _godmax_profiles(config)
    comparison_sources = comparison_source_manifest()
    profile_source = Path(__file__).with_name("compare_profiles.py")
    profile_sources = dict(comparison_sources)
    profile_sources[profile_source.relative_to(WORKSPACE_ROOT).as_posix()] = (
        sha256_file(profile_source)
    )
    profile_sources = dict(sorted(profile_sources.items()))
    convergence_sources = dict(profile_sources)
    for path in (Path(__file__).resolve(),):
        convergence_sources[path.relative_to(WORKSPACE_ROOT).as_posix()] = sha256_file(
            path
        )
    runtime_versions = runtime_version_manifest()
    candidates = {
        name: independent_normalizers(
            profiles,
            rmax_r200c=upper,
            n_points=n_points,
            method=method,
        )
        for name, (upper, n_points, method) in variants.items()
    }
    reference = candidates[reference_name]
    normalizer_errors = {}
    for name, values in candidates.items():
        normalizer_errors[name] = {
            field: float(np.max(np.abs(_relative(values[field], reference[field]))))
            for field in ("gas_norm", "Mtot")
        }

    production = candidates[production_name]
    production_fields = {
        "nfw_norm": "rho_nfw_norm_mat",
        "Mtot": "Mtot_mat",
        "fstar_total": "fstar_tot_mat",
        "fstar_central": "fstar_cen_mat",
        "fstar_satellite": "fstar_sat_mat",
        "fgas": "fgas_mat",
        "fclm": "fclm_mat",
        "gas_norm": "rho_gas_norm_mat",
    }
    actual_reproduction = {
        field: float(
            np.max(
                np.abs(
                    _relative(
                        production[field],
                        np.asarray(getattr(profiles, attribute), dtype=np.float64),
                    )
                )
            )
        )
        for field, attribute in production_fields.items()
    }
    fraction_error = float(
        np.max(
            np.abs(
                np.asarray(profiles.fgas_mat)
                + np.asarray(profiles.fstar_cen_mat)
                + np.asarray(profiles.fclm_mat)
                - 1.0
            )
        )
    )
    pressure = pressure_convergence(profiles, variants, reference_name)
    pressure_max = {
        name: float(np.max(np.abs(value)))
        for name, value in pressure["relative_errors"].items()
    }
    full_chain_pressure = full_chain_pressure_convergence(
        profiles,
        candidates,
        variants,
        production_name,
        reference_name,
    )
    full_chain_pressure_max = {
        name: float(np.max(np.abs(value)))
        for name, value in full_chain_pressure["relative_errors"].items()
    }
    full_chain_reproduction_max = max(
        full_chain_pressure["production_rebuild_max_abs_relative_error"].values()
    )
    tolerance = float(
        config["validation"]["asymptotic_convergence_max_relative_change"]
    )
    report = {
        "schema": CONVERGENCE_SCHEMA,
        "config_path": str(resolve_path(config["_config_path"])),
        "config_sha256": sha256_file(config["_config_path"]),
        "godmax_params_sha256": sha256_file(config["profiles"]["godmax_params"]),
        "profile_source_manifest_sha256": sha256_json(profile_sources),
        "convergence_source_manifest": convergence_sources,
        "convergence_source_manifest_sha256": sha256_json(convergence_sources),
        "runtime_versions": runtime_versions,
        "runtime_manifest_sha256": sha256_json(runtime_versions),
        "profiles_class_fqname": context["profiles_class_fqname"],
        "grid_shape": [int(profiles.nz), int(profiles.nM)],
        "variants": {
            name: {
                "rmax_R200c": upper,
                "n_points": n_points,
                "method": method,
            }
            for name, (upper, n_points, method) in variants.items()
        },
        "integration_memory_contract": profile_integration_contract(config)[
            "godmax"
        ],
        "production": production_name,
        "reference": reference_name,
        "normalizer_max_abs_relative_error": normalizer_errors,
        "production_numpy_vs_jax_max_abs_relative_error": actual_reproduction,
        "fraction_algebra_max_abs_error": fraction_error,
        "pressure": {
            "method": pressure["method"],
            "acceptance_role": "informational_only",
            "node_count": len(pressure["nodes"]),
            "nodes": pressure["nodes"],
            "scaled_radius_R200c": pressure["scaled_radius_R200c"].tolist(),
            "max_abs_relative_error": pressure_max,
        },
        "pressure_full_chain": {
            "method": full_chain_pressure["method"],
            "node_count": len(full_chain_pressure["nodes"]),
            "nodes": full_chain_pressure["nodes"],
            "scaled_radius_R200c": full_chain_pressure["scaled_radius_R200c"].tolist(),
            "production": production_name,
            "reference": reference_name,
            "max_abs_relative_error": full_chain_pressure_max,
            "production_rebuild_max_abs_relative_error": full_chain_pressure[
                "production_rebuild_max_abs_relative_error"
            ],
            "rebuilt_fields": full_chain_pressure["rebuilt_fields"],
            "mdmb_interpolation_counts": full_chain_pressure[
                "mdmb_interpolation_counts"
            ],
        },
        "fixed_tolerance": tolerance,
    }
    report["acceptance"] = {
        "production_gas_norm_converged": (
            normalizer_errors[production_name]["gas_norm"] <= tolerance
        ),
        "production_Mtot_converged": (
            normalizer_errors[production_name]["Mtot"] <= tolerance
        ),
        "production_full_chain_HSE_converged": (
            full_chain_pressure_max[production_name] <= tolerance
        ),
        "full_chain_rebuild_reproduces_production": (
            full_chain_reproduction_max <= 1.0e-10
        ),
        "numpy_reproduces_jax_affected_fields": (
            max(actual_reproduction.values()) <= 1.0e-10
        ),
        "fraction_algebra": fraction_error <= 1.0e-12,
        "native_core_width_restored": (
            int(profiles.num_points_trapz_int) == 64
        ),
        "extended_rule_width_not_above_native": (
            int(profiles.extended_profile_num_points)
            <= int(profiles.num_points_trapz_int)
        ),
    }
    report["ok"] = all(report["acceptance"].values())
    report["figures"] = _save_figure(
        normalizer_errors,
        pressure,
        full_chain_pressure,
        variants,
        reference_name,
        tolerance,
        figure_dir,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema"] = report["schema"]
        handle.attrs["summary_json"] = canonical_json(report)
        handle.create_dataset("z", data=np.asarray(profiles.z_array))
        handle.create_dataset("mass_hMsun", data=np.asarray(profiles.M_array))
        normalizers = handle.create_group("normalizers")
        for name, fields in candidates.items():
            child = normalizers.create_group(name)
            for field, values in fields.items():
                child.create_dataset(field, data=values, compression="lzf")
        pressure_group = handle.create_group("pressure")
        pressure_group.create_dataset(
            "scaled_radius_R200c", data=pressure["scaled_radius_R200c"]
        )
        for name, values in pressure["relative_errors"].items():
            pressure_group.create_dataset(name, data=values)
        full_chain_group = handle.create_group("pressure_full_chain")
        full_chain_group.create_dataset(
            "scaled_radius_R200c",
            data=full_chain_pressure["scaled_radius_R200c"],
        )
        for name, values in full_chain_pressure["values"].items():
            full_chain_group.create_dataset(f"{name}_values", data=values)
        for name, values in full_chain_pressure["mdmb_values"].items():
            full_chain_group.create_dataset(
                f"{name}_Mdmb_nodes", data=values, compression="lzf"
            )
        for name, values in full_chain_pressure["relative_errors"].items():
            full_chain_group.create_dataset(f"{name}_relative_error", data=values)
    os.replace(temporary, output)
    report["output_h5"] = str(output)
    report["output_h5_sha256"] = sha256_file(output)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--figure-dir")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    root = resolve_path(config["project"]["output_root"], config["_config_path"])
    output = resolve_path(
        args.output or root / "profiles" / "integration_convergence.h5"
    )
    figure_dir = resolve_path(args.figure_dir or root / "profiles" / "figures")
    report = run(config, output, figure_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
