#!/usr/bin/env python
"""Diagnose BaryonForge--GODMAX line-of-sight projection convergence.

This is an opt-in evidence generator.  It does not change either native
projector or any map/parameter configuration.  The five stored variants are

1. the legacy GODMAX Abel projection, including its endpoint clamp;
2. the accepted 12-Mpc/h GODMAX table, projected nonsingularly and truncated
   at its actual physical support;
3. an in-memory 70-Mpc/h GODMAX table, projected nonsingularly over the same
   100-comoving-Mpc line of sight as BaryonForge;
4. the native BaryonForge real-space projector; and
5. the same BaryonForge 3D profile projected by the dense nonsingular rule.

The comparison is evaluated at the nine frozen mass/redshift nodes and the
frozen 0.02--5 R200c transverse-radius grid from the comparison YAML.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import (  # noqa: E402
    REPO_ROOT,
    WORKSPACE_ROOT,
    canonical_json,
    comparison_source_manifest,
    git_is_dirty,
    git_revision,
    jsonable,
    load_config,
    load_yaml,
    profile_integration_contract,
    resolve_path,
    runtime_version_manifest,
    sha256_file,
    sha256_json,
    validate_parameter_crosswalk,
)


SCHEMA = "baryonforge_godmax_projection_convergence_v1"
VARIANTS = (
    "godmax_legacy_clamped",
    "godmax_table_truncated_nonsingular",
    "godmax_extended_nonsingular",
    "baryonforge_native",
    "baryonforge_dense_nonsingular",
)
FIELDS = ("y", "sigma_matter_physical_Msun_Mpc2", "kappa_cmb")

# This grid reaches the radial distance needed by a 100-comoving-Mpc LOS at
# every frozen transverse radius.  It is an in-memory diagnostic override,
# never a production parameter mutation.
EXTENDED_GODMAX_RMAX_COMOVING_HMPC = 70.0
EXTENDED_GODMAX_NR = 128
MATCHED_GODMAX_PROJECTION_POINTS = 128
BARYONFORGE_LOS_CUTOFF_COMOVING_MPC = 100.0

# Fixed before the durable diagnostic run.  The 512-point result is stored as
# the reference; 256 points must reproduce both dense projectors to 1e-3.
DENSE_REFERENCE_POINTS = 512
DENSE_CHECK_POINTS = 256
DENSE_CONVERGENCE_TOLERANCE = 1.0e-3
PROFILE_RESOLUTION_TOLERANCE = 2.0e-2
GODMAX_GRID_SCAN = (
    (67.11, 128),
    (70.0, 96),
    (70.0, 128),
    (80.0, 192),
)
GODMAX_GRID_REFERENCE = (80.0, 192)
GODMAX_PROJECTION_POINT_SCAN = (32, 64, 128, 256, 512)
BARYONFORGE_POINTS_PER_DECADE_SCAN = (24, 64, 128)


def required_extended_rmax_hmpc(
    transverse_comoving_hmpc: np.ndarray,
    *,
    los_cutoff_comoving_mpc: float,
    h: float,
) -> float:
    """Largest comoving GODMAX radius reached by a common BFG LOS."""

    transverse = np.asarray(transverse_comoving_hmpc, dtype=np.float64)
    if transverse.size == 0 or np.any(~np.isfinite(transverse)):
        raise ValueError("Transverse radii must be finite and non-empty.")
    if np.any(transverse < 0.0) or los_cutoff_comoving_mpc <= 0.0 or h <= 0.0:
        raise ValueError("Projection radii, LOS cutoff, and h must be positive.")
    return float(
        np.sqrt(
            np.max(transverse) ** 2 + (float(h) * float(los_cutoff_comoving_mpc)) ** 2
        )
    )


def nonsingular_projection_geometry(
    projected_radius: np.ndarray,
    line_of_sight_max: float | np.ndarray,
    *,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 3D radii and weights for ``2 integral rho dl``.

    The substitution ``l = R sinh(t)`` removes the Abel singularity at
    ``r = R``.  The returned arrays have shape ``(n_radius, n_points)`` and
    satisfy ``projected = sum(rho(radius_3d) * weights, axis=1)``.
    """

    radius = np.atleast_1d(np.asarray(projected_radius, dtype=np.float64))
    los_max = np.broadcast_to(
        np.asarray(line_of_sight_max, dtype=np.float64), radius.shape
    )
    if radius.ndim != 1 or np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("Projected radii must be finite, positive, and 1D.")
    if np.any(~np.isfinite(los_max)) or np.any(los_max < 0.0):
        raise ValueError("LOS limits must be finite and non-negative.")
    if int(n_points) < 8:
        raise ValueError("Nonsingular projection requires at least eight points.")

    nodes, base_weights = leggauss(int(n_points))
    tmax = np.arcsinh(los_max / radius)
    t = 0.5 * (nodes[None, :] + 1.0) * tmax[:, None]
    radius_3d = radius[:, None] * np.cosh(t)
    # 2 * dt/dx = tmax, while dl/dt = R cosh(t).
    weights = tmax[:, None] * base_weights[None, :] * radius[:, None] * np.cosh(t)
    return radius_3d, weights


def project_log_table_nonsingular(
    projected_radius: np.ndarray,
    line_of_sight_max: float | np.ndarray,
    table_radius: np.ndarray,
    table_values: np.ndarray,
    *,
    n_points: int,
) -> np.ndarray:
    """Project one positive log-interpolated 3D table without extrapolation."""

    radius = np.asarray(table_radius, dtype=np.float64)
    values = np.asarray(table_values, dtype=np.float64)
    if (
        radius.ndim != 1
        or values.shape != radius.shape
        or radius.size < 2
        or np.any(np.diff(radius) <= 0.0)
        or np.any(radius <= 0.0)
        or np.any(values <= 0.0)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("Projection tables must be finite, positive 1D arrays.")

    query, weights = nonsingular_projection_geometry(
        projected_radius, line_of_sight_max, n_points=n_points
    )
    active = np.any(weights != 0.0, axis=1)
    if np.any(query[active] < radius[0] * (1.0 - 1.0e-12)) or np.any(
        query[active] > radius[-1] * (1.0 + 1.0e-12)
    ):
        raise ValueError(
            "Nonsingular projection query leaves the explicit 3D table; "
            "extend the table or shorten the LOS instead of clamping."
        )
    query = np.clip(query, radius[0], radius[-1])
    sampled = np.exp(np.interp(np.log(query), np.log(radius), np.log(values)))
    projected = np.sum(sampled * weights, axis=1)
    projected[~active] = 0.0
    return projected


def project_callable_nonsingular(
    projected_radius: np.ndarray,
    line_of_sight_max: float | np.ndarray,
    evaluator: Callable[[np.ndarray], np.ndarray],
    *,
    n_points: int,
) -> np.ndarray:
    """Project a callable 3D profile on the same nonsingular geometry."""

    query, weights = nonsingular_projection_geometry(
        projected_radius, line_of_sight_max, n_points=n_points
    )
    sampled = np.asarray(evaluator(query.ravel()), dtype=np.float64).reshape(
        query.shape
    )
    if np.any(~np.isfinite(sampled)) or np.any(sampled < 0.0):
        raise ValueError("The 3D profile returned non-finite or negative values.")
    return np.sum(sampled * weights, axis=1)


def _interp2(
    values: np.ndarray,
    redshifts: np.ndarray,
    masses: np.ndarray,
    redshift: float,
    mass: float,
) -> float:
    interpolator = RegularGridInterpolator(
        (redshifts, np.log(masses)),
        np.asarray(values, dtype=np.float64),
        bounds_error=True,
    )
    return float(interpolator([[redshift, np.log(mass)]])[0])


def _interp3_positive(
    values: np.ndarray,
    radii: np.ndarray,
    redshifts: np.ndarray,
    masses: np.ndarray,
    query_radius: np.ndarray,
    redshift: float,
    mass: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    floor = max(float(np.nanmax(values)) * 1.0e-300, 1.0e-300)
    interpolator = RegularGridInterpolator(
        (np.log(radii), redshifts, np.log(masses)),
        np.log(np.maximum(values, floor)),
        bounds_error=True,
    )
    query_radius = np.asarray(query_radius, dtype=np.float64)
    points = np.column_stack(
        (
            np.log(query_radius),
            np.full(query_radius.size, float(redshift)),
            np.full(query_radius.size, np.log(float(mass))),
        )
    )
    return np.exp(interpolator(points))


def _target_radial_table(
    values: np.ndarray,
    radii: np.ndarray,
    redshifts: np.ndarray,
    masses: np.ndarray,
    redshift: float,
    mass: float,
) -> np.ndarray:
    """Interpolate only mass/redshift while retaining every radial node."""

    return _interp3_positive(
        values,
        radii,
        redshifts,
        masses,
        radii,
        redshift,
        mass,
    )


def _kappa_from_godmax_sigma(
    sigma_msun_mpc2: np.ndarray,
    *,
    redshift: float,
    projected,
    profiles,
) -> np.ndarray:
    a = 1.0 / (1.0 + float(redshift))
    wkappa = float(
        np.interp(
            redshift,
            np.asarray(projected.z_array, dtype=np.float64),
            np.asarray(projected.Wkappa_array_for_map, dtype=np.float64),
        )
    )
    # GODMAX's raw projection has units Msun h / Mpc^2; the comparison-facing
    # sigma has already been multiplied by h into Msun / Mpc^2.
    return (
        wkappa
        * a**2
        * (np.asarray(sigma_msun_mpc2, dtype=np.float64) / float(profiles.h))
        / float(profiles.rho_m_bar)
    )


def _project_godmax_shared_los(
    record: Mapping[str, Any],
    projected,
    projected_radius_physical_hmpc: np.ndarray,
    *,
    redshift: float,
    mass_hmsun: float,
    h: float,
    n_points: int,
    truncate_to_table: bool = False,
) -> dict[str, np.ndarray]:
    """Project one extended GODMAX radial-table record on the common LOS."""

    profiles = record["profiles"]
    radius = record["radius"]
    redshifts = record["redshift"]
    masses = record["mass"]
    a = 1.0 / (1.0 + float(redshift))
    pressure = _target_radial_table(
        profiles.Pe_mat_physical,
        radius,
        redshifts,
        masses,
        redshift,
        mass_hmsun,
    )
    density = _target_radial_table(
        record["rho_physical"],
        radius,
        redshifts,
        masses,
        redshift,
        mass_hmsun,
    )
    physical_radius = a * radius
    los_max = np.full(
        np.asarray(projected_radius_physical_hmpc).shape,
        a * float(h) * BARYONFORGE_LOS_CUTOFF_COMOVING_MPC,
        dtype=np.float64,
    )
    if truncate_to_table:
        table_los = np.sqrt(
            np.maximum(
                physical_radius[-1] ** 2
                - np.asarray(projected_radius_physical_hmpc) ** 2,
                0.0,
            )
        )
        los_max = np.minimum(los_max, table_los)
    y = float(projected.const_coeff) * project_log_table_nonsingular(
        projected_radius_physical_hmpc,
        los_max,
        physical_radius,
        pressure,
        n_points=n_points,
    )
    sigma = float(h) * project_log_table_nonsingular(
        projected_radius_physical_hmpc,
        los_max,
        physical_radius,
        density,
        n_points=n_points,
    )
    return {
        "y": y,
        "sigma_matter_physical_Msun_Mpc2": sigma,
        "kappa_cmb": _kappa_from_godmax_sigma(
            sigma,
            redshift=redshift,
            projected=projected,
            profiles=profiles,
        ),
    }


def _build_extended_godmax(
    native_profiles,
    context: Mapping[str, Any],
    *,
    rmax_comoving_hmpc: float,
    nr: int,
):
    """Rebuild only an in-memory extended radial table with the same model."""

    from base_class import base_class

    sim = copy.deepcopy(context["sim_params"])
    halo = copy.deepcopy(context["halo_params"])
    analysis = copy.deepcopy(context["analysis"])
    other = copy.deepcopy(context["other_params"])
    halo["rmax"] = float(rmax_comoving_hmpc)
    halo["nr"] = int(nr)
    base = base_class(sim, halo, analysis, other)
    extended = type(native_profiles)(
        sim,
        halo,
        analysis,
        other,
        base_class_obj=base,
    )
    return extended, {
        "sim_params": sim,
        "halo_params": halo,
        "analysis": analysis,
        "other_params": other,
        "profiles_class_fqname": (
            f"{type(extended).__module__}.{type(extended).__qualname__}"
        ),
    }


def _empty_profiles(n_nodes: int, n_radius: int) -> dict[str, dict[str, np.ndarray]]:
    return {
        variant: {
            field: np.empty((n_nodes, n_radius), dtype=np.float64) for field in FIELDS
        }
        for variant in VARIANTS
    }


def _relative(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if np.any(reference <= 0.0):
        raise ValueError("A positive reference is required for projection ratios.")
    return candidate / reference - 1.0


def _save_figure(
    nodes: Sequence[Mapping[str, float]],
    scaled_radius: np.ndarray,
    profiles: Mapping[str, Mapping[str, np.ndarray]],
    figure_dir: Path,
) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    reference = profiles["baryonforge_dense_nonsingular"]
    colors = {
        "godmax_legacy_clamped": "#d62728",
        "godmax_table_truncated_nonsingular": "#9467bd",
        "godmax_extended_nonsingular": "#1f77b4",
        "baryonforge_native": "#ff7f0e",
    }
    labels = {
        "godmax_legacy_clamped": "GODMAX legacy clamp",
        "godmax_table_truncated_nonsingular": "GODMAX 12 hMpc trunc.",
        "godmax_extended_nonsingular": "GODMAX 70 hMpc shared LOS",
        "baryonforge_native": "BaryonForge native",
    }
    redshifts = sorted({float(node["z"]) for node in nodes})
    linestyles = dict(zip(redshifts, ("-", "--", ":")))
    masses = sorted({float(node["mass_hMsun"]) for node in nodes})
    fields = ("y", "kappa_cmb")
    field_labels = (r"Compton-$y$", r"CMB $\kappa$")

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True)
    for column, mass in enumerate(masses):
        indices = [i for i, node in enumerate(nodes) if node["mass_hMsun"] == mass]
        for row, (field, field_label) in enumerate(zip(fields, field_labels)):
            axis = axes[row, column]
            plotted = []
            for index in indices:
                redshift = float(nodes[index]["z"])
                for variant, color in colors.items():
                    ratio = profiles[variant][field][index] / reference[field][index]
                    axis.plot(
                        scaled_radius,
                        ratio,
                        color=color,
                        linestyle=linestyles[redshift],
                        linewidth=1.4,
                    )
                    plotted.extend(ratio.tolist())
            axis.axhline(1.0, color="black", linewidth=0.8)
            axis.set_xscale("log")
            finite = np.asarray(plotted)[np.isfinite(plotted)]
            lower = max(0.05, 0.94 * float(np.min(finite)))
            upper = 1.04 * float(np.max(finite))
            axis.set_ylim(lower, upper)
            axis.grid(alpha=0.2)
            axis.set_title(rf"$M_{{200c}}={mass:.0e}\,h^{{-1}}M_\odot$: {field_label}")
            if column == 0:
                axis.set_ylabel("projection / dense BaryonForge")
            if row == 1:
                axis.set_xlabel(r"transverse radius [$R_{200c}$]")

    variant_handles = [
        Line2D([0], [0], color=color, label=labels[name], linewidth=2.0)
        for name, color in colors.items()
    ]
    redshift_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=linestyles[z],
            label=f"z={z:.2f}",
        )
        for z in redshifts
    ]
    fig.legend(
        handles=variant_handles + redshift_handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Projection-domain diagnosis on nine frozen profile nodes")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96))

    outputs = []
    for extension, kwargs in (("png", {"dpi": 180}), ("pdf", {})):
        output = figure_dir / f"projection_convergence.{extension}"
        temporary = (
            figure_dir / f".projection_convergence.tmp.{os.getpid()}.{extension}"
        )
        fig.savefig(temporary, bbox_inches="tight", **kwargs)
        os.replace(temporary, output)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def _save_resolution_figure(
    godmax_grid_errors: Mapping[str, Mapping[str, float]],
    godmax_grid_support: Mapping[str, bool],
    godmax_point_errors: Mapping[str, Mapping[str, float]],
    baryonforge_point_errors: Mapping[str, Mapping[str, float]],
    figure_dir: Path,
) -> list[str]:
    """Save the concrete radial/projection resolution recommendation."""

    fields = ("y", "kappa_cmb")
    labels = {"y": r"Compton-$y$", "kappa_cmb": r"CMB $\kappa$"}
    colors = {"y": "#d62728", "kappa_cmb": "#1f77b4"}
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    grid_names = list(godmax_grid_errors)
    for field in fields:
        axes[0].plot(
            np.arange(len(grid_names)),
            [max(godmax_grid_errors[name][field], 1.0e-16) for name in grid_names],
            marker="o",
            color=colors[field],
            label=labels[field],
        )
    axes[0].axhline(
        PROFILE_RESOLUTION_TOLERANCE,
        color="black",
        linestyle="--",
        label="fixed 2% target",
    )
    grid_labels = [
        name if godmax_grid_support[name] else f"{name}\n(LOS-short)"
        for name in grid_names
    ]
    axes[0].set_xticks(np.arange(len(grid_names)), grid_labels, rotation=20)
    axes[0].set_title("GODMAX radial table")

    point_names = list(godmax_point_errors)
    point_x = np.asarray([int(name) for name in point_names])
    for field in fields:
        axes[1].plot(
            point_x,
            [max(godmax_point_errors[name][field], 1.0e-16) for name in point_names],
            marker="o",
            color=colors[field],
            label=labels[field],
        )
    axes[1].axhline(
        DENSE_CONVERGENCE_TOLERANCE,
        color="black",
        linestyle="--",
        label="fixed $10^{-3}$ target",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("GODMAX nonsingular quadrature")
    axes[1].set_xlabel("Gauss-Legendre points")

    bfg_names = list(baryonforge_point_errors)
    bfg_x = np.asarray([int(name) for name in bfg_names])
    for field in fields:
        axes[2].plot(
            bfg_x,
            [max(baryonforge_point_errors[name][field], 1.0e-16) for name in bfg_names],
            marker="o",
            color=colors[field],
            label=labels[field],
        )
    axes[2].axhline(
        PROFILE_RESOLUTION_TOLERANCE,
        color="black",
        linestyle="--",
        label="fixed 2% target",
    )
    axes[2].set_xscale("log", base=2)
    axes[2].set_title("BaryonForge native projector")
    axes[2].set_xlabel("points per decade")

    for axis in axes:
        axis.set_yscale("log")
        axis.set_ylabel("max relative error through $5R_{200c}$")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Projection-resolution convergence on nine frozen nodes")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    outputs = []
    for extension, kwargs in (("png", {"dpi": 180}), ("pdf", {})):
        output = figure_dir / f"projection_resolution_scan.{extension}"
        temporary = (
            figure_dir / f".projection_resolution_scan.tmp.{os.getpid()}.{extension}"
        )
        fig.savefig(temporary, bbox_inches="tight", **kwargs)
        os.replace(temporary, output)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def _provenance(
    config: Mapping[str, Any],
    native_context: Mapping[str, Any],
    extended_contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_manifest = comparison_source_manifest()
    for path in (
        Path(__file__).with_name("compare_profiles.py"),
        Path(__file__).resolve(),
    ):
        source_manifest[path.relative_to(WORKSPACE_ROOT).as_posix()] = sha256_file(path)
    source_manifest = dict(sorted(source_manifest.items()))
    runtime = runtime_version_manifest()
    config_path = resolve_path(config["_config_path"])
    catalog_path = resolve_path(config["catalog"]["output_h5"], config_path)
    godmax_params = resolve_path(config["profiles"]["godmax_params"], config_path)
    baryonforge_params = resolve_path(
        config["profiles"]["baryonforge_params"], config_path
    )
    return {
        "schema": SCHEMA,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "godmax_params_path": str(godmax_params),
        "godmax_params_sha256": sha256_file(godmax_params),
        "baryonforge_params_path": str(baryonforge_params),
        "baryonforge_params_sha256": sha256_file(baryonforge_params),
        "profile_integration_contract": profile_integration_contract(config),
        "native_effective_dictionaries": jsonable(native_context),
        "native_effective_dictionaries_sha256": sha256_json(native_context),
        "extended_effective_dictionaries": jsonable(extended_contexts),
        "extended_effective_dictionaries_sha256": sha256_json(extended_contexts),
        "source_manifest": source_manifest,
        "source_manifest_sha256": sha256_json(source_manifest),
        "runtime_versions": runtime,
        "runtime_manifest_sha256": sha256_json(runtime),
        "godmax_git_sha": git_revision(REPO_ROOT),
        "godmax_git_dirty": git_is_dirty(REPO_ROOT),
        "baryonforge_git_sha": git_revision(WORKSPACE_ROOT / "BaryonForge"),
        "baryonforge_git_dirty": git_is_dirty(WORKSPACE_ROOT / "BaryonForge"),
        "variant_contract": {
            "variants": list(VARIANTS),
            "extended_godmax_rmax_comoving_hMpc": (EXTENDED_GODMAX_RMAX_COMOVING_HMPC),
            "extended_godmax_nr": EXTENDED_GODMAX_NR,
            "matched_godmax_projection_points": MATCHED_GODMAX_PROJECTION_POINTS,
            "shared_los_cutoff_comoving_Mpc": (BARYONFORGE_LOS_CUTOFF_COMOVING_MPC),
            "dense_reference_points": DENSE_REFERENCE_POINTS,
            "dense_check_points": DENSE_CHECK_POINTS,
            "dense_convergence_tolerance": DENSE_CONVERGENCE_TOLERANCE,
            "profile_resolution_tolerance": PROFILE_RESOLUTION_TOLERANCE,
            "godmax_grid_scan": [list(item) for item in GODMAX_GRID_SCAN],
            "godmax_grid_reference": list(GODMAX_GRID_REFERENCE),
            "godmax_projection_point_scan": list(GODMAX_PROJECTION_POINT_SCAN),
            "baryonforge_points_per_decade_scan": list(
                BARYONFORGE_POINTS_PER_DECADE_SCAN
            ),
            "quadrature": "Gauss-Legendre after l=R*sinh(t)",
            "out_of_table_policy": "raise; never endpoint-clamp",
        },
    }


def run(
    config: Mapping[str, Any],
    output_h5: Path,
    summary_json: Path,
    figure_dir: Path,
) -> dict[str, Any]:
    """Execute and persist the five-way nine-node projection diagnostic."""

    crosswalk = validate_parameter_crosswalk(config)
    if not crosswalk["ok"]:
        raise ValueError(f"Parameter crosswalk failed: {crosswalk['failed']}")

    # Imports occur only in the executable path so pure helper tests do not
    # initialize JAX or require the full scientific stack.
    from compare_profiles import _godmax_profiles
    from paint_baryonforge import (
        build_ccl_cosmology,
        build_direct_models,
    )

    native, projected, native_context = _godmax_profiles(config)
    extended_models: dict[str, Any] = {}
    extended_contexts: dict[str, Mapping[str, Any]] = {}
    for rmax, nr in GODMAX_GRID_SCAN:
        key = f"rmax{rmax:g}_nr{nr}"
        model, context = _build_extended_godmax(
            native,
            native_context,
            rmax_comoving_hmpc=rmax,
            nr=nr,
        )
        radius = np.asarray(model.r_array, dtype=np.float64)
        scale_factor = np.asarray(model.scale_fac_a_array, dtype=np.float64)
        extended_models[key] = {
            "profiles": model,
            "radius": radius,
            "redshift": np.asarray(model.z_array, dtype=np.float64),
            "mass": np.asarray(model.M_array, dtype=np.float64),
            "rho_physical": np.asarray(model.rho_dmb_mat, dtype=np.float64)
            / (scale_factor[None, :, None] ** 3),
            "rmax_comoving_hMpc": float(rmax),
            "nr": int(nr),
        }
        extended_contexts[key] = context
    main_extended_key = (
        f"rmax{EXTENDED_GODMAX_RMAX_COMOVING_HMPC:g}_nr{EXTENDED_GODMAX_NR}"
    )
    bparams = load_yaml(config["profiles"]["baryonforge_params"])
    cosmo = build_ccl_cosmology(bparams["cosmology"])
    bmodels = build_direct_models(bparams, cosmo)
    native_points_per_decade = int(bparams["numerics"]["n_per_decade_proj"])
    baryonforge_models = {}
    for points_per_decade in BARYONFORGE_POINTS_PER_DECADE_SCAN:
        if points_per_decade == native_points_per_decade:
            baryonforge_models[points_per_decade] = bmodels
        else:
            candidate_params = copy.deepcopy(bparams)
            candidate_params["numerics"]["n_per_decade_proj"] = points_per_decade
            baryonforge_models[points_per_decade] = build_direct_models(
                candidate_params, cosmo
            )

    h = float(bparams["cosmology"]["h"])
    mass_targets = np.asarray(config["profiles"]["masses_hMsun"], dtype=np.float64)
    redshift_targets = np.asarray(config["profiles"]["redshifts"], dtype=np.float64)
    scaled_radius = np.geomspace(
        float(config["profiles"]["radius_min_R200c"]),
        float(config["profiles"]["radius_max_R200c"]),
        int(config["profiles"]["n_radius"]),
    )
    nodes = [
        {"mass_hMsun": float(mass), "z": float(redshift)}
        for mass in mass_targets
        for redshift in redshift_targets
    ]
    profiles = _empty_profiles(len(nodes), scaled_radius.size)
    dense_convergence = {
        "baryonforge_dense_nonsingular": {
            field: np.empty((len(nodes), scaled_radius.size), dtype=np.float64)
            for field in FIELDS
        },
    }
    godmax_grid_scan = {
        key: {
            field: np.empty((len(nodes), scaled_radius.size), dtype=np.float64)
            for field in FIELDS
        }
        for key in extended_models
    }
    godmax_point_scan = {
        str(points): {
            field: np.empty((len(nodes), scaled_radius.size), dtype=np.float64)
            for field in FIELDS
        }
        for points in GODMAX_PROJECTION_POINT_SCAN
    }
    baryonforge_point_scan = {
        str(points): {
            field: np.empty((len(nodes), scaled_radius.size), dtype=np.float64)
            for field in FIELDS
        }
        for points in BARYONFORGE_POINTS_PER_DECADE_SCAN
    }

    native_r = np.asarray(native.r_array, dtype=np.float64)
    native_z = np.asarray(native.z_array, dtype=np.float64)
    native_m = np.asarray(native.M_array, dtype=np.float64)
    projected_r = np.asarray(projected.rp_array, dtype=np.float64)
    native_rho_physical = np.asarray(native.rho_dmb_mat, dtype=np.float64) / (
        np.asarray(native.scale_fac_a_array, dtype=np.float64)[None, :, None] ** 3
    )

    maximum_transverse_hmpc = []
    support_records = []
    for node_index, node in enumerate(nodes):
        mass = float(node["mass_hMsun"])
        redshift = float(node["z"])
        a = 1.0 / (1.0 + redshift)
        r200 = _interp2(native.r200c_mat, native_z, native_m, redshift, mass)
        transverse_comoving_hmpc = scaled_radius * r200
        transverse_physical_hmpc = a * transverse_comoving_hmpc
        transverse_bfg_comoving_mpc = transverse_comoving_hmpc / h
        maximum_transverse_hmpc.extend(transverse_comoving_hmpc.tolist())
        node.update(
            {
                "R200c_comoving_hMpc": r200,
                "native_table_max_comoving_hMpc": float(native_r[-1]),
                "native_table_max_physical_hMpc": float(a * native_r[-1]),
            }
        )

        # 1. Exact legacy GODMAX projected table, including the native clamp.
        legacy_y = _interp3_positive(
            projected.y2D_mat_physical,
            projected_r,
            native_z,
            native_m,
            transverse_physical_hmpc,
            redshift,
            mass,
        )
        legacy_sigma = h * _interp3_positive(
            projected.rhom2D_mat_physical,
            projected_r,
            native_z,
            native_m,
            transverse_physical_hmpc,
            redshift,
            mass,
        )
        profiles["godmax_legacy_clamped"]["y"][node_index] = legacy_y
        profiles["godmax_legacy_clamped"]["sigma_matter_physical_Msun_Mpc2"][
            node_index
        ] = legacy_sigma
        profiles["godmax_legacy_clamped"]["kappa_cmb"][node_index] = (
            _kappa_from_godmax_sigma(
                legacy_sigma,
                redshift=redshift,
                projected=projected,
                profiles=native,
            )
        )

        # 2. Same finite 3D table, but no endpoint extrapolation and no Abel
        # singularity.  This intentionally exposes the missing physical tail.
        native_pe = _target_radial_table(
            native.Pe_mat_physical,
            native_r,
            native_z,
            native_m,
            redshift,
            mass,
        )
        native_rho = _target_radial_table(
            native_rho_physical,
            native_r,
            native_z,
            native_m,
            redshift,
            mass,
        )
        native_physical_radius = a * native_r
        table_los = np.sqrt(
            np.maximum(
                native_physical_radius[-1] ** 2 - transverse_physical_hmpc**2,
                0.0,
            )
        )
        truncated_y = float(projected.const_coeff) * project_log_table_nonsingular(
            transverse_physical_hmpc,
            table_los,
            native_physical_radius,
            native_pe,
            n_points=DENSE_REFERENCE_POINTS,
        )
        truncated_sigma = h * project_log_table_nonsingular(
            transverse_physical_hmpc,
            table_los,
            native_physical_radius,
            native_rho,
            n_points=DENSE_REFERENCE_POINTS,
        )
        profiles["godmax_table_truncated_nonsingular"]["y"][node_index] = truncated_y
        profiles["godmax_table_truncated_nonsingular"][
            "sigma_matter_physical_Msun_Mpc2"
        ][node_index] = truncated_sigma
        profiles["godmax_table_truncated_nonsingular"]["kappa_cmb"][node_index] = (
            _kappa_from_godmax_sigma(
                truncated_sigma,
                redshift=redshift,
                projected=projected,
                profiles=native,
            )
        )

        # 3. Extended GODMAX table with exactly BaryonForge's comoving LOS.
        main_extended = extended_models[main_extended_key]
        extended_physical_radius = a * main_extended["radius"]
        common_los_godmax = a * h * BARYONFORGE_LOS_CUTOFF_COMOVING_MPC
        required_radius = float(
            np.max(np.sqrt(transverse_physical_hmpc**2 + common_los_godmax**2))
        )
        support_records.append(
            {
                "mass_hMsun": mass,
                "z": redshift,
                "required_max_physical_hMpc": required_radius,
                "available_max_physical_hMpc": float(extended_physical_radius[-1]),
                "ok": bool(required_radius <= extended_physical_radius[-1]),
            }
        )
        for grid_key, record in extended_models.items():
            result = _project_godmax_shared_los(
                record,
                projected,
                transverse_physical_hmpc,
                redshift=redshift,
                mass_hmsun=mass,
                h=h,
                n_points=DENSE_REFERENCE_POINTS,
                truncate_to_table=True,
            )
            for field in FIELDS:
                godmax_grid_scan[grid_key][field][node_index] = result[field]
        for n_points in GODMAX_PROJECTION_POINT_SCAN:
            result = _project_godmax_shared_los(
                main_extended,
                projected,
                transverse_physical_hmpc,
                redshift=redshift,
                mass_hmsun=mass,
                h=h,
                n_points=n_points,
            )
            for field in FIELDS:
                godmax_point_scan[str(n_points)][field][node_index] = result[field]
        for field in FIELDS:
            profiles["godmax_extended_nonsingular"][field][node_index] = (
                godmax_point_scan[str(DENSE_REFERENCE_POINTS)][field][node_index]
            )

        # 4. Native BaryonForge projection on the identical transverse radii.
        mass_physical_msun = mass / h
        native_bfg_y = np.asarray(
            bmodels["y_direct"].projected(
                cosmo,
                transverse_bfg_comoving_mpc,
                mass_physical_msun,
                a,
            ),
            dtype=np.float64,
        )
        native_bfg_sigma = (
            np.asarray(
                bmodels["matter_direct"].projected(
                    cosmo,
                    transverse_bfg_comoving_mpc,
                    mass_physical_msun,
                    a,
                ),
                dtype=np.float64,
            )
            / a**2
        )
        native_bfg_kappa = np.asarray(
            bmodels["kappa_direct"].projected(
                cosmo,
                transverse_bfg_comoving_mpc,
                mass_physical_msun,
                a,
            ),
            dtype=np.float64,
        )
        profiles["baryonforge_native"]["y"][node_index] = native_bfg_y
        profiles["baryonforge_native"]["sigma_matter_physical_Msun_Mpc2"][
            node_index
        ] = native_bfg_sigma
        profiles["baryonforge_native"]["kappa_cmb"][node_index] = native_bfg_kappa

        # 5. Dense nonsingular projection of the same BaryonForge 3D profiles.
        sigma_critical = float(
            cosmo.sigma_critical(
                a_lens=a,
                a_source=1.0 / (1.0 + float(bparams["adapter"]["cmb_source_redshift"])),
            )
        )
        for points_per_decade, candidate_models in baryonforge_models.items():
            if points_per_decade == native_points_per_decade:
                candidate_y = native_bfg_y
                candidate_sigma = native_bfg_sigma
            else:
                candidate_y = np.asarray(
                    candidate_models["y_direct"].projected(
                        cosmo,
                        transverse_bfg_comoving_mpc,
                        mass_physical_msun,
                        a,
                    ),
                    dtype=np.float64,
                )
                candidate_sigma = (
                    np.asarray(
                        candidate_models["matter_direct"].projected(
                            cosmo,
                            transverse_bfg_comoving_mpc,
                            mass_physical_msun,
                            a,
                        ),
                        dtype=np.float64,
                    )
                    / a**2
                )
            destination = baryonforge_point_scan[str(points_per_decade)]
            destination["y"][node_index] = candidate_y
            destination["sigma_matter_physical_Msun_Mpc2"][node_index] = candidate_sigma
            destination["kappa_cmb"][node_index] = candidate_sigma / sigma_critical
        for n_points, destination in (
            (
                DENSE_REFERENCE_POINTS,
                profiles["baryonforge_dense_nonsingular"],
            ),
            (
                DENSE_CHECK_POINTS,
                dense_convergence["baryonforge_dense_nonsingular"],
            ),
        ):

            def y_evaluator(radius: np.ndarray) -> np.ndarray:
                return np.asarray(
                    bmodels["y_direct"].real(cosmo, radius, mass_physical_msun, a),
                    dtype=np.float64,
                )

            def matter_evaluator(radius: np.ndarray) -> np.ndarray:
                return np.asarray(
                    bmodels["matter_direct"].real(cosmo, radius, mass_physical_msun, a),
                    dtype=np.float64,
                )

            dense_y = a * project_callable_nonsingular(
                transverse_bfg_comoving_mpc,
                BARYONFORGE_LOS_CUTOFF_COMOVING_MPC,
                y_evaluator,
                n_points=n_points,
            )
            dense_sigma = (
                project_callable_nonsingular(
                    transverse_bfg_comoving_mpc,
                    BARYONFORGE_LOS_CUTOFF_COMOVING_MPC,
                    matter_evaluator,
                    n_points=n_points,
                )
                / a**2
            )
            destination["y"][node_index] = dense_y
            destination["sigma_matter_physical_Msun_Mpc2"][node_index] = dense_sigma
            destination["kappa_cmb"][node_index] = dense_sigma / sigma_critical

    required_rmax = required_extended_rmax_hmpc(
        np.asarray(maximum_transverse_hmpc),
        los_cutoff_comoving_mpc=BARYONFORGE_LOS_CUTOFF_COMOVING_MPC,
        h=h,
    )
    reference = profiles["baryonforge_dense_nonsingular"]
    comparison_metrics = {
        variant: {
            field: {
                "max_abs_relative_difference": float(
                    np.max(np.abs(_relative(values[field], reference[field])))
                ),
                "rms_log_ratio": float(
                    np.sqrt(np.mean(np.log(values[field] / reference[field]) ** 2))
                ),
                "edge_ratio_min": float(
                    np.min(values[field][:, -1] / reference[field][:, -1])
                ),
                "edge_ratio_max": float(
                    np.max(values[field][:, -1] / reference[field][:, -1])
                ),
            }
            for field in FIELDS
        }
        for variant, values in profiles.items()
    }
    godmax_grid_reference_key = (
        f"rmax{GODMAX_GRID_REFERENCE[0]:g}_nr{GODMAX_GRID_REFERENCE[1]}"
    )
    godmax_grid_errors = {
        key: {
            field: float(
                np.max(
                    np.abs(
                        _relative(
                            values[field],
                            godmax_grid_scan[godmax_grid_reference_key][field],
                        )
                    )
                )
            )
            for field in FIELDS
        }
        for key, values in godmax_grid_scan.items()
    }
    godmax_point_reference = godmax_point_scan[str(DENSE_REFERENCE_POINTS)]
    godmax_point_errors = {
        key: {
            field: float(
                np.max(np.abs(_relative(values[field], godmax_point_reference[field])))
            )
            for field in FIELDS
        }
        for key, values in godmax_point_scan.items()
    }
    baryonforge_point_errors = {
        key: {
            field: float(np.max(np.abs(_relative(values[field], reference[field]))))
            for field in FIELDS
        }
        for key, values in baryonforge_point_scan.items()
    }
    dense_baryonforge_quadrature_errors = {
        field: float(
            np.max(
                np.abs(
                    _relative(
                        dense_convergence["baryonforge_dense_nonsingular"][field],
                        reference[field],
                    )
                )
            )
        )
        for field in FIELDS
    }

    def probe_maximum(values: Mapping[str, float]) -> float:
        return max(float(values[field]) for field in ("y", "kappa_cmb"))

    godmax_grid_support = {
        key: bool(record["rmax_comoving_hMpc"] >= required_rmax)
        for key, record in extended_models.items()
    }
    minimum_scanned_complete_grid = next(
        (
            key
            for key in godmax_grid_scan
            if godmax_grid_support[key]
            and probe_maximum(godmax_grid_errors[key]) <= PROFILE_RESOLUTION_TOLERANCE
        ),
        None,
    )
    configured_grid = (
        main_extended_key
        if godmax_grid_support[main_extended_key]
        and probe_maximum(godmax_grid_errors[main_extended_key])
        <= PROFILE_RESOLUTION_TOLERANCE
        else None
    )
    minimum_scanned_godmax_points = next(
        (
            int(key)
            for key in godmax_point_scan
            if int(key) < DENSE_REFERENCE_POINTS
            and probe_maximum(godmax_point_errors[key]) <= DENSE_CONVERGENCE_TOLERANCE
        ),
        None,
    )
    configured_godmax_points = (
        MATCHED_GODMAX_PROJECTION_POINTS
        if probe_maximum(godmax_point_errors[str(MATCHED_GODMAX_PROJECTION_POINTS)])
        <= DENSE_CONVERGENCE_TOLERANCE
        else None
    )
    recommended_baryonforge_points_per_decade = next(
        (
            int(key)
            for key in baryonforge_point_scan
            if probe_maximum(baryonforge_point_errors[key])
            <= PROFILE_RESOLUTION_TOLERANCE
        ),
        None,
    )
    finite_nonnegative = all(
        np.all(np.isfinite(values[field])) and np.all(values[field] >= 0.0)
        for values in profiles.values()
        for field in FIELDS
    )
    acceptance = {
        "extended_grid_covers_shared_los": bool(
            required_rmax <= EXTENDED_GODMAX_RMAX_COMOVING_HMPC
            and all(item["ok"] for item in support_records)
        ),
        "all_profiles_finite_nonnegative": bool(finite_nonnegative),
        "godmax_main_grid_vs_80hMpc_192": bool(
            probe_maximum(godmax_grid_errors[main_extended_key])
            <= PROFILE_RESOLUTION_TOLERANCE
        ),
        "baryonforge_dense_256_vs_512": bool(
            probe_maximum(dense_baryonforge_quadrature_errors)
            <= DENSE_CONVERGENCE_TOLERANCE
        ),
        "godmax_projection_recommendation_found": (
            configured_godmax_points is not None
        ),
        "baryonforge_projection_recommendation_found": (
            recommended_baryonforge_points_per_decade is not None
        ),
    }
    provenance = _provenance(config, native_context, extended_contexts)
    report = {
        "schema": SCHEMA,
        "status": "bounded_profile_projection_diagnostic_not_map_production",
        "node_count": len(nodes),
        "nodes": nodes,
        "scaled_radius_R200c": scaled_radius.tolist(),
        "fields": list(FIELDS),
        "units": {
            "y": "dimensionless Compton-y",
            "sigma_matter_physical_Msun_Mpc2": "physical Msun / Mpc^2",
            "kappa_cmb": "dimensionless halo-only CMB convergence",
        },
        "required_extended_rmax_comoving_hMpc": required_rmax,
        "available_extended_rmax_comoving_hMpc": (EXTENDED_GODMAX_RMAX_COMOVING_HMPC),
        "support_records": support_records,
        "comparison_reference": "baryonforge_dense_nonsingular",
        "comparison_metrics": comparison_metrics,
        "resolution_convergence": {
            "profile_resolution_fixed_tolerance": PROFILE_RESOLUTION_TOLERANCE,
            "godmax_grid_reference": godmax_grid_reference_key,
            "godmax_grid_covers_full_shared_los": godmax_grid_support,
            "godmax_grid_max_abs_relative_change": godmax_grid_errors,
            "godmax_projection_reference_points": DENSE_REFERENCE_POINTS,
            "godmax_projection_point_max_abs_relative_change": (godmax_point_errors),
            "baryonforge_dense_candidate_points": DENSE_CHECK_POINTS,
            "reference_points": DENSE_REFERENCE_POINTS,
            "dense_fixed_tolerance": DENSE_CONVERGENCE_TOLERANCE,
            "baryonforge_dense_256_vs_512_max_abs_relative_change": (
                dense_baryonforge_quadrature_errors
            ),
            "baryonforge_points_per_decade_max_abs_relative_difference": (
                baryonforge_point_errors
            ),
        },
        "recommendation": {
            "godmax_grid": configured_grid,
            "godmax_projection_points": configured_godmax_points,
            "minimum_scanned_complete_godmax_grid": (minimum_scanned_complete_grid),
            "minimum_scanned_godmax_projection_points": (minimum_scanned_godmax_points),
            "baryonforge_n_per_decade_proj": (
                recommended_baryonforge_points_per_decade
            ),
            "selection_rule": (
                "configured GODMAX 70 hMpc/128-node grid must cover the full "
                "shared LOS and pass the fixed 2% through-5R target; configured "
                "128-point Gauss/cosh quadrature must pass the fixed 1e-3 "
                "dense-reference target. Minimum passing scanned choices are "
                "reported separately. BaryonForge points/decade is accepted "
                "only if the native projector passes 2% through 5R."
            ),
        },
        "acceptance": acceptance,
        "ok": all(acceptance.values()),
        "provenance": provenance,
    }
    report["figures"] = _save_figure(
        nodes, scaled_radius, profiles, figure_dir
    ) + _save_resolution_figure(
        godmax_grid_errors,
        godmax_grid_support,
        godmax_point_errors,
        baryonforge_point_errors,
        figure_dir,
    )
    report["figure_sha256"] = {path: sha256_file(path) for path in report["figures"]}

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    temporary_h5 = output_h5.with_name(f".{output_h5.name}.tmp.{os.getpid()}")
    with h5py.File(temporary_h5, "w") as handle:
        handle.attrs["schema"] = SCHEMA
        handle.attrs["summary_json"] = canonical_json(report)
        handle.attrs["provenance_json"] = canonical_json(provenance)
        handle.create_dataset("radius_R200c", data=scaled_radius)
        node_group = handle.create_group("nodes")
        for key in ("mass_hMsun", "z", "R200c_comoving_hMpc"):
            node_group.create_dataset(key, data=[node[key] for node in nodes])
        profile_group = handle.create_group("profiles")
        for variant, fields in profiles.items():
            child = profile_group.create_group(variant)
            for field, values in fields.items():
                child.create_dataset(field, data=values, compression="lzf")
        convergence_group = handle.create_group("dense_quadrature_256")
        for variant, fields in dense_convergence.items():
            child = convergence_group.create_group(variant)
            for field, values in fields.items():
                child.create_dataset(field, data=values, compression="lzf")
        grid_group = handle.create_group("godmax_grid_scan")
        for variant, fields in godmax_grid_scan.items():
            child = grid_group.create_group(variant)
            for field, values in fields.items():
                child.create_dataset(field, data=values, compression="lzf")
        point_group = handle.create_group("godmax_projection_point_scan")
        for variant, fields in godmax_point_scan.items():
            child = point_group.create_group(variant)
            for field, values in fields.items():
                child.create_dataset(field, data=values, compression="lzf")
        bfg_group = handle.create_group("baryonforge_points_per_decade_scan")
        for variant, fields in baryonforge_point_scan.items():
            child = bfg_group.create_group(variant)
            for field, values in fields.items():
                child.create_dataset(field, data=values, compression="lzf")
    os.replace(temporary_h5, output_h5)
    report["output_h5"] = str(output_h5)
    report["output_h5_sha256"] = sha256_file(output_h5)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = summary_json.with_name(f".{summary_json.name}.tmp.{os.getpid()}")
    temporary_json.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_json, summary_json)
    report["summary_json"] = str(summary_json)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--summary-json")
    parser.add_argument("--figure-dir")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    root = resolve_path(config["project"]["output_root"], config["_config_path"])
    output = resolve_path(
        args.output or root / "profiles" / "projection_convergence.h5"
    )
    summary = resolve_path(
        args.summary_json or root / "profiles" / "projection_convergence_summary.json"
    )
    figures = resolve_path(args.figure_dir or root / "profiles" / "figures")
    report = run(config, output, summary, figures)
    print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
