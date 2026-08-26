#!/usr/bin/env python3
"""Validate the projected paste operator against the spherical theory candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLBACKEND", "Agg")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
for path in (REPO_ROOT / "src", THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from three_probe_mock_contract import sha256_array, sha256_file  # noqa: E402
from three_probe_projected_operator import (  # noqa: E402
    painter_log_interpolate,
    painter_rp_nodes,
    project_physical_profile_cosh,
    project_physical_profile_legacy,
    projected_painter_transform,
    spherical_support_transform,
    symmetric_fractional_difference,
)
from validate_three_probe_resolved_theory import build_pkz, load_contract  # noqa: E402


FIELDS = ("y", "e", "m")
TARGET_LOGM = (np.log10(5.0e11), 13.0, 14.0, 15.0)
TARGET_Z = (0.3, 0.4, 0.5)
PAINT_FACTOR = 8.0


def file_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_profiles(pkz, iz: int, im: int) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """Return (theory profile, physical painter profile, plane-volume factor)."""

    a = 1.0 / (1.0 + float(pkz.z_array[iz]))
    rhom = float(pkz.rhom_0)
    return {
        "y": (
            np.asarray(pkz.y3d_mat[:, iz, im], dtype=np.float64),
            np.asarray(pkz.y3d_mat[:, iz, im], dtype=np.float64),
            a ** -3,
        ),
        "e": (
            np.asarray(pkz.ne_mat[:, iz, im], dtype=np.float64),
            np.asarray(pkz.ne_mat_physical[:, iz, im], dtype=np.float64),
            1.0,
        ),
        "m": (
            np.asarray(pkz.rho_dmb_mat[:, iz, im], dtype=np.float64) / rhom,
            np.asarray(pkz.rho_dmb_mat[:, iz, im], dtype=np.float64) / (a**3 * rhom),
            1.0,
        ),
    }


def evaluate_grid(
    pkz,
    label: str,
    *,
    n_los: int,
    n_rp: int,
    legacy_n_los: int = 32,
) -> dict[str, np.ndarray]:
    radius = np.asarray(pkz.r_array, dtype=np.float64)
    mass_grid = np.asarray(pkz.M_array, dtype=np.float64)
    z_grid = np.asarray(pkz.z_array, dtype=np.float64)
    k = np.concatenate(([0.0], np.geomspace(1.0e-4, 2.0, 96)))
    shape = (len(FIELDS), len(TARGET_Z), len(TARGET_LOGM), k.size)
    sphere = np.empty(shape)
    cylinder = np.empty(shape)
    painter_cosh = np.empty(shape)
    painter_legacy = np.empty(shape)
    actual_mass = np.empty((len(TARGET_Z), len(TARGET_LOGM)))
    actual_z = np.empty_like(actual_mass)
    rp_min_over_aperture = np.empty_like(actual_mass)

    source_rp = painter_rp_nodes(radius)
    for jz, target_z in enumerate(TARGET_Z):
        iz = int(np.argmin(np.abs(z_grid - target_z)))
        z = float(z_grid[iz])
        a = 1.0 / (1.0 + z)
        for jm, target_logm in enumerate(TARGET_LOGM):
            im = int(np.argmin(np.abs(np.log10(mass_grid) - target_logm)))
            actual_mass[jz, jm] = mass_grid[im]
            actual_z[jz, jm] = z
            r200_com = float(pkz.r200c_mat[iz, im])
            aperture_phys = PAINT_FACTOR * r200_com * a
            rp_min_over_aperture[jz, jm] = source_rp[0] / aperture_phys
            rp_dense = np.geomspace(max(aperture_phys * 1.0e-7, 1.0e-10), aperture_phys, n_rp)

            for jf, field in enumerate(FIELDS):
                theory_profile, physical_profile, plane_factor = selected_profiles(pkz, iz, im)[field]
                sphere[jf, jz, jm] = spherical_support_transform(
                    k, radius, theory_profile, PAINT_FACTOR * r200_com
                )
                continuous_sigma = project_physical_profile_cosh(
                    radius, physical_profile, z, rp_dense, n_los=n_los
                )
                cylinder[jf, jz, jm] = projected_painter_transform(
                    k, rp_dense, continuous_sigma, z, aperture_phys,
                    physical_to_theory_volume_factor=plane_factor,
                )
                for method, destination in (
                    ("physical_table_cosh", painter_cosh),
                    ("legacy_log_radius", painter_legacy),
                ):
                    projector = (
                        project_physical_profile_cosh
                        if method == "physical_table_cosh"
                        else project_physical_profile_legacy
                    )
                    source_sigma = projector(
                        radius, physical_profile, z, source_rp,
                        n_los=n_los if method == "physical_table_cosh" else legacy_n_los,
                    ).astype(np.float32).astype(np.float64)
                    interpolated = painter_log_interpolate(source_rp, source_sigma, rp_dense)
                    destination[jf, jz, jm] = projected_painter_transform(
                        k, rp_dense, interpolated, z, aperture_phys,
                        physical_to_theory_volume_factor=plane_factor,
                    )

    return {
        "label": np.asarray(label),
        "k_hmpc": k,
        "sphere": sphere,
        "continuous_cylinder": cylinder,
        "painter_cosh": painter_cosh,
        "painter_legacy": painter_legacy,
        "actual_mass_hmsun": actual_mass,
        "actual_redshift": actual_z,
        "rp_min_over_aperture": rp_min_over_aperture,
        "radius_hmpc": radius,
    }


def metrics(result: dict[str, np.ndarray]) -> dict[str, object]:
    k = result["k_hmpc"]
    relevant = k <= 2.0
    output: dict[str, object] = {}
    for name in ("continuous_cylinder", "painter_cosh", "painter_legacy"):
        diff = symmetric_fractional_difference(result[name], result["sphere"])
        zero_ratio = result[name][..., 0] / result["sphere"][..., 0]
        normalized_abs = np.abs(result[name] - result["sphere"]) / np.abs(
            result["sphere"][..., :1]
        )
        output[name] = {
            "zero_mode_ratio_min": float(np.min(zero_ratio)),
            "zero_mode_ratio_max": float(np.max(zero_ratio)),
            "median_abs_symmetric_fractional_difference_k_le_2": float(
                np.median(np.abs(diff[..., relevant]))
            ),
            "max_abs_difference_over_spherical_zero_mode_k_le_2": float(
                np.max(normalized_abs[..., relevant])
            ),
        }
    output["rp_min_over_aperture"] = {
        "min": float(np.min(result["rp_min_over_aperture"])),
        "max": float(np.max(result["rp_min_over_aperture"])),
    }
    return output


def save_plots(output_dir: pathlib.Path, result: dict[str, np.ndarray], tag: str) -> None:
    k = result["k_hmpc"]
    colors = {"y": "tab:red", "e": "tab:blue", "m": "tab:green"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    operators = ("continuous_cylinder", "painter_cosh", "painter_legacy")
    titles = ("Continuous cylinder / sphere", "Cosh painter table / sphere", "Legacy painter table / sphere")
    for axis, operator, title in zip(axes, operators, titles):
        for jf, field in enumerate(FIELDS):
            ratio = result[operator][jf] / result["sphere"][jf]
            for curve in ratio.reshape(-1, k.size):
                axis.plot(k, curve, color=colors[field], alpha=0.25, lw=0.8)
            axis.plot([], [], color=colors[field], label=field)
        axis.axhline(1.0, color="black", ls="--", lw=0.8)
        axis.set_xscale("symlog", linthresh=1.0e-4)
        axis.set_title(title)
        axis.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("operator ratio")
    axes[0].legend()
    fig.suptitle("Projected painter versus spherical 8R200c candidate (all representative M,z)")
    fig.tight_layout()
    fig.savefig(output_dir / f"projected_operator_ratios_{tag}.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for jf, field in enumerate(FIELDS):
        zero = result["continuous_cylinder"][jf, ..., 0] / result["sphere"][jf, ..., 0]
        image = axes[jf].imshow(zero, origin="lower", aspect="auto")
        axes[jf].set_title(f"{field}: cylinder/sphere at k=0")
        axes[jf].set_xticks(range(len(TARGET_LOGM)), [f"{v:.2f}" for v in TARGET_LOGM])
        axes[jf].set_yticks(range(len(TARGET_Z)), [f"{v:.1f}" for v in TARGET_Z])
        axes[jf].set_xlabel("target log10 M")
        axes[jf].set_ylabel("target z")
        fig.colorbar(image, ax=axes[jf])
    fig.tight_layout()
    fig.savefig(output_dir / f"projected_operator_zero_modes_{tag}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=THIS_DIR / "three_probe_mock_experiment.yaml")
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--skip-catalog-sha", action="store_true")
    args = parser.parse_args()

    _config, theory, catalog_path, source_files, support, cosmology = load_contract(
        args.config, verify_catalog_sha=not args.skip_catalog_sha
    )
    kernel_path = pathlib.Path(theory["validation_output_dir"]) / theory["lens_kernel"]["output_name"]
    output_dir = args.output_dir or pathlib.Path(theory["validation_output_dir"]) / "projected_operator"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    robust_pkz = None
    for robust, label, n_los, n_rp in (
        (False, "baseline", 128, 1025),
        (True, "robust", 256, 2049),
    ):
        pkz, _resolved_support, params_path, _z_lens, _nz_lens = build_pkz(
            theory, support, cosmology, kernel_path, robust=robust
        )
        run = evaluate_grid(pkz, label, n_los=n_los, n_rp=n_rp)
        np.savez_compressed(output_dir / f"projected_operator_{label}.npz", **run)
        runs.append((run, params_path, int(pkz.nr), int(pkz.nM), int(pkz.nz)))
        if robust:
            robust_pkz = pkz

    baseline, params_path, base_nr, base_nm, base_nz = runs[0]
    robust, _, robust_nr, robust_nm, robust_nz = runs[1]
    if robust_pkz is None:
        raise RuntimeError("robust GODMAX object was not constructed")
    quadrature_control = evaluate_grid(
        robust_pkz, "robust_profiles_baseline_projection", n_los=128, n_rp=1025
    )
    np.savez_compressed(
        output_dir / "projected_operator_robust_profiles_baseline_projection.npz",
        **quadrature_control,
    )
    save_plots(output_dir, baseline, "baseline")
    save_plots(output_dir, robust, "robust")
    base_metrics = metrics(baseline)
    robust_metrics = metrics(robust)
    common = {}
    for name in ("sphere", "continuous_cylinder", "painter_cosh", "painter_legacy"):
        diff = symmetric_fractional_difference(robust[name], baseline[name])
        common[name] = {
            "median_abs_symmetric_fractional_difference": float(np.median(np.abs(diff))),
            "max_abs_symmetric_fractional_difference": float(np.max(np.abs(diff))),
            "note": "includes nearest-node M/z displacement between the two GODMAX grids",
        }
    projection_convergence = {}
    for name in ("continuous_cylinder", "painter_cosh", "painter_legacy"):
        diff = symmetric_fractional_difference(robust[name], quadrature_control[name])
        projection_convergence[name] = {
            "median_abs_symmetric_fractional_difference": float(np.median(np.abs(diff))),
            "max_abs_symmetric_fractional_difference": float(np.max(np.abs(diff))),
            "note": "same nr=45,nM=192,nz=128 profiles and identical nearest M/z nodes",
        }

    npz_paths = [
        output_dir / "projected_operator_baseline.npz",
        output_dir / "projected_operator_robust.npz",
        output_dir / "projected_operator_robust_profiles_baseline_projection.npz",
    ]
    summary = {
        "status": "SUPPORT_GEOMETRY_MISMATCH_CONFIRMED_NUMERICAL_REPLACEMENT_UNCONVERGED",
        "catalog": {"path": str(catalog_path), "sha256": theory["catalog_file_sha256"]},
        "source_shell_count": len(source_files),
        "cosmology": cosmology,
        "support": {"mass_min_hmsun": 5.0e11, "mass_max_hmsun": 1.0e16, "z_min": 0.3, "z_max": 0.5},
        "paint_factor_r200c": PAINT_FACTOR,
        "production_default_projection": "legacy_log_radius",
        "unit_consistent_candidate_projection": "physical_table_cosh",
        "smoothing_included": False,
        "legacy_projection_n_los": 32,
        "baseline_grid": {"nr": base_nr, "nM": base_nm, "nz": base_nz, "cosh_n_los": 128, "n_rp": 1025},
        "robust_grid": {"nr": robust_nr, "nM": robust_nm, "nz": robust_nz, "cosh_n_los": 256, "n_rp": 2049},
        "baseline_metrics": base_metrics,
        "robust_metrics": robust_metrics,
        "grid_comparison": common,
        "same_profile_projection_convergence": projection_convergence,
        "interpretation": (
            "The unit-consistent cosh path integrates to the physical radial-table edge; the legacy "
            "path instead uses min(comoving table edge,100*rp) on mixed coordinates. Both are then "
            "sampled through a transverse 8R200c aperture. This projected window is not the spherical 8R200c "
            "mask in the current resolved-theory candidate. The support mismatch is exact, while "
            "the reported c0000 amplitudes remain numerical diagnostics rather than a converged replacement."
        ),
        "provenance": {
            "git_commit": os.popen("git rev-parse HEAD").read().strip(),
            "validation_script_sha256": file_hash(pathlib.Path(__file__)),
            "operator_module_sha256": file_hash(THIS_DIR / "three_probe_projected_operator.py"),
            "get_sim_maps_sha256": file_hash(REPO_ROOT / "src" / "get_sim_maps.py"),
            "experiment_config_sha256": file_hash(args.config),
            "default_params_sha256": file_hash(params_path),
            "lens_kernel_sha256": sha256_file(kernel_path),
            "artifact_sha256": {path.name: file_hash(path) for path in npz_paths},
            "baseline_array_sha256": sha256_array(*[baseline[key] for key in ("k_hmpc", "sphere", "continuous_cylinder", "painter_cosh", "painter_legacy")]),
            "robust_array_sha256": sha256_array(*[robust[key] for key in ("k_hmpc", "sphere", "continuous_cylinder", "painter_cosh", "painter_legacy")]),
            "quadrature_control_array_sha256": sha256_array(*[quadrature_control[key] for key in ("k_hmpc", "sphere", "continuous_cylinder", "painter_cosh", "painter_legacy")]),
        },
    }
    with (output_dir / "projected_operator_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
