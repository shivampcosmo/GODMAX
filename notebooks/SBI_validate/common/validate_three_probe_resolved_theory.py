#!/usr/bin/env python3
"""Build and plot catalog-bound common-support resolved GODMAX powers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLBACKEND", "Agg")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import asdf
import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"
for path in (SRC_DIR, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from three_probe_mock_contract import (  # noqa: E402
    canonical_cosmology,
    canonical_json_sha256,
    sha256_array,
    sha256_file,
    validate_catalog_contract,
)
from three_probe_resolved_theory import (  # noqa: E402
    FIELD_ORDER,
    PAIR_ORDER,
    ResolvedSupport,
    assemble_from_godmax,
    map_matched_profile_transforms,
    resolved_halo_overrides,
)


C_KMS = 299792.458


def load_source_header(path: pathlib.Path) -> dict[str, float]:
    with asdf.open(path, lazy_load=True, memmap=False) as handle:
        header = handle["header"]
        keys = (
            "H0", "Omega_M", "Omega_DE", "Omega_K", "CAMB_Omega_b",
            "CAMB_sigma8", "CAMB_ns", "w0", "ParticleMassHMsun", "hMpc",
        )
        result = {key: float(header[key]) for key in keys}
        result["wa"] = float(header.get("wa", 0.0))
    if result["wa"] != 0.0 or result["hMpc"] != 1.0:
        raise ValueError("Source header is incompatible with the flat w0 / h^-1 Mpc contract")
    return result


def chi_hmpc(z: float, cosmology: dict[str, Any]) -> float:
    omega_m = float(cosmology["Om0"])
    w0 = float(cosmology["w0"])

    def inverse_e(redshift: float) -> float:
        dark_energy = (1.0 - omega_m) * (1.0 + redshift) ** (3.0 * (1.0 + w0))
        return 1.0 / np.sqrt(omega_m * (1.0 + redshift) ** 3 + dark_energy)

    return (C_KMS / 100.0) * quad(inverse_e, 0.0, float(z), epsabs=1.0e-10)[0]


def catalog_nbar(kernel_path: pathlib.Path, cosmology: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(kernel_path, "r") as handle:
        group = handle["primary"]
        edges = np.asarray(group["histogram_edges"], dtype=np.float64)
        counts = np.asarray(group["histogram_counts"], dtype=np.float64)
    chi_edges = np.asarray([chi_hmpc(value, cosmology) for value in edges])
    volumes = (4.0 * np.pi / 3.0) * np.diff(chi_edges**3)
    if np.any(volumes <= 0.0) or np.any(counts <= 0.0):
        raise ValueError("Catalog nbar bins must have positive full-sky volume and counts")
    return 0.5 * (edges[:-1] + edges[1:]), counts / volumes


def load_contract(config_path: pathlib.Path, verify_catalog_sha: bool):
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    theory = config["resolved_theory"]
    catalog_path = pathlib.Path(theory["catalog_path"])
    with h5py.File(catalog_path, "r") as handle:
        source_files = json.loads(str(handle.attrs["source_files_json"]))
    headers = [load_source_header(pathlib.Path(path)) for path in source_files]
    attrs, support, cosmology = validate_catalog_contract(
        catalog_path, theory, headers[0], verify_file_sha=verify_catalog_sha
    )
    for path, header in zip(source_files[1:], headers[1:]):
        if header != headers[0] or canonical_cosmology(attrs, header) != cosmology:
            raise ValueError(f"Source-shell cosmology mismatch: {path}")
    return config, theory, catalog_path, source_files, support, cosmology


def build_pkz(
    theory: dict[str, Any],
    support_input: Any,
    cosmology: dict[str, Any],
    kernel_path: pathlib.Path,
    *,
    robust: bool,
):
    from base_class import base_class
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles

    params_path = pathlib.Path(theory["default_params_path"])
    with params_path.open() as handle:
        params = yaml.safe_load(handle)
    sim = copy.deepcopy(params["sim_params"])
    halo = copy.deepcopy(params["halo_params"])
    analysis = copy.deepcopy(params["analysis"])
    other = copy.deepcopy(params["other_params"])
    sim["cosmo"] = dict(cosmology)
    sim["init_power"] = True

    diagnostic = theory["hmf_bias_diagnostic"]
    prefix = "robustness_" if robust else ""
    resolved_support = ResolvedSupport(
        mass_min_hmsun=float(support_input.mass_min_hmsun),
        mass_max_hmsun=float(support_input.mass_max_hmsun),
        z_min=float(support_input.z_min),
        z_max=float(support_input.z_max),
    )
    halo_overrides, analysis_overrides = resolved_halo_overrides(
        resolved_support,
        n_mass=int(diagnostic[f"{prefix}theory_nM"]),
        n_redshift=int(diagnostic[f"{prefix}theory_nz"]),
        n_k=int(diagnostic[f"{prefix}theory_nk"]),
    )
    halo.update(halo_overrides)
    base_nr = int(halo.get("nr", 23))
    halo.update({"kmin": 1.0e-4, "kmax": 1.0e3, "nr": 2 * base_nr - 1 if robust else base_nr})
    analysis.update(analysis_overrides)

    with h5py.File(kernel_path, "r") as handle:
        if str(handle.attrs["catalog_file_sha256"]) != theory["catalog_file_sha256"]:
            raise ValueError("Lens kernel catalog SHA differs from the resolved contract")
        if str(handle.attrs["cosmology_sha256"]) != canonical_json_sha256(cosmology):
            raise ValueError("Lens kernel cosmology differs from the resolved contract")
        group = handle["primary"]
        z_lens = np.asarray(group["z"], dtype=np.float64)
        nz_lens = np.asarray(group["nz"], dtype=np.float64)
        if str(group.attrs["kernel_array_sha256"]) != sha256_array(z_lens, nz_lens):
            raise ValueError("Lens kernel array hash mismatch")
    if abs(float(np.trapz(nz_lens, z_lens)) - 1.0) > 1.0e-6:
        raise ValueError("Catalog lens kernel is not normalized within 1e-6")
    if z_lens[0] != resolved_support.z_min or z_lens[-1] != resolved_support.z_max:
        raise ValueError("Catalog lens kernel support differs from resolved theory")
    analysis["nz_lens_info_dict"] = {
        "nbins_lens": 1,
        "z_edges_bins_lens": [[resolved_support.z_min, resolved_support.z_max]],
        "z_array_lens": z_lens.tolist(),
        "nz0": nz_lens.tolist(),
    }
    z_nbar, nbar = catalog_nbar(kernel_path, cosmology)
    analysis["nbar_gal_comoving_zarray"] = z_nbar.tolist()
    analysis["nbar_gal_comoving_val"] = nbar.tolist()

    base = base_class(sim, halo, analysis, other)
    effective_cosmology = {
        key: (bool(value) if key == "flat" else float(value))
        for key, value in base.cosmo_params.items()
    }
    if effective_cosmology != dict(cosmology):
        raise ValueError("GODMAX constructor cosmology differs from the catalog cosmology")
    profiles = Profiles(sim, halo, analysis, other, base_class_obj=base)
    pkz = get_Pkz(sim, halo, analysis, other, Profiles_obj=profiles)
    return pkz, resolved_support, params_path, z_lens, nz_lens


def provenance(paths: dict[str, pathlib.Path]) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    inputs = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    return {
        "inputs": inputs,
        "git_commit": commit,
        "git_worktree_dirty": bool(status),
        "git_status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "python_executable": sys.executable,
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
    }


def save_plots(output_dir: pathlib.Path, k: np.ndarray, z: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    z_indices = sorted(set((0, z.size // 2, z.size - 1)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, field in zip(axes.flat, FIELD_ORDER):
        plotted = arrays[f"b{field}_resolved"][:, z_indices]
        linthresh = max(float(np.max(np.abs(plotted))) * 1.0e-12, np.finfo(np.float64).tiny)
        for index in z_indices:
            axis.semilogx(k, arrays[f"b{field}_resolved"][:, index], label=f"z={z[index]:.3f}")
        axis.set_yscale("symlog", linthresh=linthresh)
        axis.set_title(f"raw resolved b_{field}(k,z)")
        axis.set_ylabel("signed field bias factor")
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("k [h/Mpc]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Common support: 5e11 <= Mproxy < 1e16 Msun/h, 0.3 < z < 0.5")
    fig.tight_layout()
    fig.savefig(output_dir / "resolved_effective_biases.png", dpi=180)
    plt.close(fig)

    iz = z.size // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for axis, pair in zip(axes, ("gy", "gm", "ge")):
        plotted = np.stack([arrays[f"P{pair}_{suffix}"][:, iz] for suffix in ("1h", "2h", "resolved")])
        linthresh = max(float(np.max(np.abs(plotted))) * 1.0e-12, np.finfo(np.float64).tiny)
        for suffix, style in (("1h", "--"), ("2h", ":"), ("resolved", "-")):
            axis.semilogx(k, arrays[f"P{pair}_{suffix}"][:, iz], style, label=suffix)
        axis.set_yscale("symlog", linthresh=linthresh)
        axis.set_title(f"P{pair} at z={z[iz]:.3f}")
        axis.set_xlabel("k [h/Mpc]")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("signed power component")
    axes[0].legend()
    fig.suptitle("Spherical-support resolved candidate (no low-mass completion)")
    fig.tight_layout()
    fig.savefig(output_dir / "resolved_three_probe_power_components.png", dpi=180)
    plt.close(fig)


def compare_grids(validation_dir: pathlib.Path) -> dict[str, Any]:
    """Report baseline versus doubled-grid differences without a tuned cutoff."""

    baseline_path = validation_dir / "resolved_theory" / "resolved_power_components.npz"
    robust_path = validation_dir / "resolved_theory_robust" / "resolved_power_components.npz"
    with np.load(baseline_path) as baseline_file, np.load(robust_path) as robust_file:
        baseline = {key: np.asarray(baseline_file[key]) for key in baseline_file.files}
        robust = {key: np.asarray(robust_file[key]) for key in robust_file.files}
    k_base, z_base = baseline["k_hmpc"], baseline["redshift"]
    k_robust, z_robust = robust["k_hmpc"], robust["redshift"]
    logk_mesh, z_mesh = np.meshgrid(np.log(k_base), z_base, indexing="ij")
    points = np.column_stack((logk_mesh.ravel(), z_mesh.ravel()))
    comparisons: dict[str, Any] = {}
    interpolated: dict[str, np.ndarray] = {}
    for key in sorted(set(baseline).intersection(robust)):
        if key in {"mass_hmsun", "redshift", "k_hmpc", "radius_mpch"}:
            continue
        interp = RegularGridInterpolator(
            (np.log(k_robust), z_robust), robust[key], bounds_error=True
        )(points).reshape((k_base.size, z_base.size))
        interpolated[key] = interp
        absolute = np.abs(interp - baseline[key])
        denominator = np.abs(interp) + np.abs(baseline[key])
        symmetric_fraction = np.where(denominator > 0.0, 2.0 * absolute / denominator, 0.0)
        comparisons[key] = {
            "median_absolute_difference": float(np.median(absolute)),
            "max_absolute_difference": float(np.max(absolute)),
            "median_symmetric_fractional_difference": float(np.median(symmetric_fraction)),
            "max_symmetric_fractional_difference": float(np.max(symmetric_fraction)),
        }

    output_dir = validation_dir / "resolved_theory_grid_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "GRID_DIFFERENCES_REPORTED_NO_ACCEPTANCE_TOLERANCE_INVENTED",
        "baseline_grid": {"nM": baseline["mass_hmsun"].size, "nz": z_base.size, "nk": k_base.size, "nr": baseline["radius_mpch"].size},
        "robust_grid": {"nM": robust["mass_hmsun"].size, "nz": z_robust.size, "nk": k_robust.size, "nr": robust["radius_mpch"].size},
        "metric": "2*abs(robust_interpolated-baseline)/(abs(robust_interpolated)+abs(baseline))",
        "comparisons": comparisons,
    }
    with (output_dir / "grid_convergence.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    iz = z_base.size // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for axis, pair in zip(axes, ("gy", "gm", "ge")):
        key = f"P{pair}_resolved"
        denominator = np.abs(interpolated[key][:, iz]) + np.abs(baseline[key][:, iz])
        difference = np.where(
            denominator > 0.0,
            2.0 * np.abs(interpolated[key][:, iz] - baseline[key][:, iz]) / denominator,
            0.0,
        )
        axis.loglog(k_base, difference)
        axis.set_title(f"{key}, z={z_base[iz]:.3f}")
        axis.set_xlabel("k [h/Mpc]")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("symmetric fractional grid difference")
    fig.suptitle("96x64x256 baseline versus 192x128x512 grid (reported, no tuned cutoff)")
    fig.tight_layout()
    fig.savefig(output_dir / "resolved_grid_convergence.png", dpi=180)
    plt.close(fig)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=THIS_DIR / "three_probe_mock_experiment.yaml")
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--skip-catalog-sha", action="store_true")
    args = parser.parse_args()

    config, theory, catalog_path, source_files, support, cosmology = load_contract(
        args.config, verify_catalog_sha=not args.skip_catalog_sha
    )
    kernel_path = pathlib.Path(theory["validation_output_dir"]) / theory["lens_kernel"]["output_name"]
    output_dir = args.output_dir or pathlib.Path(theory["validation_output_dir"]) / (
        "resolved_theory_robust" if args.robust else "resolved_theory"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pkz, resolved_support, params_path, z_lens, nz_lens = build_pkz(
        theory, support, cosmology, kernel_path, robust=args.robust
    )
    transforms = map_matched_profile_transforms(pkz)
    powers_jax = assemble_from_godmax(pkz, transforms, resolved_support)
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in powers_jax.items()}
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Resolved field power contains non-finite values")

    mass = np.asarray(pkz.M_array, dtype=np.float64)
    redshift = np.asarray(pkz.z_array, dtype=np.float64)
    k = np.asarray(pkz.kPk_array, dtype=np.float64)
    save_payload = {
        "mass_hmsun": mass,
        "redshift": redshift,
        "k_hmpc": k,
        "radius_mpch": np.asarray(pkz.r_array, dtype=np.float64),
        **arrays,
    }
    np.savez_compressed(output_dir / "resolved_power_components.npz", **save_payload)
    save_plots(output_dir, k, redshift, arrays)

    prov = provenance({
        "validation_script": pathlib.Path(__file__),
        "resolved_module": THIS_DIR / "three_probe_resolved_theory.py",
        "contract_module": THIS_DIR / "three_probe_mock_contract.py",
        "experiment_config": args.config,
        "default_params": params_path,
        "lens_kernel": kernel_path,
    })
    summary = {
        "status": "SPHERICAL_SUPPORT_RESOLVED_COMPONENTS_PASS_PROJECTED_OPERATOR_AND_POSTERIORS_PENDING",
        "grid": {"nM": mass.size, "nz": redshift.size, "nk": k.size, "nr": int(pkz.nr)},
        "support": {
            "mass_min_hmsun": resolved_support.mass_min_hmsun,
            "mass_max_hmsun": resolved_support.mass_max_hmsun,
            "z_min": resolved_support.z_min,
            "z_max": resolved_support.z_max,
            "upper_endpoints_are_zero_measure_quadrature_boundaries": True,
        },
        "cosmology": cosmology,
        "catalog": {
            "path": str(catalog_path),
            "sha256": theory["catalog_file_sha256"],
            "source_shell_count": len(source_files),
        },
        "lens_kernel": {
            "path": str(kernel_path),
            "sha256": sha256_file(kernel_path),
            "array_sha256": sha256_array(z_lens, nz_lens),
            "normalization_trapezoid": float(np.trapz(nz_lens, z_lens)),
        },
        "field_order": list(FIELD_ORDER),
        "pair_order": [left + right for left, right in PAIR_ORDER],
        "unresolved_completion": False,
        "projected_painter_operator_equivalence": "pending",
        "electron_convention": "absolute_comoving_electron_number_density_transform",
        "mass_semantics": resolved_support.mass_semantics,
        "arrays_sha256": sha256_array(*[save_payload[key] for key in sorted(save_payload)]),
        "provenance": prov,
    }
    with (output_dir / "resolved_theory_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.robust:
        comparison = compare_grids(pathlib.Path(theory["validation_output_dir"]))
        print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
