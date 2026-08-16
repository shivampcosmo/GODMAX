#!/usr/bin/env python
"""Quantify and plot the historical 8R-to-asymptotic profile change."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import sha256_file, sha256_json


FIELDS = {
    "gas": "rho_gas_direct_ratio_support_median",
    "matter": "rho_matter_direct_ratio_support_median",
    "collisionless": "rho_collisionless_direct_ratio_support_median",
    "stars": "rho_stars_direct_ratio_support_median",
    "y_direct": "y_projected_direct_ratio_support_median",
    "y_tabulated": "y_projected_tabulated_ratio_support_median",
    "kappa_tabulated": "kappa_cmb_tabulated_ratio_support_median",
}
CONVERGENCE_SCHEMA = "godmax_asymptotic_integration_convergence_v3"
PREVIOUS_CONVERGENCE_SCHEMA = "godmax_asymptotic_integration_convergence_v2"
LEGACY_CONVERGENCE_SCHEMA = "godmax_asymptotic_integration_convergence_v1"


def _aggregate(values: np.ndarray) -> dict[str, float]:
    log_error = np.abs(np.log(np.asarray(values, dtype=np.float64)))
    return {
        "min_ratio": float(np.min(values)),
        "max_ratio": float(np.max(values)),
        "median_abs_log_ratio": float(np.median(log_error)),
        "rms_log_ratio": float(np.sqrt(np.mean(log_error**2))),
        "worst_abs_log_ratio": float(np.max(log_error)),
    }


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension, kwargs in (("png", {"dpi": 180}), ("pdf", {})):
        output = directory / f"{stem}.{extension}"
        temporary = directory / f".{stem}.tmp.{os.getpid()}.{extension}"
        fig.savefig(temporary, bbox_inches="tight", **kwargs)
        os.replace(temporary, output)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def bind_convergence_evidence(
    convergence_path: Path, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Bind a profile summary to the convergence decision for that exact config."""

    with h5py.File(convergence_path, "r") as handle:
        try:
            summary = json.loads(str(handle.attrs["summary_json"]))
        except KeyError as error:
            raise ValueError(
                f"Convergence artifact has no summary_json: {convergence_path}"
            ) from error

    schema = str(summary.get("schema", ""))
    rejected_schemas = {PREVIOUS_CONVERGENCE_SCHEMA, LEGACY_CONVERGENCE_SCHEMA}
    if schema not in {CONVERGENCE_SCHEMA, *rejected_schemas}:
        raise ValueError(
            "Unexpected convergence schema: "
            f"{schema!r}, expected {CONVERGENCE_SCHEMA!r} or one of the "
            f"rejected historical schemas {sorted(rejected_schemas)!r}."
        )
    legacy_schema = schema != CONVERGENCE_SCHEMA
    profile_config_sha = str(provenance["comparison_config_sha256"])
    convergence_config_sha = str(summary.get("config_sha256", ""))
    if convergence_config_sha != profile_config_sha:
        raise ValueError(
            "The profile and convergence artifact were generated from different "
            "comparison configs: "
            f"profile={profile_config_sha}, convergence={convergence_config_sha}."
        )

    binding_pairs = {
        "godmax_params_sha256": (
            str(provenance["godmax_params_sha256"]),
            summary.get("godmax_params_sha256"),
        ),
        "profile_source_manifest_sha256": (
            str(provenance["source_manifest_sha256"]),
            summary.get("profile_source_manifest_sha256"),
        ),
        "runtime_manifest_sha256": (
            sha256_json(provenance["runtime_versions"]),
            summary.get("runtime_manifest_sha256"),
        ),
    }
    bindings: dict[str, Any] = {}
    for name, (expected, raw_actual) in binding_pairs.items():
        actual = None if raw_actual is None else str(raw_actual)
        matches = actual == expected
        bindings[name] = {
            "expected_from_profile": expected,
            "actual_from_convergence": actual,
            "matches": matches,
        }
        if actual is not None and not matches:
            raise ValueError(
                f"The profile and convergence artifact have different {name}: "
                f"profile={expected!r}, convergence={actual!r}."
            )

    contract = provenance["profile_integration_contract"]["godmax"]
    rmax = float(contract["r_max_R200c"])
    n_points = int(contract["extended_num_points"])
    method = str(contract["extended_integration_method"])
    method_tag = {
        "uniform_log_trapezoid": "trap",
        "gauss_legendre_log": "gl",
    }.get(method)
    if method_tag is None:
        raise ValueError(f"Unsupported profile integration method {method!r}.")
    candidate_name = f"production_{rmax:g}R_{method_tag}{n_points}"
    variants = summary.get("variants", {})
    if candidate_name not in variants:
        raise ValueError(
            f"Convergence artifact does not contain candidate {candidate_name!r}."
        )
    candidate = variants[candidate_name]
    if not (
        float(candidate.get("rmax_R200c", float("nan"))) == rmax
        and int(candidate.get("n_points", -1)) == n_points
        and str(candidate.get("method", "")) == method
    ):
        raise ValueError(
            f"Convergence candidate {candidate_name!r} does not match the profile "
            f"contract ({rmax:g} R200c, {n_points} points): {candidate!r}."
        )
    expected_class = str(contract["profiles_class_fqname"])
    actual_class = str(summary.get("profiles_class_fqname", ""))
    if actual_class != expected_class:
        raise ValueError(
            "The profile and convergence artifact used different GODMAX classes: "
            f"profile={expected_class!r}, convergence={actual_class!r}."
        )

    acceptance = summary.get("acceptance", {})
    full_chain = summary.get("pressure_full_chain", {})
    required_acceptance = {
        "production_full_chain_HSE_converged",
        "full_chain_rebuild_reproduces_production",
    }
    required_rebuilt_fields = {
        "Mtot",
        "fstar_total",
        "fstar_central",
        "fstar_satellite",
        "fgas",
        "fclm",
        "gas_norm",
        "Mdmb",
        "Ptot",
    }
    semantic_checks = {
        "current_v3_method_aware_schema": not legacy_schema,
        "full_chain_acceptance_fields": required_acceptance.issubset(acceptance),
        "full_chain_candidate_recorded": (
            full_chain.get("production") == candidate_name
            and candidate_name in full_chain.get("max_abs_relative_error", {})
        ),
        "full_chain_rebuilt_fields": required_rebuilt_fields.issubset(
            set(full_chain.get("rebuilt_fields", []))
        ),
    }
    semantic_complete = all(semantic_checks.values())
    numerical_ok = bool(summary.get("ok", False))
    binding_complete = semantic_complete and all(
        item["matches"] for item in bindings.values()
    )
    return {
        "path": str(convergence_path),
        "sha256": sha256_file(convergence_path),
        "schema": schema,
        "config_path": summary.get("config_path"),
        "config_sha256": convergence_config_sha,
        "fixed_tolerance": float(summary["fixed_tolerance"]),
        "candidate_variant": candidate_name,
        "candidate_contract": candidate,
        "acceptance": acceptance,
        "bindings": bindings,
        "semantic_checks": semantic_checks,
        "semantic_complete": semantic_complete,
        "legacy_rejection_reason": (
            (
                "legacy_v1_held_fixed_HSE_is_not_acceptance_evidence"
                if schema == LEGACY_CONVERGENCE_SCHEMA
                else "v2_trapezoid_only_schema_cannot_certify_log_GL"
            )
            if legacy_schema
            else None
        ),
        "binding_complete": binding_complete,
        "numerical_ok": numerical_ok,
        "ok": numerical_ok and binding_complete,
    }


def _is_projected_baryonforge_dataset(root: str, dataset: str) -> bool:
    """Identify arrays expected to change under the registered LOS adapter."""

    return root == "baryonforge_tabulated_for_painter" or dataset in {
        "y_projected",
        "sigma_matter_physical_Msun_Mpc2",
        "kappa_cmb",
    }


def summarize(
    old_path: Path,
    new_path: Path,
    convergence_path: Path,
    *,
    allow_registered_projection_change: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with h5py.File(old_path, "r") as old, h5py.File(new_path, "r") as new:
        keys = sorted(name for name in new if name.startswith("log10M"))
        if keys != sorted(name for name in old if name.startswith("log10M")):
            raise ValueError(
                "Historical and candidate artifacts have different profile nodes."
            )

        node_mass = np.asarray([new[key].attrs["mass_hMsun"] for key in keys])
        node_redshift = np.asarray([new[key].attrs["z"] for key in keys])
        ratios: dict[str, dict[str, np.ndarray]] = {}
        aggregate: dict[str, Any] = {}
        for field, attribute in FIELDS.items():
            old_values = np.asarray([old[key].attrs[attribute] for key in keys])
            new_values = np.asarray([new[key].attrs[attribute] for key in keys])
            ratios[field] = {"old": old_values, "new": new_values}
            old_summary = _aggregate(old_values)
            new_summary = _aggregate(new_values)
            aggregate[field] = {
                "old": old_summary,
                "new": new_summary,
                "all_nodes_abs_log_nonworse": bool(
                    np.all(
                        np.abs(np.log(new_values))
                        <= np.abs(np.log(old_values)) + 1.0e-12
                    )
                ),
                "rms_log_error_improvement_factor": float(
                    old_summary["rms_log_ratio"] / new_summary["rms_log_ratio"]
                ),
            }

        baryonforge_mismatches = []
        baryonforge_nonprojected_mismatches = []
        for key in keys:
            for root in ("baryonforge", "baryonforge_tabulated_for_painter"):
                old_datasets = set(old[key][root])
                new_datasets = set(new[key][root])
                for dataset in sorted(old_datasets - new_datasets):
                    mismatch = f"{key}/{root}/{dataset} (missing from candidate)"
                    baryonforge_mismatches.append(mismatch)
                    if not _is_projected_baryonforge_dataset(root, dataset):
                        baryonforge_nonprojected_mismatches.append(mismatch)
                for dataset in sorted(new_datasets - old_datasets):
                    mismatch = f"{key}/{root}/{dataset} (missing from historical)"
                    baryonforge_mismatches.append(mismatch)
                    if not _is_projected_baryonforge_dataset(root, dataset):
                        baryonforge_nonprojected_mismatches.append(mismatch)
                for dataset in sorted(old_datasets & new_datasets):
                    if not np.array_equal(
                        old[key][root][dataset][:],
                        new[key][root][dataset][:],
                        equal_nan=True,
                    ):
                        mismatch = f"{key}/{root}/{dataset}"
                        baryonforge_mismatches.append(mismatch)
                        if not _is_projected_baryonforge_dataset(root, dataset):
                            baryonforge_nonprojected_mismatches.append(mismatch)

        provenance = json.loads(str(new.attrs["provenance_json"]))
        projection_variant = str(
            provenance.get("projected_profile_contract", {}).get(
                "projection_variant", ""
            )
        )
        registered_projection_change = bool(
            allow_registered_projection_change
            and projection_variant == "physical_table_cosh_100mpc_v1"
        )
        convergence = bind_convergence_evidence(convergence_path, provenance)
        component = provenance["component_conservation_check"]
        report = {
            "schema": "baryonforge_godmax_integration_change_summary_v1",
            "old_profile": str(old_path),
            "old_profile_sha256": sha256_file(old_path),
            "new_profile": str(new_path),
            "new_profile_sha256": sha256_file(new_path),
            "node_count": len(keys),
            "aggregate": aggregate,
            "baryonforge_arrays_bitwise_unchanged": not baryonforge_mismatches,
            "baryonforge_array_mismatches": baryonforge_mismatches,
            "baryonforge_nonprojected_arrays_bitwise_unchanged": (
                not baryonforge_nonprojected_mismatches
            ),
            "baryonforge_nonprojected_array_mismatches": (
                baryonforge_nonprojected_mismatches
            ),
            "registered_projection_change": {
                "requested": bool(allow_registered_projection_change),
                "candidate_projection_variant": projection_variant,
                "accepted": registered_projection_change,
            },
            "component_conservation": component,
            "convergence": convergence,
        }
        gas = aggregate["gas"]
        report["acceptance"] = {
            "gas_all_ratios_within_0p99_1p01": bool(
                gas["new"]["min_ratio"] >= 0.99 and gas["new"]["max_ratio"] <= 1.01
            ),
            "gas_median_abs_log_improved": bool(
                gas["new"]["median_abs_log_ratio"] < gas["old"]["median_abs_log_ratio"]
            ),
            "gas_rms_log_improved": bool(
                gas["new"]["rms_log_ratio"] < gas["old"]["rms_log_ratio"]
            ),
            "gas_worst_abs_log_improved": bool(
                gas["new"]["worst_abs_log_ratio"] < gas["old"]["worst_abs_log_ratio"]
            ),
            "gas_all_nodes_nonworse": aggregate["gas"]["all_nodes_abs_log_nonworse"],
            "y_direct_rms_log_improved": bool(
                aggregate["y_direct"]["new"]["rms_log_ratio"]
                < aggregate["y_direct"]["old"]["rms_log_ratio"]
            ),
            "y_tabulated_rms_log_improved": bool(
                aggregate["y_tabulated"]["new"]["rms_log_ratio"]
                < aggregate["y_tabulated"]["old"]["rms_log_ratio"]
            ),
            "kappa_tabulated_rms_log_improved": bool(
                aggregate["kappa_tabulated"]["new"]["rms_log_ratio"]
                < aggregate["kappa_tabulated"]["old"]["rms_log_ratio"]
            ),
            "baryonforge_unchanged_arrays_bitwise_unchanged": bool(
                not baryonforge_nonprojected_mismatches
                and (not baryonforge_mismatches or registered_projection_change)
            ),
            "component_conservation_ok": bool(component["ok"]),
            "integration_convergence_ok": bool(convergence["ok"]),
        }
        report["ok"] = all(report["acceptance"].values())
        plot_data = {
            "mass": node_mass,
            "redshift": node_redshift,
            "ratios": ratios,
            "rmax_R200c": provenance["profile_integration_contract"]["godmax"][
                "r_max_R200c"
            ],
            "n_points": provenance["profile_integration_contract"]["godmax"][
                "extended_num_points"
            ],
            "integration_method": provenance["profile_integration_contract"][
                "godmax"
            ]["extended_integration_method"],
            "registered_projection_change": registered_projection_change,
        }
        return report, plot_data


def plot_summary(plot_data: dict[str, Any], directory: Path) -> list[str]:
    mass = plot_data["mass"]
    redshift = plot_data["redshift"]
    ratios = plot_data["ratios"]
    rmax = float(plot_data["rmax_R200c"])
    n_points = int(plot_data["n_points"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(np.unique(redshift))))
    for color, z_value in zip(colors, np.unique(redshift)):
        selected = redshift == z_value
        order = np.argsort(mass[selected])
        x = np.log10(mass[selected][order])
        axes[0].plot(
            x,
            ratios["gas"]["old"][selected][order],
            marker="o",
            linestyle="--",
            color=color,
            alpha=0.65,
            label=f"8R, z={z_value:.2f}",
        )
        axes[0].plot(
            x,
            ratios["gas"]["new"][selected][order],
            marker="s",
            linestyle="-",
            color=color,
            label=f"{rmax:g}R, z={z_value:.2f}",
        )
    axes[0].axhline(1.0, color="black", linewidth=1)
    axes[0].axhspan(0.99, 1.01, color="0.85", alpha=0.5)
    axes[0].set(
        xlabel=r"$\log_{10}(M_{200c}/[M_\odot/h])$",
        ylabel="BaryonForge / GODMAX gas density",
        title="Gas normalization at all nine nodes",
    )
    axes[0].legend(fontsize=7, ncol=2)

    labels = list(FIELDS)
    old_error = [
        np.sqrt(np.mean(np.log(ratios[label]["old"]) ** 2)) for label in labels
    ]
    new_error = [
        np.sqrt(np.mean(np.log(ratios[label]["new"]) ** 2)) for label in labels
    ]
    positions = np.arange(len(labels))
    axes[1].bar(positions - 0.18, old_error, width=0.36, label="native 8R")
    axes[1].bar(
        positions + 0.18,
        new_error,
        width=0.36,
        label=f"{rmax:g}R / GL{n_points}",
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions, labels, rotation=35, ha="right")
    axes[1].set(
        ylabel="RMS log ratio across nine nodes",
        title="Before/after profile agreement",
    )
    axes[1].legend()
    title = "BaryonForge--GODMAX asymptotic-normalization validation"
    if plot_data.get("registered_projection_change"):
        title = "BaryonForge--GODMAX integration + LOS-projection validation"
    fig.suptitle(title)
    fig.tight_layout()
    return _save_figure(fig, directory, "integration_limit_before_after")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument(
        "--convergence",
        required=True,
        help="Convergence HDF5 generated for the exact candidate profile config.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument(
        "--allow-registered-projection-change",
        action="store_true",
        help=(
            "Allow only the projected BaryonForge arrays to change when the "
            "candidate records the matched physical_table_cosh_100mpc_v1 contract."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    old_path = Path(args.old).expanduser().resolve()
    new_path = Path(args.new).expanduser().resolve()
    convergence_path = Path(args.convergence).expanduser().resolve()
    report, plot_data = summarize(
        old_path,
        new_path,
        convergence_path,
        allow_registered_projection_change=args.allow_registered_projection_change,
    )
    report["figures"] = plot_summary(
        plot_data, Path(args.figure_dir).expanduser().resolve()
    )
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
