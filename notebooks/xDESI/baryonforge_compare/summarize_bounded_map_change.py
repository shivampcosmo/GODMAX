#!/usr/bin/env python
"""Compare two bounded native-map smoke runs.

This is a read-only, non-production validation.  It independently reads the
saved map and diagnostics products, verifies that their selected objects and
geometry are unchanged, recomputes map-pair metrics from the native NSIDE-1024
arrays, and summarizes isolated-halo aperture ratios in the four registered
mass bins.  It never imports either painter or changes a map contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCHEMA = "baryonforge_godmax_bounded_map_change_summary_v1"
LABEL = "64-halo bounded smoke; not production statistics"
MASS_EDGES = np.asarray([13.0, 13.5, 14.0, 14.5, 16.0], dtype=np.float64)
MAP_DATASETS = {
    "y": "maps/map_ymap",
    "kappa": "maps/map_kappa_cmb",
}
GEOMETRY_DATASETS = (
    "source_row",
    "mass_hMsun",
    "redshift",
    "support_angle_deg",
    "isolated_5R",
)
BF_APERTURE_DATASETS = (
    "baryonforge_y_aperture_integral_sr",
    "baryonforge_kappa_aperture_integral_sr",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def arrays_bitwise_equal(first: np.ndarray, second: np.ndarray) -> bool:
    first = np.asarray(first)
    second = np.asarray(second)
    return bool(
        first.shape == second.shape
        and first.dtype == second.dtype
        and np.ascontiguousarray(first).tobytes()
        == np.ascontiguousarray(second).tobytes()
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def smoke_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "diagnostics": root / "smoke64_diagnostics.h5",
        "godmax_map": root
        / "maps"
        / "godmax_native"
        / "abacus_pasted_maps_buffered_mgt13_nside1024_split000of001.h5",
        "baryonforge_map": root / "maps" / "baryonforge_native_nside1024_smoke64.h5",
    }


def require_inputs(paths: Mapping[str, Path]) -> None:
    for name, path in paths.items():
        if name != "root" and not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")


def read_selection(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle["selection"]
        required = set(GEOMETRY_DATASETS) | {
            f"{backend}_{field}_aperture_integral_sr"
            for backend in ("godmax", "baryonforge")
            for field in ("y", "kappa")
        }
        missing = sorted(required.difference(group))
        if missing:
            raise KeyError(f"{path} is missing selection datasets: {missing}")
        return {name: np.asarray(group[name][:]) for name in sorted(required)}


def compare_selection_geometry(
    old: Mapping[str, np.ndarray], new: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    datasets = {}
    for name in GEOMETRY_DATASETS:
        equal = arrays_bitwise_equal(old[name], new[name])
        datasets[name] = {
            "bitwise_equal": equal,
            "old_sha256": array_sha256(old[name]),
            "new_sha256": array_sha256(new[name]),
            "shape": list(old[name].shape),
            "dtype": str(old[name].dtype),
        }
    bf_apertures = {}
    for name in BF_APERTURE_DATASETS:
        bf_apertures[name] = {
            "bitwise_equal": arrays_bitwise_equal(old[name], new[name]),
            "old_sha256": array_sha256(old[name]),
            "new_sha256": array_sha256(new[name]),
        }
    return {
        "datasets": datasets,
        "all_geometry_datasets_bitwise_equal": all(
            item["bitwise_equal"] for item in datasets.values()
        ),
        "baryonforge_apertures": bf_apertures,
        "baryonforge_apertures_bitwise_equal": all(
            item["bitwise_equal"] for item in bf_apertures.values()
        ),
        "halo_count": int(old["source_row"].size),
        "isolated_halo_count": int(np.count_nonzero(old["isolated_5R"])),
    }


def read_map_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        fields = {}
        for field, dataset_name in MAP_DATASETS.items():
            dataset = handle[dataset_name]
            fields[field] = {
                "shape": list(dataset.shape),
                "dtype": str(dataset.dtype),
            }
        source_rows = (
            np.asarray(handle["provenance/source_row"][:])
            if "provenance/source_row" in handle
            else None
        )
        return {
            "nside": int(handle.attrs["nside"]),
            "ordering": str(handle.attrs["ordering"]).upper(),
            "fields": fields,
            "source_rows": source_rows,
        }


def pair_metrics(godmax: np.ndarray, baryonforge: np.ndarray) -> dict[str, Any]:
    godmax = np.asarray(godmax, dtype=np.float64)
    baryonforge = np.asarray(baryonforge, dtype=np.float64)
    if godmax.shape != baryonforge.shape:
        raise ValueError(
            f"Map shapes differ: GODMAX {godmax.shape}, BaryonForge {baryonforge.shape}."
        )
    union = (godmax != 0.0) | (baryonforge != 0.0)
    intersection = (godmax != 0.0) & (baryonforge != 0.0)
    if not np.any(union):
        raise ValueError("Both maps are identically zero.")
    reference = godmax[union]
    candidate = baryonforge[union]
    reference_sum = float(np.sum(godmax, dtype=np.float64))
    candidate_sum = float(np.sum(baryonforge, dtype=np.float64))
    return {
        "difference_convention": "BaryonForge minus GODMAX",
        "pearson_r_on_union": float(np.corrcoef(reference, candidate)[0, 1]),
        "relative_l1": float(
            np.sum(np.abs(candidate - reference), dtype=np.float64)
            / np.sum(np.abs(reference), dtype=np.float64)
        ),
        "global_sum_ratio": float(candidate_sum / reference_sum),
        "godmax_sum": reference_sum,
        "baryonforge_sum": candidate_sum,
        "union_nonzero_pixels": int(np.count_nonzero(union)),
        "intersection_nonzero_pixels": int(np.count_nonzero(intersection)),
        "footprint_jaccard": float(
            np.count_nonzero(intersection) / np.count_nonzero(union)
        ),
    }


def diagnostics_metrics(path: Path, field: str) -> dict[str, float]:
    with h5py.File(path, "r") as handle:
        attrs = handle[f"map_pair_metrics/{field}"].attrs
        return {
            "pearson_r_on_union": float(attrs["pearson_r_on_union"]),
            "relative_l1": float(attrs["relative_l1"]),
            "global_sum_ratio": float(attrs["global_sum_ratio"]),
            "footprint_jaccard": float(attrs["footprint_jaccard"]),
        }


def compare_maps(
    old_paths: Mapping[str, Path],
    new_paths: Mapping[str, Path],
    old_selection: Mapping[str, np.ndarray],
    new_selection: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    metadata = {
        run: {
            backend: read_map_metadata(paths[f"{backend}_map"])
            for backend in ("godmax", "baryonforge")
        }
        for run, paths in (("old", old_paths), ("new", new_paths))
    }
    metadata_consistent = all(
        item["nside"] == 1024 and item["ordering"] == "RING"
        for run in metadata.values()
        for item in run.values()
    )
    metadata_consistent = bool(
        metadata_consistent
        and all(
            run["godmax"]["fields"] == run["baryonforge"]["fields"]
            for run in metadata.values()
        )
        and metadata["old"]["godmax"]["fields"] == metadata["new"]["godmax"]["fields"]
    )

    source_row_checks = {}
    for run, selection in (("old", old_selection), ("new", new_selection)):
        source_rows = metadata[run]["baryonforge"]["source_rows"]
        source_row_checks[run] = bool(
            source_rows is not None
            and arrays_bitwise_equal(source_rows, selection["source_row"])
        )

    fields = {}
    for field, dataset_name in MAP_DATASETS.items():
        arrays = {}
        for run, paths in (("old", old_paths), ("new", new_paths)):
            for backend in ("godmax", "baryonforge"):
                with h5py.File(paths[f"{backend}_map"], "r") as handle:
                    arrays[f"{run}_{backend}"] = np.asarray(handle[dataset_name][:])

        old_metrics = pair_metrics(arrays["old_godmax"], arrays["old_baryonforge"])
        new_metrics = pair_metrics(arrays["new_godmax"], arrays["new_baryonforge"])
        bf_unchanged = arrays_bitwise_equal(
            arrays["old_baryonforge"], arrays["new_baryonforge"]
        )
        footprint_checks = {
            "old_cross_backend": bool(
                np.array_equal(
                    arrays["old_godmax"] != 0.0,
                    arrays["old_baryonforge"] != 0.0,
                )
            ),
            "new_cross_backend": bool(
                np.array_equal(
                    arrays["new_godmax"] != 0.0,
                    arrays["new_baryonforge"] != 0.0,
                )
            ),
            "godmax_before_after": bool(
                np.array_equal(
                    arrays["old_godmax"] != 0.0,
                    arrays["new_godmax"] != 0.0,
                )
            ),
            "baryonforge_before_after": bool(
                np.array_equal(
                    arrays["old_baryonforge"] != 0.0,
                    arrays["new_baryonforge"] != 0.0,
                )
            ),
        }
        stored = {
            "old": diagnostics_metrics(old_paths["diagnostics"], field),
            "new": diagnostics_metrics(new_paths["diagnostics"], field),
        }
        recomputed = {"old": old_metrics, "new": new_metrics}
        consistency = {}
        for run in ("old", "new"):
            consistency[run] = {
                name: bool(
                    math.isclose(
                        recomputed[run][name],
                        stored[run][name],
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                )
                for name in stored[run]
            }
        fields[field] = {
            "old": old_metrics,
            "new": new_metrics,
            "change": {
                "pearson": new_metrics["pearson_r_on_union"]
                - old_metrics["pearson_r_on_union"],
                "relative_l1": new_metrics["relative_l1"] - old_metrics["relative_l1"],
                "absolute_global_sum_error": abs(new_metrics["global_sum_ratio"] - 1.0)
                - abs(old_metrics["global_sum_ratio"] - 1.0),
            },
            "acceptance": {
                "pearson_nonworse": bool(
                    new_metrics["pearson_r_on_union"]
                    >= old_metrics["pearson_r_on_union"] - 1.0e-12
                ),
                "relative_l1_improved": bool(
                    new_metrics["relative_l1"] < old_metrics["relative_l1"]
                ),
                "global_sum_ratio_closer_to_one": bool(
                    abs(new_metrics["global_sum_ratio"] - 1.0)
                    < abs(old_metrics["global_sum_ratio"] - 1.0)
                ),
            },
            "baryonforge_array_bitwise_unchanged": bf_unchanged,
            "baryonforge_array_old_sha256": array_sha256(arrays["old_baryonforge"]),
            "baryonforge_array_new_sha256": array_sha256(arrays["new_baryonforge"]),
            "footprint_checks": footprint_checks,
            "stored_diagnostics": stored,
            "stored_diagnostics_match_recomputed": consistency,
        }
        del arrays

    return {
        "metadata": jsonable(metadata),
        "metadata_consistent": metadata_consistent,
        "baryonforge_source_rows_match_diagnostics": source_row_checks,
        "fields": fields,
        "all_footprints_unchanged": all(
            all(item["footprint_checks"].values()) for item in fields.values()
        ),
        "all_stored_metrics_match_recomputed": all(
            all(
                all(checks.values())
                for checks in item["stored_diagnostics_match_recomputed"].values()
            )
            for item in fields.values()
        ),
    }


def aperture_summary(selection: Mapping[str, np.ndarray]) -> dict[str, Any]:
    log_mass = np.log10(np.asarray(selection["mass_hMsun"], dtype=np.float64))
    isolated = np.asarray(selection["isolated_5R"], dtype=bool)
    output = {}
    for field in ("y", "kappa"):
        godmax = np.asarray(
            selection[f"godmax_{field}_aperture_integral_sr"], dtype=np.float64
        )
        baryonforge = np.asarray(
            selection[f"baryonforge_{field}_aperture_integral_sr"],
            dtype=np.float64,
        )
        ratio = baryonforge / godmax
        bins = []
        for lower, upper in zip(MASS_EDGES[:-1], MASS_EDGES[1:]):
            selected = isolated & (log_mass >= lower) & (log_mass < upper)
            values = ratio[selected]
            if values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"No finite isolated {field} ratios in [{lower}, {upper})."
                )
            bins.append(
                {
                    "lower_log10_hMsun": float(lower),
                    "upper_log10_hMsun": float(upper),
                    "n_isolated": int(values.size),
                    "median": float(np.median(values)),
                    "p16": float(np.percentile(values, 16.0)),
                    "p84": float(np.percentile(values, 84.0)),
                }
            )
        output[field] = bins
    return output


def compare_apertures(
    old_selection: Mapping[str, np.ndarray],
    new_selection: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    old = aperture_summary(old_selection)
    new = aperture_summary(new_selection)
    fields = {}
    for field in ("y", "kappa"):
        bins = []
        for old_bin, new_bin in zip(old[field], new[field]):
            old_distance = abs(math.log(old_bin["median"]))
            new_distance = abs(math.log(new_bin["median"]))
            bins.append(
                {
                    "mass_bin_log10_hMsun": [
                        old_bin["lower_log10_hMsun"],
                        old_bin["upper_log10_hMsun"],
                    ],
                    "n_isolated": old_bin["n_isolated"],
                    "old": old_bin,
                    "new": new_bin,
                    "old_abs_log_median": old_distance,
                    "new_abs_log_median": new_distance,
                    "change_abs_log_median": new_distance - old_distance,
                    "median_closer_to_one": bool(new_distance < old_distance),
                }
            )
        fields[field] = {
            "bins": bins,
            "all_mass_bin_medians_closer_to_one": all(
                item["median_closer_to_one"] for item in bins
            ),
        }
    return {"mass_edges_log10_hMsun": MASS_EDGES, "fields": fields}


def spectra_finiteness(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        group = handle["spectra"]
        axes = {
            name: bool(np.all(np.isfinite(group[name][:])))
            for name in ("ell", "ell_left", "ell_right")
        }
        names = sorted(
            name for name, item in group.items() if isinstance(item, h5py.Group)
        )
        spectra = {}
        for name in names:
            spectra[name] = {
                dataset: bool(np.all(np.isfinite(group[name][dataset][:])))
                for dataset in ("cl", "dell")
            }
        return {
            "count": len(names),
            "names": names,
            "axes_finite": axes,
            "spectra": spectra,
            "all_finite": bool(
                all(axes.values())
                and all(all(item.values()) for item in spectra.values())
            ),
        }


def plot_summary(report: Mapping[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = report["map_comparison"]["fields"]
    apertures = report["aperture_comparison"]["fields"]
    display = report["display"]
    old_label = display["old_label"]
    new_label = display["new_label"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    labels = [
        r"$y$ relative L1",
        r"$y$ $|\Sigma_{BF}/\Sigma_{GM}-1|$",
        r"$\kappa$ relative L1",
        r"$\kappa$ $|\Sigma_{BF}/\Sigma_{GM}-1|$",
    ]
    old_errors = [
        fields["y"]["old"]["relative_l1"],
        abs(fields["y"]["old"]["global_sum_ratio"] - 1.0),
        fields["kappa"]["old"]["relative_l1"],
        abs(fields["kappa"]["old"]["global_sum_ratio"] - 1.0),
    ]
    new_errors = [
        fields["y"]["new"]["relative_l1"],
        abs(fields["y"]["new"]["global_sum_ratio"] - 1.0),
        fields["kappa"]["new"]["relative_l1"],
        abs(fields["kappa"]["new"]["global_sum_ratio"] - 1.0),
    ]
    positions = np.arange(len(labels))
    axes[0, 0].bar(positions - 0.19, old_errors, 0.38, label=old_label)
    axes[0, 0].bar(positions + 0.19, new_errors, 0.38, label=new_label)
    axes[0, 0].set_xticks(positions, labels, rotation=22, ha="right")
    axes[0, 0].set(ylabel="map discrepancy", title="Aggregate map agreement")
    axes[0, 0].legend()

    pearson_labels = [r"$y$", r"$\kappa$"]
    pearson_old = [1.0 - fields[field]["old"]["pearson_r_on_union"] for field in fields]
    pearson_new = [1.0 - fields[field]["new"]["pearson_r_on_union"] for field in fields]
    positions = np.arange(2)
    axes[0, 1].bar(positions - 0.19, pearson_old, 0.38, label=old_label)
    axes[0, 1].bar(positions + 0.19, pearson_new, 0.38, label=new_label)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(positions, pearson_labels)
    axes[0, 1].set(ylabel=r"$1-r_{Pearson}$", title="Pixel-pattern agreement")
    axes[0, 1].legend()
    identity = report["identity_checks"]
    baryonforge_maps_bitwise_unchanged = all(
        item["baryonforge_array_bitwise_unchanged"] for item in fields.values()
    )
    axes[0, 1].text(
        0.98,
        0.95,
        "\n".join(
            (
                f"same geometry: {identity['selection_geometry']} ",
                f"BF arrays bitwise unchanged: {baryonforge_maps_bitwise_unchanged}",
                f"BF-unchanged gate required: {report['expect_baryonforge_unchanged']}",
                f"13 spectra finite: {identity['all_spectra_finite']}",
            )
        ),
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    centers = 0.5 * (MASS_EDGES[:-1] + MASS_EDGES[1:])
    for axis, field in zip(axes[1], ("y", "kappa")):
        for label, key, offset, color, marker in (
            (old_label, "old", -0.035, "tab:blue", "o"),
            (new_label, "new", 0.035, "tab:orange", "s"),
        ):
            summaries = [item[key] for item in apertures[field]["bins"]]
            median = np.asarray([item["median"] for item in summaries])
            p16 = np.asarray([item["p16"] for item in summaries])
            p84 = np.asarray([item["p84"] for item in summaries])
            axis.errorbar(
                centers + offset,
                median,
                yerr=np.vstack([median - p16, p84 - median]),
                color=color,
                marker=marker,
                capsize=3,
                label=label,
            )
        axis.axhline(1.0, color="black", linewidth=1)
        axis.set_xticks(centers, ["13-13.5", "13.5-14", "14-14.5", "14.5-16"])
        axis.set(
            xlabel=r"$\log_{10}(M_{200c}/[M_\odot/h])$",
            ylabel="isolated-aperture BaryonForge / GODMAX",
            title=f"{field} aperture ratios (median, 16--84%)",
        )
        axis.legend(fontsize=8)

    fig.suptitle(display["title"])
    fig.text(
        0.995,
        0.005,
        LABEL,
        ha="right",
        va="bottom",
        fontsize=8,
        color="firebrick",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    outputs = []
    for extension, kwargs in (("png", {"dpi": 180}), ("pdf", {})):
        output = output_dir / f"09_bounded_map_before_after.{extension}"
        temporary = (
            output_dir / f".09_bounded_map_before_after.tmp.{os.getpid()}.{extension}"
        )
        fig.savefig(temporary, bbox_inches="tight", **kwargs)
        os.replace(temporary, output)
        outputs.append(output)
    plt.close(fig)
    return outputs


def build_report(
    old_paths: Mapping[str, Path],
    new_paths: Mapping[str, Path],
    *,
    expect_baryonforge_unchanged: bool,
    old_label: str = "native 8R",
    new_label: str = "candidate 128R",
    comparison_title: str = "Bounded native-map validation: GODMAX 8R to 128R",
    comparison_description: str = (
        "historical native GODMAX 8R versus candidate GODMAX 128R"
    ),
) -> dict[str, Any]:
    old_selection = read_selection(old_paths["diagnostics"])
    new_selection = read_selection(new_paths["diagnostics"])
    geometry = compare_selection_geometry(old_selection, new_selection)
    maps = compare_maps(old_paths, new_paths, old_selection, new_selection)
    apertures = compare_apertures(old_selection, new_selection)
    spectra = {
        "old": spectra_finiteness(old_paths["diagnostics"]),
        "new": spectra_finiteness(new_paths["diagnostics"]),
    }
    same_spectrum_inventory = bool(
        spectra["old"]["count"] == spectra["new"]["count"] == 13
        and spectra["old"]["names"] == spectra["new"]["names"]
    )

    baryonforge_maps_unchanged = all(
        field["baryonforge_array_bitwise_unchanged"]
        for field in maps["fields"].values()
    )
    map_metric_acceptance = {
        field: dict(values["acceptance"]) for field, values in maps["fields"].items()
    }
    aperture_acceptance = {
        field: values["all_mass_bin_medians_closer_to_one"]
        for field, values in apertures["fields"].items()
    }
    identity_checks = {
        "selection_geometry": geometry["all_geometry_datasets_bitwise_equal"],
        "map_metadata": maps["metadata_consistent"],
        "map_source_rows": all(
            maps["baryonforge_source_rows_match_diagnostics"].values()
        ),
        "map_footprints": maps["all_footprints_unchanged"],
        "baryonforge_maps": (
            baryonforge_maps_unchanged if expect_baryonforge_unchanged else True
        ),
        "baryonforge_apertures": (
            geometry["baryonforge_apertures_bitwise_equal"]
            if expect_baryonforge_unchanged
            else True
        ),
        "stored_metrics_match_recomputed": maps["all_stored_metrics_match_recomputed"],
        "same_13_spectrum_inventory": same_spectrum_inventory,
        "all_spectra_finite": all(item["all_finite"] for item in spectra.values()),
    }
    acceptance = {
        "identity_and_null_controls": all(identity_checks.values()),
        "map_metrics": map_metric_acceptance,
        "all_map_metric_checks_pass": all(
            all(checks.values()) for checks in map_metric_acceptance.values()
        ),
        "aperture_mass_bins": aperture_acceptance,
        "all_aperture_mass_bins_improve": all(aperture_acceptance.values()),
    }
    acceptance["ok"] = bool(
        acceptance["identity_and_null_controls"]
        and acceptance["all_map_metric_checks_pass"]
        and acceptance["all_aperture_mass_bins_improve"]
    )
    inputs = {
        f"{run}_{name}": {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for run, paths in (("old", old_paths), ("new", new_paths))
        for name, path in paths.items()
        if name != "root"
    }
    script_path = Path(__file__).resolve()
    inputs["summary_script"] = {
        "path": str(script_path),
        "sha256": sha256_file(script_path),
        "size_bytes": script_path.stat().st_size,
    }
    return {
        "schema": SCHEMA,
        "label": LABEL,
        "comparison": comparison_description,
        "display": {
            "old_label": old_label,
            "new_label": new_label,
            "title": comparison_title,
        },
        "production_statistics_eligible": False,
        "command": [sys.executable, *sys.argv],
        "input_products": inputs,
        "expect_baryonforge_unchanged": expect_baryonforge_unchanged,
        "geometry": geometry,
        "map_comparison": maps,
        "aperture_comparison": apertures,
        "spectra_finiteness": spectra,
        "identity_checks": identity_checks,
        "acceptance": acceptance,
        "ok": acceptance["ok"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-smoke-root", required=True)
    parser.add_argument("--new-smoke-root", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--figure-dir")
    parser.add_argument("--old-label", default="native 8R")
    parser.add_argument("--new-label", default="candidate 128R")
    parser.add_argument(
        "--comparison-title",
        default="Bounded native-map validation: GODMAX 8R to 128R",
    )
    parser.add_argument(
        "--comparison-description",
        default="historical native GODMAX 8R versus candidate GODMAX 128R",
    )
    parser.add_argument(
        "--allow-baryonforge-change",
        action="store_true",
        help="Do not gate on bitwise-identical BaryonForge map/aperture arrays.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    old_root = Path(args.old_smoke_root).expanduser().resolve()
    new_root = Path(args.new_smoke_root).expanduser().resolve()
    old_paths = smoke_paths(old_root)
    new_paths = smoke_paths(new_root)
    require_inputs(old_paths)
    require_inputs(new_paths)
    output = (
        Path(args.output_json or new_root / "bounded_map_before_after.json")
        .expanduser()
        .resolve()
    )
    figure_dir = Path(args.figure_dir or new_root / "figures").expanduser().resolve()
    targets = [
        output,
        figure_dir / "09_bounded_map_before_after.png",
        figure_dir / "09_bounded_map_before_after.pdf",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing outputs: {existing}; pass --overwrite."
        )

    report = build_report(
        old_paths,
        new_paths,
        expect_baryonforge_unchanged=not args.allow_baryonforge_change,
        old_label=args.old_label,
        new_label=args.new_label,
        comparison_title=args.comparison_title,
        comparison_description=args.comparison_description,
    )
    figures = plot_summary(report, figure_dir)
    report["figures"] = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in figures
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(jsonable(report), indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
