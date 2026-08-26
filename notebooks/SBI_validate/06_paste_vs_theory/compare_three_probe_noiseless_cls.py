#!/usr/bin/env python3
"""Compare frozen noiseless pasted maps with input-matched resolved theory.

This is deliberately standalone: it does not modify or monkey-patch GODMAX source.
The immutable headline product is written before any diagnostic interpretation.
"""

from __future__ import annotations

# --- keep imports working from a theme subfolder: common/ holds the
# --- modules shared by more than one stage.
import pathlib as _pl, sys as _sys
_ROOT = _pl.Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "common"):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Mapping

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLBACKEND", "Agg")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import three_probe_noiseless_estimator as estimator  # noqa: E402
import three_probe_noiseless_theory as theory_builder  # noqa: E402


SPECTRA = ("gg", "gy", "gtau", "gkappa")
LABELS = {
    "gg": r"$C_\ell^{gg}$ (shot noise included)",
    "gy": r"$C_\ell^{gy}$",
    "gtau": r"$C_\ell^{g\tau}$",
    "gkappa": r"$C_\ell^{g\kappa_{\rm CMB}}$",
}
COLORS = {"gg": "#1f77b4", "gy": "#d62728", "gtau": "#9467bd", "gkappa": "#2ca02c"}


def sha256_file(path: pathlib.Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_continuous_maps_and_bell(map_path: pathlib.Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with h5py.File(map_path, "r") as handle:
        maps = {
            "y": np.asarray(handle["maps/map_ymap"], dtype=np.float64),
            "tau": np.asarray(handle["maps/map_tau"], dtype=np.float64),
            "kappa": np.asarray(handle["maps/map_kappa_cmb"], dtype=np.float64),
        }
        bell = np.asarray(handle["kernels/profile_smoothing_Bell"], dtype=np.float64)
    return maps, bell


def _infer_supported_map_nside(map_path: pathlib.Path) -> int:
    """Read the map resolution and reject anything outside the frozen control pair."""

    with h5py.File(map_path, "r") as handle:
        map_nside = int(handle.attrs.get("nside", -1))
    if map_nside not in (512, 1024):
        raise ValueError(f"Unsupported final-map nside={map_nside}")
    return map_nside


def evaluate_frozen_acceptance(
    measured: Mapping[str, np.ndarray], predicted: Mapping[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate the pre-registered 5/10 percent criteria without changing support."""

    metrics: dict[str, Any] = {}
    residuals: dict[str, np.ndarray] = {}
    all_pass = True
    for name in SPECTRA:
        mock = np.asarray(measured[name], dtype=np.float64)
        model = np.asarray(predicted[name], dtype=np.float64)
        valid = np.isfinite(mock) & np.isfinite(model) & (model > 0.0)
        residual = np.full_like(model, np.nan)
        residual[valid] = mock[valid] / model[valid] - 1.0
        residuals[name] = residual
        absolute = np.abs(residual[valid])
        adjacent = bool(
            np.any(
                (np.abs(residual[:-1]) > 0.10)
                & (np.abs(residual[1:]) > 0.10)
                & (np.sign(residual[:-1]) == np.sign(residual[1:]))
            )
        )
        family = {
            "n_valid": int(np.count_nonzero(valid)),
            "all_mock_positive": bool(np.all(mock > 0.0)),
            "median_abs_fractional_residual": float(np.median(absolute)) if absolute.size else None,
            "max_abs_fractional_residual": float(np.max(absolute)) if absolute.size else None,
            "all_bands_within_10_percent": bool(absolute.size == len(model) and np.all(absolute <= 0.10)),
            "median_within_5_percent": bool(absolute.size and np.median(absolute) <= 0.05),
            "no_adjacent_coherent_over_10_percent": not adjacent,
        }
        family["pass"] = bool(
            family["all_mock_positive"]
            and family["all_bands_within_10_percent"]
            and family["median_within_5_percent"]
            and family["no_adjacent_coherent_over_10_percent"]
        )
        all_pass &= family["pass"]
        metrics[name] = family
    metrics["overall_pass"] = bool(all_pass)
    metrics["frozen_criteria"] = {
        "support": "native bands 0--11, integer ell 80--1267",
        "maximum_absolute_fractional_residual": 0.10,
        "maximum_family_median_absolute_fractional_residual": 0.05,
        "allow_two_adjacent_coherent_bands_over_10_percent": False,
    }
    return metrics, residuals


def _plot_comparison(
    path: pathlib.Path,
    ell_eff: np.ndarray,
    measured: Mapping[str, np.ndarray],
    predicted: Mapping[str, np.ndarray],
    residuals: Mapping[str, np.ndarray],
    gg_signal: np.ndarray,
    gg_shot: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharex="col", gridspec_kw={"height_ratios": [2.1, 1.0]})
    for column, name in enumerate(SPECTRA):
        upper, lower = axes[0, column], axes[1, column]
        color = COLORS[name]
        upper.plot(ell_eff, measured[name], "o", ms=4.5, color="black", label="pasted mock")
        upper.plot(ell_eff, predicted[name], "-", lw=2.0, color=color, label="matched theory")
        if name == "gg":
            upper.plot(ell_eff, gg_signal, "--", lw=1.2, color="#17becf", label="theory signal")
            upper.plot(ell_eff, gg_shot, ":", lw=1.5, color="#ff7f0e", label="shot noise")
        positive = np.all(measured[name] > 0.0) and np.all(predicted[name] > 0.0)
        upper.set_xscale("log")
        if positive:
            upper.set_yscale("log")
        else:
            upper.set_yscale("symlog", linthresh=1.0e-16)
        upper.set_title(LABELS[name])
        upper.grid(alpha=0.25)
        upper.legend(fontsize=8)
        lower.axhspan(-10.0, 10.0, color="#2ca02c", alpha=0.10)
        lower.axhline(0.0, color="black", lw=0.8)
        lower.plot(ell_eff, 100.0 * residuals[name], "o-", ms=3.5, lw=1.2, color=color)
        lower.set_xscale("log")
        lower.set_ylim(-55.0, 55.0)
        lower.grid(alpha=0.25)
        lower.set_xlabel(r"$\ell_{\rm eff}$")
        if column == 0:
            upper.set_ylabel(r"$C_\ell$")
            lower.set_ylabel(r"$100\,(C_\ell^{\rm mock}/C_\ell^{\rm th}-1)$ [%]")
    fig.suptitle("First matched comparison: current noiseless paste vs projected resolved theory", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_product(
    path: pathlib.Path,
    *,
    mask: np.ndarray,
    mask_metadata: Mapping[str, Any],
    count_report: Mapping[str, int],
    measurement: Mapping[str, Any],
    shot: Mapping[str, Any],
    theory: Mapping[str, Any],
    prediction: Mapping[str, Mapping[str, np.ndarray]],
    residuals: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        handle.attrs["schema_version"] = "sbi_three_probe_noiseless_cl_v1"
        handle.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        handle.attrs["mask_metadata_json"] = json.dumps(mask_metadata, sort_keys=True)
        handle.attrs["galaxy_count_report_json"] = json.dumps(count_report, sort_keys=True)
        handle.attrs["metrics_json"] = json.dumps(metrics, sort_keys=True)
        handle.create_dataset("mask", data=np.asarray(mask, dtype=np.float32), compression="gzip", shuffle=True)
        handle.create_dataset("ell_effective", data=measurement["effective_ell"])
        handle.create_dataset("ell_dense", data=theory["ell"])
        handle.create_dataset("bandpower_window", data=measurement["window"])
        handle.create_dataset("galaxy_shot_coupled", data=shot["coupled"])
        handle.create_dataset("galaxy_shot_decoupled", data=shot["decoupled"])
        handle.create_dataset("theory_redshift", data=theory["redshift"])
        handle.create_dataset("theory_realized_nz", data=theory["realized_nz_on_theory_grid"])
        handle.create_dataset("theory_cmb_efficiency_hmpc", data=theory["cmb_efficiency_hmpc"])
        for name in SPECTRA:
            group = handle.create_group(name)
            group.create_dataset("mock_coupled", data=measurement["coupled"][name])
            group.create_dataset("mock_decoupled_total", data=measurement["decoupled_raw"][name])
            group.create_dataset("theory_intrinsic_dense", data=theory["cls"][name])
            group.create_dataset("theory_spherical_8r_dense", data=theory["spherical_8r_cls"][name])
            group.create_dataset("theory_decoupled_signal", data=prediction[name]["signal"])
            group.create_dataset("theory_decoupled_noise", data=prediction[name]["noise"])
            group.create_dataset("theory_decoupled_total", data=prediction[name]["total"])
            group.create_dataset("fractional_residual", data=residuals[name])
    os.replace(tmp, path)


def _finalize_existing_artifact(
    artifact: pathlib.Path, plot: pathlib.Path, sidecar: pathlib.Path
) -> dict[str, Any]:
    """Render the plot/sidecar after a post-write plotting interruption."""

    if plot.exists() or sidecar.exists():
        raise FileExistsError("Refusing a partial overwrite of immutable first-comparison outputs")
    with h5py.File(artifact, "r") as handle:
        if str(handle.attrs.get("schema_version", "")) != "sbi_three_probe_noiseless_cl_v1":
            raise ValueError("Existing artifact does not have the frozen comparison schema")
        metrics = json.loads(str(handle.attrs["metrics_json"]))
        provenance = json.loads(str(handle.attrs["provenance_json"]))
        mask_metadata = json.loads(str(handle.attrs["mask_metadata_json"]))
        count_report = json.loads(str(handle.attrs["galaxy_count_report_json"]))
        ell_eff = np.asarray(handle["ell_effective"])
        measured = {name: np.asarray(handle[f"{name}/mock_decoupled_total"]) for name in SPECTRA}
        predicted = {name: np.asarray(handle[f"{name}/theory_decoupled_total"]) for name in SPECTRA}
        residuals = {name: np.asarray(handle[f"{name}/fractional_residual"]) for name in SPECTRA}
        gg_signal = np.asarray(handle["gg/theory_decoupled_signal"])
        gg_shot = np.asarray(handle["gg/theory_decoupled_noise"])
    _plot_comparison(plot, ell_eff, measured, predicted, residuals, gg_signal, gg_shot)
    summary = {
        "status": "PASS" if metrics["overall_pass"] else "FAIL_DIAGNOSIS_REQUIRED",
        "artifact": str(artifact.resolve()),
        "artifact_sha256": sha256_file(artifact),
        "plot": str(plot.resolve()),
        "plot_sha256": sha256_file(plot),
        "metrics": metrics,
        "galaxy_count_report": count_report,
        "mask_metadata": mask_metadata,
        "provenance": provenance,
        "finalization_mode": "rendered_from_complete_immutable_hdf5_after_plotting_interruption",
    }
    tmp_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp_sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_sidecar, sidecar)
    return summary


def run(config_path: pathlib.Path, map_path: pathlib.Path, output_dir: pathlib.Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "first_matched_current_paste_vs_projected_theory"
    artifact = output_dir / f"{stem}.h5"
    plot = output_dir / f"{stem}.png"
    sidecar = output_dir / f"{stem}.json"
    if artifact.exists() and not plot.exists() and not sidecar.exists():
        return _finalize_existing_artifact(artifact, plot, sidecar)
    for target in (artifact, plot, sidecar):
        if target.exists():
            raise FileExistsError(f"Immutable first-comparison target already exists: {target}")

    started = time.time()
    map_nside = _infer_supported_map_nside(map_path)
    contract = estimator.validate_final_map_product(map_path, expected_nside=map_nside)
    mask, mask_metadata = estimator.solve_common_c2_cap(nside=map_nside)
    counts, count_report = estimator.build_galaxy_count_map(map_path, nside=map_nside)
    delta_g, mean_count, galaxy_removed_mean = estimator.galaxy_overdensity(counts, mask)
    continuous, bell = _load_continuous_maps_and_bell(map_path)
    maps = {"g": delta_g, **continuous}
    measurement = estimator.make_scalar_namaster_measurement(maps, mask)
    shot = estimator.decoupled_galaxy_shot_template(
        mean_count, mask, measurement["workspace"], nside=map_nside
    )

    theory = theory_builder.build_noiseless_intrinsic_theory(config_path, map_path)
    pixwin = np.asarray(hp.pixwin(map_nside, lmax=estimator.LMAX), dtype=np.float64)
    # The exact projected theory is built from the already-smoothed tables.  Therefore
    # only the galaxy count-map pixel window remains external here.
    embedded_smoothing_transfers = {
        "g": pixwin,
        "y": np.ones_like(pixwin),
        "tau": np.ones_like(pixwin),
        "kappa": np.ones_like(pixwin),
    }
    prediction = estimator.apply_forward_windows(
        measurement["window"],
        theory["cls"],
        embedded_smoothing_transfers,
        galaxy_shot_decoupled=shot["decoupled"],
    )
    measured = measurement["decoupled_raw"]
    predicted_total = {name: prediction[name]["total"] for name in SPECTRA}
    metrics, residuals = evaluate_frozen_acceptance(measured, predicted_total)

    provenance = {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "map_path": str(map_path.resolve()),
        "map_sha256": contract["file_sha256"],
        "nside": map_nside,
        "comparison_script_sha256": sha256_file(pathlib.Path(__file__)),
        "estimator_module_sha256": sha256_file(pathlib.Path(estimator.__file__)),
        "theory_module_sha256": sha256_file(pathlib.Path(theory_builder.__file__)),
        "mask_sha256": mask_metadata["mask_sha256"],
        "transfer_policy": "exact projected theory embeds half-pixel Gaussian; external Tg=HEALPix pixwin; external continuous transfer=unity",
        "profile_smoothing_Bell_sha256": estimator.sha256_array(bell),
        "field_weighted_means": measurement["field_weighted_means"],
        "galaxy_mean_count_per_mask_weighted_pixel": mean_count,
        "galaxy_overdensity_removed_weighted_mean": galaxy_removed_mean,
        "galaxy_shot_full_sky_level_sr": shot["full_sky_level"],
        "theory_provenance": theory["provenance"],
        "wall_seconds": time.time() - started,
    }
    _write_product(
        artifact,
        mask=mask,
        mask_metadata=mask_metadata,
        count_report=count_report,
        measurement=measurement,
        shot=shot,
        theory=theory,
        prediction=prediction,
        residuals=residuals,
        metrics=metrics,
        provenance=provenance,
    )
    _plot_comparison(
        plot,
        measurement["effective_ell"],
        measured,
        predicted_total,
        residuals,
        prediction["gg"]["signal"],
        prediction["gg"]["noise"],
    )
    summary = {
        "status": "PASS" if metrics["overall_pass"] else "FAIL_DIAGNOSIS_REQUIRED",
        "artifact": str(artifact.resolve()),
        "artifact_sha256": sha256_file(artifact),
        "plot": str(plot.resolve()),
        "plot_sha256": sha256_file(plot),
        "metrics": metrics,
        "galaxy_count_report": count_report,
        "mask_metadata": mask_metadata,
        "provenance": provenance,
    }
    tmp_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp_sidecar.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_sidecar, sidecar)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--map", dest="map_path", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.map_path, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
