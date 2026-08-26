#!/usr/bin/env python3
"""Windowed noiseless nside-1024 pasted-map/theory comparison through ell=2048."""

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
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import three_probe_noiseless_estimator as estimator  # noqa: E402
import three_probe_noiseless_theory as theory_builder  # noqa: E402

NSIDE = 1024
LMAX = 2048
# Native logarithmic bands 12 and 13 are complete; band 14 is cut at literal ell=2048.
BAND_EDGES = np.concatenate(
    (estimator.NATIVE_12_EDGES, np.asarray([1597, 2010, 2049], dtype=np.int64))
)
SPECTRA = estimator.SPECTRUM_FIELDS


def sha256_file(path: pathlib.Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_bins(nmt_module):
    ell = np.arange(LMAX + 1, dtype=np.int32)
    bpws = np.full(ell.shape, -1, dtype=np.int32)
    for band, (left, right) in enumerate(zip(BAND_EDGES[:-1], BAND_EDGES[1:])):
        bpws[(ell >= int(left)) & (ell < int(right))] = band
    return nmt_module.NmtBin(ells=ell, bpws=bpws, lmax=LMAX)


def measure_cls(
    maps: Mapping[str, np.ndarray], mask: np.ndarray, *, nmt_module=None
) -> dict[str, Any]:
    if nmt_module is None:
        import pymaster as nmt_module

    weight = np.asarray(mask, dtype=np.float64)
    centered, means = {}, {}
    for name in ("g", "y", "tau", "kappa"):
        centered[name], means[name] = estimator.subtract_weighted_mean(maps[name], weight)
    bins = make_bins(nmt_module)
    fields = {
        name: nmt_module.NmtField(
            weight,
            [centered[name]],
            spin=0,
            beam=None,
            n_iter=0,
            n_iter_mask=0,
            lmax=LMAX,
            lmax_mask=LMAX,
            lite=True,
            masked_on_input=False,
        )
        for name in centered
    }
    workspace = nmt_module.NmtWorkspace.from_fields(
        fields["g"], fields["g"], bins, l_toeplitz=-1, l_exact=-1, dl_band=-1
    )
    windows_all = np.asarray(workspace.get_bandpower_windows(), dtype=np.float64)
    expected_shape = (1, BAND_EDGES.size - 1, 1, LMAX + 1)
    if windows_all.shape != expected_shape:
        raise RuntimeError(f"Unexpected bandpower-window shape {windows_all.shape}")
    coupled, decoupled = {}, {}
    for name, (left, right) in SPECTRA.items():
        pcl = np.asarray(
            nmt_module.compute_coupled_cell(fields[left], fields[right]), dtype=np.float64
        )
        bpw = np.asarray(workspace.decouple_cell(pcl), dtype=np.float64)
        if pcl.shape != (1, LMAX + 1) or bpw.shape != (1, BAND_EDGES.size - 1):
            raise RuntimeError(f"Invalid {name} spectrum shapes: {pcl.shape}, {bpw.shape}")
        coupled[name], decoupled[name] = pcl[0], bpw[0]
    return {
        "workspace": workspace,
        "window": windows_all[0, :, 0, :],
        "windows_all": windows_all,
        "effective_ell": np.asarray(bins.get_effective_ells(), dtype=np.float64),
        "coupled": coupled,
        "decoupled": decoupled,
        "weighted_means": means,
    }


def residual_summary(
    measured: Mapping[str, np.ndarray], predicted: Mapping[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary, residuals = {}, {}
    for name in SPECTRA:
        residual = np.asarray(measured[name]) / np.asarray(predicted[name]) - 1.0
        residuals[name] = residual
        summary[name] = {
            "median_abs_original_12": float(np.median(np.abs(residual[:12]))),
            "max_abs_original_12": float(np.max(np.abs(residual[:12]))),
            "band_1268_1596_residual": float(residual[12]),
            "band_1597_2009_residual": float(residual[13]),
            "partial_band_2010_2048_residual": float(residual[14]),
        }
    return summary, residuals


def plot_comparison(
    path: pathlib.Path,
    ell_eff: np.ndarray,
    measured: Mapping[str, np.ndarray],
    predicted: Mapping[str, np.ndarray],
    residuals: Mapping[str, np.ndarray],
) -> None:
    colors = {"gg": "#1f77b4", "gy": "#d62728", "gtau": "#9467bd", "gkappa": "#2ca02c"}
    fig, axes = plt.subplots(
        2, 4, figsize=(18, 7.5), sharex="col", gridspec_kw={"height_ratios": [2.1, 1.0]}
    )
    for column, name in enumerate(SPECTRA):
        top, bottom = axes[:, column]
        top.loglog(ell_eff, measured[name], "ko", ms=4.0, label="pasted mock")
        top.loglog(ell_eff, predicted[name], "-", color=colors[name], lw=1.8, label="resolved theory")
        top.axvspan(2010, 2048, color="black", alpha=0.08, label="partial native band")
        top.set_title(name)
        top.grid(alpha=0.25)
        top.legend(fontsize=7)
        bottom.axhspan(-10.0, 10.0, color="#2ca02c", alpha=0.10)
        bottom.axhline(0.0, color="black", lw=0.8)
        bottom.plot(ell_eff, 100.0 * residuals[name], "o-", color=colors[name], ms=3.2)
        bottom.axvspan(2010, 2048, color="black", alpha=0.08)
        bottom.set_xscale("log")
        bottom.grid(alpha=0.25)
        bottom.set_xlabel(r"$\ell_{\rm eff}$")
        if column == 0:
            top.set_ylabel(r"$C_\ell$")
            bottom.set_ylabel("mock/theory - 1 [%]")
    fig.suptitle(
        "nside=1024 noiseless pasted maps vs input-matched resolved theory, ell_max=2048",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(
    config_path: pathlib.Path,
    map_path: pathlib.Path,
    old_artifact: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = output_dir / "nside1024_ell2048_paste_vs_projected_theory.h5"
    output_png = output_dir / "nside1024_ell2048_paste_vs_projected_theory.png"
    output_json = output_dir / "nside1024_ell2048_paste_vs_projected_theory.json"
    if any(path.exists() for path in (output_h5, output_png, output_json)):
        raise FileExistsError("Ell=2048 comparison outputs are immutable")

    started = time.time()
    contract = estimator.validate_final_map_product(
        map_path, expected_nside=NSIDE, expected_lmax=1535
    )
    mask, mask_metadata = estimator.solve_common_c2_cap(nside=NSIDE)
    counts, count_report = estimator.build_galaxy_count_map(map_path, nside=NSIDE)
    delta_g, mean_count, removed_mean = estimator.galaxy_overdensity(counts, mask)
    with h5py.File(map_path, "r") as handle:
        attrs = dict(handle.attrs)
        kernels = handle["kernels"]
        sigma_rad = float(kernels.attrs["profile_smoothing_sigma_rad"])
        saved_ell = np.asarray(kernels["profile_smoothing_ell"], dtype=np.int64)
        saved_bell = np.asarray(kernels["profile_smoothing_Bell"], dtype=np.float64)
        maps = {
            "g": delta_g,
            "y": np.asarray(handle["maps/map_ymap"], dtype=np.float64),
            "tau": np.asarray(handle["maps/map_tau"], dtype=np.float64),
            "kappa": np.asarray(handle["maps/map_kappa_cmb"], dtype=np.float64),
        }
    if not np.array_equal(saved_ell, np.arange(1536)):
        raise ValueError("Map smoothing kernel does not have frozen ell=0..1535 support")
    ell_dense = np.arange(LMAX + 1, dtype=np.float64)
    bell_extended = np.exp(-0.5 * (ell_dense * sigma_rad) ** 2)
    np.testing.assert_allclose(bell_extended[:1536], saved_bell, rtol=2.0e-15, atol=0.0)

    measurement = measure_cls(maps, mask)
    shot = estimator.decoupled_galaxy_shot_template(
        mean_count, mask, measurement["workspace"], nside=NSIDE, lmax=LMAX
    )
    theory = theory_builder.build_noiseless_intrinsic_theory(
        config_path, map_path, ell_max=LMAX
    )
    pixel_window = np.asarray(hp.pixwin(NSIDE, lmax=LMAX), dtype=np.float64)
    transfers = {
        "g": pixel_window,
        "y": np.ones(LMAX + 1),
        "tau": np.ones(LMAX + 1),
        "kappa": np.ones(LMAX + 1),
    }
    prediction = estimator.apply_forward_windows(
        measurement["window"],
        theory["cls"],
        transfers,
        galaxy_shot_decoupled=shot["decoupled"],
    )
    predicted = {name: prediction[name]["total"] for name in SPECTRA}
    summary, residuals = residual_summary(measurement["decoupled"], predicted)

    with h5py.File(old_artifact, "r") as handle:
        old_mock = {name: np.asarray(handle[f"{name}/mock_decoupled_total"]) for name in SPECTRA}
        old_theory = {name: np.asarray(handle[f"{name}/theory_decoupled_total"]) for name in SPECTRA}
    old_band_null = {
        name: {
            "mock_max_fractional_change": float(
                np.max(np.abs(measurement["decoupled"][name][:12] / old_mock[name] - 1.0))
            ),
            "theory_max_fractional_change": float(
                np.max(np.abs(predicted[name][:12] / old_theory[name] - 1.0))
            ),
        }
        for name in SPECTRA
    }
    if any(
        value > 0.005
        for family in old_band_null.values()
        for value in family.values()
    ):
        raise RuntimeError(f"Ell extension moved an original band by more than 0.5%: {old_band_null}")

    plot_comparison(output_png, measurement["effective_ell"], measurement["decoupled"], predicted, residuals)
    provenance = {
        "map": str(map_path.resolve()),
        "map_sha256": contract["file_sha256"],
        "old_artifact": str(old_artifact.resolve()),
        "old_artifact_sha256": sha256_file(old_artifact),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(pathlib.Path(__file__)),
        "estimator_sha256": sha256_file(pathlib.Path(estimator.__file__)),
        "theory_builder_sha256": sha256_file(pathlib.Path(theory_builder.__file__)),
        "catalog_cosmology_sha256": str(contract["catalog_cosmology_sha256"]),
        "catalog_file_sha256": str(contract["catalog_file_sha256"]),
        "realized_hod_nz_sha256": str(contract["kernel_dataset_sha256"]["realized_hod_galaxy_nz"]),
        "realized_hod_nz_normalization": float(contract["realized_hod_nz_normalization"]),
        "nside": NSIDE,
        "lmax": LMAX,
        "map_declared_comparison_lmax": int(attrs["comparison_lmax"]),
        "band_policy": "native integer bands through 2009 plus partial [2010,2049)",
        "partial_final_band": True,
        "profile_smoothing": {
            "method": str(attrs["profile_smoothing_method"]),
            "fwhm_arcmin": float(attrs["profile_smoothing_fwhm_arcmin"]),
            "sigma_rad": sigma_rad,
            "extended_Bell_sha256": estimator.sha256_array(bell_extended),
            "application": "embedded in projected y/e/m tables; not multiplied again",
        },
        "galaxy_transfer": "HEALPix pixel window applied once",
        "mask": mask_metadata,
        "count_report": count_report,
        "weighted_means": measurement["weighted_means"],
        "galaxy_removed_mean": removed_mean,
        "theory_provenance": theory["provenance"],
        "wall_seconds": time.time() - started,
    }

    tmp_h5 = output_h5.with_suffix(".h5.tmp")
    with h5py.File(tmp_h5, "w") as handle:
        handle.attrs["schema_version"] = "sbi_three_probe_nside1024_ell2048_v1"
        handle.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        handle.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        handle.attrs["old_band_null_json"] = json.dumps(old_band_null, sort_keys=True)
        handle.create_dataset("ell_dense", data=np.arange(LMAX + 1, dtype=np.int64))
        handle.create_dataset("ell_effective", data=measurement["effective_ell"])
        handle.create_dataset("band_edges", data=BAND_EDGES)
        handle.create_dataset("bandpower_window", data=measurement["window"])
        handle.create_dataset("profile_smoothing_Bell_extended", data=bell_extended)
        handle.create_dataset("galaxy_pixel_window", data=pixel_window)
        handle.create_dataset("galaxy_shot_decoupled", data=shot["decoupled"])
        for name in SPECTRA:
            group = handle.create_group(name)
            group.create_dataset("mock_coupled", data=measurement["coupled"][name])
            group.create_dataset("mock_decoupled_total", data=measurement["decoupled"][name])
            group.create_dataset("theory_intrinsic", data=theory["cls"][name])
            group.create_dataset("theory_decoupled_signal", data=prediction[name]["signal"])
            group.create_dataset("theory_decoupled_noise", data=prediction[name]["noise"])
            group.create_dataset("theory_decoupled_total", data=predicted[name])
            group.create_dataset("fractional_residual", data=residuals[name])
    os.replace(tmp_h5, output_h5)

    result = {
        "status": "DIAGNOSTIC_COMPLETE",
        "artifact": str(output_h5.resolve()),
        "artifact_sha256": sha256_file(output_h5),
        "plot": str(output_png.resolve()),
        "plot_sha256": sha256_file(output_png),
        "summary": summary,
        "old_band_null": old_band_null,
        "provenance": provenance,
    }
    tmp_json = output_json.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_json, output_json)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--map", dest="map_path", type=pathlib.Path, required=True)
    parser.add_argument("--old-artifact", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.map_path, args.old_artifact, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
