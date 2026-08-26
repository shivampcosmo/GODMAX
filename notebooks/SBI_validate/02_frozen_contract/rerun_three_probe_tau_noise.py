#!/usr/bin/env python3
"""Versioned tau-only rerun at 1e-3 tau arcmin.

The accepted y and kappa products are immutable null controls.  This script
copies the original contract, replaces only tau N_ell and covariance blocks,
regenerates the twelve tau maps/gtau measurements with identical seeds, and
updates the requested ensemble comparison plot.
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
import json
import os
import pathlib
import shutil
import sys

import h5py
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from survey_defaults import ARCMIN_TO_RAD  # noqa: E402
from three_probe_mock_contract import sha256_array, sha256_file  # noqa: E402
from three_probe_noise_contract import (  # noqa: E402
    BASE_SEED, FIELDS, LMAX, NSIDE, N_REALIZATIONS, PAIR_FIELDS, SPECTRA,
    _synalm_seeded, build_gaussian_covariance, solve_common_c2_cap, subtract_weighted_mean,
    total_observed_cls,
)


TAU_DEPTH_ARCMIN = 1.0e-3
XLIM = (60.0, 2100.0)


def tau_white_noise(depth_arcmin: float = TAU_DEPTH_ARCMIN, lmax: int = LMAX) -> np.ndarray:
    result = np.full(int(lmax) + 1, (float(depth_arcmin) * ARCMIN_TO_RAD) ** 2,
                     dtype=np.float64)
    result[:2] = 0.0
    return result


def tau_subseed(realization: int) -> int:
    if realization < 0 or realization >= N_REALIZATIONS:
        raise ValueError(f"realization must be in [0,{N_REALIZATIONS})")
    return BASE_SEED + int(realization) + 200_000


def amplitude_snr(signal: np.ndarray, covariance: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    return float(np.sqrt(signal @ np.linalg.solve(covariance, signal)))


def solve_tau_depth_matching_gkappa(
    parent: pathlib.Path, reference: pathlib.Path
) -> tuple[float, dict[str, float]]:
    """Solve white tau depth from two exact covariance evaluations."""

    with h5py.File(parent, "r") as low, h5py.File(reference, "r") as high:
        d0 = float(low.attrs["tau_depth_arcmin"])
        d1 = float(high.attrs["tau_depth_arcmin"])
        c0 = np.asarray(low["hmc/covariance"])[28:, 28:]
        c1 = np.asarray(high["hmc/covariance"])[28:, 28:]
        signal_tau = np.asarray(high["theory_bandpowers/gtau"])
        signal_kappa = np.asarray(high["theory_bandpowers/gkappa"])
        covariance_kappa = np.asarray(high["hmc/covariance"])[14:28, 14:28]
    if not 0.0 < d0 < d1:
        raise ValueError("S/N solve requires two increasing positive tau depths")
    slope = (c1 - c0) / (d1**2 - d0**2)
    intercept = c0 - d0**2 * slope
    target = amplitude_snr(signal_kappa, covariance_kappa)

    def residual(depth: float) -> float:
        return amplitude_snr(signal_tau, intercept + depth**2 * slope) - target

    upper = d1
    while residual(upper) > 0.0 and upper < 10.0:
        upper *= 2.0
    if residual(upper) > 0.0:
        raise RuntimeError("Could not bracket the tau depth matching gkappa S/N")
    depth = float(brentq(residual, d1, upper, xtol=1.0e-14, rtol=1.0e-14))
    return depth, {
        "target_gkappa_snr": target,
        "predicted_gtau_snr": amplitude_snr(signal_tau, intercept + depth**2 * slope),
        "low_depth_arcmin": d0,
        "reference_depth_arcmin": d1,
    }


def build_revised_contract(
    parent: pathlib.Path, output: pathlib.Path, tau_depth_arcmin: float = TAU_DEPTH_ARCMIN,
    snr_metadata: dict[str, float] | None = None,
) -> pathlib.Path:
    import pymaster as nmt

    parent_sha = sha256_file(parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    shutil.copy2(parent, tmp)
    with h5py.File(tmp, "r+") as handle:
        if float(handle.attrs["tau_depth_arcmin"]) != 1.0e-5:
            raise ValueError("Parent contract is not the accepted 1e-5 tau product")
        saved_mask = np.asarray(handle["mask"])
        mask, _ = solve_common_c2_cap(nside=NSIDE)
        if not np.array_equal(mask.astype(saved_mask.dtype), saved_mask):
            raise RuntimeError("Regenerated float64 mask does not match saved float32 mask")
        pixwin = np.asarray(handle["pixel_window_g"], dtype=np.float64)
        slice_cls = {name: np.asarray(handle[f"signal_cls/{name}"]) for name in PAIR_FIELDS}
        noise = {
            "y": np.asarray(handle["noise_cls/y_effective"]),
            "kappa": np.asarray(handle["noise_cls/kappa"]),
            "tau": tau_white_noise(tau_depth_arcmin),
        }
        galaxy_shot = float(handle.attrs["galaxy_shot_noise"])
        workspace_path = pathlib.Path(str(handle.attrs["workspace_path"]))
        if sha256_file(workspace_path) != str(handle.attrs["workspace_sha256"]):
            raise ValueError("Parent workspace hash mismatch")
        workspace = nmt.NmtWorkspace.from_file(str(workspace_path))
        # Only the mask enters the covariance workspace.  The zero map is intentional.
        field = nmt.NmtField(mask, [np.zeros_like(mask)], spin=0, n_iter=0, n_iter_mask=0,
                             lmax=LMAX, lmax_mask=LMAX, lite=True)
        total = total_observed_cls(slice_cls, noise, galaxy_shot, pixwin)
        covariance = build_gaussian_covariance(nmt, field, workspace, total)
        old_cov = np.asarray(handle["hmc/covariance"])
        # Tau cannot enter the gy/gkappa principal block algebraically.  Preserve that
        # accepted block byte-for-byte rather than admitting NaMaster recomputation
        # roundoff into a declared null control; all gtau rows/columns remain recomputed.
        covariance[:28, :28] = old_cov[:28, :28]
        covariance = 0.5 * (covariance + covariance.T)
        np.linalg.cholesky(covariance)
        del handle["noise_cls/tau"]
        handle.create_dataset("noise_cls/tau", data=noise["tau"])
        for name in ("covariance", "correlation", "cholesky"):
            del handle[f"hmc/{name}"]
        handle.create_dataset("hmc/covariance", data=covariance)
        diagonal = np.sqrt(np.diag(covariance))
        handle.create_dataset("hmc/correlation", data=covariance / np.outer(diagonal, diagonal))
        handle.create_dataset("hmc/cholesky", data=np.linalg.cholesky(covariance))
        hashes = json.loads(str(handle.attrs["noise_dataset_sha256_json"]))
        hashes["tau"] = sha256_array(noise["tau"])
        handle.attrs["noise_dataset_sha256_json"] = json.dumps(hashes, sort_keys=True)
        handle.attrs["schema_version"] = "sbi_three_probe_noise_contract_tau_effective_v2"
        handle.attrs["tau_depth_arcmin"] = float(tau_depth_arcmin)
        handle.attrs["parent_contract_path"] = str(parent.resolve())
        handle.attrs["parent_contract_sha256"] = parent_sha
        handle.attrs["tau_revision_script_sha256"] = sha256_file(pathlib.Path(__file__))
        handle.attrs["revision_scope"] = "tau noise and covariance blocks only"
        if snr_metadata is not None:
            handle.attrs["tau_snr_match_metadata_json"] = json.dumps(snr_metadata, sort_keys=True)
    os.replace(tmp, output)
    return output


def realize_tau(contract: pathlib.Path, parent_contract: pathlib.Path, signal_map: pathlib.Path,
                output: pathlib.Path, realization: int) -> pathlib.Path:
    import pymaster as nmt

    with h5py.File(contract, "r") as revised, h5py.File(parent_contract, "r") as parent:
        mask = np.asarray(revised["mask"], dtype=np.float64)
        tau_cl = np.asarray(revised["noise_cls/tau"])
        fixed_g = np.asarray(revised["fixed_masked_alm/g"])
        fixed_tau = np.asarray(revised["fixed_masked_alm/tau"])
        workspace_path = pathlib.Path(str(revised.attrs["workspace_path"]))
        if str(revised.attrs["parent_contract_sha256"]) != sha256_file(parent_contract):
            raise ValueError("Revised contract parent hash mismatch")
        old_tau_cl = np.asarray(parent["noise_cls/tau"])
        tau_depth_arcmin = float(revised.attrs["tau_depth_arcmin"])
        parent_depth_arcmin = float(parent.attrs["tau_depth_arcmin"])
    supported = old_tau_cl > 0.0
    expected_power_ratio = (tau_depth_arcmin / parent_depth_arcmin) ** 2
    np.testing.assert_allclose(
        tau_cl[supported] / old_tau_cl[supported], expected_power_ratio,
        rtol=2.0e-15, atol=0.0,
    )
    workspace = nmt.NmtWorkspace.from_file(str(workspace_path))
    noise_alm = _synalm_seeded(tau_cl, tau_subseed(realization))
    noise_map = hp.alm2map(noise_alm, nside=NSIDE, lmax=LMAX)
    centered, removed_mean = subtract_weighted_mean(noise_map, mask)
    masked_noise_alm = hp.map2alm(mask * centered, lmax=LMAX, iter=0)
    coupled = hp.alm2cl(fixed_g, fixed_tau + masked_noise_alm, lmax=LMAX)
    bandpower = np.asarray(workspace.decouple_cell(coupled[None, :]))[0]
    with h5py.File(signal_map, "r") as source:
        noisy_tau = np.asarray(source["maps/map_tau"], dtype=np.float32)
    noisy_tau = noisy_tau + noise_map.astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        handle.attrs.update({
            "schema_version": "sbi_three_probe_tau_effective_realization_v2",
            "contract_path": str(contract.resolve()), "contract_sha256": sha256_file(contract),
            "parent_contract_sha256": sha256_file(parent_contract),
            "realization": int(realization), "tau_depth_arcmin": tau_depth_arcmin,
            "tau_subseed": tau_subseed(realization), "removed_noise_mean": removed_mean,
            "nside": NSIDE, "lmax": LMAX,
        })
        handle.create_dataset("maps/tau", data=noisy_tau, compression="lzf")
        handle.create_dataset("bandpowers/gtau", data=bandpower)
        handle.create_dataset("coupled_cls/gtau", data=coupled)
    os.replace(tmp, output)
    return output


def combine_and_plot(contract: pathlib.Path, parent_contract: pathlib.Path,
                     old_ensemble: pathlib.Path, realization_dir: pathlib.Path,
                     output: pathlib.Path, plot: pathlib.Path) -> pathlib.Path:
    with (
        h5py.File(contract, "r") as revised,
        h5py.File(parent_contract, "r") as parent,
        h5py.File(old_ensemble, "r") as old,
    ):
        ell = np.asarray(revised["effective_ell"])
        theory = {name: np.asarray(revised[f"theory_bandpowers/{name}"]) for name in SPECTRA}
        fixed = {name: np.asarray(revised[f"fixed_bandpowers/{name}"]) for name in SPECTRA}
        covariance = np.asarray(revised["hmc/covariance"])
        old_covariance = np.asarray(parent["hmc/covariance"])
        draws = {"gy": np.asarray(old["draws/gy"]), "gkappa": np.asarray(old["draws/gkappa"])}
        old_gtau = np.asarray(old["draws/gtau"])
    tau_draws = []
    hashes = []
    for realization in range(N_REALIZATIONS):
        path = realization_dir / f"tau_noise_realization_{realization:03d}.h5"
        with h5py.File(path, "r") as handle:
            if int(handle.attrs["realization"]) != realization:
                raise ValueError(f"Realization mismatch in {path}")
            tau_draws.append(np.asarray(handle["bandpowers/gtau"]))
        hashes.append(sha256_file(path))
    draws["gtau"] = np.stack(tau_draws)
    means = {name: np.mean(value, axis=0) for name, value in draws.items()}
    stds = {name: np.std(value, axis=0, ddof=1) for name, value in draws.items()}
    with h5py.File(parent_contract, "r") as parent:
        old_tau = np.asarray(parent["noise_cls/tau"])
    with h5py.File(contract, "r") as revised:
        new_tau = np.asarray(revised["noise_cls/tau"])
        tau_depth_arcmin = float(revised.attrs["tau_depth_arcmin"])
    supported = old_tau > 0.0
    tau_power_ratio = new_tau[supported] / old_tau[supported]
    # Same seeds imply exact linear scaling of the gtau noise perturbation.
    old_fixed = fixed["gtau"][None, :]
    old_perturbation = old_gtau - old_fixed
    new_perturbation = draws["gtau"] - old_fixed
    scale_ratio = new_perturbation / old_perturbation
    with h5py.File(parent_contract, "r") as parent:
        parent_depth_arcmin = float(parent.attrs["tau_depth_arcmin"])
    expected_amplitude_scale = tau_depth_arcmin / parent_depth_arcmin
    scale_residual = new_perturbation - expected_amplitude_scale * old_perturbation
    if not np.all(np.isfinite(scale_ratio)):
        raise RuntimeError("Same-seed gtau scaling diagnostic is non-finite")
    if not np.array_equal(covariance[:28, :28], old_covariance[:28, :28]):
        raise RuntimeError("Tau-only rerun failed an unchanged control")
    vectors = np.concatenate([draws[name] for name in SPECTRA], axis=1)
    sample_cov = np.cov(vectors, rowvar=False, ddof=1)
    summary = {
        "tau_depth_arcmin": tau_depth_arcmin,
        "tau_noise_power_ratio_min": float(np.min(tau_power_ratio)),
        "tau_noise_power_ratio_max": float(np.max(tau_power_ratio)),
        "gtau_perturbation_scale_median": float(np.median(scale_ratio)),
        "expected_gtau_amplitude_scale": expected_amplitude_scale,
        "gtau_perturbation_scale_max_abs_error": float(
            np.max(np.abs(scale_ratio - expected_amplitude_scale))
        ),
        "gtau_scaling_max_absolute_cl_residual": float(np.max(np.abs(scale_residual))),
        "gtau_scaling_global_relative_residual": float(
            np.max(np.abs(scale_residual)) / np.max(np.abs(new_perturbation))
        ),
        "sample_covariance_rank": int(np.linalg.matrix_rank(sample_cov)),
        "xlim": list(XLIM), "contract_sha256": sha256_file(contract),
        "parent_contract_sha256": sha256_file(parent_contract),
        "old_ensemble_sha256": sha256_file(old_ensemble),
        "realization_sha256": hashes, "script_sha256": sha256_file(pathlib.Path(__file__)),
    }
    snr_values = {}
    for index, name in enumerate(SPECTRA):
        block = covariance[index * 14:(index + 1) * 14, index * 14:(index + 1) * 14]
        snr_values[name] = amplitude_snr(theory[name], block)
    summary["forecast_amplitude_snr"] = snr_values
    summary["gtau_to_gkappa_snr_ratio"] = snr_values["gtau"] / snr_values["gkappa"]
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        handle.attrs["schema_version"] = "sbi_three_probe_noisy_ensemble_tau_effective_v2"
        handle.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        handle.create_dataset("effective_ell", data=ell)
        handle.create_dataset("hmc_covariance", data=covariance)
        handle.create_dataset("sample_covariance_diagnostic", data=sample_cov)
        for name in SPECTRA:
            handle.create_dataset(f"draws/{name}", data=draws[name])
            handle.create_dataset(f"mean/{name}", data=means[name])
            handle.create_dataset(f"std/{name}", data=stds[name])
            handle.create_dataset(f"theory/{name}", data=theory[name])
            handle.create_dataset(f"fixed_noiseless_mock/{name}", data=fixed[name])
    os.replace(tmp, output)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    labels = {"gy": r"$g\times y$", "gkappa": r"$g\times\kappa_{\rm CMB}$",
              "gtau": r"$g\times\tau$"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5),
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    for column, name in enumerate(SPECTRA):
        top, residual = axes[:, column]
        top.errorbar(ell, means[name], yerr=stds[name], fmt="o", ms=3.5, capsize=2,
                     label="12 noisy mocks: mean $\\pm1\\sigma$")
        top.plot(ell, theory[name], color="C3", lw=1.8, label="matched resolved theory")
        top.plot(ell, fixed[name], color="0.35", lw=1.2, ls="--",
                 label="fixed noiseless pasted map")
        top.set_xscale("log"); top.set_yscale("log"); top.set_xlim(*XLIM)
        top.set_title(labels[name]); top.grid(alpha=0.25)
        ratio = means[name] / theory[name] - 1.0
        ratio_error = stds[name] / np.abs(theory[name])
        residual.errorbar(ell, 100.0 * ratio, yerr=100.0 * ratio_error,
                          fmt="o", ms=3.5, capsize=2)
        residual.axhline(0.0, color="k", lw=1)
        residual.axhspan(-10.0, 10.0, color="C2", alpha=0.12)
        residual.set_xscale("log"); residual.set_xlim(*XLIM)
        residual.set_xlabel(r"$\ell$"); residual.set_ylabel("mock/theory - 1 [%]")
        residual.grid(alpha=0.25)
    axes[0, 0].set_ylabel(r"$C_b$"); axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        rf"nside=1024 noisy mocks; $\Delta_\tau={tau_depth_arcmin:.5g}\,\tau$ arcmin"
    )
    fig.tight_layout()
    archive = plot.with_name("noisy_mock_mean_vs_theory_tau1e-3.png")
    if plot.exists() and not archive.exists():
        shutil.copy2(plot, archive)
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-contract", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--signal-map", type=pathlib.Path, required=True)
    parser.add_argument("--old-ensemble", type=pathlib.Path, required=True)
    parser.add_argument("--realization-dir", type=pathlib.Path, required=True)
    parser.add_argument("--ensemble", type=pathlib.Path, required=True)
    parser.add_argument("--plot", type=pathlib.Path, required=True)
    parser.add_argument("--tau-depth-arcmin", type=float, default=TAU_DEPTH_ARCMIN)
    parser.add_argument("--match-gkappa-snr-reference-contract", type=pathlib.Path)
    args = parser.parse_args()
    tau_depth = float(args.tau_depth_arcmin)
    snr_metadata = None
    if args.match_gkappa_snr_reference_contract is not None:
        tau_depth, snr_metadata = solve_tau_depth_matching_gkappa(
            args.parent_contract, args.match_gkappa_snr_reference_contract
        )
        print(f"S/N-matched tau depth: {tau_depth:.17g} tau arcmin")
    build_revised_contract(args.parent_contract, args.contract, tau_depth, snr_metadata)
    for realization in range(N_REALIZATIONS):
        realize_tau(args.contract, args.parent_contract, args.signal_map,
                    args.realization_dir / f"tau_noise_realization_{realization:03d}.h5",
                    realization)
    print(combine_and_plot(args.contract, args.parent_contract, args.old_ensemble,
                           args.realization_dir, args.ensemble, args.plot))


if __name__ == "__main__":
    main()
