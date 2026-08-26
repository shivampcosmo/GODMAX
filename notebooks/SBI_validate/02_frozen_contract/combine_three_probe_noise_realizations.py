#!/usr/bin/env python3
"""Combine the twelve frozen noisy realizations and plot theory versus mean±1sigma."""

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
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from three_probe_mock_contract import sha256_file  # noqa: E402
from three_probe_noise_contract import N_REALIZATIONS, SPECTRA  # noqa: E402


def combine(contract: pathlib.Path, input_dir: pathlib.Path, output: pathlib.Path,
            plot: pathlib.Path) -> pathlib.Path:
    contract_sha = sha256_file(contract)
    with h5py.File(contract, "r") as handle:
        ell = np.asarray(handle["effective_ell"])
        theory = {name: np.asarray(handle[f"theory_bandpowers/{name}"]) for name in SPECTRA}
        fixed = {name: np.asarray(handle[f"fixed_bandpowers/{name}"]) for name in SPECTRA}
        covariance = np.asarray(handle["hmc/covariance"])
        contract_attrs = dict(handle.attrs)
    draws = {name: [] for name in SPECTRA}
    input_hashes = []
    for realization in range(N_REALIZATIONS):
        path = input_dir / f"noise_realization_{realization:03d}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            if int(handle.attrs["realization"]) != realization:
                raise ValueError(f"Realization index mismatch in {path}")
            if str(handle.attrs["contract_sha256"]) != contract_sha:
                raise ValueError(f"Contract hash mismatch in {path}")
            for name in SPECTRA:
                value = np.asarray(handle[f"bandpowers/{name}"])
                if value.shape != ell.shape or not np.all(np.isfinite(value)):
                    raise ValueError(f"Invalid {name} bandpowers in {path}")
                draws[name].append(value)
        input_hashes.append(sha256_file(path))
    arrays = {name: np.stack(values) for name, values in draws.items()}
    means = {name: np.mean(value, axis=0) for name, value in arrays.items()}
    stds = {name: np.std(value, axis=0, ddof=1) for name, value in arrays.items()}
    vectors = np.concatenate([arrays[name] for name in SPECTRA], axis=1)
    sample_cov = np.cov(vectors, rowvar=False, ddof=1)
    sample_rank = int(np.linalg.matrix_rank(sample_cov))
    studentized = {
        name: (means[name] - fixed[name]) / (stds[name] / np.sqrt(N_REALIZATIONS))
        for name in SPECTRA
    }
    all_t = np.concatenate(list(studentized.values()))
    summary = {
        "combine_script_sha256": sha256_file(pathlib.Path(__file__)),
        "contract_sha256": contract_sha,
        "input_sha256": input_hashes,
        "n_realizations": N_REALIZATIONS,
        "sample_covariance_rank": sample_rank,
        "sample_covariance_policy": "diagnostic_only_not_HMC",
        "median_abs_studentized_mean_vs_fixed_signal": float(np.median(np.abs(all_t))),
        "max_abs_studentized_mean_vs_fixed_signal": float(np.max(np.abs(all_t))),
        "hmc_covariance_shape": list(covariance.shape),
        "spectrum_order": list(SPECTRA),
        "contract_vector_order": str(contract_attrs["vector_order"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        handle.attrs["schema_version"] = "sbi_three_probe_noisy_ensemble_v1"
        handle.attrs["summary_json"] = json.dumps(summary, sort_keys=True)
        handle.create_dataset("effective_ell", data=ell)
        handle.create_dataset("sample_covariance_diagnostic", data=sample_cov)
        handle.create_dataset("hmc_covariance", data=covariance)
        for name in SPECTRA:
            handle.create_dataset(f"draws/{name}", data=arrays[name])
            handle.create_dataset(f"mean/{name}", data=means[name])
            handle.create_dataset(f"std/{name}", data=stds[name])
            handle.create_dataset(f"theory/{name}", data=theory[name])
            handle.create_dataset(f"fixed_noiseless_mock/{name}", data=fixed[name])
            handle.create_dataset(f"studentized_mean/{name}", data=studentized[name])
    os.replace(tmp, output)

    labels = {"gy": r"$g\times y$", "gkappa": r"$g\times\kappa_{\rm CMB}$",
              "gtau": r"$g\times\tau$"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    for column, name in enumerate(SPECTRA):
        ax, residual = axes[:, column]
        ax.errorbar(ell, means[name], yerr=stds[name], fmt="o", ms=3.5,
                    capsize=2, label="12 noisy mocks: mean $\pm1\sigma$")
        ax.plot(ell, theory[name], color="C3", lw=1.8, label="matched resolved theory")
        ax.plot(ell, fixed[name], color="0.35", lw=1.2, ls="--",
                label="fixed noiseless pasted map")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(labels[name])
        ax.grid(alpha=0.25)
        ratio = means[name] / theory[name] - 1.0
        ratio_err = stds[name] / np.abs(theory[name])
        residual.errorbar(ell, 100.0 * ratio, yerr=100.0 * ratio_err,
                          fmt="o", ms=3.5, capsize=2)
        residual.axhline(0.0, color="k", lw=1)
        residual.axhspan(-10.0, 10.0, color="C2", alpha=0.12)
        residual.set_xscale("log")
        residual.set_xlabel(r"$\ell$")
        residual.set_ylabel("mock/theory - 1 [%]")
        residual.grid(alpha=0.25)
    axes[0, 0].set_ylabel(r"$C_b$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("nside=1024 noisy pasted mocks and matched covariance/noise theory")
    fig.tight_layout()
    plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot, dpi=180)
    plt.close(fig)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--plot", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(combine(args.contract, args.input_dir, args.output, args.plot))


if __name__ == "__main__":
    main()
