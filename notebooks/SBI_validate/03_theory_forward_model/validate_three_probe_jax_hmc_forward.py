#!/usr/bin/env python3
"""Run the HMC forward callable once and compare it with the frozen mock catalog."""

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
import pathlib
import time

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

from three_probe_inference_contract import DEFAULT_CONTRACT_PATH, load_training_contract
from three_probe_jax_forward_model import PARAMETER_NAMES, make_three_probe_forward_model


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
AUDIT_MANIFEST = REPO_ROOT / "data/SBI_validate/three_probe_inference/experiment_manifest.yaml"
REFERENCE = REPO_ROOT / (
    "data/SBI_validate/three_probe_mock/validation/noiseless_cls_gate3_nside1024/ell2048/"
    "nside1024_ell2048_paste_vs_projected_theory.h5"
)
REFERENCE_SHA256 = "4f21e58884b80cd5247dfc5d4c2b1d8cb84b0170a03c4e4c27016b9bcdcc8662"
SPECTRA = ("gy", "gkappa", "gtau")
LABELS = (r"$g\times y$", r"$g\times\kappa$", r"$g\times\tau$")
NONREGRESSION_TOLERANCE = 0.005


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dense-radius-nodes", type=int, default=256)
    parser.add_argument("--profile-nr", type=int, default=48)
    parser.add_argument("--profile-nz", type=int, default=48)
    parser.add_argument("--limber-ell-nodes", type=int, default=2049)
    args = parser.parse_args()
    if sha256_file(REFERENCE) != REFERENCE_SHA256:
        raise ValueError("Frozen mock-comparison reference hash mismatch")
    contract = load_training_contract(DEFAULT_CONTRACT_PATH)
    with AUDIT_MANIFEST.open() as handle:
        audit = yaml.safe_load(handle)
    sampled = audit["parameters"]["sampled"]
    if tuple(item["name"] for item in sampled) != PARAMETER_NAMES:
        raise ValueError("Audit parameter order differs from the HMC callable")
    theta = np.asarray([item["truth"] for item in sampled], dtype=np.float64)

    start = time.time()
    model = make_three_probe_forward_model(
        DEFAULT_CONTRACT_PATH,
        dense_radius_nodes=args.dense_radius_nodes,
        profile_nr=args.profile_nr,
        profile_nz=args.profile_nz,
        limber_ell_nodes=args.limber_ell_nodes,
        jit_compile=True,
    )
    jax_vector = np.asarray(model.vector_fn(theta), dtype=np.float64)
    jax_vector.block if hasattr(jax_vector, "block") else None
    wall_seconds = time.time() - start
    if jax_vector.shape != (42,) or not np.all(np.isfinite(jax_vector)):
        raise ValueError("HMC forward evaluation did not produce a finite 42-vector")

    with h5py.File(REFERENCE, "r") as handle:
        ell = np.asarray(handle["ell_effective"], dtype=np.float64)[:14]
        host = {name: np.asarray(handle[f"{name}/theory_decoupled_signal"], dtype=np.float64)[:14] for name in SPECTRA}
        mock = {name: np.asarray(handle[f"{name}/mock_decoupled_total"], dtype=np.float64)[:14] for name in SPECTRA}
    jax_bands = {name: jax_vector[14 * index:14 * (index + 1)] for index, name in enumerate(SPECTRA)}
    residual = contract.data_vector - jax_vector
    whitened = np.linalg.solve(contract.cholesky, residual)
    chi2_full = float(whitened @ whitened)
    chi2_by_probe = {}
    for index, name in enumerate(SPECTRA):
        selection = slice(14 * index, 14 * (index + 1))
        covariance_block = contract.covariance[selection, selection]
        chi2_by_probe[name] = float(
            residual[selection] @ np.linalg.solve(covariance_block, residual[selection])
        )

    summary = {}
    for name in SPECTRA:
        jax_host = jax_bands[name] / host[name] - 1.0
        host_mock = mock[name] / host[name] - 1.0
        jax_mock = mock[name] / jax_bands[name] - 1.0
        summary[name] = {
            "jax_vs_host_max_abs_fraction": float(np.max(np.abs(jax_host))),
            "jax_vs_host_median_abs_fraction": float(np.median(np.abs(jax_host))),
            "mock_residual_replay_max_abs_change": float(np.max(np.abs(jax_mock - host_mock))),
            "mock_vs_jax_median_abs_fraction": float(np.median(np.abs(jax_mock))),
        }
    gate = all(
        value["jax_vs_host_max_abs_fraction"] <= NONREGRESSION_TOLERANCE
        and value["mock_residual_replay_max_abs_change"] <= NONREGRESSION_TOLERANCE
        for value in summary.values()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "jax_hmc_forward_vs_mock_three_cls"
    factor = ell * (ell + 1.0) / (2.0 * np.pi)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5), sharex="col", gridspec_kw={"height_ratios": [2.2, 1.0]})
    for index, (name, label) in enumerate(zip(SPECTRA, LABELS)):
        axes[0, index].plot(ell, factor * mock[name], "o", ms=4, label="mock catalog")
        axes[0, index].plot(ell, factor * host[name], lw=2, label="host projected theory")
        axes[0, index].plot(ell, factor * jax_bands[name], "--", lw=2, label="JAX HMC callable")
        axes[0, index].set_xscale("log")
        axes[0, index].set_yscale("log")
        axes[0, index].set_title(label)
        axes[0, index].set_ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
        axes[1, index].axhline(0.0, color="0.5", lw=1)
        axes[1, index].plot(ell, mock[name] / jax_bands[name] - 1.0, "o-", ms=3, label="mock/JAX - 1")
        axes[1, index].plot(ell, host[name] / jax_bands[name] - 1.0, "--", lw=1.5, label="host/JAX - 1")
        axes[1, index].set_xscale("log")
        axes[1, index].set_xlabel(r"$\ell_{\rm eff}$")
        axes[1, index].set_ylabel("fractional residual")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Single evaluation of the exact JAX HMC forward model vs frozen noiseless mock")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    plt.close(fig)

    with h5py.File(stem.with_suffix(".h5"), "w") as handle:
        handle.attrs["status"] = "PASS" if gate else "FAIL"
        handle.attrs["contract_sha256"] = contract.contract_sha256
        handle.attrs["reference_sha256"] = REFERENCE_SHA256
        handle.attrs["model_metadata_json"] = json.dumps(model.metadata, sort_keys=True)
        handle.attrs["absolute_chi2_at_audit_point"] = chi2_full
        handle.attrs["marginal_block_chi2_json"] = json.dumps(chi2_by_probe, sort_keys=True)
        handle.create_dataset("theta_audit_only", data=theta)
        handle.create_dataset("effective_ell", data=ell)
        handle.create_dataset("jax_vector", data=jax_vector)
        for name in SPECTRA:
            group = handle.create_group(name)
            group.create_dataset("jax_hmc_callable", data=jax_bands[name])
            group.create_dataset("host_projected_theory", data=host[name])
            group.create_dataset("mock_catalog", data=mock[name])
    payload = {
        "status": "PASS" if gate else "FAIL",
        "wall_seconds_including_compile": wall_seconds,
        "nonregression_tolerance": NONREGRESSION_TOLERANCE,
        "contract_sha256": contract.contract_sha256,
        "reference": str(REFERENCE),
        "reference_sha256": REFERENCE_SHA256,
        "model_metadata": model.metadata,
        "summary": summary,
        "absolute_chi2_at_audit_point": chi2_full,
        "marginal_block_chi2_at_audit_point": chi2_by_probe,
        "retained_rank": 42,
        "varied_parameters": 5,
        "plot": str(stem.with_suffix(".png").resolve()),
    }
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    if not gate:
        raise SystemExit("JAX HMC forward non-regression gate failed")


if __name__ == "__main__":
    main()
