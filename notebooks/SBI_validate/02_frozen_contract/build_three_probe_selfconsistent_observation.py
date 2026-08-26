#!/usr/bin/env python3
"""Build a self-consistent theory observation for the three-probe contract.

Why
---
The production contract's observation is a pasted-map measurement plus one noise
realization.  The samplers' forward model is the analytic JAX halo model.  Those
are different functions, so the chi-square floor is the paste-versus-theory
mismatch, not noise: 161.09 at the MAP for a nominal ``42 - 5 = 37``.  Every
difficulty in the v2 campaign traced back to that -- the observation is a ~13
sigma outlier for the simulator, the posterior is dragged onto a curved ridge as
the model tries to absorb a mismatch it cannot absorb, and a disagreement between
HMC and SBI cannot be separated from model misspecification.

This script writes an observation that IS the forward model's own prediction at a
chosen parameter point, reusing the production covariance, Cholesky, window,
galaxy pixel window and profile Bell byte-for-byte.  Then

    chi2(theta_generating) == 0    exactly, to float64 round-off,

so HMC-versus-SBI agreement becomes decidable: any disagreement is the inference
machinery and nothing else.

Noiseless by choice
-------------------
``--noise-realization`` can add exactly one draw of the contract noise, which
would put the chi-square at the generating point near 42 and the posterior about
one sigma off truth.  The default is noiseless because the question being asked is
whether two samplers agree, and a noiseless observation makes the whitened
residual at the generating point exactly zero -- the single most typical summary
the simulator can produce, which is also the friendliest possible conditioning
point for NPE.

Oracle discipline
-----------------
The generating point is written to a sibling ``*_generating_point.json`` that no
sampler reads, never into the contract, whose own hash is then stamped into
``three_probe_inference_contract.py``.  The samplers see an observation and a
covariance, exactly as they do in production.
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
import pathlib
import re

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import h5py
import numpy as np
import yaml

from three_probe_inference_contract import (
    DEFAULT_CONTRACT_PATH, SELFCONSISTENT_CONTRACT_PATH, VECTOR_ORDER,
    load_training_contract,
)
from three_probe_jax_forward_model import PARAMETER_NAMES, make_three_probe_forward_model

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
AUDIT_MANIFEST = REPO_ROOT / "data/SBI_validate/three_probe_inference/experiment_manifest.yaml"
OBSERVATION_PATH = REPO_ROOT / "data/SBI_validate/three_probe_inference/observation_selfconsistent.h5"
GENERATING_POINT_PATH = REPO_ROOT / "data/SBI_validate/three_probe_inference/observation_selfconsistent_generating_point.json"
GRID = (256, 48, 48, 2049)
# chi2 at the generating point must be zero to round-off.  The scale is set by
# float64 cancellation in a 42-dimensional triangular solve on values whose
# whitened magnitudes reach ~1e2, so ~1e-20 is the honest floor and 1e-16 is a
# generous ceiling that still rejects any real inconsistency by many orders.
CHI2_AT_GENERATING_POINT_TOLERANCE = 1.0e-16


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, nargs=4, default=list(GRID),
                        metavar=("APERTURE", "NR", "NZ", "ELL"))
    parser.add_argument("--generating-point", type=float, nargs=5, default=None,
                        help="Parameter point in physical units; default is the "
                             "audit manifest's point, i.e. the parameters the "
                             "maps were painted at.")
    parser.add_argument("--noise-realization", type=int, default=None,
                        help="If given, add one Cholesky-correlated noise draw "
                             "with this seed. Default: noiseless.")
    parser.add_argument("--observation", type=pathlib.Path, default=OBSERVATION_PATH)
    parser.add_argument("--contract", type=pathlib.Path, default=SELFCONSISTENT_CONTRACT_PATH)
    parser.add_argument("--stamp", action="store_true",
                        help="Write the contract hash into three_probe_inference_contract.py.")
    args = parser.parse_args()

    production = load_training_contract(DEFAULT_CONTRACT_PATH)
    if tuple(item["name"] for item in production.sampled_parameters) != PARAMETER_NAMES:
        raise RuntimeError("Sampled parameter order differs from the forward model")

    if args.generating_point is None:
        audit = yaml.safe_load(AUDIT_MANIFEST.read_text())
        sampled = audit["parameters"]["sampled"]
        if tuple(item["name"] for item in sampled) != PARAMETER_NAMES:
            raise RuntimeError("Audit parameter order differs from the forward model")
        theta = np.asarray([item["truth"] for item in sampled], dtype=np.float64)
        origin = "audit_manifest_generating_parameters"
    else:
        theta = np.asarray(args.generating_point, dtype=np.float64)
        origin = "command_line"

    low = np.asarray([p["prior"]["low"] for p in production.sampled_parameters])
    high = np.asarray([p["prior"]["high"] for p in production.sampled_parameters])
    if not np.all((theta > low) & (theta < high)):
        raise RuntimeError(f"Generating point {theta} is not strictly inside the prior box")

    aperture, profile_nr, profile_nz, limber_ell_nodes = args.grid
    forward = make_three_probe_forward_model(
        DEFAULT_CONTRACT_PATH, dense_radius_nodes=aperture, profile_nr=profile_nr,
        profile_nz=profile_nz, limber_ell_nodes=limber_ell_nodes, jit_compile=True)
    prediction = np.asarray(forward.vector_fn(theta), dtype=np.float64)
    if prediction.shape != (42,) or not np.all(np.isfinite(prediction)):
        raise RuntimeError("Forward model did not return a finite 42-vector")

    data_vector = prediction.copy()
    noise_metadata = dict(noiseless=True, seed=None)
    if args.noise_realization is not None:
        draw = np.random.default_rng(args.noise_realization).standard_normal(42)
        data_vector = prediction + production.cholesky @ draw
        noise_metadata = dict(noiseless=False, seed=int(args.noise_realization))

    # ---- the decisive check, before anything is written -------------------
    residual = data_vector - prediction
    whitened = np.linalg.solve(production.cholesky, residual)
    chi2_at_generating_point = float(whitened @ whitened)
    if noise_metadata["noiseless"]:
        if not chi2_at_generating_point <= CHI2_AT_GENERATING_POINT_TOLERANCE:
            raise RuntimeError(
                f"chi2 at the generating point is {chi2_at_generating_point:.6e}, above "
                f"{CHI2_AT_GENERATING_POINT_TOLERANCE:.1e}. A noiseless self-consistent "
                f"observation must reproduce its own generator exactly; this means the "
                f"forward model is not deterministic or the Cholesky solve is wrong.")

    args.observation.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.observation, "w") as handle:
        handle.attrs["schema_version"] = "godmax.sbi.three_probe_observation.v1"
        handle.attrs["vector_order"] = VECTOR_ORDER
        handle.attrs["observation_kind"] = "self_consistent_forward_model_prediction"
        handle.attrs["grid_json"] = json.dumps(list(args.grid))
        handle.attrs["forward_metadata_json"] = json.dumps(forward.metadata, sort_keys=True)
        handle.attrs["noise_json"] = json.dumps(noise_metadata, sort_keys=True)
        handle.attrs["chi2_at_generating_point"] = chi2_at_generating_point
        handle.create_dataset("data_vector", data=data_vector)
        for name, array in (("covariance", production.covariance),
                            ("cholesky", production.cholesky),
                            ("window", production.window),
                            ("pixel_window_g", production.pixel_window_g),
                            ("profile_smoothing_bell", production.profile_smoothing_bell),
                            ("band_edges", production.band_edges),
                            ("effective_ell", production.effective_ell)):
            handle.create_dataset(name, data=array)

    production_payload = yaml.safe_load(DEFAULT_CONTRACT_PATH.read_text())
    contract = dict(
        schema_version=production_payload["schema_version"],
        analysis=production_payload["analysis"],
        covariance=production_payload["covariance"],
        parameters=production_payload["parameters"],
        seed_namespaces=production_payload["seed_namespaces"],
        observation=dict(
            path=str(args.observation.relative_to(REPO_ROOT)),
            vector_order=VECTOR_ORDER,
            vector_sha256=sha256_array(data_vector),
            kind="self_consistent_forward_model_prediction",
        ),
        source_hashes={
            "data/SBI_validate/three_probe_inference/inference_contract.yaml":
                sha256_file(DEFAULT_CONTRACT_PATH),
            "notebooks/SBI_validate/common/three_probe_jax_forward_model.py":
                sha256_file(pathlib.Path(__file__).with_name("three_probe_jax_forward_model.py")),
            "notebooks/SBI_validate/02_frozen_contract/build_three_probe_selfconsistent_observation.py":
                sha256_file(pathlib.Path(__file__)),
        },
    )
    args.contract.write_text(yaml.safe_dump(contract, sort_keys=True))
    contract_sha = sha256_file(args.contract)

    generating = dict(
        schema="godmax.sbi.three_probe_selfconsistent_generating_point.v1",
        note="NOT a sampler input. Read only by comparison and coverage scripts.",
        origin=origin,
        parameter_names=list(PARAMETER_NAMES),
        theta=theta.tolist(),
        grid=list(args.grid),
        contract=str(args.contract.relative_to(REPO_ROOT)),
        contract_sha256=contract_sha,
        observation=str(args.observation.relative_to(REPO_ROOT)),
        observation_sha256=sha256_file(args.observation),
        data_vector_sha256=sha256_array(data_vector),
        noise=noise_metadata,
        chi2_at_generating_point=chi2_at_generating_point,
        forward_metadata=forward.metadata,
    )
    GENERATING_POINT_PATH.write_text(json.dumps(generating, indent=2, sort_keys=True) + "\n")

    module = pathlib.Path(__file__).with_name("three_probe_inference_contract.py")
    if args.stamp:
        text = module.read_text()
        stamped = re.sub(r"^SELFCONSISTENT_CONTRACT_SHA256 = .*$",
                         f'SELFCONSISTENT_CONTRACT_SHA256 = "{contract_sha}"',
                         text, count=1, flags=re.MULTILINE)
        if stamped == text:
            raise RuntimeError("Could not stamp SELFCONSISTENT_CONTRACT_SHA256")
        module.write_text(stamped)

    print(json.dumps(dict(
        status="PASS",
        generating_point=dict(zip(PARAMETER_NAMES, theta.tolist())),
        chi2_at_generating_point=chi2_at_generating_point,
        tolerance=CHI2_AT_GENERATING_POINT_TOLERANCE,
        noiseless=noise_metadata["noiseless"],
        contract=str(args.contract), contract_sha256=contract_sha,
        observation=str(args.observation),
        data_vector_sha256=sha256_array(data_vector),
        reuses_production_covariance=True,
        stamped=bool(args.stamp),
    ), indent=2, sort_keys=True))
    print("\ndata vector, first band of each probe (self-consistent vs production):")
    for index, name in enumerate(("gy", "gkappa", "gtau")):
        s = 14 * index
        print(f"   {name:7s} theory {data_vector[s]:+.6e}   pasted {production.data_vector[s]:+.6e}"
              f"   ratio {data_vector[s] / production.data_vector[s]:.4f}")


if __name__ == "__main__":
    main()
