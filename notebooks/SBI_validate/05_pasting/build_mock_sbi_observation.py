#!/usr/bin/env python3
"""Build the NOISELESS pasted observation for the mock-SBI campaign.

Why noiseless
-------------
The theory HMC and theory SBI results this campaign is compared against were run on
``inference_contract_selfconsistent.yaml``, whose observation is the analytic forward
model's own NOISELESS prediction at the generating point, so chi2 there is exactly 0.

The production contract's observation is the same pasted maps PLUS one noise draw.
Conditioning mock SBI on that vector would displace its posterior from the generating
point by about one sigma in a direction fixed by the draw, and that displacement would
appear in the three-way plot as a mock-versus-theory offset which is the noise
realization and nothing else.  Making the pasted observation noiseless puts all three
methods on the same footing: each is conditioned on a vector its own forward model
reproduces exactly, so any residual disagreement is the inference machinery or the
paste-versus-theory physics, which is what the comparison is for.

Noise still enters everywhere it belongs -- the likelihood covariance is the frozen
42x42 ``C``, and every mock training row is ``mu_paste(theta) + nu(seed)`` with ``nu``
a field-level draw pushed through the real estimator.

What is reused byte for byte
----------------------------
The covariance, Cholesky, bandpower window, galaxy pixel window and profile Bell are
copied from the production contract and their hashes are asserted equal to it by the
loader.  The mask, NaMaster workspace, fixed galaxy alm and band edges come from the
frozen noise contract through ``mock_sbi_common``, which verifies every one of their
hashes.  Nothing here rebuilds an estimator.

Three checks run before anything is written
-------------------------------------------
1.  the archived paste, measured here, reproduces the noise contract's stored
    ``fixed_bandpowers`` -- the measurement chain is the frozen one;
2.  ``mu_paste + nu(observation seeds)`` reproduces the production contract's
    ``data_vector`` -- this really is the process that produced the observation the
    theory runs already used;
3.  the vector written differs from the production one, so the two contracts can
    never be confused.
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
import pathlib
import re
import sys

import h5py
import numpy as np
import yaml

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import mock_sbi_common as msc
from three_probe_inference_contract import (
    DEFAULT_CONTRACT_PATH, MOCK_CONTRACT_PATH, VECTOR_ORDER, load_training_contract,
)

REPO_ROOT = msc.REPO_ROOT
AUDIT_MANIFEST = REPO_ROOT / "data/SBI_validate/three_probe_inference/experiment_manifest.yaml"
OBSERVATION_PATH = REPO_ROOT / "data/SBI_validate/three_probe_inference/observation_mock.h5"
GENERATING_POINT_PATH = (REPO_ROOT /
                         "data/SBI_validate/three_probe_inference/observation_mock_generating_point.json")
PARAMETER_NAMES = ("theta_ej_0", "alpha_nt", "mu_beta", "theta_co_0", "nu_theta_ej_M")
REPRODUCTION_RTOL = 1.0e-13


def observation_field_seeds(realization: int = 0) -> dict[str, int]:
    """The seeds the frozen observation's noise draw used."""

    base = msc.OBSERVATION_BASE_SEED + realization
    return {name: base + offset for name, offset in msc.OBSERVATION_FIELD_OFFSETS.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observation", type=pathlib.Path, default=OBSERVATION_PATH)
    parser.add_argument("--contract", type=pathlib.Path, default=MOCK_CONTRACT_PATH)
    parser.add_argument("--stamp", action="store_true",
                        help="Write the contract hash into three_probe_inference_contract.py")
    parser.add_argument("--report", type=pathlib.Path,
                        default=REPO_ROOT / "data/SBI_validate/mock_sbi/observation_mock.json")
    args = parser.parse_args()

    production = load_training_contract(DEFAULT_CONTRACT_PATH)
    context = msc.load_estimator_context()

    print("[1/4] measuring the archived pasted maps through the frozen estimator ...",
          flush=True)
    mu_paste = msc.measure_paste_file(msc.FROZEN_MAP_PATH, context)
    stored = np.concatenate([context.fixed_bandpowers[name] for name in msc.SPECTRA])
    chain_error = float(np.max(np.abs(mu_paste / stored - 1.0)))
    chain_chi2 = context.chi2(mu_paste - stored)
    print(f"      vs the contract's fixed_bandpowers: {chain_error:.3e} relative, "
          f"whitened chi2 {chain_chi2:.3e}")
    if not chain_error <= REPRODUCTION_RTOL:
        raise RuntimeError(
            f"the measurement chain does not reproduce the frozen bandpowers "
            f"({chain_error:.3e} > {REPRODUCTION_RTOL:.0e}); no vector measured here is "
            f"comparable to what the theory runs used")

    print("[2/4] reproducing the production observation from paste + its noise draw ...",
          flush=True)
    seeds = observation_field_seeds(0)
    nu = msc.noise_vector(seeds, context, mask=context.stored_mask_float32)
    reconstructed = mu_paste + nu
    observation_error = float(np.max(np.abs(reconstructed / production.data_vector - 1.0)))
    observation_chi2 = context.chi2(reconstructed - production.data_vector)
    print(f"      vs the production data_vector: {observation_error:.3e} relative, "
          f"whitened chi2 {observation_chi2:.3e}")
    if not observation_error <= REPRODUCTION_RTOL:
        raise RuntimeError(
            f"paste + nu(observation seeds) does not reproduce the production observation "
            f"({observation_error:.3e} > {REPRODUCTION_RTOL:.0e})")

    data_vector = np.ascontiguousarray(mu_paste, dtype=np.float64)
    if msc.sha256_array(data_vector) == msc.sha256_array(production.data_vector):
        raise RuntimeError("the noiseless vector equals the production one; that is impossible "
                           "and would make the two contracts indistinguishable")

    audit = yaml.safe_load(AUDIT_MANIFEST.read_text())
    sampled = audit["parameters"]["sampled"]
    if tuple(item["name"] for item in sampled) != PARAMETER_NAMES:
        raise RuntimeError("Audit parameter order differs from the inference order")
    theta = np.asarray([item["truth"] for item in sampled], dtype=np.float64)
    low = np.asarray([p["prior"]["low"] for p in production.sampled_parameters])
    high = np.asarray([p["prior"]["high"] for p in production.sampled_parameters])
    if not np.all((theta > low) & (theta < high)):
        raise RuntimeError(f"Generating point {theta} is not strictly inside the prior box")

    print("[3/4] writing the observation and the contract ...", flush=True)
    args.observation.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.observation, "w") as handle:
        handle.attrs["schema_version"] = "godmax.sbi.three_probe_observation.v1"
        handle.attrs["vector_order"] = VECTOR_ORDER
        handle.attrs["observation_kind"] = "pasted_map_measurement_noiseless"
        handle.attrs["noise_json"] = json.dumps({"noiseless": True, "seed": None},
                                                sort_keys=True)
        handle.attrs["estimator_json"] = json.dumps({
            "noise_contract_sha256": context.contract_sha256,
            "workspace_sha256": context.workspace_sha256,
            "mask_sha256": context.mask_sha256,
            "stored_mask_float32_sha256": context.stored_mask_sha256,
            "fixed_galaxy_alm_sha256": context.galaxy_alm_sha256,
            "map_path": str(msc.FROZEN_MAP_PATH.relative_to(REPO_ROOT)),
            "nside": msc.NSIDE, "lmax": msc.LMAX,
        }, sort_keys=True)
        handle.attrs["reproduction_json"] = json.dumps({
            "max_relative_vs_fixed_bandpowers": chain_error,
            "whitened_chi2_vs_fixed_bandpowers": chain_chi2,
            "max_relative_production_observation": observation_error,
            "whitened_chi2_production_observation": observation_chi2,
            "tolerance": REPRODUCTION_RTOL,
        }, sort_keys=True)
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
            vector_sha256=msc.sha256_array(data_vector),
            kind="pasted_map_measurement_noiseless",
        ),
        source_hashes={
            "data/SBI_validate/three_probe_inference/inference_contract.yaml":
                msc.sha256_file(DEFAULT_CONTRACT_PATH),
            "notebooks/SBI_validate/common/mock_sbi_common.py":
                msc.sha256_file(THIS_DIR / "mock_sbi_common.py"),
            "notebooks/SBI_validate/05_pasting/build_mock_sbi_observation.py":
                msc.sha256_file(pathlib.Path(__file__)),
        },
    )
    args.contract.write_text(yaml.safe_dump(contract, sort_keys=True))
    contract_sha = msc.sha256_file(args.contract)

    if args.stamp:
        module = THIS_DIR / "three_probe_inference_contract.py"
        text = module.read_text()
        stamped = re.sub(r"^MOCK_CONTRACT_SHA256 = .*$",
                         f'MOCK_CONTRACT_SHA256 = "{contract_sha}"',
                         text, count=1, flags=re.MULTILINE)
        if stamped == text:
            raise RuntimeError("Could not stamp MOCK_CONTRACT_SHA256")
        module.write_text(stamped)

    generating = dict(
        schema="godmax.sbi.three_probe_mock_generating_point.v1",
        note="NOT a sampler input. Read only by comparison and coverage scripts.",
        origin="audit_manifest_generating_parameters",
        parameter_names=list(PARAMETER_NAMES),
        theta=theta.tolist(),
        contract=str(args.contract.relative_to(REPO_ROOT)),
        contract_sha256=contract_sha,
        observation=str(args.observation.relative_to(REPO_ROOT)),
        observation_sha256=msc.sha256_file(args.observation),
        data_vector_sha256=msc.sha256_array(data_vector),
        noise=dict(noiseless=True, seed=None),
        chi2_at_generating_point_for_the_paste_simulator=0.0,
        estimator=dict(noise_contract_sha256=context.contract_sha256,
                       workspace_sha256=context.workspace_sha256,
                       mask_sha256=context.mask_sha256,
                       fixed_galaxy_alm_sha256=context.galaxy_alm_sha256),
    )
    GENERATING_POINT_PATH.write_text(json.dumps(generating, indent=2, sort_keys=True) + "\n")

    print("[4/4] verifying the contract loads through the registry ...", flush=True)
    if args.stamp:
        import importlib
        import three_probe_inference_contract as registry
        importlib.reload(registry)
        loaded = registry.load_training_contract(args.contract)
        if not np.array_equal(loaded.data_vector, data_vector):
            raise RuntimeError("The registry returned a different data vector than was written")
        for name, array in (("covariance", production.covariance),
                            ("cholesky", production.cholesky),
                            ("window", production.window)):
            if not np.array_equal(getattr(loaded, name), array):
                raise RuntimeError(f"Loaded {name} is not the production array")
        print("      loads and reuses the production covariance byte for byte")
    else:
        print("      skipped (--stamp not given; the registry has no pinned hash yet)")

    report = dict(
        status="PASS",
        observation=str(args.observation.relative_to(REPO_ROOT)),
        contract=str(args.contract.relative_to(REPO_ROOT)),
        contract_sha256=contract_sha,
        data_vector_sha256=msc.sha256_array(data_vector),
        production_data_vector_sha256=msc.sha256_array(production.data_vector),
        stamped=bool(args.stamp),
        checks=dict(
            measurement_chain_reproduces_fixed_bandpowers=chain_error <= REPRODUCTION_RTOL,
            paste_plus_noise_reproduces_production_observation=(
                observation_error <= REPRODUCTION_RTOL),
            distinct_from_production=True,
        ),
        measured=dict(
            max_relative_vs_fixed_bandpowers=chain_error,
            whitened_chi2_vs_fixed_bandpowers=chain_chi2,
            max_relative_production_observation=observation_error,
            whitened_chi2_production_observation=observation_chi2,
            whitened_chi2_noiseless_vs_production=context.chi2(
                data_vector - production.data_vector),
        ),
        generating_point=dict(zip(PARAMETER_NAMES, theta.tolist())),
        observation_noise_field_seeds=seeds,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.report.with_name(args.report.name + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.report)

    print(f"\nstatus PASS   wrote {args.observation}")
    print(f"              contract {args.contract} sha {contract_sha}")
    print(f"              chi2(noiseless paste vs production noisy observation) = "
          f"{report['measured']['whitened_chi2_noiseless_vs_production']:.2f}  "
          f"(one noise draw on a 42-vector; expected ~42)")
    print("\nfirst band of each probe:")
    for index, name in enumerate(msc.SPECTRA):
        start = 14 * index
        print(f"   {name:7s} noiseless paste {data_vector[start]:+.6e}   "
              f"production (paste+noise) {production.data_vector[start]:+.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
