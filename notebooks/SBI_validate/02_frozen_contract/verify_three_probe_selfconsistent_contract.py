#!/usr/bin/env python3
"""Verify the self-consistent contract reproduces its own generator on THIS node.

Run before sampling.  The observation was written on whichever GPU built it, so on
a different device the chi-square at the generating point is no longer exactly zero
-- it is bounded by the forward model's cross-backend agreement.  This script
states the measured value and gates on the same tolerance the backend-parity gate
uses, so a real portability regression is caught while float64 round-off is not
mistaken for one.

Reads the generating point, which is NOT a sampler input.  This is a validation
script, not a sampler.
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

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from three_probe_agreement_common import (
    GRID, PARAMETER_NAMES, PARITY_CHI2_TOLERANCE, backend_manifest, build_problem,
    probit_from_theta,
)
from three_probe_inference_contract import SELFCONSISTENT_CONTRACT_PATH

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
GENERATING_POINT_PATH = REPO_ROOT / "data/SBI_validate/three_probe_inference/observation_selfconsistent_generating_point.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=pathlib.Path, default=SELFCONSISTENT_CONTRACT_PATH)
    parser.add_argument("--generating-point", type=pathlib.Path, default=GENERATING_POINT_PATH)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--grid", type=int, nargs=4, default=list(GRID))
    args = parser.parse_args()

    generating = json.loads(args.generating_point.read_text())
    if generating["schema"] != "godmax.sbi.three_probe_selfconsistent_generating_point.v1":
        raise RuntimeError("Unexpected generating-point schema")
    if tuple(generating["parameter_names"]) != PARAMETER_NAMES:
        raise RuntimeError("Generating-point parameter order differs from the forward model")

    problem = build_problem(tuple(args.grid), jit_compile=True, contract_path=args.contract)
    if generating["contract_sha256"] != __import__("hashlib").sha256(
            args.contract.read_bytes()).hexdigest():
        raise RuntimeError("Generating point does not belong to this contract file")

    theta = np.asarray(generating["theta"], dtype=np.float64)
    u = probit_from_theta(theta[None, :], problem.low, problem.high)[0]
    chi2 = float(problem.chi2_u(jnp.asarray(u)))
    prediction = np.asarray(problem.predict_u(jnp.asarray(u)), dtype=np.float64)
    residual = problem.observation - prediction
    relative = np.max(np.abs(prediction / problem.observation - 1.0))

    passed = chi2 <= PARITY_CHI2_TOLERANCE
    payload = dict(
        schema="godmax.sbi.three_probe_selfconsistent_verification.v1",
        status="PASS" if passed else "FAIL",
        contract=str(args.contract), contract_sha256=generating["contract_sha256"],
        grid=list(args.grid), backend=backend_manifest(),
        generating_theta=dict(zip(PARAMETER_NAMES, theta.tolist())),
        generating_u=u.tolist(),
        chi2_at_generating_point=chi2,
        chi2_tolerance=PARITY_CHI2_TOLERANCE,
        chi2_recorded_at_build=generating["chi2_at_generating_point"],
        build_backend_device=generating.get("forward_metadata", {}),
        max_relative_prediction_difference=float(relative),
        max_absolute_residual=float(np.max(np.abs(residual))),
        noiseless=generating["noise"]["noiseless"],
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nchi2 at the generating point on this node: {chi2:.6e}"
          f"  (built as {generating['chi2_at_generating_point']:.3e},"
          f" gate {PARITY_CHI2_TOLERANCE:.1e})")
    print(f"max relative prediction-vs-observation difference: {relative:.3e}")
    if not passed:
        raise SystemExit(
            "The self-consistent observation does not reproduce its own generator on "
            "this node. Sampling would target a problem whose chi-square floor is not "
            "zero, which is the whole point of this contract.")


if __name__ == "__main__":
    main()
