#!/usr/bin/env python3
"""Re-evaluate a reference artifact's parity vectors on this process's backend.

Run twice on the same compute node -- once with the GPU visible and once with
``JAX_PLATFORMS=cpu`` -- to settle whether the forward model is backend-portable.

Motivation: a CPU replay of the rejected depth-6 HMC artifact disagreed with its
own stored chi-square by about 88 units at every one of its 12,000 samples, with
byte-identical sources across 31 hashed files and hash-verified pinned inputs.
The only difference that could be found was the execution backend.  If the two
runs of this script disagree, that is a numerics defect in the forward model and
every absolute chi-square this project has ever quoted from a GPU run is
suspect; if they agree, the depth-6 discrepancy has some other cause and the
search continues.  Either answer is worth one minute of a compute node.
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
    PARITY_CHI2_TOLERANCE, PARITY_VECTOR_RELATIVE_TOLERANCE, atomic_json,
    backend_manifest, build_problem, numerical_source_manifest, sha256_array,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-point", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, default=None,
                        help="Registered contract that produced the reference point. "
                             "The chi-square half of this gate depends on the "
                             "observation, so it must be the same contract or the "
                             "gate compares two different problems.")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference_point.read_text())
    grid = tuple(reference["grid"])
    backend = backend_manifest()
    problem = build_problem(grid, jit_compile=True, contract_path=args.contract)
    reference_contract = reference.get("contract_sha256")
    if reference_contract is not None and reference_contract != problem.contract.contract_sha256:
        raise RuntimeError(
            "Reference point was built against a different inference contract than "
            "the one given to this parity check. The chi-square half of the gate "
            "would compare two different problems and fail for a reason that has "
            "nothing to do with backend portability. Pass the matching --contract.")

    points = np.asarray(reference["parity"]["probit_points"], dtype=np.float64)
    expected = np.asarray(reference["parity"]["vectors"], dtype=np.float64)
    here = np.stack([np.asarray(problem.predict_u(jnp.asarray(p)), dtype=np.float64)
                     for p in points])
    chi2_here = [float(problem.chi2_u(jnp.asarray(p))) for p in points]
    chi2_reference = list(reference["parity"]["chi2"])

    relative = np.abs(here / expected - 1.0)
    u_map = np.asarray(reference["u_map"], dtype=np.float64)
    chi2_at_map_here = float(problem.chi2_u(jnp.asarray(u_map)))

    payload = dict(
        schema="godmax.sbi.three_probe_backend_parity.v1",
        grid=list(grid),
        reference_point=str(args.reference_point),
        reference_backend=reference["backend"]["default_backend"],
        reference_device_kind=reference["backend"]["device_kind"],
        this_backend=backend["default_backend"],
        this_device_kind=backend["device_kind"],
        numerical_sources_aggregate=numerical_source_manifest()["aggregate_sha256"],
        reference_numerical_sources_aggregate=reference["numerical_sources"]["aggregate_sha256"],
        sources_identical=(numerical_source_manifest()["aggregate_sha256"]
                           == reference["numerical_sources"]["aggregate_sha256"]),
        vectors_sha256_here=sha256_array(here),
        vectors_sha256_reference=reference["parity"]["vectors_sha256"],
        max_relative_difference=float(relative.max()),
        median_relative_difference=float(np.median(relative)),
        per_point_max_relative=relative.max(axis=1).tolist(),
        chi2_reference=chi2_reference,
        chi2_here=chi2_here,
        chi2_absolute_difference=[float(a - b) for a, b in zip(chi2_here, chi2_reference)],
        chi2_at_map_reference=reference["chi2_at_map"],
        chi2_at_map_here=chi2_at_map_here,
        chi2_at_map_difference=chi2_at_map_here - float(reference["chi2_at_map"]),
        vector_relative_tolerance=PARITY_VECTOR_RELATIVE_TOLERANCE,
        chi2_tolerance=PARITY_CHI2_TOLERANCE,
        vector_passed=bool(float(relative.max()) <= PARITY_VECTOR_RELATIVE_TOLERANCE),
        chi2_passed=bool(abs(chi2_at_map_here - float(reference["chi2_at_map"]))
                         <= PARITY_CHI2_TOLERANCE),
        passed=bool(float(relative.max()) <= PARITY_VECTOR_RELATIVE_TOLERANCE
                    and abs(chi2_at_map_here - float(reference["chi2_at_map"]))
                    <= PARITY_CHI2_TOLERANCE),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"\nbackend {payload['this_backend']} ({payload['this_device_kind']}) versus "
          f"reference {payload['reference_backend']} ({payload['reference_device_kind']}):")
    print(f"  max relative forward difference : {payload['max_relative_difference']:.3e}")
    print(f"  chi2 at the MAP                 : {chi2_at_map_here:.6f} here versus "
          f"{reference['chi2_at_map']:.6f} in the reference "
          f"(difference {payload['chi2_at_map_difference']:+.3e})")
    print(f"  gate: 42-vector rel <= {PARITY_VECTOR_RELATIVE_TOLERANCE:.0e}  -> "
          f"{'PASS' if payload['vector_passed'] else 'FAIL'}")
    print(f"  gate: |d chi2|      <= {PARITY_CHI2_TOLERANCE:.0e}  -> "
          f"{'PASS' if payload['chi2_passed'] else 'FAIL'}")
    print(f"  OVERALL                         : "
          f"{'PORTABLE' if payload['passed'] else 'NOT BACKEND-PORTABLE'}")


if __name__ == "__main__":
    main()
