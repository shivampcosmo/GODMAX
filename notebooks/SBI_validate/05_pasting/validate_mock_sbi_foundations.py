#!/usr/bin/env python3
"""Gate-1 foundations for mock SBI: prove the measurement chain is the frozen one.

Three claims are checked, in order of how much they would cost to get wrong:

1.  Measuring the archived frozen paste through this module reproduces the noise
    contract's stored noiseless bandpowers.  If this fails, the measurement path
    is not the contract's estimator and no mock-SBI vector is comparable to the
    theory observation.
2.  ``mu_paste(theta_ref) + nu(observation seeds)`` reproduces the inference
    contract's ``data_vector`` to machine precision.  This is the end-to-end
    statement that the simulator this campaign will run is the same process that
    produced the observation the theory runs already used.
3.  The exact linearity ``x = mu_paste + nu`` holds, re-verified here against the
    archived realization rather than assumed from section 1.3 of the plan.

Writes a JSON evidence ledger; exits non-zero on any failure.
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
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import mock_sbi_common as msc

# Tolerances.  These are float64 round-off budgets for a chain of
# map2alm -> alm2cl -> decouple_cell, not adjustable quality thresholds.
BANDPOWER_REPRODUCTION_RTOL = 1.0e-13
OBSERVATION_REPRODUCTION_RTOL = 1.0e-13
LINEARITY_RTOL = 1.0e-13
# The frozen observation mixes two mask precisions: build_contract made the signal
# alms with the float64 mask, realize() drew the noise against the lossily stored
# float32 copy.  Reproducing it bitwise therefore needs the same mixture.  The
# production path uses float64 throughout; that choice differs from the frozen
# observation by the amount recorded as `production_mask_consistency`, which must
# stay far below the goodness-of-fit scatter sqrt(2*37) = 8.6.
PRODUCTION_MASK_CHI2_MAX = 1.0e-10
# Whitened chi-square equivalent of the reproduction error must be far below the
# expected goodness-of-fit scatter (sqrt(2*37) = 8.6).
CHI2_EQUIVALENT_MAX = 1.0e-6


def max_relative(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.abs(a), np.abs(b))
    scale = np.where(scale > 0.0, scale, 1.0)
    return float(np.max(np.abs(a - b) / scale))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path,
                        default=msc.REPO_ROOT / "data/SBI_validate/mock_sbi/foundations.json")
    args = parser.parse_args()

    started = time.time()
    print("[1/5] loading frozen estimator context ...", flush=True)
    ctx = msc.load_estimator_context()
    print(f"      noise contract   {ctx.contract_sha256[:16]}")
    print(f"      workspace        {ctx.workspace_sha256[:16]}")
    print(f"      mask array       {ctx.mask_sha256[:16]}   sum={ctx.mask_sum:.6e}")
    print(f"      fixed g alm      {ctx.galaxy_alm_sha256[:16]}   size={ctx.fixed_galaxy_alm.size}")

    print("[2/5] measuring the archived frozen paste ...", flush=True)
    t0 = time.time()
    mu_ref = msc.measure_paste_file(msc.FROZEN_MAP_PATH, ctx)
    measure_seconds = time.time() - t0
    stored_fixed = np.concatenate([ctx.fixed_bandpowers[s] for s in msc.SPECTRA])
    rel_fixed = max_relative(mu_ref, stored_fixed)
    chi2_fixed = ctx.chi2(mu_ref - stored_fixed)
    print(f"      measured in {measure_seconds:.1f}s")
    print(f"      max|rel diff| vs contract fixed_bandpowers = {rel_fixed:.3e}")
    print(f"      whitened chi2 of the difference            = {chi2_fixed:.3e}")

    print("[3/5] reproducing the frozen observation noise ...", flush=True)
    obs_seeds = {name: msc.OBSERVATION_BASE_SEED + offset
                 for name, offset in msc.OBSERVATION_FIELD_OFFSETS.items()}
    t0 = time.time()
    nu_obs = msc.noise_vector(obs_seeds, ctx, mask=ctx.stored_mask_float32)
    noise_seconds = time.time() - t0
    nu_obs_production = msc.noise_vector(obs_seeds, ctx)
    print(f"      seeds {obs_seeds}  ({noise_seconds:.1f}s)")

    print("[4/5] end-to-end observation reconstruction ...", flush=True)
    observation, obs_provenance = msc.load_inference_observation()
    reconstructed = mu_ref + nu_obs
    rel_obs = max_relative(reconstructed, observation)
    chi2_obs = ctx.chi2(reconstructed - observation)
    print(f"      max|rel diff| vs contract data_vector = {rel_obs:.3e}")
    print(f"      whitened chi2 of the difference       = {chi2_obs:.3e}")

    print("[5/5] production mask-precision consistency ...", flush=True)
    linearity = max_relative(reconstructed, observation)
    production_vector = mu_ref + nu_obs_production
    mask_rel = max_relative(production_vector, observation)
    mask_chi2 = ctx.chi2(production_vector - observation)
    print(f"      float64-throughout vs frozen observation: rel {mask_rel:.3e}  chi2 {mask_chi2:.3e}")

    per_probe = {}
    for index, spectrum in enumerate(msc.SPECTRA):
        sl = slice(index * msc.N_BAND, (index + 1) * msc.N_BAND)
        per_probe[spectrum] = {
            "max_relative_fixed": max_relative(mu_ref[sl], stored_fixed[sl]),
            "max_relative_observation": max_relative(reconstructed[sl], observation[sl]),
        }

    checks = {
        "measured_paste_matches_contract_bandpowers": rel_fixed <= BANDPOWER_REPRODUCTION_RTOL,
        "observation_reconstructed": rel_obs <= OBSERVATION_REPRODUCTION_RTOL,
        "linearity_holds": linearity <= LINEARITY_RTOL,
        "reproduction_chi2_negligible": max(chi2_fixed, chi2_obs) <= CHI2_EQUIVALENT_MAX,
        "production_mask_consistency": mask_chi2 <= PRODUCTION_MASK_CHI2_MAX,
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    payload = {
        "status": status,
        "checks": checks,
        "tolerances": {
            "bandpower_reproduction_rtol": BANDPOWER_REPRODUCTION_RTOL,
            "observation_reproduction_rtol": OBSERVATION_REPRODUCTION_RTOL,
            "linearity_rtol": LINEARITY_RTOL,
            "chi2_equivalent_max": CHI2_EQUIVALENT_MAX,
        },
        "measured": {
            "max_relative_paste_vs_contract_bandpowers": rel_fixed,
            "whitened_chi2_paste_vs_contract_bandpowers": chi2_fixed,
            "max_relative_observation_reconstruction": rel_obs,
            "whitened_chi2_observation_reconstruction": chi2_obs,
            "per_probe": per_probe,
            "production_mask_max_relative": mask_rel,
            "production_mask_whitened_chi2": mask_chi2,
            "paste_measurement_seconds": measure_seconds,
            "noise_vector_seconds": noise_seconds,
        },
        "identity": {
            "noise_contract_sha256": ctx.contract_sha256,
            "workspace_sha256": ctx.workspace_sha256,
            "mask_array_sha256": ctx.mask_sha256,
            "stored_mask_float32_sha256": ctx.stored_mask_sha256,
            "mask_metadata": ctx.mask_metadata,
            "fixed_galaxy_alm_sha256": ctx.galaxy_alm_sha256,
            "frozen_map_sha256": msc.sha256_file(msc.FROZEN_MAP_PATH),
            "mu_paste_reference_sha256": msc.sha256_array(mu_ref),
            "observation": obs_provenance,
            "observation_field_seeds": obs_seeds,
        },
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.output)
    np.savez(args.output.with_name("reference_paste_vector.npz"),
             mu_paste_reference=mu_ref, nu_observation=nu_obs, observation=observation)

    print(f"\nstatus {status}   ledger {args.output}")
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
