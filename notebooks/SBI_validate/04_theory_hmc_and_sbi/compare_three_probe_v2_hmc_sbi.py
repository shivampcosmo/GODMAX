#!/usr/bin/env python3
"""Measure agreement between the v2 HMC and the v2 score-compressed SBI.

Reads only saved artifacts -- no forward model, no sampler.  Both posteriors are
in the same standard-normal probit coordinates ``u`` and on the same converged
grid, and both jobs consumed the same pinned reference point, so the comparison
is a like-for-like one by construction rather than by assertion; the script
verifies that by checking the recorded hashes agree before comparing anything.

Three comparisons are reported, not one:

* HMC versus SBI-NPE                  -- the headline agreement;
* HMC versus the SBI job's own exact-likelihood importance reference -- this
  isolates whether any disagreement is the neural density estimator or the
  sampler, because the exact reference shares SBI's code path but not its
  network;
* the absolute chi-square of both, against ``retained_rank - n_varied = 37``.

The third exists because agreement is not sufficiency.  Two methods can agree
precisely on the posterior of a model that does not fit the data, and this
observation has a large absolute misfit that neither method can remove.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

PARAMETER_NAMES = ("theta_ej_0", "alpha_nt", "mu_beta", "theta_co_0", "nu_theta_ej_M")
# Frozen agreement thresholds, matching the SBI runner's internal gates.
AGREEMENT_GATE = dict(max_mean_shift_pooled_sigma=0.30, width90_ratio=(0.85, 1.18))


def summarize(samples: np.ndarray, names: tuple[str, ...], weights: np.ndarray | None = None) -> dict:
    out = {}
    if weights is None:
        weights = np.ones(samples.shape[0])
    weights = weights / weights.sum()
    for index, name in enumerate(names):
        column = samples[:, index]
        order = np.argsort(column)
        cumulative = np.cumsum(weights[order])
        mean = float(np.sum(weights * column))
        low = float(np.interp(0.05, cumulative, column[order]))
        high = float(np.interp(0.95, cumulative, column[order]))
        out[name] = dict(mean=mean,
                         std=float(np.sqrt(np.sum(weights * (column - mean) ** 2))),
                         median=float(np.interp(0.5, cumulative, column[order])),
                         q05=low, q95=high, width90=high - low)
    return out


def compare(a: dict, b: dict, names: tuple[str, ...]) -> dict:
    out = {}
    for name in names:
        pooled = np.sqrt(0.5 * (a[name]["std"] ** 2 + b[name]["std"] ** 2))
        out[name] = dict(
            mean_shift_pooled_sigma=float(abs(a[name]["mean"] - b[name]["mean"]) / pooled),
            width90_ratio=float(a[name]["width90"] / b[name]["width90"]),
        )
    return out


def verdict(comparison: dict) -> tuple[bool, list[str]]:
    failures = []
    for name, value in comparison.items():
        if value["mean_shift_pooled_sigma"] > AGREEMENT_GATE["max_mean_shift_pooled_sigma"]:
            failures.append(f"{name}: mean shift {value['mean_shift_pooled_sigma']:.3f}")
        low, high = AGREEMENT_GATE["width90_ratio"]
        if not low <= value["width90_ratio"] <= high:
            failures.append(f"{name}: width ratio {value['width90_ratio']:.3f}")
    return not failures, failures


def table(title: str, comparison: dict) -> None:
    print(f"\n{title}")
    print(f"  {'parameter':<16} {'mean shift [pooled sigma]':>26} {'90% width ratio':>18}")
    for name, value in comparison.items():
        print(f"  {name:<16} {value['mean_shift_pooled_sigma']:>26.3f} "
              f"{value['width90_ratio']:>18.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmc-dir", type=pathlib.Path, required=True)
    parser.add_argument("--sbi-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    hmc_diagnostics = json.loads((args.hmc_dir / "hmc_diagnostics.json").read_text())
    sbi_diagnostics = json.loads((args.sbi_dir / "diagnostics.json").read_text())

    provenance = dict(
        same_reference_point=hmc_diagnostics["reference_point_sha256"] == sbi_diagnostics["reference_point_sha256"],
        same_contract=hmc_diagnostics["contract_sha256"] == sbi_diagnostics["contract_sha256"],
        same_grid=hmc_diagnostics["grid"] == sbi_diagnostics["grid"],
        same_numerical_sources=(hmc_diagnostics["numerical_sources"]["aggregate_sha256"]
                                == sbi_diagnostics["numerical_sources"]["aggregate_sha256"]),
        same_backend=(hmc_diagnostics["backend"]["device_kind"]
                      == sbi_diagnostics["backend"]["device_kind"]),
        hmc_status=hmc_diagnostics["status"], sbi_status=sbi_diagnostics["status"],
        hmc_backend_parity=hmc_diagnostics["backend_parity"]["passed"],
        sbi_backend_parity=sbi_diagnostics["backend_parity"]["passed"],
    )
    print("provenance:", json.dumps(provenance, indent=1, sort_keys=True))
    blocking = [k for k in ("same_reference_point", "same_contract", "same_grid",
                            "same_numerical_sources") if not provenance[k]]
    if blocking:
        raise SystemExit(f"Refusing to compare: the two jobs did not solve the same problem ({blocking})")

    hmc_u = np.load(args.hmc_dir / "hmc_samples.npz")["u"].reshape(-1, 5)
    sbi_round = len(sbi_diagnostics["rounds"])
    sbi_npe = np.load(args.sbi_dir / f"posterior_samples_round_{sbi_round}.npz")
    validation = np.load(args.sbi_dir / "exact_likelihood_validation.npz")
    exact_u = validation["u"]
    exact_weights = np.exp(validation["log_weights"] - np.max(validation["log_weights"]))

    u_names = tuple(f"u_{n}" for n in PARAMETER_NAMES)
    hmc_summary = summarize(hmc_u, u_names)
    npe_summary = summarize(sbi_npe["u"], u_names)
    exact_summary = summarize(exact_u, u_names, exact_weights)

    hmc_vs_npe = compare(npe_summary, hmc_summary, u_names)
    hmc_vs_exact = compare(exact_summary, hmc_summary, u_names)
    npe_vs_exact = compare(npe_summary, exact_summary, u_names)

    table("SBI-NPE versus HMC  (the headline agreement)", hmc_vs_npe)
    table("SBI exact-likelihood reference versus HMC  (isolates sampler from network)", hmc_vs_exact)
    table("SBI-NPE versus SBI exact reference  (isolates the network alone)", npe_vs_exact)

    npe_pass, npe_failures = verdict(hmc_vs_npe)
    exact_pass, exact_failures = verdict(hmc_vs_exact)

    hmc_chi2 = hmc_diagnostics["chi2_replay"]["chi2_median"]
    sbi_chi2 = sbi_diagnostics["exact_reference"]["chi2"]["posterior_weighted_mean"]
    print(f"\nABSOLUTE FIT (this is not an agreement question):")
    print(f"  HMC replayed median whitened chi2      : {hmc_chi2:.2f}")
    print(f"  SBI posterior-weighted exact chi2      : {sbi_chi2:.2f}")
    print(f"  nominal reference retained_rank - n_varied = 42 - 5 = 37, scatter ~8.6")
    # This verdict used to be hardcoded, which was correct on the pasted-map
    # contract (chi2 161-165 for a nominal 37) and false on a self-consistent
    # theory observation (chi2 ~3). A conclusion printed regardless of the numbers
    # is not evidence.
    expected, scatter = 37.0, 8.6
    worst = max(hmc_chi2, sbi_chi2)
    deviation = (worst - expected) / scatter
    if deviation > 3.0:
        print(f"  -> both posteriors describe a model that does NOT fit this data "
              f"vector ({deviation:+.1f} sigma above the nominal reference).")
    elif deviation < -3.0:
        print(f"  -> chi2 is far BELOW the nominal reference ({deviation:+.1f} sigma). "
              f"Expected for a noiseless self-consistent observation, where the "
              f"generating point has chi2 = 0 exactly; it would be suspicious for a "
              f"noisy one.")
    else:
        print(f"  -> chi2 is consistent with the nominal reference "
              f"({deviation:+.1f} sigma). The fit is acceptable.")

    print(f"\nSBI importance diagnostics: "
          f"{json.dumps(sbi_diagnostics['exact_reference']['diagnostics'], sort_keys=True)}")
    print(f"HMC divergences {hmc_diagnostics['divergences']}, "
          f"depth saturation {hmc_diagnostics['tree_depth_saturation_fraction']:.4%}, "
          f"max r_hat {hmc_diagnostics['max_rhat']:.4f}, min ESS {hmc_diagnostics['min_ess']:.0f}")

    payload = dict(
        schema="godmax.sbi.three_probe_v2_agreement.v1",
        agreement_gate=AGREEMENT_GATE, provenance=provenance,
        hmc_vs_npe=hmc_vs_npe, hmc_vs_exact=hmc_vs_exact, npe_vs_exact=npe_vs_exact,
        hmc_u_summary=hmc_summary, npe_u_summary=npe_summary, exact_u_summary=exact_summary,
        npe_agrees_with_hmc=npe_pass, npe_failures=npe_failures,
        exact_agrees_with_hmc=exact_pass, exact_failures=exact_failures,
        absolute_fit=dict(hmc_median_chi2=hmc_chi2, sbi_weighted_chi2=sbi_chi2,
                          expected=37, expected_scatter=8.6,
                          acceptable=bool(abs(hmc_chi2 - 37) <= 3 * 8.6)),
        hmc_diagnostics_summary=dict(
            status=hmc_diagnostics["status"], divergences=hmc_diagnostics["divergences"],
            saturation=hmc_diagnostics["tree_depth_saturation_fraction"],
            max_rhat=hmc_diagnostics["max_rhat"], min_ess=hmc_diagnostics["min_ess"]),
        sbi_diagnostics_summary=dict(
            status=sbi_diagnostics["status"],
            importance=sbi_diagnostics["exact_reference"]["diagnostics"]),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output}")
    print(f"\nVERDICT  NPE vs HMC: {'AGREE' if npe_pass else 'DISAGREE ' + str(npe_failures)}")
    print(f"VERDICT  exact vs HMC: {'AGREE' if exact_pass else 'DISAGREE ' + str(exact_failures)}")


if __name__ == "__main__":
    main()
