#!/usr/bin/env python3
"""Triangle plot of HMC versus SBI on the self-consistent contract, with the truth.

Only possible on this contract: the observation is the forward model's own
prediction at a known parameter point, so the generating point is a real truth and
coverage is checkable, not just agreement.  Reads saved artifacts only -- no
forward model, no sampler.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

PARAMETER_NAMES = ("theta_ej_0", "alpha_nt", "mu_beta", "theta_co_0", "nu_theta_ej_M")
LATEX = (r"\theta_{\rm ej,0}", r"\alpha_{\rm nt}", r"\mu_\beta",
         r"\theta_{\rm co,0}", r"\nu^M_{\theta_{\rm ej}}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmc-dir", type=pathlib.Path, required=True)
    parser.add_argument("--sbi-dir", type=pathlib.Path, required=True)
    parser.add_argument("--generating-point", type=pathlib.Path, required=True)
    parser.add_argument("--hmc-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--burn-in-fraction", type=float, default=0.0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--no-title", action="store_true",
                       help="draw the panels alone, with no suptitle")
    parser.add_argument("--legend-fontsize", type=float, default=10.0)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots

    generating = json.loads(args.generating_point.read_text())
    truth = np.asarray(generating["theta"], dtype=np.float64)

    source = args.hmc_checkpoint
    if source is None:
        final = args.hmc_dir / "hmc_samples.npz"
        if final.is_file():
            source = final
        else:
            candidates = sorted(args.hmc_dir.glob("checkpoint_*.npz"))
            if not candidates:
                raise SystemExit(f"No HMC samples or checkpoint in {args.hmc_dir}")
            source = candidates[-1]
    archive = np.load(source)
    hmc = np.stack([archive[f"sample_{n}"] for n in PARAMETER_NAMES], axis=-1)
    if args.burn_in_fraction > 0.0:
        cut = int(args.burn_in_fraction * hmc.shape[1])
        hmc = hmc[:, cut:]
    hmc_flat = hmc.reshape(-1, len(PARAMETER_NAMES))

    npe_files = sorted(args.sbi_dir.glob("posterior_samples_round_*.npz"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not npe_files:
        raise SystemExit(f"No SBI posterior samples in {args.sbi_dir}")
    npe = np.load(npe_files[-1])["theta"]

    datasets = [
        MCSamples(samples=hmc_flat, names=list(PARAMETER_NAMES), labels=list(LATEX),
                  label=f"HMC, {hmc.shape[1]} draws x {hmc.shape[0]} chains"),
        MCSamples(samples=npe, names=list(PARAMETER_NAMES), labels=list(LATEX),
                  label=f"SBI NPE ({npe_files[-1].stem.rsplit('_', 1)[1]} rounds)"),
    ]

    exact_path = args.sbi_dir / "exact_likelihood_validation.npz"
    exact_summary = None
    if exact_path.is_file():
        payload = np.load(exact_path)
        weights = np.exp(payload["log_weights"] - payload["log_weights"].max())
        keep = weights > 0.0
        datasets.append(MCSamples(samples=payload["theta"][keep], weights=weights[keep],
                                  names=list(PARAMETER_NAMES), labels=list(LATEX),
                                  label="SBI exact-likelihood reference"))
        exact_summary = float(1.0 / np.sum((weights / weights.sum()) ** 2))

    plotter = plots.get_subplot_plotter(width_inch=11.0)
    plotter.settings.alpha_filled_add = 0.45
    plotter.settings.legend_fontsize = args.legend_fontsize
    plotter.triangle_plot(datasets, filled=True,
                          contour_colors=["#333333", "#c0392b", "#2980b9"])
    for row in range(len(PARAMETER_NAMES)):
        for column in range(row + 1):
            axis = plotter.subplots[row, column]
            if axis is None:
                continue
            if row == column:
                axis.axvline(truth[row], color="#16a085", lw=1.6, ls="--", zorder=10)
            else:
                axis.plot(truth[column], truth[row], marker="*", ms=13,
                          color="#16a085", mec="black", mew=0.6, zorder=10)
    if not args.no_title:
        title = args.title or ("HMC versus score-compressed SBI on a self-consistent "
                               "theory observation\n"
                               "green star / dashed line = the generating parameters "
                               "(chi2 = 0 there by construction)")
        plotter.fig.suptitle(title, fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plotter.export(str(args.output))
    plt.close("all")

    print(f"wrote {args.output}")
    print(f"\n{'parameter':18s} {'truth':>10s} {'HMC':>19s} {'NPE':>19s}"
          + ("" if exact_summary is None else f" {'exact':>19s}"))
    for index, name in enumerate(PARAMETER_NAMES):
        row = (f"{name:18s} {truth[index]:10.4f}"
               f" {hmc_flat[:, index].mean():9.4f} +-{hmc_flat[:, index].std(ddof=1):7.4f}"
               f" {npe[:, index].mean():9.4f} +-{npe[:, index].std(ddof=1):7.4f}")
        if exact_summary is not None:
            payload = np.load(exact_path)
            w = np.exp(payload["log_weights"] - payload["log_weights"].max()); w /= w.sum()
            mean = float(np.sum(w * payload["theta"][:, index]))
            sd = float(np.sqrt(np.sum(w * (payload["theta"][:, index] - mean) ** 2)))
            row += f" {mean:9.4f} +-{sd:7.4f}"
        print(row)
    print("\npulls of each posterior mean against the truth, in its own sigma:")
    for index, name in enumerate(PARAMETER_NAMES):
        h = (hmc_flat[:, index].mean() - truth[index]) / hmc_flat[:, index].std(ddof=1)
        n = (npe[:, index].mean() - truth[index]) / npe[:, index].std(ddof=1)
        print(f"   {name:18s} HMC {h:+6.2f}   NPE {n:+6.2f}")
    if exact_summary is not None:
        print(f"\nexact-reference importance ESS: {exact_summary:.1f}")


if __name__ == "__main__":
    main()
