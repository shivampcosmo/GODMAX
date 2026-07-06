#!/usr/bin/env python
"""Stage-31 GetDist for one combined HMC checkpoint chain.

Drop-in replacement for plot_stage31_getdist_gas_hod_ia_checkpoint.py with the
SAME CLI (the checkpoint monitor calls it identically): it reproduces the
gas/HOD/fcen/IA subset triangles + summary JSON via that module, then ALSO renders
a full triangle over ALL sampled parameters (here 64) from the combined chain
(samples from every chain at this checkpoint)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_stage31_getdist_gas_hod_ia_checkpoint as base  # noqa: E402

# Cap samples used for the (expensive) all-parameter triangle. Uniform thinning
# over the combined chain still includes every chain; >~40k samples does not
# visibly change 2D contours but keeps per-checkpoint render time bounded.
MAX_SAMPLES_ALL = 50000


def render_all_params_triangle(chain_path: Path, out_dir: Path, *, label: str, tag: str,
                               remove_divergent: bool) -> dict:
    with np.load(chain_path, allow_pickle=True) as data:
        names = [k[len("sample__"):] for k in data.files
                 if k.startswith("sample__") and k != "sample__chi2"]
        names.sort()
        samples = np.column_stack(
            [np.asarray(data[f"sample__{n}"], np.float64).reshape(-1) for n in names])
        chi2 = np.asarray(data["sample__chi2"], np.float64).reshape(-1)
        diverging = (np.asarray(data["extra__diverging"], bool).reshape(-1)
                     if "extra__diverging" in data.files else np.zeros(chi2.size, bool))
    finite = np.all(np.isfinite(samples), axis=1) & np.isfinite(chi2)
    keep = finite & ~diverging if remove_divergent else finite
    samples, chi2 = samples[keep], chi2[keep]
    n_full = int(samples.shape[0])
    if n_full == 0:
        raise RuntimeError("No finite samples for all-parameter triangle.")
    best = samples[int(np.nanargmin(chi2))]
    if n_full > MAX_SAMPLES_ALL:
        idx = np.linspace(0, n_full - 1, MAX_SAMPLES_ALL).astype(int)
        plot_samples = samples[idx]
    else:
        plot_samples = samples
    n_used = int(plot_samples.shape[0])

    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots

    # matplotlib mathtext: escape underscores (multi-underscore names -> double subscript)
    labels = [base.LABELS.get(n, r"\rm " + n.replace("_", r"\_")) for n in names]
    gd = MCSamples(
        samples=plot_samples, names=names, labels=labels, label=label,
        settings={"contours": [0.68, 0.95], "fine_bins": 128, "fine_bins_2D": 64,
                  "smooth_scale_1D": 0.4, "smooth_scale_2D": 0.5},
    )
    g = plots.get_subplot_plotter(width_inch=44.0)
    g.settings.figure_legend_frame = False
    g.settings.axes_fontsize = 5
    g.settings.lab_fontsize = 6
    g.settings.alpha_filled_add = 0.80
    g.settings.linewidth_contour = 0.7
    g.triangle_plot([gd], filled=True, contour_colors=["#2f6f9f"],
                    line_args=[{"color": "#123c57", "lw": 0.7}],
                    markers={n: float(best[i]) for i, n in enumerate(names)},
                    marker_args={"color": "#202020", "lw": 0.5, "ls": "--"})
    g.fig.suptitle(f"All {len(names)} sampled parameters  —  {tag}", y=0.999, fontsize=15)
    g.fig.text(0.01, 0.005,
               f"{label}; all chains; samples used: {n_used:,} of {n_full:,}",
               fontsize=8, color="#333333")
    stem = f"getdist_all64_{tag}"
    pdf = out_dir / f"{stem}.pdf"
    g.fig.savefig(pdf, bbox_inches="tight")  # vector PDF only (rasterizing a 44in PNG
    plt.close(g.fig)                          # doubles per-checkpoint render time)
    return {"n_params": len(names), "n_samples_full": n_full, "n_samples_used": n_used,
            "pdf": str(pdf)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", type=Path, required=True, help="Combined checkpoint chain NPZ.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-label", default="Stage-31 64-param HMC checkpoint")
    parser.add_argument("--tag", required=True, help="Short tag used in output filenames.")
    parser.add_argument("--keep-divergent", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    chain = args.chain.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    remove_divergent = not args.keep_divergent
    # 1) existing subset triangles + summary JSON (the monitor reads this summary)
    summary = base.render_getdist_plots(
        chain, out_dir, label=args.sample_label, tag=args.tag, remove_divergent=remove_divergent)
    # 2) full all-parameter triangle
    all_params = render_all_params_triangle(
        chain, out_dir, label=args.sample_label, tag=args.tag, remove_divergent=remove_divergent)
    summary["all_params_triangle"] = all_params
    print(json.dumps({"subset_plots": list(summary.get("plot_outputs", {}).keys()),
                      "all_params_triangle": all_params}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
