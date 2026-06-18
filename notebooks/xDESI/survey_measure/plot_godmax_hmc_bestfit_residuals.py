#!/usr/bin/env python
"""Plot normalized residuals for a GODMAX Stage31 HMC best-fit params YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import godmax_multiprobe_hmc_stage31 as hmc31
import godmax_multiprobe_theory_utils as gmt

DEFAULT_PLOT_XSCALE = "log"
DEFAULT_PLOT_XLIM = "100,3000"


def parse_pair(value: object, *, option: str) -> Optional[tuple[float, float]]:
    if value is None or str(value).strip() == "":
        return None
    parts = str(value).replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"{option} must contain two values, got {value!r}.")
    lo, hi = float(parts[0]), float(parts[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
        raise ValueError(f"{option} must be finite and increasing, got {value!r}.")
    return (lo, hi)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=hmc31.DEFAULT_STAGE31_CONFIG, help="Stage31 HMC YAML config.")
    parser.add_argument("--params", required=True, help="Best-fit params YAML to evaluate and plot.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the residual PDF/PNGs. Defaults to <params parent>/residual_plots.",
    )
    parser.add_argument("--pdf", default=None, help="Optional explicit PDF path.")
    parser.add_argument("--filename-prefix", default="bestfit_dell_residual", help="Prefix for per-family PNG outputs.")
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Plot only the active likelihood vector. By default the full vector is plotted with inactive bands shaded.",
    )
    parser.add_argument(
        "--plot-ell-max",
        type=float,
        default=3000.0,
        help="Maximum ell shown. Use <=0 to show all available bandpowers.",
    )
    parser.add_argument(
        "--plot-xscale",
        default=DEFAULT_PLOT_XSCALE,
        choices=("linear", "log", "symlog"),
        help="X-axis scaling.",
    )
    parser.add_argument(
        "--plot-xlim",
        default=DEFAULT_PLOT_XLIM,
        metavar="XMIN,XMAX",
        help="Optional x-axis limits. Empty string disables explicit limits.",
    )
    parser.add_argument(
        "--residual-ylim",
        default=None,
        metavar="YMIN,YMAX",
        help="Optional y-axis limits for normalized residuals.",
    )
    parser.add_argument(
        "--ksz-scale",
        type=float,
        default=1.0,
        help="Multiplicative display scale for the kSZ pi x T residual convention.",
    )
    return parser


def make_residual_plot(args: argparse.Namespace) -> dict:
    config_path = Path(args.config)
    params_path = Path(args.params)
    output_dir = Path(args.output_dir) if args.output_dir else params_path.parent / "residual_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = Path(args.pdf) if args.pdf else output_dir / f"{args.filename_prefix}.pdf"
    plot_ell_max = None if args.plot_ell_max is not None and args.plot_ell_max <= 0.0 else float(args.plot_ell_max)
    plot_xlim = parse_pair(args.plot_xlim, option="--plot-xlim")
    residual_ylim = parse_pair(args.residual_ylim, option="--residual-ylim")

    context = hmc31.prepare_fit_context(config_path)
    sample = hmc31.pack_sample_from_params_file(context, params_path)
    models = hmc31.build_models_from_sample(context, sample)
    theory_cls = hmc31.extract_theory_cls_jax_from_models(models)

    active_theory = hmc31.theory_data_vector_jax(context.likelihood, theory_cls)
    best_chi2 = float(np.asarray(hmc31.whitened_chi2(context.likelihood, active_theory)))
    chi2_n_modes = int(context.likelihood.rank)
    n_fit_parameters = len(context.parameter_specs)
    chi2_dof = max(chi2_n_modes - n_fit_parameters, 1)
    reduced_chi2 = best_chi2 / float(chi2_dof)

    if args.active_only:
        measurement = hmc31.measurement_for_plots(context)
        theory_vector = np.asarray(active_theory)
        active_band_indices = None
        vector_kind = "active_likelihood"
    else:
        full_likelihood = hmc31.full_likelihood_for_plots(context)
        measurement = hmc31.measurement_from_likelihood(context, full_likelihood)
        theory_vector = np.asarray(hmc31.theory_data_vector_jax(full_likelihood, theory_cls))
        active_band_indices = hmc31.likelihood_active_band_indices(context)
        vector_kind = "full_measurement_with_inactive_shading"

    plot_paths = gmt.plot_family_dell_residual_comparisons(
        measurement,
        theory_vector,
        output_dir,
        pdf_path=pdf_path,
        filename_prefix=args.filename_prefix,
        ell_max=plot_ell_max,
        ksz_scale=float(args.ksz_scale),
        active_band_indices=active_band_indices,
        total_reduced_chi2=reduced_chi2,
        chi2_dof=chi2_dof,
        xscale=str(args.plot_xscale),
        xlim=plot_xlim,
        ylim=residual_ylim,
    )

    summary = {
        "config_path": str(config_path),
        "params_path": str(params_path),
        "pdf_path": str(pdf_path),
        "plot_paths": [str(path) for path in plot_paths],
        "vector_kind": vector_kind,
        "best_whitened_chi2": best_chi2,
        "best_reduced_chi2": reduced_chi2,
        "best_chi2_dof": chi2_dof,
        "best_chi2_per_mode": best_chi2 / max(float(chi2_n_modes), 1.0),
        "chi2_n_modes": chi2_n_modes,
        "n_fit_parameters": n_fit_parameters,
        "plot_settings": {
            "ell_max": plot_ell_max,
            "xscale": args.plot_xscale,
            "xlim": plot_xlim,
            "residual_ylim": residual_ylim,
            "ksz_scale": float(args.ksz_scale),
            "residual_definition": "(plotted bestfit - plotted data) / plotted sigma",
        },
    }
    summary_path = output_dir / f"{args.filename_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(gmt.to_jsonable(summary), handle, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(gmt.to_jsonable(make_residual_plot(args)), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
