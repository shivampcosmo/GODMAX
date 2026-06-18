#!/usr/bin/env python
"""Make Stage-31 GetDist contours from one combined HMC checkpoint chain."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PARAMS = [
    "theta_ej_0",
    "nu_theta_ej_M",
    "mu_beta",
    "log10M1_fshmr_pz1",
    "log10M1_fshmr_pz2",
    "log10M1_fshmr_pz3",
    "log10M1_fshmr_pz4",
    "alphasat_Nsat_pz1",
    "alphasat_Nsat_pz2",
    "alphasat_Nsat_pz3",
    "alphasat_Nsat_pz4",
    "fcen_pz1",
    "fcen_pz2",
    "fcen_pz3",
    "fcen_pz4",
    "A_IA",
    "eta_IA",
]

LABELS = {
    "theta_ej_0": r"\theta_{\rm ej,0}",
    "nu_theta_ej_M": r"\nu_{\theta_{\rm ej},M}",
    "mu_beta": r"\mu_{\beta}",
    "log10M1_fshmr_pz1": r"\log_{10} M_{1}^{\rm pz1}",
    "log10M1_fshmr_pz2": r"\log_{10} M_{1}^{\rm pz2}",
    "log10M1_fshmr_pz3": r"\log_{10} M_{1}^{\rm pz3}",
    "log10M1_fshmr_pz4": r"\log_{10} M_{1}^{\rm pz4}",
    "alphasat_Nsat_pz1": r"\alpha_{\rm sat}^{\rm pz1}",
    "alphasat_Nsat_pz2": r"\alpha_{\rm sat}^{\rm pz2}",
    "alphasat_Nsat_pz3": r"\alpha_{\rm sat}^{\rm pz3}",
    "alphasat_Nsat_pz4": r"\alpha_{\rm sat}^{\rm pz4}",
    "fcen_pz1": r"f_{\rm cen}^{\rm pz1}",
    "fcen_pz2": r"f_{\rm cen}^{\rm pz2}",
    "fcen_pz3": r"f_{\rm cen}^{\rm pz3}",
    "fcen_pz4": r"f_{\rm cen}^{\rm pz4}",
    "A_IA": r"A_{\rm IA}",
    "eta_IA": r"\eta_{\rm IA}",
}

GROUPS = {
    "gas_ia": {
        "params": ["theta_ej_0", "nu_theta_ej_M", "mu_beta", "A_IA", "eta_IA"],
        "title": "Gas and intrinsic-alignment parameters",
        "width": 8.0,
    },
    "hod_m1": {
        "params": [
            "log10M1_fshmr_pz1",
            "log10M1_fshmr_pz2",
            "log10M1_fshmr_pz3",
            "log10M1_fshmr_pz4",
        ],
        "title": "HOD M1 parameters by DESI photo-z bin",
        "width": 6.8,
    },
    "hod_alphasat": {
        "params": [
            "alphasat_Nsat_pz1",
            "alphasat_Nsat_pz2",
            "alphasat_Nsat_pz3",
            "alphasat_Nsat_pz4",
        ],
        "title": "HOD satellite-slope parameters by DESI photo-z bin",
        "width": 6.8,
    },
    "hod_fcen": {
        "params": ["fcen_pz1", "fcen_pz2", "fcen_pz3", "fcen_pz4"],
        "title": "HOD central-occupation amplitudes by DESI photo-z bin",
        "width": 6.8,
    },
    "all_selected": {
        "params": PARAMS,
        "title": "Selected gas, HOD, fcen, and IA parameters",
        "width": 20.0,
    },
}


def _available_params(files: Sequence[str], params: Sequence[str]) -> list[str]:
    return [param for param in params if f"sample__{param}" in files]


def load_chain_samples(
    chain_path: Path,
    params: Sequence[str],
    *,
    remove_divergent: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, object]]:
    with np.load(chain_path, allow_pickle=True) as data:
        available = _available_params(data.files, params)
        if not available:
            raise KeyError(f"{chain_path} does not contain any requested sample__ parameters.")
        samples = np.column_stack(
            [np.asarray(data[f"sample__{param}"], dtype=np.float64).reshape(-1) for param in available]
        )
        chi2 = np.asarray(data["sample__chi2"], dtype=np.float64).reshape(-1)
        if "extra__diverging" in data.files:
            diverging = np.asarray(data["extra__diverging"], dtype=bool).reshape(-1)
        else:
            diverging = np.zeros(chi2.size, dtype=bool)
        metadata = {}
        if "metadata_json" in data.files:
            raw = np.asarray(data["metadata_json"]).item()
            metadata = json.loads(str(raw))

    if samples.shape[0] != chi2.size or chi2.size != diverging.size:
        raise ValueError(
            f"Inconsistent chain lengths in {chain_path}: samples={samples.shape[0]} "
            f"chi2={chi2.size} diverging={diverging.size}"
        )
    finite = np.all(np.isfinite(samples), axis=1) & np.isfinite(chi2)
    keep = finite & ~diverging if remove_divergent else finite
    clean_samples = samples[keep]
    clean_chi2 = chi2[keep]
    clean_diverging = diverging[keep]
    if clean_samples.size == 0:
        raise RuntimeError("No finite samples remain after filtering.")
    summary = {
        "metadata": metadata,
        "n_raw_samples": int(samples.shape[0]),
        "n_finite_samples": int(np.count_nonzero(finite)),
        "n_divergent_samples": int(np.count_nonzero(diverging)),
        "n_samples_used": int(clean_samples.shape[0]),
        "remove_divergent": bool(remove_divergent),
        "best_chi2_used_samples": float(np.nanmin(clean_chi2)),
    }
    return clean_samples, clean_chi2, clean_diverging, available, summary


def save_triangle(
    gd_samples,
    out_dir: Path,
    *,
    stem: str,
    params: Sequence[str],
    title: str,
    width: float,
    label: str,
    best_values: Mapping[str, float],
    n_samples: int,
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from getdist import plots

    color = "#2f6f9f"
    line_color = "#123c57"
    n_param = len(params)
    g = plots.get_subplot_plotter(width_inch=float(width))
    g.settings.figure_legend_frame = False
    g.settings.axes_fontsize = 7 if n_param > 8 else 9
    g.settings.lab_fontsize = 9 if n_param > 8 else 11
    g.settings.legend_fontsize = 10
    g.settings.alpha_filled_add = 0.80
    g.settings.linewidth_contour = 1.05
    g.triangle_plot(
        [gd_samples],
        params=list(params),
        filled=True,
        contour_colors=[color],
        line_args=[{"color": line_color, "lw": 1.1}],
        contour_args=[{"alpha": 0.82}],
        markers={param: best_values[param] for param in params},
        marker_args={"color": "#202020", "lw": 0.8, "ls": "--"},
        title_limit=1,
    )
    fig = g.fig
    fig.suptitle(title, y=0.998, fontsize=13 if n_param > 8 else 14)
    fig.text(
        0.01,
        0.01,
        f"{label}; samples used: {n_samples:,}",
        fontsize=8,
        color="#333333",
    )
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def render_getdist_plots(
    chain_path: Path,
    out_dir: Path,
    *,
    label: str,
    tag: str,
    remove_divergent: bool,
) -> dict[str, object]:
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    from getdist import MCSamples

    samples, chi2, diverging, params, summary = load_chain_samples(
        chain_path,
        PARAMS,
        remove_divergent=remove_divergent,
    )
    labels = [LABELS[param] for param in params]
    best_index = int(np.nanargmin(chi2))
    best_values = {param: float(samples[best_index, i]) for i, param in enumerate(params)}
    gd_samples = MCSamples(
        samples=samples,
        names=params,
        labels=labels,
        label=label,
        settings={
            "contours": [0.68, 0.95],
            "fine_bins": 1024,
            "fine_bins_2D": 384,
            "smooth_scale_1D": 0.35,
            "smooth_scale_2D": 0.45,
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, spec in GROUPS.items():
        group_params = [param for param in spec["params"] if param in params]
        if len(group_params) < 2:
            continue
        outputs[name] = save_triangle(
            gd_samples,
            out_dir,
            stem=f"getdist_{name}_{tag}",
            params=group_params,
            title=str(spec["title"]),
            width=float(spec["width"]),
            label=label,
            best_values=best_values,
            n_samples=int(samples.shape[0]),
        )

    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    parameter_summary = {
        param: {
            "p16": float(percentiles[0, i]),
            "median": float(percentiles[1, i]),
            "p84": float(percentiles[2, i]),
            "best_chi2_sample": best_values[param],
        }
        for i, param in enumerate(params)
    }
    payload = {
        "chain_path": str(chain_path),
        "output_dir": str(out_dir),
        "tag": str(tag),
        "label": str(label),
        "parameters": params,
        "labels": {param: LABELS[param] for param in params},
        "parameter_summary": parameter_summary,
        "plot_outputs": outputs,
        **summary,
    }
    summary_path = out_dir / f"getdist_gas_hod_ia_sample_summary_{tag}.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["summary_path"] = str(summary_path)
    return payload


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
    summary = render_getdist_plots(
        args.chain.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        label=args.sample_label,
        tag=args.tag,
        remove_divergent=not args.keep_divergent,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
