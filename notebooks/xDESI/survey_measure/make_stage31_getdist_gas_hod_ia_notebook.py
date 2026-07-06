#!/usr/bin/env python
"""Build a GetDist contour notebook for the Stage-31 60-parameter HMC run."""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import nbformat
import numpy as np


DEFAULT_RUN_DIR = Path(
    "/mnt/ceph/users/spandey/ltu-godmax/GODMAX/notebooks/xDESI/survey_measure/outputs/"
    "godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu/"
    "stage31_hmc_abacus_cosmo_midres2048_simple1h2h_lmax3000_gk1000_mmin11p147538_60param_"
    "depth6_defaultacc_warm25_2000x16_checkpoint25_v1"
)

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
    "A_IA",
    "eta_IA",
]

LABELS = [
    r"\theta_{\rm ej,0}",
    r"\nu_{\theta_{\rm ej},M}",
    r"\mu_{\beta}",
    r"\log_{10} M_{1}^{\rm pz1}",
    r"\log_{10} M_{1}^{\rm pz2}",
    r"\log_{10} M_{1}^{\rm pz3}",
    r"\log_{10} M_{1}^{\rm pz4}",
    r"\alpha_{\rm sat}^{\rm pz1}",
    r"\alpha_{\rm sat}^{\rm pz2}",
    r"\alpha_{\rm sat}^{\rm pz3}",
    r"\alpha_{\rm sat}^{\rm pz4}",
    r"A_{\rm IA}",
    r"\eta_{\rm IA}",
]

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
    "all_selected": {
        "params": PARAMS,
        "title": "Selected gas, HOD, and IA parameters",
        "width": 18.0,
    },
}


def find_latest_complete_checkpoint(run_dir: Path) -> int | None:
    counts: dict[int, int] = {}
    for path in (run_dir / "workers").glob("worker_*/chain_stage31_checkpoint_*.npz"):
        name = path.name
        if name == "chain_stage31_checkpoint_latest.npz":
            continue
        try:
            draw = int(name.removeprefix("chain_stage31_checkpoint_").removesuffix(".npz"))
        except ValueError:
            continue
        counts[draw] = counts.get(draw, 0) + 1
    complete = [draw for draw, count in counts.items() if count == 4]
    return max(complete) if complete else None


def load_latest_worker_samples(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    worker_paths = sorted((run_dir / "workers").glob("worker_*/chain_stage31_checkpoint_latest.npz"))
    if not worker_paths:
        raise FileNotFoundError(f"No worker latest checkpoints found under {run_dir / 'workers'}")

    sample_blocks = []
    chi2_blocks = []
    divergent_blocks = []
    worker_summary = []
    for path in worker_paths:
        with np.load(path, allow_pickle=True) as data:
            missing = [param for param in PARAMS if f"sample__{param}" not in data.files]
            if missing:
                raise KeyError(f"{path} is missing requested sample keys: {missing}")
            n_sample = int(np.asarray(data[f"sample__{PARAMS[0]}"]).size)
            block = np.column_stack([np.asarray(data[f"sample__{param}"], dtype=np.float64) for param in PARAMS])
            chi2 = np.asarray(data["sample__chi2"], dtype=np.float64)
            if "extra__diverging" in data.files:
                diverging = np.asarray(data["extra__diverging"], dtype=bool)
            else:
                diverging = np.zeros(n_sample, dtype=bool)
            if block.shape[0] != chi2.size or chi2.size != diverging.size:
                raise ValueError(f"Inconsistent sample lengths in {path}")
            sample_blocks.append(block)
            chi2_blocks.append(chi2)
            divergent_blocks.append(diverging)
            worker_summary.append(
                {
                    "worker": path.parent.name,
                    "path": str(path),
                    "n_samples": n_sample,
                    "n_divergent": int(np.count_nonzero(diverging)),
                    "best_chi2": float(np.nanmin(chi2)),
                }
            )

    samples = np.vstack(sample_blocks)
    chi2 = np.concatenate(chi2_blocks)
    diverging = np.concatenate(divergent_blocks)
    return samples, chi2, diverging, worker_summary


def render_getdist_plots(
    run_dir: Path,
    out_dir: Path,
    *,
    label: str,
    remove_divergent: bool,
) -> dict[str, object]:
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots

    samples, chi2, diverging, worker_summary = load_latest_worker_samples(run_dir)
    finite = np.all(np.isfinite(samples), axis=1) & np.isfinite(chi2)
    keep = finite & ~diverging if remove_divergent else finite
    clean_samples = samples[keep]
    clean_chi2 = chi2[keep]
    if clean_samples.size == 0:
        raise RuntimeError("No finite samples remain after filtering")

    out_dir.mkdir(parents=True, exist_ok=True)
    latest_complete = find_latest_complete_checkpoint(run_dir)
    best_index = int(np.nanargmin(clean_chi2))
    best_values = {param: float(clean_samples[best_index, i]) for i, param in enumerate(PARAMS)}

    gd_samples = MCSamples(
        samples=clean_samples,
        names=PARAMS,
        labels=LABELS,
        label=label,
        settings={
            "contours": [0.68, 0.95],
            "fine_bins": 1024,
            "fine_bins_2D": 384,
            "smooth_scale_1D": 0.35,
            "smooth_scale_2D": 0.45,
        },
    )

    def save_triangle(stem: str, params: list[str], title: str, width: float) -> dict[str, str]:
        color = "#2f6f9f"
        line_color = "#123c57"
        g = plots.get_subplot_plotter(width_inch=width)
        n_param = len(params)
        g.settings.figure_legend_frame = False
        g.settings.axes_fontsize = 7 if n_param > 8 else 9
        g.settings.lab_fontsize = 9 if n_param > 8 else 11
        g.settings.legend_fontsize = 10
        g.settings.alpha_filled_add = 0.80
        g.settings.linewidth_contour = 1.05
        g.triangle_plot(
            [gd_samples],
            params=params,
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
            f"{label}; samples used: {clean_samples.shape[0]:,}; latest complete checkpoint: "
            f"{latest_complete if latest_complete is not None else 'none'}",
            fontsize=8,
            color="#333333",
        )
        pdf = out_dir / f"{stem}.pdf"
        png = out_dir / f"{stem}.png"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, bbox_inches="tight", dpi=220)
        plt.close(fig)
        return {"pdf": str(pdf), "png": str(png)}

    outputs = {
        name: save_triangle(
            f"getdist_{name}_stage31_60param_checkpoint_latest",
            list(spec["params"]),
            str(spec["title"]),
            float(spec["width"]),
        )
        for name, spec in GROUPS.items()
    }

    percentiles = np.percentile(clean_samples, [16, 50, 84], axis=0)
    parameter_summary = {
        param: {
            "p16": float(percentiles[0, i]),
            "median": float(percentiles[1, i]),
            "p84": float(percentiles[2, i]),
            "best_chi2_sample": best_values[param],
        }
        for i, param in enumerate(PARAMS)
    }
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(out_dir),
        "latest_complete_checkpoint_draws_per_worker": latest_complete,
        "worker_summary": worker_summary,
        "n_raw_samples": int(samples.shape[0]),
        "n_finite_samples": int(np.count_nonzero(finite)),
        "n_divergent_samples": int(np.count_nonzero(diverging)),
        "n_samples_used": int(clean_samples.shape[0]),
        "remove_divergent": bool(remove_divergent),
        "best_chi2_used_samples": float(np.nanmin(clean_chi2)),
        "parameters": PARAMS,
        "labels": dict(zip(PARAMS, LABELS)),
        "parameter_summary": parameter_summary,
        "plot_outputs": outputs,
    }
    summary_path = out_dir / "getdist_gas_hod_ia_sample_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def notebook_source(run_dir: Path, out_dir: Path, label: str, remove_divergent: bool) -> nbformat.NotebookNode:
    params_repr = repr(PARAMS)
    labels_repr = repr(LABELS)
    groups_repr = repr(GROUPS)
    run_dir_repr = repr(str(run_dir))
    out_dir_repr = repr(str(out_dir))
    label_repr = repr(label)
    remove_divergent_repr = repr(bool(remove_divergent))

    cells = [
        nbformat.v4.new_markdown_cell(
            textwrap.dedent(
                f"""
                # Stage-31 HMC GetDist contours

                This notebook makes diagnostic GetDist contours for selected gas, HOD, and IA
                parameters from the Stage-31 60-parameter HMC run.

                The code loads `chain_stage31_checkpoint_latest.npz` from each worker to include
                every worker sample currently present in the run directory. The latest complete
                synchronized checkpoint is also reported for provenance.
                """
            ).strip()
        ),
        nbformat.v4.new_code_cell(
            textwrap.dedent(
                f"""
                from pathlib import Path
                import json
                import os

                RUN_DIR = Path({run_dir_repr})
                OUT_DIR = Path({out_dir_repr})
                OUT_DIR.mkdir(parents=True, exist_ok=True)

                os.environ.setdefault("MPLCONFIGDIR", str(RUN_DIR / "matplotlib"))
                Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

                PARAMS = {params_repr}
                LABELS = {labels_repr}
                GROUPS = {groups_repr}
                SAMPLE_LABEL = {label_repr}
                REMOVE_DIVERGENT = {remove_divergent_repr}
                """
            ).strip()
        ),
        nbformat.v4.new_code_cell(
            textwrap.dedent(
                """
                import matplotlib
                matplotlib.use("Agg")

                import matplotlib.pyplot as plt
                import numpy as np
                from getdist import MCSamples, plots


                def find_latest_complete_checkpoint(run_dir):
                    counts = {}
                    for path in (run_dir / "workers").glob("worker_*/chain_stage31_checkpoint_*.npz"):
                        if path.name == "chain_stage31_checkpoint_latest.npz":
                            continue
                        draw = int(path.name.removeprefix("chain_stage31_checkpoint_").removesuffix(".npz"))
                        counts[draw] = counts.get(draw, 0) + 1
                    complete = [draw for draw, count in counts.items() if count == 4]
                    return max(complete) if complete else None


                def load_latest_worker_samples(run_dir):
                    worker_paths = sorted((run_dir / "workers").glob("worker_*/chain_stage31_checkpoint_latest.npz"))
                    sample_blocks, chi2_blocks, divergent_blocks, worker_summary = [], [], [], []
                    for path in worker_paths:
                        with np.load(path, allow_pickle=True) as data:
                            block = np.column_stack([
                                np.asarray(data[f"sample__{param}"], dtype=np.float64)
                                for param in PARAMS
                            ])
                            chi2 = np.asarray(data["sample__chi2"], dtype=np.float64)
                            diverging = (
                                np.asarray(data["extra__diverging"], dtype=bool)
                                if "extra__diverging" in data.files
                                else np.zeros(chi2.size, dtype=bool)
                            )
                        sample_blocks.append(block)
                        chi2_blocks.append(chi2)
                        divergent_blocks.append(diverging)
                        worker_summary.append({
                            "worker": path.parent.name,
                            "path": str(path),
                            "n_samples": int(block.shape[0]),
                            "n_divergent": int(np.count_nonzero(diverging)),
                            "best_chi2": float(np.nanmin(chi2)),
                        })
                    return (
                        np.vstack(sample_blocks),
                        np.concatenate(chi2_blocks),
                        np.concatenate(divergent_blocks),
                        worker_summary,
                    )


                samples, chi2, diverging, worker_summary = load_latest_worker_samples(RUN_DIR)
                finite = np.all(np.isfinite(samples), axis=1) & np.isfinite(chi2)
                keep = finite & ~diverging if REMOVE_DIVERGENT else finite
                samples = samples[keep]
                chi2 = chi2[keep]
                latest_complete = find_latest_complete_checkpoint(RUN_DIR)
                best_index = int(np.nanargmin(chi2))
                best_values = {param: float(samples[best_index, i]) for i, param in enumerate(PARAMS)}

                print(f"latest complete checkpoint: {latest_complete}")
                print(f"samples used: {samples.shape[0]:,}")
                print(json.dumps(worker_summary, indent=2))
                """
            ).strip()
        ),
        nbformat.v4.new_code_cell(
            textwrap.dedent(
                """
                gd_samples = MCSamples(
                    samples=samples,
                    names=PARAMS,
                    labels=LABELS,
                    label=SAMPLE_LABEL,
                    settings={
                        "contours": [0.68, 0.95],
                        "fine_bins": 1024,
                        "fine_bins_2D": 384,
                        "smooth_scale_1D": 0.35,
                        "smooth_scale_2D": 0.45,
                    },
                )
                """
            ).strip()
        ),
        nbformat.v4.new_code_cell(
            textwrap.dedent(
                """
                def save_triangle(name, spec):
                    color = "#2f6f9f"
                    line_color = "#123c57"
                    params = list(spec["params"])
                    n_param = len(params)
                    g = plots.get_subplot_plotter(width_inch=float(spec["width"]))
                    g.settings.figure_legend_frame = False
                    g.settings.axes_fontsize = 7 if n_param > 8 else 9
                    g.settings.lab_fontsize = 9 if n_param > 8 else 11
                    g.settings.legend_fontsize = 10
                    g.settings.alpha_filled_add = 0.80
                    g.settings.linewidth_contour = 1.05
                    g.triangle_plot(
                        [gd_samples],
                        params=params,
                        filled=True,
                        contour_colors=[color],
                        line_args=[{"color": line_color, "lw": 1.1}],
                        contour_args=[{"alpha": 0.82}],
                        markers={param: best_values[param] for param in params},
                        marker_args={"color": "#202020", "lw": 0.8, "ls": "--"},
                        title_limit=1,
                    )
                    fig = g.fig
                    fig.suptitle(str(spec["title"]), y=0.998, fontsize=13 if n_param > 8 else 14)
                    fig.text(
                        0.01,
                        0.01,
                        f"{SAMPLE_LABEL}; samples used: {samples.shape[0]:,}; "
                        f"latest complete checkpoint: {latest_complete}",
                        fontsize=8,
                        color="#333333",
                    )
                    pdf = OUT_DIR / f"getdist_{name}_stage31_60param_checkpoint_latest.pdf"
                    png = OUT_DIR / f"getdist_{name}_stage31_60param_checkpoint_latest.png"
                    fig.savefig(pdf, bbox_inches="tight")
                    fig.savefig(png, bbox_inches="tight", dpi=220)
                    plt.close(fig)
                    return {"pdf": str(pdf), "png": str(png)}


                plot_outputs = {name: save_triangle(name, spec) for name, spec in GROUPS.items()}
                print(json.dumps(plot_outputs, indent=2))
                """
            ).strip()
        ),
        nbformat.v4.new_code_cell(
            textwrap.dedent(
                """
                percentiles = np.percentile(samples, [16, 50, 84], axis=0)
                parameter_summary = {
                    param: {
                        "p16": float(percentiles[0, i]),
                        "median": float(percentiles[1, i]),
                        "p84": float(percentiles[2, i]),
                        "best_chi2_sample": best_values[param],
                    }
                    for i, param in enumerate(PARAMS)
                }
                summary = {
                    "run_dir": str(RUN_DIR),
                    "output_dir": str(OUT_DIR),
                    "latest_complete_checkpoint_draws_per_worker": latest_complete,
                    "worker_summary": worker_summary,
                    "n_samples_used": int(samples.shape[0]),
                    "best_chi2_used_samples": float(np.nanmin(chi2)),
                    "parameters": PARAMS,
                    "labels": dict(zip(PARAMS, LABELS)),
                    "parameter_summary": parameter_summary,
                    "plot_outputs": plot_outputs,
                }
                summary_path = OUT_DIR / "getdist_gas_hod_ia_sample_summary.json"
                summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n")
                print(summary_path)
                """
            ).strip()
        ),
        nbformat.v4.new_markdown_cell(
            textwrap.dedent(
                """
                ## Rendered figures

                These PNG/PDF files are generated next to this notebook by
                `make_stage31_getdist_gas_hod_ia_notebook.py`.

                [Full selected 13-parameter triangle](getdist_all_selected_stage31_60param_checkpoint_latest.pdf)

                ![Gas and IA triangle](getdist_gas_ia_stage31_60param_checkpoint_latest.png)

                ![HOD M1 triangle](getdist_hod_m1_stage31_60param_checkpoint_latest.png)

                ![HOD alpha sat triangle](getdist_hod_alphasat_stage31_60param_checkpoint_latest.png)
                """
            ).strip()
        ),
    ]

    nb = nbformat.v4.new_notebook(cells=cells)
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python (ili-sbi)",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {
        "name": "python",
        "pygments_lexer": "ipython3",
    }
    return nb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-label", default="Stage-31 60-param HMC latest")
    parser.add_argument("--keep-divergent", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "combined" / "getdist_gas_hod_ia"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = render_getdist_plots(
        run_dir,
        out_dir,
        label=args.sample_label,
        remove_divergent=not args.keep_divergent,
    )
    nb = notebook_source(run_dir, out_dir, args.sample_label, not args.keep_divergent)
    notebook_path = out_dir / "stage31_60param_getdist_gas_hod_ia_contours.ipynb"
    nbformat.write(nb, notebook_path)
    print(json.dumps({"notebook": str(notebook_path), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
