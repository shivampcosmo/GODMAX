#!/usr/bin/env python3
"""Render a compact one-row Stage-31 pz3 publication comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from plot_stage31_pz3_publication_5panel import (
    DEFAULT_CONFIG,
    array_sha256,
    decode,
    dell,
    excluded_band_spans,
    likelihood_active_mask,
    load_config,
    load_full_theory_vector,
    measurement_slice,
    write_report,
)


PANELS = (
    {
        "name": "desi_g_auto_pz3",
        "family": "desi_g_auto",
        "label": "(a)  Galaxy clustering\n$g_3 \\times g_3$",
        "log_y": True,
    },
    {
        "name": "desi_g_act_y_pz3",
        "family": "desi_g_act_y",
        "label": "(b)  Galaxy $\\times$ thermal SZ\n$g_3 \\times y$",
        "log_y": False,
    },
    {
        "name": "desi_g_des_shear_E_pz3_tomo3",
        "family": "desi_g_des_shear_E",
        "label": "(c)  Galaxy $\\times$ cosmic shear\n$g_3 \\times \\gamma_E^{(3)}$",
        "log_y": False,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--measurement", type=Path)
    parser.add_argument("--sim", type=Path)
    parser.add_argument("--full-theory-vector", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and print numerical digests without writing a figure.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Path]:
    run_root = Path(config["project"]["output_root"]).expanduser().resolve()
    run_name = str(config["pasting"]["run_name"])
    nside = int(config["pasting"]["nside"])
    lmax = int(config["pasting"]["lmax"])
    nbin = int(config["pasting"]["n_bins"])

    bestfit_dir = Path(config["godmax"]["bestfit_params"]).expanduser().resolve().parent
    vector_candidates = sorted(bestfit_dir.glob("bestfit_full_theory_data_vector_*.npz"))
    if args.full_theory_vector is None and len(vector_candidates) != 1:
        raise RuntimeError(
            f"Expected one full theory vector beside the best-fit parameters, found {len(vector_candidates)}."
        )

    paths = {
        "config": args.config.expanduser().resolve(),
        "measurement": (
            args.measurement or Path(config["godmax"]["measurement_h5"])
        ).expanduser().resolve(),
        "sim": (
            args.sim
            or run_root
            / "measurements"
            / f"sim_pz3_cap2400_map64fcen_nside{nside}_lmax{lmax}_nbin{nbin}_log.h5"
        ).expanduser().resolve(),
        "full_theory_vector": (
            args.full_theory_vector or vector_candidates[0]
        ).expanduser().resolve(),
        "output": (
            args.output
            or run_root
            / "plots"
            / f"{run_name}_publication_3panel_Dell.pdf"
        ).expanduser().resolve(),
    }
    missing = [
        f"{key}: {path}"
        for key, path in paths.items()
        if key != "output" and not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing input file(s):\n" + "\n".join(missing))
    if paths["output"].suffix.lower() != ".pdf":
        raise ValueError("--output must name a PDF; a same-stem PNG is written alongside it.")
    return paths


def load_measurement(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        names = decode(h5["joint/spectrum_names"][:])
        result = {
            "names": names,
            "ell": np.asarray(h5["joint/ell"][:], dtype=np.float64),
            "ell_left": np.asarray(h5["ell_left"][:], dtype=np.float64),
            "ell_right": np.asarray(h5["ell_right"][:], dtype=np.float64),
            "data": np.asarray(h5["joint/data_vector"][:], dtype=np.float64),
            "covariance": np.asarray(h5["joint/cov"][:], dtype=np.float64),
            "starts": np.asarray(h5["joint/slice_start"][:], dtype=np.int64),
            "stops": np.asarray(h5["joint/slice_stop"][:], dtype=np.int64),
            "windows": {
                str(panel["name"]): np.asarray(
                    h5[f"spectra/{panel['name']}/bandpower_window_selected"][:],
                    dtype=np.float64,
                )
                for panel in PANELS
            },
        }
    for panel in PANELS:
        if panel["name"] not in result["names"]:
            raise KeyError(f"Measurement is missing {panel['name']!r}.")
    return result


def load_simulation(path: Path) -> dict[str, dict[str, np.ndarray]]:
    spectra: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(path, "r") as h5:
        names = (
            decode(h5["joint/spectrum_names"][:])
            if "joint/spectrum_names" in h5
            else sorted(h5["spectra"].keys())
        )
        for panel in PANELS:
            name = str(panel["name"])
            if name not in names:
                raise KeyError(f"Simulation product is missing required spectrum {name!r}.")
            spectra[name] = {
                "ell": np.asarray(h5[f"spectra/{name}/ell"][:], dtype=np.float64),
                "cl": np.asarray(h5[f"spectra/{name}/cl"][:], dtype=np.float64),
                "window": np.asarray(
                    h5[f"spectra/{name}/bandpower_window_selected"][:],
                    dtype=np.float64,
                ),
            }
    return spectra


def build_series(
    measurement: Mapping[str, Any],
    sim: Mapping[str, Any],
    full_vector: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    spectra: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "input_identity": {
            "data_vector_equal": full_vector["data_equal"],
            "covariance_equal": full_vector["covariance_equal"],
        },
        "spectra": {},
    }
    for panel in PANELS:
        name = str(panel["name"])
        section = measurement_slice(measurement, name)
        ell = np.asarray(measurement["ell"], dtype=np.float64)
        data_cl = np.asarray(measurement["data"][section], dtype=np.float64)
        data_err = np.sqrt(
            np.clip(np.diag(measurement["covariance"][section, section]), 0.0, np.inf)
        )
        theory_cl = np.asarray(full_vector["theory"][section], dtype=np.float64)
        sim_ell = np.asarray(sim[name]["ell"], dtype=np.float64)
        sim_cl = np.asarray(sim[name]["cl"], dtype=np.float64)
        if data_cl.size != ell.size:
            raise ValueError(f"{name} has {data_cl.size} bands; expected {ell.size}.")
        if not np.array_equal(sim_ell, ell):
            raise ValueError(f"{name} simulation band centers do not exactly match the measurement.")

        data_window = np.asarray(measurement["windows"][name], dtype=np.float64)
        sim_window = np.asarray(sim[name]["window"], dtype=np.float64)
        if data_window.shape != sim_window.shape:
            raise ValueError(
                f"{name} survey and simulation bandpower windows have different shapes: "
                f"{data_window.shape} versus {sim_window.shape}."
            )
        window_row_l1 = np.sum(np.abs(data_window - sim_window), axis=1)
        active = likelihood_active_mask(config, panel, measurement)
        excluded_spans = excluded_band_spans(
            active,
            measurement["ell_left"],
            measurement["ell_right"],
        )
        arrays = {
            "data": data_cl,
            "data error": data_err,
            "theory": theory_cl,
            "simulation": sim_cl,
        }
        nonfinite = [label for label, values in arrays.items() if not np.all(np.isfinite(values))]
        if nonfinite:
            raise ValueError(f"{name} has non-finite values in: {', '.join(nonfinite)}.")

        spectra[name] = {
            "ell": ell,
            "data": dell(ell, data_cl),
            "data_err": dell(ell, data_err),
            "theory": dell(ell, theory_cl),
            "sim": dell(sim_ell, sim_cl),
            "likelihood_excluded_spans": excluded_spans,
        }
        report["spectra"][name] = {
            "theory_source": "full-survey-windowed",
            "ell_sha256": array_sha256(ell),
            "data_cl_sha256": array_sha256(data_cl),
            "data_error_cl_sha256": array_sha256(data_err),
            "theory_cl_sha256": array_sha256(theory_cl),
            "sim_cl_sha256": array_sha256(sim_cl),
            "data_window_sha256": array_sha256(data_window),
            "sim_window_sha256": array_sha256(sim_window),
            "window_row_l1_max": float(np.max(window_row_l1)),
            "window_row_l1_median": float(np.median(window_row_l1)),
            "likelihood_active_bands": int(np.count_nonzero(active)),
            "likelihood_excluded_spans": [list(span) for span in excluded_spans],
            "n_bands": int(ell.size),
        }
    return spectra, report


def render(
    spectra: Mapping[str, Mapping[str, Any]],
    measurement: Mapping[str, Any],
    output: Path,
) -> tuple[Path, Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    colors = {
        "data": "#20242B",
        "theory": "#0072B2",
        "sim": "#D55E00",
        "sim_edge": "#8B2D14",
        "zero": "#68717D",
        "grid": "#D8DEE6",
    }
    rc_params = {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12.8,
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
    with mpl.rc_context(rc_params):
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), constrained_layout=True)
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                ms=6.8,
                markerfacecolor="white",
                markeredgecolor=colors["data"],
                markeredgewidth=1.4,
                color=colors["data"],
                label="Data",
            ),
            Line2D(
                [0],
                [0],
                color=colors["theory"],
                lw=2.7,
                label="Theory (data matched)",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="--",
                lw=2.0,
                ms=6.8,
                markerfacecolor=colors["sim"],
                markeredgecolor=colors["sim_edge"],
                label="Simulations (theory matched)",
            ),
        ]

        for index, (ax, panel) in enumerate(zip(axes, PANELS)):
            name = str(panel["name"])
            values = spectra[name]
            ell = np.asarray(values["ell"])

            for lo, hi in values["likelihood_excluded_spans"]:
                ax.axvspan(
                    lo,
                    hi,
                    facecolor="#AEB4BC",
                    edgecolor="none",
                    alpha=0.36,
                    zorder=0,
                )

            ax.errorbar(
                ell,
                values["data"],
                yerr=values["data_err"],
                fmt="o",
                ms=5.2,
                lw=1.25,
                elinewidth=1.2,
                capsize=3.2,
                color=colors["data"],
                markerfacecolor="white",
                markeredgewidth=1.25,
                zorder=4,
            )
            ax.plot(ell, values["theory"], color=colors["theory"], lw=2.7, zorder=3)
            ax.plot(
                ell,
                values["sim"],
                linestyle="--",
                lw=2.0,
                marker="s",
                ms=4.7,
                color=colors["sim"],
                markeredgecolor=colors["sim_edge"],
                markeredgewidth=0.65,
                zorder=5,
            )

            ax.set_xscale("log")
            ax.set_xlim(float(measurement["ell_left"][0]) * 0.94, 3000.0)
            if bool(panel["log_y"]):
                if any(np.any(values[key] <= 0.0) for key in ("data", "theory", "sim")):
                    raise ValueError(f"{name} cannot use the requested logarithmic y axis.")
                ax.set_yscale("log")
            else:
                ax.axhline(0.0, color=colors["zero"], lw=1.0, alpha=0.8, zorder=1)
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 2))
                ax.yaxis.set_major_formatter(formatter)

            ax.set_xlabel(r"Multipole, $\ell$", labelpad=5)
            ax.set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/(2\pi)$", labelpad=7)
            ax.text(
                0.045,
                0.955,
                str(panel["label"]),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=16.5,
                linespacing=1.18,
                bbox={
                    "boxstyle": "round,pad=0.32",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
                zorder=8,
            )
            if index == 0:
                ax.legend(
                    handles=handles,
                    loc="lower right",
                    frameon=True,
                    fancybox=False,
                    facecolor="white",
                    edgecolor="#B8C0CA",
                    framealpha=0.95,
                    borderpad=0.7,
                    labelspacing=0.65,
                    handlelength=2.3,
                    handletextpad=0.7,
                )
            ax.grid(True, which="major", color=colors["grid"], lw=0.85, alpha=0.82)
            ax.grid(True, which="minor", axis="x", color=colors["grid"], lw=0.5, alpha=0.35)
            ax.tick_params(direction="out")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(y=0.12)

        output.parent.mkdir(parents=True, exist_ok=True)
        png = output.with_suffix(".png")
        fig.savefig(output, bbox_inches="tight")
        fig.savefig(png, dpi=320, bbox_inches="tight")
        plt.close(fig)
    return output, png


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(args, config)
    measurement = load_measurement(paths["measurement"])
    sim = load_simulation(paths["sim"])
    full_vector = load_full_theory_vector(paths["full_theory_vector"], measurement)
    spectra, report = build_series(measurement, sim, full_vector, config)
    report["schema"] = "stage31_pz3_publication_3panel_v1"
    report["inputs"] = {key: str(value) for key, value in paths.items() if key != "output"}
    report["layout"] = "one row of three panels; shared legend inside the gg panel"
    report["legend_labels"] = [
        "Data",
        "Theory (data matched)",
        "Simulations (theory matched)",
    ]
    report["comparison_semantics"] = {
        "data_and_theory": (
            "Saved full-survey data bandpowers and the MAP theory convolved through their "
            "saved survey bandpower windows."
        ),
        "simulations": (
            "Raw NaMaster cap-simulation bandpowers. They retain their own saved cap-estimator "
            "windows and are not re-windowed to the survey estimator."
        ),
        "legend_theory_matched_meaning": (
            "The simulation maps use the selected MAP physical parameters and simulation-matched "
            "map transfers; it does not assert equality of cap and survey bandpower windows."
        ),
    }

    if args.check_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    pdf, png = render(spectra, measurement, paths["output"])
    report_path = pdf.with_suffix(".provenance.json")
    report["outputs"] = {"pdf": str(pdf), "png": str(png), "provenance": str(report_path)}
    write_report(report_path, report)
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
