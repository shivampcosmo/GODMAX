#!/usr/bin/env python3
"""Render the Stage-31 pz3 five-panel publication figure from saved products."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "notebooks/xDESI/abacus_paste/"
    "stage31_pz3_cap2400_map64fcen_mmin11p147538_lmax3000_13log.selected.yaml"
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
    {
        "name": "desi_g_act_kappa_pz3",
        "family": "desi_g_act_kappa",
        "label": "(d)  Galaxy $\\times$ CMB lensing\n$g_3 \\times \\kappa_{\\rm CMB}$",
        "log_y": False,
    },
    {
        "name": "des_shear_EE_tomo3_tomo3",
        "family": "des_shear_EE",
        "label": "(e)  Cosmic shear auto\n$\\gamma_E^{(3)} \\times \\gamma_E^{(3)}$",
        "log_y": False,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--measurement", type=Path)
    parser.add_argument("--sim", type=Path)
    parser.add_argument("--shear-sim", type=Path)
    parser.add_argument("--full-theory-vector", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and print numerical digests without writing a figure.",
    )
    return parser.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if int(config["pasting"]["pz_bin"]) != 3:
        raise ValueError("This publication figure requires the pz3 configuration.")
    if not bool(config["godmax"].get("override_cosmology_from_catalog", False)):
        raise ValueError("Abacus-cosmology override is not enabled in the selected configuration.")
    return config


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
        "shear_sim": (
            args.shear_sim
            or run_root
            / "measurements"
            / (
                "sim_des_shear_EE_tomo3_tomo3_pz3_cap2400_map64fcen_"
                f"nside{nside}_lmax{lmax}_nbin{nbin}_log.h5"
            )
        ).expanduser().resolve(),
        "full_theory_vector": (
            args.full_theory_vector or vector_candidates[0]
        ).expanduser().resolve(),
        "output": (
            args.output
            or run_root
            / "plots"
            / f"{run_name}_publication_5panel_Dell.pdf"
        ).expanduser().resolve(),
    }
    missing = [f"{key}: {path}" for key, path in paths.items() if key != "output" and not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input file(s):\n" + "\n".join(missing))
    if paths["output"].suffix.lower() != ".pdf":
        raise ValueError("--output must name a PDF; a same-stem PNG is written alongside it.")
    return paths


def load_measurement(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        names = decode(h5["joint/spectrum_names"][:])
        starts = np.asarray(h5["joint/slice_start"][:], dtype=np.int64)
        stops = np.asarray(h5["joint/slice_stop"][:], dtype=np.int64)
        result = {
            "names": names,
            "ell": np.asarray(h5["joint/ell"][:], dtype=np.float64),
            "ell_left": np.asarray(h5["ell_left"][:], dtype=np.float64),
            "ell_right": np.asarray(h5["ell_right"][:], dtype=np.float64),
            "data": np.asarray(h5["joint/data_vector"][:], dtype=np.float64),
            "covariance": np.asarray(h5["joint/cov"][:], dtype=np.float64),
            "starts": starts,
            "stops": stops,
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


def load_simulation(*paths: Path) -> dict[str, dict[str, np.ndarray]]:
    spectra: dict[str, dict[str, np.ndarray]] = {}
    for path in paths:
        with h5py.File(path, "r") as h5:
            names = (
                decode(h5["joint/spectrum_names"][:])
                if "joint/spectrum_names" in h5
                else sorted(h5["spectra"].keys())
            )
            for name in names:
                if name in spectra:
                    raise ValueError(f"Duplicate simulation spectrum {name!r} across inputs.")
                spectra[name] = {
                    "ell": np.asarray(h5[f"spectra/{name}/ell"][:], dtype=np.float64),
                    "cl": np.asarray(h5[f"spectra/{name}/cl"][:], dtype=np.float64),
                    "window": np.asarray(
                        h5[f"spectra/{name}/bandpower_window_selected"][:],
                        dtype=np.float64,
                    ),
                }
    expected_sim = {str(panel["name"]) for panel in PANELS}
    missing = sorted(expected_sim.difference(spectra))
    if missing:
        raise KeyError(f"Simulation product is missing required spectra: {missing}")
    return spectra


def load_full_theory_vector(path: Path, measurement: Mapping[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as npz:
        names = decode(npz["spectrum_names"])
        data = np.asarray(npz["data_vector"], dtype=np.float64)
        covariance = np.asarray(npz["covariance"], dtype=np.float64)
        theory = np.asarray(npz["theory_vector"], dtype=np.float64)
    if names != measurement["names"]:
        raise ValueError("Full theory-vector spectrum ordering does not match the measurement product.")
    data_equal = bool(np.array_equal(data, measurement["data"]))
    covariance_equal = bool(np.array_equal(covariance, measurement["covariance"]))
    if not data_equal or not covariance_equal:
        raise ValueError("Full theory-vector measurement basis does not exactly match the HDF5 product.")
    if theory.shape != data.shape:
        raise ValueError("Full theory and data vectors have different shapes.")
    return {
        "names": names,
        "theory": theory,
        "data_equal": data_equal,
        "covariance_equal": covariance_equal,
    }


def measurement_slice(measurement: Mapping[str, Any], name: str) -> slice:
    index = measurement["names"].index(name)
    return slice(int(measurement["starts"][index]), int(measurement["stops"][index]))


def dell(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64)
    return ell * (ell + 1.0) * np.asarray(cl, dtype=np.float64) / (2.0 * math.pi)


def likelihood_active_mask(
    config: Mapping[str, Any],
    panel: Mapping[str, Any],
    measurement: Mapping[str, Any],
) -> np.ndarray:
    cuts = config.get("likelihood_cuts") or {}
    ell = np.asarray(measurement["ell"], dtype=np.float64)
    selection = str(cuts.get("band_selection", "center")).lower()
    if selection in {"left", "lower", "ell_left"}:
        basis = np.asarray(measurement["ell_left"], dtype=np.float64)
    elif selection in {"right", "upper", "ell_right"}:
        basis = np.asarray(measurement["ell_right"], dtype=np.float64)
    else:
        basis = ell

    name = str(panel["name"])
    family = str(panel["family"])

    def bound(which: str) -> float | None:
        for key in (name,):
            value = (cuts.get(f"spectrum_ell_{which}") or {}).get(key)
            if value is not None:
                return float(value)
        value = (cuts.get(f"family_ell_{which}") or {}).get(family)
        if value is not None:
            return float(value)
        value = cuts.get(f"default_ell_{which}")
        return None if value is None else float(value)

    keep = np.ones(ell.size, dtype=bool)
    ell_min = bound("min")
    ell_max = bound("max")
    if ell_min is not None:
        keep &= basis >= ell_min
    if ell_max is not None:
        keep &= basis <= ell_max
    return keep


def excluded_band_spans(
    active: np.ndarray,
    ell_left: np.ndarray,
    ell_right: np.ndarray,
) -> list[tuple[float, float]]:
    excluded = ~np.asarray(active, dtype=bool)
    left = np.asarray(ell_left, dtype=np.float64)
    right = np.asarray(ell_right, dtype=np.float64)
    spans: list[tuple[float, float]] = []
    start: int | None = None
    for index, is_excluded in enumerate(excluded):
        if is_excluded and start is None:
            start = index
        if start is not None and (not is_excluded or index == excluded.size - 1):
            stop = index if not is_excluded else index + 1
            spans.append((float(left[start]), float(right[stop - 1])))
            start = None
    return spans


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
        data_cl = np.asarray(measurement["data"][section], dtype=np.float64)
        data_err = np.sqrt(
            np.clip(np.diag(measurement["covariance"][section, section]), 0.0, np.inf)
        )
        ell = np.asarray(measurement["ell"], dtype=np.float64)
        if data_cl.size != ell.size:
            raise ValueError(f"{name} has {data_cl.size} bands; expected {ell.size}.")

        theory_ell = ell.copy()
        theory_cl = np.asarray(full_vector["theory"][section], dtype=np.float64)
        if not np.array_equal(theory_ell, ell):
            raise ValueError(f"{name} theory band centers do not exactly match the measurement.")

        sim_ell = np.asarray(sim[name]["ell"], dtype=np.float64)
        sim_cl = np.asarray(sim[name]["cl"], dtype=np.float64)
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
            "theory": dell(theory_ell, theory_cl),
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
        "legend.fontsize": 15,
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
        fig = plt.figure(figsize=(17.5, 10.2), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.10)
        axes = [fig.add_subplot(grid[index // 3, index % 3]) for index in range(5)]
        legend_ax = fig.add_subplot(grid[1, 2])
        legend_ax.axis("off")

        for ax, panel in zip(axes, PANELS):
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
            ax.plot(
                ell,
                values["theory"],
                color=colors["theory"],
                linestyle="-",
                lw=2.7,
                zorder=3,
            )
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
            if name == "des_shear_EE_tomo3_tomo3":
                ax.text(
                    0.045,
                    0.055,
                    r"simulation: $0.63<z_{\rm lens}<0.98$ only",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=12.5,
                    color=colors["sim_edge"],
                    zorder=8,
                )
            ax.grid(True, which="major", color=colors["grid"], lw=0.85, alpha=0.82)
            ax.grid(True, which="minor", axis="x", color=colors["grid"], lw=0.5, alpha=0.35)
            ax.tick_params(direction="out")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.margins(y=0.12)

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                ms=7.0,
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
                ms=7.2,
                markerfacecolor=colors["sim"],
                markeredgecolor=colors["sim_edge"],
                label="Simulations (theory matched)",
            ),
        ]
        legend_ax.legend(
            handles=handles,
            loc="center",
            frameon=True,
            fancybox=False,
            facecolor="white",
            edgecolor="#B8C0CA",
            framealpha=1.0,
            borderpad=1.05,
            labelspacing=1.05,
            handlelength=2.7,
            handletextpad=0.9,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        png = output.with_suffix(".png")
        fig.savefig(output, bbox_inches="tight")
        fig.savefig(png, dpi=320, bbox_inches="tight")
        plt.close(fig)
    return output, png


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(args, config)
    measurement = load_measurement(paths["measurement"])
    sim = load_simulation(paths["sim"], paths["shear_sim"])
    full_vector = load_full_theory_vector(paths["full_theory_vector"], measurement)
    spectra, report = build_series(measurement, sim, full_vector, config)
    report["schema"] = "stage31_pz3_publication_5panel_v2"
    report["inputs"] = {key: str(value) for key, value in paths.items() if key != "output"}
    report["layout"] = "three panels on top; two panels and common legend on bottom"
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
        "shear_auto_caveat": (
            "The simulated tomo3 shear auto is an E-only spin-2 proxy constructed from the "
            "cap-limited pz3 lens slice (0.63 < z_lens < 0.98), not a full line-of-sight shear map."
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
