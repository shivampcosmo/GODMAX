#!/usr/bin/env python
"""Overlay xDESI measurements, fiducial GODMAX, and Stage-31 best-fit Cls."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import godmax_multiprobe_hmc_stage31 as hmc31
import godmax_multiprobe_theory_utils as gmt


def resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    default_local = (
        "notebooks/xDESI/survey_measure/outputs/"
        "godmax_multiprobe_fast1024_true_nz_hmc_stage31_local"
    )
    p.add_argument("--config", default="param_files/xDESI/params_multiprobe_fast1024_hmc_stage31.yaml")
    p.add_argument(
        "--fiducial-vector",
        default=(
            "notebooks/xDESI/survey_measure/outputs/"
            "godmax_multiprobe_fast1024_true_nz/theory_data_vector_fast1024.npz"
        ),
    )
    p.add_argument("--bestfit-vector", default=f"{default_local}/bestfit_theory_data_vector_smoke_stage31.npz")
    p.add_argument("--fit-summary", default=f"{default_local}/fit_summary_smoke_stage31.json")
    p.add_argument("--output-dir", default=f"{default_local}/bestfit_vs_fiducial_cls_20260604")
    p.add_argument("--prefix", default="stage31_smoke_20260604_bestfit_vs_fiducial")
    return p


def load_vector_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as npz:
        return {key: npz[key] for key in npz.files}


def decode_names(names: Sequence[object]) -> list[str]:
    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]


def whitened_chi2(context: hmc31.FitContext, theory_vector: np.ndarray) -> float:
    data = np.asarray(context.likelihood.data_vector, dtype=np.float64)
    white = np.asarray(context.likelihood.whitener) @ (data - np.asarray(theory_vector, dtype=np.float64))
    return float(np.sum(white**2))


def family_block_stats(measurement: gmt.MeasurementData, fiducial: np.ndarray, bestfit: np.ndarray) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for family in sorted(set(measurement.families.values())):
        chunks = []
        nspec = 0
        for name, start, stop in zip(measurement.names, measurement.starts, measurement.stops):
            if measurement.families[name] == family:
                chunks.append(np.arange(int(start), int(stop), dtype=int))
                nspec += 1
        if not chunks:
            continue
        idx = np.concatenate(chunks)
        cov = measurement.covariance[np.ix_(idx, idx)]
        row = {"n_spectra": nspec, "n_data": int(idx.size)}
        for label, theory in (("fiducial", fiducial), ("bestfit", bestfit)):
            resid = measurement.data_vector[idx] - theory[idx]
            try:
                alpha = np.linalg.solve(cov, resid)
            except np.linalg.LinAlgError:
                alpha = np.linalg.pinv(cov) @ resid
            row[f"block_chi2_{label}"] = float(resid @ alpha)
        row["delta_block_chi2_best_minus_fiducial"] = row["block_chi2_bestfit"] - row["block_chi2_fiducial"]
        out[family] = row
    return out


def dell_factor(ell: np.ndarray) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64)
    return ell * (ell + 1.0) / (2.0 * math.pi)


def plot_overlays(
    measurement: gmt.MeasurementData,
    fiducial: np.ndarray,
    bestfit: np.ndarray,
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{prefix}_cls.pdf"
    pdf = PdfPages(pdf_path)
    outputs: list[Path] = []

    family_order = [
        "des_shear_EE",
        "act_y_des_shear_E",
        "desi_g_auto",
        "desi_g_act_y",
        "desi_g_des_shear_E",
        "desi_g_act_kappa",
        "desi_pi_act_T",
    ]
    try:
        for family in family_order:
            names = [name for name in measurement.names if measurement.families[name] == family]
            if not names:
                continue
            ncol = min(4, int(math.ceil(math.sqrt(len(names)))))
            nrow = int(math.ceil(len(names) / ncol))
            fig, axes = plt.subplots(
                nrow,
                ncol,
                figsize=(4.5 * ncol, 3.25 * nrow),
                squeeze=False,
                constrained_layout=True,
            )
            for ax, name in zip(axes.flat, names):
                i = measurement.names.index(name)
                start = int(measurement.starts[i])
                stop = int(measurement.stops[i])
                ell = np.asarray(measurement.ell, dtype=np.float64)
                data_cl = measurement.data_vector[start:stop]
                fid_cl = fiducial[start:stop]
                best_cl = bestfit[start:stop]
                err = np.sqrt(np.clip(np.diag(measurement.covariance[start:stop, start:stop]), 0.0, np.inf))

                if family == "desi_g_auto":
                    y_data, y_err = data_cl, err
                    y_fid, y_best = fid_cl, best_cl
                    ylabel = r"$C_\ell$ signal"
                else:
                    fac = dell_factor(ell)
                    sign = -1.0 if family == "desi_pi_act_T" else 1.0
                    scale = 1.0e3 if family == "desi_pi_act_T" else 1.0
                    y_data = sign * scale * fac * data_cl
                    y_err = scale * fac * err
                    y_fid = sign * scale * fac * fid_cl
                    y_best = sign * scale * fac * best_cl
                    ylabel = r"$D_\ell$"
                    if family == "desi_pi_act_T":
                        ylabel = r"$-10^3 D_\ell^{\pi T}$"

                ax.errorbar(
                    ell,
                    y_data,
                    yerr=y_err,
                    fmt="o",
                    ms=3.0,
                    lw=0.9,
                    color="#30343b",
                    alpha=0.9,
                    label="measurement",
                )
                ax.plot(ell, y_fid, "-", lw=1.35, color="#d07a00", label="fiducial")
                ax.plot(ell, y_best, "-", lw=1.55, color="#1f63b5", label="posterior best")
                ax.axhline(0.0, color="#777777", lw=0.7, alpha=0.55)
                if family == "desi_g_auto" and np.all(y_data > 0.0) and np.all(y_fid > 0.0) and np.all(y_best > 0.0):
                    ax.set_yscale("log")
                ax.grid(True, color="#d8dbe2", lw=0.7, alpha=0.75)
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(ylabel)
                ax.set_title(measurement.labels.get(name, name), fontsize=9)
                ax.legend(loc="best", fontsize=7, frameon=False)
            for ax in axes.flat[len(names) :]:
                ax.set_visible(False)
            title = f"{family}: measurement, fiducial, posterior best"
            if family == "desi_pi_act_T":
                title += " (positive kSZ convention)"
            fig.suptitle(title, fontsize=13)
            png_path = output_dir / f"{prefix}_{family}.png"
            fig.savefig(png_path, dpi=180)
            outputs.append(png_path)
            pdf.savefig(fig)
            plt.close(fig)
    finally:
        pdf.close()
    outputs.append(pdf_path)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config_path = resolve(args.config)
    fid_path = resolve(args.fiducial_vector)
    best_path = resolve(args.bestfit_vector)
    summary_path = resolve(args.fit_summary)
    output_dir = resolve(args.output_dir)

    context = hmc31.prepare_fit_context(config_path)
    measurement = gmt.load_measurement_data(context.config["paths"]["measurement_h5"])
    fid_npz = load_vector_npz(fid_path)
    best_npz = load_vector_npz(best_path)

    fid_names = decode_names(fid_npz["spectrum_names"])
    best_names = decode_names(best_npz["spectrum_names"])
    if fid_names != measurement.names:
        raise ValueError("Fiducial vector spectrum names do not match measurement order.")
    if best_names != measurement.names:
        raise ValueError("Best-fit vector spectrum names do not match measurement order.")
    if not np.allclose(fid_npz["data_vector"], measurement.data_vector):
        raise ValueError("Fiducial vector data does not match measurement vector.")
    if not np.allclose(best_npz["data_vector"], measurement.data_vector):
        raise ValueError("Best-fit vector data does not match measurement vector.")

    fiducial = np.asarray(fid_npz["theory_vector"], dtype=np.float64)
    bestfit = np.asarray(best_npz["theory_vector"], dtype=np.float64)
    if fiducial.shape != measurement.data_vector.shape or bestfit.shape != measurement.data_vector.shape:
        raise ValueError("Theory vector shape does not match measurement vector.")

    full_fid = whitened_chi2(context, fiducial)
    full_best = whitened_chi2(context, bestfit)
    families = family_block_stats(measurement, fiducial, bestfit)
    plot_paths = plot_overlays(measurement, fiducial, bestfit, output_dir, args.prefix)

    source_summary = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as handle:
            source_summary = json.load(handle)

    summary = {
        "config_path": config_path,
        "fiducial_vector_path": fid_path,
        "bestfit_vector_path": best_path,
        "source_fit_summary_path": summary_path if summary_path.exists() else None,
        "source_bestfit_whitened_chi2": source_summary.get("best_whitened_chi2"),
        "output_dir": output_dir,
        "n_spectra": len(measurement.names),
        "data_vector_size": int(measurement.data_vector.size),
        "fiducial_whitened_chi2": full_fid,
        "bestfit_whitened_chi2": full_best,
        "delta_whitened_chi2_best_minus_fiducial": full_best - full_fid,
        "families": families,
        "pdf": next(str(path) for path in plot_paths if path.suffix == ".pdf"),
        "pngs": [str(path) for path in plot_paths if path.suffix == ".png"],
    }
    summary_out = output_dir / f"{args.prefix}_summary.json"
    with open(summary_out, "w", encoding="utf-8") as handle:
        json.dump(gmt.to_jsonable(summary), handle, indent=2)

    print(json.dumps(gmt.to_jsonable({**summary, "summary": summary_out}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
