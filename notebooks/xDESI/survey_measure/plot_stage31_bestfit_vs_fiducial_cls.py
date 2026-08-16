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


def scalar_text(value: object, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must be a scalar string, got shape {array.shape}.")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def validate_cached_vector_product(
    payload: Mapping[str, object],
    path: Path,
    measurement: gmt.MeasurementData,
    *,
    expected_likelihood_identity: str | None = None,
    expected_config_identity: str | None = None,
    expected_theory_response_identity: str | None = None,
    expected_parameter_names: Sequence[str] | None = None,
    expected_parameter_contract_identity: str | None = None,
) -> None:
    """Bind a cached theory vector to the exact current measurement and likelihood."""

    required = {
        "ell_band",
        "data_vector",
        "theory_vector",
        "covariance",
        "spectrum_names",
        "slice_start",
        "slice_stop",
        "measurement_identity_sha256",
        "theory_vector_generation_json",
        "theory_vector_identity_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{path} is an unversioned cached vector missing {missing}.")
    names = decode_names(np.asarray(payload["spectrum_names"]).reshape(-1))
    saved_identity = scalar_text(
        payload["measurement_identity_sha256"],
        f"{path}:measurement_identity_sha256",
    )
    embedded_identity = gmt.measurement_data_identity_sha256(
        names=names,
        ell=payload["ell_band"],
        data_vector=payload["data_vector"],
        covariance=payload["covariance"],
        starts=payload["slice_start"],
        stops=payload["slice_stop"],
    )
    current_identity = gmt.measurement_identity_sha256(measurement)
    if saved_identity != embedded_identity:
        raise ValueError(f"{path} measurement fingerprint does not match its embedded arrays.")
    if saved_identity != current_identity:
        raise ValueError(f"{path} was built for a different measurement data/covariance basis.")
    exact_arrays = (
        ("ell_band", measurement.ell),
        ("data_vector", measurement.data_vector),
        ("covariance", measurement.covariance),
        ("slice_start", measurement.starts),
        ("slice_stop", measurement.stops),
    )
    for key, current in exact_arrays:
        if not np.array_equal(np.asarray(payload[key]), np.asarray(current)):
            raise ValueError(f"{path}:{key} is not exact for the current measurement.")
    if names != measurement.names:
        raise ValueError(f"{path} spectrum names do not match the current measurement order.")
    generation_json = scalar_text(
        payload["theory_vector_generation_json"],
        f"{path}:theory_vector_generation_json",
    )
    try:
        generation = json.loads(generation_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} theory-vector generation metadata is invalid JSON.") from exc
    if not isinstance(generation, dict) or not generation.get("product_kind"):
        raise ValueError(f"{path} has no theory-vector product_kind contract.")
    if generation.get("measurement_identity_sha256") != saved_identity:
        raise ValueError(f"{path} theory-vector metadata names a different measurement.")
    if (
        expected_config_identity is not None
        and generation.get("comparison_config_identity_sha256") != expected_config_identity
    ):
        raise ValueError(f"{path} was built for a different materialized comparison configuration.")
    if expected_theory_response_identity is not None:
        if "theory_response_identity_sha256" not in payload:
            raise ValueError(f"{path} has no saved theory-response fingerprint.")
        saved_response_identity = scalar_text(
            payload["theory_response_identity_sha256"],
            f"{path}:theory_response_identity_sha256",
        )
        if saved_response_identity != expected_theory_response_identity:
            raise ValueError(f"{path} was built for different saved theory-response content.")
        if (
            generation.get("theory_response_identity_sha256")
            != expected_theory_response_identity
        ):
            raise ValueError(f"{path} theory-vector metadata names different response content.")
    expected_vector_fields = gmt.theory_vector_cache_fields(
        payload["theory_vector"],
        saved_identity,
        {
            key: value
            for key, value in generation.items()
            if key != "measurement_identity_sha256"
        },
    )
    expected_vector_identity = scalar_text(
        expected_vector_fields["theory_vector_identity_sha256"],
        "recomputed theory_vector_identity_sha256",
    )
    saved_vector_identity = scalar_text(
        payload["theory_vector_identity_sha256"],
        f"{path}:theory_vector_identity_sha256",
    )
    if saved_vector_identity != expected_vector_identity:
        raise ValueError(f"{path} theory-vector fingerprint does not match its payload.")
    if not np.all(np.isfinite(np.asarray(payload["theory_vector"], dtype=np.float64))):
        raise ValueError(f"{path} theory vector contains non-finite values.")
    saved_sample = None
    if "best_sample_json" in payload:
        try:
            saved_sample = json.loads(scalar_text(payload["best_sample_json"], f"{path}:best_sample_json"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} best_sample_json is invalid JSON.") from exc
        if generation.get("best_sample") != saved_sample:
            raise ValueError(f"{path} theory-vector fingerprint is not linked to best_sample_json.")
    if "best_whitened_chi2" in payload:
        saved_chi2 = float(np.asarray(payload["best_whitened_chi2"]).reshape(-1)[0])
        if generation.get("best_whitened_chi2") != saved_chi2:
            raise ValueError(f"{path} theory-vector fingerprint is not linked to its saved chi2.")
    if expected_likelihood_identity is not None:
        if "likelihood_identity_sha256" not in payload:
            raise ValueError(f"{path} has no likelihood identity fingerprint.")
        saved_likelihood_identity = scalar_text(
            payload["likelihood_identity_sha256"],
            f"{path}:likelihood_identity_sha256",
        )
        if saved_likelihood_identity != expected_likelihood_identity:
            raise ValueError(f"{path} was built for a different Stage-31 likelihood.")
        if generation.get("likelihood_identity_sha256") != expected_likelihood_identity:
            raise ValueError(f"{path} theory-vector metadata names a different likelihood.")
        if generation.get("chain_contract_version") != hmc31.STAGE31_CHAIN_CONTRACT_VERSION:
            raise ValueError(f"{path} theory-vector metadata has a stale chain contract.")
    if (expected_parameter_names is None) != (expected_parameter_contract_identity is None):
        raise ValueError("Expected parameter names and parameter-contract identity must be supplied together.")
    if expected_parameter_names is not None:
        expected_names = [str(name) for name in expected_parameter_names]
        if "parameter_names" not in payload:
            raise ValueError(f"{path} has no ordered parameter_names contract.")
        saved_names = decode_names(np.asarray(payload["parameter_names"]).reshape(-1))
        if saved_names != expected_names:
            raise ValueError(f"{path} was built for a different ordered parameter contract.")
        if "parameter_contract_identity_sha256" not in payload:
            raise ValueError(f"{path} has no parameter/prior-contract fingerprint.")
        saved_parameter_contract = scalar_text(
            payload["parameter_contract_identity_sha256"],
            f"{path}:parameter_contract_identity_sha256",
        )
        if saved_parameter_contract != expected_parameter_contract_identity:
            raise ValueError(f"{path} was built for a different parameter/prior contract.")
        if generation.get("parameter_names") != expected_names:
            raise ValueError(f"{path} theory-vector metadata has a different parameter order.")
        if (
            generation.get("parameter_contract_identity_sha256")
            != expected_parameter_contract_identity
        ):
            raise ValueError(f"{path} theory-vector metadata has a different parameter/prior contract.")
        if not isinstance(saved_sample, dict) or list(saved_sample) != expected_names:
            raise ValueError(f"{path} best sample does not match the current ordered parameter contract.")


def validate_fit_summary_contract(
    summary: Mapping[str, object],
    path: Path,
    *,
    expected_likelihood_identity: str,
    expected_theory_response_identity: str,
    expected_parameter_names: Sequence[str],
    expected_parameter_contract_identity: str,
) -> None:
    """Fail closed on a fit summary from another likelihood or prior contract."""

    source_static = summary.get("static_summary")
    if not isinstance(source_static, Mapping):
        raise ValueError(f"{path} has no static likelihood identity.")
    if source_static.get("chain_contract_version") != hmc31.STAGE31_CHAIN_CONTRACT_VERSION:
        raise ValueError(f"{path} has a stale chain contract.")
    if source_static.get("likelihood_identity_sha256") != expected_likelihood_identity:
        raise ValueError(f"{path} was built for a different Stage-31 likelihood.")
    if (
        source_static.get("theory_response_identity_sha256")
        != expected_theory_response_identity
    ):
        raise ValueError(f"{path} was built for different saved theory-response content.")
    expected_names = [str(name) for name in expected_parameter_names]
    if list(source_static.get("parameter_names", [])) != expected_names:
        raise ValueError(f"{path} was built for a different ordered parameter contract.")
    if (
        source_static.get("parameter_contract_identity_sha256")
        != expected_parameter_contract_identity
    ):
        raise ValueError(f"{path} was built for a different parameter/prior contract.")


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
                    ylabel = r"$C_\ell$ (signal + shot noise)"
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
    measurement = hmc31.measurement_for_plots(context)
    fid_npz = load_vector_npz(fid_path)
    best_npz = load_vector_npz(best_path)
    current_likelihood_identity = hmc31.likelihood_identity(context.likelihood)
    current_config_identity = gmt.comparison_config_identity_sha256(context.config)
    current_theory_response_identity = gmt.theory_response_identity_sha256(context.config)
    current_parameter_names = [spec.name for spec in context.parameter_specs]
    current_parameter_contract_identity = hmc31.parameter_contract_identity_sha256(
        context.parameter_specs
    )
    validate_cached_vector_product(
        fid_npz,
        fid_path,
        measurement,
        expected_config_identity=current_config_identity,
        expected_theory_response_identity=current_theory_response_identity,
    )
    validate_cached_vector_product(
        best_npz,
        best_path,
        measurement,
        expected_likelihood_identity=current_likelihood_identity,
        expected_config_identity=current_config_identity,
        expected_theory_response_identity=current_theory_response_identity,
        expected_parameter_names=current_parameter_names,
        expected_parameter_contract_identity=current_parameter_contract_identity,
    )

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
        validate_fit_summary_contract(
            source_summary,
            summary_path,
            expected_likelihood_identity=current_likelihood_identity,
            expected_theory_response_identity=current_theory_response_identity,
            expected_parameter_names=current_parameter_names,
            expected_parameter_contract_identity=current_parameter_contract_identity,
        )

    retained_rank = int(context.likelihood.rank)
    n_varied = len(context.parameter_specs)
    chi2_reference_dof = retained_rank - n_varied
    if chi2_reference_dof <= 0:
        raise ValueError(
            f"Non-positive chi2 reference dof: rank={retained_rank}, n_varied={n_varied}."
        )
    chi2_reference_sigma = math.sqrt(2.0 * chi2_reference_dof)

    summary = {
        "config_path": config_path,
        "fiducial_vector_path": fid_path,
        "bestfit_vector_path": best_path,
        "source_fit_summary_path": summary_path if summary_path.exists() else None,
        "source_bestfit_whitened_chi2": source_summary.get("best_whitened_chi2"),
        "output_dir": output_dir,
        "n_spectra": len(measurement.names),
        "data_vector_size": int(measurement.data_vector.size),
        "retained_covariance_rank": retained_rank,
        "n_varied_parameters": n_varied,
        "chi2_reference_dof": chi2_reference_dof,
        "chi2_reference_expected": chi2_reference_dof,
        "chi2_reference_sigma": chi2_reference_sigma,
        "fiducial_whitened_chi2": full_fid,
        "fiducial_chi2_minus_expected_sigma": (
            full_fid - chi2_reference_dof
        ) / chi2_reference_sigma,
        "bestfit_whitened_chi2": full_best,
        "bestfit_chi2_minus_expected_sigma": (
            full_best - chi2_reference_dof
        ) / chi2_reference_sigma,
        "delta_whitened_chi2_best_minus_fiducial": full_best - full_fid,
        "chi2_interpretation": (
            "Absolute whitened chi2 values must be judged against retained covariance rank "
            "minus n_varied_parameters; the delta alone is not a goodness-of-fit result."
        ),
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
