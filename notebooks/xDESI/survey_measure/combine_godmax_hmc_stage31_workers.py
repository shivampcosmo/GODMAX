#!/usr/bin/env python
"""Combine independent Stage-31 worker chains and save the global best fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

import godmax_multiprobe_hmc_stage31 as hmc31
import godmax_multiprobe_theory_utils as gmt

DEFAULT_PLOT_ELL_MAX = 2800.0
DEFAULT_KSZ_YLIM = (-5.0e-5, 5.0e-5)
DEFAULT_KSZ_YLIM_ARG = f"{DEFAULT_KSZ_YLIM[0]},{DEFAULT_KSZ_YLIM[1]}"
DEFAULT_PLOT_XSCALE = "linear"
CHAIN_CHI2_RTOL = 1.0e-8
CHAIN_CHI2_ATOL = 1.0e-8


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=hmc31.DEFAULT_STAGE31_CONFIG)
    parser.add_argument("--worker-dir", required=True)
    parser.add_argument("--pattern", default="worker_*/chain_stage31.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix", default="stage31_multigpu")
    parser.add_argument(
        "--plot-ell-max",
        type=float,
        default=DEFAULT_PLOT_ELL_MAX,
        help="Maximum ell shown in posterior predictive D_ell plots. Use <=0 to show all available bandpowers.",
    )
    parser.add_argument(
        "--plot-ksz-ylim",
        default=DEFAULT_KSZ_YLIM_ARG,
        metavar="YMIN,YMAX",
        help="Y-axis limits for the kSZ pi x T D_ell panel.",
    )
    parser.add_argument(
        "--plot-ksz-scale",
        type=float,
        default=1.0,
        help="Multiplicative display scale for the kSZ pi x T D_ell panel.",
    )
    parser.add_argument(
        "--plot-xscale",
        default=DEFAULT_PLOT_XSCALE,
        choices=("linear", "log", "symlog"),
        help="X-axis scaling for posterior predictive D_ell plots.",
    )
    parser.add_argument(
        "--plot-xlim",
        default=None,
        metavar="XMIN,XMAX",
        help="Optional x-axis limits for posterior predictive D_ell plots.",
    )
    return parser


def normalize_plot_ksz_ylim_args(argv: Optional[Sequence[str]]) -> list[str]:
    """Allow negative scientific-notation y-limits after --plot-ksz-ylim."""

    raw = list(sys.argv[1:] if argv is None else argv)
    out: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "--plot-ksz-ylim" and i + 2 < len(raw):
            out.append(f"--plot-ksz-ylim={raw[i + 1]},{raw[i + 2]}")
            i += 3
            continue
        if raw[i].startswith("--plot-ksz-ylim=") and i + 1 < len(raw):
            option, value = raw[i].split("=", 1)
            if "," not in value:
                out.append(f"{option}={value},{raw[i + 1]}")
                i += 2
                continue
        out.append(raw[i])
        i += 1
    return out


def parse_plot_ksz_ylim(value: object) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    parts = str(value).replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"--plot-ksz-ylim must contain two values, got {value!r}.")
    return (float(parts[0]), float(parts[1]))


def parse_plot_xlim(value: object) -> Optional[tuple[float, float]]:
    if value is None or str(value).strip() == "":
        return None
    parts = str(value).replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"--plot-xlim must contain two values, got {value!r}.")
    lo, hi = float(parts[0]), float(parts[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or not lo < hi:
        raise ValueError(f"--plot-xlim must be finite and increasing, got {value!r}.")
    return (lo, hi)


def _load_chain(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _sample_keys(payload: dict) -> list[str]:
    return sorted(key for key in payload if key.startswith("sample__"))


def _scalar_text(value: object, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must contain exactly one scalar string, got {array.shape}.")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _validate_chain_payload(
    path: Path,
    payload: dict,
    context: hmc31.FitContext,
    *,
    expected_static_summary: Optional[dict] = None,
) -> dict:
    """Fail closed unless a worker chain is bound to this exact likelihood."""

    expected_names = [spec.name for spec in context.parameter_specs]
    if "parameter_names" not in payload:
        raise ValueError(f"{path} has no parameter_names contract.")
    actual_names = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(payload["parameter_names"]).reshape(-1)
    ]
    if actual_names != expected_names:
        raise ValueError(
            f"{path} parameter_names do not match the current ordered parameter contract: "
            f"saved={actual_names}, current={expected_names}."
        )

    required_sample_keys = {f"sample__{name}" for name in expected_names}
    required_sample_keys.add("sample__chi2")
    actual_sample_keys = set(_sample_keys(payload))
    if actual_sample_keys != required_sample_keys:
        raise ValueError(
            f"{path} sample keys do not match the current chain contract: "
            f"missing={sorted(required_sample_keys - actual_sample_keys)}, "
            f"unexpected={sorted(actual_sample_keys - required_sample_keys)}."
        )
    sample_lengths = {}
    for key in sorted(required_sample_keys):
        array = np.asarray(payload[key])
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{path}:{key} must be a non-empty 1D flattened chain.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{path}:{key} contains non-finite samples.")
        sample_lengths[key] = int(array.size)
    if len(set(sample_lengths.values())) != 1:
        raise ValueError(f"{path} has inconsistent sample lengths: {sample_lengths}.")

    if "metadata_json" not in payload:
        raise ValueError(f"{path} has no metadata_json likelihood identity.")
    try:
        metadata = json.loads(_scalar_text(payload["metadata_json"], f"{path}:metadata_json"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} metadata_json is not valid JSON.") from exc
    static = metadata.get("static_summary")
    if not isinstance(static, dict):
        raise ValueError(f"{path} metadata_json has no static_summary mapping.")
    expected = expected_static_summary or hmc31.static_summary(context)
    identity_keys = (
        "chain_contract_version",
        "likelihood_identity_sha256",
        "theory_response_identity_sha256",
        "parameter_contract_identity_sha256",
        "measurement_path",
        "measurement_map_product_id",
        "desi_galaxy_auto_mean_convention",
    )
    for key in identity_keys:
        if key not in static:
            raise ValueError(f"{path} static_summary is missing required identity key {key!r}.")
        if static[key] != expected.get(key):
            raise ValueError(
                f"{path} likelihood identity mismatch for {key}: "
                f"saved={static[key]!r}, current={expected.get(key)!r}."
            )
    if list(static.get("parameter_names", [])) != expected_names:
        raise ValueError(f"{path} static_summary parameter_names do not match the current fit.")
    expected_parameter_specs = hmc31.parameter_specs_jsonable(context.parameter_specs)
    expected_parameter_contract = hmc31.parameter_contract_identity_sha256(
        context.parameter_specs
    )
    if metadata.get("parameter_specs") != expected_parameter_specs:
        raise ValueError(f"{path} parameter_specs do not match the current prior contract.")
    if "parameter_contract_identity_sha256" not in payload:
        raise ValueError(f"{path} has no parameter-contract fingerprint.")
    saved_parameter_contract = _scalar_text(
        payload["parameter_contract_identity_sha256"],
        f"{path}:parameter_contract_identity_sha256",
    )
    if saved_parameter_contract != expected_parameter_contract:
        raise ValueError(f"{path} was built with a different parameter/prior contract.")
    return metadata


def _validated_recomputed_chi2(cached: float, recomputed: float) -> float:
    if not np.isfinite(cached) or not np.isfinite(recomputed):
        raise ValueError(
            f"Best-fit chi2 must be finite: cached={cached!r}, recomputed={recomputed!r}."
        )
    if not np.isclose(
        cached,
        recomputed,
        rtol=CHAIN_CHI2_RTOL,
        atol=CHAIN_CHI2_ATOL,
    ):
        raise ValueError(
            "Cached worker chi2 does not reproduce under the current likelihood: "
            f"cached={cached:.17g}, recomputed={recomputed:.17g}, "
            f"abs_delta={abs(cached - recomputed):.6g}."
        )
    return float(recomputed)


def _split_rhat(chains: np.ndarray) -> float:
    chains = np.asarray(chains, dtype=np.float64)
    if chains.ndim != 2:
        raise ValueError(f"Expected 2D chain array, got shape {chains.shape}.")
    n_chain, n_draw = chains.shape
    half = n_draw // 2
    if n_chain < 2 or half < 2:
        return float("nan")
    split = np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)
    n = split.shape[1]
    chain_means = np.mean(split, axis=1)
    chain_vars = np.var(split, axis=1, ddof=1)
    w = float(np.mean(chain_vars))
    b = float(n * np.var(chain_means, ddof=1))
    var_hat = ((n - 1.0) / n) * w + b / n
    if not np.isfinite(w) or w <= 0.0:
        return 1.0 if np.isfinite(var_hat) and abs(var_hat) < 1.0e-30 else float("nan")
    return float(np.sqrt(max(var_hat / w, 0.0)))


def convergence_diagnostics(payloads: Sequence[dict], parameter_names: Sequence[str]) -> dict:
    """Approximate convergence diagnostics from independent worker streams.

    Worker NPZ files currently store vectorized chains flattened within each
    GPU worker. This treats the four worker streams as independent chains and
    reports split-R-hat across them, plus sampler extra-field summaries.
    """

    n_workers = len(payloads)
    sample_draws = {
        f"sample__{name}": [np.asarray(payload[f"sample__{name}"], dtype=np.float64).reshape(-1) for payload in payloads]
        for name in list(parameter_names) + ["chi2"]
        if all(f"sample__{name}" in payload for payload in payloads)
    }
    min_draws = min((min(len(x) for x in rows) for rows in sample_draws.values()), default=0)
    rhat_by_name = {}
    for key, rows in sample_draws.items():
        if min_draws < 4:
            rhat_by_name[key.removeprefix("sample__")] = float("nan")
            continue
        chains = np.stack([row[:min_draws] for row in rows], axis=0)
        rhat_by_name[key.removeprefix("sample__")] = _split_rhat(chains)
    finite_rhats = np.asarray([value for value in rhat_by_name.values() if np.isfinite(value)], dtype=np.float64)
    threshold = 1.05
    extra = {}
    if all("extra__diverging" in payload for payload in payloads):
        diverging = np.concatenate([np.asarray(payload["extra__diverging"]).reshape(-1) for payload in payloads])
        extra["total_divergences"] = int(np.sum(diverging.astype(bool)))
    else:
        extra["total_divergences"] = None
    if all("extra__accept_prob" in payload for payload in payloads):
        accept = np.concatenate([np.asarray(payload["extra__accept_prob"], dtype=np.float64).reshape(-1) for payload in payloads])
        extra["accept_prob_mean"] = float(np.nanmean(accept))
        extra["accept_prob_min"] = float(np.nanmin(accept))
        extra["accept_prob_max"] = float(np.nanmax(accept))
    if all("extra__num_steps" in payload for payload in payloads):
        steps = np.concatenate([np.asarray(payload["extra__num_steps"], dtype=np.float64).reshape(-1) for payload in payloads])
        extra["num_steps_mean"] = float(np.nanmean(steps))
        extra["num_steps_max"] = float(np.nanmax(steps))
    max_rhat = float(np.max(finite_rhats)) if finite_rhats.size else float("nan")
    n_over = int(np.sum(finite_rhats > threshold)) if finite_rhats.size else None
    return {
        "method": "split_rhat_across_flattened_worker_streams",
        "note": (
            "Each worker NPZ flattens its vectorized chains; diagnostics treat workers "
            "as independent chains and do not recover within-worker chain R-hat."
        ),
        "n_workers": int(n_workers),
        "draws_per_worker_used": int(min_draws),
        "rhat_threshold": threshold,
        "max_split_rhat": max_rhat,
        "n_split_rhat_over_threshold": n_over,
        "split_rhat_by_name": rhat_by_name,
        "sampler_extra": extra,
        "passes_basic_gate": bool(
            np.isfinite(max_rhat)
            and max_rhat < threshold
            and extra.get("total_divergences") in (0, None)
        ),
    }


def combine_worker_chains(
    context: hmc31.FitContext,
    chain_paths: Sequence[Path],
    output_dir: Path,
    suffix: str,
    *,
    plot_ell_max: Optional[float] = DEFAULT_PLOT_ELL_MAX,
    plot_ksz_ylim: Optional[tuple[float, float]] = DEFAULT_KSZ_YLIM,
    plot_ksz_scale: float = 1.0,
    plot_xscale: str = DEFAULT_PLOT_XSCALE,
    plot_xlim: Optional[tuple[float, float]] = None,
) -> dict:
    if not chain_paths:
        raise FileNotFoundError("No worker chain files found.")

    payloads = [_load_chain(path) for path in chain_paths]
    expected_static_summary = hmc31.static_summary(context)
    for path, payload in zip(chain_paths, payloads):
        _validate_chain_payload(
            path,
            payload,
            context,
            expected_static_summary=expected_static_summary,
        )
    keys = _sample_keys(payloads[0])

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = {key: np.concatenate([payload[key] for payload in payloads], axis=0) for key in keys}
    for key in sorted(k for k in payloads[0] if k.startswith("extra__")):
        if all(key in payload for payload in payloads):
            combined[key] = np.concatenate([payload[key] for payload in payloads], axis=0)
    combined["parameter_names"] = np.asarray([spec.name for spec in context.parameter_specs])
    combined["parameter_contract_identity_sha256"] = np.asarray(
        hmc31.parameter_contract_identity_sha256(context.parameter_specs)
    )
    combined["worker_chain_paths"] = np.asarray([str(path) for path in chain_paths])

    if "sample__chi2" not in combined:
        raise KeyError("Combined chains do not contain sample__chi2.")
    chi2 = np.asarray(combined["sample__chi2"], dtype=np.float64)
    best_idx = int(np.nanargmin(chi2))
    cached_best_chi2 = float(chi2[best_idx])
    best_sample = {
        spec.name: float(np.asarray(combined[f"sample__{spec.name}"])[best_idx])
        for spec in context.parameter_specs
    }
    best_theory = np.asarray(hmc31.evaluate_sample_theory_vector(context, best_sample))
    recomputed_best_chi2 = float(
        np.asarray(hmc31.whitened_chi2(context.likelihood, best_theory))
    )
    best_chi2 = _validated_recomputed_chi2(
        cached_best_chi2,
        recomputed_best_chi2,
    )
    chi2_n_modes = int(context.likelihood.rank)
    n_fit_parameters = len(context.parameter_specs)
    chi2_dof = max(chi2_n_modes - n_fit_parameters, 1)
    reduced_chi2 = float(best_chi2) / float(chi2_dof)
    chi2_per_mode = float(best_chi2) / max(float(chi2_n_modes), 1.0)
    convergence = convergence_diagnostics(payloads, [spec.name for spec in context.parameter_specs])

    chain_path = output_dir / f"chain_{suffix}.npz"
    combined["metadata_json"] = np.asarray(
        json.dumps(
            {
                "best_sample_index": best_idx,
                "best_cached_whitened_chi2": cached_best_chi2,
                "best_recomputed_whitened_chi2": recomputed_best_chi2,
                "best_chi2_abs_delta": abs(cached_best_chi2 - recomputed_best_chi2),
                "best_whitened_chi2": best_chi2,
                "best_reduced_chi2": reduced_chi2,
                "best_chi2_dof": chi2_dof,
                "best_chi2_per_mode": chi2_per_mode,
                "chi2_n_modes": chi2_n_modes,
                "n_fit_parameters": n_fit_parameters,
                "convergence": convergence,
                "worker_chain_paths": [str(path) for path in chain_paths],
                "convergence_diagnostics": convergence,
                "static_summary": expected_static_summary,
                "parameter_specs": hmc31.parameter_specs_jsonable(context.parameter_specs),
            }
        )
    )
    np.savez_compressed(chain_path, **combined)

    best_config = hmc31.apply_sample_to_config(context.config, context.parameter_specs, best_sample)
    best_params_path = output_dir / f"bestfit_params_{suffix}.yaml"
    with open(best_params_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(gmt.to_jsonable(best_config["params"]), handle, sort_keys=False)

    models = hmc31.build_models_from_sample(context, best_sample)
    theory_cls = hmc31._dense_theory_cls_from_models(context, models)
    best_shot_noise = hmc31._sampled_shot_noise_amplitudes(best_sample)
    measurement = hmc31.measurement_for_plots(context)
    stats = gmt.comparison_statistics(measurement, best_theory)
    full_likelihood = hmc31.full_likelihood_for_plots(context)
    full_measurement = hmc31.measurement_from_likelihood(context, full_likelihood)
    full_best_theory = np.asarray(
        hmc31.theory_data_vector_jax(
            full_likelihood,
            theory_cls,
            desi_galaxy_shot_noise_amplitudes=best_shot_noise,
        )
    )
    measurement_identity = gmt.measurement_identity_sha256(measurement)
    full_measurement_identity = gmt.measurement_identity_sha256(full_measurement)
    active_likelihood_identity = hmc31.likelihood_identity(context.likelihood)
    full_likelihood_identity = hmc31.likelihood_identity(full_likelihood)
    comparison_config_identity = gmt.comparison_config_identity_sha256(context.config)
    theory_response_identity = gmt.theory_response_identity_sha256(context.config)
    parameter_names = [spec.name for spec in context.parameter_specs]
    parameter_contract_identity = hmc31.parameter_contract_identity_sha256(
        context.parameter_specs
    )
    active_vector_cache_fields = gmt.theory_vector_cache_fields(
        best_theory,
        measurement_identity,
        {
            "product_kind": "stage31_combined_bestfit_active",
            "chain_contract_version": hmc31.STAGE31_CHAIN_CONTRACT_VERSION,
            "likelihood_identity_sha256": active_likelihood_identity,
            "comparison_config_identity_sha256": comparison_config_identity,
            "theory_response_identity_sha256": theory_response_identity,
            "parameter_names": parameter_names,
            "parameter_contract_identity_sha256": parameter_contract_identity,
            "best_sample": best_sample,
            "best_whitened_chi2": float(best_chi2),
        },
    )
    full_vector_cache_fields = gmt.theory_vector_cache_fields(
        full_best_theory,
        full_measurement_identity,
        {
            "product_kind": "stage31_combined_bestfit_full",
            "chain_contract_version": hmc31.STAGE31_CHAIN_CONTRACT_VERSION,
            "likelihood_identity_sha256": full_likelihood_identity,
            "comparison_config_identity_sha256": comparison_config_identity,
            "theory_response_identity_sha256": theory_response_identity,
            "parameter_names": parameter_names,
            "parameter_contract_identity_sha256": parameter_contract_identity,
            "source_active_likelihood_identity_sha256": active_likelihood_identity,
            "best_sample": best_sample,
            "best_whitened_chi2": float(best_chi2),
        },
    )

    theory_path = output_dir / f"bestfit_theory_data_vector_{suffix}.npz"
    np.savez_compressed(
        theory_path,
        ell_band=np.asarray(measurement.ell),
        data_vector=np.asarray(measurement.data_vector),
        theory_vector=best_theory,
        covariance=np.asarray(measurement.covariance),
        spectrum_names=np.asarray(measurement.names),
        slice_start=np.asarray(measurement.starts, dtype=np.int64),
        slice_stop=np.asarray(measurement.stops, dtype=np.int64),
        measurement_identity_sha256=np.asarray(measurement_identity),
        likelihood_identity_sha256=np.asarray(active_likelihood_identity),
        chain_contract_version=np.asarray(hmc31.STAGE31_CHAIN_CONTRACT_VERSION),
        theory_response_identity_sha256=np.asarray(theory_response_identity),
        parameter_names=np.asarray(parameter_names),
        parameter_contract_identity_sha256=np.asarray(parameter_contract_identity),
        best_sample_json=np.asarray(json.dumps(best_sample)),
        best_whitened_chi2=np.asarray(best_chi2),
        **active_vector_cache_fields,
    )

    full_theory_path = output_dir / f"bestfit_full_theory_data_vector_{suffix}.npz"
    np.savez_compressed(
        full_theory_path,
        ell_band=np.asarray(full_measurement.ell),
        data_vector=np.asarray(full_measurement.data_vector),
        theory_vector=full_best_theory,
        covariance=np.asarray(full_measurement.covariance),
        spectrum_names=np.asarray(full_measurement.names),
        slice_start=np.asarray(full_measurement.starts, dtype=np.int64),
        slice_stop=np.asarray(full_measurement.stops, dtype=np.int64),
        measurement_identity_sha256=np.asarray(full_measurement_identity),
        likelihood_identity_sha256=np.asarray(full_likelihood_identity),
        source_active_likelihood_identity_sha256=np.asarray(active_likelihood_identity),
        chain_contract_version=np.asarray(hmc31.STAGE31_CHAIN_CONTRACT_VERSION),
        theory_response_identity_sha256=np.asarray(theory_response_identity),
        parameter_names=np.asarray(parameter_names),
        parameter_contract_identity_sha256=np.asarray(parameter_contract_identity),
        best_sample_json=np.asarray(json.dumps(best_sample)),
        best_whitened_chi2=np.asarray(best_chi2),
        likelihood_bestfit_theory_vector=np.asarray(str(theory_path)),
        **full_vector_cache_fields,
    )

    pdf_path = output_dir / f"posterior_predictive_comparison_{suffix}.pdf"
    plot_paths = gmt.plot_family_comparisons(measurement, best_theory, output_dir, pdf_path=pdf_path)
    dell_pdf_path = output_dir / f"posterior_predictive_dell_comparison_{suffix}.pdf"
    dell_plot_paths = gmt.plot_family_dell_comparisons(
        measurement,
        best_theory,
        output_dir,
        pdf_path=dell_pdf_path,
        filename_prefix=f"posterior_predictive_dell_{suffix}",
        ell_max=plot_ell_max,
        ksz_ylim=plot_ksz_ylim,
        ksz_scale=plot_ksz_scale,
        total_reduced_chi2=reduced_chi2,
        chi2_dof=chi2_dof,
        xscale=plot_xscale,
        xlim=plot_xlim,
    )
    full_dell_pdf_path = output_dir / f"posterior_predictive_full_dell_comparison_{suffix}.pdf"
    full_dell_plot_paths = gmt.plot_family_dell_comparisons(
        full_measurement,
        full_best_theory,
        output_dir,
        pdf_path=full_dell_pdf_path,
        filename_prefix=f"posterior_predictive_full_dell_{suffix}",
        ell_max=plot_ell_max,
        ksz_ylim=plot_ksz_ylim,
        ksz_scale=plot_ksz_scale,
        active_band_indices=hmc31.likelihood_active_band_indices(context),
        total_reduced_chi2=reduced_chi2,
        chi2_dof=chi2_dof,
        xscale=plot_xscale,
        xlim=plot_xlim,
    )

    summary_path = output_dir / f"fit_summary_{suffix}.json"
    summary = {
        "chain_path": chain_path,
        "bestfit_params_path": best_params_path,
        "bestfit_theory_vector_path": theory_path,
        "bestfit_full_theory_vector_path": full_theory_path,
        "posterior_predictive_pdf": pdf_path,
        "posterior_predictive_dell_pdf": dell_pdf_path,
        "posterior_predictive_full_dell_pdf": full_dell_pdf_path,
        "plot_paths": plot_paths,
        "dell_plot_paths": dell_plot_paths,
        "full_dell_plot_paths": full_dell_plot_paths,
        "best_sample_index": best_idx,
        "best_sample": best_sample,
        "best_cached_whitened_chi2": cached_best_chi2,
        "best_recomputed_whitened_chi2": recomputed_best_chi2,
        "best_chi2_abs_delta": abs(cached_best_chi2 - recomputed_best_chi2),
        "best_whitened_chi2": best_chi2,
        "best_reduced_chi2": reduced_chi2,
        "best_chi2_dof": chi2_dof,
        "best_chi2_per_mode": chi2_per_mode,
        "chi2_n_modes": chi2_n_modes,
        "n_fit_parameters": n_fit_parameters,
        "convergence": convergence,
        "convergence_diagnostics": convergence,
        "pseudo_inverse_stats": stats,
        "worker_chain_paths": [str(path) for path in chain_paths],
        "n_samples_total": int(chi2.size),
        "n_workers": len(chain_paths),
        "plot_settings": {
            "dell_ell_max": plot_ell_max,
            "ksz_ylim": plot_ksz_ylim,
            "ksz_scale": float(plot_ksz_scale),
            "xscale": plot_xscale,
            "xlim": plot_xlim,
        },
        "static_summary": expected_static_summary,
        "parameter_specs": hmc31.parameter_specs_jsonable(context.parameter_specs),
        "priors": context.prior_config,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(gmt.to_jsonable(summary), handle, indent=2)

    return {
        "chain": chain_path,
        "bestfit_params": best_params_path,
        "bestfit_theory_vector": theory_path,
        "bestfit_full_theory_vector": full_theory_path,
        "summary": summary_path,
        "pdf": pdf_path,
        "dell_pdf": dell_pdf_path,
        "full_dell_pdf": full_dell_pdf_path,
        "plots": plot_paths,
        "dell_plots": dell_plot_paths,
        "full_dell_plots": full_dell_plot_paths,
        "best_whitened_chi2": best_chi2,
        "best_reduced_chi2": reduced_chi2,
        "best_chi2_dof": chi2_dof,
        "best_chi2_per_mode": chi2_per_mode,
        "chi2_n_modes": chi2_n_modes,
        "n_fit_parameters": n_fit_parameters,
        "convergence": convergence,
        "convergence_diagnostics": convergence,
        "n_samples_total": int(chi2.size),
        "n_workers": len(chain_paths),
        "plot_settings": {
            "dell_ell_max": plot_ell_max,
            "ksz_ylim": plot_ksz_ylim,
            "ksz_scale": float(plot_ksz_scale),
            "xscale": plot_xscale,
            "xlim": plot_xlim,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(normalize_plot_ksz_ylim_args(argv))
    plot_ell_max = None if args.plot_ell_max is not None and args.plot_ell_max <= 0.0 else args.plot_ell_max
    plot_ksz_ylim = parse_plot_ksz_ylim(args.plot_ksz_ylim)
    plot_xlim = parse_plot_xlim(args.plot_xlim)
    context = hmc31.prepare_fit_context(args.config)
    worker_dir = Path(args.worker_dir)
    chain_paths = sorted(worker_dir.glob(args.pattern))
    result = combine_worker_chains(
        context,
        chain_paths,
        Path(args.output_dir),
        args.suffix,
        plot_ell_max=plot_ell_max,
        plot_ksz_ylim=plot_ksz_ylim,
        plot_ksz_scale=float(args.plot_ksz_scale),
        plot_xscale=str(args.plot_xscale),
        plot_xlim=plot_xlim,
    )
    print(json.dumps(gmt.to_jsonable(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
