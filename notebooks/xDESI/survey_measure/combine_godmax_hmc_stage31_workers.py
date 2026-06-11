#!/usr/bin/env python
"""Combine independent Stage-31 worker chains and save the global best fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import yaml

import godmax_multiprobe_hmc_stage31 as hmc31
import godmax_multiprobe_theory_utils as gmt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=hmc31.DEFAULT_STAGE31_CONFIG)
    parser.add_argument("--worker-dir", required=True)
    parser.add_argument("--pattern", default="worker_*/chain_stage31.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix", default="stage31_multigpu")
    return parser


def _load_chain(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _sample_keys(payload: dict) -> list[str]:
    return sorted(key for key in payload if key.startswith("sample__"))


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
) -> dict:
    if not chain_paths:
        raise FileNotFoundError("No worker chain files found.")

    payloads = [_load_chain(path) for path in chain_paths]
    keys = _sample_keys(payloads[0])
    for path, payload in zip(chain_paths, payloads):
        missing = sorted(set(keys) - set(payload))
        if missing:
            raise KeyError(f"{path} missing sample keys: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = {key: np.concatenate([payload[key] for payload in payloads], axis=0) for key in keys}
    for key in sorted(k for k in payloads[0] if k.startswith("extra__")):
        if all(key in payload for payload in payloads):
            combined[key] = np.concatenate([payload[key] for payload in payloads], axis=0)
    combined["parameter_names"] = np.asarray([spec.name for spec in context.parameter_specs])
    combined["worker_chain_paths"] = np.asarray([str(path) for path in chain_paths])

    if "sample__chi2" not in combined:
        raise KeyError("Combined chains do not contain sample__chi2.")
    chi2 = np.asarray(combined["sample__chi2"], dtype=np.float64)
    best_idx = int(np.nanargmin(chi2))
    best_chi2 = float(chi2[best_idx])
    best_sample = {
        spec.name: float(np.asarray(combined[f"sample__{spec.name}"])[best_idx])
        for spec in context.parameter_specs
    }
    convergence = convergence_diagnostics(payloads, [spec.name for spec in context.parameter_specs])

    chain_path = output_dir / f"chain_{suffix}.npz"
    combined["metadata_json"] = np.asarray(
        json.dumps(
            {
                "best_sample_index": best_idx,
                "best_whitened_chi2": best_chi2,
                "convergence": convergence,
                "worker_chain_paths": [str(path) for path in chain_paths],
                "convergence_diagnostics": convergence,
                "static_summary": hmc31.static_summary(context),
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
    theory_cls = hmc31.extract_theory_cls_jax_from_models(models)
    best_theory = np.asarray(hmc31.theory_data_vector_jax(context.likelihood, theory_cls))
    measurement = hmc31.measurement_for_plots(context)
    stats = gmt.comparison_statistics(measurement, best_theory)
    full_likelihood = hmc31.full_likelihood_for_plots(context)
    full_measurement = hmc31.measurement_from_likelihood(context, full_likelihood)
    full_best_theory = np.asarray(hmc31.theory_data_vector_jax(full_likelihood, theory_cls))

    theory_path = output_dir / f"bestfit_theory_data_vector_{suffix}.npz"
    np.savez_compressed(
        theory_path,
        ell_band=np.asarray(measurement.ell),
        data_vector=np.asarray(measurement.data_vector),
        theory_vector=best_theory,
        covariance=np.asarray(measurement.covariance),
        spectrum_names=np.asarray(measurement.names),
        best_sample_json=np.asarray(json.dumps(best_sample)),
        best_whitened_chi2=np.asarray(best_chi2),
    )

    full_theory_path = output_dir / f"bestfit_full_theory_data_vector_{suffix}.npz"
    np.savez_compressed(
        full_theory_path,
        ell_band=np.asarray(full_measurement.ell),
        data_vector=np.asarray(full_measurement.data_vector),
        theory_vector=full_best_theory,
        covariance=np.asarray(full_measurement.covariance),
        spectrum_names=np.asarray(full_measurement.names),
        best_sample_json=np.asarray(json.dumps(best_sample)),
        best_whitened_chi2=np.asarray(best_chi2),
        likelihood_bestfit_theory_vector=np.asarray(str(theory_path)),
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
    )
    full_dell_pdf_path = output_dir / f"posterior_predictive_full_dell_comparison_{suffix}.pdf"
    full_dell_plot_paths = gmt.plot_family_dell_comparisons(
        full_measurement,
        full_best_theory,
        output_dir,
        pdf_path=full_dell_pdf_path,
        filename_prefix=f"posterior_predictive_full_dell_{suffix}",
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
        "best_whitened_chi2": best_chi2,
        "convergence": convergence,
        "convergence_diagnostics": convergence,
        "pseudo_inverse_stats": stats,
        "worker_chain_paths": [str(path) for path in chain_paths],
        "static_summary": hmc31.static_summary(context),
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
        "convergence": convergence,
        "convergence_diagnostics": convergence,
        "n_samples_total": int(chi2.size),
        "n_workers": len(chain_paths),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    context = hmc31.prepare_fit_context(args.config)
    worker_dir = Path(args.worker_dir)
    chain_paths = sorted(worker_dir.glob(args.pattern))
    result = combine_worker_chains(context, chain_paths, Path(args.output_dir), args.suffix)
    print(json.dumps(gmt.to_jsonable(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
