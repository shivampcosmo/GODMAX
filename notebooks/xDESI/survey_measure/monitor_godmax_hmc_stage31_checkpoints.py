#!/usr/bin/env python
"""Combine Stage-31 HMC worker checkpoints and submit checkpoint paste jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


DRAW_RE = re.compile(r"chain_stage31_checkpoint_(\d{6})\.npz$")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker-dir", required=True)
    parser.add_argument("--combined-dir", required=True)
    parser.add_argument("--combined-suffix", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--expected-workers", type=int, default=4)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--combiner", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--make-getdist", action="store_true")
    parser.add_argument("--getdist-script", default=None)
    parser.add_argument("--getdist-python", default=None)
    parser.add_argument("--getdist-tag-prefix", default="stage31_hmc")
    parser.add_argument("--getdist-label", default=None)
    parser.add_argument("--paste-gate", default=None)
    parser.add_argument("--paste-config-template", default=None)
    parser.add_argument("--submit-paste", action="store_true")
    parser.add_argument("--paste-run-root-base", default=None)
    parser.add_argument("--nside", type=int, default=2048)
    parser.add_argument("--lmax", type=int, default=4096)
    parser.add_argument("--num-splits", type=int, default=4)
    parser.add_argument("--pixel-workers", type=int, default=16)
    parser.add_argument("--ksz-ylim-min", default="-5e-5")
    parser.add_argument("--ksz-ylim-max", default="5e-5")
    parser.add_argument("--plot-ell-max", default="2800")
    parser.add_argument("--plot-xscale", default="linear", choices=("linear", "log", "symlog"))
    parser.add_argument("--plot-xlim", default=None)
    parser.add_argument("--ksz-velocity-mode", default="photoz_reconstruction_emulation")
    parser.add_argument("--ksz-reconstruction-noise-seed", default="12345")
    parser.add_argument("--sim-matched-transfers", default="1")
    parser.add_argument("--do-direct-field", default="0")
    parser.add_argument("--do-plus-direct", default="0")
    parser.add_argument("--do-pasted-theory", default="1")
    parser.add_argument("--do-preprocess", default="1")
    parser.add_argument("--catalog-source", default=None)
    parser.add_argument("--catalog-output-name", default=None)
    parser.add_argument("--postprocess-platform", choices=("cpu", "gpu", "inherit"), default="cpu")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--require-healthy-for-paste", action=argparse.BooleanOptionalAction, default=True)
    return parser


def checkpoint_draws(worker_dir: Path, expected_workers: int) -> list[int]:
    worker0 = worker_dir / "worker_0"
    if not worker0.exists():
        return []
    draws = []
    for path in sorted(worker0.glob("chain_stage31_checkpoint_*.npz")):
        match = DRAW_RE.match(path.name)
        if not match:
            continue
        draw = int(match.group(1))
        present = all(
            (worker_dir / f"worker_{rank}" / f"chain_stage31_checkpoint_{draw:06d}.npz").is_file()
            for rank in range(expected_workers)
        )
        if present:
            draws.append(draw)
    return draws


def run_command(cmd: Sequence[str], *, log_path: Path, err_path: Path, env: Optional[dict[str, str]] = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as stdout, open(err_path, "w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, check=False, stdout=stdout, stderr=stderr, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def postprocess_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    if args.postprocess_platform == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_PLATFORM_NAME"] = "cpu"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif args.postprocess_platform == "gpu":
        env["JAX_PLATFORMS"] = "cuda"
    return env


def submit_paste(args: argparse.Namespace, checkpoint_dir: Path, suffix: str, draw: int, monitor_dir: Path) -> Optional[str]:
    if not args.submit_paste:
        return None
    if not args.paste_gate:
        raise ValueError("--submit-paste requires --paste-gate")
    if not args.paste_run_root_base:
        raise ValueError("--submit-paste requires --paste-run-root-base")

    tag = f"checkpoint_{draw:06d}"
    run_root = Path(args.paste_run_root_base).expanduser().resolve() / args.run_label / tag
    if args.catalog_source and args.catalog_output_name:
        source = Path(args.catalog_source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Missing reusable catalog source: {source}")
        catalog_dir = run_root / "halos"
        catalog_dir.mkdir(parents=True, exist_ok=True)
        target = catalog_dir / str(args.catalog_output_name)
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise FileExistsError(f"Checkpoint catalog target exists but does not point to source: {target}")
        else:
            os.symlink(source, target)
    runtime_config = run_root / "configs" / f"stage31_pz3_cap2400_hmcbestfit_mmin11p147538_nside{args.nside}_lmax{args.lmax}_{suffix}.selected.yaml"
    sim_meas = run_root / "measurements" / f"sim_pz3_cap2400_hmcbestfit_mmin11p147538_{tag}_nside{args.nside}_lmax{args.lmax}_nbin10_linear.h5"
    full_area_plot = run_root / "plots" / f"stage31_pz3_cap2400_hmcbestfit_mmin11p147538_{tag}_full_area_data_bestfit_with_cap_sim_nside{args.nside}_lmax{args.lmax}_Dell.pdf"
    sum_theory = run_root / "theory" / f"stage31_pz3_cap2400_hmcbestfit_mmin11p147538_{tag}_nside{args.nside}_lmax{args.lmax}_theory_poweradd_sum_for_sim_measurement_matched_transfers.h5"
    response_theory = run_root / "theory" / f"stage31_pz3_cap2400_hmcbestfit_mmin11p147538_{tag}_nside{args.nside}_lmax{args.lmax}_theory_response_for_sim_measurement_matched_transfers.h5"
    variant_plot = run_root / "plots" / f"stage31_pz3_cap2400_hmcbestfit_mmin11p147538_{tag}_pasted_only_full_data_theory_variants_with_cap_sim_Dell.pdf"
    paste_config_template = (
        str(Path(args.paste_config_template).expanduser().resolve())
        if args.paste_config_template
        else None
    )

    exports = {
        "HMC_COMBINED": str(checkpoint_dir),
        "COMBINED_SUFFIX": suffix,
        "FIT_SUMMARY": str(checkpoint_dir / f"fit_summary_{suffix}.json"),
        "BESTFIT_PARAMS": str(checkpoint_dir / f"bestfit_params_{suffix}.yaml"),
        "BESTFIT_VECTOR": str(checkpoint_dir / f"bestfit_full_theory_data_vector_{suffix}.npz"),
        "REQUIRE_CONVERGENCE": "0",
        "RUN_ROOT": str(run_root),
        "RUNTIME_CONFIG": str(runtime_config),
        "CHECKPOINT_TAG": tag,
        "CHECKPOINT_RUN_NAME": f"stage31_pz3_cap2400_hmcbestfit_mmin11p147538_{tag}_nside{args.nside}_lmax{args.lmax}",
        "CHECKPOINT_MEASUREMENT_TAG_BASE": f"pz3_cap2400_hmcbestfit_mmin11p147538_{tag}",
        "NSIDE": str(args.nside),
        "LMAX": str(args.lmax),
        "NUM_SPLITS": str(args.num_splits),
        "PIXEL_WORKERS": str(args.pixel_workers),
        "KSZ_VELOCITY_MODE": str(args.ksz_velocity_mode),
        "KSZ_RECONSTRUCTION_NOISE_SEED": str(args.ksz_reconstruction_noise_seed),
        "KSZ_YLIM_MIN": str(args.ksz_ylim_min),
        "KSZ_YLIM_MAX": str(args.ksz_ylim_max),
        "PLOT_ELL_MAX": str(args.plot_ell_max),
        "SIM_MATCHED_TRANSFERS": str(args.sim_matched_transfers),
        "DO_PREPROCESS": str(args.do_preprocess),
        "DO_PASTED_THEORY": str(args.do_pasted_theory),
        "DO_DIRECT_FIELD": str(args.do_direct_field),
        "DO_PLUS_DIRECT": str(args.do_plus_direct),
        "SIM_MEAS": str(sim_meas),
        "FULL_AREA_PLOT": str(full_area_plot),
        "SUM_THEORY": str(sum_theory),
        "RESPONSE_THEORY": str(response_theory),
        "VARIANT_PLOT": str(variant_plot),
    }
    if paste_config_template:
        exports["CONFIG"] = paste_config_template
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    cmd = ["sbatch", "--parsable", f"--export={export_arg}", str(Path(args.paste_gate).expanduser().resolve())]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    record = {
        "draws_per_worker": draw,
        "checkpoint_dir": str(checkpoint_dir),
        "suffix": suffix,
        "run_root": str(run_root),
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }
    with open(monitor_dir / "checkpoint_paste_submissions.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed for checkpoint {draw}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def make_getdist(args: argparse.Namespace, checkpoint_dir: Path, suffix: str, draw: int, monitor_dir: Path) -> Optional[dict]:
    if not args.make_getdist:
        return None
    if not args.getdist_script:
        raise ValueError("--make-getdist requires --getdist-script")
    chain_path = checkpoint_dir / f"chain_{suffix}.npz"
    if not chain_path.is_file():
        raise FileNotFoundError(f"Missing combined checkpoint chain for GetDist: {chain_path}")
    tag = f"{args.getdist_tag_prefix}_checkpoint_{draw:06d}"
    out_dir = checkpoint_dir / "getdist_gas_hod_ia"
    label_base = args.getdist_label or args.run_label
    sample_label = f"{label_base} checkpoint {draw:06d}"
    py = str(Path(args.getdist_python or args.python).expanduser().resolve())
    cmd = [
        py,
        "-u",
        str(Path(args.getdist_script).expanduser().resolve()),
        "--chain",
        str(chain_path),
        "--output-dir",
        str(out_dir),
        "--sample-label",
        sample_label,
        "--tag",
        tag,
    ]
    run_command(
        cmd,
        log_path=monitor_dir / f"getdist_checkpoint_{draw:06d}.out",
        err_path=monitor_dir / f"getdist_checkpoint_{draw:06d}.err",
        env=postprocess_env(args),
    )
    summary_path = out_dir / f"getdist_gas_hod_ia_sample_summary_{tag}.json"
    result = {
        "tag": tag,
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "all_selected_pdf": str(out_dir / f"getdist_all_selected_{tag}.pdf"),
    }
    if summary_path.is_file():
        result["summary"] = _json_load(summary_path)
    return result


def _json_load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def previous_checkpoint_health(monitor_dir: Path, draw: int) -> Optional[dict]:
    previous = []
    for path in monitor_dir.glob("checkpoint_*_health.json"):
        match = re.match(r"checkpoint_(\d{6})_health\.json$", path.name)
        if not match:
            continue
        other_draw = int(match.group(1))
        if other_draw < int(draw):
            previous.append((other_draw, path))
    if not previous:
        return None
    _, path = max(previous, key=lambda item: item[0])
    return _json_load(path)


def checkpoint_health(checkpoint_dir: Path, suffix: str, draw: int, monitor_dir: Path) -> dict:
    summary_path = checkpoint_dir / f"fit_summary_{suffix}.json"
    likelihood_vector_path = checkpoint_dir / f"bestfit_theory_data_vector_{suffix}.npz"
    full_vector_path = checkpoint_dir / f"bestfit_full_theory_data_vector_{suffix}.npz"
    dell_pdf = checkpoint_dir / f"posterior_predictive_dell_comparison_{suffix}.pdf"
    full_dell_pdf = checkpoint_dir / f"posterior_predictive_full_dell_comparison_{suffix}.pdf"
    params_path = checkpoint_dir / f"bestfit_params_{suffix}.yaml"

    health = {
        "draws_per_worker": int(draw),
        "suffix": str(suffix),
        "checkpoint_dir": str(checkpoint_dir),
        "summary_path": str(summary_path),
        "bestfit_params_path": str(params_path),
        "likelihood_vector_path": str(likelihood_vector_path),
        "full_vector_path": str(full_vector_path),
        "posterior_predictive_dell_pdf": str(dell_pdf),
        "posterior_predictive_full_dell_pdf": str(full_dell_pdf),
        "checks": {},
    }
    checks = health["checks"]
    for label, path in (
        ("summary_exists", summary_path),
        ("bestfit_params_exists", params_path),
        ("likelihood_vector_exists", likelihood_vector_path),
        ("full_vector_exists", full_vector_path),
        ("dell_pdf_exists", dell_pdf),
        ("full_dell_pdf_exists", full_dell_pdf),
    ):
        checks[label] = path.is_file()
    checks["dell_pdf_nonempty"] = dell_pdf.is_file() and dell_pdf.stat().st_size > 10_000
    checks["full_dell_pdf_nonempty"] = full_dell_pdf.is_file() and full_dell_pdf.stat().st_size > 10_000

    if summary_path.is_file():
        summary = _json_load(summary_path)
        health["best_whitened_chi2"] = float(summary.get("best_whitened_chi2", np.nan))
        health["best_sample_index"] = int(summary.get("best_sample_index", -1))
        health["n_samples_total"] = int(summary.get("n_samples_total", -1))
    else:
        summary = {}
        health["best_whitened_chi2"] = float("nan")
        health["best_sample_index"] = -1
        health["n_samples_total"] = -1

    vector_stats = {}
    for label, path in (("likelihood", likelihood_vector_path), ("full", full_vector_path)):
        stats = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            with np.load(path, allow_pickle=True) as data:
                data_vector = np.asarray(data["data_vector"], dtype=np.float64)
                theory_vector = np.asarray(data["theory_vector"], dtype=np.float64)
                ell = np.asarray(data["ell_band"], dtype=np.float64)
            finite_data = np.isfinite(data_vector)
            finite_theory = np.isfinite(theory_vector)
            same_shape = data_vector.shape == theory_vector.shape == ell.shape
            delta = theory_vector - data_vector if same_shape else np.asarray([], dtype=np.float64)
            stats.update(
                {
                    "same_shape": bool(same_shape),
                    "n_vector": int(theory_vector.size),
                    "n_finite_data": int(np.count_nonzero(finite_data)),
                    "n_finite_theory": int(np.count_nonzero(finite_theory)),
                    "all_finite_data": bool(np.all(finite_data)),
                    "all_finite_theory": bool(np.all(finite_theory)),
                    "max_abs_data": float(np.nanmax(np.abs(data_vector))) if data_vector.size else float("nan"),
                    "max_abs_theory": float(np.nanmax(np.abs(theory_vector))) if theory_vector.size else float("nan"),
                    "median_abs_delta": float(np.nanmedian(np.abs(delta))) if delta.size else float("nan"),
                    "max_abs_delta": float(np.nanmax(np.abs(delta))) if delta.size else float("nan"),
                }
            )
        vector_stats[label] = stats
    health["vector_stats"] = vector_stats
    checks["vectors_finite"] = all(
        stats.get("same_shape") and stats.get("all_finite_data") and stats.get("all_finite_theory")
        for stats in vector_stats.values()
    )

    previous = previous_checkpoint_health(monitor_dir, draw)
    if previous is None:
        health["previous_draws_per_worker"] = None
        health["previous_best_whitened_chi2"] = None
        health["best_chi2_nonincreasing"] = True
        health["best_chi2_improved"] = None
        health["delta_best_whitened_chi2"] = None
        health["residual_trend"] = None
    else:
        prev_chi2 = float(previous.get("best_whitened_chi2", np.nan))
        chi2 = float(health["best_whitened_chi2"])
        tol = 1.0e-8 * max(1.0, abs(prev_chi2))
        health["previous_draws_per_worker"] = int(previous.get("draws_per_worker", -1))
        health["previous_best_whitened_chi2"] = prev_chi2
        health["delta_best_whitened_chi2"] = chi2 - prev_chi2
        health["best_chi2_nonincreasing"] = bool(np.isfinite(chi2) and np.isfinite(prev_chi2) and chi2 <= prev_chi2 + tol)
        health["best_chi2_improved"] = bool(np.isfinite(chi2) and np.isfinite(prev_chi2) and chi2 < prev_chi2 - tol)
        residual_trend = {}
        for label in ("likelihood", "full"):
            current_median = float(vector_stats.get(label, {}).get("median_abs_delta", np.nan))
            prev_median = float(previous.get("vector_stats", {}).get(label, {}).get("median_abs_delta", np.nan))
            median_tol = 1.0e-8 * max(1.0, abs(prev_median))
            residual_trend[label] = {
                "previous_median_abs_delta": prev_median,
                "current_median_abs_delta": current_median,
                "delta_median_abs_delta": current_median - prev_median,
                "median_abs_delta_nonincreasing": bool(
                    np.isfinite(current_median)
                    and np.isfinite(prev_median)
                    and current_median <= prev_median + median_tol
                ),
            }
        health["residual_trend"] = residual_trend

    required_checks = [
        checks["summary_exists"],
        checks["bestfit_params_exists"],
        checks["likelihood_vector_exists"],
        checks["full_vector_exists"],
        checks["dell_pdf_exists"],
        checks["full_dell_pdf_exists"],
        checks["dell_pdf_nonempty"],
        checks["full_dell_pdf_nonempty"],
        checks["vectors_finite"],
        np.isfinite(float(health["best_whitened_chi2"])),
        bool(health["best_chi2_nonincreasing"]),
    ]
    health["healthy"] = bool(all(required_checks))
    health["health_note"] = (
        "Cumulative best chi2 is allowed to stay flat if no sample in this chunk beats the previous best; "
        "best_chi2_improved records strict improvement."
    )
    health_path = monitor_dir / f"checkpoint_{draw:06d}_health.json"
    health_path.write_text(json.dumps(health, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return health


def process_checkpoint(args: argparse.Namespace, draw: int, monitor_dir: Path) -> None:
    marker = monitor_dir / f"checkpoint_{draw:06d}.done"
    if marker.exists():
        return
    failed_marker = monitor_dir / f"checkpoint_{draw:06d}.failed"
    if failed_marker.exists() and not args.retry_failed:
        return
    if failed_marker.exists() and args.retry_failed:
        log(f"retrying previously failed checkpoint {draw:06d}")

    checkpoint_dir = Path(args.combined_dir).expanduser().resolve() / "checkpoints" / f"checkpoint_{draw:06d}"
    suffix = f"{args.combined_suffix}_checkpoint_{draw:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pattern = f"worker_*/chain_stage31_checkpoint_{draw:06d}.npz"
    log(f"combining checkpoint draws_per_worker={draw} suffix={suffix}")
    cmd = [
        str(Path(args.python).expanduser().resolve()),
        "-u",
        str(Path(args.combiner).expanduser().resolve()),
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--worker-dir",
        str(Path(args.worker_dir).expanduser().resolve()),
        "--pattern",
        pattern,
        "--output-dir",
        str(checkpoint_dir),
        "--suffix",
        suffix,
        "--plot-ell-max",
        str(args.plot_ell_max),
        f"--plot-ksz-ylim={args.ksz_ylim_min},{args.ksz_ylim_max}",
        "--plot-xscale",
        str(args.plot_xscale),
    ]
    if args.plot_xlim:
        cmd.extend(["--plot-xlim", str(args.plot_xlim)])
    try:
        run_command(
            cmd,
            log_path=monitor_dir / f"combine_checkpoint_{draw:06d}.out",
            err_path=monitor_dir / f"combine_checkpoint_{draw:06d}.err",
            env=postprocess_env(args),
        )
        health = checkpoint_health(checkpoint_dir, suffix, draw, monitor_dir)
        getdist = make_getdist(args, checkpoint_dir, suffix, draw, monitor_dir)
        if not health["healthy"] and args.require_healthy_for_paste:
            paste_job = None
            log(
                f"checkpoint {draw:06d} unhealthy; skipping paste submission "
                f"best_chi2={health.get('best_whitened_chi2')}"
            )
        else:
            paste_job = submit_paste(args, checkpoint_dir, suffix, draw, monitor_dir)
        payload = {
            "draws_per_worker": draw,
            "checkpoint_dir": str(checkpoint_dir),
            "suffix": suffix,
            "getdist": getdist,
            "paste_job": paste_job,
            "healthy": bool(health["healthy"]),
            "best_whitened_chi2": health.get("best_whitened_chi2"),
            "best_chi2_nonincreasing": health.get("best_chi2_nonincreasing"),
            "best_chi2_improved": health.get("best_chi2_improved"),
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if failed_marker.exists():
            failed_marker.unlink()
        log(f"checkpoint {draw:06d} complete paste_job={paste_job} getdist={bool(getdist)}")
    except Exception as exc:
        payload = {
            "draws_per_worker": draw,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        failed_marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log(f"checkpoint {draw:06d} failed: {exc}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    monitor_dir = Path(args.combined_dir).expanduser().resolve() / "checkpoints" / ".monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file).expanduser().resolve()
    log(
        "checkpoint monitor starting "
        f"worker_dir={args.worker_dir} combined_dir={args.combined_dir} "
        f"expected_workers={args.expected_workers} submit_paste={args.submit_paste} "
        f"make_getdist={args.make_getdist}"
    )
    while True:
        draws = checkpoint_draws(Path(args.worker_dir).expanduser().resolve(), int(args.expected_workers))
        for draw in draws:
            process_checkpoint(args, draw, monitor_dir)
        if stop_file.exists():
            draws = checkpoint_draws(Path(args.worker_dir).expanduser().resolve(), int(args.expected_workers))
            for draw in draws:
                process_checkpoint(args, draw, monitor_dir)
            log("checkpoint monitor stopping after final scan")
            return 0
        time.sleep(float(args.poll_interval))


if __name__ == "__main__":
    raise SystemExit(main())
