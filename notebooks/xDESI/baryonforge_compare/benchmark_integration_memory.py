#!/usr/bin/env python
"""Benchmark peak profile memory for native64, rejected trap256, and GL64."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (
    REPO_ROOT,
    canonical_json,
    load_config,
    resolve_path,
    sha256_file,
)


for path in (REPO_ROOT / "src", REPO_ROOT / "notebooks" / "xDESI"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


SCHEMA = "godmax_integration_memory_benchmark_v1"
MODES = ("native_trap64", "rejected_trap256", "matched_gl64")
LABELS = {
    "native_trap64": "native\ntrap64",
    "rejected_trap256": "rejected\ntrap256",
    "matched_gl64": "matched\nGL64",
}


def _vmrss_bytes() -> int:
    with Path("/proc/self/status").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status did not report VmRSS.")


def _stored_jax_array_bytes(obj: object) -> int:
    import jax

    total = 0
    seen: set[int] = set()
    for value in vars(obj).values():
        if isinstance(value, jax.Array) and id(value) not in seen:
            seen.add(id(value))
            total += int(value.nbytes)
    return total


def _prepare_inputs(config: Mapping[str, Any]):
    from abacus_pasting_helpers import prepare_godmax_config

    catalog_path = resolve_path(config["catalog"]["output_h5"], config["_config_path"])
    with h5py.File(catalog_path, "r") as handle:
        attrs = dict(handle.attrs)
        z_max = float(np.max(np.asarray(handle["z"])))
    return prepare_godmax_config(
        config,
        attrs,
        is_cmb_lensing=True,
        z_max=z_max,
        log10_mass_min=float(attrs["log10_m_min_hmsun"]),
    )


def _worker(config: Mapping[str, Any], mode: str) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from base_class import base_class, get_vmapped_func
    from get_radial_profiles import Profiles
    from matched_godmax_profiles import AsymptoticNormalizationProfiles

    class HistoricalAsymptoticTrapezoidProfiles(Profiles):
        """Reproduce the rejected 128R/global-trapezoid comparison path."""

        integration_rmax_r200c = 128.0

        def get_Mtot(self, jz, jM, rmax_r200c=None):
            return super().get_Mtot(jz, jM, rmax_r200c=128.0)

        def get_rho_gas_norm(self, jz, jM, rmax_r200c=None):
            return super().get_rho_gas_norm(jz, jM, rmax_r200c=128.0)

        def get_Ptot(self, jr, jz, jM, r_array_here=None, rmax_r200c=None):
            return super().get_Ptot(
                jr,
                jz,
                jM,
                r_array_here=r_array_here,
                rmax_r200c=128.0,
            )

    sim, halo, analysis, other = _prepare_inputs(config)
    analysis = copy.deepcopy(analysis)
    if mode == "native_trap64":
        analysis["num_points_trapz_int"] = 64
        profiles_class = Profiles
    elif mode == "rejected_trap256":
        analysis["num_points_trapz_int"] = 256
        profiles_class = HistoricalAsymptoticTrapezoidProfiles
    elif mode == "matched_gl64":
        profiles_class = AsymptoticNormalizationProfiles
    else:
        raise ValueError(f"Unknown benchmark mode {mode!r}.")

    base = base_class(sim, halo, analysis, other)
    profiles = profiles_class(
        sim,
        halo,
        analysis,
        other,
        base_class_obj=base,
    )
    digest = hashlib.sha256()
    forced_fields = (
        "Mtot_mat",
        "rho_gas_norm_mat",
        "Mdmb_mat",
        "Ptot_mat_physical",
        "Pe_mat_physical",
        "y3d_mat",
    )
    for name in forced_fields:
        array = np.asarray(jax.block_until_ready(getattr(profiles, name)))
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes(order="C"))

    def pressure_grid():
        return get_vmapped_func(profiles.get_Ptot, 3)(
            jnp.arange(profiles.nr),
            jnp.arange(profiles.nz),
            jnp.arange(profiles.nM),
        ).T

    compiled = jax.jit(pressure_grid).lower().compile()
    jax.block_until_ready(compiled())
    executable_memory = compiled.memory_analysis()
    temporary_bytes = int(executable_memory.temp_size_in_bytes)
    stored_bytes = _stored_jax_array_bytes(profiles)
    del compiled, base
    gc.collect()
    return {
        "mode": mode,
        "grid_shape_nr_nz_nM": [
            int(profiles.nr),
            int(profiles.nz),
            int(profiles.nM),
        ],
        "core_points": int(profiles.num_points_trapz_int),
        "extended_points": int(
            getattr(profiles, "extended_profile_num_points", profiles.num_points_trapz_int)
        ),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "retained_rss_bytes": _vmrss_bytes(),
        "stored_jax_array_bytes": stored_bytes,
        "ptot_executable_temporary_bytes": temporary_bytes,
        "result_sha256": digest.hexdigest(),
    }


def _run_worker(config_path: Path, mode: str, repeat: int) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update(
        {
            "JAX_ENABLE_X64": "True",
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "1",
        }
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--worker",
        mode,
        "--repeat-index",
        str(repeat),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    prefix = "BENCHMARK_JSON="
    records = [line for line in completed.stdout.splitlines() if line.startswith(prefix)]
    if len(records) != 1:
        raise RuntimeError(
            f"Worker {mode}/{repeat} emitted {len(records)} result records.\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return json.loads(records[0][len(prefix) :])


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_fields = (
        "peak_rss_bytes",
        "retained_rss_bytes",
        "stored_jax_array_bytes",
        "ptot_executable_temporary_bytes",
    )
    summary: dict[str, Any] = {"runs": runs}
    for field in scalar_fields:
        values = np.asarray([record[field] for record in runs], dtype=np.float64)
        summary[f"{field}_mean"] = float(np.mean(values))
        summary[f"{field}_min"] = int(np.min(values))
        summary[f"{field}_max"] = int(np.max(values))
    summary["repeat_hashes_identical"] = len(
        {record["result_sha256"] for record in runs}
    ) == 1
    return summary


def _plot(report: Mapping[str, Any], figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(MODES))
    colors = ("#4c78a8", "#e45756", "#59a14f")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, field, title in (
        (axes[0], "peak_rss_bytes", "Fresh-process peak RSS"),
        (
            axes[1],
            "ptot_executable_temporary_bytes",
            "Compiled full-grid HSE temporary",
        ),
    ):
        means = [report["results"][mode][f"{field}_mean"] / 2**30 for mode in MODES]
        axis.bar(positions, means, color=colors, width=0.68)
        for position, mode in zip(positions, MODES):
            values = [
                run[field] / 2**30 for run in report["results"][mode]["runs"]
            ]
            axis.scatter(
                np.full(len(values), position),
                values,
                color="black",
                s=18,
                zorder=3,
            )
        axis.set_xticks(positions, [LABELS[mode] for mode in MODES])
        axis.set_ylabel("GiB")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("GODMAX 128 x 48 x 48 integration-memory validation")
    fig.tight_layout()
    outputs = []
    for extension, kwargs in (("png", {"dpi": 180}), ("pdf", {})):
        output = figure_dir / f"integration_memory.{extension}"
        temporary = figure_dir / f".integration_memory.tmp.{os.getpid()}.{extension}"
        fig.savefig(temporary, bbox_inches="tight", **kwargs)
        os.replace(temporary, output)
        outputs.append(str(output))
    plt.close(fig)
    return outputs


def run(config_path: Path, output: Path, figure_dir: Path, repeats: int) -> dict[str, Any]:
    results = {
        mode: _summarize_runs(
            [_run_worker(config_path, mode, repeat) for repeat in range(repeats)]
        )
        for mode in MODES
    }
    native = results["native_trap64"]
    rejected = results["rejected_trap256"]
    matched = results["matched_gl64"]
    native_peak = native["peak_rss_bytes_mean"]
    rejected_peak = rejected["peak_rss_bytes_mean"]
    matched_peak = matched["peak_rss_bytes_mean"]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": (
            "CPU/x64 GODMAX Profiles construction and forced normalization, Mdmb, "
            "Ptot, Pe, and y3d evaluation; excludes setup_sim_map and map painting"
        ),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "benchmark_source_sha256": sha256_file(__file__),
        "python_executable": sys.executable,
        "repeats": repeats,
        "results": results,
        "comparisons": {
            "matched_peak_reduction_vs_rejected_fraction": (
                1.0 - matched_peak / rejected_peak
            ),
            "matched_peak_reduction_vs_native_fraction": (
                1.0 - matched_peak / native_peak
            ),
            "matched_stored_array_overhead_vs_native_bytes": int(
                round(
                    matched["stored_jax_array_bytes_mean"]
                    - native["stored_jax_array_bytes_mean"]
                )
            ),
        },
        "caveats": [
            "CPU TFRT allocator peaks do not determine absolute CUDA allocator peaks.",
            "nr=128 and the separate 128-point LOS projector are outside this integration-only benchmark.",
            "Retained RSS includes allocator/runtime noise; stored JAX-array bytes isolate persistent profile data.",
        ],
    }
    report["acceptance"] = {
        "matched_peak_not_above_native_by_more_than_5pct": (
            matched_peak <= 1.05 * native_peak
        ),
        "matched_peak_below_rejected_trap256": matched_peak < rejected_peak,
        "matched_rule_storage_overhead_exactly_1024_bytes": (
            report["comparisons"]["matched_stored_array_overhead_vs_native_bytes"]
            == 1024
        ),
        "all_repeat_hashes_identical": all(
            results[mode]["repeat_hashes_identical"] for mode in MODES
        ),
    }
    report["ok"] = all(report["acceptance"].values())
    report["figures"] = _plot(report, figure_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    report["output_json"] = str(output)
    report["output_json_sha256"] = sha256_file(output)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--figure-dir")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--worker", choices=MODES)
    parser.add_argument("--repeat-index", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    if args.worker:
        result = _worker(config, args.worker)
        result["repeat_index"] = args.repeat_index
        print(f"BENCHMARK_JSON={canonical_json(result)}")
        return 0
    if args.repeats < 1:
        raise ValueError("--repeats must be positive.")
    root = resolve_path(config["project"]["output_root"], config_path)
    output = Path(args.output).expanduser().resolve() if args.output else (
        root / "profiles" / "integration_memory_benchmark.json"
    )
    figure_dir = (
        Path(args.figure_dir).expanduser().resolve()
        if args.figure_dir
        else root / "profiles" / "figures"
    )
    report = run(config_path, output, figure_dir, args.repeats)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
