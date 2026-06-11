#!/usr/bin/env python
"""Split production driver for xDESI multi-probe NaMaster measurements."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import h5py
import numpy as np

from multiprobe_namaster import (
    SCHEMA_MAPS,
    SCHEMA_MEASUREMENT,
    SurveyBundle,
    MeasurementConfig,
    SpectrumSpec,
    _corr_from_cov,
    _covariance_workspace_from_fields,
    _json_dumps,
    _string_array,
    _write_dataset,
    add_common_cli_args,
    build_nmt_fields,
    build_probe_maps,
    compute_covariance_block_with_workspace,
    config_from_args,
    covariance_diagnostics,
    covariance_group_key_for_specs,
    covariance_input_noise_policy,
    default_spectrum_specs,
    load_map_product,
    make_bins,
    save_map_product,
    save_measurement_product,
    utc_now,
)


def _config_from_map_metadata(config: MeasurementConfig, map_metadata: Mapping[str, object]) -> MeasurementConfig:
    map_config = map_metadata.get("config", {})
    if isinstance(map_config, Mapping):
        for key in (
            "stage",
            "nside",
            "lmax",
            "ell_min",
            "n_bins",
            "binning",
            "act_downgrade",
            "shear_e_to_kappa_sign",
            "shear_mask_dataset",
            "shear_noise_attr",
        ):
            if key in map_config:
                setattr(config, key, map_config[key])
    config.validate()
    return config


def spectra_path(config: MeasurementConfig) -> Path:
    return config.output_root / f"xdesi_multiprobe_spectra_{config.product_tag}.h5"


def manifest_path(config: MeasurementConfig) -> Path:
    return config.output_root / f"covariance_manifest_{config.product_tag}.json"


def block_dir(config: MeasurementConfig) -> Path:
    return config.output_root / f"covariance_blocks_{config.product_tag}"


def block_shard_path(config: MeasurementConfig, group: Mapping[str, object]) -> Path:
    return block_dir(config) / f"cov_group_{int(group['index']):04d}_{str(group['class'])}.h5"


def _config_value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return bool(actual) == expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == int(expected)
        except Exception:
            return False
    if isinstance(expected, float):
        try:
            return float(actual) == float(expected)
        except Exception:
            return False
    return str(actual) == str(expected)


def _existing_product_matches_config(path: Path, schema: str, config: MeasurementConfig) -> Tuple[bool, str]:
    if not path.exists():
        return False, "file does not exist"
    try:
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("schema") != schema:
                return False, f"schema is {h5.attrs.get('schema')!r}, expected {schema!r}"
            if schema == SCHEMA_MAPS:
                metadata = json.loads(h5.attrs["metadata_json"])
                cfg = metadata.get("config", {})
            else:
                cfg = json.loads(h5.attrs["config_json"])
    except Exception as exc:
        return False, f"could not read product metadata: {exc}"
    for key in ("stage", "nside", "lmax", "ell_min", "n_bins", "binning", "act_downgrade"):
        expected = getattr(config, key)
        if key not in cfg:
            return False, f"missing config key {key!r}"
        if not _config_value_matches(cfg[key], expected):
            return False, f"config {key}={cfg[key]!r}, expected {expected!r}"
    return True, "compatible"


def _field_spin_from_name(name: str) -> int:
    return 2 if str(name).startswith("s") else 0


def build_covariance_manifest(config: MeasurementConfig) -> Dict[str, object]:
    specs = default_spectrum_specs()
    groups: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    for i, spec_i in enumerate(specs):
        for j, spec_j in enumerate(specs[i:], start=i):
            key = covariance_group_key_for_specs(spec_i, spec_j)
            if key not in groups:
                representative_fields = [spec_i.fields[0], spec_i.fields[1], spec_j.fields[0], spec_j.fields[1]]
                spins = [_field_spin_from_name(name) for name in representative_fields]
                groups[key] = {
                    "key": list(key),
                    "representative_fields": representative_fields,
                    "spins": spins,
                    "class": "scalar" if all(spin == 0 for spin in spins) else "spin2",
                    "blocks": [],
                }
            groups[key]["blocks"].append(
                {
                    "spec_i": spec_i.name,
                    "spec_j": spec_j.name,
                    "spec_i_index": i,
                    "spec_j_index": j,
                    "fields_i": list(spec_i.fields),
                    "fields_j": list(spec_j.fields),
                }
            )
    out_groups = []
    for index, group in enumerate(groups.values()):
        group = dict(group)
        group["index"] = index
        group["n_blocks"] = len(group["blocks"])
        out_groups.append(group)
    n_by_class = {
        "scalar": int(sum(1 for group in out_groups if group["class"] == "scalar")),
        "spin2": int(sum(1 for group in out_groups if group["class"] == "spin2")),
    }
    return {
        "created_utc": utc_now(),
        "stage": config.stage,
        "config": {
            "stage": config.stage,
            "nside": int(config.nside),
            "lmax": int(config.lmax),
            "ell_min": int(config.ell_min),
            "n_bins": int(config.n_bins),
            "binning": str(config.binning),
        },
        "n_spectra": len(specs),
        "n_covariance_blocks": len(specs) * (len(specs) + 1) // 2,
        "n_covariance_groups": len(out_groups),
        "n_covariance_groups_by_class": n_by_class,
        "spectrum_names": [spec.name for spec in specs],
        "groups": out_groups,
    }


def write_covariance_manifest(path: Path, config: MeasurementConfig, overwrite: bool = False) -> Dict[str, object]:
    manifest = build_covariance_manifest(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return json.loads(path.read_text())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, path)
    return manifest


def load_covariance_manifest(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing covariance manifest: {path}")
    return json.loads(path.read_text())


def _groups_for_class(manifest: Mapping[str, object], cov_class: str) -> List[Mapping[str, object]]:
    groups = list(manifest["groups"])
    if cov_class == "all":
        return groups
    return [group for group in groups if group["class"] == cov_class]


def _read_string_dataset(ds: h5py.Dataset) -> List[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in ds[:]]


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - 3600 * hours - 60 * minutes
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:04.1f}s"
    if minutes:
        return f"{minutes:d}m{secs:04.1f}s"
    return f"{secs:.1f}s"


def _proc_cpu_seconds(pid: int) -> float:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        close = text.rfind(")")
        if close < 0:
            return 0.0
        fields = text[close + 2 :].split()
        ticks = int(fields[11]) + int(fields[12])
        return float(ticks) / float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
    except Exception:
        return 0.0


def _proc_status(pid: int) -> Dict[str, object]:
    out: Dict[str, object] = {"state": "unknown", "threads": np.nan, "rss_gb": np.nan, "vms_gb": np.nan}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                out["state"] = line.split(":", 1)[1].strip()
            elif line.startswith("Threads:"):
                out["threads"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("VmRSS:"):
                out["rss_gb"] = float(line.split()[1]) / 1024.0**2
            elif line.startswith("VmSize:"):
                out["vms_gb"] = float(line.split()[1]) / 1024.0**2
    except Exception:
        pass
    return out


@contextmanager
def heartbeat(label: str, interval: float = 120.0, allocated_cpus: Optional[int] = None):
    """Print process CPU/memory status periodically while inside long C calls."""

    interval = float(interval)
    if interval <= 0.0:
        yield
        return
    pid = os.getpid()
    stop = threading.Event()
    start_wall = time.monotonic()
    start_cpu = _proc_cpu_seconds(pid)
    last_wall = start_wall
    last_cpu = start_cpu

    def run() -> None:
        nonlocal last_wall, last_cpu
        while not stop.wait(interval):
            now = time.monotonic()
            cpu = _proc_cpu_seconds(pid)
            recent_cores = (cpu - last_cpu) / max(now - last_wall, 1.0e-9)
            avg_cores = (cpu - start_cpu) / max(now - start_wall, 1.0e-9)
            status = _proc_status(pid)
            alloc = ""
            if allocated_cpus and allocated_cpus > 0:
                alloc = f", recent_alloc={100.0 * recent_cores / allocated_cpus:.1f}%"
            print(
                f"[{utc_now()}] heartbeat {label}: elapsed={_format_seconds(now - start_wall)}, "
                f"cpu_recent={recent_cores:.2f} cores, cpu_avg={avg_cores:.2f} cores{alloc}, "
                f"rss={float(status['rss_gb']):.2f} GB, vms={float(status['vms_gb']):.2f} GB, "
                f"threads={status['threads']}, state={status['state']}",
                flush=True,
            )
            last_wall = now
            last_cpu = cpu

    thread = threading.Thread(target=run, name=f"heartbeat:{label}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)


@contextmanager
def timed_step(label: str):
    start = time.monotonic()
    print(f"[{utc_now()}] START {label}", flush=True)
    try:
        yield
    finally:
        print(f"[{utc_now()}] DONE  {label} in {_format_seconds(time.monotonic() - start)}", flush=True)


def _read_spectra_product(path: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema") != SCHEMA_MEASUREMENT:
            raise ValueError(f"{path} is not a {SCHEMA_MEASUREMENT} product.")
        map_metadata = json.loads(h5.attrs["map_metadata_json"])
        config = json.loads(h5.attrs["config_json"])
        spectra: Dict[str, Dict[str, object]] = {}
        for name in h5["spectra"]:
            g = h5[f"spectra/{name}"]
            spectra[name] = {
                "name": name,
                "family": str(g.attrs["family"]),
                "label": str(g.attrs["label"]),
                "theory_key": str(g.attrs["theory_key"]),
                "component_label": str(g.attrs["component_label"]),
                "fields": tuple(json.loads(g.attrs["fields"])),
                "component": int(g.attrs["component"]),
                "component_labels": json.loads(g.attrs["component_labels"]),
                "metadata": json.loads(g.attrs["metadata_json"]),
                "ell": g["ell"][:],
                "cl": g["cl"][:],
                "cl_all_components": g["cl_all_components"][:],
                "pcl_all_components": g["pcl_all_components"][:],
                "bandpower_window_selected": g["bandpower_window_selected"][:],
                "noise_decoupled_all_components": (
                    None if "noise_decoupled_all_components" not in g else g["noise_decoupled_all_components"][:]
                ),
            }
        null_tests: Dict[str, Dict[str, object]] = {}
        if "null_tests" in h5:
            for name in h5["null_tests"]:
                g = h5[f"null_tests/{name}"]
                item = {"ell": g["ell"][:], "cl": g["cl"][:]}
                for key, value in g.attrs.items():
                    if isinstance(value, str):
                        try:
                            item[key] = json.loads(value)
                        except json.JSONDecodeError:
                            item[key] = value
                    else:
                        item[key] = value
                null_tests[name] = item
        field_metadata = {}
        if "fields" in h5 and "metadata_json" in h5["fields"].attrs:
            field_metadata = json.loads(h5["fields"].attrs["metadata_json"])
        result = {
            "schema": SCHEMA_MEASUREMENT,
            "created_utc": utc_now(),
            "config": config,
            "ell": h5["ell"][:],
            "ell_left": h5["ell_left"][:],
            "ell_right": h5["ell_right"][:],
            "binning": str(h5.attrs.get("binning", config.get("binning", "sqrt"))),
            "ell_max_inclusive": int(h5.attrs.get("ell_max_inclusive", config.get("lmax", 0))),
            "spectra": spectra,
            "covariance_blocks": {},
            "joint": None,
            "null_tests": null_tests,
            "input_cls_for_covariance": {},
            "workspace_keys": [],
            "covariance_workspace_keys": [],
            "field_metadata": field_metadata,
        }
    return result, map_metadata


def _write_input_cls_group(parent: h5py.Group, input_cl_cache: Mapping[Tuple[str, ...], np.ndarray]) -> None:
    group = parent.create_group("input_cls_for_covariance")
    for key, cl in input_cl_cache.items():
        if len(key) == 3:
            input_mode, a, b = key
            dataset_name = f"{input_mode}__{a}__x__{b}"
        else:
            input_mode = "legacy"
            a, b = key
            dataset_name = f"{a}__x__{b}"
        ds = _write_dataset(group, dataset_name, np.asarray(cl), dtype="f8")
        ds.attrs["input_mode"] = str(input_mode)
        ds.attrs["field_a"] = str(a)
        ds.attrs["field_b"] = str(b)


def _read_input_cls_group(parent: h5py.Group) -> Dict[Tuple[str, ...], np.ndarray]:
    out: Dict[Tuple[str, ...], np.ndarray] = {}
    if "input_cls_for_covariance" not in parent:
        return out
    for name in parent["input_cls_for_covariance"]:
        ds = parent[f"input_cls_for_covariance/{name}"]
        key = (str(ds.attrs["input_mode"]), str(ds.attrs["field_a"]), str(ds.attrs["field_b"]))
        out[key] = ds[:]
    return out


def run_prepare(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    bundle = SurveyBundle.from_root(args.survey_root)
    output = Path(args.maps_out).resolve() if args.maps_out else config.default_maps_path
    if output.exists() and not args.force:
        ok, reason = _existing_product_matches_config(output, SCHEMA_MAPS, config)
        if ok:
            print(f"[{utc_now()}] Reusing existing compatible map product: {output}", flush=True)
            return
        raise FileExistsError(f"{output} exists but is not compatible ({reason}); pass --force to replace it.")
    print(f"[{utc_now()}] Preparing maps for {config.stage}: {output}", flush=True)
    fields, metadata = build_probe_maps(bundle, config)
    save_map_product(output, fields, metadata, overwrite=args.force)
    print(f"[{utc_now()}] Wrote {output}", flush=True)


def run_spectra(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    maps = Path(args.maps_path).resolve() if args.maps_path else config.default_maps_path
    output = Path(args.spectra_out).resolve() if args.spectra_out else spectra_path(config)
    if output.exists() and not args.force:
        ok, reason = _existing_product_matches_config(output, SCHEMA_MEASUREMENT, config)
        if ok:
            print(f"[{utc_now()}] Reusing existing compatible spectra product: {output}", flush=True)
            return
        raise FileExistsError(f"{output} exists but is not compatible ({reason}); pass --force to replace it.")
    print(f"[{utc_now()}] Loading maps for spectra: {maps}", flush=True)
    map_fields, map_metadata = load_map_product(maps)
    config = _config_from_map_metadata(config, map_metadata)
    config.output_dir = args.output_dir
    config.compute_covariance = False
    from multiprobe_namaster import measure_all

    result = measure_all(map_fields, config, verbose=not args.quiet)
    save_measurement_product(output, result, map_metadata, overwrite=args.force)
    print(f"[{utc_now()}] Wrote spectra product {output}", flush=True)


def run_make_cov_manifest(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    output = Path(args.manifest_out).resolve() if args.manifest_out else manifest_path(config)
    manifest = write_covariance_manifest(output, config, overwrite=args.force)
    print(
        f"[{utc_now()}] Wrote covariance manifest {output} "
        f"({manifest['n_covariance_groups']} groups; {manifest['n_covariance_blocks']} blocks)",
        flush=True,
    )


def run_cov_key(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    manifest_file = Path(args.manifest_path).resolve() if args.manifest_path else manifest_path(config)
    manifest = load_covariance_manifest(manifest_file)
    groups = _groups_for_class(manifest, args.cov_class)
    task_id = int(args.task_id if args.task_id is not None else os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if task_id >= len(groups):
        print(f"[{utc_now()}] task_id={task_id} outside {args.cov_class} group count={len(groups)}; skipping.", flush=True)
        return
    group = groups[task_id]
    output = block_shard_path(config, group)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        print(f"[{utc_now()}] Existing covariance shard {output}; skipping.", flush=True)
        return

    maps = Path(args.maps_path).resolve() if args.maps_path else config.default_maps_path
    print(
        f"[{utc_now()}] Computing covariance group {group['index']} "
        f"({group['class']}, {group['n_blocks']} blocks) from {maps}",
        flush=True,
    )
    allocated_cpus = int(os.environ.get("OMP_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    with heartbeat(
        f"cov-key group={group['index']} class={group['class']}",
        interval=float(args.heartbeat_interval),
        allocated_cpus=allocated_cpus,
    ):
        with timed_step(f"group {group['index']} load map product"):
            map_fields, map_metadata = load_map_product(maps)
        config = _config_from_map_metadata(config, map_metadata)
        config.output_dir = args.output_dir
        with timed_step(f"group {group['index']} make bins"):
            bins = make_bins(config)
        with timed_step(f"group {group['index']} build NaMaster fields"):
            fields = build_nmt_fields(map_fields, config)
        representatives = list(group["representative_fields"])
        with timed_step(
            f"group {group['index']} build covariance workspace "
            f"({','.join(representatives)})"
        ):
            cw = _covariance_workspace_from_fields(
                fields[representatives[0]].cov_field,
                fields[representatives[1]].cov_field,
                fields[representatives[2]].cov_field,
                fields[representatives[3]].cov_field,
                config,
            )
        specs = {spec.name: spec for spec in default_spectrum_specs()}
        workspace_cache = {}
        input_cl_cache: Dict[Tuple[str, ...], np.ndarray] = {}
        blocks: Dict[Tuple[str, str], np.ndarray] = {}
        for block_info in group["blocks"]:
            spec_i = specs[str(block_info["spec_i"])]
            spec_j = specs[str(block_info["spec_j"])]
            with timed_step(f"group {group['index']} block {spec_i.name} x {spec_j.name}"):
                blocks[(spec_i.name, spec_j.name)] = compute_covariance_block_with_workspace(
                    spec_i,
                    spec_j,
                    fields,
                    bins,
                    workspace_cache,
                    cw,
                    input_cl_cache,
                    config,
                )

    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w", track_order=True) as h5:
        h5.attrs["created_utc"] = utc_now()
        h5.attrs["stage"] = config.stage
        h5.attrs["config_json"] = _json_dumps(config.__dict__)
        h5.attrs["group_json"] = json.dumps(group)
        h5.attrs["group_index"] = int(group["index"])
        h5.attrs["group_class"] = str(group["class"])
        bg = h5.create_group("covariance_blocks")
        for (name_i, name_j), block in blocks.items():
            ds = _write_dataset(bg, f"{name_i}__x__{name_j}", block, dtype="f8")
            ds.attrs["spectrum_i"] = name_i
            ds.attrs["spectrum_j"] = name_j
        _write_input_cls_group(h5, input_cl_cache)
    os.replace(tmp, output)
    print(f"[{utc_now()}] Wrote covariance shard {output}", flush=True)


def _compute_covariance_group(
    group: Mapping[str, object],
    fields: Mapping[str, object],
    bins: object,
    config: MeasurementConfig,
    *,
    force: bool,
) -> Path:
    output = block_shard_path(config, group)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        print(f"[{utc_now()}] Existing covariance shard {output}; skipping.", flush=True)
        return output
    representatives = list(group["representative_fields"])
    with timed_step(
        f"group {group['index']} build covariance workspace "
        f"({','.join(representatives)})"
    ):
        cw = _covariance_workspace_from_fields(
            fields[representatives[0]].cov_field,
            fields[representatives[1]].cov_field,
            fields[representatives[2]].cov_field,
            fields[representatives[3]].cov_field,
            config,
        )
    specs = {spec.name: spec for spec in default_spectrum_specs()}
    workspace_cache = {}
    input_cl_cache: Dict[Tuple[str, ...], np.ndarray] = {}
    blocks: Dict[Tuple[str, str], np.ndarray] = {}
    for block_info in group["blocks"]:
        spec_i = specs[str(block_info["spec_i"])]
        spec_j = specs[str(block_info["spec_j"])]
        with timed_step(f"group {group['index']} block {spec_i.name} x {spec_j.name}"):
            blocks[(spec_i.name, spec_j.name)] = compute_covariance_block_with_workspace(
                spec_i,
                spec_j,
                fields,
                bins,
                workspace_cache,
                cw,
                input_cl_cache,
                config,
            )
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w", track_order=True) as h5:
        h5.attrs["created_utc"] = utc_now()
        h5.attrs["stage"] = config.stage
        h5.attrs["config_json"] = _json_dumps(config.__dict__)
        h5.attrs["group_json"] = json.dumps(group)
        h5.attrs["group_index"] = int(group["index"])
        h5.attrs["group_class"] = str(group["class"])
        bg = h5.create_group("covariance_blocks")
        for (name_i, name_j), block in blocks.items():
            ds = _write_dataset(bg, f"{name_i}__x__{name_j}", block, dtype="f8")
            ds.attrs["spectrum_i"] = name_i
            ds.attrs["spectrum_j"] = name_j
        _write_input_cls_group(h5, input_cl_cache)
    os.replace(tmp, output)
    print(f"[{utc_now()}] Wrote covariance shard {output}", flush=True)
    return output


def run_cov_batch(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    manifest_file = Path(args.manifest_path).resolve() if args.manifest_path else manifest_path(config)
    manifest = load_covariance_manifest(manifest_file)
    groups = _groups_for_class(manifest, args.cov_class)
    batch_id = int(args.batch_id if args.batch_id is not None else os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    start = batch_id * batch_size
    stop = min(start + batch_size, len(groups))
    if start >= len(groups):
        print(f"[{utc_now()}] batch_id={batch_id} outside {args.cov_class} group count={len(groups)}; skipping.", flush=True)
        return
    selected_groups = groups[start:stop]
    parallel_groups = int(getattr(args, "parallel_groups", 1))
    if parallel_groups > 1 and len(selected_groups) > 1:
        env_base = os.environ.copy()
        omp_threads = max(1, int(getattr(args, "omp_threads_per_group", 1)))
        env_base["OMP_NUM_THREADS"] = str(omp_threads)
        env_base["OMP_PROC_BIND"] = "spread"
        env_base["OMP_PLACES"] = "cores"
        env_base["OMP_MAX_ACTIVE_LEVELS"] = "1"
        env_base["MKL_NUM_THREADS"] = "1"
        env_base["OPENBLAS_NUM_THREADS"] = "1"
        env_base["NUMEXPR_NUM_THREADS"] = "1"
        common = [
            sys.executable,
            str(Path(__file__).resolve()),
            "cov-key",
            "--stage",
            str(config.stage),
            "--output-dir",
            str(args.output_dir),
            "--cov-class",
            str(args.cov_class),
            "--manifest-path",
            str(manifest_file),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
        ]
        if args.maps_path:
            common.extend(["--maps-path", str(args.maps_path)])
        if args.force:
            common.append("--force")
        running = []
        for group in selected_groups:
            task_id = groups.index(group)
            cmd = [*common, "--task-id", str(task_id)]
            print(
                f"[{utc_now()}] Launching group {group['index']} as subprocess "
                f"(task_id={task_id}, OMP_NUM_THREADS={omp_threads})",
                flush=True,
            )
            running.append(subprocess.Popen(cmd, env=env_base))
            while len(running) >= parallel_groups:
                proc = running.pop(0)
                ret = proc.wait()
                if ret != 0:
                    raise subprocess.CalledProcessError(ret, proc.args)
        for proc in running:
            ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, proc.args)
        return

    maps = Path(args.maps_path).resolve() if args.maps_path else config.default_maps_path
    print(
        f"[{utc_now()}] Computing covariance batch {batch_id} "
        f"({args.cov_class} groups {start}..{stop - 1}) from {maps}",
        flush=True,
    )
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "1")))
    with heartbeat(
        f"cov-batch id={batch_id} class={args.cov_class} groups={start}..{stop - 1}",
        interval=float(args.heartbeat_interval),
        allocated_cpus=allocated_cpus,
    ):
        with timed_step(f"batch {batch_id} load map product"):
            map_fields, map_metadata = load_map_product(maps)
        config = _config_from_map_metadata(config, map_metadata)
        config.output_dir = args.output_dir
        with timed_step(f"batch {batch_id} make bins"):
            bins = make_bins(config)
        with timed_step(f"batch {batch_id} build NaMaster fields"):
            fields = build_nmt_fields(map_fields, config)
        for group in selected_groups:
            print(
                f"[{utc_now()}] Group {group['index']} ({group['class']}, {group['n_blocks']} blocks)",
                flush=True,
            )
            _compute_covariance_group(group, fields, bins, config, force=args.force)


def run_assemble(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    spec_file = Path(args.spectra_path).resolve() if args.spectra_path else spectra_path(config)
    manifest_file = Path(args.manifest_path).resolve() if args.manifest_path else manifest_path(config)
    output = Path(args.measurement_out).resolve() if args.measurement_out else config.default_measurement_path
    result, map_metadata = _read_spectra_product(spec_file)
    manifest = load_covariance_manifest(manifest_file)
    specs = default_spectrum_specs()
    ell = np.asarray(result["ell"], dtype=np.float64)
    n_per = ell.size
    n_data = n_per * len(specs)
    cov = np.zeros((n_data, n_data), dtype=np.float64)
    slices = {spec.name: (i * n_per, (i + 1) * n_per) for i, spec in enumerate(specs)}
    covariance_blocks: Dict[Tuple[str, str], np.ndarray] = {}
    input_cls: Dict[Tuple[str, ...], np.ndarray] = {}

    for group in manifest["groups"]:
        shard = block_shard_path(config, group)
        if not shard.exists():
            raise FileNotFoundError(f"Missing covariance shard for group {group['index']}: {shard}")
        with h5py.File(shard, "r") as h5:
            input_cls.update(_read_input_cls_group(h5))
            for name in h5["covariance_blocks"]:
                ds = h5[f"covariance_blocks/{name}"]
                name_i = str(ds.attrs["spectrum_i"])
                name_j = str(ds.attrs["spectrum_j"])
                block = ds[:]
                covariance_blocks[(name_i, name_j)] = block
                si = slice(*slices[name_i])
                sj = slice(*slices[name_j])
                cov[si, sj] = block
                if name_i != name_j:
                    cov[sj, si] = block.T

    for spec in specs:
        name = spec.name
        start, stop = slices[name]
        block = cov[start:stop, start:stop]
        result["spectra"][name]["cov"] = block
        result["spectra"][name]["err"] = np.sqrt(np.clip(np.diag(block), 0.0, np.inf))

    data_vector = np.concatenate([np.asarray(result["spectra"][spec.name]["cl"]) for spec in specs])
    result["covariance_blocks"] = covariance_blocks
    result["input_cls_for_covariance"] = input_cls
    result["covariance_workspace_keys"] = [group["key"] for group in manifest["groups"]]
    result["joint"] = {
        "spectrum_names": [spec.name for spec in specs],
        "ell": ell,
        "data_vector": data_vector,
        "cov": cov,
        "corr": _corr_from_cov(cov),
        "slices": slices,
        "diagnostics": covariance_diagnostics(cov, compute_eig=not args.skip_cov_eig),
    }
    save_measurement_product(output, result, map_metadata, overwrite=args.force)
    print(f"[{utc_now()}] Wrote assembled measurement {output}", flush=True)


def run_validate(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    path = Path(args.measurement_path).resolve() if args.measurement_path else config.default_measurement_path
    with h5py.File(path, "r") as h5:
        cov = h5["joint/cov"][:]
        names = _read_string_dataset(h5["joint/spectrum_names"])
    expected = 46 * int(config.n_bins)
    if cov.shape != (expected, expected):
        raise ValueError(f"Covariance shape {cov.shape} does not match expected {(expected, expected)}.")
    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance contains non-finite values.")
    if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-20):
        raise ValueError("Covariance is not symmetric.")
    diag = np.diag(cov)
    if np.any(diag <= 0) or not np.all(np.isfinite(diag)):
        raise ValueError("Covariance diagonal is not strictly positive and finite.")
    if len(names) != 46:
        raise ValueError(f"Expected 46 spectra, found {len(names)}.")
    print(
        f"[{utc_now()}] Validation passed for {path}: shape={cov.shape}, "
        f"diag=[{diag.min():.3e}, {diag.max():.3e}]",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        add_common_cli_args(subparser)

    p = sub.add_parser("prepare")
    add_common(p)
    p.add_argument("--maps-out", default=None)
    p.set_defaults(func=run_prepare)

    p = sub.add_parser("spectra")
    add_common(p)
    p.add_argument("--maps-path", default=None)
    p.add_argument("--spectra-out", default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=run_spectra)

    p = sub.add_parser("make-cov-manifest")
    add_common(p)
    p.add_argument("--manifest-out", default=None)
    p.set_defaults(func=run_make_cov_manifest)

    p = sub.add_parser("cov-key")
    add_common(p)
    p.add_argument("--maps-path", default=None)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--cov-class", choices=["all", "scalar", "spin2"], default="all")
    p.add_argument("--heartbeat-interval", type=float, default=120.0)
    p.set_defaults(func=run_cov_key)

    p = sub.add_parser("cov-batch")
    add_common(p)
    p.add_argument("--maps-path", default=None)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--batch-id", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--parallel-groups", type=int, default=1)
    p.add_argument("--omp-threads-per-group", type=int, default=1)
    p.add_argument("--cov-class", choices=["all", "scalar", "spin2"], default="all")
    p.add_argument("--heartbeat-interval", type=float, default=120.0)
    p.set_defaults(func=run_cov_batch)

    p = sub.add_parser("assemble")
    add_common(p)
    p.add_argument("--spectra-path", default=None)
    p.add_argument("--manifest-path", default=None)
    p.add_argument("--measurement-out", default=None)
    p.add_argument("--skip-cov-eig", action="store_true")
    p.set_defaults(func=run_assemble)

    p = sub.add_parser("validate")
    add_common(p)
    p.add_argument("--measurement-path", default=None)
    p.set_defaults(func=run_validate)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
