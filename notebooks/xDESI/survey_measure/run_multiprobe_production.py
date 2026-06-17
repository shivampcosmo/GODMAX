#!/usr/bin/env python
"""Split production driver for xDESI multi-probe NaMaster measurements."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Set, Tuple

import h5py
import numpy as np
import pymaster as nmt

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
            "act_downgrade",
            "shear_e_to_kappa_sign",
            "shear_mask_dataset",
            "shear_noise_attr",
            "mask_apodization_deg",
            "mask_apodization_type",
        ):
            if key in map_config:
                setattr(config, key, map_config[key])
        if "lmax" in map_config and int(config.lmax) > int(map_config["lmax"]):
            raise ValueError(f"Requested lmax={config.lmax} exceeds cached-map lmax={map_config['lmax']}.")
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


def cov_workspace_cache_dir(config: MeasurementConfig) -> Path:
    return block_dir(config) / "cov_workspaces"


def _field_names_for_groups(groups: Iterable[Mapping[str, object]]) -> Set[str]:
    """Return only the field names a set of covariance groups actually reference.

    A group's covariance only needs the fields that appear in its blocks (plus the
    representative fields used for the covariance workspace). Building just these instead
    of all ~15 probe fields cuts per-process memory by ~3-5x (the spin-2 alms dominate),
    which lets many more single-threaded groups pack onto one node.
    """

    names: Set[str] = set()
    for group in groups:
        names.update(str(n) for n in group.get("representative_fields", []))
        for block in group.get("blocks", []):
            names.update(str(n) for n in block.get("fields_i", []))
            names.update(str(n) for n in block.get("fields_j", []))
    return names


def _build_cov_fields(
    map_fields: Mapping[str, object],
    config: MeasurementConfig,
    groups: Iterable[Mapping[str, object]],
) -> Dict[str, object]:
    """Build NaMaster fields for only the probes referenced by ``groups``."""

    needed = _field_names_for_groups(groups)
    subset = {name: fmap for name, fmap in map_fields.items() if name in needed}
    missing = sorted(needed - set(subset))
    if missing:
        raise KeyError(f"Covariance group references field(s) absent from the map product: {missing}")
    return build_nmt_fields(subset, config)


def _cov_workspace_cache_path(config: MeasurementConfig, group: Mapping[str, object]) -> Path:
    """Path of the on-disk covariance workspace for a group's mask/spin signature.

    The covariance workspace depends only on the four masks (the alias key), the spins,
    and lmax/Toeplitz settings -- never on the field data or the noise model. So it is
    valid across any rerun that keeps the same map product (masks) and binning, e.g. a
    shape-noise/data-vector change. It is scoped under the tag-specific block dir, so a
    different product (different nside/apodization) gets a separate cache.
    """

    signature = {
        "key": [str(k) for k in group.get("key", [])],
        "spins": [int(s) for s in group.get("spins", [])],
        "lmax": int(config.lmax),
        "l_toeplitz": int(config.covariance_l_toeplitz),
        "l_exact": int(config.covariance_l_exact),
        "dl_band": int(config.covariance_dl_band),
    }
    digest = hashlib.md5(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    return cov_workspace_cache_dir(config) / f"cw_{digest}.fits"


def _get_or_build_cov_workspace(
    group: Mapping[str, object],
    fields: Mapping[str, object],
    config: MeasurementConfig,
    *,
    use_cache: bool = True,
) -> object:
    """Load the group's covariance workspace from disk if cached, else build and cache it."""

    representatives = list(group["representative_fields"])
    path = _cov_workspace_cache_path(config, group)
    if use_cache and path.exists():
        try:
            cw = nmt.NmtCovarianceWorkspace.from_file(str(path))
            print(f"[{utc_now()}] group {group['index']} reused cached covariance workspace {path.name}", flush=True)
            return cw
        except Exception as exc:  # pragma: no cover - corrupt/old cache, rebuild
            print(f"[{utc_now()}] group {group['index']} cached workspace {path.name} unreadable ({exc}); rebuilding", flush=True)
    cw = _covariance_workspace_from_fields(
        fields[representatives[0]].cov_field,
        fields[representatives[1]].cov_field,
        fields[representatives[2]].cov_field,
        fields[representatives[3]].cov_field,
        config,
    )
    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".fits.tmp")
            if tmp.exists():
                tmp.unlink()
            cw.write_to(str(tmp))
            os.replace(tmp, path)
        except Exception as exc:  # pragma: no cover - cache write best effort
            print(f"[{utc_now()}] group {group['index']} could not cache workspace ({exc})", flush=True)
    return cw


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
    for key in (
        "stage",
        "nside",
        "lmax",
        "ell_min",
        "n_bins",
        "binning",
        "act_downgrade",
        "mask_apodization_deg",
        "mask_apodization_type",
        "pair_overlap_mean_subtract",
    ):
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
            "mask_apodization_deg": float(config.mask_apodization_deg),
            "mask_apodization_type": str(config.mask_apodization_type),
            "pair_overlap_mean_subtract": bool(config.pair_overlap_mean_subtract),
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
            if "pair_overlap_mean_subtraction_json" in g.attrs:
                spectra[name]["pair_overlap_mean_subtraction"] = json.loads(g.attrs["pair_overlap_mean_subtraction_json"])
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


SHEAR_AUTO_SPEC_NAMES = tuple(f"des_shear_EE_tomo{i}_tomo{i}" for i in range(1, 5))
_PATCHABLE_SPECTRUM_DATASETS = (
    "cl",
    "cl_all_components",
    "pcl_all_components",
    "noise_decoupled_all_components",
)


def _patch_shear_autos_in_spectra_product(
    output: Path,
    maps: Path,
    config: MeasurementConfig,
    verbose: bool = True,
) -> int:
    """Recompute only the 4 same-bin DES shear EE autos and overwrite them in place.

    The shape-noise fix changes ONLY the shear-auto spectra; the other 42 (cross-tomo
    shear, galaxy autos/crosses, kSZ, ...) have an untouched noise path and are bit-identical.
    So when a compatible spectra product already exists, we recompute just the 4 autos
    (loading only the shear fields and building only their workspaces) and replace their
    datasets in the HDF5, instead of regenerating all 46 (~2.9 h -> ~25 min). assemble
    rebuilds the data vector from the per-spectrum ``cl``, so no ``joint`` group exists to fix.
    """

    from multiprobe_namaster import build_nmt_fields, default_spectrum_specs, make_bins, measure_spectrum

    shear_map_fields, _ = load_map_product(maps, field_names={f"s{i}" for i in range(1, 5)})
    fields = build_nmt_fields(shear_map_fields, config)
    bins = make_bins(config)
    specs = [spec for spec in default_spectrum_specs() if spec.name in SHEAR_AUTO_SPEC_NAMES]
    workspace_cache: Dict[Tuple[str, str], object] = {}
    with h5py.File(output, "r+") as h5:
        for spec in specs:
            if verbose:
                print(f"[{utc_now()}] Patching shear auto {spec.name}", flush=True)
            res = measure_spectrum(spec, fields, bins, workspace_cache, config)
            grp = h5[f"spectra/{spec.name}"]
            for key in _PATCHABLE_SPECTRUM_DATASETS:
                value = np.asarray(res[key], dtype=np.float64)
                if key in grp:
                    del grp[key]
                grp.create_dataset(key, data=value)
    print(f"[{utc_now()}] Patched {len(specs)} shear-auto spectra in {output}", flush=True)
    return len(specs)


def run_spectra(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    maps = Path(args.maps_path).resolve() if args.maps_path else config.default_maps_path

    if getattr(args, "patch_shear_only", False):
        # Resolve the output tag from the map metadata without loading any field maps.
        _, meta_for_tag = load_map_product(maps, field_names=set())
        cfg = _config_from_map_metadata(config, meta_for_tag)
        cfg.output_dir = args.output_dir
        out = Path(args.spectra_out).resolve() if args.spectra_out else spectra_path(cfg)
        if out.exists():
            ok, reason = _existing_product_matches_config(out, SCHEMA_MEASUREMENT, cfg)
            if not ok:
                raise FileExistsError(
                    f"--patch-shear-only: existing {out} is incompatible ({reason}); "
                    "rerun without --patch-shear-only (optionally with --force) to regenerate fully."
                )
            cfg.compute_covariance = False
            print(f"[{utc_now()}] Patch mode: recomputing only shear autos in {out}", flush=True)
            _patch_shear_autos_in_spectra_product(out, maps, cfg, verbose=not args.quiet)
            return
        print(
            f"[{utc_now()}] --patch-shear-only requested but {out} is absent; "
            "doing a full spectra recompute instead.",
            flush=True,
        )

    print(f"[{utc_now()}] Loading maps for spectra: {maps}", flush=True)
    map_fields, map_metadata = load_map_product(maps)
    config = _config_from_map_metadata(config, map_metadata)
    config.output_dir = args.output_dir
    output = Path(args.spectra_out).resolve() if args.spectra_out else spectra_path(config)
    if output.exists() and not args.force:
        ok, reason = _existing_product_matches_config(output, SCHEMA_MEASUREMENT, config)
        if ok:
            print(f"[{utc_now()}] Reusing existing compatible spectra product: {output}", flush=True)
            return
        raise FileExistsError(f"{output} exists but is not compatible ({reason}); pass --force to replace it.")
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
            map_fields, map_metadata = load_map_product(maps, field_names=_field_names_for_groups([group]))
        config = _config_from_map_metadata(config, map_metadata)
        config.output_dir = args.output_dir
        with timed_step(f"group {group['index']} make bins"):
            bins = make_bins(config)
        with timed_step(f"group {group['index']} build NaMaster fields"):
            fields = _build_cov_fields(map_fields, config, [group])
        with timed_step(
            f"group {group['index']} get/build covariance workspace "
            f"({','.join(group['representative_fields'])})"
        ):
            cw = _get_or_build_cov_workspace(
                group, fields, config, use_cache=not getattr(args, "no_cov_workspace_cache", False)
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
    use_cache: bool = True,
) -> Path:
    output = block_shard_path(config, group)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        print(f"[{utc_now()}] Existing covariance shard {output}; skipping.", flush=True)
        return output
    with timed_step(
        f"group {group['index']} get/build covariance workspace "
        f"({','.join(group['representative_fields'])})"
    ):
        cw = _get_or_build_cov_workspace(group, fields, config, use_cache=use_cache)
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
        if getattr(args, "no_cov_workspace_cache", False):
            common.append("--no-cov-workspace-cache")
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
            map_fields, map_metadata = load_map_product(maps, field_names=_field_names_for_groups(selected_groups))
        config = _config_from_map_metadata(config, map_metadata)
        config.output_dir = args.output_dir
        with timed_step(f"batch {batch_id} make bins"):
            bins = make_bins(config)
        with timed_step(f"batch {batch_id} build NaMaster fields"):
            fields = _build_cov_fields(map_fields, config, selected_groups)
        use_cache = not getattr(args, "no_cov_workspace_cache", False)
        for group in selected_groups:
            print(
                f"[{utc_now()}] Group {group['index']} ({group['class']}, {group['n_blocks']} blocks)",
                flush=True,
            )
            _compute_covariance_group(group, fields, bins, config, force=args.force, use_cache=use_cache)


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
        data = h5["joint/data_vector"][:]
        names = _read_string_dataset(h5["joint/spectrum_names"])
        diagnostics = json.loads(str(h5["joint"].attrs.get("diagnostics_json", "{}")))
    expected = 46 * int(config.n_bins)
    if cov.shape != (expected, expected):
        raise ValueError(f"Covariance shape {cov.shape} does not match expected {(expected, expected)}.")
    if data.shape != (expected,):
        raise ValueError(f"Data vector shape {data.shape} does not match expected {(expected,)}.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Data vector contains non-finite values.")
    if not np.all(np.isfinite(cov)):
        raise ValueError("Covariance contains non-finite values.")
    if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-20):
        raise ValueError("Covariance is not symmetric.")
    diag = np.diag(cov)
    if np.any(diag <= 0) or not np.all(np.isfinite(diag)):
        raise ValueError("Covariance diagonal is not strictly positive and finite.")
    if len(names) != 46:
        raise ValueError(f"Expected 46 spectra, found {len(names)}.")
    sigma = np.sqrt(diag)
    corr = cov / np.outer(sigma, sigma)
    corr = 0.5 * (corr + corr.T)
    corr_eig = np.linalg.eigvalsh(corr)
    cov_eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    if not np.all(np.isfinite(corr_eig)) or not np.all(np.isfinite(cov_eig)):
        raise ValueError("Covariance/correlation eigenvalues contain non-finite values.")
    corr_threshold = float(args.corr_eigen_threshold)
    rank = int(np.sum(corr_eig > corr_threshold))
    if rank <= 0:
        raise ValueError(f"Correlation eigencut threshold {corr_threshold:g} retains zero modes.")
    if float(np.min(corr_eig)) < -1.0e-6:
        raise ValueError(f"Correlation matrix has a strongly negative eigenvalue: {np.min(corr_eig):.6e}.")
    report = {
        "measurement_path": str(path),
        "n_spectra": len(names),
        "n_bins": int(config.n_bins),
        "data_vector_size": int(data.size),
        "covariance_shape": list(cov.shape),
        "data_finite": bool(np.all(np.isfinite(data))),
        "covariance_finite": bool(np.all(np.isfinite(cov))),
        "covariance_symmetric": bool(np.allclose(cov, cov.T, rtol=1e-8, atol=1e-20)),
        "diag_min": float(np.min(diag)),
        "diag_max": float(np.max(diag)),
        "cov_eigen_min": float(np.min(cov_eig)),
        "cov_eigen_max": float(np.max(cov_eig)),
        "corr_eigen_min": float(np.min(corr_eig)),
        "corr_eigen_max": float(np.max(corr_eig)),
        "corr_eigen_threshold": corr_threshold,
        "corr_eigencut_rank": rank,
        "corr_eigencut_dropped_modes": int(corr_eig.size - rank),
        "hdf5_diagnostics": diagnostics,
    }
    report_path = path.with_name(f"measurement_validation_{config.product_tag}.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[{utc_now()}] Validation passed for {path}: shape={cov.shape}, "
        f"diag=[{diag.min():.3e}, {diag.max():.3e}], "
        f"corr_eig=[{corr_eig.min():.3e}, {corr_eig.max():.3e}], "
        f"rank@{corr_threshold:g}={rank}/{corr_eig.size}, report={report_path}",
        flush=True,
    )


def _parse_ksz_ylim(value: object) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    parts = str(value).replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError(f"Expected two values for --plot-ksz-ylim, got {value!r}.")
    return float(parts[0]), float(parts[1])


def run_plot_measurement_dell(args: argparse.Namespace) -> None:
    config = config_from_args(args)
    path = Path(args.measurement_path).resolve() if args.measurement_path else config.default_measurement_path
    output_dir = Path(args.plot_dir).resolve() if args.plot_dir else config.output_root / "plots"
    pdf = Path(args.pdf_out).resolve() if args.pdf_out else output_dir / f"measurement_dell_{config.product_tag}.pdf"
    ell_max = None if args.plot_ell_max is not None and float(args.plot_ell_max) <= 0.0 else args.plot_ell_max
    ksz_ylim = _parse_ksz_ylim(args.plot_ksz_ylim)

    import godmax_multiprobe_theory_utils as gmt

    measurement = gmt.load_measurement_data(path)
    outputs = gmt.plot_measurement_dell(
        measurement,
        output_dir,
        pdf_path=pdf,
        filename_prefix=f"measurement_dell_{config.product_tag}",
        ell_max=ell_max,
        ksz_ylim=ksz_ylim,
        ksz_scale=float(args.plot_ksz_scale),
    )
    summary = {
        "measurement_h5": str(path),
        "pdf": str(pdf),
        "pngs": [str(p) for p in outputs],
        "ell_max": ell_max,
        "ksz_ylim": ksz_ylim,
        "ksz_scale": float(args.plot_ksz_scale),
    }
    summary_path = output_dir / f"measurement_dell_{config.product_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{utc_now()}] Wrote measurement D_ell plot {pdf}", flush=True)
    print(f"[{utc_now()}] Wrote measurement D_ell plot summary {summary_path}", flush=True)


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
    p.add_argument("--patch-shear-only", action="store_true",
                   help="If a compatible spectra product exists, recompute ONLY the 4 shear-auto "
                        "spectra and overwrite them in place (the other 42 are unaffected by the "
                        "shape-noise fix). Falls back to a full recompute if no product exists.")
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
    p.add_argument("--no-cov-workspace-cache", action="store_true",
                   help="Do not read/write the on-disk covariance-workspace cache (rebuild every time).")
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
    p.add_argument("--no-cov-workspace-cache", action="store_true",
                   help="Do not read/write the on-disk covariance-workspace cache (rebuild every time).")
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
    p.add_argument("--corr-eigen-threshold", type=float, default=1.0e-8)
    p.set_defaults(func=run_validate)

    p = sub.add_parser("plot-measurement-dell")
    add_common(p)
    p.add_argument("--measurement-path", default=None)
    p.add_argument("--plot-dir", default=None)
    p.add_argument("--pdf-out", default=None)
    p.add_argument("--plot-ell-max", type=float, default=2800.0)
    p.add_argument("--plot-ksz-ylim", default="-5e-5,5e-5")
    p.add_argument("--plot-ksz-scale", type=float, default=1.0)
    p.set_defaults(func=run_plot_measurement_dell)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
