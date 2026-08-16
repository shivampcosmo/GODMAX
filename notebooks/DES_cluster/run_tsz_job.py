#!/usr/bin/env python
"""Run one configured tSZ paste and fail closed on the saved product contract."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-des-cluster")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tsz_pasting as tp


def _scan_map(path: Path, block_size: int = 4_000_000) -> dict:
    with h5py.File(path, "r") as handle:
        if list(handle.keys()) != ["maps"] or list(handle["maps"].keys()) != ["map_ymap"]:
            raise AssertionError(f"Unexpected product structure in {path}.")
        dataset = handle[tp.MAP_DATASET]
        finite = True
        nonnegative = True
        nonzero = 0
        map_min = np.inf
        map_max = -np.inf
        map_sum = 0.0
        for start in range(0, len(dataset), block_size):
            values = np.asarray(dataset[start : min(start + block_size, len(dataset))])
            finite &= bool(np.all(np.isfinite(values)))
            nonnegative &= bool(np.all(values >= 0.0))
            nonzero += int(np.count_nonzero(values))
            map_min = min(map_min, float(np.min(values, initial=np.inf)))
            map_max = max(map_max, float(np.max(values, initial=-np.inf)))
            map_sum += float(np.sum(values, dtype=np.float64))
        dtype = str(dataset.dtype)
        npix = int(len(dataset))
        attrs = {str(key): value for key, value in handle.attrs.items()}
    return {
        "dtype": dtype,
        "npix": npix,
        "finite": finite,
        "nonnegative": nonnegative,
        "nonzero": nonzero,
        "map_min": map_min,
        "map_max": map_max,
        "map_sum": map_sum,
        "n_halos_painted": int(attrs["n_halos_painted"]),
        "selected_rows_available": int(attrs["selected_rows_available"]),
        "selected_z_max": float(attrs["selected_z_max"]),
        "selection_redshift_max": float(attrs["selection_redshift_max"]),
        "selection_redshift_max_inclusive": bool(attrs["selection_redshift_max_inclusive"]),
        "selected_row_index_sha256": str(attrs["selected_row_index_sha256"]),
        "painted_row_index_sha256": str(attrs["painted_row_index_sha256"]),
        "complete_selected_catalog_painted": bool(attrs["complete_selected_catalog_painted"]),
        "n_pairs_below_projected_grid": int(attrs["n_pairs_below_projected_grid"]),
        "selection_predicate": str(attrs["selection_predicate"]),
        "helper_sha256": str(attrs["helper_sha256"]),
        "config_sha256": str(attrs["config_sha256"]),
        "config_sources_json": str(attrs["config_sources_json"]),
        "diagnostics": json.loads(str(attrs["diagnostics_json"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        default=str(HERE / "params_tsz_zmax0p85.yaml"),
        help="Complete params file or a base_params override YAML.",
    )
    parser.add_argument("--max-halos", type=int, default=None, help="Bounded benchmark only.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--expected-config-sha256",
        default=None,
        help="Fail unless the merged configuration has this exact SHA256.",
    )
    parser.add_argument(
        "--expected-selected-row-sha256",
        default=None,
        help="Fail unless preflight selects this exact ordered source-row identity.",
    )
    parser.add_argument(
        "--expected-halos",
        type=int,
        default=None,
        help="Fail unless the complete configured selection has this row count.",
    )
    args = parser.parse_args()

    cfg = tp.load_params(args.params)
    config_sha256 = tp._configuration_hash(cfg)
    if (
        args.expected_config_sha256 is not None
        and config_sha256 != args.expected_config_sha256
    ):
        raise AssertionError(
            "Merged configuration does not match the approved contract: "
            f"{config_sha256} != {args.expected_config_sha256}."
        )
    preflight = tp.preflight_catalog(cfg)
    if (
        args.expected_selected_row_sha256 is not None
        and preflight["selected_row_index_sha256"]
        != args.expected_selected_row_sha256
    ):
        raise AssertionError(
            "Selected-row identity does not match the approved contract: "
            f"{preflight['selected_row_index_sha256']} != "
            f"{args.expected_selected_row_sha256}."
        )
    if (
        args.expected_halos is not None
        and int(preflight["selected_rows"]) != int(args.expected_halos)
    ):
        raise AssertionError(
            "Selected-row count does not match the approved contract: "
            f"{preflight['selected_rows']} != {args.expected_halos}."
        )
    print("[tsz-job] preflight=" + json.dumps(preflight, sort_keys=True, default=str), flush=True)
    result = tp.run_tsz_paste(
        args.params,
        max_halos=args.max_halos,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    path = Path(result["path"])
    product = _scan_map(path)
    expected_halos = (
        int(preflight["selected_rows"])
        if args.max_halos is None
        else min(int(args.max_halos), int(preflight["selected_rows"]))
    )
    failures = []
    if product["dtype"] != "float32":
        failures.append(f"dtype={product['dtype']}")
    if product["config_sha256"] != config_sha256:
        failures.append(
            f"saved config_sha256={product['config_sha256']} expected={config_sha256}"
        )
    if product["npix"] != 12 * int(cfg["map"]["nside"]) ** 2:
        failures.append(f"npix={product['npix']}")
    if not product["finite"] or not product["nonnegative"]:
        failures.append("map is nonfinite or negative")
    if float(cfg["map"]["pressure_amplitude"]) > 0.0 and product["nonzero"] == 0:
        failures.append("positive-amplitude map is identically zero")
    if product["n_halos_painted"] != expected_halos:
        failures.append(f"painted={product['n_halos_painted']} expected={expected_halos}")
    if product["n_pairs_below_projected_grid"] != 0:
        failures.append(f"below_grid={product['n_pairs_below_projected_grid']}")
    configured_zmax = cfg["catalog"]["selection"].get("redshift_max")
    if configured_zmax is not None:
        if not product["selection_redshift_max_inclusive"]:
            failures.append("selection_redshift_max_inclusive=false")
        if product["selection_redshift_max"] != float(configured_zmax):
            failures.append(
                f"selection_redshift_max={product['selection_redshift_max']}"
            )
        if product["selected_z_max"] > float(configured_zmax):
            failures.append(f"selected_z_max={product['selected_z_max']}")
    if args.max_halos is None:
        if not product["complete_selected_catalog_painted"]:
            failures.append("complete_selected_catalog_painted=false")
        if product["selected_row_index_sha256"] != product["painted_row_index_sha256"]:
            failures.append("selected/painted row-index hashes differ")
    diagnostics = product["diagnostics"]
    requested_backend = str(cfg["runtime"]["jax_platforms"]).lower()
    backend_aliases = {requested_backend}
    if requested_backend == "cuda":
        backend_aliases.add("gpu")
    if requested_backend != "auto" and str(diagnostics["jax_backend"]).lower() not in backend_aliases:
        failures.append(
            f"jax_backend={diagnostics['jax_backend']} requested={requested_backend}"
        )
    if diagnostics.get("jax_x64") is not True:
        failures.append(f"jax_x64={diagnostics.get('jax_x64')}")
    if failures:
        raise AssertionError("; ".join(failures))
    product["output_sha256"] = tp._sha256_file(path)
    marker = path.with_name(path.name + ".validated.json")
    if marker.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite validation marker {marker}.")
    marker_payload = {
        "schema": "godmax_des_cluster_tsz_validation_v1",
        "validated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "path": str(path),
        "preflight_selected_rows": int(preflight["selected_rows"]),
        **product,
    }
    staging = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
    staging.write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, marker)
    print(
        "[tsz-job] result="
        + json.dumps(
            {
                "path": str(path),
                "validation_marker": str(marker),
                "preflight_selected_rows": int(preflight["selected_rows"]),
                **product,
            },
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
