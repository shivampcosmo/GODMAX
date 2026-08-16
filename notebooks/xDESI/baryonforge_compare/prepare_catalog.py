#!/usr/bin/env python
"""Create the immutable, buffered M200c_hMsun > 1e13 comparison catalog."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from common import load_config, resolve_path, sha256_file
from validate_config import validate_catalog


def _copy_selected_catalog(
    source: Path,
    output: Path,
    *,
    mass_field: str,
    mass_cut: float,
    predicate: str,
    chunk_rows: int,
    overwrite: bool,
) -> dict:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite explicitly.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()

    parent_hash = sha256_file(source)
    selected_total = 0
    try:
        with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
            nrows = int(src[mass_field].shape[0])
            names = [
                name
                for name, dataset in src.items()
                if isinstance(dataset, h5py.Dataset) and dataset.ndim == 1 and dataset.shape[0] == nrows
            ]
            if mass_field not in names:
                raise KeyError(f"Mass field {mass_field!r} is not a row-aligned dataset in {source}.")

            outputs = {}
            for name in names:
                dataset = src[name]
                outputs[name] = dst.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dataset.dtype,
                    chunks=(min(int(chunk_rows), 262_144),),
                    compression="lzf",
                    shuffle=True,
                )
            outputs["source_row"] = dst.create_dataset(
                "source_row",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(min(int(chunk_rows), 262_144),),
                compression="lzf",
                shuffle=True,
            )

            for start in range(0, nrows, int(chunk_rows)):
                stop = min(nrows, start + int(chunk_rows))
                mass = np.asarray(src[mass_field][start:stop], dtype=np.float64)
                keep = mass > float(mass_cut)
                nkeep = int(np.count_nonzero(keep))
                if nkeep == 0:
                    continue
                old_stop = selected_total
                selected_total += nkeep
                for name in names:
                    out = outputs[name]
                    out.resize((selected_total,))
                    out[old_stop:selected_total] = src[name][start:stop][keep]
                source_row = outputs["source_row"]
                source_row.resize((selected_total,))
                source_row[old_stop:selected_total] = np.arange(start, stop, dtype=np.int64)[keep]

            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["comparison_schema"] = "baryonforge_godmax_catalog_v1"
            dst.attrs["source_catalog_path"] = str(source)
            dst.attrs["source_catalog_sha256"] = parent_hash
            dst.attrs["selection_predicate"] = str(predicate)
            dst.attrs["selection_mass_field"] = str(mass_field)
            dst.attrs["selection_mass_cut_hMsun"] = float(mass_cut)
            dst.attrs["selection_is_strict"] = True
            dst.attrs["selection_parent_rows"] = int(nrows)
            dst.attrs["selection_rows"] = int(selected_total)
            dst.attrs["n_halos"] = int(selected_total)
            dst.attrs["retains_catalog_edge_buffer"] = True
            dst.attrs["log10_m_min_hmsun"] = float(np.log10(mass_cut))
            dst.attrs["filter_selection"] = str(predicate)
            dst.attrs["R200c_coordinate_type"] = "proper/physical radius"
            dst.attrs["DA_coordinate_type"] = "proper/physical angular-diameter distance"
            if selected_total:
                selected_z = np.asarray(outputs["z"][:], dtype=np.float64)
                selected_mass = np.asarray(outputs[mass_field][:], dtype=np.float64)
                dst.attrs["z_min"] = float(np.min(selected_z))
                dst.attrs["z_max"] = float(np.max(selected_z))
                dst.attrs["mass_min_hMsun"] = float(np.min(selected_mass))
                dst.attrs["mass_max_hMsun"] = float(np.max(selected_mass))
            dst.flush()
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    return {
        "source_h5": str(source),
        "source_sha256": parent_hash,
        "output_h5": str(output),
        "selection_predicate": predicate,
        "selected_rows": int(selected_total),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.check_only:
        report = validate_catalog(config, chunk_rows=args.chunk_rows)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 2

    source = resolve_path(config["catalog"]["source_h5"], config["_config_path"])
    output = resolve_path(config["catalog"]["output_h5"], config["_config_path"])
    report = _copy_selected_catalog(
        source,
        output,
        mass_field=str(config["catalog"]["mass_field"]),
        mass_cut=float(config["catalog"]["mass_cut_hMsun"]),
        predicate=str(config["catalog"]["predicate"]),
        chunk_rows=int(args.chunk_rows),
        overwrite=bool(args.overwrite),
    )
    expected = int(config["catalog"]["expected_selected_count"])
    report["expected_selected_rows"] = expected
    report["ok"] = int(report["selected_rows"]) == expected
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
