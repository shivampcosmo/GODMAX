#!/usr/bin/env python
"""Reproducible preflight evidence for the three-probe halo catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import asdf
import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "notebooks" / "xDESI"))
import abacus_lightcone_catalog as alc  # noqa: E402


def source_report(config_path: Path) -> dict:
    config = alc.load_config(config_path)
    spec = alc.catalog_specs_from_config(config)[0]
    input_root = Path(config["abacus"]["input_root"])
    files = alc.list_snapshot_files(
        input_root, spec.z_max, config["abacus"]["source_dirs"]
    )
    alc.validate_frozen_source_files(
        input_root, files, config["abacus"].get("source_files")
    )
    counts = {}
    reference_header = None
    for _, path in files:
        with asdf.open(path, lazy_load=True) as af:
            header = dict(af["header"])
            reference_header = reference_header or header
            alc._validate_source_header(reference_header, header, config, path)
            lightcone = af["halo_lightcone"]
            n_raw, _ = alc._get_first(lightcone, alc.FIELD_ALIASES["n_interp"])
            chi_raw, _ = alc._get_first(lightcone, alc.FIELD_ALIASES["chi"])
            n_interp = np.asarray(n_raw[:], dtype=np.float32)
            chi = np.asarray(chi_raw[:], dtype=np.float32)
            z = alc.make_chi_to_z_interpolator(header, spec.z_max + 0.2)(chi)
            mass = n_interp.astype(np.float64) * float(header["ParticleMassHMsun"])
            counts[path.parent.name] = int(
                np.count_nonzero(
                    alc.catalog_selection_mask(spec, z, mass, n_interp)
                )
            )
    coverage = alc.validate_explicit_source_coverage(
        input_root, config, [spec], reference_header
    )
    return {
        "counts_by_source_dir": counts,
        "n_selected": int(sum(counts.values())),
        "source_size_bytes": int(sum(path.stat().st_size for _, path in files)),
        "coverage": coverage,
    }


def catalog_report(path: Path, config_path: Path) -> dict:
    config = alc.load_config(config_path)
    spec = alc.catalog_specs_from_config(config)[0]
    with h5py.File(path, "r") as handle:
        n_halos = int(handle.attrs["n_halos"])
        predicate_ok = True
        mass_equation_ok = True
        row_order_ok = True
        previous_file = -1
        previous_row = -1
        particle_mass = float(handle.attrs["particle_mass_hmsun"])
        for start in range(0, n_halos, 262144):
            stop = min(start + 262144, n_halos)
            z = handle["z"][start:stop]
            mass = handle["M_particle_proxy_hMsun"][start:stop]
            n_interp = handle["N_interp"][start:stop]
            file_index = handle["source_file_index"][start:stop]
            row_index = handle["source_row_index"][start:stop]
            predicate_ok &= bool(
                np.all(alc.catalog_selection_mask(spec, z, mass, n_interp))
            )
            mass_equation_ok &= bool(
                np.array_equal(
                    mass, n_interp.astype(np.float64) * particle_mass
                )
            )
            row_order_ok &= bool(np.all(np.diff(file_index) >= 0))
            if len(file_index):
                row_order_ok &= int(file_index[0]) >= previous_file
                if int(file_index[0]) == previous_file:
                    row_order_ok &= int(row_index[0]) > previous_row
            transitions = np.flatnonzero(np.diff(file_index) != 0) + 1
            for rows in np.split(row_index, transitions):
                row_order_ok &= bool(np.all(np.diff(rows) > 0))
            if len(file_index):
                previous_file = int(file_index[-1])
                previous_row = int(row_index[-1])

        ranges = json.loads(handle.attrs["source_row_ranges_json"])
        identity_hash = hashlib.sha256()
        content_hash = hashlib.sha256()
        for source_range in ranges:
            start = int(source_range["output_start"])
            stop = int(source_range["output_stop"])
            n_rows = stop - start
            if n_rows == 0:
                continue
            row_order_ok &= bool(
                np.all(
                    handle["source_file_index"][start:stop]
                    == int(source_range["source_file_index"])
                )
            )
            for name, dtype in alc.CATALOG_DTYPES.items():
                values = np.asarray(
                    handle[name][start:stop],
                    dtype=np.dtype(dtype).newbyteorder("<"),
                )
                content_hash.update(name.encode("ascii") + b"\0")
                content_hash.update(np.asarray(n_rows, dtype="<i8").tobytes())
                content_hash.update(values.tobytes(order="C"))
                if name in alc.ROW_IDENTITY_FIELDS:
                    identity_hash.update(name.encode("ascii") + b"\0")
                    identity_hash.update(
                        np.asarray(n_rows, dtype="<i8").tobytes()
                    )
                    identity_hash.update(values.tobytes(order="C"))

        content_hash_ok = (
            content_hash.hexdigest()
            == str(handle.attrs["catalog_row_content_sha256"])
        )
        identity_hash_ok = (
            identity_hash.hexdigest() == str(handle.attrs["row_identity_sha256"])
        )
        source_content_hash_ok = True
        if config["abacus"].get("source_checksum_algorithm") == "sha256":
            source_content_json = str(handle.attrs["source_content_manifest_json"])
            _, recomputed = alc.canonical_json_sha256(
                json.loads(source_content_json)
            )
            source_content_hash_ok = (
                recomputed
                == str(handle.attrs["source_content_manifest_sha256"])
            )
        report = {
            "catalog_row_content_hash_valid": content_hash_ok,
            "catalog_row_content_sha256": str(
                handle.attrs["catalog_row_content_sha256"]
            ),
            "compression": str(handle["z"].compression),
            "mass_alias_same_object": bool(
                handle["M200c_hMsun"].id
                == handle["M_particle_proxy_hMsun"].id
            ),
            "mass_equation_exact": mass_equation_ok,
            "n_halos": n_halos,
            "predicate_all_rows": predicate_ok,
            "row_identity_sha256": str(handle.attrs["row_identity_sha256"]),
            "row_identity_hash_valid": identity_hash_ok,
            "source_content_manifest_hash_valid": source_content_hash_ok,
            "source_row_order_strict": row_order_ok,
            "working_mass_mode": str(handle.attrs["working_mass_mode"]),
        }
        required = {
            "catalog_row_content_hash_valid": content_hash_ok,
            "compression_is_lzf": handle["z"].compression == "lzf",
            "mass_alias_same_object": report["mass_alias_same_object"],
            "mass_equation_exact": mass_equation_ok,
            "predicate_all_rows": predicate_ok,
            "row_identity_hash_valid": identity_hash_ok,
            "source_content_manifest_hash_valid": source_content_hash_ok,
            "source_row_order_strict": row_order_ok,
            "working_mass_mode": report["working_mass_mode"]
            == alc.WORKING_MASS_MODE,
        }
        failed = [name for name, passed in required.items() if not passed]
        if failed:
            raise RuntimeError(f"Catalog validation failed: {failed}; report={report}")
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "notebooks/SBI_validate/three_probe_mock_experiment.yaml",
    )
    parser.add_argument("--catalog-path", type=Path)
    args = parser.parse_args()
    report = (
        catalog_report(args.catalog_path, args.config)
        if args.catalog_path
        else source_report(args.config)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
