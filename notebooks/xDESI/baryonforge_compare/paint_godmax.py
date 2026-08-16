#!/usr/bin/env python
"""Run the existing GODMAX native painter with the matched comparison config."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from common import (
    MAP_PRODUCT_SCHEMA,
    REPO_ROOT,
    assert_map_contract_unchanged,
    canonical_json,
    current_map_contract,
    godmax_profiles_class_path,
    load_config,
    load_config_and_freeze_map_contract,
    resolve_path,
    validate_parameter_crosswalk,
)


XDESI_DIR = REPO_ROOT / "notebooks" / "xDESI"
if str(XDESI_DIR) not in sys.path:
    sys.path.insert(0, str(XDESI_DIR))


def _helpers():
    from abacus_pasting_helpers import partial_map_path, run_paste_split

    return partial_map_path, run_paste_split


def annotate_product(
    path: Path,
    config: dict,
    nside: int,
    split_index: int,
    num_splits: int,
    *,
    pixel_workers: int | None = None,
    contract_override: dict | None = None,
) -> dict:
    catalog_path = resolve_path(config["catalog"]["output_h5"], config["_config_path"])
    with h5py.File(catalog_path, "r") as catalog_handle:
        catalog_attrs = dict(catalog_handle.attrs)
        halo_count = int(catalog_handle["z"].shape[0])
        selected_redshift = np.asarray(catalog_handle["z"][:], dtype=np.float64)
    contract = contract_override or current_map_contract(config)
    with h5py.File(path, "r") as handle:
        painted_halos = int(
            handle.attrs.get(
                "n_split_halos",
                halo_count if int(num_splits) == 1 else -1,
            )
        )
        actual_profiles_class = str(
            handle.attrs.get("profiles_class_fqname", "")
        )
    expected_profiles_class = str(
        contract["profile_integration_contract"]["godmax"][
            "profiles_class_fqname"
        ]
    )
    if actual_profiles_class != expected_profiles_class:
        raise ValueError(
            "The native map was produced by the wrong GODMAX profile class: "
            f"actual={actual_profiles_class!r}, expected={expected_profiles_class!r}."
        )
    complete_catalog_paint = bool(
        int(num_splits) == 1
        and int(split_index) == 0
        and painted_halos == halo_count
    )
    provenance = {
        **contract,
        "schema": MAP_PRODUCT_SCHEMA,
        "backend": "godmax",
        "nside": int(nside),
        "ordering": "RING",
        "catalog_path": contract["catalog_path"],
        "catalog_sha256": contract["catalog_sha256"],
        "params_path": contract["godmax_params_path"],
        "params_sha256": contract["godmax_params_sha256"],
        "mass_predicate": str(config["catalog"]["predicate"]),
        "selection_predicate": str(config["catalog"]["predicate"]),
        "halo_count": halo_count,
        "halo_only": True,
        "z_min": float(np.min(selected_redshift)),
        "z_max": float(np.max(selected_redshift)),
        "h": float(catalog_attrs["h"]),
        "H0": float(catalog_attrs["H0"]),
        "Omega_M": float(catalog_attrs["Omega_M"]),
        "Omega_b": float(catalog_attrs["Omega_b"]),
        "max_paint_R200c_factor": float(config["pasting"]["max_paint_R200c_factor"]),
        "smooth_profiles": bool(config["pasting"]["smooth_profiles"]),
        "n_halos_painted": painted_halos,
        "complete_catalog_paint": complete_catalog_paint,
        "unit_boundary": {
            "catalog_mass": "M200c_hMsun in Msun/h",
            "catalog_radius": "R200c_hMpc is physical Mpc/h",
            "catalog_distance": "DA_hMpc is physical angular-diameter distance in Mpc/h",
            "godmax_profile_radius": "3D r_array is comoving Mpc/h",
            "godmax_projected_radius": "DA_hMpc times angle is physical Mpc/h",
            "map_ymap": "dimensionless Compton-y",
            "map_kappa_cmb": "dimensionless halo-only CMB convergence",
        },
        "split_index": int(split_index),
        "num_splits": int(num_splits),
        "godmax_pixel_workers": int(
            pixel_workers
            if pixel_workers is not None
            else config["pasting"].get("pixel_workers", 1)
        ),
        "profiles_class_fqname": actual_profiles_class,
    }
    with h5py.File(path, "r+") as handle:
        handle.attrs["schema"] = provenance["schema"]
        handle.attrs["comparison_schema"] = provenance["schema"]
        handle.attrs["backend"] = "godmax"
        handle.attrs["nside"] = int(nside)
        handle.attrs["ordering"] = "RING"
        handle.attrs["catalog_sha256"] = provenance["catalog_sha256"]
        handle.attrs["catalog_path"] = provenance["catalog_path"]
        handle.attrs["params_sha256"] = provenance["params_sha256"]
        handle.attrs["comparison_config_sha256"] = provenance[
            "comparison_config_sha256"
        ]
        handle.attrs["source_manifest_sha256"] = provenance[
            "source_manifest_sha256"
        ]
        handle.attrs["effective_godmax_config_sha256"] = provenance[
            "effective_godmax_config_sha256"
        ]
        handle.attrs["selection_predicate"] = provenance["selection_predicate"]
        handle.attrs["halo_count"] = provenance["halo_count"]
        handle.attrs["halo_only"] = True
        for key in ("z_min", "z_max", "h", "H0", "Omega_M", "Omega_b"):
            handle.attrs[key] = provenance[key]
        handle.attrs["max_paint_R200c_factor"] = provenance["max_paint_R200c_factor"]
        handle.attrs["smooth_profiles"] = provenance["smooth_profiles"]
        handle.attrs["complete_catalog_paint"] = provenance[
            "complete_catalog_paint"
        ]
        handle.attrs["n_halos_painted"] = provenance["n_halos_painted"]
        handle.attrs["profiles_class_fqname"] = provenance[
            "profiles_class_fqname"
        ]
        handle.attrs["noise_policy"] = provenance["noise_policy"]
        handle.attrs["provisional_status"] = provenance["provisional_status"]
        group = handle.require_group("comparison_provenance")
        group.attrs["json"] = canonical_json(provenance)
    provenance["output_h5"] = str(path)
    return provenance


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--nside", type=int)
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--num-splits", type=int)
    parser.add_argument("--pixel-workers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    frozen_contract = None
    if args.dry_run:
        config = load_config(config_path)
    else:
        config, frozen_contract = load_config_and_freeze_map_contract(config_path)
    crosswalk = validate_parameter_crosswalk(config)
    if not crosswalk["ok"]:
        print(json.dumps(crosswalk, indent=2, sort_keys=True))
        return 2
    catalog_path = resolve_path(config["catalog"]["output_h5"], config["_config_path"])
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Missing filtered catalog {catalog_path}. Run prepare_catalog.py first."
        )
    nside = int(args.nside or config["pasting"]["nside"])
    configured_pixel_workers = int(config["pasting"]["pixel_workers"])
    actual_pixel_workers = int(
        args.pixel_workers
        if args.pixel_workers is not None
        else configured_pixel_workers
    )
    if actual_pixel_workers != configured_pixel_workers:
        raise ValueError(
            "Full GODMAX production must use the configured pixel worker count so the "
            "saved comparison contract identifies the executed path: "
            f"pixel_workers={actual_pixel_workers}, configured={configured_pixel_workers}."
        )
    default_splits = config["pasting"].get("num_splits_by_nside", {})
    num_splits = int(args.num_splits or default_splits.get(nside, default_splits.get(str(nside), 1)))
    if not (0 <= int(args.split_index) < num_splits):
        raise ValueError(f"split-index must satisfy 0 <= split-index < {num_splits}.")
    if args.dry_run:
        partial_map_path, _ = _helpers()
        expected = partial_map_path(
            config,
            config["pasting"]["catalog_key"],
            nside,
            args.split_index,
            num_splits,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "config": str(config_path),
                    "catalog": str(catalog_path),
                    "nside": nside,
                    "split_index": int(args.split_index),
                    "num_splits": num_splits,
                    "expected_output": str(expected),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if frozen_contract is None:
        raise RuntimeError("Internal error: production map contract was not frozen.")
    assert_map_contract_unchanged(
        frozen_contract,
        current_map_contract(config),
        context="GODMAX pre-painter validation",
    )
    # Must be set before abacus_pasting_helpers imports JAX/GODMAX modules.
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    _, run_paste_split = _helpers()
    output = Path(
        run_paste_split(
            config_path,
            config["pasting"]["catalog_key"],
            int(args.split_index),
            num_splits,
            nside,
            overwrite=bool(args.overwrite),
            pixel_workers=actual_pixel_workers,
            profiles_class_path=godmax_profiles_class_path(config),
        )
    )
    temporary = output.with_name(f".{output.name}.comparison.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(
            f"Refusing to replace pre-existing comparison staging file {temporary}."
        )
    os.replace(output, temporary)
    assert_map_contract_unchanged(
        frozen_contract,
        current_map_contract(config),
        context=(
            "GODMAX post-paint validation; the unvalidated native map remains at "
            f"{temporary}"
        ),
    )
    report = annotate_product(
        temporary,
        config,
        nside,
        args.split_index,
        num_splits,
        pixel_workers=actual_pixel_workers,
        contract_override=frozen_contract,
    )
    assert_map_contract_unchanged(
        frozen_contract,
        current_map_contract(config),
        context=(
            "GODMAX pre-publication validation; the staged map remains at "
            f"{temporary}"
        ),
    )
    os.replace(temporary, output)
    report["output_h5"] = str(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
