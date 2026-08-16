#!/usr/bin/env python
"""Validate the paired profile files and the shared Backlight catalog contract."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Sequence

import h5py
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from common import (
    catalog_cosmology,
    jsonable,
    load_config,
    load_yaml,
    resolve_path,
    validate_parameter_crosswalk,
)


REQUIRED_CATALOG_FIELDS = (
    "ra_deg",
    "dec_deg",
    "z",
    "M200c_hMsun",
    "log10M200c_hMsun",
    "vlos_kms",
    "R200c_hMpc",
    "DA_hMpc",
)


def angular_cap_membership(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cra = math.radians(center_ra_deg)
    cdec = math.radians(center_dec_deg)
    cosine = np.sin(dec) * math.sin(cdec) + np.cos(dec) * math.cos(cdec) * np.cos(ra - cra)
    return cosine >= math.cos(math.radians(radius_deg))


def validate_catalog(config: dict, chunk_rows: int = 1_000_000) -> dict:
    config_path = config.get("_config_path")
    path = resolve_path(config["catalog"]["source_h5"], config_path)
    if not path.exists():
        return {"ok": False, "path": str(path), "error": "catalog does not exist"}

    cut = float(config["catalog"]["mass_cut_hMsun"])
    patch = config["sky_patch"]
    count = 0
    count_inner = 0
    min_angle_deg = math.inf
    max_angle_deg = 0.0
    max_roundtrip_mass_rel = 0.0
    with h5py.File(path, "r") as handle:
        missing = [name for name in REQUIRED_CATALOG_FIELDS if name not in handle]
        attrs = {str(key): jsonable(value) for key, value in handle.attrs.items()}
        if missing:
            return {"ok": False, "path": str(path), "missing_fields": missing, "attrs": attrs}
        cosmo = catalog_cosmology(attrs)
        nrows = int(handle["M200c_hMsun"].shape[0])
        h = float(cosmo["h"])
        for start in range(0, nrows, int(chunk_rows)):
            stop = min(nrows, start + int(chunk_rows))
            mass = np.asarray(handle["M200c_hMsun"][start:stop], dtype=np.float64)
            keep = mass > cut
            if not np.any(keep):
                continue
            selected_mass = mass[keep]
            count += int(np.count_nonzero(keep))
            roundtrip = (selected_mass / h) * h
            max_roundtrip_mass_rel = max(
                max_roundtrip_mass_rel,
                float(np.max(np.abs(roundtrip - selected_mass) / selected_mass)),
            )
            ra = np.asarray(handle["ra_deg"][start:stop], dtype=np.float64)[keep]
            dec = np.asarray(handle["dec_deg"][start:stop], dtype=np.float64)[keep]
            count_inner += int(
                np.count_nonzero(
                    angular_cap_membership(
                        ra,
                        dec,
                        float(patch["center_ra_deg"]),
                        float(patch["center_dec_deg"]),
                        float(patch["radius_deg"]),
                    )
                )
            )
            radius = np.asarray(handle["R200c_hMpc"][start:stop], dtype=np.float64)[keep]
            distance = np.asarray(handle["DA_hMpc"][start:stop], dtype=np.float64)[keep]
            angle = np.rad2deg(
                float(config["pasting"]["max_paint_R200c_factor"])
                * radius
                / np.maximum(distance, 1.0e-30)
            )
            min_angle_deg = min(min_angle_deg, float(np.min(angle)))
            max_angle_deg = max(max_angle_deg, float(np.max(angle)))

    expected = int(config["catalog"]["expected_selected_count"])
    buffer_deg = float(patch["edge_buffer_deg"])
    import healpy as hp

    requested_nsides = {
        "smoke": int(config["validation"]["smoke_nside"]),
        "production": int(config["validation"]["production_nside"]),
    }
    pixel_covering_radius_deg = {
        label: math.degrees(float(hp.max_pixrad(nside)))
        for label, nside in requested_nsides.items()
    }
    catalog_geometry_disc_safe = {
        label: min_angle_deg >= radius
        for label, radius in pixel_covering_radius_deg.items()
    }
    return {
        "ok": (
            count == expected
            and max_angle_deg < buffer_deg
            and all(catalog_geometry_disc_safe.values())
        ),
        "path": str(path),
        "n_parent": nrows,
        "n_selected_buffered": count,
        "n_selected_inner_cap_centers": count_inner,
        "n_selected_outer_buffer_centers": count - count_inner,
        "expected_selected_count": expected,
        "predicate": config["catalog"]["predicate"],
        "min_paint_angle_deg": min_angle_deg,
        "max_paint_angle_deg": max_angle_deg,
        "edge_buffer_deg": buffer_deg,
        "edge_buffer_safe": max_angle_deg < buffer_deg,
        "healpix_max_pixel_radius_deg": pixel_covering_radius_deg,
        "catalog_geometry_query_disc_safe": catalog_geometry_disc_safe,
        "mass_h_roundtrip_max_relative_error": max_roundtrip_mass_rel,
        "cosmology": cosmo,
        "attrs": attrs,
    }


def validate_catalog_cosmology(config: dict, catalog_report: dict) -> dict:
    if "cosmology" not in catalog_report:
        return {"ok": False, "error": "catalog cosmology unavailable"}
    bpath = resolve_path(config["profiles"]["baryonforge_params"], config.get("_config_path"))
    bcosmo = load_yaml(bpath)["cosmology"]
    ccosmo = catalog_report["cosmology"]
    pairs = {
        "h": (ccosmo["h"], bcosmo["h"]),
        "Omega_m": (ccosmo["Omega_m"], bcosmo["Omega_m"]),
        "Omega_b": (ccosmo["Omega_b"], bcosmo["Omega_b"]),
        "sigma8": (ccosmo["sigma8"], bcosmo["sigma8"]),
        "n_s": (ccosmo["n_s"], bcosmo["n_s"]),
        "w0": (ccosmo["w0"], bcosmo["w0"]),
    }
    mismatches = {
        name: {"catalog": float(actual), "baryonforge": float(expected)}
        for name, (actual, expected) in pairs.items()
        if not math.isclose(float(actual), float(expected), rel_tol=2.0e-12, abs_tol=2.0e-12)
    }
    return {"ok": not mismatches, "mismatches": mismatches}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-data", action="store_true", help="Only validate the two parameter files.")
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    crosswalk = validate_parameter_crosswalk(config)
    report = {"schema": config.get("schema"), "crosswalk": crosswalk}
    ok = bool(crosswalk["ok"])
    if not args.no_data:
        catalog = validate_catalog(config, chunk_rows=args.chunk_rows)
        cosmology = validate_catalog_cosmology(config, catalog)
        report.update({"catalog": catalog, "catalog_cosmology_match": cosmology})
        ok = ok and bool(catalog["ok"]) and bool(cosmology["ok"])
    report["ok"] = ok
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
