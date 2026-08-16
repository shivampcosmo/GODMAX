#!/usr/bin/env python
"""Audit stored-catalog versus native BaryonForge five-R200c pixel support."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from common import (
    catalog_cosmology,
    load_config,
    resolve_path,
    sha256_file,
    validate_parameter_crosswalk,
)
from paint_baryonforge import (
    _scientific_imports,
    build_ccl_cosmology,
    native_painter_geometry,
)


def audit(config: dict, *, nside: int, n_jobs: int) -> dict:
    crosswalk = validate_parameter_crosswalk(config)
    if not crosswalk["ok"]:
        raise ValueError(f"Parameter crosswalk failed: {crosswalk['failed']}")
    catalog_path = resolve_path(config["catalog"]["output_h5"], config["_config_path"])
    with h5py.File(catalog_path, "r") as handle:
        ra = np.asarray(handle["ra_deg"][:], dtype=np.float64)
        dec = np.asarray(handle["dec_deg"][:], dtype=np.float64)
        redshift = np.asarray(handle["z"][:], dtype=np.float64)
        mass_hmsun = np.asarray(handle["M200c_hMsun"][:], dtype=np.float64)
        radius_hmpc = np.asarray(handle["R200c_hMpc"][:], dtype=np.float64)
        distance_hmpc = np.asarray(handle["DA_hMpc"][:], dtype=np.float64)
        cosmo_values = catalog_cosmology(dict(handle.attrs))

    _, hp, ccl = _scientific_imports()
    max_paint = float(config["pasting"]["max_paint_R200c_factor"])
    native = native_painter_geometry(
        redshift,
        mass_hmsun / float(cosmo_values["h"]),
        cosmo=build_ccl_cosmology(cosmo_values),
        ccl=ccl,
        n_jobs=int(n_jobs),
        seed=int(config["pasting"]["random_seed"]),
        max_paint=max_paint,
    )
    catalog_support = max_paint * radius_hmpc / distance_hmpc
    native_support = np.asarray(native["support_rad"], dtype=np.float64)
    halo_vectors = hp.ang2vec(ra, dec, lonlat=True)

    different = 0
    catalog_extra = 0
    baryonforge_extra = 0
    for index in range(redshift.size):
        stored_pixels = hp.query_disc(
            int(nside),
            halo_vectors[index],
            catalog_support[index],
            inclusive=False,
            nest=False,
        )
        native_pixels = hp.query_disc(
            int(nside),
            halo_vectors[index],
            native_support[index],
            inclusive=False,
            nest=False,
        )
        if not np.array_equal(stored_pixels, native_pixels):
            different += 1
            catalog_extra += len(
                np.setdiff1d(stored_pixels, native_pixels, assume_unique=True)
            )
            baryonforge_extra += len(
                np.setdiff1d(native_pixels, stored_pixels, assume_unique=True)
            )

    max_pixel_radius = float(hp.max_pixrad(int(nside)))
    return {
        "schema": "baryonforge_godmax_support_audit_v1",
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "halo_count": int(redshift.size),
        "nside": int(nside),
        "n_jobs": int(n_jobs),
        "nominal_R200c_factor": max_paint,
        "support_baryonforge_over_catalog": (
            float(np.min(native_support / catalog_support)),
            float(np.max(native_support / catalog_support)),
        ),
        "catalog_support_angle_deg": (
            math.degrees(float(np.min(catalog_support))),
            math.degrees(float(np.max(catalog_support))),
        ),
        "baryonforge_support_angle_deg": (
            math.degrees(float(np.min(native_support))),
            math.degrees(float(np.max(native_support))),
        ),
        "healpix_max_pixel_radius_deg": math.degrees(max_pixel_radius),
        "baryonforge_query_disc_safe": bool(np.min(native_support) >= max_pixel_radius),
        "different_pixel_sets": int(different),
        "catalog_extra_boundary_pixels": int(catalog_extra),
        "baryonforge_extra_boundary_pixels": int(baryonforge_extra),
        "partition_z_max": native["partition_z_max"],
        "interpretation": (
            "Both painters use the same nominal five-R200c cutoff, but their native "
            "R200c and distance implementations do not produce bit-identical pixel support."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--nside", type=int)
    parser.add_argument("--n-jobs", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    report = audit(
        config,
        nside=int(args.nside or config["pasting"]["nside"]),
        n_jobs=int(args.n_jobs or config["baryonforge"]["n_jobs"]),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["baryonforge_query_disc_safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
