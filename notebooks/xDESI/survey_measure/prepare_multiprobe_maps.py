#!/usr/bin/env python
"""Prepare cached HEALPix maps for xDESI multi-probe NaMaster measurements."""

from __future__ import annotations

import argparse
from pathlib import Path

from multiprobe_namaster import (
    SurveyBundle,
    add_common_cli_args,
    build_probe_maps,
    config_from_args,
    save_map_product,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_args(parser)
    parser.add_argument("--maps-out", default=None, help="Output HDF5 map product. Defaults to stage output path.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate input paths; do not build maps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    bundle = SurveyBundle.from_root(args.survey_root)
    output = Path(args.maps_out).resolve() if args.maps_out else config.default_maps_path

    print(f"[{utc_now()}] Validating survey bundle: {bundle.root}", flush=True)
    bundle.validate_files()
    des_nz_path = Path(config.des_y3_source_nz_fits)
    if not des_nz_path.exists():
        raise FileNotFoundError(f"Missing DES Y3 source n(z) FITS: {des_nz_path}")
    if args.validate_only:
        print(f"[{utc_now()}] Validation passed.", flush=True)
        return

    print(
        f"[{utc_now()}] Building {config.stage} map product "
        f"(nside={config.nside}, lmax={config.lmax}, act_downgrade={config.act_downgrade})",
        flush=True,
    )
    fields, metadata = build_probe_maps(bundle, config)
    save_map_product(output, fields, metadata, overwrite=args.force)
    print(f"[{utc_now()}] Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
