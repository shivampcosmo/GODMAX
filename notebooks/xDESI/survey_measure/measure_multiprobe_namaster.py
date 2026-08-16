#!/usr/bin/env python
"""Measure xDESI multi-probe spectra and Gaussian covariance with NaMaster."""

from __future__ import annotations

import argparse
from pathlib import Path

from multiprobe_namaster import (
    MAP_CONSTRUCTION_VERSION,
    MEASUREMENT_PIPELINE_VERSION,
    MeasurementConfig,
    add_common_cli_args,
    config_from_args,
    load_map_product,
    measure_all,
    save_measurement_product,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_cli_args(parser)
    parser.add_argument("--maps-path", default=None, help="Input cached map HDF5. Defaults to stage map path.")
    parser.add_argument("--measurement-out", default=None, help="Output spectra/covariance HDF5. Defaults to stage path.")
    parser.add_argument("--no-covariance", action="store_true", help="Measure spectra only; skip joint covariance.")
    parser.add_argument("--skip-cov-eig", action="store_true", help="Skip covariance eigenvalue diagnostics.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging.")
    return parser.parse_args()


def _config_from_map_metadata(config: MeasurementConfig, map_metadata: dict) -> MeasurementConfig:
    if str(map_metadata.get("pipeline_version", "")) != MEASUREMENT_PIPELINE_VERSION:
        raise ValueError("Cached map product is from a stale pipeline version; regenerate it.")
    if str(map_metadata.get("map_construction_version", "")) != MAP_CONSTRUCTION_VERSION:
        raise ValueError("Cached map product uses a stale map-construction algorithm; regenerate it.")
    if not str(map_metadata.get("map_product_id", "")):
        raise ValueError("Cached map product has no map_product_id; regenerate it.")
    map_config = map_metadata.get("config", {})
    for key in (
        "pipeline_version",
        "stage",
        "nside",
        "act_downgrade",
        "shear_e_to_kappa_sign",
        "shear_mask_dataset",
        "shear_noise_attr",
        "subtract_masked_mean",
        "mask_apodization_deg",
        "mask_apodization_type",
        "pair_overlap_mean_subtract",
    ):
        if key not in map_config:
            raise ValueError(f"Cached map product is missing construction config key {key!r}.")
        if map_config[key] != getattr(config, key):
            raise ValueError(
                f"Requested {key}={getattr(config, key)!r} does not match cached map value {map_config[key]!r}."
            )
    if "lmax" in map_config:
        map_lmax = int(map_config["lmax"])
        if int(config.lmax) > map_lmax:
            raise ValueError(f"Requested lmax={config.lmax} exceeds cached-map lmax={map_lmax}.")
    config.validate()
    return config


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    maps_path = Path(args.maps_path).resolve() if args.maps_path else config.default_maps_path

    print(f"[{utc_now()}] Loading cached maps: {maps_path}", flush=True)
    fields, map_metadata = load_map_product(maps_path)
    config = _config_from_map_metadata(config, map_metadata)
    config.output_dir = args.output_dir
    config.compute_covariance = not args.no_covariance
    config.compute_covariance_eigenvalues = not args.skip_cov_eig
    output = Path(args.measurement_out).resolve() if args.measurement_out else config.default_measurement_path

    print(
        f"[{utc_now()}] Measuring spectra "
        f"(nside={config.nside}, lmax={config.lmax}, n_bins={config.n_bins}, "
        f"covariance={config.compute_covariance})",
        flush=True,
    )
    result = measure_all(fields, config, verbose=not args.quiet)
    save_measurement_product(output, result, map_metadata, overwrite=args.force)
    print(f"[{utc_now()}] Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
