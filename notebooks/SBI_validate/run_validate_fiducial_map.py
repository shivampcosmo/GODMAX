"""Regenerate the corrected fiducial theory and pasted-map validation products."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from backlight_metadata import backlight_validation_settings
from fiducial_theory_datavector import DEFAULT_OUTPUT, build_and_save_fiducial
from pasted_map_cls_validation import (
    DEFAULT_HALO_CATALOG,
    DEFAULT_MAP_PATH,
    DEFAULT_VALIDATION_OUTPUT,
    measure_pasted_map_cls,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory-output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--map-output", default=str(DEFAULT_VALIDATION_OUTPUT))
    parser.add_argument("--map-path", default=str(DEFAULT_MAP_PATH))
    parser.add_argument("--halo-catalog", default=str(DEFAULT_HALO_CATALOG))
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--gal-zmin", type=float, default=0.4)
    parser.add_argument("--gal-zmax", type=float, default=0.6)
    parser.add_argument("--nbar-comoving", type=float, default=1.0e-4)
    parser.add_argument("--kappa-source", choices=("cmb", "lsst"), default="cmb")
    parser.add_argument("--regenerate-theory", action="store_true")
    parser.add_argument("--regenerate-map-validation", action="store_true")
    parser.add_argument("--regenerate-pasted-maps", action="store_true")
    args = parser.parse_args()

    settings = backlight_validation_settings(args.halo_catalog)
    theory_output = pathlib.Path(args.theory_output)
    map_output = pathlib.Path(args.map_output)

    print("Backlight matched settings:")
    print(json.dumps({
        "hod_mass_cut": settings["hod_mass_cut"],
        "halo_param_overrides": settings["halo_param_overrides"],
        "sim_param_overrides": settings["sim_param_overrides"],
        "mass_metadata": settings["mass_metadata"],
        "source_asdf": settings["source_metadata"].get("source_asdf"),
    }, indent=2, sort_keys=True))

    if args.regenerate_theory or not theory_output.exists():
        extra_metadata = {
            "backlight_mass_corrected": True,
            "backlight_source_metadata": settings["source_metadata"],
            "backlight_mass_metadata": settings["mass_metadata"],
            "backlight_raw_halo_mass_cut": 1.0e13,
        }
        theory_result = build_and_save_fiducial(
            output_path=theory_output,
            gal_zmin=args.gal_zmin,
            gal_zmax=args.gal_zmax,
            nbar_comoving=args.nbar_comoving,
            hod_mass_cut=settings["hod_mass_cut"],
            kappa_source=args.kappa_source,
            sim_param_overrides=settings["sim_param_overrides"],
            halo_param_overrides=settings["halo_param_overrides"],
            extra_metadata=extra_metadata,
        )
        print(f"Saved corrected theory product: {theory_result['output_path']}")
        print(json.dumps(theory_result["metadata"]["quality_checks"], indent=2, sort_keys=True))
    else:
        print(f"Using existing theory product: {theory_output}")

    if args.regenerate_map_validation or not map_output.exists():
        map_result = measure_pasted_map_cls(
            theory_path=theory_output,
            map_path=args.map_path,
            halo_catalog=args.halo_catalog,
            output_path=map_output,
            nside=args.nside,
            gal_zmin=args.gal_zmin,
            gal_zmax=args.gal_zmax,
            regenerate_maps=args.regenerate_pasted_maps,
        )
        print(f"Saved corrected map validation: {map_result['output_path']}")
        print(json.dumps(map_result["metadata"]["map_theory_ratio_diagnostics"], indent=2, sort_keys=True))
    else:
        print(f"Using existing map validation product: {map_output}")


if __name__ == "__main__":
    main()
