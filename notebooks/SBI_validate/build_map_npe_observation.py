"""Build one pasted-map observation for active map-NPE validation."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from theory_sbi_utils import DEFAULT_FIDUCIAL_PATH, default_parameter_specs, parse_probe_list
from map_npe_utils import (
    MeasurementConfig,
    generate_pasted_map_product,
    measure_binned_cls,
    save_map_measurement,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--theory-path", type=pathlib.Path, default=DEFAULT_FIDUCIAL_PATH)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--theta-ej-0", type=float, default=2.0)
    parser.add_argument("--nu-theta-ej-M", type=float, default=-0.1)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--probes", default="gg,gy,gtau,gkappa")
    parser.add_argument("--fsky", type=float, default=0.34)
    parser.add_argument("--add-survey-noise", action="store_true")
    parser.add_argument("--save-map-product", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    param_specs = default_parameter_specs()
    theta = np.array([args.theta_ej_0, args.nu_theta_ej_M], dtype=float)
    map_path = args.output_dir / "observation_map.pkl" if args.save_map_product else None
    map_data = generate_pasted_map_product(
        theta,
        param_specs=param_specs,
        nside=args.nside,
        random_seed=args.seed,
        save_path=map_path,
    )
    config = MeasurementConfig(
        nside=args.nside,
        fsky=args.fsky,
        add_survey_noise=bool(args.add_survey_noise),
    )
    measurement = measure_binned_cls(
        map_data,
        theory_path=args.theory_path,
        probes=parse_probe_list(args.probes),
        config=config,
        seed=args.seed + 17,
    )
    if not np.all(np.isfinite(measurement["data_vector"])):
        bad = int(measurement["data_vector"].size - np.count_nonzero(np.isfinite(measurement["data_vector"])))
        raise ValueError(f"Measured observation contains {bad} non-finite datavector entries")
    metadata = {
        "kind": "validation_pseudo_observation",
        "theta_truth": theta.tolist(),
        "theta_truth_is_for_validation_only": True,
        "nside": int(args.nside),
        "seed": int(args.seed),
        "add_survey_noise": bool(args.add_survey_noise),
        "probes": list(parse_probe_list(args.probes)),
        "map_path": None if map_path is None else str(map_path),
        "measurement": {
            "ngal": int(measurement["ngal"]),
            "fsky": float(measurement["fsky"]),
            "shot_noise_gg": float(measurement["shot_noise_gg"]),
            "estimator": str(measurement["estimator"]),
            "ell_min": float(np.min(measurement["ell"])),
            "ell_max": float(np.max(measurement["ell"])),
            "nell": int(len(measurement["ell"])),
        },
    }
    output_path = args.output_dir / "observation.npz"
    save_map_measurement(output_path, measurement, metadata)
    (args.output_dir / "observation_truth.json").write_text(
        json.dumps({"theta_truth": theta.tolist(), "seed": int(args.seed)}, indent=2, sort_keys=True)
    )
    print(f"Saved observation to {output_path}")
    print(json.dumps(metadata["measurement"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
