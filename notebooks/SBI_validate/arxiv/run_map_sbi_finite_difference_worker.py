"""Generate paired finite-difference pasted-map simulations for score estimates."""

from __future__ import annotations

import argparse
import json
import pathlib
import traceback

import numpy as np

from map_npe_utils import MeasurementConfig, measure_binned_cls
from map_sbi_pasted_utils import (
    DEFAULT_FIDUCIAL_PATH,
    generate_component_map_product,
    merge_signal_and_galaxy_products,
    save_json,
)
from theory_sbi_utils import default_parameter_specs, fiducial_theta, parse_param_specs, parse_probe_list, prior_bounds


def finite_difference_deltas(param_specs, step_fraction: float, explicit: list[str]) -> np.ndarray:
    theta0 = fiducial_theta(param_specs)
    prior_min, prior_max = prior_bounds(param_specs)
    deltas = np.asarray(step_fraction * (prior_max - prior_min), dtype=float)
    by_name = {}
    for item in explicit:
        if ":" not in item:
            raise ValueError("--fd-delta entries must have the form name:delta")
        name, value = item.split(":", 1)
        by_name[name.strip()] = float(value)
    for i, spec in enumerate(param_specs):
        if spec.name in by_name:
            deltas[i] = by_name[spec.name]
        if theta0[i] - deltas[i] <= prior_min[i] or theta0[i] + deltas[i] >= prior_max[i]:
            raise ValueError(
                f"Finite-difference step for {spec.name} leaves the prior: "
                f"theta0={theta0[i]}, delta={deltas[i]}, prior=[{prior_min[i]}, {prior_max[i]}]"
            )
    return deltas


def measure_product(map_data, fiducial_path, probes, config, seed):
    measurement = measure_binned_cls(
        map_data,
        theory_path=fiducial_path,
        probes=probes,
        config=config,
        seed=seed,
    )
    vector = np.asarray(measurement["data_vector"], dtype=float)
    if not np.all(np.isfinite(vector)):
        bad = int(vector.size - np.count_nonzero(np.isfinite(vector)))
        raise ValueError(f"Measured datavector contains {bad} non-finite entries")
    return measurement, vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--npairs-per-param", type=int, default=32)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--base-seed", type=int, default=20260626)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--fiducial-path", type=pathlib.Path, default=DEFAULT_FIDUCIAL_PATH)
    parser.add_argument("--probes", default="gg,gy,gtau,gkappa")
    parser.add_argument("--fsky", type=float, default=0.34)
    parser.add_argument("--param-spec", action="append", default=[])
    parser.add_argument("--fd-step-fraction", type=float, default=0.05)
    parser.add_argument("--fd-delta", action="append", default=[])
    parser.add_argument(
        "--vary-galaxies-with-params",
        action="store_true",
        help=(
            "Generate plus/minus galaxy catalogs at the plus/minus parameter values. "
            "The default reuses fiducial galaxies, appropriate for baryonic profile parameters."
        ),
    )
    parser.add_argument("--save-map-products", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "fd_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    map_dir = args.output_dir / "fd_maps" / f"rank{args.rank:02d}"
    if args.save_map_products:
        map_dir.mkdir(parents=True, exist_ok=True)

    param_specs = parse_param_specs(args.param_spec) if args.param_spec else default_parameter_specs()
    theta0 = fiducial_theta(param_specs)
    deltas = finite_difference_deltas(param_specs, args.fd_step_fraction, args.fd_delta)
    probes = parse_probe_list(args.probes)
    config = MeasurementConfig(nside=args.nside, fsky=args.fsky, add_survey_noise=True)
    pair_ids = np.arange(args.rank, args.npairs_per_param, args.world_size, dtype=int)

    signal_products = {}
    for ip, spec in enumerate(param_specs):
        for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
            theta = theta0.copy()
            theta[ip] += sign * deltas[ip]
            delta_tag = f"{deltas[ip]:.6g}".replace("-", "m").replace(".", "p")
            signal_path = (
                args.output_dir
                / f"fd_signal_maps_rank{args.rank:02d}_{spec.name}_{sign_name}_delta{delta_tag}_nside{args.nside}.pkl"
            )
            signal_products[(ip, sign_name)] = generate_component_map_product(
                theta,
                param_specs,
                nside=args.nside,
                random_seed=args.base_seed + 1009 * (ip + 1) + (1 if sign_name == "plus" else 2),
                get_signal_maps=True,
                get_galaxies=False,
                save_path=signal_path,
                use_cache=True,
            )

    ell = None
    delta_ell = None
    estimator = None
    status_by_param = {spec.name: [] for spec in param_specs}
    errors_by_param = {spec.name: [] for spec in param_specs}
    pair_ids_by_param = {spec.name: [] for spec in param_specs}
    plus_by_param = {spec.name: [] for spec in param_specs}
    minus_by_param = {spec.name: [] for spec in param_specs}
    ngal_by_param = {spec.name: [] for spec in param_specs}
    shot_by_param = {spec.name: [] for spec in param_specs}
    fsky_by_param = {spec.name: [] for spec in param_specs}

    for ip, spec in enumerate(param_specs):
        theta_plus = theta0.copy()
        theta_minus = theta0.copy()
        theta_plus[ip] += deltas[ip]
        theta_minus[ip] -= deltas[ip]
        for pair_id in pair_ids:
            seed = int(args.base_seed + 100000 * (ip + 1) + pair_id)
            try:
                if args.vary_galaxies_with_params:
                    plus_gal_path = map_dir / f"{spec.name}_plus_pair{pair_id:05d}.pkl" if args.save_map_products else None
                    minus_gal_path = map_dir / f"{spec.name}_minus_pair{pair_id:05d}.pkl" if args.save_map_products else None
                    galaxy_plus = generate_component_map_product(
                        theta_plus,
                        param_specs,
                        nside=args.nside,
                        random_seed=seed,
                        get_signal_maps=False,
                        get_galaxies=True,
                        save_path=plus_gal_path,
                        use_cache=False,
                    )
                    galaxy_minus = generate_component_map_product(
                        theta_minus,
                        param_specs,
                        nside=args.nside,
                        random_seed=seed,
                        get_signal_maps=False,
                        get_galaxies=True,
                        save_path=minus_gal_path,
                        use_cache=False,
                    )
                else:
                    gal_path = map_dir / f"{spec.name}_fidgal_pair{pair_id:05d}.pkl" if args.save_map_products else None
                    galaxy_plus = generate_component_map_product(
                        theta0,
                        param_specs,
                        nside=args.nside,
                        random_seed=seed,
                        get_signal_maps=False,
                        get_galaxies=True,
                        save_path=gal_path,
                        use_cache=False,
                    )
                    galaxy_minus = galaxy_plus

                map_plus = merge_signal_and_galaxy_products(signal_products[(ip, "plus")], galaxy_plus)
                map_minus = merge_signal_and_galaxy_products(signal_products[(ip, "minus")], galaxy_minus)
                measurement_plus, vector_plus = measure_product(
                    map_plus,
                    args.fiducial_path,
                    probes,
                    config,
                    seed + 1000003,
                )
                measurement_minus, vector_minus = measure_product(
                    map_minus,
                    args.fiducial_path,
                    probes,
                    config,
                    seed + 1000003,
                )
                plus_by_param[spec.name].append(vector_plus)
                minus_by_param[spec.name].append(vector_minus)
                pair_ids_by_param[spec.name].append(int(pair_id))
                ngal_by_param[spec.name].append(int(measurement_plus["ngal"]))
                shot_by_param[spec.name].append(float(measurement_plus["shot_noise_gg"]))
                fsky_by_param[spec.name].append(float(measurement_plus["fsky"]))
                ell = np.asarray(measurement_plus["ell"], dtype=float)
                delta_ell = np.asarray(measurement_plus["delta_ell"], dtype=float)
                estimator = str(measurement_plus["estimator"])
                status_by_param[spec.name].append("ok")
                errors_by_param[spec.name].append("")
                print(f"[fd rank {args.rank}] param={spec.name} pair={pair_id} seed={seed} ok")
            except Exception as exc:
                status_by_param[spec.name].append("failed")
                errors_by_param[spec.name].append(traceback.format_exc())
                print(f"[fd rank {args.rank}] param={spec.name} pair={pair_id} failed: {exc!r}")

    if ell is None:
        ell = np.array([], dtype=float)
        delta_ell = np.array([], dtype=float)

    payload = {
        "param_names": np.asarray([spec.name for spec in param_specs]),
        "deltas": np.asarray(deltas, dtype=float),
        "theta_fiducial": theta0,
        "ell": np.asarray(ell, dtype=float),
        "delta_ell": np.asarray(delta_ell, dtype=float),
        "probes": np.asarray(probes),
        "rank": np.asarray(args.rank),
        "world_size": np.asarray(args.world_size),
    }
    metadata = {
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "npairs_per_param": int(args.npairs_per_param),
        "nside": int(args.nside),
        "base_seed": int(args.base_seed),
        "fiducial_path": str(args.fiducial_path),
        "probes": list(probes),
        "fd_step_fraction": float(args.fd_step_fraction),
        "deltas": {spec.name: float(deltas[ip]) for ip, spec in enumerate(param_specs)},
        "vary_galaxies_with_params": bool(args.vary_galaxies_with_params),
        "estimator": estimator,
        "status_by_param": status_by_param,
        "errors_by_param": errors_by_param,
    }
    for spec in param_specs:
        name = spec.name
        payload[f"pair_id__{name}"] = np.asarray(pair_ids_by_param[name], dtype=int)
        payload[f"plus__{name}"] = (
            np.vstack(plus_by_param[name]) if plus_by_param[name] else np.empty((0, 0), dtype=float)
        )
        payload[f"minus__{name}"] = (
            np.vstack(minus_by_param[name]) if minus_by_param[name] else np.empty((0, 0), dtype=float)
        )
        payload[f"ngal__{name}"] = np.asarray(ngal_by_param[name], dtype=int)
        payload[f"shot_noise_gg__{name}"] = np.asarray(shot_by_param[name], dtype=float)
        payload[f"fsky__{name}"] = np.asarray(fsky_by_param[name], dtype=float)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, indent=2, sort_keys=True))

    shard_path = shard_dir / f"fd_shard_rank{args.rank:02d}_of{args.world_size:02d}.npz"
    np.savez_compressed(shard_path, **payload)
    save_json(
        shard_path.with_suffix(".json"),
        {
            "path": str(shard_path),
            "rank": int(args.rank),
            "world_size": int(args.world_size),
            "n_success_by_param": {name: int(len(vals)) for name, vals in pair_ids_by_param.items()},
        },
    )
    print(f"Saved finite-difference shard to {shard_path}")


if __name__ == "__main__":
    main()
