"""Generate pasted-map simulations for score-compressed likelihood tests."""

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
from theory_sbi_utils import default_parameter_specs, fiducial_theta, parse_param_specs, parse_probe_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--nsim-total", type=int, default=256)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--base-seed", type=int, default=20260526)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--fiducial-path", type=pathlib.Path, default=DEFAULT_FIDUCIAL_PATH)
    parser.add_argument("--probes", default="gg,gy,gtau,gkappa")
    parser.add_argument("--fsky", type=float, default=0.34)
    parser.add_argument("--param-spec", action="append", default=[])
    parser.add_argument("--save-map-products", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    map_dir = args.output_dir / "maps" / f"rank{args.rank:02d}"
    if args.save_map_products:
        map_dir.mkdir(parents=True, exist_ok=True)

    param_specs = parse_param_specs(args.param_spec) if args.param_spec else default_parameter_specs()
    theta0 = fiducial_theta(param_specs)
    probes = parse_probe_list(args.probes)
    config = MeasurementConfig(nside=args.nside, fsky=args.fsky, add_survey_noise=True)
    sim_ids = np.arange(args.rank, args.nsim_total, args.world_size, dtype=int)

    signal_path = args.output_dir / f"signal_maps_rank{args.rank:02d}_nside{args.nside}.pkl"
    signal_product = generate_component_map_product(
        theta0,
        param_specs,
        nside=args.nside,
        random_seed=args.base_seed,
        get_signal_maps=True,
        get_galaxies=False,
        save_path=signal_path,
        use_cache=True,
    )

    data_vectors = []
    sim_done = []
    ngal = []
    shot_noise = []
    fsky = []
    status = []
    errors = []
    cl_by_probe = {probe: [] for probe in probes}
    ell = None
    delta_ell = None
    estimator = None

    for sim_id in sim_ids:
        seed = int(args.base_seed + int(sim_id))
        try:
            gal_path = map_dir / f"galaxies_sim{sim_id:06d}.pkl" if args.save_map_products else None
            galaxy_product = generate_component_map_product(
                theta0,
                param_specs,
                nside=args.nside,
                random_seed=seed,
                get_signal_maps=False,
                get_galaxies=True,
                save_path=gal_path,
                use_cache=False,
            )
            map_data = merge_signal_and_galaxy_products(signal_product, galaxy_product)
            measurement = measure_binned_cls(
                map_data,
                theory_path=args.fiducial_path,
                probes=probes,
                config=config,
                seed=seed + 1000003,
            )
            vector = np.asarray(measurement["data_vector"], dtype=float)
            if not np.all(np.isfinite(vector)):
                bad = int(vector.size - np.count_nonzero(np.isfinite(vector)))
                raise ValueError(f"Measured datavector contains {bad} non-finite entries")
            data_vectors.append(vector)
            sim_done.append(int(sim_id))
            ngal.append(int(measurement["ngal"]))
            shot_noise.append(float(measurement["shot_noise_gg"]))
            fsky.append(float(measurement["fsky"]))
            ell = np.asarray(measurement["ell"], dtype=float)
            delta_ell = np.asarray(measurement["delta_ell"], dtype=float)
            estimator = str(measurement["estimator"])
            for probe in probes:
                cl_by_probe[probe].append(np.asarray(measurement["cl_by_probe"][probe], dtype=float))
            status.append("ok")
            errors.append("")
            print(f"[rank {args.rank}] sim_id={sim_id} seed={seed} ok")
        except Exception as exc:
            status.append("failed")
            errors.append(traceback.format_exc())
            print(f"[rank {args.rank}] sim_id={sim_id} failed: {exc!r}")

    if data_vectors:
        data_vectors_arr = np.vstack(data_vectors)
        nell = len(ell)
    else:
        data_vectors_arr = np.empty((0, 0), dtype=float)
        nell = 0
        ell = np.array([], dtype=float)
        delta_ell = np.array([], dtype=float)

    payload = {
        "sim_id": np.asarray(sim_done, dtype=int),
        "data_vector": data_vectors_arr,
        "ell": np.asarray(ell, dtype=float),
        "delta_ell": np.asarray(delta_ell, dtype=float),
        "probes": np.asarray(probes),
        "theta_fiducial": theta0,
        "ngal": np.asarray(ngal, dtype=int),
        "shot_noise_gg": np.asarray(shot_noise, dtype=float),
        "fsky": np.asarray(fsky, dtype=float),
        "rank": np.asarray(args.rank),
        "world_size": np.asarray(args.world_size),
        "metadata_json": np.asarray(json.dumps({
            "rank": int(args.rank),
            "world_size": int(args.world_size),
            "nsim_total": int(args.nsim_total),
            "n_requested": int(len(sim_ids)),
            "n_success": int(len(data_vectors)),
            "nside": int(args.nside),
            "base_seed": int(args.base_seed),
            "fiducial_path": str(args.fiducial_path),
            "probes": list(probes),
            "nell": int(nell),
            "estimator": estimator,
            "status": status,
            "errors": errors,
        }, indent=2, sort_keys=True)),
    }
    for probe, values in cl_by_probe.items():
        payload[f"cl_{probe}"] = np.vstack(values) if values else np.empty((0, 0), dtype=float)

    shard_path = shard_dir / f"shard_rank{args.rank:02d}_of{args.world_size:02d}.npz"
    np.savez_compressed(shard_path, **payload)
    save_json(
        shard_path.with_suffix(".json"),
        {
            "path": str(shard_path),
            "rank": int(args.rank),
            "world_size": int(args.world_size),
            "n_requested": int(len(sim_ids)),
            "n_success": int(len(data_vectors)),
        },
    )
    print(f"Saved shard to {shard_path}")


if __name__ == "__main__":
    main()
