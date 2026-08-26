"""Worker process for one active pasted-map NPE round."""

from __future__ import annotations

import argparse
import json
import pathlib
import traceback

import numpy as np

from theory_sbi_utils import DEFAULT_FIDUCIAL_PATH, default_parameter_specs, parse_probe_list
from map_npe_utils import (
    MeasurementConfig,
    generate_pasted_map_product,
    measure_binned_cls,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta-table", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--theory-path", type=pathlib.Path, default=DEFAULT_FIDUCIAL_PATH)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--base-seed", type=int, default=20260527)
    parser.add_argument("--probes", default="gg,gy,gtau,gkappa")
    parser.add_argument("--fsky", type=float, default=0.34)
    parser.add_argument("--add-survey-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-map-products", action="store_true")
    args = parser.parse_args()

    table = np.load(args.theta_table, allow_pickle=True)
    theta_all = np.asarray(table["theta"], dtype=float)
    sim_id_all = np.asarray(table["sim_id"], dtype=int)
    row_all = np.arange(len(theta_all), dtype=int)
    local_rows = row_all[row_all % int(args.world_size) == int(args.rank)]

    shard_dir = args.output_dir / "shards" / f"round{args.round_index:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    map_dir = args.output_dir / "maps" / f"round{args.round_index:02d}" / f"rank{args.rank:02d}"
    if args.save_map_products:
        map_dir.mkdir(parents=True, exist_ok=True)

    param_specs = default_parameter_specs()
    probes = parse_probe_list(args.probes)
    config = MeasurementConfig(
        nside=args.nside,
        fsky=args.fsky,
        add_survey_noise=bool(args.add_survey_noise),
    )

    data_vectors = []
    theta_done = []
    sim_ids_done = []
    cl_by_probe = {probe: [] for probe in probes}
    ngal = []
    shot_noise = []
    fsky = []
    status = []
    errors = []

    for row in local_rows:
        theta = theta_all[row]
        sim_id = int(sim_id_all[row])
        seed = int(args.base_seed + 100000 * args.round_index + sim_id)
        try:
            map_path = map_dir / f"map_sim{sim_id:06d}.pkl" if args.save_map_products else None
            map_data = generate_pasted_map_product(
                theta,
                param_specs=param_specs,
                nside=args.nside,
                random_seed=seed,
                save_path=map_path,
            )
            measurement = measure_binned_cls(
                map_data,
                theory_path=args.theory_path,
                probes=probes,
                config=config,
                seed=seed + 17,
            )
            if not np.all(np.isfinite(measurement["data_vector"])):
                bad = int(
                    measurement["data_vector"].size
                    - np.count_nonzero(np.isfinite(measurement["data_vector"]))
                )
                raise ValueError(f"Measured simulation contains {bad} non-finite datavector entries")
            data_vectors.append(np.asarray(measurement["data_vector"], dtype=float))
            theta_done.append(theta)
            sim_ids_done.append(sim_id)
            for probe in probes:
                cl_by_probe[probe].append(np.asarray(measurement["cl_by_probe"][probe], dtype=float))
            ngal.append(int(measurement["ngal"]))
            shot_noise.append(float(measurement["shot_noise_gg"]))
            fsky.append(float(measurement["fsky"]))
            status.append("ok")
            errors.append("")
            print(f"[rank {args.rank}] round={args.round_index} sim_id={sim_id} theta={theta} ok")
        except Exception as exc:
            status.append("failed")
            errors.append(traceback.format_exc())
            print(f"[rank {args.rank}] round={args.round_index} sim_id={sim_id} failed: {exc!r}")

    if data_vectors:
        first_len = len(data_vectors[0])
        data_vectors_arr = np.vstack(data_vectors)
    else:
        first_len = 0
        data_vectors_arr = np.empty((0, 0), dtype=float)

    payload = {
        "theta": np.asarray(theta_done, dtype=float).reshape((-1, len(param_specs))),
        "sim_id": np.asarray(sim_ids_done, dtype=int),
        "data_vector": data_vectors_arr,
        "ngal": np.asarray(ngal, dtype=int),
        "shot_noise_gg": np.asarray(shot_noise, dtype=float),
        "fsky": np.asarray(fsky, dtype=float),
        "round_index": np.asarray(args.round_index),
        "rank": np.asarray(args.rank),
        "world_size": np.asarray(args.world_size),
        "metadata_json": np.asarray(json.dumps({
            "theta_table": str(args.theta_table),
            "round_index": int(args.round_index),
            "rank": int(args.rank),
            "world_size": int(args.world_size),
            "nside": int(args.nside),
            "base_seed": int(args.base_seed),
            "probes": list(probes),
            "status": status,
            "errors": errors,
            "n_success": int(len(data_vectors)),
            "data_vector_length": int(first_len),
        }, indent=2, sort_keys=True)),
    }
    for probe, values in cl_by_probe.items():
        payload[f"cl_{probe}"] = np.vstack(values) if values else np.empty((0, 0), dtype=float)

    shard_path = shard_dir / f"shard_rank{args.rank:02d}_of{args.world_size:02d}.npz"
    np.savez_compressed(shard_path, **payload)
    save_json(
        shard_path.with_suffix(".json"),
        {
            "shard_path": str(shard_path),
            "round_index": int(args.round_index),
            "rank": int(args.rank),
            "n_success": int(len(data_vectors)),
            "n_requested": int(len(local_rows)),
        },
    )
    print(f"Saved shard to {shard_path}")


if __name__ == "__main__":
    main()
