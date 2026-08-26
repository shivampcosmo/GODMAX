"""Combine pasted-map simulations into score-compressed likelihood posteriors."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time

import numpy as np

from map_sbi_pasted_utils import (
    DEFAULT_FIDUCIAL_PATH,
    block_to_payload,
    build_simulation_score_block,
    build_two_point_score_block,
    compress_with_block_at_theta0,
    parse_analysis_specs,
    raw_block_indices,
    sample_compressed_posterior,
    save_json,
    simulation_score_stability_diagnostics,
    stable_cholesky,
)
from theory_sbi_utils import default_parameter_specs, fiducial_theta, parse_param_specs, prior_bounds


def load_shards(input_dir: pathlib.Path, nsim_total: int) -> dict[str, object]:
    shard_paths = sorted((input_dir / "shards").glob("shard_rank*_of*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {input_dir / 'shards'}")
    shards = [np.load(path, allow_pickle=True) for path in shard_paths]
    sim_id = np.concatenate([np.asarray(s["sim_id"], dtype=int) for s in shards])
    data_vector = np.vstack([np.asarray(s["data_vector"], dtype=float) for s in shards if len(s["data_vector"])])
    order = np.argsort(sim_id)
    sim_id = sim_id[order]
    data_vector = data_vector[order]
    if len(sim_id) != nsim_total:
        raise RuntimeError(f"Expected {nsim_total} valid simulations but found {len(sim_id)}")
    if len(np.unique(sim_id)) != nsim_total:
        raise RuntimeError("Simulation IDs are not unique")
    if not np.all(np.isfinite(data_vector)):
        bad = int(data_vector.size - np.count_nonzero(np.isfinite(data_vector)))
        raise RuntimeError(f"Combined datavectors contain {bad} non-finite entries")
    first = shards[0]
    raw_probes = tuple(str(p) for p in first["probes"])
    ell = np.asarray(first["ell"], dtype=float)
    delta_ell = np.asarray(first["delta_ell"], dtype=float)
    ngal = np.concatenate([np.asarray(s["ngal"], dtype=int) for s in shards])[order]
    shot_noise = np.concatenate([np.asarray(s["shot_noise_gg"], dtype=float) for s in shards])[order]
    fsky = np.concatenate([np.asarray(s["fsky"], dtype=float) for s in shards])[order]
    cl_by_probe = {}
    for probe in raw_probes:
        cl_by_probe[probe] = np.vstack([np.asarray(s[f"cl_{probe}"], dtype=float) for s in shards if len(s[f"cl_{probe}"])])[order]
    return {
        "shard_paths": shard_paths,
        "sim_id": sim_id,
        "data_vector": data_vector,
        "raw_probes": raw_probes,
        "ell": ell,
        "delta_ell": delta_ell,
        "ngal": ngal,
        "shot_noise_gg": shot_noise,
        "fsky": fsky,
        "cl_by_probe": cl_by_probe,
    }


def load_fd_shards(input_dir: pathlib.Path) -> dict[str, object]:
    shard_paths = sorted((input_dir / "fd_shards").glob("fd_shard_rank*_of*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No finite-difference shard files found in {input_dir / 'fd_shards'}")
    shards = [np.load(path, allow_pickle=True) for path in shard_paths]
    first = shards[0]
    param_names = tuple(str(p) for p in first["param_names"])
    deltas = np.asarray(first["deltas"], dtype=float)
    out = {
        "shard_paths": shard_paths,
        "param_names": param_names,
        "deltas": deltas,
        "raw_probes": tuple(str(p) for p in first["probes"]),
        "ell": np.asarray(first["ell"], dtype=float),
        "delta_ell": np.asarray(first["delta_ell"], dtype=float),
        "plus": {},
        "minus": {},
        "pair_id": {},
    }
    for name in param_names:
        pair = np.concatenate([np.asarray(s[f"pair_id__{name}"], dtype=int) for s in shards])
        plus = np.vstack([np.asarray(s[f"plus__{name}"], dtype=float) for s in shards if len(s[f"plus__{name}"])])
        minus = np.vstack([np.asarray(s[f"minus__{name}"], dtype=float) for s in shards if len(s[f"minus__{name}"])])
        order = np.argsort(pair)
        pair = pair[order]
        plus = plus[order]
        minus = minus[order]
        if len(pair) == 0:
            raise RuntimeError(f"No finite-difference pairs were generated for parameter {name}")
        if len(np.unique(pair)) != len(pair):
            raise RuntimeError(f"Finite-difference pair IDs are not unique for parameter {name}")
        if plus.shape != minus.shape:
            raise RuntimeError(f"Plus/minus finite-difference shapes disagree for parameter {name}")
        if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(minus)):
            raise RuntimeError(f"Finite-difference datavectors contain non-finite entries for parameter {name}")
        out["pair_id"][name] = pair
        out["plus"][name] = plus
        out["minus"][name] = minus
    return out


def selected_raw_labels(probes: tuple[str, ...], ell: np.ndarray) -> tuple[str, ...]:
    labels = []
    for probe in probes:
        for ell_value in ell:
            labels.append(f"{probe}:ell={ell_value:.6g}")
    return tuple(labels)


def posterior_width(samples: np.ndarray) -> np.ndarray:
    return np.std(np.asarray(samples, dtype=float), axis=0)


def bootstrap_widths(
    u_centered: np.ndarray,
    theta0: np.ndarray,
    prior_min: np.ndarray,
    prior_max: np.ndarray,
    fisher_transform: np.ndarray,
    likelihood: str,
    nsim: int,
    ngrid: int,
    nboot: int,
    seed: int,
) -> np.ndarray:
    if len(theta0) != 2 or nboot <= 0:
        return np.empty((0, len(theta0)), dtype=float)
    rng = np.random.default_rng(seed)
    widths = []
    for _ in range(nboot):
        choice = rng.integers(0, len(u_centered), size=len(u_centered))
        cov = np.cov(u_centered[choice].T, ddof=1)
        samples, _ = sample_compressed_posterior(
            prior_min,
            prior_max,
            theta0,
            fisher_transform,
            np.zeros(len(theta0), dtype=float),
            cov,
            likelihood,
            nsim,
            ngrid,
            posterior_samples=4000,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        widths.append(posterior_width(samples))
    return np.asarray(widths, dtype=float)


def run_posterior_for_block(
    score_method: str,
    block,
    data: np.ndarray,
    theta0: np.ndarray,
    prior_min: np.ndarray,
    prior_max: np.ndarray,
    combined: dict[str, object],
    ngrid: int,
    posterior_samples: int,
    nboot: int,
    npseudo: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    theta_summary, u_raw = compress_with_block_at_theta0(data, block, theta0)
    u_mean = np.mean(u_raw, axis=0)
    u_centered = u_raw - u_mean[None, :]
    cov_u = np.cov(u_centered.T, ddof=1)
    cov_u = 0.5 * (cov_u + cov_u.T)
    chol_u, chol_jitter = stable_cholesky(cov_u)
    corr_u = cov_u / np.sqrt(np.clip(np.outer(np.diag(cov_u), np.diag(cov_u)), 1.0e-300, np.inf))
    nsim = len(data)

    likelihoods = {}
    for ilike, likelihood in enumerate(("student_t", "gaussian")):
        samples, info = sample_compressed_posterior(
            prior_min,
            prior_max,
            theta0,
            block.fisher_transform,
            np.zeros(len(theta0), dtype=float),
            cov_u,
            likelihood,
            nsim,
            ngrid,
            posterior_samples,
            seed + 101 * (ilike + 1),
        )
        likelihoods[likelihood] = {"samples": samples, "info": info}

    n128 = min(128, nsim)
    samples_128 = None
    if n128 >= len(theta0) + 2:
        cov_128 = np.cov(u_centered[:n128].T, ddof=1)
        samples_128, _info_128 = sample_compressed_posterior(
            prior_min,
            prior_max,
            theta0,
            block.fisher_transform,
            np.zeros(len(theta0), dtype=float),
            cov_128,
            "student_t",
            n128,
            ngrid,
            posterior_samples,
            seed + 909,
        )

    boot_width = bootstrap_widths(
        u_centered,
        theta0,
        prior_min,
        prior_max,
        block.fisher_transform,
        "student_t",
        nsim,
        ngrid,
        nboot,
        seed + 12345,
    )

    pseudo = []
    for i in range(min(npseudo, nsim)):
        samples_i, _info_i = sample_compressed_posterior(
            prior_min,
            prior_max,
            theta0,
            block.fisher_transform,
            u_centered[i],
            cov_u,
            "student_t",
            nsim,
            ngrid,
            min(8000, posterior_samples),
            seed + 20000 + i,
        )
        pseudo.append({
            "sim_id": int(np.asarray(combined["sim_id"])[i]),
            "mean": samples_i.mean(axis=0).tolist(),
            "std": samples_i.std(axis=0).tolist(),
        })

    st_samples = likelihoods["student_t"]["samples"]
    gauss_samples = likelihoods["gaussian"]["samples"]
    width_full = posterior_width(st_samples)
    width_128 = None if samples_128 is None else posterior_width(samples_128)
    eig = np.linalg.eigvalsh(block.fisher)
    mean_sigma = u_mean / np.sqrt(np.clip(np.diag(cov_u) / nsim, 1.0e-300, np.inf))
    diagnostics = {
        "score_method": score_method,
        "nsim": int(nsim),
        "n128": int(n128),
        "ngrid": int(ngrid),
        "posterior_samples": int(posterior_samples),
        "theta_fiducial": theta0.tolist(),
        "prior_min": prior_min.tolist(),
        "prior_max": prior_max.tolist(),
        "fisher_eigenvalues": eig.tolist(),
        "fisher_condition": float(np.max(eig) / np.clip(np.min(eig), 1.0e-300, np.inf)),
        "map_compressed_mean_u": u_mean.tolist(),
        "map_compressed_mean_sigma": mean_sigma.tolist(),
        "compressed_cov_cholesky_jitter": float(chol_jitter),
        "student_t_mean": st_samples.mean(axis=0).tolist(),
        "student_t_std": st_samples.std(axis=0).tolist(),
        "gaussian_mean": gauss_samples.mean(axis=0).tolist(),
        "gaussian_std": gauss_samples.std(axis=0).tolist(),
        "student_t_128_mean": None if samples_128 is None else samples_128.mean(axis=0).tolist(),
        "student_t_128_std": None if samples_128 is None else samples_128.std(axis=0).tolist(),
        "width_128_over_full": None if width_128 is None else (width_128 / np.clip(width_full, 1.0e-300, np.inf)).tolist(),
        "bootstrap_width_ratio_p16_p50_p84": None if len(boot_width) == 0 else np.percentile(
            boot_width / np.clip(width_full[None, :], 1.0e-300, np.inf), [16, 50, 84], axis=0
        ).tolist(),
        "pseudo_observation_checks": pseudo,
    }
    payload = {
        "samples_student_t": st_samples,
        "samples_gaussian": gauss_samples,
        "samples_student_t_128": np.empty((0, len(theta0))) if samples_128 is None else samples_128,
        "theta_summary": theta_summary,
        "summary_u_raw": u_raw,
        "summary_u_mean": u_mean,
        "summary_u_centered": u_centered,
        "summary_u_cov": cov_u,
        "summary_u_corr": corr_u,
        "summary_u_chol": chol_u,
        "bootstrap_widths": boot_width,
    }
    return payload, diagnostics


def run_one_analysis(
    name: str,
    probes: tuple[str, ...],
    combined: dict[str, object],
    fd_combined: dict[str, object] | None,
    fiducial_path: pathlib.Path,
    param_specs,
    output_dir: pathlib.Path,
    ngrid: int,
    posterior_samples: int,
    nboot: int,
    npseudo: int,
    seed: int,
    theory_backend: str,
    score_methods: tuple[str, ...],
) -> dict[str, object]:
    t0 = time.time()
    raw_probes = tuple(combined["raw_probes"])
    ell = np.asarray(combined["ell"], dtype=float)
    nell = len(ell)
    idx = raw_block_indices(raw_probes, probes, nell)
    data = np.asarray(combined["data_vector"], dtype=float)[:, idx]
    theta0 = fiducial_theta(param_specs)
    prior_min, prior_max = prior_bounds(param_specs)
    analysis_dir = output_dir / "analyses" / name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    method_payloads: dict[str, dict[str, np.ndarray]] = {}
    method_diagnostics: dict[str, dict[str, object]] = {}
    block_payloads: dict[str, dict[str, np.ndarray]] = {}

    if "analytic" in score_methods:
        block = build_two_point_score_block(
            fiducial_path,
            probes=probes,
            param_specs=param_specs,
            ell_min=float(np.min(ell)),
            ell_max=float(np.max(ell)),
            theory_backend=theory_backend,
            jit_compile=True,
        )
        if data.shape[1] != len(block.mu0):
            raise RuntimeError(f"Data length {data.shape[1]} does not match analytical block length {len(block.mu0)}")
        payload_a, diag_a = run_posterior_for_block(
            "analytic_theory_covariance_score",
            block,
            data,
            theta0,
            prior_min,
            prior_max,
            combined,
            ngrid,
            posterior_samples,
            nboot,
            npseudo,
            seed,
        )
        method_payloads["analytic"] = payload_a
        method_diagnostics["analytic"] = diag_a
        block_payloads["analytic"] = block_to_payload(block)

    if "finite_difference" in score_methods:
        if fd_combined is None:
            raise RuntimeError("Finite-difference score was requested, but no finite-difference shards were loaded")
        fd_raw_probes = tuple(fd_combined["raw_probes"])
        if len(np.asarray(fd_combined["ell"])) != nell or not np.allclose(np.asarray(fd_combined["ell"], dtype=float), ell):
            raise RuntimeError("Finite-difference ell grid does not match the fiducial simulation ell grid")
        fd_idx = raw_block_indices(fd_raw_probes, probes, nell)
        fd_param_names = tuple(fd_combined["param_names"])
        fd_deltas_all = np.asarray(fd_combined["deltas"], dtype=float)
        plus_by_param = []
        minus_by_param = []
        fd_deltas = []
        n_fd_measurements = 0
        for spec in param_specs:
            if spec.name not in fd_param_names:
                raise RuntimeError(f"Finite-difference shards are missing parameter {spec.name}")
            plus = np.asarray(fd_combined["plus"][spec.name], dtype=float)[:, fd_idx]
            minus = np.asarray(fd_combined["minus"][spec.name], dtype=float)[:, fd_idx]
            plus_by_param.append(plus)
            minus_by_param.append(minus)
            fd_deltas.append(float(fd_deltas_all[fd_param_names.index(spec.name)]))
            n_fd_measurements += int(len(plus) + len(minus))
        block_fd, fd_score_diag = build_simulation_score_block(
            name="2pt_finite_difference",
            fiducial_samples=data,
            plus_samples_by_param=plus_by_param,
            minus_samples_by_param=minus_by_param,
            deltas=fd_deltas,
            labels=selected_raw_labels(probes, ell),
            probes=probes,
            shrinkage=None,
        )
        score_stability = simulation_score_stability_diagnostics(
            data,
            plus_by_param,
            minus_by_param,
            fd_deltas,
            nboot=nboot,
            seed=seed + 777,
        )
        payload_fd, diag_fd = run_posterior_for_block(
            "finite_difference_simulation_score",
            block_fd,
            data,
            theta0,
            prior_min,
            prior_max,
            combined,
            ngrid,
            posterior_samples,
            nboot,
            npseudo,
            seed + 500000,
        )
        diag_fd["score_calibration"] = fd_score_diag
        diag_fd["score_stability"] = score_stability
        diag_fd["finite_difference_deltas"] = {spec.name: float(delta) for spec, delta in zip(param_specs, fd_deltas)}
        diag_fd["n_finite_difference_measurements"] = int(n_fd_measurements)
        diag_fd["n_finite_difference_pairs_by_param"] = {
            spec.name: int(min(len(plus_by_param[ip]), len(minus_by_param[ip])))
            for ip, spec in enumerate(param_specs)
        }
        method_payloads["finite_difference"] = payload_fd
        method_diagnostics["finite_difference"] = diag_fd
        block_payloads["finite_difference"] = block_to_payload(block_fd)

    if not method_payloads:
        raise RuntimeError("No score methods were requested")

    primary_method = "analytic" if "analytic" in method_payloads else next(iter(method_payloads))
    primary = method_payloads[primary_method]
    primary_block = block_payloads[primary_method]
    for method, diag in method_diagnostics.items():
        diag["name"] = name
        diag["runtime_sec"] = time.time() - t0
        diag["probes"] = list(probes)
        diag["ell_min"] = float(np.min(ell))
        diag["ell_max"] = float(np.max(ell))
        diag["nell"] = int(nell)
        diag["ndata"] = int(data.shape[1])

    diagnostics = {
        **method_diagnostics[primary_method],
        "name": name,
        "runtime_sec": time.time() - t0,
        "probes": list(probes),
        "ell_min": float(np.min(ell)),
        "ell_max": float(np.max(ell)),
        "nell": int(nell),
        "ndata": int(data.shape[1]),
        "primary_score_method": primary_method,
        "available_score_methods": sorted(method_payloads),
        "score_methods": method_diagnostics,
    }

    payload = {
        "samples_student_t": primary["samples_student_t"],
        "samples_gaussian": primary["samples_gaussian"],
        "samples_student_t_128": primary["samples_student_t_128"],
        "theta_fiducial": theta0,
        "prior_min": prior_min,
        "prior_max": prior_max,
        "ell": ell,
        "delta_ell": np.asarray(combined["delta_ell"], dtype=float),
        "selected_data_vectors": data,
        "theta_summary": primary["theta_summary"],
        "summary_u_raw": primary["summary_u_raw"],
        "summary_u_mean": primary["summary_u_mean"],
        "summary_u_centered": primary["summary_u_centered"],
        "summary_u_cov": primary["summary_u_cov"],
        "summary_u_corr": primary["summary_u_corr"],
        "summary_u_chol": primary["summary_u_chol"],
        "bootstrap_widths": primary["bootstrap_widths"],
        "map_minus_theory_raw_mean": np.mean(data, axis=0) - primary_block["mu0"],
        "score_method_names": np.asarray(sorted(method_payloads)),
        "metadata_json": np.asarray(json.dumps(diagnostics, indent=2, sort_keys=True)),
    }
    payload.update({f"block_{key}": value for key, value in primary_block.items()})
    for method, method_payload in method_payloads.items():
        suffix = "" if method == primary_method else f"_{method}"
        if not suffix:
            continue
        for key, value in method_payload.items():
            payload[f"{key}{suffix}"] = value
        for key, value in block_payloads[method].items():
            payload[f"block_{key}{suffix}"] = value

    samples_path = analysis_dir / "map_sbi_compressed_samples.npz"
    np.savez_compressed(samples_path, **payload)
    save_json(analysis_dir / "map_sbi_diagnostics.json", diagnostics)
    return {"name": name, "samples_path": str(samples_path), "diagnostics": diagnostics}


def save_cls_ensemble(input_dir: pathlib.Path, combined: dict[str, object]) -> pathlib.Path:
    ell = np.asarray(combined["ell"], dtype=float)
    payload = {
        "ell": ell,
        "delta_ell": np.asarray(combined["delta_ell"], dtype=float),
        "probes": np.asarray(combined["raw_probes"]),
        "sim_id": np.asarray(combined["sim_id"], dtype=int),
    }
    for probe, arr in combined["cl_by_probe"].items():
        arr = np.asarray(arr, dtype=float)
        payload[f"cl_{probe}"] = arr
        payload[f"cl_{probe}_mean"] = np.mean(arr, axis=0)
        payload[f"cl_{probe}_std"] = np.std(arr, axis=0, ddof=1)
        payload[f"cl_{probe}_p16"] = np.percentile(arr, 16, axis=0)
        payload[f"cl_{probe}_p84"] = np.percentile(arr, 84, axis=0)
    path = input_dir / "map_sbi_cls_ensemble.npz"
    np.savez_compressed(path, **payload)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--fiducial-path", type=pathlib.Path, default=DEFAULT_FIDUCIAL_PATH)
    parser.add_argument("--nsim-total", type=int, default=256)
    parser.add_argument("--analysis", default="all:gg,gy,gtau,gkappa;gg_only:gg;pressure:gy;lensing:gkappa")
    parser.add_argument("--param-spec", action="append", default=[])
    parser.add_argument("--posterior-samples", type=int, default=50000)
    parser.add_argument("--ngrid", type=int, default=900)
    parser.add_argument("--nboot", type=int, default=32)
    parser.add_argument("--npseudo", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--theory-backend", choices=("linearized", "direct"), default="linearized")
    parser.add_argument(
        "--score-methods",
        default="analytic,finite_difference",
        help="Comma-separated score methods to run: analytic,finite_difference",
    )
    args = parser.parse_args()

    t0 = time.time()
    param_specs = parse_param_specs(args.param_spec) if args.param_spec else default_parameter_specs()
    combined = load_shards(args.input_dir, args.nsim_total)
    score_methods = tuple(item.strip() for item in args.score_methods.split(",") if item.strip())
    unknown_methods = sorted(set(score_methods).difference({"analytic", "finite_difference"}))
    if unknown_methods:
        raise ValueError(f"Unknown score methods {unknown_methods}; allowed analytic, finite_difference")
    fd_combined = load_fd_shards(args.input_dir) if "finite_difference" in score_methods else None
    analyses = parse_analysis_specs(args.analysis, combined["raw_probes"])
    cls_path = save_cls_ensemble(args.input_dir, combined)
    results = []
    for i, (name, probes) in enumerate(analyses.items()):
        results.append(run_one_analysis(
            name,
            probes,
            combined,
            fd_combined,
            args.fiducial_path,
            param_specs,
            args.input_dir,
            args.ngrid,
            args.posterior_samples,
            args.nboot,
            args.npseudo,
            args.seed + 1000 * i,
            args.theory_backend,
            score_methods,
        ))

    main_result = results[0]
    shutil.copyfile(main_result["samples_path"], args.input_dir / "map_sbi_compressed_samples.npz")
    diagnostics = {
        "runtime_sec": time.time() - t0,
        "input_dir": str(args.input_dir),
        "fiducial_path": str(args.fiducial_path),
        "nsim_total": int(args.nsim_total),
        "score_methods": list(score_methods),
        "fd_shard_count": 0 if fd_combined is None else len(fd_combined["shard_paths"]),
        "raw_probes": list(combined["raw_probes"]),
        "analysis_order": list(analyses.keys()),
        "cls_ensemble_path": str(cls_path),
        "main_samples_path": str(args.input_dir / "map_sbi_compressed_samples.npz"),
        "analyses": results,
    }
    save_json(args.input_dir / "map_sbi_diagnostics.json", diagnostics)
    print(json.dumps({
        "nsim_total": int(args.nsim_total),
        "analysis_order": list(analyses.keys()),
        "main_samples_path": str(args.input_dir / "map_sbi_compressed_samples.npz"),
    }, indent=2))


if __name__ == "__main__":
    main()
