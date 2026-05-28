"""Utilities for pasted-map score-compressed likelihood validation."""

from __future__ import annotations

import json
import pathlib
import pickle as pk
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from fiducial_theory_datavector import build_theory_objects, ensure_repo_paths, load_validation_product
from map_npe_utils import (
    DEFAULT_HALO_CATALOG,
    MeasurementConfig,
    add_survey_noise_maps,
    common_band_mask,
    generate_pasted_map_product,
    load_halo_catalog,
    measure_binned_cls,
)
from theory_sbi_utils import (
    DEFAULT_FIDUCIAL_PATH,
    ParameterSpec,
    default_parameter_specs,
    fiducial_theta,
    make_inference_theory_vector_function,
    parse_probe_list,
    prior_bounds,
    selected_product_arrays,
)


TARGET_PROBES = ("gg", "gy", "gtau", "gkappa")


@dataclass(frozen=True)
class StatisticBlock:
    name: str
    probes: tuple[str, ...]
    labels: tuple[str, ...]
    mu0: np.ndarray
    jacobian: np.ndarray
    covariance_model: np.ndarray
    precision_model: np.ndarray
    fisher: np.ndarray
    fisher_inv: np.ndarray
    fisher_transform: np.ndarray
    compression_theta: np.ndarray
    compression_u: np.ndarray
    selection_indices: np.ndarray
    ell_indices: np.ndarray


def save_json(path: pathlib.Path | str, payload: Mapping[str, object]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_analysis_specs(text: str | Sequence[str], available_probes: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Parse analysis specs like ``all:gg,gy`` or ``gg_only=gg``."""

    if isinstance(text, str):
        items = [item.strip() for item in text.split(";") if item.strip()]
    else:
        items = [str(item).strip() for item in text if str(item).strip()]
    out: dict[str, tuple[str, ...]] = {}
    available = set(parse_probe_list(available_probes))
    for item in items:
        if ":" in item:
            name, probes_text = item.split(":", 1)
        elif "=" in item:
            name, probes_text = item.split("=", 1)
        else:
            name, probes_text = item, item
        probes = parse_probe_list(probes_text)
        missing = sorted(set(probes).difference(available))
        if missing:
            raise ValueError(f"Analysis {name!r} asks for unavailable probes {missing}; available={sorted(available)}")
        out[name.strip()] = probes
    if not out:
        raise ValueError("At least one analysis spec is required")
    return out


def supported_ell_range(
    fiducial_path: pathlib.Path | str,
    nside: int,
    ell_min: float | None = None,
    ell_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    theory = load_validation_product(fiducial_path)
    ell = np.asarray(theory["ell"], dtype=float)
    delta_ell = np.asarray(theory["delta_ell"], dtype=float)
    lmax = min(3 * int(nside) - 1, int(np.nanmax(ell + 0.5 * delta_ell)))
    ok = np.ceil(ell + 0.5 * delta_ell).astype(int) <= lmax + 1
    if ell_min is not None:
        ok &= ell >= float(ell_min)
    if ell_max is not None:
        ok &= ell <= float(ell_max)
    if not np.any(ok):
        raise ValueError(f"No ell bins survive nside={nside}, ell_min={ell_min}, ell_max={ell_max}")
    return ell[ok], delta_ell[ok]


def raw_block_indices(raw_probes: Sequence[str], selected_probes: Sequence[str], nell: int) -> np.ndarray:
    raw = tuple(raw_probes)
    idx: list[int] = []
    for probe in selected_probes:
        block = raw.index(probe)
        idx.extend(range(block * nell, (block + 1) * nell))
    return np.asarray(idx, dtype=int)


def stable_inverse(cov: np.ndarray, jitter_fraction: float = 1.0e-10) -> tuple[np.ndarray, float]:
    cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    jitter = 0.0
    try:
        return np.linalg.inv(cov), jitter
    except np.linalg.LinAlgError:
        eig_min = float(np.min(np.linalg.eigvalsh(cov)))
        floor = jitter_fraction * max(float(np.median(np.diag(cov))), 1.0e-300)
        jitter = max(floor, -eig_min + floor)
        return np.linalg.inv(cov + np.eye(cov.shape[0]) * jitter), jitter


def stable_cholesky(cov: np.ndarray, jitter_fraction: float = 1.0e-10) -> tuple[np.ndarray, float]:
    cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    jitter = 0.0
    try:
        return np.linalg.cholesky(cov), jitter
    except np.linalg.LinAlgError:
        eig_min = float(np.min(np.linalg.eigvalsh(cov)))
        floor = jitter_fraction * max(float(np.median(np.diag(cov))), 1.0e-300)
        jitter = max(floor, -eig_min + floor)
        return np.linalg.cholesky(cov + np.eye(cov.shape[0]) * jitter), jitter


def shrinkage_covariance(samples: np.ndarray, shrinkage: float | None = None) -> tuple[np.ndarray, float]:
    """Return a diagonal-shrinkage covariance for simulation-derived score blocks."""

    x = np.asarray(samples, dtype=float)
    cov = np.cov(x.T, ddof=1)
    cov = np.atleast_2d(0.5 * (cov + cov.T))
    diag = np.diag(np.diag(cov))
    if shrinkage is None:
        n, p = x.shape
        shrinkage = float(np.clip((p + 1.0) / max(n - 1.0, 1.0), 0.02, 0.50))
    cov_shrink = (1.0 - shrinkage) * cov + shrinkage * diag
    return cov_shrink, float(shrinkage)


def build_two_point_score_block(
    fiducial_path: pathlib.Path | str,
    probes: Sequence[str],
    param_specs: Sequence[ParameterSpec],
    ell_min: float | None,
    ell_max: float | None,
    theory_backend: str = "linearized",
    jit_compile: bool = True,
) -> StatisticBlock:
    """Build analytical 2pt Fisher-score compression for selected probes."""

    selected = selected_product_arrays(fiducial_path, probes=probes, ell_min=ell_min, ell_max=ell_max)
    vector_fn, theory_info = make_inference_theory_vector_function(
        param_specs,
        selected["selection"],
        fiducial_vector=selected["data_vector"],
        backend=theory_backend,
        fiducial_offset=True,
        jit_compile=jit_compile,
    )
    if "jacobian" in theory_info:
        jac = np.asarray(theory_info["jacobian"], dtype=float)
    else:
        import jax
        import jax.numpy as jnp

        jac = np.asarray(jax.jacfwd(vector_fn)(jnp.asarray(fiducial_theta(param_specs))), dtype=float)
    precision = np.asarray(selected["precision"], dtype=float)
    fisher = jac.T @ precision @ jac
    fisher = 0.5 * (fisher + fisher.T)
    fisher_inv = np.linalg.pinv(fisher)
    fisher_transform, _ = stable_cholesky(fisher)
    compression_theta = fisher_inv @ jac.T @ precision
    compression_u = compression_theta.T @ fisher_transform
    return StatisticBlock(
        name="2pt",
        probes=tuple(probes),
        labels=tuple(selected["selection"].labels),
        mu0=np.asarray(selected["data_vector"], dtype=float),
        jacobian=jac,
        covariance_model=np.asarray(selected["cov"], dtype=float),
        precision_model=precision,
        fisher=fisher,
        fisher_inv=fisher_inv,
        fisher_transform=fisher_transform,
        compression_theta=compression_theta,
        compression_u=compression_u,
        selection_indices=np.asarray(selected["selection"].indices, dtype=int),
        ell_indices=np.asarray(selected["selection"].ell_indices, dtype=int),
    )


def build_simulation_score_block(
    name: str,
    fiducial_samples: np.ndarray,
    plus_samples_by_param: Sequence[np.ndarray],
    minus_samples_by_param: Sequence[np.ndarray],
    deltas: Sequence[float],
    labels: Sequence[str],
    probes: Sequence[str] = (),
    shrinkage: float | None = None,
) -> tuple[StatisticBlock, dict[str, object]]:
    """Build a future HOS score block from paired finite differences."""

    fid = np.asarray(fiducial_samples, dtype=float)
    mu0 = np.mean(fid, axis=0)
    cov, shrink = shrinkage_covariance(fid, shrinkage=shrinkage)
    precision, jitter = stable_inverse(cov)
    jac_cols = []
    for plus, minus, delta in zip(plus_samples_by_param, minus_samples_by_param, deltas):
        plus = np.asarray(plus, dtype=float)
        minus = np.asarray(minus, dtype=float)
        n = min(len(plus), len(minus))
        deriv = (plus[:n] - minus[:n]) / (2.0 * float(delta))
        jac_cols.append(np.mean(deriv, axis=0))
    jac = np.column_stack(jac_cols)
    fisher = jac.T @ precision @ jac
    fisher = 0.5 * (fisher + fisher.T)
    fisher_inv = np.linalg.pinv(fisher)
    fisher_transform, chol_jitter = stable_cholesky(fisher)
    compression_theta = fisher_inv @ jac.T @ precision
    compression_u = compression_theta.T @ fisher_transform
    block = StatisticBlock(
        name=name,
        probes=tuple(probes),
        labels=tuple(labels),
        mu0=mu0,
        jacobian=jac,
        covariance_model=cov,
        precision_model=precision,
        fisher=fisher,
        fisher_inv=fisher_inv,
        fisher_transform=fisher_transform,
        compression_theta=compression_theta,
        compression_u=compression_u,
        selection_indices=np.arange(len(mu0), dtype=int),
        ell_indices=np.array([], dtype=int),
    )
    diagnostics = {
        "shrinkage": shrink,
        "precision_jitter": jitter,
        "fisher_cholesky_jitter": chol_jitter,
        "n_fiducial": int(len(fid)),
        "n_derivative_pairs": [int(min(len(p), len(m))) for p, m in zip(plus_samples_by_param, minus_samples_by_param)],
    }
    return block, diagnostics


def simulation_score_stability_diagnostics(
    fiducial_samples: np.ndarray,
    plus_samples_by_param: Sequence[np.ndarray],
    minus_samples_by_param: Sequence[np.ndarray],
    deltas: Sequence[float],
    nboot: int = 64,
    seed: int = 0,
) -> dict[str, object]:
    """Low-cost stability diagnostics for simulation-derived HOS scores."""

    fid = np.asarray(fiducial_samples, dtype=float)
    cov, shrink = shrinkage_covariance(fid)
    precision, jitter = stable_inverse(cov)

    def jac_from_indices(indices_by_param: Sequence[np.ndarray]) -> np.ndarray:
        cols = []
        for ip, (plus, minus, delta) in enumerate(zip(plus_samples_by_param, minus_samples_by_param, deltas)):
            plus = np.asarray(plus, dtype=float)
            minus = np.asarray(minus, dtype=float)
            idx = indices_by_param[ip]
            deriv = (plus[idx] - minus[idx]) / (2.0 * float(delta))
            cols.append(np.mean(deriv, axis=0))
        return np.column_stack(cols)

    n_pair = [min(len(p), len(m)) for p, m in zip(plus_samples_by_param, minus_samples_by_param)]
    full_indices = [np.arange(n, dtype=int) for n in n_pair]
    jac_full = jac_from_indices(full_indices)
    fisher_full = 0.5 * (jac_full.T @ precision @ jac_full + (jac_full.T @ precision @ jac_full).T)
    eval_full, evec_full = np.linalg.eigh(fisher_full)

    split_angles = []
    for half in (0, 1):
        split_indices = []
        for n in n_pair:
            cut = n // 2
            split_indices.append(np.arange(0, cut, dtype=int) if half == 0 else np.arange(cut, n, dtype=int))
        if all(len(idx) > 1 for idx in split_indices):
            jac_split = jac_from_indices(split_indices)
            fisher_split = 0.5 * (jac_split.T @ precision @ jac_split + (jac_split.T @ precision @ jac_split).T)
            _, evec_split = np.linalg.eigh(fisher_split)
            dots = np.abs(np.sum(evec_full * evec_split, axis=0))
            split_angles.append(np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))).tolist())

    rng = np.random.default_rng(seed)
    boot_evals = []
    boot_angles = []
    for _ in range(int(nboot)):
        boot_indices = [rng.integers(0, n, size=n) for n in n_pair]
        jac_boot = jac_from_indices(boot_indices)
        fisher_boot = 0.5 * (jac_boot.T @ precision @ jac_boot + (jac_boot.T @ precision @ jac_boot).T)
        eval_boot, evec_boot = np.linalg.eigh(fisher_boot)
        dots = np.abs(np.sum(evec_full * evec_boot, axis=0))
        boot_evals.append(eval_boot)
        boot_angles.append(np.degrees(np.arccos(np.clip(dots, -1.0, 1.0))))

    boot_evals = np.asarray(boot_evals, dtype=float)
    boot_angles = np.asarray(boot_angles, dtype=float)
    return {
        "n_pair": [int(n) for n in n_pair],
        "fiducial_covariance_shrinkage": float(shrink),
        "fiducial_precision_jitter": float(jitter),
        "fisher_eigenvalues": eval_full.tolist(),
        "split_eigenvector_angles_deg": split_angles,
        "bootstrap_eigenvalue_p16_p50_p84": np.percentile(boot_evals, [16, 50, 84], axis=0).tolist() if len(boot_evals) else None,
        "bootstrap_eigenvector_angle_p16_p50_p84_deg": np.percentile(boot_angles, [16, 50, 84], axis=0).tolist() if len(boot_angles) else None,
    }


def compress_with_block(data_vectors: np.ndarray, block: StatisticBlock) -> tuple[np.ndarray, np.ndarray]:
    x = np.atleast_2d(np.asarray(data_vectors, dtype=float))
    resid = x - block.mu0[None, :]
    theta_summary = fiducial_theta(default_parameter_specs())[None, :] + resid @ block.compression_theta.T
    u_summary = resid @ block.compression_u
    return theta_summary, u_summary


def compress_with_block_at_theta0(
    data_vectors: np.ndarray,
    block: StatisticBlock,
    theta0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.atleast_2d(np.asarray(data_vectors, dtype=float))
    resid = x - block.mu0[None, :]
    theta_summary = theta0[None, :] + resid @ block.compression_theta.T
    u_summary = resid @ block.compression_u
    return theta_summary, u_summary


def loglike_compressed(
    theta: np.ndarray,
    theta0: np.ndarray,
    fisher_transform: np.ndarray,
    x_obs_u: np.ndarray,
    cov_u: np.ndarray,
    likelihood: str,
    nsim: int,
) -> float:
    model_u = (np.asarray(theta, dtype=float) - theta0) @ fisher_transform
    resid = x_obs_u - model_u
    inv_cov, jitter = stable_inverse(cov_u)
    cov_eval = cov_u + np.eye(cov_u.shape[0]) * jitter
    sign, logdet = np.linalg.slogdet(cov_eval)
    if sign <= 0:
        return -np.inf
    chi2 = float(resid @ inv_cov @ resid)
    if likelihood == "gaussian":
        return -0.5 * (chi2 + logdet)
    if likelihood == "student_t":
        return -0.5 * logdet - 0.5 * nsim * np.log1p(chi2 / max(nsim - 1.0, 1.0))
    raise ValueError("likelihood must be gaussian or student_t")


def grid_posterior_2d(
    prior_min: np.ndarray,
    prior_max: np.ndarray,
    theta0: np.ndarray,
    fisher_transform: np.ndarray,
    x_obs_u: np.ndarray,
    cov_u: np.ndarray,
    likelihood: str,
    nsim: int,
    ngrid: int,
) -> dict[str, np.ndarray | float]:
    theta0_grid = np.linspace(prior_min[0], prior_max[0], ngrid)
    theta1_grid = np.linspace(prior_min[1], prior_max[1], ngrid)
    grid0, grid1 = np.meshgrid(theta0_grid, theta1_grid, indexing="ij")
    theta_stack = np.stack([grid0, grid1], axis=-1)
    model_u = (theta_stack - theta0[None, None, :]) @ fisher_transform
    resid = x_obs_u[None, None, :] - model_u
    inv_cov, jitter = stable_inverse(cov_u)
    cov_eval = cov_u + np.eye(cov_u.shape[0]) * jitter
    sign, logdet = np.linalg.slogdet(cov_eval)
    if sign <= 0:
        raise RuntimeError("Compressed covariance is not positive definite")
    chi2 = np.einsum("...i,ij,...j->...", resid, inv_cov, resid)
    if likelihood == "gaussian":
        loglike = -0.5 * (chi2 + logdet)
    elif likelihood == "student_t":
        loglike = -0.5 * logdet - 0.5 * nsim * np.log1p(chi2 / max(nsim - 1.0, 1.0))
    else:
        raise ValueError("likelihood must be gaussian or student_t")
    logw = loglike - np.max(loglike)
    weight = np.exp(logw)
    norm = np.sum(weight)
    weight = weight / norm
    mean = np.array([np.sum(weight * grid0), np.sum(weight * grid1)])
    cov = np.array([
        [np.sum(weight * (grid0 - mean[0]) ** 2), np.sum(weight * (grid0 - mean[0]) * (grid1 - mean[1]))],
        [np.sum(weight * (grid0 - mean[0]) * (grid1 - mean[1])), np.sum(weight * (grid1 - mean[1]) ** 2)],
    ])
    return {
        "theta0_grid": theta0_grid,
        "theta1_grid": theta1_grid,
        "posterior_weight_grid": weight,
        "log_posterior_grid": np.log(np.clip(weight, 1.0e-300, np.inf)),
        "mean": mean,
        "cov": cov,
        "jitter": jitter,
    }


def sample_grid_posterior(grid: Mapping[str, np.ndarray], nsamples: int, rng: np.random.Generator) -> np.ndarray:
    weight = np.ravel(np.asarray(grid["posterior_weight_grid"], dtype=float))
    weight = weight / np.sum(weight)
    choice = rng.choice(weight.size, size=nsamples, replace=True, p=weight)
    i0, i1 = np.unravel_index(choice, np.asarray(grid["posterior_weight_grid"]).shape)
    return np.column_stack([np.asarray(grid["theta0_grid"])[i0], np.asarray(grid["theta1_grid"])[i1]])


def metropolis_posterior(
    prior_min: np.ndarray,
    prior_max: np.ndarray,
    theta0: np.ndarray,
    fisher_transform: np.ndarray,
    x_obs_u: np.ndarray,
    cov_u: np.ndarray,
    likelihood: str,
    nsim: int,
    nsamples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    ndim = len(theta0)
    fisher = fisher_transform @ fisher_transform.T
    proposal_cov = np.linalg.pinv(fisher) * (2.38**2 / max(ndim, 1))
    proposal_chol, _ = stable_cholesky(proposal_cov)
    theta = theta0.copy()
    logp = loglike_compressed(theta, theta0, fisher_transform, x_obs_u, cov_u, likelihood, nsim)
    burn = max(1000, nsamples // 5)
    draws = []
    accepted = 0
    total = burn + nsamples
    for i in range(total):
        prop = theta + proposal_chol @ rng.normal(size=ndim)
        if np.all(prop >= prior_min) and np.all(prop <= prior_max):
            logp_prop = loglike_compressed(prop, theta0, fisher_transform, x_obs_u, cov_u, likelihood, nsim)
            if np.log(rng.uniform()) < logp_prop - logp:
                theta = prop
                logp = logp_prop
                accepted += 1
        if i >= burn:
            draws.append(theta.copy())
    return np.asarray(draws), {"acceptance_fraction": accepted / float(total)}


def sample_compressed_posterior(
    prior_min: np.ndarray,
    prior_max: np.ndarray,
    theta0: np.ndarray,
    fisher_transform: np.ndarray,
    x_obs_u: np.ndarray,
    cov_u: np.ndarray,
    likelihood: str,
    nsim: int,
    ngrid: int,
    posterior_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(seed)
    if len(theta0) == 2:
        grid = grid_posterior_2d(prior_min, prior_max, theta0, fisher_transform, x_obs_u, cov_u, likelihood, nsim, ngrid)
        samples = sample_grid_posterior(grid, posterior_samples, rng)
        return samples, {"sampler": "grid", **grid}
    samples, diagnostics = metropolis_posterior(
        prior_min,
        prior_max,
        theta0,
        fisher_transform,
        x_obs_u,
        cov_u,
        likelihood,
        nsim,
        posterior_samples,
        seed,
    )
    return samples, {"sampler": "metropolis", **diagnostics}


def generate_component_map_product(
    theta: np.ndarray,
    param_specs: Sequence[ParameterSpec],
    nside: int,
    random_seed: int,
    get_signal_maps: bool,
    get_galaxies: bool,
    halo_catalog: pathlib.Path | str = DEFAULT_HALO_CATALOG,
    save_path: pathlib.Path | str | None = None,
    use_cache: bool = False,
) -> dict:
    """Generate signal-only, galaxy-only, or full pasted products."""

    if get_signal_maps and get_galaxies:
        return generate_pasted_map_product(
            theta,
            param_specs=param_specs,
            nside=nside,
            random_seed=random_seed,
            halo_catalog=halo_catalog,
            save_path=save_path,
            use_cached_signal_if_available=use_cache,
        )
    if use_cache and save_path is not None and pathlib.Path(save_path).exists():
        with open(save_path, "rb") as handle:
            return pk.load(handle)

    ensure_repo_paths()
    from get_sim_maps import setup_sim_map
    from paste_backlight_utils import generate_maps

    theta = np.asarray(theta, dtype=float)
    sim_overrides = {spec.name: float(theta[ip]) for ip, spec in enumerate(param_specs) if spec.target == "sim"}
    other_overrides = {spec.name: float(theta[ip]) for ip, spec in enumerate(param_specs) if spec.target == "other"}
    for ip, spec in enumerate(param_specs):
        if spec.target == "cosmo":
            sim_overrides[f"cosmo.{spec.name}"] = float(theta[ip])
    context = build_theory_objects(
        sim_param_overrides=sim_overrides,
        other_param_overrides=other_overrides,
        kappa_source="cmb",
    )
    halo_params_map = dict(context["halo_params_dict"])
    halo_params_map.update({
        "rmin": 0.005,
        "rmax": 10.0,
        "nr": 48,
        "zmin": 0.005,
        "zmax": 0.8,
        "nz": 52,
        "lg10_Mmin": 12.0,
        "lg10_Mmax": 15.75,
        "nM": 42,
    })
    mock_params_setup = {
        "nside": int(nside),
        "get_ymap": bool(get_signal_maps),
        "get_kSZmap": False,
        "get_taumap": bool(get_signal_maps),
        "get_kappamap": bool(get_signal_maps),
        "get_galmap": bool(get_galaxies),
        "smooth_profiles": True,
        "random_seed": int(random_seed),
    }
    map_profiles = setup_sim_map(
        context["sim_params_dict"],
        halo_params_map,
        context["analysis_dict"],
        context["other_params_dict"],
        mock_params_setup,
        Profiles_obj=context["profiles"],
    )
    ra, dec, z, mass, vlos = load_halo_catalog(halo_catalog)
    valid = (
        (ra > 2.0e-5)
        & (ra < 360.0 - 2.0e-5)
        & (dec > -90.0 + 2.0e-5)
        & (dec < 90.0 - 2.0e-5)
    )
    return generate_maps(
        ra[valid],
        dec[valid],
        z[valid],
        mass[valid],
        vlos[valid],
        map_profiles,
        mock_params_setup,
        int(nside),
        context["sim_params_dict"],
        halo_params_map,
        context["analysis_dict"],
        context["other_params_dict"],
        save_path=None if save_path is None else str(save_path),
        profile_timing=False,
    )


def merge_signal_and_galaxy_products(signal_product: Mapping[str, object], galaxy_product: Mapping[str, object]) -> dict:
    return {
        "map_ymap": np.asarray(signal_product["map_ymap"], dtype=float),
        "map_tau": np.asarray(signal_product["map_tau"], dtype=float),
        "map_kappa": np.asarray(signal_product["map_kappa"], dtype=float),
        "mock_gals_all": galaxy_product["mock_gals_all"],
    }


def block_to_payload(block: StatisticBlock) -> dict[str, np.ndarray]:
    return {
        "mu0": block.mu0,
        "jacobian": block.jacobian,
        "covariance_model": block.covariance_model,
        "precision_model": block.precision_model,
        "fisher": block.fisher,
        "fisher_inv": block.fisher_inv,
        "fisher_transform": block.fisher_transform,
        "compression_theta": block.compression_theta,
        "compression_u": block.compression_u,
        "selection_indices": block.selection_indices,
        "ell_indices": block.ell_indices,
        "labels": np.asarray(block.labels),
        "probes": np.asarray(block.probes),
    }
