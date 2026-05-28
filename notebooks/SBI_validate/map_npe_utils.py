"""Utilities for active NPE on pasted Backlight maps.

The helpers here deliberately separate the expensive pasted-map forward model
from the NPE orchestration code.  The analytical theory product is used only to
define proposal guides and fixed multi-anchor score features; the posterior
training data are measured from pasted-map simulations.
"""

from __future__ import annotations

import json
import pathlib
import pickle as pk
from dataclasses import dataclass
from typing import Mapping, Sequence

import healpy as hp
import numpy as np
from scipy.interpolate import interp1d

from fiducial_theory_datavector import (
    REPO_ROOT,
    build_theory_objects,
    ensure_repo_paths,
    load_validation_product,
)
from theory_sbi_utils import (
    DEFAULT_FIDUCIAL_PATH,
    ParameterSpec,
    fiducial_theta,
    make_inference_theory_vector_function,
    parse_probe_list,
    prior_bounds,
    selected_product_arrays,
)


THIS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_HALO_CATALOG = (
    REPO_ROOT / "data" / "backlight" / "halo_catalog_Mlim_1e13_zlim_0.4_0.6.h5"
)
FIELD_BY_PROBE = {"gy": "y", "gtau": "tau", "gkappa": "kappa"}
MAP_KEY_BY_FIELD = {"y": "map_ymap", "tau": "map_tau", "kappa": "map_kappa"}


@dataclass(frozen=True)
class MeasurementConfig:
    nside: int = 512
    gal_zmin: float = 0.4
    gal_zmax: float = 0.6
    fsky: float = 0.34
    add_survey_noise: bool = True


@dataclass(frozen=True)
class GuideProduct:
    anchors: np.ndarray
    guide_mean: np.ndarray
    guide_cov: np.ndarray
    guide_weight_grid: np.ndarray
    theta0_grid: np.ndarray
    theta1_grid: np.ndarray


def save_json(path: pathlib.Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def split_rounds(text: str) -> list[int]:
    rounds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not rounds:
        raise ValueError("At least one round is required")
    return rounds


def common_band_mask(nside: int, fsky: float = 0.34) -> np.ndarray:
    """Return a deterministic full-RA equatorial band with the requested fsky."""

    npix = hp.nside2npix(nside)
    _, dec = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    dec_max = np.degrees(np.arcsin(np.clip(fsky, 0.0, 1.0)))
    return (np.abs(dec) <= dec_max).astype(float)


def stack_mock_galaxies(mock_gals_all: Mapping[object, object]) -> np.ndarray:
    chunks = []
    for chunk in mock_gals_all.values():
        arr = np.asarray(chunk)
        if arr.size:
            chunks.append(arr)
    if not chunks:
        return np.empty((0, 6), dtype=float)
    return np.vstack(chunks)


def make_galaxy_delta_map(
    mock_gals: np.ndarray,
    nside: int,
    mask: np.ndarray,
    zmin: float = 0.4,
    zmax: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    npix = hp.nside2npix(nside)
    if mock_gals.size == 0:
        raise ValueError("No mock galaxies were produced")
    ra = np.mod(mock_gals[:, 0], 360.0)
    dec = np.clip(mock_gals[:, 1], -90.0, 90.0)
    z = mock_gals[:, 2]
    sel = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & (z >= zmin) & (z <= zmax)
    pix = hp.ang2pix(nside, ra[sel], dec[sel], lonlat=True)
    counts = np.bincount(pix, minlength=npix).astype(float)
    mask_bool = np.asarray(mask) > 0
    ngal = int(np.sum(counts[mask_bool]))
    if ngal <= 0:
        raise ValueError("No selected galaxies inside the mask")
    mean_counts = ngal / np.count_nonzero(mask_bool)
    delta = np.zeros(npix, dtype=float)
    delta[mask_bool] = counts[mask_bool] / mean_counts - 1.0
    shot_noise = 4.0 * np.pi * float(np.mean(mask_bool)) / float(ngal)
    return delta, counts, shot_noise, ngal


def theory_bin_edges(ell: np.ndarray, delta_ell: np.ndarray, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    lows = np.floor(np.asarray(ell) - 0.5 * np.asarray(delta_ell)).astype(int)
    highs = np.ceil(np.asarray(ell) + 0.5 * np.asarray(delta_ell)).astype(int)
    lows = np.clip(lows, 2, lmax)
    highs = np.clip(highs, lows + 1, lmax + 1)
    for i in range(1, len(lows)):
        if lows[i] <= lows[i - 1]:
            lows[i] = highs[i - 1]
            highs[i] = max(highs[i], lows[i] + 1)
    ok = highs <= lmax + 1
    return lows[ok], highs[ok]


def bin_integer_spectrum(
    ell_int: np.ndarray,
    cl_int: np.ndarray,
    ell: np.ndarray,
    delta_ell: np.ndarray,
) -> np.ndarray:
    out = np.full(len(ell), np.nan, dtype=float)
    for i, (center, width) in enumerate(zip(ell, delta_ell)):
        lo = center - 0.5 * width
        hi = center + 0.5 * width
        sel = (ell_int >= lo) & (ell_int < hi) & np.isfinite(cl_int)
        if np.any(sel):
            out[i] = np.average(cl_int[sel], weights=2.0 * ell_int[sel] + 1.0)
    return out


def binned_pixwin(nside: int, ell: np.ndarray, delta_ell: np.ndarray, lmax: int) -> np.ndarray:
    ell_int = np.arange(lmax + 1, dtype=float)
    pixwin = hp.pixwin(nside, lmax=lmax)
    return np.clip(bin_integer_spectrum(ell_int, pixwin, ell, delta_ell), 1.0e-30, np.inf)


def binned_pixwin2(nside: int, ell: np.ndarray, delta_ell: np.ndarray, lmax: int) -> np.ndarray:
    ell_int = np.arange(lmax + 1, dtype=float)
    pixwin = hp.pixwin(nside, lmax=lmax)
    return np.clip(bin_integer_spectrum(ell_int, pixwin**2, ell, delta_ell), 1.0e-30, np.inf)


def interpolate_noise_to_integer_ell(
    theory: Mapping[str, object],
    field: str,
    ell_int: np.ndarray,
    pixwin: np.ndarray,
) -> np.ndarray:
    noise_dict = theory["noise"]
    key = field if field in noise_dict else f"noise_{field}"
    if key not in noise_dict:
        raise KeyError(
            f"Noise field {field!r} is missing from the theory product; "
            f"available noise keys are {sorted(noise_dict)}"
        )
    ell_th = np.asarray(theory["ell"], dtype=float)
    noise_th = np.asarray(noise_dict[key], dtype=float)
    noise_signal = interp1d(
        ell_th,
        noise_th,
        bounds_error=False,
        fill_value=(noise_th[0], noise_th[-1]),
    )(ell_int)
    raw = np.clip(noise_signal, 0.0, np.inf) * np.clip(pixwin, 0.0, np.inf) ** 2
    raw[:2] = 0.0
    return raw


def add_survey_noise_maps(
    maps: Mapping[str, np.ndarray],
    theory: Mapping[str, object],
    nside: int,
    lmax: int,
    seed: int,
    add_noise: bool = True,
) -> dict[str, np.ndarray]:
    out = {key: np.asarray(value, dtype=float).copy() for key, value in maps.items()}
    if not add_noise:
        return out
    rng = np.random.default_rng(seed)
    ell_int = np.arange(lmax + 1, dtype=float)
    pixwin = hp.pixwin(nside, lmax=lmax)
    for field in ("y", "tau", "kappa"):
        map_key = MAP_KEY_BY_FIELD[field]
        if map_key not in out:
            continue
        cl_raw = interpolate_noise_to_integer_ell(theory, field, ell_int, pixwin)
        noise_seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(noise_seed)
        noise_map = hp.synfast(cl_raw, nside=nside, lmax=lmax, new=True, verbose=False)
        out[map_key] = out[map_key] + np.asarray(noise_map, dtype=float)
    return out


def measure_binned_cls(
    map_data: Mapping[str, object],
    theory_path: pathlib.Path | str = DEFAULT_FIDUCIAL_PATH,
    probes: Sequence[str] = ("gg", "gy", "gtau", "gkappa"),
    config: MeasurementConfig = MeasurementConfig(),
    seed: int = 0,
) -> dict[str, object]:
    """Measure the selected map datavector with mask and pixel-window rules."""

    theory = load_validation_product(theory_path)
    probes = parse_probe_list(probes)
    ell_full = np.asarray(theory["ell"], dtype=float)
    delta_full = np.asarray(theory["delta_ell"], dtype=float)
    lmax = min(3 * config.nside - 1, int(np.nanmax(ell_full + 0.5 * delta_full)))
    supported = np.ceil(ell_full + 0.5 * delta_full).astype(int) <= lmax + 1
    if not np.any(supported):
        raise ValueError(f"No theory ell bins are measurable for nside={config.nside}, lmax={lmax}")
    ell = ell_full[supported]
    delta_ell = delta_full[supported]
    mask = common_band_mask(config.nside, config.fsky)
    fsky_eff = float(np.mean(mask > 0))
    pixwin2_bin = binned_pixwin2(config.nside, ell, delta_ell, lmax)
    maps_required = {}
    for probe in probes:
        if probe in FIELD_BY_PROBE:
            map_key = MAP_KEY_BY_FIELD[FIELD_BY_PROBE[probe]]
            maps_required[map_key] = np.asarray(map_data[map_key], dtype=float)
    noisy_maps = add_survey_noise_maps(
        maps_required,
        theory,
        config.nside,
        lmax,
        seed=seed,
        add_noise=config.add_survey_noise,
    )
    mock_gals = stack_mock_galaxies(map_data["mock_gals_all"])
    delta_g, counts_g, shot_noise, ngal = make_galaxy_delta_map(
        mock_gals,
        config.nside,
        mask,
        zmin=config.gal_zmin,
        zmax=config.gal_zmax,
    )

    try:
        import pymaster as nmt

        lows, highs = theory_bin_edges(ell, delta_ell, lmax)
        if len(lows) != len(ell):
            raise RuntimeError("Rounded NaMaster bins do not match the theory ell grid")
        bins = nmt.NmtBin.from_edges(lows, highs)
        field_g = nmt.NmtField(mask, [delta_g])

        def master(field_map):
            field_x = nmt.NmtField(mask, [field_map])
            return np.asarray(nmt.compute_full_master(field_g, field_x, bins)[0], dtype=float)

        cl_gg = (
            np.asarray(nmt.compute_full_master(field_g, field_g, bins)[0], dtype=float)
            - shot_noise
        ) / pixwin2_bin
        cl_cross = {}
        for spec in probes:
            if spec in FIELD_BY_PROBE:
                cl_cross[spec] = master(noisy_maps[MAP_KEY_BY_FIELD[FIELD_BY_PROBE[spec]]]) / pixwin2_bin
        estimator = "namaster"
    except Exception as exc:
        raw_gg = hp.anafast(mask * delta_g, lmax=lmax) / max(fsky_eff, 1.0e-30)
        cl_gg = (
            bin_integer_spectrum(np.arange(lmax + 1), raw_gg, ell, delta_ell)
            - shot_noise
        ) / pixwin2_bin
        cl_cross = {}
        for spec in probes:
            if spec not in FIELD_BY_PROBE:
                continue
            field = FIELD_BY_PROBE[spec]
            raw = hp.anafast(mask * delta_g, mask * noisy_maps[MAP_KEY_BY_FIELD[field]], lmax=lmax)
            raw = raw / max(fsky_eff, 1.0e-30)
            cl_cross[spec] = bin_integer_spectrum(np.arange(lmax + 1), raw, ell, delta_ell) / pixwin2_bin
        estimator = f"anafast_fallback:{exc!r}"

    cl_by_probe = {"gg": cl_gg, **cl_cross}
    data_vector = np.concatenate([np.asarray(cl_by_probe[p], dtype=float) for p in probes])
    return {
        "data_vector": data_vector,
        "cl_by_probe": cl_by_probe,
        "ell": ell,
        "delta_ell": delta_ell,
        "shot_noise_gg": shot_noise,
        "ngal": ngal,
        "fsky": fsky_eff,
        "mask": mask,
        "estimator": estimator,
    }


def load_halo_catalog(path: pathlib.Path | str = DEFAULT_HALO_CATALOG):
    ensure_repo_paths()
    from paste_backlight_utils import load_halo_catalog as _load

    return _load(str(path))


def generate_pasted_map_product(
    theta: np.ndarray,
    param_specs: Sequence[ParameterSpec],
    nside: int = 512,
    random_seed: int = 42,
    halo_catalog: pathlib.Path | str = DEFAULT_HALO_CATALOG,
    save_path: pathlib.Path | str | None = None,
    use_cached_signal_if_available: bool = False,
) -> dict:
    """Generate y/tau/kappa maps and a stochastic HOD galaxy catalog."""

    if use_cached_signal_if_available and save_path is not None and pathlib.Path(save_path).exists():
        with open(save_path, "rb") as handle:
            return pk.load(handle)

    ensure_repo_paths()
    from get_sim_maps import setup_sim_map
    from paste_backlight_utils import generate_maps

    theta = np.asarray(theta, dtype=float)
    sim_overrides = {
        spec.name: float(theta[ip])
        for ip, spec in enumerate(param_specs)
        if spec.target == "sim"
    }
    other_overrides = {
        spec.name: float(theta[ip])
        for ip, spec in enumerate(param_specs)
        if spec.target == "other"
    }
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
        "get_ymap": True,
        "get_kSZmap": True,
        "get_taumap": True,
        "get_kappamap": True,
        "get_galmap": True,
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
    data = generate_maps(
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
    return data


def guide_posterior_from_observation(
    observation_vector: np.ndarray,
    fiducial_path: pathlib.Path | str,
    probes: Sequence[str],
    param_specs: Sequence[ParameterSpec],
    ell_min: float | None = None,
    ell_max: float | None = None,
    backend: str = "linearized",
    ngrid: int = 240,
    broadening: float = 1.0,
) -> tuple[GuideProduct, Mapping[str, object]]:
    selected = selected_product_arrays(fiducial_path, probes=probes, ell_min=ell_min, ell_max=ell_max)
    selection = selected["selection"]
    vector_fn, theory_info = make_inference_theory_vector_function(
        param_specs,
        selection,
        fiducial_vector=selected["data_vector"],
        backend=backend,
        fiducial_offset=True,
        jit_compile=True,
    )
    low, high = prior_bounds(param_specs)
    t0 = np.linspace(low[0], high[0], ngrid)
    t1 = np.linspace(low[1], high[1], ngrid)
    grid0, grid1 = np.meshgrid(t0, t1, indexing="ij")
    theta_grid = np.stack([grid0.ravel(), grid1.ravel()], axis=1)
    if "jacobian" in theory_info:
        theta0 = fiducial_theta(param_specs)
        jac = np.asarray(theory_info["jacobian"], dtype=float)
        mu0 = np.asarray(theory_info["mu0"], dtype=float)
        mu_grid = mu0[None, :] + (theta_grid - theta0[None, :]) @ jac.T
    else:
        mu_grid = np.vstack([np.asarray(vector_fn(row), dtype=float) for row in theta_grid])
    resid = np.asarray(observation_vector, dtype=float)[None, :] - mu_grid
    white = np.linalg.solve(np.asarray(selected["chol"], dtype=float), resid.T).T
    logw = -0.5 * np.sum(white**2, axis=1)
    logw -= np.max(logw)
    weight = np.exp(logw).reshape(grid0.shape)
    weight = weight / np.sum(weight)
    mean = np.array([np.sum(weight * grid0), np.sum(weight * grid1)])
    cov = np.array([
        [np.sum(weight * (grid0 - mean[0]) ** 2), np.sum(weight * (grid0 - mean[0]) * (grid1 - mean[1]))],
        [np.sum(weight * (grid0 - mean[0]) * (grid1 - mean[1])), np.sum(weight * (grid1 - mean[1]) ** 2)],
    ])
    evals, evecs = np.linalg.eigh(cov)
    anchors = [mean]
    for ie in range(2):
        step = 2.0 * np.sqrt(max(evals[ie], 0.0)) * evecs[:, ie]
        anchors.append(np.clip(mean + step, low, high))
        anchors.append(np.clip(mean - step, low, high))
    anchors.append(0.5 * (low + high))
    anchors = np.unique(np.asarray(anchors, dtype=float), axis=0)
    if broadening != 1.0:
        cov = cov * float(broadening) ** 2
    guide = GuideProduct(
        anchors=anchors,
        guide_mean=mean,
        guide_cov=cov,
        guide_weight_grid=weight,
        theta0_grid=t0,
        theta1_grid=t1,
    )
    return guide, {"selected": selected, "vector_fn": vector_fn, "theory_info": theory_info}


def build_multi_anchor_compressor(
    anchors: np.ndarray,
    fiducial_path: pathlib.Path | str,
    probes: Sequence[str],
    param_specs: Sequence[ParameterSpec],
    ell_min: float | None = None,
    ell_max: float | None = None,
    backend: str = "linearized",
) -> dict[str, np.ndarray]:
    selected = selected_product_arrays(fiducial_path, probes=probes, ell_min=ell_min, ell_max=ell_max)
    selection = selected["selection"]
    vector_fn, theory_info = make_inference_theory_vector_function(
        param_specs,
        selection,
        fiducial_vector=selected["data_vector"],
        backend=backend,
        fiducial_offset=True,
        jit_compile=True,
    )
    precision = np.asarray(selected["precision"], dtype=float)
    mus = []
    jacobians = []
    for anchor in np.asarray(anchors, dtype=float):
        mus.append(np.asarray(vector_fn(anchor), dtype=float))
        if "jacobian" in theory_info:
            jacobians.append(np.asarray(theory_info["jacobian"], dtype=float))
        else:
            import jax
            import jax.numpy as jnp

            jacobians.append(np.asarray(jax.jacfwd(vector_fn)(jnp.asarray(anchor)), dtype=float))
    return {
        "anchors": np.asarray(anchors, dtype=float),
        "mu": np.asarray(mus, dtype=float),
        "jacobian": np.asarray(jacobians, dtype=float),
        "precision": precision,
        "data_vector_fiducial": np.asarray(selected["data_vector"], dtype=float),
        "cov": np.asarray(selected["cov"], dtype=float),
        "selection_indices": np.asarray(selection.indices),
        "selection_ell_indices": np.asarray(selection.ell_indices),
    }


def compress_datavectors(data_vectors: np.ndarray, compressor: Mapping[str, np.ndarray]) -> np.ndarray:
    x = np.atleast_2d(np.asarray(data_vectors, dtype=float))
    mus = np.asarray(compressor["mu"], dtype=float)
    jacs = np.asarray(compressor["jacobian"], dtype=float)
    precision = np.asarray(compressor["precision"], dtype=float)
    features = []
    for mu, jac in zip(mus, jacs):
        resid = x - mu[None, :]
        features.append(resid @ precision.T @ jac)
    return np.concatenate(features, axis=1)


def save_compressor(path: pathlib.Path | str, compressor: Mapping[str, np.ndarray]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in compressor.items()})


def load_compressor(path: pathlib.Path | str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def save_map_measurement(path: pathlib.Path | str, measurement: Mapping[str, object], metadata: Mapping[str, object]) -> None:
    payload = {
        "data_vector": np.asarray(measurement["data_vector"], dtype=float),
        "ell": np.asarray(measurement["ell"], dtype=float),
        "delta_ell": np.asarray(measurement["delta_ell"], dtype=float),
        "shot_noise_gg": np.asarray(measurement["shot_noise_gg"]),
        "ngal": np.asarray(measurement["ngal"]),
        "fsky": np.asarray(measurement["fsky"]),
        "metadata_json": np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
    }
    for spec, arr in measurement["cl_by_probe"].items():
        payload[f"cl_{spec}"] = np.asarray(arr, dtype=float)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
