"""Measure pasted-map Cls and compare them to the fiducial theory product."""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle as pk
import sys
from typing import Dict, Iterable, Mapping, Tuple

import healpy as hp
import numpy as np
from scipy.interpolate import interp1d

from fiducial_theory_datavector import (
    DEFAULT_OUTPUT as DEFAULT_THEORY_OUTPUT,
    REPO_ROOT,
    build_theory_objects,
    ensure_repo_paths,
    load_validation_product,
)


THIS_DIR = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "outputs"
DEFAULT_VALIDATION_OUTPUT = OUTPUT_DIR / "pasted_map_cls_validation_nside512.npz"
DEFAULT_MAP_PATH = (
    REPO_ROOT / "data" / "backlight" / "allmaps_sim_B12_nside512_split0of1_paste_testlowmass.pkl"
)
DEFAULT_HALO_CATALOG = (
    REPO_ROOT / "data" / "backlight" / "halo_catalog_Mlim_1e13_zlim_0.4_0.6.h5"
)


TARGET_MAP_KEYS = {
    "gy": "map_ymap",
    "gtau": "map_tau",
    "gkappa": "map_kappa",
}


def load_pickle(path: pathlib.Path | str) -> dict:
    with open(path, "rb") as handle:
        return pk.load(handle)


def map_product_is_compatible(data: Mapping[str, object], nside: int) -> bool:
    """Check whether an existing pickle has the required map structure."""

    npix = hp.nside2npix(nside)
    required = ("mock_gals_all", "map_ymap", "map_tau", "map_kappa")
    for key in required:
        if key not in data:
            return False
    for key in ("map_ymap", "map_tau", "map_kappa"):
        if len(np.asarray(data[key])) != npix:
            return False
    return True


def load_or_generate_maps(
    map_path: pathlib.Path | str = DEFAULT_MAP_PATH,
    halo_catalog: pathlib.Path | str = DEFAULT_HALO_CATALOG,
    nside: int = 512,
    regenerate: bool = False,
) -> Tuple[dict, pathlib.Path, bool]:
    """Load a pasted map product or regenerate the analytic-test map product."""

    map_path = pathlib.Path(map_path)
    halo_catalog = pathlib.Path(halo_catalog)
    if map_path.exists() and not regenerate:
        data = load_pickle(map_path)
        if map_product_is_compatible(data, nside):
            return data, map_path, False

    ensure_repo_paths()
    import jax
    import jax.numpy as jnp
    from get_sim_maps import setup_sim_map
    from paste_backlight_utils import generate_maps, load_halo_catalog

    context = build_theory_objects()
    sim_params_dict = context["sim_params_dict"]
    halo_params_dict = context["halo_params_dict"]
    analysis_dict = context["analysis_dict"]
    other_params_dict = context["other_params_dict"]
    profiles = context["profiles"]

    halo_params_map = halo_params_dict.copy()
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
        "nside": nside,
        "get_ymap": True,
        "get_kSZmap": True,
        "get_taumap": True,
        "get_kappamap": True,
        "get_galmap": True,
        "smooth_profiles": True,
    }
    map_profiles = setup_sim_map(
        sim_params_dict,
        halo_params_map,
        analysis_dict,
        other_params_dict,
        mock_params_setup,
        Profiles_obj=profiles,
    )

    ra, dec, z, mass, vlos = load_halo_catalog(str(halo_catalog))
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
        nside,
        sim_params_dict,
        halo_params_dict,
        analysis_dict,
        other_params_dict,
        save_path=str(map_path),
        profile_timing=True,
    )
    jax.clear_caches()
    return data, map_path, True


def stack_mock_galaxies(mock_gals_all: Mapping[object, object]) -> np.ndarray:
    """Stack chunked mock galaxy catalogs into a single array."""

    chunks = []
    for chunk in mock_gals_all.values():
        arr = np.asarray(chunk)
        if arr.size > 0:
            chunks.append(arr)
    if not chunks:
        return np.empty((0, 6), dtype=float)
    return np.vstack(chunks)


def make_galaxy_delta_map(mock_gals: np.ndarray, nside: int,
                          zmin: float = 0.4, zmax: float = 0.6,
                          mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Create a galaxy overdensity map and return counts plus shot noise."""

    npix = hp.nside2npix(nside)
    if mask is None:
        mask = np.ones(npix, dtype=bool)
    if mock_gals.size == 0:
        raise ValueError("No mock galaxies found in map product")

    ra = np.mod(mock_gals[:, 0], 360.0)
    dec = np.clip(mock_gals[:, 1], -90.0, 90.0)
    z = mock_gals[:, 2]
    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
    zsel = finite & (z >= zmin) & (z <= zmax)
    pix = hp.ang2pix(nside, ra[zsel], dec[zsel], lonlat=True)
    counts = np.bincount(pix, minlength=npix).astype(float)

    fsky = float(np.mean(mask))
    ngal = float(np.sum(counts[mask]))
    if ngal <= 0:
        raise ValueError("No selected mock galaxies inside the validation mask")

    mean_counts = ngal / np.sum(mask)
    delta = np.zeros(npix, dtype=float)
    delta[mask] = counts[mask] / mean_counts - 1.0
    shot_noise = 4.0 * np.pi * fsky / ngal
    return delta, counts, shot_noise


def bin_spectrum(ell_int: np.ndarray, cl_int: np.ndarray,
                 ell_centers: np.ndarray, delta_ell: np.ndarray) -> np.ndarray:
    """Bin integer-ell spectra into the theory log-ell bins."""

    ell_int = np.asarray(ell_int, dtype=float)
    cl_int = np.asarray(cl_int, dtype=float)
    out = np.full_like(ell_centers, np.nan, dtype=float)
    for i, (center, width) in enumerate(zip(ell_centers, delta_ell)):
        lo = center - 0.5 * width
        hi = center + 0.5 * width
        sel = (ell_int >= lo) & (ell_int < hi) & np.isfinite(cl_int)
        if np.any(sel):
            weights = 2.0 * ell_int[sel] + 1.0
            out[i] = np.average(cl_int[sel], weights=weights)
    return out


def interpolate_theory_to_integer_ell(theory: Mapping[str, object],
                                      ell_int: np.ndarray) -> Dict[str, np.ndarray]:
    """Interpolate target theory spectra to integer ell values."""

    ell_th = np.asarray(theory["ell"], dtype=float)
    cl_signal = theory["cl_signal"]
    out = {}
    for spec in ("gg", "gy", "gtau", "gkappa"):
        out[spec] = interp1d(
            ell_th,
            np.asarray(cl_signal[spec], dtype=float),
            bounds_error=False,
            fill_value=np.nan,
        )(ell_int)
    return out


def interpolate_all_theory_cls(theory: Mapping[str, object],
                               ell_int: np.ndarray) -> Dict[str, np.ndarray]:
    """Interpolate every saved theory Cl-like array to integer ell."""

    ell_th = np.asarray(theory["ell"], dtype=float)
    out = {}
    for key, value in theory["cl_signal"].items():
        arr = np.asarray(value, dtype=float)
        if arr.shape != ell_th.shape:
            continue
        out[key] = interp1d(
            ell_th,
            arr,
            bounds_error=False,
            fill_value=np.nan,
        )(ell_int)
    return out


def safe_npz_key(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in key)


def band_ratio_diagnostic(arrays: Mapping[str, np.ndarray],
                          ell: np.ndarray,
                          spec: str,
                          ell_min: float = 300.0,
                          ell_max: float = 800.0) -> Dict[str, float]:
    """Summarize map/theory agreement without modifying the theory target."""

    required = (f"{spec}_signal_map_binned", f"{spec}_signal_theory_binned")
    if any(key not in arrays for key in required):
        return {}

    map_cl = np.asarray(arrays[required[0]], dtype=float)
    theory_cl = np.asarray(arrays[required[1]], dtype=float)
    sel = (
        (ell >= ell_min)
        & (ell <= ell_max)
        & np.isfinite(map_cl)
        & np.isfinite(theory_cl)
        & (np.abs(theory_cl) > 1.0e-300)
    )
    if not np.any(sel):
        return {}

    ratio = map_cl[sel] / theory_cl[sel]
    return {
        "ell_min": float(ell_min),
        "ell_max": float(ell_max),
        "n_bins": int(np.count_nonzero(sel)),
        "median_map_over_theory": float(np.nanmedian(ratio)),
        "p16_map_over_theory": float(np.nanpercentile(ratio, 16.0)),
        "p84_map_over_theory": float(np.nanpercentile(ratio, 84.0)),
    }


def estimate_signal_space_sigma(
    ell_int: np.ndarray,
    delta_ell: np.ndarray,
    deconvolution_window: np.ndarray,
    fsky: float,
    raw_cross: np.ndarray,
    raw_gg: np.ndarray,
    raw_xx: np.ndarray,
    shot_noise: float = 0.0,
    is_gg: bool = False,
) -> np.ndarray:
    """Approximate deconvolved signal-space sigma from raw map spectra."""

    denom = np.clip((2.0 * ell_int + 1.0) * fsky, 1.0e-30, np.inf)
    if is_gg:
        var_raw = 2.0 * raw_gg ** 2 / denom
    else:
        var_raw = (raw_gg * raw_xx + raw_cross ** 2) / denom
    return (
        np.sqrt(np.clip(var_raw, 0.0, np.inf))
        / np.clip(deconvolution_window, 1.0e-30, np.inf)
    )


def measure_pasted_map_cls(
    theory_path: pathlib.Path | str = DEFAULT_THEORY_OUTPUT,
    map_path: pathlib.Path | str = DEFAULT_MAP_PATH,
    halo_catalog: pathlib.Path | str = DEFAULT_HALO_CATALOG,
    output_path: pathlib.Path | str = DEFAULT_VALIDATION_OUTPUT,
    nside: int = 512,
    gal_zmin: float = 0.4,
    gal_zmax: float = 0.6,
    regenerate_maps: bool = False,
) -> Dict[str, object]:
    """Measure pasted-map spectra and save comparison arrays."""

    theory = load_validation_product(theory_path)
    map_data, used_map_path, regenerated = load_or_generate_maps(
        map_path=map_path,
        halo_catalog=halo_catalog,
        nside=nside,
        regenerate=regenerate_maps,
    )

    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=bool)
    fsky = float(np.mean(mask))
    mock_gals = stack_mock_galaxies(map_data["mock_gals_all"])
    delta_g, counts_g, shot_noise = make_galaxy_delta_map(
        mock_gals,
        nside,
        zmin=gal_zmin,
        zmax=gal_zmax,
        mask=mask,
    )

    lmax = min(3 * nside - 1, int(np.nanmax(theory["ell"] + 0.5 * theory["delta_ell"])))
    ell_int = np.arange(lmax + 1, dtype=float)
    pixwin = hp.pixwin(nside, lmax=lmax)
    pixwin_one = np.clip(pixwin, 1.0e-30, np.inf)
    pixwin2 = np.clip(pixwin ** 2, 1.0e-30, np.inf)
    unity_window = np.ones_like(pixwin_one)
    theory_int = interpolate_theory_to_integer_ell(theory, ell_int)
    all_theory_int = interpolate_all_theory_cls(theory, ell_int)

    raw_gg = hp.anafast(delta_g, lmax=lmax) / fsky
    signal_gg = raw_gg - shot_noise
    raw_theory_gg = theory_int["gg"] + shot_noise
    signal_theory_gg = theory_int["gg"]

    spectra = {
        "gg": {
            "available": True,
            "raw_map": raw_gg,
            "raw_theory": raw_theory_gg,
            "signal_map": signal_gg,
            "signal_theory": signal_theory_gg,
            "deconvolution_window": unity_window,
            "map_key": "galaxy_counts",
        }
    }

    auto_raw = {
        "g": raw_gg,
        "y": hp.anafast(np.asarray(map_data["map_ymap"], dtype=float), lmax=lmax) / fsky,
        "tau": hp.anafast(np.asarray(map_data["map_tau"], dtype=float), lmax=lmax) / fsky,
    }
    if "map_kappa" in map_data:
        auto_raw["kappa"] = hp.anafast(np.asarray(map_data["map_kappa"], dtype=float), lmax=lmax) / fsky

    for spec, map_key in TARGET_MAP_KEYS.items():
        if map_key not in map_data:
            spectra[spec] = {
                "available": False,
                "reason": f"{map_key} is not present in the map product",
                "map_key": map_key,
            }
            continue
        field_map = np.asarray(map_data[map_key], dtype=float)
        raw = hp.anafast(delta_g, field_map, lmax=lmax) / fsky
        signal = raw / pixwin_one
        spectra[spec] = {
            "available": True,
            "raw_map": raw,
            "raw_theory": theory_int[spec] * pixwin_one,
            "signal_map": signal,
            "signal_theory": theory_int[spec],
            "signal_map_no_window": raw,
            "signal_map_two_window": raw / pixwin2,
            "raw_theory_no_window": theory_int[spec],
            "raw_theory_two_window": theory_int[spec] * pixwin2,
            "deconvolution_window": pixwin_one,
            "map_key": map_key,
        }

    if "map_rhom" in map_data:
        raw = hp.anafast(delta_g, np.asarray(map_data["map_rhom"], dtype=float), lmax=lmax) / fsky
        spectra["grhom"] = {
            "available": True,
            "raw_map": raw,
            "signal_map": raw / pixwin_one,
            "signal_map_no_window": raw,
            "signal_map_two_window": raw / pixwin2,
            "deconvolution_window": pixwin_one,
            "map_key": "map_rhom",
            "diagnostic_only": True,
            "reason": "map_rhom is not a lensing-weighted kappa map",
        }

    ell_th = np.asarray(theory["ell"], dtype=float)
    delta_ell = np.asarray(theory["delta_ell"], dtype=float)
    save_payload = {
        "ell_int": ell_int,
        "ell": ell_th,
        "delta_ell": delta_ell,
        "pixwin": pixwin,
        "cross_deconvolution_window": pixwin_one,
        "shot_noise_gg": np.asarray(shot_noise),
        "fsky_map": np.asarray(fsky),
        "ngal": np.asarray(np.sum(counts_g)),
    }
    binned = {}

    for key, arr in all_theory_int.items():
        npz_key = safe_npz_key(key)
        save_payload[f"theory_{npz_key}"] = np.asarray(arr, dtype=float)
        binned[f"theory_{npz_key}_binned"] = bin_spectrum(ell_int, arr, ell_th, delta_ell)

    array_keys_to_save = (
        "raw_map",
        "raw_theory",
        "raw_theory_no_window",
        "raw_theory_two_window",
        "signal_map",
        "signal_map_no_window",
        "signal_map_two_window",
        "signal_theory",
        "deconvolution_window",
    )
    for spec, values in spectra.items():
        if not values.get("available", False):
            continue
        for arr_key in array_keys_to_save:
            if arr_key in values:
                arr = np.asarray(values[arr_key], dtype=float)
                save_payload[f"{spec}_{arr_key}"] = arr
                binned[f"{spec}_{arr_key}_binned"] = bin_spectrum(ell_int, arr, ell_th, delta_ell)

        if spec == "gg":
            sigma = estimate_signal_space_sigma(
                ell_int,
                delta_ell,
                unity_window,
                fsky,
                raw_gg,
                raw_gg,
                raw_gg,
                shot_noise=shot_noise,
                is_gg=True,
            )
        elif spec in ("gy", "gtau", "gkappa"):
            field = {"gy": "y", "gtau": "tau", "gkappa": "kappa"}[spec]
            sigma = estimate_signal_space_sigma(
                ell_int,
                delta_ell,
                np.asarray(values["deconvolution_window"], dtype=float),
                fsky,
                np.asarray(values["raw_map"], dtype=float),
                raw_gg,
                auto_raw[field],
            )
        else:
            sigma = np.full_like(ell_int, np.nan, dtype=float)
        save_payload[f"{spec}_signal_sigma"] = sigma
        binned[f"{spec}_signal_sigma_binned"] = bin_spectrum(ell_int, sigma, ell_th, delta_ell)

    save_payload.update(binned)
    ratio_diagnostics = {}
    for spec in ("gg", "gy", "gtau", "gkappa"):
        map_key = f"{spec}_signal_map_binned"
        theory_key = f"{spec}_signal_theory_binned"
        if map_key in save_payload and theory_key in save_payload:
            ratio = save_payload[map_key] / np.where(
                np.abs(save_payload[theory_key]) > 1.0e-300,
                save_payload[theory_key],
                np.nan,
            )
            save_payload[f"{spec}_map_over_theory_binned"] = ratio
            ratio_diagnostics[spec] = band_ratio_diagnostic(save_payload, ell_th, spec)

    availability = {
        spec: {
            "available": bool(values.get("available", False)),
            "map_key": values.get("map_key"),
            "reason": values.get("reason", ""),
        }
        for spec, values in spectra.items()
    }
    metadata = {
        "theory_path": str(theory_path),
        "theory_mode": theory["metadata"].get("theory_mode"),
        "theory_corrections": theory["metadata"].get("corrections", {}),
        "map_path": str(used_map_path),
        "map_metadata": map_data.get("map_metadata", {}),
        "halo_catalog": str(halo_catalog),
        "map_regenerated": bool(regenerated),
        "nside": int(nside),
        "lmax": int(lmax),
        "map_paint_r200c_factor": 8.0,
        "gal_zmin": float(gal_zmin),
        "gal_zmax": float(gal_zmax),
        "ell_max_compare": 1000.0,
        "fsky_map": float(fsky),
        "ngal": int(np.sum(counts_g)),
        "shot_noise_gg": float(shot_noise),
        "availability": availability,
        "map_theory_ratio_diagnostics": ratio_diagnostics,
        "map_derived_calibrations_applied": False,
        "pixel_window_rule": "gg is treated as an unwindowed count-map auto after shot-noise subtraction. Galaxy-cross spectra use one HEALPix pixel window for the non-galaxy pasted field: raw theory = signal theory * hp.pixwin(nside), deconvolved signal map = raw map / hp.pixwin(nside). The older two-window convention is saved as *_two_window for diagnostics only.",
    }
    save_payload["metadata_json"] = np.asarray(json.dumps(metadata, indent=2, sort_keys=True))

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_payload)
    return {
        "output_path": output_path,
        "metadata": metadata,
        "spectra": spectra,
        "theory": theory,
    }


def load_map_validation_product(path: pathlib.Path | str = DEFAULT_VALIDATION_OUTPUT) -> Dict[str, object]:
    path = pathlib.Path(path)
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"]))
    return {
        "path": path,
        "metadata": metadata,
        "arrays": {key: data[key] for key in data.files if key != "metadata_json"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theory", default=str(DEFAULT_THEORY_OUTPUT))
    parser.add_argument("--map", default=str(DEFAULT_MAP_PATH))
    parser.add_argument("--halo-catalog", default=str(DEFAULT_HALO_CATALOG))
    parser.add_argument("--output", default=str(DEFAULT_VALIDATION_OUTPUT))
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--regenerate-maps", action="store_true")
    args = parser.parse_args()

    result = measure_pasted_map_cls(
        theory_path=args.theory,
        map_path=args.map,
        halo_catalog=args.halo_catalog,
        output_path=args.output,
        nside=args.nside,
        regenerate_maps=args.regenerate_maps,
    )
    print(f"Saved pasted-map Cl validation to {result['output_path']}")
    print(json.dumps(result["metadata"]["availability"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
