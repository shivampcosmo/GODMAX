"""Large-scale halo clustering diagnostics for Backlight pasted-map validation."""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, Mapping, Tuple

import h5py
import healpy as hp
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
OUTPUT_DIR = THIS_DIR / "outputs"
DEFAULT_HALO_CATALOG = REPO_ROOT / "data" / "backlight" / "halo_catalog_Mlim_1e13_zlim_0.4_0.6.h5"
DEFAULT_OUTPUT = OUTPUT_DIR / "halo_clustering_large_scale_diagnostic.npz"


def ensure_repo_paths() -> None:
    for path in (REPO_ROOT / "src", REPO_ROOT / "notebooks" / "pasting", THIS_DIR, REPO_ROOT):
        spath = str(path)
        if spath not in sys.path:
            sys.path.insert(0, spath)


def load_theory_context() -> Mapping[str, object]:
    ensure_repo_paths()
    from fiducial_theory_datavector import build_theory_objects

    return build_theory_objects()


def load_halo_catalog(path: pathlib.Path | str = DEFAULT_HALO_CATALOG) -> Dict[str, np.ndarray]:
    path = pathlib.Path(path)
    with h5py.File(path, "r") as handle:
        return {key: handle[key][()] for key in ("ra", "dec", "z", "M200c", "vlos")}


def _bin_spectrum(ell_int: np.ndarray, cl_int: np.ndarray,
                  ell_bins: np.ndarray, delta_ell: np.ndarray) -> np.ndarray:
    out = np.full_like(ell_bins, np.nan, dtype=float)
    for i, (ell, dell) in enumerate(zip(ell_bins, delta_ell)):
        lo = ell - 0.5 * dell
        hi = ell + 0.5 * dell
        sel = (ell_int >= lo) & (ell_int < hi)
        if np.any(sel):
            out[i] = np.nanmean(cl_int[sel])
    return out


def _catalog_bias_interpolator(context: Mapping[str, object]) -> RegularGridInterpolator:
    pkz = context["pkz"]
    return RegularGridInterpolator(
        (np.asarray(pkz.z_array, dtype=float), np.log(np.asarray(pkz.M_array, dtype=float))),
        np.asarray(pkz.bias_Mz_mat, dtype=float),
        bounds_error=False,
        fill_value=None,
    )


def _catalog_bias_values(context: Mapping[str, object], catalog: Mapping[str, np.ndarray]) -> np.ndarray:
    interp_bias = _catalog_bias_interpolator(context)
    return interp_bias(np.column_stack([
        np.asarray(catalog["z"], dtype=float),
        np.log(np.asarray(catalog["M200c"], dtype=float)),
    ]))


def halo_map_cls(catalog: Mapping[str, np.ndarray], nside: int = 256,
                 lmax: int = 768) -> Dict[str, np.ndarray]:
    """Measure full-sky halo overdensity C_ell from the catalog."""

    npix = hp.nside2npix(nside)
    pix = hp.ang2pix(
        nside,
        np.asarray(catalog["ra"], dtype=float),
        np.asarray(catalog["dec"], dtype=float),
        lonlat=True,
    )
    counts = np.bincount(pix, minlength=npix).astype(float)
    delta = counts / np.mean(counts) - 1.0
    shot_noise = 4.0 * np.pi / float(len(catalog["z"]))
    cl_raw = hp.anafast(delta, lmax=lmax)
    return {
        "ell_int": np.arange(lmax + 1, dtype=float),
        "cl_raw": cl_raw,
        "cl_signal": cl_raw - shot_noise,
        "shot_noise": np.asarray(shot_noise),
        "counts": counts,
        "delta": delta,
    }


def catalog_window_and_bias(
    context: Mapping[str, object],
    catalog: Mapping[str, np.ndarray],
    z_edges: np.ndarray,
    smooth_sigma: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Build realized catalog dN/dz and catalog-mass-weighted bias curves."""

    z = np.asarray(catalog["z"], dtype=float)
    bias = _catalog_bias_values(context, catalog)
    z_centers = 0.5 * (z_edges[1:] + z_edges[:-1])
    counts, _ = np.histogram(z, bins=z_edges)
    bias_sum, _ = np.histogram(z, bins=z_edges, weights=bias)
    bias_mean = np.divide(
        bias_sum,
        counts,
        out=np.full_like(bias_sum, np.nan, dtype=float),
        where=counts > 0,
    )
    counts_smooth = gaussian_filter1d(counts.astype(float), smooth_sigma)

    cls = context["cls"]
    z_for = np.asarray(cls.z_array_for_Cls, dtype=float)
    window = interp1d(z_centers, counts_smooth, bounds_error=False, fill_value=0.0)(z_for)
    norm = np.trapz(window, z_for)
    if norm > 0:
        window = window / norm
    bias_for = interp1d(
        z_centers,
        np.nan_to_num(bias_mean, nan=np.nanmedian(bias_mean)),
        bounds_error=False,
        fill_value="extrapolate",
    )(z_for)
    return {
        "z_centers": z_centers,
        "counts": counts,
        "counts_smooth": counts_smooth,
        "bias_mean": bias_mean,
        "z_for": z_for,
        "window_for": window,
        "bias_for_catalog_mass_weighted": bias_for,
    }


def hmf_effective_bias_curves(
    context: Mapping[str, object],
    catalog: Mapping[str, np.ndarray],
    mass_min: float = 1.0e13,
) -> Dict[str, np.ndarray]:
    """Compute number-selected b_eff(z) using HMF and catalog-weighted masses."""

    pkz = context["pkz"]
    mass = np.asarray(pkz.M_array, dtype=float)
    z_grid = np.asarray(pkz.z_array, dtype=float)
    ln_mass = np.log(mass)
    hmf = np.asarray(pkz.hmf_Mz_mat, dtype=float)
    bias = np.asarray(pkz.bias_Mz_mat, dtype=float)
    cat_max = float(np.nanmax(catalog["M200c"]))

    out = {"z_grid": z_grid}
    for name, sel in {
        "hmf_gridmax": mass >= mass_min,
        "hmf_catalog_max": (mass >= mass_min) & (mass <= cat_max),
    }.items():
        nden = np.trapz(hmf[:, sel], x=ln_mass[sel], axis=1)
        bnum = np.trapz(hmf[:, sel] * bias[:, sel], x=ln_mass[sel], axis=1)
        out[name] = bnum / nden
        out[f"{name}_number_density"] = nden

    return out


def mass_bin_summary(
    context: Mapping[str, object],
    catalog: Mapping[str, np.ndarray],
    mass_edges_log10: np.ndarray | None = None,
    z_edges: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """Summarize catalog/HMF abundance and bias by mass bin."""

    ensure_repo_paths()
    from paste_backlight_utils import compute_dV_dz_per_sr

    if mass_edges_log10 is None:
        mass_edges_log10 = np.asarray([13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 15.0, 15.75])
    if z_edges is None:
        z_edges = np.linspace(0.4, 0.6, 9)

    pkz = context["pkz"]
    mass = np.asarray(pkz.M_array, dtype=float)
    z_grid = np.asarray(pkz.z_array, dtype=float)
    ln_mass = np.log(mass)
    hmf = np.asarray(pkz.hmf_Mz_mat, dtype=float)
    bias_grid = np.asarray(pkz.bias_Mz_mat, dtype=float)

    z_centers = 0.5 * (z_edges[1:] + z_edges[:-1])
    dvol = compute_dV_dz_per_sr(context["cosmo_jax"], z_centers) * 4.0 * np.pi * np.diff(z_edges)
    cat_z = np.asarray(catalog["z"], dtype=float)
    cat_logm = np.log10(np.asarray(catalog["M200c"], dtype=float))
    cat_bias = _catalog_bias_values(context, catalog)

    rows = []
    for lo, hi in zip(mass_edges_log10[:-1], mass_edges_log10[1:]):
        grid_sel = (np.log10(mass) >= lo) & (np.log10(mass) < hi)
        cat_sel_all = (cat_logm >= lo) & (cat_logm < hi)
        if np.count_nonzero(grid_sel) < 2:
            continue

        nden = np.trapz(hmf[:, grid_sel], x=ln_mass[grid_sel], axis=1)
        b_hmf = np.trapz(
            hmf[:, grid_sel] * bias_grid[:, grid_sel],
            x=ln_mass[grid_sel],
            axis=1,
        ) / nden
        expected = np.interp(z_centers, z_grid, nden) * dvol
        counts = []
        for zlo, zhi in zip(z_edges[:-1], z_edges[1:]):
            counts.append(np.count_nonzero(cat_sel_all & (cat_z >= zlo) & (cat_z < zhi)))
        counts = np.asarray(counts, dtype=float)
        abundance_ratio = counts / expected
        rows.append((
            lo,
            hi,
            np.count_nonzero(cat_sel_all),
            np.nanmedian(abundance_ratio),
            np.nanpercentile(abundance_ratio, 16.0),
            np.nanpercentile(abundance_ratio, 84.0),
            np.nanmean(np.interp(z_centers, z_grid, b_hmf)),
            np.nanmean(cat_bias[cat_sel_all]) if np.any(cat_sel_all) else np.nan,
        ))

    names = [
        "log10m_lo",
        "log10m_hi",
        "catalog_count",
        "abundance_ratio_median",
        "abundance_ratio_p16",
        "abundance_ratio_p84",
        "bias_hmf_mean",
        "bias_catalog_mean",
    ]
    return {name: np.asarray([row[i] for row in rows]) for i, name in enumerate(names)}


def godmax_limber_halo_cls(
    context: Mapping[str, object],
    ell_int: np.ndarray,
    window_for: np.ndarray,
    bias_for: np.ndarray,
) -> np.ndarray:
    """Limber halo C_ell using GODMAX/JAX-cosmo P_lin and supplied W(z), b(z)."""

    cls = context["cls"]
    pkz = context["pkz"]
    z_for = np.asarray(cls.z_array_for_Cls, dtype=float)
    chi = np.asarray(cls.chi_array_for_Cls, dtype=float)
    dchi_dz = np.asarray(cls.dchi_dz_array_for_Cls, dtype=float)
    logk = np.log(np.asarray(cls.kPk_array, dtype=float))
    z_grid = np.asarray(pkz.z_array, dtype=float)
    log_power = np.log(np.clip(np.asarray(pkz.plin_kz_mat, dtype=float), 1.0e-300, np.inf))
    interp_power = RegularGridInterpolator(
        (logk, z_grid),
        log_power,
        bounds_error=False,
        fill_value=None,
    )

    out = np.empty_like(ell_int, dtype=float)
    for i, ell in enumerate(ell_int):
        kval = (ell + 0.5) / np.clip(chi, 1.0, np.inf)
        pk = np.exp(interp_power(np.column_stack([np.log(kval), z_for])))
        integrand = window_for ** 2 * bias_for ** 2 * pk / (chi ** 2 * dchi_dz)
        out[i] = np.trapz(integrand, z_for)
    return out


def pyccl_limber_halo_cls(
    context: Mapping[str, object],
    ell_int: np.ndarray,
    z_for: np.ndarray,
    window_for: np.ndarray,
    bias_for: np.ndarray,
) -> Tuple[np.ndarray | None, str]:
    """Optional public-code comparison using PyCCL."""

    try:
        import pyccl as ccl
    except Exception as exc:  # pragma: no cover - depends on local environment
        return None, f"pyccl unavailable: {exc}"

    cosmo_params = context["sim_params_dict"]["cosmo"]
    try:
        cosmo = ccl.Cosmology(
            Omega_c=cosmo_params["Om0"] - cosmo_params["Ob0"],
            Omega_b=cosmo_params["Ob0"],
            h=cosmo_params["H0"] / 100.0,
            sigma8=cosmo_params["sigma8"],
            n_s=cosmo_params["ns"],
            w0=cosmo_params["w0"],
            transfer_function="eisenstein_hu",
            matter_power_spectrum="linear",
        )
        tracer = ccl.NumberCountsTracer(
            cosmo,
            has_rsd=False,
            dndz=(z_for, window_for),
            bias=(z_for, bias_for),
        )
        valid = ell_int >= 2
        out = np.full_like(ell_int, np.nan, dtype=float)
        out[valid] = ccl.angular_cl(cosmo, tracer, tracer, ell_int[valid])
        return out, "pyccl angular_cl completed"
    except Exception as exc:  # pragma: no cover - depends on PyCCL version
        return None, f"pyccl failed: {exc}"


def pyccl_nonlimber_halo_cls_binned(
    context: Mapping[str, object],
    ell_bins: np.ndarray,
    z_for: np.ndarray,
    window_for: np.ndarray,
    bias_for: np.ndarray,
    l_limber: float = 2000.0,
    fkem_Nchi: int = 512,
) -> Tuple[np.ndarray | None, str]:
    """Optional public-code FKEM non-Limber comparison using PyCCL."""

    try:
        import pyccl as ccl
    except Exception as exc:  # pragma: no cover - depends on local environment
        return None, f"pyccl unavailable: {exc}"

    cosmo_params = context["sim_params_dict"]["cosmo"]
    try:
        cosmo = ccl.Cosmology(
            Omega_c=cosmo_params["Om0"] - cosmo_params["Ob0"],
            Omega_b=cosmo_params["Ob0"],
            h=cosmo_params["H0"] / 100.0,
            sigma8=cosmo_params["sigma8"],
            n_s=cosmo_params["ns"],
            w0=cosmo_params["w0"],
            transfer_function="eisenstein_hu",
            matter_power_spectrum="linear",
        )
        tracer = ccl.NumberCountsTracer(
            cosmo,
            has_rsd=False,
            dndz=(z_for, window_for),
            bias=(z_for, bias_for),
        )
        valid = ell_bins >= 2
        out = np.full_like(ell_bins, np.nan, dtype=float)
        out[valid] = ccl.angular_cl(
            cosmo,
            tracer,
            tracer,
            ell_bins[valid],
            l_limber=l_limber,
            non_limber_integration_method="FKEM",
            fkem_Nchi=fkem_Nchi,
        )
        return out, "pyccl FKEM non-Limber angular_cl completed"
    except Exception as exc:  # pragma: no cover - depends on PyCCL version
        return None, f"pyccl non-Limber failed: {exc}"


def run_diagnostic(
    halo_catalog: pathlib.Path | str = DEFAULT_HALO_CATALOG,
    output_path: pathlib.Path | str = DEFAULT_OUTPUT,
    nside: int = 256,
    lmax: int = 768,
) -> Dict[str, object]:
    """Run and save the halo-clustering diagnostic."""

    context = load_theory_context()
    catalog = load_halo_catalog(halo_catalog)
    halo_cls = halo_map_cls(catalog, nside=nside, lmax=lmax)
    z_edges = np.linspace(0.4, 0.6, 81)
    window_bias = catalog_window_and_bias(context, catalog, z_edges=z_edges)
    hmf_bias = hmf_effective_bias_curves(context, catalog)

    cls = context["cls"]
    ell_bins = np.asarray(cls.ell_array, dtype=float)
    delta_ell = np.asarray(context["analysis_dict"]["dl_array_survey"], dtype=float)
    ell_int = halo_cls["ell_int"]
    map_binned = _bin_spectrum(ell_int, halo_cls["cl_signal"], ell_bins, delta_ell)

    z_for = window_bias["z_for"]
    window_for = window_bias["window_for"]
    bias_catalog_for = window_bias["bias_for_catalog_mass_weighted"]
    bias_grid = hmf_bias["z_grid"]

    theory_curves = {}
    for name in ("hmf_gridmax", "hmf_catalog_max"):
        bias_for = np.interp(z_for, bias_grid, hmf_bias[name])
        theory_curves[name] = godmax_limber_halo_cls(context, ell_int, window_for, bias_for)
    theory_curves["catalog_mass_weighted"] = godmax_limber_halo_cls(
        context,
        ell_int,
        window_for,
        bias_catalog_for,
    )

    pyccl_cl, pyccl_status = pyccl_limber_halo_cls(
        context,
        ell_int,
        z_for,
        window_for,
        bias_catalog_for,
    )
    if pyccl_cl is not None:
        theory_curves["pyccl_catalog_mass_weighted"] = pyccl_cl

    pyccl_limber_bincenter, pyccl_limber_bincenter_status = pyccl_limber_halo_cls(
        context,
        ell_bins,
        z_for,
        window_for,
        bias_catalog_for,
    )
    pyccl_nonlimber_binned, pyccl_nonlimber_status = pyccl_nonlimber_halo_cls_binned(
        context,
        ell_bins,
        z_for,
        window_for,
        bias_catalog_for,
    )

    theory_binned = {
        name: _bin_spectrum(ell_int, cl, ell_bins, delta_ell)
        for name, cl in theory_curves.items()
    }
    if pyccl_nonlimber_binned is not None:
        theory_binned["pyccl_nonlimber_catalog_mass_weighted"] = pyccl_nonlimber_binned
    if pyccl_limber_bincenter is not None:
        theory_binned["pyccl_limber_bincenter_catalog_mass_weighted"] = pyccl_limber_bincenter
    ratio_binned = {
        name: map_binned / theory
        for name, theory in theory_binned.items()
    }

    mass_summary = mass_bin_summary(context, catalog)
    metadata = {
        "halo_catalog": str(halo_catalog),
        "nside": int(nside),
        "lmax": int(lmax),
        "mass_min": float(np.nanmin(catalog["M200c"])),
        "mass_max": float(np.nanmax(catalog["M200c"])),
        "catalog_count": int(len(catalog["z"])),
        "pyccl_status": pyccl_status,
        "pyccl_limber_bincenter_status": pyccl_limber_bincenter_status,
        "pyccl_nonlimber_status": pyccl_nonlimber_status,
    }

    payload = {
        "ell_int": ell_int,
        "ell": ell_bins,
        "delta_ell": delta_ell,
        "halo_cl_raw": halo_cls["cl_raw"],
        "halo_cl_signal": halo_cls["cl_signal"],
        "halo_cl_signal_binned": map_binned,
        "shot_noise": halo_cls["shot_noise"],
        "z_for": z_for,
        "catalog_window_for": window_for,
        "catalog_bias_for": bias_catalog_for,
        "hmf_z_grid": bias_grid,
        "metadata_json": np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
    }
    for name, arr in hmf_bias.items():
        payload[f"bias_{name}"] = np.asarray(arr)
    for name, arr in mass_summary.items():
        payload[f"mass_summary_{name}"] = np.asarray(arr)
    for name, arr in theory_curves.items():
        payload[f"theory_{name}"] = arr
    for name, arr in theory_binned.items():
        payload[f"theory_{name}_binned"] = arr
    for name, arr in ratio_binned.items():
        payload[f"ratio_{name}_binned"] = arr

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    return {
        "output_path": output_path,
        "metadata": metadata,
        "mass_summary": mass_summary,
        "ratio_binned": ratio_binned,
    }


def load_diagnostic(path: pathlib.Path | str = DEFAULT_OUTPUT) -> Dict[str, object]:
    path = pathlib.Path(path)
    data = np.load(path, allow_pickle=True)
    return {
        "path": path,
        "metadata": json.loads(str(data["metadata_json"])),
        "arrays": {key: data[key] for key in data.files if key != "metadata_json"},
    }


if __name__ == "__main__":
    result = run_diagnostic()
    print(f"Saved halo clustering diagnostic to {result['output_path']}")
    print(json.dumps(result["metadata"], indent=2, sort_keys=True))
