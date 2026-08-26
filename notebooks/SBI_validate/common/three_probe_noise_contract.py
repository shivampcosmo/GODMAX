#!/usr/bin/env python3
"""Build and realize the frozen noisy three-probe mock/covariance contract.

This is deliberately standalone: it does not modify the map painter or GODMAX
source.  ``build`` constructs the matched signal theory, the effective y noise,
the exact NaMaster covariance, and fixed masked signal alms.  ``realize`` draws
one deterministic set of y/tau/kappa noise maps and measures its three galaxy
cross spectra using the saved workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any, Mapping

os.environ.setdefault("JAX_ENABLE_X64", "True")

import h5py
import healpy as hp
import numpy as np
from scipy.interpolate import RegularGridInterpolator


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
for _path in (THIS_DIR, REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "notebooks" / "xDESI"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from survey_defaults import (  # noqa: E402
    ARCMIN_TO_RAD,
    SO_KAPPA_MV_COLUMN,
    SO_KAPPA_NOISE_PATH,
    SO_TSZ_DEPROJ2_COLUMN,
    SO_TSZ_NOISE_PATH,
    _load_tabulated_noise_table,
    so_noise_provenance,
)
from three_probe_fast_paste import _catalog_attrs, prepare_fast_paste_godmax_config  # noqa: E402
from three_probe_mock_contract import sha256_array, sha256_file  # noqa: E402
from three_probe_noiseless_estimator import (  # noqa: E402
    build_galaxy_count_map,
    galaxy_overdensity,
    solve_common_c2_cap,
    subtract_weighted_mean,
)
from three_probe_noiseless_theory import build_noiseless_intrinsic_theory  # noqa: E402


NSIDE = 1024
LMAX = 2048
INFERENCE_EDGES = np.asarray(
    [80, 101, 127, 160, 201, 253, 319, 401, 505, 636, 801, 1008, 1268, 1597, 2010],
    dtype=np.int64,
)
SPECTRA = ("gy", "gkappa", "gtau")
FIELDS = ("g", "y", "kappa", "tau")
PAIR_FIELDS = {
    "gg": ("g", "g"), "gy": ("g", "y"), "gtau": ("g", "tau"),
    "gkappa": ("g", "kappa"), "yy": ("y", "y"),
    "ytau": ("y", "tau"), "ykappa": ("y", "kappa"),
    "tautau": ("tau", "tau"), "taukappa": ("tau", "kappa"),
    "kappakappa": ("kappa", "kappa"),
}
BASE_SEED = 2026082000
N_REALIZATIONS = 12
TAU_DEPTH_ARCMIN = 1.0e-5


def inference_bins(nmt_module, lmax: int = LMAX):
    ell = np.arange(int(lmax) + 1, dtype=np.int32)
    bpws = np.full(ell.size, -1, dtype=np.int32)
    for index, (left, right) in enumerate(zip(INFERENCE_EDGES[:-1], INFERENCE_EDGES[1:])):
        bpws[(ell >= left) & (ell < right)] = index
    return nmt_module.NmtBin(ells=ell, bpws=bpws, lmax=int(lmax))


def dense_tabulated_noise(path: pathlib.Path, column: int, lmax: int, *, zero_below: bool) -> np.ndarray:
    source_ell, source = _load_tabulated_noise_table(path, column, path.name)
    target = np.arange(int(lmax) + 1)
    result = np.zeros(target.size, dtype=np.float64)
    supported = (target >= source_ell[0]) & (target <= source_ell[-1])
    result[supported] = source[(target[supported] - int(source_ell[0])).astype(int)]
    if not zero_below and np.any(target < source_ell[0]):
        result[target < source_ell[0]] = source[0]
    if target[-1] > source_ell[-1]:
        raise ValueError(f"Requested ell={target[-1]} exceeds {path.name} support")
    result[:2] = 0.0
    if np.any(result[2:] < 0.0) or not np.all(np.isfinite(result)):
        raise ValueError(f"Invalid dense noise from {path}")
    return result


def project_all_slice_cls(product: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Project every g/y/e/m resolved pair using the exact saved map kernels."""

    ell = np.asarray(product["ell"], dtype=np.float64)
    k = np.asarray(product["k_hmpc"], dtype=np.float64)
    z = np.asarray(product["redshift"], dtype=np.float64)
    chi = np.asarray(product["chi_hmpc"], dtype=np.float64)
    dchi = np.asarray(product["dchi_dz_hmpc"], dtype=np.float64)
    nz = np.asarray(product["realized_nz_on_theory_grid"], dtype=np.float64)
    wk = np.asarray(product["cmb_efficiency_hmpc"], dtype=np.float64)
    tau = float(product["tau_constant_mpc3_h3"]) * (1.0 + z) ** 2
    kernels = {"g": nz / dchi, "y": 1.0 / (1.0 + z), "tau": tau, "kappa": wk}
    power_keys = {
        "gg": "Pgg_resolved", "gy": "Pgy_resolved", "gtau": "Pge_resolved",
        "gkappa": "Pgm_resolved", "yy": "Pyy_resolved", "ytau": "Pye_resolved",
        "ykappa": "Pym_resolved", "tautau": "Pee_resolved",
        "taukappa": "Pem_resolved", "kappakappa": "Pmm_resolved",
    }
    output: dict[str, np.ndarray] = {}
    for name, key in power_keys.items():
        values = np.asarray(product["powers"][key], dtype=np.float64)
        interp = RegularGridInterpolator((np.log(k), z), values, bounds_error=True)
        left, right = PAIR_FIELDS[name]
        result = np.empty(ell.size, dtype=np.float64)
        for index, multipole in enumerate(ell):
            kval = (multipole + 0.5) / chi
            p = interp(np.column_stack((np.log(kval), z)))
            result[index] = np.trapz(dchi * kernels[left] * kernels[right] * p / chi**2, z)
        output[name] = result
    return output


def build_all_sky_yy(
    config_path: pathlib.Path, map_path: pathlib.Path, ell: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Numerical Tinker all-z/all-mass yy used only as missing-sky variance."""

    import yaml
    from base_class import base_class
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles

    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    catalog_path = pathlib.Path(config["resolved_theory"]["catalog_path"])
    sim, halo, analysis, other = prepare_fast_paste_godmax_config(
        config, _catalog_attrs(catalog_path), config_path=config_path
    )
    halo.update({
        "lg10_Mmin": 10.0, "lg10_Mmax": 16.0, "nM": 64,
        "zmin": 0.01, "zmax": 3.0, "nz": 96, "nr": 48,
        "nk": 128, "kmin": 1.0e-4, "kmax": 150.0,
    })
    analysis.update({
        "zmin_for_Cls": 0.01, "zmax_for_Cls": 3.0,
        "symbolic_hmf": False, "symbolic_pk": False,
    })
    base = base_class(sim, halo, analysis, other)
    profiles = Profiles(sim, halo, analysis, other, base_class_obj=base)
    pkz = get_Pkz(sim, halo, analysis, other, Profiles_obj=profiles)
    mass = np.asarray(pkz.M_array, dtype=np.float64)
    hmf = np.asarray(pkz.hmf_Mz_mat, dtype=np.float64)
    bias = np.asarray(pkz.bias_Mz_mat, dtype=np.float64)
    uy = np.asarray(pkz.uk_y, dtype=np.float64)
    plin = np.asarray(pkz.plin_kz_mat, dtype=np.float64)
    logm = np.log(mass)
    p1 = np.trapz(uy * uy * hmf[None, :, :], x=logm, axis=-1)
    by = np.trapz(uy * hmf[None, :, :] * bias[None, :, :], x=logm, axis=-1)
    pyy = p1 + by * by * plin
    k = np.asarray(pkz.kPk_array, dtype=np.float64)
    z = np.asarray(pkz.z_array, dtype=np.float64)
    chi = np.asarray(pkz.chi_array, dtype=np.float64)
    dchi = np.asarray(pkz.dchi_dz_array, dtype=np.float64)
    interp = RegularGridInterpolator((np.log(k), z), pyy, bounds_error=True)
    cl = np.empty(len(ell), dtype=np.float64)
    for index, multipole in enumerate(np.asarray(ell, dtype=np.float64)):
        kval = (multipole + 0.5) / chi
        values = interp(np.column_stack((np.log(kval), z)))
        cl[index] = np.trapz(dchi * values / ((1.0 + z) ** 2 * chi**2), z)
    with h5py.File(map_path, "r") as handle:
        sigma = float(handle["kernels"].attrs["profile_smoothing_sigma_rad"])
    bell = np.exp(-0.5 * (np.asarray(ell) * sigma) ** 2)
    cl *= bell**2
    metadata = {
        "z_support": [0.01, 3.0], "log10_mass_support_hmsun": [10.0, 16.0],
        "nM": 64, "nz": 96, "nr": 48, "nk": 128,
        "hmf": "numerical_Tinker", "pk": "numerical", "profile": "GODMAX_uk_y",
        "smoothing": "same_nside1024_half_pixel_flat_sky_Bell_squared",
    }
    return cl, metadata


def total_observed_cls(slice_cls: Mapping[str, np.ndarray], noise: Mapping[str, np.ndarray],
                       galaxy_shot: float, pixel_window: np.ndarray) -> dict[str, np.ndarray]:
    total = {name: np.asarray(value, dtype=np.float64).copy() for name, value in slice_cls.items()}
    pg = np.asarray(pixel_window, dtype=np.float64)
    for name, (left, right) in PAIR_FIELDS.items():
        factor = pg ** ((left == "g") + (right == "g"))
        total[name] *= factor
    total["gg"] += float(galaxy_shot)
    total["yy"] += noise["y"]
    total["tautau"] += noise["tau"]
    total["kappakappa"] += noise["kappa"]
    return total


def build_gaussian_covariance(nmt, field, workspace, total_cls: Mapping[str, np.ndarray]) -> np.ndarray:
    cw = nmt.NmtCovarianceWorkspace.from_fields(field, field, field, field)
    nband = len(INFERENCE_EDGES) - 1
    result = np.empty((len(SPECTRA) * nband, len(SPECTRA) * nband), dtype=np.float64)

    def pair(a: str, b: str) -> np.ndarray:
        key = a + b
        reverse = b + a
        if key in total_cls:
            return total_cls[key]
        if reverse in total_cls:
            return total_cls[reverse]
        raise KeyError((a, b))

    for i, spec_a in enumerate(SPECTRA):
        a1, a2 = PAIR_FIELDS[spec_a]
        for j, spec_b in enumerate(SPECTRA):
            b1, b2 = PAIR_FIELDS[spec_b]
            block = nmt.gaussian_covariance(
                cw, 0, 0, 0, 0,
                pair(a1, b1)[None, :], pair(a1, b2)[None, :],
                pair(a2, b1)[None, :], pair(a2, b2)[None, :],
                workspace, workspace, coupled=False,
            )
            result[i*nband:(i+1)*nband, j*nband:(j+1)*nband] = np.asarray(block).reshape(nband, nband)
    result = 0.5 * (result + result.T)
    if not np.all(np.isfinite(result)) or np.any(np.diag(result) <= 0.0):
        raise RuntimeError("NaMaster returned an invalid covariance")
    np.linalg.cholesky(result)
    return result


def _read_map_centered(path: pathlib.Path, dataset: str, mask: np.ndarray) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        values = np.asarray(handle[dataset], dtype=np.float64)
    return subtract_weighted_mean(values, mask)


def build_contract(config_path: pathlib.Path, map_path: pathlib.Path, output: pathlib.Path,
                   workspace_path: pathlib.Path, n_projected_radius: int = 256) -> pathlib.Path:
    import pymaster as nmt

    product = build_noiseless_intrinsic_theory(
        config_path, map_path, n_projected_radius=n_projected_radius, ell_max=LMAX
    )
    slice_cls = project_all_slice_cls(product)
    all_yy, all_yy_meta = build_all_sky_yy(config_path, map_path, product["ell"])
    missing_yy = all_yy - slice_cls["yy"]
    if not np.all(np.isfinite(missing_yy)):
        raise RuntimeError("All-minus-slice yy contains non-finite values")
    if np.any(missing_yy[80:] < -1.0e-12 * np.maximum(all_yy[80:], 1.0e-300)):
        bad = np.where(missing_yy[80:] < 0.0)[0] + 80
        raise RuntimeError(f"All-minus-slice yy is materially negative at ell={bad[:10].tolist()}")
    missing_yy[np.abs(missing_yy) < 1.0e-15 * np.maximum(all_yy, 1.0e-300)] = 0.0

    ell = np.arange(LMAX + 1)
    y_inst = dense_tabulated_noise(SO_TSZ_NOISE_PATH, SO_TSZ_DEPROJ2_COLUMN, LMAX, zero_below=True)
    kappa_noise = dense_tabulated_noise(SO_KAPPA_NOISE_PATH, SO_KAPPA_MV_COLUMN, LMAX, zero_below=True)
    tau_noise = np.full(LMAX + 1, (TAU_DEPTH_ARCMIN * ARCMIN_TO_RAD) ** 2, dtype=np.float64)
    tau_noise[:2] = 0.0
    y_noise = y_inst + missing_yy
    y_noise[:2] = 0.0
    noise = {"y": y_noise, "tau": tau_noise, "kappa": kappa_noise}

    mask, mask_meta = solve_common_c2_cap(nside=NSIDE)
    counts, galaxy_report = build_galaxy_count_map(map_path, nside=NSIDE)
    delta_g, mean_count, removed = galaxy_overdensity(counts, mask)
    galaxy_shot = float(hp.nside2pixarea(NSIDE) / mean_count)
    bins = inference_bins(nmt)
    field = nmt.NmtField(mask, [delta_g], spin=0, n_iter=0, n_iter_mask=0,
                         lmax=LMAX, lmax_mask=LMAX, lite=True)
    workspace = nmt.NmtWorkspace.from_fields(field, field, bins, l_toeplitz=-1,
                                              l_exact=-1, dl_band=-1)
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace.write_to(str(workspace_path))
    window = np.asarray(workspace.get_bandpower_windows())[0, :, 0, :]
    pixwin = np.asarray(hp.pixwin(NSIDE, lmax=LMAX), dtype=np.float64)
    total_cls = total_observed_cls(slice_cls, noise, galaxy_shot, pixwin)
    covariance = build_gaussian_covariance(nmt, field, workspace, total_cls)

    fixed_alms: dict[str, np.ndarray] = {}
    fixed_alms["g"] = hp.map2alm(mask * delta_g, lmax=LMAX, iter=0)
    means = {"g_overdensity_removed": removed}
    for name, dataset in (("y", "maps/map_ymap"), ("tau", "maps/map_tau"),
                          ("kappa", "maps/map_kappa_cmb")):
        centered, mean = _read_map_centered(map_path, dataset, mask)
        means[name] = mean
        fixed_alms[name] = hp.map2alm(mask * centered, lmax=LMAX, iter=0)
    fixed_bp = {}
    for spec in SPECTRA:
        left, right = PAIR_FIELDS[spec]
        coupled = hp.alm2cl(fixed_alms[left], fixed_alms[right], lmax=LMAX)[None, :]
        fixed_bp[spec] = np.asarray(workspace.decouple_cell(coupled))[0]
    theory_bp = {spec: window @ (pixwin * slice_cls[spec]) for spec in SPECTRA}

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        handle.attrs.update({
            "schema_version": "sbi_three_probe_noise_contract_v1", "nside": NSIDE,
            "lmax": LMAX, "n_realizations": N_REALIZATIONS, "base_seed": BASE_SEED,
            "spectrum_order_json": json.dumps(SPECTRA),
            "vector_order": "spectrum-major gy[14],gkappa[14],gtau[14]",
            "map_path": str(map_path.resolve()), "map_sha256": sha256_file(map_path),
            "config_path": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
            "workspace_path": str(workspace_path.resolve()),
            "workspace_sha256": sha256_file(workspace_path),
            "script_sha256": sha256_file(pathlib.Path(__file__)),
            "tau_depth_arcmin": TAU_DEPTH_ARCMIN,
            "y_low_ell_policy": "Nell_instrument_zero_below_ell80; physical_missing_yy_retained",
            "noise_basis": "observed continuous map field; no profile Bell applied to noise",
            "mock_covariance_identity": "same dense noise datasets used by synalm and analytic covariance",
            "sample_covariance_policy": "diagnostic_only_rank_at_most_11",
            "all_yy_metadata_json": json.dumps(all_yy_meta, sort_keys=True),
            "so_noise_provenance_json": json.dumps(so_noise_provenance(), sort_keys=True),
            "mask_metadata_json": json.dumps(mask_meta, sort_keys=True),
            "galaxy_report_json": json.dumps(galaxy_report, sort_keys=True),
            "field_weighted_means_json": json.dumps(means, sort_keys=True),
            "galaxy_mean_count_per_masked_pixel": mean_count,
            "galaxy_shot_noise": galaxy_shot,
        })
        handle.create_dataset("ell", data=ell)
        handle.create_dataset("band_edges", data=INFERENCE_EDGES)
        handle.create_dataset("effective_ell", data=np.asarray(bins.get_effective_ells()))
        handle.create_dataset("window", data=window)
        handle.create_dataset("mask", data=mask.astype(np.float32), compression="lzf")
        handle.create_dataset("pixel_window_g", data=pixwin)
        for name, value in fixed_alms.items():
            handle.create_dataset(f"fixed_masked_alm/{name}", data=value)
        for name, value in slice_cls.items():
            handle.create_dataset(f"signal_cls/{name}", data=value)
        handle.create_dataset("noise_cls/y_instrument", data=y_inst)
        handle.create_dataset("noise_cls/y_missing_sky", data=missing_yy)
        handle.create_dataset("noise_cls/y_effective", data=y_noise)
        handle.create_dataset("noise_cls/tau", data=tau_noise)
        handle.create_dataset("noise_cls/kappa", data=kappa_noise)
        handle.create_dataset("diagnostics/yy_all", data=all_yy)
        for name, value in fixed_bp.items():
            handle.create_dataset(f"fixed_bandpowers/{name}", data=value)
        for name, value in theory_bp.items():
            handle.create_dataset(f"theory_bandpowers/{name}", data=value)
        handle.create_dataset("hmc/covariance", data=covariance)
        diagonal = np.sqrt(np.diag(covariance))
        handle.create_dataset("hmc/correlation", data=covariance / np.outer(diagonal, diagonal))
        handle.create_dataset("hmc/cholesky", data=np.linalg.cholesky(covariance))
        noise_hashes = {name: sha256_array(value) for name, value in noise.items()}
        handle.attrs["noise_dataset_sha256_json"] = json.dumps(noise_hashes, sort_keys=True)
    os.replace(tmp, output)
    return output


def _synalm_seeded(cl: np.ndarray, seed: int) -> np.ndarray:
    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        return hp.synalm(np.asarray(cl, dtype=np.float64), lmax=LMAX, new=True)
    finally:
        np.random.set_state(state)


def realize(contract: pathlib.Path, output: pathlib.Path, realization: int) -> pathlib.Path:
    import pymaster as nmt

    if realization < 0 or realization >= N_REALIZATIONS:
        raise ValueError(f"realization must be in [0,{N_REALIZATIONS})")
    with h5py.File(contract, "r") as handle:
        if str(handle.attrs["schema_version"]) != "sbi_three_probe_noise_contract_v1":
            raise ValueError("Noise contract schema mismatch")
        mask = np.asarray(handle["mask"], dtype=np.float64)
        fixed = {name: np.asarray(handle[f"fixed_masked_alm/{name}"]) for name in FIELDS}
        noise_cls = {name: np.asarray(handle[f"noise_cls/{'y_effective' if name == 'y' else name}"])
                     for name in ("y", "tau", "kappa")}
        expected_hashes = json.loads(str(handle.attrs["noise_dataset_sha256_json"]))
        for name, value in noise_cls.items():
            if sha256_array(value) != expected_hashes[name]:
                raise ValueError(f"Noise curve hash mismatch for {name}")
        workspace_path = pathlib.Path(str(handle.attrs["workspace_path"]))
        workspace_sha = str(handle.attrs["workspace_sha256"])
        map_path = str(handle.attrs["map_path"])
    if sha256_file(workspace_path) != workspace_sha:
        raise ValueError("NaMaster workspace hash mismatch")
    workspace = nmt.NmtWorkspace.from_file(str(workspace_path))
    seed = BASE_SEED + int(realization)
    subseeds = {"y": seed + 100_000, "tau": seed + 200_000, "kappa": seed + 300_000}
    noisy_alms = {}
    noisy_maps = {}
    noise_means = {}
    for name in ("y", "tau", "kappa"):
        noise_alm = _synalm_seeded(noise_cls[name], subseeds[name])
        noise_map = hp.alm2map(noise_alm, nside=NSIDE, lmax=LMAX)
        centered, mean = subtract_weighted_mean(noise_map, mask)
        masked_noise_alm = hp.map2alm(mask * centered, lmax=LMAX, iter=0)
        noisy_alms[name] = fixed[name] + masked_noise_alm
        noise_means[name] = mean
        # Saved noisy maps are full-sky signal plus the original full-sky draw.
        with h5py.File(map_path, "r") as source:
            dataset = {"y": "maps/map_ymap", "tau": "maps/map_tau",
                       "kappa": "maps/map_kappa_cmb"}[name]
            noisy_maps[name] = np.asarray(source[dataset], dtype=np.float32) + noise_map.astype(np.float32)
    bandpowers = {}
    coupled = {}
    for spec in SPECTRA:
        _, right = PAIR_FIELDS[spec]
        cl = hp.alm2cl(fixed["g"], noisy_alms[right], lmax=LMAX)
        coupled[spec] = cl
        bandpowers[spec] = np.asarray(workspace.decouple_cell(cl[None, :]))[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        handle.attrs.update({
            "schema_version": "sbi_three_probe_noisy_realization_v1",
            "contract_path": str(contract.resolve()), "contract_sha256": sha256_file(contract),
            "realization": int(realization), "base_seed": BASE_SEED,
            "field_subseeds_json": json.dumps(subseeds, sort_keys=True),
            "noise_weighted_means_json": json.dumps(noise_means, sort_keys=True),
            "spectrum_order_json": json.dumps(SPECTRA), "nside": NSIDE, "lmax": LMAX,
        })
        for name, value in noisy_maps.items():
            handle.create_dataset(f"maps/{name}", data=value, compression="lzf")
        for name, value in bandpowers.items():
            handle.create_dataset(f"bandpowers/{name}", data=value)
            handle.create_dataset(f"coupled_cls/{name}", data=coupled[name])
    os.replace(tmp, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--config", type=pathlib.Path, required=True)
    build.add_argument("--map", type=pathlib.Path, required=True)
    build.add_argument("--output", type=pathlib.Path, required=True)
    build.add_argument("--workspace", type=pathlib.Path, required=True)
    build.add_argument("--n-projected-radius", type=int, default=256)
    draw = sub.add_parser("realize")
    draw.add_argument("--contract", type=pathlib.Path, required=True)
    draw.add_argument("--output", type=pathlib.Path, required=True)
    draw.add_argument("--realization", type=int, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_contract(args.config, args.map, args.output, args.workspace,
                                n_projected_radius=args.n_projected_radius)
    else:
        result = realize(args.contract, args.output, args.realization)
    print(result)


if __name__ == "__main__":
    main()
