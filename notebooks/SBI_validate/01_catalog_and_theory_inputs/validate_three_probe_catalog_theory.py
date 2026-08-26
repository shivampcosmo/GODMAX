#!/usr/bin/env python3
"""Generate the fail-closed Gate-2 catalog/theory validation bundle.

This reads the large catalog in chunks, derives its lens n(z), evaluates the
numerical GODMAX Tinker-2010 HMF/bias at the exact c0000 cosmology, and writes
inspectable plots plus all plotted arrays.  The HMF comparison is explicitly
conditional on treating the particle-count mass proxy as M200c; it is not a
calibration of that provisional mass definition.
"""

from __future__ import annotations

# --- keep imports working from a theme subfolder: common/ holds the
# --- modules shared by more than one stage.
import pathlib as _pl, sys as _sys
_ROOT = _pl.Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "common"):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import argparse
import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Mapping

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLBACKEND", "Agg")

import asdf
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator

THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from three_probe_mock_contract import (  # noqa: E402
    canonical_cosmology,
    canonical_json_sha256,
    make_normalized_kernel,
    sha256_file,
    sha256_array,
    validate_catalog_contract,
)


C_KMS = 299792.458
THRESHOLDS = ("primary", "Nge125", "Nge150")
THRESHOLD_VALUES = {"primary": None, "Nge125": 125.0, "Nge150": 150.0}
COLORS = {"primary": "black", "Nge125": "#0072B2", "Nge150": "#D55E00"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def load_source_header(path: pathlib.Path) -> dict[str, Any]:
    with asdf.open(path, lazy_load=True, memmap=False) as af:
        header = af["header"]
        keys = (
            "H0", "Omega_M", "Omega_DE", "Omega_K", "CAMB_Omega_b",
            "CAMB_sigma8", "CAMB_ns", "w0", "ParticleMassHMsun", "hMpc",
        )
        result = {key: float(header[key]) for key in keys}
        result["wa"] = float(header.get("wa", 0.0))
    if result["wa"] != 0.0:
        raise ValueError(f"Current GODMAX contract requires wa=0, found {result['wa']}")
    if result["hMpc"] != 1.0:
        raise ValueError(f"Source coordinates are not marked h^-1 Mpc: hMpc={result['hMpc']}")
    return result


def build_provenance(config_path: pathlib.Path, params_path: pathlib.Path) -> dict[str, Any]:
    input_paths = {
        "validation_script": pathlib.Path(__file__).resolve(),
        "contract_module": (THIS_DIR / "three_probe_mock_contract.py").resolve(),
        "experiment_config": config_path.resolve(),
        "default_params": params_path.resolve(),
    }
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    src_files = sorted(path for path in SRC_DIR.rglob("*.py") if "__pycache__" not in path.parts)
    src_manifest = [
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)} for path in src_files
    ]
    return {
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in input_paths.items()
        },
        "git_commit": commit,
        "git_worktree_dirty": bool(status),
        "git_status_porcelain_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "godmax_src_tree": {
            "algorithm": "sha256(canonical-json(sorted repo-relative *.py path and file sha256))",
            "file_count": len(src_manifest),
            "sha256": canonical_json_sha256({"files": src_manifest}),
            "files": src_manifest,
        },
        "python_executable": sys.executable,
        "argv": [str(pathlib.Path(sys.argv[0]).resolve()), *sys.argv[1:]],
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
    }


def _identity_digest_update(digest: Any, arrays: list[np.ndarray]) -> None:
    dtype = np.dtype([("source_file_index", "<i4"), ("source_row_index", "<i8"), ("halo_timeslice_index", "<i8")])
    rows = np.empty(arrays[0].size, dtype=dtype)
    for name, array in zip(dtype.names, arrays):
        rows[name] = array
    digest.update(rows.tobytes())


def stream_catalog(
    path: pathlib.Path,
    z_min: float,
    z_max: float,
    mass_min: float,
    mass_max: float,
    *,
    nz_bins: int,
    robust_nz_bins: int,
    nside: int,
    chunk_rows: int,
) -> dict[str, Any]:
    z_edges = np.linspace(z_min, z_max, nz_bins + 1)
    robust_z_edges = np.linspace(z_min, z_max, robust_nz_bins + 1)
    mass_edges = np.logspace(np.log10(mass_min), np.log10(mass_max), 25)
    diagnostic_z_edges = np.linspace(z_min, z_max, 5)
    nz_counts = {key: np.zeros(nz_bins, dtype=np.int64) for key in THRESHOLDS}
    robust_counts = {key: np.zeros(robust_nz_bins, dtype=np.int64) for key in THRESHOLDS}
    mass_z_counts = {
        key: np.zeros((diagnostic_z_edges.size - 1, mass_edges.size - 1), dtype=np.int64)
        for key in THRESHOLDS
    }
    occupancy = np.zeros((diagnostic_z_edges.size - 1, hp.nside2npix(nside)), dtype=bool)
    identity_digests = {key: hashlib.sha256() for key in THRESHOLDS}
    total_counts = {key: 0 for key in THRESHOLDS}
    source_file_counts: dict[int, int] = {}
    order_ok = True
    previous_pair: tuple[int, int] | None = None
    selected_mass_min = np.inf
    selected_mass_max = -np.inf
    selected_z_min = np.inf
    selected_z_max = -np.inf

    with h5py.File(path, "r") as handle:
        nrows = int(handle["z"].shape[0])
        for start in range(0, nrows, chunk_rows):
            stop = min(start + chunk_rows, nrows)
            z = np.asarray(handle["z"][start:stop], dtype=np.float64)
            mass = np.asarray(handle["M200c_hMsun"][start:stop], dtype=np.float64)
            npart = np.asarray(handle["N_interp"][start:stop], dtype=np.float64)
            source_file = np.asarray(handle["source_file_index"][start:stop], dtype=np.int32)
            source_row = np.asarray(handle["source_row_index"][start:stop], dtype=np.int64)
            timeslice = np.asarray(handle["halo_timeslice_index"][start:stop], dtype=np.int64)

            if not (np.all(z > z_min) and np.all(z < z_max)):
                raise ValueError(f"Catalog chunk {start}:{stop} violates strict redshift support")
            if not (np.all(mass >= mass_min) and np.all(mass < mass_max)):
                raise ValueError(f"Catalog chunk {start}:{stop} violates mass support")
            if not np.array_equal(mass, np.asarray(handle["M_particle_proxy_hMsun"][start:stop])):
                raise ValueError("Painter mass alias differs from the particle-proxy mass")
            selected_mass_min = min(selected_mass_min, float(mass.min()))
            selected_mass_max = max(selected_mass_max, float(mass.max()))
            selected_z_min = min(selected_z_min, float(z.min()))
            selected_z_max = max(selected_z_max, float(z.max()))

            pairs = np.column_stack((source_file.astype(np.int64), source_row))
            if previous_pair is not None and tuple(pairs[0]) <= previous_pair:
                order_ok = False
            if np.any(pairs[1:, 0] < pairs[:-1, 0]) or np.any(
                (pairs[1:, 0] == pairs[:-1, 0]) & (pairs[1:, 1] <= pairs[:-1, 1])
            ):
                order_ok = False
            previous_pair = tuple(pairs[-1])
            unique_sources, chunk_source_counts = np.unique(source_file, return_counts=True)
            for source_index, count in zip(unique_sources, chunk_source_counts):
                source_file_counts[int(source_index)] = source_file_counts.get(int(source_index), 0) + int(count)

            masks = {
                "primary": np.ones(z.size, dtype=bool),
                "Nge125": npart >= 125.0,
                "Nge150": npart >= 150.0,
            }
            for key, mask in masks.items():
                total_counts[key] += int(mask.sum())
                nz_counts[key] += np.histogram(z[mask], bins=z_edges)[0]
                robust_counts[key] += np.histogram(z[mask], bins=robust_z_edges)[0]
                mass_z_counts[key] += np.histogram2d(z[mask], mass[mask], bins=(diagnostic_z_edges, mass_edges))[0].astype(np.int64)
                _identity_digest_update(
                    identity_digests[key], [source_file[mask], source_row[mask], timeslice[mask]]
                )

            pix = hp.ang2pix(
                nside,
                np.asarray(handle["ra_deg"][start:stop], dtype=np.float64),
                np.asarray(handle["dec_deg"][start:stop], dtype=np.float64),
                lonlat=True,
            )
            z_bin = np.searchsorted(diagnostic_z_edges, z, side="right") - 1
            for index in range(diagnostic_z_edges.size - 1):
                occupancy[index, np.unique(pix[z_bin == index])] = True

    if not order_ok:
        raise ValueError("Catalog source ordering is not strictly preserved")
    if not (total_counts["primary"] >= total_counts["Nge125"] >= total_counts["Nge150"]):
        raise ValueError("Particle-threshold counts are not monotone")
    for key in THRESHOLDS:
        if not np.array_equal(robust_counts[key].reshape(nz_bins, -1).sum(axis=1), nz_counts[key]):
            raise ValueError(f"128-bin and 256-bin {key} n(z) counts do not aggregate exactly")

    return {
        "z_edges": z_edges,
        "robust_z_edges": robust_z_edges,
        "nz_counts": nz_counts,
        "robust_nz_counts": robust_counts,
        "diagnostic_z_edges": diagnostic_z_edges,
        "mass_edges": mass_edges,
        "mass_z_counts": mass_z_counts,
        "counts": total_counts,
        "source_file_counts": source_file_counts,
        "identity_sha256": {key: value.hexdigest() for key, value in identity_digests.items()},
        "source_order_preserved": order_ok,
        "occupied_pixels_by_z": occupancy.sum(axis=1),
        "total_pixels": occupancy.shape[1],
        "full_sky": bool(np.all(occupancy)),
        "selected_extrema": {
            "z_min": selected_z_min,
            "z_max": selected_z_max,
            "mass_min_hmsun": selected_mass_min,
            "mass_max_hmsun": selected_mass_max,
        },
    }


def build_godmax_hmf_bias(
    params_path: pathlib.Path,
    cosmology: Mapping[str, Any],
    support: Any,
    diagnostic: Mapping[str, Any],
    *,
    robust: bool,
) -> dict[str, Any]:
    import jax.numpy as jnp
    from base_class import base_class, get_vmapped_func
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles

    with params_path.open() as handle:
        config = yaml.safe_load(handle)
    sim = copy.deepcopy(config["sim_params"])
    halo = copy.deepcopy(config["halo_params"])
    analysis = copy.deepcopy(config["analysis"])
    other = copy.deepcopy(config["other_params"])
    sim["cosmo"] = dict(cosmology)
    sim["init_power"] = True
    prefix = "robustness_" if robust else ""
    halo.update({
        "zmin": support.z_min,
        "zmax": support.z_max,
        "nz": int(diagnostic[f"{prefix}theory_nz"]),
        "lg10_Mmin": float(np.log10(support.mass_min_hmsun)),
        "lg10_Mmax": float(np.log10(support.mass_max_hmsun)),
        "nM": int(diagnostic[f"{prefix}theory_nM"]),
        "kmin": 1.0e-4,
        "kmax": 1.0e3,
        "nk": int(diagnostic[f"{prefix}theory_nk"]),
        "mdef_Delta": 200,
        "hmf_model": "T10",
    })
    analysis.update({
        "symbolic_hmf": False,
        "symbolic_pk": False,
        "nbar_gal_comoving_val": 5.0e-4,
        "zmin_for_Cls": support.z_min,
        "zmax_for_Cls": support.z_max,
        "nz_for_Cls": int(diagnostic[f"{prefix}theory_nz"]),
    })
    nz_stub = np.linspace(support.z_min, support.z_max, 16)
    analysis["nz_source_info_dict"] = {
        "nbins": 1, "z_array_source": nz_stub.tolist(), "nz0": np.ones(16).tolist()
    }
    analysis["nz_lens_info_dict"] = {
        "nbins_lens": 1,
        "z_edges_bins_lens": [[support.z_min, support.z_max]],
        "z_array_lens": nz_stub.tolist(),
        "nz0": np.ones(16).tolist(),
    }

    obj = Profiles.__new__(Profiles)
    base_class.__init__(obj, sim, halo, analysis, other)
    Profiles.setup_hmf(obj)
    Profiles.get_hmf(obj)
    bound_bias = get_Pkz.get_bias_Mz.__get__(obj, Profiles)
    bias = get_vmapped_func(bound_bias, 2)(jnp.arange(obj.nz), jnp.arange(obj.nM)).T
    effective = {key: (bool(value) if key == "flat" else float(value)) for key, value in obj.cosmo_params.items()}
    if effective != dict(cosmology):
        raise ValueError(f"Constructed GODMAX cosmology differs from catalog: {effective} != {cosmology}")
    if obj.hmf_model != "T10" or int(obj.mdef_Delta) != 200:
        raise ValueError("GODMAX HMF/bias model is not frozen to T10/200c")
    mass = np.asarray(obj.M_array, dtype=np.float64)
    redshift = np.asarray(obj.z_array, dtype=np.float64)
    endpoint_rtol = 8.0 * np.finfo(np.float64).eps
    if not (
        np.isclose(mass[0], support.mass_min_hmsun, rtol=endpoint_rtol, atol=0.0)
        and np.isclose(mass[-1], support.mass_max_hmsun, rtol=endpoint_rtol, atol=0.0)
    ):
        raise ValueError(
            "Constructed GODMAX mass grid does not have the contract endpoints "
            f"to floating-point precision: {mass[[0, -1]]}"
        )
    if redshift[0] != support.z_min or redshift[-1] != support.z_max:
        raise ValueError("Constructed GODMAX redshift grid does not have the exact contract endpoints")
    return {
        "mass_hmsun": mass,
        "z": redshift,
        "hmf_dndlnm_h3mpc3": np.asarray(obj.hmf_Mz_mat, dtype=np.float64),
        "bias": np.asarray(bias, dtype=np.float64),
        "rho_m0_h2msun_mpc3": float(obj.rhom_0),
        "effective_cosmology": effective,
        "grid": {"nM": int(obj.nM), "nz": int(obj.nz), "nk": int(obj.nk), "kmin": 1e-4, "kmax": 1e3},
    }


def expansion_e(z: np.ndarray | float, om0: float, w0: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return np.sqrt(om0 * (1.0 + z) ** 3 + (1.0 - om0) * (1.0 + z) ** (3.0 * (1.0 + w0)))


def chi_hmpc(z: float, om0: float, w0: float) -> float:
    value = quad(lambda zp: 1.0 / float(expansion_e(zp, om0, w0)), 0.0, z, epsabs=1e-10, epsrel=1e-10)[0]
    return (C_KMS / 100.0) * value


def dvolume_dz_fullsky(z: np.ndarray, om0: float, w0: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    chi = np.asarray([chi_hmpc(float(value), om0, w0) for value in z])
    return 4.0 * np.pi * chi**2 * (C_KMS / 100.0) / expansion_e(z, om0, w0)


def shell_volume(z_lo: float, z_hi: float, om0: float, w0: float) -> float:
    return 4.0 * np.pi / 3.0 * (chi_hmpc(z_hi, om0, w0) ** 3 - chi_hmpc(z_lo, om0, w0) ** 3)


def binned_theory_hmf(theory: Mapping[str, Any], z_edges: np.ndarray, mass_edges: np.ndarray, cosmology: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    z_model = np.asarray(theory["z"])
    ln_mass_model = np.log(np.asarray(theory["mass_hmsun"]))
    interp_hmf = RegularGridInterpolator((z_model, ln_mass_model), np.asarray(theory["hmf_dndlnm_h3mpc3"]))
    expected = np.empty((z_edges.size - 1, mass_edges.size - 1), dtype=np.float64)
    density = np.empty_like(expected)
    for iz, (zlo, zhi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        z_eval = np.linspace(zlo, zhi, 33)
        dv = dvolume_dz_fullsky(z_eval, float(cosmology["Om0"]), float(cosmology["w0"]))
        volume = shell_volume(zlo, zhi, float(cosmology["Om0"]), float(cosmology["w0"]))
        for im, (mlo, mhi) in enumerate(zip(mass_edges[:-1], mass_edges[1:])):
            lnm_eval = np.linspace(np.log(mlo), np.log(mhi), 17)
            zz, mm = np.meshgrid(z_eval, lnm_eval, indexing="ij")
            hmf = interp_hmf(np.column_stack((zz.ravel(), mm.ravel()))).reshape(zz.shape)
            per_z = np.trapz(hmf, lnm_eval, axis=1)
            expected[iz, im] = np.trapz(per_z * dv, z_eval)
            density[iz, im] = expected[iz, im] / (volume * (np.log(mhi) - np.log(mlo)))
    return expected, density


def bias_summaries(theory: Mapping[str, Any]) -> dict[str, np.ndarray]:
    mass = np.asarray(theory["mass_hmsun"])
    lnm = np.log(mass)
    hmf = np.asarray(theory["hmf_dndlnm_h3mpc3"])
    bias = np.asarray(theory["bias"])
    number_effective = np.trapz(hmf * bias, lnm, axis=1) / np.trapz(hmf, lnm, axis=1)
    raw_mass_integral = np.trapz(hmf * bias * mass[None, :], lnm, axis=1) / float(theory["rho_m0_h2msun_mpc3"])
    return {"number_weighted_bias": number_effective, "resolved_mass_weighted_bias_integral": raw_mass_integral}


def catalog_conditioned_bias(scan: Mapping[str, Any], theory: Mapping[str, Any]) -> dict[str, np.ndarray]:
    z_edges = np.asarray(scan["diagnostic_z_edges"])
    mass_edges = np.asarray(scan["mass_edges"])
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    mass_mid = np.sqrt(mass_edges[:-1] * mass_edges[1:])
    interp = RegularGridInterpolator((np.asarray(theory["z"]), np.log(np.asarray(theory["mass_hmsun"]))), np.asarray(theory["bias"]))
    zz, mm = np.meshgrid(z_mid, np.log(mass_mid), indexing="ij")
    bmid = interp(np.column_stack((zz.ravel(), mm.ravel()))).reshape(zz.shape)
    result: dict[str, np.ndarray] = {"z_mid": z_mid}
    for key in THRESHOLDS:
        counts = np.asarray(scan["mass_z_counts"][key], dtype=np.float64)
        result[key] = np.divide((counts * bmid).sum(axis=1), counts.sum(axis=1), out=np.full(z_mid.size, np.nan), where=counts.sum(axis=1) > 0)
    return result


def save_kernel(path: pathlib.Path, scan: Mapping[str, Any], cosmology: Mapping[str, Any], metadata: Mapping[str, Any], provenance: Mapping[str, Any], tolerance: float) -> dict[str, Any]:
    kernels: dict[str, Any] = {}
    with h5py.File(path, "w") as handle:
        handle.attrs["format_version"] = 1
        handle.attrs["catalog_file_sha256"] = metadata["catalog_file_sha256"]
        handle.attrs["catalog_row_identity_sha256"] = metadata["catalog_row_identity_sha256"]
        handle.attrs["cosmology_json"] = json.dumps(_jsonable(cosmology), sort_keys=True)
        handle.attrs["cosmology_sha256"] = canonical_json_sha256(cosmology)
        handle.attrs["producer_provenance_json"] = json.dumps(_jsonable(provenance), sort_keys=True)
        handle.attrs["producer_provenance_sha256"] = canonical_json_sha256(provenance)
        handle.attrs["z_min_exclusive"] = True
        handle.attrs["z_max_exclusive"] = True
        for key in THRESHOLDS:
            z_grid, nz_grid, integral = make_normalized_kernel(scan["nz_counts"][key], scan["z_edges"])
            if abs(integral - 1.0) > tolerance or np.any(nz_grid < 0):
                raise ValueError(f"Kernel normalization invariant failed for {key}: {integral}")
            group = handle.create_group(key)
            group.create_dataset("z", data=z_grid)
            group.create_dataset("nz", data=nz_grid)
            group.create_dataset("histogram_edges", data=scan["z_edges"])
            group.create_dataset("histogram_counts", data=scan["nz_counts"][key], compression="gzip", shuffle=True)
            group.attrs["n_halos"] = int(scan["counts"][key])
            group.attrs["normalization_trapezoid"] = integral
            group.attrs["retained_row_identity_sha256"] = scan["identity_sha256"][key]
            group.attrs["kernel_array_sha256"] = sha256_array(z_grid, nz_grid)
            kernels[key] = {"z": z_grid, "nz": nz_grid, "integral": integral, "sha256": group.attrs["kernel_array_sha256"]}
    return kernels


def plot_nz(path: pathlib.Path, scan: Mapping[str, Any], kernels: Mapping[str, Any]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    centers = 0.5 * (scan["z_edges"][:-1] + scan["z_edges"][1:])
    dz = np.diff(scan["z_edges"])
    primary_density = scan["nz_counts"]["primary"] / dz
    for key in THRESHOLDS:
        label = {"primary": r"$M_{proxy}\geq5\times10^{11}$", "Nge125": r"$N_{interp}\geq125$", "Nge150": r"$N_{interp}\geq150$"}[key]
        axes[0].step(centers, scan["nz_counts"][key] / dz, where="mid", color=COLORS[key], label=f"{label}; N={scan['counts'][key]:,}")
        axes[1].plot(centers, (scan["nz_counts"][key] / dz) / primary_density, color=COLORS[key])
    axes[0].set_ylabel(r"$dN/dz$")
    axes[0].legend(frameon=False)
    axes[0].set_title("c0000 catalog redshift distribution — exact 0.3 < z < 0.5 support")
    axes[1].set_ylabel("ratio to primary")
    axes[1].set_xlabel("redshift z")
    axes[1].set_ylim(0.75, 1.02)
    axes[1].grid(alpha=0.25)
    text = "  ".join(f"{key}: ∫n(z)dz={kernels[key]['integral']:.12f}" for key in THRESHOLDS)
    fig.text(0.5, 0.01, text, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_hmf(path: pathlib.Path, scan: Mapping[str, Any], theory_density: np.ndarray, expected: np.ndarray, volume: np.ndarray) -> None:
    z_edges = scan["diagnostic_z_edges"]
    mass_edges = scan["mass_edges"]
    mass_mid = np.sqrt(mass_edges[:-1] * mass_edges[1:])
    dlnm = np.diff(np.log(mass_edges))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey="row")
    for index, axis in enumerate(axes.flat):
        if index >= 4:
            break
        ax = axes.flat[index]
        counts = scan["mass_z_counts"]["primary"][index]
        observed = counts / (volume[index] * dlnm)
        positive = counts > 0
        ax.loglog(mass_mid[positive], observed[positive], "o", ms=3.5, color="black", label="catalog proxy HMF")
        ax.loglog(mass_mid, theory_density[index], color="#D55E00", lw=1.8, label="GODMAX T10 200c")
        ax.text(0.04, 0.06, f"{z_edges[index]:.2f} < z < {z_edges[index + 1]:.2f}", transform=ax.transAxes)
        ax.grid(alpha=0.2, which="both")
        if index == 0:
            ax.legend(frameon=False, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$M_{particle\ proxy}$ [$M_\odot/h$]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$dn/d\ln M$ [$(h/{\rm Mpc})^3$]")
    fig.suptitle("Conditional proxy-HMF check: particle-count proxy identified provisionally with Tinker M200c\nPoisson scatter/sample variance are not an acceptance test")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_hmf_residuals(path: pathlib.Path, scan: Mapping[str, Any], expected: np.ndarray) -> None:
    z_edges = np.asarray(scan["diagnostic_z_edges"])
    mass_mid = np.sqrt(np.asarray(scan["mass_edges"][:-1]) * np.asarray(scan["mass_edges"][1:]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for index, ax in enumerate(axes.flat):
        counts = np.asarray(scan["mass_z_counts"]["primary"][index], dtype=np.float64)
        valid = counts > 0
        ratio = counts[valid] / expected[index, valid]
        poisson_only = np.sqrt(counts[valid]) / expected[index, valid]
        ax.errorbar(mass_mid[valid], ratio, yerr=poisson_only, fmt="o", ms=3.5, color="black", capsize=1.5)
        ax.axhline(1.0, color="#D55E00", lw=1.5)
        ax.set_xscale("log")
        ax.set_xlim(scan["mass_edges"][0], scan["mass_edges"][-1])
        ax.text(0.04, 0.07, f"{z_edges[index]:.2f} < z < {z_edges[index + 1]:.2f}\nmedian={np.median(ratio):.3f}", transform=ax.transAxes)
        ax.grid(alpha=0.2, which="both")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$M_{particle\ proxy}$ [$M_\odot/h$]")
    for ax in axes[:, 0]:
        ax.set_ylabel("catalog / GODMAX T10")
    fig.suptitle("Conditional proxy-HMF residuals — error bars are Poisson-only; sample variance omitted")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bias(path: pathlib.Path, theory: Mapping[str, Any], conditioned: Mapping[str, Any], summaries: Mapping[str, Any]) -> None:
    mass = np.asarray(theory["mass_hmsun"])
    z = np.asarray(theory["z"])
    bias = np.asarray(theory["bias"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for target, color in zip((0.30, 0.40, 0.50), ("#0072B2", "#009E73", "#D55E00")):
        index = int(np.argmin(np.abs(z - target)))
        axes[0].semilogx(mass, bias[index], color=color, label=f"T10 b(M,z={z[index]:.3f})")
    axes[0].set_xlabel(r"$M_{200c}$ [$M_\odot/h$] (conditional proxy identification)")
    axes[0].set_ylabel("dimensionless T10 halo bias")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2, which="both")
    axes[1].plot(z, summaries["number_weighted_bias"], color="black", label="resolved HMF number-weighted")
    for key in THRESHOLDS:
        axes[1].plot(conditioned["z_mid"], conditioned[key], "o-", color=COLORS[key], label=f"catalog-weighted theory: {key}")
    axes[1].set_xlabel("redshift z")
    axes[1].set_ylabel("effective theory bias")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Theory bias diagnostic only — no empirical catalog bias measurement or low-mass completion")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_summary(path: pathlib.Path, report: Mapping[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    cosmo = report["cosmology"]
    counts = report["catalog"]["counts"]
    checks = report["checks"]
    lines = [
        "THREE-PROBE CATALOG → GODMAX INPUT VALIDATION",
        "",
        f"Catalog SHA-256: {report['catalog']['file_sha256']}",
        f"Rows: primary={counts['primary']:,}; N≥125={counts['Nge125']:,}; N≥150={counts['Nge150']:,}",
        f"Contributing source shells: {len(report['catalog']['source_files'])}/{len(report['catalog']['source_files'])}",
        f"Support: 0.3 < z < 0.5; 5e11 ≤ Mproxy < 1e16 Msun/h; full sky={report['catalog']['full_sky']}",
        "",
        "Catalog/source/effective GODMAX cosmology:",
        f"H0={cosmo['H0']:.15g}, h={report['catalog']['h']:.15g}, Ωm={cosmo['Om0']:.15g}, Ωb={cosmo['Ob0']:.15g}",
        f"σ8={cosmo['sigma8']:.15g}, ns={cosmo['ns']:.15g}, w0={cosmo['w0']:.15g}, flat={cosmo['flat']}",
        "",
        f"Kernel normalization passed: {checks['kernel_normalization']}",
        f"Cosmology exact-match passed: {checks['cosmology_exact_match']}",
        f"Source ordering passed: {checks['source_order_preserved']}",
        f"Grid robustness reported (no tuned tolerance): {checks['grid_robustness_reported']}",
        "",
        "LIMITATION: Mparticle proxy is only provisionally identified with M200c.",
        "Bias curves are GODMAX theory conditioned on catalog masses, not an empirical bias measurement.",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=THIS_DIR / "three_probe_mock_experiment.yaml")
    parser.add_argument("--catalog", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--chunk-rows", type=int, default=1_048_576)
    parser.add_argument("--skip-file-sha", action="store_true", help="Only for fast development; final evidence must verify SHA")
    args = parser.parse_args()

    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    theory_config = config["resolved_theory"]
    catalog_path = args.catalog or pathlib.Path(theory_config["catalog_path"])
    output_dir = args.output_dir or pathlib.Path(theory_config["validation_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if theory_config.get("validate_all_source_headers_from_catalog") is not True:
        raise ValueError("validate_all_source_headers_from_catalog must be literally true")
    with h5py.File(catalog_path, "r") as handle:
        source_files_for_headers = json.loads(str(handle.attrs["source_files_json"]))
    source_headers = [
        {"path": path, "header": load_source_header(pathlib.Path(path))}
        for path in source_files_for_headers
    ]
    source_header = source_headers[0]["header"]
    attrs, support, cosmology = validate_catalog_contract(
        catalog_path, theory_config, source_header, verify_file_sha=not args.skip_file_sha
    )
    for item in source_headers[1:]:
        shell_cosmology = canonical_cosmology(attrs, item["header"])
        if shell_cosmology != cosmology or item["header"] != source_header:
            raise ValueError(f"Source-shell header mismatch: {item['path']}")
    provenance = build_provenance(args.config, pathlib.Path(theory_config["default_params_path"]))

    lens = theory_config["lens_kernel"]
    diagnostic = theory_config["hmf_bias_diagnostic"]
    scan = stream_catalog(
        catalog_path,
        support.z_min,
        support.z_max,
        support.mass_min_hmsun,
        support.mass_max_hmsun,
        nz_bins=int(lens["histogram_bins"]),
        robust_nz_bins=int(lens["robustness_histogram_bins"]),
        nside=int(diagnostic["sky_fraction_nside"]),
        chunk_rows=args.chunk_rows,
    )
    if scan["counts"]["primary"] != int(attrs["n_halos"]):
        raise ValueError("Streamed primary count differs from catalog n_halos")
    source_files = json.loads(str(attrs["source_files_json"]))
    if sorted(scan["source_file_counts"]) != list(range(len(source_files))) or any(
        count <= 0 for count in scan["source_file_counts"].values()
    ):
        raise ValueError("Not every frozen source shell contributes at least one selected row")
    if not scan["full_sky"]:
        raise ValueError("Pre-registered full-sky selection was falsified")

    kernels = save_kernel(
        output_dir / lens["output_name"], scan, cosmology, theory_config, provenance, float(lens["normalization_tolerance"])
    )
    theory = build_godmax_hmf_bias(pathlib.Path(theory_config["default_params_path"]), cosmology, support, diagnostic, robust=False)
    robust_theory = build_godmax_hmf_bias(pathlib.Path(theory_config["default_params_path"]), cosmology, support, diagnostic, robust=True)

    expected, theory_density = binned_theory_hmf(theory, scan["diagnostic_z_edges"], scan["mass_edges"], cosmology)
    robust_expected, robust_density = binned_theory_hmf(robust_theory, scan["diagnostic_z_edges"], scan["mass_edges"], cosmology)
    finite = (expected > 0) & (robust_expected > 0)
    fractional_grid_difference = np.abs(robust_expected[finite] / expected[finite] - 1.0)
    summaries = bias_summaries(theory)
    robust_summaries = bias_summaries(robust_theory)
    conditioned = catalog_conditioned_bias(scan, theory)
    volumes = np.asarray([
        shell_volume(lo, hi, float(cosmology["Om0"]), float(cosmology["w0"]))
        for lo, hi in zip(scan["diagnostic_z_edges"][:-1], scan["diagnostic_z_edges"][1:])
    ])

    plot_nz(output_dir / "catalog_nz_thresholds.png", scan, kernels)
    plot_hmf(output_dir / "proxy_hmf_vs_tinker200c.png", scan, theory_density, expected, volumes)
    plot_hmf_residuals(output_dir / "proxy_hmf_residuals.png", scan, expected)
    plot_bias(output_dir / "conditional_theory_bias.png", theory, conditioned, summaries)

    primary_counts = np.asarray(scan["mass_z_counts"]["primary"], dtype=np.float64)
    occupied_hmf_bins = primary_counts > 0
    hmf_ratio = np.divide(primary_counts, expected, out=np.full_like(expected, np.nan), where=occupied_hmf_bins)
    grid_expected_ge1 = finite & (expected >= 1.0)
    grid_expected_ge100 = finite & (expected >= 100.0)

    report = {
        "status": "GATE2_INPUT_CONTRACT_PASS_COMMON_FIELD_INTEGRALS_AND_THRESHOLD_POSTERIORS_PENDING",
        "catalog": {
            "path": str(catalog_path),
            "file_size_bytes": catalog_path.stat().st_size,
            "file_sha256": theory_config["catalog_file_sha256"],
            "row_identity_sha256": str(attrs["row_identity_sha256"]),
            "selection_contract_sha256": str(attrs["selection_contract_sha256"]),
            "counts": scan["counts"],
            "source_files": source_files,
            "source_file_counts": scan["source_file_counts"],
            "retained_identity_sha256": scan["identity_sha256"],
            "selected_extrema": scan["selected_extrema"],
            "occupied_pixels_by_z": scan["occupied_pixels_by_z"],
            "total_pixels": scan["total_pixels"],
            "full_sky": scan["full_sky"],
            "h": float(attrs["h"]),
            "mass_unit": str(attrs["mass_unit"]),
            "mass_semantics": str(attrs["mass_semantics"]),
        },
        "cosmology": cosmology,
        "source_header": source_header,
        "source_headers": source_headers,
        "provenance": provenance,
        "resolved_support": _jsonable(support.__dict__),
        "kernel": {key: {"integral": kernels[key]["integral"], "sha256": kernels[key]["sha256"]} for key in THRESHOLDS},
        "theory": {
            "model": "numerical GODMAX T10, Delta=200c",
            "symbolic_hmf": False,
            "symbolic_pk": False,
            "unresolved_completion": False,
            "baseline_grid": theory["grid"],
            "robustness_grid": robust_theory["grid"],
            "grid_fractional_expected_count_difference": {
                "median": float(np.median(fractional_grid_difference)),
                "max": float(np.max(fractional_grid_difference)),
                "max_expected_ge_1": float(np.max(np.abs(robust_expected[grid_expected_ge1] / expected[grid_expected_ge1] - 1.0))),
                "max_expected_ge_100": float(np.max(np.abs(robust_expected[grid_expected_ge100] / expected[grid_expected_ge100] - 1.0))),
            },
            "catalog_over_theory_hmf_occupied_bins": {
                "median": float(np.nanmedian(hmf_ratio)),
                "min": float(np.nanmin(hmf_ratio)),
                "max": float(np.nanmax(hmf_ratio)),
            },
            "fullsky_volume_hmpc3": float(volumes.sum()),
            "z_bin_volumes_hmpc3": volumes,
            "resolved_mass_weighted_bias_integral_range": [
                float(np.min(summaries["resolved_mass_weighted_bias_integral"])),
                float(np.max(summaries["resolved_mass_weighted_bias_integral"])),
            ],
            "interpretation": diagnostic["interpretation"],
        },
        "checks": {
            "catalog_identity": True,
            "cosmology_exact_match": theory["effective_cosmology"] == cosmology and robust_theory["effective_cosmology"] == cosmology,
            "kernel_normalization": all(abs(kernels[key]["integral"] - 1.0) <= float(lens["normalization_tolerance"]) for key in THRESHOLDS),
            "source_order_preserved": scan["source_order_preserved"],
            "threshold_counts_monotone": scan["counts"]["primary"] >= scan["counts"]["Nge125"] >= scan["counts"]["Nge150"],
            "full_sky": scan["full_sky"],
            "grid_robustness_reported": True,
        },
        "limitations": [
            "The particle-count mass proxy is not a recovered spherical-overdensity M200c mass.",
            "HMF comparison is conditional on the provisional proxy-as-M200c identification.",
            "Catalog-weighted bias is theory assigned to proxy masses, not an empirical bias measurement.",
            "No unresolved/low-mass bias completion is applied to the reported resolved integrals.",
            "Gate 2 is not complete until common-support g/y/electron/matter integrals and the 125/150-particle posterior tests are run.",
        ],
    }
    if not all(report["checks"].values()):
        raise ValueError(f"One or more validation checks failed: {report['checks']}")

    np.savez_compressed(
        output_dir / "catalog_theory_validation_arrays.npz",
        z_edges=scan["z_edges"],
        robust_z_edges=scan["robust_z_edges"],
        primary_robust_nz_counts=scan["robust_nz_counts"]["primary"],
        Nge125_robust_nz_counts=scan["robust_nz_counts"]["Nge125"],
        Nge150_robust_nz_counts=scan["robust_nz_counts"]["Nge150"],
        diagnostic_z_edges=scan["diagnostic_z_edges"],
        mass_edges_hmsun=scan["mass_edges"],
        baseline_mass_hmsun=theory["mass_hmsun"],
        baseline_z=theory["z"],
        baseline_hmf=theory["hmf_dndlnm_h3mpc3"],
        baseline_bias=theory["bias"],
        robust_mass_hmsun=robust_theory["mass_hmsun"],
        robust_z=robust_theory["z"],
        robust_hmf=robust_theory["hmf_dndlnm_h3mpc3"],
        robust_bias=robust_theory["bias"],
        expected_counts=expected,
        robust_expected_counts=robust_expected,
        primary_catalog_over_theory_hmf=hmf_ratio,
        primary_mass_z_counts=scan["mass_z_counts"]["primary"],
        Nge125_mass_z_counts=scan["mass_z_counts"]["Nge125"],
        Nge150_mass_z_counts=scan["mass_z_counts"]["Nge150"],
        resolved_mass_weighted_bias_integral=summaries["resolved_mass_weighted_bias_integral"],
        number_weighted_bias=summaries["number_weighted_bias"],
        conditioned_bias_z=conditioned["z_mid"],
        conditioned_bias_primary=conditioned["primary"],
        conditioned_bias_Nge125=conditioned["Nge125"],
        conditioned_bias_Nge150=conditioned["Nge150"],
        provenance_json=np.asarray(json.dumps(_jsonable(provenance), sort_keys=True)),
    )
    with (output_dir / "catalog_theory_validation.json").open("w") as handle:
        json.dump(_jsonable(report), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    plot_summary(output_dir / "validation_summary.png", report)

    output_names = (
        "catalog_lens_kernel.h5",
        "catalog_nz_thresholds.png",
        "catalog_theory_validation.json",
        "catalog_theory_validation_arrays.npz",
        "conditional_theory_bias.png",
        "proxy_hmf_residuals.png",
        "proxy_hmf_vs_tinker200c.png",
        "validation_summary.png",
    )
    manifest = {
        "producer_provenance": provenance,
        "outputs": {
            name: {"path": str(output_dir / name), "sha256": sha256_file(output_dir / name)}
            for name in output_names
        },
    }
    with (output_dir / "validation_manifest.json").open("w") as handle:
        json.dump(_jsonable(manifest), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    print(json.dumps({
        "status": report["status"],
        "output_dir": str(output_dir),
        "counts": scan["counts"],
        "fullsky_volume_hmpc3": report["theory"]["fullsky_volume_hmpc3"],
        "cosmology": cosmology,
        "kernel_integrals": {key: kernels[key]["integral"] for key in THRESHOLDS},
        "grid_fractional_expected_count_difference": report["theory"]["grid_fractional_expected_count_difference"],
        "checks": report["checks"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
