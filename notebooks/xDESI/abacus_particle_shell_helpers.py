"""Utilities for Abacus particle-shell residual validation.

This module is intentionally local to ``notebooks/xDESI``.  It reads Abacus
HEALPix shell products read-only and writes only reusable caches under the
xDESI output tree.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import asdf
import h5py
import healpy as hp
import numpy as np
from astropy import constants as astro_const

import abacus_pasting_helpers as aph
from abacus_lightcone_catalog import ensure_under_xdesi


C_KM_S = 299792.458
Y_HE = 0.24
SHELL_STEP_RE = re.compile(r"(Step\d+-\d+)")


def step_id_from_path(path: Path | str) -> str:
    match = SHELL_STEP_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Could not parse Step range from {path}")
    return match.group(1)


def _quantity_dir(total_root: Path | str, quantity: str) -> Path:
    return Path(total_root).expanduser().resolve() / quantity


def _read_quantity_metadata(path: Path | str) -> dict:
    path = Path(path)
    with asdf.open(path, lazy_load=True) as af:
        post = af.tree["header_post"]
        headers = af.tree["headers"]
        data_key = next(iter(af.tree["data"].keys()))
        arr = af.tree["data"][data_key]
        return {
            "path": str(path),
            "name": path.name,
            "step_id": step_id_from_path(path),
            "data_key": data_key,
            "quantity": str(post.get("healpix_map_quantity", data_key)),
            "nside": int(post["healpix_map_nside"]),
            "order": str(post["healpix_order"]).lower(),
            "mean_fp64": float(post.get("healpix_map_mean_fp64", np.nan)),
            "n_input_steps": len(post.get("input_files", [])),
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "z_hi": float(headers[0]["Redshift"]),
            "z_lo": float(headers[-1]["Redshift"]),
            "chi_hi_hMpc": float(headers[0]["CoordinateDistanceHMpc"]),
            "chi_lo_hMpc": float(headers[-1]["CoordinateDistanceHMpc"]),
            "particle_mass_hMsun": float(headers[0]["ParticleMassHMsun"]),
            "h": float(headers[0]["H0"]) / 100.0,
            "Omega_M": float(headers[0]["Omega_M"]),
            "Omega_b": float(headers[0]["CAMB_Omega_b"]),
        }


def _read_crc32_table(quantity_dir: Path) -> dict:
    checksum_path = quantity_dir / "checksums.crc32"
    if not checksum_path.exists():
        return {}
    rows = {}
    with open(checksum_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 3:
                rows[parts[2]] = {"crc32": parts[0], "size_bytes": int(parts[1])}
    return rows


def summarize_healpix_quantity_dirs(total_root: Path | str) -> List[dict]:
    """Return one-row-per-quantity metadata for available shell products."""

    total_root = Path(total_root).expanduser().resolve()
    rows = []
    for quantity in ("heal-counts", "heal-vel-los", "heal-vel-theta", "heal-vel-phi"):
        qdir = total_root / quantity
        files = sorted(qdir.glob("*.asdf"))
        if not files:
            continue
        first = _read_quantity_metadata(files[0])
        last = _read_quantity_metadata(files[-1])
        rows.append(
            {
                "quantity_dir": quantity,
                "n_files": len(files),
                "data_key": first["data_key"],
                "map_quantity": first["quantity"],
                "nside": first["nside"],
                "order": first["order"],
                "dtype": first["dtype"],
                "first_step": first["step_id"],
                "last_step": last["step_id"],
                "z_max": first["z_hi"],
                "z_min": last["z_lo"],
            }
        )
    return rows


def discover_total_shells(total_root: Path | str, *, z_max_hint: Optional[float] = None) -> List[dict]:
    """Match total counts and LOS velocity shell files by Step range."""

    total_root = Path(total_root).expanduser().resolve()
    count_files = sorted(_quantity_dir(total_root, "heal-counts").glob("*.asdf"))
    files_to_scan = reversed(count_files) if z_max_hint is not None else count_files
    vel_file_list = sorted(_quantity_dir(total_root, "heal-vel-los").glob("*.asdf"))
    count_checksums = _read_crc32_table(_quantity_dir(total_root, "heal-counts"))
    vel_checksums = _read_crc32_table(_quantity_dir(total_root, "heal-vel-los"))
    vel_files = {step_id_from_path(path): path for path in vel_file_list}
    vel_template_meta = _read_quantity_metadata(vel_file_list[0]) if vel_file_list else {}
    shells = []
    for count_path in files_to_scan:
        meta_counts = _read_quantity_metadata(count_path)
        if z_max_hint is not None and float(meta_counts["z_lo"]) > float(z_max_hint):
            break
        step_id = meta_counts["step_id"]
        vel_path = vel_files.get(step_id)
        count_crc = count_checksums.get(Path(count_path).name, {})
        vel_crc = vel_checksums.get(Path(vel_path).name, {}) if vel_path else {}
        z_hi = float(meta_counts["z_hi"])
        z_lo = float(meta_counts["z_lo"])
        chi_hi = float(meta_counts["chi_hi_hMpc"])
        chi_lo = float(meta_counts["chi_lo_hMpc"])
        shells.append(
            {
                "step_id": step_id,
                "name": Path(count_path).name,
                "path_counts": str(count_path),
                "path_vel_los": str(vel_path) if vel_path else "",
                "counts_crc32": str(count_crc.get("crc32", "")),
                "counts_size_bytes": int(count_crc.get("size_bytes", -1)),
                "vel_los_crc32": str(vel_crc.get("crc32", "")),
                "vel_los_size_bytes": int(vel_crc.get("size_bytes", -1)),
                "z_hi": z_hi,
                "z_lo": z_lo,
                "z_mid": 0.5 * (z_hi + z_lo),
                "chi_hi_hMpc": chi_hi,
                "chi_lo_hMpc": chi_lo,
                "dchi_hMpc": abs(chi_hi - chi_lo),
                "nside_counts": int(meta_counts["nside"]),
                "nside_vel_los": int(vel_template_meta.get("nside", -1)) if vel_path else -1,
                "order_counts": str(meta_counts["order"]),
                "order_vel_los": str(vel_template_meta.get("order", "")) if vel_path else "",
                "mean_count_fine": float(meta_counts["mean_fp64"]),
                "particle_mass_hMsun": float(meta_counts["particle_mass_hMsun"]),
                "h": float(meta_counts["h"]),
                "Omega_M": float(meta_counts["Omega_M"]),
                "Omega_b": float(meta_counts["Omega_b"]),
                "n_input_steps": int(meta_counts["n_input_steps"]),
            }
        )
    return sorted(shells, key=lambda row: row["z_mid"], reverse=True)


def select_shells(shells: Sequence[Mapping[str, object]], z_min: float, z_max: float, max_shells: Optional[int] = None) -> List[dict]:
    selected = [dict(m) for m in shells if float(m["z_hi"]) > z_min and float(m["z_lo"]) < z_max]
    selected = sorted(selected, key=lambda row: row["z_mid"], reverse=True)
    return selected[:max_shells] if max_shells is not None else selected


def _assert_close_metadata(
    total_meta: Mapping[str, object],
    halo_meta: Mapping[str, object],
    keys: Sequence[str],
    *,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
) -> None:
    step = total_meta.get("step_id", "<unknown>")
    for key in keys:
        a = total_meta.get(key)
        b = halo_meta.get(key)
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            if int(a) != int(b):
                raise ValueError(f"Matched total/halo shell {step} metadata mismatch for {key}: {a} vs {b}")
        elif isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            if not np.isclose(float(a), float(b), rtol=rtol, atol=atol):
                raise ValueError(f"Matched total/halo shell {step} metadata mismatch for {key}: {a} vs {b}")
        else:
            if str(a) != str(b):
                raise ValueError(f"Matched total/halo shell {step} metadata mismatch for {key}: {a} vs {b}")


def discover_matched_total_halo_shells(
    total_root: Path | str,
    halo_root: Path | str,
    z_min: float,
    z_max: float,
    *,
    max_shells: Optional[int] = None,
) -> List[dict]:
    """Return shell metadata for the direct non-halo particle field.

    The returned rows are keyed by Abacus ``Step*`` range and contain matched
    all-particle and identified-halo-particle HEALPix shell products.  These
    rows are intended for a physically additive field split:

    ``field = total particles - particles already assigned to identified halos``.
    """

    total_root = Path(total_root).expanduser().resolve()
    halo_root = Path(halo_root).expanduser().resolve()
    total_shells = discover_total_shells(total_root, z_max_hint=float(z_max))
    halo_shells = discover_total_shells(halo_root, z_max_hint=float(z_max))
    halo_by_step = {str(row["step_id"]): row for row in halo_shells}
    selected_total = select_shells(total_shells, float(z_min), float(z_max), max_shells=max_shells)

    rows = []
    for total in selected_total:
        step = str(total["step_id"])
        if step not in halo_by_step:
            raise FileNotFoundError(f"Missing matched halo shell for {step} under {halo_root}")
        halo = dict(halo_by_step[step])
        _assert_close_metadata(
            total,
            halo,
            (
                "z_hi",
                "z_lo",
                "chi_hi_hMpc",
                "chi_lo_hMpc",
                "dchi_hMpc",
                "nside_counts",
                "nside_vel_los",
                "order_counts",
                "order_vel_los",
                "particle_mass_hMsun",
                "h",
                "Omega_M",
                "Omega_b",
            ),
        )
        if not total.get("path_vel_los") or not halo.get("path_vel_los"):
            raise FileNotFoundError(f"Missing heal-vel-los product for matched shell {step}")
        rows.append(
            {
                "step_id": step,
                "z_hi": float(total["z_hi"]),
                "z_lo": float(total["z_lo"]),
                "z_mid": float(total["z_mid"]),
                "chi_hi_hMpc": float(total["chi_hi_hMpc"]),
                "chi_lo_hMpc": float(total["chi_lo_hMpc"]),
                "dchi_hMpc": float(total["dchi_hMpc"]),
                "nside_counts": int(total["nside_counts"]),
                "nside_vel_los": int(total["nside_vel_los"]),
                "order_counts": str(total["order_counts"]),
                "order_vel_los": str(total["order_vel_los"]),
                "particle_mass_hMsun": float(total["particle_mass_hMsun"]),
                "h": float(total["h"]),
                "Omega_M": float(total["Omega_M"]),
                "Omega_b": float(total["Omega_b"]),
                "n_input_steps": int(total["n_input_steps"]),
                "total_root": str(total_root),
                "halo_root": str(halo_root),
                "path_counts_total": str(total["path_counts"]),
                "path_vel_los_total": str(total["path_vel_los"]),
                "path_counts_halo": str(halo["path_counts"]),
                "path_vel_los_halo": str(halo["path_vel_los"]),
                "total_counts_crc32": str(total.get("counts_crc32", "")),
                "total_counts_size_bytes": int(total.get("counts_size_bytes", -1)),
                "total_vel_los_crc32": str(total.get("vel_los_crc32", "")),
                "total_vel_los_size_bytes": int(total.get("vel_los_size_bytes", -1)),
                "halo_counts_crc32": str(halo.get("counts_crc32", "")),
                "halo_counts_size_bytes": int(halo.get("counts_size_bytes", -1)),
                "halo_vel_los_crc32": str(halo.get("vel_los_crc32", "")),
                "halo_vel_los_size_bytes": int(halo.get("vel_los_size_bytes", -1)),
                "mean_count_total_fine": float(total["mean_count_fine"]),
                "mean_count_halo_fine": float(halo["mean_count_fine"]),
                "shell_product": "direct_nonhalo_field_from_total_minus_halo",
            }
        )
    return sorted(rows, key=lambda row: row["z_mid"], reverse=True)


def particle_shell_cache_root(config: Mapping[str, object], nside: int) -> Path:
    path = Path(config["project"]["output_root"]).expanduser().resolve() / "abacus_particle_shell_tests" / f"nside{nside}"
    ensure_under_xdesi(path)
    return path


def shell_cache_path(cache_root: Path | str, shell_meta: Mapping[str, object], nside: int) -> Path:
    path = Path(cache_root).expanduser().resolve() / "downgraded_shells" / f"{shell_meta['step_id']}_nside{nside}.h5"
    ensure_under_xdesi(path)
    return path


def direct_field_shell_cache_path(cache_root: Path | str, shell_meta: Mapping[str, object], nside: int) -> Path:
    path = Path(cache_root).expanduser().resolve() / "direct_field_shells" / f"{shell_meta['step_id']}_nside{nside}.h5"
    ensure_under_xdesi(path)
    return path


def nfw_template_cache_path(
    cache_root: Path | str,
    catalog_key: str,
    shell_meta: Mapping[str, object],
    nside: int,
    velocity_bins: int,
) -> Path:
    path = (
        Path(cache_root).expanduser().resolve()
        / "selected_halo_templates"
        / f"{catalog_key}_{shell_meta['step_id']}_nside{nside}_vb{velocity_bins}.h5"
    )
    ensure_under_xdesi(path)
    return path


def _write_hdf5_atomic(path: Path, datasets: Mapping[str, np.ndarray], attrs: Mapping[str, object]) -> None:
    ensure_under_xdesi(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as handle:
        for key, value in datasets.items():
            handle.create_dataset(key, data=value, compression="lzf", shuffle=True)
        for key, value in attrs.items():
            if value is None:
                handle.attrs[key] = ""
            elif isinstance(value, (dict, list, tuple)):
                handle.attrs[key] = json.dumps(value)
            else:
                handle.attrs[key] = value
    os.replace(tmp_path, path)


def read_or_create_downgraded_shell_cache(
    shell_meta: Mapping[str, object],
    nside: int,
    cache_root: Path | str,
    *,
    overwrite: bool = False,
    batch_parent_pixels: int = 262_144,
) -> Path:
    """Cache one shell downgraded to RING order at ``nside``.

    ``heal-vel-los`` is treated as a mean velocity in each fine pixel, so the
    coarse velocity is count-weighted and ``momentum_los`` is ``counts * v``.
    """

    out_path = shell_cache_path(cache_root, shell_meta, nside)
    if out_path.exists() and not overwrite:
        return out_path

    count_path = Path(str(shell_meta["path_counts"]))
    vel_path = Path(str(shell_meta.get("path_vel_los", "")))
    if not vel_path.exists():
        raise FileNotFoundError(f"Missing matched heal-vel-los file for {shell_meta['step_id']}")

    with asdf.open(count_path, lazy_load=True) as fc, asdf.open(vel_path, lazy_load=True) as fv:
        cpost = fc.tree["header_post"]
        vpost = fv.tree["header_post"]
        nside_in = int(cpost["healpix_map_nside"])
        nside_vel = int(vpost["healpix_map_nside"])
        order = str(cpost["healpix_order"]).lower()
        if order != "nest" or str(vpost["healpix_order"]).lower() != "nest":
            raise ValueError("Expected NESTED input maps for counts and velocity.")
        if nside_in != nside_vel:
            raise ValueError(f"Counts and vel-los NSIDE mismatch: {nside_in} vs {nside_vel}")
        if nside_in % nside != 0:
            raise ValueError(f"Input NSIDE {nside_in} is not divisible by output NSIDE {nside}")

        group = (nside_in // nside) ** 2
        npix_out = hp.nside2npix(nside)
        counts_nested = np.empty(npix_out, dtype=np.float32)
        momentum_nested = np.empty(npix_out, dtype=np.float32)
        vel_mean_nested = np.empty(npix_out, dtype=np.float32)
        counts_data = fc.tree["data"]["heal-counts"]
        vel_data = fv.tree["data"]["heal-vel-los"]

        input_count_sum = 0.0
        input_momentum_sum = 0.0
        nonzero_count_pix = 0
        nonzero_vel_pix = 0
        vel_nonzero_count_zero = 0
        t0 = time.perf_counter()
        for start in range(0, npix_out, batch_parent_pixels):
            end = min(start + batch_parent_pixels, npix_out)
            count_block = np.asarray(counts_data[start * group : end * group], dtype=np.float64).reshape(end - start, group)
            vel_block = np.asarray(vel_data[start * group : end * group], dtype=np.float64).reshape(end - start, group)
            count_sum = count_block.sum(axis=1)
            momentum_sum = (count_block * vel_block).sum(axis=1)
            vel_mean = np.divide(momentum_sum, count_sum, out=np.zeros_like(momentum_sum), where=count_sum > 0)

            counts_nested[start:end] = count_sum.astype(np.float32)
            momentum_nested[start:end] = momentum_sum.astype(np.float32)
            vel_mean_nested[start:end] = vel_mean.astype(np.float32)

            input_count_sum += float(count_block.sum())
            input_momentum_sum += float(momentum_sum.sum())
            nonzero_count_pix += int(np.count_nonzero(count_block))
            nonzero_vel_pix += int(np.count_nonzero(vel_block))
            vel_nonzero_count_zero += int(np.count_nonzero((vel_block != 0.0) & (count_block == 0.0)))

        counts_ring = hp.reorder(counts_nested, n2r=True)
        momentum_ring = hp.reorder(momentum_nested, n2r=True)
        vel_mean_ring = hp.reorder(vel_mean_nested, n2r=True)

    attrs = {
        **{k: shell_meta[k] for k in shell_meta if isinstance(shell_meta[k], (str, int, float, bool))},
        "nside_out": int(nside),
        "nside_in": int(nside_in),
        "input_order": order,
        "downgrade_method": "sum NESTED children; vel-los count-weighted mean",
        "velocity_interpretation": "fine-pixel mean velocity; momentum_los=sum(counts*vel_los)",
        "input_count_sum": input_count_sum,
        "output_count_sum": float(np.sum(counts_ring, dtype=np.float64)),
        "input_momentum_los_sum": input_momentum_sum,
        "output_momentum_los_sum": float(np.sum(momentum_ring, dtype=np.float64)),
        "nonzero_count_fine_pixels": nonzero_count_pix,
        "nonzero_vel_fine_pixels": nonzero_vel_pix,
        "vel_nonzero_count_zero_fine_pixels": vel_nonzero_count_zero,
        "cache_runtime_sec": time.perf_counter() - t0,
    }
    _write_hdf5_atomic(
        out_path,
        {
            f"counts_{nside}": counts_ring.astype(np.float32),
            f"vel_los_count_weighted_{nside}": vel_mean_ring.astype(np.float32),
            f"momentum_los_{nside}": momentum_ring.astype(np.float32),
        },
        attrs,
    )
    return out_path


def load_downgraded_shell_cache(path: Path | str, nside: int) -> Tuple[dict, dict]:
    with h5py.File(path, "r") as handle:
        data = {
            "counts": handle[f"counts_{nside}"][:],
            "vel_los": handle[f"vel_los_count_weighted_{nside}"][:],
            "momentum_los": handle[f"momentum_los_{nside}"][:],
        }
        attrs = dict(handle.attrs)
    return data, attrs


def read_or_create_direct_field_shell_cache(
    shell_meta: Mapping[str, object],
    nside: int,
    cache_root: Path | str,
    *,
    overwrite: bool = False,
    batch_parent_pixels: int = 262_144,
    clip_negative_counts: bool = False,
) -> Path:
    """Cache one direct non-halo field shell at ``nside`` in RING order.

    The input ``heal-vel-los`` products are mean velocities in occupied fine
    pixels.  Therefore the direct field momentum is formed before downgrading:

    ``counts_total * vlos_total - counts_halo * vlos_halo``.

    By default, negative ``counts_total - counts_halo`` is treated as a hard
    data-integrity failure.  ``clip_negative_counts`` is intended only for
    explicit diagnostics.
    """

    out_path = direct_field_shell_cache_path(cache_root, shell_meta, nside)
    if out_path.exists() and not overwrite:
        return out_path

    count_total_path = Path(str(shell_meta["path_counts_total"]))
    vel_total_path = Path(str(shell_meta["path_vel_los_total"]))
    count_halo_path = Path(str(shell_meta["path_counts_halo"]))
    vel_halo_path = Path(str(shell_meta["path_vel_los_halo"]))
    for path in (count_total_path, vel_total_path, count_halo_path, vel_halo_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with (
        asdf.open(count_total_path, lazy_load=True) as fc_total,
        asdf.open(vel_total_path, lazy_load=True) as fv_total,
        asdf.open(count_halo_path, lazy_load=True) as fc_halo,
        asdf.open(vel_halo_path, lazy_load=True) as fv_halo,
    ):
        cpost_total = fc_total.tree["header_post"]
        vpost_total = fv_total.tree["header_post"]
        cpost_halo = fc_halo.tree["header_post"]
        vpost_halo = fv_halo.tree["header_post"]
        nside_in = int(cpost_total["healpix_map_nside"])
        nside_vel = int(vpost_total["healpix_map_nside"])
        order = str(cpost_total["healpix_order"]).lower()
        if order != "nest" or str(vpost_total["healpix_order"]).lower() != "nest":
            raise ValueError("Expected NESTED total input maps for counts and velocity.")
        if str(cpost_halo["healpix_order"]).lower() != "nest" or str(vpost_halo["healpix_order"]).lower() != "nest":
            raise ValueError("Expected NESTED halo input maps for counts and velocity.")
        if int(cpost_halo["healpix_map_nside"]) != nside_in or nside_vel != nside_in or int(vpost_halo["healpix_map_nside"]) != nside_in:
            raise ValueError("Counts and vel-los NSIDE mismatch between total and halo shell products.")
        if nside_in % nside != 0:
            raise ValueError(f"Input NSIDE {nside_in} is not divisible by output NSIDE {nside}")

        group = (nside_in // nside) ** 2
        npix_out = hp.nside2npix(nside)
        counts_field_nested = np.empty(npix_out, dtype=np.float32)
        momentum_field_nested = np.empty(npix_out, dtype=np.float32)
        vel_field_nested = np.empty(npix_out, dtype=np.float32)

        counts_total_data = fc_total.tree["data"]["heal-counts"]
        vel_total_data = fv_total.tree["data"]["heal-vel-los"]
        counts_halo_data = fc_halo.tree["data"]["heal-counts"]
        vel_halo_data = fv_halo.tree["data"]["heal-vel-los"]

        total_count_sum = 0.0
        halo_count_sum = 0.0
        field_count_sum = 0.0
        total_momentum_sum = 0.0
        halo_momentum_sum = 0.0
        field_momentum_sum = 0.0
        negative_count_fine_pixels = 0
        negative_count_parent_pixels = 0
        min_field_count_fine = np.inf
        min_field_count_parent = np.inf
        t0 = time.perf_counter()

        for start in range(0, npix_out, batch_parent_pixels):
            end = min(start + batch_parent_pixels, npix_out)
            count_total_block = np.asarray(counts_total_data[start * group : end * group], dtype=np.float64).reshape(end - start, group)
            vel_total_block = np.asarray(vel_total_data[start * group : end * group], dtype=np.float64).reshape(end - start, group)
            count_halo_block = np.asarray(counts_halo_data[start * group : end * group], dtype=np.float64).reshape(end - start, group)
            vel_halo_block = np.asarray(vel_halo_data[start * group : end * group], dtype=np.float64).reshape(end - start, group)

            field_count_block = count_total_block - count_halo_block
            neg_fine = field_count_block < 0.0
            n_neg_fine = int(np.count_nonzero(neg_fine))
            if n_neg_fine and not clip_negative_counts:
                raise ValueError(
                    f"{shell_meta['step_id']} has {n_neg_fine} fine pixels with total counts < halo counts; "
                    "refusing to build a direct field cache without clip_negative_counts=True."
                )
            field_momentum_block = count_total_block * vel_total_block - count_halo_block * vel_halo_block
            if n_neg_fine and clip_negative_counts:
                field_count_block = np.where(neg_fine, 0.0, field_count_block)
                field_momentum_block = np.where(neg_fine, 0.0, field_momentum_block)

            field_count_parent = field_count_block.sum(axis=1)
            field_momentum_parent = field_momentum_block.sum(axis=1)
            field_vel_parent = np.divide(
                field_momentum_parent,
                field_count_parent,
                out=np.zeros_like(field_momentum_parent),
                where=field_count_parent > 0.0,
            )

            counts_field_nested[start:end] = field_count_parent.astype(np.float32)
            momentum_field_nested[start:end] = field_momentum_parent.astype(np.float32)
            vel_field_nested[start:end] = field_vel_parent.astype(np.float32)

            total_count_sum += float(count_total_block.sum())
            halo_count_sum += float(count_halo_block.sum())
            field_count_sum += float(field_count_block.sum())
            total_momentum_sum += float((count_total_block * vel_total_block).sum())
            halo_momentum_sum += float((count_halo_block * vel_halo_block).sum())
            field_momentum_sum += float(field_momentum_block.sum())
            negative_count_fine_pixels += n_neg_fine
            negative_count_parent_pixels += int(np.count_nonzero(field_count_parent < 0.0))
            min_field_count_fine = min(min_field_count_fine, float(np.min(field_count_block)))
            min_field_count_parent = min(min_field_count_parent, float(np.min(field_count_parent)))

        counts_field_ring = hp.reorder(counts_field_nested, n2r=True)
        momentum_field_ring = hp.reorder(momentum_field_nested, n2r=True)
        vel_field_ring = hp.reorder(vel_field_nested, n2r=True)

    output_field_count_sum = float(np.sum(counts_field_ring, dtype=np.float64))
    output_field_momentum_sum = float(np.sum(momentum_field_ring, dtype=np.float64))
    attrs = {
        **{k: shell_meta[k] for k in shell_meta if isinstance(shell_meta[k], (str, int, float, bool))},
        "nside_out": int(nside),
        "nside_in": int(nside_in),
        "input_order": order,
        "cache_kind": "direct_nonhalo_field_shell",
        "field_definition": "total particle shell minus identified-halo particle shell",
        "downgrade_method": "sum NESTED children; direct field momentum=sum(total_counts*total_vlos - halo_counts*halo_vlos)",
        "velocity_interpretation": "fine-pixel mean velocity; field momentum formed before downgrade",
        "clip_negative_counts": bool(clip_negative_counts),
        "input_total_count_sum": total_count_sum,
        "input_halo_count_sum": halo_count_sum,
        "input_field_count_sum": field_count_sum,
        "output_field_count_sum": output_field_count_sum,
        "input_total_momentum_los_sum": total_momentum_sum,
        "input_halo_momentum_los_sum": halo_momentum_sum,
        "input_field_momentum_los_sum": field_momentum_sum,
        "output_field_momentum_los_sum": output_field_momentum_sum,
        "mean_total_counts_out": total_count_sum / float(npix_out),
        "mean_halo_counts_out": halo_count_sum / float(npix_out),
        "mean_field_counts_out": field_count_sum / float(npix_out),
        "field_mass_fraction": field_count_sum / max(total_count_sum, 1.0),
        "halo_mass_fraction": halo_count_sum / max(total_count_sum, 1.0),
        "negative_count_fine_pixels": int(negative_count_fine_pixels),
        "negative_count_parent_pixels": int(negative_count_parent_pixels),
        "min_field_count_fine": 0.0 if not np.isfinite(min_field_count_fine) else float(min_field_count_fine),
        "min_field_count_parent": 0.0 if not np.isfinite(min_field_count_parent) else float(min_field_count_parent),
        "cache_runtime_sec": time.perf_counter() - t0,
    }
    _write_hdf5_atomic(
        out_path,
        {
            f"counts_{nside}": counts_field_ring.astype(np.float32),
            f"vel_los_count_weighted_{nside}": vel_field_ring.astype(np.float32),
            f"momentum_los_{nside}": momentum_field_ring.astype(np.float32),
        },
        attrs,
    )
    return out_path


def load_direct_field_shell_cache(path: Path | str, nside: int) -> Tuple[dict, dict]:
    return load_downgraded_shell_cache(path, nside)


def velocity_interpretation_diagnostic(shell_meta: Mapping[str, object], sample_pixels: int = 5_000_000) -> dict:
    """Check whether vel-los behaves like a mean velocity rather than a sum."""

    with asdf.open(shell_meta["path_counts"], lazy_load=True) as fc, asdf.open(shell_meta["path_vel_los"], lazy_load=True) as fv:
        counts = np.asarray(fc.tree["data"]["heal-counts"][:sample_pixels])
        vel = np.asarray(fv.tree["data"]["heal-vel-los"][:sample_pixels])
    nonzero = counts > 0
    rows = {
        "step_id": shell_meta["step_id"],
        "sample_pixels": int(len(counts)),
        "count_nonzero_frac": float(np.mean(nonzero)),
        "vel_nonzero_frac": float(np.mean(vel != 0)),
        "vel_nonzero_where_count_zero": int(np.count_nonzero((vel != 0) & (~nonzero))),
    }
    for count_value in (1, 2, 4, 8, 16, 32):
        mask = counts == count_value
        if np.count_nonzero(mask) >= 20:
            rows[f"vel_std_count{count_value}"] = float(np.std(vel[mask]))
            rows[f"vel_mean_count{count_value}"] = float(np.mean(vel[mask]))
            rows[f"n_count{count_value}"] = int(np.count_nonzero(mask))
    return rows


def tau_prefactor_per_hMpc(cosmo_params: Mapping[str, float]) -> float:
    h = float(cosmo_params["H0"]) / 100.0
    ob0 = float(cosmo_params["Ob0"])
    ne0_cm3 = (1.878e-29 * h**2) * ob0 * (1.0 - Y_HE / 2.0) / astro_const.m_p.to("g").value
    sigma_t_cm2 = astro_const.sigma_T.to("cm2").value
    mpc_cm = astro_const.pc.to("cm").value * 1.0e6
    return float(sigma_t_cm2 * ne0_cm3 * mpc_cm / h)


def _trapz_np(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _kernel_vector_for_z(kernel, z_len: int, name: str, bin_index: int = 0) -> np.ndarray:
    """Return one redshift kernel vector from GODMAX kernel arrays."""

    raw = np.asarray(kernel, dtype=np.float64)
    arr = np.squeeze(raw)
    if arr.ndim == 1:
        if arr.shape[0] == z_len:
            return arr
    elif arr.ndim == 2:
        if arr.shape[1] == z_len:
            if bin_index >= arr.shape[0]:
                raise ValueError(f"{name} bin_index={bin_index} is out of range for shape {raw.shape}.")
            return arr[bin_index, :]
        if arr.shape[0] == z_len:
            if bin_index >= arr.shape[1]:
                raise ValueError(f"{name} bin_index={bin_index} is out of range for shape {raw.shape}.")
            return arr[:, bin_index]

    raise ValueError(
        f"Could not extract a 1D redshift kernel from {name}: "
        f"raw shape={raw.shape}, squeezed shape={arr.shape}, expected z length={z_len}."
    )


def compute_shell_weights(
    shell_meta: Mapping[str, object],
    cls_cmb,
    cls_wl,
    *,
    mode: str = "average",
    n_samples: int = 32,
) -> dict:
    """Integrate map kernels across one shell in h^-1 Mpc."""

    z_grid = np.asarray(cls_cmb.z_array_for_Cls, dtype=np.float64)
    chi_grid = np.asarray(cls_cmb.chi_array_for_Cls, dtype=np.float64)
    w_cmb_grid = _kernel_vector_for_z(cls_cmb.Wk_mat, len(z_grid), "cls_cmb.Wk_mat")
    w_wl_grid = _kernel_vector_for_z(cls_wl.Wk_gravonly_mat, len(z_grid), "cls_wl.Wk_gravonly_mat")
    tau_prefac = tau_prefactor_per_hMpc(cls_cmb.cosmo_params)

    chi_lo = float(shell_meta["chi_lo_hMpc"])
    chi_hi = float(shell_meta["chi_hi_hMpc"])
    z_mid = float(shell_meta["z_mid"])
    dchi = abs(chi_hi - chi_lo)
    if mode == "midpoint":
        return {
            "kappa_cmb": float(np.interp(z_mid, z_grid, w_cmb_grid) * dchi),
            "kappa_wl": float(np.interp(z_mid, z_grid, w_wl_grid) * dchi),
            "tau": float(tau_prefac * (1.0 + z_mid) ** 2 * dchi),
            "mode": mode,
        }
    if mode != "average":
        raise ValueError(f"Unknown shell-weight mode {mode}")

    chi_samples = np.linspace(min(chi_lo, chi_hi), max(chi_lo, chi_hi), int(n_samples))
    z_samples = np.interp(chi_samples, chi_grid, z_grid)
    w_cmb = np.interp(z_samples, z_grid, w_cmb_grid)
    w_wl = np.interp(z_samples, z_grid, w_wl_grid)
    tau_w = tau_prefac * (1.0 + z_samples) ** 2
    return {
        "kappa_cmb": _trapz_np(w_cmb, chi_samples),
        "kappa_wl": _trapz_np(w_wl, chi_samples),
        "tau": _trapz_np(tau_w, chi_samples),
        "mode": mode,
    }


def _integrated_shell_kernel(
    shell_meta: Mapping[str, object],
    z_grid: np.ndarray,
    chi_grid: np.ndarray,
    kernel_grid: np.ndarray,
    *,
    mode: str,
    n_samples: int,
) -> float:
    chi_lo = float(shell_meta["chi_lo_hMpc"])
    chi_hi = float(shell_meta["chi_hi_hMpc"])
    z_mid = float(shell_meta["z_mid"])
    dchi = abs(chi_hi - chi_lo)
    if mode == "midpoint":
        return float(np.interp(z_mid, z_grid, kernel_grid) * dchi)
    if mode != "average":
        raise ValueError(f"Unknown shell-weight mode {mode}")
    chi_samples = np.linspace(min(chi_lo, chi_hi), max(chi_lo, chi_hi), int(n_samples))
    z_samples = np.interp(chi_samples, chi_grid, z_grid)
    return _trapz_np(np.interp(z_samples, z_grid, kernel_grid), chi_samples)


def compute_dataset_shell_weights(
    shell_meta: Mapping[str, object],
    cls_cmb,
    cls_wl,
    *,
    wl_source_bins: Sequence[int] = (1,),
    mode: str = "average",
    n_samples: int = 32,
) -> dict:
    """Return shell weights keyed by pasted HDF5 map dataset name."""

    z_grid = np.asarray(cls_cmb.z_array_for_Cls, dtype=np.float64)
    chi_grid = np.asarray(cls_cmb.chi_array_for_Cls, dtype=np.float64)
    z_grid_wl = np.asarray(cls_wl.z_array_for_Cls, dtype=np.float64)
    chi_grid_wl = np.asarray(cls_wl.chi_array_for_Cls, dtype=np.float64)
    if len(z_grid) != len(z_grid_wl) or not np.allclose(z_grid, z_grid_wl):
        raise ValueError("CMB and WL theory objects have different z grids; cannot build shared shell weights.")
    if len(chi_grid) != len(chi_grid_wl) or not np.allclose(chi_grid, chi_grid_wl):
        raise ValueError("CMB and WL theory objects have different chi grids; cannot build shared shell weights.")

    weights = {"mode": mode}
    w_cmb_grid = _kernel_vector_for_z(cls_cmb.Wk_mat, len(z_grid), "cls_cmb.Wk_mat")
    weights["map_kappa_cmb"] = _integrated_shell_kernel(
        shell_meta,
        z_grid,
        chi_grid,
        w_cmb_grid,
        mode=mode,
        n_samples=n_samples,
    )
    for source_bin in sorted({int(value) for value in wl_source_bins if int(value) >= 1}):
        dataset = "map_kappa_wl" if source_bin == 1 else f"map_kappa_wl_tomo{source_bin}"
        w_wl_grid = _kernel_vector_for_z(
            cls_wl.Wk_gravonly_mat,
            len(z_grid),
            "cls_wl.Wk_gravonly_mat",
            bin_index=source_bin - 1,
        )
        weights[dataset] = _integrated_shell_kernel(
            shell_meta,
            z_grid,
            chi_grid,
            w_wl_grid,
            mode=mode,
            n_samples=n_samples,
        )

    tau_prefac = tau_prefactor_per_hMpc(cls_cmb.cosmo_params)
    tau_grid = tau_prefac * (1.0 + z_grid) ** 2
    tau_weight = _integrated_shell_kernel(
        shell_meta,
        z_grid,
        chi_grid,
        tau_grid,
        mode=mode,
        n_samples=n_samples,
    )
    weights["map_tau"] = tau_weight
    weights["map_ksz"] = tau_weight
    return weights


def point_halo_templates_for_shell(catalog: Mapping[str, np.ndarray], shell_meta: Mapping[str, object], nside: int) -> dict:
    """Selected-halo count and momentum templates using point-mass halos."""

    mask = (catalog["z"] >= float(shell_meta["z_lo"])) & (catalog["z"] < float(shell_meta["z_hi"]))
    out_count = np.zeros(hp.nside2npix(nside), dtype=np.float32)
    out_momentum = np.zeros_like(out_count)
    if not np.any(mask):
        return {"count_template": out_count, "momentum_template": out_momentum, "n_halos": 0, "mass_sum_hMsun": 0.0}
    pix = hp.ang2pix(nside, catalog["ra_deg"][mask], catalog["dec_deg"][mask], lonlat=True)
    particle_equiv = catalog["M200c_hMsun"][mask] / float(shell_meta["particle_mass_hMsun"])
    momentum_equiv = particle_equiv * catalog["vlos_kms"][mask]
    np.add.at(out_count, pix, particle_equiv.astype(np.float32))
    np.add.at(out_momentum, pix, momentum_equiv.astype(np.float32))
    return {
        "count_template": out_count,
        "momentum_template": out_momentum,
        "n_halos": int(np.count_nonzero(mask)),
        "mass_sum_hMsun": float(np.sum(catalog["M200c_hMsun"][mask])),
    }


def empty_shell_field_maps(nside: int, map_keys: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    npix = hp.nside2npix(nside)
    keys = tuple(map_keys) if map_keys is not None else ("map_kappa_cmb", "map_kappa_wl", "map_tau", "map_ksz")
    return {str(key): np.zeros(npix, dtype=np.float32) for key in keys}


def add_shell_to_field_maps(
    maps: Dict[str, np.ndarray],
    counts: np.ndarray,
    momentum_los: np.ndarray,
    weights: Mapping[str, float],
    *,
    count_template: Optional[np.ndarray] = None,
    momentum_template: Optional[np.ndarray] = None,
    clip_negative_counts: bool = False,
    mean_total_counts: Optional[float] = None,
) -> dict:
    """Add one shell to kappa/tau/kSZ maps.

    Residual fields are normalized by the total shell mean, not by the residual
    mean.  This keeps the missing-mass contribution in full-matter units.
    """

    counts = np.asarray(counts, dtype=np.float32)
    momentum_los = np.asarray(momentum_los, dtype=np.float32)
    mean_total = float(np.mean(counts) if mean_total_counts is None else mean_total_counts)
    if mean_total <= 0:
        raise ValueError("Shell mean count is non-positive.")

    if count_template is None:
        source_counts = counts
        source_momentum = momentum_los
    else:
        source_counts = counts - np.asarray(count_template, dtype=np.float32)
        source_momentum = momentum_los - np.asarray(momentum_template if momentum_template is not None else 0.0, dtype=np.float32)
        if clip_negative_counts:
            neg = source_counts < 0.0
            source_counts = np.where(neg, 0.0, source_counts)
            source_momentum = np.where(neg, 0.0, source_momentum)

    mean_source = float(np.mean(source_counts))
    delta_source_totalnorm = (source_counts - mean_source) / mean_total
    dataset_weights = {
        "map_kappa_cmb": weights.get("map_kappa_cmb", weights.get("kappa_cmb")),
        "map_kappa_wl": weights.get("map_kappa_wl", weights.get("kappa_wl")),
        "map_tau": weights.get("map_tau", weights.get("tau")),
        "map_ksz": weights.get("map_ksz", weights.get("map_tau", weights.get("tau"))),
    }
    for key, value in weights.items():
        if str(key).startswith("map_kappa_wl_tomo"):
            dataset_weights[str(key)] = value

    for key, weight in dataset_weights.items():
        if key not in maps or weight is None:
            continue
        if key == "map_ksz":
            maps[key] += -float(weight) * (source_momentum / mean_total) / C_KM_S
        else:
            maps[key] += float(weight) * delta_source_totalnorm
    return {
        "mean_total_counts": mean_total,
        "mean_source_counts": mean_source,
        "source_mass_fraction": mean_source / mean_total,
        "frac_negative_source_pix": float(np.mean(source_counts < 0.0)) if count_template is not None else 0.0,
        "min_source_counts": float(np.min(source_counts)),
    }


def build_shell_field_maps(
    shell_meta: Sequence[Mapping[str, object]],
    cache_paths: Mapping[str, Path],
    weights_by_step: Mapping[str, Mapping[str, float]],
    nside: int,
    *,
    templates_by_step: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
    clip_negative_counts: bool = False,
    map_keys: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    maps = empty_shell_field_maps(nside, map_keys=map_keys)
    diagnostics = []
    for meta in shell_meta:
        step = str(meta["step_id"])
        data, _ = load_downgraded_shell_cache(cache_paths[step], nside)
        template = templates_by_step.get(step) if templates_by_step else None
        diag = add_shell_to_field_maps(
            maps,
            data["counts"],
            data["momentum_los"],
            weights_by_step[step],
            count_template=None if template is None else template["count_template"],
            momentum_template=None if template is None else template.get("momentum_template"),
            clip_negative_counts=clip_negative_counts,
        )
        diagnostics.append({"step_id": step, **diag, **{f"w_{k}": float(v) for k, v in weights_by_step[step].items() if k != "mode"}})
    for key in maps:
        maps[key] = np.asarray(maps[key] - np.mean(maps[key]), dtype=np.float32)
    return maps, diagnostics


def build_direct_field_maps(
    shell_meta: Sequence[Mapping[str, object]],
    cache_paths: Mapping[str, Path],
    weights_by_step: Mapping[str, Mapping[str, float]],
    nside: int,
    *,
    map_keys: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    """Build additive direct non-halo field maps from direct field caches."""

    maps = empty_shell_field_maps(nside, map_keys=map_keys)
    diagnostics = []
    for meta in shell_meta:
        step = str(meta["step_id"])
        data, attrs = load_direct_field_shell_cache(cache_paths[step], nside)
        mean_total = float(attrs.get("mean_total_counts_out", np.nan))
        if not np.isfinite(mean_total) or mean_total <= 0.0:
            total_sum = float(attrs.get("input_total_count_sum", np.nan))
            mean_total = total_sum / float(hp.nside2npix(nside))
        diag = add_shell_to_field_maps(
            maps,
            data["counts"],
            data["momentum_los"],
            weights_by_step[step],
            mean_total_counts=mean_total,
        )
        diagnostics.append(
            {
                "step_id": step,
                **diag,
                "cache_path": str(cache_paths[step]),
                "field_mass_fraction_cache": float(attrs.get("field_mass_fraction", np.nan)),
                "halo_mass_fraction_cache": float(attrs.get("halo_mass_fraction", np.nan)),
                "negative_count_fine_pixels": int(attrs.get("negative_count_fine_pixels", 0)),
                "negative_count_parent_pixels": int(attrs.get("negative_count_parent_pixels", 0)),
                "input_total_count_sum": float(attrs.get("input_total_count_sum", np.nan)),
                "input_halo_count_sum": float(attrs.get("input_halo_count_sum", np.nan)),
                "input_field_count_sum": float(attrs.get("input_field_count_sum", np.nan)),
                **{f"w_{k}": float(v) for k, v in weights_by_step[step].items() if k != "mode"},
            }
        )
    for key in maps:
        maps[key] = np.asarray(maps[key] - np.mean(maps[key]), dtype=np.float32)
    return maps, diagnostics


def add_field_maps(base_maps: Mapping[str, np.ndarray], residual_maps: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {key: np.array(value, copy=True) for key, value in base_maps.items()}
    for key, value in residual_maps.items():
        out[key] = np.asarray(out.get(key, 0.0), dtype=np.float32) + np.asarray(value, dtype=np.float32)
    return out


def complete_maps_for_measurement(field_maps: Mapping[str, np.ndarray], nside: int) -> Dict[str, np.ndarray]:
    npix = hp.nside2npix(nside)
    out = {
        key: np.zeros(npix, dtype=np.float32)
        for key in (
            "map_ymap",
            "map_ksz",
            "map_tau",
            "map_kappa_cmb",
            "map_kappa_wl",
            "map_kappa_wl_tomo2",
            "map_kappa_wl_tomo3",
            "map_kappa_wl_tomo4",
        )
    }
    for key, value in field_maps.items():
        out[key] = np.asarray(value, dtype=np.float32)
    return out


def harmonic_blend_map(base: np.ndarray, replacement: np.ndarray, nside: int, ell_cut: int, *, width_fraction: float = 0.3) -> np.ndarray:
    """Use replacement modes at low ell and base modes at high ell."""

    lmax = 3 * int(nside) - 1
    ell = np.arange(lmax + 1, dtype=np.float64)
    lo = max(2.0, float(ell_cut) * (1.0 - width_fraction))
    hi = max(lo + 1.0, float(ell_cut) * (1.0 + width_fraction))
    taper = np.zeros(lmax + 1, dtype=np.float64)
    taper[ell <= lo] = 1.0
    mid = (ell > lo) & (ell < hi)
    taper[mid] = 0.5 * (1.0 + np.cos(np.pi * (ell[mid] - lo) / (hi - lo)))

    base_alm = hp.map2alm(np.asarray(base, dtype=np.float64), lmax=lmax, iter=0)
    delta_alm = hp.map2alm(np.asarray(replacement - base, dtype=np.float64), lmax=lmax, iter=0)
    hp.almxfl(delta_alm, taper, inplace=True)
    blended = hp.alm2map(base_alm + delta_alm, nside, lmax=lmax, verbose=False)
    return np.asarray(blended, dtype=np.float32)


def harmonic_blend_maps(
    base_maps: Mapping[str, np.ndarray],
    replacement_maps: Mapping[str, np.ndarray],
    nside: int,
    ell_cut: int,
    fields: Sequence[str] = ("map_kappa_cmb", "map_kappa_wl", "map_tau", "map_ksz"),
) -> Dict[str, np.ndarray]:
    out = {key: np.array(value, copy=True) for key, value in base_maps.items()}
    for key in fields:
        if key in base_maps and key in replacement_maps:
            out[key] = harmonic_blend_map(base_maps[key], replacement_maps[key], nside, ell_cut)
    return out


def fit_residual_amplitude(
    ell: np.ndarray,
    base_cross: np.ndarray,
    residual_cross: np.ndarray,
    theory_cross: np.ndarray,
    train_range: Tuple[float, float],
) -> dict:
    ell = np.asarray(ell)
    base_cross = np.asarray(base_cross)
    residual_cross = np.asarray(residual_cross)
    theory_cross = np.asarray(theory_cross)
    mask = (
        np.isfinite(base_cross)
        & np.isfinite(residual_cross)
        & np.isfinite(theory_cross)
        & (ell >= float(train_range[0]))
        & (ell <= float(train_range[1]))
        & (np.abs(residual_cross) > 0)
    )
    if np.count_nonzero(mask) < 3:
        return {"amplitude": np.nan, "n_train": int(np.count_nonzero(mask)), "rms_frac_after": np.nan}
    x = residual_cross[mask]
    y = theory_cross[mask] - base_cross[mask]
    amp = float(np.sum(x * y) / np.sum(x * x))
    pred = base_cross[mask] + amp * x
    denom = np.maximum(np.abs(theory_cross[mask]), 1.0e-30)
    return {
        "amplitude": amp,
        "n_train": int(np.count_nonzero(mask)),
        "rms_frac_after": float(np.sqrt(np.mean(((pred - theory_cross[mask]) / denom) ** 2))),
    }


def bin_spectrum(
    ell: np.ndarray,
    values: np.ndarray,
    *,
    ell_min: int = 10,
    ell_max: int = 3000,
    delta_ell: int = 20,
) -> dict:
    """Average a spectrum in linear ell bins."""

    ell = np.asarray(ell, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    edges = np.arange(int(ell_min), int(ell_max) + int(delta_ell), int(delta_ell), dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(len(centers), np.nan, dtype=np.float64)
    counts = np.zeros(len(centers), dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (ell >= lo) & (ell < hi) & np.isfinite(values)
        counts[i] = int(np.count_nonzero(mask))
        if counts[i]:
            mean[i] = float(np.mean(values[mask]))
    keep = counts > 0
    return {"ell": centers[keep], "value": mean[keep], "counts": counts[keep], "edges": edges}


def bin_spectrum_pair(
    ell: np.ndarray,
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    ell_min: int = 10,
    ell_max: int = 3000,
    delta_ell: int = 20,
) -> dict:
    """Average two spectra using the same finite-value mask in each ell bin."""

    ell = np.asarray(ell, dtype=np.float64)
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    edges = np.arange(int(ell_min), int(ell_max) + int(delta_ell), int(delta_ell), dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_a = np.full(len(centers), np.nan, dtype=np.float64)
    mean_b = np.full(len(centers), np.nan, dtype=np.float64)
    ratio = np.full(len(centers), np.nan, dtype=np.float64)
    counts = np.zeros(len(centers), dtype=np.int64)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (ell >= lo) & (ell < hi) & np.isfinite(values_a) & np.isfinite(values_b)
        counts[i] = int(np.count_nonzero(mask))
        if counts[i]:
            mean_a[i] = float(np.mean(values_a[mask]))
            mean_b[i] = float(np.mean(values_b[mask]))
            if mean_b[i] != 0.0:
                ratio[i] = mean_a[i] / mean_b[i]
    keep = counts > 0
    return {
        "ell": centers[keep],
        "value_a": mean_a[keep],
        "value_b": mean_b[keep],
        "ratio": ratio[keep],
        "counts": counts[keep],
        "edges": edges,
    }


class DmoTemplatePainter:
    """Paint selected-halo DMO/NFW-like mass templates with GODMAX machinery."""

    def __init__(self, config_path: Path | str, catalog_key: str, nside: int):
        import jax
        import jax.numpy as jnp
        from base_class import base_class
        from get_radial_profiles import Profiles
        from get_sim_maps import setup_sim_map, get_sim_map

        self.jax = jax
        self.jnp = jnp
        self.get_sim_map = get_sim_map
        self.config_path = Path(config_path)
        self.config = aph.load_config(config_path)
        self.catalog_key = catalog_key
        self.nside = int(nside)
        self.max_paint = float(self.config["pasting"]["max_paint_R200c_factor"])
        _, attrs = aph.load_halo_catalog(aph.catalog_path(self.config, catalog_key), indices=np.asarray([], dtype=np.int64))
        self.sim_params, self.halo_params, self.analysis, self.other_params = aph.prepare_godmax_config(
            self.config,
            attrs,
            is_cmb_lensing=False,
            z_max=float(attrs.get("z_max", 0.5)),
            log10_mass_min=float(attrs.get("log10_m_min_hmsun", 14.0)),
        )
        self.base = base_class(self.sim_params, self.halo_params, self.analysis, self.other_params)
        self.profiles = Profiles(self.sim_params, self.halo_params, self.analysis, self.other_params, base_class_obj=self.base)
        self.setup_params = {
            "nside": self.nside,
            "smooth_profiles": bool(self.config["pasting"].get("smooth_profiles", True)),
            "profile_timing": False,
            "get_galmap": False,
            "get_ymap": False,
            "get_kSZmap": False,
            "get_taumap": False,
            "get_kappamap": True,
            "get_baryonifiedmap": True,
            "kappa_source_bin": 0,
        }
        self.setup = setup_sim_map(
            self.sim_params,
            self.halo_params,
            self.analysis,
            self.other_params,
            self.setup_params,
            Profiles_obj=self.profiles,
        )

    def _mock_params_for_pixels(self, catalog: Mapping[str, np.ndarray], pixels: Mapping[str, np.ndarray]) -> dict:
        params = dict(self.setup_params)
        params.update(
            {
                "get_galmap": False,
                "get_ymap": False,
                "get_kSZmap": False,
                "get_taumap": False,
                "get_kappamap": False,
                "get_baryonifiedmap": False,
                "halo_z": self.jnp.array(catalog["z"], dtype=self.jnp.float32),
                "halo_ra": self.jnp.array(catalog["ra_deg"], dtype=self.jnp.float32),
                "halo_dec": self.jnp.array(catalog["dec_deg"], dtype=self.jnp.float32),
                "halo_M": self.jnp.array(catalog["M200c_hMsun"], dtype=self.jnp.float64),
                "halo_vlos": self.jnp.array(catalog["vlos_kms"], dtype=self.jnp.float32),
                "nearby_pix_all": self.jnp.array(pixels["nearby_pix_all"]),
                "pix_prop_all": self.jnp.array(
                    [np.log(pixels["distances"]), pixels["z"], pixels["logM"], pixels["vlos"]],
                    dtype=self.jnp.float32,
                ).T,
                "start_ind": self.jnp.array(pixels["start_ind"], dtype=self.jnp.int32),
                "end_ind": self.jnp.array(pixels["end_ind"], dtype=self.jnp.int32),
                "ang_distance_all": self.jnp.array(pixels["ang_distance_all"], dtype=self.jnp.float32),
                "rp_max_all": self.jnp.array(pixels["rp_max_all"], dtype=self.jnp.float32),
                "random_seed": int(self.config["pasting"].get("random_seed", 42)),
            }
        )
        return params

    def _dmo_pixel_values(self, catalog: Mapping[str, np.ndarray], pixels: Mapping[str, np.ndarray]) -> np.ndarray:
        params = self._mock_params_for_pixels(catalog, pixels)
        mock = self.get_sim_map(self.sim_params, self.halo_params, self.analysis, self.other_params, params, Profiles_obj=self.setup)
        values = mock._chunked_vmap(mock.get_rhom_dmo_healpix, len(mock.pix_prop_all))
        del mock, params
        self.jax.clear_caches()
        gc.collect()
        return np.nan_to_num(values).astype(np.float32)

    def _paint_mass_map(self, catalog: Mapping[str, np.ndarray]) -> np.ndarray:
        if len(catalog["z"]) == 0:
            return np.zeros(hp.nside2npix(self.nside), dtype=np.float32)
        pixels = aph.build_pixel_work_package(
            catalog,
            self.nside,
            self.max_paint,
            int(self.config["pasting"].get("pixel_batch_size", 2000)),
        )
        if pixels is None:
            return np.zeros(hp.nside2npix(self.nside), dtype=np.float32)
        values = self._dmo_pixel_values(catalog, pixels)
        mass_map = np.zeros(hp.nside2npix(self.nside), dtype=np.float32)
        np.add.at(mass_map, pixels["nearby_pix_all"], values)
        del values, pixels
        gc.collect()
        return mass_map

    @staticmethod
    def _subset_catalog(catalog: Mapping[str, np.ndarray], mask: np.ndarray) -> dict:
        return {key: np.asarray(value)[mask] for key, value in catalog.items()}

    def _chunk_size(self) -> int:
        mapping = self.config["pasting"].get("chunk_halos_by_nside", {})
        if self.nside in mapping:
            return int(mapping[self.nside])
        if str(self.nside) in mapping:
            return int(mapping[str(self.nside)])
        return 50000

    def paint_shell_templates(
        self,
        catalog: Mapping[str, np.ndarray],
        shell_meta: Sequence[Mapping[str, object]],
        cache_root: Path | str,
        *,
        overwrite: bool = False,
        velocity_bins: int = 1,
    ) -> Dict[str, dict]:
        """Paint all requested shell templates in halo chunks.

        This avoids the expensive old pattern of rebuilding pixel work and
        launching a GODMAX map evaluation separately for every redshift shell.
        The momentum template uses each halo's own LOS velocity, so
        ``velocity_bins`` is kept only for cache-path compatibility.
        """

        out: Dict[str, dict] = {}
        missing = []
        for meta in shell_meta:
            path = nfw_template_cache_path(cache_root, self.catalog_key, meta, self.nside, velocity_bins)
            if path.exists() and not overwrite:
                with h5py.File(path, "r") as handle:
                    attrs = dict(handle.attrs)
                    is_current_bulk = bool(attrs.get("bulk_template", False)) and attrs.get("momentum_kind", "") == "DMO profile weighted by each halo LOS velocity"
                    if is_current_bulk:
                        out[str(meta["step_id"])] = {
                            "count_template": handle["count_template"][:],
                            "momentum_template": handle["momentum_template"][:],
                            "attrs": attrs,
                        }
                    else:
                        missing.append(dict(meta))
            else:
                missing.append(dict(meta))
        if not missing:
            return out

        npix = hp.nside2npix(self.nside)
        nshell = len(missing)
        raw_mass_maps = np.zeros((nshell, npix), dtype=np.float32)
        raw_momentum_maps = np.zeros((nshell, npix), dtype=np.float32)
        shell_halo_count = np.zeros(nshell, dtype=np.int64)
        shell_mass_sum = np.zeros(nshell, dtype=np.float64)
        halo_shell = np.full(len(catalog["z"]), -1, dtype=np.int32)

        z = np.asarray(catalog["z"])
        mass = np.asarray(catalog["M200c_hMsun"], dtype=np.float64)
        for ishell, meta in enumerate(missing):
            mask = (z >= float(meta["z_lo"])) & (z < float(meta["z_hi"]))
            halo_shell[mask] = ishell
            shell_halo_count[ishell] = int(np.count_nonzero(mask))
            shell_mass_sum[ishell] = float(np.sum(mass[mask]))

        selected = np.where(halo_shell >= 0)[0]
        chunk_size = self._chunk_size()
        for ichunk, start in enumerate(range(0, len(selected), chunk_size), 1):
            idx = selected[start : start + chunk_size]
            if len(idx) == 0:
                continue
            chunk = self._subset_catalog(catalog, idx)
            chunk_shell = halo_shell[idx]
            print(f"[nfw-template] chunk {ichunk}: halos {start:,}:{start + len(idx):,} pixel work")
            pixels = aph.build_pixel_work_package(
                chunk,
                self.nside,
                self.max_paint,
                int(self.config["pasting"].get("pixel_batch_size", 2000)),
            )
            if pixels is None:
                continue

            values = self._dmo_pixel_values(chunk, pixels)
            lengths = np.asarray(pixels["end_ind"], dtype=np.int64) - np.asarray(pixels["start_ind"], dtype=np.int64)
            pair_shell = np.repeat(chunk_shell, lengths)
            pair_vlos = np.repeat(np.asarray(chunk["vlos_kms"], dtype=np.float32), lengths)
            pix = np.asarray(pixels["nearby_pix_all"], dtype=np.int64)

            for ishell in np.unique(pair_shell):
                mask = pair_shell == ishell
                np.add.at(raw_mass_maps[ishell], pix[mask], values[mask])
                np.add.at(raw_momentum_maps[ishell], pix[mask], values[mask] * pair_vlos[mask])

            del pixels, values, lengths, pair_shell, pair_vlos, pix, chunk
            self.jax.clear_caches()
            gc.collect()

        for ishell, meta in enumerate(missing):
            raw_sum = float(np.sum(raw_mass_maps[ishell], dtype=np.float64))
            if raw_sum > 0.0 and shell_mass_sum[ishell] > 0.0:
                norm = shell_mass_sum[ishell] / raw_sum / float(meta["particle_mass_hMsun"])
                count_template = (raw_mass_maps[ishell] * norm).astype(np.float32)
                momentum_template = (raw_momentum_maps[ishell] * norm).astype(np.float32)
            else:
                count_template = np.zeros(npix, dtype=np.float32)
                momentum_template = np.zeros(npix, dtype=np.float32)

            attrs = {
                "step_id": meta["step_id"],
                "catalog_key": self.catalog_key,
                "nside": self.nside,
                "velocity_bins": int(velocity_bins),
                "n_halos": int(shell_halo_count[ishell]),
                "mass_sum_hMsun": float(shell_mass_sum[ishell]),
                "raw_mass_template_sum": raw_sum,
                "count_template_sum": float(np.sum(count_template, dtype=np.float64)),
                "expected_count_sum": float(shell_mass_sum[ishell] / float(meta["particle_mass_hMsun"])) if shell_mass_sum[ishell] > 0 else 0.0,
                "template_kind": "bulk GODMAX DMO projected mass, normalized to selected halo M200c sum",
                "momentum_kind": "DMO profile weighted by each halo LOS velocity",
                "bulk_template": True,
            }
            path = nfw_template_cache_path(cache_root, self.catalog_key, meta, self.nside, velocity_bins)
            _write_hdf5_atomic(path, {"count_template": count_template, "momentum_template": momentum_template}, attrs)
            out[str(meta["step_id"])] = {"count_template": count_template, "momentum_template": momentum_template, "attrs": attrs}

        del raw_mass_maps, raw_momentum_maps, halo_shell
        gc.collect()
        return out

    def paint_shell_template(
        self,
        catalog: Mapping[str, np.ndarray],
        shell_meta: Mapping[str, object],
        cache_root: Path | str,
        *,
        overwrite: bool = False,
        velocity_bins: int = 1,
    ) -> dict:
        path = nfw_template_cache_path(cache_root, self.catalog_key, shell_meta, self.nside, velocity_bins)
        if path.exists() and not overwrite:
            with h5py.File(path, "r") as handle:
                return {
                    "count_template": handle["count_template"][:],
                    "momentum_template": handle["momentum_template"][:],
                    "attrs": dict(handle.attrs),
                }

        mask = (catalog["z"] >= float(shell_meta["z_lo"])) & (catalog["z"] < float(shell_meta["z_hi"]))
        shell_catalog = self._subset_catalog(catalog, mask)
        npix = hp.nside2npix(self.nside)
        count_template = np.zeros(npix, dtype=np.float32)
        momentum_template = np.zeros(npix, dtype=np.float32)
        n_halos = len(shell_catalog["z"])
        mass_sum = float(np.sum(shell_catalog["M200c_hMsun"])) if n_halos else 0.0
        if n_halos:
            mass_map = self._paint_mass_map(shell_catalog)
            raw_sum = float(np.sum(mass_map, dtype=np.float64))
            if raw_sum > 0 and mass_sum > 0:
                count_template = (mass_map * (mass_sum / raw_sum) / float(shell_meta["particle_mass_hMsun"])).astype(np.float32)

            if int(velocity_bins) <= 1:
                v_mean = float(np.average(shell_catalog["vlos_kms"], weights=shell_catalog["M200c_hMsun"]))
                momentum_template = (count_template * v_mean).astype(np.float32)
            else:
                momentum_template = self._paint_velocity_binned_momentum(shell_catalog, shell_meta, int(velocity_bins))

        attrs = {
            "step_id": shell_meta["step_id"],
            "catalog_key": self.catalog_key,
            "nside": self.nside,
            "velocity_bins": int(velocity_bins),
            "n_halos": int(n_halos),
            "mass_sum_hMsun": mass_sum,
            "count_template_sum": float(np.sum(count_template, dtype=np.float64)),
            "expected_count_sum": mass_sum / float(shell_meta["particle_mass_hMsun"]) if mass_sum else 0.0,
            "template_kind": "GODMAX DMO projected mass, normalized to selected halo M200c sum",
        }
        _write_hdf5_atomic(path, {"count_template": count_template, "momentum_template": momentum_template}, attrs)
        return {"count_template": count_template, "momentum_template": momentum_template, "attrs": attrs}

    def _paint_velocity_binned_momentum(
        self,
        shell_catalog: Mapping[str, np.ndarray],
        shell_meta: Mapping[str, object],
        velocity_bins: int,
    ) -> np.ndarray:
        momentum_template = np.zeros(hp.nside2npix(self.nside), dtype=np.float32)
        vlos = np.asarray(shell_catalog["vlos_kms"], dtype=np.float64)
        if len(vlos) == 0:
            return momentum_template
        edges = np.unique(np.quantile(vlos, np.linspace(0.0, 1.0, velocity_bins + 1)))
        if len(edges) <= 2:
            v_mean = float(np.average(vlos, weights=shell_catalog["M200c_hMsun"]))
            mass_map = self._paint_mass_map(shell_catalog)
            raw_sum = float(np.sum(mass_map, dtype=np.float64))
            mass_sum = float(np.sum(shell_catalog["M200c_hMsun"]))
            if raw_sum > 0 and mass_sum > 0:
                return (mass_map * (mass_sum / raw_sum) / float(shell_meta["particle_mass_hMsun"]) * v_mean).astype(np.float32)
            return momentum_template

        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (vlos >= lo) & (vlos <= hi if i == len(edges) - 2 else vlos < hi)
            if not np.any(mask):
                continue
            subcat = self._subset_catalog(shell_catalog, mask)
            mass_sum = float(np.sum(subcat["M200c_hMsun"]))
            mass_map = self._paint_mass_map(subcat)
            raw_sum = float(np.sum(mass_map, dtype=np.float64))
            if raw_sum <= 0 or mass_sum <= 0:
                continue
            count_map = mass_map * (mass_sum / raw_sum) / float(shell_meta["particle_mass_hMsun"])
            v_mean = float(np.average(subcat["vlos_kms"], weights=subcat["M200c_hMsun"]))
            momentum_template += (count_map * v_mean).astype(np.float32)
        return momentum_template
