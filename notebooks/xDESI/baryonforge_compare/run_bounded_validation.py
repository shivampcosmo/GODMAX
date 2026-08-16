#!/usr/bin/env python
"""Run and plot a matched 64-halo bounded BaryonForge--GODMAX validation.

This is an explicitly non-production diagnostic.  It selects 64 halo centers
inside the configured 600 deg2 cap, stratified across four mass bins, paints
the exact same small catalog with both native backends at NSIDE 1024, and
saves common-window maps, aperture summaries, profiles, and low-resolution
NaMaster bandpowers.  It never calls or weakens the production statistics
gate in :mod:`measure_statistics`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import healpy as hp
import matplotlib
import numpy as np
import pymaster as nmt
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, SymLogNorm  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from common import (  # noqa: E402
    canonical_json,
    cap_mask,
    current_map_contract,
    load_config,
    read_map_file,
    resolve_path,
    sha256_file,
)
from measure_statistics import (  # noqa: E402
    SPECTRUM_SPECS,
    _make_bins,
    _masked_map,
)


SCHEMA = "baryonforge_godmax_bounded_validation_v1"
LABEL = "64-halo bounded smoke; not production statistics"
MASS_EDGES = np.asarray([13.0, 13.5, 14.0, 14.5, 16.0], dtype=np.float64)
HALOS_PER_MASS_BIN = 16
N_HALOS = 64
PAINT_NSIDE = 1024
DIAGNOSTIC_NSIDE = 128
DIAGNOSTIC_LMAX = 256
DIAGNOSTIC_N_BINS = 8
MAP_KEYS = ("map_ymap", "map_kappa_cmb")


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def angular_separation_deg(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    center_ra_deg: float,
    center_dec_deg: float,
) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    center_ra = math.radians(float(center_ra_deg))
    center_dec = math.radians(float(center_dec_deg))
    cosine = np.sin(dec) * math.sin(center_dec) + np.cos(dec) * math.cos(
        center_dec
    ) * np.cos(ra - center_ra)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def select_stratified_inner_cap_indices(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    mass_hmsun: np.ndarray,
    *,
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
    seed: int,
    mass_edges: np.ndarray = MASS_EDGES,
    halos_per_bin: int = HALOS_PER_MASS_BIN,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a deterministic, mass-stratified sample wholly inside the cap."""

    separation = angular_separation_deg(ra_deg, dec_deg, center_ra_deg, center_dec_deg)
    log_mass = np.log10(np.asarray(mass_hmsun, dtype=np.float64))
    inside = separation <= float(radius_deg)
    rng = np.random.default_rng(int(seed))
    selections = []
    counts = []
    for lower, upper in zip(mass_edges[:-1], mass_edges[1:]):
        candidates = np.flatnonzero(
            inside & (log_mass >= float(lower)) & (log_mass < float(upper))
        )
        if candidates.size < int(halos_per_bin):
            raise ValueError(
                f"Mass bin [{lower}, {upper}) has {candidates.size} inner-cap halos; "
                f"cannot select {halos_per_bin}."
            )
        chosen = np.sort(rng.choice(candidates, size=int(halos_per_bin), replace=False))
        selections.append(chosen)
        counts.append(int(candidates.size))
    selected = np.sort(np.concatenate(selections)).astype(np.int64)
    if selected.size != int(halos_per_bin) * (len(mass_edges) - 1):
        raise RuntimeError("Internal bounded-selection count mismatch.")
    if not np.all(separation[selected] <= float(radius_deg)):
        raise RuntimeError("Bounded selection contains a center outside the inner cap.")
    return selected, {
        "algorithm": "fixed-seed uniform-without-replacement within four log10 mass bins",
        "seed": int(seed),
        "mass_edges_log10_hMsun": np.asarray(mass_edges),
        "halos_per_mass_bin": int(halos_per_bin),
        "candidate_count_per_mass_bin": counts,
        "selected_parent_index_sha256": array_sha256(selected.astype("<i8")),
        "maximum_center_separation_deg": float(np.max(separation[selected])),
    }


def validation_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    root = resolve_path(config["project"]["output_root"], config["_config_path"])
    base = root / "validation" / "smoke64"
    catalog_name = "bounded_innercap_stratified_M200cgt1e13_64halos.h5"
    catalog = base / "catalog" / catalog_name
    derived_config = base / "config" / "bounded_smoke64_cpu.yaml"
    godmax_map = (
        base
        / "maps"
        / "godmax_native"
        / "abacus_pasted_maps_buffered_mgt13_nside1024_split000of001.h5"
    )
    baryonforge_map = base / "maps" / "baryonforge_native_nside1024_smoke64.h5"
    return {
        "base": base,
        "catalog": catalog,
        "config": derived_config,
        "godmax_map": godmax_map,
        "baryonforge_map": baryonforge_map,
        "diagnostics": base / "smoke64_diagnostics.h5",
        "figures": base / "figures",
        "logs": base / "logs",
        "manifest": base / "plot_manifest.json",
    }


def _replace_or_refuse(path: Path, temporary: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    os.replace(temporary, path)


def write_subset_catalog(
    config: Mapping[str, Any], output: Path, *, overwrite: bool
) -> dict[str, Any]:
    parent = resolve_path(config["catalog"]["output_h5"], config["_config_path"])
    if not parent.exists():
        raise FileNotFoundError(parent)
    parent_sha = sha256_file(parent)
    sky = config["sky_patch"]
    seed = int(config["pasting"]["random_seed"])
    with h5py.File(parent, "r") as source:
        selected, selection = select_stratified_inner_cap_indices(
            source["ra_deg"][:],
            source["dec_deg"][:],
            source["M200c_hMsun"][:],
            center_ra_deg=float(sky["center_ra_deg"]),
            center_dec_deg=float(sky["center_dec_deg"]),
            radius_deg=float(sky["radius_deg"]),
            seed=seed,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            with h5py.File(temporary, "w") as target:
                for key, value in source.attrs.items():
                    target.attrs[key] = value
                for name, dataset in source.items():
                    target.create_dataset(
                        name,
                        data=np.asarray(dataset[selected]),
                        compression="lzf",
                    )
                target.create_dataset(
                    "parent_catalog_index", data=selected, compression="lzf"
                )
                selected_mass = np.asarray(
                    source["M200c_hMsun"][selected], dtype=np.float64
                )
                selected_z = np.asarray(source["z"][selected], dtype=np.float64)
                target.attrs.update(
                    {
                        "comparison_schema": "baryonforge_godmax_bounded_catalog_v1",
                        "catalog_key": "bounded_innercap_stratified_smoke64",
                        "created_utc": utc_now(),
                        "updated_utc": utc_now(),
                        "selection_rows": int(selected.size),
                        "n_halos": int(selected.size),
                        "mass_min_hMsun": float(np.min(selected_mass)),
                        "mass_max_hMsun": float(np.max(selected_mass)),
                        "z_min": float(np.min(selected_z)),
                        "z_max": float(np.max(selected_z)),
                        "catalog_selection_radius_deg": float(sky["radius_deg"]),
                        "retains_catalog_edge_buffer": False,
                        "bounded_smoke_only": True,
                        "bounded_label": LABEL,
                        "bounded_parent_catalog_path": str(parent),
                        "bounded_parent_catalog_sha256": parent_sha,
                        "bounded_parent_catalog_rows": int(source["z"].shape[0]),
                        "bounded_selection_json": canonical_json(selection),
                    }
                )
            _replace_or_refuse(output, temporary, overwrite)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return {
        **selection,
        "path": str(output),
        "sha256": sha256_file(output),
        "parent_path": str(parent),
        "parent_sha256": parent_sha,
        "parent_indices": selected,
        "source_rows": _read_dataset(output, "source_row"),
    }


def _read_dataset(path: Path, name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[name][:])


def write_derived_config(
    primary: Mapping[str, Any],
    paths: Mapping[str, Path],
    selection: Mapping[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    derived = copy.deepcopy(dict(primary))
    derived.pop("_config_path", None)
    relative_base = "validation/smoke64"
    derived["project"]["catalog_subdir"] = f"{relative_base}/catalog"
    derived["project"]["map_subdir"] = f"{relative_base}/maps"
    derived["catalog"]["output_h5"] = str(paths["catalog"])
    derived["catalog"]["expected_selected_count"] = N_HALOS
    derived["catalog"]["retain_outer_buffer_halos"] = False
    derived["catalogs"]["buffered_mgt13"]["output_name"] = paths["catalog"].name
    derived["catalogs"]["buffered_mgt13"]["metadata"]["bounded_smoke_only"] = True
    derived["catalogs"]["buffered_mgt13"]["metadata"]["retain_outer_buffer_halos"] = (
        False
    )
    derived["baryonforge"]["output_h5"] = str(paths["baryonforge_map"])
    derived["pasting"]["run_name"] = "godmax_native"
    derived["pasting"]["verbose"] = False
    derived["pasting"]["chunk_halos_by_nside"][PAINT_NSIDE] = N_HALOS
    derived["pasting"]["num_splits_by_nside"][PAINT_NSIDE] = 1
    derived["pasting"]["jax"]["platforms"] = "cpu"
    derived["pasting"]["jax"]["preallocate"] = False
    derived["bounded_validation"] = {
        "schema": SCHEMA,
        "label": LABEL,
        "primary_config_path": str(primary["_config_path"]),
        "primary_config_sha256": sha256_file(primary["_config_path"]),
        "parent_catalog_path": selection["parent_path"],
        "parent_catalog_sha256": selection["parent_sha256"],
        "selected_parent_index_sha256": selection["selected_parent_index_sha256"],
        "selection_seed": int(selection["seed"]),
        "mass_edges_log10_hMsun": MASS_EDGES.tolist(),
        "halos_per_mass_bin": HALOS_PER_MASS_BIN,
        "paint_nside": PAINT_NSIDE,
        "diagnostic_nside": DIAGNOSTIC_NSIDE,
        "diagnostic_lmax": DIAGNOSTIC_LMAX,
        "production_statistics_eligible": False,
    }
    output = paths["config"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(derived, handle, sort_keys=False)
    _replace_or_refuse(output, temporary, overwrite)
    return load_config(output)


def prepare(
    primary: Mapping[str, Any], paths: Mapping[str, Path], *, overwrite: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    if paths["catalog"].exists() and paths["config"].exists() and not overwrite:
        selection = {
            "path": str(paths["catalog"]),
            "sha256": sha256_file(paths["catalog"]),
        }
        with h5py.File(paths["catalog"], "r") as handle:
            selection.update(json.loads(handle.attrs["bounded_selection_json"]))
            selection["parent_path"] = handle.attrs["bounded_parent_catalog_path"]
            selection["parent_sha256"] = handle.attrs["bounded_parent_catalog_sha256"]
            selection["parent_indices"] = np.asarray(
                handle["parent_catalog_index"][:], dtype=np.int64
            )
            selection["source_rows"] = np.asarray(
                handle["source_row"][:], dtype=np.int64
            )
        if sha256_file(selection["parent_path"]) != selection["parent_sha256"]:
            raise RuntimeError("Prepared subset parent catalog has changed.")
        bounded = load_config(paths["config"])
        if (
            resolve_path(bounded["catalog"]["output_h5"], paths["config"])
            != paths["catalog"]
        ):
            raise RuntimeError("Prepared bounded config points to a different catalog.")
        return bounded, selection
    if paths["catalog"].exists() != paths["config"].exists() and not overwrite:
        raise FileExistsError(
            "Only one prepared artifact exists; pass --overwrite to rebuild the pair."
        )
    selection = write_subset_catalog(primary, paths["catalog"], overwrite=overwrite)
    bounded = write_derived_config(primary, paths, selection, overwrite=overwrite)
    return bounded, selection


def run_logged(command: list[str], log_path: Path, *, env: Mapping[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dict(env),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
        log.write(f"EXIT_CODE {code}\n")
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def run_painters(
    bounded: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    overwrite: bool,
) -> None:
    script_dir = Path(__file__).resolve().parent
    environment = dict(os.environ)
    environment.update(
        {
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "JAX_ENABLE_X64": "True",
            "JAX_PLATFORMS": "cpu",
            "PASTE_JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    godmax_command = [
        sys.executable,
        str(script_dir / "paint_godmax.py"),
        "--config",
        str(paths["config"]),
        "--nside",
        str(PAINT_NSIDE),
        "--split-index",
        "0",
        "--num-splits",
        "1",
        "--pixel-workers",
        "1",
    ]
    baryonforge_command = [
        sys.executable,
        str(script_dir / "paint_baryonforge.py"),
        "--config",
        str(paths["config"]),
        "--nside",
        str(PAINT_NSIDE),
        "--max-halos",
        str(N_HALOS),
        "--n-jobs",
        str(int(bounded["baryonforge"]["n_jobs"])),
        "--output",
        str(paths["baryonforge_map"]),
        "--no-verbose",
    ]
    if overwrite:
        godmax_command.append("--overwrite")
        baryonforge_command.append("--overwrite")
    run_logged(
        godmax_command,
        paths["logs"] / "paint_godmax.log",
        env=environment,
    )
    run_logged(
        baryonforge_command,
        paths["logs"] / "paint_baryonforge.log",
        env=environment,
    )


def _product_provenance(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    maps, attrs = read_map_file(path)
    attrs = dict(attrs)
    nested = attrs.get("provenance")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            attrs.setdefault(str(key), value)
    return {key: np.asarray(maps[key], dtype=np.float64) for key in MAP_KEYS}, attrs


def validate_products(
    bounded: Mapping[str, Any],
    selection: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]], dict[str, Any]]:
    expected = current_map_contract(bounded)
    products = {}
    metadata = {}
    for backend, key in (("godmax", "godmax_map"), ("baryonforge", "baryonforge_map")):
        path = paths[key]
        if not path.exists():
            raise FileNotFoundError(path)
        maps, attrs = _product_provenance(path)
        products[backend] = maps
        metadata[backend] = attrs
        if attrs.get("backend") != backend:
            raise ValueError(
                f"{path} backend is {attrs.get('backend')!r}, expected {backend!r}."
            )
        for name in (
            "catalog_sha256",
            "comparison_config_sha256",
            "source_manifest_sha256",
            "effective_godmax_config_sha256",
        ):
            if attrs.get(name) != expected[name]:
                raise ValueError(
                    f"{backend} {name} does not match the current bounded contract."
                )
        if int(attrs.get("nside", -1)) != PAINT_NSIDE:
            raise ValueError(f"{backend} map is not NSIDE {PAINT_NSIDE}.")
        if str(attrs.get("ordering", "")).upper() != "RING":
            raise ValueError(f"{backend} map is not RING ordered.")
        if int(attrs.get("n_halos_painted", -1)) != N_HALOS:
            raise ValueError(f"{backend} did not paint {N_HALOS} halos.")
        for field, values in maps.items():
            if values.size != hp.nside2npix(PAINT_NSIDE):
                raise ValueError(f"{backend}:{field} has invalid map length.")
            if not np.all(np.isfinite(values)) or np.count_nonzero(values) == 0:
                raise ValueError(
                    f"{backend}:{field} is non-finite or identically zero."
                )
    godmax = metadata["godmax"]
    if not (
        int(godmax.get("split_index", -1)) == 0
        and int(godmax.get("num_splits", -1)) == 1
        and int(godmax.get("split_start", -1)) == 0
        and int(godmax.get("split_stop", -1)) == N_HALOS
    ):
        raise ValueError("GODMAX bounded map is not the exact [0:64] one-split subset.")
    baryonforge = metadata["baryonforge"]
    if baryonforge.get("complete_catalog_paint") is not False:
        raise ValueError("BaryonForge bounded map must remain explicitly incomplete.")
    with h5py.File(paths["baryonforge_map"], "r") as handle:
        source_rows = np.asarray(handle["provenance/source_row"][:], dtype=np.int64)
    if not np.array_equal(
        source_rows, np.asarray(selection["source_rows"], dtype=np.int64)
    ):
        raise ValueError(
            "BaryonForge source-row identity differs from the bounded catalog."
        )
    return products, metadata, expected


def _pair_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    union = (reference != 0.0) | (candidate != 0.0)
    intersection = (reference != 0.0) & (candidate != 0.0)
    ref = reference[union]
    test = candidate[union]
    dot = float(np.dot(ref, test))
    ref_norm = float(np.dot(ref, ref))
    test_norm = float(np.dot(test, test))
    correlation = (
        float(np.corrcoef(ref, test)[0, 1])
        if ref.size > 1 and np.std(ref) > 0.0 and np.std(test) > 0.0
        else float("nan")
    )
    return {
        "difference_convention": "BaryonForge minus GODMAX",
        "union_nonzero_pixels": int(np.count_nonzero(union)),
        "intersection_nonzero_pixels": int(np.count_nonzero(intersection)),
        "footprint_jaccard": float(
            np.count_nonzero(intersection) / np.count_nonzero(union)
        ),
        "pearson_r_on_union": correlation,
        "cosine_similarity_on_union": (
            float(dot / math.sqrt(ref_norm * test_norm))
            if ref_norm > 0.0 and test_norm > 0.0
            else float("nan")
        ),
        "gain_through_origin_on_union": (
            float(dot / ref_norm) if ref_norm > 0.0 else float("nan")
        ),
        "global_sum_ratio": (
            float(np.sum(candidate) / np.sum(reference))
            if np.sum(reference) != 0.0
            else float("nan")
        ),
        "relative_l1": (
            float(np.sum(np.abs(candidate - reference)) / np.sum(np.abs(reference)))
            if np.sum(np.abs(reference)) > 0.0
            else float("nan")
        ),
    }


def aperture_diagnostics(
    catalog_path: Path,
    products: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    with h5py.File(catalog_path, "r") as handle:
        ra = np.asarray(handle["ra_deg"][:], dtype=np.float64)
        dec = np.asarray(handle["dec_deg"][:], dtype=np.float64)
        mass = np.asarray(handle["M200c_hMsun"][:], dtype=np.float64)
        redshift = np.asarray(handle["z"][:], dtype=np.float64)
        support = (
            5.0
            * np.asarray(handle["R200c_hMpc"][:], dtype=np.float64)
            / np.asarray(handle["DA_hMpc"][:], dtype=np.float64)
        )
        source_row = np.asarray(handle["source_row"][:], dtype=np.int64)
    vectors = np.asarray(hp.ang2vec(ra, dec, lonlat=True), dtype=np.float64)
    apertures = [
        hp.query_disc(PAINT_NSIDE, vector, float(radius), inclusive=False, nest=False)
        for vector, radius in zip(vectors, support)
    ]
    pair_angle = np.arccos(np.clip(vectors @ vectors.T, -1.0, 1.0))
    np.fill_diagonal(pair_angle, np.inf)
    isolated = np.all(pair_angle > support[:, None] + support[None, :], axis=1)
    out = {
        "mass_hMsun": mass,
        "redshift": redshift,
        "source_row": source_row,
        "support_angle_deg": np.degrees(support),
        "isolated_5R": isolated,
    }
    pixel_area = hp.nside2pixarea(PAINT_NSIDE)
    for field in MAP_KEYS:
        short = "y" if field == "map_ymap" else "kappa"
        for backend in ("godmax", "baryonforge"):
            values = products[backend][field]
            out[f"{backend}_{short}_aperture_integral_sr"] = np.asarray(
                [
                    np.sum(values[pixels], dtype=np.float64) * pixel_area
                    for pixels in apertures
                ]
            )
    return out


def bounded_spectra(
    config: Mapping[str, Any], products: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, Any]:
    sky = config["sky_patch"]
    degraded = {
        "godmax_y": hp.ud_grade(
            products["godmax"]["map_ymap"], DIAGNOSTIC_NSIDE, power=0
        ),
        "godmax_kappa": hp.ud_grade(
            products["godmax"]["map_kappa_cmb"], DIAGNOSTIC_NSIDE, power=0
        ),
        "baryonforge_y": hp.ud_grade(
            products["baryonforge"]["map_ymap"], DIAGNOSTIC_NSIDE, power=0
        ),
        "baryonforge_kappa": hp.ud_grade(
            products["baryonforge"]["map_kappa_cmb"], DIAGNOSTIC_NSIDE, power=0
        ),
    }
    binary = cap_mask(
        DIAGNOSTIC_NSIDE,
        float(sky["center_ra_deg"]),
        float(sky["center_dec_deg"]),
        float(sky["radius_deg"]),
    )
    apodized = np.asarray(nmt.mask_apodization(binary, 1.0, apotype="C2"))
    processed = {}
    means = {}
    for name, values in degraded.items():
        processed[name], means[name] = _masked_map(values, apodized, True)
    processed["residual_y"] = processed["baryonforge_y"] - processed["godmax_y"]
    processed["residual_kappa"] = (
        processed["baryonforge_kappa"] - processed["godmax_kappa"]
    )
    bins, ell_left, ell_right = _make_bins(
        8, DIAGNOSTIC_LMAX, DIAGNOSTIC_N_BINS, "linear"
    )
    fields = {
        name: nmt.NmtField(
            apodized,
            [values],
            spin=0,
            n_iter=0,
            n_iter_mask=0,
            lmax=DIAGNOSTIC_LMAX,
            lmax_mask=DIAGNOSTIC_LMAX,
            lite=True,
        )
        for name, values in processed.items()
    }
    workspace = nmt.NmtWorkspace.from_fields(
        fields["godmax_y"], fields["godmax_y"], bins
    )
    ell = np.asarray(bins.get_effective_ells(), dtype=np.float64)
    spectra = {}
    for name, field_a, field_b in SPECTRUM_SPECS:
        coupled = np.asarray(
            nmt.compute_coupled_cell(fields[field_a], fields[field_b])[0],
            dtype=np.float64,
        )
        cell = np.asarray(workspace.decouple_cell(coupled[None, :])[0])
        spectra[name] = {
            "cl": cell,
            "dell": ell * (ell + 1.0) * cell / (2.0 * np.pi),
        }
    return {
        "ell": ell,
        "ell_left": ell_left,
        "ell_right": ell_right,
        "binary_mask": binary,
        "apodized_mask": apodized,
        "degraded_maps": degraded,
        "subtracted_means": means,
        "spectra": spectra,
        "method": (
            "common C2 one-degree apodized cap, common scalar NaMaster workspace, "
            "identical NSIDE1024-to-128 scalar averaging, lmax=256; bounded smoke only"
        ),
    }


def write_diagnostics(
    path: Path,
    *,
    primary_config: Mapping[str, Any],
    bounded_config: Mapping[str, Any],
    paths: Mapping[str, Path],
    products: Mapping[str, Mapping[str, np.ndarray]],
    metadata: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Any],
    aperture: Mapping[str, np.ndarray],
    spectra: Mapping[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    pair_metrics = {
        "y": _pair_metrics(
            products["godmax"]["map_ymap"],
            products["baryonforge"]["map_ymap"],
        ),
        "kappa": _pair_metrics(
            products["godmax"]["map_kappa_cmb"],
            products["baryonforge"]["map_kappa_cmb"],
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            {
                "schema": SCHEMA,
                "label": LABEL,
                "created_utc": utc_now(),
                "production_statistics_eligible": False,
                "difference_convention": "BaryonForge minus GODMAX",
                "primary_config_path": str(primary_config["_config_path"]),
                "primary_config_sha256": sha256_file(primary_config["_config_path"]),
                "bounded_config_path": str(bounded_config["_config_path"]),
                "bounded_config_sha256": sha256_file(bounded_config["_config_path"]),
                "catalog_path": str(paths["catalog"]),
                "catalog_sha256": sha256_file(paths["catalog"]),
                "godmax_map_path": str(paths["godmax_map"]),
                "godmax_map_sha256": sha256_file(paths["godmax_map"]),
                "baryonforge_map_path": str(paths["baryonforge_map"]),
                "baryonforge_map_sha256": sha256_file(paths["baryonforge_map"]),
                "profile_path": str(primary_config["profiles"]["output_h5"]),
                "profile_sha256": sha256_file(primary_config["profiles"]["output_h5"]),
                "source_manifest_sha256": expected["source_manifest_sha256"],
                "map_metadata_json": canonical_json(metadata),
                "spectra_method": spectra["method"],
            }
        )
        selection_group = handle.create_group("selection")
        for key, values in aperture.items():
            selection_group.create_dataset(key, data=values, compression="lzf")
        metrics_group = handle.create_group("map_pair_metrics")
        for field, values in pair_metrics.items():
            child = metrics_group.create_group(field)
            for key, value in values.items():
                if isinstance(value, str):
                    child.attrs[key] = value
                else:
                    child.attrs[key] = value
        spectra_group = handle.create_group("spectra")
        spectra_group.create_dataset("ell", data=spectra["ell"])
        spectra_group.create_dataset("ell_left", data=spectra["ell_left"])
        spectra_group.create_dataset("ell_right", data=spectra["ell_right"])
        for name, values in spectra["spectra"].items():
            child = spectra_group.create_group(name)
            child.create_dataset("cl", data=values["cl"])
            child.create_dataset("dell", data=values["dell"])
        diagnostic_maps = handle.create_group("diagnostic_maps_nside128")
        diagnostic_maps.attrs["nside"] = DIAGNOSTIC_NSIDE
        diagnostic_maps.attrs["ordering"] = "RING"
        for name, values in spectra["degraded_maps"].items():
            diagnostic_maps.create_dataset(
                name, data=np.asarray(values, dtype=np.float32), compression="lzf"
            )
        diagnostic_maps.create_dataset(
            "binary_mask",
            data=spectra["binary_mask"].astype(np.uint8),
            compression="lzf",
        )
        diagnostic_maps.create_dataset(
            "apodized_mask",
            data=np.asarray(spectra["apodized_mask"], dtype=np.float32),
            compression="lzf",
        )
    _replace_or_refuse(path, temporary, overwrite)
    return pair_metrics


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension, kwargs in (
        ("png", {"dpi": 180}),
        ("pdf", {}),
    ):
        output = directory / f"{stem}.{extension}"
        temporary = directory / f".{stem}.tmp.{os.getpid()}.{extension}"
        fig.savefig(temporary, bbox_inches="tight", **kwargs)
        os.replace(temporary, output)
        outputs.append(output)
    plt.close(fig)
    return outputs


def _smoke_label(fig: plt.Figure) -> None:
    fig.text(
        0.995,
        0.005,
        LABEL,
        ha="right",
        va="bottom",
        fontsize=8,
        color="firebrick",
    )


def plot_catalog(
    primary: Mapping[str, Any], catalog_path: Path, figures: Path
) -> list[Path]:
    parent_path = resolve_path(primary["catalog"]["output_h5"], primary["_config_path"])
    with h5py.File(parent_path, "r") as handle:
        parent_ra = np.asarray(handle["ra_deg"][:])
        parent_dec = np.asarray(handle["dec_deg"][:])
        parent_mass = np.asarray(handle["M200c_hMsun"][:])
        parent_z = np.asarray(handle["z"][:])
    with h5py.File(catalog_path, "r") as handle:
        ra = np.asarray(handle["ra_deg"][:])
        dec = np.asarray(handle["dec_deg"][:])
        mass = np.asarray(handle["M200c_hMsun"][:])
        redshift = np.asarray(handle["z"][:])
        support = (
            5.0 * np.asarray(handle["R200c_hMpc"][:]) / np.asarray(handle["DA_hMpc"][:])
        )
    sky = primary["sky_patch"]
    ra0 = float(sky["center_ra_deg"])
    dec0 = float(sky["center_dec_deg"])
    radius = float(sky["radius_deg"])
    rng = np.random.default_rng(int(primary["pasting"]["random_seed"]))
    sample = rng.choice(parent_ra.size, size=min(25000, parent_ra.size), replace=False)
    x_parent = ((parent_ra[sample] - ra0 + 180.0) % 360.0 - 180.0) * math.cos(
        math.radians(dec0)
    )
    y_parent = parent_dec[sample] - dec0
    x = ((ra - ra0 + 180.0) % 360.0 - 180.0) * math.cos(math.radians(dec0))
    y = dec - dec0
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(
        x_parent, y_parent, s=1, color="0.75", alpha=0.25, rasterized=True
    )
    points = axes[0, 0].scatter(
        x, y, c=np.log10(mass), s=34, cmap="viridis", edgecolor="k", linewidth=0.25
    )
    axes[0, 0].add_patch(
        Circle((0.0, 0.0), radius, fill=False, color="k", lw=1.2, label="inner cap")
    )
    axes[0, 0].add_patch(
        Circle(
            (0.0, 0.0),
            radius + float(sky["edge_buffer_deg"]),
            fill=False,
            color="0.4",
            ls="--",
            lw=1.0,
            label="catalog buffer",
        )
    )
    axes[0, 0].set(
        xlabel=r"$\Delta$RA cos(dec) [deg]",
        ylabel=r"$\Delta$dec [deg]",
        title="Parent catalog and selected centers",
        aspect="equal",
    )
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.colorbar(points, ax=axes[0, 0], label=r"$\log_{10}(M_{200c}/[M_\odot/h])$")
    axes[0, 1].hist(
        np.log10(parent_mass),
        bins=np.linspace(13, 16, 35),
        histtype="step",
        density=True,
        color="0.4",
        label="parent",
    )
    axes[0, 1].hist(
        np.log10(mass),
        bins=MASS_EDGES,
        histtype="stepfilled",
        alpha=0.45,
        density=True,
        label="selected 64",
    )
    axes[0, 1].set(
        xlabel=r"$\log_{10}(M_{200c}/[M_\odot/h])$",
        ylabel="density",
        title="Mass-stratified smoke selection",
    )
    axes[0, 1].legend()
    axes[1, 0].hist(
        parent_z, bins=35, histtype="step", density=True, color="0.4", label="parent"
    )
    axes[1, 0].hist(
        redshift,
        bins=12,
        histtype="stepfilled",
        alpha=0.45,
        density=True,
        label="selected 64",
    )
    axes[1, 0].set(xlabel="redshift", ylabel="density", title="Redshift coverage")
    axes[1, 0].legend()
    axes[1, 1].scatter(
        np.log10(mass),
        np.degrees(support),
        c=redshift,
        cmap="plasma",
        edgecolor="k",
        linewidth=0.25,
    )
    axes[1, 1].axhline(
        np.degrees(hp.max_pixrad(PAINT_NSIDE)),
        color="firebrick",
        ls="--",
        label="NSIDE1024 max pixel radius",
    )
    axes[1, 1].set(
        xlabel=r"$\log_{10}(M_{200c}/[M_\odot/h])$",
        ylabel="5R200c support [deg]",
        title="Native paint support safety",
    )
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Bounded Backlight catalog validation")
    _smoke_label(fig)
    return save_figure(fig, figures, "01_catalog_selection")


def _profile_group(profile: h5py.File, mass: float, redshift: float) -> h5py.Group:
    return profile[f"log10M{math.log10(mass):.2f}_z{redshift:.3f}"]


def plot_profiles(profile_path: Path, figures: Path) -> list[Path]:
    masses = (1.0e13, 1.0e14, 1.0e15)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(masses)))
    outputs = []
    with h5py.File(profile_path, "r") as profile:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="col")
        for mass, color in zip(masses, colors):
            group = _profile_group(profile, mass, 0.8)
            radius = group["radius_R200c"][:]
            label = rf"$10^{{{math.log10(mass):.0f}}}\,M_\odot/h$"
            for column, field in enumerate(("rho_gas", "rho_matter")):
                godmax = group[f"godmax/{field}"][:]
                baryonforge = group[f"baryonforge/{field}"][:]
                axes[0, column].loglog(radius, godmax, color=color, label=label)
                axes[0, column].loglog(radius, baryonforge, color=color, ls="--")
                axes[1, column].semilogx(radius, baryonforge / godmax, color=color)
        axes[0, 0].set(
            ylabel=r"$\rho_{gas}$ [native GODMAX units]",
            title="Gas: GODMAX solid, BaryonForge dashed",
        )
        axes[0, 1].set(
            ylabel=r"$\rho_{matter}$ [native GODMAX units]", title="Total matter"
        )
        for axis in axes[1]:
            axis.axhline(1.0, color="0.3", lw=1)
            axis.set(xlabel=r"$r/R_{200c}$", ylabel="BaryonForge / GODMAX")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Matched three-dimensional profiles at z=0.8")
        _smoke_label(fig)
        outputs.extend(save_figure(fig, figures, "02_profiles_3d"))

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="col")
        for mass, color in zip(masses, colors):
            group = _profile_group(profile, mass, 0.8)
            radius = group["radius_R200c"][:]
            label = rf"$10^{{{math.log10(mass):.0f}}}\,M_\odot/h$"
            for column, field in enumerate(("y_projected", "kappa_cmb")):
                godmax = group[f"godmax/{field}"][:]
                tabulated = group[f"baryonforge_tabulated_for_painter/{field}"][:]
                direct = group[f"baryonforge/{field}"][:]
                axes[0, column].loglog(radius, godmax, color=color, label=label)
                axes[0, column].loglog(radius, tabulated, color=color, ls="--")
                axes[0, column].loglog(radius, direct, color=color, ls=":", alpha=0.65)
                axes[1, column].semilogx(radius, tabulated / godmax, color=color)
        axes[0, 0].set(
            ylabel="dimensionless Compton-y",
            title="y: GODMAX solid, BF table dashed, BF direct dotted",
        )
        axes[0, 1].set(ylabel="dimensionless CMB convergence", title="CMB kappa")
        for axis in axes[1]:
            axis.axhline(1.0, color="0.3", lw=1)
            axis.set(xlabel=r"$R/R_{200c}$", ylabel="BF painter table / GODMAX")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Matched projected profiles at z=0.8")
        _smoke_label(fig)
        outputs.extend(save_figure(fig, figures, "03_profiles_projected"))

        fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)
        redshifts = (0.65, 0.8, 0.95)
        ratio_specs = (
            ("baryonforge_direct_over_godmax/rho_gas", "gas 3D"),
            ("baryonforge_direct_over_godmax/rho_matter", "matter 3D"),
            ("baryonforge_tabulated_over_godmax/y_projected", "y table"),
            ("baryonforge_tabulated_over_godmax/kappa_cmb", "kappa table"),
        )
        for row, mass in enumerate(masses):
            for column, redshift in enumerate(redshifts):
                group = _profile_group(profile, mass, redshift)
                radius = group["radius_R200c"][:]
                axis = axes[row, column]
                for field, label in ratio_specs:
                    axis.semilogx(radius, group[field][:], label=label)
                axis.axhline(1.0, color="0.3", lw=0.8)
                axis.set_ylim(0.45, 1.55)
                axis.set_title(rf"$M=10^{{{math.log10(mass):.0f}}}$, z={redshift:.2f}")
                if row == 2:
                    axis.set_xlabel(r"$R/R_{200c}$")
                if column == 0:
                    axis.set_ylabel("BaryonForge / GODMAX")
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle("Profile-ratio grid across all nine validation nodes")
        _smoke_label(fig)
        outputs.extend(save_figure(fig, figures, "04_profile_ratio_grid"))
    return outputs


def project_map(values: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    sky = config["sky_patch"]
    projector = hp.projector.GnomonicProj(
        rot=(float(sky["center_ra_deg"]), float(sky["center_dec_deg"]), 0.0),
        xsize=620,
        ysize=620,
        reso=3.0,
    )
    return np.asarray(
        projector.projmap(
            values,
            vec2pix_func=lambda x, y, z: hp.vec2pix(PAINT_NSIDE, x, y, z),
        ),
        dtype=np.float64,
    )


def plot_map_triplet(
    config: Mapping[str, Any],
    products: Mapping[str, Mapping[str, np.ndarray]],
    field: str,
    label: str,
    stem: str,
    figures: Path,
) -> list[Path]:
    godmax = project_map(products["godmax"][field], config)
    baryonforge = project_map(products["baryonforge"][field], config)
    residual = baryonforge - godmax
    positive = np.concatenate((godmax[godmax > 0.0], baryonforge[baryonforge > 0.0]))
    vmin = max(float(np.percentile(positive, 2.0)), float(np.max(positive)) * 1.0e-7)
    vmax = float(np.max(positive))
    residual_scale = float(np.max(np.abs(residual)))
    extent_deg = 620 * 3.0 / 120.0
    extent = (-extent_deg, extent_deg, -extent_deg, extent_deg)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.3))
    for axis, image, title in zip(
        axes[:2], (godmax, baryonforge), ("GODMAX", "BaryonForge")
    ):
        plotted = axis.imshow(
            np.where(image > 0.0, image, np.nan),
            origin="lower",
            extent=extent,
            cmap="magma",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        axis.set_title(title)
        fig.colorbar(plotted, ax=axis, fraction=0.042, pad=0.025, label=label)
    threshold = max(residual_scale * 1.0e-4, np.finfo(float).tiny)
    plotted = axes[2].imshow(
        residual,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=SymLogNorm(
            linthresh=threshold,
            vmin=-residual_scale,
            vmax=residual_scale,
        ),
    )
    axes[2].set_title("BaryonForge minus GODMAX")
    fig.colorbar(
        plotted,
        ax=axes[2],
        fraction=0.042,
        pad=0.025,
        label=f"residual {label}",
    )
    radius = float(config["sky_patch"]["radius_deg"])
    for index, axis in enumerate(axes):
        axis.add_patch(Circle((0.0, 0.0), radius, fill=False, color="cyan", lw=0.8))
        axis.set_xlabel("tangent-plane x [deg]")
        axis.set_ylabel("tangent-plane y [deg]" if index == 0 else "")
    fig.subplots_adjust(wspace=0.36)
    fig.suptitle(f"{label} maps on the common bounded catalog")
    _smoke_label(fig)
    return save_figure(fig, figures, stem)


def plot_apertures(
    aperture: Mapping[str, np.ndarray],
    pair_metrics: Mapping[str, Mapping[str, Any]],
    figures: Path,
) -> list[Path]:
    mass = np.asarray(aperture["mass_hMsun"])
    isolated = np.asarray(aperture["isolated_5R"], dtype=bool)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for column, field in enumerate(("y", "kappa")):
        godmax = np.asarray(aperture[f"godmax_{field}_aperture_integral_sr"])
        baryonforge = np.asarray(aperture[f"baryonforge_{field}_aperture_integral_sr"])
        valid = (godmax > 0.0) & (baryonforge > 0.0)
        axes[0, column].scatter(
            godmax[valid & ~isolated],
            baryonforge[valid & ~isolated],
            c=np.log10(mass[valid & ~isolated]),
            cmap="viridis",
            marker="x",
            label="overlapping 5R apertures",
        )
        axes[0, column].scatter(
            godmax[valid & isolated],
            baryonforge[valid & isolated],
            c=np.log10(mass[valid & isolated]),
            cmap="viridis",
            edgecolor="k",
            linewidth=0.3,
            label="isolated 5R aperture",
        )
        limits = [
            min(np.min(godmax[valid]), np.min(baryonforge[valid])),
            max(np.max(godmax[valid]), np.max(baryonforge[valid])),
        ]
        axes[0, column].plot(limits, limits, color="0.3", lw=1)
        axes[0, column].set_xscale("log")
        axes[0, column].set_yscale("log")
        axes[0, column].set(
            xlabel=f"GODMAX {field} aperture integral [sr]",
            ylabel=f"BaryonForge {field} aperture integral [sr]",
            title=f"Per-halo 5R {field} apertures",
        )
        axes[0, column].legend(fontsize=7)
        ratio = np.divide(
            baryonforge,
            godmax,
            out=np.full_like(godmax, np.nan),
            where=godmax != 0.0,
        )
        axes[1, column].scatter(
            np.log10(mass[~isolated]), ratio[~isolated], marker="x", color="0.5"
        )
        axes[1, column].scatter(
            np.log10(mass[isolated]),
            ratio[isolated],
            c="tab:blue",
            edgecolor="k",
            linewidth=0.3,
        )
        axes[1, column].axhline(1.0, color="0.3", lw=1)
        metrics = pair_metrics[field]
        axes[1, column].set(
            xlabel=r"$\log_{10}(M_{200c}/[M_\odot/h])$",
            ylabel="BaryonForge / GODMAX",
            title=(
                f"global sum ratio={metrics['global_sum_ratio']:.3f}; "
                f"footprint J={metrics['footprint_jaccard']:.3f}"
            ),
        )
    fig.suptitle("Bounded map and aperture diagnostics")
    _smoke_label(fig)
    return save_figure(fig, figures, "07_aperture_map_statistics")


def plot_spectra(spectra: Mapping[str, Any], figures: Path) -> list[Path]:
    ell = np.asarray(spectra["ell"])
    specs = (
        ("yy", "godmax_yy", "baryonforge_yy"),
        ("kk", "godmax_kk", "baryonforge_kk"),
        ("yk", "godmax_yk", "baryonforge_yk"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex="col")
    for column, (label, godmax_name, baryonforge_name) in enumerate(specs):
        godmax = np.asarray(spectra["spectra"][godmax_name]["dell"])
        baryonforge = np.asarray(spectra["spectra"][baryonforge_name]["dell"])
        axes[0, column].plot(ell, godmax, marker="o", label="GODMAX")
        axes[0, column].plot(ell, baryonforge, marker="s", label="BaryonForge")
        axes[0, column].set_yscale(
            "symlog",
            linthresh=max(np.max(np.abs(godmax)) * 1.0e-5, np.finfo(float).tiny),
        )
        axes[0, column].set(ylabel=rf"$D_\ell^{{{label}}}$", title=f"bounded {label}")
        ratio = np.divide(
            baryonforge,
            godmax,
            out=np.full_like(godmax, np.nan),
            where=np.abs(godmax) > np.max(np.abs(godmax)) * 1.0e-10,
        )
        axes[1, column].plot(ell, ratio, marker="o")
        axes[1, column].axhline(1.0, color="0.3", lw=1)
        axes[1, column].set(xlabel=r"multipole $\ell$", ylabel="BaryonForge / GODMAX")
    axes[0, 0].legend()
    fig.suptitle(
        "Exploratory common-window NaMaster spectra after identical NSIDE128 downgrade"
    )
    _smoke_label(fig)
    return save_figure(fig, figures, "08_exploratory_bounded_spectra")


def make_plots(
    primary: Mapping[str, Any],
    bounded: Mapping[str, Any],
    paths: Mapping[str, Path],
    products: Mapping[str, Mapping[str, np.ndarray]],
    aperture: Mapping[str, np.ndarray],
    pair_metrics: Mapping[str, Mapping[str, Any]],
    spectra: Mapping[str, Any],
) -> list[Path]:
    outputs = []
    outputs.extend(plot_catalog(primary, paths["catalog"], paths["figures"]))
    profile_path = resolve_path(
        primary["profiles"]["output_h5"], primary["_config_path"]
    )
    outputs.extend(plot_profiles(profile_path, paths["figures"]))
    outputs.extend(
        plot_map_triplet(
            bounded,
            products,
            "map_ymap",
            "dimensionless Compton-y",
            "05_tsz_maps",
            paths["figures"],
        )
    )
    outputs.extend(
        plot_map_triplet(
            bounded,
            products,
            "map_kappa_cmb",
            "dimensionless CMB convergence",
            "06_cmb_lensing_maps",
            paths["figures"],
        )
    )
    outputs.extend(plot_apertures(aperture, pair_metrics, paths["figures"]))
    outputs.extend(plot_spectra(spectra, paths["figures"]))
    return outputs


def write_manifest(
    primary: Mapping[str, Any],
    bounded: Mapping[str, Any],
    selection: Mapping[str, Any],
    paths: Mapping[str, Path],
    plots: Sequence[Path],
) -> dict[str, Any]:
    inputs = {
        "primary_config": Path(primary["_config_path"]),
        "bounded_config": Path(bounded["_config_path"]),
        "parent_catalog": Path(selection["parent_path"]),
        "bounded_catalog": paths["catalog"],
        "profile_comparison": resolve_path(
            primary["profiles"]["output_h5"], primary["_config_path"]
        ),
        "godmax_map": paths["godmax_map"],
        "baryonforge_map": paths["baryonforge_map"],
        "diagnostics": paths["diagnostics"],
        "plot_driver": Path(__file__).resolve(),
    }
    manifest = {
        "schema": SCHEMA,
        "label": LABEL,
        "created_utc": utc_now(),
        "production_statistics_eligible": False,
        "selection": {
            key: jsonable(value)
            for key, value in selection.items()
            if key not in {"parent_indices", "source_rows"}
        },
        "input_products": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in inputs.items()
        },
        "plots": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in plots
        ],
        "method_notes": [
            "Both native painters consume the exact same 64-row bounded catalog.",
            "Halo centers are mass-stratified and all lie inside the configured 600 deg2 cap.",
            "Native painting stays at NSIDE1024; only explicitly exploratory spectra are identically downgraded to NSIDE128.",
            "Production measure_statistics.py is not invoked and its completeness gate is not weakened.",
            "Residuals are BaryonForge minus GODMAX.",
        ],
    }
    temporary = paths["manifest"].with_name(
        f".{paths['manifest'].name}.tmp.{os.getpid()}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, paths["manifest"])
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the bounded catalog/config and stop before painting.",
    )
    parser.add_argument(
        "--skip-paint",
        action="store_true",
        help="Reuse existing bounded map products and regenerate diagnostics/plots.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    primary = load_config(args.config)
    paths = validation_paths(primary)
    # ``--skip-paint --overwrite`` is a safe plot/diagnostics refresh: retain
    # the exact catalog/config bytes to which the existing maps are bound.
    overwrite_prepared = bool(args.overwrite and not args.skip_paint)
    bounded, selection = prepare(primary, paths, overwrite=overwrite_prepared)
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "ok": True,
                    "label": LABEL,
                    "catalog": str(paths["catalog"]),
                    "catalog_sha256": sha256_file(paths["catalog"]),
                    "config": str(paths["config"]),
                    "config_sha256": sha256_file(paths["config"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.skip_paint:
        run_painters(bounded, paths, overwrite=bool(args.overwrite))
    products, metadata, expected = validate_products(bounded, selection, paths)
    aperture = aperture_diagnostics(paths["catalog"], products)
    spectra = bounded_spectra(bounded, products)
    pair_metrics = write_diagnostics(
        paths["diagnostics"],
        primary_config=primary,
        bounded_config=bounded,
        paths=paths,
        products=products,
        metadata=metadata,
        expected=expected,
        aperture=aperture,
        spectra=spectra,
        overwrite=bool(args.overwrite),
    )
    plots = make_plots(
        primary,
        bounded,
        paths,
        products,
        aperture,
        pair_metrics,
        spectra,
    )
    manifest = write_manifest(primary, bounded, selection, paths, plots)
    print(
        json.dumps(
            {
                "ok": True,
                "label": LABEL,
                "catalog": str(paths["catalog"]),
                "godmax_map": str(paths["godmax_map"]),
                "baryonforge_map": str(paths["baryonforge_map"]),
                "diagnostics": str(paths["diagnostics"]),
                "manifest": str(paths["manifest"]),
                "n_plots": len(manifest["plots"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
