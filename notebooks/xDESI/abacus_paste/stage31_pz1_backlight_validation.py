"""Single-DESI-bin 600 deg^2 Abacus Backlight paste validation workflow.

This driver keeps the validation path narrow:

* choose a 600 deg^2 cap from the common DESI/DES/ACT footprint,
* stream Abacus Backlight halos once into a reusable HDF5 catalog,
* measure only one DESI-bin galaxy auto/cross spectra set on the cap,
* build single-pz Stage-31 theory curves and the unresolved mass correction,
* plot data, theory, and optional simulation measurements together.

The input halo lightcone is read-only.  Derived catalogs and products are
written under data/xDESI/processed/abacus_backlight/<run_name>.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import healpy as hp
import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
XDESI_DIR = THIS_DIR.parent
REPO_ROOT = XDESI_DIR.parents[1]
SURVEY_MEASURE_DIR = XDESI_DIR / "survey_measure"
for _path in (XDESI_DIR, SURVEY_MEASURE_DIR, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from abacus_lightcone_catalog import (  # noqa: E402
    angular_cap_mask,
    cap_radius_deg_for_area,
    catalog_specs_from_config,
    ensure_under_xdesi,
    preprocess_abacus_catalogs,
    validate_catalog_file,
)
from abacus_pasting_helpers import (  # noqa: E402
    MAP_DATASETS,
    build_pixel_work_package,
    build_theory_cls,
    catalog_path,
    combine_partial_maps,
    configure_jax_runtime_for_pasting,
    final_map_path,
    load_halo_catalog,
    load_maps_h5,
    partial_map_path,
    prepare_godmax_config,
    run_paste_split,
    write_maps_h5,
    wl_source_bins_from_config,
)
import abacus_particle_shell_helpers as psh  # noqa: E402
import godmax_multiprobe_theory_utils as gmt  # noqa: E402
import multiprobe_namaster as mpn  # noqa: E402


DEFAULT_CONFIG = THIS_DIR / "stage31_pz1_cap600.yaml"
CATALOG_KEY = "pz1cap600_z0p30_0p62_logMgt11p0"
STAGE31_MULTIGPU_COMBINED = (
    REPO_ROOT
    / "notebooks/xDESI/survey_measure/outputs/"
    / "godmax_multiprobe_fast1024_true_nz_hmc_stage31_multigpu/"
    / "stage31_hmc_8000x16_v2/combined"
)
DEFAULT_STAGE31_FIDUCIAL_VECTOR = (
    REPO_ROOT / "notebooks/xDESI/survey_measure/outputs/godmax_multiprobe_fast1024_true_nz/theory_data_vector_fast1024.npz"
)
DEFAULT_STAGE31_BESTFIT_VECTOR = STAGE31_MULTIGPU_COMBINED / "bestfit_theory_data_vector_stage31_multigpu_v2.npz"
DEFAULT_TOTAL_SHELL_ROOT = Path("/mnt/ceph/users/backlight/AbacusBacklight_base_c9999_ph9999/lightcone_healpix/total")
DEFAULT_HALO_SHELL_ROOT = Path("/mnt/storone/nfs1/backlight/AbacusBacklight_base_c9999_ph9999/lightcone_healpix/halo")


def pz_bin_from_config(config: Mapping[str, object]) -> int:
    pz = int(config.get("pasting", {}).get("pz_bin", 1))
    if pz < 1 or pz > 4:
        raise ValueError(f"Expected pasting.pz_bin in [1, 4], got {pz}.")
    return pz


def pz_tag(config: Mapping[str, object]) -> str:
    return f"pz{pz_bin_from_config(config)}"


def pz_measurement_tag(config: Mapping[str, object]) -> str:
    return str(config.get("pasting", {}).get("measurement_tag_base", f"{pz_tag(config)}_cap600"))


def cap_area_deg2_from_config(config: Mapping[str, object]) -> float:
    return float(config.get("sky_patch", {}).get("area_deg2", 600.0))


def cap_tag_from_area(area_deg2: float) -> str:
    area = float(area_deg2)
    if abs(area - round(area)) < 1.0e-6:
        return f"cap{int(round(area))}"
    return f"cap{area:g}".replace(".", "p")


def cap_tag_from_config(config: Mapping[str, object]) -> str:
    return cap_tag_from_area(cap_area_deg2_from_config(config))


def cap_area_latex_from_config(config: Mapping[str, object]) -> str:
    area = cap_area_deg2_from_config(config)
    if abs(area - round(area)) < 1.0e-6:
        text = f"{int(round(area))}"
    else:
        text = f"{area:g}"
    return rf"{text} deg$^2$"


def run_name_from_config(config: Mapping[str, object]) -> str:
    return str(config.get("pasting", {}).get("run_name", config.get("project", {}).get("name", pz_measurement_tag(config))))


def default_catalog_key(config: Mapping[str, object]) -> str:
    pasting_key = config.get("pasting", {}).get("catalog_key")
    if pasting_key:
        return str(pasting_key)
    if config.get("catalogs"):
        return str(next(iter(config["catalogs"])))
    return CATALOG_KEY


def core_spectra_for_pz(pz_bin: int) -> Tuple[str, ...]:
    tag = f"pz{int(pz_bin)}"
    return (
        f"desi_g_auto_{tag}",
        f"desi_g_act_y_{tag}",
        f"desi_g_des_shear_E_{tag}_tomo1",
        f"desi_g_des_shear_E_{tag}_tomo2",
        f"desi_g_des_shear_E_{tag}_tomo3",
        f"desi_g_des_shear_E_{tag}_tomo4",
        f"desi_g_act_kappa_{tag}",
        f"desi_pi_act_T_{tag}",
    )


def read_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path | str, data: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)
    return path


def read_config(path: Path | str) -> dict:
    cfg = read_yaml(path)
    ensure_under_xdesi(Path(cfg["project"]["output_root"]).expanduser().resolve())
    return cfg


def output_dir(config: Mapping[str, object], subdir_key: str) -> Path:
    project = config["project"]
    path = Path(project["output_root"]).expanduser().resolve() / str(project[subdir_key])
    ensure_under_xdesi(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cap_radius_from_config(config: Mapping[str, object]) -> float:
    patch = config["sky_patch"]
    if patch.get("radius_deg") is not None:
        return float(patch["radius_deg"])
    return cap_radius_deg_for_area(float(patch["area_deg2"]))


def require_cap_center(config: Mapping[str, object]) -> Tuple[float, float, float]:
    patch = config["sky_patch"]
    if patch.get("center_ra_deg") is None or patch.get("center_dec_deg") is None:
        raise ValueError("sky_patch.center_ra_deg and center_dec_deg are not set. Run select-cap first.")
    return float(patch["center_ra_deg"]), float(patch["center_dec_deg"]), cap_radius_from_config(config)


def cap_pixel_mask(nside: int, ra_deg: float, dec_deg: float, radius_deg: float) -> np.ndarray:
    nside = int(nside)
    mask = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    vec = hp.ang2vec(float(ra_deg), float(dec_deg), lonlat=True)
    pix = hp.query_disc(nside, vec, math.radians(float(radius_deg)), inclusive=False)
    mask[np.asarray(pix, dtype=np.int64)] = 1.0
    return mask


def field_metadata(fields: Mapping[str, mpn.FieldMap]) -> Dict[str, dict]:
    out = {}
    for name, field in fields.items():
        out[name] = {
            "label": field.label,
            "kind": field.kind,
            "spin": int(field.spin),
            "mask_name": field.mask_name,
            "metadata": copy.deepcopy(field.metadata),
        }
    return out


def pz_spectrum_specs(
    pz_bin: int,
    *,
    include_gtau: bool = False,
    available_fields: Optional[Iterable[str]] = None,
    require_core: bool = True,
) -> List[mpn.SpectrumSpec]:
    pz_bin = int(pz_bin)
    pz = f"pz{pz_bin}"
    g_field = f"g{pz_bin}"
    default_by_name = {spec.name: spec for spec in mpn.default_spectrum_specs()}
    core_spectra = core_spectra_for_pz(pz_bin)
    specs = [default_by_name[name] for name in core_spectra]
    if include_gtau:
        specs.append(
            mpn.SpectrumSpec(
                name=f"desi_g_tau_{pz}",
                family="desi_g_tau",
                fields=(g_field, "tau"),
                component=0,
                label=f"DESI g pz {pz_bin} x tau",
                theory_key=f"desi_g_tau_{pz}",
                metadata={"desi_pz": pz_bin, "diagnostic": "simulation-only g x tau"},
            )
        )
    if available_fields is None:
        return specs
    available = set(available_fields)
    missing_core = [spec.name for spec in specs[: len(core_spectra)] if any(f not in available for f in spec.fields)]
    if require_core and missing_core:
        raise KeyError(f"Missing fields for required pz{pz_bin} spectra: {missing_core}")
    return [spec for spec in specs if all(f in available for f in spec.fields)]


def needed_fields_for_specs(specs: Sequence[mpn.SpectrumSpec]) -> List[str]:
    seen = []
    for spec in specs:
        for name in spec.fields:
            if name not in seen:
                seen.append(name)
    return seen


def resample_field(field: mpn.FieldMap, nside: int) -> mpn.FieldMap:
    target_nside = int(nside)
    current_nside = hp.npix2nside(len(field.mask))
    if current_nside == target_nside:
        return copy.deepcopy(field)
    maps = [
        hp.ud_grade(np.asarray(component, dtype=np.float64), nside_out=target_nside, power=0).astype(np.float64)
        for component in field.maps
    ]
    mask = hp.ud_grade(np.asarray(field.mask, dtype=np.float64), nside_out=target_nside, power=0)
    metadata = copy.deepcopy(field.metadata)
    metadata["resampled_from_nside"] = int(current_nside)
    metadata["resampled_to_nside"] = int(target_nside)
    return mpn.FieldMap(
        name=field.name,
        label=field.label,
        kind=field.kind,
        spin=field.spin,
        maps=maps,
        mask=np.clip(mask, 0.0, None).astype(np.float64),
        mask_name=field.mask_name,
        metadata=metadata,
        catalog=copy.deepcopy(field.catalog),
    )


def cap_field(
    field: mpn.FieldMap,
    cap_mask: np.ndarray,
    center: Tuple[float, float],
    radius_deg: float,
    *,
    cap_tag: str = "cap600",
) -> mpn.FieldMap:
    cap_mask = np.asarray(cap_mask, dtype=np.float64)
    maps = [(np.asarray(component, dtype=np.float64) * cap_mask).astype(np.float64) for component in field.maps]
    mask = (np.asarray(field.mask, dtype=np.float64) * cap_mask).astype(np.float64)
    catalog = copy.deepcopy(field.catalog)
    if catalog:
        required = {"ra_deg", "dec_deg"}
        if required.issubset(catalog):
            keep = angular_cap_mask(catalog["ra_deg"], catalog["dec_deg"], center[0], center[1], radius_deg)
            catalog = {key: np.asarray(value)[keep] for key, value in catalog.items()}
    metadata = copy.deepcopy(field.metadata)
    metadata["cap_center_ra_deg"] = float(center[0])
    metadata["cap_center_dec_deg"] = float(center[1])
    metadata["cap_radius_deg"] = float(radius_deg)
    metadata["cap_area_deg2_requested"] = float(2.0 * math.pi * (1.0 - math.cos(math.radians(radius_deg))) * (180.0 / math.pi) ** 2)
    return mpn.FieldMap(
        name=field.name,
        label=field.label,
        kind=field.kind,
        spin=field.spin,
        maps=maps,
        mask=mask,
        mask_name=f"{field.mask_name}_{cap_tag}",
        metadata=metadata,
        catalog=catalog,
    )


def load_capped_map_fields(
    config: Mapping[str, object],
    *,
    nside: int,
    include_gtau: bool = False,
    require_core: bool = True,
) -> Tuple[Dict[str, mpn.FieldMap], dict, List[mpn.SpectrumSpec]]:
    fields, metadata = mpn.load_map_product(config["godmax"]["map_h5"])
    pz_bin = pz_bin_from_config(config)
    specs = pz_spectrum_specs(
        pz_bin,
        include_gtau=include_gtau,
        available_fields=fields.keys(),
        require_core=require_core,
    )
    need = needed_fields_for_specs(specs)
    center_ra, center_dec, radius_deg = require_cap_center(config)
    cap = cap_pixel_mask(int(nside), center_ra, center_dec, radius_deg)
    cap_tag = cap_tag_from_config(config)
    capped = {}
    for name in need:
        capped[name] = cap_field(
            resample_field(fields[name], int(nside)),
            cap,
            (center_ra, center_dec),
            radius_deg,
            cap_tag=cap_tag,
        )
    meta = copy.deepcopy(metadata)
    meta[run_name_from_config(config)] = {
        "center_ra_deg": center_ra,
        "center_dec_deg": center_dec,
        "radius_deg": radius_deg,
        "area_deg2_requested": float(config["sky_patch"]["area_deg2"]),
        "nside": int(nside),
        "spectra": [spec.name for spec in specs],
        "photoz_vs_truez": (
            f"pz{pz_bin} is a photometric tracer label. The map/theory comparison uses the calibrated "
            f"pz{pz_bin} true-redshift kernel; no simulated photo-z assignment or photo-z cut is applied."
        ),
    }
    return capped, meta, specs


def select_cap(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    fields, _ = mpn.load_map_product(config["godmax"]["map_h5"])
    selection_fields = list(config["sky_patch"].get("selection_fields", []))
    missing = [name for name in selection_fields if name not in fields]
    if missing:
        raise KeyError(f"Missing selection field(s) in map product: {missing}")

    first = fields[selection_fields[0]]
    map_nside = hp.npix2nside(len(first.mask))
    common = np.ones_like(first.mask, dtype=bool)
    for name in selection_fields:
        if len(fields[name].mask) != len(first.mask):
            raise ValueError(f"Field {name} mask has different nside from {selection_fields[0]}.")
        common &= np.asarray(fields[name].mask) > 0.0

    candidate_nside = int(args.candidate_nside or config.get("validation", {}).get("cap_candidate_nside", 64))
    common_low = hp.ud_grade(common.astype(np.float64), nside_out=candidate_nside, power=0)
    candidates = np.flatnonzero(common_low > float(args.min_candidate_fraction))
    if candidates.size == 0:
        candidates = np.flatnonzero(common_low > 0.0)
    if candidates.size == 0:
        raise RuntimeError("No positive common-footprint pixels found for cap selection.")

    radius_deg = cap_radius_from_config(config)
    radius_rad = math.radians(radius_deg)
    coarse_scores = []
    for pix in candidates:
        vec = hp.pix2vec(candidate_nside, int(pix))
        cap_pix = hp.query_disc(candidate_nside, vec, radius_rad, inclusive=False)
        score = float(np.mean(common_low[np.asarray(cap_pix, dtype=np.int64)])) if len(cap_pix) else 0.0
        coarse_scores.append((score, int(pix)))
    coarse_scores.sort(reverse=True)
    top_n = int(args.refine_top_n or config.get("validation", {}).get("cap_refine_top_n", 128))
    refine = coarse_scores[: max(1, top_n)]

    best = None
    pix_area_deg2 = hp.nside2pixarea(map_nside, degrees=True)
    for _, pix in refine:
        ra, dec = hp.pix2ang(candidate_nside, pix, lonlat=True)
        cap = cap_pixel_mask(map_nside, float(ra), float(dec), radius_deg).astype(bool)
        cap_area = float(np.count_nonzero(cap) * pix_area_deg2)
        common_area = float(np.count_nonzero(common & cap) * pix_area_deg2)
        common_fraction = common_area / cap_area if cap_area > 0.0 else 0.0
        score = (common_fraction, common_area)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "center_ra_deg": float(ra),
                "center_dec_deg": float(dec),
                "radius_deg": float(radius_deg),
                "actual_cap_area_deg2": cap_area,
                "common_area_deg2": common_area,
                "common_fraction": common_fraction,
                "map_nside": int(map_nside),
                "candidate_nside": int(candidate_nside),
                "selection_fields": selection_fields,
            }
    assert best is not None

    out_config = copy.deepcopy(config)
    out_config["sky_patch"]["center_ra_deg"] = best["center_ra_deg"]
    out_config["sky_patch"]["center_dec_deg"] = best["center_dec_deg"]
    out_config["sky_patch"]["radius_deg"] = best["radius_deg"]
    out_config["sky_patch"]["selected_common_area_deg2"] = best["common_area_deg2"]
    out_config["sky_patch"]["selected_common_fraction"] = best["common_fraction"]
    out_config["sky_patch"]["selected_actual_cap_area_deg2"] = best["actual_cap_area_deg2"]
    output = Path(args.output_config) if args.output_config else Path(args.config).with_suffix(".selected.yaml")
    write_yaml(output, out_config)
    print(json.dumps({"output_config": str(output), "cap": best}, indent=2, sort_keys=True))


def preprocess(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    catalogs = args.catalog or [default_catalog_key(config)]
    counts = preprocess_abacus_catalogs(
        args.config,
        only_catalogs=catalogs,
        max_files=args.max_files,
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
    )
    summary = {"counts": counts}
    if not args.dry_run:
        for spec in catalog_specs_from_config(config, catalogs):
            path = catalog_path(config, spec.key)
            summary[spec.key] = {"path": str(path), "validation": validate_catalog_file(path)}
        out = output_dir(config, "catalog_subdir") / "preprocess_summary.json"
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary["summary_path"] = str(out)
    print(json.dumps(summary, indent=2, sort_keys=True))


def catalog_summary(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    key = args.catalog or default_catalog_key(config)
    path = catalog_path(config, key)
    summary = validate_catalog_file(path)
    with h5py.File(path, "r") as h5:
        attrs = {name: h5.attrs[name] for name in h5.attrs}
        summary["path"] = str(path)
        summary["attrs"] = {
            str(name): (value.decode("utf-8") if isinstance(value, bytes) else value)
            for name, value in attrs.items()
            if str(name).startswith(("z_", "log10_", "sky_patch_", "catalog_", "metadata_", "particle_mass"))
        }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


def measurement_config_from_workflow(config: Mapping[str, object], nside: int, stage_label: str) -> mpn.MeasurementConfig:
    pasting = config["pasting"]
    mcfg = mpn.MeasurementConfig.for_stage("fast1024")
    mcfg.stage = stage_label
    mcfg.nside = int(nside)
    mcfg.lmax = min(int(pasting.get("lmax", 1024)), 3 * int(nside) - 1)
    mcfg.ell_min = int(pasting.get("ell_min", 8))
    mcfg.n_bins = int(pasting.get("n_bins", 10))
    mcfg.binning = str(pasting.get("binning", "linear"))
    mcfg.compute_covariance = bool(pasting.get("compute_covariance", False))
    mcfg.compute_covariance_eigenvalues = False
    mcfg.include_ksz_velocity_shuffle = False
    return mcfg


def default_measurement_path(config: Mapping[str, object], kind: str, nside: int) -> Path:
    pasting = config["pasting"]
    lmax = min(int(pasting.get("lmax", 1024)), 3 * int(nside) - 1)
    tag = (
        f"{kind}_{pz_measurement_tag(config)}_nside{int(nside)}_lmax{lmax}_"
        f"nbin{int(pasting.get('n_bins', 10))}_{pasting.get('binning', 'linear')}"
    )
    return output_dir(config, "measurement_subdir") / f"{tag}.h5"


def measure_data(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    nside = int(args.nside or config["pasting"].get("nside", 1024))
    include_gtau = bool(args.include_gtau)
    fields, metadata, specs = load_capped_map_fields(
        config,
        nside=nside,
        include_gtau=include_gtau,
        require_core=True,
    )
    mcfg = measurement_config_from_workflow(config, nside, f"{run_name_from_config(config)}_data_nside{nside}")
    result = mpn.measure_all(fields, mcfg, specs=specs, verbose=not args.quiet)
    output = Path(args.output) if args.output else default_measurement_path(config, "data", nside)
    mpn.save_measurement_product(output, result, metadata, overwrite=bool(args.overwrite))
    print(json.dumps({"output": str(output), "spectra": [spec.name for spec in specs]}, indent=2, sort_keys=True))


def reference_field_info(map_path: Path, field_name: str, nside: int) -> dict:
    with h5py.File(map_path, "r") as h5:
        group = h5[f"fields/{field_name}"]
        mask_name = str(group.attrs["mask_ref"])
        mask = np.asarray(h5[f"masks/{mask_name}"][:], dtype=np.float64)
        in_nside = hp.npix2nside(mask.size)
        if in_nside != int(nside):
            mask = hp.ud_grade(mask, nside_out=int(nside), power=0)
        return {
            "name": str(group.attrs["name"]),
            "label": str(group.attrs["label"]),
            "kind": str(group.attrs["kind"]),
            "spin": int(group.attrs["spin"]),
            "mask_name": mask_name,
            "mask": np.clip(mask, 0.0, None).astype(np.float64),
            "metadata": json.loads(str(group.attrs["metadata_json"])),
        }


def subtract_weighted_mask_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float64)
    out = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    good = mask > 0.0
    if np.any(good):
        denom = float(np.sum(mask[good]))
        mean = float(np.sum(out[good] * mask[good]) / denom) if denom > 0.0 else 0.0
        out[good] -= mean
    out[~good] = 0.0
    return out.astype(np.float64)


def galaxy_delta_for_mask(galaxies: np.ndarray, nside: int, mask: np.ndarray) -> Tuple[np.ndarray, dict]:
    counts = np.zeros(hp.nside2npix(int(nside)), dtype=np.float64)
    n_total = 0.0
    if galaxies.size:
        valid = np.asarray(galaxies[:, 5]) > 0.5 if galaxies.shape[1] > 5 else np.ones(len(galaxies), dtype=bool)
        if np.any(valid):
            gals = galaxies[valid]
            pix = hp.ang2pix(int(nside), gals[:, 0], gals[:, 1], lonlat=True)
            in_mask = np.asarray(mask[pix], dtype=np.float64) > 0.0
            if np.any(in_mask):
                pix = pix[in_mask]
                np.add.at(counts, pix, 1.0)
                n_total = float(len(pix))
    mask = np.asarray(mask, dtype=np.float64)
    good = mask > 0.0
    delta = np.zeros_like(counts, dtype=np.float64)
    mask_sum = float(np.sum(mask[good])) if np.any(good) else 0.0
    mean_count_per_weighted_pixel = float(np.sum(counts[good] * mask[good]) / mask_sum) if mask_sum > 0.0 else 0.0
    if mean_count_per_weighted_pixel > 0.0:
        delta[good] = counts[good] / mean_count_per_weighted_pixel - 1.0
    pix_area_sr = float(hp.nside2pixarea(int(nside), degrees=False))
    area_sr = mask_sum * pix_area_sr
    shot_noise = area_sr / float(n_total) if n_total > 0 and area_sr > 0.0 else 0.0
    meta = {
        "n_gal": float(n_total),
        "mean_count_per_weighted_mask_pixel": mean_count_per_weighted_pixel,
        "mask_weight_sum_pixels": mask_sum,
        "area_sr": area_sr,
        "shot_noise": shot_noise,
        "shot_noise_convention": "pasted sampled galaxies: weighted_mask_area_sr / n_gal",
    }
    return delta, meta


def galaxy_momentum_for_mask(
    galaxies: np.ndarray,
    nside: int,
    mask: np.ndarray,
    *,
    velocity_mode: str = "true_velocity",
    velocity_correlation_r: Optional[float] = None,
    sigma_rec_over_c: Optional[float] = None,
    reconstruction_noise_seed: int = 12345,
) -> Tuple[np.ndarray, dict, dict]:
    velocity_mode = str(velocity_mode).lower()
    if velocity_mode not in {"true_velocity", "photoz_reconstruction_emulation"}:
        raise ValueError(
            f"Unknown velocity_mode {velocity_mode!r}; expected 'true_velocity' or 'photoz_reconstruction_emulation'."
        )
    npix = hp.nside2npix(int(nside))
    pi_map = np.zeros(npix, dtype=np.float64)
    empty_catalog = {
        "ra_deg": np.empty(0, dtype=np.float64),
        "dec_deg": np.empty(0, dtype=np.float64),
        "weight": np.empty(0, dtype=np.float64),
        "field": np.empty(0, dtype=np.float64),
    }
    if galaxies.size == 0:
        return pi_map, {}, empty_catalog
    if galaxies.shape[1] < 7:
        raise ValueError(
            "Pasted galaxy catalog does not contain host_vlos_kms. "
            "Rerun paste-split after regenerating with the 7-column galaxy catalog schema."
        )

    valid = np.asarray(galaxies[:, 5]) > 0.5
    valid &= np.all(np.isfinite(galaxies[:, [0, 1, 6]]), axis=1)
    ra = np.asarray(galaxies[valid, 0], dtype=np.float64)
    dec = np.asarray(galaxies[valid, 1], dtype=np.float64)
    vr_over_c = np.asarray(galaxies[valid, 6], dtype=np.float64) / 299792.458
    weights = np.asarray(galaxies[valid, 7], dtype=np.float64) if galaxies.shape[1] > 7 else np.ones_like(vr_over_c, dtype=np.float64)
    if ra.size:
        pix = hp.ang2pix(int(nside), ra, dec, lonlat=True)
        in_mask = np.asarray(mask[pix], dtype=np.float64) > 0.0
        ra = ra[in_mask]
        dec = dec[in_mask]
        vr_over_c = vr_over_c[in_mask]
        weights = weights[in_mask]
        pix = pix[in_mask]
    else:
        pix = np.empty(0, dtype=np.int64)

    vr_true_over_c = vr_over_c
    mean_vr_true = float(np.average(vr_true_over_c, weights=weights)) if vr_true_over_c.size else np.nan
    rms_vr_true = float(np.sqrt(np.average(vr_true_over_c**2, weights=weights))) if vr_true_over_c.size else np.nan
    sigma_vr_true = (
        float(np.sqrt(np.average((vr_true_over_c - mean_vr_true) ** 2, weights=weights))) if vr_true_over_c.size else np.nan
    )
    emulation_meta: dict = {}
    if velocity_mode == "photoz_reconstruction_emulation":
        if velocity_correlation_r is None or sigma_rec_over_c is None:
            raise ValueError(
                "photoz_reconstruction_emulation requires velocity_correlation_r and sigma_rec_over_c "
                "(take them from the reference data pi field metadata)."
            )
        r_corr = float(velocity_correlation_r)
        sigma_rec = float(sigma_rec_over_c)
        if not (0.0 < r_corr <= 1.0):
            raise ValueError(f"velocity_correlation_r must be in (0, 1], got {r_corr}.")
        if sigma_rec <= 0.0 or not np.isfinite(rms_vr_true) or rms_vr_true <= 0.0:
            raise ValueError(
                f"Invalid emulation inputs: sigma_rec_over_c={sigma_rec}, rms(v_true/c)={rms_vr_true}."
            )
        scale = r_corr * sigma_rec / rms_vr_true
        noise_sigma = sigma_rec * math.sqrt(max(0.0, 1.0 - r_corr**2))
        rng = np.random.default_rng(int(reconstruction_noise_seed))
        noise = rng.normal(0.0, noise_sigma, size=vr_true_over_c.size)
        vr_over_c = scale * vr_true_over_c + noise
        achieved_corr = (
            float(np.corrcoef(vr_true_over_c, vr_over_c)[0, 1]) if vr_true_over_c.size > 1 else np.nan
        )
        emulation_meta = {
            "vrec_emulation_velocity_correlation_r": r_corr,
            "vrec_emulation_sigma_rec_over_c_target": sigma_rec,
            "vrec_emulation_rms_true_vr_over_c": rms_vr_true,
            "vrec_emulation_scale_on_true_velocity": scale,
            "vrec_emulation_noise_sigma_over_c": noise_sigma,
            "vrec_emulation_noise_seed": int(reconstruction_noise_seed),
            "vrec_emulation_achieved_corr_v_true": achieved_corr,
            "vrec_emulation_expected_cross_amplitude": r_corr * sigma_rec * rms_vr_true,
            "vrec_emulation_model": "v_rec/c = r*sigma_rec/rms(v_true) * v_true/c + N(0, sigma_rec*sqrt(1-r^2))",
        }

    counts = np.zeros(npix, dtype=np.float64)
    vsum = np.zeros(npix, dtype=np.float64)
    if pix.size:
        np.add.at(counts, pix, weights)
        np.add.at(vsum, pix, weights * vr_over_c)
    mask = np.asarray(mask, dtype=np.float64)
    good = mask > 0.0
    mask_sum = float(np.sum(mask[good])) if np.any(good) else 0.0
    mean_count_per_weighted_pixel = float(np.sum(counts[good] * mask[good]) / mask_sum) if mask_sum > 0.0 else 0.0
    pix_area_sr = float(hp.nside2pixarea(int(nside), degrees=False))
    area_sr = mask_sum * pix_area_sr
    sum_weight = float(np.sum(weights, dtype=np.float64)) if weights.size else 0.0
    sum_weight2 = float(np.sum(weights**2, dtype=np.float64)) if weights.size else 0.0
    expected = mean_count_per_weighted_pixel * mask
    denom_good = good & (expected > 0.0)
    pi_map[denom_good] = vsum[denom_good] / expected[denom_good]
    pi_map = subtract_weighted_mask_mean(pi_map, mask)

    mean_vr = float(np.average(vr_over_c, weights=weights)) if vr_over_c.size else np.nan
    rms_vr = float(np.sqrt(np.average(vr_over_c**2, weights=weights))) if vr_over_c.size else np.nan
    sigma_vr = float(np.sqrt(np.average((vr_over_c - mean_vr) ** 2, weights=weights))) if vr_over_c.size else np.nan
    meta = {
        "n_gal": int(vr_over_c.size),
        "n_gal_momentum": int(vr_over_c.size),
        "mean_vr_over_c": mean_vr,
        "rms_rec_vr_over_c": rms_vr,
        "sigma_rec_vr_over_c": sigma_vr,
        "mean_count_per_weighted_mask_pixel": mean_count_per_weighted_pixel,
        "mask_weight_sum_pixels": mask_sum,
        "area_sr": area_sr,
        "nbar_per_sr": float(sum_weight / area_sr) if area_sr > 0.0 else 0.0,
        "n_eff_per_sr": float((sum_weight * sum_weight) / (sum_weight2 * area_sr)) if sum_weight2 > 0.0 and area_sr > 0.0 else 0.0,
        "sum_weight": sum_weight,
        "sum_weight2": sum_weight2,
        "shot_noise": float(area_sr * sum_weight2 / (sum_weight * sum_weight)) if sum_weight > 0.0 and area_sr > 0.0 else 0.0,
        "shot_noise_convention": "weighted Poisson: area_sr * sum(weight^2) / sum(weight)^2",
        "catalog_column_vlos": "host_vlos_kms",
        "velocity_mode": velocity_mode,
        "mean_true_vr_over_c": mean_vr_true,
        "rms_true_vr_over_c": rms_vr_true,
        "sigma_true_vr_over_c": sigma_vr_true,
        "velocity_source": "Abacus halo true line-of-sight velocity; assumed equal to true gas velocity for this sim validation.",
    }
    if velocity_mode == "photoz_reconstruction_emulation":
        meta.update(emulation_meta)
        meta["velocity_source"] = (
            "Photo-z velocity-reconstruction emulation of the Abacus halo true line-of-sight velocity: "
            "the momentum field carries the data-facing correlation r and reconstructed RMS sigma_rec, "
            "so the data theory amplitude A_v = r*sigma_rec*sigma_true applies directly."
        )
        meta["rms_rec_vr_over_c_weighted"] = rms_vr
        meta["sigma_rec_vr_over_c_weighted"] = sigma_vr
    catalog = {
        "ra_deg": ra,
        "dec_deg": dec,
        "weight": weights,
        "field": vr_over_c,
    }
    return pi_map, gmt.to_jsonable(meta), catalog


def halo_velocity_catalog_for_galaxies(
    galaxies: np.ndarray,
    nside: int,
    mask: np.ndarray,
    halo_catalog_path: Path,
    z_min: float,
    z_max: float,
    *,
    n_z_bins: int = 64,
) -> Tuple[np.ndarray, dict, dict]:
    npix = hp.nside2npix(int(nside))
    empty_catalog = {
        "ra_deg": np.empty(0, dtype=np.float64),
        "dec_deg": np.empty(0, dtype=np.float64),
        "weight": np.empty(0, dtype=np.float64),
        "field": np.empty(0, dtype=np.float64),
    }
    if galaxies.size == 0:
        return np.zeros(npix, dtype=np.float64), {}, empty_catalog
    if not halo_catalog_path.exists():
        raise FileNotFoundError(f"Missing halo catalog for velocity-field fallback: {halo_catalog_path}")

    n_z_bins = int(max(1, n_z_bins))
    z_edges = np.linspace(float(z_min), float(z_max), n_z_bins + 1, dtype=np.float64)
    with h5py.File(halo_catalog_path, "r") as h5:
        halo_ra = np.asarray(h5["ra_deg"][:], dtype=np.float64)
        halo_dec = np.asarray(h5["dec_deg"][:], dtype=np.float64)
        halo_z = np.asarray(h5["z"][:], dtype=np.float64)
        halo_vlos = np.asarray(h5["vlos_kms"][:], dtype=np.float64)

    finite_halo = np.isfinite(halo_ra) & np.isfinite(halo_dec) & np.isfinite(halo_z) & np.isfinite(halo_vlos)
    finite_halo &= (halo_z >= z_edges[0]) & (halo_z < z_edges[-1])
    halo_ra = halo_ra[finite_halo]
    halo_dec = halo_dec[finite_halo]
    halo_z = halo_z[finite_halo]
    halo_vlos = halo_vlos[finite_halo]
    halo_pix = hp.ang2pix(int(nside), halo_ra, halo_dec, lonlat=True)
    halo_zbin = np.searchsorted(z_edges, halo_z, side="right") - 1
    halo_zbin = np.clip(halo_zbin, 0, n_z_bins - 1).astype(np.int64)

    cell_key = halo_zbin * np.int64(npix) + halo_pix.astype(np.int64)
    order = np.argsort(cell_key, kind="mergesort")
    sorted_key = cell_key[order]
    sorted_v = halo_vlos[order]
    unique_key, start = np.unique(sorted_key, return_index=True)
    cell_sum = np.add.reduceat(sorted_v, start)
    cell_count = np.diff(np.r_[start, sorted_v.size]).astype(np.float64)
    cell_mean = cell_sum / np.maximum(cell_count, 1.0)

    pix_order = np.argsort(halo_pix, kind="mergesort")
    sorted_pix = halo_pix[pix_order].astype(np.int64)
    sorted_pix_v = halo_vlos[pix_order]
    unique_pix, pix_start = np.unique(sorted_pix, return_index=True)
    pix_sum = np.add.reduceat(sorted_pix_v, pix_start)
    pix_count = np.diff(np.r_[pix_start, sorted_pix_v.size]).astype(np.float64)
    pix_mean = pix_sum / np.maximum(pix_count, 1.0)

    valid = np.asarray(galaxies[:, 5]) > 0.5 if galaxies.shape[1] > 5 else np.ones(len(galaxies), dtype=bool)
    valid &= np.all(np.isfinite(galaxies[:, [0, 1, 2]]), axis=1)
    ra = np.asarray(galaxies[valid, 0], dtype=np.float64)
    dec = np.asarray(galaxies[valid, 1], dtype=np.float64)
    gal_z = np.asarray(galaxies[valid, 2], dtype=np.float64)
    if ra.size:
        gal_pix = hp.ang2pix(int(nside), ra, dec, lonlat=True)
        in_mask = np.asarray(mask[gal_pix], dtype=np.float64) > 0.0
        ra = ra[in_mask]
        dec = dec[in_mask]
        gal_z = gal_z[in_mask]
        gal_pix = gal_pix[in_mask]
    else:
        gal_pix = np.empty(0, dtype=np.int64)

    gal_zbin = np.searchsorted(z_edges, gal_z, side="right") - 1
    gal_zbin = np.clip(gal_zbin, 0, n_z_bins - 1).astype(np.int64)
    gal_key = gal_zbin * np.int64(npix) + gal_pix.astype(np.int64)
    pos = np.searchsorted(unique_key, gal_key)
    matched = (pos < unique_key.size) & (unique_key[np.minimum(pos, unique_key.size - 1)] == gal_key)
    gal_vlos = np.zeros(gal_key.size, dtype=np.float64)
    if np.any(matched):
        gal_vlos[matched] = cell_mean[pos[matched]]
    if np.any(~matched):
        ppos = np.searchsorted(unique_pix, gal_pix[~matched])
        pmatch = (ppos < unique_pix.size) & (unique_pix[np.minimum(ppos, unique_pix.size - 1)] == gal_pix[~matched])
        fallback = np.zeros(np.count_nonzero(~matched), dtype=np.float64)
        if np.any(pmatch):
            fallback[pmatch] = pix_mean[ppos[pmatch]]
        gal_vlos[~matched] = fallback

    vr_over_c = gal_vlos / 299792.458
    weights = np.ones_like(vr_over_c, dtype=np.float64)
    counts = np.zeros(npix, dtype=np.float64)
    vsum = np.zeros(npix, dtype=np.float64)
    if gal_pix.size:
        np.add.at(counts, gal_pix, weights)
        np.add.at(vsum, gal_pix, weights * vr_over_c)
    mask = np.asarray(mask, dtype=np.float64)
    good = mask > 0.0
    pi_map = np.zeros(npix, dtype=np.float64)
    mask_sum = float(np.sum(mask[good])) if np.any(good) else 0.0
    mean_count_per_weighted_pixel = float(np.sum(counts[good] * mask[good]) / mask_sum) if mask_sum > 0.0 else 0.0
    expected = mean_count_per_weighted_pixel * mask
    denom_good = good & (expected > 0.0)
    pi_map[denom_good] = vsum[denom_good] / expected[denom_good]
    pi_map = subtract_weighted_mask_mean(pi_map, mask)

    mean_vr = float(np.average(vr_over_c, weights=weights)) if vr_over_c.size else np.nan
    rms_vr = float(np.sqrt(np.average(vr_over_c**2, weights=weights))) if vr_over_c.size else np.nan
    sigma_vr = float(np.sqrt(np.average((vr_over_c - mean_vr) ** 2, weights=weights))) if vr_over_c.size else np.nan
    meta = {
        "n_gal_momentum": int(vr_over_c.size),
        "mean_vr_over_c": mean_vr,
        "rms_rec_vr_over_c": rms_vr,
        "sigma_rec_vr_over_c": sigma_vr,
        "mean_count_per_weighted_mask_pixel": mean_count_per_weighted_pixel,
        "mask_weight_sum_pixels": mask_sum,
        "velocity_source": "Binned Abacus halo velocity field queried at pasted galaxy positions.",
        "velocity_field_method": "mean halo v_los in (z_bin, healpix_pixel), fallback to same-pixel mean over redshift, then zero.",
        "velocity_field_halo_catalog": str(halo_catalog_path),
        "velocity_field_n_z_bins": n_z_bins,
        "velocity_field_z_min": float(z_edges[0]),
        "velocity_field_z_max": float(z_edges[-1]),
        "velocity_field_galaxy_cell_match_fraction": float(np.mean(matched)) if matched.size else np.nan,
        "velocity_field_galaxy_pixel_fallback_fraction": float(np.mean(~matched)) if matched.size else np.nan,
        "velocity_field_exact_host_velocity": False,
    }
    catalog = {
        "ra_deg": ra,
        "dec_deg": dec,
        "weight": weights,
        "field": vr_over_c,
    }
    return pi_map, gmt.to_jsonable(meta), catalog


def kappa_to_namaster_shear_maps(kappa: np.ndarray, nside: int, lmax: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a convergence map to an E-only spin-2 shear proxy.

    The spherical relation is gamma_E_lm = sqrt((l+2)(l-1)/(l(l+1))) kappa_lm
    for l >= 2.  This is the finite-cap proxy used for the first validation;
    full production should paste shear for each source bin directly or build it
    from a padded/full-sky convergence map.
    """

    kappa_clean = np.nan_to_num(np.asarray(kappa, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    lmax = int(min(lmax, 3 * int(nside) - 1))
    alm = hp.map2alm(kappa_clean, lmax=lmax, iter=0)
    ell, _ = hp.Alm.getlm(lmax)
    factor = np.zeros_like(ell, dtype=np.float64)
    good = ell >= 2
    factor[good] = np.sqrt((ell[good] + 2.0) * (ell[good] - 1.0) / (ell[good] * (ell[good] + 1.0)))
    e_alm = alm * factor
    b_alm = np.zeros_like(e_alm)
    gamma1, gamma2 = hp.alm2map_spin([e_alm, b_alm], int(nside), spin=2, lmax=lmax)
    return np.asarray(gamma1, dtype=np.float64), np.asarray(gamma2, dtype=np.float64)


def build_sim_measurement_fields(
    config: Mapping[str, object],
    *,
    map_path: Path,
    nside: int,
    include_gtau: bool,
) -> Tuple[Dict[str, mpn.FieldMap], dict, List[mpn.SpectrumSpec]]:
    if not map_path.exists():
        raise FileNotFoundError(f"Missing pasted map HDF5: {map_path}")

    maps, galaxies, attrs = load_maps_h5(map_path)
    pz_bin = pz_bin_from_config(config)
    pz = f"pz{pz_bin}"
    g_field = f"g{pz_bin}"
    mcfg = measurement_config_from_workflow(config, int(nside), f"{run_name_from_config(config)}_sim_nside{nside}")
    center_ra, center_dec, radius_deg = require_cap_center(config)
    cap = cap_pixel_mask(int(nside), center_ra, center_dec, radius_deg)
    cap_tag = cap_tag_from_config(config)
    sim_mask_mode = str(config.get("pasting", {}).get("sim_measurement_mask_mode", "reference")).lower()
    use_common_cap_mask = sim_mask_mode in {"cap", "common_cap", "binary_cap"}

    def measurement_mask(ref_info: Mapping[str, object]) -> np.ndarray:
        if use_common_cap_mask:
            return cap.copy()
        return np.clip(np.asarray(ref_info["mask"], dtype=np.float64) * cap, 0.0, None)

    ref_map_path = Path(config["godmax"]["map_h5"])

    g_ref = reference_field_info(ref_map_path, g_field, int(nside))
    g_mask = measurement_mask(g_ref)
    g_delta, g_meta = galaxy_delta_for_mask(galaxies, int(nside), g_mask)
    g_metadata = copy.deepcopy(g_ref["metadata"])
    g_metadata.update(
        {
            **g_meta,
            "source": "Abacus Backlight pasted galaxy catalog",
            "pasted_map_h5": str(map_path),
            "sim_measurement_mask_mode": sim_mask_mode,
            "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            "photoz_vs_truez": f"{pz} label uses Stage-31 HOD/true-n(z); simulated galaxies are not assigned or cut by photo-z.",
        }
    )
    fields: Dict[str, mpn.FieldMap] = {
        g_field: mpn.FieldMap(
            name=g_field,
            label=f"Abacus Backlight pasted DESI {pz} galaxy overdensity",
            kind="desi_galaxy",
            spin=0,
            maps=[g_delta],
            mask=g_mask,
            mask_name=f"desi_dr9_random_{cap_tag}",
            metadata=g_metadata,
        )
    }

    if "map_ksz" in maps:
        pi_ref = reference_field_info(ref_map_path, f"pi{pz_bin}", int(nside))
        pi_mask = g_mask.copy()
        ksz_velocity_mode = str(config.get("pasting", {}).get("ksz_velocity_mode", "true_velocity")).lower()
        if ksz_velocity_mode == "photoz_reconstruction_emulation" and galaxies.shape[1] < 7:
            raise ValueError(
                "pasting.ksz_velocity_mode=photoz_reconstruction_emulation requires the 7-column pasted "
                "galaxy catalog with host_vlos_kms; rerun paste-split with the current schema."
            )
        if galaxies.shape[1] >= 7 and ksz_velocity_mode == "photoz_reconstruction_emulation":
            ref_pi_meta = pi_ref["metadata"]
            sigma_rec_over_c = float(ref_pi_meta["rms_rec_vr_over_c_weighted"])
            r_corr = float(
                ref_pi_meta.get("ksz_photoz_velocity_correlation_r", mpn.KSZ_PHOTOZ_VELOCITY_CORRELATION_R)
            )
            noise_seed = int(config.get("pasting", {}).get("ksz_reconstruction_noise_seed", 12345))
            pi_map, pi_meta, pi_catalog = galaxy_momentum_for_mask(
                galaxies,
                int(nside),
                pi_mask,
                velocity_mode="photoz_reconstruction_emulation",
                velocity_correlation_r=r_corr,
                sigma_rec_over_c=sigma_rec_over_c,
                reconstruction_noise_seed=noise_seed,
            )
            pi_estimator_note = (
                "Simulation catalog momentum estimator with photo-z velocity-reconstruction emulation: "
                "positions=(ra_deg, dec_deg), weights=1, "
                "field = r*sigma_rec/rms(v_true) * host_vlos_kms/c + N(0, sigma_rec*sqrt(1-r^2)). "
                f"r={r_corr:g} and sigma_rec/c={sigma_rec_over_c:g} are taken from the reference data pi field "
                "metadata, so this momentum field has the same RMS and the same correlation with true velocities "
                "as the data reconstructed-velocity field; the data-facing kSZ theory "
                "C_ell^piT = -T_CMB_uK * (r*sigma_rec*sigma_true) * C_ell^gtau applies without modification."
            )
        elif galaxies.shape[1] >= 7:
            pi_map, pi_meta, pi_catalog = galaxy_momentum_for_mask(galaxies, int(nside), pi_mask)
            pi_estimator_note = (
                "Simulation catalog momentum estimator: positions=(ra_deg, dec_deg), weights=1, "
                "field=host_vlos_kms/c. This uses true halo velocities and does not apply the "
                "photo-z reconstruction correlation r used by theory-to-data conversion."
            )
        else:
            cat_path = Path(str(attrs.get("catalog_path", catalog_path(config, default_catalog_key(config)))))
            z_min = float(config["catalogs"][default_catalog_key(config)]["z_min"])
            z_max = float(config["catalogs"][default_catalog_key(config)]["z_max"])
            pi_map, pi_meta, pi_catalog = halo_velocity_catalog_for_galaxies(
                galaxies,
                int(nside),
                pi_mask,
                cat_path,
                z_min,
                z_max,
                n_z_bins=int(config.get("pasting", {}).get("ksz_velocity_z_bins", 64)),
            )
            pi_estimator_note = (
                "Simulation catalog momentum estimator fallback for legacy six-column pasted catalogs: "
                "positions=(ra_deg, dec_deg), weights=1, field=v_halo_field/c. The velocity field is the "
                "mean Abacus halo v_los in each (z_bin, HEALPix pixel), queried at pasted galaxy positions. "
                "This is a diagnostic approximation; rerun paste-split with the 7-column catalog for exact "
                "host_vlos_kms per galaxy."
            )
        pi_metadata = copy.deepcopy(pi_ref["metadata"])
        pi_metadata.update(
            {
                **pi_meta,
                "source": "Abacus Backlight pasted galaxy catalog with simulated velocity field",
                "pasted_map_h5": str(map_path),
                "namaster_field_class": "NmtFieldCatalogMomentum",
                "ksz_estimator": pi_estimator_note,
                "ksz_velocity_mode": ksz_velocity_mode,
                "catalog_field_is_weighted": False,
                "catalog_lonlat": True,
                "sim_measurement_mask_mode": sim_mask_mode,
                "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            }
        )
        pi_label = (
            f"Abacus Backlight reconstruction-emulated momentum {pz}"
            if ksz_velocity_mode == "photoz_reconstruction_emulation"
            else f"Abacus Backlight true-velocity momentum {pz}"
        )
        fields[f"pi{pz_bin}"] = mpn.FieldMap(
            name=f"pi{pz_bin}",
            label=pi_label,
            kind="desi_momentum",
            spin=0,
            maps=[pi_map],
            mask=pi_mask,
            mask_name=f"desi_dr9_random_{cap_tag}",
            metadata=pi_metadata,
            catalog=pi_catalog,
        )

        t_ref = reference_field_info(ref_map_path, "T", int(nside))
        t_mask = measurement_mask(t_ref)
        t_map_uk = subtract_weighted_mask_mean(mpn.TCMB_UK * np.asarray(maps["map_ksz"], dtype=np.float64), t_mask)
        t_metadata = copy.deepcopy(t_ref["metadata"])
        t_metadata.update(
            {
                "source": "Abacus Backlight pasted kSZ temperature map",
                "pasted_map_h5": str(map_path),
                "pasted_dataset": "map_ksz",
                "map_ksz_input_units": "Delta T / T_CMB",
                "temperature_units": "uK",
                "temperature_conversion": f"T_uK = {mpn.TCMB_UK:g} * map_ksz",
                "masked_mean_subtracted_for_measurement": True,
                "sim_measurement_mask_mode": sim_mask_mode,
                "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            }
        )
        fields["T"] = mpn.FieldMap(
            name="T",
            label="Abacus Backlight pasted kSZ temperature",
            kind=t_ref["kind"],
            spin=0,
            maps=[t_map_uk],
            mask=t_mask,
            mask_name=f"{t_ref['mask_name']}_{cap_tag}",
            metadata=t_metadata,
        )

    scalar_map_to_field = {
        "y": ("map_ymap", "y", "Abacus Backlight pasted tSZ Compton-y"),
        "kappa": ("map_kappa_cmb", "kappa", "Abacus Backlight pasted CMB lensing kappa"),
    }
    for out_name, (dataset, ref_name, label) in scalar_map_to_field.items():
        if dataset not in maps:
            continue
        ref = reference_field_info(ref_map_path, ref_name, int(nside))
        mask = measurement_mask(ref)
        metadata = copy.deepcopy(ref["metadata"])
        metadata.update(
            {
                "source": "Abacus Backlight pasted map",
                "pasted_map_h5": str(map_path),
                "pasted_dataset": dataset,
                "masked_mean_subtracted_for_measurement": True,
                "sim_measurement_mask_mode": sim_mask_mode,
                "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            }
        )
        fields[out_name] = mpn.FieldMap(
            name=out_name,
            label=label,
            kind=ref["kind"],
            spin=0,
            maps=[subtract_weighted_mask_mean(maps[dataset], mask)],
            mask=mask,
            mask_name=f"{ref['mask_name']}_{cap_tag}",
            metadata=metadata,
        )

    wl_source_bins = {1} if "map_kappa_wl" in maps else set()
    if "wl_source_bins_json" in attrs:
        try:
            wl_source_bins = {int(value) for value in json.loads(str(attrs["wl_source_bins_json"]))}
        except Exception:
            wl_source_bins = {1} if "map_kappa_wl" in maps else set()
    for tomo in range(1, 5):
        dataset = "map_kappa_wl" if tomo == 1 else f"map_kappa_wl_tomo{tomo}"
        if tomo not in wl_source_bins or dataset not in maps:
            continue
        ref = reference_field_info(ref_map_path, f"s{tomo}", int(nside))
        mask = measurement_mask(ref)
        kappa_wl = subtract_weighted_mask_mean(maps[dataset], mask)
        gamma1, gamma2 = kappa_to_namaster_shear_maps(kappa_wl, int(nside), int(mcfg.lmax))
        gamma1[mask <= 0.0] = 0.0
        gamma2[mask <= 0.0] = 0.0
        metadata = copy.deepcopy(ref["metadata"])
        metadata.update(
            {
                "source": f"Abacus Backlight pasted {dataset}",
                "pasted_map_h5": str(map_path),
                "pasted_dataset": dataset,
                "des_source_tomo": tomo,
                "shape_noise_pseudo_cl": 0.0,
                "shape_noise_note": "No DES shape noise is added for the pasted simulation cross-spectrum.",
                "input_spin_convention": "E-only spin-2 shear proxy generated from pasted convergence with healpy.alm2map_spin.",
                "finite_cap_caveat": "The shear proxy is built from the cap-limited convergence map; full production should paste shears directly or use padded/full-sky kappa.",
                "shear_e_to_kappa_sign": -1.0,
                "sim_measurement_mask_mode": sim_mask_mode,
                "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            }
        )
        fields[f"s{tomo}"] = mpn.FieldMap(
            name=f"s{tomo}",
            label=f"Abacus Backlight pasted DES source-bin {tomo} shear-E proxy",
            kind="des_shear",
            spin=2,
            maps=[gamma1, gamma2],
            mask=mask,
            mask_name=f"des_shear_tomo{tomo}_{cap_tag}",
            metadata=metadata,
        )

    if include_gtau and "map_tau" in maps:
        tau_mask = cap.copy()
        tau = subtract_weighted_mask_mean(maps["map_tau"], tau_mask)
        fields["tau"] = mpn.FieldMap(
            name="tau",
            label="Abacus Backlight pasted optical-depth tau",
            kind="sim_tau",
            spin=0,
            maps=[tau],
            mask=tau_mask,
            mask_name=cap_tag,
            metadata={
                "source": "Abacus Backlight pasted map",
                "pasted_map_h5": str(map_path),
                "pasted_dataset": "map_tau",
                "diagnostic": "simulation-only optical-depth cross-spectrum",
                "masked_mean_subtracted_for_measurement": True,
                "sim_measurement_mask_mode": sim_mask_mode,
                "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            },
        )

    specs = pz_spectrum_specs(pz_bin, include_gtau=include_gtau, available_fields=fields.keys(), require_core=False)
    if not specs:
        raise RuntimeError(f"No {pz} spectra can be measured from available simulated fields: {sorted(fields)}")
    missing = [
        spec.name
        for spec in pz_spectrum_specs(pz_bin, include_gtau=include_gtau, require_core=False)
        if spec.name not in {s.name for s in specs}
    ]
    metadata = {
        "schema": "stage31_single_pz_cap_sim_maps_for_namaster_v1",
        "pasted_map_h5": str(map_path),
        "pasted_map_attrs": {
            str(key): (value.decode("utf-8") if isinstance(value, bytes) else value)
            for key, value in attrs.items()
        },
        "cap": {
            "center_ra_deg": center_ra,
            "center_dec_deg": center_dec,
            "radius_deg": radius_deg,
            "area_deg2_requested": float(config["sky_patch"]["area_deg2"]),
            "nside": int(nside),
        },
        "sim_measurement_mask_mode": sim_mask_mode,
        "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
        "spectra_measured": [spec.name for spec in specs],
        "spectra_skipped_missing_sim_fields": missing,
        "field_metadata": field_metadata(fields),
        "comparison_caveat": (
            "This cap product measures only fields present in the pasted HDF5. "
            "It includes g auto, g x y, any pasted DES source-bin shear proxies, "
            "g x ACT kappa, optional g x tau, and pi x pasted-kSZ T when the pasted galaxy catalog "
            "contains host_vlos_kms."
        ),
    }
    return fields, gmt.to_jsonable(metadata), specs


def measure_sim(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    nside = int(args.nside or config["pasting"].get("nside", 1024))
    include_gtau = (
        bool(config["pasting"].get("include_diagnostic_gtau", True))
        if args.include_gtau is None
        else bool(args.include_gtau)
    )
    catalog_key = args.catalog or default_catalog_key(config)
    map_path = Path(args.maps) if args.maps else final_map_path(config, catalog_key, nside)
    fields, metadata, specs = build_sim_measurement_fields(
        config,
        map_path=map_path,
        nside=nside,
        include_gtau=include_gtau,
    )
    mcfg = measurement_config_from_workflow(config, nside, f"{run_name_from_config(config)}_sim_nside{nside}")
    result = mpn.measure_all(fields, mcfg, specs=specs, verbose=not args.quiet)
    output = Path(args.output) if args.output else default_measurement_path(config, "sim", nside)
    mpn.save_measurement_product(output, result, metadata, overwrite=bool(args.overwrite))
    print(
        json.dumps(
            {
                "output": str(output),
                "maps": str(map_path),
                "spectra": [spec.name for spec in specs],
                "skipped": metadata["spectra_skipped_missing_sim_fields"],
                "nside": nside,
                "lmax": int(mcfg.lmax),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _sim_cap_context(config: Mapping[str, object], nside: int):
    center_ra, center_dec, radius_deg = require_cap_center(config)
    cap = cap_pixel_mask(int(nside), center_ra, center_dec, radius_deg)
    sim_mask_mode = str(config.get("pasting", {}).get("sim_measurement_mask_mode", "reference")).lower()
    use_common_cap_mask = sim_mask_mode in {"cap", "common_cap", "binary_cap"}

    def measurement_mask(ref_info: Mapping[str, object]) -> np.ndarray:
        if use_common_cap_mask:
            return cap.copy()
        return np.clip(np.asarray(ref_info["mask"], dtype=np.float64) * cap, 0.0, None)

    return center_ra, center_dec, radius_deg, cap, sim_mask_mode, use_common_cap_mask, measurement_mask


def _pasted_galaxy_field(
    config: Mapping[str, object],
    *,
    galaxies: np.ndarray,
    map_path: Path,
    nside: int,
    measurement_mask,
    sim_mask_mode: str,
    use_common_cap_mask: bool,
) -> mpn.FieldMap:
    pz_bin = pz_bin_from_config(config)
    g_field = f"g{pz_bin}"
    ref_map_path = Path(config["godmax"]["map_h5"])
    cap_tag = cap_tag_from_config(config)
    g_ref = reference_field_info(ref_map_path, g_field, int(nside))
    g_mask = measurement_mask(g_ref)
    g_delta, g_meta = galaxy_delta_for_mask(galaxies, int(nside), g_mask)
    g_metadata = copy.deepcopy(g_ref["metadata"])
    g_metadata.update(
        {
            **g_meta,
            "source": "Abacus Backlight pasted galaxy catalog",
            "pasted_map_h5": str(map_path),
            "sim_measurement_mask_mode": sim_mask_mode,
            "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            "photoz_vs_truez": f"pz{pz_bin} label uses Stage-31 HOD/true-n(z); simulated galaxies are not assigned or cut by photo-z.",
        }
    )
    return mpn.FieldMap(
        name=g_field,
        label=f"Abacus Backlight pasted DESI pz{pz_bin} galaxy overdensity",
        kind="desi_galaxy",
        spin=0,
        maps=[g_delta],
        mask=g_mask,
        mask_name=f"desi_dr9_random_{cap_tag}",
        metadata=g_metadata,
    )


def measure_scalar_wl(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    catalog_key = args.catalog or default_catalog_key(config)
    nside = int(args.nside or config["pasting"].get("nside", 1024))
    map_path = Path(args.maps).expanduser().resolve() if args.maps else final_map_path(config, catalog_key, nside)
    maps, galaxies, attrs = load_maps_h5(map_path)
    center_ra, center_dec, radius_deg, _cap, sim_mask_mode, use_common_cap_mask, measurement_mask = _sim_cap_context(config, nside)
    pz_bin = pz_bin_from_config(config)
    pz = f"pz{pz_bin}"
    cap_tag = cap_tag_from_config(config)
    ref_map_path = Path(config["godmax"]["map_h5"])

    g_field = _pasted_galaxy_field(
        config,
        galaxies=galaxies,
        map_path=map_path,
        nside=nside,
        measurement_mask=measurement_mask,
        sim_mask_mode=sim_mask_mode,
        use_common_cap_mask=use_common_cap_mask,
    )
    fields: Dict[str, mpn.FieldMap] = {g_field.name: g_field}
    specs: List[mpn.SpectrumSpec] = []

    wl_source_bins = {1} if "map_kappa_wl" in maps else set()
    if "wl_source_bins_json" in attrs:
        try:
            wl_source_bins = {int(value) for value in json.loads(str(attrs["wl_source_bins_json"]))}
        except Exception:
            wl_source_bins = {1} if "map_kappa_wl" in maps else set()
    for tomo in range(1, 5):
        dataset = "map_kappa_wl" if tomo == 1 else f"map_kappa_wl_tomo{tomo}"
        if tomo not in wl_source_bins or dataset not in maps:
            continue
        ref = reference_field_info(ref_map_path, f"s{tomo}", int(nside))
        mask = measurement_mask(ref)
        field_name = f"kappa_wl_tomo{tomo}"
        metadata = copy.deepcopy(ref["metadata"])
        metadata.update(
            {
                "source": f"Abacus Backlight pasted scalar {dataset}",
                "pasted_map_h5": str(map_path),
                "pasted_dataset": dataset,
                "des_source_tomo": tomo,
                "masked_mean_subtracted_for_measurement": True,
                "diagnostic": "scalar convergence cross bypassing finite-cap spin-2 shear proxy",
                "sim_measurement_mask_mode": sim_mask_mode,
                "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            }
        )
        fields[field_name] = mpn.FieldMap(
            name=field_name,
            label=f"Abacus Backlight pasted DES source-bin {tomo} scalar convergence",
            kind="sim_scalar_wl_kappa",
            spin=0,
            maps=[subtract_weighted_mask_mean(maps[dataset], mask)],
            mask=mask,
            mask_name=f"des_kappa_wl_tomo{tomo}_{cap_tag}",
            metadata=metadata,
        )
        specs.append(
            mpn.SpectrumSpec(
                name=f"desi_g_kappa_wl_scalar_{pz}_tomo{tomo}",
                family="desi_g_kappa_wl_scalar",
                fields=(g_field.name, field_name),
                component=0,
                label=f"DESI g pz {pz_bin} x scalar WL kappa tomo {tomo}",
                theory_key=f"desi_g_des_shear_E_{pz}_tomo{tomo}",
                metadata={
                    "desi_pz": pz_bin,
                    "des_source_tomo": tomo,
                    "diagnostic": "scalar convergence cross for shear-proxy transfer isolation",
                },
            )
        )
    if not specs:
        raise RuntimeError(f"No scalar WL source-bin maps found in {map_path}")

    mcfg = measurement_config_from_workflow(config, nside, f"{run_name_from_config(config)}_scalar_wl_nside{nside}")
    result = mpn.measure_all(fields, mcfg, specs=specs, verbose=not args.quiet)
    metadata = {
        "schema": "stage31_scalar_wl_sim_maps_for_namaster_v1",
        "pasted_map_h5": str(map_path),
        "pasted_map_attrs": {str(key): (value.decode("utf-8") if isinstance(value, bytes) else value) for key, value in attrs.items()},
        "cap": {
            "center_ra_deg": center_ra,
            "center_dec_deg": center_dec,
            "radius_deg": radius_deg,
            "area_deg2_requested": float(config["sky_patch"]["area_deg2"]),
            "nside": int(nside),
        },
        "sim_measurement_mask_mode": sim_mask_mode,
        "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
        "spectra_measured": [spec.name for spec in specs],
        "field_metadata": field_metadata(fields),
        "comparison_caveat": "Scalar WL convergence diagnostic; bypasses the spin-2 shear proxy.",
    }
    output = Path(args.output).expanduser().resolve() if args.output else output_dir(config, "measurement_subdir") / (
        f"sim_scalar_wl_{pz_measurement_tag(config)}_nside{nside}_lmax{int(mcfg.lmax)}_nbin{int(mcfg.n_bins)}_{mcfg.binning}.h5"
    )
    mpn.save_measurement_product(output, result, metadata, overwrite=bool(args.overwrite))
    print(json.dumps({"output": str(output), "spectra": [spec.name for spec in specs]}, indent=2))


def _extract_kernel_vector(kernel, z_len: int, *, bin_index: int = 0) -> np.ndarray:
    arr = np.squeeze(np.asarray(kernel, dtype=np.float64))
    if arr.ndim == 1 and arr.shape[0] == z_len:
        return arr
    if arr.ndim == 2:
        if arr.shape[1] == z_len:
            return arr[int(bin_index)]
        if arr.shape[0] == z_len:
            return arr[:, int(bin_index)]
    raise ValueError(f"Could not extract kernel vector with z_len={z_len}; got shape={arr.shape}.")


def _integrate_shell_kernel(shell_meta: Mapping[str, object], z_grid: np.ndarray, chi_grid: np.ndarray, kernel_grid: np.ndarray, *, mode: str, n_samples: int) -> float:
    chi_lo = float(shell_meta["chi_lo_hMpc"])
    chi_hi = float(shell_meta["chi_hi_hMpc"])
    z_mid = float(shell_meta["z_mid"])
    dchi = abs(chi_hi - chi_lo)
    if mode == "midpoint":
        return float(np.interp(z_mid, z_grid, kernel_grid) * dchi)
    if mode != "average":
        raise ValueError(f"Unknown shell-weight mode {mode!r}")
    chi_samples = np.linspace(min(chi_lo, chi_hi), max(chi_lo, chi_hi), int(n_samples))
    z_samples = np.interp(chi_samples, chi_grid, z_grid)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(np.interp(z_samples, z_grid, kernel_grid), chi_samples))
    return float(np.trapz(np.interp(z_samples, z_grid, kernel_grid), chi_samples))


def _lensweighted_theory_cls(cls) -> Dict[str, np.ndarray]:
    z = np.asarray(cls.z_array_for_Cls, dtype=np.float64)
    chi = np.asarray(cls.chi_array_for_Cls, dtype=np.float64)
    dchi_dz = np.asarray(cls.dchi_dz_array_for_Cls, dtype=np.float64)
    wg = _extract_kernel_vector(cls.Wg_mat, len(z), bin_index=0)
    weight = wg * wg / np.clip(dchi_dz * chi * chi, 1.0e-30, np.inf)
    powers = {
        "mm": np.asarray(cls.cached_power_spectra[0, 0], dtype=np.float64),
        "gm": np.asarray(cls.cached_power_spectra[2, 0], dtype=np.float64),
        "gg": np.asarray(cls.cached_power_spectra[2, 2], dtype=np.float64),
    }
    out = {}
    for key, power in powers.items():
        integrand = power * weight[None, :]
        if hasattr(np, "trapezoid"):
            out[key] = np.trapezoid(integrand, z, axis=1)
        else:
            out[key] = np.trapz(integrand, z, axis=1)
    return out


def _vector_by_names(vector: np.ndarray, names: Sequence[str], n_per: int) -> Dict[str, np.ndarray]:
    return {name: np.asarray(vector[i * n_per : (i + 1) * n_per], dtype=np.float64) for i, name in enumerate(names)}


def measure_total_matter_bias_closure(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    catalog_key = args.catalog or default_catalog_key(config)
    nside = int(args.nside or config["pasting"].get("nside", 1024))
    map_path = Path(args.maps).expanduser().resolve() if args.maps else final_map_path(config, catalog_key, nside)
    maps, galaxies, attrs = load_maps_h5(map_path)
    center_ra, center_dec, radius_deg, cap, sim_mask_mode, use_common_cap_mask, measurement_mask = _sim_cap_context(config, nside)
    pz_bin = pz_bin_from_config(config)
    pz = f"pz{pz_bin}"
    g_field = _pasted_galaxy_field(
        config,
        galaxies=galaxies,
        map_path=map_path,
        nside=nside,
        measurement_mask=measurement_mask,
        sim_mask_mode=sim_mask_mode,
        use_common_cap_mask=use_common_cap_mask,
    )

    catalog_cfg = config.get("catalogs", {}).get(catalog_key, {})
    z_min = float(args.z_min if args.z_min is not None else catalog_cfg.get("z_min", attrs.get("z_min", 1.0e-4)))
    z_max = float(args.z_max if args.z_max is not None else catalog_cfg.get("z_max", attrs.get("z_max", 0.5)))
    resolved_cut = float(
        args.log10_mass_min
        if args.log10_mass_min is not None
        else config.get("godmax", {}).get("resolved_catalog_log10_m_min_hmsun", catalog_cfg.get("log10_m_min_hmsun", attrs.get("log10_m_min_hmsun", 11.0)))
    )
    total_root = Path(args.total_root).expanduser().resolve()
    halo_root = Path(args.halo_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve() if args.cache_root else psh.particle_shell_cache_root(config, nside)
    ensure_under_xdesi(cache_root)
    shell_meta = psh.discover_matched_total_halo_shells(total_root, halo_root, z_min=z_min, z_max=z_max, max_shells=args.max_shells)
    if not shell_meta:
        raise RuntimeError(f"No matched particle shells selected for z=[{z_min}, {z_max}]")

    cls = build_theory_cls(
        args.config,
        catalog_key,
        is_cmb_lensing=False,
        log10_mass_min=resolved_cut,
        z_max=z_max,
        include_ia=False,
    )
    z_grid = np.asarray(cls.z_array_for_Cls, dtype=np.float64)
    chi_grid = np.asarray(cls.chi_array_for_Cls, dtype=np.float64)
    wg_grid = _extract_kernel_vector(cls.Wg_mat, len(z_grid), bin_index=0)
    weights_by_step = {
        str(meta["step_id"]): _integrate_shell_kernel(
            meta,
            z_grid,
            chi_grid,
            wg_grid,
            mode=str(args.shell_weight_mode),
            n_samples=int(args.shell_weight_nsamples),
        )
        for meta in shell_meta
    }

    npix = hp.nside2npix(nside)
    matter_map = np.zeros(npix, dtype=np.float64)
    shell_rows = []
    for idx, meta in enumerate(shell_meta, 1):
        if not bool(args.quiet):
            print(f"[total-matter-closure] shell {idx}/{len(shell_meta)} {meta['step_id']}", flush=True)
        total_meta = dict(meta)
        total_meta["path_counts"] = meta["path_counts_total"]
        total_meta["path_vel_los"] = meta["path_vel_los_total"]
        cache_path = psh.read_or_create_downgraded_shell_cache(
            total_meta,
            nside,
            cache_root,
            overwrite=bool(args.overwrite_shell_cache),
            batch_parent_pixels=int(args.batch_parent_pixels),
        )
        data, cache_attrs = psh.load_downgraded_shell_cache(cache_path, nside)
        counts = np.asarray(data["counts"], dtype=np.float64)
        mean_total = float(np.mean(counts))
        if mean_total <= 0.0:
            raise ValueError(f"Shell {meta['step_id']} has non-positive mean count.")
        delta = (counts - mean_total) / mean_total
        matter_map += float(weights_by_step[str(meta["step_id"])]) * delta
        shell_rows.append(
            {
                "step_id": str(meta["step_id"]),
                "z_lo": float(meta["z_lo"]),
                "z_hi": float(meta["z_hi"]),
                "weight": float(weights_by_step[str(meta["step_id"])]),
                "cache_path": str(cache_path),
                "mean_total_counts": mean_total,
                "input_count_sum": float(cache_attrs.get("input_count_sum", np.nan)),
            }
        )
    matter_map = subtract_weighted_mask_mean(matter_map, cap)

    fields = {
        g_field.name: g_field,
        "m_lens": mpn.FieldMap(
            name="m_lens",
            label=f"Total-particle lens-weighted matter overdensity for DESI {pz}",
            kind="sim_total_matter_lens_weighted",
            spin=0,
            maps=[matter_map],
            mask=cap,
            mask_name=cap_tag_from_config(config),
            metadata={
                "source": "Abacus Backlight total particle shell counts",
                "total_root": str(total_root),
                "halo_root_for_shell_matching": str(halo_root),
                "cache_root": str(cache_root),
                "z_min": z_min,
                "z_max": z_max,
                "kernel": f"DESI {pz} lens Wg integrated over shell comoving distance",
                "masked_mean_subtracted_for_measurement": True,
                "sim_measurement_mask_mode": sim_mask_mode,
                "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            },
        ),
    }
    specs = [
        mpn.SpectrumSpec(
            name=f"desi_g_total_matter_lens_{pz}",
            family="desi_g_total_matter_lens",
            fields=(g_field.name, "m_lens"),
            component=0,
            label=f"DESI g {pz} x total matter lens-weighted field",
            theory_key=f"desi_g_total_matter_lens_{pz}",
            metadata={"desi_pz": pz_bin},
        ),
        mpn.SpectrumSpec(
            name=f"total_matter_lens_auto_{pz}",
            family="total_matter_lens_auto",
            fields=("m_lens", "m_lens"),
            component=0,
            label=f"Total matter lens-weighted auto {pz}",
            theory_key=f"total_matter_lens_auto_{pz}",
            metadata={"desi_pz": pz_bin},
        ),
        mpn.SpectrumSpec(
            name=f"desi_g_auto_closure_{pz}",
            family="desi_g_auto",
            fields=(g_field.name, g_field.name),
            component=0,
            label=f"DESI g auto closure {pz}",
            theory_key=f"desi_g_auto_lensweighted_{pz}",
            metadata={"desi_pz": pz_bin, "diagnostic": "same galaxy field used in total-matter bias closure"},
        ),
    ]
    mcfg = measurement_config_from_workflow(config, nside, f"{run_name_from_config(config)}_matter_bias_nside{nside}")
    result = mpn.measure_all(fields, mcfg, specs=specs, verbose=not args.quiet)
    metadata = {
        "schema": "stage31_total_matter_bias_closure_maps_for_namaster_v1",
        "pasted_map_h5": str(map_path),
        "pasted_map_attrs": {str(key): (value.decode("utf-8") if isinstance(value, bytes) else value) for key, value in attrs.items()},
        "cap": {
            "center_ra_deg": center_ra,
            "center_dec_deg": center_dec,
            "radius_deg": radius_deg,
            "area_deg2_requested": float(config["sky_patch"]["area_deg2"]),
            "nside": int(nside),
        },
        "sim_measurement_mask_mode": sim_mask_mode,
        "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
        "spectra_measured": [spec.name for spec in specs],
        "field_metadata": field_metadata(fields),
        "shells": shell_rows,
    }
    output = Path(args.output).expanduser().resolve() if args.output else output_dir(config, "measurement_subdir") / (
        f"total_matter_bias_closure_{pz_measurement_tag(config)}_nside{nside}_lmax{int(mcfg.lmax)}_nbin{int(mcfg.n_bins)}_{mcfg.binning}.h5"
    )
    mpn.save_measurement_product(output, result, metadata, overwrite=bool(args.overwrite))

    lens_theory = _lensweighted_theory_cls(cls)
    theory_dict = {
        f"desi_g_total_matter_lens_{pz}": lens_theory["gm"],
        f"total_matter_lens_auto_{pz}": lens_theory["mm"],
        f"desi_g_auto_lensweighted_{pz}": lens_theory["gg"],
    }
    theory_vec, theory_names = mpn.theory_to_data_vector(
        output,
        theory_dict,
        ell=np.asarray(cls.ell_array, dtype=np.float64),
        shear_m_bias=None,
        theory_shear_e_is_positive_kappa=True,
        include_default_pixel_windows=True,
        include_default_act_beams=False,
    )
    theory_by_name = _vector_by_names(theory_vec, theory_names, len(result["ell"]))
    sim_by_name = {name: np.asarray(result["spectra"][name]["cl"], dtype=np.float64) for name in result["spectra"]}
    gm_name = f"desi_g_total_matter_lens_{pz}"
    mm_name = f"total_matter_lens_auto_{pz}"
    gg_name = f"desi_g_auto_closure_{pz}"
    b_cross_sim = np.divide(sim_by_name[gm_name], sim_by_name[mm_name], out=np.full_like(sim_by_name[gm_name], np.nan), where=sim_by_name[mm_name] != 0.0)
    b_auto_sim = np.sqrt(np.divide(sim_by_name[gg_name], sim_by_name[mm_name], out=np.full_like(sim_by_name[gg_name], np.nan), where=sim_by_name[mm_name] > 0.0))
    b_cross_theory = np.divide(theory_by_name[gm_name], theory_by_name[mm_name], out=np.full_like(theory_by_name[gm_name], np.nan), where=theory_by_name[mm_name] != 0.0)
    b_auto_theory = np.sqrt(np.divide(theory_by_name[gg_name], theory_by_name[mm_name], out=np.full_like(theory_by_name[gg_name], np.nan), where=theory_by_name[mm_name] > 0.0))
    ratio_summary = {
        "ell": np.asarray(result["ell"], dtype=np.float64).tolist(),
        "b_cross_sim": b_cross_sim.tolist(),
        "b_auto_sim": b_auto_sim.tolist(),
        "b_cross_theory": b_cross_theory.tolist(),
        "b_auto_theory": b_auto_theory.tolist(),
        "b_cross_sim_over_theory": np.divide(b_cross_sim, b_cross_theory, out=np.full_like(b_cross_sim, np.nan), where=b_cross_theory != 0.0).tolist(),
        "b_auto_sim_over_theory": np.divide(b_auto_sim, b_auto_theory, out=np.full_like(b_auto_sim, np.nan), where=b_auto_theory != 0.0).tolist(),
        "b_cross_stats": _finite_ratio_stats(result["ell"], np.divide(b_cross_sim, b_cross_theory, out=np.full_like(b_cross_sim, np.nan), where=b_cross_theory != 0.0)),
        "b_auto_stats": _finite_ratio_stats(result["ell"], np.divide(b_auto_sim, b_auto_theory, out=np.full_like(b_auto_sim, np.nan), where=b_auto_theory != 0.0)),
    }
    summary_output = Path(args.summary_output).expanduser().resolve() if args.summary_output else output.with_suffix(output.suffix + ".bias_summary.json")
    _write_json_atomic(
        summary_output,
        {
            "schema": "stage31_total_matter_bias_closure_summary_v1",
            "measurement": str(output),
            "config": str(args.config),
            "maps": str(map_path),
            "nside": int(nside),
            "z_min": z_min,
            "z_max": z_max,
            "log10_mass_min_for_theory": resolved_cut,
            "shells": shell_rows,
            "bias": ratio_summary,
            "caveat": "Bin 1 should not be used for closure claims until cap mean-subtraction is modeled.",
        },
    )
    print(json.dumps({"output": str(output), "summary": str(summary_output)}, indent=2))


def _selected_halo_mass_by_shell(catalog_h5: Path, shell_meta: Sequence[Mapping[str, object]], chunk_size: int) -> Dict[str, dict]:
    out = {
        str(meta["step_id"]): {
            "n_selected_halos": 0,
            "selected_M200c_hMsun_sum": 0.0,
            "selected_M200c_hMsun_mean": np.nan,
        }
        for meta in shell_meta
    }
    z_los = np.asarray([float(meta["z_lo"]) for meta in shell_meta], dtype=np.float64)
    z_his = np.asarray([float(meta["z_hi"]) for meta in shell_meta], dtype=np.float64)
    step_ids = [str(meta["step_id"]) for meta in shell_meta]
    with h5py.File(catalog_h5, "r") as h5:
        n = int(h5["z"].shape[0])
        for start in range(0, n, int(chunk_size)):
            stop = min(start + int(chunk_size), n)
            z = np.asarray(h5["z"][start:stop], dtype=np.float64)
            mass = np.asarray(h5["M200c_hMsun"][start:stop], dtype=np.float64)
            for step, zlo, zhi in zip(step_ids, z_los, z_his):
                mask = (z >= zlo) & (z < zhi)
                count = int(np.count_nonzero(mask))
                if count:
                    out[step]["n_selected_halos"] += count
                    out[step]["selected_M200c_hMsun_sum"] += float(np.sum(mass[mask], dtype=np.float64))
    for step, row in out.items():
        if row["n_selected_halos"] > 0:
            row["selected_M200c_hMsun_mean"] = row["selected_M200c_hMsun_sum"] / float(row["n_selected_halos"])
    return out


def direct_field_mass_ledger(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    catalog_key = args.catalog or default_catalog_key(config)
    nside = int(args.nside or config["pasting"].get("nside", 1024))
    catalog_h5 = catalog_path(config, catalog_key)
    if not catalog_h5.exists():
        raise FileNotFoundError(catalog_h5)
    with h5py.File(catalog_h5, "r") as h5:
        catalog_attrs = {str(key): _jsonable_attr(value) for key, value in h5.attrs.items()}
    catalog_cfg = config.get("catalogs", {}).get(catalog_key, {})
    z_min = float(args.z_min if args.z_min is not None else catalog_cfg.get("z_min", catalog_attrs.get("z_min", 1.0e-4)))
    z_max = float(args.z_max if args.z_max is not None else catalog_cfg.get("z_max", catalog_attrs.get("z_max", 0.5)))
    total_root = Path(args.total_root).expanduser().resolve()
    halo_root = Path(args.halo_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve() if args.cache_root else psh.particle_shell_cache_root(config, nside)
    ensure_under_xdesi(cache_root)
    shell_meta = psh.discover_matched_total_halo_shells(total_root, halo_root, z_min=z_min, z_max=z_max, max_shells=args.max_shells)
    selected_by_step = _selected_halo_mass_by_shell(catalog_h5, shell_meta, int(args.catalog_chunk_size))

    painted_templates: Dict[str, dict] = {}
    if bool(args.build_painted_template_proxy):
        catalog, _attrs = load_halo_catalog(catalog_h5)
        painter = psh.DmoTemplatePainter(args.config, catalog_key, nside)
        painted_templates = painter.paint_shell_templates(
            catalog,
            shell_meta,
            cache_root,
            overwrite=bool(args.overwrite_painted_template_proxy),
            velocity_bins=1,
        )

    rows = []
    particle_mass_total = 0.0
    particle_mass_halo = 0.0
    selected_mass_sum = 0.0
    painted_proxy_sum = 0.0
    for meta in shell_meta:
        step = str(meta["step_id"])
        particle_mass = float(meta["particle_mass_hMsun"])
        npix_fine = float(hp.nside2npix(int(meta["nside_counts"])))
        total_count = float(meta["mean_count_total_fine"]) * npix_fine
        halo_count = float(meta["mean_count_halo_fine"]) * npix_fine
        selected = selected_by_step[step]
        selected_mass = float(selected["selected_M200c_hMsun_sum"])
        template_path = psh.nfw_template_cache_path(cache_root, catalog_key, meta, nside, 1)
        painted_count = np.nan
        painted_mass = np.nan
        painted_template_status = "missing"
        if step in painted_templates:
            attrs = painted_templates[step].get("attrs", {})
            painted_count = float(attrs.get("count_template_sum", np.nan))
            painted_mass = painted_count * particle_mass if np.isfinite(painted_count) else np.nan
            painted_template_status = "built_or_loaded"
        elif template_path.exists():
            with h5py.File(template_path, "r") as h5:
                painted_count = float(h5.attrs.get("count_template_sum", np.nan))
                painted_mass = painted_count * particle_mass if np.isfinite(painted_count) else np.nan
                painted_template_status = "existing_cache"

        direct_cache_path = psh.direct_field_shell_cache_path(cache_root, meta, nside)
        direct_cache_attrs = {}
        if direct_cache_path.exists():
            with h5py.File(direct_cache_path, "r") as h5:
                direct_cache_attrs = {str(key): _jsonable_attr(value) for key, value in h5.attrs.items()}

        row = {
            "step_id": step,
            "z_lo": float(meta["z_lo"]),
            "z_hi": float(meta["z_hi"]),
            "particle_mass_hMsun": particle_mass,
            "total_particle_count_est": total_count,
            "halo_particle_count_est": halo_count,
            "field_particle_count_est": total_count - halo_count,
            "total_particle_mass_hMsun_est": total_count * particle_mass,
            "identified_halo_particle_mass_hMsun_est": halo_count * particle_mass,
            "field_particle_mass_hMsun_est": (total_count - halo_count) * particle_mass,
            **selected,
            "painted_profile_proxy_count_sum": painted_count,
            "painted_profile_proxy_mass_hMsun": painted_mass,
            "painted_profile_proxy_status": painted_template_status,
            "painted_template_cache_path": str(template_path),
            "selected_over_identified_halo_particle_mass": selected_mass / (halo_count * particle_mass) if halo_count > 0.0 else np.nan,
            "painted_proxy_over_selected_M200c": painted_mass / selected_mass if selected_mass > 0.0 and np.isfinite(painted_mass) else np.nan,
            "direct_field_cache_path": str(direct_cache_path),
            "direct_field_cache_attrs": direct_cache_attrs,
        }
        rows.append(row)
        particle_mass_total += row["total_particle_mass_hMsun_est"]
        particle_mass_halo += row["identified_halo_particle_mass_hMsun_est"]
        selected_mass_sum += selected_mass
        if np.isfinite(painted_mass):
            painted_proxy_sum += painted_mass

    summary = {
        "n_shells": int(len(rows)),
        "total_particle_mass_hMsun_est": particle_mass_total,
        "identified_halo_particle_mass_hMsun_est": particle_mass_halo,
        "field_particle_mass_hMsun_est": particle_mass_total - particle_mass_halo,
        "selected_pasted_halo_M200c_hMsun_sum": selected_mass_sum,
        "painted_profile_proxy_mass_hMsun_sum_existing": painted_proxy_sum if painted_proxy_sum > 0.0 else np.nan,
        "selected_over_identified_halo_particle_mass": selected_mass_sum / particle_mass_halo if particle_mass_halo > 0.0 else np.nan,
        "field_mass_fraction_est": (particle_mass_total - particle_mass_halo) / particle_mass_total if particle_mass_total > 0.0 else np.nan,
    }
    output = Path(args.output).expanduser().resolve() if args.output else output_dir(config, "measurement_subdir") / (
        f"direct_field_mass_ledger_{pz_measurement_tag(config)}_nside{nside}.json"
    )
    _write_json_atomic(
        output,
        {
            "schema": "stage31_direct_field_mass_ledger_v1",
            "config": str(args.config),
            "catalog_key": str(catalog_key),
            "catalog_path": str(catalog_h5),
            "catalog_attrs": catalog_attrs,
            "nside": int(nside),
            "z_min": z_min,
            "z_max": z_max,
            "total_root": str(total_root),
            "halo_root": str(halo_root),
            "cache_root": str(cache_root),
            "summary": summary,
            "shells": rows,
            "painted_profile_proxy_note": (
                "painted_profile_proxy_* is populated from selected-halo DMO template caches. "
                "Pass --build-painted-template-proxy to create missing caches."
            ),
        },
    )
    print(json.dumps({"output": str(output), "summary": summary}, indent=2, default=gmt.to_jsonable))


def merge_bestfit_params(config: Mapping[str, object]) -> dict:
    gcfg = config["godmax"]
    cfg = gmt.load_comparison_config(gcfg["comparison_config"])
    bestfit = read_yaml(gcfg["bestfit_params"])
    cfg["params"] = gmt.deep_update(cfg["params"], bestfit)
    return gmt.materialize_nz_inputs(cfg)


def build_one_godmax_model(
    config: Mapping[str, object],
    *,
    is_cmb_lensing: bool,
    log10_mass_cut: Optional[float] = None,
    gg_transition_model: Optional[str] = None,
    tsz_transition_model: Optional[str] = None,
    galaxy_matter_transition_model: Optional[str] = None,
    galaxy_electron_transition_model: Optional[str] = None,
):
    gmt.ensure_godmax_import_paths(Path(config["repo_root"]))
    import jax.numpy as jnp
    from base_class import base_class
    from get_Cls import get_Cl
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles

    sim_params, halo_params, analysis, other_params = gmt._params_for_model(config, is_cmb_lensing=is_cmb_lensing)
    if gg_transition_model is not None:
        analysis["gg_transition_model"] = str(gg_transition_model)
    if tsz_transition_model is not None:
        analysis["tSZ_transition_model"] = str(tsz_transition_model)
    if galaxy_matter_transition_model is not None:
        analysis["galaxy_matter_transition_model"] = str(galaxy_matter_transition_model)
    if galaxy_electron_transition_model is not None:
        analysis["galaxy_electron_transition_model"] = str(galaxy_electron_transition_model)
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = Profiles(sim_params, halo_params, analysis, other_params, base_class_obj=base)
    if log10_mass_cut is not None:
        mass_mask = jnp.asarray(jnp.log10(profiles.M_array) >= float(log10_mass_cut))
        profiles.Ncen_mat = profiles.Ncen_mat * mass_mask[None, :]
        profiles.Nsat_mat = profiles.Nsat_mat * mass_mask[None, :]
    pkz = get_Pkz(sim_params, halo_params, analysis, other_params, Profiles_obj=profiles)
    cls = get_Cl(sim_params, halo_params, analysis, other_params, Pkz_obj=pkz)
    return cls


def _resolve_transition_model(value: Optional[str], default: Optional[str]) -> Optional[str]:
    model = default if value is None else value
    if model is None:
        return None
    text = str(model).strip().lower()
    if text in {"", "none", "config"}:
        return None
    if text not in {"poweradd", "response"}:
        raise ValueError(f"Unknown transition model {model!r}; expected poweradd, response, or config.")
    return text


def pz_theory_from_models(pz_wl, pz_cmb, pz_bin: int) -> Dict[str, np.ndarray]:
    pz = f"pz{int(pz_bin)}"
    theory: Dict[str, np.ndarray] = {
        f"desi_g_auto_{pz}": np.asarray(pz_wl.Cl_gal_gal_tot_mat[:, 0, 0], dtype=np.float64),
        f"desi_g_act_y_{pz}": np.asarray(pz_wl.Cl_gal_y_tot_mat[:, 0], dtype=np.float64),
        f"desi_g_act_kappa_{pz}": np.asarray(pz_cmb.Cl_gal_kappa_tot_mat[:, 0, 0], dtype=np.float64),
        f"desi_g_tau_{pz}": gmt.corrected_gal_tau_cls_zdependent(pz_wl)[:, 0],
    }
    for tomo in range(1, 5):
        theory[f"desi_g_des_shear_E_{pz}_tomo{tomo}"] = np.asarray(
            pz_wl.Cl_gal_kappa_tot_mat[:, 0, tomo - 1],
            dtype=np.float64,
        )
    return theory


def sim_matched_transfer_options(measurement_path: Path) -> Tuple[dict, dict]:
    """Return transfer options matching the current pasted-map construction.

    The default NaMaster theory wrapper is configured for real data: HEALPix
    pixel windows plus ACT beams.  Pasted validation maps instead contain
    binned catalog fields for DESI galaxy/momentum and paint-time Gaussian
    profile smoothing for the continuous pasted fields.  This helper builds the
    explicit per-field transfers for that simulation comparison mode.
    """

    with h5py.File(measurement_path, "r") as h5:
        config = json.loads(str(h5.attrs["config_json"]))
        nside = int(config["nside"])
        lmax = int(config["lmax"])
        field_meta = json.loads(str(h5["fields"].attrs["metadata_json"]))

    pix_t, _pix_p = hp.pixwin(nside, lmax=lmax, pol=True)
    fwhm_arcmin = float(hp.nside2resol(nside, arcmin=True))
    gaussian = mpn.gaussian_beam_transfer(lmax, fwhm_arcmin)
    transfers: Dict[str, np.ndarray] = {}
    transfer_notes: Dict[str, str] = {}
    for field_name in sorted(field_meta):
        if field_name.startswith("g") or field_name.startswith("pi"):
            transfers[field_name] = np.asarray(pix_t, dtype=np.float64)
            transfer_notes[field_name] = "healpix_temperature_pixwin; binned catalog field"
        elif field_name in {"y", "T", "kappa", "tau"} or field_name.startswith("s"):
            transfers[field_name] = np.asarray(gaussian, dtype=np.float64)
            transfer_notes[field_name] = f"gaussian_profile_smoothing_fwhm_arcmin={fwhm_arcmin:.8g}"

    metadata = {
        "mode": "sim_matched_transfers",
        "measurement_path": str(measurement_path),
        "nside": int(nside),
        "lmax": int(lmax),
        "painted_field_gaussian_fwhm_arcmin": fwhm_arcmin,
        "include_default_pixel_windows": False,
        "include_default_act_beams": False,
        "shear_m_bias_applied": False,
        "transfer_notes": transfer_notes,
        "caveat": (
            "Pasted profile smoothing is approximated as an isotropic Gaussian transfer. "
            "The finite-cap shear spin-2 roundtrip is not included in this transfer model."
        ),
    }
    options = {
        "transfer_functions": transfers,
        "transfer_ell": None,
        "include_default_pixel_windows": False,
        "include_default_act_beams": False,
        "shear_m_bias": None,
    }
    return options, metadata


def write_theory_product(
    path: Path,
    *,
    measurement_path: Path,
    names: Sequence[str],
    ell_band: np.ndarray,
    ell_smooth: np.ndarray,
    full_theory: Mapping[str, np.ndarray],
    resolved_theory: Mapping[str, np.ndarray],
    full_windowed: np.ndarray,
    resolved_windowed: np.ndarray,
    config: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(tmp, "w") as h5:
        h5.attrs["schema"] = "stage31_single_pz_cap600_theory_v1"
        h5.attrs["measurement_path"] = str(measurement_path)
        h5.attrs["config_json"] = json.dumps(config, default=gmt.to_jsonable, sort_keys=True)
        h5.create_dataset("ell_smooth", data=np.asarray(ell_smooth, dtype=np.float64))
        sg = h5.create_group("smooth_cls")
        for name in sorted(full_theory):
            g = sg.create_group(name)
            g.create_dataset("full_hod_floor10p5", data=np.asarray(full_theory[name], dtype=np.float64))
            if name in resolved_theory:
                g.create_dataset("resolved_log10Mgt11", data=np.asarray(resolved_theory[name], dtype=np.float64))
                g.create_dataset(
                    "unresolved_delta",
                    data=np.asarray(full_theory[name], dtype=np.float64) - np.asarray(resolved_theory[name], dtype=np.float64),
                )
        wg = h5.create_group("windowed")
        wg.create_dataset("spectrum_names", data=np.asarray(list(names), dtype=h5py.string_dtype("utf-8")))
        wg.create_dataset("ell", data=np.asarray(ell_band, dtype=np.float64))
        wg.create_dataset("full_hod_floor10p5", data=np.asarray(full_windowed, dtype=np.float64))
        wg.create_dataset("resolved_log10Mgt11", data=np.asarray(resolved_windowed, dtype=np.float64))
        wg.create_dataset("unresolved_delta", data=np.asarray(full_windowed - resolved_windowed, dtype=np.float64))
    os.replace(tmp, path)


def build_theory(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    measurement_path = Path(args.measurement) if args.measurement else default_measurement_path(config, "data", args.nside)
    if not measurement_path.exists():
        raise FileNotFoundError(f"Missing measurement product: {measurement_path}")
    cfg = merge_bestfit_params(config)
    with h5py.File(measurement_path, "r") as h5:
        measurement_config = json.loads(str(h5.attrs["config_json"]))
        ell_band = h5["joint/ell"][:]
    cfg["metadata"]["lmax"] = int(measurement_config["lmax"])
    cfg = gmt.compute_desi_nbar_comoving(cfg)
    pz_bin = pz_bin_from_config(config)
    pz_cfg = gmt.config_for_single_desi_pz(cfg, pz_bin)

    resolved_cut = float(config["godmax"].get("resolved_catalog_log10_m_min_hmsun", 11.0))
    transition_default = _resolve_transition_model(args.transition_model, "poweradd")
    gg_transition_model = _resolve_transition_model(args.gg_transition_model, transition_default)
    tsz_transition_model = _resolve_transition_model(args.tsz_transition_model, transition_default)
    galaxy_matter_transition_model = _resolve_transition_model(args.galaxy_matter_transition_model, transition_default)
    galaxy_electron_transition_model = _resolve_transition_model(args.galaxy_electron_transition_model, transition_default)
    transition_kwargs = {
        "gg_transition_model": gg_transition_model,
        "tsz_transition_model": tsz_transition_model,
        "galaxy_matter_transition_model": galaxy_matter_transition_model,
        "galaxy_electron_transition_model": galaxy_electron_transition_model,
    }
    full_wl = build_one_godmax_model(pz_cfg, is_cmb_lensing=False, **transition_kwargs)
    full_cmb = build_one_godmax_model(pz_cfg, is_cmb_lensing=True, **transition_kwargs)
    resolved_wl = build_one_godmax_model(
        pz_cfg,
        is_cmb_lensing=False,
        log10_mass_cut=resolved_cut,
        **transition_kwargs,
    )
    resolved_cmb = build_one_godmax_model(
        pz_cfg,
        is_cmb_lensing=True,
        log10_mass_cut=resolved_cut,
        **transition_kwargs,
    )

    ell_smooth = np.asarray(full_wl.ell_array, dtype=np.float64)
    full_theory = pz_theory_from_models(full_wl, full_cmb, pz_bin)
    resolved_theory = pz_theory_from_models(resolved_wl, resolved_cmb, pz_bin)
    av = np.asarray(cfg["metadata"].get("ksz_default_A_v_by_pz", np.full(4, np.nan)), dtype=np.float64)
    ksz_amplitudes = {pz_bin: float(av[pz_bin - 1])} if av.size >= pz_bin and np.isfinite(av[pz_bin - 1]) else None
    shear_m = cfg["metadata"].get("shear_m_bias_means")
    if bool(args.sim_matched_transfers):
        windowing_options, windowing_metadata = sim_matched_transfer_options(measurement_path)
    else:
        windowing_options = {
            "transfer_functions": None,
            "transfer_ell": None,
            "include_default_pixel_windows": True,
            "include_default_act_beams": True,
            "shear_m_bias": shear_m,
        }
        windowing_metadata = {
            "mode": "real_data_default_transfers",
            "measurement_path": str(measurement_path),
            "include_default_pixel_windows": True,
            "include_default_act_beams": True,
            "shear_m_bias_applied": shear_m is not None,
            "note": "Default wrapper behavior for real-data comparison products.",
        }
    full_vec, names = mpn.theory_to_data_vector(
        measurement_path,
        full_theory,
        ell=ell_smooth,
        ksz_velocity_amplitudes=ksz_amplitudes,
        shear_m_bias=windowing_options["shear_m_bias"],
        theory_shear_e_is_positive_kappa=True,
        transfer_functions=windowing_options["transfer_functions"],
        transfer_ell=windowing_options["transfer_ell"],
        include_default_pixel_windows=windowing_options["include_default_pixel_windows"],
        include_default_act_beams=windowing_options["include_default_act_beams"],
    )
    resolved_vec, _ = mpn.theory_to_data_vector(
        measurement_path,
        resolved_theory,
        ell=ell_smooth,
        ksz_velocity_amplitudes=ksz_amplitudes,
        shear_m_bias=windowing_options["shear_m_bias"],
        theory_shear_e_is_positive_kappa=True,
        transfer_functions=windowing_options["transfer_functions"],
        transfer_ell=windowing_options["transfer_ell"],
        include_default_pixel_windows=windowing_options["include_default_pixel_windows"],
        include_default_act_beams=windowing_options["include_default_act_beams"],
    )
    output = Path(args.output) if args.output else output_dir(config, "theory_subdir") / (
        f"{run_name_from_config(config)}_theory_for_{measurement_path.stem}.h5"
    )
    write_config = copy.deepcopy(dict(config))
    write_config["theory_transition_models"] = {
        "default": transition_default or "config",
        "gg_transition_model": gg_transition_model or "config",
        "tSZ_transition_model": tsz_transition_model or "config",
        "galaxy_matter_transition_model": galaxy_matter_transition_model or "config",
        "galaxy_electron_transition_model": galaxy_electron_transition_model or "config",
        "note": (
            "These overrides are applied after merging the best-fit parameter file. "
            "poweradd means direct 1h+2h addition; response multiplies the corresponding "
            "combined term by the matter response factor where implemented."
        ),
    }
    write_config["theory_windowing"] = copy.deepcopy(windowing_metadata)
    write_theory_product(
        output,
        measurement_path=measurement_path,
        names=names,
        ell_band=ell_band,
        ell_smooth=ell_smooth,
        full_theory=full_theory,
        resolved_theory=resolved_theory,
        full_windowed=full_vec,
        resolved_windowed=resolved_vec,
        config=write_config,
    )
    with h5py.File(output, "a") as h5:
        h5.attrs["theory_transition_models_json"] = json.dumps(
            write_config["theory_transition_models"], sort_keys=True
        )
        h5.attrs["theory_windowing_json"] = json.dumps(windowing_metadata, sort_keys=True, default=gmt.to_jsonable)
    print(
        json.dumps(
            {
                "output": str(output),
                "measurement": str(measurement_path),
                "spectra": names,
                "transition_models": write_config["theory_transition_models"],
                "windowing": windowing_metadata,
            },
            indent=2,
            default=gmt.to_jsonable,
        )
    )


def paste_split(args: argparse.Namespace) -> None:
    nside = int(args.nside)
    config = read_config(args.config)
    catalog_key = args.catalog or default_catalog_key(config)
    path = run_paste_split(
        args.config,
        catalog_key,
        split_index=int(args.split_index),
        num_splits=int(args.num_splits),
        nside=nside,
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
        pixel_workers=args.pixel_workers,
        pixel_start_method=args.pixel_start_method,
        pixel_backend=args.pixel_backend,
        query_disc_buffer_safety_factor=args.query_disc_buffer_safety_factor,
    )
    print(json.dumps({"output": str(path)}, indent=2))


def _parse_int_list(text: str | Sequence[int]) -> List[int]:
    if isinstance(text, str):
        return [int(item) for item in text.replace(" ", "").split(",") if item]
    return [int(item) for item in text]


def _parse_float_list(text: str | Sequence[float]) -> List[float]:
    if isinstance(text, str):
        return [float(item) for item in text.replace(" ", "").split(",") if item]
    return [float(item) for item in text]


def _parse_str_list(text: str | Sequence[str]) -> List[str]:
    if isinstance(text, str):
        return [item for item in text.replace(" ", "").split(",") if item]
    return [str(item) for item in text]


def _percentile_summary(values: np.ndarray, percentiles: Sequence[float] = (0, 1, 5, 16, 50, 84, 95, 99, 100)) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{int(p):02d}": None for p in percentiles}
    pct = np.percentile(arr, percentiles)
    return {f"p{int(p):02d}": float(v) for p, v in zip(percentiles, pct)}


def _gpu_memory_snapshot_mb() -> Optional[List[float]]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    values = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            return None
    return values or None


def _jsonable_attr(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_galaxies_only(path: Path | str) -> Tuple[np.ndarray, dict]:
    with h5py.File(path, "r") as handle:
        galaxies = handle["galaxies"][:]
        attrs = dict(handle.attrs)
    return galaxies, attrs


def _galaxy_density_paths(
    config: Mapping[str, object],
    catalog_key: str,
    nside: int,
    maps: Optional[str],
    num_splits: Optional[int],
) -> List[Path]:
    if maps:
        return [Path(maps).expanduser().resolve()]
    if num_splits is not None:
        return [
            partial_map_path(config, catalog_key, int(nside), split, int(num_splits))
            for split in range(int(num_splits))
        ]
    return [final_map_path(config, catalog_key, int(nside))]


def _density_target_info(
    config: Mapping[str, object],
    catalog_key: str,
    args: argparse.Namespace,
    area_deg2: float,
) -> dict:
    pz_bin = pz_bin_from_config(config)
    validation = config.get("validation", {})
    catalog_meta = config.get("catalogs", {}).get(catalog_key, {}).get("metadata", {})
    target_key = f"target_pz{pz_bin}_surface_density_per_deg2"
    retained_key = f"pz{pz_bin}_true_nz_retained_fraction"
    target_density = (
        float(args.target_density_per_deg2)
        if args.target_density_per_deg2 is not None
        else validation.get(target_key, validation.get("target_surface_density_per_deg2"))
    )
    retained_fraction = (
        float(args.retained_fraction)
        if args.retained_fraction is not None
        else validation.get(retained_key, catalog_meta.get(retained_key, 1.0))
    )
    if target_density is None:
        return {
            "pz_bin": int(pz_bin),
            "target_density_per_deg2": None,
            "retained_fraction": None,
            "target_count_raw": None,
            "target_count_retained_true_z": None,
        }
    target_density = float(target_density)
    retained_fraction = float(retained_fraction)
    return {
        "pz_bin": int(pz_bin),
        "target_density_key": target_key,
        "retained_fraction_key": retained_key,
        "target_density_per_deg2": target_density,
        "retained_fraction": retained_fraction,
        "target_density_retained_true_z_per_deg2": target_density * retained_fraction,
        "target_count_raw": target_density * float(area_deg2),
        "target_count_retained_true_z": target_density * retained_fraction * float(area_deg2),
    }


def _accumulate_galaxy_density(
    acc: dict,
    galaxies: np.ndarray,
    nside: int,
    cap_mask: np.ndarray,
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
    z_edges: Optional[np.ndarray],
) -> None:
    galaxies = np.asarray(galaxies)
    acc["n_rows"] += int(galaxies.shape[0]) if galaxies.ndim == 2 else 0
    if galaxies.ndim != 2 or galaxies.shape[0] == 0:
        return
    if galaxies.shape[1] < 2:
        raise ValueError(f"Expected at least ra/dec columns in galaxy catalog, got shape={galaxies.shape}.")
    valid = np.ones(galaxies.shape[0], dtype=bool)
    if galaxies.shape[1] > 5:
        valid &= np.asarray(galaxies[:, 5], dtype=np.float64) > 0.5
    valid &= np.isfinite(galaxies[:, 0]) & np.isfinite(galaxies[:, 1])
    if galaxies.shape[1] > 2:
        valid &= np.isfinite(galaxies[:, 2])
    acc["n_valid_buffer"] += int(np.count_nonzero(valid))
    if not np.any(valid):
        return

    gals = galaxies[valid]
    weights = np.asarray(gals[:, 7], dtype=np.float64) if gals.shape[1] > 7 else np.ones(len(gals), dtype=np.float64)
    pix = hp.ang2pix(int(nside), gals[:, 0], gals[:, 1], lonlat=True)
    in_cap = np.asarray(cap_mask[pix], dtype=np.float64) > 0.0
    in_exact_cap = angular_cap_mask(gals[:, 0], gals[:, 1], center_ra_deg, center_dec_deg, radius_deg)
    central = np.asarray(gals[:, 4], dtype=np.float64) > 0.5 if gals.shape[1] > 4 else np.zeros(len(gals), dtype=bool)
    satellite = ~central

    acc["weight_valid_buffer"] += float(np.sum(weights))
    acc["n_valid_in_cap"] += int(np.count_nonzero(in_cap))
    acc["weight_valid_in_cap"] += float(np.sum(weights[in_cap]))
    acc["n_valid_in_exact_cap"] += int(np.count_nonzero(in_exact_cap))
    acc["weight_valid_in_exact_cap"] += float(np.sum(weights[in_exact_cap]))
    acc["n_valid_outside_cap"] += int(np.count_nonzero(~in_cap))
    acc["weight_valid_outside_cap"] += float(np.sum(weights[~in_cap]))
    acc["n_central_buffer"] += int(np.count_nonzero(central))
    acc["n_satellite_buffer"] += int(np.count_nonzero(satellite))
    acc["n_central_in_cap"] += int(np.count_nonzero(central & in_cap))
    acc["n_satellite_in_cap"] += int(np.count_nonzero(satellite & in_cap))
    acc["n_central_in_exact_cap"] += int(np.count_nonzero(central & in_exact_cap))
    acc["n_satellite_in_exact_cap"] += int(np.count_nonzero(satellite & in_exact_cap))

    if gals.shape[1] > 2:
        z = np.asarray(gals[:, 2], dtype=np.float64)
        acc["z_min"] = float(np.nanmin(z)) if acc["z_min"] is None else float(min(acc["z_min"], np.nanmin(z)))
        acc["z_max"] = float(np.nanmax(z)) if acc["z_max"] is None else float(max(acc["z_max"], np.nanmax(z)))
        acc["z_sum_valid_buffer"] += float(np.sum(z))
        acc["z_sum_valid_in_cap"] += float(np.sum(z[in_cap]))
        acc["z_sum_valid_in_exact_cap"] += float(np.sum(z[in_exact_cap]))
        if z_edges is not None:
            hist_all, _ = np.histogram(z, bins=z_edges)
            hist_cap, _ = np.histogram(z[in_cap], bins=z_edges)
            hist_exact_cap, _ = np.histogram(z[in_exact_cap], bins=z_edges)
            acc["z_hist_valid_buffer"] += hist_all.astype(np.int64)
            acc["z_hist_valid_in_cap"] += hist_cap.astype(np.int64)
            acc["z_hist_valid_in_exact_cap"] += hist_exact_cap.astype(np.int64)


def _estimate_hod_mean_counts(
    config: Mapping[str, object],
    catalog_key: str,
    nside: int,
    cap_mask: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    platform = str(args.hod_platform)
    os.environ["PASTE_JAX_PLATFORMS"] = platform
    os.environ["JAX_PLATFORMS"] = platform
    if platform == "cpu":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    cfg_for_jax = copy.deepcopy(dict(config))
    cfg_for_jax.setdefault("pasting", {})
    cfg_for_jax["pasting"] = dict(cfg_for_jax["pasting"])
    cfg_for_jax["pasting"]["jax"] = dict(cfg_for_jax["pasting"].get("jax", {}))
    cfg_for_jax["pasting"]["jax"]["platforms"] = platform
    cfg_for_jax["pasting"]["jax"]["preallocate"] = False if platform == "cpu" else bool(
        cfg_for_jax["pasting"]["jax"].get("preallocate", True)
    )
    configure_jax_runtime_for_pasting(cfg_for_jax, verbose=not args.quiet)

    import jax
    import jax.numpy as jnp
    from jax import vmap
    from base_class import base_class
    from get_radial_profiles import Profiles
    from get_sim_maps import setup_sim_map, get_sim_map

    cat_path = catalog_path(config, catalog_key)
    catalog, attrs = load_halo_catalog(cat_path)
    n_full = int(len(catalog["z"]))
    n_hod = int(args.hod_max_halos)
    if n_hod > 0 and n_hod < n_full:
        sample = _sample_catalog_for_benchmark(catalog, n_hod, str(args.hod_sample_mode), int(args.hod_seed))
    else:
        sample = {key: np.asarray(value) for key, value in catalog.items()}
        n_hod = n_full

    halo_pix = hp.ang2pix(int(nside), sample["ra_deg"], sample["dec_deg"], lonlat=True)
    halo_in_cap = np.asarray(cap_mask[halo_pix], dtype=np.float64) > 0.0

    sim_params, halo_params, analysis, other_params = prepare_godmax_config(
        config,
        attrs,
        is_cmb_lensing=False,
        z_max=float(attrs.get("z_max", np.max(sample["z"]) if len(sample["z"]) else 0.5)),
        log10_mass_min=float(attrs.get("log10_m_min_hmsun", np.min(sample["log10M200c_hMsun"]))),
    )
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = Profiles(sim_params, halo_params, analysis, other_params, base_class_obj=base)
    setup_mock = {
        "nside": int(nside),
        "get_galmap": True,
        "get_ymap": False,
        "get_kSZmap": False,
        "get_taumap": False,
        "get_kappamap": False,
        "get_multi_kappamap": False,
        "get_baryonifiedmap": False,
        "smooth_profiles": bool(config["pasting"].get("smooth_profiles", True)),
        "return_sparse_maps": True,
        "store_projected_matter_maps": False,
    }
    setup = setup_sim_map(sim_params, halo_params, analysis, other_params, setup_mock, Profiles_obj=profiles)
    mock = {
        "nside": int(nside),
        "nearby_pix_all": np.asarray([0], dtype=np.int64),
        "pix_prop_all": jnp.zeros((1, 4), dtype=jnp.float32),
        "pix_unique": np.asarray([0], dtype=np.int64),
        "sort_idx": np.asarray([0], dtype=np.int64),
        "boundaries": np.asarray([0, 1], dtype=np.int64),
        "get_galmap": False,
        "get_ymap": False,
        "get_kSZmap": False,
        "get_taumap": False,
        "get_kappamap": False,
        "get_multi_kappamap": False,
        "get_baryonifiedmap": False,
        "smooth_profiles": bool(config["pasting"].get("smooth_profiles", True)),
        "return_sparse_maps": True,
        "store_projected_matter_maps": False,
    }
    hod = get_sim_map(sim_params, halo_params, analysis, other_params, mock, Profiles_obj=setup)
    eval_hod = jax.jit(vmap(lambda mass, z: hod.get_hod_params(mass, z)))

    expected_ncen = 0.0
    expected_nsat = 0.0
    expected_ncen_cap = 0.0
    expected_nsat_cap = 0.0
    chunk_size = max(1, int(args.hod_chunk_size))
    for start in range(0, n_hod, chunk_size):
        stop = min(start + chunk_size, n_hod)
        ncen, nsat = eval_hod(
            jnp.asarray(sample["M200c_hMsun"][start:stop], dtype=jnp.float64),
            jnp.asarray(sample["z"][start:stop], dtype=jnp.float32),
        )
        ncen_np = np.asarray(ncen, dtype=np.float64)
        nsat_np = np.asarray(nsat, dtype=np.float64)
        in_cap = halo_in_cap[start:stop]
        expected_ncen += float(np.sum(ncen_np))
        expected_nsat += float(np.sum(nsat_np))
        expected_ncen_cap += float(np.sum(ncen_np[in_cap]))
        expected_nsat_cap += float(np.sum(nsat_np[in_cap]))

    scale_to_full_catalog = float(n_full / max(1, n_hod))
    representative_scaling = bool(n_hod == n_full or str(args.hod_sample_mode) == "random")
    return {
        "catalog_path": str(cat_path),
        "jax_backend": str(jax.default_backend()),
        "jax_devices": [str(device) for device in jax.devices()],
        "n_halos_full_catalog": int(n_full),
        "n_halos_evaluated": int(n_hod),
        "sample_mode": str(args.hod_sample_mode) if n_hod < n_full else "full",
        "scale_to_full_catalog": scale_to_full_catalog,
        "scaled_counts_representative": representative_scaling,
        "n_halo_centers_in_cap_evaluated": int(np.count_nonzero(halo_in_cap)),
        "expected_ncen_buffer": expected_ncen,
        "expected_nsat_buffer": expected_nsat,
        "expected_ngal_buffer": expected_ncen + expected_nsat,
        "expected_ncen_halo_centers_in_cap": expected_ncen_cap,
        "expected_nsat_halo_centers_in_cap": expected_nsat_cap,
        "expected_ngal_halo_centers_in_cap": expected_ncen_cap + expected_nsat_cap,
        "expected_ngal_buffer_scaled": (expected_ncen + expected_nsat) * scale_to_full_catalog,
        "expected_ngal_halo_centers_in_cap_scaled": (expected_ncen_cap + expected_nsat_cap) * scale_to_full_catalog,
    }


def diagnose_galaxy_density(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    catalog_key = args.catalog or default_catalog_key(config)
    nside = int(args.nside or config["pasting"].get("nside", 1024))
    center_ra, center_dec, radius_deg = require_cap_center(config)
    cap_mask = cap_pixel_mask(nside, center_ra, center_dec, radius_deg)
    cap_area_deg2 = float(np.sum(cap_mask > 0.0) * hp.nside2pixarea(nside, degrees=True))
    config_area_deg2 = float(config.get("sky_patch", {}).get("area_deg2", cap_area_deg2))
    area_for_target = float(args.area_deg2) if args.area_deg2 is not None else config_area_deg2
    paths = _galaxy_density_paths(config, catalog_key, nside, args.maps, args.num_splits)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing pasted map HDF5 files:\n" + "\n".join(missing))

    z_edges = None
    catalog_cfg = config.get("catalogs", {}).get(catalog_key, {})
    if int(args.z_bins) > 0:
        z_min = float(args.z_min if args.z_min is not None else catalog_cfg.get("z_min", 0.0))
        z_max = float(args.z_max if args.z_max is not None else catalog_cfg.get("z_max", 2.0))
        z_edges = np.linspace(z_min, z_max, int(args.z_bins) + 1)

    acc = {
        "n_rows": 0,
        "n_valid_buffer": 0,
        "n_valid_in_cap": 0,
        "n_valid_in_exact_cap": 0,
        "n_valid_outside_cap": 0,
        "weight_valid_buffer": 0.0,
        "weight_valid_in_cap": 0.0,
        "weight_valid_in_exact_cap": 0.0,
        "weight_valid_outside_cap": 0.0,
        "n_central_buffer": 0,
        "n_satellite_buffer": 0,
        "n_central_in_cap": 0,
        "n_satellite_in_cap": 0,
        "n_central_in_exact_cap": 0,
        "n_satellite_in_exact_cap": 0,
        "z_min": None,
        "z_max": None,
        "z_sum_valid_buffer": 0.0,
        "z_sum_valid_in_cap": 0.0,
        "z_sum_valid_in_exact_cap": 0.0,
        "z_hist_valid_buffer": np.zeros(int(args.z_bins), dtype=np.int64) if z_edges is not None else None,
        "z_hist_valid_in_cap": np.zeros(int(args.z_bins), dtype=np.int64) if z_edges is not None else None,
        "z_hist_valid_in_exact_cap": np.zeros(int(args.z_bins), dtype=np.int64) if z_edges is not None else None,
    }
    source_rows = []
    attrs_last = {}
    for path in paths:
        galaxies, attrs = _load_galaxies_only(path)
        attrs_last = attrs
        before = dict(acc)
        _accumulate_galaxy_density(acc, galaxies, nside, cap_mask, center_ra, center_dec, radius_deg, z_edges)
        source_rows.append(
            {
                "path": str(path),
                "n_rows": int(galaxies.shape[0]) if galaxies.ndim == 2 else 0,
                "n_valid_buffer": int(acc["n_valid_buffer"] - before["n_valid_buffer"]),
                "n_valid_in_cap": int(acc["n_valid_in_cap"] - before["n_valid_in_cap"]),
                "n_valid_in_exact_cap": int(acc["n_valid_in_exact_cap"] - before["n_valid_in_exact_cap"]),
                "attrs": {
                    key: _jsonable_attr(attrs[key])
                    for key in sorted(attrs)
                    if key in {"n_galaxies", "split_index", "num_splits", "n_split_halos", "n_input_halos", "split_strategy"}
                },
            }
        )

    target = _density_target_info(config, catalog_key, args, area_for_target)
    density = {
        "nside": int(nside),
        "cap_center_ra_deg": float(center_ra),
        "cap_center_dec_deg": float(center_dec),
        "cap_radius_deg": float(radius_deg),
        "cap_area_deg2_from_nside_mask": cap_area_deg2,
        "area_deg2_used_for_target": area_for_target,
        "config_area_deg2": config_area_deg2,
        "config_selected_actual_cap_area_deg2": float(config.get("sky_patch", {}).get("selected_actual_cap_area_deg2", np.nan)),
        "n_rows": int(acc["n_rows"]),
        "n_valid_buffer": int(acc["n_valid_buffer"]),
        "n_valid_in_cap": int(acc["n_valid_in_cap"]),
        "n_valid_in_exact_cap": int(acc["n_valid_in_exact_cap"]),
        "n_valid_outside_cap": int(acc["n_valid_outside_cap"]),
        "valid_outside_cap_fraction": float(acc["n_valid_outside_cap"] / max(1, acc["n_valid_buffer"])),
        "surface_density_in_cap_per_deg2": float(acc["n_valid_in_cap"] / area_for_target),
        "surface_density_exact_cap_per_deg2": float(acc["n_valid_in_exact_cap"] / area_for_target),
        "surface_density_buffer_valid_per_target_area_deg2": float(acc["n_valid_buffer"] / area_for_target),
        "weighted_valid_buffer": float(acc["weight_valid_buffer"]),
        "weighted_valid_in_cap": float(acc["weight_valid_in_cap"]),
        "weighted_valid_in_exact_cap": float(acc["weight_valid_in_exact_cap"]),
        "weighted_surface_density_in_cap_per_deg2": float(acc["weight_valid_in_cap"] / area_for_target),
        "weighted_surface_density_exact_cap_per_deg2": float(acc["weight_valid_in_exact_cap"] / area_for_target),
        "n_central_buffer": int(acc["n_central_buffer"]),
        "n_satellite_buffer": int(acc["n_satellite_buffer"]),
        "n_central_in_cap": int(acc["n_central_in_cap"]),
        "n_satellite_in_cap": int(acc["n_satellite_in_cap"]),
        "n_central_in_exact_cap": int(acc["n_central_in_exact_cap"]),
        "n_satellite_in_exact_cap": int(acc["n_satellite_in_exact_cap"]),
        "satellite_fraction_in_cap": float(acc["n_satellite_in_cap"] / max(1, acc["n_valid_in_cap"])),
        "satellite_fraction_exact_cap": float(acc["n_satellite_in_exact_cap"] / max(1, acc["n_valid_in_exact_cap"])),
        "mean_z_valid_buffer": float(acc["z_sum_valid_buffer"] / acc["n_valid_buffer"]) if acc["n_valid_buffer"] else None,
        "mean_z_valid_in_cap": float(acc["z_sum_valid_in_cap"] / acc["n_valid_in_cap"]) if acc["n_valid_in_cap"] else None,
        "mean_z_valid_in_exact_cap": float(acc["z_sum_valid_in_exact_cap"] / acc["n_valid_in_exact_cap"]) if acc["n_valid_in_exact_cap"] else None,
        "z_min_valid": acc["z_min"],
        "z_max_valid": acc["z_max"],
    }
    if target["target_density_per_deg2"] is not None:
        retained_count = float(target["target_count_retained_true_z"])
        raw_count = float(target["target_count_raw"])
        density.update(
            {
                "target_count_raw": raw_count,
                "target_count_retained_true_z": retained_count,
                "target_density_per_deg2": float(target["target_density_per_deg2"]),
                "target_density_retained_true_z_per_deg2": float(target["target_density_retained_true_z_per_deg2"]),
                "retained_fraction": float(target["retained_fraction"]),
                "deficit_vs_raw_target_count": float(acc["n_valid_in_cap"] - raw_count),
                "deficit_vs_retained_target_count": float(acc["n_valid_in_cap"] - retained_count),
                "deficit_exact_cap_vs_raw_target_count": float(acc["n_valid_in_exact_cap"] - raw_count),
                "deficit_exact_cap_vs_retained_target_count": float(acc["n_valid_in_exact_cap"] - retained_count),
                "ratio_to_raw_target": float(acc["n_valid_in_cap"] / raw_count) if raw_count > 0 else None,
                "ratio_to_retained_target": float(acc["n_valid_in_cap"] / retained_count) if retained_count > 0 else None,
                "ratio_exact_cap_to_raw_target": float(acc["n_valid_in_exact_cap"] / raw_count) if raw_count > 0 else None,
                "ratio_exact_cap_to_retained_target": float(acc["n_valid_in_exact_cap"] / retained_count) if retained_count > 0 else None,
                "required_multiplier_to_raw_target": float(raw_count / max(1, acc["n_valid_in_cap"])),
                "required_multiplier_to_retained_target": float(retained_count / max(1, acc["n_valid_in_cap"])),
                "required_multiplier_exact_cap_to_raw_target": float(raw_count / max(1, acc["n_valid_in_exact_cap"])),
                "required_multiplier_exact_cap_to_retained_target": float(retained_count / max(1, acc["n_valid_in_exact_cap"])),
            }
        )

    z_hist = None
    if z_edges is not None:
        z_hist = {
            "edges": z_edges.tolist(),
            "valid_buffer": acc["z_hist_valid_buffer"].astype(int).tolist(),
            "valid_in_cap": acc["z_hist_valid_in_cap"].astype(int).tolist(),
            "valid_in_exact_cap": acc["z_hist_valid_in_exact_cap"].astype(int).tolist(),
        }

    hod_mean = None
    if bool(args.include_hod_mean):
        hod_mean = _estimate_hod_mean_counts(config, catalog_key, nside, cap_mask, args)
        target_count = target.get("target_count_retained_true_z")
        if target_count is not None:
            hod_mean["ratio_expected_halo_centers_in_cap_to_retained_target"] = float(
                hod_mean["expected_ngal_halo_centers_in_cap_scaled"] / float(target_count)
            )
            hod_mean["ratio_expected_buffer_to_retained_target"] = float(
                hod_mean["expected_ngal_buffer_scaled"] / float(target_count)
            )

    payload = {
        "config": str(args.config),
        "catalog_key": str(catalog_key),
        "nside": int(nside),
        "input_mode": "explicit_maps" if args.maps else ("partial_splits" if args.num_splits is not None else "combined_map"),
        "sources": source_rows,
        "target": target,
        "density": density,
        "z_histogram": z_hist,
        "hod_mean": hod_mean,
        "last_map_attrs": {key: _jsonable_attr(value) for key, value in attrs_last.items()},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = Path(args.output) if args.output else output_dir(config, "measurement_subdir") / (
        f"galaxy_density_diagnostic_{pz_measurement_tag(config)}_nside{nside}.json"
    )
    ensure_under_xdesi(out_path.resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, out_path)
    print(
        json.dumps(
            {
                "output": str(out_path),
                "n_valid_buffer": density["n_valid_buffer"],
                "n_valid_in_cap": density["n_valid_in_cap"],
                "n_valid_in_exact_cap": density["n_valid_in_exact_cap"],
                "surface_density_in_cap_per_deg2": density["surface_density_in_cap_per_deg2"],
                "surface_density_exact_cap_per_deg2": density["surface_density_exact_cap_per_deg2"],
                "ratio_to_retained_target": density.get("ratio_to_retained_target"),
                "ratio_exact_cap_to_retained_target": density.get("ratio_exact_cap_to_retained_target"),
            },
            indent=2,
        )
    )


def _sample_catalog_for_benchmark(catalog: Mapping[str, np.ndarray], n_halos: int, mode: str, seed: int) -> dict:
    n_total = len(catalog["z"])
    n_halos = min(int(n_halos), int(n_total))
    if n_halos <= 0:
        raise ValueError("Benchmark sample size must be positive.")
    if mode == "head":
        idx = np.arange(n_halos, dtype=np.int64)
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(n_total, size=n_halos, replace=False).astype(np.int64))
    elif mode == "largest-paint":
        radius = np.asarray(catalog["R200c_hMpc"], dtype=np.float64) / np.maximum(catalog["DA_hMpc"], 1.0e-8)
        idx = np.argsort(radius)[-n_halos:]
        idx.sort()
    elif mode == "largest-mass":
        idx = np.argsort(np.asarray(catalog["M200c_hMsun"], dtype=np.float64))[-n_halos:]
        idx.sort()
    elif mode == "lowest-z":
        idx = np.argsort(np.asarray(catalog["z"], dtype=np.float64))[:n_halos]
        idx.sort()
    elif mode == "highest-z":
        idx = np.argsort(np.asarray(catalog["z"], dtype=np.float64))[-n_halos:]
        idx.sort()
    else:
        raise ValueError(f"Unknown benchmark sample mode {mode!r}.")
    return {key: np.asarray(value)[idx] for key, value in catalog.items()}


def benchmark_pixel_work(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    catalog_key = args.catalog or default_catalog_key(config)
    cat_path = catalog_path(config, catalog_key)
    catalog, attrs = load_halo_catalog(cat_path)
    nside = int(args.nside or config["pasting"].get("nside", 512))
    max_paint = float(args.max_paint or config["pasting"]["max_paint_R200c_factor"])
    batch_size = int(args.pixel_batch_size or config["pasting"].get("pixel_batch_size", 2000))
    pixel_gc_collect_every_n_batches = int(
        config["pasting"].get("pixel_gc_collect_every_n_batches", 0)
        if args.pixel_gc_collect_every_n_batches is None
        else args.pixel_gc_collect_every_n_batches
    )
    single_pixel_angle_factor = float(args.single_pixel_angle_factor)
    stencil_pixel_angle_factor = float(getattr(args, "stencil_pixel_angle_factor", 1.0))
    pixel_backend = str(getattr(args, "pixel_backend", "healpy"))
    query_disc_buffer_safety_factor = float(getattr(args, "query_disc_buffer_safety_factor", 2.0))
    precompute_pixel_groups = not bool(getattr(args, "no_precompute_pixel_groups", False))
    sample_sizes = _parse_int_list(args.sample_sizes)
    worker_values = _parse_int_list(args.workers)
    chunksize_values = _parse_int_list(args.pool_chunksizes)
    out_path = Path(args.output) if args.output else output_dir(config, "measurement_subdir") / (
        f"pixel_work_benchmark_{pz_measurement_tag(config)}_nside{nside}.json"
    )
    ensure_under_xdesi(out_path.resolve())

    rows = []
    print(
        f"[benchmark:pixel] catalog={cat_path} n_total={len(catalog['z']):,} "
        f"nside={nside} max_paint={max_paint} batch_size={batch_size} "
        f"gc_collect_every_n_batches={pixel_gc_collect_every_n_batches} "
        f"single_pixel_angle_factor={single_pixel_angle_factor} "
        f"stencil_pixel_angle_factor={stencil_pixel_angle_factor} "
        f"pixel_backend={pixel_backend} precompute_pixel_groups={precompute_pixel_groups}",
        flush=True,
    )
    for n_halos in sample_sizes:
        sample = _sample_catalog_for_benchmark(catalog, n_halos, args.sample_mode, int(args.seed))
        exact_pixels = None
        if bool(args.compare_exact):
            t_exact = time.perf_counter()
            exact_pixels = build_pixel_work_package(
                sample,
                nside,
                max_paint,
                batch_size,
                workers=int(args.exact_workers),
                start_method=str(args.pixel_start_method),
                pool=None,
                pool_chunksize=int(args.exact_pool_chunksize),
                single_pixel_angle_factor=0.0,
                stencil_pixel_angle_factor=stencil_pixel_angle_factor,
                pixel_backend="healpy",
                precompute_pixel_groups=precompute_pixel_groups,
                pixel_gc_collect_every_n_batches=pixel_gc_collect_every_n_batches,
                verbose=not args.quiet,
            )
            print(
                f"[benchmark:pixel] exact reference halos={len(sample['z']):,} "
                f"time={time.perf_counter() - t_exact:.2f}s pairs={len(exact_pixels['nearby_pix_all']) if exact_pixels else 0:,}",
                flush=True,
            )
        for workers in worker_values:
            for pool_chunksize in chunksize_values:
                t0 = time.perf_counter()
                pixels = build_pixel_work_package(
                    sample,
                    nside,
                    max_paint,
                    batch_size,
                    workers=int(workers),
                    start_method=str(args.pixel_start_method),
                    pool=None,
                    pool_chunksize=int(pool_chunksize),
                    single_pixel_angle_factor=single_pixel_angle_factor,
                    stencil_pixel_angle_factor=stencil_pixel_angle_factor,
                    pixel_backend=pixel_backend,
                    query_disc_buffer_safety_factor=query_disc_buffer_safety_factor,
                    precompute_pixel_groups=precompute_pixel_groups,
                    pixel_gc_collect_every_n_batches=pixel_gc_collect_every_n_batches,
                    verbose=not args.quiet,
                )
                dt = time.perf_counter() - t0
                n_pairs = int(len(pixels["nearby_pix_all"])) if pixels is not None else 0
                row = {
                    "n_halos": int(len(sample["z"])),
                    "n_pairs": int(n_pairs),
                    "pairs_per_halo": float(n_pairs / max(1, len(sample["z"]))),
                    "nside": int(nside),
                    "max_paint_R200c_factor": float(max_paint),
                    "pixel_batch_size": int(batch_size),
                    "pixel_gc_collect_every_n_batches": int(pixel_gc_collect_every_n_batches),
                    "workers": int(workers),
                    "pool_chunksize": int(pool_chunksize),
                    "pixel_backend": str(pixel_backend),
                    "query_disc_buffer_safety_factor": float(query_disc_buffer_safety_factor),
                    "precompute_pixel_groups": bool(precompute_pixel_groups),
                    "single_pixel_angle_factor": float(single_pixel_angle_factor),
                    "stencil_pixel_angle_factor": float(stencil_pixel_angle_factor),
                    "n_single_pixel_shortcut": int(pixels.get("n_single_pixel_shortcut", 0)) if pixels is not None else 0,
                    "n_ring": int(pixels.get("n_ring", 0)) if pixels is not None else 0,
                    "n_query_disc": int(pixels.get("n_query_disc", 0)) if pixels is not None else 0,
                    "n_query_disc_buffer_grows": int(pixels.get("n_query_disc_buffer_grows", 0)) if pixels is not None else 0,
                    "sample_mode": str(args.sample_mode),
                    "runtime_s": float(dt),
                    "halos_per_s": float(len(sample["z"]) / max(dt, 1.0e-9)),
                    "pairs_per_s": float(n_pairs / max(dt, 1.0e-9)),
                }
                if exact_pixels is not None and pixels is not None:
                    same_size = len(exact_pixels["nearby_pix_all"]) == len(pixels["nearby_pix_all"])
                    row["compare_exact_same_pair_count"] = bool(same_size)
                    row["compare_exact_pixel_mismatch_count"] = (
                        int(np.sum(exact_pixels["nearby_pix_all"] != pixels["nearby_pix_all"])) if same_size else None
                    )
                    row["compare_exact_pixel_mismatch_fraction"] = (
                        float(row["compare_exact_pixel_mismatch_count"] / max(1, len(pixels["nearby_pix_all"]))) if same_size else None
                    )
                    row["compare_exact_max_distance_hMpc"] = (
                        float(np.max(np.abs(exact_pixels["distances"] - pixels["distances"]))) if same_size else None
                    )
                rows.append(row)
                print(
                    "[benchmark:pixel] "
                    f"halos={row['n_halos']:,} workers={workers} chunksize={pool_chunksize} "
                    f"pairs={n_pairs:,} time={dt:.2f}s "
                    f"halos/s={row['halos_per_s']:.1f} pairs/s={row['pairs_per_s']:.1f}",
                    flush=True,
                )
                del pixels

    payload = {
        "config": str(args.config),
        "catalog_key": str(catalog_key),
        "catalog_path": str(cat_path),
        "catalog_attrs": {key: _jsonable_attr(value) for key, value in attrs.items()},
        "rows": rows,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, out_path)
    print(json.dumps({"output": str(out_path), "n_rows": len(rows)}, indent=2))


def _finite_map_diff(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return {"max_abs": None, "rms": None, "n": 0}
    diff = a[mask] - b[mask]
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff**2))),
        "n": int(mask.sum()),
    }


def benchmark_gpu_chunk(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    configure_jax_runtime_for_pasting(config, verbose=not args.quiet)
    catalog_key = args.catalog or default_catalog_key(config)
    cat_path = catalog_path(config, catalog_key)
    catalog, attrs = load_halo_catalog(cat_path)
    nside = int(args.nside or config["pasting"].get("nside", 512))
    sample = _sample_catalog_for_benchmark(catalog, int(args.n_halos), args.sample_mode, int(args.seed))
    max_paint = float(args.max_paint or config["pasting"]["max_paint_R200c_factor"])
    batch_size = int(args.pixel_batch_size or config["pasting"].get("pixel_batch_size", 2000))
    pixel_gc_collect_every_n_batches = int(
        config["pasting"].get("pixel_gc_collect_every_n_batches", 0)
        if getattr(args, "pixel_gc_collect_every_n_batches", None) is None
        else args.pixel_gc_collect_every_n_batches
    )
    pixel_workers = int(args.pixel_workers)
    pixel_pool_chunksize = int(args.pixel_pool_chunksize)
    single_pixel_angle_factor = float(args.single_pixel_angle_factor)
    stencil_pixel_angle_factor = float(getattr(args, "stencil_pixel_angle_factor", 1.0))
    pixel_backend = str(getattr(args, "pixel_backend", "healpy"))
    query_disc_buffer_safety_factor = float(getattr(args, "query_disc_buffer_safety_factor", 2.0))

    print(
        f"[benchmark:gpu] build pixels n_halos={len(sample['z']):,} nside={nside} "
        f"workers={pixel_workers} pool_chunksize={pixel_pool_chunksize} "
        f"pixel_backend={pixel_backend}",
        flush=True,
    )
    gpu_memory_before_mb = _gpu_memory_snapshot_mb()
    t_pixel = time.perf_counter()
    pixels = build_pixel_work_package(
        sample,
        nside,
        max_paint,
        batch_size,
        workers=pixel_workers,
        start_method=str(args.pixel_start_method),
        pool=None,
        pool_chunksize=pixel_pool_chunksize,
        single_pixel_angle_factor=single_pixel_angle_factor,
        stencil_pixel_angle_factor=stencil_pixel_angle_factor,
        pixel_backend=pixel_backend,
        query_disc_buffer_safety_factor=query_disc_buffer_safety_factor,
        precompute_pixel_groups=True,
        pixel_gc_collect_every_n_batches=pixel_gc_collect_every_n_batches,
        verbose=not args.quiet,
    )
    pixel_time = time.perf_counter() - t_pixel
    gpu_memory_after_pixels_mb = _gpu_memory_snapshot_mb()
    if pixels is None:
        raise RuntimeError("Pixel benchmark sample produced no pixel work.")

    import jax
    import jax.numpy as jnp
    from base_class import base_class
    from get_radial_profiles import Profiles
    from get_sim_maps import setup_sim_map, get_sim_map

    print(f"[benchmark:gpu] backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    sim_params, halo_params, analysis, other_params = prepare_godmax_config(
        config,
        attrs,
        is_cmb_lensing=False,
        z_max=float(attrs.get("z_max", np.max(sample["z"]) if len(sample["z"]) else 0.5)),
        log10_mass_min=float(attrs.get("log10_m_min_hmsun", np.min(sample["log10M200c_hMsun"]))),
    )
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = Profiles(sim_params, halo_params, analysis, other_params, base_class_obj=base)
    store_projected_matter_maps = bool(config["pasting"].get("store_projected_matter_maps", True))
    use_multi_kappa_maps = bool(config["pasting"].get("use_multi_kappa_maps", False))
    wl_source_bins = wl_source_bins_from_config(config) if bool(config["pasting"].get("get_kappa_wl", True)) else []
    get_kappa_cmb = bool(config["pasting"].get("get_kappa_cmb", True))
    common_setup = {
        "nside": int(nside),
        "smooth_profiles": bool(config["pasting"].get("smooth_profiles", True)),
        "profile_timing": True,
        "return_sparse_maps": bool(config["pasting"].get("return_sparse_maps", True)),
        "store_projected_matter_maps": store_projected_matter_maps,
        "galaxy_population_chunk_size": int(config["pasting"].get("galaxy_population_chunk_size", 20000)),
        "galaxy_max_gals_round_to": int(config["pasting"].get("galaxy_max_gals_round_to", 16)),
        "galaxy_population_group_by_max_gals": bool(config["pasting"].get("galaxy_population_group_by_max_gals", False)),
        "galaxy_population_backend": str(args.galaxy_population_backend or config["pasting"].get("galaxy_population_backend", "padded_precomputed")),
        "galaxy_compact_max_satellite_groups": int(args.galaxy_compact_max_satellite_groups or config["pasting"].get("galaxy_compact_max_satellite_groups", 32)),
        "get_galmap": bool(args.include_galaxies),
        "get_ymap": True,
        "get_kSZmap": True,
        "get_taumap": True,
        "get_kappamap": not use_multi_kappa_maps,
        "get_multi_kappamap": use_multi_kappa_maps,
        "multi_kappa_source_bins": [int(source_bin) - 1 for source_bin in wl_source_bins],
        "multi_kappa_include_cmb": bool(use_multi_kappa_maps and get_kappa_cmb),
        "get_baryonifiedmap": bool(config["pasting"].get("get_baryonifiedmap", store_projected_matter_maps)),
        "kappa_source_bin": int(args.kappa_source_bin),
    }
    pix_prop_all = np.column_stack(
        (
            np.log(pixels["distances"]),
            pixels["z"],
            pixels["logM"],
            pixels["vlos"],
        )
    ).astype(np.float32, copy=False)
    mock_common = {
        "halo_z": jnp.array(sample["z"], dtype=jnp.float32),
        "halo_ra": jnp.array(sample["ra_deg"], dtype=jnp.float32),
        "halo_dec": jnp.array(sample["dec_deg"], dtype=jnp.float32),
        "halo_M": jnp.array(sample["M200c_hMsun"], dtype=jnp.float64),
        "halo_DA": jnp.array(sample["DA_hMpc"], dtype=jnp.float32),
        "halo_vlos": jnp.array(sample["vlos_kms"], dtype=jnp.float32),
        "nearby_pix_all": pixels["nearby_pix_all"],
        "pix_unique": pixels.get("pix_unique"),
        "sort_idx": pixels.get("sort_idx"),
        "boundaries": pixels.get("boundaries"),
        "pix_prop_all": jnp.array(pix_prop_all, dtype=jnp.float32),
        "random_seed": int(config["pasting"].get("random_seed", 42)),
    }

    outputs = {}
    rows = []
    reusable_setup = None
    for fused in (False, True):
        setup_params = dict(common_setup, use_fused_profile_maps=bool(fused))
        setup = setup_sim_map(
            sim_params,
            halo_params,
            analysis,
            other_params,
            setup_params,
            Profiles_obj=profiles if reusable_setup is None else reusable_setup,
        )
        if reusable_setup is None:
            reusable_setup = setup
        mock_params = dict(setup_params, **mock_common)
        print(f"[benchmark:gpu] get_sim_map fused={fused} start", flush=True)
        t0 = time.perf_counter()
        mock_map = get_sim_map(sim_params, halo_params, analysis, other_params, mock_params, Profiles_obj=setup)
        runtime = time.perf_counter() - t0
        gpu_memory_after_map_mb = _gpu_memory_snapshot_mb()
        row = {
            "fused": bool(fused),
            "runtime_s": float(runtime),
            "timing_results": copy.deepcopy(getattr(mock_map, "timing_results", {})),
            "galaxy_population_diagnostics": copy.deepcopy(getattr(mock_map, "galaxy_population_diagnostics", {})),
            "gpu_memory_used_mb_after_map": gpu_memory_after_map_mb,
        }
        rows.append(row)
        outputs[bool(fused)] = {
            "ymap": np.asarray(getattr(mock_map, "ymap_final", np.array([], dtype=np.float32))),
            "ksz": np.asarray(getattr(mock_map, "kszmap_final", np.array([], dtype=np.float32))),
            "tau": np.asarray(getattr(mock_map, "taumap_final", np.array([], dtype=np.float32))),
            "rhom": np.asarray(getattr(mock_map, "rhommap_final", np.array([], dtype=np.float32))),
            "kappa": np.asarray(getattr(mock_map, "kappamap_final", np.array([], dtype=np.float32))),
            "rhom_dmo": np.asarray(getattr(mock_map, "rhom_dmo_map_final", np.array([], dtype=np.float32))),
        }
        for label, value in getattr(mock_map, "multi_kappamaps_final", {}).items():
            if isinstance(value, tuple) and len(value) == 2:
                outputs[bool(fused)][f"multi_kappa_{label}"] = np.asarray(value[1], dtype=np.float32)
            else:
                outputs[bool(fused)][f"multi_kappa_{label}"] = np.asarray(value, dtype=np.float32)
        print(f"[benchmark:gpu] fused={fused} runtime={runtime:.2f}s", flush=True)

    diffs = {
        name: _finite_map_diff(outputs[False][name], outputs[True][name])
        for name in outputs[False]
        if outputs[False][name].size and outputs[True][name].size
    }
    out_path = Path(args.output) if args.output else output_dir(config, "measurement_subdir") / (
        f"gpu_chunk_benchmark_{pz_measurement_tag(config)}_nside{nside}_halos{len(sample['z'])}.json"
    )
    ensure_under_xdesi(out_path.resolve())
    payload = {
        "config": str(args.config),
        "catalog_key": str(catalog_key),
        "catalog_path": str(cat_path),
        "nside": int(nside),
        "n_halos": int(len(sample["z"])),
        "n_pairs": int(len(pixels["nearby_pix_all"])),
        "n_single_pixel_shortcut": int(pixels.get("n_single_pixel_shortcut", 0)),
        "pixel_time_s": float(pixel_time),
        "gpu_memory_used_mb_before": gpu_memory_before_mb,
        "gpu_memory_used_mb_after_pixels": gpu_memory_after_pixels_mb,
        "sample_mode": str(args.sample_mode),
        "pixel_batch_size": int(batch_size),
        "pixel_gc_collect_every_n_batches": int(pixel_gc_collect_every_n_batches),
        "pixel_workers": int(pixel_workers),
        "pixel_pool_chunksize": int(pixel_pool_chunksize),
        "pixel_start_method": str(args.pixel_start_method),
        "pixel_backend": str(pixel_backend),
        "include_galaxies": bool(args.include_galaxies),
        "galaxy_population_backend": str(common_setup["galaxy_population_backend"]),
        "query_disc_buffer_safety_factor": float(query_disc_buffer_safety_factor),
        "stencil_pixel_angle_factor": float(stencil_pixel_angle_factor),
        "n_query_disc": int(pixels.get("n_query_disc", 0)),
        "n_ring": int(pixels.get("n_ring", 0)),
        "n_query_disc_buffer_grows": int(pixels.get("n_query_disc_buffer_grows", 0)),
        "single_pixel_angle_factor": float(single_pixel_angle_factor),
        "store_projected_matter_maps": bool(store_projected_matter_maps),
        "use_multi_kappa_maps": bool(use_multi_kappa_maps),
        "rows": rows,
        "diffs_fused_minus_unfused": diffs,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, out_path)
    print(json.dumps({"output": str(out_path), "diffs": diffs}, indent=2))


def diagnose_hod_stress(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    catalog_key = args.catalog or default_catalog_key(config)
    cat_path = catalog_path(config, catalog_key)
    catalog, attrs = load_halo_catalog(cat_path)
    n_total = int(len(catalog["z"]))
    if n_total <= 0:
        raise ValueError(f"Catalog is empty: {cat_path}")

    platform = str(args.hod_platform)
    os.environ["PASTE_JAX_PLATFORMS"] = platform
    os.environ["JAX_PLATFORMS"] = platform
    if platform == "cpu":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    cfg_for_jax = copy.deepcopy(dict(config))
    cfg_for_jax.setdefault("pasting", {})
    cfg_for_jax["pasting"] = dict(cfg_for_jax["pasting"])
    cfg_for_jax["pasting"]["jax"] = dict(cfg_for_jax["pasting"].get("jax", {}))
    cfg_for_jax["pasting"]["jax"]["platforms"] = platform
    cfg_for_jax["pasting"]["jax"]["preallocate"] = False if platform == "cpu" else bool(
        cfg_for_jax["pasting"]["jax"].get("preallocate", True)
    )
    configure_jax_runtime_for_pasting(cfg_for_jax, verbose=not args.quiet)

    import jax
    import jax.numpy as jnp
    from jax import vmap
    from jax.random import PRNGKey, fold_in, split
    from base_class import base_class
    from get_radial_profiles import Profiles
    from get_sim_maps import setup_sim_map, get_sim_map

    nside_values = _parse_int_list(args.nside_list)
    sample_modes = _parse_str_list(args.sample_modes)
    max_paint = float(args.max_paint or config["pasting"]["max_paint_R200c_factor"])
    hod_chunk_size = max(1, int(args.hod_chunk_size))
    pixel_sample_halos = max(1, int(args.pixel_sample_halos))
    pixel_batch_size = int(args.pixel_batch_size or config["pasting"].get("pixel_batch_size", 2000))
    pixel_gc_collect_every_n_batches = int(
        config["pasting"].get("pixel_gc_collect_every_n_batches", 0)
        if getattr(args, "pixel_gc_collect_every_n_batches", None) is None
        else args.pixel_gc_collect_every_n_batches
    )
    pixel_workers = int(args.pixel_workers)
    pixel_pool_chunksize = int(args.pixel_pool_chunksize)
    pixel_backend = str(args.pixel_backend)
    single_pixel_angle_factor = float(args.single_pixel_angle_factor)
    stencil_pixel_angle_factor = float(getattr(args, "stencil_pixel_angle_factor", 1.0))

    sim_params, halo_params, analysis, other_params = prepare_godmax_config(
        config,
        attrs,
        is_cmb_lensing=False,
        z_max=float(attrs.get("z_max", np.max(catalog["z"]))),
        log10_mass_min=float(attrs.get("log10_m_min_hmsun", np.min(catalog["log10M200c_hMsun"]))),
    )
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = Profiles(sim_params, halo_params, analysis, other_params, base_class_obj=base)
    setup_mock = {
        "nside": int(nside_values[0]),
        "get_galmap": True,
        "get_ymap": False,
        "get_kSZmap": False,
        "get_taumap": False,
        "get_kappamap": False,
        "get_multi_kappamap": False,
        "get_baryonifiedmap": False,
        "smooth_profiles": bool(config["pasting"].get("smooth_profiles", True)),
        "return_sparse_maps": True,
        "store_projected_matter_maps": False,
    }
    setup = setup_sim_map(sim_params, halo_params, analysis, other_params, setup_mock, Profiles_obj=profiles)
    mock = {
        "nside": int(nside_values[0]),
        "nearby_pix_all": np.asarray([0], dtype=np.int64),
        "pix_prop_all": jnp.zeros((1, 4), dtype=jnp.float32),
        "pix_unique": np.asarray([0], dtype=np.int64),
        "sort_idx": np.asarray([0], dtype=np.int64),
        "boundaries": np.asarray([0, 1], dtype=np.int64),
        "get_galmap": False,
        "get_ymap": False,
        "get_kSZmap": False,
        "get_taumap": False,
        "get_kappamap": False,
        "get_multi_kappamap": False,
        "get_baryonifiedmap": False,
        "smooth_profiles": bool(config["pasting"].get("smooth_profiles", True)),
        "return_sparse_maps": True,
        "store_projected_matter_maps": False,
    }
    hod = get_sim_map(sim_params, halo_params, analysis, other_params, mock, Profiles_obj=setup)
    eval_hod = jax.jit(vmap(lambda mass, z: hod.get_hod_params(mass, z)))

    z = np.asarray(catalog["z"], dtype=np.float64)
    mass = np.asarray(catalog["M200c_hMsun"], dtype=np.float64)
    log_mass = np.asarray(catalog["log10M200c_hMsun"], dtype=np.float64)
    da = np.asarray(catalog["DA_hMpc"], dtype=np.float64)
    r200 = np.asarray(catalog["R200c_hMpc"], dtype=np.float64)
    angle_rad = max_paint * r200 / np.maximum(da, 1.0e-8)

    mean_ncen = np.empty(n_total, dtype=np.float32)
    mean_nsat = np.empty(n_total, dtype=np.float32)
    ncen_realized = np.empty(n_total, dtype=np.int32)
    nsat_realized = np.empty(n_total, dtype=np.int32)
    nsat_raw = np.empty(n_total, dtype=np.int32)
    max_sats = np.empty(n_total, dtype=np.int32)

    t_hod = time.perf_counter()
    base_key = PRNGKey(int(args.seed))
    for start in range(0, n_total, hod_chunk_size):
        stop = min(start + hod_chunk_size, n_total)
        ncen_j, nsat_j = eval_hod(
            jnp.asarray(mass[start:stop], dtype=jnp.float64),
            jnp.asarray(z[start:stop], dtype=jnp.float32),
        )
        ncen_np = np.asarray(ncen_j, dtype=np.float32)
        nsat_np = np.asarray(nsat_j, dtype=np.float32)
        mean_ncen[start:stop] = ncen_np
        mean_nsat[start:stop] = nsat_np
        raw_max_gals = np.ceil(nsat_np + np.sqrt(np.maximum(nsat_np, 0.0))).astype(np.int64) + 2
        round_to = max(1, int(config["pasting"].get("galaxy_max_gals_round_to", 16)))
        max_gals = (
            np.ceil(np.maximum(2, raw_max_gals) / float(round_to)).astype(np.int64)
            * int(round_to)
        )
        max_sats[start:stop] = np.maximum(1, max_gals - 1).astype(np.int32)
        keys = split(fold_in(base_key, int(start)), stop - start)
        ncen_s, nsat_s, nsat_raw_s = hod.sample_hod_counts_from_means(
            keys,
            jnp.asarray(ncen_np, dtype=jnp.float32),
            jnp.asarray(nsat_np, dtype=jnp.float32),
            jnp.asarray(max_sats[start:stop], dtype=jnp.int32),
            bool(config["pasting"].get("use_poisson_centrals", False)),
        )
        ncen_realized[start:stop] = np.asarray(ncen_s, dtype=np.int32)
        nsat_realized[start:stop] = np.asarray(nsat_s, dtype=np.int32)
        nsat_raw[start:stop] = np.asarray(nsat_raw_s, dtype=np.int32)
    hod_eval_time = time.perf_counter() - t_hod

    if args.z_edges:
        z_edges = np.asarray(_parse_float_list(args.z_edges), dtype=np.float64)
    else:
        z_min = float(args.z_min) if args.z_min is not None else float(np.nanmin(z))
        z_max = float(args.z_max) if args.z_max is not None else float(np.nanmax(z))
        z_edges = np.linspace(z_min, z_max, int(args.z_bins) + 1)
    if len(z_edges) < 2 or not np.all(np.diff(z_edges) > 0):
        raise ValueError("--z-edges must contain at least two increasing values.")

    shell_rows = []
    pixel_rows = []
    rng = np.random.default_rng(int(args.seed))
    hod_score = mean_ncen.astype(np.float64) + mean_nsat.astype(np.float64)
    for iz in range(len(z_edges) - 1):
        lo = float(z_edges[iz])
        hi = float(z_edges[iz + 1])
        if iz == len(z_edges) - 2:
            mask = (z >= lo) & (z <= hi)
        else:
            mask = (z >= lo) & (z < hi)
        idx_bin = np.flatnonzero(mask)
        if idx_bin.size == 0:
            continue
        clipped = nsat_raw[idx_bin] - nsat_realized[idx_bin]
        shell_rows.append(
            {
                "z_bin_index": int(iz),
                "z_min": lo,
                "z_max": hi,
                "n_halos": int(idx_bin.size),
                "log10M200c_hMsun": _percentile_summary(log_mass[idx_bin]),
                "z": _percentile_summary(z[idx_bin]),
                "paint_angle_arcmin": _percentile_summary(angle_rad[idx_bin] * (180.0 / np.pi) * 60.0),
                "expected_ncen": float(np.sum(mean_ncen[idx_bin], dtype=np.float64)),
                "expected_nsat": float(np.sum(mean_nsat[idx_bin], dtype=np.float64)),
                "expected_ngal": float(np.sum(mean_ncen[idx_bin] + mean_nsat[idx_bin], dtype=np.float64)),
                "realized_ncen": int(np.sum(ncen_realized[idx_bin])),
                "realized_nsat": int(np.sum(nsat_realized[idx_bin])),
                "realized_ngal": int(np.sum(ncen_realized[idx_bin]) + np.sum(nsat_realized[idx_bin])),
                "n_nonzero_satellite_halos": int(np.count_nonzero(nsat_realized[idx_bin] > 0)),
                "n_clipped_halos": int(np.count_nonzero(clipped > 0)),
                "n_clipped_sats": int(np.sum(clipped)),
                "max_nsat_raw": int(np.max(nsat_raw[idx_bin])),
                "max_nsat_clipped": int(np.max(nsat_realized[idx_bin])),
                "max_sats_capacity": int(np.max(max_sats[idx_bin])),
            }
        )

        for mode in sample_modes:
            n_sample = min(pixel_sample_halos, int(idx_bin.size))
            if mode == "random":
                sample_idx = np.sort(rng.choice(idx_bin, size=n_sample, replace=False).astype(np.int64))
            elif mode == "head":
                sample_idx = idx_bin[:n_sample]
            elif mode == "largest-paint":
                sample_idx = idx_bin[np.argsort(angle_rad[idx_bin])[-n_sample:]]
                sample_idx.sort()
            elif mode == "largest-mass":
                sample_idx = idx_bin[np.argsort(mass[idx_bin])[-n_sample:]]
                sample_idx.sort()
            elif mode == "lowest-z":
                sample_idx = idx_bin[np.argsort(z[idx_bin])[:n_sample]]
                sample_idx.sort()
            elif mode == "highest-hod":
                sample_idx = idx_bin[np.argsort(hod_score[idx_bin])[-n_sample:]]
                sample_idx.sort()
            else:
                raise ValueError(f"Unknown stress sample mode {mode!r}.")
            sample = {key: np.asarray(value)[sample_idx] for key, value in catalog.items()}
            for nside in nside_values:
                row = {
                    "z_bin_index": int(iz),
                    "z_min": lo,
                    "z_max": hi,
                    "sample_mode": str(mode),
                    "nside": int(nside),
                    "n_sample_halos": int(n_sample),
                    "sample_expected_ncen": float(np.sum(mean_ncen[sample_idx], dtype=np.float64)),
                    "sample_expected_nsat": float(np.sum(mean_nsat[sample_idx], dtype=np.float64)),
                    "sample_realized_ncen": int(np.sum(ncen_realized[sample_idx])),
                    "sample_realized_nsat": int(np.sum(nsat_realized[sample_idx])),
                    "sample_paint_angle_arcmin": _percentile_summary(angle_rad[sample_idx] * (180.0 / np.pi) * 60.0),
                }
                if bool(args.include_pixel_work):
                    t0 = time.perf_counter()
                    pixels = build_pixel_work_package(
                        sample,
                        int(nside),
                        max_paint,
                        pixel_batch_size,
                        workers=pixel_workers,
                        start_method=str(args.pixel_start_method),
                        pool=None,
                        pool_chunksize=pixel_pool_chunksize,
                        single_pixel_angle_factor=single_pixel_angle_factor,
                        stencil_pixel_angle_factor=stencil_pixel_angle_factor,
                        pixel_backend=pixel_backend,
                        query_disc_buffer_safety_factor=float(args.query_disc_buffer_safety_factor),
                        precompute_pixel_groups=True,
                        pixel_gc_collect_every_n_batches=pixel_gc_collect_every_n_batches,
                        verbose=not args.quiet,
                    )
                    runtime = time.perf_counter() - t0
                    n_pairs = int(len(pixels["nearby_pix_all"])) if pixels is not None else 0
                    row.update(
                        {
                            "pixel_time_s": float(runtime),
                            "n_pairs": int(n_pairs),
                            "pairs_per_halo": float(n_pairs / max(1, n_sample)),
                            "n_single_pixel_shortcut": int(pixels.get("n_single_pixel_shortcut", 0)) if pixels is not None else 0,
                            "n_ring": int(pixels.get("n_ring", 0)) if pixels is not None else 0,
                            "n_query_disc": int(pixels.get("n_query_disc", 0)) if pixels is not None else 0,
                            "n_query_disc_buffer_grows": int(pixels.get("n_query_disc_buffer_grows", 0)) if pixels is not None else 0,
                        }
                    )
                    del pixels
                pixel_rows.append(row)

    out_path = Path(args.output) if args.output else output_dir(config, "measurement_subdir") / (
        f"hod_stress_{pz_measurement_tag(config)}_halos{n_total}.json"
    )
    ensure_under_xdesi(out_path.resolve())
    payload = {
        "config": str(args.config),
        "catalog_key": str(catalog_key),
        "catalog_path": str(cat_path),
        "catalog_attrs": {key: _jsonable_attr(value) for key, value in attrs.items()},
        "n_halos": int(n_total),
        "nside_list": [int(value) for value in nside_values],
        "sample_modes": sample_modes,
        "pixel_sample_halos": int(pixel_sample_halos),
        "max_paint_R200c_factor": float(max_paint),
        "pixel_workers": int(pixel_workers),
        "pixel_pool_chunksize": int(pixel_pool_chunksize),
        "single_pixel_angle_factor": float(single_pixel_angle_factor),
        "stencil_pixel_angle_factor": float(stencil_pixel_angle_factor),
        "pixel_backend": str(pixel_backend),
        "jax_backend": str(jax.default_backend()),
        "jax_devices": [str(device) for device in jax.devices()],
        "gpu_memory_used_mb_after": _gpu_memory_snapshot_mb(),
        "hod_eval_time_s": float(hod_eval_time),
        "z_edges": z_edges.astype(float).tolist(),
        "shell_rows": shell_rows,
        "pixel_rows": pixel_rows,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, out_path)
    print(json.dumps({"output": str(out_path), "shell_rows": len(shell_rows), "pixel_rows": len(pixel_rows)}, indent=2))


def benchmark_healpix_functions(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    catalog_key = args.catalog or default_catalog_key(config)
    cat_path = catalog_path(config, catalog_key)
    catalog, attrs = load_halo_catalog(cat_path)
    nside = int(args.nside or config["pasting"].get("nside", 512))
    sample = _sample_catalog_for_benchmark(catalog, int(args.n_halos), args.sample_mode, int(args.seed))
    max_paint = float(args.max_paint or config["pasting"]["max_paint_R200c_factor"])
    ra = np.asarray(sample["ra_deg"], dtype=np.float64)
    dec = np.asarray(sample["dec_deg"], dtype=np.float64)
    angles = max_paint * np.asarray(sample["R200c_hMpc"], dtype=np.float64) / np.maximum(sample["DA_hMpc"], 1.0e-8)
    rows = []

    def add_row(name: str, runtime_s: float, **values) -> None:
        row = {"name": name, "runtime_s": float(runtime_s)}
        row.update(values)
        rows.append(row)
        print(f"[benchmark:healpix-func] {name} time={runtime_s:.4f}s", flush=True)

    t0 = time.perf_counter()
    hp_pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    add_row("healpy.ang2pix", time.perf_counter() - t0, n=int(len(hp_pix)))

    t0 = time.perf_counter()
    hp_lon, hp_lat = hp.pix2ang(nside, hp_pix, lonlat=True)
    add_row("healpy.pix2ang", time.perf_counter() - t0, n=int(len(hp_pix)))

    t0 = time.perf_counter()
    hp_neigh = hp.get_all_neighbours(nside, hp_pix, nest=False)
    add_row("healpy.get_all_neighbours", time.perf_counter() - t0, shape=list(np.shape(hp_neigh)))

    query_count = min(int(args.query_disc_count), int(len(ra)))
    if query_count > 0:
        query_order = np.argsort(angles)[::-1][:query_count]
        t0 = time.perf_counter()
        query_lengths = []
        for idx in query_order:
            vec = hp.ang2vec(float(ra[idx]), float(dec[idx]), lonlat=True)
            query_lengths.append(len(hp.query_disc(nside, vec, float(angles[idx]), inclusive=False)))
        add_row(
            "healpy.query_disc.loop",
            time.perf_counter() - t0,
            n=int(query_count),
            mean_len=float(np.mean(query_lengths)) if query_lengths else 0.0,
            max_len=int(np.max(query_lengths)) if query_lengths else 0,
        )

        safety = float(args.query_disc_buffer_safety_factor)
        max_len = max(
            16,
            max(
                int(math.ceil(safety * 6.0 * nside * nside * (1.0 - math.cos(float(angles[idx]))))) + 64
                for idx in query_order
            ),
        )
        buff = np.empty(max_len, dtype=np.int64)
        t0 = time.perf_counter()
        buff_lengths = []
        buff_grows = 0
        for idx in query_order:
            vec = hp.ang2vec(float(ra[idx]), float(dec[idx]), lonlat=True)
            while True:
                try:
                    out = hp.query_disc(nside, vec, float(angles[idx]), inclusive=False, buff=buff)
                    buff_lengths.append(len(out))
                    break
                except ValueError as exc:
                    if "Buffer too small" not in str(exc):
                        raise
                    buff = np.empty(len(buff) * 2, dtype=np.int64)
                    buff_grows += 1
        add_row(
            "healpy.query_disc.buff_loop",
            time.perf_counter() - t0,
            n=int(query_count),
            mean_len=float(np.mean(buff_lengths)) if buff_lengths else 0.0,
            max_len=int(np.max(buff_lengths)) if buff_lengths else 0,
            buffer_len=int(len(buff)),
            buffer_grows=int(buff_grows),
        )

    if args.jax_healpy_path:
        jax_path = str(Path(args.jax_healpy_path).resolve())
        if jax_path not in sys.path:
            sys.path.insert(0, jax_path)
    if args.run_jax_healpy:
        os.environ.setdefault("JAX_PLATFORMS", str(args.jax_device))
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            import jax_healpy as jhp
        except Exception as exc:
            add_row("jax_healpy.import_failed", 0.0, error=f"{type(exc).__name__}: {exc}")
        else:
            jra = jnp.asarray(ra)
            jdec = jnp.asarray(dec)
            t0 = time.perf_counter()
            jpix = jhp.ang2pix(nside, jra, jdec, lonlat=True)
            jpix.block_until_ready()
            jpix_np = np.asarray(jpix)
            add_row(
                "jax_healpy.ang2pix",
                time.perf_counter() - t0,
                n=int(len(jpix_np)),
                mismatch_count=int(np.count_nonzero(jpix_np != hp_pix)),
                backend=str(jax.default_backend()),
            )

            t0 = time.perf_counter()
            jlon, jlat = jhp.pix2ang(nside, jpix, lonlat=True)
            jlon.block_until_ready()
            add_row(
                "jax_healpy.pix2ang",
                time.perf_counter() - t0,
                max_abs_lon_diff=float(np.max(np.abs(np.asarray(jlon) - hp_lon))) if len(hp_lon) else 0.0,
                max_abs_lat_diff=float(np.max(np.abs(np.asarray(jlat) - hp_lat))) if len(hp_lat) else 0.0,
            )

            t0 = time.perf_counter()
            jneigh = jhp.get_all_neighbours(nside, jpix, nest=False, get_center=True)
            jneigh.block_until_ready()
            add_row("jax_healpy.get_all_neighbours_center", time.perf_counter() - t0, shape=list(np.shape(np.asarray(jneigh))))

            if query_count > 0 and bool(args.include_jax_query_disc):
                qvecs = np.asarray([hp.ang2vec(float(ra[idx]), float(dec[idx]), lonlat=True) for idx in query_order], dtype=np.float64)
                qradii = np.asarray([float(angles[idx]) for idx in query_order], dtype=np.float64)
                max_length = max(
                    16,
                    max(
                        int(math.ceil(safety * 6.0 * nside * nside * (1.0 - math.cos(float(radius))))) + 64
                        for radius in qradii
                    ),
                )
                npix = hp.nside2npix(nside)

                @jax.jit
                def _batched_query_disc(vecs, radii):
                    return jax.vmap(
                        lambda vec, radius: jhp.query_disc(
                            nside,
                            vec,
                            radius,
                            inclusive=False,
                            max_length=max_length,
                        )
                    )(vecs, radii)

                t0 = time.perf_counter()
                jout = _batched_query_disc(jnp.asarray(qvecs), jnp.asarray(qradii))
                jout.block_until_ready()
                batched_runtime = time.perf_counter() - t0
                jout_np = np.asarray(jout)
                mismatch = 0
                lengths = []
                for row, idx in zip(jout_np, query_order):
                    valid = np.sort(row[row < npix])
                    lengths.append(int(len(valid)))
                    ref_vec = hp.ang2vec(float(ra[idx]), float(dec[idx]), lonlat=True)
                    ref = np.sort(hp.query_disc(nside, ref_vec, float(angles[idx]), inclusive=False))
                    mismatch += int(not np.array_equal(valid, ref))
                add_row(
                    "jax_healpy.query_disc.vmap",
                    batched_runtime,
                    n=int(query_count),
                    max_length=int(max_length),
                    output_shape=list(jout_np.shape),
                    mean_len=float(np.mean(lengths)) if lengths else 0.0,
                    max_len=int(np.max(lengths)) if lengths else 0,
                    mismatch_queries=int(mismatch),
                    backend=str(jax.default_backend()),
                )

            if query_count > 0 and bool(args.include_jax_query_disc_loop):
                t0 = time.perf_counter()
                mismatch = 0
                for idx in query_order:
                    vec = jnp.asarray(hp.ang2vec(float(ra[idx]), float(dec[idx]), lonlat=True))
                    max_length = max(
                        16,
                        int(math.ceil(safety * 6.0 * nside * nside * (1.0 - math.cos(float(angles[idx]))))) + 64,
                    )
                    jout = jhp.query_disc(nside, vec, float(angles[idx]), inclusive=False, max_length=max_length)
                    jout.block_until_ready()
                    valid = np.asarray(jout)
                    valid = np.sort(valid[valid < hp.nside2npix(nside)])
                    ref_vec = hp.ang2vec(float(ra[idx]), float(dec[idx]), lonlat=True)
                    ref = np.sort(hp.query_disc(nside, ref_vec, float(angles[idx]), inclusive=False))
                    mismatch += int(not np.array_equal(valid, ref))
                add_row("jax_healpy.query_disc.loop", time.perf_counter() - t0, n=int(query_count), mismatch_queries=int(mismatch))

    out_path = Path(args.output) if args.output else output_dir(config, "measurement_subdir") / (
        f"healpix_function_benchmark_{pz_measurement_tag(config)}_nside{nside}_halos{len(sample['z'])}.json"
    )
    ensure_under_xdesi(out_path.resolve())
    payload = {
        "config": str(args.config),
        "catalog_key": str(catalog_key),
        "catalog_path": str(cat_path),
        "catalog_attrs": {key: _jsonable_attr(value) for key, value in attrs.items()},
        "nside": int(nside),
        "n_halos": int(len(sample["z"])),
        "sample_mode": str(args.sample_mode),
        "rows": rows,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, out_path)
    print(json.dumps({"output": str(out_path), "n_rows": len(rows)}, indent=2))


def combine_maps(args: argparse.Namespace) -> None:
    config = read_config(args.config)
    catalog_key = args.catalog or default_catalog_key(config)
    path = combine_partial_maps(
        args.config,
        catalog_key,
        num_splits=int(args.num_splits),
        nside=int(args.nside),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps({"output": str(path)}, indent=2))


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    ensure_under_xdesi(path.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=gmt.to_jsonable)
        handle.write("\n")
    os.replace(tmp_path, path)


def _direct_field_summary(diagnostics: Sequence[Mapping[str, object]], map_keys: Sequence[str]) -> dict:
    field_fracs = np.asarray([float(row.get("field_mass_fraction_cache", np.nan)) for row in diagnostics], dtype=np.float64)
    halo_fracs = np.asarray([float(row.get("halo_mass_fraction_cache", np.nan)) for row in diagnostics], dtype=np.float64)
    return {
        "n_shells": int(len(diagnostics)),
        "map_keys": [str(key) for key in map_keys],
        "field_mass_fraction_mean": float(np.nanmean(field_fracs)) if field_fracs.size else np.nan,
        "field_mass_fraction_min": float(np.nanmin(field_fracs)) if field_fracs.size else np.nan,
        "field_mass_fraction_max": float(np.nanmax(field_fracs)) if field_fracs.size else np.nan,
        "halo_mass_fraction_mean": float(np.nanmean(halo_fracs)) if halo_fracs.size else np.nan,
        "negative_count_fine_pixels_total": int(sum(int(row.get("negative_count_fine_pixels", 0)) for row in diagnostics)),
        "negative_count_parent_pixels_total": int(sum(int(row.get("negative_count_parent_pixels", 0)) for row in diagnostics)),
        "normalization": "field delta and kSZ momentum are divided by mean all-particle shell counts",
    }


def _direct_field_cache_diagnostic(path: Path, nside: int, shell_index: int, selected_index: int) -> dict:
    _, attrs = psh.load_direct_field_shell_cache(path, nside)
    keys = (
        "step_id",
        "z_lo",
        "z_hi",
        "z_mid",
        "chi_lo_hMpc",
        "chi_hi_hMpc",
        "input_total_count_sum",
        "input_halo_count_sum",
        "input_field_count_sum",
        "output_field_count_sum",
        "input_total_momentum_los_sum",
        "input_halo_momentum_los_sum",
        "input_field_momentum_los_sum",
        "output_field_momentum_los_sum",
        "mean_total_counts_out",
        "mean_field_counts_out",
        "field_mass_fraction",
        "halo_mass_fraction",
        "negative_count_fine_pixels",
        "negative_count_parent_pixels",
        "min_field_count_fine",
        "min_field_count_parent",
        "cache_runtime_sec",
    )
    row = {
        "shell_index": int(shell_index),
        "selected_index": int(selected_index),
        "cache_path": str(path),
    }
    for key in keys:
        if key in attrs:
            row[key] = _jsonable_attr(attrs[key])
    return row


def cache_direct_field_shells(args: argparse.Namespace) -> None:
    """Precompute direct non-halo field shell caches for later map products."""

    config = read_config(args.config)
    catalog_key = args.catalog or default_catalog_key(config)
    nside = int(args.nside or config["pasting"].get("nside", 1024))
    catalog_cfg = config.get("catalogs", {}).get(catalog_key, {})
    z_min = float(args.z_min if args.z_min is not None else catalog_cfg.get("z_min", 1.0e-4))
    z_max = float(args.z_max if args.z_max is not None else catalog_cfg.get("z_max", 0.5))
    total_root = Path(args.total_root).expanduser().resolve()
    halo_root = Path(args.halo_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve() if args.cache_root else psh.particle_shell_cache_root(config, nside)
    ensure_under_xdesi(cache_root)

    shell_index_mod = int(args.shell_index_mod)
    shell_index_rem = int(args.shell_index_rem)
    if shell_index_mod <= 0:
        raise ValueError("--shell-index-mod must be positive.")
    if shell_index_rem < 0 or shell_index_rem >= shell_index_mod:
        raise ValueError("--shell-index-rem must satisfy 0 <= rem < mod.")

    shell_meta = psh.discover_matched_total_halo_shells(
        total_root,
        halo_root,
        z_min=z_min,
        z_max=z_max,
        max_shells=args.max_shells,
    )
    if not shell_meta:
        raise RuntimeError(f"No matched total/halo shells selected for z=[{z_min}, {z_max}]")

    selected = [(idx, meta) for idx, meta in enumerate(shell_meta) if idx % shell_index_mod == shell_index_rem]
    if not selected:
        raise RuntimeError(
            f"No shells assigned to shell-index-rem={shell_index_rem} with shell-index-mod={shell_index_mod}; "
            f"selected from {len(shell_meta)} total shells."
        )

    diagnostics = []
    for selected_index, (shell_index, meta) in enumerate(selected):
        if not bool(args.quiet):
            print(
                f"[direct-field-cache] task {shell_index_rem}/{shell_index_mod} "
                f"shell {shell_index + 1}/{len(shell_meta)} {meta['step_id']}",
                flush=True,
            )
        path = psh.read_or_create_direct_field_shell_cache(
            meta,
            nside,
            cache_root,
            overwrite=bool(args.overwrite_field_cache),
            batch_parent_pixels=int(args.batch_parent_pixels),
            clip_negative_counts=bool(args.clip_negative_field_counts),
        )
        diagnostics.append(_direct_field_cache_diagnostic(path, nside, shell_index, selected_index))

    field_fracs = np.asarray([float(row.get("field_mass_fraction", np.nan)) for row in diagnostics], dtype=np.float64)
    output = Path(args.output).expanduser().resolve() if args.output else output_dir(config, "measurement_subdir") / (
        f"direct_field_shell_cache_{pz_measurement_tag(config)}_nside{nside}_task{shell_index_rem}of{shell_index_mod}.json"
    )
    payload = {
        "config": str(args.config),
        "catalog_key": str(catalog_key),
        "nside": int(nside),
        "total_root": str(total_root),
        "halo_root": str(halo_root),
        "cache_root": str(cache_root),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "n_shells_total": int(len(shell_meta)),
        "n_shells_selected": int(len(selected)),
        "shell_index_mod": int(shell_index_mod),
        "shell_index_rem": int(shell_index_rem),
        "batch_parent_pixels": int(args.batch_parent_pixels),
        "clip_negative_counts": bool(args.clip_negative_field_counts),
        "overwrite_field_cache": bool(args.overwrite_field_cache),
        "field_mass_fraction_mean": float(np.nanmean(field_fracs)) if field_fracs.size else np.nan,
        "negative_count_fine_pixels_total": int(sum(int(row.get("negative_count_fine_pixels", 0)) for row in diagnostics)),
        "negative_count_parent_pixels_total": int(sum(int(row.get("negative_count_parent_pixels", 0)) for row in diagnostics)),
        "shells": diagnostics,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_atomic(output, payload)
    print(json.dumps({"output": str(output), "n_shells_selected": len(selected), "cache_root": str(cache_root)}, indent=2))


def build_direct_field_map(args: argparse.Namespace) -> None:
    """Build pasted halo-profile plus direct non-halo field-shell map product."""

    config = read_config(args.config)
    catalog_key = args.catalog or default_catalog_key(config)
    nside = int(args.nside or config["pasting"].get("nside", 1024))
    map_path = Path(args.maps).expanduser().resolve() if args.maps else final_map_path(config, catalog_key, nside)
    if not map_path.exists():
        raise FileNotFoundError(f"Missing base pasted map HDF5: {map_path}")
    maps, galaxies, attrs = load_maps_h5(map_path)

    field_keys_all = (
        "map_kappa_cmb",
        "map_kappa_wl",
        "map_kappa_wl_tomo2",
        "map_kappa_wl_tomo3",
        "map_kappa_wl_tomo4",
        "map_tau",
        "map_ksz",
    )
    field_keys = [key for key in field_keys_all if key in maps]
    if not field_keys:
        raise RuntimeError(f"No direct-field target datasets found in {map_path}; available maps={sorted(maps)}")

    catalog_cfg = config.get("catalogs", {}).get(catalog_key, {})
    z_min = float(args.z_min if args.z_min is not None else catalog_cfg.get("z_min", attrs.get("z_min", 1.0e-4)))
    z_max = float(args.z_max if args.z_max is not None else catalog_cfg.get("z_max", attrs.get("z_max", 0.5)))
    log10_mass_min = float(
        args.log10_mass_min
        if args.log10_mass_min is not None
        else config.get("godmax", {}).get(
            "resolved_catalog_log10_m_min_hmsun",
            catalog_cfg.get("log10_m_min_hmsun", attrs.get("log10_m_min_hmsun", 11.0)),
        )
    )
    total_root = Path(args.total_root).expanduser().resolve()
    halo_root = Path(args.halo_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve() if args.cache_root else psh.particle_shell_cache_root(config, nside)
    ensure_under_xdesi(cache_root)

    shell_meta = psh.discover_matched_total_halo_shells(
        total_root,
        halo_root,
        z_min=z_min,
        z_max=z_max,
        max_shells=args.max_shells,
    )
    if not shell_meta:
        raise RuntimeError(f"No matched total/halo shells selected for z=[{z_min}, {z_max}]")

    gg_transition_model = None if str(args.gg_transition_model).lower() in {"", "none", "config"} else str(args.gg_transition_model)
    cls_cmb = build_theory_cls(
        args.config,
        catalog_key,
        is_cmb_lensing=True,
        log10_mass_min=log10_mass_min,
        z_max=z_max,
        gg_transition_model=gg_transition_model,
    )
    cls_wl = build_theory_cls(
        args.config,
        catalog_key,
        is_cmb_lensing=False,
        log10_mass_min=log10_mass_min,
        z_max=z_max,
        gg_transition_model=gg_transition_model,
    )
    wl_source_bins = wl_source_bins_from_config(config) if bool(config.get("pasting", {}).get("get_kappa_wl", True)) else []
    weights_by_step = {
        str(meta["step_id"]): psh.compute_dataset_shell_weights(
            meta,
            cls_cmb,
            cls_wl,
            wl_source_bins=wl_source_bins,
            mode=str(args.shell_weight_mode),
            n_samples=int(args.shell_weight_nsamples),
        )
        for meta in shell_meta
    }

    cache_paths = {}
    for idx, meta in enumerate(shell_meta, 1):
        if not bool(args.quiet):
            print(f"[direct-field] cache {idx}/{len(shell_meta)} {meta['step_id']}", flush=True)
        cache_paths[str(meta["step_id"])] = psh.read_or_create_direct_field_shell_cache(
            meta,
            nside,
            cache_root,
            overwrite=bool(args.overwrite_field_cache),
            batch_parent_pixels=int(args.batch_parent_pixels),
            clip_negative_counts=bool(args.clip_negative_field_counts),
        )

    direct_maps, diagnostics = psh.build_direct_field_maps(
        shell_meta,
        cache_paths,
        weights_by_step,
        nside,
        map_keys=field_keys,
    )
    for key, value in direct_maps.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Direct field map {key} contains non-finite values.")

    output = Path(args.output).expanduser().resolve() if args.output else map_path.with_name(f"{map_path.stem}_plus_direct_field_shells.h5")
    ensure_under_xdesi(output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it.")

    out_maps = {key: np.array(value, copy=True) for key, value in maps.items()}
    for key in field_keys:
        out_maps[key] = (np.asarray(out_maps[key], dtype=np.float32) + np.asarray(direct_maps[key], dtype=np.float32)).astype(np.float32)

    summary = _direct_field_summary(diagnostics, field_keys)
    diagnostics_path = output.with_suffix(output.suffix + ".direct_field_shells.json")
    out_attrs = dict(attrs)
    out_attrs.update(
        {
            "map_product": "pasted_plus_direct_nonhalo_field_shells",
            "base_pasted_map_path": str(map_path),
            "direct_field_total_shell_root": str(total_root),
            "direct_field_halo_shell_root": str(halo_root),
            "direct_field_cache_root": str(cache_root),
            "direct_field_diagnostics_path": str(diagnostics_path),
            "direct_field_z_min": float(z_min),
            "direct_field_z_max": float(z_max),
            "direct_field_n_shells": int(len(shell_meta)),
            "direct_field_map_keys_json": field_keys,
            "direct_field_shell_weight_mode": str(args.shell_weight_mode),
            "direct_field_shell_weight_nsamples": int(args.shell_weight_nsamples),
            "direct_field_normalization": "delta_field=(counts_total-counts_halo-mean_field)/mean_total; kSZ=-Wtau*momentum_field/(mean_total*c)",
            "direct_field_velocity_interpretation": "heal-vel-los is a mean velocity per occupied fine pixel; momentum is count-weighted before total-minus-halo subtraction",
            "direct_field_no_ymap": True,
            "direct_field_clip_negative_counts": bool(args.clip_negative_field_counts),
            "direct_field_population_caveat": "Physically complete only if the pasted halo profiles cover the same identified-halo population removed by the halo shell; otherwise lower-mass identified halos remain missing.",
            "direct_field_summary_json": summary,
        }
    )
    write_maps_h5(output, out_maps, galaxies, out_attrs)

    _write_json_atomic(
        diagnostics_path,
        {
            "output": str(output),
            "base_pasted_map_path": str(map_path),
            "config": str(args.config),
            "catalog_key": str(catalog_key),
            "nside": int(nside),
            "total_root": str(total_root),
            "halo_root": str(halo_root),
            "cache_root": str(cache_root),
            "z_min": float(z_min),
            "z_max": float(z_max),
            "log10_mass_min_for_weight_setup": float(log10_mass_min),
            "gg_transition_model_for_weight_setup": gg_transition_model,
            "field_keys": field_keys,
            "summary": summary,
            "shells": shell_meta,
            "diagnostics": diagnostics,
        },
    )
    print(json.dumps({"output": str(output), "diagnostics": str(diagnostics_path), "summary": summary}, indent=2, default=gmt.to_jsonable))


def read_measurement_spectra(path: Path) -> Dict[str, dict]:
    out = {}
    with h5py.File(path, "r") as h5:
        names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["joint/spectrum_names"][:]]
        for name in names:
            group = h5[f"spectra/{name}"]
            theory_key = group.attrs.get("theory_key", name)
            if isinstance(theory_key, bytes):
                theory_key = theory_key.decode("utf-8")
            out[name] = {
                "ell": group["ell"][:],
                "cl": group["cl"][:],
                "err": group["err"][:] if "err" in group else None,
                "label": str(group.attrs.get("label", name)),
                "theory_key": str(theory_key),
            }
    return out


def read_reference_errors(path: Path, names: Sequence[str]) -> Dict[str, np.ndarray]:
    out = {}
    with h5py.File(path, "r") as h5:
        if "joint/cov" in h5:
            joint_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["joint/spectrum_names"][:]]
            starts = h5["joint/slice_start"][:].astype(int)
            stops = h5["joint/slice_stop"][:].astype(int)
            diag = np.diag(h5["joint/cov"][:])
            for name, start, stop in zip(joint_names, starts, stops):
                if name in names:
                    out[name] = np.sqrt(np.clip(diag[start:stop], 0.0, np.inf))
        for name in names:
            if name not in out and f"spectra/{name}/err" in h5:
                out[name] = h5[f"spectra/{name}/err"][:]
    return out


def field_masks_for_specs(map_path: Path, specs: Sequence[mpn.SpectrumSpec], nside: int) -> Dict[str, np.ndarray]:
    needed = needed_fields_for_specs(specs)
    masks = {}
    with h5py.File(map_path, "r") as h5:
        for field_name in needed:
            group = h5[f"fields/{field_name}"]
            mask_name = str(group.attrs["mask_ref"])
            mask = np.asarray(h5[f"masks/{mask_name}"][:], dtype=np.float64)
            in_nside = hp.npix2nside(mask.size)
            if in_nside != int(nside):
                mask = hp.ud_grade(mask, nside_out=int(nside), power=0)
            masks[field_name] = np.clip(mask, 0.0, None)
    return masks


def scaled_reference_errors(
    config: Mapping[str, object],
    names: Sequence[str],
    nside: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    ref_path = Path(config["godmax"]["measurement_h5"])
    errors = read_reference_errors(ref_path, names)
    if not errors:
        return {}, {}
    specs_by_name = {spec.name: spec for spec in pz_spectrum_specs(pz_bin_from_config(config))}
    specs = [specs_by_name[name] for name in names if name in specs_by_name]
    masks = field_masks_for_specs(Path(config["godmax"]["map_h5"]), specs, int(nside))
    center_ra, center_dec, radius_deg = require_cap_center(config)
    cap = cap_pixel_mask(int(nside), center_ra, center_dec, radius_deg)
    scales = {}
    for spec in specs:
        ma = masks[spec.fields[0]]
        mb = masks[spec.fields[1]]
        full_fsky = float(np.mean(ma * mb))
        cap_fsky = float(np.mean(ma * mb * cap))
        scales[spec.name] = math.sqrt(full_fsky / cap_fsky) if cap_fsky > 0.0 and full_fsky > 0.0 else 1.0
    return {name: errors[name] * scales.get(name, 1.0) for name in names if name in errors}, scales


def read_windowed_theory(path: Path) -> Dict[str, dict]:
    with h5py.File(path, "r") as h5:
        names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["windowed/spectrum_names"][:]]
        ell = h5["windowed/ell"][:]
        full = h5["windowed/full_hod_floor10p5"][:]
        resolved = h5["windowed/resolved_log10Mgt11"][:]
        delta = h5["windowed/unresolved_delta"][:]
    n = len(ell)
    return {
        name: {
            "ell": ell,
            "full": full[i * n : (i + 1) * n],
            "resolved": resolved[i * n : (i + 1) * n],
            "delta": delta[i * n : (i + 1) * n],
        }
        for i, name in enumerate(names)
    }


def _finite_ratio_stats(ell: np.ndarray, ratio: np.ndarray) -> dict:
    ell = np.asarray(ell, dtype=np.float64)
    ratio = np.asarray(ratio, dtype=np.float64)
    good = np.isfinite(ratio)
    if not np.any(good):
        return {
            "n_finite": 0,
            "first": np.nan,
            "last": np.nan,
            "median_all": np.nan,
            "median_excluding_bin1": np.nan,
            "median_body_excluding_first_last": np.nan,
            "high_ell_tilt_last_over_body": np.nan,
            "last_over_first": np.nan,
        }
    body = ratio[1:-1]
    body_good = body[np.isfinite(body)]
    excl1 = ratio[1:]
    excl1_good = excl1[np.isfinite(excl1)]
    median_body = float(np.median(body_good)) if body_good.size else np.nan
    first = float(ratio[0]) if ratio.size and np.isfinite(ratio[0]) else np.nan
    last = float(ratio[-1]) if ratio.size and np.isfinite(ratio[-1]) else np.nan
    return {
        "n_finite": int(np.count_nonzero(good)),
        "ell_first": float(ell[0]) if ell.size else np.nan,
        "ell_last": float(ell[-1]) if ell.size else np.nan,
        "first": first,
        "last": last,
        "median_all": float(np.median(ratio[good])),
        "median_excluding_bin1": float(np.median(excl1_good)) if excl1_good.size else np.nan,
        "median_body_excluding_first_last": median_body,
        "high_ell_tilt_last_over_body": float(last / median_body) if np.isfinite(last) and np.isfinite(median_body) and median_body != 0.0 else np.nan,
        "last_over_first": float(last / first) if np.isfinite(last) and np.isfinite(first) and first != 0.0 else np.nan,
    }


def summarize_sim_theory_ratios(args: argparse.Namespace) -> None:
    sim_path = Path(args.sim).expanduser().resolve()
    theory_path = Path(args.theory).expanduser().resolve()
    component = str(args.component)
    if component not in {"full", "resolved", "delta"}:
        raise ValueError("--component must be one of full, resolved, or delta.")
    sim = read_measurement_spectra(sim_path)
    theory = read_windowed_theory(theory_path)
    rows = {}
    skipped = {}
    for name in sorted(sim):
        theory_candidates = [name]
        sim_theory_key = str(sim[name].get("theory_key", name))
        if sim_theory_key and sim_theory_key not in theory_candidates:
            theory_candidates.append(sim_theory_key)
        theory_name = next((candidate for candidate in theory_candidates if candidate in theory), None)
        if theory_name is None:
            skipped[name] = {
                "measurement_theory_key": sim_theory_key,
                "candidate_theory_spectra": theory_candidates,
            }
            continue
        sim_cl = np.asarray(sim[name]["cl"], dtype=np.float64)
        th_key = {"full": "full", "resolved": "resolved", "delta": "delta"}[component]
        th_cl = np.asarray(theory[theory_name][th_key], dtype=np.float64)
        n = min(len(sim_cl), len(th_cl))
        ratio = np.divide(sim_cl[:n], th_cl[:n], out=np.full(n, np.nan, dtype=np.float64), where=th_cl[:n] != 0.0)
        rows[name] = {
            "theory_spectrum_used": theory_name,
            "measurement_theory_key": sim_theory_key,
            "ell": np.asarray(sim[name]["ell"][:n], dtype=np.float64).tolist(),
            "ratio": ratio.tolist(),
            "stats": _finite_ratio_stats(sim[name]["ell"][:n], ratio),
        }
    payload = {
        "schema": "stage31_sim_theory_ratio_summary_v1",
        "sim": str(sim_path),
        "theory": str(theory_path),
        "component": component,
        "bin1_policy": "Report bin-1-excluded medians; do not use bin 1 for closure claims until cap mean-subtraction is modeled.",
        "spectra": rows,
        "spectra_skipped_missing_theory": skipped,
    }
    output = Path(args.output).expanduser().resolve() if args.output else sim_path.with_name(
        f"{sim_path.stem}_over_{theory_path.stem}_{component}_ratio_summary.json"
    )
    _write_json_atomic(output, payload)
    print(json.dumps({"output": str(output), "n_spectra": len(rows), "n_skipped_missing_theory": len(skipped)}, indent=2))


def load_vector_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as npz:
        return {key: npz[key] for key in npz.files}


def decode_names(values: Sequence[object]) -> List[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def stage31_plot_transform(name: str, ell: np.ndarray, cl: np.ndarray, err: Optional[np.ndarray] = None):
    ell = np.asarray(ell, dtype=np.float64)
    cl = np.asarray(cl, dtype=np.float64)
    if name.startswith("desi_g_auto"):
        y = cl
        yerr = None if err is None else np.asarray(err, dtype=np.float64)
        ylabel = r"$C_\ell$ signal"
        return y, yerr, ylabel

    fac = ell * (ell + 1.0) / (2.0 * math.pi)
    sign = -1.0 if name.startswith("desi_pi_act_T") else 1.0
    scale = 1.0e3 if name.startswith("desi_pi_act_T") else 1.0
    y = sign * scale * fac * cl
    yerr = None if err is None else scale * fac * np.asarray(err, dtype=np.float64)
    ylabel = r"$-10^3 D_\ell^{\pi T}$" if name.startswith("desi_pi_act_T") else r"$D_\ell$"
    return y, yerr, ylabel


def full_data_panel_title(name: str, pz_bin: int) -> str:
    pz = rf"\mathrm{{pz}}{int(pz_bin)}"
    if name.startswith("desi_g_auto"):
        return rf"DESI galaxy clustering, ${pz}\times {pz}$"
    if name.startswith("desi_g_act_y"):
        return rf"DESI galaxies $\times$ ACT thermal SZ, ${pz}\times y$"
    if name.startswith("desi_g_act_kappa"):
        return rf"DESI galaxies $\times$ ACT CMB lensing, ${pz}\times\kappa_{{\rm CMB}}$"
    if name.startswith("desi_pi_act_T"):
        return rf"DESI velocity tracer $\times$ ACT temperature, $\pi_{{{int(pz_bin)}}}\times T$"
    if name.startswith("desi_g_des_shear_E"):
        tomo = name.rsplit("tomo", 1)[-1]
        return rf"DESI galaxies $\times$ DES shear, ${pz}\times\gamma_E^{{({tomo})}}$"
    return name.replace("_", r"\_")


def _plot_ell_max_from_args(args: argparse.Namespace) -> Optional[float]:
    raw = getattr(args, "plot_ell_max", None)
    if raw is None:
        return None
    ell_max = float(raw)
    if ell_max <= 0.0:
        return None
    if not np.isfinite(ell_max):
        raise ValueError(f"--plot-ell-max must be finite, got {raw!r}.")
    return ell_max


def _clip_plot_ell(
    ell: np.ndarray,
    *arrays: Optional[np.ndarray],
    ell_max: Optional[float],
) -> Tuple[np.ndarray, ...]:
    ell = np.asarray(ell, dtype=np.float64)
    if ell_max is None:
        return (ell, *arrays)
    keep = ell <= float(ell_max)
    clipped = [ell[keep]]
    for arr in arrays:
        clipped.append(None if arr is None else np.asarray(arr)[keep])
    return tuple(clipped)


def plot_full_data(args: argparse.Namespace) -> None:
    import shutil
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if bool(args.raw_cl) and bool(args.dell):
        raise ValueError("Choose only one of --raw-cl or --dell.")

    config = read_config(args.config)
    measurement_path = Path(args.measurement) if args.measurement else Path(config["godmax"]["measurement_h5"])
    fid_path = Path(args.fiducial_vector)
    best_path = Path(args.bestfit_vector)
    measurement = gmt.load_measurement_data(measurement_path)
    fid_npz = load_vector_npz(fid_path)
    best_npz = load_vector_npz(best_path)
    fid_names = decode_names(fid_npz["spectrum_names"])
    best_names = decode_names(best_npz["spectrum_names"])
    if fid_names != measurement.names:
        raise ValueError("Fiducial vector spectrum names do not match full measurement order.")
    if best_names != measurement.names:
        raise ValueError("Bestfit vector spectrum names do not match full measurement order.")
    if not np.allclose(fid_npz["data_vector"], measurement.data_vector):
        raise ValueError("Fiducial vector data does not match the full measurement vector.")
    if not np.allclose(best_npz["data_vector"], measurement.data_vector):
        raise ValueError("Bestfit vector data does not match the full measurement vector.")

    fiducial = np.asarray(fid_npz["theory_vector"], dtype=np.float64)
    bestfit = np.asarray(best_npz["theory_vector"], dtype=np.float64)
    sim = read_measurement_spectra(Path(args.sim)) if args.sim else {}
    pz_bin = pz_bin_from_config(config)
    nside = int(config.get("pasting", {}).get("nside", 1024))
    cap_area_latex = cap_area_latex_from_config(config)
    names = [name for name in core_spectra_for_pz(pz_bin) if name in measurement.names]
    plot_ell_max = _plot_ell_max_from_args(args)

    def transform(name: str, ell: np.ndarray, cl: np.ndarray, err: Optional[np.ndarray] = None):
        if bool(args.raw_cl):
            y = np.asarray(cl, dtype=np.float64)
            yerr = None if err is None else np.asarray(err, dtype=np.float64)
            return y, yerr, r"$C_\ell$"
        if bool(args.dell):
            ell = np.asarray(ell, dtype=np.float64)
            factor = ell * (ell + 1.0) / (2.0 * math.pi)
            sign = -1.0 if name.startswith("desi_pi_act_T") else 1.0
            y = sign * factor * np.asarray(cl, dtype=np.float64)
            yerr = None if err is None else factor * np.asarray(err, dtype=np.float64)
            if name.startswith("desi_pi_act_T"):
                return y, yerr, r"$D_\ell^{\rm kSZ}=-\ell(\ell+1)C_\ell^{\pi T}/2\pi$"
            return y, yerr, r"$D_\ell = \ell(\ell+1)C_\ell/2\pi$"
        return stage31_plot_transform(name, ell, cl, err)

    latex_available = shutil.which("latex") is not None
    rc_params = {
        "text.usetex": latex_available,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,
        "axes.linewidth": 0.85,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with mpl.rc_context(rc_params):
        ncols = 2
        nrows = int(math.ceil(len(names) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(13.2, 3.55 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        fig.patch.set_facecolor("white")
        ell = np.asarray(measurement.ell, dtype=np.float64)
        colors = {
            "data": "#20242b",
            "bestfit": "#0072B2",
            "fiducial": "#D55E00",
            "sim": "#009E73",
            "zero": "#6f7782",
            "grid": "#d7dce2",
        }
        labels = {
            "data": "Data measurements (full survey footprint)",
            "bestfit": "Best-fit theory",
            "fiducial": "Fiducial theory",
            "sim": rf"Abacus Backlight paste ({cap_area_latex} cap, $N_{{\rm side}}={nside}$)",
        }
        for ax, name in zip(axes.ravel(), names):
            idx = measurement.names.index(name)
            start = int(measurement.starts[idx])
            stop = int(measurement.stops[idx])
            data_cl = measurement.data_vector[start:stop]
            err = np.sqrt(np.clip(np.diag(measurement.covariance[start:stop, start:stop]), 0.0, np.inf))
            ell_panel, data_cl, err, fid_cl, best_cl = _clip_plot_ell(
                ell,
                data_cl,
                err,
                fiducial[start:stop],
                bestfit[start:stop],
                ell_max=plot_ell_max,
            )
            y_data, y_err, ylabel = transform(name, ell_panel, data_cl, err)
            y_fid, _, _ = transform(name, ell_panel, fid_cl)
            y_best, _, _ = transform(name, ell_panel, best_cl)

            ax.errorbar(
                ell_panel,
                y_data,
                yerr=y_err,
                fmt="o",
                ms=3.7,
                lw=1.0,
                elinewidth=0.9,
                capsize=2.4,
                color=colors["data"],
                markerfacecolor="white",
                markeredgewidth=1.0,
                alpha=0.95,
                label=labels["data"],
                zorder=4,
            )
            if bool(args.include_fiducial):
                ax.plot(ell_panel, y_fid, "-", lw=1.45, color=colors["fiducial"], label=labels["fiducial"], zorder=2)
            ax.plot(ell_panel, y_best, "-", lw=2.0, color=colors["bestfit"], label=labels["bestfit"], zorder=3)
            if name in sim:
                sim_ell = np.asarray(sim[name]["ell"], dtype=np.float64)
                sim_ell, sim_cl = _clip_plot_ell(sim_ell, sim[name]["cl"], ell_max=plot_ell_max)
                y_sim, _, _ = transform(name, sim_ell, sim_cl)
                ax.plot(
                    sim_ell,
                    y_sim,
                    "s",
                    ms=4.0,
                    color=colors["sim"],
                    markeredgecolor="#004d3b",
                    markeredgewidth=0.6,
                    label=labels["sim"],
                    zorder=5,
                )
            ax.axhline(0.0, color=colors["zero"], lw=0.85, alpha=0.75, zorder=1)
            if name.startswith("desi_pi_act_T"):
                ksz_ylim = getattr(args, "ksz_ylim", None) or (-5.0e-5, 5.0e-5)
                ax.set_ylim(float(ksz_ylim[0]), float(ksz_ylim[1]))
            elif name.startswith("desi_g_auto") and np.all(y_data > 0.0) and np.all(y_best > 0.0):
                ax.set_yscale("log")
            if plot_ell_max is not None:
                ax.set_xlim(right=float(plot_ell_max))
            ax.grid(True, color=colors["grid"], lw=0.75, alpha=0.72)
            ax.tick_params(direction="out", length=3.2, width=0.8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.set_xlabel(r"Multipole, $\ell$")
            ax.set_ylabel(ylabel)
            ax.set_title(full_data_panel_title(name, pz_bin), pad=7.0)
            ax.legend(
                loc="best",
                frameon=True,
                facecolor="white",
                edgecolor="#c5ccd3",
                framealpha=0.92,
                borderpad=0.55,
                handlelength=2.3,
            )
        for ax in axes.ravel()[len(names) :]:
            ax.set_visible(False)
        if bool(args.raw_cl):
            quantity = r"$C_\ell$"
        elif bool(args.dell):
            quantity = r"$D_\ell=\ell(\ell+1)C_\ell/(2\pi)$; kSZ panel uses $-D_\ell^{\pi T}$"
        else:
            quantity = "Stage-31 plotting convention"
        fig.suptitle(
            rf"DESI $\mathrm{{pz}}{pz_bin}$ validation: data, best-fit theory, and {cap_area_latex} Abacus paste ({quantity})",
            fontsize=14,
        )
        output = Path(args.output) if args.output else output_dir(config, "plot_subdir") / (
            f"{run_name_from_config(config)}_full_area_data_bestfit_for_cap_paste_validation.pdf"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(output),
                "measurement": str(measurement_path),
                "fiducial_vector": str(fid_path),
                "bestfit_vector": str(best_path),
                "sim": str(args.sim) if args.sim else None,
                "spectra": names,
                "data_layer": "full-footprint Stage-31 measurement/covariance",
                "raw_cl": bool(args.raw_cl),
                "dell": bool(args.dell),
                "plot_ell_max": plot_ell_max,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _likelihood_cut_bound(
    cut_config: Mapping[str, object],
    name: str,
    family: str,
    theory_key: str,
    which: str,
) -> Optional[float]:
    """Resolve the likelihood ell_min/ell_max for a spectrum.

    Mirrors _ell_min_for_spectrum / _ell_max_for_spectrum in
    godmax_multiprobe_hmc_stage31.py: precedence is spectrum_ell_<which>[name or
    theory_key] -> family_ell_<which>[family] -> default_ell_<which>.  ``which``
    is "min" or "max".  Replicated here to avoid importing the numpyro-heavy HMC
    module into the plotting job.
    """

    if not cut_config:
        return None
    spectrum_map = cut_config.get(f"spectrum_ell_{which}") or {}
    for key in (name, theory_key):
        if key in spectrum_map:
            return float(spectrum_map[key])
    family_map = cut_config.get(f"family_ell_{which}") or {}
    if family in family_map:
        return float(family_map[family])
    default = cut_config.get(f"default_ell_{which}")
    return None if default is None else float(default)


def _likelihood_active_band_mask(
    ell_centers: np.ndarray,
    ell_left: Optional[np.ndarray],
    ell_right: Optional[np.ndarray],
    cut_config: Mapping[str, object],
    name: str,
    family: str,
    theory_key: str,
) -> np.ndarray:
    """Boolean mask of bandpowers kept by the likelihood scale cuts.

    Mirrors _selected_band_indices in godmax_multiprobe_hmc_stage31.py with the
    configured band_selection basis (default "center").
    """

    centers = np.asarray(ell_centers, dtype=np.float64)
    n_band = int(centers.size)
    ell_min = _likelihood_cut_bound(cut_config, name, family, theory_key, "min")
    ell_max = _likelihood_cut_bound(cut_config, name, family, theory_key, "max")
    if ell_min is None and ell_max is None:
        return np.ones(n_band, dtype=bool)
    selection = str(cut_config.get("band_selection", "center")).lower()
    if selection in {"left", "lower", "ell_left"} and ell_left is not None:
        basis = np.asarray(ell_left, dtype=np.float64)
    elif selection in {"right", "upper", "ell_right"} and ell_right is not None:
        basis = np.asarray(ell_right, dtype=np.float64)
    else:
        basis = centers
    keep = np.ones(n_band, dtype=bool)
    if ell_min is not None:
        keep &= basis >= float(ell_min)
    if ell_max is not None:
        keep &= basis <= float(ell_max)
    return keep


def _inactive_band_spans(
    active_mask: np.ndarray,
    ell_left: Optional[np.ndarray],
    ell_right: Optional[np.ndarray],
) -> List[Tuple[float, float]]:
    """(lo, hi) ell ranges of bandpowers excluded by the likelihood cuts."""

    if ell_left is None or ell_right is None:
        return []
    excluded = ~np.asarray(active_mask, dtype=bool)
    left = np.asarray(ell_left, dtype=np.float64)
    right = np.asarray(ell_right, dtype=np.float64)
    return [(float(lo), float(hi)) for lo, hi in zip(left[excluded], right[excluded])]


def plot_full_data_theory_variants(args: argparse.Namespace) -> None:
    import shutil
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    config = read_config(args.config)
    measurement_path = Path(args.measurement) if args.measurement else Path(config["godmax"]["measurement_h5"])
    sum_theory_path = Path(args.sum_theory)
    response_theory_path = Path(args.response_theory) if args.response_theory else None
    sim_path = Path(args.sim) if args.sim else None
    extra_sim_paths = [Path(path) for path in (args.extra_sim or [])]
    extra_sim_labels = list(args.extra_sim_label or [])
    if len(extra_sim_labels) > len(extra_sim_paths):
        raise ValueError("--extra-sim-label was supplied more times than --extra-sim.")

    measurement = gmt.load_measurement_data(measurement_path)
    sum_theory = read_windowed_theory(sum_theory_path)
    response_theory = read_windowed_theory(response_theory_path) if response_theory_path is not None else {}
    sim = read_measurement_spectra(sim_path) if sim_path is not None and sim_path.exists() else {}
    pz_bin = pz_bin_from_config(config)
    nside = int(args.nside or config.get("pasting", {}).get("nside", 1024))
    cap_area_latex = cap_area_latex_from_config(config)
    theory_component = str(args.theory_component)
    if theory_component not in {"full", "resolved"}:
        raise ValueError("--theory-component must be 'full' or 'resolved'.")
    plot_ell_max = _plot_ell_max_from_args(args)
    cut_config = config.get("likelihood_cuts") or {}
    gray_unused = bool(cut_config) if args.gray_unused_bands is None else bool(args.gray_unused_bands)
    extra_sims = []
    for idx, path in enumerate(extra_sim_paths):
        if not path.exists():
            raise FileNotFoundError(path)
        extra_sims.append(
            {
                "path": path,
                "spectra": read_measurement_spectra(path),
                "label": extra_sim_labels[idx] if idx < len(extra_sim_labels) else f"Additional simulation {idx + 1}",
            }
        )

    names = [
        name
        for name in core_spectra_for_pz(pz_bin)
        if name in measurement.names and name in sum_theory
        and (response_theory_path is None or name in response_theory)
    ]
    if not names:
        raise RuntimeError("No overlapping spectra found between data and theory products.")

    def transform(name: str, ell: np.ndarray, cl: np.ndarray, err: Optional[np.ndarray] = None):
        ell = np.asarray(ell, dtype=np.float64)
        cl = np.asarray(cl, dtype=np.float64)
        if bool(args.raw_cl):
            return cl, None if err is None else np.asarray(err, dtype=np.float64), r"$C_\ell$"
        factor = ell * (ell + 1.0) / (2.0 * math.pi)
        sign = -1.0 if name.startswith("desi_pi_act_T") else 1.0
        y = sign * factor * cl
        yerr = None if err is None else factor * np.asarray(err, dtype=np.float64)
        ylabel = r"$-D_\ell^{\pi T}$" if name.startswith("desi_pi_act_T") else r"$D_\ell$"
        return y, yerr, ylabel

    latex_available = shutil.which("latex") is not None
    rc_params = {
        "text.usetex": latex_available,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.6,
        "legend.title_fontsize": 8.6,
        "axes.linewidth": 0.85,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with mpl.rc_context(rc_params):
        ncols = 2
        nrows = int(math.ceil(len(names) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(13.2, 3.55 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        fig.patch.set_facecolor("white")
        ell = np.asarray(measurement.ell, dtype=np.float64)
        colors = {
            "data": "#20242b",
            "sum": "#0072B2",
            "response": "#D55E00",
            "sim": "#009E73",
            "extra_sim_0": "#CC79A7",
            "extra_sim_1": "#E69F00",
            "extra_sim_2": "#56B4E9",
            "zero": "#6f7782",
            "grid": "#d7dce2",
        }
        sim_series = []
        if sim:
            sim_series.append(
                {
                    "spectra": sim,
                    "label": args.sim_label
                    or rf"Abacus Backlight paste ({cap_area_latex}, $N_{{\rm side}}={nside}$)",
                    "marker": "s",
                    "color": colors["sim"],
                    "edgecolor": "#004d3b",
                    "zorder": 5,
                }
            )
        extra_markers = ["^", "D", "v", "P", "X"]
        for idx, item in enumerate(extra_sims):
            sim_series.append(
                {
                    "spectra": item["spectra"],
                    "label": item["label"],
                    "marker": extra_markers[idx % len(extra_markers)],
                    "color": colors.get(f"extra_sim_{idx}", "#7f7f7f"),
                    "edgecolor": "#4a2740" if idx == 0 else "#4a4a4a",
                    "zorder": 6 + idx,
                }
            )
        for ax, name in zip(axes.ravel(), names):
            idx = measurement.names.index(name)
            start = int(measurement.starts[idx])
            stop = int(measurement.stops[idx])
            data_cl = measurement.data_vector[start:stop]
            err = np.sqrt(np.clip(np.diag(measurement.covariance[start:stop, start:stop]), 0.0, np.inf))
            ell_panel, data_cl, err = _clip_plot_ell(ell, data_cl, err, ell_max=plot_ell_max)
            y_data, y_err, ylabel = transform(name, ell_panel, data_cl, err)
            if gray_unused:
                active_mask = _likelihood_active_band_mask(
                    ell,
                    measurement.ell_left,
                    measurement.ell_right,
                    cut_config,
                    name,
                    measurement.families.get(name, ""),
                    measurement.theory_keys.get(name, name),
                )
                first_inactive = True
                for lo, hi in _inactive_band_spans(active_mask, measurement.ell_left, measurement.ell_right):
                    ax.fill_between(
                        [lo, hi],
                        [0.0, 0.0],
                        [1.0, 1.0],
                        transform=ax.get_xaxis_transform(),
                        color="#b8bcc5",
                        alpha=0.28,
                        lw=0,
                        zorder=0,
                        label="not in likelihood" if first_inactive else None,
                    )
                    first_inactive = False
            ax.errorbar(
                ell_panel,
                y_data,
                yerr=y_err,
                fmt="o",
                ms=3.7,
                lw=1.0,
                elinewidth=0.9,
                capsize=2.4,
                color=colors["data"],
                markerfacecolor="white",
                markeredgewidth=1.0,
                alpha=0.95,
                label="Data measurements (full survey footprint)",
                zorder=4,
            )

            th_sum = sum_theory[name]
            ell_sum, cl_sum = _clip_plot_ell(
                th_sum["ell"],
                th_sum[theory_component],
                ell_max=plot_ell_max,
            )
            y_sum, _, _ = transform(name, ell_sum, cl_sum)
            sum_label = "Theory (1h+2h)" if response_theory_path is None else "Theory power-add transitions"
            ax.plot(
                ell_sum,
                y_sum,
                "-",
                lw=2.0,
                color=colors["sum"],
                label=sum_label,
                zorder=3,
            )
            if response_theory_path is not None and name in response_theory:
                th_response = response_theory[name]
                ell_response, cl_response = _clip_plot_ell(
                    th_response["ell"],
                    th_response[theory_component],
                    ell_max=plot_ell_max,
                )
                y_response, _, _ = transform(name, ell_response, cl_response)
                ax.plot(
                    ell_response,
                    y_response,
                    "--",
                    lw=1.8,
                    color=colors["response"],
                    label="Theory response transitions",
                    zorder=3,
                )
            for series in sim_series:
                spectra = series["spectra"]
                if name not in spectra:
                    continue
                s = spectra[name]
                sim_ell, sim_cl = _clip_plot_ell(s["ell"], s["cl"], ell_max=plot_ell_max)
                y_sim, _, _ = transform(name, sim_ell, sim_cl)
                ax.plot(
                    sim_ell,
                    y_sim,
                    linestyle="None",
                    marker=series["marker"],
                    ms=4.2,
                    color=series["color"],
                    markeredgecolor=series["edgecolor"],
                    markeredgewidth=0.6,
                    label=series["label"],
                    zorder=series["zorder"],
                )
            ax.axhline(0.0, color=colors["zero"], lw=0.85, alpha=0.75, zorder=1)
            if name.startswith("desi_pi_act_T"):
                ksz_ylim = getattr(args, "ksz_ylim", None) or (-5.0e-5, 5.0e-5)
                ax.set_ylim(float(ksz_ylim[0]), float(ksz_ylim[1]))
            elif name.startswith("desi_g_auto") and np.all(y_data > 0.0) and np.all(y_sum > 0.0):
                ax.set_yscale("log")
            if str(getattr(args, "plot_xscale", "linear")) == "log":
                ax.set_xscale("log")
            if plot_ell_max is not None:
                ax.set_xlim(right=float(plot_ell_max))
            ax.grid(True, color=colors["grid"], lw=0.75, alpha=0.72)
            ax.tick_params(direction="out", length=3.2, width=0.8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.set_xlabel(r"Multipole, $\ell$")
            ax.set_ylabel(ylabel)
            ax.set_title(full_data_panel_title(name, pz_bin), pad=7.0)
            ax.legend(
                loc="best",
                frameon=True,
                facecolor="white",
                edgecolor="#c5ccd3",
                framealpha=0.92,
                borderpad=0.55,
                handlelength=2.3,
            )
        for ax in axes.ravel()[len(names) :]:
            ax.set_visible(False)
        quantity = r"$C_\ell$" if bool(args.raw_cl) else r"$D_\ell=\ell(\ell+1)C_\ell/(2\pi)$; kSZ panel uses $-D_\ell^{\pi T}$"
        fig.suptitle(
            rf"DESI $\mathrm{{pz}}{pz_bin}$ validation: theory variants, full-footprint data, and {cap_area_latex} Abacus paste ({quantity})",
            fontsize=14,
        )
        output = Path(args.output) if args.output else output_dir(config, "plot_subdir") / (
            f"{run_name_from_config(config)}_full_data_theory_variants_with_cap_sim_Dell.pdf"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)

    print(
        json.dumps(
            {
                "output": str(output),
                "measurement": str(measurement_path),
                "sum_theory": str(sum_theory_path),
                "response_theory": str(response_theory_path) if response_theory_path is not None else None,
                "sim": str(sim_path) if sim_path else None,
                "sim_label": args.sim_label,
                "extra_sim": [str(path) for path in extra_sim_paths],
                "extra_sim_label": extra_sim_labels,
                "spectra": names,
                "raw_cl": bool(args.raw_cl),
                "theory_component": theory_component,
                "plot_ell_max": plot_ell_max,
            },
            indent=2,
            sort_keys=True,
        )
    )


def map_percentile_limits(
    values: np.ndarray,
    symmetric: bool = False,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite != hp.UNSEEN) & (finite > hp.UNSEEN / 2.0)]
    if finite.size == 0:
        return 0.0, 1.0
    if symmetric:
        vmax = float(np.nanpercentile(np.abs(finite), upper_percentile))
        vmax = vmax if vmax > 0.0 else float(np.nanmax(np.abs(finite)))
        return -vmax, vmax
    lo, hi = np.nanpercentile(finite, [lower_percentile, upper_percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def masked_for_cap(values: np.ndarray, cap_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out[cap_mask <= 0.0] = hp.UNSEEN
    return out


def counts_to_delta(counts: np.ndarray, cap_mask: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    inside = cap_mask > 0.0
    mean = float(np.mean(counts[inside])) if np.any(inside) else 0.0
    delta = np.zeros_like(counts, dtype=np.float64)
    if mean > 0.0:
        delta[inside] = counts[inside] / mean - 1.0
    delta[~inside] = hp.UNSEEN
    return delta


def halo_count_map(config: Mapping[str, object], nside: int, cap_mask: np.ndarray, catalog_key: str) -> np.ndarray:
    path = catalog_path(config, catalog_key)
    counts = np.zeros(hp.nside2npix(int(nside)), dtype=np.float64)
    center_ra, center_dec, radius_deg = require_cap_center(config)
    chunk = 1_000_000
    with h5py.File(path, "r") as h5:
        n = len(h5["ra_deg"])
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            ra = h5["ra_deg"][start:stop]
            dec = h5["dec_deg"][start:stop]
            keep = angular_cap_mask(ra, dec, center_ra, center_dec, radius_deg)
            if np.any(keep):
                pix = hp.ang2pix(int(nside), ra[keep], dec[keep], lonlat=True)
                np.add.at(counts, pix, 1.0)
    return counts_to_delta(counts, cap_mask)


def galaxy_count_delta(galaxies: np.ndarray, nside: int, cap_mask: np.ndarray) -> np.ndarray:
    counts = np.zeros(hp.nside2npix(int(nside)), dtype=np.float64)
    if galaxies.size:
        valid = np.asarray(galaxies[:, 5]) > 0.5 if galaxies.shape[1] > 5 else np.ones(len(galaxies), dtype=bool)
        if np.any(valid):
            gals = galaxies[valid]
            weights = np.asarray(gals[:, 7], dtype=np.float64) if gals.shape[1] > 7 else np.ones(len(gals), dtype=np.float64)
            pix = hp.ang2pix(int(nside), gals[:, 0], gals[:, 1], lonlat=True)
            in_cap = cap_mask[pix] > 0.0
            if np.any(in_cap):
                np.add.at(counts, pix[in_cap], weights[in_cap])
    return counts_to_delta(counts, cap_mask)


def map_title_and_limits(name: str, values: np.ndarray, pz_bin: int = 1) -> Tuple[str, bool]:
    titles = {
        "halo_delta": r"Halo overdensity, $\delta_h$",
        "galaxy_delta": rf"Pasted DESI $\mathrm{{pz}}{int(pz_bin)}$ galaxies, $\delta_g$",
        "map_ymap": r"Thermal SZ Compton-$y$",
        "map_ksz": r"Kinetic SZ temperature, $\Delta T_{\rm kSZ}$",
        "map_tau": r"Electron optical depth, $\tau$",
        "map_kappa_cmb": r"CMB lensing convergence, $\kappa_{\rm CMB}$",
        "map_kappa_wl": r"DES source bin 1 convergence, $\kappa_s^{(1)}$",
        "map_kappa_wl_tomo2": r"DES source bin 2 convergence, $\kappa_s^{(2)}$",
        "map_kappa_wl_tomo3": r"DES source bin 3 convergence, $\kappa_s^{(3)}$",
        "map_kappa_wl_tomo4": r"DES source bin 4 convergence, $\kappa_s^{(4)}$",
        "map_rhom": r"Projected matter surface density",
        "map_rhom_dmb": r"Baryonified projected matter",
        "map_rhom_dmo": r"Dark-matter-only projected matter",
    }
    symmetric = name in {"halo_delta", "galaxy_delta", "map_ksz", "map_kappa_cmb"} or name.startswith("map_kappa_wl")
    return titles.get(name, name), symmetric


def display_map_for_panel(name: str, values: np.ndarray) -> Tuple[np.ndarray, str, bool, str]:
    arr = np.asarray(values, dtype=np.float64).copy()
    unseen = arr == hp.UNSEEN
    sparse_profile_map = name in {"map_ksz", "map_ymap", "map_tau"}
    if sparse_profile_map:
        unseen |= np.isfinite(arr) & (arr == 0.0)
    if name == "map_ksz":
        arr = arr * mpn.TCMB_UK
        arr[unseen] = hp.UNSEEN
        return arr, r"$\mu{\rm K}$", True, "RdBu_r"
    if name == "map_ymap":
        arr = arr * 1.0e6
        arr[unseen] = hp.UNSEEN
        return arr, r"$10^6\,y$", False, "magma"
    if name == "map_tau":
        arr = arr * 1.0e3
        arr[unseen] = hp.UNSEEN
        return arr, r"$10^3\,\tau$", False, "viridis"
    if name in {"halo_delta", "galaxy_delta"}:
        return arr, r"overdensity", True, "RdBu_r"
    if name == "map_kappa_cmb" or name.startswith("map_kappa_wl"):
        return arr, r"$\kappa$", True, "PuOr_r"
    if name.startswith("map_rhom"):
        return arr, r"map value", False, "cividis"
    return arr, "", False, "viridis"


def plot_healpix_maps(args: argparse.Namespace) -> None:
    import shutil
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    config = read_config(args.config)
    nside = int(args.nside)
    pz_bin = pz_bin_from_config(config)
    catalog_key = args.catalog or default_catalog_key(config)
    map_path = Path(args.maps) if args.maps else final_map_path(config, catalog_key, nside)
    if not map_path.exists():
        raise FileNotFoundError(
            f"Missing pasted map HDF5: {map_path}. Run paste-split/combine-maps first, or pass --maps."
        )
    maps, galaxies, attrs = load_maps_h5(map_path)
    center_ra, center_dec, radius_deg = require_cap_center(config)
    cap_mask = cap_pixel_mask(nside, center_ra, center_dec, radius_deg)

    panels: List[Tuple[str, np.ndarray]] = []
    panels.append(("halo_delta", halo_count_map(config, nside, cap_mask, catalog_key)))
    if galaxies.size:
        panels.append(("galaxy_delta", galaxy_count_delta(galaxies, nside, cap_mask)))
    projected_matter_maps = {"map_rhom", "map_rhom_dmb", "map_rhom_dmo"}
    for name in MAP_DATASETS:
        if name in projected_matter_maps:
            continue
        if name in maps:
            panels.append((name, masked_for_cap(maps[name], cap_mask)))

    ncols = int(args.ncols)
    nrows = int(math.ceil(len(panels) / ncols))
    xsize = int(args.xsize)
    reso_arcmin = float(args.reso_arcmin) if args.reso_arcmin is not None else 2.1 * radius_deg * 60.0 / xsize
    output = Path(args.output) if args.output else output_dir(config, "plot_subdir") / (
        f"{run_name_from_config(config)}_pasted_healpix_maps_nside{nside}.pdf"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    latex_available = shutil.which("latex") is not None
    rc_params = {
        "text.usetex": latex_available,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 10.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with mpl.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.45 * ncols, 4.25 * nrows),
            squeeze=False,
            constrained_layout=True,
        )
        fig.patch.set_facecolor("white")
        half_width_arcmin = 0.5 * xsize * reso_arcmin
        extent = [-half_width_arcmin / 60.0, half_width_arcmin / 60.0, -half_width_arcmin / 60.0, half_width_arcmin / 60.0]
        for ax, (name, values) in zip(axes.ravel(), panels):
            title, default_symmetric = map_title_and_limits(name, values, pz_bin=pz_bin)
            display_values, unit, symmetric, cmap_name = display_map_for_panel(name, values)
            symmetric = bool(symmetric or default_symmetric)
            if name in {"map_ksz", "map_ymap", "map_tau"}:
                lower_percentile, upper_percentile = 2.0, 98.0
            else:
                lower_percentile, upper_percentile = 1.0, 99.0
            projected = hp.gnomview(
                display_values,
                rot=(center_ra, center_dec, 0.0),
                xsize=xsize,
                ysize=xsize,
                reso=reso_arcmin,
                notext=True,
                cbar=False,
                no_plot=True,
                return_projected_map=True,
            )
            img = np.ma.masked_invalid(np.asarray(projected, dtype=np.float64))
            img = np.ma.masked_where(np.asarray(img) <= hp.UNSEEN / 2.0, img)
            vmin, vmax = map_percentile_limits(
                np.asarray(img.compressed(), dtype=np.float64),
                symmetric=symmetric,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            )
            cmap = mpl.colormaps.get_cmap(cmap_name).copy()
            cmap.set_bad("#f1f3f5")
            im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(title, pad=6.0)
            ax.set_xlabel(r"$\Delta{\rm RA}$ [deg]")
            ax.set_ylabel(r"$\Delta{\rm Dec}$ [deg]")
            ax.set_aspect("equal")
            ax.tick_params(direction="out", length=3.0, width=0.75)
            for spine in ax.spines.values():
                spine.set_color("#a7adb5")
                spine.set_linewidth(0.75)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
            cbar.ax.tick_params(labelsize=7.4, length=2.4, width=0.6)
            if unit:
                cbar.set_label(unit, rotation=90, labelpad=7.0)
        for ax in axes.ravel()[len(panels) :]:
            ax.set_visible(False)
        fig.suptitle(
            rf"Abacus Backlight $\mathrm{{pz}}{pz_bin}$ pasted fields on the {cap_area_latex_from_config(config)} validation cap "
            rf"($N_{{\rm side}}={nside}$, center $=({center_ra:.2f}^\circ,{center_dec:.2f}^\circ)$)",
            fontsize=14,
        )
        with PdfPages(output) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(output),
                "maps": str(map_path),
                "catalog": str(catalog_path(config, catalog_key)),
                "nside": nside,
                "n_panels": len(panels),
                "panel_names": [name for name, _ in panels],
                "n_galaxies": int(len(galaxies)),
                "map_attrs": {key: (val.decode("utf-8") if isinstance(val, bytes) else str(val)) for key, val in attrs.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


def plot_overlay(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    config = read_config(args.config)
    data_path = Path(args.data) if args.data else default_measurement_path(config, "data", args.nside)
    theory_path = Path(args.theory)
    sim_path = Path(args.sim) if args.sim else None
    data = read_measurement_spectra(data_path)
    theory = read_windowed_theory(theory_path)
    sim = read_measurement_spectra(sim_path) if sim_path is not None and sim_path.exists() else {}

    names = [name for name in core_spectra_for_pz(pz_bin_from_config(config)) if name in data and name in theory]
    ref_errors, error_scales = ({}, {})
    if bool(args.scaled_reference_errors):
        ref_errors, error_scales = scaled_reference_errors(config, names, int(args.nside))

    def convert(ell: np.ndarray, values: np.ndarray) -> np.ndarray:
        if not args.dell:
            return np.asarray(values, dtype=np.float64)
        factor = np.asarray(ell, dtype=np.float64) * (np.asarray(ell, dtype=np.float64) + 1.0) / (2.0 * math.pi)
        return factor * np.asarray(values, dtype=np.float64)

    ncols = 2
    nrows = int(math.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.0 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        d = data[name]
        err = d["err"] if d["err"] is not None else ref_errors.get(name)
        y_data = convert(d["ell"], d["cl"])
        y_err = convert(d["ell"], err) if err is not None else None
        if err is None:
            ax.plot(d["ell"], y_data, "o", ms=3.0, label="data cap")
        else:
            ax.errorbar(d["ell"], y_data, yerr=y_err, fmt="o", ms=3.0, lw=0.8, capsize=2.0, label="data cap")
        th = theory[name]
        ax.plot(th["ell"], convert(th["ell"], th["full"]), "-", lw=1.6, label="Stage-31 theory")
        ax.plot(th["ell"], convert(th["ell"], th["resolved"]), "--", lw=1.2, label="resolved theory")
        if name in sim:
            s = sim[name]
            ax.plot(s["ell"], convert(s["ell"], s["cl"]), "s", ms=3.0, label="sim resolved")
            corrected = np.asarray(s["cl"]) + np.asarray(th["delta"])
            ax.plot(s["ell"], convert(s["ell"], corrected), "d", ms=3.0, label="sim + unresolved")
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$D_\ell = \ell(\ell+1)C_\ell/2\pi$" if args.dell else r"$C_\ell$")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    suffix = "_Dell" if args.dell else ""
    output = Path(args.output) if args.output else output_dir(config, "plot_subdir") / (
        f"{run_name_from_config(config)}_overlay{suffix}_{data_path.stem}.pdf"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(output),
                "data": str(data_path),
                "theory": str(theory_path),
                "sim": str(sim_path) if sim_path else None,
                "dell": bool(args.dell),
                "scaled_reference_errors": bool(args.scaled_reference_errors),
                "error_scales": error_scales,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Workflow YAML config.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("select-cap", help="Choose and write a 600 deg^2 cap center from common masks.")
    _add_common(p)
    p.add_argument("--output-config", default=None, help="Derived config with cap center. Defaults to *.selected.yaml.")
    p.add_argument("--candidate-nside", type=int, default=None)
    p.add_argument("--min-candidate-fraction", type=float, default=0.5)
    p.add_argument("--refine-top-n", type=int, default=None)
    p.set_defaults(func=select_cap)

    p = sub.add_parser("preprocess", help="Stream ASDF halos into the cap-selected HDF5 catalog.")
    _add_common(p)
    p.add_argument("--catalog", action="append", help="Catalog key to build. Defaults to pasting.catalog_key.")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=preprocess)

    p = sub.add_parser("catalog-summary", help="Validate and summarize the saved cap catalog.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.set_defaults(func=catalog_summary)

    p = sub.add_parser("measure-data", help="Measure single-pz cap data spectra with NaMaster.")
    _add_common(p)
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--include-gtau", action="store_true", help="Include g x tau only if a tau field exists.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=measure_data)

    p = sub.add_parser("measure-sim", help="Measure pasted single-pz cap simulation spectra with NaMaster.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--maps", default=None, help="Combined pasted map HDF5. Defaults to the combined map path for --nside.")
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument(
        "--include-gtau",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include the simulation-only g x tau diagnostic. Defaults to pasting.include_diagnostic_gtau.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=measure_sim)

    p = sub.add_parser("measure-scalar-wl", help="Measure scalar g x kappa_wl_tomo diagnostics, bypassing the shear proxy.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--maps", default=None, help="Combined pasted map HDF5. Defaults to the combined map path for --nside.")
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=measure_scalar_wl)

    p = sub.add_parser("measure-total-matter-bias", help="Build total-particle lens-weighted matter closure spectra and b_sim/b_theory JSON.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--maps", default=None, help="Combined pasted map HDF5. Defaults to the combined map path for --nside.")
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--total-root", default=str(DEFAULT_TOTAL_SHELL_ROOT))
    p.add_argument("--halo-root", default=str(DEFAULT_HALO_SHELL_ROOT))
    p.add_argument("--cache-root", default=None)
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    p.add_argument("--log10-mass-min", type=float, default=None)
    p.add_argument("--max-shells", type=int, default=None)
    p.add_argument("--shell-weight-mode", default="average", choices=("average", "midpoint"))
    p.add_argument("--shell-weight-nsamples", type=int, default=32)
    p.add_argument("--batch-parent-pixels", type=int, default=262144)
    p.add_argument("--overwrite-shell-cache", action="store_true")
    p.add_argument("--output", default=None)
    p.add_argument("--summary-output", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=measure_total_matter_bias_closure)

    p = sub.add_parser("direct-field-mass-ledger", help="Write per-shell mass ledger for total, identified-halo, selected pasted-halo, and painted proxy masses.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--total-root", default=str(DEFAULT_TOTAL_SHELL_ROOT))
    p.add_argument("--halo-root", default=str(DEFAULT_HALO_SHELL_ROOT))
    p.add_argument("--cache-root", default=None)
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    p.add_argument("--max-shells", type=int, default=None)
    p.add_argument("--catalog-chunk-size", type=int, default=2000000)
    p.add_argument("--build-painted-template-proxy", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--overwrite-painted-template-proxy", action="store_true")
    p.add_argument("--output", default=None)
    p.set_defaults(func=direct_field_mass_ledger)

    p = sub.add_parser("diagnose-galaxy-density", help="Count pasted galaxies against the configured Stage-31 target density.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--maps", default=None, help="Combined or partial pasted map HDF5. Defaults to combined map unless --num-splits is set.")
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--num-splits", type=int, default=None, help="Read all partial split files for this nside instead of the combined map.")
    p.add_argument("--area-deg2", type=float, default=None, help="Override the area used for target counts. Defaults to the nside cap mask area.")
    p.add_argument("--target-density-per-deg2", type=float, default=None)
    p.add_argument("--retained-fraction", type=float, default=None)
    p.add_argument("--z-bins", type=int, default=20)
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    p.add_argument("--include-hod-mean", action="store_true", help="Also evaluate the HOD mean occupation on the halo catalog.")
    p.add_argument("--hod-platform", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--hod-max-halos", type=int, default=0, help="Limit HOD-mean evaluation; 0 evaluates the full catalog.")
    p.add_argument("--hod-sample-mode", default="random", choices=("head", "random", "largest-paint", "largest-mass", "lowest-z", "highest-z"))
    p.add_argument("--hod-seed", type=int, default=12345)
    p.add_argument("--hod-chunk-size", type=int, default=200000)
    p.add_argument("--output", default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=diagnose_galaxy_density)

    p = sub.add_parser("theory", help="Build single-pz Stage-31 theory and unresolved correction.")
    _add_common(p)
    p.add_argument("--nside", type=int, default=1024, help="Used only to locate the default data measurement path.")
    p.add_argument("--measurement", default=None)
    p.add_argument("--output", default=None)
    p.add_argument(
        "--transition-model",
        default="poweradd",
        choices=("poweradd", "response", "config", "none"),
        help="Default 1h/2h transition model for all supported probe combinations.",
    )
    p.add_argument(
        "--gg-transition-model",
        default=None,
        choices=("poweradd", "response", "config", "none"),
        help="Override galaxy-auto transition only. Defaults to --transition-model.",
    )
    p.add_argument(
        "--tsz-transition-model",
        default=None,
        choices=("poweradd", "response", "config", "none"),
        help="Override y-related transitions only. Defaults to --transition-model.",
    )
    p.add_argument(
        "--galaxy-matter-transition-model",
        default=None,
        choices=("poweradd", "response", "config", "none"),
        help="Override galaxy-matter transitions used by g x kappa and g x shear. Defaults to --transition-model.",
    )
    p.add_argument(
        "--galaxy-electron-transition-model",
        default=None,
        choices=("poweradd", "response", "config", "none"),
        help="Override galaxy-electron transitions used by g x tau and kSZ theory conversion. Defaults to --transition-model.",
    )
    p.add_argument(
        "--sim-matched-transfers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Window theory for pasted-simulation closure: no ACT beams, no default pixwin layer, no DES shear m-bias, and explicit pasted-map transfers.",
    )
    p.set_defaults(func=build_theory)

    p = sub.add_parser("summarize-ratios", help="Write JSON sim/theory ratio diagnostics with bin-1-excluded amplitudes and high-ell tilt.")
    p.add_argument("--sim", required=True, help="Simulation measurement HDF5.")
    p.add_argument("--theory", required=True, help="Windowed theory HDF5.")
    p.add_argument("--component", default="resolved", choices=("full", "resolved", "delta"))
    p.add_argument("--output", default=None)
    p.set_defaults(func=summarize_sim_theory_ratios)

    p = sub.add_parser("paste-split", help="Run one Abacus map-pasting split using the saved cap catalog.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--nside", type=int, required=True)
    p.add_argument("--split-index", type=int, required=True)
    p.add_argument("--num-splits", type=int, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pixel-workers", type=int, default=None, help="CPU workers for pixel-neighbor packages. Use 1 to avoid fork/JAX deadlocks.")
    p.add_argument("--pixel-start-method", default=None, choices=("forkserver", "spawn", "fork"))
    p.add_argument("--pixel-backend", default=None, choices=("healpy", "healpy-buff", "healpy_buff", "healpy-ring", "healpy_ring", "healpy-stencil", "healpy_stencil"))
    p.add_argument("--query-disc-buffer-safety-factor", type=float, default=None)
    p.set_defaults(func=paste_split)

    p = sub.add_parser("benchmark-pixel-work", help="Benchmark CPU HEALPix neighbor-package generation on halo subsets.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--sample-sizes", default="2000,10000,50000", help="Comma-separated halo counts to benchmark.")
    p.add_argument("--sample-mode", default="random", choices=("head", "random", "largest-paint", "largest-mass", "lowest-z", "highest-z"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--workers", default="1,8,16,32,64", help="Comma-separated CPU worker counts.")
    p.add_argument("--pool-chunksizes", default="0,32,128,512", help="Comma-separated Pool.map chunksizes; 0 uses auto.")
    p.add_argument("--pixel-batch-size", type=int, default=None)
    p.add_argument("--pixel-gc-collect-every-n-batches", type=int, default=None)
    p.add_argument("--pixel-start-method", default="fork", choices=("forkserver", "spawn", "fork"))
    p.add_argument("--pixel-backend", default="healpy", choices=("healpy", "healpy-buff", "healpy_buff", "healpy-ring", "healpy_ring", "healpy-stencil", "healpy_stencil"))
    p.add_argument("--query-disc-buffer-safety-factor", type=float, default=2.0)
    p.add_argument("--no-precompute-pixel-groups", action="store_true")
    p.add_argument("--stencil-pixel-angle-factor", type=float, default=1.0)
    p.add_argument(
        "--single-pixel-angle-factor",
        type=float,
        default=0.0,
        help="Shortcut hp.query_disc to ang2pix when paint angle is below this factor times hp.nside2resol. 0 disables.",
    )
    p.add_argument("--max-paint", type=float, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--compare-exact", action="store_true", help="Also build an exact query_disc reference and record mismatch counts.")
    p.add_argument("--exact-workers", type=int, default=1)
    p.add_argument("--exact-pool-chunksize", type=int, default=128)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=benchmark_pixel_work)

    p = sub.add_parser("benchmark-gpu-chunk", help="Benchmark one pasted map chunk with fused/unfused JAX profile evaluation.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--n-halos", type=int, default=2000)
    p.add_argument("--sample-mode", default="random", choices=("head", "random", "largest-paint", "largest-mass", "lowest-z", "highest-z"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--pixel-workers", type=int, default=1)
    p.add_argument("--pixel-pool-chunksize", type=int, default=128)
    p.add_argument("--pixel-batch-size", type=int, default=None)
    p.add_argument("--pixel-gc-collect-every-n-batches", type=int, default=None)
    p.add_argument("--pixel-start-method", default="forkserver", choices=("forkserver", "spawn", "fork"))
    p.add_argument("--pixel-backend", default="healpy", choices=("healpy", "healpy-buff", "healpy_buff", "healpy-ring", "healpy_ring", "healpy-stencil", "healpy_stencil"))
    p.add_argument("--query-disc-buffer-safety-factor", type=float, default=2.0)
    p.add_argument("--stencil-pixel-angle-factor", type=float, default=1.0)
    p.add_argument("--single-pixel-angle-factor", type=float, default=0.5)
    p.add_argument("--max-paint", type=float, default=None)
    p.add_argument("--kappa-source-bin", type=int, default=0)
    p.add_argument("--include-galaxies", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--galaxy-population-backend", default=None, choices=("compact", "padded", "padded_precomputed", "padded-precomputed"))
    p.add_argument("--galaxy-compact-max-satellite-groups", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=benchmark_gpu_chunk)

    p = sub.add_parser("diagnose-hod-stress", help="Summarize HOD, clipping, paint-radius, and optional pixel-work stress by redshift shell.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--nside-list", default="1024,2048")
    p.add_argument("--sample-modes", default="random,largest-paint,largest-mass,lowest-z,highest-hod")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--z-bins", type=int, default=8)
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    p.add_argument("--z-edges", default=None)
    p.add_argument("--hod-platform", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--hod-chunk-size", type=int, default=200000)
    p.add_argument("--pixel-sample-halos", type=int, default=50000)
    p.add_argument("--include-pixel-work", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pixel-workers", type=int, default=16)
    p.add_argument("--pixel-pool-chunksize", type=int, default=32)
    p.add_argument("--pixel-batch-size", type=int, default=None)
    p.add_argument("--pixel-gc-collect-every-n-batches", type=int, default=None)
    p.add_argument("--pixel-start-method", default="fork", choices=("forkserver", "spawn", "fork"))
    p.add_argument("--pixel-backend", default="healpy", choices=("healpy", "healpy-buff", "healpy_buff", "healpy-ring", "healpy_ring", "healpy-stencil", "healpy_stencil"))
    p.add_argument("--query-disc-buffer-safety-factor", type=float, default=2.0)
    p.add_argument("--stencil-pixel-angle-factor", type=float, default=1.0)
    p.add_argument("--single-pixel-angle-factor", type=float, default=0.5)
    p.add_argument("--max-paint", type=float, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=diagnose_hod_stress)

    p = sub.add_parser("benchmark-healpix-functions", help="Benchmark individual healpy and optional jax-healpy pixel functions.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--n-halos", type=int, default=50000)
    p.add_argument("--sample-mode", default="random", choices=("head", "random", "largest-paint", "largest-mass", "lowest-z", "highest-z"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-paint", type=float, default=None)
    p.add_argument("--query-disc-count", type=int, default=2000)
    p.add_argument("--query-disc-buffer-safety-factor", type=float, default=2.0)
    p.add_argument("--run-jax-healpy", action="store_true")
    p.add_argument("--jax-healpy-path", default="/mnt/ceph/users/spandey/ltu-godmax/jax-healpy")
    p.add_argument("--jax-device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--include-jax-query-disc", action="store_true", help="Benchmark batched/JIT jax-healpy query_disc with vmap.")
    p.add_argument("--include-jax-query-disc-loop", action="store_true", help="Also benchmark slow per-query jax-healpy query_disc calls.")
    p.add_argument("--output", default=None)
    p.set_defaults(func=benchmark_healpix_functions)

    p = sub.add_parser("combine-maps", help="Combine pasted map splits.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--nside", type=int, required=True)
    p.add_argument("--num-splits", type=int, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=combine_maps)

    p = sub.add_parser("cache-direct-field-shells", help="Precompute direct non-halo field shell caches.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--total-root", default=str(DEFAULT_TOTAL_SHELL_ROOT), help="All-particle HEALPix shell root.")
    p.add_argument("--halo-root", default=str(DEFAULT_HALO_SHELL_ROOT), help="Identified-halo-particle HEALPix shell root.")
    p.add_argument("--cache-root", default=None, help="Cache root. Defaults to the config particle-shell cache root.")
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    p.add_argument("--max-shells", type=int, default=None)
    p.add_argument("--shell-index-mod", type=int, default=1, help="Cache only shells with zero-based index %% mod == rem.")
    p.add_argument("--shell-index-rem", type=int, default=0, help="Remainder used with --shell-index-mod for Slurm arrays.")
    p.add_argument("--batch-parent-pixels", type=int, default=262_144, help="Output HEALPix pixels processed per input read block.")
    p.add_argument("--overwrite-field-cache", action="store_true")
    p.add_argument("--clip-negative-field-counts", action="store_true", help="Diagnostic only; default raises if total counts are below halo counts.")
    p.add_argument("--output", default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cache_direct_field_shells)

    p = sub.add_parser("build-direct-field-map", help="Build pasted plus direct non-halo field-shell HDF5 product.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--maps", default=None, help="Base combined pasted map HDF5. Defaults to the combined map path for --nside.")
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--total-root", default=str(DEFAULT_TOTAL_SHELL_ROOT), help="All-particle HEALPix shell root.")
    p.add_argument("--halo-root", default=str(DEFAULT_HALO_SHELL_ROOT), help="Identified-halo-particle HEALPix shell root.")
    p.add_argument("--cache-root", default=None, help="Cache root. Defaults to the config particle-shell cache root.")
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=None)
    p.add_argument("--max-shells", type=int, default=None)
    p.add_argument("--log10-mass-min", type=float, default=None, help="Mass cut used only while setting up GODMAX kernels.")
    p.add_argument("--shell-weight-mode", default="average", choices=("average", "midpoint"))
    p.add_argument("--shell-weight-nsamples", type=int, default=48)
    p.add_argument("--gg-transition-model", default="poweradd", help="Theory setup option; use 'config' to leave config default.")
    p.add_argument("--batch-parent-pixels", type=int, default=262_144, help="Output HEALPix pixels processed per input read block.")
    p.add_argument("--overwrite-field-cache", action="store_true")
    p.add_argument("--clip-negative-field-counts", action="store_true", help="Diagnostic only; default raises if total counts are below halo counts.")
    p.add_argument("--output", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=build_direct_field_map)

    p = sub.add_parser("plot-healpix-maps", help="Plot pasted HEALPix maps plus halo/galaxy fields in one PDF.")
    _add_common(p)
    p.add_argument("--catalog", default=None)
    p.add_argument("--maps", default=None, help="Pasted map HDF5. Defaults to the combined map path for --nside.")
    p.add_argument("--nside", type=int, required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--xsize", type=int, default=360)
    p.add_argument("--reso-arcmin", type=float, default=None)
    p.add_argument("--ncols", type=int, default=3)
    p.set_defaults(func=plot_healpix_maps)

    p = sub.add_parser("plot", help="Plot single-pz data, theory, optional sim, and corrected sim overlays.")
    _add_common(p)
    p.add_argument("--nside", type=int, default=1024, help="Used only to locate the default data measurement path.")
    p.add_argument("--data", default=None)
    p.add_argument("--theory", required=True)
    p.add_argument("--sim", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--dell", action="store_true", help="Plot D_ell = ell(ell+1) C_ell / 2pi.")
    p.add_argument(
        "--scaled-reference-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use full Stage-31 covariance errors scaled by mask-overlap fsky when the cap product has no covariance.",
    )
    p.set_defaults(func=plot_overlay)

    p = sub.add_parser(
        "plot-full-data",
        help="Plot one DESI pz bin using the full-footprint Stage-31 data/covariance and optional cap simulation overlay.",
    )
    _add_common(p)
    p.add_argument("--measurement", default=None, help="Full-footprint measurement HDF5. Defaults to godmax.measurement_h5.")
    p.add_argument("--fiducial-vector", default=str(DEFAULT_STAGE31_FIDUCIAL_VECTOR))
    p.add_argument("--bestfit-vector", default=str(DEFAULT_STAGE31_BESTFIT_VECTOR))
    p.add_argument("--sim", default=None, help="Optional cap simulation measurement HDF5.")
    p.add_argument("--output", default=None)
    p.add_argument("--include-fiducial", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--raw-cl", action="store_true", help="Plot raw C_ell for every panel instead of the Stage-31 D_ell convention.")
    p.add_argument("--dell", action="store_true", help="Plot D_ell = ell(ell+1) C_ell / 2pi for every panel.")
    p.add_argument(
        "--plot-ell-max",
        type=float,
        default=None,
        help="Maximum ell to show in the plot. Use <=0 to show all available bandpowers.",
    )
    p.add_argument(
        "--ksz-ylim",
        type=_parse_ksz_ylim,
        default=None,
        metavar="YMIN,YMAX",
        help="y-axis limits for the kSZ pi x T panel. Accepts YMIN,YMAX or YMIN YMAX. Defaults to -5e-5 5e-5.",
    )
    p.set_defaults(func=plot_full_data)

    p = sub.add_parser(
        "plot-full-data-theory-variants",
        help="Plot one DESI pz bin with full data, simple-sum theory, response theory, and optional cap simulation.",
    )
    _add_common(p)
    p.add_argument("--measurement", default=None, help="Full-footprint measurement HDF5. Defaults to godmax.measurement_h5.")
    p.add_argument("--sum-theory", required=True, help="Theory HDF5 built with all supported transition models set to poweradd.")
    p.add_argument("--response-theory", required=False, default=None, help="Theory HDF5 built with all supported transition models set to response. Optional: if omitted, only the power-add (1h+2h) theory curve is shown.")
    p.add_argument("--sim", default=None, help="Optional cap simulation measurement HDF5.")
    p.add_argument("--sim-label", default=None, help="Legend label for --sim.")
    p.add_argument("--extra-sim", action="append", default=[], help="Additional simulation measurement HDF5 to overlay. Repeatable.")
    p.add_argument(
        "--extra-sim-label",
        action="append",
        default=[],
        help="Legend label for an --extra-sim entry. Repeatable in the same order as --extra-sim.",
    )
    p.add_argument("--nside", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--theory-component", default="resolved", choices=("full", "resolved"))
    p.add_argument("--raw-cl", action="store_true", help="Plot raw C_ell instead of D_ell.")
    p.add_argument(
        "--plot-ell-max",
        type=float,
        default=None,
        help="Maximum ell to show in the plot. Use <=0 to show all available bandpowers.",
    )
    p.add_argument(
        "--ksz-ylim",
        type=_parse_ksz_ylim,
        default=None,
        metavar="YMIN,YMAX",
        help="y-axis limits for the kSZ pi x T panel. Accepts YMIN,YMAX or YMIN YMAX. Defaults to -5e-5 5e-5.",
    )
    p.add_argument(
        "--gray-unused-bands",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Shade bandpowers excluded by the likelihood scale cuts (config 'likelihood_cuts'), "
            "matching the survey full_dell_comparison figure. Default: auto-on when the config "
            "carries a 'likelihood_cuts' block."
        ),
    )
    p.add_argument(
        "--plot-xscale",
        default="linear",
        choices=("linear", "log"),
        help="X-axis (multipole) scale for the panels. Default linear.",
    )
    p.set_defaults(func=plot_full_data_theory_variants)
    return parser


def _parse_ksz_ylim(value: str) -> Tuple[float, float]:
    parts = [part for part in str(value).replace(",", " ").split() if part]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--ksz-ylim expects two values: YMIN,YMAX.")
    return (float(parts[0]), float(parts[1]))


def _normalize_ksz_ylim_argv(argv: Optional[Sequence[str]]) -> Sequence[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    out = []
    i = 0
    while i < len(raw):
        token = raw[i]
        if token == "--ksz-ylim" and i + 2 < len(raw):
            out.append(f"--ksz-ylim={raw[i + 1]},{raw[i + 2]}")
            i += 3
            continue
        if token.startswith("--ksz-ylim="):
            value = token.split("=", 1)[1]
            if "," not in value and i + 1 < len(raw) and not raw[i + 1].startswith("--"):
                out.append(f"--ksz-ylim={value},{raw[i + 1]}")
                i += 2
                continue
        out.append(token)
        i += 1
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(_normalize_ksz_ylim_argv(argv))
    args.func(args)


if __name__ == "__main__":
    main()
