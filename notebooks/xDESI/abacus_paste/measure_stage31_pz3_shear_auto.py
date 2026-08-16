#!/usr/bin/env python3
"""Measure the pasted DES source-bin-3 shear auto with the Stage-31 NaMaster setup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import stage31_pz1_backlight_validation as workflow


SPECTRUM_NAME = "des_shear_EE_tomo3_tomo3"
SOURCE_DATASET = "maps/map_kappa_wl_tomo3"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "stage31_pz3_cap2400_map64fcen_mmin11p147538_lmax3000_13log.selected.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--maps", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths and conventions without loading or transforming the full map.",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path, int]:
    config_path = args.config.expanduser().resolve()
    config = workflow.read_config(config_path)
    if workflow.pz_bin_from_config(config) != 3:
        raise ValueError("The focused publication measurement requires pasting.pz_bin=3.")
    if not bool(config["godmax"].get("override_cosmology_from_catalog", False)):
        raise ValueError("The selected configuration must use the Abacus catalog cosmology.")
    nside = int(config["pasting"]["nside"])
    catalog_key = workflow.default_catalog_key(config)
    map_path = (
        args.maps or workflow.final_map_path(config, catalog_key, nside)
    ).expanduser().resolve()
    run_root = Path(config["project"]["output_root"]).expanduser().resolve()
    output = (
        args.output
        or run_root
        / "measurements"
        / f"sim_{SPECTRUM_NAME}_pz3_cap2400_map64fcen_nside{nside}_lmax{int(config['pasting']['lmax'])}_nbin{int(config['pasting']['n_bins'])}_log.h5"
    ).expanduser().resolve()
    if not map_path.is_file():
        raise FileNotFoundError(map_path)
    with h5py.File(map_path, "r") as h5:
        if SOURCE_DATASET not in h5:
            raise KeyError(f"{map_path} is missing {SOURCE_DATASET!r}.")
        source = h5[SOURCE_DATASET]
        expected_npix = 12 * nside**2
        if source.shape != (expected_npix,):
            raise ValueError(
                f"{SOURCE_DATASET} has shape {source.shape}; expected {(expected_npix,)}."
            )
        if int(h5.attrs.get("nside", -1)) != nside:
            raise ValueError("Pasted-map nside does not match the selected configuration.")
    return config, map_path, output, nside


def make_shear_field(
    config: dict[str, Any],
    map_path: Path,
    nside: int,
    measurement_config: workflow.mpn.MeasurementConfig,
) -> tuple[workflow.mpn.FieldMap, dict[str, Any]]:
    with h5py.File(map_path, "r") as h5:
        kappa = np.asarray(h5[SOURCE_DATASET][:], dtype=np.float64)
        pasted_attrs = {
            str(key): (value.decode("utf-8") if isinstance(value, bytes) else value)
            for key, value in h5.attrs.items()
        }

    center_ra, center_dec, radius_deg = workflow.require_cap_center(config)
    cap = workflow.cap_pixel_mask(nside, center_ra, center_dec, radius_deg)
    mode = str(config["pasting"].get("sim_measurement_mask_mode", "reference")).lower()
    common_cap = mode in {"cap", "common_cap", "binary_cap"}
    reference = workflow.reference_field_info(Path(config["godmax"]["map_h5"]), "s3", nside)
    mask = cap if common_cap else np.clip(np.asarray(reference["mask"]) * cap, 0.0, None)

    kappa = workflow.subtract_weighted_mask_mean(kappa, mask)
    gamma1, gamma2 = workflow.kappa_to_namaster_shear_maps(
        kappa,
        nside,
        int(measurement_config.lmax),
    )
    gamma1[mask <= 0.0] = 0.0
    gamma2[mask <= 0.0] = 0.0
    metadata = dict(reference["metadata"])
    metadata.update(
        {
            "source": "Abacus Backlight pasted map",
            "pasted_map_h5": str(map_path),
            "pasted_dataset": SOURCE_DATASET,
            "des_source_tomo": 3,
            "shape_noise_pseudo_cl": 0.0,
            "shape_noise_note": "No DES shape noise is added to the pasted simulation.",
            "mask_apodization_applied": False,
            "mask_apodization_deg": 0.0,
            "mask_apodization_type": "none",
            "mask_apodization_note": (
                "The simulation field uses the selected binary cap mask, not the "
                "apodized DES survey mask inherited from the reference field metadata."
            ),
            "input_spin_convention": (
                "E-only spin-2 shear proxy generated from pasted convergence with "
                "healpy.alm2map_spin."
            ),
            "finite_cap_caveat": (
                "The shear proxy is built from the cap-limited convergence map; it is not "
                "a padded or full-sky shear construction."
            ),
            "shear_e_to_kappa_sign": -1.0,
            "sim_measurement_mask_mode": mode,
            "sim_measurement_common_cap_mask": common_cap,
        }
    )
    field = workflow.mpn.FieldMap(
        name="s3",
        label="Abacus Backlight pasted DES source-bin 3 shear-E proxy",
        kind="des_shear",
        spin=2,
        maps=[gamma1, gamma2],
        mask=np.asarray(mask, dtype=np.float64),
        mask_name=f"des_shear_tomo3_{workflow.cap_tag_from_config(config)}",
        metadata=metadata,
    )
    provenance = {
        "schema": "stage31_pz3_pasted_shear_auto_inputs_v1",
        "pasted_map_h5": str(map_path),
        "pasted_dataset": SOURCE_DATASET,
        "pasted_map_attrs": pasted_attrs,
        "reference_map_h5": str(Path(config["godmax"]["map_h5"])),
        "reference_field": "s3",
        "cap": {
            "center_ra_deg": center_ra,
            "center_dec_deg": center_dec,
            "radius_deg": radius_deg,
            "area_deg2_requested": float(config["sky_patch"]["area_deg2"]),
        },
        "sim_measurement_mask_mode": mode,
        "sim_measurement_common_cap_mask": common_cap,
        "field_metadata": {"s3": metadata},
    }
    return field, provenance


def write_measurement(
    output: Path,
    result: dict[str, Any],
    provenance: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    spectrum = result["spectra"][SPECTRUM_NAME]
    provenance = workflow.gmt.to_jsonable(provenance)
    measurement_config = workflow.gmt.to_jsonable(result["config"])
    with h5py.File(tmp, "w", track_order=True) as h5:
        h5.attrs["schema"] = "stage31_pz3_pasted_shear_auto_measurement_v1"
        h5.attrs["created_utc"] = str(result["created_utc"])
        h5.attrs["spectrum_name"] = SPECTRUM_NAME
        h5.attrs["estimator"] = "NaMaster decoupled spin-2 EE pseudo-Cl"
        h5.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        h5.attrs["measurement_config_json"] = json.dumps(measurement_config, sort_keys=True)
        h5.create_dataset("ell", data=np.asarray(result["ell"], dtype=np.float64))
        h5.create_dataset("ell_left", data=np.asarray(result["ell_left"], dtype=np.int32))
        h5.create_dataset("ell_right", data=np.asarray(result["ell_right"], dtype=np.int32))
        group = h5.create_group(f"spectra/{SPECTRUM_NAME}")
        group.attrs["family"] = str(spectrum["family"])
        group.attrs["component"] = int(spectrum["component"])
        group.attrs["component_label"] = str(spectrum["component_label"])
        group.attrs["component_labels_json"] = json.dumps(spectrum["component_labels"])
        group.attrs["fields_json"] = json.dumps(list(spectrum["fields"]))
        group.attrs["metadata_json"] = json.dumps(
            workflow.gmt.to_jsonable(spectrum["metadata"]),
            sort_keys=True,
        )
        group.create_dataset("ell", data=np.asarray(spectrum["ell"], dtype=np.float64))
        group.create_dataset("cl", data=np.asarray(spectrum["cl"], dtype=np.float64))
        group.create_dataset(
            "cl_all_components",
            data=np.asarray(spectrum["cl_all_components"], dtype=np.float64),
        )
        group.create_dataset(
            "pcl_all_components",
            data=np.asarray(spectrum["pcl_all_components"], dtype=np.float64),
        )
        group.create_dataset(
            "bandpower_window_selected",
            data=np.asarray(spectrum["bandpower_window_selected"], dtype=np.float64),
        )
    os.replace(tmp, output)


def main() -> None:
    args = parse_args()
    config, map_path, output, nside = resolve_inputs(args)
    summary = {
        "config": str(args.config.expanduser().resolve()),
        "maps": str(map_path),
        "source_dataset": SOURCE_DATASET,
        "output": str(output),
        "spectrum": SPECTRUM_NAME,
        "nside": nside,
        "lmax": int(config["pasting"]["lmax"]),
        "n_bins": int(config["pasting"]["n_bins"]),
        "binning": str(config["pasting"]["binning"]),
        "random_seed": int(config["pasting"]["random_seed"]),
    }
    if args.check_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    measurement_config = workflow.measurement_config_from_workflow(
        config,
        nside,
        "stage31_pz3_cap2400_map64fcen_shear_auto",
    )
    if bool(measurement_config.compute_covariance):
        raise ValueError("This focused noiseless simulation measurement must not compute covariance.")
    field, provenance = make_shear_field(config, map_path, nside, measurement_config)
    spec = next(
        spec for spec in workflow.mpn.default_spectrum_specs() if spec.name == SPECTRUM_NAME
    )
    result = workflow.mpn.measure_all(
        {"s3": field},
        measurement_config,
        specs=[spec],
        verbose=not args.quiet,
    )
    cl = np.asarray(result["spectra"][SPECTRUM_NAME]["cl"], dtype=np.float64)
    if cl.size != int(config["pasting"]["n_bins"]) or not np.all(np.isfinite(cl)):
        raise ValueError("Measured shear-auto bandpowers have an invalid size or non-finite values.")
    write_measurement(output, result, provenance, overwrite=bool(args.overwrite))
    summary["ell"] = np.asarray(result["ell"], dtype=np.float64).tolist()
    summary["cl"] = cl.tolist()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
