"""Measure all four DESI pz bins against a shared all-z Abacus cap paste."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import h5py
import healpy as hp
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
XDESI_DIR = THIS_DIR.parent
REPO_ROOT = XDESI_DIR.parents[1]
SURVEY_MEASURE_DIR = XDESI_DIR / "survey_measure"
for _path in (THIS_DIR, XDESI_DIR, SURVEY_MEASURE_DIR, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import godmax_multiprobe_theory_utils as gmt  # noqa: E402
import multiprobe_namaster as mpn  # noqa: E402
import stage31_pz1_backlight_validation as st  # noqa: E402
from abacus_lightcone_catalog import ensure_under_xdesi  # noqa: E402


def _jsonable_attr(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _read_h5_attrs(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        return {str(key): _jsonable_attr(value) for key, value in h5.attrs.items()}


def _load_galaxies(path: Path) -> Tuple[np.ndarray, dict]:
    with h5py.File(path, "r") as h5:
        galaxies = h5["galaxies"][:]
        attrs = {str(key): _jsonable_attr(value) for key, value in h5.attrs.items()}
    return galaxies, attrs


def _parse_pz_path(values: Optional[Sequence[str]], *, option: str) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"{option} entries must look like pz1=/path/to/file.h5, got {raw!r}.")
        key, path = raw.split("=", 1)
        key = key.strip().lower().replace("pz", "")
        pz_bin = int(key)
        if pz_bin not in {1, 2, 3, 4}:
            raise ValueError(f"{option} pz bin must be 1..4, got {pz_bin}.")
        out[pz_bin] = Path(path).expanduser().resolve()
    return out


def _require_all_pz(paths: Mapping[int, Path], *, label: str) -> None:
    missing = [pz for pz in range(1, 5) if pz not in paths]
    if missing:
        raise ValueError(f"Missing {label} for pz bins: {missing}")
    missing_paths = {pz: str(path) for pz, path in paths.items() if not path.exists()}
    if missing_paths:
        raise FileNotFoundError(f"Missing {label} files: {missing_paths}")


def _sim_cap_context(config: Mapping[str, object], nside: int):
    center_ra, center_dec, radius_deg = st.require_cap_center(config)
    cap = st.cap_pixel_mask(int(nside), center_ra, center_dec, radius_deg)
    sim_mask_mode = str(config.get("pasting", {}).get("sim_measurement_mask_mode", "reference")).lower()
    use_common_cap_mask = sim_mask_mode in {"cap", "common_cap", "binary_cap"}

    def measurement_mask(ref_info: Mapping[str, object]) -> np.ndarray:
        if use_common_cap_mask:
            return cap.copy()
        return np.clip(np.asarray(ref_info["mask"], dtype=np.float64) * cap, 0.0, None)

    return center_ra, center_dec, radius_deg, cap, sim_mask_mode, use_common_cap_mask, measurement_mask


def _galaxy_fields_for_pz(
    config: Mapping[str, object],
    *,
    pz_bin: int,
    galaxies: np.ndarray,
    map_path: Path,
    nside: int,
    measurement_mask,
    sim_mask_mode: str,
    use_common_cap_mask: bool,
    build_pi: bool,
) -> Dict[str, mpn.FieldMap]:
    ref_map_path = Path(config["godmax"]["map_h5"])
    cap_tag = st.cap_tag_from_config(config)
    pz = f"pz{int(pz_bin)}"
    out: Dict[str, mpn.FieldMap] = {}

    g_ref = st.reference_field_info(ref_map_path, f"g{pz_bin}", int(nside))
    g_mask = measurement_mask(g_ref)
    g_delta, g_meta = st.galaxy_delta_for_mask(galaxies, int(nside), g_mask)
    g_metadata = copy.deepcopy(g_ref["metadata"])
    g_metadata.update(
        {
            **g_meta,
            "source": "Abacus Backlight pasted galaxy catalog",
            "pasted_map_h5": str(map_path),
            "sim_measurement_mask_mode": sim_mask_mode,
            "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            "photoz_vs_truez": (
                f"{pz} label uses Stage-31 HOD/true-n(z); simulated galaxies are not assigned or cut by photo-z."
            ),
        }
    )
    out[f"g{pz_bin}"] = mpn.FieldMap(
        name=f"g{pz_bin}",
        label=f"Abacus Backlight pasted DESI {pz} galaxy overdensity",
        kind="desi_galaxy",
        spin=0,
        maps=[g_delta],
        mask=g_mask,
        mask_name=f"desi_dr9_random_{cap_tag}",
        metadata=g_metadata,
    )

    if not build_pi:
        return out

    pi_ref = st.reference_field_info(ref_map_path, f"pi{pz_bin}", int(nside))
    pi_mask = g_mask.copy()
    ksz_velocity_mode = str(config.get("pasting", {}).get("ksz_velocity_mode", "true_velocity")).lower()
    if ksz_velocity_mode == "photoz_reconstruction_emulation" and galaxies.shape[1] < 7:
        raise ValueError(
            "pasting.ksz_velocity_mode=photoz_reconstruction_emulation requires a pasted galaxy catalog "
            "with host_vlos_kms."
        )

    if ksz_velocity_mode == "photoz_reconstruction_emulation":
        ref_pi_meta = pi_ref["metadata"]
        sigma_rec_over_c = float(ref_pi_meta["rms_rec_vr_over_c_weighted"])
        r_corr = float(ref_pi_meta.get("ksz_photoz_velocity_correlation_r", mpn.KSZ_PHOTOZ_VELOCITY_CORRELATION_R))
        noise_seed = int(config.get("pasting", {}).get("ksz_reconstruction_noise_seed", 12345))
        pi_map, pi_meta, pi_catalog = st.galaxy_momentum_for_mask(
            galaxies,
            int(nside),
            pi_mask,
            velocity_mode="photoz_reconstruction_emulation",
            velocity_correlation_r=r_corr,
            sigma_rec_over_c=sigma_rec_over_c,
            reconstruction_noise_seed=noise_seed,
        )
        pi_estimator_note = (
            "Simulation catalog momentum estimator with photo-z velocity-reconstruction emulation. "
            "The data-facing kSZ theory C_ell^piT = -T_CMB_uK * A_v * C_ell^gtau applies without modification."
        )
    else:
        pi_map, pi_meta, pi_catalog = st.galaxy_momentum_for_mask(galaxies, int(nside), pi_mask)
        pi_estimator_note = (
            "Simulation catalog momentum estimator: positions=(ra_deg, dec_deg), weights=1, "
            "field=host_vlos_kms/c."
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
    label = (
        f"Abacus Backlight reconstruction-emulated momentum {pz}"
        if ksz_velocity_mode == "photoz_reconstruction_emulation"
        else f"Abacus Backlight true-velocity momentum {pz}"
    )
    out[f"pi{pz_bin}"] = mpn.FieldMap(
        name=f"pi{pz_bin}",
        label=label,
        kind="desi_momentum",
        spin=0,
        maps=[pi_map],
        mask=pi_mask,
        mask_name=f"desi_dr9_random_{cap_tag}",
        metadata=pi_metadata,
        catalog=pi_catalog,
    )
    return out


def _continuous_fields(
    config: Mapping[str, object],
    *,
    maps: Mapping[str, np.ndarray],
    attrs: Mapping[str, object],
    map_path: Path,
    nside: int,
    measurement_mask,
    cap: np.ndarray,
    sim_mask_mode: str,
    use_common_cap_mask: bool,
    include_gtau: bool,
) -> Dict[str, mpn.FieldMap]:
    ref_map_path = Path(config["godmax"]["map_h5"])
    cap_tag = st.cap_tag_from_config(config)
    mcfg = st.measurement_config_from_workflow(config, int(nside), f"{st.run_name_from_config(config)}_allpz_sim_nside{nside}")
    fields: Dict[str, mpn.FieldMap] = {}

    if "map_ksz" in maps:
        t_ref = st.reference_field_info(ref_map_path, "T", int(nside))
        t_mask = measurement_mask(t_ref)
        fields["T"] = mpn.FieldMap(
            name="T",
            label="Abacus Backlight all-z pasted kSZ temperature",
            kind=t_ref["kind"],
            spin=0,
            maps=[st.subtract_weighted_mask_mean(mpn.TCMB_UK * np.asarray(maps["map_ksz"], dtype=np.float64), t_mask)],
            mask=t_mask,
            mask_name=f"{t_ref['mask_name']}_{cap_tag}",
            metadata={
                **copy.deepcopy(t_ref["metadata"]),
                "source": "Abacus Backlight all-z pasted kSZ temperature map",
                "pasted_map_h5": str(map_path),
                "pasted_dataset": "map_ksz",
                "map_ksz_input_units": "Delta T / T_CMB",
                "temperature_units": "uK",
                "temperature_conversion": f"T_uK = {mpn.TCMB_UK:g} * map_ksz",
                "masked_mean_subtracted_for_measurement": True,
                "sim_measurement_mask_mode": sim_mask_mode,
                "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            },
        )

    scalar_map_to_field = {
        "y": ("map_ymap", "y", "Abacus Backlight all-z pasted tSZ Compton-y"),
        "kappa": ("map_kappa_cmb", "kappa", "Abacus Backlight all-z pasted CMB lensing kappa"),
    }
    for out_name, (dataset, ref_name, label) in scalar_map_to_field.items():
        if dataset not in maps:
            continue
        ref = st.reference_field_info(ref_map_path, ref_name, int(nside))
        mask = measurement_mask(ref)
        fields[out_name] = mpn.FieldMap(
            name=out_name,
            label=label,
            kind=ref["kind"],
            spin=0,
            maps=[st.subtract_weighted_mask_mean(maps[dataset], mask)],
            mask=mask,
            mask_name=f"{ref['mask_name']}_{cap_tag}",
            metadata={
                **copy.deepcopy(ref["metadata"]),
                "source": "Abacus Backlight all-z pasted map",
                "pasted_map_h5": str(map_path),
                "pasted_dataset": dataset,
                "masked_mean_subtracted_for_measurement": True,
                "sim_measurement_mask_mode": sim_mask_mode,
                "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            },
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
        ref = st.reference_field_info(ref_map_path, f"s{tomo}", int(nside))
        mask = measurement_mask(ref)
        kappa_wl = st.subtract_weighted_mask_mean(maps[dataset], mask)
        gamma1, gamma2 = st.kappa_to_namaster_shear_maps(kappa_wl, int(nside), int(mcfg.lmax))
        gamma1[mask <= 0.0] = 0.0
        gamma2[mask <= 0.0] = 0.0
        fields[f"s{tomo}"] = mpn.FieldMap(
            name=f"s{tomo}",
            label=f"Abacus Backlight all-z pasted DES source-bin {tomo} shear-E proxy",
            kind="des_shear",
            spin=2,
            maps=[gamma1, gamma2],
            mask=mask,
            mask_name=f"des_shear_tomo{tomo}_{cap_tag}",
            metadata={
                **copy.deepcopy(ref["metadata"]),
                "source": f"Abacus Backlight all-z pasted {dataset}",
                "pasted_map_h5": str(map_path),
                "pasted_dataset": dataset,
                "des_source_tomo": tomo,
                "shape_noise_pseudo_cl": 0.0,
                "shape_noise_note": "No DES shape noise is added for the pasted simulation cross-spectrum.",
                "input_spin_convention": "E-only spin-2 shear proxy generated from pasted convergence with healpy.alm2map_spin.",
                "finite_cap_caveat": "The shear proxy is built from the cap-limited convergence map.",
                "shear_e_to_kappa_sign": -1.0,
                "sim_measurement_mask_mode": sim_mask_mode,
                "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            },
        )

    if include_gtau and "map_tau" in maps:
        tau_mask = cap.copy()
        fields["tau"] = mpn.FieldMap(
            name="tau",
            label="Abacus Backlight all-z pasted optical-depth tau",
            kind="sim_tau",
            spin=0,
            maps=[st.subtract_weighted_mask_mean(maps["map_tau"], tau_mask)],
            mask=tau_mask,
            mask_name=cap_tag,
            metadata={
                "source": "Abacus Backlight all-z pasted map",
                "pasted_map_h5": str(map_path),
                "pasted_dataset": "map_tau",
                "diagnostic": "simulation-only optical-depth cross-spectrum",
                "masked_mean_subtracted_for_measurement": True,
                "sim_measurement_mask_mode": sim_mask_mode,
                "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            },
        )

    return fields


def _specs_for_pz_bins(
    pz_bins: Iterable[int],
    *,
    include_gtau: bool,
    available_fields: Iterable[str],
) -> list[mpn.SpectrumSpec]:
    specs: list[mpn.SpectrumSpec] = []
    for pz_bin in pz_bins:
        specs.extend(
            st.pz_spectrum_specs(
                int(pz_bin),
                include_gtau=include_gtau,
                available_fields=available_fields,
                require_core=False,
            )
        )
    return specs


def _metadata(
    config: Mapping[str, object],
    *,
    continuous_path: Path,
    continuous_attrs: Mapping[str, object],
    galaxy_paths: Mapping[int, Path],
    galaxy_attrs: Mapping[int, Mapping[str, object]],
    fields: Mapping[str, mpn.FieldMap],
    specs: Sequence[mpn.SpectrumSpec],
    nside: int,
    cap: Tuple[float, float, float],
    sim_mask_mode: str,
    use_common_cap_mask: bool,
) -> dict:
    center_ra, center_dec, radius_deg = cap
    return gmt.to_jsonable(
        {
            "schema": "stage31_allpz_cap_sim_maps_for_namaster_v1",
            "continuous_map_h5": str(continuous_path),
            "continuous_map_attrs": dict(continuous_attrs),
            "galaxy_map_h5_by_pz": {f"pz{pz}": str(path) for pz, path in sorted(galaxy_paths.items())},
            "galaxy_map_attrs_by_pz": {f"pz{pz}": dict(galaxy_attrs[pz]) for pz in sorted(galaxy_attrs)},
            "cap": {
                "center_ra_deg": float(center_ra),
                "center_dec_deg": float(center_dec),
                "radius_deg": float(radius_deg),
                "area_deg2_requested": float(config["sky_patch"]["area_deg2"]),
                "nside": int(nside),
            },
            "sim_measurement_mask_mode": sim_mask_mode,
            "sim_measurement_common_cap_mask": bool(use_common_cap_mask),
            "spectra_measured": [spec.name for spec in specs],
            "field_metadata": st.field_metadata(fields),
            "comparison_caveat": (
                "Shared continuous y/T/kappa/shear/tau maps are pasted over 0 < z < 1.2. "
                "DESI galaxy and momentum fields are pz-specific HOD pastes; pz3 is reused from the "
                "existing hmcfailed cap2400 product."
            ),
        }
    )


def _write_result(
    output: Path,
    *,
    fields: Mapping[str, mpn.FieldMap],
    specs: Sequence[mpn.SpectrumSpec],
    mcfg: mpn.MeasurementConfig,
    metadata: Mapping[str, object],
    overwrite: bool,
    quiet: bool,
) -> None:
    ensure_under_xdesi(output.resolve())
    result = mpn.measure_all(fields, mcfg, specs=specs, verbose=not quiet)
    mpn.save_measurement_product(output, result, metadata, overwrite=overwrite)


def measure_allpz(args: argparse.Namespace) -> None:
    config = st.read_config(args.config)
    nside = int(args.nside or config["pasting"].get("nside", 2048))
    include_gtau = (
        bool(config["pasting"].get("include_diagnostic_gtau", True))
        if args.include_gtau is None
        else bool(args.include_gtau)
    )
    continuous_path = Path(args.continuous_maps).expanduser().resolve()
    if not continuous_path.exists():
        raise FileNotFoundError(continuous_path)
    galaxy_paths = _parse_pz_path(args.galaxy_map, option="--galaxy-map")
    _require_all_pz(galaxy_paths, label="galaxy map")
    per_pz_outputs = _parse_pz_path(args.per_pz_output, option="--per-pz-output")

    maps, _unused_galaxies, continuous_attrs = st.load_maps_h5(continuous_path)
    continuous_attrs = {str(key): _jsonable_attr(value) for key, value in continuous_attrs.items()}
    center_ra, center_dec, radius_deg, cap_mask, sim_mask_mode, use_common_cap_mask, measurement_mask = _sim_cap_context(
        config, nside
    )

    fields = _continuous_fields(
        config,
        maps=maps,
        attrs=continuous_attrs,
        map_path=continuous_path,
        nside=nside,
        measurement_mask=measurement_mask,
        cap=cap_mask,
        sim_mask_mode=sim_mask_mode,
        use_common_cap_mask=use_common_cap_mask,
        include_gtau=include_gtau,
    )
    galaxy_attrs: Dict[int, dict] = {}
    build_pi = "T" in fields
    for pz_bin in range(1, 5):
        galaxies, attrs = _load_galaxies(galaxy_paths[pz_bin])
        galaxy_attrs[pz_bin] = attrs
        fields.update(
            _galaxy_fields_for_pz(
                config,
                pz_bin=pz_bin,
                galaxies=galaxies,
                map_path=galaxy_paths[pz_bin],
                nside=nside,
                measurement_mask=measurement_mask,
                sim_mask_mode=sim_mask_mode,
                use_common_cap_mask=use_common_cap_mask,
                build_pi=build_pi,
            )
        )

    specs = _specs_for_pz_bins(range(1, 5), include_gtau=include_gtau, available_fields=fields.keys())
    if not specs:
        raise RuntimeError(f"No spectra can be measured from fields: {sorted(fields)}")

    mcfg = st.measurement_config_from_workflow(config, nside, f"{st.run_name_from_config(config)}_allpz_sim_nside{nside}")
    output = Path(args.output).expanduser().resolve()
    metadata = _metadata(
        config,
        continuous_path=continuous_path,
        continuous_attrs=continuous_attrs,
        galaxy_paths=galaxy_paths,
        galaxy_attrs=galaxy_attrs,
        fields=fields,
        specs=specs,
        nside=nside,
        cap=(center_ra, center_dec, radius_deg),
        sim_mask_mode=sim_mask_mode,
        use_common_cap_mask=use_common_cap_mask,
    )
    _write_result(output, fields=fields, specs=specs, mcfg=mcfg, metadata=metadata, overwrite=args.overwrite, quiet=args.quiet)

    per_pz_written = {}
    for pz_bin, out_path in sorted(per_pz_outputs.items()):
        pz_specs = _specs_for_pz_bins([pz_bin], include_gtau=include_gtau, available_fields=fields.keys())
        if not pz_specs:
            continue
        pz_metadata = copy.deepcopy(metadata)
        pz_metadata["schema"] = "stage31_single_pz_from_allpz_cap_sim_maps_for_namaster_v1"
        pz_metadata["selected_pz_bin"] = int(pz_bin)
        pz_metadata["spectra_measured"] = [spec.name for spec in pz_specs]
        mcfg_pz = copy.deepcopy(mcfg)
        mcfg_pz.stage = f"{st.run_name_from_config(config)}_pz{pz_bin}_sim_nside{nside}"
        _write_result(
            out_path.expanduser().resolve(),
            fields=fields,
            specs=pz_specs,
            mcfg=mcfg_pz,
            metadata=pz_metadata,
            overwrite=args.overwrite,
            quiet=args.quiet,
        )
        per_pz_written[f"pz{pz_bin}"] = str(out_path.expanduser().resolve())

    print(
        json.dumps(
            {
                "output": str(output),
                "per_pz_outputs": per_pz_written,
                "continuous_maps": str(continuous_path),
                "galaxy_maps": {f"pz{pz}": str(path) for pz, path in sorted(galaxy_paths.items())},
                "n_spectra": len(specs),
                "spectra": [spec.name for spec in specs],
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="All-z selected config carrying the shared cap and measurement settings.")
    parser.add_argument("--continuous-maps", required=True, help="All-z pasted maps HDF5.")
    parser.add_argument(
        "--galaxy-map",
        action="append",
        default=[],
        help="Tomographic galaxy map, formatted pzN=/path/to/map.h5. Repeat for pz1..pz4.",
    )
    parser.add_argument("--output", required=True, help="All-pz combined measurement output HDF5.")
    parser.add_argument(
        "--per-pz-output",
        action="append",
        default=[],
        help="Optional per-pz measurement output, formatted pzN=/path/to/out.h5. Repeat as needed.",
    )
    parser.add_argument("--nside", type=int, default=None)
    parser.add_argument("--include-gtau", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    measure_allpz(args)


if __name__ == "__main__":
    main()
