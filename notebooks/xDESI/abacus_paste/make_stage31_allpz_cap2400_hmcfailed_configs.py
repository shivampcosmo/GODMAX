"""Generate Stage-31 all-pz cap2400 hmcfailed Abacus paste configs.

The all-pz diagnostic uses one shared continuous-field paste over
0 < z < 1.2, plus pz-specific galaxy-only pastes.  The existing pz3 cap2400
hmcfailed galaxy paste is reused and is therefore not regenerated here.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Mapping

import yaml


REPO = Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX")
SCRIPT_DIR = REPO / "notebooks/xDESI/abacus_paste"
OUTPUT_ROOT = REPO / "data/xDESI/processed/abacus_backlight/stage31_allpz_cap2400_hmcfailed_mmin11p147538"
BASE_CONFIG = SCRIPT_DIR / "stage31_pz3_cap2400_hmcfailed_mmin11p147538_nside2048_lmax4096.selected.yaml"

MMIN = 11.147538
NSIDE = 2048
LMAX = 4096

PZ_SPECS = {
    1: {
        "z_min": 0.30,
        "z_max": 0.62,
        "z_tag": "z0p30_0p62",
        "target_surface_density_per_deg2": 167.866089696,
        "retained_fraction": 0.9905739901982026,
        "central_z_min": 0.29560290047792287,
        "central_z_max": 0.6165910586223279,
        "central_label": "central99",
    },
    2: {
        "z_min": 0.43110627652982897,
        "z_max": 0.8035931265106552,
        "z_tag": "z0p431_0p804",
        "target_surface_density_per_deg2": 284.753817344,
        "retained_fraction": 0.9901655519259698,
        "central_z_min": 0.43110627652982897,
        "central_z_max": 0.8035931265106552,
        "central_label": "central99",
    },
    4: {
        "z_min": 0.7131674616590881,
        "z_max": 1.1898555882069786,
        "z_tag": "z0p713_1p19",
        "target_surface_density_per_deg2": 341.297435565,
        "retained_fraction": 0.9900608345851288,
        "central_z_min": 0.7131674616590881,
        "central_z_max": 1.1898555882069786,
        "central_label": "central99",
    },
}


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a YAML mapping.")
    return data


def write_yaml(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def _common_project(cfg: dict) -> None:
    cfg["project"]["name"] = "stage31_allpz_cap2400_hmcfailed_mmin11p147538"
    cfg["project"]["output_root"] = str(OUTPUT_ROOT)
    cfg.setdefault("diagnostics", {})["allpz_config_note"] = (
        "Generated for all-pz cap2400 hmcfailed diagnostic.  pz3 galaxy map is reused "
        "from the existing stage31_pz3_cap2400_hmcfailed_mmin11p147538 product."
    )


def _common_pasting(cfg: dict, tag: str, catalog_key: str, *, num_splits: int) -> None:
    pasting = cfg["pasting"]
    pasting["run_name"] = f"stage31_{tag}_nside{NSIDE}_lmax{LMAX}"
    pasting["catalog_key"] = catalog_key
    pasting["measurement_tag_base"] = tag
    pasting["nside"] = NSIDE
    pasting["lmax"] = LMAX
    pasting["sim_measurement_mask_mode"] = "cap"
    pasting["ksz_velocity_mode"] = "photoz_reconstruction_emulation"
    pasting["ksz_reconstruction_noise_seed"] = 12345
    pasting.setdefault("jax", {})["compilation_cache_dir"] = str(OUTPUT_ROOT / f"jax_cache/{tag}_nside{NSIDE}_lmax{LMAX}")
    pasting.setdefault("num_splits_by_nside", {})[NSIDE] = int(num_splits)
    pasting.setdefault("chunk_halos_by_nside", {})[NSIDE] = int(pasting.get("chunk_halos_by_nside", {}).get(2048, 500000))
    pasting["pixel_workers"] = 16
    pasting["split_block_halos"] = 250000
    pasting["include_diagnostic_gtau"] = True
    pasting["source_bins_for_galaxy_cross"] = [1, 2, 3, 4]


def build_allz_config(base: Mapping[str, object]) -> dict:
    cfg = copy.deepcopy(dict(base))
    _common_project(cfg)
    catalog_key = "allzcap2400_hmcfailed_z0p001_1p2_logMgt11p147538"
    _common_pasting(cfg, "allz_cap2400_hmcfailed_mmin11p147538", catalog_key, num_splits=16)
    cfg["catalogs"] = {
        catalog_key: {
            "output_name": f"abacus_c9999_ph9999_{catalog_key}_halos.h5",
            "z_min": 0.001,
            "z_max": 1.2,
            "log10_m_min_hmsun": MMIN,
            "metadata": {
                "redshift_kind": "spectroscopic_calibrated_true_redshift",
                "selection_note": "Shared all-pz continuous-field paste; galaxies are disabled for this config.",
                "analytic_hod_log10_m_floor": MMIN,
                "resolved_catalog_log10_m_min_hmsun": MMIN,
                "diagnostic_note": "Non-converged HMC bestfit diagnostic cap2400 paste; do not use as final converged product.",
            },
        }
    }
    cfg["sky_patch"]["selection_fields"] = ["g1", "pi1", "g2", "pi2", "g3", "pi3", "g4", "pi4", "y", "T", "kappa", "s1", "s2", "s3", "s4"]
    pasting = cfg["pasting"]
    pasting["pz_bin"] = 3
    pasting["get_galmap"] = False
    pasting["get_ymap"] = True
    pasting["get_kszmap"] = True
    pasting["get_taumap"] = True
    pasting["get_kappa_wl"] = True
    pasting["get_kappa_cmb"] = True
    pasting["use_multi_kappa_maps"] = True
    cfg["validation"]["production_nside"] = NSIDE
    cfg["validation"]["resolved_catalog_log10_m_min_hmsun"] = MMIN
    return cfg


def build_pz_galaxy_config(base: Mapping[str, object], pz_bin: int, spec: Mapping[str, object]) -> dict:
    cfg = copy.deepcopy(dict(base))
    _common_project(cfg)
    catalog_key = f"pz{pz_bin}cap2400_hmcfailed_{spec['z_tag']}_logMgt11p147538"
    tag = f"pz{pz_bin}_cap2400_hmcfailed_mmin11p147538"
    _common_pasting(cfg, tag, catalog_key, num_splits=4)
    central_label = str(spec["central_label"])
    metadata = {
        "desi_pz_bin": int(pz_bin),
        "redshift_kind": "spectroscopic_calibrated_true_redshift",
        "selection_note": (
            f"pz{pz_bin} is a photometric tracer label; halos are selected by true redshift only. "
            "Continuous fields are disabled so this config only supplies the tomographic galaxy catalog."
        ),
        f"pz{pz_bin}_true_nz_retained_fraction": float(spec["retained_fraction"]),
        f"pz{pz_bin}_true_nz_target_surface_density_per_deg2": float(spec["target_surface_density_per_deg2"]),
        f"pz{pz_bin}_true_nz_{central_label}_z_min": float(spec["central_z_min"]),
        f"pz{pz_bin}_true_nz_{central_label}_z_max": float(spec["central_z_max"]),
        "analytic_hod_log10_m_floor": MMIN,
        "resolved_catalog_log10_m_min_hmsun": MMIN,
        "mass_selection_note": "Direct Abacus Backlight cap catalog for all-pz cap2400 hmcfailed diagnostic.",
        "diagnostic_note": "Non-converged HMC bestfit diagnostic cap2400 paste; do not use as final converged product.",
    }
    cfg["catalogs"] = {
        catalog_key: {
            "output_name": f"abacus_c9999_ph9999_{catalog_key}_halos.h5",
            "z_min": float(spec["z_min"]),
            "z_max": float(spec["z_max"]),
            "log10_m_min_hmsun": MMIN,
            "metadata": metadata,
        }
    }
    cfg["sky_patch"]["selection_fields"] = [f"g{pz_bin}", f"pi{pz_bin}", "y", "T", "kappa", "s1", "s2", "s3", "s4"]
    pasting = cfg["pasting"]
    pasting["pz_bin"] = int(pz_bin)
    pasting["get_galmap"] = True
    pasting["get_ymap"] = False
    pasting["get_kszmap"] = False
    pasting["get_taumap"] = False
    pasting["get_kappa_wl"] = False
    pasting["get_kappa_cmb"] = False
    pasting["use_multi_kappa_maps"] = False
    cfg["validation"]["production_nside"] = NSIDE
    cfg["validation"][f"target_pz{pz_bin}_surface_density_per_deg2"] = float(spec["target_surface_density_per_deg2"])
    cfg["validation"][f"pz{pz_bin}_true_nz_retained_fraction"] = float(spec["retained_fraction"])
    cfg["validation"]["resolved_catalog_log10_m_min_hmsun"] = MMIN
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = read_yaml(Path(args.base_config))
    out_dir = Path(args.output_dir)
    products = {}

    allz_cfg = build_allz_config(base)
    allz_path = out_dir / "stage31_allz_cap2400_hmcfailed_mmin11p147538_nside2048_lmax4096.selected.yaml"
    write_yaml(allz_path, allz_cfg)
    products["allz"] = str(allz_path)

    for pz_bin, spec in PZ_SPECS.items():
        cfg = build_pz_galaxy_config(base, pz_bin, spec)
        path = out_dir / f"stage31_pz{pz_bin}_cap2400_hmcfailed_mmin11p147538_nside2048_lmax4096.selected.yaml"
        write_yaml(path, cfg)
        products[f"pz{pz_bin}"] = str(path)

    manifest = {
        "schema": "stage31_allpz_cap2400_hmcfailed_config_manifest_v1",
        "base_config": str(Path(args.base_config).resolve()),
        "output_root": str(OUTPUT_ROOT),
        "configs": products,
        "reused_pz3_config": str(BASE_CONFIG),
        "reused_pz3_note": "pz3 galaxy map is reused from the existing hmcfailed cap2400 product.",
    }
    manifest_path = out_dir / "stage31_allpz_cap2400_hmcfailed_mmin11p147538_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
