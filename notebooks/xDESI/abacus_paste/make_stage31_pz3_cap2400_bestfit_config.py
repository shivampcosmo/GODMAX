"""Generate the Stage-31 pz3 cap2400 bestfit Abacus paste config.

The file produced here is intentionally an unselected cap config: run
``stage31_pz1_backlight_validation.py select-cap`` afterward to choose the
actual cap center from the common data mask.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


REPO = Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX")
SCRIPT_DIR = REPO / "notebooks/xDESI/abacus_paste"
OUTPUT_ROOT = REPO / "data/xDESI/processed/abacus_backlight"

DEFAULT_BASE_CONFIG = SCRIPT_DIR / "stage31_pz3_cap4800_mmin11p147538_nside2048_lmax4096.yaml"
DEFAULT_OUTPUT = SCRIPT_DIR / "stage31_pz3_cap2400_hmcbestfit_mmin11p147538_nside2048_lmax4096.yaml"
DEFAULT_COMPARISON_CONFIG = (
    REPO / "param_files/xDESI/params_multiprobe_midres2048_true_nz_theory_abacus_cosmo_simple1h2h.yaml"
)
DEFAULT_HMC_RUN = (
    REPO
    / "notebooks/xDESI/survey_measure/outputs/"
    / "godmax_multiprobe_midres2048_true_nz_hmc_stage31_multigpu/"
    / "stage31_hmc_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_1600x16_v1/"
    / "combined"
)
DEFAULT_BESTFIT_PARAMS = (
    DEFAULT_HMC_RUN
    / "bestfit_params_stage31_multigpu_abacus_cosmo_midres2048_simple1h2h_lmax4096_gk1024_mmin11p147538_1600x16_v1.yaml"
)
DEFAULT_MEASUREMENT_H5 = (
    REPO
    / "data/xDESI/processed/multiprobe_namaster_true_nz/midres2048/"
    / "xdesi_multiprobe_cls_cov_nside2048_lmax4096_nbin10_linear.h5"
)
DEFAULT_MAP_H5 = (
    REPO
    / "data/xDESI/processed/multiprobe_namaster_true_nz/midres2048/"
    / "xdesi_multiprobe_maps_nside2048_lmax4096_nbin10_linear.h5"
)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not contain a YAML mapping.")
    return data


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def build_config(args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(read_yaml(Path(args.base_config)))
    area_tag = f"cap{int(round(float(args.area_deg2)))}"
    project_name = args.project_name or f"stage31_pz3_{area_tag}_hmcbestfit_mmin11p147538"
    catalog_key = args.catalog_key or f"pz3{area_tag}_hmcbestfit_z0p63_0p98_logMgt11p147538"
    output_root = OUTPUT_ROOT / project_name
    run_name = f"{project_name}_nside{int(args.nside)}_lmax{int(args.lmax)}"
    measurement_tag_base = f"pz3_{area_tag}_hmcbestfit_mmin11p147538"

    cfg["project"]["name"] = project_name
    cfg["project"]["output_root"] = str(output_root)

    old_catalogs = cfg.get("catalogs", {})
    if len(old_catalogs) != 1:
        raise ValueError(f"Expected one source catalog in {args.base_config}, found {list(old_catalogs)}")
    old_spec = copy.deepcopy(next(iter(old_catalogs.values())))
    old_spec["output_name"] = f"abacus_c9999_ph9999_{catalog_key}_halos.h5"
    metadata = old_spec.setdefault("metadata", {})
    metadata["mass_selection_note"] = (
        "Direct Abacus Backlight cap catalog for the Stage-31 midres2048 HMC bestfit "
        "Abacus-cosmology mmin11p147538 comparison."
    )
    cfg["catalogs"] = {catalog_key: old_spec}

    sky = cfg["sky_patch"]
    sky["area_deg2"] = float(args.area_deg2)
    sky["center_ra_deg"] = None
    sky["center_dec_deg"] = None
    sky["radius_deg"] = None
    for key in (
        "selected_common_area_deg2",
        "selected_common_fraction",
        "selected_actual_cap_area_deg2",
    ):
        sky.pop(key, None)

    godmax = cfg["godmax"]
    godmax["comparison_config"] = str(Path(args.comparison_config))
    godmax["bestfit_params"] = str(Path(args.bestfit_params))
    godmax["measurement_h5"] = str(Path(args.measurement_h5))
    godmax["map_h5"] = str(Path(args.map_h5))

    pasting = cfg["pasting"]
    pasting["run_name"] = run_name
    pasting["catalog_key"] = catalog_key
    pasting["measurement_tag_base"] = measurement_tag_base
    pasting["nside"] = int(args.nside)
    pasting["lmax"] = int(args.lmax)
    pasting["num_splits_by_nside"][int(args.nside)] = int(args.num_splits)
    pasting["jax"]["compilation_cache_dir"] = str(output_root / f"jax_cache/nside{args.nside}_lmax{args.lmax}")

    validation = cfg.setdefault("validation", {})
    validation["production_nside"] = int(args.nside)
    validation["cap_candidate_nside"] = int(args.cap_candidate_nside)
    validation["cap_refine_top_n"] = int(args.cap_refine_top_n)

    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--catalog-key", default=None)
    parser.add_argument("--area-deg2", type=float, default=2400.0)
    parser.add_argument("--nside", type=int, default=2048)
    parser.add_argument("--lmax", type=int, default=4096)
    parser.add_argument("--num-splits", type=int, default=4)
    parser.add_argument("--cap-candidate-nside", type=int, default=64)
    parser.add_argument("--cap-refine-top-n", type=int, default=256)
    parser.add_argument("--comparison-config", default=str(DEFAULT_COMPARISON_CONFIG))
    parser.add_argument("--bestfit-params", default=str(DEFAULT_BESTFIT_PARAMS))
    parser.add_argument("--measurement-h5", default=str(DEFAULT_MEASUREMENT_H5))
    parser.add_argument("--map-h5", default=str(DEFAULT_MAP_H5))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    cfg = build_config(args)
    write_yaml(output, cfg)
    print(
        json.dumps(
            {
                "output": str(output),
                "project": cfg["project"]["name"],
                "catalog_key": cfg["pasting"]["catalog_key"],
                "bestfit_params": cfg["godmax"]["bestfit_params"],
                "comparison_config": cfg["godmax"]["comparison_config"],
                "measurement_h5": cfg["godmax"]["measurement_h5"],
                "map_h5": cfg["godmax"]["map_h5"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
