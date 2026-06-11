"""Generate Stage-31 full-sky Abacus paste configs for all DESI pz bins."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


REPO = Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX")
SCRIPT_DIR = REPO / "notebooks/xDESI/abacus_paste"
OUTPUT_ROOT = REPO / "data/xDESI/processed/abacus_backlight/stage31_fullsky_logMgt13p8"


PZ_BINS = {
    1: {
        "z_min": 0.30,
        "z_max": 0.62,
        "retained_fraction": 0.9905739901982026,
        "target_surface_density_per_deg2": 167.866089696,
        "central99_z_min": 0.29560290047792287,
        "central99_z_max": 0.6165910586223279,
    },
    2: {
        "z_min": 0.43110627652982897,
        "z_max": 0.8035931265106552,
        "retained_fraction": 0.9901655519259698,
        "target_surface_density_per_deg2": 284.753817344,
        "central99_z_min": 0.43110627652982897,
        "central99_z_max": 0.8035931265106552,
    },
    3: {
        "z_min": 0.63,
        "z_max": 0.98,
        "retained_fraction": 0.9814082135162306,
        "target_surface_density_per_deg2": 334.86400893955306,
        "central98_z_min": 0.6294281441841235,
        "central98_z_max": 0.9748499080501475,
        "stage31_map_h5_surface_density_per_deg2": 372.777282547,
    },
    4: {
        "z_min": 0.7131674616590881,
        "z_max": 1.1898555882069786,
        "retained_fraction": 0.9900608345851288,
        "target_surface_density_per_deg2": 341.297435565,
        "central99_z_min": 0.7131674616590881,
        "central99_z_max": 1.1898555882069786,
    },
}


def z_tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def catalog_key(pz: int, cfg: dict) -> str:
    return f"pz{pz}fullsky_z{z_tag(cfg['z_min'])}_{z_tag(cfg['z_max'])}_logMgt13p8"


def catalog_output_name(pz: int, cfg: dict) -> str:
    return f"abacus_c9999_ph9999_{catalog_key(pz, cfg)}_halos.h5"


def build_catalogs() -> dict:
    catalogs = {}
    for pz, cfg in PZ_BINS.items():
        key = catalog_key(pz, cfg)
        meta = {
            "desi_pz_bin": int(pz),
            "redshift_kind": "spectroscopic_calibrated_true_redshift",
            "selection_note": (
                f"pz{pz} is a photometric tracer label; full-sky halos are selected by true redshift "
                "and high mass only. No photo-z assignment is applied."
            ),
            f"pz{pz}_true_nz_retained_fraction": float(cfg["retained_fraction"]),
            "analytic_hod_log10_m_floor": 10.5,
            "resolved_catalog_log10_m_min_hmsun": 13.8,
            "fullsky_mass_cut_reason": (
                "Chosen from 2026-06-08 count and pixel-pair scans to keep the four-bin full-sky "
                "diagnostic paste close to a 30 minute total budget with current healpy settings."
            ),
        }
        for name, value in cfg.items():
            if name not in {"z_min", "z_max"}:
                meta[f"pz{pz}_true_nz_{name}"] = float(value)
        catalogs[key] = {
            "output_name": catalog_output_name(pz, cfg),
            "z_min": float(cfg["z_min"]),
            "z_max": float(cfg["z_max"]),
            "log10_m_min_hmsun": 13.8,
            "metadata": meta,
        }
    return catalogs


def base_config(default_pz: int = 1) -> dict:
    pz_cfg = PZ_BINS[int(default_pz)]
    key = catalog_key(int(default_pz), pz_cfg)
    return {
        "project": {
            "name": "stage31_fullsky_logMgt13p8",
            "xdesi_dir": str(REPO / "notebooks/xDESI"),
            "output_root": str(OUTPUT_ROOT),
            "catalog_subdir": "halos",
            "map_subdir": "maps",
            "measurement_subdir": "measurements",
            "theory_subdir": "theory",
            "plot_subdir": "plots",
        },
        "abacus": {
            "input_root": "/mnt/ceph/users/backlight/AbacusBacklight_base_c9999_ph9999/lightcone_halos",
            "sim_name": "AbacusBacklight_base_c9999_ph9999",
            "read_only": True,
            "redshift_dir_padding": 0.08,
        },
        "catalogs": build_catalogs(),
        "godmax": {
            "comparison_config": str(REPO / "param_files/xDESI/params_multiprobe_fast1024_true_nz_theory.yaml"),
            "bestfit_params": str(
                REPO
                / "notebooks/xDESI/survey_measure/outputs/"
                / "godmax_multiprobe_fast1024_true_nz_hmc_stage31_multigpu/"
                / "stage31_hmc_8000x16_v2/combined/bestfit_params_stage31_multigpu_v2.yaml"
            ),
            "measurement_h5": str(
                REPO
                / "data/xDESI/processed/multiprobe_namaster_true_nz/fast1024/"
                / "xdesi_multiprobe_cls_cov_nside1024_lmax1024_nbin10_linear.h5"
            ),
            "map_h5": str(
                REPO
                / "data/xDESI/processed/multiprobe_namaster_true_nz/fast1024/"
                / "xdesi_multiprobe_maps_nside1024_lmax1024_nbin10_linear.h5"
            ),
            "analytic_hod_log10_m_floor": 10.5,
            "resolved_catalog_log10_m_min_hmsun": 13.8,
            "override_cosmology_from_catalog": True,
        },
        "pasting": {
            "run_name": f"stage31_fullsky_pz{int(default_pz)}_logMgt13p8",
            "catalog_key": key,
            "measurement_tag_base": f"pz{int(default_pz)}_fullsky_logMgt13p8",
            "pz_bin": int(default_pz),
            "nside": 1024,
            "lmax": 1024,
            "ell_min": 8,
            "n_bins": 10,
            "binning": "linear",
            "random_seed": 42,
            "verbose": True,
            "pixel_log_batches": False,
            "pixel_backend": "healpy",
            "pixel_workers": 16,
            "pixel_start_method": "fork",
            "persistent_pixel_pool": True,
            "pixel_pool_warmup": True,
            "split_strategy": "block_striped",
            "split_block_halos": 50000,
            "jax": {"preallocate": True, "memory_fraction": 0.95, "platforms": "cuda"},
            "max_paint_R200c_factor": 5.0,
            "smooth_profiles": True,
            "profile_timing": False,
            "use_fused_profile_maps": True,
            "use_multi_kappa_maps": True,
            "return_sparse_maps": True,
            "store_projected_matter_maps": False,
            "get_baryonifiedmap": False,
            "include_legacy_pixel_arrays": False,
            "galaxy_population_chunk_size": 20000,
            "galaxy_max_gals_round_to": 16,
            "galaxy_population_backend": "padded_precomputed",
            "galaxy_compact_max_satellite_groups": 8,
            "galaxy_population_group_by_max_gals": False,
            "pixel_batch_size": 1000000,
            "pixel_gc_collect_every_n_batches": 0,
            "pixel_pool_chunksize": 32,
            "pixel_prefetch_next_chunk": False,
            "single_pixel_angle_factor": 0.5,
            "jax_clear_caches_every_n_chunks": 1,
            "compute_covariance": False,
            "include_diagnostic_gtau": True,
            "source_bins_for_galaxy_cross": [1, 2, 3, 4],
            "chunk_halos_by_nside": {128: 50000, 512: 50000, 1024: 50000, 2048: 25000},
            "num_splits_by_nside": {128: 1, 512: 4, 1024: 4, 2048: 4},
        },
        "validation": {
            "smoke_nside": 128,
            "production_nside": 1024,
            f"target_pz{int(default_pz)}_surface_density_per_deg2": float(
                pz_cfg["target_surface_density_per_deg2"]
            ),
            f"pz{int(default_pz)}_true_nz_retained_fraction": float(pz_cfg["retained_fraction"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined = base_config(default_pz=1)
    combined["pasting"]["run_name"] = "stage31_fullsky_allpz_logMgt13p8_preprocess"
    combined_path = args.out_dir / "stage31_fullsky_logMgt13p8.catalogs.yaml"
    combined_path.write_text(yaml.safe_dump(combined, sort_keys=False), encoding="utf-8")
    print(combined_path)
    for pz in range(1, 5):
        cfg = base_config(default_pz=pz)
        path = args.out_dir / f"stage31_fullsky_pz{pz}_logMgt13p8.selected.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
