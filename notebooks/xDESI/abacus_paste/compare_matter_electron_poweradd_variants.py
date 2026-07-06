#!/usr/bin/env python3
"""Scan power-add matter/electron 1h/2h variants for Abacus pasted spectra.

This is a focused diagnostic for galaxy-kappa, galaxy-shear, and galaxy-tau.
It leaves the galaxy and tSZ transitions at the requested physical baseline:

* ``gg_transition_model=poweradd``
* ``tSZ_transition_model=poweradd``
* ``alpha_gg=alpha_gy=alpha_ky=1``

This is now mostly a historical diagnostic: the main theory builder defaults
galaxy-matter/electron totals to power-add directly.  The script still scans
alternate alpha values and can compare against an older response-suppressed
baseline product.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import h5py
import numpy as np

import compare_physical_transition_variants as trans
import stage31_pz1_backlight_validation as stage31
from stage31_pz1_backlight_validation import gmt, mpn


CONFIG_DEFAULT = trans.CONFIG_DEFAULT
CURRENT_BASELINE_DEFAULT = Path(
    "data/xDESI/processed/abacus_backlight/stage31_pz3_cap600_mmin11p147538/theory/"
    "physical_transition_variants/stage31_pz3_cap600_mmin11p147538_theory_variant_baseline.h5"
)
PASTED_SIM_DEFAULT = trans.PASTED_SIM_DEFAULT
DIRECT_FIELD_SIM_DEFAULT = trans.DIRECT_FIELD_SIM_DEFAULT

FOCUS_SPECTRA = (
    "desi_g_act_kappa_pz3",
    "desi_g_des_shear_E_pz3_tomo1",
    "desi_g_des_shear_E_pz3_tomo2",
    "desi_g_des_shear_E_pz3_tomo3",
    "desi_g_des_shear_E_pz3_tomo4",
    "desi_g_tau_pz3",
)


@dataclass(frozen=True)
class MEVariant:
    variant_id: str
    label: str
    alpha_matter: Optional[float]
    alpha_electron: Optional[float]
    note: str


VARIANTS: Sequence[MEVariant] = (
    MEVariant(
        "current_response",
        "current response",
        None,
        None,
        "Current baseline matter/electron treatment: (P1h + P2h) * Pmm_response.",
    ),
    MEVariant(
        "poweradd_alpha0p75",
        "poweradd alpha=0.75",
        0.75,
        0.75,
        "Power-add galaxy-matter and galaxy-electron totals with alpha=0.75.",
    ),
    MEVariant(
        "poweradd_alpha1p00",
        "poweradd alpha=1.00",
        1.0,
        1.0,
        "Power-add galaxy-matter and galaxy-electron totals with alpha=1.0, i.e. direct 1h+2h sum without response suppression.",
    ),
    MEVariant(
        "poweradd_alpha1p25",
        "poweradd alpha=1.25",
        1.25,
        1.25,
        "Power-add galaxy-matter and galaxy-electron totals with alpha=1.25.",
    ),
)


def poweradd(p1, p2, alpha: float):
    import jax.numpy as jnp

    a = float(alpha)
    p1_min = float(np.min(np.asarray(p1)))
    p2_min = float(np.min(np.asarray(p2)))
    if p1_min < -1.0e-12 or p2_min < -1.0e-12:
        raise ValueError(f"Power-add requires non-negative inputs; mins are {p1_min:.6e}, {p2_min:.6e}.")
    p1c = jnp.clip(p1, 1.0e-60, jnp.inf)
    p2c = jnp.clip(p2, 1.0e-60, jnp.inf)
    return (p1c**a + p2c**a) ** (1.0 / a)


def build_model_with_me_poweradd(
    config: Mapping[str, object],
    *,
    is_cmb_lensing: bool,
    log10_mass_cut: Optional[float],
    variant: MEVariant,
):
    gmt.ensure_godmax_import_paths(Path(config["repo_root"]))
    import jax.numpy as jnp
    from base_class import base_class
    from get_Cls import get_Cl
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles

    sim_params, halo_params, analysis, other_params = gmt._params_for_model(config, is_cmb_lensing=is_cmb_lensing)
    analysis = dict(analysis)
    other_params = dict(other_params)
    analysis.update(trans.BASELINE_ANALYSIS_DEFAULTS)
    other_params.update(trans.BASELINE_OTHER_DEFAULTS)

    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = Profiles(sim_params, halo_params, analysis, other_params, base_class_obj=base)
    if log10_mass_cut is not None:
        mass_mask = jnp.asarray(jnp.log10(profiles.M_array) >= float(log10_mass_cut))
        profiles.Ncen_mat = profiles.Ncen_mat * mass_mask[None, :]
        profiles.Nsat_mat = profiles.Nsat_mat * mass_mask[None, :]
    pkz = get_Pkz(sim_params, halo_params, analysis, other_params, Profiles_obj=profiles)

    if variant.alpha_matter is not None:
        pkz.Pgm_tot_mat = poweradd(pkz.Pgm_1h_kz_mat, pkz.Pgm_2h_kz_mat, float(variant.alpha_matter))
        pkz.Pgm_nfw_tot_mat = poweradd(pkz.Pgm_nfw_1h_kz_mat, pkz.Pgm_nfw_2h_kz_mat, float(variant.alpha_matter))
    if variant.alpha_electron is not None:
        pkz.Pge_tot_mat = poweradd(pkz.Pge_1h_kz_mat, pkz.Pge_2h_kz_mat, float(variant.alpha_electron))
    return get_Cl(sim_params, halo_params, analysis, other_params, Pkz_obj=pkz)


def build_variant_theory(
    *,
    config_path: Path,
    config: Mapping[str, object],
    base_cfg: Mapping[str, object],
    variant: MEVariant,
    measurement_path: Path,
    output_path: Path,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path

    cfg = copy.deepcopy(dict(base_cfg))
    with h5py.File(measurement_path, "r") as h5:
        measurement_config = json.loads(str(h5.attrs["config_json"]))
        ell_band = np.asarray(h5["joint/ell"][:], dtype=np.float64)
    cfg["metadata"]["lmax"] = int(measurement_config["lmax"])
    cfg = gmt.compute_desi_nbar_comoving(cfg)
    pz_bin = stage31.pz_bin_from_config(config)
    pz_cfg = gmt.config_for_single_desi_pz(cfg, pz_bin)
    resolved_cut = float(config["godmax"].get("resolved_catalog_log10_m_min_hmsun", 11.0))

    full_wl = build_model_with_me_poweradd(pz_cfg, is_cmb_lensing=False, log10_mass_cut=None, variant=variant)
    full_cmb = build_model_with_me_poweradd(pz_cfg, is_cmb_lensing=True, log10_mass_cut=None, variant=variant)
    resolved_wl = build_model_with_me_poweradd(
        pz_cfg, is_cmb_lensing=False, log10_mass_cut=resolved_cut, variant=variant
    )
    resolved_cmb = build_model_with_me_poweradd(
        pz_cfg, is_cmb_lensing=True, log10_mass_cut=resolved_cut, variant=variant
    )

    ell_smooth = np.asarray(full_wl.ell_array, dtype=np.float64)
    full_theory = stage31.pz_theory_from_models(full_wl, full_cmb, pz_bin)
    resolved_theory = stage31.pz_theory_from_models(resolved_wl, resolved_cmb, pz_bin)
    av = np.asarray(cfg["metadata"].get("ksz_default_A_v_by_pz", np.full(4, np.nan)), dtype=np.float64)
    ksz_amplitudes = {pz_bin: float(av[pz_bin - 1])} if av.size >= pz_bin and np.isfinite(av[pz_bin - 1]) else None
    shear_m = cfg["metadata"].get("shear_m_bias_means")
    full_vec, names = mpn.theory_to_data_vector(
        measurement_path,
        full_theory,
        ell=ell_smooth,
        ksz_velocity_amplitudes=ksz_amplitudes,
        shear_m_bias=shear_m,
        theory_shear_e_is_positive_kappa=True,
        include_default_pixel_windows=True,
        include_default_act_beams=True,
    )
    resolved_vec, _ = mpn.theory_to_data_vector(
        measurement_path,
        resolved_theory,
        ell=ell_smooth,
        ksz_velocity_amplitudes=ksz_amplitudes,
        shear_m_bias=shear_m,
        theory_shear_e_is_positive_kappa=True,
        include_default_pixel_windows=True,
        include_default_act_beams=True,
    )

    write_config = copy.deepcopy(dict(config))
    write_config["matter_electron_poweradd_variant"] = {
        "variant_id": variant.variant_id,
        "label": variant.label,
        "alpha_matter": variant.alpha_matter,
        "alpha_electron": variant.alpha_electron,
        "note": variant.note,
        "config_path": str(config_path),
        "baseline_gg_tsz": {
            "analysis": trans.BASELINE_ANALYSIS_DEFAULTS,
            "other": trans.BASELINE_OTHER_DEFAULTS,
        },
    }
    stage31.write_theory_product(
        output_path,
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
    with h5py.File(output_path, "a") as h5:
        h5.attrs["matter_electron_poweradd_variant_json"] = json.dumps(
            write_config["matter_electron_poweradd_variant"], sort_keys=True
        )
    return output_path


def compare_focus_spectra(sim_payload: Mapping[str, object], theories: Mapping[str, Mapping[str, object]]) -> dict:
    out = {}
    for variant_id, theory in theories.items():
        variant_out = {}
        for name in FOCUS_SPECTRA:
            if name not in sim_payload["spectra"] or name not in theory["names"]:
                continue
            sim_cl = np.asarray(sim_payload["spectra"][name]["cl"], dtype=np.float64)
            th_cl = trans.vector_by_name(theory, name)
            variant_out[name] = trans.ratio_stats(sim_cl, th_cl)
        out[variant_id] = variant_out
    return out


def mean_focus_residual(stats_by_spectrum: Mapping[str, Mapping[str, float]]) -> float:
    vals = [float(v["median_abs_frac"]) for v in stats_by_spectrum.values() if np.isfinite(v["median_abs_frac"])]
    return float(np.mean(vals)) if vals else math.nan


def plot_focus_pdf(
    *,
    output_path: Path,
    sims: Mapping[str, Mapping[str, object]],
    theories: Mapping[str, Mapping[str, object]],
    variant_labels: Mapping[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = {
        "current_response": "#111111",
        "poweradd_alpha0p75": "#0072B2",
        "poweradd_alpha1p00": "#D55E00",
        "poweradd_alpha1p25": "#009E73",
    }
    styles = {
        "current_response": "-",
        "poweradd_alpha0p75": "--",
        "poweradd_alpha1p00": "-.",
        "poweradd_alpha1p25": ":",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for sim_label, sim in sims.items():
            fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.3), squeeze=False)
            for ax, name in zip(axes.ravel(), FOCUS_SPECTRA):
                s = sim["spectra"][name]
                ax.plot(
                    s["ell"],
                    trans.dell(s["ell"], s["cl"]),
                    "o",
                    color="#000000",
                    markerfacecolor="white",
                    ms=4.0,
                    label=sim_label,
                    zorder=5,
                )
                for variant_id, theory in theories.items():
                    ell = np.asarray(theory["ell"], dtype=np.float64)
                    cl = trans.vector_by_name(theory, name)
                    ax.plot(
                        ell,
                        trans.dell(ell, cl),
                        color=colors.get(variant_id),
                        linestyle=styles.get(variant_id, "-"),
                        lw=1.8 if variant_id == "current_response" else 1.4,
                        label=variant_labels[variant_id],
                    )
                ax.axhline(0.0, color="0.72", lw=0.8)
                ax.set_title(name, fontsize=9)
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(r"$D_\ell$")
                ax.grid(alpha=0.22)
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
            fig.suptitle(f"Matter/electron power-add variants vs {sim_label}", fontsize=13)
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.3), squeeze=False)
            for ax, name in zip(axes.ravel(), FOCUS_SPECTRA):
                s = sim["spectra"][name]
                sim_cl = np.asarray(s["cl"], dtype=np.float64)
                for variant_id, theory in theories.items():
                    th_cl = trans.vector_by_name(theory, name)
                    ratio = np.divide(sim_cl, th_cl, out=np.full_like(sim_cl, np.nan), where=th_cl != 0.0)
                    ax.plot(
                        s["ell"],
                        ratio,
                        "o",
                        ms=3.0,
                        color=colors.get(variant_id),
                        linestyle=styles.get(variant_id, "-"),
                        lw=1.2,
                        label=variant_labels[variant_id],
                    )
                ax.axhspan(0.9, 1.1, color="#cfe8cf", alpha=0.45, lw=0)
                ax.axhline(1.0, color="0.3", lw=0.9)
                ax.set_title(name, fontsize=9)
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel("sim / theory")
                ax.grid(alpha=0.22)
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
            fig.suptitle(f"Matter/electron power-add ratios vs {sim_label}", fontsize=13)
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def write_markdown(
    *,
    path: Path,
    config_path: Path,
    current_baseline: Path,
    sims: Mapping[str, Mapping[str, object]],
    theory_paths: Mapping[str, Path],
    comparisons: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    variant_map: Mapping[str, MEVariant],
    plot_path: Path,
) -> None:
    lines = []
    lines.append("# Matter/Electron Power-Add Variant Comparison")
    lines.append("")
    lines.append("This scan changes only the galaxy-matter and galaxy-electron 1h/2h combination. Galaxy and tSZ transitions are fixed to the requested power-add alpha=1 baseline.")
    lines.append("")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Current response baseline: `{current_baseline}`")
    lines.append(f"- Plot: `{plot_path}`")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| variant | alpha_matter | alpha_electron | note |")
    lines.append("|---|---:|---:|---|")
    for variant_id, variant in variant_map.items():
        am = "response" if variant.alpha_matter is None else f"{variant.alpha_matter:.2f}"
        ae = "response" if variant.alpha_electron is None else f"{variant.alpha_electron:.2f}"
        lines.append(f"| `{variant_id}` | {am} | {ae} | {variant.note} |")
    lines.append("")
    for sim_label, comparison in comparisons.items():
        lines.append(f"## {sim_label}")
        lines.append("")
        lines.append(f"Simulation product: `{sims[sim_label]['path']}`")
        lines.append("")
        lines.append("| variant | mean median abs frac over focus spectra |")
        lines.append("|---|---:|")
        for variant_id in variant_map:
            lines.append(f"| `{variant_id}` | {mean_focus_residual(comparison[variant_id]):.3f} |")
        lines.append("")
        lines.append("| spectrum | current response median ratio | best variant | best median ratio | best median abs frac | within 10% bins |")
        lines.append("|---|---:|---|---:|---:|---:|")
        for spectrum in FOCUS_SPECTRA:
            current = comparison["current_response"][spectrum]
            candidates = []
            for variant_id, stats_by_spectrum in comparison.items():
                stats = stats_by_spectrum[spectrum]
                candidates.append((float(stats["median_abs_frac"]), variant_id, stats))
            _, best_id, best = min(candidates, key=lambda item: item[0])
            lines.append(
                f"| `{spectrum}` | {current['median_ratio']:.3f} | `{best_id}` | "
                f"{best['median_ratio']:.3f} | {best['median_abs_frac']:.3f} | {best['within10']}/{best['n']} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--current-baseline", default=str(CURRENT_BASELINE_DEFAULT))
    parser.add_argument("--pasted-sim", default=str(PASTED_SIM_DEFAULT))
    parser.add_argument("--direct-field-sim", default=str(DIRECT_FIELD_SIM_DEFAULT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = stage31.read_config(config_path)
    out_root = stage31.output_dir(config, "theory_subdir")
    measure_root = stage31.output_dir(config, "measurement_subdir")
    plot_root = stage31.output_dir(config, "plot_subdir")
    variant_dir = out_root / "matter_electron_poweradd_variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    current_baseline = Path(args.current_baseline)
    pasted_sim_path = Path(args.pasted_sim)
    direct_sim_path = Path(args.direct_field_sim)
    base_cfg = stage31.merge_bestfit_params(config)
    run_name = stage31.run_name_from_config(config)
    tag = trans.measurement_tag_base(config)

    variant_map = {variant.variant_id: variant for variant in VARIANTS}
    theory_paths: Dict[str, Path] = {"current_response": current_baseline}
    for variant in VARIANTS:
        if variant.variant_id == "current_response":
            continue
        out_path = variant_dir / f"{run_name}_theory_matter_electron_{variant.variant_id}.h5"
        theory_paths[variant.variant_id] = out_path
        if not args.skip_build:
            print(f"[build] {variant.variant_id}: {out_path}", flush=True)
            build_variant_theory(
                config_path=config_path,
                config=config,
                base_cfg=base_cfg,
                variant=variant,
                measurement_path=pasted_sim_path,
                output_path=out_path,
                overwrite=bool(args.overwrite),
            )

    theories = {variant_id: trans.read_windowed_theory_vector(path) for variant_id, path in theory_paths.items()}
    sims = {
        "pasted_only": trans.read_measurement(pasted_sim_path),
        "pasted_plus_direct_field": trans.read_measurement(direct_sim_path),
    }
    comparisons = {label: compare_focus_spectra(payload, theories) for label, payload in sims.items()}

    output_json = measure_root / f"sim_theory_matter_electron_poweradd_variants_{tag}.json"
    output_md = measure_root / f"sim_theory_matter_electron_poweradd_variants_{tag}.md"
    output_pdf = plot_root / f"{run_name}_matter_electron_poweradd_variants_Dell_ratio.pdf"

    payload = {
        "config": str(config_path),
        "current_baseline": str(current_baseline),
        "theory_paths": {key: str(value) for key, value in theory_paths.items()},
        "variants": {
            key: {
                "label": value.label,
                "alpha_matter": value.alpha_matter,
                "alpha_electron": value.alpha_electron,
                "note": value.note,
            }
            for key, value in variant_map.items()
        },
        "focus_spectra": list(FOCUS_SPECTRA),
        "sims": {key: value["path"] for key, value in sims.items()},
        "comparisons": comparisons,
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(
        path=output_md,
        config_path=config_path,
        current_baseline=current_baseline,
        sims=sims,
        theory_paths=theory_paths,
        comparisons=comparisons,
        variant_map=variant_map,
        plot_path=output_pdf,
    )
    plot_focus_pdf(
        output_path=output_pdf,
        sims=sims,
        theories=theories,
        variant_labels={key: variant.label for key, variant in variant_map.items()},
    )
    print(
        json.dumps(
            {
                "json": str(output_json),
                "markdown": str(output_md),
                "plot": str(output_pdf),
                "theory_paths": {key: str(value) for key, value in theory_paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
