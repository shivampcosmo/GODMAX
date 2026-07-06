#!/usr/bin/env python3
"""Compare physical 1h/2h transition variants to Abacus pasted spectra.

This script is intentionally limited to physical knobs already exposed by the
active GODMAX theory path:

* ``gg_transition_model``
* ``tSZ_transition_model``
* ``galaxy_matter_transition_model``
* ``galaxy_electron_transition_model``
* ``alpha_gg``
* ``alpha_gy``
* ``alpha_ky``

It does not fit residual transfers, per-spectrum amplitudes, or component
rescalings.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np

import stage31_pz1_backlight_validation as stage31
from stage31_pz1_backlight_validation import gmt, mpn


CONFIG_DEFAULT = Path("notebooks/xDESI/abacus_paste/stage31_pz3_cap600_mmin11p147538.selected.yaml")
BASELINE_THEORY_DEFAULT = Path(
    "data/xDESI/processed/abacus_backlight/stage31_pz3_cap600_mmin11p147538/theory/"
    "stage31_pz3_cap600_mmin11p147538_theory_for_sim_measurement_no_pgg_clip.h5"
)
PASTED_SIM_DEFAULT = Path(
    "data/xDESI/processed/abacus_backlight/stage31_pz3_cap600_mmin11p147538/measurements/"
    "sim_pz3_cap600_mmin11p147538_nside1024_lmax1024_nbin10_linear.h5"
)
DIRECT_FIELD_SIM_DEFAULT = Path(
    "data/xDESI/processed/abacus_backlight/stage31_pz3_cap600_mmin11p147538/measurements/"
    "sim_pz3_cap600_mmin11p147538_plus_direct_field_nside1024_lmax1024_nbin10_linear.h5"
)


@dataclass(frozen=True)
class TheoryVariant:
    variant_id: str
    label: str
    analysis_updates: Mapping[str, object]
    other_updates: Mapping[str, object]
    note: str


BASELINE_ANALYSIS_DEFAULTS = {
    "gg_transition_model": "poweradd",
    "tSZ_transition_model": "poweradd",
    "galaxy_matter_transition_model": "poweradd",
    "galaxy_electron_transition_model": "poweradd",
}
BASELINE_OTHER_DEFAULTS = {
    "alpha_gg": 1.0,
    "alpha_gy": 1.0,
    "alpha_ky": 1.0,
}


VARIANTS: Sequence[TheoryVariant] = (
    TheoryVariant(
        "baseline",
        "baseline: gg poweradd, tSZ poweradd, alpha=1",
        {},
        {},
        "Requested physical baseline: power-add 1h/2h combination for galaxies and tSZ with alpha_gg=alpha_gy=alpha_ky=1.",
    ),
    TheoryVariant(
        "tsz_response_alpha1",
        "previous default: tSZ response alpha=1",
        {"tSZ_transition_model": "response"},
        {},
        "Previous no-Pgg-clip reference product: y-related transitions multiplied by the matter response factor.",
    ),
    TheoryVariant(
        "gg_response",
        "gg response",
        {"gg_transition_model": "response"},
        {},
        "Galaxy auto uses (P1h + P2h) multiplied by the matter response factor.",
    ),
    TheoryVariant(
        "gg_alpha2",
        "gg poweradd alpha_gg=2",
        {"gg_transition_model": "poweradd"},
        {"alpha_gg": 2.0},
        "Galaxy auto uses a p-norm smooth transition that reduces double-counting near 1h/2h equality.",
    ),
    TheoryVariant(
        "gg_alpha4",
        "gg poweradd alpha_gg=4",
        {"gg_transition_model": "poweradd"},
        {"alpha_gg": 4.0},
        "Sharper galaxy-auto p-norm transition, approaching max(P1h,P2h).",
    ),
    TheoryVariant(
        "tsz_response_alpha2",
        "tSZ response alpha_gy=alpha_ky=2",
        {"tSZ_transition_model": "response"},
        {"alpha_gy": 2.0, "alpha_ky": 2.0},
        "Keeps response suppression but smooths y-related 1h/2h addition with alpha=2.",
    ),
    TheoryVariant(
        "tsz_poweradd_alpha2",
        "tSZ poweradd alpha_gy=alpha_ky=2",
        {"tSZ_transition_model": "poweradd"},
        {"alpha_gy": 2.0, "alpha_ky": 2.0},
        "Combines power-add tSZ transition with alpha=2 p-norm smoothing.",
    ),
)


def decode_names(values: Sequence[object]) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def read_measurement(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        names = decode_names(h5["joint/spectrum_names"][:])
        starts = np.asarray(h5["joint/slice_start"][:], dtype=int)
        stops = np.asarray(h5["joint/slice_stop"][:], dtype=int)
        vector = np.asarray(h5["joint/data_vector"][:], dtype=np.float64)
        spectra = {}
        for name, start, stop in zip(names, starts, stops):
            group = h5[f"spectra/{name}"]
            spectra[name] = {
                "ell": np.asarray(group["ell"][:], dtype=np.float64),
                "cl": np.asarray(group["cl"][:], dtype=np.float64),
                "err": np.asarray(group["err"][:], dtype=np.float64) if "err" in group else None,
                "slice": (int(start), int(stop)),
                "label": str(group.attrs.get("label", name)),
            }
        return {
            "path": str(path),
            "names": names,
            "starts": starts,
            "stops": stops,
            "vector": vector,
            "spectra": spectra,
            "config_json": json.loads(h5.attrs["config_json"]),
        }


def read_windowed_theory_vector(path: Path) -> dict:
    with h5py.File(path, "r") as h5:
        names = decode_names(h5["windowed/spectrum_names"][:])
        ell = np.asarray(h5["windowed/ell"][:], dtype=np.float64)
        vector = np.asarray(h5["windowed/full_hod_floor10p5"][:], dtype=np.float64)
        resolved = np.asarray(h5["windowed/resolved_log10Mgt11"][:], dtype=np.float64)
        attrs = {key: h5.attrs[key] for key in h5.attrs}
    return {"path": str(path), "names": names, "ell": ell, "vector": vector, "resolved": resolved, "attrs": attrs}


def measurement_tag_base(config: Mapping[str, object]) -> str:
    return str(config.get("pasting", {}).get("measurement_tag_base", stage31.run_name_from_config(config)))


def apply_variant_updates(cfg: dict, variant: TheoryVariant) -> dict:
    out = copy.deepcopy(cfg)
    analysis = out["params"]["analysis"]
    other = out["params"]["other_params"]
    analysis.update(BASELINE_ANALYSIS_DEFAULTS)
    other.update(BASELINE_OTHER_DEFAULTS)
    for key, value in variant.analysis_updates.items():
        analysis[key] = value
    for key, value in variant.other_updates.items():
        other[key] = value
    return out


def build_variant_theory(
    *,
    config_path: Path,
    config: Mapping[str, object],
    base_cfg: Mapping[str, object],
    variant: TheoryVariant,
    measurement_path: Path,
    output_path: Path,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path

    cfg = apply_variant_updates(dict(base_cfg), variant)
    with h5py.File(measurement_path, "r") as h5:
        measurement_config = json.loads(str(h5.attrs["config_json"]))
        ell_band = np.asarray(h5["joint/ell"][:], dtype=np.float64)
    cfg["metadata"]["lmax"] = int(measurement_config["lmax"])
    cfg = gmt.compute_desi_nbar_comoving(cfg)
    pz_bin = stage31.pz_bin_from_config(config)
    pz_cfg = gmt.config_for_single_desi_pz(cfg, pz_bin)
    resolved_cut = float(config["godmax"].get("resolved_catalog_log10_m_min_hmsun", 11.0))

    full_wl = stage31.build_one_godmax_model(pz_cfg, is_cmb_lensing=False)
    full_cmb = stage31.build_one_godmax_model(pz_cfg, is_cmb_lensing=True)
    resolved_wl = stage31.build_one_godmax_model(pz_cfg, is_cmb_lensing=False, log10_mass_cut=resolved_cut)
    resolved_cmb = stage31.build_one_godmax_model(pz_cfg, is_cmb_lensing=True, log10_mass_cut=resolved_cut)

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
    write_config["physical_transition_variant"] = {
        "variant_id": variant.variant_id,
        "label": variant.label,
        "analysis_updates": dict(variant.analysis_updates),
        "other_updates": dict(variant.other_updates),
        "note": variant.note,
        "config_path": str(config_path),
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
        h5.attrs["physical_transition_variant_json"] = json.dumps(write_config["physical_transition_variant"], sort_keys=True)
    return output_path


def vector_by_name(payload: Mapping[str, object], name: str) -> np.ndarray:
    names = list(payload["names"])
    ell = np.asarray(payload["ell"], dtype=np.float64)
    n = len(ell)
    i = names.index(name)
    return np.asarray(payload["vector"][i * n : (i + 1) * n], dtype=np.float64)


def ratio_stats(sim: np.ndarray, theory: np.ndarray) -> dict:
    sim = np.asarray(sim, dtype=np.float64)
    theory = np.asarray(theory, dtype=np.float64)
    good = np.isfinite(sim) & np.isfinite(theory) & (np.abs(theory) > 0.0)
    if not np.any(good):
        return {
            "n": 0,
            "same_sign": 0,
            "within10": 0,
            "median_ratio": math.nan,
            "mean_ratio": math.nan,
            "min_ratio": math.nan,
            "max_ratio": math.nan,
            "median_abs_frac": math.nan,
            "max_abs_frac": math.nan,
        }
    ratio = sim[good] / theory[good]
    abs_frac = np.abs(ratio - 1.0)
    return {
        "n": int(ratio.size),
        "same_sign": int(np.count_nonzero(np.signbit(sim[good]) == np.signbit(theory[good]))),
        "within10": int(np.count_nonzero(abs_frac <= 0.10)),
        "median_ratio": float(np.nanmedian(ratio)),
        "mean_ratio": float(np.nanmean(ratio)),
        "min_ratio": float(np.nanmin(ratio)),
        "max_ratio": float(np.nanmax(ratio)),
        "median_abs_frac": float(np.nanmedian(abs_frac)),
        "max_abs_frac": float(np.nanmax(abs_frac)),
    }


def compare_sim_to_variants(sim_payload: Mapping[str, object], theories: Mapping[str, Mapping[str, object]]) -> dict:
    out = {}
    names = list(sim_payload["names"])
    for variant_id, theory in theories.items():
        variant_out = {}
        for name in names:
            if name not in theory["names"]:
                continue
            sim_cl = np.asarray(sim_payload["spectra"][name]["cl"], dtype=np.float64)
            th_cl = vector_by_name(theory, name)
            variant_out[name] = ratio_stats(sim_cl, th_cl)
        out[variant_id] = variant_out
    return out


def clean_mean_abs_frac(stats_by_spectrum: Mapping[str, Mapping[str, float]], *, include_ksz: bool) -> float:
    vals = []
    for name, stats in stats_by_spectrum.items():
        if not include_ksz and name.startswith("desi_pi_act_T"):
            continue
        value = float(stats.get("median_abs_frac", math.nan))
        if np.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else math.nan


def best_variant_by_spectrum(comparison: Mapping[str, Mapping[str, Mapping[str, float]]], spectrum: str) -> dict:
    candidates = []
    for variant_id, stats_by_spec in comparison.items():
        if spectrum in stats_by_spec:
            stats = stats_by_spec[spectrum]
            val = float(stats["median_abs_frac"])
            if np.isfinite(val):
                candidates.append((val, variant_id, stats))
    if not candidates:
        return {}
    candidates.sort(key=lambda item: item[0])
    val, variant_id, stats = candidates[0]
    baseline = next((item for item in candidates if item[1] == "baseline"), None)
    if baseline is not None and abs(float(baseline[0]) - float(val)) <= 1.0e-10:
        val, variant_id, stats = baseline
    return {"variant_id": variant_id, **stats}


def dell(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64)
    return ell * (ell + 1.0) * np.asarray(cl, dtype=np.float64) / (2.0 * math.pi)


def plot_variants(
    *,
    output_path: Path,
    sims: Mapping[str, Mapping[str, object]],
    theories: Mapping[str, Mapping[str, object]],
    variant_labels: Mapping[str, str],
    spectra: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = {
        "baseline": "#111111",
        "gg_response": "#0072B2",
        "gg_alpha2": "#56B4E9",
        "gg_alpha4": "#009E73",
        "tsz_response_alpha1": "#D55E00",
        "tsz_response_alpha2": "#CC79A7",
        "tsz_poweradd_alpha2": "#E69F00",
    }
    linestyles = {
        "baseline": "-",
        "gg_response": "--",
        "gg_alpha2": "-.",
        "gg_alpha4": ":",
        "tsz_response_alpha1": "--",
        "tsz_response_alpha2": "-.",
        "tsz_poweradd_alpha2": ":",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for sim_label, sim in sims.items():
            ncols = 3
            nrows = int(math.ceil(len(spectra) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 3.25 * nrows), squeeze=False)
            for ax, name in zip(axes.ravel(), spectra):
                if name not in sim["spectra"]:
                    ax.axis("off")
                    continue
                s = sim["spectra"][name]
                ax.plot(
                    s["ell"],
                    dell(s["ell"], s["cl"]),
                    "o",
                    color="#000000",
                    markerfacecolor="white",
                    ms=4.0,
                    label=sim_label,
                    zorder=5,
                )
                for variant_id, theory in theories.items():
                    if name not in theory["names"]:
                        continue
                    ell = np.asarray(theory["ell"], dtype=np.float64)
                    y = dell(ell, vector_by_name(theory, name))
                    ax.plot(
                        ell,
                        y,
                        linestyle=linestyles.get(variant_id, "-"),
                        color=colors.get(variant_id, None),
                        lw=1.35 if variant_id != "baseline" else 1.9,
                        alpha=0.9,
                        label=variant_labels[variant_id],
                    )
                ax.axhline(0.0, color="0.72", lw=0.8)
                ax.set_title(name, fontsize=9)
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(r"$D_\ell$")
                ax.grid(alpha=0.22)
            for ax in axes.ravel()[len(spectra) :]:
                ax.axis("off")
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
            fig.suptitle(f"Physical 1h/2h transition variants vs {sim_label}", fontsize=13)
            fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def write_markdown(
    *,
    path: Path,
    config_path: Path,
    baseline_theory: Path,
    sims: Mapping[str, Mapping[str, object]],
    theory_paths: Mapping[str, Path],
    comparisons: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    variant_map: Mapping[str, TheoryVariant],
    plot_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Physical Transition Variant Comparison")
    lines.append("")
    lines.append("This scan uses only physical 1h/2h transition controls already exposed by the active GODMAX theory path. It does not fit residual transfers, per-spectrum amplitudes, or map-component rescalings.")
    lines.append("")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Requested baseline: `gg_transition_model=poweradd`, `tSZ_transition_model=poweradd`, `alpha_gg=alpha_gy=alpha_ky=1`")
    lines.append(f"- Previous tSZ-response reference: `{baseline_theory}`")
    lines.append(f"- Plot: `{plot_path}`")
    lines.append("")
    lines.append("## Variants")
    lines.append("")
    lines.append("| variant | analysis updates | other updates | note |")
    lines.append("|---|---|---|---|")
    for variant_id, variant in variant_map.items():
        lines.append(
            f"| `{variant_id}` | `{json.dumps(dict(variant.analysis_updates), sort_keys=True)}` | "
            f"`{json.dumps(dict(variant.other_updates), sort_keys=True)}` | {variant.note} |"
        )
    lines.append("")
    for sim_label, comparison in comparisons.items():
        lines.append(f"## {sim_label}")
        lines.append("")
        lines.append(f"Simulation product: `{sims[sim_label]['path']}`")
        lines.append("")
        lines.append("| variant | mean median abs frac, clean spectra | mean median abs frac, incl. kSZ |")
        lines.append("|---|---:|---:|")
        for variant_id in variant_map:
            clean = clean_mean_abs_frac(comparison[variant_id], include_ksz=False)
            allv = clean_mean_abs_frac(comparison[variant_id], include_ksz=True)
            lines.append(f"| `{variant_id}` | {clean:.3f} | {allv:.3f} |")
        lines.append("")
        spectra = list(next(iter(comparison.values())).keys())
        lines.append("| spectrum | baseline median ratio | baseline median abs frac | best physical theory variant | best median ratio | best median abs frac | within 10% bins |")
        lines.append("|---|---:|---:|---|---:|---:|---:|")
        for spectrum in spectra:
            base = comparison["baseline"][spectrum]
            best = best_variant_by_spectrum(comparison, spectrum)
            lines.append(
                f"| `{spectrum}` | {base['median_ratio']:.3f} | {base['median_abs_frac']:.3f} | "
                f"`{best['variant_id']}` | {best['median_ratio']:.3f} | {best['median_abs_frac']:.3f} | "
                f"{best['within10']}/{best['n']} |"
            )
        lines.append("")
    lines.append("## Physical Readout")
    lines.append("")
    lines.append("- `gg_transition_model` and `alpha_gg` only move `desi_g_auto_pz3` in this saved comparison set.")
    lines.append("- `tSZ_transition_model`, `alpha_gy`, and `alpha_ky` only move y-related theory here; the saved pz3 measurement contains `desi_g_act_y_pz3` but not y autos, y-kappa, or y-shear.")
    lines.append("- The active matter/electron path for `desi_g_act_kappa_pz3`, DES shear, and `desi_g_tau_pz3` now defaults to direct power-add via `galaxy_matter_transition_model=poweradd` and `galaxy_electron_transition_model=poweradd`; set those keys to `response` to recover the previous response-suppressed convention.")
    lines.append("- `desi_pi_act_T_pz3` remains a convention/noise diagnostic because the simulation uses true host/field velocities while the theory vector applies the Stage-31 photo-z velocity reconstruction amplitude.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DEFAULT))
    parser.add_argument("--baseline-theory", default=str(BASELINE_THEORY_DEFAULT))
    parser.add_argument("--pasted-sim", default=str(PASTED_SIM_DEFAULT))
    parser.add_argument("--direct-field-sim", default=str(DIRECT_FIELD_SIM_DEFAULT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="Only read already-built variant HDF5 products.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = stage31.read_config(config_path)
    out_root = stage31.output_dir(config, "theory_subdir")
    measure_root = stage31.output_dir(config, "measurement_subdir")
    plot_root = stage31.output_dir(config, "plot_subdir")
    variant_dir = out_root / "physical_transition_variants"
    variant_dir.mkdir(parents=True, exist_ok=True)

    pasted_sim_path = Path(args.pasted_sim)
    direct_sim_path = Path(args.direct_field_sim)
    measurement_for_windows = pasted_sim_path
    baseline_theory = Path(args.baseline_theory)
    run_name = stage31.run_name_from_config(config)
    base_cfg = stage31.merge_bestfit_params(config)

    variant_map = {variant.variant_id: variant for variant in VARIANTS}
    theory_paths: Dict[str, Path] = {}
    for variant in VARIANTS:
        if variant.variant_id == "tsz_response_alpha1":
            theory_paths[variant.variant_id] = baseline_theory
            continue
        out_path = variant_dir / f"{run_name}_theory_variant_{variant.variant_id}.h5"
        theory_paths[variant.variant_id] = out_path
        if not args.skip_build:
            print(f"[build] {variant.variant_id}: {out_path}", flush=True)
            build_variant_theory(
                config_path=config_path,
                config=config,
                base_cfg=base_cfg,
                variant=variant,
                measurement_path=measurement_for_windows,
                output_path=out_path,
                overwrite=bool(args.overwrite),
            )

    theories = {variant_id: read_windowed_theory_vector(path) for variant_id, path in theory_paths.items()}
    sims = {
        "pasted_only": read_measurement(pasted_sim_path),
        "pasted_plus_direct_field": read_measurement(direct_sim_path),
    }
    spectra = [name for name in sims["pasted_only"]["names"] if name in theories["baseline"]["names"]]
    comparisons = {label: compare_sim_to_variants(payload, theories) for label, payload in sims.items()}

    tag = measurement_tag_base(config)
    output_json = measure_root / f"sim_theory_physical_transition_variants_{tag}.json"
    output_md = measure_root / f"sim_theory_physical_transition_variants_{tag}.md"
    output_pdf = plot_root / f"{run_name}_physical_transition_variants_Dell.pdf"

    payload = {
        "config": str(config_path),
        "baseline_theory": str(baseline_theory),
        "theory_paths": {key: str(value) for key, value in theory_paths.items()},
        "variants": {
            key: {
                "label": value.label,
                "analysis_updates": dict(value.analysis_updates),
                "other_updates": dict(value.other_updates),
                "note": value.note,
            }
            for key, value in variant_map.items()
        },
        "sims": {key: value["path"] for key, value in sims.items()},
        "comparisons": comparisons,
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(
        path=output_md,
        config_path=config_path,
        baseline_theory=baseline_theory,
        sims=sims,
        theory_paths=theory_paths,
        comparisons=comparisons,
        variant_map=variant_map,
        plot_path=output_pdf,
    )
    plot_variants(
        output_path=output_pdf,
        sims=sims,
        theories=theories,
        variant_labels={key: variant.label for key, variant in variant_map.items()},
        spectra=spectra,
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
