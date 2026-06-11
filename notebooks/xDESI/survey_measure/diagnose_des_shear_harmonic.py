#!/usr/bin/env python
"""DES-only checks against the DES Y3 harmonic-space shear setup.

This script is intentionally separate from the 46-spectrum multi-probe
measurement.  It compares the fast low-res settings to the DES Y3 paper's
fiducial shear-only choices: nside=1024, ell=8..2048, 32 sqrt-spaced bins,
raw weighted masks, and HEALPix polarization pixel-window deconvolution in
the bandpower weights.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pymaster as nmt

from multiprobe_namaster import (
    MeasurementConfig,
    SpectrumSpec,
    SurveyBundle,
    build_nmt_fields,
    build_shear_fields,
    compute_covariance_block,
    make_sqrt_bandpower_edges,
    measure_spectrum,
    utc_now,
)


@dataclass(frozen=True)
class ShearScenario:
    name: str
    lmax: int
    n_bins: int
    mask_dataset: str
    noise_attr: str
    deconvolve_pixel_window: bool


SCENARIOS: Tuple[ShearScenario, ...] = (
    ShearScenario(
        name="current_lowres_norm_mask_ell8_1024",
        lmax=1024,
        n_bins=24,
        mask_dataset="mask_weight",
        noise_attr="shape_noise_pseudo_cl_normalized_weight_mask",
        deconvolve_pixel_window=False,
    ),
    ShearScenario(
        name="paper_range_norm_mask_no_pixwin_ell8_2048",
        lmax=2048,
        n_bins=32,
        mask_dataset="mask_weight",
        noise_attr="shape_noise_pseudo_cl_normalized_weight_mask",
        deconvolve_pixel_window=False,
    ),
    ShearScenario(
        name="paper_like_raw_mask_pixwin_ell8_2048",
        lmax=2048,
        n_bins=32,
        mask_dataset="mask_weight_raw",
        noise_attr="shape_noise_pseudo_cl_raw_weight_mask",
        deconvolve_pixel_window=True,
    ),
)


def shear_pair_spec(tomo_i: int, tomo_j: int) -> SpectrumSpec:
    """Return the DES shear EE spectrum spec for one tomographic pair."""

    tomo_i = int(tomo_i)
    tomo_j = int(tomo_j)
    if not (1 <= tomo_i <= 4 and 1 <= tomo_j <= 4):
        raise ValueError("DES source tomography bins must be in 1..4.")
    if tomo_j < tomo_i:
        tomo_i, tomo_j = tomo_j, tomo_i
    return SpectrumSpec(
        name=f"des_shear_EE_tomo{tomo_i}x{tomo_j}",
        family="des_shear_EE",
        fields=(f"s{tomo_i}", f"s{tomo_j}"),
        component=0,
        label=f"DES shear E tomo {tomo_i} x {tomo_j}",
        theory_key=f"des_shear_EE_tomo{tomo_i}x{tomo_j}",
        metadata={"source_tomo_i": tomo_i, "source_tomo_j": tomo_j},
    )


def shear_specs() -> List[SpectrumSpec]:
    specs: List[SpectrumSpec] = []
    for i in range(1, 5):
        for j in range(i, 5):
            specs.append(shear_pair_spec(i, j))
    return specs


def scenario_band_edges(bundle: SurveyBundle, scenario: ShearScenario) -> Tuple[np.ndarray, np.ndarray]:
    shear_h5 = bundle.shear_path_for_nside(1024)
    if scenario.lmax == 2048 and scenario.n_bins == 32:
        with h5py.File(shear_h5, "r") as h5:
            left = np.asarray(h5["bandpowers/ell_left"][:], dtype=np.int32)
            right = np.asarray(h5["bandpowers/ell_right"][:], dtype=np.int32)
    else:
        left, right = make_sqrt_bandpower_edges(8, scenario.lmax, scenario.n_bins)
    return left, right


def make_scenario_bins(bundle: SurveyBundle, scenario: ShearScenario) -> nmt.NmtBin:
    shear_h5 = bundle.shear_path_for_nside(1024)
    left, right = scenario_band_edges(bundle, scenario)
    if not scenario.deconvolve_pixel_window:
        return nmt.NmtBin.from_edges(left, right)

    with h5py.File(shear_h5, "r") as h5:
        pixwin_full = np.asarray(h5["pixel_window/polarization"][: scenario.lmax + 1], dtype=np.float64)
    ells = np.concatenate([np.arange(li, ri, dtype=np.int64) for li, ri in zip(left, right)])
    pixwin = pixwin_full[ells]
    f_ell = np.ones_like(pixwin, dtype=np.float64)
    good = pixwin > 0
    f_ell[good] = 1.0 / np.square(pixwin[good])
    return nmt.NmtBin.from_edges(left, right, f_ell=f_ell)


def scenario_config(base: MeasurementConfig, scenario: ShearScenario) -> MeasurementConfig:
    return MeasurementConfig(
        stage=f"diagnostic_{scenario.name}",
        nside=1024,
        lmax=scenario.lmax,
        ell_min=8,
        n_bins=scenario.n_bins,
        act_downgrade=base.act_downgrade,
        catalog_chunk=base.catalog_chunk,
        shear_mask_dataset=scenario.mask_dataset,
        shear_noise_attr=scenario.noise_attr,
        shear_e_to_kappa_sign=base.shear_e_to_kappa_sign,
        subtract_masked_mean=base.subtract_masked_mean,
        n_iter=base.n_iter,
        n_iter_mask=base.n_iter_mask,
        covariance_l_toeplitz=base.covariance_l_toeplitz,
        covariance_l_exact=base.covariance_l_exact,
        covariance_dl_band=base.covariance_dl_band,
        covariance_workspace_cache_size=base.covariance_workspace_cache_size,
        covariance_input_mode=base.covariance_input_mode,
        covariance_input_smooth_bandpowers=base.covariance_input_smooth_bandpowers,
        covariance_input_smooth_window=base.covariance_input_smooth_window,
        covariance_zero_parity_odd_inputs=base.covariance_zero_parity_odd_inputs,
        compute_covariance=True,
        compute_covariance_eigenvalues=False,
        include_ksz_velocity_shuffle=False,
        ksz_shuffle_seed=base.ksz_shuffle_seed,
        des_y3_source_nz_fits=base.des_y3_source_nz_fits,
        output_dir=base.output_dir,
    )


def run_scenario(bundle: SurveyBundle, base_config: MeasurementConfig, scenario: ShearScenario) -> Dict[str, object]:
    config = scenario_config(base_config, scenario)
    bins = make_scenario_bins(bundle, scenario)
    fields = build_shear_fields(bundle, config)
    nmt_fields = build_nmt_fields(fields, config)
    specs = shear_specs()
    workspace_cache = {}
    cov_workspace_cache = {}
    input_cl_cache = {}

    spectra = {}
    diag_snr2 = 0.0
    negative_bands = 0
    total_bands = 0
    for spec in specs:
        measured = measure_spectrum(spec, nmt_fields, bins, workspace_cache, config)
        cov = compute_covariance_block(
            spec,
            spec,
            nmt_fields,
            bins,
            workspace_cache,
            cov_workspace_cache,
            input_cl_cache,
            config,
        )
        cl = np.asarray(measured["cl"], dtype=np.float64)
        diag = np.diag(cov)
        good = np.isfinite(cl) & np.isfinite(diag) & (diag > 0)
        snr = float(np.sqrt(np.sum(np.square(cl[good]) / diag[good])))
        diag_snr2 += snr * snr
        negative_bands += int(np.count_nonzero(cl < 0))
        total_bands += int(cl.size)
        spectra[spec.name] = {
            "ell_min": float(np.min(measured["ell"])),
            "ell_max": float(np.max(measured["ell"])),
            "cl_median": float(np.nanmedian(cl)),
            "dell_median": float(np.nanmedian(np.asarray(measured["ell"]) * cl)),
            "diag_snr": snr,
            "err_median": float(np.nanmedian(np.sqrt(np.clip(diag, 0.0, np.inf)))),
        }

    return {
        "scenario": scenario.name,
        "lmax": scenario.lmax,
        "n_bins": scenario.n_bins,
        "mask_dataset": scenario.mask_dataset,
        "noise_attr": scenario.noise_attr,
        "deconvolve_pixel_window": scenario.deconvolve_pixel_window,
        "effective_ells": bins.get_effective_ells().tolist(),
        "combined_diag_snr": float(np.sqrt(diag_snr2)),
        "negative_band_fraction": float(negative_bands / max(total_bands, 1)),
        "spectra": spectra,
    }


def run_shear_pair_scenario(
    bundle: SurveyBundle,
    base_config: Optional[MeasurementConfig] = None,
    scenario: Optional[ShearScenario] = None,
    tomo_i: int = 4,
    tomo_j: int = 4,
    compute_covariance: bool = True,
) -> Dict[str, object]:
    """Measure one DES shear EE tomographic pair with one diagnostic scenario.

    The default scenario is the closest local reproduction of Fig. 4 in the
    DES Y3 harmonic-space paper: raw Metacalibration-weight mask, 32
    square-root-spaced equal-weight bands over ell=8..2048, and polarization
    HEALPix pixel-window deconvolution in the NaMaster bandpower weights.
    """

    base_config = base_config or MeasurementConfig.for_stage("lowres")
    scenario = scenario or next(s for s in SCENARIOS if s.name == "paper_like_raw_mask_pixwin_ell8_2048")
    config = scenario_config(base_config, scenario)
    config.compute_covariance = bool(compute_covariance)
    bins = make_scenario_bins(bundle, scenario)
    left, right = scenario_band_edges(bundle, scenario)
    fields = build_shear_fields(bundle, config)
    nmt_fields = build_nmt_fields(fields, config)
    spec = shear_pair_spec(tomo_i, tomo_j)

    workspace_cache = {}
    measured = measure_spectrum(spec, nmt_fields, bins, workspace_cache, config)
    cl = np.asarray(measured["cl"], dtype=np.float64)
    ell = np.asarray(measured["ell"], dtype=np.float64)

    covariance = None
    err = np.full_like(cl, np.nan, dtype=np.float64)
    diag_snr = np.nan
    cov_workspace_cache = {}
    input_cl_cache = {}
    if compute_covariance:
        covariance = compute_covariance_block(
            spec,
            spec,
            nmt_fields,
            bins,
            workspace_cache,
            cov_workspace_cache,
            input_cl_cache,
            config,
        )
        diag = np.diag(covariance)
        good = np.isfinite(cl) & np.isfinite(diag) & (diag > 0)
        err = np.sqrt(np.clip(diag, 0.0, np.inf))
        diag_snr = float(np.sqrt(np.sum(np.square(cl[good]) / diag[good])))

    cl_all = np.asarray(measured["cl_all_components"], dtype=np.float64)
    noise = measured["noise_decoupled_all_components"]
    noise_all = None if noise is None else np.asarray(noise, dtype=np.float64)
    return {
        "scenario": scenario.name,
        "lmax": int(scenario.lmax),
        "n_bins": int(scenario.n_bins),
        "mask_dataset": scenario.mask_dataset,
        "noise_attr": scenario.noise_attr,
        "deconvolve_pixel_window": bool(scenario.deconvolve_pixel_window),
        "tomo_i": int(spec.metadata["source_tomo_i"]),
        "tomo_j": int(spec.metadata["source_tomo_j"]),
        "spectrum_name": spec.name,
        "component_labels": list(measured["component_labels"]),
        "ell_left": left.astype(int).tolist(),
        "ell_right": right.astype(int).tolist(),
        "ell": ell.tolist(),
        "cl": cl.tolist(),
        "err": err.tolist(),
        "ell_cl_1e7": (ell * cl * 1.0e7).tolist(),
        "ell_err_1e7": (ell * err * 1.0e7).tolist(),
        "cl_all_components": cl_all.tolist(),
        "noise_decoupled_all_components": None if noise_all is None else noise_all.tolist(),
        "diag_snr": None if not np.isfinite(diag_snr) else float(diag_snr),
        "covariance": None if covariance is None else np.asarray(covariance, dtype=np.float64).tolist(),
        "paper_comparison_note": (
            "Fig. 4 plots the noise-bias-corrected E-mode data as Lbar*C_L^EE "
            "in units of 1e-7. The ell_cl_1e7 array is the same plotting "
            "quantity using the NaMaster effective multipole for each band."
        ),
    }


def write_shear_pair_result(result: Dict[str, object], output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    tmp.replace(out)
    return out


def plot_shear_pair_result(
    result: Dict[str, object],
    output: Optional[str | Path] = None,
    *,
    show_components: bool = True,
    x_axis: str = "paper_sqrt",
):
    """Plot the single-pair result in the Fig. 4 display convention."""

    import matplotlib.pyplot as plt

    ell = np.asarray(result["ell"], dtype=np.float64)
    y = np.asarray(result["ell_cl_1e7"], dtype=np.float64)
    yerr = np.asarray(result["ell_err_1e7"], dtype=np.float64)
    labels = list(result["component_labels"])
    cl_all = np.asarray(result["cl_all_components"], dtype=np.float64)

    x_axis = str(x_axis)
    if x_axis == "paper_sqrt":
        x = np.sqrt(ell)
        x_label = "Multipole ell"
    elif x_axis == "log":
        x = ell
        x_label = "Multipole ell"
    else:
        raise ValueError("x_axis must be 'paper_sqrt' or 'log'.")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.errorbar(x, y, yerr=yerr, fmt="o", ms=4, lw=1, capsize=2, label="EE noise-subtracted")
    if show_components and cl_all.ndim == 2 and cl_all.shape[0] > 3:
        ax.plot(x, ell * cl_all[1] * 1.0e7, color="0.55", lw=1, alpha=0.8, label=labels[1])
        ax.plot(x, ell * cl_all[2] * 1.0e7, color="0.70", lw=1, alpha=0.8, label=labels[2])
        ax.plot(x, ell * cl_all[3] * 1.0e7, color="0.35", lw=1, alpha=0.8, label=labels[3])
    ax.axhline(0.0, color="0.2", lw=0.8)
    if x_axis == "log":
        ax.set_xscale("log")
        ax.set_xlim(max(5.0, 0.8 * np.nanmin(ell)), 1.15 * np.nanmax(ell))
    else:
        ticks_ell = np.asarray([0, 100, 400, 900, 1600], dtype=np.float64)
        ax.set_xticks(np.sqrt(ticks_ell))
        ax.set_xticklabels([str(int(v)) for v in ticks_ell])
        ax.set_xlim(0.0, np.sqrt(2048.0) * 1.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\bar{L}\,\hat{C}^{EE}_L$  [$10^{-7}$]")
    ax.set_title(
        f"DES shear tomo {result['tomo_i']} x {result['tomo_j']} "
        f"({result['scenario']})"
    )
    snr = result.get("diag_snr")
    if snr is not None:
        ax.text(0.03, 0.95, f"diag S/N = {float(snr):.2f}", transform=ax.transAxes, va="top")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey-root", default="data/xDESI/survey_data")
    parser.add_argument("--output", default="data/xDESI/processed/multiprobe_namaster/diagnostics/des_shear_harmonic_diagnosis.json")
    parser.add_argument("--scenario", choices=[s.name for s in SCENARIOS], action="append")
    parser.add_argument("--single-pair", action="store_true", help="Run only one tomographic pair instead of all 10 DES shear spectra.")
    parser.add_argument("--tomo-i", type=int, default=4)
    parser.add_argument("--tomo-j", type=int, default=4)
    parser.add_argument("--no-covariance", action="store_true")
    parser.add_argument("--plot-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = SurveyBundle.from_root(args.survey_root)
    bundle.validate_files()
    selected = [s for s in SCENARIOS if args.scenario is None or s.name in args.scenario]
    base_config = MeasurementConfig.for_stage("lowres")

    results = []
    for scenario in selected:
        print(f"[{utc_now()}] Running {scenario.name}", flush=True)
        if args.single_pair:
            result = run_shear_pair_scenario(
                bundle,
                base_config,
                scenario,
                tomo_i=args.tomo_i,
                tomo_j=args.tomo_j,
                compute_covariance=not args.no_covariance,
            )
        else:
            result = run_scenario(bundle, base_config, scenario)
        if args.single_pair:
            print(f"[{utc_now()}] {scenario.name}: diag SNR = {result['diag_snr']:.2f}", flush=True)
        else:
            print(
                f"[{utc_now()}] {scenario.name}: combined diag SNR = "
                f"{result['combined_diag_snr']:.2f}; negative-band fraction = "
                f"{result['negative_band_fraction']:.3f}",
                flush=True,
            )
        results.append(result)

    if args.single_pair and len(results) == 1:
        payload = {**results[0], "created_utc": utc_now()}
    else:
        payload = {"created_utc": utc_now(), "results": results}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(out)
    print(f"[{utc_now()}] Wrote {out.resolve()}", flush=True)
    if args.single_pair and len(results) == 1:
        result = results[0]
        arrays_out = out.with_name(f"{out.stem}_arrays.npz")
        np.savez_compressed(
            arrays_out,
            ell=np.asarray(result["ell"], dtype=np.float64),
            ell_left=np.asarray(result["ell_left"], dtype=np.int32),
            ell_right=np.asarray(result["ell_right"], dtype=np.int32),
            cl=np.asarray(result["cl"], dtype=np.float64),
            err=np.asarray(result["err"], dtype=np.float64),
            ell_cl_1e7=np.asarray(result["ell_cl_1e7"], dtype=np.float64),
            ell_err_1e7=np.asarray(result["ell_err_1e7"], dtype=np.float64),
            cl_all_components=np.asarray(result["cl_all_components"], dtype=np.float64),
            covariance=np.asarray(result["covariance"], dtype=np.float64)
            if result.get("covariance") is not None
            else np.asarray([], dtype=np.float64),
        )
        print(f"[{utc_now()}] Wrote {arrays_out.resolve()}", flush=True)
    if args.plot_output and len(results) == 1:
        plot_shear_pair_result(results[0], args.plot_output)
        print(f"[{utc_now()}] Wrote {Path(args.plot_output).resolve()}", flush=True)


if __name__ == "__main__":
    main()
