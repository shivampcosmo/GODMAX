#!/usr/bin/env python3
"""Plot high-resolution pilot bandpowers with per-spectrum iNKA errors.

This diagnostic is deliberately narrower than the production plotting path.
It can plot the validated spectra-only HDF5 product with no uncertainties, or
combine it with covariance shards that contain the 46 self-covariance blocks.
It never constructs or saves a joint covariance: off-diagonal
spectrum-to-spectrum blocks are neither needed for error bars nor implied to be
zero.

The resulting figures are therefore suitable for inspecting measured
bandpowers and their marginal one-sigma Gaussian/iNKA errors, but not for
joint S/N, chi-square, likelihood, or correlation-matrix calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import godmax_multiprobe_theory_utils as gmt
import multiprobe_namaster as mp
import run_multiprobe_production as production


FAMILY_LABELS: Mapping[str, str] = {
    "des_shear_EE": "DES shear EE",
    "act_y_des_shear_E": "ACT y × DES shear E",
    "desi_g_auto": "DESI DR9 Extended LRG auto (signal + shot noise)",
    "desi_g_act_y": "DESI DR9 Extended LRG × ACT y",
    "desi_g_des_shear_E": "DESI DR9 Extended LRG × DES shear E",
    "desi_g_act_kappa": "DESI DR9 Extended LRG × ACT CMB κ",
    "desi_pi_act_T": "DESI momentum × ACT temperature",
}

FAMILY_COLORS: Mapping[str, str] = {
    "des_shear_EE": "#2457a6",
    "act_y_des_shear_E": "#b43c2f",
    "desi_g_auto": "#1e7a49",
    "desi_g_act_y": "#7a4aa0",
    "desi_g_des_shear_E": "#c26a1b",
    "desi_g_act_kappa": "#00838f",
    "desi_pi_act_T": "#5e5147",
}


@dataclass(frozen=True)
class PilotSpectrum:
    name: str
    label: str
    family: str
    ell: np.ndarray
    cl: np.ndarray
    sigma_cl: np.ndarray | None
    valid: np.ndarray
    ell_left: np.ndarray
    ell_right: np.ndarray


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def decode_strings(values: Iterable[object]) -> List[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def load_pilot_spectra_values(spectra_path: Path) -> List[PilotSpectrum]:
    """Load and validate the spectra pilot without inventing uncertainties."""

    spectra_path = spectra_path.expanduser().resolve()
    with h5py.File(spectra_path, "r") as h5:
        mp.validate_measurement_product_identity(h5)
        config_payload = json.loads(str(h5.attrs["config_json"]))
        stage = str(config_payload.get("stage", ""))
        config = mp.MeasurementConfig.for_stage(stage)
        config.output_dir = str(spectra_path.parent.parent)
        config.compute_covariance = False
        config.validate()
        for key, expected in config.to_dict().items():
            if key == "output_dir":
                continue
            if key not in config_payload or config_payload[key] != expected:
                raise ValueError(
                    f"Spectra config {key}={config_payload.get(key)!r}, expected {expected!r}."
                )
        names = decode_strings(h5["joint/spectrum_names"][:])
        starts = np.asarray(h5["joint/slice_start"][:], dtype=int)
        stops = np.asarray(h5["joint/slice_stop"][:], dtype=int)
        raw = np.asarray(h5["joint/data_vector_raw"][:], dtype=np.float64)
        valid = np.asarray(h5["joint/data_vector_valid"][:], dtype=bool)
        ell_left = np.asarray(h5["ell_left"][:], dtype=np.int64)
        ell_right = np.asarray(h5["ell_right"][:], dtype=np.int64)
        n_band = int(ell_left.size)
        if ell_right.shape != ell_left.shape or n_band != int(config.n_bins):
            raise ValueError("Spectra bandpower edges do not match the configured bin count.")
        spectra: List[PilotSpectrum] = []
        for name, start, stop in zip(names, starts, stops):
            group = h5[f"spectra/{name}"]
            ell = np.asarray(group["ell"][:], dtype=np.float64)
            cl = np.asarray(group["cl"][:], dtype=np.float64)
            local_valid = valid[int(start) : int(stop)]
            if ell.shape != (n_band,) or cl.shape != (n_band,) or local_valid.shape != (n_band,):
                raise ValueError(f"Spectrum {name!r} does not use the common {n_band}-band grid.")
            if not np.all(np.isfinite(ell)) or not np.all(np.isfinite(cl)):
                raise ValueError(f"Spectrum {name!r} contains non-finite ell or C_ell values.")
            if not np.array_equal(cl, raw[int(start) : int(stop)]):
                raise ValueError(f"Spectrum {name!r} is not the exact raw-vector slice.")
            spectra.append(
                PilotSpectrum(
                    name=name,
                    label=str(group.attrs.get("label", name)),
                    family=str(group.attrs.get("family", "unknown")),
                    ell=ell,
                    cl=cl,
                    sigma_cl=None,
                    valid=local_valid,
                    ell_left=ell_left.copy(),
                    ell_right=ell_right.copy(),
                )
            )
    return spectra


def required_self_covariance_groups(
    manifest: Mapping[str, object],
    spectrum_names: Sequence[str],
) -> Dict[str, Mapping[str, object]]:
    """Return the unique covariance group containing each spectrum self-block."""

    required = set(str(name) for name in spectrum_names)
    found: Dict[str, Mapping[str, object]] = {}
    for group in manifest.get("groups", []):
        for block in group.get("blocks", []):
            name_i = str(block.get("spec_i", ""))
            name_j = str(block.get("spec_j", ""))
            if name_i != name_j or name_i not in required:
                continue
            if name_i in found:
                raise ValueError(f"Spectrum {name_i!r} has more than one manifest self-block group.")
            found[name_i] = group
    missing = sorted(required - set(found))
    extra = sorted(set(found) - required)
    if missing or extra:
        raise ValueError(f"Self-covariance manifest coverage mismatch: missing={missing}, extra={extra}.")
    return found


def _validate_shard_and_read_self_blocks(
    shard_path: Path,
    group: Mapping[str, object],
    required_names: Sequence[str],
    *,
    manifest_digest: str,
    covariance_config_digest: str,
    map_product_id: str,
    n_band: int,
) -> Dict[str, np.ndarray]:
    if not shard_path.is_file():
        raise FileNotFoundError(f"Required covariance shard is missing: {shard_path}")
    required = set(str(name) for name in required_names)
    blocks: Dict[str, np.ndarray] = {}
    with h5py.File(shard_path, "r") as h5:
        expected_attrs = {
            "pipeline_version": mp.MEASUREMENT_PIPELINE_VERSION,
            "covariance_estimator_version": mp.COVARIANCE_ESTIMATOR_VERSION,
            "manifest_digest": str(manifest_digest),
            "covariance_config_digest": str(covariance_config_digest),
            "map_product_id": str(map_product_id),
            "group_digest": production._group_digest(group),
            "group_class": str(group["class"]),
        }
        for key, expected in expected_attrs.items():
            actual = str(h5.attrs.get(key, ""))
            if actual != expected:
                raise ValueError(
                    f"Shard {shard_path.name} attribute {key}={actual!r}, expected {expected!r}."
                )
        if int(h5.attrs.get("group_index", -1)) != int(group["index"]):
            raise ValueError(f"Shard {shard_path.name} has the wrong covariance group index.")
        saved_group = json.loads(str(h5.attrs.get("group_json", "{}")))
        if saved_group != group:
            raise ValueError(f"Shard {shard_path.name} group_json does not match the manifest.")
        if "covariance_blocks" not in h5:
            raise ValueError(f"Shard {shard_path.name} has no covariance_blocks group.")
        for name in sorted(required):
            dataset_name = f"{name}__x__{name}"
            if dataset_name not in h5["covariance_blocks"]:
                raise ValueError(f"Shard {shard_path.name} is missing self-block {dataset_name!r}.")
            dataset = h5[f"covariance_blocks/{dataset_name}"]
            if str(dataset.attrs.get("spectrum_i", "")) != name or str(
                dataset.attrs.get("spectrum_j", "")
            ) != name:
                raise ValueError(f"Shard {shard_path.name} has inconsistent block labels for {name!r}.")
            block = np.asarray(dataset[:], dtype=np.float64)
            if block.shape != (n_band, n_band):
                raise ValueError(
                    f"Self-block {name!r} has shape {block.shape}; expected {(n_band, n_band)}."
                )
            if not np.all(np.isfinite(block)):
                raise ValueError(f"Self-block {name!r} contains non-finite values.")
            diagonal = np.diag(block)
            if np.any(diagonal <= 0.0):
                bad = np.flatnonzero(diagonal <= 0.0).tolist()
                raise ValueError(f"Self-block {name!r} has non-positive variance at bands {bad}.")
            blocks[name] = block
    return blocks


def load_pilot_spectra_and_errors(
    spectra_path: Path,
    manifest_path: Path,
    shard_dir: Path,
) -> Tuple[List[PilotSpectrum], Mapping[str, object], List[Mapping[str, object]]]:
    spectra_path = spectra_path.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    shard_dir = shard_dir.expanduser().resolve()

    with h5py.File(spectra_path, "r") as h5:
        map_product_id = mp.validate_measurement_product_identity(h5)
        if "joint/cov" in h5:
            raise ValueError(
                "This is a spectra-only pilot diagnostic. A joint covariance is already present; "
                "use run_multiprobe_production.py plot-measurement-dell instead."
            )
        config_payload = json.loads(str(h5.attrs["config_json"]))
        stage = str(config_payload.get("stage", ""))
        config = mp.MeasurementConfig.for_stage(stage)
        config.output_dir = str(spectra_path.parent.parent)
        config.compute_covariance = False
        config.validate()
        for key, expected in config.to_dict().items():
            if key == "output_dir":
                continue
            if key not in config_payload or config_payload[key] != expected:
                raise ValueError(
                    f"Spectra config {key}={config_payload.get(key)!r}, expected {expected!r}."
                )
        names = decode_strings(h5["joint/spectrum_names"][:])
        starts = np.asarray(h5["joint/slice_start"][:], dtype=int)
        stops = np.asarray(h5["joint/slice_stop"][:], dtype=int)
        raw = np.asarray(h5["joint/data_vector_raw"][:], dtype=np.float64)
        valid = np.asarray(h5["joint/data_vector_valid"][:], dtype=bool)
        ell_left = np.asarray(h5["ell_left"][:], dtype=np.int64)
        ell_right = np.asarray(h5["ell_right"][:], dtype=np.int64)
        n_band = int(ell_left.size)
        if ell_right.shape != ell_left.shape or n_band != int(config.n_bins):
            raise ValueError("Spectra bandpower edges do not match the configured bin count.")
        spectra_records = []
        for name, start, stop in zip(names, starts, stops):
            group = h5[f"spectra/{name}"]
            ell = np.asarray(group["ell"][:], dtype=np.float64)
            cl = np.asarray(group["cl"][:], dtype=np.float64)
            local_valid = valid[int(start) : int(stop)]
            if ell.shape != (n_band,) or cl.shape != (n_band,) or local_valid.shape != (n_band,):
                raise ValueError(f"Spectrum {name!r} does not use the common {n_band}-band grid.")
            if not np.all(np.isfinite(ell)) or not np.all(np.isfinite(cl)):
                raise ValueError(f"Spectrum {name!r} contains non-finite ell or C_ell values.")
            if not np.array_equal(cl, raw[int(start) : int(stop)]):
                raise ValueError(f"Spectrum {name!r} is not the exact raw-vector slice.")
            spectra_records.append(
                {
                    "name": name,
                    "label": str(group.attrs.get("label", name)),
                    "family": str(group.attrs.get("family", "unknown")),
                    "ell": ell,
                    "cl": cl,
                    "valid": local_valid,
                }
            )

    manifest = production.load_covariance_manifest(manifest_path)
    production.validate_covariance_manifest(manifest, config)
    if str(manifest["manifest_digest"]) == "":
        raise ValueError("Covariance manifest digest is empty.")
    if list(manifest["spectrum_names"]) != names:
        raise ValueError("Covariance manifest spectrum order does not match the spectra HDF5 product.")
    group_for_name = required_self_covariance_groups(manifest, names)
    names_by_group: Dict[int, List[str]] = {}
    groups_by_index: Dict[int, Mapping[str, object]] = {}
    for name, group in group_for_name.items():
        index = int(group["index"])
        names_by_group.setdefault(index, []).append(name)
        groups_by_index[index] = group

    self_blocks: Dict[str, np.ndarray] = {}
    shard_records: List[Mapping[str, object]] = []
    for index in sorted(groups_by_index):
        group = groups_by_index[index]
        shard_path = shard_dir / f"cov_group_{index:04d}_{str(group['class'])}.h5"
        blocks = _validate_shard_and_read_self_blocks(
            shard_path,
            group,
            names_by_group[index],
            manifest_digest=str(manifest["manifest_digest"]),
            covariance_config_digest=str(manifest["covariance_config_digest"]),
            map_product_id=map_product_id,
            n_band=n_band,
        )
        self_blocks.update(blocks)
        shard_records.append(
            {
                "group_index": index,
                "group_class": str(group["class"]),
                "path": str(shard_path),
                "sha256": sha256_file(shard_path),
                "self_spectra": sorted(names_by_group[index]),
            }
        )

    if set(self_blocks) != set(names):
        raise ValueError("Loaded covariance shards do not cover all 46 spectrum self-blocks.")
    spectra = [
        PilotSpectrum(
            name=str(record["name"]),
            label=str(record["label"]),
            family=str(record["family"]),
            ell=np.asarray(record["ell"], dtype=np.float64),
            cl=np.asarray(record["cl"], dtype=np.float64),
            sigma_cl=np.sqrt(np.diag(self_blocks[str(record["name"])])),
            valid=np.asarray(record["valid"], dtype=bool),
            ell_left=ell_left.copy(),
            ell_right=ell_right.copy(),
        )
        for record in spectra_records
    ]
    return spectra, manifest, shard_records


def _family_grid_size(n_spectra: int) -> Tuple[int, int]:
    n_col = min(4, int(math.ceil(math.sqrt(n_spectra))))
    n_row = int(math.ceil(n_spectra / n_col))
    return n_row, n_col


def _set_robust_linear_limits(ax: plt.Axes, values: Sequence[np.ndarray]) -> None:
    finite_parts = [np.asarray(value)[np.isfinite(value)] for value in values]
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        return
    all_values = np.concatenate(finite_parts)
    low = float(np.min(all_values))
    high = float(np.max(all_values))
    if not np.isfinite(low) or not np.isfinite(high):
        return
    if low == high:
        pad = max(abs(low) * 0.1, 1.0e-12)
    else:
        pad = 0.09 * (high - low)
    ax.set_ylim(low - pad, high + pad)


def plot_pilot_dell(
    spectra: Sequence[PilotSpectrum],
    output_dir: Path,
    *,
    stage_label: str,
    ksz_scale: float,
    show_kappa_null_diagnostics: bool,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    error_flags = [spectrum.sigma_cl is not None for spectrum in spectra]
    if any(error_flags) and not all(error_flags):
        raise ValueError("Pilot plot cannot mix spectra with and without covariance errors.")
    has_error_bars = bool(error_flags and all(error_flags))
    by_family: Dict[str, List[PilotSpectrum]] = {}
    for spectrum in spectra:
        by_family.setdefault(spectrum.family, []).append(spectrum)
    missing_families = [family for family in gmt.MEASUREMENT_FAMILY_ORDER if family not in by_family]
    if missing_families:
        raise ValueError(f"Pilot spectra are missing required families: {missing_families}.")

    suffix = "Dell_with_iNKA_errors" if has_error_bars else "Dell_no_errorbars"
    pdf_path = output_dir / f"highres4096_pilot_all_spectra_{suffix}.pdf"
    outputs: List[Path] = []
    with PdfPages(pdf_path) as pdf:
        for family in gmt.MEASUREMENT_FAMILY_ORDER:
            family_spectra = by_family[family]
            n_row, n_col = _family_grid_size(len(family_spectra))
            fig, axes = plt.subplots(
                n_row,
                n_col,
                figsize=(4.45 * n_col, 3.25 * n_row),
                squeeze=False,
                constrained_layout=True,
            )
            color = FAMILY_COLORS[family]
            for ax, spectrum in zip(axes.flat, family_spectra):
                keep = spectrum.valid
                ell = spectrum.ell[keep]
                sigma_cl = (
                    np.asarray(spectrum.sigma_cl, dtype=np.float64)
                    if spectrum.sigma_cl is not None
                    else np.zeros_like(spectrum.cl)
                )
                y, yerr, ylabel = gmt.measurement_plot_values(
                    ell,
                    spectrum.cl[keep],
                    sigma_cl[keep],
                    family=family,
                    quantity="dell",
                    ksz_scale=ksz_scale,
                )
                if has_error_bars:
                    ax.errorbar(
                        ell,
                        y,
                        yerr=yerr,
                        fmt="o",
                        ms=3.8,
                        color=color,
                        ecolor=color,
                        elinewidth=1.05,
                        capsize=2.3,
                        alpha=0.94,
                        label="pilot ±1σ iNKA",
                        zorder=3,
                    )
                    ylim_values: List[np.ndarray] = [y - yerr, y + yerr]
                else:
                    ax.plot(
                        ell,
                        y,
                        "o-",
                        ms=3.8,
                        lw=1.0,
                        color=color,
                        alpha=0.94,
                        label="pilot bandpower",
                        zorder=3,
                    )
                    ylim_values = [y]
                if family == "desi_g_act_kappa":
                    boundary = 3001.0
                    ax.axvspan(boundary, float(spectrum.ell_right[-1]), color="#bbbbbb", alpha=0.18)
                    ax.axvline(boundary, color="#666666", ls="--", lw=0.9)
                    invalid = ~keep
                    if show_kappa_null_diagnostics and np.any(invalid):
                        null_y, _, _ = gmt.measurement_plot_values(
                            spectrum.ell[invalid],
                            spectrum.cl[invalid],
                            sigma_cl[invalid],
                            family=family,
                            quantity="dell",
                            ksz_scale=ksz_scale,
                        )
                        ax.plot(
                            spectrum.ell[invalid],
                            null_y,
                            "x",
                            ms=4.0,
                            mew=1.0,
                            color="#888888",
                            alpha=0.75,
                            label="raw transfer-null diagnostic",
                            zorder=2,
                        )
                        ylim_values.append(null_y)
                ax.axhline(0.0, color="#777777", lw=0.75, alpha=0.65, zorder=1)
                ax.set_xscale("log")
                ax.set_xlim(float(spectrum.ell_left[0]) * 0.92, float(spectrum.ell_right[-1]) * 1.03)
                _set_robust_linear_limits(ax, ylim_values)
                ax.grid(True, color="#d8dbe2", lw=0.7, alpha=0.75)
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(ylabel)
                ax.set_title(spectrum.label, fontsize=9)
                if family == "desi_g_act_kappa":
                    ax.legend(loc="best", fontsize=6.5, frameon=False)
            for ax in axes.flat[len(family_spectra) :]:
                ax.set_visible(False)
            title = (
                f"{stage_label}: {FAMILY_LABELS[family]} in $D_\\ell$\n"
            )
            if has_error_bars:
                title += "per-spectrum Gaussian/iNKA errors; no joint covariance implied"
            else:
                title += "spectra only — no covariance or error bars"
            if family == "desi_g_act_kappa":
                title += r"; grey ×: ACT-κ transfer-null at $\ell\geq3001$ (excluded)"
            if family == "desi_pi_act_T":
                title += r"; display: $-10^3D_\ell^{\pi T}$"
            fig.suptitle(title, fontsize=10.0)
            png = output_dir / f"highres4096_pilot_{family}_{suffix}.png"
            fig.savefig(png, dpi=190)
            pdf.savefig(fig)
            plt.close(fig)
            outputs.append(png)
    outputs.append(pdf_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-path", required=True)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-label", default="highres4096 pilot")
    parser.add_argument("--ksz-scale", type=float, default=1000.0)
    parser.add_argument(
        "--no-error-bars",
        action="store_true",
        help="Plot the validated spectra values only; do not require or imply covariance errors.",
    )
    parser.add_argument(
        "--omit-kappa-null-diagnostics",
        action="store_true",
        help="Omit the raw ell>3000 ACT-kappa transfer-null estimator markers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spectra_path = Path(args.spectra_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.no_error_bars:
        spectra = load_pilot_spectra_values(spectra_path)
        manifest = None
        manifest_path = None
        shard_records: List[Mapping[str, object]] = []
    else:
        if not args.manifest_path or not args.shard_dir:
            raise ValueError("--manifest-path and --shard-dir are required unless --no-error-bars is set.")
        manifest_path = Path(args.manifest_path).expanduser().resolve()
        shard_dir = Path(args.shard_dir).expanduser().resolve()
        spectra, manifest, shard_records = load_pilot_spectra_and_errors(
            spectra_path,
            manifest_path,
            shard_dir,
        )
    outputs = plot_pilot_dell(
        spectra,
        output_dir,
        stage_label=str(args.stage_label),
        ksz_scale=float(args.ksz_scale),
        show_kappa_null_diagnostics=not bool(args.omit_kappa_null_diagnostics),
    )
    n_archive = sum(spectrum.cl.size for spectrum in spectra)
    n_active = sum(int(np.count_nonzero(spectrum.valid)) for spectrum in spectra)
    has_error_bars = not bool(args.no_error_bars)
    if has_error_bars:
        scientific_scope = (
            "Per-spectrum Gaussian/iNKA self-covariance error bars only. Off-diagonal "
            "spectrum covariance is absent and not represented as zero; this diagnostic "
            "must not be used for joint S/N, chi-square, likelihood, or inference."
        )
    else:
        scientific_scope = (
            "Spectra-only D_ell diagnostic. No covariance was loaded, no uncertainty is "
            "shown or implied, and this diagnostic must not be used for S/N, chi-square, "
            "likelihood, or inference."
        )
    attestation = {
        "schema": "xdesi_highres_pilot_dell_diagnostic_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": scientific_scope,
        "error_bars_shown": has_error_bars,
        "spectra_h5": str(spectra_path),
        "spectra_sha256": sha256_file(spectra_path),
        "manifest_json": str(manifest_path) if manifest_path is not None else None,
        "manifest_file_sha256": sha256_file(manifest_path) if manifest_path is not None else None,
        "manifest_digest": str(manifest["manifest_digest"]) if manifest is not None else None,
        "covariance_estimator": "Gaussian/iNKA" if has_error_bars else None,
        "covariance_shards": shard_records,
        "n_spectra": len(spectra),
        "n_bands_per_spectrum": 20,
        "archive_elements": n_archive,
        "active_elements": n_active,
        "invalid_kappa_transfer_null_elements": n_archive - n_active,
        "kappa_validity": "right-exclusive band edge <= 3001 (ell <= 3000 supported)",
        "ksz_display": "-1000 * D_ell^{pi,T}; saved C_ell remains raw and unchanged",
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in outputs
        ],
    }
    attestation_suffix = "with_iNKA_errors" if has_error_bars else "no_errorbars"
    attestation_path = output_dir / f"highres4096_pilot_Dell_{attestation_suffix}_manifest.json"
    attestation_path.write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(outputs)} figures plus {attestation_path}; "
        f"archive={n_archive}, active={n_active}, invalid={n_archive - n_active}."
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
