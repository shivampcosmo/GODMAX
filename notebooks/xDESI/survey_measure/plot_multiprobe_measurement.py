"""Plot xDESI multi-probe NaMaster measurement products.

This module is intentionally independent of the measurement code path.  It
only reads the final HDF5 product, so the same functions can be used for the
fast1024 smoke run, the midres2048 run, or any later product with the same
schema.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter


DEFAULT_PUBLISHED_GG_CLS = Path("/mnt/ceph/users/spandey/xdesi/lrg_xcorr_2023/v1/clustering/combined_cls.json")

FAMILY_LABELS: Mapping[str, str] = {
    "des_shear_EE": "DES Shear EE",
    "act_y_des_shear_E": "ACT y x DES Shear E",
    "desi_g_auto": "DESI Galaxy Auto",
    "desi_g_act_y": "DESI Galaxy x ACT y",
    "desi_g_des_shear_E": "DESI Galaxy x DES Shear E",
    "desi_g_act_kappa": "DESI Galaxy x ACT Kappa",
    "desi_pi_act_T": "DESI Velocity Template x ACT T",
}


FAMILY_ORDER: Sequence[str] = (
    "des_shear_EE",
    "act_y_des_shear_E",
    "desi_g_auto",
    "desi_g_act_y",
    "desi_g_des_shear_E",
    "desi_g_act_kappa",
    "desi_pi_act_T",
)


FAMILY_COLORS: Mapping[str, str] = {
    "des_shear_EE": "#2457a6",
    "act_y_des_shear_E": "#b43c2f",
    "desi_g_auto": "#1e7a49",
    "desi_g_act_y": "#7a4aa0",
    "desi_g_des_shear_E": "#c26a1b",
    "desi_g_act_kappa": "#00838f",
    "desi_pi_act_T": "#5e5147",
}

FAMILY_Y_DISPLAY_SCALE: Mapping[str, float] = {
    "desi_pi_act_T": 1.0e3,
}

FAMILY_PLOT_QUANTITY: Mapping[str, str] = {
    "desi_g_auto": "cl",
}

# Display total galaxy clustering for both current and historical products.
# Current `_gshot` products already store signal + shot noise; historical
# explicitly labelled signal-only products get their saved template restored
# in memory.  The per-spectrum convention prevents a second addition.
FAMILY_USE_TOTAL_CL: Mapping[str, bool] = {
    "desi_g_auto": True,
}

FAMILY_Y_DISPLAY_LABEL: Mapping[str, str] = {
    "desi_g_auto": r"$C_\ell$ (signal + shot noise)",
    "desi_pi_act_T": r"$10^3 D_\ell = 10^3 \ell(\ell+1)C_\ell/2\pi$",
}

FAMILY_YLIM: Mapping[str, tuple[float, float]] = {
    "desi_pi_act_T": (-0.075, 0.075),
}

FAMILY_YSCALE: Mapping[str, str] = {
    "desi_g_auto": "log",
}


@dataclass(frozen=True)
class PublishedGgCls:
    ell: np.ndarray
    cls_by_pz: Dict[int, np.ndarray]
    cls_key: str
    label: str



@dataclass(frozen=True)
class SpectrumPlotData:
    name: str
    label: str
    family: str
    ell: np.ndarray
    cl: np.ndarray
    cov: np.ndarray
    start: int
    stop: int
    noise_decoupled: np.ndarray | None = None
    cl_convention: str = ""

    @property
    def err(self) -> np.ndarray:
        diag = np.diag(self.cov)
        return np.sqrt(np.where(diag >= 0.0, diag, np.nan))

    @property
    def cl_total(self) -> np.ndarray:
        if self.cl_convention != "shot_noise_subtracted_signal" or self.noise_decoupled is None:
            return self.cl
        return self.cl + self.noise_decoupled

    @property
    def dell(self) -> np.ndarray:
        return dell_factor(self.ell) * self.cl

    @property
    def dell_err(self) -> np.ndarray:
        return dell_factor(self.ell) * self.err

    @property
    def dell_total(self) -> np.ndarray:
        return dell_factor(self.ell) * self.cl_total


@dataclass(frozen=True)
class MeasurementPlotData:
    path: Path
    stage: str
    schema: str
    spectrum_names: List[str]
    spectra: List[SpectrumPlotData]
    covariance: np.ndarray
    correlation: np.ndarray
    data_vector: np.ndarray
    ell: np.ndarray
    config_json: str


def decode_strings(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def dell_factor(ell: np.ndarray) -> np.ndarray:
    return ell * (ell + 1.0) / (2.0 * np.pi)


def load_published_gg_cls(
    path: str | Path | None = DEFAULT_PUBLISHED_GG_CLS,
    cls_key: str = "cls_ext",
) -> PublishedGgCls | None:
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.exists():
        return None
    with path.open("r") as handle:
        data = json.load(handle)
    if cls_key not in data:
        raise KeyError(f"{path} does not contain published gg spectra key {cls_key!r}; available keys are {sorted(data)}")
    ell = np.asarray(data["ell"], dtype=np.float64)
    cls_fid = data.get(cls_key, {})
    cls_by_pz: Dict[int, np.ndarray] = {}
    for pz in range(1, 5):
        key = f"s0{pz}"
        if key in cls_fid:
            cls_by_pz[pz] = np.asarray(cls_fid[key], dtype=np.float64)
    if not cls_by_pz:
        return None
    sample = "extended" if cls_key == "cls_ext" else "fiducial" if cls_key == "cls_fid" else cls_key
    return PublishedGgCls(ell=ell, cls_by_pz=cls_by_pz, cls_key=cls_key, label=f"Published {sample} incl. shot")


def desi_pz_from_spectrum_name(name: str) -> int | None:
    for token in name.split("_"):
        if token.startswith("pz"):
            try:
                return int(token[2:])
            except ValueError:
                return None
    return None


def spectrum_snr(cl: np.ndarray, cov: np.ndarray) -> float:
    finite = np.isfinite(cl) & np.all(np.isfinite(cov), axis=0) & np.all(np.isfinite(cov), axis=1)
    if not np.any(finite):
        return float("nan")
    d = cl[finite]
    c = cov[np.ix_(finite, finite)]
    c = 0.5 * (c + c.T)
    vals, vecs = np.linalg.eigh(c)
    if vals.size == 0:
        return float("nan")
    scale = np.nanmax(vals)
    if not np.isfinite(scale) or scale <= 0.0:
        return float("nan")
    keep = vals > scale * 1.0e-12
    if not np.any(keep):
        return float("nan")
    proj = vecs[:, keep].T @ d
    snr2 = np.sum((proj * proj) / vals[keep])
    return float(np.sqrt(max(snr2, 0.0)))


def robust_linear_ylim(y: np.ndarray, yerr: np.ndarray) -> tuple[float, float]:
    """Return linear y-limits that include all finite points and errorbars."""

    lower = y - yerr
    upper = y + yerr
    values = np.concatenate([lower, upper, y, np.asarray([0.0])])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if hi == lo:
        pad = max(abs(hi), 1.0) * 0.15
        return lo - pad, hi + pad
    pad = 0.12 * (hi - lo)
    return lo - pad, hi + pad


def robust_log_ylim(y: np.ndarray, yerr: np.ndarray) -> tuple[float, float]:
    """Return positive y-limits that include finite positive points/errorbars."""

    lower = y - yerr
    upper = y + yerr
    values = np.concatenate([lower, upper, y])
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0e-12, 1.0
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= 0.0:
        return 1.0e-12, 1.0
    if hi <= lo:
        return lo / 1.5, hi * 1.5
    return lo / 1.35, hi * 1.35


def nice_grid_size(n_panel: int) -> tuple[int, int]:
    if n_panel <= 0:
        return 1, 1
    n_col = min(4, int(math.ceil(math.sqrt(n_panel))))
    n_row = int(math.ceil(n_panel / n_col))
    return n_row, n_col


def load_measurement(
    path: str | Path,
    stage_label: str | None = None,
    *,
    allow_legacy_product: bool = False,
) -> MeasurementPlotData:
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as h5:
        from multiprobe_namaster import (
            DESI_GALAXY_AUTO_MEAN_CONVENTION,
            validate_measurement_product_identity,
        )

        validate_measurement_product_identity(
            h5,
            allow_legacy_product=allow_legacy_product,
        )
        schema = str(h5.attrs.get("schema", ""))
        config_json = str(h5.attrs.get("config_json", ""))
        spectrum_names = decode_strings(h5["joint/spectrum_names"][:])
        starts = h5["joint/slice_start"][:].astype(int)
        stops = h5["joint/slice_stop"][:].astype(int)
        archive_covariance = np.asarray(h5["joint/cov"][:], dtype=np.float64)
        archive_data_vector = np.asarray(h5["joint/data_vector"][:], dtype=np.float64)
        archive_raw_vector = np.asarray(
            h5["joint/data_vector_raw"][:] if "joint/data_vector_raw" in h5 else archive_data_vector,
            dtype=np.float64,
        )
        archive_valid = np.asarray(
            h5["joint/data_vector_valid"][:]
            if "joint/data_vector_valid" in h5
            else np.ones(archive_data_vector.size, dtype=bool),
            dtype=bool,
        )
        if archive_valid.shape != archive_data_vector.shape:
            raise ValueError("Saved data-vector validity mask has the wrong shape.")
        if archive_covariance.shape != (archive_data_vector.size, archive_data_vector.size):
            raise ValueError("Saved covariance shape does not match the archived data vector.")
        # Plot statistically active estimator values only. In validity-mask products,
        # joint/data_vector contains zero layout placeholders above the ACT-kappa
        # response support, while joint/data_vector_raw retains the diagnostic
        # estimator values. Neither should be displayed as a physical measurement.
        covariance = archive_covariance[np.ix_(archive_valid, archive_valid)]
        data_vector = archive_data_vector[archive_valid]
        sigma = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(sigma, sigma)
        correlation = 0.5 * (correlation + correlation.T)
        ell = h5["joint/ell"][:]
        spectra: List[SpectrumPlotData] = []
        active_cursor = 0
        for name, start, stop in zip(spectrum_names, starts, stops):
            group = h5[f"spectra/{name}"]
            family = str(group.attrs.get("family", "unknown"))
            cl_convention = str(group.attrs.get("cl_convention", ""))
            if family == "desi_g_auto":
                allowed_conventions = {
                    DESI_GALAXY_AUTO_MEAN_CONVENTION,
                    "shot_noise_subtracted_signal",
                }
                if cl_convention not in allowed_conventions:
                    raise ValueError(
                        f"DESI galaxy auto {name!r} has unknown/missing cl_convention "
                        f"{cl_convention!r}; refusing to label it signal + shot noise."
                    )
                if (
                    cl_convention != DESI_GALAXY_AUTO_MEAN_CONVENTION
                    and not allow_legacy_product
                ):
                    raise ValueError(
                        f"DESI galaxy auto {name!r} is an historical signal-only product; "
                        "pass allow_legacy_product=True only for an explicit historical plot."
                    )
                if "noise_decoupled_all_components" not in group:
                    raise ValueError(
                        f"DESI galaxy auto {name!r} has no saved shot-noise template."
                    )
            local_valid = archive_valid[start:stop]
            if not np.any(local_valid):
                raise ValueError(f"Spectrum {name!r} has no statistically valid bandpowers.")
            spec_ell = np.asarray(group["ell"][:], dtype=np.float64)[local_valid]
            cl = archive_raw_vector[start:stop][local_valid]
            archive_cov = archive_covariance[start:stop, start:stop]
            cov = archive_cov[np.ix_(local_valid, local_valid)]
            active_start = active_cursor
            active_cursor += int(np.count_nonzero(local_valid))
            active_stop = active_cursor
            component = int(group.attrs.get("component", 0))
            noise_decoupled = None
            if "noise_decoupled_all_components" in group:
                noise_all = group["noise_decoupled_all_components"][:]
                if 0 <= component < noise_all.shape[0]:
                    noise_decoupled = np.asarray(noise_all[component], dtype=np.float64)[local_valid]
            spectra.append(
                SpectrumPlotData(
                    name=name,
                    label=str(group.attrs.get("label", name)),
                    family=family,
                    ell=spec_ell,
                    cl=cl,
                    cov=cov,
                    start=int(active_start),
                    stop=int(active_stop),
                    noise_decoupled=noise_decoupled,
                    cl_convention=cl_convention,
                )
            )

    stage = stage_label
    if stage is None:
        stage = path.parent.name
    return MeasurementPlotData(
        path=path,
        stage=stage,
        schema=schema,
        spectrum_names=spectrum_names,
        spectra=spectra,
        covariance=covariance,
        correlation=correlation,
        data_vector=data_vector,
        ell=ell,
        config_json=config_json,
    )


def spectra_by_family(measurement: MeasurementPlotData) -> Dict[str, List[SpectrumPlotData]]:
    grouped: Dict[str, List[SpectrumPlotData]] = {}
    for spec in measurement.spectra:
        grouped.setdefault(spec.family, []).append(spec)
    ordered: Dict[str, List[SpectrumPlotData]] = {}
    for family in FAMILY_ORDER:
        if family in grouped:
            ordered[family] = grouped.pop(family)
    for family in sorted(grouped):
        ordered[family] = grouped[family]
    return ordered


def format_axis(ax: plt.Axes) -> None:
    if ax.get_yscale() == "linear":
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, color="#d8dbe2", linewidth=0.8, alpha=0.75)
    if ax.get_yscale() == "linear":
        ax.axhline(0.0, color="#4d4d4d", linewidth=0.8, alpha=0.65)


def plot_family_spectra(
    measurement: MeasurementPlotData,
    family: str,
    spectra: Sequence[SpectrumPlotData],
    output_dir: str | Path,
    *,
    pdf: PdfPages | None = None,
    published_gg_cls: PublishedGgCls | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_row, n_col = nice_grid_size(len(spectra))
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(4.4 * n_col, 3.35 * n_row),
        squeeze=False,
        constrained_layout=True,
    )
    color = FAMILY_COLORS.get(family, "#2f5f8f")
    family_label = FAMILY_LABELS.get(family, family)
    quantity = FAMILY_PLOT_QUANTITY.get(family, "dell")
    use_total_cl = FAMILY_USE_TOTAL_CL.get(family, False)
    y_display_scale = FAMILY_Y_DISPLAY_SCALE.get(family, 1.0)
    y_label = FAMILY_Y_DISPLAY_LABEL.get(family, r"$D_\ell = \ell(\ell+1)C_\ell/2\pi$")
    yscale = FAMILY_YSCALE.get(family, "linear")
    for ax, spec in zip(axes.flat, spectra):
        x = spec.ell
        base_cl = spec.cl_total if use_total_cl else spec.cl
        if quantity == "cl":
            y = y_display_scale * base_cl
            yerr = y_display_scale * spec.err
        else:
            y = y_display_scale * dell_factor(spec.ell) * base_cl
            yerr = y_display_scale * spec.dell_err
        this_work_label = "This work"
        if family == "desi_g_auto" and use_total_cl:
            this_work_label = "This work (signal + shot)"
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            color=color,
            ecolor=color,
            elinewidth=1.05,
            linewidth=1.25,
            markersize=3.6,
            capsize=2.4,
            alpha=0.92,
            label=this_work_label,
        )
        ylim_y = y
        ylim_yerr = yerr
        if family == "desi_g_auto" and published_gg_cls is not None:
            pz = desi_pz_from_spectrum_name(spec.name)
            if pz in published_gg_cls.cls_by_pz:
                pub_ell = published_gg_cls.ell
                pub_cl = published_gg_cls.cls_by_pz[pz]
                if quantity == "cl":
                    pub_y = y_display_scale * pub_cl
                else:
                    pub_y = y_display_scale * dell_factor(pub_ell) * pub_cl
                good = np.isfinite(pub_ell) & np.isfinite(pub_y)
                if np.any(good):
                    ax.plot(
                        pub_ell[good],
                        pub_y[good],
                        "s--",
                        color="#30343b",
                        markerfacecolor="white",
                        markeredgewidth=1.0,
                        linewidth=1.0,
                        markersize=3.2,
                        alpha=0.88,
                        label=published_gg_cls.label,
                    )
                    ylim_y = np.concatenate([ylim_y, pub_y[good]])
                    ylim_yerr = np.concatenate([ylim_yerr, np.zeros(np.count_nonzero(good), dtype=np.float64)])
        ax.set_xlim(float(np.nanmin(x)) * 0.92, float(np.nanmax(x)) * 1.04)
        ax.set_yscale(yscale)
        if family in FAMILY_YLIM:
            ax.set_ylim(*FAMILY_YLIM[family])
        elif yscale == "log":
            ax.set_ylim(*robust_log_ylim(ylim_y, ylim_yerr))
        else:
            ax.set_ylim(*robust_linear_ylim(ylim_y, ylim_yerr))
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(y_label)
        if family == "desi_g_auto":
            ax.set_title(f"{spec.label}\nsignal + shot-noise data vector", fontsize=10)
        else:
            snr = spectrum_snr(spec.cl, spec.cov)
            ax.set_title(f"{spec.label}\nS/N = {snr:.2f}", fontsize=10)
        format_axis(ax)
        if family == "desi_g_auto":
            ax.legend(loc="best", fontsize=7, frameon=False)

    for ax in axes.flat[len(spectra) :]:
        ax.set_visible(False)
    fig.suptitle(f"{measurement.stage}: {family_label}", fontsize=14, fontweight="bold")
    suffix = "Cell" if quantity == "cl" else "Dell"
    out = output_dir / f"{measurement.stage}_spectra_{family}_{suffix}.png"
    fig.savefig(out, dpi=180)
    if family == "desi_g_auto":
        legacy = output_dir / f"{measurement.stage}_spectra_{family}_Dell.png"
        if legacy != out:
            fig.savefig(legacy, dpi=180)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)
    return out


def family_block_boundaries(measurement: MeasurementPlotData) -> tuple[List[int], List[float], List[str]]:
    boundaries = [0]
    centers: List[float] = []
    labels: List[str] = []
    cursor = 0
    grouped = spectra_by_family(measurement)
    name_to_spec = {spec.name: spec for spec in measurement.spectra}
    for family, spectra in grouped.items():
        names_in_order = [name for name in measurement.spectrum_names if name_to_spec[name].family == family]
        if not names_in_order:
            continue
        start = min(name_to_spec[name].start for name in names_in_order)
        stop = max(name_to_spec[name].stop for name in names_in_order)
        if start != cursor:
            boundaries.append(start)
        centers.append(0.5 * (start + stop))
        labels.append(FAMILY_LABELS.get(family, family))
        boundaries.append(stop)
        cursor = stop
    boundaries = sorted(set(boundaries))
    return boundaries, centers, labels


def draw_block_lines(ax: plt.Axes, boundaries: Sequence[int]) -> None:
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.5, alpha=0.85)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.5, alpha=0.85)


def plot_covariance_products(measurement: MeasurementPlotData, output_dir: str | Path) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    boundaries, centers, labels = family_block_boundaries(measurement)

    cov = measurement.covariance
    finite_abs = np.abs(cov[np.isfinite(cov)])
    if finite_abs.size == 0:
        vmax = 1.0
    else:
        vmax = float(np.nanpercentile(finite_abs, 99.5))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.nanmax(finite_abs)) if finite_abs.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0

    fig, ax = plt.subplots(figsize=(10.5, 9.0), constrained_layout=True)
    im = ax.imshow(cov, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    draw_block_lines(ax, boundaries)
    ax.set_title(f"{measurement.stage}: joint covariance, linear color scale\nclipped at 99.5% of |Cov|")
    ax.set_xlabel("Data-vector index")
    ax.set_ylabel("Data-vector index")
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(centers)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.82, label="Covariance")
    out = output_dir / f"{measurement.stage}_joint_covariance_matrix.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    outputs.append(out)

    corr = np.clip(measurement.correlation, -1.0, 1.0)
    fig, ax = plt.subplots(figsize=(10.5, 9.0), constrained_layout=True)
    im = ax.imshow(corr, origin="lower", cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    draw_block_lines(ax, boundaries)
    ax.set_title(f"{measurement.stage}: joint correlation matrix")
    ax.set_xlabel("Data-vector index")
    ax.set_ylabel("Data-vector index")
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(centers)
    ax.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.82, label="Correlation")
    out = output_dir / f"{measurement.stage}_joint_correlation_matrix.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    outputs.append(out)

    diag = np.diag(measurement.covariance)
    err = np.sqrt(np.where(diag >= 0.0, diag, np.nan))
    fig, ax = plt.subplots(figsize=(12.0, 4.2), constrained_layout=True)
    ax.plot(np.arange(err.size), err, color="#2b4a67", linewidth=1.1)
    for boundary in boundaries:
        ax.axvline(boundary, color="#9aa1aa", linewidth=0.8, alpha=0.7)
    ax.set_title(f"{measurement.stage}: data-vector 1-sigma errors from joint covariance")
    ax.set_xlabel("Data-vector index")
    ax.set_ylabel(r"$\sigma(C_\ell)$")
    ax.grid(True, color="#d8dbe2", linewidth=0.8, alpha=0.75)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(formatter)
    out = output_dir / f"{measurement.stage}_joint_covariance_diagonal_errors.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    outputs.append(out)

    return outputs


def make_all_plots(
    measurement_path: str | Path,
    output_dir: str | Path,
    stage_label: str | None = None,
    published_gg_cls_path: str | Path | None = DEFAULT_PUBLISHED_GG_CLS,
    published_gg_cls_key: str = "cls_ext",
    *,
    allow_legacy_product: bool = False,
) -> List[Path]:
    measurement = load_measurement(
        measurement_path,
        stage_label=stage_label,
        allow_legacy_product=allow_legacy_product,
    )
    published_gg_cls = load_published_gg_cls(published_gg_cls_path, cls_key=published_gg_cls_key)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    pdf_path = output_dir / f"{measurement.stage}_all_spectra_Dell.pdf"
    with PdfPages(pdf_path) as pdf:
        for family, spectra in spectra_by_family(measurement).items():
            outputs.append(
                plot_family_spectra(
                    measurement,
                    family,
                    spectra,
                    output_dir,
                    pdf=pdf,
                    published_gg_cls=published_gg_cls,
                )
            )
    outputs.append(pdf_path)
    outputs.extend(plot_covariance_products(measurement, output_dir))
    return outputs


def default_measurement_for_stage(stage: str, root: str | Path = ".") -> Path:
    root = Path(root)
    if stage == "fast1024":
        return root / (
            "data/xDESI/processed/multiprobe_namaster/fast1024/"
            "xdesi_multiprobe_cls_cov_nside1024_lmax1024_nbin10_linear_pipev2_gshot.h5"
        )
    if stage == "midres2048":
        return root / (
            "data/xDESI/processed/multiprobe_namaster/midres2048/"
            "xdesi_multiprobe_cls_cov_nside2048_ell128_lmax4096_lmask6143_nbin16_log_pipev2_gshot.h5"
        )
    raise ValueError(f"No default measurement path is registered for stage {stage!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", default=None, help="Final measurement HDF5 product.")
    parser.add_argument("--stage", default="fast1024", help="Stage label used for titles and default path lookup.")
    parser.add_argument("--root", default=".", help="Repository root for default stage paths.")
    parser.add_argument("--output-dir", default=None, help="Directory for plots. Defaults to <measurement-dir>/diagnostics.")
    parser.add_argument(
        "--published-gg-cls",
        default=str(DEFAULT_PUBLISHED_GG_CLS),
        help="Optional published DESI gg auto C_ell JSON to overlay. Use an empty string to disable.",
    )
    parser.add_argument(
        "--published-gg-cls-key",
        default="cls_ext",
        help="JSON key to overlay for DESI gg auto spectra. Use cls_ext for the extended catalog or cls_fid for the fiducial LRG catalog.",
    )
    parser.add_argument(
        "--allow-legacy-product",
        action="store_true",
        help="Explicitly allow a historical signal-only product; current plots reject it by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.measurement is None:
        measurement = default_measurement_for_stage(args.stage, root=args.root)
    else:
        measurement = Path(args.measurement)
    if args.output_dir is None:
        output_dir = Path(measurement).expanduser().resolve().parent / "diagnostics"
    else:
        output_dir = Path(args.output_dir)
    published_gg_cls = args.published_gg_cls if args.published_gg_cls else None
    outputs = make_all_plots(
        measurement,
        output_dir,
        stage_label=args.stage,
        published_gg_cls_path=published_gg_cls,
        published_gg_cls_key=args.published_gg_cls_key,
        allow_legacy_product=bool(args.allow_legacy_product),
    )
    print(f"Wrote {len(outputs)} plot products to {Path(output_dir).resolve()}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
