"""Utilities for LRG, tSZ, CMB-lensing, and DES-kappa NaMaster spectra.

The notebook in this folder uses these functions to keep the data selection,
map construction, power-spectrum estimation, and Gaussian covariance assembly
in one reproducible place.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import fitsio
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pymaster as nmt
from astropy.table import Table


@dataclass
class LrgCuts:
    min_nobs: int = 2
    max_ebv: float = 0.15
    max_stardens: float = 2500.0
    stardens_nside: int = 64
    remove_ngc_islands: bool = True
    target_min_nobs: int = 1
    target_maskbits: Tuple[int, ...] = (1, 12, 13)


@dataclass
class XcorrPaths:
    lrg_dir: str = "/mnt/ceph/users/spandey/xdesi/lrg_xcorr_2023/v1"
    randoms_catalog: str = "/mnt/ceph/users/spandey/xdesi/lrg_xcorr_2023/v1/randoms-1-0.fits"
    act_lensing_dir: str = "/mnt/ceph/users/spandey/xdesi/act/lensing/baseline"
    des_kappa_dir: str = "/mnt/ceph/users/spandey/xdesi/des/kappa_WL"
    tsz_dir: str = "/mnt/ceph/users/spandey/xdesi/act/tsz/cib_cibdBeta"

    @property
    def lrg_catalog(self) -> str:
        return f"{self.lrg_dir}/catalogs/dr9_lrg_pzbins.fits"

    @property
    def lrg_weights(self) -> str:
        return f"{self.lrg_dir}/catalogs/more/dr9_lrg_pzbins-weights.fits"

    @property
    def lrg_random_mask(self) -> str:
        return f"{self.lrg_dir}/catalogs/lrgmask_v1.1/randoms-1-0-lrgmask_v1.1.fits.gz"

    @property
    def stardens_map(self) -> str:
        return f"{self.lrg_dir}/misc/pixweight-dr7.1-0.22.0_stardens_64_ring.fits"

    @property
    def cmb_kappa_alm(self) -> str:
        return f"{self.act_lensing_dir}/kappa_alm_data_act_dr6_lensing_v1_baseline.fits"

    @property
    def cmb_kappa_mask(self) -> str:
        return f"{self.act_lensing_dir}/mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits"

    @property
    def des_kappa_map(self) -> str:
        return f"{self.des_kappa_dir}/KS_tomo4.fits"

    @property
    def des_mask(self) -> str:
        return f"{self.des_kappa_dir}/glimpse_mask.fits"

    @property
    def y_map(self) -> str:
        return f"{self.tsz_dir}/ilc_actplanck_ymap_deproj_cib_cibdBeta_1.7_10.7.fits"

    @property
    def y_mask(self) -> str:
        return f"{self.tsz_dir}/wide_mask_GAL080_apod_1.50_deg_wExtended.fits"


@dataclass
class MeasurementConfig:
    nside: int = 1024
    lmax: Optional[int] = None
    nlb: int = 100
    nbins_gal: int = 4
    zmin_nz: float = 0.0
    zmax_nz: float = 1.6
    nz_nbins: int = 300
    n_iter: int = 0
    n_iter_mask: int = 0
    y_downgrade: int = 2
    y_reproject_lmax: Optional[int] = None
    cmb_kappa_mask_power: float = 2.0
    subtract_masked_mean: bool = True
    compute_full_joint_covariance: bool = True
    covariance_l_toeplitz: int = -1
    covariance_l_exact: int = -1
    covariance_dl_band: int = -1
    output_pickle: str = "outputs/lrg_tsz_shear_cls_namaster.pkl"

    def resolved_lmax(self) -> int:
        return int(self.lmax if self.lmax is not None else 2 * self.nside)

    def resolved_y_reproject_lmax(self) -> int:
        if self.y_reproject_lmax is not None:
            return int(self.y_reproject_lmax)
        return self.resolved_lmax()


@dataclass
class ScalarField:
    name: str
    label: str
    field: nmt.NmtField
    mask_name: str
    shot_noise: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SpectrumSpec:
    name: str
    family: str
    fields: Tuple[str, str]
    label: str
    title: str


def _as_path_dict(paths: XcorrPaths) -> Dict[str, str]:
    out = asdict(paths)
    for key in (
        "lrg_catalog",
        "lrg_weights",
        "lrg_random_mask",
        "stardens_map",
        "cmb_kappa_alm",
        "cmb_kappa_mask",
        "des_kappa_map",
        "des_mask",
        "y_map",
        "y_mask",
    ):
        out[key] = getattr(paths, key)
    return out


def _record_cut(
    cutflow: List[Dict[str, object]],
    step: str,
    before: int,
    mask: np.ndarray,
) -> None:
    kept = int(np.count_nonzero(mask))
    cutflow.append(
        {
            "step": step,
            "before": int(before),
            "kept": kept,
            "removed": int(before - kept),
            "kept_fraction": float(kept / before) if before > 0 else np.nan,
        }
    )


def _apply_cut(table: Table, mask: np.ndarray, step: str, cutflow: List[Dict[str, object]]) -> Table:
    mask = np.asarray(mask, dtype=bool)
    _record_cut(cutflow, step, len(table), mask)
    return table[mask]


def _stardens_good_mask(table: Table, paths: XcorrPaths, cuts: LrgCuts) -> np.ndarray:
    stardens = fitsio.read(paths.stardens_map)
    bad = stardens["STARDENS"] >= cuts.max_stardens
    bad_hp_idx = stardens["HPXPIXEL"][bad]
    cat_hp_idx = hp.ang2pix(
        cuts.stardens_nside,
        np.asarray(table["RA"]),
        np.asarray(table["DEC"]),
        lonlat=True,
        nest=False,
    )
    return ~np.isin(cat_hp_idx, bad_hp_idx)


def load_lrg_catalog(
    paths: Optional[XcorrPaths] = None,
    cuts: Optional[LrgCuts] = None,
    verbose: bool = True,
) -> Tuple[Table, List[Dict[str, object]]]:
    """Load and cut the DESI LRG catalog following the template notebook."""

    paths = paths or XcorrPaths()
    cuts = cuts or LrgCuts()
    cutflow: List[Dict[str, object]] = []

    cat = Table(fitsio.read(paths.lrg_catalog))
    weights = Table(fitsio.read(paths.lrg_weights))["weight"]
    if len(weights) != len(cat):
        raise ValueError(f"LRG weight length {len(weights)} does not match catalog length {len(cat)}")
    cat["weight"] = weights

    if cuts.remove_ngc_islands:
        mask = ~((cat["DEC"] < -10.5) & (cat["RA"] > 120) & (cat["RA"] < 260))
        cat = _apply_cut(cat, mask, "Remove NGC islands", cutflow)

    mask = (
        (cat["PIXEL_NOBS_G"] >= cuts.min_nobs)
        & (cat["PIXEL_NOBS_R"] >= cuts.min_nobs)
        & (cat["PIXEL_NOBS_Z"] >= cuts.min_nobs)
    )
    cat = _apply_cut(cat, mask, "NOBS", cutflow)

    cat = _apply_cut(cat, cat["lrg_mask"] == 0, "LRG mask", cutflow)
    cat = _apply_cut(cat, cat["EBV"] < cuts.max_ebv, "EBV", cutflow)
    cat = _apply_cut(cat, _stardens_good_mask(cat, paths, cuts), "STARDENS", cutflow)

    if verbose:
        print_cutflow("LRG data", cutflow)
        print(f"Selected LRGs: {len(cat):,}")
    return cat, cutflow


def load_lrg_randoms(
    paths: Optional[XcorrPaths] = None,
    cuts: Optional[LrgCuts] = None,
    verbose: bool = True,
) -> Tuple[Table, List[Dict[str, object]]]:
    """Load randoms and apply the same footprint and quality cuts as the LRGs."""

    paths = paths or XcorrPaths()
    cuts = cuts or LrgCuts()
    cutflow: List[Dict[str, object]] = []

    columns = ["RA", "DEC", "NOBS_G", "NOBS_R", "NOBS_Z", "MASKBITS", "EBV"]
    randoms = Table(fitsio.read(paths.randoms_catalog, columns=columns))
    lrgmask = fitsio.read(paths.lrg_random_mask, columns=["lrg_mask"])["lrg_mask"]
    if len(lrgmask) != len(randoms):
        raise ValueError(f"Random LRG-mask length {len(lrgmask)} does not match random length {len(randoms)}")
    randoms["lrg_mask"] = lrgmask

    mask = (
        (randoms["NOBS_G"] >= cuts.target_min_nobs)
        & (randoms["NOBS_R"] >= cuts.target_min_nobs)
        & (randoms["NOBS_Z"] >= cuts.target_min_nobs)
    )
    randoms = _apply_cut(randoms, mask, "Target NOBS", cutflow)

    mask = np.ones(len(randoms), dtype=bool)
    for bit in cuts.target_maskbits:
        mask &= (randoms["MASKBITS"] & 2**bit) == 0
    randoms = _apply_cut(randoms, mask, "Target MASKBITS", cutflow)

    if cuts.remove_ngc_islands:
        mask = ~((randoms["DEC"] < -10.5) & (randoms["RA"] > 120) & (randoms["RA"] < 260))
        randoms = _apply_cut(randoms, mask, "Remove NGC islands", cutflow)

    mask = (
        (randoms["NOBS_G"] >= cuts.min_nobs)
        & (randoms["NOBS_R"] >= cuts.min_nobs)
        & (randoms["NOBS_Z"] >= cuts.min_nobs)
    )
    randoms = _apply_cut(randoms, mask, "NOBS", cutflow)

    randoms = _apply_cut(randoms, randoms["lrg_mask"] == 0, "LRG mask", cutflow)
    randoms = _apply_cut(randoms, randoms["EBV"] < cuts.max_ebv, "EBV", cutflow)
    randoms = _apply_cut(randoms, _stardens_good_mask(randoms, paths, cuts), "STARDENS", cutflow)

    if verbose:
        print_cutflow("LRG randoms", cutflow)
        print(f"Selected randoms: {len(randoms):,}")
    return randoms, cutflow


def print_cutflow(name: str, cutflow: Iterable[Mapping[str, object]]) -> None:
    print(f"\n{name} cutflow")
    for row in cutflow:
        print(
            f"  {row['step']:<20s}"
            f" kept={row['kept']:>10,}"
            f" removed={row['removed']:>10,}"
            f" kept_fraction={row['kept_fraction']:.5f}"
        )


def compute_lrg_nz(
    cat: Table,
    config: Optional[MeasurementConfig] = None,
) -> Dict[str, np.ndarray]:
    config = config or MeasurementConfig()
    z_edges = np.linspace(config.zmin_nz, config.zmax_nz, config.nz_nbins)
    z_mid = 0.5 * (z_edges[1:] + z_edges[:-1])
    out: Dict[str, np.ndarray] = {"z_edges": z_edges, "z_mid": z_mid}

    for bin_index in range(1, config.nbins_gal + 1):
        sel = np.asarray(cat["pz_bin"]) == bin_index
        nz, _ = np.histogram(
            np.asarray(cat["Z_PHOT_MEDIAN"])[sel],
            bins=z_edges,
            weights=np.asarray(cat["weight"])[sel],
        )
        norm = np.sum(nz)
        out[f"bin_{bin_index}"] = nz / norm if norm > 0 else nz
    return out


def randoms_to_mask(randoms: Table, nside: int) -> np.ndarray:
    ipix = hp.ang2pix(nside, np.asarray(randoms["RA"]), np.asarray(randoms["DEC"]), lonlat=True)
    mask = np.bincount(ipix, minlength=hp.nside2npix(nside)).astype(np.float64)
    mask[mask > 0] = 1.0
    return mask


def _clean_map(map_in: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(map_in, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _clean_mask(mask_in: np.ndarray) -> np.ndarray:
    mask = _clean_map(mask_in)
    mask[mask < 0] = 0.0
    return mask


def _subtract_weighted_mean(map_in: np.ndarray, mask: np.ndarray) -> np.ndarray:
    good = mask > 0
    if not np.any(good):
        raise ValueError("Mask has no positive pixels.")
    mean = np.sum(mask[good] * map_in[good]) / np.sum(mask[good])
    out = map_in.copy()
    out[good] -= mean
    out[~good] = 0.0
    return out


def _make_nmt_field(mask: np.ndarray, map_in: np.ndarray, config: MeasurementConfig) -> nmt.NmtField:
    lmax = config.resolved_lmax()
    return nmt.NmtField(
        mask,
        [map_in],
        lmax=lmax,
        lmax_mask=lmax,
        n_iter=config.n_iter,
        n_iter_mask=config.n_iter_mask,
        lite=True,
    )


def make_lrg_field(
    cat: Table,
    random_mask: np.ndarray,
    bin_index: int,
    config: Optional[MeasurementConfig] = None,
) -> ScalarField:
    """Create a pixelized LRG overdensity field for one p(z) bin."""

    config = config or MeasurementConfig()
    nside = config.nside
    lmax = config.resolved_lmax()
    npix = hp.nside2npix(nside)
    if len(random_mask) != npix:
        raise ValueError(f"Random mask has length {len(random_mask)}, expected {npix}.")

    sel = np.asarray(cat["pz_bin"]) == bin_index
    if not np.any(sel):
        raise ValueError(f"No LRGs found for pz_bin={bin_index}")

    ra = np.asarray(cat["RA"])[sel]
    dec = np.asarray(cat["DEC"])[sel]
    weights = np.asarray(cat["weight"], dtype=np.float64)[sel]
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights /= np.mean(weights[weights > 0])

    ipix = hp.ang2pix(nside, ra, dec, lonlat=True)
    nmap = np.bincount(ipix, minlength=npix, weights=weights).astype(np.float64)

    mask = _clean_mask(random_mask)
    good = mask > 0
    area = hp.nside2pixarea(nside) * np.sum(mask)
    sumw = np.sum(weights)
    sumw2 = np.sum(weights**2)
    if area <= 0 or sumw <= 0:
        raise ValueError(f"Invalid LRG area or weight sum for bin {bin_index}.")

    nbar_pix = np.sum(nmap[good]) / np.sum(mask[good])
    expected = nbar_pix * mask
    delta = np.zeros(npix, dtype=np.float64)
    delta[good] = nmap[good] / expected[good] - 1.0
    delta[~good] = 0.0

    shot_noise = area * sumw2 / sumw**2
    field_obj = _make_nmt_field(mask, delta, config)
    return ScalarField(
        name=f"g{bin_index}",
        label=f"LRG bin {bin_index}",
        field=field_obj,
        mask_name="lrg",
        shot_noise=float(shot_noise),
        metadata={
            "bin_index": int(bin_index),
            "n_gal": int(np.count_nonzero(sel)),
            "sum_weight": float(sumw),
            "sum_weight2": float(sumw2),
            "area_sr": float(area),
            "fsky": float(area / (4.0 * np.pi)),
            "nbar_per_sr": float(sumw / area),
            "nbar_per_pix": float(nbar_pix),
            "shot_noise": float(shot_noise),
            "lmax": int(lmax),
        },
    )


def load_cmb_kappa_field(
    paths: Optional[XcorrPaths] = None,
    config: Optional[MeasurementConfig] = None,
) -> ScalarField:
    paths = paths or XcorrPaths()
    config = config or MeasurementConfig()
    lmax = config.resolved_lmax()

    alm = hp.read_alm(paths.cmb_kappa_alm)
    alm = np.nan_to_num(alm.astype(np.complex128), nan=0.0, posinf=0.0, neginf=0.0)
    alm_lmax = hp.Alm.getlmax(len(alm))
    map_lmax = min(lmax, alm_lmax)
    if map_lmax < alm_lmax:
        alm = hp.resize_alm(alm, alm_lmax, alm_lmax, map_lmax, map_lmax)
    kappa = hp.alm2map(alm, config.nside, lmax=map_lmax)
    kappa = _clean_map(kappa)

    mask = hp.read_map(paths.cmb_kappa_mask)
    mask = hp.ud_grade(mask, nside_out=config.nside)
    mask = _clean_mask(mask) ** config.cmb_kappa_mask_power
    if config.subtract_masked_mean:
        kappa = _subtract_weighted_mean(kappa, mask)

    return ScalarField(
        name="kappa_CMB",
        label=r"$\kappa_{\rm CMB}$",
        field=_make_nmt_field(mask, kappa, config),
        mask_name="kappa_CMB",
        metadata={
            "source": paths.cmb_kappa_alm,
            "mask": paths.cmb_kappa_mask,
            "alm_lmax": int(alm_lmax),
            "map_lmax": int(map_lmax),
            "mask_power": float(config.cmb_kappa_mask_power),
            "fsky": float(np.mean(mask > 0)),
            "weighted_fsky": float(np.mean(mask)),
        },
    )


def load_des_kappa_field(
    paths: Optional[XcorrPaths] = None,
    config: Optional[MeasurementConfig] = None,
) -> ScalarField:
    paths = paths or XcorrPaths()
    config = config or MeasurementConfig()
    kappa = hp.ud_grade(hp.read_map(paths.des_kappa_map), nside_out=config.nside)
    mask = hp.ud_grade(hp.read_map(paths.des_mask), nside_out=config.nside)
    kappa = _clean_map(kappa)
    mask = _clean_mask(mask)
    if config.subtract_masked_mean:
        kappa = _subtract_weighted_mean(kappa, mask)

    return ScalarField(
        name="kappa_DES",
        label=r"$\kappa_{\rm DES}$",
        field=_make_nmt_field(mask, kappa, config),
        mask_name="kappa_DES",
        metadata={
            "source": paths.des_kappa_map,
            "mask": paths.des_mask,
            "fsky": float(np.mean(mask > 0)),
            "weighted_fsky": float(np.mean(mask)),
        },
    )


def load_tsz_y_field(
    paths: Optional[XcorrPaths] = None,
    config: Optional[MeasurementConfig] = None,
) -> ScalarField:
    paths = paths or XcorrPaths()
    config = config or MeasurementConfig()
    try:
        from pixell import enmap, reproject
    except ImportError as exc:
        raise ImportError("pixell is required to read and reproject the ACT y-map.") from exc

    ymask = enmap.read_map(paths.y_mask)
    ymap = enmap.read_map(paths.y_map)
    if config.y_downgrade and config.y_downgrade > 1:
        ymask = enmap.downgrade(ymask, config.y_downgrade)
        ymap = enmap.downgrade(ymap, config.y_downgrade)

    reproj_lmax = config.resolved_y_reproject_lmax()
    ymask_hp = reproject.map2healpix(ymask, nside=config.nside, lmax=reproj_lmax)
    ymap_hp = reproject.map2healpix(ymap, nside=config.nside, lmax=reproj_lmax)
    ymask_hp = _clean_mask(ymask_hp)
    ymap_hp = _clean_map(ymap_hp)
    if config.subtract_masked_mean:
        ymap_hp = _subtract_weighted_mean(ymap_hp, ymask_hp)

    return ScalarField(
        name="y",
        label="Compton-y",
        field=_make_nmt_field(ymask_hp, ymap_hp, config),
        mask_name="y",
        metadata={
            "source": paths.y_map,
            "mask": paths.y_mask,
            "downgrade": int(config.y_downgrade),
            "reproject_lmax": int(reproj_lmax),
            "fsky": float(np.mean(ymask_hp > 0)),
            "weighted_fsky": float(np.mean(ymask_hp)),
        },
    )


def build_scalar_fields(
    cat: Table,
    randoms: Table,
    paths: Optional[XcorrPaths] = None,
    config: Optional[MeasurementConfig] = None,
    verbose: bool = True,
) -> Dict[str, ScalarField]:
    paths = paths or XcorrPaths()
    config = config or MeasurementConfig()
    fields: Dict[str, ScalarField] = {}

    if verbose:
        print("Building LRG random footprint mask")
    random_mask = randoms_to_mask(randoms, config.nside)
    for bin_index in range(1, config.nbins_gal + 1):
        if verbose:
            print(f"Building LRG overdensity field for bin {bin_index}")
        sf = make_lrg_field(cat, random_mask, bin_index, config)
        fields[sf.name] = sf

    if verbose:
        print("Loading ACT DR6 CMB-lensing kappa field")
    fields["kappa_CMB"] = load_cmb_kappa_field(paths, config)

    if verbose:
        print("Loading DES kappa field")
    fields["kappa_DES"] = load_des_kappa_field(paths, config)

    if verbose:
        print("Loading ACT+Planck Compton-y field")
    fields["y"] = load_tsz_y_field(paths, config)

    return fields


def default_target_spectra(nbins_gal: int = 4) -> List[SpectrumSpec]:
    specs: List[SpectrumSpec] = []
    for i in range(1, nbins_gal + 1):
        specs.append(
            SpectrumSpec(
                name=f"Cl_gg_bin{i}",
                family="Cl_gg",
                fields=(f"g{i}", f"g{i}"),
                label=rf"$C_\ell^{{g_{i}g_{i}}}$",
                title=f"LRG bin {i}",
            )
        )
    for i in range(1, nbins_gal + 1):
        specs.append(
            SpectrumSpec(
                name=f"Cl_gkappa_DES_bin{i}",
                family="Cl_gkappa_DES",
                fields=(f"g{i}", "kappa_DES"),
                label=rf"$C_\ell^{{g_{i}\kappa_{{\rm DES}}}}$",
                title=f"LRG bin {i}",
            )
        )
    for i in range(1, nbins_gal + 1):
        specs.append(
            SpectrumSpec(
                name=f"Cl_gkappa_CMB_bin{i}",
                family="Cl_gkappa_CMB",
                fields=(f"g{i}", "kappa_CMB"),
                label=rf"$C_\ell^{{g_{i}\kappa_{{\rm CMB}}}}$",
                title=f"LRG bin {i}",
            )
        )
    for i in range(1, nbins_gal + 1):
        specs.append(
            SpectrumSpec(
                name=f"Cl_yg_bin{i}",
                family="Cl_yg",
                fields=("y", f"g{i}"),
                label=rf"$C_\ell^{{y g_{i}}}$",
                title=f"LRG bin {i}",
            )
        )
    specs.append(
        SpectrumSpec(
            name="Cl_ykappa_DES",
            family="Cl_ykappa_DES",
            fields=("y", "kappa_DES"),
            label=r"$C_\ell^{y\kappa_{\rm DES}}$",
            title=r"$y \times \kappa_{\rm DES}$",
        )
    )
    specs.append(
        SpectrumSpec(
            name="Cl_yy",
            family="Cl_yy",
            fields=("y", "y"),
            label=r"$C_\ell^{yy}$",
            title=r"$y \times y$",
        )
    )
    return specs


def make_bins(config: Optional[MeasurementConfig] = None) -> nmt.NmtBin:
    config = config or MeasurementConfig()
    return nmt.NmtBin.from_lmax_linear(config.resolved_lmax(), nlb=config.nlb)


def _field_pair_key(a: ScalarField, b: ScalarField) -> Tuple[str, str]:
    return (a.mask_name, b.mask_name)


def _mean_mask_product(a: nmt.NmtField, b: nmt.NmtField) -> float:
    mean = float(np.mean(a.get_mask() * b.get_mask()))
    if mean <= 0:
        raise ValueError("Two fields have zero mask overlap.")
    return mean


def _get_workspace(
    spec_fields: Tuple[str, str],
    fields: Mapping[str, ScalarField],
    bins: nmt.NmtBin,
    cache: MutableMapping[Tuple[str, str], nmt.NmtWorkspace],
) -> nmt.NmtWorkspace:
    a = fields[spec_fields[0]]
    b = fields[spec_fields[1]]
    key = _field_pair_key(a, b)
    if key not in cache:
        cache[key] = nmt.NmtWorkspace.from_fields(a.field, b.field, bins)
    return cache[key]


def compute_pair_input_cls(fields: Mapping[str, ScalarField], verbose: bool = True) -> Dict[Tuple[str, str], np.ndarray]:
    """Compute full-resolution input spectra used by the Gaussian covariance.

    These are pseudo-Cl estimates divided by the mean product of the masks,
    following the improved narrow-kernel approximation recommended in the
    NaMaster covariance documentation. Galaxy auto-spectra include their
    Poisson shot noise because covariance inputs must be total spectra.
    """

    names = list(fields)
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for i, name_a in enumerate(names):
        for name_b in names[i:]:
            fa = fields[name_a].field
            fb = fields[name_b].field
            cl = nmt.compute_coupled_cell(fa, fb)
            cl = cl / _mean_mask_product(fa, fb)
            out[(name_a, name_b)] = cl
            out[(name_b, name_a)] = cl
            if verbose:
                print(f"Input Cl for covariance: {name_a} x {name_b}")
    return out


def _shot_noise_coupled(sf: ScalarField, workspace: nmt.NmtWorkspace, lmax: int) -> Optional[np.ndarray]:
    if sf.shot_noise <= 0:
        return None
    return workspace.couple_cell(np.full((1, lmax + 1), sf.shot_noise, dtype=np.float64))


def measure_spectrum(
    spec: SpectrumSpec,
    fields: Mapping[str, ScalarField],
    bins: nmt.NmtBin,
    workspace_cache: MutableMapping[Tuple[str, str], nmt.NmtWorkspace],
    config: Optional[MeasurementConfig] = None,
) -> Dict[str, object]:
    config = config or MeasurementConfig()
    lmax = config.resolved_lmax()
    field_a = fields[spec.fields[0]]
    field_b = fields[spec.fields[1]]
    workspace = _get_workspace(spec.fields, fields, bins, workspace_cache)

    pcl = nmt.compute_coupled_cell(field_a.field, field_b.field)
    noise_coupled = None
    shot_noise = 0.0
    if spec.fields[0] == spec.fields[1] and field_a.shot_noise > 0:
        noise_coupled = _shot_noise_coupled(field_a, workspace, lmax)
        shot_noise = field_a.shot_noise

    cl = workspace.decouple_cell(pcl, cl_noise=noise_coupled)[0]
    noise_bandpower = None
    if noise_coupled is not None:
        noise_bandpower = workspace.decouple_cell(noise_coupled)[0]

    window = workspace.get_bandpower_windows()[0, :, 0, :]
    return {
        "name": spec.name,
        "family": spec.family,
        "fields": spec.fields,
        "label": spec.label,
        "title": spec.title,
        "ell": bins.get_effective_ells(),
        "cl": cl,
        "shot_noise": float(shot_noise),
        "shot_noise_bandpower": noise_bandpower,
        "bandpower_window": window,
        "mask_names": (field_a.mask_name, field_b.mask_name),
    }


def _get_covariance_workspace(
    spec_a: SpectrumSpec,
    spec_b: SpectrumSpec,
    fields: Mapping[str, ScalarField],
    config: MeasurementConfig,
    cache: MutableMapping[Tuple[str, str, str, str], nmt.NmtCovarianceWorkspace],
) -> nmt.NmtCovarianceWorkspace:
    a1, a2 = (fields[spec_a.fields[0]], fields[spec_a.fields[1]])
    b1, b2 = (fields[spec_b.fields[0]], fields[spec_b.fields[1]])
    key = (a1.mask_name, a2.mask_name, b1.mask_name, b2.mask_name)
    if key not in cache:
        cache[key] = nmt.NmtCovarianceWorkspace.from_fields(
            a1.field,
            a2.field,
            b1.field,
            b2.field,
            l_toeplitz=config.covariance_l_toeplitz,
            l_exact=config.covariance_l_exact,
            dl_band=config.covariance_dl_band,
            spin0_only=True,
        )
    return cache[key]


def compute_covariance_block(
    spec_a: SpectrumSpec,
    spec_b: SpectrumSpec,
    fields: Mapping[str, ScalarField],
    input_cls: Mapping[Tuple[str, str], np.ndarray],
    bins: nmt.NmtBin,
    workspace_cache: MutableMapping[Tuple[str, str], nmt.NmtWorkspace],
    cov_workspace_cache: MutableMapping[Tuple[str, str, str, str], nmt.NmtCovarianceWorkspace],
    config: Optional[MeasurementConfig] = None,
) -> np.ndarray:
    config = config or MeasurementConfig()
    a1, a2 = spec_a.fields
    b1, b2 = spec_b.fields
    wa = _get_workspace(spec_a.fields, fields, bins, workspace_cache)
    wb = _get_workspace(spec_b.fields, fields, bins, workspace_cache)
    cw = _get_covariance_workspace(spec_a, spec_b, fields, config, cov_workspace_cache)

    cov = nmt.gaussian_covariance(
        cw,
        0,
        0,
        0,
        0,
        input_cls[(a1, b1)],
        input_cls[(a1, b2)],
        input_cls[(a2, b1)],
        input_cls[(a2, b2)],
        wa,
        wb,
        coupled=False,
    )
    if cov.ndim == 4:
        cov = cov[:, 0, :, 0]
    return np.asarray(cov, dtype=np.float64)


def measure_target_spectra_and_covariances(
    fields: Mapping[str, ScalarField],
    config: Optional[MeasurementConfig] = None,
    specs: Optional[List[SpectrumSpec]] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    config = config or MeasurementConfig()
    specs = specs or default_target_spectra(config.nbins_gal)
    bins = make_bins(config)
    ell = bins.get_effective_ells()
    workspace_cache: Dict[Tuple[str, str], nmt.NmtWorkspace] = {}
    cov_workspace_cache: Dict[Tuple[str, str, str, str], nmt.NmtCovarianceWorkspace] = {}

    if verbose:
        print("Computing full-resolution spectra for covariance inputs")
    input_cls = compute_pair_input_cls(fields, verbose=verbose)

    spectra: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        if verbose:
            print(f"Measuring {spec.name}")
        spectra[spec.name] = measure_spectrum(spec, fields, bins, workspace_cache, config)

    blocks: Dict[Tuple[str, str], np.ndarray] = {}
    if config.compute_full_joint_covariance:
        n_per = len(ell)
        n_data = n_per * len(specs)
        joint_cov = np.zeros((n_data, n_data), dtype=np.float64)
        slices: Dict[str, Tuple[int, int]] = {}
        for i, spec in enumerate(specs):
            slices[spec.name] = (i * n_per, (i + 1) * n_per)

        for i, spec_i in enumerate(specs):
            for j, spec_j in enumerate(specs[i:], start=i):
                if verbose:
                    print(f"Covariance block: {spec_i.name} x {spec_j.name}")
                block = compute_covariance_block(
                    spec_i,
                    spec_j,
                    fields,
                    input_cls,
                    bins,
                    workspace_cache,
                    cov_workspace_cache,
                    config,
                )
                blocks[(spec_i.name, spec_j.name)] = block
                if spec_i.name == spec_j.name:
                    spectra[spec_i.name]["cov"] = block
                    spectra[spec_i.name]["err"] = np.sqrt(np.clip(np.diag(block), 0.0, np.inf))
                s_i = slice(*slices[spec_i.name])
                s_j = slice(*slices[spec_j.name])
                joint_cov[s_i, s_j] = block
                if i != j:
                    joint_cov[s_j, s_i] = block.T
        joint = {
            "spectrum_names": [spec.name for spec in specs],
            "ell": ell,
            "data_vector": np.concatenate([np.asarray(spectra[spec.name]["cl"]) for spec in specs]),
            "cov": joint_cov,
            "slices": slices,
        }
    else:
        joint = None
        for spec in specs:
            if verbose:
                print(f"Covariance block: {spec.name}")
            block = compute_covariance_block(
                spec,
                spec,
                fields,
                input_cls,
                bins,
                workspace_cache,
                cov_workspace_cache,
                config,
            )
            spectra[spec.name]["cov"] = block
            spectra[spec.name]["err"] = np.sqrt(np.clip(np.diag(block), 0.0, np.inf))
            blocks[(spec.name, spec.name)] = block

    return {
        "ell": ell,
        "spectra": spectra,
        "covariance_blocks": blocks,
        "joint": joint,
        "input_cls_for_covariance": input_cls,
        "workspace_mask_pair_keys": list(workspace_cache.keys()),
        "covariance_mask_quad_keys": list(cov_workspace_cache.keys()),
    }


def run_lrg_tsz_shear_measurement(
    paths: Optional[XcorrPaths] = None,
    cuts: Optional[LrgCuts] = None,
    config: Optional[MeasurementConfig] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    paths = paths or XcorrPaths()
    cuts = cuts or LrgCuts()
    config = config or MeasurementConfig()

    cat, data_cutflow = load_lrg_catalog(paths, cuts, verbose=verbose)
    randoms, random_cutflow = load_lrg_randoms(paths, cuts, verbose=verbose)
    nz = compute_lrg_nz(cat, config)
    fields = build_scalar_fields(cat, randoms, paths, config, verbose=verbose)
    measured = measure_target_spectra_and_covariances(fields, config, verbose=verbose)

    field_metadata = {
        name: {
            "label": sf.label,
            "mask_name": sf.mask_name,
            "shot_noise": sf.shot_noise,
            "metadata": sf.metadata,
        }
        for name, sf in fields.items()
    }
    measured.update(
        {
            "metadata": {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "paths": _as_path_dict(paths),
                "cuts": asdict(cuts),
                "config": asdict(config),
                "field_metadata": field_metadata,
                "lrg_data_cutflow": data_cutflow,
                "lrg_random_cutflow": random_cutflow,
                "notes": (
                    "All fields are spin-0 HEALPix NaMaster fields. LRG spectra "
                    "are measured from pixelized overdensity maps selected with "
                    "the same cuts as the template notebook. Galaxy auto-spectra "
                    "are shot-noise subtracted for the saved Cl values; covariance "
                    "inputs use total spectra."
                ),
            },
            "nz": nz,
        }
    )
    return measured


def save_measurement_pickle(result: Mapping[str, object], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dict(result), f, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def plot_nz(nz: Mapping[str, np.ndarray], nbins_gal: int = 4, ax: Optional[plt.Axes] = None) -> plt.Axes:
    ax = ax or plt.gca()
    for i in range(1, nbins_gal + 1):
        ax.plot(nz["z_mid"], nz[f"bin_{i}"], lw=2.0, label=f"LRG bin {i}")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"Normalized $n(z)$")
    ax.set_xlim(0.3, 1.2)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", alpha=0.25, lw=0.8)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


_SPECTRUM_FAMILY_STYLE = {
    "Cl_gg": {
        "title": r"LRG auto-correlation: $C_\ell^{gg}$",
        "color": "#1f77b4",
    },
    "Cl_gkappa_DES": {
        "title": r"LRG x DES shear convergence: $C_\ell^{g\kappa_{\rm DES}}$",
        "color": "#2ca02c",
    },
    "Cl_gkappa_CMB": {
        "title": r"LRG x CMB lensing convergence: $C_\ell^{g\kappa_{\rm CMB}}$",
        "color": "#9467bd",
    },
    "Cl_yg": {
        "title": r"Compton-y x LRG: $C_\ell^{y g}$",
        "color": "#d62728",
    },
    "Cl_ykappa_DES": {
        "title": r"Compton-y x DES shear convergence: $C_\ell^{y\kappa_{\rm DES}}$",
        "color": "#ff7f0e",
    },
    "Cl_yy": {
        "title": r"Compton-y auto-correlation: $C_\ell^{yy}$",
        "color": "#17becf",
    },
}


def _positive_y_limits(cl: np.ndarray, err: np.ndarray) -> Tuple[str, float, float, bool]:
    """Choose a non-negative y-axis range that emphasizes the bandpower shape."""

    cl = np.asarray(cl, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    finite = np.isfinite(cl) & np.isfinite(err)
    if not np.any(finite):
        return "linear", 0.0, 1.0, False

    cl_f = cl[finite]
    err_f = np.clip(err[finite], 0.0, np.inf)
    upper = cl_f + err_f
    positive_upper = upper[upper > 0]
    positive_cl = cl_f[cl_f > 0]
    has_nonpositive = np.any(cl_f <= 0)
    lower_1sigma_positive = np.all(cl_f - err_f > 0)

    if positive_upper.size:
        y_top = float(np.nanpercentile(positive_upper, 98))
    elif positive_cl.size:
        y_top = float(np.nanpercentile(positive_cl, 98))
    else:
        y_top = float(np.nanpercentile(err_f[err_f > 0], 98)) if np.any(err_f > 0) else 1.0
    if not np.isfinite(y_top) or y_top <= 0:
        y_top = 1.0
    y_top *= 1.25

    if (not has_nonpositive) and lower_1sigma_positive and positive_cl.size:
        dynamic_range = np.nanmax(positive_cl) / max(np.nanmin(positive_cl), 1e-300)
        if dynamic_range > 8:
            y_bottom = float(np.nanpercentile(positive_cl, 2) / 1.6)
            y_bottom = max(y_bottom, y_top / 1.0e5, 1.0e-300)
            return "log", y_bottom, y_top, False

    return "linear", 0.0, y_top, bool(has_nonpositive)


def _style_power_axis(ax: plt.Axes, scale: str, y_bottom: float, y_top: float) -> None:
    ax.set_xscale("log")
    ax.set_yscale(scale)
    ax.set_ylim(y_bottom, y_top)
    ax.grid(True, which="major", alpha=0.25, lw=0.8)
    if scale == "log":
        ax.grid(True, which="minor", alpha=0.12, lw=0.5)
    ax.tick_params(direction="in", which="both", top=False, right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_spectrum_family(
    result: Mapping[str, object],
    family: str,
    ncols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    spectra = result["spectra"]
    names = [name for name, spec in spectra.items() if spec["family"] == family]
    if not names:
        raise KeyError(f"No spectra found for family {family}")

    ncols = min(ncols, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    figsize = figsize or (5.0 * ncols, 3.8 * nrows)
    style = _SPECTRUM_FAMILY_STYLE.get(family, {"title": family, "color": "#1f77b4"})
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, name in zip(axes_flat, names):
        spec = spectra[name]
        ell = np.asarray(spec["ell"])
        cl = np.asarray(spec["cl"])
        err = np.asarray(spec["err"])
        scale, y_bottom, y_top, has_nonpositive = _positive_y_limits(cl, err)
        color = style["color"]
        ax.errorbar(
            ell,
            cl,
            yerr=err,
            fmt="o-",
            ms=4.0,
            lw=1.6,
            capsize=2.5,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            alpha=0.95,
        )
        if has_nonpositive:
            nonpositive = np.isfinite(cl) & (cl <= 0)
            ax.scatter(
                ell[nonpositive],
                np.zeros(np.count_nonzero(nonpositive)),
                marker="v",
                s=30,
                facecolors="white",
                edgecolors="0.25",
                linewidths=0.8,
                zorder=4,
                label=r"$C_\ell \leq 0$",
            )
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        _style_power_axis(ax, scale, y_bottom, y_top)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(spec["label"])
        ax.set_title(spec["title"], loc="left", fontsize=11, fontweight="semibold")
    for ax in axes_flat[len(names) :]:
        ax.axis("off")
    fig.suptitle(style["title"], fontsize=15, fontweight="semibold")
    return fig, axes


def plot_all_target_spectra(result: Mapping[str, object]) -> Dict[str, Tuple[plt.Figure, np.ndarray]]:
    figures = {}
    for family in ("Cl_gg", "Cl_gkappa_DES", "Cl_gkappa_CMB", "Cl_yg", "Cl_ykappa_DES", "Cl_yy"):
        figures[family] = plot_spectrum_family(result, family)
    return figures
