"""Multi-probe NaMaster measurements for the xDESI survey transfer bundle.

This module implements the two-stage harmonic-space pipeline described in the
survey measurement plan.  It deliberately keeps the expensive map-preparation
step separate from the NaMaster measurement step, so low-resolution products
can be inspected and reused before launching production measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import healpy as hp
import numpy as np
import pymaster as nmt
from astropy.io import fits
from astropy.wcs import WCS
from pixell import enmap, reproject

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover - optional smoothing dependency
    gaussian_filter1d = None


TCMB_UK = 2.7255e6
ACT_TSZ_BEAM_FWHM_ARCMIN = 1.6
ACT_CMB_TEMPERATURE_BEAM_FWHM_ARCMIN = 1.6
KSZ_PHOTOZ_VELOCITY_CORRELATION_R = 0.3
KSZ_PHOTOZ_VELOCITY_CORRELATION_FRACERR = 0.10
KSZ_SPECTRO_VELOCITY_CORRELATION_R = 0.64
KSZ_REFERENCE_PAPER = "notebooks/xDESI/papers/ksz/2407.07152v2.pdf"
KSZ_SIGMA_TRUE_GAS_DOC = "data/xDESI/survey_data/docs/DESI_ABACUS_SIGMA_TRUE_GAS.md"
KSZ_SIGMA_TRUE_GAS_JSON_REL = (
    "data/desi_abacus_velocity_calibration/"
    "sigma_true_gas_abacus_extended_lrg_zerr0p0_ph201_photometric_bins.json"
)
KSZ_SIGMA_TRUE_GAS_OVER_C_3E5 = {
    1: 0.001055808793736867,
    2: 0.001049158647903646,
    3: 0.0010358254819693483,
    4: 0.001017605495377451,
}
KSZ_SIGMA_TRUE_GAS_KM_S = {
    1: 316.74263812106005,
    2: 314.7475943710938,
    3: 310.7476445908045,
    4: 305.2816486132353,
}
DES_Y3_SOURCE_NZ_FITS_DEFAULT = (
    "/mnt/ceph/users/spandey/GODMAX/data/DESxACT/"
    "2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"
)
DES_Y3_SOURCE_NZ_HDU = "nz_source"


def _trapz(y: np.ndarray, x: Optional[np.ndarray] = None, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)
DES_Y3_GAUSSIAN_PRIORS = {
    "Delta_z_bias_bin1": (0.0, 1.8e-2),
    "Delta_z_bias_bin2": (0.0, 1.5e-2),
    "Delta_z_bias_bin3": (0.0, 1.1e-2),
    "Delta_z_bias_bin4": (0.0, 1.7e-2),
    "mult_shear_bias_bin1": (-6.0e-3, 9.0e-3),
    "mult_shear_bias_bin2": (-2.0e-2, 8.0e-3),
    "mult_shear_bias_bin3": (-2.4e-2, 8.0e-3),
    "mult_shear_bias_bin4": (-3.7e-2, 8.0e-3),
}
DESI_DR9_SELECTION_DATASET = "catalog/valid_for_cl"
DESI_DR9_WEIGHT_DATASET = "catalog/weight_imaging_mean1"
DESI_DR9_TRUE_NZ_HDF5_REL = "data/desi_dr9_redshift_distributions/desi_dr9_extended_lrg_sigmaz0p05_true_nz.h5"
DESI_DR9_TRUE_NZ_GROUP_FULL_CL = "zphot_std0p05_spec_ratio_corrected"
DESI_DR9_TRUE_NZ_DATASET = "nz_unit_integral"
DESI_DR9_SUPPORTED_RANDOM_NSIDE = (1024, 4096)
DESI_DR9_RANDOM_DERIVED_NSIDE = (2048,)
SCHEMA_MAPS = "xdesi_multiprobe_maps_v1"
SCHEMA_MEASUREMENT = "xdesi_multiprobe_measurement_v1"
SCHEMA_MEASUREMENT_VALIDITY_MASK = "xdesi_multiprobe_measurement_v2"
MEASUREMENT_PIPELINE_VERSION = "namaster_v2"
MAP_CONSTRUCTION_VERSION = "shear_act_desi_masks_v2"
SPECTRUM_ESTIMATOR_VERSION = "catalog_noise_and_transfer_v2"
COVARIANCE_ESTIMATOR_VERSION = "inka_data_v1"
DESI_GALAXY_AUTO_MEAN_CONVENTION = "signal_plus_weighted_poisson_shot_noise"
DESI_GALAXY_AUTO_PRODUCT_TAG = "gshot"
DESI_GALAXY_AUTO_VIEWS_CONTRACT_VERSION = "conditional_weighted_poisson_views_v1"
DESI_GALAXY_AUTO_PRIMARY_VIEW = "total"
DESI_GALAXY_AUTO_SUBTRACTED_VIEW = "weighted_poisson_subtracted"
DATA_VECTOR_VALIDITY_CONTRACT_VERSION = "kappa_band_validity_v1"
ACT_CMB_TEMPERATURE_UNITS = "uK_CMB"
DESI_GALAXY_AUTO_THEORY_INTERFACE_NOTE = (
    "For DESI galaxy autos, window the clustering signal with the saved signal transfers "
    "and then add A_shot,pz times spectra/<name>/noise_decoupled_all_components[component]. "
    "That template is already decoupled; do not window it or add it to the measurement again."
)


@dataclass(frozen=True)
class SurveyBundle:
    """Resolved paths for products inside ``data/xDESI/survey_data``."""

    root: Path
    manifest: Mapping[str, object]
    desi_catalog: Path
    desi_randoms: Path
    desi_random_count_maps: Path
    shear_nside1024: Path
    shear_nside2048: Path
    shear_nside4096: Path
    act_y: Path
    act_cmb: Path
    act_kappa: Path
    desi_true_nz: Path
    sigma_true_gas_calibration: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "SurveyBundle":
        root = Path(root).resolve()
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        products = manifest["products"]
        shear_products = products["des_y3_shear_maps"]
        desi_products = products["desi_dr9_extended_velocity_catalogs"]
        random_products = products["desi_dr9_imaging_randoms"]
        nz_products = products.get("desi_dr9_redshift_distributions", {})
        true_nz_rel = nz_products.get("extended_lrg_sigmaz0p05_true_nz_hdf5", DESI_DR9_TRUE_NZ_HDF5_REL)
        return cls(
            root=root,
            manifest=manifest,
            desi_catalog=root / desi_products["combined"],
            desi_randoms=root / random_products["quality_cut_randoms"],
            desi_random_count_maps=root / random_products["count_maps_nside1024_4096"],
            shear_nside1024=root / shear_products["nside1024"],
            shear_nside2048=root / shear_products.get("nside2048", shear_products["nside4096"]),
            shear_nside4096=root / shear_products["nside4096"],
            act_y=root / products["act_dr6_tsz_compton_y"],
            act_cmb=root / products["act_dr6_cmb_temperature"],
            act_kappa=root / products["act_dr6_lensing_kappa"],
            desi_true_nz=root / true_nz_rel,
            sigma_true_gas_calibration=root / KSZ_SIGMA_TRUE_GAS_JSON_REL,
        )

    def shear_path_for_nside(self, nside: int) -> Path:
        if int(nside) == 1024:
            return self.shear_nside1024
        if int(nside) == 2048:
            return self.shear_nside2048
        if int(nside) == 4096:
            return self.shear_nside4096
        raise ValueError(f"No transferred DES shear product exists for nside={nside}.")

    def validate_files(self) -> Dict[str, Dict[str, object]]:
        paths = {
            "manifest": self.root / "manifest.json",
            "desi_catalog": self.desi_catalog,
            "desi_randoms": self.desi_randoms,
            "desi_random_count_maps": self.desi_random_count_maps,
            "shear_nside1024": self.shear_nside1024,
            "shear_nside2048": self.shear_nside2048,
            "shear_nside4096": self.shear_nside4096,
            "act_y": self.act_y,
            "act_cmb": self.act_cmb,
            "act_kappa": self.act_kappa,
            "desi_true_nz": self.desi_true_nz,
        }
        out: Dict[str, Dict[str, object]] = {}
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing required input {name}: {path}")
            stat = path.stat()
            out[name] = {
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        self._validate_dr9_catalog_schema()
        self._validate_dr9_random_count_schema()
        random_meta = self.manifest["products"]["desi_dr9_imaging_randoms"]
        if isinstance(random_meta, Mapping) and "sha256" in random_meta:
            out["desi_random_count_maps"]["manifest_sha256"] = str(random_meta["sha256"])
        if self.sigma_true_gas_calibration.exists():
            stat = self.sigma_true_gas_calibration.stat()
            out["sigma_true_gas_calibration"] = {
                "path": str(self.sigma_true_gas_calibration),
                "size_bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        return out

    def _validate_dr9_catalog_schema(self) -> None:
        required = (
            "catalog/ra_deg",
            "catalog/dec_deg",
            "catalog/z",
            "catalog/vr_over_c",
            "catalog/pz_bin",
            DESI_DR9_SELECTION_DATASET,
            DESI_DR9_WEIGHT_DATASET,
        )
        with h5py.File(self.desi_catalog, "r") as h5:
            missing = [name for name in required if name not in h5]
        if missing:
            raise KeyError(f"DESI DR9 catalog is missing required dataset(s): {missing}")

    def _validate_dr9_random_count_schema(self) -> None:
        with h5py.File(self.desi_random_count_maps, "r") as h5:
            ordering = str(h5.attrs.get("ordering", "")).upper()
            if ordering and ordering != "RING":
                raise ValueError(f"DESI DR9 random-count maps must be RING ordered, got {ordering!r}.")
            missing = [
                f"nside{nside}/random_count"
                for nside in DESI_DR9_SUPPORTED_RANDOM_NSIDE
                if f"nside{nside}/random_count" not in h5
            ]
        if missing:
            raise KeyError(f"DESI DR9 random-count map file is missing required dataset(s): {missing}")


@dataclass
class MeasurementConfig:
    pipeline_version: str = MEASUREMENT_PIPELINE_VERSION
    stage: str = "lowres"
    nside: int = 1024
    lmax: int = 1024
    # Harmonic cutoff used for survey masks. ``None`` follows the science
    # cutoff for backwards-compatible ad-hoc configurations; production
    # stages set this explicitly when the mask bandwidth must be larger.
    lmax_mask: Optional[int] = None
    ell_min: int = 8
    n_bins: int = 24
    binning: str = "sqrt"
    # Inclusive ACT CMB-lensing science cutoff.  ``None`` means that every
    # common-grid band is active.  The high-resolution stage uses the measured
    # ACT response support, which is non-zero through ell=3000 and null from
    # ell=3001 onward.
    kappa_cmb_lmax: Optional[int] = None
    act_cmb_temperature_units_confirmed: bool = False
    minimum_desi_random_realizations: int = 1
    act_downgrade: int = 1
    catalog_chunk: int = 2_000_000
    shear_mask_dataset: str = "mask_weight"
    shear_noise_attr: str = "shape_noise_pseudo_cl_normalized_weight_mask"
    shear_e_to_kappa_sign: float = -1.0
    subtract_masked_mean: bool = True
    mask_apodization_deg: float = 0.0
    mask_apodization_type: str = "C1"
    pair_overlap_mean_subtract: bool = False
    n_iter: int = 0
    n_iter_mask: int = 0
    covariance_l_toeplitz: int = -1
    covariance_l_exact: int = -1
    covariance_dl_band: int = -1
    covariance_workspace_cache_size: int = 0
    covariance_input_mode: str = "inka_data"
    covariance_input_smooth_bandpowers: bool = True
    covariance_input_smooth_window: int = 5
    covariance_zero_parity_odd_inputs: bool = True
    compute_covariance: bool = True
    compute_covariance_eigenvalues: bool = True
    include_ksz_velocity_shuffle: bool = True
    ksz_shuffle_seed: int = 12345
    des_y3_source_nz_fits: str = DES_Y3_SOURCE_NZ_FITS_DEFAULT
    output_dir: str = "data/xDESI/processed/multiprobe_namaster"

    @classmethod
    def for_stage(cls, stage: str) -> "MeasurementConfig":
        stage = str(stage).lower()
        if stage == "lowres":
            return cls(stage="lowres", nside=1024, lmax=2048, n_bins=32, act_downgrade=1)
        if stage == "fast1024":
            return cls(
                stage="fast1024",
                nside=1024,
                lmax=1024,
                n_bins=10,
                binning="linear",
                # CAR block averaging has an anisotropic transfer that is not
                # represented by a HEALPix pixel window. Reproject the native
                # map instead of introducing an unmodelled filter.
                act_downgrade=1,
                include_ksz_velocity_shuffle=False,
            )
        if stage == "midres2048":
            return cls(
                stage="midres2048",
                nside=2048,
                lmax=4096,
                lmax_mask=6143,
                ell_min=128,
                n_bins=16,
                binning="log",
                act_downgrade=1,
                # The DES shear and DESI random-count masks contain internal
                # zero-count pixels. Treating each zero as an apodization
                # boundary destroys their effective area. ACT T/kappa masks
                # are already tapered upstream, so no second apodization is
                # applied here either.
                mask_apodization_deg=0.0,
                mask_apodization_type="C2",
                pair_overlap_mean_subtract=False,
                include_ksz_velocity_shuffle=False,
            )
        if stage == "full":
            return cls(
                stage="full",
                nside=4096,
                lmax=4096,
                n_bins=48,
                act_downgrade=1,
                include_ksz_velocity_shuffle=False,
            )
        if stage == "highres4096":
            return cls(
                stage="highres4096",
                nside=4096,
                lmax=8192,
                lmax_mask=12287,
                ell_min=128,
                n_bins=20,
                binning="log",
                kappa_cmb_lmax=3000,
                act_cmb_temperature_units_confirmed=True,
                minimum_desi_random_realizations=8,
                act_downgrade=1,
                mask_apodization_deg=0.0,
                mask_apodization_type="C2",
                pair_overlap_mean_subtract=False,
                include_ksz_velocity_shuffle=False,
            )
        raise ValueError(
            "stage must be 'lowres', 'fast1024', 'midres2048', 'full', or 'highres4096'."
        )

    def validate(self) -> None:
        if str(self.pipeline_version) != MEASUREMENT_PIPELINE_VERSION:
            raise ValueError(
                f"pipeline_version={self.pipeline_version!r} is not supported by this code; "
                f"expected {MEASUREMENT_PIPELINE_VERSION!r}. Regenerate stale map/spectrum products."
            )
        if int(self.ell_min) < 0:
            raise ValueError("ell_min must be non-negative.")
        if int(self.lmax) < int(self.ell_min):
            raise ValueError(f"lmax={self.lmax} must be >= ell_min={self.ell_min}.")
        if int(self.n_bins) <= 0:
            raise ValueError("n_bins must be positive.")
        if int(self.minimum_desi_random_realizations) <= 0:
            raise ValueError("minimum_desi_random_realizations must be positive.")
        if int(self.lmax) > 3 * int(self.nside) - 1:
            raise ValueError(f"lmax={self.lmax} exceeds the HEALPix limit 3*nside-1={3 * int(self.nside) - 1}.")
        if int(self.effective_lmax_mask) < int(self.lmax):
            raise ValueError(
                f"lmax_mask={self.effective_lmax_mask} must be >= the science lmax={self.lmax}."
            )
        if int(self.effective_lmax_mask) > 3 * int(self.nside) - 1:
            raise ValueError(
                f"lmax_mask={self.effective_lmax_mask} exceeds the HEALPix limit "
                f"3*nside-1={3 * int(self.nside) - 1}."
            )
        if str(self.binning).lower() not in {"sqrt", "linear", "log"}:
            raise ValueError(f"Unsupported binning={self.binning!r}; expected 'sqrt', 'linear', or 'log'.")
        if float(self.mask_apodization_deg) < 0.0:
            raise ValueError("mask_apodization_deg must be non-negative.")
        if str(self.mask_apodization_type) not in {"C1", "C2", "Smooth"}:
            raise ValueError("mask_apodization_type must be one of 'C1', 'C2', or 'Smooth'.")
        if self.kappa_cmb_lmax is not None:
            if int(self.kappa_cmb_lmax) < int(self.ell_min):
                raise ValueError("kappa_cmb_lmax must be >= ell_min when supplied.")
            if int(self.kappa_cmb_lmax) > int(self.lmax):
                raise ValueError("kappa_cmb_lmax must be <= lmax when supplied.")
        if bool(self.act_cmb_temperature_units_confirmed) and ACT_CMB_TEMPERATURE_UNITS != "uK_CMB":
            raise ValueError("The confirmed ACT CMB temperature-unit contract must be uK_CMB.")

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir).resolve() / self.stage

    @property
    def effective_lmax_mask(self) -> int:
        """Resolved NaMaster mask-harmonic cutoff saved in product provenance."""

        return int(self.lmax if self.lmax_mask is None else self.lmax_mask)

    def to_dict(self) -> Dict[str, object]:
        """Serialize configuration with defaults resolved to executed values."""

        payload = asdict(self)
        payload["lmax_mask"] = int(self.effective_lmax_mask)
        return payload

    @property
    def map_product_tag(self) -> str:
        """Tag for reusable map products, which do not depend on the saved-mean policy."""

        parts = [f"nside{self.nside}"]
        if int(self.ell_min) != 8:
            parts.append(f"ell{int(self.ell_min)}")
        parts.append(f"lmax{self.lmax}")
        if int(self.effective_lmax_mask) != int(self.lmax):
            parts.append(f"lmask{self.effective_lmax_mask}")
        if not (self.stage in {"lowres", "full"} and str(self.binning).lower() == "sqrt"):
            parts.append(f"nbin{self.n_bins}")
            parts.append(str(self.binning).lower())
        apo = float(self.mask_apodization_deg)
        if apo > 0.0:
            apo_tag = ("%g" % apo).replace(".", "p")
            parts.append(f"apo{apo_tag}deg")
            parts.append(str(self.mask_apodization_type))
        if bool(self.pair_overlap_mean_subtract):
            parts.append("pairmean")
        parts.append("pipev2")
        return "_".join(parts)

    @property
    def product_tag(self) -> str:
        """Tag for spectra/assembled products with galaxy shot noise in the mean."""

        parts = [self.map_product_tag, DESI_GALAXY_AUTO_PRODUCT_TAG]
        if self.kappa_cmb_lmax is not None:
            parts.extend((f"gkell{int(self.kappa_cmb_lmax)}", "dvvalidv1"))
        return "_".join(parts)

    @property
    def covariance_product_tag(self) -> str:
        """Tag for covariance artifacts, unchanged by the saved galaxy-mean policy."""

        return self.map_product_tag

    @property
    def default_maps_path(self) -> Path:
        return self.output_root / f"xdesi_multiprobe_maps_{self.map_product_tag}.h5"

    @property
    def default_measurement_path(self) -> Path:
        return self.output_root / f"xdesi_multiprobe_cls_cov_{self.product_tag}.h5"


@dataclass
class FieldMap:
    name: str
    label: str
    kind: str
    spin: int
    maps: List[np.ndarray]
    mask: np.ndarray
    mask_name: str
    metadata: Dict[str, object] = field(default_factory=dict)
    catalog: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        return len(self.maps)

    @property
    def has_catalog_momentum(self) -> bool:
        required = {"ra_deg", "dec_deg", "weight", "field"}
        return self.kind in {"desi_momentum", "desi_momentum_null"} and required.issubset(self.catalog)


@dataclass(frozen=True)
class SpectrumSpec:
    name: str
    family: str
    fields: Tuple[str, str]
    component: int
    label: str
    theory_key: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class NmtProbeField:
    info: FieldMap
    field: nmt.NmtField
    covariance_field: Optional[nmt.NmtField] = None

    @property
    def spin(self) -> int:
        return int(self.info.spin)

    @property
    def mask(self) -> np.ndarray:
        return self.info.mask

    @property
    def cov_field(self) -> nmt.NmtField:
        return self.field if self.covariance_field is None else self.covariance_field

    @property
    def is_catalog_momentum(self) -> bool:
        return self.info.has_catalog_momentum and bool(getattr(self.field, "is_catalog", False))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(value: object) -> object:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r} to JSON")


def _json_dumps(value: object) -> str:
    return json.dumps(value, default=_json_default, indent=2, sort_keys=True)


def _stable_sha256(value: object) -> str:
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _array_content_sha256(values: np.ndarray) -> str:
    """Return a stable identity for one saved numeric array."""

    arr = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.dtype.str.encode("ascii"))
    digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


def _array_raw_sha256(values: np.ndarray) -> str:
    """Hash only contiguous array bytes, matching transferred count-map provenance."""

    arr = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(memoryview(arr).cast("B")).hexdigest()


def map_content_digests(fields: Mapping[str, FieldMap]) -> Dict[str, object]:
    """Content-address the complete estimator contract for every saved field."""

    masks: Dict[str, str] = {}
    field_mask_names: Dict[str, str] = {}
    field_descriptors: Dict[str, str] = {}
    field_arrays: Dict[str, object] = {}
    for name in sorted(fields):
        field_map = fields[name]
        mask_name = str(field_map.mask_name)
        if mask_name not in masks:
            masks[mask_name] = _array_content_sha256(
                np.asarray(field_map.mask, dtype=np.float32)
            )
        field_mask_names[name] = mask_name
        field_descriptors[name] = _stable_sha256(
            {
                "name": str(field_map.name),
                "label": str(field_map.label),
                "kind": str(field_map.kind),
                "spin": int(field_map.spin),
                "mask_name": mask_name,
                "metadata": field_map.metadata,
            }
        )
        field_arrays[name] = {
            "maps": [
                _array_content_sha256(np.asarray(values, dtype=np.float32))
                for values in field_map.maps
            ],
            "catalog": {
                key: _array_content_sha256(
                    np.asarray(field_map.catalog[key], dtype=np.float64)
                )
                for key in sorted(field_map.catalog)
            },
        }
    return {
        "masks": masks,
        "field_mask_names": field_mask_names,
        "field_descriptors": field_descriptors,
        "fields": field_arrays,
    }


def map_product_id_from_metadata(metadata: Mapping[str, object]) -> str:
    """Derive the map-product ID from algorithms, inputs, config and array content."""

    content = metadata.get("map_content_digests")
    if not isinstance(content, Mapping) or not content:
        raise ValueError("Map metadata has no map_content_digests; regenerate the map product.")
    return _stable_sha256(
        {
            "pipeline_version": metadata.get("pipeline_version"),
            "map_construction_version": metadata.get("map_construction_version"),
            "config": metadata.get("config"),
            "input_files": metadata.get("input_files"),
            "desi_summary": metadata.get("desi_summary"),
            "des_y3_source_nz": metadata.get("des_y3_source_nz"),
            "map_content_digests": content,
        }
    )


def map_metadata_with_content_identity(
    fields: Mapping[str, FieldMap],
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    """Return map metadata bound to the exact saved mask/map/catalog arrays."""

    out = dict(metadata)
    out["map_content_digests"] = map_content_digests(fields)
    out["map_product_id"] = map_product_id_from_metadata(out)
    return out


def validate_map_metadata_identity(metadata: Mapping[str, object]) -> str:
    """Validate and return the content-addressed map-product ID."""

    product_id = str(metadata.get("map_product_id", ""))
    expected = map_product_id_from_metadata(metadata)
    if not product_id or product_id != expected:
        raise ValueError(
            "Map product identity is missing or does not match its config/input/content digests; "
            "regenerate it instead of reusing it."
        )
    return product_id


def validate_loaded_map_content(
    fields: Mapping[str, FieldMap],
    metadata: Mapping[str, object],
) -> None:
    """Check loaded arrays against the content identities stored in map metadata."""

    expected = metadata.get("map_content_digests")
    if not isinstance(expected, Mapping):
        raise ValueError("Map metadata has no content digests; regenerate the product.")
    actual = map_content_digests(fields)
    for section in ("masks", "field_mask_names", "field_descriptors", "fields"):
        expected_section = expected.get(section)
        if not isinstance(expected_section, Mapping):
            raise ValueError(f"Map metadata content-digest section {section!r} is missing.")
        for name, value in actual[section].items():
            if name not in expected_section or expected_section[name] != value:
                raise ValueError(
                    f"Loaded map content for {section}/{name} does not match map_product_id; "
                    "the HDF5 product is stale or has been modified."
                )


def _json_hdf5_attr(attrs: h5py.AttributeManager, key: str) -> object:
    if key not in attrs:
        raise ValueError(f"HDF5 product is missing required {key!r} provenance.")
    raw = attrs[key]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(str(raw))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"HDF5 product has invalid JSON provenance in {key!r}.") from exc


def validate_map_product_hdf_identity(
    h5: h5py.File,
    *,
    allow_legacy_product: bool = False,
) -> str:
    """Validate the algorithm and originating-content identity of an open map HDF5."""

    schema = str(h5.attrs.get("schema", ""))
    if schema != SCHEMA_MAPS:
        raise ValueError(f"HDF5 product schema={schema!r}, expected {SCHEMA_MAPS!r}.")
    pipeline_version = str(h5.attrs.get("pipeline_version", ""))
    if pipeline_version != MEASUREMENT_PIPELINE_VERSION:
        if allow_legacy_product:
            return str(h5.attrs.get("map_product_id", ""))
        raise ValueError(
            f"Map product pipeline_version={pipeline_version!r} is legacy; expected "
            f"{MEASUREMENT_PIPELINE_VERSION!r}. Regenerate it, or explicitly opt into "
            "allow_legacy_product=True for a historical-only comparison."
        )
    map_version = str(h5.attrs.get("map_construction_version", ""))
    if map_version != MAP_CONSTRUCTION_VERSION:
        raise ValueError(
            f"Map product map_construction_version={map_version!r}, expected "
            f"{MAP_CONSTRUCTION_VERSION!r}."
        )
    metadata = _json_hdf5_attr(h5.attrs, "metadata_json")
    if not isinstance(metadata, Mapping):
        raise ValueError("Map metadata_json must decode to a mapping.")
    product_id = validate_map_metadata_identity(metadata)
    root_product_id = str(h5.attrs.get("map_product_id", ""))
    if root_product_id != product_id:
        raise ValueError("Map HDF5 map_product_id does not match its embedded metadata.")
    for key, expected in (
        ("pipeline_version", MEASUREMENT_PIPELINE_VERSION),
        ("map_construction_version", MAP_CONSTRUCTION_VERSION),
        ("spectrum_estimator_version", SPECTRUM_ESTIMATOR_VERSION),
        ("covariance_estimator_version", COVARIANCE_ESTIMATOR_VERSION),
    ):
        if str(metadata.get(key, "")) != expected:
            raise ValueError(
                f"Map metadata {key}={metadata.get(key)!r}, expected {expected!r}."
            )
    return product_id


def validate_measurement_product_identity(
    h5: h5py.File,
    *,
    allow_legacy_product: bool = False,
) -> str:
    """Validate an open measurement and return its originating map-product ID.

    Legacy products are rejected by default because their masks, noise subtraction,
    transfer functions and covariance estimator differ from pipeline v2.  The opt-in
    exists only to make deliberately historical comparisons explicit.
    """

    schema = str(h5.attrs.get("schema", ""))
    supported_schemas = {SCHEMA_MEASUREMENT, SCHEMA_MEASUREMENT_VALIDITY_MASK}
    if schema not in supported_schemas:
        raise ValueError(
            f"HDF5 product schema={schema!r}, expected one of {sorted(supported_schemas)!r}."
        )
    pipeline_version = str(h5.attrs.get("pipeline_version", ""))
    if pipeline_version != MEASUREMENT_PIPELINE_VERSION:
        if allow_legacy_product:
            return str(h5.attrs.get("map_product_id", ""))
        raise ValueError(
            f"Measurement pipeline_version={pipeline_version!r} is legacy; expected "
            f"{MEASUREMENT_PIPELINE_VERSION!r}. Regenerate it, or explicitly opt into "
            "allow_legacy_product=True for a historical-only comparison."
        )
    for key, expected in (
        ("map_construction_version", MAP_CONSTRUCTION_VERSION),
        ("spectrum_estimator_version", SPECTRUM_ESTIMATOR_VERSION),
        ("covariance_estimator_version", COVARIANCE_ESTIMATOR_VERSION),
    ):
        actual = str(h5.attrs.get(key, ""))
        if actual != expected:
            raise ValueError(f"Measurement {key}={actual!r}, expected {expected!r}.")
    mean_convention = str(h5.attrs.get("desi_galaxy_auto_mean_convention", ""))
    if mean_convention != DESI_GALAXY_AUTO_MEAN_CONVENTION and not allow_legacy_product:
        raise ValueError(
            "Measurement desi_galaxy_auto_mean_convention="
            f"{mean_convention!r}, expected {DESI_GALAXY_AUTO_MEAN_CONVENTION!r}. "
            "Regenerate it, or explicitly opt into allow_legacy_product=True for a "
            "historical shot-noise-subtracted comparison."
        )
    map_metadata = _json_hdf5_attr(h5.attrs, "map_metadata_json")
    if not isinstance(map_metadata, Mapping):
        raise ValueError("Measurement map_metadata_json must decode to a mapping.")
    map_product_id = validate_map_metadata_identity(map_metadata)
    if str(h5.attrs.get("map_product_id", "")) != map_product_id:
        raise ValueError(
            "Measurement map_product_id does not match its embedded map metadata."
        )
    for key, expected in (
        ("pipeline_version", MEASUREMENT_PIPELINE_VERSION),
        ("map_construction_version", MAP_CONSTRUCTION_VERSION),
        ("spectrum_estimator_version", SPECTRUM_ESTIMATOR_VERSION),
        ("covariance_estimator_version", COVARIANCE_ESTIMATOR_VERSION),
    ):
        if str(map_metadata.get(key, "")) != expected:
            raise ValueError(
                f"Embedded map metadata {key}={map_metadata.get(key)!r}, expected {expected!r}."
            )
    measurement_config = _json_hdf5_attr(h5.attrs, "config_json")
    map_config = map_metadata.get("config")
    if not isinstance(measurement_config, Mapping) or not isinstance(map_config, Mapping):
        raise ValueError("Measurement and embedded map configs must both be mappings.")
    expected_schema = measurement_schema_for_config(measurement_config)
    if schema != expected_schema:
        raise ValueError(
            f"Measurement schema={schema!r} does not match its packing config; expected {expected_schema!r}."
        )
    # Split production intentionally changes execution-only settings such as
    # compute_covariance after loading the prepared map.  Bind only settings that
    # determine the saved map realization; the measurement HDF5 is authoritative for
    # its own binning/covariance settings.
    map_construction_keys = (
        "pipeline_version",
        "nside",
        "act_downgrade",
        "shear_mask_dataset",
        "shear_noise_attr",
        "shear_e_to_kappa_sign",
        "subtract_masked_mean",
        "mask_apodization_deg",
        "mask_apodization_type",
        "include_ksz_velocity_shuffle",
        "ksz_shuffle_seed",
    )
    if schema == SCHEMA_MEASUREMENT_VALIDITY_MASK:
        map_construction_keys += (
            "act_cmb_temperature_units_confirmed",
            "minimum_desi_random_realizations",
        )
    for key in map_construction_keys:
        if key not in measurement_config or key not in map_config:
            raise ValueError(f"Measurement/map provenance is missing config key {key!r}.")
        if measurement_config[key] != map_config[key]:
            raise ValueError(
                f"Measurement config {key}={measurement_config[key]!r} does not match "
                f"its originating map config value {map_config[key]!r}."
            )
    if int(measurement_config.get("lmax", -1)) > int(map_config.get("lmax", -1)):
        raise ValueError(
            "Measurement lmax exceeds the lmax used to construct its originating map product."
        )
    measurement_lmax_mask = measurement_config.get("lmax_mask")
    if measurement_lmax_mask is None:
        measurement_lmax_mask = measurement_config.get("lmax")
    map_lmax_mask = map_config.get("lmax_mask")
    if map_lmax_mask is None:
        map_lmax_mask = map_config.get("lmax")
    if int(measurement_lmax_mask) != int(map_lmax_mask):
        raise ValueError(
            "Measurement lmax_mask does not match the mask bandwidth of its originating map product."
        )
    if schema == SCHEMA_MEASUREMENT_VALIDITY_MASK:
        covariance_present = "joint/cov" in h5
        covariance_declared = bool(measurement_config.get("compute_covariance", False))
        if covariance_present != covariance_declared:
            raise ValueError(
                "Validity-mask product covariance presence disagrees with "
                "config.compute_covariance; regenerate the spectra intermediate or final assembly."
            )
        required = (
            "joint/data_vector",
            "joint/data_vector_raw",
            "joint/data_vector_valid",
            "joint/spectrum_names",
            "joint/slice_start",
            "joint/slice_stop",
            "ell_left",
            "ell_right",
        )
        missing = [key for key in required if key not in h5]
        if missing:
            raise ValueError(f"Validity-mask measurement is missing required dataset(s): {missing}.")
        packed = np.asarray(h5["joint/data_vector"][:], dtype=np.float64)
        raw = np.asarray(h5["joint/data_vector_raw"][:], dtype=np.float64)
        valid = np.asarray(h5["joint/data_vector_valid"][:], dtype=bool)
        if packed.shape != raw.shape or packed.shape != valid.shape:
            raise ValueError("Packed, raw, and validity vectors must have identical shapes.")
        if not np.array_equal(packed[valid], raw[valid]) or not np.all(packed[~valid] == 0.0):
            raise ValueError("Packed data-vector values do not obey the saved validity mask.")
        names = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in h5["joint/spectrum_names"][:]
        ]
        starts = np.asarray(h5["joint/slice_start"][:], dtype=int)
        stops = np.asarray(h5["joint/slice_stop"][:], dtype=int)
        left = np.asarray(h5["ell_left"][:], dtype=np.int64)
        right = np.asarray(h5["ell_right"][:], dtype=np.int64)
        canonical_names = [spec.name for spec in default_spectrum_specs()]
        if names != canonical_names:
            raise ValueError(
                "Validity-mask spectrum names/order do not match the canonical 46-spectrum contract."
            )
        n_band = int(left.size)
        canonical_starts = np.arange(len(canonical_names), dtype=int) * n_band
        canonical_stops = canonical_starts + n_band
        if not np.array_equal(starts, canonical_starts) or not np.array_equal(
            stops, canonical_stops
        ):
            raise ValueError(
                "Validity-mask spectrum slices are not the canonical contiguous band-major layout."
            )
        expected_vector_size = len(canonical_names) * n_band
        if packed.shape != (expected_vector_size,):
            raise ValueError(
                "Validity-mask archive vectors do not match the canonical spectrum-by-band size."
            )
        cutoff = int(measurement_config["kappa_cmb_lmax"])
        boundary = cutoff + 1
        if boundary not in set(right.tolist()) or np.any((left < boundary) & (right > boundary)):
            raise ValueError("Saved band edges do not exactly resolve the ACT kappa response cutoff.")
        expected_valid = np.ones_like(valid)
        raw_chunks = []
        for name, start, stop in zip(names, starts, stops):
            if stop - start != left.size:
                raise ValueError(f"Spectrum {name!r} slice is inconsistent with the common ell grid.")
            group = h5[f"spectra/{name}"]
            cl = np.asarray(group["cl"][:], dtype=np.float64)
            if cl.shape != (n_band,):
                raise ValueError(
                    f"Spectrum {name!r} cl shape {cl.shape} does not match the common ell grid."
                )
            raw_chunks.append(cl)
            if str(group.attrs.get("family", "")) == "desi_g_act_kappa":
                expected_valid[start:stop] = right <= boundary
            if "data_vector_valid" not in group or not np.array_equal(
                np.asarray(group["data_vector_valid"][:], dtype=bool), expected_valid[start:stop]
            ):
                raise ValueError(f"Spectrum {name!r} has missing or inconsistent validity metadata.")
        if not np.array_equal(valid, expected_valid):
            raise ValueError("joint/data_vector_valid does not match spectrum families and band support.")
        if not np.array_equal(raw, np.concatenate(raw_chunks)):
            raise ValueError("joint/data_vector_raw is not the exact concatenation of spectra/<name>/cl.")
    if (
        "desi_galaxy_auto_views_contract_version" in h5.attrs
        or "joint/views" in h5
    ):
        validate_galaxy_auto_views(h5, require=True)
    return map_product_id


def validate_galaxy_auto_views(
    h5: h5py.File,
    *,
    require: bool = False,
) -> Dict[str, object]:
    """Validate the two DESI galaxy-auto mean views in an open product.

    The total view is the primary HMC estimator.  The weighted-Poisson-
    subtracted view is an exact deterministic shift of the four galaxy-auto
    slices.  Both covariance links must resolve to the same HDF5 object as
    ``joint/cov`` so the file cannot silently advertise a signal-only
    covariance.
    """

    contract = str(h5.attrs.get("desi_galaxy_auto_views_contract_version", ""))
    present = "joint/views" in h5
    if not present:
        if require:
            raise ValueError("Measurement is missing the required joint/views galaxy-auto contract.")
        return {"present": False}
    if contract != DESI_GALAXY_AUTO_VIEWS_CONTRACT_VERSION:
        raise ValueError(
            f"Galaxy-auto views contract={contract!r}, expected "
            f"{DESI_GALAXY_AUTO_VIEWS_CONTRACT_VERSION!r}."
        )

    joint = h5["joint"]
    views = joint["views"]
    if str(views.attrs.get("contract_version", "")) != contract:
        raise ValueError("joint/views contract version disagrees with the root product contract.")
    if str(views.attrs.get("primary_view", "")) != DESI_GALAXY_AUTO_PRIMARY_VIEW:
        raise ValueError("joint/views does not identify the total estimator as the primary HMC view.")
    required_views = {DESI_GALAXY_AUTO_PRIMARY_VIEW, DESI_GALAXY_AUTO_SUBTRACTED_VIEW}
    if not required_views.issubset(set(views.keys())):
        raise ValueError("joint/views is missing a required galaxy-auto mean view.")

    total = views[DESI_GALAXY_AUTO_PRIMARY_VIEW]
    subtracted = views[DESI_GALAXY_AUTO_SUBTRACTED_VIEW]
    for name in ("data_vector", "data_vector_raw", "data_vector_valid"):
        if name not in total or name not in subtracted:
            raise ValueError(f"Galaxy-auto view is missing {name!r}.")
    if total["data_vector"].id != joint["data_vector"].id:
        raise ValueError("Total-view data_vector is not a hard link to joint/data_vector.")
    if total["data_vector_raw"].id != joint["data_vector_raw"].id:
        raise ValueError("Total-view raw vector is not a hard link to joint/data_vector_raw.")
    if total["data_vector_valid"].id != joint["data_vector_valid"].id:
        raise ValueError("Total-view validity is not a hard link to the joint validity mask.")
    if subtracted["data_vector_valid"].id != joint["data_vector_valid"].id:
        raise ValueError("Subtracted-view validity is not a hard link to the joint validity mask.")

    raw = np.asarray(joint["data_vector_raw"][:], dtype=np.float64)
    valid = np.asarray(joint["data_vector_valid"][:], dtype=bool)
    template_path = "galaxy_auto_weighted_poisson_template"
    if template_path not in views:
        raise ValueError("joint/views has no saved galaxy-auto weighted-Poisson template.")
    template = np.asarray(views[template_path][:], dtype=np.float64)
    if template.shape != raw.shape or not np.all(np.isfinite(template)):
        raise ValueError("Joint galaxy-auto weighted-Poisson template has invalid shape or values.")

    names = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in joint["spectrum_names"][:]
    ]
    starts = np.asarray(joint["slice_start"][:], dtype=int)
    stops = np.asarray(joint["slice_stop"][:], dtype=int)
    expected_template = np.zeros_like(raw)
    galaxy_names: List[str] = []
    for name, start, stop in zip(names, starts, stops):
        group = h5[f"spectra/{name}"]
        if str(group.attrs.get("family", "")) != "desi_g_auto":
            continue
        galaxy_names.append(name)
        component = int(group.attrs["component"])
        if "noise_decoupled_all_components" not in group:
            raise ValueError(f"Galaxy auto {name!r} is missing its shot-noise template.")
        noise_all = np.asarray(group["noise_decoupled_all_components"][:], dtype=np.float64)
        cl_all = np.asarray(group["cl_all_components"][:], dtype=np.float64)
        if noise_all.shape != cl_all.shape or not (0 <= component < noise_all.shape[0]):
            raise ValueError(f"Galaxy auto {name!r} has incompatible component/noise shapes.")
        expected_template[start:stop] = noise_all[component]
        expected_cl = np.asarray(group["cl"][:], dtype=np.float64) - noise_all[component]
        if "cl_weighted_poisson_subtracted" not in group or not np.array_equal(
            np.asarray(group["cl_weighted_poisson_subtracted"][:], dtype=np.float64),
            expected_cl,
        ):
            raise ValueError(f"Galaxy auto {name!r} has an inconsistent subtracted mean view.")
        if "cl_all_components_weighted_poisson_subtracted" not in group or not np.array_equal(
            np.asarray(
                group["cl_all_components_weighted_poisson_subtracted"][:],
                dtype=np.float64,
            ),
            cl_all - noise_all,
        ):
            raise ValueError(f"Galaxy auto {name!r} has an inconsistent all-component subtracted view.")
    if len(galaxy_names) != 4:
        raise ValueError(f"Expected four DESI galaxy-auto spectra, found {len(galaxy_names)}.")
    if not np.array_equal(template, expected_template):
        raise ValueError("Joint weighted-Poisson template does not match the four saved galaxy autos.")

    expected_subtracted_raw = raw - template
    expected_subtracted = expected_subtracted_raw.copy()
    expected_subtracted[~valid] = 0.0
    if not np.array_equal(
        np.asarray(subtracted["data_vector_raw"][:], dtype=np.float64),
        expected_subtracted_raw,
    ):
        raise ValueError("Subtracted raw data vector is not total minus the saved template.")
    if not np.array_equal(
        np.asarray(subtracted["data_vector"][:], dtype=np.float64),
        expected_subtracted,
    ):
        raise ValueError("Subtracted packed data vector does not obey the common validity mask.")

    covariance_shared = False
    if "cov" in joint:
        if "cov" not in total or "cov" not in subtracted:
            raise ValueError("Both galaxy-auto views must expose the full estimator covariance.")
        if total["cov"].id != joint["cov"].id or subtracted["cov"].id != joint["cov"].id:
            raise ValueError("Galaxy-auto view covariance is not the shared joint/cov HDF5 object.")
        covariance_shared = True
    elif "cov" in total or "cov" in subtracted:
        raise ValueError("A spectra-only view must not advertise a covariance that is absent from joint.")

    changed = template != 0.0
    galaxy_support = np.zeros_like(changed)
    for name, start, stop in zip(names, starts, stops):
        if name in galaxy_names:
            galaxy_support[start:stop] = True
    if np.any(changed & ~galaxy_support):
        raise ValueError("The weighted-Poisson view changes a non-galaxy-auto data-vector element.")
    return {
        "present": True,
        "contract_version": contract,
        "primary_view": DESI_GALAXY_AUTO_PRIMARY_VIEW,
        "subtracted_view": DESI_GALAXY_AUTO_SUBTRACTED_VIEW,
        "galaxy_auto_spectra": galaxy_names,
        "galaxy_auto_elements": int(np.count_nonzero(galaxy_support)),
        "changed_elements": int(np.count_nonzero(changed)),
        "covariance_shared_hard_link": covariance_shared,
    }


def gaussian_beam_transfer(lmax: int, fwhm_arcmin: float) -> np.ndarray:
    """Return the harmonic-space Gaussian beam B_ell for a FWHM in arcmin."""

    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("lmax must be non-negative.")
    ell = np.arange(lmax + 1, dtype=np.float64)
    sigma_rad = math.radians(float(fwhm_arcmin) / 60.0) / math.sqrt(8.0 * math.log(2.0))
    return np.exp(-0.5 * ell * (ell + 1.0) * sigma_rad**2)


def _write_act_beam_transfers(group: h5py.Group, lmax: int) -> None:
    y_beam = gaussian_beam_transfer(lmax, ACT_TSZ_BEAM_FWHM_ARCMIN)
    t_beam = gaussian_beam_transfer(lmax, ACT_CMB_TEMPERATURE_BEAM_FWHM_ARCMIN)

    y_ds = _write_dataset(group, "act_y_gaussian_beam", y_beam, dtype="f8")
    y_ds.attrs["applies_to_field"] = "y"
    y_ds.attrs["fwhm_arcmin"] = ACT_TSZ_BEAM_FWHM_ARCMIN
    y_ds.attrs["note"] = "Default ACT Compton-y beam to apply to sky theory before NaMaster bandpower windows."

    t_ds = _write_dataset(group, "act_cmb_temperature_gaussian_beam", t_beam, dtype="f8")
    t_ds.attrs["applies_to_field"] = "T"
    t_ds.attrs["fwhm_arcmin"] = ACT_CMB_TEMPERATURE_BEAM_FWHM_ARCMIN
    t_ds.attrs["reference"] = KSZ_REFERENCE_PAPER
    t_ds.attrs["note"] = "ACT DR6 hILC temperature effective FWHM used by the kSZ reference analysis."

    group.attrs["act_y_effective_transfer_missing"] = False
    group.attrs["act_cmb_temperature_effective_transfer_missing"] = False
    group.attrs["act_y_effective_transfer_note"] = (
        "The default theory wrapper multiplies y-theory spectra by a 1.6 arcmin Gaussian beam. "
        "Pass an extra transfer_functions['y'] if using a map-specific filter beyond this beam."
    )
    group.attrs["act_cmb_temperature_effective_transfer_note"] = (
        "The default theory wrapper multiplies temperature-theory spectra by a 1.6 arcmin Gaussian beam. "
        "Pass an extra transfer_functions['T'] if using additional temperature filtering."
    )


def _pz_label_to_int(label: object) -> int:
    text = str(label).strip().lower()
    if text.startswith("pz"):
        text = text[2:]
    return int(text)


def default_ksz_sigma_true_gas_calibration() -> Dict[str, object]:
    return {
        "source": "module_constants",
        "doc": KSZ_SIGMA_TRUE_GAS_DOC,
        "json_relpath": KSZ_SIGMA_TRUE_GAS_JSON_REL,
        "bin_assignment": "by_z_photo",
        "velocity_column": "v_true",
        "velocity_units": "km/s",
        "c_km_s_convention": 300000.0,
        "sigma_true_gas_over_c": dict(KSZ_SIGMA_TRUE_GAS_OVER_C_3E5),
        "sigma_true_gas_km_s": dict(KSZ_SIGMA_TRUE_GAS_KM_S),
    }


def load_ksz_sigma_true_gas_calibration(
    path: Optional[str | Path] = None,
    bin_assignment: str = "by_z_photo",
) -> Dict[str, object]:
    """Load Abacus true-velocity RMS values for the DESI photometric bins."""

    if path is None:
        return default_ksz_sigma_true_gas_calibration()
    path = Path(path)
    if not path.exists():
        return default_ksz_sigma_true_gas_calibration()

    raw = json.loads(path.read_text())
    rows = raw.get(bin_assignment)
    if not isinstance(rows, list):
        raise KeyError(f"{path} does not contain a {bin_assignment!r} calibration list.")
    sigma_over_c: Dict[int, float] = {}
    sigma_km_s: Dict[int, float] = {}
    n_objects: Dict[int, int] = {}
    corrcoef: Dict[int, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        pz_bin = _pz_label_to_int(row["bin"])
        sigma_over_c[pz_bin] = float(row["sigma_true_gas_over_c_3e5"])
        sigma_km_s[pz_bin] = float(row["sigma_true_gas_km_s"])
        n_objects[pz_bin] = int(row["n_objects"])
        if "corrcoef_v_rec_v_true" in row:
            corrcoef[pz_bin] = float(row["corrcoef_v_rec_v_true"])
    if set(sigma_over_c) != {1, 2, 3, 4}:
        raise ValueError(f"{path} did not provide sigma_true_gas for all four pz bins.")
    return {
        "source": "json",
        "path": str(path),
        "doc": KSZ_SIGMA_TRUE_GAS_DOC,
        "bin_assignment": bin_assignment,
        "description": raw.get("description", ""),
        "paper_definition": raw.get("paper_definition", ""),
        "source_npz_nersc": raw.get("source_npz_nersc", ""),
        "velocity_column": raw.get("velocity_column", "v_true"),
        "velocity_units": raw.get("velocity_units", "km/s"),
        "std_convention": raw.get("std_convention", ""),
        "c_km_s_convention": float(raw.get("c_km_s_convention", 300000.0)),
        "sigma_true_gas_over_c": sigma_over_c,
        "sigma_true_gas_km_s": sigma_km_s,
        "n_objects": n_objects,
        "corrcoef_v_rec_v_true": corrcoef,
    }


def des_y3_gaussian_priors() -> Dict[str, Dict[str, float]]:
    return {
        name: {"mu": float(mu), "sigma": float(sigma)}
        for name, (mu, sigma) in DES_Y3_GAUSSIAN_PRIORS.items()
    }


def load_des_y3_source_nz(path: str | Path, hdu_name: str = DES_Y3_SOURCE_NZ_HDU) -> Dict[str, object]:
    """Load DES Y3 source-bin n(z) curves and normalize them for theory use."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing DES Y3 source n(z) FITS: {path}")
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[hdu_name]
        data = hdu.data
        z_low = np.asarray(data["Z_LOW"], dtype=np.float64)
        z_mid = np.asarray(data["Z_MID"], dtype=np.float64)
        z_high = np.asarray(data["Z_HIGH"], dtype=np.float64)
        raw = np.vstack([np.asarray(data[f"BIN{i}"], dtype=np.float64) for i in range(1, 5)])
        trapz_norm = _trapz(raw, x=z_mid, axis=1)
        width_norm = np.sum(raw * (z_high - z_low)[None, :], axis=1)
        if np.any(~np.isfinite(trapz_norm)) or np.any(trapz_norm <= 0):
            raise ValueError(f"Invalid DES Y3 source n(z) normalization in {path}.")
        dndz = raw / trapz_norm[:, None]
        mean_z = np.asarray([_trapz(z_mid * dndz_i, x=z_mid) for dndz_i in dndz], dtype=np.float64)
        sigma_e = np.asarray([float(hdu.header.get(f"SIG_E_{i}", np.nan)) for i in range(1, 5)], dtype=np.float64)
        ngal_arcmin2 = np.asarray([float(hdu.header.get(f"NGAL_{i}", np.nan)) for i in range(1, 5)], dtype=np.float64)
    return {
        "source_fits": str(path),
        "hdu": hdu_name,
        "z_low": z_low,
        "z_mid": z_mid,
        "z_high": z_high,
        "raw_bin_values_by_bin": raw,
        "dndz_by_bin": dndz,
        "raw_trapz_norm_by_bin": trapz_norm,
        "raw_width_norm_by_bin": width_norm,
        "mean_z_by_bin": mean_z,
        "sigma_e_by_bin": sigma_e,
        "ngal_arcmin2_by_bin": ngal_arcmin2,
        "raw_column_note": (
            "FITS BIN columns are stored separately from normalized theory dN/dz. "
            "Use dndz_by_bin for theory; it is normalized by trapezoidal integration over Z_MID."
        ),
        "priors": des_y3_gaussian_priors(),
    }


def _redshift_edges_from_minmax(zmin: np.ndarray, zmax: np.ndarray) -> np.ndarray:
    zmin = np.asarray(zmin, dtype=np.float64)
    zmax = np.asarray(zmax, dtype=np.float64)
    if zmin.shape != zmax.shape or zmin.ndim != 1:
        raise ValueError(f"Expected 1D zmin/zmax arrays with matching shape, got {zmin.shape} and {zmax.shape}.")
    if zmin.size == 0:
        raise ValueError("True DESI n(z) redshift grid is empty.")
    return np.concatenate([[zmin[0]], zmax])


def load_desi_dr9_calibrated_true_nz(
    path: str | Path,
    group: str = DESI_DR9_TRUE_NZ_GROUP_FULL_CL,
) -> Dict[str, object]:
    """Load calibrated true-redshift n(z) for the full DR9 Extended valid_for_cl sample."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing DESI DR9 calibrated true n(z) HDF5: {path}")
    dndz_rows = []
    n_per_deg2_rows = []
    dndz_per_deg2_rows = []
    mean_z = []
    sigma_z = []
    surface_density = []
    with h5py.File(path, "r") as h5:
        z_mid = np.asarray(h5["redshift_bins/zmid"][:], dtype=np.float64)
        z_min = np.asarray(h5["redshift_bins/zmin"][:], dtype=np.float64)
        z_max = np.asarray(h5["redshift_bins/zmax"][:], dtype=np.float64)
        dz = np.asarray(h5["redshift_bins/dz"][:], dtype=np.float64)
        z_edges = _redshift_edges_from_minmax(z_min, z_max)
        for pz_bin in range(1, 5):
            key = f"{group}/pz{pz_bin}"
            if key not in h5:
                raise KeyError(f"Missing calibrated true n(z) group {key!r} in {path}.")
            g = h5[key]
            nz = np.asarray(g[DESI_DR9_TRUE_NZ_DATASET][:], dtype=np.float64)
            if nz.shape != z_mid.shape:
                raise ValueError(f"{key}/{DESI_DR9_TRUE_NZ_DATASET} has shape {nz.shape}, expected {z_mid.shape}.")
            norm = float(np.sum(nz * dz))
            if not np.isfinite(norm) or norm <= 0.0:
                raise ValueError(f"{key}/{DESI_DR9_TRUE_NZ_DATASET} has invalid normalization {norm}.")
            dndz_rows.append(nz / norm)
            if "n_per_deg2_bin" in g:
                n_per_deg2_rows.append(np.asarray(g["n_per_deg2_bin"][:], dtype=np.float64))
            if "dndz_per_deg2" in g:
                dndz_per_deg2_rows.append(np.asarray(g["dndz_per_deg2"][:], dtype=np.float64))
            mean = float(g.attrs.get("mean_z", np.sum(z_mid * nz * dz) / norm))
            mean_z.append(mean)
            sigma_z.append(float(g.attrs.get("sigma_z", np.sqrt(np.sum((z_mid - mean) ** 2 * nz * dz) / norm))))
            surface_density.append(float(g.attrs.get("surface_density_per_deg2", np.nan)))

    return {
        "source_hdf5": str(path),
        "group": group,
        "redshift_kind": "spectroscopic_calibrated_true_redshift",
        "sample": "full_valid_for_cl_dr9_extended_lrg",
        "normalized_kernel_dataset": DESI_DR9_TRUE_NZ_DATASET,
        "z_edges": z_edges,
        "z_mid": z_mid,
        "z_min": z_min,
        "z_max": z_max,
        "dz": dz,
        "nz_dndz_by_pz": np.asarray(dndz_rows, dtype=np.float64),
        "n_per_deg2_bin_by_pz": np.asarray(n_per_deg2_rows, dtype=np.float64) if n_per_deg2_rows else None,
        "dndz_per_deg2_by_pz": np.asarray(dndz_per_deg2_rows, dtype=np.float64) if dndz_per_deg2_rows else None,
        "mean_true_z_by_pz": np.asarray(mean_z, dtype=np.float64),
        "sigma_true_z_by_pz": np.asarray(sigma_z, dtype=np.float64),
        "surface_density_per_deg2_by_pz": np.asarray(surface_density, dtype=np.float64),
    }


def _clean_map(values: np.ndarray, dtype: str = "f4") -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)


def _clean_mask(values: np.ndarray) -> np.ndarray:
    mask = _clean_map(values, dtype="f4")
    mask[mask < 0.0] = 0.0
    return mask


def _clean_bounded_mask(values: np.ndarray) -> np.ndarray:
    """Clean a mask whose source convention is explicitly bounded in [0, 1]."""

    return np.clip(_clean_mask(values), 0.0, 1.0)


def _subtract_masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    good = mask > 0
    if not np.any(good):
        raise ValueError("Cannot subtract masked mean from a zero-overlap mask.")
    out = np.asarray(values, dtype=np.float32).copy()
    mean = float(np.sum(out[good] * mask[good]) / np.sum(mask[good]))
    out[good] -= mean
    out[~good] = 0.0
    return out


def _subtract_premasked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove a monopole from a map that already contains one factor of ``mask``.

    If ``values = mask * signal``, subtracting a bare constant would no longer
    leave a masked field when the mask is tapered.  The correct fixed-field
    projection is ``values - <signal> * mask``.
    """

    good = np.asarray(mask) > 0
    if not np.any(good):
        raise ValueError("Cannot subtract a premasked mean from a zero-overlap mask.")
    out = np.asarray(values, dtype=np.float32).copy()
    weight = np.asarray(mask, dtype=np.float64)
    mean = float(np.sum(out[good], dtype=np.float64) / np.sum(weight[good], dtype=np.float64))
    out[good] -= mean * weight[good]
    out[~good] = 0.0
    return out


def _weighted_mean_on_mask(values: np.ndarray, mask: np.ndarray) -> float:
    good = np.asarray(mask) > 0
    if not np.any(good):
        raise ValueError("Cannot compute a weighted mean on a zero-overlap mask.")
    arr = np.asarray(values, dtype=np.float64)
    w = np.asarray(mask, dtype=np.float64)
    return float(np.sum(arr[good] * w[good]) / np.sum(w[good]))


def apply_mask_apodization(
    fields: Mapping[str, FieldMap],
    config: MeasurementConfig,
) -> Dict[str, FieldMap]:
    """Return fields with NaMaster-apodized masks, preserving shared-mask refs.

    DES shear and DESI random-count masks are deliberately left unapodized.
    They contain isolated zero-count pixels, so treating every zero as an
    apodization boundary removes most of the effective footprint. ACT T and
    kappa masks are already tapered in their source products. Keeping these
    masks unchanged also preserves their catalog/noise normalization.
    """

    apo_deg = float(config.mask_apodization_deg)
    if apo_deg <= 0.0:
        out: Dict[str, FieldMap] = {}
        for name, field_map in fields.items():
            metadata = dict(field_map.metadata)
            metadata.update(
                {
                    "mask_apodization_applied": False,
                    "mask_apodization_deg": 0.0,
                    "mask_apodization_type": "none",
                    "mask_apodization_note": (
                        "No additional pipeline apodization was applied; any taper already "
                        "present in the source mask is preserved."
                    ),
                }
            )
            out[name] = replace(field_map, metadata=metadata)
        return out

    apotype = str(config.mask_apodization_type)
    apodized_masks: Dict[str, np.ndarray] = {}
    out: Dict[str, FieldMap] = {}
    for name, field_map in fields.items():
        preserve_reasons = {
            "des_shear": (
                "DES Y3 weighted-count shear mask kept unapodized. Apodizing its "
                "internal zero-count pixels destroys effective area and invalidates "
                "the catalog-derived shape-noise pseudo-Cl."
            ),
            "desi_galaxy": (
                "DESI random-count mask kept unapodized. Its internal zero-count pixels "
                "come from a sparse random realization and are not survey boundaries."
            ),
            "desi_momentum": (
                "DESI catalog-momentum selection mask kept unapodized so it remains "
                "matched to the random catalog used to normalize the estimator."
            ),
            "desi_momentum_null": (
                "DESI shuffled catalog-momentum selection mask kept unapodized so it "
                "remains matched to the random catalog normalization."
            ),
            "act_cmb_temperature": "ACT temperature mask is already apodized upstream.",
            "act_cmb_lensing_kappa": "ACT lensing mask is already apodized upstream.",
        }
        if field_map.kind in preserve_reasons:
            metadata = dict(field_map.metadata)
            metadata.update(
                {
                    "mask_apodization_applied": False,
                    "mask_apodization_deg": 0.0,
                    "mask_apodization_type": "none",
                    "mask_apodization_note": preserve_reasons[field_map.kind],
                }
            )
            out[name] = replace(field_map, metadata=metadata)
            continue
        if field_map.mask_name not in apodized_masks:
            mask = np.asarray(field_map.mask, dtype=np.float64)
            print(
                f"[{utc_now()}] Apodizing mask {field_map.mask_name} "
                f"({apo_deg:g} deg, {apotype})",
                flush=True,
            )
            apo_mask = nmt.mask_apodization(mask, apo_deg, apotype=apotype)
            apodized_masks[field_map.mask_name] = _clean_mask(apo_mask).astype(np.float32, copy=False)
            print(f"[{utc_now()}] Finished apodizing mask {field_map.mask_name}", flush=True)
        metadata = dict(field_map.metadata)
        metadata.update(
            {
                "mask_apodization_applied": True,
                "mask_apodization_deg": apo_deg,
                "mask_apodization_type": apotype,
                "mask_apodization_note": (
                    "Mask was apodized with pymaster.mask_apodization before "
                    "constructing NaMaster fields."
                ),
            }
        )
        out[name] = replace(field_map, mask=apodized_masks[field_map.mask_name], metadata=metadata)
    return out


def _accumulate_pixels(
    out: np.ndarray,
    pix: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> None:
    pix = np.asarray(pix, dtype=np.int64)
    if pix.size == 0:
        return
    order = np.argsort(pix)
    pix_sorted = pix[order]
    unique, start = np.unique(pix_sorted, return_index=True)
    if weights is None:
        vals = np.diff(np.r_[start, pix_sorted.size]).astype(out.dtype, copy=False)
    else:
        w_sorted = np.asarray(weights, dtype=np.float64)[order]
        vals = np.add.reduceat(w_sorted, start).astype(out.dtype, copy=False)
    out[unique] += vals


def make_sqrt_bandpower_edges(ell_min: int, ell_max: int, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return DES-Y3-style sqrt-spaced NaMaster bandpower edges.

    The DES Y3 harmonic-space shear product used equal-weight bins whose
    edges are uniformly spaced in sqrt(ell), rounded to integer ell, with
    right edges exclusive. This intentionally matches the transferred DES
    shear map preparation script.
    """

    edges = np.rint(np.linspace(np.sqrt(ell_min), np.sqrt(ell_max), n_bins + 1) ** 2).astype(np.int64)
    edges[0] = int(ell_min)
    edges[-1] = int(ell_max) + 1
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    if edges[-1] != int(ell_max) + 1:
        edges[-1] = int(ell_max) + 1
    return edges[:-1].astype(np.int32), edges[1:].astype(np.int32)


def make_linear_bandpower_edges(ell_min: int, ell_max: int, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return linearly-spaced integer NaMaster bandpower edges.

    ``ell_max`` is inclusive in the public configuration, while
    ``NmtBin.from_edges`` treats right edges as exclusive.
    """

    edges = np.ceil(np.linspace(int(ell_min), int(ell_max) + 1, int(n_bins) + 1)).astype(np.int64)
    edges[0] = int(ell_min)
    edges[-1] = int(ell_max) + 1
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1
    if edges[-1] != int(ell_max) + 1:
        edges[-1] = int(ell_max) + 1
    return edges[:-1].astype(np.int32), edges[1:].astype(np.int32)


MIDRES2048_LOG_ELL_LEFT_3000 = np.asarray(
    [128, 160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315, 1730, 2280],
    dtype=np.int32,
)
MIDRES2048_LOG_ELL_RIGHT_3000 = np.asarray(
    [160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315, 1730, 2280, 3001],
    dtype=np.int32,
)

MIDRES2048_LOG_ELL_LEFT_4096 = np.asarray(
    [128, 160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315, 1730, 2280, 3001, 3329, 3693],
    dtype=np.int32,
)
MIDRES2048_LOG_ELL_RIGHT_4096 = np.asarray(
    [160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315, 1730, 2280, 3001, 3329, 3693, 4097],
    dtype=np.int32,
)

# Preserve the established thirteen ACT-supported bands through inclusive
# ell=3000, then extend the common grid with seven nearly equal logarithmic
# intervals through inclusive ell=8192.  Right edges are exclusive.
HIGHRES4096_LOG_ELL_LEFT_8192 = np.asarray(
    [
        128, 160, 200, 255, 320, 400, 500, 630, 795, 1000,
        1315, 1730, 2280, 3001, 3464, 3998, 4615, 5327, 6149, 7098,
    ],
    dtype=np.int32,
)
HIGHRES4096_LOG_ELL_RIGHT_8192 = np.asarray(
    [
        160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315,
        1730, 2280, 3001, 3464, 3998, 4615, 5327, 6149, 7098, 8193,
    ],
    dtype=np.int32,
)


def make_log_bandpower_edges(ell_min: int, ell_max: int, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Stage-31 midres2048 hybrid-log NaMaster bandpower edges."""

    requested = (int(ell_min), int(ell_max), int(n_bins))
    edge_tables = {
        (128, 3000, 13): (MIDRES2048_LOG_ELL_LEFT_3000, MIDRES2048_LOG_ELL_RIGHT_3000),
        (128, 4096, 16): (MIDRES2048_LOG_ELL_LEFT_4096, MIDRES2048_LOG_ELL_RIGHT_4096),
        (128, 8192, 20): (HIGHRES4096_LOG_ELL_LEFT_8192, HIGHRES4096_LOG_ELL_RIGHT_8192),
    }
    if requested not in edge_tables:
        supported = ", ".join(str(key) for key in edge_tables)
        raise ValueError(
            "binning='log' uses a preregistered xDESI edge table; "
            f"supported ell_min/lmax/n_bins tuples are {supported}, got {requested}."
        )
    left, right = edge_tables[requested]
    return left.copy(), right.copy()


def make_bandpower_edges(config: MeasurementConfig) -> Tuple[np.ndarray, np.ndarray]:
    binning = str(config.binning).lower()
    if binning == "sqrt":
        return make_sqrt_bandpower_edges(config.ell_min, config.lmax, config.n_bins)
    if binning == "linear":
        return make_linear_bandpower_edges(config.ell_min, config.lmax, config.n_bins)
    if binning == "log":
        return make_log_bandpower_edges(config.ell_min, config.lmax, config.n_bins)
    raise ValueError(f"Unsupported binning={config.binning!r}; expected 'sqrt', 'linear', or 'log'.")


def make_bins(config: MeasurementConfig) -> nmt.NmtBin:
    config.validate()
    left, right = make_bandpower_edges(config)
    return nmt.NmtBin.from_edges(left, right)


def measurement_schema_for_config(config: MeasurementConfig | Mapping[str, object]) -> str:
    """Return the archive schema required by the configured packing policy.

    Keeping the validity-mask product on a new schema is intentional: an older
    checkout that knows only v1 must reject, rather than silently fit, the
    high-ell zero placeholders.
    """

    if isinstance(config, Mapping):
        cutoff = config.get("kappa_cmb_lmax")
    else:
        cutoff = config.kappa_cmb_lmax
    return SCHEMA_MEASUREMENT_VALIDITY_MASK if cutoff is not None else SCHEMA_MEASUREMENT


def data_vector_validity_by_spectrum(
    specs: Sequence[SpectrumSpec],
    config: MeasurementConfig,
    ell_left: np.ndarray,
    ell_right: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return the exact band-validity mask for every archived spectrum.

    The ACT lensing response is supported through inclusive ``kappa_cmb_lmax``.
    Validity is determined from complete band support, never from an effective
    ell.  A bin that straddles the cutoff is rejected as an invalid product
    definition rather than assigned ambiguously.
    """

    left = np.asarray(ell_left, dtype=np.int64)
    right = np.asarray(ell_right, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Bandpower left/right edges must be same-length one-dimensional arrays.")
    if left.size != int(config.n_bins):
        raise ValueError(f"Bandpower edge count {left.size} does not match n_bins={config.n_bins}.")
    if np.any(right <= left):
        raise ValueError("Every bandpower right edge must exceed its left edge.")

    all_valid = np.ones(left.size, dtype=bool)
    out = {spec.name: all_valid.copy() for spec in specs}
    if config.kappa_cmb_lmax is None:
        return out

    boundary = int(config.kappa_cmb_lmax) + 1
    if boundary not in set(right.tolist()):
        raise ValueError(
            f"kappa_cmb_lmax={config.kappa_cmb_lmax} requires an exact right-exclusive "
            f"band edge at {boundary}; the common grid would otherwise straddle the ACT response cutoff."
        )
    straddles = (left < boundary) & (right > boundary)
    if np.any(straddles):
        raise ValueError(
            f"Band(s) {np.flatnonzero(straddles).tolist()} straddle the ACT kappa cutoff at {boundary}."
        )
    kappa_valid = right <= boundary
    for spec in specs:
        if str(spec.family) == "desi_g_act_kappa":
            out[spec.name] = kappa_valid.copy()
    return out


def pack_joint_data_vector(
    specs: Sequence[SpectrumSpec],
    spectra: Mapping[str, Mapping[str, object]],
    config: MeasurementConfig,
    ell_left: np.ndarray,
    ell_right: np.ndarray,
) -> Dict[str, object]:
    """Pack total and conditional shot-subtracted archival data-vector views.

    ``data_vector`` remains the primary measured estimator: DESI galaxy autos
    contain weighted Poisson shot noise exactly once.  The secondary view is an
    affine shift by the saved, already-decoupled catalog template.  Its
    covariance is therefore the same estimator covariance conditional on the
    fixed catalog and mask; callers must not remove shot noise from covariance
    inputs when using that view.
    """

    validity = data_vector_validity_by_spectrum(specs, config, ell_left, ell_right)
    raw_chunks: List[np.ndarray] = []
    shot_subtracted_raw_chunks: List[np.ndarray] = []
    shot_template_chunks: List[np.ndarray] = []
    valid_chunks: List[np.ndarray] = []
    for spec in specs:
        spectrum = spectra[spec.name]
        values = np.asarray(spectrum["cl"], dtype=np.float64)
        valid = np.asarray(validity[spec.name], dtype=bool)
        if values.shape != valid.shape:
            raise ValueError(
                f"Spectrum {spec.name!r} has shape {values.shape}, expected {valid.shape} from the common grid."
            )
        shot_template = np.zeros_like(values)
        if str(spec.family) == "desi_g_auto":
            noise_all = spectrum.get("noise_decoupled_all_components")
            if noise_all is None:
                raise ValueError(
                    f"DESI galaxy auto {spec.name!r} has no saved decoupled shot-noise template."
                )
            noise_all = np.asarray(noise_all, dtype=np.float64)
            component = int(spec.component)
            if noise_all.ndim != 2 or not (0 <= component < noise_all.shape[0]):
                raise ValueError(
                    f"DESI galaxy auto {spec.name!r} component {component} is incompatible "
                    f"with shot-noise template shape {noise_all.shape}."
                )
            shot_template = np.asarray(noise_all[component], dtype=np.float64)
            if shot_template.shape != values.shape or not np.all(np.isfinite(shot_template)):
                raise ValueError(
                    f"DESI galaxy auto {spec.name!r} shot-noise template has invalid shape or values."
                )
        raw_chunks.append(values)
        shot_subtracted_raw_chunks.append(values - shot_template)
        shot_template_chunks.append(shot_template)
        valid_chunks.append(valid)
    raw = np.concatenate(raw_chunks)
    shot_subtracted_raw = np.concatenate(shot_subtracted_raw_chunks)
    shot_template = np.concatenate(shot_template_chunks)
    valid = np.concatenate(valid_chunks)
    packed = raw.copy()
    packed[~valid] = 0.0
    shot_subtracted_packed = shot_subtracted_raw.copy()
    shot_subtracted_packed[~valid] = 0.0
    return {
        "data_vector": packed,
        "data_vector_raw": raw,
        "data_vector_valid": valid,
        "data_vector_weighted_poisson_subtracted": shot_subtracted_packed,
        "data_vector_raw_weighted_poisson_subtracted": shot_subtracted_raw,
        "galaxy_auto_weighted_poisson_template": shot_template,
        "spectrum_validity": validity,
    }


def component_labels(spin_a: int, spin_b: int) -> List[str]:
    if spin_a == 0 and spin_b == 0:
        return ["00"]
    if spin_a == 0 and spin_b == 2:
        return ["0E", "0B"]
    if spin_a == 2 and spin_b == 0:
        return ["E0", "B0"]
    if spin_a == 2 and spin_b == 2:
        return ["EE", "EB", "BE", "BB"]
    raise ValueError(f"Unsupported spin pair ({spin_a}, {spin_b}).")


def ncls_for_spins(spin_a: int, spin_b: int) -> int:
    return len(component_labels(spin_a, spin_b))


def n_maps_for_spin(spin: int) -> int:
    if int(spin) == 0:
        return 1
    if int(spin) == 2:
        return 2
    raise ValueError(f"Unsupported NaMaster spin {spin}; this pipeline supports only spin 0 and spin 2.")


def validate_field_map_for_namaster(field_map: FieldMap) -> None:
    """Validate map/mask shapes before constructing a NaMaster field."""

    expected = n_maps_for_spin(field_map.spin)
    if len(field_map.maps) != expected:
        raise ValueError(
            f"Field {field_map.name!r} has spin={field_map.spin} and {len(field_map.maps)} map component(s); "
            f"expected {expected}."
        )
    mask = np.asarray(field_map.mask)
    if mask.ndim != 1:
        raise ValueError(f"Field {field_map.name!r} mask must be a 1D HEALPix RING array, got shape {mask.shape}.")
    if not np.any(np.isfinite(mask) & (mask > 0)):
        raise ValueError(f"Field {field_map.name!r} mask has no positive finite pixels.")
    for i, values in enumerate(field_map.maps):
        arr = np.asarray(values)
        if arr.shape != mask.shape:
            raise ValueError(
                f"Field {field_map.name!r} map{i} has shape {arr.shape}; expected mask shape {mask.shape}."
            )
    if field_map.kind in {"desi_momentum", "desi_momentum_null"}:
        if not field_map.has_catalog_momentum:
            raise ValueError(
                f"Field {field_map.name!r} is a kSZ momentum field but does not carry the "
                "catalog arrays required by NaMaster NmtFieldCatalogMomentum. Regenerate the "
                "map product with prepare_multiprobe_maps.py after the NaMaster 2.7 update."
            )
        lengths = {
            key: np.asarray(field_map.catalog[key]).size
            for key in ("ra_deg", "dec_deg", "weight", "field")
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Field {field_map.name!r} catalog arrays have inconsistent lengths: {lengths}.")
        if next(iter(lengths.values())) == 0:
            raise ValueError(f"Field {field_map.name!r} catalog arrays are empty.")
        for key in ("ra_deg", "dec_deg", "weight", "field"):
            arr = np.asarray(field_map.catalog[key])
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Field {field_map.name!r} catalog/{key} contains non-finite values.")
        if np.any(np.asarray(field_map.catalog["weight"], dtype=np.float64) <= 0.0):
            raise ValueError(f"Field {field_map.name!r} catalog weights must be strictly positive.")


def validate_spectrum_spec(spec: SpectrumSpec, fields: Mapping[str, NmtProbeField]) -> None:
    missing = [name for name in spec.fields if name not in fields]
    if missing:
        raise KeyError(f"Spectrum {spec.name!r} references missing field(s): {missing}")
    labels = component_labels(fields[spec.fields[0]].spin, fields[spec.fields[1]].spin)
    if not (0 <= int(spec.component) < len(labels)):
        raise ValueError(
            f"Spectrum {spec.name!r} asks for component {spec.component}, "
            f"but fields {spec.fields} expose components {labels}."
        )


def default_spectrum_specs() -> List[SpectrumSpec]:
    specs: List[SpectrumSpec] = []
    for i in range(1, 5):
        for j in range(i, 5):
            specs.append(
                SpectrumSpec(
                    name=f"des_shear_EE_tomo{i}_tomo{j}",
                    family="des_shear_EE",
                    fields=(f"s{i}", f"s{j}"),
                    component=0,
                    label=rf"DES shear EE tomo {i} x {j}",
                    theory_key=f"des_shear_EE_tomo{i}_tomo{j}",
                    metadata={"source_tomo_i": i, "source_tomo_j": j},
                )
            )
    for i in range(1, 5):
        specs.append(
            SpectrumSpec(
                name=f"act_y_des_shear_E_tomo{i}",
                family="act_y_des_shear_E",
                fields=("y", f"s{i}"),
                component=0,
                label=rf"ACT y x DES shear E tomo {i}",
                theory_key=f"act_y_des_shear_E_tomo{i}",
                metadata={"source_tomo": i},
            )
        )
    for i in range(1, 5):
        specs.append(
            SpectrumSpec(
                name=f"desi_g_auto_pz{i}",
                family="desi_g_auto",
                fields=(f"g{i}", f"g{i}"),
                component=0,
                label=rf"DESI g auto pz {i}",
                theory_key=f"desi_g_auto_pz{i}",
                metadata={"desi_pz": i},
            )
        )
    for i in range(1, 5):
        specs.append(
            SpectrumSpec(
                name=f"desi_g_act_y_pz{i}",
                family="desi_g_act_y",
                fields=(f"g{i}", "y"),
                component=0,
                label=rf"DESI g pz {i} x ACT y",
                theory_key=f"desi_g_act_y_pz{i}",
                metadata={"desi_pz": i},
            )
        )
    for i in range(1, 5):
        for j in range(1, 5):
            specs.append(
                SpectrumSpec(
                    name=f"desi_g_des_shear_E_pz{i}_tomo{j}",
                    family="desi_g_des_shear_E",
                    fields=(f"g{i}", f"s{j}"),
                    component=0,
                    label=rf"DESI g pz {i} x DES shear E tomo {j}",
                    theory_key=f"desi_g_des_shear_E_pz{i}_tomo{j}",
                    metadata={"desi_pz": i, "source_tomo": j},
                )
            )
    for i in range(1, 5):
        specs.append(
            SpectrumSpec(
                name=f"desi_g_act_kappa_pz{i}",
                family="desi_g_act_kappa",
                fields=(f"g{i}", "kappa"),
                component=0,
                label=rf"DESI g pz {i} x ACT kappa",
                theory_key=f"desi_g_act_kappa_pz{i}",
                metadata={"desi_pz": i},
            )
        )
    for i in range(1, 5):
        specs.append(
            SpectrumSpec(
                name=f"desi_pi_act_T_pz{i}",
                family="desi_pi_act_T",
                fields=(f"pi{i}", "T"),
                component=0,
                label=rf"DESI pi pz {i} x ACT T",
                theory_key=f"desi_g_tau_pz{i}",
                metadata={"desi_pz": i, "ksz_model": "-T_CMB_uK * A_v_bin * C_ell^g_tau"},
            )
        )
    return specs


def h5_attrs_to_jsonable(attrs: h5py.AttributeManager, max_string: int = 4000) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key in attrs:
        val = attrs[key]
        if isinstance(val, bytes):
            val = val.decode("utf-8", errors="replace")
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, str) and len(val) > max_string:
            val = val[:max_string] + "...[truncated]"
        if isinstance(val, (np.integer, np.floating)):
            val = val.item()
        out[str(key)] = val
    return out


def read_enmap_from_h5(path: Path, dataset: str, header_attr: str, downgrade: int = 1) -> enmap.ndmap:
    with h5py.File(path, "r") as h5:
        arr = h5[dataset][:]
        header_string = str(h5["geometry"].attrs[header_attr])
    sep = "\n" if "\n" in header_string.strip() else ""
    header = fits.Header.fromstring(header_string, sep=sep)
    wcs = WCS(header)
    em = enmap.enmap(arr, wcs)
    if int(downgrade) > 1:
        em = enmap.downgrade(em, int(downgrade))
    return em


def enmap_h5_to_healpix(
    path: Path,
    map_dataset: str,
    map_header_attr: str,
    mask_dataset: str,
    mask_header_attr: str,
    nside: int,
    lmax: int,
    downgrade: int,
    subtract_mean: bool,
    map_is_masked: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    map_em = read_enmap_from_h5(path, map_dataset, map_header_attr, downgrade=downgrade)
    mask_em = read_enmap_from_h5(path, mask_dataset, mask_header_attr, downgrade=downgrade)
    # Harmonic reprojection is appropriate for sky maps, but it rings at mask
    # edges and creates positive support outside the actual footprint. Pixell
    # recommends local spline interpolation for masks.
    mask_hp = reproject.map2healpix(mask_em, nside=nside, method="spline", order=1, spin=[0])
    map_hp = reproject.map2healpix(map_em, nside=nside, lmax=lmax)
    mask_hp = _clean_bounded_mask(mask_hp)
    map_hp = _clean_map(map_hp)
    if subtract_mean:
        if map_is_masked:
            map_hp = _subtract_premasked_mean(map_hp, mask_hp)
        else:
            map_hp = _subtract_masked_mean(map_hp, mask_hp)
    else:
        map_hp[mask_hp <= 0] = 0.0
    return map_hp.astype(np.float32, copy=False), mask_hp.astype(np.float32, copy=False)


def _load_healpix_random_count_map(
    path: Path,
    nside: int,
    allowed_nsides: Sequence[int] = DESI_DR9_SUPPORTED_RANDOM_NSIDE,
) -> np.ndarray:
    nside = int(nside)
    allowed = tuple(int(x) for x in allowed_nsides)
    if nside not in allowed:
        raise ValueError(
            f"DESI DR9 random-count maps are available only for nside={allowed}; got nside={nside}."
        )
    dataset = f"nside{nside}/random_count"
    with h5py.File(path, "r") as h5:
        ordering = str(h5.attrs.get("ordering", "")).upper()
        if ordering and ordering != "RING":
            raise ValueError(f"DESI DR9 random-count map must be RING ordered, got {ordering!r}.")
        if dataset not in h5:
            raise KeyError(f"Missing DESI DR9 random-count dataset {dataset!r} in {path}.")
        counts = np.asarray(h5[dataset][:], dtype=np.float32)
    expected = hp.nside2npix(nside)
    if counts.shape != (expected,):
        raise ValueError(f"{dataset} has shape {counts.shape}, expected {(expected,)}.")
    return counts


def sum_preserving_ud_grade_counts(
    counts: np.ndarray,
    nside_out: int,
    *,
    nside_in: Optional[int] = None,
) -> np.ndarray:
    """Downgrade or upgrade extensive HEALPix count maps preserving total counts."""

    arr = np.asarray(counts, dtype=np.float64)
    if nside_in is None:
        nside_in = hp.npix2nside(arr.size)
    out = hp.ud_grade(arr, nside_out=int(nside_out), power=-2, order_in="RING", order_out="RING")
    return np.asarray(out, dtype=np.float32)


def load_dr9_random_counts_with_metadata(
    bundle: SurveyBundle,
    config: MeasurementConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Load DR9 random counts, deriving nside=2048 from 4096 if needed."""

    nside = int(config.nside)
    dataset = f"nside{nside}/random_count"
    metadata: Dict[str, object] = {
        "random_count_source": str(bundle.desi_random_count_maps),
        "random_count_dataset": dataset,
        "random_count_nside": nside,
        "random_count_derivation": "native",
    }
    random_manifest = bundle.manifest["products"]["desi_dr9_imaging_randoms"]
    if not isinstance(random_manifest, Mapping):
        raise ValueError("DESI random-count manifest entry must be a mapping.")
    strict_multi_random = int(config.minimum_desi_random_realizations) > 1
    with h5py.File(bundle.desi_random_count_maps, "r") as h5:
        ordering = str(h5.attrs.get("ordering", "")).upper()
        if ordering and ordering != "RING":
            raise ValueError(f"DESI DR9 random-count map must be RING ordered, got {ordering!r}.")
        metadata["random_count_ordering"] = ordering or "RING"
        realization_count = int(h5.attrs.get("n_random_realizations", h5.attrs.get("random_realization_count", 1)))
        metadata["random_realization_count"] = realization_count
        if "random_realization_indices" in h5.attrs:
            realization_indices = [
                int(value) for value in np.asarray(h5.attrs["random_realization_indices"]).reshape(-1)
            ]
        elif "random_realization_indices_json" in h5.attrs:
            try:
                realization_indices = [int(value) for value in json.loads(
                    str(h5.attrs["random_realization_indices_json"])
                )]
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("DESI random-count product has invalid random_realization_indices_json.") from exc
        else:
            realization_indices = []
        metadata["random_realization_indices"] = realization_indices
        if realization_indices:
            if len(realization_indices) != realization_count:
                raise ValueError(
                    "DESI random-count realization index count does not match its realization-count attribute."
                )
            if len(set(realization_indices)) != len(realization_indices):
                raise ValueError("DESI random-count realization indices contain duplicates.")
        if realization_count < int(config.minimum_desi_random_realizations):
            raise ValueError(
                f"DESI random-count product has {realization_count} realization(s), but stage "
                f"{config.stage!r} requires at least {config.minimum_desi_random_realizations}."
            )
        expected_count = random_manifest.get("random_realization_count")
        if expected_count is not None and realization_count != int(expected_count):
            raise ValueError(
                f"DESI random-count product records {realization_count} realizations, but the manifest "
                f"requires {int(expected_count)}."
            )
        expected_indices = tuple(int(value) for value in random_manifest.get("random_indices", ()))
        if expected_indices and tuple(realization_indices) != expected_indices:
            raise ValueError(
                f"DESI random-count realization indices {realization_indices} do not match the "
                f"manifest set {list(expected_indices)}."
            )
        expected_identity = str(random_manifest.get("input_identity_sha256", ""))
        product_identity = str(h5.attrs.get("input_identity_sha256", ""))
        metadata["random_count_input_identity_sha256"] = product_identity
        if expected_identity and product_identity != expected_identity:
            raise ValueError(
                "DESI random-count input identity does not match the survey manifest."
            )
        expected_schema = str(random_manifest.get("random_product_schema", ""))
        product_schema = str(h5.attrs.get("schema_version", ""))
        metadata["random_count_schema_version"] = product_schema
        if expected_schema and product_schema != expected_schema:
            raise ValueError(
                f"DESI random-count schema {product_schema!r} does not match manifest "
                f"schema {expected_schema!r}."
            )
        if strict_multi_random:
            missing_contract = []
            if not expected_identity or not product_identity:
                missing_contract.append("input_identity_sha256")
            if not expected_indices or not realization_indices:
                missing_contract.append("random realization indices")
            if not expected_schema or not product_schema:
                missing_contract.append("random product schema")
            try:
                full_source_sha256_verified = int(
                    h5.attrs.get("full_source_sha256_verified", 0)
                )
            except (TypeError, ValueError):
                full_source_sha256_verified = 0
            sha256_ledger_sha256_raw = h5.attrs.get("sha256_ledger_sha256", "")
            if isinstance(sha256_ledger_sha256_raw, bytes):
                sha256_ledger_sha256_raw = sha256_ledger_sha256_raw.decode("utf-8")
            sha256_ledger_sha256 = str(sha256_ledger_sha256_raw).strip()
            full_source_sha256_json_raw = h5.attrs.get("full_source_sha256_json", "")
            if isinstance(full_source_sha256_json_raw, bytes):
                full_source_sha256_json_raw = full_source_sha256_json_raw.decode("utf-8")
            full_source_sha256_json = str(full_source_sha256_json_raw).strip()
            if full_source_sha256_verified != 1:
                missing_contract.append("full_source_sha256_verified=1")
            if not sha256_ledger_sha256:
                missing_contract.append("sha256_ledger_sha256")
            elif len(sha256_ledger_sha256) != 64 or any(
                character not in "0123456789abcdefABCDEF"
                for character in sha256_ledger_sha256
            ):
                raise ValueError(
                    "Multi-random production has an invalid sha256_ledger_sha256 digest."
                )
            full_source_sha256: object = {}
            if not full_source_sha256_json:
                missing_contract.append("full_source_sha256_json")
            else:
                try:
                    full_source_sha256 = json.loads(full_source_sha256_json)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Multi-random production has invalid full_source_sha256_json provenance."
                    ) from exc
                if not isinstance(full_source_sha256, Mapping) or not full_source_sha256:
                    missing_contract.append("nonempty full_source_sha256_json mapping")
            if missing_contract:
                raise ValueError(
                    "Multi-random production requires manifest/HDF provenance for "
                    + ", ".join(missing_contract)
                    + "."
                )
            expected_source_paths = {
                *(
                    f"randoms/resolve/randoms-1-{index}.fits"
                    for index in expected_indices
                ),
                *(
                    "zhou-lrg-xcorr-2023-v1/catalogs/lrgmask_v1.1/"
                    f"randoms-1-{index}-lrgmask_v1.1.fits.gz"
                    for index in expected_indices
                ),
                (
                    "zhou-lrg-xcorr-2023-v1/misc/"
                    "pixweight-dr7.1-0.22.0_stardens_64_ring.fits"
                ),
                "zhou-lrg-xcorr-2023-v1/catalogs/randoms_quality_cuts.py",
            }
            if isinstance(full_source_sha256, Mapping) and full_source_sha256:
                observed_source_paths = {str(key) for key in full_source_sha256}
                if observed_source_paths != expected_source_paths:
                    raise ValueError(
                        "Multi-random production full_source_sha256_json does not contain "
                        "the exact selected random/mask and shared-source inventory."
                    )
                invalid_source_digests = [
                    str(path)
                    for path, digest in full_source_sha256.items()
                    if len(str(digest)) != 64
                    or any(
                        character not in "0123456789abcdefABCDEF"
                        for character in str(digest)
                    )
                ]
                if invalid_source_digests:
                    raise ValueError(
                        "Multi-random production has invalid full-source SHA256 digest(s) "
                        f"for {invalid_source_digests}."
                    )
                source_inventory_sha256 = hashlib.sha256(
                    json.dumps(
                        dict(full_source_sha256),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                manifest_ledger_sha256 = str(
                    random_manifest.get("sha256_ledger_sha256", "")
                )
                manifest_inventory_sha256 = str(
                    random_manifest.get("full_source_sha256_inventory_sha256", "")
                )
                if not manifest_ledger_sha256:
                    missing_contract.append("manifest sha256_ledger_sha256")
                elif manifest_ledger_sha256 != sha256_ledger_sha256:
                    raise ValueError(
                        "DESI random-count SHA256-ledger identity does not match the survey manifest."
                    )
                if not manifest_inventory_sha256:
                    missing_contract.append(
                        "manifest full_source_sha256_inventory_sha256"
                    )
                elif manifest_inventory_sha256 != source_inventory_sha256:
                    raise ValueError(
                        "DESI random-count full-source inventory does not match the survey manifest."
                    )
            if missing_contract:
                raise ValueError(
                    "Multi-random production requires manifest/HDF provenance for "
                    + ", ".join(missing_contract)
                    + "."
                )
            metadata["full_source_sha256_verified"] = True
            metadata["sha256_ledger_sha256"] = sha256_ledger_sha256
            metadata["full_source_sha256"] = dict(full_source_sha256)
            metadata["full_source_sha256_inventory_sha256"] = source_inventory_sha256
        if dataset in h5:
            raw_counts = np.asarray(h5[dataset][:])
            expected_digest = str(h5[f"nside{nside}"].attrs.get("random_count_sha256", ""))
            if strict_multi_random and not expected_digest:
                raise ValueError(f"DESI random-count dataset {dataset!r} has no content checksum.")
            if expected_digest:
                actual_digest = _array_raw_sha256(raw_counts)
                if actual_digest != expected_digest:
                    raise ValueError(
                        f"DESI random-count dataset {dataset!r} failed its content checksum."
                    )
                metadata["random_count_sha256"] = actual_digest
            expected_sum = h5[f"nside{nside}"].attrs.get("count_sum")
            actual_sum = int(np.sum(raw_counts, dtype=np.uint64))
            if expected_sum is not None and actual_sum != int(expected_sum):
                raise ValueError(
                    f"DESI random-count dataset {dataset!r} sum {actual_sum} does not match "
                    f"its recorded count_sum={int(expected_sum)}."
                )
            counts = np.asarray(raw_counts, dtype=np.float32)
            metadata["random_count_sum"] = actual_sum
        elif nside == 2048 and "nside4096/random_count" in h5:
            source_dataset = "nside4096/random_count"
            source_counts = np.asarray(h5[source_dataset][:], dtype=np.float64)
            counts = sum_preserving_ud_grade_counts(source_counts, 2048, nside_in=4096)
            metadata.update(
                {
                    "random_count_dataset": source_dataset,
                    "random_count_source_dataset": source_dataset,
                    "random_count_target_dataset": dataset,
                    "random_count_derivation": "sum_preserving_ud_grade_from_nside4096",
                    "random_count_source_nside": 4096,
                    "random_count_sum_source": float(np.sum(source_counts, dtype=np.float64)),
                    "random_count_sum_target": float(np.sum(counts, dtype=np.float64)),
                }
            )
        else:
            available = sorted(k for k in h5 if str(k).startswith("nside"))
            raise KeyError(
                f"Missing DESI DR9 random-count dataset {dataset!r} in {bundle.desi_random_count_maps}; "
                f"available nside groups are {available}."
            )
    expected = hp.nside2npix(nside)
    if counts.shape != (expected,):
        raise ValueError(f"{metadata['random_count_dataset']} has shape {counts.shape}, expected {(expected,)}.")
    return counts, metadata


def load_dr9_random_counts(bundle: SurveyBundle, config: MeasurementConfig) -> np.ndarray:
    counts, _ = load_dr9_random_counts_with_metadata(bundle, config)
    return counts


def covariance_alias_for_field_name(field_name: str) -> str:
    """Return the conservative mask/spin alias used for covariance grouping."""

    if field_name.startswith("g") or field_name.startswith("pi"):
        return "desi_dr9_scalar"
    if field_name.startswith("s"):
        return f"des_shear_{field_name}"
    if field_name == "y":
        return "act_y_scalar"
    if field_name == "T":
        return "act_T_scalar"
    if field_name == "kappa":
        return "act_kappa_scalar"
    return field_name


def covariance_group_key_for_specs(spec_a: SpectrumSpec, spec_b: SpectrumSpec) -> Tuple[str, str, str, str]:
    return tuple(covariance_alias_for_field_name(name) for name in (*spec_a.fields, *spec_b.fields))


def _hist_density(
    values: np.ndarray,
    edges: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    if weights is not None:
        finite &= np.isfinite(weights) & (weights > 0)
        hist_weights = np.asarray(weights[finite], dtype=np.float64)
    else:
        hist_weights = None
    counts, _ = np.histogram(values[finite], bins=edges, weights=hist_weights)
    mid = 0.5 * (edges[1:] + edges[:-1])
    norm = _trapz(counts.astype(np.float64), x=mid)
    if norm > 0:
        dndz = counts.astype(np.float64) / norm
    else:
        dndz = counts.astype(np.float64)
    return counts, dndz.astype(np.float64)


def _weighted_mean_std_rms(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float]:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan, np.nan, np.nan
    x = np.asarray(values[good], dtype=np.float64)
    w = np.asarray(weights[good], dtype=np.float64)
    sumw = float(np.sum(w))
    if sumw <= 0:
        return np.nan, np.nan, np.nan
    mean = float(np.sum(w * x) / sumw)
    var = float(np.sum(w * (x - mean) ** 2) / sumw)
    rms = float(np.sqrt(np.sum(w * x**2) / sumw))
    return mean, float(np.sqrt(max(var, 0.0))), rms


def build_desi_fields(
    bundle: SurveyBundle,
    config: MeasurementConfig,
    random_counts: np.ndarray,
    random_count_metadata: Optional[Mapping[str, object]] = None,
) -> Tuple[Dict[str, FieldMap], Dict[str, object]]:
    npix = hp.nside2npix(config.nside)
    if len(random_counts) != npix:
        raise ValueError(f"DESI random-count map has length {len(random_counts)}, expected {npix}.")
    pixarea = hp.nside2pixarea(config.nside)
    random_counts = np.asarray(random_counts, dtype=np.float64)
    valid = random_counts > 0
    if not np.any(valid):
        raise ValueError("DESI random mask has no valid pixels after cuts.")
    random_mean = float(np.mean(random_counts[valid]))
    random_sum_valid = float(np.sum(random_counts[valid]))
    desi_mask = np.zeros(npix, dtype=np.float32)
    desi_mask[valid] = random_counts[valid] / random_mean
    area_sr = float(pixarea * np.sum(desi_mask, dtype=np.float64))

    random_attrs: Dict[str, object] = {}
    with h5py.File(bundle.desi_random_count_maps, "r") as h5:
        random_attrs = h5_attrs_to_jsonable(h5.attrs)
    random_count_metadata = dict(random_count_metadata or {})
    random_count_dataset = str(random_count_metadata.get("random_count_dataset", f"nside{config.nside}/random_count"))
    random_count_derivation = str(random_count_metadata.get("random_count_derivation", "native"))

    fields: Dict[str, FieldMap] = {}
    true_nz = load_desi_dr9_calibrated_true_nz(bundle.desi_true_nz, DESI_DR9_TRUE_NZ_GROUP_FULL_CL)
    photoz_edges = np.linspace(0.0, 2.0, 201)
    vr_edges = np.linspace(-0.01, 0.01, 201)
    nz_counts_unweighted = []
    nz_weighted_counts = []
    nz_dndz = []
    nz_dndz_unweighted = []
    vr_counts = []
    vr_weighted_counts = []
    bin_summary: Dict[str, object] = {
        "desi_release": "DR9 Extended LRG",
        "catalog_source": str(bundle.desi_catalog),
        "random_count_source": str(bundle.desi_random_count_maps),
        "random_count_dataset": random_count_dataset,
        "random_count_derivation": random_count_derivation,
        "random_count_metadata": random_count_metadata,
        "random_count_ordering": random_attrs.get("ordering", "RING"),
        "selection_dataset": DESI_DR9_SELECTION_DATASET,
        "weight_dataset": DESI_DR9_WEIGHT_DATASET,
        "nside": int(config.nside),
        "z_edges": true_nz["z_edges"],
        "z_mid": true_nz["z_mid"],
        "dz": true_nz["dz"],
        "z_min": true_nz["z_min"],
        "z_max": true_nz["z_max"],
        "redshift_kind": true_nz["redshift_kind"],
        "theory_nz_source_hdf5": true_nz["source_hdf5"],
        "theory_nz_group": true_nz["group"],
        "theory_nz_sample": true_nz["sample"],
        "theory_nz_dataset": true_nz["normalized_kernel_dataset"],
        "theory_nz_note": (
            "Primary nz/desi/* arrays are calibrated true-redshift kernels for theory. "
            "Catalog Z_PHOT_MEDIAN histograms are saved only under nz/desi_photoz_diagnostic."
        ),
        "vr_edges": vr_edges,
        "vr_mid": 0.5 * (vr_edges[1:] + vr_edges[:-1]),
        "bins": {},
    }
    sigma_true_calibration = load_ksz_sigma_true_gas_calibration(bundle.sigma_true_gas_calibration)
    sigma_true_over_c_by_pz = sigma_true_calibration["sigma_true_gas_over_c"]
    sigma_true_km_s_by_pz = sigma_true_calibration["sigma_true_gas_km_s"]
    bin_summary["ksz_sigma_true_gas_calibration"] = sigma_true_calibration
    rng = np.random.default_rng(config.ksz_shuffle_seed)

    with h5py.File(bundle.desi_catalog, "r") as h5:
        product_type = str(h5.attrs.get("product_type", ""))
        n_objects = int(h5.attrs.get("n_objects", 0))
        n_valid_for_cl = int(h5.attrs.get("n_valid_for_cl", 0))
        cat = h5["catalog"]
        ra_all = np.asarray(cat["ra_deg"][:], dtype=np.float64)
        dec_all = np.asarray(cat["dec_deg"][:], dtype=np.float64)
        z_all = np.asarray(cat["z"][:], dtype=np.float64)
        vr_all = np.asarray(cat["vr_over_c"][:], dtype=np.float64)
        pz_all = np.asarray(cat["pz_bin"][:], dtype=np.int16)
        valid_for_cl_all = np.asarray(cat["valid_for_cl"][:], dtype=bool)
        weight_all = np.asarray(cat["weight_imaging_mean1"][:], dtype=np.float64)
        bin_summary["catalog_product_type"] = product_type
        bin_summary["catalog_n_objects"] = n_objects
        bin_summary["catalog_n_valid_for_cl"] = n_valid_for_cl
        for pz_bin in range(1, 5):
            selected = (
                (pz_all == pz_bin)
                & valid_for_cl_all
                & np.isfinite(ra_all)
                & np.isfinite(dec_all)
                & np.isfinite(z_all)
                & np.isfinite(vr_all)
                & np.isfinite(weight_all)
                & (weight_all > 0)
            )
            n_selected_before_mask = int(np.count_nonzero(selected))
            if n_selected_before_mask == 0:
                raise ValueError(f"No valid DR9 DESI galaxies found for pz_bin={pz_bin}.")

            ra = ra_all[selected]
            dec = dec_all[selected]
            z = z_all[selected]
            vr = vr_all[selected]
            weights = weight_all[selected]
            pix = hp.ang2pix(config.nside, ra, dec, lonlat=True, nest=False)
            in_random_mask = valid[pix]
            n_excluded_by_random_mask = int(np.count_nonzero(~in_random_mask))
            if n_excluded_by_random_mask:
                ra = ra[in_random_mask]
                dec = dec[in_random_mask]
                z = z[in_random_mask]
                vr = vr[in_random_mask]
                weights = weights[in_random_mask]
                pix = pix[in_random_mask]
            if pix.size == 0:
                raise ValueError(f"No DR9 DESI galaxies remain inside the random mask for pz_bin={pz_bin}.")

            mean_vr_weighted, sigma_rec_weighted, rms_rec_weighted = _weighted_mean_std_rms(vr, weights)
            _, sigma_rec_unweighted, rms_rec_unweighted = _weighted_mean_std_rms(vr, np.ones_like(vr))

            counts = np.zeros(npix, dtype=np.float32)
            vsum = np.zeros(npix, dtype=np.float32)
            _accumulate_pixels(counts, pix, weights=weights)
            _accumulate_pixels(vsum, pix, weights=weights * vr)

            sumw = float(np.sum(weights, dtype=np.float64))
            sumw2 = float(np.sum(weights**2, dtype=np.float64))
            alpha = float(np.sum(counts[valid], dtype=np.float64) / random_sum_valid)
            expected = alpha * random_counts
            delta = np.zeros(npix, dtype=np.float32)
            pi_map = np.zeros(npix, dtype=np.float32)
            denom_good = valid & (expected > 0)
            delta[denom_good] = counts[denom_good] / expected[denom_good] - 1.0
            pi_map[denom_good] = vsum[denom_good] / expected[denom_good]
            delta[~denom_good] = 0.0
            pi_map[~denom_good] = 0.0
            if config.subtract_masked_mean:
                pi_map = _subtract_masked_mean(pi_map, desi_mask)

            n_gal = int(pix.size)
            shot = float(area_sr * sumw2 / sumw**2)
            # The observed masked overdensity is G/(alpha*rbar)-R/rbar.
            # Its conditional galaxy-Poisson bias is therefore a constant
            # *coupled pseudo-Cl*, not a homogeneous full-sky Cl to be coupled
            # through the variable random-count mask a second time.
            shot_pseudo_cl = float(
                pixarea * sumw2 / (float(npix) * (alpha * random_mean) ** 2)
            )
            counts_z_unweighted, dndz_unweighted = _hist_density(z, photoz_edges)
            counts_z_weighted, dndz = _hist_density(z, photoz_edges, weights=weights)
            counts_vr, _ = np.histogram(vr[np.isfinite(vr)], bins=vr_edges)
            counts_vr_weighted, _ = np.histogram(
                vr[np.isfinite(vr)],
                bins=vr_edges,
                weights=weights[np.isfinite(vr)],
            )
            nz_counts_unweighted.append(counts_z_unweighted.astype(np.int64))
            nz_weighted_counts.append(counts_z_weighted.astype(np.float64))
            nz_dndz.append(dndz)
            nz_dndz_unweighted.append(dndz_unweighted)
            vr_counts.append(counts_vr.astype(np.int64))
            vr_weighted_counts.append(counts_vr_weighted.astype(np.float64))

            true_mean_z = float(true_nz["mean_true_z_by_pz"][pz_bin - 1])
            true_sigma_z = float(true_nz["sigma_true_z_by_pz"][pz_bin - 1])
            true_surface_density = float(true_nz["surface_density_per_deg2_by_pz"][pz_bin - 1])
            meta = {
                "desi_release": "DR9 Extended LRG",
                "source_hdf5": str(bundle.desi_catalog),
                "catalog_product_type": product_type,
                "selection_dataset": DESI_DR9_SELECTION_DATASET,
                "weight_dataset": DESI_DR9_WEIGHT_DATASET,
                "random_count_source": str(bundle.desi_random_count_maps),
                "random_count_dataset": random_count_dataset,
                "random_count_derivation": random_count_derivation,
                "random_count_metadata": random_count_metadata,
                "random_count_ordering": random_attrs.get("ordering", "RING"),
                "desi_pz_bin": pz_bin,
                "n_gal": n_gal,
                "n_selected_before_random_mask": n_selected_before_mask,
                "n_excluded_by_random_mask": n_excluded_by_random_mask,
                "alpha_galaxy_to_random": alpha,
                "area_sr": area_sr,
                "fsky_weighted": float(area_sr / (4.0 * np.pi)),
                "shot_noise": shot,
                "shot_noise_convention": (
                    "full-sky-equivalent weighted Poisson level: "
                    "area_sr * sum(weight^2) / sum(weight)^2; provenance only"
                ),
                "shot_noise_pseudo_cl": shot_pseudo_cl,
                "shot_noise_pseudo_cl_convention": (
                    "exact conditional coupled pseudo-Cl for M*delta = "
                    "G/(alpha*random_mean)-R/random_mean: "
                    "Omega_pix*sum(weight^2)/(Npix*(alpha*random_mean)^2)"
                ),
                "random_catalog_shot_noise_subtracted": False,
                "random_catalog_shot_noise_note": (
                    "The transferred random-count realization is conditioned on as the selection mask. "
                    "Its finite-density uncertainty is not an additive Poisson term in this conditional estimator; "
                    "denser/independent randoms are required to test that uncertainty."
                ),
                "sum_weight": sumw,
                "sum_weight2": sumw2,
                "mean_weight": float(np.mean(weights)) if weights.size else np.nan,
                "nbar_per_sr": float(sumw / area_sr) if area_sr > 0 else np.nan,
                "n_eff_per_sr": float(sumw**2 / (sumw2 * area_sr)) if area_sr > 0 and sumw2 > 0 else np.nan,
                "sigma_rec_vr_over_c": sigma_rec_weighted,
                "rms_rec_vr_over_c": rms_rec_weighted,
                "sigma_rec_vr_over_c_weighted": sigma_rec_weighted,
                "rms_rec_vr_over_c_weighted": rms_rec_weighted,
                "sigma_rec_vr_over_c_unweighted": sigma_rec_unweighted,
                "rms_rec_vr_over_c_unweighted": rms_rec_unweighted,
                "ksz_sigma_rec_default": "weighted_rms_rec_vr_over_c",
                "sigma_true_gas_over_c": float(sigma_true_over_c_by_pz[pz_bin]),
                "sigma_true_gas_km_s": float(sigma_true_km_s_by_pz[pz_bin]),
                "ksz_sigma_true_gas_calibration_source": sigma_true_calibration.get("source", ""),
                "ksz_sigma_true_gas_calibration_path": sigma_true_calibration.get("path", ""),
                "ksz_sigma_true_gas_calibration_doc": sigma_true_calibration.get("doc", KSZ_SIGMA_TRUE_GAS_DOC),
                "ksz_sigma_true_gas_bin_assignment": sigma_true_calibration.get("bin_assignment", "by_z_photo"),
                "ksz_photoz_velocity_correlation_r": KSZ_PHOTOZ_VELOCITY_CORRELATION_R,
                "ksz_photoz_velocity_correlation_fracerr": KSZ_PHOTOZ_VELOCITY_CORRELATION_FRACERR,
                "ksz_velocity_calibration_reference": KSZ_REFERENCE_PAPER,
                "mean_z": float(np.average(z, weights=weights)) if z.size else np.nan,
                "mean_z_convention": "catalog_z_phot_median",
                "mean_photoz": float(np.average(z, weights=weights)) if z.size else np.nan,
                "mean_photoz_unweighted": float(np.mean(z)) if z.size else np.nan,
                "mean_true_z": true_mean_z,
                "sigma_true_z": true_sigma_z,
                "true_nz_surface_density_per_deg2": true_surface_density,
                "true_nz_source_hdf5": true_nz["source_hdf5"],
                "true_nz_group": true_nz["group"],
                "true_nz_dataset": true_nz["normalized_kernel_dataset"],
                "mean_z_unweighted": float(np.mean(z)) if z.size else np.nan,
                "median_z": float(np.median(z)) if z.size else np.nan,
                "mean_vr_over_c": mean_vr_weighted,
                "mean_vr_over_c_weighted": mean_vr_weighted,
                "mean_vr_over_c_unweighted": float(np.mean(vr)) if vr.size else np.nan,
                "random_counts_mean_valid": random_mean,
                "random_counts_sum_valid": random_sum_valid,
                "random_counts_n_valid_pixels": int(np.count_nonzero(valid)),
            }
            bin_summary["bins"][f"pz{pz_bin}"] = meta
            fields[f"g{pz_bin}"] = FieldMap(
                name=f"g{pz_bin}",
                label=f"DESI DR9 weighted galaxy overdensity pz {pz_bin}",
                kind="desi_galaxy",
                spin=0,
                maps=[delta],
                mask=desi_mask,
                mask_name="desi_dr9_random",
                metadata=meta,
            )
            fields[f"pi{pz_bin}"] = FieldMap(
                name=f"pi{pz_bin}",
                label=f"DESI DR9 weighted velocity momentum pz {pz_bin}",
                kind="desi_momentum",
                spin=0,
                maps=[pi_map],
                mask=desi_mask,
                mask_name="desi_dr9_random",
                metadata={
                    **meta,
                    "namaster_field_class": "NmtFieldCatalogMomentum",
                    "ksz_estimator": (
                        "Catalog momentum estimator: positions=(ra_deg, dec_deg), weights=weight_imaging_mean1, "
                        "field=vr_over_c, mask=DESI DR9 random-count mask. The diagnostic map is saved only for "
                        "visual inspection; spectra use the catalog field to avoid pixelized momentum aliasing."
                    ),
                    "catalog_field_is_weighted": False,
                    "catalog_lonlat": True,
                },
                catalog={
                    "ra_deg": np.asarray(ra, dtype=np.float64),
                    "dec_deg": np.asarray(dec, dtype=np.float64),
                    "weight": np.asarray(weights, dtype=np.float64),
                    "field": np.asarray(vr, dtype=np.float64),
                },
            )

            if config.include_ksz_velocity_shuffle:
                vr_shuf = vr.copy()
                rng.shuffle(vr_shuf)
                vsum_shuf = np.zeros(npix, dtype=np.float32)
                _accumulate_pixels(vsum_shuf, pix, weights=weights * vr_shuf)
                pi_shuf = np.zeros(npix, dtype=np.float32)
                pi_shuf[denom_good] = vsum_shuf[denom_good] / expected[denom_good]
                if config.subtract_masked_mean:
                    pi_shuf = _subtract_masked_mean(pi_shuf, desi_mask)
                fields[f"pi_shuf{pz_bin}"] = FieldMap(
                    name=f"pi_shuf{pz_bin}",
                    label=f"DESI DR9 weighted shuffled velocity momentum pz {pz_bin}",
                    kind="desi_momentum_null",
                    spin=0,
                    maps=[pi_shuf],
                    mask=desi_mask,
                    mask_name="desi_dr9_random",
                    metadata={
                        **meta,
                        "shuffle_seed": int(config.ksz_shuffle_seed),
                        "namaster_field_class": "NmtFieldCatalogMomentum",
                        "ksz_estimator": (
                            "Catalog shuffled-momentum null: same positions and imaging weights as pi, "
                            "with vr_over_c shuffled within the pz bin before constructing the catalog field."
                        ),
                        "catalog_field_is_weighted": False,
                        "catalog_lonlat": True,
                    },
                    catalog={
                        "ra_deg": np.asarray(ra, dtype=np.float64),
                        "dec_deg": np.asarray(dec, dtype=np.float64),
                        "weight": np.asarray(weights, dtype=np.float64),
                        "field": np.asarray(vr_shuf, dtype=np.float64),
                    },
                )

    bin_summary["nz_dndz_by_pz"] = np.asarray(true_nz["nz_dndz_by_pz"], dtype=np.float64)
    bin_summary["nz_mean_true_z_by_pz"] = np.asarray(true_nz["mean_true_z_by_pz"], dtype=np.float64)
    bin_summary["nz_sigma_true_z_by_pz"] = np.asarray(true_nz["sigma_true_z_by_pz"], dtype=np.float64)
    bin_summary["nz_surface_density_per_deg2_by_pz"] = np.asarray(true_nz["surface_density_per_deg2_by_pz"], dtype=np.float64)
    if true_nz["n_per_deg2_bin_by_pz"] is not None:
        bin_summary["nz_n_per_deg2_bin_by_pz"] = np.asarray(true_nz["n_per_deg2_bin_by_pz"], dtype=np.float64)
    if true_nz["dndz_per_deg2_by_pz"] is not None:
        bin_summary["nz_dndz_per_deg2_by_pz"] = np.asarray(true_nz["dndz_per_deg2_by_pz"], dtype=np.float64)
    bin_summary["photoz_diagnostic"] = {
        "description": "Weighted and unweighted histograms of catalog Z_PHOT_MEDIAN. Diagnostic only; do not use for theory kernels.",
        "redshift_kind": "catalog_z_phot_median",
        "z_edges": photoz_edges,
        "z_mid": 0.5 * (photoz_edges[1:] + photoz_edges[:-1]),
        "nz_counts_by_pz": np.asarray(nz_counts_unweighted, dtype=np.int64),
        "nz_counts_unweighted_by_pz": np.asarray(nz_counts_unweighted, dtype=np.int64),
        "nz_weighted_counts_by_pz": np.asarray(nz_weighted_counts, dtype=np.float64),
        "nz_dndz_by_pz": np.asarray(nz_dndz, dtype=np.float64),
        "nz_dndz_unweighted_by_pz": np.asarray(nz_dndz_unweighted, dtype=np.float64),
    }
    bin_summary["vr_counts_by_pz"] = np.asarray(vr_counts, dtype=np.int64)
    bin_summary["vr_weighted_counts_by_pz"] = np.asarray(vr_weighted_counts, dtype=np.float64)
    realization_count = int(random_count_metadata.get("random_realization_count", 1))
    default_random_caveat = (
        "The transferred bundle contains one DR9 random realization. Raw high-resolution "
        "DESI masks are sparse and provisional; combine independent randoms or validate a "
        "separate completeness-mask construction."
        if realization_count <= 1
        else (
            f"This mask combines {realization_count} independent DR9 random realizations. "
            "Its saved cumulative-support diagnostics must be retained, and finite-random/"
            "catalog-selection uncertainty is not part of the Gaussian/iNKA covariance."
        )
    )
    bin_summary["desi_random_mask_caveat"] = random_attrs.get(
        "caveat",
        default_random_caveat,
    )
    return fields, bin_summary


def build_shear_fields(bundle: SurveyBundle, config: MeasurementConfig) -> Dict[str, FieldMap]:
    path = bundle.shear_path_for_nside(config.nside)
    fields: Dict[str, FieldMap] = {}
    shear_sign = float(config.shear_e_to_kappa_sign)
    if shear_sign not in (-1.0, 1.0):
        raise ValueError("shear_e_to_kappa_sign must be +1 or -1.")
    with h5py.File(path, "r") as h5:
        for tomo in range(4):
            group = h5[f"maps/tomo{tomo}"]
            expected_noise_attrs = {
                "mask_weight": "shape_noise_pseudo_cl_normalized_weight_mask",
                "mask_weight_raw": "shape_noise_pseudo_cl_raw_weight_mask",
            }
            expected_noise_attr = expected_noise_attrs.get(str(config.shear_mask_dataset))
            if expected_noise_attr is None:
                raise ValueError(
                    f"Unsupported DES shear mask dataset {config.shear_mask_dataset!r}; "
                    f"expected one of {sorted(expected_noise_attrs)} so shape noise is unambiguous."
                )
            if str(config.shear_noise_attr) != expected_noise_attr:
                raise ValueError(
                    f"DES shear mask {config.shear_mask_dataset!r} requires noise attribute "
                    f"{expected_noise_attr!r}, got {config.shear_noise_attr!r}."
                )
            mask = _clean_mask(group[config.shear_mask_dataset][:])
            gamma1 = shear_sign * _clean_map(group["gamma1"][:])
            gamma2 = shear_sign * _clean_map(group["gamma2_namaster"][:])
            gamma1[mask <= 0] = 0.0
            gamma2[mask <= 0] = 0.0
            noise = float(group.attrs[config.shear_noise_attr])
            metadata = h5_attrs_to_jsonable(group.attrs)
            metadata.update(
                {
                    "source_hdf5": str(path),
                    "mask_dataset": config.shear_mask_dataset,
                    "shape_noise_pseudo_cl": noise,
                    "shape_noise_attr": config.shear_noise_attr,
                    "input_spin_convention": "DES product stores [gamma1, gamma2_namaster] for NaMaster spin=2.",
                    "shear_e_to_kappa_sign": shear_sign,
                    "saved_e_mode_convention": (
                        "Shear maps are multiplied by shear_e_to_kappa_sign before "
                        "NaMaster so scalar x shear-E spectra have the same sign as "
                        "scalar x positive-convergence theory. DES shear EE is unchanged."
                    ),
                }
            )
            fields[f"s{tomo + 1}"] = FieldMap(
                name=f"s{tomo + 1}",
                label=f"DES Y3 shear tomo {tomo + 1}",
                kind="des_shear",
                spin=2,
                maps=[gamma1, gamma2],
                mask=mask,
                mask_name=f"des_shear_tomo{tomo + 1}",
                metadata=metadata,
            )
    return fields


def build_act_fields(bundle: SurveyBundle, config: MeasurementConfig) -> Dict[str, FieldMap]:
    fields: Dict[str, FieldMap] = {}
    y_map, y_mask = enmap_h5_to_healpix(
        bundle.act_y,
        "maps/compton_y",
        "map_wcs_header",
        "masks/footprint_mask",
        "footprint_mask_wcs_header",
        config.nside,
        config.lmax,
        config.act_downgrade,
        config.subtract_masked_mean,
    )
    fields["y"] = FieldMap(
        name="y",
        label="ACT DR6 Compton-y",
        kind="act_tsz_y",
        spin=0,
        maps=[y_map],
        mask=y_mask,
        mask_name="act_y_footprint",
        metadata={
            "source_hdf5": str(bundle.act_y),
            "mask_dataset": "masks/footprint_mask",
            "beam_fwhm_arcmin": ACT_TSZ_BEAM_FWHM_ARCMIN,
            "beam_transfer_dataset": "transfer_functions/act_y_gaussian_beam",
            "effective_transfer_status": "Default theory wrapper applies a 1.6 arcmin Gaussian beam.",
        },
    )

    t_map, t_mask = enmap_h5_to_healpix(
        bundle.act_cmb,
        "maps/cmb_temperature",
        "map_wcs_header",
        "maps/analysis_mask",
        "mask_wcs_header",
        config.nside,
        config.lmax,
        config.act_downgrade,
        config.subtract_masked_mean,
    )
    fields["T"] = FieldMap(
        name="T",
        label="ACT DR6 CMB temperature",
        kind="act_cmb_temperature",
        spin=0,
        maps=[t_map],
        mask=t_mask,
        mask_name="act_cmb_temperature",
        metadata={
            "source_hdf5": str(bundle.act_cmb),
            "mask_dataset": "maps/analysis_mask",
            "units": (
                ACT_CMB_TEMPERATURE_UNITS
                if config.act_cmb_temperature_units_confirmed
                else "uK_CMB_likely"
            ),
            "units_confirmed": bool(config.act_cmb_temperature_units_confirmed),
            "units_confirmation_source": (
                "user-supplied source-map provenance"
                if config.act_cmb_temperature_units_confirmed
                else "unconfirmed"
            ),
            "beam_fwhm_arcmin": ACT_CMB_TEMPERATURE_BEAM_FWHM_ARCMIN,
            "beam_transfer_dataset": "transfer_functions/act_cmb_temperature_gaussian_beam",
            "effective_transfer_status": "Default theory wrapper applies a 1.6 arcmin Gaussian beam.",
            "beam_reference": KSZ_REFERENCE_PAPER,
        },
    )

    k_map, k_mask = enmap_h5_to_healpix(
        bundle.act_kappa,
        "maps/kappa",
        "map_wcs_header",
        "masks/lensing_mask_apodized",
        "map_wcs_header",
        config.nside,
        config.lmax,
        max(1, config.act_downgrade // 2),
        config.subtract_masked_mean,
        map_is_masked=True,
    )
    fields["kappa"] = FieldMap(
        name="kappa",
        label="ACT DR6 CMB lensing kappa",
        kind="act_cmb_lensing_kappa",
        spin=0,
        maps=[k_map],
        mask=k_mask,
        mask_name="act_cmb_lensing",
        metadata={
            "source_hdf5": str(bundle.act_kappa),
            "mask_dataset": "masks/lensing_mask_apodized",
            "map_note": "Transferred map is the ACTxDESI baseline masked CAR kappa product.",
            "analysis_lmax_inclusive": (
                None if config.kappa_cmb_lmax is None else int(config.kappa_cmb_lmax)
            ),
            "transfer_null_from_ell": (
                None if config.kappa_cmb_lmax is None else int(config.kappa_cmb_lmax) + 1
            ),
            "namaster_masked_on_input": True,
            "masked_on_input_caveat": (
                "The available ACT kappa CAR map already contains the upstream lensing mask. "
                "NaMaster must not multiply it by the mask a second time. The projected mask "
                "is used for mode coupling; an unmasked kappa source would be preferable."
            ),
        },
    )
    return fields


def build_probe_maps(bundle: SurveyBundle, config: MeasurementConfig) -> Tuple[Dict[str, FieldMap], Dict[str, object]]:
    bundle_meta = bundle.validate_files()
    des_source_nz = load_des_y3_source_nz(config.des_y3_source_nz_fits)
    des_source_nz_path = Path(config.des_y3_source_nz_fits)
    stat = des_source_nz_path.stat()
    bundle_meta["des_y3_source_nz"] = {
        "path": str(des_source_nz_path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "hdu": DES_Y3_SOURCE_NZ_HDU,
    }
    fields: Dict[str, FieldMap] = {}
    fields.update(build_shear_fields(bundle, config))
    random_counts, random_count_metadata = load_dr9_random_counts_with_metadata(bundle, config)
    desi_fields, desi_summary = build_desi_fields(bundle, config, random_counts, random_count_metadata)
    fields.update(desi_fields)
    fields.update(build_act_fields(bundle, config))
    fields = apply_mask_apodization(fields, config)

    metadata = {
        "schema": SCHEMA_MAPS,
        "created_utc": utc_now(),
        "pipeline_version": MEASUREMENT_PIPELINE_VERSION,
        "map_construction_version": MAP_CONSTRUCTION_VERSION,
        "spectrum_estimator_version": SPECTRUM_ESTIMATOR_VERSION,
        "covariance_estimator_version": COVARIANCE_ESTIMATOR_VERSION,
        "bundle_root": str(bundle.root),
        "input_files": bundle_meta,
        "config": config.to_dict(),
        "desi_summary": desi_summary,
        "des_y3_source_nz": des_source_nz,
        "missing_inputs": missing_inputs_metadata(
            config=config,
            random_realization_count=int(
                desi_summary.get("random_count_metadata", {}).get("random_realization_count", 1)
            ),
        ),
    }
    return fields, map_metadata_with_content_identity(fields, metadata)


def missing_inputs_metadata(
    *,
    config: Optional[MeasurementConfig] = None,
    random_realization_count: int = 1,
) -> Dict[str, object]:
    temperature_units_confirmed = bool(
        config is not None and config.act_cmb_temperature_units_confirmed
    )
    required_randoms = int(
        1 if config is None else config.minimum_desi_random_realizations
    )
    randoms_present = int(random_realization_count) >= required_randoms
    return {
        "des_y3_source_nz": {
            "present": True,
            "needed_for": "DES shear theory and final MCMC.",
            "source_fits": DES_Y3_SOURCE_NZ_FITS_DEFAULT,
            "hdu": DES_Y3_SOURCE_NZ_HDU,
            "note": "DES Y3 source-bin n(z) curves are loaded from the 2pt FITS and saved as normalized dN/dz.",
        },
        "des_y3_shear_photoz_priors": {
            "present": True,
            "needed_for": "DES shear theory nuisance parameters and final MCMC.",
            "prior_gaussian": des_y3_gaussian_priors(),
        },
        "act_y_effective_transfer": {
            "present": True,
            "needed_for": "Exact y/shear and g/y theory comparison.",
            "note": "Use a 1.6 arcmin Gaussian beam by default. Supply an additional transfer only if the y map has filtering beyond this beam.",
        },
        "act_cmb_temperature_effective_transfer": {
            "present": True,
            "needed_for": "Exact kSZ theory comparison.",
            "note": "The kSZ reference paper uses ACT DR6 hILC temperature maps with effective FWHM 1.6 arcmin. Supply an additional transfer only if applying extra temperature filtering.",
        },
        "act_cmb_temperature_unit_validation": {
            "present": temperature_units_confirmed,
            "needed_for": "Absolute kSZ amplitude calibration.",
            "units": ACT_CMB_TEMPERATURE_UNITS if temperature_units_confirmed else "uK_CMB_likely",
            "confirmation_source": "user-supplied source-map provenance" if temperature_units_confirmed else "unconfirmed",
            "fallback": (
                "Confirmed for this product."
                if temperature_units_confirmed
                else "Verify the transferred source-map units before quoting a physical kSZ amplitude."
            ),
        },
        "ksz_velocity_calibration": {
            "present": True,
            "needed_for": "Convert C_ell^g,tau into C_ell^pi,T without fitting free A_v per bin.",
            "available": (
                "Theory wrapper uses r=0.3 from the photometric DESI velocity-reconstruction reference, "
                "sigma_rec from each saved DESI bin, and Abacus sigma_true_gas values from "
                f"{KSZ_SIGMA_TRUE_GAS_DOC}."
            ),
            "sigma_true_gas_over_c": KSZ_SIGMA_TRUE_GAS_OVER_C_3E5,
            "caveat": "The r calibration is approximate and may vary with bin/sample; A_v_bin can still be fit directly.",
            "fallback": "Fit A_v_bin as a free amplitude.",
        },
        "additional_desi_randoms_for_nside4096": {
            "present": randoms_present,
            "needed_for": "Production raw nside=4096 DESI high-ell mask stability.",
            "random_realization_count": int(random_realization_count),
            "minimum_required_by_stage": required_randoms,
            "fallback": (
                "Requirement satisfied; retain mask-convergence diagnostics in product provenance."
                if randoms_present
                else "Use the provided DR9 one-random count map as provisional, or rerun after adding more DR9 random realizations."
            ),
        },
        "non_gaussian_and_mock_covariance_validation": {
            "present": False,
            "needed_for": "Final full-data-vector errors and detection significance.",
            "current_model": "Gaussian improved narrow-kernel approximation (iNKA)",
            "omitted": [
                "connected non-Gaussian covariance",
                "super-sample covariance",
                "foreground covariance",
                "finite-random/catalog-selection uncertainty",
                "mock calibration",
            ],
        },
        "desi_dr9_imaging_weights": {
            "present": True,
            "needed_for": "DESI galaxy auto/cross spectra and weighted kSZ velocity-momentum templates.",
            "selection_dataset": DESI_DR9_SELECTION_DATASET,
            "weight_dataset": DESI_DR9_WEIGHT_DATASET,
        },
        "desi_dr9_calibrated_true_nz": {
            "present": True,
            "needed_for": "DESI photometric-bin theory kernels and HOD abundance targets.",
            "source_hdf5_relpath": DESI_DR9_TRUE_NZ_HDF5_REL,
            "full_cl_group": DESI_DR9_TRUE_NZ_GROUP_FULL_CL,
            "normalized_kernel_dataset": DESI_DR9_TRUE_NZ_DATASET,
            "note": "Catalog Z_PHOT_MEDIAN histograms are diagnostic only and are not valid theory n(z) kernels.",
        },
    }


def _write_dataset(group: h5py.Group, name: str, data: np.ndarray, dtype: Optional[str] = None) -> h5py.Dataset:
    arr = np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)
    if arr.dtype.kind == "O":
        return group.create_dataset(name, data=arr)
    return group.create_dataset(
        name,
        data=arr,
        chunks=True,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def _write_des_shear_nz_group(parent: h5py.Group, des_nz: Optional[Mapping[str, object]]) -> None:
    sg = parent.create_group("des_shear")
    if not isinstance(des_nz, Mapping):
        sg.attrs["missing"] = True
        sg.attrs["note"] = "DES Y3 source n(z) was not present in this product metadata."
        return
    sg.attrs["missing"] = False
    sg.attrs["source_fits"] = str(des_nz.get("source_fits", ""))
    sg.attrs["hdu"] = str(des_nz.get("hdu", DES_Y3_SOURCE_NZ_HDU))
    sg.attrs["raw_column_note"] = str(des_nz.get("raw_column_note", ""))
    for key in (
        "z_low",
        "z_mid",
        "z_high",
        "raw_bin_values_by_bin",
        "dndz_by_bin",
        "raw_trapz_norm_by_bin",
        "raw_width_norm_by_bin",
        "mean_z_by_bin",
        "sigma_e_by_bin",
        "ngal_arcmin2_by_bin",
    ):
        if key in des_nz:
            _write_dataset(sg, key, np.asarray(des_nz[key]), dtype="f8")
    _write_dataset(sg, "bin_names", _string_array([f"source_bin{i}" for i in range(1, 5)]))


def _write_des_y3_priors_group(parent: h5py.Group) -> None:
    pg = parent.create_group("des_y3_gaussian")
    priors = des_y3_gaussian_priors()
    names = list(priors)
    _write_dataset(pg, "name", _string_array(names))
    _write_dataset(pg, "mu", np.asarray([priors[name]["mu"] for name in names], dtype=np.float64), dtype="f8")
    _write_dataset(pg, "sigma", np.asarray([priors[name]["sigma"] for name in names], dtype=np.float64), dtype="f8")
    pg.attrs["prior_gaussian_json"] = _json_dumps(priors)
    pg.attrs["note"] = "DES Y3 photo-z Delta_z and multiplicative shear-bias Gaussian priors."


def save_map_product(
    path: str | Path,
    fields: Mapping[str, FieldMap],
    metadata: Mapping[str, object],
    overwrite: bool = False,
) -> Path:
    path = Path(path)
    product_id = validate_map_metadata_identity(metadata)
    validate_loaded_map_content(fields, metadata)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    masks: Dict[str, np.ndarray] = {}
    for field_map in fields.values():
        if field_map.mask_name not in masks:
            masks[field_map.mask_name] = field_map.mask

    with h5py.File(tmp, "w", track_order=True) as h5:
        h5.attrs["schema"] = SCHEMA_MAPS
        h5.attrs["created_utc"] = utc_now()
        h5.attrs["pipeline_version"] = str(metadata.get("pipeline_version", MEASUREMENT_PIPELINE_VERSION))
        h5.attrs["map_construction_version"] = str(
            metadata.get("map_construction_version", MAP_CONSTRUCTION_VERSION)
        )
        h5.attrs["map_product_id"] = product_id
        h5.attrs["metadata_json"] = _json_dumps(metadata)
        cfg = metadata.get("config", {})
        if isinstance(cfg, Mapping):
            h5.attrs["binning"] = str(cfg.get("binning", "sqrt"))
            h5.attrs["ell_max_inclusive"] = int(cfg.get("lmax", 0))
            try:
                cfg_obj = MeasurementConfig(**{**asdict(MeasurementConfig()), **dict(cfg)})
                left, right = make_bandpower_edges(cfg_obj)
                _write_dataset(h5, "ell_left", left, dtype="i4")
                _write_dataset(h5, "ell_right", right, dtype="i4")
            except Exception as exc:  # pragma: no cover - metadata best effort
                h5.attrs["bandpower_edge_write_error"] = str(exc)

        mg = h5.create_group("masks")
        for name, mask in masks.items():
            ds = _write_dataset(mg, name, mask, dtype="f4")
            ds.attrs["description"] = f"Shared mask {name}"

        fg = h5.create_group("fields")
        for name, field_map in fields.items():
            g = fg.create_group(name)
            g.attrs["name"] = field_map.name
            g.attrs["label"] = field_map.label
            g.attrs["kind"] = field_map.kind
            g.attrs["spin"] = int(field_map.spin)
            g.attrs["mask_ref"] = field_map.mask_name
            g.attrs["metadata_json"] = _json_dumps(field_map.metadata)
            for i, map_i in enumerate(field_map.maps):
                _write_dataset(g, f"map{i}", map_i, dtype="f4")
            if field_map.catalog:
                cg = g.create_group("catalog")
                cg.attrs["description"] = (
                    "Catalog inputs used to construct NaMaster NmtFieldCatalogMomentum for kSZ spectra. "
                    "The pixelized map datasets in this field are diagnostics only."
                )
                cg.attrs["lonlat"] = bool(field_map.metadata.get("catalog_lonlat", True))
                cg.attrs["field_is_weighted"] = bool(field_map.metadata.get("catalog_field_is_weighted", False))
                for key in ("ra_deg", "dec_deg", "weight", "field"):
                    if key in field_map.catalog:
                        _write_dataset(cg, key, np.asarray(field_map.catalog[key]), dtype="f8")

        nzg = h5.create_group("nz")
        desi = metadata.get("desi_summary", {})
        if isinstance(desi, Mapping):
            dg = nzg.create_group("desi")
            for key in (
                "z_edges",
                "z_mid",
                "dz",
                "z_min",
                "z_max",
                "vr_edges",
                "vr_mid",
                "nz_dndz_by_pz",
                "nz_n_per_deg2_bin_by_pz",
                "nz_dndz_per_deg2_by_pz",
                "nz_mean_true_z_by_pz",
                "nz_sigma_true_z_by_pz",
                "nz_surface_density_per_deg2_by_pz",
                "vr_counts_by_pz",
                "vr_weighted_counts_by_pz",
            ):
                if key in desi:
                    _write_dataset(dg, key, np.asarray(desi[key]))
            for key in (
                "redshift_kind",
                "theory_nz_source_hdf5",
                "theory_nz_group",
                "theory_nz_sample",
                "theory_nz_dataset",
                "theory_nz_note",
                "selection_dataset",
                "weight_dataset",
            ):
                if key in desi:
                    dg.attrs[key] = str(desi[key])
            dg.attrs["summary_json"] = _json_dumps(desi.get("bins", {}))
            if "ksz_sigma_true_gas_calibration" in desi:
                dg.attrs["ksz_sigma_true_gas_calibration_json"] = _json_dumps(desi["ksz_sigma_true_gas_calibration"])
            photoz = desi.get("photoz_diagnostic", {})
            if isinstance(photoz, Mapping):
                pgz = nzg.create_group("desi_photoz_diagnostic")
                for key, value in photoz.items():
                    if key in {"description", "redshift_kind"}:
                        pgz.attrs[key] = str(value)
                    else:
                        _write_dataset(pgz, key, np.asarray(value))
        _write_des_shear_nz_group(nzg, metadata.get("des_y3_source_nz"))

        pg = h5.create_group("priors")
        _write_des_y3_priors_group(pg)

        tf = h5.create_group("transfer_functions")
        cfg = metadata.get("config", {})
        lmax = int(cfg.get("lmax", 0)) if isinstance(cfg, Mapping) else 0
        nside = int(cfg.get("nside", 0)) if isinstance(cfg, Mapping) else 0
        if nside > 0 and lmax > 0:
            pix_t, pix_p = hp.pixwin(nside, lmax=lmax, pol=True)
            _write_dataset(tf, "healpix_temperature_pixwin", pix_t, dtype="f8")
            _write_dataset(tf, "healpix_polarization_pixwin", pix_p, dtype="f8")
        if lmax >= 0:
            _write_act_beam_transfers(tf, lmax)
        input_files = metadata.get("input_files", {})
        kappa_path = None
        if isinstance(input_files, Mapping) and "act_kappa" in input_files:
            kappa_path = input_files["act_kappa"].get("path")
        if kappa_path:
            try:
                with h5py.File(kappa_path, "r") as kh5:
                    _write_dataset(tf, "act_kappa_filter_baseline", kh5["curves/kappa_filter_baseline"][:], dtype="f8")
                    _write_dataset(tf, "act_kappa_noise_N_L_baseline", kh5["curves/N_L_kk_baseline"][:], dtype="f8")
            except Exception as exc:  # pragma: no cover - metadata best effort
                tf.attrs["act_kappa_curve_read_error"] = str(exc)

    os.replace(tmp, path)
    return path


def load_map_product(
    path: str | Path,
    field_names: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, FieldMap], Dict[str, object]]:
    """Load cached probe maps. If ``field_names`` is given, only those fields (and the
    masks they reference) are read from disk -- this keeps covariance workers from holding
    all ~15 probe maps in memory when a group only needs a few, so many more single-threaded
    groups can be packed onto one node."""

    path = Path(path)
    want = None if field_names is None else {str(n) for n in field_names}
    with h5py.File(path, "r") as h5:
        validate_map_product_hdf_identity(h5)
        metadata = json.loads(h5.attrs["metadata_json"])
        selected = [name for name in h5["fields"] if want is None or name in want]
        if want is not None:
            missing = sorted(want - set(selected))
            if missing:
                raise KeyError(f"{path} is missing requested field(s): {missing}")
        needed_masks = {str(h5[f"fields/{name}"].attrs["mask_ref"]) for name in selected}
        masks = {name: h5[f"masks/{name}"][:] for name in h5["masks"] if want is None or name in needed_masks}
        fields: Dict[str, FieldMap] = {}
        for name in selected:
            g = h5[f"fields/{name}"]
            maps = [g[f"map{i}"][:] for i in range(len([k for k in g if k.startswith("map")]))]
            mask_name = str(g.attrs["mask_ref"])
            catalog: Dict[str, np.ndarray] = {}
            if "catalog" in g:
                cg = g["catalog"]
                catalog = {key: np.asarray(cg[key][:], dtype=np.float64) for key in cg}
            fields[name] = FieldMap(
                name=str(g.attrs["name"]),
                label=str(g.attrs["label"]),
                kind=str(g.attrs["kind"]),
                spin=int(g.attrs["spin"]),
                maps=maps,
                mask=masks[mask_name],
                mask_name=mask_name,
                metadata=json.loads(g.attrs["metadata_json"]),
                catalog=catalog,
            )
    validate_loaded_map_content(fields, metadata)
    return fields, metadata


def build_nmt_fields(fields: Mapping[str, FieldMap], config: MeasurementConfig) -> Dict[str, NmtProbeField]:
    out: Dict[str, NmtProbeField] = {}
    for name, field_map in fields.items():
        validate_field_map_for_namaster(field_map)
        if field_map.has_catalog_momentum:
            catalog = field_map.catalog
            cat_field = nmt.NmtFieldCatalogMomentum(
                np.asarray([catalog["ra_deg"], catalog["dec_deg"]], dtype=np.float64),
                np.asarray(catalog["weight"], dtype=np.float64),
                np.asarray(catalog["field"], dtype=np.float64),
                None,
                None,
                lmax=config.lmax,
                lmax_mask=config.effective_lmax_mask,
                spin=field_map.spin,
                field_is_weighted=bool(field_map.metadata.get("catalog_field_is_weighted", False)),
                lonlat=bool(field_map.metadata.get("catalog_lonlat", True)),
                mask=np.asarray(field_map.mask, dtype=np.float64),
                n_iter_mask=config.n_iter_mask,
            )
            cov_mask_field = nmt.NmtField(
                field_map.mask,
                None,
                spin=0,
                n_iter=config.n_iter,
                n_iter_mask=config.n_iter_mask,
                lmax=config.lmax,
                lmax_mask=config.effective_lmax_mask,
                lite=True,
            )
            out[name] = NmtProbeField(info=field_map, field=cat_field, covariance_field=cov_mask_field)
            continue
        out[name] = NmtProbeField(
            info=field_map,
            field=nmt.NmtField(
                field_map.mask,
                field_map.maps,
                spin=field_map.spin,
                purify_e=False,
                purify_b=False,
                n_iter=config.n_iter,
                n_iter_mask=config.n_iter_mask,
                lmax=config.lmax,
                lmax_mask=config.effective_lmax_mask,
                lite=True,
                masked_on_input=bool(field_map.metadata.get("namaster_masked_on_input", False)),
            ),
        )
    return out


def _new_map_nmt_field(info: FieldMap, maps: Sequence[np.ndarray], config: MeasurementConfig) -> nmt.NmtField:
    return nmt.NmtField(
        info.mask,
        list(maps),
        spin=info.spin,
        purify_e=False,
        purify_b=False,
        n_iter=config.n_iter,
        n_iter_mask=config.n_iter_mask,
        lmax=config.lmax,
        lmax_mask=config.effective_lmax_mask,
        lite=True,
        masked_on_input=bool(info.metadata.get("namaster_masked_on_input", False)),
    )


def _pair_overlap_mean_subtract_enabled(
    a: str,
    b: str,
    fields: Mapping[str, NmtProbeField],
    config: MeasurementConfig,
) -> bool:
    if not bool(config.pair_overlap_mean_subtract):
        return False
    return not (fields[a].is_catalog_momentum or fields[b].is_catalog_momentum)


def _pair_overlap_demeaned_probe_field(
    name: str,
    partner: str,
    fields: Mapping[str, NmtProbeField],
    config: MeasurementConfig,
) -> Tuple[NmtProbeField, Dict[str, object]]:
    base = fields[name]
    partner_field = fields[partner]
    overlap_mask = np.asarray(base.mask, dtype=np.float64) * np.asarray(partner_field.mask, dtype=np.float64)
    overlap_sum = float(np.sum(overlap_mask, dtype=np.float64))
    if not np.isfinite(overlap_sum) or overlap_sum <= 0.0:
        raise ValueError(f"Fields {name!r} and {partner!r} have zero overlap for pair-overlap mean subtraction.")

    positive_self = np.asarray(base.mask) > 0
    maps = []
    means = []
    for values in base.info.maps:
        out = np.asarray(values, dtype=np.float32).copy()
        mean = _weighted_mean_on_mask(out, overlap_mask)
        out[positive_self] -= mean
        out[~positive_self] = 0.0
        maps.append(out)
        means.append(mean)
    meta = {
        "field": name,
        "partner": partner,
        "overlap_mask_sum": overlap_sum,
        "overlap_fsky_weighted": float(np.mean(overlap_mask)),
        "component_means_subtracted": means,
    }
    return NmtProbeField(info=base.info, field=_new_map_nmt_field(base.info, maps, config)), meta


def get_pair_probe_fields(
    a: str,
    b: str,
    fields: Mapping[str, NmtProbeField],
    config: MeasurementConfig,
) -> Tuple[NmtProbeField, NmtProbeField, Dict[str, object]]:
    if not _pair_overlap_mean_subtract_enabled(a, b, fields, config):
        return fields[a], fields[b], {
            "enabled": False,
            "reason": (
                "disabled"
                if not bool(config.pair_overlap_mean_subtract)
                else "catalog_momentum_pair"
            ),
        }
    fa, meta_a = _pair_overlap_demeaned_probe_field(a, b, fields, config)
    fb, meta_b = _pair_overlap_demeaned_probe_field(b, a, fields, config)
    return fa, fb, {"enabled": True, "fields": [meta_a, meta_b]}


def _mean_mask_product(a: NmtProbeField, b: NmtProbeField) -> float:
    mean = float(np.mean(a.mask * b.mask))
    if mean <= 0:
        raise ValueError(f"Fields {a.info.name} and {b.info.name} have zero mask overlap.")
    return mean


def _workspace_key(a: str, b: str, fields: Mapping[str, NmtProbeField]) -> Tuple:
    """Cache key for a power-spectrum workspace.

    The NaMaster mode-coupling matrix and bandpower windows depend only on the two masks,
    the two spins, and the binning -- NOT on the field values (verified: identical coupling
    matrices/windows for two fields sharing a mask). Keying by (mask_name, spin) therefore
    lets field pairs that share masks reuse one workspace -- e.g. all g_i x y, all g_i x s_j,
    the galaxy autos -- collapsing ~46 single-threaded builds to ~22 unique mask/spin pairs.
    Catalog-momentum (kSZ pi) fields are keyed per field pair (conservative: their catalog
    mask handling is not deduplicated here).
    """

    fa = fields[a]
    fb = fields[b]
    if fa.is_catalog_momentum or fb.is_catalog_momentum:
        return ("byfield", a, b)
    return ("bymask", fa.info.mask_name, int(fa.spin), fb.info.mask_name, int(fb.spin))


def get_workspace(
    a: str,
    b: str,
    fields: Mapping[str, NmtProbeField],
    bins: nmt.NmtBin,
    cache: MutableMapping[Tuple, nmt.NmtWorkspace],
    config: MeasurementConfig,
) -> nmt.NmtWorkspace:
    key = _workspace_key(a, b, fields)
    if key not in cache:
        fa, fb, _ = get_pair_probe_fields(a, b, fields, config)
        cache[key] = nmt.NmtWorkspace.from_fields(
            fa.field,
            fb.field,
            bins,
            l_toeplitz=config.covariance_l_toeplitz,
            l_exact=config.covariance_l_exact,
            dl_band=config.covariance_dl_band,
        )
    return cache[key]


def _cov_workspace_key(spec_a: SpectrumSpec, spec_b: SpectrumSpec) -> Tuple[str, str, str, str]:
    return (*spec_a.fields, *spec_b.fields)


def _covariance_workspace_from_fields(
    f_a1: nmt.NmtField,
    f_a2: nmt.NmtField,
    f_b1: nmt.NmtField,
    f_b2: nmt.NmtField,
    config: MeasurementConfig,
) -> nmt.NmtCovarianceWorkspace:
    """Construct a covariance workspace across NaMaster API versions."""

    kwargs = {
        "l_toeplitz": config.covariance_l_toeplitz,
        "l_exact": config.covariance_l_exact,
        "dl_band": config.covariance_dl_band,
    }
    try:
        return nmt.NmtCovarianceWorkspace.from_fields(
            f_a1,
            f_a2,
            f_b1,
            f_b2,
            all_spins=True,
            **kwargs,
        )
    except TypeError:
        # NaMaster <= 2.3 used spin0_only instead of all_spins.
        return nmt.NmtCovarianceWorkspace.from_fields(
            f_a1,
            f_a2,
            f_b1,
            f_b2,
            spin0_only=False,
            **kwargs,
        )


def get_covariance_workspace(
    spec_a: SpectrumSpec,
    spec_b: SpectrumSpec,
    fields: Mapping[str, NmtProbeField],
    cache: MutableMapping[Tuple[str, str, str, str], nmt.NmtCovarianceWorkspace],
    config: MeasurementConfig,
) -> nmt.NmtCovarianceWorkspace:
    key = _cov_workspace_key(spec_a, spec_b)
    a1, a2 = spec_a.fields
    b1, b2 = spec_b.fields
    fa1, fa2, _ = get_pair_probe_fields(a1, a2, fields, config)
    fb1, fb2, _ = get_pair_probe_fields(b1, b2, fields, config)
    max_cache = int(config.covariance_workspace_cache_size)
    if max_cache == 0:
        return _covariance_workspace_from_fields(
            fa1.cov_field,
            fa2.cov_field,
            fb1.cov_field,
            fb2.cov_field,
            config,
        )
    if key not in cache:
        cache[key] = _covariance_workspace_from_fields(
            fa1.cov_field,
            fa2.cov_field,
            fb1.cov_field,
            fb2.cov_field,
            config,
        )
        if max_cache > 0:
            while len(cache) > max_cache:
                cache.pop(next(iter(cache)))
    return cache[key]


def compute_input_cl_for_covariance(
    a: str,
    b: str,
    fields: Mapping[str, NmtProbeField],
    bins: nmt.NmtBin,
    workspace_cache: MutableMapping[Tuple[str, str], nmt.NmtWorkspace],
    cache: MutableMapping[Tuple[str, ...], np.ndarray],
    config: MeasurementConfig,
    *,
    force_pseudo_over_fsky: bool = False,
) -> np.ndarray:
    """Return full-ell total spectra for NaMaster decoupled covariance.

    ``nmt.gaussian_covariance(..., coupled=False)`` expects theory-like
    full-ell spectra.  The measurement files do not yet carry a cosmological
    covariance model, so we build local data-derived total spectra.

    The default ``inka_data`` mode follows NaMaster's improved narrow-kernel
    approximation: ordinary map pairs use :func:`nmt.get_iNKA_cell`, i.e. the
    measured coupled pseudo-spectrum divided by the mean mask product.  Inputs
    involving catalog-momentum fields use the equivalent explicit pseudo-Cl /<
    mask-product> construction and add back the catalog zero-lag ``Nf`` term
    for momentum autos.  NaMaster subtracts ``Nf`` from catalog autos in
    ``compute_coupled_cell`` by default, but it is a real covariance noise term.

    The legacy decouple/smooth/unbin construction remains available only for
    reproducing old products; it is no longer the default covariance model.
    """

    input_mode = str(config.covariance_input_mode)
    supported_modes = {"inka_data", "decoupled_total_bandpowers_unbinned"}
    if input_mode not in supported_modes:
        raise ValueError(
            "Unsupported covariance_input_mode "
            f"{config.covariance_input_mode!r}; expected one of {sorted(supported_modes)}."
        )
    use_pseudo_over_fsky = bool(force_pseudo_over_fsky) or fields[a].is_catalog_momentum or fields[b].is_catalog_momentum
    if use_pseudo_over_fsky:
        mode = "pseudo_over_fsky"
    elif input_mode == "inka_data":
        mode = "inka_data"
    else:
        mode = "decoupled_total"
    key = (mode, a, b)
    if key not in cache:
        if use_pseudo_over_fsky:
            cache[key] = compute_pseudo_over_fsky_input_cl_for_covariance(a, b, fields, config)
            return cache[key]
        fa, fb, _ = get_pair_probe_fields(a, b, fields, config)
        if input_mode == "inka_data":
            inka_cl = nmt.get_iNKA_cell(fa.field, fb.field)
            inka_cl = _pad_or_trim_full_ell(np.asarray(inka_cl, dtype=np.float64), config.lmax)
            if not np.all(np.isfinite(inka_cl)):
                raise ValueError(f"Non-finite iNKA covariance input for fields {a!r} x {b!r}.")
            cache[key] = inka_cl
            return cache[key]
        workspace = get_workspace(a, b, fields, bins, workspace_cache, config)
        pcl = nmt.compute_coupled_cell(fa.field, fb.field)
        noise_coupled, noise_decoupled = coupled_noise_for_field_pair(a, b, fields, workspace, config, pcl=pcl)
        signal_bpw = workspace.decouple_cell(pcl, cl_noise=noise_coupled)
        total_bpw = np.asarray(signal_bpw, dtype=np.float64)
        if noise_decoupled is not None:
            total_bpw = total_bpw + np.asarray(noise_decoupled, dtype=np.float64)
        total_bpw = prepare_total_bandpowers_for_covariance(a, b, fields, total_bpw, config, noise_decoupled)
        cache[key] = unbin_covariance_bandpowers_to_full_ell(bins, total_bpw, config.lmax)
    return cache[key]


def _pad_or_trim_full_ell(cl: np.ndarray, lmax: int) -> np.ndarray:
    arr = np.asarray(cl, dtype=np.float64)
    target = int(lmax) + 1
    if arr.ndim != 2:
        raise ValueError(f"Expected covariance input Cl array with rank 2, got shape {arr.shape}.")
    if arr.shape[1] == target:
        return arr
    out = np.zeros((arr.shape[0], target), dtype=np.float64)
    n = min(target, arr.shape[1])
    out[:, :n] = arr[:, :n]
    if n > 0 and n < target:
        out[:, n:] = arr[:, n - 1 : n]
    return out


def _add_catalog_auto_zero_lag_noise(pcl: np.ndarray, field: NmtProbeField) -> np.ndarray:
    out = np.asarray(pcl, dtype=np.float64).copy()
    nf = float(getattr(field.field, "Nf", 0.0))
    if nf == 0.0:
        return out
    labels = component_labels(field.spin, field.spin)
    for icomp, label in enumerate(labels):
        if label in {"00", "EE", "BB"}:
            out[icomp, :] += nf
    return out


def compute_pseudo_over_fsky_input_cl_for_covariance(
    a: str,
    b: str,
    fields: Mapping[str, NmtProbeField],
    config: MeasurementConfig,
) -> np.ndarray:
    """Return kSZ-tutorial-style pseudo-Cl/fsky covariance input spectra."""

    fa, fb, _ = get_pair_probe_fields(a, b, fields, config)
    pcl = nmt.compute_coupled_cell(fa.field, fb.field)
    if a == b and fa.is_catalog_momentum:
        pcl = _add_catalog_auto_zero_lag_noise(pcl, fa)
    fsky = float(np.mean(fa.mask * fb.mask))
    if not np.isfinite(fsky) or fsky <= 0.0:
        raise ValueError(f"Fields {a!r} and {b!r} have zero mask overlap for kSZ covariance input.")
    return _pad_or_trim_full_ell(np.asarray(pcl, dtype=np.float64) / fsky, config.lmax)


def compute_catalog_momentum_input_cl_for_covariance(
    a: str,
    b: str,
    fields: Mapping[str, NmtProbeField],
    config: MeasurementConfig,
) -> np.ndarray:
    """Backward-compatible alias for the kSZ tutorial covariance input helper."""

    return compute_pseudo_over_fsky_input_cl_for_covariance(a, b, fields, config)


def coupled_noise_for_field_pair(
    a: str,
    b: str,
    fields: Mapping[str, NmtProbeField],
    workspace: nmt.NmtWorkspace,
    config: MeasurementConfig,
    pcl: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    fa = fields[a].info
    fb = fields[b].info
    if a != b:
        return None, None

    labels = component_labels(fa.spin, fb.spin)
    ncls = len(labels)
    if fa.kind == "des_shear":
        if bool(fa.metadata.get("mask_apodization_applied", False)):
            raise ValueError(
                f"DES shear field {a!r} is apodized, but its saved analytic shape-noise "
                "pseudo-Cl is defined for the original weighted-count mask."
            )
        n_white = float(fa.metadata["shape_noise_pseudo_cl"])
        if not np.isfinite(n_white) or n_white < 0.0:
            raise ValueError(f"Invalid catalog-derived DES shear shape noise for field {a!r}: {n_white}")
        full = np.zeros((ncls, config.lmax + 1), dtype=np.float64)
        full[0, :] = n_white
        if ncls == 4:
            full[3, :] = n_white
        return full, workspace.decouple_cell(full)

    if fa.kind == "desi_galaxy":
        shot_pseudo = float(fa.metadata["shot_noise_pseudo_cl"])
        if not np.isfinite(shot_pseudo) or shot_pseudo < 0.0:
            raise ValueError(f"Invalid DESI coupled shot-noise pseudo-Cl for field {a!r}: {shot_pseudo}")
        coupled = np.full((1, config.lmax + 1), shot_pseudo, dtype=np.float64)
        return coupled, workspace.decouple_cell(coupled)

    return None, None


def coupled_noise_for_spectrum(
    spec: SpectrumSpec,
    fields: Mapping[str, NmtProbeField],
    workspace: nmt.NmtWorkspace,
    config: MeasurementConfig,
    pcl: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    return coupled_noise_for_field_pair(
        spec.fields[0], spec.fields[1], fields, workspace, config, pcl=pcl
    )


def _fill_nonfinite_bandpowers(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if arr.ndim != 1:
        raise ValueError("_fill_nonfinite_bandpowers expects one component at a time.")
    good = np.isfinite(arr)
    if np.all(good):
        return arr
    if not np.any(good):
        return np.zeros_like(arr)
    x = np.arange(arr.size, dtype=np.float64)
    arr[~good] = np.interp(x[~good], x[good], arr[good])
    return arr


def _smooth_bandpower_component(values: np.ndarray, window: int) -> np.ndarray:
    arr = _fill_nonfinite_bandpowers(values)
    window = int(window)
    if window <= 1 or arr.size < 3:
        return arr
    if window % 2 == 0:
        window += 1
    window = min(window, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if window <= 1:
        return arr
    pad = window // 2
    padded = np.pad(arr, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _smooth_positive_bandpowers(values: np.ndarray, floor: float = 1.0e-20) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=np.float64), float(floor))
    if gaussian_filter1d is None or arr.size < 5:
        return arr
    return np.exp(gaussian_filter1d(np.log(arr), sigma=1.0, mode="nearest"))


def _is_auto_total_component(field_a: str, field_b: str, label: str) -> bool:
    return field_a == field_b and label in {"00", "EE", "BB"}


def _component_is_theory_null_for_covariance(field_a: str, field_b: str, label: str) -> bool:
    if "B" not in label:
        return False
    if label in {"EB", "BE", "0B", "B0"}:
        return True
    if label == "BB" and field_a != field_b:
        return True
    return False


def prepare_total_bandpowers_for_covariance(
    a: str,
    b: str,
    fields: Mapping[str, NmtProbeField],
    total_bpw: np.ndarray,
    config: MeasurementConfig,
    noise_decoupled: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Make decoupled total bandpowers usable as covariance input spectra."""

    fa = fields[a]
    fb = fields[b]
    labels = component_labels(fa.spin, fb.spin)
    arr = np.asarray(total_bpw, dtype=np.float64).copy()
    if arr.shape[0] != len(labels):
        raise ValueError(f"{a} x {b} total bandpowers have {arr.shape[0]} components, expected {len(labels)}.")
    if arr.ndim != 2:
        raise ValueError(f"{a} x {b} total bandpowers must be 2D, got shape {arr.shape}.")

    smooth = bool(config.covariance_input_smooth_bandpowers)
    window = int(config.covariance_input_smooth_window)
    noise_bpw = None if noise_decoupled is None else np.asarray(noise_decoupled, dtype=np.float64)
    for icomp, label in enumerate(labels):
        if bool(config.covariance_zero_parity_odd_inputs) and _component_is_theory_null_for_covariance(a, b, label):
            arr[icomp, :] = 0.0
            continue
        if smooth and _is_auto_total_component(a, b, label):
            component = arr[icomp, :]
            if label == "BB" and noise_bpw is not None and noise_bpw.shape == arr.shape:
                component = np.maximum(component, noise_bpw[icomp, :])
            arr[icomp, :] = _smooth_positive_bandpowers(component)
        elif smooth:
            arr[icomp, :] = _smooth_bandpower_component(arr[icomp, :], window)
        else:
            arr[icomp, :] = _fill_nonfinite_bandpowers(arr[icomp, :])
        if _is_auto_total_component(a, b, label):
            finite_pos = arr[icomp, np.isfinite(arr[icomp]) & (arr[icomp] > 0)]
            floor = 1.0e-20 if finite_pos.size == 0 else max(1.0e-20, float(np.nanmin(finite_pos)) * 1.0e-6)
            arr[icomp, :] = np.clip(arr[icomp, :], floor, np.inf)
    return arr


def covariance_input_noise_policy(
    a: str,
    b: str,
    field_metadata: Mapping[str, object],
    zero_parity_odd: bool = True,
    input_mode: str = "inka_data",
) -> str:
    """Describe the total-spectrum convention used for one covariance input pair."""

    meta_a_outer = field_metadata.get(a, {})
    meta_b_outer = field_metadata.get(b, {})
    kind_a = str(meta_a_outer.get("kind", "")) if isinstance(meta_a_outer, Mapping) else ""
    kind_b = str(meta_b_outer.get("kind", "")) if isinstance(meta_b_outer, Mapping) else ""
    is_catalog = kind_a in {"desi_momentum", "desi_momentum_null"} or kind_b in {
        "desi_momentum",
        "desi_momentum_null",
    }
    if str(input_mode) != "inka_data" and not is_catalog:
        suffix = (
            " Parity-odd/B cross components are set to zero before smoothing."
            if zero_parity_odd
            else " All measured components are retained."
        )
        return (
            "Legacy covariance input: measured decoupled bandpowers are converted to total spectra "
            "by restoring same-field noise, sanitizing/smoothing, and unbinning to full ell."
            + suffix
        )
    suffix = " Coupled spin components are retained because mask mixing is part of the iNKA input."
    if a == b and kind_a == "des_shear":
        return (
            "DES shear auto: data-derived NaMaster iNKA total pseudo-spectrum; the map auto "
            "contains the catalog shape noise that was subtracted only from the saved mean bandpowers."
            + suffix
        )
    if a == b and kind_a == "desi_galaxy":
        return (
            "DESI galaxy auto: data-derived NaMaster iNKA total pseudo-spectrum; the map auto "
            "and saved mean bandpowers both contain the weighted Poisson shot noise exactly once."
            + suffix
        )
    if is_catalog:
        if a == b:
            return (
                "DESI kSZ catalog-momentum auto: covariance input follows the NaMaster kSZ tutorial, "
                "using compute_coupled_cell(field, field) plus the catalog zero-lag Nf term, divided "
                "by mean(mask^2)."
                + suffix
            )
        return (
            "DESI kSZ catalog-momentum cross: covariance input follows the NaMaster kSZ tutorial, "
            "using the coupled pseudo-Cl divided by mean(mask_a * mask_b). No cross-noise template "
            "is subtracted."
            + suffix
        )
    if a == b:
        return (
            f"{kind_a or 'field'} auto: no explicit noise template is subtracted in the data vector; "
            "the measured map auto-spectrum is converted with NaMaster's data-derived iNKA rule."
            + suffix
        )
    return (
        f"{kind_a or a} x {kind_b or b} cross: no explicit cross-noise template is subtracted; "
        "the measured coupled cross-spectrum divided by mask overlap is used as the iNKA input."
        + suffix
    )


def unbin_covariance_bandpowers_to_full_ell(bins: nmt.NmtBin, bandpowers: np.ndarray, lmax: int) -> np.ndarray:
    """Expand covariance bandpowers to ``C_ell`` arrays of length ``lmax + 1``."""

    target = int(lmax) + 1
    bpw = np.asarray(bandpowers, dtype=np.float64)
    if bpw.ndim != 2:
        raise ValueError(f"Covariance bandpowers must be 2D, got shape {bpw.shape}.")
    n_bands = int(bins.get_n_bands())
    if bpw.shape[1] != n_bands:
        raise ValueError(f"Covariance bandpowers have {bpw.shape[1]} bands; expected {n_bands}.")
    full = np.zeros((bpw.shape[0], target), dtype=np.float64)
    for ib in range(n_bands):
        lo = max(int(bins.get_ell_min(ib)), 0)
        hi = min(int(bins.get_ell_max(ib)) + 1, target)
        if hi > lo:
            full[:, lo:hi] = bpw[:, ib][:, None]
    if n_bands > 0 and int(bins.get_ell_min(0)) > 0:
        full[:, : int(bins.get_ell_min(0))] = bpw[:, 0][:, None]
    return full


def measure_spectrum(
    spec: SpectrumSpec,
    fields: Mapping[str, NmtProbeField],
    bins: nmt.NmtBin,
    workspace_cache: MutableMapping[Tuple[str, str], nmt.NmtWorkspace],
    config: MeasurementConfig,
) -> Dict[str, object]:
    a, b = spec.fields
    fa = fields[a]
    fb = fields[b]
    labels = component_labels(fa.spin, fb.spin)
    if spec.component >= len(labels):
        raise ValueError(f"{spec.name} asks for component {spec.component}, but {labels} are available.")
    pair_fa, pair_fb, pair_mean_meta = get_pair_probe_fields(a, b, fields, config)
    workspace = get_workspace(a, b, fields, bins, workspace_cache, config)
    pcl = nmt.compute_coupled_cell(pair_fa.field, pair_fb.field)
    noise_coupled, noise_decoupled = coupled_noise_for_spectrum(spec, fields, workspace, config, pcl=pcl)
    # DESI galaxy autos are deliberately saved as raw total bandpowers.  Keep the
    # catalog-derived template in the product for provenance and nuisance modelling,
    # but do not pass it as ``cl_noise`` here: that would turn the saved mean back into
    # a signal-only estimator.  DES shear shape-noise subtraction remains unchanged.
    mean_noise_coupled = noise_coupled
    if a == b and fa.info.kind == "desi_galaxy":
        mean_noise_coupled = None
    cl_all = workspace.decouple_cell(pcl, cl_noise=mean_noise_coupled)
    windows = workspace.get_bandpower_windows()
    selected_window = windows[spec.component, :, spec.component, :]
    result = {
        "name": spec.name,
        "family": spec.family,
        "fields": spec.fields,
        "component": int(spec.component),
        "component_label": labels[spec.component],
        "component_labels": labels,
        "label": spec.label,
        "theory_key": spec.theory_key,
        "metadata": dict(spec.metadata),
        "ell": bins.get_effective_ells(),
        "cl": np.asarray(cl_all[spec.component], dtype=np.float64),
        "cl_all_components": np.asarray(cl_all, dtype=np.float64),
        "pcl_all_components": np.asarray(pcl, dtype=np.float64),
        "noise_decoupled_all_components": None if noise_decoupled is None else np.asarray(noise_decoupled, dtype=np.float64),
        "bandpower_window_selected": np.asarray(selected_window, dtype=np.float64),
        "pair_overlap_mean_subtraction": pair_mean_meta,
    }
    return result


def _select_covariance_component_block(
    cov: np.ndarray,
    n_bands: int,
    ncomp_a: int,
    ncomp_b: int,
    component_a: int,
    component_b: int,
) -> np.ndarray:
    if not (0 <= int(component_a) < int(ncomp_a)):
        raise ValueError(f"component_a={component_a} is outside 0..{int(ncomp_a) - 1}.")
    if not (0 <= int(component_b) < int(ncomp_b)):
        raise ValueError(f"component_b={component_b} is outside 0..{int(ncomp_b) - 1}.")
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 4:
        if cov.shape == (n_bands, ncomp_a, n_bands, ncomp_b):
            return cov[:, component_a, :, component_b]
        if cov.shape == (ncomp_a, n_bands, ncomp_b, n_bands):
            return cov[component_a, :, component_b, :]
        if cov.shape == (n_bands, n_bands, ncomp_a, ncomp_b):
            return cov[:, :, component_a, component_b]
        raise ValueError(
            f"unexpected 4D covariance block shape {cov.shape}; expected one of "
            f"{(n_bands, ncomp_a, n_bands, ncomp_b)}, "
            f"{(ncomp_a, n_bands, ncomp_b, n_bands)}, or "
            f"{(n_bands, n_bands, ncomp_a, ncomp_b)}."
        )
    if cov.ndim != 2:
        raise ValueError(f"unexpected covariance block rank {cov.ndim}")
    if cov.shape == (n_bands, n_bands):
        return cov
    expected = (ncomp_a * n_bands, ncomp_b * n_bands)
    if cov.shape != expected:
        raise ValueError(f"unexpected flattened covariance block shape {cov.shape}; expected {expected}")
    # NaMaster's flattened decoupled covariance is band-major:
    # cov.reshape(n_band, n_comp_a, n_band, n_comp_b).
    rows = np.arange(n_bands) * ncomp_a + component_a
    cols = np.arange(n_bands) * ncomp_b + component_b
    return cov[np.ix_(rows, cols)]


def compute_covariance_block(
    spec_a: SpectrumSpec,
    spec_b: SpectrumSpec,
    fields: Mapping[str, NmtProbeField],
    bins: nmt.NmtBin,
    workspace_cache: MutableMapping[Tuple[str, str], nmt.NmtWorkspace],
    cov_workspace_cache: MutableMapping[Tuple[str, str, str, str], nmt.NmtCovarianceWorkspace],
    input_cl_cache: MutableMapping[Tuple[str, ...], np.ndarray],
    config: MeasurementConfig,
) -> np.ndarray:
    validate_spectrum_spec(spec_a, fields)
    validate_spectrum_spec(spec_b, fields)
    a1, a2 = spec_a.fields
    b1, b2 = spec_b.fields
    wa = get_workspace(a1, a2, fields, bins, workspace_cache, config)
    wb = get_workspace(b1, b2, fields, bins, workspace_cache, config)
    cw = get_covariance_workspace(spec_a, spec_b, fields, cov_workspace_cache, config)
    cov = nmt.gaussian_covariance(
        cw,
        fields[a1].spin,
        fields[a2].spin,
        fields[b1].spin,
        fields[b2].spin,
        compute_input_cl_for_covariance(
            a1,
            b1,
            fields,
            bins,
            workspace_cache,
            input_cl_cache,
            config,
        ),
        compute_input_cl_for_covariance(
            a1,
            b2,
            fields,
            bins,
            workspace_cache,
            input_cl_cache,
            config,
        ),
        compute_input_cl_for_covariance(
            a2,
            b1,
            fields,
            bins,
            workspace_cache,
            input_cl_cache,
            config,
        ),
        compute_input_cl_for_covariance(
            a2,
            b2,
            fields,
            bins,
            workspace_cache,
            input_cl_cache,
            config,
        ),
        wa,
        wb,
        coupled=False,
    )
    try:
        block = _select_covariance_component_block(
            cov,
            bins.get_n_bands(),
            ncls_for_spins(fields[a1].spin, fields[a2].spin),
            ncls_for_spins(fields[b1].spin, fields[b2].spin),
            spec_a.component,
            spec_b.component,
        )
    except ValueError as exc:
        raise ValueError(f"Unexpected covariance block for {spec_a.name} x {spec_b.name}: {exc}") from exc
    if spec_a.name == spec_b.name:
        block = 0.5 * (block + block.T)
    return block


def compute_covariance_block_with_workspace(
    spec_a: SpectrumSpec,
    spec_b: SpectrumSpec,
    fields: Mapping[str, NmtProbeField],
    bins: nmt.NmtBin,
    workspace_cache: MutableMapping[Tuple[str, str], nmt.NmtWorkspace],
    covariance_workspace: nmt.NmtCovarianceWorkspace,
    input_cl_cache: MutableMapping[Tuple[str, ...], np.ndarray],
    config: MeasurementConfig,
) -> np.ndarray:
    """Compute a covariance block using a caller-supplied covariance workspace."""

    validate_spectrum_spec(spec_a, fields)
    validate_spectrum_spec(spec_b, fields)
    a1, a2 = spec_a.fields
    b1, b2 = spec_b.fields
    wa = get_workspace(a1, a2, fields, bins, workspace_cache, config)
    wb = get_workspace(b1, b2, fields, bins, workspace_cache, config)
    cov = nmt.gaussian_covariance(
        covariance_workspace,
        fields[a1].spin,
        fields[a2].spin,
        fields[b1].spin,
        fields[b2].spin,
        compute_input_cl_for_covariance(
            a1,
            b1,
            fields,
            bins,
            workspace_cache,
            input_cl_cache,
            config,
        ),
        compute_input_cl_for_covariance(
            a1,
            b2,
            fields,
            bins,
            workspace_cache,
            input_cl_cache,
            config,
        ),
        compute_input_cl_for_covariance(
            a2,
            b1,
            fields,
            bins,
            workspace_cache,
            input_cl_cache,
            config,
        ),
        compute_input_cl_for_covariance(
            a2,
            b2,
            fields,
            bins,
            workspace_cache,
            input_cl_cache,
            config,
        ),
        wa,
        wb,
        coupled=False,
    )
    block = _select_covariance_component_block(
        cov,
        bins.get_n_bands(),
        ncls_for_spins(fields[a1].spin, fields[a2].spin),
        ncls_for_spins(fields[b1].spin, fields[b2].spin),
        spec_a.component,
        spec_b.component,
    )
    if spec_a.name == spec_b.name:
        block = 0.5 * (block + block.T)
    return block


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    diag = np.clip(np.diag(cov), 0.0, np.inf)
    sigma = np.sqrt(diag)
    denom = sigma[:, None] * sigma[None, :]
    corr = np.zeros_like(cov)
    good = denom > 0
    corr[good] = cov[good] / denom[good]
    return corr


def covariance_diagnostics(cov: np.ndarray, compute_eig: bool = True) -> Dict[str, object]:
    diag = np.diag(cov)
    symmetric_cov = 0.5 * (np.asarray(cov, dtype=np.float64) + np.asarray(cov, dtype=np.float64).T)
    out: Dict[str, object] = {
        "shape": tuple(int(x) for x in cov.shape),
        "finite": bool(np.all(np.isfinite(cov))),
        "symmetric": bool(np.allclose(cov, np.asarray(cov).T, rtol=1.0e-8, atol=1.0e-20)),
        "diag_strictly_positive": bool(np.all(np.isfinite(diag)) and np.all(diag > 0.0)),
        "diag_min": float(np.nanmin(diag)),
        "diag_max": float(np.nanmax(diag)),
    }
    if compute_eig:
        try:
            evals = np.linalg.eigvalsh(symmetric_cov)
            corr = _corr_from_cov(symmetric_cov)
            corr_evals = np.linalg.eigvalsh(0.5 * (corr + corr.T))
            positive = evals[evals > 0]
            eig_scale = max(float(np.max(np.abs(evals))), np.finfo(np.float64).tiny)
            out.update(
                {
                    "eig_min": float(evals[0]),
                    "eig_max": float(evals[-1]),
                    "n_negative_eig": int(np.count_nonzero(evals < 0.0)),
                    "n_negative_eig_relative_1e-12": int(np.count_nonzero(evals < -1.0e-12 * eig_scale)),
                    "condition_positive": float(positive[-1] / positive[0]) if positive.size else np.inf,
                    "corr_eig_min": float(corr_evals[0]),
                    "corr_eig_max": float(corr_evals[-1]),
                    "n_negative_corr_eig": int(np.count_nonzero(corr_evals < 0.0)),
                }
            )
        except np.linalg.LinAlgError as exc:
            out["eig_error"] = str(exc)
    return out


def measure_ksz_nulls(
    fields: Mapping[str, NmtProbeField],
    bins: nmt.NmtBin,
    workspace_cache: MutableMapping[Tuple[str, str], nmt.NmtWorkspace],
    config: MeasurementConfig,
    spectra: Mapping[str, Mapping[str, object]],
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for pz_bin in range(1, 5):
        main_name = f"desi_pi_act_T_pz{pz_bin}"
        if main_name in spectra:
            out[f"signflip_pi_act_T_pz{pz_bin}"] = {
                "ell": np.asarray(spectra[main_name]["ell"]),
                "cl": -np.asarray(spectra[main_name]["cl"]),
                "note": "Exact sign-flip null from pi -> -pi.",
            }
        shuf_name = f"pi_shuf{pz_bin}"
        if shuf_name in fields:
            spec = SpectrumSpec(
                name=f"shuffle_pi_act_T_pz{pz_bin}",
                family="ksz_velocity_shuffle_null",
                fields=(shuf_name, "T"),
                component=0,
                label=f"Shuffled DESI pi pz {pz_bin} x ACT T",
                theory_key="null",
                metadata={"desi_pz": pz_bin},
            )
            out[spec.name] = measure_spectrum(spec, fields, bins, workspace_cache, config)
    return out


def measure_all(
    map_fields: Mapping[str, FieldMap],
    config: MeasurementConfig,
    specs: Optional[List[SpectrumSpec]] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    using_default_specs = specs is None
    specs = default_spectrum_specs() if specs is None else list(specs)
    if using_default_specs and len(specs) != 46:
        raise ValueError(f"Expected 46 target spectra, got {len(specs)}.")
    if not specs:
        raise ValueError("At least one spectrum spec is required.")
    bins = make_bins(config)
    fields = build_nmt_fields(map_fields, config)
    for spec in specs:
        validate_spectrum_spec(spec, fields)
    workspace_cache: Dict[Tuple[str, str], nmt.NmtWorkspace] = {}
    cov_workspace_cache: Dict[Tuple[str, str, str, str], nmt.NmtCovarianceWorkspace] = {}
    input_cl_cache: Dict[Tuple[str, ...], np.ndarray] = {}

    spectra: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        if verbose:
            print(f"[{utc_now()}] Measuring {spec.name}", flush=True)
        spectra[spec.name] = measure_spectrum(spec, fields, bins, workspace_cache, config)

    covariance_blocks: Dict[Tuple[str, str], np.ndarray] = {}
    ell = bins.get_effective_ells()
    n_per = len(ell)
    slices: Dict[str, Tuple[int, int]] = {
        spec.name: (i * n_per, (i + 1) * n_per) for i, spec in enumerate(specs)
    }
    ell_left, ell_right = make_bandpower_edges(config)
    packed = pack_joint_data_vector(specs, spectra, config, ell_left, ell_right)
    joint = {
        "spectrum_names": [spec.name for spec in specs],
        "ell": ell,
        "data_vector": packed["data_vector"],
        "data_vector_raw": packed["data_vector_raw"],
        "data_vector_valid": packed["data_vector_valid"],
        "data_vector_weighted_poisson_subtracted": packed[
            "data_vector_weighted_poisson_subtracted"
        ],
        "data_vector_raw_weighted_poisson_subtracted": packed[
            "data_vector_raw_weighted_poisson_subtracted"
        ],
        "galaxy_auto_weighted_poisson_template": packed[
            "galaxy_auto_weighted_poisson_template"
        ],
        "spectrum_validity": packed["spectrum_validity"],
        "cov": None,
        "corr": None,
        "slices": slices,
        "diagnostics": {"covariance_computed": False},
    }
    if config.compute_covariance:
        n_data = n_per * len(specs)
        cov = np.zeros((n_data, n_data), dtype=np.float64)
        for i, spec_i in enumerate(specs):
            for j, spec_j in enumerate(specs[i:], start=i):
                if verbose:
                    print(f"[{utc_now()}] Covariance {spec_i.name} x {spec_j.name}", flush=True)
                block = compute_covariance_block(
                    spec_i,
                    spec_j,
                    fields,
                    bins,
                    workspace_cache,
                    cov_workspace_cache,
                    input_cl_cache,
                    config,
                )
                covariance_blocks[(spec_i.name, spec_j.name)] = block
                si = slice(*slices[spec_i.name])
                sj = slice(*slices[spec_j.name])
                cov[si, sj] = block
                if i != j:
                    cov[sj, si] = block.T
                if i == j:
                    spectra[spec_i.name]["cov"] = block
                    spectra[spec_i.name]["err"] = np.sqrt(np.clip(np.diag(block), 0.0, np.inf))
        joint["cov"] = cov
        joint["corr"] = _corr_from_cov(cov)
        joint["diagnostics"] = covariance_diagnostics(cov, compute_eig=config.compute_covariance_eigenvalues)
        joint["diagnostics"]["covariance_computed"] = True

    null_tests = measure_ksz_nulls(fields, bins, workspace_cache, config, spectra)
    return {
        "schema": measurement_schema_for_config(config),
        "created_utc": utc_now(),
        "config": config.to_dict(),
        "ell": bins.get_effective_ells(),
        "ell_left": ell_left,
        "ell_right": ell_right,
        "binning": str(config.binning).lower(),
        "ell_max_inclusive": int(config.lmax),
        "spectra": spectra,
        "covariance_blocks": covariance_blocks,
        "joint": joint,
        "null_tests": null_tests,
        "input_cls_for_covariance": input_cl_cache,
        "workspace_keys": list(workspace_cache.keys()),
        "covariance_workspace_keys": list(cov_workspace_cache.keys()),
        "spectrum_specs": [
            {
                "name": spec.name,
                "family": spec.family,
                "fields": list(spec.fields),
                "component": int(spec.component),
                "label": spec.label,
                "theory_key": spec.theory_key,
                "metadata": dict(spec.metadata),
            }
            for spec in specs
        ],
        "field_metadata": {
            name: {
                "label": f.info.label,
                "kind": f.info.kind,
                "spin": f.info.spin,
                "mask_name": f.info.mask_name,
                "metadata": f.info.metadata,
            }
            for name, f in fields.items()
            if not f.info.kind.endswith("_null")
        },
    }


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


def save_measurement_product(
    path: str | Path,
    result: Mapping[str, object],
    map_metadata: Mapping[str, object],
    overwrite: bool = False,
) -> Path:
    path = Path(path)
    map_product_id = validate_map_metadata_identity(map_metadata)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with h5py.File(tmp, "w", track_order=True) as h5:
        schema = str(result.get("schema", measurement_schema_for_config(result["config"])))
        h5.attrs["schema"] = schema
        h5.attrs["created_utc"] = str(result["created_utc"])
        h5.attrs["pipeline_version"] = MEASUREMENT_PIPELINE_VERSION
        h5.attrs["map_construction_version"] = str(
            map_metadata.get("map_construction_version", MAP_CONSTRUCTION_VERSION)
        )
        h5.attrs["spectrum_estimator_version"] = SPECTRUM_ESTIMATOR_VERSION
        h5.attrs["covariance_estimator_version"] = COVARIANCE_ESTIMATOR_VERSION
        h5.attrs["desi_galaxy_auto_mean_convention"] = DESI_GALAXY_AUTO_MEAN_CONVENTION
        joint_for_views = result.get("joint")
        has_galaxy_auto_views = isinstance(joint_for_views, Mapping) and all(
            key in joint_for_views
            for key in (
                "data_vector_weighted_poisson_subtracted",
                "data_vector_raw_weighted_poisson_subtracted",
                "galaxy_auto_weighted_poisson_template",
            )
        )
        if has_galaxy_auto_views:
            h5.attrs["desi_galaxy_auto_views_contract_version"] = (
                DESI_GALAXY_AUTO_VIEWS_CONTRACT_VERSION
            )
        h5.attrs["map_product_id"] = map_product_id
        h5.attrs["config_json"] = _json_dumps(result["config"])
        h5.attrs["map_metadata_json"] = _json_dumps(map_metadata)
        if "sim_measurement_mask_mode" in map_metadata:
            h5.attrs["sim_measurement_mask_mode"] = str(map_metadata["sim_measurement_mask_mode"])
        if "sim_measurement_common_cap_mask" in map_metadata:
            h5.attrs["sim_measurement_common_cap_mask"] = bool(map_metadata["sim_measurement_common_cap_mask"])
        h5.attrs["missing_inputs_json"] = _json_dumps(
            map_metadata.get("missing_inputs", missing_inputs_metadata())
        )
        h5.attrs["binning"] = str(result.get("binning", result["config"].get("binning", "sqrt")))
        h5.attrs["ell_max_inclusive"] = int(result.get("ell_max_inclusive", result["config"].get("lmax", 0)))

        _write_dataset(h5, "ell", np.asarray(result["ell"]), dtype="f8")
        _write_dataset(h5, "ell_left", np.asarray(result["ell_left"]), dtype="i4")
        _write_dataset(h5, "ell_right", np.asarray(result["ell_right"]), dtype="i4")

        sg = h5.create_group("spectra")
        for name, spec in result["spectra"].items():
            g = sg.create_group(name)
            for key in ("family", "label", "theory_key", "component_label"):
                g.attrs[key] = str(spec[key])
            g.attrs["fields"] = _json_dumps(list(spec["fields"]))
            g.attrs["component"] = int(spec["component"])
            g.attrs["component_labels"] = _json_dumps(spec["component_labels"])
            g.attrs["metadata_json"] = _json_dumps(spec["metadata"])
            if "pair_overlap_mean_subtraction" in spec:
                g.attrs["pair_overlap_mean_subtraction_json"] = _json_dumps(spec["pair_overlap_mean_subtraction"])
            if str(spec.get("family", "")) == "desi_g_auto":
                g.attrs["cl_convention"] = DESI_GALAXY_AUTO_MEAN_CONVENTION
                g.attrs["shot_noise_plotting_note"] = (
                    "The saved spectra/<name>/cl values already contain clustering signal plus "
                    "the weighted Poisson shot noise exactly once. The saved "
                    "noise_decoupled_all_components[component] is a provenance/nuisance template; "
                    "do not add it to the measured bandpowers a second time."
                )
            _write_dataset(g, "ell", spec["ell"], dtype="f8")
            _write_dataset(g, "cl", spec["cl"], dtype="f8")
            if str(spec.get("family", "")) == "desi_g_auto":
                noise_all = spec.get("noise_decoupled_all_components")
                if noise_all is None:
                    raise ValueError(
                        f"DESI galaxy auto {name!r} has no saved decoupled shot-noise template."
                    )
                noise_all = np.asarray(noise_all, dtype=np.float64)
                cl_all = np.asarray(spec["cl_all_components"], dtype=np.float64)
                component = int(spec["component"])
                if noise_all.shape != cl_all.shape or not (0 <= component < noise_all.shape[0]):
                    raise ValueError(
                        f"DESI galaxy auto {name!r} has incompatible total/noise component shapes."
                    )
                _write_dataset(
                    g,
                    "cl_weighted_poisson_subtracted",
                    np.asarray(spec["cl"], dtype=np.float64) - noise_all[component],
                    dtype="f8",
                )
                _write_dataset(
                    g,
                    "cl_all_components_weighted_poisson_subtracted",
                    cl_all - noise_all,
                    dtype="f8",
                )
            joint_result = result.get("joint")
            if isinstance(joint_result, Mapping):
                per_spectrum_validity = joint_result.get("spectrum_validity", {})
                if isinstance(per_spectrum_validity, Mapping) and name in per_spectrum_validity:
                    _write_dataset(g, "data_vector_valid", per_spectrum_validity[name], dtype="?")
            _write_dataset(g, "cl_all_components", spec["cl_all_components"], dtype="f8")
            _write_dataset(g, "pcl_all_components", spec["pcl_all_components"], dtype="f8")
            _write_dataset(g, "bandpower_window_selected", spec["bandpower_window_selected"], dtype="f8")
            if spec.get("err") is not None:
                _write_dataset(g, "err", spec["err"], dtype="f8")
            if spec.get("cov") is not None:
                _write_dataset(g, "cov", spec["cov"], dtype="f8")
            if spec.get("noise_decoupled_all_components") is not None:
                _write_dataset(g, "noise_decoupled_all_components", spec["noise_decoupled_all_components"], dtype="f8")

        if result.get("joint") is not None:
            joint = result["joint"]
            jg = h5.create_group("joint")
            _write_dataset(jg, "spectrum_names", _string_array(joint["spectrum_names"]))
            _write_dataset(jg, "ell", joint["ell"], dtype="f8")
            _write_dataset(jg, "data_vector", joint["data_vector"], dtype="f8")
            _write_dataset(jg, "data_vector_raw", joint.get("data_vector_raw", joint["data_vector"]), dtype="f8")
            _write_dataset(
                jg,
                "data_vector_valid",
                joint.get("data_vector_valid", np.ones_like(joint["data_vector"], dtype=bool)),
                dtype="?",
            )
            if joint.get("cov") is not None:
                _write_dataset(jg, "cov", joint["cov"], dtype="f8")
            else:
                jg.attrs["covariance_missing"] = True
            if joint.get("corr") is not None:
                _write_dataset(jg, "corr", joint["corr"], dtype="f8")
            starts = [joint["slices"][name][0] for name in joint["spectrum_names"]]
            stops = [joint["slices"][name][1] for name in joint["spectrum_names"]]
            _write_dataset(jg, "slice_start", np.asarray(starts, dtype=np.int64), dtype="i8")
            _write_dataset(jg, "slice_stop", np.asarray(stops, dtype=np.int64), dtype="i8")
            jg.attrs["diagnostics_json"] = _json_dumps(joint["diagnostics"])
            jg.attrs["data_vector_validity_contract_version"] = (
                DATA_VECTOR_VALIDITY_CONTRACT_VERSION
                if schema == SCHEMA_MEASUREMENT_VALIDITY_MASK
                else "all_bands_valid"
            )
            jg.attrs["covariance_basis"] = "joint/data_vector_raw full estimator bandpowers"
            jg.attrs["active_covariance_rule"] = (
                "Use joint/cov[np.ix_(joint/data_vector_valid, joint/data_vector_valid)]; "
                "this is the ordinary principal submatrix, not a Schur complement."
            )
            default_valid = np.ones(np.asarray(joint["data_vector"]).shape, dtype=bool)
            saved_valid = np.asarray(joint.get("data_vector_valid", default_valid), dtype=bool)
            jg.attrs["n_archive"] = int(np.asarray(joint["data_vector"]).size)
            jg.attrs["n_active"] = int(np.count_nonzero(saved_valid))
            jg.attrs["n_zero_placeholders"] = int(np.count_nonzero(~saved_valid))
            jg.attrs["data_vector_convention"] = (
                "joint/data_vector is assembled from spectra/<name>/cl, except explicitly invalid "
                "ACT-kappa transfer-null bands are exact zero placeholders and are marked false in "
                "joint/data_vector_valid; joint/data_vector_raw preserves every raw estimator band. "
                "DESI galaxy auto entries "
                "contain clustering signal plus the weighted Poisson shot noise exactly once. "
                "Theory comparisons window the clustering signal normally, then add their freely "
                "modelled amplitude times the saved, already-decoupled shot-noise template in "
                "bandpower space. Measurement readers must not add that template a second time."
            )
            if has_galaxy_auto_views:
                views = jg.create_group("views")
                views.attrs["contract_version"] = DESI_GALAXY_AUTO_VIEWS_CONTRACT_VERSION
                views.attrs["primary_view"] = DESI_GALAXY_AUTO_PRIMARY_VIEW
                views.attrs["covariance_conditioning"] = (
                    "Conditional on the fixed catalog, imaging weights, and random-mask realization. "
                    "Subtracting the deterministic weighted-Poisson template shifts the mean only; "
                    "both views therefore share the identical full Gaussian/iNKA estimator covariance."
                )
                template = np.asarray(
                    joint["galaxy_auto_weighted_poisson_template"], dtype=np.float64
                )
                if template.shape != np.asarray(joint["data_vector"]).shape:
                    raise ValueError("Joint galaxy-auto shot template has the wrong shape.")
                _write_dataset(
                    views,
                    "galaxy_auto_weighted_poisson_template",
                    template,
                    dtype="f8",
                )
                total_view = views.create_group(DESI_GALAXY_AUTO_PRIMARY_VIEW)
                subtracted_view = views.create_group(DESI_GALAXY_AUTO_SUBTRACTED_VIEW)
                total_view.attrs["mean_convention"] = DESI_GALAXY_AUTO_MEAN_CONVENTION
                total_view.attrs["hmc_primary"] = True
                subtracted_view.attrs["mean_convention"] = (
                    "total estimator minus the fixed saved weighted-Poisson bandpower template"
                )
                subtracted_view.attrs["hmc_primary"] = False
                for view in (total_view, subtracted_view):
                    view["data_vector_valid"] = jg["data_vector_valid"]
                    view["spectrum_names"] = jg["spectrum_names"]
                    view["slice_start"] = jg["slice_start"]
                    view["slice_stop"] = jg["slice_stop"]
                    view["ell"] = jg["ell"]
                    if "cov" in jg:
                        view["cov"] = jg["cov"]
                    if "corr" in jg:
                        view["corr"] = jg["corr"]
                total_view["data_vector"] = jg["data_vector"]
                total_view["data_vector_raw"] = jg["data_vector_raw"]
                _write_dataset(
                    subtracted_view,
                    "data_vector",
                    joint["data_vector_weighted_poisson_subtracted"],
                    dtype="f8",
                )
                _write_dataset(
                    subtracted_view,
                    "data_vector_raw",
                    joint["data_vector_raw_weighted_poisson_subtracted"],
                    dtype="f8",
                )

        bg = h5.create_group("covariance_blocks")
        for (name_i, name_j), block in result["covariance_blocks"].items():
            ds = _write_dataset(bg, f"{name_i}__x__{name_j}", block, dtype="f8")
            ds.attrs["spectrum_i"] = name_i
            ds.attrs["spectrum_j"] = name_j

        ng = h5.create_group("null_tests")
        for name, null in result["null_tests"].items():
            g = ng.create_group(name)
            _write_dataset(g, "ell", null["ell"], dtype="f8")
            _write_dataset(g, "cl", null["cl"], dtype="f8")
            for key, value in null.items():
                if key not in {"ell", "cl"}:
                    g.attrs[key] = _json_dumps(value) if not isinstance(value, str) else value

        ig = h5.create_group("input_cls_for_covariance")
        cfg = result["config"]
        covariance_input_mode = str(cfg.get("covariance_input_mode", "inka_data"))
        ig.attrs["mode"] = covariance_input_mode
        ig.attrs["namaster_coupled_argument"] = False
        if covariance_input_mode == "inka_data":
            ig.attrs["description"] = (
                "Full-ell total spectra passed to nmt.gaussian_covariance with coupled=False. "
                "Ordinary map pairs use nmt.get_iNKA_cell (measured coupled pseudo-Cl divided "
                "by mean mask product). Inputs involving DESI catalog-momentum fields use the "
                "equivalent explicit pseudo-Cl/mask-overlap form, with catalog Nf added back "
                "for momentum autos."
            )
        else:
            ig.attrs["description"] = (
                "Legacy full-ell inputs built from decoupled measured bandpowers, with auto noise "
                "added back, sanitized/smoothed, and unbinned. Retained only for reproducing old products."
            )
        ig.attrs["smooth_bandpowers"] = bool(
            covariance_input_mode != "inka_data" and cfg.get("covariance_input_smooth_bandpowers", True)
        )
        ig.attrs["smooth_window"] = int(cfg.get("covariance_input_smooth_window", 5))
        zero_parity_odd_inputs = bool(
            covariance_input_mode != "inka_data" and cfg.get("covariance_zero_parity_odd_inputs", True)
        )
        ig.attrs["zero_parity_odd_inputs"] = zero_parity_odd_inputs
        field_meta_for_cov = result.get("field_metadata", {})
        for key, cl in result["input_cls_for_covariance"].items():
            if len(key) == 3:
                input_mode, a, b = key
                dataset_name = f"{input_mode}__{a}__x__{b}"
            elif len(key) == 2:
                input_mode = "legacy"
                a, b = key
                dataset_name = f"{a}__x__{b}"
            else:
                raise ValueError(f"Unexpected covariance input cache key: {key!r}")
            ds = _write_dataset(ig, dataset_name, cl, dtype="f8")
            ds.attrs["input_mode"] = str(input_mode)
            ds.attrs["field_a"] = a
            ds.attrs["field_b"] = b
            if isinstance(field_meta_for_cov, Mapping) and a in field_meta_for_cov and b in field_meta_for_cov:
                ma = field_meta_for_cov[a]
                mb = field_meta_for_cov[b]
                spin_a = int(ma["spin"])
                spin_b = int(mb["spin"])
                ds.attrs["spin_a"] = spin_a
                ds.attrs["spin_b"] = spin_b
                ds.attrs["kind_a"] = str(ma.get("kind", ""))
                ds.attrs["kind_b"] = str(mb.get("kind", ""))
                ds.attrs["component_labels"] = _json_dumps(component_labels(spin_a, spin_b))
                ds.attrs["noise_policy"] = covariance_input_noise_policy(
                    a,
                    b,
                    field_meta_for_cov,
                    zero_parity_odd=zero_parity_odd_inputs,
                    input_mode=covariance_input_mode,
                )

        fg = h5.create_group("fields")
        fg.attrs["metadata_json"] = _json_dumps(result["field_metadata"])

        nzg = h5.create_group("nz")
        _write_des_shear_nz_group(nzg, map_metadata.get("des_y3_source_nz"))
        pg = h5.create_group("priors")
        _write_des_y3_priors_group(pg)

        tfg = h5.create_group("transfer_functions")
        cfg = result["config"]
        nside = int(cfg["nside"])
        lmax = int(cfg["lmax"])
        pix_t, pix_p = hp.pixwin(nside, lmax=lmax, pol=True)
        _write_dataset(tfg, "healpix_temperature_pixwin", pix_t, dtype="f8")
        _write_dataset(tfg, "healpix_polarization_pixwin", pix_p, dtype="f8")
        _write_act_beam_transfers(tfg, lmax)
        input_files = map_metadata.get("input_files", {})
        kappa_path = None
        if isinstance(input_files, Mapping) and "act_kappa" in input_files:
            kappa_path = input_files["act_kappa"].get("path")
        if kappa_path:
            try:
                with h5py.File(kappa_path, "r") as kh5:
                    _write_dataset(tfg, "act_kappa_filter_baseline", kh5["curves/kappa_filter_baseline"][:], dtype="f8")
                    _write_dataset(tfg, "act_kappa_noise_N_L_baseline", kh5["curves/N_L_kk_baseline"][:], dtype="f8")
            except Exception as exc:  # pragma: no cover - metadata best effort
                tfg.attrs["act_kappa_curve_read_error"] = str(exc)

        tig = h5.create_group("theory_interface")
        spectrum_names_for_interface = (
            list(result["joint"]["spectrum_names"])
            if result.get("joint") is not None
            else list(result["spectra"].keys())
        )
        spectrum_entries_for_interface = [result["spectra"][name] for name in spectrum_names_for_interface]
        tig.attrs["description"] = (
            "Use theory_to_data_vector(measurement_path, theory_cls, ...) to apply "
            "saved bandpower windows, default pixel-window transfers, and ACT Gaussian beams."
        )
        tig.attrs["desi_galaxy_auto_mean_convention"] = DESI_GALAXY_AUTO_MEAN_CONVENTION
        if has_galaxy_auto_views:
            tig.attrs["desi_galaxy_auto_views_contract_version"] = (
                DESI_GALAXY_AUTO_VIEWS_CONTRACT_VERSION
            )
            tig.attrs["desi_galaxy_auto_primary_hmc_view"] = (
                f"joint/views/{DESI_GALAXY_AUTO_PRIMARY_VIEW}; identical hard links to "
                "joint/data_vector and joint/cov"
            )
        tig.attrs["desi_galaxy_auto_shot_noise_nuisance"] = (
            DESI_GALAXY_AUTO_THEORY_INTERFACE_NOTE
        )
        tig.attrs["act_y_beam_fwhm_arcmin"] = ACT_TSZ_BEAM_FWHM_ARCMIN
        tig.attrs["act_cmb_temperature_beam_fwhm_arcmin"] = ACT_CMB_TEMPERATURE_BEAM_FWHM_ARCMIN
        tig.attrs["ksz_photoz_velocity_correlation_r"] = KSZ_PHOTOZ_VELOCITY_CORRELATION_R
        tig.attrs["ksz_photoz_velocity_correlation_fracerr"] = KSZ_PHOTOZ_VELOCITY_CORRELATION_FRACERR
        tig.attrs["ksz_velocity_calibration_reference"] = KSZ_REFERENCE_PAPER
        tig.attrs["ksz_sigma_true_gas_doc"] = KSZ_SIGMA_TRUE_GAS_DOC
        tig.attrs["ksz_sigma_true_gas_json_relpath"] = KSZ_SIGMA_TRUE_GAS_JSON_REL
        des_nz_meta = map_metadata.get("des_y3_source_nz", {})
        if isinstance(des_nz_meta, Mapping):
            tig.attrs["des_y3_source_nz_fits"] = str(des_nz_meta.get("source_fits", ""))
            tig.attrs["des_y3_source_nz_hdu"] = str(des_nz_meta.get("hdu", DES_Y3_SOURCE_NZ_HDU))
        desi_meta = map_metadata.get("desi_summary", {})
        if isinstance(desi_meta, Mapping):
            for src_key, attr_key in (
                ("redshift_kind", "desi_lens_nz_redshift_kind"),
                ("theory_nz_source_hdf5", "desi_lens_nz_source_hdf5"),
                ("theory_nz_group", "desi_lens_nz_group"),
                ("theory_nz_sample", "desi_lens_nz_sample"),
                ("theory_nz_dataset", "desi_lens_nz_dataset"),
                ("selection_dataset", "desi_selection_dataset"),
                ("weight_dataset", "desi_weight_dataset"),
            ):
                if src_key in desi_meta:
                    tig.attrs[attr_key] = str(desi_meta[src_key])
        tig.attrs["des_y3_gaussian_priors_json"] = _json_dumps(des_y3_gaussian_priors())
        field_meta = result["field_metadata"]
        if all(f"pi{i}" in field_meta for i in range(1, 5)):
            sigma_true_over_c = _sigma_true_by_pz_from_field_metadata(field_meta)
            sigma_true_km_s = []
            for pz_bin in range(1, 5):
                meta_outer = field_meta.get(f"pi{pz_bin}", {})
                meta = meta_outer.get("metadata", {}) if isinstance(meta_outer, Mapping) else {}
                sigma_true_km_s.append(float(meta.get("sigma_true_gas_km_s", KSZ_SIGMA_TRUE_GAS_KM_S[pz_bin])))
            default_ksz_amps = ksz_velocity_amplitudes_from_field_metadata(field_meta)
            _write_dataset(
                tig,
                "ksz_sigma_true_gas_over_c_by_pz",
                np.asarray([sigma_true_over_c[i] for i in range(1, 5)], dtype=np.float64),
                dtype="f8",
            )
            _write_dataset(tig, "ksz_sigma_true_gas_km_s_by_pz", np.asarray(sigma_true_km_s, dtype=np.float64), dtype="f8")
            _write_dataset(
                tig,
                "ksz_default_A_v_by_pz",
                np.asarray([default_ksz_amps[i] for i in range(1, 5)], dtype=np.float64),
                dtype="f8",
            )
        _write_dataset(tig, "spectrum_names", _string_array(spectrum_names_for_interface))
        _write_dataset(tig, "theory_keys", _string_array([str(s["theory_key"]) for s in spectrum_entries_for_interface]))
        _write_dataset(tig, "field_a", _string_array([str(s["fields"][0]) for s in spectrum_entries_for_interface]))
        _write_dataset(tig, "field_b", _string_array([str(s["fields"][1]) for s in spectrum_entries_for_interface]))
        _write_dataset(
            tig,
            "component",
            np.asarray([int(s["component"]) for s in spectrum_entries_for_interface], dtype=np.int16),
            dtype="i2",
        )

    os.replace(tmp, path)
    return path


def _interp_to_lmax(values: np.ndarray, ell_in: Optional[np.ndarray], lmax: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == lmax + 1 and ell_in is None:
        return values
    ell_full = np.arange(lmax + 1, dtype=np.float64)
    if ell_in is None:
        ell_in = np.arange(values.size, dtype=np.float64)
    ell_in = np.asarray(ell_in, dtype=np.float64)
    return np.interp(ell_full, ell_in, values, left=values[0], right=values[-1])


def _value_for_pz(values: object, pz_bin: int, label: str) -> float:
    if isinstance(values, Mapping):
        for key in (pz_bin, str(pz_bin), f"pz{pz_bin}"):
            if key in values:
                return float(values[key])
        raise KeyError(f"Missing {label} for pz bin {pz_bin}.")
    if isinstance(values, np.ndarray) or (
        isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray))
    ):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            return float(arr)
        if arr.size >= 4:
            return float(arr[pz_bin - 1])
        raise ValueError(f"{label} sequence must be scalar-like or contain at least four pz-bin values.")
    return float(values)


def _value_for_bin(values: object, bin_index: int, label: str) -> float:
    if isinstance(values, Mapping):
        for key in (bin_index, str(bin_index), f"bin{bin_index}", f"s{bin_index}", f"tomo{bin_index}"):
            if key in values:
                return float(values[key])
        raise KeyError(f"Missing {label} for bin {bin_index}.")
    if isinstance(values, np.ndarray) or (
        isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray))
    ):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            return float(arr)
        if arr.size >= 4:
            return float(arr[bin_index - 1])
        raise ValueError(f"{label} sequence must be scalar-like or contain at least four bin values.")
    return float(values)


def _shear_multiplicative_factor(field_names: Sequence[str], shear_m_bias: Optional[object]) -> float:
    if shear_m_bias is None:
        return 1.0
    factor = 1.0
    for field_name in field_names:
        if len(field_name) > 1 and field_name[0] == "s" and field_name[1:].isdigit():
            bin_index = int(field_name[1:])
            factor *= 1.0 + _value_for_bin(shear_m_bias, bin_index, "shear_m_bias")
    return float(factor)


def _shear_e_to_positive_kappa_factor(
    field_names: Sequence[str],
    field_meta: Mapping[str, object],
) -> float:
    """Convert positive-convergence shear theory to the saved E-mode convention."""

    factor = 1.0
    for field_name in field_names:
        meta_outer = field_meta.get(field_name, {})
        if not isinstance(meta_outer, Mapping):
            continue
        if str(meta_outer.get("kind", "")) != "des_shear":
            continue
        meta = meta_outer.get("metadata", {})
        if not isinstance(meta, Mapping):
            meta = {}
        sign = float(meta.get("shear_e_to_kappa_sign", 1.0))
        factor *= -sign
    return float(factor)


def _sigma_rec_by_pz_from_field_metadata(field_meta: Mapping[str, object]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for pz_bin in range(1, 5):
        meta_outer = field_meta.get(f"pi{pz_bin}", {})
        if not isinstance(meta_outer, Mapping):
            continue
        meta = meta_outer.get("metadata", {})
        if not isinstance(meta, Mapping):
            continue
        sigma = meta.get(
            "rms_rec_vr_over_c_weighted",
            meta.get("rms_rec_vr_over_c", meta.get("sigma_rec_vr_over_c")),
        )
        if sigma is not None and np.isfinite(float(sigma)):
            out[pz_bin] = float(sigma)
    return out


def _sigma_true_by_pz_from_field_metadata(field_meta: Mapping[str, object]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for pz_bin in range(1, 5):
        meta_outer = field_meta.get(f"pi{pz_bin}", {})
        if not isinstance(meta_outer, Mapping):
            continue
        meta = meta_outer.get("metadata", {})
        if not isinstance(meta, Mapping):
            continue
        sigma = meta.get("sigma_true_gas_over_c")
        if sigma is not None and np.isfinite(float(sigma)):
            out[pz_bin] = float(sigma)
    if set(out) == {1, 2, 3, 4}:
        return out
    return dict(KSZ_SIGMA_TRUE_GAS_OVER_C_3E5)


def ksz_velocity_amplitudes_from_field_metadata(
    field_meta: Mapping[str, object],
    sigma_true_over_c: Optional[object] = None,
    velocity_correlation: object = KSZ_PHOTOZ_VELOCITY_CORRELATION_R,
) -> Dict[int, float]:
    """Build A_v_bin = r_bin * sigma_rec_bin/c * sigma_true_bin/c for kSZ theory."""

    sigma_rec_by_pz = _sigma_rec_by_pz_from_field_metadata(field_meta)
    if sigma_true_over_c is None:
        sigma_true_over_c = _sigma_true_by_pz_from_field_metadata(field_meta)
    out: Dict[int, float] = {}
    for pz_bin in range(1, 5):
        if pz_bin not in sigma_rec_by_pz:
            raise KeyError(f"Missing reconstructed velocity RMS metadata for DESI pz bin {pz_bin}.")
        r_bin = _value_for_pz(velocity_correlation, pz_bin, "velocity_correlation")
        sigma_true_bin = _value_for_pz(sigma_true_over_c, pz_bin, "sigma_true_over_c")
        out[pz_bin] = float(r_bin) * float(sigma_rec_by_pz[pz_bin]) * float(sigma_true_bin)
    return out


def ksz_velocity_amplitudes_from_measurement(
    measurement_path: str | Path,
    sigma_true_over_c: Optional[object] = None,
    velocity_correlation: object = KSZ_PHOTOZ_VELOCITY_CORRELATION_R,
) -> Dict[int, float]:
    """Read saved DESI velocity RMS values and return paper-calibrated kSZ A_v bins."""

    with h5py.File(measurement_path, "r") as h5:
        field_meta = json.loads(h5["fields"].attrs["metadata_json"])
    return ksz_velocity_amplitudes_from_field_metadata(field_meta, sigma_true_over_c, velocity_correlation)


def _beam_transfer_from_h5_or_default(
    h5: h5py.File,
    dataset: str,
    fwhm_arcmin: float,
    lmax: int,
) -> np.ndarray:
    if dataset in h5:
        return _interp_to_lmax(h5[dataset][:], None, lmax)
    return gaussian_beam_transfer(lmax, fwhm_arcmin)


def _load_default_transfers(
    h5: h5py.File,
    lmax: int,
    include_pixel_windows: bool = True,
    include_act_beams: bool = True,
) -> Dict[str, np.ndarray]:
    config = json.loads(h5.attrs["config_json"])
    nside = int(config["nside"])
    if include_pixel_windows and "transfer_functions/healpix_temperature_pixwin" in h5:
        pix_t = h5["transfer_functions/healpix_temperature_pixwin"][:]
        pix_p = h5["transfer_functions/healpix_polarization_pixwin"][:]
    elif include_pixel_windows:
        pix_t, pix_p = hp.pixwin(nside, lmax=lmax, pol=True)
    else:
        pix_t = np.ones(lmax + 1, dtype=np.float64)
        pix_p = np.ones(lmax + 1, dtype=np.float64)
    field_meta = json.loads(h5["fields"].attrs["metadata_json"])
    transfers: Dict[str, np.ndarray] = {}
    for name, meta in field_meta.items():
        kind = str(meta.get("kind", ""))
        if not include_pixel_windows:
            transfers[name] = np.ones(lmax + 1, dtype=np.float64)
        elif kind == "des_shear":
            transfers[name] = pix_p
        elif kind == "desi_galaxy":
            transfers[name] = pix_t
        else:
            # ACT maps are harmonically reprojected to HEALPix samples (no
            # output HEALPix pixel window), and catalog momentum fields are
            # measured directly from object positions. Applying hp.pixwin to
            # either class would spuriously damp their theory templates.
            transfers[name] = np.ones(lmax + 1, dtype=np.float64)
    if include_act_beams:
        if "y" in transfers:
            transfers["y"] = transfers["y"] * _beam_transfer_from_h5_or_default(
                h5,
                "transfer_functions/act_y_gaussian_beam",
                ACT_TSZ_BEAM_FWHM_ARCMIN,
                lmax,
            )
        if "T" in transfers:
            transfers["T"] = transfers["T"] * _beam_transfer_from_h5_or_default(
                h5,
                "transfer_functions/act_cmb_temperature_gaussian_beam",
                ACT_CMB_TEMPERATURE_BEAM_FWHM_ARCMIN,
                lmax,
            )
    if "kappa" in transfers and "transfer_functions/act_kappa_filter_baseline" in h5:
        curve = h5["transfer_functions/act_kappa_filter_baseline"][:]
        filt = np.interp(
            np.arange(lmax + 1, dtype=np.float64),
            np.asarray(curve[:, 0], dtype=np.float64),
            np.asarray(curve[:, 1], dtype=np.float64),
            left=float(curve[0, 1]),
            right=float(curve[-1, 1]),
        )
        transfers["kappa"] = transfers["kappa"] * filt
    return transfers


def theory_to_data_vector(
    measurement_path: str | Path,
    theory_cls: Mapping[str, np.ndarray],
    ell: Optional[np.ndarray] = None,
    transfer_functions: Optional[Mapping[str, np.ndarray]] = None,
    transfer_ell: Optional[np.ndarray] = None,
    ksz_velocity_amplitudes: Optional[Mapping[int, float]] = None,
    ksz_sigma_true_over_c: Optional[object] = None,
    ksz_velocity_correlation: object = KSZ_PHOTOZ_VELOCITY_CORRELATION_R,
    desi_galaxy_shot_noise_amplitudes: Optional[object] = 1.0,
    shear_m_bias: Optional[object] = None,
    theory_shear_e_is_positive_kappa: bool = True,
    tcmb_uk: float = TCMB_UK,
    include_default_pixel_windows: bool = True,
    include_default_act_beams: bool = True,
    include_invalid_placeholders: bool = False,
    allow_legacy_product: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Convolve smooth theory spectra with the saved measurement windows.

    ``theory_cls`` may be keyed by exact measured spectrum name.  For kSZ
    spectra, it may instead provide ``desi_g_tau_pz{i}``; the wrapper then
    applies ``-T_CMB_uK * A_v_bin``.  Provide ``ksz_velocity_amplitudes``
    directly, or let the wrapper use Abacus ``sigma_true_gas`` with the paper
    default ``r=0.3`` and the saved reconstructed velocity RMS values.
    If ``shear_m_bias`` is provided, any spectrum involving DES shear field
    ``s{i}`` is multiplied by ``prod_i(1 + m_i)``.
    By default, DES shear theory inputs are interpreted as positive-convergence
    E-mode spectra; the wrapper converts them to the saved E-mode sign
    convention recorded in the measurement metadata.
    Current DESI galaxy-auto measurements contain signal plus weighted Poisson
    shot noise.  Supply their ``theory_cls`` arrays as clustering signal only;
    after windowing that signal, this wrapper adds
    ``desi_galaxy_shot_noise_amplitudes[pz]`` times the saved, estimator-matched
    decoupled shot-noise template.  The default amplitude 1 reproduces the
    catalog Poisson prediction and the amplitudes can be varied as nuisance
    parameters.  Do not put a flat shot term into ``theory_cls``: the ordinary
    galaxy pixel transfer is not the response of this conditional pseudo-Cl
    noise template.
    Legacy products are rejected unless ``allow_legacy_product=True`` is supplied
    explicitly for a historical-only comparison.
    Products carrying ``joint/data_vector_valid`` return only the statistically
    active entries by default.  Set ``include_invalid_placeholders=True`` only
    for archive-layout diagnostics; the returned invalid entries are exact zeros.
    """

    measurement_path = Path(measurement_path)
    out: List[np.ndarray] = []
    names: List[str] = []
    data_vector_valid: Optional[np.ndarray] = None
    with h5py.File(measurement_path, "r") as h5:
        validate_measurement_product_identity(
            h5,
            allow_legacy_product=allow_legacy_product,
        )
        config = json.loads(h5.attrs["config_json"])
        lmax = int(config["lmax"])
        field_meta = json.loads(h5["fields"].attrs["metadata_json"])
        default_transfers = _load_default_transfers(
            h5,
            lmax,
            include_pixel_windows=include_default_pixel_windows,
            include_act_beams=include_default_act_beams,
        )
        transfers = dict(default_transfers)
        if transfer_functions is not None:
            for field_name, values in transfer_functions.items():
                base = transfers.get(field_name, np.ones(lmax + 1, dtype=np.float64))
                transfers[field_name] = base * _interp_to_lmax(np.asarray(values), transfer_ell, lmax)
        ksz_amps: Optional[Dict[int, float]] = None

        spectrum_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["joint/spectrum_names"][:]]
        if "joint/data_vector_valid" in h5:
            data_vector_valid = np.asarray(h5["joint/data_vector_valid"][:], dtype=bool)
        for name in spectrum_names:
            g = h5[f"spectra/{name}"]
            fields = json.loads(g.attrs["fields"])
            theory_key = str(g.attrs["theory_key"])
            meta = json.loads(g.attrs["metadata_json"])
            if name in theory_cls:
                cl = _interp_to_lmax(np.asarray(theory_cls[name]), ell, lmax)
            elif theory_key in theory_cls and str(g.attrs["family"]) == "desi_pi_act_T":
                pz_bin = int(meta["desi_pz"])
                if ksz_velocity_amplitudes is not None:
                    amp = _value_for_pz(ksz_velocity_amplitudes, pz_bin, "ksz_velocity_amplitudes")
                else:
                    if ksz_amps is None:
                        ksz_amps = ksz_velocity_amplitudes_from_field_metadata(
                            field_meta,
                            ksz_sigma_true_over_c,
                            ksz_velocity_correlation,
                        )
                    if pz_bin not in ksz_amps:
                        raise KeyError(
                            f"Missing kSZ velocity amplitude for pz bin {pz_bin}. "
                            "Provide ksz_velocity_amplitudes directly, or provide "
                            "ksz_sigma_true_over_c if the saved/default Abacus calibration is not appropriate."
                        )
                    amp = float(ksz_amps[pz_bin])
                cl_tau = _interp_to_lmax(np.asarray(theory_cls[theory_key]), ell, lmax)
                cl = -float(tcmb_uk) * amp * cl_tau
            elif theory_key in theory_cls:
                cl = _interp_to_lmax(np.asarray(theory_cls[theory_key]), ell, lmax)
            else:
                raise KeyError(f"No theory spectrum provided for {name} or theory key {theory_key}.")
            if theory_shear_e_is_positive_kappa:
                cl = cl * _shear_e_to_positive_kappa_factor(fields, field_meta)
            cl = cl * _shear_multiplicative_factor(fields, shear_m_bias)
            tf = transfers.get(fields[0], np.ones(lmax + 1)) * transfers.get(fields[1], np.ones(lmax + 1))
            window = g["bandpower_window_selected"][:]
            bandpower = window @ (cl[: window.shape[1]] * tf[: window.shape[1]])
            family = str(g.attrs["family"])
            if family == "desi_g_auto":
                cl_convention = str(g.attrs.get("cl_convention", ""))
                if cl_convention == DESI_GALAXY_AUTO_MEAN_CONVENTION:
                    if desi_galaxy_shot_noise_amplitudes is None:
                        raise ValueError(
                            "Current DESI galaxy-auto data include shot noise. Provide "
                            "desi_galaxy_shot_noise_amplitudes (1 matches the catalog "
                            "Poisson template) rather than comparing to signal-only theory."
                        )
                    if "noise_decoupled_all_components" not in g:
                        raise ValueError(f"DESI galaxy auto {name!r} has no saved shot-noise template.")
                    component = int(g.attrs["component"])
                    noise_all = np.asarray(g["noise_decoupled_all_components"][:], dtype=np.float64)
                    if not (0 <= component < noise_all.shape[0]):
                        raise ValueError(
                            f"DESI galaxy auto {name!r} component {component} is outside "
                            f"saved noise shape {noise_all.shape}."
                        )
                    shot_template = np.asarray(noise_all[component], dtype=np.float64)
                    if shot_template.shape != bandpower.shape:
                        raise ValueError(
                            f"DESI galaxy auto {name!r} shot template has shape "
                            f"{shot_template.shape}, expected {bandpower.shape}."
                        )
                    pz_bin = int(meta["desi_pz"])
                    shot_amplitude = _value_for_pz(
                        desi_galaxy_shot_noise_amplitudes,
                        pz_bin,
                        "desi_galaxy_shot_noise_amplitudes",
                    )
                    bandpower = bandpower + shot_amplitude * shot_template
                elif not allow_legacy_product:
                    raise ValueError(
                        f"DESI galaxy auto {name!r} has cl_convention={cl_convention!r}, "
                        f"expected {DESI_GALAXY_AUTO_MEAN_CONVENTION!r}."
                    )
            out.append(bandpower)
            names.append(name)
    vector = np.concatenate(out)
    if data_vector_valid is not None:
        if data_vector_valid.shape != vector.shape:
            raise ValueError(
                f"Saved data-vector validity shape {data_vector_valid.shape} does not match theory shape {vector.shape}."
            )
        if include_invalid_placeholders:
            vector = vector.copy()
            vector[~data_vector_valid] = 0.0
        else:
            vector = vector[data_vector_valid]
    return vector, names


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stage",
        choices=["lowres", "fast1024", "midres2048", "full", "highres4096"],
        default="lowres",
    )
    parser.add_argument("--survey-root", default="data/xDESI/survey_data")
    parser.add_argument("--output-dir", default="data/xDESI/processed/multiprobe_namaster")
    parser.add_argument("--nside", type=int, default=None)
    parser.add_argument("--lmax", type=int, default=None)
    parser.add_argument("--lmax-mask", type=int, default=None)
    parser.add_argument("--ell-min", type=int, default=None)
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--binning", choices=["sqrt", "linear", "log"], default=None)
    parser.add_argument("--act-downgrade", type=int, default=None)
    parser.add_argument("--catalog-chunk", type=int, default=2_000_000)
    parser.add_argument("--shear-e-to-kappa-sign", type=float, choices=[-1.0, 1.0], default=None)
    parser.add_argument("--des-y3-source-nz-fits", default=None)
    parser.add_argument("--mask-apodization-deg", type=float, default=None)
    parser.add_argument("--mask-apodization-type", choices=["C1", "C2", "Smooth"], default=None)
    parser.add_argument(
        "--pair-overlap-mean-subtraction",
        dest="pair_overlap_mean_subtract",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-pair-overlap-mean-subtraction",
        dest="pair_overlap_mean_subtract",
        action="store_false",
    )
    parser.add_argument("--covariance-l-toeplitz", type=int, default=None)
    parser.add_argument("--covariance-l-exact", type=int, default=None)
    parser.add_argument("--covariance-dl-band", type=int, default=None)
    parser.add_argument("--covariance-workspace-cache-size", type=int, default=None)
    parser.add_argument("--covariance-input-smooth-window", type=int, default=None)
    parser.add_argument("--no-covariance-input-smoothing", action="store_true")
    parser.add_argument("--keep-covariance-parity-odd-inputs", action="store_true")
    parser.add_argument("--force", action="store_true")


def config_from_args(args: argparse.Namespace) -> MeasurementConfig:
    config = MeasurementConfig.for_stage(args.stage)
    config.output_dir = args.output_dir
    config.catalog_chunk = int(args.catalog_chunk)
    if args.nside is not None:
        config.nside = int(args.nside)
    if args.lmax is not None:
        config.lmax = int(args.lmax)
    if args.lmax_mask is not None:
        config.lmax_mask = int(args.lmax_mask)
    if args.ell_min is not None:
        config.ell_min = int(args.ell_min)
    if args.n_bins is not None:
        config.n_bins = int(args.n_bins)
    if args.binning is not None:
        config.binning = str(args.binning)
    if args.act_downgrade is not None:
        config.act_downgrade = int(args.act_downgrade)
    if args.shear_e_to_kappa_sign is not None:
        config.shear_e_to_kappa_sign = float(args.shear_e_to_kappa_sign)
    if args.des_y3_source_nz_fits is not None:
        config.des_y3_source_nz_fits = str(args.des_y3_source_nz_fits)
    if args.mask_apodization_deg is not None:
        config.mask_apodization_deg = float(args.mask_apodization_deg)
    if args.mask_apodization_type is not None:
        config.mask_apodization_type = str(args.mask_apodization_type)
    if args.pair_overlap_mean_subtract is not None:
        config.pair_overlap_mean_subtract = bool(args.pair_overlap_mean_subtract)
    if args.covariance_l_toeplitz is not None:
        config.covariance_l_toeplitz = int(args.covariance_l_toeplitz)
    if args.covariance_l_exact is not None:
        config.covariance_l_exact = int(args.covariance_l_exact)
    if args.covariance_dl_band is not None:
        config.covariance_dl_band = int(args.covariance_dl_band)
    if args.covariance_workspace_cache_size is not None:
        config.covariance_workspace_cache_size = int(args.covariance_workspace_cache_size)
    if args.covariance_input_smooth_window is not None:
        config.covariance_input_smooth_window = int(args.covariance_input_smooth_window)
    if args.no_covariance_input_smoothing:
        config.covariance_input_smooth_bandpowers = False
    if args.keep_covariance_parity_odd_inputs:
        config.covariance_zero_parity_odd_inputs = False
    config.validate()
    return config
