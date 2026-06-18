"""Helpers for GODMAX theory comparisons to xDESI multi-probe measurements."""

from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import yaml


SCHEMA_MEASUREMENT = "xdesi_multiprobe_measurement_v1"
DESI_TRUE_NZ_REDSHIFT_KIND = "spectroscopic_calibrated_true_redshift"
HALO_MASS_FLOOR_TOL = 5.0e-7
HOD_ARRAY_BASE_NAMES = (
    "log10M1_fshmr",
    "log10M1_a_fshmr",
    "log10Mstar0_fshmr",
    "log10Mstar0_a_fshmr",
    "beta_fshmr",
    "beta_a_fshmr",
    "delta_fshmr",
    "delta_a_fshmr",
    "gamma_fshmr",
    "gamma_a_fshmr",
    "siglogMstar_Ncen",
    "alphasat_Nsat",
    "Bcut_Nsat",
    "Bsat_Nsat",
    "betacut_Nsat",
    "betasat_Nsat",
    "fcen",
)


@dataclass(frozen=True)
class MeasurementData:
    path: Path
    names: List[str]
    ell: np.ndarray
    data_vector: np.ndarray
    covariance: np.ndarray
    starts: np.ndarray
    stops: np.ndarray
    families: Dict[str, str]
    labels: Dict[str, str]
    theory_keys: Dict[str, str]
    ell_left: Optional[np.ndarray] = None
    ell_right: Optional[np.ndarray] = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: Mapping[str, object], override: Mapping[str, object]) -> dict:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_repo_path(path: str | Path, root: Optional[Path] = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (root or repo_root()) / path


def _json_load_attr(attrs: h5py.AttributeManager, key: str, default):
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, axis=axis)
    return np.trapz(y, x=x, axis=axis)


def _normalize_rows(
    z: np.ndarray,
    values: np.ndarray,
    label: str,
    warnings: List[str],
    rtol: float = 5.0e-4,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {arr.shape}.")
    norms = _trapz(arr, z, axis=1)
    for i, norm in enumerate(norms, start=1):
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"{label} bin {i} has invalid normalization {norm}.")
        if not np.isclose(norm, 1.0, rtol=rtol, atol=rtol):
            warnings.append(f"{label} bin {i} renormalized from integral {norm:.8g}.")
            arr[i - 1] /= norm
    return arr


def _support_edges(z_mid: np.ndarray, z_edges: Optional[np.ndarray], dndz: np.ndarray) -> np.ndarray:
    edges = []
    for row in np.asarray(dndz):
        support = np.flatnonzero(np.isfinite(row) & (row > 0.0))
        if support.size == 0:
            raise ValueError("DESI lens n(z) has an empty pz-bin support.")
        first = int(support[0])
        last = int(support[-1])
        if z_edges is not None and len(z_edges) == len(z_mid) + 1:
            edges.append([float(z_edges[first]), float(z_edges[last + 1])])
        else:
            edges.append([float(z_mid[first]), float(z_mid[last])])
    return np.asarray(edges, dtype=np.float64)


def _priors_from_measurement(measurement_path: Path) -> dict:
    with h5py.File(measurement_path, "r") as h5:
        if "theory_interface" not in h5:
            return {}
        raw = h5["theory_interface"].attrs.get("des_y3_gaussian_priors_json", "{}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _shear_m_means(priors: Mapping[str, object]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for i in range(1, 5):
        entry = priors.get(f"mult_shear_bias_bin{i}", {})
        out[i] = float(entry.get("mu", 0.0)) if isinstance(entry, Mapping) else 0.0
    return out


def _delta_z_means(priors: Mapping[str, object]) -> List[float]:
    out: List[float] = []
    for i in range(1, 5):
        entry = priors.get(f"Delta_z_bias_bin{i}", {})
        out.append(float(entry.get("mu", 0.0)) if isinstance(entry, Mapping) else 0.0)
    return out


def _serializable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes, bytearray)):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, Mapping):
        return {str(key): _serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(value) for value in obj]
    return obj


def to_jsonable(obj):
    return _serializable(obj)


def validate_comparison_halo_mass_floor(params: Mapping[str, object], raw: Mapping[str, object]) -> dict:
    required = raw.get("minimum_halo_log10_m200c_hmsun")
    halo_params = params.get("halo_params", {})
    actual = halo_params.get("lg10_Mmin")
    if required is None:
        return {
            "required": None,
            "actual": float(actual) if actual is not None else None,
            "enforced": False,
        }
    if actual is None:
        raise ValueError(
            "Comparison config requires minimum_halo_log10_m200c_hmsun="
            f"{float(required):.6f}, but merged halo_params.lg10_Mmin is missing."
        )
    actual_float = float(actual)
    required_float = float(required)
    if not math.isfinite(actual_float):
        raise ValueError(f"halo_params.lg10_Mmin is not finite: {actual!r}.")
    if abs(actual_float - required_float) > HALO_MASS_FLOOR_TOL:
        raise ValueError(
            "Backlight-compatible theory comparisons require the GODMAX halo mass grid "
            f"to start at log10(M200c/[Msun/h])={required_float:.6f}; "
            f"merged halo_params.lg10_Mmin={actual_float:.6f}."
        )
    return {
        "required": required_float,
        "actual": actual_float,
        "enforced": True,
        "mass_definition": "M200c",
        "mass_units": "Msun/h",
    }


def load_comparison_config(config_path: str | Path) -> dict:
    """Load and merge the path-backed comparison YAML into GODMAX parameters."""

    root = repo_root()
    config_path = resolve_repo_path(config_path, root)
    raw = read_yaml(config_path)
    base_path = resolve_repo_path(raw["base_params"], root)
    xdesi_path = resolve_repo_path(raw["xdesi_params"], root)
    params = deep_update(read_yaml(base_path), read_yaml(xdesi_path))
    if "cosmology" in raw:
        params.setdefault("sim_params", {}).setdefault("cosmo", {}).update(copy.deepcopy(raw["cosmology"]))
    if "overrides" in raw:
        params = deep_update(params, raw["overrides"])
    halo_mass_floor = validate_comparison_halo_mass_floor(params, raw)

    paths = {
        "config": config_path,
        "base_params": base_path,
        "xdesi_params": xdesi_path,
        "measurement_h5": resolve_repo_path(raw["measurement_h5"], root),
        "map_h5": resolve_repo_path(raw["map_h5"], root),
        "output_dir": resolve_repo_path(raw["output_dir"], root),
    }
    return {
        "repo_root": root,
        "raw": raw,
        "params": params,
        "paths": paths,
        "warnings": [],
        "metadata": {"halo_mass_floor": halo_mass_floor},
    }


def load_measurement_data(measurement_path: str | Path) -> MeasurementData:
    path = resolve_repo_path(measurement_path)
    with h5py.File(path, "r") as h5:
        schema = str(h5.attrs.get("schema", ""))
        if schema != SCHEMA_MEASUREMENT:
            raise ValueError(f"{path} is not a {SCHEMA_MEASUREMENT} product; schema={schema!r}.")
        names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in h5["joint/spectrum_names"][:]]
        starts = h5["joint/slice_start"][:].astype(int)
        stops = h5["joint/slice_stop"][:].astype(int)
        families: Dict[str, str] = {}
        labels: Dict[str, str] = {}
        theory_keys: Dict[str, str] = {}
        for name in names:
            group = h5[f"spectra/{name}"]
            families[name] = str(group.attrs.get("family", "unknown"))
            labels[name] = str(group.attrs.get("label", name))
            theory_keys[name] = str(group.attrs.get("theory_key", name))
        return MeasurementData(
            path=path,
            names=names,
            ell=h5["joint/ell"][:],
            data_vector=h5["joint/data_vector"][:],
            covariance=h5["joint/cov"][:],
            starts=starts,
            stops=stops,
            families=families,
            labels=labels,
            theory_keys=theory_keys,
            ell_left=h5["ell_left"][:] if "ell_left" in h5 else None,
            ell_right=h5["ell_right"][:] if "ell_right" in h5 else None,
        )


def materialize_nz_inputs(config: Mapping[str, object]) -> dict:
    """Read HDF5 n(z) products and inject GODMAX-ready n(z) dictionaries."""

    cfg = copy.deepcopy(dict(config))
    raw = cfg["raw"]
    datasets = raw["datasets"]
    paths = cfg["paths"]
    measurement_path = Path(paths["measurement_h5"])
    map_path = Path(paths["map_h5"])
    if not measurement_path.exists():
        raise FileNotFoundError(f"Missing measurement HDF5: {measurement_path}")
    if not map_path.exists():
        raise FileNotFoundError(f"Missing map HDF5 with DESI weighted n(z): {map_path}")

    warnings = list(cfg.get("warnings", []))
    with h5py.File(measurement_path, "r") as mh5, h5py.File(map_path, "r") as map_h5:
        schema = str(mh5.attrs.get("schema", ""))
        if schema != SCHEMA_MEASUREMENT:
            raise ValueError(f"{measurement_path} schema={schema!r}, expected {SCHEMA_MEASUREMENT!r}.")
        measurement_config = json.loads(str(mh5.attrs["config_json"]))
        field_meta = _json_load_attr(mh5["fields"].attrs, "metadata_json", {})
        priors = _priors_from_measurement(measurement_path)

        des_cfg = datasets["des_source_nz"]
        src_file = mh5 if des_cfg["file"] == "measurement_h5" else map_h5
        z_source = np.asarray(src_file[des_cfg["z_mid"]][:], dtype=np.float64)
        dndz_source = _normalize_rows(
            z_source,
            np.asarray(src_file[des_cfg["dndz_by_bin"]][:], dtype=np.float64),
            "DES source n(z)",
            warnings,
        )

        lens_cfg = datasets["desi_lens_nz"]
        lens_file = map_h5 if lens_cfg["file"] == "map_h5" else mh5
        z_lens = np.asarray(lens_file[lens_cfg["z_mid"]][:], dtype=np.float64)
        z_edges = np.asarray(lens_file[lens_cfg["z_edges"]][:], dtype=np.float64)
        lens_group_path = str(Path(lens_cfg["dndz_by_pz"]).parent)
        lens_group = lens_file[lens_group_path] if lens_group_path in lens_file else None
        lens_redshift_kind = str(lens_group.attrs.get("redshift_kind", "")) if lens_group is not None else ""
        if lens_redshift_kind != DESI_TRUE_NZ_REDSHIFT_KIND:
            raise ValueError(
                "DESI lens n(z) is not the calibrated true-redshift kernel. "
                f"Read redshift_kind={lens_redshift_kind!r} from {lens_cfg['dndz_by_pz']}; "
                "rerun the fast1024 measurement into the true-n(z) output directory."
            )
        dndz_lens = _normalize_rows(
            z_lens,
            np.asarray(lens_file[lens_cfg["dndz_by_pz"]][:], dtype=np.float64),
            "DESI calibrated true lens n(z)",
            warnings,
        )
        lens_edges = _support_edges(z_lens, z_edges, dndz_lens)
        lens_nz_provenance = {}
        if lens_group is not None:
            for key, value in lens_group.attrs.items():
                lens_nz_provenance[str(key)] = value.decode("utf-8") if isinstance(value, bytes) else value
        photoz_diagnostic = None
        if "nz/desi_photoz_diagnostic" in lens_file:
            pg = lens_file["nz/desi_photoz_diagnostic"]
            photoz_diagnostic = {
                "z_mid": np.asarray(pg["z_mid"][:], dtype=np.float64) if "z_mid" in pg else None,
                "z_edges": np.asarray(pg["z_edges"][:], dtype=np.float64) if "z_edges" in pg else None,
                "dndz_by_pz": np.asarray(pg["nz_dndz_by_pz"][:], dtype=np.float64) if "nz_dndz_by_pz" in pg else None,
                "description": str(pg.attrs.get("description", "")),
                "redshift_kind": str(pg.attrs.get("redshift_kind", "")),
            }

        ksz_default_a_v = (
            mh5["theory_interface/ksz_default_A_v_by_pz"][:]
            if "theory_interface/ksz_default_A_v_by_pz" in mh5
            else np.full(4, np.nan)
        )
        ksz_sigma_true = (
            mh5["theory_interface/ksz_sigma_true_gas_over_c_by_pz"][:]
            if "theory_interface/ksz_sigma_true_gas_over_c_by_pz" in mh5
            else np.full(4, np.nan)
        )

    source_info = {
        "nbins": int(dndz_source.shape[0]),
        "z_array_source": z_source.tolist(),
    }
    for i in range(dndz_source.shape[0]):
        source_info[f"nz{i}"] = dndz_source[i].tolist()

    lens_info = {
        "nbins_lens": int(dndz_lens.shape[0]),
        "z_array_lens": z_lens.tolist(),
        "z_edges_bins_lens": lens_edges.tolist(),
    }
    for i in range(dndz_lens.shape[0]):
        lens_info[f"nz{i}"] = dndz_lens[i].tolist()

    params = cfg["params"]
    params.setdefault("analysis", {})["nz_source_info_dict"] = source_info
    params.setdefault("analysis", {})["nz_lens_info_dict"] = lens_info
    other_params = params.setdefault("other_params", {})
    if not bool(other_params.get("sampled_des_y3_nuisance", False)):
        other_params["Delta_z_bias_array"] = _delta_z_means(priors)

    metadata = dict(cfg.get("metadata", {}))
    metadata.update(
        {
            "measurement_config": measurement_config,
            "field_metadata": field_meta,
            "des_y3_gaussian_priors": priors,
            "shear_m_bias_means": _shear_m_means(priors),
            "source_z_mid": z_source,
            "source_dndz": dndz_source,
            "lens_z_mid": z_lens,
            "lens_z_edges": z_edges,
            "lens_dndz": dndz_lens,
            "lens_edges": lens_edges,
            "lens_redshift_kind": lens_redshift_kind,
            "lens_nz_provenance": lens_nz_provenance,
            "lens_photoz_diagnostic": photoz_diagnostic,
            "ksz_default_A_v_by_pz": np.asarray(ksz_default_a_v, dtype=np.float64),
            "ksz_sigma_true_gas_over_c_by_pz": np.asarray(ksz_sigma_true, dtype=np.float64),
            "lmax": int(measurement_config["lmax"]),
        }
    )
    cfg["metadata"] = metadata
    cfg["warnings"] = warnings
    return cfg


def _jax_cosmo_distances(cosmo_params: Mapping[str, float], z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax_cosmo.background as bkgrd
    from astropy import constants as const
    from jax_cosmo import Cosmology

    cosmo = Cosmology(
        Omega_c=float(cosmo_params["Om0"]) - float(cosmo_params["Ob0"]),
        Omega_b=float(cosmo_params["Ob0"]),
        h=float(cosmo_params["H0"]) / 100.0,
        sigma8=float(cosmo_params["sigma8"]),
        n_s=float(cosmo_params["ns"]),
        Omega_k=0.0,
        w0=float(cosmo_params["w0"]),
        wa=0.0,
    )
    a = jnp.asarray(1.0 / (1.0 + np.asarray(z, dtype=np.float64)))
    chi = np.asarray(bkgrd.radial_comoving_distance(cosmo, a), dtype=np.float64)
    dchi_dz = np.asarray((const.c.value * 1.0e-3) / bkgrd.H(cosmo, a), dtype=np.float64)
    return chi, dchi_dz


def _hod_bin_indices_for_z(z: np.ndarray, z_edges_bins_lens: np.ndarray) -> np.ndarray:
    """Match Profiles.setup_hod_params bin assignment on an arbitrary z grid.

    Returned indices are 0 for no lens bin and 1..nbin for populated bins.
    In overlapping support regions, the highest matching bin index wins, which
    matches the jnp.max(where(mask, indices + 1, 0)) convention in Profiles.
    """

    z = np.asarray(z, dtype=np.float64)
    edges = np.asarray(z_edges_bins_lens, dtype=np.float64)
    out = np.zeros(z.shape, dtype=np.int64)
    for i, (lo, hi) in enumerate(edges, start=1):
        out[(z > lo) & (z < hi)] = i
    return out


def compute_desi_nbar_comoving(config: Mapping[str, object]) -> dict:
    """Compute per-photo-z-bin HOD nbar(z) from measured angular densities and true DESI n(z)."""

    cfg = copy.deepcopy(dict(config))
    metadata = cfg.get("metadata", {})
    if "lens_z_mid" not in metadata or "lens_dndz" not in metadata:
        cfg = materialize_nz_inputs(cfg)
        metadata = cfg["metadata"]

    analysis = cfg.get("params", {}).get("analysis", {})
    cached_keys = (
        "nbar_per_sr_by_pz",
        "nbar_comoving_by_pz",
        "nbar_comoving_total",
        "nbar_comoving_target_by_pz",
        "nbar_physical_by_pz",
        "chi_lens_hmpc",
        "dchi_dz_lens_hmpc",
    )
    if all(key in metadata for key in cached_keys) and {
        "nbar_gal_comoving_zarray",
        "nbar_gal_comoving_val",
    }.issubset(analysis):
        return cfg

    z = np.asarray(metadata["lens_z_mid"], dtype=np.float64)
    dndz = np.asarray(metadata["lens_dndz"], dtype=np.float64)
    field_meta = metadata["field_metadata"]
    nbar_per_sr = []
    for i in range(1, 5):
        meta_outer = field_meta.get(f"g{i}", {})
        meta = meta_outer.get("metadata", {}) if isinstance(meta_outer, Mapping) else {}
        if "nbar_per_sr" not in meta:
            raise KeyError(f"fields metadata for g{i} lacks nbar_per_sr.")
        nbar_per_sr.append(float(meta["nbar_per_sr"]))
    nbar_per_sr_arr = np.asarray(nbar_per_sr, dtype=np.float64)

    chi, dchi_dz = _jax_cosmo_distances(cfg["params"]["sim_params"]["cosmo"], z)
    denom = chi**2 * dchi_dz
    nbar_by_pz = np.divide(
        nbar_per_sr_arr[:, None] * dndz,
        denom[None, :],
        out=np.zeros_like(dndz, dtype=np.float64),
        where=np.isfinite(denom[None, :]) & (denom[None, :] > 0.0),
    )
    nbar_sum = np.sum(nbar_by_pz, axis=0)
    lens_edges = np.asarray(metadata["lens_edges"], dtype=np.float64)
    hod_bin_indices = _hod_bin_indices_for_z(z, lens_edges)
    floor = float(cfg["raw"].get("nbar", {}).get("floor_comoving_hmpc3", 1.0e-10))
    nbar_sum = np.maximum(np.nan_to_num(nbar_sum, nan=0.0, posinf=0.0, neginf=0.0), floor)
    nbar_by_pz = np.maximum(np.nan_to_num(nbar_by_pz, nan=0.0, posinf=0.0, neginf=0.0), floor)
    nbar_physical_by_pz = nbar_by_pz * (1.0 + z[None, :]) ** 3
    nbar_physical_sum = nbar_sum * (1.0 + z) ** 3

    cfg["params"]["analysis"]["nbar_gal_comoving_zarray"] = z.tolist()
    cfg["params"]["analysis"]["nbar_gal_comoving_val"] = nbar_sum.tolist()
    cfg["metadata"] = dict(metadata)
    cfg["metadata"].update(
        {
            "nbar_per_sr_by_pz": nbar_per_sr_arr,
            "nbar_comoving_by_pz": nbar_by_pz,
            "nbar_comoving_total": nbar_sum,
            "nbar_comoving_sum_by_pz": nbar_sum,
            "nbar_comoving_target": nbar_sum,
            "nbar_comoving_target_by_pz": nbar_by_pz,
            "nbar_physical_by_pz": nbar_physical_by_pz,
            "nbar_physical_target": nbar_physical_sum,
            "nbar_physical_target_by_pz": nbar_physical_by_pz,
            "nbar_hod_bin_indices": hod_bin_indices,
            "nbar_target_convention": "per_photoz_bin_single_godmax_models",
            "chi_lens_hmpc": chi,
            "dchi_dz_lens_hmpc": dchi_dz,
        }
    )
    return cfg


def validation_summary(config: Mapping[str, object]) -> dict:
    metadata = config["metadata"]
    return {
        "measurement_h5": config["paths"]["measurement_h5"],
        "map_h5": config["paths"]["map_h5"],
        "lmax": metadata["lmax"],
        "des_source_nz_shape": np.asarray(metadata["source_dndz"]).shape,
        "desi_lens_nz_shape": np.asarray(metadata["lens_dndz"]).shape,
        "desi_lens_redshift_kind": metadata.get("lens_redshift_kind", ""),
        "desi_lens_nz_provenance": metadata.get("lens_nz_provenance", {}),
        "desi_lens_edges": np.asarray(metadata["lens_edges"]),
        "ksz_default_A_v_by_pz": np.asarray(metadata["ksz_default_A_v_by_pz"]),
        "ksz_sigma_true_gas_over_c_by_pz": np.asarray(metadata["ksz_sigma_true_gas_over_c_by_pz"]),
        "shear_m_bias_means": metadata["shear_m_bias_means"],
        "warnings": config.get("warnings", []),
    }


def ensure_godmax_import_paths(root: Optional[Path] = None) -> None:
    root = root or repo_root()
    for path in (root / "src", root / "notebooks" / "xDESI" / "survey_measure"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _collapse_hod_array_for_pz(values: object, pz_bin: int):
    import jax.numpy as jnp

    arr = jnp.atleast_1d(jnp.asarray(values, dtype=jnp.float64))
    if arr.shape[0] == 1:
        return arr
    if arr.shape[0] <= int(pz_bin):
        raise ValueError(f"HOD array has length {arr.shape[0]}, cannot select pz bin {pz_bin}.")
    return jnp.stack([arr[0], arr[int(pz_bin)]])


def _collapse_hod_arrays_for_single_pz(params: dict, pz_bin: int) -> None:
    sim_params = params["sim_params"]
    for base in HOD_ARRAY_BASE_NAMES:
        key = f"{base}_array"
        if key in sim_params:
            sim_params[key] = _collapse_hod_array_for_pz(sim_params[key], pz_bin)


def config_without_galaxies(config: Mapping[str, object]) -> dict:
    cfg = copy.deepcopy(dict(config))
    cfg = compute_desi_nbar_comoving(cfg)
    cfg["params"] = copy.deepcopy(cfg["params"])
    cfg["params"].setdefault("analysis", {})["model_galaxies"] = False
    return cfg


def config_for_single_desi_pz(config: Mapping[str, object], pz_bin: int) -> dict:
    """Return a GODMAX config whose lone lens bin is one photometric DESI pz bin."""

    pz_bin = int(pz_bin)
    if pz_bin < 1 or pz_bin > 4:
        raise ValueError(f"pz_bin must be 1..4, got {pz_bin}.")
    cfg = copy.deepcopy(dict(config))
    cfg = compute_desi_nbar_comoving(cfg)
    metadata = cfg["metadata"]
    z_lens = np.asarray(metadata["lens_z_mid"], dtype=np.float64)
    dndz = np.asarray(metadata["lens_dndz"], dtype=np.float64)[pz_bin - 1]
    z_edges = np.asarray(metadata["lens_edges"], dtype=np.float64)[pz_bin - 1 : pz_bin]
    nbar = np.asarray(metadata["nbar_comoving_by_pz"], dtype=np.float64)[pz_bin - 1]

    params = copy.deepcopy(cfg["params"])
    params["analysis"]["nz_lens_info_dict"] = {
        "nbins_lens": 1,
        "z_array_lens": z_lens.tolist(),
        "z_edges_bins_lens": z_edges.tolist(),
        "nz0": dndz.tolist(),
    }
    params["analysis"]["nbar_gal_comoving_zarray"] = z_lens.tolist()
    params["analysis"]["nbar_gal_comoving_val"] = nbar.tolist()
    params["analysis"]["model_galaxies"] = True
    params["analysis"]["hod_params_model"] = "perbin"
    params["analysis"]["single_photometric_pz_bin"] = pz_bin
    _collapse_hod_arrays_for_single_pz(params, pz_bin)

    cfg["params"] = params
    cfg["metadata"] = dict(metadata)
    cfg["metadata"].update(
        {
            "single_photometric_pz_bin": pz_bin,
            "single_pz_lens_dndz": dndz,
            "single_pz_nbar_comoving": nbar,
        }
    )
    return cfg


def _params_for_model(config: Mapping[str, object], *, is_cmb_lensing: bool) -> Tuple[dict, dict, dict, dict]:
    import jax.numpy as jnp

    cfg = compute_desi_nbar_comoving(config)
    params = copy.deepcopy(cfg["params"])
    lmax = int(cfg["metadata"]["lmax"])
    params["halo_params"]["ell_array"] = jnp.arange(2, lmax + 1, dtype=jnp.float64)
    params["analysis"]["is_cmb_lensing"] = bool(is_cmb_lensing)
    params["analysis"]["symbolic_pk"] = False
    params["analysis"]["symbolic_hmf"] = False
    if is_cmb_lensing:
        z_source = params["analysis"]["nz_source_info_dict"]["z_array_source"]
        params["analysis"]["nz_source_info_dict"] = {
            "nbins": 1,
            "z_array_source": z_source,
            "nz0": np.ones(len(z_source), dtype=np.float64).tolist(),
        }
        params["other_params"]["Delta_z_bias_array"] = [0.0]
        params["other_params"]["mult_shear_bias_array"] = [0.0]
    return (
        params["sim_params"],
        params["halo_params"],
        params["analysis"],
        params["other_params"],
    )


def build_one_godmax_model(config: Mapping[str, object], *, is_cmb_lensing: bool):
    ensure_godmax_import_paths(Path(config["repo_root"]))
    from base_class import base_class
    from get_Cls import get_Cl
    from get_Pkzs import get_Pkz
    from get_radial_profiles import Profiles

    sim_params, halo_params, analysis, other_params = _params_for_model(config, is_cmb_lensing=is_cmb_lensing)
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = Profiles(sim_params, halo_params, analysis, other_params, base_class_obj=base)
    pkz = get_Pkz(sim_params, halo_params, analysis, other_params, Profiles_obj=profiles)
    cls = get_Cl(sim_params, halo_params, analysis, other_params, Pkz_obj=pkz)
    return cls


def build_godmax_models(config: Mapping[str, object]) -> Dict[str, object]:
    """Build GODMAX theory objects for a photometric-bin-consistent comparison.

    The shared ``wl`` object has ``model_galaxies=False`` and is used for pure
    DES shear and ACT-y x shear spectra.  DESI spectra are assembled from one
    single-lens-bin WL and CMB-lensing object per pz bin, so overlapping
    calibrated true-redshift kernels never force HOD parameters onto true-z
    support intervals.
    """

    cfg = compute_desi_nbar_comoving(config)
    wl_nongal = build_one_godmax_model(config_without_galaxies(cfg), is_cmb_lensing=False)
    gal_wl_by_pz = {}
    gal_cmb_by_pz = {}
    for pz_bin in range(1, 5):
        pz_cfg = config_for_single_desi_pz(cfg, pz_bin)
        gal_wl_by_pz[pz_bin] = build_one_godmax_model(pz_cfg, is_cmb_lensing=False)
        gal_cmb_by_pz[pz_bin] = build_one_godmax_model(pz_cfg, is_cmb_lensing=True)
    return {
        "wl": wl_nongal,
        "cmb": gal_cmb_by_pz[1],
        "gal_wl_by_pz": gal_wl_by_pz,
        "gal_cmb_by_pz": gal_cmb_by_pz,
        "modeling": "single_godmax_object_per_photometric_pz_bin",
    }


def ne0_cm3(cosmo_params: Mapping[str, float], helium_mass_fraction: float = 0.24) -> float:
    from astropy import constants as const

    h = float(cosmo_params["H0"]) / 100.0
    rho_crit_0 = 1.878e-29 * h**2
    return float(rho_crit_0 * float(cosmo_params["Ob0"]) * (1.0 - helium_mass_fraction / 2.0) / const.m_p.to("g").value)


def corrected_gal_tau_cls_zdependent(cls_obj) -> np.ndarray:
    """Apply physical ne0*(1+z)^3 tau normalization inside the Limber integral."""

    z = np.asarray(cls_obj.z_array_for_Cls, dtype=np.float64)
    chi = np.asarray(cls_obj.chi_array_for_Cls, dtype=np.float64)
    dchi_dz = np.asarray(cls_obj.dchi_dz_array_for_Cls, dtype=np.float64)
    wg = np.asarray(cls_obj.Wg_mat, dtype=np.float64)
    wtau = np.asarray(cls_obj.Wtau_array, dtype=np.float64)
    pge = np.asarray(cls_obj.cached_power_spectra[2, 4], dtype=np.float64)
    wtau_corrected = wtau * ne0_cm3(cls_obj.cosmo_params) * (1.0 + z) ** 3
    prefac_tau = wtau_corrected / chi**2
    out = np.zeros((len(cls_obj.ell_array), wg.shape[0]), dtype=np.float64)
    for j in range(wg.shape[0]):
        prefac_g = wg[j] / (dchi_dz * chi**2)
        integrand = pge * prefac_g[None, :] * prefac_tau[None, :] * chi[None, :] ** 2 * dchi_dz[None, :]
        out[:, j] = _trapz(integrand, z, axis=1)
    return out


def corrected_gal_tau_cls_effz(cls_obj, lens_mean_z: Sequence[float]) -> np.ndarray:
    raw = np.asarray(cls_obj.Cl_gal_tau_tot_mat, dtype=np.float64).copy()
    out = np.zeros_like(raw)
    ne0 = ne0_cm3(cls_obj.cosmo_params)
    for j, z_mean in enumerate(lens_mean_z):
        out[:, j] = raw[:, j] * ne0 * (1.0 + float(z_mean)) ** 3
    return out


def lens_mean_z_from_metadata(metadata: Mapping[str, object]) -> np.ndarray:
    field_meta = metadata["field_metadata"]
    vals = []
    for i in range(1, 5):
        meta_outer = field_meta.get(f"g{i}", {})
        meta = meta_outer.get("metadata", {}) if isinstance(meta_outer, Mapping) else {}
        vals.append(float(meta.get("mean_true_z", meta.get("mean_z", np.nan))))
    return np.asarray(vals, dtype=np.float64)


def model_ell_array(models: Mapping[str, object]) -> np.ndarray:
    return np.asarray(models["wl"].ell_array, dtype=np.float64)


def extract_theory_cls_from_models(
    models: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    return_diagnostics: bool = False,
):
    """Extract smooth spectra from the pz-specific GODMAX model bundle."""

    cls_wl = models["wl"]
    gal_wl_by_pz = models["gal_wl_by_pz"]
    gal_cmb_by_pz = models["gal_cmb_by_pz"]
    theory: Dict[str, np.ndarray] = {}
    for i in range(4):
        for j in range(i, 4):
            theory[f"des_shear_EE_tomo{i + 1}_tomo{j + 1}"] = np.asarray(
                cls_wl.Cl_kappa_kappa_tot_mat[:, i, j], dtype=np.float64
            )
    for i in range(4):
        theory[f"act_y_des_shear_E_tomo{i + 1}"] = np.asarray(cls_wl.Cl_kappa_y_tot_mat[:, i], dtype=np.float64)

    tau_zdep_cols = []
    tau_effz_cols = []
    lens_mean_z = lens_mean_z_from_metadata(metadata)
    for pz_bin in range(1, 5):
        pz_wl = gal_wl_by_pz[pz_bin]
        pz_cmb = gal_cmb_by_pz[pz_bin]
        theory[f"desi_g_auto_pz{pz_bin}"] = np.asarray(pz_wl.Cl_gal_gal_tot_mat[:, 0, 0], dtype=np.float64)
        theory[f"desi_g_act_y_pz{pz_bin}"] = np.asarray(pz_wl.Cl_gal_y_tot_mat[:, 0], dtype=np.float64)
        theory[f"desi_g_act_kappa_pz{pz_bin}"] = np.asarray(pz_cmb.Cl_gal_kappa_tot_mat[:, 0, 0], dtype=np.float64)
        for tomo in range(1, 5):
            theory[f"desi_g_des_shear_E_pz{pz_bin}_tomo{tomo}"] = np.asarray(
                pz_wl.Cl_gal_kappa_tot_mat[:, 0, tomo - 1], dtype=np.float64
            )
        tau_zdep = corrected_gal_tau_cls_zdependent(pz_wl)[:, 0]
        tau_effz = corrected_gal_tau_cls_effz(pz_wl, [lens_mean_z[pz_bin - 1]])[:, 0]
        theory[f"desi_g_tau_pz{pz_bin}"] = tau_zdep
        tau_zdep_cols.append(tau_zdep)
        tau_effz_cols.append(tau_effz)

    if not return_diagnostics:
        return theory

    tau_zdep_mat = np.stack(tau_zdep_cols, axis=1)
    tau_effz_mat = np.stack(tau_effz_cols, axis=1)
    diagnostics = {
        "ell": np.asarray(cls_wl.ell_array, dtype=np.float64),
        "lens_mean_z": lens_mean_z,
        "gal_tau_zdependent": tau_zdep_mat,
        "gal_tau_effective_z_approx": tau_effz_mat,
        "gal_tau_effective_over_zdependent": np.divide(
            tau_effz_mat,
            tau_zdep_mat,
            out=np.full_like(tau_zdep_mat, np.nan),
            where=tau_zdep_mat != 0.0,
        ),
        "modeling": models.get("modeling", ""),
    }
    return theory, diagnostics


def extract_theory_cls(
    cls_wl,
    cls_cmb,
    metadata: Mapping[str, object],
    *,
    return_diagnostics: bool = False,
):
    """Extract smooth GODMAX spectra keyed by the measurement theory interface."""

    if isinstance(cls_wl, Mapping) and "gal_wl_by_pz" in cls_wl:
        return extract_theory_cls_from_models(cls_wl, metadata, return_diagnostics=return_diagnostics)

    theory: Dict[str, np.ndarray] = {}
    for i in range(4):
        for j in range(i, 4):
            theory[f"des_shear_EE_tomo{i + 1}_tomo{j + 1}"] = np.asarray(
                cls_wl.Cl_kappa_kappa_tot_mat[:, i, j], dtype=np.float64
            )
    for i in range(4):
        theory[f"act_y_des_shear_E_tomo{i + 1}"] = np.asarray(cls_wl.Cl_kappa_y_tot_mat[:, i], dtype=np.float64)
        theory[f"desi_g_auto_pz{i + 1}"] = np.asarray(cls_wl.Cl_gal_gal_tot_mat[:, i, i], dtype=np.float64)
        theory[f"desi_g_act_y_pz{i + 1}"] = np.asarray(cls_wl.Cl_gal_y_tot_mat[:, i], dtype=np.float64)
        theory[f"desi_g_act_kappa_pz{i + 1}"] = np.asarray(cls_cmb.Cl_gal_kappa_tot_mat[:, i, 0], dtype=np.float64)
    for i in range(4):
        for j in range(4):
            theory[f"desi_g_des_shear_E_pz{i + 1}_tomo{j + 1}"] = np.asarray(
                cls_wl.Cl_gal_kappa_tot_mat[:, i, j], dtype=np.float64
            )

    tau_zdep = corrected_gal_tau_cls_zdependent(cls_wl)
    for i in range(4):
        theory[f"desi_g_tau_pz{i + 1}"] = tau_zdep[:, i]

    if not return_diagnostics:
        return theory

    lens_mean_z = lens_mean_z_from_metadata(metadata)
    tau_effz = corrected_gal_tau_cls_effz(cls_wl, lens_mean_z)
    diagnostics = {
        "ell": np.asarray(cls_wl.ell_array, dtype=np.float64),
        "lens_mean_z": lens_mean_z,
        "gal_tau_zdependent": tau_zdep,
        "gal_tau_effective_z_approx": tau_effz,
        "gal_tau_effective_over_zdependent": np.divide(
            tau_effz,
            tau_zdep,
            out=np.full_like(tau_zdep, np.nan),
            where=tau_zdep != 0.0,
        ),
    }
    return theory, diagnostics


def validate_theory_keys(measurement_path: str | Path, theory_cls: Mapping[str, np.ndarray]) -> List[str]:
    missing = []
    with h5py.File(resolve_repo_path(measurement_path), "r") as h5:
        for raw_name in h5["joint/spectrum_names"][:]:
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            group = h5[f"spectra/{name}"]
            theory_key = str(group.attrs.get("theory_key", name))
            family = str(group.attrs.get("family", ""))
            if name not in theory_cls and not (family == "desi_pi_act_T" and theory_key in theory_cls):
                missing.append(f"{name} or {theory_key}")
    return missing


def theory_data_vector(config: Mapping[str, object], theory_cls: Mapping[str, np.ndarray], ell: np.ndarray):
    ensure_godmax_import_paths(Path(config["repo_root"]))
    from multiprobe_namaster import theory_to_data_vector

    raw = config["raw"].get("theory_to_data_vector", {})
    metadata = config.get("metadata", {})
    other_params = config.get("params", {}).get("other_params", {})
    shear_m_bias = config["metadata"]["shear_m_bias_means"]
    if bool(metadata.get("sampled_des_shear_m_bias_in_model", False)) or bool(
        other_params.get("sampled_des_shear_m_bias_in_model", False)
    ):
        shear_m_bias = None
    return theory_to_data_vector(
        config["paths"]["measurement_h5"],
        theory_cls,
        ell=np.asarray(ell, dtype=np.float64),
        shear_m_bias=shear_m_bias,
        ksz_velocity_correlation=float(raw.get("ksz_velocity_correlation", 0.3)),
        include_default_pixel_windows=bool(raw.get("include_default_pixel_windows", True)),
        include_default_act_beams=bool(raw.get("include_default_act_beams", True)),
        theory_shear_e_is_positive_kappa=bool(raw.get("theory_shear_e_is_positive_kappa", True)),
    )


def comparison_statistics(measurement: MeasurementData, theory_vector: np.ndarray) -> dict:
    if theory_vector.shape != measurement.data_vector.shape:
        raise ValueError(
            f"Theory vector shape {theory_vector.shape} does not match data shape {measurement.data_vector.shape}."
        )
    resid = measurement.data_vector - theory_vector
    try:
        alpha = np.linalg.solve(measurement.covariance, resid)
    except np.linalg.LinAlgError:
        alpha = np.linalg.pinv(measurement.covariance) @ resid
    stats = {
        "full": {
            "chi2": float(resid @ alpha),
            "ndof": int(resid.size),
        },
        "families": {},
    }
    for family in sorted(set(measurement.families.values())):
        idx_chunks = []
        for name, start, stop in zip(measurement.names, measurement.starts, measurement.stops):
            if measurement.families[name] == family:
                idx_chunks.append(np.arange(start, stop, dtype=int))
        if not idx_chunks:
            continue
        idx = np.concatenate(idx_chunks)
        sub_resid = resid[idx]
        sub_cov = measurement.covariance[np.ix_(idx, idx)]
        try:
            sub_alpha = np.linalg.solve(sub_cov, sub_resid)
        except np.linalg.LinAlgError:
            sub_alpha = np.linalg.pinv(sub_cov) @ sub_resid
        stats["families"][family] = {
            "chi2": float(sub_resid @ sub_alpha),
            "ndof": int(idx.size),
        }
    return stats


def dell_factor(ell: np.ndarray) -> np.ndarray:
    ell = np.asarray(ell, dtype=np.float64)
    return ell * (ell + 1.0) / (2.0 * math.pi)


def measurement_ell_slice(measurement: MeasurementData, start: int, stop: int) -> np.ndarray:
    ell = np.asarray(measurement.ell, dtype=np.float64)
    n_band = int(stop) - int(start)
    if ell.size == n_band:
        return ell
    if ell.size == measurement.data_vector.size:
        return ell[int(start) : int(stop)]
    if ell.size >= n_band:
        return ell[:n_band]
    raise ValueError(f"Cannot match ell array of length {ell.size} to band count {n_band}.")


def measurement_ell_edge_slice(
    measurement: MeasurementData,
    start: int,
    stop: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if measurement.ell_left is None or measurement.ell_right is None:
        return None, None
    left = np.asarray(measurement.ell_left, dtype=np.float64)
    right = np.asarray(measurement.ell_right, dtype=np.float64)
    n_band = int(stop) - int(start)
    if left.size == n_band and right.size == n_band:
        return left, right
    if left.size == measurement.data_vector.size and right.size == measurement.data_vector.size:
        return left[int(start) : int(stop)], right[int(start) : int(stop)]
    if left.size >= n_band and right.size >= n_band:
        return left[:n_band], right[:n_band]
    return None, None


def plot_family_comparisons(
    measurement: MeasurementData,
    theory_vector: np.ndarray,
    output_dir: str | Path,
    *,
    pdf_path: Optional[str | Path] = None,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    outputs: List[Path] = []
    family_order = [
        "des_shear_EE",
        "act_y_des_shear_E",
        "desi_g_auto",
        "desi_g_act_y",
        "desi_g_des_shear_E",
        "desi_g_act_kappa",
        "desi_pi_act_T",
    ]
    colors = {
        "des_shear_EE": "#2457a6",
        "act_y_des_shear_E": "#b43c2f",
        "desi_g_auto": "#1e7a49",
        "desi_g_act_y": "#7a4aa0",
        "desi_g_des_shear_E": "#c26a1b",
        "desi_g_act_kappa": "#00838f",
        "desi_pi_act_T": "#5e5147",
    }
    try:
        for family in family_order:
            names = [name for name in measurement.names if measurement.families[name] == family]
            if not names:
                continue
            ncol = min(4, int(math.ceil(math.sqrt(len(names)))))
            nrow = int(math.ceil(len(names) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False, constrained_layout=True)
            for ax, name in zip(axes.flat, names):
                index = measurement.names.index(name)
                start = int(measurement.starts[index])
                stop = int(measurement.stops[index])
                ell = measurement_ell_slice(measurement, start, stop)
                data_cl = measurement.data_vector[start:stop]
                theory_cl = theory_vector[start:stop]
                err = np.sqrt(np.clip(np.diag(measurement.covariance[start:stop, start:stop]), 0.0, np.inf))
                if family == "desi_g_auto":
                    y_data = data_cl
                    y_theory = theory_cl
                    y_err = err
                    ylabel = r"$C_\ell$ signal"
                else:
                    fac = dell_factor(ell)
                    sign = -1.0 if family == "desi_pi_act_T" else 1.0
                    scale = 1.0e3 if family == "desi_pi_act_T" else 1.0
                    y_data = sign * scale * fac * data_cl
                    y_theory = sign * scale * fac * theory_cl
                    y_err = scale * fac * err
                    ylabel = r"$D_\ell$"
                    if family == "desi_pi_act_T":
                        ylabel = r"$-10^3 D_\ell^{\pi T}$"
                ax.errorbar(ell, y_data, yerr=y_err, fmt="o", ms=3.2, lw=1.0, color=colors.get(family, "#333333"), label="measurement")
                ax.plot(ell, y_theory, "-", lw=1.6, color="#111111", label="GODMAX windowed")
                ax.axhline(0.0, color="#777777", lw=0.7, alpha=0.55)
                if family == "desi_g_auto" and np.all(y_data > 0.0) and np.all(y_theory > 0.0):
                    ax.set_yscale("log")
                ax.grid(True, color="#d8dbe2", lw=0.7, alpha=0.75)
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(ylabel)
                ax.set_title(measurement.labels.get(name, name), fontsize=9)
                ax.legend(loc="best", fontsize=7, frameon=False)
            for ax in axes.flat[len(names) :]:
                ax.set_visible(False)
            title = f"{family}: measurement vs GODMAX"
            if family == "desi_pi_act_T":
                title += " (positive kSZ convention)"
            fig.suptitle(title, fontsize=13)
            out = output_dir / f"fast1024_godmax_comparison_{family}.png"
            fig.savefig(out, dpi=180)
            outputs.append(out)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)

    finally:
        if pdf is not None:
            pdf.close()
    return outputs


def plot_family_dell_comparisons(
    measurement: MeasurementData,
    theory_vector: np.ndarray,
    output_dir: str | Path,
    *,
    pdf_path: Optional[str | Path] = None,
    filename_prefix: str = "godmax_dell_comparison",
    ell_max: Optional[float] = None,
    ksz_ylim: Optional[Tuple[float, float]] = None,
    ksz_scale: float = 1.0e3,
    active_band_indices: Optional[Mapping[str, Sequence[int]]] = None,
    total_reduced_chi2: Optional[float] = None,
    chi2_dof: Optional[int] = None,
    xscale: str = "linear",
    xlim: Optional[Tuple[float, float]] = None,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if theory_vector.shape != measurement.data_vector.shape:
        raise ValueError(
            f"Theory vector shape {theory_vector.shape} does not match data shape {measurement.data_vector.shape}."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    outputs: List[Path] = []
    family_order = [
        "des_shear_EE",
        "act_y_des_shear_E",
        "desi_g_auto",
        "desi_g_act_y",
        "desi_g_des_shear_E",
        "desi_g_act_kappa",
        "desi_pi_act_T",
    ]
    colors = {
        "des_shear_EE": "#2457a6",
        "act_y_des_shear_E": "#b43c2f",
        "desi_g_auto": "#1e7a49",
        "desi_g_act_y": "#7a4aa0",
        "desi_g_des_shear_E": "#c26a1b",
        "desi_g_act_kappa": "#00838f",
        "desi_pi_act_T": "#5e5147",
    }
    try:
        for family in family_order:
            names = [name for name in measurement.names if measurement.families[name] == family]
            if not names:
                continue
            ncol = min(4, int(math.ceil(math.sqrt(len(names)))))
            nrow = int(math.ceil(len(names) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False, constrained_layout=True)
            for ax, name in zip(axes.flat, names):
                index = measurement.names.index(name)
                start = int(measurement.starts[index])
                stop = int(measurement.stops[index])
                ell = measurement_ell_slice(measurement, start, stop)
                ell_left, ell_right = measurement_ell_edge_slice(measurement, start, stop)
                data_cl = measurement.data_vector[start:stop]
                theory_cl = theory_vector[start:stop]
                err = np.sqrt(np.clip(np.diag(measurement.covariance[start:stop, start:stop]), 0.0, np.inf))
                local_index = np.arange(ell.size, dtype=int)
                if ell_max is not None:
                    keep = ell <= float(ell_max)
                    local_index = local_index[keep]
                    ell = ell[keep]
                    if ell_left is not None and ell_right is not None:
                        ell_left = ell_left[keep]
                        ell_right = ell_right[keep]
                    data_cl = data_cl[keep]
                    theory_cl = theory_cl[keep]
                    err = err[keep]
                if active_band_indices is not None and name in active_band_indices:
                    active = set(int(x) for x in active_band_indices[name])
                    excluded = np.asarray([int(i) not in active for i in local_index], dtype=bool)
                    inactive_spans = (
                        [(float(lo), float(hi)) for lo, hi in zip(ell_left[excluded], ell_right[excluded])]
                        if np.any(excluded) and ell_left is not None and ell_right is not None
                        else []
                    )
                else:
                    inactive_spans = []
                fac = dell_factor(ell)
                sign = -1.0 if family == "desi_pi_act_T" else 1.0
                scale = float(ksz_scale) if family == "desi_pi_act_T" else 1.0
                y_data = sign * scale * fac * data_cl
                y_theory = sign * scale * fac * theory_cl
                y_err = scale * fac * err
                ylabel = r"$D_\ell$"
                if family == "desi_pi_act_T":
                    ylabel = r"$-D_\ell^{\pi T}$" if np.isclose(scale, 1.0) else r"$-10^3 D_\ell^{\pi T}$"
                if inactive_spans:
                    first_inactive = True
                    for lo, hi in inactive_spans:
                        ax.fill_between(
                            [lo, hi],
                            [0.0, 0.0],
                            [1.0, 1.0],
                            transform=ax.get_xaxis_transform(),
                            color="#b8bcc5",
                            alpha=0.28,
                            lw=0,
                            zorder=0,
                            label="not in likelihood" if first_inactive else None,
                        )
                        first_inactive = False
                ax.errorbar(ell, y_data, yerr=y_err, fmt="o", ms=3.2, lw=1.0, color=colors.get(family, "#333333"), label="data", zorder=2)
                ax.plot(ell, y_theory, "-", lw=1.6, color="#111111", label="bestfit theory", zorder=3)
                ax.axhline(0.0, color="#777777", lw=0.7, alpha=0.55)
                if ell_max is not None:
                    ax.set_xlim(right=float(ell_max))
                if family == "desi_pi_act_T" and ksz_ylim is not None:
                    ax.set_ylim(float(ksz_ylim[0]), float(ksz_ylim[1]))
                ax.grid(True, color="#d8dbe2", lw=0.7, alpha=0.75)
                if str(xscale) != "linear":
                    ax.set_xscale(str(xscale))
                if xlim is not None:
                    ax.set_xlim(float(xlim[0]), float(xlim[1]))
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(ylabel)
                ax.set_title(measurement.labels.get(name, name), fontsize=9)
                ax.legend(loc="best", fontsize=7, frameon=False)
            for ax in axes.flat[len(names) :]:
                ax.set_visible(False)
            title = f"{family}: data vs bestfit theory in D_ell"
            if family == "desi_pi_act_T":
                title += " (positive kSZ convention)"
            if total_reduced_chi2 is not None:
                title += "\n" + rf"best-fit reduced $\chi^2={float(total_reduced_chi2):.2f}$"
                if chi2_dof is not None:
                    title += f" ({int(chi2_dof)} dof)"
            fig.suptitle(title, fontsize=12)
            out = output_dir / f"{filename_prefix}_{family}.png"
            fig.savefig(out, dpi=180)
            outputs.append(out)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return outputs


def plot_family_dell_residual_comparisons(
    measurement: MeasurementData,
    theory_vector: np.ndarray,
    output_dir: str | Path,
    *,
    pdf_path: Optional[str | Path] = None,
    filename_prefix: str = "godmax_dell_residual",
    ell_max: Optional[float] = None,
    ksz_scale: float = 1.0,
    active_band_indices: Optional[Mapping[str, Sequence[int]]] = None,
    total_reduced_chi2: Optional[float] = None,
    chi2_dof: Optional[int] = None,
    xscale: str = "linear",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if theory_vector.shape != measurement.data_vector.shape:
        raise ValueError(
            f"Theory vector shape {theory_vector.shape} does not match data shape {measurement.data_vector.shape}."
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    outputs: List[Path] = []
    family_order = [
        "des_shear_EE",
        "act_y_des_shear_E",
        "desi_g_auto",
        "desi_g_act_y",
        "desi_g_des_shear_E",
        "desi_g_act_kappa",
        "desi_pi_act_T",
    ]
    colors = {
        "des_shear_EE": "#2457a6",
        "act_y_des_shear_E": "#b43c2f",
        "desi_g_auto": "#1e7a49",
        "desi_g_act_y": "#7a4aa0",
        "desi_g_des_shear_E": "#c26a1b",
        "desi_g_act_kappa": "#00838f",
        "desi_pi_act_T": "#5e5147",
    }
    try:
        for family in family_order:
            names = [name for name in measurement.names if measurement.families[name] == family]
            if not names:
                continue
            ncol = min(4, int(math.ceil(math.sqrt(len(names)))))
            nrow = int(math.ceil(len(names) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False, constrained_layout=True)
            for ax, name in zip(axes.flat, names):
                index = measurement.names.index(name)
                start = int(measurement.starts[index])
                stop = int(measurement.stops[index])
                ell = measurement_ell_slice(measurement, start, stop)
                ell_left, ell_right = measurement_ell_edge_slice(measurement, start, stop)
                data_cl = measurement.data_vector[start:stop]
                theory_cl = theory_vector[start:stop]
                err = np.sqrt(np.clip(np.diag(measurement.covariance[start:stop, start:stop]), 0.0, np.inf))
                local_index = np.arange(ell.size, dtype=int)
                if ell_max is not None:
                    keep = ell <= float(ell_max)
                    local_index = local_index[keep]
                    ell = ell[keep]
                    if ell_left is not None and ell_right is not None:
                        ell_left = ell_left[keep]
                        ell_right = ell_right[keep]
                    data_cl = data_cl[keep]
                    theory_cl = theory_cl[keep]
                    err = err[keep]
                if active_band_indices is not None and name in active_band_indices:
                    active = set(int(x) for x in active_band_indices[name])
                    excluded = np.asarray([int(i) not in active for i in local_index], dtype=bool)
                    inactive_spans = (
                        [(float(lo), float(hi)) for lo, hi in zip(ell_left[excluded], ell_right[excluded])]
                        if np.any(excluded) and ell_left is not None and ell_right is not None
                        else []
                    )
                else:
                    inactive_spans = []
                fac = dell_factor(ell)
                sign = -1.0 if family == "desi_pi_act_T" else 1.0
                scale = float(ksz_scale) if family == "desi_pi_act_T" else 1.0
                y_data = sign * scale * fac * data_cl
                y_theory = sign * scale * fac * theory_cl
                y_err = scale * fac * err
                residual = np.full_like(y_data, np.nan, dtype=np.float64)
                valid = np.isfinite(y_data) & np.isfinite(y_theory) & np.isfinite(y_err) & (y_err > 0.0)
                residual[valid] = (y_theory[valid] - y_data[valid]) / y_err[valid]
                if inactive_spans:
                    first_inactive = True
                    for lo, hi in inactive_spans:
                        ax.fill_between(
                            [lo, hi],
                            [0.0, 0.0],
                            [1.0, 1.0],
                            transform=ax.get_xaxis_transform(),
                            color="#b8bcc5",
                            alpha=0.28,
                            lw=0,
                            zorder=0,
                            label="not in likelihood" if first_inactive else None,
                        )
                        first_inactive = False
                ax.axhline(0.0, color="#555555", lw=0.8, alpha=0.75, zorder=1)
                ax.axhline(1.0, color="#4f4f4f", lw=1.6, ls="--", alpha=0.95, zorder=1, label=r"$\pm 1\sigma$")
                ax.axhline(-1.0, color="#4f4f4f", lw=1.6, ls="--", alpha=0.95, zorder=1)
                ax.plot(ell, residual, "o-", ms=3.2, lw=1.1, color=colors.get(family, "#333333"), label=r"$(\mathrm{bestfit}-\mathrm{data})/\sigma$", zorder=3)
                if ell_max is not None:
                    ax.set_xlim(right=float(ell_max))
                if str(xscale) != "linear":
                    ax.set_xscale(str(xscale))
                if xlim is not None:
                    ax.set_xlim(float(xlim[0]), float(xlim[1]))
                if ylim is not None:
                    ax.set_ylim(float(ylim[0]), float(ylim[1]))
                ax.grid(True, color="#d8dbe2", lw=0.7, alpha=0.75)
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(r"$(D_\ell^\mathrm{bf}-D_\ell^\mathrm{data})/\sigma(D_\ell)$")
                ax.set_title(measurement.labels.get(name, name), fontsize=9)
                ax.legend(loc="best", fontsize=7, frameon=False)
            for ax in axes.flat[len(names) :]:
                ax.set_visible(False)
            title = f"{family}: bestfit residuals"
            if family == "desi_pi_act_T":
                title += " (positive kSZ convention)"
            if total_reduced_chi2 is not None:
                title += "\n" + rf"best-fit reduced $\chi^2={float(total_reduced_chi2):.2f}$"
                if chi2_dof is not None:
                    title += f" ({int(chi2_dof)} dof)"
            fig.suptitle(title, fontsize=12)
            out = output_dir / f"{filename_prefix}_{family}.png"
            fig.savefig(out, dpi=180)
            outputs.append(out)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return outputs


def plot_measurement_dell(
    measurement: MeasurementData,
    output_dir: str | Path,
    *,
    pdf_path: Optional[str | Path] = None,
    filename_prefix: str = "measurement_dell",
    ell_max: Optional[float] = None,
    ksz_ylim: Optional[Tuple[float, float]] = None,
    ksz_scale: float = 1.0,
    xscale: str = "linear",
    xlim: Optional[Tuple[float, float]] = None,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = PdfPages(pdf_path) if pdf_path is not None else None
    outputs: List[Path] = []
    family_order = [
        "des_shear_EE",
        "act_y_des_shear_E",
        "desi_g_auto",
        "desi_g_act_y",
        "desi_g_des_shear_E",
        "desi_g_act_kappa",
        "desi_pi_act_T",
    ]
    colors = {
        "des_shear_EE": "#2457a6",
        "act_y_des_shear_E": "#b43c2f",
        "desi_g_auto": "#1e7a49",
        "desi_g_act_y": "#7a4aa0",
        "desi_g_des_shear_E": "#c26a1b",
        "desi_g_act_kappa": "#00838f",
        "desi_pi_act_T": "#5e5147",
    }
    try:
        for family in family_order:
            names = [name for name in measurement.names if measurement.families[name] == family]
            if not names:
                continue
            ncol = min(4, int(math.ceil(math.sqrt(len(names)))))
            nrow = int(math.ceil(len(names) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False, constrained_layout=True)
            for ax, name in zip(axes.flat, names):
                index = measurement.names.index(name)
                start = int(measurement.starts[index])
                stop = int(measurement.stops[index])
                ell = measurement_ell_slice(measurement, start, stop)
                data_cl = measurement.data_vector[start:stop]
                err = np.sqrt(np.clip(np.diag(measurement.covariance[start:stop, start:stop]), 0.0, np.inf))
                if ell_max is not None:
                    keep = ell <= float(ell_max)
                    ell = ell[keep]
                    data_cl = data_cl[keep]
                    err = err[keep]
                fac = dell_factor(ell)
                sign = -1.0 if family == "desi_pi_act_T" else 1.0
                scale = float(ksz_scale) if family == "desi_pi_act_T" else 1.0
                y_data = sign * scale * fac * data_cl
                y_err = scale * fac * err
                ylabel = r"$D_\ell$"
                if family == "desi_pi_act_T":
                    ylabel = r"$-D_\ell^{\pi T}$" if np.isclose(scale, 1.0) else r"$-10^3 D_\ell^{\pi T}$"
                ax.errorbar(ell, y_data, yerr=y_err, fmt="o", ms=3.2, lw=1.0, color=colors.get(family, "#333333"), label="measurement")
                ax.axhline(0.0, color="#777777", lw=0.7, alpha=0.55)
                if ell_max is not None:
                    ax.set_xlim(right=float(ell_max))
                if family == "desi_pi_act_T" and ksz_ylim is not None:
                    ax.set_ylim(float(ksz_ylim[0]), float(ksz_ylim[1]))
                ax.grid(True, color="#d8dbe2", lw=0.7, alpha=0.75)
                if str(xscale) != "linear":
                    ax.set_xscale(str(xscale))
                if xlim is not None:
                    ax.set_xlim(float(xlim[0]), float(xlim[1]))
                ax.set_xlabel(r"$\ell$")
                ax.set_ylabel(ylabel)
                ax.set_title(measurement.labels.get(name, name), fontsize=9)
                ax.legend(loc="best", fontsize=7, frameon=False)
            for ax in axes.flat[len(names) :]:
                ax.set_visible(False)
            title = f"{family}: measurement in D_ell"
            if family == "desi_pi_act_T":
                title += " (positive kSZ convention)"
            fig.suptitle(title, fontsize=13)
            out = output_dir / f"{filename_prefix}_{family}.png"
            fig.savefig(out, dpi=180)
            outputs.append(out)
            if pdf is not None:
                pdf.savefig(fig)
            plt.close(fig)
    finally:
        if pdf is not None:
            pdf.close()
    return outputs


def save_outputs(
    config: Mapping[str, object],
    measurement: MeasurementData,
    theory_vector: np.ndarray,
    theory_names: Sequence[str],
    theory_cls: Mapping[str, np.ndarray],
    ell_theory: np.ndarray,
    stats: Mapping[str, object],
    tau_diagnostics: Optional[Mapping[str, np.ndarray]] = None,
) -> dict:
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "theory_data_vector_fast1024.npz"
    np.savez_compressed(
        npz_path,
        ell_band=measurement.ell,
        data_vector=measurement.data_vector,
        theory_vector=theory_vector,
        covariance=measurement.covariance,
        spectrum_names=np.asarray(measurement.names),
        theory_names=np.asarray(list(theory_names)),
        ell_theory=np.asarray(ell_theory, dtype=np.float64),
        theory_cls_keys=np.asarray(sorted(theory_cls)),
        ksz_default_A_v_by_pz=np.asarray(config["metadata"]["ksz_default_A_v_by_pz"], dtype=np.float64),
    )
    summary_path = output_dir / "comparison_summary_fast1024.json"
    summary = {
        "config_path": config["paths"]["config"],
        "measurement_path": measurement.path,
        "map_path": config["paths"]["map_h5"],
        "npz_path": npz_path,
        "stats": stats,
        "validation": validation_summary(config),
    }
    if tau_diagnostics is not None:
        ratio = np.asarray(tau_diagnostics["gal_tau_effective_over_zdependent"], dtype=np.float64)
        summary["tau_effective_z_over_zdependent_median_by_pz"] = np.nanmedian(ratio, axis=0)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_serializable(summary), handle, indent=2)
    return {"npz": npz_path, "summary": summary_path}
