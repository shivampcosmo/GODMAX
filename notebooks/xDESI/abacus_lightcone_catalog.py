"""Read-only Abacus Backlight halo preprocessing for xDESI map pasting.

The public entry point is ``preprocess_abacus_catalogs``.  It streams the
Abacus ASDF lightcone files, applies the requested redshift and mass cuts, and
writes compact HDF5 catalogs under controlled repo output directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import asdf
import h5py
import numpy as np
import yaml


RHO_CRIT_0_HMSUN_HMPC3 = 2.77536627245708e11
C_KM_S = 299792.458

FIELD_ALIASES = {
    "chi": ("Interpolated_ComovingDist", "InterpolatedComovingDist"),
    "n_interp": ("Interpolated_N", "InterpolatedN"),
    "position": ("Interpolated_x_L2com", "InterpolatedPosition"),
    "velocity": ("Interpolated_v_L2com", "InterpolatedVelocity"),
    "timeslice_index": ("halo_timeslice_index", "HaloIndex", "index_halo"),
}

CATALOG_DTYPES = {
    "ra_deg": np.float32,
    "dec_deg": np.float32,
    "z": np.float32,
    "M_particle_proxy_hMsun": np.float64,
    "log10M200c_hMsun": np.float32,
    "vlos_kms": np.float32,
    "chi_hMpc": np.float32,
    "R200c_hMpc": np.float32,
    "DA_hMpc": np.float32,
    "N_interp": np.float32,
    "snapshot_z": np.float32,
    "source_file_index": np.int32,
    "source_row_index": np.int64,
    "halo_timeslice_index": np.int64,
}
ROW_IDENTITY_FIELDS = (
    "source_file_index",
    "source_row_index",
    "halo_timeslice_index",
)
WORKING_MASS_MODE = "interpolated_particle_proxy_as_m200c"


@dataclass(frozen=True)
class CatalogSpec:
    key: str
    output_name: str
    z_min: float
    z_max: float
    mass_min_hmsun: float
    mass_max_hmsun: Optional[float] = None
    z_min_exclusive: bool = False
    z_max_exclusive: bool = True
    mass_min_inclusive: bool = False
    mass_max_exclusive: bool = True
    cap_center_ra_deg: Optional[float] = None
    cap_center_dec_deg: Optional[float] = None
    cap_radius_deg: Optional[float] = None
    cap_area_deg2: Optional[float] = None
    cap_edge_buffer_deg: float = 0.0
    metadata: Optional[Mapping[str, object]] = None

    @property
    def log10_m_min_hmsun(self) -> float:
        return float(np.log10(self.mass_min_hmsun))

    @property
    def has_angular_cap(self) -> bool:
        return (
            self.cap_center_ra_deg is not None
            and self.cap_center_dec_deg is not None
            and self.cap_radius_deg is not None
        )

    @property
    def catalog_cap_radius_deg(self) -> Optional[float]:
        if self.cap_radius_deg is None:
            return None
        return float(self.cap_radius_deg) + float(self.cap_edge_buffer_deg)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def xdesi_dir() -> Path:
    return Path(__file__).resolve().parent


def read_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(path: Path | str) -> dict:
    config = read_yaml(path)
    output_root = Path(config["project"]["output_root"]).expanduser().resolve()
    ensure_under_xdesi(output_root)
    return config


def ensure_under_xdesi(path: Path) -> None:
    """Prevent accidental writes outside approved repo output directories."""

    roots = (xdesi_dir().resolve(), (repo_root() / "data").resolve())
    resolved = Path(path).expanduser().resolve()
    for root in roots:
        try:
            resolved.relative_to(root)
            return
        except ValueError:
            continue
    allowed = ", ".join(str(root) for root in roots)
    raise ValueError(f"Refusing to write outside approved output roots ({allowed}): {resolved}")


def cap_radius_deg_for_area(area_deg2: float) -> float:
    area_sr = float(area_deg2) * (math.pi / 180.0) ** 2
    if area_sr <= 0.0 or area_sr >= 4.0 * math.pi:
        raise ValueError(f"Invalid angular cap area {area_deg2!r} deg^2.")
    return math.degrees(math.acos(1.0 - area_sr / (2.0 * math.pi)))


def angular_cap_mask(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    center_ra_deg: float,
    center_dec_deg: float,
    radius_deg: float,
) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    ra0 = math.radians(float(center_ra_deg))
    dec0 = math.radians(float(center_dec_deg))
    cosang = (
        np.sin(dec) * math.sin(dec0)
        + np.cos(dec) * math.cos(dec0) * np.cos(ra - ra0)
    )
    return cosang >= math.cos(math.radians(float(radius_deg)))


def catalog_specs_from_config(config: Mapping[str, object], only: Optional[Sequence[str]] = None) -> List[CatalogSpec]:
    specs = []
    only_set = set(only) if only else None
    default_patch = config.get("sky_patch", {})
    for key, raw in config["catalogs"].items():
        if only_set and key not in only_set:
            continue
        patch = {}
        if isinstance(default_patch, Mapping):
            patch.update(default_patch)
        if isinstance(raw.get("sky_patch"), Mapping):
            patch.update(raw["sky_patch"])
        cap_area = patch.get("area_deg2")
        cap_radius = patch.get("radius_deg")
        if cap_radius is None and cap_area is not None:
            cap_radius = cap_radius_deg_for_area(float(cap_area))
        has_mass = "mass_min_hmsun" in raw
        has_logmass = "log10_m_min_hmsun" in raw
        if has_mass == has_logmass:
            raise ValueError(
                f"Catalog {key!r} must define exactly one of mass_min_hmsun or "
                "log10_m_min_hmsun."
            )
        if has_mass and config["abacus"].get("working_mass_mode") != WORKING_MASS_MODE:
            raise ValueError(
                f"Catalog {key!r} with mass_min_hmsun requires explicit "
                f"abacus.working_mass_mode: {WORKING_MASS_MODE}"
            )
        mass_min = (
            float(raw["mass_min_hmsun"])
            if has_mass
            else float(10.0 ** float(raw["log10_m_min_hmsun"]))
        )
        mass_max = raw.get("mass_max_hmsun")
        if not np.isfinite(mass_min) or mass_min <= 0.0:
            raise ValueError(f"Catalog {key!r} has invalid mass_min_hmsun={mass_min!r}.")
        if mass_max is not None and (
            not np.isfinite(float(mass_max)) or float(mass_max) <= mass_min
        ):
            raise ValueError(f"Catalog {key!r} has invalid mass_max_hmsun={mass_max!r}.")
        specs.append(
            CatalogSpec(
                key=key,
                output_name=str(raw["output_name"]),
                z_min=float(raw["z_min"]),
                z_max=float(raw["z_max"]),
                mass_min_hmsun=mass_min,
                mass_max_hmsun=None if mass_max is None else float(mass_max),
                z_min_exclusive=bool(raw.get("z_min_exclusive", False)),
                z_max_exclusive=bool(raw.get("z_max_exclusive", True)),
                mass_min_inclusive=bool(raw.get("mass_min_inclusive", False)),
                mass_max_exclusive=bool(raw.get("mass_max_exclusive", True)),
                cap_center_ra_deg=None if patch.get("center_ra_deg") is None else float(patch["center_ra_deg"]),
                cap_center_dec_deg=None if patch.get("center_dec_deg") is None else float(patch["center_dec_deg"]),
                cap_radius_deg=None if cap_radius is None else float(cap_radius),
                cap_area_deg2=None if cap_area is None else float(cap_area),
                cap_edge_buffer_deg=float(patch.get("edge_buffer_deg", 0.0)),
                metadata=raw.get("metadata", {}),
            )
        )
    if not specs:
        raise ValueError("No catalog specs selected.")
    return specs


def list_snapshot_files(
    input_root: Path,
    max_z_for_dirs: float,
    source_dirs: Optional[Sequence[str]] = None,
) -> List[Tuple[float, Path]]:
    files = []
    if source_dirs:
        requested = [str(name) for name in source_dirs]
        if len(set(requested)) != len(requested):
            raise ValueError(f"Duplicate Abacus source directories requested: {requested}")
        children = [input_root / name for name in requested]
        missing = [str(path) for path in children if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing requested Abacus source directories: {missing}")
    else:
        children = list(input_root.iterdir())
    for child in children:
        if not child.is_dir() or not child.name.startswith("z"):
            continue
        try:
            z_dir = float(child.name[1:])
        except ValueError:
            continue
        if source_dirs or z_dir <= max_z_for_dirs:
            matches = sorted(child.glob("lightcone_halo_info_*.asdf"))
            if source_dirs and not matches:
                raise FileNotFoundError(f"No halo-lightcone ASDF files found in {child}")
            files.extend((z_dir, info) for info in matches)
    return sorted(files, key=lambda item: item[0])


def validate_frozen_source_files(
    input_root: Path,
    snapshot_files: Sequence[Tuple[float, Path]],
    expected_relative_paths: Optional[Sequence[str]],
) -> None:
    """Require the discovered source set to equal the experiment manifest."""

    if not expected_relative_paths:
        return
    expected = [str(Path(path)) for path in expected_relative_paths]
    actual = [str(path.relative_to(input_root)) for _, path in snapshot_files]
    if len(set(expected)) != len(expected):
        raise ValueError(f"Duplicate paths in abacus.source_files: {expected}")
    if actual != expected:
        raise ValueError(
            "Discovered Abacus source files differ from frozen experiment manifest: "
            f"actual={actual}, expected={expected}"
        )


def _get_first(mapping: Mapping[str, object], aliases: Sequence[str]):
    for name in aliases:
        if name in mapping:
            return mapping[name], name
    raise KeyError(f"None of the aliases are present: {aliases}")


def _e2_of_z(z: np.ndarray, omega_m: float, omega_de: float, omega_k: float = 0.0, w0: float = -1.0) -> np.ndarray:
    de = omega_de * np.power(1.0 + z, 3.0 * (1.0 + w0))
    return omega_m * np.power(1.0 + z, 3.0) + omega_k * np.power(1.0 + z, 2.0) + de


def make_chi_to_z_interpolator(header: Mapping[str, object], z_max: float):
    omega_m = float(header.get("Omega_M", header.get("CAMB_Omega_m", 0.3175)))
    omega_k = float(header.get("Omega_K", 0.0))
    omega_de = float(header.get("Omega_DE", 1.0 - omega_m - omega_k))
    w0 = float(header.get("w0", -1.0))
    z_grid = np.linspace(0.0, max(0.1, z_max), 12000, dtype=np.float64)
    e_grid = np.sqrt(_e2_of_z(z_grid, omega_m, omega_de, omega_k, w0))
    dz = np.diff(z_grid)
    inv_e_mid = 0.5 * (1.0 / e_grid[1:] + 1.0 / e_grid[:-1])
    chi = np.empty_like(z_grid)
    chi[0] = 0.0
    chi[1:] = (C_KM_S / 100.0) * np.cumsum(dz * inv_e_mid)

    def chi_to_z(chi_hmpc: np.ndarray) -> np.ndarray:
        return np.interp(chi_hmpc, chi, z_grid).astype(np.float32)

    return chi_to_z


def position_to_radec(pos_hmpc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(pos_hmpc, dtype=np.float64)
    radius = np.linalg.norm(pos, axis=1)
    safe_radius = np.maximum(radius, 1.0e-12)
    ra = np.degrees(np.arctan2(pos[:, 1], pos[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(pos[:, 2] / safe_radius, -1.0, 1.0)))
    return ra.astype(np.float32), dec.astype(np.float32), radius.astype(np.float32)


def radial_velocity_kms(pos_hmpc: np.ndarray, vel_kms: np.ndarray) -> np.ndarray:
    radius = np.linalg.norm(pos_hmpc, axis=1)
    unit = np.divide(pos_hmpc, np.maximum(radius[:, None], 1.0e-12))
    return np.sum(vel_kms * unit, axis=1).astype(np.float32)


def r200c_hmpc(mass_hmsun: np.ndarray, z: np.ndarray, header: Mapping[str, object]) -> np.ndarray:
    omega_m = float(header.get("Omega_M", header.get("CAMB_Omega_m", 0.3175)))
    omega_k = float(header.get("Omega_K", 0.0))
    omega_de = float(header.get("Omega_DE", 1.0 - omega_m - omega_k))
    w0 = float(header.get("w0", -1.0))
    rho_crit = RHO_CRIT_0_HMSUN_HMPC3 * _e2_of_z(z, omega_m, omega_de, omega_k, w0)
    radius = np.power(3.0 * mass_hmsun / (4.0 * math.pi * 200.0 * rho_crit), 1.0 / 3.0)
    return radius.astype(np.float32)


def catalog_selection_mask(
    spec: CatalogSpec,
    z: np.ndarray,
    mass_hmsun: np.ndarray,
    n_interp: np.ndarray,
) -> np.ndarray:
    """Return the exact finite redshift/mass predicate for one catalog."""

    z = np.asarray(z)
    mass_hmsun = np.asarray(mass_hmsun)
    n_interp = np.asarray(n_interp)
    finite = np.isfinite(z) & np.isfinite(mass_hmsun) & np.isfinite(n_interp)
    z_lower = z > spec.z_min if spec.z_min_exclusive else z >= spec.z_min
    z_upper = z < spec.z_max if spec.z_max_exclusive else z <= spec.z_max
    mass_lower = (
        mass_hmsun >= spec.mass_min_hmsun
        if spec.mass_min_inclusive
        else mass_hmsun > spec.mass_min_hmsun
    )
    if spec.mass_max_hmsun is None:
        mass_upper = np.ones_like(finite, dtype=bool)
    elif spec.mass_max_exclusive:
        mass_upper = mass_hmsun < spec.mass_max_hmsun
    else:
        mass_upper = mass_hmsun <= spec.mass_max_hmsun
    return finite & z_lower & z_upper & mass_lower & mass_upper


def selection_predicate_text(spec: CatalogSpec) -> str:
    zlo = ">" if spec.z_min_exclusive else ">="
    zhi = "<" if spec.z_max_exclusive else "<="
    mlo = ">=" if spec.mass_min_inclusive else ">"
    pieces = [
        f"z {zlo} {spec.z_min:.17g}",
        f"z {zhi} {spec.z_max:.17g}",
        f"M_particle_proxy_hMsun {mlo} {spec.mass_min_hmsun:.17g}",
    ]
    if spec.mass_max_hmsun is not None:
        mhi = "<" if spec.mass_max_exclusive else "<="
        pieces.append(
            f"M_particle_proxy_hMsun {mhi} {spec.mass_max_hmsun:.17g}"
        )
    return " & ".join(f"({piece})" for piece in pieces)


def canonical_json_sha256(value: object) -> Tuple[str, str]:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _catalog_attrs(spec: CatalogSpec, config: Mapping[str, object], first_header: Mapping[str, object], files: Sequence[Path]) -> dict:
    h0 = float(first_header["H0"])
    selection_contract = {
        "catalog_key": spec.key,
        "mass_column": "M_particle_proxy_hMsun",
        "mass_equation": "M_particle_proxy_hMsun = N_interp * ParticleMassHMsun",
        "mass_semantics": "interpolated_particle_count_proxy_treated_as_M200c",
        "working_mass_mode": str(
            config["abacus"].get("working_mass_mode", "legacy_implicit")
        ),
        "particle_mass_hmsun": float(first_header["ParticleMassHMsun"]),
        "predicate": selection_predicate_text(spec),
        "source_dirs": list(map(str, config["abacus"].get("source_dirs", ()))),
        "source_files": list(map(str, config["abacus"].get("source_files", ()))),
    }
    selection_contract_json, selection_contract_hash = canonical_json_sha256(
        selection_contract
    )
    attrs = {
        "catalog_format_version": "godmax_abacus_halo_catalog_v2",
        "catalog_key": spec.key,
        "abacus_input_root": str(config["abacus"]["input_root"]),
        "abacus_sim_name": str(config["abacus"].get("sim_name", "")),
        "mass_column": "M_particle_proxy_hMsun",
        "painter_compatibility_mass_column": "M200c_hMsun",
        "mass_unit": "Msun/h",
        "mass_equation": "M_particle_proxy_hMsun = N_interp * ParticleMassHMsun",
        "mass_semantics": "interpolated_particle_count_proxy_treated_as_M200c",
        "mass_definition_status": "provisional_assumption",
        "working_mass_mode": str(
            config["abacus"].get("working_mass_mode", "legacy_implicit")
        ),
        "mass_provenance": "InterpolatedN * ParticleMassHMsun from halo_lightcone",
        "mass_min_hmsun": float(spec.mass_min_hmsun),
        "mass_max_hmsun": (
            np.nan if spec.mass_max_hmsun is None else float(spec.mass_max_hmsun)
        ),
        "log10_m_min_hmsun": float(spec.log10_m_min_hmsun),
        "z_min": float(spec.z_min),
        "z_max": float(spec.z_max),
        "z_min_exclusive": bool(spec.z_min_exclusive),
        "z_max_exclusive": bool(spec.z_max_exclusive),
        "mass_min_inclusive": bool(spec.mass_min_inclusive),
        "mass_max_exclusive": bool(spec.mass_max_exclusive),
        "selection_predicate": selection_predicate_text(spec),
        "selection_contract_json": selection_contract_json,
        "selection_contract_sha256": selection_contract_hash,
        "velocity_unit": "km/s",
        "vlos_definition": "dot(Interpolated velocity, line-of-sight unit vector)",
        "chi_unit": "Mpc/h",
        "R200c_unit": "Mpc/h",
        "position_unit": "Mpc/h",
        "H0": h0,
        "h": h0 / 100.0,
        "Omega_M": float(first_header["Omega_M"]),
        "Omega_b": float(first_header.get("CAMB_Omega_b", np.nan)),
        "sigma8": float(first_header.get("CAMB_sigma8", np.nan)),
        "ns": float(first_header.get("CAMB_ns", np.nan)),
        "w0": float(first_header.get("w0", -1.0)),
        "particle_mass_hmsun": float(first_header["ParticleMassHMsun"]),
        "particle_mass_msun": float(first_header["ParticleMassMsun"]),
        "mass_min_particle_equivalent": float(
            spec.mass_min_hmsun / float(first_header["ParticleMassHMsun"])
        ),
        "source_files_json": json.dumps([str(path) for path in files]),
        "storage_layout": "columnar_shell_ordered_chunked_lzf_shuffle",
        "source_tree_read_only": bool(config["abacus"].get("read_only", False)),
        "source_checksum_algorithm": str(
            config["abacus"].get("source_checksum_algorithm", "none")
        ),
    }
    if spec.has_angular_cap:
        attrs.update(
            {
                "sky_patch_type": "angular_cap",
                "sky_patch_center_ra_deg": float(spec.cap_center_ra_deg),
                "sky_patch_center_dec_deg": float(spec.cap_center_dec_deg),
                "sky_patch_radius_deg": float(spec.cap_radius_deg),
                "sky_patch_area_deg2": float(spec.cap_area_deg2) if spec.cap_area_deg2 is not None else np.nan,
                "sky_patch_edge_buffer_deg": float(spec.cap_edge_buffer_deg),
                "catalog_selection_radius_deg": float(spec.catalog_cap_radius_deg),
            }
        )
    if spec.metadata:
        for key, value in spec.metadata.items():
            attrs[f"metadata_{key}"] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
    return attrs


class H5CatalogWriter:
    def __init__(self, path: Path, attrs: Mapping[str, object], overwrite: bool):
        ensure_under_xdesi(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.tmp_path = path.with_suffix(path.suffix + ".tmp")
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} exists; pass --overwrite to replace it.")
        if self.tmp_path.exists():
            self.tmp_path.unlink()
        self.handle = h5py.File(self.tmp_path, "w")
        self.count = 0
        self.row_identity_hash = hashlib.sha256()
        self.row_content_hash = hashlib.sha256()
        for key, value in attrs.items():
            self.handle.attrs[key] = value
        for name, dtype in CATALOG_DTYPES.items():
            self.handle.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=(min(262144, max(1024, 262144)),),
                dtype=dtype,
                compression="lzf",
                shuffle=True,
            )
        # Existing painters consume M200c_hMsun.  Keep it as a zero-copy HDF5
        # hard link to the explicitly named provisional proxy.
        self.handle["M200c_hMsun"] = self.handle["M_particle_proxy_hMsun"]

    def append(self, data: Mapping[str, np.ndarray]) -> Tuple[int, int]:
        n_new = len(next(iter(data.values()))) if data else 0
        if n_new == 0:
            return self.count, self.count
        start = self.count
        end = start + n_new
        for name, dtype in CATALOG_DTYPES.items():
            ds = self.handle[name]
            ds.resize((end,))
            values = np.asarray(data[name], dtype=dtype)
            ds[start:end] = values
            little_endian = np.asarray(
                values, dtype=np.dtype(dtype).newbyteorder("<")
            )
            self.row_content_hash.update(name.encode("ascii") + b"\0")
            self.row_content_hash.update(np.asarray(n_new, dtype="<i8").tobytes())
            self.row_content_hash.update(little_endian.tobytes(order="C"))
            if name in ROW_IDENTITY_FIELDS:
                self.row_identity_hash.update(name.encode("ascii") + b"\0")
                self.row_identity_hash.update(
                    np.asarray(n_new, dtype="<i8").tobytes()
                )
                self.row_identity_hash.update(little_endian.tobytes(order="C"))
        self.count = end
        self.handle.attrs["n_halos"] = int(self.count)
        return start, end

    def update_attrs(self, attrs: Mapping[str, object]) -> None:
        for key, value in attrs.items():
            self.handle.attrs[key] = value

    def close(self) -> None:
        self.handle.attrs["n_halos"] = int(self.count)
        self.handle.attrs["row_identity_fields_json"] = json.dumps(
            ROW_IDENTITY_FIELDS
        )
        self.handle.attrs["row_identity_sha256"] = self.row_identity_hash.hexdigest()
        self.handle.attrs["catalog_row_content_sha256"] = (
            self.row_content_hash.hexdigest()
        )
        self.handle.close()
        os.replace(self.tmp_path, self.path)

    def abort(self) -> None:
        self.handle.close()
        if self.tmp_path.exists():
            self.tmp_path.unlink()


def _build_selected_data(
    *,
    pos: np.ndarray,
    vel: np.ndarray,
    chi: np.ndarray,
    z: np.ndarray,
    mass: np.ndarray,
    n_interp: np.ndarray,
    timeslice_index: np.ndarray,
    snapshot_z: float,
    source_file_index: int,
    source_row_index: np.ndarray,
    header: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    ra, dec, _ = position_to_radec(pos)
    vlos = radial_velocity_kms(pos.astype(np.float64), vel.astype(np.float64))
    da = chi / (1.0 + z)
    logm = np.log10(mass)
    return {
        "ra_deg": ra,
        "dec_deg": dec,
        "z": z.astype(np.float32),
        "M_particle_proxy_hMsun": mass.astype(np.float64),
        "log10M200c_hMsun": logm.astype(np.float32),
        "vlos_kms": vlos,
        "chi_hMpc": chi.astype(np.float32),
        "R200c_hMpc": r200c_hmpc(mass, z, header),
        "DA_hMpc": da.astype(np.float32),
        "N_interp": n_interp.astype(np.float32),
        "snapshot_z": np.full(len(z), snapshot_z, dtype=np.float32),
        "source_file_index": np.full(len(z), source_file_index, dtype=np.int32),
        "source_row_index": np.asarray(source_row_index, dtype=np.int64),
        "halo_timeslice_index": timeslice_index.astype(np.int64),
    }


def _validate_source_header(
    reference: Mapping[str, object],
    current: Mapping[str, object],
    config: Mapping[str, object],
    path: Path,
) -> None:
    expected_sim = str(config["abacus"].get("sim_name", ""))
    if expected_sim and str(current.get("SimName", "")) != expected_sim:
        raise ValueError(
            f"Source simulation mismatch for {path}: "
            f"{current.get('SimName')!r} != {expected_sim!r}"
        )
    match_keys = config["abacus"].get(
        "header_match_keys",
        (
            "ParticleMassHMsun",
            "H0",
            "Omega_M",
            "Omega_DE",
            "Omega_K",
            "CAMB_Omega_b",
            "CAMB_sigma8",
            "CAMB_ns",
            "w0",
            "hMpc",
        ),
    )
    for key in match_keys:
        if key not in reference or key not in current:
            raise KeyError(f"Required source-header key {key!r} missing in {path}")
        if current[key] != reference[key]:
            raise ValueError(
                f"Source-header mismatch for {key!r} in {path}: "
                f"{current[key]!r} != {reference[key]!r}"
            )


def validate_explicit_source_coverage(
    input_root: Path,
    config: Mapping[str, object],
    specs: Sequence[CatalogSpec],
    reference_header: Mapping[str, object],
) -> dict:
    """Prove an explicit source-dir interval is contiguous and boundary-complete."""

    requested = config["abacus"].get("source_dirs")
    boundary_dirs = config["abacus"].get("source_boundary_null_dirs")
    if not requested or not boundary_dirs:
        return {"status": "not_requested"}
    if len(boundary_dirs) != 2:
        raise ValueError("abacus.source_boundary_null_dirs must contain [lower, upper].")

    available = []
    for child in input_root.iterdir():
        if not child.is_dir() or not child.name.startswith("z"):
            continue
        try:
            z_dir = float(child.name[1:])
        except ValueError:
            continue
        if any(child.glob("lightcone_halo_info_*.asdf")):
            available.append((z_dir, child.name))
    available.sort()
    names = [name for _, name in available]
    lower_name, upper_name = map(str, boundary_dirs)
    if lower_name not in names or upper_name not in names:
        raise FileNotFoundError(
            f"Coverage boundary directories are unavailable: {boundary_dirs}"
        )
    lower_index, upper_index = names.index(lower_name), names.index(upper_name)
    if upper_index <= lower_index + 1:
        raise ValueError(f"Invalid coverage boundary ordering: {boundary_dirs}")
    expected = names[lower_index + 1 : upper_index]
    if list(map(str, requested)) != expected:
        raise ValueError(
            "Explicit source directories are not the complete contiguous interval between "
            f"coverage nulls: requested={list(requested)}, expected={expected}"
        )

    report = {"status": "passed", "expected_source_dirs": expected, "boundaries": []}
    for dirname in (lower_name, upper_name):
        paths = sorted((input_root / dirname).glob("lightcone_halo_info_*.asdf"))
        if not paths:
            raise FileNotFoundError(f"No ASDF files found in coverage boundary {dirname}")
        for path in paths:
            with asdf.open(path, lazy_load=True) as af:
                header = dict(af["header"])
                _validate_source_header(reference_header, header, config, path)
                halo_lc = af["halo_lightcone"]
                chi_arr, _ = _get_first(halo_lc, FIELD_ALIASES["chi"])
                chi = np.asarray(chi_arr[:], dtype=np.float32)
                z = make_chi_to_z_interpolator(
                    header, max(spec.z_max for spec in specs) + 0.2
                )(chi)
                overlaps = {}
                for spec in specs:
                    lower = z > spec.z_min if spec.z_min_exclusive else z >= spec.z_min
                    upper = z < spec.z_max if spec.z_max_exclusive else z <= spec.z_max
                    overlaps[spec.key] = int(
                        np.count_nonzero(np.isfinite(z) & lower & upper)
                    )
                if any(overlaps.values()):
                    raise ValueError(
                        f"Excluded coverage boundary {path} contributes rows: {overlaps}"
                    )
                report["boundaries"].append(
                    {
                        "path": str(path),
                        "z_min": float(np.min(z[np.isfinite(z)])),
                        "z_max": float(np.max(z[np.isfinite(z)])),
                        "overlap_counts": overlaps,
                    }
                )
    return report


def preprocess_abacus_catalogs(
    config_path: Path | str,
    only_catalogs: Optional[Sequence[str]] = None,
    max_files: Optional[int] = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Dict[str, int]:
    config = load_config(config_path)
    specs = catalog_specs_from_config(config, only_catalogs)
    input_root = Path(config["abacus"]["input_root"]).expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if config["abacus"].get("require_read_only", False) and not config["abacus"].get(
        "read_only", False
    ):
        raise ValueError("The experiment requires abacus.read_only: true.")

    max_z = max(spec.z_max for spec in specs) + float(config["abacus"].get("redshift_dir_padding", 0.0))
    snapshot_files = list_snapshot_files(
        input_root,
        max_z,
        source_dirs=config["abacus"].get("source_dirs"),
    )
    validate_frozen_source_files(
        input_root,
        snapshot_files,
        config["abacus"].get("source_files"),
    )
    if max_files is not None:
        snapshot_files = snapshot_files[: int(max_files)]
    if not snapshot_files:
        raise FileNotFoundError(f"No Abacus halo ASDF files found under {input_root}")

    catalog_subdir = str(config["project"].get("catalog_subdir", "abacus_halos"))
    output_root = Path(config["project"]["output_root"]).expanduser().resolve() / catalog_subdir
    ensure_under_xdesi(output_root)
    if not dry_run and not overwrite:
        existing = [
            output_root / spec.output_name
            for spec in specs
            if (output_root / spec.output_name).exists()
        ]
        if existing:
            raise FileExistsError(
                f"Catalog outputs already exist; pass --overwrite to replace: {existing}"
            )
    first_header = None
    with asdf.open(snapshot_files[0][1], lazy_load=True) as af:
        first_header = dict(af["header"])
    _validate_source_header(first_header, first_header, config, snapshot_files[0][1])
    coverage_report = validate_explicit_source_coverage(
        input_root, config, specs, first_header
    )

    source_manifest = []
    checksum_algorithm = str(
        config["abacus"].get("source_checksum_algorithm", "none")
    ).lower()
    if checksum_algorithm not in {"none", "sha256"}:
        raise ValueError(
            "abacus.source_checksum_algorithm must be one of: none, sha256"
        )
    for z_dir, path in snapshot_files:
        stat = path.stat()
        item = {
            "z_dir": float(z_dir),
            "path": str(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        if checksum_algorithm == "sha256" and not dry_run:
            item["sha256"] = file_sha256(path)
        source_manifest.append(item)

    writers: Dict[str, H5CatalogWriter] = {}
    counts = {spec.key: 0 for spec in specs}
    stats = {
        spec.key: {
            "n_source_rows": 0,
            "n_nonfinite": 0,
            "n_failed_z_min": 0,
            "n_failed_z_max": 0,
            "n_failed_mass_min": 0,
            "n_failed_mass_max": 0,
            "n_failed_angular_cap": 0,
            "n_selected": 0,
            "selected_z_min": None,
            "selected_z_max": None,
            "selected_mass_min_hmsun": None,
            "selected_mass_max_hmsun": None,
            "selected_n_interp_min": None,
            "selected_n_interp_max": None,
        }
        for spec in specs
    }
    source_ranges = {spec.key: [] for spec in specs}
    if not dry_run:
        for spec in specs:
            writers[spec.key] = H5CatalogWriter(
                output_root / spec.output_name,
                _catalog_attrs(spec, config, first_header, [path for _, path in snapshot_files]),
                overwrite=overwrite,
            )

    success = False
    try:
        for file_index, (z_dir, path) in enumerate(snapshot_files):
            print(f"[preprocess] {file_index + 1}/{len(snapshot_files)} zdir={z_dir:.3f} {path}")
            with asdf.open(path, lazy_load=True) as af:
                header = dict(af["header"])
                _validate_source_header(first_header, header, config, path)
                halo_lc = af["halo_lightcone"]
                n_arr, _ = _get_first(halo_lc, FIELD_ALIASES["n_interp"])
                n_interp_all = np.asarray(n_arr[:], dtype=np.float32)
                mass_all = n_interp_all.astype(np.float64) * float(header["ParticleMassHMsun"])
                chi_arr, _ = _get_first(halo_lc, FIELD_ALIASES["chi"])
                chi_all = np.asarray(chi_arr[:], dtype=np.float32)
                chi_to_z = make_chi_to_z_interpolator(header, max(spec.z_max for spec in specs) + 0.2)
                z_all = chi_to_z(chi_all)
                spec_masks = {
                    spec.key: catalog_selection_mask(
                        spec, z_all, mass_all, n_interp_all
                    )
                    for spec in specs
                }
                redshift_counts = {}
                for spec in specs:
                    z_lower = (
                        z_all > spec.z_min
                        if spec.z_min_exclusive
                        else z_all >= spec.z_min
                    )
                    z_upper = (
                        z_all < spec.z_max
                        if spec.z_max_exclusive
                        else z_all <= spec.z_max
                    )
                    redshift_counts[spec.key] = int(
                        np.count_nonzero(np.isfinite(z_all) & z_lower & z_upper)
                    )
                union_mask = np.logical_or.reduce(list(spec_masks.values()))
                selected_union = np.flatnonzero(union_mask)

                finite = (
                    np.isfinite(z_all)
                    & np.isfinite(mass_all)
                    & np.isfinite(n_interp_all)
                )
                for spec in specs:
                    item = stats[spec.key]
                    item["n_source_rows"] += int(len(z_all))
                    item["n_nonfinite"] += int(np.count_nonzero(~finite))
                    item["n_failed_z_min"] += int(
                        np.count_nonzero(
                            finite
                            & (
                                z_all <= spec.z_min
                                if spec.z_min_exclusive
                                else z_all < spec.z_min
                            )
                        )
                    )
                    item["n_failed_z_max"] += int(
                        np.count_nonzero(
                            finite
                            & (
                                z_all >= spec.z_max
                                if spec.z_max_exclusive
                                else z_all > spec.z_max
                            )
                        )
                    )
                    item["n_failed_mass_min"] += int(
                        np.count_nonzero(
                            finite
                            & (
                                mass_all < spec.mass_min_hmsun
                                if spec.mass_min_inclusive
                                else mass_all <= spec.mass_min_hmsun
                            )
                        )
                    )
                    if spec.mass_max_hmsun is not None:
                        item["n_failed_mass_max"] += int(
                            np.count_nonzero(
                                finite
                                & (
                                    mass_all >= spec.mass_max_hmsun
                                    if spec.mass_max_exclusive
                                    else mass_all > spec.mass_max_hmsun
                                )
                            )
                        )
                if len(selected_union) == 0:
                    for spec in specs:
                        source_ranges[spec.key].append(
                            {
                                "source_file_index": int(file_index),
                                "z_dir": float(z_dir),
                                "path": str(path),
                                "n_redshift_rows": redshift_counts[spec.key],
                                "n_selected": 0,
                                "output_start": int(counts[spec.key]),
                                "output_stop": int(counts[spec.key]),
                            }
                        )
                    continue

                pos_arr, _ = _get_first(halo_lc, FIELD_ALIASES["position"])
                vel_arr, _ = _get_first(halo_lc, FIELD_ALIASES["velocity"])
                ts_arr, _ = _get_first(halo_lc, FIELD_ALIASES["timeslice_index"])
                pos = np.asarray(pos_arr[selected_union], dtype=np.float32)
                vel = np.asarray(vel_arr[selected_union], dtype=np.float32)
                timeslice_index = np.asarray(ts_arr[selected_union])
                chi = chi_all[selected_union]
                z = z_all[selected_union]
                mass = mass_all[selected_union]
                n_interp = n_interp_all[selected_union]
                snapshot_z = float(header.get("Redshift", z_dir))
                ra_pre = dec_pre = None
                if any(spec.has_angular_cap for spec in specs):
                    ra_pre, dec_pre, _ = position_to_radec(pos)

                for spec in specs:
                    mask = spec_masks[spec.key][selected_union]
                    if spec.has_angular_cap:
                        angular = angular_cap_mask(
                            ra_pre,
                            dec_pre,
                            float(spec.cap_center_ra_deg),
                            float(spec.cap_center_dec_deg),
                            float(spec.catalog_cap_radius_deg),
                        )
                        stats[spec.key]["n_failed_angular_cap"] += int(
                            np.count_nonzero(mask & ~angular)
                        )
                        mask &= angular
                    n_keep = int(np.count_nonzero(mask))
                    counts[spec.key] += n_keep
                    stats[spec.key]["n_selected"] += n_keep
                    output_start = int(counts[spec.key] - n_keep)
                    output_stop = int(counts[spec.key])
                    source_ranges[spec.key].append(
                        {
                            "source_file_index": int(file_index),
                            "z_dir": float(z_dir),
                            "path": str(path),
                            "n_redshift_rows": redshift_counts[spec.key],
                            "n_selected": n_keep,
                            "output_start": output_start,
                            "output_stop": output_stop,
                        }
                    )
                    if n_keep == 0:
                        continue
                    selected_z = z[mask]
                    selected_mass = mass[mask]
                    selected_n = n_interp[mask]
                    extrema = {
                        "selected_z_min": float(np.min(selected_z)),
                        "selected_z_max": float(np.max(selected_z)),
                        "selected_mass_min_hmsun": float(np.min(selected_mass)),
                        "selected_mass_max_hmsun": float(np.max(selected_mass)),
                        "selected_n_interp_min": float(np.min(selected_n)),
                        "selected_n_interp_max": float(np.max(selected_n)),
                    }
                    for name, value in extrema.items():
                        old = stats[spec.key][name]
                        if old is None:
                            stats[spec.key][name] = value
                        elif name.endswith("_min") or "_min_" in name:
                            stats[spec.key][name] = min(old, value)
                        else:
                            stats[spec.key][name] = max(old, value)
                    if dry_run:
                        continue
                    data = _build_selected_data(
                        pos=pos[mask],
                        vel=vel[mask],
                        chi=chi[mask],
                        z=z[mask],
                        mass=mass[mask],
                        n_interp=n_interp[mask],
                        timeslice_index=timeslice_index[mask],
                        snapshot_z=snapshot_z,
                        source_file_index=file_index,
                        source_row_index=selected_union[mask],
                        header=header,
                    )
                    writers[spec.key].append(data)
                    print(f"  [{spec.key}] appended {n_keep:,} halos; total {writers[spec.key].count:,}")
        for item in source_manifest:
            final_stat = Path(item["path"]).stat()
            if (
                int(final_stat.st_size) != item["size_bytes"]
                or int(final_stat.st_mtime_ns) != item["mtime_ns"]
            ):
                raise RuntimeError(
                    f"Abacus source changed during catalog build: {item['path']}"
                )
        success = True
    finally:
        if not dry_run:
            for key, writer in writers.items():
                if success:
                    source_manifest_json, source_manifest_hash = canonical_json_sha256(
                        source_manifest
                    )
                    content_manifest = [
                        {
                            "relative_path": str(
                                Path(item["path"]).relative_to(input_root)
                            ),
                            "sha256": item["sha256"],
                            "size_bytes": item["size_bytes"],
                        }
                        for item in source_manifest
                        if "sha256" in item
                    ]
                    content_manifest_json = ""
                    content_manifest_hash = ""
                    if len(content_manifest) == len(source_manifest):
                        content_manifest_json, content_manifest_hash = (
                            canonical_json_sha256(content_manifest)
                        )
                    writer.update_attrs(
                        {
                            "selection_statistics_json": json.dumps(stats[key], sort_keys=True),
                            "source_row_ranges_json": json.dumps(source_ranges[key], sort_keys=True),
                            "source_manifest_json": source_manifest_json,
                            "source_observation_manifest_sha256": source_manifest_hash,
                            "source_content_manifest_json": content_manifest_json,
                            "source_content_manifest_sha256": content_manifest_hash,
                            "source_coverage_report_json": json.dumps(coverage_report, sort_keys=True),
                        }
                    )
                    writer.close()
                else:
                    writer.abort()

    for key, count in counts.items():
        print(f"[preprocess] {key}: selected {count:,} halos")
    return counts


def validate_catalog_file(path: Path | str) -> Dict[str, float]:
    path = Path(path)
    with h5py.File(path, "r") as handle:
        n = int(handle.attrs.get("n_halos", len(handle["z"])))
        out = {
            "n_halos": n,
            "z_min": float(np.min(handle["z"][:])) if n else np.nan,
            "z_max": float(np.max(handle["z"][:])) if n else np.nan,
            "log10M_min": float(np.min(handle["log10M200c_hMsun"][:])) if n else np.nan,
            "log10M_max": float(np.max(handle["log10M200c_hMsun"][:])) if n else np.nan,
            "nonpositive_DA": int(np.count_nonzero(handle["DA_hMpc"][:] <= 0.0)) if n else 0,
        }
    return out


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(xdesi_dir() / "abacus_pasting_config.yaml"))
    parser.add_argument("--catalog", action="append", help="Catalog key to build. Can be repeated.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of redshift files, useful for dry-runs.")
    parser.add_argument("--dry-run", action="store_true", help="Read and count selected halos without writing outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing xDESI output catalogs.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    preprocess_abacus_catalogs(
        args.config,
        only_catalogs=args.catalog,
        max_files=args.max_files,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
