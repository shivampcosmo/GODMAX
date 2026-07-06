"""Read-only Abacus Backlight halo preprocessing for xDESI map pasting.

The public entry point is ``preprocess_abacus_catalogs``.  It streams the
Abacus ASDF lightcone files, applies the requested redshift and mass cuts, and
writes compact HDF5 catalogs under controlled repo output directories.
"""

from __future__ import annotations

import argparse
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
    "M200c_hMsun": np.float64,
    "log10M200c_hMsun": np.float32,
    "vlos_kms": np.float32,
    "chi_hMpc": np.float32,
    "R200c_hMpc": np.float32,
    "DA_hMpc": np.float32,
    "N_interp": np.float32,
    "snapshot_z": np.float32,
    "source_file_index": np.int32,
    "halo_timeslice_index": np.int64,
}


@dataclass(frozen=True)
class CatalogSpec:
    key: str
    output_name: str
    z_min: float
    z_max: float
    log10_m_min_hmsun: float
    cap_center_ra_deg: Optional[float] = None
    cap_center_dec_deg: Optional[float] = None
    cap_radius_deg: Optional[float] = None
    cap_area_deg2: Optional[float] = None
    cap_edge_buffer_deg: float = 0.0
    metadata: Optional[Mapping[str, object]] = None

    @property
    def mass_min_hmsun(self) -> float:
        return float(10.0 ** self.log10_m_min_hmsun)

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
        specs.append(
            CatalogSpec(
                key=key,
                output_name=str(raw["output_name"]),
                z_min=float(raw["z_min"]),
                z_max=float(raw["z_max"]),
                log10_m_min_hmsun=float(raw["log10_m_min_hmsun"]),
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


def list_snapshot_files(input_root: Path, max_z_for_dirs: float) -> List[Tuple[float, Path]]:
    files = []
    for child in input_root.iterdir():
        if not child.is_dir() or not child.name.startswith("z"):
            continue
        try:
            z_dir = float(child.name[1:])
        except ValueError:
            continue
        if z_dir <= max_z_for_dirs:
            info = child / "lightcone_halo_info_000.asdf"
            if info.exists():
                files.append((z_dir, info))
    return sorted(files, key=lambda item: item[0])


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


def _catalog_attrs(spec: CatalogSpec, config: Mapping[str, object], first_header: Mapping[str, object], files: Sequence[Path]) -> dict:
    h0 = float(first_header["H0"])
    attrs = {
        "catalog_key": spec.key,
        "abacus_input_root": str(config["abacus"]["input_root"]),
        "abacus_sim_name": str(config["abacus"].get("sim_name", "")),
        "mass_column": "M200c_hMsun",
        "mass_unit": "Msun/h",
        "mass_provenance": "Interpolated_N * ParticleMassHMsun from halo_lightcone",
        "log10_m_min_hmsun": float(spec.log10_m_min_hmsun),
        "z_min": float(spec.z_min),
        "z_max": float(spec.z_max),
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
        "source_files_json": json.dumps([str(path) for path in files]),
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
            )

    def append(self, data: Mapping[str, np.ndarray]) -> None:
        n_new = len(next(iter(data.values()))) if data else 0
        if n_new == 0:
            return
        start = self.count
        end = start + n_new
        for name, dtype in CATALOG_DTYPES.items():
            ds = self.handle[name]
            ds.resize((end,))
            ds[start:end] = np.asarray(data[name], dtype=dtype)
        self.count = end
        self.handle.attrs["n_halos"] = int(self.count)

    def close(self) -> None:
        self.handle.attrs["n_halos"] = int(self.count)
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
        "M200c_hMsun": mass.astype(np.float64),
        "log10M200c_hMsun": logm.astype(np.float32),
        "vlos_kms": vlos,
        "chi_hMpc": chi.astype(np.float32),
        "R200c_hMpc": r200c_hmpc(mass, z, header),
        "DA_hMpc": da.astype(np.float32),
        "N_interp": n_interp.astype(np.float32),
        "snapshot_z": np.full(len(z), snapshot_z, dtype=np.float32),
        "source_file_index": np.full(len(z), source_file_index, dtype=np.int32),
        "halo_timeslice_index": timeslice_index.astype(np.int64),
    }


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

    max_z = max(spec.z_max for spec in specs) + float(config["abacus"].get("redshift_dir_padding", 0.0))
    snapshot_files = list_snapshot_files(input_root, max_z)
    if max_files is not None:
        snapshot_files = snapshot_files[: int(max_files)]
    if not snapshot_files:
        raise FileNotFoundError(f"No Abacus halo ASDF files found under {input_root}")

    catalog_subdir = str(config["project"].get("catalog_subdir", "abacus_halos"))
    output_root = Path(config["project"]["output_root"]).expanduser().resolve() / catalog_subdir
    ensure_under_xdesi(output_root)
    first_header = None
    with asdf.open(snapshot_files[0][1], lazy_load=True) as af:
        first_header = dict(af["header"])

    writers: Dict[str, H5CatalogWriter] = {}
    counts = {spec.key: 0 for spec in specs}
    if not dry_run:
        for spec in specs:
            writers[spec.key] = H5CatalogWriter(
                output_root / spec.output_name,
                _catalog_attrs(spec, config, first_header, [path for _, path in snapshot_files]),
                overwrite=overwrite,
            )

    success = False
    try:
        global_mass_min = min(spec.mass_min_hmsun for spec in specs)
        for file_index, (z_dir, path) in enumerate(snapshot_files):
            print(f"[preprocess] {file_index + 1}/{len(snapshot_files)} zdir={z_dir:.3f} {path}")
            with asdf.open(path, lazy_load=True) as af:
                header = af["header"]
                halo_lc = af["halo_lightcone"]
                n_arr, n_name = _get_first(halo_lc, FIELD_ALIASES["n_interp"])
                n_interp_all = np.asarray(n_arr[:], dtype=np.float32)
                mass_all = n_interp_all.astype(np.float64) * float(header["ParticleMassHMsun"])
                pre_idx = np.flatnonzero(mass_all > global_mass_min)
                if len(pre_idx) == 0:
                    continue

                chi_arr, _ = _get_first(halo_lc, FIELD_ALIASES["chi"])
                pos_arr, _ = _get_first(halo_lc, FIELD_ALIASES["position"])
                vel_arr, _ = _get_first(halo_lc, FIELD_ALIASES["velocity"])
                ts_arr, _ = _get_first(halo_lc, FIELD_ALIASES["timeslice_index"])

                chi = np.asarray(chi_arr[pre_idx], dtype=np.float32)
                chi_to_z = make_chi_to_z_interpolator(header, max(spec.z_max for spec in specs) + 0.2)
                z = chi_to_z(chi)
                pos = np.asarray(pos_arr[pre_idx], dtype=np.float32)
                vel = np.asarray(vel_arr[pre_idx], dtype=np.float32)
                timeslice_index = np.asarray(ts_arr[pre_idx])
                mass = mass_all[pre_idx]
                n_interp = n_interp_all[pre_idx]
                snapshot_z = float(header.get("Redshift", z_dir))
                ra_pre = dec_pre = None
                if any(spec.has_angular_cap for spec in specs):
                    ra_pre, dec_pre, _ = position_to_radec(pos)

                for spec in specs:
                    mask = (
                        (z >= spec.z_min)
                        & (z < spec.z_max)
                        & (mass > spec.mass_min_hmsun)
                        & np.isfinite(z)
                        & np.isfinite(mass)
                    )
                    if spec.has_angular_cap:
                        mask &= angular_cap_mask(
                            ra_pre,
                            dec_pre,
                            float(spec.cap_center_ra_deg),
                            float(spec.cap_center_dec_deg),
                            float(spec.catalog_cap_radius_deg),
                        )
                    n_keep = int(np.count_nonzero(mask))
                    counts[spec.key] += n_keep
                    if n_keep == 0 or dry_run:
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
                        header=header,
                    )
                    writers[spec.key].append(data)
                    print(f"  [{spec.key}] appended {n_keep:,} halos; total {writers[spec.key].count:,}")
        success = True
    finally:
        if not dry_run:
            for writer in writers.values():
                if success:
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
