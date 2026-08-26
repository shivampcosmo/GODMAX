"""Backlight catalog metadata and mass-unit helpers for SBI validation."""

from __future__ import annotations

import pathlib
import re
from typing import Dict, Mapping, Tuple

import h5py
import numpy as np


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
DEFAULT_HALO_CATALOG = REPO_ROOT / "data" / "backlight" / "halo_catalog_Mlim_1e13_zlim_0.4_0.6.h5"
DEFAULT_BACKLIGHT_LIGHTCONE_DIR = pathlib.Path(
    "/mnt/ceph/users/backlight/AbacusBacklight_base_c0000_ph000/lightcone_halos"
)


def load_halo_catalog_h5(path: pathlib.Path | str = DEFAULT_HALO_CATALOG) -> Dict[str, np.ndarray]:
    """Load the local Backlight HDF5 halo catalog."""

    path = pathlib.Path(path)
    with h5py.File(path, "r") as handle:
        catalog = {key: handle[key][()] for key in ("ra", "dec", "z", "M200c", "vlos")}
        catalog["hdf_attrs"] = dict(handle.attrs)
    return catalog


def _extract_z_from_path(path: pathlib.Path) -> float | None:
    match = re.search(r"z(\d+\.\d+)", str(path))
    return float(match.group(1)) if match else None


def find_representative_backlight_asdf(
    catalog: Mapping[str, np.ndarray],
    lightcone_dir: pathlib.Path | str = DEFAULT_BACKLIGHT_LIGHTCONE_DIR,
) -> pathlib.Path | None:
    """Find the source ASDF slice closest to the catalog median redshift."""

    lightcone_dir = pathlib.Path(lightcone_dir)
    if not lightcone_dir.exists():
        return None

    z_target = float(np.nanmedian(np.asarray(catalog["z"], dtype=float)))
    candidates = []
    for path in lightcone_dir.glob("z*/lightcone_halo_info_000.asdf"):
        z_val = _extract_z_from_path(path)
        if z_val is not None:
            candidates.append((abs(z_val - z_target), path))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def load_backlight_source_metadata(
    catalog: Mapping[str, np.ndarray],
    lightcone_dir: pathlib.Path | str = DEFAULT_BACKLIGHT_LIGHTCONE_DIR,
) -> Dict[str, object]:
    """Read source cosmology and mass-unit metadata from a representative ASDF."""

    path = find_representative_backlight_asdf(catalog, lightcone_dir=lightcone_dir)
    if path is None:
        return {
            "source_asdf": None,
            "status": "Backlight ASDF directory unavailable; using catalog-native masses.",
            "raw_mass_unit": "unknown",
            "theory_mass_factor": 1.0,
            "cosmo_overrides": {},
        }

    try:
        import asdf

        with asdf.open(path, lazy_load=True) as af:
            header = af["header"]
            h = float(header["H0"]) / 100.0
            cosmo_overrides = {
                "H0": float(header["H0"]),
                "Om0": float(header["Omega_M"]),
                "Ob0": float(header["CAMB_Omega_b"]),
                "sigma8": float(header["CAMB_sigma8"]),
                "ns": float(header["CAMB_ns"]),
                "w0": float(header.get("w0", -1.0)),
            }
            return {
                "source_asdf": str(path),
                "status": "Backlight ASDF header loaded.",
                "raw_mass_unit": "physical_Msun_from_InterpolatedN_times_ParticleMassMsun",
                "theory_mass_unit": "Msun_over_h_particle_count_proxy",
                "theory_mass_factor": h,
                "particle_mass_msun": float(header["ParticleMassMsun"]),
                "particle_mass_hmsun": float(header["ParticleMassHMsun"]),
                "h": h,
                "cosmo_overrides": cosmo_overrides,
                "mass_definition_warning": (
                    "The saved catalog column is a particle-count mass proxy, not a recovered SO M200c. "
                    "The comparison now handles its h units consistently, but a true SO-M200c comparison "
                    "requires a catalog with HaloIndex/SO mass information."
                ),
            }
    except Exception as exc:  # pragma: no cover - depends on local ASDF install.
        return {
            "source_asdf": str(path),
            "status": f"Failed to read Backlight ASDF header: {exc}",
            "raw_mass_unit": "unknown",
            "theory_mass_factor": 1.0,
            "cosmo_overrides": {},
        }


def prepare_catalog_for_theory(
    catalog: Mapping[str, np.ndarray],
    source_metadata: Mapping[str, object],
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Return a catalog whose ``M200c`` column is in GODMAX theory mass units."""

    mass_raw = np.asarray(catalog["M200c"], dtype=float)
    factor = float(source_metadata.get("theory_mass_factor", 1.0))
    mass_theory = mass_raw * factor
    out = {
        key: np.asarray(value)
        for key, value in catalog.items()
        if isinstance(value, np.ndarray)
    }
    out["M200c_raw"] = mass_raw
    out["M200c"] = mass_theory
    mass_metadata = {
        "raw_mass_column": "M200c",
        "raw_mass_min": float(np.nanmin(mass_raw)),
        "raw_mass_max": float(np.nanmax(mass_raw)),
        "theory_mass_min": float(np.nanmin(mass_theory)),
        "theory_mass_max": float(np.nanmax(mass_theory)),
        "theory_mass_factor_applied": factor,
        "theory_mass_unit": source_metadata.get("theory_mass_unit", "catalog_native"),
    }
    return out, mass_metadata


def mass_grid_log10_min_for_catalog(mass_min: float) -> float:
    """Choose a conservative lower theory grid edge below the converted catalog minimum."""

    return max(10.0, float(np.floor(20.0 * np.log10(mass_min)) / 20.0) - 0.05)


def backlight_validation_settings(
    halo_catalog: pathlib.Path | str = DEFAULT_HALO_CATALOG,
    lightcone_dir: pathlib.Path | str = DEFAULT_BACKLIGHT_LIGHTCONE_DIR,
) -> Dict[str, object]:
    """Return cosmology, mass cut, and halo-grid settings matching the Backlight catalog."""

    raw_catalog = load_halo_catalog_h5(halo_catalog)
    source_metadata = load_backlight_source_metadata(raw_catalog, lightcone_dir=lightcone_dir)
    theory_catalog, mass_metadata = prepare_catalog_for_theory(raw_catalog, source_metadata)
    hod_mass_cut = float(np.nanmin(theory_catalog["M200c"]))
    halo_lg10_min = mass_grid_log10_min_for_catalog(hod_mass_cut)
    sim_param_overrides = {
        f"cosmo.{key}": float(value)
        for key, value in source_metadata.get("cosmo_overrides", {}).items()
    }
    halo_param_overrides = {"lg10_Mmin": halo_lg10_min}
    return {
        "halo_catalog": str(halo_catalog),
        "source_metadata": source_metadata,
        "mass_metadata": mass_metadata,
        "hod_mass_cut": hod_mass_cut,
        "halo_param_overrides": halo_param_overrides,
        "sim_param_overrides": sim_param_overrides,
        "raw_catalog_count": int(len(raw_catalog["z"])),
    }
