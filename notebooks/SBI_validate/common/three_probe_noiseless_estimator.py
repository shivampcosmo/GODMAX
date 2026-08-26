"""Fail-closed scalar estimator helpers for the noiseless three-probe paste test.

This module deliberately contains no GODMAX physics construction.  It turns the
frozen pasted HDF5 product into raw NaMaster bandpowers and applies an explicitly
supplied dense theory vector through the saved estimator windows.  In particular,
the measured bandpowers never depend on a theory curve or transfer choice.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Mapping, Sequence

import h5py
import healpy as hp
import numpy as np


NSIDE = 512
LMAX = 1535
TARGET_FSKY2 = 0.4
MASK_FSKY2_ATOL = 5.0e-13
NZ_NORMALIZATION_ATOL = 1.0e-6

# These are the exact integer supports pre-registered in the experiment plan.
# NmtBin treats the upper edge as exclusive.
NATIVE_12_EDGES = np.asarray(
    [80, 101, 127, 160, 201, 253, 319, 401, 505, 636, 801, 1008, 1268],
    dtype=np.int64,
)
SPECTRUM_FIELDS = {
    "gg": ("g", "g"),
    "gy": ("g", "y"),
    "gtau": ("g", "tau"),
    "gkappa": ("g", "kappa"),
}
MAP_DATASETS = {
    "y": "maps/map_ymap",
    "tau": "maps/map_tau",
    "kappa": "maps/map_kappa_cmb",
}
GALAXY_COLUMNS = (
    "ra_deg",
    "dec_deg",
    "z",
    "host_M200c_hMsun",
    "is_central",
    "valid",
    "host_vlos_kms",
)


def native_12_band_edges() -> np.ndarray:
    """Return a copy of the frozen, non-overlapping native-band edges."""

    return NATIVE_12_EDGES.copy()


def make_native_12_bins(nmt_module, lmax: int = LMAX):
    """Build the 12 frozen bands while advertising the full field bandlimit.

    NaMaster requires ``bins.lmax == field.lmax``.  Multipoles above the last
    frozen band are assigned ``bpw=-1`` (ignored), but remain available to the
    exact coupling matrix and bandpower windows up to the field bandlimit.
    """

    ell = np.arange(int(lmax) + 1, dtype=np.int32)
    bpws = np.full(ell.shape, -1, dtype=np.int32)
    for band, (left, right) in enumerate(zip(NATIVE_12_EDGES[:-1], NATIVE_12_EDGES[1:])):
        bpws[(ell >= int(left)) & (ell < int(right))] = int(band)
    return nmt_module.NmtBin(ells=ell, bpws=bpws, lmax=int(lmax))


def sha256_array(value: np.ndarray) -> str:
    """Hash an array using the exact convention used by the paste combiner."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def sha256_file(path: pathlib.Path | str, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_h5_dataset(dataset: h5py.Dataset, chunk_rows: int = 1_000_000) -> str:
    """Hash an HDF5 dataset without materializing a large galaxy catalog."""

    digest = hashlib.sha256()
    digest.update(np.dtype(dataset.dtype).str.encode("ascii"))
    digest.update(np.asarray(dataset.shape, dtype=np.int64).tobytes())
    if dataset.ndim == 0:
        digest.update(np.ascontiguousarray(dataset[()]).tobytes())
        return digest.hexdigest()
    for start in range(0, dataset.shape[0], int(chunk_rows)):
        stop = min(start + int(chunk_rows), dataset.shape[0])
        digest.update(np.ascontiguousarray(dataset[start:stop]).tobytes())
    return digest.hexdigest()


def _axis_separation_rad(
    nside: int, center_lon_deg: float, center_lat_deg: float
) -> np.ndarray:
    pix = np.arange(hp.nside2npix(int(nside)), dtype=np.int64)
    x, y, z = hp.pix2vec(int(nside), pix)
    lon = np.radians(float(center_lon_deg))
    lat = np.radians(float(center_lat_deg))
    center = np.asarray(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )
    return np.arccos(np.clip(x * center[0] + y * center[1] + z * center[2], -1.0, 1.0))


def _analytic_c2_cap(
    separation_rad: np.ndarray, radius_rad: float, apodization_rad: float
) -> np.ndarray:
    """Evaluate the NaMaster-documented C2 taper about an ideal cap edge."""

    separation = np.asarray(separation_rad, dtype=np.float64)
    distance_inside = float(radius_rad) - separation
    mask = np.zeros_like(separation)
    inside = distance_inside > 0.0
    deep = inside & (distance_inside >= float(apodization_rad))
    mask[deep] = 1.0
    edge = inside & ~deep
    denominator = 1.0 - np.cos(float(apodization_rad))
    if denominator <= 0.0:
        raise ValueError("C2 apodization width must be positive")
    x = np.sqrt(
        np.clip((1.0 - np.cos(distance_inside[edge])) / denominator, 0.0, 1.0)
    )
    mask[edge] = 0.5 * (1.0 - np.cos(np.pi * x))
    return mask


def solve_common_c2_cap(
    nside: int = NSIDE,
    target_fsky2: float = TARGET_FSKY2,
    apodization_deg: float = 1.0,
    center_lon_deg: float = 0.0,
    center_lat_deg: float = 90.0,
    atol: float = MASK_FSKY2_ATOL,
    max_iterations: int = 80,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve an axisymmetric analytic C2 cap with the requested ``mean(mask**2)``.

    The continuous cap radius, rather than an amplitude renormalization, is
    varied.  This keeps the unmasked cap interior exactly one and the exterior
    exactly zero.
    """

    target = float(target_fsky2)
    if not 0.0 < target < 1.0:
        raise ValueError("target_fsky2 must lie strictly between zero and one")
    apo = np.radians(float(apodization_deg))
    separation = _axis_separation_rad(nside, center_lon_deg, center_lat_deg)
    lower = apo
    upper = np.pi
    mask = None
    for _ in range(int(max_iterations)):
        radius = 0.5 * (lower + upper)
        trial = _analytic_c2_cap(separation, radius, apo)
        fsky2 = float(np.mean(trial * trial, dtype=np.float64))
        mask = trial
        if abs(fsky2 - target) <= float(atol):
            break
        if fsky2 < target:
            lower = radius
        else:
            upper = radius
    if mask is None:
        raise RuntimeError("C2 cap solver did not execute")
    fsky2 = float(np.mean(mask * mask, dtype=np.float64))
    if abs(fsky2 - target) > float(atol):
        raise RuntimeError(
            f"C2 cap solve missed mean(mask**2)={target}: got {fsky2}"
        )
    metadata = {
        "implementation": "analytic_axisymmetric_namaster_documented_c2_v1",
        "nside": int(nside),
        "ordering": "RING",
        "coordinate_frame": "ICRS_lonlat_degrees",
        "center_lon_deg": float(center_lon_deg),
        "center_lat_deg": float(center_lat_deg),
        "cap_radius_deg": float(np.degrees(radius)),
        "apodization_type": "C2",
        "apodization_deg": float(apodization_deg),
        "mean_mask": float(np.mean(mask, dtype=np.float64)),
        "mean_mask2": fsky2,
        "mean_mask4": float(np.mean(mask**4, dtype=np.float64)),
        "nonzero_fraction": float(np.mean(mask > 0.0)),
        "mask_sha256": sha256_array(mask),
    }
    return mask, metadata


def subtract_weighted_mean(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Subtract the mean weighted by the exact analysis mask."""

    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(mask, dtype=np.float64)
    if array.shape != weight.shape or array.ndim != 1:
        raise ValueError("values and mask must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(array)) or not np.all(np.isfinite(weight)):
        raise ValueError("values and mask must be finite")
    if np.any(weight < 0.0):
        raise ValueError("mask weights must be non-negative")
    denominator = float(np.sum(weight, dtype=np.float64))
    if denominator <= 0.0:
        raise ValueError("mask has zero weight")
    mean = float(np.sum(weight * array, dtype=np.float64) / denominator)
    return array - mean, mean


def build_galaxy_count_map(
    map_path: pathlib.Path | str,
    nside: int = NSIDE,
    z_min: float = 0.3,
    z_max: float = 0.5,
    chunk_rows: int = 1_000_000,
) -> tuple[np.ndarray, dict[str, int]]:
    """Pixelize the frozen HOD catalog in bounded memory.

    Invalid declinations are dropped rather than clipped and are reported
    separately.  The redshift predicate is strict, matching the halo catalog.
    """

    counts = np.zeros(hp.nside2npix(int(nside)), dtype=np.int64)
    report = {
        "rows_total": 0,
        "rows_selected": 0,
        "rows_invalid_flag": 0,
        "rows_nonfinite_coordinates_or_redshift": 0,
        "rows_invalid_declination": 0,
        "rows_outside_strict_redshift": 0,
    }
    with h5py.File(map_path, "r") as handle:
        if "galaxies" not in handle:
            raise KeyError("Map product has no galaxies dataset")
        dataset = handle["galaxies"]
        if dataset.ndim != 2 or dataset.shape[1] != len(GALAXY_COLUMNS):
            raise ValueError("Galaxy dataset does not have the frozen seven-column schema")
        declared = tuple(json.loads(str(handle.attrs["galaxy_catalog_columns_json"])))
        if declared != GALAXY_COLUMNS:
            raise ValueError(f"Galaxy column contract mismatch: {declared}")
        for start in range(0, dataset.shape[0], int(chunk_rows)):
            chunk = np.asarray(dataset[start : start + int(chunk_rows)])
            report["rows_total"] += int(len(chunk))
            ra, dec, redshift, valid = chunk[:, 0], chunk[:, 1], chunk[:, 2], chunk[:, 5]
            finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(redshift)
            valid_flag = np.isfinite(valid) & (valid > 0.5)
            valid_dec = finite & (dec >= -90.0) & (dec <= 90.0)
            in_redshift = finite & (redshift > float(z_min)) & (redshift < float(z_max))
            report["rows_invalid_flag"] += int(np.count_nonzero(~valid_flag))
            report["rows_nonfinite_coordinates_or_redshift"] += int(np.count_nonzero(~finite))
            report["rows_invalid_declination"] += int(np.count_nonzero(finite & ~valid_dec))
            report["rows_outside_strict_redshift"] += int(np.count_nonzero(finite & ~in_redshift))
            selected = valid_flag & valid_dec & in_redshift
            if np.any(selected):
                pix = hp.ang2pix(
                    int(nside), np.mod(ra[selected], 360.0), dec[selected], lonlat=True
                )
                counts += np.bincount(pix, minlength=counts.size).astype(np.int64)
                report["rows_selected"] += int(np.count_nonzero(selected))
    if int(np.sum(counts, dtype=np.int64)) != report["rows_selected"]:
        raise RuntimeError("Galaxy count-map sum differs from the selected row count")
    return counts, report


def galaxy_overdensity(
    counts: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Create a count overdensity normalized and demeaned on the common mask."""

    count_array = np.asarray(counts, dtype=np.float64)
    weight = np.asarray(mask, dtype=np.float64)
    if count_array.shape != weight.shape or np.any(count_array < 0.0):
        raise ValueError("counts and mask must be aligned and counts non-negative")
    mean_count = float(
        np.sum(weight * count_array, dtype=np.float64)
        / np.sum(weight, dtype=np.float64)
    )
    if not np.isfinite(mean_count) or mean_count <= 0.0:
        raise ValueError("No galaxies contribute to the common mask")
    delta, removed_mean = subtract_weighted_mean(count_array / mean_count - 1.0, weight)
    return delta, mean_count, removed_mean


def make_scalar_namaster_measurement(
    maps: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    lmax: int = LMAX,
    edges: Sequence[int] = NATIVE_12_EDGES,
    nmt_module=None,
) -> dict[str, object]:
    """Measure raw scalar bandpowers and save the exact common window.

    No theory curve or noise template is accepted by this function, which makes
    the measured result invariant under all later theory hypotheses.
    """

    if nmt_module is None:
        import pymaster as nmt_module

    edge_array = np.asarray(edges, dtype=np.int64)
    if not np.array_equal(edge_array, NATIVE_12_EDGES):
        raise ValueError("Only the frozen native 12-band edges are allowed")
    if int(edge_array[-1]) > int(lmax) + 1:
        raise ValueError("Estimator lmax does not contain all 12 frozen bands")
    weight = np.asarray(mask, dtype=np.float64)
    required = ("g", "y", "tau", "kappa")
    if set(maps) != set(required):
        raise ValueError(f"Expected exactly scalar maps {required}, got {tuple(maps)}")
    centered = {}
    means = {}
    for name in required:
        centered[name], means[name] = subtract_weighted_mean(maps[name], weight)

    bins = make_native_12_bins(nmt_module, lmax=int(lmax))
    fields = {
        name: nmt_module.NmtField(
            weight,
            [centered[name]],
            spin=0,
            beam=None,
            purify_e=False,
            purify_b=False,
            n_iter=0,
            n_iter_mask=0,
            lmax=int(lmax),
            lmax_mask=int(lmax),
            lite=True,
            masked_on_input=False,
        )
        for name in required
    }
    workspace = nmt_module.NmtWorkspace.from_fields(
        fields["g"],
        fields["g"],
        bins,
        l_toeplitz=-1,
        l_exact=-1,
        dl_band=-1,
    )
    windows_all = np.asarray(workspace.get_bandpower_windows(), dtype=np.float64)
    expected_window_shape = (1, len(edge_array) - 1, 1, int(lmax) + 1)
    if windows_all.shape != expected_window_shape:
        raise RuntimeError(
            f"Unexpected scalar bandpower-window shape {windows_all.shape}; "
            f"expected {expected_window_shape}"
        )
    coupled = {}
    decoupled = {}
    for spectrum, (left, right) in SPECTRUM_FIELDS.items():
        pcl = np.asarray(
            nmt_module.compute_coupled_cell(fields[left], fields[right]), dtype=np.float64
        )
        if pcl.shape != (1, int(lmax) + 1) or not np.all(np.isfinite(pcl)):
            raise RuntimeError(f"Invalid coupled spectrum for {spectrum}: {pcl.shape}")
        bpw = np.asarray(workspace.decouple_cell(pcl), dtype=np.float64)
        if bpw.shape != (1, len(edge_array) - 1) or not np.all(np.isfinite(bpw)):
            raise RuntimeError(f"Invalid decoupled spectrum for {spectrum}: {bpw.shape}")
        coupled[spectrum] = pcl[0].copy()
        decoupled[spectrum] = bpw[0].copy()
    return {
        "coupled": coupled,
        "decoupled_raw": decoupled,
        "effective_ell": np.asarray(bins.get_effective_ells(), dtype=np.float64),
        "windows_all": windows_all,
        "window": windows_all[0, :, 0, :].copy(),
        "field_weighted_means": means,
        "workspace": workspace,
    }


def decoupled_galaxy_shot_template(
    mean_count_per_pixel: float,
    mask: np.ndarray,
    workspace,
    *,
    nside: int = NSIDE,
    lmax: int = LMAX,
) -> dict[str, object]:
    """Return full, coupled, and estimator-decoupled Poisson shot noise."""

    mean_count = float(mean_count_per_pixel)
    if not np.isfinite(mean_count) or mean_count <= 0.0:
        raise ValueError("mean_count_per_pixel must be finite and positive")
    weight = np.asarray(mask, dtype=np.float64)
    fsky2 = float(np.mean(weight * weight, dtype=np.float64))
    full_sky_level = float(hp.nside2pixarea(int(nside)) / mean_count)
    coupled_level = fsky2 * full_sky_level
    coupled = np.full((1, int(lmax) + 1), coupled_level, dtype=np.float64)
    decoupled = np.asarray(workspace.decouple_cell(coupled), dtype=np.float64)
    if decoupled.ndim != 2 or decoupled.shape[0] != 1 or not np.all(np.isfinite(decoupled)):
        raise RuntimeError("Workspace returned an invalid galaxy shot-noise template")
    return {
        "full_sky_level": full_sky_level,
        "coupled_level": coupled_level,
        "coupled": coupled[0],
        "decoupled": decoupled[0],
        "fsky2": fsky2,
    }


def build_field_transfers(
    profile_smoothing_bell: np.ndarray,
    *,
    nside: int = NSIDE,
    lmax: int = LMAX,
    continuous_pixel_window_diagnostic: bool = False,
    pixel_window: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build the frozen baseline transfers or the explicit pixel diagnostic.

    Baseline: ``Tg=HEALPix pixel window`` and
    ``Ty=Ttau=Tkappa=saved Bell``.  Continuous maps are point-sampled by the
    painter, so multiplying them by a HEALPix pixel window is diagnostic only.
    """

    bell = np.asarray(profile_smoothing_bell, dtype=np.float64)
    if bell.shape != (int(lmax) + 1,) or not np.all(np.isfinite(bell)) or np.any(bell <= 0.0):
        raise ValueError("Saved profile_smoothing_Bell is invalid or incomplete")
    if pixel_window is None:
        pixel_window = hp.pixwin(int(nside), lmax=int(lmax))
    pixwin = np.asarray(pixel_window, dtype=np.float64)
    if pixwin.shape != bell.shape or not np.all(np.isfinite(pixwin)) or np.any(pixwin <= 0.0):
        raise ValueError("HEALPix pixel window is invalid or incomplete")
    continuous = bell * pixwin if continuous_pixel_window_diagnostic else bell.copy()
    return {
        "g": pixwin.copy(),
        "y": continuous.copy(),
        "tau": continuous.copy(),
        "kappa": continuous.copy(),
    }


def apply_forward_windows(
    window: np.ndarray,
    theory_cls: Mapping[str, np.ndarray],
    transfers: Mapping[str, np.ndarray],
    *,
    galaxy_shot_decoupled: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Apply the exact pair transfer before the saved NaMaster window."""

    operator = np.asarray(window, dtype=np.float64)
    if operator.ndim != 2 or not np.all(np.isfinite(operator)):
        raise ValueError("Scalar bandpower window must be a finite 2D array")
    nell = operator.shape[1]
    result = {}
    for spectrum, (left, right) in SPECTRUM_FIELDS.items():
        cl = np.asarray(theory_cls[spectrum], dtype=np.float64)
        tl = np.asarray(transfers[left], dtype=np.float64)
        tr = np.asarray(transfers[right], dtype=np.float64)
        if cl.shape != (nell,) or tl.shape != (nell,) or tr.shape != (nell,):
            raise ValueError(f"Dense theory/transfer support mismatch for {spectrum}")
        if not np.all(np.isfinite(cl * tl * tr)):
            raise ValueError(f"Non-finite forward theory for {spectrum}")
        signal = operator @ (tl * tr * cl)
        noise = np.zeros_like(signal)
        if spectrum == "gg" and galaxy_shot_decoupled is not None:
            noise = np.asarray(galaxy_shot_decoupled, dtype=np.float64)
            if noise.shape != signal.shape or not np.all(np.isfinite(noise)):
                raise ValueError("Galaxy shot template does not match the bandpower grid")
        result[spectrum] = {
            "signal": signal,
            "noise": noise.copy(),
            "total": signal + noise,
            "pair_transfer": tl * tr,
        }
    return result


def validate_final_map_product(
    map_path: pathlib.Path | str,
    *,
    expected_nside: int = NSIDE,
    expected_lmax: int = LMAX,
    verify_dataset_hashes: bool = True,
) -> dict[str, object]:
    """Validate the frozen map, kernel, transfer, and content-hash contract."""

    path = pathlib.Path(map_path)
    required_maps = ("map_ymap", "map_tau", "map_kappa_cmb")
    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)
        required_attrs = (
            "nside",
            "comparison_lmax",
            "ordering",
            "schema_version",
            "catalog_cosmology_sha256",
            "catalog_file_sha256",
            "galaxy_catalog_sha256",
            "map_dataset_sha256_json",
            "kernel_dataset_sha256_json",
            "profile_smoothing_applied",
            "healpix_pixel_window_applied_during_paste",
        )
        missing = [name for name in required_attrs if name not in attrs]
        if missing:
            raise ValueError(f"Map product is missing contract attrs: {missing}")
        if int(attrs["nside"]) != int(expected_nside) or int(attrs["comparison_lmax"]) != int(expected_lmax):
            raise ValueError("Map nside/lmax differ from the frozen estimator contract")
        if str(attrs["ordering"]) != "RING" or str(attrs["schema_version"]) != "sbi_three_probe_signal_v1":
            raise ValueError("Map ordering/schema differs from the frozen estimator contract")
        if not bool(attrs["profile_smoothing_applied"]):
            raise ValueError("Final map does not declare applied profile smoothing")
        if bool(attrs["healpix_pixel_window_applied_during_paste"]):
            raise ValueError("Continuous pasted maps unexpectedly declare a HEALPix pixel window")
        if "maps" not in handle or tuple(sorted(handle["maps"].keys())) != tuple(sorted(required_maps)):
            raise ValueError("Final map datasets differ from the frozen three-map schema")
        expected_shape = (hp.nside2npix(int(expected_nside)),)
        declared_map_hashes = json.loads(str(attrs["map_dataset_sha256_json"]))
        observed_map_hashes = {}
        for name in required_maps:
            dataset = handle[f"maps/{name}"]
            if dataset.shape != expected_shape or dataset.dtype != np.float32:
                raise ValueError(f"Map {name} has wrong shape or dtype")
            if not np.all(np.isfinite(dataset[:])):
                raise ValueError(f"Map {name} contains non-finite values")
            if verify_dataset_hashes:
                observed_map_hashes[name] = sha256_h5_dataset(dataset)
        if verify_dataset_hashes and observed_map_hashes != declared_map_hashes:
            raise ValueError("Map arrays do not match their declared hashes")

        if "kernels" not in handle:
            raise ValueError("Final map has no saved kernel bundle")
        kernels = handle["kernels"]
        declared_kernel_hashes = json.loads(str(attrs["kernel_dataset_sha256_json"]))
        if set(kernels.keys()) != set(declared_kernel_hashes):
            raise ValueError("Kernel inventory differs from the declared hash inventory")
        observed_kernel_hashes = {
            name: sha256_h5_dataset(kernels[name]) for name in kernels
        }
        if verify_dataset_hashes and observed_kernel_hashes != declared_kernel_hashes:
            raise ValueError("Kernel arrays do not match their declared hashes")
        kernel_attrs = dict(kernels.attrs)
        if str(kernel_attrs.get("catalog_cosmology_sha256", "")) != str(attrs["catalog_cosmology_sha256"]):
            raise ValueError("Map and kernel cosmology hashes differ")
        if kernel_attrs.get("continuous_field_transfer_policy") != (
            "T_y=T_tau=T_kappa=profile_smoothing_Bell; no HEALPix pixel window is applied during painting"
        ):
            raise ValueError("Continuous-field transfer policy is missing or incompatible")
        if kernel_attrs.get("galaxy_field_transfer_policy") != (
            "T_g=HEALPix pixel window for the count map"
        ):
            raise ValueError("Galaxy transfer policy is missing or incompatible")
        ell = np.asarray(kernels["profile_smoothing_ell"][:], dtype=np.int64)
        bell = np.asarray(kernels["profile_smoothing_Bell"][:], dtype=np.float64)
        if not np.array_equal(ell, np.arange(int(expected_lmax) + 1)) or bell.shape != ell.shape:
            raise ValueError("Saved Gaussian transfer does not cover every estimator ell")
        z = np.asarray(kernels["realized_hod_galaxy_redshift"][:], dtype=np.float64)
        nz = np.asarray(kernels["realized_hod_galaxy_nz"][:], dtype=np.float64)
        normalization = float(np.trapz(nz, z))
        if z.shape != nz.shape or abs(normalization - 1.0) > NZ_NORMALIZATION_ATOL:
            raise ValueError("Realized HOD galaxy n(z) is invalid or not normalized")
        if "galaxies" not in handle or handle["galaxies"].shape != (int(attrs["n_galaxies"]), 7):
            raise ValueError("Frozen galaxy catalog shape differs from its declaration")
        galaxy_hash = None
        if verify_dataset_hashes:
            galaxy_hash = sha256_h5_dataset(handle["galaxies"])
            if galaxy_hash != str(attrs["galaxy_catalog_sha256"]):
                raise ValueError("Galaxy catalog does not match its declared hash")
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "catalog_cosmology_sha256": str(attrs["catalog_cosmology_sha256"]),
        "catalog_file_sha256": str(attrs["catalog_file_sha256"]),
        "map_dataset_sha256": declared_map_hashes,
        "kernel_dataset_sha256": declared_kernel_hashes,
        "galaxy_catalog_sha256": str(attrs["galaxy_catalog_sha256"]),
        "galaxy_hash_recomputed": galaxy_hash,
        "profile_smoothing_Bell": bell,
        "realized_hod_nz_normalization": normalization,
    }
