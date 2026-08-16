from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_tsz_halo_correlations as diagnostics
import tsz_pasting as tp


BASE_PARAMS = HERE / "params_tsz.yaml"


def test_validated_map_loader_checks_marker_provenance(tmp_path: Path) -> None:
    path = tmp_path / "map.h5"
    selected_digest = "a" * 64
    ymap = np.zeros(hp.nside2npix(8), dtype=np.float32)
    ymap[7] = np.float32(1.0e-6)
    with h5py.File(path, "w") as handle:
        maps = handle.create_group("maps")
        maps.create_dataset("map_ymap", data=ymap)
        handle.attrs["schema"] = diagnostics.MAP_SCHEMA
        handle.attrs["ordering"] = "RING"
        handle.attrs["map_units"] = "dimensionless Compton-y"
        handle.attrs["complete_selected_catalog_painted"] = True
        handle.attrs["selected_row_index_sha256"] = selected_digest
        handle.attrs["painted_row_index_sha256"] = selected_digest
        handle.attrs["selected_rows_available"] = 1
        handle.attrs["nside"] = 8
    marker_path = path.with_name(path.name + ".validated.json")
    marker = {
        "schema": diagnostics.MARKER_SCHEMA,
        "complete_selected_catalog_painted": True,
        "finite": True,
        "nonnegative": True,
        "selected_row_index_sha256": selected_digest,
        "painted_row_index_sha256": selected_digest,
        "selected_rows_available": 1,
        "npix": len(ymap),
        "output_sha256": diagnostics.sha256_file(path),
    }
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    loaded, metadata = diagnostics.load_validated_ymap(path)
    np.testing.assert_array_equal(loaded, ymap)
    assert metadata["marker_sha256"] == diagnostics.sha256_file(marker_path)

    marker["schema"] = "wrong"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="marker schema"):
        diagnostics.load_validated_ymap(path)


def _synthetic_catalog(path: Path) -> None:
    cfg = tp.load_params(BASE_PARAMS)
    redshift = np.array([0.2, 0.5, 1.0], dtype=np.float64)
    chi = tp.comoving_distance_hmpc(redshift, cfg["cosmology"])
    origin = np.asarray(cfg["catalog"]["observer_xyz_hmpc"], dtype=np.float64)
    xyz = origin[None, :] + np.array(
        [[chi[0], 0.0, 0.0], [0.0, chi[1], 0.0], [0.0, 0.0, chi[2]]]
    )
    dtype = np.dtype(
        [
            ("M_interp", "f8"),
            ("X_interp", "f8"),
            ("Y_interp", "f8"),
            ("Z_interp", "f8"),
            ("redshift_interp", "f8"),
        ]
    )
    rows = np.empty(3, dtype=dtype)
    rows["M_interp"] = [1.1e13, 2.0e13, 3.0e13]
    rows["X_interp"], rows["Y_interp"], rows["Z_interp"] = xyz.T
    rows["redshift_interp"] = redshift
    with h5py.File(path, "w") as handle:
        handle.create_dataset("halos_selected", data=rows)
        handle.attrs["complete"] = True
        handle.attrs["mass_threshold"] = 1.0e13
        handle.attrs["selected_halo_count"] = 3
        handle.attrs["selection"] = "M_interp > 1e13"


def _params(path: Path, catalog_path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                f"base_params: {BASE_PARAMS}",
                "catalog:",
                f"  path: {catalog_path}",
                "  selection:",
                "    redshift_max: 0.5",
                "    expected_rows: 2",
                "map:",
                "  nside: 8",
                "runtime:",
                "  jax_platforms: cpu",
                "output:",
                f"  directory: {path.parent}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_exact_selected_halo_map_uses_inclusive_octant(tmp_path: Path) -> None:
    catalog_path = tmp_path / "halos.h5"
    params_path = tmp_path / "params.yaml"
    _synthetic_catalog(catalog_path)
    _params(params_path, catalog_path)
    cfg = tp.load_params(params_path)
    expected_sha = hashlib.sha256(np.asarray([0, 1], dtype="<i8").tobytes()).hexdigest()
    expected = {
        "selected_rows_available": 2,
        "selected_row_index_sha256": expected_sha,
        "source_catalog": str(catalog_path),
        "catalog_sha256": diagnostics.sha256_file(catalog_path),
        "config_sha256": tp._configuration_hash(cfg),
    }

    delta_h, footprint, metadata = diagnostics.build_selected_halo_overdensity(
        params_path,
        8,
        expected_map_metadata=expected,
        chunk_size=2,
    )
    assert metadata["selected_rows"] == 2
    assert metadata["count_map_sum"] == 2
    assert metadata["selected_row_index_sha256"] == expected_sha
    assert metadata["catalog_sha256"] == diagnostics.sha256_file(catalog_path)
    assert metadata["halos_outside_footprint"] == 0
    assert 0.12 < metadata["footprint_fsky"] < 0.2
    assert abs(float(np.mean(delta_h[footprint], dtype=np.float64))) < 1.0e-6
    assert np.count_nonzero(delta_h[~footprint]) == 0


def test_masked_pseudo_spectra_are_finite(tmp_path: Path) -> None:
    catalog_path = tmp_path / "halos.h5"
    params_path = tmp_path / "params.yaml"
    _synthetic_catalog(catalog_path)
    _params(params_path, catalog_path)
    delta_h, footprint, _ = diagnostics.build_selected_halo_overdensity(
        params_path,
        8,
        chunk_size=2,
    )
    ymap = np.zeros(hp.nside2npix(8), dtype=np.float32)
    active = np.flatnonzero(delta_h > 0.0)
    ymap[active] = np.linspace(1.0e-7, 2.0e-7, len(active), dtype=np.float32)
    result = diagnostics.compute_masked_pseudo_spectra(
        ymap,
        delta_h,
        footprint,
        lmax=16,
        iter_count=0,
        n_bins=6,
    )
    assert np.all(np.isfinite(result["cl_yy"]))
    assert np.all(np.isfinite(result["cl_hy"]))
    assert np.min(result["cl_yy"]) >= -1.0e-30
    assert abs(result["masked_y_mean_after_centering"]) < 1.0e-12
    assert abs(result["masked_halo_mean_after_centering"]) < 1.0e-6
    assert result["policy"]["mode_coupling_correction"] == "none"
    assert result["policy"]["f_sky_correction"] == "none"


def test_weighted_log_bins_preserve_constant() -> None:
    ell = np.arange(2, 101)
    values = np.full_like(ell, 3.25, dtype=np.float64)
    binned = diagnostics.weighted_log_bins(ell, values, ell_min=2, ell_max=100, n_bins=8)
    np.testing.assert_allclose(binned["value"], 3.25, rtol=0.0, atol=1.0e-14)
    assert np.all(np.diff(binned["ell_eff"]) > 0.0)
