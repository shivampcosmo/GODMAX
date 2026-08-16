from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tsz_pasting as tp


PARAMS = HERE / "params_tsz.yaml"
ZMAX_PARAMS = HERE / "params_tsz_zmax0p85.yaml"


def _synthetic_catalog(path: Path) -> None:
    cfg = tp.load_params(PARAMS)
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


def _overrides(catalog_path: Path, output_dir: Path, pressure_amplitude: float = 0.0) -> dict:
    return {
        "catalog": {
            "path": str(catalog_path),
            "selection": {"expected_rows": 3},
        },
        "map": {"nside": 8, "pressure_amplitude": pressure_amplitude},
        "runtime": {
            "halo_chunk_size": 2,
            "pixel_batch_size": 2,
            "pair_batch_size": 8,
            "pixel_workers": 1,
            "verbose": False,
        },
        "output": {
            "directory": str(output_dir),
            "compression": None,
            "overwrite": False,
        },
        "validation": {"preflight_chunk_size": 2},
    }


def test_adapter_uses_observer_relative_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "halos.h5"
    _synthetic_catalog(path)
    cfg = tp.load_params(PARAMS, _overrides(path, tmp_path))
    with h5py.File(path, "r") as handle:
        rows = handle["halos_selected"][:]
    catalog = tp.adapt_records(rows, cfg)
    np.testing.assert_allclose(catalog["ra_deg"], [0.0, 90.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(catalog["dec_deg"], [0.0, 0.0, 90.0], atol=1.0e-12)
    expected_chi = tp.comoving_distance_hmpc(catalog["z"], cfg["cosmology"])
    np.testing.assert_allclose(catalog["chi_hMpc"], expected_chi, rtol=1.0e-12)
    np.testing.assert_allclose(catalog["DA_hMpc"], expected_chi / (1.0 + catalog["z"]), rtol=1.0e-12)
    assert np.all(catalog["R200c_hMpc"] > 0.0)


def test_full_stream_preflight_and_higher_cut(tmp_path: Path) -> None:
    path = tmp_path / "halos.h5"
    _synthetic_catalog(path)
    cfg = tp.load_params(PARAMS, _overrides(path, tmp_path))
    report = tp.preflight_catalog(cfg)
    assert report["source_rows"] == 3
    assert report["selected_rows"] == 3
    assert report["all_source_rows_pass_cut"] is True
    assert report["max_distance_redshift_relative_error"] < 1.0e-12

    higher = tp.load_params(
        PARAMS,
        tp._deep_update(
            _overrides(path, tmp_path),
            {"catalog": {"selection": {"mass_min_hmsun": 1.5e13}}},
        ),
    )
    assert tp.preflight_catalog(higher)["selected_rows"] == 2
    np.testing.assert_array_equal(tp.stratified_row_indices(cfg, 3), [0, 1, 2])


def test_inclusive_redshift_limit_filters_stream_and_records_predicate(tmp_path: Path) -> None:
    path = tmp_path / "halos.h5"
    output = tmp_path / "zmax.h5"
    _synthetic_catalog(path)
    overrides = tp._deep_update(
        _overrides(path, tmp_path),
        {"catalog": {"selection": {"redshift_max": 0.5, "expected_rows": 2}}},
    )
    cfg = tp.load_params(PARAMS, overrides)
    report = tp.preflight_catalog(cfg)
    assert report["source_rows"] == 3
    assert report["selected_rows"] == 2
    assert report["all_source_rows_pass_cut"] is False
    assert report["all_source_rows_pass_mass_cut"] is True
    assert report["z_max"] == 0.5
    expected_index_hash = hashlib.sha256(np.asarray([0, 1], dtype="<i8").tobytes()).hexdigest()
    assert report["selected_row_index_sha256"] == expected_index_hash
    np.testing.assert_array_equal(tp.stratified_row_indices(cfg, 3), [0, 1])
    chunks = list(tp._iter_selected_chunks(cfg, max_halos=None))
    np.testing.assert_array_equal(np.concatenate([chunk["z"] for chunk in chunks]), [0.2, 0.5])

    result = tp.run_tsz_paste(PARAMS, overrides=overrides, output_path=output)
    with h5py.File(result["path"], "r") as handle:
        assert list(handle["maps"].keys()) == ["map_ymap"]
        assert handle.attrs["n_halos_painted"] == 2
        assert handle.attrs["selected_rows_available"] == 2
        assert handle.attrs["redshift_max"] == 0.5
        assert bool(handle.attrs["redshift_max_is_inclusive"])
        assert handle.attrs["selection_predicate"].endswith("redshift_interp <= 0.5")
        assert handle.attrs["selected_row_index_sha256"] == expected_index_hash
        assert handle.attrs["painted_row_index_sha256"] == expected_index_hash
        assert bool(handle.attrs["complete_selected_catalog_painted"])

    with pytest.raises(ValueError, match="fail the configured selection"):
        list(tp._iter_selected_chunks(cfg, max_halos=None, row_indices=np.asarray([0, 2])))


def test_run_specific_config_inherits_base_and_records_sources() -> None:
    cfg = tp.load_params(ZMAX_PARAMS)
    assert cfg["catalog"]["selection"]["redshift_max"] == 0.85
    assert cfg["catalog"]["selection"]["expected_rows"] == 1299336
    assert cfg["map"]["nside"] == 2048
    assert cfg["runtime"]["jax_platforms"] == "cuda"
    assert [Path(path).name for path in cfg["_config_sources"]] == [
        "params_tsz.yaml",
        "params_tsz_zmax0p85.yaml",
    ]
    boundary = np.asarray([np.nextafter(0.85, -np.inf), 0.85, np.nextafter(0.85, np.inf)])
    mask = tp._selection_mask(np.full(3, 2.0e13), boundary, cfg)
    np.testing.assert_array_equal(mask, [True, True, False])


def test_wrong_observer_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "halos.h5"
    _synthetic_catalog(path)
    overrides = tp._deep_update(
        _overrides(path, tmp_path),
        {"catalog": {"observer_xyz_hmpc": [0.0, 0.0, 0.0]}},
    )
    with pytest.raises(ValueError, match="observer_xyz_hmpc"):
        tp.load_params(PARAMS, overrides)


def test_zero_amplitude_writes_only_exact_zero_ymap(tmp_path: Path) -> None:
    catalog_path = tmp_path / "halos.h5"
    output_path = tmp_path / "zero.h5"
    _synthetic_catalog(catalog_path)
    result = tp.run_tsz_paste(
        PARAMS,
        overrides=_overrides(catalog_path, tmp_path, pressure_amplitude=0.0),
        output_path=output_path,
    )
    assert result["diagnostics"]["n_halos_painted"] == 3
    with h5py.File(output_path, "r") as handle:
        assert list(handle.keys()) == ["maps"]
        assert list(handle["maps"].keys()) == ["map_ymap"]
        ymap = handle[tp.MAP_DATASET][:]
        assert ymap.dtype == np.float32
        assert np.count_nonzero(ymap) == 0
        assert handle.attrs["n_halos_painted"] == 3
        assert bool(handle.attrs["mass_definition_is_provisional"])
        assert handle.attrs["map_units"] == "dimensionless Compton-y"
        assert handle.attrs["central_radius_policy"] == "extended_projected_grid_no_extrapolation"
        assert handle.attrs["n_pairs_below_projected_grid"] == 0

    sampled_path = tmp_path / "sampled_zero.h5"
    sampled = tp.run_tsz_paste(
        PARAMS,
        overrides=_overrides(catalog_path, tmp_path, pressure_amplitude=0.0),
        row_indices=np.array([0, 2], dtype=np.int64),
        output_path=sampled_path,
    )
    assert sampled["diagnostics"]["n_halos_painted"] == 2
    assert sampled["diagnostics"]["catalog_sampling"] == "explicit_row_indices"


def test_atomic_publication_does_not_clobber_concurrent_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path = tmp_path / "halos.h5"
    output_path = tmp_path / "race.h5"
    _synthetic_catalog(catalog_path)
    winner = b"concurrent-winner"

    def concurrent_link(_staging: str | Path, target: str | Path) -> None:
        Path(target).write_bytes(winner)
        raise FileExistsError(target)

    monkeypatch.setattr(tp.os, "link", concurrent_link)
    with pytest.raises(FileExistsError, match="concurrently-created"):
        tp.run_tsz_paste(
            PARAMS,
            overrides=_overrides(catalog_path, tmp_path, pressure_amplitude=0.0),
            output_path=output_path,
        )
    assert output_path.read_bytes() == winner
    assert list(tmp_path.glob(f".{output_path.name}.tmp.*")) == []


def test_fixed_pair_padding_does_not_change_values() -> None:
    import jax.numpy as jnp

    work = {
        "distances": np.array([0.2, 0.4, 0.8, 1.6, 3.2], dtype=np.float32),
        "z": np.array([0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        "logM": np.log(np.array([1.1e13, 1.2e13, 1.3e13, 1.4e13, 1.5e13])).astype(np.float32),
    }

    def evaluator(props):
        return jnp.exp(props[:, 0]) + props[:, 1] + 1.0e-15 * props[:, 2]

    batch3 = tp.evaluate_pairs_fixed(evaluator, work, 3)
    batch8 = tp.evaluate_pairs_fixed(evaluator, work, 8)
    np.testing.assert_array_equal(batch3, batch8)


def test_profile_evaluator_rejects_radial_extrapolation() -> None:
    import jax.numpy as jnp

    class Setup:
        rp_array = jnp.array([0.01, 0.1], dtype=jnp.float32)
        z_array = jnp.array([0.2, 0.3], dtype=jnp.float32)
        M_array = jnp.array([1.0e13, 2.0e13], dtype=jnp.float64)

        @staticmethod
        def log_y2D_interp(log_radius, redshift, log_mass):
            del redshift, log_mass
            return log_radius

    evaluator = tp.make_pair_evaluator(Setup(), 4)
    below = {
        "distances": np.array([0.001], dtype=np.float32),
        "z": np.array([0.2], dtype=np.float32),
        "logM": np.log(np.array([1.0e13])).astype(np.float32),
    }
    with pytest.raises(ValueError, match="below the extended projected profile grid"):
        tp.evaluate_pairs_fixed(evaluator, below, 4)

    above = {**below, "distances": np.array([0.11], dtype=np.float32)}
    with pytest.raises(ValueError, match="above the projected profile grid"):
        tp.evaluate_pairs_fixed(evaluator, above, 4)


def test_invalid_physics_switches_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "halos.h5"
    _synthetic_catalog(path)
    with pytest.raises(ValueError, match="physical_table_cosh"):
        tp.load_params(
            PARAMS,
            tp._deep_update(
                _overrides(path, tmp_path),
                {"profiles": {"overrides": {"analysis": {"projected_profile_integration_method": "legacy_log_radius"}}}},
            ),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        tp.load_params(PARAMS, tp._deep_update(_overrides(path, tmp_path), {"map": {"pressure_amplitude": -1.0}}))
    with pytest.raises(ValueError, match="central_radius_policy"):
        tp.load_params(
            PARAMS,
            tp._deep_update(
                _overrides(path, tmp_path),
                {"map": {"central_radius_policy": "sentinel_fill"}},
            ),
        )
