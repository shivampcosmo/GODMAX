from __future__ import annotations

import copy
import sys
from pathlib import Path

import healpy as hp
import numpy as np
from pixell import enmap


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import redmapper_y_cross as cross


def test_real_catalog_strict_cuts_and_row_digests() -> None:
    cases = (
        (
            cross.DEFAULT_CONFIG_PATH,
            {
                "simulation": (5616, "893e493e6c042c1ad6dfa8a8b008de131183d9f69171b91dfc64176b0e0f8967"),
                "data": (9245, "cd075371a789add09c8373fe3641d09d126936785d906aba25235fa197902eb2"),
            },
        ),
        (
            HERE / "params_redmapper_y_cross_z0p4_0p6.yaml",
            {
                "simulation": (6591, "e65eee2e0f0734ec66d44cd528eec9d61a83c6b890c1ed514eade432fddd6683"),
                "data": (5441, "a205a793333a521a51fd5aa4aa0d96120b8ec53be24e400fce215d387d575483"),
            },
        ),
    )
    for config_path, expected in cases:
        cfg = cross.load_config(config_path)
        zmin = float(cfg["selection"]["redshift_min"])
        zmax = float(cfg["selection"]["redshift_max"])
        for section, (expected_rows, expected_digest) in expected.items():
            result = cross.load_cluster_selection(
                cfg[section]["cluster_fits"], cfg, expected_rows, expected_digest
            )
            assert result["pre_mask_rows"] == expected_rows
            assert result["pre_mask_row_index_sha256"] == expected_digest
            assert np.all(result["richness"] > 20.0)
            assert np.all((result["redshift"] > zmin) & (result["redshift"] < zmax))

    cfg = cross.load_config(HERE / "params_redmapper_y_cross_z0p4_0p6.yaml")
    assert cross.selection_label(cfg) == "lambda > 20 and 0.4 < z < 0.6 (strict; z column)"


def test_deterministic_nested_random_selection() -> None:
    rows = np.arange(100_000, dtype=np.int64)
    first = cross.deterministic_nested_sample(rows, 20_000, 91)
    repeated = cross.deterministic_nested_sample(rows, 20_000, 91)
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(
        first[:5_000], cross.deterministic_nested_sample(rows, 5_000, 91)
    )
    assert len(np.unique(first)) == len(first)
    assert len(np.intersect1d(first[:10_000], first[10_000:])) == 0


def test_car_extraction_uses_positive_nonuniform_solid_angle_weights() -> None:
    cfg = copy.deepcopy(cross.load_config())
    shape, wcs = enmap.geometry(
        pos=np.deg2rad([[0.0, 0.0], [60.0, 2.0]]),
        res=np.deg2rad(1.0),
        proj="car",
    )
    ymap = enmap.ones(shape, wcs, dtype=np.float32)
    mask = np.ones(hp.nside2npix(8), dtype=np.float32)
    cfg["data"]["mask_threshold"] = 0.9
    result = cross.extract_data_y_pixels(ymap, mask, cfg, factor=1, row_block=8)
    assert len(result["k"]) > 0
    assert np.all(result["weights"] > 0.0)
    assert np.ptp(result["weights"]) > 0.1
    assert np.allclose(result["k"], 1.0)


def test_compensated_treecorr_profile_and_null_rank() -> None:
    cfg = copy.deepcopy(cross.load_config())
    cfg["treecorr"].update(
        {
            "nbins": 5,
            "min_sep_arcmin": 0.5,
            "max_sep_arcmin": 20.0,
            "num_threads": 1,
        }
    )
    offsets_deg = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0]) / 60.0
    y_pixels = {
        "ra_deg": np.concatenate([offsets_deg, 5.0 + offsets_deg]),
        "dec_deg": np.zeros(10),
        "k": np.concatenate([np.ones(5), np.zeros(5)]),
        "weights": np.ones(10),
    }
    clusters = {
        "ra_deg": np.asarray([0.0]),
        "dec_deg": np.asarray([0.0]),
        "selected_rows": 1,
    }
    randoms = {"ra_deg": np.asarray([5.0]), "dec_deg": np.asarray([0.0])}
    result = cross.measure_unpatched_profile(clusters, randoms, y_pixels, cfg, 1)
    assert np.mean(result["xi"]) > 0.0

    null = cross._null_chi2(np.asarray([1.0, 2.0]), np.eye(2), 1.0e-10)
    assert null["rank"] == 2
    assert np.isclose(null["chi2"], 5.0)
    assert 0.0 < null["pte"] < 1.0
