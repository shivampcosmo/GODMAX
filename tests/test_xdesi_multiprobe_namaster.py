import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import healpy as hp
import numpy as np
import pytest
from astropy.io import fits

MODULE_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "xDESI" / "survey_measure"
sys.path.insert(0, str(MODULE_DIR))

import multiprobe_namaster as mp  # noqa: E402


def _write_minimal_dr9_catalog(path: Path, nside: int = 1) -> None:
    rows = []
    for pz_bin in range(1, 5):
        lon, lat = hp.pix2ang(nside, [0, 1], lonlat=True)
        rows.extend(
            [
                (float(lon[0]), float(lat[0]), 0.2 + 0.1 * pz_bin, 0.001 * pz_bin, pz_bin, 2.0),
                (float(lon[1]), float(lat[1]), 0.25 + 0.1 * pz_bin, -0.002 * pz_bin, pz_bin, 1.0),
            ]
        )
    with h5py.File(path, "w") as h5:
        h5.attrs["product_type"] = "synthetic DR9 weighted catalog"
        h5.attrs["n_objects"] = len(rows)
        h5.attrs["n_valid_for_cl"] = len(rows)
        cat = h5.create_group("catalog")
        cat.create_dataset("ra_deg", data=np.asarray([row[0] for row in rows], dtype=np.float32))
        cat.create_dataset("dec_deg", data=np.asarray([row[1] for row in rows], dtype=np.float32))
        cat.create_dataset("z", data=np.asarray([row[2] for row in rows], dtype=np.float32))
        cat.create_dataset("vr_over_c", data=np.asarray([row[3] for row in rows], dtype=np.float32))
        cat.create_dataset("pz_bin", data=np.asarray([row[4] for row in rows], dtype=np.uint8))
        cat.create_dataset("valid_for_cl", data=np.ones(len(rows), dtype=bool))
        cat.create_dataset("weight_imaging_mean1", data=np.asarray([row[5] for row in rows], dtype=np.float32))


def _write_minimal_random_count_file(path: Path, nside: int = 1) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["ordering"] = "RING"
        h5.attrs["caveat"] = "synthetic one-random caveat"
        g = h5.create_group(f"nside{nside}")
        g.create_dataset("random_count", data=np.ones(hp.nside2npix(nside), dtype=np.uint32))


def _write_minimal_shear_maps(path: Path, nside: int = 1) -> None:
    npix = hp.nside2npix(nside)
    with h5py.File(path, "w") as h5:
        maps = h5.create_group("maps")
        for tomo in range(4):
            g = maps.create_group(f"tomo{tomo}")
            g.create_dataset("mask_weight", data=np.ones(npix, dtype=np.float32))
            g.create_dataset("gamma1", data=np.full(npix, tomo + 1.0, dtype=np.float32))
            g.create_dataset("gamma2_namaster", data=np.full(npix, tomo + 2.0, dtype=np.float32))
            g.attrs["shape_noise_pseudo_cl_normalized_weight_mask"] = 0.1 * (tomo + 1)


class _FakeShearBundle:
    def __init__(self, path: Path):
        self.path = path

    def shear_path_for_nside(self, nside: int) -> Path:
        return self.path


def _fake_bundle(tmp_path: Path, nside: int = 1) -> SimpleNamespace:
    cat_path = tmp_path / "desi_dr9_catalog.h5"
    random_path = tmp_path / "desi_dr9_random_counts.h5"
    _write_minimal_dr9_catalog(cat_path, nside=nside)
    _write_minimal_random_count_file(random_path, nside=nside)
    return SimpleNamespace(
        desi_catalog=cat_path,
        desi_random_count_maps=random_path,
        sigma_true_gas_calibration=tmp_path / "missing_sigma_true.json",
    )


def test_default_spectrum_inventory_is_46():
    specs = mp.default_spectrum_specs()
    assert len(specs) == 46
    families = {}
    for spec in specs:
        families[spec.family] = families.get(spec.family, 0) + 1
    assert families == {
        "des_shear_EE": 10,
        "act_y_des_shear_E": 4,
        "desi_g_auto": 4,
        "desi_g_act_y": 4,
        "desi_g_des_shear_E": 16,
        "desi_g_act_kappa": 4,
        "desi_pi_act_T": 4,
    }


def test_covariance_grouping_reduces_raw_block_count_conservatively():
    specs = mp.default_spectrum_specs()
    groups = {}
    for i, spec_i in enumerate(specs):
        for spec_j in specs[i:]:
            key = mp.covariance_group_key_for_specs(spec_i, spec_j)
            groups.setdefault(key, 0)
            groups[key] += 1
    assert len(specs) * (len(specs) + 1) // 2 == 1081
    assert len(groups) == 259
    assert max(groups.values()) == 16


def test_sqrt_bandpowers_cover_requested_ell_range():
    left, right = mp.make_sqrt_bandpower_edges(8, 1024, 24)
    assert left[0] == 8
    assert right[-1] == 1025
    assert np.all(right > left)
    assert np.all(left[1:] >= right[:-1])


def test_des_y3_fiducial_bandpowers_match_transferred_edge_rule():
    left, right = mp.make_sqrt_bandpower_edges(8, 2048, 32)
    assert left[:5].tolist() == [8, 17, 30, 46, 66]
    assert right[:5].tolist() == [17, 30, 46, 66, 89]
    assert left[-5:].tolist() == [1492, 1596, 1704, 1815, 1930]
    assert right[-5:].tolist() == [1596, 1704, 1815, 1930, 2049]
    assert np.all(left[1:] == right[:-1])


def test_linear_bandpowers_match_cpu_production_edges():
    left, right = mp.make_linear_bandpower_edges(8, 1024, 10)
    assert left.tolist() == [8, 110, 212, 314, 415, 517, 619, 720, 822, 924]
    assert right.tolist() == [110, 212, 314, 415, 517, 619, 720, 822, 924, 1025]

    left, right = mp.make_linear_bandpower_edges(8, 4096, 10)
    assert left.tolist() == [8, 417, 826, 1235, 1644, 2053, 2462, 2871, 3280, 3689]
    assert right.tolist() == [417, 826, 1235, 1644, 2053, 2462, 2871, 3280, 3689, 4097]


def test_cpu_production_stage_configs_use_linear_binning():
    fast = mp.MeasurementConfig.for_stage("fast1024")
    assert fast.nside == 1024
    assert fast.lmax == 1024
    assert fast.n_bins == 10
    assert fast.binning == "linear"
    assert fast.default_measurement_path.name == "xdesi_multiprobe_cls_cov_nside1024_lmax1024_nbin10_linear.h5"

    mid = mp.MeasurementConfig.for_stage("midres2048")
    assert mid.nside == 2048
    assert mid.lmax == 4096
    assert mid.n_bins == 10
    assert mid.binning == "linear"
    assert mid.default_measurement_path.name == "xdesi_multiprobe_cls_cov_nside2048_lmax4096_nbin10_linear.h5"


def test_component_labels_match_namaster_ordering():
    assert mp.component_labels(0, 0) == ["00"]
    assert mp.component_labels(0, 2) == ["0E", "0B"]
    assert mp.component_labels(2, 0) == ["E0", "B0"]
    assert mp.component_labels(2, 2) == ["EE", "EB", "BE", "BB"]


def test_validate_field_map_requires_spin_consistent_map_count():
    valid = mp.FieldMap(
        name="s1",
        label="s1",
        kind="des_shear",
        spin=2,
        maps=[np.zeros(12, dtype=np.float32), np.ones(12, dtype=np.float32)],
        mask=np.ones(12, dtype=np.float32),
        mask_name="unit",
    )
    mp.validate_field_map_for_namaster(valid)

    invalid = mp.FieldMap(
        name="bad_s1",
        label="bad_s1",
        kind="des_shear",
        spin=2,
        maps=[np.zeros(12, dtype=np.float32)],
        mask=np.ones(12, dtype=np.float32),
        mask_name="unit",
    )
    with pytest.raises(ValueError, match="expected 2"):
        mp.validate_field_map_for_namaster(invalid)


def test_build_shear_fields_defaults_to_positive_convergence_scalar_cross_convention(tmp_path):
    shear_path = tmp_path / "shear.h5"
    _write_minimal_shear_maps(shear_path, nside=1)
    bundle = _FakeShearBundle(shear_path)

    fields = mp.build_shear_fields(bundle, mp.MeasurementConfig(nside=1, lmax=3))

    np.testing.assert_allclose(fields["s1"].maps[0], -1.0)
    np.testing.assert_allclose(fields["s1"].maps[1], -2.0)
    assert fields["s1"].metadata["shear_e_to_kappa_sign"] == -1.0

    raw_config = mp.MeasurementConfig(nside=1, lmax=3, shear_e_to_kappa_sign=1.0)
    raw_fields = mp.build_shear_fields(bundle, raw_config)
    np.testing.assert_allclose(raw_fields["s1"].maps[0], 1.0)
    np.testing.assert_allclose(raw_fields["s1"].maps[1], 2.0)


def test_covariance_component_selection_handles_flattened_namaster_ordering():
    n_bands = 3
    ncomp_a = 4
    ncomp_b = 2
    cov = np.arange(ncomp_a * n_bands * ncomp_b * n_bands, dtype=float).reshape(
        ncomp_a * n_bands,
        ncomp_b * n_bands,
    )
    selected = mp._select_covariance_component_block(
        cov,
        n_bands=n_bands,
        ncomp_a=ncomp_a,
        ncomp_b=ncomp_b,
        component_a=2,
        component_b=1,
    )
    rows = np.arange(n_bands) * ncomp_a + 2
    cols = np.arange(n_bands) * ncomp_b + 1
    np.testing.assert_allclose(selected, cov[np.ix_(rows, cols)])


def test_covariance_component_selection_handles_explicit_4d_layouts():
    n_bands = 3
    ncomp_a = 4
    ncomp_b = 2
    band_major = np.arange(n_bands * ncomp_a * n_bands * ncomp_b, dtype=float).reshape(
        n_bands,
        ncomp_a,
        n_bands,
        ncomp_b,
    )
    selected = mp._select_covariance_component_block(
        band_major,
        n_bands=n_bands,
        ncomp_a=ncomp_a,
        ncomp_b=ncomp_b,
        component_a=2,
        component_b=1,
    )
    np.testing.assert_allclose(selected, band_major[:, 2, :, 1])

    component_major = np.transpose(band_major, (1, 0, 3, 2))
    selected_component_major = mp._select_covariance_component_block(
        component_major,
        n_bands=n_bands,
        ncomp_a=ncomp_a,
        ncomp_b=ncomp_b,
        component_a=2,
        component_b=1,
    )
    np.testing.assert_allclose(selected_component_major, band_major[:, 2, :, 1])

    band_band_component_component = np.transpose(band_major, (0, 2, 1, 3))
    selected_band_band = mp._select_covariance_component_block(
        band_band_component_component,
        n_bands=n_bands,
        ncomp_a=ncomp_a,
        ncomp_b=ncomp_b,
        component_a=2,
        component_b=1,
    )
    np.testing.assert_allclose(selected_band_band, band_major[:, 2, :, 1])


def _probe_for_covariance_test(name: str, kind: str, spin: int) -> mp.NmtProbeField:
    info = mp.FieldMap(
        name=name,
        label=name,
        kind=kind,
        spin=spin,
        maps=[np.zeros(12, dtype=np.float32)],
        mask=np.ones(12, dtype=np.float32),
        mask_name="unit",
        metadata={},
    )
    return mp.NmtProbeField(info=info, field=None)


def test_covariance_total_bandpowers_zero_theory_null_spin_components():
    fields = {
        "s1": _probe_for_covariance_test("s1", "des_shear", 2),
        "s2": _probe_for_covariance_test("s2", "des_shear", 2),
        "g1": _probe_for_covariance_test("g1", "desi_galaxy", 0),
    }
    config = mp.MeasurementConfig(covariance_input_smooth_bandpowers=False)

    shear_auto = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 10.0, 10.0],
            [-10.0, -10.0, -10.0],
            [0.2, 0.3, 0.4],
        ]
    )
    prepared_auto = mp.prepare_total_bandpowers_for_covariance("s1", "s1", fields, shear_auto, config)
    np.testing.assert_allclose(prepared_auto[1], 0.0)
    np.testing.assert_allclose(prepared_auto[2], 0.0)
    assert np.all(prepared_auto[0] > 0.0)
    assert np.all(prepared_auto[3] > 0.0)

    shear_cross = shear_auto.copy()
    prepared_cross = mp.prepare_total_bandpowers_for_covariance("s1", "s2", fields, shear_cross, config)
    np.testing.assert_allclose(prepared_cross[1], 0.0)
    np.testing.assert_allclose(prepared_cross[2], 0.0)
    np.testing.assert_allclose(prepared_cross[3], 0.0)

    scalar_shear = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    prepared_scalar_shear = mp.prepare_total_bandpowers_for_covariance("g1", "s1", fields, scalar_shear, config)
    np.testing.assert_allclose(prepared_scalar_shear[1], 0.0)


def test_covariance_auto_total_bandpowers_are_clipped_positive():
    fields = {"g1": _probe_for_covariance_test("g1", "desi_galaxy", 0)}
    config = mp.MeasurementConfig(covariance_input_smooth_bandpowers=False)

    total = np.array([[-1.0, 0.0, 2.0]])
    prepared = mp.prepare_total_bandpowers_for_covariance("g1", "g1", fields, total, config)

    assert np.all(prepared > 0.0)
    assert prepared[0, 2] == 2.0


def test_covariance_bandpowers_unbin_to_full_ell_arrays():
    bins = mp.nmt.NmtBin.from_edges([2, 5], [5, 9])
    bandpowers = np.array([[1.0, 2.0]])

    full = mp.unbin_covariance_bandpowers_to_full_ell(bins, bandpowers, lmax=10)

    assert full.shape == (1, 11)
    np.testing.assert_allclose(full[0, :2], 1.0)
    np.testing.assert_allclose(full[0, 2:5], 1.0)
    np.testing.assert_allclose(full[0, 5:9], 2.0)
    np.testing.assert_allclose(full[0, 9:], 0.0)


def test_covariance_input_mode_must_be_supported():
    config = mp.MeasurementConfig(covariance_input_mode="pseudo_cl_mask_overlap")
    with pytest.raises(ValueError, match="Unsupported covariance_input_mode"):
        mp.compute_input_cl_for_covariance("a", "b", {}, None, {}, {}, config)


def test_gaussian_beam_transfer_is_normalized_and_damps_high_ell():
    beam = mp.gaussian_beam_transfer(4096, 1.6)
    assert beam.shape == (4097,)
    assert beam[0] == 1.0
    assert 0.0 < beam[-1] < beam[100]


def test_ksz_velocity_amplitudes_use_saved_sigma_rec_and_paper_r():
    field_meta = {
        f"pi{i}": {"metadata": {"rms_rec_vr_over_c": 0.001 * i}}
        for i in range(1, 5)
    }
    amps = mp.ksz_velocity_amplitudes_from_field_metadata(field_meta, sigma_true_over_c=[0.002] * 4)
    assert amps == {
        1: 0.3 * 0.001 * 0.002,
        2: 0.3 * 0.002 * 0.002,
        3: 0.3 * 0.003 * 0.002,
        4: 0.3 * 0.004 * 0.002,
    }


def test_default_ksz_velocity_amplitudes_use_abacus_sigma_true():
    field_meta = {
        f"pi{i}": {"metadata": {"rms_rec_vr_over_c": 0.001 * i}}
        for i in range(1, 5)
    }
    amps = mp.ksz_velocity_amplitudes_from_field_metadata(field_meta)
    assert amps[1] == 0.3 * 0.001 * mp.KSZ_SIGMA_TRUE_GAS_OVER_C_3E5[1]
    assert amps[4] == 0.3 * 0.004 * mp.KSZ_SIGMA_TRUE_GAS_OVER_C_3E5[4]


def test_ksz_velocity_amplitudes_prefer_weighted_sigma_rec():
    field_meta = {
        f"pi{i}": {
            "metadata": {
                "rms_rec_vr_over_c": 0.001 * i,
                "rms_rec_vr_over_c_weighted": 0.01 * i,
            }
        }
        for i in range(1, 5)
    }
    amps = mp.ksz_velocity_amplitudes_from_field_metadata(field_meta, sigma_true_over_c=[0.002] * 4)
    assert amps[1] == 0.3 * 0.01 * 0.002
    assert amps[4] == 0.3 * 0.04 * 0.002


def test_dr9_random_count_loader_reads_ring_count_map(tmp_path):
    path = tmp_path / "random_counts.h5"
    _write_minimal_random_count_file(path, nside=1)
    counts = mp._load_healpix_random_count_map(path, 1, allowed_nsides=(1,))
    assert counts.shape == (hp.nside2npix(1),)
    assert counts.dtype == np.float32
    np.testing.assert_allclose(counts, 1.0)


def test_sum_preserving_ud_grade_counts_preserves_total():
    counts = np.arange(1, hp.nside2npix(2) + 1, dtype=np.float64)
    downgraded = mp.sum_preserving_ud_grade_counts(counts, nside_out=1, nside_in=2)
    assert downgraded.shape == (hp.nside2npix(1),)
    np.testing.assert_allclose(np.sum(downgraded), np.sum(counts), rtol=1e-6)


def test_survey_bundle_resolves_and_validates_dr9_products(tmp_path):
    manifest = {
        "products": {
            "des_y3_shear_maps": {
                "nside1024": "shear1024.h5",
                "nside2048": "shear2048.h5",
                "nside4096": "shear4096.h5",
            },
            "desi_dr9_extended_velocity_catalogs": {
                "combined": "desi_dr9_catalog.h5",
            },
            "desi_dr9_imaging_randoms": {
                "quality_cut_randoms": "desi_dr9_randoms.h5",
                "count_maps_nside1024_4096": "desi_dr9_random_count_maps.h5",
            },
            "act_dr6_tsz_compton_y": "act_y.h5",
            "act_dr6_cmb_temperature": "act_T.h5",
            "act_dr6_lensing_kappa": "act_kappa.h5",
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _write_minimal_dr9_catalog(tmp_path / "desi_dr9_catalog.h5")
    with h5py.File(tmp_path / "desi_dr9_random_count_maps.h5", "w") as h5:
        h5.attrs["ordering"] = "RING"
        h5.create_group("nside1024").create_dataset("random_count", data=np.ones(1, dtype=np.uint32))
        h5.create_group("nside4096").create_dataset("random_count", data=np.ones(1, dtype=np.uint32))
    for name in ("desi_dr9_randoms.h5", "shear1024.h5", "shear2048.h5", "shear4096.h5", "act_y.h5", "act_T.h5", "act_kappa.h5"):
        (tmp_path / name).touch()

    bundle = mp.SurveyBundle.from_root(tmp_path)
    assert bundle.desi_catalog == tmp_path.resolve() / "desi_dr9_catalog.h5"
    assert bundle.desi_randoms == tmp_path.resolve() / "desi_dr9_randoms.h5"
    assert bundle.desi_random_count_maps == tmp_path.resolve() / "desi_dr9_random_count_maps.h5"
    assert bundle.shear_path_for_nside(2048) == tmp_path.resolve() / "shear2048.h5"
    info = bundle.validate_files()
    assert "desi_random_count_maps" in info


def test_build_desi_fields_uses_dr9_imaging_weights_and_weighted_shot_noise(tmp_path):
    nside = 1
    bundle = _fake_bundle(tmp_path, nside=nside)
    config = mp.MeasurementConfig(nside=nside, lmax=3, include_ksz_velocity_shuffle=True)
    random_counts = np.ones(hp.nside2npix(nside), dtype=np.float32)

    fields, summary = mp.build_desi_fields(bundle, config, random_counts)

    assert fields["g1"].mask_name == "desi_dr9_random"
    assert fields["pi1"].mask_name == "desi_dr9_random"
    assert "pi_shuf1" in fields
    assert fields["pi1"].has_catalog_momentum
    assert fields["pi_shuf1"].has_catalog_momentum
    assert fields["pi1"].metadata["namaster_field_class"] == "NmtFieldCatalogMomentum"
    np.testing.assert_allclose(fields["pi1"].catalog["weight"], [2.0, 1.0])
    np.testing.assert_allclose(fields["pi1"].catalog["field"], [0.001, -0.002])
    assert summary["desi_release"] == "DR9 Extended LRG"
    assert summary["selection_dataset"] == mp.DESI_DR9_SELECTION_DATASET
    assert summary["weight_dataset"] == mp.DESI_DR9_WEIGHT_DATASET

    g1 = fields["g1"].maps[0]
    expected_delta = np.full(hp.nside2npix(nside), -1.0, dtype=np.float32)
    expected_delta[0] = 7.0
    expected_delta[1] = 3.0
    np.testing.assert_allclose(g1, expected_delta, rtol=1e-6)

    pi1 = fields["pi1"].maps[0]
    np.testing.assert_allclose(np.mean(pi1), 0.0, atol=1e-10)

    meta = fields["g1"].metadata
    area_sr = 4.0 * np.pi
    expected_shot = area_sr * (2.0**2 + 1.0**2) / (2.0 + 1.0) ** 2
    np.testing.assert_allclose(meta["shot_noise"], expected_shot)
    assert meta["nbar_per_sr"] == meta["sum_weight"] / meta["area_sr"]

    vr = np.array([0.001, -0.002])
    weights = np.array([2.0, 1.0])
    expected_rms = np.sqrt(np.sum(weights * vr**2) / np.sum(weights))
    np.testing.assert_allclose(meta["rms_rec_vr_over_c"], expected_rms, rtol=1e-6)
    np.testing.assert_allclose(meta["rms_rec_vr_over_c_weighted"], expected_rms, rtol=1e-6)

    z_mid = summary["z_mid"]
    nz = summary["nz_dndz_by_pz"][0]
    np.testing.assert_allclose(np.trapezoid(nz, x=z_mid), 1.0)


def test_ksz_catalog_momentum_roundtrips_through_map_product(tmp_path):
    nside = 2
    bundle = _fake_bundle(tmp_path, nside=nside)
    config = mp.MeasurementConfig(nside=nside, lmax=3, include_ksz_velocity_shuffle=True)
    random_counts = np.ones(hp.nside2npix(nside), dtype=np.float32)
    fields, summary = mp.build_desi_fields(bundle, config, random_counts)
    output = tmp_path / "maps.h5"

    mp.save_map_product(output, fields, {"config": {}, "desi_summary": summary}, overwrite=True)
    loaded, _ = mp.load_map_product(output)

    assert loaded["pi1"].has_catalog_momentum
    assert loaded["pi_shuf1"].has_catalog_momentum
    np.testing.assert_allclose(loaded["pi1"].catalog["ra_deg"], fields["pi1"].catalog["ra_deg"])
    np.testing.assert_allclose(loaded["pi1"].catalog["dec_deg"], fields["pi1"].catalog["dec_deg"])
    np.testing.assert_allclose(loaded["pi1"].catalog["weight"], fields["pi1"].catalog["weight"])
    np.testing.assert_allclose(loaded["pi1"].catalog["field"], fields["pi1"].catalog["field"])


def test_build_nmt_fields_uses_catalog_momentum_and_mask_covariance_field(tmp_path):
    nside = 2
    bundle = _fake_bundle(tmp_path, nside=nside)
    config = mp.MeasurementConfig(nside=nside, lmax=3, include_ksz_velocity_shuffle=False)
    random_counts = np.ones(hp.nside2npix(nside), dtype=np.float32)
    fields, _ = mp.build_desi_fields(bundle, config, random_counts)

    nmt_fields = mp.build_nmt_fields({"pi1": fields["pi1"]}, config)
    pi = nmt_fields["pi1"]

    assert pi.is_catalog_momentum
    assert type(pi.field).__name__ == "NmtFieldCatalogMomentum"
    assert type(pi.cov_field).__name__ == "NmtField"
    assert not bool(getattr(pi.cov_field, "is_catalog", False))
    assert float(pi.field.Nf) > 0.0


def test_catalog_momentum_covariance_input_adds_back_zero_lag_noise(tmp_path):
    nside = 2
    bundle = _fake_bundle(tmp_path, nside=nside)
    config = mp.MeasurementConfig(nside=nside, lmax=3, include_ksz_velocity_shuffle=False)
    random_counts = np.ones(hp.nside2npix(nside), dtype=np.float32)
    fields, _ = mp.build_desi_fields(bundle, config, random_counts)
    nmt_fields = mp.build_nmt_fields({"pi1": fields["pi1"]}, config)

    total = mp.compute_catalog_momentum_input_cl_for_covariance("pi1", "pi1", nmt_fields, config)
    pcl_subtracted = mp.nmt.compute_coupled_cell(nmt_fields["pi1"].field, nmt_fields["pi1"].field)
    expected = pcl_subtracted.copy()
    expected[0, :] += nmt_fields["pi1"].field.Nf
    expected /= np.mean(nmt_fields["pi1"].mask * nmt_fields["pi1"].mask)

    np.testing.assert_allclose(total, expected)


def test_ksz_covariance_block_forces_all_inputs_to_pseudo_over_fsky(tmp_path):
    nside = 2
    bundle = _fake_bundle(tmp_path, nside=nside)
    config = mp.MeasurementConfig(nside=nside, lmax=3, ell_min=2, n_bins=1, include_ksz_velocity_shuffle=False)
    random_counts = np.ones(hp.nside2npix(nside), dtype=np.float32)
    fields, _ = mp.build_desi_fields(bundle, config, random_counts)
    fields = {"pi1": fields["pi1"]}
    fields["T"] = mp.FieldMap(
        name="T",
        label="T",
        kind="act_cmb_temperature",
        spin=0,
        maps=[np.linspace(-1.0, 1.0, hp.nside2npix(nside), dtype=np.float32)],
        mask=np.ones(hp.nside2npix(nside), dtype=np.float32),
        mask_name="unit",
        metadata={},
    )
    nmt_fields = mp.build_nmt_fields(fields, config)
    bins = mp.make_bins(config)
    spec = mp.SpectrumSpec(
        name="desi_pi_act_T_pz1",
        family="desi_pi_act_T",
        fields=("pi1", "T"),
        component=0,
        label="pi x T",
        theory_key="desi_g_tau_pz1",
    )
    input_cache = {}

    block = mp.compute_covariance_block(spec, spec, nmt_fields, bins, {}, {}, input_cache, config)

    assert block.shape == (1, 1)
    assert ("pseudo_over_fsky", "pi1", "pi1") in input_cache
    assert ("pseudo_over_fsky", "pi1", "T") in input_cache
    assert ("pseudo_over_fsky", "T", "pi1") in input_cache
    assert ("pseudo_over_fsky", "T", "T") in input_cache
    assert ("decoupled_total", "T", "T") not in input_cache


def test_load_des_y3_source_nz_normalizes_for_theory(tmp_path):
    path = tmp_path / "des_nz.fits"
    z_low = np.array([0.0, 0.1, 0.2])
    z_high = np.array([0.1, 0.2, 0.3])
    z_mid = 0.5 * (z_low + z_high)
    cols = [
        fits.Column(name="Z_LOW", format="D", array=z_low),
        fits.Column(name="Z_MID", format="D", array=z_mid),
        fits.Column(name="Z_HIGH", format="D", array=z_high),
    ]
    for i in range(1, 5):
        cols.append(fits.Column(name=f"BIN{i}", format="D", array=np.array([0.2, 0.6, 0.2])))
    hdu = fits.BinTableHDU.from_columns(cols, name="nz_source")
    for i in range(1, 5):
        hdu.header[f"SIG_E_{i}"] = 0.2 + 0.01 * i
        hdu.header[f"NGAL_{i}"] = 1.0 + 0.1 * i
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    nz = mp.load_des_y3_source_nz(path)
    assert nz["dndz_by_bin"].shape == (4, 3)
    np.testing.assert_allclose(np.trapezoid(nz["dndz_by_bin"], x=nz["z_mid"], axis=1), np.ones(4))
    np.testing.assert_allclose(nz["sigma_e_by_bin"], [0.21, 0.22, 0.23, 0.24])
    assert nz["priors"]["Delta_z_bias_bin1"]["sigma"] == 1.8e-2


def test_theory_to_data_vector_uses_saved_windows(tmp_path):
    path = tmp_path / "measurement.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MEASUREMENT
        h5.attrs["config_json"] = json.dumps({"nside": 8, "lmax": 16})
        fields = h5.create_group("fields")
        fields.attrs["metadata_json"] = json.dumps(
            {
                "g1": {"spin": 0, "kind": "desi_galaxy"},
                "y": {"spin": 0, "kind": "act_tsz_y"},
            }
        )
        joint = h5.create_group("joint")
        joint.create_dataset("spectrum_names", data=np.array(["desi_g_act_y_pz1"], dtype=string_dtype))
        spectra = h5.create_group("spectra")
        spec = spectra.create_group("desi_g_act_y_pz1")
        spec.attrs["fields"] = json.dumps(["g1", "y"])
        spec.attrs["theory_key"] = "desi_g_act_y_pz1"
        spec.attrs["family"] = "desi_g_act_y"
        spec.attrs["metadata_json"] = json.dumps({"desi_pz": 1})
        window = np.zeros((2, 17))
        window[0, 4] = 1.0
        window[1, 8] = 1.0
        spec.create_dataset("bandpower_window_selected", data=window)

    theory = {"desi_g_act_y_pz1": np.arange(17, dtype=float)}
    transfers = {"g1": np.ones(17), "y": np.ones(17)}
    dv, names = mp.theory_to_data_vector(
        path,
        theory,
        transfer_functions=transfers,
        include_default_pixel_windows=False,
        include_default_act_beams=False,
    )
    assert names == ["desi_g_act_y_pz1"]
    np.testing.assert_allclose(dv, [4.0, 8.0])


def test_theory_to_data_vector_applies_optional_shear_m_bias(tmp_path):
    path = tmp_path / "measurement.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MEASUREMENT
        h5.attrs["config_json"] = json.dumps({"nside": 8, "lmax": 16})
        fields = h5.create_group("fields")
        fields.attrs["metadata_json"] = json.dumps(
            {
                "s1": {"spin": 2, "kind": "des_shear"},
                "s2": {"spin": 2, "kind": "des_shear"},
            }
        )
        joint = h5.create_group("joint")
        joint.create_dataset("spectrum_names", data=np.array(["des_shear_EE_tomo1_tomo2"], dtype=string_dtype))
        spectra = h5.create_group("spectra")
        spec = spectra.create_group("des_shear_EE_tomo1_tomo2")
        spec.attrs["fields"] = json.dumps(["s1", "s2"])
        spec.attrs["theory_key"] = "des_shear_EE_tomo1_tomo2"
        spec.attrs["family"] = "des_shear_EE"
        spec.attrs["metadata_json"] = json.dumps({"source_tomo_a": 1, "source_tomo_b": 2})
        window = np.zeros((1, 17))
        window[0, 4] = 1.0
        spec.create_dataset("bandpower_window_selected", data=window)

    theory = {"des_shear_EE_tomo1_tomo2": np.ones(17, dtype=float)}
    dv, names = mp.theory_to_data_vector(
        path,
        theory,
        shear_m_bias={1: 0.01, 2: -0.02},
        include_default_pixel_windows=False,
    )
    assert names == ["des_shear_EE_tomo1_tomo2"]
    np.testing.assert_allclose(dv, [(1.01 * 0.98)])


def test_theory_to_data_vector_converts_positive_kappa_shear_theory_to_saved_sign(tmp_path):
    path = tmp_path / "measurement.h5"
    string_dtype = h5py.string_dtype("utf-8")

    def write_product(sign):
        with h5py.File(path, "w") as h5:
            h5.attrs["schema"] = mp.SCHEMA_MEASUREMENT
            h5.attrs["config_json"] = json.dumps({"nside": 8, "lmax": 16})
            fields = h5.create_group("fields")
            fields.attrs["metadata_json"] = json.dumps(
                {
                    "g1": {"spin": 0, "kind": "desi_galaxy"},
                    "s1": {
                        "spin": 2,
                        "kind": "des_shear",
                        "metadata": {"shear_e_to_kappa_sign": sign},
                    },
                }
            )
            joint = h5.create_group("joint")
            joint.create_dataset(
                "spectrum_names",
                data=np.array(["desi_g_des_shear_E_pz1_tomo1"], dtype=string_dtype),
            )
            spectra = h5.create_group("spectra")
            spec = spectra.create_group("desi_g_des_shear_E_pz1_tomo1")
            spec.attrs["fields"] = json.dumps(["g1", "s1"])
            spec.attrs["theory_key"] = "desi_g_des_shear_E_pz1_tomo1"
            spec.attrs["family"] = "desi_g_des_shear_E"
            spec.attrs["metadata_json"] = json.dumps({"desi_pz": 1, "source_tomo": 1})
            window = np.zeros((1, 17))
            window[0, 4] = 1.0
            spec.create_dataset("bandpower_window_selected", data=window)

    theory = {"desi_g_des_shear_E_pz1_tomo1": np.ones(17, dtype=float)}

    write_product(-1.0)
    dv, _ = mp.theory_to_data_vector(path, theory, include_default_pixel_windows=False)
    np.testing.assert_allclose(dv, [1.0])

    write_product(1.0)
    dv, _ = mp.theory_to_data_vector(path, theory, include_default_pixel_windows=False)
    np.testing.assert_allclose(dv, [-1.0])


def test_theory_to_data_vector_can_calibrate_ksz_from_sigma_true(tmp_path):
    path = tmp_path / "measurement.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MEASUREMENT
        h5.attrs["config_json"] = json.dumps({"nside": 8, "lmax": 16})
        fields = h5.create_group("fields")
        fields.attrs["metadata_json"] = json.dumps(
            {
                "T": {"spin": 0, "kind": "act_cmb_temperature"},
                **{
                    f"pi{i}": {
                        "spin": 0,
                        "kind": "desi_momentum",
                        "metadata": {"rms_rec_vr_over_c": 0.001 * i},
                    }
                    for i in range(1, 5)
                },
            }
        )
        joint = h5.create_group("joint")
        joint.create_dataset("spectrum_names", data=np.array(["desi_pi_act_T_pz1"], dtype=string_dtype))
        spectra = h5.create_group("spectra")
        spec = spectra.create_group("desi_pi_act_T_pz1")
        spec.attrs["fields"] = json.dumps(["pi1", "T"])
        spec.attrs["theory_key"] = "desi_g_tau_pz1"
        spec.attrs["family"] = "desi_pi_act_T"
        spec.attrs["metadata_json"] = json.dumps({"desi_pz": 1})
        window = np.zeros((1, 17))
        window[0, 4] = 1.0
        spec.create_dataset("bandpower_window_selected", data=window)

    theory = {"desi_g_tau_pz1": np.ones(17, dtype=float)}
    dv, names = mp.theory_to_data_vector(
        path,
        theory,
        ksz_sigma_true_over_c=[0.002] * 4,
        include_default_pixel_windows=False,
        include_default_act_beams=False,
    )
    assert names == ["desi_pi_act_T_pz1"]
    np.testing.assert_allclose(dv, [-mp.TCMB_UK * 0.3 * 0.001 * 0.002])


def test_theory_to_data_vector_uses_default_abacus_ksz_calibration(tmp_path):
    path = tmp_path / "measurement.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MEASUREMENT
        h5.attrs["config_json"] = json.dumps({"nside": 8, "lmax": 16})
        fields = h5.create_group("fields")
        fields.attrs["metadata_json"] = json.dumps(
            {
                "T": {"spin": 0, "kind": "act_cmb_temperature"},
                **{
                    f"pi{i}": {
                        "spin": 0,
                        "kind": "desi_momentum",
                        "metadata": {"rms_rec_vr_over_c": 0.001 * i},
                    }
                    for i in range(1, 5)
                },
            }
        )
        joint = h5.create_group("joint")
        joint.create_dataset("spectrum_names", data=np.array(["desi_pi_act_T_pz1"], dtype=string_dtype))
        spectra = h5.create_group("spectra")
        spec = spectra.create_group("desi_pi_act_T_pz1")
        spec.attrs["fields"] = json.dumps(["pi1", "T"])
        spec.attrs["theory_key"] = "desi_g_tau_pz1"
        spec.attrs["family"] = "desi_pi_act_T"
        spec.attrs["metadata_json"] = json.dumps({"desi_pz": 1})
        window = np.zeros((1, 17))
        window[0, 4] = 1.0
        spec.create_dataset("bandpower_window_selected", data=window)

    theory = {"desi_g_tau_pz1": np.ones(17, dtype=float)}
    dv, names = mp.theory_to_data_vector(
        path,
        theory,
        include_default_pixel_windows=False,
        include_default_act_beams=False,
    )
    assert names == ["desi_pi_act_T_pz1"]
    np.testing.assert_allclose(dv, [-mp.TCMB_UK * 0.3 * 0.001 * mp.KSZ_SIGMA_TRUE_GAS_OVER_C_3E5[1]])
