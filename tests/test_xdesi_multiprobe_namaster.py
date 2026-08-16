import hashlib
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
import godmax_multiprobe_theory_utils as gmt  # noqa: E402
import migrate_galaxy_auto_shot_noise as migrate_gshot  # noqa: E402
import plot_highres_pilot_dell as plot_highres_pilot  # noqa: E402
import plot_multiprobe_measurement as plot_measurement  # noqa: E402
import run_multiprobe_production as prod  # noqa: E402


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
            g.create_dataset("mask_weight_raw", data=np.full(npix, 2.0, dtype=np.float32))
            g.create_dataset("gamma1", data=np.full(npix, tomo + 1.0, dtype=np.float32))
            g.create_dataset("gamma2_namaster", data=np.full(npix, tomo + 2.0, dtype=np.float32))
            g.attrs["shape_noise_pseudo_cl_normalized_weight_mask"] = 0.1 * (tomo + 1)
            g.attrs["shape_noise_pseudo_cl_raw_weight_mask"] = 0.4 * (tomo + 1)


def _write_minimal_true_nz(path: Path) -> None:
    zmin = np.array([0.0, 0.5], dtype=np.float64)
    zmax = np.array([0.5, 1.0], dtype=np.float64)
    zmid = 0.5 * (zmin + zmax)
    dz = zmax - zmin
    with h5py.File(path, "w") as h5:
        bins = h5.create_group("redshift_bins")
        bins.create_dataset("zmin", data=zmin)
        bins.create_dataset("zmax", data=zmax)
        bins.create_dataset("zmid", data=zmid)
        bins.create_dataset("dz", data=dz)
        root = h5.create_group(mp.DESI_DR9_TRUE_NZ_GROUP_FULL_CL)
        for pz_bin in range(1, 5):
            group = root.create_group(f"pz{pz_bin}")
            group.create_dataset(mp.DESI_DR9_TRUE_NZ_DATASET, data=np.ones_like(zmid))
            group.attrs["mean_z"] = 0.5
            group.attrs["sigma_z"] = 0.25
            group.attrs["surface_density_per_deg2"] = 1.0


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
    true_nz_path = tmp_path / "desi_dr9_true_nz.h5"
    _write_minimal_true_nz(true_nz_path)
    return SimpleNamespace(
        desi_catalog=cat_path,
        desi_random_count_maps=random_path,
        desi_true_nz=true_nz_path,
        sigma_true_gas_calibration=tmp_path / "missing_sigma_true.json",
    )


def _content_addressed_map_metadata(fields, config, **extra):
    metadata = {
        "pipeline_version": mp.MEASUREMENT_PIPELINE_VERSION,
        "map_construction_version": mp.MAP_CONSTRUCTION_VERSION,
        "spectrum_estimator_version": mp.SPECTRUM_ESTIMATOR_VERSION,
        "covariance_estimator_version": mp.COVARIANCE_ESTIMATOR_VERSION,
        "config": dict(config.__dict__),
        "input_files": {},
        **extra,
    }
    return mp.map_metadata_with_content_identity(fields, metadata)


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


def test_product_metadata_flags_unvalidated_covariance_and_act_temperature_units():
    missing = mp.missing_inputs_metadata()
    assert missing["act_cmb_temperature_unit_validation"]["present"] is False
    assert missing["non_gaussian_and_mock_covariance_validation"]["present"] is False
    assert "super-sample covariance" in missing["non_gaussian_and_mock_covariance_validation"]["omitted"]


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


def test_midres4096_bandpowers_preserve_the_3000_contract_and_add_three_log_bands():
    old_left, old_right = mp.make_log_bandpower_edges(128, 3000, 13)
    left, right = mp.make_log_bandpower_edges(128, 4096, 16)
    np.testing.assert_array_equal(left[:13], old_left)
    np.testing.assert_array_equal(right[:13], old_right)
    assert left.tolist() == [
        128, 160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315, 1730, 2280, 3001, 3329, 3693
    ]
    assert right.tolist() == [
        160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315, 1730, 2280, 3001, 3329, 3693, 4097
    ]


def test_highres8192_bandpowers_preserve_supported_range_and_add_seven_log_bands():
    old_left, old_right = mp.make_log_bandpower_edges(128, 3000, 13)
    left, right = mp.make_log_bandpower_edges(128, 8192, 20)
    np.testing.assert_array_equal(left[:13], old_left)
    np.testing.assert_array_equal(right[:13], old_right)
    assert left.tolist() == [
        128, 160, 200, 255, 320, 400, 500, 630, 795, 1000,
        1315, 1730, 2280, 3001, 3464, 3998, 4615, 5327, 6149, 7098,
    ]
    assert right.tolist() == [
        160, 200, 255, 320, 400, 500, 630, 795, 1000, 1315,
        1730, 2280, 3001, 3464, 3998, 4615, 5327, 6149, 7098, 8193,
    ]


def test_highres_kappa_packing_keeps_raw_covariance_basis_and_marks_exactly_28_placeholders():
    config = mp.MeasurementConfig.for_stage("highres4096")
    specs = mp.default_spectrum_specs()
    left, right = mp.make_bandpower_edges(config)
    spectra = {
        spec.name: {
            "cl": np.arange(config.n_bins, dtype=np.float64) + 100.0 * i + 1.0,
            "noise_decoupled_all_components": (
                np.full((1, config.n_bins), 0.25 + i, dtype=np.float64)
                if spec.family == "desi_g_auto"
                else None
            ),
        }
        for i, spec in enumerate(specs)
    }
    packed = mp.pack_joint_data_vector(specs, spectra, config, left, right)
    raw = np.concatenate([spectra[spec.name]["cl"] for spec in specs])
    valid = packed["data_vector_valid"]

    assert mp.measurement_schema_for_config(config) == mp.SCHEMA_MEASUREMENT_VALIDITY_MASK
    assert raw.size == 920
    assert np.count_nonzero(~valid) == 28
    assert np.count_nonzero(valid) == 892
    np.testing.assert_array_equal(packed["data_vector_raw"], raw)
    np.testing.assert_array_equal(packed["data_vector"][valid], raw[valid])
    np.testing.assert_array_equal(packed["data_vector"][~valid], 0.0)
    shot_template = packed["galaxy_auto_weighted_poisson_template"]
    subtracted_raw = packed["data_vector_raw_weighted_poisson_subtracted"]
    np.testing.assert_array_equal(subtracted_raw, raw - shot_template)
    assert np.count_nonzero(shot_template) == 80
    non_galaxy = np.ones(raw.size, dtype=bool)
    for index, spec in enumerate(specs):
        if spec.family == "desi_g_auto":
            non_galaxy[index * config.n_bins : (index + 1) * config.n_bins] = False
    np.testing.assert_array_equal(subtracted_raw[non_galaxy], raw[non_galaxy])
    np.testing.assert_array_equal(
        packed["data_vector_weighted_poisson_subtracted"][~valid], 0.0
    )
    for spec in specs:
        expected = np.arange(13) if spec.family == "desi_g_act_kappa" else np.arange(20)
        np.testing.assert_array_equal(np.flatnonzero(packed["spectrum_validity"][spec.name]), expected)


def test_highres_pilot_plot_self_covariance_groups_cover_all_spectra_once():
    config = mp.MeasurementConfig.for_stage("highres4096")
    manifest = prod.build_covariance_manifest(config)
    names = [spec.name for spec in mp.default_spectrum_specs()]
    group_for_name = plot_highres_pilot.required_self_covariance_groups(manifest, names)

    assert set(group_for_name) == set(names)
    assert len(group_for_name) == 46
    assert sorted({int(group["index"]) for group in group_for_name.values()}) == [
        0, 22, 43, 63, 82, 100, 117, 133, 148, 162, 175,
        187, 198, 208, 217, 225, 232, 238, 244, 250, 256, 258,
    ]


def test_highres_pilot_plot_shard_validation_fails_closed_on_identity_tamper(tmp_path):
    config = mp.MeasurementConfig.for_stage("highres4096")
    manifest = prod.build_covariance_manifest(config)
    group = manifest["groups"][0]
    name = str(group["blocks"][0]["spec_i"])
    path = tmp_path / "cov_group_0000_spin2.h5"
    map_product_id = "a" * 64
    block = np.eye(config.n_bins, dtype=np.float64)
    with h5py.File(path, "w") as h5:
        h5.attrs["pipeline_version"] = mp.MEASUREMENT_PIPELINE_VERSION
        h5.attrs["covariance_estimator_version"] = mp.COVARIANCE_ESTIMATOR_VERSION
        h5.attrs["manifest_digest"] = manifest["manifest_digest"]
        h5.attrs["covariance_config_digest"] = manifest["covariance_config_digest"]
        h5.attrs["map_product_id"] = map_product_id
        h5.attrs["group_digest"] = prod._group_digest(group)
        h5.attrs["group_index"] = int(group["index"])
        h5.attrs["group_class"] = str(group["class"])
        h5.attrs["group_json"] = json.dumps(group)
        dataset = h5.create_group("covariance_blocks").create_dataset(
            f"{name}__x__{name}", data=block
        )
        dataset.attrs["spectrum_i"] = name
        dataset.attrs["spectrum_j"] = name

    loaded = plot_highres_pilot._validate_shard_and_read_self_blocks(
        path,
        group,
        [name],
        manifest_digest=str(manifest["manifest_digest"]),
        covariance_config_digest=str(manifest["covariance_config_digest"]),
        map_product_id=map_product_id,
        n_band=config.n_bins,
    )
    np.testing.assert_array_equal(loaded[name], block)

    with h5py.File(path, "r+") as h5:
        h5.attrs["map_product_id"] = "b" * 64
    with pytest.raises(ValueError, match="map_product_id"):
        plot_highres_pilot._validate_shard_and_read_self_blocks(
            path,
            group,
            [name],
            manifest_digest=str(manifest["manifest_digest"]),
            covariance_config_digest=str(manifest["covariance_config_digest"]),
            map_product_id=map_product_id,
            n_band=config.n_bins,
        )


def test_highres_pilot_plot_end_to_end_uses_only_self_covariance_blocks(tmp_path):
    config = mp.MeasurementConfig.for_stage("highres4096")
    config.output_dir = str(tmp_path)
    spectra_path = config.output_root / "synthetic_spectra.h5"
    spectra_path.parent.mkdir(parents=True)
    _write_highres_validity_identity_fixture(spectra_path)
    left, right = mp.make_bandpower_edges(config)
    ell = 0.5 * (left + right - 1.0)
    with h5py.File(spectra_path, "r+") as h5:
        h5["joint"].create_dataset("ell", data=ell)
        for spec in mp.default_spectrum_specs():
            group = h5[f"spectra/{spec.name}"]
            group.attrs["label"] = spec.label
            group.create_dataset("ell", data=ell)
        map_product_id = str(h5.attrs["map_product_id"])

    manifest_path = config.output_root / "manifest.json"
    manifest = prod.write_covariance_manifest(manifest_path, config)
    names = [spec.name for spec in mp.default_spectrum_specs()]
    group_for_name = plot_highres_pilot.required_self_covariance_groups(manifest, names)
    names_by_group = {}
    for name, group in group_for_name.items():
        names_by_group.setdefault(int(group["index"]), []).append(name)
    shard_dir = tmp_path / "self_covariance_shards"
    shard_dir.mkdir()
    for group in manifest["groups"]:
        index = int(group["index"])
        if index not in names_by_group:
            continue
        path = shard_dir / f"cov_group_{index:04d}_{group['class']}.h5"
        with h5py.File(path, "w") as h5:
            h5.attrs["pipeline_version"] = mp.MEASUREMENT_PIPELINE_VERSION
            h5.attrs["covariance_estimator_version"] = mp.COVARIANCE_ESTIMATOR_VERSION
            h5.attrs["manifest_digest"] = manifest["manifest_digest"]
            h5.attrs["covariance_config_digest"] = manifest["covariance_config_digest"]
            h5.attrs["map_product_id"] = map_product_id
            h5.attrs["group_digest"] = prod._group_digest(group)
            h5.attrs["group_index"] = index
            h5.attrs["group_class"] = str(group["class"])
            h5.attrs["group_json"] = json.dumps(group)
            covariance_blocks = h5.create_group("covariance_blocks")
            for name in names_by_group[index]:
                dataset = covariance_blocks.create_dataset(
                    f"{name}__x__{name}",
                    data=np.eye(config.n_bins, dtype=np.float64),
                )
                dataset.attrs["spectrum_i"] = name
                dataset.attrs["spectrum_j"] = name

    spectra, loaded_manifest, shard_records = plot_highres_pilot.load_pilot_spectra_and_errors(
        spectra_path,
        manifest_path,
        shard_dir,
    )
    assert loaded_manifest["manifest_digest"] == manifest["manifest_digest"]
    assert len(spectra) == 46
    assert len(shard_records) == 22
    assert sum(int(np.count_nonzero(spec.valid)) for spec in spectra) == 892
    assert all(np.array_equal(spec.sigma_cl, np.ones(config.n_bins)) for spec in spectra)

    outputs = plot_highres_pilot.plot_pilot_dell(
        spectra,
        tmp_path / "plots",
        stage_label="synthetic pilot",
        ksz_scale=1000.0,
        show_kappa_null_diagnostics=True,
    )
    assert len(outputs) == 8
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)

    value_only_spectra = plot_highres_pilot.load_pilot_spectra_values(spectra_path)
    assert len(value_only_spectra) == 46
    assert all(spec.sigma_cl is None for spec in value_only_spectra)
    value_only_outputs = plot_highres_pilot.plot_pilot_dell(
        value_only_spectra,
        tmp_path / "plots_value_only",
        stage_label="synthetic spectra-only pilot",
        ksz_scale=1000.0,
        show_kappa_null_diagnostics=True,
    )
    assert len(value_only_outputs) == 8
    assert all(path.is_file() and path.stat().st_size > 0 for path in value_only_outputs)


def _write_highres_validity_identity_fixture(path: Path) -> None:
    config = mp.MeasurementConfig.for_stage("highres4096")
    config.compute_covariance = False
    fields = {
        "g1": mp.FieldMap(
            name="g1",
            label="synthetic",
            kind="desi_galaxy",
            spin=0,
            maps=[np.zeros(12, dtype=np.float32)],
            mask=np.ones(12, dtype=np.float32),
            mask_name="mask",
        )
    }
    map_metadata = _content_addressed_map_metadata(fields, config)
    specs = mp.default_spectrum_specs()
    left, right = mp.make_bandpower_edges(config)
    spectra = {
        spec.name: {
            "cl": np.zeros(config.n_bins),
            "noise_decoupled_all_components": (
                np.zeros((1, config.n_bins)) if spec.family == "desi_g_auto" else None
            ),
        }
        for spec in specs
    }
    packed = mp.pack_joint_data_vector(specs, spectra, config, left, right)
    starts = np.arange(len(specs), dtype=np.int64) * config.n_bins
    string_dtype = h5py.string_dtype("utf-8")

    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MEASUREMENT_VALIDITY_MASK
        h5.attrs["pipeline_version"] = mp.MEASUREMENT_PIPELINE_VERSION
        h5.attrs["map_construction_version"] = mp.MAP_CONSTRUCTION_VERSION
        h5.attrs["spectrum_estimator_version"] = mp.SPECTRUM_ESTIMATOR_VERSION
        h5.attrs["covariance_estimator_version"] = mp.COVARIANCE_ESTIMATOR_VERSION
        h5.attrs["desi_galaxy_auto_mean_convention"] = mp.DESI_GALAXY_AUTO_MEAN_CONVENTION
        h5.attrs["map_product_id"] = map_metadata["map_product_id"]
        h5.attrs["map_metadata_json"] = json.dumps(map_metadata)
        h5.attrs["config_json"] = json.dumps(config.to_dict())
        h5.create_dataset("ell_left", data=left)
        h5.create_dataset("ell_right", data=right)
        joint = h5.create_group("joint")
        joint.create_dataset(
            "spectrum_names",
            data=np.asarray([spec.name for spec in specs], dtype=string_dtype),
        )
        joint.create_dataset("slice_start", data=starts)
        joint.create_dataset("slice_stop", data=starts + config.n_bins)
        joint.create_dataset("data_vector", data=packed["data_vector"])
        joint.create_dataset("data_vector_raw", data=packed["data_vector_raw"])
        joint.create_dataset("data_vector_valid", data=packed["data_vector_valid"])
        spectra_group = h5.create_group("spectra")
        for spec in specs:
            group = spectra_group.create_group(spec.name)
            group.attrs["family"] = spec.family
            group.create_dataset("cl", data=spectra[spec.name]["cl"])
            group.create_dataset(
                "data_vector_valid",
                data=packed["spectrum_validity"][spec.name],
            )


@pytest.mark.parametrize("tamper", ["name_order", "slice_order", "vector_size", "cl_shape"])
def test_highres_measurement_identity_rejects_noncanonical_archive_layout(tmp_path, tamper):
    path = tmp_path / f"validity_{tamper}.h5"
    _write_highres_validity_identity_fixture(path)
    with h5py.File(path, "r") as h5:
        mp.validate_measurement_product_identity(h5)

    with h5py.File(path, "r+") as h5:
        if tamper == "name_order":
            names = h5["joint/spectrum_names"][:]
            names[[0, 1]] = names[[1, 0]]
            h5["joint/spectrum_names"][:] = names
        elif tamper == "slice_order":
            starts = h5["joint/slice_start"][:]
            stops = h5["joint/slice_stop"][:]
            starts[[0, 1]] = starts[[1, 0]]
            stops[[0, 1]] = stops[[1, 0]]
            h5["joint/slice_start"][:] = starts
            h5["joint/slice_stop"][:] = stops
        elif tamper == "vector_size":
            for name in ("data_vector", "data_vector_raw", "data_vector_valid"):
                values = h5[f"joint/{name}"][:-1]
                del h5[f"joint/{name}"]
                h5["joint"].create_dataset(name, data=values)
        else:
            name = mp.default_spectrum_specs()[0].name
            del h5[f"spectra/{name}/cl"]
            h5[f"spectra/{name}"].create_dataset("cl", data=np.zeros(19))

    with h5py.File(path, "r") as h5, pytest.raises(ValueError, match="canonical|cl shape"):
        mp.validate_measurement_product_identity(h5)


def test_measurement_loader_marginalizes_saved_placeholders_by_principal_submatrix(tmp_path, monkeypatch):
    path = tmp_path / "validity_product.h5"
    names = ["desi_g_act_kappa_pz1", "desi_g_auto_pz1"]
    valid = np.asarray([True, True, False, True, True, True])
    packed = np.asarray([1.0, 2.0, 0.0, 4.0, 5.0, 6.0])
    cov = np.diag(np.arange(1.0, 7.0))
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MEASUREMENT_VALIDITY_MASK
        joint = h5.create_group("joint")
        joint.create_dataset("spectrum_names", data=np.asarray(names, dtype=h5py.string_dtype("utf-8")))
        joint.create_dataset("slice_start", data=np.asarray([0, 3]))
        joint.create_dataset("slice_stop", data=np.asarray([3, 6]))
        joint.create_dataset("ell", data=np.asarray([100.0, 200.0, 400.0]))
        joint.create_dataset("data_vector", data=packed)
        joint.create_dataset("data_vector_valid", data=valid)
        joint.create_dataset("cov", data=cov)
        h5.create_dataset("ell_left", data=np.asarray([80, 160, 320]))
        h5.create_dataset("ell_right", data=np.asarray([160, 320, 500]))
        spectra = h5.create_group("spectra")
        for name, family in zip(names, ["desi_g_act_kappa", "desi_g_auto"]):
            group = spectra.create_group(name)
            group.attrs["family"] = family
            group.attrs["label"] = name
            group.attrs["theory_key"] = name
    monkeypatch.setattr(mp, "validate_measurement_product_identity", lambda *_args, **_kwargs: "map")

    active = gmt.load_measurement_data(path)
    np.testing.assert_array_equal(active.data_vector, [1.0, 2.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(active.covariance, cov[np.ix_(valid, valid)])
    np.testing.assert_array_equal(active.starts, [0, 2])
    np.testing.assert_array_equal(active.stops, [2, 5])
    np.testing.assert_array_equal(active.archive_indices, [0, 1, 3, 4, 5])
    np.testing.assert_array_equal(active.ell, [100.0, 200.0, 100.0, 200.0, 400.0])

    archive = gmt.load_measurement_data(path, include_invalid_placeholders=True)
    np.testing.assert_array_equal(archive.data_vector, packed)
    np.testing.assert_array_equal(archive.data_vector_valid, valid)
    stats = gmt.comparison_statistics(archive, np.zeros_like(packed))
    expected_chi2 = float(np.sum(packed[valid] ** 2 / np.diag(cov)[valid]))
    assert stats["full"]["chi2"] == pytest.approx(expected_chi2)
    assert stats["full"]["ndof"] == 5


def test_kappa_packing_rejects_a_band_that_straddles_the_response_cutoff():
    config = mp.MeasurementConfig.for_stage("highres4096")
    specs = mp.default_spectrum_specs()
    left, right = mp.make_bandpower_edges(config)
    left = left.copy()
    right = right.copy()
    right[12] = 3100
    left[13] = 3100
    spectra = {spec.name: {"cl": np.ones(config.n_bins)} for spec in specs}
    with pytest.raises(ValueError, match="exact right-exclusive band edge"):
        mp.pack_joint_data_vector(specs, spectra, config, left, right)


def test_cpu_production_stage_configs_and_versioned_paths():
    fast = mp.MeasurementConfig.for_stage("fast1024")
    assert fast.nside == 1024
    assert fast.lmax == 1024
    assert fast.n_bins == 10
    assert fast.binning == "linear"
    assert fast.act_downgrade == 1
    assert fast.default_measurement_path.name == (
        "xdesi_multiprobe_cls_cov_nside1024_lmax1024_nbin10_linear_pipev2_gshot.h5"
    )
    assert fast.default_maps_path.name == "xdesi_multiprobe_maps_nside1024_lmax1024_nbin10_linear_pipev2.h5"
    assert prod.manifest_path(fast).name == "covariance_manifest_nside1024_lmax1024_nbin10_linear_pipev2.json"

    mid = mp.MeasurementConfig.for_stage("midres2048")
    assert mid.nside == 2048
    assert mid.lmax == 4096
    assert mid.effective_lmax_mask == 6143
    assert mid.ell_min == 128
    assert mid.n_bins == 16
    assert mid.binning == "log"
    assert mid.act_downgrade == 1
    assert mid.mask_apodization_deg == 0.0
    assert mid.pair_overlap_mean_subtract is False
    assert mid.default_measurement_path.name == (
        "xdesi_multiprobe_cls_cov_nside2048_ell128_lmax4096_lmask6143_nbin16_log_pipev2_gshot.h5"
    )
    assert mid.default_maps_path.name == (
        "xdesi_multiprobe_maps_nside2048_ell128_lmax4096_lmask6143_nbin16_log_pipev2.h5"
    )
    assert prod.manifest_path(mid).name == (
        "covariance_manifest_nside2048_ell128_lmax4096_lmask6143_nbin16_log_pipev2.json"
    )

    high = mp.MeasurementConfig.for_stage("highres4096")
    assert high.nside == 4096
    assert high.lmax == 8192
    assert high.effective_lmax_mask == 12287
    assert high.n_bins == 20
    assert high.kappa_cmb_lmax == 3000
    assert high.minimum_desi_random_realizations == 8
    assert high.act_cmb_temperature_units_confirmed is True
    assert high.default_measurement_path.name == (
        "xdesi_multiprobe_cls_cov_nside4096_ell128_lmax8192_lmask12287_nbin20_log_"
        "pipev2_gshot_gkell3000_dvvalidv1.h5"
    )
    assert prod.manifest_path(high).name == (
        "covariance_manifest_nside4096_ell128_lmax8192_lmask12287_nbin20_log_pipev2.json"
    )


def test_standalone_plot_defaults_resolve_pipeline_v2_products():
    fast = plot_measurement.default_measurement_for_stage("fast1024", root="/repo")
    mid = plot_measurement.default_measurement_for_stage("midres2048", root="/repo")
    assert fast.name == "xdesi_multiprobe_cls_cov_nside1024_lmax1024_nbin10_linear_pipev2_gshot.h5"
    assert mid.name == (
        "xdesi_multiprobe_cls_cov_nside2048_ell128_lmax4096_lmask6143_nbin16_log_pipev2_gshot.h5"
    )


def test_measurement_cl_and_dell_plot_values_scale_means_and_errors_consistently():
    ell = np.asarray([10.0, 20.0])
    cl = np.asarray([-2.0, 3.0])
    err = np.asarray([0.5, 0.25])

    cl_y, cl_err, cl_label = gmt.measurement_plot_values(
        ell, cl, err, family="desi_pi_act_T", quantity="cl"
    )
    np.testing.assert_array_equal(cl_y, cl)
    np.testing.assert_array_equal(cl_err, err)
    assert cl_label == r"$C_\ell^{\pi T}$"

    dell_y, dell_err, dell_label = gmt.measurement_plot_values(
        ell, cl, err, family="desi_pi_act_T", quantity="dell"
    )
    factor = gmt.dell_factor(ell)
    np.testing.assert_allclose(dell_y, -factor * cl)
    np.testing.assert_allclose(dell_err, factor * err)
    assert dell_label == r"$-D_\ell^{\pi T}$"

    scaled_cl_y, scaled_cl_err, scaled_cl_label = gmt.measurement_plot_values(
        ell,
        cl,
        err,
        family="desi_pi_act_T",
        quantity="cl",
        ksz_scale=1.0e3,
    )
    np.testing.assert_array_equal(scaled_cl_y, 1.0e3 * cl)
    np.testing.assert_array_equal(scaled_cl_err, 1.0e3 * err)
    assert scaled_cl_label == r"$10^{3}\,C_\ell^{\pi T}$"

    with pytest.raises(ValueError, match="ksz_scale must be finite and positive"):
        gmt.measurement_plot_values(
            ell,
            cl,
            err,
            family="desi_pi_act_T",
            quantity="dell",
            ksz_scale=-1.0,
        )


def test_ell_axis_configuration_keeps_log_limits_positive_and_ordered():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        ax.plot([128.0, 8192.0], [0.0, 1.0])
        gmt._configure_ell_axis(
            ax,
            np.asarray([128.0, 8192.0]),
            ell_left=np.asarray([128.0, 7098.0]),
            xscale="log",
            ell_max=8192.0,
            xlim=None,
        )
        left, right = ax.get_xlim()
        assert ax.get_xscale() == "log"
        assert left == pytest.approx(128.0)
        assert right == pytest.approx(8192.0)
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        ax.plot([128.0, 8192.0], [0.0, 1.0])
        left_before = ax.get_xlim()[0]
        gmt._configure_ell_axis(
            ax,
            np.asarray([128.0, 8192.0]),
            ell_left=np.asarray([128.0, 7098.0]),
            xscale="linear",
            ell_max=8192.0,
            xlim=None,
        )
        left, right = ax.get_xlim()
        assert ax.get_xscale() == "linear"
        assert left == pytest.approx(left_before)
        assert right == pytest.approx(8192.0)
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match=r"requires xlim\[0\] > 0"):
            gmt._configure_ell_axis(
                ax,
                np.asarray([128.0, 8192.0]),
                ell_left=np.asarray([128.0, 7098.0]),
                xscale="log",
                ell_max=None,
                xlim=(0.0, 8192.0),
            )
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        ax.plot([128.0, 8192.0], [0.0, 1.0])
        gmt._configure_ell_axis(
            ax,
            np.asarray([128.0, 8192.0]),
            ell_left=np.asarray([128.0, 7098.0]),
            xscale="log",
            ell_max=8192.0,
            xlim=(100.0, 4096.0),
        )
        assert ax.get_xlim() == pytest.approx((100.0, 4096.0))
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="finite positive band support"):
            gmt._configure_ell_axis(
                ax,
                np.asarray([], dtype=np.float64),
                ell_left=None,
                xscale="log",
                ell_max=8192.0,
                xlim=None,
            )
    finally:
        plt.close(fig)


def test_measurement_plotter_writes_cl_and_dell_for_every_probe_family(
    tmp_path,
    monkeypatch,
):
    all_specs = mp.default_spectrum_specs()
    specs_by_family = {}
    for spec in all_specs:
        specs_by_family.setdefault(spec.family, spec)
    specs = list(specs_by_family.values())
    assert len(all_specs) == 46
    assert len(specs) == 7
    n_band = 2
    names = [spec.name for spec in specs]
    starts = np.arange(len(specs), dtype=np.int64) * n_band
    stops = starts + n_band
    n_data = len(specs) * n_band
    measurement = gmt.MeasurementData(
        path=tmp_path / "synthetic_pipev2.h5",
        names=names,
        ell=np.asarray([143.5, 447.0]),
        data_vector=np.linspace(-1.0e-6, 1.0e-6, n_data),
        covariance=np.eye(n_data, dtype=np.float64) * 1.0e-14,
        starts=starts,
        stops=stops,
        families={spec.name: spec.family for spec in specs},
        labels={spec.name: spec.label for spec in specs},
        theory_keys={spec.name: spec.theory_key for spec in specs},
        ell_left=np.asarray([128.0, 400.0]),
        ell_right=np.asarray([160.0, 500.0]),
    )
    plot_dir = tmp_path / "plots"
    cl_pdf = plot_dir / "all_cl.pdf"
    dell_pdf = plot_dir / "all_dell.pdf"

    observed_axes = []
    configure_ell_axis = gmt._configure_ell_axis

    def record_ell_axis(ax, ell, **kwargs):
        configure_ell_axis(ax, ell, **kwargs)
        observed_axes.append((ax.get_xscale(), ax.get_xlim()))

    monkeypatch.setattr(gmt, "_configure_ell_axis", record_ell_axis)

    cl_outputs = gmt.plot_measurement_cl(
        measurement,
        plot_dir,
        pdf_path=cl_pdf,
        ell_max=8192.0,
        xscale="log",
    )
    dell_outputs = gmt.plot_measurement_dell(
        measurement,
        plot_dir,
        pdf_path=dell_pdf,
        ell_max=8192.0,
        xscale="log",
    )

    assert len(cl_outputs) == 7
    assert len(dell_outputs) == 7
    assert cl_pdf.is_file()
    assert dell_pdf.is_file()
    assert all(path.is_file() for path in [*cl_outputs, *dell_outputs])
    assert len(observed_axes) == 14
    assert all(scale == "log" for scale, _ in observed_axes)
    assert all(limits == pytest.approx((128.0, 8192.0)) for _, limits in observed_axes)


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


def test_build_shear_fields_requires_mask_matched_catalog_noise(tmp_path):
    shear_path = tmp_path / "shear.h5"
    _write_minimal_shear_maps(shear_path, nside=1)
    bundle = _FakeShearBundle(shear_path)

    raw_config = mp.MeasurementConfig(
        nside=1,
        lmax=3,
        shear_mask_dataset="mask_weight_raw",
        shear_noise_attr="shape_noise_pseudo_cl_raw_weight_mask",
    )
    fields = mp.build_shear_fields(bundle, raw_config)
    assert fields["s1"].metadata["shape_noise_pseudo_cl"] == 0.4

    mismatched = mp.MeasurementConfig(
        nside=1,
        lmax=3,
        shear_mask_dataset="mask_weight_raw",
        shear_noise_attr="shape_noise_pseudo_cl_normalized_weight_mask",
    )
    with pytest.raises(ValueError, match="requires noise attribute"):
        mp.build_shear_fields(bundle, mismatched)


def test_mask_apodization_preserves_catalog_masks_but_can_apply_to_act_y(monkeypatch):
    mask = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    shear = mp.FieldMap(
        name="s1",
        label="s1",
        kind="des_shear",
        spin=2,
        maps=[np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)],
        mask=mask,
        mask_name="shear",
        metadata={"shape_noise_pseudo_cl": 0.1},
    )
    galaxy = mp.FieldMap(
        name="g1",
        label="g1",
        kind="desi_galaxy",
        spin=0,
        maps=[np.zeros(4, dtype=np.float32)],
        mask=mask,
        mask_name="desi",
    )
    act_y = mp.FieldMap(
        name="y",
        label="y",
        kind="act_tsz_y",
        spin=0,
        maps=[np.zeros(4, dtype=np.float32)],
        mask=mask,
        mask_name="act_y",
    )
    calls = []

    def fake_apodization(values, aposize, apotype):
        calls.append((np.asarray(values).copy(), aposize, apotype))
        return 0.5 * np.asarray(values)

    monkeypatch.setattr(mp.nmt, "mask_apodization", fake_apodization)
    config = mp.MeasurementConfig(mask_apodization_deg=1.0, mask_apodization_type="C2")
    out = mp.apply_mask_apodization({"s1": shear, "g1": galaxy, "y": act_y}, config)

    np.testing.assert_array_equal(out["s1"].mask, mask)
    np.testing.assert_array_equal(out["g1"].mask, mask)
    np.testing.assert_allclose(out["y"].mask, 0.5 * mask)
    assert out["s1"].metadata["mask_apodization_applied"] is False
    assert out["g1"].metadata["mask_apodization_applied"] is False
    assert out["y"].metadata["mask_apodization_applied"] is True
    assert len(calls) == 1


def test_act_masks_use_bounded_spline_reprojection(monkeypatch):
    map_em = object()
    mask_em = object()
    calls = []

    def fake_read(path, dataset, header_attr, downgrade=1):
        return mask_em if dataset == "mask" else map_em

    def fake_reproject(values, **kwargs):
        calls.append((values, dict(kwargs)))
        if values is mask_em:
            return np.array([-0.2, 0.25, 1.2, 0.0], dtype=np.float64)
        return np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)

    monkeypatch.setattr(mp, "read_enmap_from_h5", fake_read)
    monkeypatch.setattr(mp.reproject, "map2healpix", fake_reproject)
    sky, mask = mp.enmap_h5_to_healpix(
        Path("unused.h5"),
        "map",
        "map_header",
        "mask",
        "mask_header",
        nside=1,
        lmax=3,
        downgrade=1,
        subtract_mean=False,
    )

    np.testing.assert_allclose(mask, [0.0, 0.25, 1.0, 0.0])
    np.testing.assert_allclose(sky, [0.0, 6.0, 7.0, 0.0])
    mask_call = next(kwargs for values, kwargs in calls if values is mask_em)
    map_call = next(kwargs for values, kwargs in calls if values is map_em)
    assert mask_call == {"nside": 1, "method": "spline", "order": 1, "spin": [0]}
    assert map_call == {"nside": 1, "lmax": 3}


def test_premasked_mean_subtraction_preserves_one_mask_factor():
    mask = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    premasked = np.array([0.0, 1.0, 4.0], dtype=np.float32)
    result = mp._subtract_premasked_mean(premasked, mask)
    assert result[0] == 0.0
    np.testing.assert_allclose(np.sum(result), 0.0, atol=1.0e-7)
    ratio = (premasked[1:] - result[1:]) / mask[1:]
    np.testing.assert_allclose(ratio[0], ratio[1])


def test_build_nmt_fields_honors_premasked_input_flag(monkeypatch):
    captured = []

    def fake_field(mask, maps, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(is_catalog=False)

    monkeypatch.setattr(mp.nmt, "NmtField", fake_field)
    field = mp.FieldMap(
        name="kappa",
        label="kappa",
        kind="act_cmb_lensing_kappa",
        spin=0,
        maps=[np.ones(12, dtype=np.float32)],
        mask=np.ones(12, dtype=np.float32),
        mask_name="kappa",
        metadata={"namaster_masked_on_input": True},
    )
    mp.build_nmt_fields({"kappa": field}, mp.MeasurementConfig(nside=2, lmax=3, lmax_mask=5))
    assert captured[0]["masked_on_input"] is True
    assert captured[0]["lmax"] == 3
    assert captured[0]["lmax_mask"] == 5


def test_shear_auto_noise_uses_saved_catalog_value_not_observed_pcl():
    info = mp.FieldMap(
        name="s1",
        label="s1",
        kind="des_shear",
        spin=2,
        maps=[np.zeros(12, dtype=np.float32), np.zeros(12, dtype=np.float32)],
        mask=np.ones(12, dtype=np.float32),
        mask_name="unit",
        metadata={"shape_noise_pseudo_cl": 0.25, "mask_apodization_applied": False},
    )
    fields = {"s1": mp.NmtProbeField(info=info, field=None)}
    workspace = SimpleNamespace(decouple_cell=lambda values: np.asarray(values) * 2.0)
    config = mp.MeasurementConfig(nside=1, lmax=3)

    coupled_a, decoupled_a = mp.coupled_noise_for_field_pair(
        "s1", "s1", fields, workspace, config, pcl=np.full((4, 4), 100.0)
    )
    coupled_b, decoupled_b = mp.coupled_noise_for_field_pair(
        "s1", "s1", fields, workspace, config, pcl=np.full((4, 4), -100.0)
    )

    np.testing.assert_allclose(coupled_a[[0, 3]], 0.25)
    np.testing.assert_allclose(coupled_a[[1, 2]], 0.0)
    np.testing.assert_array_equal(coupled_a, coupled_b)
    np.testing.assert_array_equal(decoupled_a, decoupled_b)


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
    assert mp.MeasurementConfig().covariance_input_mode == "inka_data"
    config = mp.MeasurementConfig(covariance_input_mode="unsupported")
    with pytest.raises(ValueError, match="Unsupported covariance_input_mode"):
        mp.compute_input_cl_for_covariance("a", "b", {}, None, {}, {}, config)


def test_map_covariance_input_uses_namaster_inka_cell(monkeypatch):
    fields = {
        "a": _probe_for_covariance_test("a", "scalar_a", 0),
        "b": _probe_for_covariance_test("b", "scalar_b", 0),
    }
    fields["a"].field = object()
    fields["b"].field = object()
    expected = np.arange(6, dtype=np.float64).reshape(1, 6)
    calls = []

    def fake_inka(field_a, field_b):
        calls.append((field_a, field_b))
        return expected

    monkeypatch.setattr(mp.nmt, "get_iNKA_cell", fake_inka)
    cache = {}
    config = mp.MeasurementConfig(nside=2, lmax=3, covariance_input_mode="inka_data")
    result = mp.compute_input_cl_for_covariance("a", "b", fields, None, {}, cache, config)

    np.testing.assert_array_equal(result, expected[:, :4])
    assert calls == [(fields["a"].field, fields["b"].field)]
    assert ("inka_data", "a", "b") in cache


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


def test_multi_random_loader_binds_manifest_identity_indices_and_map_checksum(tmp_path):
    path = tmp_path / "random_counts_v3.h5"
    values = np.arange(1, hp.nside2npix(1) + 1, dtype=np.uint32)
    indices = [0, 1, 10, 11, 12, 13, 14, 15]
    identity = "a" * 64
    schema = "desi-dr9-extended-lrg-random-mask-v3"
    ledger_sha256 = "b" * 64
    full_source_sha256 = {
        **{
            f"randoms/resolve/randoms-1-{index}.fits": "c" * 64
            for index in indices
        },
        **{
            (
                "zhou-lrg-xcorr-2023-v1/catalogs/lrgmask_v1.1/"
                f"randoms-1-{index}-lrgmask_v1.1.fits.gz"
            ): "d" * 64
            for index in indices
        },
        (
            "zhou-lrg-xcorr-2023-v1/misc/"
            "pixweight-dr7.1-0.22.0_stardens_64_ring.fits"
        ): "e" * 64,
        "zhou-lrg-xcorr-2023-v1/catalogs/randoms_quality_cuts.py": "f" * 64,
    }
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            full_source_sha256, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    with h5py.File(path, "w") as h5:
        h5.attrs["ordering"] = "RING"
        h5.attrs["n_random_realizations"] = 8
        h5.attrs["random_realization_count"] = 8
        h5.attrs["random_realization_indices"] = np.asarray(indices, dtype=np.int32)
        h5.attrs["input_identity_sha256"] = identity
        h5.attrs["schema_version"] = schema
        h5.attrs["full_source_sha256_verified"] = 1
        h5.attrs["sha256_ledger_sha256"] = ledger_sha256
        h5.attrs["full_source_sha256_json"] = json.dumps(full_source_sha256)
        group = h5.create_group("nside1")
        group.attrs["random_count_sha256"] = mp._array_raw_sha256(values)
        group.attrs["count_sum"] = int(np.sum(values, dtype=np.uint64))
        group.create_dataset("random_count", data=values)
    manifest_randoms = {
        "random_realization_count": 8,
        "random_indices": indices,
        "input_identity_sha256": identity,
        "random_product_schema": schema,
        "sha256_ledger_sha256": ledger_sha256,
        "full_source_sha256_inventory_sha256": inventory_sha256,
    }
    bundle = SimpleNamespace(
        desi_random_count_maps=path,
        manifest={"products": {"desi_dr9_imaging_randoms": manifest_randoms}},
    )
    config = mp.MeasurementConfig(
        stage="synthetic-highres",
        nside=1,
        lmax=3,
        minimum_desi_random_realizations=8,
    )
    counts, metadata = mp.load_dr9_random_counts_with_metadata(bundle, config)
    np.testing.assert_array_equal(counts, values.astype(np.float32))
    assert metadata["random_realization_indices"] == indices
    assert metadata["random_count_input_identity_sha256"] == identity
    assert metadata["random_count_sha256"] == mp._array_raw_sha256(values)
    assert metadata["full_source_sha256_verified"] is True
    assert metadata["sha256_ledger_sha256"] == ledger_sha256
    assert metadata["full_source_sha256"] == full_source_sha256

    for key, stale_value in (
        ("full_source_sha256_verified", 0),
        ("sha256_ledger_sha256", ""),
        ("full_source_sha256_json", "{}"),
    ):
        with h5py.File(path, "r+") as h5:
            original = h5.attrs[key]
            h5.attrs[key] = stale_value
        with pytest.raises(ValueError, match="full_source_sha256|sha256_ledger_sha256"):
            mp.load_dr9_random_counts_with_metadata(bundle, config)
        with h5py.File(path, "r+") as h5:
            h5.attrs[key] = original

    incomplete_inventory = dict(full_source_sha256)
    incomplete_inventory.pop("randoms/resolve/randoms-1-15.fits")
    with h5py.File(path, "r+") as h5:
        h5.attrs["full_source_sha256_json"] = json.dumps(incomplete_inventory)
    with pytest.raises(ValueError, match="exact selected random/mask"):
        mp.load_dr9_random_counts_with_metadata(bundle, config)
    invalid_inventory = dict(full_source_sha256)
    invalid_inventory["randoms/resolve/randoms-1-15.fits"] = "not-a-sha256"
    with h5py.File(path, "r+") as h5:
        h5.attrs["full_source_sha256_json"] = json.dumps(invalid_inventory)
    with pytest.raises(ValueError, match="invalid full-source SHA256"):
        mp.load_dr9_random_counts_with_metadata(bundle, config)
    with h5py.File(path, "r+") as h5:
        h5.attrs["full_source_sha256_json"] = json.dumps(full_source_sha256)

    manifest_randoms["sha256_ledger_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ledger identity does not match"):
        mp.load_dr9_random_counts_with_metadata(bundle, config)
    manifest_randoms["sha256_ledger_sha256"] = ledger_sha256
    manifest_randoms["full_source_sha256_inventory_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source inventory does not match"):
        mp.load_dr9_random_counts_with_metadata(bundle, config)
    manifest_randoms["full_source_sha256_inventory_sha256"] = inventory_sha256

    manifest_randoms["random_indices"] = indices[:-1] + [16]
    with pytest.raises(ValueError, match="do not match the manifest set"):
        mp.load_dr9_random_counts_with_metadata(bundle, config)
    manifest_randoms["random_indices"] = indices
    with h5py.File(path, "r+") as h5:
        h5["nside1/random_count"][0] += 1
    with pytest.raises(ValueError, match="failed its content checksum"):
        mp.load_dr9_random_counts_with_metadata(bundle, config)


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
            "desi_dr9_redshift_distributions": {
                "extended_lrg_sigmaz0p05_true_nz_hdf5": "desi_dr9_true_nz.h5",
            },
            "act_dr6_tsz_compton_y": "act_y.h5",
            "act_dr6_cmb_temperature": "act_T.h5",
            "act_dr6_lensing_kappa": "act_kappa.h5",
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _write_minimal_dr9_catalog(tmp_path / "desi_dr9_catalog.h5")
    _write_minimal_true_nz(tmp_path / "desi_dr9_true_nz.h5")
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
    np.testing.assert_allclose(meta["shot_noise_pseudo_cl"], expected_shot)
    assert meta["nbar_per_sr"] == meta["sum_weight"] / meta["area_sr"]

    vr = np.array([0.001, -0.002])
    weights = np.array([2.0, 1.0])
    expected_rms = np.sqrt(np.sum(weights * vr**2) / np.sum(weights))
    np.testing.assert_allclose(meta["rms_rec_vr_over_c"], expected_rms, rtol=1e-6)
    np.testing.assert_allclose(meta["rms_rec_vr_over_c_weighted"], expected_rms, rtol=1e-6)

    z_mid = summary["z_mid"]
    nz = summary["nz_dndz_by_pz"][0]
    np.testing.assert_allclose(np.sum(nz * summary["dz"]), 1.0)


def test_desi_variable_selection_shot_noise_template_is_coupled_pseudo_cl(tmp_path):
    nside = 1
    bundle = _fake_bundle(tmp_path, nside=nside)
    config = mp.MeasurementConfig(nside=nside, lmax=3, include_ksz_velocity_shuffle=False)
    random_counts = np.arange(1, hp.nside2npix(nside) + 1, dtype=np.float64)
    fields, _ = mp.build_desi_fields(bundle, config, random_counts)
    meta = fields["g1"].metadata
    alpha = float(meta["alpha_galaxy_to_random"])
    random_mean = float(meta["random_counts_mean_valid"])
    expected = (
        hp.nside2pixarea(nside)
        * float(meta["sum_weight2"])
        / (hp.nside2npix(nside) * (alpha * random_mean) ** 2)
    )
    np.testing.assert_allclose(meta["shot_noise_pseudo_cl"], expected)

    class Workspace:
        def couple_cell(self, values):
            raise AssertionError("variable-mask galaxy shot noise must not be coupled a second time")

        def decouple_cell(self, values):
            return 2.0 * np.asarray(values)

    probe = mp.NmtProbeField(info=fields["g1"], field=None)
    coupled, decoupled = mp.coupled_noise_for_field_pair(
        "g1", "g1", {"g1": probe}, Workspace(), config
    )
    np.testing.assert_allclose(coupled, expected)
    np.testing.assert_allclose(decoupled, 2.0 * expected)


def test_measure_spectrum_keeps_galaxy_shot_noise_but_not_shear_or_cross_noise_policy(monkeypatch):
    config = mp.MeasurementConfig(nside=1, lmax=3, ell_min=0, n_bins=1)
    mask = np.ones(hp.nside2npix(1), dtype=np.float32)

    def probe(name, kind, metadata):
        info = mp.FieldMap(
            name=name,
            label=name,
            kind=kind,
            spin=0,
            maps=[np.zeros_like(mask)],
            mask=mask,
            mask_name=f"{name}_mask",
            metadata=metadata,
        )
        return mp.NmtProbeField(info=info, field=object())

    fields = {
        "g1": probe("g1", "desi_galaxy", {"shot_noise_pseudo_cl": 2.0}),
        "s0": probe(
            "s0",
            "des_shear",
            {"shape_noise_pseudo_cl": 2.0, "mask_apodization_applied": False},
        ),
        "y": probe("y", "act_tsz_y", {}),
    }

    class Workspace:
        def decouple_cell(self, values, cl_noise=None):
            values = np.asarray(values, dtype=np.float64)
            if cl_noise is not None:
                values = values - np.asarray(cl_noise, dtype=np.float64)
            return values[:, :1]

        def get_bandpower_windows(self):
            return np.ones((1, 1, 1, config.lmax + 1), dtype=np.float64)

    workspace = Workspace()
    bins = SimpleNamespace(get_effective_ells=lambda: np.asarray([1.0]))
    monkeypatch.setattr(mp, "get_workspace", lambda *args, **kwargs: workspace)
    monkeypatch.setattr(
        mp.nmt,
        "compute_coupled_cell",
        lambda *args, **kwargs: np.full((1, config.lmax + 1), 10.0, dtype=np.float64),
    )

    def spec(name, family, pair):
        return mp.SpectrumSpec(
            name=name,
            family=family,
            fields=pair,
            component=0,
            label=name,
            theory_key=name,
        )

    galaxy = mp.measure_spectrum(
        spec("desi_g_auto_pz1", "desi_g_auto", ("g1", "g1")),
        fields,
        bins,
        {},
        config,
    )
    shear = mp.measure_spectrum(
        spec("des_shear_EE_tomo1_tomo1", "des_shear_EE", ("s0", "s0")),
        fields,
        bins,
        {},
        config,
    )
    cross = mp.measure_spectrum(
        spec("desi_g_act_y_pz1", "desi_g_act_y", ("g1", "y")),
        fields,
        bins,
        {},
        config,
    )

    np.testing.assert_array_equal(galaxy["cl"], [10.0])
    np.testing.assert_array_equal(galaxy["noise_decoupled_all_components"], [[2.0]])
    np.testing.assert_array_equal(galaxy["cl"], shear["cl"] + galaxy["noise_decoupled_all_components"][0])
    np.testing.assert_array_equal(shear["cl"], [8.0])
    np.testing.assert_array_equal(cross["cl"], [10.0])


def test_galaxy_covariance_input_uses_raw_total_inka_auto(monkeypatch):
    field = _probe_for_covariance_test("g1", "desi_galaxy", 0)
    field.field = object()
    expected_total = np.asarray([[7.0, 8.0, 9.0, 10.0]], dtype=np.float64)
    monkeypatch.setattr(mp.nmt, "get_iNKA_cell", lambda *args: expected_total)
    monkeypatch.setattr(
        mp,
        "coupled_noise_for_field_pair",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("default iNKA covariance must not subtract or restore shot noise")
        ),
    )
    config = mp.MeasurementConfig(nside=2, lmax=3, covariance_input_mode="inka_data")
    result = mp.compute_input_cl_for_covariance(
        "g1",
        "g1",
        {"g1": field},
        None,
        {},
        {},
        config,
    )
    np.testing.assert_array_equal(result, expected_total)


def test_ksz_catalog_momentum_roundtrips_through_map_product(tmp_path):
    nside = 2
    bundle = _fake_bundle(tmp_path, nside=nside)
    config = mp.MeasurementConfig(nside=nside, lmax=3, include_ksz_velocity_shuffle=True)
    random_counts = np.ones(hp.nside2npix(nside), dtype=np.float32)
    fields, summary = mp.build_desi_fields(bundle, config, random_counts)
    output = tmp_path / "maps.h5"

    metadata = _content_addressed_map_metadata(fields, config, desi_summary=summary)
    metadata["config"] = {}
    metadata["map_product_id"] = mp.map_product_id_from_metadata(metadata)
    mp.save_map_product(output, fields, metadata, overwrite=True)
    loaded, _ = mp.load_map_product(output)

    assert loaded["pi1"].has_catalog_momentum
    assert loaded["pi_shuf1"].has_catalog_momentum
    np.testing.assert_allclose(loaded["pi1"].catalog["ra_deg"], fields["pi1"].catalog["ra_deg"])
    np.testing.assert_allclose(loaded["pi1"].catalog["dec_deg"], fields["pi1"].catalog["dec_deg"])
    np.testing.assert_allclose(loaded["pi1"].catalog["weight"], fields["pi1"].catalog["weight"])
    np.testing.assert_allclose(loaded["pi1"].catalog["field"], fields["pi1"].catalog["field"])
    with h5py.File(output, "r+") as h5:
        h5["fields/pi1/catalog/field"][0] += 1.0
    with pytest.raises(ValueError, match="does not match map_product_id"):
        mp.load_map_product(output)


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


def test_ksz_covariance_block_uses_catalog_nf_and_map_inka_inputs(tmp_path):
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
    assert ("inka_data", "T", "T") in input_cache
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
    np.testing.assert_allclose(np.trapz(nz["dndz_by_bin"], x=nz["z_mid"], axis=1), np.ones(4))
    np.testing.assert_allclose(nz["sigma_e_by_bin"], [0.21, 0.22, 0.23, 0.24])
    assert nz["priors"]["Delta_z_bias_bin1"]["sigma"] == 1.8e-2


def test_default_pixel_windows_are_field_specific(tmp_path):
    path = tmp_path / "measurement.h5"
    lmax = 8
    with h5py.File(path, "w") as h5:
        h5.attrs["config_json"] = json.dumps({"nside": 8, "lmax": lmax})
        fields = h5.create_group("fields")
        fields.attrs["metadata_json"] = json.dumps(
            {
                "s1": {"spin": 2, "kind": "des_shear"},
                "g1": {"spin": 0, "kind": "desi_galaxy"},
                "pi1": {"spin": 0, "kind": "desi_momentum"},
                "T": {"spin": 0, "kind": "act_cmb_temperature"},
                "y": {"spin": 0, "kind": "act_tsz_y"},
                "kappa": {"spin": 0, "kind": "act_cmb_lensing_kappa"},
            }
        )
        transfers = h5.create_group("transfer_functions")
        pix_t = np.linspace(1.0, 0.8, lmax + 1)
        pix_p = np.linspace(1.0, 0.7, lmax + 1)
        transfers.create_dataset("healpix_temperature_pixwin", data=pix_t)
        transfers.create_dataset("healpix_polarization_pixwin", data=pix_p)
        loaded = mp._load_default_transfers(h5, lmax, include_act_beams=False)

    np.testing.assert_allclose(loaded["s1"], pix_p)
    np.testing.assert_allclose(loaded["g1"], pix_t)
    for name in ("pi1", "T", "y", "kappa"):
        np.testing.assert_allclose(loaded[name], np.ones(lmax + 1))


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
    with pytest.raises(ValueError, match="legacy"):
        mp.theory_to_data_vector(
            path,
            theory,
            transfer_functions=transfers,
            include_default_pixel_windows=False,
            include_default_act_beams=False,
        )
    dv, names = mp.theory_to_data_vector(
        path,
        theory,
        transfer_functions=transfers,
        include_default_pixel_windows=False,
        include_default_act_beams=False,
        allow_legacy_product=True,
    )
    assert names == ["desi_g_act_y_pz1"]
    np.testing.assert_allclose(dv, [4.0, 8.0])


def test_theory_galaxy_shot_nuisance_uses_saved_bandpower_template_after_signal_window(tmp_path):
    path = tmp_path / "measurement_gshot.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MEASUREMENT
        h5.attrs["config_json"] = json.dumps({"nside": 8, "lmax": 4})
        fields = h5.create_group("fields")
        fields.attrs["metadata_json"] = json.dumps(
            {"g1": {"spin": 0, "kind": "desi_galaxy"}}
        )
        joint = h5.create_group("joint")
        joint.create_dataset(
            "spectrum_names",
            data=np.array(["desi_g_auto_pz1"], dtype=string_dtype),
        )
        spec = h5.create_group("spectra/desi_g_auto_pz1")
        spec.attrs["fields"] = json.dumps(["g1", "g1"])
        spec.attrs["theory_key"] = "desi_g_auto_pz1"
        spec.attrs["family"] = "desi_g_auto"
        spec.attrs["component"] = 0
        spec.attrs["cl_convention"] = mp.DESI_GALAXY_AUTO_MEAN_CONVENTION
        spec.attrs["metadata_json"] = json.dumps({"desi_pz": 1})
        window = np.zeros((1, 5), dtype=np.float64)
        window[0, 2] = 1.0
        spec.create_dataset("bandpower_window_selected", data=window)
        spec.create_dataset("noise_decoupled_all_components", data=np.asarray([[5.0]]))

    signal = np.full(5, 2.0, dtype=np.float64)
    signal_only, _ = mp.theory_to_data_vector(
        path,
        {"desi_g_auto_pz1": signal},
        transfer_functions={"g1": np.full(5, 0.5)},
        desi_galaxy_shot_noise_amplitudes={1: 0.0},
        include_default_pixel_windows=False,
        include_default_act_beams=False,
        allow_legacy_product=True,
    )
    dv, names = mp.theory_to_data_vector(
        path,
        {"desi_g_auto_pz1": signal},
        transfer_functions={"g1": np.full(5, 0.5)},
        desi_galaxy_shot_noise_amplitudes={1: 3.0},
        include_default_pixel_windows=False,
        include_default_act_beams=False,
        allow_legacy_product=True,
    )
    assert names == ["desi_g_auto_pz1"]
    # Cosmological signal receives g1*g1 = 0.25; the already-decoupled shot
    # response is then added in bandpower space and is not pixel-window damped.
    np.testing.assert_allclose(dv, [2.0 * 0.25 + 3.0 * 5.0])
    np.testing.assert_allclose(signal_only, [2.0 * 0.25])

    with pytest.raises(ValueError, match="include shot noise"):
        mp.theory_to_data_vector(
            path,
            {"desi_g_auto_pz1": signal},
            desi_galaxy_shot_noise_amplitudes=None,
            include_default_pixel_windows=False,
            include_default_act_beams=False,
            allow_legacy_product=True,
        )


def test_hmc_galaxy_shot_nuisance_matches_saved_bandpower_template_after_signal_window():
    import godmax_multiprobe_hmc_stage31 as hmc

    window = np.zeros((1, 5), dtype=np.float64)
    window[0, 2] = 1.0
    spec = hmc.SpectrumSpec(
        name="desi_g_auto_pz1",
        family="desi_g_auto",
        theory_key="desi_g_auto_pz1",
        fields=("g1", "g1"),
        pz_bin=1,
        window=hmc.jnp.asarray(window),
        transfer=hmc.jnp.full(5, 0.25, dtype=hmc.jnp.float64),
        scalar_factor=1.0,
        ksz_amp=0.0,
        shot_noise_template=hmc.jnp.asarray([5.0], dtype=hmc.jnp.float64),
        shot_noise_amplitude=1.0,
        source_band_count=1,
        selected_band_indices=(0,),
        ell_band=(2.0,),
    )
    likelihood = SimpleNamespace(spectrum_specs=(spec,))
    theory = {"desi_g_auto_pz1": hmc.jnp.full(3, 2.0, dtype=hmc.jnp.float64)}

    fixed = hmc.theory_data_vector_jax(likelihood, theory)
    signal_only = hmc.theory_data_vector_jax(
        likelihood,
        theory,
        desi_galaxy_shot_noise_amplitudes={1: 0.0},
    )
    sampled = hmc.theory_data_vector_jax(
        likelihood,
        theory,
        desi_galaxy_shot_noise_amplitudes={1: 3.0},
    )
    np.testing.assert_allclose(np.asarray(fixed), [2.0 * 0.25 + 5.0])
    np.testing.assert_allclose(np.asarray(signal_only), [2.0 * 0.25])
    np.testing.assert_allclose(np.asarray(sampled), [2.0 * 0.25 + 3.0 * 5.0])


def test_hmc_galaxy_shot_amplitudes_can_be_registered_as_four_free_scalars():
    import godmax_multiprobe_hmc_stage31 as hmc

    names = [f"desi_galaxy_shot_noise_amplitude_pz{i}" for i in range(1, 5)]
    config = {
        "params": {
            "sim_params": {},
            "other_params": {name: 1.0 for name in names},
        }
    }
    priors = {
        "expected_parameter_count": 4,
        "prior_uniform": {name: [0.0, 2.0] for name in names},
        "prior_gaussian": {},
        "vary": {
            "baryon_scalars": [],
            "other_scalars": names,
            "hod_arrays": [],
            "hod_indices": [],
        },
    }
    specs = hmc.build_parameter_specs(config, priors)
    registry = hmc.validate_parameter_specs(config, specs, priors)
    sample = {name: float(index) for index, name in enumerate(names, start=1)}

    assert registry["parameter_names"] == names
    assert all(spec.target == "other_scalar" for spec in specs)
    assert hmc._sampled_shot_noise_amplitudes(sample) == {
        1: 1.0,
        2: 2.0,
        3: 3.0,
        4: 4.0,
    }

    saved_config = hmc.apply_sample_to_config(config, specs, sample)
    assert gmt.desi_galaxy_shot_noise_amplitudes_from_config(saved_config) == {
        1: 1.0,
        2: 2.0,
        3: 3.0,
        4: 4.0,
    }
    explicit_config = dict(saved_config)
    explicit_config["raw"] = {
        "theory_to_data_vector": {"desi_galaxy_shot_noise_amplitudes": 0.75}
    }
    with pytest.raises(ValueError, match="Ambiguous DESI galaxy shot-noise amplitudes"):
        gmt.desi_galaxy_shot_noise_amplitudes_from_config(explicit_config)


def test_stage31_combiner_rejects_stale_or_mixed_chain_identity(tmp_path):
    import combine_godmax_hmc_stage31_workers as combine
    import godmax_multiprobe_hmc_stage31 as hmc

    parameter_specs = (
        hmc.ParameterSpec(
            name="theta_a",
            base_name="theta_a",
            target="sim_scalar",
            fiducial=0.1,
            prior_min=0.0,
            prior_max=1.0,
        ),
        hmc.ParameterSpec(
            name="theta_b",
            base_name="theta_b",
            target="sim_scalar",
            fiducial=1.1,
            prior_kind="normal",
            prior_mean=1.0,
            prior_sigma=0.2,
        ),
    )
    parameter_names = [spec.name for spec in parameter_specs]
    parameter_contract = hmc.parameter_contract_identity_sha256(parameter_specs)
    parameter_specs_payload = hmc.parameter_specs_jsonable(parameter_specs)
    context = SimpleNamespace(parameter_specs=parameter_specs)
    static_summary = {
        "chain_contract_version": hmc.STAGE31_CHAIN_CONTRACT_VERSION,
        "likelihood_identity_sha256": "likelihood-a",
        "theory_response_identity_sha256": "response-a",
        "measurement_path": "/measurement-a.h5",
        "measurement_map_product_id": "map-a",
        "desi_galaxy_auto_mean_convention": mp.DESI_GALAXY_AUTO_MEAN_CONVENTION,
        "parameter_names": parameter_names,
        "parameter_contract_identity_sha256": parameter_contract,
    }
    payload = {
        "parameter_names": np.asarray(parameter_names),
        "sample__theta_a": np.asarray([0.1, 0.2]),
        "sample__theta_b": np.asarray([1.1, 1.2]),
        "sample__chi2": np.asarray([10.0, 9.0]),
        "parameter_contract_identity_sha256": np.asarray(parameter_contract),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "static_summary": static_summary,
                    "parameter_specs": parameter_specs_payload,
                }
            )
        ),
    }
    path = tmp_path / "worker_chain.npz"
    metadata = combine._validate_chain_payload(
        path,
        payload,
        context,
        expected_static_summary=static_summary,
    )
    assert metadata["static_summary"] == static_summary
    assert combine._validated_recomputed_chi2(9.0, 9.0) == 9.0

    changed_prior_specs = (
        parameter_specs[0],
        hmc.ParameterSpec(
            name="theta_b",
            base_name="theta_b",
            target="sim_scalar",
            fiducial=1.1,
            prior_kind="normal",
            prior_mean=1.0,
            prior_sigma=0.3,
        ),
    )
    changed_prior_static = dict(static_summary)
    changed_prior_static["parameter_contract_identity_sha256"] = (
        hmc.parameter_contract_identity_sha256(changed_prior_specs)
    )
    with pytest.raises(ValueError, match="parameter_contract_identity_sha256"):
        combine._validate_chain_payload(
            path,
            payload,
            SimpleNamespace(parameter_specs=changed_prior_specs),
            expected_static_summary=changed_prior_static,
        )

    reversed_parameters = dict(payload)
    reversed_parameters["parameter_names"] = np.asarray(parameter_names[::-1])
    with pytest.raises(ValueError, match="ordered parameter contract"):
        combine._validate_chain_payload(
            path,
            reversed_parameters,
            context,
            expected_static_summary=static_summary,
        )

    stale_metadata = json.loads(str(payload["metadata_json"]))
    stale_metadata["static_summary"]["likelihood_identity_sha256"] = "likelihood-b"
    stale_payload = dict(payload)
    stale_payload["metadata_json"] = np.asarray(json.dumps(stale_metadata))
    with pytest.raises(ValueError, match="likelihood identity mismatch"):
        combine._validate_chain_payload(
            path,
            stale_payload,
            context,
            expected_static_summary=static_summary,
        )

    stale_response_metadata = json.loads(str(payload["metadata_json"]))
    stale_response_metadata["static_summary"]["theory_response_identity_sha256"] = "response-b"
    stale_response_payload = dict(payload)
    stale_response_payload["metadata_json"] = np.asarray(json.dumps(stale_response_metadata))
    with pytest.raises(ValueError, match="likelihood identity mismatch"):
        combine._validate_chain_payload(
            path,
            stale_response_payload,
            context,
            expected_static_summary=static_summary,
        )

    missing_convention = json.loads(str(payload["metadata_json"]))
    del missing_convention["static_summary"]["desi_galaxy_auto_mean_convention"]
    missing_convention_payload = dict(payload)
    missing_convention_payload["metadata_json"] = np.asarray(json.dumps(missing_convention))
    with pytest.raises(ValueError, match="missing required identity key"):
        combine._validate_chain_payload(
            path,
            missing_convention_payload,
            context,
            expected_static_summary=static_summary,
        )

    unexpected_sample = dict(payload)
    unexpected_sample["sample__theta_stale"] = np.asarray([3.0, 4.0])
    with pytest.raises(ValueError, match="sample keys do not match"):
        combine._validate_chain_payload(
            path,
            unexpected_sample,
            context,
            expected_static_summary=static_summary,
        )

    missing_sample = dict(payload)
    del missing_sample["sample__theta_b"]
    with pytest.raises(ValueError, match="sample keys do not match"):
        combine._validate_chain_payload(
            path,
            missing_sample,
            context,
            expected_static_summary=static_summary,
        )

    with pytest.raises(ValueError, match="does not reproduce"):
        combine._validated_recomputed_chi2(9.0, 9.1)


def test_stage31_map_outputs_use_objective_dense_ell_path_with_shot_noise(monkeypatch):
    import jax.numpy as jnp
    import godmax_multiprobe_hmc_stage31 as hmc
    import godmax_multiprobe_map_stage31 as map_stage31

    window = np.zeros((2, 9), dtype=np.float64)
    window[0, 2] = 1.0
    window[1, 3] = 1.0
    spec = hmc.SpectrumSpec(
        name="gg",
        family="desi_g_auto",
        theory_key="gg",
        fields=("g1", "g1"),
        pz_bin=1,
        window=jnp.asarray(window),
        transfer=jnp.ones(9, dtype=jnp.float64),
        scalar_factor=1.0,
        ksz_amp=0.0,
        shot_noise_template=jnp.asarray([2.0, 3.0]),
        shot_noise_amplitude=1.0,
        source_band_count=2,
        selected_band_indices=(0, 1),
        ell_band=(20.0, 40.0),
    )
    active_likelihood = SimpleNamespace(spectrum_specs=(spec,))
    full_likelihood = SimpleNamespace(spectrum_specs=(spec,))
    context = SimpleNamespace(
        likelihood=active_likelihood,
        config={"metadata": {"lmax": 8}},
    )
    sample = {"desi_galaxy_shot_noise_amplitude_pz1": 1.5}
    models = {"wl": SimpleNamespace(ell_array=jnp.asarray([2.0, 4.0, 8.0]))}
    dense_cls = {"gg": jnp.arange(1.0, 8.0, dtype=jnp.float64)}

    monkeypatch.setattr(hmc, "build_models_from_sample", lambda *_args, **_kwargs: models)
    monkeypatch.setattr(hmc, "_dense_theory_cls_from_models", lambda *_args, **_kwargs: dense_cls)
    monkeypatch.setattr(
        hmc,
        "extract_theory_cls_jax_from_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("MAP writer bypassed the required dense-ell helper")
        ),
    )
    monkeypatch.setattr(hmc, "full_likelihood_for_plots", lambda _context: full_likelihood)

    expected_active = np.asarray(hmc.evaluate_sample_theory_vector(context, sample))
    active, returned_full_likelihood, full = map_stage31.active_and_full_theory_vectors(
        context,
        sample,
    )
    np.testing.assert_array_equal(active, expected_active)
    np.testing.assert_array_equal(full, expected_active)
    np.testing.assert_array_equal(active, [4.0, 6.5])
    assert returned_full_likelihood is full_likelihood


def test_galaxy_shot_migration_is_lossless_and_leaves_covariance_unchanged(tmp_path):
    source = tmp_path / "synthetic_pipev2.h5"
    destination = migrate_gshot.default_destination(source)
    string_dtype = h5py.string_dtype("utf-8")
    covariance = np.asarray([[4.0, 0.5], [0.5, 9.0]], dtype=np.float64)
    with h5py.File(source, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MEASUREMENT
        h5.attrs["covariance_estimator_version"] = mp.COVARIANCE_ESTIMATOR_VERSION
        covariance_inputs = h5.create_group("input_cls_for_covariance")
        covariance_inputs.attrs["mode"] = "inka_data"
        galaxy_auto_covariance_input = covariance_inputs.create_dataset(
            "inka_data__g1__x__g1", data=np.asarray([[13.0, 17.0]])
        )
        galaxy_auto_covariance_input.attrs["field_a"] = "g1"
        galaxy_auto_covariance_input.attrs["field_b"] = "g1"
        galaxy_auto_covariance_input.attrs["kind_a"] = "desi_galaxy"
        galaxy_auto_covariance_input.attrs["kind_b"] = "desi_galaxy"
        galaxy_auto_covariance_input.attrs["noise_policy"] = (
            "DESI galaxy auto: data-derived NaMaster iNKA total pseudo-spectrum; the map auto "
            "contains the weighted Poisson shot noise subtracted only from the saved mean bandpowers."
        )
        joint = h5.create_group("joint")
        joint.create_dataset(
            "spectrum_names",
            data=np.asarray(["desi_g_auto_pz1", "desi_g_act_y_pz1"], dtype=string_dtype),
        )
        joint.create_dataset("slice_start", data=np.asarray([0, 1], dtype=np.int64))
        joint.create_dataset("slice_stop", data=np.asarray([1, 2], dtype=np.int64))
        joint.create_dataset("data_vector", data=np.asarray([7.0, 11.0]))
        joint.create_dataset("cov", data=covariance)

        auto = h5.create_group("spectra/desi_g_auto_pz1")
        auto.attrs["family"] = "desi_g_auto"
        auto.attrs["component"] = 0
        auto.attrs["cl_convention"] = migrate_gshot.LEGACY_SIGNAL_ONLY_CONVENTION
        auto.create_dataset("cl", data=np.asarray([7.0]))
        auto.create_dataset("cl_all_components", data=np.asarray([[7.0]]))
        auto.create_dataset("noise_decoupled_all_components", data=np.asarray([[2.0]]))

        cross = h5.create_group("spectra/desi_g_act_y_pz1")
        cross.attrs["family"] = "desi_g_act_y"
        cross.create_dataset("cl", data=np.asarray([11.0]))

    report = migrate_gshot.migrate_product(source, destination)
    assert report["changed_spectra"] == ["desi_g_auto_pz1"]
    assert report["changed_data_vector_elements"] == 1
    assert report["audit"]["status"] == "PASS"

    with h5py.File(source, "r") as old, h5py.File(destination, "r") as new:
        assert "desi_galaxy_auto_mean_convention" not in old.attrs
        np.testing.assert_array_equal(old["joint/data_vector"][:], [7.0, 11.0])
        assert (
            new.attrs["desi_galaxy_auto_mean_convention"]
            == mp.DESI_GALAXY_AUTO_MEAN_CONVENTION
        )
        np.testing.assert_array_equal(new["joint/data_vector"][:], [9.0, 11.0])
        np.testing.assert_array_equal(new["joint/cov"][:], covariance)
        np.testing.assert_array_equal(
            new["input_cls_for_covariance/inka_data__g1__x__g1"][:],
            old["input_cls_for_covariance/inka_data__g1__x__g1"][:],
        )
        assert "saved mean bandpowers both contain" in str(
            new["input_cls_for_covariance/inka_data__g1__x__g1"].attrs["noise_policy"]
        )
        np.testing.assert_array_equal(
            new["spectra/desi_g_auto_pz1/cl_shot_noise_subtracted_signal"][:],
            [7.0],
        )
        np.testing.assert_array_equal(new["spectra/desi_g_auto_pz1/cl"][:], [9.0])
        np.testing.assert_array_equal(new["spectra/desi_g_act_y_pz1/cl"][:], [11.0])

    audit = migrate_gshot.audit_migration(source, destination)
    assert audit["status"] == "PASS"
    assert audit["changed_data_vector_elements"] == 1

    with pytest.raises(FileExistsError, match="Destination already exists"):
        migrate_gshot.migrate_product(source, destination)


def test_plotter_adds_shot_template_only_for_explicit_signal_only_products():
    kwargs = dict(
        name="desi_g_auto_pz1",
        label="g1 auto",
        family="desi_g_auto",
        ell=np.asarray([100.0]),
        cl=np.asarray([7.0]),
        cov=np.asarray([[1.0]]),
        start=0,
        stop=1,
        noise_decoupled=np.asarray([2.0]),
    )
    current = plot_measurement.SpectrumPlotData(
        **kwargs,
        cl_convention=mp.DESI_GALAXY_AUTO_MEAN_CONVENTION,
    )
    historical = plot_measurement.SpectrumPlotData(
        **kwargs,
        cl_convention="shot_noise_subtracted_signal",
    )
    np.testing.assert_array_equal(current.cl_total, [7.0])
    np.testing.assert_array_equal(historical.cl_total, [9.0])


def test_standalone_plot_loader_rejects_unknown_galaxy_auto_convention(tmp_path, monkeypatch):
    path = tmp_path / "measurement.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        joint = h5.create_group("joint")
        joint.create_dataset(
            "spectrum_names",
            data=np.asarray(["desi_g_auto_pz1"], dtype=string_dtype),
        )
        joint.create_dataset("slice_start", data=np.asarray([0]))
        joint.create_dataset("slice_stop", data=np.asarray([1]))
        joint.create_dataset("data_vector", data=np.asarray([7.0]))
        joint.create_dataset("cov", data=np.asarray([[1.0]]))
        joint.create_dataset("corr", data=np.asarray([[1.0]]))
        joint.create_dataset("ell", data=np.asarray([100.0]))
        group = h5.create_group("spectra/desi_g_auto_pz1")
        group.attrs["family"] = "desi_g_auto"
        group.attrs["component"] = 0
        group.create_dataset("ell", data=np.asarray([100.0]))

    monkeypatch.setattr(mp, "validate_measurement_product_identity", lambda *args, **kwargs: "map")
    with pytest.raises(ValueError, match="unknown/missing cl_convention"):
        plot_measurement.load_measurement(path)

    with h5py.File(path, "r+") as h5:
        group = h5["spectra/desi_g_auto_pz1"]
        group.attrs["cl_convention"] = mp.DESI_GALAXY_AUTO_MEAN_CONVENTION
    with pytest.raises(ValueError, match="no saved shot-noise template"):
        plot_measurement.load_measurement(path)

    with h5py.File(path, "r+") as h5:
        h5["spectra/desi_g_auto_pz1"].create_dataset(
            "noise_decoupled_all_components", data=np.asarray([[2.0]])
        )
    loaded = plot_measurement.load_measurement(path)
    np.testing.assert_array_equal(loaded.spectra[0].cl_total, [7.0])


def test_standalone_plot_loader_omits_validity_placeholders_without_conditioning(tmp_path, monkeypatch):
    path = tmp_path / "measurement_validity.h5"
    string_dtype = h5py.string_dtype("utf-8")
    packed = np.asarray([1.0, 0.0, 0.0, 4.0, 5.0, 6.0])
    raw = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    valid = np.asarray([True, False, False, True, True, True])
    covariance = np.diag(np.arange(1.0, 7.0))
    with h5py.File(path, "w") as h5:
        joint = h5.create_group("joint")
        joint.create_dataset(
            "spectrum_names",
            data=np.asarray(
                ["desi_g_act_kappa_pz1", "desi_g_act_y_pz1"],
                dtype=string_dtype,
            ),
        )
        joint.create_dataset("slice_start", data=np.asarray([0, 3]))
        joint.create_dataset("slice_stop", data=np.asarray([3, 6]))
        joint.create_dataset("data_vector", data=packed)
        joint.create_dataset("data_vector_raw", data=raw)
        joint.create_dataset("data_vector_valid", data=valid)
        joint.create_dataset("cov", data=covariance)
        joint.create_dataset("corr", data=np.eye(6))
        joint.create_dataset("ell", data=np.asarray([100.0, 1000.0, 4000.0]))
        for name, family in (
            ("desi_g_act_kappa_pz1", "desi_g_act_kappa"),
            ("desi_g_act_y_pz1", "desi_g_act_y"),
        ):
            group = h5.create_group(f"spectra/{name}")
            group.attrs["family"] = family
            group.attrs["component"] = 0
            group.create_dataset("ell", data=np.asarray([100.0, 1000.0, 4000.0]))

    monkeypatch.setattr(mp, "validate_measurement_product_identity", lambda *args, **kwargs: "map")
    loaded = plot_measurement.load_measurement(path)
    np.testing.assert_array_equal(loaded.data_vector, [1.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(np.diag(loaded.covariance), [1.0, 4.0, 5.0, 6.0])
    np.testing.assert_array_equal(loaded.spectra[0].cl, [1.0])
    np.testing.assert_array_equal(loaded.spectra[1].cl, [4.0, 5.0, 6.0])
    assert (loaded.spectra[0].start, loaded.spectra[0].stop) == (0, 1)
    assert (loaded.spectra[1].start, loaded.spectra[1].stop) == (1, 4)


def test_comparison_config_identity_tracks_theory_inputs_not_output_location(tmp_path):
    config = {
        "raw": {
            "output_dir": "output-a",
            "theory_to_data_vector": {"include_default_act_beams": True},
        },
        "params": {"sim_params": {"sigma8": 0.8}},
        "metadata": {"lens_dndz": np.asarray([[0.0, 1.0, 0.0]])},
        "paths": {
            "config": tmp_path / "comparison.yaml",
            "measurement_h5": tmp_path / "measurement.h5",
            "map_h5": tmp_path / "maps.h5",
            "output_dir": tmp_path / "output-a",
        },
    }
    identity = gmt.comparison_config_identity_sha256(config)

    relocated = dict(config)
    relocated["paths"] = dict(config["paths"], output_dir=tmp_path / "output-b")
    relocated["raw"] = dict(config["raw"], output_dir="output-b")
    assert gmt.comparison_config_identity_sha256(relocated) == identity

    changed = dict(config)
    changed["params"] = {"sim_params": {"sigma8": 0.81}}
    assert gmt.comparison_config_identity_sha256(changed) != identity


def test_theory_response_identity_rejects_fixed_path_response_mutations(tmp_path, monkeypatch):
    import plot_stage31_bestfit_vs_fiducial_cls as plot_stage31

    monkeypatch.setattr(mp, "validate_measurement_product_identity", lambda *_args, **_kwargs: None)
    path = tmp_path / "measurement.h5"
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs["config_json"] = json.dumps({"nside": 8, "lmax": 4})
        fields = h5.create_group("fields")
        fields.attrs["metadata_json"] = json.dumps(
            {
                "g1": {"spin": 0, "kind": "desi_galaxy"},
                "y": {"spin": 0, "kind": "act_tsz_y"},
            }
        )
        transfer = h5.create_group("transfer_functions")
        transfer.create_dataset("healpix_temperature_pixwin", data=np.ones(5))
        transfer.create_dataset("healpix_polarization_pixwin", data=np.ones(5))
        transfer.create_dataset("act_y_gaussian_beam", data=np.ones(5))
        joint = h5.create_group("joint")
        joint.create_dataset(
            "spectrum_names",
            data=np.asarray(["desi_g_auto_pz1", "desi_g_act_y_pz1"], dtype=string_dtype),
        )
        spectra = h5.create_group("spectra")
        auto = spectra.create_group("desi_g_auto_pz1")
        auto.attrs["fields"] = json.dumps(["g1", "g1"])
        auto.attrs["family"] = "desi_g_auto"
        auto.attrs["theory_key"] = "desi_g_auto_pz1"
        auto.attrs["metadata_json"] = json.dumps({"desi_pz": 1})
        auto.attrs["component"] = 0
        auto.attrs["cl_convention"] = mp.DESI_GALAXY_AUTO_MEAN_CONVENTION
        auto.create_dataset(
            "bandpower_window_selected",
            data=np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0]]),
        )
        auto.create_dataset("noise_decoupled_all_components", data=np.asarray([[5.0]]))
        cross = spectra.create_group("desi_g_act_y_pz1")
        cross.attrs["fields"] = json.dumps(["g1", "y"])
        cross.attrs["family"] = "desi_g_act_y"
        cross.attrs["theory_key"] = "desi_g_act_y_pz1"
        cross.attrs["metadata_json"] = json.dumps({"desi_pz": 1})
        cross.create_dataset(
            "bandpower_window_selected",
            data=np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0]]),
        )

    config = {
        "repo_root": Path(__file__).resolve().parents[1],
        "raw": {
            "theory_to_data_vector": {
                "allow_legacy_product": True,
                "include_default_pixel_windows": True,
                "include_default_act_beams": True,
            }
        },
        "params": {},
        "metadata": {"shear_m_bias_means": {}},
        "paths": {
            "config": tmp_path / "comparison.yaml",
            "measurement_h5": path,
            "map_h5": tmp_path / "maps.h5",
            "output_dir": tmp_path / "output-a",
        },
    }
    response_identity = gmt.theory_response_identity_sha256(config)
    config_identity = gmt.comparison_config_identity_sha256(config)
    measurement = gmt.MeasurementData(
        path=path,
        names=["desi_g_auto_pz1", "desi_g_act_y_pz1"],
        ell=np.asarray([20.0, 20.0]),
        data_vector=np.asarray([6.0, 2.0]),
        covariance=np.eye(2),
        starts=np.asarray([0, 1]),
        stops=np.asarray([1, 2]),
        families={"desi_g_auto_pz1": "desi_g_auto", "desi_g_act_y_pz1": "desi_g_act_y"},
        labels={"desi_g_auto_pz1": "auto", "desi_g_act_y_pz1": "cross"},
        theory_keys={
            "desi_g_auto_pz1": "desi_g_auto_pz1",
            "desi_g_act_y_pz1": "desi_g_act_y_pz1",
        },
    )
    measurement_identity = gmt.measurement_identity_sha256(measurement)
    cached = {
        "ell_band": measurement.ell,
        "data_vector": measurement.data_vector,
        "theory_vector": np.asarray([5.5, 1.5]),
        "covariance": measurement.covariance,
        "spectrum_names": np.asarray(measurement.names),
        "slice_start": measurement.starts,
        "slice_stop": measurement.stops,
        "measurement_identity_sha256": np.asarray(measurement_identity),
        "theory_response_identity_sha256": np.asarray(response_identity),
    }
    cached.update(
        gmt.theory_vector_cache_fields(
            cached["theory_vector"],
            measurement_identity,
            {
                "product_kind": "configured_theory_vector",
                "comparison_config_identity_sha256": config_identity,
                "theory_response_identity_sha256": response_identity,
            },
        )
    )
    plot_stage31.validate_cached_vector_product(
        cached,
        path,
        measurement,
        expected_config_identity=config_identity,
        expected_theory_response_identity=response_identity,
    )

    mutations = (
        ("spectra/desi_g_auto_pz1/noise_decoupled_all_components", (0, 0), 6.0),
        ("spectra/desi_g_auto_pz1/bandpower_window_selected", (0, 2), 0.5),
        ("transfer_functions/act_y_gaussian_beam", (2,), 0.8),
    )
    for dataset, index, replacement in mutations:
        with h5py.File(path, "r+") as h5:
            original = float(h5[dataset][index])
            h5[dataset][index] = replacement
        changed_response = gmt.theory_response_identity_sha256(config)
        assert changed_response != response_identity
        assert gmt.comparison_config_identity_sha256(config) == config_identity
        with pytest.raises(ValueError, match="different saved theory-response content"):
            plot_stage31.validate_cached_vector_product(
                cached,
                path,
                measurement,
                expected_config_identity=config_identity,
                expected_theory_response_identity=changed_response,
            )
        with h5py.File(path, "r+") as h5:
            h5[dataset][index] = original
        assert gmt.theory_response_identity_sha256(config) == response_identity

    relocated = dict(config)
    relocated["paths"] = dict(config["paths"], output_dir=tmp_path / "output-b")
    assert gmt.theory_response_identity_sha256(relocated) == response_identity


def test_stage31_cached_theory_vectors_are_measurement_fingerprint_bound(tmp_path):
    import godmax_multiprobe_hmc_stage31 as hmc
    import plot_stage31_bestfit_vs_fiducial_cls as plot_stage31

    measurement = gmt.MeasurementData(
        path=tmp_path / "measurement.h5",
        names=["spectrum_a"],
        ell=np.asarray([20.0, 40.0]),
        data_vector=np.asarray([1.0, 2.0]),
        covariance=np.asarray([[4.0, 0.5], [0.5, 9.0]]),
        starts=np.asarray([0], dtype=np.int64),
        stops=np.asarray([2], dtype=np.int64),
        families={"spectrum_a": "family_a"},
        labels={"spectrum_a": "Spectrum A"},
        theory_keys={"spectrum_a": "theory_a"},
    )
    identity = gmt.measurement_identity_sha256(measurement)
    parameter_names = ["theta_a", "theta_b"]
    parameter_contract = "parameter-contract-a"
    best_sample = {"theta_a": 0.2, "theta_b": 1.2}
    payload = {
        "ell_band": np.asarray(measurement.ell),
        "data_vector": np.asarray(measurement.data_vector),
        "theory_vector": np.asarray([0.9, 2.1]),
        "covariance": np.asarray(measurement.covariance),
        "spectrum_names": np.asarray(measurement.names),
        "slice_start": np.asarray(measurement.starts),
        "slice_stop": np.asarray(measurement.stops),
        "measurement_identity_sha256": np.asarray(identity),
        "likelihood_identity_sha256": np.asarray("likelihood-a"),
        "theory_response_identity_sha256": np.asarray("response-a"),
        "parameter_names": np.asarray(parameter_names),
        "parameter_contract_identity_sha256": np.asarray(parameter_contract),
        "best_sample_json": np.asarray(json.dumps(best_sample)),
    }
    payload.update(
        gmt.theory_vector_cache_fields(
            payload["theory_vector"],
            identity,
            {
                "product_kind": "stage31_bestfit_active",
                "chain_contract_version": hmc.STAGE31_CHAIN_CONTRACT_VERSION,
                "likelihood_identity_sha256": "likelihood-a",
                "comparison_config_identity_sha256": "config-a",
                "theory_response_identity_sha256": "response-a",
                "parameter_names": parameter_names,
                "parameter_contract_identity_sha256": parameter_contract,
                "best_sample": best_sample,
            },
        )
    )
    path = tmp_path / "cached_vector.npz"
    plot_stage31.validate_cached_vector_product(
        payload,
        path,
        measurement,
        expected_likelihood_identity="likelihood-a",
        expected_config_identity="config-a",
        expected_theory_response_identity="response-a",
        expected_parameter_names=parameter_names,
        expected_parameter_contract_identity=parameter_contract,
    )

    stale = dict(payload)
    stale["data_vector"] = np.asarray([1.0, 2.01])
    with pytest.raises(ValueError, match="fingerprint does not match"):
        plot_stage31.validate_cached_vector_product(stale, path, measurement)

    corrupt_theory = dict(payload)
    corrupt_theory["theory_vector"] = np.asarray([9.0e99, 9.0e99])
    with pytest.raises(ValueError, match="theory-vector fingerprint"):
        plot_stage31.validate_cached_vector_product(
            corrupt_theory,
            path,
            measurement,
            expected_likelihood_identity="likelihood-a",
            expected_config_identity="config-a",
            expected_theory_response_identity="response-a",
            expected_parameter_names=parameter_names,
            expected_parameter_contract_identity=parameter_contract,
        )

    wrong_config = dict(payload)
    wrong_config.update(
        gmt.theory_vector_cache_fields(
            wrong_config["theory_vector"],
            identity,
            {
                "product_kind": "stage31_bestfit_active",
                "chain_contract_version": hmc.STAGE31_CHAIN_CONTRACT_VERSION,
                "likelihood_identity_sha256": "likelihood-a",
                "comparison_config_identity_sha256": "config-b",
                "theory_response_identity_sha256": "response-a",
                "parameter_names": parameter_names,
                "parameter_contract_identity_sha256": parameter_contract,
                "best_sample": best_sample,
            },
        )
    )
    with pytest.raises(ValueError, match="different materialized comparison configuration"):
        plot_stage31.validate_cached_vector_product(
            wrong_config,
            path,
            measurement,
            expected_likelihood_identity="likelihood-a",
            expected_config_identity="config-a",
            expected_theory_response_identity="response-a",
            expected_parameter_names=parameter_names,
            expected_parameter_contract_identity=parameter_contract,
        )

    wrong_prior = dict(payload)
    wrong_prior["parameter_contract_identity_sha256"] = np.asarray("parameter-contract-b")
    wrong_prior.update(
        gmt.theory_vector_cache_fields(
            wrong_prior["theory_vector"],
            identity,
            {
                "product_kind": "stage31_bestfit_active",
                "chain_contract_version": hmc.STAGE31_CHAIN_CONTRACT_VERSION,
                "likelihood_identity_sha256": "likelihood-a",
                "comparison_config_identity_sha256": "config-a",
                "theory_response_identity_sha256": "response-a",
                "parameter_names": parameter_names,
                "parameter_contract_identity_sha256": "parameter-contract-b",
                "best_sample": best_sample,
            },
        )
    )
    with pytest.raises(ValueError, match="different parameter/prior contract"):
        plot_stage31.validate_cached_vector_product(
            wrong_prior,
            path,
            measurement,
            expected_likelihood_identity="likelihood-a",
            expected_config_identity="config-a",
            expected_theory_response_identity="response-a",
            expected_parameter_names=parameter_names,
            expected_parameter_contract_identity=parameter_contract,
        )

    old_parameter_names = ["theta_a"]
    old_best_sample = {"theta_a": 0.2}
    old_parameter_cache = dict(payload)
    old_parameter_cache["parameter_names"] = np.asarray(old_parameter_names)
    old_parameter_cache["parameter_contract_identity_sha256"] = np.asarray("old-contract")
    old_parameter_cache["best_sample_json"] = np.asarray(json.dumps(old_best_sample))
    old_parameter_cache.update(
        gmt.theory_vector_cache_fields(
            old_parameter_cache["theory_vector"],
            identity,
            {
                "product_kind": "stage31_bestfit_active",
                "chain_contract_version": hmc.STAGE31_CHAIN_CONTRACT_VERSION,
                "likelihood_identity_sha256": "likelihood-a",
                "comparison_config_identity_sha256": "config-a",
                "theory_response_identity_sha256": "response-a",
                "parameter_names": old_parameter_names,
                "parameter_contract_identity_sha256": "old-contract",
                "best_sample": old_best_sample,
            },
        )
    )
    with pytest.raises(ValueError, match="different ordered parameter contract"):
        plot_stage31.validate_cached_vector_product(
            old_parameter_cache,
            path,
            measurement,
            expected_likelihood_identity="likelihood-a",
            expected_config_identity="config-a",
            expected_theory_response_identity="response-a",
            expected_parameter_names=parameter_names,
            expected_parameter_contract_identity=parameter_contract,
        )

    wrong_likelihood = dict(payload)
    wrong_likelihood["likelihood_identity_sha256"] = np.asarray("likelihood-b")
    with pytest.raises(ValueError, match="different Stage-31 likelihood"):
        plot_stage31.validate_cached_vector_product(
            wrong_likelihood,
            path,
            measurement,
            expected_likelihood_identity="likelihood-a",
            expected_config_identity="config-a",
            expected_theory_response_identity="response-a",
            expected_parameter_names=parameter_names,
            expected_parameter_contract_identity=parameter_contract,
        )

    fit_summary = {
        "static_summary": {
            "chain_contract_version": hmc.STAGE31_CHAIN_CONTRACT_VERSION,
            "likelihood_identity_sha256": "likelihood-a",
            "theory_response_identity_sha256": "response-a",
            "parameter_names": parameter_names,
            "parameter_contract_identity_sha256": parameter_contract,
        }
    }
    plot_stage31.validate_fit_summary_contract(
        fit_summary,
        tmp_path / "fit_summary.json",
        expected_likelihood_identity="likelihood-a",
        expected_theory_response_identity="response-a",
        expected_parameter_names=parameter_names,
        expected_parameter_contract_identity=parameter_contract,
    )
    stale_fit_summary = json.loads(json.dumps(fit_summary))
    stale_fit_summary["static_summary"]["parameter_contract_identity_sha256"] = (
        "parameter-contract-b"
    )
    with pytest.raises(ValueError, match="different parameter/prior contract"):
        plot_stage31.validate_fit_summary_contract(
            stale_fit_summary,
            tmp_path / "fit_summary.json",
            expected_likelihood_identity="likelihood-a",
            expected_theory_response_identity="response-a",
            expected_parameter_names=parameter_names,
            expected_parameter_contract_identity=parameter_contract,
        )


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
        allow_legacy_product=True,
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
    dv, _ = mp.theory_to_data_vector(
        path,
        theory,
        include_default_pixel_windows=False,
        allow_legacy_product=True,
    )
    np.testing.assert_allclose(dv, [1.0])

    write_product(1.0)
    dv, _ = mp.theory_to_data_vector(
        path,
        theory,
        include_default_pixel_windows=False,
        allow_legacy_product=True,
    )
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
        allow_legacy_product=True,
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
        allow_legacy_product=True,
    )
    assert names == ["desi_pi_act_T_pz1"]
    np.testing.assert_allclose(dv, [-mp.TCMB_UK * 0.3 * 0.001 * mp.KSZ_SIGMA_TRUE_GAS_OVER_C_3E5[1]])


def test_covariance_diagnostics_exposes_negative_correlation_mode():
    cov = np.full((3, 3), -0.6, dtype=np.float64)
    np.fill_diagonal(cov, 1.0)
    diagnostics = mp.covariance_diagnostics(cov)
    assert diagnostics["diag_strictly_positive"] is True
    assert diagnostics["corr_eig_min"] < -0.1
    assert diagnostics["n_negative_corr_eig"] == 1
    assert diagnostics["n_negative_eig_relative_1e-12"] == 1


def test_legacy_products_are_not_compatible_with_pipeline_v2(tmp_path, monkeypatch):
    config = mp.MeasurementConfig(nside=1, lmax=3, n_bins=1, ell_min=0)
    path = tmp_path / "maps.h5"
    metadata = {"config": dict(config.__dict__)}
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MAPS
        h5.attrs["metadata_json"] = json.dumps(metadata)
    compatible, reason = prod._existing_product_matches_config(path, mp.SCHEMA_MAPS, config)
    assert compatible is False
    assert "pipeline_version" in reason

    current_path = tmp_path / "current_maps.h5"
    fields = {
        "g1": mp.FieldMap(
            name="g1",
            label="synthetic",
            kind="desi_galaxy",
            spin=0,
            maps=[np.zeros(12, dtype=np.float32)],
            mask=np.ones(12, dtype=np.float32),
            mask_name="mask",
        )
    }
    current_metadata = _content_addressed_map_metadata(fields, config)
    monkeypatch.setattr(
        mp.hp,
        "pixwin",
        lambda nside, lmax, pol: (np.ones(lmax + 1), np.ones(lmax + 1)),
    )
    mp.save_map_product(current_path, fields, current_metadata, overwrite=True)
    compatible, reason = prod._existing_product_matches_config(
        current_path, mp.SCHEMA_MAPS, config
    )
    assert (compatible, reason) == (True, "compatible")


@pytest.mark.parametrize("stage", ["fast1024", "midres2048"])
def test_lowres_map_reuse_does_not_require_highres_only_contract_keys(tmp_path, stage):
    config = mp.MeasurementConfig.for_stage(stage)
    fields = {
        "g1": mp.FieldMap(
            name="g1",
            label="synthetic",
            kind="desi_galaxy",
            spin=0,
            maps=[np.zeros(12, dtype=np.float32)],
            mask=np.ones(12, dtype=np.float32),
            mask_name="mask",
        )
    }
    metadata = _content_addressed_map_metadata(fields, config)
    del metadata["config"]["act_cmb_temperature_units_confirmed"]
    del metadata["config"]["minimum_desi_random_realizations"]
    metadata["map_product_id"] = mp.map_product_id_from_metadata(metadata)
    path = tmp_path / f"{stage}_maps.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MAPS
        h5.attrs["pipeline_version"] = mp.MEASUREMENT_PIPELINE_VERSION
        h5.attrs["map_construction_version"] = mp.MAP_CONSTRUCTION_VERSION
        h5.attrs["map_product_id"] = metadata["map_product_id"]
        h5.attrs["metadata_json"] = json.dumps(metadata)

    assert prod._existing_product_matches_config(path, mp.SCHEMA_MAPS, config) == (
        True,
        "compatible",
    )


@pytest.mark.parametrize(
    ("key", "stale_value"),
    [
        ("act_cmb_temperature_units_confirmed", False),
        ("minimum_desi_random_realizations", 1),
    ],
)
def test_highres_map_reuse_rejects_stale_units_or_random_contract(
    tmp_path, key, stale_value
):
    config = mp.MeasurementConfig.for_stage("highres4096")
    fields = {
        "g1": mp.FieldMap(
            name="g1",
            label="synthetic",
            kind="desi_galaxy",
            spin=0,
            maps=[np.zeros(12, dtype=np.float32)],
            mask=np.ones(12, dtype=np.float32),
            mask_name="mask",
        )
    }
    metadata = _content_addressed_map_metadata(fields, config)
    metadata["config"][key] = stale_value
    metadata["map_product_id"] = mp.map_product_id_from_metadata(metadata)
    path = tmp_path / f"stale_{key}.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MAPS
        h5.attrs["pipeline_version"] = mp.MEASUREMENT_PIPELINE_VERSION
        h5.attrs["map_construction_version"] = mp.MAP_CONSTRUCTION_VERSION
        h5.attrs["map_product_id"] = metadata["map_product_id"]
        h5.attrs["metadata_json"] = json.dumps(metadata)

    compatible, reason = prod._existing_product_matches_config(
        path, mp.SCHEMA_MAPS, config
    )
    assert compatible is False
    assert key in reason


def test_map_product_load_rejects_array_content_tampering(tmp_path, monkeypatch):
    config = mp.MeasurementConfig(nside=1, lmax=3, n_bins=1, ell_min=0)
    fields = {
        "g1": mp.FieldMap(
            name="g1",
            label="synthetic",
            kind="desi_galaxy",
            spin=0,
            maps=[np.zeros(12, dtype=np.float32)],
            mask=np.ones(12, dtype=np.float32),
            mask_name="mask",
        )
    }
    metadata = _content_addressed_map_metadata(fields, config)
    path = tmp_path / "maps.h5"
    monkeypatch.setattr(
        mp.hp,
        "pixwin",
        lambda nside, lmax, pol: (np.ones(lmax + 1), np.ones(lmax + 1)),
    )
    mp.save_map_product(path, fields, metadata, overwrite=True)
    with h5py.File(path, "r+") as h5:
        h5["masks/mask"][0] = 0.5
    with pytest.raises(ValueError, match="does not match map_product_id"):
        mp.load_map_product(path)
    with h5py.File(path, "r+") as h5:
        h5["masks/mask"][0] = 1.0
        h5["fields/g1/map0"][0] = 2.0
    with pytest.raises(ValueError, match="does not match map_product_id"):
        mp.load_map_product(path)


@pytest.mark.parametrize("tamper", ["metadata", "spin", "kind"])
def test_map_product_identity_binds_field_estimator_contract(tmp_path, monkeypatch, tamper):
    config = mp.MeasurementConfig(nside=1, lmax=3, n_bins=1, ell_min=0)
    fields = {
        "s1": mp.FieldMap(
            name="s1",
            label="synthetic shear",
            kind="des_shear",
            spin=2,
            maps=[np.zeros(12, dtype=np.float32), np.zeros(12, dtype=np.float32)],
            mask=np.ones(12, dtype=np.float32),
            mask_name="mask",
            metadata={"shape_noise_pseudo_cl": 0.1, "namaster_masked_on_input": False},
        )
    }
    metadata = _content_addressed_map_metadata(fields, config)
    path = tmp_path / f"maps_{tamper}.h5"
    monkeypatch.setattr(
        mp.hp,
        "pixwin",
        lambda nside, lmax, pol: (np.ones(lmax + 1), np.ones(lmax + 1)),
    )
    mp.save_map_product(path, fields, metadata, overwrite=True)
    with h5py.File(path, "r+") as h5:
        group = h5["fields/s1"]
        if tamper == "metadata":
            group.attrs["metadata_json"] = json.dumps(
                {"shape_noise_pseudo_cl": 999.0, "namaster_masked_on_input": True}
            )
        elif tamper == "spin":
            group.attrs["spin"] = 0
        else:
            group.attrs["kind"] = "act_cmb_temperature"
    with pytest.raises(ValueError, match="does not match map_product_id"):
        mp.load_map_product(path)


def test_current_measurement_identity_accepts_split_execution_but_rejects_mixed_map(tmp_path):
    config = mp.MeasurementConfig(nside=1, lmax=3, n_bins=1, ell_min=0)
    field_a = mp.FieldMap(
        name="g1",
        label="map a",
        kind="desi_galaxy",
        spin=0,
        maps=[np.zeros(12, dtype=np.float32)],
        mask=np.ones(12, dtype=np.float32),
        mask_name="mask",
    )
    field_b = mp.FieldMap(
        name="g1",
        label="map b",
        kind="desi_galaxy",
        spin=0,
        maps=[np.ones(12, dtype=np.float32)],
        mask=np.ones(12, dtype=np.float32),
        mask_name="mask",
    )
    metadata_a = _content_addressed_map_metadata({"g1": field_a}, config)
    metadata_b = _content_addressed_map_metadata({"g1": field_b}, config)
    map_path = tmp_path / "maps.h5"
    measurement_path = tmp_path / "measurement.h5"

    with h5py.File(map_path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MAPS
        h5.attrs["pipeline_version"] = mp.MEASUREMENT_PIPELINE_VERSION
        h5.attrs["map_construction_version"] = mp.MAP_CONSTRUCTION_VERSION
        h5.attrs["map_product_id"] = metadata_a["map_product_id"]
        h5.attrs["metadata_json"] = json.dumps(metadata_a)

    spectra_config = dict(config.__dict__)
    spectra_config["compute_covariance"] = False
    with h5py.File(measurement_path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MEASUREMENT
        h5.attrs["pipeline_version"] = mp.MEASUREMENT_PIPELINE_VERSION
        h5.attrs["map_construction_version"] = mp.MAP_CONSTRUCTION_VERSION
        h5.attrs["spectrum_estimator_version"] = mp.SPECTRUM_ESTIMATOR_VERSION
        h5.attrs["covariance_estimator_version"] = mp.COVARIANCE_ESTIMATOR_VERSION
        h5.attrs["desi_galaxy_auto_mean_convention"] = mp.DESI_GALAXY_AUTO_MEAN_CONVENTION
        h5.attrs["map_product_id"] = metadata_a["map_product_id"]
        h5.attrs["map_metadata_json"] = json.dumps(metadata_a)
        h5.attrs["config_json"] = json.dumps(spectra_config)

    with h5py.File(measurement_path, "r") as mh5, h5py.File(map_path, "r") as map_h5:
        assert mp.validate_measurement_product_identity(mh5) == metadata_a["map_product_id"]
        assert gmt.validate_measurement_map_identity(mh5, map_h5) == metadata_a["map_product_id"]

    with h5py.File(measurement_path, "r+") as h5:
        h5.attrs["map_product_id"] = metadata_b["map_product_id"]
        h5.attrs["map_metadata_json"] = json.dumps(metadata_b)
    with h5py.File(measurement_path, "r") as mh5, h5py.File(map_path, "r") as map_h5:
        with pytest.raises(ValueError, match="different map_product_id"):
            gmt.validate_measurement_map_identity(mh5, map_h5)


def test_spectra_reuse_requires_exact_config_and_originating_map(tmp_path):
    config = mp.MeasurementConfig(nside=1, lmax=3, n_bins=1, ell_min=0)
    field_old = mp.FieldMap(
        name="s1",
        label="old",
        kind="des_shear",
        spin=2,
        maps=[np.zeros(12, dtype=np.float32), np.zeros(12, dtype=np.float32)],
        mask=np.ones(12, dtype=np.float32),
        mask_name="shear",
    )
    field_new = mp.FieldMap(
        name="s1",
        label="new",
        kind="des_shear",
        spin=2,
        maps=[np.zeros(12, dtype=np.float32), np.zeros(12, dtype=np.float32)],
        mask=np.arange(12, dtype=np.float32),
        mask_name="shear",
    )
    old_meta = _content_addressed_map_metadata({"s1": field_old}, config)
    new_meta = _content_addressed_map_metadata({"s1": field_new}, config)
    path = tmp_path / "spectra.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = mp.SCHEMA_MEASUREMENT
        h5.attrs["pipeline_version"] = mp.MEASUREMENT_PIPELINE_VERSION
        h5.attrs["spectrum_estimator_version"] = mp.SPECTRUM_ESTIMATOR_VERSION
        h5.attrs["desi_galaxy_auto_mean_convention"] = mp.DESI_GALAXY_AUTO_MEAN_CONVENTION
        h5.attrs["map_product_id"] = old_meta["map_product_id"]
        h5.attrs["map_metadata_json"] = json.dumps(old_meta)
        h5.attrs["config_json"] = json.dumps(dict(config.__dict__))
        ell_left, ell_right = mp.make_bandpower_edges(config)
        h5.create_dataset("ell_left", data=ell_left)
        h5.create_dataset("ell_right", data=ell_right)

    compatible, reason = prod._existing_product_matches_config(
        path,
        mp.SCHEMA_MEASUREMENT,
        config,
        expected_map_product_id=new_meta["map_product_id"],
    )
    assert compatible is False
    assert "map_product_id" in reason

    compatible, reason = prod._existing_product_matches_config(
        path,
        mp.SCHEMA_MEASUREMENT,
        config,
        expected_map_product_id=old_meta["map_product_id"],
    )
    assert (compatible, reason) == (True, "compatible")

    with h5py.File(path, "r+") as h5:
        del h5.attrs["desi_galaxy_auto_mean_convention"]
    compatible, reason = prod._existing_product_matches_config(
        path,
        mp.SCHEMA_MEASUREMENT,
        config,
        expected_map_product_id=old_meta["map_product_id"],
    )
    assert compatible is False
    assert "desi_galaxy_auto_mean_convention" in reason
    with h5py.File(path, "r+") as h5:
        h5.attrs["desi_galaxy_auto_mean_convention"] = mp.DESI_GALAXY_AUTO_MEAN_CONVENTION

    changed_config = mp.MeasurementConfig(**dict(config.__dict__))
    changed_config.shear_mask_dataset = "mask_weight_raw"
    compatible, reason = prod._existing_product_matches_config(
        path,
        mp.SCHEMA_MEASUREMENT,
        changed_config,
        expected_map_product_id=old_meta["map_product_id"],
    )
    assert compatible is False
    assert "shear_mask_dataset" in reason

    with h5py.File(path, "r+") as h5:
        h5["ell_right"][-1] -= 1
    compatible, reason = prod._existing_product_matches_config(
        path,
        mp.SCHEMA_MEASUREMENT,
        config,
        expected_map_product_id=old_meta["map_product_id"],
    )
    assert compatible is False
    assert "ell_right" in reason


def test_covariance_manifest_is_versioned_and_tamper_evident():
    config = mp.MeasurementConfig.for_stage("fast1024")
    manifest = prod.build_covariance_manifest(config)
    prod.validate_covariance_manifest(manifest, config)
    assert manifest["covariance_estimator_version"] == mp.COVARIANCE_ESTIMATOR_VERSION
    assert len(manifest["groups"]) == 259
    assert manifest["ell_left"] == mp.make_bandpower_edges(config)[0].tolist()
    assert manifest["ell_right"] == mp.make_bandpower_edges(config)[1].tolist()

    tampered = json.loads(json.dumps(manifest))
    tampered["groups"][0]["representative_fields"][0] = "tampered"
    with pytest.raises(ValueError, match="digest"):
        prod.validate_covariance_manifest(tampered, config)

    forged = json.loads(json.dumps(manifest))
    forged["groups"][0]["representative_fields"][0] = "tampered"
    forged["manifest_digest"] = prod._sha256_json(
        {
            key: value
            for key, value in forged.items()
            if key not in {"created_utc", "manifest_digest"}
        }
    )
    with pytest.raises(ValueError, match="canonical covariance-group contract"):
        prod.validate_covariance_manifest(forged, config)

    tampered_edges = json.loads(json.dumps(manifest))
    tampered_edges["ell_right"][-1] -= 1
    with pytest.raises(ValueError, match="ell_right"):
        prod.validate_covariance_manifest(tampered_edges, config)


def test_covariance_workspace_cache_key_contains_mask_bytes(tmp_path):
    config = mp.MeasurementConfig(nside=1, lmax=3, n_bins=1, output_dir=str(tmp_path))
    group = {
        "key": ["a", "b", "c", "d"],
        "spins": [0, 0, 0, 0],
        "representative_fields": ["a", "b", "c", "d"],
    }
    fields_a = {
        name: SimpleNamespace(mask=np.ones(12, dtype=np.float32))
        for name in group["representative_fields"]
    }
    fields_b = dict(fields_a)
    fields_b["c"] = SimpleNamespace(mask=np.arange(12, dtype=np.float32))
    path_a = prod._cov_workspace_cache_path(config, group, fields_a)
    path_b = prod._cov_workspace_cache_path(config, group, fields_b)
    assert path_a != path_b


def test_covariance_shard_validation_rejects_mixed_map_product(tmp_path):
    config = mp.MeasurementConfig.for_stage("fast1024")
    manifest = prod.build_covariance_manifest(config)
    group = manifest["groups"][0]
    path = tmp_path / "shard.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["pipeline_version"] = mp.MEASUREMENT_PIPELINE_VERSION
        h5.attrs["covariance_estimator_version"] = mp.COVARIANCE_ESTIMATOR_VERSION
        h5.attrs["covariance_config_digest"] = prod.covariance_config_digest(config)
        h5.attrs["manifest_digest"] = manifest["manifest_digest"]
        h5.attrs["group_digest"] = prod._group_digest(group)
        h5.attrs["map_product_id"] = "map-a"
        h5.attrs["group_mask_digest"] = "mask-a"
        h5.attrs["group_index"] = group["index"]
        blocks = h5.create_group("covariance_blocks")
        for block in group["blocks"]:
            name = f"{block['spec_i']}__x__{block['spec_j']}"
            blocks.create_dataset(name, data=np.eye(config.n_bins))
        h5.create_group("input_cls_for_covariance")

    compatible, reason = prod._covariance_shard_compatibility(
        path, group, manifest, config, "map-a", "mask-a"
    )
    assert (compatible, reason) == (True, "compatible")
    compatible, reason = prod._covariance_shard_compatibility(
        path, group, manifest, config, "map-b", "mask-a"
    )
    assert compatible is False
    assert "map_product_id" in reason
    compatible, reason = prod._covariance_shard_compatibility(
        path, group, manifest, config, "map-a", "mask-b"
    )
    assert compatible is False
    assert "group_mask_digest" in reason


def _write_dual_galaxy_auto_view_fixture(tmp_path, monkeypatch):
    config = mp.MeasurementConfig(
        stage="synthetic",
        nside=1,
        lmax=3,
        lmax_mask=3,
        ell_min=0,
        n_bins=1,
        binning="linear",
        compute_covariance=True,
        output_dir=str(tmp_path),
    )
    specs = mp.default_spectrum_specs()
    spectra = {}
    for index, spec in enumerate(specs):
        spin_a = 2 if str(spec.fields[0]).startswith("s") else 0
        spin_b = 2 if str(spec.fields[1]).startswith("s") else 0
        n_components = mp.ncls_for_spins(spin_a, spin_b)
        cl_all = np.full((n_components, 1), index + 1.0, dtype=np.float64)
        noise_all = None
        if spec.family == "desi_g_auto":
            noise_all = np.full((n_components, 1), 0.25 * (index + 1.0))
        spectra[spec.name] = {
            "name": spec.name,
            "family": spec.family,
            "fields": spec.fields,
            "component": spec.component,
            "component_label": mp.component_labels(spin_a, spin_b)[spec.component],
            "component_labels": mp.component_labels(spin_a, spin_b),
            "label": spec.label,
            "theory_key": spec.theory_key,
            "metadata": dict(spec.metadata),
            "ell": np.asarray([1.5]),
            "cl": cl_all[spec.component].copy(),
            "cl_all_components": cl_all,
            "pcl_all_components": np.zeros((n_components, 4), dtype=np.float64),
            "noise_decoupled_all_components": noise_all,
            "bandpower_window_selected": np.ones((1, 4), dtype=np.float64),
        }
    left, right = mp.make_bandpower_edges(config)
    packed = mp.pack_joint_data_vector(specs, spectra, config, left, right)
    n_data = len(specs)
    covariance = np.diag(np.arange(1.0, n_data + 1.0))
    slices = {spec.name: (index, index + 1) for index, spec in enumerate(specs)}
    joint = {
        "spectrum_names": [spec.name for spec in specs],
        "ell": np.asarray([1.5]),
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
        "cov": covariance,
        "corr": np.eye(n_data),
        "slices": slices,
        "diagnostics": {"synthetic": True},
    }
    result = {
        "schema": mp.measurement_schema_for_config(config),
        "created_utc": mp.utc_now(),
        "config": config.to_dict(),
        "ell": np.asarray([1.5]),
        "ell_left": left,
        "ell_right": right,
        "binning": config.binning,
        "ell_max_inclusive": config.lmax,
        "spectra": spectra,
        "covariance_blocks": {},
        "joint": joint,
        "null_tests": {},
        "input_cls_for_covariance": {},
        "field_metadata": {},
    }
    field = mp.FieldMap(
        name="g1",
        label="synthetic",
        kind="desi_galaxy",
        spin=0,
        maps=[np.zeros(hp.nside2npix(1), dtype=np.float32)],
        mask=np.ones(hp.nside2npix(1), dtype=np.float32),
        mask_name="mask",
    )
    map_metadata = _content_addressed_map_metadata({"g1": field}, config)
    monkeypatch.setattr(
        mp.hp,
        "pixwin",
        lambda nside, lmax, pol: (
            np.ones(lmax + 1, dtype=np.float64),
            np.ones(lmax + 1, dtype=np.float64),
        ),
    )
    output = tmp_path / "dual_views.h5"
    mp.save_measurement_product(output, result, map_metadata)
    return output, packed


def test_dual_galaxy_auto_views_share_covariance_and_preserve_primary_hmc_vector(
    tmp_path, monkeypatch
):
    path, packed = _write_dual_galaxy_auto_view_fixture(tmp_path, monkeypatch)
    with h5py.File(path, "r") as h5:
        mp.validate_measurement_product_identity(h5)
        report = mp.validate_galaxy_auto_views(h5, require=True)
        assert report["galaxy_auto_elements"] == 4
        assert report["changed_elements"] == 4
        assert report["covariance_shared_hard_link"] is True
        total = h5["joint/views/total"]
        subtracted = h5["joint/views/weighted_poisson_subtracted"]
        assert total["data_vector"].id == h5["joint/data_vector"].id
        assert total["cov"].id == h5["joint/cov"].id
        assert subtracted["cov"].id == h5["joint/cov"].id
        np.testing.assert_array_equal(
            subtracted["data_vector_raw"][:],
            packed["data_vector_raw_weighted_poisson_subtracted"],
        )
        np.testing.assert_array_equal(
            h5["joint/data_vector"][:], packed["data_vector"]
        )

    total_loaded = gmt.load_measurement_data(path)
    subtracted_loaded = gmt.load_measurement_data(
        path, galaxy_auto_view="weighted_poisson_subtracted"
    )
    assert total_loaded.galaxy_auto_view == "total"
    assert subtracted_loaded.galaxy_auto_view == "weighted_poisson_subtracted"
    np.testing.assert_array_equal(total_loaded.covariance, subtracted_loaded.covariance)
    assert np.count_nonzero(total_loaded.data_vector != subtracted_loaded.data_vector) == 4


def test_dual_galaxy_auto_view_validator_rejects_a_copied_covariance(tmp_path, monkeypatch):
    path, _ = _write_dual_galaxy_auto_view_fixture(tmp_path, monkeypatch)
    with h5py.File(path, "r+") as h5:
        copied = h5["joint/cov"][:]
        del h5["joint/views/weighted_poisson_subtracted/cov"]
        h5["joint/views/weighted_poisson_subtracted"].create_dataset("cov", data=copied)
    with h5py.File(path, "r") as h5, pytest.raises(ValueError, match="shared joint/cov"):
        mp.validate_galaxy_auto_views(h5, require=True)


def test_highres_covariance_work_plan_balances_only_missing_global_groups():
    config = mp.MeasurementConfig.for_stage("highres4096")
    manifest = prod.build_covariance_manifest(config)
    reused = {116, 237, 257}
    missing = [group for group in manifest["groups"] if int(group["index"]) not in reused]
    bundles = prod._balanced_covariance_bundles(
        missing,
        11,
        stress_group_indices=prod.HIGHRES_RESOURCE_STRESS_GROUP_INDICES,
    )
    indices = [int(group["index"]) for bundle in bundles for group in bundle]
    assert len(indices) == 256
    assert len(indices) == len(set(indices))
    assert not reused.intersection(indices)
    assert set(indices) == set(range(259)) - reused
    assert len(bundles) == 24
    assert [len(bundle) for bundle in bundles] == [11] * 16 + [10] * 8
    assert [int(group["index"]) for group in bundles[0]] == list(
        prod.HIGHRES_RESOURCE_STRESS_GROUP_INDICES
    )


def test_covariance_work_plan_digest_and_group_inventory_are_fail_closed():
    config = mp.MeasurementConfig.for_stage("fast1024")
    manifest = prod.build_covariance_manifest(config)
    bundles = prod._balanced_covariance_bundles(manifest["groups"], 11)
    bundle_payload = [
        {
            "bundle_index": index,
            "group_indices": [int(group["index"]) for group in bundle],
            "group_digests": [prod._group_digest(group) for group in bundle],
            "n_blocks": sum(int(group["n_blocks"]) for group in bundle),
        }
        for index, bundle in enumerate(bundles)
    ]
    plan = {
        "created_utc": "synthetic",
        "version": prod.COVARIANCE_WORK_PLAN_VERSION,
        "stage": config.stage,
        "pipeline_version": mp.MEASUREMENT_PIPELINE_VERSION,
        "covariance_estimator_version": mp.COVARIANCE_ESTIMATOR_VERSION,
        "manifest_digest": manifest["manifest_digest"],
        "covariance_config_digest": prod.covariance_config_digest(config),
        "groups_per_bundle": 11,
        "n_manifest_groups": len(manifest["groups"]),
        "n_reused_groups": 0,
        "n_missing_groups": 259,
        "n_bundles": len(bundle_payload),
        "stress_bundle_index": 0,
        "stress_group_indices": bundle_payload[0]["group_indices"],
        "reused_groups": [],
        "bundles": bundle_payload,
    }
    plan["plan_digest"] = prod._sha256_json(
        {key: value for key, value in plan.items() if key not in {"created_utc", "plan_digest"}}
    )
    prod.validate_covariance_work_plan(plan, manifest, config)

    tampered = json.loads(json.dumps(plan))
    tampered["bundles"][0]["group_indices"][0] = tampered["bundles"][1][
        "group_indices"
    ][0]
    tampered["plan_digest"] = prod._sha256_json(
        {
            key: value
            for key, value in tampered.items()
            if key not in {"created_utc", "plan_digest"}
        }
    )
    with pytest.raises(ValueError, match="identity|more than once|cover"):
        prod.validate_covariance_work_plan(tampered, manifest, config)

    wrong_stage = json.loads(json.dumps(plan))
    wrong_stage["stage"] = "highres4096"
    wrong_stage["plan_digest"] = prod._sha256_json(
        {
            key: value
            for key, value in wrong_stage.items()
            if key not in {"created_utc", "plan_digest"}
        }
    )
    with pytest.raises(ValueError, match="different measurement stage"):
        prod.validate_covariance_work_plan(wrong_stage, manifest, config)


def test_efficient_highres_submission_has_hard_five_node_cap_and_no_remeasurement():
    submit = (
        MODULE_DIR / "submit_multiprobe_highres4096_efficient.sh"
    ).read_text(encoding="utf-8")
    worker = (
        MODULE_DIR / "run_multiprobe_cov_bundle_worker.sbatch"
    ).read_text(encoding="utf-8")
    finalize_worker = (
        MODULE_DIR / "run_multiprobe_finalize_worker.sbatch"
    ).read_text(encoding="utf-8")
    driver = (MODULE_DIR / "run_multiprobe_production.py").read_text(encoding="utf-8")
    assert "--array=\"1-$((n_bundles - 1))%${MAX_NODES}\"" in submit
    assert "^[[1-5]$" not in submit  # sanity: regex remains readable, not ANSI escaped
    assert "if ! [[ \"${MAX_NODES}\" =~ ^[1-5]$ ]]" in submit
    assert " prepare " not in submit
    assert " spectra " not in submit
    assert "--ntasks=11 --cpus-per-task=11" in submit
    assert "--mem=\"${COV_NODE_MEMORY}\"" in submit
    assert "--dependency=\"afterok:${stress_job}\"" in submit
    assert "srun --exclusive --exact" in worker
    assert "--cpus-per-task=11" in worker
    assert "--mem=80G" in worker
    assert "--cov-class all" in worker
    assert "--no-cov-workspace-cache" in worker
    assert "cp -- \"${SOURCE_MAP}\" \"${local_map}.part\"" in worker
    assert "flock -n 9" in submit
    assert submit.index("active_same_name=") < submit.index("make-cov-manifest")
    assert "--plan-path \"${PLAN_PATH}\"" in finalize_worker
    finalize_parser = driver[driver.index('sub.add_parser("finalize")') :]
    assert 'p.add_argument("--plan-path", default=None)' in finalize_parser
