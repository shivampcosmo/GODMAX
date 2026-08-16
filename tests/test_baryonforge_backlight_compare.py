import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import scipy
import yaml


GODMAX_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GODMAX_ROOT.parent
PIPELINE_DIR = GODMAX_ROOT / "notebooks" / "xDESI" / "baryonforge_compare"
CONFIG_PATH = PIPELINE_DIR / "backlight_compare.yaml"
ASYMPTOTIC_CONFIG_PATH = PIPELINE_DIR / "backlight_compare_asymptotic.yaml"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import common as comparison_common  # noqa: E402
import check_integration_convergence as integration_convergence  # noqa: E402
import paint_baryonforge as baryonforge_painter  # noqa: E402
import projection_convergence as projection_convergence  # noqa: E402
import summarize_integration_change as integration_summary  # noqa: E402
import validate_config as comparison_validate  # noqa: E402


def test_effective_grid_canonicalization_removes_cross_architecture_ulp_jitter():
    xdesi_dir = PIPELINE_DIR.parent
    if str(xdesi_dir) not in sys.path:
        sys.path.insert(0, str(xdesi_dir))
    import abacus_pasting_helpers

    left = np.asarray(
        [
            1.8044760663370383,
            125.24480745890082,
            1696.8554134062015,
            0.6599982216986185,
        ]
    )
    cross_architecture = np.asarray(
        [
            1.8044760663370347,
            125.24480745890128,
            1696.8554134062017,
            0.6599982216986187,
        ]
    )
    changed_grid = left.copy()
    changed_grid[0] *= 1.0 + 1.0e-10

    canonical_left = abacus_pasting_helpers._canonicalize_effective_grid(left)
    canonical_cross_architecture = (
        abacus_pasting_helpers._canonicalize_effective_grid(cross_architecture)
    )
    canonical_changed = abacus_pasting_helpers._canonicalize_effective_grid(
        changed_grid
    )

    np.testing.assert_array_equal(canonical_left, canonical_cross_architecture)
    assert not np.array_equal(canonical_left, canonical_changed)
    assert np.max(np.abs(canonical_left / left - 1.0)) < 5.0e-13

    final_grids = abacus_pasting_helpers._effective_survey_grids(67.11, 1024)
    for grid in final_grids:
        np.testing.assert_array_equal(
            grid,
            abacus_pasting_helpers._canonicalize_effective_grid(grid),
        )
    assert abacus_pasting_helpers.effective_grid_canonicalization_contract() == {
        "method": "decimal_significant_digits",
        "significant_digits": 13,
        "affected_effective_config_paths": [
            "analysis.k_array_survey",
            "analysis.l_array_survey",
            "analysis.dl_array_survey",
            "halo_params.ell_array",
        ],
        "applies_to_catalog_or_profile_values": False,
    }
    with pytest.raises(ValueError, match="must be finite"):
        abacus_pasting_helpers._canonicalize_effective_grid(
            np.asarray([1.0, np.nan])
        )


def test_runtime_manifest_binds_imported_module_not_stale_distribution_metadata():
    manifest = comparison_common.runtime_version_manifest()

    for distribution, module in (("scipy", scipy), ("PyYAML", yaml)):
        record = manifest[distribution]
        resolved = Path(module.__file__).resolve()
        assert record["import_status"] == "ok"
        assert record["imported_version"] == str(module.__version__)
        assert Path(record["resolved_file"]) == resolved
        assert record["resolved_file_sha256"] == comparison_common.sha256_file(resolved)
        expected_metadata_match = record["distribution_metadata_version"] == str(
            module.__version__
        )
        assert record["metadata_matches_import"] is expected_metadata_match


def _portable_config() -> dict:
    """Load the real comparison contract but replace cluster-absolute code paths."""

    config = comparison_common.load_config(CONFIG_PATH)
    config["profiles"] = copy.deepcopy(config["profiles"])
    config["profiles"]["godmax_params"] = str(
        GODMAX_ROOT / "param_files" / "Pge" / "params_baryonforge_backlight_godmax.yaml"
    )
    config["profiles"]["baryonforge_params"] = str(
        WORKSPACE_ROOT
        / "BaryonForge"
        / "examples"
        / "params_baryonforge_backlight_godmax.yaml"
    )
    return config


def _portable_asymptotic_config() -> dict:
    config = comparison_common.load_config(ASYMPTOTIC_CONFIG_PATH)
    config["profiles"] = copy.deepcopy(config["profiles"])
    godmax_params = (
        GODMAX_ROOT
        / "param_files"
        / "Pge"
        / "params_baryonforge_backlight_godmax_asymptotic.yaml"
    )
    baryonforge_params = (
        WORKSPACE_ROOT
        / "BaryonForge"
        / "examples"
        / "params_baryonforge_backlight_godmax.yaml"
    )
    config["profiles"]["godmax_params"] = str(godmax_params)
    config["profiles"]["baryonforge_params"] = str(baryonforge_params)
    config["godmax"]["xdesi_params"] = str(godmax_params)
    config["baryonforge"]["params"] = str(baryonforge_params)
    return config


def test_parameter_crosswalk_rejects_backend_parameter_path_alias_drift():
    config = _portable_config()
    assert comparison_common.validate_parameter_crosswalk(config)["ok"]

    drifted = copy.deepcopy(config)
    drifted["baryonforge"]["params"] = drifted["profiles"]["godmax_params"]
    report = comparison_common.validate_parameter_crosswalk(drifted)
    failed = {item["name"] for item in report["failed"]}
    assert "paths.baryonforge_runtime_params" in failed


def test_asymptotic_variant_selects_and_freezes_the_comparison_subclass():
    native = _portable_config()
    asymptotic = _portable_asymptotic_config()

    assert comparison_common.godmax_profiles_class_path(native) is None
    assert comparison_common.godmax_profiles_class_path(asymptotic) == (
        "matched_godmax_profiles.AsymptoticNormalizationProfiles"
    )
    contract = comparison_common.profile_integration_contract(asymptotic)
    assert contract["godmax"] == {
        "normalization_variant": "asymptotic_total_mass_v1",
        "profiles_class_fqname": (
            "matched_godmax_profiles.AsymptoticNormalizationProfiles"
        ),
        "r_min_R200c": 0.01,
        "r_max_R200c": 128.0,
        "core_integration_method": "uniform_log_trapezoid",
        "core_num_points": 64,
        "num_points_trapz_int": 64,
        "extended_integration_method": "gauss_legendre_log",
        "extended_num_points": 64,
        "max_simultaneous_integration_nodes": 64,
        "quadrature_rule_storage_bytes": 1024,
        "radius_unit": "comoving Mpc/h after multiplication by R200c",
    }


def test_asymptotic_parameter_crosswalk_rejects_limit_drift(tmp_path):
    config = _portable_asymptotic_config()
    assert comparison_common.validate_parameter_crosswalk(config)["ok"]
    params = comparison_common.load_yaml(config["profiles"]["godmax_params"])
    params["analysis"]["comparison_extended_profile_rmax_r200c"] = 8.0
    drifted_path = tmp_path / "drifted_godmax.yaml"
    drifted_path.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
    config["profiles"]["godmax_params"] = str(drifted_path)
    config["godmax"]["xdesi_params"] = str(drifted_path)

    report = comparison_common.validate_parameter_crosswalk(config)
    failed = {item["name"] for item in report["failed"]}
    assert "godmax.asymptotic_rmax_R200c" in failed


def test_asymptotic_subclass_fail_closes_memory_neutral_quadrature(monkeypatch):
    src_dir = GODMAX_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import matched_godmax_profiles as matched

    monkeypatch.setattr(matched.Profiles, "__init__", lambda *args, **kwargs: None)
    analysis = {
        "comparison_extended_profile_rmax_r200c": 128.0,
        "comparison_extended_profile_integration_method": "gauss_legendre_log",
        "comparison_extended_profile_num_points": 64,
        "num_points_trapz_int": 64,
    }
    instance = matched.AsymptoticNormalizationProfiles({}, {}, analysis, {})
    assert instance.integration_rmax_r200c == 128.0
    assert instance.extended_profile_num_points == 64
    assert instance.extended_profile_log_nodes.nbytes == 512
    assert instance.extended_profile_log_weights.nbytes == 512

    too_wide = copy.deepcopy(analysis)
    too_wide["comparison_extended_profile_num_points"] = 65
    with pytest.raises(ValueError, match=r"must be in \[2, 64\]"):
        matched.AsymptoticNormalizationProfiles({}, {}, too_wide, {})

    global_regression = copy.deepcopy(analysis)
    global_regression["num_points_trapz_int"] = 256
    with pytest.raises(ValueError, match="native 64"):
        matched.AsymptoticNormalizationProfiles({}, {}, global_regression, {})
    assert "get_fz_Pnt" not in matched.AsymptoticNormalizationProfiles.__dict__


def test_profile_summary_is_bound_to_the_exact_convergence_decision(tmp_path):
    provenance = {
        "comparison_config_sha256": "candidate-config-sha",
        "godmax_params_sha256": "candidate-params-sha",
        "source_manifest_sha256": "candidate-source-sha",
        "runtime_versions": {"python": {"version": "synthetic-runtime"}},
        "profile_integration_contract": {
            "godmax": {
                "profiles_class_fqname": (
                    "matched_godmax_profiles.AsymptoticNormalizationProfiles"
                ),
                "r_max_R200c": 128.0,
                "num_points_trapz_int": 64,
                "extended_integration_method": "gauss_legendre_log",
                "extended_num_points": 64,
            }
        },
    }

    def write_convergence(
        path,
        *,
        ok,
        config_sha="candidate-config-sha",
        schema=integration_summary.CONVERGENCE_SCHEMA,
    ):
        summary = {
            "schema": schema,
            "config_path": "/synthetic/config.yaml",
            "config_sha256": config_sha,
            "godmax_params_sha256": "candidate-params-sha",
            "profile_source_manifest_sha256": "candidate-source-sha",
            "runtime_manifest_sha256": comparison_common.sha256_json(
                provenance["runtime_versions"]
            ),
            "fixed_tolerance": 1.0e-3,
            "profiles_class_fqname": (
                "matched_godmax_profiles.AsymptoticNormalizationProfiles"
            ),
            "variants": {
                "production_128R_gl64": {
                    "rmax_R200c": 128.0,
                    "n_points": 64,
                    "method": "gauss_legendre_log",
                }
            },
            "pressure_full_chain": {
                "production": "production_128R_gl64",
                "reference": "reference_256R_gl512",
                "max_abs_relative_error": {
                    "production_128R_gl64": 4.0e-4,
                },
                "rebuilt_fields": [
                    "Mtot",
                    "fstar_total",
                    "fstar_central",
                    "fstar_satellite",
                    "fgas",
                    "fclm",
                    "gas_norm",
                    "Mdmb",
                    "Ptot",
                ],
            },
            "acceptance": {
                "production_gas_norm_converged": ok,
                "production_full_chain_HSE_converged": ok,
                "full_chain_rebuild_reproduces_production": True,
            },
            "ok": ok,
        }
        with h5py.File(path, "w") as handle:
            handle.attrs["summary_json"] = json.dumps(summary)

    accepted = tmp_path / "accepted.h5"
    write_convergence(accepted, ok=True)
    record = integration_summary.bind_convergence_evidence(accepted, provenance)
    assert record["ok"] is True
    assert record["binding_complete"] is True
    assert record["candidate_variant"] == "production_128R_gl64"

    rejected = tmp_path / "rejected.h5"
    write_convergence(rejected, ok=False)
    assert (
        integration_summary.bind_convergence_evidence(rejected, provenance)["ok"]
        is False
    )

    wrong_config = tmp_path / "wrong_config.h5"
    write_convergence(wrong_config, ok=True, config_sha="different-config-sha")
    with pytest.raises(ValueError, match="different comparison configs"):
        integration_summary.bind_convergence_evidence(wrong_config, provenance)

    wrong_params = tmp_path / "wrong_params.h5"
    write_convergence(wrong_params, ok=True)
    with h5py.File(wrong_params, "r+") as handle:
        summary = json.loads(handle.attrs["summary_json"])
        summary["godmax_params_sha256"] = "different-params-sha"
        handle.attrs["summary_json"] = json.dumps(summary)
    with pytest.raises(ValueError, match="different godmax_params_sha256"):
        integration_summary.bind_convergence_evidence(wrong_params, provenance)

    legacy = tmp_path / "legacy_v1.h5"
    write_convergence(
        legacy,
        ok=True,
        schema=integration_summary.LEGACY_CONVERGENCE_SCHEMA,
    )
    legacy_record = integration_summary.bind_convergence_evidence(legacy, provenance)
    assert legacy_record["numerical_ok"] is True
    assert legacy_record["semantic_complete"] is False
    assert legacy_record["binding_complete"] is False
    assert legacy_record["ok"] is False
    assert legacy_record["legacy_rejection_reason"] == (
        "legacy_v1_held_fixed_HSE_is_not_acceptance_evidence"
    )

    incomplete_v2 = tmp_path / "incomplete_v2.h5"
    write_convergence(incomplete_v2, ok=True)
    with h5py.File(incomplete_v2, "r+") as handle:
        summary = json.loads(handle.attrs["summary_json"])
        summary.pop("pressure_full_chain")
        handle.attrs["summary_json"] = json.dumps(summary)
    incomplete_record = integration_summary.bind_convergence_evidence(
        incomplete_v2, provenance
    )
    assert incomplete_record["numerical_ok"] is True
    assert incomplete_record["semantic_complete"] is False
    assert incomplete_record["ok"] is False


def test_convergence_scan_consumes_the_configured_reference():
    config = _portable_asymptotic_config()
    config["validation"] = copy.deepcopy(config["validation"])
    config["validation"]["asymptotic_convergence_reference_rmax_R200c"] = 300.0
    config["validation"]["asymptotic_convergence_reference_points"] = 1024

    variants, production, reference = integration_convergence.convergence_variants(
        config
    )
    assert production == "production_128R_gl64"
    assert reference == "reference_300R_gl1024"
    assert variants[production] == (128.0, 64, "gauss_legendre_log")
    assert variants[reference] == (300.0, 1024, "gauss_legendre_log")
    assert variants["points_128R_gl1024"] == (
        128.0,
        1024,
        "gauss_legendre_log",
    )
    assert variants["bound_300R_gl64"] == (
        300.0,
        64,
        "gauss_legendre_log",
    )


def test_nonsingular_projection_recovers_constant_profile_and_hard_support():
    table_radius = np.geomspace(0.1, 10.0, 128)
    density = np.full(table_radius.size, 2.5)
    projected_radius = np.asarray([0.5, 2.0, 7.0])
    line_of_sight = np.sqrt(table_radius[-1] ** 2 - projected_radius**2)

    actual = projection_convergence.project_log_table_nonsingular(
        projected_radius,
        line_of_sight,
        table_radius,
        density,
        n_points=64,
    )

    np.testing.assert_allclose(actual, 2.0 * density[0] * line_of_sight, rtol=1e-12)
    outside = np.asarray([11.0])
    zero = projection_convergence.project_log_table_nonsingular(
        outside,
        np.asarray([0.0]),
        table_radius,
        density,
        n_points=64,
    )
    np.testing.assert_array_equal(zero, np.zeros(1))


def test_nonsingular_projection_refuses_endpoint_extrapolation():
    with pytest.raises(ValueError, match="leaves the explicit 3D table"):
        projection_convergence.project_log_table_nonsingular(
            np.asarray([2.0]),
            np.asarray([20.0]),
            np.geomspace(0.1, 10.0, 64),
            np.geomspace(100.0, 1.0, 64),
            n_points=32,
        )


def test_baryonforge_matched_los_wrapper_preserves_shapes_and_scale_factor():
    class FakeHaloProfile:
        def __init__(self, mass_def=None):
            self.mass_def = mass_def

        def real(self, cosmo, r, M, a):
            return self._real(cosmo, r, M, a)

        def projected(self, cosmo, r, M, a):
            return self._projected(cosmo, r, M, a)

    fake_ccl = SimpleNamespace(
        halos=SimpleNamespace(
            profiles=SimpleNamespace(HaloProfile=FakeHaloProfile)
        )
    )

    class ConstantByMassProfile:
        mass_def = object()

        def real(self, cosmo, r, M, a):
            radius = np.atleast_1d(r)
            mass = np.atleast_1d(M)
            return np.broadcast_to(mass[:, None], (mass.size, radius.size)).copy()

        def projected(self, cosmo, r, M, a):
            raise AssertionError("The matched wrapper must not call native projected().")

    base = ConstantByMassProfile()
    adapter = {
        "projected_profile_integration_method": "nonsingular_gauss_legendre",
        "projected_profile_num_points": 64,
        "projected_profile_los_max_comoving_Mpc": 10.0,
    }
    wrapped = baryonforge_painter.apply_projection_adapter(
        base,
        ccl=fake_ccl,
        adapter=adapter,
        projected_scale_factor_power=1.0,
    )

    radii = np.asarray([0.2, 2.0, 8.0])
    masses = np.asarray([2.0, 5.0])
    scale_factor = 0.4
    expected = np.broadcast_to(
        (2.0 * 10.0 * masses * scale_factor)[:, None],
        (masses.size, radii.size),
    )
    np.testing.assert_allclose(
        wrapped.projected(None, radii, masses, scale_factor),
        expected,
        rtol=2.0e-13,
    )
    assert np.shape(wrapped.projected(None, 1.0, 2.0, scale_factor)) == ()
    assert np.shape(wrapped.projected(None, radii, 2.0, scale_factor)) == radii.shape
    assert np.shape(wrapped.projected(None, 1.0, masses, scale_factor)) == masses.shape
    np.testing.assert_array_equal(
        wrapped.real(None, radii, masses, scale_factor),
        base.real(None, radii, masses, scale_factor),
    )


def test_baryonforge_projection_adapter_is_strictly_opt_in():
    profile = object()
    fake_ccl = SimpleNamespace()
    assert (
        baryonforge_painter.apply_projection_adapter(
            profile,
            ccl=fake_ccl,
            adapter={},
            projected_scale_factor_power=0.0,
        )
        is profile
    )
    assert (
        baryonforge_painter.apply_projection_adapter(
            profile,
            ccl=fake_ccl,
            adapter={"projected_profile_integration_method": "native"},
            projected_scale_factor_power=0.0,
        )
        is profile
    )
    with pytest.raises(ValueError, match="Unsupported adapter"):
        baryonforge_painter.apply_projection_adapter(
            profile,
            ccl=fake_ccl,
            adapter={"projected_profile_integration_method": "unregistered"},
            projected_scale_factor_power=0.0,
        )


def test_projection_support_contract_includes_transverse_radius():
    required = projection_convergence.required_extended_rmax_hmpc(
        np.asarray([10.95]),
        los_cutoff_comoving_mpc=100.0,
        h=0.6711,
    )

    assert required == pytest.approx(np.hypot(10.95, 67.11))
    assert required > 67.11
    assert required < projection_convergence.EXTENDED_GODMAX_RMAX_COMOVING_HMPC
    assert projection_convergence.BARYONFORGE_POINTS_PER_DECADE_SCAN == (
        24,
        64,
        128,
    )


def test_opt_in_godmax_projector_uses_physical_support_and_preserves_legacy():
    jnp = pytest.importorskip("jax.numpy")
    jsi = pytest.importorskip("jax.scipy.integrate")
    from get_sim_maps import setup_sim_map

    table_radius = jnp.linspace(0.1, 3.0, 128)
    density = jnp.ones((table_radius.size, 1, 1))
    projector = setup_sim_map._generic_2D_projection.__wrapped__
    state = SimpleNamespace(
        z_array=jnp.asarray([0.5]),
        rp_array=jnp.asarray([1.0]),
        r_array=table_radius,
        h=0.6711,
        num_points_projected_profile=1024,
        projected_profile_los_max_comoving_mpc=None,
        projected_profile_integration_method="physical_table_cosh",
    )

    physical = projector(state, 0, 0, 0, density)
    expected = 2.0 * np.sqrt((3.0 / 1.5) ** 2 - 1.0**2)
    assert float(physical) == pytest.approx(expected, rel=2.0e-7)

    # The matched production contract requests a finite comoving LOS.  Check
    # the actual core Gauss/cosh path applies that cutoff in physical h^-1 Mpc.
    state.num_points_projected_profile = 128
    state.projected_profile_los_max_comoving_mpc = 1.0
    physical_cutoff = projector(state, 0, 0, 0, density)
    expected_cutoff = 2.0 * state.h / (1.0 + state.z_array[0])
    assert float(physical_cutoff) == pytest.approx(float(expected_cutoff), rel=2.0e-7)

    state.projected_profile_integration_method = "legacy_log_radius"
    legacy = projector(state, 0, 0, 0, density, 1.0, 32)
    rp = state.rp_array[0]
    legacy_rmax = jnp.minimum(jnp.max(state.r_array), rp * 100.0)
    radius = jnp.exp(jnp.linspace(jnp.log(rp * 1.01), jnp.log(legacy_rmax), 32))
    sampled = jnp.exp(
        jnp.interp(
            jnp.log(radius),
            jnp.log(state.r_array / (1.0 + state.z_array[0])),
            jnp.log(density[:, 0, 0]),
        )
    )
    old_formula = 2.0 * jsi.trapezoid(
        radius**2 * sampled / jnp.sqrt(radius**2 - rp**2),
        jnp.log(radius),
    )
    np.testing.assert_array_equal(np.asarray(legacy), np.asarray(old_formula))


def _synthetic_simple_profiles() -> SimpleNamespace:
    return SimpleNamespace(
        model_galaxies=False,
        backreaction=False,
        nz=1,
        M_array=np.asarray([1.0e14]),
        z_array=np.asarray([0.8]),
        r_array=np.geomspace(0.003, 12.0, 12),
        r200c_mat=np.asarray([[1.0]]),
        conc_Mz_mat=np.asarray([[4.0]]),
        rt_mat=np.asarray([[4.0]]),
        r_co_mat=np.asarray([[0.05]]),
        r_ej_mat=np.asarray([[2.0]]),
        beta_mat=np.asarray([[1.5]]),
        Rh_mat=np.asarray([[0.015]]),
        nfw_trunc=True,
        gamma_rhogas=2.0,
        delta_rhogas=7.0,
        A_starcga=0.09,
        M1_starcga=10.0**11.4,
        eta_star=0.3,
        eta_cga=0.6,
        Ob0=0.049,
        Om0=0.3175,
        cosmo_params={"H0": 67.11},
    )


def test_numpy_full_chain_rebuilds_variant_component_amplitudes_and_mdmb():
    profiles = _synthetic_simple_profiles()
    state = integration_convergence.independent_normalizers(
        profiles,
        rmax_r200c=128.0,
        n_points=64,
        method="gauss_legendre_log",
    )
    np.testing.assert_allclose(
        state["fgas"] + state["fstar_central"] + state["fclm"], 1.0
    )

    changed = {name: value.copy() for name, value in state.items()}
    changed["Mtot"] *= 1.2
    changed["gas_norm"] *= 0.8
    changed["nfw_norm"] *= 0.9
    changed["fclm"] *= 1.1
    radius = np.asarray([0.02, 0.2, 1.0])
    original_density = integration_convergence._variant_component_densities(
        profiles, state, 0, 0, radius
    )
    changed_density = integration_convergence._variant_component_densities(
        profiles, changed, 0, 0, radius
    )
    assert not np.array_equal(original_density["gas"], changed_density["gas"])
    assert not np.array_equal(original_density["central"], changed_density["central"])
    assert not np.array_equal(
        original_density["collisionless"], changed_density["collisionless"]
    )

    original_mdmb = integration_convergence._rebuild_mdmb_node(
        profiles, state, 0, 0, 64, "gauss_legendre_log"
    )
    changed_mdmb = integration_convergence._rebuild_mdmb_node(
        profiles, changed, 0, 0, 64, "gauss_legendre_log"
    )
    assert np.all(np.isfinite(original_mdmb) & (original_mdmb > 0.0))
    assert not np.array_equal(original_mdmb, changed_mdmb)


def test_numpy_full_chain_fails_closed_for_unimplemented_profile_modes():
    profiles = _synthetic_simple_profiles()
    profiles.model_galaxies = True
    with pytest.raises(ValueError, match="model_galaxies=false"):
        integration_convergence.independent_normalizers(
            profiles,
            rmax_r200c=128.0,
            n_points=64,
            method="gauss_legendre_log",
        )

    profiles.model_galaxies = False
    profiles.backreaction = True
    with pytest.raises(ValueError, match="backreaction=false"):
        integration_convergence.full_chain_pressure_convergence(
            profiles,
            {},
            {
                "production": (128.0, 64, "gauss_legendre_log"),
                "reference": (256.0, 128, "gauss_legendre_log"),
            },
            "production",
            "reference",
        )


def test_bounded_selection_is_deterministic_mass_stratified_and_inside_cap():
    pytest.importorskip("pymaster")
    import run_bounded_validation as bounded

    masses = np.concatenate(
        [np.full(20, 10.0**midpoint) for midpoint in (13.25, 13.75, 14.25, 14.75)]
    )
    ra = np.linspace(-4.0, 4.0, masses.size)
    dec = np.linspace(-3.0, 3.0, masses.size)
    first, report = bounded.select_stratified_inner_cap_indices(
        ra,
        dec,
        masses,
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        radius_deg=6.0,
        seed=20260804,
    )
    second, second_report = bounded.select_stratified_inner_cap_indices(
        ra,
        dec,
        masses,
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        radius_deg=6.0,
        seed=20260804,
    )

    np.testing.assert_array_equal(first, second)
    assert first.size == np.unique(first).size == 64
    assert (
        report["selected_parent_index_sha256"]
        == second_report["selected_parent_index_sha256"]
    )
    selected_log_mass = np.log10(masses[first])
    for lower, upper in zip(bounded.MASS_EDGES[:-1], bounded.MASS_EDGES[1:]):
        assert (
            np.count_nonzero((selected_log_mass >= lower) & (selected_log_mass < upper))
            == 16
        )
    assert report["maximum_center_separation_deg"] <= 6.0


def _write_synthetic_catalog(path: Path) -> None:
    masses = np.asarray([0.9e13, 1.0e13, 1.0001e13, 2.0e13], dtype=np.float64)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "H0": 67.11,
                "Omega_M": 0.3175,
                "Omega_b": 0.049,
                "sigma8": 0.834,
                "ns": 0.9624,
                "w0": -1.0,
            }
        )
        handle.create_dataset("ra_deg", data=np.asarray([0.0, 0.0, 0.0, 10.0]))
        handle.create_dataset("dec_deg", data=np.zeros(masses.size))
        handle.create_dataset("z", data=np.full(masses.size, 0.8))
        handle.create_dataset("M200c_hMsun", data=masses)
        handle.create_dataset("log10M200c_hMsun", data=np.log10(masses))
        handle.create_dataset("vlos_kms", data=np.zeros(masses.size))
        # The native BaryonForge painter queries pixel centers inside the
        # angular support, so keep even this synthetic halo wider than an
        # NSIDE=1024 pixel while remaining well inside the one-degree buffer.
        handle.create_dataset("R200c_hMpc", data=np.full(masses.size, 0.5))
        handle.create_dataset("DA_hMpc", data=np.full(masses.size, 1000.0))


def _synthetic_catalog_config(path: Path) -> dict:
    config = _portable_config()
    config["catalog"] = {
        "source_h5": str(path),
        "mass_cut_hMsun": 1.0e13,
        "predicate": "M200c_hMsun > 1.0e13",
        "expected_selected_count": 2,
    }
    config["sky_patch"] = {
        "center_ra_deg": 0.0,
        "center_dec_deg": 0.0,
        "radius_deg": 5.0,
        "edge_buffer_deg": 1.0,
        "ordering": "RING",
    }
    config["pasting"] = {
        "max_paint_R200c_factor": 5.0,
        "smooth_profiles": False,
    }
    return config


def _statistics_module():
    pytest.importorskip("pymaster")
    import measure_statistics

    return measure_statistics


def _synthetic_map_attrs(
    backend: str,
    *,
    nside: int = 1,
    catalog_path: str = "/synthetic/shared-catalog.h5",
    halo_count: int = 3,
) -> dict:
    source_manifest = {"synthetic/source.py": "synthetic-source-sha256"}
    effective_body = {"synthetic_effective_godmax_config": True}
    effective_manifest = {
        **effective_body,
        "sha256": comparison_common.sha256_json(effective_body),
    }
    return {
        "schema": comparison_common.MAP_PRODUCT_SCHEMA,
        "backend": backend,
        "comparison_config_path": "/synthetic/backlight_compare.yaml",
        "comparison_config_sha256": "synthetic-config-sha256",
        "godmax_params_path": "/synthetic/godmax.yaml",
        "godmax_params_sha256": "synthetic-godmax-params-sha256",
        "baryonforge_params_path": "/synthetic/baryonforge.yaml",
        "baryonforge_params_sha256": "synthetic-baryonforge-params-sha256",
        "effective_godmax_config_sha256": effective_manifest["sha256"],
        "effective_godmax_config_manifest": effective_manifest,
        "source_manifest": source_manifest,
        "source_manifest_sha256": comparison_common.sha256_json(source_manifest),
        "godmax_git_sha": "synthetic-godmax-git-sha",
        "baryonforge_git_sha": "synthetic-baryonforge-git-sha",
        "godmax_git_dirty": False,
        "baryonforge_git_dirty": False,
        "runtime_versions": {"python": "synthetic"},
        "profile_integration_contract": {
            "godmax": {
                "profiles_class_fqname": "get_radial_profiles.Profiles",
            }
        },
        "profiles_class_fqname": "get_radial_profiles.Profiles",
        "smoke_table": False,
        "max_halos": None,
        "baryonforge_splitjoin_n_jobs": 8,
        "godmax_pixel_workers": 1,
        "split_index": 0,
        "num_splits": 1,
        "n_jobs": 8,
        "catalog_sha256": "synthetic-catalog-sha256",
        "catalog_selection_sha256": "synthetic-selection-sha256",
        "catalog_path": catalog_path,
        "selection_predicate": "M200c_hMsun > 1.0e13",
        "mass_cut_predicate": "M200c_hMsun > 1.0e13",
        "halo_count": halo_count,
        "n_halos_painted": halo_count,
        "complete_catalog_paint": True,
        "nside": nside,
        "ordering": "RING",
        "max_paint_R200c_factor": 5.0,
        "smooth_profiles": False,
        "halo_only": True,
        "z_min": 0.63,
        "z_max": 0.98,
        "h": 0.6711,
        "H0": 67.11,
        "Omega_M": 0.3175,
        "Omega_b": 0.049,
        "map_semantics": comparison_common.MAP_SEMANTICS,
        "noise_policy": comparison_common.NOISE_POLICY,
        "mass_proxy_semantics": comparison_common.MASS_PROXY_SEMANTICS,
        "provisional_status": comparison_common.PROVISIONAL_STATUS,
        "provisional_reasons": list(comparison_common.PROVISIONAL_REASONS),
        "analysis_mask_policy": (
            "none in map product; one inner-cap mask is applied by measure_statistics.py"
        ),
        "cmb_source_redshift": 1100.0,
        "unit_boundary": {
            "catalog_mass": "M200c_hMsun in Msun/h",
            "catalog_radius": "R200c_hMpc is physical Mpc/h",
            "catalog_distance": (
                "DA_hMpc is physical angular-diameter distance in Mpc/h"
            ),
            "map_ymap": "dimensionless Compton-y",
            "map_kappa_cmb": "dimensionless halo-only CMB convergence",
        },
    }


def test_real_parameter_files_satisfy_the_explicit_crosswalk():
    report = comparison_common.validate_parameter_crosswalk(_portable_config())

    assert report["ok"] is True
    assert report["failed"] == []
    checks = {item["name"]: item for item in report["checks"]}
    for required in (
        "mass_pivot.log10_Mc0->M_c",
        "simple_stars.eta_delta",
        "godmax.backreaction",
        "mass_definition",
        "concentration",
        "electron_pressure_factor",
        "paint_cutoff",
        "internal_smoothing",
        "strict_mass_predicate",
    ):
        assert checks[required]["ok"] is True


def test_mass_radius_density_conversions_preserve_shell_mass():
    h = 0.6711
    mass_godmax_hmsun = np.asarray([1.0e13, 3.0e14, 1.0e15])
    radius_godmax_hmpc = np.asarray([0.02, 0.5, 5.0])
    density_baryonforge_msun_mpc3 = np.asarray([2.0e10, 7.0e12, 4.0e14])

    mass_baryonforge_msun = mass_godmax_hmsun / h
    radius_baryonforge_mpc = radius_godmax_hmpc / h
    density_godmax_msun_h2_mpc3 = density_baryonforge_msun_mpc3 / h**2

    np.testing.assert_allclose(mass_baryonforge_msun * h, mass_godmax_hmsun)
    np.testing.assert_allclose(radius_baryonforge_mpc * h, radius_godmax_hmpc)
    np.testing.assert_allclose(
        density_godmax_msun_h2_mpc3 * h**2,
        density_baryonforge_msun_mpc3,
    )

    shell_volume_factor = 4.0 * np.pi / 3.0
    shell_mass_godmax_hmsun = (
        density_godmax_msun_h2_mpc3 * shell_volume_factor * radius_godmax_hmpc**3
    )
    shell_mass_baryonforge_msun = (
        density_baryonforge_msun_mpc3 * shell_volume_factor * radius_baryonforge_mpc**3
    )
    np.testing.assert_allclose(
        shell_mass_baryonforge_msun,
        shell_mass_godmax_hmsun / h,
    )


def test_catalog_validation_uses_strict_mass_cut_and_keeps_outer_buffer(tmp_path):
    catalog_path = tmp_path / "synthetic_backlight.h5"
    _write_synthetic_catalog(catalog_path)
    config = _synthetic_catalog_config(catalog_path)

    report = comparison_validate.validate_catalog(config, chunk_rows=2)

    assert report["ok"] is True
    assert report["n_parent"] == 4
    # The row exactly at 1e13 is excluded: only 1.0001e13 and 2e13 survive.
    assert report["n_selected_buffered"] == 2
    assert report["n_selected_inner_cap_centers"] == 1
    assert report["n_selected_outer_buffer_centers"] == 1
    assert report["predicate"] == "M200c_hMsun > 1.0e13"
    assert report["edge_buffer_safe"] is True
    assert report["max_paint_angle_deg"] < report["edge_buffer_deg"]
    assert report["mass_h_roundtrip_max_relative_error"] < 1.0e-15

    cosmology_report = comparison_validate.validate_catalog_cosmology(config, report)
    assert cosmology_report == {"ok": True, "mismatches": {}}


def test_cap_mask_is_binary_ring_map_containing_the_requested_center():
    hp = pytest.importorskip("healpy")
    nside = 8
    ra_deg = 27.421875
    dec_deg = -24.62431835216408

    mask = comparison_common.cap_mask(nside, ra_deg, dec_deg, radius_deg=12.0)

    assert mask.shape == (hp.nside2npix(nside),)
    assert mask.dtype == np.float64
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert 0 < np.count_nonzero(mask) < mask.size
    center_pixel = hp.ang2pix(nside, ra_deg, dec_deg, lonlat=True, nest=False)
    assert mask[center_pixel] == 1.0


def test_map_reader_accepts_aliases_and_preserves_nested_provenance(tmp_path):
    path = tmp_path / "native_alias_maps.h5"
    ymap = np.arange(12, dtype=np.float32)
    kappa = -np.arange(12, dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.attrs["nside"] = 1
        handle.attrs["ordering"] = "RING"
        maps = handle.create_group("maps")
        maps.create_dataset("y", data=ymap)
        maps.create_dataset("kappa", data=kappa)
        provenance = handle.create_group("provenance")
        provenance.attrs["catalog_sha256"] = "synthetic-catalog-hash"
        provenance.attrs["selection_predicate"] = "M200c_hMsun > 1.0e13"

    maps, attrs = comparison_common.read_map_file(path)

    assert set(maps) == {"map_ymap", "map_kappa_cmb"}
    assert maps["map_ymap"].dtype == np.float64
    assert maps["map_kappa_cmb"].dtype == np.float64
    np.testing.assert_array_equal(maps["map_ymap"], ymap)
    np.testing.assert_array_equal(maps["map_kappa_cmb"], kappa)
    assert attrs["nside"] == 1
    assert attrs["ordering"] == "RING"
    assert attrs["path"] == str(path.resolve())
    assert attrs["provenance"] == {
        "catalog_sha256": "synthetic-catalog-hash",
        "selection_predicate": "M200c_hMsun > 1.0e13",
    }


def test_current_contract_binds_catalog_bytes_and_rejects_a_matched_stale_pair(
    tmp_path, monkeypatch
):
    stats = _statistics_module()
    config_path = tmp_path / "comparison.yaml"
    godmax_params = tmp_path / "godmax.yaml"
    baryonforge_params = tmp_path / "baryonforge.yaml"
    catalog_path = tmp_path / "selected_catalog.h5"
    config_path.write_text("schema: synthetic\n", encoding="utf-8")
    godmax_params.write_text("synthetic: godmax\n", encoding="utf-8")
    baryonforge_params.write_text("synthetic: baryonforge\n", encoding="utf-8")
    selected_redshift = np.asarray([0.63, 0.71, 0.98], dtype=np.float64)
    catalog_attrs = {
        "selection_rows": selected_redshift.size,
        "log10_m_min_hmsun": 13.0,
        "h": 0.6711,
        "H0": 67.11,
        "Omega_M": 0.3175,
        "Omega_b": 0.049,
    }
    with h5py.File(catalog_path, "w") as handle:
        handle.attrs.update(catalog_attrs)
        handle.create_dataset("z", data=selected_redshift)

    config = {
        "_config_path": str(config_path),
        "catalog": {
            "output_h5": str(catalog_path),
            "predicate": "M200c_hMsun > 1.0e13",
        },
        "profiles": {
            "godmax_params": str(godmax_params),
            "baryonforge_params": str(baryonforge_params),
        },
        "baryonforge": {"n_jobs": 8},
        "pasting": {
            "pixel_workers": 1,
            "nside": 1024,
            "max_paint_R200c_factor": 5.0,
            "smooth_profiles": False,
        },
        "sky_patch": {"ordering": "RING"},
    }
    monkeypatch.setattr(
        comparison_common,
        "effective_godmax_config_manifest",
        lambda *_args, **_kwargs: {"sha256": "synthetic-effective-sha256"},
    )
    monkeypatch.setattr(
        comparison_common,
        "comparison_source_manifest",
        lambda: {"synthetic/source.py": "synthetic-source-sha256"},
    )
    monkeypatch.setattr(
        comparison_common,
        "profile_integration_contract",
        lambda _config: {"synthetic_profile_integration": True},
    )
    monkeypatch.setattr(
        comparison_common,
        "projected_profile_contract",
        lambda _config: {"synthetic_projected_profile": True},
    )
    monkeypatch.setattr(
        comparison_common, "git_revision", lambda _path: "synthetic-git"
    )
    monkeypatch.setattr(comparison_common, "git_is_dirty", lambda _path: False)

    expected = comparison_common.current_map_contract(config)

    assert expected["catalog_path"] == str(catalog_path.resolve())
    assert expected["catalog_sha256"] == comparison_common.sha256_file(catalog_path)
    assert expected["halo_count"] == selected_redshift.size
    assert expected["selection_predicate"] == "M200c_hMsun > 1.0e13"
    assert expected["z_min"] == pytest.approx(0.63)
    assert expected["z_max"] == pytest.approx(0.98)

    godmax_attrs = _synthetic_map_attrs(
        "godmax", catalog_path=str(catalog_path), halo_count=selected_redshift.size
    )
    baryonforge_attrs = _synthetic_map_attrs(
        "baryonforge", catalog_path=str(catalog_path), halo_count=selected_redshift.size
    )
    godmax_attrs["catalog_sha256"] = "same-stale-catalog-sha256"
    baryonforge_attrs["catalog_sha256"] = "same-stale-catalog-sha256"
    with pytest.raises(ValueError, match="catalog_sha256 differs from current"):
        stats._check_shared_provenance(
            godmax_attrs,
            baryonforge_attrs,
            expected_contract={"catalog_sha256": expected["catalog_sha256"]},
        )


def test_contract_freeze_rejects_any_input_change_before_publication():
    frozen = {
        "catalog_sha256": "catalog-at-start",
        "comparison_config_sha256": "config-at-start",
        "source_manifest": {"model.py": "source-at-start"},
    }
    comparison_common.assert_map_contract_unchanged(
        frozen, copy.deepcopy(frozen), context="synthetic publication"
    )

    changed = copy.deepcopy(frozen)
    changed["catalog_sha256"] = "catalog-after-paint"
    with pytest.raises(RuntimeError, match="refusing to publish.*catalog_sha256"):
        comparison_common.assert_map_contract_unchanged(
            frozen, changed, context="synthetic publication"
        )


def test_source_manifest_covers_transitive_local_painter_dependencies():
    manifest = comparison_common.comparison_source_manifest()
    required = {
        "GODMAX/notebooks/xDESI/abacus_lightcone_catalog.py",
        "GODMAX/notebooks/xDESI/baryonforge_compare/matched_godmax_profiles.py",
        "GODMAX/src/helpers/constants.py",
        "GODMAX/src/helpers/jax_cosmo_power.py",
        "GODMAX/src/hmf_symbolic.py",
        "GODMAX/src/matter_pk_symbolic.py",
        "BaryonForge/BaryonForge/Profiles/misc.py",
        "BaryonForge/BaryonForge/utils/Parallelize.py",
        "BaryonForge/BaryonForge/utils/constants.py",
    }
    assert required.issubset(manifest)
    assert all(len(digest) == 64 for digest in manifest.values())


def test_godmax_annotation_uses_selected_row_redshift_extrema(tmp_path):
    import paint_godmax

    catalog_path = tmp_path / "filtered_catalog_with_stale_parent_attrs.h5"
    with h5py.File(catalog_path, "w") as handle:
        handle.attrs.update(
            {
                "z_min": 0.60,
                "z_max": 1.00,
                "h": 0.6711,
                "H0": 67.11,
                "Omega_M": 0.3175,
                "Omega_b": 0.049,
            }
        )
        handle.create_dataset("z", data=np.asarray([0.6302, 0.75, 0.9798]))

    map_path = tmp_path / "godmax_native.h5"
    with h5py.File(map_path, "w") as handle:
        handle.attrs["profiles_class_fqname"] = "get_radial_profiles.Profiles"
    config = {
        "_config_path": str(CONFIG_PATH),
        "catalog": {
            "output_h5": str(catalog_path),
            "predicate": "M200c_hMsun > 1.0e13",
        },
        "profiles": {
            "godmax_params": str(
                GODMAX_ROOT
                / "param_files"
                / "Pge"
                / "params_baryonforge_backlight_godmax.yaml"
            )
        },
        "pasting": {
            "max_paint_R200c_factor": 5.0,
            "smooth_profiles": False,
        },
    }

    report = paint_godmax.annotate_product(
        map_path,
        config,
        1024,
        0,
        1,
        contract_override=_synthetic_map_attrs("godmax"),
    )

    assert report["z_min"] == pytest.approx(0.6302)
    assert report["z_max"] == pytest.approx(0.9798)
    with h5py.File(map_path, "r") as handle:
        assert handle.attrs["z_min"] == pytest.approx(0.6302)
        assert handle.attrs["z_max"] == pytest.approx(0.9798)


def test_statistics_pure_helpers_enforce_shared_geometry_and_conventions():
    stats = _statistics_module()
    maps = {
        "map_ymap": np.arange(12, dtype=np.float64),
        "map_kappa_cmb": np.arange(12, dtype=np.float64) / 10.0,
    }
    attrs = _synthetic_map_attrs("godmax")
    baryonforge_attrs = _synthetic_map_attrs("baryonforge")

    assert (
        stats._infer_and_validate_nside(maps, maps, attrs, attrs, expected_nside=1) == 1
    )
    provenance = stats._check_shared_provenance(attrs, baryonforge_attrs)
    assert provenance["catalog_sha256"]["status"] == "equal"

    mismatched = dict(attrs)
    mismatched["catalog_sha256"] = "different-catalog"
    with pytest.raises(ValueError, match="same comparison object"):
        stats._check_shared_provenance(mismatched, baryonforge_attrs)

    missing_required = dict(baryonforge_attrs)
    missing_required.pop("halo_count")
    with pytest.raises(ValueError, match="halo_count.missing_one"):
        stats._check_shared_provenance(attrs, missing_required)

    partial = dict(attrs)
    partial["complete_catalog_paint"] = False
    partial["n_halos_painted"] = 1
    with pytest.raises(ValueError, match="complete_catalog_paint"):
        stats._check_shared_provenance(partial, baryonforge_attrs)

    with pytest.raises(ValueError, match="differs from current config/source"):
        stats._check_shared_provenance(
            attrs,
            baryonforge_attrs,
            expected_contract={"comparison_config_sha256": "new-config-sha256"},
        )

    bad_manifest = copy.deepcopy(baryonforge_attrs)
    bad_manifest["source_manifest"]["synthetic/source.py"] = "changed-source"
    with pytest.raises(ValueError, match="source_manifest_sha256"):
        stats._check_shared_provenance(attrs, bad_manifest)

    values = np.asarray([1.0, 3.0, 99.0])
    mask = np.asarray([1.0, 2.0, 0.0])
    masked, weighted_mean = stats._masked_map(values, mask, subtract_weighted_mean=True)
    assert weighted_mean == pytest.approx(7.0 / 3.0)
    np.testing.assert_allclose(masked, [1.0 - 7.0 / 3.0, 3.0 - 7.0 / 3.0, 0.0])

    pair = stats._pair_summary(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 4.0, 6.0]),
    )
    assert pair["difference_convention"] == "BaryonForge minus GODMAX"
    assert pair["gain_through_origin"] == pytest.approx(2.0)
    assert pair["pearson_r"] == pytest.approx(1.0)
    assert pair["relative_rmse_to_godmax"] == pytest.approx(1.0)


def test_statistics_residual_spectrum_closures_are_exact_nulls():
    stats = _statistics_module()
    spectra = {
        "godmax_yy": {"cl": np.asarray([4.0, 16.0])},
        "baryonforge_yy": {"cl": np.asarray([9.0, 25.0])},
        "cross_backend_yy": {"cl": np.asarray([6.0, 20.0])},
        "residual_yy": {"cl": np.asarray([1.0, 1.0])},
        "godmax_kk": {"cl": np.asarray([16.0, 36.0])},
        "baryonforge_kk": {"cl": np.asarray([25.0, 49.0])},
        "cross_backend_kk": {"cl": np.asarray([20.0, 42.0])},
        "residual_kk": {"cl": np.asarray([1.0, 1.0])},
        "godmax_yk": {"cl": np.asarray([8.0, 24.0])},
        "baryonforge_yk": {"cl": np.asarray([15.0, 35.0])},
        "baryonforge_y_godmax_k": {"cl": np.asarray([12.0, 30.0])},
        "godmax_y_baryonforge_k": {"cl": np.asarray([10.0, 28.0])},
        "residual_yk": {"cl": np.asarray([1.0, 1.0])},
    }

    closures = stats._closure_diagnostics(spectra)

    for field in ("yy", "kk", "yk"):
        np.testing.assert_array_equal(closures[field]["difference"], 0.0)
        assert closures[field]["max_abs_difference"] == 0.0
        assert closures[field]["max_abs_difference_over_reference_scale"] == 0.0


def test_statistics_end_to_end_nonidentical_and_zero_map_nulls(tmp_path):
    hp = pytest.importorskip("healpy")
    stats = _statistics_module()
    nside = 8
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    godmax_y = 2.0e-6 * (0.6 * np.sin(theta) * np.cos(phi) + 0.4 * np.cos(2.0 * phi))
    godmax_kappa = 1.0e-3 * (0.5 * np.cos(theta) + 0.3 * np.sin(3.0 * phi))
    baryonforge_y = 0.94 * godmax_y + 1.0e-7 * np.sin(4.0 * phi)
    baryonforge_kappa = 0.97 * godmax_kappa - 2.0e-5 * np.cos(5.0 * phi)

    def write_maps(path, ymap, kappa, backend):
        attrs = _synthetic_map_attrs(
            backend,
            nside=nside,
            catalog_path=str(tmp_path / "synthetic_shared_catalog.h5"),
        )
        with h5py.File(path, "w") as handle:
            handle.attrs["nside"] = nside
            handle.attrs["ordering"] = "RING"
            maps = handle.create_group("maps")
            maps.create_dataset("map_ymap", data=np.asarray(ymap, dtype=np.float64))
            maps.create_dataset(
                "map_kappa_cmb", data=np.asarray(kappa, dtype=np.float64)
            )
            provenance = handle.create_group("provenance")
            provenance.attrs["json"] = comparison_common.canonical_json(attrs)

    godmax_path = tmp_path / "godmax.h5"
    baryonforge_path = tmp_path / "baryonforge.h5"
    zero_godmax_path = tmp_path / "zero_godmax.h5"
    zero_baryonforge_path = tmp_path / "zero_baryonforge.h5"
    write_maps(godmax_path, godmax_y, godmax_kappa, "godmax")
    write_maps(baryonforge_path, baryonforge_y, baryonforge_kappa, "baryonforge")
    write_maps(zero_godmax_path, np.zeros(npix), np.zeros(npix), "godmax")
    write_maps(
        zero_baryonforge_path,
        np.zeros(npix),
        np.zeros(npix),
        "baryonforge",
    )

    config_path = tmp_path / "statistics.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sky_patch": {
                    "center_ra_deg": 27.421875,
                    "center_dec_deg": -24.62431835216408,
                    "radius_deg": 60.0,
                    "ordering": "RING",
                },
                "pasting": {"nside": nside},
                "statistics": {
                    "apodization_deg": 0.0,
                    "ell_min": 2,
                    "lmax": 16,
                    "n_bins": 3,
                    "binning": "linear",
                    "subtract_weighted_mean": True,
                    "deconvolve_pixel_window": False,
                    "n_iter": 0,
                    "n_iter_mask": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    nonidentical_output = tmp_path / "nonidentical_statistics.h5"
    report = stats.measure(
        SimpleNamespace(
            config=str(config_path),
            godmax_maps=str(godmax_path),
            baryonforge_maps=str(baryonforge_path),
            output=str(nonidentical_output),
            nside=None,
            overwrite=False,
            _allow_synthetic_provenance=True,
        )
    )
    assert report["n_spectra"] == len(stats.SPECTRUM_SPECS) == 13
    assert report["radial_stacks_included"] is False
    with h5py.File(nonidentical_output, "r") as handle:
        assert handle.attrs["noise_policy"] == comparison_common.NOISE_POLICY
        assert (
            handle.attrs["provisional_status"] == comparison_common.PROVISIONAL_STATUS
        )
        assert handle.attrs["comparison_config_sha256"] == "synthetic-config-sha256"
        assert set(handle["spectra"]) == {name for name, _, _ in stats.SPECTRUM_SPECS}
        for group in handle["spectra"].values():
            assert np.all(np.isfinite(group["cl"][:]))
        for group in handle["null_tests"]["linear_residual_closure"].values():
            assert np.all(np.isfinite(group["difference"][:]))

    zero_output = tmp_path / "zero_statistics.h5"
    stats.measure(
        SimpleNamespace(
            config=str(config_path),
            godmax_maps=str(zero_godmax_path),
            baryonforge_maps=str(zero_baryonforge_path),
            output=str(zero_output),
            nside=None,
            overwrite=False,
            _allow_synthetic_provenance=True,
        )
    )
    with h5py.File(zero_output, "r") as handle:
        for group in handle["spectra"].values():
            np.testing.assert_array_equal(group["cl"][:], 0.0)
        summary_scalars = []
        for field_group in handle["map_summaries"].values():
            for variant in field_group.values():
                summary_scalars.extend(
                    float(value)
                    for value in variant.attrs.values()
                    if isinstance(value, (int, float, np.integer, np.floating))
                )
        assert np.all(np.isfinite(summary_scalars))
def test_registered_projection_change_classification_is_fail_closed():
    assert integration_summary._is_projected_baryonforge_dataset(
        "baryonforge", "y_projected"
    )
    assert integration_summary._is_projected_baryonforge_dataset(
        "baryonforge", "sigma_matter_physical_Msun_Mpc2"
    )
    assert integration_summary._is_projected_baryonforge_dataset(
        "baryonforge_tabulated_for_painter", "kappa_cmb"
    )
    assert not integration_summary._is_projected_baryonforge_dataset(
        "baryonforge", "rho_gas"
    )
