# run_hmcvssbi_theory.py
# ──────────────────────────────────────────────────────────────────────────────
# Runs theory-based HMC and SBI separately for each probe:
#   gy, gtau, gkappa, all_2pt
# Outputs go to sbi_Cls/outputs/theory_sbi/{probe}_linearized/
#                               theory_sbi/{probe}_linearized_fisher_mdn5/
# ──────────────────────────────────────────────────────────────────────────────

import json
import pathlib
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# PATHS
# =============================================================================

SBI_CLS_DIR      = pathlib.Path.cwd()
SBI_VALIDATE_DIR = pathlib.Path(
    "/work/hdd/bdne/aacharya2/GODMAX/notebooks/SBI_validate"
)

for p in [str(SBI_VALIDATE_DIR), str(SBI_CLS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# =============================================================================
# IMPORTS
# =============================================================================

from theory_sbi_utils import (
    DEFAULT_FIDUCIAL_PATH,
    default_parameter_specs,
    ensure_default_fiducial_product,
    make_inference_theory_vector_function,
    parse_probe_list,
    selected_product_arrays,
    validate_theory_vector,
)
from run_hmc_theory_cls import run_hmc
from run_sbi_theory_cls import run_sbi

plt.rcParams.update({"figure.dpi": 130})

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

PARAM_SPECS             = default_parameter_specs()
ELL_MIN                 = 100.0
ELL_MAX                 = 1500.0
THEORY_BACKEND          = "linearized"
FIDUCIAL_OFFSET         = True
SBI_SUMMARY_COMPRESSION = "score"

RUN_HMC = True
RUN_SBI = True

OUTPUT_DIR = SBI_CLS_DIR / "outputs" / "theory_sbi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("SBI_validate dir :", SBI_VALIDATE_DIR)
print("sbi_Cls dir      :", SBI_CLS_DIR)
print("Output root      :", OUTPUT_DIR)

# =============================================================================
# PER-PROBE PROBE-LIST MAPPING
# =============================================================================

PROBE_TO_THEORY_PROBES = {
    "gy":      parse_probe_list("gy"),
    "gtau":    parse_probe_list("gtau"),
    "gkappa":  parse_probe_list("gkappa"),
    "all_2pt": parse_probe_list("gy,gtau,gkappa"),
}

PROBES_TO_RUN = ["gy", "gtau", "gkappa", "all_2pt"]

# =============================================================================
# FIDUCIAL PRODUCT  (shared across all probes — no regeneration)
# =============================================================================

FIDUCIAL_PATH = ensure_default_fiducial_product(
    DEFAULT_FIDUCIAL_PATH,
    param_specs=PARAM_SPECS,
    force=False,
)
print("Fiducial path    :", FIDUCIAL_PATH)

names     = [p.name      for p in PARAM_SPECS]
labels    = [p.label     for p in PARAM_SPECS]
fiducial  = np.array([p.fiducial  for p in PARAM_SPECS], dtype=float)
prior_min = np.array([p.prior_min for p in PARAM_SPECS], dtype=float)
prior_max = np.array([p.prior_max for p in PARAM_SPECS], dtype=float)


def _hmc_sample_array(hmc_npz):
    return np.column_stack([hmc_npz[f"samples_{name}"] for name in names])


def _filter_prior(samples):
    samples = np.asarray(samples, dtype=float)
    mask    = np.all(
        (samples >= prior_min[None, :]) & (samples <= prior_max[None, :]),
        axis=1,
    )
    return samples[mask], mask
# ── DIAGNOSTIC: inspect the raw file before anything processes it ────────────
import numpy as np

_raw = np.load(FIDUCIAL_PATH, allow_pickle=True)
print("\n── RAW FILE DIAGNOSTIC ──")
print("Keys            :", list(_raw.keys()))
print("ell shape       :", _raw['ell'].shape)
print("ell values      :", _raw['ell'])
print("data_vector shape:", _raw['data_vector'].shape)
print("cov shape       :", _raw['cov'].shape)
print("──────────────────────────\n")

# =============================================================================
# MAIN LOOP — one HMC + SBI run per probe
# =============================================================================

for probe_name in PROBES_TO_RUN:
    theory_probes = PROBE_TO_THEORY_PROBES[probe_name]
    hmc_run_name  = f"{probe_name}_linearized"
    sbi_run_name  = f"{probe_name}_linearized_fisher_mdn5"
    hmc_run_dir   = OUTPUT_DIR / hmc_run_name
    sbi_run_dir   = OUTPUT_DIR / sbi_run_name
    hmc_run_dir.mkdir(parents=True, exist_ok=True)
    sbi_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  Probe : {probe_name}  |  theory_probes = {theory_probes}")
    print(f"  HMC   : {hmc_run_dir}")
    print(f"  SBI   : {sbi_run_dir}")
    print(f"{'='*72}")

    # ── Data vector + covariance ──────────────────────────────────────────────
    selected = selected_product_arrays(
        FIDUCIAL_PATH, probes=theory_probes, ell_min=ELL_MIN, ell_max=ELL_MAX
    )

    product = selected["product"]
    meta    = product["metadata"]

    print(f"  datavector shape : {selected['data_vector'].shape}")
    print(f"  cov shape        : {selected['cov'].shape}")
    print(f"  cov jitter       : {selected['jitter']}")
    print(f"  fiducial overrides: {meta.get('sim_param_overrides')}")
    print(f"  theory mode      : {meta.get('theory_mode')}")

    eig = np.linalg.eigvalsh(selected["cov"])
    print(f"  min/max eigenvalue: {eig.min():.4e}  {eig.max():.4e}")

    # ── Fiducial data-vector plot ─────────────────────────────────────────────
    ell   = product["ell"]
    dv    = product["data_vector"]
    order = product["spectra_order"]
    nell  = len(ell)
    ncols = min(len(order), 3)
    nrows = int(np.ceil(len(order) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             sharex=True, squeeze=False)
    for idx, spec in enumerate(order):
        ax = axes[idx // ncols][idx % ncols]
        y  = dv[idx * nell:(idx + 1) * nell]
        ax.loglog(ell, np.abs(y), marker="o", ms=3)
        ax.set_title(spec, fontsize=10)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$|C_\ell|$")
    for idx in range(len(order), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(f"Fiducial data vector — {probe_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"fiducial_data_vector_{probe_name}.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Theory vector + validation ────────────────────────────────────────────
    vector_fn, theory_info = make_inference_theory_vector_function(
        PARAM_SPECS,
        selected["selection"],
        fiducial_vector=selected["data_vector"],
        backend=THEORY_BACKEND,
        fiducial_offset=FIDUCIAL_OFFSET,
        jit_compile=True,
    )
    # ── LINEARIZED JACOBIAN DIAGNOSTIC ──────────────────────────────────────────
    print(f"\n  Linearized backend Jacobian check for probe '{probe_name}':")
    fid_vec = np.array(vector_fn(fiducial))
    print(f"  Fiducial vector (first 5): {fid_vec[:5]}")
    print(f"  Fiducial vector norm     : {np.linalg.norm(fid_vec):.4e}")
    print()

    for i, spec in enumerate(PARAM_SPECS):
        step     = abs(spec.fiducial) * 0.1 if abs(spec.fiducial) > 1e-10 else 0.01
        theta_p  = fiducial.copy(); theta_p[i] += step
        theta_m  = fiducial.copy(); theta_m[i] -= step
        dvec     = (np.array(vector_fn(theta_p)) - np.array(vector_fn(theta_m))) / (2 * step)
        rel      = np.max(np.abs(dvec)) / (np.max(np.abs(fid_vec)) + 1e-300)
        print(f"  {spec.name:<25}  step={step:.4f}  "
              f"max|dv|={np.max(np.abs(dvec)):.4e}  "
              f"rel={rel:.4e}  "
              f"{'⚠ ZERO' if np.max(np.abs(dvec)) < 1e-15 else 'OK'}")
    # ─────────────────────────────────────────────────────────────────────────────
    
    checks = validate_theory_vector(vector_fn, selected, PARAM_SPECS)
    print(f"  Theory validation: {checks}")

    # ── HMC ───────────────────────────────────────────────────────────────────
    if RUN_HMC:
        # Verify vector_fn actually changes with nu at the point HMC will use it
        test_fid = vector_fn(fiducial)
        test_nu  = vector_fn(fiducial + np.array([0.0, 0.05]))
        print(f"  vector_fn change when nudging nu: {np.max(np.abs(test_nu - test_fid)):.4e}")
        # If this prints ~0 → vector_fn is not the one with the FD Jacobian
        # If this prints nonzero → bug is inside run_hmc itself
        run_hmc(
            fiducial_path=FIDUCIAL_PATH,
            output_dir=hmc_run_dir,
            probes=theory_probes,
            param_specs=PARAM_SPECS,
            ell_min=ELL_MIN,
            ell_max=ELL_MAX,
            num_warmup=4000,
            num_samples=4000,
            num_chains=4,
            max_tree_depth=6,
            dense_mass=True,
            seed=42,
            chain_method="vectorized",
            jit_compile=True,
            fiducial_offset=FIDUCIAL_OFFSET,
            theory_backend=THEORY_BACKEND,
            target_accept_prob=0.9,
        )

    # ── SBI ───────────────────────────────────────────────────────────────────
    if RUN_SBI:
        run_sbi(
            fiducial_path=FIDUCIAL_PATH,
            output_dir=sbi_run_dir,
            probes=theory_probes,
            param_specs=PARAM_SPECS,
            ell_min=ELL_MIN,
            ell_max=ELL_MAX,
            simulations_per_round=[4096, 4096, 8192],
            posterior_samples=30000,
            seed=123,
            hidden_features=64,
            num_transforms=5,
            density_estimator_model="mdn",
            num_components=5,
            num_bins=10,
            training_batch_size=256,
            max_num_epochs=200,
            jit_compile=True,
            fiducial_offset=FIDUCIAL_OFFSET,
            theory_backend=THEORY_BACKEND,
            summary_compression=SBI_SUMMARY_COMPRESSION,
            device="auto",
            discard_prior_samples=True,
            retrain_from_scratch=True,
            force_first_round_loss=False,
            num_atoms=20,
            validation_fraction=0.1,
            learning_rate=5.0e-4,
            parameter_transform="fisher",
        )

    # ── Load results + diagnostics ────────────────────────────────────────────
    hmc_path = hmc_run_dir / "hmc_samples.npz"
    sbi_path = sbi_run_dir / "sbi_posterior_samples.npz"

    hmc = np.load(hmc_path, allow_pickle=True) if hmc_path.exists() else None
    sbi = np.load(sbi_path, allow_pickle=True) if sbi_path.exists() else None

    print(f"\n  HMC available : {hmc is not None}  ({hmc_path})")
    print(f"  SBI available : {sbi is not None}  ({sbi_path})")

    diag_hmc = hmc_run_dir / "hmc_diagnostics.json"
    if diag_hmc.exists():
        print(f"\n  HMC diagnostics [{probe_name}]")
        print(json.dumps(json.loads(diag_hmc.read_text()), indent=2)[:2000])

    diag_sbi = sbi_run_dir / "sbi_diagnostics.json"
    if diag_sbi.exists():
        print(f"\n  SBI diagnostics [{probe_name}]")
        print(json.dumps(json.loads(diag_sbi.read_text()), indent=2)[:2000])

    # ── GetDist triangle plot ─────────────────────────────────────────────────
    if hmc is not None and sbi is not None:
        from getdist import MCSamples, plots

        hmc_samples     = _hmc_sample_array(hmc)
        sbi_samples_all = np.asarray(sbi["samples"], dtype=float)
        sbi_samples, _  = _filter_prior(sbi_samples_all)

        print(f"\n  GetDist: HMC {hmc_samples.shape}  "
              f"SBI {sbi_samples.shape} "
              f"(from {len(sbi_samples_all)} raw samples)")

        mc_hmc = MCSamples(
            samples=hmc_samples,
            names=names,
            labels=labels,
            name_tag=f"HMC ({probe_name})",
        )
        mc_sbi = MCSamples(
            samples=sbi_samples,
            names=names,
            labels=labels,
            name_tag=f"SBI ({probe_name})",
        )

        g = plots.get_subplot_plotter(width_inch=10)
        g.triangle_plot(
            [mc_hmc, mc_sbi],
            filled=True,
            legend_labels=["HMC", "SBI (MDN-5)"],
            contour_colors=["steelblue", "tomato"],
        )
        tri_path = OUTPUT_DIR / f"triangle_{probe_name}.png"
        g.export(str(tri_path))
        plt.close("all")
        print(f"  Triangle plot saved → {tri_path}")

    # ── 1-D marginal comparison ───────────────────────────────────────────────
    if hmc is not None and sbi is not None:
        hmc_samples    = _hmc_sample_array(hmc)
        sbi_samples, _ = _filter_prior(np.asarray(sbi["samples"], dtype=float))

        n_params = len(names)
        ncols    = min(n_params, 4)
        nrows    = int(np.ceil(n_params / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.0 * ncols, 3.2 * nrows),
            squeeze=False,
        )

        for idx, (name, label, fid) in enumerate(
            zip(names, labels, fiducial)
        ):
            ax = axes[idx // ncols][idx % ncols]
            h_col = hmc_samples[:, idx]
            s_col = sbi_samples[:, idx]

            lo = min(h_col.min(), s_col.min())
            hi = max(h_col.max(), s_col.max())
            bins = np.linspace(lo, hi, 60)

            ax.hist(h_col, bins=bins, density=True,
                    alpha=0.55, color="steelblue", label="HMC")
            ax.hist(s_col, bins=bins, density=True,
                    alpha=0.55, color="tomato",    label="SBI")
            ax.axvline(fid, color="k", lw=1.4, ls="--", label="fiducial")
            ax.set_xlabel(f"${label}$", fontsize=11)
            ax.set_ylabel("density",    fontsize=9)
            ax.legend(fontsize=8)

        for idx in range(n_params, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(
            f"1-D marginals — {probe_name}  (HMC vs SBI MDN-5)",
            fontsize=12,
        )
        fig.tight_layout()
        marg_path = OUTPUT_DIR / f"marginals_1d_{probe_name}.png"
        fig.savefig(marg_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  1-D marginals saved → {marg_path}")

    # ── Numerical summary table ───────────────────────────────────────────────
    if hmc is not None and sbi is not None:
        hmc_samples    = _hmc_sample_array(hmc)
        sbi_samples, _ = _filter_prior(np.asarray(sbi["samples"], dtype=float))

        summary_rows = []
        header = (
            f"{'param':<14} "
            f"{'fid':>10} "
            f"{'HMC mean':>12} {'HMC std':>10} "
            f"{'SBI mean':>12} {'SBI std':>10} "
            f"{'Δmean/σ_HMC':>13}"
        )
        print(f"\n  {'─'*len(header)}")
        print(f"  {header}")
        print(f"  {'─'*len(header)}")

        for idx, (name, fid) in enumerate(zip(names, fiducial)):
            hm, hs = hmc_samples[:, idx].mean(), hmc_samples[:, idx].std()
            sm, ss = sbi_samples[:, idx].mean(), sbi_samples[:, idx].std()
            delta  = (sm - hm) / hs if hs > 0 else float("nan")
            row = dict(
                param=name, fiducial=float(fid),
                hmc_mean=float(hm), hmc_std=float(hs),
                sbi_mean=float(sm), sbi_std=float(ss),
                delta_mean_over_hmc_std=float(delta),
            )
            summary_rows.append(row)
            print(
                f"  {name:<14} "
                f"{fid:>10.4f} "
                f"{hm:>12.4f} {hs:>10.4f} "
                f"{sm:>12.4f} {ss:>10.4f} "
                f"{delta:>13.3f}"
            )
        print(f"  {'─'*len(header)}")

        summary_path = sbi_run_dir / "summary_table.json"
        summary_path.write_text(json.dumps(summary_rows, indent=2))
        print(f"  Summary table saved → {summary_path}")

    print(f"\n  ✓  Probe '{probe_name}' complete.")

# =============================================================================
# FINAL BANNER
# =============================================================================

print(f"\n{'='*72}")
print("  All probes finished.")
print(f"  Results root : {OUTPUT_DIR}")
print(f"{'='*72}")
