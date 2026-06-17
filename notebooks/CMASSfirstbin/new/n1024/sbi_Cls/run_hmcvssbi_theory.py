# run_hmcvssbi_theory.py
# ──────────────────────────────────────────────────────────────────────────────
# Ports the SBI_validate notebook to the sbi_Cls working directory.
# All imports come from SBI_validate; all outputs go to sbi_Cls/outputs/theory_sbi/.
# The fiducial .npz product is reused from SBI_validate (not regenerated).
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

SBI_CLS_DIR      = pathlib.Path.cwd()           # sbi_Cls/
SBI_VALIDATE_DIR = pathlib.Path(
    "/work/hdd/bdne/aacharya2/GODMAX/notebooks/SBI_validate"
)

for p in [str(SBI_VALIDATE_DIR), str(SBI_CLS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# =============================================================================
# IMPORTS  (all from SBI_validate)
# =============================================================================

from theory_sbi_utils import (
    DEFAULT_FIDUCIAL_PATH,
    THEORY_SBI_DIR,
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
# CONFIGURATION  (identical to SBI_validate notebook)
# =============================================================================

PARAM_SPECS              = default_parameter_specs()
PROBES                   = parse_probe_list("gg,gy,gtau,gkappa")
ELL_MIN                  = 100.0
ELL_MAX                  = 1500.0
THEORY_BACKEND           = "linearized"
FIDUCIAL_OFFSET          = True
SBI_SUMMARY_COMPRESSION  = "score"

HMC_RUN_NAME = "joint_gg_gy_gtau_gkappa_linearized"
SBI_RUN_NAME = "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn5"

SBI_DIAGNOSTIC_RUNS = {
    "raw active NSF":                    "joint_gg_gy_gtau_gkappa_linearized",
    "raw active NSF, discard round 0":   "joint_gg_gy_gtau_gkappa_linearized_al_rounds12_nsf",
    "Fisher NSF, SNPE-C":                "joint_gg_gy_gtau_gkappa_linearized_fisher_nsf64",
    "Fisher NSF, 8x1024 small rounds":   "joint_gg_gy_gtau_gkappa_linearized_fisher_nsf64_8x1024",
    "Fisher MDN5, SNPE-C":               "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn5",
    "Fisher MDN5, SNPE-C 2x sims":       "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn5_2x",
    "Fisher NSF, first-round loss diag": "joint_gg_gy_gtau_gkappa_linearized_fisher_nsf64_ffl",
}

# =============================================================================
# OUTPUT DIRECTORIES  (rooted in sbi_Cls, not SBI_validate)
# =============================================================================

OUTPUT_DIR  = SBI_CLS_DIR / "outputs" / "theory_sbi"
HMC_RUN_DIR = OUTPUT_DIR / HMC_RUN_NAME
SBI_RUN_DIR = OUTPUT_DIR / SBI_RUN_NAME
HMC_RUN_DIR.mkdir(parents=True, exist_ok=True)
SBI_RUN_DIR.mkdir(parents=True, exist_ok=True)

print("SBI_validate dir :", SBI_VALIDATE_DIR)
print("sbi_Cls dir      :", SBI_CLS_DIR)
print("HMC output dir   :", HMC_RUN_DIR)
print("SBI output dir   :", SBI_RUN_DIR)

# =============================================================================
# FIDUCIAL PRODUCT  (reuse from SBI_validate — no regeneration)
# =============================================================================

FIDUCIAL_PATH = ensure_default_fiducial_product(
    DEFAULT_FIDUCIAL_PATH,
    param_specs=PARAM_SPECS,
    force=False,
)
print("Fiducial path    :", FIDUCIAL_PATH)

# =============================================================================
# LOAD SELECTED DATA VECTOR + COVARIANCE
# =============================================================================

selected = selected_product_arrays(
    FIDUCIAL_PATH, probes=PROBES, ell_min=ELL_MIN, ell_max=ELL_MAX
)
product = selected["product"]
meta    = product["metadata"]

print("datavector shape :", selected["data_vector"].shape)
print("cov shape        :", selected["cov"].shape)
print("cov jitter       :", selected["jitter"])
print("fiducial overrides:", meta.get("sim_param_overrides"))
print("theory mode      :", meta.get("theory_mode"))
print("paint R200c factor:", meta.get("paint_r200c_factor"))

eig = np.linalg.eigvalsh(selected["cov"])
print("min/max eigenvalue:", eig.min(), eig.max())

# =============================================================================
# THEORY VECTOR FUNCTION + VALIDATION
# =============================================================================

vector_fn, theory_info = make_inference_theory_vector_function(
    PARAM_SPECS,
    selected["selection"],
    fiducial_vector=selected["data_vector"],
    backend=THEORY_BACKEND,
    fiducial_offset=FIDUCIAL_OFFSET,
    jit_compile=True,
)
checks = validate_theory_vector(vector_fn, selected, PARAM_SPECS)
print("Theory validation:", checks)

# =============================================================================
# PLOT FIDUCIAL DATA VECTOR
# =============================================================================

ell   = product["ell"]
dv    = product["data_vector"]
order = product["spectra_order"]
nell  = len(ell)

fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), sharex=True)
for ax, spec in zip(axes.ravel(), order):
    i = order.index(spec)
    y = dv[i * nell:(i + 1) * nell]
    ax.loglog(ell, np.abs(y), marker="o", ms=3)
    ax.set_title(spec)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$|C_\ell|$")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "fiducial_data_vector.png", dpi=130, bbox_inches='tight')
plt.close(fig)
# =============================================================================
# RUN HMC / SBI  (set flags to True to execute)
# =============================================================================

RUN_HMC = True
RUN_SBI = True

if RUN_HMC:
    run_hmc(
        fiducial_path=FIDUCIAL_PATH,
        output_dir=HMC_RUN_DIR,
        probes=PROBES,
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

if RUN_SBI:
    run_sbi(
        fiducial_path=FIDUCIAL_PATH,
        output_dir=SBI_RUN_DIR,
        probes=PROBES,
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

# =============================================================================
# LOAD RESULTS
# =============================================================================

hmc_path = HMC_RUN_DIR / "hmc_samples.npz"
sbi_path = SBI_RUN_DIR / "sbi_posterior_samples.npz"

hmc = np.load(hmc_path, allow_pickle=True) if hmc_path.exists() else None
sbi = np.load(sbi_path, allow_pickle=True) if sbi_path.exists() else None

print("HMC run dir  :", HMC_RUN_DIR)
print("SBI run dir  :", SBI_RUN_DIR)
print("HMC available:", hmc is not None)
print("SBI available:", sbi is not None)

try:
    import torch
    print("torch CUDA available:", torch.cuda.is_available(),
          "device count:", torch.cuda.device_count())
except Exception as exc:
    print("torch CUDA check failed:", exc)

# =============================================================================
# DIAGNOSTICS
# =============================================================================

if (HMC_RUN_DIR / "hmc_diagnostics.json").exists():
    print("\nHMC diagnostics")
    print(json.dumps(
        json.loads((HMC_RUN_DIR / "hmc_diagnostics.json").read_text()),
        indent=2,
    )[:4000])

if (SBI_RUN_DIR / "sbi_diagnostics.json").exists():
    print("\nSBI diagnostics")
    print(json.dumps(
        json.loads((SBI_RUN_DIR / "sbi_diagnostics.json").read_text()),
        indent=2,
    )[:4000])

# =============================================================================
# GETDIST TRIANGLE PLOT
# =============================================================================

names    = [p.name     for p in PARAM_SPECS]
labels   = [p.label    for p in PARAM_SPECS]
fiducial = np.array([p.fiducial  for p in PARAM_SPECS], dtype=float)
prior_min = np.array([p.prior_min for p in PARAM_SPECS], dtype=float)
prior_max = np.array([p.prior_max for p in PARAM_SPECS], dtype=float)


def hmc_sample_array(hmc_npz):
    return np.column_stack([hmc_npz[f"samples_{name}"] for name in names])


def filter_prior(samples):
    samples = np.asarray(samples, dtype=float)
    mask = np.all(
        (samples >= prior_min[None, :]) & (samples <= prior_max[None, :]),
        axis=1,
    )
    return samples[mask], mask


if hmc is not None and sbi is not None:
    from getdist import MCSamples, plots

    hmc_samples      = hmc_sample_array(hmc)
    sbi_samples_all  = np.asarray(sbi["samples"], dtype=float)
    sbi_samples, sbi_prior_mask = filter_prior(sbi_samples_all)
    print(f"Using {len(sbi_samples)} / {len(sbi_samples_all)} SBI samples "
          f"inside the physical prior.")

    getdist_settings = {"smooth_scale_1D": 0.35, "smooth_scale_2D": 0.35}

    hmc_gd = MCSamples(
        samples=hmc_samples,
        names=names,
        labels=labels,
        label="HMC / NUTS",
        settings=getdist_settings,
    )
    sbi_gd = MCSamples(
        samples=sbi_samples,
        names=names,
        labels=labels,
        label="SBI / SNPE MDN5 (Fisher)",
        settings=getdist_settings,
    )

    g = plots.get_subplot_plotter(width_inch=6.2)
    g.settings.legend_fontsize = 9
    g.settings.axes_labelsize  = 10
    g.triangle_plot(
        [hmc_gd, sbi_gd],
        params=names,
        filled=True,
        legend_labels=["HMC / NUTS", "SBI / SNPE MDN5 (Fisher)"],
        contour_colors=["#1f77b4", "#d62728"],
        markers={name: value for name, value in zip(names, fiducial)},
        marker_args={"color": "black", "lw": 1.2, "ls": "--"},
    )

    # Save to sbi_Cls/outputs/theory_sbi/
    fig_path = OUTPUT_DIR / "hmc_vs_sbi_theory_cls_triangle.pdf"
    g.export(str(fig_path))
    import gc
    gc.collect()
    plt.close('all')
    print(f"Saved triangle plot to {fig_path}")

else:
    print("Run or load both HMC and SBI outputs to make the GetDist overlay.")
