"""Transition-model scan for the cap4800 pz3 closure study.

Builds the resolved Stage-31 theory model with:
  - poweradd alpha in {1.0, 0.85, 0.7} (applied jointly to gg/gm/ge/gy/ky),
  - exact windowed 1-halo-only and 2-halo-only decompositions at alpha=1,
and windows everything through the same measurement bandpower windows used by
build_theory. Also dumps k-space 1h/2h matrices and the matter response
factor Pmm_sup = halofit / Pmm_nfw_tot.

Read-only on products; writes one npz. Run with JAX on GPU or CPU.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys, copy
from pathlib import Path
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import stage31_pz1_backlight_validation as st
import godmax_multiprobe_theory_utils as gmt
import multiprobe_namaster as mpn
import h5py

CONFIG = THIS_DIR / "stage31_pz3_cap4800_mmin11p147538_nside2048_lmax4096.selected.yaml"
ROOT = Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/xDESI/processed/abacus_backlight/stage31_pz3_cap4800_mmin11p147538")
MEAS = ROOT / "measurements/sim_pz3_cap4800_mmin11p147538_nside2048_lmax4096_nbin10_linear.h5"
OUT = ROOT / "measurements/theory_alpha_scan_claude_20260610.npz"

config = st.read_config(CONFIG)
cfg = st.merge_bestfit_params(config)
cfg["metadata"]["lmax"] = 4096
cfg = gmt.compute_desi_nbar_comoving(cfg)
pz_cfg = gmt.config_for_single_desi_pz(cfg, 3)
RESOLVED_CUT = 11.147538

gmt.ensure_godmax_import_paths(Path(pz_cfg["repo_root"]))
import jax.numpy as jnp
from base_class import base_class
from get_Cls import get_Cl
from get_Pkzs import get_Pkz
from get_radial_profiles import Profiles


def build_model(is_cmb, alpha=None, components=False):
    cfgv = copy.deepcopy(pz_cfg)
    if alpha is not None:
        for key in ("alpha_gg", "alpha_gm", "alpha_ge", "alpha_gy", "alpha_ky"):
            cfgv["params"]["other_params"][key] = float(alpha)
    sim_params, halo_params, analysis, other_params = gmt._params_for_model(cfgv, is_cmb_lensing=is_cmb)
    for key in ("gg_transition_model", "tSZ_transition_model",
                "galaxy_matter_transition_model", "galaxy_electron_transition_model"):
        analysis[key] = "poweradd"
    base = base_class(sim_params, halo_params, analysis, other_params)
    profiles = Profiles(sim_params, halo_params, analysis, other_params, base_class_obj=base)
    mass_mask = jnp.asarray(jnp.log10(profiles.M_array) >= RESOLVED_CUT)
    profiles.Ncen_mat = profiles.Ncen_mat * mass_mask[None, :]
    profiles.Nsat_mat = profiles.Nsat_mat * mass_mask[None, :]
    pkz = get_Pkz(sim_params, halo_params, analysis, other_params, Profiles_obj=profiles)
    cls = get_Cl(sim_params, halo_params, analysis, other_params, Pkz_obj=pkz)
    comp = {}
    if components:
        for tag, attr in (("1h", "1h"), ("2h", "2h")):
            pkz_c = copy.copy(pkz)
            pkz_c.Pgg_tot_mat = getattr(pkz, f"Pgg_{attr}_kz_mat")
            pkz_c.Pgm_tot_mat = getattr(pkz, f"Pgm_{attr}_kz_mat")
            pkz_c.Pge_tot_mat = getattr(pkz, f"Pge_{attr}_kz_mat")
            pkz_c.Pgy_tot_mat = getattr(pkz, f"Pgy_{attr}_kz_mat")
            pkz_c.Pym_tot_mat = getattr(pkz, f"Pym_{attr}_kz_mat")
            comp[tag] = get_Cl(sim_params, halo_params, analysis, other_params, Pkz_obj=pkz_c)
    return cls, pkz, comp


def windowed_vector(model_wl, model_cmb):
    theory = st.pz_theory_from_models(model_wl, model_cmb, 3)
    av = np.asarray(cfg["metadata"].get("ksz_default_A_v_by_pz", np.full(4, np.nan)), dtype=np.float64)
    ksz_amplitudes = {3: float(av[2])} if np.isfinite(av[2]) else None
    shear_m = cfg["metadata"].get("shear_m_bias_means")
    vec, names = mpn.theory_to_data_vector(
        MEAS, theory, ell=np.asarray(model_wl.ell_array, dtype=np.float64),
        ksz_velocity_amplitudes=ksz_amplitudes, shear_m_bias=shear_m,
        theory_shear_e_is_positive_kappa=True,
        include_default_pixel_windows=True, include_default_act_beams=True,
    )
    return vec, names

out = {}

# alpha = 1 with exact 1h/2h decomposition
wl, pkz_wl, comp_wl = build_model(False, alpha=1.0, components=True)
cmb, pkz_cmb, comp_cmb = build_model(True, alpha=1.0, components=True)
vec, names = windowed_vector(wl, cmb)
out["names"] = np.asarray(names, dtype=object)
out["windowed_alpha1.0"] = vec
for tag in ("1h", "2h"):
    v, _ = windowed_vector(comp_wl[tag], comp_cmb[tag])
    out[f"windowed_{tag}_alpha1.0"] = v
print("alpha 1.0 + components done", flush=True)

# k-space dumps from the alpha=1 wl model
for name in ("kPk_array", "z_array", "Pgg_1h_kz_mat", "Pgg_2h_kz_mat",
             "Pgm_1h_kz_mat", "Pgm_2h_kz_mat", "Pge_1h_kz_mat", "Pge_2h_kz_mat",
             "Pgy_1h_kz_mat", "Pgy_2h_kz_mat", "Pym_1h_kz_mat", "Pym_2h_kz_mat",
             "Pmm_nfw_1h_kz_mat", "Pmm_nfw_2h_kz_mat", "Pmm_dmb_1h_kz_mat",
             "Pmm_dmb_2h_kz_mat", "phfit_kz_mat", "Pmm_sup_tot_mat",
             "bg_kz_mat", "bm_nfw_kz_mat", "nbarz"):
    if hasattr(pkz_wl, name):
        out[f"k_{name}"] = np.asarray(getattr(pkz_wl, name))

# alpha variants
for alpha in (0.85, 0.7):
    wl_a, _, _ = build_model(False, alpha=alpha)
    cmb_a, _, _ = build_model(True, alpha=alpha)
    v, _ = windowed_vector(wl_a, cmb_a)
    out[f"windowed_alpha{alpha}"] = v
    print(f"alpha {alpha} done", flush=True)

np.savez(OUT, **out)
print("WROTE", OUT)
for k in sorted(out):
    arr = out[k]
    print(" ", k, getattr(arr, "shape", None))
