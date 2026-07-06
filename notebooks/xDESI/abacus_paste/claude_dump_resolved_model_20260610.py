"""Dump resolved/full theory-model internals (nbarz, bias, HMF, HOD) for the
cap4800 pz3 closure investigation. Read-only on products; writes one npz.

Run: JAX_PLATFORMS=cpu python claude_dump_resolved_model_20260610.py
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import sys
from pathlib import Path
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import stage31_pz1_backlight_validation as st

CONFIG = THIS_DIR / "stage31_pz3_cap4800_mmin11p147538_nside2048_lmax4096.selected.yaml"
OUT = Path(
    "/mnt/ceph/users/spandey/ltu-godmax/GODMAX/data/xDESI/processed/abacus_backlight/"
    "stage31_pz3_cap4800_mmin11p147538/measurements/theory_model_dump_claude_20260610.npz"
)

config = st.read_config(CONFIG)
cfg = st.merge_bestfit_params(config)
cfg["metadata"]["lmax"] = 4096
cfg = st.gmt.compute_desi_nbar_comoving(cfg)
pz_cfg = st.gmt.config_for_single_desi_pz(cfg, 3)

kw = dict(
    gg_transition_model="poweradd",
    tsz_transition_model="poweradd",
    galaxy_matter_transition_model="poweradd",
    galaxy_electron_transition_model="poweradd",
)
resolved = st.build_one_godmax_model(pz_cfg, is_cmb_lensing=False, log10_mass_cut=11.147538, **kw)
full = st.build_one_godmax_model(pz_cfg, is_cmb_lensing=False, **kw)

def grab(model, prefix, out):
    for name in [
        "z_array", "M_array", "nbarz", "hmf_Mz_mat", "bias_Mz_mat",
        "Ncen_mat", "Nsat_mat", "bg_kz_mat", "kPk_array", "chi_array",
        "ell_array", "Cl_gal_gal_tot_mat",
    ]:
        if hasattr(model, name):
            out[f"{prefix}{name}"] = np.asarray(getattr(model, name))
    # Wg and Cls z grid
    for name in ["z_array_for_Cls", "chi_array_for_Cls", "dchi_dz_array_for_Cls", "Wg_mat"]:
        if hasattr(model, name):
            out[f"{prefix}{name}"] = np.asarray(getattr(model, name))

out = {}
grab(resolved, "res_", out)
grab(full, "full_", out)
out["res_mult_shear_bias_array"] = np.asarray(resolved.mult_shear_bias_array)
np.savez(OUT, **out)
print("WROTE", OUT)
for k in sorted(out):
    v = out[k]
    print(f"  {k}: shape {v.shape}")
