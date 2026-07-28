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

PARAM_SPECS = default_parameter_specs()
names = [p.name for p in PARAM_SPECS]
nu_idx = names.index("nu_theta_ej_M")  # adjust exact name if needed
print(f"nu is parameter index {nu_idx}: {names[nu_idx]}")

theta_fid = np.array([p.fiducial for p in PARAM_SPECS], dtype=float)
theta_nu_plus = theta_fid.copy()
theta_nu_plus[nu_idx] += 0.2

v0 = np.array(vector_fn(theta_fid))
v1 = np.array(vector_fn(theta_nu_plus))

print("Max |delta_theory| when nu += 0.2:", np.max(np.abs(v1 - v0)))
print("Per-ell delta:", v1 - v0)
