# compute_linear_d.py
import pathlib, sys
import numpy as np

SBI_CLS_DIR      = pathlib.Path.cwd()
SBI_VALIDATE_DIR = pathlib.Path(
    "/work/hdd/bdne/aacharya2/GODMAX/notebooks/SBI_validate"
)
for p in [str(SBI_VALIDATE_DIR), str(SBI_CLS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from theory_sbi_utils import (
    DEFAULT_FIDUCIAL_PATH,
    default_parameter_specs,
    ensure_default_fiducial_product,
    selected_product_arrays,
    make_linearized_theory_vector_function,
    parse_probe_list,
)

PARAM_SPECS   = default_parameter_specs()
FIDUCIAL_PATH = ensure_default_fiducial_product(DEFAULT_FIDUCIAL_PATH, param_specs=PARAM_SPECS)
ELL_MIN, ELL_MAX = 100.0, 1500.0

# Use whichever probe combination you want the reparameterization tuned for.
# "all_2pt" (combined info) is a reasonable default since that's the
# hardest / most-informative combined inference target.
PROBE_SETS = {
    "gy":      parse_probe_list("gy"),
    "gtau":    parse_probe_list("gtau"),
    "gkappa":  parse_probe_list("gkappa"),
    "all_2pt": parse_probe_list("gy,gtau,gkappa"),
}

def fisher_d(probes):
    selected = selected_product_arrays(
        FIDUCIAL_PATH, probes=probes, ell_min=ELL_MIN, ell_max=ELL_MAX,
    )
    # Reuse the EXACT Jacobian used by the "linearized" backend elsewhere
    # in the pipeline -- guarantees the Fisher matrix here matches the
    # Gaussian model actually driving HMC/SBI.
    _, info = make_linearized_theory_vector_function(
        PARAM_SPECS,
        selected["selection"],
        fiducial_vector=selected["data_vector"],
    )
    jac = info["jacobian"]            # (N_data, 2), columns = (theta_ej_0, nu)
    precision = selected["precision"] # (N_data, N_data)

    fisher = jac.T @ precision @ jac  # (2, 2)
    eigvals, eigvecs = np.linalg.eigh(fisher)   # ascending eigenvalues

    # Smallest-eigenvalue eigenvector of F == largest-eigenvalue
    # (most degenerate) eigenvector of Cov = F^-1.
    dtheta, dnu = eigvecs[:, 0]
    d = float(dnu / dtheta)           # signed -- do NOT take abs()

    cov = np.linalg.inv(fisher)
    sigma_t, sigma_nu = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / (sigma_t * sigma_nu)

    return {
        "fisher": fisher, "eigvals": eigvals, "d": d,
        "sigma_theta_ej_0": sigma_t, "sigma_nu": sigma_nu, "rho": rho,
    }


if __name__ == "__main__":
    for name, probes in PROBE_SETS.items():
        r = fisher_d(probes)
        print(f"\n[{name}]  probes={probes}")
        print(f"  Fisher matrix:\n{r['fisher']}")
        print(f"  Eigenvalues (ascending): {r['eigvals']}")
        print(f"  sigma(theta_ej_0)={r['sigma_theta_ej_0']:.4g}  "
              f"sigma(nu)={r['sigma_nu']:.4g}  rho={r['rho']:.4f}")
        print(f"  => LINEAR_D = {r['d']:.6g}")
