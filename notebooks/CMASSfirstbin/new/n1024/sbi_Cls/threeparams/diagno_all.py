import numpy as np

PARAM_NAMES  = ["theta_ej_0", "nu_theta_ej_M", "nu_theta_ej_z"]
PRIOR_WIDTHS = [5.0, 0.3, 6.0]

for probe, path in [
    ("gy",             "outputs/theory_sbi/gy_linearized/hmc_samples.npz"),
    ("gtau",           "outputs/theory_sbi/gtau_linearized/hmc_samples.npz"),
    ("gkappa",         "outputs/theory_sbi/gkappa_linearized/hmc_samples.npz"),
    ("gy+gtau+gkappa", "outputs/theory_sbi/all_2pt_linearized/hmc_samples.npz"),
]:
    d = np.load(path, allow_pickle=True)
    J    = d['theory_jacobian']   # shape (N_ell, 3)
    chol = d['theory_chol'] if 'theory_chol' in d else d['chol']

    chol_diag = np.diag(chol)
    W = J / chol_diag[:, None]   # shape (N_ell, 3) — whitened Jacobian

    # 3x3 Fisher matrix
    F = W.T @ W

    print(f"\n{'='*60}")
    print(f"Probe: {probe}")
    print(f"  Jacobian shape : {J.shape}")
    print(f"  Fisher matrix  :")
    for i, n in enumerate(PARAM_NAMES):
        row = "  ".join(f"{F[i,j]:>12.4e}" for j in range(3))
        print(f"    {n:<20} {row}")

    # Marginal sigmas from F^-1
    try:
        F_inv = np.linalg.inv(F)
        print(f"\n  Marginal sigmas (from F^-1):")
        for i, (n, pw) in enumerate(zip(PARAM_NAMES, PRIOR_WIDTHS)):
            sigma = np.sqrt(F_inv[i, i])
            flag  = "⚠ > prior" if sigma > pw else "OK"
            print(f"    sigma({n:<20}) = {sigma:.4f}  "
                  f"(prior width = {pw:.1f})  {flag}")

        # Correlation matrix
        D    = np.sqrt(np.diag(F_inv))
        corr = F_inv / np.outer(D, D)
        print(f"\n  Correlation matrix:")
        header = "  ".join(f"{n:>20}" for n in PARAM_NAMES)
        print(f"    {'':20} {header}")
        for i, ni in enumerate(PARAM_NAMES):
            row = "  ".join(f"{corr[i,j]:>20.6f}" for j in range(3))
            print(f"    {ni:<20} {row}")

    except np.linalg.LinAlgError:
        print("  ⚠ Fisher matrix singular — cannot invert")

    # Eigenvalue decomposition — constrained vs degenerate directions
    eigvals, eigvecs = np.linalg.eigh(F)
    print(f"\n  Eigenvalues (ascending = most degenerate first):")
    for i, ev in enumerate(eigvals):
        sigma_dir = 1.0 / np.sqrt(ev) if ev > 0 else np.inf
        vec_str   = "  ".join(f"{eigvecs[j,i]:+.3f} {PARAM_NAMES[j]}"
                              for j in range(3))
        print(f"    λ={ev:.4e}  σ={sigma_dir:.4f}  direction: [{vec_str}]")

    # Pairwise column correlations (raw Fisher, before inversion)
    print(f"\n  Pairwise Jacobian column correlations:")
    for i in range(3):
        for j in range(i+1, 3):
            r = F[i,j] / np.sqrt(F[i,i] * F[j,j])
            print(f"    r({PARAM_NAMES[i]}, {PARAM_NAMES[j]}) = {r:+.6f}")
