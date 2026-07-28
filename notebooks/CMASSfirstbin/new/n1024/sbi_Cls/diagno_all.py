import numpy as np

for probe, path in [
    ("gy",     "outputs/theory_sbi/gy_linearized/hmc_samples.npz"),
    ("gtau",   "outputs/theory_sbi/gtau_linearized/hmc_samples.npz"),
    ("gkappa", "outputs/theory_sbi/gkappa_linearized/hmc_samples.npz"),
    ("gy+gtau+gkappa", "outputs/theory_sbi/all_2pt_linearized/hmc_samples.npz"),
]:
    d = np.load(path, allow_pickle=True)
    J     = d['theory_jacobian']   # shape (N_ell, 2)
    chol  = d['theory_chol'] if 'theory_chol' in d else d['chol']
    
    chol_diag = np.diag(chol)
    w_t = J[:, 0] / chol_diag
    w_n = J[:, 1] / chol_diag

    F = np.array([
        [np.dot(w_t, w_t), np.dot(w_t, w_n)],
        [np.dot(w_t, w_n), np.dot(w_n, w_n)],
    ])
    det = np.linalg.det(F)
    r   = F[0,1] / np.sqrt(F[0,0] * F[1,1])
    
    sigma_t_marginal = np.sqrt(F[1,1] / det)
    sigma_n_marginal = np.sqrt(F[0,0] / det)
    
    # The degeneracy direction: what combination is actually constrained?
    ratio = np.median(w_n / w_t)

    print(f"\n{probe}:")
    print(f"  correlation r          = {r:.6f}")
    print(f"  sigma_theta (marginal) = {sigma_t_marginal:.4f}  (prior width = 5.0)")
    print(f"  sigma_nu    (marginal) = {sigma_n_marginal:.4f}  (prior width = 0.3)")
    print(f"  median w_nu/w_theta    = {ratio:.4f}  <-- degeneracy direction")
    print(f"  => constrained combo   ~ nu + ({-1/ratio:.3f}) * theta = const")
