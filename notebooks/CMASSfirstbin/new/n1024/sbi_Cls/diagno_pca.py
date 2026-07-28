import numpy as np, joblib, os

WORK_DIR = '.'
name = 'all_2pt'  # or 'JOINT'

x_train     = np.load(os.path.join(WORK_DIR, f'x_train_full_noisy.npy'))
theta_train = np.load(os.path.join(WORK_DIR, 'theta_train_full.npy'))
x_mean      = np.load(os.path.join(WORK_DIR, f'scaler_{name}_mean.npy'))
x_std       = np.load(os.path.join(WORK_DIR, f'scaler_{name}_std.npy'))
pca         = joblib.load(os.path.join(WORK_DIR, f'pca_{name}.pkl'))
n_comp      = int(np.load(os.path.join(WORK_DIR, f'pca_{name}_n_comp.npy')))

idx = list(range(len(x_mean)))  # JOINT uses all
xt_norm = ((x_train[:, idx] - x_mean) / x_std).astype(np.float32)
xt_pca  = pca.transform(xt_norm)  # all components

# Correlation of EACH PCA component with nu and theta
r_theta = np.array([np.corrcoef(xt_pca[:, i], theta_train[:, 0])[0,1]
                    for i in range(xt_pca.shape[1])])
r_nu    = np.array([np.corrcoef(xt_pca[:, i], theta_train[:, 1])[0,1]
                    for i in range(xt_pca.shape[1])])

cumvar = np.cumsum(pca.explained_variance_ratio_)

print(f"Components kept at 99% variance: {n_comp}")
print(f"\nTop 5 components by |r_nu|:")
for i in np.argsort(np.abs(r_nu))[::-1][:5]:
    kept = "KEPT" if i < n_comp else "DROPPED"
    print(f"  PC{i:3d}: r_nu={r_nu[i]:+.3f}  r_theta={r_theta[i]:+.3f}  "
          f"cumvar={cumvar[i]:.4f}  [{kept}]")

print(f"\nTop 5 components by |r_theta|:")
for i in np.argsort(np.abs(r_theta))[::-1][:5]:
    kept = "KEPT" if i < n_comp else "DROPPED"
    print(f"  PC{i:3d}: r_nu={r_nu[i]:+.3f}  r_theta={r_theta[i]:+.3f}  "
          f"cumvar={cumvar[i]:.4f}  [{kept}]")

# How much nu signal is retained vs dropped?
r2_nu_kept    = np.sum(r_nu[:n_comp]**2)
r2_nu_total   = np.sum(r_nu**2)
print(f"\nFraction of nu R² retained by kept PCs: {r2_nu_kept/r2_nu_total:.3f}")
print(f"Fraction of theta R² retained:           "
      f"{np.sum(r_theta[:n_comp]**2)/np.sum(r_theta**2):.3f}")
