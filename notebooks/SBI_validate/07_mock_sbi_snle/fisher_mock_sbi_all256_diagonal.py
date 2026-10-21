#!/usr/bin/env python3
"""Simple diagonal-covariance Fisher check for gy, g2y, and gy+g2y.

Run from the GODMAX repository root. Uses the native 128x5 + 128x10 master,
its frozen training split, and the runner's affine response in probit u.
This is a global-affine Fisher approximation, NOT local finite differences.
No training-count or replica-count division is applied to the noise variance.
The N(0,I_5) prior is added ONCE; all five parameters are marginalized before
plotting theta_ej_0, alpha_nt, nu_theta_ej_M.

Default covariance: diagonal within-point replica covariance (empirical).
--covariance knox instead uses a conditional noise-only f_sky approximation
from the registered noise HDF5 and fixed_g2_cache.npz. The masked template
auto-spectrum is divided by mean(mask^2); the effective sky fraction is
mean(mask^2)^2/mean(mask^4). This neglects exact mask/centering covariance.
It is deliberately labelled an approximation, not exact masked covariance.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import chi2

PARAMS = ['theta_ej_0', 'alpha_nt', 'mu_beta', 'theta_co_0', 'nu_theta_ej_M']
LOW = np.array([0.5, 0., 0.005, 0.001, -1.])
HIGH = np.array([8., 0.5, 1.5, 0.5, 1.])
BASE = Path('data/SBI_validate/mock_sbi/all256_native')


def sha_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(a.dtype.str.encode('ascii'))
    h.update(np.array(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def component_response(data, key):
    values = data[key][:, :14]       # Stored order: y, kappa, tau.
    points = np.flatnonzero(data['point_split'] == 0)
    means, residuals = [], []
    for point in points:
        rows = values[data['point_index'] == point]
        means.append(rows.mean(axis=0))
        residuals.append(rows - means[-1])
    residuals = np.concatenate(residuals)
    dof = len(residuals) - len(points)
    covariance = residuals.T @ residuals / dof
    covariance = (covariance + covariance.T) / 2
    design = np.column_stack([np.ones(len(points)), data['point_u'][points]])
    coef, _, rank, _ = np.linalg.lstsq(design, np.asarray(means), rcond=None)
    if rank != 6:
        raise ValueError('Affine design does not have full rank')
    return coef[1:].T, np.diag(covariance)


def knox_diagonals(args, data, manifest):
    import h5py
    import healpy as hp
    contract = args.noise_contract
    if contract is None:
        contract = args.master.resolve().parents[4] / (
            'data/SBI_validate/three_probe_mock/validation/noisy_nside1024_ell2048/'
            'noise_contract_tau_snrmatch_gkappa.h5')
    cache_path = args.g2_cache or args.master.parent / 'fixed_g2_cache.npz'
    with contract.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != manifest['estimator']['noise_contract_sha256']:
        raise ValueError('Noise contract differs from the master')
    with h5py.File(contract, 'r') as h:
        mask = np.asarray(h['mask'], float)
        galm = np.asarray(h['fixed_masked_alm/g'])
        noise = np.asarray(h['noise_cls/y_effective'], float)
        window = np.asarray(h['window'], float)
        edges = np.asarray(h['band_edges'])
    with np.load(cache_path, allow_pickle=False) as h:
        qalm = np.asarray(h['g2_alm'])
        cache_manifest = json.loads(h['manifest_json'].item())
    identity = manifest['estimator']
    if sha_array(galm) != identity['fixed_galaxy_alm_sha256']:
        raise ValueError('Fixed galaxy template differs')
    if sha_array(qalm) != identity['g2_alm_sha256'] or sha_array(qalm) != cache_manifest['g2_alm_sha256']:
        raise ValueError('Fixed quadratic template differs')
    if not np.array_equal(edges, data['band_edges']) or window.shape != (14, len(noise)):
        raise ValueError('Knox bandpower windows differ')
    w2, w4 = np.mean(mask**2), np.mean(mask**4)
    fsky = w2**2 / w4
    if not 0 < fsky <= 1:
        raise ValueError('Invalid effective sky fraction')
    ell = np.arange(len(noise))
    # Approximate conditional variance: fixed template crossed with Gaussian y-noise.
    # No C_yy cosmic variance, extra galaxy shot noise, or additional beam factors.
    def variance(alm):
        auto = hp.alm2cl(alm, lmax=len(noise)-1) / w2
        return window**2 @ (auto * noise / ((2*ell+1) * fsky))
    print(f'Knox noise-only mask approximation: f_eff={fsky:.6g}')
    return variance(galm), variance(qalm)


def fisher(jacobian, variance):
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError('Non-positive diagonal covariance; no flooring is applied')
    return jacobian.T @ (jacobian / variance[:, None])


def plot(results, truth, out, mode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {0: r'$\theta_{\rm ej,0}$', 1: r'$\alpha_{\rm nt}$',
              4: r'$\nu^M_{\theta_{\rm ej}}$'}
    pairs = [(0, 1), (0, 4), (1, 4)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    circle = np.array([np.cos(np.linspace(0, 2*np.pi, 400)),
                       np.sin(np.linspace(0, 2*np.pi, 400))])
    for ax, (i, j) in zip(axes, pairs):
        for (name, cov), color in zip(results.items(), ['#2a78d6', '#eb6834', '#1baf7a']):
            vals, vecs = np.linalg.eigh(cov[np.ix_([i, j], [i, j])])
            for mass, style in [(0.68, '-'), (0.95, '--')]:
                ellipse = (vecs * np.sqrt(vals * chi2.ppf(mass, 2))) @ circle
                ax.plot(truth[i]+ellipse[0], truth[j]+ellipse[1], color=color,
                        ls=style, label=name if mass == 0.68 else None)
        ax.axvline(truth[i], color='gray', ls=':', lw=.8)
        ax.axhline(truth[j], color='gray', ls=':', lw=.8)
        ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j]); ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle(f'Fisher control: {mode} diagonal raw covariance; solid 68%, dashed 95%')
    fig.text(.5, .015, 'All five parameters marginalized. Global linear response in probit coordinates; prior included once',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(out / 'fisher_ellipses.png', dpi=180)
    fig.savefig(out / 'fisher_ellipses.pdf')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--master', type=Path, default=BASE / 'gx_g2x_native_all256.npz')
    p.add_argument('--covariance', choices=['empirical', 'knox'], default='empirical')
    p.add_argument('--noise-contract', type=Path)
    p.add_argument('--g2-cache', type=Path)
    p.add_argument('--outdir', type=Path)
    a = p.parse_args()
    with np.load(a.master, allow_pickle=False) as h:
        manifest = json.loads(h['manifest_json'].item())
        data = {k: np.asarray(h[k]) for k in ['gx', 'g2x', 'point_split', 'point_index',
            'point_u', 'theta_reference', 'u_reference', 'replica_count', 'band_edges']}
    if manifest['schema_version'] != 'godmax.mock_sbi.gx_g2x_native_heterogeneous.v3':
        raise ValueError('Unexpected master schema')
    if manifest['parameter_names'] != PARAMS or data['gx'].shape != (1920, 42) or data['g2x'].shape != (1920, 42):
        raise ValueError('Expected the current native all-256 master and parameter order')
    if not np.array_equal(data['replica_count'], np.r_[np.full(128, 5), np.full(128, 10)]):
        raise ValueError('Expected 128x5 + 128x10 replicas')
    if data['point_u'].shape != (256, 5) or set(data['point_split'].tolist()) != {0, 1}:
        raise ValueError('Invalid point design or split')
    split = data['point_split']
    if split[:128].sum() != 13 or split[128:].sum() != 13:
        raise ValueError('Unexpected frozen point split')
    if not np.array_equal(data['point_index'], np.repeat(np.arange(256), data['replica_count'])):
        raise ValueError('Invalid point-major replica table')
    for k in ['gx', 'g2x', 'point_u', 'u_reference', 'theta_reference']:
        if not np.isfinite(data[k]).all():
            raise ValueError(f'Non-finite {k}')
    np.testing.assert_allclose(data['theta_reference'],
        LOW + (HIGH-LOW)*ndtr(data['u_reference']), rtol=0, atol=2e-13)
    print('128x5 + 128x10 = 1920 mock rows; fit uses the frozen 230 training points.')
    print('Response: GLOBAL affine OLS in u, as in the runner; not local finite differences.')
    j1, v1 = component_response(data, 'gx')
    j2, v2 = component_response(data, 'g2x')
    if a.covariance == 'knox':
        v1, v2 = knox_diagonals(a, data, manifest)
    f1, f2 = fisher(j1, v1), fisher(j2, v2)
    matrices = {'gy': f1, 'g2y': f2, 'gy_g2y': f1 + f2}
    scale = (HIGH-LOW)*np.exp(-data['u_reference']**2/2)/np.sqrt(2*np.pi)
    covariances, saved = {}, {}
    for name, f in matrices.items():
        # Add the prior once, invert the FULL 5x5 matrix, then select plotted pairs.
        chol = np.linalg.cholesky(f + np.eye(5))
        inv_chol = np.linalg.solve(chol, np.eye(5))
        cov_u = inv_chol.T @ inv_chol
        cov = scale[:, None] * cov_u * scale[None, :]
        covariances[name] = cov
        saved[f'{name}_fisher_likelihood_u'] = f
        saved[f'{name}_covariance_u'] = cov_u
        saved[f'{name}_covariance_theta_local'] = cov
        print(f'{name:10s} sigma(theta): {np.round(np.sqrt(np.diag(cov)), 5)}')
    out = a.outdir or a.master.parent / f'fisher_diagonal_{a.covariance}'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'fisher_matrices.npz', **saved, parameter_names=PARAMS,
        theta_reference=data['theta_reference'], gy_variance=v1, g2y_variance=v2,
        gy_jacobian_u=j1, g2y_jacobian_u=j2, covariance_method=a.covariance)
    with (out / 'fisher_sigmas.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['probe', *PARAMS])
        for name, cov in covariances.items():
            writer.writerow([name, *np.sqrt(np.diag(cov))])
    plot(covariances, data['theta_reference'], out, a.covariance)
    print(f'Saved matrices, marginal sigmas, and ellipse PNG/PDF to {out}')


if __name__ == '__main__':
    main()
