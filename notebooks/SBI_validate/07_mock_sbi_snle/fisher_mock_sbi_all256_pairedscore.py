#!/usr/bin/env python3
"""Fisher benchmark on the EXACT saved gy/g2y scores, with paired covariance.

Run from the GODMAX repository root. No SBI retraining or new maps are needed.
Reconstructs the saved 5D gy and 5D g2y scores, then concatenates paired rows.
Estimates their full 10x10 within-point noise covariance on the frozen 230
training points (1725 rows; 1495 residual degrees of freedom). No division by
the number of replicas, covariance repair, or eigenvalue flooring is applied.

Also evaluates a block-diagonal control: retain each FULL 5x5 covariance but
set the gy--g2y cross block to zero. The parameter response remains a GLOBAL
affine OLS fit in probit u, not local finite differences. The N(0,I_5) prior
is included once. Ellipses are centred at the input reference by construction.

Also saves one Fisher-versus-SBI figure for each of gy, g2y and gy_g2y.
Each compares the three pairs of theta_ej_0, alpha_nt and nu_theta_ej_M,
marginalizing the other parameters. SBI contours use ALL saved physical
posterior samples and the notebook's 44-bin, sigma=1.35 smoothed histograms.
Both methods show 68% and 95% regions. SBI locations are never recentered;
the Fisher ellipses remain the local physical-coordinate approximation.
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


def require(ok, message):
    if not ok:
        raise ValueError(message)


def manifest(archive):
    return json.loads(archive['manifest_json'].item())


def load_master(path):
    with np.load(path, allow_pickle=False) as h:
        man = manifest(h)
        keys = ['gx', 'g2x', 'gx_data_vector', 'g2x_data_vector', 'point_split',
                'point_index', 'point_u', 'theta_reference', 'u_reference', 'replica_count']
        data = {key: np.asarray(h[key]) for key in keys}
    require(man['schema_version'] == 'godmax.mock_sbi.gx_g2x_native_heterogeneous.v3',
            'Unexpected master schema')
    require(man['parameter_names'] == PARAMS, 'Master parameter order differs')
    require(data['gx'].shape == data['g2x'].shape == (1920, 42),
            'Expected the 128x5 + 128x10 master')
    counts = np.r_[np.full(128, 5), np.full(128, 10)]
    require(np.array_equal(data['replica_count'], counts), 'Replica counts differ')
    require(np.array_equal(data['point_index'], np.repeat(np.arange(256), counts)),
            'Invalid point-major replica pairing')
    split = data['point_split']
    require(split.shape == (256,) and set(split.tolist()) == {0, 1}, 'Invalid split')
    require(split[:128].sum() == split[128:].sum() == 13, 'Frozen split differs')
    require(data['point_u'].shape == (256, 5), 'Invalid parameter design')
    for key, value in data.items():
        require(np.isfinite(value).all(), f'Non-finite master array: {key}')
    np.testing.assert_allclose(data['theta_reference'],
        LOW + (HIGH-LOW)*ndtr(data['u_reference']), rtol=0, atol=2e-13)
    return data, man


def check_posterior(h, data, master_man, master_digest, probe, dimension):
    man = manifest(h)
    require(man['probe'] == probe and man['parameter_names'] == PARAMS,
            f'{probe}: probe or parameter order differs')
    require(man['compression'] == 'score' and man['summary_dimension'] == dimension,
            f'{probe}: expected saved {dimension}D score compression')
    require((man['cohort'], man['training_replica_policy'], man['contract_replica_policy'])
            == ('combined', 'native', 'native'), f'{probe}: expected the native combined run')
    require(man['master_sha256'] == master_digest, f'{probe}: different master file')
    require(man['point_split_sha256'] == master_man['point_split']['sha256'],
            f'{probe}: split hash differs')
    np.testing.assert_array_equal(h['point_split'], data['point_split'])
    np.testing.assert_allclose(h['theta_reference'], data['theta_reference'], rtol=0, atol=0)
    np.testing.assert_allclose(h['u_reference'], data['u_reference'], rtol=0, atol=0)
    return man


def saved_scores(joint_path, single_paths, data, master_man, master_digest):
    """Use saved component maps, and verify them against the individual runs."""
    parts, observations = [], []
    with np.load(joint_path, allow_pickle=False) as joint:
        joint_man = check_posterior(joint, data, master_man, master_digest, 'gy_g2y', 10)
        require(tuple(joint_man['component_order']) == ('gy', 'g2y'),
                'Expected component order [gy, g2y]')
        for component, key in [('gy', 'gx'), ('g2y', 'g2x')]:
            covariance = np.asarray(joint[f'covariance_{component}'], float)
            chol = np.asarray(joint[f'score_cholesky_{component}'], float)
            operator = np.asarray(joint[f'score_operator_{component}'], float)
            reference = np.asarray(joint[f'score_reference_prediction_{component}'], float)
            require(covariance.shape == chol.shape == (14, 14)
                    and operator.shape == (5, 14) and reference.shape == (14,),
                    f'{component}: invalid score contract shapes')
            np.testing.assert_allclose(chol @ chol.T, covariance,
                rtol=1e-12, atol=1e-12*np.max(np.abs(covariance)))
            with np.load(single_paths[component], allow_pickle=False) as single:
                single_man = check_posterior(single, data, master_man, master_digest, component, 5)
                for name, actual in [('covariance', covariance), ('score_operator', operator),
                                     ('score_reference_prediction', reference)]:
                    np.testing.assert_array_equal(actual, single[name],
                        err_msg=f'{component}: joint and individual {name} differ')
                values = np.asarray(data[key][:, :14], float)
                score = (operator @ np.linalg.solve(chol, (values-reference).T)).T
                observed = operator @ np.linalg.solve(chol, data[f'{key}_data_vector'][:14]-reference)
                np.testing.assert_allclose(observed, single_man['observed_summary'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(observed, joint[f'observed_score_{component}'], rtol=0, atol=1e-12)
            parts.append(score)
            observations.append(observed)
        observed = np.concatenate(observations)
        np.testing.assert_allclose(observed, joint_man['observed_summary'], rtol=0, atol=1e-12)
    print('PASS: saved joint and individual component contracts/observations match.')
    return np.concatenate(parts, axis=1), observed


def fit_statistics(scores, data):
    """Subtract a separate point mean; preserve gy/g2y pairing in every row."""
    points = np.flatnonzero(data['point_split'] == 0)
    means, residuals = [], []
    for point in points:
        rows = scores[data['point_index'] == point]
        means.append(rows.mean(axis=0))
        residuals.append(rows-means[-1])
    means = np.asarray(means)
    residuals = np.concatenate(residuals)
    dof = len(residuals)-len(points)
    require((len(points), len(residuals), dof) == (230, 1725, 1495), 'Training layout differs')
    covariance = residuals.T @ residuals / dof
    covariance = (covariance+covariance.T)/2
    np.linalg.cholesky(covariance)  # Fail rather than repair a singular covariance.
    design = np.column_stack([np.ones(len(points)), data['point_u'][points]])
    coef, _, rank, _ = np.linalg.lstsq(design, means, rcond=None)
    require(rank == 6, 'Affine parameter design does not have full rank')
    return coef[1:].T, covariance, dof


def fisher(jacobian, covariance):
    whitened = np.linalg.solve(np.linalg.cholesky(covariance), jacobian)
    return whitened.T @ whitened


def plot(results, truth, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {0: r'$\theta_{\rm ej,0}$', 1: r'$\alpha_{\rm nt}$',
              4: r'$\nu^M_{\theta_{\rm ej}}$'}
    names = ['gy', 'g2y', 'joint: full paired', 'joint: cross block zero']
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#777777']
    angles = np.linspace(0, 2*np.pi, 400)
    circle = np.array([np.cos(angles), np.sin(angles)])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (i, j) in zip(axes, [(0, 1), (0, 4), (1, 4)]):
        for cov, name, color in zip(results.values(), names, colors):
            vals, vecs = np.linalg.eigh(cov[np.ix_([i, j], [i, j])])
            for mass, style in [(0.68, '-'), (0.95, '--')]:
                ellipse = (vecs*np.sqrt(vals*chi2.ppf(mass, 2))) @ circle
                ax.plot(truth[i]+ellipse[0], truth[j]+ellipse[1], color=color,
                        ls=style, label=name if mass == 0.68 else None)
        ax.axvline(truth[i], color='gray', ls=':', lw=.8)
        ax.axhline(truth[j], color='gray', ls=':', lw=.8)
        ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j]); ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Fisher control: saved 5D+5D scores; solid 68%, dashed 95%')
    fig.text(.5, .015, 'All five parameters marginalized. Global linear response in probit coordinates; prior included once.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(out/'fisher_ellipses.png', dpi=180)
    fig.savefig(out/'fisher_ellipses.pdf')
    plt.close(fig)


def load_sbi_samples(paths):
    """Load actual samples; do not assume a fixed 400000/80000 pooled count."""
    samples = {}
    for probe, path in paths.items():
        with np.load(path, allow_pickle=False) as archive:
            man = manifest(archive)
            require('theta' in archive.files, f'{probe}: posterior has no theta samples')
            theta = np.asarray(archive['theta'], dtype=float)
        require(theta.ndim == 2 and theta.shape[1] == len(PARAMS)
                and theta.shape[0] > 0 and np.isfinite(theta).all(),
                f'{probe}: expected finite (N, 5) theta samples')
        require(np.all(theta >= LOW-1e-12) and np.all(theta <= HIGH+1e-12),
                f'{probe}: samples lie outside the physical prior')
        if 'pooled_samples' in man:
            require(theta.shape[0] == man['pooled_samples'],
                    f'{probe}: saved sample count differs from its manifest')
        samples[probe] = theta
    return samples


def credible_density_levels(density, masses=(0.68, 0.95)):
    """Highest-density thresholds on an equally spaced 2D histogram grid."""
    ranked = np.sort(np.asarray(density, dtype=float).ravel())[::-1]
    require(ranked.size > 0 and np.isfinite(ranked).all()
            and ranked.min() >= 0 and ranked.sum() > 0, 'Invalid contour density')
    cumulative = np.cumsum(ranked) / ranked.sum()
    return [float(ranked[min(int(np.searchsorted(cumulative, mass)), ranked.size-1)])
            for mass in masses]


def plot_fisher_sbi(results, samples, truth, out):
    """Three separate comparisons; draw SBI HPD regions, not covariance ellipses."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter

    labels = {0: r'$\theta_{\rm ej,0}$', 1: r'$\alpha_{\rm nt}$',
              4: r'$\nu^M_{\theta_{\rm ej}}$'}
    pairs = [(0, 1), (0, 4), (1, 4)]
    cases = [('gy', 'gy', r'$g\times y$', '#2a78d6'),
             ('g2y', 'g2y', r'$g^2\times y$', '#eb6834'),
             ('gy_g2y', 'gy_g2y_full', r'$gy+g^2y$ (paired 10D)', '#1baf7a')]
    fisher_color = '#303030'
    angles = np.linspace(0, 2*np.pi, 400)
    circle = np.array([np.cos(angles), np.sin(angles)])
    radius95 = np.sqrt(chi2.ppf(0.95, 2))
    written = []

    for probe, fisher_key, title, color in cases:
        chain, covariance = samples[probe], results[fisher_key]
        # Include ALL SBI samples and the entire Fisher 95% region. No clipped
        # sample range is renormalized to obtain an artificially tight contour.
        radius = radius95*np.sqrt(np.diag(covariance))
        lower = np.minimum(chain.min(axis=0), truth-radius)
        upper = np.maximum(chain.max(axis=0), truth+radius)
        padding = 0.06*(upper-lower)
        lower, upper = lower-padding, upper+padding
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
        for ax, (i, j) in zip(axes, pairs):
            h, xe, ye = np.histogram2d(chain[:, i], chain[:, j], bins=44,
                range=[[lower[i], upper[i]], [lower[j], upper[j]]])
            density = gaussian_filter(h.T, 1.35)
            level68, level95 = credible_density_levels(density)
            x, y = 0.5*(xe[:-1]+xe[1:]), 0.5*(ye[:-1]+ye[1:])
            ceiling = float(density.max())*1.001
            for level, alpha in [(level95, 0.10), (level68, 0.15)]:
                ax.contourf(x, y, density, levels=[level, ceiling],
                            colors=[color], alpha=alpha)
            for level, style in [(level95, '--'), (level68, '-')]:
                ax.contour(x, y, density, levels=[level], colors=[color],
                           linestyles=[style], linewidths=1.8)

            values, vectors = np.linalg.eigh(covariance[np.ix_([i, j], [i, j])])
            for mass, style in [(0.68, '-'), (0.95, '--')]:
                ellipse = (vectors*np.sqrt(values*chi2.ppf(mass, 2))) @ circle
                ax.plot(truth[i]+ellipse[0], truth[j]+ellipse[1],
                        color=fisher_color, ls=style, lw=2.0)
            ax.axvline(truth[i], color='gray', ls=':', lw=0.9)
            ax.axhline(truth[j], color='gray', ls=':', lw=0.9)
            ax.set_xlim(lower[i], upper[i]); ax.set_ylim(lower[j], upper[j])
            ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j]); ax.grid(alpha=0.2)

        handles = [Line2D([], [], color=c, ls=style, lw=2, label=f'{name} {mass}%')
                   for name, c in [('Fisher', fisher_color), ('SBI', color)]
                   for mass, style in [(68, '-'), (95, '--')]]
        handles.append(Line2D([], [], color='gray', ls=':', label='reference'))
        fig.suptitle(f'{title}: Fisher and SBI', fontsize=15)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.91),
                   ncol=5, frameon=False, fontsize=10)
        fig.text(0.5, 0.025,
            'Other parameters marginalized. Fisher: local ellipses at reference; '
            'SBI: smoothed sample contours at fitted locations.',
            ha='center', fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.82))
        for extension in ('png', 'pdf'):
            path = out/f'fisher_vs_sbi_{probe}.{extension}'
            fig.savefig(path, dpi=180)
            written.append(path)
        plt.close(fig)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--master', type=Path, default=BASE/'gx_g2x_native_all256.npz')
    parser.add_argument('--run-label', default='combined_native_contractnative')
    parser.add_argument('--posterior-dir', type=Path, help='Defaults to the master directory')
    parser.add_argument('--joint-posterior', type=Path, help='Override the sepscore10 gy_g2y filename')
    parser.add_argument('--outdir', type=Path)
    args = parser.parse_args()
    data, master_man = load_master(args.master)
    with args.master.open('rb') as stream:
        master_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    post_dir = args.posterior_dir or args.master.parent
    prefix = f'mock_snle_all256_{args.run_label}'
    joint_path = args.joint_posterior or post_dir/f'{prefix}_sepscore10_gy_g2y_posterior.npz'
    single_paths = {p: post_dir/f'{prefix}_{p}_posterior.npz' for p in ['gy', 'g2y']}
    scores, observed = saved_scores(joint_path, single_paths, data, master_man, master_digest)
    sbi_samples = load_sbi_samples({**single_paths, 'gy_g2y': joint_path})
    jacobian, covariance, dof = fit_statistics(scores, data)
    block_covariance = covariance.copy()
    block_covariance[:5, 5:] = 0
    block_covariance[5:, :5] = 0
    matrices = {
        'gy': fisher(jacobian[:5], covariance[:5, :5]),
        'g2y': fisher(jacobian[5:], covariance[5:, 5:]),
        'gy_g2y_full': fisher(jacobian, covariance),
        'gy_g2y_block': fisher(jacobian, block_covariance),
    }
    print('128x5 + 128x10 = 1920 mock rows; 230 training points / 1725 training rows.')
    print(f'Full paired 10D score covariance; residual df={dof}; no replica-count division.')
    print('Response: GLOBAL affine OLS in u, not local finite differences.')
    correlation = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    print(f'Max |gy--g2y score correlation| = {np.max(np.abs(correlation[:5, 5:])):.6g}')
    print(f'10D covariance condition number = {np.linalg.cond(covariance):.6g}')
    scale = (HIGH-LOW)*np.exp(-data['u_reference']**2/2)/np.sqrt(2*np.pi)
    covariances, saved = {}, {}
    for name, f in matrices.items():
        chol = np.linalg.cholesky(f+np.eye(5))  # Include the prior ONCE.
        inverse = np.linalg.solve(chol, np.eye(5))
        cov_u = inverse.T @ inverse
        cov = scale[:, None]*cov_u*scale[None, :]
        covariances[name] = cov
        saved[f'{name}_fisher_likelihood_u'] = f
        saved[f'{name}_covariance_u'] = cov_u
        saved[f'{name}_covariance_theta_local'] = cov
        print(f'{name:17s} sigma(theta): {np.round(np.sqrt(np.diag(cov)), 5)}')
    out = args.outdir or args.master.parent/'fisher_pairedscore_full'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'fisher_matrices.npz', **saved, parameter_names=PARAMS,
        score_covariance_full=covariance, score_covariance_block=block_covariance,
        score_correlation=correlation, score_jacobian_u=jacobian, observed_score=observed,
        covariance_dof=dof, theta_reference=data['theta_reference'], u_reference=data['u_reference'],
        master_sha256=master_digest, joint_posterior=str(joint_path),
        response_method='global affine OLS; constant within-point covariance; no covariance derivative',
        component_order=['gy', 'g2y'])
    with (out/'fisher_sigmas.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['probe', *PARAMS])
        for name, cov in covariances.items():
            writer.writerow([name, *np.sqrt(np.diag(cov))])
    plot(covariances, data['theta_reference'], out)
    for path in plot_fisher_sbi(covariances, sbi_samples, data['theta_reference'], out):
        print(f'Saved Fisher/SBI comparison: {path}')
    print(f'Saved matrices, sigmas, and ellipse PNG/PDF to {out}')


if __name__ == '__main__':
    main()
