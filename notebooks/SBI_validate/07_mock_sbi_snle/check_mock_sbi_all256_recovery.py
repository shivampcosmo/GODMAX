#!/usr/bin/env python3
"""Read-only diagnostics for the native 128x5 + 128x10 gy/g2y SBI experiment.

Uses NumPy, SciPy and Matplotlib only; no new maps or neural-network fitting.
Checks full/block Fisher modes, the affine mean on held-out points, saved
held-out NLE log probabilities, and an observation-conditioned Gaussian.
Outputs are model diagnostics, NOT SBC or a test of posterior coverage.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve_triangular
from scipy.special import ndtr
from scipy.stats import chi2

PARAMS = ['theta_ej_0', 'alpha_nt', 'mu_beta', 'theta_co_0', 'nu_theta_ej_M']
LOW = np.array([.5, 0., .005, .001, -1.])
HIGH = np.array([8., .5, 1.5, .5, 1.])
BASE = Path('data/SBI_validate/mock_sbi/all256_native')
PROBES = ['gy', 'g2y', 'gy_g2y']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def covariance(draws, weights=None):
    mean = np.average(draws, axis=0, weights=weights)
    delta = draws - mean
    w = np.full(len(draws), 1 / len(draws)) if weights is None else weights / weights.sum()
    cov = (delta * w[:, None]).T @ delta
    np.linalg.cholesky(cov)
    return mean, cov


def posterior_path(base, label, probe, offset=None):
    marker = '_sepscore10' if probe == 'gy_g2y' else ''
    if offset is None:
        return base / f'mock_snle_all256_{label}{marker}_{probe}_posterior.npz'
    path = base / 'ablations' / f'mock_snle_all256_{label}_abl_baseline_s{offset}{marker}_{probe}_posterior.npz'
    return posterior_path(base, label, probe) if offset == 0 and not path.is_file() else path


def load_inputs(args):
    master_path = args.master or args.base / 'gx_g2x_native_all256.npz'
    with np.load(master_path, allow_pickle=False) as h:
        master_man = json.loads(h['manifest_json'].item())
        keys = ['gx', 'g2x', 'gx_data_vector', 'g2x_data_vector', 'point_u',
                'point_index', 'point_split', 'replica_count', 'u_reference', 'theta_reference']
        data = {k: np.asarray(h[k]) for k in keys}
        if 'noise_seeds' in h.files:
            data['noise_seeds'] = np.asarray(h['noise_seeds'])
    require(master_man['parameter_names'] == PARAMS, 'Master parameter order differs')
    require(master_man['schema_version'] == 'godmax.mock_sbi.gx_g2x_native_heterogeneous.v3',
            'Unexpected master schema')
    counts = np.r_[np.full(128, 5), np.full(128, 10)]
    require(data['gx'].shape == data['g2x'].shape == (1920, 42), 'Expected 1920 paired rows')
    np.testing.assert_array_equal(data['replica_count'], counts)
    np.testing.assert_array_equal(data['point_index'], np.repeat(np.arange(256), counts))
    require(data['point_u'].shape == (256, 5), 'Invalid point_u shape')
    split = data['point_split']
    require(split.shape == (256,) and set(split.tolist()) == {0, 1}
            and split[:128].sum() == split[128:].sum() == 13, 'Frozen split differs')
    for k, value in data.items():
        require(np.isfinite(value).all(), f'Non-finite master {k}')
    noise_audit = {'available': 'noise_seeds' in data}
    if noise_audit['available']:
        order = master_man.get('seed_field_order')
        require(order is not None and 'y' in order, 'Missing canonical noise seed field order')
        require(data['noise_seeds'].shape == (1920, 3), 'Invalid noise seed table')
        y_seeds = data['noise_seeds'][:, order.index('y')]
        for i in range(256):
            point_seeds = y_seeds[data['point_index'] == i]
            require(len(np.unique(point_seeds)) == len(point_seeds), 'Repeated y noise seed within a point')
        training_rows = split[data['point_index']] == 0
        train_seeds = set(y_seeds[training_rows].tolist())
        heldout_seeds = set(y_seeds[~training_rows].tolist())
        noise_audit.update(unique_y_seeds=len(set(y_seeds.tolist())),
                           shared_train_heldout_y_seeds=len(train_seeds & heldout_seeds),
                           repeated_y_seed_rows=len(y_seeds)-len(set(y_seeds.tolist())))
        print('Noise seed audit:', noise_audit)
        if noise_audit['repeated_y_seed_rows']:
            print('NOTE: noise seeds recur across parameter points; fitted-mean uncertainty and point SE below use an independent-point approximation. Treat them descriptively, not as significance tests.')
    np.testing.assert_allclose(data['theta_reference'], LOW + (HIGH - LOW) * ndtr(data['u_reference']),
                               rtol=0, atol=2e-13)
    with master_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    paths = {p: posterior_path(args.base, args.run_label, p) for p in PROBES[:2]}
    joint_paths = ([posterior_path(args.base, args.run_label, 'gy_g2y', s) for s in args.seed_offsets]
                   if args.seed_offsets is not None else [posterior_path(args.base, args.run_label, 'gy_g2y')])
    require(len(set(joint_paths)) == len(joint_paths), 'Duplicated joint files')
    posts = {p: [] for p in PROBES}
    for probe, files in [(p, [paths[p]]) for p in PROBES[:2]] + [('gy_g2y', joint_paths)]:
        for path in files:
            with np.load(path, allow_pickle=False) as h:
                man = json.loads(h['manifest_json'].item())
                require(man['probe'] == probe and man['parameter_names'] == PARAMS, f'{path.name}: wrong probe/order')
                require((man['cohort'], man['training_replica_policy'], man['contract_replica_policy'])
                        == ('combined', 'native', 'native'), f'{path.name}: expected combined/native/native')
                require(man['compression'] == 'score' and man['summary_dimension'] == (10 if probe == 'gy_g2y' else 5),
                        f'{path.name}: wrong summary')
                require(man['master_sha256'] == digest
                        and man['point_split_sha256'] == master_man['point_split']['sha256'], f'{path.name}: master/split differs')
                for k in ['point_split', 'u_reference', 'theta_reference']:
                    np.testing.assert_array_equal(h[k], data[k])
                observed = np.asarray(man['observed_summary'], float)
                if 'observed_summary' in h.files:
                    np.testing.assert_allclose(h['observed_summary'], observed, rtol=0, atol=1e-12)
                u, theta = np.asarray(h['u'], float), np.asarray(h['theta'], float)
                require(u.ndim == 2 and u.shape == theta.shape and u.shape[1] == 5
                        and len(u) == man['pooled_samples'] and np.isfinite(u).all(), f'{path.name}: invalid samples')
                np.testing.assert_allclose(theta, LOW + (HIGH - LOW) * ndtr(u), rtol=0, atol=2e-12)
                components = ['gy', 'g2y'] if probe == 'gy_g2y' else [probe]
                if probe == 'gy_g2y':
                    require(man['component_order'] == components, 'Wrong joint component order')
                contracts = {}
                for component in components:
                    suffix = '_' + component if probe == 'gy_g2y' else ''
                    contracts[component] = {k: np.asarray(h[k + suffix], float)
                        for k in ['covariance', 'score_operator', 'score_reference_prediction']}
                    if probe == 'gy_g2y':
                        contracts[component]['cholesky'] = np.asarray(h['score_cholesky_' + component], float)
                logs = {k: np.asarray(h[k], float) for k in h.files if k.startswith('heldout_log_prob_seed_')}
                n = int(man['network_seeds'])
                require(len(u) == n * man['posterior_samples_per_network'], 'Pooled sample count differs')
                post = dict(path=path, man=man, u=u, theta=theta, networks=n,
                            observed=observed, contracts=contracts, logs=logs)
            posts[probe].append(post)
    reference = posts['gy_g2y'][0]
    training_seeds = []
    for post in posts['gy_g2y']:
        for name in ['training', 'mcmc']:
            require(post['man'][name] == reference['man'][name], f'Joint {name} settings changed')
        np.testing.assert_allclose(post['observed'], reference['observed'], rtol=0, atol=1e-12)
        records = post['man'].get('per_network', [])
        if records:
            require(len(records) == post['networks'], 'Incomplete network metadata')
            training_seeds.extend(r['training_seed'] for r in records)
    require(len(training_seeds) == len(set(training_seeds)), 'Duplicated joint training seeds')
    score_parts, obs_parts = [], []
    for component, key in [('gy', 'gx'), ('g2y', 'g2x')]:
        c = reference['contracts'][component]
        require(c['covariance'].shape == c['cholesky'].shape == (14, 14)
                and c['score_operator'].shape == (5, 14)
                and c['score_reference_prediction'].shape == (14,), 'Invalid component shapes')
        np.testing.assert_allclose(c['cholesky'] @ c['cholesky'].T, c['covariance'],
                                   rtol=1e-12, atol=1e-12*np.abs(c['covariance']).max())
        for post in posts['gy_g2y'] + posts[component]:
            for name in ['covariance', 'score_operator', 'score_reference_prediction']:
                np.testing.assert_array_equal(post['contracts'][component][name], c[name])
        score_parts.append(solve_triangular(c['cholesky'],
            (data[key][:, :14] - c['score_reference_prediction']).T, lower=True).T @ c['score_operator'].T)
        obs = c['score_operator'] @ solve_triangular(c['cholesky'],
            data[key + '_data_vector'][:14] - c['score_reference_prediction'], lower=True)
        np.testing.assert_allclose(obs, posts[component][0]['observed'], rtol=0, atol=1e-12)
        obs_parts.append(obs)
    observed = np.concatenate(obs_parts)
    np.testing.assert_allclose(observed, reference['observed'], rtol=0, atol=1e-12)
    print('PASS: native layout, frozen split, paired component contracts, and observation match.')
    return data, posts, np.concatenate(score_parts, axis=1), observed, digest, noise_audit


def affine_fit(scores, data):
    index, split = data['point_index'], data['point_split']
    train = np.flatnonzero(split == 0)
    means = np.asarray([scores[index == i].mean(axis=0) for i in range(256)])
    residual = np.concatenate([scores[index == i] - means[i] for i in train])
    dof = len(residual) - len(train)
    require(dof == 1495, 'Unexpected residual degrees of freedom')
    C = residual.T @ residual / dof
    np.linalg.cholesky(C)
    X = np.column_stack([np.ones(256), data['point_u']])
    coefficients, _, rank, _ = np.linalg.lstsq(X[train], means[train], rcond=None)
    require(rank == 6, 'Affine fit has insufficient rank')
    inverse = np.linalg.inv(X[train].T @ X[train])
    # OLS coefficient uncertainty for heterogeneous mean-noise C/R_i.
    coefficient_scale = inverse @ (X[train].T @ (X[train] / data['replica_count'][train, None])) @ inverse
    leverage = np.einsum('ij,jk,ik->i', X, coefficient_scale, X)
    return coefficients[0], coefficients[1:].T, C, means, leverage, dof


def gaussian(a, J, C, observed):
    L = np.linalg.cholesky(C)
    W = solve_triangular(L, J, lower=True)
    F = W.T @ W
    S = np.linalg.solve(np.eye(5) + F, np.eye(5))
    mean = S @ W.T @ solve_triangular(L, observed - a, lower=True)
    return mean, (S + S.T) / 2, F


def csv_write(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--master', type=Path)
    parser.add_argument('--fisher', type=Path)
    parser.add_argument('--run-label', default='combined_native_contractnative')
    parser.add_argument('--seed-offsets', nargs='+', type=int, help='Optional equal-weight joint seed ensembles, e.g. 0 1 ... 9')
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--gaussian-draws', type=int, default=50000)
    args = parser.parse_args()
    require(args.gaussian_draws >= 1000, 'Use at least 1000 Gaussian draws')
    if args.seed_offsets is not None:
        require(len(set(args.seed_offsets)) == len(args.seed_offsets) and min(args.seed_offsets) >= 0,
                'Seed offsets must be distinct and nonnegative')
    data, posts, scores, observed, digest, noise_audit = load_inputs(args)
    a, J, C, means, leverage, dof = affine_fit(scores, data)
    Cblock = C.copy(); Cblock[:5, 5:] = 0; Cblock[5:, :5] = 0
    cases = {'gy': (slice(0, 5), C[:5, :5]), 'g2y': (slice(5, 10), C[5:, 5:]),
             'gy_g2y': (slice(0, 10), C), 'block': (slice(0, 10), Cblock)}
    models = {p: gaussian(a[s], J[s], c, observed[s]) for p, (s, c) in cases.items()}
    fisher_path = args.fisher or args.base / 'fisher_pairedscore_full' / 'fisher_matrices.npz'
    with np.load(fisher_path, allow_pickle=False) as f:
        require(tuple(f['parameter_names'].astype(str)) == tuple(PARAMS), 'Fisher parameter order differs')
        require(str(f['master_sha256'].item()) == digest, 'Fisher master digest differs')
        np.testing.assert_allclose(f['score_covariance_full'], C, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(f['score_jacobian_u'], J, rtol=1e-10, atol=1e-12)
        for p, tag in [('gy', 'gy'), ('g2y', 'g2y'), ('gy_g2y', 'gy_g2y_full'), ('block', 'gy_g2y_block')]:
            np.testing.assert_allclose(f[tag + '_covariance_u'], models[p][1], rtol=1e-8, atol=1e-10)
    print('PASS: reconstructed covariance, affine response, and Fisher matrices reproduce the saved benchmark.')
    draws, weights, moments = {}, {}, {}
    for p in PROBES:
        draws[p] = np.concatenate([post['u'] for post in posts[p]])
        total_networks = sum(post['networks'] for post in posts[p])
        weights[p] = np.concatenate([np.full(len(post['u']), post['networks'] / (total_networks * len(post['u'])))
                                      for post in posts[p]])
        moments[p] = covariance(draws[p], weights[p])
        print(f'{p}: {len(posts[p])} runs, {total_networks} networks, {len(draws[p]):,} posterior samples')
    mode_rows = []
    for single in PROBES[:2]:
        for selected, Cs, Cj in [('Fisher', models[single][1], models['gy_g2y'][1]),
                                 ('SBI', moments[single][1], moments['gy_g2y'][1])]:
            _, vectors = eigh(Cs, Cj)
            v = vectors[:, -1].copy(); v /= np.linalg.norm(v)
            if v[np.argmax(np.abs(v))] < 0: v *= -1
            sf = np.sqrt(v @ models[single][1] @ v)
            sb = np.sqrt(v @ models['block'][1] @ v)
            sj = np.sqrt(v @ models['gy_g2y'][1] @ v)
            ss = np.sqrt(v @ moments[single][1] @ v)
            sn = np.sqrt(v @ moments['gy_g2y'][1] @ v)
            row = dict(relative_to=single, selected_by=selected,
                       sigma_single_F=sf, sigma_single_SBI=ss,
                       sigma_joint_block=sb, sigma_joint_full=sj, sigma_joint_SBI=sn,
                       gain_block=sf/sb, gain_full=sf/sj, gain_SBI=ss/sn,
                       block_to_full_width_ratio=sb/sj)
            row.update({f'coefficient_{p}': x for p, x in zip(PARAMS, v)})
            mode_rows.append(row)
            print(f'\n{selected}-selected mode relative to {single}:')
            print('q = ' + ' '.join(f'{x:+.5f} u[{p}]' for p, x in zip(PARAMS, v)))
            print(f'  joint SD: block={sb:.6f}, full={sj:.6f}, SBI={sn:.6f}')
            print(f'  gains: block={sf/sb:.3f}, full={sf/sj:.3f}, SBI={ss/sn:.3f}')
    heldout = np.flatnonzero(data['point_split'] == 1)
    row_mask = data['point_split'][data['point_index']] == 1
    heldout_rows = np.flatnonzero(row_mask)
    predicted = a + data['point_u'] @ J.T
    point_rows, log_rows, summaries = [], [], {}
    for p in PROBES:
        s, c = cases[p]
        L = np.linalg.cholesky(c); dimension = c.shape[0]
        delta = means[heldout, s] - predicted[heldout, s]
        whitened = solve_triangular(L, delta.T, lower=True).T
        mean_scale = 1 / data['replica_count'][heldout] + leverage[heldout]
        Q = np.sum(whitened**2, axis=1) / mean_scale
        for i, q, lev in zip(heldout, Q, leverage[heldout]):
            point_rows.append(dict(probe=p, point_index=int(i), replicas=int(data['replica_count'][i]),
                                   mean_Q=float(q), summary_dimension=dimension, Q_per_dimension=float(q/dimension),
                                   fitted_mean_uncertainty_scale=float(lev)))
        residual = scores[heldout_rows, s] - predicted[data['point_index'][heldout_rows], s]
        wr = solve_triangular(L, residual.T, lower=True).T
        gaussian_logp = -.5 * (dimension*np.log(2*np.pi) + 2*np.log(np.diag(L)).sum() + np.sum(wr**2, axis=1))
        obs_delta = observed[s] - a[s] - J[s] @ data['u_reference']
        obs_white = solve_triangular(L, obs_delta, lower=True)
        summaries[p] = dict(heldout_points=len(heldout), heldout_rows=len(heldout_rows),
                            mean_Q_per_dimension=float(np.mean(Q)/dimension),
                            median_Q_per_dimension=float(np.median(Q)/dimension),
                            reference_affine_residual_Q=float(obs_white @ obs_white))
        print(f'\n{p}: held-out mean-response Q/d: mean={np.mean(Q)/dimension:.3f}, median={np.median(Q)/dimension:.3f}')
        print('  Approximate affine-Gaussian expectation is 1; this is not posterior coverage.')
        for run_index, post in enumerate(posts[p]):
            if not post['logs']:
                print(f"  No saved held-out NLE log probabilities in {post['path'].name}; skipping this density comparison.")
            for key, lp in sorted(post['logs'].items()):
                require(lp.shape == gaussian_logp.shape and np.isfinite(lp).all(), 'Held-out log-probability shape/order differs')
                difference = lp - gaussian_logp
                point_difference = np.asarray([difference[data['point_index'][heldout_rows] == i].mean() for i in heldout])
                row = dict(probe=p, posterior_file=str(post['path']), run_index=run_index,
                           network=int(key.rsplit('_', 1)[1]),
                           gaussian_row_mean_logp=float(gaussian_logp.mean()), NLE_row_mean_logp=float(lp.mean()),
                           NLE_minus_gaussian_row_mean=float(difference.mean()),
                           NLE_minus_gaussian_point_mean=float(point_difference.mean()),
                           point_SE=float(point_difference.std(ddof=1)/np.sqrt(len(heldout))))
                log_rows.append(row)
                print(f"  run {run_index}, network {row['network']}: NLE-Gaussian logp per point = "
                      f"{row['NLE_minus_gaussian_point_mean']:+.3f} (point scatter SE {row['point_SE']:.3f})")
    out = args.outdir or args.base / ('recovery_checks_seed_average' if args.seed_offsets is not None else 'recovery_checks_baseline')
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261013)
    gaussian_draws = {p: rng.multivariate_normal(models[p][0], models[p][1], args.gaussian_draws) for p in PROBES}
    recovery_rows = []
    for p in PROBES:
        model_mean, model_cov, _ = models[p]
        # Physical Gaussian moments use the FULL inverse-probit map, not a local Jacobian.
        gt = LOW + (HIGH-LOW) * ndtr(gaussian_draws[p])
        st = LOW + (HIGH-LOW) * ndtr(draws[p])
        gm, gc = covariance(gt); sm, sc = covariance(st, weights[p])
        for j, name in enumerate(PARAMS):
            recovery_rows.append(dict(probe=p, parameter=name, injected_theta=float(data['theta_reference'][j]),
                Gaussian_mean_u=float(model_mean[j]), SBI_mean_u=float(moments[p][0][j]),
                Gaussian_sd_u=float(np.sqrt(model_cov[j, j])), SBI_sd_u=float(np.sqrt(moments[p][1][j, j])),
                Gaussian_mean_theta=float(gm[j]), SBI_mean_theta=float(sm[j]),
                Gaussian_sd_theta=float(np.sqrt(gc[j, j])), SBI_sd_theta=float(np.sqrt(sc[j, j]))))
        print(f'\n{p} observation-conditioned Gaussian mean u: {np.round(model_mean, 4)}')
        print(f'{p} SBI mean u:                            {np.round(moments[p][0], 4)}')
    csv_write(out / 'mode_full_block_SBI.csv', mode_rows)
    csv_write(out / 'heldout_affine_mean.csv', point_rows)
    csv_write(out / 'heldout_density_comparison.csv', log_rows)
    csv_write(out / 'conditioned_gaussian_vs_SBI.csv', recovery_rows)
    saved = {'parameter_names': np.asarray(PARAMS), 'intercept': a, 'jacobian_u': J,
             'score_covariance': C, 'score_covariance_block': Cblock, 'observed_score': observed,
             'u_reference': data['u_reference'], 'theta_reference': data['theta_reference'],
             'master_sha256': np.asarray(digest)}
    for p in cases:
        saved[p + '_Gaussian_mean_u'], saved[p + '_Gaussian_covariance_u'] = models[p][:2]
    for p in PROBES:
        saved[p + '_SBI_mean_u'], saved[p + '_SBI_covariance_u'] = moments[p]
    np.savez_compressed(out / 'recovery_check_matrices.npz', **saved)
    notes = dict(master=str(args.master or args.base / 'gx_g2x_native_all256.npz'),
        seed_offsets=args.seed_offsets, posterior_files={p: [str(x['path']) for x in posts[p]] for p in PROBES},
        noise_covariance_dof=dof, noise_seed_audit=noise_audit,
        heldout=summaries, Gaussian_draws=args.gaussian_draws,
        limitations=['No NLE retraining or new simulations are performed.',
            'Gaussian posterior conditions on the actual observed scores and includes the prior once.',
            'Changing the affine Gaussian centre leaves its u covariance unchanged.',
            'Held-out mean Q includes OLS fitted-mean uncertainty and C/R_i mean noise.',
            'The OLS fitted-mean uncertainty and point SE assume independent point-mean noise; seed reuse is audited and flagged.',
            'Held-out log probabilities compare fixed fitted densities, not posterior coverage.',
            'Parameter points were proposal-selected; held-out checks are not prior-drawn SBC.',
            'Physical Gaussian moments and contours use inverse-probit samples, not local ellipses.',
            'The posterior NPZ files do not contain trained NLE weights for new-observation recovery.'])
    (out / 'diagnostic_notes.json').write_text(json.dumps(notes, indent=2) + '\n')
    plot_outputs(out, data, draws, weights, gaussian_draws, models, point_rows, mode_rows)
    print(f'\nSaved diagnostics to {out}')
    return 0


def plot_outputs(out, data, draws, weights, gaussian_draws, models, point_rows, mode_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter
    pairs = [(0, 1), (0, 4), (1, 4)]
    labels = [r'$\theta_{\rm ej,0}$', r'$\alpha_{\rm nt}$', r'$\mu_\beta$', r'$\theta_{\rm co,0}$', r'$\nu^M_{\theta_{\rm ej}}$']
    colors = ['#2a78d6', '#eb6834', '#1baf7a']
    for physical in [False, True]:
        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        for row, (p, color) in enumerate(zip(PROBES, colors)):
            transform = (lambda x: LOW + (HIGH-LOW)*ndtr(x)) if physical else (lambda x: x)
            chains = [transform(draws[p]), transform(gaussian_draws[p])]
            truth = data['theta_reference'] if physical else data['u_reference']
            mean, cov, _ = models[p]
            for ax, (i, j) in zip(axes[row], pairs):
                lo = np.minimum(chains[0].min(axis=0), chains[1].min(axis=0))
                hi = np.maximum(chains[0].max(axis=0), chains[1].max(axis=0))
                lo = np.minimum(lo, truth); hi = np.maximum(hi, truth)
                pad = .03*(hi-lo); lo -= pad; hi += pad
                for k, (chain, col) in enumerate(zip(chains, [color, '#333333'])):
                    if k == 1 and not physical:
                        values, vectors = np.linalg.eigh(cov[np.ix_([i, j], [i, j])])
                        angles = np.linspace(0, 2*np.pi, 300)
                        for mass, style in [(.68, '-'), (.95, '--')]:
                            ellipse = (vectors*np.sqrt(values*chi2.ppf(mass, 2))) @ np.array([np.cos(angles), np.sin(angles)])
                            ax.plot(mean[i]+ellipse[0], mean[j]+ellipse[1], color=col, ls=style, lw=1.6)
                    else:
                        h, xe, ye = np.histogram2d(chain[:, i], chain[:, j], bins=55,
                            weights=weights[p] if k == 0 else None,
                            range=[[lo[i], hi[i]], [lo[j], hi[j]]])
                        h = gaussian_filter(h.T, 1.1)
                        ranked = np.sort(h.ravel())[::-1]; cumulative = ranked.cumsum()/ranked.sum()
                        for mass, style in [(.68, '-'), (.95, '--')]:
                            level = ranked[min(np.searchsorted(cumulative, mass), len(ranked)-1)]
                            ax.contour((xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2, h,
                                       levels=[level], colors=[col], linestyles=[style], linewidths=1.6)
                ax.axvline(truth[i], color='gray', ls=':', lw=.8)
                ax.axhline(truth[j], color='gray', ls=':', lw=.8)
                ax.set_xlabel(labels[i] if physical else '$u$: ' + labels[i])
                ax.set_ylabel(labels[j] if physical else '$u$: ' + labels[j])
                ax.set_title(p); ax.grid(alpha=.2)
        fig.legend(handles=[Line2D([], [], color='#333333', label='Observation-conditioned affine Gaussian'),
                            Line2D([], [], color='#1baf7a', label='SBI (row colour)'),
                            Line2D([], [], color='gray', ls=':', label='Injected reference')],
                   loc='upper center', ncol=3, frameon=False)
        fig.suptitle('Solid 68%, dashed 95%; physical Gaussian uses the full inverse-probit map' if physical
                     else 'Observation-conditioned Gaussian vs SBI in u; solid 68%, dashed 95%', y=.965)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(out / ('conditioned_gaussian_vs_SBI_theta.png' if physical else 'conditioned_gaussian_vs_SBI_u.png'), dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for p, col in zip(PROBES, colors):
        rows = [r for r in point_rows if r['probe'] == p]
        ax.plot([r['point_index'] for r in rows], [r['Q_per_dimension'] for r in rows], 'o-', color=col, label=p)
    ax.axhline(1, color='gray', ls='--', label='Approximate affine-Gaussian expectation')
    ax.set_yscale('log'); ax.set_xlabel('Held-out parameter-point index'); ax.set_ylabel('Mean-response Q / summary dimension')
    ax.legend(frameon=False); ax.set_title('Held-out affine response: noise averaging and fitted-mean uncertainty included')
    fig.tight_layout(); fig.savefig(out / 'heldout_affine_mean.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    keys = ['sigma_single_F', 'sigma_single_SBI', 'sigma_joint_block', 'sigma_joint_full', 'sigma_joint_SBI']
    labels = ['Single F', 'Single SBI', 'Joint block', 'Joint full', 'Joint SBI']
    for ax, row in zip(axes.flat, mode_rows):
        ax.bar(labels, [row[k] for k in keys], color=['#777777', '#2a78d6', '#aaaaaa', '#444444', '#1baf7a'])
        ax.set_yscale('log'); ax.tick_params(axis='x', rotation=25)
        ax.set_title(f"{row['selected_by']}-selected, relative to {row['relative_to']}")
        ax.set_ylabel('SD of the same combination in u')
    fig.tight_layout(); fig.savefig(out / 'mode_full_block_SBI.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    raise SystemExit(main())

