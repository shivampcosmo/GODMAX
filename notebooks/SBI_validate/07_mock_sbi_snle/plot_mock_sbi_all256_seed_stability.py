#!/usr/bin/env python3
"""Compare baseline random-seed offsets, keeping all simulations/settings fixed.

Defaults: gy_g2y, offsets 0..9, combined/native/contract-native, 128x5+128x10.
Each offset is one four-network run; two MCMC repeats are NOT extra networks.
Offset 0 uses the original posterior if no explicit baseline_s0 file exists.
Only baseline filenames are selected. Missing offsets or changed settings fail.

Writes three PNG/PDF figures, four CSV tables, and small moment arrays (NPZ).
The corner plot uses the overview notebook's smoothed 68%/95% histograms.
Every curve retains its fitted location. The black curve averages posterior
densities with equal weight per network, even if saved sample counts differ.
No draws are averaged coordinate by coordinate; no independent likelihoods
are multiplied. Diagnostic moments are in probit u, with population variance
so the within/between-network covariance decomposition is exact.

Dependencies: NumPy, SciPy, Matplotlib. No new maps or training required.
"""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import ndtr

PARAMS = ['theta_ej_0', 'alpha_nt', 'mu_beta', 'theta_co_0', 'nu_theta_ej_M']
LABELS = [r'$\theta_{\rm ej,0}$', r'$\alpha_{\rm nt}$', r'$\mu_\beta$',
          r'$\theta_{\rm co,0}$', r'$\nu^M_{\theta_{\rm ej}}$']
LOW = np.array([0.5, 0., 0.005, 0.001, -1.])
HIGH = np.array([8., 0.5, 1.5, 0.5, 1.])
BASE = Path('data/SBI_validate/mock_sbi/all256_native')
TITLES = {'gy': r'$g\times y$', 'g2y': r'$g^2\times y$',
          'gy_g2y': r'$gy+g^2y$ (paired 10D)'}
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#d64949', '#8a5cc2',
           '#95643c', '#d175b5', '#737373', '#a09b22', '#3da6a6']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def moments(values, weights=None):
    """Population moments, optionally for an exactly weighted empirical mixture."""
    mean = np.average(values, axis=0, weights=weights)
    delta = values - mean
    w = np.full(len(values), 1/len(values)) if weights is None else weights/weights.sum()
    covariance = (delta*w[:, None]).T @ delta
    sd = np.sqrt(np.diag(covariance))
    require(np.all(sd > 0), 'Zero posterior variance')
    return dict(mean=mean, covariance=covariance, sd=sd,
                correlation=covariance/np.outer(sd, sd))


def samples(archive, theta_key, u_key, path):
    theta, u = np.asarray(archive[theta_key], float), np.asarray(archive[u_key], float)
    require(theta.ndim == 2 and theta.shape[1] == 5 and len(theta) > 1
            and u.shape == theta.shape and np.isfinite(theta).all()
            and np.isfinite(u).all(), f'{path.name}: invalid {theta_key}/{u_key}')
    require(np.all(theta >= LOW-1e-12) and np.all(theta <= HIGH+1e-12),
            f'{path.name}: samples outside physical priors')
    np.testing.assert_allclose(theta, LOW+(HIGH-LOW)*ndtr(u), rtol=0, atol=2e-12,
                               err_msg=f'{path.name}: theta/u transform differs')
    return theta, u


def load_run(path, offset, probe):
    with np.load(path, allow_pickle=False) as h:
        man = json.loads(h['manifest_json'].item())
        require(man['probe'] == probe and man['parameter_names'] == PARAMS,
                f'{path.name}: probe/parameter order differs')
        require(tuple(man[k] for k in ('cohort', 'training_replica_policy',
                'contract_replica_policy')) == ('combined', 'native', 'native'),
                f'{path.name}: expected combined/native/contract-native')
        require(man['compression'] == 'score'
                and man['summary_dimension'] == (10 if probe == 'gy_g2y' else 5),
                f'{path.name}: summary construction differs')
        if probe == 'gy_g2y':
            require(man['component_order'] == ['gy', 'g2y'],
                    f'{path.name}: component order differs')
        for key, expected in [('n_master_points', 256), ('n_master_native_rows', 1920)]:
            if key in man:
                require(man[key] == expected, f'{path.name}: {key} differs')
        require(man['network_seeds'] == 4, f'{path.name}: expected four networks')
        theta, u = samples(h, 'theta', 'u', path)
        require(len(theta) == man['pooled_samples'], f'{path.name}: pooled count differs')
        records = man['per_network']
        require(len(records) == 4, f'{path.name}: per-network records incomplete')
        networks = []
        for k in range(4):
            nt, nu = samples(h, f'theta_seed_{k}', f'u_seed_{k}', path)
            require(len(nt) == man['posterior_samples_per_network'],
                    f'{path.name}: network {k} sample count differs')
            record = records[k]
            require(record['network_seed'] == k, f'{path.name}: network records out of order')
            networks.append(dict(offset=offset, network=k, theta=nt, u=nu,
                training_seed=int(record['training_seed']), record=record,
                stats=moments(nu)))
        np.testing.assert_array_equal(u, np.concatenate([n['u'] for n in networks]))
        np.testing.assert_array_equal(theta, np.concatenate([n['theta'] for n in networks]))
        observed = np.asarray(man['observed_summary'], float)
        if 'observed_summary' in h.files:
            np.testing.assert_allclose(h['observed_summary'], observed, rtol=0, atol=1e-12)
        contracts = {}
        for component in (['gy', 'g2y'] if probe == 'gy_g2y' else [probe]):
            suffix = '_'+component if probe == 'gy_g2y' else ''
            for key in ('covariance', 'score_operator', 'score_reference_prediction'):
                contracts[component+':'+key] = np.asarray(h[key+suffix])
        return dict(path=path, offset=offset, label=f'seed offset {offset}', manifest=man,
            theta=theta, u=u, stats=moments(u), networks=networks, contracts=contracts,
            truth=np.asarray(h['theta_reference'], float),
            truth_u=np.asarray(h['u_reference'], float), observed=observed,
            split=np.asarray(h['point_split']))


def discover(args):
    directory = args.ablation_dir or args.base/'ablations'
    marker = '_sepscore10' if args.probe == 'gy_g2y' else ''
    label = 'combined_native_contractnative'
    paths, missing = [], []
    for offset in args.offsets:
        path = directory/f'mock_snle_all256_{label}_abl_baseline_s{offset}{marker}_{args.probe}_posterior.npz'
        if offset == 0 and not path.is_file():
            path = args.base/f'mock_snle_all256_{label}{marker}_{args.probe}_posterior.npz'
        if not path.is_file():
            missing.append(str(path))
        paths.append((offset, path))
    if missing:
        raise FileNotFoundError('Missing requested seed-offset results:\n  '+'\n  '.join(missing))
    return [load_run(path, offset, args.probe) for offset, path in paths]


def check_comparable(runs):
    ref = runs[0]
    namespace = list(ref['manifest']['namespace'])
    if ref['offset']:
        require(namespace[-1] == ref['offset'], 'Filename offset differs from saved namespace')
        namespace = namespace[:-1]
    training_seeds = []
    for run in runs:
        man = run['manifest']
        expected = namespace+([run['offset']] if run['offset'] else [])
        require(man['namespace'] == expected, f"{run['path'].name}: seed namespace differs")
        # The internal random training/validation assignment can vary by seed.
        # Its fraction/settings and the frozen outer point split stay fixed.
        for key in ('training', 'mcmc', 'master_sha256', 'point_split_sha256'):
            require(man[key] == ref['manifest'][key], f"{run['path'].name}: {key} differs")
        for key in ('training_summary_sha256', 'heldout_summary_sha256'):
            if key in man and key in ref['manifest']:
                require(man[key] == ref['manifest'][key], f"{run['path'].name}: {key} differs")
        for key in ('truth', 'truth_u', 'split'):
            np.testing.assert_array_equal(run[key], ref[key])
        np.testing.assert_allclose(run['observed'], ref['observed'], rtol=0, atol=1e-12)
        for key, array in ref['contracts'].items():
            np.testing.assert_allclose(run['contracts'][key], array, rtol=1e-10, atol=1e-12,
                                       err_msg=f"{run['path'].name}: compression changed ({key})")
        training_seeds.extend(n['training_seed'] for n in run['networks'])
    require(len(set(training_seeds)) == len(training_seeds),
            'Repeated training seeds: refusing to count duplicated networks as independent fits')


def mixture(runs):
    networks = [n for r in runs for n in r['networks']]
    theta = np.concatenate([n['theta'] for n in networks])
    u = np.concatenate([n['u'] for n in networks])
    weights = np.concatenate([np.full(len(n['u']), 1/(len(networks)*len(n['u'])))
                              for n in networks])
    stats = moments(u, weights)
    within = np.mean([n['stats']['covariance'] for n in networks], axis=0)
    means = np.asarray([n['stats']['mean'] for n in networks])
    delta = means-stats['mean']
    between = delta.T @ delta/len(networks)
    np.testing.assert_allclose(stats['covariance'], within+between, rtol=1e-11, atol=1e-12)
    return dict(label=f'equal-weight average ({len(networks)} networks)', theta=theta,
                u=u, weights=weights, stats=stats, within=within, between=between,
                theta_stats=moments(theta, weights), networks=networks)


def save_figure(fig, stem, show):
    import matplotlib.pyplot as plt
    for extension in ('png', 'pdf'):
        fig.savefig(stem.with_suffix('.'+extension), dpi=180, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def corner(runs, average, args, out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    truth = runs[0]['truth']
    lo = np.minimum(average['theta'].min(axis=0), truth)
    hi = np.maximum(average['theta'].max(axis=0), truth)
    pad = .04*np.maximum(hi-lo, np.finfo(float).eps)
    lo, hi = lo-pad, hi+pad
    curves = runs+[average]
    fig, axes = plt.subplots(5, 5, figsize=(12.3, 11.4))
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            for curve in curves:
                chain, color = curve['theta'], curve['color']
                weights = curve.get('weights')
                width = 2.3 if curve is average else 1.15
                if i == j:
                    h, edges = np.histogram(chain[:, j], bins=64, weights=weights,
                                            range=(lo[j], hi[j]))
                    h = gaussian_filter(h.astype(float), 2.4)
                    ax.plot(.5*(edges[:-1]+edges[1:]), h/h.max(), color=color, lw=width)
                else:
                    h, xe, ye = np.histogram2d(chain[:, j], chain[:, i], bins=44,
                        weights=weights, range=[[lo[j], hi[j]], [lo[i], hi[i]]])
                    h = gaussian_filter(h.T, 1.35)
                    ranked = np.sort(h.ravel())[::-1]
                    cumulative = np.cumsum(ranked)/ranked.sum()
                    xc, yc = .5*(xe[:-1]+xe[1:]), .5*(ye[:-1]+ye[1:])
                    for mass, style in [(.95, '--'), (.68, '-')]:
                        index = min(int(np.searchsorted(cumulative, mass)), len(ranked)-1)
                        ax.contour(xc, yc, h, levels=[ranked[index]], colors=[color],
                                   linestyles=[style], linewidths=width)
            ax.axvline(truth[j], color='#777777', lw=.9, ls=':', zorder=1)
            if i == j:
                ax.set_ylim(0, 1.13)
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
            else:
                ax.axhline(truth[i], color='#777777', lw=.9, ls=':', zorder=1)
                ax.set_ylim(lo[i], hi[i])
                ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune='both'))
            ax.set_xlim(lo[j], hi[j])
            ax.xaxis.set_major_locator(plt.MaxNLocator(4, prune='both'))
            ax.tick_params(labelsize=8)
            if i == 4:
                ax.set_xlabel(LABELS[j], fontsize=12)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS[i], fontsize=12)
            elif j != 0:
                ax.tick_params(labelleft=False)
    handles = [Line2D([], [], color=c['color'], lw=2.4, label=c['label']) for c in curves]
    handles.append(Line2D([], [], color='#777777', ls=':', label='injected reference'))
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.985, .92),
               frameon=False, fontsize=9)
    fig.suptitle(f'{TITLES[args.probe]}: random-seed comparison ({len(runs)} runs)',
                 fontsize=16, y=.98)
    fig.text(.70, .555, 'Solid: 68%   Dashed: 95%\nFour networks per seed offset.\n'
             'Same data and training settings; locations retained.\n'
             'Black: equal-weight posterior-density average.', fontsize=9,
             ha='left', va='top', color='#555555')
    fig.subplots_adjust(left=.09, right=.985, top=.94, bottom=.08, wspace=.09, hspace=.09)
    save_figure(fig, out/'seed_offsets_triangle', args.show)


def variability_plot(runs, average, args, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    ref = average['stats']
    shifts = np.asarray([(r['stats']['mean']-ref['mean'])/ref['sd'] for r in runs]).T
    ratios = np.asarray([r['stats']['sd']/ref['sd'] for r in runs]).T
    pairs = list(combinations(range(5), 2))
    correlations = np.asarray([[r['stats']['correlation'][i, j]-ref['correlation'][i, j]
                                for i, j in pairs] for r in runs]).T
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.9), layout='constrained')
    panels = [(shifts, LABELS, 'Posterior mean shift in u / average SD', 0.),
              (ratios, LABELS, 'Posterior SD in u / average SD', 1.),
              (correlations, [LABELS[i]+' / '+LABELS[j] for i, j in pairs],
               'Correlation change in u', 0.)]
    for ax, (values, labels, title, centre) in zip(axes, panels):
        extent = max(float(np.max(np.abs(values-centre))), .02)
        im = ax.imshow(values, aspect='auto', cmap='RdBu_r',
                       norm=TwoSlopeNorm(centre, centre-extent, centre+extent))
        ax.set_xticks(np.arange(len(runs)), [str(r['offset']) for r in runs])
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
        ax.set_xlabel('random seed offset')
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=.78)
    fig.suptitle(f'{TITLES[args.probe]}: variability across four-network runs\n'
                 'Reference: equally weighted average of all networks', fontsize=13)
    save_figure(fig, out/'seed_offsets_variability', args.show)


def network_plot(runs, average, args, out):
    import matplotlib.pyplot as plt
    networks, ref = average['networks'], average['stats']
    shifts = np.asarray([(n['stats']['mean']-ref['mean'])/ref['sd'] for n in networks])
    ratios = np.asarray([n['stats']['sd']/ref['sd'] for n in networks])
    repeat = [n['record'].get('max_repeat_mean_shift_sigma', np.nan) for n in networks]
    heldout = [n['record'].get('heldout_mean_log_prob', np.nan) for n in networks]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6), layout='constrained')
    x = np.arange(len(networks))
    for j in range(5):
        axes[0, 0].plot(x, shifts[:, j], '.-', lw=.65, ms=5, label=LABELS[j])
        axes[0, 1].plot(x, ratios[:, j], '.-', lw=.65, ms=5, label=LABELS[j])
    axes[0, 0].axhline(0, color='#555555', lw=.8)
    axes[0, 1].axhline(1, color='#555555', lw=.8)
    axes[0, 0].set_ylabel('Network mean shift in u / average SD')
    axes[0, 1].set_ylabel('Network SD in u / average SD')
    axes[0, 0].legend(ncol=3, fontsize=9, frameon=False)
    for k, run in enumerate(runs):
        segment = slice(4*k, 4*k+4)
        axes[1, 0].scatter(x[segment], np.asarray(heldout)[segment], color=run['color'], s=28)
        axes[1, 1].scatter(x[segment], np.asarray(repeat)[segment], color=run['color'], s=28)
    axes[1, 0].set_ylabel('Held-out mean log likelihood')
    axes[1, 1].set_ylabel('Max MCMC-repeat mean shift / network SD')
    for ax in axes.flat:
        ax.set_xticks(4*np.arange(len(runs))+1.5, [str(r['offset']) for r in runs])
        ax.set_xlabel('random seed offset (four networks per group)')
        ax.set_xlim(-.8, len(networks)-.2)
        for edge in 4*np.arange(1, len(runs))-.5:
            ax.axvline(edge, color='#bbbbbb', lw=.5)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'{TITLES[args.probe]}: {len(networks)} individual network fits\n'
                 'Training variation and MCMC-repeat variation are shown separately', fontsize=13)
    save_figure(fig, out/'seed_network_diagnostics', args.show)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tables(runs, average, out):
    ref = average['stats']
    parameters, diagnostics, network_rows, summary = [], [], [], []
    for run in runs:
        stats, man = run['stats'], run['manifest']
        shift, ratio = (stats['mean']-ref['mean'])/ref['sd'], stats['sd']/ref['sd']
        q16, q50, q84 = np.percentile(run['theta'], [16, 50, 84], axis=0)
        for j, parameter in enumerate(PARAMS):
            parameters.append(dict(seed_offset=run['offset'], parameter=parameter,
                mean_theta=run['theta'][:, j].mean(), median_theta=q50[j],
                sd_theta=run['theta'][:, j].std(), q16_theta=q16[j], q84_theta=q84[j],
                mean_u=stats['mean'][j], sd_u=stats['sd'][j],
                mean_shift_over_average_sd_u=shift[j], sd_ratio_to_average_u=ratio[j]))
        diagnostics.append(dict(seed_offset=run['offset'], networks=4, samples=len(run['u']),
            namespace=json.dumps(man['namespace']), training=json.dumps(man['training'], sort_keys=True),
            max_mean_shift_over_average_sd_u=float(np.abs(shift).max()),
            max_abs_sd_ratio_minus_one_u=float(np.abs(ratio-1).max()),
            max_abs_correlation_change_u=float(np.abs(stats['correlation']-ref['correlation']).max()),
            max_network_mean_departure_sigma=man.get('max_network_mean_departure_sigma', np.nan),
            max_mcmc_repeat_mean_shift_sigma=man.get('max_independent_mcmc_repeat_mean_shift_sigma', np.nan),
            posterior_file=str(run['path'])))
    for n in average['networks']:
        shift = (n['stats']['mean']-ref['mean'])/ref['sd']
        ratio = n['stats']['sd']/ref['sd']
        row = dict(seed_offset=n['offset'], network=n['network'], training_seed=n['training_seed'],
            samples=len(n['u']), heldout_mean_log_prob=n['record'].get('heldout_mean_log_prob', np.nan),
            max_mcmc_repeat_mean_shift_sigma=n['record'].get('max_repeat_mean_shift_sigma', np.nan))
        for j, parameter in enumerate(PARAMS):
            row[parameter+'_mean_shift_over_average_sd_u'] = shift[j]
            row[parameter+'_sd_ratio_to_average_u'] = ratio[j]
        network_rows.append(row)
    means = np.asarray([r['stats']['mean'] for r in runs])
    widths = np.asarray([r['stats']['sd'] for r in runs])
    for j, parameter in enumerate(PARAMS):
        summary.append(dict(parameter=parameter, average_mean_u=ref['mean'][j],
            average_sd_u=ref['sd'][j], average_mean_theta=average['theta_stats']['mean'][j],
            average_sd_theta=average['theta_stats']['sd'][j],
            injected_theta=runs[0]['truth'][j],
            sd_of_run_means_over_average_sd_u=means[:, j].std()/ref['sd'][j],
            sd_of_network_means_over_average_sd_u=np.sqrt(average['between'][j, j])/ref['sd'][j],
            between_network_fraction_of_variance_u=average['between'][j, j]/ref['covariance'][j, j],
            min_run_sd_ratio_u=widths[:, j].min()/ref['sd'][j],
            max_run_sd_ratio_u=widths[:, j].max()/ref['sd'][j]))
    write_csv(out/'seed_parameter_comparison.csv', parameters)
    write_csv(out/'seed_run_diagnostics.csv', diagnostics)
    write_csv(out/'seed_network_diagnostics.csv', network_rows)
    write_csv(out/'seed_variability_summary.csv', summary)
    np.savez_compressed(out/'seed_stability_statistics.npz', parameter_names=np.asarray(PARAMS),
        seed_offsets=np.asarray([r['offset'] for r in runs]),
        run_mean_u=means, run_sd_u=widths,
        run_correlation_u=np.asarray([r['stats']['correlation'] for r in runs]),
        network_training_seeds=np.asarray([n['training_seed'] for n in average['networks']]),
        network_seed_offsets=np.asarray([n['offset'] for n in average['networks']]),
        network_mean_u=np.asarray([n['stats']['mean'] for n in average['networks']]),
        mixture_mean_u=ref['mean'], mixture_covariance_u=ref['covariance'],
        mixture_mean_theta=average['theta_stats']['mean'],
        mixture_covariance_theta=average['theta_stats']['covariance'],
        within_network_covariance_u=average['within'], between_network_covariance_u=average['between'],
        theta_reference=runs[0]['truth'], u_reference=runs[0]['truth_u'])
    print('\nVariability relative to the equally weighted posterior average:')
    print(f"{'parameter':20s} {'run mean scatter/SD':>20s} {'network variance fraction':>26s} {'run SD range/SD':>21s}")
    for row in summary:
        print(f"{row['parameter']:20s} {row['sd_of_run_means_over_average_sd_u']:20.3f} "
              f"{row['between_network_fraction_of_variance_u']:26.3f} "
              f"{row['min_run_sd_ratio_u']:9.3f} .. {row['max_run_sd_ratio_u']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--ablation-dir', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--probe', choices=list(TITLES), default='gy_g2y')
    parser.add_argument('--offsets', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    require(len(args.offsets) >= 2 and len(set(args.offsets)) == len(args.offsets)
            and min(args.offsets) >= 0, 'Supply at least two distinct non-negative offsets')
    args.offsets.sort()
    if not args.show:
        import matplotlib
        matplotlib.use('Agg')
    runs = discover(args)
    check_comparable(runs)
    average = mixture(runs)
    for k, run in enumerate(runs):
        run['color'] = PALETTE[k % len(PALETTE)]
        print(f"offset {run['offset']:2d}: {run['path'].name} ({len(run['u']):,} samples)")
    average['color'] = '#101010'
    print(f'PASS: {len(runs)} offsets, {len(average["networks"])} distinct network training seeds; '
          'same master, split, observations, score contracts, training and MCMC settings.')
    out = args.outdir or (args.ablation_dir or args.base/'ablations')/'figures'/'seed_stability'/args.probe
    out.mkdir(parents=True, exist_ok=True)
    tables(runs, average, out)
    corner(runs, average, args, out)
    variability_plot(runs, average, args, out)
    network_plot(runs, average, args, out)
    print(f'\nSaved three PNG/PDF figures, four CSV tables and moment arrays to {out}')
    print('Small seed variation supports reproducibility; it does not establish posterior coverage. '
          'No automatic stability threshold is applied. MCMC repeats are not additional trained networks.')


if __name__ == '__main__':
    main()
