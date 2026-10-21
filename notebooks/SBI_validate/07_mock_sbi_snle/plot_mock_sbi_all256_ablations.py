#!/usr/bin/env python3
"""Overlay the original and ablation SBI posteriors for gy, g2y and gy_g2y.

Run from the GODMAX root, or pass --base explicitly. Uses NumPy, SciPy and
Matplotlib already used by the overview notebook; no corner/GetDist needed.
Three five-parameter corner figures are saved as PNG/PDF. --show displays them
when the script is invoked from Jupyter with %run.

By default, compare only combined/native/contract-native runs on the existing
128x5 + 128x10 master. The other data selections can be chosen with the policy
arguments. Files from other selections are skipped rather than pooled.
Each curve is a separate run's saved four-network pool; pools from different
experiments are never combined. Diagnostics also report the saved within-run
network and MCMC-repeat checks.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


PARAMS = ['theta_ej_0', 'alpha_nt', 'mu_beta', 'theta_co_0', 'nu_theta_ej_M']
LABELS = [r'$\theta_{\rm ej,0}$', r'$\alpha_{\rm nt}$', r'$\mu_\beta$',
          r'$\theta_{\rm co,0}$', r'$\nu^M_{\theta_{\rm ej}}$']
LOW = np.array([0.5, 0., 0.005, 0.001, -1.])
HIGH = np.array([8., 0.5, 1.5, 0.5, 1.])
PROBES = ['gy', 'g2y', 'gy_g2y']
TITLES = {'gy': r'$g\times y$', 'g2y': r'$g^2\times y$',
          'gy_g2y': r'$gy+g^2y$ (paired 10D)'}
CASES = ['baseline', 'batch128', 'lr2e4', 'patience100', 'epochs2000']
BASE = Path('data/SBI_validate/mock_sbi/all256_native')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_run(path, selection, original=False):
    with np.load(path, allow_pickle=False) as archive:
        man = json.loads(archive['manifest_json'].item())
        probe = man['probe']
        policies = tuple(man[k] for k in
                         ('cohort', 'training_replica_policy', 'contract_replica_policy'))
        if probe not in PROBES or policies != selection:
            return None
        require(man['parameter_names'] == PARAMS, f'{path.name}: parameter order differs')
        require(man['compression'] == 'score' and
                man['summary_dimension'] == (10 if probe == 'gy_g2y' else 5),
                f'{path.name}: expected the current separate-score construction')
        if probe == 'gy_g2y':
            require(man['component_order'] == ['gy', 'g2y'],
                    f'{path.name}: joint component order differs')
        theta = np.asarray(archive['theta'], dtype=float)
        u = np.asarray(archive['u'], dtype=float)
        require(theta.ndim == 2 and theta.shape[1] == 5 and theta.shape[0] > 1
                and u.shape == theta.shape and np.isfinite(theta).all()
                and np.isfinite(u).all(), f'{path.name}: invalid posterior samples')
        require(theta.shape[0] == man['pooled_samples'],
                f'{path.name}: sample count differs from its own manifest')
        require(np.all(theta >= LOW-1e-12) and np.all(theta <= HIGH+1e-12),
                f'{path.name}: samples outside the physical prior')
        contracts = {}
        for component in (['gy', 'g2y'] if probe == 'gy_g2y' else [probe]):
            suffix = f'_{component}' if probe == 'gy_g2y' else ''
            for key in ('covariance', 'score_operator', 'score_reference_prediction'):
                contracts[f'{component}:{key}'] = np.asarray(archive[key+suffix])
        truth = np.asarray(archive['theta_reference'], dtype=float)

    match = re.search(r'_abl_(.+?)_s(\d+)(?:_sepscore10)?_'+probe+r'_posterior\.npz$',
                      path.name)
    require(original or match is not None, f'{path.name}: unrecognized ablation filename')
    case, offset = ('baseline', 0) if original else (match[1], int(match[2]))
    label = ('original baseline' if original else case)+f' (seed {offset})'
    rank = -1 if original else (CASES.index(case) if case in CASES else len(CASES))
    return dict(path=path, manifest=man, probe=probe, theta=theta, u=u,
                contracts=contracts, truth=truth, case=case, offset=offset,
                label=label, original=original, order=(rank, offset, path.name))


def check_comparable(runs):
    baseline = next((r for r in runs if r['original']),
                    next((r for r in runs if r['case'] == 'baseline' and r['offset'] == 0),
                         runs[0]))
    base = baseline['manifest']
    for run in runs:
        man = run['manifest']
        for key in ('master_sha256', 'point_split_sha256'):
            require(man[key] == base[key], f"{run['path'].name}: {key} differs")
        # These hashes are available in the current runner. If an older product
        # omitted one, still compare the actual contracts and observation below.
        for key in ('training_summary_sha256', 'heldout_summary_sha256'):
            if key in man and key in base:
                require(man[key] == base[key], f"{run['path'].name}: {key} differs")
        np.testing.assert_array_equal(run['truth'], baseline['truth'])
        np.testing.assert_allclose(man['observed_summary'], base['observed_summary'],
                                   rtol=0, atol=1e-12)
        for key in baseline['contracts']:
            np.testing.assert_allclose(run['contracts'][key], baseline['contracts'][key],
                rtol=1e-10, atol=1e-12,
                err_msg=f"{run['path'].name}: compression changed ({key})")
    return baseline


def corner(runs, colors, out, show=False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    truth = runs[0]['truth']
    # Include all samples in the contour histograms, without percentile cuts.
    lo = np.minimum.reduce([r['theta'].min(axis=0) for r in runs]+[truth])
    hi = np.maximum.reduce([r['theta'].max(axis=0) for r in runs]+[truth])
    pad = 0.04*np.maximum(hi-lo, np.finfo(float).eps)
    lo, hi = lo-pad, hi+pad
    fig, axes = plt.subplots(5, 5, figsize=(11.2, 10.6))
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            for run in runs:
                chain, color = run['theta'], colors[run['label']]
                zorder = 12 if run['original'] else 5
                if i == j:
                    h, edge = np.histogram(chain[:, j], bins=64, range=(lo[j], hi[j]))
                    h = gaussian_filter(h.astype(float), 2.4)
                    ax.plot(0.5*(edge[:-1]+edge[1:]), h/h.max(),
                            color=color, lw=1.8, zorder=zorder)
                else:
                    h, xe, ye = np.histogram2d(chain[:, j], chain[:, i], bins=44,
                        range=[[lo[j], hi[j]], [lo[i], hi[i]]])
                    h = gaussian_filter(h.T, 1.35)
                    ranked = np.sort(h.ravel())[::-1]
                    cumulative = np.cumsum(ranked)/ranked.sum()
                    xc, yc = 0.5*(xe[:-1]+xe[1:]), 0.5*(ye[:-1]+ye[1:])
                    for mass, style in [(0.95, '--'), (0.68, '-')]:
                        level = ranked[min(int(np.searchsorted(cumulative, mass)), ranked.size-1)]
                        ax.contour(xc, yc, h, levels=[level], colors=[color],
                                   linestyles=[style], linewidths=1.3, zorder=zorder)
            ax.axvline(truth[j], color='#777777', lw=0.9, ls=':', zorder=1)
            if i == j:
                ax.set_ylim(0, 1.13); ax.set_yticks([])
                ax.spines['left'].set_visible(False)
            else:
                ax.axhline(truth[i], color='#777777', lw=0.9, ls=':', zorder=1)
                ax.set_ylim(lo[i], hi[i])
            ax.set_xlim(lo[j], hi[j]); ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(plt.MaxNLocator(4, prune='both'))
            if i != j:
                ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune='both'))
            if i == 4: ax.set_xlabel(LABELS[j], fontsize=12)
            else: ax.tick_params(labelbottom=False)
            if j == 0 and i > 0: ax.set_ylabel(LABELS[i], fontsize=12)
            elif j != 0: ax.tick_params(labelleft=False)

    handles = [Line2D([], [], color=colors[r['label']], lw=2.4, label=r['label']) for r in runs]
    handles += [Line2D([], [], color='#777777', ls=':', label='reference')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.92),
               frameon=False, fontsize=10)
    fig.suptitle(f"{TITLES[runs[0]['probe']]}: SBI ablations ({len(runs)} runs)",
                 fontsize=15, y=0.98)
    fig.text(0.70, 0.59, 'Solid: 68%   Dashed: 95%\nEach curve is a separate run; '
             'locations retained.', ha='left', va='top', fontsize=9, color='#555555')
    fig.subplots_adjust(left=0.09, right=0.985, top=0.94, bottom=0.08,
                        wspace=0.09, hspace=0.09)
    for extension in ('png', 'pdf'):
        fig.savefig(out/f"ablation_{runs[0]['probe']}_triangle.{extension}",
                    dpi=180, bbox_inches='tight')
    if show: plt.show()
    plt.close(fig)


def comparisons(runs, baseline):
    base_u = baseline['u']
    mean0, sd0 = base_u.mean(axis=0), base_u.std(axis=0, ddof=1)
    corr0 = np.corrcoef(base_u, rowvar=False)
    require(np.all(sd0 > 0), 'Baseline has a zero parameter variance')
    parameters, diagnostics = [], []
    for run in runs:
        theta, u, man = run['theta'], run['u'], run['manifest']
        shift = (u.mean(axis=0)-mean0)/sd0
        ratio = u.std(axis=0, ddof=1)/sd0
        q16, q50, q84 = np.percentile(theta, [16, 50, 84], axis=0)
        training = man['training']
        for j, parameter in enumerate(PARAMS):
            parameters.append(dict(probe=run['probe'], run=run['label'],
                reference_run=baseline['label'], parameter=parameter,
                mean_theta=theta[:, j].mean(), median_theta=q50[j],
                sd_theta=theta[:, j].std(ddof=1), q16_theta=q16[j], q84_theta=q84[j],
                mean_shift_in_baseline_sd_u=shift[j], sd_u_ratio_to_baseline=ratio[j]))
        logps = [r.get('heldout_mean_log_prob') for r in man.get('per_network', [])]
        logps = [v for v in logps if v is not None and np.isfinite(v)]
        diagnostics.append(dict(probe=run['probe'], run=run['label'],
            reference_run=baseline['label'], samples=len(theta),
            training_batch_size=training['training_batch_size'],
            learning_rate=training['learning_rate'], patience=training['stop_after_epochs'],
            max_epochs=training['max_num_epochs'],
            training_same_as_reference=training == baseline['manifest']['training'],
            max_abs_mean_shift_in_baseline_sd_u=float(np.abs(shift).max()),
            max_abs_sd_u_ratio_minus_one=float(np.abs(ratio-1).max()),
            max_abs_correlation_change_u=float(np.abs(np.corrcoef(u, rowvar=False)-corr0).max()),
            heldout_mean_log_prob=float(np.mean(logps)) if logps else np.nan,
            max_network_mean_departure_sigma=man.get('max_network_mean_departure_sigma', np.nan),
            max_mcmc_repeat_mean_shift_sigma=man.get('max_independent_mcmc_repeat_mean_shift_sigma', np.nan),
            posterior_file=str(run['path'])))
    return parameters, diagnostics


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--run-label', default='combined_native_contractnative')
    parser.add_argument('--ablation-dir', type=Path)
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--cohort', choices=['combined', 'old', 'new'], default='combined')
    parser.add_argument('--training-replicas', choices=['native', 'first5'], default='native')
    parser.add_argument('--contract-replicas', choices=['native', 'first5'], default='native')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    if not args.show:
        import matplotlib
        matplotlib.use('Agg')
    selection = (args.cohort, args.training_replicas, args.contract_replicas)
    directory = args.ablation_dir or args.base/'ablations'
    out = args.outdir or directory/'figures'/('_'.join(selection))
    by_probe = {p: [] for p in PROBES}
    for probe in PROBES:
        marker = '_sepscore10' if probe == 'gy_g2y' else ''
        path = args.base/f'mock_snle_all256_{args.run_label}{marker}_{probe}_posterior.npz'
        if path.is_file():
            run = load_run(path, selection, original=True)
            if run is not None: by_probe[probe].append(run)
    for path in sorted(directory.glob('mock_snle_all256_*_abl_*_posterior.npz')):
        run = load_run(path, selection)
        if run is not None: by_probe[run['probe']].append(run)
    require(any(by_probe.values()), f'No matching posterior files under {args.base} and {directory}')
    all_runs = sorted([r for runs in by_probe.values() for r in runs], key=lambda r: r['order'])
    labels = list(dict.fromkeys(r['label'] for r in all_runs))
    palette = ['#303030', '#2a78d6', '#eb6834', '#8a5cc2', '#1baf7a',
               '#d64949', '#b08b32', '#d175b5', '#3da6a6', '#7c6a58']
    colors = {label: palette[i % len(palette)] for i, label in enumerate(labels)}
    out.mkdir(parents=True, exist_ok=True)
    parameters, diagnostics = [], []
    for probe, runs in by_probe.items():
        if not runs:
            print(f'{probe}: no matching results; figure skipped.')
            continue
        runs.sort(key=lambda r: r['order'])
        baseline = check_comparable(runs)
        print(f'\n{probe}: {len(runs)} runs; reference = {baseline["label"]}')
        if len(runs) == 1:
            print('  Only one run exists for this probe; there are no ablation overlays yet.')
        print(f'  {"run":32s} {"batch":>5s} {"lr":>8s} {"patience":>8s} {"maxep":>6s}'
              f' {"max shift/u SD":>14s} {"max width change":>16s}')
        p, d = comparisons(runs, baseline)
        parameters.extend(p); diagnostics.extend(d)
        for row in d:
            print(f'  {row["run"]:32s} {row["training_batch_size"]:5d} '
                  f'{row["learning_rate"]:8.1e} {row["patience"]:8d} {row["max_epochs"]:6d} '
                  f'{row["max_abs_mean_shift_in_baseline_sd_u"]:14.3f} '
                  f'{100*row["max_abs_sd_u_ratio_minus_one"]:15.1f}%')
        corner(runs, colors, out, args.show)
    write_csv(out/'ablation_parameter_comparison.csv', parameters)
    write_csv(out/'ablation_run_diagnostics.csv', diagnostics)
    print(f'\nSaved corner PNG/PDF figures and two comparison CSVs to {out}')
    print('Compare held-out log probabilities within a probe only. The diagnostics CSV '
          'also records network and MCMC-repeat stability; pooled contours alone do not establish calibration.')


if __name__ == '__main__':
    main()

