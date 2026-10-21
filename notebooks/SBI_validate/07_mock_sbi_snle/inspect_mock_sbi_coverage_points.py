#!/usr/bin/env python3
"""Inspect saved coverage posteriors at points 124 and 8; no fitting or MCMC.

Run from the GODMAX repository root:
    python inspect_mock_sbi_coverage_points.py --points 124 8

Requires NumPy, SciPy and Matplotlib. Reads coverage_summary.npz, recovery.csv
and mock_posteriors/point_PPP_replica_RR.npz from the completed validation.
All plots and numerical summaries use probit coordinates u, whose prior is
independent N(0,1). The omitted parameters remain marginalized over.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr, ndtri
from scipy.stats import chi2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARAMETERS = ["theta_ej_0", "alpha_nt", "mu_beta", "theta_co_0", "nu_theta_ej_M"]
FOCUS = [0, 1, 4]
LABELS = [r"$u(\theta_{\rm ej,0})$", r"$u(\alpha_{\rm nt})$",
          r"$u(\mu_\beta)$", r"$u(\theta_{\rm co,0})$",
          r"$u(\nu_{\theta_{\rm ej},M})$"]
COLORS = ["#2675bd", "#e87520", "#15966c", "#9854bf"]
DEFAULT_DIR = Path("data/SBI_validate/mock_sbi/all256_native/coverage_checks/gy_g2y")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def truth_cdf(draws, truth):
    """Empirical CDF with midpoint treatment of ties; uses every saved draw."""
    return np.mean(draws < truth, axis=0) + 0.5 * np.mean(draws == truth, axis=0)


def load_cases(root, requested_points):
    with np.load(root / "coverage_summary.npz", allow_pickle=False) as archive:
        names = archive["names"].astype(str).tolist()
        points = np.asarray(archive["point_indices"], dtype=np.int64)
        weights = np.asarray(archive["point_weights"], dtype=float)
        rows = np.asarray(archive["selected_rows"], dtype=np.int64)
    require(names[:5] == PARAMETERS, "Unexpected saved parameter order")
    require(points.ndim == 1 and len(points) == len(np.unique(points)), "Invalid point IDs")
    require(weights.shape == points.shape and np.isfinite(weights).all()
            and np.all(weights >= 0) and weights.sum() > 0, "Invalid validation weights")
    require(rows.ndim == 1 and len(rows) == len(np.unique(rows)), "Invalid selected rows")
    weights = weights / weights.sum()
    point_weights = dict(zip(map(int, points), map(float, weights)))
    metadata = {}
    with (root / "recovery.csv").open(newline="") as stream:
        for record in csv.DictReader(stream):
            row = int(record["row"])
            pair = int(record["point"]), int(record["replica"])
            require(metadata.setdefault(row, pair) == pair,
                    f"Inconsistent recovery metadata for row {row}")
    require(all(int(row) in metadata for row in rows), "Recovery table is incomplete")
    cases, common_signature = [], None
    for point in requested_points:
        selected = [int(row) for row in rows if metadata[int(row)][0] == point]
        require(selected, f"Point {point} is not in this completed validation")
        for row in selected:
            _, replica = metadata[row]
            path = root / "mock_posteriors" / f"point_{point:03d}_replica_{replica:02d}.npz"
            with np.load(path, allow_pickle=False) as archive:
                require(int(archive["point"]) == point and int(archive["replica"]) == replica
                        and int(archive["row"]) == row, f"Metadata mismatch: {path}")
                draws = np.asarray(archive["u_network_repeat"], dtype=float)
                truth = np.asarray(archive["truth_u"], dtype=float)
                signature = json.loads(archive["signature_json"].item())
            require(draws.ndim == 4 and draws.shape[:2] == (4, 2)
                    and draws.shape[2] >= 20 and draws.shape[3] == 5
                    and np.isfinite(draws).all(), f"Invalid posterior array: {path}")
            require(truth.shape == (5,) and np.isfinite(truth).all(), f"Invalid truth: {path}")
            require(signature["samples_per_network"] == 2 * draws.shape[2],
                    f"Sample count differs from checkpoint signature: {path}")
            seeds = signature["fit"]["training_seeds"]
            require(len(seeds) == 4, f"Expected four training seeds: {path}")
            if common_signature is None:
                common_signature = signature
            require(signature == common_signature, "Requested checkpoints use different fits/settings")
            require(point in point_weights, f"Missing weight for point {point}")
            cases.append(dict(point=point, replica=replica, row=row, draws=draws,
                              truth=truth, weight=point_weights[point], seeds=seeds,
                              source=str(path.resolve())))
    return cases, float(1 / np.square(weights).sum())


def groups(draws):
    yield "pool", -1, -1, draws.reshape(-1, 5)
    for n in range(4):
        yield f"network_{n}", n, -1, draws[n].reshape(-1, 5)
    for n in range(4):
        for repeat in range(2):
            yield f"network_{n}_repeat_{repeat}", n, repeat, draws[n, repeat]


def ellipse_mass_at_truth(draws, truth, seed):
    """Same empirical 3D ellipsoid calculation as the marginal coverage script.

    Returns the posterior mass of the smallest fitted ellipsoid containing the
    truth. Above 0.95 means the truth misses this 95% region. This is neither
    a p-value nor an HPD probability. No Gaussian posterior is assumed.
    """
    rng = np.random.default_rng(seed)
    fitting, testing = np.array_split(draws[rng.permutation(len(draws))], 2)
    centre = fitting.mean(axis=0)
    chol = np.linalg.cholesky(np.cov(fitting, rowvar=False))
    radii = np.square(solve_triangular(chol, (testing - centre).T, lower=True)).sum(axis=0)
    true_radius = float(np.square(solve_triangular(chol, truth - centre, lower=True)).sum())
    mass = (np.sum(radii < true_radius)
            + rng.random() * (np.sum(radii == true_radius) + 1)) / (len(radii) + 1)
    return float(mass)


def numerical_checks(case):
    draws, truth = case["draws"], case["truth"]
    base = dict(point=case["point"], replica=case["replica"], row=case["row"],
                validation_point_weight=case["weight"])
    marginals, joint, repeats, variability = [], [], [], []
    for group_index, (name, n, repeat, sample) in enumerate(groups(draws)):
        quantiles = np.quantile(sample, [.025, .16, .5, .84, .975], axis=0)
        means, sd = sample.mean(axis=0), sample.std(axis=0, ddof=1)
        require(np.all(sd > 0), f"Degenerate posterior variance: {name}")
        ranks = truth_cdf(sample, truth)
        for j, parameter in enumerate(PARAMETERS):
            lo95, lo68, median, hi68, hi95 = quantiles[:, j]
            marginals.append(dict(**base, group=name, parameter=parameter,
                training_seed="" if n < 0 else int(case["seeds"][n]), n_draws=len(sample),
                truth_u=float(truth[j]), mean_u=float(means[j]), sd_u=float(sd[j]),
                lo95_u=float(lo95), lo68_u=float(lo68), median_u=float(median),
                hi68_u=float(hi68), hi95_u=float(hi95), truth_cdf=float(ranks[j]),
                contains68=bool(lo68 <= truth[j] <= hi68),
                contains95=bool(lo95 <= truth[j] <= hi95)))
        # Pool seed exactly matches the previous 3D coverage script.
        mass = ellipse_mass_at_truth(sample[:, FOCUS], truth[FOCUS],
                                     20261020 + case["row"] + 100000 * group_index)
        joint.append(dict(**base, group=name, ellipsoid_mass_at_truth=mass,
                          contains68=mass <= .68, contains95=mass <= .95))
    prior_quantiles = ndtri(np.array([.025, .16, .5, .84, .975]))
    for j, parameter in enumerate(PARAMETERS):
        lo95, lo68, median, hi68, hi95 = prior_quantiles
        marginals.append(dict(**base, group="prior", parameter=parameter,
            training_seed="", n_draws=0, truth_u=float(truth[j]), mean_u=0.0, sd_u=1.0,
            lo95_u=float(lo95), lo68_u=float(lo68), median_u=float(median),
            hi68_u=float(hi68), hi95_u=float(hi95), truth_cdf=float(ndtr(truth[j])),
            contains68=bool(lo68 <= truth[j] <= hi68),
            contains95=bool(lo95 <= truth[j] <= hi95)))
    prior_mass = float(chi2.cdf(np.square(truth[FOCUS]).sum(), df=3))
    joint.append(dict(**base, group="prior", ellipsoid_mass_at_truth=prior_mass,
                      contains68=prior_mass <= .68, contains95=prior_mass <= .95))

    network_draws = draws.reshape(4, -1, 5)
    network_means = network_draws.mean(axis=1)
    network_sd = network_draws.std(axis=1, ddof=1)
    within = network_draws.var(axis=1, ddof=0).mean(axis=0)
    between = network_means.var(axis=0, ddof=0)
    require(np.all(within + between > 0), "Degenerate mixture variance")
    pool_sd = draws.reshape(-1, 5).std(axis=0, ddof=1)
    repeat_means = draws.mean(axis=2)
    for n in range(4):
        cdf0 = truth_cdf(draws[n, 0], truth)
        cdf1 = truth_cdf(draws[n, 1], truth)
        for j, parameter in enumerate(PARAMETERS):
            repeats.append(dict(**base, network=n, training_seed=int(case["seeds"][n]),
                parameter=parameter, mean_repeat0_u=float(repeat_means[n, 0, j]),
                mean_repeat1_u=float(repeat_means[n, 1, j]),
                repeat_mean_shift_over_sd=float(abs(repeat_means[n, 0, j]
                                                  - repeat_means[n, 1, j]) / network_sd[n, j]),
                truth_cdf_repeat0=float(cdf0[j]), truth_cdf_repeat1=float(cdf1[j]),
                absolute_cdf_difference=float(abs(cdf0[j] - cdf1[j]))))
    for j, parameter in enumerate(PARAMETERS):
        variability.append(dict(**base, parameter=parameter,
            network_mean_scatter_over_pool_sd=float(network_means[:, j].std(ddof=1) / pool_sd[j]),
            between_network_variance_fraction=float(between[j] / (within[j] + between[j])),
            max_repeat_mean_shift_over_sd=max(r["repeat_mean_shift_over_sd"]
                                             for r in repeats if r["parameter"] == parameter)))
    return marginals, joint, repeats, variability


def display_range(pooled, truth):
    lo, hi = np.quantile(pooled, [.001, .999])
    lo, hi = min(float(lo), float(truth), -2.5), max(float(hi), float(truth), 2.5)
    pad = .04 * (hi - lo)
    return lo - pad, hi + pad


def density(ax, sample, edges, color, label, lw=1.6):
    # Same absolute bins/smoothing for all groups in each parameter panel.
    # Divide by ALL draws, so displayed tails are not renormalized away.
    counts, _ = np.histogram(sample, bins=edges)
    value = gaussian_filter1d(counts.astype(float), sigma=1.0, mode="constant")
    value /= len(sample) * (edges[1] - edges[0])
    ax.plot(.5 * (edges[:-1] + edges[1:]), value, color=color, lw=lw, label=label)


def prior_and_truth(ax, edges, truth):
    x = np.linspace(edges[0], edges[-1], 400)
    ax.plot(x, np.exp(-.5 * x*x) / np.sqrt(2*np.pi), color="0.6", ls=":", label="Prior")
    ax.axvline(truth, color="black", ls="--", lw=1.3, label="Truth")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.18)


def make_plots(case, marginals, outdir):
    draws, truth = case["draws"], case["truth"]
    pooled = draws.reshape(-1, 5)
    edges = [np.linspace(*display_range(pooled[:, j], truth[j]), 121) for j in range(5)]
    stem = f"point_{case['point']:03d}_replica_{case['replica']:02d}"
    heading = f"Point {case['point']}, replica {case['replica']}; validation weight {case['weight']:.1%}"
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), constrained_layout=True)
    for ax, j in zip(axes.flat, FOCUS + [2, 3]):
        for n in range(4):
            density(ax, draws[n, :, :, j].ravel(), edges[j], COLORS[n], f"Network {n}")
        density(ax, pooled[:, j], edges[j], "black", "Four-network pool", lw=2.5)
        prior_and_truth(ax, edges[j], truth[j])
        ax.set(xlabel=LABELS[j], ylabel="Marginal density")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles, labels, loc="upper left", frameon=False)
    axes.flat[-1].text(.02, .12,
        "All axes: probit coordinates u.\nCurves use common smoothed histograms.\n"
        "Numerical checks use every saved draw.\nExtreme plot tails may be outside the axes.",
        transform=axes.flat[-1].transAxes, fontsize=9)
    fig.suptitle(heading + "\nPosteriors by network")
    fig.savefig(outdir / f"{stem}_networks.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(13.5, 10), sharex="col", sharey="col",
                             constrained_layout=True)
    for n in range(4):
        for col, j in enumerate(FOCUS):
            ax = axes[n, col]
            for repeat, color in enumerate(("#2675bd", "#e87520")):
                density(ax, draws[n, repeat, :, j], edges[j], color, f"Repeat {repeat}")
            prior_and_truth(ax, edges[j], truth[j])
            if n == 0:
                ax.set_title(PARAMETERS[j])
            if col == 0:
                ax.set_ylabel(f"Network {n}\nDensity")
            if n == 3:
                ax.set_xlabel(LABELS[j])
    for col in range(3):
        top = max(ax.dataLim.ymax for ax in axes[:, col])
        axes[0, col].set_ylim(0, 1.08 * top)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(heading + "\nIndependent MCMC repeats within each network")
    fig.savefig(outdir / f"{stem}_repeats.png", dpi=160)
    plt.close(fig)

    chosen = ["prior", "network_0", "network_1", "network_2", "network_3", "pool"]
    color_by_group = dict(zip(chosen, ["0.6", *COLORS, "black"]))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), constrained_layout=True)
    for ax, j in zip(axes, FOCUS):
        for y, group in enumerate(chosen):
            r = next(r for r in marginals if r["group"] == group and r["parameter"] == PARAMETERS[j])
            color = color_by_group[group]
            ax.plot([r["lo95_u"], r["hi95_u"]], [y, y], color=color, lw=1.2)
            ax.plot([r["lo68_u"], r["hi68_u"]], [y, y], color=color, lw=5, solid_capstyle="butt")
            ax.plot(r["median_u"], y, "o", color=color, ms=4)
        ax.axvline(truth[j], color="black", ls="--", lw=1.3)
        ax.set_yticks(range(len(chosen)), [g.replace("_", " ") for g in chosen])
        ax.invert_yaxis()
        ax.set_xlabel(LABELS[j])
        ax.grid(alpha=.18)
    fig.suptitle(heading + "\nMedian; thick bars: 68%; thin bars: 95%; dashed line: truth")
    fig.savefig(outdir / f"{stem}_intervals.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--points", nargs="+", type=int, default=[124, 8])
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    require(len(args.points) == len(set(args.points)), "Duplicate requested points")
    cases, effective_points = load_cases(args.coverage_dir, args.points)
    outdir = args.outdir or args.coverage_dir / "point_checks"
    outdir.mkdir(parents=True, exist_ok=True)
    all_marginals, all_joint, all_repeats, all_variability = [], [], [], []
    print(f"Validation importance ESS: {effective_points:.3f}")
    for case in cases:
        marginals, joint, repeats, variability = numerical_checks(case)
        all_marginals.extend(marginals)
        all_joint.extend(joint)
        all_repeats.extend(repeats)
        all_variability.extend(variability)
        make_plots(case, marginals, outdir)
        print(f"\nPoint {case['point']}, replica {case['replica']}, weight={case['weight']:.6f}")
        print("Truth CDF: fraction of draws below truth; 95% central interval uses 0.025..0.975.")
        for j in FOCUS:
            print(f"  {PARAMETERS[j]}: true u={case['truth'][j]:.6f}")
            for group in ("prior", "network_0", "network_1", "network_2", "network_3", "pool"):
                r = next(r for r in marginals if r["group"] == group and r["parameter"] == PARAMETERS[j])
                print(f"    {group:10s} CDF={r['truth_cdf']:.5f}; "
                      f"95%=[{r['lo95_u']:.4f}, {r['hi95_u']:.4f}]; contains={r['contains95']}")
        print("  3D ellipsoid mass at truth (>0.95 means outside its 95% region):")
        for r in joint:
            if "repeat" not in r["group"]:
                print(f"    {r['group']:10s}: {r['ellipsoid_mass_at_truth']:.5f}")
        print("  Largest repeat mean shift / network SD: "
              f"{max(r['repeat_mean_shift_over_sd'] for r in repeats):.3f}")
    for name, records in [("marginals.csv", all_marginals), ("joint_3d.csv", all_joint),
                          ("repeat_checks.csv", all_repeats), ("network_variability.csv", all_variability)]:
        write_csv(outdir / name, records)
    notes = dict(parameter_order=PARAMETERS, focus_parameters=[PARAMETERS[j] for j in FOCUS],
        sources=[case["source"] for case in cases], importance_effective_points=effective_points,
        posterior_pool="Equal weight for four networks, each with two equally sized MCMC repeats",
        validation_weights="p(u)/q(u), retained for context only; no posterior draws are reweighted",
        axes="Probit u; prior independent standard normal",
        ellipsoid="Empirical split-sample posterior mass in three dimensions; not an HPD region or p-value",
        limitations="The archive has flattened draws, not chain-indexed trajectories. No R-hat or chain ESS "
                    "is inferred. Repeat agreement and network agreement do not certify convergence or coverage.")
    (outdir / "inspection_notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    print(f"\nSaved figures and four CSV tables to {outdir}")
    print("These are pointwise recovery diagnostics, not a new population coverage estimate.")


if __name__ == "__main__":
    main()

