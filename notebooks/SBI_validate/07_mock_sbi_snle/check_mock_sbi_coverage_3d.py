#!/usr/bin/env python3
"""Assess the 3D marginal using completed coverage-validation checkpoints."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr, ndtri

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PARAMETERS = [
    "theta_ej_0", "alpha_nt", "mu_beta",
    "theta_co_0", "nu_theta_ej_M",
]
KEEP = [0, 1, 4]

DEFAULT_DIR = Path(
    "data/SBI_validate/mock_sbi/all256_native/"
    "coverage_checks/gy_g2y"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rank(samples, truth, rng):
    """Randomized posterior CDF rank, including finite-sample correction."""
    n = len(samples)
    below = np.sum(samples < truth, axis=0)
    ties = np.sum(samples == truth, axis=0)
    return (below + rng.random(np.shape(truth)) * (ties + 1)) / (n + 1)


def joint_checks(draws, truth, levels, seed, references):
    rng = np.random.default_rng(seed)

    # Fit an ellipsoidal geometry using half the posterior draws.
    # Measure its enclosed posterior mass using the other half.
    shuffled = draws[rng.permutation(len(draws))]
    fitting, testing = np.array_split(shuffled, 2)
    centre = fitting.mean(axis=0)
    chol = np.linalg.cholesky(np.cov(fitting, rowvar=False))

    radii = np.square(
        solve_triangular(chol, (testing - centre).T, lower=True)
    ).sum(axis=0)
    true_radius = np.square(
        solve_triangular(chol, truth - centre, lower=True)
    ).sum()

    ellipsoid = (rank(radii, true_radius, rng) <= levels).astype(float)

    # TARP in physical coordinates scaled to the fixed prior box:
    # (theta - theta_min)/(theta_max - theta_min) = Phi(u).
    unit_draws = ndtr(draws)
    unit_truth = ndtr(truth)
    refs = rng.uniform(size=(references, 3))

    distances = np.square(
        unit_draws[:, None, :] - refs[None, :, :]
    ).sum(axis=2)
    true_distances = np.square(unit_truth - refs).sum(axis=1)
    ball_ranks = rank(distances, true_distances, rng)
    tarp = (ball_ranks[None, :] <= levels[:, None]).mean(axis=1)

    return np.column_stack([ellipsoid, tarp])


def uncertainty_bands(values, weights, points, repetitions):
    """Match the original approximate point-level uncertainty calculation."""
    rng = np.random.default_rng(20261020)

    # Registered point numbering: 0..127 first campaign, 128..255 second.
    groups = [
        np.flatnonzero(points < 128),
        np.flatnonzero(points >= 128),
    ]
    groups = [group for group in groups if len(group)]

    boots = np.empty((repetitions, *values.shape[1:]))
    for b in range(repetitions):
        selected = np.concatenate([
            rng.choice(group, size=len(group), replace=True)
            for group in groups
        ])
        boots[b] = np.average(
            values[selected], axis=0, weights=weights[selected]
        )

    bands = np.quantile(boots, [0.05, 0.95], axis=0)

    # Prevent all-success/all-failure bootstrap bands from collapsing.
    # This is an approximation, not an exact weighted-binomial interval.
    estimate = np.clip(
        np.average(values, axis=0, weights=weights), 0, 1
    )
    effective_points = 1 / np.square(weights).sum()
    z = ndtri(0.95)
    denominator = 1 + z**2 / effective_points
    centre = (
        estimate + z**2 / (2 * effective_points)
    ) / denominator
    half_width = z / denominator * np.sqrt(
        estimate * (1 - estimate) / effective_points
        + z**2 / (4 * effective_points**2)
    )

    bands[0] = np.minimum(bands[0], np.maximum(0, centre - half_width))
    bands[1] = np.maximum(bands[1], np.minimum(1, centre + half_width))
    return bands


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_dir", nargs="?", type=Path,
                        default=DEFAULT_DIR)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--references", type=int, default=32)
    args = parser.parse_args()
    require(args.bootstrap >= 200 and args.references >= 1,
            "Invalid bootstrap/reference counts")

    root = args.coverage_dir
    out = args.outdir or root / "marginal3d"

    with np.load(root / "coverage_summary.npz",
                 allow_pickle=False) as archive:
        old_names = archive["names"].astype(str).tolist()
        levels = np.asarray(archive["levels"], float)
        rows = np.asarray(archive["selected_rows"], np.int64)
        points = np.asarray(archive["point_indices"], np.int64)
        weights = np.asarray(archive["point_weights"], float)
        old_point_values = np.asarray(archive["point_coverage"], float)
        old_weighted = np.asarray(
            archive["prior_weighted_coverage"], float
        )

    require(old_names[:5] == PARAMETERS, "Unexpected parameter order")
    require(np.array_equal(points, np.unique(points)),
            "Point IDs must be unique and sorted")
    require(len(points) >= 2 and np.all((points >= 0) & (points < 256)),
            "Unexpected registered point IDs")
    require(len(rows) == len(np.unique(rows)), "Duplicate selected rows")
    require(weights.shape == points.shape
            and np.isfinite(weights).all() and np.all(weights > 0),
            "Invalid saved point weights")
    require(old_point_values.shape ==
            (len(points), len(levels), len(old_names)),
            "Unexpected saved coverage-array shape")
    weights = weights / weights.sum()
    require(weights.max() < 1, "A single point has all the weight")

    # Recover exact checkpoint names from the completed recovery table.
    metadata = {}
    with (root / "recovery.csv").open(newline="") as stream:
        for record in csv.DictReader(stream):
            row = int(record["row"])
            pair = (int(record["point"]), int(record["replica"]))
            require(metadata.setdefault(row, pair) == pair,
                    f"Inconsistent recovery metadata for row {row}")

    position = {int(point): i for i, point in enumerate(points)}
    joint_sum = np.zeros((len(points), len(levels), 2))
    counts = np.zeros(len(points), dtype=int)
    common_signature = None

    for number, row in enumerate(rows, start=1):
        row = int(row)
        require(row in metadata, f"Missing recovery metadata for row {row}")
        point, replica = metadata[row]
        require(point in position, f"Unexpected point {point}")

        path = root / "mock_posteriors" / (
            f"point_{point:03d}_replica_{replica:02d}.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            require(int(archive["row"]) == row
                    and int(archive["point"]) == point
                    and int(archive["replica"]) == replica,
                    f"Checkpoint metadata mismatch: {path}")
            signature = str(archive["signature_json"].item())
            samples = np.asarray(archive["u_network_repeat"], float)
            truth = np.asarray(archive["truth_u"], float)

        if common_signature is None:
            common_signature = signature
        require(signature == common_signature,
                f"Mixed validation configurations: {path}")
        require(samples.ndim == 4
                and samples.shape[:2] == (4, 2)
                and samples.shape[-1] == 5
                and samples.shape[2] >= 20
                and np.isfinite(samples).all(),
                f"Invalid posterior samples: {path}")
        require(truth.shape == (5,) and np.isfinite(truth).all(),
                f"Invalid saved truth: {path}")

        # Selecting columns marginalizes over the omitted parameters.
        # No conditioning, cuts, or posterior reweighting are applied.
        draws3 = samples[..., KEEP].reshape(-1, 3)
        values = joint_checks(
            draws3, truth[KEEP], levels,
            seed=20261020 + row, references=args.references,
        )
        i = position[point]
        joint_sum[i] += values
        counts[i] += 1
        print(f"[{number}/{len(rows)}] point {point}, replica {replica}",
              flush=True)

    require(np.all(counts > 0), "Some saved validation points have no mocks")
    joint_values = joint_sum / counts[:, None, None]

    # Individual marginal coverage is unchanged by taking a joint marginal.
    values = np.concatenate([
        old_point_values[:, :, KEEP], joint_values
    ], axis=2)
    names = [PARAMETERS[j] for j in KEEP] + [
        "joint_ellipsoid_3D", "joint_TARP_3D"
    ]

    weighted = np.clip(
        np.average(values, axis=0, weights=weights), 0, 1
    )
    guided = values.mean(axis=0)
    bands = uncertainty_bands(
        values, weights, points, args.bootstrap
    )
    effective_points = 1 / np.square(weights).sum()
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for j, name in enumerate(names):
        for k, mass in enumerate(levels):
            records.append(dict(
                statistic=name,
                nominal_mass=float(mass),
                prior_weighted_coverage=float(weighted[k, j]),
                approximate_90_low=float(bands[0, k, j]),
                approximate_90_high=float(bands[1, k, j]),
                guided_recovery_fraction=float(guided[k, j]),
            ))
    write_csv(out / "coverage_3d.csv", records)

    # Sensitivity diagnostic only: the main estimate retains every point.
    influence = []
    for i in np.argsort(weights)[::-1]:
        without = (
            weighted - weights[i] * values[i]
        ) / (1 - weights[i])
        for mass in (0.68, 0.95):
            k = int(np.argmin(np.abs(levels - mass)))
            require(np.isclose(levels[k], mass), f"Missing level {mass}")
            for j in (3, 4):
                influence.append(dict(
                    point=int(points[i]),
                    point_weight=float(weights[i]),
                    statistic=names[j],
                    nominal_mass=mass,
                    point_recovery=float(values[i, k, j]),
                    coverage_without_point=float(without[k, j]),
                    change_from_full=float(without[k, j] - weighted[k, j]),
                ))
    write_csv(out / "point_influence_3d.csv", influence)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             constrained_layout=True)
    for ax, j, old_name in zip(
        axes[:2], (3, 4), ("joint_ellipsoid_5D", "joint_TARP_5D")
    ):
        ax.plot([0, 1], [0, 1], "k--", lw=1,
                label="Calibrated prior average")
        ax.fill_between(levels, bands[0, :, j], bands[1, :, j],
                        color="tab:blue", alpha=0.2,
                        label="3D approximate 90% uncertainty")
        ax.plot(levels, weighted[:, j], "o-", color="tab:blue",
                ms=3, label="3D marginal")
        ax.plot(levels, old_weighted[:, old_names.index(old_name)],
                color="tab:orange", ls=":", label="Original 5D check")
        ax.set(title=names[j], xlabel="Nominal credible mass",
               ylabel="Prior-weighted coverage", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)

    order = np.argsort(weights)[::-1]
    axes[2].bar(np.arange(len(points)), weights[order], color="tab:blue")
    axes[2].set_xticks(np.arange(len(points)), points[order],
                      rotation=90, fontsize=7)
    axes[2].set(title="Validation point weights",
                xlabel="Point ID", ylabel="Normalized importance weight")
    fig.suptitle(
        f"3D marginal: {len(points)} points, {len(rows)} noisy mocks; "
        f"importance ESS = {effective_points:.2f}"
    )
    fig.savefig(out / "coverage_3d.png", dpi=180)
    plt.close(fig)

    print(f"\nImportance ESS: {effective_points:.3f}")
    print(f"Largest point weight: {weights.max():.1%}")
    for mass in (0.68, 0.95):
        k = int(np.argmin(np.abs(levels - mass)))
        print(f"\nNominal mass {mass:.0%}:")
        for j, name in enumerate(names):
            print(
                f"{name:24s} weighted={weighted[k, j]:.3f} "
                f"[{bands[0, k, j]:.3f}, {bands[1, k, j]:.3f}]; "
                f"guided recovery={guided[k, j]:.3f}"
            )
    print("\nThe diagonal target applies to the prior-weighted estimate.")
    print("Uncertainty bands are approximate and pointwise.")
    print(f"Saved outputs to {out}")


if __name__ == "__main__":
    main()
