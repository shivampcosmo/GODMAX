#!/usr/bin/env python3
"""Held-out noisy-mock recovery for the native 128x5 + 128x10 experiment.

Place beside run_mock_sbi_all256_snle.py. No new maps or parameter points.
Uses that runner's master reader and separately fitted component scores.
Retrains the four likelihoods using the reference posterior's RECORDED
training settings/seeds, saves the estimators, and conditions them on each
selected held-out noisy row. Existing inference products are never modified.

Calibration summaries: marginal equal-tail intervals, empirical joint
ellipsoidal credible regions in u, and joint TARP random balls in physical
coordinates scaled to the fixed prior box. Ellipsoid masses use posterior
sample distances; no Gaussian/chi-squared posterior approximation is made.
The guided design is not prior predictive: prior-averaged curves use p(u)/q(u)
weights from point_log_q. These are finite-design, self-normalized estimates,
not a new prior-drawn SBC experiment or an automatic calibration certificate.
Bootstrap uncertainty resamples parameter points, stratified by campaign.

Run once with --probe gy_g2y; preflight is included. --preflight alone is
an optional read-only check. Rerunning resumes completed model/mock checkpoints,
not an interrupted training call. progress.json records the current stage.
Use another --outdir if changing the reference, sampler settings, or samples
per network.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import time

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

BASE = Path("data/SBI_validate/mock_sbi/all256_native")
FIELDS = {"gy": "y", "g2y": "y", "gkappa": "kappa",
          "g2kappa": "kappa", "gtau": "tau", "g2tau": "tau"}
LEVELS = np.unique(np.round(np.r_[np.arange(.05, 1., .05), .68, .95], 10))


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def progress(args, stage, **details):
    """Current stage only; resumable weights/draws live in the checkpoint files."""
    path = args.outdir / "progress.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(
        updated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        stage=stage, probe=args.probe, **details), indent=2) + "\n")
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def assert_roundoff_close(actual, expected, label):
    """Compare numerical contracts without requiring identical floating-point bits.

    The absolute tolerance follows the array's own units. A fixed absolute
    tolerance would be inappropriate for very small bandpower covariances.
    """
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape, f"{label}: shapes differ")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), f"{label}: non-finite values")
    scale = max(float(np.max(np.abs(expected), initial=0)), np.finfo(np.float64).tiny)
    np.testing.assert_allclose(actual, expected, rtol=1e-10,
                               atol=128 * np.finfo(np.float64).eps * scale,
                               err_msg=f"{label} differs beyond floating-point roundoff")


def replay_saved_scores(gx, g2x, components, contracts):
    """Independent replay of the saved elementary 14-to-5 transformations."""
    pieces = []
    starts = {"y": 0, "kappa": 14, "tau": 28}
    for component in components:
        start = starts[FIELDS[component]]
        values = np.atleast_2d(np.asarray(g2x if component.startswith("g2") else gx,
                                         dtype=np.float64))[..., start:start + 14]
        c = contracts[component]
        whitened = np.linalg.solve(c["cholesky"], (values - c["reference"]).T)
        pieces.append((c["operator"] @ whitened).T)
    return np.concatenate(pieces, axis=1)


def file_hash(path):
    """SHA-256 of the file bytes, including on Python 3.10."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def import_runner(path):
    path = path.expanduser().resolve()
    require(path.is_file(), f"Missing current runner: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_coverage_source_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("_load_master", "_build_component_contracts", "_summarise_probe"):
        require(hasattr(module, name), f"Runner lacks required interface {name}")
    return module


def choose_rows(arrays, seed_order, components, limit):
    """Choose noisy rows without train/test or test/test noise reuse.

    Selection uses only point IDs, replica IDs, and noise seeds. No observed
    statistic or inference result is used. Round-robin selection gives each
    point a chance before taking additional replicas at other points.
    """
    index = np.asarray(arrays["point_index"], int)
    split = np.asarray(arrays["point_split"], int)
    training = split[index] == 0
    columns = [seed_order.index(field) for field in sorted({FIELDS[c] for c in components})]
    seeds = np.asarray(arrays["noise_seeds"])
    train_seeds = [set(seeds[training, col].tolist()) for col in columns]
    heldout = np.flatnonzero(~training)
    eligible = [int(row) for row in heldout if all(
        int(seeds[row, col]) not in used for col, used in zip(columns, train_seeds))]
    points = np.flatnonzero(split == 1)
    pending = {int(p): sorted([r for r in eligible if index[r] == p],
                             key=lambda r: int(arrays["replica"][r])) for p in points}
    selected = []
    chosen_seeds = [set() for _ in columns]
    counts = {int(p): 0 for p in points}
    while True:
        added = 0
        for point in points:
            point = int(point)
            if limit and counts[point] >= limit:
                continue
            while pending[point]:
                row = pending[point].pop(0)
                if any(int(seeds[row, col]) in used
                       for col, used in zip(columns, chosen_seeds)):
                    continue
                selected.append(row)
                counts[point] += 1
                for col, used in zip(columns, chosen_seeds):
                    used.add(int(seeds[row, col]))
                added += 1
                break
        if not added:
            break
    require(selected, "No held-out rows have noise seeds disjoint from training")
    audit = dict(heldout_rows=int(len(heldout)), train_noise_disjoint_rows=len(eligible),
                 selected_rows=len(selected), selected_points=sum(c > 0 for c in counts.values()),
                 unavailable_points=[p for p, count in counts.items() if not count],
                 fields=[seed_order[c] for c in columns],
                 test_noise_policy="disjoint from training and other selected test rows")
    return np.asarray(selected, dtype=int), audit


def prepare(args):
    log("Preflight: loading runner, master, and saved score contracts")
    runner = import_runner(args.runner)
    require(args.probe in runner.PROBE_SPECS, f"Unknown probe {args.probe}")
    import mock_sbi_common as msc
    master = runner._load_master(args.master or args.base / "gx_g2x_native_all256.npz", msc)
    arrays = master["arrays"]
    components = runner._components_for_probe(args.probe)
    contracts = runner._build_component_contracts(arrays, args.probe, "native")
    summary = runner._summarise_probe(arrays["gx"], arrays["g2x"], args.probe, contracts, "score")
    index = np.asarray(arrays["point_index"], int)
    training_mask = arrays["point_split"][index] == 0
    require(np.sum(training_mask) == 1725 and np.unique(index[training_mask]).size == 230,
            "Expected the frozen 230-point / 1725-row native training split")
    if args.reference is None:
        marker = f"_sepscore{5 * len(components)}" if len(components) > 1 else ""
        args.reference = args.base / f"mock_snle_all256_combined_native_contractnative{marker}_{args.probe}_posterior.npz"
    require(args.reference.is_file(), f"Missing reference posterior: {args.reference}")
    with np.load(args.reference, allow_pickle=False) as archive:
        man = json.loads(archive["manifest_json"].item())
        require(man["probe"] == args.probe and man["compression"] == "score",
                "Reference probe/compression differs")
        require((man["cohort"], man["training_replica_policy"], man["contract_replica_policy"])
                == ("combined", "native", "native"), "Reference must use combined/native/native")
        require(man["parameter_names"] == list(runner.PARAMETER_NAMES), "Parameter order differs")
        require(man["master_sha256"] == master["file_sha256"], "Reference master hash differs")
        require(man["point_split_sha256"] == master["manifest"]["point_split"]["sha256"],
                "Reference frozen split differs")
        np.testing.assert_array_equal(archive["point_split"], arrays["point_split"])
        require(man["summary_dimension"] == summary.shape[1] == 5 * len(components),
                "Reference summary dimension differs")
        require(man.get("component_order", list(components)) == list(components),
                "Component order differs")
        for component in components:
            suffix = "_" + component if len(components) > 1 else ""
            for saved, current in (("covariance", "covariance"), ("score_operator", "operator"),
                                   ("score_reference_prediction", "reference")):
                value = np.asarray(archive[saved + suffix], dtype=np.float64)
                assert_roundoff_close(value, contracts[component][current], f"{component}: {saved}")
                # Replay the saved transformation after checking that it was
                # fitted on the current frozen training set.
                contracts[component][current] = value
            cholesky_key = "score_cholesky_" + component
            L = (np.asarray(archive[cholesky_key], dtype=np.float64) if len(components) > 1
                 else np.linalg.cholesky(contracts[component]["covariance"]))
            assert_roundoff_close(L @ L.T, contracts[component]["covariance"], f"{component}: Cholesky factor")
            contracts[component]["cholesky"] = L
        summary = replay_saved_scores(arrays["gx"], arrays["g2x"], components, contracts)
        runner_summary = runner._summarise_probe(arrays["gx"], arrays["g2x"], args.probe, contracts, "score")
        assert_roundoff_close(runner_summary, summary, "runner versus saved-score replay")
        observed = replay_saved_scores(arrays["gx_data_vector"], arrays["g2x_data_vector"],
                                        components, contracts)[0]
        np.testing.assert_allclose(observed, man["observed_summary"], rtol=1e-10, atol=1e-12,
                                   err_msg="saved observed summary differs")
    require(isinstance(man.get("training_summary_sha256"), str)
            and len(man["training_summary_sha256"]) == 64, "Missing historical training-summary hash")
    summary_hash_match = man["training_summary_sha256"] == msc.sha256_array(summary[training_mask])
    print("Historical float64 training-summary hash: " + ("exact match" if summary_hash_match else
          "differs; exact master/split and numerical saved-score replay checks passed"), flush=True)
    records = man["per_network"]
    require(man["network_seeds"] == len(records) == 4, "Expected one four-network reference run")
    seeds = [int(record["training_seed"]) for record in records]
    require(len(set(seeds)) == 4, "Repeated network training seeds")
    order = master["manifest"]["seed_field_order"]
    log(f"Requested test replicas per held-out point: {args.replicas_per_point} (0 = all eligible)")
    rows, audit = choose_rows(arrays, order, components, args.replicas_per_point)
    points = np.unique(index[rows])
    require(np.isfinite(arrays["point_log_q"][points]).all(), "Non-finite guided point_log_q")
    logp = -.5 * np.square(arrays["point_u"][points]).sum(axis=1) - 2.5 * np.log(2 * np.pi)
    logw = logp - arrays["point_log_q"][points]
    weights = np.exp(logw - logsumexp(logw))
    audit.update(importance_effective_points=float(1 / np.square(weights).sum()),
                 largest_point_weight=float(weights.max()),
                 historical_training_summary_hash_match=summary_hash_match)
    require(man["mcmc"]["independent_repeats"] == 2, "Expected two independent sampler repeats")
    require(args.samples_per_network % (2 * man["mcmc"]["num_chains"]) == 0,
            "--samples-per-network must be divisible by twice the recorded chain count")
    print(f"128x5 + 128x10 = 1920 rows; train: 230 points / 1725 rows. Probe: {args.probe}", flush=True)
    print(f"Score components: {components}; dimension: {summary.shape[1]}", flush=True)
    print(f"Recorded training: {man['training']}; recorded MCMC: {man['mcmc']}", flush=True)
    print("Noisy-test selection:", json.dumps(audit, indent=2), flush=True)
    print("Each test uses one noisy row; point means are NOT used as observations.", flush=True)
    print("Prior weights are for assessment only; training and posterior priors are unchanged.", flush=True)
    return runner, master, man, summary, training_mask, rows, weights, audit


def randomized_rank(samples, truth, rng):
    """CDF rank with finite-draw randomization; also handles exact ties."""
    samples = np.asarray(samples)
    return (np.sum(samples < truth, axis=0) + rng.random(np.shape(truth))
            * (np.sum(samples == truth, axis=0) + 1)) / (len(samples) + 1)


def assess_draws(draws, truth, seed, references):
    """Marginal intervals, empirical joint ellipsoids, and TARP balls."""
    draws, truth = np.asarray(draws, float), np.asarray(truth, float)
    require(draws.ndim == 2 and draws.shape[1] == truth.size == 5
            and np.isfinite(draws).all(), "Invalid posterior draws")
    rng = np.random.default_rng(seed)
    ranks = randomized_rank(draws, truth, rng)
    central = np.abs(2 * ranks - 1)[None, :] <= LEVELS[:, None]
    # Split samples to estimate geometry and measure its credible mass with
    # separate draws. Empirical distance quantiles work for non-Gaussian
    # posteriors too; the region shape is simply chosen to be ellipsoidal.
    shuffled = draws[rng.permutation(len(draws))]
    fitting, testing = np.array_split(shuffled, 2)
    centre = fitting.mean(axis=0)
    L = np.linalg.cholesky(np.cov(fitting, rowvar=False))
    from scipy.linalg import solve_triangular
    radii = np.square(solve_triangular(L, (testing - centre).T, lower=True)).sum(axis=0)
    true_radius = np.square(solve_triangular(L, truth - centre, lower=True)).sum()
    ellipse_rank = randomized_rank(radii, true_radius, rng)
    ellipse_coverage = ellipse_rank <= LEVELS
    # Fixed physical-prior scaling: (theta-low)/(high-low) = Phi(u).
    # References have full support on the prior box and do not use the truth.
    unit_draws, unit_truth = ndtr(draws), ndtr(truth)
    refs = rng.uniform(size=(references, 5))
    distances = np.square(unit_draws[:, None, :] - refs[None, :, :]).sum(axis=2)
    true_distances = np.square(unit_truth - refs).sum(axis=1)
    ball_ranks = randomized_rank(distances, true_distances, rng)
    ball_coverage = (ball_ranks[None, :] <= LEVELS[:, None]).mean(axis=1)
    quantiles = np.quantile(draws, [.025, .16, .5, .84, .975], axis=0)
    return ranks, central, ellipse_coverage, ball_coverage, quantiles


def point_bootstrap(point_values, weights, cohorts, repetitions, seed):
    """Point-cluster bootstrap with self-normalized importance weights."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cohorts == label) for label in np.unique(cohorts)]
    values = []
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        values.append(np.average(point_values[indices], axis=0, weights=weights[indices]))
    return np.quantile(values, [.05, .95], axis=0)


def report(args, runner, master, rows, weights, results, audit):
    """Point-equal guided recovery and importance-weighted coverage."""
    arrays = master["arrays"]
    index = arrays["point_index"][rows]
    points = np.unique(index)
    marginal = np.stack([record["central"] for record in results])
    ellipsoid = np.stack([record["ellipsoid"] for record in results])
    tarp = np.stack([record["tarp"] for record in results])
    all_values = np.concatenate([marginal.astype(float), ellipsoid[:, :, None], tarp[:, :, None]], axis=2)
    point_values = np.stack([all_values[index == point].mean(axis=0) for point in points])
    guided = point_values.mean(axis=0)
    weighted = np.average(point_values, axis=0, weights=weights)
    bands = point_bootstrap(point_values, weights, arrays["point_cohort"][points],
                           args.bootstrap, 20261013)
    # A pure bootstrap collapses at all-success/all-failure boundaries. Widen
    # it with an effective-point Wilson approximation; these remain approximate
    # pointwise uncertainty bands, not exact weighted-binomial confidence bounds.
    effective_points = 1 / np.square(weights).sum()
    z = ndtri(.95)
    denominator = 1 + z**2 / effective_points
    centre = (weighted + z**2 / (2 * effective_points)) / denominator
    half_width = z / denominator * np.sqrt(weighted * (1 - weighted) / effective_points
                                           + z**2 / (4 * effective_points**2))
    bands[0] = np.minimum(bands[0], np.maximum(0, centre - half_width))
    bands[1] = np.maximum(bands[1], np.minimum(1, centre + half_width))
    names = [*runner.PARAMETER_NAMES, "joint_ellipsoid_5D", "joint_TARP_5D"]
    records = []
    for dimension, name in enumerate(names):
        for k, level in enumerate(LEVELS):
            records.append(dict(statistic=name, nominal_mass=float(level),
                                prior_weighted_coverage=float(weighted[k, dimension]),
                                point_bootstrap_90_low=float(bands[0, k, dimension]),
                                point_bootstrap_90_high=float(bands[1, k, dimension]),
                                guided_recovery_fraction=float(guided[k, dimension])))
    with (args.outdir / "coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)
    ranks = np.stack([record["ranks"] for record in results])
    quantiles = np.stack([record["quantiles"] for record in results])
    observation_weights = np.asarray([weights[np.flatnonzero(points == point)[0]]
                                     / np.sum(index == point) for point in index])
    recovery_records = []
    for row, record, q in zip(rows, results, quantiles):
        point = int(arrays["point_index"][row])
        for j, name in enumerate(runner.PARAMETER_NAMES):
            recovery_records.append(dict(row=int(row), point=point, replica=int(arrays["replica"][row]),
                parameter=name, truth_u=float(arrays["point_u"][point, j]),
                lo95_u=float(q[0, j]), lo68_u=float(q[1, j]), median_u=float(q[2, j]),
                hi68_u=float(q[3, j]), hi95_u=float(q[4, j]), rank=float(record["ranks"][j]),
                max_repeat_mean_shift_sigma=float(record["repeat_shift"])))
    with (args.outdir / "recovery.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=recovery_records[0].keys())
        writer.writeheader(); writer.writerows(recovery_records)
    atomic_npz(args.outdir / "coverage_summary.npz", levels=LEVELS, selected_rows=rows,
               point_indices=points, point_weights=weights, observation_weights=observation_weights,
               marginal_ranks=ranks, quantiles_u=quantiles, point_coverage=point_values,
               prior_weighted_coverage=weighted, guided_recovery_fraction=guided,
               point_bootstrap_90=bands, names=np.asarray(names))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for j, (ax, name) in enumerate(zip(axes.flat, names)):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibrated prior average")
        ax.fill_between(LEVELS, bands[0, :, j], bands[1, :, j], color="#1479b8", alpha=.2,
                        label="Approx. 90% point uncertainty")
        ax.plot(LEVELS, weighted[:, j], "o-", color="#1479b8", ms=3, label="prior-weighted estimate")
        ax.set(xlim=(0, 1), ylim=(0, 1), title=name, xlabel="Nominal credible mass",
               ylabel="Estimated coverage"); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle(f"{args.probe}: {len(points)} held-out points, {len(rows)} noisy mocks; "
                 f"importance ESS = {audit['importance_effective_points']:.1f}\n"
                 "Below diagonal: undercoverage; above: overcoverage. Bands are pointwise.")
    fig.savefig(args.outdir / "coverage_marginals_and_joint.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    bins = np.linspace(0, 1, 9)
    for j, (ax, name) in enumerate(zip(axes.flat, runner.PARAMETER_NAMES)):
        ax.hist(ranks[:, j], bins=bins, weights=observation_weights / np.diff(bins)[0],
                color="#1479b8", alpha=.8)
        ax.axhline(1, color="black", ls="--", lw=1)
        ax.set(title=name, xlabel="Truth's posterior CDF rank", ylabel="Prior-weighted density")
        ax.grid(alpha=.2)
    axes.flat[-1].bar(np.arange(len(points)), weights, color="#1479b8")
    axes.flat[-1].set(title="Importance weights by held-out point", xlabel="Point position",
                      ylabel="Normalized weight")
    fig.suptitle("Marginal ranks: edge excess can indicate narrow intervals; asymmetry can indicate bias.\n"
                 "Interpret with the coverage bands and limited number of distinct parameter points.")
    fig.savefig(args.outdir / "calibration_ranks_and_weights.png", dpi=180)
    plt.close(fig)
    max_shift = max(record["repeat_shift"] for record in results)
    notes = dict(noise_audit=audit, samples_per_network=args.samples_per_network,
                 max_repeat_mean_shift_sigma=max_shift,
                 heldout_parameters_are_guided=True, importance_weights="p(u)/q(u), assessment only",
                 bootstrap="campaign-stratified point bootstrap, widened by an effective-point Wilson approximation; approximate 90% pointwise bands",
                 interpretation="Undercoverage may reflect bias and/or narrow posteriors. Limited design/weight ESS and MCMC error prevent automatic certification.",
                 references="TARP: Lemos et al., 2023, arXiv:2302.03026; SBC: Talts et al., 2018, arXiv:1805.09294")
    (args.outdir / "diagnostic_notes.json").write_text(json.dumps(notes, indent=2) + "\n")
    print("\nPrior-weighted coverage (approximate 90% point uncertainty):", flush=True)
    for level in (.68, .95):
        k = int(np.argmin(np.abs(LEVELS - level)))
        for j, name in enumerate(names):
            print(f"{name:20s} {level:.0%}: {weighted[k,j]:.3f} "
                  f"[{bands[0,k,j]:.3f}, {bands[1,k,j]:.3f}]", flush=True)
    print(f"Largest MCMC-repeat mean shift / posterior SD: {max_shift:.3f}", flush=True)
    print("Below diagonal means undercoverage (narrowness and/or bias); above means overcoverage.", flush=True)
    print("The diagonal applies to the prior-weighted estimate, not the unweighted guided-design recovery.", flush=True)
    print(f"Saved plots, coverage.csv, recovery.csv and checkpoints to {args.outdir}", flush=True)


def run_validation(args, prepared):
    runner, master, man, summary, train_mask, rows, weights, audit = prepared
    import torch
    import sbi
    from sbi.inference import SNLE
    from torch.distributions import Independent, Normal
    # The small density estimators need little CPU parallelism. Keep this
    # explicit rather than inheriting a large BLAS/OpenMP thread pool.
    torch.set_num_threads(1)
    log(f"torch={torch.__version__}; sbi={sbi.__version__}; device=cpu; "
        f"torch threads={torch.get_num_threads()}")
    arrays = master["arrays"]
    args.outdir.mkdir(parents=True, exist_ok=True)
    model_dir, draws_dir = args.outdir / "models", args.outdir / "mock_posteriors"
    model_dir.mkdir(exist_ok=True); draws_dir.mkdir(exist_ok=True)
    log(f"Completed-network checkpoints: {model_dir}")
    log(f"Completed-mock checkpoints: {draws_dir}")
    log("A checkpoint is saved after each network and each complete mock, not during a network's training.")
    progress(args, "preflight_complete", selected_mocks=int(len(rows)))
    prior = Independent(Normal(torch.zeros(5), torch.ones(5)), 1)
    train_u = torch.as_tensor(arrays["u"][train_mask], dtype=torch.float32)
    train_x = torch.as_tensor(summary[train_mask], dtype=torch.float32)
    fit_signature = dict(master=master["file_sha256"], reference=file_hash(args.reference),
                         runner=file_hash(args.runner), probe=args.probe, training=man["training"],
                         training_seeds=[int(r["training_seed"]) for r in man["per_network"]],
                         torch=str(torch.__version__), sbi=str(sbi.__version__),
                         cache_format="posterior_template.v1")
    mcmc = man["mcmc"]
    posterior_templates = []
    for network, record in enumerate(man["per_network"]):
        path = model_dir / f"network_{network}.pt"
        if path.exists():
            progress(args, "loading_network", network_index=network)
            # These are this script's own model checkpoints, not uploaded objects.
            try:
                cache = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:  # older Torch lacks weights_only
                cache = torch.load(path, map_location="cpu")
            require(cache["signature"] == fit_signature, f"Model checkpoint differs: {path}; use another --outdir")
            posterior = cache["posterior"]
            print(f"Loaded network {network}", flush=True)
        else:
            seed = int(record["training_seed"])
            torch.manual_seed(seed); np.random.seed(seed)
            inference = SNLE(prior=prior, density_estimator="nsf", device="cpu", show_progress_bars=True)
            inference.append_simulations(train_u, train_x)
            log(f"Training network {network} with recorded seed {seed}; settings={man['training']}")
            progress(args, "training", network_index=network, training_seed=seed)
            started = time.monotonic()
            estimator = inference.train(**man["training"], show_train_summary=True)
            log(f"Network {network} training finished in {time.monotonic() - started:.1f} seconds")
            estimator.eval()
            # Build on the trained inference object, preserving version-specific
            # metadata. Save the resulting picklable posterior with its network.
            posterior = inference.build_posterior(estimator, prior=prior, sample_with="mcmc",
                mcmc_method=mcmc["method"], mcmc_parameters={key: mcmc[key]
                for key in ("num_chains", "warmup_steps", "thin")})
            temporary = path.with_name(path.name + ".tmp")
            torch.save(dict(signature=fit_signature, posterior=posterior), temporary)
            os.replace(temporary, path)
            log(f"Saved {path}")
        posterior_templates.append(posterior)
        progress(args, "network_ready", completed_networks=len(posterior_templates))
    sample_signature = dict(fit=fit_signature, mcmc=man["mcmc"],
                            samples_per_network=args.samples_per_network, sampling_namespace=20261013)
    serialized_signature = json.dumps(sample_signature, sort_keys=True)
    per_repeat = args.samples_per_network // 2
    results = []
    for position, row in enumerate(rows):
        row = int(row)
        point = int(arrays["point_index"][row]); replica = int(arrays["replica"][row])
        path = draws_dir / f"point_{point:03d}_replica_{replica:02d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                require(archive["signature_json"].item() == serialized_signature,
                        f"Sample checkpoint differs: {path}; use another --outdir")
                require(int(archive["row"]) == row, f"Sample row differs: {path}")
                np.testing.assert_allclose(archive["observed_summary"], summary[row], rtol=1e-10, atol=1e-12,
                                           err_msg="cached observed summary differs")
                draws = np.asarray(archive["u_network_repeat"], float)
        else:
            x = torch.as_tensor(summary[row], dtype=torch.float32)
            networks = []
            for network, template in enumerate(posterior_templates):
                repetitions = []
                for repeat in range(2):
                    log(f"Mock {position + 1}/{len(rows)}: point {point}, replica {replica}; "
                        f"sampling network {network}, repeat {repeat}")
                    progress(args, "sampling", completed_mocks=position,
                             selected_mocks=int(len(rows)), point=point, replica=replica,
                             network_index=network, repeat_index=repeat)
                    seed = int(np.random.SeedSequence([20261013, int(man["per_network"][network]["training_seed"]),
                                                      point, replica, repeat]).generate_state(1)[0])
                    torch.manual_seed(seed); np.random.seed(seed)
                    # Fresh posterior per repeat avoids inheriting chain state from another mock.
                    posterior = copy.deepcopy(template)
                    repetitions.append(np.asarray(posterior.sample((per_repeat,), x=x,
                                                show_progress_bars=False), dtype=float))
                networks.append(np.stack(repetitions))
            draws = np.stack(networks)
            require(draws.shape == (4, 2, per_repeat, 5) and np.isfinite(draws).all(), "Invalid MCMC samples")
            atomic_npz(path, u_network_repeat=draws, row=np.asarray(row), point=np.asarray(point),
                       replica=np.asarray(replica), observed_summary=summary[row],
                       truth_u=arrays["point_u"][point], signature_json=np.asarray(serialized_signature))
        require(draws.shape == (4, 2, per_repeat, 5) and np.isfinite(draws).all(), "Invalid cached MCMC samples")
        means = draws.mean(axis=2)
        sd = draws.reshape(4, -1, 5).std(axis=1, ddof=1)
        require(np.all(sd > 0), "Degenerate MCMC sample variance")
        shift = float(np.max(np.abs(means[:, 0] - means[:, 1]) / sd))
        pooled = draws.reshape(-1, 5)  # equal samples per network => equal posterior mixture weights
        ranks, central, ellipsoid, tarp, quantiles = assess_draws(pooled, arrays["point_u"][point],
                                                                 20261013 + row, args.tarp_references)
        results.append(dict(ranks=ranks, central=central, ellipsoid=ellipsoid,
                            tarp=tarp, quantiles=quantiles, repeat_shift=shift))
        print(f"[{position+1}/{len(rows)}] point {point}, replica {replica}; repeat shift={shift:.3f}", flush=True)
        progress(args, "mock_complete", completed_mocks=position + 1, selected_mocks=int(len(rows)))
    progress(args, "reporting", completed_mocks=len(results))
    report(args, runner, master, rows, weights, results, audit)
    progress(args, "complete", completed_mocks=len(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", default="gy_g2y")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--master", type=Path)
    parser.add_argument("--runner", type=Path, default=Path(__file__).with_name("run_mock_sbi_all256_snle.py"))
    parser.add_argument("--reference", type=Path, help="Original four-network posterior NPZ; its manifest supplies training settings/seeds")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--samples-per-network", type=int, default=4000, help="Validation samples per network, split across two repeats")
    parser.add_argument("--replicas-per-point", type=int, default=1, help="Maximum noise-disjoint test replicas per point; 0 uses all eligible replicas")
    parser.add_argument("--tarp-references", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    require(args.samples_per_network >= 1000 and args.replicas_per_point >= 0
            and args.tarp_references >= 1 and args.bootstrap >= 200, "Invalid sample/test/bootstrap counts")
    args.outdir = args.outdir or args.base / "coverage_checks" / args.probe
    prepared = prepare(args)
    if args.preflight:
        print(f"PREFLIGHT PASS; output directory: {args.outdir}; no files written", flush=True)
        return 0
    run_validation(args, prepared)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

