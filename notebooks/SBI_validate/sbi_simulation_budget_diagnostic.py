"""Diagnose SBI simulation-budget choices for the analytical Cl validation.

The linearized two-parameter validation problem has an exact posterior once the
score-compressed Fisher matrix is known.  This script uses that exact posterior
as the reference, compares existing SNPE runs to it, and estimates how many
simulations are needed by a score-compressed Gaussian synthetic likelihood that
uses simulations only to calibrate the 2x2 summary covariance.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from dataclasses import dataclass

import numpy as np


THIS_DIR = pathlib.Path(__file__).resolve().parent
THEORY_SBI_DIR = THIS_DIR / "outputs" / "theory_sbi"
DEFAULT_REFERENCE_RUN = "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn5"
DEFAULT_HMC_RUN = "joint_gg_gy_gtau_gkappa_linearized"


@dataclass(frozen=True)
class PosteriorMoments:
    mean: np.ndarray
    cov: np.ndarray
    constrained_vec: np.ndarray
    degenerate_vec: np.ndarray

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def corr(self) -> float:
        return float(self.cov[0, 1] / np.sqrt(self.cov[0, 0] * self.cov[1, 1]))


def posterior_grid_moments(
    cov_summary: np.ndarray,
    fiducial: np.ndarray,
    prior_min: np.ndarray,
    prior_max: np.ndarray,
    ngrid: int,
) -> PosteriorMoments:
    """Return exact box-prior posterior moments for x_obs=theta_fiducial."""

    theta0 = np.linspace(prior_min[0], prior_max[0], ngrid)
    theta1 = np.linspace(prior_min[1], prior_max[1], ngrid)
    grid0, grid1 = np.meshgrid(theta0, theta1, indexing="ij")
    d0 = grid0 - fiducial[0]
    d1 = grid1 - fiducial[1]
    precision = np.linalg.inv(cov_summary)
    chi2 = precision[0, 0] * d0 * d0 + 2.0 * precision[0, 1] * d0 * d1 + precision[1, 1] * d1 * d1
    logw = -0.5 * chi2
    logw -= np.max(logw)
    weight = np.exp(logw)
    norm = np.sum(weight)
    mean = np.array([
        np.sum(weight * grid0) / norm,
        np.sum(weight * grid1) / norm,
    ])
    var0 = np.sum(weight * (grid0 - mean[0]) ** 2) / norm
    var1 = np.sum(weight * (grid1 - mean[1]) ** 2) / norm
    cov01 = np.sum(weight * (grid0 - mean[0]) * (grid1 - mean[1])) / norm
    cov = np.array([[var0, cov01], [cov01, var1]])
    evals, evecs = np.linalg.eigh(cov)
    constrained_vec = evecs[:, 0]
    degenerate_vec = evecs[:, 1]
    if constrained_vec[1] > 0:
        constrained_vec = -constrained_vec
    if degenerate_vec[0] < 0:
        degenerate_vec = -degenerate_vec
    return PosteriorMoments(mean, cov, constrained_vec, degenerate_vec)


def sample_moments(samples: np.ndarray, reference: PosteriorMoments) -> dict[str, float]:
    samples = np.asarray(samples, dtype=float)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    cov = np.cov(samples.T)
    fiducial = np.array([2.0, -0.1])
    q = (samples - fiducial[None, :]) @ reference.constrained_vec
    d = (samples - fiducial[None, :]) @ reference.degenerate_vec
    q_ref_mean = (reference.mean - fiducial) @ reference.constrained_vec
    d_ref_mean = (reference.mean - fiducial) @ reference.degenerate_vec
    q_ref_std = float(np.sqrt(reference.constrained_vec @ reference.cov @ reference.constrained_vec))
    d_ref_std = float(np.sqrt(reference.degenerate_vec @ reference.cov @ reference.degenerate_vec))
    return {
        "theta_shift_grid_sigma": float((mean[0] - reference.mean[0]) / reference.std[0]),
        "nu_shift_grid_sigma": float((mean[1] - reference.mean[1]) / reference.std[1]),
        "theta_width_grid_ratio": float(std[0] / reference.std[0]),
        "nu_width_grid_ratio": float(std[1] / reference.std[1]),
        "q_shift_grid_sigma": float((np.mean(q) - q_ref_mean) / q_ref_std),
        "q_width_grid_ratio": float(np.std(q) / q_ref_std),
        "d_shift_grid_sigma": float((np.mean(d) - d_ref_mean) / d_ref_std),
        "d_width_grid_ratio": float(np.std(d) / d_ref_std),
        "corr": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
    }


def load_hmc_samples(path: pathlib.Path) -> np.ndarray:
    data = np.load(path)
    return np.column_stack([data["samples_theta_ej_0"], data["samples_nu_theta_ej_M"]])


def load_sbi_samples(path: pathlib.Path, prior_min: np.ndarray, prior_max: np.ndarray) -> np.ndarray:
    samples = np.load(path, allow_pickle=True)["samples"]
    inside = np.all((samples >= prior_min[None, :]) & (samples <= prior_max[None, :]), axis=1)
    return samples[inside]


def compare_existing_runs(
    base_dir: pathlib.Path,
    reference: PosteriorMoments,
    prior_min: np.ndarray,
    prior_max: np.ndarray,
) -> list[dict[str, object]]:
    runs = [
        ("HMC raw", DEFAULT_HMC_RUN, "hmc"),
        ("HMC strict", "joint_gg_gy_gtau_gkappa_linearized_hmc_strict", "hmc"),
        ("Score t 128", "compressed_likelihood_student_t_nsim128", "compressed"),
        ("Score Gaussian 128", "compressed_likelihood_gaussian_nsim128", "compressed"),
        ("Score t 256", "compressed_likelihood_student_t_nsim256", "compressed"),
        ("Score Gaussian 256", "compressed_likelihood_gaussian_nsim256", "compressed"),
        ("MDN1 2k", "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn1_512_512_1024", "sbi"),
        ("MDN1 4k", "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn1_1024_1024_2048", "sbi"),
        ("MDN5 2k", "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn5_512_512_1024", "sbi"),
        ("MDN5 4k", "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn5_1024_1024_2048", "sbi"),
        ("MDN5 8k", "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn5_2048_2048_4096", "sbi"),
        ("MDN5 8x512", "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn5_8x512", "sbi"),
        ("MDN5 16k", "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn5", "sbi"),
        ("MDN5 32k", "joint_gg_gy_gtau_gkappa_linearized_fisher_mdn5_2x", "sbi"),
        ("NSF 8x1024", "joint_gg_gy_gtau_gkappa_linearized_fisher_nsf64_8x1024", "sbi"),
        ("NSF 16k", "joint_gg_gy_gtau_gkappa_linearized_fisher_nsf64", "sbi"),
    ]
    rows: list[dict[str, object]] = []
    for label, run_name, kind in runs:
        if kind == "hmc":
            filename = "hmc_samples.npz"
            diagnostics_filename = "hmc_diagnostics.json"
        elif kind == "compressed":
            filename = "compressed_likelihood_samples.npz"
            diagnostics_filename = "compressed_likelihood_diagnostics.json"
        else:
            filename = "sbi_posterior_samples.npz"
            diagnostics_filename = "sbi_diagnostics.json"
        path = base_dir / run_name / filename
        if not path.exists():
            continue
        if kind == "hmc":
            samples = load_hmc_samples(path)
        elif kind == "compressed":
            samples = np.load(path, allow_pickle=True)["samples"]
        else:
            samples = load_sbi_samples(path, prior_min, prior_max)
        diagnostics_path = base_dir / run_name / diagnostics_filename
        diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
        sims = diagnostics.get("simulations_per_round")
        if sims is None:
            sims = diagnostics.get("nsim")
        row: dict[str, object] = {
            "label": label,
            "run_name": run_name,
            "kind": kind,
            "total_sims": "" if sims is None else int(np.sum(sims)),
            "runtime_sec": diagnostics.get("runtime_sec", ""),
        }
        row.update(sample_moments(samples, reference))
        rows.append(row)
    return rows


def covariance_budget_experiment(
    cov_summary: np.ndarray,
    fiducial: np.ndarray,
    prior_min: np.ndarray,
    prior_max: np.ndarray,
    reference: PosteriorMoments,
    ngrid: int,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    q_ref_std = float(np.sqrt(reference.constrained_vec @ reference.cov @ reference.constrained_vec))
    d_ref_std = float(np.sqrt(reference.degenerate_vec @ reference.cov @ reference.degenerate_vec))
    rows = []
    for nsim in (16, 32, 64, 128, 256, 512, 1024):
        trials = []
        for _ in range(80):
            summaries = rng.multivariate_normal(fiducial, cov_summary, size=nsim)
            cov_hat = np.cov(summaries.T, ddof=1)
            moments = posterior_grid_moments(cov_hat, fiducial, prior_min, prior_max, ngrid)
            q_std = float(np.sqrt(reference.constrained_vec @ moments.cov @ reference.constrained_vec))
            d_std = float(np.sqrt(reference.degenerate_vec @ moments.cov @ reference.degenerate_vec))
            trials.append([
                np.max(np.abs((moments.mean - reference.mean) / reference.std)),
                np.max(np.abs(moments.std / reference.std - 1.0)),
                abs(q_std / q_ref_std - 1.0),
                abs(d_std / d_ref_std - 1.0),
            ])
        trials_np = np.asarray(trials)
        rows.append({
            "nsim": nsim,
            "median_abs_mean_shift_sigma": float(np.median(trials_np[:, 0])),
            "median_abs_width_error": float(np.median(trials_np[:, 1])),
            "median_abs_q_width_error": float(np.median(trials_np[:, 2])),
            "p90_abs_q_width_error": float(np.percentile(trials_np[:, 2], 90)),
            "median_abs_d_width_error": float(np.median(trials_np[:, 3])),
        })
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: pathlib.Path,
    reference: PosteriorMoments,
    comparison_rows: list[dict[str, object]],
    budget_rows: list[dict[str, float]],
) -> None:
    lines = [
        "# SBI Simulation Budget Diagnostic",
        "",
        "Reference is the exact box-prior posterior for the score-compressed linearized likelihood.",
        "",
        f"- reference mean: `{reference.mean.tolist()}`",
        f"- reference std: `{reference.std.tolist()}`",
        f"- reference corr: `{reference.corr:.6f}`",
        "",
        "## Existing Runs",
        "",
        "| run | sims | theta shift | nu shift | theta width | nu width | q width | d width |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            "| {label} | {total_sims} | {theta_shift_grid_sigma:.3f} | {nu_shift_grid_sigma:.3f} | "
            "{theta_width_grid_ratio:.3f} | {nu_width_grid_ratio:.3f} | "
            "{q_width_grid_ratio:.3f} | {d_width_grid_ratio:.3f} |".format(**row)
        )
    lines.extend([
        "",
        "## Score-Gaussian Covariance Budget",
        "",
        "| simulations | median mean shift | median width err | median q width err | p90 q width err | median d width err |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in budget_rows:
        lines.append(
            "| {nsim} | {median_abs_mean_shift_sigma:.3f} | {median_abs_width_error:.3f} | "
            "{median_abs_q_width_error:.3f} | {p90_abs_q_width_error:.3f} | "
            "{median_abs_d_width_error:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=pathlib.Path, default=THEORY_SBI_DIR)
    parser.add_argument("--reference-run", default=DEFAULT_REFERENCE_RUN)
    parser.add_argument("--output-dir", type=pathlib.Path, default=THEORY_SBI_DIR / "sbi_simulation_budget_diagnostics")
    parser.add_argument("--ngrid", type=int, default=800)
    parser.add_argument("--cov-ngrid", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    reference_dir = args.base_dir / args.reference_run
    fisher = np.load(reference_dir / "sbi_fisher.npy")
    cov_summary = np.linalg.inv(fisher)
    fiducial = np.array([2.0, -0.1])
    prior_min = np.array([0.5, -1.0])
    prior_max = np.array([4.0, 0.0])

    reference = posterior_grid_moments(cov_summary, fiducial, prior_min, prior_max, args.ngrid)
    comparison_rows = compare_existing_runs(args.base_dir, reference, prior_min, prior_max)
    budget_rows = covariance_budget_experiment(
        cov_summary,
        fiducial,
        prior_min,
        prior_max,
        reference,
        ngrid=args.cov_ngrid,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "existing_run_comparison.csv", comparison_rows)
    write_csv(args.output_dir / "score_gaussian_covariance_budget.csv", budget_rows)
    write_markdown(args.output_dir / "summary.md", reference, comparison_rows, budget_rows)
    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
