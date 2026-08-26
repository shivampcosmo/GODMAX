"""Compare strict HMC to score-compressed Gaussian/Student-t likelihood runs."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np


THIS_DIR = pathlib.Path(__file__).resolve().parent
THEORY_SBI_DIR = THIS_DIR / "outputs" / "theory_sbi"
DEFAULT_HMC_RUN = "joint_gg_gy_gtau_gkappa_linearized_hmc_strict"
DEFAULT_RUNS = (
    "compressed_likelihood_student_t_nsim128",
    "compressed_likelihood_gaussian_nsim128",
    "compressed_likelihood_student_t_nsim256",
    "compressed_likelihood_gaussian_nsim256",
)


def load_hmc(path: pathlib.Path) -> np.ndarray:
    data = np.load(path)
    return np.column_stack([data["samples_theta_ej_0"], data["samples_nu_theta_ej_M"]])


def load_compressed(path: pathlib.Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)["samples"]


def comparison_row(label: str, samples: np.ndarray, ref: np.ndarray, fiducial: np.ndarray) -> dict[str, float | str]:
    ref_cov = np.cov(ref.T)
    evals, evecs = np.linalg.eigh(ref_cov)
    constrained_vec = evecs[:, 0]
    degenerate_vec = evecs[:, 1]
    if constrained_vec[1] > 0:
        constrained_vec = -constrained_vec
    if degenerate_vec[0] < 0:
        degenerate_vec = -degenerate_vec
    q_ref = (ref - fiducial[None, :]) @ constrained_vec
    d_ref = (ref - fiducial[None, :]) @ degenerate_vec
    q = (samples - fiducial[None, :]) @ constrained_vec
    d = (samples - fiducial[None, :]) @ degenerate_vec
    return {
        "run": label,
        "theta_shift_sigma": float((samples[:, 0].mean() - ref[:, 0].mean()) / ref[:, 0].std()),
        "nu_shift_sigma": float((samples[:, 1].mean() - ref[:, 1].mean()) / ref[:, 1].std()),
        "theta_width_ratio": float(samples[:, 0].std() / ref[:, 0].std()),
        "nu_width_ratio": float(samples[:, 1].std() / ref[:, 1].std()),
        "q_shift_sigma": float((q.mean() - q_ref.mean()) / q_ref.std()),
        "q_width_ratio": float(q.std() / q_ref.std()),
        "d_shift_sigma": float((d.mean() - d_ref.mean()) / d_ref.std()),
        "d_width_ratio": float(d.std() / d_ref.std()),
        "corr": float(np.corrcoef(samples.T)[0, 1]),
    }


def write_markdown(path: pathlib.Path, rows: list[dict[str, float | str]], hmc_info: dict[str, object]) -> None:
    lines = [
        "# Strict HMC vs Compressed Likelihood",
        "",
        f"- HMC run: `{hmc_info['run']}`",
        f"- HMC divergences: `{hmc_info['divergences']}`",
        f"- HMC max Rhat: `{hmc_info.get('max_rhat')}`",
        f"- HMC min ESS bulk: `{hmc_info.get('min_ess_bulk')}`",
        "",
        "| run | theta shift | nu shift | theta width | nu width | q shift | q width | d width | corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run} | {theta_shift_sigma:.3f} | {nu_shift_sigma:.3f} | "
            "{theta_width_ratio:.3f} | {nu_width_ratio:.3f} | "
            "{q_shift_sigma:.3f} | {q_width_ratio:.3f} | {d_width_ratio:.3f} | {corr:.5f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=pathlib.Path, default=THEORY_SBI_DIR)
    parser.add_argument("--hmc-run", default=DEFAULT_HMC_RUN)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--output-dir", type=pathlib.Path, default=THEORY_SBI_DIR / "strict_hmc_compressed_comparison")
    args = parser.parse_args()

    run_names = args.run or list(DEFAULT_RUNS)
    hmc_path = args.base_dir / args.hmc_run / "hmc_samples.npz"
    hmc = load_hmc(hmc_path)
    hmc_npz = np.load(hmc_path)
    diagnostics_path = args.base_dir / args.hmc_run / "hmc_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    hmc_info = {
        "run": args.hmc_run,
        "divergences": int(hmc_npz["extra_diverging"].sum()) if "extra_diverging" in hmc_npz else "",
        "max_rhat": diagnostics.get("max_rhat"),
        "min_ess_bulk": diagnostics.get("min_ess_bulk"),
    }
    fiducial = np.array([2.0, -0.1])
    rows = []
    for run_name in run_names:
        path = args.base_dir / run_name / "compressed_likelihood_samples.npz"
        if not path.exists():
            continue
        rows.append(comparison_row(run_name, load_compressed(path), hmc, fiducial))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.output_dir / "summary.md", rows, hmc_info)
    print(f"Wrote comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
