"""Create comparison notebook and figures for pasted-map compressed likelihood."""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import numpy as np


THIS_DIR = pathlib.Path(__file__).resolve().parent
THEORY_SBI_DIR = THIS_DIR / "outputs" / "theory_sbi"
STRICT_HMC = THEORY_SBI_DIR / "joint_gg_gy_gtau_gkappa_linearized_hmc_strict" / "hmc_samples.npz"
ANALYTIC_COMPRESSED = THEORY_SBI_DIR / "compressed_likelihood_student_t_nsim256" / "compressed_likelihood_samples.npz"
FIDUCIAL_PRODUCT = THEORY_SBI_DIR / "fiducial_thetaej2_nuejm_minus0p1.npz"


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.split("\n")]}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")],
    }


def load_hmc(path: pathlib.Path) -> np.ndarray:
    data = np.load(path)
    return np.column_stack([data["samples_theta_ej_0"], data["samples_nu_theta_ej_M"]])


def summarize(label: str, samples: np.ndarray, ref: np.ndarray, fid: np.ndarray, nsim: object, like: str) -> dict[str, object]:
    ref_cov = np.cov(ref.T)
    evals, evecs = np.linalg.eigh(ref_cov)
    constrained = evecs[:, 0]
    if constrained[1] > 0:
        constrained = -constrained
    q_ref = (ref - fid[None, :]) @ constrained
    q = (samples - fid[None, :]) @ constrained
    return {
        "method": label,
        "simulations": str(nsim),
        "likelihood": like,
        "theta shift / HMC sigma": float((samples[:, 0].mean() - ref[:, 0].mean()) / ref[:, 0].std()),
        "nu shift / HMC sigma": float((samples[:, 1].mean() - ref[:, 1].mean()) / ref[:, 1].std()),
        "theta width / HMC": float(samples[:, 0].std() / ref[:, 0].std()),
        "nu width / HMC": float(samples[:, 1].std() / ref[:, 1].std()),
        "q width / HMC": float(q.std() / q_ref.std()),
    }


def make_figures(input_dir: pathlib.Path) -> tuple[list[dict[str, object]], pathlib.Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots

    fig_dir = input_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    hmc = load_hmc(STRICT_HMC)
    analytic = np.load(ANALYTIC_COMPRESSED, allow_pickle=True)["samples"]
    map_npz = np.load(input_dir / "map_sbi_compressed_samples.npz", allow_pickle=True)
    map_student = map_npz["samples_student_t"]
    map_fd_student = map_npz["samples_student_t_finite_difference"] if "samples_student_t_finite_difference" in map_npz else None
    diag = json.loads((input_dir / "map_sbi_diagnostics.json").read_text())
    main_diag = diag["analyses"][0]["diagnostics"]
    score_diags = main_diag.get("score_methods", {})
    fid = np.asarray(map_npz["theta_fiducial"], dtype=float)
    analytic_nsim = score_diags.get("analytic", main_diag).get("nsim", main_diag.get("nsim", 256))
    fd_diag = score_diags.get("finite_difference", {})

    rows = [
        summarize("Strict HMC", hmc, hmc, fid, 0, "exact Gaussian"),
        summarize("Analytical compressed", analytic, hmc, fid, 256, "Student-t"),
        summarize("Map SBI theory-score", map_student, hmc, fid, analytic_nsim, "Student-t"),
    ]
    if map_fd_student is not None:
        rows.append(
            summarize(
                "Map SBI finite-difference-score",
                map_fd_student,
                hmc,
                fid,
                f"{fd_diag.get('nsim', analytic_nsim)} + {fd_diag.get('n_finite_difference_measurements', 0)} FD",
                "Student-t",
            )
        )

    names = ["theta_ej_0", "nu_theta_ej_M"]
    labels = [r"\theta_{\rm ej,0}", r"\nu^M_{\theta_{\rm ej}}"]
    mc = [
        MCSamples(samples=hmc, names=names, labels=labels, label="Strict HMC", settings={"ignore_rows": 0.0}),
        MCSamples(samples=analytic, names=names, labels=labels, label="Analytical Student-t 256", settings={"ignore_rows": 0.0}),
        MCSamples(samples=map_student, names=names, labels=labels, label="Map theory-score Student-t", settings={"ignore_rows": 0.0}),
    ]
    colors = ["#1f77b4", "black", "#d62728"]
    filled = [False, False, True]
    if map_fd_student is not None:
        mc.append(
            MCSamples(
                samples=map_fd_student,
                names=names,
                labels=labels,
                label="Map finite-difference-score Student-t",
                settings={"ignore_rows": 0.0},
            )
        )
        colors.append("#2ca02c")
        filled.append(False)
    g = plots.get_subplot_plotter(width_inch=7.0)
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.18
    g.triangle_plot(
        mc,
        filled=filled,
        contour_colors=colors,
    )
    g.export(str(fig_dir / "map_sbi_triangle.png"))
    plt.close("all")

    ref_cov = np.cov(hmc.T)
    evals, evecs = np.linalg.eigh(ref_cov)
    constrained = evecs[:, 0]
    if constrained[1] > 0:
        constrained = -constrained
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    q_inputs = [
        ("Strict HMC", hmc, "#1f77b4"),
        ("Analytical Student-t 256", analytic, "black"),
        ("Map theory-score Student-t", map_student, "#d62728"),
    ]
    if map_fd_student is not None:
        q_inputs.append(("Map finite-difference-score Student-t", map_fd_student, "#2ca02c"))
    for label, samples, color in q_inputs:
        q = (samples - fid[None, :]) @ constrained
        ax.hist(q, bins=70, density=True, histtype="step", lw=1.7, color=color, label=label)
    ax.set_xlabel(r"$q = (\theta-\theta_{\rm fid})\cdot e_{\rm constrained}$")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.savefig(fig_dir / "map_sbi_constrained_direction.png", dpi=180)
    plt.close(fig)

    cls = np.load(input_dir / "map_sbi_cls_ensemble.npz", allow_pickle=True)
    fid_product = np.load(FIDUCIAL_PRODUCT, allow_pickle=True)
    ell = np.asarray(cls["ell"], dtype=float)
    fid_ell = np.asarray(fid_product["ell"], dtype=float)
    fid_idx = np.array([int(np.argmin(np.abs(fid_ell - val))) for val in ell], dtype=int)
    probes = [str(p) for p in cls["probes"]]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    axes = axes.ravel()
    for ax, probe in zip(axes, probes):
        mean = np.asarray(cls[f"cl_{probe}_mean"], dtype=float)
        p16 = np.asarray(cls[f"cl_{probe}_p16"], dtype=float)
        p84 = np.asarray(cls[f"cl_{probe}_p84"], dtype=float)
        theory = np.asarray(fid_product[f"cl_{probe}"], dtype=float)[fid_idx]
        ax.loglog(ell, np.abs(theory), color="black", lw=1.5, label="theory abs")
        ax.loglog(ell, np.abs(mean), color="#d62728", lw=1.4, label="map mean abs")
        ax.fill_between(ell, np.abs(p16), np.abs(p84), color="#d62728", alpha=0.20, linewidth=0)
        ax.set_title(probe)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$|C_\ell|$")
        ax.legend(fontsize=8)
    for ax in axes[len(probes):]:
        ax.axis("off")
    fig.savefig(fig_dir / "map_sbi_cls_ensemble.png", dpi=180)
    plt.close(fig)

    n_cov = 3 if "summary_u_cov_finite_difference" in map_npz else 2
    fig, axes = plt.subplots(1, n_cov, figsize=(4.8 * n_cov, 4.0), constrained_layout=True)
    im0 = axes[0].imshow(np.log10(np.abs(map_npz["summary_u_cov"]) + 1.0e-300), origin="lower", cmap="magma")
    axes[0].set_title(r"theory-score $\log_{10}|{\rm Cov}(u)|$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(map_npz["summary_u_corr"], origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title(r"theory-score ${\rm Corr}(u)$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    if n_cov == 3:
        im2 = axes[2].imshow(
            map_npz["summary_u_corr_finite_difference"],
            origin="lower",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        axes[2].set_title(r"finite-difference-score ${\rm Corr}(u)$")
        fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.savefig(fig_dir / "map_sbi_compressed_covariance.png", dpi=180)
    plt.close(fig)
    return rows, fig_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=THIS_DIR / "06_compare_map_sbi_compressed_likelihood.ipynb")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()
    rows, fig_dir = make_figures(input_dir)
    table = [
        "| method | simulations | likelihood | theta shift | nu shift | theta width | nu width | q width |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['method']} | {row['simulations']} | {row['likelihood']} | "
            f"{row['theta shift / HMC sigma']:.3f} | {row['nu shift / HMC sigma']:.3f} | "
            f"{row['theta width / HMC']:.3f} | {row['nu width / HMC']:.3f} | {row['q width / HMC']:.3f} |"
        )
    rel_fig = pathlib.Path(os.path.relpath(fig_dir, output_path.parent))
    cells = [
        md_cell("# Pasted-Map Compressed-Likelihood SBI\n\n" + "\n".join(table)),
        md_cell(f"## Posterior Triangle\n\n![triangle]({rel_fig}/map_sbi_triangle.png)"),
        md_cell(f"## Constrained Direction\n\n![q]({rel_fig}/map_sbi_constrained_direction.png)"),
        md_cell(f"## Map Cl Ensemble\n\n![cls]({rel_fig}/map_sbi_cls_ensemble.png)"),
        md_cell(f"## Compressed Covariance\n\n![cov]({rel_fig}/map_sbi_compressed_covariance.png)"),
        code_cell(
            "from pathlib import Path\n"
            "import json, numpy as np\n"
            f"INPUT_DIR = Path('{input_dir}')\n"
            "diag = json.loads((INPUT_DIR / 'map_sbi_diagnostics.json').read_text())\n"
            "print('analysis order:', diag['analysis_order'])\n"
            "print('score methods:', diag.get('score_methods'))\n"
            "print('main diagnostics:')\n"
            "print(json.dumps(diag['analyses'][0]['diagnostics'], indent=2)[:4000])\n"
        ),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(nb, indent=1))
    print(f"Wrote notebook to {output_path}")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
