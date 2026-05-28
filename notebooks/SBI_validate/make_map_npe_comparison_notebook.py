"""Create the pasted-map active-NPE comparison notebook and figures."""

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
    z = np.load(path)
    return np.column_stack([z["samples_theta_ej_0"], z["samples_nu_theta_ej_M"]])


def make_figures(input_dir: pathlib.Path) -> tuple[list[dict[str, object]], pathlib.Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from getdist import MCSamples, plots

    fig_dir = input_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    hmc = load_hmc(STRICT_HMC)
    analytic = np.load(ANALYTIC_COMPRESSED, allow_pickle=True)["samples"]
    map_npe_file = np.load(input_dir / "map_npe_posterior_samples.npz", allow_pickle=True)
    map_npe = map_npe_file["samples"]
    sims = np.load(input_dir / "map_npe_simulations.npz", allow_pickle=True)
    diag = json.loads((input_dir / "map_npe_diagnostics.json").read_text())
    fid = np.asarray(
        map_npe_file["theta_validation_truth"]
        if "theta_validation_truth" in map_npe_file.files
        else np.array([2.0, -0.1]),
        dtype=float,
    )
    if not np.all(np.isfinite(fid)):
        fid = np.array([2.0, -0.1])
    ref_cov = np.cov(hmc.T)
    evals, evecs = np.linalg.eigh(ref_cov)
    constrained_vec = evecs[:, 0]
    degenerate_vec = evecs[:, 1]
    if constrained_vec[1] > 0:
        constrained_vec = -constrained_vec
    if degenerate_vec[0] < 0:
        degenerate_vec = -degenerate_vec
    q_hmc = (hmc - fid[None, :]) @ constrained_vec
    d_hmc = (hmc - fid[None, :]) @ degenerate_vec

    rows = []
    nmap = int(diag.get("n_valid_simulations", diag["nsim_total"]))
    for label, samples, nsim in [
        ("Strict HMC", hmc, 0),
        ("Analytical compressed Student-t", analytic, 256),
        ("Pasted-map active NPE", map_npe, nmap),
    ]:
        q = (samples - fid[None, :]) @ constrained_vec
        d = (samples - fid[None, :]) @ degenerate_vec
        rows.append({
            "method": label,
            "total simulations": nsim,
            "theta mean": float(samples[:, 0].mean()),
            "nu mean": float(samples[:, 1].mean()),
            "theta shift / HMC sigma": float((samples[:, 0].mean() - hmc[:, 0].mean()) / hmc[:, 0].std()),
            "nu shift / HMC sigma": float((samples[:, 1].mean() - hmc[:, 1].mean()) / hmc[:, 1].std()),
            "theta width / HMC": float(samples[:, 0].std() / hmc[:, 0].std()),
            "nu width / HMC": float(samples[:, 1].std() / hmc[:, 1].std()),
            "q width / HMC": float(q.std() / q_hmc.std()),
            "d width / HMC": float(d.std() / d_hmc.std()),
        })

    names = ["theta_ej_0", "nu_theta_ej_M"]
    labels = [r"\theta_{\rm ej,0}", r"\nu^M_{\theta_{\rm ej}}"]
    mc = [
        MCSamples(samples=hmc, names=names, labels=labels, label="Strict HMC", settings={"ignore_rows": 0.0}),
        MCSamples(samples=analytic, names=names, labels=labels, label="Analytical compressed 256", settings={"ignore_rows": 0.0}),
        MCSamples(samples=map_npe, names=names, labels=labels, label="Pasted-map active NPE 256", settings={"ignore_rows": 0.0}),
    ]
    g = plots.get_subplot_plotter(width_inch=7.0)
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.20
    g.triangle_plot(
        mc,
        filled=[False, False, True],
        contour_colors=["#1f77b4", "black", "#d62728"],
        legend_labels=["Strict HMC", "Analytical compressed 256", "Pasted-map active NPE 256"],
    )
    g.export(str(fig_dir / "map_npe_triangle.png"))
    plt.close("all")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    bins = np.linspace(np.percentile(q_hmc, 0.3), np.percentile(q_hmc, 99.7), 65)
    axes[0].hist(q_hmc, bins=bins, density=True, histtype="step", lw=2.0, color="#1f77b4", label="Strict HMC")
    for label, samples, color in [
        ("Analytical compressed 256", analytic, "black"),
        ("Pasted-map active NPE 256", map_npe, "#d62728"),
    ]:
        q = (samples - fid[None, :]) @ constrained_vec
        axes[0].hist(q, bins=bins, density=True, histtype="step", lw=1.5, label=label, color=color)
    axes[0].set_xlabel(r"$q = (\theta-\theta_{\rm fid})\cdot e_{\rm constrained}$")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Most constrained direction")
    rng = np.random.default_rng(13)
    for label, samples, color, alpha in [
        ("Strict HMC", hmc, "#1f77b4", 0.18),
        ("Pasted-map active NPE 256", map_npe, "#d62728", 0.12),
    ]:
        q = (samples - fid[None, :]) @ constrained_vec
        keep = rng.choice(len(samples), size=min(5000, len(samples)), replace=False)
        axes[1].scatter(samples[keep, 0], q[keep], s=3, alpha=alpha, color=color, label=label)
    axes[1].axhline(0.0, color="black", ls="--", lw=0.8)
    axes[1].axvline(fid[0], color="black", ls="--", lw=0.8)
    axes[1].set_xlabel(r"$\theta_{\rm ej,0}$")
    axes[1].set_ylabel(r"$q$ constrained")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Thin direction")
    fig.savefig(fig_dir / "map_npe_constrained_direction.png", dpi=180)
    plt.close(fig)

    fid_product = np.load(FIDUCIAL_PRODUCT, allow_pickle=True)
    ell = np.asarray(sims["ell"], dtype=float) if "ell" in sims.files else np.asarray(fid_product["ell"], dtype=float)
    fid_ell = np.asarray(fid_product["ell"], dtype=float)
    fid_ell_idx = np.array([int(np.argmin(np.abs(fid_ell - val))) for val in ell], dtype=int)
    spectra_order = [str(x) for x in fid_product["spectra_order"]]
    nell = len(ell)
    data_vectors = np.asarray(sims["data_vector"], dtype=float)
    probe_order = [str(spec) for spec in diag.get("probes", spectra_order)]
    plotted_specs = [spec for spec in probe_order if spec in ("gg", "gy", "gtau", "gkappa")]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    axes = axes.ravel()
    for ax, spec in zip(axes, plotted_specs):
        block = probe_order.index(spec)
        sl = slice(block * nell, (block + 1) * nell)
        cl_sim = data_vectors[:, sl]
        abs_cl_sim = np.abs(cl_sim)
        mean = np.nanmean(abs_cl_sim, axis=0)
        p16, p84 = np.nanpercentile(abs_cl_sim, [16, 84], axis=0)
        theory = np.asarray(fid_product[f"cl_{spec}"], dtype=float)[fid_ell_idx]
        ax.loglog(ell, np.abs(theory), color="black", lw=1.5, label="theory abs")
        ax.loglog(ell, mean, color="#d62728", lw=1.3, label="map mean abs")
        ax.fill_between(ell, p16, p84, color="#d62728", alpha=0.20, linewidth=0)
        ax.set_title(spec)
        ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$|C_\ell|$")
        ax.legend(fontsize=8)
    for ax in axes[len(plotted_specs):]:
        ax.axis("off")
    fig.savefig(fig_dir / "map_npe_cls_ensemble.png", dpi=180)
    plt.close(fig)
    return rows, fig_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=THIS_DIR / "05_compare_map_npe_to_hmc.ipynb")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()

    rows, fig_dir = make_figures(input_dir)
    table = [
        "| method | total simulations | theta shift / HMC sigma | nu shift / HMC sigma | theta width / HMC | nu width / HMC | q width / HMC | d width / HMC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['method']} | {row['total simulations']} | "
            f"{row['theta shift / HMC sigma']:.3f} | {row['nu shift / HMC sigma']:.3f} | "
            f"{row['theta width / HMC']:.3f} | {row['nu width / HMC']:.3f} | "
            f"{row['q width / HMC']:.3f} | {row['d width / HMC']:.3f} |"
        )
    rel_fig = pathlib.Path(os.path.relpath(input_dir / "figures", output_path.parent))
    cells = [
        md_cell("# Pasted-Map Active NPE vs Analytical HMC\n\nThis notebook compares the simulation-efficient pasted-map active NPE posterior against the strict analytical HMC benchmark. The validation truth is used only for plotting diagnostics, not by the NPE training code."),
        md_cell("## Posterior Summary\n\n" + "\n".join(table)),
        md_cell(f"## Posterior Triangle\n\n![triangle]({rel_fig}/map_npe_triangle.png)"),
        md_cell(f"## Constrained Direction\n\n![q]({rel_fig}/map_npe_constrained_direction.png)"),
        md_cell(f"## Map Cl Ensemble\n\n![cls]({rel_fig}/map_npe_cls_ensemble.png)"),
        code_cell(
            "from pathlib import Path\n"
            "import json, numpy as np\n"
            f"INPUT_DIR = Path('{input_dir}')\n"
            "samples = np.load(INPUT_DIR / 'map_npe_posterior_samples.npz')\n"
            "diag = json.loads((INPUT_DIR / 'map_npe_diagnostics.json').read_text())\n"
            "print('total simulations:', diag['nsim_total'])\n"
            "print('valid simulations:', diag.get('n_valid_simulations', diag['nsim_total']))\n"
            "print('rounds:', diag['rounds'])\n"
            "print('posterior mean:', samples['samples'].mean(axis=0))\n"
            "print('posterior std:', samples['samples'].std(axis=0))\n"
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
