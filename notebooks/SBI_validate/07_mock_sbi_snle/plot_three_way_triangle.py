"""Triangle plot of the three posteriors on the same five gas parameters.

theory HMC   analytic-theory NUTS on the noiseless self-consistent observation
theory SBI   analytic-theory NPE, final round, same observation
mock SBI     NLE on the pasted-map simulator, noiseless pasted mock observation

Contours are 68 and 95 per cent of the 2-D marginal probability, taken from a smoothed
histogram of the samples rather than a KDE: getdist's automatic bandwidth collapses on a
chain with duplicate rows, which silently renders an empty panel.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

PARAMS = ["theta_ej_0", "alpha_nt", "mu_beta", "theta_co_0", "nu_theta_ej_M"]
LABELS = [r"$\theta_{\rm ej,0}$", r"$\alpha_{\rm nt}$", r"$\mu_\beta$",
          r"$\theta_{\rm co,0}$", r"$\nu_{\theta_{\rm ej},M}$"]
PRIOR_LOW = np.array([0.5, 0.0, 0.005, 0.001, -1.0])
PRIOR_HIGH = np.array([8.0, 0.5, 1.5, 0.5, 1.0])
TRUTH = np.array([2.0, 0.18, 0.6, 0.05, 0.0])

# dataviz reference palette, light surface, categorical slots 1/2/3 in fixed order.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedcd6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

LEVELS = (0.68, 0.95)
# The 2-D grid is deliberately coarse.  theory HMC has only 7200 samples, so a 96x96 grid
# holds under one sample per cell and the "contours" in a flat direction are pure Poisson
# speckle -- structure that is not in the posterior.  44x44 keeps ~4 per cell.
BINS_1D, SMOOTH_1D = 64, 2.4
BINS_2D, SMOOTH_2D = 44, 1.35


def contour_levels(h: np.ndarray) -> list[float]:
    """Density thresholds enclosing each fraction of the total probability."""
    flat = np.sort(h.ravel())[::-1]
    csum = np.cumsum(flat)
    csum /= csum[-1]
    out = []
    for frac in LEVELS:
        idx = int(np.searchsorted(csum, frac))
        out.append(float(flat[min(idx, flat.size - 1)]))
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=pathlib.Path,
                   default=pathlib.Path("/mnt/ceph/users/spandey/ltu-godmax/GODMAX"))
    p.add_argument("--mock", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--truth", action="store_true", help="mark the generating point")
    p.add_argument("--dpi", type=int, default=170)
    args = p.parse_args()

    inf = args.repo / "data/SBI_validate/three_probe_inference"
    hmc_raw = np.load(inf / "hmc_sc/run01/hmc_samples.npz")
    hmc = np.stack([np.asarray(hmc_raw[f"sample_{n}"]).reshape(-1) for n in PARAMS], axis=1)
    npe = np.asarray(np.load(inf / "sbi_sc/run01/posterior_samples_round_4.npz")["theta"])
    mock = np.asarray(np.load(args.mock, allow_pickle=True)["theta"])

    chains = [("theory HMC", hmc), ("theory SBI", npe), ("mock SBI", mock)]
    for name, c in chains:
        print(f"{name:<12} {c.shape[0]:>6d} samples")

    # A common range per parameter: the union of all three, clipped to the prior box.
    lo = np.empty(5)
    hi = np.empty(5)
    for j in range(5):
        q = np.concatenate([np.percentile(c[:, j], [0.3, 99.7]) for _, c in chains])
        pad = 0.06 * (q.max() - q.min())
        lo[j] = max(PRIOR_LOW[j], q.min() - pad)
        hi[j] = min(PRIOR_HIGH[j], q.max() + pad)

    n = 5
    fig, axes = plt.subplots(n, n, figsize=(11.6, 11.0))
    fig.patch.set_facecolor(SURFACE)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            if j > i:
                ax.axis("off")
                continue
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=8.5, direction="out")

            if i == j:
                for k, (_, c) in enumerate(chains):
                    counts, edges = np.histogram(c[:, j], bins=BINS_1D,
                                                 range=(lo[j], hi[j]), density=True)
                    dens = gaussian_filter(counts, SMOOTH_1D)
                    centres = 0.5 * (edges[:-1] + edges[1:])
                    ax.plot(centres, dens / dens.max(), color=SERIES[k], lw=1.9)
                ax.set_ylim(0, 1.13)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                if args.truth:
                    ax.axvline(TRUTH[j], color=INK_2, lw=1.0, ls=(0, (4, 2.5)), zorder=0)
            else:
                for k, (_, c) in enumerate(chains):
                    h, xe, ye = np.histogram2d(
                        c[:, j], c[:, i], bins=BINS_2D,
                        range=[[lo[j], hi[j]], [lo[i], hi[i]]])
                    h = gaussian_filter(h.T, SMOOTH_2D)
                    if h.max() <= 0:
                        continue
                    lv = contour_levels(h)
                    xc = 0.5 * (xe[:-1] + xe[1:])
                    yc = 0.5 * (ye[:-1] + ye[1:])
                    ax.contourf(xc, yc, h, levels=lv + [h.max() * 1.001],
                                colors=SERIES[k], alpha=0.22, zorder=1 + k)
                    ax.contour(xc, yc, h, levels=lv, colors=SERIES[k],
                               linewidths=1.5, zorder=5 + k)
                ax.set_ylim(lo[i], hi[i])
                if args.truth:
                    ax.axvline(TRUTH[j], color=INK_2, lw=0.9, ls=(0, (4, 2.5)), zorder=0)
                    ax.axhline(TRUTH[i], color=INK_2, lw=0.9, ls=(0, (4, 2.5)), zorder=0)
            ax.set_xlim(lo[j], hi[j])

            if i == n - 1:
                ax.set_xlabel(LABELS[j], color=INK, fontsize=13, labelpad=6)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS[i], color=INK, fontsize=13, labelpad=6)
            elif j != 0:
                ax.set_yticklabels([])
            ax.locator_params = None
            ax.xaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
            if i != j:
                ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))

    handles = [Line2D([], [], color=SERIES[k], lw=2.6, label=name)
               for k, (name, _) in enumerate(chains)]
    if args.truth:
        handles.append(Line2D([], [], color=INK_2, lw=1.2, ls=(0, (4, 2.5)),
                              label="truth"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.975, 0.965),
               frameon=False, fontsize=14, labelcolor=INK, handlelength=1.9,
               borderaxespad=0.0)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.075,
                        wspace=0.09, hspace=0.09)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
