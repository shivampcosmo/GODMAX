"""Plot the mock-SBI training bandpowers against the truth data vector.

Three panels, one per spectrum (gy, gkappa, gtau), in the frozen 14-band inference
binning.  Each panel carries three things:

*   every ``signal + noise`` training row (n_points x n_replicas of them) as a
    low-alpha line -- the object the NLE is actually trained on;
*   the noiseless pasted response ``mu(theta)`` for each design point -- the SIGNAL
    spread alone, i.e. how far the design moves the model;
*   the truth data vector (the noiseless pasted mock observation) as markers.

Separating the second from the first is the point of the figure: it shows at a glance
whether the design's signal excursion dominates the per-realization noise, which is
what decides whether the training set carries information about theta at all.

The lower row divides by the truth so the spread is readable, with the +/-1 sigma
band from the frozen covariance diagonal drawn for reference -- that band is the
noise scale the design has to be compared against.

Nothing here is fitted or corrected; the pasted vectors are used exactly as measured.
"""

from __future__ import annotations

# --- keep imports working from a theme subfolder: common/ holds the
# --- modules shared by more than one stage.
import pathlib as _pl, sys as _sys
_ROOT = _pl.Path(__file__).resolve().parents[1]
for _d in (_ROOT, _ROOT / "common"):
    if str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))

import argparse
import pathlib
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mock_sbi_common as msc

# Validated categorical slots (dataviz reference palette, light surface):
#   node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light  -> ALL CHECKS PASS
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
SERIES_NOISY = "#2a78d6"   # slot 1, blue   -- signal + noise rows
SERIES_SIGNAL = "#eb6834"  # slot 2, orange -- noiseless mu(theta)
GRID = "#d8d7d2"

PANEL_TITLE = {"gy": r"$g \times y$", "gkappa": r"$g \times \kappa_{\rm CMB}$",
               "gtau": r"$g \times \tau$"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--training-set", type=pathlib.Path, required=True)
    p.add_argument("--observation", type=pathlib.Path,
                   default=msc.REPO_ROOT / "data/SBI_validate/three_probe_inference/observation_mock.h5")
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--ratio-ylim", type=float, nargs=2, default=(-0.6, 3.0))
    p.add_argument("--dpi", type=int, default=170)
    args = p.parse_args()

    data = np.load(args.training_set, allow_pickle=True)
    x, mu = np.asarray(data["x"]), np.asarray(data["mu"])
    with h5py.File(args.observation, "r") as handle:
        truth = np.asarray(handle["data_vector"], dtype=np.float64)
        kind = str(handle.attrs["observation_kind"])
    ctx = msc.load_estimator_context()
    ell = np.asarray(ctx.effective_ell, dtype=np.float64)
    sigma = np.sqrt(np.diag(np.asarray(ctx.covariance, dtype=np.float64)))

    n_rows, n_points = x.shape[0], mu.shape[0]
    print(f"{n_rows} training rows from {n_points} design points; "
          f"observation {args.observation.name} ({kind})")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.09,
                                          "wspace": 0.235})
    fig.patch.set_facecolor(SURFACE)

    for col, name in enumerate(msc.SPECTRA):
        sl = slice(msc.N_BAND * col, msc.N_BAND * (col + 1))
        rows, sig, t, s = x[:, sl], mu[:, sl], truth[sl], sigma[sl]
        top, bot = axes[0, col], axes[1, col]
        for ax in (top, bot):
            ax.set_facecolor(SURFACE)
            ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=9)

        # --- top: the bandpowers themselves --------------------------------------
        # Mask non-positive values to NaN before drawing.  Left as-is, matplotlib draws
        # the segment from a positive band down to the bottom of a log axis, producing
        # vertical streaks that look like data and are not.
        rows_pos = np.where(rows > 0, rows, np.nan)
        sig_pos = np.where(sig > 0, sig, np.nan)
        top.plot(ell, rows_pos.T, color=SERIES_NOISY, lw=0.35, alpha=0.012,
                 solid_capstyle="round", zorder=1)
        top.plot(ell, sig_pos.T, color=SERIES_SIGNAL, lw=0.6, alpha=0.30, zorder=2)
        top.plot(ell, t, "o", ms=6.5, color=INK, mec=SURFACE, mew=1.4, zorder=4)
        top.set_xscale("log")
        # Log axis: values <= 0 cannot be drawn here.  Rather than let symlog give the
        # handful of negative gy excursions half the panel -- which buries the signal --
        # the top row stays log and SAYS how many points it cannot show, and the ratio
        # row below is linear and displays every one of them.
        top.set_yscale("log")
        n_neg = int((rows <= 0).sum())
        top.set_title(PANEL_TITLE[name], color=INK, fontsize=12, pad=8)
        if col == 0:
            top.set_ylabel(r"$C_\ell$", color=INK, fontsize=11)

        # --- bottom: ratio to truth, with the 1-sigma noise scale ----------------
        ratio = rows / t[None, :]
        bot.plot(ell, ratio.T, color=SERIES_NOISY, lw=0.35, alpha=0.012, zorder=2)
        bot.plot(ell, (sig / t[None, :]).T, color=SERIES_SIGNAL, lw=0.6, alpha=0.30,
                 zorder=3)
        # The +/-1 sigma envelope goes ON TOP as lines, not behind as a fill: with 6016
        # overlapping rows a translucent fill is simply invisible.
        for edge in (1 - s / t, 1 + s / t):
            bot.plot(ell, edge, color=INK_2, lw=1.3, ls=(0, (5, 2.5)), zorder=6)
        bot.axhline(1.0, color=INK, lw=1.1, zorder=4)
        bot.plot(ell, np.ones_like(ell), "o", ms=5.0, color=INK, mec=SURFACE,
                 mew=1.2, zorder=5)
        bot.set_xscale("log")
        bot.set_ylim(*args.ratio_ylim)
        bot.set_xlabel(r"$\ell_{\rm eff}$", color=INK, fontsize=11)
        if col == 0:
            bot.set_ylabel(r"$C_\ell\,/\,C_\ell^{\rm truth}$", color=INK, fontsize=11)

        # Never truncate silently: say how much left the frame, and how far.
        outside = int(((ratio < args.ratio_ylim[0]) | (ratio > args.ratio_ylim[1])).any(axis=1).sum())
        bot.text(0.975, 0.95,
                 f"{outside}/{n_rows} rows leave frame (max {ratio.max():.0f}x, "
                 f"min {ratio.min():.1f}x)",
                 transform=bot.transAxes, ha="right", va="top", fontsize=7.4,
                 color=INK_2)
        if n_neg:
            top.text(0.035, 0.045,
                     f"{n_neg} band values $\\leq 0$ ({100*n_neg/rows.size:.1f}%)\n"
                     f"not drawable on a log axis \u2014 see ratio panel",
                     transform=top.transAxes, ha="left", va="bottom", fontsize=7.4,
                     color=INK_2)

    # Legend: identity is never colour-alone -- each entry is also direct-labelled
    # on the first panel below.
    handles = [
        plt.Line2D([], [], color=SERIES_NOISY, lw=2.0,
                   label=f"signal + noise training rows  ({n_rows:,})"),
        plt.Line2D([], [], color=SERIES_SIGNAL, lw=2.0,
                   label=f"noiseless pasted $\\mu(\\theta)$  ({n_points} design points)"),
        plt.Line2D([], [], color=INK, marker="o", ms=6.5, mec=SURFACE, mew=1.4,
                   ls="none", label="truth (noiseless pasted observation)"),
        plt.Line2D([], [], color=INK_2, lw=1.3, ls=(0, (5, 2.5)),
                   label=r"$\pm 1\sigma$ from the frozen covariance"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.2, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Mock-SBI round 1: training bandpowers against the truth data vector",
                 color=INK, fontsize=13.5, y=0.985)
    fig.subplots_adjust(top=0.90, bottom=0.115, left=0.062, right=0.988)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
