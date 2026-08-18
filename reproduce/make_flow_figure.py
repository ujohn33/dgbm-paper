#!/usr/bin/env python3
"""Draw the flexible-family / calibration figure from run_flow_calibration.py output.

    python reproduce/make_flow_figure.py --out reproduce/figures

Three panels, one argument:

(a) why the Gaussian is wrong -- fitted predictive densities at a single test
    point against the known true conditional density;
(b) what that costs -- change in NLL and CRPS relative to the Gaussian, showing
    the two metrics disagreeing in sign;
(c) what NLL and CRPS cannot show -- PIT histograms against uniform.

Reads ``flow_calibration.json`` and ``synthetic_samples.npz``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

# Categorical slots 1-3 of the reference palette, validated for CVD separation
# against a light surface (worst adjacent pair dE 9.2 deutan / 27.6 normal).
TRUTH = "#8a8985"
COLORS = {"Gaussian": "#2a78d6", "Mixture(2)": "#eb6834", "SplineFlow": "#1baf7a"}
INK, MUTED = "#0b0b0b", "#52514e"
FAMILIES = ["Gaussian", "Mixture(2)", "SplineFlow"]


def true_density(x, grid):
    """Conditional density of the synthetic DGP, known in closed form."""
    from scipy.stats import norm
    sep = 1.0 + 1.5 * (x[0] + 2) / 4
    w = 1 / (1 + np.exp(-x[1]))
    lo, hi = -sep + 0.3 * x[2], sep + 0.3 * x[2]
    return (1 - w) * norm.pdf(grid, lo, 0.35) + w * norm.pdf(grid, hi, 0.35)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy.stats import gaussian_kde

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[1]
                    / "reproduce" / "flow_calibration")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[1]
                    / "reproduce" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import glob as _glob
    seed_files = sorted(_glob.glob(str(args.data / "flow_calibration_seed*.json")))
    runs = [json.loads(pathlib.Path(f).read_text())["Synthetic bimodal"]
            for f in seed_files]
    print(f"pooling {len(runs)} seeds")
    # panel (a) is illustrative and drawn from one seed; (b) and (c) are pooled
    npz = np.load(args.data / "synthetic_samples_seed0.npz")
    X, y = npz["X_test"], npz["y_test"]

    # A test point where the truth is unambiguously bimodal: modes far apart
    # (large x0) and near-equal weight (x1 close to 0).
    idx = int(np.argmax(X[:, 0] - 3 * np.abs(X[:, 1])))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(9.6, 5.0))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.25, 1.0],
                  hspace=0.74, wspace=0.30)

    # (a) predictive densities against the truth ---------------------------
    ax = fig.add_subplot(gs[0, :2])
    grid = np.linspace(-6, 6, 600)
    ax.fill_between(grid, true_density(X[idx], grid), color=TRUTH, alpha=0.30,
                    lw=0, label="True density")
    for fam in FAMILIES:
        kde = gaussian_kde(npz[f"samples_{fam}"][idx], bw_method=0.12)
        ax.plot(grid, kde(grid), color=COLORS[fam], lw=2, label=fam)
    ax.set_xlabel("y")
    ax.set_yticks([])
    ax.set_title("(a)  Predictive density at one test point", loc="left",
                 fontsize=10, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.annotate("Gaussian fills the gap\nbetween the modes",
                xy=(0.02, 0.74), xycoords="axes fraction", fontsize=8,
                color=MUTED, ha="left")

    # (b) the two metrics disagree in sign ---------------------------------
    ax = fig.add_subplot(gs[0, 2])
    flex = [f for f in FAMILIES if f != "Gaussian"]
    width, offs = 0.34, {"NLL": -0.19, "CRPS": 0.19}
    for metric, hatch in (("NLL", None), ("CRPS", "///")):
        # per-seed percentage change vs the Gaussian, pooled across seeds
        per_seed = {f: [100 * (r[f][metric] - r["Gaussian"][metric])
                        / r["Gaussian"][metric] for r in runs] for f in flex}
        vals = [float(np.mean(per_seed[f])) for f in flex]
        errs = [float(np.std(per_seed[f], ddof=1)) for f in flex]
        pos = np.arange(len(flex)) + offs[metric]
        ax.bar(pos, vals, width, hatch=hatch, yerr=errs, capsize=3,
               error_kw=dict(ecolor=MUTED, lw=1),
               color=[COLORS[f] for f in flex], edgecolor="white", linewidth=0.8)
        for p, v, e in zip(pos, vals, errs):
            off = e + 2.0
            ax.text(p, v + (off if v >= 0 else -off), f"{v:+.0f}%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(np.arange(len(flex)))
    ax.set_xticklabels([f.replace("(2)", "") for f in flex], fontsize=8)
    ax.set_ylabel("change vs Gaussian")
    ax.set_ylim(-46, 40)
    ax.set_yticks([])
    # Colour already carries the family here, so the legend distinguishes the
    # metrics by hatch alone, on a neutral swatch.
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=MUTED, edgecolor="white", label="NLL"),
                       Patch(facecolor=MUTED, edgecolor="white", hatch="///",
                             label="CRPS")],
              frameon=False, fontsize=8, loc="upper left", handlelength=1.4,
              borderpad=0.1, labelspacing=0.3)
    ax.set_title("(b)  Better on NLL, worse on CRPS", loc="left",
                 fontsize=10, pad=8)

    # (c) PIT histograms ---------------------------------------------------
    for j, fam in enumerate(FAMILIES):
        ax = fig.add_subplot(gs[1, j])
        hs = np.array([r[fam]["PIT_hist"] for r in runs], dtype=float)
        h = hs.mean(axis=0)
        ax.bar(np.arange(10) / 10 + 0.05, h, 0.085, color=COLORS[fam],
               edgecolor="white", linewidth=0.8,
               yerr=hs.std(axis=0, ddof=1), capsize=2,
               error_kw=dict(ecolor=MUTED, lw=0.8))
        ax.axhline(0.1, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.set_ylim(0, max(0.22, h.max() * 1.25))
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([])
        ax.set_xlabel("PIT", fontsize=8)
        devs = [r[fam]["PIT_deviation"] for r in runs]
        ax.set_title(f"{fam}   dev {np.mean(devs):.2f}$\\pm${np.std(devs, ddof=1):.2f}",
                     loc="left", fontsize=9, color=INK, pad=4)
    fig.text(0.077, 0.425, "(c)  PIT histograms, mean over 10 seeds — flat (dashed) is calibrated",
             fontsize=10, color=INK, ha="left")

    path_pdf = args.out / "flow_calibration_figure.pdf"
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(args.out / "flow_calibration_figure.png", dpi=200,
                bbox_inches="tight")
    print(f"wrote {path_pdf} (+ .png), test point index {idx} "
          f"x0={X[idx,0]:.2f} x1={X[idx,1]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
