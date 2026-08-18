#!/usr/bin/env python3
"""Draw the regressor-informativeness sweep figure.

    python reproduce/make_sweep_figure.py --out reproduce/figures

One panel, one message: the mixture's NLL advantage over the Gaussian as a
function of how informative the observed component indicator is. The advantage
is flat up to p = 0.99 and collapses only at exactly p = 1, because for any
p < 1 the conditional law given the regressors is still a two-component
mixture, and the log score heavily penalizes assigning near-zero density to
even a rare minority component.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

INK, MUTED, ACCENT = "#0b0b0b", "#52514e", "#2a78d6"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[1]
                    / "reproduce" / "flow_calibration")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[1]
                    / "reproduce" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    parsed = args.data / "regressor_sweep_parsed.json"
    if parsed.exists():
        runs = list(json.loads(parsed.read_text()).values())
        advs = {p: [r[p]["adv"] for r in runs] for p in runs[0]}
    else:
        files = sorted(glob.glob(str(args.data / "regressor_sweep_seed*.json")))
        runs = [json.loads(pathlib.Path(f).read_text()) for f in files]
        advs = {p: [r[p]["advantage_NLL"] for r in runs] for p in runs[0]}

    ps = sorted(float(p) for p in advs)
    mean = np.array([np.mean(advs[f"{p:.2f}"]) for p in ps])
    sd = np.array([np.std(advs[f"{p:.2f}"], ddof=1) for p in ps])

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for p in ps:
        vals = advs[f"{p:.2f}"]
        ax.plot([p] * len(vals), vals, "o", color=ACCENT, alpha=0.35, ms=4,
                mec="none")
    ax.errorbar(ps, mean, yerr=sd, color=ACCENT, lw=2, marker="o", ms=5,
                capsize=3, zorder=3, label="mean $\\pm$ sd over seeds")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("informativeness $p$ of the observed component indicator")
    ax.set_ylabel("NLL advantage of Mixture(2)\nover Gaussian (nats)")
    ax.annotate("advantage persists at $p=0.99$ ...", xy=(0.99, mean[-2]),
                xytext=(0.62, 0.72), fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("... and vanishes only at $p=1$", xy=(1.0, mean[-1]),
                xytext=(0.62, -0.18), fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"regressor_sweep_figure.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"wrote {args.out}/regressor_sweep_figure.pdf (+ .png)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
