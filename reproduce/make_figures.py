#!/usr/bin/env python3
"""Regenerate the paper's figures from the result CSVs in ``results/``.

    python reproduce/make_figures.py                 # everything, into reproduce/figures/
    python reproduce/make_figures.py --out figures   # somewhere else
    python reproduce/make_figures.py --only cd       # just the CD diagrams
    python reproduce/make_figures.py --only time     # just the runtime boxplots

Produces
--------
    cd_diagram_{NLL,CRPS,RMSE}_{uci,openml}.png   critical-difference diagrams
    run_time_figure.png                            training time per method
    hp_time_figure.png                             tuning time per method

Method
------
CD diagrams follow Demsar (2006) as refined by Benavoli et al. (2016): methods
are ranked per dataset, a Friedman test checks whether the ranks differ at all,
and pairwise Wilcoxon signed-rank tests with Holm correction decide which
methods are joined by a bar (i.e. are not significantly different). The
published figures were drawn with the ``critdd`` package; this reimplements the
same procedure with scipy so the repository has no extra dependency.

Only datasets on which *every* method has a result are used, so all methods are
ranked on the same footing. The count is printed for each diagram, and it is
what the paper reports: 9 UCI datasets (GPBoost timed out on Year Prediction
MSD) and, where GPBoost OpenML results are present, 17 OpenML datasets.

Provenance
----------
Built from the result CSVs in this repository. GPBoost on OpenML and DGBM-XGB
on UCI were rerun to fill gaps in the original set: GPBoost had no OpenML result
file at all, and DGBM-XGB on UCI had been run with a tenth of DGBM-LGB's
boosting budget. DGBM-XGB therefore reads the matched-protocol run
(``*_n_est_2000_seed*.csv``). Rank values consequently differ from the published
figures on the affected methods.

Requires numpy, pandas, scipy and matplotlib -- all in requirements-dgbm.txt.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import pathlib
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Display name -> result file, for the configuration reported in the paper
# (no natural gradient, exp response function, no stabilization).
SOURCES = {
    "uci": {
        "DGBM-LGB": "results/uci/uci_LSSboost_no_natural_exp_None.csv",
        # one file per dataset index, from the run whose protocol matches DGBM-LGB
        "DGBM-XGB": "results/uci/uci_XGBoostLSS_*n_est_2000_seed*.csv",
        "GPBoost":  "results/uci/uci_GPboost.csv",
        "NGBoost":  "results/uci/NGboost_natural_crps_calibration_sharpness.csv",
        "PGBM":     "results/uci/uci_pgbm.csv",
        "XLSF":     "results/uci/uci_GluonTS_LSF.csv",
    },
    "openml": {
        "DGBM-LGB": "results/openml/openml_LSSboost_no_natural_exp_None_std_False_safety_False_job_11752126.csv",
        "DGBM-XGB": "results/openml/openml_XGBoostLSS_no_natural_exp_None_safety_False_job_11752710.csv",
        "GPBoost":  "results/openml/openml_GPboost.csv",
        "NGBoost":  "logs/openml/openml_NGBoost_natural.csv",
        "PGBM":     "results/openml/openml_PGBM_NLL_seeded.csv",
        "XLSF":     "results/openml/openml_GluonTS_LSF.csv",
    },
}

# Canonical column order, used for the files that were written without a header.
HEADERLESS_COLUMNS = [
    "dset", "RMSE-mean", "RMSE-std", "NLL-mean", "NLL-std", "CRPS-mean",
    "CRPS-std", "CRPS-calibration-mean", "CRPS-calibration-std",
    "CRPS-sharpness-mean", "CRPS-sharpness-std", "time_run", "time_HP",
]

METRICS = {"NLL": "NLL-mean", "CRPS": "CRPS-mean", "RMSE": "RMSE-mean"}


def _read_one(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.columns[0] != "dset":
        # written without a header: re-read positionally
        df = pd.read_csv(path, header=None)
        df = df.iloc[:, : len(HEADERLESS_COLUMNS)]
        df.columns = HEADERLESS_COLUMNS[: df.shape[1]]
    df["dset"] = df["dset"].astype(str).str.strip()
    # "Year Prediciton MSD" is misspelled in the loaders; normalise it
    df["dset"] = df["dset"].replace({"Year Prediciton MSD": "Year Prediction MSD"})
    # some files concatenate repeated runs; keep the first occurrence
    return df.drop_duplicates(subset="dset", keep="first")


def load(spec) -> pd.DataFrame | None:
    """Read a result source: a single CSV, or a glob over per-dataset files."""
    spec = str(spec)
    paths = ([pathlib.Path(p) for p in sorted(glob.glob(spec))] if any(c in spec for c in "*?[")
             else ([pathlib.Path(spec)] if pathlib.Path(spec).exists() else []))
    if not paths:
        return None
    frames = [_read_one(p) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset="dset", keep="first").set_index("dset")


def matrix(suite: str, column: str) -> tuple[pd.DataFrame, list[str]]:
    """Build a datasets x methods matrix for one metric; report missing sources."""
    series, missing = {}, []
    for method, rel in SOURCES[suite].items():
        df = load(REPO_ROOT / rel)
        if df is None:
            missing.append(f"{method} ({rel})")
            continue
        if column not in df.columns:
            missing.append(f"{method} (no '{column}' column)")
            continue
        series[method] = pd.to_numeric(df[column], errors="coerce")
    if not series:
        return pd.DataFrame(), missing
    return pd.DataFrame(series).dropna(how="any"), missing


def holm_cliques(data: np.ndarray, names: list[str], alpha: float = 0.05):
    """Maximal sets of methods that are pairwise not significantly different."""
    from scipy.stats import wilcoxon

    k = len(names)
    pairs, pvals = [], []
    for i, j in itertools.combinations(range(k), 2):
        try:
            p = wilcoxon(data[:, i], data[:, j]).pvalue
        except ValueError:          # identical columns
            p = 1.0
        pairs.append((i, j))
        pvals.append(p)

    order = np.argsort(pvals)
    m = len(pvals)
    different = set()
    for rank, idx in enumerate(order):
        if pvals[idx] * (m - rank) < alpha:
            different.add(pairs[idx])
        else:
            break                    # Holm stops at the first non-rejection

    cliques = []
    for size in range(k, 1, -1):
        for combo in itertools.combinations(range(k), size):
            if any((a, b) in different for a, b in itertools.combinations(combo, 2)):
                continue
            if any(set(combo) <= set(c) for c in cliques):
                continue
            cliques.append(combo)
    return cliques


def cd_diagram(ranks: pd.Series, cliques, n_datasets: int, title: str, out: pathlib.Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(ranks.index)
    values = ranks.to_numpy()
    k = len(names)
    lo, hi = 1, k
    order = np.argsort(values)              # best (lowest rank) first

    half = (k + 1) // 2
    lowest = 0.82 - 0.14 - 0.11 * (half - 1)
    fig, ax = plt.subplots(figsize=(9, 1.7 + 0.34 * half))
    ax.set_xlim(hi + 0.4, lo - 0.4)         # reversed: rank 1 on the right
    ax.set_ylim(lowest - 0.13, 1.02)
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=12)

    axis_y = 0.82
    ax.plot([lo, hi], [axis_y, axis_y], color="black", lw=1.6)
    for tick in range(lo, hi + 1):
        ax.plot([tick, tick], [axis_y, axis_y + 0.05], color="black", lw=1.6)
        ax.text(tick, axis_y + 0.09, str(tick), ha="center", va="bottom", fontsize=11)
        if tick < hi:
            ax.plot([tick + 0.5, tick + 0.5], [axis_y, axis_y + 0.03], color="black", lw=1.0)

    for slot, idx in enumerate(order):
        right = slot < half                 # better half labelled on the right
        row = slot if right else k - 1 - slot
        y = axis_y - 0.14 - 0.11 * row
        edge = lo - 0.30 if right else hi + 0.30
        ax.plot([values[idx], values[idx]], [axis_y, y], color="black", lw=1.1)
        ax.plot([values[idx], edge], [y, y], color="black", lw=1.1)
        ax.text(edge + (-0.04 if right else 0.04), y, names[idx],
                ha="left" if right else "right", va="center", fontsize=12)
        ax.text(values[idx] + (-0.03 if right else 0.03), y + 0.025, f"{values[idx]:.4f}",
                ha="left" if right else "right", va="bottom", fontsize=9)

    for level, combo in enumerate(cliques):
        member = [values[i] for i in combo]
        y = axis_y - 0.045 - 0.035 * level
        ax.plot([min(member) - 0.03, max(member) + 0.03], [y, y], color="black", lw=4.5,
                solid_capstyle="butt")

    ax.text(0.5, 0.015, f"{n_datasets} datasets", transform=ax.transAxes,
            ha="center", fontsize=8, color="0.45")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_cd_diagrams(outdir: pathlib.Path) -> None:
    from scipy.stats import friedmanchisquare

    for suite in ("uci", "openml"):
        for metric, column in METRICS.items():
            df, missing = matrix(suite, column)
            if missing:
                print(f"  [{suite}/{metric}] unavailable: {', '.join(missing)}")
            if df.shape[0] < 3 or df.shape[1] < 3:
                print(f"  [{suite}/{metric}] skipped: need >=3 datasets and >=3 methods, "
                      f"have {df.shape[0]} x {df.shape[1]}")
                continue

            ranks = df.rank(axis=1, ascending=True).mean().sort_values()
            data = df[list(ranks.index)].to_numpy()
            p = friedmanchisquare(*data.T).pvalue
            cliques = holm_cliques(data, list(ranks.index))

            out = outdir / f"cd_diagram_{metric}_{suite}.png"
            cd_diagram(ranks, cliques, df.shape[0], f"{metric} ({suite.upper()})", out)
            print(f"  wrote {out.name}  n={df.shape[0]}  Friedman p={p:.2e}")
            print("        ranks: " + ", ".join(f"{m} {v:.4f}" for m, v in ranks.items()))


def make_time_figures(outdir: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for column, fname, label in [("time_run", "run_time_figure.png", "Training time"),
                                 ("time_HP", "hp_time_figure.png",
                                  "Hyperparameter optimization time")]:
        pooled: dict[str, list[float]] = {}
        for suite in ("uci", "openml"):
            for method, rel in SOURCES[suite].items():
                df = load(REPO_ROOT / rel)
                if df is None or column not in df.columns:
                    continue
                vals = pd.to_numeric(df[column], errors="coerce").dropna()
                vals = vals[vals > 0]
                pooled.setdefault(method, []).extend(vals.tolist())

        methods = [m for m in SOURCES["uci"] if pooled.get(m)]
        if not methods:
            print(f"  [{column}] skipped: no data")
            continue

        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.set_facecolor("0.92")
        ax.grid(True, color="white", lw=0.9)
        ax.set_axisbelow(True)
        bp = ax.boxplot([pooled[m] for m in methods], patch_artist=True, widths=0.62,
                        medianprops=dict(color="black", lw=1.4),
                        flierprops=dict(marker="o", markersize=4,
                                        markerfacecolor="none", markeredgecolor="black"))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(str(0.25 + 0.10 * i))
            patch.set_edgecolor("black")

        ax.set_yscale("log")
        ax.set_ylabel("Time (s) - log scale")
        ax.set_xlabel("Method")
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()
        out = outdir / fname
        fig.savefig(out, dpi=200)
        plt.close(fig)
        n = sum(len(pooled[m]) for m in methods)
        print(f"  wrote {out.name}  ({label}, {n} observations across {len(methods)} methods)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO_ROOT / "reproduce" / "figures")
    ap.add_argument("--only", choices=["cd", "time"], default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.only in (None, "cd"):
        print("Critical-difference diagrams:")
        make_cd_diagrams(args.out)
    if args.only in (None, "time"):
        print("Runtime figures:")
        make_time_figures(args.out)

    print(f"\nFigures written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
