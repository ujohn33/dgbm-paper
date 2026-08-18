#!/usr/bin/env python3
"""Pool the per-seed flow/calibration runs into mean +/- std.

    python reproduce/aggregate_flow_calibration.py --data reproduce/flow_calibration

Single-split numbers are not reportable here: the UCI test sets are 30-103
observations, so a 50% coverage estimate carries a standard error of several
percentage points. This pools the per-seed JSONs and reports mean +/- std across
seeds, matching how the rest of the paper reports fold variation.

Also flags instability rather than averaging it away: a family whose predictive
samples occasionally escape to extreme values produces a mean that describes no
individual run, so the per-seed spread and the worst seed are printed alongside.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

FAMILIES = ["Gaussian", "Mixture(2)", "SplineFlow"]
METRICS = ["NLL", "CRPS", "PIT_deviation", "coverage_50", "coverage_90",
           "width_90", "best_iteration"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, default=pathlib.Path("."))
    ap.add_argument("--tex", type=pathlib.Path, default=None,
                    help="also write a LaTeX table here")
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.data / "flow_calibration_seed*.json")))
    if not files:
        print("no per-seed files found", file=sys.stderr)
        return 1
    runs = [json.loads(pathlib.Path(f).read_text()) for f in files]
    print(f"pooling {len(runs)} seeds\n")

    datasets = list(runs[0])
    pooled = {}
    for ds in datasets:
        print(f"=== {ds} ===")
        print(f"  {'family':<12}{'NLL':>16}{'CRPS':>16}{'PIT dev':>15}"
              f"{'cov50':>14}{'cov90':>14}{'iters':>8}")
        pooled[ds] = {}
        for fam in FAMILIES:
            vals = {m: np.array([r[ds][fam][m] for r in runs if fam in r.get(ds, {})],
                                dtype=float) for m in METRICS}
            # Degenerate seeds -- runaway predictive samples -- corrupt every
            # sample-based metric at once, so they are excluded from those
            # columns (never from NLL) and counted explicitly instead of being
            # averaged into a number that describes no actual run.
            crps = vals["CRPS"]
            clean = np.isfinite(crps) & (np.nan_to_num(crps, nan=np.inf)
                                         <= 10 * np.nanmedian(crps))
            n_degen = int((~clean).sum())
            for m in ("CRPS", "PIT_deviation", "coverage_50", "coverage_90",
                      "width_90"):
                vals[m] = vals[m][clean]
            pooled[ds][fam] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                                   "min": float(np.min(v)), "max": float(np.max(v)),
                                   "n": int(len(v))}
                               for m, v in vals.items()}
            pooled[ds][fam]["n_degenerate"] = n_degen
            f = lambda m, d=2: (f"{np.mean(vals[m]):.{d}f}±{np.std(vals[m], ddof=1):.{d}f}")
            print(f"  {fam:<12}{f('NLL'):>16}{f('CRPS'):>16}{f('PIT_deviation'):>15}"
                  f"{f('coverage_50'):>14}{f('coverage_90'):>14}"
                  f"{np.mean(vals['best_iteration']):>8.0f}")
        # Instability check: a spread far larger than the level means the mean
        # is describing no actual run.
        for fam in FAMILIES:
            c = pooled[ds][fam]["CRPS"]
            if c["std"] > 0.5 * abs(c["mean"]) and c["n"] > 1:
                print(f"    ! {fam}: CRPS unstable across seeds "
                      f"(min {c['min']:.2f}, max {c['max']:.2f})")
        print()

    out = args.data / "flow_calibration_pooled.json"
    out.write_text(json.dumps(pooled, indent=1))
    print(f"wrote {out}")

    if args.tex:
        rows = []
        for ds in datasets:
            rows.append(r"      \midrule")
            rows.append(rf"      \multicolumn{{7}}{{c}}{{\textbf{{{ds}}}}} \\")
            rows.append(r"      \midrule")
            for fam in FAMILIES:
                p = pooled[ds][fam]
                g = lambda m, d=2: f"{p[m]['mean']:.{d}f}$\\pm${p[m]['std']:.{d}f}"
                mark = (rf"$^{{\dagger({p['n_degenerate']})}}$"
                        if p["n_degenerate"] else "")
                rows.append(f"      {fam}{mark} & {g('NLL')} & {g('CRPS')} & "
                            f"{g('PIT_deviation')} & {g('coverage_50')} & "
                            f"{g('coverage_90')} & {g('width_90')} \\\\")
        tex = "\n".join([
            r"\begin{table}[h!]", r"    \centering",
            r"    \caption{Distributional families on a synthetic bimodal target and "
            r"four UCI datasets: mean $\pm$ std over " + str(len(runs)) +
            r" seeds. Coverage is of central intervals at the stated nominal level; "
            r"PIT deviation is the $L_1$ distance of the PIT histogram from uniform "
            r"(0 = calibrated). $^{\dagger(k)}$: $k$ seeds produced degenerate "
            r"predictive samples (a runaway mixture component, "
            r"Section~\ref{sec:flow_lessons}); they are excluded from the "
            r"sample-based columns of that row but not from NLL.}",
            r"    \label{tab:flow_calibration}",
            r"    \adjustbox{max width=\textwidth}{",
            r"      \begin{tabular}{lrrrrrr}", r"      \toprule",
            r"      \textbf{Family} & \textbf{NLL} & \textbf{CRPS} & "
            r"\textbf{PIT dev.} & \textbf{Cov.\ 50\%} & \textbf{Cov.\ 90\%} & "
            r"\textbf{Width 90\%} \\", *rows,
            r"      \bottomrule", r"      \end{tabular}%", r"    }", r"\end{table}", ""])
        args.tex.write_text(tex)
        print(f"wrote {args.tex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
