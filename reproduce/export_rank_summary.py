#!/usr/bin/env python3
"""Compact mean-rank summary for the main text.

    python reproduce/export_rank_summary.py --out reproduce/tables

One small table: the mean rank of every method on NLL and CRPS, per benchmark
suite, over the complete-case datasets used by the critical-difference
diagrams (identical inputs to make_figures.py, so the two cannot disagree).
Lower is better; best per column in bold. The per-dataset values behind these
ranks are in the supplementary material's full tables.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("mf", HERE / "make_figures.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

METHODS = ["DGBM-LGB", "DGBM-XGB", "GPBoost", "NGBoost", "PGBM", "XLSF"]
CELLS = [("uci", "NLL"), ("uci", "CRPS"), ("openml", "NLL"), ("openml", "CRPS")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path,
                    default=mf.REPO_ROOT / "reproduce" / "tables")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ranks, counts = {}, {}
    for suite, metric in CELLS:
        data, missing = mf.matrix(suite, mf.METRICS[metric])
        for m in missing:
            print(f"  note ({suite}/{metric}): missing {m}")
        ranks[(suite, metric)] = data.rank(axis=1).mean()
        counts[(suite, metric)] = data.shape[0]

    hdr = (r"      \textbf{Method} & \textbf{NLL} & \textbf{CRPS} & "
           r"\textbf{NLL} & \textbf{CRPS} \\")
    L = [r"\begin{table}[h!]", r"    \centering",
         r"    \caption{Mean rank (lower is better) of each method on the two "
         r"proper scoring rules, over the complete-case datasets of each suite "
         rf"(UCI: {counts[('uci','NLL')]} datasets; OpenML: "
         rf"{counts[('openml','NLL')]} datasets), matching the "
         r"critical-difference diagrams. Best per column in bold; per-dataset "
         r"values are in Section~6 of the supplementary material.}",
         r"    \label{tab:rank_summary}",
         r"      \begin{tabular}{lcccc}", r"      \toprule",
         r"      & \multicolumn{2}{c}{\textbf{UCI}} & "
         r"\multicolumn{2}{c}{\textbf{OpenML}} \\",
         r"      \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         hdr, r"      \midrule"]
    best = {c: min(ranks[c][m] for m in METHODS if m in ranks[c]) for c in CELLS}
    for m in METHODS:
        row = [m]
        for c in CELLS:
            v = ranks[c].get(m)
            if v is None:
                row.append("---")
                continue
            txt = f"{v:.2f}"
            if abs(v - best[c]) < 1e-9:
                txt = rf"\textbf{{{txt}}}"
            row.append(txt)
        L.append("      " + " & ".join(row) + r" \\")
    L += [r"      \bottomrule", r"      \end{tabular}", r"\end{table}", ""]

    path = args.out / "table_rank_summary.tex"
    path.write_text("\n".join(L))
    print(f"wrote {path}")
    for c in CELLS:
        print(f"  {c[0]:>7}/{c[1]:<5} n={counts[c]:>2}  " +
              "  ".join(f"{m}:{ranks[c].get(m, float('nan')):.2f}" for m in METHODS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
