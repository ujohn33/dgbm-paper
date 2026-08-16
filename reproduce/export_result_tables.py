#!/usr/bin/env python3
"""Regenerate the main NLL and CRPS result tables from the result CSVs.

    python reproduce/export_result_tables.py --out reproduce/tables

Writes ``table_nll_main.tex`` and ``table_crps_main.tex``: complete ``table``
environments in the same shape as the ones in the manuscript, so they can be
pasted over the existing ones.

Formatting follows the paper: two decimal places, ``mean±std``, the best value
per row in bold (all of them when rounded values tie), ``---`` where a
dataset-method pair has no result, and the OpenML block above the UCI block in
the paper's row order.

Reads the same sources as make_figures.py, so the two never disagree.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("mf", HERE / "make_figures.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

METHODS = ["DGBM-LGB", "DGBM-XGB", "GPBoost", "NGBoost", "PGBM", "XLSF"]

OPENML_ORDER = [
    "Ailerons", "Bike_Sharing_Demand", "Brazilian_houses", "MiamiHousing2016",
    "abalone", "cpu_act", "diamonds", "elevators", "house_16H", "house_sales",
    "houses", "medical_charges", "nyc-taxi-green-dec-2016", "pol", "sulfur",
    "superconduct", "wine_quality", "yprop_4_1",
]
UCI_ORDER = [
    "Boston Housing", "Combined Cycle Power Plant", "Concrete Compression Strength",
    "Energy Efficiency", "Kin8nm", "Naval Propulsion", "Protein Structure",
    "Wine Quality Red", "Yacht Hydrodynamics", "Year Prediction MSD",
]

CAPTIONS = {
    "NLL": ("Negative log-likelihood (NLL, lower is better) of DGBM-LGB, DGBM-XGB, "
            "and benchmark methods on OpenML and UCI datasets (mean $\\pm$ std over "
            "folds). Best values per dataset in bold; ``---'' marks runs that "
            "exceeded their wall-clock limit (Section~\\ref{sec:reproducibility}); "
            "$^{\\ddagger}$ marks cells carried over from the earlier runs, whose "
            "result files were not preserved and whose regeneration is still in progress.",
            "tab:nll_main"),
    "CRPS": ("Continuous ranked probability score (CRPS, lower is better) of "
             "DGBM-LGB, DGBM-XGB, and benchmark methods on OpenML and UCI datasets "
             "(mean $\\pm$ std over folds). Best values per dataset in bold; "
             "``---'' marks runs that exceeded their wall-clock limit "
             "(Section~\\ref{sec:reproducibility}); $^{\\ddagger}$ marks cells carried "
             "over from the earlier runs, whose result files were not preserved "
             "and whose regeneration is still in progress.",
             "tab:crps_main"),
}


# Cells the released result files do not cover. Their runs were never saved and
# the regeneration jobs have not finished, so the previously published values are
# carried over and flagged, rather than silently shown as timeouts.
CARRIED = [
    ("openml", "nyc-taxi-green-dec-2016", "NGBoost"),
    ("openml", "nyc-taxi-green-dec-2016", "PGBM"),
]


def esc(name: str) -> str:
    return name.replace("_", r"\_")


def collect(metric: str) -> dict:
    """{suite: {dataset: {method: (mean, std) or None}}}"""
    mean_col, std_col = f"{metric}-mean", f"{metric}-std"
    out = {}
    for suite in ("openml", "uci"):
        table = {}
        frames = {}
        for method, rel in mf.SOURCES[suite].items():
            df = mf.load(rel if isinstance(rel, (list, tuple)) else mf.REPO_ROOT / rel)
            frames[method] = df
        order = OPENML_ORDER if suite == "openml" else UCI_ORDER
        for ds in order:
            row = {}
            for method in METHODS:
                df = frames.get(method)
                if df is None or ds not in df.index or mean_col not in df.columns:
                    row[method] = None
                    continue
                m = pd.to_numeric(pd.Series([df.loc[ds, mean_col]]), errors="coerce").iloc[0]
                s = (pd.to_numeric(pd.Series([df.loc[ds, std_col]]), errors="coerce").iloc[0]
                     if std_col in df.columns else float("nan"))
                row[method] = None if pd.isna(m) else (float(m), float(s))
            table[ds] = row
        out[suite] = table
    return out


def render_row(ds: str, row: dict, carried: set) -> str:
    vals = {m: v for m, v in row.items() if v is not None}
    best = min((round(v[0], 2) for v in vals.values()), default=None)
    cells = []
    for m in METHODS:
        v = row.get(m)
        if v is None:
            cells.append("---")
            continue
        mean, std = v
        txt = f"{mean:.2f}±{0.0 if pd.isna(std) else std:.2f}"
        if best is not None and round(mean, 2) == best:
            txt = rf"\textbf{{{txt}}}"
        if (ds, m) in carried:
            txt += r"$^{\ddagger}$"
        cells.append(txt)
    return f"      {esc(ds)} & " + " & ".join(cells) + r" \\"


def render_table(metric: str, data: dict) -> str:
    caption, label = CAPTIONS[metric]
    L = [
        r"\begin{table}[h!]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \adjustbox{max width=\textwidth}{",
        r"      \begin{tabular}{lrrrrrr}",
        r"      \toprule",
        r"      \textbf{Dataset} & " + " & ".join(rf"\textbf{{{m}}}" for m in METHODS) + r" \\",
        r"      \midrule",
        r"      \multicolumn{7}{c}{\textbf{OpenML Datasets}} \\",
        r"      \midrule",
    ]
    carried_o = {(d, m) for su, d, m in CARRIED if su == "openml"}
    carried_u = {(d, m) for su, d, m in CARRIED if su == "uci"}
    L += [render_row(ds, data["openml"][ds], carried_o) for ds in OPENML_ORDER]
    L += [
        r"      \midrule",
        r"      \multicolumn{7}{c}{\textbf{UCI Datasets}} \\",
        r"      \midrule",
    ]
    L += [render_row(ds, data["uci"][ds], carried_u) for ds in UCI_ORDER]
    L += [r"      \bottomrule", r"      \end{tabular}%", r"    }", r"\end{table}", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path,
                    default=mf.REPO_ROOT / "reproduce" / "tables")
    ap.add_argument("--carry-over", type=pathlib.Path, default=None,
                    help="JSON of previously published values, used only for the "
                         "cells listed in CARRIED")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    published = {}
    if args.carry_over and args.carry_over.exists():
        published = json.loads(args.carry_over.read_text())

    for metric in ("NLL", "CRPS"):
        data = collect(metric)
        for suite, ds, meth in CARRIED:
            if data[suite].get(ds, {}).get(meth) is None:
                v = published.get(metric, {}).get(suite, {}).get(ds, {}).get(meth)
                if v:
                    data[suite][ds][meth] = (float(v[0]), float(v[1]))
        path = args.out / f"table_{metric.lower()}_main.tex"
        path.write_text(render_table(metric, data))
        missing = [f"{s}/{d}/{m}"
                   for s in data for d in data[s] for m in METHODS
                   if data[s][d][m] is None]
        print(f"wrote {path.name}  ({len(missing)} empty cells)")
        for cell in missing:
            print(f"    --- {cell}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
