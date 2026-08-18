#!/usr/bin/env python3
"""Aggregate the stabilization/natural-gradient ablation into one table.

    python reproduce/export_stabilization.py --out reproduce/tables

For each suite and backend, reports mean NLL and CRPS per (natural gradient,
stabilization) configuration over the datasets that every configuration of
that suite and backend completed, so all cells within a block are directly
comparable. UCI covers {None, L2, MAD}; OpenML runs cover {None, MAD}. Where
several job files exist for one configuration, earlier entries in the priority
list win, matching make_figures.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("mf", HERE / "make_figures.py")
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)
R = mf.REPO_ROOT

# (suite, backend) -> {(natural, stabilization): source spec}
SOURCES = {
    ("uci", "DGBM-LGB"): {
        ("No", "None"): "results/uci/uci_LSSboost_no_natural_exp_None.csv",
        ("No", "L2"):   "results/uci/uci_LSSboost_no_natural_exp_L2.csv",
        ("No", "MAD"):  "results/uci/uci_LSSboost_no_natural_exp_MAD.csv",
        ("Yes", "None"): "results/uci/uci_LSSboost_natural_exp_None.csv",
        ("Yes", "L2"):  "results/uci/uci_LSSboost_natural_exp_L2.csv",
        ("Yes", "MAD"): "results/uci/uci_LSSboost_natural_exp_MAD.csv",
    },
    ("uci", "DGBM-XGB"): {
        ("No", "None"): "results/uci/uci_XGLSSboost_no_natural_exp_None.csv",
        ("No", "L2"):   "results/uci/uci_XGLSSboost_no_natural_exp_L2.csv",
        ("No", "MAD"):  "results/uci/uci_XGLSSboost_no_natural_exp_MAD.csv",
        ("Yes", "None"): "results/uci/uci_XGLSSboost_natural_exp_None.csv",
        ("Yes", "L2"):  "results/uci/uci_XGLSSboost_natural_exp_L2.csv",
        ("Yes", "MAD"): "results/uci/uci_XGLSSboost_natural_exp_MAD.csv",
    },
    ("openml", "DGBM-LGB"): {
        ("No", "None"): [
            "results/openml/openml_LSSboost_no_natural_exp_None_std_False_safety_False_job_11752126.csv",
            "results/openml/openml_LSSboost_no_natural_exp_None_std_False_safety_False_job_13033317.csv",
            "results/openml/openml_LSSboost_no_natural_exp_None_std_False_safety_False_job_11750058.csv",
            "results/openml/openml_LSSboost_no_natural_exp_None_std_False_safety_False_job_11732371.csv",
        ],
        ("No", "MAD"): "results/openml/openml_LSSboost_no_natural_exp_MAD_std_False_safety_False_job_11732986.csv",
        ("Yes", "None"): [
            "results/openml/openml_LSSboost_natural_exp_None_std_False_safety_False_job_11750069.csv",
            "results/openml/openml_LSSboost_natural_exp_None_std_False_safety_False_job_11732546.csv",
        ],
        ("Yes", "MAD"): [
            "results/openml/openml_LSSboost_natural_exp_MAD_std_False_safety_False_job_11757226.csv",
            "results/openml/openml_LSSboost_natural_exp_MAD_std_False_safety_False_job_11733005.csv",
        ],
    },
    ("openml", "DGBM-XGB"): {
        ("No", "None"): [
            "results/openml/openml_XGBoostLSS_no_natural_exp_None_safety_False_job_11752710.csv",
            "results/openml/openml_XGBoostLSS_no_natural_exp_None_safety_False_job_11733821.csv",
        ],
        ("No", "MAD"): "results/openml/openml_XGBoostLSS_no_natural_exp_MAD_safety_False_job_11749986.csv",
        ("Yes", "None"): "results/openml/openml_XGBoostLSS_natural_exp_None_safety_False_job_11733823.csv",
        ("Yes", "MAD"): "results/openml/openml_XGBoostLSS_natural_exp_MAD_safety_False_job_11749999.csv",
    },
}

CONFIGS = [("No", "None"), ("No", "L2"), ("No", "MAD"),
           ("Yes", "None"), ("Yes", "L2"), ("Yes", "MAD")]


def _load(spec):
    """mf.load, with a repair pass for files where two rows share one line.

    A handful of ablation CSVs were appended without a trailing newline, so a
    record occasionally starts in the middle of the previous line. Rows are
    re-split on the known dataset names before parsing.
    """
    import io

    try:
        return mf.load([R / p for p in spec] if isinstance(spec, list) else R / spec)
    except Exception:
        pass
    frames = []
    for path in (spec if isinstance(spec, list) else [spec]):
        lines = (R / path).read_text().splitlines()
        n_cols = lines[0].count(",") + 1
        # a row that was appended twice without a newline carries extra fields;
        # keep the first record on the line
        lines = [",".join(l.split(",")[:n_cols]) for l in lines]
        df = pd.read_csv(io.StringIO("\n".join(lines)))
        df["dset"] = (df["dset"].astype(str).str.strip()
                      .replace({"Year Prediciton MSD": "Year Prediction MSD"}))
        frames.append(df.drop_duplicates(subset="dset", keep="first"))
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset="dset", keep="first").set_index("dset")


def block(suite, backend):
    frames = {}
    for cfg, spec in SOURCES[(suite, backend)].items():
        df = _load(spec)
        if df is not None:
            frames[cfg] = df
    common = None
    for df in frames.values():
        idx = set(df.index)
        common = idx if common is None else common & idx
    common = sorted(common or [])
    rows = {}
    for cfg, df in frames.items():
        sub = df.loc[common]
        # medians: a single diverged dataset dominates the mean (values up to
        # 1e19 occur), so the mean describes no typical run
        rows[cfg] = (pd.to_numeric(sub["NLL-mean"], errors="coerce").median(),
                     pd.to_numeric(sub["CRPS-mean"], errors="coerce").median())
    return rows, len(common)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=R / "reproduce" / "tables")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    blocks, counts = {}, {}
    for key in SOURCES:
        blocks[key], counts[key] = block(*key)
        print(f"{key}: {counts[key]} common datasets")
        for cfg in CONFIGS:
            if cfg in blocks[key]:
                n, c = blocks[key][cfg]
                print(f"    natural={cfg[0]:<4} stab={cfg[1]:<5} NLL {n:8.3f}  CRPS {c:8.3f}")

    L = [r"\begin{table}[h!]", r"    \centering",
         r"    \caption{Stabilization and natural-gradient ablation: median "
         r"NLL and CRPS over the datasets that every configuration of a "
         r"block completed "
         rf"(UCI: {counts[('uci','DGBM-LGB')]} datasets for DGBM-LGB and "
         rf"{counts[('uci','DGBM-XGB')]} for DGBM-XGB; OpenML: "
         rf"{counts[('openml','DGBM-LGB')]} and "
         rf"{counts[('openml','DGBM-XGB')]}). Lower is better; best per "
         r"column within each suite in bold. The OpenML ablation covered "
         r"L2 normalization only on UCI. The paper's headline configuration "
         r"is no natural gradient with no stabilization.}",
         r"    \label{tab:stabilization_ablation}",
         r"    \adjustbox{max width=\textwidth}{",
         r"      \begin{tabular}{llrrrr}", r"      \toprule",
         r"      & & \multicolumn{2}{c}{\textbf{DGBM-LGB}} & "
         r"\multicolumn{2}{c}{\textbf{DGBM-XGB}} \\",
         r"      \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"      \textbf{Natural} & \textbf{Stab.} & \textbf{NLL} & "
         r"\textbf{CRPS} & \textbf{NLL} & \textbf{CRPS} \\"]
    for suite, label in (("uci", "UCI"), ("openml", "OpenML")):
        L += [r"      \midrule",
              rf"      \multicolumn{{6}}{{c}}{{\textbf{{{label}}}}} \\",
              r"      \midrule"]
        cols = {}
        for bi, backend in enumerate(("DGBM-LGB", "DGBM-XGB")):
            for mi in range(2):
                vals = [blocks[(suite, backend)][c][mi]
                        for c in CONFIGS if c in blocks[(suite, backend)]]
                cols[(bi, mi)] = min(vals)
        for cfg in CONFIGS:
            if not any(cfg in blocks[(suite, b)] for b in ("DGBM-LGB", "DGBM-XGB")):
                continue
            cells = []
            for bi, backend in enumerate(("DGBM-LGB", "DGBM-XGB")):
                v = blocks[(suite, backend)].get(cfg)
                for mi in range(2):
                    if v is None:
                        cells.append("---")
                        continue
                    txt = f"{v[mi]:.3f}"
                    if abs(v[mi] - cols[(bi, mi)]) < 1e-9:
                        txt = rf"\textbf{{{txt}}}"
                    cells.append(txt)
            L.append(f"      {cfg[0]} & {cfg[1]} & " + " & ".join(cells) + r" \\")
    L += [r"      \bottomrule", r"      \end{tabular}%", r"    }", r"\end{table}", ""]
    path = args.out / "table_stabilization_ablation.tex"
    path.write_text("\n".join(L))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
