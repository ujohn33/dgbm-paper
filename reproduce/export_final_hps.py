#!/usr/bin/env python
"""Export the final selected hyperparameters to the LaTeX tables in the paper.

The tuning stage of the OpenML experiments writes one JSON file per dataset:

    logs/openml/lssboost/<natural|normal>/<mode>/<dataset>_opt_params.json   # DGBM-LGB
    logs/openml/xgboost/<natural|normal>/<mode>/<dataset>_opt_params.json    # DGBM-XGB

`normal` means natural-gradient updates disabled and `exp` is the response
function for the scale parameter -- that combination is the configuration
reported in the paper, so it is the default here.

On the UCI suite the search is repeated inside every fold and no per-dataset
JSON is written; see the README for why there is no UCI table.

Usage (from the repository root):

    python reproduce/export_final_hps.py --out reproduce/tables

Then \\input the generated .tex files in the supplementary section
"Final Selected Hyperparameters per Dataset".
"""
import argparse
import json
import pathlib
import sys

# (method label, log directory, columns to show, constant columns to fold into
#  the caption, LaTeX label)
METHODS = [
    (
        "DGBM-LGB",
        "logs/openml/lssboost/{grad}/{mode}",
        ["eta", "max_depth", "num_leaves", "min_data_in_leaf", "lambda_l1", "opt_rounds"],
        ["feature_pre_filter", "boosting", "device"],
        "tab:final_hps_lgb",
    ),
    (
        "DGBM-XGB",
        "logs/openml/xgboost/{grad}/{mode}",
        ["eta", "max_depth", "min_child_weight", "subsample", "reg_alpha", "opt_rounds"],
        ["booster", "device"],
        "tab:final_hps_xgb",
    ),
]


def esc(value) -> str:
    return str(value).replace("_", r"\_")


def fmt(value) -> str:
    if isinstance(value, bool):
        return r"\texttt{%s}" % value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return ("%.3g" % value).replace("e-0", "e-").replace("e+0", "e")
    return esc(value)


def load(directory: pathlib.Path) -> dict:
    rows = {}
    for path in sorted(directory.glob("*_opt_params.json")):
        dataset = path.name[: -len("_opt_params.json")]
        rows[dataset] = json.loads(path.read_text())
    return rows


def to_latex(method: str, rows: dict, columns: list, constants: list, label: str) -> str:
    fixed = {}
    for key in constants:
        values = {json.dumps(params.get(key)) for params in rows.values()}
        if len(values) == 1:
            fixed[key] = next(iter(rows.values())).get(key)

    caption = (
        f"Final hyperparameters selected by TPE for {esc(method)} on each OpenML "
        r"dataset. \texttt{opt\_rounds} is the number of boosting rounds returned "
        r"by the tuning stage (the upper bound was 2000)."
    )
    if fixed:
        caption += " " + ", ".join(
            r"\texttt{%s} was fixed to \texttt{%s}" % (esc(k), esc(v))
            for k, v in fixed.items()
        ) + " for all datasets."

    lines = [
        r"\begin{table}[h!]",
        r"  \centering",
        r"  \caption{%s}" % caption,
        r"  \label{%s}" % label,
        r"  \adjustbox{max width=\textwidth}{",
        r"  \begin{tabular}{l" + "r" * len(columns) + "}",
        r"  \toprule",
        r"  \textbf{Dataset} & "
        + " & ".join(r"\texttt{%s}" % esc(c) for c in columns)
        + r" \\",
        r"  \midrule",
    ]
    for dataset in sorted(rows, key=str.lower):
        params = rows[dataset]
        cells = [fmt(params[c]) if c in params else "--" for c in columns]
        lines.append("  " + esc(dataset) + " & " + " & ".join(cells) + r" \\")
    lines += [r"  \bottomrule", r"  \end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("reproduce/tables"))
    parser.add_argument(
        "--grad",
        default="normal",
        choices=["normal", "natural"],
        help="'normal' (reported configuration) or 'natural' (ablation)",
    )
    parser.add_argument("--mode", default="exp", choices=["exp", "softplus"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wrote = False
    for method, template, columns, constants, label in METHODS:
        directory = args.repo_root / template.format(grad=args.grad, mode=args.mode)
        if not directory.is_dir():
            print(f"skip {method}: {directory} does not exist", file=sys.stderr)
            continue
        rows = load(directory)
        if not rows:
            print(f"skip {method}: no *_opt_params.json in {directory}", file=sys.stderr)
            continue
        path = args.out / f"final_hps_{method.lower().replace('-', '_')}.tex"
        path.write_text(to_latex(method, rows, columns, constants, label))
        print(f"wrote {path} ({len(rows)} datasets)")
        wrote = True

    if not wrote:
        sys.exit("nothing written -- check --repo-root, --grad and --mode")


if __name__ == "__main__":
    main()
