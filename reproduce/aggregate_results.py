#!/usr/bin/env python
"""Merge the per-job result CSVs into one long-format table per benchmark suite.

Each Slurm array job writes its own CSV under `results/openml/` or
`results/uci/`, and several jobs cover overlapping subsets of datasets. This
script parses the configuration out of every filename, merges the rows, and
makes duplicate coverage explicit instead of silently picking one file.

Filename conventions
--------------------
    openml_<model>_<natural|no_natural>_<mode>_<stab>[_std_<bool>][_safety_<bool>][_job_<id>].csv
    uci_<model>_<natural|no_natural>_<mode>_<stab>[...].csv

Usage (from the repository root):

    python reproduce/aggregate_results.py --out reproduce/tables
    python reproduce/aggregate_results.py --config no_natural/exp/None --out reproduce/tables

Outputs
-------
    <out>/results_openml_long.csv   all runs, one row per (config, dataset)
    <out>/results_uci_long.csv
    <out>/duplicates.csv            (config, dataset) pairs covered by >1 job
"""
import argparse
import csv
import pathlib
import re
import sys
from collections import defaultdict

JOB_RE = re.compile(r"_job_(\d+)$")
BOOL_RE = re.compile(r"^(std|safety)_(True|False)$")
STABILIZATIONS = {"None", "L2", "MAD"}
MODES = {"exp", "softplus"}


def parse_name(stem: str) -> dict:
    """Turn a result filename stem into a configuration dictionary."""
    meta = {
        "model": None,
        "natural": None,
        "mode": None,
        "stabilization": None,
        "standardize": None,
        "safety": None,
        "job": None,
    }
    job = JOB_RE.search(stem)
    if job:
        meta["job"] = job.group(1)
        stem = stem[: job.start()]

    parts = stem.split("_")
    if parts and parts[0] in ("openml", "uci"):
        parts = parts[1:]

    model_parts = []
    i = 0
    while i < len(parts):
        token = parts[i]
        if token == "no" and i + 1 < len(parts) and parts[i + 1] == "natural":
            meta["natural"] = False
            i += 2
            continue
        if token == "natural":
            meta["natural"] = True
            i += 1
            continue
        if token in MODES and meta["natural"] is not None:
            meta["mode"] = token
            i += 1
            continue
        if token in STABILIZATIONS and meta["mode"] is not None:
            if meta["stabilization"] is None:
                meta["stabilization"] = token
            i += 1
            continue
        flag = BOOL_RE.match("_".join(parts[i : i + 2]))
        if flag:
            meta["standardize" if flag.group(1) == "std" else "safety"] = (
                flag.group(2) == "True"
            )
            i += 2
            continue
        if meta["natural"] is None:
            model_parts.append(token)
        i += 1

    meta["model"] = "_".join(model_parts) or None
    return meta


def config_key(meta: dict) -> str:
    natural = {True: "natural", False: "no_natural", None: "?"}[meta["natural"]]
    return "/".join(
        [
            str(meta["model"]),
            natural,
            str(meta["mode"]),
            str(meta["stabilization"]),
        ]
    )


def collect(directory: pathlib.Path, suite: str, wanted: str | None):
    rows = []
    coverage = defaultdict(set)
    for path in sorted(directory.glob("*.csv")):
        meta = parse_name(path.stem)
        key = config_key(meta)
        if wanted and not key.endswith(wanted):
            continue
        with path.open(newline="") as handle:
            for record in csv.DictReader(handle):
                dataset = record.get("dset")
                if not dataset:
                    continue
                coverage[(key, dataset)].add(path.name)
                rows.append(
                    {
                        "suite": suite,
                        "config": key,
                        "model": meta["model"],
                        "natural_gradient": meta["natural"],
                        "mode": meta["mode"],
                        "stabilization": meta["stabilization"],
                        "standardize": meta["standardize"],
                        "safety": meta["safety"],
                        "job": meta["job"],
                        "source_file": path.name,
                        **record,
                    }
                )
    return rows, coverage


def write_csv(path: pathlib.Path, rows: list) -> None:
    if not rows:
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("reproduce/tables"))
    parser.add_argument(
        "--config",
        default=None,
        help="keep only configurations whose key ends with this, "
        "e.g. 'no_natural/exp/None' (the configuration reported in the paper)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    duplicates = []
    found_any = False
    for suite in ("openml", "uci"):
        directory = args.repo_root / "results" / suite
        if not directory.is_dir():
            print(f"skip {suite}: {directory} does not exist", file=sys.stderr)
            continue
        rows, coverage = collect(directory, suite, args.config)
        if rows:
            found_any = True
        write_csv(args.out / f"results_{suite}_long.csv", rows)
        for (key, dataset), files in sorted(coverage.items()):
            if len(files) > 1:
                duplicates.append(
                    {
                        "suite": suite,
                        "config": key,
                        "dataset": dataset,
                        "n_files": len(files),
                        "files": ";".join(sorted(files)),
                    }
                )

    if duplicates:
        write_csv(args.out / "duplicates.csv", duplicates)
        print(
            f"\n{len(duplicates)} (config, dataset) pairs are covered by more than one "
            "job file. Resolve them explicitly before building a paper table.",
            file=sys.stderr,
        )
    if not found_any:
        sys.exit("no result rows found -- check --repo-root and --config")


if __name__ == "__main__":
    main()
