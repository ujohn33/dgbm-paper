#!/usr/bin/env python3
"""Unified OpenML benchmarking runner.

This script mirrors the heterogeneous OpenML runs currently spread across
multiple SLURM files and executes them from one place.

Outputs are stored under:
  reproduce/runs/
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "reproduce" / "runs"


def parse_task_indices(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid task range: {part}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    unique_sorted = sorted(set(values))
    if not unique_sorted:
        raise ValueError("No task indices parsed from --tasks.")
    return unique_sorted


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


@dataclass(frozen=True)
class Profile:
    name: str
    script: str
    fixed_args: tuple[str, ...]
    supports_seed: bool = False
    include_by_default: bool = True
    notes: str = ""

    def command(self, task_idx: int, python_bin: str, run_seed: int) -> list[str]:
        cmd = [python_bin, self.script, str(task_idx), *self.fixed_args]
        if self.supports_seed:
            cmd.append(str(run_seed))
        return cmd


PROFILES: tuple[Profile, ...] = (
    # Mirrors run_LSS_eval_HP_openml.sbatch defaults.
    Profile(
        name="lssboost_no_natural",
        script="openml/openml_lssboost_HP_single_run.py",
        fixed_args=("exp", "False", "None", "None", "False"),
        supports_seed=True,
        notes="SLURM-like LSSBoost CPU profile.",
    ),
    Profile(
        name="lssboost_no_natural_mad",
        script="openml/openml_lssboost_HP_single_run.py",
        fixed_args=("exp", "False", "MAD", "None", "False"),
        supports_seed=True,
        include_by_default=False,
        notes="LSSBoost CPU profile with MAD stabilization.",
    ),
    # Mirrors run_LSS_clip_openml.sbatch variants.
    Profile(
        name="lssboost_natural",
        script="openml/openml_lssboost_HP_single_run.py",
        fixed_args=("exp", "True", "None", "None", "False"),
        supports_seed=True,
        notes="SLURM-like LSSBoost natural-gradient profile.",
    ),
    # Mirrors run_LSS-XGB_clip_openml.sbatch variants.
    Profile(
        name="xglssboost_no_natural",
        script="openml/openml_xglssboost_HP_single_run.py",
        fixed_args=("exp", "False", "None", "None", "False"),
        supports_seed=True,
        notes="SLURM-like XGBoostLSS profile.",
    ),
    Profile(
        name="xglssboost_natural",
        script="openml/openml_xglssboost_HP_single_run.py",
        fixed_args=("exp", "True", "None", "None", "False"),
        supports_seed=True,
        notes="SLURM-like XGBoostLSS natural-gradient profile.",
    ),
    # Updated PGBM HP script (seeded).
    Profile(
        name="pgbm_hp",
        script="openml/openml_pgbm_HP_single_run.py",
        fixed_args=(),
        supports_seed=True,
        notes="Current PGBM HP script.",
    ),
    # Historical PGBM NLL profile from run_PGBM_gpu_HP_openml.sbatch.
    Profile(
        name="pgbm_nll_hp",
        script="openml/openml_pgbm_NLL_HP_single_run.py",
        fixed_args=(),
        supports_seed=False,
        include_by_default=False,
        notes="Legacy PGBM NLL HP script (no CLI seed).",
    ),
    Profile(
        name="gpboost_hp",
        script="openml/openml_gpboost_HP_single_run.py",
        fixed_args=(),
        supports_seed=True,
        notes="GPBoost OpenML profile.",
    ),
    Profile(
        name="ngboost_hp",
        script="openml/openml_ngboost_HP_single_run.py",
        fixed_args=(),
        supports_seed=True,
        notes="NGBoost OpenML profile.",
    ),
    Profile(
        name="lsf_gluonts",
        script="openml/openml_lsf_HP_single_run.py",
        fixed_args=(),
        supports_seed=True,
        include_by_default=False,
        notes="Exact package implementation from gluonts.ext.rotbaum (LSF = QRX).",
    ),
    # Archive SLURM profiles (optional).
    Profile(
        name="lgbm_hp",
        script="openml/openml_LGBM_HP.py",
        fixed_args=(),
        supports_seed=False,
        include_by_default=False,
        notes="Archive LightGBM HP script.",
    ),
    Profile(
        name="autogluon_hp",
        script="openml/openml_autogluon_HP_single_run.py",
        fixed_args=(),
        supports_seed=False,
        include_by_default=False,
        notes="Archive AutoGluon HP script.",
    ),
)


def get_profile_map() -> dict[str, Profile]:
    return {profile.name: profile for profile in PROFILES}


def default_profile_names() -> list[str]:
    return [p.name for p in PROFILES if p.include_by_default]


PROFILE_ARTIFACT_GLOBS: dict[str, tuple[str, ...]] = {
    "lssboost_no_natural": (
        "results/openml/openml_LSSboost_no_natural_exp_None.csv",
        "logs/openml/predictions/LSSboost_no_natural_exp_None.csv",
        "logs/openml/lssboost/normal/exp/*.json",
    ),
    "lssboost_no_natural_mad": (
        "results/openml/openml_LSSboost_no_natural_exp_MAD.csv",
        "logs/openml/predictions/LSSboost_no_natural_exp_MAD.csv",
        "logs/openml/lssboost/normal/exp/*.json",
    ),
    "lssboost_natural": (
        "results/openml/openml_LSSboost_natural_exp_None.csv",
        "logs/openml/predictions/LSSboost_natural_exp_None.csv",
        "logs/openml/lssboost/natural/exp/*.json",
    ),
    "xglssboost_no_natural": (
        "results/openml/openml_XGBoostLSS_no_natural_exp_None.csv",
        "logs/openml/predictions/XGBoostLSS_no_natural_exp_None_exp_None.csv",
        "logs/openml/xgboost/normal/exp/*.json",
    ),
    "xglssboost_natural": (
        "results/openml/openml_XGBoostLSS_natural_exp_None.csv",
        "logs/openml/predictions/XGBoostLSS_natural_exp_None_exp_None.csv",
        "logs/openml/xgboost/natural/exp/*.json",
    ),
    "pgbm_hp": (
        "results/openml/openml_PGBM.csv",
    ),
    "pgbm_nll_hp": (
        "results/openml/openml_PGBM.csv",
    ),
    "gpboost_hp": (
        "results/openml/openml_GPboost.csv",
        "logs/openml/predictions/GPboost.csv",
    ),
    "ngboost_hp": (
        "logs/openml/openml_NGBoost_natural.csv",
        "logs/openml/openml_NGBoost_no_natural.csv",
        "logs/openml/ngboost/*/exp/*.json",
    ),
    "lsf_gluonts": (
        "results/openml/openml_GluonTS_LSF.csv",
        "logs/openml/predictions/GluonTS_LSF.csv",
        "logs/openml/lsf/*.json",
    ),
    "lgbm_hp": (
        "results/openml/openml_LGBM.csv",
    ),
    "autogluon_hp": (
        "results/openml/openml_autogluon.csv",
    ),
}


def profile_artifact_paths(profile_name: str) -> list[Path]:
    patterns = PROFILE_ARTIFACT_GLOBS.get(profile_name, ())
    out: list[Path] = []
    for pattern in patterns:
        out.extend(path for path in REPO_ROOT.glob(pattern) if path.is_file())
    return sorted(set(out))


def snapshot_profile_artifacts(profile_name: str) -> dict[Path, float]:
    return {path: path.stat().st_mtime for path in profile_artifact_paths(profile_name)}


def changed_profile_artifacts(profile_name: str, before: dict[Path, float]) -> list[Path]:
    changed: list[Path] = []
    for path in profile_artifact_paths(profile_name):
        mtime = path.stat().st_mtime
        if path not in before or mtime > before[path]:
            changed.append(path)
    return sorted(changed)


def copy_artifacts(paths: list[Path], destination_root: Path) -> None:
    for src in paths:
        rel = src.relative_to(REPO_ROOT)
        dst = destination_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def ensure_dirs() -> dict[str, Path]:
    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = RESULTS_ROOT / f"run_{run_ts}"
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return {"run_root": run_root, "logs": logs_dir}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified OpenML benchmark launcher based on existing SLURM workflows."
    )
    parser.add_argument(
        "--tasks",
        default="0-18",
        help="Task index selection, e.g. '0-18' or '0,2,5-7'. Default: 0-18",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(default_profile_names()),
        help=(
            "Comma-separated profile names to run. "
            "Use --list-profiles to inspect all available options."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Global run seed passed to profiles that support CLI seeding.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for launched benchmarks.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on total launched runs (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit.",
    )

    args = parser.parse_args()
    profile_map = get_profile_map()

    if args.list_profiles:
        print("Available profiles:")
        for profile in PROFILES:
            default_mark = " [default]" if profile.include_by_default else ""
            seed_mark = "seeded" if profile.supports_seed else "unseeded"
            print(f"- {profile.name}{default_mark}: {profile.script} ({seed_mark})")
            if profile.notes:
                print(f"  notes: {profile.notes}")
        return 0

    try:
        task_indices = parse_task_indices(args.tasks)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    selected_profile_names = [p.strip() for p in args.profiles.split(",") if p.strip()]
    if not selected_profile_names:
        print("[error] No profiles selected.", file=sys.stderr)
        return 2

    unknown = [name for name in selected_profile_names if name not in profile_map]
    if unknown:
        print(f"[error] Unknown profiles: {', '.join(unknown)}", file=sys.stderr)
        print("Use --list-profiles to see valid names.", file=sys.stderr)
        return 2

    selected_profiles = [profile_map[name] for name in selected_profile_names]
    combinations: list[tuple[Profile, int]] = [
        (profile, task_idx)
        for profile in selected_profiles
        for task_idx in task_indices
    ]

    if args.max_runs is not None:
        combinations = combinations[: args.max_runs]

    dirs = ensure_dirs()
    run_root = dirs["run_root"]
    logs_dir = dirs["logs"]

    summary_path = run_root / "summary.csv"
    metadata_path = run_root / "run_metadata.json"

    metadata = {
        "created_at": dt.datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "tasks": task_indices,
        "profiles": selected_profile_names,
        "seed": args.seed,
        "python": args.python,
        "dry_run": args.dry_run,
        "max_runs": args.max_runs,
        "planned_runs": len(combinations),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fieldnames = [
        "run_id",
        "profile",
        "task_idx",
        "command",
        "status",
        "return_code",
        "duration_sec",
        "log_file",
        "artifact_count",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f_summary:
        writer = csv.DictWriter(f_summary, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (profile, task_idx) in enumerate(combinations, start=1):
            run_id = f"{idx:04d}_{slugify(profile.name)}_task{task_idx}"
            cmd = profile.command(task_idx=task_idx, python_bin=args.python, run_seed=args.seed)
            cmd_str = " ".join(cmd)
            log_path = logs_dir / f"{run_id}.log"

            print(f"[{idx}/{len(combinations)}] {run_id}")
            print(f"  $ {cmd_str}")

            start = time.time()
            status = "dry_run"
            return_code = 0
            artifact_count = 0

            if not args.dry_run:
                with log_path.open("w", encoding="utf-8") as log_f:
                    proc = subprocess.run(
                        cmd,
                        cwd=REPO_ROOT,
                        env={**os.environ, "OPENML_SKIP_PARQUET": "true"},
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        text=True,
                )
                return_code = proc.returncode
                status = "ok" if return_code == 0 else "failed"
            else:
                log_path.write_text("[dry-run] command not executed\n", encoding="utf-8")

            duration = time.time() - start
            writer.writerow(
                {
                    "run_id": run_id,
                    "profile": profile.name,
                    "task_idx": task_idx,
                    "command": cmd_str,
                    "status": status,
                    "return_code": return_code,
                    "duration_sec": f"{duration:.3f}",
                    "log_file": str(log_path.relative_to(run_root)),
                    "artifact_count": artifact_count,
                }
            )
            f_summary.flush()

    print()
    print(f"Run complete. Outputs saved in: {run_root}")
    print(f"- summary: {summary_path}")
    print(f"- metadata: {metadata_path}")
    print(f"- logs: {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
