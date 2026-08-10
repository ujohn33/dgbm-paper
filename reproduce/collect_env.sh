#!/bin/bash
# Capture the environment for the record.
#
#     bash reproduce/collect_env.sh > reproduce/env_report.txt
#
# Probes whatever Python environment is currently active, so activate the one
# you want to describe first. On systems that layer an environment-module stack
# under a virtualenv, load the modules before activating, because the
# module-provided packages take precedence and are what actually gets imported.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

echo "===== Python packages (currently active environment) ====="
python3 - <<'PY'
import importlib, sys
print("  python:", sys.version.split()[0], "at", sys.executable)
for p in ["lightgbm", "lightgbmlss", "xgboost", "xgboostlss", "torch", "numpy",
          "pandas", "scipy", "sklearn", "ngboost", "pgbm", "gpboost", "gluonts",
          "optuna", "openml", "properscoring", "pyro"]:
    try:
        m = importlib.import_module(p)
        print(f"  {p}: {getattr(m, '__version__', 'installed (no __version__)')}")
    except Exception:
        print(f"  {p}: -")
PY

echo
echo "===== Pinned submodule commits ====="
git submodule status 2>/dev/null
for d in LightGBMLSS XGBoostLSS ngboost pgbm properscoring; do
  [ -e "$d/.git" ] || continue
  printf '  %-14s %s  %s\n' "$d" \
    "$(git -C "$d" rev-parse --short HEAD 2>/dev/null)" \
    "$(git -C "$d" describe --tags 2>/dev/null)"
done

echo
echo "===== Experiment settings read back from the code ====="
echo "--- TPE budgets, early stopping, boosting rounds ---"
grep -rn --include='*.py' -E 'n_trials|early_stopping_rounds|max_minutes|num_boost_round' \
  openml uci 2>/dev/null | grep -v '__pycache__'
echo "--- seeds ---"
grep -rn --include='*.py' -E 'run_seed|random_state|default_rng|seed_everything' \
  openml uci 2>/dev/null | grep -v '__pycache__'

echo
echo "===== Host ====="
uname -srm
python3 -c "import platform; print(' ', platform.platform())"
if command -v lscpu >/dev/null 2>&1; then
  lscpu | grep -E 'Model name|^CPU\(s\)|Socket|Thread'
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  echo "  no GPU detected"
fi
[ -r /etc/os-release ] && head -2 /etc/os-release

echo
echo "===== Scheduler (if this is a batch cluster) ====="
if command -v sinfo >/dev/null 2>&1; then
  sinfo -o "%20P %5D %14C %10m %20G" 2>/dev/null | head -15
  echo "--- wall-clock limits requested by the job scripts ---"
  grep -H -E 'SBATCH --(time|partition|gpus|mem)' slurm/*.sbatch 2>/dev/null | sed 's|slurm/||'
else
  echo "  no Slurm on this host"
fi
