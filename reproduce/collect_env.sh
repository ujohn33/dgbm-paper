#!/bin/bash
# Environment capture for the DGBM paper's Reproducibility section.
#
# Run from the repository root on VSC Hydra:
#     bash reproduce/collect_env.sh > reproduce/env_report.txt
#
# It probes each of the module+venv combinations that the Slurm jobs actually
# use, because the EasyBuild modules take precedence over the same packages
# installed inside the venvs -- probing the venv alone reports the wrong
# versions.

SCRATCH="${VSC_SCRATCH:-/scratch/brussel/105/vsc10528}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

probe() {
  # Probe from /tmp: the repository root contains directories named `openml/`
  # and `ngboost/`, which otherwise shadow the installed packages.
  cd /tmp || return
  python - <<'PY'
import importlib, sys
print("  python:", sys.version.split()[0])
for p in ["lightgbm", "lightgbmlss", "xgboost", "xgboostlss", "torch", "numpy",
          "pandas", "scipy", "sklearn", "ngboost", "pgbm", "gpboost", "gluonts",
          "optuna", "openml", "properscoring", "pyro"]:
    try:
        m = importlib.import_module(p)
        print(f"  {p}: {getattr(m, '__version__', 'installed (no __version__)')}")
    except Exception:
        print(f"  {p}: -")
PY
}

echo "===== A. distiboost_311 : DGBM-LGB, DGBM-XGB (CPU), NGBoost ====="
( module purge >/dev/null 2>&1
  module load Python/3.11.3-GCCcore-12.3.0 IPython/8.14.0-GCCcore-12.3.0 \
              jupyter-server/2.14.0-GCCcore-12.3.0 aiohttp/3.8.5-GCCcore-12.3.0 \
              LightGBM/4.5.0-foss-2023a Optuna/3.5.0-foss-2023a \
              SciPy-bundle/2023.07-gfbf-2023a Pillow/10.0.0-GCCcore-12.3.0 \
              openpyxl/3.1.2-GCCcore-12.3.0 >/dev/null 2>&1
  source "$SCRATCH/distiboost_311/bin/activate" 2>/dev/null && probe )

echo "===== B. gpu_friendly : PGBM, CUDA benchmarks ====="
( module purge >/dev/null 2>&1
  module load Python/3.11.3-GCCcore-12.3.0 IPython/8.14.0-GCCcore-12.3.0 \
              jupyter-server/2.14.0-GCCcore-12.3.0 aiohttp/3.8.5-GCCcore-12.3.0 \
              PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 \
              LightGBM/4.5.0-foss-2023a-CUDA-12.1.1 \
              Optuna/3.5.0-foss-2023a Pillow/10.0.0-GCCcore-12.3.0 >/dev/null 2>&1
  source "$SCRATCH/gpu_friendly/bin/activate" 2>/dev/null && probe )

echo "===== C. lsf_openml_311 : XLSF / GluonTS ====="
( module purge >/dev/null 2>&1
  module load Python/3.11.3-GCCcore-12.3.0 >/dev/null 2>&1
  source "$SCRATCH/venvs/lsf_openml_311/bin/activate" 2>/dev/null && probe )

echo "===== Pinned submodule commits ====="
cd "$REPO" || exit 1
git submodule status 2>/dev/null
for d in LightGBMLSS XGBoostLSS ngboost; do
  [ -d "$d/.git" ] || [ -f "$d/.git" ] || continue
  printf '  %-14s %s  %s\n' "$d" \
    "$(git -C "$d" rev-parse --short HEAD 2>/dev/null)" \
    "$(git -C "$d" describe --tags 2>/dev/null)"
done

echo "===== Slurm wall-clock limits per job script ====="
grep -H -E "SBATCH --(time|partition|gpus|mem)" slurm/*.sbatch | sed 's|slurm/||'

echo "===== TPE budgets and early stopping (grepped from the scripts) ====="
grep -rn --include="*.py" -E "n_trials|early_stopping_rounds|max_minutes|num_boost_round" \
  openml uci | grep -v "__pycache__"

echo "===== Seeds (grepped from the scripts) ====="
grep -rn --include="*.py" -E "run_seed|random_state|default_rng|seed_everything" \
  openml uci | grep -v "__pycache__"

echo "===== Hardware of the partitions used ====="
sinfo -o "%20P %5D %14C %10m %20G" 2>/dev/null
for p in zen4 zen5_mpi ampere_gpu; do
  echo "[$p]"
  scontrol show partition "$p" 2>/dev/null | grep -E "PartitionName|Nodes=|MaxTime"
done
echo "(CPU model per partition: run e.g. 'srun -p zen4 -n1 -t 00:02:00 lscpu | grep \"Model name\"')"

echo "===== OS ====="
head -3 /etc/os-release
