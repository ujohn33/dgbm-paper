#!/bin/bash

#SBATCH --output=./report/%A_unified_openml/slurm-%A_%a.out
#SBATCH --job-name=unified_openml
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --array=0-18
#SBATCH --mail-type=ALL

set -euo pipefail

# ------------------------------------------------------------------------------
# Configurable submission notes:
# - You can override SBATCH resources at submit time, e.g.:
#   sbatch --time=1-00:00:00 --cpus-per-task=4 --mem=32G reproduction/run_openml_benchmarks.sbatch
# - You can override runtime variables via --export (examples at bottom).
# ------------------------------------------------------------------------------

# -------- Optional module setup (adjust to your cluster) --------
module purge
try_module () {
  module load "$1" || echo "[warn] module not available: $1"
}

try_module "Python/3.11.3-GCCcore-12.3.0"
try_module "IPython/8.14.0-GCCcore-12.3.0"
try_module "LightGBM/4.5.0-foss-2023a"
try_module "jupyter-server/2.14.0-GCCcore-12.3.0"
try_module "aiohttp/3.8.5-GCCcore-12.3.0"
try_module "Optuna/3.5.0-foss-2023a"
try_module "SciPy-bundle/2023.07-gfbf-2023a"
try_module "Pillow/10.0.0-GCCcore-12.3.0"
try_module "openpyxl/3.1.2-GCCcore-12.3.0"

# Optional venv activation used in your existing SLURM scripts.
BENCHMARK_VENV="${BENCHMARK_VENV:-/scratch/brussel/105/vsc10528/distiboost_311}"
if [[ -f "${BENCHMARK_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${BENCHMARK_VENV}/bin/activate"
else
  echo "[warn] BENCHMARK_VENV activate script not found: ${BENCHMARK_VENV}/bin/activate"
  echo "[warn] Continuing without venv activation."
fi

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/brussel/105/vsc10528/LSSboost}"
cd "${PROJECT_ROOT}"

export OPENML_SKIP_PARQUET=true
export CUDA_VISIBLE_DEVICES=""

# -------- Configurable knobs --------
# Comma-separated profile names from:
#   python reproduction/run_openml_benchmarks.py --list-profiles
OPENML_PROFILES="${OPENML_PROFILES:-lssboost_no_natural,xglssboost_no_natural,pgbm_hp,gpboost_hp,ngboost_hp,lsf_gluonts}"
RUN_SEED="${RUN_SEED:-123}"
PYTHON_BIN_USER="${PYTHON_BIN:-}"
OPENML_DRY_RUN="${OPENML_DRY_RUN:-0}"
OPENML_MAX_RUNS="${OPENML_MAX_RUNS:-}"
OPENML_EXTRA_ARGS="${OPENML_EXTRA_ARGS:-}"
OPENML_TASK_IDS="${OPENML_TASK_IDS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}"
OPENML_AUTO_INSTALL_DEPS="${OPENML_AUTO_INSTALL_DEPS:-0}"
LSF_VENV="${LSF_VENV:-/scratch/brussel/105/vsc10528/venvs/lsf_openml_311}"

if [[ "${OPENML_PROFILES}" == *"lsf_gluonts"* ]]; then
  if [[ -f "${LSF_VENV}/bin/activate" ]]; then
    if declare -f deactivate >/dev/null 2>&1; then
      deactivate || true
    fi
    # shellcheck disable=SC1090
    source "${LSF_VENV}/bin/activate"
    # Avoid module-injected site-packages (notably old typing_extensions)
    # shadowing this dedicated LSF environment.
    unset PYTHONPATH
    if [[ -z "${PYTHON_BIN_USER}" ]]; then
      PYTHON_BIN="${LSF_VENV}/bin/python"
    fi
  else
    echo "[error] lsf_gluonts requested but LSF_VENV activate script not found: ${LSF_VENV}/bin/activate"
    exit 1
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_USER}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

# Map SLURM array index to task id using OPENML_TASK_IDS.
IFS=',' read -r -a TASKS <<< "${OPENML_TASK_IDS}"

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#TASKS[@]} )); then
  if (( ${#TASKS[@]} == 1 )); then
    echo "[warn] SLURM array index (${SLURM_ARRAY_TASK_ID}) exceeds OPENML_TASK_IDS size=1 (${OPENML_TASK_IDS})."
    echo "[warn] Falling back to TASK_IDX=SLURM_ARRAY_TASK_ID to avoid job failure."
    TASK_IDX="${SLURM_ARRAY_TASK_ID}"
  else
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} for TASKS size=${#TASKS[@]}"
    echo "OPENML_TASK_IDS=${OPENML_TASK_IDS}"
    exit 1
  fi
else
  TASK_IDX="${TASKS[$SLURM_ARRAY_TASK_ID]}"
fi

echo "Running unified OpenML benchmark"
echo "  task_idx=${TASK_IDX}"
echo "  profiles=${OPENML_PROFILES}"
echo "  seed=${RUN_SEED}"
echo "  python=${PYTHON_BIN}"
echo "  benchmark_venv=${BENCHMARK_VENV}"
echo "  lsf_venv=${LSF_VENV}"
echo "  project_root=${PROJECT_ROOT}"
echo "  dry_run=${OPENML_DRY_RUN}"
echo "  max_runs=${OPENML_MAX_RUNS:-<none>}"
echo "  extra_args=${OPENML_EXTRA_ARGS:-<none>}"
echo "  auto_install_deps=${OPENML_AUTO_INSTALL_DEPS}"

if [[ "${OPENML_PROFILES}" == *"lsf_gluonts"* ]]; then
  if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
from gluonts.ext.rotbaum._model import LSF
print(LSF)
PY
  then
    echo "[warn] lsf_gluonts requested but gluonts is not importable in current environment."
    if [[ "${OPENML_AUTO_INSTALL_DEPS}" == "1" ]]; then
      echo "[warn] Attempting automatic dependency install in active environment..."
      "${PYTHON_BIN}" -m pip install gluonts xgboost || true
    fi
    if ! "${PYTHON_BIN}" - <<'PY'
import sys, traceback
print("python_executable:", sys.executable)
print("python_version:", sys.version)
print("sys_path_head:", sys.path[:5])
try:
    from gluonts.ext.rotbaum._model import LSF
    print("gluonts_import_ok:", LSF)
except Exception as e:
    print("gluonts_import_error:", repr(e))
    traceback.print_exc()
    raise SystemExit(1)
PY
    then
      echo "[error] Could not import gluonts for lsf_gluonts profile."
      echo "[error] In this exact environment run: ${PYTHON_BIN} -m pip install gluonts xgboost"
      exit 1
    fi
  fi
fi

cmd=("${PYTHON_BIN}" reproduction/run_openml_benchmarks.py \
  --tasks "${TASK_IDX}" \
  --profiles "${OPENML_PROFILES}" \
  --seed "${RUN_SEED}" \
  --python "${PYTHON_BIN}")

if [[ "${OPENML_DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

if [[ -n "${OPENML_MAX_RUNS}" ]]; then
  cmd+=(--max-runs "${OPENML_MAX_RUNS}")
fi

if [[ -n "${OPENML_EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${OPENML_EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

echo "  command=${cmd[*]}"
"${cmd[@]}"

# Example (safe export style; preserves commas in OPENML_TASK_IDS):
# export OPENML_TASK_IDS="0,7,14"
# sbatch --array=0-2 \
#   --export=ALL,OPENML_TASK_IDS,OPENML_PROFILES="pgbm_hp,lsf_gluonts",RUN_SEED=123,OPENML_DRY_RUN=0 \
#   reproduction/run_openml_benchmarks.sbatch
