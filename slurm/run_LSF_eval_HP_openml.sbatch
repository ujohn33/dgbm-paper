#!/bin/bash

#SBATCH --output=./report/%A_lsf_openml/slurm-%A_%a.out
#SBATCH --job-name=lsf_openml
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --array=10
#SBATCH --mail-type=ALL

set -euo pipefail

module purge
try_module () {
  module load "$1" || echo "[warn] module not available: $1"
}

try_module "Python/3.11.3-GCCcore-12.3.0"

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/brussel/105/vsc10528/LSSboost}"
LSF_VENV="${LSF_VENV:-/scratch/brussel/105/vsc10528/venvs/lsf_openml_311}"
RUN_SEED="${RUN_SEED:-123}"
TASK_IDX="${SLURM_ARRAY_TASK_ID}"
export TASK_IDX
OPENML_LSF_WRITE_PREDICTIONS="${OPENML_LSF_WRITE_PREDICTIONS:-0}"
export OPENML_LSF_WRITE_PREDICTIONS
OPENML_LSF_WRITE_OPT_PARAMS="${OPENML_LSF_WRITE_OPT_PARAMS:-0}"
export OPENML_LSF_WRITE_OPT_PARAMS
OPENML_AUTO_INSTALL_DEPS="${OPENML_AUTO_INSTALL_DEPS:-0}"

if [[ ! -f "${LSF_VENV}/bin/activate" ]]; then
  echo "[error] LSF venv not found at ${LSF_VENV}" >&2
  echo "[error] Create it and install deps: gluonts xgboost optuna scipy scikit-learn openml pandas numpy pydantic" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${LSF_VENV}/bin/activate"
unset PYTHONHOME || true
unset PYTHONUSERBASE || true
export PYTHONNOUSERSITE=1
export PYTHONWARNINGS="ignore::UserWarning:sklearn.utils.parallel,ignore::UserWarning:gluonts.json"

cd "${PROJECT_ROOT}"

export OPENML_SKIP_PARQUET=true
export CUDA_VISIBLE_DEVICES=""

if [[ "${OPENML_AUTO_INSTALL_DEPS}" == "1" ]]; then
  python -m pip install --quiet --upgrade gluonts xgboost openml optuna scipy scikit-learn pandas numpy pydantic || true
fi

if ! python - <<'PY'
import importlib.util
import pathlib
import sys

required = [
    "gluonts",
    "xgboost",
    "openml",
    "optuna",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "pydantic",
]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("Missing Python packages in active venv: " + ", ".join(missing))

import typing_extensions
from typing_extensions import Sentinel  # noqa: F401
import pydantic_core
from gluonts.ext.rotbaum._model import LSF  # noqa: F401

venv = pathlib.Path(sys.prefix).resolve()
te_path = pathlib.Path(typing_extensions.__file__).resolve()
pc_path = pathlib.Path(pydantic_core.__file__).resolve()
if venv not in te_path.parents:
    raise SystemExit(f"typing_extensions is not coming from venv: {te_path}")
if venv not in pc_path.parents:
    raise SystemExit(f"pydantic_core is not coming from venv: {pc_path}")

print("Dependency preflight OK")
print(f"typing_extensions={te_path}")
print(f"pydantic_core={pc_path}")
print(f"python={sys.executable}")
PY
then
  echo "[error] Python environment preflight failed for ${LSF_VENV}" >&2
  echo "[error] Reinstall in this venv:" >&2
  echo "        python -m pip install --upgrade typing_extensions pydantic pydantic-core gluonts xgboost openml optuna scipy scikit-learn pandas numpy" >&2
  exit 1
fi

echo "Running LSF OpenML benchmark"
echo "  task_idx=${TASK_IDX}"
echo "  run_seed=${RUN_SEED}"
echo "  write_predictions=${OPENML_LSF_WRITE_PREDICTIONS}"
echo "  write_opt_params=${OPENML_LSF_WRITE_OPT_PARAMS}"
echo "  auto_install_deps=${OPENML_AUTO_INSTALL_DEPS}"
echo "  python=$(which python)"
echo "  lsf_venv=${LSF_VENV}"
echo "  project_root=${PROJECT_ROOT}"

python -u openml/openml_lsf_HP_single_run.py "${TASK_IDX}" "${RUN_SEED}"
