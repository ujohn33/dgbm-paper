#!/bin/bash

#SBATCH --output=./report/%A_lsf_uci/slurm-%A_%a.out
#SBATCH --job-name=lsf_uci
#SBATCH --time=2-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --array=0-9
#SBATCH --mail-type=ALL

set -euo pipefail

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load IPython/8.14.0-GCCcore-12.3.0
module load jupyter-server/2.14.0-GCCcore-12.3.0
module load aiohttp/3.8.5-GCCcore-12.3.0
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0

LSF_VENV="${LSF_VENV:-/scratch/brussel/105/vsc10528/venvs/lsf_openml_311}"
RUN_SEED="${RUN_SEED:-123}"
DATASET_IDX="${SLURM_ARRAY_TASK_ID}"
export DATASET_IDX
UCI_WRITE_PREDICTIONS="${UCI_WRITE_PREDICTIONS:-0}"
export UCI_WRITE_PREDICTIONS
LSF_AUTO_INSTALL_EXCEL_DEPS="${LSF_AUTO_INSTALL_EXCEL_DEPS:-1}"

if [[ ! -f "${LSF_VENV}/bin/activate" ]]; then
  echo "[error] LSF venv not found at ${LSF_VENV}" >&2
  echo "[error] Create it and install deps: gluonts xgboost optuna scipy scikit-learn openpyxl xlrd pandas" >&2
  exit 1
fi

source "${LSF_VENV}/bin/activate"
unset PYTHONPATH || true
unset PYTHONHOME || true
unset PYTHONUSERBASE || true
export PYTHONNOUSERSITE=1

cd /scratch/brussel/105/vsc10528/LSSboost

if [[ "${LSF_AUTO_INSTALL_EXCEL_DEPS}" == "1" ]]; then
  case "${DATASET_IDX}" in
    1)
      python -m pip install --quiet --upgrade xlrd || true
      ;;
    2|5)
      python -m pip install --quiet --upgrade openpyxl || true
      ;;
  esac
fi

if ! python - <<'PY'
import importlib.util
import os
import pathlib
import sys

required = ["gluonts", "xgboost", "optuna", "pandas", "scipy", "sklearn", "pydantic"]
dataset_idx = int(os.environ.get("DATASET_IDX", "-1"))
# UCI dataset that uses pd.read_excel on .xls: 1=Concrete Compression Strength
if dataset_idx == 1:
    required.append("xlrd")
# UCI datasets that use pd.read_excel: 2=Energy Efficiency, 5=Combined Cycle Power Plant
if dataset_idx in {2, 5}:
    required.append("openpyxl")
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("Missing Python packages in active venv: " + ", ".join(missing))

import typing_extensions
from typing_extensions import Sentinel  # noqa: F401
import pydantic_core

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
PY
then
  echo "[error] Python environment preflight failed for ${LSF_VENV}" >&2
  echo "[error] Reinstall in this venv:" >&2
  echo "        python -m pip install --upgrade typing_extensions pydantic pydantic-core gluonts xgboost openpyxl xlrd" >&2
  exit 1
fi

echo "Running LSF UCI benchmark"
echo "  dataset_idx=${DATASET_IDX}"
echo "  run_seed=${RUN_SEED}"
echo "  write_predictions=${UCI_WRITE_PREDICTIONS}"
echo "  auto_install_excel_deps=${LSF_AUTO_INSTALL_EXCEL_DEPS}"
echo "  python=$(which python)"
echo "  lsf_venv=${LSF_VENV}"

python uci/UCI_lsf_HP_single_run.py "${DATASET_IDX}" "${RUN_SEED}"
