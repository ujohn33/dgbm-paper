#!/bin/bash

#SBATCH --output=/scratch/brussel/105/vsc10528/LSSboost/report/%A_DGB-XGB_cuda_openml/slurm-%A_%a.out # STDOUT
#SBATCH --job-name=DGB-XGB_cuda_openml
#SBATCH --time=2-12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=90G
#SBATCH --array=0-18
#SBATCH --mail-type=ALL


# File to run on HPC
# First activate your python environment (you might have to create your own) with:
# virtualenv --system-site-packages $VSC_DATA/scenarioenv
# And then install the requirements with
# pip install -r requirements.txt

module load Python/3.11.3-GCCcore-12.3.0
module load IPython/8.14.0-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load jupyter-server/2.14.0-GCCcore-12.3.0
module load aiohttp/3.8.5-GCCcore-12.3.0
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0
module load openpyxl/3.1.2-GCCcore-12.3.0

source $VSC_SCRATCH/gpu_friendly/bin/activate

cd /scratch/brussel/105/vsc10528/LSSboost

export OPENML_SKIP_PARQUET=true
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OPENML_OUTPUT_BASE_DIR="${VSC_SCRATCH}/LSSboost"
export OPENML_BASE_DIR="${VSC_DATA:-$VSC_SCRATCH}/openml_runtime"
export TMPDIR="$OPENML_BASE_DIR/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export OPENML_CACHE_DIR="$OPENML_BASE_DIR/cache"
export OPENML_DEBUG_DIR="$OPENML_OUTPUT_BASE_DIR/logs/openml/debug"
export OPENML_FOLD_METRICS_DIR="$OPENML_OUTPUT_BASE_DIR/logs/openml/fold_metrics"
export OPENML_RESULTS_DIR="$OPENML_OUTPUT_BASE_DIR/results/openml"
mkdir -p \
  "$TMPDIR" \
  "$OPENML_CACHE_DIR" \
  "$OPENML_DEBUG_DIR" \
  "$OPENML_FOLD_METRICS_DIR" \
  "$OPENML_RESULTS_DIR" \
  "$OPENML_OUTPUT_BASE_DIR/report/${SLURM_ARRAY_JOB_ID}_DGB-XGB_cuda_openml"

# Fixed values
dataset=$SLURM_ARRAY_TASK_ID
mode="exp"
natural_grad="True"
stabilization="None"
clip_value="None"
standardize="False"
run_seed="123"
safety="False"
if [ "$natural_grad" = "True" ]; then
  export OPENML_XGB_OPT_PARAMS_DIR="$OPENML_OUTPUT_BASE_DIR/logs/openml/xgboost/natural/exp"
else
  export OPENML_XGB_OPT_PARAMS_DIR="$OPENML_OUTPUT_BASE_DIR/logs/openml/xgboost/normal/exp"
fi
mkdir -p "$OPENML_XGB_OPT_PARAMS_DIR"

# Echo parameters and dataset for logging
echo "Running DGB-XGB CUDA with dataset=$dataset, mode=$mode, natural_grad=$natural_grad, stabilization=$stabilization, clip_value=$clip_value, standardize=$standardize, run_seed=$run_seed, safety=$safety"
echo "Host=$(hostname) JOB=$SLURM_JOB_ID ARRAY_JOB=$SLURM_ARRAY_JOB_ID ARRAY_TASK=$SLURM_ARRAY_TASK_ID"
echo "PWD=$(pwd)"
echo "TMPDIR=$TMPDIR"
echo "OPENML_CACHE_DIR=$OPENML_CACHE_DIR"
echo "OPENML_BASE_DIR=$OPENML_BASE_DIR"
echo "OPENML_OUTPUT_BASE_DIR=$OPENML_OUTPUT_BASE_DIR"
echo "OPENML_DEBUG_DIR=$OPENML_DEBUG_DIR"
echo "OPENML_FOLD_METRICS_DIR=$OPENML_FOLD_METRICS_DIR"
echo "OPENML_RESULTS_DIR=$OPENML_RESULTS_DIR"
echo "Python=$(which python)"
df -h "$VSC_SCRATCH" || true
df -h "$HOME" || true
df -h "${VSC_DATA:-$VSC_SCRATCH}" || true
df -h "$VSC_SCRATCH" || true
nvidia-smi || true

GPU_INFO_PATH="$OPENML_OUTPUT_BASE_DIR/logs/openml/gpu_dgbxgb_info_${SLURM_ARRAY_JOB_ID}.txt"
GPU_USAGE_PATH="$OPENML_OUTPUT_BASE_DIR/logs/openml/gpu_dgbxgb_usage_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
mkdir -p "$(dirname "$GPU_INFO_PATH")"
nvidia-smi > "$GPU_INFO_PATH"
nvidia-smi pmon -o DT -s um > "$GPU_USAGE_PATH" &
GPU_MONITOR_PID=$!

python -u openml/openml_xglssboost_HP_single_run.py $dataset $mode $natural_grad $stabilization $clip_value $standardize $run_seed $safety
status=$?

kill "$GPU_MONITOR_PID" 2>/dev/null || true
wait "$GPU_MONITOR_PID" 2>/dev/null || true

exit $status
