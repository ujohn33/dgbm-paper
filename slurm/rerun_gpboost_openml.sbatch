#!/bin/bash
#SBATCH --output=./report/%A_rerun_gpboost_openml/slurm-%A_%a.out
#SBATCH --job-name=rr_gpb_openml
#SBATCH --time=2-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=60G
#SBATCH --array=0-18

module load Python/3.11.3-GCCcore-12.3.0
module load IPython/8.14.0-GCCcore-12.3.0
module load jupyter-server/2.14.0-GCCcore-12.3.0
module load aiohttp/3.8.5-GCCcore-12.3.0
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0
module load openpyxl/3.1.2-GCCcore-12.3.0
source $VSC_SCRATCH/venvs/gpboost_repro/bin/activate
cd /scratch/brussel/105/vsc10528/LSSboost

mkdir -p logs/openml/predictions logs/openml/fold_metrics results/openml
export OPENML_CACHE_DIR="$VSC_SCRATCH/openml_cache"
mkdir -p "$OPENML_CACHE_DIR"

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID gpboost=$(python -c 'import gpboost;print(gpboost.__version__)')"
python -u openml/openml_gpboost_HP_single_run.py $SLURM_ARRAY_TASK_ID
