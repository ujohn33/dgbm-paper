#!/bin/bash
#SBATCH --output=./report/%A_uci_init_moment/slurm-%A_%a.out
#SBATCH --job-name=init_moment
#SBATCH --time=5-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=60G
#SBATCH --array=0-9

module load Python/3.11.3-GCCcore-12.3.0
module load IPython/8.14.0-GCCcore-12.3.0
module load jupyter-server/2.14.0-GCCcore-12.3.0
module load aiohttp/3.8.5-GCCcore-12.3.0
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0
module load openpyxl/3.1.2-GCCcore-12.3.0
module load LightGBM/4.5.0-foss-2023a
source $VSC_SCRATCH/distiboost_311/bin/activate
cd /scratch/brussel/105/vsc10528/LSSboost
mkdir -p logs/uci/predictions logs/uci/fold_metrics results/uci

# Marginal-MLE initialisation of the additive predictor; everything else is the
# configuration reported in the paper.
export DGBM_INIT=moment

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID variant=$VARIANT DGBM_INIT=$DGBM_INIT"
case "$VARIANT" in
  lgb) python -u uci/UCI_lssboost_single_run_HP.py   "$SLURM_ARRAY_TASK_ID" exp False None None False ;;
  xgb) python -u uci/UCI_xglssboost_HP_single_run.py "$SLURM_ARRAY_TASK_ID" exp False None None False ;;
  *)   echo "set VARIANT=lgb|xgb"; exit 2 ;;
esac
