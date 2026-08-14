#!/bin/bash
#SBATCH --output=./report/%A_rerun_gaps_openml/slurm-%A_%a.out
#SBATCH --job-name=rr_gaps
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G

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
mkdir -p logs/openml/predictions logs/openml/fold_metrics results/openml

case "$WHICH" in
  ngboost_superconduct)
    python -u openml/openml_ngboost_HP_single_run.py 15 123 ;;
  lgb_nyctaxi)
    python -u openml/openml_lssboost_HP_single_run.py 10 exp False None None False 123 False ;;
  *) echo "set WHICH"; exit 2 ;;
esac
