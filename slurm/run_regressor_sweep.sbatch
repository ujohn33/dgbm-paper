#!/bin/bash
#SBATCH --output=./report/%A_regressor_sweep/slurm-%A_%a.out
#SBATCH --job-name=reg_sweep
#SBATCH --time=0-12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --array=0-4

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

echo "host=$(hostname) seed=${SLURM_ARRAY_TASK_ID}"
python -u reproduce/run_regressor_sweep.py --seed "${SLURM_ARRAY_TASK_ID}"
