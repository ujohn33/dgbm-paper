#!/bin/bash

#SBATCH --output=./report/%A_LSSboost_LGBM_M5/slurm-%A_%a.out # STDOUT
#SBATCH --job-name=LSSboost_LGBM_M5
#SBATCH --time=1-12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=LSSboost_LGBM_M5
#SBATCH --partition=skylake,skylake_mpi
#SBATCH --gpus-per-node=0
#SBATCH --mem=100G
#SBATCH --array=13-15
#SBATCH --mail-type=ALL


# File to run on HPC
# First activate your python environment (you might have to create your own) with:
# virtualenv --system-site-packages $VSC_DATA/scenarioenv
# And then intall the requirements with
# pip install -r requirements.txt

module load Python/3.11.3-GCCcore-12.3.0
module load IPython/8.14.0-GCCcore-12.3.0
module load LightGBM/4.5.0-foss-2023a
module load jupyter-server/2.14.0-GCCcore-12.3.0
module load aiohttp/3.8.5-GCCcore-12.3.0
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0
module load Seaborn/0.13.2-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a
module load openpyxl/3.1.2-GCCcore-12.3.0

source $VSC_SCRATCH/distiboost_311/bin/activate

cd /scratch/brussel/105/vsc10528/LSSboost

python -u m5/m5_lgbm_winner.py --level $SLURM_ARRAY_TASK_ID