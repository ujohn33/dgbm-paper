#!/bin/bash

#SBATCH --output=./report/%A_gpboost_uci/slurm-%A_%a.out # STDOUT

#SBATCH --job-name=gpboost_UCI

#SBATCH --time=2-12:00:00

#SBATCH --ntasks-per-node=1

#SBATCH --partition=skylake,skylake_mpi

#SBATCH --mem-per-cpu=50G

#SBATCH --array=3-7,9
#SBATCH --mail-type=ALL


# File to run on HPC
# First activate your python environment (you might have to create your own) with:
# virtualenv --system-site-packages $VSC_DATA/scenarioenv
# And then intall the requirements with
# pip install -r requirements.txt

module load Python/3.10.4-GCCcore-11.3.0
module load IPython/8.5.0-GCCcore-11.3.0
module load jupyter-server/1.21.0-GCCcore-11.3.0
module load aiohttp/3.8.3-GCCcore-11.3.0
module load Optuna/3.1.0-foss-2022a
module load SciPy-bundle/2022.05-foss-2022a
module load scikit-learn/1.1.2-foss-2022a
module load Pillow/9.1.1-GCCcore-11.3.0
module load openpyxl/3.0.10-GCCcore-11.3.0

source $VSC_SCRATCH/distiboost/bin/activate

cd /scratch/brussel/105/vsc10528/LSSboost

python uci/UCI_gpboost_HP_single_run.py $SLURM_ARRAY_TASK_ID
