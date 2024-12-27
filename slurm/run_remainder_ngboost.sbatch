#!/bin/bash

#SBATCH --output=./report/%A_ngboost_openml/slurm-%A_%a.out # STDOUT

#SBATCH --partition=skylake,skylake_mpi

#SBATCH --mem-per-cpu=60G

#SBATCH --job-name=ngboost_openml

#SBATCH --time=2-00:00:00

#SBATCH --ntasks-per-node=1

#SBATCH --array=10
#SBATCH --mail-type=ALL


# File to run on HPC
# First activate your python environment (you might have to create your own) with:
# virtualenv --system-site-packages $VSC_DATA/scenarioenv
# And then intall the requirements with
# pip install -r requirements.txt

module load Python/3.11.3-GCCcore-12.3.0
module load IPython/8.14.0-GCCcore-12.3.0
module load jupyter-server/2.14.0-GCCcore-12.3.0
module load aiohttp/3.8.5-GCCcore-12.3.0
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0

source $VSC_SCRATCH/distiboost_311/bin/activate

cd /scratch/brussel/105/vsc10528/LSSboost

python openml/openml_ngboost_HP_single_run.py $SLURM_ARRAY_TASK_ID
