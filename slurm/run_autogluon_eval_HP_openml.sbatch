#!/bin/bash

#SBATCH --output=./report/%A_autogluon_openml/slurm-%A_%a.out # STDOUT

#SBATCH --job-name=autogluon_openml

#SBATCH --time=1-12:00:00

#SBATCH --ntasks-per-node=1

#SBATCH --partition=skylake,skylake_mpi

#SBATCH --mem-per-cpu=60G

#####SBATCH --array=0-18
#SBATCH --array=0-18
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
module load SciPy-bundle/2023.07-gfbf-2023a
module load scikit-learn/1.4.2-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0

source $VSC_DATA/auto/bin/activate

cd /scratch/brussel/105/vsc10528/LSSboost

python openml/openml_autogluon_HP_single_run.py $SLURM_ARRAY_TASK_ID
