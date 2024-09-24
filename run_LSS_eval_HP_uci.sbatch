#!/bin/bash

#SBATCH --output=./report/%A_LSSboost_uci/slurm-%A_%a.out # STDOUT

#SBATCH --job-name=LSSboost_uci

#SBATCH --time=1-12:00:00

#SBATCH --ntasks-per-node=1

#SBATCH --job-name=LSSboost_uci
#SBATCH --partition=skylake,skylake_mpi
#SBATCH --output=/scratch/brussel/105/vsc10528/LSSboost/experiments/slurm_output_%A_%a.out

#SBATCH --mem-per-cpu=60G

#####SBATCH --array=0-9
#SBATCH --array=0-9
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
module load SciPy-bundle/2022.05-foss-2022a
module load scikit-learn/1.1.2-foss-2022a
module load Pillow/9.1.1-GCCcore-11.3.0

source $VSC_SCRATCH/distiboost_311/bin/activate

cd /scratch/brussel/105/vsc10528/LSSboost

python uci/UCI_lssboost_single_run_HP.py $SLURM_ARRAY_TASK_ID
