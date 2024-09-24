#!/bin/bash

#SBATCH --output=./report/%A/slurm-%A_%a.out # STDOUT

#SBATCH --job-name=LSSboost

#SBATCH --time=1-00:00:00

#SBATCH --ntasks-per-node=1

#SBATCH --partition=skylake,skylake_mpi

#SBATCH --mem-per-cpu=20G

#####SBATCH --array=0-0
#SBATCH --array=0-0
#SBATCH --mail-type=ALL


# File to run on HPC
# First activate your python environment (you might have to create your own) with:
# virtualenv --system-site-packages $VSC_DATA/scenarioenv
# And then intall the requirements with
# pip install -r requirements.txt

module load Python/3.10.4-GCCcore-11.3.0
module load JupyterHub/3.1.1-GCCcore-11.3.0
# module load IPython/8.5.0-GCCcore-11.3.0
# module load jupyter-server/1.21.0-GCCcore-11.3.0
module load SciPy-bundle/2022.05-foss-2022a
module load Pillow/9.1.1-GCCcore-11.3.0

source $VSC_DATA/distiboost/bin/activate

cd /data/brussel/105/vsc10528/LSSboost

python lssboost_HP.py 
