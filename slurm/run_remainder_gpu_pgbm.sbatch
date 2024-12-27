#!/bin/bash

#SBATCH --output=./report/%A_pgbm_rest_openml/slurm-%A_%a.out # STDOUT
#SBATCH --job-name=pgbm_rest_openml
#SBATCH --time=2-12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=60G
#SBATCH --partition=pascal_gpu,ampere_gpu
#SBATCH --array=10
#SBATCH --mail-type=ALL


# File to run on HPC
# First activate your python environment (you might have to create your own) with:
# virtualenv --system-site-packages $VSC_DATA/scenarioenv
# And then intall the requirements with
# pip install -r requirements.txt

module load Python/3.11.3-GCCcore-12.3.0
module load IPython/8.14.0-GCCcore-12.3.0
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load jupyter-server/2.14.0-GCCcore-12.3.0
module load aiohttp/3.8.5-GCCcore-12.3.0
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0
module load openpyxl/3.1.2-GCCcore-12.3.0

source $VSC_SCRATCH/gpu_friendly/bin/activate

# print GPU info:
nvidia-smi >logs/gpu_info.txt
# start monitoring GPU usage:
nvidia-smi pmon -o DT -s um >logs/gpu_usage.txt &

cd /scratch/brussel/105/vsc10528/LSSboost

python openml/openml_pgbm_HP_single_run.py $SLURM_ARRAY_TASK_ID

# stop monitoring GPU usage:
killall nvidia-smi
