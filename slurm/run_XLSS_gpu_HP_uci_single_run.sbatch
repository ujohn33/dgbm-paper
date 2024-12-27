#!/bin/bash

#SBATCH --output=./report/%A_XLSSboost_openml/slurm-%A_%a.out # STDOUT
#SBATCH --job-name=LSSboost_openml
#SBATCH --time=1-12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=100G
#SBATCH --array=8
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

cd /scratch/brussel/105/vsc10528/LSSboost

# print GPU info:
nvidia-smi >logs/openml/gpu_lgblss_info.txt
# start monitoring GPU usage:
nvidia-smi pmon -o DT -s um >logs/openml/gpu_lgblss_usage.txt &

mode='exp'
natural_grad='False'
stabilization='None'

# Define variables
# mode='exp'
# natural_grad='False'
# stabilization='L2'

# Echo the parameters for logging
echo "Running with mode=$mode, natural_grad=$natural_grad, stabilization=$stabilization"

python uci/UCI_xglssboost_HP_single_run.py $SLURM_ARRAY_TASK_ID $mode $natural_grad $stabilization

# stop monitoring GPU usage:
killall nvidia-smi
