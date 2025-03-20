#!/bin/bash

#SBATCH --output=./report/%A_LSSboost_uci/slurm-%A_%a.out # STDOUT
#SBATCH --job-name=LSSboost_o
#SBATCH --time=1-12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=80G
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
module load LightGBM/4.5.0-foss-2023a-CUDA-12.1.1 
module load jupyter-server/2.14.0-GCCcore-12.3.0
module load aiohttp/3.8.5-GCCcore-12.3.0
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0
module load openpyxl/3.1.2-GCCcore-12.3.0

source $VSC_SCRATCH/gpu_friendly/bin/activate

cd /scratch/brussel/105/vsc10528/LSSboost

# Define variables
mode='exp'
natural_grad='True'
stabilization='MAD'
clipping='False'

echo "Running with mode=$mode, natural_grad=$natural_grad, stabilization=$stabilization"

# Run the Python script with the selected parameters
python uci/UCI_lssboost_single_run_HP.py $SLURM_ARRAY_TASK_ID $mode $natural_grad $stabilization $clipping 

