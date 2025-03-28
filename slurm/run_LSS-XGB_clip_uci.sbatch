#!/bin/bash

#SBATCH --output=./report/%A_LSS-XGB_UCI/slurm-%A_%a.out # STDOUT
#SBATCH --job-name=LSS-XGB_UCI
#SBATCH --time=1-12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=60G
#SBATCH --array=0-19
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

# print GPU info:
nvidia-smi >logs/uci/gpu_lgblss_info.txt
# start monitoring GPU usage:
nvidia-smi pmon -o DT -s um >logs/uci/gpu_lgblss_usage.txt &

#datasets=(10 11 12 13 14 15 16 17)
datasets=(0 1 2 3 4 5 6 7 8 9)
natural_grad_values=("True" "False")

num_datasets=${#datasets[@]}
num_param_combinations=${#natural_grad_values[@]}

# Total combinations: datasets × natural_grad_values
dataset_index=$((SLURM_ARRAY_TASK_ID / num_param_combinations))
ng_index=$((SLURM_ARRAY_TASK_ID % num_param_combinations))

dataset=${datasets[$dataset_index]}
natural_grad=${natural_grad_values[$ng_index]}

# Fixed values
mode="exp"
stabilization="None"
clip_value="None"
standardize="True"

# Echo the parameters and dataset for logging
echo "Running with dataset=$dataset, mode=$mode, natural_grad=$natural_grad, stabilization=$stabilization, clipping=$clip_value and standardize=$standardize"

python uci/UCI_xglssboost_HP_single_run.py $dataset $mode $natural_grad $stabilization $clip_value $standardize

# stop monitoring GPU usage:
killall nvidia-smi
