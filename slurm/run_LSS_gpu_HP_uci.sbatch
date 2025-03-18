#!/bin/bash

#SBATCH --output=./report/%A_LSSboost_uci/slurm-%A_%a.out # STDOUT
#SBATCH --job-name=LSSboost_o
#SBATCH --time=1-12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=60G
#SBATCH --array=0-5
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
nvidia-smi >logs/openml/gpu_lgblss_info.txt
# start monitoring GPU usage:
nvidia-smi pmon -o DT -s um >logs/openml/gpu_lgblss_usage.txt &

# # Parameter combinations
modes=("exp")  # Add more modes if needed
natural_grad_values=("True" "False")
stabilizations=("MAD" "L2" "None")

# Calculate the index combinations
ARRAYID=$SLURM_ARRAY_TASK_ID
total_combinations=$((${#modes[@]} * ${#natural_grad_values[@]} * ${#stabilizations[@]}))

if [ $ARRAYID -ge $total_combinations ]; then
  echo "Error: ARRAYID exceeds number of parameter combinations."
  exit 1
fi

# Calculate the specific combination based on ARRAYID
mode_index=$((ARRAYID / (${#natural_grad_values[@]} * ${#stabilizations[@]})))
ng_index=$(( (ARRAYID / ${#stabilizations[@]}) % ${#natural_grad_values[@]} ))
stabilization_index=$((ARRAYID % ${#stabilizations[@]}))

mode=${modes[$mode_index]}
natural_grad=${natural_grad_values[$ng_index]}
stabilization=${stabilizations[$stabilization_index]}

# Define variables
# mode='exp'
# natural_grad='False'
# stabilization='L2'

# Echo the parameters for logging
echo "Running with mode=$mode, natural_grad=$natural_grad, stabilization=$stabilization"

python uci/UCI_lssboost_single_run_HP.py 8 $mode $natural_grad $stabilization

# stop monitoring GPU usage:
killall nvidia-smi
