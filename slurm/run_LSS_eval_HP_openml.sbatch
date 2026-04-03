#!/bin/bash

#SBATCH --output=./report/%A_LSSboost_openml/slurm-%A_%a.out # STDOUT
#SBATCH --job-name=LSSboost_openml
#SBATCH --time=1-12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition=zen5_mpi
#SBATCH --mem-per-cpu=60G
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
module load LightGBM/4.5.0-foss-2023a
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0

source $VSC_SCRATCH/distiboost_311/bin/activate

cd /scratch/brussel/105/vsc10528/LSSboost

# Fixed values
dataset=$SLURM_ARRAY_TASK_ID
mode="exp"
natural_grad="False" #("True" "False")
stabilization="None"
clip_value="None"
standardize="False"
run_seed="123"
safety="False"

# Echo the parameters and dataset for logging
echo "Running with dataset=$dataset, mode=$mode, natural_grad=$natural_grad, stabilization=$stabilization, clipping=$clip_value, standardize=$standardize, run_seed=$run_seed, safety=$safety"

python openml/openml_lssboost_HP_single_run.py $dataset $mode $natural_grad $stabilization $clip_value $standardize $run_seed $safety
