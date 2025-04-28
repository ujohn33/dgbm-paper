#!/bin/bash

#SBATCH --output=./report/%A_LSSboost_LGBM_M5/slurm-%A_%a.out # STDOUT
#SBATCH --job-name=LSSboost_LGBM_M5
#SBATCH --time=1-12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=LSSboost_LGBM_M5
#SBATCH --partition=skylake,skylake_mpi
#SBATCH --mem-per-cpu=25G
#SBATCH --array=1-100
#SBATCH --mail-type=ALL


# File to run on HPC
# First activate your python environment (you might have to create your own) with:
# virtualenv --system-site-packages $VSC_DATA/scenarioenv
# And then intall the requirements with
# pip install -r requirements.txt

module load Python/3.11.3-GCCcore-12.3.0
module load IPython/8.14.0-GCCcore-12.3.0
module load LightGBM/4.5.0-foss-2023a
module load jupyter-server/2.14.0-GCCcore-12.3.0
module load aiohttp/3.8.5-GCCcore-12.3.0
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0
module load openpyxl/3.1.2-GCCcore-12.3.0

source $VSC_SCRATCH/distiboost_311/bin/activate

cd /scratch/brussel/105/vsc10528/LSSboost

# Fixed values
dataset=$SLURM_ARRAY_TASK_ID
mode="softplus" # ("softplus" "relu" "exp")
natural_grad="False" #("True" "False")
stabilization="None"
clip_value="None"
standardize="False"
dist='NegativeBinomial' #("NegativeBinomial" "Poisson" "Gaussian")

# Echo the parameters and dataset for logging
echo "Running with dataset=$dataset, mode=$mode, stabilization=$stabilization"

python m5/m5_comparison_lssboost_vs_lightgbm.py $dataset $mode $stabilization $dist
