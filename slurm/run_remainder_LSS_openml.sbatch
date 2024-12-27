#!/bin/bash

#SBATCH --output=./report/%A_ExtraRAMLSS_openml/slurm-%A_%a.out # STDOUT

#SBATCH --job-name=leftover_LSS_openml_nyc

#SBATCH --time=1-12:00:00

#SBATCH --ntasks-per-node=1

#SBATCH --partition=skylake,skylake_mpi

#SBATCH --mem-per-cpu=120G

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
module load Optuna/3.5.0-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load Pillow/10.0.0-GCCcore-12.3.0
module load openpyxl/3.1.2-GCCcore-12.3.0
#module load LightGBM/4.5.0-foss-2023a-CUDA-12.1.1

source $VSC_SCRATCH/distiboost_311/bin/activate

cd /scratch/brussel/105/vsc10528/LSSboost

# Define variables
mode='exp'
natural_grad='False'
stabilization='None'

# Echo the parameters for logging
echo "Running with mode=$mode, natural_grad=$natural_grad, stabilization=$stabilization"

# Run the Python script with the selected parameters
python openml/openml_lssboost_HP_single_run.py 10 $mode $natural_grad $stabilization

