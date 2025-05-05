#!/bin/bash

#SBATCH --output=./report/%A_LGBM_M5/slurm-%A_%a.out # STDOUT
#SBATCH --job-name=LGBM_M5
#SBATCH --time=1-12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition=skylake,skylake_mpi
#SBATCH --mem-per-cpu=25G
#SBATCH --array=1-3
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

# Determine which level to train based on SLURM_ARRAY_TASK_ID
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    LEVEL=13
    LEVEL_NAME="HOBBIES"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    LEVEL=14
    LEVEL_NAME="HOUSEHOLD"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    LEVEL=15
    LEVEL_NAME="FOODS"
else
    echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Training models for level $LEVEL ($LEVEL_NAME) with SUPER_SPEED enabled"
echo "============================================================"
echo "Training level $LEVEL ($LEVEL_NAME)"
echo "============================================================"
python m5/run_m5_forecaster.py --level $LEVEL --super-speed

echo "Training for level $LEVEL ($LEVEL_NAME) complete!"
