#!/bin/bash
#SBATCH --output=./report/%A_ExtraXGLSS_uci/slurm-%A_%a.out # STDOUT
#SBATCH --job-name=leftover_XGLSS_year_MSD
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition=skylake,skylake_mpi
#SBATCH --mem-per-cpu=60G
#SBATCH --array=3
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

source $VSC_SCRATCH/gpu_friendly/bin/activate

cd /scratch/brussel/105/vsc10528/LSSboost

# Parameter combinations
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

echo "Running with mode=$mode, natural_grad=$natural_grad, stabilization=$stabilization"

# Run the Python script with the selected parameters
python uci/UCI_xglssboost_HP_single_run.py 6 $mode $natural_grad $stabilization

