#!/bin/bash
# Load modules
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

cd /scratch/brussel/105/vsc10528/LSSboost

# Activate virtual environment
source $VSC_SCRATCH/gpu_friendly/bin/activate

# Run the debugger
python -Xfrozen_modules=off -m debugpy --listen 5267 --wait-for-client openml/openml_lssboost_HP_single_run.py "$@"
