#!/bin/bash
#SBATCH --output=./report/%A_rerun_gaps_lsf/slurm-%A_%a.out
#SBATCH --job-name=rr_gaps_lsf
#SBATCH --time=3-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G

module load Python/3.11.3-GCCcore-12.3.0
source /scratch/brussel/105/vsc10528/venvs/lsf_openml_311/bin/activate
cd /scratch/brussel/105/vsc10528/LSSboost
mkdir -p logs/openml/predictions logs/openml/fold_metrics results/openml
python -u openml/openml_lsf_HP_single_run.py 13 123
