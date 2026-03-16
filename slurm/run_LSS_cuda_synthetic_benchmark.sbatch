#!/bin/bash

#SBATCH --output=./report/%A_LSSboost_synth_cuda/slurm-%A_%a.out
#SBATCH --job-name=LSSboost_synth_cuda
#SBATCH --time=0-08:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-17
#SBATCH --mail-type=ALL


# First activate your python environment. It needs LightGBMLSS dependencies,
# including pyro-ppl, plus a CUDA-enabled LightGBM build.

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

export PYTHONPATH="$PWD/LightGBMLSS:$PYTHONPATH"

mkdir -p logs/synthetic
mkdir -p results/synthetic/lightgbmlss_cuda_benchmark
mkdir -p report/${SLURM_JOB_ID}_LSSboost_synth_cuda

# print GPU info:
nvidia-smi > logs/synthetic/gpu_lgblss_synth_info_${SLURM_JOB_ID}.txt
# start monitoring GPU usage:
nvidia-smi pmon -o DT -s um > logs/synthetic/gpu_lgblss_synth_usage_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt &

data_sizes=(10000 25000 50000 100000 250000 500000)
repeat_ids=(0 1 2)

num_sizes=${#data_sizes[@]}
num_repeats=${#repeat_ids[@]}

size_index=$((SLURM_ARRAY_TASK_ID / num_repeats))
repeat_index=$((SLURM_ARRAY_TASK_ID % num_repeats))

n_samples=${data_sizes[$size_index]}
repeat_id=${repeat_ids[$repeat_index]}
run_seed=$((123 + repeat_id))

output_path="results/synthetic/lightgbmlss_cuda_benchmark/benchmark_n${n_samples}_repeat${repeat_id}.csv"

echo "Running synthetic LightGBMLSS CUDA benchmark with n_samples=${n_samples}, repeat_id=${repeat_id}, seed=${run_seed}"

python synthetic/lightgbmlss_cuda_benchmark.py \
  --data-sizes ${n_samples} \
  --n-features 50 \
  --n-informative 20 \
  --train-fraction 0.8 \
  --num-boost-round 200 \
  --early-stopping-rounds 30 \
  --num-leaves 63 \
  --min-data-in-leaf 50 \
  --feature-fraction 0.9 \
  --bagging-fraction 0.9 \
  --bagging-freq 1 \
  --eta 0.05 \
  --num-threads ${SLURM_CPUS_PER_TASK} \
  --devices cpu cuda \
  --response-fn exp \
  --stabilization None \
  --loss-fn nll \
  --base-seed ${run_seed} \
  --repeats 1 \
  --precompute-start-values \
  --time-prediction \
  --warmup \
  --output-path ${output_path}

# stop monitoring GPU usage:
killall nvidia-smi
