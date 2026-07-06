#!/bin/bash
#SBATCH --job-name=rvfl_bench
#SBATCH --array=1-42%4
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Adjust these lines for your university cluster.
# Examples:
# module load python/3.10
# module load cuda
# source ~/rvfl_env/bin/activate

cd "$SLURM_SUBMIT_DIR"

CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" benchmarking/configs.txt)

echo "============================================================"
echo "Running benchmark job"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "CONFIG=${CONFIG}"
echo "============================================================"

python benchmarking/benchmark_job.py \
    $CONFIG \
    --device cuda \
    --dtype float32 \
    --lr 3e-2 \
    --lamb 1e-3 \
    --rvfl-epochs 20 \
    --nn-epochs 200 \
    --ortho-method qr \
    --result-root benchmark_results