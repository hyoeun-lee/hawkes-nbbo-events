#!/bin/bash
# Simulation study, step 2: one replicate per array task (plan.csv from sim_truth.sbatch:
# 4 stocks x (50 base + 4 x 25 other) = 600 tasks). Submit after step 1 has finished:
#
#     sbatch --dependency=afterok:<sim_truth job id> scripts/cluster/sim_run.sbatch
#
# Each task is resumable and writes ~/hawkes-taq-data/sim/out/task_NNNNN.csv; combine and
# summarize with scripts/summarize_simulation.py. At most 64 tasks run at once (%64). One email
# when the whole array ends (not one per task).
#
#SBATCH --job-name=sim_run
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-599%64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sim/run_%a.o
#SBATCH --error=logs/sim/run_%a.e

set -euo pipefail
mkdir -p logs/sim
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

DATA=~/hawkes-taq-data
python scripts/simulate_study.py run --simdir $DATA/sim --plan $DATA/sim/plan.csv \
    --task $SLURM_ARRAY_TASK_ID --out-dir $DATA/sim/out
