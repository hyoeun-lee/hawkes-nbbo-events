#!/bin/bash
# Simulation study, second batch (subms_clean, powerlaw, actscale; hawkes_taq/simstudy.py):
# 4 stocks x (25 subms_clean + 25 powerlaw + 50 actscale) = 400 tasks, one replicate each.
# Uses the truths and brief-state calibration from the first batch (~/hawkes-taq-data/sim),
# so run it after sim_truth.sbatch. Writes its own plan and outputs (plan2.csv, out2/):
#
#     sbatch scripts/cluster/sim_run2.sbatch
#
# Each task is resumable and writes ~/hawkes-taq-data/sim/out2/task_NNNNN.csv; combine and
# summarize with scripts/summarize_simulation.py. At most 64 tasks at once (%64). One email
# when the whole array ends.
#
#SBATCH --job-name=sim_run2
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-399%64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sim2/run_%a.o
#SBATCH --error=logs/sim2/run_%a.e

set -euo pipefail
mkdir -p logs/sim2
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

DATA=~/hawkes-taq-data
# every task writes the same plan; the plan is deterministic, so this is safe
if [ ! -f $DATA/sim/plan2.csv ]; then
    python scripts/simulate_study.py plan --batch 2 --out $DATA/sim/plan2.csv.$SLURM_ARRAY_TASK_ID
    mv -f $DATA/sim/plan2.csv.$SLURM_ARRAY_TASK_ID $DATA/sim/plan2.csv
fi
python scripts/simulate_study.py run --simdir $DATA/sim --plan $DATA/sim/plan2.csv \
    --task $SLURM_ARRAY_TASK_ID --out-dir $DATA/sim/out2
