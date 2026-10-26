#!/bin/bash
# Simulation, share0 experiment rerun with a fit of the unfiltered truth (revision item A13 b):
# the same 25 replicates per stock as the first batch (same seeds, so the same simulated months),
# refitted with the four core fits plus a K = 4 fit of the simulated clean process itself before
# the observation layer, ungated (definition TRUE). The TRUE fit's bias is the estimator's alone;
# the E1m fit's bias minus it is what the persistence filter and the gate add.
# 100 array tasks = 4 stocks x 25 replicates. Needs ~/hawkes-taq-data/sim/{truth,chains}_SYM.*
# from sim_truth.sbatch.
#
#     sbatch scripts/cluster/sim_share0_truth.sbatch
#
# Output: ~/hawkes-taq-data/sim/out3/share0_SYM_REP.csv (resumable).
#
#SBATCH --job-name=sim_share0
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-99%50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sim/share0_%a_%A.o
#SBATCH --error=logs/sim/share0_%a_%A.e

set -euo pipefail
mkdir -p logs/sim
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MDLZ BAC TGT)
SYM=${SYMS[$((SLURM_ARRAY_TASK_ID / 25))]}
REP=$((SLURM_ARRAY_TASK_ID % 25))
DATA=~/hawkes-taq-data
mkdir -p $DATA/sim/out3
python scripts/simulate_study.py run --simdir $DATA/sim --sym $SYM --experiment share0 \
    --reps $REP --fits truth --out $DATA/sim/out3/share0_${SYM}_${REP}.csv
