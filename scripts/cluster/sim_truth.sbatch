#!/bin/bash
# Simulation study, step 1: truth models and brief-state calibration for the four simulation
# stocks (May 2017, E1m, K = 4, 1 ms dead time), then the task plan for step 2.
#
#     sbatch scripts/cluster/sim_truth.sbatch
#
# Output: ~/hawkes-taq-data/sim/truth_SYM.npz, chains_SYM.csv.gz, plan.csv
#
#SBATCH --job-name=sim_truth
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sim_truth_%a_%A.o
#SBATCH --error=logs/sim_truth_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MDLZ BAC TGT)
SYM=${SYMS[$SLURM_ARRAY_TASK_ID]}
DATA=~/hawkes-taq-data
python scripts/simulate_study.py truth --nbbo $DATA/nbbo_2017Q2 --month 201705 \
    --syms $SYM --simdir $DATA/sim
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
    python scripts/simulate_study.py plan --reps-base 50 --reps-other 25 --out $DATA/sim/plan.csv
fi
