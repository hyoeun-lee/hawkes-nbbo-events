#!/bin/bash
# Bootstrap standard errors and calibrated KS tests for the main specification, May 2017:
# E1m, 1 ms dead time, K = 4 (K = 3 for BAC), full start grid, 99 replicates, eight stocks
# (one array task each). Every replicate's parameters are saved, so standard errors, percentile
# intervals and the bootstrap estimate of bias can be computed for every parameter.
#
#     sbatch scripts/cluster/bootstrap_se.sbatch
#
# Output: ~/hawkes-taq-data/boot/SYM_201705/{fit,replicates,summary}.csv (resumable: resubmit
# the same task to continue).
#
#SBATCH --job-name=boot_se
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/boot_se_%a_%A.o
#SBATCH --error=logs/boot_se_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
SYM=${SYMS[$SLURM_ARRAY_TASK_ID]}
K=4
if [ "$SYM" = BAC ]; then K=3; fi
DATA=~/hawkes-taq-data
python scripts/bootstrap_ks.py --nbbo $DATA/nbbo_2017Q2 --sym $SYM --month 201705 \
    --definition E1m --K $K --delta 0.001 --starts full --n-boot 99 \
    --workers $SLURM_CPUS_PER_TASK --out-dir $DATA/boot/${SYM}_201705
