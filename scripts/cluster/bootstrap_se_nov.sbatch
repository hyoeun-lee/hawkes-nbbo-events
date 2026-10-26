#!/bin/bash
# Calibrated KS tests and bootstrap standard errors for November 2017 (revision item A14), the
# busiest month: E1m, K = 4 and h = delta = 1 ms (BAC: K = 3, h = delta = 2 ms), full start grid,
# 99 replicates, eight stocks (one array task each). Same as bootstrap_se.sbatch for May.
#
#     sbatch scripts/cluster/bootstrap_se_nov.sbatch
#
# Output: ~/hawkes-taq-data/boot/SYM_201711/{fit,replicates,summary}.csv (resumable).
#
#SBATCH --job-name=boot_nov
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --output=logs/boot_nov_%a_%A.o
#SBATCH --error=logs/boot_nov_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
SYM=${SYMS[$SLURM_ARRAY_TASK_ID]}
K=4; H=0.001
if [ "$SYM" = BAC ]; then K=3; H=0.002; fi
DATA=~/hawkes-taq-data
python scripts/bootstrap_ks.py --nbbo $DATA/nbbo_2017Q4 --sym $SYM --month 201711 \
    --definition E1m --K $K --min-state $H --delta $H --starts full --n-boot 99 \
    --workers $SLURM_CPUS_PER_TASK --out-dir $DATA/boot/${SYM}_201711
