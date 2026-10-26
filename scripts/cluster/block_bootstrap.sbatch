#!/bin/bash
# Paired day-block bootstrap (blocks of 5 trading days, same days for E0 and E1m), May 2017,
# eight stocks, 100 replicates each: 80 array tasks = 8 stocks x 10 chunks of 10 replicates.
# E1m K = 4 with the 1 ms dead time and E0 K = 4, started from the main fits. Review point E.
#
#     sbatch scripts/cluster/block_bootstrap.sbatch
#
# Output: ~/hawkes-taq-data/blockboot/SYM_201705.csv (one row per replicate; resumable: a
# resubmitted task skips replicates already in the file). The ten chunks of a stock append to
# the same file; rows are keyed by replicate number, so concurrent appends are harmless.
# TGT's E0 fits take about 30 min each with a single start, hence 12 hours per chunk.
#
#SBATCH --job-name=blockboot
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-79%40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/blockboot_%a_%A.o
#SBATCH --error=logs/blockboot_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
SYM=${SYMS[$((SLURM_ARRAY_TASK_ID / 10))]}
CHUNK=$((SLURM_ARRAY_TASK_ID % 10))
DATA=~/hawkes-taq-data
python scripts/block_bootstrap.py \
    --nbbo $DATA/nbbo_2017Q2 --sym $SYM --month 201705 \
    --fits-clean $DATA/fits_e1m_delta_K4.csv --fits-raw $DATA/fits_e0_k4_$SYM.csv \
    --rep-start $((CHUNK * 10)) --rep-count 10 \
    --out $DATA/blockboot/${SYM}_201705.csv
