#!/bin/bash
# Day-block bootstrap of the clean fits for the whole panel (8 stocks x 14 months): blocks of
# 5 trading days, 100 replicates per stock-month, E1m only, started from the main fits. Gives
# the model-free standard errors of n(1 s) and n(10 s) for every stock-month (revision item A7,
# for the rebuilt main table). Main specification per stock: K = 4 at h = 1 ms, except BAC at
# h = 2 ms with K = 3 (revision decision D2).
#
#     sbatch scripts/cluster/block_bootstrap_panel.sbatch
#
# 224 array tasks = 112 stock-months x 2 chunks of 50 replicates. Output:
# ~/hawkes-taq-data/blockboot_panel/SYM_YYYYMM.csv (resumable; the two chunks append to the
# same file, keyed by replicate). A single-start E1m fit takes 1-20 minutes, hence 24 hours.
#
#SBATCH --job-name=bbpanel
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-223%60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bbpanel_%a_%A.o
#SBATCH --error=logs/bbpanel_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
MONTHS=(201611 201612 201701 201702 201703 201704 201705 201706 201707 201708 201709 201710 201711 201712)
SM=$((SLURM_ARRAY_TASK_ID / 2))
CHUNK=$((SLURM_ARRAY_TASK_ID % 2))
SYM=${SYMS[$((SM / 14))]}
MONTH=${MONTHS[$((SM % 14))]}
DATA=~/hawkes-taq-data
if [ "$SYM" = "BAC" ]; then
    SPEC="--h 0.002 --K-clean 3 --fits-clean $DATA/fits_e1m_h2_K3.csv"
else
    SPEC="--h 0.001 --K-clean 4 --fits-clean $DATA/fits_e1m_delta_K4.csv"
fi
python scripts/block_bootstrap.py \
    --nbbo $DATA/nbbo_2016Q4 $DATA/nbbo_2017Q1 $DATA/nbbo_2017Q2 $DATA/nbbo_2017Q3 $DATA/nbbo_2017Q4 \
    --sym $SYM --month $MONTH --definitions clean $SPEC --block 5 \
    --rep-start $((CHUNK * 50)) --rep-count 50 \
    --out $DATA/blockboot_panel/${SYM}_${MONTH}.csv
