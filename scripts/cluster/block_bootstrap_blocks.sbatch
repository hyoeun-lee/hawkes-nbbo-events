#!/bin/bash
# Paired day-block bootstrap, May 2017, eight stocks, with block lengths of 1, 5 and 10 trading
# days (revision item A7: sensitivity to the block length). Same days for E0 and E1m; 100
# replicates per stock and block length. Clean: K = 4 at h = 1 ms, BAC K = 3 at h = 2 ms
# (decision D2); raw: E0 with K = 4 started from the refitted raw panel (fit_e0_k4_panel.sbatch),
# whose ladder starts reach better optima than the earlier May fits (INTC May: +120 log-likelihood
# units). Run after that job: collect its output first,
#
#     python -c "import glob,pandas as pd; pd.concat([pd.read_csv(f,dtype={'month':str}) for f in sorted(glob.glob('$HOME/hawkes-taq-data/e0k4_panel/*.csv'))]).to_csv('$HOME/hawkes-taq-data/fits_e0_k4_panel.csv',index=False)"
#     sbatch scripts/cluster/block_bootstrap_blocks.sbatch
#
# 240 array tasks = 8 stocks x 3 block lengths x 10 chunks of 10 replicates. Output:
# ~/hawkes-taq-data/blockboot_b{1,5,10}/SYM_201705.csv (resumable). TGT's raw fits take about
# 30 minutes each with a single start, hence 12 hours per chunk.
#
#SBATCH --job-name=bbblocks
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-239%60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bbblocks_%a_%A.o
#SBATCH --error=logs/bbblocks_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
BLOCKS=(1 5 10)
SYM=${SYMS[$((SLURM_ARRAY_TASK_ID / 30))]}
BLOCK=${BLOCKS[$(((SLURM_ARRAY_TASK_ID % 30) / 10))]}
CHUNK=$((SLURM_ARRAY_TASK_ID % 10))
DATA=~/hawkes-taq-data
if [ "$SYM" = "BAC" ]; then
    SPEC="--h 0.002 --K-clean 3 --fits-clean $DATA/fits_e1m_h2_K3.csv"
else
    SPEC="--h 0.001 --K-clean 4 --fits-clean $DATA/fits_e1m_delta_K4.csv"
fi
python scripts/block_bootstrap.py \
    --nbbo $DATA/nbbo_2017Q2 --sym $SYM --month 201705 \
    --definitions both $SPEC --K-raw 4 --fits-raw $DATA/fits_e0_k4_panel.csv \
    --block $BLOCK --rep-start $((CHUNK * 10)) --rep-count 10 \
    --out $DATA/blockboot_b${BLOCK}/${SYM}_201705.csv
