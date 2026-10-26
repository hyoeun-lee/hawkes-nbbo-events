#!/bin/bash
# Baseline resolution for all eight stocks (revision item A16): the main specification refitted
# with baseline bins of 5, 10, 60 and 110 minutes and one level per day (330 minutes), May and
# November 2017. E1m, K = 4, h = delta = 1 ms, full start grid; BAC: K = 3, h = delta = 2 ms.
# The 30-minute fits are the main panel (fits_e1m_delta_K4.csv / fits_e1m_h2_K3.csv).
# 80 array tasks = 8 stocks x 2 months x 5 resolutions.
#
#     sbatch scripts/cluster/fit_baseline_bins.sbatch
#
# Output: ~/hawkes-taq-data/baseline_bins/SYM_YYYYMM_B<minutes>.csv, one file per task (resumable).
#
#SBATCH --job-name=base_bins
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-79%40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bbins_%a_%A.o
#SBATCH --error=logs/bbins_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
MONTHS=(201705 201711)
BINS=(5 10 60 110 330)
SYM=${SYMS[$((SLURM_ARRAY_TASK_ID / 10))]}
MONTH=${MONTHS[$(((SLURM_ARRAY_TASK_ID / 5) % 2))]}
B=${BINS[$((SLURM_ARRAY_TASK_ID % 5))]}
K=4; H=0.001
if [ "$SYM" = BAC ]; then K=3; H=0.002; fi
DATA=~/hawkes-taq-data
mkdir -p $DATA/baseline_bins
python scripts/fit_panel.py \
    --nbbo $DATA/nbbo_2017Q2 $DATA/nbbo_2017Q4 \
    --months $MONTH --syms $SYM \
    --definitions E1m --K $K --min-state $H --delta $H --starts full --bin-minutes $B \
    --workers 1 \
    --out $DATA/baseline_bins/${SYM}_${MONTH}_B${B}.csv
