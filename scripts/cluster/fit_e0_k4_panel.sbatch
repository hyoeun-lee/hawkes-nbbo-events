#!/bin/bash
# Raw events (E0) with K = 4 for the whole panel (8 stocks x 14 months), one array task per
# stock-month. Starts: the default six-point grid plus two points built from the saved K = 3
# raw fit of the same stock-month (results/fits_e0_K3_panel.csv), so the K = 4 fit can do no
# worse than the K = 3 one (revision decision D1: K = 4 for every series).
#
#     sbatch scripts/cluster/fit_e0_k4_panel.sbatch
#
# Output: ~/hawkes-taq-data/e0k4_panel/SYM_YYYYMM.csv, one file per task (resumable). The MU
# months of late 2017 have up to 300,000 raw events and take 10-30 hours, hence 48 hours.
# Afterwards: cat ~/hawkes-taq-data/e0k4_panel/*.csv (one header) -> fits_e0_k4_panel.csv.
#
#SBATCH --job-name=e0k4p
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-111%56
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/e0k4p_%a_%A.o
#SBATCH --error=logs/e0k4p_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
MONTHS=(201611 201612 201701 201702 201703 201704 201705 201706 201707 201708 201709 201710 201711 201712)
SYM=${SYMS[$((SLURM_ARRAY_TASK_ID / 14))]}
MONTH=${MONTHS[$((SLURM_ARRAY_TASK_ID % 14))]}
DATA=~/hawkes-taq-data
mkdir -p $DATA/e0k4_panel
python scripts/fit_panel.py \
    --nbbo $DATA/nbbo_2016Q4 $DATA/nbbo_2017Q1 $DATA/nbbo_2017Q2 $DATA/nbbo_2017Q3 $DATA/nbbo_2017Q4 \
    --months $MONTH --syms $SYM \
    --definitions E0 --K 4 \
    --extra-starts-from results/fits_e0_K3_panel.csv \
    --workers 1 \
    --out $DATA/e0k4_panel/${SYM}_${MONTH}.csv
