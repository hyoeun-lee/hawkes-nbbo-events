#!/bin/bash
# Kernel by time of day for the whole panel: E1m, K = 4, 1 ms dead time, three windows per day
# (10:00-11:00, 11:00-14:30, 14:30-15:30), 14 months, one array task per stock, started from the
# main fits (fits_e1m_delta_K4.csv). Results §23.B extended to the panel.
#
#     sbatch scripts/cluster/fit_by_hour.sbatch
#
# Output: ~/hawkes-taq-data/by_hour_SYM.csv (resumable).
#
#SBATCH --job-name=by_hour
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/by_hour_%a_%A.o
#SBATCH --error=logs/by_hour_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
SYM=${SYMS[$SLURM_ARRAY_TASK_ID]}
DATA=~/hawkes-taq-data
python scripts/fit_by_hour.py \
    --nbbo $DATA/nbbo_2016Q4 $DATA/nbbo_2017Q1 $DATA/nbbo_2017Q2 $DATA/nbbo_2017Q3 $DATA/nbbo_2017Q4 \
    --fits $DATA/fits_e1m_delta_K4.csv --syms $SYM \
    --out $DATA/by_hour_$SYM.csv
