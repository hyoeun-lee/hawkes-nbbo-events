#!/bin/bash
# Outlier days: flag days with more than 3x the stock-month's median number of NBBO changes and
# refit those stock-months (E1m, K = 4, 1 ms dead time) without them. One array task per stock,
# all 14 months. Results §24.
#
#     sbatch scripts/cluster/outliers.sbatch
#
# Output: ~/hawkes-taq-data/outliers_SYM.csv (one row per stock-month; resumable).
#
#SBATCH --job-name=outliers
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/outliers_%a_%A.o
#SBATCH --error=logs/outliers_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
SYM=${SYMS[$SLURM_ARRAY_TASK_ID]}
DATA=~/hawkes-taq-data
python scripts/fit_without_outliers.py \
    --nbbo $DATA/nbbo_2016Q4 $DATA/nbbo_2017Q1 $DATA/nbbo_2017Q2 $DATA/nbbo_2017Q3 $DATA/nbbo_2017Q4 \
    --fits $DATA/fits_e1m_delta_K4.csv --syms $SYM \
    --out $DATA/outliers_$SYM.csv
