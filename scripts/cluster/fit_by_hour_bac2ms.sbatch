#!/bin/bash
# Kernel by time of day for BAC at the main specification of decision D2: E1m with h = delta = 2 ms,
# K = 3, started from the h = 2 ms panel fits (fits_e1m_h2_K3.csv). Replaces BAC's row of the
# by-hour table, which was fitted at 1 ms with K = 3. One task, 14 months x 3 windows.
#
#     sbatch scripts/cluster/fit_by_hour_bac2ms.sbatch
#
# Output: ~/hawkes-taq-data/by_hour_BAC_h2.csv (resumable).
#
#SBATCH --job-name=by_hour_bac
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/by_hour_bac_%A.o
#SBATCH --error=logs/by_hour_bac_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

DATA=~/hawkes-taq-data
python scripts/fit_by_hour.py \
    --nbbo $DATA/nbbo_2016Q4 $DATA/nbbo_2017Q1 $DATA/nbbo_2017Q2 $DATA/nbbo_2017Q3 $DATA/nbbo_2017Q4 \
    --fits $DATA/fits_e1m_h2_K3.csv --syms BAC --K 3 --delta 0.002 --min-state 0.002 \
    --out $DATA/by_hour_BAC_h2.csv
