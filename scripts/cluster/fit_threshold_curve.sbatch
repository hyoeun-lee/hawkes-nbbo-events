#!/bin/bash
# Threshold sensitivity: merge every state shorter than h (and use a dead time of h), for
# h = 0.25, 0.5, 2 and 5 ms (1 ms is the main panel), all eight stocks, May and November 2017,
# K = 4, full start grid. One array task per threshold.
#
#     sbatch scripts/cluster/fit_threshold_curve.sbatch
#
# Output: ~/hawkes-taq-data/fits_threshold_<h>.csv
#
#SBATCH --job-name=threshold
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/threshold_%a_%A.o
#SBATCH --error=logs/threshold_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

H=(0.00025 0.0005 0.002 0.005)
h=${H[$SLURM_ARRAY_TASK_ID]}
DATA=~/hawkes-taq-data
python scripts/fit_panel.py \
    --nbbo $DATA/nbbo_2017Q2 $DATA/nbbo_2017Q4 \
    --months 201705 201711 \
    --definitions E1m --K 4 --starts full --min-state $h --delta $h \
    --workers $SLURM_CPUS_PER_TASK \
    --out $DATA/fits_threshold_$h.csv
