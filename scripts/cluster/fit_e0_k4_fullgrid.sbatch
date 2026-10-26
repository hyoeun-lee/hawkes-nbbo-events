#!/bin/bash
# Raw events with the full start grid (revision item A13 a, data side): E0, K = 4, May 2017,
# eight stocks, every grid combination (15) plus the three ladders plus the two points built from
# the K = 3 fit, against the panel fits of fit_e0_k4_panel.sbatch (default grid plus the two
# K = 3 points). If the log-likelihood does not improve, the panel's raw optima stand.
# One array task per stock.
#
#     sbatch scripts/cluster/fit_e0_k4_fullgrid.sbatch
#
# Output: ~/hawkes-taq-data/e0k4_fullgrid/SYM_201705.csv (resumable).
#
#SBATCH --job-name=e0k4_full
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/e0k4f_%a_%A.o
#SBATCH --error=logs/e0k4f_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
SYM=${SYMS[$SLURM_ARRAY_TASK_ID]}
DATA=~/hawkes-taq-data
mkdir -p $DATA/e0k4_fullgrid
python scripts/fit_panel.py \
    --nbbo $DATA/nbbo_2017Q2 \
    --months 201705 --syms $SYM \
    --definitions E0 --K 4 --starts full \
    --extra-starts-from results/fits_e0_K3_panel.csv \
    --workers 1 \
    --out $DATA/e0k4_fullgrid/${SYM}_201705.csv
