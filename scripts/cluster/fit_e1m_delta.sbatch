#!/bin/bash
# Main E1m panel with the 1 ms dead time, K = 3 and K = 4, full start grid, half days excluded.
# Becomes the main results table (K = 4; K = 3 for BAC).
#
#     sbatch scripts/cluster/fit_e1m_delta.sbatch
#
# Output: ~/hawkes-taq-data/fits_e1m_delta_K3.csv and fits_e1m_delta_K4.csv (resumable).
#
#SBATCH --job-name=e1m_delta
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=3-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/e1m_delta_K%a_%A.o
#SBATCH --error=logs/e1m_delta_K%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

K=$SLURM_ARRAY_TASK_ID
DATA=~/hawkes-taq-data
python scripts/fit_panel.py \
    --nbbo $DATA/nbbo_2016Q4 $DATA/nbbo_2017Q1 $DATA/nbbo_2017Q2 $DATA/nbbo_2017Q3 $DATA/nbbo_2017Q4 \
    --months 201611 201612 201701 201702 201703 201704 201705 201706 201707 201708 201709 201710 201711 201712 \
    --definitions E1m --K $K --starts full --delta 0.001 \
    --workers $SLURM_CPUS_PER_TASK \
    --out $DATA/fits_e1m_delta_K$K.csv
