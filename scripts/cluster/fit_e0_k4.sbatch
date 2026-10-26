#!/bin/bash
# Raw events (E0) with K = 4 on real data, May and November 2017, one array task per stock
# (two fits each). Fair comparison of slow timescales with E1m (K = 4) and a direct check of
# immigrant invariance (baseline events per day under E0 and E1m). Half days excluded.
#
#     sbatch scripts/cluster/fit_e0_k4.sbatch
#
# Output: ~/hawkes-taq-data/fits_e0_k4_SYM.csv, one file per stock (resumable).
# E0 fits are slow in busy months (MU, November 2017, has about 300,000 events), hence 48 hours.
#
#SBATCH --job-name=e0_k4
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/e0_k4_%a_%A.o
#SBATCH --error=logs/e0_k4_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
SYM=${SYMS[$SLURM_ARRAY_TASK_ID]}
DATA=~/hawkes-taq-data
python scripts/fit_panel.py \
    --nbbo $DATA/nbbo_2017Q2 $DATA/nbbo_2017Q4 \
    --months 201705 201711 --syms $SYM \
    --definitions E0 --K 4 \
    --workers $SLURM_CPUS_PER_TASK \
    --out $DATA/fits_e0_k4_$SYM.csv
