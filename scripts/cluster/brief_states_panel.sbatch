#!/bin/bash
# Brief-state table for the whole panel (revision item A20): for every stock-month, the shares of
# clean moves that pass directly, through a lock or through a widening, the flicker rate, the
# chain durations, the time at one tick and the Dayri-Rosenbaum eta, from
# scripts/calibrate_brief_states.py (the simulation's calibration step, here for all 112
# stock-months). One array task per stock, 14 months each.
#
#     sbatch scripts/cluster/brief_states_panel.sbatch
#
# Output: ~/hawkes-taq-data/briefstates/SYM/calibration_summary.csv and chains_SYM_YYYYMM.csv.gz.
#
#SBATCH --job-name=brief_panel
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/brief_%a_%A.o
#SBATCH --error=logs/brief_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

SYMS=(INTC MU MDLZ BAC WFC T TGT ATVI)
SYM=${SYMS[$SLURM_ARRAY_TASK_ID]}
DATA=~/hawkes-taq-data
python scripts/calibrate_brief_states.py \
    --nbbo $DATA/nbbo_2016Q4 $DATA/nbbo_2017Q1 $DATA/nbbo_2017Q2 $DATA/nbbo_2017Q3 $DATA/nbbo_2017Q4 \
    --syms $SYM \
    --months 201611 201612 201701 201702 201703 201704 201705 201706 201707 201708 201709 201710 201711 201712 \
    --durations --out $DATA/briefstates/$SYM
