#!/bin/bash
# The two missing bias cells of the baseline-resolution table (revision item A16): parametric
# bootstrap of INTC, May 2017, E1m with K = 4 and the 1 ms dead time, baseline bins of 5 and
# 10 minutes, 30 replicates each (as for the 60-, 110- and 330-minute rows). Two array tasks.
#
#     sbatch scripts/cluster/bootstrap_bins.sbatch
#
# Output: ~/hawkes-taq-data/boot_bins/INTC_201705_B<minutes>/{fit,replicates,summary}.csv
# (resumable).
#
#SBATCH --job-name=boot_bins
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/boot_bins_%a_%A.o
#SBATCH --error=logs/boot_bins_%a_%A.e

set -euo pipefail
mkdir -p logs
module load python/3.11.11
source ~/venvs/hawkes/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

BINS=(5 10)
B=${BINS[$SLURM_ARRAY_TASK_ID]}
DATA=~/hawkes-taq-data
python scripts/bootstrap_ks.py --nbbo $DATA/nbbo_2017Q2 --sym INTC --month 201705 \
    --definition E1m --K 4 --delta 0.001 --starts full --bin-minutes $B --n-boot 30 \
    --workers $SLURM_CPUS_PER_TASK --out-dir $DATA/boot_bins/INTC_201705_B${B}
