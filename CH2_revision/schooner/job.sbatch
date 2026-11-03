#!/bin/bash
#SBATCH --job-name=mwr_revision
#SBATCH --partition=32gb_20core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=28G
#SBATCH --time=24:00:00
#SBATCH --output=mwr_revision_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-20}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mwr_matplotlib_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
MWR_PYTHON="${MWR_PYTHON:-python}"
MWR_DATA_ROOT="${MWR_DATA_ROOT:-/condo/swat_z6/tpardun/phd_stuff}"
"$MWR_PYTHON" -m unittest -v test_barnes_uncertainty.py
"$MWR_PYTHON" run_revision.py --root "$MWR_DATA_ROOT" --output "results/run_${SLURM_JOB_ID}"
