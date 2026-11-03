#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export MWR_DATA_ROOT="${1:-/condo/swat_z6/tpardun/phd_stuff}"
export MWR_PYTHON="${MWR_PYTHON:-$(command -v python)}"
"$MWR_PYTHON" -c 'import numpy, scipy, pandas, matplotlib, openpyxl; print("Python dependencies available")'
partition="${MWR_PARTITION:-32gb_20core}"
if ! sinfo -h -o '%P' | sed 's/\*$//' | awk -v wanted="$partition" '$0==wanted{ok=1} END{exit !ok}'; then
    printf 'Partition %s is not visible. Check sinfo and set MWR_PARTITION to the appropriate 20-core partition.\n' "$partition" >&2
    exit 1
fi
mkdir -p results
sbatch --partition="$partition" --export=ALL job.sbatch
