#!/bin/bash
#SBATCH --job-name=ncto_smoke
#SBATCH --account=def-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=ncto_smoke_%j.log
# ---------------------------------------------------------------------------
# One MD point on a compute node: validates the module env, the solver's
# HDF5 write, and that the driver reads /trajectory/spins back correctly
# BEFORE committing the multi-hour campaign.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
python3 -c "import numpy,h5py; print('py',numpy.__version__,'h5py',h5py.__version__,h5py.version.hdf5_version)"
# fresh single MD point (distinct lambda so it is not the corrupt login-node cache)
python3 NCTO_project/scripts/run_campaign.py drive-pd --all-channels \
  --j7-values -0.40 --lambda-values 0.03 --e0-values 4 --workers 1
echo "=== SMOKE RESULT ==="
python3 - <<'PY'
import glob, h5py, numpy as np
fs = glob.glob("NCTO_project/tuned_kruger_campaign/phase_diagram_allchan/runs/J70400_lK030_E00400/sample_0/trajectory.h5")
assert fs, "no trajectory written"
with h5py.File(fs[0],"r") as f:
    sp = np.array(f["trajectory/spins"][-1])
print("READABLE h5, final spins", sp.shape, "-> SMOKE PASS")
PY
