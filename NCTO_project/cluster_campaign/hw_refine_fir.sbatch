#!/bin/bash
#SBATCH --job-name=ncto_hwref
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=ncto_hwref_%j.log
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
B1="$REPO/NCTO_project/scripts/gneb_finite_size_scan.py"
echo "==== $(date) refined hw scan (heavier GNEB convergence) ===="
# separate output tree so we don't clobber the coarse scan; --force fresh endpoints
for hw in 1.40 1.50 1.60 1.70 1.80; do
  python3 "$B1" --lattices 36 --half-width "$hw" --force \
    --n-iter 3000 --climb-start 1800 --plateau-win 250 > hwref_L36_hw${hw}.log 2>&1 &
done
wait
echo "==== $(date) HW REFINE DONE ===="
