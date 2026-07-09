#!/bin/bash
#SBATCH --job-name=ncto_wsweep
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=ncto_wsweep_%j.log
# Clean, consistent GNEB kinetic-barrier vs defect half-width at L=36 (3000 iters).
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
B1="$REPO/NCTO_project/scripts/gneb_finite_size_scan.py"
echo "==== $(date) clean width sweep L=36 (3000 iters) ===="
for hw in 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 3.00; do
  python3 "$B1" --lattices 36 --half-width "$hw" --force \
    --n-iter 3000 --climb-start 1800 --plateau-win 250 > wsweep_hw${hw}.log 2>&1 &
done
wait
echo "==== $(date) WIDTH SWEEP DONE ===="
