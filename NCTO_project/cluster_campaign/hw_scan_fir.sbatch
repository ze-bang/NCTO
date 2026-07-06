#!/bin/bash
#SBATCH --job-name=ncto_hwscan
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=ncto_hwscan_%j.log
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
B1="$REPO/NCTO_project/scripts/gneb_finite_size_scan.py"
echo "==== $(date) hw scan: dG(hw) at L=36 (+ L=54 anchors) to hit 0.8 ms => 17.7 meV ===="
# L=36 depinning barrier vs hw (hw=2.0 already cached from B1)
for hw in 1.00 1.25 1.50 1.75; do
  python3 "$B1" --lattices 36 --half-width "$hw" > hwscan_L36_hw${hw}.log 2>&1 &
done
# L=54 anchors near the expected hw* for the L-independence / line-stiffness check
for hw in 1.50 1.75; do
  python3 "$B1" --lattices 54 --half-width "$hw" > hwscan_L54_hw${hw}.log 2>&1 &
done
wait
echo "==== $(date) HW SCAN DONE ===="
