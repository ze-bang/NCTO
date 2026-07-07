#!/bin/bash
#SBATCH --job-name=ncto_swfix
#SBATCH --account=rrg-ybkim_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=ncto_swfix_%j.log
# Validate the seed fix restores a threshold drive: clean line (sigma=0), hw=1.8,
# E0 sweep, 1 seed. Expect final_r3 to drop / quench ZZ to rise above some E0.
set -uo pipefail
REPO=/project/6003507/zhouzb79/NCTO
source "$REPO/NCTO_project/cluster_campaign/env_fir.sh"
cd "$REPO"
python3 NCTO_project/scripts/run_campaign.py switching \
  --half-width 1.8 --sigma-k-values 0.0 --seeds 1 \
  --e0-values 0 8 12 16 20 25 30 40 \
  --output-suffix fixtest --workers 16
echo "=== fixtest raw metrics vs E0 ==="
python3 - <<'PY'
import csv, glob
f=glob.glob("NCTO_project/tuned_kruger_campaign/pinning_switching_crosscheck_L36/analysis/*fixtest*crosscheck.csv")
rows=sorted(csv.DictReader(open(f[0])), key=lambda r:float(r['e0']))
print(f"{'E0':>5} {'final_r3':>9} {'final_zzf':>9} {'quench_zzf':>10} {'written':>8}")
for r in rows:
    print(f"{r['e0']:>5} {float(r['final_r3']):>9.3f} {float(r['final_zz_frac']):>9.3f} {float(r.get('quench_zz_frac',r['final_zz_frac'])):>10.3f} {r['written']:>8}")
PY
echo "==== $(date) FIXTEST DONE ===="
