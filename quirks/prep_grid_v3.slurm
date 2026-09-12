#!/bin/bash
#SBATCH --job-name=quirk-prep-v3
#SBATCH --partition=sooner_test
#SBATCH --account=general
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --array=0-10
#SBATCH --output=/ourdisk/hpc/ouhep/jburzyns/dont_archive/Quirks/hepattn/logs/prepv3_%A_%a.out
# grid_v3 dataset, one array task per mass:
#   selection: exactly 2 quirks with pt>5, |eta|<2.5, >=5 spacepoints
#   train/val: Lambda <= 3000 only (7 points), merged across points
#   test:      ALL Lambda, per-point dirs test/m<M>.L<L>/ (efficiency grid)
#   file split per point: last 2 -> test, next 2 -> val, rest -> train
MASSES=(50 75 100 125 150 175 200 225 250 275 300)
MI=$SLURM_ARRAY_TASK_ID
M=${MASSES[$MI]}
C=/home/jburzyns/ATLAS/Quirks/quirk-chain
D=/ourdisk/hpc/ouhep/jburzyns/dont_archive/Quirks/chain/gnndump
T=/ourdisk/hpc/ouhep/jburzyns/dont_archive/Quirks/hepattn
PY="$C/.venv/bin/python"
G=$T/data/grid_v3
mkdir -p $G/train $G/val
LAMS=(100 167 278 464 774 1292 2154 3594 5995 10000)
for li in "${!LAMS[@]}"; do
  l=${LAMS[$li]}
  (
    files=($D/m$M.L$l/*.root); n=${#files[@]}
    [ "$n" -eq 0 ] || [ ! -e "${files[0]}" ] && { echo "m$M.L$l: NO FILES"; exit 0; }
    mkdir -p $G/test/m$M.L$l
    base=$(( (MI*10 + li) * 100000 ))
    for i in "${!files[@]}"; do
      off=$(( base + i*1000 ))
      if   (( i >= n-2 )); then out=$G/test/m$M.L$l
      elif (( i >= n-4 )); then out=$G/val;   (( l > 3000 )) && continue
      else                      out=$G/train; (( l > 3000 )) && continue; fi
      env -u PYTHONPATH $PY $C/Tracking/quirks/prep_quirks.py "${files[$i]}" $out \
        --mass $M --lam $l --quirks-only --min-hits 5 --exactly-two --offset $off 2>/dev/null
    done
    echo "m$M.L$l done ($n files)"
  ) &
done
wait
echo "PREPV3_M${M}_DONE"
