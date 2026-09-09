#!/bin/bash
# One-time setup: bakes your repo path and scratch area into the slurm
# scripts. Run from the repo root:  ./bootstrap.sh [scratch_dir]
set -e
QT=$(cd "$(dirname "$0")" && pwd)
SCRATCH=${1:-/ourdisk/hpc/ouhep/$USER/dont_archive/quirk-tracking}
mkdir -p $SCRATCH/{containers,pixi-home,pixi-cache,apptainer-cache,data,logs}
mkdir -p $QT/slurm
for t in $QT/templates/*.slurm.in; do
  sed -e "s|@QT@|$QT|g" -e "s|@SCRATCH@|$SCRATCH|g" $t > $QT/slurm/$(basename ${t%.in})
done
echo "QT=$QT"        >  $QT/env.sh
echo "SCRATCH=$SCRATCH" >> $QT/env.sh
echo "Wrote slurm/*.slurm and env.sh (scratch: $SCRATCH)"
