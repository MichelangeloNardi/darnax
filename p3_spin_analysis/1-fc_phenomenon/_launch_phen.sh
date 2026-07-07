#!/bin/bash
# Cluster launcher for the phenomenon analysis. Usage: _launch_phen.sh <frozen|trainwin>
cd ~/projects/darnax || exit 1
VARIANT="$1"
CFG="replicate/tuned_fc_${VARIANT}.json"
RES=p3_spin_analysis/1-fc_phenomenon/results
mkdir -p "$RES"
PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python \
  p3_spin_analysis/1-fc_phenomenon/phenomenon.py --cfg "$CFG" \
  >> "$RES/phen_${VARIANT}.log" 2>&1
