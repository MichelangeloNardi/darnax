#!/bin/bash
# Cluster launcher (avoids nested-ssh quoting). Usage: _launch_tune.sh <frozen|trainwin> [trials]
cd ~/projects/darnax || exit 1
VARIANT="$1"
TRIALS="${2:-40}"
RES=p3_spin_analysis/1-fc_phenomenon/results
mkdir -p "$RES"
PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/miniforge3/envs/darnax_hpc/bin/python \
  p3_spin_analysis/1-fc_phenomenon/tune.py --variant "$VARIANT" --trials "$TRIALS" \
  >> "$RES/tune_${VARIANT}.log" 2>&1
