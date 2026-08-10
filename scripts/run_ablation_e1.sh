#!/bin/bash
# E1 ablation runs for the PRX submission — four sequential 2-epoch trainings
# on GPUs 0-3 (8192/GPU x 4 = 32,768 effective batch, matching the paper run).
#
# Usage:  bash scripts/run_ablation_e1.sh [start_variant]
#   start_variant: optional (V0|V1|V2|V3) — skip variants before it, e.g. to
#   resume the sequence after a crash without retraining finished variants.
#
# Order: V0 (reference) first so the baseline exists even if later variants
# hit trouble; Euclidean variants after their manifold counterparts.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0,1,2,3

LOGDIR=logs/ablation_e1
mkdir -p "$LOGDIR"

VARIANTS=(V0_manifold_aux V1_euclidean_aux V2_manifold_noaux V3_euclidean_noaux)
START=${1:-V0}
skipping=1

for V in "${VARIANTS[@]}"; do
  [[ "$V" == "$START"* ]] && skipping=0
  if [[ $skipping -eq 1 ]]; then
    echo "skip $V (starting from $START)"
    continue
  fi
  stamp=$(date +%Y%m%d_%H%M%S)
  log="$LOGDIR/${V}_${stamp}.log"
  echo "=== $(date -u +%FT%TZ) launching $V -> $log ==="
  uv run python scripts/train_riemannian_DiTs.py \
    --config "configs/ablation_e1/train_riemannian_DiTs_ablE1_${V}.yaml" \
    > "$log" 2>&1
  status=$?
  echo "=== $(date -u +%FT%TZ) $V finished with exit $status ==="
  if [[ $status -ne 0 ]]; then
    echo "ABORTING SEQUENCE: $V failed (exit $status). Later variants not started."
    echo "Diagnose via $log; resume with: bash scripts/run_ablation_e1.sh <variant>"
    exit $status
  fi
done
echo "=== ALL E1 ABLATION RUNS COMPLETE ==="
