#!/bin/bash
# Convert all datasets with --max-particles 8 filter.
# Output goes to data/run2_data_*_reduced/
# Order: 4lep (smallest) -> exactly3lep -> GamGam -> 2to4lep (largest)

set -e

MAX_PARTICLES=8
BASE_OUT="data"

echo "============================================================"
echo "Converting all datasets with max_particles=${MAX_PARTICLES}"
echo "============================================================"

echo ""
echo "[1/4] 4lep (smallest, ~1.5M events)"
uv run python -m shellflow.rootfile_to_numpy \
    --dataset 4lep \
    --max-particles ${MAX_PARTICLES} \
    --output-dir ${BASE_OUT}/run2_data_4lep_reduced

echo ""
echo "[2/4] exactly3lep (~15.7M events)"
uv run python -m shellflow.rootfile_to_numpy \
    --dataset exactly3lep \
    --max-particles ${MAX_PARTICLES} \
    --output-dir ${BASE_OUT}/run2_data_exactly3lep_reduced

echo ""
echo "[3/4] GamGam (~36.6M events)"
uv run python -m shellflow.rootfile_to_numpy \
    --dataset GamGam \
    --max-particles ${MAX_PARTICLES} \
    --output-dir ${BASE_OUT}/run2_data_GamGam_reduced

echo ""
echo "[4/4] 2to4lep (largest, ~284M events)"
uv run python -m shellflow.rootfile_to_numpy \
    --dataset 2to4lep \
    --max-particles ${MAX_PARTICLES} \
    --output-dir ${BASE_OUT}/run2_data_2to4lep_reduced

echo ""
echo "============================================================"
echo "All conversions complete!"
echo "Output directories:"
ls -lhd ${BASE_OUT}/run2_data_*_reduced/
echo "============================================================"
