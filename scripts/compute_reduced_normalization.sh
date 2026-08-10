#!/bin/bash
# Compute normalization + manifold stats for all reduced datasets.
# Creates new base configs pointing to the reduced data directories,
# then runs the normalizer on each.
#
# Requires: the reduced datasets to already exist in
#   data/run2_data_{name}_reduced/

set -e

CONFIG_DIR="src/shellflow/config"
BASE_OUT="data"

DATASETS=("4lep" "exactly3lep" "GamGam" "2to4lep")

echo "============================================================"
echo "Creating base configs for reduced datasets"
echo "============================================================"

for ds in "${DATASETS[@]}"; do
    src="${CONFIG_DIR}/dataloader_config_base_06_${ds}.yaml"
    dst="${CONFIG_DIR}/dataloader_config_base_06_${ds}_reduced.yaml"
    out="dataloader_config_06_${ds}_reduced.yaml"

    if [ ! -f "$src" ]; then
        echo "WARNING: Source config not found: $src — skipping $ds"
        continue
    fi

    # Copy base config and replace data paths
    sed "s|run2_data_${ds}_converted|run2_data_${ds}_reduced|g" "$src" > "$dst"
    echo "Created $dst"
done

echo ""
echo "============================================================"
echo "Computing normalization + manifold stats for reduced datasets"
echo "============================================================"

for ds in "${DATASETS[@]}"; do
    base="dataloader_config_base_06_${ds}_reduced.yaml"
    out="dataloader_config_06_${ds}_reduced.yaml"

    if [ ! -f "${CONFIG_DIR}/${base}" ]; then
        echo "WARNING: Config not found: ${CONFIG_DIR}/${base} — skipping $ds"
        continue
    fi

    echo ""
    echo "[${ds}] Running normalizer..."
    echo "  Input:  ${base}"
    echo "  Output: ${out}"
    uv run python -m shellflow.data.normalizer \
        --config "$base" \
        --output "$out" \
        --manifold-stats

    echo "[${ds}] Done."
done

echo ""
echo "============================================================"
echo "All normalization computations complete!"
echo "Output configs:"
for ds in "${DATASETS[@]}"; do
    echo "  ${CONFIG_DIR}/dataloader_config_06_${ds}_reduced.yaml"
done
echo "============================================================"
