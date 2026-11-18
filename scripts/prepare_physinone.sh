#!/bin/bash
# Fuse the cached scenes into voxel grids (about 9 GB):
#   data/physinone/fused           every scene, all cameras (training inputs and ground truth)
#   data/physinone/fused_holdout   the six comparison scenes without views 0 and 1 (their inputs), on the same grids
#
#   bash scripts/prepare_physinone.sh [workers]
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
W=${1:-4}
"$PYTHON" dataset/fuse.py "$DATA_ROOT/cache" "$DATA_ROOT/fused" --split splits/physinone.json --workers "$W"
COMP=$("$PYTHON" -c "import json; print(','.join(json.load(open('splits/physinone.json'))['comparison']))")
"$PYTHON" dataset/fuse.py "$DATA_ROOT/cache" "$DATA_ROOT/fused_holdout" --scenes "$COMP" --holdout 0,1 \
    --bounds-like "$DATA_ROOT/fused" --workers "$W"
