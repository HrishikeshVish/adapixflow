#!/bin/bash
# Estimated inputs of the comparison scenes (Sec. 4): depth from Depth Anything 3, masks from SAM 2.1, fused on the grids
# of data/physinone/fused (after scripts/prepare_physinone.sh). Needs a GPU, and DA3 in its own environment:
#   git submodule update --init third_party/Depth-Anything-3
#   (DA3 env)  pip install -e third_party/Depth-Anything-3
#   (main env) pip install transformers==5.17.0      # SAM 2.1 (facebook/sam2.1-hiera-large)
#
#   DA3_PYTHON=/path/to/da3/env/bin/python bash scripts/prepare_estimated.sh [gpu]
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
GPU=${1:-0}; DA3_PYTHON=${DA3_PYTHON:-$PYTHON}
P=$DATA_ROOT; E=$P/estimated
SCENES=$("$PYTHON" -c "import json; print(' '.join(json.load(open('splits/physinone.json'))['comparison']))")
for n in $SCENES; do
  "$PYTHON" dataset/estimated/training_views.py $P/cache/$n.npz $E/views
  CUDA_VISIBLE_DEVICES=$GPU "$DA3_PYTHON" dataset/estimated/da3_depth.py $E/views/$n.npz $E/depth_da3giant/$n.npz \
      --model da3-giant --frames stored --device cuda
  for task in inbetween future; do
    # future fusions only for the scenes with future targets (splits/estimated_far.json)
    "$PYTHON" -c "import json, sys; sys.exit('$task' not in json.load(open('splits/estimated_far.json'))['$n'])" || continue
    frames=s4; [ $task = future ] && frames=future
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" dataset/estimated/sam_masks.py $E/views/$n.npz $E/depth_da3giant/$n.npz \
        $E/masks_$task/$n.npz --frames $frames --v2 --device cuda
    "$PYTHON" dataset/estimated/fuse_estimated.py $n --task $task --views $E/views --depth $E/depth_da3giant \
        --masks $E/masks_$task --bounds-like $P/fused --out $E/fused_$task
  done
done
