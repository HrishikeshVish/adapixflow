#!/bin/bash
# Evaluate a checkpoint with one of the configs in config/.
#
#   bash scripts/evaluate.sh physinone/comparison [checkpoint] [gpu]   # Tab. 1 and the motion split of Tab. 9
#   bash scripts/evaluate.sh physinone/heldout [checkpoint] [gpu]      # Tab. 3, Test and Val scenes
#   bash scripts/evaluate.sh physinone/estimated [checkpoint] [gpu]    # estimated depth and masks
#   bash scripts/evaluate.sh springgaus/comparison [checkpoint] [gpu]  # Tab. 2 and 10 (also physgaia/, kubric4d/)
# The default checkpoint is the released model, checkpoints/adapixflow_physinone.pt.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
[ $# -ge 1 ] || { sed -n 2,9p "$0"; exit 1; }
CKPT=${2:-checkpoints/adapixflow_physinone.pt}
GPU=${3:-${CUDA_VISIBLE_DEVICES:-0}}
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" evaluate.py "config/${1%.yaml}.yaml" --checkpoint "$CKPT"
