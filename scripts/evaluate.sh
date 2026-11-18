#!/bin/bash
# Evaluate a checkpoint.
#
#   bash scripts/evaluate.sh heldout [checkpoint] [gpu]      # Tab. 3, Test and Val scenes
#   bash scripts/evaluate.sh comparison [checkpoint] [gpu]   # Tab. 1, comparison scenes
# The default checkpoint is the released model, checkpoints/adapixflow_physinone.pt.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
[ $# -ge 1 ] || { sed -n 2,7p "$0"; exit 1; }
CKPT=${2:-checkpoints/adapixflow_physinone.pt}
GPU=${3:-${CUDA_VISIBLE_DEVICES:-0}}
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" evaluate.py "config/physinone/$1.yaml" --checkpoint "$CKPT"
