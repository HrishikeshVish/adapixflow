#!/bin/bash
# Train AdaPixFlow.
#
#   bash scripts/train.sh [config] [gpu]
#   bash scripts/train.sh config/physinone/train.yaml 0
# Output lands in experiments/<name>/ (last.pt, eval_step<k>.json, train.log).
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
CFG=${1:-config/physinone/train.yaml}
GPU=${2:-${CUDA_VISIBLE_DEVICES:-0}}
echo "[adapixflow] training with $CFG on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" train.py "$CFG"
