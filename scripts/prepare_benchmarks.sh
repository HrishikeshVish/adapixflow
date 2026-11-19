#!/bin/bash
# Convert a physics benchmark into scene caches and fuse them (after scripts/download_benchmarks.sh):
#   data/<benchmark>/cache           scene caches (the PhysInOne layout)
#   data/<benchmark>/fused           all cameras (ground truth)
#   data/<benchmark>/fused_holdout   without views 0 and 1 (the model's input), on the same grids
#
#   bash scripts/prepare_benchmarks.sh springgaus|physgaia|kubric4d [workers]
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
[ $# -ge 1 ] || { sed -n 2,7p "$0"; exit 1; }
B=$1; W=${2:-2}; D=data/$B
FUSE=(--voxel 0.0125 --max-extent 4.0)
case $B in
  springgaus)
    # every native frame written twice (--repeat 2), so that one step (two video frames) is one native frame
    for s in torus cross apple cream toothpaste chess banana; do
      "$PYTHON" -I dataset/benchmarks/springgaus.py $s --repeat 2 --sim $D/raw/mpm_synthetic/simulation \
          --render $D/raw/mpm_synthetic/render --out $D/cache
    done ;;
  physgaia)
    "$PYTHON" -I dataset/benchmarks/physgaia_particles.py $D/visco_bouncing_balls $D/_particles/visco_bouncing_balls
    PHYSGAIA_ROOT=$D "$PYTHON" -I dataset/benchmarks/physgaia.py visco_bouncing_balls --scale 0.2 --step 1 --first 26 \
        --n-latent 90 --name visco_bouncing_balls__s0.2_n1_f26 --out $D/cache ;;
  kubric4d)
    for s in scn02700 scn02800 scn02801 scn02802 scn02803 scn02804; do
      src=$D/raw/val/extract/$s; [ $s = scn02700 ] && src=$D/raw/tiny/extract/$s
      "$PYTHON" -I dataset/benchmarks/kubric4d.py $src --out $D/cache --dup 2 --compress --moving d4 --suffix _f1
    done
    FUSE=(--voxel 0.1 --max-extent 16.0) ;;
  *) echo "unknown benchmark $B"; exit 1 ;;
esac
"$PYTHON" dataset/fuse.py $D/cache $D/fused --split splits/$B.json "${FUSE[@]}" --workers "$W"
"$PYTHON" dataset/fuse.py $D/cache $D/fused_holdout --split splits/$B.json --holdout 0,1 --bounds-like $D/fused \
    "${FUSE[@]}" --workers "$W"
