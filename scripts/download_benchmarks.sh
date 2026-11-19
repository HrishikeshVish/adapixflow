#!/bin/bash
# Download the raw data of a physics benchmark (Tab. 2).
#
#   bash scripts/download_benchmarks.sh springgaus   # Spring-Gaus mpm_synthetic (2.1 GB)
#   bash scripts/download_benchmarks.sh physgaia     # PhysGaia visco_bouncing_balls (Hugging Face)
#   bash scripts/download_benchmarks.sh kubric4d     # Kubric-4D scn02700, scn02800-02804 (GCD release, streamed)
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
[ $# -ge 1 ] || { sed -n 2,6p "$0"; exit 1; }
case $1 in
  springgaus)
    # Spring-Gaus (Zhong et al., ECCV 2024): mpm_synthetic.zip from the authors' Google Drive folder
    # https://drive.google.com/drive/folders/1SN4vrPZSuBnbkDTLl0nu9g-boVUr5yO1 (see github.com/Colmar-zlicheng/Spring-Gaus).
    R=data/springgaus/raw
    mkdir -p "$R"
    if [ ! -f "$R/mpm_synthetic.zip" ]; then
      echo "Download mpm_synthetic.zip from the folder above into $R/ (or: gdown --folder 1SN4vrPZSuBnbkDTLl0nu9g-boVUr5yO1 -O $R)."
      exit 1
    fi
    unzip -q -n "$R/mpm_synthetic.zip" -d "$R"   # -> $R/mpm_synthetic/{render,simulation}
    ;;
  physgaia)
    # PhysGaia (Kim et al.): https://huggingface.co/datasets/mijeongkim/PhysGaia (CC BY-NC 4.0)
    R=data/physgaia
    mkdir -p "$R/raw"
    [ -f "$R/raw/visco_bouncing_balls.zip" ] || curl -L --fail -o "$R/raw/visco_bouncing_balls.zip" \
        https://huggingface.co/datasets/mijeongkim/PhysGaia/resolve/main/visco_bouncing_balls.zip
    "$PYTHON" -I dataset/benchmarks/physgaia_unzip.py "$R/raw/visco_bouncing_balls.zip" visco_bouncing_balls "$R"
    ;;
  kubric4d)
    # Kubric-4D (Van Hoorick et al., GCD, ECCV 2024): only the members of the six scenes are kept
    U=https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/urp/datasets/tcow
    "$PYTHON" -I dataset/benchmarks/kubric4d_download.py $U/gcd_kubric4d_tiny.tar.gz data/kubric4d/raw/tiny \
        --scenes scn02700 --max-gb 6
    "$PYTHON" -I dataset/benchmarks/kubric4d_download.py $U/gcd_kubric4d_val.tar.gz data/kubric4d/raw/val \
        --scenes scn02800,scn02801,scn02802,scn02803,scn02804 --max-gb 16
    ;;
  *) echo "unknown benchmark $1"; exit 1 ;;
esac
