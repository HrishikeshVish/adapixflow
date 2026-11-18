#!/bin/bash
# Download the 175 PhysInOne scenes of the paper (121 GB of zips, 48 GB of caches) and convert each one to a
# cache in data/physinone/cache. Each zip is deleted after conversion, and an interrupted run resumes where it stopped.
#
#   bash scripts/download_physinone.sh [workers]
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
"$PYTHON" dataset/fetch_physinone.py splits/physinone_sources.json "$DATA_ROOT/cache" --workers "${1:-4}"
