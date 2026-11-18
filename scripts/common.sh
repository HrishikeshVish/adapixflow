#!/bin/bash
# Shared settings of the scripts. Override PYTHON or DATA_ROOT from the environment.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data/physinone}
