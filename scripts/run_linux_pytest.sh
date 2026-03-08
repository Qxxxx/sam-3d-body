#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <pytest-args...>" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "run_linux_pytest.sh only supports Linux." >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "sam_3d_body" ]]; then
  echo "expected active conda environment: sam_3d_body" >&2
  exit 1
fi

python -m pytest "$@"
