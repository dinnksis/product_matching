#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="$(cd "$(dirname "$0")" && pwd)"
PYARROW_RUNTIME="$SUBMISSION_DIR/vendor/pyarrow_runtime"

if ! python -c 'import pyarrow' >/dev/null 2>&1; then
    export PYTHONPATH="$PYARROW_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
fi

python -c 'import pyarrow' >/dev/null
exec python -u "$SUBMISSION_DIR/run.py" "$@"
