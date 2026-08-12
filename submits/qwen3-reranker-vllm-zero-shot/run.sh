#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="$(cd "$(dirname "$0")" && pwd)"
PYARROW_RUNTIME="$SUBMISSION_DIR/vendor/pyarrow_runtime"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter is unavailable: $PYTHON_BIN" >&2
    exit 127
fi

if ! "$PYTHON_BIN" -c 'import pyarrow' >/dev/null 2>&1; then
    export PYTHONPATH="$PYARROW_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
fi

"$PYTHON_BIN" -c 'import pyarrow' >/dev/null
exec "$PYTHON_BIN" -u "$SUBMISSION_DIR/run.py" "$@"
