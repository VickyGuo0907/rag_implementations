#!/bin/bash
# RAG Runner - uses venv Python automatically

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PYTHON"
    exit 1
fi

# Run with venv Python
exec "$VENV_PYTHON" "$SCRIPT_DIR/scripts/run_technique.py" "$@"
