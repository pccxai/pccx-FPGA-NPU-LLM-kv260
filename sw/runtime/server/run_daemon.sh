#!/usr/bin/env bash
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)

cd "$REPO_ROOT"
exec python3 -m sw.runtime.server \
    --model "$HOME/models/gemma-3n-e4b-int4" \
    --host 0.0.0.0 \
    --port 7860 \
    "$@"
