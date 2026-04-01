#!/bin/bash
# Legacy startup script (compatibility wrapper).
# The main entrypoint is `./server.sh`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Optional first argument: config file path
if [ "${1:-}" != "" ]; then
  ./server.sh start "$1"
else
  ./server.sh start
fi

