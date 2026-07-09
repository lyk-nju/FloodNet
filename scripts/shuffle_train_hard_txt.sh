#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/data/home/shengqiuProf_user_yuankai/miniconda3/envs/flooddiffusion/bin/python"
TARGET="/data/home/shengqiuProf_user_yuankai/FloodDiffusion/raw_data/HumanML3D/train_hard.txt"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python env not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$TARGET" ]]; then
    echo "Target txt not found: $TARGET" >&2
    exit 1
fi

"$PYTHON_BIN" - "$TARGET" <<'PY'
import random
import sys
from datetime import datetime
from pathlib import Path

target = Path(sys.argv[1])
lines = [line.strip() for line in target.read_text().splitlines() if line.strip()]
if not lines:
    raise SystemExit(f"Target txt is empty: {target}")

backup = target.with_name(
    f"{target.name}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
tmp = target.with_name(f"{target.name}.tmp_shuffle")

backup.write_text("\n".join(lines) + "\n")
random.shuffle(lines)
tmp.write_text("\n".join(lines) + "\n")
tmp.replace(target)

print(f"shuffled: {target}")
print(f"backup:   {backup}")
print(f"num_lines: {len(lines)}")
print("head:")
for item in lines[:10]:
    print(item)
PY
