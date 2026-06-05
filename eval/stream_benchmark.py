"""Compatibility wrapper for the layered runtime benchmark module."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.runtime.benchmark import *  # noqa: F401,F403
from eval.runtime.benchmark import main
from eval.runtime import benchmark as _compat_impl

for _compat_name in dir(_compat_impl):
    if _compat_name.startswith("_") and not _compat_name.startswith("__"):
        globals()[_compat_name] = getattr(_compat_impl, _compat_name)
del _compat_impl, _compat_name


if __name__ == "__main__":
    main()
