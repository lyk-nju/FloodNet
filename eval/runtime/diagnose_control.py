import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval.diagnose_stream_control import *  # noqa: F401,F403
from eval.diagnose_stream_control import main


if __name__ == "__main__":
    main()
