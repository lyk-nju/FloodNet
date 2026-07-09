"""True initial z_T oracle entrypoint for stream_generate_step.

This line-B oracle keeps an independent Gaussian initial-noise buffer and
optimizes that buffer through differentiable stream denoising.  It intentionally
does not reuse ``model.generated[start:end]`` as the optimization variable,
because that cache contains the current active noisy state ``x_beta`` during
streaming.

The first implementation delegates to the full-stream initial-noise optimizer
(``optimize_stream_noise.py``).  It is a true z_T baseline, but not yet the
local receding-horizon z_T MPC variant; summary files explicitly record
``commit_strategy=full_stream_shadow_rollout``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.ldf.latent_initializer.optimize_stream_noise import main


if __name__ == "__main__":
    raise SystemExit(main())
