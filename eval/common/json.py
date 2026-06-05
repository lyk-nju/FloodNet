from __future__ import annotations

import json
import math
from pathlib import Path


def json_sanitize(value):
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None

    if isinstance(value, dict):
        return {key: json_sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_sanitize(val) for val in value]
    if isinstance(value, tuple):
        return [json_sanitize(val) for val in value]
    if np is not None and isinstance(value, np.generic):
        return json_sanitize(value.item())
    if np is not None and isinstance(value, np.ndarray):
        return json_sanitize(value.tolist())
    if torch is not None and torch.is_tensor(value):
        if value.ndim == 0:
            return json_sanitize(value.detach().cpu().item())
        return json_sanitize(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_strict(path: str | Path, payload) -> None:
    with open(path, "w") as f:
        json.dump(json_sanitize(payload), f, indent=2, allow_nan=False)
