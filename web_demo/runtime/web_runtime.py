"""Web runtime facade placeholder.

The concrete runtime is still exposed through `web_demo.model_manager.ModelManager`
during the staged refactor. This module anchors the new dependency direction.
"""

from __future__ import annotations


class WebRuntime:
    """Marker base for the staged web runtime facade."""


__all__ = ["WebRuntime"]

