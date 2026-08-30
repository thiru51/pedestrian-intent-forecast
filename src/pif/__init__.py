"""Multi-modal pedestrian crossing-intent forecasting."""

from __future__ import annotations

__version__ = "0.1.0"

# Frame rate both JAAD and PIE were recorded at. Everything that talks about a
# lead time in seconds converts through this, so it lives in one place.
FPS: int = 30
