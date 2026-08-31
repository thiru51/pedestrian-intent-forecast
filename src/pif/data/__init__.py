"""Dataset parsing, windowing and torch plumbing.

Deliberately does not import .dataset here. Parsing JAAD/PIE annotations and
building windows is pure numpy, and keeping torch off this import path means
`python -m scripts.parse_annotations` works in an environment that has not
installed it. Import pif.data.dataset explicitly when you want tensors.
"""

from __future__ import annotations

from .jaad import JAADAnnotations
from .pie import PIEAnnotations
from .schema import PedTrack, TrafficObject, VideoAnnotation
from .windows import Window, summarise, track_windows, video_windows

__all__ = [
    "JAADAnnotations",
    "PIEAnnotations",
    "PedTrack",
    "TrafficObject",
    "VideoAnnotation",
    "Window",
    "summarise",
    "track_windows",
    "video_windows",
]
