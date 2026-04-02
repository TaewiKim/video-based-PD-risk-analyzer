"""Compatibility package for the legacy source tree.

The repository's code lives under ``src/gait-analyzer`` while the rest of the
project imports ``research_automation``. Extend the package search path so
existing imports resolve without moving the entire source tree.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_LEGACY_DIR = _PACKAGE_DIR.parent / "gait-analyzer"

__path__ = [str(_PACKAGE_DIR)]
if _LEGACY_DIR.is_dir():
    __path__.append(str(_LEGACY_DIR))

__version__ = "0.1.0"

from .core import Base, Settings, StorageManager, get_session, get_settings, init_db

__all__ = [
    "Base",
    "Settings",
    "StorageManager",
    "__version__",
    "get_session",
    "get_settings",
    "init_db",
]
