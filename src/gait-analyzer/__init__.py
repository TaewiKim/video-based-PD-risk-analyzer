"""Research Automation: Video-based Health Monitoring.

A comprehensive toolkit for automating research workflows in video-based
health monitoring, including literature search, video collection, and
quality assessment.
"""

__version__ = "0.1.0"

from research_automation.core import (
    Base,
    Settings,
    StorageManager,
    get_session,
    get_settings,
    init_db,
)

__all__ = [
    "Base",
    "Settings",
    "StorageManager",
    "__version__",
    "get_session",
    "get_settings",
    "init_db",
]
