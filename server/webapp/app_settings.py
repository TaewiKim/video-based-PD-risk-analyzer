from dataclasses import dataclass
from pathlib import Path

from django.conf import settings


@dataclass(frozen=True)
class WebappSettings:
    runtime_dir: Path
    upload_dir: Path
    results_dir: Path
    data_dir: Path
    profile_dir: Path
    allowed_video_extensions: tuple[str, ...]


app_settings = WebappSettings(
    runtime_dir=Path(settings.RUNTIME_DIR),
    upload_dir=Path(settings.UPLOAD_DIR),
    results_dir=Path(settings.RESULTS_DIR),
    data_dir=Path(settings.DATA_DIR),
    profile_dir=Path(settings.PROFILE_DIR),
    allowed_video_extensions=tuple(settings.ALLOWED_VIDEO_EXTENSIONS),
)
