"""Data augmentation for video and pose sequences."""

from .pose import (
    AugmentationConfig,
    PoseAugmentor,
    SequenceDataset,
    augment_sequence,
)
from .video import (
    VideoAugmentationConfig,
    VideoAugmentor,
    augment_video,
)

__all__ = [
    # Pose
    "AugmentationConfig",
    "PoseAugmentor",
    "SequenceDataset",
    "augment_sequence",
    # Video
    "VideoAugmentationConfig",
    "VideoAugmentor",
    "augment_video",
]
