"""Analysis pipelines for video-based health monitoring."""

from .gait_baseline import (
    SMPL_JOINT_NAMES,
    extract_features_batch,
    extract_gait_features,
    get_feature_importance,
    load_care_pd_data,
    print_results,
    train_updrs_classifier,
)

try:
    from .gait_sequence_model import (
        SequenceTrainingConfig,
        build_sequence_dataset,
        train_sequence_cv,
    )
except ImportError:
    SequenceTrainingConfig = None
    build_sequence_dataset = None
    train_sequence_cv = None

__all__ = [
    "SMPL_JOINT_NAMES",
    "extract_features_batch",
    "extract_gait_features",
    "get_feature_importance",
    "load_care_pd_data",
    "print_results",
    "train_updrs_classifier",
    "SequenceTrainingConfig",
    "build_sequence_dataset",
    "train_sequence_cv",
]
