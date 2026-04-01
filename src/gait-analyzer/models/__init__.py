"""Machine learning models for video-based health analysis."""

from .base import (
    BaseModel,
    FeatureExtractor,
    InferenceResult,
    ModelConfig,
    PoseModel,
    SequenceModel,
    VideoClassificationModel,
    get_model,
    list_models,
    register_model,
)
from .classifiers import (
    ClassificationMetrics,
    ClassifierConfig,
    FeatureClassifier,
    UPDRSClassifier,
)

__all__ = [
    # Base
    "BaseModel",
    "FeatureExtractor",
    "InferenceResult",
    "ModelConfig",
    "PoseModel",
    "SequenceModel",
    "VideoClassificationModel",
    "get_model",
    "list_models",
    "register_model",
    # Classifiers
    "ClassificationMetrics",
    "ClassifierConfig",
    "FeatureClassifier",
    "UPDRSClassifier",
]
