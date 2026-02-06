"""
Base Model Classes
==================

Abstract base classes for video/pose/face analysis models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class ModelConfig:
    """Base model configuration."""

    name: str
    version: str = "1.0"
    device: str = "cpu"  # cpu, cuda, mps
    batch_size: int = 1
    num_workers: int = 0


@dataclass
class InferenceResult:
    """Base inference result."""

    predictions: np.ndarray
    confidence: float
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def load(self, checkpoint_path: str | Path | None = None) -> None:
        """Load model weights."""
        pass

    @abstractmethod
    def predict(self, inputs: Any) -> InferenceResult:
        """Run inference."""
        pass

    def unload(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._is_loaded = False

    def __enter__(self):
        if not self._is_loaded:
            self.load()
        return self

    def __exit__(self, *args):
        self.unload()


class PoseModel(BaseModel):
    """Base class for pose estimation models."""

    @abstractmethod
    def estimate_pose(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate pose from image.

        Args:
            image: BGR image (H, W, 3)

        Returns:
            Tuple of (keypoints, confidence)
            - keypoints: (N, 2 or 3) normalized coordinates
            - confidence: (N,) confidence per keypoint
        """
        pass

    def estimate_poses_batch(
        self,
        images: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Batch pose estimation."""
        return [self.estimate_pose(img) for img in images]


class VideoClassificationModel(BaseModel):
    """Base class for video classification models."""

    @abstractmethod
    def classify(
        self,
        frames: np.ndarray | list[np.ndarray],
    ) -> tuple[int, np.ndarray]:
        """
        Classify video frames.

        Args:
            frames: (T, H, W, 3) or list of frames

        Returns:
            Tuple of (predicted_class, class_probabilities)
        """
        pass


class SequenceModel(BaseModel):
    """Base class for sequence (pose/feature) classification."""

    @abstractmethod
    def classify_sequence(
        self,
        sequence: np.ndarray,
    ) -> tuple[int, np.ndarray]:
        """
        Classify a sequence of features.

        Args:
            sequence: (T, D) feature sequence

        Returns:
            Tuple of (predicted_class, class_probabilities)
        """
        pass


class FeatureExtractor(BaseModel):
    """Base class for feature extraction models."""

    @abstractmethod
    def extract_features(
        self,
        inputs: Any,
    ) -> np.ndarray:
        """
        Extract features from input.

        Args:
            inputs: Model-specific input

        Returns:
            Feature vector or sequence
        """
        pass


# Model Registry
_MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, config: ModelConfig | None = None) -> BaseModel:
    """Get a registered model by name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}")

    model_cls = _MODEL_REGISTRY[name]
    if config is None:
        config = ModelConfig(name=name)

    return model_cls(config)


def list_models() -> list[str]:
    """List all registered models."""
    return list(_MODEL_REGISTRY.keys())
