"""
Pose Augmentation
=================

Data augmentation for pose sequences to improve model robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class AugmentationConfig:
    """Configuration for pose augmentation."""

    # Spatial
    scale_range: tuple[float, float] = (0.9, 1.1)
    rotation_range: tuple[float, float] = (-15, 15)  # degrees
    translation_range: tuple[float, float] = (-0.1, 0.1)
    flip_probability: float = 0.5

    # Temporal
    speed_range: tuple[float, float] = (0.8, 1.2)
    temporal_crop_ratio: tuple[float, float] = (0.8, 1.0)
    frame_drop_probability: float = 0.1

    # Noise
    gaussian_noise_std: float = 0.01
    joint_dropout_probability: float = 0.05

    # Joint-specific
    jitter_std: float = 0.005


class PoseAugmentor:
    """Apply augmentations to pose sequences."""

    def __init__(self, config: AugmentationConfig | None = None):
        self.config = config or AugmentationConfig()

    def augment(
        self,
        keypoints: np.ndarray,
        augmentations: list[str] | None = None,
    ) -> np.ndarray:
        """
        Apply augmentations to keypoint sequence.

        Args:
            keypoints: (T, N, 2 or 3) keypoint array
            augmentations: List of augmentation names to apply
                          None = apply all with configured probabilities

        Returns:
            Augmented keypoints
        """
        if augmentations is None:
            augmentations = [
                "scale", "rotate", "translate", "flip",
                "speed", "noise", "jitter",
            ]

        result = keypoints.copy()

        for aug in augmentations:
            if aug == "scale":
                result = self.scale(result)
            elif aug == "rotate":
                result = self.rotate(result)
            elif aug == "translate":
                result = self.translate(result)
            elif aug == "flip":
                result = self.flip(result)
            elif aug == "speed":
                result = self.temporal_speed(result)
            elif aug == "crop":
                result = self.temporal_crop(result)
            elif aug == "drop":
                result = self.frame_drop(result)
            elif aug == "noise":
                result = self.add_noise(result)
            elif aug == "jitter":
                result = self.joint_jitter(result)
            elif aug == "dropout":
                result = self.joint_dropout(result)

        return result

    def scale(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply random scaling."""
        scale = np.random.uniform(*self.config.scale_range)

        # Scale around center
        center = keypoints[:, :, :2].mean(axis=(0, 1), keepdims=True)
        scaled = keypoints.copy()
        scaled[:, :, :2] = (keypoints[:, :, :2] - center) * scale + center

        return scaled

    def rotate(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply random rotation around center."""
        angle = np.random.uniform(*self.config.rotation_range)
        angle_rad = np.deg2rad(angle)

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a],
        ])

        # Rotate around center
        center = keypoints[:, :, :2].mean(axis=(0, 1), keepdims=True)
        centered = keypoints[:, :, :2] - center

        rotated = keypoints.copy()
        for t in range(len(keypoints)):
            rotated[t, :, :2] = centered[t] @ rotation_matrix.T + center[0, 0]

        return rotated

    def translate(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply random translation."""
        tx = np.random.uniform(*self.config.translation_range)
        ty = np.random.uniform(*self.config.translation_range)

        translated = keypoints.copy()
        translated[:, :, 0] += tx
        translated[:, :, 1] += ty

        return translated

    def flip(self, keypoints: np.ndarray) -> np.ndarray:
        """Horizontally flip with probability."""
        if np.random.random() > self.config.flip_probability:
            return keypoints

        flipped = keypoints.copy()

        # Flip x coordinates
        center_x = keypoints[:, :, 0].mean()
        flipped[:, :, 0] = 2 * center_x - keypoints[:, :, 0]

        # Swap left/right joints (MediaPipe order)
        # This is a simplified swap - adjust based on skeleton
        left_right_pairs = [
            (1, 4), (2, 5), (3, 6),  # Eyes
            (7, 8),  # Ears
            (11, 12), (13, 14), (15, 16),  # Arms
            (17, 18), (19, 20), (21, 22),  # Hands
            (23, 24), (25, 26), (27, 28),  # Legs
            (29, 30), (31, 32),  # Feet
        ]

        n_joints = keypoints.shape[1]
        for left, right in left_right_pairs:
            if left < n_joints and right < n_joints:
                flipped[:, [left, right], :] = flipped[:, [right, left], :]

        return flipped

    def temporal_speed(self, keypoints: np.ndarray) -> np.ndarray:
        """Change temporal speed (resample)."""
        speed = np.random.uniform(*self.config.speed_range)
        T = len(keypoints)
        new_T = int(T / speed)

        if new_T < 2:
            return keypoints

        # Interpolate
        old_indices = np.linspace(0, T - 1, T)
        new_indices = np.linspace(0, T - 1, new_T)

        resampled = np.zeros((new_T, *keypoints.shape[1:]))
        for j in range(keypoints.shape[1]):
            for d in range(keypoints.shape[2]):
                resampled[:, j, d] = np.interp(new_indices, old_indices, keypoints[:, j, d])

        return resampled

    def temporal_crop(self, keypoints: np.ndarray) -> np.ndarray:
        """Random temporal crop."""
        ratio = np.random.uniform(*self.config.temporal_crop_ratio)
        T = len(keypoints)
        new_T = int(T * ratio)

        if new_T < 2:
            return keypoints

        start = np.random.randint(0, T - new_T + 1)
        return keypoints[start:start + new_T]

    def frame_drop(self, keypoints: np.ndarray) -> np.ndarray:
        """Randomly drop frames."""
        T = len(keypoints)
        keep_mask = np.random.random(T) > self.config.frame_drop_probability

        # Ensure at least 2 frames
        if keep_mask.sum() < 2:
            keep_mask[:2] = True

        return keypoints[keep_mask]

    def add_noise(self, keypoints: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to coordinates."""
        noise = np.random.normal(
            0,
            self.config.gaussian_noise_std,
            keypoints[:, :, :2].shape,
        )

        noisy = keypoints.copy()
        noisy[:, :, :2] += noise

        return noisy

    def joint_jitter(self, keypoints: np.ndarray) -> np.ndarray:
        """Add per-joint random jitter."""
        jitter = np.random.normal(
            0,
            self.config.jitter_std,
            keypoints[:, :, :2].shape,
        )

        jittered = keypoints.copy()
        jittered[:, :, :2] += jitter

        return jittered

    def joint_dropout(self, keypoints: np.ndarray) -> np.ndarray:
        """Randomly zero out joints."""
        T, N = keypoints.shape[:2]
        dropout_mask = np.random.random((T, N)) < self.config.joint_dropout_probability

        dropped = keypoints.copy()
        dropped[dropout_mask, :2] = 0

        # Also zero confidence if present
        if keypoints.shape[2] > 2:
            dropped[dropout_mask, 2] = 0

        return dropped


def augment_sequence(
    keypoints: np.ndarray,
    config: AugmentationConfig | None = None,
    n_augmented: int = 1,
) -> list[np.ndarray]:
    """
    Generate augmented versions of a pose sequence.

    Args:
        keypoints: (T, N, 2 or 3) keypoint array
        config: Augmentation configuration
        n_augmented: Number of augmented versions to generate

    Returns:
        List of augmented sequences (including original)
    """
    augmentor = PoseAugmentor(config)
    results = [keypoints]  # Include original

    for _ in range(n_augmented):
        results.append(augmentor.augment(keypoints))

    return results


class SequenceDataset:
    """Dataset with on-the-fly augmentation."""

    def __init__(
        self,
        sequences: list[np.ndarray],
        labels: np.ndarray,
        augment: bool = True,
        config: AugmentationConfig | None = None,
    ):
        """
        Initialize dataset.

        Args:
            sequences: List of (T, N, D) keypoint sequences
            labels: (N,) array of labels
            augment: Whether to apply augmentation
            config: Augmentation config
        """
        self.sequences = sequences
        self.labels = labels
        self.augment = augment
        self.augmentor = PoseAugmentor(config) if augment else None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        seq = self.sequences[idx]
        label = self.labels[idx]

        if self.augment and self.augmentor is not None:
            seq = self.augmentor.augment(seq)

        return seq, label

    def get_batch(
        self,
        indices: list[int],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Get a batch of samples."""
        sequences = []
        labels = []

        for idx in indices:
            seq, label = self[idx]
            sequences.append(seq)
            labels.append(label)

        return sequences, np.array(labels)
