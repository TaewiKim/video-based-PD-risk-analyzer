"""
Video Augmentation
==================

Data augmentation for video frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import cv2
import numpy as np


@dataclass
class VideoAugmentationConfig:
    """Configuration for video augmentation."""

    # Spatial
    scale_range: tuple[float, float] = (0.9, 1.1)
    rotation_range: tuple[float, float] = (-10, 10)
    crop_ratio: tuple[float, float] = (0.8, 1.0)
    flip_probability: float = 0.5

    # Color
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.8, 1.2)
    hue_range: tuple[float, float] = (-0.1, 0.1)

    # Temporal
    speed_range: tuple[float, float] = (0.8, 1.2)
    frame_drop_probability: float = 0.05

    # Noise
    gaussian_noise_std: float = 10.0  # pixel value std
    blur_probability: float = 0.1
    blur_kernel_range: tuple[int, int] = (3, 7)


class VideoAugmentor:
    """Apply augmentations to video frames."""

    def __init__(self, config: VideoAugmentationConfig | None = None):
        self.config = config or VideoAugmentationConfig()

    def augment(
        self,
        frames: np.ndarray | list[np.ndarray],
        augmentations: list[str] | None = None,
    ) -> np.ndarray:
        """
        Apply augmentations to video frames.

        Args:
            frames: (T, H, W, 3) or list of frames
            augmentations: List of augmentation names

        Returns:
            Augmented frames
        """
        if isinstance(frames, list):
            frames = np.array(frames)

        if augmentations is None:
            augmentations = [
                "scale", "rotate", "crop", "flip",
                "brightness", "contrast", "noise",
            ]

        result = frames.copy()

        # Determine spatial augmentation parameters (consistent across frames)
        spatial_params = self._sample_spatial_params()

        for aug in augmentations:
            if aug == "scale":
                result = self._scale(result, spatial_params["scale"])
            elif aug == "rotate":
                result = self._rotate(result, spatial_params["rotation"])
            elif aug == "crop":
                result = self._random_crop(result, spatial_params["crop"])
            elif aug == "flip":
                result = self._flip(result, spatial_params["flip"])
            elif aug == "brightness":
                result = self._adjust_brightness(result)
            elif aug == "contrast":
                result = self._adjust_contrast(result)
            elif aug == "saturation":
                result = self._adjust_saturation(result)
            elif aug == "noise":
                result = self._add_noise(result)
            elif aug == "blur":
                result = self._random_blur(result)
            elif aug == "speed":
                result = self._temporal_speed(result)
            elif aug == "drop":
                result = self._frame_drop(result)

        return result

    def _sample_spatial_params(self) -> dict:
        """Sample consistent spatial augmentation parameters."""
        return {
            "scale": np.random.uniform(*self.config.scale_range),
            "rotation": np.random.uniform(*self.config.rotation_range),
            "crop": np.random.uniform(*self.config.crop_ratio),
            "flip": np.random.random() < self.config.flip_probability,
        }

    def _scale(self, frames: np.ndarray, scale: float) -> np.ndarray:
        """Scale frames."""
        T, H, W = frames.shape[:3]
        new_H, new_W = int(H * scale), int(W * scale)

        scaled = np.zeros((T, new_H, new_W, 3), dtype=frames.dtype)
        for t in range(T):
            scaled[t] = cv2.resize(frames[t], (new_W, new_H))

        # Pad or crop to original size
        if new_H > H:
            start_h = (new_H - H) // 2
            scaled = scaled[:, start_h:start_h + H, :, :]
        elif new_H < H:
            pad_h = (H - new_H) // 2
            padded = np.zeros((T, H, new_W, 3), dtype=frames.dtype)
            padded[:, pad_h:pad_h + new_H, :, :] = scaled
            scaled = padded

        if scaled.shape[2] > W:
            start_w = (scaled.shape[2] - W) // 2
            scaled = scaled[:, :, start_w:start_w + W, :]
        elif scaled.shape[2] < W:
            pad_w = (W - scaled.shape[2]) // 2
            padded = np.zeros((T, H, W, 3), dtype=frames.dtype)
            padded[:, :, pad_w:pad_w + scaled.shape[2], :] = scaled
            scaled = padded

        return scaled

    def _rotate(self, frames: np.ndarray, angle: float) -> np.ndarray:
        """Rotate frames."""
        T, H, W = frames.shape[:3]
        center = (W // 2, H // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = np.zeros_like(frames)
        for t in range(T):
            rotated[t] = cv2.warpAffine(frames[t], rotation_matrix, (W, H))

        return rotated

    def _random_crop(self, frames: np.ndarray, crop_ratio: float) -> np.ndarray:
        """Random crop and resize."""
        T, H, W = frames.shape[:3]
        new_H, new_W = int(H * crop_ratio), int(W * crop_ratio)

        # Random crop position
        start_h = np.random.randint(0, H - new_H + 1)
        start_w = np.random.randint(0, W - new_W + 1)

        cropped = frames[:, start_h:start_h + new_H, start_w:start_w + new_W, :]

        # Resize back to original
        resized = np.zeros_like(frames)
        for t in range(T):
            resized[t] = cv2.resize(cropped[t], (W, H))

        return resized

    def _flip(self, frames: np.ndarray, do_flip: bool) -> np.ndarray:
        """Horizontal flip."""
        if not do_flip:
            return frames
        return np.flip(frames, axis=2).copy()

    def _adjust_brightness(self, frames: np.ndarray) -> np.ndarray:
        """Adjust brightness."""
        factor = np.random.uniform(*self.config.brightness_range)
        adjusted = frames.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _adjust_contrast(self, frames: np.ndarray) -> np.ndarray:
        """Adjust contrast."""
        factor = np.random.uniform(*self.config.contrast_range)
        mean = frames.mean()
        adjusted = (frames.astype(np.float32) - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _adjust_saturation(self, frames: np.ndarray) -> np.ndarray:
        """Adjust saturation."""
        factor = np.random.uniform(*self.config.saturation_range)

        adjusted = np.zeros_like(frames)
        for t in range(len(frames)):
            hsv = cv2.cvtColor(frames[t], cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= factor
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            adjusted[t] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return adjusted

    def _add_noise(self, frames: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = np.random.normal(0, self.config.gaussian_noise_std, frames.shape)
        noisy = frames.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _random_blur(self, frames: np.ndarray) -> np.ndarray:
        """Apply random Gaussian blur."""
        if np.random.random() > self.config.blur_probability:
            return frames

        kernel_size = np.random.randint(*self.config.blur_kernel_range)
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = np.zeros_like(frames)
        for t in range(len(frames)):
            blurred[t] = cv2.GaussianBlur(frames[t], (kernel_size, kernel_size), 0)

        return blurred

    def _temporal_speed(self, frames: np.ndarray) -> np.ndarray:
        """Change temporal speed."""
        speed = np.random.uniform(*self.config.speed_range)
        T = len(frames)
        new_T = int(T / speed)

        if new_T < 2:
            return frames

        # Resample frames
        indices = np.linspace(0, T - 1, new_T).astype(int)
        return frames[indices]

    def _frame_drop(self, frames: np.ndarray) -> np.ndarray:
        """Randomly drop frames."""
        T = len(frames)
        keep_mask = np.random.random(T) > self.config.frame_drop_probability

        # Ensure at least 2 frames
        if keep_mask.sum() < 2:
            keep_mask[:2] = True

        return frames[keep_mask]


def augment_video(
    frames: np.ndarray,
    config: VideoAugmentationConfig | None = None,
    n_augmented: int = 1,
) -> list[np.ndarray]:
    """
    Generate augmented versions of video.

    Args:
        frames: (T, H, W, 3) video frames
        config: Augmentation configuration
        n_augmented: Number of augmented versions

    Returns:
        List of augmented videos (including original)
    """
    augmentor = VideoAugmentor(config)
    results = [frames]

    for _ in range(n_augmented):
        results.append(augmentor.augment(frames))

    return results
