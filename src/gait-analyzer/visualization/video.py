"""
Video Visualization and Annotation
==================================

Annotate videos with pose, face landmarks, and analysis results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np


@dataclass
class AnnotationStyle:
    """Video annotation styling."""

    # Pose
    pose_color: tuple[int, int, int] = (0, 255, 0)  # Green
    pose_thickness: int = 2
    joint_radius: int = 4
    joint_color: tuple[int, int, int] = (255, 0, 0)  # Blue

    # Face
    face_color: tuple[int, int, int] = (255, 255, 0)  # Cyan
    face_thickness: int = 1
    landmark_radius: int = 1

    # Bounding box
    bbox_color: tuple[int, int, int] = (0, 255, 255)  # Yellow
    bbox_thickness: int = 2

    # Text
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.6
    font_color: tuple[int, int, int] = (255, 255, 255)  # White
    font_thickness: int = 1
    text_bg_color: tuple[int, int, int] = (0, 0, 0)  # Black


# Pose connections (MediaPipe style)
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
    (9, 10),  # Mouth
    # Body
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 29), (29, 31),  # Left foot
    (28, 30), (30, 32),  # Right foot
]

# Limb colors for visualization
LIMB_COLORS = {
    "left_arm": (255, 128, 0),    # Orange
    "right_arm": (0, 128, 255),   # Blue
    "left_leg": (255, 0, 128),    # Pink
    "right_leg": (128, 0, 255),   # Purple
    "torso": (0, 255, 128),       # Green
    "face": (255, 255, 0),        # Yellow
}


def draw_pose(
    frame: np.ndarray,
    keypoints: np.ndarray,
    style: AnnotationStyle | None = None,
    connections: list[tuple[int, int]] | None = None,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """
    Draw pose skeleton on frame.

    Args:
        frame: BGR image
        keypoints: (N, 3) array with x, y, confidence
        style: Annotation style
        connections: List of joint connections
        confidence_threshold: Min confidence to draw

    Returns:
        Annotated frame
    """
    style = style or AnnotationStyle()
    connections = connections or POSE_CONNECTIONS

    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Draw connections
    for i, j in connections:
        if i >= len(keypoints) or j >= len(keypoints):
            continue

        x1, y1, c1 = keypoints[i]
        x2, y2, c2 = keypoints[j]

        if c1 < confidence_threshold or c2 < confidence_threshold:
            continue

        # Convert to pixel coordinates if normalized
        if x1 <= 1 and y1 <= 1:
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)
        else:
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

        cv2.line(annotated, (x1, y1), (x2, y2), style.pose_color, style.pose_thickness)

    # Draw joints
    for i, (x, y, c) in enumerate(keypoints):
        if c < confidence_threshold:
            continue

        if x <= 1 and y <= 1:
            x, y = int(x * w), int(y * h)
        else:
            x, y = int(x), int(y)

        cv2.circle(annotated, (x, y), style.joint_radius, style.joint_color, -1)

    return annotated


def draw_face_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    style: AnnotationStyle | None = None,
    draw_mesh: bool = False,
) -> np.ndarray:
    """
    Draw face landmarks on frame.

    Args:
        frame: BGR image
        landmarks: (N, 3) normalized landmarks
        style: Annotation style
        draw_mesh: Whether to draw full mesh

    Returns:
        Annotated frame
    """
    style = style or AnnotationStyle()

    h, w = frame.shape[:2]
    annotated = frame.copy()

    for x, y, _ in landmarks:
        px, py = int(x * w), int(y * h)
        cv2.circle(annotated, (px, py), style.landmark_radius, style.face_color, -1)

    return annotated


def draw_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str = "",
    confidence: float | None = None,
    style: AnnotationStyle | None = None,
) -> np.ndarray:
    """
    Draw bounding box on frame.

    Args:
        frame: BGR image
        bbox: (x, y, w, h) bounding box
        label: Optional label text
        confidence: Optional confidence score
        style: Annotation style

    Returns:
        Annotated frame
    """
    style = style or AnnotationStyle()
    annotated = frame.copy()

    x, y, bw, bh = bbox
    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), style.bbox_color, style.bbox_thickness)

    if label or confidence is not None:
        text = label
        if confidence is not None:
            text = f"{label} {confidence:.2f}" if label else f"{confidence:.2f}"

        (tw, th), _ = cv2.getTextSize(text, style.font, style.font_scale, style.font_thickness)
        cv2.rectangle(annotated, (x, y - th - 4), (x + tw + 4, y), style.text_bg_color, -1)
        cv2.putText(annotated, text, (x + 2, y - 2), style.font, style.font_scale,
                    style.font_color, style.font_thickness)

    return annotated


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int] = (10, 30),
    style: AnnotationStyle | None = None,
) -> np.ndarray:
    """Draw text with background on frame."""
    style = style or AnnotationStyle()
    annotated = frame.copy()

    (tw, th), _ = cv2.getTextSize(text, style.font, style.font_scale, style.font_thickness)
    x, y = position
    cv2.rectangle(annotated, (x - 2, y - th - 2), (x + tw + 2, y + 2), style.text_bg_color, -1)
    cv2.putText(annotated, text, (x, y), style.font, style.font_scale,
                style.font_color, style.font_thickness)

    return annotated


def draw_metrics(
    frame: np.ndarray,
    metrics: dict[str, float],
    position: tuple[int, int] = (10, 30),
    style: AnnotationStyle | None = None,
) -> np.ndarray:
    """Draw metrics overlay on frame."""
    style = style or AnnotationStyle()
    annotated = frame.copy()

    y = position[1]
    for name, value in metrics.items():
        text = f"{name}: {value:.3f}"
        annotated = draw_text(annotated, text, (position[0], y), style)
        y += 25

    return annotated


class VideoAnnotator:
    """Annotate videos with pose, face, and analysis results."""

    def __init__(self, style: AnnotationStyle | None = None):
        self.style = style or AnnotationStyle()

    def annotate_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        pose_data: list[np.ndarray] | None = None,
        face_data: list[np.ndarray] | None = None,
        metrics_data: list[dict] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> bool:
        """
        Annotate a video with pose/face/metrics.

        Args:
            input_path: Input video path
            output_path: Output video path
            pose_data: List of keypoints per frame
            face_data: List of face landmarks per frame
            metrics_data: List of metrics per frame
            progress_callback: Called with (current_frame, total_frames)

        Returns:
            True if successful
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Add pose
            if pose_data and frame_idx < len(pose_data):
                keypoints = pose_data[frame_idx]
                if keypoints is not None and len(keypoints) > 0:
                    frame = draw_pose(frame, keypoints, self.style)

            # Add face
            if face_data and frame_idx < len(face_data):
                landmarks = face_data[frame_idx]
                if landmarks is not None and len(landmarks) > 0:
                    frame = draw_face_landmarks(frame, landmarks, self.style)

            # Add metrics
            if metrics_data and frame_idx < len(metrics_data):
                metrics = metrics_data[frame_idx]
                if metrics:
                    frame = draw_metrics(frame, metrics, style=self.style)

            out.write(frame)
            frame_idx += 1

            if progress_callback:
                progress_callback(frame_idx, total_frames)

        cap.release()
        out.release()
        return True

    def extract_frames_with_annotations(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        pose_data: list[np.ndarray] | None = None,
        face_data: list[np.ndarray] | None = None,
        frame_indices: list[int] | None = None,
        sample_rate: int = 30,
    ) -> list[Path]:
        """
        Extract annotated frames from video.

        Args:
            video_path: Input video path
            output_dir: Output directory for frames
            pose_data: Pose keypoints
            face_data: Face landmarks
            frame_indices: Specific frames to extract (None = sample)
            sample_rate: Extract every Nth frame if indices not specified

        Returns:
            List of saved frame paths
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_indices is None:
            frame_indices = list(range(0, total_frames, sample_rate))

        saved_paths = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Add annotations
            if pose_data and idx < len(pose_data) and pose_data[idx] is not None:
                frame = draw_pose(frame, pose_data[idx], self.style)

            if face_data and idx < len(face_data) and face_data[idx] is not None:
                frame = draw_face_landmarks(frame, face_data[idx], self.style)

            # Save frame
            frame_path = output_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            saved_paths.append(frame_path)

        cap.release()
        return saved_paths


def create_comparison_video(
    video_paths: list[str | Path],
    output_path: str | Path,
    labels: list[str] | None = None,
    layout: str = "horizontal",  # horizontal, vertical, grid
) -> bool:
    """
    Create side-by-side comparison video.

    Args:
        video_paths: List of input video paths
        output_path: Output video path
        labels: Labels for each video
        layout: Layout type

    Returns:
        True if successful
    """
    caps = [cv2.VideoCapture(str(p)) for p in video_paths]

    if not all(cap.isOpened() for cap in caps):
        for cap in caps:
            cap.release()
        return False

    fps = caps[0].get(cv2.CAP_PROP_FPS)
    widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
    heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]

    n = len(caps)

    if layout == "horizontal":
        out_w = sum(widths)
        out_h = max(heights)
    elif layout == "vertical":
        out_w = max(widths)
        out_h = sum(heights)
    else:  # grid
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        out_w = cols * max(widths)
        out_h = rows * max(heights)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if len(frames) != n:
            break

        # Add labels
        if labels:
            for i, (frame, label) in enumerate(zip(frames, labels)):
                frames[i] = draw_text(frame, label, (10, 30))

        # Combine frames
        if layout == "horizontal":
            combined = np.hstack(frames)
        elif layout == "vertical":
            combined = np.vstack(frames)
        else:  # grid
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            # Pad if necessary
            while len(frames) < rows * cols:
                frames.append(np.zeros_like(frames[0]))
            row_imgs = []
            for r in range(rows):
                row_frames = frames[r * cols:(r + 1) * cols]
                row_imgs.append(np.hstack(row_frames))
            combined = np.vstack(row_imgs)

        out.write(combined)

    for cap in caps:
        cap.release()
    out.release()

    return True
