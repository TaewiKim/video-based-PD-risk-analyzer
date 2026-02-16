from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np

from research_automation.collection.quality import VideoQualityChecker
from research_automation.pipeline.extractors.pose import PoseExtractor, PoseSequence


LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


@dataclass
class SkeletonPipelineConfig:
    sample_rate: int = 1
    min_detection_rate: float = 0.3
    min_valid_frames: int = 45
    min_mean_visibility: float = 0.35
    target_fps: float = 30.0
    smooth_window: int = 5


def _safe_keypoints_array(sequence: PoseSequence) -> np.ndarray:
    rows = [f.keypoints for f in sequence.frames if f.keypoints is not None]
    if not rows:
        n_keypoints = int(sequence.n_keypoints) if sequence.n_keypoints > 0 else 0
        if n_keypoints > 0:
            return np.empty((0, n_keypoints, 3), dtype=np.float32)
        return np.empty((0, 0, 0), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def _resample_series(arr: np.ndarray, src_fps: float, target_fps: float) -> np.ndarray:
    if arr.ndim < 1 or len(arr) < 2:
        return arr
    if src_fps <= 0 or target_fps <= 0:
        return arr
    if abs(src_fps - target_fps) < 1e-6:
        return arr

    n_src = arr.shape[0]
    duration = max((n_src - 1) / src_fps, 0.0)
    n_tgt = max(2, int(round(duration * target_fps)) + 1)
    t_src = np.linspace(0.0, duration, n_src)
    t_tgt = np.linspace(0.0, duration, n_tgt)

    flat = arr.reshape(n_src, -1)
    out = np.empty((n_tgt, flat.shape[1]), dtype=float)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(t_tgt, t_src, flat[:, i])
    return out.reshape((n_tgt,) + arr.shape[1:])


def _smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty_like(flat, dtype=float)
    for i in range(flat.shape[1]):
        out[:, i] = np.convolve(flat[:, i], kernel, mode="same")
    return out.reshape(arr.shape)


def _normalize_skeleton(
    keypoints: np.ndarray, fps: float, cfg: SkeletonPipelineConfig
) -> tuple[np.ndarray, float]:
    if keypoints.ndim != 3 or keypoints.shape[1] < 25:
        return keypoints, fps

    k = keypoints.astype(float, copy=True)
    xy = k[:, :, :2]
    vis = k[:, :, 2:3]

    pelvis = (xy[:, LEFT_HIP, :] + xy[:, RIGHT_HIP, :]) / 2.0
    xy = xy - pelvis[:, None, :]

    shoulder_dist = np.linalg.norm(xy[:, LEFT_SHOULDER, :] - xy[:, RIGHT_SHOULDER, :], axis=1)
    scale = (
        float(np.median(shoulder_dist[shoulder_dist > 1e-6]))
        if np.any(shoulder_dist > 1e-6)
        else 1.0
    )
    if scale <= 1e-6:
        scale = 1.0
    xy = xy / scale

    out = np.concatenate([xy, vis], axis=2)
    out = _resample_series(out, src_fps=fps, target_fps=cfg.target_fps)
    out = _smooth_series(out, window=cfg.smooth_window)
    return out.astype(np.float32), cfg.target_fps


def _build_validation_report(
    sequence: PoseSequence,
    quality_checker: VideoQualityChecker,
    video_path: Path,
    cfg: SkeletonPipelineConfig,
) -> dict[str, Any]:
    quality = quality_checker.check_video(video_path)
    arr = _safe_keypoints_array(sequence)
    mean_visibility = float(arr[:, :, 2].mean()) if len(arr) > 0 else 0.0

    checks = {
        "detection_rate_ok": sequence.detection_rate >= cfg.min_detection_rate,
        "valid_frames_ok": len(arr) >= cfg.min_valid_frames,
        "mean_visibility_ok": mean_visibility >= cfg.min_mean_visibility,
        "quality_ok": bool(quality.is_usable),
    }
    passed = all(checks.values())

    return {
        "passed": passed,
        "checks": checks,
        "n_valid_frames": int(len(arr)),
        "detection_rate": float(sequence.detection_rate),
        "mean_visibility": mean_visibility,
        "quality": {
            "overall_score": float(quality.overall_score),
            "is_usable": bool(quality.is_usable),
            "pose_detection_rate": float(quality.pose_detection_rate),
            "face_detection_rate": float(quality.face_detection_rate),
            "fps": float(quality.fps),
            "resolution": [int(quality.resolution[0]), int(quality.resolution[1])],
            "issues": list(quality.issues),
        },
    }


def preprocess_video_to_skeleton(
    video_path: str | Path,
    output_dir: str | Path,
    label: str | None = None,
    config: SkeletonPipelineConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SkeletonPipelineConfig()
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with PoseExtractor() as extractor:
        sequence = extractor.extract_from_video(video_path, sample_rate=cfg.sample_rate)

    quality_checker = VideoQualityChecker(
        sample_rate=max(1, int(sequence.fps // 2) if sequence.fps > 0 else 15)
    )
    validation = _build_validation_report(sequence, quality_checker, video_path, cfg)

    raw = _safe_keypoints_array(sequence)
    normalized, normalized_fps = (
        _normalize_skeleton(raw, sequence.fps, cfg) if len(raw) > 0 else (raw, sequence.fps)
    )

    stem = video_path.stem
    raw_path = output_dir / f"{stem}_skeleton_raw.npy"
    norm_path = output_dir / f"{stem}_skeleton_norm.npy"
    meta_path = output_dir / f"{stem}_skeleton_meta.json"

    np.save(raw_path, raw)
    np.save(norm_path, normalized)

    metadata = {
        "video_path": str(video_path),
        "label": label,
        "pipeline": "video->skeleton->validation->normalization",
        "config": {
            "sample_rate": cfg.sample_rate,
            "min_detection_rate": cfg.min_detection_rate,
            "min_valid_frames": cfg.min_valid_frames,
            "min_mean_visibility": cfg.min_mean_visibility,
            "target_fps": cfg.target_fps,
            "smooth_window": cfg.smooth_window,
        },
        "input": {
            "fps": float(sequence.fps),
            "n_frames": int(len(sequence.frames)),
            "n_keypoints": int(sequence.n_keypoints),
        },
        "output": {
            "raw_path": str(raw_path),
            "normalized_path": str(norm_path),
            "metadata_path": str(meta_path),
            "normalized_fps": float(normalized_fps),
            "raw_shape": list(raw.shape),
            "normalized_shape": list(normalized.shape),
        },
        "validation": validation,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
