"""Visualization utilities for research outputs."""

from .plots import (
    PlotStyle,
    PUBLICATION_STYLE,
    plot_confusion_matrix,
    plot_experiment_comparison,
    plot_feature_comparison,
    plot_gait_features,
    plot_joint_trajectories,
    plot_roc_curve,
    plot_time_series,
)
from .video import (
    AnnotationStyle,
    VideoAnnotator,
    create_comparison_video,
    draw_bbox,
    draw_face_landmarks,
    draw_metrics,
    draw_pose,
    draw_text,
)

__all__ = [
    # Plots
    "PlotStyle",
    "PUBLICATION_STYLE",
    "plot_confusion_matrix",
    "plot_experiment_comparison",
    "plot_feature_comparison",
    "plot_gait_features",
    "plot_joint_trajectories",
    "plot_roc_curve",
    "plot_time_series",
    # Video
    "AnnotationStyle",
    "VideoAnnotator",
    "create_comparison_video",
    "draw_bbox",
    "draw_face_landmarks",
    "draw_metrics",
    "draw_pose",
    "draw_text",
]
