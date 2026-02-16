"""
Plotting Utilities for Research Visualization
==============================================

Create publication-quality plots for gait analysis, facial features, and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Optional matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    Figure = None


@dataclass
class PlotStyle:
    """Plot styling configuration."""

    figsize: tuple[float, float] = (10, 6)
    dpi: int = 150
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 10
    grid: bool = True
    style: str = "seaborn-v0_8-whitegrid"

    def apply(self):
        """Apply style to matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            return
        try:
            plt.style.use(self.style)
        except OSError:
            plt.style.use("seaborn-v0_8-white")

        plt.rcParams.update({
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "legend.fontsize": self.legend_size,
            "figure.figsize": self.figsize,
            "figure.dpi": self.dpi,
        })


# Default style for publication
PUBLICATION_STYLE = PlotStyle(
    figsize=(8, 6),
    dpi=300,
    font_size=11,
    title_size=12,
    style="seaborn-v0_8-white",
)


def plot_gait_features(
    features: dict[str, float],
    title: str = "Gait Features",
    save_path: str | Path | None = None,
    style: PlotStyle | None = None,
) -> Figure | None:
    """
    Create bar plot of gait features.

    Args:
        features: Dict of feature name -> value
        title: Plot title
        save_path: Optional path to save figure
        style: Plot style configuration

    Returns:
        Matplotlib figure or None if not available
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    style = style or PUBLICATION_STYLE
    style.apply()

    fig, ax = plt.subplots(figsize=style.figsize)

    names = list(features.keys())
    values = list(features.values())

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(names, values, color=colors)

    ax.set_xlabel("Value")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=style.tick_size)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=style.dpi, bbox_inches="tight")

    return fig


def plot_feature_comparison(
    features_list: list[dict[str, float]],
    labels: list[str],
    feature_names: list[str] | None = None,
    title: str = "Feature Comparison",
    save_path: str | Path | None = None,
    style: PlotStyle | None = None,
) -> Figure | None:
    """
    Compare features across multiple subjects/conditions.

    Args:
        features_list: List of feature dicts
        labels: Labels for each feature set
        feature_names: Which features to plot (None = all)
        title: Plot title
        save_path: Optional path to save
        style: Plot style

    Returns:
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    style = style or PUBLICATION_STYLE
    style.apply()

    # Get feature names
    if feature_names is None:
        feature_names = list(features_list[0].keys())

    n_features = len(feature_names)
    n_groups = len(features_list)

    fig, ax = plt.subplots(figsize=(max(10, n_features * 0.8), 6))

    x = np.arange(n_features)
    width = 0.8 / n_groups

    for i, (features, label) in enumerate(zip(features_list, labels)):
        values = [features.get(name, 0) for name in feature_names]
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=label, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=style.dpi, bbox_inches="tight")

    return fig


def plot_time_series(
    data: np.ndarray | list,
    timestamps: np.ndarray | list | None = None,
    labels: list[str] | None = None,
    title: str = "Time Series",
    xlabel: str = "Time (s)",
    ylabel: str = "Value",
    save_path: str | Path | None = None,
    style: PlotStyle | None = None,
) -> Figure | None:
    """
    Plot time series data.

    Args:
        data: (T,) or (T, N) array of values
        timestamps: Optional time values
        labels: Labels for each series
        title: Plot title
        xlabel, ylabel: Axis labels
        save_path: Optional save path
        style: Plot style

    Returns:
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    style = style or PUBLICATION_STYLE
    style.apply()

    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    T, N = data.shape

    if timestamps is None:
        timestamps = np.arange(T)

    fig, ax = plt.subplots(figsize=style.figsize)

    for i in range(N):
        label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
        ax.plot(timestamps, data[:, i], label=label, alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if N > 1:
        ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=style.dpi, bbox_inches="tight")

    return fig


def plot_joint_trajectories(
    keypoints: np.ndarray,
    joint_indices: list[int],
    joint_names: list[str],
    fps: float = 30.0,
    title: str = "Joint Trajectories",
    save_path: str | Path | None = None,
) -> Figure | None:
    """
    Plot joint position trajectories over time.

    Args:
        keypoints: (T, N, 2 or 3) array of keypoints
        joint_indices: Which joints to plot
        joint_names: Names for each joint
        fps: Frames per second
        title: Plot title
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    T = keypoints.shape[0]
    timestamps = np.arange(T) / fps

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for idx, name in zip(joint_indices, joint_names):
        axes[0].plot(timestamps, keypoints[:, idx, 0], label=name, alpha=0.8)
        axes[1].plot(timestamps, keypoints[:, idx, 1], label=name, alpha=0.8)

    axes[0].set_ylabel("X Position")
    axes[0].set_title(f"{title} - Horizontal Movement")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("Y Position")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(f"{title} - Vertical Movement")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    cmap: str = "Blues",
    save_path: str | Path | None = None,
) -> Figure | None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix array
        class_names: Class labels
        title: Plot title
        normalize: Whether to normalize
        cmap: Colormap name
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    title: str = "ROC Curve",
    save_path: str | Path | None = None,
) -> Figure | None:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve
        title: Plot title
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.3, color="darkorange")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_experiment_comparison(
    experiments: list[dict],
    metrics: list[str],
    title: str = "Experiment Comparison",
    save_path: str | Path | None = None,
) -> Figure | None:
    """
    Compare metrics across experiments.

    Args:
        experiments: List of dicts with 'name' and metric values
        metrics: List of metric names to compare
        title: Plot title
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    n_exp = len(experiments)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(max(10, n_exp * 1.5), 6))

    x = np.arange(n_exp)
    width = 0.8 / n_metrics

    for i, metric in enumerate(metrics):
        values = [exp.get(metric, 0) for exp in experiments]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([exp["name"] for exp in experiments], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
