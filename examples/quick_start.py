#!/usr/bin/env python3
"""
Quick Start Examples
====================

Simple examples to get started with research-automation.

Usage:
    uv run python examples/quick_start.py
"""

# ============================================================================
# 1. Literature Search
# ============================================================================

def example_literature_search():
    """Search for papers on PubMed and arXiv."""
    from research_automation.literature.search import LiteratureSearch

    with LiteratureSearch() as searcher:
        # Search PubMed
        papers = searcher.search_pubmed("Parkinson disease gait", limit=5)

        for paper in papers:
            print(f"Title: {paper.title}")
            print(f"Authors: {', '.join(paper.authors[:3])}")
            print(f"DOI: {paper.doi}")
            print()


# ============================================================================
# 2. Gait Feature Extraction
# ============================================================================

def example_gait_features():
    """Extract gait features from pose data."""
    import numpy as np
    from research_automation.pipeline.extractors import (
        PoseFrame,
        PoseSequence,
        extract_gait_features,
    )

    # Create pose frames (in real use, these come from video)
    frames = []
    for i in range(60):
        keypoints = np.zeros((33, 3), dtype=np.float32)
        # Set joint positions (normalized 0-1)
        keypoints[23] = [0.45, 0.5, 0.9]  # left hip
        keypoints[24] = [0.55, 0.5, 0.9]  # right hip
        keypoints[27] = [0.45, 0.9, 0.9]  # left ankle
        keypoints[28] = [0.55, 0.9, 0.9]  # right ankle
        keypoints[25] = [0.45, 0.7, 0.9]  # left knee
        keypoints[26] = [0.55, 0.7, 0.9]  # right knee
        keypoints[11] = [0.45, 0.3, 0.9]  # left shoulder
        keypoints[12] = [0.55, 0.3, 0.9]  # right shoulder

        frames.append(PoseFrame(
            frame_idx=i,
            timestamp=i / 30.0,
            keypoints=keypoints,
        ))

    sequence = PoseSequence(
        frames=frames,
        fps=30.0,
        video_width=1920,
        video_height=1080,
        n_keypoints=33,
    )

    # Extract gait features
    gait = extract_gait_features(sequence)

    print(f"Walking speed: {gait.walking_speed:.4f}")
    print(f"Hip asymmetry: {gait.hip_asymmetry:.2f}°")
    print(f"Trunk sway: {gait.trunk_sway:.4f}")

    # Get feature vector for ML
    feature_vector = gait.to_array()
    print(f"Feature vector: {feature_vector.shape}")


# ============================================================================
# 3. Experiment Tracking
# ============================================================================

def example_experiment_tracking():
    """Track ML experiments with MLflow."""
    import tempfile
    from pathlib import Path
    from research_automation.experiment import ExperimentTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(
            experiment_name="my-experiment",
            tracking_uri=str(Path(tmpdir) / "mlruns"),
        )

        with tracker.start_run("baseline-v1") as run:
            # Log parameters
            tracker.log_params({
                "model": "RandomForest",
                "n_estimators": 100,
            })

            # Log metrics
            tracker.log_metrics({
                "accuracy": 0.85,
                "f1": 0.82,
            })

            print(f"Run ID: {run.info.run_id}")

        # Find best run
        best = tracker.get_best_run("accuracy")
        print(f"Best accuracy: {best.data.metrics['accuracy']}")


# ============================================================================
# 4. Report Generation
# ============================================================================

def example_report_generation():
    """Generate experiment reports."""
    from research_automation.report import generate_experiment_report

    report = generate_experiment_report(
        name="Gait Classification",
        description="Random Forest baseline for PD severity",
        dataset="CARE-PD",
        methods="5-fold cross-validation",
        metrics={"accuracy": 0.78, "f1": 0.76},
    )

    print(report)


# ============================================================================
# 5. Video from Real File (if you have one)
# ============================================================================

def example_real_video(video_path: str):
    """Extract features from a real video file."""
    from pathlib import Path
    from research_automation.pipeline.extractors import (
        PoseExtractor,
        extract_gait_features,
    )

    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        return

    # Extract pose
    with PoseExtractor() as extractor:
        sequence = extractor.extract_from_video(
            video_path,
            sample_rate=2,  # Process every 2nd frame
        )

    print(f"Extracted {len(sequence.frames)} frames")
    print(f"Detection rate: {sequence.detection_rate * 100:.0f}%")

    # Extract gait features
    gait = extract_gait_features(sequence)
    print(f"Walking speed: {gait.walking_speed:.4f}")


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("1. Literature Search")
    print("=" * 50)
    example_literature_search()

    print("\n" + "=" * 50)
    print("2. Gait Feature Extraction")
    print("=" * 50)
    example_gait_features()

    print("\n" + "=" * 50)
    print("3. Experiment Tracking")
    print("=" * 50)
    example_experiment_tracking()

    print("\n" + "=" * 50)
    print("4. Report Generation")
    print("=" * 50)
    example_report_generation()

    # Uncomment to test with real video:
    # example_real_video("/path/to/your/video.mp4")
