#!/usr/bin/env python3
"""
Research Automation Demo
========================

Run this script to test all major features of the research-automation system.

Usage:
    uv run python examples/demo.py

Or run specific demos:
    uv run python examples/demo.py --demo literature
    uv run python examples/demo.py --demo gait
    uv run python examples/demo.py --demo experiment
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


def demo_literature_search():
    """Demonstrate literature search functionality."""
    print("\n" + "=" * 60)
    print("📚 Literature Search Demo")
    print("=" * 60)

    from research_automation.literature.search import LiteratureSearch

    with LiteratureSearch() as searcher:
        # PubMed search
        print("\n[1] Searching PubMed for 'Parkinson gait video'...")
        results = searcher.search_pubmed("Parkinson gait video analysis", limit=5)
        print(f"   Found {len(results)} papers")
        for r in results[:3]:
            print(f"   - {r.title[:60]}...")
            print(f"     Authors: {', '.join(r.authors[:2])} et al.")
            print(f"     DOI: {r.doi or 'N/A'}")

        # arXiv search
        print("\n[2] Searching arXiv for 'pose estimation deep learning'...")
        results = searcher.search_arxiv("pose estimation deep learning", limit=5)
        print(f"   Found {len(results)} papers")
        for r in results[:3]:
            print(f"   - {r.title[:60]}...")
            print(f"     arXiv: {r.arxiv_id}")


def demo_pose_extraction():
    """Demonstrate pose feature extraction with synthetic data."""
    print("\n" + "=" * 60)
    print("🏃 Pose & Gait Feature Extraction Demo")
    print("=" * 60)

    from research_automation.pipeline.extractors import (
        GAIT_JOINTS,
        GaitFeatures,
        PoseFrame,
        PoseSequence,
        extract_gait_features,
    )

    # Create synthetic walking data
    print("\n[1] Creating synthetic walking sequence (60 frames @ 30fps)...")

    frames = []
    fps = 30.0

    for i in range(60):
        t = i / fps
        keypoints = np.zeros((33, 3), dtype=np.float32)

        # Simulate walking motion
        phase = t * 2 * np.pi  # Complete cycle per second

        # Hip positions (slight lateral sway)
        keypoints[23] = [0.45 + 0.02 * np.sin(phase), 0.5, 0.9]  # left hip
        keypoints[24] = [0.55 + 0.02 * np.sin(phase), 0.5, 0.9]  # right hip

        # Ankle positions (alternating forward/backward)
        keypoints[27] = [0.45 + 0.05 * np.sin(phase), 0.9, 0.9]  # left ankle
        keypoints[28] = [0.55 + 0.05 * np.sin(phase + np.pi), 0.9, 0.9]  # right ankle

        # Knee positions
        keypoints[25] = [0.45 + 0.02 * np.sin(phase), 0.7, 0.9]  # left knee
        keypoints[26] = [0.55 + 0.02 * np.sin(phase + np.pi), 0.7, 0.9]  # right knee

        # Shoulder positions (arm swing)
        keypoints[11] = [0.42 + 0.03 * np.sin(phase + np.pi), 0.3, 0.9]  # left shoulder
        keypoints[12] = [0.58 + 0.03 * np.sin(phase), 0.3, 0.9]  # right shoulder

        frames.append(PoseFrame(
            frame_idx=i,
            timestamp=t,
            keypoints=keypoints,
        ))

    sequence = PoseSequence(
        frames=frames,
        fps=fps,
        video_width=1920,
        video_height=1080,
        n_keypoints=33,
    )

    print(f"   Duration: {sequence.duration:.2f}s")
    print(f"   Detection rate: {sequence.detection_rate * 100:.0f}%")

    # Extract gait features
    print("\n[2] Extracting gait features...")
    gait = extract_gait_features(sequence)

    print(f"\n   📊 Gait Analysis Results:")
    print(f"   ├── Walking speed: {gait.walking_speed:.4f} (normalized units)")
    print(f"   ├── Speed variability: {gait.speed_variability:.4f}")
    print(f"   ├── Step length (mean): {gait.step_length_mean:.4f}")
    print(f"   ├── Step width (mean): {gait.step_width_mean:.4f}")
    print(f"   ├── Hip flexion (mean): {gait.hip_flexion_mean:.1f}°")
    print(f"   ├── Knee flexion (mean): {gait.knee_flexion_mean:.1f}°")
    print(f"   ├── Hip asymmetry: {gait.hip_asymmetry:.2f}°")
    print(f"   ├── Knee asymmetry: {gait.knee_asymmetry:.2f}°")
    print(f"   ├── Trunk sway: {gait.trunk_sway:.4f}")
    print(f"   └── Vertical oscillation: {gait.vertical_oscillation:.4f}")

    # Feature vector
    feature_vector = gait.to_array()
    print(f"\n   Feature vector shape: {feature_vector.shape}")


def demo_face_extraction():
    """Demonstrate facial feature extraction with synthetic data."""
    print("\n" + "=" * 60)
    print("😀 Facial Feature Extraction Demo")
    print("=" * 60)

    from research_automation.pipeline.extractors import (
        AU_DEFINITIONS,
        FACE_LANDMARKS,
        FaceFrame,
        FaceSequence,
        FacialFeatures,
        extract_facial_features,
    )

    # Create synthetic face data
    print("\n[1] Creating synthetic face sequence (90 frames @ 30fps)...")

    frames = []
    fps = 30.0

    # Base neutral face landmarks (normalized 0-1)
    np.random.seed(42)
    base_landmarks = np.random.rand(468, 3).astype(np.float32) * 0.3 + 0.35

    for i in range(90):
        t = i / fps
        landmarks = base_landmarks.copy()

        # Simulate expressions (subtle movements)
        phase = t * np.pi  # Slow variation

        # Eye blink (every ~3 seconds)
        if 25 <= i % 90 <= 27:
            # Close eyes briefly
            landmarks[FACE_LANDMARKS["left_eye_upper"], 1] += 0.02
            landmarks[FACE_LANDMARKS["left_eye_lower"], 1] -= 0.02
            landmarks[FACE_LANDMARKS["right_eye_upper"], 1] += 0.02
            landmarks[FACE_LANDMARKS["right_eye_lower"], 1] -= 0.02

        # Mouth movement (talking simulation)
        landmarks[FACE_LANDMARKS["mouth_upper"], 1] += 0.01 * np.sin(phase * 5)
        landmarks[FACE_LANDMARKS["mouth_lower"], 1] -= 0.01 * np.sin(phase * 5)

        # Eyebrow raise
        landmarks[FACE_LANDMARKS["left_eyebrow_inner"], 1] -= 0.005 * np.sin(phase)
        landmarks[FACE_LANDMARKS["right_eyebrow_inner"], 1] -= 0.005 * np.sin(phase)

        frames.append(FaceFrame(
            frame_idx=i,
            timestamp=t,
            landmarks=landmarks,
            detected=True,
        ))

    sequence = FaceSequence(
        frames=frames,
        fps=fps,
        video_width=1920,
        video_height=1080,
        n_landmarks=468,
    )

    print(f"   Duration: {sequence.duration:.2f}s")
    print(f"   Detection rate: {sequence.detection_rate * 100:.0f}%")

    # Extract facial features
    print("\n[2] Extracting facial features...")
    features = extract_facial_features(sequence)

    print(f"\n   📊 Facial Analysis Results:")
    print(f"   ├── Eye blink rate: {features.eye_blink_rate:.1f}/min")
    print(f"   ├── Mouth movement: {features.mouth_movement:.4f}")
    print(f"   ├── Eyebrow movement: {features.eyebrow_movement:.4f}")
    print(f"   ├── Eye asymmetry: {features.eye_asymmetry:.4f}")
    print(f"   ├── Mouth asymmetry: {features.mouth_asymmetry:.4f}")
    print(f"   ├── Overall asymmetry: {features.overall_asymmetry:.4f}")
    print(f"   ├── Expression variability: {features.expression_variability:.4f}")
    print(f"   └── Expression range: {features.expression_range:.4f}")

    print(f"\n   Action Unit Intensities:")
    for au, intensity in sorted(features.au_intensities.items()):
        print(f"   ├── {au}: {intensity:.4f}")


def demo_experiment_tracking():
    """Demonstrate experiment tracking with MLflow."""
    print("\n" + "=" * 60)
    print("📈 Experiment Tracking Demo")
    print("=" * 60)

    from research_automation.experiment import (
        ExperimentResult,
        ExperimentTracker,
        run_experiment,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = str(Path(tmpdir) / "mlruns")

        # Create tracker
        print("\n[1] Creating experiment tracker...")
        tracker = ExperimentTracker(
            experiment_name="gait-classification",
            tracking_uri=tracking_uri,
            description="Demo experiment for gait-based PD classification",
        )
        print(f"   Experiment ID: {tracker.experiment_id}")

        # Run multiple experiments
        print("\n[2] Running experiments with different hyperparameters...")

        configs = [
            {"model": "RandomForest", "n_estimators": 50, "max_depth": 5},
            {"model": "RandomForest", "n_estimators": 100, "max_depth": 10},
            {"model": "RandomForest", "n_estimators": 200, "max_depth": 15},
        ]

        for i, params in enumerate(configs):
            with tracker.start_run(f"run-{i+1}") as run:
                tracker.log_params(params)

                # Simulate training metrics
                accuracy = 0.75 + 0.05 * i + np.random.rand() * 0.05
                f1 = accuracy - 0.02
                precision = accuracy - 0.01
                recall = accuracy - 0.03

                tracker.log_metrics({
                    "accuracy": accuracy,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                })

                print(f"   Run {i+1}: accuracy={accuracy:.3f}, f1={f1:.3f}")

        # Find best run
        print("\n[3] Finding best run...")
        best_run = tracker.get_best_run("accuracy", maximize=True)
        if best_run:
            result = ExperimentResult.from_run(best_run, "gait-classification")
            print(f"   Best run: {result.run_name}")
            print(f"   Best accuracy: {result.metrics['accuracy']:.3f}")
            print(f"   Parameters: {result.params}")

        # Compare runs
        print("\n[4] Comparing all runs...")
        runs = tracker.list_runs()
        print(f"   Total runs: {len(runs)}")

        for run in runs:
            print(f"   - {run.info.run_name}: "
                  f"acc={run.data.metrics.get('accuracy', 0):.3f}, "
                  f"f1={run.data.metrics.get('f1', 0):.3f}")


def demo_report_generation():
    """Demonstrate report generation."""
    print("\n" + "=" * 60)
    print("📄 Report Generation Demo")
    print("=" * 60)

    from research_automation.report import (
        ExperimentReport,
        LiteratureReport,
        ReportGenerator,
        generate_experiment_report,
        generate_literature_report,
    )

    # Create generator
    generator = ReportGenerator()

    # Experiment report
    print("\n[1] Generating experiment report...")

    exp_report = ExperimentReport(
        name="UPDRS Prediction Baseline",
        description="Baseline Random Forest model for UPDRS severity classification using gait features from CARE-PD dataset.",
        date=datetime.now(),
        dataset="CARE-PD (362 subjects, 8,476 walks)",
        methods="Random Forest classifier with 100 trees, 5-fold cross-validation, 41 handcrafted gait features.",
        results={
            "confusion_matrix": [[45, 5, 2, 1], [4, 38, 3, 2], [2, 4, 35, 4], [1, 2, 3, 40]],
            "feature_importance": ["hip_asymmetry", "speed_mean", "trunk_sway", "knee_range"],
        },
        metrics={
            "accuracy": 0.781,
            "f1_macro": 0.762,
            "precision": 0.775,
            "recall": 0.751,
            "binary_accuracy": 0.869,
            "roc_auc": 0.943,
        },
        conclusions="The baseline achieves 78.1% accuracy on 4-class UPDRS prediction and 86.9% on binary mild/severe classification.",
    )

    content = generator.generate_experiment_report(exp_report)
    print("\n" + "-" * 40)
    print(content[:1500])
    print("..." if len(content) > 1500 else "")
    print("-" * 40)

    # Literature report
    print("\n[2] Generating literature report...")

    lit_report = LiteratureReport(
        title="Video-based Parkinson's Disease Assessment",
        date=datetime.now(),
        query="Parkinson gait video deep learning",
        n_papers=3,
        papers=[
            {
                "title": "CARE-PD: A Large-Scale Parkinson's Gait Dataset",
                "authors": ["Smith, J.", "Johnson, A.", "Williams, B."],
                "source": "NeurIPS 2025",
                "doi": "10.1234/neurips.2025.001",
                "abstract": "We present CARE-PD, the largest publicly available dataset for PD gait analysis with 363 subjects from 8 clinical sites...",
            },
            {
                "title": "OpenGait: A Unified Gait Recognition Framework",
                "authors": ["Chen, X.", "Li, Y."],
                "source": "CVPR 2023",
                "doi": "10.1109/cvpr.2023.002",
                "abstract": "OpenGait provides a modular framework for gait recognition with support for various backbone networks...",
            },
            {
                "title": "ViTPose: Vision Transformer for Pose Estimation",
                "authors": ["Wang, M.", "Zhang, K."],
                "source": "NeurIPS 2022",
                "doi": "10.1234/neurips.2022.003",
                "abstract": "We propose ViTPose, a simple vision transformer baseline for human pose estimation achieving 81.1 AP on COCO...",
            },
        ],
        summary="Recent advances in video-based PD assessment leverage deep learning for automated gait and movement analysis.",
        key_findings=[
            "Large-scale datasets enable better generalization",
            "Transformer-based models achieve state-of-the-art",
            "Smartphone videos are viable for clinical assessment",
        ],
    )

    content = generator.generate_literature_report(lit_report)
    print("\n" + "-" * 40)
    print(content[:1500])
    print("..." if len(content) > 1500 else "")
    print("-" * 40)


def demo_baseline_training():
    """Demonstrate UPDRS baseline training with synthetic data."""
    print("\n" + "=" * 60)
    print("🧠 UPDRS Baseline Training Demo (Synthetic Data)")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report

    # Create synthetic gait features
    print("\n[1] Generating synthetic gait features...")

    np.random.seed(42)
    n_samples = 500
    n_features = 18  # Same as GaitFeatures

    # Generate features for 4 UPDRS severity levels (0-3)
    X = []
    y = []

    for severity in range(4):
        n_class = n_samples // 4

        # Base features with severity-dependent variations
        features = np.random.randn(n_class, n_features)

        # Add severity-specific patterns
        features[:, 0] += severity * 0.5  # walking speed decreases
        features[:, 4] += severity * 0.3  # hip asymmetry increases
        features[:, 6] += severity * 0.2  # trunk sway increases

        X.append(features)
        y.extend([severity] * n_class)

    X = np.vstack(X)
    y = np.array(y)

    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Class distribution: {np.bincount(y)}")

    # Train classifier
    print("\n[2] Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # Cross-validation
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"   5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Train on full data for feature importance
    clf.fit(X, y)

    # Feature importance
    print("\n[3] Feature importance (top 5):")
    feature_names = [
        "duration", "n_frames", "detection_rate", "walking_speed",
        "speed_var", "step_len_mean", "step_len_std", "step_width_mean",
        "step_width_std", "hip_flex_mean", "hip_flex_range", "knee_flex_mean",
        "knee_flex_range", "hip_asym", "knee_asym", "ankle_asym",
        "trunk_sway", "vert_osc"
    ]

    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1][:5]

    for i, idx in enumerate(indices):
        print(f"   {i+1}. {feature_names[idx]}: {importance[idx]:.4f}")

    # Binary classification (mild vs severe)
    print("\n[4] Binary classification (mild: 0-1 vs severe: 2-3)...")
    y_binary = (y >= 2).astype(int)

    scores_binary = cross_val_score(clf, X, y_binary, cv=5, scoring='accuracy')
    print(f"   5-fold CV accuracy: {scores_binary.mean():.3f} ± {scores_binary.std():.3f}")


def demo_clinical_scales():
    """Demonstrate clinical assessment scales."""
    print("\n" + "=" * 60)
    print("📋 Clinical Assessment Scales Demo")
    print("=" * 60)

    from research_automation.collection.questionnaire import (
        get_scale,
        list_scales,
        format_scale,
    )

    print("\n[1] Available scales:")
    scales = list_scales()
    for s in scales:
        print(f"   - {s.abbreviation}: {s.name} ({s.item_count} items, range {s.total_min}-{s.total_max})")

    print("\n[2] MDS-UPDRS Part III (Motor Examination):")
    updrs = get_scale("MDS-UPDRS-III")
    if updrs:
        print(f"   Name: {updrs.name}")
        print(f"   Items: {updrs.item_count}")
        print(f"   Score range: {updrs.total_min}-{updrs.total_max}")
        print(f"\n   Sample items (first 5):")
        for item in updrs.items[:5]:
            print(f"   - {item.number}. {item.name}")


def main():
    parser = argparse.ArgumentParser(description="Research Automation Demo")
    parser.add_argument(
        "--demo",
        choices=["all", "literature", "pose", "face", "experiment", "report", "baseline", "scales"],
        default="all",
        help="Which demo to run",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🔬 Research Automation Demo")
    print("=" * 60)

    demos = {
        "literature": demo_literature_search,
        "pose": demo_pose_extraction,
        "face": demo_face_extraction,
        "experiment": demo_experiment_tracking,
        "report": demo_report_generation,
        "baseline": demo_baseline_training,
        "scales": demo_clinical_scales,
    }

    if args.demo == "all":
        for name, func in demos.items():
            try:
                func()
            except Exception as e:
                print(f"\n⚠️  {name} demo failed: {e}")
    else:
        demos[args.demo]()

    print("\n" + "=" * 60)
    print("✅ Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
