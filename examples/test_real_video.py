#!/usr/bin/env python3
"""
Real Video Test Script
======================

Download a sample video and run the full analysis pipeline.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research_automation.pipeline.extractors import PoseExtractor, extract_gait_features
from research_automation.analysis import detect_tremor, detect_fog, analyze_bradykinesia


def download_sample_video(output_path: Path) -> bool:
    """Download a sample walking video using yt-dlp."""
    # Short walking video from Pexels (Creative Commons)
    # This is a sample - you can replace with any video URL
    video_url = "https://www.pexels.com/video/853889/download/"

    print(f"Downloading sample video...")

    try:
        # Try using curl first (more reliable for direct downloads)
        result = subprocess.run(
            ["curl", "-L", "-o", str(output_path), video_url],
            capture_output=True,
            timeout=60
        )
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 10000:
            return True
    except Exception as e:
        print(f"curl failed: {e}")

    # Alternative: Try yt-dlp for YouTube videos
    youtube_sample = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with actual gait video
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[height<=480]", "-o", str(output_path), youtube_sample],
            capture_output=True,
            timeout=120
        )
        if result.returncode == 0:
            return True
    except Exception:
        pass

    return False


def create_synthetic_walking_video(output_path: Path, duration: float = 3.0, fps: int = 30):
    """Create a synthetic video with a walking stick figure for testing."""
    print("Creating synthetic walking video...")

    width, height = 640, 480
    n_frames = int(duration * fps)

    # Use .avi with MJPG codec for better compatibility
    output_path = output_path.with_suffix('.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Walking animation parameters
    step_frequency = 1.5  # Hz
    step_length = 50  # pixels

    for frame_idx in range(n_frames):
        t = frame_idx / fps

        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background

        # Calculate walking phase
        phase = (t * step_frequency * 2 * np.pi) % (2 * np.pi)

        # Body center moves across screen
        center_x = 100 + int(t * 80)  # Move right
        center_y = height // 2

        # Head
        head_y = center_y - 120
        cv2.circle(frame, (center_x, head_y), 25, (100, 100, 100), -1)

        # Torso
        torso_top = head_y + 25
        torso_bottom = center_y + 20
        cv2.line(frame, (center_x, torso_top), (center_x, torso_bottom), (80, 80, 80), 8)

        # Arms (swinging opposite to legs)
        arm_swing = 30 * np.sin(phase)
        shoulder_y = torso_top + 20
        # Left arm
        left_hand = (center_x - 40 + int(arm_swing), shoulder_y + 60)
        cv2.line(frame, (center_x - 20, shoulder_y), left_hand, (80, 80, 80), 6)
        # Right arm
        right_hand = (center_x + 40 - int(arm_swing), shoulder_y + 60)
        cv2.line(frame, (center_x + 20, shoulder_y), right_hand, (80, 80, 80), 6)

        # Legs (walking motion)
        hip_y = torso_bottom
        leg_length = 100
        leg_swing = 25 * np.sin(phase)

        # Left leg
        left_knee_x = center_x - 15 + int(leg_swing * 0.5)
        left_knee_y = hip_y + 50
        left_foot_x = center_x - 20 + int(leg_swing)
        left_foot_y = hip_y + leg_length
        cv2.line(frame, (center_x - 15, hip_y), (left_knee_x, left_knee_y), (80, 80, 80), 8)
        cv2.line(frame, (left_knee_x, left_knee_y), (left_foot_x, left_foot_y), (80, 80, 80), 8)

        # Right leg (opposite phase)
        right_knee_x = center_x + 15 - int(leg_swing * 0.5)
        right_knee_y = hip_y + 50
        right_foot_x = center_x + 20 - int(leg_swing)
        right_foot_y = hip_y + leg_length
        cv2.line(frame, (center_x + 15, hip_y), (right_knee_x, right_knee_y), (80, 80, 80), 8)
        cv2.line(frame, (right_knee_x, right_knee_y), (right_foot_x, right_foot_y), (80, 80, 80), 8)

        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx + 1}/{n_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

        writer.write(frame)

    writer.release()
    print(f"Created {n_frames} frames at {fps} FPS")
    print(f"Saved to: {output_path}")
    return output_path


def analyze_video(video_path: Path):
    """Run full analysis pipeline on video."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {video_path.name}")
    print(f"{'='*60}")

    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    print(f"\n[Video Info]")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {duration:.1f}s")

    # Step 1: Extract poses
    print(f"\n[1] Extracting poses...")
    with PoseExtractor() as extractor:
        pose_sequence = extractor.extract_from_video(str(video_path))

    # Count frames with valid keypoints
    valid_frames = [f for f in pose_sequence.frames if f.keypoints is not None]
    print(f"    Total frames: {len(pose_sequence.frames)}")
    print(f"    Valid detections: {len(valid_frames)}/{frame_count}")
    detection_rate = len(valid_frames) / frame_count * 100 if frame_count > 0 else 0
    print(f"    Detection rate: {detection_rate:.1f}%")

    if len(valid_frames) < 10:
        print("    ⚠ Not enough detections for analysis (need real human video)")
        print("    Tip: Try with a video containing a person, or use webcam")
        return None

    # Convert to numpy array
    keypoints = pose_sequence.to_array()
    print(f"    Keypoints shape: {keypoints.shape}")

    # Step 2: Gait features
    print(f"\n[2] Extracting gait features...")
    gait = extract_gait_features(pose_sequence)

    print(f"    Walking speed: {gait.walking_speed:.4f}")
    print(f"    Speed variability: {gait.speed_variability:.4f}")
    print(f"    Step length (mean): {gait.step_length_mean:.4f}")
    print(f"    Hip asymmetry: {gait.hip_asymmetry:.2f}°")
    print(f"    Knee asymmetry: {gait.knee_asymmetry:.2f}°")

    # Step 3: Tremor analysis
    print(f"\n[3] Tremor detection...")
    tremor = detect_tremor(keypoints, fps)

    print(f"    Tremor detected: {tremor.tremor_detected}")
    print(f"    Dominant frequency: {tremor.dominant_frequency:.2f} Hz")
    print(f"    Max amplitude: {tremor.amplitude_max:.4f}")
    print(f"    Regularity: {tremor.regularity:.2f}")
    if tremor.tremor_detected:
        print(f"    Type: {tremor.tremor_type.value}")

    # Step 4: FOG detection
    print(f"\n[4] Freezing of Gait detection...")
    fog = detect_fog(keypoints, fps)

    print(f"    FOG detected: {fog.fog_detected}")
    print(f"    Episodes: {fog.n_episodes}")
    print(f"    Freezing index: {fog.freezing_index:.3f}")
    if fog.fog_detected:
        print(f"    Total FOG duration: {fog.total_fog_duration:.2f}s")
        print(f"    FOG percentage: {fog.fog_percentage:.1f}%")

    # Step 5: Bradykinesia analysis
    print(f"\n[5] Bradykinesia analysis...")
    brady = analyze_bradykinesia(keypoints, fps)

    print(f"    Mean speed: {brady.mean_speed:.4f}")
    print(f"    Speed decrement: {brady.speed_decrement:.4f}")
    print(f"    Amplitude decrement: {brady.amplitude_decrement:.4f}")
    print(f"    Hesitation ratio: {brady.hesitation_ratio:.2%}")
    print(f"    Pause count: {brady.pause_count}")
    print(f"    Bradykinesia score: {brady.bradykinesia_score:.1f}/4 (UPDRS-like)")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Gait quality: {'Normal' if gait.walking_speed > 0.05 else 'Slow'}")
    print(f"  Tremor: {'Yes' if tremor.tremor_detected else 'No'}")
    print(f"  FOG episodes: {fog.n_episodes}")
    print(f"  Movement quality: {['Normal', 'Mild', 'Moderate', 'Severe', 'Very Severe'][int(brady.bradykinesia_score)]}")

    return {
        'gait': gait,
        'tremor': tremor,
        'fog': fog,
        'bradykinesia': brady
    }


def analyze_synthetic_poses():
    """Analyze synthetic pose data to demonstrate capabilities."""
    print(f"\n{'='*60}")
    print("Analyzing SYNTHETIC pose data (demo)")
    print(f"{'='*60}")

    fps = 30.0
    duration = 5.0
    n_frames = int(fps * duration)
    n_joints = 33  # MediaPipe pose

    print(f"\n[Synthetic Data]")
    print(f"  Duration: {duration}s")
    print(f"  FPS: {fps}")
    print(f"  Frames: {n_frames}")

    # Create realistic walking motion
    print(f"\n[1] Generating walking keypoints...")
    t = np.linspace(0, duration, n_frames)

    # Base skeleton (normalized coordinates 0-1)
    base_pose = np.zeros((n_joints, 3))
    # Head (0)
    base_pose[0] = [0.5, 0.15, 1.0]
    # Shoulders (11, 12)
    base_pose[11] = [0.4, 0.25, 1.0]
    base_pose[12] = [0.6, 0.25, 1.0]
    # Hips (23, 24)
    base_pose[23] = [0.45, 0.5, 1.0]
    base_pose[24] = [0.55, 0.5, 1.0]
    # Knees (25, 26)
    base_pose[25] = [0.45, 0.7, 1.0]
    base_pose[26] = [0.55, 0.7, 1.0]
    # Ankles (27, 28)
    base_pose[27] = [0.45, 0.9, 1.0]
    base_pose[28] = [0.55, 0.9, 1.0]

    # Walking animation
    keypoints = np.zeros((n_frames, n_joints, 3))
    walk_freq = 1.2  # Hz (steps per second)

    for i, time in enumerate(t):
        phase = 2 * np.pi * walk_freq * time

        # Copy base pose
        keypoints[i] = base_pose.copy()

        # Leg swing
        left_swing = 0.03 * np.sin(phase)
        right_swing = 0.03 * np.sin(phase + np.pi)

        # Left leg
        keypoints[i, 25, 0] += left_swing  # knee
        keypoints[i, 27, 0] += left_swing * 1.5  # ankle
        keypoints[i, 27, 1] += 0.01 * np.abs(np.sin(phase))  # foot lift

        # Right leg
        keypoints[i, 26, 0] += right_swing
        keypoints[i, 28, 0] += right_swing * 1.5
        keypoints[i, 28, 1] += 0.01 * np.abs(np.sin(phase + np.pi))

        # Arm swing (opposite to legs)
        keypoints[i, 11, 0] += -left_swing * 0.5  # left shoulder
        keypoints[i, 12, 0] += -right_swing * 0.5  # right shoulder

        # Subtle trunk sway
        keypoints[i, 0, 0] += 0.005 * np.sin(phase * 2)  # head
        keypoints[i, 11:13, 0] += 0.003 * np.sin(phase * 2)  # shoulders

        # Add small noise (sensor noise simulation)
        keypoints[i, :, :2] += np.random.normal(0, 0.002, (n_joints, 2))

    print(f"    Generated keypoints: {keypoints.shape}")

    # Run analysis
    from research_automation.pipeline.extractors import GaitFeatures

    # Step 2: Gait features
    print(f"\n[2] Extracting gait features...")

    # Calculate basic gait metrics manually
    ankle_left = keypoints[:, 27, :2]
    ankle_right = keypoints[:, 28, :2]

    # Speed from ankle movement
    velocity_left = np.diff(ankle_left, axis=0) * fps
    velocity_right = np.diff(ankle_right, axis=0) * fps
    speed_left = np.sqrt(velocity_left[:, 0]**2 + velocity_left[:, 1]**2)
    speed_right = np.sqrt(velocity_right[:, 0]**2 + velocity_right[:, 1]**2)
    walking_speed = (np.mean(speed_left) + np.mean(speed_right)) / 2

    print(f"    Walking speed: {walking_speed:.4f}")
    print(f"    Speed variability: {np.std(speed_left):.4f}")

    # Step 3: Tremor analysis
    print(f"\n[3] Tremor detection...")
    tremor = detect_tremor(keypoints, fps)

    print(f"    Tremor detected: {tremor.tremor_detected}")
    print(f"    Dominant frequency: {tremor.dominant_frequency:.2f} Hz")
    print(f"    Max amplitude: {tremor.amplitude_max:.4f}")
    print(f"    Regularity: {tremor.regularity:.2f}")

    # Step 4: FOG detection
    print(f"\n[4] Freezing of Gait detection...")
    fog = detect_fog(keypoints, fps)

    print(f"    FOG detected: {fog.fog_detected}")
    print(f"    Episodes: {fog.n_episodes}")
    print(f"    Freezing index: {fog.freezing_index:.3f}")

    # Step 5: Bradykinesia analysis
    print(f"\n[5] Bradykinesia analysis...")
    brady = analyze_bradykinesia(keypoints, fps)

    print(f"    Mean speed: {brady.mean_speed:.4f}")
    print(f"    Speed decrement: {brady.speed_decrement:.4f}")
    print(f"    Amplitude decrement: {brady.amplitude_decrement:.4f}")
    print(f"    Hesitation ratio: {brady.hesitation_ratio:.2%}")
    print(f"    Pause count: {brady.pause_count}")
    print(f"    Bradykinesia score: {brady.bradykinesia_score:.1f}/4 (UPDRS-like)")

    # Summary
    print(f"\n{'='*60}")
    print("Summary (Synthetic Normal Walking)")
    print(f"{'='*60}")
    print(f"  Tremor: {'Yes' if tremor.tremor_detected else 'No'}")
    print(f"  FOG episodes: {fog.n_episodes}")
    score = int(brady.bradykinesia_score)
    labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Very Severe']
    print(f"  Movement quality: {labels[min(score, 4)]}")

    return {
        'tremor': tremor,
        'fog': fog,
        'bradykinesia': brady
    }


def main():
    print("="*60)
    print("🎬 Real Video Analysis Test")
    print("="*60)

    # Check for command line video path
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--synthetic":
            # Run synthetic pose analysis
            analyze_synthetic_poses()
            return

        video_path = Path(arg)
        if video_path.exists():
            result = analyze_video(video_path)
            if result is None:
                print("\n" + "="*60)
                print("Running synthetic demo instead...")
                analyze_synthetic_poses()
            return
        else:
            print(f"Video not found: {video_path}")

    # Check for existing videos in data/videos/raw
    raw_dir = Path("data/videos/raw")
    if raw_dir.exists():
        videos = list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.avi")) + list(raw_dir.glob("*.mov"))
        # Filter out synthetic videos
        real_videos = [v for v in videos if "synthetic" not in v.name.lower()]
        if real_videos:
            print(f"Found {len(real_videos)} video(s) in {raw_dir}")
            for video in real_videos:
                analyze_video(video)
            return

    # No real videos - run synthetic demo
    print("\nNo real video files found.")
    print("Usage: python test_real_video.py <video_path>")
    print("       python test_real_video.py --synthetic")
    print("\nRunning synthetic pose analysis demo...")
    analyze_synthetic_poses()


if __name__ == "__main__":
    main()
