#!/usr/bin/env python3
"""
Record a short video from webcam for testing.
"""

import cv2
import sys
from pathlib import Path

def record_webcam(output_path: str, duration: float = 5.0, fps: int = 30):
    """Record video from webcam."""
    print(f"Recording {duration}s video from webcam...")
    print("Stand and walk in front of the camera.")
    print("Press 'q' to stop early, 's' to start recording.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False

    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    print(f"Webcam: {width}x{height} @ {actual_fps:.1f} FPS")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    max_frames = int(duration * fps)
    frame_count = 0
    recording = False

    print("\nPress 's' to START recording, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display status on frame
        status = "RECORDING" if recording else "Press 's' to start"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if recording:
            cv2.putText(frame, f"Frame: {frame_count}/{max_frames}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)
            frame_count += 1

            if frame_count >= max_frames:
                break

        cv2.imshow('Webcam Recording', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not recording:
            recording = True
            print("Recording started!")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        print(f"\n✓ Saved: {output_path}")
        print(f"  Frames: {frame_count}")
        print(f"  Duration: {frame_count/fps:.1f}s")
        return True
    else:
        print("No frames recorded")
        return False


def main():
    output_path = "data/videos/raw/webcam_test.mp4"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    duration = 5.0
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            pass

    if record_webcam(output_path, duration=duration):
        print("\nNow run analysis:")
        print(f"  uv run python examples/test_real_video.py {output_path}")


if __name__ == "__main__":
    main()
