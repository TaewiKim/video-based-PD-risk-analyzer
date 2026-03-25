# Analysis Runtime And Fallback

This note records the runtime regression and the backend fallback behavior that now protects the Django service from hard failures.

## Problem

`Full PD Scan` was failing with:

- `Symptom Error: No persons detected in video`

Observed on:

- [`examples/Parkinson's Disease Freezing & Festinating Gait [EQ0HG16EC3g].webm`](/workspace/video-based-PD-risk-analyzer/examples/Parkinson's Disease%20Freezing%20%26%20Festinating%20Gait%20%5BEQ0HG16EC3g%5D.webm)

Root cause:

- the Django service defaulted to `mediapipe`
- for some videos, `mediapipe` returned effectively zero usable pose detections
- symptom tracking then produced zero people
- full-scan surfaced that directly as an error

## Current Behavior

## Full-Scan Execution Path

`POST /analyze-symptoms` no longer reruns a second full pose extraction for the gait card.

Current path:

1. symptom tracker extracts tracks once
2. symptom analyzers run on those tracks
3. gait detail is computed from the selected tracked person's precomputed pose series

Files:

- [`server/webapp/analysis_views.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/analysis_views.py)
- [`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py)
- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py)

This removed the previous duplicate gait pose pass inside full-scan.

### Symptom Tracker

File:

- [`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py)

Rule:

- run tracking with requested backend
- if requested backend is `mediapipe` and no valid tracks are produced
- retry once with `rtmw`

Response evidence:

- `video_info.pose_backend`
- `video_info.pose_backend_requested`
- `video_info.pose_backend_fallback_used`

### Gait Analyzer

File:

- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py)

Rule:

- run gait pose extraction with requested backend
- if requested backend is `mediapipe` and preprocessing quality is effectively unusable
- retry once with `rtmw`

Current fallback trigger:

- `detection_rate < 0.05` or
- `mean_pose_quality < 0.10`

Response evidence:

- top-level `pose_backend`
- `pose_backend_requested`
- `pose_backend_used`
- `backend_selection`

## Runtime Tradeoff

The fallback prevents hard failure, but it is expensive.

Measured on the sample full video before unifying the full-scan path:

- symptom + gait full-scan completed successfully
- elapsed time was about `241.62s`

Measured on the same sample full video after unifying the full-scan path:

- elapsed time dropped to about `153.95s`
- symptom tracker used fallback from `mediapipe` to `rtmw`
- gait detail reused the tracked pose from `person_1`
- resulting gait summary produced `2` usable gait segments

Measured on the same sample full video after resizing the symptom-tracker RTMW fallback path:

- elapsed time dropped again to about `100.17s`
- improvement over the unified baseline was about `53.78s`
- relative reduction was about `34.9%`
- the key fix was stopping multi-person RTMW tracking from running whole-body ONNX inference on the original `2560x1440` frame
- the tracker now applies the same `max_inference_size` downscale used by the shared pose extractor before rescaling keypoints back to full-resolution coordinates

Measured on the 320px sample for gait-only:

- `mediapipe`: about `0.75s`, but zero usable gait segments
- `rtmw`: about `105.54s`, but produced one usable gait segment

## Current Interpretation

- `mediapipe` is fast but unreliable for this specific video family
- `rtmw` is slow but recovers detections
- current service behavior prioritizes completing analysis over failing fast

## MediaPipe Retry Test

We added a MediaPipe crop-retry path in:

- [`src/research_automation/pipeline/extractors/pose.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/extractors/pose.py)

Intent:

- retry MediaPipe on a few centered crops so distant subjects appear larger

Observed result on the 320px sample:

- `frames = 460`
- `valid_frames = 0`
- `seq_detection_rate = 0.0`

Interpretation:

- on this video family, simple centered crop retries are not enough to recover MediaPipe
- the remaining likely fix is person-first detection/cropping rather than blind center crops

## Person-First Crop Test

We also added a person-first crop proposal path in:

- [`src/research_automation/pipeline/extractors/pose.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/pipeline/extractors/pose.py)

Intent:

- use OpenCV face detection to propose a tighter body crop before retrying MediaPipe Pose

Observed result on the 320px sample:

- `frames = 460`
- `valid_frames = 0`
- `seq_detection_rate = 0.0`

Interpretation:

- both centered crops and the lightweight face-based person-first crop still fail on this video family
- for this dataset, the practical path remains `rtmw` fallback plus aggressive runtime optimization

## Remaining Work

- reduce fallback cost further with frame stride or proxy-video analysis on the RTMW fallback path
- consider async execution for long `rtmw` fallback cases
- make multi-person gait detail selection explicit in the UI when more than one person is detected
