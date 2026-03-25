# UI API Alignment

This note records the current alignment rules between the Django UI and Django API so the frontend does not drift away from the response schema again.

## Current Contracts

### Full PD Scan

Endpoint:

- `POST /analyze-symptoms`

UI consumers:

- [`displaySymptomResults(...)`](/workspace/video-based-PD-risk-analyzer/server/webapp/templates/webapp/index.html)
- [`displayResults(...)`](/workspace/video-based-PD-risk-analyzer/server/webapp/templates/webapp/index.html)

Required payload:

- root `video_info`
- root `n_persons`
- root `activity_schema`
- root `persons[*].activity_segments`
- root `persons[*].activity_breakdown`
- root `persons[*].activity_schema`
- root `persons[*].symptoms`
- root `persons[*].skeleton_track`
- root `gait_analysis`

Required `gait_analysis` fields:

- `pose_backend`
- `video_info`
- `user`
- `walking_detection`
- `analysis_results`
- `statistical_analysis`
- `ml_inference`
- `summary`
- `activity_timeline`
- `turn_detection`
- `turn_analysis`
- `turn_methodology`

Required `activity_schema` fields:

- root `activity_schema.routing.summary`
- root `activity_schema.gait_phase.summary`
- `persons[*].activity_schema.routing.segments`
- `persons[*].activity_schema.routing.summary`
- `persons[*].activity_schema.gait_phase` when the person is the gait source person

Required `persons[*].skeleton_track` fields for overlay rendering:

- `start_frame`
- `frame_stride`
- `n_frames`
- `frames`
- `keypoints`
- `pose_quality`

Current overlay rule:

- in full-scan mode, the review overlay is expected to match the actual gait-analysis source person
- `skeleton_track.frame_stride` is now `1` so the visible review skeleton stays closer to the frame-level pose used by analysis
- `frames` is the stable UI field; `keypoints` is kept as an alias for backward compatibility
- the overlay uses the same tracked person as gait analysis, but not the same processed pose representation

Reason:

- full-scan mode now renders both the symptom section and the reused gait-detail section
- gait-detail now reuses the tracked pose from the selected primary symptom track instead of triggering a second full gait extraction
- the gait-detail section uses the same renderer as the gait-only flow

### Gait Only

Endpoint:

- `POST /analyze`

UI consumer:

- [`displayResults(...)`](/workspace/video-based-PD-risk-analyzer/server/webapp/templates/webapp/index.html)

Required payload:

- `video_info`
- `walking_detection`
- `analysis_results`
- `summary`
- `statistical_analysis`
- `ml_inference`

## Mismatches Found And Fixed

### 1. Full-scan hid the old gait-detail UI

Problem:

- symptom mode showed only the symptom container
- the older gait charts, segment list, and statistical tables were still in the template but hidden

Fix:

- `displaySymptomResults(...)` now reuses `displayResults(...)`
- full-scan shows both symptom summary and gait-detail review

### 2. `/analyze-symptoms` did not return all fields expected by the reused gait UI

Problem:

- `displayResults(...)` expects `video_info` and `user`
- `gait_analysis` in the symptom response did not always include them

Fix:

- [`server/webapp/analysis_views.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/analysis_views.py) now copies `video_info`, `user`, and `pose_backend` into `gait_analysis`

### 5. Gait analyzer omitted its own backend field

Problem:

- the UI contract expected `gait_analysis.pose_backend`
- `analysis_views.py` forwarded the field correctly
- but [`SmartGaitAnalyzer`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py) did not include it in its response payload

Fix:

- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py) now emits top-level `pose_backend`
- preprocessing checks also include numeric `detection_rate` and `mean_pose_quality` values for UI/debug use

### 3. Duplicate DOM id for festination

Problem:

- the symptom FOG card and transition FOG card both used `id="fogFestination"`
- JS updates were writing to the wrong card

Fix:

- transition card now uses `fogTransitionFestination`

### 4. Video response content type was wrong for non-MP4 uploads

Problem:

- `/videos/<filename>` always returned `video/mp4`
- stored uploads are often `.webm`

Fix:

- [`server/webapp/media_views.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/media_views.py) now uses `mimetypes.guess_type(...)`

### 6. Skeleton overlay looked worse than the underlying pose quality

Problem:

- the UI overlay drew every joint and every edge without checking confidence
- low-confidence foot and toe landmarks made the on-screen skeleton look much noisier than the backend gait quality metrics

Fix:

- [`server/webapp/templates/webapp/index.html`](/workspace/video-based-PD-risk-analyzer/server/webapp/templates/webapp/index.html) now filters joints and connections by confidence
- foot joints use a stricter threshold than torso joints
- low-quality frames are skipped instead of drawing a misleading full skeleton

### 7. Symptom classification could disagree with the actual gait analysis summary

Problem:

- the upper symptom classification box used only `persons[*].symptoms[*].summary.overall_severity`
- for the gait source person, that could disagree with the dedicated `gait_analysis.summary.overall_pd_risk_level`

Fix:

- the symptom classification box now prioritizes `gait_analysis.summary` when the selected person is the gait analysis source person
- this keeps the visible classification consistent with the actual gait result shown in the same screen

### 8. Video overlay status used the wrong activity source

Problem:

- the video badge used `walking_detection.segments` only
- that collapsed `turning` and `standing` into the same `Not Walking` state even when `activity_timeline` had a dedicated turn segment

Fix:

- the video badge now reads `gait_analysis.activity_timeline.segments`
- visible states are `Walking`, `Turning`, and `Standing`
- gait phase review lists also use `label` as well as `activity_type`, so turn segments remain visible in the UI

### 9. Activity payloads were split across multiple incompatible shapes

Problem:

- root `activity_summary`, person `activity_breakdown`, person `activity_segments`, and `gait_analysis.activity_timeline` all represented activity in different ways

Fix:

- the API now emits a common `activity_schema`
- saved history payloads are normalized on load so older files also expose `activity_schema`

### 10. Saved-result preload and History load could stop on a JS exception

Problem:

- `displayResults(...)` referenced `walking.segments` without defining `walking`
- this threw `ReferenceError: walking is not defined`
- both `?result=<file>` preload and History `Load` used the same path, so both failed silently into the catch handler

Fix:

- `displayResults(...)` now initializes `const walking = results.walking_detection || { segments: [] }`
- `?result=` preload and History `Load` both render the saved result again

### 11. Root activity schema and skeleton review payload were inconsistent across live vs history paths

Problem:

- root `activity_schema.gait_phase` could be `null` in saved payloads even when `gait_analysis.activity_timeline` existed
- some saved results exposed only `skeleton_track.keypoints`, while UI/debug tooling increasingly expected `frames`

Fix:

- `normalize_activity_schema(...)` now backfills root and person `gait_phase`
- `normalize_activity_schema(...)` also adds `frames <-> keypoints` aliases for `skeleton_track`
- new saved results emit both `frames` and `keypoints`

## Known Remaining Gaps

### Old saved results

Some saved JSON files were produced before the current `gait_analysis` schema was finalized.

Impact:

- loading a historical result can still show empty gait-detail cards if `analysis_results` is empty

Current UI behavior:

- explicit empty state: `No Gait Segment`

### Root-level analysis errors in saved symptom results

If an older saved symptom result contains a root `error`, the UI may still try to render partial data.

Current status:

- live analysis path blocks on `error`
- saved-result path is still tolerant rather than strict

## Regression Guardrails

- if `displayResults(...)` is reused in another flow, include its required payload fields explicitly in the backend response
- do not reuse DOM ids between symptom panels
- do not hardcode video MIME types in API responses
- when changing response fields, update this document and [`docs/UI_DATA_FLOW.md`](/workspace/video-based-PD-risk-analyzer/docs/UI_DATA_FLOW.md) together
