# UI Data Flow

This document records which backend payload drives each major UI section so the frontend does not regress back into mixed or duplicated timeline behavior.

공통 activity payload 정의는 [`docs/ACTIVITY_SCHEMA.md`](/workspace/video-based-PD-risk-analyzer/docs/ACTIVITY_SCHEMA.md)를 기준으로 합니다.

## Current Rule

There are two different timeline concepts in the Django UI and they must stay separate:

1. Symptom activity segmentation
2. Gait phase segmentation

They are not interchangeable.

The current UI grouping is:

1. `Activity Classification`
2. `Symptom Screening`
3. `Gait And Turning`

`turning` belongs to the gait section. It must not be presented as a standing subtype.

The video review panel now lives inside the right-side analysis card when results are visible.

Reason:

- users need to compare the visible skeleton overlay against the same tracked person used by the analysis without looking back and forth between two cards
- the left-side `Video Input` card is hidden once results are shown, so the analysis view does not keep two competing video panels on screen

## Symptom Activity Segmentation

UI section:

- [`server/webapp/templates/webapp/index.html`](/workspace/video-based-PD-risk-analyzer/server/webapp/templates/webapp/index.html)
- heading: `Activity Classification`
- subheading: `Symptom Routing View`

Backend source:

- `persons[*].activity_schema.routing.segments`
- `persons[*].activity_schema.routing.summary`

Meaning:

- Used by the symptom analyzer
- Groups time into `walking`, `resting`, `task`, `standing`
- Purpose is to decide which symptom modules are appropriate for each segment

## Gait Phase Timeline

UI section:

- [`server/webapp/templates/webapp/index.html`](/workspace/video-based-PD-risk-analyzer/server/webapp/templates/webapp/index.html)
- heading: `Gait Phase Timeline`
- section group: `Gait And Turning`

Backend source:

- `persons[*].activity_schema.gait_phase.segments`
- `persons[*].activity_schema.gait_phase.summary`
- fallback: `gait_analysis.activity_timeline`

Meaning:

- Used by the gait biomarker engine
- Groups time into `walking`, `turning`, `standing`
- Purpose is to show the newer gait-specific split, including turn analysis

The gait detail card in the same section also surfaces:

- `turn_detection.summary.total_segments`
- `turn_detection.summary.hesitant_turns`
- `turn_detection.summary.possible_fog_turns`
- `turn_detection.summary.turning_while_walking`
- `activity_timeline.summary.turning_time`

## Empty Section Rule

Detailed symptom sections must not remain visible when there is no summary payload for that symptom.

Current UI behavior:

- top overview cards are shown only for symptoms with actual summary data
- detailed sections for `tremor`, `bradykinesia`, and `posture` are hidden if the selected person has no summary for them
- the `fog transition` section is hidden when there are no detected transitions
- `Activity Classification` and `Gait Phase Timeline` both expose clickable segment lists that jump the review video to the segment start time

Reason:

- many videos produce only a subset of symptom outputs
- leaving all detail cards visible fills the page with placeholder `-` values and makes the UI look broken

## Why The UI Looked Broken

The old segment UI was still present, but it was reading only:

- `persons[*].activity_segments`

At the same time, the new gait split was being returned under:

- `gait_analysis.activity_timeline`

That made it look like the previous segmented UI had disappeared, when in reality the UI and backend were using two different segmentation systems.

There was a second regression in the full-scan path:

- `displaySymptomResults(...)` showed only the symptom container
- the older gait detail container stayed hidden
- that removed the previous gait charts, segment list, statistical summary, and benchmark table from the `Full PD Scan` flow

This is now fixed by rendering symptom results and the detailed gait review together in the same right-side card.

Evidence screenshots captured from the running Django UI:

- before analysis: [`outputs/ui_review/ui_before_analysis.png`](/workspace/video-based-PD-risk-analyzer/outputs/ui_review/ui_before_analysis.png)
- full-scan top section after loading saved symptom result: [`outputs/ui_review/ui_after_analysis_fullscan_top.png`](/workspace/video-based-PD-risk-analyzer/outputs/ui_review/ui_after_analysis_fullscan_top.png)
- detailed gait review section in the same full-scan flow: [`outputs/ui_review/ui_after_analysis_fullscan_gait_detail.png`](/workspace/video-based-PD-risk-analyzer/outputs/ui_review/ui_after_analysis_fullscan_gait_detail.png)

## Consolidation Done

To reduce duplicate behavior:

- timeline drawing now uses a shared `renderTimeline(...)` helper in the Django template
- pose backend default selection is centralized in:
  - [`server/webapp/analysis_config.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/analysis_config.py)
- symptom-mode gait detail rendering now reuses the existing gait result section instead of maintaining a second copy of the same charts and segment UI

Current full-scan rendering rule:

- symptom summary uses [`displaySymptomResults(...)`](/workspace/video-based-PD-risk-analyzer/server/webapp/templates/webapp/index.html)
- gait detail cards in the same flow reuse [`displayResults(...)`](/workspace/video-based-PD-risk-analyzer/server/webapp/templates/webapp/index.html)
- `/analyze-symptoms` therefore must continue returning `gait_analysis.video_info`, `gait_analysis.user`, and the full gait payload expected by `displayResults(...)`

Configured backend preference remains:

- `mediapipe`

Effective runtime backend for many real videos is often:

- `rtmw` fallback

Do not conflate configured default with effective runtime backend in UI or docs.

## Regression Guardrails

When editing the UI:

- do not replace symptom activity data with gait timeline data unless the section is explicitly renamed
- do not merge `task` and `turning` as if they are the same concept
- do not render `turning` only in the timeline while omitting it from the gait summary cards
- do not change pose backend defaults separately in `smart_analyzer.py` and `pd_symptoms_analyzer.py`
- document any new payload field in this file before wiring it into the template
