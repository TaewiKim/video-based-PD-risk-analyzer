# Stroke Gait Analysis V1

## Scope

This module is a literature-based gait pattern screen for stroke-linked walking impairment.
It is not a cerebrovascular event predictor and it must not be presented as a stroke diagnosis tool.

The output is intended for:

- screening of post-stroke-like gait patterns from a short walking video
- rehabilitation-oriented review of gait burden over time
- branch-level interpretation downstream of the shared gait analysis layer

## Literature Basis

The rule model follows three consistent findings from the gait literature:

1. Walking speed is a core functional marker after stroke.
2. Spatial and temporal asymmetry provide clinically relevant information beyond speed alone.
3. Video-derived gait parameters are usable for stroke-related gait assessment when pose quality is adequate.

Primary references used for the v1 design:

- Wonsetler et al. 2017, Top Stroke Rehabil. Systematic review on gait parameters related to function after stroke.
- Xu et al. 2025, Gait & Posture. Meta-analysis of stroke-related factors associated with gait asymmetry in ambulatory stroke survivors.
- Stenum et al. 2024, PLOS Digital Health. Video-based estimation of clinically relevant gait parameters in Parkinson's disease and stroke.
- Cleland et al. 2023, Physical Therapy. Video-pose estimation for gait parameter extraction in people with stroke.

## Model Type

- schema version: `stroke_gait_analysis_v1`
- analysis type: `literature_rule_screen`
- score family: 4-domain additive burden score

## Input Features

The model consumes gait summary features already produced by the platform:

- `avg_speed`
- `avg_cadence`
- `avg_asymmetry`
- `avg_stability`
- `avg_stride_time_cv`
- `avg_step_time_asymmetry`

When available from external clinical cohorts, the model also consumes rehabilitation-oriented mobility metadata:

- `tug_seconds`
- `turn_duration_seconds`
- `visual_gait_assessment`

## Domains

The score contains four domains, each scored from 0 to 2:

1. Walking capacity
2. Spatial asymmetry
3. Temporal variability
4. Dynamic stability

Maximum total score: 8

## Rules

### 1. Walking Capacity

Evidence:

- slow walking speed
- reduced cadence

Rules:

- speed < 0.4 m/s -> severe domain burden
- 0.4 m/s <= speed < 0.8 m/s -> moderate domain burden
- cadence < 90 steps/min can also push the domain to severe
- cadence < 100 steps/min can push the domain to moderate

Speed band output:

- `household_only` for < 0.4 m/s
- `limited_community` for 0.4 to < 0.8 m/s
- `community_plus` for >= 0.8 m/s

### 2. Spatial Asymmetry

Evidence:

- mean gait asymmetry

Rules:

- asymmetry > 0.35 -> severe
- asymmetry > 0.20 -> moderate

### 3. Temporal Variability

Evidence:

- stride time coefficient of variation
- step-time asymmetry

Rules:

- stride time CV > 8% -> severe
- stride time CV > 4% -> moderate
- step-time asymmetry > 0.08 -> severe
- step-time asymmetry > 0.05 -> moderate

The domain score is the max score from these temporal indicators.

### 4. Dynamic Stability

Evidence:

- platform stability score

Rules:

- stability < 0.08 -> severe
- stability < 0.20 -> moderate

## Total Pattern Level

Total domain score:

- 0 to 1 -> `Low`
- 2 to 4 -> `Moderate`
- 5 to 8 -> `High`

The user-facing interpretation should describe the result as a stroke-linked gait pattern burden, not stroke probability.

## Output Schema

Top-level fields:

- `available`
- `schema_version`
- `analysis_type`
- `label`
- `pattern_level`
- `pattern_score`
- `pattern_score_max`
- `risk_score`
- `speed_band`
- `domain_scores`
- `flagged_indicators`
- `summary_note`
- `clinical_note`
- `technical_note`

Backward compatibility:

- the same object is currently exposed under both `stroke_screening` and `stroke_gait_analysis`

## UI Guidance

Recommended labels:

- Korean: `뇌졸중 연관 보행 패턴`
- English: `Stroke-linked Gait Pattern`

Do not label the main number as "stroke probability".
Prefer:

- `패턴 점수`
- `패턴 수준`
- `보행 속도 밴드`

## Known Limitations

- the current model is rule-based and not dataset-calibrated
- no large open raw-video stroke gait dataset is currently wired into training
- thresholds should be re-calibrated once public stroke cohorts are integrated
- low-quality pose tracks should suppress overconfident interpretation

## Current Profiles

Available threshold profiles in code:

- `DEFAULT_STROKE_GAIT_THRESHOLDS`: literature-first baseline
- `VOISARD_HS_CALIBRATED_THRESHOLDS`: stricter profile adjusted to reduce `HS` false positive burden on the Voisard 2025 cohort

## Output Modes

The stroke gait output is now split into two interpretation layers:

- `video_only`: uses only gait-derived summary features and is the default product path
- `clinical_augmented`: adds cohort-level mobility metadata such as `TUG`, `turn_duration_seconds`, and `visual_gait_assessment` when available

For platform UI and product behavior, `video_only` should remain the default. `clinical_augmented` is intended for dataset analysis or clinically enriched workflows.

## Next Steps

1. Calibrate thresholds on public stroke gait datasets.
2. Add turning and segment-consistency features.
3. Evaluate sensitivity and false positive burden with clinician review.
4. Promote the model to v2 only after dataset-backed threshold revision.
