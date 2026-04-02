# Stroke Gait Calibration Workflow

## Purpose

This workflow is for offline review of the literature-based stroke gait screen.
It is used to inspect score distributions, speed-band distributions, and likely threshold pressure before dataset-backed recalibration.

## Inputs

Supported inputs today:

- stored platform result JSON files with a top-level `summary`
- stored symptom result JSON files with `gait_analysis.summary`
- external summary JSON files
- external summary CSV files

The calibration utilities extract or normalize the gait summary and rebuild the stroke gait analysis output in a consistent format.

## Core Modules

- scorer: [`server/webapp/stroke_gait_analysis.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/stroke_gait_analysis.py)
- calibration helpers: [`server/webapp/stroke_gait_calibration.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/stroke_gait_calibration.py)
- dataset adapters: [`server/webapp/stroke_gait_dataset_adapters.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/stroke_gait_dataset_adapters.py)
- CLI: [`scripts/score_stroke_gait_results.py`](/workspace/video-based-PD-risk-analyzer/scripts/score_stroke_gait_results.py)

## CLI Usage

Single-file scoring:

```bash
python scripts/score_stroke_gait_results.py server/runtime/results/fce59db8c5a44e32a2a3028c541faad9_results.json
```

Directory-wide aggregate report:

```bash
python scripts/score_stroke_gait_results.py \
  --report \
  --output output/stroke_gait_calibration_report.json \
  server/runtime/results
```

Custom glob:

```bash
python scripts/score_stroke_gait_results.py \
  --report \
  --glob '*_symptoms.json' \
  server/runtime/results
```

External summary JSON:

```bash
python scripts/score_stroke_gait_results.py \
  --input-format summary_json \
  --report \
  data/stroke_cohort_summary.json
```

External summary CSV:

```bash
python scripts/score_stroke_gait_results.py \
  --input-format summary_csv \
  --report \
  data/stroke_cohort_summary.csv
```

Voisard 2025 root directory:

```bash
python scripts/score_stroke_gait_results.py \
  --input-format voisard_2025 \
  --report \
  data/external/voisard_2025/unpacked
```

Voisard 2025 CVA-only report:

```bash
python scripts/score_stroke_gait_results.py \
  --input-format voisard_2025 \
  --cohort CVA \
  --report \
  data/external/voisard_2025/unpacked
```

Voisard 2025 CVA vs HS comparison:

```bash
python scripts/score_stroke_gait_results.py \
  --input-format voisard_2025 \
  --cohort CVA \
  --cohort HS \
  --compare-groups CVA HS \
  --report \
  data/external/voisard_2025/unpacked
```

Threshold profile comparison:

```bash
python scripts/tune_stroke_gait_thresholds.py \
  data/external/voisard_2025/unpacked \
  output/voisard_hs_threshold_profiles.json
```

Preset-based adapter:

```bash
python scripts/score_stroke_gait_results.py \
  --input-format summary_csv \
  --adapter-preset clinical_gait_parameters \
  --report \
  data/clinical_gait_parameters.csv
```

Preset plus mapping override:

```bash
python scripts/score_stroke_gait_results.py \
  --input-format summary_csv \
  --adapter-preset clinical_gait_parameters \
  --mapping-file configs/stroke_gait_adapter.json \
  --report \
  data/public_cohort.csv
```

Auto mode rules:

- `.csv` files use the summary CSV adapter
- JSON files with `summary` or `gait_analysis.summary` use the platform adapter
- other JSON files use the summary JSON adapter

## Adapter Mapping

The external dataset adapter normalizes common field aliases into the platform summary schema:

- `avg_speed`: `speed`, `gait_speed`, `walking_speed`, `speed_mps`
- `avg_cadence`: `cadence`, `step_cadence`, `cadence_spm`
- `avg_asymmetry`: `asymmetry`, `gait_asymmetry`, `spatial_asymmetry`
- `avg_stability`: `stability`, `gait_stability`, `dynamic_stability`
- `avg_stride_time_cv`: `stride_time_cv`, `stride_time_cv_pct`, `temporal_variability`
- `avg_step_time_asymmetry`: `step_time_asymmetry`, `temporal_asymmetry`, `step_timing_asymmetry`

Optional metadata columns are preserved on each calibration record when present:

- `record_id`
- `subject_id`
- `participant_id`
- `label`
- `cohort`

Built-in presets today:

- `default`: generic platform-like summary aliases
- `clinical_gait_parameters`: clinical cohort CSV aliases such as `Speed`, `Cadence`, `StrideTimeCV`, `StepTimeAsymmetry`

Special input formats today:

- `voisard_2025`: reads trial folders directly from the Voisard dataset layout and derives gait summary values from `*_meta.json` plus `*_processed_data.txt`

Mapping file shape:

```json
{
  "summary_fields": {
    "avg_speed": ["preferred_speed"],
    "avg_cadence": ["cadence_steps_per_min"]
  },
  "identity_fields": ["subject_id", "pathology", "visit"],
  "metadata_renames": {
    "pathology": "cohort"
  }
}
```

## Report Fields

The aggregate report currently includes:

- `n_records`
- `pattern_level_counts`
- `speed_band_counts`
- `mean_pattern_score`
- `mean_flagged_count`
- `domain_flag_counts`
- `cohort_counts`
- `label_counts`
- `threshold_review`
- `top_examples`
- `group_comparison` when `--compare-groups` is used

`threshold_review` is intended for fast human review before changing thresholds:

- `prominent_pattern_share`
- `limited_community_share`
- `max_pattern_score`
- `min_pattern_score`

## Interpretation

If the report shows:

- too many `High` cases in apparently non-stroke samples
- too many `household_only` or `limited_community` bands in mixed cohorts
- one domain dominating nearly every case

then the current thresholds should be reviewed before wider rollout.

## Next Calibration Step

The next implementation target is a dataset adapter layer that can ingest public stroke gait cohorts and map them into the same summary schema used by the platform:
That adapter is now in place for CSV and summary JSON sources, with preset and mapping-file support. The next step is to add dataset-specific preset files for concrete public cohorts and calibration notebooks for threshold review.
