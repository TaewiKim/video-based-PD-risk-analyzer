from __future__ import annotations

import json

from server.webapp.stroke_gait_dataset_adapters import (
    build_adapter_metadata,
    load_summary_csv_records,
    load_summary_json_records,
    load_voisard_2025_records,
    normalize_summary_record,
    resolve_adapter_spec,
)


def test_normalize_summary_record_accepts_common_aliases() -> None:
    row = {
        "speed_mps": "0.61",
        "cadence_spm": "103",
        "spatial_asymmetry": "0.18",
        "dynamic_stability": "0.22",
        "stride_time_cv_pct": "4.8",
        "temporal_asymmetry": "0.06",
    }
    summary = normalize_summary_record(row)
    assert summary == {
        "avg_speed": 0.61,
        "avg_cadence": 103.0,
        "avg_asymmetry": 0.18,
        "avg_stability": 0.22,
        "avg_stride_time_cv": 4.8,
        "avg_step_time_asymmetry": 0.06,
    }


def test_load_summary_json_records_supports_records_wrapper(tmp_path) -> None:
    path = tmp_path / "cohort.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "subject_id": "S-01",
                        "speed_mps": 0.73,
                        "cadence_spm": 108,
                        "spatial_asymmetry": 0.14,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    records = load_summary_json_records(path)
    assert len(records) == 1
    assert records[0]["metadata"]["subject_id"] == "S-01"
    assert records[0]["summary"]["avg_speed"] == 0.73


def test_load_summary_csv_records_extracts_metadata(tmp_path) -> None:
    path = tmp_path / "cohort.csv"
    path.write_text(
        "subject_id,cohort,speed_mps,cadence_spm,spatial_asymmetry\n"
        "P-01,stroke,0.51,98,0.22\n",
        encoding="utf-8",
    )
    records = load_summary_csv_records(path)
    assert len(records) == 1
    assert records[0]["metadata"]["subject_id"] == "P-01"
    assert records[0]["metadata"]["cohort"] == "stroke"
    assert records[0]["summary"]["avg_asymmetry"] == 0.22


def test_resolve_adapter_spec_merges_preset_and_mapping() -> None:
    spec = resolve_adapter_spec(
        preset_name="clinical_gait_parameters",
        mapping={
            "summary_fields": {"avg_speed": ["preferred_speed"]},
            "identity_fields": ["subject_id", "pathology"],
            "metadata_renames": {"pathology": "cohort"},
        },
    )
    assert tuple(spec["summary_fields"]["avg_speed"]) == ("preferred_speed",)
    metadata = build_adapter_metadata(
        {"subject_id": "A1", "pathology": "CVA"},
        fallback_name="row-1",
        identity_fields=spec["identity_fields"],
        metadata_renames=spec["metadata_renames"],
    )
    assert metadata["subject_id"] == "A1"
    assert metadata["cohort"] == "CVA"


def test_load_voisard_2025_records_builds_summary_from_meta_and_processed(tmp_path) -> None:
    trial_dir = tmp_path / "dataset" / "data" / "neuro" / "CVA" / "CVA_43" / "CVA_43_1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "CVA_43_1_meta.json").write_text(
        json.dumps(
            {
                "subject": "CVA_43",
                "group": "neuro",
                "pathology": "cerebrovascular accident",
                "pathologyKey": "CVA",
                "evaluationScoreName": "FMA-LE (/34)",
                "evaluationScoreValue": 30.0,
                "TUG": 16.6,
                "visualGaitAssessment": 3.0,
                "clinicalDeficitSide": "left",
                "session": 1,
                "trial": 1,
                "freq": 100.0,
                "uturnBoundaries": [1000, 1300],
                "leftGaitEvents": [[100, 150], [300, 350], [500, 550], [700, 750], [1400, 1450], [1600, 1650]],
                "rightGaitEvents": [[200, 250], [400, 450], [600, 650], [800, 850], [1500, 1550], [1700, 1750]],
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "CVA_43_1_processed_data.txt").write_text(
        "PacketCounter\tSignal\n0\t0\n1800\t1\n",
        encoding="utf-8",
    )

    records = load_voisard_2025_records(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["metadata"]["cohort"] == "CVA"
    assert record["metadata"]["label"] == "cerebrovascular accident"
    assert record["summary"]["avg_speed"] > 0
    assert record["summary"]["avg_cadence"] > 0
    assert record["summary"]["avg_stride_time_cv"] >= 0
    assert record["summary"]["tug_seconds"] == 16.6
    assert record["summary"]["turn_duration_seconds"] == 3.0
    assert record["summary"]["visual_gait_assessment"] == 3.0
