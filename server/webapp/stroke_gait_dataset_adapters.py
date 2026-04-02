from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

DEFAULT_SUMMARY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "avg_speed": ("avg_speed", "speed", "gait_speed", "walking_speed", "speed_mps"),
    "avg_cadence": ("avg_cadence", "cadence", "step_cadence", "cadence_spm"),
    "avg_asymmetry": ("avg_asymmetry", "asymmetry", "gait_asymmetry", "spatial_asymmetry"),
    "avg_stability": ("avg_stability", "stability", "gait_stability", "dynamic_stability"),
    "avg_stride_time_cv": (
        "avg_stride_time_cv",
        "stride_time_cv",
        "stride_time_cv_pct",
        "temporal_variability",
    ),
    "avg_step_time_asymmetry": (
        "avg_step_time_asymmetry",
        "step_time_asymmetry",
        "temporal_asymmetry",
        "step_timing_asymmetry",
    ),
}

IDENTITY_FIELDS = ("file", "record_id", "subject_id", "participant_id", "label", "cohort")

ADAPTER_PRESETS: dict[str, dict[str, Any]] = {
    "default": {
        "summary_fields": DEFAULT_SUMMARY_FIELD_ALIASES,
        "identity_fields": IDENTITY_FIELDS,
    },
    "clinical_gait_parameters": {
        "summary_fields": {
            "avg_speed": ("Speed", "speed", "walking_speed", "speed_mps"),
            "avg_cadence": ("Cadence", "cadence", "cadence_spm"),
            "avg_asymmetry": ("Asymmetry", "spatial_asymmetry", "step_length_asymmetry", "gait_asymmetry"),
            "avg_stability": ("Stability", "dynamic_stability", "gait_stability"),
            "avg_stride_time_cv": ("StrideTimeCV", "stride_time_cv", "stride_time_cv_pct"),
            "avg_step_time_asymmetry": ("StepTimeAsymmetry", "step_time_asymmetry", "temporal_asymmetry"),
        },
        "identity_fields": ("record_id", "subject_id", "participant_id", "pathology", "label", "cohort", "visit"),
        "metadata_renames": {"pathology": "cohort", "visit": "visit"},
    },
}


def _coerce_float(value: Any) -> float | None:
    if value in (None, "", "NA", "N/A", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_adapter_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Adapter mapping file must contain a JSON object.")
    return payload


def resolve_adapter_spec(
    *,
    preset_name: str = "default",
    mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preset = ADAPTER_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(f"Unknown adapter preset: {preset_name}")
    spec = {
        "summary_fields": dict(preset.get("summary_fields", {})),
        "identity_fields": tuple(preset.get("identity_fields", IDENTITY_FIELDS)),
        "metadata_renames": dict(preset.get("metadata_renames", {})),
    }
    if mapping:
        if "summary_fields" in mapping:
            spec["summary_fields"].update(mapping["summary_fields"])
        if "identity_fields" in mapping:
            spec["identity_fields"] = tuple(mapping["identity_fields"])
        if "metadata_renames" in mapping:
            spec["metadata_renames"].update(mapping["metadata_renames"])
    return spec


def normalize_summary_record(
    row: dict[str, Any],
    *,
    field_aliases: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    aliases = field_aliases or DEFAULT_SUMMARY_FIELD_ALIASES
    summary: dict[str, Any] = {}
    for canonical_field, field_names in aliases.items():
        for alias in field_names:
            if alias not in row:
                continue
            value = _coerce_float(row.get(alias))
            if value is not None:
                summary[canonical_field] = value
                break
    return summary


def build_adapter_metadata(
    row: dict[str, Any],
    *,
    fallback_name: str,
    identity_fields: tuple[str, ...] | None = None,
    metadata_renames: dict[str, str] | None = None,
) -> dict[str, Any]:
    fields = identity_fields or IDENTITY_FIELDS
    renames = metadata_renames or {}
    metadata = {"file": str(row.get("file") or fallback_name)}
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            metadata[renames.get(field, field)] = value
    return metadata


def _iter_summary_json_records(
    path: str | Path,
    payload: Any,
    *,
    adapter_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    source_name = Path(path).name
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            records = []
            for index, row in enumerate(payload["records"]):
                if not isinstance(row, dict):
                    continue
                records.append(
                    {
                        "summary": normalize_summary_record(row, field_aliases=adapter_spec["summary_fields"]),
                        "metadata": build_adapter_metadata(
                            row,
                            fallback_name=f"{source_name}#{index + 1}",
                            identity_fields=adapter_spec["identity_fields"],
                            metadata_renames=adapter_spec["metadata_renames"],
                        ),
                    }
                )
            return records
        return [
            {
                "summary": normalize_summary_record(payload, field_aliases=adapter_spec["summary_fields"]),
                "metadata": build_adapter_metadata(
                    payload,
                    fallback_name=source_name,
                    identity_fields=adapter_spec["identity_fields"],
                    metadata_renames=adapter_spec["metadata_renames"],
                ),
            }
        ]
    if isinstance(payload, list):
        records = []
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            records.append(
                {
                    "summary": normalize_summary_record(row, field_aliases=adapter_spec["summary_fields"]),
                    "metadata": build_adapter_metadata(
                        row,
                        fallback_name=f"{source_name}#{index + 1}",
                        identity_fields=adapter_spec["identity_fields"],
                        metadata_renames=adapter_spec["metadata_renames"],
                    ),
                }
            )
        return records
    return []


def load_summary_json_records(
    path: str | Path,
    *,
    adapter_spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _iter_summary_json_records(path, payload, adapter_spec=adapter_spec or resolve_adapter_spec())


def load_summary_csv_records(
    path: str | Path,
    *,
    adapter_spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    spec = adapter_spec or resolve_adapter_spec()
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            summary = normalize_summary_record(row, field_aliases=spec["summary_fields"])
            metadata = build_adapter_metadata(
                row,
                fallback_name=f"{Path(path).name}#{index + 1}",
                identity_fields=spec["identity_fields"],
                metadata_renames=spec["metadata_renames"],
            )
            records.append({"summary": summary, "metadata": metadata})
    return records


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _extract_heel_strikes(events: list[Any], *, freq: float) -> list[float]:
    heel_strikes: list[float] = []
    for event in events:
        if isinstance(event, (list, tuple)) and len(event) >= 2:
            heel_strikes.append(float(event[1]) / freq)
        elif isinstance(event, (int, float)):
            heel_strikes.append(float(event) / freq)
    return heel_strikes


def _stride_durations(times: list[float]) -> list[float]:
    if len(times) < 2:
        return []
    return [times[index + 1] - times[index] for index in range(len(times) - 1) if times[index + 1] > times[index]]


def _read_last_packet_counter(processed_path: Path) -> float | None:
    if not processed_path.exists():
        return None
    with processed_path.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        block = 4096
        data = b""
        while end > 0:
            read_size = min(block, end)
            end -= read_size
            handle.seek(end)
            data = handle.read(read_size) + data
            lines = data.splitlines()
            if len(lines) >= 2:
                last_line = lines[-1].decode("utf-8", errors="ignore").strip()
                if last_line:
                    try:
                        return float(last_line.split("\t", 1)[0])
                    except ValueError:
                        return None
        return None


def _resolve_voisard_data_root(path: str | Path) -> Path:
    root = Path(path)
    if (root / "dataset" / "data").exists():
        return root / "dataset" / "data"
    if (root / "data").exists():
        return root / "data"
    raise ValueError(f"Voisard 2025 data root not found under: {root}")


def _build_voisard_summary(meta: dict[str, Any], processed_path: Path) -> dict[str, Any]:
    freq = float(meta.get("freq") or 100.0)
    left_hs = _extract_heel_strikes(list(meta.get("leftGaitEvents") or []), freq=freq)
    right_hs = _extract_heel_strikes(list(meta.get("rightGaitEvents") or []), freq=freq)
    left_stride = _stride_durations(left_hs)
    right_stride = _stride_durations(right_hs)
    all_stride = left_stride + right_stride

    packet_counter = _read_last_packet_counter(processed_path)
    total_duration = (packet_counter / freq) if packet_counter is not None else None
    uturn = meta.get("uturnBoundaries") or []
    uturn_duration = 0.0
    if isinstance(uturn, list) and len(uturn) >= 2:
        uturn_duration = max(0.0, (float(uturn[1]) - float(uturn[0])) / freq)

    walking_duration = None
    if total_duration is not None:
        walking_duration = max(total_duration - uturn_duration, 0.0)
    if not walking_duration:
        tug = _coerce_float(meta.get("TUG"))
        if tug:
            walking_duration = max(tug, 0.0)

    step_count = len(left_hs) + len(right_hs)
    mean_left_stride = mean(left_stride) if left_stride else None
    mean_right_stride = mean(right_stride) if right_stride else None
    mean_stride = mean(all_stride) if all_stride else None
    stride_time_cv = (pstdev(all_stride) / mean_stride * 100.0) if all_stride and mean_stride else 0.0

    asymmetry = 0.0
    if mean_left_stride and mean_right_stride:
        asymmetry = abs(mean_left_stride - mean_right_stride) / max(mean_left_stride, mean_right_stride)

    cadence = (step_count / walking_duration * 60.0) if walking_duration and step_count else 0.0
    speed = (20.0 / walking_duration) if walking_duration else 0.0

    tug = _coerce_float(meta.get("TUG")) or 0.0
    visual = _coerce_float(meta.get("visualGaitAssessment")) or 0.0
    variability_penalty = min(0.16, stride_time_cv / 50.0)
    asymmetry_penalty = min(0.12, asymmetry * 0.6)
    tug_penalty = min(0.08, max(tug - 12.0, 0.0) / 50.0)
    visual_penalty = min(0.06, visual / 50.0)
    stability = _clip(0.36 - variability_penalty - asymmetry_penalty - tug_penalty - visual_penalty, 0.02, 0.4)

    return {
        "avg_speed": round(speed, 3),
        "avg_cadence": round(cadence, 1),
        "avg_asymmetry": round(asymmetry, 3),
        "avg_stability": round(stability, 3),
        "avg_stride_time_cv": round(stride_time_cv, 2),
        "avg_step_time_asymmetry": round(asymmetry, 3),
        "tug_seconds": round(tug, 2),
        "turn_duration_seconds": round(uturn_duration, 2),
        "visual_gait_assessment": round(visual, 2),
    }


def _build_voisard_metadata(meta_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": str(meta_path),
        "subject_id": str(meta.get("subject") or meta_path.parent.parent.name),
        "record_id": meta_path.stem.removesuffix("_meta"),
        "cohort": str(meta.get("pathologyKey") or meta.get("group") or "unknown"),
        "label": str(meta.get("pathology") or meta.get("pathologyKey") or "unknown"),
        "pathology": meta.get("pathology"),
        "evaluation_score_name": meta.get("evaluationScoreName"),
        "evaluation_score_value": meta.get("evaluationScoreValue"),
        "tug_seconds": meta.get("TUG"),
        "visual_gait_assessment": meta.get("visualGaitAssessment"),
        "clinical_deficit_side": meta.get("clinicalDeficitSide"),
        "session": meta.get("session"),
        "trial": meta.get("trial"),
    }


def load_voisard_2025_records(
    path: str | Path,
    *,
    cohort_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    data_root = _resolve_voisard_data_root(path)
    records: list[dict[str, Any]] = []
    search_roots: list[Path]
    if cohort_filter:
        search_roots = []
        for cohort in sorted(cohort_filter):
            cohort_path = data_root / ("healthy" if cohort == "HS" else "neuro" if cohort in {"CVA", "PD", "RIL", "CIPN"} else "ortho") / cohort
            if cohort_path.exists():
                search_roots.append(cohort_path)
    else:
        search_roots = [data_root]

    for search_root in search_roots:
        meta_paths = sorted(search_root.rglob("*_meta.json"))
        for meta_path in meta_paths:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            processed_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_processed_data.txt"))
            summary = _build_voisard_summary(meta, processed_path)
            metadata = _build_voisard_metadata(meta_path, meta)
            records.append({"summary": summary, "metadata": metadata})
    return records
