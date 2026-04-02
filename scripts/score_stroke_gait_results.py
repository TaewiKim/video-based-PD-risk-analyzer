from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse
from datetime import UTC, datetime


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.webapp.stroke_gait_calibration import (
    build_calibration_record,
    build_calibration_record_from_summary,
    build_group_comparison,
    build_calibration_report,
    filter_calibration_records,
    load_result_payload,
)
from server.webapp.stroke_gait_dataset_adapters import (
    load_summary_csv_records,
    load_summary_json_records,
    load_voisard_2025_records,
    load_adapter_mapping,
    resolve_adapter_spec,
)


def _resolve_input_paths(paths: list[str], glob_pattern: str, *, keep_dirs: bool = False) -> list[Path]:
    input_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            if keep_dirs or (path / "dataset" / "data").exists() or (path / "data").exists():
                input_paths.append(path)
            else:
                input_paths.extend(sorted(path.glob(glob_pattern)))
        else:
            input_paths.append(path)
    return input_paths


def _load_records(
    path: Path,
    input_format: str,
    *,
    adapter_spec: dict[str, object],
) -> list[dict[str, object]]:
    if input_format == "platform_json":
        payload = load_result_payload(path)
        return [build_calibration_record(path, payload)]
    if input_format == "summary_json":
        summaries = load_summary_json_records(path, adapter_spec=adapter_spec)
        return [
            build_calibration_record_from_summary(
                item["metadata"]["file"],
                item["summary"],
                metadata=item["metadata"],
            )
            for item in summaries
        ]
    if input_format == "summary_csv":
        summaries = load_summary_csv_records(path, adapter_spec=adapter_spec)
        return [
            build_calibration_record_from_summary(
                item["metadata"]["file"],
                item["summary"],
                metadata=item["metadata"],
            )
            for item in summaries
        ]
    if input_format == "voisard_2025":
        summaries = load_voisard_2025_records(path)
        return [
            build_calibration_record_from_summary(
                item["metadata"]["file"],
                item["summary"],
                metadata=item["metadata"],
            )
            for item in summaries
        ]
    if path.suffix.lower() == ".csv":
        return _load_records(path, "summary_csv", adapter_spec=adapter_spec)
    if path.is_dir() and ((path / "dataset" / "data").exists() or (path / "data").exists()):
        return _load_records(path, "voisard_2025", adapter_spec=adapter_spec)
    payload = load_result_payload(path)
    if isinstance(payload, dict) and ("summary" in payload or "gait_analysis" in payload):
        return [build_calibration_record(path, payload)]
    return _load_records(path, "summary_json", adapter_spec=adapter_spec)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Score stroke-linked gait analysis outputs from stored result JSON files."
    )
    parser.add_argument("paths", nargs="+", help="Result JSON file paths or directories")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print an aggregate calibration report instead of one line per file.",
    )
    parser.add_argument(
        "--glob",
        default="*_results.json",
        help="Glob pattern used when a directory is provided. Default: *_results.json",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the JSON report or line-delimited JSON output.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "platform_json", "summary_json", "summary_csv", "voisard_2025"),
        default="auto",
        help="How to interpret each input path. Default: auto",
    )
    parser.add_argument(
        "--adapter-preset",
        default="default",
        help="Summary adapter preset name for external datasets. Default: default",
    )
    parser.add_argument(
        "--mapping-file",
        help="Optional JSON file that extends or overrides the selected adapter preset.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        help="Keep only records whose metadata.cohort matches this value. Repeatable.",
    )
    parser.add_argument(
        "--exclude-cohort",
        action="append",
        help="Drop records whose metadata.cohort matches this value. Repeatable.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Keep only records whose metadata.label matches this value. Repeatable.",
    )
    parser.add_argument(
        "--compare-groups",
        nargs=2,
        metavar=("GROUP_A", "GROUP_B"),
        help="When reporting, add a two-group comparison using metadata.cohort by default.",
    )
    parser.add_argument(
        "--compare-field",
        default="cohort",
        help="Metadata field used for --compare-groups. Default: cohort",
    )
    args = parser.parse_args(argv[1:])

    keep_dirs = args.input_format == "voisard_2025"
    input_paths = _resolve_input_paths(args.paths, args.glob, keep_dirs=keep_dirs)

    if not input_paths:
        print("No matching result files found.")
        return 1

    mapping = load_adapter_mapping(args.mapping_file) if args.mapping_file else None
    adapter_spec = resolve_adapter_spec(preset_name=args.adapter_preset, mapping=mapping)

    records = []
    for path in input_paths:
        records.extend(_load_records(path, args.input_format, adapter_spec=adapter_spec))

    records = filter_calibration_records(
        records,
        cohort_in=set(args.cohort or []) or None,
        cohort_exclude=set(args.exclude_cohort or []) or None,
        label_in=set(args.label or []) or None,
    )

    if not records:
        print("No usable records found after applying the selected adapter.")
        return 1

    if args.report:
        report = build_calibration_report(records)
        report["generated_at"] = datetime.now(UTC).isoformat()
        report["source_paths"] = [str(path) for path in input_paths]
        report["input_format"] = args.input_format
        report["adapter_preset"] = args.adapter_preset
        if args.mapping_file:
            report["mapping_file"] = str(Path(args.mapping_file))
        if args.cohort:
            report["cohort_filter"] = args.cohort
        if args.exclude_cohort:
            report["exclude_cohort_filter"] = args.exclude_cohort
        if args.label:
            report["label_filter"] = args.label
        if args.compare_groups:
            comparison = build_group_comparison(
                records,
                group_field=args.compare_field,
                groups=(args.compare_groups[0], args.compare_groups[1]),
            )
            if comparison:
                report["group_comparison"] = comparison
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0

    rendered_lines = []
    for record in records:
        compact = {
            "file": record["file"],
            "pattern_level": record["pattern_level"],
            "pattern_score": record["pattern_score"],
            "speed_band": record.get("speed_band"),
            "flagged_count": record["flagged_count"],
            "summary_note": record["stroke_gait_analysis"]["summary_note"],
        }
        rendered = json.dumps(compact, ensure_ascii=False)
        rendered_lines.append(rendered)
        print(rendered)
    if args.output:
        Path(args.output).write_text("\n".join(rendered_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
