from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.webapp.stroke_gait_calibration import (
    build_calibration_record,
    build_calibration_report,
    load_result_payload,
)


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
    args = parser.parse_args(argv[1:])

    input_paths: list[Path] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        if path.is_dir():
            input_paths.extend(sorted(path.glob(args.glob)))
        else:
            input_paths.append(path)

    if not input_paths:
        print("No matching result files found.")
        return 1

    records = []
    for path in input_paths:
        payload = load_result_payload(path)
        records.append(build_calibration_record(path, payload))

    if args.report:
        print(json.dumps(build_calibration_report(records), ensure_ascii=False, indent=2))
        return 0

    for record in records:
        compact = {
            "file": record["file"],
            "pattern_level": record["pattern_level"],
            "pattern_score": record["pattern_score"],
            "speed_band": record.get("speed_band"),
            "flagged_count": record["flagged_count"],
            "summary_note": record["stroke_gait_analysis"]["summary_note"],
        }
        print(json.dumps(compact, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
