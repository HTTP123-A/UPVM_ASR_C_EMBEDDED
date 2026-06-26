#!/usr/bin/env python3
"""Summarize timing CSV columns as mean +- std and write a sibling log file.

Usage:
    python3 timing_summarize.py

Edit INPUT_CSV_PATH below to point to your timing CSV.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

# ============================================================================
# User config: set the input CSV path here.
# ============================================================================
INPUT_CSV_PATH = (
    "/home/phuong/QNAP/Projects/AuAx/Model/UPVM-ASR-FOR-C/"
    "results/sq_int8_v9/inner_8/detail_stage_timing.csv"
)

# Columns to skip from averaging/std reporting.
_METADATA_COLUMNS = {"timing_index", "file_name", "segment_index"}


def _safe_float(value: str) -> float | None:
    """Convert string to finite float, otherwise return None."""
    text = value.strip()
    if not text:
        return None

    try:
        parsed = float(text)
    except ValueError:
        return None

    if not math.isfinite(parsed):
        return None

    return parsed


def _compute_mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def _compute_population_std(values: list[float], mean_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((v - mean_value) ** 2 for v in values) / float(len(values))
    return math.sqrt(variance)


def _build_output_log_path(csv_path: Path) -> Path:
    # "detail_stage_timing.csv" -> "detail_stage_timing_summ.log"
    return csv_path.with_name(f"{csv_path.stem}_summ.log")


def summarize_timing_csv(input_csv_path: str) -> Path:
    csv_path = Path(input_csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        ordered_columns = [
            col for col in reader.fieldnames if col not in _METADATA_COLUMNS
        ]
        values_by_col: dict[str, list[float]] = {col: [] for col in ordered_columns}

        row_count = 0
        for row in reader:
            row_count += 1
            for col in ordered_columns:
                raw = row.get(col, "")
                parsed = _safe_float(raw)
                if parsed is not None:
                    values_by_col[col].append(parsed)

    summary_lines: list[str] = []
    summary_lines.append("Timing Summary")
    summary_lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    summary_lines.append(f"Input CSV: {csv_path}")
    summary_lines.append(f"Rows read: {row_count}")
    summary_lines.append("Std: population standard deviation (ddof=0)")
    summary_lines.append("")

    for col in ordered_columns:
        vals = values_by_col[col]
        if not vals:
            summary_lines.append(f"- {col}: no valid numeric samples")
            continue

        mean_val = _compute_mean(vals)
        std_val = _compute_population_std(vals, mean_val)
        summary_lines.append(
            f"- {col}: {mean_val:.6f} +- {std_val:.6f} (n={len(vals)})"
        )

    output_log_path = _build_output_log_path(csv_path)
    output_log_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return output_log_path


def main() -> None:
    out_path = summarize_timing_csv(INPUT_CSV_PATH)
    print(f"Summary written to: {out_path}")


if __name__ == "__main__":
    main()
