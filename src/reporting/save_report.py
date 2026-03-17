from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReportPaths:
    output_dir: str = "outputs"

    def daily_report_path(self, date_str: str) -> Path:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        return Path(self.output_dir) / f"market_report_{date_str}.txt"


def build_report_text(
    combined_table: pd.DataFrame,
    used_cache: dict[str, bool],
    title: str = "Market Email Bot Report",
) -> str:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Make table pretty + stable
    table_text = combined_table.to_string(index=False)

    lines: list[str] = []
    lines.append(title)
    lines.append(f"Run Time: {time_str}")
    lines.append("")
    lines.append("=== Market Snapshot + Forecast ===")
    lines.append(table_text)
    lines.append("")
    lines.append("=== Data Source ===")
    for t, was_cache in used_cache.items():
        lines.append(f"{t:>6}  ->  {'CACHE' if was_cache else 'FRESH'}")

    lines.append("")  # newline at end
    return "\n".join(lines)


def save_daily_report(
    combined_table: pd.DataFrame,
    used_cache: dict[str, bool],
    output_dir: str = "outputs",
) -> Path:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    paths = ReportPaths(output_dir=output_dir)
    out_path = paths.daily_report_path(date_str)

    report_text = build_report_text(combined_table, used_cache)
    out_path.write_text(report_text, encoding="utf-8")

    return out_path