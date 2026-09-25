"""
Dataset analyzer for the parsed JSON output of parse_and_label.py.

Reads a processed dataset (list of parsed log records) and prints structured
statistics: total count, error rate, category distribution, log level distribution,
template diversity, severity breakdown, and per-service stats.

Usage:
    python analyze_parsed_dataset.py dataset_slm_procesado_web.json
    python analyze_parsed_dataset.py dataset_slm_procesado_web.json --output analysis/dataset_stats.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Any


def load_dataset(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Load a parsed dataset JSON file."""
    path = pathlib.Path(path)
    if not path.exists():
        print(f"[error] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"[error] Expected a JSON array, got {type(data).__name__}", file=sys.stderr)
        sys.exit(1)
    return data


def analyze_dataset(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute statistics over a parsed dataset."""
    total = len(data)
    if total == 0:
        return {"total_records": 0, "warning": "Dataset is empty"}

    errors = 0
    categories: dict[str, int] = collections.Counter()
    levels: dict[str, int] = collections.Counter()
    severities: dict[str, int] = collections.Counter()
    services: dict[str, int] = collections.Counter()
    templates: set[str] = set()
    templates_per_category: dict[str, set[str]] = collections.defaultdict(set)
    raw_lengths: list[int] = []
    template_lengths: list[int] = []

    for record in data:
        # Ground truth
        gt = record.get("ground_truth_label", {})
        is_error = gt.get("is_error", False)
        category = gt.get("error_category", "UNKNOWN")
        severity = gt.get("severity", "NONE")

        if is_error:
            errors += 1
        categories[category] += 1
        severities[severity] += 1

        # Log data
        ld = record.get("log_data", {})
        level = ld.get("level", "UNKNOWN")
        template = ld.get("template", "")
        raw = ld.get("raw_message", "")

        levels[level] += 1
        templates.add(template)
        templates_per_category[category].add(template)
        raw_lengths.append(len(raw))
        template_lengths.append(len(template))

        # Service
        svc = record.get("service", {}).get("name", "unknown")
        services[svc] += 1

    # Error rate
    error_rate = errors / total if total > 0 else 0.0

    # Build category detail (with template counts)
    category_detail = {}
    for cat, count in categories.most_common():
        category_detail[cat] = {
            "count": count,
            "percentage": round(count / total * 100, 2),
            "unique_templates": len(templates_per_category.get(cat, set())),
        }

    return {
        "total_records": total,
        "error_records": errors,
        "normal_records": total - errors,
        "error_rate": round(error_rate, 4),
        "error_rate_percentage": round(error_rate * 100, 2),
        "log_level_distribution": dict(levels.most_common()),
        "error_category_distribution": category_detail,
        "severity_distribution": dict(severities.most_common()),
        "service_distribution": dict(services.most_common()),
        "unique_templates": len(templates),
        "template_ratio": round(len(templates) / total, 4) if total > 0 else 0,
        "avg_raw_length": round(sum(raw_lengths) / len(raw_lengths), 1) if raw_lengths else 0,
        "avg_template_length": round(sum(template_lengths) / len(template_lengths), 1) if template_lengths else 0,
        "max_raw_length": max(raw_lengths) if raw_lengths else 0,
        "max_template_length": max(template_lengths) if template_lengths else 0,
    }


def print_report(stats: dict[str, Any]) -> None:
    """Print a formatted human-readable report."""
    if stats.get("total_records", 0) == 0:
        print("Dataset is empty.")
        return

    total = stats["total_records"]
    errors = stats["error_records"]

    print("=" * 62)
    print("  PARSED DATASET ANALYSIS REPORT")
    print("=" * 62)
    print(f"  Total records:        {total}")
    print(f"  Error records:        {errors}  ({stats['error_rate_percentage']}%)")
    print(f"  Normal records:       {stats['normal_records']}  ({100 - stats['error_rate_percentage']:.2f}%)")
    print("-" * 62)
    print("  LOG LEVEL DISTRIBUTION")
    print("-" * 62)
    for level, count in stats["log_level_distribution"].items():
        bar = "█" * int(count / total * 40)
        print(f"  {level:<12} {count:>6}  {bar}  {count/total*100:5.1f}%")
    print("-" * 62)
    print("  ERROR CATEGORY DISTRIBUTION")
    print("-" * 62)
    for cat, info in stats["error_category_distribution"].items():
        bar = "█" * int(info["count"] / total * 40)
        print(f"  {cat:<30} {info['count']:>6}  {bar}  {info['percentage']:5.1f}%  "
              f"({info['unique_templates']} unique templates)")
    print("-" * 62)
    print("  SEVERITY DISTRIBUTION")
    print("-" * 62)
    for sev, count in stats["severity_distribution"].items():
        print(f"  {sev:<12} {count:>6}  ({count/total*100:5.1f}%)")
    print("-" * 62)
    print("  SERVICE DISTRIBUTION")
    print("-" * 62)
    for svc, count in stats["service_distribution"].items():
        print(f"  {svc:<25} {count:>6}  ({count/total*100:5.1f}%)")
    print("-" * 62)
    print("  TEMPLATE STATISTICS")
    print("-" * 62)
    print(f"  Unique templates:     {stats['unique_templates']}")
    print(f"  Template ratio:       {stats['template_ratio']:.4f}  (unique / total)")
    print(f"  Avg raw length:       {stats['avg_raw_length']} chars")
    print(f"  Avg template length:  {stats['avg_template_length']} chars")
    print(f"  Max raw length:       {stats['max_raw_length']} chars")
    print(f"  Max template length:  {stats['max_template_length']} chars")
    print("=" * 62)


def export_json(stats: dict[str, Any], output_path: str | pathlib.Path) -> None:
    """Export stats as a JSON file."""
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a parsed log dataset (output of parse_and_label.py)"
    )
    parser.add_argument(
        "dataset",
        type=pathlib.Path,
        help="Path to the parsed dataset JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        type=pathlib.Path,
        default=None,
        help="Optional JSON file to export stats to",
    )
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    stats = analyze_dataset(data)
    print_report(stats)

    if args.output:
        export_json(stats, args.output)


if __name__ == "__main__":
    main()
