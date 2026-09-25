"""
Dataset builder for structured log classification.

Produces a labeled dataset from raw log files suitable for training and
evaluating SLMs on anomaly detection. Supports both auto-labeling (heuristic)
and manual labeling workflows.

Output format (JSONL):
  {"id": int, "source_file": str, "timestamp": str, "level": str,
   "service": str, "template": str, "raw_text": str,
   "error_type": str | null, "is_error": bool, "label_source": str}
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Error taxonomy for university infrastructure context (DTIC - UCuenca)
#TODO Estos patrones son de ejemplo, por ahora no se tiene el patron de DTIC, se debe actualizar con los patrones reales.
# ---------------------------------------------------------------------------
ERROR_TAXONOMY = {
    "connection_error": {
        "description": "Network or service connection failures",
        "patterns": [
            r"\bconnection (?:refused|reset|timeout|timed out|closed|broken)\b",
            r"\bcannot connect to\b",
            r"\bconnection (?:pool|refused|reset)\b",
            r"\bunable to connect\b",
            r"\bconnection error\b",
            r"\bconnect_failed\b",
            r"\bETIMEDOUT\b",
            r"\bECONNREFUSED\b",
            r"\bECONNRESET\b",
        ],
    },
    "authentication_error": {
        "description": "Authentication and authorization failures",
        "patterns": [
            r"\bauthentication (?:failed|error|failed|denied|invalid|failure)\b",
            r"\bunauthorized\b",
            r"\baccess denied\b",
            r"\bforbidden\b",
            r"\bpermission denied\b",
            r"\binvalid credentials\b",
            r"\blogin (?:failed|error|attempt)\b",
            r"\b401\b",
            r"\b403\b",
        ],
    },
    "database_error": {
        "description": "Database query, connection, or integrity errors",
        "patterns": [
            r"\bdatabase (?:error|failed|connection|timeout|corrupt)\b",
            r"\bquery (?:failed|error|timeout)\b",
            r"\bsql (?:error|exception|syntax)\b",
            r"\bdeadlock\b",
            r"\brecord not found\b",
            r"\bdata integrity\b",
            r"\bout of memory\b.*\bdatabase\b",
        ],
    },
    "disk_space_error": {
        "description": "Disk space, I/O, and filesystem errors",
        "patterns": [
            r"\bdisk (?:full|space|error|failure)\b",
            r"\bout of disk\b",
            r"\bno space left\b",
            r"\bfilesystem (?:full|error|read-only)\b",
            r"\bI/O error\b",
            r"\binput/output error\b",
            r"\bwrite (?:failed|error)\b.*\bdisk\b",
            r"\bENOSPC\b",
        ],
    },
    "memory_error": {
        "description": "Memory exhaustion and leak errors",
        "patterns": [
            r"\bmemory (?:leak|error|exhausted|overflow|usage)\b",
            r"\bout of memory\b",
            r"\bmemory allocation (?:failed|error)\b",
            r"\bhugepages?\b.*\berror\b",
            r"\bvirtual memory\b.*\b exhausted\b",
        ],
    },
    "application_crash": {
        "description": "Application process crashes and fatal errors",
        "patterns": [
            r"\b(fatal|critical|abort(?:ed|ing)?)\b",
            r"\bsegmentation fault\b",
            r"\bsegfault\b",
            r"\bsigsegv\b",
            r"\bcore (?:dump|registered)\b",
            r"\bprocess (?:died|exited|terminated|killed|crashed)\b",
            r"\buncaught (?:exception|error|exception)\b",
            r"\bpanic\b",
        ],
    },
    "service_unavailable": {
        "description": "Service downtime and unavailability",
        "patterns": [
            r"\bservice (?:unavailable|down|offline|not responding)\b",
            r"\bservice (?:error|failed|exception)\b",
            r"\bupstream (?:connect|timeout|error)\b",
            r"\b502\b",
            r"\b503\b",
            r"\b504\b",
            r"\bhealth (?:check|check) (?:failed|error)\b",
            r"\brequest (?:timed? ?out|failed|error)\b",
        ],
    },
    "configuration_error": {
        "description": "Misconfiguration and invalid settings",
        "patterns": [
            r"\bconfiguration (?:error|invalid|failed|missing)\b",
            r"\bconfig (?:error|invalid|file not found)\b",
            r"\binvalid (?:config|setting|parameter|argument)\b",
            r"\bmisconfiguration\b",
            r"\bunexpected (?:config|setting|value)\b",
        ],
    },
    "timeout_error": {
        "description": "General timeout errors",
        "patterns": [
            r"\btimeout\b",
            r"\btimed out\b",
            r"\bexceeded (?:the )?timeout\b",
            r"\boperation (?:timed? ?out|aborted)\b",
        ],
    },
    " ssl_error": {
        "description": "SSL/TLS certificate and handshake errors",
        "patterns": [
            r"\bssl (?:error|certificate|handshake|verification)\b",
            r"\btls (?:error|handshake|alert)\b",
            r"\bcertificate (?:expired|invalid|verify)\b",
            r"\bssl(?:_)?certificate_error\b",
        ],
    },
}


def _match_error_type(raw_text: str) -> str | None:
    """Return the first matching error type from the taxonomy, or None."""
    text_lower = raw_text.lower()
    for error_type, info in ERROR_TAXONOMY.items():
        for pattern in info["patterns"]:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return error_type
    return None


def _extract_level(raw_text: str) -> str:
    """Extract log level (ERROR, WARN, INFO, DEBUG, etc.) from text."""
    m = re.search(r"\b(ERROR|WARN(?:ING)?|INFO|DEBUG|FATAL|CRITICAL|TRACE)\b", raw_text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def _extract_timestamp(raw_text: str) -> str | None:
    """Try to extract a timestamp from common log formats."""
    # Apache/Nginx: [22/Jan/2019:03:56:14 +0330]
    m = re.search(r"\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\]", raw_text)
    if m:
        return m.group(1)
    # Syslog: Jan 22 03:56:14
    m = re.search(r"^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})", raw_text)
    if m:
        return m.group(1)
    # ISO 8601: 2019-01-22T03:56:14
    m = re.search(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", raw_text)
    if m:
        return m.group(1)
    return None


def _has_error_indicators(raw_text: str) -> bool:
    """Heuristic: does this log line look like an error?"""
    text_lower = raw_text.lower()
    # Explicit error level
    if re.search(r"\b(ERROR|FATAL|CRITICAL|SEVERE)\b", text_lower):
        return True
    # HTTP error status codes (4xx, 5xx) in access logs
    if re.search(r'"\s+(4\d{2}|5\d{2})\s+', raw_text):
        return True
    # Known error patterns from taxonomy
    if _match_error_type(raw_text) is not None:
        return True
    return False


def parse_log_line(line: str, source_file: str, global_id: int) -> dict[str, Any] | None:
    """Parse a single log line into a structured record."""
    line = line.rstrip("\n").rstrip("\r")
    if not line.strip():
        return None

    return {
        "id": global_id,
        "source_file": source_file,
        "timestamp": _extract_timestamp(line) or "",
        "level": _extract_level(line),
        "service": "",
        "template": "",  # filled by template extractor if available
        "raw_text": line,
        "error_type": _match_error_type(line),
        "is_error": _has_error_indicators(line),
        "label_source": "auto_heuristic",
    }


def process_log_file(
    log_path: Path,
    output_path: Path,
    append: bool = False,
    global_id_start: int = 0,
) -> int:
    """Process a single log file and append structured records to output."""
    records_written = 0
    mode = "a" if append else "w"

    with open(log_path, "r", encoding="utf-8", errors="replace") as infile, \
         open(output_path, mode, encoding="utf-8") as outfile:

        for line in infile:
            record = parse_log_line(line, log_path.name, global_id_start + records_written)
            if record is None:
                continue
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

    return records_written


def build_dataset(
    log_dir: str | Path,
    output_path: str | Path,
    append: bool = False,
    file_glob: str = "*.log",
) -> dict[str, Any]:
    """
    Build a structured dataset from all log files in a directory.

    Returns statistics about the build.
    """
    log_dir = Path(log_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_files = sorted(log_dir.glob(file_glob))
    if not log_files:
        raise FileNotFoundError(f"No log files matching '{file_glob}' in {log_dir}")

    global_id = 0
    total_records = 0
    error_count = 0
    type_counts: dict[str, int] = {}

    for log_file in log_files:
        written = process_log_file(log_file, output_path, append=append, global_id_start=global_id)
        total_records += written
        global_id += written

    # Compute stats from the output file
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("is_error"):
                error_count += 1
                et = rec.get("error_type") or "unclassified_error"
                type_counts[et] = type_counts.get(et, 0) + 1

    return {
        "log_files_processed": len(log_files),
        "log_files": [f.name for f in log_files],
        "total_records": total_records,
        "error_records": error_count,
        "normal_records": total_records - error_count,
        "error_rate": error_count / total_records if total_records > 0 else 0,
        "error_type_distribution": type_counts,
    }


# ---------------------------------------------------------------------------
# Manual labeling helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL dataset into memory."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def update_label(
    dataset_path: str | Path,
    record_id: int,
    error_type: str | None,
    is_error: bool,
    label_source: str = "manual",
) -> bool:
    """Update a single record's labels in-place in the JSONL file."""
    path = Path(dataset_path)
    records = []
    found = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("id") == record_id:
                rec["error_type"] = error_type
                rec["is_error"] = is_error
                rec["label_source"] = label_source
                found = True
            records.append(rec)

    if not found:
        return False

    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def export_labeled_subset(
    dataset_path: str | Path,
    output_path: str | Path,
    only_errors: bool = False,
    error_types: list[str] | None = None,
) -> int:
    """Export a filtered subset of the dataset for focused analysis."""
    path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            rec = json.loads(line)
            if only_errors and not rec.get("is_error"):
                continue
            if error_types and rec.get("error_type") not in error_types:
                continue
            outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a structured log dataset for SLM evaluation")
    parser.add_argument("log_dir", type=Path, help="Directory containing .log files", default=Path("../logs"))
    parser.add_argument("output", type=Path, help="Output JSONL dataset path", default =Path("../dataset/log_dataset.jsonl"))
    parser.add_argument("--append", action="store_true", help="Append to existing dataset")
    parser.add_argument("--glob", default="*.log", help="File glob pattern (default: *.log)")
    args = parser.parse_args()

    stats = build_dataset(args.log_dir, args.output, append=args.append, file_glob=args.glob)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
