"""
Módulo 0: Log Prefilter (CLI)
=============================
Lee los logs crudos de `ms-users` (Logback / Spring Boot, ~15 GB en 309 archivos), agrupa las líneas
de continuación (stack traces) con su evento padre y descarta el ruido de infraestructura.

El parseo y las reglas viven en `log_analysis.core.log_filter` (clase `LogbackEventFilter`), la misma
pieza que usa el pipeline con `--log-filter logback`; este módulo aporta la CLI y la escritura a
JSONL. `iter_events`, `classify` y `prefilter` se siguen exportando desde aquí.

Salida (JSONL, un evento por línea):
{
    "timestamp": "2026-09-23T00:31:22.382-05:00", "level": "ERROR", "pid": "9",
    "thread": "Thread-1698", "logger": "e.e.u.m.i.a.o.e.a.UsuarioAs400Service",
    "message": "User not found ...", "stack_trace": "", "raw_line": "...",
    "source_file": "logFile....log"
}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator

# permite `python pseudo_labeling/<script>.py` desde scripts/ (sys.path[0] sería pseudo_labeling/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from log_analysis.core.log_filter import (
    DEFAULT_MAX_TRACE_LINES,
    LogbackEventFilter,
    classify,
    format_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("log_prefilter")


def iter_events(path: Path, max_trace_lines: int = DEFAULT_MAX_TRACE_LINES) -> Iterator[dict]:
    """Todos los eventos de un archivo, sin aplicar reglas."""
    return LogbackEventFilter(max_trace_lines).iter_file(path)


def prefilter(
    input_dir: Path,
    output_path: Path,
    max_trace_lines: int = DEFAULT_MAX_TRACE_LINES,
    limit_files: int | None = None,
    pattern: str = "*.log",
    per_file: bool = False,
) -> dict:
    """Filtra los `*.log` de un directorio a un JSONL (o uno por archivo con `per_file`)."""
    files = sorted(input_dir.glob(pattern))
    if limit_files:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"No hay archivos '{pattern}' en {input_dir}")

    # per_file: `output_path` es un directorio y se escribe un .jsonl por log de entrada
    (output_path if per_file else output_path.parent).mkdir(parents=True, exist_ok=True)
    event_filter = LogbackEventFilter(max_trace_lines)
    kept = 0

    single = None if per_file else open(output_path, "w", encoding="utf-8")
    try:
        for i, path in enumerate(files, start=1):
            file_kept = 0
            out = single or open(output_path / f"{path.stem}.jsonl", "w", encoding="utf-8")
            try:
                for event in event_filter.iter_file_kept(path):
                    out.write(json.dumps(event, ensure_ascii=False) + "\n")
                    kept += 1
                    file_kept += 1
            finally:
                if single is None:
                    out.close()
            logger.info("[%d/%d] %s → %d eventos conservados", i, len(files), path.name, file_kept)
    finally:
        if single is not None:
            single.close()

    stats = event_filter.stats.as_dict()
    assert stats["events_kept"] == kept
    stats["files"] = len(files)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Módulo 0 — Prefiltrado de logs ms-users")
    parser.add_argument("--input-dir", default="logs/ms-users.ucuenca-logs")
    parser.add_argument("--output", default="pseudo_labeling/data/filtered_events.jsonl")
    parser.add_argument("--report", default="pseudo_labeling/data/prefilter_report.txt")
    parser.add_argument("--max-trace-lines", type=int, default=DEFAULT_MAX_TRACE_LINES,
                        help="Máximo de líneas de stack trace por evento (0 = sin límite)")
    parser.add_argument("--limit-files", type=int, default=None,
                        help="Procesar solo los primeros N archivos (pruebas)")
    parser.add_argument("--per-file", action="store_true",
                        help="Escribir un .jsonl por log de entrada; --output pasa a ser un directorio")
    args = parser.parse_args()

    stats = prefilter(Path(args.input_dir), Path(args.output), args.max_trace_lines,
                      args.limit_files, per_file=args.per_file)
    report = format_report(stats)
    Path(args.report).write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
