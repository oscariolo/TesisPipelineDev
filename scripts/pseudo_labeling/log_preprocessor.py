"""
Módulo 1: Log Preprocessor
==========================
Toma los eventos Logback ya filtrados (salida del Módulo 0, `filtered_events.jsonl`) o,
con `--from-raw`, los logs crudos de un directorio, asigna un `log_id` único incremental
y exporta un JSONL estandarizado para el Módulo 2.

Output por registro:
{
    "log_id": "LOG-000001",
    "timestamp": "2026-09-23T00:31:22.382-05:00",
    "level": "ERROR",
    "thread": "Thread-1698",
    "logger": "e.e.u.m.i.a.o.e.a.UsuarioAs400Service",
    "message": "User not found with personal identification as400: 0150369817",
    "stack_trace": "",
    "source_file": "logFile.2026-09-23.0.log",
    "raw_text": "<línea original + stack trace>"
}
No se colapsan duplicados: cada evento conservado recibe su propio `log_id`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

# permite `python pseudo_labeling/<script>.py` desde scripts/ (sys.path[0] sería pseudo_labeling/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from log_analysis.core.log_filter import DEFAULT_MAX_TRACE_LINES, LogbackEventFilter, render

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("log_preprocessor")


class ProcessedLogEntry(BaseModel):
    """Esquema de un log procesado listo para ser enviado a los modelos."""

    log_id: str = Field(..., description="Identificador único incremental: LOG-XXXXXX")
    timestamp: str = Field("", description="Timestamp original del evento")
    level: str = Field("", description="TRACE/DEBUG/INFO/WARN/ERROR/FATAL")
    thread: str = Field("", description="Hilo que emitió el evento")
    logger: str = Field("", description="Logger (nombre abreviado por Logback)")
    message: str = Field("", description="Mensaje del evento")
    stack_trace: str = Field("", description="Stack trace (truncado) asociado al evento")
    source_file: str = Field("", description="Archivo de log de origen")
    raw_text: str = Field(..., description="Línea original + stack trace, sin modificar")


def _iter_filtered(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Línea JSON inválida en %s, omitida.", path.name)


def _iter_raw(input_dir: Path, max_trace_lines: int, limit_files: int | None) -> Iterator[dict]:
    files = sorted(input_dir.glob("*.log"))
    if limit_files:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"No hay archivos *.log en {input_dir}")
    event_filter = LogbackEventFilter(max_trace_lines)
    for path in files:
        yield from event_filter.iter_file_kept(path)


def preprocess_logs(
    input_path: str | Path,
    output_path: str | Path,
    id_prefix: str = "LOG",
    id_width: int = 6,
    from_raw: bool = False,
    max_trace_lines: int = DEFAULT_MAX_TRACE_LINES,
    limit_files: int | None = None,
) -> int:
    """
    Asigna `log_id` a cada evento y exporta a JSONL.

    Args:
        input_path:      JSONL filtrado (Módulo 0) o, con `from_raw`, directorio de *.log.
        output_path:     Ruta de salida para `processed_logs.jsonl`.
        id_prefix/width: Formato del id (default LOG-000001).
        from_raw:        Filtrar directamente los logs crudos (equivale a correr el Módulo 0 en memoria).

    Returns:
        Total de registros escritos.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe {input_path}. Ejecuta primero el Módulo 0 (log_prefilter.py) o usa --from-raw."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events = _iter_raw(input_path, max_trace_lines, limit_files) if from_raw else _iter_filtered(input_path)
    logger.info(" Leyendo eventos desde: %s", input_path)

    written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for written, event in enumerate(events, start=1):
            entry = ProcessedLogEntry(
                log_id=f"{id_prefix}-{written:0{id_width}d}",
                timestamp=event.get("timestamp", ""),
                level=event.get("level", ""),
                thread=event.get("thread", ""),
                logger=event.get("logger", ""),
                message=event.get("message", ""),
                stack_trace=event.get("stack_trace", ""),
                source_file=event.get("source_file", ""),
                raw_text=render(event),
            )
            out_f.write(entry.model_dump_json() + "\n")
            if written % 10000 == 0:
                logger.info("  → Procesados: %d eventos...", written)

    logger.info(" Preprocesamiento completado: %d registros → %s", written, output_path)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Módulo 1 — Asignación de log_id a los eventos Logback filtrados"
    )
    parser.add_argument("--input", default="pseudo_labeling/data/filtered_events.jsonl",
                        help="JSONL del Módulo 0 (o directorio de *.log con --from-raw)")
    parser.add_argument("--output", default="pseudo_labeling/data/ms_users/processed_logs.jsonl")
    parser.add_argument("--from-raw", action="store_true",
                        help="Filtrar directamente los .log crudos (input = directorio)")
    parser.add_argument("--max-trace-lines", type=int, default=DEFAULT_MAX_TRACE_LINES)
    parser.add_argument("--limit-files", type=int, default=None,
                        help="Solo con --from-raw: primeros N archivos")
    args = parser.parse_args()

    total = preprocess_logs(
        input_path=args.input,
        output_path=args.output,
        from_raw=args.from_raw,
        max_trace_lines=args.max_trace_lines,
        limit_files=args.limit_files,
    )
    print(f"\n Listo. {total:,} registros exportados → {args.output}")
