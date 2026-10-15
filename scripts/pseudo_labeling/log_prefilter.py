"""
Módulo 0: Log Prefilter
=======================
Lee los logs crudos de `ms-users` (Logback / Spring Boot, ~15 GB en 309 archivos),
agrupa las líneas de continuación (stack traces) con su evento padre y descarta
el ruido de infraestructura (cadena TRACE de Spring Security, sondeos a /actuator).

Reglas:
    - El filtrado es por LOGGER / NIVEL / MENSAJE, nunca por IP: las IPs 10.16.0.x
      son el ingress de Kubernetes y también llevan tráfico legítimo.
    - Se conservan: loggers de negocio (ec.edu.ucuenca...), cualquier WARN/ERROR/FATAL
      que no sea ruido de actuator, y un pequeño conjunto de loggers de framework
      que reflejan peticiones reales (DispatcherServlet, RequestMappingHandlerMapping...).

Salida (JSONL, un evento por línea):
{
    "timestamp": "2026-09-23T00:31:22.382-05:00", "level": "ERROR", "pid": "9",
    "thread": "Thread-1698", "logger": "e.e.u.m.i.a.o.e.a.UsuarioAs400Service",
    "message": "User not found ...", "stack_trace": "", "source_file": "logFile....log"
}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("log_prefilter")

# 2026-09-23T00:31:22.382-05:00 ERROR 9 --- [   Thread-1698] logger.name  : mensaje
_EVENT_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:.]+(?:[+-]\d{2}:\d{2}|Z))\s+"
    r"(?P<level>TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\s+"
    r"(?P<pid>\d+)\s+---\s+"
    r"\[\s*(?P<thread>[^\]]*?)\s*\]\s+"
    r"(?P<logger>\S+)\s*:\s?(?P<message>.*)$"
)

# Loggers de la aplicación (nombres abreviados por Logback: e.e.u.m..., u.m.i..., e.u.m...)
_BUSINESS_LOGGER_RE = re.compile(r"^(?:e\.e\.u\.m|e\.u\.m|u\.m|m\.i|ec\.edu\.ucuenca|e\.e\.ucuenca)")

# Loggers de framework que reflejan peticiones reales y se conservan en DEBUG/INFO
_KEEP_FRAMEWORK_LOGGERS = (
    "o.s.web.servlet.DispatcherServlet",
    "s.w.s.m.m.a.RequestMappingHandlerMapping",
    ".m.m.a.ExceptionHandlerExceptionResolver",
)

_ACTUATOR_RE = re.compile(r"/actuator\b")
_LOGIN_REDIRECT_RE = re.compile(r"Redirecting to .*/login\b|Securing GET /login\b")
_FRAMEWORK_NOISE_MSG_RE = re.compile(r"ResourceHttpRequestHandler")
_ACTUATOR_DENIED_RE = re.compile(r"AccessDeniedException")


def iter_events(path: Path, max_trace_lines: int) -> Iterator[dict]:
    """Agrupa las líneas de continuación con su evento padre (streaming)."""
    current: dict | None = None
    extra: list[str] = []
    truncated = 0

    def finish() -> dict | None:
        if current is None:
            return None
        trace = [ln for ln in extra if ln.strip()]
        if max_trace_lines and len(trace) > max_trace_lines:
            # conserva las primeras líneas (excepción raíz) y la última (causa)
            trace = trace[: max_trace_lines - 1] + trace[-1:]
        current["stack_trace"] = "\n".join(trace)
        return current

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _EVENT_RE.match(line)
            if m:
                done = finish()
                if done is not None:
                    yield done
                current = m.groupdict()
                current["source_file"] = path.name
                extra = []
            elif current is not None:
                extra.append(line)
    done = finish()
    if done is not None:
        yield done


def classify(event: dict) -> tuple[bool, str]:
    """Devuelve (conservar, regla). `regla` se usa en el reporte."""
    level, lg, msg = event["level"], event["logger"], event["message"]
    trace = event.get("stack_trace", "")

    if _BUSINESS_LOGGER_RE.match(lg):
        return True, "keep:business_logger"

    if level == "TRACE":
        return False, "drop:trace"
    if _ACTUATOR_RE.search(msg) or _LOGIN_REDIRECT_RE.search(msg):
        return False, "drop:actuator_or_login_redirect"
    if _FRAMEWORK_NOISE_MSG_RE.search(msg):
        return False, "drop:static_resource_handler"
    if _ACTUATOR_DENIED_RE.search(trace) and level not in ("ERROR", "FATAL"):
        return False, "drop:access_denied_cascade"

    if level in ("WARN", "ERROR", "FATAL"):
        return True, "keep:warn_error"
    if lg in _KEEP_FRAMEWORK_LOGGERS:
        return True, "keep:framework_request"
    return False, "drop:framework_other"


def prefilter(
    input_dir: Path,
    output_path: Path,
    max_trace_lines: int = 15,
    limit_files: int | None = None,
    pattern: str = "*.log",
    per_file: bool = False,
) -> dict:
    files = sorted(input_dir.glob(pattern))
    if limit_files:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"No hay archivos '{pattern}' en {input_dir}")

    # per_file: `output_path` es un directorio y se escribe un .jsonl por log de entrada
    (output_path if per_file else output_path.parent).mkdir(parents=True, exist_ok=True)
    rules: Counter[str] = Counter()
    kept_levels: Counter[str] = Counter()
    kept_loggers: Counter[str] = Counter()
    total_events = 0
    kept = 0

    single = None if per_file else open(output_path, "w", encoding="utf-8")
    try:
        for i, path in enumerate(files, start=1):
            file_kept = 0
            out = single or open(output_path / f"{path.stem}.jsonl", "w", encoding="utf-8")
            try:
                for event in iter_events(path, max_trace_lines):
                    total_events += 1
                    keep, rule = classify(event)
                    rules[rule] += 1
                    if not keep:
                        continue
                    out.write(json.dumps(event, ensure_ascii=False) + "\n")
                    kept += 1
                    file_kept += 1
                    kept_levels[event["level"]] += 1
                    kept_loggers[event["logger"]] += 1
            finally:
                if single is None:
                    out.close()
            logger.info("[%d/%d] %s → %d eventos conservados", i, len(files), path.name, file_kept)
    finally:
        if single is not None:
            single.close()

    return {
        "files": len(files),
        "events_total": total_events,
        "events_kept": kept,
        "rules": dict(rules),
        "kept_levels": dict(kept_levels),
        "kept_loggers": dict(kept_loggers.most_common(30)),
    }


def format_report(stats: dict) -> str:
    total = stats["events_total"] or 1
    lines = [
        "=" * 60,
        "REPORTE DE PREFILTRADO",
        "=" * 60,
        f"Archivos procesados : {stats['files']}",
        f"Eventos totales     : {stats['events_total']:,}",
        f"Eventos conservados : {stats['events_kept']:,} ({stats['events_kept'] / total * 100:.3f}%)",
        "-" * 60,
        "Decisiones por regla:",
    ]
    lines += [f"  {r:<36}{n:>12,}" for r, n in sorted(stats["rules"].items(), key=lambda x: -x[1])]
    lines += ["-" * 60, "Niveles conservados:"]
    lines += [f"  {lv:<36}{n:>12,}" for lv, n in sorted(stats["kept_levels"].items(), key=lambda x: -x[1])]
    lines += ["-" * 60, "Top loggers conservados:"]
    lines += [f"  {lg:<46}{n:>8,}" for lg, n in stats["kept_loggers"].items()]
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Módulo 0 — Prefiltrado de logs ms-users")
    parser.add_argument("--input-dir", default="logs/ms-users.ucuenca-logs")
    parser.add_argument("--output", default="pseudo_labeling/data/filtered_events.jsonl")
    parser.add_argument("--report", default="pseudo_labeling/data/prefilter_report.txt")
    parser.add_argument("--max-trace-lines", type=int, default=15,
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
