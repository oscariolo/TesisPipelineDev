"""Logback / Spring Boot event parsing and prefilter rules.

The pipeline reads large application logs (the ms-users corpus is 309 files, ~16 GB) in which the
vast majority of lines are infrastructure noise. This module turns the line stream into *events*
(attaching stack-trace continuation lines to the line that opened them), decides which events are
worth analysing, and renders a kept event back to the text handed to the model.

Rules are by LOGGER / LEVEL / MESSAGE, never by IP: the 10.16.0.x addresses are the Kubernetes
ingress and also carry legitimate traffic.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

# Stack-trace lines kept per event (first N-1 + the last one, so the root cause survives).
DEFAULT_MAX_TRACE_LINES = 15

# 2026-09-23T00:31:22.382-05:00 ERROR 9 --- [   Thread-1698] logger.name  : mensaje
EVENT_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T[\d:.]+(?:[+-]\d{2}:\d{2}|Z))\s+"
    r"(?P<level>TRACE|DEBUG|INFO|WARN|ERROR|FATAL)\s+"
    r"(?P<pid>\d+)\s+---\s+"
    r"\[\s*(?P<thread>[^\]]*?)\s*\]\s+"
    r"(?P<logger>\S+)\s*:\s?(?P<message>.*)$"
)

# Application loggers (Logback abbreviates package names)
BUSINESS_LOGGER_RE = re.compile(r"^(?:e\.e\.u\.m|e\.u\.m|u\.m|m\.i|ec\.edu\.ucuenca|e\.e\.ucuenca)")

# Framework loggers that reflect real requests, kept in DEBUG/INFO
KEEP_FRAMEWORK_LOGGERS = (
    "o.s.web.servlet.DispatcherServlet",
    "s.w.s.m.m.a.RequestMappingHandlerMapping",
    ".m.m.a.ExceptionHandlerExceptionResolver",
)

ACTUATOR_RE = re.compile(r"/actuator\b")
LOGIN_REDIRECT_RE = re.compile(r"Redirecting to .*/login\b|Securing GET /login\b")
FRAMEWORK_NOISE_MSG_RE = re.compile(r"ResourceHttpRequestHandler")
ACTUATOR_DENIED_RE = re.compile(r"AccessDeniedException")

SEVERITY_LEVELS = ("WARN", "ERROR", "FATAL")


@dataclass
class FilterStats:
    """Run-wide counters for a filtered ingestion."""

    events_total: int = 0
    events_kept: int = 0
    rules: Counter = field(default_factory=Counter)
    kept_levels: Counter = field(default_factory=Counter)
    kept_loggers: Counter = field(default_factory=Counter)

    @property
    def keep_ratio(self) -> float:
        return self.events_kept / self.events_total if self.events_total else 0.0

    def as_dict(self, top_loggers: int = 30) -> dict:
        return {
            "events_total": self.events_total,
            "events_kept": self.events_kept,
            "keep_ratio": round(self.keep_ratio, 6),
            "rules": dict(self.rules),
            "kept_levels": dict(self.kept_levels),
            "kept_loggers": dict(self.kept_loggers.most_common(top_loggers)),
        }


def classify(event: dict) -> tuple[bool, str]:
    """Return ``(keep, rule)`` for one event; ``rule`` is reported so a run can be audited."""
    level, lg, msg = event["level"], event["logger"], event["message"]
    trace = event.get("stack_trace", "")

    if BUSINESS_LOGGER_RE.match(lg):
        return True, "keep:business_logger"

    if level == "TRACE":
        return False, "drop:trace"
    if ACTUATOR_RE.search(msg) or LOGIN_REDIRECT_RE.search(msg):
        return False, "drop:actuator_or_login_redirect"
    if FRAMEWORK_NOISE_MSG_RE.search(msg):
        return False, "drop:static_resource_handler"
    if ACTUATOR_DENIED_RE.search(trace) and level not in ("ERROR", "FATAL"):
        return False, "drop:access_denied_cascade"

    if level in SEVERITY_LEVELS:
        return True, "keep:warn_error"
    if lg in KEEP_FRAMEWORK_LOGGERS:
        return True, "keep:framework_request"
    return False, "drop:framework_other"


def render(event: dict) -> str:
    """The text handed to the model: the original header line plus its stack trace.

    The header is the source line verbatim (``raw_line``), so analysed text can be grepped back into
    the corpus.
    """
    line = event.get("raw_line") or (
        f'{event["timestamp"]} {event["level"]} {event["pid"]} --- [{event["thread"]}] '
        f'{event["logger"]} : {event["message"]}'
    )
    trace = event.get("stack_trace", "")
    return f"{line}\n{trace}" if trace else line


class LogbackEventFilter:
    """Stateful Logback event parser + rule filter, usable line by line.

    ``feed``/``flush`` take one line at a time so files and HTTP streams share the same parser and
    the same counters; ``iter_events``/``iter_kept`` wrap them for a whole line iterable.
    ``events_total`` is counted when an event completes, ``events_kept``/``rules`` on the kept path.
    """

    def __init__(self, max_trace_lines: int = DEFAULT_MAX_TRACE_LINES):
        self.max_trace_lines = max_trace_lines
        self.stats = FilterStats()
        self._current: dict | None = None
        self._extra: list[str] = []
        self._source = ""

    # --- parsing ---------------------------------------------------------
    def feed(self, line: str) -> dict | None:
        """Feed one line; return the event this line completed (if it started a new one)."""
        line = line.rstrip("\n").rstrip("\r")
        match = EVENT_RE.match(line)
        if match:
            done = self.flush()
            self._current = match.groupdict()
            self._current["raw_line"] = line
            self._current["source_file"] = self._source
            self._extra = []
            return done
        if self._current is not None:
            self._extra.append(line)
        return None

    def flush(self) -> dict | None:
        """Close the event in progress (call at end of file/stream)."""
        if self._current is None:
            return None
        event = self._current
        event["stack_trace"] = self._truncated_trace()
        self._current = None
        self._extra = []
        self.stats.events_total += 1
        return event

    def _truncated_trace(self) -> str:
        trace = [line for line in self._extra if line.strip()]
        if self.max_trace_lines and len(trace) > self.max_trace_lines:
            # keep the first lines (root exception) and the last one (cause)
            trace = trace[: self.max_trace_lines - 1] + trace[-1:]
        return "\n".join(trace)

    # --- iteration -------------------------------------------------------
    def iter_events(self, lines: Iterable[str], source: str = "") -> Iterator[dict]:
        """Every event in a line stream, rules not applied."""
        saved, self._source = self._source, source or self._source
        try:
            for line in lines:
                event = self.feed(line)
                if event is not None:
                    yield event
            last = self.flush()
            if last is not None:
                yield last
        finally:
            self._source = saved

    def iter_kept(self, lines: Iterable[str], source: str = "") -> Iterator[dict]:
        """Only the events the rules keep, with the run-wide counters updated."""
        saved, self._source = self._source, source or self._source
        try:
            for line in lines:
                event = self.feed_kept(line)
                if event is not None:
                    yield event
            last = self.flush_kept()
            if last is not None:
                yield last
        finally:
            self._source = saved

    def iter_file(self, path: str | Path) -> Iterator[dict]:
        path = Path(path)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            yield from self.iter_events(handle, source=path.name)

    def iter_file_kept(self, path: str | Path) -> Iterator[dict]:
        path = Path(path)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            yield from self.iter_kept(handle, source=path.name)

    # --- streaming (one line at a time) ----------------------------------
    def feed_kept(self, line: str) -> dict | None:
        """Feed one line; return the kept event it completed, if any."""
        event = self.feed(line)
        return None if event is None else self._keep_or_none(event)

    def flush_kept(self) -> dict | None:
        event = self.flush()
        return None if event is None else self._keep_or_none(event)

    def _keep_or_none(self, event: dict) -> dict | None:
        keep, rule = classify(event)
        self.stats.rules[rule] += 1
        if not keep:
            return None
        self.stats.events_kept += 1
        self.stats.kept_levels[event["level"]] += 1
        self.stats.kept_loggers[event["logger"]] += 1
        return event

    # --- reporting -------------------------------------------------------
    def report(self) -> str:
        return format_report(self.stats.as_dict())


def format_report(stats: dict) -> str:
    """Human-readable summary of a filtered run (same layout as the prefilter CLI)."""
    total = stats.get("events_total") or 1
    lines = [
        "=" * 60,
        "REPORTE DE PREFILTRADO",
        "=" * 60,
        f"Eventos totales     : {stats.get('events_total', 0):,}",
        f"Eventos conservados : {stats.get('events_kept', 0):,} "
        f"({stats.get('events_kept', 0) / total * 100:.3f}%)",
    ]
    if "files" in stats:
        lines.insert(3, f"Archivos procesados : {stats['files']}")
    lines += ["-" * 60, "Decisiones por regla:"]
    lines += [f"  {r:<36}{n:>12,}" for r, n in sorted(stats.get("rules", {}).items(), key=lambda x: -x[1])]
    lines += ["-" * 60, "Niveles conservados:"]
    lines += [f"  {lv:<36}{n:>12,}" for lv, n in sorted(stats.get("kept_levels", {}).items(), key=lambda x: -x[1])]
    lines += ["-" * 60, "Top loggers conservados:"]
    lines += [f"  {lg:<46}{n:>8,}" for lg, n in stats.get("kept_loggers", {}).items()]
    lines.append("=" * 60)
    return "\n".join(lines)
