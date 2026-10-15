import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from log_analysis.core.log_filter import (
    LogbackEventFilter,
    classify,
    format_report,
    render,
)
from log_analysis.core.log_ingestor import FileLogIngestor, LogMasker, StreamLogIngestor

# Two kept events (one with a 3-line trace), plus noise that the rules drop.
SAMPLE = """\
2026-09-23T00:00:00.136-05:00 DEBUG 9 --- [nio-8080-exec-6] o.s.security.web.FilterChainProxy        : Securing GET /actuator/health
2026-09-23T00:00:00.136-05:00 TRACE 9 --- [nio-8080-exec-6] o.s.s.w.a.logout.LogoutFilter            : Did not match request
org.springframework.security.access.AccessDeniedException: Access Denied
\tat org.springframework.security.X.check(X.java:1)
2026-09-23T00:00:00.137-05:00 DEBUG 9 --- [nio-8080-exec-6] o.s.s.web.DefaultRedirectStrategy        : Redirecting to http://10.16.0.1:8080/ms/users/api/login
2026-09-23T00:31:22.382-05:00 ERROR 9 --- [    Thread-1698] e.e.u.m.i.a.o.e.a.UsuarioAs400Service    : User not found with personal identification as400: 0150369817
2026-09-23T00:31:22.392-05:00 ERROR 9 --- [    Thread-1700] e.e.u.m.o.o.UsuarioOracleService   : Oracle failure
java.sql.SQLException: boom
\tat a.b.C.d(C.java:1)
\tat a.b.C.e(C.java:2)
2026-09-23T00:40:00.000-05:00 DEBUG 9 --- [nio-8080-exec-2] o.s.web.servlet.DispatcherServlet        : Completed 200 OK
2026-09-23T00:41:00.000-05:00 INFO 9 --- [nio-8080-exec-2] c.z.SomeFrameworkThing                   : nothing special
"""


def _write(dir_path: Path, name: str, text: str) -> Path:
    path = dir_path / name
    path.write_text(text, encoding="utf-8")
    return path


class ParsingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_groups_continuation_lines_into_one_event(self):
        path = _write(self.tmp, "logFile.test.log", SAMPLE)
        events = list(LogbackEventFilter(15).iter_file(path))

        self.assertEqual(len(events), 7)
        self.assertIn("AccessDeniedException", events[1]["stack_trace"])
        self.assertIn("SQLException", events[4]["stack_trace"])
        self.assertEqual(events[4]["source_file"], "logFile.test.log")

    def test_trace_is_truncated_keeping_first_and_last(self):
        path = _write(self.tmp, "logFile.test.log", SAMPLE)
        events = list(LogbackEventFilter(2).iter_file(path))

        self.assertEqual(len(events[4]["stack_trace"].splitlines()), 2)
        self.assertIn("SQLException", events[4]["stack_trace"])   # first line (root cause)
        self.assertIn("a.b.C.e", events[4]["stack_trace"])        # last line (cause)

    def test_feed_and_flush_work_across_chunks(self):
        event_filter = LogbackEventFilter(15)
        events = []
        for line in SAMPLE.splitlines():
            done = event_filter.feed(line)
            if done is not None:
                events.append(done)
        final = event_filter.flush()
        if final is not None:
            events.append(final)

        self.assertEqual(len(events), 7)
        self.assertEqual(event_filter.stats.events_total, 7)

    def test_leading_continuation_lines_are_ignored(self):
        event_filter = LogbackEventFilter(15)
        self.assertIsNone(event_filter.feed("\tat nowhere(N.java:1)"))
        self.assertIsNone(event_filter.flush())


class RuleTest(unittest.TestCase):
    def _event(self, level, logger, message, trace=""):
        return {"level": level, "logger": logger, "message": message, "stack_trace": trace}

    def test_business_logger_is_always_kept(self):
        event = self._event("DEBUG", "e.e.u.m.i.a.o.e.a.UsuarioAs400Service", "lookup ok")
        self.assertEqual(classify(event), (True, "keep:business_logger"))

    def test_trace_level_is_dropped(self):
        event = self._event("TRACE", "o.s.s.w.a.ExceptionTranslationFilter", "Sending token")
        self.assertEqual(classify(event), (False, "drop:trace"))

    def test_actuator_probes_are_dropped(self):
        event = self._event("DEBUG", "o.s.security.web.FilterChainProxy", "Securing GET /actuator/health")
        self.assertEqual(classify(event), (False, "drop:actuator_or_login_redirect"))

    def test_static_resource_handler_is_dropped(self):
        event = self._event("DEBUG", "o.s.w.s.handler.SimpleUrlHandlerMapping",
                            "Mapped to ResourceHttpRequestHandler [classpath [META-INF/resources/]]")
        self.assertEqual(classify(event), (False, "drop:static_resource_handler"))

    def test_business_logger_precedence_over_severity(self):
        # A business-logger ERROR is reported as keep:business_logger, before the severity rule.
        event = self._event("ERROR", "e.e.u.m.i.a.o.e.a.UsuarioAs400Service", "Oracle failure")
        self.assertEqual(classify(event), (True, "keep:business_logger"))
        framework = self._event("ERROR", "c.z.SomeFrameworkThing", "boom")
        self.assertEqual(classify(framework), (True, "keep:warn_error"))

    def test_access_denied_cascade_is_dropped_below_error(self):
        event = self._event("DEBUG", "o.s.s.w.a.ExceptionTranslationFilter", "Access is denied",
                            trace="AccessDeniedException: Access Denied")
        self.assertEqual(classify(event), (False, "drop:access_denied_cascade"))

    def test_error_with_access_denied_is_kept(self):
        event = self._event("ERROR", "o.s.s.w.a.ExceptionTranslationFilter", "Access is denied",
                            trace="AccessDeniedException: Access Denied")
        self.assertEqual(classify(event), (True, "keep:warn_error"))

    def test_framework_request_logger_is_kept(self):
        event = self._event("DEBUG", "o.s.web.servlet.DispatcherServlet", "Completed 200 OK")
        self.assertEqual(classify(event), (True, "keep:framework_request"))

    def test_other_framework_info_is_dropped(self):
        event = self._event("INFO", "c.z.SomeFrameworkThing", "nothing special")
        self.assertEqual(classify(event), (False, "drop:framework_other"))

    def test_ip_is_not_a_filter_criterion(self):
        event = self._event("DEBUG", "o.s.web.servlet.DispatcherServlet", "Completed 200 OK (from 10.16.0.189)")
        self.assertEqual(classify(event), (True, "keep:framework_request"))


class RenderTest(unittest.TestCase):
    def test_renders_the_source_line_verbatim_plus_trace(self):
        path = Path(tempfile.mkdtemp()) / "logFile.test.log"
        path.write_text(SAMPLE, encoding="utf-8")
        events = list(LogbackEventFilter(15).iter_file(path))

        rendered = render(events[4])
        self.assertTrue(rendered.startswith(events[4]["raw_line"]))       # option (a): exact line
        self.assertIn("[    Thread-1700]", rendered)                      # padding preserved
        self.assertIn("SQLException: boom", rendered)                     # trace attached
        self.assertEqual(render(events[3]), events[3]["raw_line"])        # no trace -> line only


class StatsTest(unittest.TestCase):
    def test_counters_and_report(self):
        event_filter = LogbackEventFilter(15)
        for line in SAMPLE.splitlines():
            event_filter.feed_kept(line)
        event_filter.flush_kept()

        stats = event_filter.stats
        self.assertEqual(stats.events_total, 7)
        self.assertEqual(stats.events_kept, 3)                            # 2 business ERROR + DispatcherServlet
        self.assertEqual(stats.kept_levels["ERROR"], 2)
        self.assertEqual(stats.rules["keep:business_logger"], 2)           # precedence over keep:warn_error
        self.assertAlmostEqual(stats.keep_ratio, 3 / 7)

        report = event_filter.report()
        self.assertIn("Eventos conservados", report)
        self.assertIn("keep:business_logger", report)
        self.assertIn("drop:trace", report)

    def test_format_report_accepts_the_cli_dict(self):
        report = format_report({"files": 1, "events_total": 10, "events_kept": 2,
                                "rules": {"drop:trace": 8}, "kept_levels": {"ERROR": 2},
                                "kept_loggers": {"a.b.C": 2}})
        self.assertIn("Archivos procesados : 1", report)


class FilteredIngestorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_one_entry_per_kept_event(self):
        _write(self.tmp, "logFile.test.log", SAMPLE)
        ingestor = FileLogIngestor(self.tmp, batch_size=100, log_filter=LogbackEventFilter(15))

        batches = list(ingestor.iter_batches())
        entries = [entry for batch in batches for entry in batch.entries]

        self.assertEqual(len(entries), 3)                                  # only the kept events
        self.assertTrue(entries[0].raw_text.startswith("2026-09-23T00:31:22.382-05:00 ERROR"))
        self.assertIn("SQLException", entries[1].raw_text)                 # trace reaches the model
        self.assertNotIn("actuator", " ".join(e.raw_text for e in entries))
        self.assertTrue(all(e.source_file == "logFile.test.log" for e in entries))

    def test_batch_size_counts_kept_events(self):
        _write(self.tmp, "logFile.test.log", SAMPLE)
        ingestor = FileLogIngestor(self.tmp, batch_size=2, log_filter=LogbackEventFilter(15))

        batches = list(ingestor.iter_batches())
        self.assertEqual([len(b.entries) for b in batches], [2, 1])

    def test_filter_stats_are_exposed(self):
        _write(self.tmp, "logFile.test.log", SAMPLE)
        ingestor = FileLogIngestor(self.tmp, batch_size=100, log_filter=LogbackEventFilter(15))
        list(ingestor.iter_batches())

        self.assertEqual(ingestor.filter_stats.events_kept, 3)
        self.assertEqual(ingestor.filter_stats.events_total, 7)

    def test_no_filter_means_no_stats(self):
        _write(self.tmp, "logFile.test.log", SAMPLE)
        self.assertIsNone(FileLogIngestor(self.tmp).filter_stats)

    def test_masker_still_applies_after_filtering(self):
        _write(self.tmp, "logFile.test.log", SAMPLE)
        masker = LogMasker(mask_ips=False, mask_uuids=False, mask_numbers=True)
        ingestor = FileLogIngestor(self.tmp, batch_size=100, masker=masker,
                                   log_filter=LogbackEventFilter(15))

        entries = [e for b in ingestor.iter_batches() for e in b.entries]
        self.assertTrue(any("<NUM>" in e.raw_text for e in entries))

    def test_lines_are_still_the_unit_without_the_filter(self):
        _write(self.tmp, "logFile.test.log", SAMPLE)
        ingestor = FileLogIngestor(self.tmp, batch_size=100)

        entries = [e for b in ingestor.iter_batches() for e in b.entries]
        self.assertEqual(len(entries), 12)                                 # every non-empty line

    def test_foreign_format_is_rejected_loudly(self):
        _write(self.tmp, "access.log",
               '54.36.149.41 - - [22/Jan/2019:03:56:14 +0330] "GET /filter/x HTTP/1.1" 200 30577\n'
               '54.36.149.42 - - [22/Jan/2019:03:56:15 +0330] "GET / HTTP/1.1" 200 12\n')
        ingestor = FileLogIngestor(self.tmp, batch_size=100, log_filter=LogbackEventFilter(15))

        with self.assertRaises(ValueError) as ctx:
            list(ingestor.iter_batches())
        self.assertIn("not in Logback/Spring Boot format", str(ctx.exception))


class _FakeStream:
    """Minimal stand-in for httpx.stream(...) used as a context manager."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self._lines)


class StreamingFilterTest(unittest.TestCase):
    def test_kept_events_are_emitted_and_flushed_at_poll_end(self):
        # header + 2 trace lines, then a dropped TRACE event: only the first one is an entry.
        lines = SAMPLE.splitlines()
        ingestor = StreamLogIngestor("http://example.test/logs", batch_size=10, poll_interval=0.0,
                                     log_filter=LogbackEventFilter(15))

        with mock.patch.object(httpx, "stream", lambda *a, **k: _FakeStream(lines)):
            generator = ingestor.iter_batches()
            batch = next(generator)
            generator.close()

        entries = [e for b in [batch] for e in b.entries]
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0].source_file, "http://example.test/logs")   # no file name
        self.assertIn("SQLException", entries[1].raw_text)

    def test_stream_without_filter_emits_raw_lines(self):
        ingestor = StreamLogIngestor("http://example.test/logs", batch_size=100, poll_interval=0.0)

        with mock.patch.object(httpx, "stream", lambda *a, **k: _FakeStream(["a", "b", ""])):
            generator = ingestor.iter_batches()
            batch = next(generator)
            generator.close()

        self.assertEqual([e.raw_text for e in batch.entries], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
