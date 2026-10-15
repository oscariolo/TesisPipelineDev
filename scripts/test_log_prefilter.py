import tempfile
import unittest
from pathlib import Path

from pseudo_labeling.log_prefilter import classify, iter_events, prefilter

SAMPLE = """\
2026-09-23T00:00:00.136-05:00 DEBUG 9 --- [nio-8080-exec-6] o.s.security.web.FilterChainProxy        : Securing GET /actuator/health
2026-09-23T00:00:00.136-05:00 TRACE 9 --- [nio-8080-exec-6] o.s.s.w.a.logout.LogoutFilter            : Did not match request

org.springframework.security.access.AccessDeniedException: Access Denied
\tat org.springframework.security.X.check(X.java:1)
2026-09-23T00:00:00.137-05:00 DEBUG 9 --- [nio-8080-exec-6] o.s.s.web.DefaultRedirectStrategy        : Redirecting to http://10.16.0.1:8080/ms/users/api/login
2026-09-23T00:31:22.382-05:00 ERROR 9 --- [    Thread-1698] e.e.u.m.i.a.o.e.a.UsuarioAs400Service    : User not found with personal identification as400: 0150369817
2026-09-23T00:31:22.392-05:00 ERROR 9 --- [    Thread-1700] e.e.u.m.i.a.o.e.o.UsuarioOracleService   : Oracle failure
java.sql.SQLException: boom
\tat a.b.C.d(C.java:1)
\tat a.b.C.e(C.java:2)
2026-09-23T00:40:00.000-05:00 DEBUG 9 --- [nio-8080-exec-2] o.s.web.servlet.DispatcherServlet        : Completed 200 OK
"""


class PrefilterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "logFile.test.log").write_text(SAMPLE, encoding="utf-8")

    def test_groups_continuation_lines(self):
        events = list(iter_events(self.tmp / "logFile.test.log", 15))
        self.assertEqual(len(events), 6)
        self.assertIn("AccessDeniedException", events[1]["stack_trace"])
        self.assertIn("SQLException", events[4]["stack_trace"])

    def test_trace_truncated(self):
        events = list(iter_events(self.tmp / "logFile.test.log", 2))
        self.assertEqual(len(events[4]["stack_trace"].splitlines()), 2)

    def test_drops_actuator_cascade_keeps_business(self):
        out = self.tmp / "out.jsonl"
        stats = prefilter(self.tmp, out)
        self.assertEqual(stats["events_total"], 6)
        self.assertEqual(stats["events_kept"], 3)
        self.assertEqual(stats["kept_levels"], {"ERROR": 2, "DEBUG": 1})

    def test_ip_is_not_a_filter_criterion(self):
        event = {"level": "DEBUG", "logger": "o.s.web.servlet.DispatcherServlet",
                 "message": "Completed 200 OK (from 10.16.0.189)", "stack_trace": ""}
        self.assertEqual(classify(event), (True, "keep:framework_request"))


if __name__ == "__main__":
    unittest.main()
