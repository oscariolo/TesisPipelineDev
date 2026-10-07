"""Pipeline observability dashboard.

Reads the pipeline's JSONL output (and the evaluation report) and serves a small
read-only web UI so the analysis process can be watched while it runs.

Stdlib only: no Flask/FastAPI. Run from the repository's scripts/ directory:

    python dashboard/app.py                      # http://127.0.0.1:8050
    python dashboard/app.py --port 9000
    python dashboard/app.py --input analysis/log_analysis.jsonl

Endpoints:
    /              HTML dashboard (polls /api/summary and /api/batches)
    /api/summary   aggregated metrics as JSON
    /api/batches   recent batch records as JSON
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_ANALYSIS_DIR = _SCRIPTS_DIR / "analysis"


def load_records(path: Path) -> list[dict]:
    """Read a JSONL file of batch summaries, skipping blank/unparseable lines."""
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_list(path: Path) -> list:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return payload if isinstance(payload, list) else []


def load_runs(analysis_dir: Path) -> list[dict]:
    """Load per-run status/config files written by the pipeline (run_<id>.json)."""
    runs: list[dict] = []
    for path in sorted(analysis_dir.glob("run_*.json")):
        payload = load_json(path)
        if payload.get("run_id"):
            runs.append(payload)
    return runs


def pick_run(runs: list[dict], run_id: str | None) -> dict | None:
    """Return the requested run, else the most recently started one."""
    if run_id:
        return next((r for r in runs if r.get("run_id") == run_id), None)
    if not runs:
        return None
    return max(runs, key=lambda r: r.get("started_at") or "")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def build_summary(records: list[dict], report: dict, unclassified: list, run: dict | None = None, runs: list[dict] | None = None) -> dict:
    runs = runs or []
    if run and run.get("run_id"):
        records = [r for r in records if r.get("run_id") == run["run_id"]]

    total = len(records)
    errors = sum(1 for r in records if r.get("error_found"))
    valid = sum(1 for r in records if r.get("is_valid_response"))
    invalid = sum(1 for r in records if r.get("is_valid_response") is False)
    entries = sum(r.get("entry_count") or 0 for r in records)

    elapsed = [float(r["elapsed_seconds"]) for r in records if r.get("elapsed_seconds") is not None]
    hits = sum(1 for r in records if r.get("embedding_hit") is True)
    misses = sum(1 for r in records if r.get("embedding_hit") is False)
    decided = hits + misses

    tokens = sum(r.get("token_usage") or 0 for r in records)

    per_file: Counter = Counter()
    for record in records:
        for source in record.get("source_files") or []:
            per_file[source] += 1

    recent = sorted(records, key=lambda r: r.get("batch_id") or 0)[-50:]
    recent = list(reversed(recent))

    expected = ((run or {}).get("run_info") or {}).get("expected_batches")
    progress = round(total / expected, 4) if expected else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": run,
        "progress": progress,
        "expected_batches": expected,
        "runs": [
            {
                "run_id": r.get("run_id"),
                "status": r.get("status"),
                "started_at": r.get("started_at"),
                "model_name": (r.get("run_info") or {}).get("model_name"),
                "batches": (r.get("totals") or {}).get("batches"),
            }
            for r in runs[-20:]
        ],
        "totals": {
            "batches": total,
            "entries": entries,
            "errors_detected": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "valid_responses": valid,
            "invalid_responses": invalid,
            "tokens": tokens,
        },
        "timing": {
            "batches_timed": len(elapsed),
            "avg_seconds": round(statistics.fmean(elapsed), 4) if elapsed else None,
            "p50_seconds": _percentile(elapsed, 0.50),
            "p95_seconds": _percentile(elapsed, 0.95),
            "max_seconds": round(max(elapsed), 4) if elapsed else None,
            "total_seconds": round(sum(elapsed), 3) if elapsed else None,
            "batches_per_second": round(len(elapsed) / sum(elapsed), 3) if elapsed and sum(elapsed) else None,
        },
        "embedding": {
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / decided, 4) if decided else None,
        },
        "models": {
            "generative": dict(Counter(r.get("model_name") for r in records if r.get("model_name"))),
            "embedder": dict(Counter(r.get("embedder_model_name") for r in records if r.get("embedder_model_name"))),
        },
        "sources": dict(per_file.most_common(20)),
        "evaluation": {
            "model_name": report.get("model_name"),
            "embedder_model_name": report.get("embedder_model_name"),
            "accuracy": report.get("accuracy"),
            "precision": report.get("precision"),
            "recall": report.get("recall"),
            "f1_score": report.get("f1_score"),
            "confusion_matrix": report.get("confusion_matrix"),
            "classes": report.get("classes"),
            "positive_class": report.get("positive_class"),
        },
        "unclassified_count": len(unclassified) if isinstance(unclassified, list) else 0,
        "recent": recent,
    }


def make_handler(summary_fn, batches_fn):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            parsed = urlparse(self.path)
            run_override = (parse_qs(parsed.query).get("run") or [None])[0]
            if parsed.path == "/api/summary":
                body = json.dumps(summary_fn(run_override), default=str).encode("utf-8")
                self._send(200, body, "application/json")
            elif parsed.path == "/api/batches":
                body = json.dumps(batches_fn(run_override), default=str).encode("utf-8")
                self._send(200, body, "application/json")
            elif parsed.path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, format, *args) -> None:  # keep the console quiet
            pass

    return Handler


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Log Analysis Pipeline</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }
  header { padding: 16px 24px; border-bottom: 1px solid #1e293b; display: flex; justify-content: space-between; align-items: baseline; }
  h1 { font-size: 18px; margin: 0; }
  .muted { color: #94a3b8; font-size: 12px; }
  main { padding: 20px 24px; display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
  .card { background: #111c33; border: 1px solid #1e293b; border-radius: 10px; padding: 14px 16px; }
  .card h2 { margin: 0 0 10px; font-size: 13px; text-transform: uppercase; letter-spacing: .05em; color: #94a3b8; }
  .kpis { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .kpi .v { font-size: 22px; font-weight: 700; }
  .kpi .l { font-size: 11px; color: #94a3b8; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: left; padding: 4px 6px; border-bottom: 1px solid #1e293b; vertical-align: top; }
  th { color: #94a3b8; font-weight: 600; }
  .wide { grid-column: 1 / -1; }
  .err { color: #f87171; }
  .ok { color: #4ade80; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #4ade80; margin-right: 6px; }
  code { color: #cbd5e1; }
  .bar { height: 8px; background: #1e293b; border-radius: 6px; overflow: hidden; margin-top: 10px; }
  .barfill { height: 100%; width: 0%; background: #2563eb; transition: width .4s; }
  .chip { display: inline-block; background: #1e293b; border-radius: 6px; padding: 2px 8px; margin: 2px 4px 2px 0; font-size: 11px; }
  select { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 4px 6px; max-width: 100%; }
</style>
</head>
<body>
<header>
  <h1><span class="dot" id="live"></span>Log Analysis Pipeline</h1>
  <span class="muted" id="stamp">loading…</span>
</header>
<main>
  <div class="card wide">
    <h2>Current run</h2>
    <div id="runhead" class="muted">–</div>
    <div class="bar"><div id="runbar" class="barfill"></div></div>
    <div id="runmeta" style="margin-top:8px"></div>
    <div id="runlist" style="margin-top:10px"></div>
  </div>

  <div class="card">
    <h2>Errors detected</h2>
    <div class="kpis">
      <div class="kpi"><div class="v" id="errors">–</div><div class="l">error batches</div></div>
      <div class="kpi"><div class="v" id="errrate">–</div><div class="l">error rate</div></div>
      <div class="kpi"><div class="v" id="batches">–</div><div class="l">batches</div></div>
      <div class="kpi"><div class="v" id="entries">–</div><div class="l">logs processed</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Batch timing</h2>
    <div class="kpis">
      <div class="kpi"><div class="v" id="avg">–</div><div class="l">avg seconds</div></div>
      <div class="kpi"><div class="v" id="p95">–</div><div class="l">p95 seconds</div></div>
      <div class="kpi"><div class="v" id="total">–</div><div class="l">total seconds</div></div>
      <div class="kpi"><div class="v" id="bps">–</div><div class="l">batches / sec</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Embedding cache</h2>
    <div class="kpis">
      <div class="kpi"><div class="v" id="hits">–</div><div class="l">hits (no LLM)</div></div>
      <div class="kpi"><div class="v" id="misses">–</div><div class="l">misses (LLM)</div></div>
      <div class="kpi"><div class="v" id="hitrate">–</div><div class="l">hit rate</div></div>
      <div class="kpi"><div class="v" id="tokens">–</div><div class="l">tokens</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Models in use</h2>
    <div id="models" class="muted">–</div>
  </div>

  <div class="card">
    <h2>Evaluation (positive = error)</h2>
    <div class="kpis">
      <div class="kpi"><div class="v" id="acc">–</div><div class="l">accuracy</div></div>
      <div class="kpi"><div class="v" id="prec">–</div><div class="l">precision</div></div>
      <div class="kpi"><div class="v" id="rec">–</div><div class="l">recall</div></div>
      <div class="kpi"><div class="v" id="f1">–</div><div class="l">F1</div></div>
    </div>
    <div id="cm" class="muted" style="margin-top:10px"></div>
  </div>

  <div class="card">
    <h2>Sources / servers</h2>
    <table id="sources"><tbody><tr><td class="muted">–</td></tr></tbody></table>
  </div>

  <div class="card wide">
    <h2>Recent batches</h2>
    <table>
      <thead><tr><th>#</th><th>time</th><th>result</th><th>model</th><th>cache</th><th>secs</th><th>description / action</th></tr></thead>
      <tbody id="rows"><tr><td colspan="7" class="muted">–</td></tr></tbody>
    </table>
  </div>
</main>
<script>
const fmt = (v, digits = 3) => (v === null || v === undefined) ? "–" : Number(v).toFixed(digits);
async function tick() {
  try {
    const runParam = new URLSearchParams(location.search).get("run");
    const url = "/api/summary" + (runParam ? "?run=" + encodeURIComponent(runParam) : "");
    const s = await (await fetch(url)).json();
    const t = s.totals, tm = s.timing, e = s.embedding, ev = s.evaluation;
    document.getElementById("stamp").textContent = "updated " + s.generated_at;
    const run = s.run, ri = (run && run.run_info) || {};
    if (run) {
      const pct = (s.progress === null || s.progress === undefined) ? 100 : Math.min(100, s.progress * 100);
      document.getElementById("runhead").innerHTML = "<b>" + (run.status === "running" ? "running" : run.status) + "</b> · <code>" + run.run_id + "</code> · started " + (run.started_at || "–");
      document.getElementById("runbar").style.width = pct.toFixed(1) + "%";
      document.getElementById("runbar").style.background = run.status === "running" ? "#2563eb" : "#22c55e";
      document.getElementById("runmeta").innerHTML =
        "<span class='chip'>mode: " + (ri.mode || "–") + "</span>" +
        "<span class='chip'>backend: " + (ri.backend || "–") + "</span>" +
        "<span class='chip'>model: " + (ri.model_name || "–") + "</span>" +
        "<span class='chip'>batch size: " + (ri.batch_size === undefined ? "–" : ri.batch_size) + "</span>" +
        (ri.embedding_model_name ? "<span class='chip'>embedder: " + ri.embedding_model_name + "</span>" : "") +
        (ri.embedding_threshold === undefined ? "" : "<span class='chip'>cosine ≥ " + ri.embedding_threshold + "</span>") +
        "<span class='chip'>source: " + (ri.log_dir || ri.json_file || ri.stream_url || "–") + "</span>" +
        "<span class='chip'>batches: " + t.batches + (s.expected_batches ? " / " + s.expected_batches : "") + "</span>";
    } else {
      document.getElementById("runhead").innerHTML = "<span class='muted'>no run files yet — aggregating all records</span>";
      document.getElementById("runmeta").innerHTML = "";
      document.getElementById("runbar").style.width = "0%";
    }
    const runOpts = (s.runs || []).map(r => "<option value='" + r.run_id + "'" + (run && r.run_id === run.run_id ? " selected" : "") + ">" + (r.started_at || "") + " · " + (r.model_name || "?") + " · " + (r.status || "") + " · " + (r.batches === undefined ? 0 : r.batches) + "</option>").join("");
    const runSig = JSON.stringify((s.runs || []).map(r => r.run_id));
    if (window.__runSig !== runSig) {
      window.__runSig = runSig;
      const runList = document.getElementById("runlist");
      runList.innerHTML = runOpts ? "<select>" + runOpts + "</select>" : "";
      const runSelect = runList.querySelector("select");
      if (runSelect) runSelect.addEventListener("change", () => { location.search = "?run=" + runSelect.value; });
    }
    document.getElementById("errors").textContent = t.errors_detected;
    document.getElementById("errrate").textContent = (t.error_rate * 100).toFixed(1) + "%";
    document.getElementById("batches").textContent = t.batches;
    document.getElementById("entries").textContent = t.entries;
    document.getElementById("avg").textContent = fmt(tm.avg_seconds);
    document.getElementById("p95").textContent = fmt(tm.p95_seconds);
    document.getElementById("total").textContent = fmt(tm.total_seconds, 2);
    document.getElementById("bps").textContent = fmt(tm.batches_per_second);
    document.getElementById("hits").textContent = e.hits;
    document.getElementById("misses").textContent = e.misses;
    document.getElementById("hitrate").textContent = e.hit_rate === null ? "–" : (e.hit_rate * 100).toFixed(1) + "%";
    document.getElementById("tokens").textContent = t.tokens;
    const models = Object.entries(s.models.generative).map(([k, v]) => `<div><code>${k}</code> ×${v}</div>`).join("");
    const embedders = Object.entries(s.models.embedder).map(([k, v]) => `<div>embedder: <code>${k}</code> ×${v}</div>`).join("");
    document.getElementById("models").innerHTML = (models + embedders) || "–";
    document.getElementById("acc").textContent = fmt(ev.accuracy);
    document.getElementById("prec").textContent = fmt(ev.precision);
    document.getElementById("rec").textContent = fmt(ev.recall);
    document.getElementById("f1").textContent = fmt(ev.f1_score);
    document.getElementById("cm").innerHTML = ev.confusion_matrix
      ? "<div class='muted'>confusion matrix (" + (ev.classes || "") + ")</div>" +
        ev.confusion_matrix.map(r => "<code>[" + r.join(", ") + "]</code>").join("<br>")
      : "<span class='muted'>no evaluation report</span>";
    const srcRows = Object.entries(s.sources).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    document.getElementById("sources").innerHTML = "<tbody>" + (srcRows || "<tr><td class='muted'>–</td></tr>") + "</tbody>";
    document.getElementById("rows").innerHTML = s.recent.map(r => {
      const res = r.error_found ? "<span class='err'>error</span>" : "<span class='ok'>ok</span>";
      const cache = r.embedding_hit === true ? "hit" : r.embedding_hit === false ? "miss" : "–";
      const note = [r.error_description, r.recommended_action].filter(Boolean).join(" → ") || "<span class='muted'>–</span>";
      return `<tr><td>${r.batch_id}</td><td>${r.timestamp || "–"}</td><td>${res}</td><td><code>${r.model_name || "–"}</code></td><td>${cache}</td><td>${fmt(r.elapsed_seconds)}</td><td>${note}</td></tr>`;
    }).join("") || "<tr><td colspan='7' class='muted'>–</td></tr>";
    document.getElementById("live").style.background = "#4ade80";
  } catch (err) {
    document.getElementById("live").style.background = "#f87171";
    document.getElementById("stamp").textContent = "connection lost";
  }
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline observability dashboard (read-only).")
    parser.add_argument("--analysis-dir", default=str(_ANALYSIS_DIR),
                        help="Directory holding the pipeline output and run_*.json files.")
    parser.add_argument("--input", default=None, help="Pipeline JSONL to read (default: <analysis-dir>/log_analysis.jsonl).")
    parser.add_argument("--report", default=None, help="Evaluation report JSON (default: <analysis-dir>/eval_plots/eval_report.json).")
    parser.add_argument("--unclassified", default=None, help="Unclassified values file (default: <analysis-dir>/unclassified.json).")
    parser.add_argument("--run-id", default=None, help="Show a specific run (default: the most recent run).")
    parser.add_argument("--all", action="store_true", help="Aggregate every run instead of following the latest one.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    input_path = Path(args.input) if args.input else analysis_dir / "log_analysis.jsonl"
    report_path = Path(args.report) if args.report else analysis_dir / "eval_plots" / "eval_report.json"
    unclassified_path = Path(args.unclassified) if args.unclassified else analysis_dir / "unclassified.json"

    def select_run(run_id: str | None) -> dict | None:
        if args.all:
            return None
        return pick_run(load_runs(analysis_dir), run_id or args.run_id)

    def summary_fn(run_id: str | None = None) -> dict:
        return build_summary(
            load_records(input_path), load_json(report_path), load_list(unclassified_path),
            select_run(run_id), load_runs(analysis_dir),
        )

    def batches_fn(run_id: str | None = None) -> list:
        selected = select_run(run_id)
        records = load_records(input_path)
        if selected and selected.get("run_id"):
            records = [r for r in records if r.get("run_id") == selected["run_id"]]
        return records[-200:]

    handler = make_handler(summary_fn, batches_fn)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[dashboard] serving {input_path} on http://{args.host}:{args.port} (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
