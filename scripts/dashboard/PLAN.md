# Dashboard plan — SLM log-analysis pipeline

Maps the thesis objectives (ThesisSources/MainObjectives.txt) to what the dashboard must
show, and lists what the pipeline has to emit for those panels to be real.

## Objectives → dashboard requirement

| Objective | What it demands | Dashboard panel |
|---|---|---|
| 1. Structured dataset classified by type | dataset stats (counts, classes, sources) | Dataset panel (from analysis/dataset_stats.json) |
| 2. SLMs adapted for log analysis | which model/backend is running | "Model in use" panel |
| 3. Validate error detection (Accuracy, F1, ROC) | precision/recall/F1/ROC vs ground truth | Model performance panel (from evaluation report) |
| 4. Pipeline + real-time analysis + suggestions, **graphical interface** | errors detected, throughput/latency, anomalies, recommended actions | Live pipeline panel + error feed |
| 5. Verify via simulated errors | simulation run summaries | Simulation / run-comparison panel (future) |

## What the pipeline emits today

`analysis/log_analysis.jsonl` (one line per batch): `batch_id`, `error_found`,
`model_name`, `embedder_model_name`, `token_usage`, `is_valid_response`, `raw_response`.
Plus `analysis/log_analysis.json` (accumulates across runs) and the evaluation report.

## Gaps — what the pipeline still needs (this is the "plan first" part)

1. **Timing** — `Pipeline._analyze_batch` computes `elapsed` then throws it away. Must be
   stored per batch → `elapsed_seconds`. (Blocks any latency/throughput panel.)
2. **Batch size** — entry count per batch isn't recorded → `entry_count`. Needed for
   logs/sec and per-log cost.
3. **Source / server** — `LogEntry.source_file` exists but is dropped → `source_files`.
   DTIC has several servers; the dashboard must break errors down per server.
4. **Embedding hit/miss + similarity** — `EmbeddingModel` knows both but records neither →
   `embedding_hit`, `similarity`. Directly the requested "embedding hits" metric and the
   cache-effectiveness story.
5. **Recommended action / description** — the model returns `error_description` and
   `recommended_action`; only `raw_response` survives → store both. This is the thesis's
   "recommend to the developer" objective, and the dashboard's action feed.
6. **Timestamp** — no per-batch time → `timestamp` (UTC ISO). Needed for the timeline and
   live view.
7. **Run identity** — the JSONL is append-only across runs; there is no `run_id`/config.
   Future: a `run_id` + run summary file so panels can be filtered per experiment.
8. **Notification** — the general objective says "notificar". Not built; dashboard shows an
   in-app alert feed first; email/webhook is a later step.

Items 1–7 are implemented now (additive, optional fields, backward compatible). Item 8
(notification) is scoped for the next iteration.

## Dashboard MVP (this iteration)

Stdlib-only (`http.server`, no new dependencies; Grafana stays a later option).
Entry point: `scripts/dashboard/app.py`.

Panels:
- Model in use — model_name(s) + embedder_model_name(s), backend implicit.
- Errors detected — count, error rate, valid/invalid responses.
- Timing — avg / p50 / p95 / max / total batch time, batches/sec.
- Embedding hits — hits / misses / hit rate (embedding mode).
- Volume — batches, logs processed, tokens.
- Evaluation — accuracy/precision/recall/F1 + confusion matrix from the eval report.
- Error feed — recent error batches with description + recommended action.

Endpoints: `/` (HTML), `/api/summary` (JSON), `/api/batches` (JSON). Polls `/api/summary`
so a live run updates in place. Data sources: `analysis/log_analysis.jsonl`,
`analysis/eval_plots/eval_report.json`, `analysis/unclassified.json`.

## Next iteration (not built yet)

- Real-time stream view (StreamLogIngestor) with a live tail.
- Notifications (email/webhook) raised on error batches.
- Grafana/OTel path if a push-based stack is preferred over the JSONL poller.
- Simulation comparison view for objective 5 (compare two runs side by side).
