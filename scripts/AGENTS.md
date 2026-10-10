# Log Analysis Pipeline

Classifies log batches as error / non-error with small language models (SLMs), through either
Ollama or Hugging Face Transformers, with an optional embedding-cache layer. Everything here runs
from `scripts/` — see the repo-root `AGENTS.md` for the venv, `HF_TOKEN` and environment setup.

## Quick start

```bash
source .venv/bin/activate
python main.py --mode generative --backend ollama --model-name qwen3.6:latest
python main.py --mode generative --backend huggingface --model-name Qwen/Qwen3-0.6B --hf-device cuda:0
python main.py --mode embedding   # vector lookup first; LLM only on first sight of a log group
python dashboard/app.py           # read-only dashboard over analysis/ (http://127.0.0.1:8050)
```

## Key dependencies

| Package | Version | Notes |
|---|---|---|
| `transformers` | 5.14.1 | HF model loading |
| `torch` | 2.13.0+rocm7.2 | **ROCm, not CUDA** |
| `ollama` | 0.6.2 | API client |
| `pydantic` | 2.13.4 | Data models |
| `pymilvus[milvus_lite]` | 3.x | Milvus Lite client (local `.db` file) |
| `sentence-transformers` | 5.x | Sentence embedding encoder for embedding mode |

## Architecture

```
log_analysis/
  core/
    log_entry.py       LogEntry, LogBatch, BatchAnalysisResult (Pydantic)
    log_ingestor.py    FileLogIngestor (logs/*.log), StreamLogIngestor (HTTP poll),
                       JsonLogIngestor (parsed JSON, uses log_data.template), LogMasker
    log_embedder.py    LogVectorStore — Milvus Lite wrapper (create, insert, cosine search)
    pipeline.py        Orchestrator ingestor -> model -> writer; writes analysis/run_<run_id>.json
  models/
    base.py            Abstract BaseModel (analyze(LogBatch) -> BatchAnalysisResult)
    generative.py      GenerativeModel: HF Transformers or Ollama; _strip_reasoning/_parse_response
    embedding.py       EmbeddingModel: sentence-embed -> vector lookup -> reuse, else LLM + store
  output/
    json_writer.py     Appends one JSON line per batch to analysis/log_analysis.jsonl
dashboard/app.py       Read-only web dashboard over analysis/ (stdlib http.server)
evaluation/            modelEvaluation.py (metrics + JSON report), model_evaluation_viz.py (PNGs)
training/training.py   LoRA fine-tuning over a labeled JSON dataset
dataBuilder/           Dataset building / parsing scripts
main.py                CLI entry point
```

## CLI arguments (main.py)

| Argument | Default | Notes |
|---|---|---|
| `--mode` | `generative` | `generative`, `embedding` |
| `--backend` | `ollama` | `huggingface`, `ollama` |
| `--model-name` | `qwen3.6:latest` | any Ollama tag or HF repo |
| `--ollama-host` / `--ollama-port` | `localhost` / `11434` | Ollama endpoint |
| `--hf-device` | `gpu` | informational; HF still loads with `device_map="auto"` |
| `--tokenizer-name` / `--gguf-file` | `None` | for GGUF repos (tokenizer repo, in-repo `.gguf` file) |
| `--thinking` | off | enable model reasoning (also reads the `THINKING` env var) |
| `--max-new-tokens` | `200` | completion budget; auto-raised to 1024 when `--thinking` |
| `--embedding-model-name` | `sentence-transformers/all-MiniLM-L6-v2` | embedding mode |
| `--embedding-db` | `./embeddings/log_embeddings.db` | Milvus Lite `.db` path |
| `--embedding-threshold` | `0.8` | min cosine similarity to reuse a stored classification |
| `--json-file` | `None` | ingest parsed JSON instead of `--log-dir` |
| `--log-dir` | `./logs` | `*.log` files (file mode) |
| `--batch-size` | `100` | logs per batch |
| `--max-batches` | `None` | stop after N batches |
| `--stream-url` / `--poll-interval` | `None` / `5.0` | live HTTP log stream |
| `--contextWindow` | auto | max context window; otherwise from the model config |
| `--keepHistory` | `perPrompt` | `perPrompt`, `always`, `tokenLimit` |
| `--tokenLimitPercentage` | `1.0` | history-clear threshold for `tokenLimit` mode |
| `--mask-ips` / `--mask-uuids` / `--mask-numbers` | off | redact values before the model sees them |
| `--output-dir` | `./analysis` | JSONL + run files |

## Important facts

- **GPU is ROCm**, not CUDA. `torch` is built for ROCm 7.2.
- **Output format**: JSONL, one line per batch in `analysis/log_analysis.jsonl` —
  `run_id`, `batch_id`, `timestamp`, `error_found`, `model_name`, `embedder_model_name`,
  `token_usage`, `is_valid_response`, `entry_count`, `source_files`, `elapsed_seconds`,
  `embedding_hit`, `similarity`, `error_description`, `recommended_action`.
- **One JSON object per batch**, not per log line: the batch is sent as numbered lines in a single
  system + user message, and the model answers `{is_error, error_description, recommended_action}`.
  A batch-level failure never crashes the run; it is written as a batch result and processing continues.
- **Runs are scoped**: each `Pipeline` run gets a `run_id` and writes `analysis/run_<run_id>.json`
  (status `running` -> `finished`, config, totals). The dashboard follows the newest one.
- **Model loading**: `GenerativeModel` loads the HF model at construction (`instanceModelClient`
  -> `_load_hf`); `EmbeddingModel` loads its encoder lazily on first `analyze()`.
- **Response parsing** is lenient by design: `_strip_reasoning` removes reasoning blocks and code
  fences, then `_parse_response` scans for the first decodable JSON object that has an `is_error`
  key, with a brace/paren fixup fallback. Keep that tolerance when editing.
- **`thinking`** is off by default. For Qwen3 the switch only works through
  `apply_chat_template(enable_thinking=...)`; passing it to the tokenizer constructor is ignored.
  Reasoning text is stripped before parsing, and `--thinking` raises the budget to 1024 tokens
  because reasoning otherwise consumes `--max-new-tokens` and truncates the answer.
- **Embedding mode** wraps a generative model: on first sight of a log group it classifies with the
  LLM and stores the embedding + label in the Milvus Lite `.db`; similar groups (cosine >=
  `--embedding-threshold`) are answered without the LLM. `clearCollectionAtStartup=True` in
  `main.py` wipes that collection every run, so cross-run cache hits do not happen.
- **Evaluation** (`evaluation/modelEvaluation.py`) reports precision/recall/F1 for the positive
  (error-found) class, records unclassified `None` values to a separate file, and writes a JSON
  report with the model + embedding model names.

## Tests

- `python -m unittest test_generative_response` (11 tests) — reasoning stripping, response parsing,
  the thinking switch and token budget on the HF path.
- `python -m unittest evaluation.test_model_evaluation` (7 tests) — metrics, JSON report, None policy.
- `python -m unittest test_pipeline_telemetry` (2 tests) — per-batch telemetry and run files.
- `test_batch_processing.py` is pytest-style (`monkeypatch`, `tmp_path`) but **pytest is not
  installed**, so it does nothing under unittest. Install pytest before relying on it.
- No CI, no lint/typecheck/format config, no `pyproject.toml`/`setup.py`.

## Pitfalls

- Only Ollama needs to be reachable (`localhost:11434` by default); the HF backend loads models
  locally with `device_map="auto"` and offloads to `./offloading` when VRAM is short.
- `--hf-device` is not wired into HF loading — it does not move the model.
- Small models (Qwen3-0.6B) wrap JSON in ``` fences and add prose; the parser tolerates it, but a
  stricter prompt or a larger model reduces noise.
- Generated artifacts, do not hand-edit: `analysis/*`, `embeddings/log_embeddings.db`,
  `fine_tuned_model/checkpoint-*`, `dataset/dataset_slm_procesado_web.json`.
