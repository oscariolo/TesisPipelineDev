# Log Analysis Pipeline

## Quick start

```bash
source .venv/bin/activate
python main.py --mode generative --backend ollama --model-name qwen3.6:latest
python main.py --mode generative --backend huggingface --model-name microsoft/Phi-3-mini-4k-instruct --hf-device gpu
python main.py --mode embedding  # embedding + vector-store lookup, falls back to LLM on first sight
```

## Virtual env

`.venv/` at repo root. Python 3.13. Activate before running anything.

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
    log_entry.py       LogEntry, LogBatch, AnalysisResult, BatchAnalysisResult (Pydantic)
    log_ingestor.py    Batch-based ingesters: FileLogIngestor (logs/*.log) and
                       StreamLogIngestor (HTTP service, polls after each batch)
    log_embedder.py    LogVectorStore — Milvus Lite wrapper (create collection, insert,
                       cosine search) persisting classified groups to a local .db file
    pipeline.py        Orchestrator: ingestor → model → writer (batch oriented)
  models/
    base.py            Abstract BaseModel (analyze(LogBatch) → BatchAnalysisResult)
    generative.py      GenerativeModel: HF Transformers or Ollama backend
    embedding.py       EmbeddingModel: sentence-embed → vector lookup → reuse stored result,
                       else fall back to GenerativeModel and store the new classification
  output/
    json_writer.py     Appends BatchAnalysisResult as JSONL to analysis/log_analysis.jsonl
main.py                CLI entry point
```

## CLI arguments (main.py)

| Argument | Default | Choices |
|---|---|---|
| `--mode` | `generative` | `generative`, `embedding` |
| `--backend` | `ollama` | `huggingface`, `ollama` |
| `--model-name` | `qwen3.6:latest` | any |
| `--ollama-host` | `localhost` | any |
| `--ollama-port` | `11434` | any int |
| `--hf-device` | `gpu` | any (e.g. `cpu`, `cuda:0`) |
| `--embedding-model-name` | `sentence-transformers/all-MiniLM-L6-v2` | HF sentence embedding repo |
| `--embedding-db` | `./embeddings/log_embeddings.db` | Milvus Lite `.db` file path |
| `--embedding-threshold` | `0.8` | min cosine similarity to reuse a stored classification |
| `--log-dir` | `./logs` | path (used only in file mode) |
| `--batch-size` | `100` | logs per batch |
| `--stream-url` | `None` | HTTP URL of a live log stream; enables stream mode |
| `--poll-interval` | `5.0` | seconds to wait between stream batch reads |
| `--output-dir` | `./analysis` | path |

## Important facts

- **GPU is ROCm**, not CUDA. `torch` is built for ROCm 7.2.
- **Output format**: JSONL, one `BatchAnalysisResult` per line (`batch_id`, `is_error`, `error_count`,
  per-entry `results`) appended to `analysis/log_analysis.jsonl`.
- **Batches**: logs are consumed in `--batch-size` chunks and the whole batch is sent to the
  model in one call; per-log failures never crash the batch — they count toward `error_count`.
- **No state tracking**: every run re-processes all `.log` files in the input dir.
- **Error resilience**: pipeline never crashes on a single log failure — writes an error batch result and continues.
- **HF models load lazily** on first `analyze()` call, not at construction.
- **Prompt**: generative models receive the full batch (every entry numbered by index) in a strict
  JSON-only prompt; the model returns a `results` array keyed by `log_index`, regex-parsed for `{...}`/`[...]` fallback.
- **`thinking` flag** available in `GenerativeConfig` for Ollama models that support chain-of-thought.
- **Embedding mode** wraps a generative model: on first sight of a log group it classifies with the LLM and
  stores the sentence-embedding + label in the Milvus Lite `.db`; similar groups answered from the vector DB
  (cosine ≥ `--embedding-threshold`) never hit the LLM again. Persistence survives across runs because the
  `.db` file is reopened.
- **Generative fallback backends**: embedding mode reuses `--backend`/`--model-name`, so it can classify
  first-time groups with either Ollama or any repository loaded via `transformers` (e.g.
  `--backend huggingface --model-name HuggingFaceTB/SmolLM2-1.7B`). The fallback model and the embedding
  encoder share the same `--hf-device`.
- **No tests**, **no CI**, **no lint/typecheck config**.
