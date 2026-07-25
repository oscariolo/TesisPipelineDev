# Log Analysis Pipeline

## Quick start

```bash
source .venv/bin/activate
python main.py --mode generative --backend ollama --model-name qwen3.6:latest
python main.py --mode generative --backend huggingface --model-name microsoft/Phi-3-mini-4k-instruct --hf-device gpu
python main.py --mode embedding  # stub, raises NotImplementedError per log
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

## Architecture

```
log_analysis/
  core/
    log_entry.py       LogEntry, AnalysisResult (Pydantic)
    log_ingestor.py    Polls logs/*.log, yields LogEntry stream (no state tracking)
    pipeline.py        Orchestrator: ingestor → model → writer
  models/
    base.py            Abstract BaseModel (analyze(LogEntry) → AnalysisResult)
    generative.py      GenerativeModel: HF Transformers or Ollama backend
    embedding.py       Stub, raises NotImplementedError
  output/
    json_writer.py     Appends AnalysisResult as JSONL to analysis/log_analysis.jsonl
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
| `--log-dir` | `./logs` | path |
| `--output-dir` | `./analysis` | path |

## Important facts

- **GPU is ROCm**, not CUDA. `torch` is built for ROCm 7.2.
- **Output format**: JSONL (one JSON object per line), appended to `analysis/log_analysis.jsonl`.
- **No state tracking**: every run re-processes all `.log` files in the input dir.
- **Error resilience**: pipeline never crashes on a single log failure — writes an error `AnalysisResult` and continues.
- **HF models load lazily** on first `analyze()` call, not at construction.
- **Prompt**: generative models receive a strict JSON-only prompt; model output is regex-parsed for `{...}` fallback.
- **`thinking` flag** available in `GenerativeConfig` for Ollama models that support chain-of-thought.
- **Embedding mode** is unimplemented — only the interface exists.
- **No tests**, **no CI**, **no lint/typecheck config**.
