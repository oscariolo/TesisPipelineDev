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
    naming.py          sanitize_model_name / model_output_dir — per-model folder naming
    log_ingestor.py    FileLogIngestor (logs/*.log), StreamLogIngestor (HTTP poll),
                       JsonLogIngestor (parsed JSON, uses log_data.template), LogMasker
    log_embedder.py    LogVectorStore — Milvus Lite wrapper (create, insert, cosine search)
    pipeline.py        Orchestrator ingestor -> model -> writer; writes <model>/run_<run_id>.json
  models/
    base.py            Abstract BaseModel (analyze(LogBatch) -> BatchAnalysisResult)
    generative.py      GenerativeModel: HF Transformers or Ollama; _strip_reasoning/_parse_response
    embedding.py       EmbeddingModel: sentence-embed -> vector lookup -> reuse, else LLM + store
  output/
    json_writer.py     Appends one JSON line per batch to <analysis-dir>/<model>/log_analysis.jsonl
dashboard/app.py       Read-only web dashboard; follows the newest <analysis-dir>/<model>/ folder
evaluation/            modelEvaluation.py (metrics + JSON report), model_evaluation_viz.py (PNGs
                       + the reusable run_evaluation())
training/training.py   LoRA fine-tuning over a labeled JSON dataset
dataBuilder/           Dataset building / parsing scripts
pseudo_labeling/       3-stage ground-truth builder: preprocess -> annotate -> consensus
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
| `--analysis-dir` | `./analysis` | parent folder; output goes to `<analysis-dir>/<model>/` |
| `--output-dir` | `None` | alias for `--analysis-dir`; the `<model>` folder is always appended |
| `--evaluate` | off | run the evaluation once the pipeline stops |
| `--eval-reference` / `--eval-ref-key` | `dataset/dataset_slm_procesado_web.json` / `ground_truth_label.is_error` | ground truth and its label key |
| `--eval-comp-key` | `error_found` | prediction key inside the pipeline output |
| `--eval-batch-size` | `--batch-size` | how the ground truth is grouped to match batches |
| `--eval-none-as` | `negative` | how `--evaluate` treats `None` labels |
| `--eval-score-key` | `error_probability` | per-record score fed to ROC AUC |

## Pseudo-labeling pipeline (ground truth)

Three stages, run in order from `scripts/` (this package's docstrings and log messages are in
Spanish; the rest of the repo is English). Outputs land in `pseudo_labeling/data/`.

```bash
# 1. parse raw Apache/Nginx combined logs -> log_id + structured fields
python pseudo_labeling/log_preprocessor.py                     # logs/webServer.log -> pseudo_labeling/data/processed_logs.jsonl
python pseudo_labeling/log_preprocessor.py --skip-unparseable  # drop non-matching lines instead of keeping them empty

# 2. annotate EVERY log individually with 3 Ollama models (qwen2.5:7b, llama3.1:8b, mistral:7b)
python pseudo_labeling/batch_runner.py --mode test --limit 30  # smoke test (--limit required in test mode)
python pseudo_labeling/batch_runner.py --mode full             # all logs
python pseudo_labeling/batch_runner.py --mode test --limit 30 --only-model model_a

# 3. majority vote (k >= 2) + Fleiss' kappa -> ground truth dataset
python pseudo_labeling/consensus_engine.py
```

- Stage 2 fails fast if `processed_logs.jsonl` is missing (run stage 1 first), and needs the three
  models present in Ollama; it writes `results_model_{a,b,c}.jsonl` and **resumes** by skipping
  `log_id`s already in the output file. Per-log JSON is retried up to `--max-retries` (3), falling
  back to `NORMAL` with confidence 0.0.
- Stage 3 needs **2 or more** result files, uses **pandas** (installed, but *not* pinned in
  `requierements.txt`), and writes `ground_truth_dataset.jsonl/.csv`, `expert_validation_sample.csv`
  (stratified 5% sample) and `consensus_report.txt`.
- Annotation categories: `NORMAL`, `HTTP_CLIENT_ERROR`, `HTTP_SERVER_ERROR`, `HTTP_REDIRECT_ISSUE`,
  `SUSPICIOUS_REQUEST`, `GENERIC_ERROR`; a row's category comes from the majority of the non-`NORMAL`
  votes.
- `ground_truth_dataset.jsonl` rows are **per log**, and the error label is `consensus_is_error`
  (int 0/1; `consensus_category` holds the category) — **not** `error_found` / `ground_truth_label.is_error`.
  Evaluate it with `--ref-key consensus_is_error` and a `--batch-size` matching the pipeline run, e.g.
  `python evaluation/model_evaluation_viz.py pseudo_labeling/data/ground_truth_dataset.jsonl --ref-key consensus_is_error --batch-size 100`.

## Important facts

- **GPU is ROCm**, not CUDA. `torch` is built for ROCm 7.2.
- **Output layout**: one folder per model — `analysis/<model>/{log_analysis.jsonl, log_analysis.json,
  run_<run_id>.json}` plus `eval_report.json` and the PNGs when evaluated. `<model>` is the model
  name with `/ \ :` replaced by `-` (`log_analysis/core/naming.py`); `--analysis-dir` / `--output-dir`
  set the parent and the `<model>` folder is always appended, so no flag can write flat into it.
- **Output format**: JSONL, one line per batch —
  `run_id`, `batch_id`, `timestamp`, `error_found`, `error_probability`, `model_name`,
  `embedder_model_name`, `token_usage`, `is_valid_response`, `entry_count`, `source_files`,
  `elapsed_seconds`, `embedding_hit`, `similarity`, `error_description`, `recommended_action`.
- **One JSON object per batch**, not per log line: the batch is sent as numbered lines in a single
  system + user message, and the model answers `{is_error, error_description, recommended_action}`.
  A batch-level failure never crashes the run; it is written as a batch result and processing continues.
- **Runs are scoped**: each `Pipeline` run gets a `run_id` and writes `run_<run_id>.json` in the
  model folder (status `running` -> `finished`, config, totals). The dashboard follows the newest one.
- **`--evaluate`** evaluates once the pipeline stops and writes the report/plots into the same model
  folder, restricted to that `run_id` so earlier runs in the same file are ignored. In streaming mode
  the loop never ends on its own, so it fires on Ctrl-C (handled) or `--max-batches`. Fail-soft: a
  missing or mismatched reference only warns. ROC AUC comes from `error_probability` — the model's own
  P(error) at the token that spells the JSON boolean, i.e. `softmax` over the true/false logits at that
  step (HF backend only; Ollama leaves it `None`, and an embedding cache hit reuses the score stored
  with the entry it matched) — is read through the same
  `run_id` / batch filtering as the confusion matrix, and is skipped (no `roc_auc` key at all) when no
  scores or a single ground-truth class are present.
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
  LLM and stores the embedding + label + `error_probability` in the Milvus Lite `.db`; similar groups
  (cosine >= `--embedding-threshold`) are answered without the LLM, reusing that stored score.
  `clearCollectionAtStartup=True` in `main.py` wipes that collection every run, so cross-run cache
  hits do not happen. The encoder gets the same `--hf-device` value as the LLM, so `gpu` is mapped to
  `cuda` (`SentenceTransformer` rejects `gpu`).
- **Evaluation** (`evaluation/modelEvaluation.py`) reports precision/recall/F1 for the positive
  (error-found) class, records unclassified `None` values to a separate file, and writes a JSON
  report. Nothing is sniffed from the inputs: `--model-name` and `--embedding-model-name` are stated
  on the command line, and the default output folder is `analysis/<model>/` (`<model>` = `--model-name`
  with `/ \ :` turned into `-`; `analysis/eval_plots` when no name is given), overridable with
  `--analysis-dir` / `--output-dir`. `run_evaluation(reference, comparison, output_dir,
  model_name=..., embedder_model_name=..., ...)` takes folder and model **explicitly** — `main.py
  --evaluate` passes its own output dir plus its `--model-name` / `--embedding-model-name`, so it
  cannot miss them; only the standalone CLI derives the default folder and picks the newest
  `<analysis-dir>/*/log_analysis.jsonl` when no `comparison_file` is given. Both print the resolved
  `reference` / `comparison` / `output` paths, record them in the report (`reference_file`,
  `comparison_file`, `run_id`), and exit listing the model folders when no pipeline output is found.
  `score_key` (`--score-key` / `--eval-score-key`, default `error_probability`) names the field that
  drives ROC; `compare_files` / `compare_batches` read it under the same filtering as the labels, so a
  score that skips the `run_id` filter raises the usual length mismatch instead of mispairing.

## Tests

- `python -m unittest test_generative_response` (19 tests) — reasoning stripping, response parsing,
  the thinking switch and token budget on the HF path, and the true/false decision probability
  (key-then-boolean scan, raw re-forward when the warpers masked a group).
- `python -m unittest evaluation.test_model_evaluation` (15 tests) — metrics, JSON report, None
  policy, `run_id` filtering, output-folder naming, ROC from a score key.
- `python -m unittest test_embedding_probability` (6 tests) — a cache hit reports the stored
  `error_probability`, a miss stores the fresh one, and a cache written before that field is rebuilt.
- `python -m unittest test_naming` (5 tests) — model-name sanitizing and the per-model output folder.
- `python -m unittest test_evaluation_runner` (2 tests) — `run_evaluation` writes the report and the
  plots into one folder, run-scoped.
- `python -m unittest test_pipeline_telemetry` (2 tests) — per-batch telemetry and run files.
- `test_batch_processing.py` is pytest-style (`monkeypatch`, `tmp_path`) but **pytest is not
  installed**, so it does nothing under unittest. Install pytest before relying on it.
- No CI, no lint/typecheck/format config, no `pyproject.toml`/`setup.py`.

## Pitfalls

- `error_probability` comes from the logits at the boolean token, and `generate(output_scores=True)`
  returns them **after the logits warpers**: Qwen3's generation config (`do_sample=True, temperature=0.6,
  top_k=20, top_p=0.95`) fills most of the vocab with `-inf`, so when a true/false group is masked the
  model re-forwards the prefix once for raw logits — expect one extra forward pass per batch, and
  `None` (no ROC) if that also fails. The same config means generation is sampled: identical inputs do
  not reproduce identical outputs. In embedding mode the column is only empty for Ollama runs, failed
  responses and entries cached before the score existed — cache hits inherit the matched entry's score.
- Only Ollama needs to be reachable (`localhost:11434` by default); the HF backend loads models
  locally with `device_map="auto"` and offloads to `./offloading` when VRAM is short.
- `--hf-device` is not wired into HF loading — it does not move the generative model (it does pick
  the device for the embedding encoder, where `gpu` is mapped to `cuda`).
- Small models (Qwen3-0.6B) wrap the JSON in markdown code fences and add prose; the parser tolerates
  it, but a stricter prompt or a larger model reduces noise.
- Generated artifacts, do not hand-edit: `analysis/*`, `embeddings/log_embeddings.db`,
  `fine_tuned_model/checkpoint-*`, `dataset/dataset_slm_procesado_web.json`.
- `pseudo_labeling/` holds large generated artifacts (`data/`, `batch_full.log`). Stage 2 appends and
  resumes, so delete a `results_*.jsonl` to force it to re-annotate from scratch.
