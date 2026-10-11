# TesisPipelineDev

Thesis project: a batch log-analysis pipeline for the DTIC (Universidad de Cuenca) that classifies
logs as error/non-error using small language models (SLMs), plus dataset building, LoRA fine-tuning,
and model evaluation, plus a three-stage multi-model pseudo-labeling pipeline for the ground
truth. Objectives live in `ThesisSources/MainObjectives.txt`. The runnable code is all
under `scripts/`; the repo root is just docs and scratch files.

For the pipeline architecture and the full `main.py` CLI table, see `scripts/AGENTS.md`. The pinned
versions in `scripts/requierements.txt` describe this machine's environment, so re-check them if an
install fails.

## Dev environment

- Python venv: `scripts/.venv` (CPython 3.13). `scripts/.python-version` says `3.12.12` — the venv
  is the source of truth.
- Activate: `source scripts/.venv/bin/activate`
- Deps: `scripts/requierements.txt` (note the misspelling — the filename really is that). It pins
  **ROCm** torch (`torch==2.13.0+rocm7.2`); this machine's GPU is AMD/ROCm, not CUDA.
  CUDA installs use `scripts/requirements.nvidia.txt`.
- Required secret: `HF_TOKEN` in `scripts/.env` (gitignored). `main.py` and `training/training.py`
  call `huggingface_hub.login()` at import time with `os.getenv("HF_TOKEN", "")`; without it you get
  warnings and unauthenticated model pulls.
- **Run everything from `scripts/`**, not the repo root — imports are `log_analysis.*` and all CLI
  paths (`./logs`, `./analysis`, `./embeddings`) are relative to the CWD.

## Build & run

```bash
cd scripts && source .venv/bin/activate

python main.py --help                                              # verified
python main.py --mode generative --backend ollama  --model-name qwen3.6:latest
python main.py --mode generative --backend huggingface --model-name HuggingFaceTB/SmolLM2-1.7B --hf-device cuda:0
python main.py --mode embedding                                    # vector lookup, LLM fallback on first sight
python main.py --analysis-dir ./analysis --evaluate                 # output + evaluation in analysis/<model>/

python dataBuilder/builder.py ./logs ./dataset/log_dataset.jsonl   # build dataset
python dataBuilder/analyze_parsed_dataset.py ./dataset/dataset_slm_procesado_web.json --output ./analysis/dataset_stats.json

# Ground truth, 3 ordered stages (preprocess -> annotate with 3 Ollama models -> consensus)
python pseudo_labeling/log_preprocessor.py
python pseudo_labeling/batch_runner.py --mode test --limit 30      # smoke test; --mode full for everything
python pseudo_labeling/consensus_engine.py
python evaluation/model_evaluation_viz.py                          # reads dataset/ vs analysis/, writes analysis/<model>/

# Dashboard: reads analysis/ output, auto-follows the newest run (run_<id>.json)
python dashboard/app.py                                            # http://127.0.0.1:8050
python dashboard/app.py --run-id <id>                              # pin a run; --all aggregates them all

# Fine-tune (args mirror .vscode/launch.json)
python training/training.py --model_name mradermacher/llama-3.2-1B-log-analyzer-GGUF \
  --gguf-file llama-3.2-1B-log-analyzer.f16.gguf \
  --train_file ./dataset/dataset_slm_procesado_web.json --output_dir ./fine_tuned_model \
  --label_column ground_truth_label.is_error --num_train_epochs 3 --test_size 0.2 --max_length 512
```

Docker (from `scripts/`; backend selected via `GPU_BACKEND`, default `rocm`):

```bash
GPU_BACKEND=rocm docker compose build
GPU_BACKEND=rocm docker compose -f docker-compose.yml -f docker-compose.rocm.yml up
docker compose up                    # CPU, no GPU
```

## Tests

- Runnable unittest suites (run from `scripts/`, 32 tests total): `test_generative_response` (11),
  `evaluation.test_model_evaluation` (12), `test_naming` (5), `test_evaluation_runner` (2) and
  `test_pipeline_telemetry` (2) — e.g. `python -m unittest test_naming`.
- `test_batch_processing.py` is pytest-style (`monkeypatch`, `tmp_path`) but **pytest is not
  installed** in the venv — it silently does nothing under unittest. Install it (`pip install
  pytest`) before running; do not add it to `requierements.txt` casually.
- No CI, no lint/typecheck/format config anywhere in the repo.

## Conventions

- Package layout: `log_analysis/` with `core/` (ingestors, pipeline, pydantic `log_entry`),
  `models/` (`base.py` abstract `analyze(LogBatch) -> BatchAnalysisResult`, `generative.py`,
  `embedding.py`), `output/` (JSONL writer). Entry points are sibling scripts: `main.py`,
  `dataBuilder/`, `evaluation/`, `training/`.
- Data structures are **Pydantic v2** models; configs are pydantic `*Config` classes.
- Every CLI is a plain `argparse` `main()` guarded by `if __name__ == "__main__":`.
- Output is one folder per model: `analysis/<model>/log_analysis.jsonl` (one `BatchAnalysisResult`
  per line — `error_found`, `is_valid_response`, `raw_response`, plus telemetry: `run_id`,
  `elapsed_seconds`, `entry_count`, `source_files`, `embedding_hit`, `similarity`,
  `error_description`), the matching `log_analysis.json`, and `run_<run_id>.json` (status, config,
  totals) which the dashboard reads. `--analysis-dir` / `--output-dir` set the parent; the `<model>`
  folder is always appended.
- Log an LLM failure and continue; a bad batch must never crash the pipeline. Model JSON output is
  parsed leniently (regex for `{`, brace/paren fixups) — keep that tolerance when editing.

## Pitfalls

- `--hf-device` is largely informational: models load with `device_map="auto"` and spill to a
  `./offloading` folder when VRAM is short. Its `main.py` default is `"gpu"`, while `launch.json`
  uses `cuda:0`.
- Qwen3 HF models default to reasoning mode in their chat template. Thinking is off here by default
  (`--thinking` enables it); the switch only works through `apply_chat_template(enable_thinking=...)`,
  and enabling it auto-raises `--max-new-tokens` to 1024 because reasoning otherwise eats the budget
  and truncates the JSON answer.
- `EmbeddingConfig` hardcodes `clearCollectionAtStartup=True` in `main.py` — the Milvus Lite
  collection is wiped every run.
- Generated artifacts, do not hand-edit: `analysis/*`, `embeddings/log_embeddings.db` (Milvus Lite),
  `fine_tuned_model/checkpoint-*`, `dataset/dataset_slm_procesado_web.json`.
- `pseudo_labeling/` runs in order (1 preprocess, 2 batch_runner, 3 consensus). Stage 2 needs three
  Ollama models pulled — `qwen2.5:7b`, `llama3.1:8b`, `mistral:7b` — and stage 3 needs **pandas**,
  which is installed in the venv but not pinned in `scripts/requierements.txt`.
- Pipeline output and its evaluation live together in `analysis/<model>/`; `python main.py --evaluate`
  fills the report and plots for that run only (scoped by `run_id`). In streaming mode the run never
  ends on its own, so the evaluation fires on Ctrl-C or `--max-batches`.
- `.vscode/launch.json` is stale: its "Builder log dataset" config points at
  `scripts/log_analysis/dataset/builder.py`, which does not exist — the builder lives at
  `scripts/dataBuilder/builder.py`.
- Ollama on port `11434`, Grafana on `3000`. On the host, Ollama is expected running at
  `localhost:11434`; inside compose it is the `ollama` service.
- Repo root has two large untracked text files (`structurefixes`, `thesisProjectHelper`) and an empty
  `log_analysis.jsonl` — scratch, not code.
