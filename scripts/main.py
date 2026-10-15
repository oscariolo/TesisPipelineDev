import argparse
import json
import logging
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from log_analysis.core.log_filter import DEFAULT_MAX_TRACE_LINES, LogbackEventFilter
from log_analysis.core.log_ingestor import FileLogIngestor, StreamLogIngestor, JsonLogIngestor, LogMasker
from log_analysis.core.naming import model_output_dir
from log_analysis.core.pipeline import DEFAULT_FAILURES_INPUT_CHARS, Pipeline
from log_analysis.models.embedding import EmbeddingConfig, EmbeddingModel
from log_analysis.models.generative import GenerativeConfig, GenerativeModel
from log_analysis.observability import init_observability
from log_analysis.output.json_writer import JsonWriter

#init_observability()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


models = [
    "llama-3.2-1B-log-analyzer-GGUF:Q4_K_M", #tokenizer: meta-llama/Llama-3.2-1B
    "Qwen/Qwen3-0.6B",
    "google/gemma-4-26B-A4B",
    "HuggingFaceTB/SmolLM2-1.7B"
]

load_dotenv()
# huggingFaceToken = os.getenv("HF_TOKEN", None)
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN", ""))

def _estimate_batches(args) -> int | None:
    """Estimate how many batches the input will produce (None for streams or on error).

    Mirrors the ingestor counting rules: non-empty lines for *.log files, items with a
    template/raw_message for JSON input. With `--log-filter logback` the kept-event count is only
    known after reading the corpus, so this returns None instead of pre-scanning 16 GB for a number
    the filter invalidates.
    """
    try:
        if args.stream_url or args.log_filter == "logback":
            return None
        count = 0
        if args.json_file:
            with open(args.json_file, encoding="utf-8") as handle:
                data = json.load(handle)
            for item in data:
                if not isinstance(item, dict):
                    continue
                log_data = item.get("log_data", {}) or {}
                if log_data.get("template") or log_data.get("raw_message"):
                    count += 1
        else:
            for path in sorted(Path(args.log_dir).glob("*.log")):
                with open(path, errors="ignore") as handle:
                    count += sum(1 for line in handle if line.strip())
        return math.ceil(count / args.batch_size) if count else 0
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Log Analysis Pipeline")
    parser.add_argument("--mode", choices=["generative", "embedding"], default="generative")
    parser.add_argument("--backend", choices=["huggingface", "ollama"], default="ollama")
    parser.add_argument("--model-name", default="qwen3.6:latest")
    parser.add_argument("--ollama-host", default="localhost")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--hf-device", default="gpu")
    parser.add_argument("--tokenizer-name", default=None, help="Tokenizer repo to use (e.g. base model for GGUF repos)")
    parser.add_argument("--gguf-file", default=None, help="GGUF file inside the model repo to load, e.g. Qwen3.6-27B-Q4_K_M.gguf")
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence embedding model for embedding mode")
    parser.add_argument("--embedding-db", default="./embeddings/log_embeddings.db", help="Milvus Lite .db file storing classified log embeddings")
    parser.add_argument("--embedding-threshold", type=float, default=0.8, help="Minimum cosine similarity to reuse a stored classification")
    parser.add_argument("--ignore-embedding-cache", dest="use_embedding_cache", action="store_false",
                        help="Still query the Milvus cache and keep saving to it, but always take the LLM's own answer.")
    parser.add_argument("--json-file", default=None, help="Path to a JSON file containing parsed logs to ingest")
    parser.add_argument("--log-dir", type=Path, default="./logs")
    parser.add_argument("--log-filter", choices=["none", "logback"], default="none",
                        help="Filter before analysis: 'logback' parses Logback/Spring Boot events and keeps "
                             "only business/severity/framework-request ones (default: none).")
    parser.add_argument("--max-trace-lines", type=int, default=DEFAULT_MAX_TRACE_LINES,
                        help="Stack-trace lines kept per event with --log-filter logback (default: 15, 0 = unlimited).")
    parser.add_argument("--failures-file", default=None,
                        help="Failure register path (default: <model folder>/failed_batches.jsonl).")
    parser.add_argument("--failures-input-chars", type=int, default=DEFAULT_FAILURES_INPUT_CHARS,
                        help="Chars of the offending input stored per failure (0 = metadata only, negative = unlimited).")
    parser.add_argument("--analysis-dir", type=Path, default=Path("./analysis"),
                        help="Parent folder holding one subfolder per model (default: ./analysis).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Alias for --analysis-dir; the per-model folder is always appended.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run the evaluation once the pipeline stops (default: off).")
    parser.add_argument("--eval-reference", default="pseudo_labeling/data/ground_truth_dataset.jsonl",
                        help="Ground-truth file compared against by --evaluate.")
    parser.add_argument("--eval-ref-key", default="consensus_is_error",
                        help="Label key inside --eval-reference (default: consensus_is_error, the pseudo-labeling 0/1 vote).")
    parser.add_argument("--eval-comp-key", default="error_found",
                        help="Prediction key inside the pipeline output (default: error_found).")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Batch size for grouping the ground truth in --evaluate (default: --batch-size).")
    parser.add_argument("--eval-none-as", choices=["negative", "unclassified"], default="negative",
                        help="How --evaluate treats None labels (default: negative).")
    parser.add_argument("--eval-score-key", default="error_probability",
                        help="Per-record score for ROC AUC (default: error_probability).")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=None, help="Maximum number of batches to process")
    parser.add_argument("--stream-url", default=None, help="Read logs as a stream from this service URL instead of a file")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds to wait between stream batch reads")
    parser.add_argument("--contextWindow", type=int, default=None, help="Maximum context window for the model (default: auto-detect)")
    parser.add_argument("--thinking", action="store_true",
                        default=os.getenv("THINKING", "").strip().lower() in {"1", "true", "yes", "y"},
                        help="Enable model reasoning/thinking mode (default: off). Qwen3 defaults to ON in its chat template.")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="Completion budget per call (default: 200). Thinking mode needs ~1024+.")
    parser.add_argument("--keepHistory", choices=["perPrompt", "always", "tokenLimit"], default="perPrompt", help="How to keep the conversation history")
    parser.add_argument("--tokenLimitPercentage", type=float, default=1.0, help="Percentage of context window to trigger history clearing in tokenLimit mode")
    parser.add_argument("--mask-ips", action="store_true", help="Mask IP addresses in logs")
    parser.add_argument("--mask-uuids", action="store_true", help="Mask UUIDs in logs")
    parser.add_argument("--mask-numbers", action="store_true", help="Mask numbers in logs")
    parser.add_argument("--testingMode", "-testingMode", action="store_true", help="Enable testing mode (for development only)")
    args = parser.parse_args()

    masker = None
    if args.mask_ips or args.mask_uuids or args.mask_numbers:
        masker = LogMasker(mask_ips=args.mask_ips, mask_uuids=args.mask_uuids, mask_numbers=args.mask_numbers)

    log_filter = None
    if args.log_filter == "logback":
        if args.json_file:
            print("Warning: --log-filter is ignored with --json-file (the JSON input is already parsed).")
        else:
            log_filter = LogbackEventFilter(max_trace_lines=args.max_trace_lines)

    if args.json_file:
        ingestor = JsonLogIngestor(args.json_file, batch_size=args.batch_size)
    elif args.stream_url:
        ingestor = StreamLogIngestor(args.stream_url, batch_size=args.batch_size, poll_interval=args.poll_interval, masker=masker, log_filter=log_filter)
    else:
        ingestor = FileLogIngestor(args.log_dir, batch_size=args.batch_size, masker=masker, log_filter=log_filter)
    output_dir = model_output_dir(args.output_dir or args.analysis_dir, args.model_name)
    writer = JsonWriter(output_dir)

    if args.mode == "generative":
        config = GenerativeConfig(
            model_name=args.model_name,
            backend=args.backend,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            hf_device=args.hf_device,
            thinking=args.thinking,
            max_new_tokens=args.max_new_tokens,
            tokenizer_name=args.tokenizer_name,
            gguf_file=args.gguf_file,
            keepHistory=args.keepHistory,
            tokenLimitPercentage=args.tokenLimitPercentage,
        )
        model = GenerativeModel(config)
    else:
        config = EmbeddingConfig(
            generative=GenerativeConfig(
                model_name=args.model_name,
                backend=args.backend,
                ollama_host=args.ollama_host,
                ollama_port=args.ollama_port,
                hf_device=args.hf_device,
                thinking=args.thinking,
                max_new_tokens=args.max_new_tokens,
                tokenizer_name=args.tokenizer_name,
                gguf_file=args.gguf_file,
                keepHistory=args.keepHistory,
                tokenLimitPercentage=args.tokenLimitPercentage,
            ),
            embedding_model_name=args.embedding_model_name,
            db_path=args.embedding_db,
            similarity_threshold=args.embedding_threshold,
            use_embedding_cache=args.use_embedding_cache,
            hf_device=args.hf_device,
            clearCollectionAtStartup=args.testingMode, 
        )
        model = EmbeddingModel(config)

    run_info = {
        "mode": args.mode,
        "backend": args.backend,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "ollama_host": f"{args.ollama_host}:{args.ollama_port}",
        "hf_device": args.hf_device,
        "keep_history": args.keepHistory,
        "thinking": args.thinking,
        "max_new_tokens": args.max_new_tokens,
        "log_dir": None if (args.json_file or args.stream_url) else str(args.log_dir),
        "json_file": args.json_file,
        "stream_url": args.stream_url,
        "output_dir": str(output_dir),
        "masking": {"ips": args.mask_ips, "uuids": args.mask_uuids, "numbers": args.mask_numbers},
        "log_filter": args.log_filter,
        "max_trace_lines": args.max_trace_lines if log_filter is not None else None,
        # What one --batch-size unit counts (kept events when the Logback filter is on).
        "batch_unit": "kept_events" if log_filter is not None else "entries",
        "expected_batches": _estimate_batches(args),
    }
    if args.mode == "embedding":
        run_info.update({
            "embedding_model_name": args.embedding_model_name,
            "embedding_db": args.embedding_db,
            "embedding_threshold": args.embedding_threshold,
            "embedding_cache": args.use_embedding_cache,
        })
    pipeline = Pipeline(model, ingestor, writer, max_batches=args.max_batches, run_info=run_info,
                        failures_path=args.failures_file,
                        failures_input_chars=args.failures_input_chars)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        # The normal way to end a stream; fall through so --evaluate still runs.
        print("\nInterrupted - pipeline stopped.")

    if log_filter is not None:
        report = log_filter.report()
        print("\n" + report)
        try:
            (output_dir / "filter_report.txt").write_text(report + "\n", encoding="utf-8")
        except OSError as e:
            print(f"Could not write the filter report: {e}")

    if args.evaluate:
        _evaluate_run(args, output_dir, pipeline.run_id)


def _evaluate_run(args, output_dir: Path, run_id: str) -> None:
    """Evaluate this run's output in place; never let the evaluation break the pipeline."""
    reference = Path(args.eval_reference)
    if not reference.exists():
        print(f"Evaluation skipped: reference file not found ({reference}).")
        return
    try:
        from evaluation.model_evaluation_viz import run_evaluation

        result = run_evaluation(
            reference_file=reference,
            comparison_file=output_dir / "log_analysis.jsonl",
            output_dir=output_dir,
            batch_size=args.eval_batch_size or args.batch_size,
            reference_key=args.eval_ref_key,
            comparison_key=args.eval_comp_key,
            none_as=args.eval_none_as,
            score_key=args.eval_score_key,
            model_name=args.model_name,
            embedder_model_name=args.embedding_model_name if args.mode == "embedding" else None,
            run_id=run_id,
        )
        print(f"Evaluation written to {result['output_dir']}")
    except Exception as exc:  # metrics may be impossible; the pipeline output still stands
        print(f"Evaluation failed ({type(exc).__name__}: {exc}); the pipeline output is intact.")


if __name__ == "__main__":
    main()
