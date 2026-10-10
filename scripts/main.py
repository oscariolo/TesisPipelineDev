import argparse
import json
import logging
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from log_analysis.core.log_ingestor import FileLogIngestor, StreamLogIngestor, JsonLogIngestor, LogMasker
from log_analysis.core.pipeline import Pipeline
from log_analysis.models.embedding import EmbeddingConfig, EmbeddingModel
from log_analysis.models.generative import GenerativeConfig, GenerativeModel
from log_analysis.observability import init_observability
from log_analysis.output.json_writer import JsonWriter

#init_observability()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


models = [
    "openai-community/gpt2",
    "qwen3.6:latest",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B"

]

load_dotenv()
# huggingFaceToken = os.getenv("HF_TOKEN", None)
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN", ""))

def _estimate_batches(args) -> int | None:
    """Estimate how many batches the input will produce (None for streams or on error).

    Mirrors the ingestor counting rules: non-empty lines for *.log files, items with a
    template/raw_message for JSON input.
    """
    try:
        if args.stream_url:
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
    parser.add_argument("--json-file", default=None, help="Path to a JSON file containing parsed logs to ingest")
    parser.add_argument("--log-dir", type=Path, default="./logs")
    parser.add_argument("--output-dir", type=Path, default="./analysis")
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
    args = parser.parse_args()

    masker = None
    if args.mask_ips or args.mask_uuids or args.mask_numbers:
        masker = LogMasker(mask_ips=args.mask_ips, mask_uuids=args.mask_uuids, mask_numbers=args.mask_numbers)

    if args.json_file:
        ingestor = JsonLogIngestor(args.json_file, batch_size=args.batch_size)
    elif args.stream_url:
        ingestor = StreamLogIngestor(args.stream_url, batch_size=args.batch_size, poll_interval=args.poll_interval, masker=masker)
    else:
        ingestor = FileLogIngestor(args.log_dir, batch_size=args.batch_size, masker=masker)
    writer = JsonWriter(args.output_dir)

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
            hf_device=args.hf_device,
            clearCollectionAtStartup=True,  # Set to True if you want to clear the Milvus collection on startup (testing only)
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
        "masking": {"ips": args.mask_ips, "uuids": args.mask_uuids, "numbers": args.mask_numbers},
        "expected_batches": _estimate_batches(args),
    }
    if args.mode == "embedding":
        run_info.update({
            "embedding_model_name": args.embedding_model_name,
            "embedding_db": args.embedding_db,
            "embedding_threshold": args.embedding_threshold,
        })
    Pipeline(model, ingestor, writer, max_batches=args.max_batches, run_info=run_info).run()


if __name__ == "__main__":
    main()
