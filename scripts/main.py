import argparse
import logging
from pathlib import Path

from log_analysis.core.log_ingestor import LogIngestor
from log_analysis.core.pipeline import Pipeline
from log_analysis.models.embedding import EmbeddingModel
from log_analysis.models.generative import GenerativeConfig, GenerativeModel
from log_analysis.output.json_writer import JsonWriter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Log Analysis Pipeline")
    parser.add_argument("--mode", choices=["generative", "embedding"], default="generative")
    parser.add_argument("--backend", choices=["huggingface", "ollama"], default="ollama")
    parser.add_argument("--model-name", default="qwen3.6:latest")
    parser.add_argument("--ollama-host", default="localhost")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--hf-device", default="gpu")
    parser.add_argument("--log-dir", type=Path, default="./logs")
    parser.add_argument("--output-dir", type=Path, default="./analysis")
    args = parser.parse_args()

    ingestor = LogIngestor(args.log_dir)
    writer = JsonWriter(args.output_dir)

    if args.mode == "generative":
        config = GenerativeConfig(
            model_name=args.model_name,
            backend=args.backend,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            hf_device=args.hf_device,
            thinking=False,  # Set to True if you want to enable thinking mode
        )
        model = GenerativeModel(config)
    else:
        model = EmbeddingModel()

    Pipeline(model, ingestor, writer).run()


if __name__ == "__main__":
    main()
