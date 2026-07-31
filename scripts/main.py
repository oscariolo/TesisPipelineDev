import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from log_analysis.core.log_ingestor import LogIngestor
from log_analysis.core.pipeline import Pipeline
from log_analysis.models.embedding import EmbeddingModel
from log_analysis.models.generative import GenerativeConfig, GenerativeModel
from log_analysis.observability import init_observability
from log_analysis.output.json_writer import JsonWriter

#init_observability()

#logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


models = [
    "openai-community/gpt2",
    "qwen3.6:latest",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B"

]

load_dotenv()
huggingFaceToken = os.getenv("HF_TOKEN", None)

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
            token=huggingFaceToken,
            tokenizer_name=args.tokenizer_name,
            gguf_file=args.gguf_file,
        )
        model = GenerativeModel(config)
    else:
        model = EmbeddingModel()

    Pipeline(model, ingestor, writer).run()


if __name__ == "__main__":
    main()
