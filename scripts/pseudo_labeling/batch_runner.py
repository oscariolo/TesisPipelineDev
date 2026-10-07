"""
Módulo 2: Batch Runner — Inferencia Individual por Modelo
=========================================================
Lee el archivo `processed_logs.jsonl` (salida del Módulo 1),
envía CADA LOG INDIVIDUALMENTE a cada modelo configurado vía Ollama,
y genera un archivo de resultados independiente por modelo.

Modos:
    --mode test --limit N   →  Procesa solo las primeras N líneas
    --mode full             →  Procesa todos los logs

Características:
    - Resume automático si el proceso se interrumpe.
    - Retry por log: hasta 3 reintentos si el modelo no devuelve JSON válido.
    - Progreso detallado con ETA.
    - Escritura inmediata (flush) tras cada log.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Iterator

import ollama
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_runner")

# ---------------------------------------------------------------------------
# Modelos anotadores
# ---------------------------------------------------------------------------
ANNOTATOR_MODELS: list[dict] = [
    {"id": "model_a", "name": "qwen2.5:7b",   "label": "Qwen 2.5 7B"},
    {"id": "model_b", "name": "llama3.1:8b",  "label": "LLaMA 3.1 8B"},
    {"id": "model_c", "name": "mistral:7b",   "label": "Mistral 7B"},
]

VALID_CATEGORIES = {
    "NORMAL", "HTTP_CLIENT_ERROR", "HTTP_SERVER_ERROR",
    "HTTP_REDIRECT_ISSUE", "SUSPICIOUS_REQUEST", "GENERIC_ERROR",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a cybersecurity and DevOps expert specialized in web server log analysis.
Analyze HTTP access log entries and classify them as errors/anomalies or normal traffic.

Rules:
- HTTP 5xx → HTTP_SERVER_ERROR (always an error)
- HTTP 4xx → HTTP_CLIENT_ERROR (always an error)
- Suspicious paths, scanners, path traversal → SUSPICIOUS_REQUEST
- HTTP 2xx, normal 3xx → NORMAL

Respond ONLY with valid JSON, no extra text:
{
  "is_error": true or false,
  "error_category": "NORMAL" | "HTTP_CLIENT_ERROR" | "HTTP_SERVER_ERROR" | "HTTP_REDIRECT_ISSUE" | "SUSPICIOUS_REQUEST" | "GENERIC_ERROR",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one concise sentence>"
}"""


def build_user_prompt(entry: dict) -> str:
    return (
        f"Analyze this HTTP access log entry:\n\n"
        f"IP: {entry.get('ip', 'unknown')}\n"
        f"Timestamp: {entry.get('timestamp', 'unknown')}\n"
        f"Method: {entry.get('method', 'unknown')}\n"
        f"Path: {entry.get('path', 'unknown')}\n"
        f"Protocol: {entry.get('protocol', 'unknown')}\n"
        f"Status Code: {entry.get('status_code', 'unknown')}\n"
        f"Bytes Sent: {entry.get('bytes_sent', 0)}\n"
        f"User-Agent: {entry.get('user_agent', '-')}\n\n"
        f"Classify this log entry."
    )


# ---------------------------------------------------------------------------
# Schema de resultado
# ---------------------------------------------------------------------------
class LogAnnotation(BaseModel):
    log_id: str
    is_error: bool
    error_category: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("error_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        return v.upper() if v.upper() in VALID_CATEGORIES else "GENERIC_ERROR"

    @field_validator("confidence")
    @classmethod
    def clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_model_response(raw: str, log_id: str) -> dict | None:
    clean = re.sub(r"```(?:json)?", "", raw).strip()
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", clean):
        try:
            data, _ = decoder.raw_decode(clean[m.start():])
            if isinstance(data, dict) and "is_error" in data:
                return data
        except json.JSONDecodeError:
            continue
    logger.warning("[%s] No JSON en respuesta: %.120s", log_id, raw)
    return None


def iter_processed_logs(path: Path, limit: int | None = None) -> Iterator[dict]:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                count += 1
                if limit and count >= limit:
                    break
            except json.JSONDecodeError:
                continue


def load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if "log_id" in r:
                    done.add(r["log_id"])
            except json.JSONDecodeError:
                continue
    logger.info("  ↩️  Resume: %d logs ya procesados en '%s'", len(done), path.name)
    return done


# ---------------------------------------------------------------------------
# Inferencia individual
# ---------------------------------------------------------------------------
def annotate_log(
    client: ollama.Client,
    model_name: str,
    entry: dict,
    max_retries: int = 3,
) -> LogAnnotation:
    log_id = entry["log_id"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(entry)},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat(
                model=model_name,
                messages=messages,
                options={"temperature": 0.1, "num_predict": 300},
            )
            raw = resp["message"]["content"].strip()
            data = parse_model_response(raw, log_id)
            if data:
                return LogAnnotation(
                    log_id=log_id,
                    is_error=bool(data.get("is_error", False)),
                    error_category=str(data.get("error_category", "GENERIC_ERROR")),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=str(data.get("reasoning", "")),
                )
        except Exception as exc:
            logger.warning("[%s] Intento %d/%d error: %s", log_id, attempt, max_retries, exc)
            time.sleep(attempt)

    # Fallback si todo falla
    return LogAnnotation(
        log_id=log_id, is_error=False, error_category="NORMAL",
        confidence=0.0,
        reasoning="[FALLBACK] Modelo no respondió tras reintentos.",
    )


# ---------------------------------------------------------------------------
# Runner por modelo
# ---------------------------------------------------------------------------
def run_model(
    cfg: dict,
    input_path: Path,
    output_dir: Path,
    limit: int | None,
    ollama_host: str,
    ollama_port: int,
    max_retries: int,
) -> dict:
    model_id, model_name, label = cfg["id"], cfg["name"], cfg["label"]
    out_path = output_dir / f"results_{model_id}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🤖 %s (%s)", label, model_name)
    logger.info("   Salida: %s", out_path)
    logger.info("=" * 60)

    client = ollama.Client(host=f"http://{ollama_host}:{ollama_port}")
    try:
        client.show(model_name)
        logger.info("   ✅ Modelo disponible en Ollama.")
    except Exception as e:
        logger.error("❌ Modelo '%s' no disponible: %s", model_name, e)
        raise RuntimeError(f"Modelo no disponible: {model_name}") from e

    already_done = load_processed_ids(out_path)
    total_in_file = sum(1 for _ in iter_processed_logs(input_path, limit=limit))
    pending = total_in_file - len(already_done)

    logger.info("   Total: %d | Ya procesados: %d | Pendientes: %d",
                total_in_file, len(already_done), pending)

    if pending == 0:
        logger.info("   ✅ Todos los logs ya fueron procesados.")
        return {"model": model_name, "written": 0, "failed": 0, "skipped": total_in_file}

    stats = {"model": model_name, "written": 0, "failed": 0, "skipped": len(already_done)}
    start = time.time()
    processed_now = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for entry in iter_processed_logs(input_path, limit=limit):
            if entry["log_id"] in already_done:
                continue

            annotation = annotate_log(client, model_name, entry, max_retries)
            f.write(annotation.model_dump_json() + "\n")
            f.flush()

            if annotation.confidence == 0.0 and "FALLBACK" in annotation.reasoning:
                stats["failed"] += 1
            else:
                stats["written"] += 1

            processed_now += 1
            if processed_now % 50 == 0:
                elapsed = time.time() - start
                rate = processed_now / elapsed if elapsed > 0 else 1
                eta_min = (pending - processed_now) / rate / 60
                logger.info("  ⏳ [%s] %d/%d | %.2f logs/s | ETA: %.1f min",
                            model_id, processed_now, pending, rate, eta_min)

    elapsed_total = time.time() - start
    stats["elapsed_seconds"] = elapsed_total
    logger.info("✅ '%s' completado en %.1f min. Escritos: %d | Fallidos: %d",
                label, elapsed_total / 60, stats["written"], stats["failed"])
    return stats


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------
def run_pipeline(
    input_path: Path,
    output_dir: Path,
    limit: int | None,
    models: list[dict],
    ollama_host: str,
    ollama_port: int,
    max_retries: int,
    only_model: str | None,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {input_path}\n"
            "Ejecuta primero el Módulo 1: log_preprocessor.py"
        )

    active = [m for m in models if only_model in (None, m["id"], m["name"])] if only_model else models
    if only_model and not active:
        raise ValueError(f"Modelo '{only_model}' no encontrado.")

    logger.info("🚀 Pipeline de Anotación Multimodelo")
    logger.info("   Modelos : %s", [m["label"] for m in active])
    logger.info("   Límite  : %s", f"{limit} logs" if limit else "todos")

    summary = []
    for cfg in active:
        try:
            res = run_model(cfg, input_path, output_dir, limit, ollama_host, ollama_port, max_retries)
            summary.append(res)
        except RuntimeError as e:
            logger.error("❌ Omitido '%s': %s", cfg["name"], e)
            summary.append({"model": cfg["name"], "error": str(e)})

    logger.info("\n%s", "=" * 60)
    logger.info("📊 RESUMEN FINAL")
    logger.info("=" * 60)
    for s in summary:
        if "error" in s:
            logger.info("  ❌ %s — %s", s["model"], s["error"])
        else:
            logger.info("  ✅ %s — Escritos: %d | Fallidos: %d | %.1f min",
                        s["model"], s.get("written", 0), s.get("failed", 0),
                        s.get("elapsed_seconds", 0) / 60)
    logger.info("=" * 60)
    logger.info("📁 Resultados en: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Módulo 2 — Inferencia individual por modelo (Ground Truth)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python batch_runner.py --mode test --limit 30
  python batch_runner.py --mode test --limit 30 --only-model model_a
  python batch_runner.py --mode full
        """,
    )
    parser.add_argument("--mode", choices=["test", "full"], required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Logs a procesar en modo test.")
    parser.add_argument("--input", default="pseudo_labeling/data/processed_logs.jsonl")
    parser.add_argument("--output-dir", default="pseudo_labeling/data")
    parser.add_argument("--only-model", default=None, metavar="MODEL_ID")
    parser.add_argument("--ollama-host", default="localhost")
    parser.add_argument("--ollama-port", type=int, default=11434)
    parser.add_argument("--max-retries", type=int, default=3)

    args = parser.parse_args()

    if args.mode == "test" and args.limit is None:
        parser.error("--limit es requerido en modo test. Ej: --mode test --limit 30")
    if args.mode == "full":
        args.limit = None

    run_pipeline(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        models=ANNOTATOR_MODELS,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        max_retries=args.max_retries,
        only_model=args.only_model,
    )
