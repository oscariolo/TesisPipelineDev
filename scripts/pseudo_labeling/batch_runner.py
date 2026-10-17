"""
Módulo 2: Batch Runner — Inferencia Individual por Modelo
=========================================================
Lee el archivo `processed_logs.jsonl` (salida del Módulo 1),
envía CADA LOG INDIVIDUALMENTE a cada modelo configurado vía Ollama,
y genera un archivo de resultados independiente por modelo.

Modos:
    --mode test --limit N   →  Procesa solo las primeras N líneas
    --mode full             →  Procesa todos los logs
    --fast                  →  Prompt mínimo: el modelo responde solo `true`/`false`
                               (temperature 0, num_predict 5). ~10-20x más rápido.

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
    "NORMAL", "AUTH_OR_POLICY_ERROR", "USER_NOT_FOUND", "BACKEND_FAILURE",
    "APPLICATION_EXCEPTION", "SUSPICIOUS_ACTIVITY", "GENERIC_ERROR",
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a DevOps expert analysing logs of ms-users, a Spring Boot (Logback) \
microservice of Universidad de Cuenca that manages user accounts against Active Directory, \
LDAP, Google Workspace, Oracle and AS400.
Classify ONE log event as an error/anomaly or as normal operation.

Error (is_error=true): failed operations, exceptions or stack traces, password policy \
rejections, users not found in AD/LDAP/Google/Oracle/AS400, backend connectivity or timeout \
failures, HTTP 4xx/5xx responses, anything a system administrator would need to investigate.
Normal (is_error=false): successful operations (password changed, user created, "Completed 200 OK"), \
routine request mapping and informational/debug messages.
Judge by the content of the message, not only by its log level.

Respond ONLY with valid JSON, no extra text:
{
  "is_error": true or false,
  "error_category": "NORMAL" | "AUTH_OR_POLICY_ERROR" | "USER_NOT_FOUND" | "BACKEND_FAILURE" | "APPLICATION_EXCEPTION" | "SUSPICIOUS_ACTIVITY" | "GENERIC_ERROR",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one concise sentence>"
}"""

FAST_SYSTEM_PROMPT = """You classify ONE log event of ms-users, a Spring Boot microservice that \
manages user accounts (Active Directory, LDAP, Google Workspace, Oracle, AS400).
Answer true if the event is an error or anomaly an administrator should investigate (failed \
operation, exception, stack trace, password policy rejection, user not found, backend failure, \
HTTP 4xx/5xx). Answer false for normal operation (successful operations, "Completed 200 OK", \
routine request mapping, informational messages). Judge the message, not only its level.
Reply with exactly one word: true or false."""

MAX_STACK_CHARS = 1500


def build_user_prompt(entry: dict) -> str:
    stack = (entry.get("stack_trace") or "")[:MAX_STACK_CHARS]
    prompt = (
        "Analyze this log event:\n\n"
        f"Timestamp: {entry.get('timestamp', 'unknown')}\n"
        f"Level: {entry.get('level', 'unknown')}\n"
        f"Logger: {entry.get('logger', 'unknown')}\n"
        f"Message: {entry.get('message', '')}\n"
    )
    if stack:
        prompt += f"Stack trace:\n{stack}\n"
    return prompt + "\nClassify this log event."


# ---------------------------------------------------------------------------
# Schema de resultado
# ---------------------------------------------------------------------------
class LogAnnotation(BaseModel):
    log_id: str
    is_error: bool
    error_category: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str = ""
    mode: str = "normal"          # "normal" | "fast"
    from_cache: bool = False      # copiada de un log idéntico ya anotado por este mismo modelo
    failed: bool = False          # el modelo no devolvió una respuesta válida tras los reintentos

    @field_validator("error_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        return v.upper() if v.upper() in VALID_CATEGORIES else "GENERIC_ERROR"

    @field_validator("confidence")
    @classmethod
    def clamp(cls, v: float | None) -> float | None:
        return None if v is None else max(0.0, min(1.0, float(v)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BOOL_RE = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)


def parse_fast_response(raw: str) -> bool | None:
    """Primera palabra true/false (o yes/no) de la respuesta; None si no hay ninguna."""
    m = _BOOL_RE.search(raw)
    if not m:
        return None
    return m.group(1).lower() in ("true", "yes")


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
    logger.info(" Resume: %d logs ya procesados en '%s'", len(done), path.name)
    return done


def event_key(entry: dict) -> tuple:
    """Dos logs con la misma clave son idénticos para el modelo (el prompt solo usa estos campos,
    salvo el timestamp, que no cambia el veredicto)."""
    return (entry.get("level", ""), entry.get("logger", ""),
            entry.get("message", ""), entry.get("stack_trace", ""))


def load_annotations(path: Path) -> dict[str, dict]:
    """log_id -> registro ya escrito (para resume y para reconstruir la memoización)."""
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if "log_id" in r:
                    records[r["log_id"]] = r
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Inferencia individual
# ---------------------------------------------------------------------------
def annotate_log(
    client: ollama.Client,
    model_name: str,
    entry: dict,
    max_retries: int = 3,
    fast: bool = False,
) -> LogAnnotation:
    """Anota un log. Si agota los reintentos devuelve `failed=True` (no se escribe en disco)."""
    log_id = entry["log_id"]
    messages = [
        {"role": "system", "content": FAST_SYSTEM_PROMPT if fast else SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(entry)},
    ]
    options = {"temperature": 0, "num_predict": 5} if fast else {"temperature": 0.1, "num_predict": 300}
    mode = "fast" if fast else "normal"

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat(model=model_name, messages=messages, options=options, keep_alive="30m")
            raw = resp["message"]["content"].strip()
            if fast:
                verdict = parse_fast_response(raw)
                if verdict is None:
                    logger.warning("[%s] Respuesta fast sin true/false: %.60r", log_id, raw)
                    continue
                return LogAnnotation(
                    log_id=log_id, is_error=verdict,
                    error_category="GENERIC_ERROR" if verdict else "NORMAL",
                    mode=mode,
                )
            data = parse_model_response(raw, log_id)
            if data:
                return LogAnnotation(
                    log_id=log_id,
                    is_error=bool(data.get("is_error", False)),
                    error_category=str(data.get("error_category", "GENERIC_ERROR")),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=str(data.get("reasoning", "")),
                    mode=mode,
                )
        except Exception as exc:
            logger.warning("[%s] Intento %d/%d error: %s", log_id, attempt, max_retries, exc)
            time.sleep(attempt)

    return LogAnnotation(
        log_id=log_id, is_error=False, error_category="NORMAL",
        reasoning="[FAILED] Modelo no respondió tras reintentos.", mode=mode, failed=True,
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
    fast: bool = False,
) -> dict:
    model_id, model_name, label = cfg["id"], cfg["name"], cfg["label"]
    out_path = output_dir / f"results_{model_id}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🤖 %s (%s) — modo %s", label, model_name, "FAST" if fast else "normal")
    logger.info("   Salida: %s", out_path)
    logger.info("=" * 60)

    client = ollama.Client(host=f"http://{ollama_host}:{ollama_port}")
    try:
        client.show(model_name)
        logger.info("   ✅ Modelo disponible en Ollama.")
    except Exception as e:
        logger.error("❌ Modelo '%s' no disponible: %s", model_name, e)
        raise RuntimeError(f"Modelo no disponible: {model_name}") from e

    done_records = load_annotations(out_path)
    already_done = set(done_records)
    logger.info(" Resume: %d logs ya procesados en '%s'", len(already_done), out_path.name)

    # Memoización de logs idénticos (mismo nivel/logger/mensaje/stack): todos los logs reciben
    # etiqueta propia en el dataset, pero el LLM solo se consulta una vez por contenido único.
    memo: dict[tuple, dict] = {}
    total_in_file = 0
    for e in iter_processed_logs(input_path, limit=limit):
        total_in_file += 1
        rec = done_records.get(e["log_id"])
        if rec is not None and not rec.get("from_cache"):
            memo.setdefault(event_key(e), rec)
    pending = total_in_file - len(already_done)

    logger.info("   Total: %d | Ya procesados: %d | Pendientes: %d",
                total_in_file, len(already_done), pending)

    if pending <= 0:
        logger.info("   ✅ Todos los logs ya fueron procesados.")
        return {"model": model_name, "written": 0, "failed": 0, "skipped": total_in_file}

    stats = {"model": model_name, "written": 0, "failed": 0, "cached": 0,
             "llm_calls": 0, "skipped": len(already_done)}
    start = time.time()
    processed_now = 0

    with open(out_path, "a", encoding="utf-8") as f:
        for entry in iter_processed_logs(input_path, limit=limit):
            if entry["log_id"] in already_done:
                continue

            key = event_key(entry)
            processed_now += 1
            if key in memo:
                annotation = LogAnnotation(**{**memo[key], "log_id": entry["log_id"], "from_cache": True})
                stats["cached"] += 1
            else:
                annotation = annotate_log(client, model_name, entry, max_retries, fast=fast)
                stats["llm_calls"] += 1
                if not annotation.failed:
                    memo[key] = annotation.model_dump()

            if annotation.failed:
                # No se escribe: queda pendiente y se reintenta en la próxima corrida (resume).
                stats["failed"] += 1
            else:
                f.write(annotation.model_dump_json() + "\n")
                f.flush()
                stats["written"] += 1

            if processed_now % 500 == 0:
                elapsed = time.time() - start
                rate = processed_now / elapsed if elapsed > 0 else 1
                eta_min = (pending - processed_now) / rate / 60
                logger.info("  ⏳ [%s] %d/%d | %.1f logs/s | LLM: %d, cache: %d | ETA: %.1f min",
                            model_id, processed_now, pending, rate,
                            stats["llm_calls"], stats["cached"], eta_min)

    elapsed_total = time.time() - start
    stats["elapsed_seconds"] = elapsed_total
    logger.info("✅ '%s' completado en %.1f min. Escritos: %d (LLM %d, cache %d) | Fallidos: %d",
                label, elapsed_total / 60, stats["written"], stats["llm_calls"],
                stats["cached"], stats["failed"])
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
    fast: bool = False,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {input_path}\n"
            "Ejecuta primero el Módulo 1: log_preprocessor.py"
        )

    active = [m for m in models if only_model in (m["id"], m["name"])] if only_model else models
    if only_model and not active:
        raise ValueError(f"Modelo '{only_model}' no encontrado.")

    logger.info("🚀 Pipeline de Anotación Multimodelo")
    logger.info("   Modelos : %s", [m["label"] for m in active])
    logger.info("   Límite  : %s", f"{limit} logs" if limit else "todos")
    logger.info("   Modo    : %s", "FAST (true/false)" if fast else "normal (JSON)")

    summary = []
    for cfg in active:
        try:
            res = run_model(cfg, input_path, output_dir, limit, ollama_host, ollama_port,
                            max_retries, fast=fast)
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


def parse_models_arg(value: str) -> list[dict]:
    """`model_a=qwen2.5:7b,model_b=llama3.1:8b` → lista de configs."""
    models = []
    for part in value.split(","):
        model_id, _, name = part.strip().partition("=")
        if not name:
            raise argparse.ArgumentTypeError(f"Formato inválido '{part}'. Usa id=tag_ollama")
        models.append({"id": model_id, "name": name, "label": name})
    return models


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Módulo 2 — Inferencia individual por modelo (Ground Truth)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python pseudo_labeling/batch_runner.py --mode test --limit 30
  python pseudo_labeling/batch_runner.py --mode test --limit 30 --fast
  python pseudo_labeling/batch_runner.py --mode full --fast --only-model model_a
  python pseudo_labeling/batch_runner.py --mode full --fast --models model_a=gemma2:9b,model_b=llama3.1:8b,model_c=mistral:7b
        """,
    )
    parser.add_argument("--mode", choices=["test", "full"], required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Logs a procesar en modo test.")
    parser.add_argument("--fast", action="store_true",
                        help="Respuesta true/false, temperature 0, num_predict 5.")
    parser.add_argument("--input", default="pseudo_labeling/data/ms_users/processed_logs.jsonl")
    parser.add_argument("--output-dir", default="pseudo_labeling/data/ms_users")
    parser.add_argument("--models", type=parse_models_arg, default=None,
                        help="Sustituye el trío por defecto: id=tag,id=tag,...")
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
        models=args.models or ANNOTATOR_MODELS,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        max_retries=args.max_retries,
        only_model=args.only_model,
        fast=args.fast,
    )
