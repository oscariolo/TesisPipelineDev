"""
Módulo 1: Log Preprocessor
==========================
Lee el archivo de logs crudo (formato Apache/Nginx Combined Log Format),
parsea cada línea, asigna un `log_id` único incremental y exporta
el resultado a un archivo JSONL estandarizado.

Output por registro:
{
    "log_id": "LOG-00001",
    "ip":      "54.36.149.41",
    "timestamp": "22/Jan/2019:03:56:14 +0330",
    "method":  "GET",
    "path":    "/filter/...",
    "protocol": "HTTP/1.1",
    "status_code": 200,
    "bytes":   30577,
    "referer": "-",
    "user_agent": "Mozilla/5.0 ...",
    "raw_text": "<línea original completa>"
}
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("log_preprocessor")

# ---------------------------------------------------------------------------
# Schema del registro procesado (validado con Pydantic)
# ---------------------------------------------------------------------------

class ProcessedLogEntry(BaseModel):
    """Esquema de un log procesado listo para ser enviado a los modelos."""

    log_id: str = Field(..., description="Identificador único incremental: LOG-XXXXX")
    ip: str = Field("", description="Dirección IP del cliente")
    timestamp: str = Field("", description="Timestamp original del log")
    method: str = Field("", description="Método HTTP: GET, POST, etc.")
    path: str = Field("", description="Ruta del recurso solicitado")
    protocol: str = Field("", description="Protocolo HTTP")
    status_code: int = Field(0, description="Código de respuesta HTTP")
    bytes_sent: int = Field(0, description="Bytes enviados en la respuesta")
    referer: str = Field("", description="Referer HTTP o '-' si no aplica")
    user_agent: str = Field("", description="User-Agent del cliente")
    raw_text: str = Field(..., description="Línea original sin modificar")


# ---------------------------------------------------------------------------
# Regex para Apache/Nginx Combined Log Format
# El archivo webServer.log tiene una variante con prefijo numérico opcional:
#   [id_num,]"IP - - [timestamp] ""METHOD path PROTO"" STATUS BYTES ..."
# ---------------------------------------------------------------------------

# Patrón principal: captura el núcleo del log dentro de posibles comillas dobles
# Soporta las variantes con prefijo CSV y comillas dobles duplicadas del dataset
_LOG_PATTERN = re.compile(
    r'(?:^\d+,)?'                          # prefijo numérico opcional (CSV)
    r'"?'                                  # comilla de apertura opcional
    r'(?P<ip>\d{1,3}(?:\.\d{1,3}){3})'   # IP del cliente
    r'\s+-\s+-\s+'                         # identd y usuario (siempre -)
    r'\[(?P<timestamp>[^\]]+)\]'           # [timestamp]
    r'\s+""?'                              # separador con comillas dobles
    r'(?P<method>[A-Z]+)\s+'              # METHOD
    r'(?P<path>\S+)\s+'                   # /path
    r'(?P<protocol>HTTP/[\d.]+)'          # HTTP/1.x
    r'""?\s+'                             # cierre de comillas
    r'(?P<status>\d{3})\s+'              # status code
    r'(?P<bytes>\d+|-)'                   # bytes (puede ser -)
    r'(?:\s+""?(?P<referer>[^""]*?)""?'  # referer (opcional)
    r'\s+""?(?P<ua>[^""]*?)""?)?'        # user-agent (opcional)
)


def _parse_line(raw_line: str) -> dict | None:
    """
    Intenta parsear una línea de log con el patrón regex.
    Retorna un dict con los campos capturados, o None si no coincide.
    """
    match = _LOG_PATTERN.search(raw_line)
    if not match:
        return None

    g = match.groupdict()
    bytes_value = g.get("bytes", "0") or "0"

    return {
        "ip":          g.get("ip", ""),
        "timestamp":   g.get("timestamp", ""),
        "method":      g.get("method", ""),
        "path":        g.get("path", ""),
        "protocol":    g.get("protocol", ""),
        "status_code": int(g.get("status", 0) or 0),
        "bytes_sent":  int(bytes_value) if bytes_value.isdigit() else 0,
        "referer":     (g.get("referer") or "").strip('"'),
        "user_agent":  (g.get("ua") or "").strip('"'),
    }


def _iter_raw_lines(log_path: Path) -> Iterator[str]:
    """Itera las líneas del archivo ignorando líneas vacías."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield stripped


# ---------------------------------------------------------------------------
# Función principal de procesamiento
# ---------------------------------------------------------------------------

def preprocess_logs(
    input_path: str | Path,
    output_path: str | Path,
    id_prefix: str = "LOG",
    id_width: int = 5,
    skip_unparseable: bool = False,
) -> int:
    """
    Lee el archivo de logs crudo, parsea cada línea y exporta a JSONL.

    Args:
        input_path:      Ruta al archivo .log crudo.
        output_path:     Ruta de salida para el archivo .jsonl procesado.
        id_prefix:       Prefijo para los log_id (default: "LOG").
        id_width:        Ancho del número en el log_id (default: 5 → LOG-00001).
        skip_unparseable: Si True, omite líneas que no coincidan con el regex.
                          Si False (default), las guarda con campos vacíos.

    Returns:
        Total de registros escritos al archivo de salida.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Archivo de logs no encontrado: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    written = 0

    logger.info(" Leyendo logs desde: %s", input_path)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for total, raw_line in enumerate(
            _iter_raw_lines(input_path), start=1
        ):
            log_id = f"{id_prefix}-{total:0{id_width}d}"
            parsed = _parse_line(raw_line)

            if parsed is None:
                skipped += 1
                if skip_unparseable:
                    logger.debug(" Línea %d sin parsear, omitida.", total)
                    continue
                # Guardar la línea cruda con campos vacíos para no perder trazabilidad
                parsed = {
                    "ip": "", "timestamp": "", "method": "",
                    "path": "", "protocol": "", "status_code": 0,
                    "bytes_sent": 0, "referer": "", "user_agent": "",
                }

            entry = ProcessedLogEntry(
                log_id=log_id,
                raw_text=raw_line,
                **parsed,
            )

            out_f.write(entry.model_dump_json() + "\n")
            written += 1

            if total % 1000 == 0:
                logger.info("  → Procesados: %d líneas...", total)

    logger.info(" Preprocesamiento completado.")
    logger.info("   Total líneas leídas : %d", total)
    logger.info("   Registros escritos  : %d", written)
    logger.info("   Líneas sin parsear  : %d", skipped)
    logger.info("   Salida              : %s", output_path)

    return written


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Módulo 1 — Preprocesamiento de logs y asignación de log_id"
    )
    parser.add_argument(
        "--input",
        default="logs/webServer.log",
        help="Ruta al archivo de logs crudo (default: logs/webServer.log)",
    )
    parser.add_argument(
        "--output",
        default="pseudo_labeling/data/processed_logs.jsonl",
        help="Ruta de salida JSONL (default: pseudo_labeling/data/processed_logs.jsonl)",
    )
    parser.add_argument(
        "--skip-unparseable",
        action="store_true",
        help="Omitir líneas que no coincidan con el patrón (default: guardar con campos vacíos)",
    )
    args = parser.parse_args()

    total_written = preprocess_logs(
        input_path=args.input,
        output_path=args.output,
        skip_unparseable=args.skip_unparseable,
    )
    print(f"\n Listo. {total_written:,} registros exportados → {args.output}")
