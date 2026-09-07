import logging
import time
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator
import json

import httpx

from log_analysis.core.log_entry import LogBatch, LogEntry

logger = logging.getLogger(__name__)


class LogIngestor(ABC):
    def __init__(self, batch_size: int = 100):
        self.batch_size = max(1, batch_size)

    @abstractmethod
    def iter_batches(self) -> Generator[LogBatch, None, None]:
        ...


class FileLogIngestor(LogIngestor):
    def __init__(self, log_dir: str | Path, batch_size: int = 100, masker=None):
        super().__init__(batch_size)
        self.log_dir = Path(log_dir)
        self.masker = masker

    def iter_batches(self) -> Generator[LogBatch, None, None]:
        file_paths = sorted(self.log_dir.glob("*.log"))
        if not file_paths:
            raise FileNotFoundError(f"Not found in {self.log_dir}")

        batch_id = 0
        entries: list[LogEntry] = []
        global_idx = 0
        for file_path in file_paths:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.rstrip("\n").rstrip("\r")
                    if not line:
                        continue
                    entries.append(
                        LogEntry(
                            index=global_idx,
                            source_file=file_path.name,
                            raw_text=self.masker.mask(line) if self.masker else line,
                        )
                    )
                    global_idx += 1
                    if len(entries) >= self.batch_size:
                        yield LogBatch(batch_id=batch_id, entries=entries)
                        batch_id += 1
                        entries = []
        if entries:
            yield LogBatch(batch_id=batch_id, entries=entries)


class StreamLogIngestor(LogIngestor):
    def __init__(self, url: str, batch_size: int = 100, poll_interval: float = 5.0, masker=None):
        super().__init__(batch_size)
        self.url = url
        self.poll_interval = max(0.0, poll_interval)
        self.masker = masker

    def iter_batches(self) -> Generator[LogBatch, None, None]:
        batch_id = 0
        global_idx = 0
        while True:
            entries: list[LogEntry] = []
            try:
                with httpx.stream("GET", self.url) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        entries.append(
                            LogEntry(
                                index=global_idx,
                                source_file=self.url,
                                raw_text=self.masker.mask(line) if self.masker else line,
                            )
                        )
                        global_idx += 1
                        if len(entries) >= self.batch_size:
                            yield LogBatch(batch_id=batch_id, entries=entries)
                            batch_id += 1
                            entries = []
            except httpx.HTTPError as e:
                logger.warning(
                    "Stream read failed for %s: %s — retrying in %.1fs",
                    self.url,
                    e,
                    self.poll_interval,
                )
            if entries:
                yield LogBatch(batch_id=batch_id, entries=entries)
                batch_id += 1
            logger.info("Batch exhausted — polling %s again in %.1fs", self.url, self.poll_interval)
            time.sleep(self.poll_interval)


class JsonLogIngestor(LogIngestor):
    """
    Ingestor diseñado para leer archivos JSON procesados previamente (ej. por parse_and_label.py),
    extrayendo unicamente el 'template' del log para ahorrar tokens en el SLM.
    """
    def __init__(self, json_path: str | Path, batch_size: int = 100):
        super().__init__(batch_size)
        self.json_path = Path(json_path)

    def iter_batches(self) -> Generator[LogBatch, None, None]:
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
            
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        batch_id = 0
        entries: list[LogEntry] = []
        global_idx = 0
        
        for item in data:
            # Extraer de tu estructura JSON el 'template'
            # (fallback a 'raw_message' por si el json esta incompleto)
            try:
                log_data = item.get("log_data", {})
                template_text = log_data.get("template") or log_data.get("raw_message", "")
                if not template_text:
                    continue
            except AttributeError:
                # En caso de que un item no sea un diccionario valido
                continue
                
            entries.append(
                LogEntry(
                    index=global_idx,
                    source_file=self.json_path.name,
                    raw_text=template_text,
                )
            )
            global_idx += 1
            if len(entries) >= self.batch_size:
                yield LogBatch(batch_id=batch_id, entries=entries)
                batch_id += 1
                entries = []
                
        if entries:
            yield LogBatch(batch_id=batch_id, entries=entries)

class LogMasker:
    """Clase para ocultar variables en los logs a conveniencia."""
    def __init__(self, mask_ips: bool = True, mask_uuids: bool = True, mask_numbers: bool = True):
        self.mask_ips = mask_ips
        self.mask_uuids = mask_uuids
        self.mask_numbers = mask_numbers

    def mask(self, text: str) -> str:
        if self.mask_ips:
            text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', text)
        if self.mask_uuids:
            text = re.sub(r'\b[a-fA-F0-9]{8}(-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}\b', '<UUID>', text)
        if self.mask_numbers:
            text = re.sub(r'\b\d+\b', '<NUM>', text)
        return text