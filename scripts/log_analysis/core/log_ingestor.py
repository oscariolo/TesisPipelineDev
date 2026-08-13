import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

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
    def __init__(self, log_dir: str | Path, batch_size: int = 100):
        super().__init__(batch_size)
        self.log_dir = Path(log_dir)

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
                            raw_text=line,
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
    def __init__(self, url: str, batch_size: int = 100, poll_interval: float = 5.0):
        super().__init__(batch_size)
        self.url = url
        self.poll_interval = max(0.0, poll_interval)

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
                                raw_text=line,
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