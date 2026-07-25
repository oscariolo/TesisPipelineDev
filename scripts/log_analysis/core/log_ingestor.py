from pathlib import Path
from typing import Generator

from log_analysis.core.log_entry import LogEntry


class LogIngestor:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)

    def ingest(self) -> Generator[LogEntry, None, None]:
        file_paths = sorted(self.log_dir.glob("*.log"))
        if not file_paths:
            raise FileNotFoundError(f"No .log files found in {self.log_dir}")

        global_idx = 0
        for file_path in file_paths:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.rstrip("\n").rstrip("\r")
                    if not line:
                        continue
                    yield LogEntry(
                        index=global_idx,
                        source_file=file_path.name,
                        raw_text=line,
                    )
                    global_idx += 1
