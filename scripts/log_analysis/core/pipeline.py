import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from log_analysis.core.log_entry import BatchAnalysisResult, LogBatch
from log_analysis.core.log_ingestor import LogIngestor
from log_analysis.models.base import BaseModel
from log_analysis.output.json_writer import JsonWriter

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        model: BaseModel,
        ingestor: LogIngestor,
        writer: JsonWriter,
        max_batches: int | None = None,
        run_id: str | None = None,
        run_info: dict | None = None,
    ):
        self.model = model
        self.ingestor = ingestor
        self.writer = writer
        self.max_batches = max_batches
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_info = dict(run_info or {})
        self._run_path = Path(getattr(writer, "output_dir", ".")) / f"run_{self.run_id}.json"
        self._stats = {
            "batches": 0,
            "entries": 0,
            "errors": 0,
            "invalid": 0,
            "embedding_hits": 0,
            "embedding_misses": 0,
            "elapsed_seconds": 0.0,
        }
        self._started_at: str | None = None

    def run(self) -> None:
        self._started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._write_run_file("running")
        try:
            for batch in self.ingestor.iter_batches():
                if self.max_batches is not None and batch.batch_id >= self.max_batches:
                    logger.info("Reached max_batches (%d). Stopping pipeline.", self.max_batches)
                    break
                result = self._analyze_batch(batch)
                self.writer.write(result)
                self._record(result)
                logger.info(
                    "Processed batch #%d (%d logs, error_found=%s)",
                    batch.batch_id,
                    len(batch.entries),
                    result.error_found,
                )
        finally:
            self._write_run_file("finished")

    def _record(self, result: BatchAnalysisResult) -> None:
        self._stats["batches"] += 1
        self._stats["entries"] += result.entry_count or 0
        if result.error_found:
            self._stats["errors"] += 1
        if result.is_valid_response is False:
            self._stats["invalid"] += 1
        if result.embedding_hit is True:
            self._stats["embedding_hits"] += 1
        elif result.embedding_hit is False:
            self._stats["embedding_misses"] += 1
        if result.elapsed_seconds is not None:
            self._stats["elapsed_seconds"] = round(self._stats["elapsed_seconds"] + result.elapsed_seconds, 4)

    def _write_run_file(self, status: str) -> None:
        """Write the per-run status/config file the dashboard reads (running -> finished)."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload = {
            "run_id": self.run_id,
            "status": status,
            "started_at": self._started_at,
            "updated_at": now,
            "run_info": self.run_info,
            "totals": dict(self._stats),
        }
        if status == "finished":
            payload["finished_at"] = now
        try:
            self._run_path.parent.mkdir(parents=True, exist_ok=True)
            self._run_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as e:
            logger.warning("Could not write run file %s: %s", self._run_path, e)

    def _analyze_batch(self, batch: LogBatch) -> BatchAnalysisResult:
        start = time.perf_counter()
        try:
            result = self.model.analyze(batch)
        except Exception as e:
            if torch is not None and isinstance(e, torch.OutOfMemoryError):
                logger.exception(
                    "CUDA OOM analyzing batch #%d (%d entries); "
                    "the batch may exceed GPU memory — try reducing --batch-size",
                    batch.batch_id, len(batch.entries),
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                logger.exception("Error analyzing batch #%d: %s", batch.batch_id, e)
            result = BatchAnalysisResult(
                batch_id=batch.batch_id,
                error_found=True,
                model_name=getattr(self.model.config, "model_name", None),
                token_usage=0,
            )

        elapsed = time.perf_counter() - start
        result.elapsed_seconds = round(elapsed, 4)
        result.entry_count = len(batch.entries)
        result.source_files = sorted({entry.source_file for entry in batch.entries})
        result.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        result.run_id = self.run_id
        return result