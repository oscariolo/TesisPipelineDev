import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from log_analysis.core.log_entry import BatchAnalysisResult, LogBatch, render_batch_text
from log_analysis.core.log_ingestor import LogIngestor
from log_analysis.models.base import BaseModel
from log_analysis.output.json_writer import JsonWriter

logger = logging.getLogger(__name__)

# Failure register: full input up to this many chars (0 = metadata only, negative = unlimited), plus
# smaller caps for the exception text and the model's own answer.
DEFAULT_FAILURES_INPUT_CHARS = 20000
FAILURES_RAW_RESPONSE_CHARS = 2000
FAILURES_MESSAGE_CHARS = 500


def _truncate(text: str | None, limit: int | None) -> str | None:
    """Trim a value for the failure register; a negative/None limit means no cap."""
    if text is None:
        return None
    if limit is None or limit < 0 or len(text) <= limit:
        return text
    return text[:limit] + f"…[{len(text) - limit} chars truncated]"


class Pipeline:
    def __init__(
        self,
        model: BaseModel,
        ingestor: LogIngestor,
        writer: JsonWriter,
        max_batches: int | None = None,
        run_id: str | None = None,
        run_info: dict | None = None,
        failures_path: str | Path | None = None,
        failures_input_chars: int = DEFAULT_FAILURES_INPUT_CHARS,
    ):
        self.model = model
        self.ingestor = ingestor
        self.writer = writer
        self.max_batches = max_batches
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_info = dict(run_info or {})
        self._run_path = Path(getattr(writer, "output_dir", ".")) / f"run_{self.run_id}.json"
        self.failures_input_chars = failures_input_chars
        # Register of batches the pipeline could not process (created on the first failure).
        self._failures_path = Path(failures_path) if failures_path else (
            Path(getattr(writer, "output_dir", ".")) / "failed_batches.jsonl"
        )
        self._stats = {
            "batches": 0,
            "entries": 0,
            "errors": 0,
            "invalid": 0,
            "failed": 0,
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
                result, failure = self._analyze_batch(batch)
                self.writer.write(result)
                self._record(result, failure)
                logger.info(
                    "Processed batch #%d (%d logs, error_found=%s)",
                    batch.batch_id,
                    len(batch.entries),
                    result.error_found,
                )
        finally:
            self._write_run_file("finished")

    def _record(self, result: BatchAnalysisResult, failure: str | None = None) -> None:
        self._stats["batches"] += 1
        self._stats["entries"] += result.entry_count or 0
        if result.error_found:
            self._stats["errors"] += 1
        if failure is None:
            if result.is_valid_response is False:
                self._stats["invalid"] += 1
        elif failure == "invalid_response":
            self._stats["invalid"] += 1
        else:
            # A raised call is a failure of the run, not an unparseable answer.
            self._stats["failed"] += 1
        if result.embedding_hit is True:
            self._stats["embedding_hits"] += 1
        elif result.embedding_hit is False:
            self._stats["embedding_misses"] += 1
        if result.elapsed_seconds is not None:
            self._stats["elapsed_seconds"] = round(self._stats["elapsed_seconds"] + result.elapsed_seconds, 4)

    def _record_failure(self, batch: LogBatch, result: BatchAnalysisResult, failure: str,
                        error_type: str | None = None, error_message: str | None = None) -> None:
        """Append one line to the failure register: what failed, why, and the input behind it."""
        text = render_batch_text(batch)
        record = {
            "run_id": self.run_id,
            "batch_id": batch.batch_id,
            "timestamp": result.timestamp,
            "failure": failure,
            "error_type": error_type,
            "error_message": _truncate(error_message, FAILURES_MESSAGE_CHARS),
            "is_valid_response": result.is_valid_response,
            "raw_response": _truncate(result.raw_response, FAILURES_RAW_RESPONSE_CHARS),
            "entry_count": result.entry_count,
            "source_files": result.source_files,
            "token_usage": result.token_usage,
            "elapsed_seconds": result.elapsed_seconds,
            "input_chars": len(text),
            "input": _truncate(text, self.failures_input_chars) if self.failures_input_chars != 0 else None,
        }
        try:
            self._failures_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._failures_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.warning("Could not write the failure register %s: %s", self._failures_path, e)

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
        # Event-filter counters (kept/total, per-rule, keep ratio) when a log filter is in use.
        filter_stats = getattr(self.ingestor, "filter_stats", None)
        if filter_stats is not None:
            payload["filter"] = filter_stats.as_dict()
        if status == "finished":
            payload["finished_at"] = now
        try:
            self._run_path.parent.mkdir(parents=True, exist_ok=True)
            self._run_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        except OSError as e:
            logger.warning("Could not write run file %s: %s", self._run_path, e)

    def _analyze_batch(self, batch: LogBatch) -> tuple[BatchAnalysisResult, str | None]:
        """Analyze one batch and never raise. Returns the result and, when it failed, why."""
        start = time.perf_counter()
        failure: str | None = None
        error_type: str | None = None
        error_message: str | None = None
        try:
            result = self.model.analyze(batch)
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            failure = f"exception:{error_type}"
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

        if failure is None and result.is_valid_response is False:
            failure = "invalid_response"
        if failure is not None:
            self._record_failure(batch, result, failure, error_type, error_message)
        return result, failure