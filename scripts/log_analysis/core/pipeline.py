import logging
import time

from log_analysis.core.log_entry import BatchAnalysisResult, LogBatch
from log_analysis.core.log_ingestor import LogIngestor
from log_analysis.models.base import BaseModel
from log_analysis.output.json_writer import JsonWriter

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, model: BaseModel, ingestor: LogIngestor, writer: JsonWriter):
        self.model = model
        self.ingestor = ingestor
        self.writer = writer

    def run(self) -> None:
        total_logs = 0
        for batch in self.ingestor.iter_batches():
            result = self._analyze_batch(batch)
            self.writer.write(result)
            total_logs += len(batch.entries)
            logger.info(
                "Processed batch #%d (%d logs, error_found=%s)",
                batch.batch_id,
                len(batch.entries),
                result.error_found,
            )

    def _analyze_batch(self, batch: LogBatch) -> BatchAnalysisResult:
        start = time.perf_counter()
        try:
            result = self.model.analyze(batch)
        except Exception as e:
            logger.error("Error analyzing batch #%d: %s", batch.batch_id, e)
            result = BatchAnalysisResult(
                batch_id=batch.batch_id,
                error_found=True,
                model_name=self.model.config.generative.model_name if hasattr(self.model, "config") else None,
                embedder_model_name=self.model.config.embedder_model_name if hasattr(self.model, "config") else None,
            )

        elapsed = time.perf_counter() - start
        logger.info(
            "Batch #%d complete in %.3fs: error_found=%s",
            batch.batch_id,
            elapsed,
            result.error_found,
        )
        return result