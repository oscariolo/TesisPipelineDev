import logging

from log_analysis.core.log_entry import AnalysisResult
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
        for entry in self.ingestor.ingest():
            try:
                result = self.model.analyze(entry)
            except Exception as e:
                logger.error("Error analyzing log #%d: %s", entry.index, e)
                result = AnalysisResult(
                    log_index=entry.index,
                    is_error=True,
                    error_description=f"Pipeline error: {e}",
                    recommended_action="Check pipeline logs for details",
                )
            self.writer.write(result)
            logger.info("Processed log #%d (%s)", entry.index, entry.source_file)
