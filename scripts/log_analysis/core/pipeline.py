import logging
from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from log_analysis.core.log_entry import AnalysisResult
from log_analysis.core.log_ingestor import LogIngestor
from log_analysis.models.base import BaseModel
from log_analysis.output.json_writer import JsonWriter

logger = logging.getLogger(__name__)

tracer = trace.get_tracer("log-analysis.pipeline")
meter = metrics.get_meter("log-analysis.pipeline")

logs_analyzed = meter.create_counter("pipeline.logs.analyzed", unit="1")
logs_errored = meter.create_counter("pipeline.logs.errors", unit="1")
analysis_duration = meter.create_histogram(
    "pipeline.analysis.duration", unit="s", description="Per-log analysis duration"
)


class Pipeline:
    def __init__(self, model: BaseModel, ingestor: LogIngestor, writer: JsonWriter):
        self.model = model
        self.ingestor = ingestor
        self.writer = writer

    def run(self) -> None:
        with tracer.start_as_current_span("pipeline.run") as run_span:
            entries = list(self.ingestor.ingest())
            run_span.set_attribute("log.count", len(entries))

            for entry in entries:
                try:
                    with tracer.start_as_current_span("pipeline.analyze") as span:
                        span.set_attributes({
                            "log.index": entry.index,
                            "log.source_file": entry.source_file,
                            "log.text_length": len(entry.raw_text),
                        })
                        result = self.model.analyze(entry)
                        span.set_attribute("log.is_error", result.is_error)
                        logs_analyzed.add(1)
                        if result.is_error:
                            logs_errored.add(1)
                            logger.warning(
                                "Error detected in log #%d: %s",
                                entry.index,
                                result.error_description,
                            )
                except Exception as e:
                    span = trace.get_current_span()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.error("Error analyzing log #%d: %s", entry.index, e)
                    result = AnalysisResult(
                        log_index=entry.index,
                        is_error=True,
                        error_description=f"Pipeline error: {e}",
                        recommended_action="Check pipeline logs for details",
                    )
                self.writer.write(result)
                logger.info("Processed log #%d (%s)", entry.index, entry.source_file)
