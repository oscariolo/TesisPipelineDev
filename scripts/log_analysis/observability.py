import logging
import atexit
from opentelemetry import _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

OTLP_ENDPOINT = "http://localhost:4318"
LOCAL_PROJECT_ID = "f6a30e72-e9fc-42ef-88d9-c1b2c281cfbe"

HEADERS = {
    "x-superlog-project-id": LOCAL_PROJECT_ID,
}

INITIALIZED = False


def _flush_all() -> None:
    trace.get_tracer_provider().force_flush()
    metrics.get_meter_provider().force_flush()
    _logs.get_logger_provider().force_flush()


def init_observability() -> None:
    global INITIALIZED
    if INITIALIZED:
        return
    INITIALIZED = True

    resource = Resource.create({
        "service.name": "log-analysis-pipeline",
        "service.version": "0.1.0",
        "deployment.environment.name": "development",
    })

    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{OTLP_ENDPOINT}/v1/traces", headers=HEADERS))
    )
    trace.set_tracer_provider(trace_provider)

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{OTLP_ENDPOINT}/v1/metrics", headers=HEADERS)
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{OTLP_ENDPOINT}/v1/logs", headers=HEADERS))
    )
    _logs.set_logger_provider(logger_provider)

    LoggingInstrumentor().instrument(set_logging_format=True, log_level=logging.INFO)

    atexit.register(_flush_all)

    logging.info("Observability initialized — exporting to %s", OTLP_ENDPOINT)
