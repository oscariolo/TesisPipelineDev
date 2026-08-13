import json

from log_analysis.core.log_entry import BatchAnalysisResult, LogBatch, LogEntry
from log_analysis.models.generative import GenerativeConfig, GenerativeModel
from log_analysis.output.json_writer import JsonWriter


def test_generative_model_works_on_batches(monkeypatch):
    model = GenerativeModel(GenerativeConfig(model_name="demo-model", backend="ollama"))
    monkeypatch.setattr(
        model,
        "_call_ollama",
        lambda prompt: '{"is_error": true, "error_description": "database connection error", "recommended_action": "restart service"}',
    )

    batch = LogBatch(
        batch_id=7,
        entries=[
            LogEntry(index=0, source_file="app.log", raw_text="ERROR: database connection timed out"),
            LogEntry(index=1, source_file="app.log", raw_text="WARN: retries exhausted"),
        ],
    )

    result = model.analyze(batch)

    assert result.batch_id == 7
    assert result.error_found is True


def test_json_writer_emits_batch_summary(tmp_path):
    writer = JsonWriter(tmp_path)
    writer.write(BatchAnalysisResult(batch_id=3, error_found=True))

    payload = json.loads((tmp_path / "log_analysis.jsonl").read_text().strip())
    assert payload == {"batch_id": 3, "error_found": True}
