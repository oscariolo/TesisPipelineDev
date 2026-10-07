import json
import tempfile
import unittest
from pathlib import Path

from log_analysis.core.log_entry import BatchAnalysisResult, LogBatch, LogEntry
from log_analysis.core.log_ingestor import LogIngestor
from log_analysis.core.pipeline import Pipeline
from log_analysis.models.base import BaseModel
from log_analysis.output.json_writer import JsonWriter


class _FakeModel(BaseModel):
    def __init__(self) -> None:
        self.config = type("Config", (), {"model_name": "demo-model"})()

    def analyze(self, batch: LogBatch) -> BatchAnalysisResult:
        return BatchAnalysisResult(
            batch_id=batch.batch_id,
            error_found=True,
            model_name="demo-model",
            is_valid_response=True,
            error_description="boom",
            recommended_action="restart",
        )


class _FakeIngestor(LogIngestor):
    def iter_batches(self):
        yield LogBatch(batch_id=0, entries=[LogEntry(index=0, source_file="a.log", raw_text="ERROR")])
        yield LogBatch(
            batch_id=1,
            entries=[
                LogEntry(index=0, source_file="b.log", raw_text="ok"),
                LogEntry(index=1, source_file="a.log", raw_text="ok"),
            ],
        )


class PipelineTelemetryTest(unittest.TestCase):
    def test_pipeline_writes_telemetry_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Pipeline(_FakeModel(), _FakeIngestor(), JsonWriter(temp_dir)).run()

            path = Path(temp_dir) / "log_analysis.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["entry_count"], 1)
            self.assertEqual(rows[0]["source_files"], ["a.log"])
            self.assertIsInstance(rows[0]["elapsed_seconds"], float)
            self.assertTrue(rows[0]["timestamp"])
            self.assertEqual(rows[0]["error_description"], "boom")
            self.assertEqual(rows[0]["recommended_action"], "restart")
            self.assertIsNone(rows[0]["embedding_hit"])

            self.assertEqual(rows[1]["entry_count"], 2)
            self.assertEqual(rows[1]["source_files"], ["a.log", "b.log"])
            self.assertTrue(all(row["run_id"] for row in rows))

    def test_pipeline_writes_run_file_with_config_and_totals(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_info = {
                "mode": "generative",
                "backend": "ollama",
                "model_name": "demo-model",
                "batch_size": 100,
                "log_dir": "./logs",
                "expected_batches": 2,
            }
            Pipeline(
                _FakeModel(), _FakeIngestor(), JsonWriter(temp_dir),
                run_id="testrun", run_info=run_info,
            ).run()

            run_file = Path(temp_dir) / "run_testrun.json"
            self.assertTrue(run_file.exists())
            payload = json.loads(run_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "testrun")
            self.assertEqual(payload["status"], "finished")
            self.assertEqual(payload["run_info"]["expected_batches"], 2)
            self.assertEqual(payload["run_info"]["model_name"], "demo-model")
            self.assertEqual(payload["totals"]["batches"], 2)
            self.assertEqual(payload["totals"]["errors"], 2)

            rows = [
                json.loads(line)
                for line in (Path(temp_dir) / "log_analysis.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(all(row["run_id"] == "testrun" for row in rows))


if __name__ == "__main__":
    unittest.main()
