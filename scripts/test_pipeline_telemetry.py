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


class _RaisingModel(BaseModel):
    """Analyzer that blows up, to exercise the failure register."""

    def __init__(self) -> None:
        self.config = type("Config", (), {"model_name": "demo-model"})()

    def analyze(self, batch: LogBatch) -> BatchAnalysisResult:
        raise RuntimeError("GPU exploded")


class _GarbageModel(BaseModel):
    """Analyzer that answers with something the parser cannot read."""

    def __init__(self) -> None:
        self.config = type("Config", (), {"model_name": "demo-model"})()

    def analyze(self, batch: LogBatch) -> BatchAnalysisResult:
        return BatchAnalysisResult(
            batch_id=batch.batch_id,
            error_found=False,
            is_valid_response=False,
            raw_response='{"oops": "not the expected shape"}',
            model_name="demo-model",
        )


class FailureRegisterTest(unittest.TestCase):
    def _register(self, temp_dir: str) -> list[dict]:
        path = Path(temp_dir) / "failed_batches.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_a_raising_model_is_registered_with_reason_and_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Pipeline(_RaisingModel(), _FakeIngestor(), JsonWriter(temp_dir), run_id="r1").run()

            records = self._register(temp_dir)
            self.assertEqual(len(records), 2)                      # both batches raised
            record = records[0]
            self.assertEqual(record["failure"], "exception:RuntimeError")
            self.assertEqual(record["error_type"], "RuntimeError")
            self.assertEqual(record["error_message"], "GPU exploded")
            self.assertEqual(record["batch_id"], 0)
            self.assertEqual(record["run_id"], "r1")
            self.assertEqual(record["entry_count"], 1)
            self.assertEqual(record["source_files"], ["a.log"])
            self.assertIsNone(record["raw_response"])
            # the input the model never got to see
            self.assertIn("[0] ERROR", record["input"])
            self.assertEqual(record["input_chars"], len(record["input"]))

            run_file = json.loads((Path(temp_dir) / "run_r1.json").read_text(encoding="utf-8"))
            self.assertEqual(run_file["totals"]["failed"], 2)      # both batches raised
            self.assertEqual(run_file["totals"]["invalid"], 0)     # a crash is not an invalid answer

    def test_an_unparseable_answer_is_registered_with_the_raw_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Pipeline(_GarbageModel(), _FakeIngestor(), JsonWriter(temp_dir), run_id="r2").run()

            records = self._register(temp_dir)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["failure"], "invalid_response")
            self.assertIsNone(records[0]["error_type"])
            self.assertIn("not the expected shape", records[0]["raw_response"])
            self.assertFalse(records[0]["is_valid_response"])

            run_file = json.loads((Path(temp_dir) / "run_r2.json").read_text(encoding="utf-8"))
            self.assertEqual(run_file["totals"]["invalid"], 2)
            self.assertEqual(run_file["totals"]["failed"], 0)

    def test_no_register_file_when_nothing_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Pipeline(_FakeModel(), _FakeIngestor(), JsonWriter(temp_dir)).run()

            self.assertEqual(self._register(temp_dir), [])
            self.assertFalse((Path(temp_dir) / "failed_batches.jsonl").exists())

    def test_the_input_can_be_left_out(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Pipeline(_RaisingModel(), _FakeIngestor(), JsonWriter(temp_dir),
                     failures_input_chars=0).run()

            record = self._register(temp_dir)[0]
            self.assertIsNone(record["input"])
            self.assertGreater(record["input_chars"], 0)          # length still reported

    def test_the_stored_input_is_capped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Pipeline(_RaisingModel(), _FakeIngestor(), JsonWriter(temp_dir),
                     failures_input_chars=5).run()

            record = self._register(temp_dir)[0]
            self.assertTrue(record["input"].startswith("[0] E"))
            self.assertIn("chars truncated", record["input"])

    def test_a_custom_register_path_is_honoured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "nested" / "custom.jsonl"
            Pipeline(_RaisingModel(), _FakeIngestor(), JsonWriter(Path(temp_dir) / "out"),
                     failures_path=target).run()

            self.assertTrue(target.exists())
            rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 2)
            self.assertFalse((Path(temp_dir) / "out" / "failed_batches.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
