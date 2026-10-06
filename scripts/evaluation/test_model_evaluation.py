import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("modelEvaluation.py")
SPEC = importlib.util.spec_from_file_location("modelEvaluation", MODULE_PATH)
MODEL_EVALUATION_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODEL_EVALUATION_MODULE)
ModelEvaluator = MODEL_EVALUATION_MODULE.ModelEvaluator


class ModelEvaluatorTest(unittest.TestCase):
    def test_accuracy_and_confusion_matrix_for_binary_labels(self):
        evaluator = ModelEvaluator(label_key="error_found")
        target = [False, True, True, False]
        prediction = [False, True, False, False]

        metrics = evaluator.evaluate(target, prediction)

        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["f1_score"], 0.6666666666666666)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [1, 1]])

    def test_compare_files_uses_same_file_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "sample.jsonl"
            rows = [
                {"batch_id": 1, "error_found": False},
                {"batch_id": 2, "error_found": True},
                {"batch_id": 3, "error_found": True},
                {"batch_id": 4, "error_found": False},
            ]

            with data_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(f"{row}\n")

            evaluator = ModelEvaluator()
            summary = evaluator.compare_files(data_path, data_path)

            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["f1_score"], 1.0)
            self.assertEqual(summary["confusion_matrix"], [[2, 0], [0, 2]])

    def test_save_report_includes_model_metadata_and_positive_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "nested" / "report.json"
            evaluator = ModelEvaluator(label_key="error_found")
            target = [False, True, True, False]
            prediction = [False, True, False, False]
            metrics = evaluator.evaluate(target, prediction)

            evaluator.save_report(
                metrics,
                report_path,
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                embedder_model_name="sentence-transformers/all-MiniLM-L6-v2",
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["model_name"], "Qwen/Qwen2.5-1.5B-Instruct")
            self.assertEqual(report["embedder_model_name"], "sentence-transformers/all-MiniLM-L6-v2")
            # precision/recall/f1 must target the positive (error_found) class
            self.assertIsInstance(report["precision"], float)
            self.assertEqual(report["precision"], 1.0)
            self.assertEqual(report["recall"], 0.5)
            self.assertAlmostEqual(report["f1_score"], 0.6666666666666666)

    def test_read_model_metadata_skips_records_without_model_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "log_analysis.jsonl"
            path.write_text(
                json.dumps({"batch_id": 1, "raw_response": "not json"}) + "\n"
                + json.dumps({
                    "batch_id": 2,
                    "error_found": True,
                    "model_name": "qwen3.6:latest",
                    "embedder_model_name": "sentence-transformers/all-MiniLM-L6-v2",
                }) + "\n",
                encoding="utf-8",
            )

            metadata = ModelEvaluator().read_model_metadata(path)

            self.assertEqual(metadata["model_name"], "qwen3.6:latest")
            self.assertEqual(metadata["embedder_model_name"], "sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    unittest.main()
