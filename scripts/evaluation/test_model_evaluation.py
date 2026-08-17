import importlib.util
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


if __name__ == "__main__":
    unittest.main()
