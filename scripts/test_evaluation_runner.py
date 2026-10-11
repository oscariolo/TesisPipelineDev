import json
import tempfile
import unittest
from pathlib import Path

from evaluation.model_evaluation_viz import run_evaluation


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class RunEvaluationTest(unittest.TestCase):
    def test_metrics_report_and_plots_land_in_one_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.jsonl"
            comparison = root / "comparison.jsonl"
            _write_jsonl(reference, [{"error_found": True}, {"error_found": False}])
            _write_jsonl(comparison, [
                {"run_id": "R1", "error_found": True, "model_name": "org/model"},
                {"run_id": "R1", "error_found": False, "model_name": "org/model"},
                {"run_id": "OTHER", "error_found": False, "model_name": "org/model"},
            ])

            out = root / "out"
            result = run_evaluation(
                reference_file=reference,
                comparison_file=comparison,
                output_dir=out,
                model_name="org/model",
                batch_size=None,
                run_id="R1",
            )

            self.assertEqual(Path(result["output_dir"]), out)
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["unclassified_file"]).exists())
            self.assertGreaterEqual(len(result["plots"]), 1)

            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["model_name"], "org/model")
            # Provenance: the report records exactly which files it came from.
            self.assertEqual(Path(report["reference_file"]), reference.resolve())
            self.assertEqual(Path(report["comparison_file"]), comparison.resolve())
            self.assertEqual(report["run_id"], "R1")
            # Only run R1 was compared, so the other run's batch is excluded.
            self.assertEqual(report["total_samples"], 2)
            self.assertEqual(report["accuracy"], 1.0)
            # No continuous confidence source -> no ROC data.
            self.assertNotIn("roc_auc", report)

    def test_uses_the_folder_and_model_name_it_is_given(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference.jsonl"
            comparison = root / "comparison.jsonl"
            _write_jsonl(reference, [{"error_found": True}])
            # The comparison file names a different model; the explicit argument must win.
            _write_jsonl(comparison, [{"error_found": True, "model_name": "someone-else/model"}])

            out = root / "explicit-folder"
            result = run_evaluation(
                reference_file=reference,
                comparison_file=comparison,
                output_dir=out,
                model_name="Qwen/Qwen3-0.6B",
                batch_size=None,
            )

            self.assertEqual(Path(result["output_dir"]), out)
            report = json.loads((out / "eval_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["model_name"], "Qwen/Qwen3-0.6B")


if __name__ == "__main__":
    unittest.main()
