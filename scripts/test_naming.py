import unittest
from pathlib import Path

from log_analysis.core.naming import model_output_dir, sanitize_model_name


class SanitizeModelNameTest(unittest.TestCase):
    def test_repo_id_becomes_a_single_folder_name(self):
        self.assertEqual(
            sanitize_model_name("mradermacher/llama-3.2-1B-log-analyzer-GGUF"),
            "mradermacher-llama-3.2-1B-log-analyzer-GGUF",
        )
        self.assertEqual(sanitize_model_name("Qwen/Qwen3-0.6B"), "Qwen-Qwen3-0.6B")
        self.assertEqual(sanitize_model_name("qwen3.6:latest"), "qwen3.6-latest")

    def test_empty_name_falls_back(self):
        self.assertEqual(sanitize_model_name("///"), "unknown-model")


class ModelOutputDirTest(unittest.TestCase):
    def test_folder_is_under_the_analysis_dir(self):
        self.assertEqual(
            model_output_dir("analysis", "Qwen/Qwen3-0.6B"),
            Path("analysis") / "Qwen-Qwen3-0.6B",
        )

    def test_custom_parent_still_gets_the_model_folder(self):
        self.assertEqual(
            model_output_dir("/tmp/base", "Qwen/Qwen3-0.6B"),
            Path("/tmp/base") / "Qwen-Qwen3-0.6B",
        )

    def test_no_model_name_uses_the_analysis_dir(self):
        self.assertEqual(model_output_dir("analysis", None), Path("analysis"))


if __name__ == "__main__":
    unittest.main()
