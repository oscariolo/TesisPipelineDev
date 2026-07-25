from pathlib import Path

from log_analysis.core.log_entry import AnalysisResult


class JsonWriter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, result: AnalysisResult) -> None:
        file_path = self.output_dir / "log_analysis.jsonl"
        with open(file_path, "a") as f:
            f.write(result.model_dump_json() + "\n")
