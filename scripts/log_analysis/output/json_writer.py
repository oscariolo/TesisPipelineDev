import json
from pathlib import Path

from log_analysis.core.log_entry import BatchAnalysisResult


class JsonWriter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, result: BatchAnalysisResult) -> None:
        summary = result.to_summary()

        jsonl_path = self.output_dir / "log_analysis.jsonl"
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(summary, separators=(",", ":")) + "\n")

        json_path = self.output_dir / "log_analysis.json"
        existing = []
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = f.read().strip()
                    if data:
                        existing = json.loads(data)
            except (json.JSONDecodeError, OSError):
                existing = []
        if not isinstance(existing, list):
            existing = []
        existing.append(summary)
        with open(json_path, "w") as f:
            json.dump(existing, f, separators=(",", ":"))
