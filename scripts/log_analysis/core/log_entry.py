from pydantic import BaseModel
from typing import Optional


class LogEntry(BaseModel):
    index: int
    source_file: str
    raw_text: str


class LogBatch(BaseModel):
    batch_id: int
    entries: list[LogEntry]


class AnalysisResult(BaseModel):
    log_index: int
    is_error: bool
    error_description: Optional[str] = None
    recommended_action: Optional[str] = None


class BatchAnalysisResult(BaseModel):
    batch_id: int
    error_found: bool = False
    error_count: int = 0
    results: list[AnalysisResult] = []

    @property
    def is_error(self) -> bool:
        return self.error_found

    @is_error.setter
    def is_error(self, value: bool) -> None:
        self.error_found = value

    def to_summary(self) -> dict[str, bool | int]:
        return {
            "batch_id": self.batch_id,
            "error_found": self.error_found,
        }
