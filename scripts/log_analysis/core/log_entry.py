from pydantic import BaseModel
from typing import Any, Optional


class LogEntry(BaseModel):
    index: int
    source_file: str
    raw_text: str


class LogBatch(BaseModel):
    batch_id: int
    entries: list[LogEntry]


class AnalysisResult(BaseModel):
    log_index: int
    error_description: Optional[str] = None
    recommended_action: Optional[str] = None


class BatchAnalysisResult(BaseModel):
    batch_id: int
    error_found: bool = False
    model_name: Optional[str] = None
    embedder_model_name: Optional[str] = None
    token_usage: Optional[int] = None
    is_valid_response: Optional[bool] = False
    raw_response: Optional[str] = None
    # Pipeline telemetry (optional so existing records still parse).
    run_id: Optional[str] = None
    timestamp: Optional[str] = None
    entry_count: Optional[int] = None
    source_files: Optional[list[str]] = None
    elapsed_seconds: Optional[float] = None
    embedding_hit: Optional[bool] = None
    similarity: Optional[float] = None
    error_description: Optional[str] = None
    recommended_action: Optional[str] = None

    @property
    def is_error(self) -> bool:
        return self.error_found

    @is_error.setter
    def is_error(self, value: bool) -> None:
        self.error_found = value

    def to_summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "batch_id": self.batch_id,
            "timestamp": self.timestamp,
            "error_found": self.error_found,
            "model_name": self.model_name,
            "embedder_model_name": self.embedder_model_name,
            "token_usage": self.token_usage,
            "is_valid_response": self.is_valid_response,
            "entry_count": self.entry_count,
            "source_files": self.source_files,
            "elapsed_seconds": self.elapsed_seconds,
            "embedding_hit": self.embedding_hit,
            "similarity": self.similarity,
            "error_description": self.error_description,
            "recommended_action": self.recommended_action,
        }
