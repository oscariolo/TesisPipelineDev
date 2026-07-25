from pydantic import BaseModel
from typing import Optional


class LogEntry(BaseModel):
    index: int
    source_file: str
    raw_text: str


class AnalysisResult(BaseModel):
    log_index: int
    is_error: bool
    error_description: Optional[str] = None
    recommended_action: Optional[str] = None
