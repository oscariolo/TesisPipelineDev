from log_analysis.core.log_entry import LogEntry, AnalysisResult
from log_analysis.models.base import BaseModel


class EmbeddingModel(BaseModel):
    def analyze(self, entry: LogEntry) -> AnalysisResult:
        raise NotImplementedError("Embedding approach not yet implemented")
