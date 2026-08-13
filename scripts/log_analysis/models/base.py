from abc import ABC, abstractmethod

from log_analysis.core.log_entry import LogBatch, BatchAnalysisResult


class BaseModel(ABC):
    @abstractmethod
    def analyze(self, batch: LogBatch) -> BatchAnalysisResult:
        ...
