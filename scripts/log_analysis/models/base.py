from abc import ABC, abstractmethod

from log_analysis.core.log_entry import LogEntry, AnalysisResult


class BaseModel(ABC):
    @abstractmethod
    def analyze(self, entry: LogEntry) -> AnalysisResult:
        ...
