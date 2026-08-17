from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    zero_one_loss,
)


class ModelEvaluator:
    """Evaluate model predictions using scikit-learn's standard metrics.

    The defaults are safe and explicit: binary labels are coerced to 0/1, and all metric
    calculations follow scikit-learn conventions so they are easy to adjust later by changing
    the average, pos_label, or labels arguments.
    """

    def __init__(
        self,
        positive_label: Any = True,
        negative_label: Any = False,
        label_key: str = "error_found",
        labels: Sequence[int] = (0, 1),
    ) -> None:
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.label_key = label_key
        self.labels = tuple(labels)

    def _as_binary(self, value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return 1 if float(value) == 1.0 else 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            truthy = {"1", "true", "yes", "y", "error", "error_found", "positive"}
            falsy = {"0", "false", "no", "n", "ok", "normal", "negative"}
            if normalized in truthy:
                return 1
            if normalized in falsy:
                return 0
        if value is None:
            print("Warning: None value encountered; treating as negative label.")
            return 0
        return 1 if value == self.positive_label else 0

    def _normalize_labels(self, labels: Iterable[Any]) -> list[int]:
        return [self._as_binary(label) for label in labels]

    def confusion_matrix(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        labels: Sequence[Any] | None = None,
    ) -> list[list[int]]:
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Mismatch between true labels ({len(y_true)}) and predicted labels ({len(y_pred)})."
            )

        true_labels = self._normalize_labels(y_true)
        pred_labels = self._normalize_labels(y_pred)
        matrix_labels = list(labels) if labels is not None else [0, 1]
        return confusion_matrix(true_labels, pred_labels, labels=matrix_labels).tolist()

    def accuracy(self, y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
        return accuracy_score(self._normalize_labels(y_true), self._normalize_labels(y_pred))

    def balanced_accuracy(self, y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
        return balanced_accuracy_score(self._normalize_labels(y_true), self._normalize_labels(y_pred))

    def precision(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        average: str = "binary",
        pos_label: int = 1,
        zero_division: float = 0.0,
    ) -> float:
        return precision_score(
            self._normalize_labels(y_true),
            self._normalize_labels(y_pred),
            average=average,
            pos_label=pos_label,
            zero_division=zero_division,
        )

    def recall(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        average: str = "binary",
        pos_label: int = 1,
        zero_division: float = 0.0,
    ) -> float:
        return recall_score(
            self._normalize_labels(y_true),
            self._normalize_labels(y_pred),
            average=average,
            pos_label=pos_label,
            zero_division=zero_division,
        )

    def f1_score(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        average: str = "binary",
        pos_label: int = 1,
        zero_division: float = 0.0,
    ) -> float:
        return f1_score(
            self._normalize_labels(y_true),
            self._normalize_labels(y_pred),
            average=average,
            pos_label=pos_label,
            zero_division=zero_division,
        )

    def precision_recall_fscore_support(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        average: str | None = None,
        labels: Sequence[int] | None = None,
        zero_division: float = 0.0,
        pos_label: int = 1,
    ) -> tuple[Any, ...]:
        return precision_recall_fscore_support(
            self._normalize_labels(y_true),
            self._normalize_labels(y_pred),
            labels=list(labels) if labels is not None else [0, 1],
            average=average,
            zero_division=zero_division,
            pos_label=pos_label,
        )

    def matthews_corrcoef(self, y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
        return matthews_corrcoef(self._normalize_labels(y_true), self._normalize_labels(y_pred))

    def classification_report(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        labels: Sequence[int] | None = None,
        zero_division: float = 0.0,
    ) -> str:
        return classification_report(
            self._normalize_labels(y_true),
            self._normalize_labels(y_pred),
            labels=list(labels) if labels is not None else [0, 1],
            zero_division=zero_division,
        )

    def zero_one_loss(self, y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
        return zero_one_loss(self._normalize_labels(y_true), self._normalize_labels(y_pred))

    def hamming_loss(self, y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
        return hamming_loss(self._normalize_labels(y_true), self._normalize_labels(y_pred))

    def evaluate(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        labels: Sequence[Any] | None = None,
        zero_division: float = 0.0,
        average: str = "binary",
    ) -> dict[str, Any]:
        true_labels = self._normalize_labels(y_true)
        pred_labels = self._normalize_labels(y_pred)

        matrix = self.confusion_matrix(y_true, y_pred, labels=labels)
        tn, fp = matrix[0]
        fn, tp = matrix[1]

        precision = precision_score(
            true_labels,
            pred_labels,
            average=average,
            pos_label=1,
            zero_division=zero_division,
        )
        recall = recall_score(
            true_labels,
            pred_labels,
            average=average,
            pos_label=1,
            zero_division=zero_division,
        )
        f1 = f1_score(
            true_labels,
            pred_labels,
            average=average,
            pos_label=1,
            zero_division=zero_division,
        )

        return {
            "accuracy": accuracy_score(true_labels, pred_labels),
            "balanced_accuracy": balanced_accuracy_score(true_labels, pred_labels),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": matrix,
            "classification_report": self.classification_report(y_true, y_pred, labels=[0, 1], zero_division=zero_division),
            "matthews_corrcoef": matthews_corrcoef(true_labels, pred_labels),
            "zero_one_loss": zero_one_loss(true_labels, pred_labels),
            "hamming_loss": hamming_loss(true_labels, pred_labels),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "support": {"negative": tn + fp, "positive": fn + tp},
            "total_samples": len(true_labels),
        }

    def _read_jsonl_records(self, file_path: str | Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue
            if isinstance(parsed, dict):
                records.append(parsed)
        return records

    def _coerce_file_labels(self, file_path: str | Path, key: str | None = None) -> list[Any]:
        path = Path(file_path)
        if path.suffix.lower() == ".jsonl":
            records = self._read_jsonl_records(path)
            if not records:
                return []
            field_name = key or self.label_key
            return [record.get(field_name) for record in records]

        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            field_name = key or self.label_key
            if isinstance(payload, dict):
                if field_name in payload:
                    return payload[field_name]
                if isinstance(payload.get("results"), list):
                    return [item.get(field_name) if isinstance(item, dict) else item for item in payload["results"]]
            if isinstance(payload, list):
                if payload and all(isinstance(item, dict) for item in payload):
                    return [item.get(field_name) for item in payload]
                return payload

        values: list[Any] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue
            field_name = key or self.label_key
            if isinstance(parsed, dict):
                values.append(parsed.get(field_name, parsed.get(self.label_key)))
            else:
                values.append(parsed)
        return values

    def compare_files(
        self,
        reference_file: str | Path,
        comparison_file: str | Path | None = None,
        reference_key: str | None = None,
        comparison_key: str | None = None,
        zero_division: float = 0.0,
        average: str = "binary",
    ) -> dict[str, Any]:
        """Compare a reference file (ground truth) against another file.

        By default, the second file is the same as the first one. This makes it easy to
        use the current long-analysis output as both the reference and the candidate while
        later swapping in a different model output.
        """
        reference_path = Path(reference_file)
        comparison_path = Path(reference_file if comparison_file is None else comparison_file)

        y_true = self._coerce_file_labels(reference_path, key=reference_key)
        y_pred = self._coerce_file_labels(comparison_path, key=comparison_key)

        if len(y_true) != len(y_pred):
            raise ValueError(
                "The reference and comparison files do not contain the same number of labels: "
                f"{len(y_true)} != {len(y_pred)}."
            )

        return self.evaluate(y_true, y_pred, zero_division=zero_division, average=average)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model predictions against a reference log-analysis output.")
    parser.add_argument("reference_file", nargs="?", default="../analysis/log_analysis.jsonl")
    parser.add_argument("comparison_file", nargs="?", default=None)
    parser.add_argument("--label-key", default="error_found")
    parser.add_argument("--positive-label", default="True")
    parser.add_argument("--negative-label", default="False")
    args = parser.parse_args()

    evaluator = ModelEvaluator(
        positive_label=args.positive_label.lower() in {"1", "true", "yes", "y"},
        negative_label=args.negative_label.lower() in {"1", "true", "yes", "y"},
        label_key=args.label_key,
    )
    metrics = evaluator.compare_files(
        args.reference_file,
        args.comparison_file,
        reference_key=args.label_key,
        comparison_key=args.label_key,
    )
    print(json.dumps(metrics, indent=2, default=str))
