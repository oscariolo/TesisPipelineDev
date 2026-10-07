from __future__ import annotations

import ast
import json
from datetime import datetime
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
    roc_auc_score,
    roc_curve,
    zero_one_loss,
)
from sklearn.preprocessing import label_binarize


def _to_jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays into plain JSON-serializable types."""
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, bool):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (AttributeError, ValueError):
            return value
    return value


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
        none_as: str = "negative",
        unclassified_label: int = 2,
    ) -> None:
        if none_as not in {"negative", "unclassified"}:
            raise ValueError("none_as must be 'negative' or 'unclassified'")
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.label_key = label_key
        self.labels = tuple(labels)
        self.none_as = none_as
        self.unclassified_label = unclassified_label
        # Records {source, file, index, value} for every None label encountered in a file.
        self.unclassified: list[dict[str, Any]] = []

    def _encode_label(self, value: Any) -> int:
        """Map a raw label to 0 (negative), 1 (positive) or, in unclassified mode, 2."""
        if value is None:
            return self.unclassified_label if self.none_as == "unclassified" else 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if numeric in (0.0, 1.0, 2.0):
                return int(numeric)
            return 1 if numeric == 1.0 else 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            truthy = {"1", "true", "yes", "y", "error", "error_found", "positive"}
            falsy = {"0", "false", "no", "n", "ok", "normal", "negative"}
            unclassified = {"2", "unclassified", "non-classified", "non classified", "not classified", "unknown", "none", "null", "nan"}
            if normalized in unclassified:
                return self.unclassified_label if self.none_as == "unclassified" else 0
            if normalized in truthy:
                return 1
            if normalized in falsy:
                return 0
        return 1 if value == self.positive_label else 0

    def _normalize_labels(self, labels: Iterable[Any]) -> list[int]:
        return [self._encode_label(label) for label in labels]

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

    def roc_auc(
        self,
        y_true: Sequence[Any],
        y_score: Sequence[float],
        average: str = "macro",
        multi_class: str = "ovr",
    ) -> float:
        """Compute ROC AUC. Requires continuous scores (not binary labels) for y_score."""
        true_bin = self._normalize_labels(y_true)
        score_arr = [float(s) for s in y_score]
        if len(set(true_bin)) < 2:
            return 0.0
        try:
            return float(roc_auc_score(true_bin, score_arr, average=average, multi_class=multi_class))
        except ValueError:
            return 0.0

    def roc_curve_data(
        self,
        y_true: Sequence[Any],
        y_score: Sequence[float],
    ) -> dict[str, list[float]]:
        """Return ROC curve points: {fpr, tpr, thresholds}."""
        true_bin = self._normalize_labels(y_true)
        score_arr = [float(s) for s in y_score]
        if len(set(true_bin)) < 2:
            return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [1.0, 0.0]}
        fpr, tpr, thresholds = roc_curve(true_bin, score_arr)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

    def evaluate_with_roc(
        self,
        y_true: Sequence[Any],
        y_score: Sequence[float],
        labels: Sequence[Any] | None = None,
        zero_division: float = 0.0,
        average: str = "binary",
    ) -> dict[str, Any]:
        """Full evaluation including ROC AUC. y_score must be continuous confidence scores."""
        base = self.evaluate(y_true=y_true, y_pred=[s >= 0.5 for s in y_score],
                             labels=labels, zero_division=zero_division, average=average)
        base["roc_auc"] = self.roc_auc(y_true, y_score, average=average)
        base["roc_curve"] = self.roc_curve_data(y_true, y_score)
        return base

    def evaluate(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        labels: Sequence[Any] | None = None,
        zero_division: float = 0.0,
        average: str = "binary",
    ) -> dict[str, Any]:
        """Evaluate predictions; precision/recall/f1 always target the positive (error) class.

        Labels are encoded to 0 (negative), 1 (positive) and, when ``none_as`` is
        ``"unclassified"``, 2 for None. Two-class (binary) metrics are reported for binary
        data; a third class switches on the 3x3 confusion matrix and per-class breakdown.
        """
        true_labels = self._normalize_labels(y_true)
        pred_labels = self._normalize_labels(y_pred)

        present = set(true_labels) | set(pred_labels)
        multiclass = not present.issubset({0, 1})
        if labels is not None:
            class_labels = list(labels)
        elif multiclass:
            class_labels = [0, 1, 2]
        else:
            class_labels = [0, 1]

        matrix = confusion_matrix(true_labels, pred_labels, labels=class_labels).tolist()
        positive_index = class_labels.index(1)

        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            pred_labels,
            labels=class_labels,
            average=None,
            zero_division=zero_division,
        )

        total = len(true_labels)
        tp = matrix[positive_index][positive_index]
        fn = sum(matrix[positive_index]) - tp
        fp = sum(row[positive_index] for row in matrix) - tp
        tn = total - tp - fn - fp

        result: dict[str, Any] = {
            "accuracy": accuracy_score(true_labels, pred_labels),
            "balanced_accuracy": balanced_accuracy_score(true_labels, pred_labels),
            "precision": precision[positive_index],
            "recall": recall[positive_index],
            "f1_score": f1[positive_index],
            "confusion_matrix": matrix,
            "classes": list(class_labels),
            "matthews_corrcoef": matthews_corrcoef(true_labels, pred_labels),
            "zero_one_loss": zero_one_loss(true_labels, pred_labels),
            "hamming_loss": hamming_loss(true_labels, pred_labels),
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "support": {
                "negative": int(sum(matrix[0])),
                "positive": int(sum(matrix[positive_index])),
            },
            "total_samples": total,
        }

        if multiclass:
            names = {0: "negative", 1: "positive", 2: "unclassified"}
            result["per_class"] = {
                names.get(label, str(label)): {
                    "precision": precision[index],
                    "recall": recall[index],
                    "f1_score": f1[index],
                    "support": int(support[index]),
                }
                for index, label in enumerate(class_labels)
            }
            if len(class_labels) > 2:
                result["support"]["unclassified"] = int(sum(matrix[2]))
            macro = precision_recall_fscore_support(
                true_labels, pred_labels, labels=class_labels, average="macro", zero_division=zero_division
            )
            weighted = precision_recall_fscore_support(
                true_labels, pred_labels, labels=class_labels, average="weighted", zero_division=zero_division
            )
            result["precision_macro"], result["recall_macro"], result["f1_score_macro"] = macro[0], macro[1], macro[2]
            result["precision_weighted"], result["recall_weighted"], result["f1_score_weighted"] = (
                weighted[0],
                weighted[1],
                weighted[2],
            )
            result["none_as"] = self.none_as

        return result

    def read_model_metadata(self, file_path: str | Path) -> dict[str, Any]:
        """Return {model_name, embedder_model_name} from the first labelled record.

        Pipeline JSONL/JSON outputs carry ``model_name`` and ``embedder_model_name`` per
        batch; the first record that has a ``model_name`` is used. Missing fields stay None.
        """
        path = Path(file_path)
        record: dict[str, Any] | None = None
        if path.suffix.lower() == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and candidate.get("model_name") is not None:
                    record = candidate
                    break
        elif path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                candidates = payload
            elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
                candidates = payload["results"]
            else:
                candidates = []
            for candidate in candidates:
                if isinstance(candidate, dict) and candidate.get("model_name") is not None:
                    record = candidate
                    break
        if not isinstance(record, dict):
            return {"model_name": None, "embedder_model_name": None}
        return {
            "model_name": record.get("model_name"),
            "embedder_model_name": record.get("embedder_model_name"),
        }

    def build_report(
        self,
        metrics: dict[str, Any],
        model_name: str | None = None,
        embedder_model_name: str | None = None,
    ) -> dict[str, Any]:
        """Attach model metadata and positive-class framing to an evaluation result."""
        report: dict[str, Any] = {
            "model_name": model_name,
            "embedder_model_name": embedder_model_name,
            "label_key": self.label_key,
            "positive_class": "error_found (label=1)",
            "averaging": "positive class only (binary); per-class + macro/weighted for 3 classes",
            "none_as": self.none_as,
            "unclassified_count": len(self.unclassified),
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        }
        report.update(_to_jsonable(metrics))
        return report

    def save_report(
        self,
        metrics: dict[str, Any],
        output_path: str | Path,
        model_name: str | None = None,
        embedder_model_name: str | None = None,
    ) -> Path:
        """Write the evaluation result (model metadata + metrics) as a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = self.build_report(metrics, model_name=model_name, embedder_model_name=embedder_model_name)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {out}")
        return out

    def save_unclassified(self, output_path: str | Path) -> Path:
        """Write the values that could not be classified (None labels) as a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(_to_jsonable(self.unclassified), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[saved] {out} ({len(self.unclassified)} unclassified value(s))")
        return out

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

    def _coerce_file_labels(
        self,
        file_path: str | Path,
        key: str | None = None,
        source: str | None = None,
    ) -> list[Any]:
        path = Path(file_path)
        values = self._read_label_values(path, key)
        if source is not None:
            for index, value in enumerate(values):
                if value is None:
                    self.unclassified.append(
                        {"source": source, "file": str(path), "index": index, "value": None}
                    )
        return values

    def _read_label_values(self, path: Path, key: str | None = None) -> list[Any]:
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
            def get_nested(item: dict, k: str) -> Any:
                for part in k.split('.'):
                    if isinstance(item, dict):
                        item = item.get(part)
                    else:
                        return None
                return item

            if isinstance(payload, dict):
                if field_name in payload:
                    return payload[field_name]
                if isinstance(payload.get("results"), list):
                    # return [item.get(field_name) if isinstance(item, dict) else item for item in payload["results"]]
                    return [get_nested(item, field_name) if isinstance(item, dict) else item for item in payload["results"]]
                    
            if isinstance(payload, list):
                if payload and all(isinstance(item, dict) for item in payload):
                    # return [item.get(field_name) for item in payload]
                    return [get_nested(item, field_name) for item in payload]
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
        unclassified_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """Compare a reference file (ground truth) against another file.

        By default, the second file is the same as the first one. This makes it easy to
        use the current long-analysis output as both the reference and the candidate while
        later swapping in a different model output.

        When ``unclassified_file`` is given, every None label found in either file is
        written there as ``{source, file, index, value}``.
        """
        reference_path = Path(reference_file)
        comparison_path = Path(reference_file if comparison_file is None else comparison_file)

        self.unclassified = []
        y_true = self._coerce_file_labels(reference_path, key=reference_key, source="reference")
        y_pred = self._coerce_file_labels(comparison_path, key=comparison_key, source="comparison")

        if len(y_true) != len(y_pred):
            raise ValueError(
                "The reference and comparison files do not contain the same number of labels: "
                f"{len(y_true)} != {len(y_pred)}."
            )

        metrics = self.evaluate(y_true, y_pred, zero_division=zero_division, average=average)
        if unclassified_file is not None:
            self.save_unclassified(unclassified_file)
        return metrics

    def compare_batches(
        self,
        reference_file: str | Path,
        comparison_file: str | Path,
        batch_size: int,
        reference_key: str | None = None,
        comparison_key: str | None = None,
        zero_division: float = 0.0,
        average: str = "binary",
        unclassified_file: str | Path | None = None,
    ) -> dict[str, Any]:
        """Compare a per-log reference file against a per-batch comparison file."""
        
        self.unclassified = []
        # 1. Load all individual logs from the reference file
        all_logs = self._coerce_file_labels(reference_file, key=reference_key, source="reference")
        
        # 2. Group them into batches
        reference_batches = []
        for i in range(0, len(all_logs), batch_size):
            batch_slice = all_logs[i:i+batch_size]
            # If ANY log in the batch is an error, the whole batch is marked as an error
            batch_has_error = any(self._encode_label(label) == 1 for label in batch_slice)
            reference_batches.append(batch_has_error)
            
        # 3. Load the comparison file (already per-batch from the SLM)
        y_pred = self._coerce_file_labels(comparison_file, key=comparison_key, source="comparison")
        
        # 4. Truncate both to match the minimum length
        processed_count = len(y_pred)
        if processed_count > len(reference_batches):
            print(f"Warning: The model evaluated MORE batches ({processed_count}) than there are in the dataset ({len(reference_batches)}).")
        
        min_len = min(len(reference_batches), len(y_pred))
        y_true = reference_batches[:min_len]
        y_pred = y_pred[:min_len]
        
        print(f"Evaluando {min_len} batches (cada uno de {batch_size} logs)...")
        metrics = self.evaluate(y_true, y_pred, zero_division=zero_division, average=average)
        if unclassified_file is not None:
            self.save_unclassified(unclassified_file)
        return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model predictions against a reference log-analysis output.")
    parser.add_argument("reference_file", nargs="?", default="dataset/analysis/log_analysis.jsonl")
    parser.add_argument("comparison_file", nargs="?", default=None)
    parser.add_argument("--label-key", default="error_found")
    parser.add_argument("--ref-key", default=None, help="Label key for reference file (defaults to --label-key)")
    parser.add_argument("--comp-key", default=None, help="Label key for comparison file (defaults to --label-key)")
    parser.add_argument("--positive-label", default="True")
    parser.add_argument("--negative-label", default="False")
    parser.add_argument("--batch-size", type=int, default=None, help="If set, groups the reference file into batches of this size before comparing.")
    parser.add_argument("--none-as", choices=["negative", "unclassified"], default="negative",
                        help="How to treat None labels: as negative (binary) or as a separate third class.")
    parser.add_argument("--report-file", default="analysis/evaluation_report.json", help="Path to write the JSON evaluation report.")
    parser.add_argument("--unclassified-file", default="analysis/unclassified.json", help="Path to write the values that could not be classified (None labels).")
    parser.add_argument("--model-name", default=None, help="Override the evaluated model name (default: read from the comparison file).")
    parser.add_argument("--embedding-model-name", default=None, help="Override the embedding model name (default: read from the comparison file).")
    args = parser.parse_args()

    evaluator = ModelEvaluator(
        positive_label=args.positive_label.lower() in {"1", "true", "yes", "y"},
        negative_label=args.negative_label.lower() in {"1", "true", "yes", "y"},
        label_key=args.label_key,
        none_as=args.none_as,
    )
    
    # metrics = evaluator.compare_files(
    #         args.reference_file,
    #         args.comparison_file,
    #         reference_key=args.label_key,
    #         comparison_key=args.label_key,
    #     )
    
    ref_key = args.ref_key or args.label_key
    comp_key = args.comp_key or args.label_key

    if args.batch_size:
        metrics = evaluator.compare_batches(
            args.reference_file,
            args.comparison_file,
            batch_size=args.batch_size,
            reference_key=ref_key,
            comparison_key=comp_key,
            unclassified_file=args.unclassified_file,
        )
    else:
        metrics = evaluator.compare_files(
            args.reference_file,
            args.comparison_file,
            reference_key=ref_key,
            comparison_key=comp_key,
            unclassified_file=args.unclassified_file,
        )

    metadata_file = args.comparison_file or args.reference_file
    metadata = evaluator.read_model_metadata(metadata_file) if metadata_file else {}
    evaluator.save_report(
        metrics,
        args.report_file,
        model_name=args.model_name or metadata.get("model_name"),
        embedder_model_name=args.embedding_model_name or metadata.get("embedder_model_name"),
    )

    print(json.dumps(metrics, indent=2, default=str))
