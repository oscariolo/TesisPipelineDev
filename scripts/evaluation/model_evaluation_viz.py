"""
Visual evaluation reports for SLM log-analysis models.

Generates publication-quality diagrams from ModelEvaluator metrics:
  - ROC curve (with AUC annotation)
  - Confusion matrix heatmap
  - Per-metric bar chart (accuracy, precision, recall, F1, MCC)
  - Multi-model comparison dashboard

All plots are saved as PNG files. No JSON/HTML reports are produced.
"""

from __future__ import annotations

import io
import os
import pathlib
import warnings
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np

# Import modelEvaluation from the same directory
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from modelEvaluation import ModelEvaluator

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _style_axes(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def _save(fig: plt.Figure, path: pathlib.Path, dpi: int = 150) -> pathlib.Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {path}")
    return path


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    fpr: list[float],
    tpr: list[float],
    auc: float,
    output_path: str | pathlib.Path,
) -> pathlib.Path:
    """Save a single ROC curve plot."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linewidth=1.2, linestyle="--", label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#2563eb")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    _style_axes(ax, "Receiver Operating Characteristic (ROC) Curve", "False Positive Rate", "True Positive Rate")
    ax.legend(loc="lower right", fontsize=11, frameon=True, facecolor="white", edgecolor="#e2e8f0")
    return _save(fig, pathlib.Path(output_path))


def plot_roc_from_metrics(
    metrics: dict[str, Any],
    output_path: str | pathlib.Path,
) -> pathlib.Path | None:
    """Extract ROC data from a metrics dict and plot."""
    roc = metrics.get("roc_curve", {})
    auc = metrics.get("roc_auc", 0.0)
    if not roc or "fpr" not in roc:
        print("[warn] No ROC data in metrics dict — skipping ROC plot.")
        return None
    return plot_roc_curve(roc["fpr"], roc["tpr"], auc, output_path)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    matrix: list[list[int]],
    output_path: str | pathlib.Path,
    class_labels: list[str] | None = None,
) -> pathlib.Path:
    """Save a confusion matrix heatmap."""
    if class_labels is None:
        class_labels = ["Negative (OK)", "Positive (Error)"]
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.85)
    ax.set(xticks=np.arange(arr.shape[1]),
           yticks=np.arange(arr.shape[0]),
           xticklabels=class_labels, yticklabels=class_labels,
           ylabel="Actual", xlabel="Predicted",
           )
    _style_axes(ax, "Confusion Matrix", "Predicted label", "Actual label")
    ax.set_ylim(len(class_labels) - 0.5, -0.5)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:d}", ha="center", va="center",
                    color="white" if arr[i, j] > arr.max() * 0.55 else "#1e293b",
                    fontsize=14, fontweight="bold")
    return _save(fig, pathlib.Path(output_path))


def plot_confusion_from_metrics(
    metrics: dict[str, Any],
    output_path: str | pathlib.Path,
) -> pathlib.Path | None:
    """Extract confusion matrix from metrics dict and plot."""
    matrix = metrics.get("confusion_matrix")
    if not matrix:
        print("[warn] No confusion_matrix in metrics dict — skipping.")
        return None
    return plot_confusion_matrix(matrix, output_path)


# ---------------------------------------------------------------------------
# Metrics bar chart
# ---------------------------------------------------------------------------

def plot_metrics_bar(
    metrics: dict[str, Any],
    output_path: str | pathlib.Path,
    include_keys: list[str] | None = None,
) -> pathlib.Path | None:
    """Bar chart of scalar metrics from a single evaluation."""
    default_keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "matthews_corrcoef"]
    keys = include_keys or default_keys
    values = []
    labels = []
    for k in keys:
        v = metrics.get(k)
        if v is not None:
            values.append(float(v))
            label = k.replace("_", " ").title()
            labels.append(label)

    if not values:
        print("[warn] No scalar metrics found — skipping bar chart.")
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2563eb" if v >= 0.7 else "#f59e0b" if v >= 0.4 else "#ef4444" for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    _style_axes(ax, "Model Performance Metrics", "", "Score")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    return _save(fig, pathlib.Path(output_path))


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    model_results: dict[str, dict[str, Any]],
    output_path: str | pathlib.Path,
    metric_keys: list[str] | None = None,
) -> pathlib.Path | None:
    """Grouped bar chart comparing multiple models across chosen metrics."""
    if metric_keys is None:
        metric_keys = ["accuracy", "precision", "recall", "f1_score"]

    model_names = list(model_results.keys())
    n_models = len(model_names)
    n_metrics = len(metric_keys)
    if n_models == 0 or n_metrics == 0:
        print("[warn] Empty model_results — skipping comparison plot.")
        return None

    x = np.arange(n_metrics)
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(8, 5))

    cmap = matplotlib.colormaps.get_cmap("tab10")
    for i, name in enumerate(model_names):
        m = model_results[name]
        vals = [float(m.get(k, 0.0)) for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=cmap(i),
                      edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in metric_keys], fontsize=11)
    _style_axes(ax, "Multi-Model Comparison", "", "Score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=10, frameon=True, facecolor="white", edgecolor="#e2e8f0")
    return _save(fig, pathlib.Path(output_path))


# ---------------------------------------------------------------------------
# Combined report (all diagrams for one evaluation)
# ---------------------------------------------------------------------------

def generate_report(
    metrics: dict[str, Any],
    output_dir: str | pathlib.Path,
    prefix: str = "evaluation",
) -> list[pathlib.Path]:
    """Generate all diagrams for a single evaluation and return saved file paths."""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[pathlib.Path] = []

    saved.append(plot_confusion_from_metrics(metrics, out / f"{prefix}_confusion_matrix.png") or out / f"{prefix}_confusion_matrix.png")
    saved.append(plot_roc_from_metrics(metrics, out / f"{prefix}_roc_curve.png") or out / f"{prefix}_roc_curve.png")
    saved.append(plot_metrics_bar(metrics, out / f"{prefix}_metrics_bar.png") or out / f"{prefix}_metrics_bar.png")

    return saved


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import pathlib as _pl

    # Resolve defaults relative to the scripts/ directory (one level up from this script)
    _scripts_dir = _pl.Path(__file__).resolve().parent.parent
    _dataset_dir = _scripts_dir / "dataset"
    _analysis_dir = _scripts_dir / "analysis"

    parser = argparse.ArgumentParser(
        description="Evaluate an SLM log-analysis output and generate diagram reports (PNG)."
    )
    parser.add_argument("reference_file", nargs="?", default=str(_dataset_dir / "dataset_slm_procesado_web.json"),
                        help="Ground-truth dataset from parse_and_label.py (default: dataset/dataset_slm_procesado_web.json)")
    parser.add_argument("comparison_file", nargs="?", default=str(_analysis_dir / "log_analysis.jsonl"),
                        help="Pipeline SLM output (default: analysis/log_analysis.jsonl)")
    parser.add_argument("--output-dir", default=str(_analysis_dir / "eval_plots"),
                        help="Directory to save PNG diagrams")
    parser.add_argument("--label-key", default="error_found")
    parser.add_argument("--ref-key", default="ground_truth_label.is_error",
                        help="Label key for reference file (default: ground_truth_label.is_error for parse_and_label output)")
    parser.add_argument("--comp-key", default="error_found",
                        help="Label key for comparison file (default: error_found for pipeline output)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for grouping ground-truth logs to match pipeline output (default: 100)")
    parser.add_argument("--roc", action="store_true",
                        help="Also require ROC data: reads a second comparison file with 'confidence' scores")
    parser.add_argument("--confidence-file", default=None,
                        help="JSONL file with per-batch 'confidence' float field for ROC plot")
    parser.add_argument("--prefix", default="eval",
                        help="Filename prefix for saved diagrams")
    args = parser.parse_args()

    evaluator = ModelEvaluator(
        positive_label=True,
        negative_label=False,
        label_key=args.label_key,
    )

    ref_key = args.ref_key or args.label_key
    comp_key = args.comp_key or args.label_key

    # Standard metrics
    if args.batch_size:
        metrics = evaluator.compare_batches(
            args.reference_file, args.comparison_file,
            batch_size=args.batch_size, reference_key=ref_key, comparison_key=comp_key,
        )
    else:
        metrics = evaluator.compare_files(
            args.reference_file, args.comparison_file,
            reference_key=ref_key, comparison_key=comp_key,
        )

    # ROC data if requested
    if args.roc or args.confidence_file:
        conf_path = pathlib.Path(args.confidence_file or args.comparison_file or args.reference_file)
        y_conf = []
        if conf_path.suffix == ".jsonl":
            for line in conf_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    y_conf.append(float(rec.get("confidence", rec.get("error_found", 0))))
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue
        elif conf_path.suffix == ".json":
            with conf_path.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                y_conf = [float(r.get("confidence", r.get("error_found", 0))) for r in data if isinstance(r, dict)]
            elif isinstance(data, dict) and isinstance(data.get("results"), list):
                y_conf = [float(r.get("confidence", r.get("error_found", 0))) for r in data["results"] if isinstance(r, dict)]

        if y_conf:
            ref_path = pathlib.Path(args.reference_file)
            y_true = evaluator._coerce_file_labels(ref_path, key=ref_key)
            metrics["roc_auc"] = evaluator.roc_auc(y_true, y_conf)
            metrics["roc_curve"] = evaluator.roc_curve_data(y_true, y_conf)

    # Generate all diagrams
    paths = generate_report(metrics, args.output_dir, prefix=args.prefix)
    print(f"\nDone. {len(paths)} diagram(s) saved to {args.output_dir}/")
