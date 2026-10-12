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
from modelEvaluation import ModelEvaluator, model_evaluation_dir

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
    size = len(matrix)
    if size == 2:
        class_labels = ["Negative (OK)", "Positive (Error)"]
    elif size == 3:
        class_labels = ["Negative (OK)", "Positive (Error)", "Unclassified"]
    else:
        class_labels = [str(i) for i in range(size)]
    return plot_confusion_matrix(matrix, output_path, class_labels=class_labels)


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
# Reusable evaluation entry point (also used by main.py --evaluate)
# ---------------------------------------------------------------------------

def run_evaluation(
    reference_file: str | pathlib.Path,
    comparison_file: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    *,
    model_name: str | None = None,
    embedder_model_name: str | None = None,
    label_key: str = "error_found",
    reference_key: str | None = None,
    comparison_key: str | None = None,
    batch_size: int | None = 100,
    none_as: str = "negative",
    prefix: str = "eval",
    run_id: str | None = None,
    score_key: str | None = "error_probability",
    report_file: str | pathlib.Path | None = None,
    unclassified_file: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Evaluate a pipeline output and write its diagrams and JSON report.

    Everything is stated by the caller: ``output_dir`` is where the report and plots go (normally
    the model folder the pipeline just wrote into) and ``model_name`` / ``embedder_model_name`` only
    label the report. ``run_id`` restricts the comparison file to one pipeline run. ``score_key``
    names the per-record score used for ROC AUC (default ``error_probability``, i.e. the model's own
    P(error) from its logits); ROC is skipped when no scores or a single class are present. Returns
    ``output_dir``, ``report_path``, ``unclassified_file``, ``plots`` and ``metrics``.
    """
    evaluator = ModelEvaluator(
        positive_label=True,
        negative_label=False,
        label_key=label_key,
        none_as=none_as,
    )

    ref_key = reference_key or label_key
    comp_key = comparison_key or label_key

    out_dir = pathlib.Path(output_dir)

    print(
        "[evaluate] reference : {}  (key: {})\n"
        "[evaluate] comparison: {}  (key: {}, run: {})\n"
        "[evaluate] output    : {}\n"
        "[evaluate] batch size: {} | none_as: {}".format(
            pathlib.Path(reference_file).resolve(),
            ref_key,
            pathlib.Path(comparison_file).resolve(),
            comp_key,
            run_id or "all runs in the file",
            out_dir.resolve(),
            batch_size if batch_size else "off (one label per record)",
            none_as,
        )
    )

    report_path = pathlib.Path(report_file) if report_file else out_dir / f"{prefix}_report.json"
    unclassified_path = (
        pathlib.Path(unclassified_file) if unclassified_file else out_dir / f"{prefix}_unclassified.json"
    )

    if batch_size:
        metrics = evaluator.compare_batches(
            reference_file, comparison_file,
            batch_size=batch_size, reference_key=ref_key, comparison_key=comp_key,
            unclassified_file=unclassified_path, run_id=run_id, score_key=score_key,
        )
    else:
        metrics = evaluator.compare_files(
            reference_file, comparison_file,
            reference_key=ref_key, comparison_key=comp_key,
            unclassified_file=unclassified_path, run_id=run_id, score_key=score_key,
        )

    # Record the provenance so a report can be traced back to the files it came from.
    metrics["reference_file"] = str(pathlib.Path(reference_file).resolve())
    metrics["comparison_file"] = str(pathlib.Path(comparison_file).resolve())
    if run_id:
        metrics["run_id"] = run_id

    paths = generate_report(metrics, out_dir, prefix=prefix)
    evaluator.save_report(
        metrics,
        report_path,
        model_name=model_name,
        embedder_model_name=embedder_model_name,
    )

    return {
        "output_dir": str(out_dir),
        "report_path": str(report_path),
        "unclassified_file": str(unclassified_path),
        "plots": [str(path) for path in paths],
        "metrics": metrics,
    }


def list_model_outputs(analysis_dir: str | pathlib.Path) -> list[pathlib.Path]:
    """Every ``<analysis_dir>/*/log_analysis.jsonl``, newest first."""
    analysis_dir = pathlib.Path(analysis_dir)
    return sorted(
        analysis_dir.glob("*/log_analysis.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def default_comparison_file(analysis_dir: str | pathlib.Path) -> str:
    """Newest ``<analysis_dir>/*/log_analysis.jsonl``, else the legacy flat file."""
    analysis_dir = pathlib.Path(analysis_dir)
    candidates = list_model_outputs(analysis_dir)
    if candidates:
        return str(candidates[0])
    return str(analysis_dir / "log_analysis.jsonl")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pathlib as _pl

    # Resolve defaults relative to the scripts/ directory (one level up from this script)
    _scripts_dir = _pl.Path(__file__).resolve().parent.parent
    _ground_truth_file = _scripts_dir / "pseudo_labeling" / "data" / "ground_truth_dataset.jsonl"
    _analysis_dir = _scripts_dir / "analysis"

    parser = argparse.ArgumentParser(
        description="Evaluate an SLM log-analysis output and generate diagram reports (PNG)."
    )
    parser.add_argument("reference_file", nargs="?", default=str(_ground_truth_file),
                        help="Ground truth, per log (default: pseudo_labeling/data/ground_truth_dataset.jsonl)")
    parser.add_argument("comparison_file", nargs="?", default=None,
                        help="Pipeline SLM output (default: newest <analysis-dir>/<model>/log_analysis.jsonl)")
    parser.add_argument("--analysis-dir", default=str(_analysis_dir),
                        help="Parent directory for per-model evaluation output (default: scripts/analysis)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for PNG diagrams (default: <analysis-dir>/<model>/)")
    parser.add_argument("--label-key", default="error_found")
    parser.add_argument("--ref-key", default="consensus_is_error",
                        help="Label key for reference file (default: consensus_is_error, the pseudo-labeling 0/1 vote)")
    parser.add_argument("--comp-key", default="error_found",
                        help="Label key for comparison file (default: error_found for pipeline output)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for grouping ground-truth logs to match pipeline output (default: 100)")
    parser.add_argument("--none-as", choices=["negative", "unclassified"], default="negative",
                        help="How to treat None labels: as negative (binary) or as a separate third class.")
    parser.add_argument("--unclassified-file", default=None,
                        help="Path to write the values that could not be classified (default: <model dir>/<prefix>_unclassified.json)")
    parser.add_argument("--score-key", default="error_probability",
                        help="Per-record score used for ROC AUC (default: error_probability).")
    parser.add_argument("--prefix", default="eval",
                        help="Filename prefix for saved diagrams")
    parser.add_argument("--report-file", default=None,
                        help="Path to write the JSON evaluation report (default: <model dir>/<prefix>_report.json)")
    parser.add_argument("--model-name", default=None,
                        help="Evaluated model name; also names the default output folder (<analysis-dir>/<model>).")
    parser.add_argument("--embedding-model-name", default=None,
                        help="Embedding model name recorded in the report.")
    args = parser.parse_args()

    if args.comparison_file:
        comparison_file = args.comparison_file
    else:
        comparison_file = default_comparison_file(args.analysis_dir)
        print("[evaluate] no comparison_file given; using the newest pipeline output")

    if not pathlib.Path(args.reference_file).exists():
        raise SystemExit(f"[evaluate] reference file not found: {args.reference_file}")
    if not pathlib.Path(comparison_file).exists():
        available = [path.parent.name for path in list_model_outputs(args.analysis_dir)]
        raise SystemExit(
            f"[evaluate] no pipeline output at {comparison_file}\n"
            f"[evaluate] model folders with output under {args.analysis_dir}: "
            f"{', '.join(available) if available else 'none'}"
        )

    # Model and embedder are stated on the command line; nothing is sniffed from the output file.
    model_name = args.model_name
    embedder_model_name = args.embedding_model_name
    output_dir = (
        pathlib.Path(args.output_dir) if args.output_dir
        else model_evaluation_dir(args.analysis_dir, model_name)
    )
    if not model_name:
        print("[warn] no --model-name given; writing to the shared eval_plots folder")

    result = run_evaluation(
        reference_file=args.reference_file,
        comparison_file=comparison_file,
        output_dir=output_dir,
        model_name=model_name,
        embedder_model_name=embedder_model_name,
        label_key=args.label_key,
        reference_key=args.ref_key,
        comparison_key=args.comp_key,
        batch_size=args.batch_size,
        none_as=args.none_as,
        prefix=args.prefix,
        score_key=args.score_key,
        report_file=args.report_file,
        unclassified_file=args.unclassified_file,
    )

    print(f"\nDone. {len(result['plots'])} diagram(s) saved to {result['output_dir']}/")
