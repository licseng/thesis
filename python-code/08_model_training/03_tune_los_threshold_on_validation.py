"""Tune LOS classifier decision thresholds on validation predictions.

This script does not retrain the model. It reads validation probabilities from
the trained Bio_ClinicalBERT output and recomputes threshold-dependent metrics
across a grid of probability cutoffs.

Use the selected validation-derived threshold later for test/fairness
evaluation. Do not choose a threshold on the test set.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_OUTPUT_DIR = SCRIPT_DIR / "bioclinicalbert_los_classifier_output"
DEFAULT_INPUT_PATH = TRAINING_OUTPUT_DIR / "validation_predictions.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "analysis_output_los_threshold_tuning"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR.parent / ".matplotlib"))

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune probability thresholds using validation predictions."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(os.environ.get("LOS_VALIDATION_PREDICTIONS_PATH", DEFAULT_INPUT_PATH)),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("LOS_THRESHOLD_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=float(os.environ.get("LOS_THRESHOLD_STEP", "0.005")),
    )
    parser.add_argument(
        "--output-prefix",
        default=os.environ.get("LOS_THRESHOLD_OUTPUT_PREFIX", "los_validation"),
    )
    parser.add_argument(
        "--task-label",
        default=os.environ.get("LOS_THRESHOLD_TASK_LABEL", "Prolonged LOS"),
    )
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    """Load validation predictions and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing validation predictions CSV: {path}")
    table = pd.read_csv(path)
    required = {"true_label", "predicted_probability"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    table = table.dropna(subset=["true_label", "predicted_probability"]).copy()
    table["true_label"] = table["true_label"].astype(int)
    table["predicted_probability"] = table["predicted_probability"].astype(float)
    return table


def metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute threshold-dependent binary metrics."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = recall_score(labels, predictions, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = precision_score(labels, predictions, zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "youden_j": float(sensitivity + specificity - 1),
        "false_positive_rate": float(1 - specificity),
        "false_negative_rate": float(1 - sensitivity),
        "n_predicted_positive": int(predictions.sum()),
        "pct_predicted_positive": float(100 * predictions.mean()),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def threshold_grid(step: float) -> np.ndarray:
    """Create a stable threshold grid avoiding exactly 0 and 1."""
    if step <= 0 or step >= 1:
        raise ValueError("threshold-step must be between 0 and 1.")
    return np.round(np.arange(step, 1.0, step), 6)


def select_recommended_thresholds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Choose useful validation-derived thresholds for later test evaluation."""
    recommendations = []

    best_f1 = metrics.sort_values(
        ["f1", "youden_j", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    recommendations.append({"criterion": "best_f1", **best_f1.to_dict()})

    best_youden = metrics.sort_values(
        ["youden_j", "f1", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    recommendations.append({"criterion": "best_youden_j", **best_youden.to_dict()})

    balanced = metrics.assign(
        sensitivity_specificity_gap=(
            metrics["recall_sensitivity"] - metrics["specificity"]
        ).abs()
    ).sort_values(["sensitivity_specificity_gap", "f1", "threshold"], ascending=[True, False, True]).iloc[0]
    recommendations.append({"criterion": "closest_sensitivity_specificity", **balanced.drop(labels=["sensitivity_specificity_gap"]).to_dict()})

    for target_recall in [0.50, 0.60, 0.70, 0.80]:
        candidates = metrics.loc[metrics["recall_sensitivity"].ge(target_recall)]
        if candidates.empty:
            continue
        chosen = candidates.sort_values(
            ["precision", "specificity", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        recommendations.append(
            {
                "criterion": f"highest_precision_with_recall_at_least_{target_recall:.2f}",
                **chosen.to_dict(),
            }
        )

    default_row = metrics.iloc[(metrics["threshold"] - 0.5).abs().argsort()[:1][0]]
    recommendations.append({"criterion": "default_0.50", **default_row.to_dict()})

    return pd.DataFrame(recommendations)


def plot_threshold_metrics(
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    output_prefix: str,
    task_label: str,
) -> None:
    """Plot threshold-dependent metrics."""
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for column in ["precision", "recall_sensitivity", "specificity", "f1"]:
        ax.plot(metrics["threshold"], metrics[column], label=column)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.set_title(f"Validation Metrics Across {task_label} Probability Thresholds")
    ax.legend()
    fig.savefig(output_dir / f"{output_prefix}_threshold_metric_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(metrics["threshold"], metrics["pct_predicted_positive"], color="#4c78a8")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Predicted positive (%)")
    ax.set_title(f"Predicted {task_label} Rate by Threshold")
    fig.savefig(output_dir / f"{output_prefix}_predicted_positive_by_threshold.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Run threshold tuning and write CSV/PNG outputs."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(args.input_path)
    labels = predictions["true_label"].to_numpy()
    probabilities = predictions["predicted_probability"].to_numpy()

    metrics = pd.DataFrame(
        [
            metrics_at_threshold(labels, probabilities, threshold)
            for threshold in threshold_grid(args.threshold_step)
        ]
    )
    recommendations = select_recommended_thresholds(metrics)

    threshold_free_metrics = {
        "n_validation_rows": int(len(predictions)),
        "n_positive": int(labels.sum()),
        "pct_positive": float(100 * labels.mean()),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }

    metrics.to_csv(args.output_dir / f"{args.output_prefix}_threshold_metrics.csv", index=False)
    recommendations.to_csv(
        args.output_dir / f"{args.output_prefix}_recommended_thresholds.csv",
        index=False,
    )
    with (args.output_dir / f"{args.output_prefix}_threshold_free_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(threshold_free_metrics, handle, indent=2)
    plot_threshold_metrics(
        metrics,
        args.output_dir,
        output_prefix=args.output_prefix,
        task_label=args.task_label,
    )

    print(f"Read validation predictions from: {args.input_path}")
    print(f"Wrote threshold tuning outputs to: {args.output_dir}")
    print("\nThreshold-free validation metrics:")
    print(json.dumps(threshold_free_metrics, indent=2))
    print("\nRecommended thresholds:")
    display_columns = [
        "criterion",
        "threshold",
        "f1",
        "precision",
        "recall_sensitivity",
        "specificity",
        "youden_j",
        "pct_predicted_positive",
    ]
    print(recommendations.loc[:, display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
