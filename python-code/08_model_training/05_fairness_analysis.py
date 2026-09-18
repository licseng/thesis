"""Summarize model inference results separately for MHH1 and MHC0.

This is a descriptive summary, not yet a formal fairness analysis. It reads
the held-out MHH1/MHC0 prediction CSV produced by the inference script and
writes cohort-specific performance metrics into the same output directory.
Formal fairness analyses can be added here later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_OUTPUT_DIRS = {
    "los": SCRIPT_DIR
    / "bioclinicalbert_los_classifier_inference_output_3epoch_NOclassweights",
    "readmission": SCRIPT_DIR
    / "bioclinicalbert_readmission_classifier_inference_output_2epoch_NOclassweights",
}

PREDICTIONS_FILENAME = "test_fairness_mhh1_mhc0_predictions.csv"
CONFIG_FILENAME = "inference_config.json"
SUMMARY_FILENAME = "mhh1_mhc0_inference_metrics.csv"
DIFFERENCE_FILENAME = "mhh1_minus_mhc0_inference_metric_differences.csv"

COHORTS = {
    "MHH1": "is_mhh1_psychotic_admission",
    "MHC0": "is_mhc0_admission",
}


def safe_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Return AUROC when both outcome classes are present."""
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def summarize_cohort(table: pd.DataFrame, cohort: str) -> dict[str, float | int | str]:
    """Calculate descriptive prediction metrics for one cohort."""
    labels = table["true_label"].astype(int).to_numpy()
    predictions = table["predicted_label"].astype(int).to_numpy()
    probabilities = table["predicted_probability"].astype(float).to_numpy()
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "cohort": cohort,
        "n_admissions": int(len(table)),
        "n_subjects": int(table["subject_id"].nunique()),
        "n_positive": int(labels.sum()),
        "outcome_prevalence_pct": float(100 * labels.mean()),
        "mean_predicted_probability_pct": float(100 * probabilities.mean()),
        "n_predicted_positive": int(predictions.sum()),
        "predicted_positive_pct": float(100 * predictions.mean()),
        "auroc": safe_auroc(labels, probabilities),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def load_threshold(output_dir: Path) -> float:
    """Read the decision threshold saved by the inference run."""
    config_path = output_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Missing inference configuration: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return float(config["threshold"])


def summarize_inference(task: str, output_dir: Path) -> None:
    """Write MHH1/MHC0 summaries for one inference output directory."""
    predictions_path = output_dir / PREDICTIONS_FILENAME
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing fairness prediction file: {predictions_path}")

    table = pd.read_csv(predictions_path)
    required = {
        "subject_id",
        "true_label",
        "predicted_probability",
        "predicted_label",
        *COHORTS.values(),
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{predictions_path} is missing columns: {missing}")

    overlap = table[list(COHORTS.values())].astype(bool).sum(axis=1)
    if overlap.ne(1).any():
        raise ValueError(
            f"Expected every row in {predictions_path} to belong to exactly one cohort."
        )

    threshold = load_threshold(output_dir)
    rows = []
    for cohort, indicator_column in COHORTS.items():
        cohort_table = table.loc[table[indicator_column].astype(bool)].copy()
        row = summarize_cohort(cohort_table, cohort)
        row["task"] = task
        row["decision_threshold"] = threshold
        rows.append(row)

    summary = pd.DataFrame(rows)
    ordered_columns = [
        "task",
        "cohort",
        "decision_threshold",
        *[column for column in summary.columns if column not in {"task", "cohort", "decision_threshold"}],
    ]
    summary = summary.loc[:, ordered_columns]
    summary.to_csv(output_dir / SUMMARY_FILENAME, index=False)

    indexed = summary.set_index("cohort")
    difference_metrics = [
        column
        for column in summary.select_dtypes(include="number").columns
        if column != "decision_threshold"
    ]
    differences = pd.DataFrame(
        {
            "task": task,
            "contrast": "MHH1_minus_MHC0",
            "metric": difference_metrics,
            "difference": [
                indexed.loc["MHH1", metric] - indexed.loc["MHC0", metric]
                for metric in difference_metrics
            ],
        }
    )
    differences.to_csv(output_dir / DIFFERENCE_FILENAME, index=False)

    print(f"\n{task.upper()} (threshold={threshold:g})")
    display_columns = [
        "cohort",
        "n_admissions",
        "n_subjects",
        "outcome_prevalence_pct",
        "mean_predicted_probability_pct",
        "auroc",
        "auprc",
        "recall_sensitivity",
        "specificity",
        "f1",
    ]
    print(summary.loc[:, display_columns].to_string(index=False))
    print(f"Wrote: {output_dir / SUMMARY_FILENAME}")
    print(f"Wrote: {output_dir / DIFFERENCE_FILENAME}")


def main() -> None:
    """Summarize all configured inference runs."""
    for task, output_dir in INFERENCE_OUTPUT_DIRS.items():
        summarize_inference(task, output_dir)


if __name__ == "__main__":
    main()
