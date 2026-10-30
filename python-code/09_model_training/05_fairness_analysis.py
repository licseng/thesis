"""Summarize model inference results separately for MHC1 and MHC0.

This is a descriptive summary, not yet a formal fairness analysis. It reads
the held-out MHC1/MHC0 prediction CSV produced by the inference script and
writes cohort-specific performance metrics into the same output directory.
Formal fairness analyses can be added here later.
"""

from __future__ import annotations

import json
import hashlib
import os
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
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR.parent / ".matplotlib"))
INFERENCE_OUTPUT_DIRS = {
    "los": SCRIPT_DIR
    / "bioclinicalbert_los_classifier_inference_output_current_cohort",
    "readmission": SCRIPT_DIR
    / "bioclinicalbert_readmission_classifier_inference_output_current_cohort",
}

PREDICTIONS_FILENAME = "test_fairness_mhc1_mhc0_predictions.csv"
CONFIG_FILENAME = "inference_config.json"
SUMMARY_FILENAME = "mhc1_mhc0_inference_metrics.csv"
DIFFERENCE_FILENAME = "mhc1_minus_mhc0_inference_metric_differences.csv"

COHORTS = {
    "MHC1-psychosis": "is_mhc1_psychotic_admission",
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


def calibration_bins(table: pd.DataFrame) -> pd.DataFrame:
    """Ten fixed probability intervals, shared by cohorts; retain empty bins."""
    data = table.copy()
    data["probability_bin"] = np.minimum((data.predicted_probability * 10).astype(int), 9)
    rows = []
    for index in range(10):
        part = data[data.probability_bin.eq(index)]
        rows.append({"probability_bin": index, "bin_lower": index / 10,
            "bin_upper": (index + 1) / 10, "n_admissions": len(part),
            "n_subjects": part.subject_id.nunique(), "n_events": int(part.true_label.sum()),
            "mean_predicted_probability": part.predicted_probability.mean(),
            "observed_event_rate": part.true_label.mean()})
    return pd.DataFrame(rows)


def summarize_calibration(task: str, table: pd.DataFrame, output_dir: Path) -> None:
    """Descriptive admission-weighted calibration; do not recalibrate predictions.

    No naive observation-independent CIs or p values are reported. Calibration
    GLMs estimate intercept/slope point summaries only; clustering uncertainty
    remains a separate analysis. Sparse fixed-width bins are explicitly marked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import statsmodels.api as sm

    rows, frames = [], []
    for cohort, flag in COHORTS.items():
        part = table[table[flag].eq(1)]
        y = part.true_label.to_numpy(dtype=float)
        probabilities = part.predicted_probability.to_numpy(dtype=float)
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        log_odds = np.log(clipped / (1 - clipped))
        offset_fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=log_odds).fit()
        slope_fit = sm.GLM(y, sm.add_constant(log_odds), family=sm.families.Binomial()).fit()
        if not offset_fit.converged or not slope_fit.converged:
            raise RuntimeError(f"Calibration diagnostic GLM did not converge: {task}, {cohort}")
        bins = calibration_bins(part)
        bins.insert(0, "cohort", cohort)
        bins.insert(0, "task", task)
        frames.append(bins)
        rows.append({"task": task, "cohort": cohort, "n_admissions": len(part),
            "n_subjects": part.subject_id.nunique(), "mean_predicted_probability": probabilities.mean(),
            "observed_event_rate": y.mean(), "observed_minus_predicted_percentage_points": 100 * (y.mean() - probabilities.mean()),
            "observed_expected_ratio": y.sum() / probabilities.sum(),
            "calibration_in_the_large_intercept": float(offset_fit.params[0]),
            "joint_calibration_intercept": float(slope_fit.params[0]),
            "calibration_slope": float(slope_fit.params[1]),
            "brier_score": float(np.mean((y - probabilities) ** 2)),
            "n_probabilities_clipped_for_logit": int((clipped != probabilities).sum())})
    summary = pd.DataFrame(rows)
    binned = pd.concat(frames, ignore_index=True)
    summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    binned.to_csv(output_dir / "calibration_bins.csv", index=False)
    colors = {"MHC1-psychosis": "#7851A9", "MHC0": "#27864C"}
    fig, (ax, hist) = plt.subplots(2, 1, figsize=(7.4, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1.2, label="Perfect calibration")
    for position, cohort in enumerate(COHORTS):
        bins = binned[binned.cohort.eq(cohort)]
        present = bins[bins.n_admissions.gt(0)]
        ax.plot(present.mean_predicted_probability, present.observed_event_rate,
            color=colors[cohort], linewidth=1.5, label=cohort)
        dense, sparse = present[present.n_admissions.ge(50)], present[present.n_admissions.lt(50)]
        ax.scatter(dense.mean_predicted_probability, dense.observed_event_rate, color=colors[cohort], s=35, zorder=3)
        ax.scatter(sparse.mean_predicted_probability, sparse.observed_event_rate,
            edgecolor=colors[cohort], facecolor="white", s=35, zorder=3)
        centers = (bins.bin_lower + bins.bin_upper) / 2 + (-0.019 if position == 0 else 0.019)
        hist.bar(centers, bins.n_admissions, width=0.036, color=colors[cohort], alpha=0.8)
    ax.set(title="Hospital stay >7 days" if task == "los" else "Readmission within 30 days",
           ylabel="Observed outcome proportion", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.02, "Hollow points: fewer than 50 admissions\nDescriptive; no confidence intervals",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=9, color="0.3")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    hist.set(xlabel="Predicted probability", ylabel="Admissions", xticks=np.linspace(0, 1, 11))
    hist.xaxis.set_major_formatter(PercentFormatter(1))
    for axis in (ax, hist):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
    fig.savefig(output_dir / "calibration_plot.png", dpi=180)
    fig.savefig(output_dir / "calibration_plot.pdf")
    plt.close(fig)
    (output_dir / "calibration_manifest.json").write_text(json.dumps({
        "predictions_sha256": hashlib.sha256((output_dir / PREDICTIONS_FILENAME).read_bytes()).hexdigest(),
        "bin_edges": list(np.linspace(0, 1, 11)), "bin_rule": "[lower, upper), final interval includes 1.0",
        "weighting": "Each admission contributes once; readmitted subjects can contribute multiple admissions",
        "uncertainty": "Descriptive point estimates only; no independence-based CIs or significance tests",
        "calibration_in_the_large": "Logistic intercept with logit(predicted probability) as offset; ideal 0",
        "joint_intercept_and_slope": "Logistic outcome on intercept plus logit(predicted probability); ideals 0 and 1",
        "predictions_recalibrated": False}, indent=2))
    print("Calibration:")
    print(summary[["cohort", "mean_predicted_probability", "observed_event_rate",
        "observed_minus_predicted_percentage_points", "calibration_slope"]].to_string(index=False))


def summarize_inference(task: str, output_dir: Path) -> None:
    """Write MHC1/MHC0 summaries for one inference output directory."""
    predictions_path = output_dir / PREDICTIONS_FILENAME
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing fairness prediction file: {predictions_path}")

    table = pd.read_csv(predictions_path)
    required = {
        "hadm_id",
        "partition",
        "subject_id",
        "true_label",
        "predicted_probability",
        "predicted_label",
        *COHORTS.values(),
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{predictions_path} is missing columns: {missing}")

    if table.empty or table.hadm_id.duplicated().any() or table.subject_id.isna().any():
        raise ValueError("Prediction admissions must be nonempty, unique, and have subject IDs")
    if table.partition.ne("test_pool").any():
        raise ValueError("Fairness analysis requires held-out test-pool admissions")
    if not table[["true_label", "predicted_label", *COHORTS.values()]].isin([0, 1]).all().all():
        raise ValueError("Labels and cohort indicators must be binary and nonmissing")
    if not table.predicted_probability.between(0, 1).all():
        raise ValueError("Prediction probabilities must be finite and between zero and one")
    overlap = table[list(COHORTS.values())].astype(bool).sum(axis=1)
    if overlap.ne(1).any():
        raise ValueError(
            f"Expected every row in {predictions_path} to belong to exactly one cohort."
        )

    threshold = load_threshold(output_dir)
    if not table.predicted_label.eq(table.predicted_probability.ge(threshold).astype(int)).all():
        raise ValueError("Saved decisions do not match the configured threshold")
    if table.groupby("subject_id")[list(COHORTS.values())].nunique().gt(1).any().any():
        raise ValueError("A subject appears in both cohorts")
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
            "contrast": "MHC1_minus_MHC0",
            "metric": difference_metrics,
            "difference": [
                indexed.loc["MHC1-psychosis", metric] - indexed.loc["MHC0", metric]
                for metric in difference_metrics
            ],
        }
    )
    differences.to_csv(output_dir / DIFFERENCE_FILENAME, index=False)
    summarize_calibration(task, table, output_dir)

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
