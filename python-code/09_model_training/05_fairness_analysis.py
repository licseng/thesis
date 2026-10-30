"""Compare frozen classifiers on complete held-out matched admission pairs.

Default: paired evaluation with connected-patient-block bootstrap confidence
intervals. --all-test-descriptive reproduces the earlier all-test summaries;
--audit-test-pairs checks pair retention. No model or threshold is fitted here.
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-test-pairs", action="store_true",
                        help="Audit pair retention and covariate balance without rerunning calibration")
    parser.add_argument("--all-test-descriptive", action="store_true",
                        help="Reproduce the previous unpaired all-test descriptive analysis")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("Use at least 100 bootstrap samples (default 5000)")
    if args.audit_test_pairs and args.all_test_descriptive:
        parser.error("Choose one analysis mode")
    if args.audit_test_pairs:
        audit_test_pairs()
        return
    for task, output_dir in INFERENCE_OUTPUT_DIRS.items():
        if args.all_test_descriptive:
            summarize_inference(task, output_dir)
        else:
            paired_fairness_analysis(task, output_dir, args.bootstrap_samples, args.seed)


BOOTSTRAP_METRICS = [
    "auroc", "auprc", "recall_sensitivity", "specificity", "false_positive_rate",
    "false_negative_rate", "precision", "f1", "accuracy", "brier_score",
    "outcome_prevalence_pct", "mean_predicted_probability_pct", "predicted_positive_pct",
    "observed_minus_predicted_percentage_points", "observed_expected_ratio",
    "calibration_in_the_large_intercept", "joint_calibration_intercept", "calibration_slope",
]


def patient_pair_blocks(table: pd.DataFrame) -> np.ndarray:
    """Connected components of patient nodes joined by original matched pairs.

    Every retained admission of a patient and both members of every retained
    pair get the same block ID. Components are constructed on the task-specific
    complete-pair evaluation population, not on training/validation admissions.
    """
    parent = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for subjects in table.groupby("pair_id", sort=True).subject_id:
        members = subjects[1].to_numpy()
        if len(members) != 2:
            raise ValueError("Expected two admissions per retained pair")
        first, second = find(int(members[0])), find(int(members[1]))
        if first != second:
            parent[max(first, second)] = min(first, second)
    roots = [find(int(subject)) for subject in table.subject_id]
    return pd.factorize(roots, sort=True)[0]


def logistic_calibration(logit, y, weights, offset_only=False):
    """Weighted logistic calibration via Newton steps, with convergence checks.

    This calculates descriptive calibration parameters within each bootstrap
    sample; it never changes the frozen model's predictions.
    """
    from scipy.special import expit

    if weights.sum() <= 0 or (weights * y).sum() <= 0 or (weights * (1 - y)).sum() <= 0:
        return np.array([np.nan, np.nan])
    prevalence = (weights * y).sum() / weights.sum()
    beta = np.array([0., 1.]) if offset_only else np.array([np.log(prevalence / (1 - prevalence)), 0.])
    for _ in range(60):
        p = expit(beta[0] + beta[1] * logit)
        residual = weights * (y - p)
        curvature = weights * p * (1 - p)
        if offset_only:
            hessian = curvature.sum()
            if hessian <= 1e-12:
                break
            step = np.array([residual.sum() / hessian, 0.])
        else:
            hessian = np.array([[curvature.sum(), (curvature * logit).sum()],
                                [(curvature * logit).sum(), (curvature * logit * logit).sum()]])
            try:
                step = np.linalg.solve(hessian, np.array([residual.sum(), (residual * logit).sum()]))
            except np.linalg.LinAlgError:
                break
        # Backtracking prevents Newton overshoot for weak/reversed calibration.
        eta = beta[0] + beta[1] * logit
        likelihood = (weights * (y * eta - np.logaddexp(0., eta))).sum()
        for _ in range(25):
            candidate = beta + step
            eta = candidate[0] + candidate[1] * logit
            candidate_likelihood = (weights * (y * eta - np.logaddexp(0., eta))).sum()
            if candidate_likelihood >= likelihood - 1e-10:
                break
            step *= .5
        else:
            return np.array([np.nan, np.nan])
        beta = candidate
        if np.max(np.abs(step)) < 1e-8:
            return beta
    return np.array([np.nan, np.nan])


class WeightedEvaluation:
    """Fast weighted metrics for cluster-multiplicity bootstrap samples.

    Sorting/tie groups are fixed across samples. Integer observation weights
    are exactly equivalent to duplicating all rows of a sampled block.
    """

    def __init__(self, table):
        self.y = table.true_label.to_numpy(dtype=float)
        self.p = table.predicted_probability.to_numpy(dtype=float)
        self.d = table.predicted_label.to_numpy(dtype=float)
        clipped = np.clip(self.p, 1e-6, 1 - 1e-6)
        self.logit = np.log(clipped / (1 - clipped))
        self.order = np.argsort(-self.p, kind="stable")
        ordered = self.p[self.order]
        self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        self.bin = np.minimum((self.p * 10).astype(int), 9)

    def calculate(self, weights):
        y, p, d = self.y, self.p, self.d
        n = weights.sum()
        positive = (weights * y).sum()
        negative = n - positive
        tp = (weights * y * d).sum()
        fp = (weights * (1 - y) * d).sum()
        fn, tn = positive - tp, negative - fp

        def ratio(a, b):
            return a / b if b > 0 else np.nan

        ordered_weight, ordered_y = weights[self.order], y[self.order]
        group_positive = np.add.reduceat(ordered_weight * ordered_y, self.starts)
        group_negative = np.add.reduceat(ordered_weight * (1 - ordered_y), self.starts)
        cumulative_positive, cumulative_negative = np.cumsum(group_positive), np.cumsum(group_negative)
        auc = ratio((group_negative * (cumulative_positive - .5 * group_positive)).sum(), positive * negative)
        precision_curve = np.divide(cumulative_positive, cumulative_positive + cumulative_negative,
                                    out=np.zeros_like(cumulative_positive),
                                    where=cumulative_positive + cumulative_negative > 0)
        ap = ratio((group_positive * precision_curve).sum(), positive)
        mean_p, prevalence = ratio((weights * p).sum(), n), ratio(positive, n)
        offset = logistic_calibration(self.logit, y, weights, offset_only=True)
        joint = logistic_calibration(self.logit, y, weights)
        metrics = np.array([
            auc, ap, ratio(tp, positive), ratio(tn, negative), ratio(fp, negative), ratio(fn, positive),
            ratio(tp, tp + fp), ratio(2 * tp, 2 * tp + fp + fn), ratio(tp + tn, n),
            ratio((weights * (y - p) ** 2).sum(), n),
            100 * prevalence, 100 * mean_p, 100 * ratio(tp + fp, n), 100 * (prevalence - mean_p),
            ratio(prevalence, mean_p), offset[0], joint[0], joint[1],
        ])
        count = np.bincount(self.bin, weights=weights, minlength=10)
        events = np.bincount(self.bin, weights=weights * y, minlength=10)
        predicted = np.bincount(self.bin, weights=weights * p, minlength=10)
        bins = np.stack([np.divide(predicted, count, out=np.full(10, np.nan), where=count > 0),
                         np.divide(events, count, out=np.full(10, np.nan), where=count > 0)])
        return metrics, bins


def interval_summary(samples):
    valid = np.isfinite(samples)
    if not valid.any():
        return np.nan, np.nan, 0
    lower, upper = np.quantile(samples[valid], [.025, .975])
    return float(lower), float(upper), int(valid.sum())


def paired_fairness_analysis(task, output_dir, n_bootstrap, seed):
    """Evaluate complete pairs with patient/pair-preserving block resampling."""
    source = output_dir / PREDICTIONS_FILENAME
    table = pd.read_csv(source)
    required = {"pair_id", "subject_id", "hadm_id", "cohort", "partition", "true_label",
                "predicted_label", "predicted_probability", *COHORTS.values()}
    if required - set(table):
        raise ValueError(f"Missing prediction fields: {sorted(required - set(table))}")
    if table[list(required - {"predicted_probability"})].isna().any().any() or table.hadm_id.duplicated().any():
        raise ValueError("Missing identifiers/labels or duplicated prediction admissions")
    if table.partition.ne("test_pool").any() or not table.predicted_probability.between(0, 1).all():
        raise ValueError("Expected held-out predictions with finite probabilities")
    if not table[["true_label", "predicted_label", *COHORTS.values()]].isin([0, 1]).all().all():
        raise ValueError("Expected binary outcomes, decisions and cohort indicators")
    if table[list(COHORTS.values())].sum(axis=1).ne(1).any():
        raise ValueError("Cohort indicators must be mutually exclusive")
    if table.groupby("subject_id")[list(COHORTS.values())].nunique().gt(1).any().any():
        raise ValueError("A patient appears in both cohorts")
    threshold = load_threshold(output_dir)
    if not table.predicted_label.eq(table.predicted_probability.ge(threshold).astype(int)).all():
        raise ValueError("Saved predictions differ from the frozen decision threshold")
    original_keys = pd.read_parquet(SCRIPT_DIR.parent / "02_cohort_matching/matched_cohort_output/matched_pairs.parquet",
                                   columns=["pair_id", "mhc1_subject_id", "mhc1_hadm_id", "mhc0_subject_id", "mhc0_hadm_id"])
    expected = []
    for role, flag in [("mhc1", "is_mhc1_psychotic_admission"), ("mhc0", "is_mhc0_admission")]:
        part = original_keys[["pair_id", f"{role}_subject_id", f"{role}_hadm_id"]].rename(
            columns={f"{role}_subject_id": "subject_id", f"{role}_hadm_id": "hadm_id"})
        part[flag] = 1
        expected.append(part)
    expected = pd.concat(expected).fillna(0)
    check = table.merge(expected, on=["pair_id", "subject_id", "hadm_id"], how="left",
                        validate="one_to_one", indicator=True, suffixes=("", "_expected"))
    if not check._merge.eq("both").all() or any(not check[flag].eq(check[f"{flag}_expected"]).all() for flag in COHORTS.values()):
        raise ValueError("Prediction identifiers/cohort roles differ from the original matching")
    pair_counts = table.groupby("pair_id").size()
    if pair_counts.gt(2).any():
        raise ValueError("More than two admissions in a matched pair")
    complete_ids = pair_counts.index[pair_counts.eq(2)]
    complete = table.loc[table.pair_id.isin(complete_ids)].sort_values(["pair_id", "hadm_id"]).reset_index(drop=True)
    if complete.empty or not complete.groupby("pair_id")[list(COHORTS.values())].sum().eq(1).all().all():
        raise ValueError("Complete pairs must contain exactly one admission per cohort")
    complete["bootstrap_block_id"] = patient_pair_blocks(complete)
    if complete.groupby("pair_id").bootstrap_block_id.nunique().ne(1).any() or complete.groupby("subject_id").bootstrap_block_id.nunique().ne(1).any():
        raise AssertionError("A bootstrap block split a patient or pair")
    n_blocks = complete.bootstrap_block_id.nunique()
    if n_blocks < 2:
        raise ValueError("At least two independent patient/pair blocks are required")
    block_sizes = complete.groupby("bootstrap_block_id").agg(n_admissions=("hadm_id", "size"),
        n_subjects=("subject_id", "nunique"), n_pairs=("pair_id", "nunique"))
    destination = output_dir / "paired_fairness_analysis"
    destination.mkdir(exist_ok=True)
    evaluators, ids, parts = [], [], []
    point = np.empty((2, len(BOOTSTRAP_METRICS)))
    point_bins = np.empty((2, 2, 10))
    for i, (cohort, flag) in enumerate(COHORTS.items()):
        part = complete.loc[complete[flag].eq(1)]
        parts.append(part)
        ids.append(part.bootstrap_block_id.to_numpy())
        evaluator = WeightedEvaluation(part)
        evaluators.append(evaluator)
        point[i], point_bins[i] = evaluator.calculate(np.ones(len(part)))
    bootstrap = np.full((n_bootstrap, 2, len(BOOTSTRAP_METRICS)), np.nan)
    bootstrap_bins = np.full((n_bootstrap, 2, 2, 10), np.nan)
    rng = np.random.default_rng(seed)
    print(f"{task}: {len(complete_ids):,} complete pairs, {n_blocks:,} patient/pair blocks; bootstrapping {n_bootstrap:,} times", flush=True)
    for draw in range(n_bootstrap):
        multiplicity = rng.multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks))
        for cohort_index in range(2):
            bootstrap[draw, cohort_index], bootstrap_bins[draw, cohort_index] = evaluators[cohort_index].calculate(multiplicity[ids[cohort_index]])
        if (draw + 1) % 500 == 0:
            print(f"{task}: bootstrap {draw + 1:,}/{n_bootstrap:,}", flush=True)
    summaries, differences, bin_rows = [], [], []
    for index, cohort in enumerate(COHORTS):
        for metric_index, metric in enumerate(BOOTSTRAP_METRICS):
            lower, upper, valid = interval_summary(bootstrap[:, index, metric_index])
            summaries.append({"task": task, "cohort": cohort, "metric": metric,
                "estimate": point[index, metric_index], "ci_lower": lower, "ci_upper": upper,
                "n_valid_bootstrap": valid, "n_bootstrap": n_bootstrap,
                "n_admissions": len(parts[index]), "n_subjects": parts[index].subject_id.nunique()})
        source_bins = calibration_bins(parts[index])
        for bin_index, row in source_bins.iterrows():
            result = {"task": task, "cohort": cohort, **row.to_dict()}
            for measure_index, measure in enumerate(["mean_predicted_probability", "observed_event_rate"]):
                lower, upper, valid = interval_summary(bootstrap_bins[:, index, measure_index, bin_index])
                result.update({f"{measure}_ci_lower": lower, f"{measure}_ci_upper": upper,
                               f"{measure}_n_valid_bootstrap": valid})
            bin_rows.append(result)
    for index, metric in enumerate(BOOTSTRAP_METRICS):
        lower, upper, valid = interval_summary(bootstrap[:, 0, index] - bootstrap[:, 1, index])
        differences.append({"task": task, "contrast": "MHC1_minus_MHC0", "metric": metric,
            "estimate": point[0, index] - point[1, index], "ci_lower": lower, "ci_upper": upper,
            "n_valid_bootstrap": valid, "n_bootstrap": n_bootstrap})
    summary = pd.DataFrame(summaries)
    contrast = pd.DataFrame(differences)
    bins = pd.DataFrame(bin_rows)
    summary.to_csv(destination / "cohort_metrics_with_ci.csv", index=False)
    contrast.to_csv(destination / "cohort_differences_with_ci.csv", index=False)
    bins.to_csv(destination / "calibration_bins_with_ci.csv", index=False)
    complete.to_csv(destination / "paired_test_predictions.csv", index=False)
    block_sizes.to_csv(destination / "bootstrap_block_sizes.csv")
    counts = pd.DataFrame([{"task": task, "cohort": cohort, "n_pairs": len(complete_ids),
        "n_admissions": len(part), "n_subjects": part.subject_id.nunique(), "n_positive": int(part.true_label.sum()),
        "decision_threshold": threshold} for cohort, part in zip(COHORTS, parts)])
    counts.to_csv(destination / "cohort_counts.csv", index=False)
    np.savez_compressed(destination / "bootstrap_replicates.npz", metrics=bootstrap,
                        calibration_bins=bootstrap_bins, metric_names=np.array(BOOTSTRAP_METRICS),
                        cohorts=np.array(list(COHORTS)))
    manifest = {
        "analysis": "primary_complete_pair_fairness", "task": task,
        "source_predictions_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "n_complete_pairs": len(complete_ids), "n_excluded_singleton_admissions": int(pair_counts.eq(1).sum()),
        "n_bootstrap_blocks": int(n_blocks), "largest_block_n_pairs": int(block_sizes.n_pairs.max()),
        "largest_block_fraction_of_pairs": float(block_sizes.n_pairs.max() / len(complete_ids)),
        "resampling_unit": "Connected component of patients linked by complete matched pairs in this evaluation population",
        "resampling_rule": "Sample n_blocks components with replacement; every admission in a selected component inherits its multiplicity; use the same draw for both cohorts",
        "bootstrap_samples": n_bootstrap, "seed": seed, "confidence_level": .95,
        "interval_method": "Pointwise percentile bootstrap, 2.5th and 97.5th percentiles; not simultaneous or multiplicity-adjusted",
        "weighting": "Point estimates are admission-weighted. Bootstrap sampling uses equal probabilities for blocks and retains block sizes.",
        "scope_of_uncertainty": "Evaluation-sample uncertainty conditional on frozen models, thresholds, original matching and retained-pair selection; excludes training/model-selection uncertainty",
        "calibration": "Probability clipping to [1e-6,1-1e-6] only for calibration logits; original probabilities used for other metrics; no recalibration",
        "zero_denominators_and_fit_failures": "Undefined metrics/nonconverged calibration fits are NaN; valid replicate counts reported per interval",
        "minimum_valid_bootstrap_count": int(summary.n_valid_bootstrap.min()),
        "largest_component_caution": "Unequal block sizes and large components can make percentile intervals sensitive to component sampling; sizes are exported",
        "model_retrained": False, "threshold_retuned": False, "original_predictions_modified": False,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_paired_calibration(task, bins, destination)
    print(contrast.loc[contrast.metric.isin(["auroc", "recall_sensitivity", "false_positive_rate", "observed_minus_predicted_percentage_points"])].to_string(index=False), flush=True)
    print(f"Saved paired primary analysis: {destination}", flush=True)


def plot_paired_calibration(task, bins, destination):
    """Fixed-bin reliability curves with pointwise block-bootstrap intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, (ax, hist) = plt.subplots(2, 1, figsize=(7.4, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    colors = {"MHC1-psychosis": "#7851A9", "MHC0": "#27864C"}
    ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1.2, label="Perfect calibration")
    for index, cohort in enumerate(COHORTS):
        part = bins.loc[bins.cohort.eq(cohort) & bins.n_admissions.gt(0)]
        ax.plot(part.mean_predicted_probability, part.observed_event_rate, color=colors[cohort], label=cohort)
        # Draw intervals as line segments rather than requiring the point
        # estimate to lie within a percentile bootstrap interval.
        ax.vlines(part.mean_predicted_probability, part.observed_event_rate_ci_lower,
                  part.observed_event_rate_ci_upper, color=colors[cohort], alpha=.65)
        for sparse in [False, True]:
            chosen = part.loc[part.n_admissions.lt(50).eq(sparse)]
            ax.scatter(chosen.mean_predicted_probability, chosen.observed_event_rate,
                       edgecolor=colors[cohort], facecolor="white" if sparse else colors[cohort], s=35, zorder=3)
        centers = (part.bin_lower + part.bin_upper) / 2 + (-.019 if index == 0 else .019)
        hist.bar(centers, part.n_admissions, width=.036, color=colors[cohort], alpha=.8)
    ax.set(title=("Hospital stay >7 days" if task == "los" else "Readmission within 30 days") + " — complete test pairs",
           ylabel="Observed outcome proportion", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", frameon=False)
    ax.text(.98, .02, "Pointwise 95% block-bootstrap intervals\nHollow points: fewer than 50 admissions",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=9, color=".3")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    hist.set(xlabel="Predicted probability", ylabel="Admissions", xticks=np.linspace(0, 1, 11))
    hist.xaxis.set_major_formatter(PercentFormatter(1))
    for axis in (ax, hist):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
    fig.savefig(destination / "calibration_plot.png", dpi=180)
    fig.savefig(destination / "calibration_plot.pdf")
    plt.close(fig)


def audit_test_pairs() -> None:
    """Inspect the existing test populations; do not change splits or predictions."""
    python_dir = SCRIPT_DIR.parent
    covariate_path = python_dir / "03_discharge_note_text_analysis/01_language_complexity/analysis_output_language_complexity_inference/language_complexity_model_covariates.csv"
    partition_path = SCRIPT_DIR / "01_training_data_creation/analysis_output_patient_level_prediction_partitions/current_matched_admission_partitions.csv"
    keys = ["pair_id", "cohort", "subject_id", "hadm_id"]
    variables = ["age_at_admission", "elixhauser_score", "n_prior_all_admissions_for_subject"]
    covariates = pd.read_csv(covariate_path, usecols=keys + variables)
    partitions = pd.read_csv(partition_path, usecols=keys + ["partition"])
    for name, frame in [("covariates", covariates), ("partitions", partitions)]:
        if frame[keys].isna().any().any() or frame.hadm_id.duplicated().any():
            raise ValueError(f"Invalid admission keys in {name}")
        grouped = frame.groupby("pair_id")
        if not grouped.size().eq(2).all() or not grouped.cohort.nunique().eq(2).all():
            raise ValueError(f"Expected complete original pairs in {name}")
    original = covariates.merge(partitions, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not original._merge.eq("both").all() or original[variables].isna().any().any():
        raise ValueError("Original partitions and covariates do not cover identical admissions")
    original = original.drop(columns="_merge")
    if original.groupby("subject_id").partition.nunique().gt(1).any():
        raise ValueError("A patient crosses the original training/validation/test partitions")
    if not set(original.cohort).issubset({"MHC0", "MHC1_psychotic"}):
        raise ValueError("Unexpected cohort labels")

    for task, output_dir in INFERENCE_OUTPUT_DIRS.items():
        prediction_path = output_dir / PREDICTIONS_FILENAME
        predictions = pd.read_csv(prediction_path)
        if predictions.hadm_id.duplicated().any() or predictions.partition.ne("test_pool").any():
            raise ValueError("Expected unique held-out prediction admissions")
        joined = predictions.merge(original[keys + variables], on=keys, how="left", validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all():
            raise ValueError("Prediction admissions do not match the original pair/covariate metadata")
        joined = joined.drop(columns="_merge")
        membership = original.copy()
        membership["in_task_predictions"] = membership.hadm_id.isin(predictions.hadm_id)
        membership["n_pair_members_in_task"] = membership.groupby("pair_id").in_task_predictions.transform("sum")
        complete_ids = membership.loc[membership.n_pair_members_in_task.eq(2), "pair_id"].unique()
        complete = joined.loc[joined.pair_id.isin(complete_ids)]
        pair_sizes = membership.groupby("pair_id").n_pair_members_in_task.first()
        retention = {
            "task": task, "n_original_pairs": int(len(pair_sizes)),
            "n_complete_test_pairs": int(pair_sizes.eq(2).sum()),
            "n_singleton_test_pairs": int(pair_sizes.eq(1).sum()),
            "n_pairs_with_neither_member_in_task": int(pair_sizes.eq(0).sum()),
            "n_test_admissions": len(joined), "n_test_admissions_in_complete_pairs": len(complete),
            "pct_test_admissions_in_complete_pairs": 100 * len(complete) / len(joined),
        }
        missing_partners = membership.loc[membership.n_pair_members_in_task.eq(1) & ~membership.in_task_predictions].copy()
        missing_partners["reason"] = missing_partners.partition.map({
            "train": "partner_in_training", "validation": "partner_in_validation",
            "test_pool": "partner_in_test_pool_but_not_in_task_predictions"})
        if missing_partners.reason.isna().any():
            raise ValueError("Unknown original partition")
        reasons = missing_partners.groupby(["cohort", "reason"]).size().rename("n_missing_partners").reset_index()
        reasons.insert(0, "task", task)
        balances, differences, metric_rows = [], [], []
        for population, frame in [("original_matched_cohort", original),
                                  ("task_test_all", joined), ("task_test_complete_pairs", complete)]:
            for variable in variables:
                for cohort, rows in frame.groupby("cohort"):
                    values = rows[variable]
                    balances.append({"task": task, "population": population, "cohort": cohort,
                        "variable": variable, "n_admissions": len(rows), "n_subjects": rows.subject_id.nunique(),
                        "mean": values.mean(), "sd": values.std(), "median": values.median(),
                        "q25": values.quantile(.25), "q75": values.quantile(.75)})
                case = frame.loc[frame.cohort.eq("MHC1_psychotic"), variable]
                control = frame.loc[frame.cohort.eq("MHC0"), variable]
                denominator = np.sqrt((case.var() + control.var()) / 2)
                differences.append({"task": task, "population": population, "variable": variable,
                    "mean_difference_mhc1_minus_mhc0": case.mean() - control.mean(),
                    "standardized_mean_difference": (case.mean() - control.mean()) / denominator if denominator > 0 else np.nan})
            if population != "original_matched_cohort":
                for cohort, flag in COHORTS.items():
                    rows = frame.loc[frame[flag].eq(1)]
                    metric_rows.append({"task": task, "population": population, **summarize_cohort(rows, cohort)})
        destination = output_dir / "test_pair_audit"
        destination.mkdir(exist_ok=True)
        pd.DataFrame([retention]).to_csv(destination / "pair_retention_summary.csv", index=False)
        reasons.to_csv(destination / "missing_partner_reasons.csv", index=False)
        pd.DataFrame(balances).to_csv(destination / "covariate_summary.csv", index=False)
        pd.DataFrame(differences).to_csv(destination / "covariate_balance.csv", index=False)
        pd.DataFrame(metric_rows).to_csv(destination / "performance_by_pair_retention.csv", index=False)
        membership[keys + ["partition", "in_task_predictions", "n_pair_members_in_task"]].to_csv(
            destination / "admission_pair_membership.csv", index=False)
        (destination / "manifest.json").write_text(json.dumps({
            "task": task, "covariate_source": str(covariate_path), "partition_source": str(partition_path),
            "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "balance_definition": "Admission-weighted means; SMD=(MHC1 mean-MHC0 mean)/sqrt((MHC1 sample variance+MHC0 sample variance)/2)",
            "complete_pair_rule": "Both original matched admissions are in this task's saved predictions",
            "selection_note": "Complete-pair results are a descriptive sensitivity check; not a replacement test set or new model selection",
            "splits_and_predictions_modified": False, "confidence_intervals_or_tests": False,
        }, indent=2) + "\n")
        print(json.dumps(retention, indent=2))
        print(reasons.to_string(index=False))
        print(pd.DataFrame(differences).to_string(index=False))
        print(f"Wrote audit: {destination}")


if __name__ == "__main__":
    main()
