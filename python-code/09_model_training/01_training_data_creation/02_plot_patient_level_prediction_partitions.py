"""Plot patient-level prediction partitions created in DuckDB.

This script joins current full/matched cohort IDs to the original patient
assignments in DuckDB, verifies those assignments against archived datasets,
and rebuilds cohort summaries. It never resamples patients or alters DuckDB.
"""

from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_CODE_DIR = SCRIPT_DIR.parents[2]
THESIS_DIR = THESIS_CODE_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(THESIS_CODE_DIR / ".matplotlib"))

import matplotlib.pyplot as plt

DB_PATH = Path(
    os.environ.get(
        "PREDICTION_PARTITION_DB_PATH",
        str(THESIS_DIR / "DataBase"),
    )
)
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_patient_level_prediction_partitions"

PARTITION_ORDER = ["train", "validation", "test_pool"]
MATCHED_PAIRS_PATH = THESIS_CODE_DIR / "python-code/02_cohort_matching/matched_cohort_output/matched_pairs.csv"
ARCHIVED_DATA_DIR = (
    THESIS_CODE_DIR / "analysis-archive/MHH1_original_2026-09-25/python-code/08_model_training"
    / "01_training_data_creation/prediction_model_dataset"
)


def quote_identifier(identifier: str) -> str:
    """Quote a DuckDB identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def load_duckdb_table(table_name: str) -> pd.DataFrame:
    """Load one table from the configured DuckDB database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DuckDB database: {DB_PATH}")
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        tables = set(con.execute("SHOW TABLES").fetchdf().iloc[:, 0].astype(str))
        if table_name not in tables:
            raise FileNotFoundError(
                f"Missing DuckDB table {table_name!r} in {DB_PATH}"
            )
        table = con.execute(f"SELECT * FROM {quote_identifier(table_name)}").fetchdf()
    table.columns = [str(column).strip().lower() for column in table.columns]
    return table


def write_summary_tables() -> dict[str, pd.DataFrame]:
    """Recalculate current group summaries using unchanged original assignments."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DuckDB database: {DB_PATH}")
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        partitions = con.execute(
            'SELECT subject_id, partition, selected_for_general_test FROM patient_partition_with_test_flags'
        ).fetchdf()
        eligible = con.execute(
            'SELECT DISTINCT subject_id, hadm_id, partition FROM eligible_prediction_admissions_with_partition'
        ).fetchdf()
        full_case = con.execute('SELECT DISTINCT subject_id, hadm_id FROM export_MHC1_psychotic').fetchdf()
        full_control = con.execute('SELECT DISTINCT subject_id, hadm_id FROM export_only_MHC0').fetchdf()
    if partitions.subject_id.duplicated().any() or eligible.duplicated(["subject_id", "hadm_id"]).any():
        raise ValueError("Duplicate original partition identifiers")
    if not partitions.partition.isin(PARTITION_ORDER).all():
        raise ValueError("Unknown original partition")
    if not partitions.selected_for_general_test.isin([0, 1]).all():
        raise ValueError("Invalid original general-test flag")
    if partitions.loc[partitions.partition.ne("test_pool"), "selected_for_general_test"].ne(0).any():
        raise ValueError("General-test flag outside the original test pool")
    original = eligible.merge(partitions, on="subject_id", suffixes=("_admission", ""),
                              how="left", validate="many_to_one")
    if not original.partition_admission.eq(original.partition).all():
        raise ValueError("Patient and admission partition assignments disagree")
    original = original.drop(columns="partition_admission")
    qc = []
    for split, partition in [("train", "train"), ("validation", "validation"),
                             ("test_general", "test_pool"), ("test_fairness_mhh1_mhc0", "test_pool")]:
        archived = pd.read_parquet(ARCHIVED_DATA_DIR / f"{split}.parquet", columns=["subject_id", "hadm_id"]).drop_duplicates()
        checked = archived.merge(original, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
        if not checked.partition.eq(partition).all():
            raise ValueError(f"Stored partitions no longer match archived {split}")
        if split in {"train", "validation"}:
            if set(archived.subject_id) != set(partitions.loc[partitions.partition.eq(partition), "subject_id"]):
                raise ValueError(f"Original {partition} patient membership has changed")
        if split == "test_general" and not checked.selected_for_general_test.eq(1).all():
            raise ValueError("Original general-test membership has changed")
        qc.append({"check": f"archived_{split}_partition_mismatches", "value": 0})
    pairs = pd.read_csv(MATCHED_PAIRS_PATH, usecols=["pair_id", "mhc1_subject_id", "mhc1_hadm_id", "mhc0_subject_id", "mhc0_hadm_id"])
    matched = {}
    for cohort, prefix in [("MHC1_psychotic", "mhc1"), ("MHC0", "mhc0")]:
        rows = pairs[["pair_id", f"{prefix}_subject_id", f"{prefix}_hadm_id"]].rename(
            columns={f"{prefix}_subject_id": "subject_id", f"{prefix}_hadm_id": "hadm_id"})
        rows["cohort"] = cohort
        matched[cohort] = rows
    if set(full_case.subject_id) & set(full_control.subject_id):
        raise ValueError("Current full case/control patients overlap")
    group_inputs = {
        "MHC1_psychotic_full_cohort": full_case,
        "MHC0_full_cohort": full_control,
        "MHC1_psychotic_matched_cohort": matched["MHC1_psychotic"],
        "MHC0_matched_cohort": matched["MHC0"],
    }
    for cohort, full in [("MHC1_psychotic", full_case), ("MHC0", full_control)]:
        check = matched[cohort].merge(full, on=["subject_id", "hadm_id"], how="left", indicator=True)
        if not check._merge.eq("both").all():
            raise ValueError(f"Current matched {cohort} is not a subset of its full export")
    group_summaries, unseen_summaries, overlap_summaries, annotated_matched = [], [], [], []
    for name, rows in group_inputs.items():
        if rows.duplicated(["subject_id", "hadm_id"]).any():
            raise ValueError(f"Duplicate current admissions: {name}")
        joined = rows.merge(original, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
        if joined.partition.isna().any():
            raise ValueError(f"Current admissions missing an original partition: {name}")
        summary = joined.groupby("partition").agg(n_subjects=("subject_id", "nunique"), n_admissions=("hadm_id", "nunique"))
        summary = summary.reindex(PARTITION_ORDER, fill_value=0).reset_index()
        summary["group_name"] = name
        summary["pct_subjects_within_group"] = 100 * summary.n_subjects / joined.subject_id.nunique()
        summary["pct_admissions_within_group"] = 100 * summary.n_admissions / len(joined)
        group_summaries.append(summary)
        unseen = joined.loc[joined.partition.eq("test_pool")]
        unseen_summaries.append({"group_name": name, "n_unseen_subjects_in_test_pool": unseen.subject_id.nunique(),
                                 "n_unseen_admissions_in_test_pool": len(unseen)})
        general = unseen.loc[unseen.selected_for_general_test.eq(1)]
        overlap_summaries.append({"group_name": name, "n_subjects_in_general_test": general.subject_id.nunique(),
                                  "n_admissions_in_general_test": len(general)})
        if name.endswith("matched_cohort"):
            annotated_matched.append(joined)
        qc.append({"check": f"{name}_unassigned_admissions", "value": 0})
    summary = original.groupby("partition", as_index=False).agg(
        n_subjects=("subject_id", "nunique"), n_admissions=("hadm_id", "nunique"))
    general_summary = original.loc[original.partition.eq("test_pool")].groupby(
        "selected_for_general_test", as_index=False).agg(n_subjects=("subject_id", "nunique"), n_admissions=("hadm_id", "nunique"))
    tables = {
        "patient_partition_summary": summary,
        "patient_partition_group_summary": pd.concat(group_summaries, ignore_index=True),
        "patient_partition_unseen_fairness_pool_summary": pd.DataFrame(unseen_summaries),
        "patient_partition_general_test_summary": general_summary,
        "patient_partition_general_test_group_overlap": pd.DataFrame(overlap_summaries),
        "current_matched_admission_partitions": pd.concat(annotated_matched, ignore_index=True),
        "patient_partition_qc": pd.DataFrame(qc),
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    fingerprint = hashlib.sha256(partitions.sort_values("subject_id").to_csv(index=False).encode()).hexdigest()
    (OUTPUT_DIR / "partition_summary_manifest.json").write_text(json.dumps({
        "assignment_source": str(DB_PATH), "original_assignments_sha256": fingerprint,
        "matched_pairs_sha256": hashlib.sha256(MATCHED_PAIRS_PATH.read_bytes()).hexdigest(),
        "cohort_source": "current export_MHC1_psychotic / export_only_MHC0 and matched_pairs.csv",
        "archived_splits_verified": True, "repartitioned": False,
    }, indent=2))
    return tables


def plot_group_partition_percentages(group_summary: pd.DataFrame) -> None:
    """Show patient/admission percentages without treating counts as quotas."""
    colors = ["#4477AA", "#EEAA33", "#228833"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, column, title in zip(axes, ["pct_subjects_within_group", "pct_admissions_within_group"],
                                 ["Patients", "Admissions"]):
        pivot = group_summary.pivot(index="group_name", columns="partition", values=column).reindex(columns=PARTITION_ORDER)
        labels = [name.replace("MHC1_psychotic", "MHC1-psychosis").replace("_full_cohort", " (full)")
                  .replace("_matched_cohort", " (matched)") for name in pivot.index]
        left = pd.Series(0.0, index=pivot.index)
        for partition, color in zip(PARTITION_ORDER, colors):
            values = pivot[partition]
            ax.barh(labels, values, left=left, color=color, label=partition.replace("_", " "))
            for y, (start, value) in enumerate(zip(left, values)):
                if value >= 1:
                    ax.text(start + value / 2, y, f"{value:.1f}%", ha="center", va="center", fontsize=9)
            left = left + values
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage within cohort")
        ax.set_title(title)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.suptitle("Current cohorts in original prediction partitions (unchanged assignments)")
    fig.savefig(OUTPUT_DIR / "patient_partition_group_percentages.png", dpi=200)
    plt.close(fig)


def add_partition_order(table: pd.DataFrame) -> pd.DataFrame:
    """Attach a stable plotting order for train/validation/test_pool."""
    output = table.copy()
    output["partition"] = pd.Categorical(
        output["partition"],
        categories=PARTITION_ORDER,
        ordered=True,
    )
    return output.sort_values("partition")


def annotate_bars(ax: plt.Axes) -> None:
    """Print exact counts above bars."""
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:,.0f}",
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_partition_totals(summary: pd.DataFrame) -> None:
    """Plot total subjects and admissions by partition."""
    summary = add_partition_order(summary)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, column, title in [
        (axes[0], "n_subjects", "Subjects"),
        (axes[1], "n_admissions", "Admissions"),
    ]:
        ax.bar(summary["partition"].astype(str), summary[column], color="#4c78a8")
        ax.set_title(title)
        ax.set_xlabel("Partition")
        ax.set_ylabel("Count")
        annotate_bars(ax)
    fig.suptitle("Patient-Level Prediction Partitions")
    fig.savefig(OUTPUT_DIR / "patient_partition_total_counts.png", dpi=200)
    plt.close(fig)


def plot_group_partition_counts(group_summary: pd.DataFrame) -> None:
    """Plot MHC1/MHC0 full and matched cohort counts by partition."""
    group_summary = add_partition_order(group_summary)
    group_order = sorted(group_summary["group_name"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for ax, column, title in [
        (axes[0], "n_subjects", "Subjects by Cohort Flag"),
        (axes[1], "n_admissions", "Admissions by Cohort Flag"),
    ]:
        pivot = (
            group_summary.pivot_table(
                index="group_name",
                columns="partition",
                values=column,
                aggfunc="sum",
                fill_value=0,
                observed=False,
            )
            .reindex(group_order)
            .reindex(columns=PARTITION_ORDER)
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend(title="Partition")
        annotate_bars(ax)
    fig.suptitle("MHC1/MHC0 Representation in Prediction Partitions")
    fig.savefig(OUTPUT_DIR / "patient_partition_group_counts.png", dpi=200)
    plt.close(fig)


def plot_unseen_fairness_pool(unseen: pd.DataFrame) -> None:
    """Plot unseen test-pool MHC1/MHC0 counts available for fairness work."""
    unseen = unseen.sort_values("group_name")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, column, title in [
        (axes[0], "n_unseen_subjects_in_test_pool", "Unseen Subjects"),
        (axes[1], "n_unseen_admissions_in_test_pool", "Unseen Admissions"),
    ]:
        ax.bar(unseen["group_name"], unseen[column], color="#59a14f")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=25)
        annotate_bars(ax)
    fig.suptitle("Held-Out Fairness Evaluation Pool")
    fig.savefig(OUTPUT_DIR / "patient_partition_unseen_fairness_pool_counts.png", dpi=200)
    plt.close(fig)


def plot_general_test_summary(general_test: pd.DataFrame) -> None:
    """Plot the selected general test sample size inside test_pool."""
    selected = general_test.loc[general_test["selected_for_general_test"].eq(1)].copy()
    if selected.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for ax, column, title in [
        (axes[0], "n_subjects", "Selected Subjects"),
        (axes[1], "n_admissions", "Selected Admissions"),
    ]:
        ax.bar(["test_general"], selected[column], color="#f28e2b")
        ax.set_title(title)
        ax.set_ylabel("Count")
        annotate_bars(ax)
    fig.suptitle("General Test Sample Selected From Test Pool")
    fig.savefig(OUTPUT_DIR / "patient_partition_general_test_counts.png", dpi=200)
    plt.close(fig)


def plot_general_test_group_overlap(overlap: pd.DataFrame) -> None:
    """Plot MHC1/MHC0 rows that also landed in the general test sample."""
    overlap = overlap.sort_values("group_name")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, column, title in [
        (axes[0], "n_subjects_in_general_test", "Subjects"),
        (axes[1], "n_admissions_in_general_test", "Admissions"),
    ]:
        ax.bar(overlap["group_name"], overlap[column], color="#e15759")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=25)
        annotate_bars(ax)
    fig.suptitle("MHC1/MHC0 Overlap With General Test Sample")
    fig.savefig(OUTPUT_DIR / "patient_partition_general_test_group_overlap.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Write partition summaries and plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = write_summary_tables()
    plot_partition_totals(tables["patient_partition_summary"])
    plot_group_partition_counts(tables["patient_partition_group_summary"])
    plot_group_partition_percentages(tables["patient_partition_group_summary"])
    plot_unseen_fairness_pool(
        tables["patient_partition_unseen_fairness_pool_summary"]
    )
    plot_general_test_summary(tables["patient_partition_general_test_summary"])
    plot_general_test_group_overlap(
        tables["patient_partition_general_test_group_overlap"]
    )

    print(f"Read partition tables from: {DB_PATH}")
    print(f"Wrote CSV summaries and plots to: {OUTPUT_DIR}")
    print("\nPartition summary:")
    print(tables["patient_partition_summary"].to_string(index=False))
    print("\nMHC1/MHC0 partition summary:")
    print(tables["patient_partition_group_summary"].to_string(index=False))
    print("\nGeneral test sample summary:")
    print(tables["patient_partition_general_test_summary"].to_string(index=False))
    print("\nMHC1/MHC0 overlap with general test sample:")
    print(tables["patient_partition_general_test_group_overlap"].to_string(index=False))


if __name__ == "__main__":
    main()
