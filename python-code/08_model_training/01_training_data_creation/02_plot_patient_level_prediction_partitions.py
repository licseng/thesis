"""Plot patient-level prediction partitions created in DuckDB.

This script reads the partition summary tables created by:

    sql-scripts/06_save_tables/03_create_patient_level_prediction_partitions.sql

It does not create or alter the train/validation/test split. It only exports
the SQL summary tables to CSV and makes annotated count plots.
"""

from __future__ import annotations

import os
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
    """Load and save the SQL summary tables used for plotting."""
    tables = {
        "patient_partition_summary": load_duckdb_table("patient_partition_summary"),
        "patient_partition_group_summary": load_duckdb_table(
            "patient_partition_group_summary"
        ),
        "patient_partition_unseen_fairness_pool_summary": load_duckdb_table(
            "patient_partition_unseen_fairness_pool_summary"
        ),
        "patient_partition_general_test_summary": load_duckdb_table(
            "patient_partition_general_test_summary"
        ),
        "patient_partition_general_test_group_overlap": load_duckdb_table(
            "patient_partition_general_test_group_overlap"
        ),
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    return tables


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
    """Plot MHH1/MHC0 full and matched cohort counts by partition."""
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
    fig.suptitle("MHH1/MHC0 Representation in Prediction Partitions")
    fig.savefig(OUTPUT_DIR / "patient_partition_group_counts.png", dpi=200)
    plt.close(fig)


def plot_unseen_fairness_pool(unseen: pd.DataFrame) -> None:
    """Plot unseen test-pool MHH1/MHC0 counts available for fairness work."""
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
    """Plot MHH1/MHC0 rows that also landed in the general test sample."""
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
    fig.suptitle("MHH1/MHC0 Overlap With General Test Sample")
    fig.savefig(OUTPUT_DIR / "patient_partition_general_test_group_overlap.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Write partition summaries and plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = write_summary_tables()
    plot_partition_totals(tables["patient_partition_summary"])
    plot_group_partition_counts(tables["patient_partition_group_summary"])
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
    print("\nMHH1/MHC0 partition summary:")
    print(tables["patient_partition_group_summary"].to_string(index=False))
    print("\nGeneral test sample summary:")
    print(tables["patient_partition_general_test_summary"].to_string(index=False))
    print("\nMHH1/MHC0 overlap with general test sample:")
    print(tables["patient_partition_general_test_group_overlap"].to_string(index=False))


if __name__ == "__main__":
    main()
