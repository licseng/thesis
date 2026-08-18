"""Characterize matched cohorts with exported admission descriptors/utilization.

This script analyzes extra matched-cohort descriptors exported from DBeaver. It
expects one small descriptor table and optional event/order tables already
restricted to the matched MHH1_psychotic and MHC0 admissions.

Default input folder:
    matched_cohort_dbeaver_exports/

Expected file basenames, with .csv or .parquet extension:
    export_matched_cohort_descriptors
    export_matched_cohort_diagnoses
    export_matched_cohort_labevents
    export_matched_cohort_microbiologyevents
    export_matched_cohort_poe
    export_matched_cohort_poe_detail

Outputs are aggregate summaries only. The script does not write row-level event
details or clinical free text.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
THESIS_DIR = PROJECT_DIR.parent.parent
DB_PATH = Path(
    os.environ.get(
        "MATCHED_COHORT_CHARACTERIZATION_DB_PATH",
        str(THESIS_DIR / "DataBase"),
    )
)
MATCHED_IDS_PATH = SCRIPT_DIR / "matched_cohort_output" / "matched_admission_ids_for_dbeaver.csv"
INPUT_DIR = Path(
    os.environ.get(
        "MATCHED_COHORT_CHARACTERIZATION_INPUT_DIR",
        str(SCRIPT_DIR / "matched_cohort_dbeaver_exports"),
    )
)
BASE_OUTPUT_DIR = Path(
    os.environ.get(
        "MATCHED_COHORT_CHARACTERIZATION_OUTPUT_DIR",
        str(SCRIPT_DIR / "analysis_output_matched_cohort_characterization"),
    )
)
ADMISSION_LEVEL_OUTPUT_DIR = Path(
    os.environ.get(
        "MATCHED_COHORT_ADMISSION_CHARACTERIZATION_OUTPUT_DIR",
        str(SCRIPT_DIR / "analysis_output_matched_cohort_characterization_admission_level"),
    )
)
SUBJECT_LEVEL_OUTPUT_DIR = Path(
    os.environ.get(
        "MATCHED_COHORT_SUBJECT_CHARACTERIZATION_OUTPUT_DIR",
        str(SCRIPT_DIR / "analysis_output_matched_cohort_characterization_subject_level"),
    )
)

EXPORT_BASENAMES = {
    "descriptors": "export_matched_cohort_descriptors",
    "diagnoses": "export_matched_cohort_diagnoses",
    "subject_admission_history": "export_matched_cohort_subject_admission_history",
    "labevents": "export_matched_cohort_labevents",
    "microbiologyevents": "export_matched_cohort_microbiologyevents",
    "poe": "export_matched_cohort_poe",
    "poe_detail": "export_matched_cohort_poe_detail",
}
SUPPORTED_SUFFIXES = [".parquet", ".csv", ".csv.gz"]
ID_COLUMNS = ["cohort", "subject_id", "hadm_id"]

CATEGORICAL_DESCRIPTOR_COLUMNS = [
    "insurance",
    "race",
    "race_group",
    "ethnicity",
    "ethnicity_from_race",
    "language",
    "marital_status",
    "admission_type",
    "admission_location",
    "discharge_location",
]


def find_export_path(basename: str, required: bool = False) -> Path | None:
    """Find an exported table by basename and supported file extension."""
    for suffix in SUPPORTED_SUFFIXES:
        path = INPUT_DIR / f"{basename}{suffix}"
        if path.exists():
            return path
    if required:
        expected = "\n".join(str(INPUT_DIR / f"{basename}{suffix}") for suffix in SUPPORTED_SUFFIXES)
        raise FileNotFoundError(
            f"Missing required DBeaver export for {basename}. Expected one of:\n{expected}"
        )
    return None


def quote_identifier(identifier: str) -> str:
    """Quote a DuckDB table/column identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def duckdb_table_exists(table_name: str) -> bool:
    """Return whether a table exists in the configured DuckDB database."""
    if not DB_PATH.exists():
        return False
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tables = con.execute("SHOW TABLES").fetchdf().iloc[:, 0].astype(str).tolist()
        return table_name in set(tables)
    finally:
        con.close()


def load_duckdb_table(table_name: str) -> pd.DataFrame:
    """Load a table from the configured DuckDB database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DuckDB database: {DB_PATH}")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        table = con.execute(
            f"SELECT * FROM {quote_identifier(table_name)}"
        ).fetchdf()
    finally:
        con.close()
    table.columns = [str(column).strip().lower() for column in table.columns]
    return table


def load_table(path: Path) -> pd.DataFrame:
    """Load a CSV/parquet export and normalize column names."""
    if path.suffix == ".parquet":
        table = pd.read_parquet(path)
    else:
        table = pd.read_csv(path)
    table.columns = [str(column).strip().lower() for column in table.columns]
    return table


def load_required_table(name: str) -> pd.DataFrame:
    """Load a required DBeaver export from file or DuckDB table."""
    table_name = EXPORT_BASENAMES[name]
    path = find_export_path(table_name, required=False)
    if path is not None:
        return load_table(path)
    if duckdb_table_exists(table_name):
        return load_duckdb_table(table_name)
    expected = "\n".join(
        str(INPUT_DIR / f"{table_name}{suffix}") for suffix in SUPPORTED_SUFFIXES
    )
    raise FileNotFoundError(
        f"Missing required DBeaver export for {table_name}. Expected either a "
        f"DuckDB table named {table_name} in {DB_PATH}, or one of:\n{expected}"
    )


def load_optional_table(name: str) -> pd.DataFrame | None:
    """Load an optional DBeaver export from file or DuckDB table."""
    table_name = EXPORT_BASENAMES[name]
    path = find_export_path(table_name, required=False)
    if path is None:
        if duckdb_table_exists(table_name):
            return load_duckdb_table(table_name)
        return None
    return load_table(path)


def validate_id_columns(table: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Validate and standardize cohort/admission identifier columns."""
    missing = sorted(set(ID_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required ID columns: {missing}")
    clean = table.copy()
    clean["cohort"] = clean["cohort"].astype("string").str.strip()
    clean["subject_id"] = pd.to_numeric(clean["subject_id"], errors="raise").astype(int)
    clean["hadm_id"] = pd.to_numeric(clean["hadm_id"], errors="raise").astype(int)
    return clean


def load_expected_matched_ids() -> pd.DataFrame:
    """Load the matched admission ID helper table used for DBeaver filtering."""
    if not MATCHED_IDS_PATH.exists():
        raise FileNotFoundError(f"Missing matched ID file: {MATCHED_IDS_PATH}")
    matched_ids = pd.read_csv(MATCHED_IDS_PATH)
    return validate_id_columns(matched_ids, "matched_admission_ids_for_dbeaver")


def make_admission_key_set(table: pd.DataFrame) -> set[tuple[str, int, int]]:
    """Return cohort + subject_id + hadm_id keys for an admission-level table."""
    return set(
        map(
            tuple,
            table.loc[:, ID_COLUMNS].drop_duplicates().to_numpy(),
        )
    )


def derive_race_group(value: object) -> str:
    """Map MIMIC race strings to coarse race/ethnicity groups."""
    race = str(value).strip().upper()
    if not race or race in {"NAN", "NONE"}:
        return "missing"
    if "DECLINED" in race or "UNABLE" in race or "UNKNOWN" in race:
        return "unknown_or_declined"
    if "HISPANIC" in race or "LATINO" in race:
        return "hispanic_or_latino"
    if "WHITE" in race:
        return "white"
    if "BLACK" in race or "AFRICAN" in race:
        return "black"
    if "ASIAN" in race:
        return "asian"
    if "AMERICAN INDIAN" in race or "ALASKA" in race:
        return "american_indian_or_alaska_native"
    if "NATIVE HAWAIIAN" in race or "PACIFIC ISLANDER" in race:
        return "native_hawaiian_or_pacific_islander"
    if "MULTIPLE" in race:
        return "multiple"
    return "other"


def derive_ethnicity_from_race(value: object) -> str:
    """Derive a Hispanic/Latino indicator from MIMIC's combined race field."""
    race = str(value).strip().upper()
    if not race or race in {"NAN", "NONE"}:
        return "missing"
    if "DECLINED" in race or "UNABLE" in race or "UNKNOWN" in race:
        return "unknown_or_declined"
    if "HISPANIC" in race or "LATINO" in race:
        return "hispanic_or_latino"
    return "not_hispanic_or_latino"


def add_derived_descriptor_columns(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Add coarse race/ethnicity variables when raw race is available."""
    output = descriptors.copy()
    if "race" in output.columns:
        output["race_group"] = output["race"].map(derive_race_group)
        if "ethnicity" not in output.columns:
            output["ethnicity_from_race"] = output["race"].map(derive_ethnicity_from_race)
    return output


def build_descriptor_completeness(
    matched_ids: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize descriptor coverage against the expected matched admissions."""
    expected_keys = make_admission_key_set(matched_ids)
    descriptor_keys = make_admission_key_set(descriptors)
    duplicated_descriptor_rows = int(
        descriptors.duplicated(ID_COLUMNS, keep=False).sum()
    )
    return pd.DataFrame(
        [
            {
                "n_expected_matched_admissions": len(expected_keys),
                "n_descriptor_rows": len(descriptors),
                "n_unique_descriptor_admissions": len(descriptor_keys),
                "n_missing_descriptor_admissions": len(expected_keys - descriptor_keys),
                "n_unexpected_descriptor_admissions": len(descriptor_keys - expected_keys),
                "n_duplicated_descriptor_rows": duplicated_descriptor_rows,
            }
        ]
    )


def build_categorical_distribution(
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    """Build n/% tables for available categorical descriptor columns."""
    available_columns = [
        column for column in CATEGORICAL_DESCRIPTOR_COLUMNS if column in descriptors.columns
    ]
    rows = []
    cohort_denominators = descriptors.groupby("cohort")["hadm_id"].nunique().to_dict()
    for variable in available_columns:
        values = descriptors.loc[:, ["cohort", "hadm_id", variable]].copy()
        values[variable] = values[variable].fillna("missing").astype(str).str.strip()
        values.loc[values[variable].eq(""), variable] = "missing"
        counts = (
            values.groupby(["cohort", variable], as_index=False)["hadm_id"]
            .nunique()
            .rename(columns={variable: "category", "hadm_id": "n_admissions"})
        )
        counts["variable"] = variable
        counts["pct_within_cohort"] = counts.apply(
            lambda row: 100.0
            * row["n_admissions"]
            / cohort_denominators.get(row["cohort"], 0),
            axis=1,
        )
        rows.append(counts)
    if not rows:
        return pd.DataFrame(
            columns=["variable", "cohort", "category", "n_admissions", "pct_within_cohort"]
        )
    distribution = pd.concat(rows, ignore_index=True)
    return distribution.loc[
        :,
        ["variable", "cohort", "category", "n_admissions", "pct_within_cohort"],
    ].sort_values(["variable", "cohort", "n_admissions"], ascending=[True, True, False])


def build_categorical_balance(categorical_distribution: pd.DataFrame) -> pd.DataFrame:
    """Pivot categorical percentages and compute MHH1-MHC0 percentage difference."""
    if categorical_distribution.empty:
        return categorical_distribution.copy()
    pivot = categorical_distribution.pivot_table(
        index=["variable", "category"],
        columns="cohort",
        values=["n_admissions", "pct_within_cohort"],
        fill_value=0,
        aggfunc="sum",
    )
    pivot.columns = [
        f"{metric}_{cohort}".lower()
        for metric, cohort in pivot.columns.to_flat_index()
    ]
    pivot = pivot.reset_index()
    mhh_pct = "pct_within_cohort_mhh1_psychotic"
    mhc0_pct = "pct_within_cohort_mhc0"
    if mhh_pct in pivot.columns and mhc0_pct in pivot.columns:
        pivot["pct_point_difference_mhh1_minus_mhc0"] = pivot[mhh_pct] - pivot[mhc0_pct]
    return pivot.sort_values(["variable", "category"])


def build_event_counts_by_admission(
    matched_ids: pd.DataFrame,
    event_tables: dict[str, pd.DataFrame | None],
) -> pd.DataFrame:
    """Count optional event/order rows per matched admission."""
    counts = matched_ids.loc[:, ["cohort", "matched_role", "subject_id", "hadm_id"]].copy()
    for event_name, table in event_tables.items():
        count_column = f"n_{event_name}_rows"
        if table is None:
            counts[count_column] = 0
            continue
        clean = validate_id_columns(table, event_name)
        event_counts = (
            clean.groupby(ID_COLUMNS, as_index=False)
            .size()
            .rename(columns={"size": count_column})
        )
        counts = counts.merge(event_counts, on=ID_COLUMNS, how="left")
        counts[count_column] = counts[count_column].fillna(0).astype(int)
    return counts


def build_utilization_summary(counts_by_admission: pd.DataFrame) -> pd.DataFrame:
    """Summarize event/order counts by cohort."""
    count_columns = [
        column for column in counts_by_admission.columns if column.startswith("n_")
    ]
    rows = []
    for cohort, group in counts_by_admission.groupby("cohort"):
        for column in count_columns:
            values = group[column]
            rows.append(
                {
                    "cohort": cohort,
                    "measure": column,
                    "n_admissions": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "iqr": values.quantile(0.75) - values.quantile(0.25),
                    "min": values.min(),
                    "max": values.max(),
                    "n_with_any": int(values.gt(0).sum()),
                    "pct_with_any": 100.0 * values.gt(0).mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["measure", "cohort"])


def build_optional_category_distribution(
    table: pd.DataFrame | None,
    table_name: str,
    candidate_columns: list[str],
) -> pd.DataFrame:
    """Summarize optional event/order category columns if present."""
    if table is None:
        return pd.DataFrame()
    clean = validate_id_columns(table, table_name)
    rows = []
    for column in candidate_columns:
        if column not in clean.columns:
            continue
        values = clean.loc[:, ["cohort", "hadm_id", column]].copy()
        values[column] = values[column].fillna("missing").astype(str).str.strip()
        values.loc[values[column].eq(""), column] = "missing"
        counts = (
            values.groupby(["cohort", column], as_index=False)
            .agg(
                n_rows=("hadm_id", "size"),
                n_admissions=("hadm_id", "nunique"),
            )
            .rename(columns={column: "category"})
        )
        counts["source_table"] = table_name
        counts["variable"] = column
        rows.append(counts)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).loc[
        :, ["source_table", "variable", "cohort", "category", "n_rows", "n_admissions"]
    ].sort_values(["source_table", "variable", "cohort", "n_rows"], ascending=[True, True, True, False])


def build_admissions_per_subject_summary(matched_ids: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated admissions per subject by cohort."""
    admissions_per_subject = (
        matched_ids.groupby(["cohort", "subject_id"], as_index=False)
        .agg(n_matched_admissions=("hadm_id", "nunique"))
    )
    rows = []
    for cohort, group in admissions_per_subject.groupby("cohort"):
        values = group["n_matched_admissions"]
        rows.append(
            {
                "cohort": cohort,
                "n_subjects": len(values),
                "n_admissions": int(values.sum()),
                "mean_admissions_per_subject": values.mean(),
                "sd_admissions_per_subject": values.std(ddof=1),
                "median_admissions_per_subject": values.median(),
                "q1_admissions_per_subject": values.quantile(0.25),
                "q3_admissions_per_subject": values.quantile(0.75),
                "max_admissions_per_subject": values.max(),
                "n_subjects_with_1_admission": int(values.eq(1).sum()),
                "n_subjects_with_multiple_admissions": int(values.gt(1).sum()),
                "pct_subjects_with_multiple_admissions": 100.0 * values.gt(1).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("cohort")


def build_admissions_per_subject_distribution(matched_ids: pd.DataFrame) -> pd.DataFrame:
    """Count subjects by exact number of matched admissions."""
    admissions_per_subject = (
        matched_ids.groupby(["cohort", "subject_id"], as_index=False)
        .agg(n_matched_admissions=("hadm_id", "nunique"))
    )
    distribution = (
        admissions_per_subject.groupby(["cohort", "n_matched_admissions"], as_index=False)
        .agg(n_subjects=("subject_id", "nunique"))
        .sort_values(["cohort", "n_matched_admissions"])
    )
    denominators = admissions_per_subject.groupby("cohort")["subject_id"].nunique().to_dict()
    distribution["pct_subjects_within_cohort"] = distribution.apply(
        lambda row: 100.0
        * row["n_subjects"]
        / denominators.get(row["cohort"], 0),
        axis=1,
    )
    distribution["n_admissions_represented"] = (
        distribution["n_subjects"] * distribution["n_matched_admissions"]
    )
    return distribution


def prior_all_admission_bucket(n_prior: int) -> str:
    """Return a compact bucket for prior admissions across all MIMIC admissions."""
    if n_prior <= 5:
        return str(n_prior)
    if n_prior <= 10:
        return "6-10"
    return "11+"


def build_prior_all_mimic_admission_summary(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Summarize true prior hospital admissions for matched admissions.

    This uses the full MIMIC admissions table-derived column
    `n_prior_all_admissions_for_subject`, not the count of repeated admissions
    inside the matched cohort.
    """
    required_columns = {
        "cohort",
        "subject_id",
        "hadm_id",
        "n_prior_all_admissions_for_subject",
    }
    missing = sorted(required_columns - set(descriptors.columns))
    if missing:
        raise ValueError(
            "descriptors is missing full-admission-history columns. "
            f"Rerun sql-scripts/06_save_tables/02_Additional_info_export_on_cohort.sql. "
            f"Missing: {missing}"
        )
    clean = descriptors.copy()
    clean["n_prior_all_admissions_for_subject"] = pd.to_numeric(
        clean["n_prior_all_admissions_for_subject"],
        errors="raise",
    ).astype(int)
    rows = []
    for cohort, group in clean.groupby("cohort"):
        values = group["n_prior_all_admissions_for_subject"]
        rows.append(
            {
                "cohort": cohort,
                "n_matched_admissions": group["hadm_id"].nunique(),
                "n_subjects": group["subject_id"].nunique(),
                "mean_prior_all_mimic_admissions": values.mean(),
                "sd_prior_all_mimic_admissions": values.std(ddof=1),
                "median_prior_all_mimic_admissions": values.median(),
                "q1_prior_all_mimic_admissions": values.quantile(0.25),
                "q3_prior_all_mimic_admissions": values.quantile(0.75),
                "max_prior_all_mimic_admissions": values.max(),
                "n_matched_admissions_with_no_prior_mimic_admission": int(values.eq(0).sum()),
                "n_matched_admissions_with_any_prior_mimic_admission": int(values.gt(0).sum()),
                "pct_matched_admissions_with_any_prior_mimic_admission": 100.0
                * values.gt(0).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("cohort")


def build_prior_all_mimic_admission_distribution(
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    """Count matched admissions by exact number of prior MIMIC admissions."""
    required_columns = {
        "cohort",
        "subject_id",
        "hadm_id",
        "n_prior_all_admissions_for_subject",
    }
    missing = sorted(required_columns - set(descriptors.columns))
    if missing:
        raise ValueError(
            "descriptors is missing full-admission-history columns. "
            f"Rerun sql-scripts/06_save_tables/02_Additional_info_export_on_cohort.sql. "
            f"Missing: {missing}"
        )
    clean = descriptors.copy()
    clean["n_prior_all_admissions_for_subject"] = pd.to_numeric(
        clean["n_prior_all_admissions_for_subject"],
        errors="raise",
    ).astype(int)
    distribution = (
        clean.groupby(["cohort", "n_prior_all_admissions_for_subject"], as_index=False)
        .agg(
            n_matched_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
        )
        .sort_values(["cohort", "n_prior_all_admissions_for_subject"])
    )
    denominators = clean.groupby("cohort")["hadm_id"].nunique().to_dict()
    distribution["pct_matched_admissions_within_cohort"] = distribution.apply(
        lambda row: 100.0
        * row["n_matched_admissions"]
        / denominators.get(row["cohort"], 0),
        axis=1,
    )
    return distribution


def build_prior_all_mimic_admission_bucket_distribution(
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    """Count matched admissions by bucketed number of prior MIMIC admissions."""
    exact = build_prior_all_mimic_admission_distribution(descriptors)
    exact["prior_all_mimic_admission_bucket"] = exact[
        "n_prior_all_admissions_for_subject"
    ].map(prior_all_admission_bucket)
    bucket_order = ["0", "1", "2", "3", "4", "5", "6-10", "11+"]
    bucketed = (
        exact.groupby(["cohort", "prior_all_mimic_admission_bucket"], as_index=False)
        .agg(
            n_matched_admissions=("n_matched_admissions", "sum"),
            n_subjects=("n_subjects", "sum"),
        )
    )
    denominators = (
        exact.groupby("cohort")["n_matched_admissions"].sum().to_dict()
    )
    bucketed["pct_matched_admissions_within_cohort"] = bucketed.apply(
        lambda row: 100.0
        * row["n_matched_admissions"]
        / denominators.get(row["cohort"], 0),
        axis=1,
    )
    bucketed["prior_all_mimic_admission_bucket"] = pd.Categorical(
        bucketed["prior_all_mimic_admission_bucket"],
        categories=bucket_order,
        ordered=True,
    )
    return bucketed.sort_values(["cohort", "prior_all_mimic_admission_bucket"])


def build_prior_all_mimic_window_summary(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Summarize recent prior admissions within 30/90/365 days."""
    window_columns = [
        "n_prior_admissions_within_30d_for_subject",
        "n_prior_admissions_within_90d_for_subject",
        "n_prior_admissions_within_365d_for_subject",
        "has_prior_admission_within_30d_for_subject",
        "has_prior_admission_within_90d_for_subject",
        "has_prior_admission_within_365d_for_subject",
    ]
    missing = sorted(set(["cohort", "subject_id", "hadm_id", *window_columns]) - set(descriptors.columns))
    if missing:
        raise ValueError(
            "descriptors is missing recent full-admission-history columns. "
            f"Rerun sql-scripts/06_save_tables/02_Additional_info_export_on_cohort.sql. "
            f"Missing: {missing}"
        )
    clean = descriptors.copy()
    for column in window_columns:
        clean[column] = pd.to_numeric(clean[column], errors="raise")

    rows = []
    for cohort, group in clean.groupby("cohort"):
        for days in (30, 90, 365):
            count_column = f"n_prior_admissions_within_{days}d_for_subject"
            flag_column = f"has_prior_admission_within_{days}d_for_subject"
            rows.append(
                {
                    "cohort": cohort,
                    "window_days": days,
                    "n_matched_admissions": group["hadm_id"].nunique(),
                    "n_subjects": group["subject_id"].nunique(),
                    "n_matched_admissions_with_prior_admission_in_window": int(
                        group[flag_column].sum()
                    ),
                    "pct_matched_admissions_with_prior_admission_in_window": 100.0
                    * group[flag_column].mean(),
                    "mean_prior_admissions_in_window": group[count_column].mean(),
                    "median_prior_admissions_in_window": group[count_column].median(),
                    "max_prior_admissions_in_window": group[count_column].max(),
                }
            )
    return pd.DataFrame(rows).sort_values(["cohort", "window_days"])


def readmission_interval_bucket(days: float | int | pd.NA) -> str:
    """Bucket time from previous discharge to current matched admission."""
    if pd.isna(days):
        return "no_prior_admission"
    if days < 0:
        return "overlap_or_negative"
    if days <= 7:
        return "0-7d"
    if days <= 30:
        return "8-30d"
    if days <= 90:
        return "31-90d"
    if days <= 365:
        return "91-365d"
    return ">365d"


def build_prior_all_mimic_interval_summary(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Summarize time since the previous hospital admission/discharge."""
    required_columns = {
        "cohort",
        "subject_id",
        "hadm_id",
        "admittime",
        "previous_admittime_for_subject",
        "previous_dischtime_for_subject",
        "days_since_previous_discharge_for_subject",
        "n_prior_admissions_within_30d_for_subject",
        "n_prior_admissions_within_90d_for_subject",
        "n_prior_admissions_within_365d_for_subject",
    }
    missing = sorted(required_columns - set(descriptors.columns))
    if missing:
        raise ValueError(
            "descriptors is missing full-admission interval columns. "
            f"Rerun sql-scripts/06_save_tables/02_Additional_info_export_on_cohort.sql. "
            f"Missing: {missing}"
        )

    clean = descriptors.copy()
    clean["admittime"] = pd.to_datetime(clean["admittime"], errors="coerce")
    clean["previous_admittime_for_subject"] = pd.to_datetime(
        clean["previous_admittime_for_subject"],
        errors="coerce",
    )
    clean["days_since_previous_admission_for_subject"] = (
        clean["admittime"] - clean["previous_admittime_for_subject"]
    ).dt.total_seconds() / 86400.0
    numeric_columns = [
        "days_since_previous_discharge_for_subject",
        "n_prior_admissions_within_30d_for_subject",
        "n_prior_admissions_within_90d_for_subject",
        "n_prior_admissions_within_365d_for_subject",
    ]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    rows = []
    for cohort, group in clean.groupby("cohort"):
        has_prior = group["previous_admittime_for_subject"].notna()
        prior_group = group.loc[has_prior]
        rows.append(
            {
                "cohort": cohort,
                "n_matched_admissions": group["hadm_id"].nunique(),
                "n_subjects": group["subject_id"].nunique(),
                "n_matched_admissions_with_previous_mimic_admission": int(has_prior.sum()),
                "pct_matched_admissions_with_previous_mimic_admission": 100.0
                * has_prior.mean(),
                "mean_days_since_previous_admission": prior_group[
                    "days_since_previous_admission_for_subject"
                ].mean(),
                "median_days_since_previous_admission": prior_group[
                    "days_since_previous_admission_for_subject"
                ].median(),
                "q1_days_since_previous_admission": prior_group[
                    "days_since_previous_admission_for_subject"
                ].quantile(0.25),
                "q3_days_since_previous_admission": prior_group[
                    "days_since_previous_admission_for_subject"
                ].quantile(0.75),
                "mean_days_since_previous_discharge": prior_group[
                    "days_since_previous_discharge_for_subject"
                ].mean(),
                "median_days_since_previous_discharge": prior_group[
                    "days_since_previous_discharge_for_subject"
                ].median(),
                "q1_days_since_previous_discharge": prior_group[
                    "days_since_previous_discharge_for_subject"
                ].quantile(0.25),
                "q3_days_since_previous_discharge": prior_group[
                    "days_since_previous_discharge_for_subject"
                ].quantile(0.75),
                "mean_prior_admissions_within_30d": group[
                    "n_prior_admissions_within_30d_for_subject"
                ].mean(),
                "mean_prior_admissions_within_90d": group[
                    "n_prior_admissions_within_90d_for_subject"
                ].mean(),
                "mean_prior_admissions_within_365d": group[
                    "n_prior_admissions_within_365d_for_subject"
                ].mean(),
                "pct_matched_admissions_with_2plus_prior_admissions_within_365d": 100.0
                * group["n_prior_admissions_within_365d_for_subject"].ge(2).mean(),
                "pct_matched_admissions_with_3plus_prior_admissions_within_365d": 100.0
                * group["n_prior_admissions_within_365d_for_subject"].ge(3).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("cohort")


def build_prior_all_mimic_interval_bucket_distribution(
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    """Count matched admissions by time since previous discharge bucket."""
    required_columns = {
        "cohort",
        "subject_id",
        "hadm_id",
        "days_since_previous_discharge_for_subject",
    }
    missing = sorted(required_columns - set(descriptors.columns))
    if missing:
        raise ValueError(
            "descriptors is missing prior discharge interval columns. "
            f"Rerun sql-scripts/06_save_tables/02_Additional_info_export_on_cohort.sql. "
            f"Missing: {missing}"
        )

    clean = descriptors.copy()
    clean["days_since_previous_discharge_for_subject"] = pd.to_numeric(
        clean["days_since_previous_discharge_for_subject"],
        errors="coerce",
    )
    clean["previous_discharge_interval_bucket"] = clean[
        "days_since_previous_discharge_for_subject"
    ].map(readmission_interval_bucket)
    bucket_order = [
        "no_prior_admission",
        "overlap_or_negative",
        "0-7d",
        "8-30d",
        "31-90d",
        "91-365d",
        ">365d",
    ]
    distribution = (
        clean.groupby(["cohort", "previous_discharge_interval_bucket"], as_index=False)
        .agg(
            n_matched_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
        )
    )
    denominators = clean.groupby("cohort")["hadm_id"].nunique().to_dict()
    distribution["pct_matched_admissions_within_cohort"] = distribution.apply(
        lambda row: 100.0
        * row["n_matched_admissions"]
        / denominators.get(row["cohort"], 0),
        axis=1,
    )
    distribution["previous_discharge_interval_bucket"] = pd.Categorical(
        distribution["previous_discharge_interval_bucket"],
        categories=bucket_order,
        ordered=True,
    )
    return distribution.sort_values(["cohort", "previous_discharge_interval_bucket"])


def build_prior_all_mimic_admission_rate_summary(
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize prior admission frequency per observed patient-year.

    Rates are missing for first observed admissions because they have no prior
    observation interval. This avoids treating zero prior admissions as a true
    zero-rate estimate.
    """
    required_columns = {
        "cohort",
        "subject_id",
        "hadm_id",
        "n_prior_all_admissions_for_subject",
        "n_prior_emergency_or_urgent_admissions_for_subject",
    }
    missing = sorted(required_columns - set(descriptors.columns))
    if missing:
        raise ValueError(
            "descriptors is missing rate-denominator columns. "
            f"Rerun sql-scripts/06_save_tables/02_Additional_info_export_on_cohort.sql. "
            f"Missing: {missing}"
        )

    clean = add_prior_all_mimic_admission_rates(descriptors)
    rows = []
    for cohort, group in clean.groupby("cohort"):
        valid_rate = group["prior_admission_rate_per_observed_year"].notna()
        valid_group = group.loc[valid_rate]
        rows.append(
            {
                "cohort": cohort,
                "n_matched_admissions": group["hadm_id"].nunique(),
                "n_subjects": group["subject_id"].nunique(),
                "n_rate_observable_matched_admissions": int(valid_rate.sum()),
                "n_rate_missing_first_observed_admissions": int((~valid_rate).sum()),
                "pct_rate_missing_first_observed_admissions": 100.0 * (~valid_rate).mean(),
                "mean_prior_admission_rate_per_observed_year": valid_group[
                    "prior_admission_rate_per_observed_year"
                ].mean(),
                "median_prior_admission_rate_per_observed_year": valid_group[
                    "prior_admission_rate_per_observed_year"
                ].median(),
                "q1_prior_admission_rate_per_observed_year": valid_group[
                    "prior_admission_rate_per_observed_year"
                ].quantile(0.25),
                "q3_prior_admission_rate_per_observed_year": valid_group[
                    "prior_admission_rate_per_observed_year"
                ].quantile(0.75),
                "mean_prior_emergency_or_urgent_rate_per_observed_year": valid_group[
                    "prior_emergency_or_urgent_admission_rate_per_observed_year"
                ].mean(),
                "median_prior_emergency_or_urgent_rate_per_observed_year": valid_group[
                    "prior_emergency_or_urgent_admission_rate_per_observed_year"
                ].median(),
            }
        )
    return pd.DataFrame(rows).sort_values("cohort")


def build_prior_all_mimic_admission_rate_distribution(
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    """Count matched admissions by prior admission-rate buckets."""
    clean = add_prior_all_mimic_admission_rates(descriptors)
    bucket_order = [
        "missing_first_observed",
        "0-0.5/year",
        "0.5-1/year",
        "1-2/year",
        "2-4/year",
        "4+/year",
    ]
    clean["prior_admission_rate_bucket"] = clean[
        "prior_admission_rate_per_observed_year"
    ].map(prior_admission_rate_bucket)
    distribution = (
        clean.groupby(["cohort", "prior_admission_rate_bucket"], as_index=False)
        .agg(
            n_matched_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
        )
    )
    denominators = clean.groupby("cohort")["hadm_id"].nunique().to_dict()
    distribution["pct_matched_admissions_within_cohort"] = distribution.apply(
        lambda row: 100.0
        * row["n_matched_admissions"]
        / denominators.get(row["cohort"], 0),
        axis=1,
    )
    distribution["prior_admission_rate_bucket"] = pd.Categorical(
        distribution["prior_admission_rate_bucket"],
        categories=bucket_order,
        ordered=True,
    )
    return distribution.sort_values(["cohort", "prior_admission_rate_bucket"])


def add_prior_all_mimic_admission_rates(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Add prior admission rates per observed patient-year to descriptor rows."""
    required_columns = {
        "subject_id",
        "admittime",
        "n_prior_all_admissions_for_subject",
        "n_prior_emergency_or_urgent_admissions_for_subject",
    }
    missing = sorted(required_columns - set(descriptors.columns))
    if missing:
        raise ValueError(
            "descriptors is missing prior admission columns. "
            f"Rerun sql-scripts/06_save_tables/02_Additional_info_export_on_cohort.sql. "
            f"Missing: {missing}"
        )
    clean = descriptors.copy()
    if "days_since_first_observed_admission_for_subject" not in clean.columns:
        history = load_optional_table("subject_admission_history")
        if history is None:
            history = load_duckdb_table("admissions")
        if not {"subject_id", "admittime"}.issubset(history.columns):
            raise ValueError(
                "Could not derive first observed admission time because the "
                "admission history table is missing subject_id/admittime."
            )
        history = history.loc[:, ["subject_id", "admittime"]].copy()
        history["subject_id"] = pd.to_numeric(
            history["subject_id"],
            errors="raise",
        ).astype(int)
        history["admittime"] = pd.to_datetime(history["admittime"], errors="coerce")
        first_admissions = (
            history.dropna(subset=["admittime"])
            .groupby("subject_id", as_index=False)["admittime"]
            .min()
            .rename(columns={"admittime": "first_observed_admittime_for_subject"})
        )
        clean["subject_id"] = pd.to_numeric(
            clean["subject_id"],
            errors="raise",
        ).astype(int)
        clean["admittime"] = pd.to_datetime(clean["admittime"], errors="coerce")
        clean = clean.merge(first_admissions, on="subject_id", how="left")
        clean["days_since_first_observed_admission_for_subject"] = (
            clean["admittime"] - clean["first_observed_admittime_for_subject"]
        ).dt.total_seconds() / 86400.0
    clean["n_prior_all_admissions_for_subject"] = pd.to_numeric(
        clean["n_prior_all_admissions_for_subject"],
        errors="raise",
    )
    clean["n_prior_emergency_or_urgent_admissions_for_subject"] = pd.to_numeric(
        clean["n_prior_emergency_or_urgent_admissions_for_subject"],
        errors="coerce",
    )
    clean["days_since_first_observed_admission_for_subject"] = pd.to_numeric(
        clean["days_since_first_observed_admission_for_subject"],
        errors="coerce",
    )
    clean["observed_years_before_matched_admission"] = (
        clean["days_since_first_observed_admission_for_subject"] / 365.25
    )
    valid_denominator = clean["observed_years_before_matched_admission"].gt(0)
    clean["prior_admission_rate_per_observed_year"] = pd.NA
    clean.loc[valid_denominator, "prior_admission_rate_per_observed_year"] = (
        clean.loc[valid_denominator, "n_prior_all_admissions_for_subject"]
        / clean.loc[valid_denominator, "observed_years_before_matched_admission"]
    )
    clean["prior_emergency_or_urgent_admission_rate_per_observed_year"] = pd.NA
    clean.loc[
        valid_denominator,
        "prior_emergency_or_urgent_admission_rate_per_observed_year",
    ] = (
        clean.loc[
            valid_denominator,
            "n_prior_emergency_or_urgent_admissions_for_subject",
        ].fillna(0)
        / clean.loc[valid_denominator, "observed_years_before_matched_admission"]
    )
    return clean


def prior_admission_rate_bucket(rate: object) -> str:
    """Return a compact bucket for prior admission rate per observed year."""
    if pd.isna(rate):
        return "missing_first_observed"
    rate = float(rate)
    if rate <= 0.5:
        return "0-0.5/year"
    if rate <= 1:
        return "0.5-1/year"
    if rate <= 2:
        return "1-2/year"
    if rate <= 4:
        return "2-4/year"
    return "4+/year"


def build_readmission_cap_loss_summary(
    matched_admissions: pd.DataFrame,
    caps: tuple[int, ...] = (3, 4, 5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate admission/pair loss under readmission caps.

    The independent mode caps admissions within each cohort/subject. The
    pair-preserving mode then drops a full matched pair if either admission in
    the pair falls above the cap.
    """
    required_columns = {"pair_id", "cohort", "subject_id", "hadm_id"}
    missing = sorted(required_columns - set(matched_admissions.columns))
    if missing:
        raise ValueError(f"matched_admissions is missing columns: {missing}")

    clean = matched_admissions.copy()
    clean["cohort"] = clean["cohort"].astype("string").str.strip()
    clean["pair_id"] = pd.to_numeric(clean["pair_id"], errors="raise").astype(int)
    clean["subject_id"] = pd.to_numeric(clean["subject_id"], errors="raise").astype(int)
    clean["hadm_id"] = pd.to_numeric(clean["hadm_id"], errors="raise").astype(int)
    if "admittime" in clean.columns:
        clean["admittime"] = pd.to_datetime(clean["admittime"], errors="coerce")
        sort_columns = ["cohort", "subject_id", "admittime", "pair_id", "hadm_id"]
    else:
        sort_columns = ["cohort", "subject_id", "pair_id", "hadm_id"]
    clean = clean.sort_values(sort_columns).copy()
    clean["admission_rank_within_subject_cohort"] = (
        clean.groupby(["cohort", "subject_id"]).cumcount() + 1
    )

    admission_rows = []
    pair_rows = []
    original_pairs = clean["pair_id"].nunique()
    for cap in caps:
        keep_column = f"kept_by_independent_cap_{cap}"
        clean[keep_column] = clean["admission_rank_within_subject_cohort"].le(cap)

        for cohort, group in clean.groupby("cohort"):
            kept = int(group[keep_column].sum())
            lost = len(group) - kept
            subject_counts = group.groupby("subject_id").size()
            admission_rows.append(
                {
                    "cap": cap,
                    "mode": "independent_within_cohort",
                    "cohort": cohort,
                    "original_admissions": len(group),
                    "kept_admissions": kept,
                    "lost_admissions": lost,
                    "pct_admissions_lost": 100.0 * lost / len(group),
                    "original_subjects": group["subject_id"].nunique(),
                    "subjects_with_more_than_cap": int(subject_counts.gt(cap).sum()),
                }
            )

        pair_keep = clean.groupby("pair_id")[keep_column].all()
        kept_pair_ids = set(pair_keep.loc[pair_keep].index)
        lost_pairs = original_pairs - len(kept_pair_ids)
        pair_rows.append(
            {
                "cap": cap,
                "original_pairs": original_pairs,
                "kept_pairs": len(kept_pair_ids),
                "lost_pairs": lost_pairs,
                "pct_pairs_lost": 100.0 * lost_pairs / original_pairs,
            }
        )

        pair_preserving = clean.assign(
            kept_pair_preserving=clean["pair_id"].isin(kept_pair_ids)
        )
        for cohort, group in pair_preserving.groupby("cohort"):
            kept = group.loc[group["kept_pair_preserving"]]
            lost = len(group) - len(kept)
            subject_counts = group.groupby("subject_id").size()
            admission_rows.append(
                {
                    "cap": cap,
                    "mode": "pair_preserving",
                    "cohort": cohort,
                    "original_admissions": len(group),
                    "kept_admissions": len(kept),
                    "lost_admissions": lost,
                    "pct_admissions_lost": 100.0 * lost / len(group),
                    "original_subjects": group["subject_id"].nunique(),
                    "subjects_with_more_than_cap": int(subject_counts.gt(cap).sum()),
                }
            )

    admission_loss = pd.DataFrame(admission_rows).sort_values(["cap", "mode", "cohort"])
    pair_loss = pd.DataFrame(pair_rows).sort_values("cap")
    return admission_loss, pair_loss


def choose_subject_category(values: pd.Series) -> str:
    """Collapse admission-level categories to one subject-level category."""
    normalized = values.fillna("missing").astype(str).str.strip()
    normalized.loc[normalized.eq("")] = "missing"
    unique_values = sorted(set(normalized))
    if len(unique_values) == 1:
        return unique_values[0]
    nonmissing_values = [value for value in unique_values if value != "missing"]
    if len(nonmissing_values) == 1:
        return nonmissing_values[0]
    if len(nonmissing_values) == 0:
        return "missing"
    return "multiple_values"


def build_subject_categorical_distribution(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Build subject-level n/% tables for categorical descriptors."""
    available_columns = [
        column for column in CATEGORICAL_DESCRIPTOR_COLUMNS if column in descriptors.columns
    ]
    if not available_columns:
        return pd.DataFrame(
            columns=["variable", "cohort", "category", "n_subjects", "pct_within_cohort"]
        )

    rows = []
    for variable in available_columns:
        collapsed = (
            descriptors.groupby(["cohort", "subject_id"])[variable]
            .agg(choose_subject_category)
            .reset_index(name="category")
        )
        denominators = collapsed.groupby("cohort")["subject_id"].nunique().to_dict()
        counts = (
            collapsed.groupby(["cohort", "category"], as_index=False)["subject_id"]
            .nunique()
            .rename(columns={"subject_id": "n_subjects"})
        )
        counts["variable"] = variable
        counts["pct_within_cohort"] = counts.apply(
            lambda row: 100.0
            * row["n_subjects"]
            / denominators.get(row["cohort"], 0),
            axis=1,
        )
        rows.append(counts)
    return pd.concat(rows, ignore_index=True).loc[
        :, ["variable", "cohort", "category", "n_subjects", "pct_within_cohort"]
    ].sort_values(["variable", "cohort", "n_subjects"], ascending=[True, True, False])


def build_subject_categorical_balance(
    subject_categorical_distribution: pd.DataFrame,
) -> pd.DataFrame:
    """Pivot subject-level categorical percentages by cohort."""
    if subject_categorical_distribution.empty:
        return subject_categorical_distribution.copy()
    pivot = subject_categorical_distribution.pivot_table(
        index=["variable", "category"],
        columns="cohort",
        values=["n_subjects", "pct_within_cohort"],
        fill_value=0,
        aggfunc="sum",
    )
    pivot.columns = [
        f"{metric}_{cohort}".lower()
        for metric, cohort in pivot.columns.to_flat_index()
    ]
    pivot = pivot.reset_index()
    mhh_pct = "pct_within_cohort_mhh1_psychotic"
    mhc0_pct = "pct_within_cohort_mhc0"
    if mhh_pct in pivot.columns and mhc0_pct in pivot.columns:
        pivot["pct_point_difference_mhh1_minus_mhc0"] = pivot[mhh_pct] - pivot[mhc0_pct]
    return pivot.sort_values(["variable", "category"])


def build_subject_utilization_counts(
    utilization_counts_by_admission: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse admission-level utilization counts to subject-level measures."""
    count_columns = [
        column
        for column in utilization_counts_by_admission.columns
        if column.startswith("n_") and column.endswith("_rows")
    ]
    grouped = utilization_counts_by_admission.groupby(
        ["cohort", "subject_id"],
        as_index=False,
    )
    subject_counts = grouped.agg(n_matched_admissions=("hadm_id", "nunique"))
    for column in count_columns:
        totals = grouped[column].sum().rename(columns={column: f"total_{column}"})
        means = grouped[column].mean().rename(columns={column: f"mean_{column}_per_admission"})
        subject_counts = subject_counts.merge(totals, on=["cohort", "subject_id"])
        subject_counts = subject_counts.merge(means, on=["cohort", "subject_id"])
    return subject_counts.sort_values(["cohort", "subject_id"]).reset_index(drop=True)


def build_subject_utilization_summary(
    subject_utilization_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize subject-level utilization measures by cohort."""
    measure_columns = [
        column
        for column in subject_utilization_counts.columns
        if column not in {"cohort", "subject_id"}
    ]
    rows = []
    for cohort, group in subject_utilization_counts.groupby("cohort"):
        for column in measure_columns:
            values = group[column]
            rows.append(
                {
                    "cohort": cohort,
                    "measure": column,
                    "n_subjects": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "iqr": values.quantile(0.75) - values.quantile(0.25),
                    "min": values.min(),
                    "max": values.max(),
                    "n_with_any": int(values.gt(0).sum()),
                    "pct_with_any": 100.0 * values.gt(0).mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["measure", "cohort"])


def write_outputs(
    descriptor_completeness: pd.DataFrame,
    categorical_distribution: pd.DataFrame,
    categorical_balance: pd.DataFrame,
    utilization_counts: pd.DataFrame,
    utilization_summary: pd.DataFrame,
    optional_category_distribution: pd.DataFrame,
    admissions_per_subject_summary: pd.DataFrame,
    subject_categorical_distribution: pd.DataFrame,
    subject_categorical_balance: pd.DataFrame,
    subject_utilization_counts: pd.DataFrame,
    subject_utilization_summary: pd.DataFrame,
) -> None:
    """Write characterization aggregate outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    descriptor_completeness.to_csv(
        OUTPUT_DIR / "matched_cohort_descriptor_completeness.csv",
        index=False,
    )
    categorical_distribution.to_csv(
        OUTPUT_DIR / "matched_cohort_categorical_distribution.csv",
        index=False,
    )
    categorical_balance.to_csv(
        OUTPUT_DIR / "matched_cohort_categorical_balance.csv",
        index=False,
    )
    utilization_counts.to_csv(
        OUTPUT_DIR / "matched_cohort_utilization_counts_by_admission.csv",
        index=False,
    )
    utilization_summary.to_csv(
        OUTPUT_DIR / "matched_cohort_utilization_summary.csv",
        index=False,
    )
    optional_category_distribution.to_csv(
        OUTPUT_DIR / "matched_cohort_optional_category_distribution.csv",
        index=False,
    )
    admissions_per_subject_summary.to_csv(
        OUTPUT_DIR / "matched_cohort_admissions_per_subject_summary.csv",
        index=False,
    )
    subject_categorical_distribution.to_csv(
        OUTPUT_DIR / "matched_cohort_subject_categorical_distribution.csv",
        index=False,
    )
    subject_categorical_balance.to_csv(
        OUTPUT_DIR / "matched_cohort_subject_categorical_balance.csv",
        index=False,
    )
    subject_utilization_counts.to_csv(
        OUTPUT_DIR / "matched_cohort_subject_utilization_counts.csv",
        index=False,
    )
    subject_utilization_summary.to_csv(
        OUTPUT_DIR / "matched_cohort_subject_utilization_summary.csv",
        index=False,
    )


def main() -> None:
    """Run matched-cohort characterization from DBeaver exports."""
    matched_ids = load_expected_matched_ids()
    descriptors = add_derived_descriptor_columns(
        validate_id_columns(load_required_table("descriptors"), "descriptors")
    )
    event_tables = {
        "labevents": load_optional_table("labevents"),
        "microbiologyevents": load_optional_table("microbiologyevents"),
        "poe": load_optional_table("poe"),
        "poe_detail": load_optional_table("poe_detail"),
    }

    descriptor_completeness = build_descriptor_completeness(matched_ids, descriptors)
    categorical_distribution = build_categorical_distribution(descriptors)
    categorical_balance = build_categorical_balance(categorical_distribution)
    utilization_counts = build_event_counts_by_admission(matched_ids, event_tables)
    utilization_summary = build_utilization_summary(utilization_counts)
    admissions_per_subject_summary = build_admissions_per_subject_summary(matched_ids)
    subject_categorical_distribution = build_subject_categorical_distribution(
        descriptors
    )
    subject_categorical_balance = build_subject_categorical_balance(
        subject_categorical_distribution
    )
    subject_utilization_counts = build_subject_utilization_counts(utilization_counts)
    subject_utilization_summary = build_subject_utilization_summary(
        subject_utilization_counts
    )
    optional_category_distribution = pd.concat(
        [
            build_optional_category_distribution(
                event_tables["poe"],
                "poe",
                ["order_type", "order_subtype", "transaction_type"],
            ),
            build_optional_category_distribution(
                event_tables["poe_detail"],
                "poe_detail",
                ["field_name", "field_value"],
            ),
            build_optional_category_distribution(
                event_tables["microbiologyevents"],
                "microbiologyevents",
                ["spec_type_desc", "test_name", "org_name"],
            ),
        ],
        ignore_index=True,
    )

    write_outputs(
        descriptor_completeness,
        categorical_distribution,
        categorical_balance,
        utilization_counts,
        utilization_summary,
        optional_category_distribution,
        admissions_per_subject_summary,
        subject_categorical_distribution,
        subject_categorical_balance,
        subject_utilization_counts,
        subject_utilization_summary,
    )

    print(f"Read DBeaver exports from: {INPUT_DIR}")
    print(f"Saved matched-cohort characterization outputs to: {OUTPUT_DIR}")
    print("\n=== Descriptor Completeness ===")
    print(descriptor_completeness.to_string(index=False))
    print("\n=== Utilization Summary ===")
    print(utilization_summary.to_string(index=False))
    print("\n=== Admissions Per Subject Summary ===")
    print(admissions_per_subject_summary.to_string(index=False))
    print("\n=== Subject-Level Utilization Summary ===")
    print(subject_utilization_summary.to_string(index=False))


if __name__ == "__main__":
    main()
