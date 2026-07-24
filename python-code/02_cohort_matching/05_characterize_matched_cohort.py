"""Characterize matched cohorts with exported admission descriptors/utilization.

This script analyzes extra matched-cohort descriptors exported from DBeaver. It
expects one small descriptor table and optional event/order tables already
restricted to the matched MHH1_psychotic and MHC0 admissions.

Default input folder:
    matched_cohort_dbeaver_exports/

Expected file basenames, with .csv or .parquet extension:
    export_matched_cohort_descriptors
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
OUTPUT_DIR = Path(
    os.environ.get(
        "MATCHED_COHORT_CHARACTERIZATION_OUTPUT_DIR",
        str(SCRIPT_DIR / "analysis_output_matched_cohort_characterization"),
    )
)

EXPORT_BASENAMES = {
    "descriptors": "export_matched_cohort_descriptors",
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


def write_outputs(
    descriptor_completeness: pd.DataFrame,
    categorical_distribution: pd.DataFrame,
    categorical_balance: pd.DataFrame,
    utilization_counts: pd.DataFrame,
    utilization_summary: pd.DataFrame,
    optional_category_distribution: pd.DataFrame,
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
    )

    print(f"Read DBeaver exports from: {INPUT_DIR}")
    print(f"Saved matched-cohort characterization outputs to: {OUTPUT_DIR}")
    print("\n=== Descriptor Completeness ===")
    print(descriptor_completeness.to_string(index=False))
    print("\n=== Utilization Summary ===")
    print(utilization_summary.to_string(index=False))


if __name__ == "__main__":
    main()
