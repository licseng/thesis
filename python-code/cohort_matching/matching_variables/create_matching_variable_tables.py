from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
COHORT_MATCHING_DIR = SCRIPT_DIR.parent
PROJECT_DIR = COHORT_MATCHING_DIR.parent
THESIS_DIR = PROJECT_DIR.parent.parent
DB_PATH = THESIS_DIR / "DataBase"

CHIEF_COMPLAINT_EMBEDDING_DIR = (
    PROJECT_DIR
    / "admission_notes"
    / "chief_complaint"
    / "embedding"
    / "chief_complaint_embeddings"
)
ELIXHAUSER_OUTPUT_DIR = COHORT_MATCHING_DIR / "elixhauser" / "elixhauser_scores_output"
OUTPUT_DIR = SCRIPT_DIR / "matching_variable_tables_output"

COHORTS = {
    "MHH_psychotic": {
        "source_table": "export_MHH_psychotic",
        "embedding_group": "MHH1_psychotic",
        "elixhauser_basename": "elixhauser_MHH_psychotic",
    },
    "only_MHC0": {
        "source_table": "export_only_MHC0",
        "embedding_group": "MHC0",
        "elixhauser_basename": "elixhauser_only_MHC0",
    },
}


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def load_demographics(source_table: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        demographics = con.execute(
            f"""
            SELECT DISTINCT
                subject_id,
                hadm_id,
                sex,
                age_at_admission
            FROM {quote_identifier(source_table)}
            WHERE subject_id IS NOT NULL
              AND hadm_id IS NOT NULL
            """
        ).fetchdf()
    finally:
        con.close()

    duplicated = demographics.duplicated(["subject_id", "hadm_id"], keep=False)
    if duplicated.any():
        raise ValueError(
            f"{source_table} has inconsistent duplicate demographics for "
            f"{duplicated.sum()} admission rows."
        )

    demographics["sex"] = demographics["sex"].astype("string").str.strip().str.upper()
    demographics["age_at_admission"] = pd.to_numeric(
        demographics["age_at_admission"], errors="coerce"
    )
    return demographics


def load_embedding_metadata(embedding_group: str) -> pd.DataFrame:
    metadata_path = (
        CHIEF_COMPLAINT_EMBEDDING_DIR
        / f"{embedding_group}_chief_complaint_embedding_metadata.parquet"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing embedding metadata: {metadata_path}")

    metadata = pd.read_parquet(metadata_path)
    required_columns = {"embedding_row_id", "subject_id", "hadm_id"}
    missing_columns = sorted(required_columns - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"{metadata_path} is missing columns: {missing_columns}")

    keep_columns = [
        "embedding_row_id",
        "subject_id",
        "hadm_id",
        "chief_complaint_raw",
        "chief_complaint_normalized",
    ]
    keep_columns = [column for column in keep_columns if column in metadata.columns]
    metadata = metadata.loc[:, keep_columns].copy()
    metadata["embedding_group"] = embedding_group
    metadata["embedding_file"] = (
        CHIEF_COMPLAINT_EMBEDDING_DIR
        / f"{embedding_group}_chief_complaint_embeddings.npy"
    ).as_posix()
    return metadata


def load_elixhauser(elixhauser_basename: str) -> pd.DataFrame:
    elixhauser_path = ELIXHAUSER_OUTPUT_DIR / f"{elixhauser_basename}.parquet"
    if not elixhauser_path.exists():
        raise FileNotFoundError(f"Missing Elixhauser table: {elixhauser_path}")

    elixhauser = pd.read_parquet(elixhauser_path)
    required_columns = {"subject_id", "hadm_id", "elixhauser_score"}
    missing_columns = sorted(required_columns - set(elixhauser.columns))
    if missing_columns:
        raise ValueError(f"{elixhauser_path} is missing columns: {missing_columns}")

    return elixhauser.loc[:, ["subject_id", "hadm_id", "elixhauser_score"]].copy()


def create_matching_table(cohort_name: str, config: dict[str, str]) -> pd.DataFrame:
    embeddings = load_embedding_metadata(config["embedding_group"])
    demographics = load_demographics(config["source_table"])
    elixhauser = load_elixhauser(config["elixhauser_basename"])

    table = embeddings.merge(
        demographics,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )
    table = table.merge(
        elixhauser,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )

    table.insert(0, "cohort", cohort_name)
    table["elixhauser_score"] = table["elixhauser_score"].fillna(0)

    missing_demographics = table["sex"].isna() | table["age_at_admission"].isna()
    if missing_demographics.any():
        print(
            f"Warning: {cohort_name} has {missing_demographics.sum():,} embedded "
            "admissions without complete sex/age metadata.",
            flush=True,
        )

    output_columns = [
        "cohort",
        "subject_id",
        "hadm_id",
        "embedding_group",
        "embedding_row_id",
        "embedding_file",
        "age_at_admission",
        "sex",
        "elixhauser_score",
    ]
    output_columns.extend(
        column
        for column in ["chief_complaint_raw", "chief_complaint_normalized"]
        if column in table.columns
    )
    return table.loc[:, output_columns].sort_values(["subject_id", "hadm_id"])


def write_outputs(cohort_tables: dict[str, pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    combined = pd.concat(cohort_tables.values(), ignore_index=True)
    for cohort_name, table in cohort_tables.items():
        table.to_parquet(OUTPUT_DIR / f"{cohort_name}_matching_variables.parquet", index=False)
        table.to_csv(OUTPUT_DIR / f"{cohort_name}_matching_variables.csv", index=False)
        print(f"{cohort_name}: saved {len(table):,} matching-variable rows", flush=True)

    combined.to_parquet(OUTPUT_DIR / "combined_matching_variables.parquet", index=False)
    combined.to_csv(OUTPUT_DIR / "combined_matching_variables.csv", index=False)

    summary = (
        combined.groupby("cohort")
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
            n_with_embedding=("embedding_row_id", "count"),
            mean_age=("age_at_admission", "mean"),
            mean_elixhauser_score=("elixhauser_score", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "matching_variables_summary.csv", index=False)
    print("Matching-variable summary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved matching-variable tables to: {OUTPUT_DIR}", flush=True)


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"No DuckDB database found at: {DB_PATH}")

    cohort_tables = {}
    for cohort_name, config in COHORTS.items():
        print(f"Creating matching-variable table for {cohort_name}", flush=True)
        cohort_tables[cohort_name] = create_matching_table(cohort_name, config)

    write_outputs(cohort_tables)


if __name__ == "__main__":
    main()
