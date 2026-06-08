from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import pandas as pd
import pandas.core.common as pandas_common
from pandas.errors import SettingWithCopyWarning


# comorbidipy 0.5.0 imports SettingWithCopyWarning from an older pandas
# internal path. This keeps the import working with newer pandas releases.
pandas_common.SettingWithCopyWarning = SettingWithCopyWarning

import comorbidipy as com  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
THESIS_DIR = PROJECT_DIR.parent.parent
DB_PATH = THESIS_DIR / "DataBase"
OUTPUT_DIR = SCRIPT_DIR / "elixhauser_scores_output"

REQUIRED_COLUMNS = {"subject_id", "hadm_id", "icd_version", "icd_code"}
SCORE = "elixhauser"
VARIANT = "quan"
WEIGHTING = "vanwalraven"
COMORBIDIPY_WEIGHTING = "vw"

INPUTS = {
    "MHH_psychotic": {
        "source_table": "admission_icd_lists_MHH_psychotic",
        "output_basename": "elixhauser_MHH_psychotic",
    },
    "only_MHC0": {
        "source_table": "admission_icd_lists_only_MHC0",
        "output_basename": "elixhauser_only_MHC0",
    },
}


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def validate_columns(df: pd.DataFrame, path: str) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")


def clean_icd_codes(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["icd_code"] = (
        clean["icd_code"]
        .astype("string")
        .str.strip()
        .str.replace(".", "", regex=False)
        .str.upper()
    )
    clean = clean.dropna(subset=["subject_id", "hadm_id", "icd_version", "icd_code"])
    clean = clean.loc[clean["icd_code"].str.len() > 0].copy()
    return clean


def add_admission_id(df: pd.DataFrame) -> pd.DataFrame:
    with_id = df.copy()
    with_id["admission_id"] = (
        with_id["subject_id"].astype(str) + "_" + with_id["hadm_id"].astype(str)
    )
    return with_id


def run_comorbidipy_for_version(df: pd.DataFrame, icd_label: str) -> pd.DataFrame:
    if df.empty:
        print(f"No rows for {icd_label}; skipping comorbidipy.", flush=True)
        return pd.DataFrame(columns=["admission_id"])

    comorbidity_input = df[["admission_id", "icd_code"]].drop_duplicates().copy()
    print(
        f"Running comorbidipy for {icd_label}: "
        f"{len(comorbidity_input):,} diagnosis rows, "
        f"{comorbidity_input['admission_id'].nunique():,} admissions",
        flush=True,
    )

    result = com.comorbidity(
        comorbidity_input,
        id="admission_id",
        code="icd_code",
        age=None,
        score=SCORE,
        icd=icd_label,
        variant=VARIANT,
        weighting=COMORBIDIPY_WEIGHTING,
    )
    print(f"{icd_label} comorbidipy columns: {list(result.columns)}", flush=True)
    return result


def standardize_score_column(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.copy()
    if standardized.empty:
        return standardized

    score_candidates = [
        column
        for column in standardized.columns
        if column == "comorbidity_score"
        or ("score" in column.lower() and column != "elixhauser_score")
    ]
    if score_candidates:
        score_column = score_candidates[0]
        if score_column != "elixhauser_score":
            standardized = standardized.rename(columns={score_column: "elixhauser_score"})
        return standardized

    id_columns = {"admission_id", "subject_id", "hadm_id"}
    numeric_columns = [
        column
        for column in standardized.columns
        if column not in id_columns and pd.api.types.is_numeric_dtype(standardized[column])
    ]
    if numeric_columns:
        warnings.warn(
            "No weighted comorbidity score column returned by comorbidipy. "
            "Using unweighted count of numeric Elixhauser flags.",
            RuntimeWarning,
            stacklevel=2,
        )
        standardized["elixhauser_score_unweighted_count"] = (
            standardized[numeric_columns].fillna(0).astype(int).sum(axis=1)
        )
        return standardized

    warnings.warn(
        "No score or numeric Elixhauser flag columns returned by comorbidipy. "
        "Assigning elixhauser_score = 0.",
        RuntimeWarning,
        stacklevel=2,
    )
    standardized["elixhauser_score"] = 0
    return standardized


def ensure_all_admissions_present(original_dx: pd.DataFrame, elix_df: pd.DataFrame) -> pd.DataFrame:
    admissions = (
        original_dx[["admission_id", "subject_id", "hadm_id"]]
        .drop_duplicates()
        .sort_values(["subject_id", "hadm_id"])
        .reset_index(drop=True)
    )

    if elix_df.empty:
        completed = admissions.copy()
        completed["elixhauser_score"] = 0
        return completed

    completed = admissions.merge(elix_df, on="admission_id", how="left")

    score_columns = [
        column
        for column in ("elixhauser_score", "elixhauser_score_unweighted_count")
        if column in completed.columns
    ]
    for column in score_columns:
        completed[column] = completed[column].fillna(0)

    metadata_columns = {"admission_id", "subject_id", "hadm_id"}
    indicator_columns = [
        column
        for column in completed.columns
        if column not in metadata_columns
        and column not in score_columns
        and pd.api.types.is_numeric_dtype(completed[column])
    ]
    for column in indicator_columns:
        completed[column] = completed[column].fillna(0).astype(int)

    if "elixhauser_score" not in completed.columns:
        completed["elixhauser_score"] = 0

    ordered_columns = ["subject_id", "hadm_id", "elixhauser_score"]
    if "elixhauser_score_unweighted_count" in completed.columns:
        ordered_columns.append("elixhauser_score_unweighted_count")
    ordered_columns.extend(
        column
        for column in completed.columns
        if column not in set(ordered_columns) | {"admission_id"}
    )
    return completed.loc[:, ordered_columns]


def combine_version_outputs(version_outputs: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty_outputs = [df for df in version_outputs if not df.empty]
    if not non_empty_outputs:
        return pd.DataFrame(columns=["admission_id"])

    combined = pd.concat(non_empty_outputs, ignore_index=True, sort=False).fillna(0)
    numeric_columns = [
        column
        for column in combined.columns
        if column != "admission_id" and pd.api.types.is_numeric_dtype(combined[column])
    ]
    aggregations = {
        column: ("sum" if "score" in column.lower() else "max")
        for column in numeric_columns
    }
    return combined.groupby("admission_id", as_index=False).agg(aggregations)


def calculate_for_file(input_path: str, output_basename: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        dx = con.execute(
            f"SELECT * FROM {quote_identifier(input_path)}"
        ).fetchdf()
    finally:
        con.close()

    validate_columns(dx, input_path)
    dx = add_admission_id(clean_icd_codes(dx))

    icd_version = dx["icd_version"].astype("string").str.strip()
    icd9 = dx.loc[icd_version == "9"].copy()
    icd10 = dx.loc[icd_version == "10"].copy()

    results = []
    for subset, icd_label in ((icd9, "icd9"), (icd10, "icd10")):
        result = run_comorbidipy_for_version(subset, icd_label)
        results.append(standardize_score_column(result))

    combined = combine_version_outputs(results)
    completed = ensure_all_admissions_present(dx, combined)

    parquet_path = OUTPUT_DIR / f"{output_basename}.parquet"
    csv_path = OUTPUT_DIR / f"{output_basename}.csv"
    completed.to_parquet(parquet_path, index=False)
    completed.to_csv(csv_path, index=False)

    print(f"Saved {len(completed):,} admission rows to: {parquet_path}", flush=True)
    print(f"Saved {len(completed):,} admission rows to: {csv_path}", flush=True)
    print(
        completed["elixhauser_score"]
        .describe()
        .rename(output_basename)
        .to_string(),
        flush=True,
    )
    return completed


def write_summary(results: dict[str, pd.DataFrame]) -> None:
    rows = []
    for cohort_name, df in results.items():
        score = df["elixhauser_score"]
        rows.append(
            {
                "cohort": cohort_name,
                "n_admissions": len(df),
                "n_subjects": df["subject_id"].nunique(),
                "mean_elixhauser_score": score.mean(),
                "sd_elixhauser_score": score.std(),
                "median_elixhauser_score": score.median(),
                "min_elixhauser_score": score.min(),
                "max_elixhauser_score": score.max(),
                "n_score_gt_0": int((score > 0).sum()),
                "pct_score_gt_0": 100.0 * (score > 0).mean(),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "elixhauser_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Elixhauser summary:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved summary to: {summary_path}", flush=True)


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"No DuckDB database found at: {DB_PATH}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Reading ICD tables from DuckDB database: {DB_PATH}", flush=True)
    print(
        f"Using comorbidipy score={SCORE!r}, variant={VARIANT!r}, "
        f"weighting={WEIGHTING!r} ({COMORBIDIPY_WEIGHTING!r} in comorbidipy)",
        flush=True,
    )

    results = {}
    for cohort_name, config in INPUTS.items():
        print(f"\nCalculating Elixhauser scores for {cohort_name}", flush=True)
        results[cohort_name] = calculate_for_file(
            config["source_table"],
            config["output_basename"],
        )

    write_summary(results)


if __name__ == "__main__":
    main()
