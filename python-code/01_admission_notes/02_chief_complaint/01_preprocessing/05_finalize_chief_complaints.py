"""Finalize chief complaints for downstream embedding/matching.

This script reads the preprocessed chief-complaint parquet files and keeps only
admissions that have:
    - a usable normalized chief complaint,
    - no psychiatric/substance/self-harm TargetMatcher flag, and
    - at least one QuickUMLS match,
    - no negated QuickUMLS term, and
    - no likely parsing leakage based on QuickUMLS term count.

The goal is to create a clean chief-complaint dataset for downstream semantic
embedding and cohort matching.

Inputs:
    chief_complaint_preprocessed/MHH1_psychotic_chief_complaints_preprocessed.parquet
    chief_complaint_preprocessed/MHC0_chief_complaints_preprocessed.parquet

Outputs:
    chief_complaint_final/MHH1_psychotic_chief_complaints_final.parquet
    chief_complaint_final/MHH1_psychotic_chief_complaints_final.csv
    chief_complaint_final/MHC0_chief_complaints_final.parquet
    chief_complaint_final/MHC0_chief_complaints_final.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "chief_complaint_preprocessed"
OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_final"

# Cohort-specific preprocessed parquet files.
INPUTS = {
    "MHH1_psychotic": INPUT_DIR / "MHH1_psychotic_chief_complaints_preprocessed.parquet",
    "MHC0": INPUT_DIR / "MHC0_chief_complaints_preprocessed.parquet",
}

# Exclude likely parsing-leakage rows. Manual review showed that chief complaints
# with 9 or more QuickUMLS terms often contain HPI/history text rather than only
# the chief complaint.
MAX_QUICKUMLS_TERMS = 8

# Required columns for filtering and final output.
REQUIRED_COLUMNS = {
    "source_table",
    "subject_id",
    "hadm_id",
    "chief_complaint_raw",
    "chief_complaint_normalized",
    "has_chief_complaint",
    "psych_substance_self_harm_entities_affirmed",
    "psych_substance_self_harm_entities_negated",
    "has_affirmed_psych_substance_self_harm_entity",
    "has_quickumls_match",
    "quickumls_terms",
    "quickumls_negated_terms",
}

# Columns to preserve in the final dataset when present.
FINAL_COLUMNS = [
    "cohort",
    "source_table",
    "subject_id",
    "hadm_id",
    "chief_complaint_raw",
    "chief_complaint_normalized",
    "chief_complaint_tokens",
    "quickumls_terms",
    "quickumls_semtypes",
    "quickumls_extracted_text",
]


# Load one preprocessed cohort parquet and validate the expected schema.
def load_preprocessed(path: Path, cohort: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessed parquet for {cohort}: {path}")

    df = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {', '.join(missing)}. "
            "Rerun preprocessing before finalizing chief complaints."
        )

    df = df.copy()
    df.insert(0, "cohort", cohort)
    return df


# Return True for rows with any psychiatric/substance/self-harm TargetMatcher
# signal, including negated matches.
def has_any_psych_flag(df: pd.DataFrame) -> pd.Series:
    affirmed = df["psych_substance_self_harm_entities_affirmed"].fillna("").ne("")
    negated = df["psych_substance_self_harm_entities_negated"].fillna("").ne("")
    return affirmed | negated


# Count pipe-separated QuickUMLS terms in one row.
def count_quickumls_terms(value: object) -> int:
    if pd.isna(value):
        return 0
    return len([term.strip() for term in str(value).split("|") if term.strip()])


# Keep only rows that are usable, not psych-flagged, QuickUMLS-matched, not
# negated by QuickUMLS context, and not suspiciously long by QuickUMLS term count.
def filter_final_chief_complaints(df: pd.DataFrame) -> pd.DataFrame:
    quickumls_term_count = df["quickumls_terms"].map(count_quickumls_terms)
    has_negated_quickumls_term = df["quickumls_negated_terms"].fillna("").ne("")
    mask = (
        df["has_chief_complaint"]
        & ~has_any_psych_flag(df)
        & df["has_quickumls_match"]
        & ~has_negated_quickumls_term
        & (quickumls_term_count <= MAX_QUICKUMLS_TERMS)
    )
    final = df.loc[mask].copy()
    columns = [column for column in FINAL_COLUMNS if column in final.columns]
    return final[columns]


# Write a dataframe to both parquet and CSV.
def write_outputs(df: pd.DataFrame, basename: str) -> None:
    parquet_path = OUTPUT_DIR / f"{basename}.parquet"
    csv_path = OUTPUT_DIR / f"{basename}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)


# Script entry point: load, filter, and save cohort-specific outputs.
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    for cohort, path in INPUTS.items():
        original = load_preprocessed(path, cohort)
        final = filter_final_chief_complaints(original)

        write_outputs(final, f"{cohort}_chief_complaints_final")
        print(f"{cohort}: saved {len(final):,} final chief complaints", flush=True)

    print(f"Saved final chief-complaint outputs to: {OUTPUT_DIR}")


# Allow the script to be run directly with:
#     python 05_finalize_chief_complaints.py
if __name__ == "__main__":
    main()
