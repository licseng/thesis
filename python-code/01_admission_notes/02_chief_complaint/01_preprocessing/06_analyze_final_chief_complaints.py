"""Analyze final chief-complaint QuickUMLS extractions.

This script reads the finalized chief-complaint files produced by
`05_finalize_chief_complaints.py` and summarizes how many QuickUMLS terms were
kept per admission. It also checks the pre-finalized preprocessing outputs for
rows above the finalization cap, because those rows are removed before they can
appear in the final files.

Inputs:
    chief_complaint_preprocessed/MHH1_psychotic_chief_complaints_preprocessed.parquet
    chief_complaint_preprocessed/MHC0_chief_complaints_preprocessed.parquet
    chief_complaint_final/MHH1_psychotic_chief_complaints_final.parquet
    chief_complaint_final/MHC0_chief_complaints_final.parquet

Outputs:
    analysis_output_chief_complaint_final/final_quickumls_term_count_summary.csv
    analysis_output_chief_complaint_final/pre_final_chief_complaints_more_than_<threshold>_quickumls_terms.csv
    analysis_output_chief_complaint_final/pre_final_chief_complaints_more_than_<threshold>_quickumls_terms_sample.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PRE_FINAL_INPUT_DIR = SCRIPT_DIR / "chief_complaint_preprocessed"
FINAL_INPUT_DIR = SCRIPT_DIR / "chief_complaint_final"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_chief_complaint_final"

# Pre-finalized cohort files. These still contain rows removed by
# `05_finalize_chief_complaints.py`, including rows above the QuickUMLS term cap.
PRE_FINAL_INPUTS = {
    "MHH1_psychotic": PRE_FINAL_INPUT_DIR / "MHH1_psychotic_chief_complaints_preprocessed.parquet",
    "MHC0": PRE_FINAL_INPUT_DIR / "MHC0_chief_complaints_preprocessed.parquet",
}

# Final cohort-specific chief-complaint files.
FINAL_INPUTS = {
    "MHH1_psychotic": FINAL_INPUT_DIR / "MHH1_psychotic_chief_complaints_final.parquet",
    "MHC0": FINAL_INPUT_DIR / "MHC0_chief_complaints_final.parquet",
}

# Review threshold for suspected parsing leakage.
LONG_TERM_COUNT_THRESHOLD = 8
LONG_TERM_SAMPLE_SIZE_PER_COHORT = 100

# Required columns in the final files.
REQUIRED_COLUMNS = {
    "cohort",
    "source_table",
    "subject_id",
    "hadm_id",
    "chief_complaint_raw",
    "chief_complaint_normalized",
    "quickumls_terms",
    "quickumls_cuis",
    "quickumls_semtypes",
    "quickumls_extracted_text",
}

# Required columns used to reconstruct the pre-finalization candidate pool before
# the QuickUMLS term cap is applied.
PRE_FINAL_REQUIRED_COLUMNS = {
    "source_table",
    "subject_id",
    "hadm_id",
    "chief_complaint_raw",
    "chief_complaint_normalized",
    "has_chief_complaint",
    "psych_substance_self_harm_entities_affirmed",
    "psych_substance_self_harm_entities_negated",
    "has_quickumls_match",
    "quickumls_terms",
    "quickumls_cuis",
    "quickumls_semtypes",
    "quickumls_extracted_text",
}

# Columns to include in local review CSVs.
REVIEW_COLUMNS = [
    "cohort",
    "source_table",
    "subject_id",
    "hadm_id",
    "quickumls_term_count",
    "chief_complaint_raw",
    "chief_complaint_normalized",
    "chief_complaint_context_text",
    "quickumls_terms",
    "quickumls_cuis",
    "quickumls_semtypes",
    "quickumls_extracted_text",
    "quickumls_matches_json",
]


# Load one final cohort file and validate that expected columns are present.
def load_final_chief_complaints(path: Path, cohort: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing final chief-complaint file for {cohort}: {path}")

    df = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")

    df = df.copy()
    if "cohort" not in df.columns:
        df.insert(0, "cohort", cohort)
    return df


# Load all configured final chief-complaint files.
def load_all_final_chief_complaints() -> pd.DataFrame:
    frames = [load_final_chief_complaints(path, cohort) for cohort, path in FINAL_INPUTS.items()]
    return pd.concat(frames, ignore_index=True)


# Load one pre-finalized cohort parquet and validate that expected columns exist.
def load_pre_final_chief_complaints(path: Path, cohort: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing pre-finalized chief-complaint file for {cohort}: {path}")

    df = pd.read_parquet(path)
    missing = sorted(PRE_FINAL_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df.insert(0, "cohort", cohort)
    return df


# Load all configured pre-finalized chief-complaint files.
def load_all_pre_final_chief_complaints() -> pd.DataFrame:
    frames = [
        load_pre_final_chief_complaints(path, cohort)
        for cohort, path in PRE_FINAL_INPUTS.items()
    ]
    return pd.concat(frames, ignore_index=True)


# Count pipe-separated QuickUMLS terms in one row.
def count_quickumls_terms(value: object) -> int:
    if pd.isna(value):
        return 0
    return len([term.strip() for term in str(value).split("|") if term.strip()])


# Add QuickUMLS term-count columns used by summaries and review outputs.
def add_term_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["quickumls_term_count"] = output["quickumls_terms"].map(count_quickumls_terms)
    output["has_long_quickumls_term_list"] = (
        output["quickumls_term_count"] > LONG_TERM_COUNT_THRESHOLD
    )
    return output


# Return True for rows with any psychiatric/substance/self-harm TargetMatcher
# signal, including negated matches.
def has_any_psych_flag(df: pd.DataFrame) -> pd.Series:
    affirmed = df["psych_substance_self_harm_entities_affirmed"].fillna("").ne("")
    negated = df["psych_substance_self_harm_entities_negated"].fillna("").ne("")
    return affirmed | negated


# Reconstruct the pre-finalization candidate pool immediately before the
# QuickUMLS term-count cap is applied.
def pre_final_candidate_pool(df: pd.DataFrame) -> pd.DataFrame:
    candidates = add_term_count_columns(df)
    mask = (
        candidates["has_chief_complaint"]
        & ~has_any_psych_flag(candidates)
        & candidates["has_quickumls_match"]
    )
    return candidates.loc[mask].copy()


# Build cohort-level distribution statistics for final QuickUMLS term counts,
# plus the number of pre-finalization candidates removed by the term-count cap.
def build_term_count_summary(
    final_df: pd.DataFrame,
    pre_final_candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    pre_final_long_counts = (
        pre_final_candidates.groupby("cohort")["has_long_quickumls_term_list"]
        .sum()
        .astype(int)
        .to_dict()
    )
    pre_final_counts = pre_final_candidates.groupby("cohort").size().to_dict()

    for cohort, group in final_df.groupby("cohort", sort=True):
        term_counts = group["quickumls_term_count"]
        n_long_before_finalization = int(pre_final_long_counts.get(cohort, 0))
        n_pre_final_candidates = int(pre_final_counts.get(cohort, 0))
        rows.append(
            {
                "cohort": cohort,
                "n_pre_final_candidates_before_term_cap": n_pre_final_candidates,
                f"n_more_than_{LONG_TERM_COUNT_THRESHOLD}_quickumls_terms_before_finalization": n_long_before_finalization,
                f"pct_more_than_{LONG_TERM_COUNT_THRESHOLD}_quickumls_terms_before_finalization": (
                    100.0 * n_long_before_finalization / n_pre_final_candidates
                    if n_pre_final_candidates
                    else 0.0
                ),
                "n_final_admissions": len(group),
                "mean_quickumls_terms_per_admission": term_counts.mean(),
                "sd_quickumls_terms_per_admission": term_counts.std(ddof=1),
                "median_quickumls_terms_per_admission": term_counts.median(),
                "q1_quickumls_terms_per_admission": term_counts.quantile(0.25),
                "q3_quickumls_terms_per_admission": term_counts.quantile(0.75),
                "min_quickumls_terms_per_admission": term_counts.min(),
                "max_quickumls_terms_per_admission": term_counts.max(),
            }
        )
    return pd.DataFrame(rows)


# Return review columns that are available in the current dataframe.
def available_review_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in REVIEW_COLUMNS if column in df.columns]


# Write all rows and a per-cohort sample of pre-finalization rows with long
# QuickUMLS term lists.
def write_long_term_review_files(df: pd.DataFrame) -> tuple[Path, Path]:
    long_rows = df.loc[df["has_long_quickumls_term_list"]].copy()
    long_rows = long_rows.sort_values(
        ["cohort", "quickumls_term_count", "subject_id", "hadm_id"],
        ascending=[True, False, True, True],
    )

    all_path = (
        OUTPUT_DIR
        / f"pre_final_chief_complaints_more_than_{LONG_TERM_COUNT_THRESHOLD}_quickumls_terms.csv"
    )
    sample_path = (
        OUTPUT_DIR
        / f"pre_final_chief_complaints_more_than_{LONG_TERM_COUNT_THRESHOLD}_quickumls_terms_sample.csv"
    )

    review_columns = available_review_columns(long_rows)
    long_rows[review_columns].to_csv(all_path, index=False)

    sample = (
        long_rows.groupby("cohort", group_keys=False)
        .head(LONG_TERM_SAMPLE_SIZE_PER_COHORT)
    )
    sample[review_columns].to_csv(sample_path, index=False)

    return all_path, sample_path


# Script entry point: summarize final QuickUMLS extraction counts and write local
# review CSVs for rows removed by the pre-finalization term cap.
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    final_df = add_term_count_columns(load_all_final_chief_complaints())
    pre_final_candidates = pre_final_candidate_pool(load_all_pre_final_chief_complaints())
    summary = build_term_count_summary(final_df, pre_final_candidates)

    summary_path = OUTPUT_DIR / "final_quickumls_term_count_summary.csv"
    summary.to_csv(summary_path, index=False)
    all_long_path, sample_long_path = write_long_term_review_files(pre_final_candidates)

    print("\n=== Final QuickUMLS term-count summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved all >{LONG_TERM_COUNT_THRESHOLD}-term rows to: {all_long_path}")
    print(f"Saved >{LONG_TERM_COUNT_THRESHOLD}-term review sample to: {sample_long_path}")


# Allow the script to be run directly with:
#     python 06_analyze_final_chief_complaints.py
if __name__ == "__main__":
    main()
