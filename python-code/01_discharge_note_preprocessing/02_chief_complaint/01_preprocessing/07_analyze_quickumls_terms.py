"""Analyze finalized QuickUMLS term extractions.

This script summarizes the `quickumls_terms` column in the finalized
chief-complaint files. It is a quality-control step for understanding which
clinical concepts are most common after preprocessing and final filtering.

Inputs:
    chief_complaint_final/MHH1_psychotic_chief_complaints_final.parquet
    chief_complaint_final/MHC0_chief_complaints_final.parquet

Outputs:
    analysis_output_chief_complaint_final/quickumls_term_summary.csv
    analysis_output_chief_complaint_final/quickumls_top_terms.csv
    analysis_output_chief_complaint_final/quickumls_top_term_pairs.csv
    analysis_output_chief_complaint_final/quickumls_term_count_distribution.csv
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "chief_complaint_final"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_chief_complaint_final"

# Finalized cohort files produced by `05_finalize_chief_complaints.py`.
INPUTS = {
    "MHH1_psychotic": INPUT_DIR / "MHH1_psychotic_chief_complaints_final.parquet",
    "MHC0": INPUT_DIR / "MHC0_chief_complaints_final.parquet",
}

# Output limits.
TOP_N_TERMS = 100
TOP_N_PAIRS = 100
PRINT_TOP_N = 10

# Minimal expected schema of the finalized chief-complaint files.
REQUIRED_COLUMNS = {
    "cohort",
    "subject_id",
    "hadm_id",
    "quickumls_terms",
    "quickumls_semtypes",
    "quickumls_extracted_text",
}


# Load one finalized cohort file and validate the expected columns.
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


# Load all configured finalized chief-complaint files.
def load_all_final_chief_complaints() -> pd.DataFrame:
    frames = [
        load_final_chief_complaints(path, cohort)
        for cohort, path in INPUTS.items()
    ]
    return pd.concat(frames, ignore_index=True)


# Split a pipe-separated QuickUMLS term string into clean terms.
def split_terms(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [term.strip() for term in str(value).split("|") if term.strip()]


# Add term-list and term-count columns used by all downstream summaries.
def add_term_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["quickumls_term_list"] = output["quickumls_terms"].map(split_terms)
    output["quickumls_term_count"] = output["quickumls_term_list"].map(len)
    return output


# Safely calculate percentages while avoiding division by zero.
def pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


# Build cohort-level summary statistics for QuickUMLS term counts.
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in df.groupby("cohort", sort=True):
        term_counts = group["quickumls_term_count"]
        rows.append(
            {
                "cohort": cohort,
                "n_admissions": len(group),
                "n_subjects": group["subject_id"].nunique(),
                "n_total_extracted_terms": int(term_counts.sum()),
                "mean_terms_per_admission": term_counts.mean(),
                "median_terms_per_admission": term_counts.median(),
                "q1_terms_per_admission": term_counts.quantile(0.25),
                "q3_terms_per_admission": term_counts.quantile(0.75),
                "min_terms_per_admission": term_counts.min(),
                "max_terms_per_admission": term_counts.max(),
                "n_admissions_with_1_term": int((term_counts == 1).sum()),
                "n_admissions_with_2_to_3_terms": int(term_counts.between(2, 3).sum()),
                "n_admissions_with_4_to_5_terms": int(term_counts.between(4, 5).sum()),
                "n_admissions_with_6_to_8_terms": int(term_counts.between(6, 8).sum()),
            }
        )
    return pd.DataFrame(rows)


# Convert finalized rows to one term per row for frequency summaries.
def explode_terms(df: pd.DataFrame) -> pd.DataFrame:
    exploded = df[["cohort", "subject_id", "hadm_id", "quickumls_term_list"]].explode(
        "quickumls_term_list"
    )
    exploded = exploded.rename(columns={"quickumls_term_list": "quickumls_term"})
    exploded = exploded.loc[exploded["quickumls_term"].fillna("").ne("")].copy()
    return exploded


# Count common QuickUMLS terms by cohort.
def build_top_terms(df: pd.DataFrame) -> pd.DataFrame:
    exploded = explode_terms(df)
    denominators = df.groupby("cohort")["hadm_id"].nunique().to_dict()
    rows = []
    for cohort, group in exploded.groupby("cohort", sort=True):
        counts = (
            group.groupby("quickumls_term")
            .agg(
                n_term_occurrences=("quickumls_term", "size"),
                n_admissions=("hadm_id", "nunique"),
                n_subjects=("subject_id", "nunique"),
            )
            .reset_index()
            .sort_values(
                ["n_admissions", "n_term_occurrences", "quickumls_term"],
                ascending=[False, False, True],
            )
            .head(TOP_N_TERMS)
        )
        counts.insert(0, "cohort", cohort)
        counts["pct_admissions"] = counts["n_admissions"].map(
            lambda value: pct(value, denominators.get(cohort, 0))
        )
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# Count common within-admission pairs of QuickUMLS terms by cohort.
def build_top_term_pairs(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in df.itertuples(index=False):
        unique_terms = sorted(set(row.quickumls_term_list))
        for term_a, term_b in combinations(unique_terms, 2):
            records.append(
                {
                    "cohort": row.cohort,
                    "subject_id": row.subject_id,
                    "hadm_id": row.hadm_id,
                    "term_a": term_a,
                    "term_b": term_b,
                }
            )
    if not records:
        return pd.DataFrame()

    pair_df = pd.DataFrame(records)
    denominators = df.groupby("cohort")["hadm_id"].nunique().to_dict()
    rows = []
    for cohort, group in pair_df.groupby("cohort", sort=True):
        counts = (
            group.groupby(["term_a", "term_b"])
            .agg(
                n_admissions=("hadm_id", "nunique"),
                n_subjects=("subject_id", "nunique"),
            )
            .reset_index()
            .sort_values(["n_admissions", "term_a", "term_b"], ascending=[False, True, True])
            .head(TOP_N_PAIRS)
        )
        counts.insert(0, "cohort", cohort)
        counts["pct_admissions"] = counts["n_admissions"].map(
            lambda value: pct(value, denominators.get(cohort, 0))
        )
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# Count how many admissions have each number of extracted terms.
def build_term_count_distribution(df: pd.DataFrame) -> pd.DataFrame:
    denominators = df.groupby("cohort")["hadm_id"].nunique().to_dict()
    counts = (
        df.groupby(["cohort", "quickumls_term_count"])
        .agg(n_admissions=("hadm_id", "nunique"))
        .reset_index()
        .sort_values(["cohort", "quickumls_term_count"])
    )
    counts["pct_admissions"] = counts.apply(
        lambda row: pct(row["n_admissions"], denominators.get(row["cohort"], 0)),
        axis=1,
    )
    return counts


# Write the analysis CSV outputs.
def write_outputs(
    summary: pd.DataFrame,
    top_terms: pd.DataFrame,
    top_pairs: pd.DataFrame,
    term_count_distribution: pd.DataFrame,
) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    outputs = {
        "summary": OUTPUT_DIR / "quickumls_term_summary.csv",
        "top_terms": OUTPUT_DIR / "quickumls_top_terms.csv",
        "top_pairs": OUTPUT_DIR / "quickumls_top_term_pairs.csv",
        "term_count_distribution": OUTPUT_DIR / "quickumls_term_count_distribution.csv",
    }
    summary.to_csv(outputs["summary"], index=False)
    top_terms.to_csv(outputs["top_terms"], index=False)
    top_pairs.to_csv(outputs["top_pairs"], index=False)
    term_count_distribution.to_csv(outputs["term_count_distribution"], index=False)
    return outputs


# Print compact aggregate output for quick inspection in the terminal.
def print_console_summary(
    summary: pd.DataFrame,
    top_terms: pd.DataFrame,
    top_pairs: pd.DataFrame,
    outputs: dict[str, Path],
) -> None:
    print("\n=== QuickUMLS Term Summary ===")
    print(summary.to_string(index=False))

    print(f"\n=== Top {PRINT_TOP_N} QuickUMLS Terms Per Cohort ===")
    for cohort, group in top_terms.groupby("cohort", sort=True):
        print(f"\n{cohort}")
        print(
            group.head(PRINT_TOP_N)[
                ["quickumls_term", "n_admissions", "pct_admissions"]
            ].to_string(index=False)
        )

    print(f"\n=== Top {PRINT_TOP_N} QuickUMLS Term Pairs Per Cohort ===")
    for cohort, group in top_pairs.groupby("cohort", sort=True):
        print(f"\n{cohort}")
        print(
            group.head(PRINT_TOP_N)[
                ["term_a", "term_b", "n_admissions", "pct_admissions"]
            ].to_string(index=False)
        )

    print("\nSaved outputs:")
    for path in outputs.values():
        print(f"  {path}")


# Script entry point: load finalized chief complaints, summarize QuickUMLS terms,
# and write local CSV outputs.
def main() -> None:
    df = add_term_columns(load_all_final_chief_complaints())
    summary = build_summary(df)
    top_terms = build_top_terms(df)
    top_pairs = build_top_term_pairs(df)
    term_count_distribution = build_term_count_distribution(df)
    outputs = write_outputs(summary, top_terms, top_pairs, term_count_distribution)
    print_console_summary(summary, top_terms, top_pairs, outputs)


# Allow the script to be run directly with:
#     python 07_analyze_quickumls_terms.py
if __name__ == "__main__":
    main()
