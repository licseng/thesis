"""Analyze preprocessed chief-complaint outputs.

This script summarizes the preprocessed chief-complaint parquet files without
printing raw clinical text to the terminal. Raw chief complaints are written only
to local CSV files that are intended for manual review.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


 # Paths
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "chief_complaint_preprocessed"
OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_preprocessed_analysis_output"

 # Cohort-specific preprocessed parquet files to summarize.
INPUTS = {
    "MHH1_psychotic": INPUT_DIR / "MHH1_psychotic_chief_complaints_preprocessed.parquet",
    "MHC0": INPUT_DIR / "MHC0_chief_complaints_preprocessed.parquet",
}

 # Number of no-QuickUMLS-match rows to export per cohort for manual review.
NO_QUICKUMLS_SAMPLE_SIZE = 200

 # Columns expected from the preprocessing step. Missing columns usually mean the
 # preprocessing script needs to be rerun or updated.
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
}

 # Load one cohort parquet and validate that required preprocessing columns exist.
def load_preprocessed(path: Path, cohort: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing preprocessed parquet for {cohort}: {path}")

    df = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {', '.join(missing)}. "
            "Rerun preprocessing before analysis."
        )

    df = df.copy()
    df["cohort"] = cohort
    return df

 # Load and concatenate all configured cohort files into one dataframe.
def load_all_preprocessed() -> pd.DataFrame:
    frames = [load_preprocessed(path, cohort) for cohort, path in INPUTS.items()]
    return pd.concat(frames, ignore_index=True)

 # Safe percentage helper that returns 0 when the denominator is 0.
def pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0

 # Build cohort-level quality-control counts for usability, psychiatric flags,
 # and QuickUMLS coverage.
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in df.groupby("cohort", sort=True):
        total = len(group)
        usable = int(group["has_chief_complaint"].sum())
        unusable = total - usable
        psych_affirmed = int(
            group["has_affirmed_psych_substance_self_harm_entity"].sum()
        )
        psych_negated = int(
            (
                group["has_chief_complaint"]
                & group["psych_substance_self_harm_entities_negated"].fillna("").ne("")
            ).sum()
        )
        quickumls_matches = int(
            (group["has_chief_complaint"] & group["has_quickumls_match"]).sum()
        )
        no_quickumls = int(
            (group["has_chief_complaint"] & ~group["has_quickumls_match"]).sum()
        )

        rows.append(
            {
                "cohort": cohort,
                "n_rows_total": total,
                "n_subjects": group["subject_id"].nunique(),
                "n_admissions": group["hadm_id"].nunique(),
                "n_usable_for_preprocessing": usable,
                "pct_usable_for_preprocessing": pct(usable, total),
                "n_unusable_empty_or_placeholder_chief_complaint": unusable,
                "pct_unusable_empty_or_placeholder_chief_complaint": pct(
                    unusable,
                    total,
                ),
                "n_with_affirmed_psych_substance_self_harm_flag": psych_affirmed,
                "pct_with_affirmed_psych_substance_self_harm_flag_among_usable": pct(
                    psych_affirmed,
                    usable,
                ),
                "n_with_negated_psych_substance_self_harm_flag": psych_negated,
                "pct_with_negated_psych_substance_self_harm_flag_among_usable": pct(
                    psych_negated,
                    usable,
                ),
                "n_with_quickumls_match": quickumls_matches,
                "pct_with_quickumls_match_among_usable": pct(quickumls_matches, usable),
                "n_without_quickumls_match": no_quickumls,
                "pct_without_quickumls_match_among_usable": pct(no_quickumls, usable),
            }
        )

    return pd.DataFrame(rows)

 # Select available columns that are useful for manual review CSVs.
def review_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        "cohort",
        "source_table",
        "subject_id",
        "hadm_id",
        "chief_complaint_raw",
        "chief_complaint_normalized",
        "psych_substance_self_harm_entities_affirmed",
        "psych_substance_self_harm_entities_negated",
        "medspacy_entities_all",
        "quickumls_terms",
        "quickumls_cuis",
        "quickumls_semtypes",
        "quickumls_extracted_text",
        "quickumls_matches_json",
        "has_quickumls_match",
    ]
    return [column for column in columns if column in df.columns]

 # Export usable chief complaints with affirmed psych/substance/self-harm flags.
def write_psych_flagged_chief_complaints(df: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "chief_complaints_with_psychiatric_flags.csv"
    flagged = df.loc[
        df["has_chief_complaint"]
        & df["has_affirmed_psych_substance_self_harm_entity"]
    ].copy()
    flagged = flagged[review_columns(flagged)]
    flagged.to_csv(output_path, index=False)
    return output_path

 # Export a per-cohort sample of usable chief complaints without QuickUMLS matches.
def write_no_quickumls_samples(df: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "chief_complaints_without_quickumls_match_sample.csv"
    no_match = df.loc[
        df["has_chief_complaint"]
        & ~df["has_quickumls_match"]
    ].copy()
    no_match = (
        no_match.sort_values(["cohort", "subject_id", "hadm_id"])
        .groupby("cohort", group_keys=False)
        .head(NO_QUICKUMLS_SAMPLE_SIZE)
    )
    no_match = no_match[review_columns(no_match)]
    no_match.to_csv(output_path, index=False)
    return output_path

 # Script entry point: load data, write summary/review CSVs, and print output paths.
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_all_preprocessed()
    summary = build_summary(df)
    summary_path = OUTPUT_DIR / "preprocessed_chief_complaint_summary.csv"
    summary.to_csv(summary_path, index=False)

    psych_flagged_path = write_psych_flagged_chief_complaints(df)
    no_quickumls_path = write_no_quickumls_samples(df)

    print("\n=== Preprocessed chief complaint summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved psychiatric-flagged chief complaints to: {psych_flagged_path}")
    print(f"Saved no-QuickUMLS-match samples to: {no_quickumls_path}")

 # Allow the script to be run directly with:
 #     python 04_analyze_preprocessed_chief_complaints.py
if __name__ == "__main__":
    main()
