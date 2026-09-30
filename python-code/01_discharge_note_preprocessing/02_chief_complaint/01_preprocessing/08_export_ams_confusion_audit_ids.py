"""Export aggregate-audit identifiers for ambiguous AMS/confusion admissions.

This script does not export chief-complaint or note text. It selects finalized
MHC1-psychosis admissions whose QuickUMLS concepts contain altered mental
status and/or confusion and whose current admission has a secondary psychosis
ICD code. The resulting ID-and-flag CSV can be imported into DuckDB to audit
whether these admissions contain a substantive physical disease/injury code.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_CC_PATH = (
    SCRIPT_DIR
    / "chief_complaint_final"
    / "MHC1_psychotic_chief_complaints_final.parquet"
)
PARSED_CONTEXT_PATH = (
    SCRIPT_DIR.parent.parent
    / "01_discharge_note_parsing"
    / "parsed_chief_complaints"
    / "MHC1_psychotic_chief_complaints_from_discharge_notes.parquet"
)
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_chief_complaint_final"
OUTPUT_PATH = (
    OUTPUT_DIR
    / "MHC1_psychotic_current_secondary_psychosis_ams_confusion_ids_for_dbeaver.csv"
)


def split_terms(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        term.strip().casefold()
        for term in str(value).split("|")
        if term.strip()
    }


def main() -> None:
    final_cc = pd.read_parquet(
        FINAL_CC_PATH,
        columns=["subject_id", "hadm_id", "quickumls_terms"],
    )
    context = pd.read_parquet(
        PARSED_CONTEXT_PATH,
        columns=[
            "subject_id",
            "hadm_id",
            "has_prior_psychosis",
            "has_current_secondary_psychosis",
            "psychosis_context_version",
        ],
    )

    terms = final_cc["quickumls_terms"].map(split_terms)
    final_cc = final_cc.assign(
        has_altered_mental_status_cc=terms.map(
            lambda values: "altered mental status" in values
        ),
        has_confusion_cc=terms.map(lambda values: "confusion" in values),
    )

    audit = final_cc.merge(
        context,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )
    audit = audit.loc[
        audit["has_current_secondary_psychosis"].eq(1)
        & (
            audit["has_altered_mental_status_cc"]
            | audit["has_confusion_cc"]
        )
    ].copy()

    output_columns = [
        "subject_id",
        "hadm_id",
        "has_altered_mental_status_cc",
        "has_confusion_cc",
        "has_prior_psychosis",
        "has_current_secondary_psychosis",
        "psychosis_context_version",
    ]
    audit = audit[output_columns].sort_values(["subject_id", "hadm_id"])

    OUTPUT_DIR.mkdir(exist_ok=True)
    audit.to_csv(OUTPUT_PATH, index=False)

    print("=== Aggregate AMS/confusion audit export summary ===")
    print(f"Admissions: {audit['hadm_id'].nunique():,}")
    print(f"Subjects: {audit['subject_id'].nunique():,}")
    print(
        audit.groupby("psychosis_context_version", dropna=False)
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
        )
        .reset_index()
        .to_string(index=False)
    )
    print(f"Saved ID-and-flag export to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
