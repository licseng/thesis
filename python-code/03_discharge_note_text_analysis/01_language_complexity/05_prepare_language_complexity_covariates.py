#!/usr/bin/env python3
"""Prepare covariates for the language-complexity mixed-effects models.

The matched-pairs file is the source of pair membership, age at admission, and
Elixhauser score. Prior hospital utilization and recorded patient language are
read from the finalized matched-cohort descriptor table in the local DuckDB
database. Only the variables required for modelling are exported.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR.parent.parent
THESIS_DIR = SCRIPT_DIR.parents[3]

PAIRS_PATH = (
    PYTHON_DIR
    / "02_cohort_matching"
    / "matched_cohort_output"
    / "matched_pairs.csv"
)
DATABASE_PATH = THESIS_DIR / "DataBase"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_language_complexity_inference"
OUTPUT_PATH = OUTPUT_DIR / "language_complexity_model_covariates.csv"


def pair_covariates(pairs: pd.DataFrame) -> pd.DataFrame:
    """Reshape pair-level matching variables to one row per admission."""
    mhc1 = pd.DataFrame(
        {
            "pair_id": pairs["pair_id"],
            "cohort": "MHC1_psychotic",
            "subject_id": pairs["mhc1_subject_id"],
            "hadm_id": pairs["mhc1_hadm_id"],
            "age_at_admission": pairs["mhc1_age_at_admission"],
            "elixhauser_score": pairs["mhc1_elixhauser_score"],
        }
    )
    mhc0 = pd.DataFrame(
        {
            "pair_id": pairs["pair_id"],
            "cohort": "MHC0",
            "subject_id": pairs["mhc0_subject_id"],
            "hadm_id": pairs["mhc0_hadm_id"],
            "age_at_admission": pairs["mhc0_age_at_admission"],
            "elixhauser_score": pairs["mhc0_elixhauser_score"],
        }
    )
    return pd.concat([mhc1, mhc0], ignore_index=True)


def load_descriptor_covariates() -> pd.DataFrame:
    """Read utilization and language for matched admissions from DuckDB."""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Missing local DuckDB database: {DATABASE_PATH}")

    connection = duckdb.connect(str(DATABASE_PATH), read_only=True)
    try:
        descriptors = connection.execute(
            """
            SELECT
                cohort,
                subject_id,
                hadm_id,
                n_prior_all_admissions_for_subject,
                language
            FROM export_matched_cohort_descriptors
            """
        ).fetchdf()
    finally:
        connection.close()
    return descriptors


def main() -> None:
    if not PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing matched-pairs file: {PAIRS_PATH}")

    pairs = pd.read_csv(PAIRS_PATH)
    covariates = pair_covariates(pairs)
    descriptors = load_descriptor_covariates()

    join_columns = ["cohort", "subject_id", "hadm_id"]
    if descriptors.duplicated(join_columns).any():
        raise ValueError("Matched-cohort descriptors contain duplicate admission keys.")

    covariates = covariates.merge(
        descriptors,
        on=join_columns,
        how="left",
        validate="one_to_one",
    )
    expected_rows = 2 * len(pairs)
    if len(covariates) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} admission rows, found {len(covariates)}."
        )

    numeric_columns = [
        "age_at_admission",
        "elixhauser_score",
        "n_prior_all_admissions_for_subject",
    ]
    for column in numeric_columns:
        covariates[column] = pd.to_numeric(covariates[column], errors="coerce")
    if covariates[numeric_columns].isna().any().any():
        missing = covariates[numeric_columns].isna().sum()
        raise ValueError(f"Missing numeric model covariates:\n{missing[missing > 0]}")
    if (covariates["n_prior_all_admissions_for_subject"] < 0).any():
        raise ValueError("Prior-admission counts cannot be negative.")

    covariates["age_at_admission_per_10y"] = (
        covariates["age_at_admission"] / 10.0
    )
    covariates["elixhauser_score_per_5pt"] = (
        covariates["elixhauser_score"] / 5.0
    )
    covariates["log1p_prior_all_admissions"] = np.log1p(
        covariates["n_prior_all_admissions_for_subject"]
    )
    covariates["language_group"] = np.select(
        [
            covariates["language"].eq("English"),
            covariates["language"].notna(),
        ],
        ["English", "Non-English"],
        default="Missing",
    )

    output_columns = [
        "pair_id",
        "cohort",
        "subject_id",
        "hadm_id",
        "age_at_admission",
        "elixhauser_score",
        "n_prior_all_admissions_for_subject",
        "language_group",
        "age_at_admission_per_10y",
        "elixhauser_score_per_5pt",
        "log1p_prior_all_admissions",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    covariates[output_columns].to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(covariates):,} model-covariate rows to {OUTPUT_PATH}")
    print("\nRecorded patient-language groups:")
    print(
        covariates.groupby(["cohort", "language_group"], observed=True)
        .size()
        .rename("n_admissions")
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
