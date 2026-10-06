#!/usr/bin/env python3
"""Prepare covariates for the language-complexity mixed-effects models.

The matched-pairs file is the source of pair membership, age at admission, and
Elixhauser score. Prior hospital utilization and recorded patient language are
read from the finalized matched-cohort descriptor table in the local DuckDB
database. Only the variables required for modelling are exported.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

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


def prepare_race_sensitivity() -> None:
    """Add recorded race to a separate copy; preserve existing model inputs."""
    sys.path.insert(0, str(PYTHON_DIR / "02_cohort_matching"))
    from _matched_cohort_characterization_common import derive_race_group

    original = pd.read_csv(OUTPUT_PATH)
    keys = ["cohort", "subject_id", "hadm_id"]
    with duckdb.connect(str(DATABASE_PATH), read_only=True) as connection:
        race = connection.execute(
            "SELECT cohort, subject_id, hadm_id, race FROM export_matched_cohort_descriptors"
        ).fetchdf()
    if original.duplicated(keys).any() or race.duplicated(keys).any():
        raise ValueError("Duplicate admission keys in race sensitivity inputs")
    data = original.merge(race, on=keys, how="left", validate="one_to_one", indicator=True)
    if not data._merge.eq("both").all():
        raise ValueError("Some model admissions have no descriptor row")
    broad = data.race.map(derive_race_group).where(data.race.notna(), "missing")
    data["race_ethnicity_group"] = broad.map({
        "white": "White", "black": "Black", "asian": "Asian",
        "hispanic_or_latino": "Hispanic/Latino",
        "missing": "Unknown/declined/missing", "unknown_or_declined": "Unknown/declined/missing",
    }).fillna("Other recorded categories")
    out = OUTPUT_DIR / "race_ethnicity_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    data.drop(columns=["race", "_merge"]).to_csv(out / "model_covariates.csv", index=False)
    counts = data.groupby(["cohort", "race_ethnicity_group"]).agg(
        n_admissions=("hadm_id", "size"), n_patients=("subject_id", "nunique")
    ).reset_index()
    counts.to_csv(out / "race_ethnicity_category_counts.csv", index=False)
    data[["race", "race_ethnicity_group"]].drop_duplicates().to_csv(out / "race_ethnicity_mapping.csv", index=False)
    print(counts.to_string(index=False))
    print(f"Preserved all {len(data):,} admission covariate rows; saved sensitivity inputs to {out}")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-sensitivity", action="store_true", help="Add race to a separate copy of existing covariates only")
    options = parser.parse_args()
    if options.race_sensitivity:
        prepare_race_sensitivity()
    else:
        main()
