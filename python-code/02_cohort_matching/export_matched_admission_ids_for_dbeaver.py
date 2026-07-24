"""Export matched admission IDs for DBeaver filtering.

This script converts the matched-pair table into one row per matched admission:
one MHH1_psychotic row and one MHC0 row per pair. The output is intended as a
small helper table for DBeaver joins against larger MIMIC tables such as
admissions, procedures_icd, poe, or labevents.

No note text or clinical free text is exported.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MATCHED_PAIRS_PATH = SCRIPT_DIR / "matched_cohort_output" / "matched_pairs.csv"
OUTPUT_DIR = SCRIPT_DIR / "matched_cohort_output"
OUTPUT_PATH = OUTPUT_DIR / "matched_admission_ids_for_dbeaver.csv"

REQUIRED_COLUMNS = {
    "pair_id",
    "mhh_subject_id",
    "mhh_hadm_id",
    "mhc0_subject_id",
    "mhc0_hadm_id",
}


def load_matched_pairs() -> pd.DataFrame:
    """Load matched pairs and validate required ID columns."""
    if not MATCHED_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing matched pairs file: {MATCHED_PAIRS_PATH}")

    matched_pairs = pd.read_csv(MATCHED_PAIRS_PATH)
    missing_columns = sorted(REQUIRED_COLUMNS - set(matched_pairs.columns))
    if missing_columns:
        raise ValueError(
            f"{MATCHED_PAIRS_PATH} is missing columns: {missing_columns}"
        )
    return matched_pairs


def build_matched_admission_ids(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    """Return one row per matched admission with cohort and pair identifiers."""
    mhh = matched_pairs.loc[
        :,
        ["pair_id", "mhh_subject_id", "mhh_hadm_id"],
    ].rename(
        columns={
            "mhh_subject_id": "subject_id",
            "mhh_hadm_id": "hadm_id",
        }
    )
    mhh.insert(1, "matched_role", "case")
    mhh.insert(2, "cohort", "MHH1_psychotic")

    mhc0 = matched_pairs.loc[
        :,
        ["pair_id", "mhc0_subject_id", "mhc0_hadm_id"],
    ].rename(
        columns={
            "mhc0_subject_id": "subject_id",
            "mhc0_hadm_id": "hadm_id",
        }
    )
    mhc0.insert(1, "matched_role", "control")
    mhc0.insert(2, "cohort", "MHC0")

    matched_ids = pd.concat([mhh, mhc0], ignore_index=True)
    matched_ids["pair_id"] = matched_ids["pair_id"].astype(int)
    matched_ids["subject_id"] = matched_ids["subject_id"].astype(int)
    matched_ids["hadm_id"] = matched_ids["hadm_id"].astype(int)

    duplicated_admissions = matched_ids.duplicated(
        ["cohort", "subject_id", "hadm_id"],
        keep=False,
    )
    if duplicated_admissions.any():
        raise ValueError(
            "Matched admission ID output would contain duplicate cohort + "
            f"subject_id + hadm_id rows: {duplicated_admissions.sum()}"
        )

    return matched_ids.sort_values(["pair_id", "matched_role"]).reset_index(drop=True)


def main() -> None:
    """Write the DBeaver helper CSV."""
    matched_pairs = load_matched_pairs()
    matched_ids = build_matched_admission_ids(matched_pairs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matched_ids.to_csv(OUTPUT_PATH, index=False)

    summary = (
        matched_ids.groupby(["cohort", "matched_role"], as_index=False)
        .agg(
            n_rows=("hadm_id", "size"),
            n_subjects=("subject_id", "nunique"),
            n_admissions=("hadm_id", "nunique"),
        )
        .sort_values(["matched_role", "cohort"])
    )

    print(f"Saved matched admission IDs to: {OUTPUT_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
