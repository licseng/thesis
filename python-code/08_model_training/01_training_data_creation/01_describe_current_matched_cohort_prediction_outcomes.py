"""Describe candidate prediction outcomes in the current matched cohort.

This script is an early planning check for the clinical prediction model. It
does not build the full training population yet. It only asks whether the
current matched MHH1/MHC0 cohort has enough outcome events for:

    - 30-day readmission after discharge
    - prolonged hospital length of stay, defined as LOS > 7 days

The 30-day readmission label uses all MIMIC admissions for matched subjects,
not only admissions inside the matched cohort.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
COHORT_DIR = PROJECT_DIR / "02_cohort_matching"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_current_matched_cohort_outcomes"
ID_COLUMNS = ["cohort", "subject_id", "hadm_id"]
ONE_DAY_SECONDS = 24 * 60 * 60

sys.path.insert(0, str(COHORT_DIR))
import _matched_cohort_characterization_common as common  # noqa: E402


def load_current_matched_admissions() -> pd.DataFrame:
    """Load one row per current matched-cohort admission."""
    descriptors = common.validate_id_columns(
        common.load_required_table("descriptors"),
        "matched_cohort_descriptors",
    )
    required_columns = {
        "pair_id",
        "matched_role",
        "cohort",
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
        "hospital_expire_flag",
    }
    available_required = required_columns & set(descriptors.columns)
    missing = sorted(required_columns - set(descriptors.columns))
    if missing:
        raise ValueError(f"Descriptor table is missing columns: {missing}")

    output = descriptors.loc[:, sorted(available_required)].copy()
    output["admittime"] = pd.to_datetime(output["admittime"], errors="coerce")
    output["dischtime"] = pd.to_datetime(output["dischtime"], errors="coerce")
    output["deathtime"] = pd.to_datetime(output["deathtime"], errors="coerce")
    output["hospital_expire_flag"] = pd.to_numeric(
        output["hospital_expire_flag"],
        errors="coerce",
    ).fillna(0).astype(int)
    output["hospital_los_days"] = (
        output["dischtime"] - output["admittime"]
    ).dt.total_seconds() / ONE_DAY_SECONDS
    output.loc[output["hospital_los_days"].lt(0), "hospital_los_days"] = pd.NA
    output["prolonged_los_gt_7d"] = output["hospital_los_days"].gt(7)
    return output


def load_subject_admission_history() -> pd.DataFrame:
    """Load all MIMIC admissions for subjects represented in the matched cohort."""
    history = common.load_optional_table("subject_admission_history")
    if history is None:
        raise FileNotFoundError(
            "Missing subject admission history table. Rerun "
            "sql-scripts/06_save_tables/02_Additional_info_export_on_cohort.sql."
        )
    history = history.copy()
    history["subject_id"] = pd.to_numeric(history["subject_id"], errors="raise").astype(int)
    history["hadm_id"] = pd.to_numeric(history["hadm_id"], errors="raise").astype(int)
    history["admittime"] = pd.to_datetime(history["admittime"], errors="coerce")
    history["dischtime"] = pd.to_datetime(history["dischtime"], errors="coerce")
    return history.loc[:, ["subject_id", "hadm_id", "admittime", "dischtime"]]


def add_next_readmission_label(
    matched: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Add next admission timing and 30-day readmission label."""
    rows = []
    history_by_subject = {
        subject_id: group.sort_values(["admittime", "hadm_id"]).reset_index(drop=True)
        for subject_id, group in history.dropna(subset=["admittime"]).groupby("subject_id")
    }

    for row in matched.itertuples(index=False):
        subject_history = history_by_subject.get(row.subject_id)
        next_hadm_id = pd.NA
        next_admittime = pd.NaT
        days_to_next_admission = pd.NA
        if subject_history is not None and pd.notna(row.dischtime):
            later = subject_history.loc[
                (subject_history["admittime"] > row.dischtime)
                & (subject_history["hadm_id"].ne(row.hadm_id))
            ]
            if not later.empty:
                next_row = later.iloc[0]
                next_hadm_id = int(next_row["hadm_id"])
                next_admittime = next_row["admittime"]
                days_to_next_admission = (
                    next_admittime - row.dischtime
                ).total_seconds() / ONE_DAY_SECONDS
        rows.append(
            {
                "cohort": row.cohort,
                "subject_id": row.subject_id,
                "hadm_id": row.hadm_id,
                "next_hadm_id_after_discharge": next_hadm_id,
                "next_admittime_after_discharge": next_admittime,
                "days_to_next_admission_after_discharge": days_to_next_admission,
            }
        )

    next_admissions = pd.DataFrame(rows)
    output = matched.merge(next_admissions, on=ID_COLUMNS, how="left", validate="one_to_one")
    output["died_in_hospital"] = output["hospital_expire_flag"].eq(1)
    output["eligible_for_30d_readmission"] = (
        output["dischtime"].notna() & ~output["died_in_hospital"]
    )
    output["readmission_within_30d"] = (
        output["eligible_for_30d_readmission"]
        & pd.to_numeric(
            output["days_to_next_admission_after_discharge"],
            errors="coerce",
        ).between(0, 30, inclusive="both")
    )
    return output


def summarize_admission_level(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate outcome prevalence by admission."""
    rows = []
    for cohort, group in outcomes.groupby("cohort"):
        at_risk = group.loc[group["eligible_for_30d_readmission"]]
        rows.append(
            {
                "cohort": cohort,
                "n_admissions": len(group),
                "n_subjects": group["subject_id"].nunique(),
                "n_in_hospital_deaths": int(group["died_in_hospital"].sum()),
                "n_admissions_eligible_for_30d_readmission": len(at_risk),
                "n_admissions_readmitted_within_30d": int(
                    at_risk["readmission_within_30d"].sum()
                ),
                "pct_eligible_admissions_readmitted_within_30d": 100
                * at_risk["readmission_within_30d"].mean()
                if len(at_risk)
                else pd.NA,
                "n_admissions_with_los_available": int(group["hospital_los_days"].notna().sum()),
                "n_admissions_prolonged_los_gt_7d": int(
                    group["prolonged_los_gt_7d"].sum()
                ),
                "pct_admissions_prolonged_los_gt_7d": 100
                * group.loc[group["hospital_los_days"].notna(), "prolonged_los_gt_7d"].mean(),
                "mean_los_days": group["hospital_los_days"].mean(),
                "median_los_days": group["hospital_los_days"].median(),
            }
        )
    return pd.DataFrame(rows)


def summarize_subject_level(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether subjects ever have candidate outcomes in matched admissions."""
    rows = []
    for cohort, group in outcomes.groupby("cohort"):
        subject_summary = (
            group.groupby("subject_id", as_index=False)
            .agg(
                n_matched_admissions=("hadm_id", "nunique"),
                any_eligible_for_30d_readmission=("eligible_for_30d_readmission", "max"),
                any_readmission_within_30d=("readmission_within_30d", "max"),
                any_prolonged_los_gt_7d=("prolonged_los_gt_7d", "max"),
            )
        )
        readmission_subjects = subject_summary.loc[
            subject_summary["any_eligible_for_30d_readmission"]
        ]
        rows.append(
            {
                "cohort": cohort,
                "n_subjects": len(subject_summary),
                "n_matched_admissions": int(subject_summary["n_matched_admissions"].sum()),
                "mean_matched_admissions_per_subject": subject_summary[
                    "n_matched_admissions"
                ].mean(),
                "n_subjects_eligible_for_30d_readmission": len(readmission_subjects),
                "n_subjects_with_any_30d_readmission_after_matched_admission": int(
                    readmission_subjects["any_readmission_within_30d"].sum()
                ),
                "pct_eligible_subjects_with_any_30d_readmission": 100
                * readmission_subjects["any_readmission_within_30d"].mean()
                if len(readmission_subjects)
                else pd.NA,
                "n_subjects_with_any_prolonged_los_gt_7d": int(
                    subject_summary["any_prolonged_los_gt_7d"].sum()
                ),
                "pct_subjects_with_any_prolonged_los_gt_7d": 100
                * subject_summary["any_prolonged_los_gt_7d"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run outcome prevalence checks and write CSV outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matched = load_current_matched_admissions()
    history = load_subject_admission_history()
    outcomes = add_next_readmission_label(matched, history)

    outcomes.to_csv(OUTPUT_DIR / "current_matched_cohort_prediction_outcomes.csv", index=False)
    admission_summary = summarize_admission_level(outcomes)
    subject_summary = summarize_subject_level(outcomes)
    admission_summary.to_csv(OUTPUT_DIR / "current_matched_cohort_outcome_admission_summary.csv", index=False)
    subject_summary.to_csv(OUTPUT_DIR / "current_matched_cohort_outcome_subject_summary.csv", index=False)

    print(f"Wrote outputs to {OUTPUT_DIR}")
    print("\nAdmission-level summary:")
    print(admission_summary.to_string(index=False))
    print("\nSubject-level summary:")
    print(subject_summary.to_string(index=False))


if __name__ == "__main__":
    main()
