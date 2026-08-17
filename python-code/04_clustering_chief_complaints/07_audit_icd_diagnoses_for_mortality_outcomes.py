"""Audit primary ICD diagnoses for mortality outcome admissions.

This script compares ICD diagnoses for:

    - admissions followed by post-discharge death within 1 year
    - admissions ending in in-hospital death

It uses the admission-level mortality dataset produced by
`06_analyze_subject_level_mortality_risk.py` and the matched-cohort diagnosis
export. Outputs are aggregate diagnosis counts plus an ID-level audit table. No
note text or chief complaint text is written.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
COHORT_MATCHING_DIR = PROJECT_DIR / "02_cohort_matching"
sys.path.insert(0, str(COHORT_MATCHING_DIR))

import _matched_cohort_characterization_common as common  # noqa: E402


MORTALITY_DATASET_PATH = (
    SCRIPT_DIR
    / "analysis_output_subject_level_mortality_risk"
    / "admission_level_mortality_all_matched_dataset.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_mortality_icd_audit"

ID_COLUMNS = ["cohort", "subject_id", "hadm_id"]
OUTCOME_COLUMNS = [
    "post_discharge_death_within_1y",
    "died_in_hospital",
]
GROUP_COLUMNS = [
    "combined_chief_complaint_group",
    "combined_chief_complaint_group_label",
    "pure_chief_complaint_group",
    "pure_chief_complaint_group_label",
]


def load_mortality_dataset() -> pd.DataFrame:
    """Load admission-level mortality rows and normalize IDs/outcomes."""
    if not MORTALITY_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Missing admission-level mortality dataset. Run "
            "04_clustering_chief_complaints/06_analyze_subject_level_mortality_risk.py first: "
            f"{MORTALITY_DATASET_PATH}"
        )
    data = pd.read_csv(MORTALITY_DATASET_PATH)
    required = {*ID_COLUMNS, "pair_id", *OUTCOME_COLUMNS, "dod", "dischtime"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Mortality dataset is missing columns: {missing}")

    data = common.validate_id_columns(data, "admission_level_mortality_dataset")
    data["pair_id"] = pd.to_numeric(data["pair_id"], errors="raise").astype(int)
    for column in OUTCOME_COLUMNS:
        data[column] = data[column].astype(bool)
    for column in ["admittime", "dischtime", "dod", "deathtime"]:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")
    for column in GROUP_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    return data


def load_diagnoses() -> pd.DataFrame:
    """Load matched-cohort diagnosis ICD rows."""
    diagnoses = common.validate_id_columns(
        common.load_required_table("diagnoses"),
        "diagnoses",
    )
    required = {
        *ID_COLUMNS,
        "seq_num",
        "icd_version",
        "icd_code",
        "long_title",
        "is_psychiatric_icd",
        "is_grey_zone_physical_icd",
        "is_primary_diagnosis",
    }
    missing = sorted(required - set(diagnoses.columns))
    if missing:
        raise ValueError(f"Diagnosis table is missing required columns: {missing}")

    diagnoses = diagnoses.copy()
    diagnoses["seq_num"] = pd.to_numeric(diagnoses["seq_num"], errors="coerce")
    diagnoses["icd_version"] = pd.to_numeric(
        diagnoses["icd_version"], errors="coerce"
    ).astype("Int64")
    diagnoses["icd_code"] = diagnoses["icd_code"].fillna("").astype(str).str.strip()
    diagnoses["long_title"] = diagnoses["long_title"].fillna("").astype(str).str.strip()
    for column in [
        "is_psychiatric_icd",
        "is_grey_zone_physical_icd",
        "is_primary_diagnosis",
    ]:
        diagnoses[column] = (
            pd.to_numeric(diagnoses[column], errors="coerce").fillna(0).astype(int)
        )
    return diagnoses


def build_primary_diagnosis_table(diagnoses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one primary diagnosis row per admission plus primary-count QC.

    If an admission has more than one row marked primary, the lowest seq_num row
    is kept for the audit and the issue is written to the QC table.
    """
    primary = diagnoses.loc[diagnoses["is_primary_diagnosis"].eq(1)].copy()
    all_admissions = diagnoses.loc[:, ID_COLUMNS].drop_duplicates()
    primary_counts = (
        primary.groupby(ID_COLUMNS)
        .size()
        .rename("n_primary_diagnosis_rows")
        .reset_index()
    )
    primary_qc = all_admissions.merge(primary_counts, on=ID_COLUMNS, how="left")
    primary_qc["n_primary_diagnosis_rows"] = (
        primary_qc["n_primary_diagnosis_rows"].fillna(0).astype(int)
    )
    primary_qc = primary_qc.loc[
        primary_qc["n_primary_diagnosis_rows"].ne(1)
    ].copy()

    primary = (
        primary.sort_values([*ID_COLUMNS, "seq_num", "icd_version", "icd_code"])
        .drop_duplicates(ID_COLUMNS, keep="first")
        .copy()
    )

    primary = primary.loc[
        :,
        ID_COLUMNS
        + [
            "icd_version",
            "icd_code",
            "long_title",
            "is_psychiatric_icd",
            "is_grey_zone_physical_icd",
        ],
    ].rename(
        columns={
            "icd_version": "primary_icd_version",
            "icd_code": "primary_icd_code",
            "long_title": "primary_icd_title",
            "is_psychiatric_icd": "primary_is_psychiatric_icd",
            "is_grey_zone_physical_icd": "primary_is_grey_zone_physical_icd",
        }
    )
    return primary, primary_qc


def summarize_all_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Summarize all diagnosis rows per admission."""
    return (
        diagnoses.groupby(ID_COLUMNS, as_index=False)
        .agg(
            n_total_icd_codes=("icd_code", "count"),
            n_total_psychiatric_icd_codes=("is_psychiatric_icd", "sum"),
            n_total_grey_zone_icd_codes=("is_grey_zone_physical_icd", "sum"),
            all_icd_codes=("icd_code", lambda values: " | ".join(map(str, values))),
            all_icd_titles=("long_title", lambda values: " | ".join(map(str, values))),
        )
    )


def build_mortality_icd_audit(
    mortality: pd.DataFrame,
    diagnoses: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach primary/all ICD summaries to mortality outcome admissions."""
    primary, primary_qc = build_primary_diagnosis_table(diagnoses)
    all_diagnoses = summarize_all_diagnoses(diagnoses)
    audit = mortality.merge(primary, on=ID_COLUMNS, how="left", validate="one_to_one")
    audit = audit.merge(all_diagnoses, on=ID_COLUMNS, how="left", validate="one_to_one")

    audit["mortality_outcome_group"] = pd.NA
    audit.loc[audit["post_discharge_death_within_1y"], "mortality_outcome_group"] = (
        "post_discharge_death_within_1y"
    )
    audit.loc[audit["died_in_hospital"], "mortality_outcome_group"] = (
        "died_in_hospital"
    )

    outcome_rows = []
    for outcome in OUTCOME_COLUMNS:
        subset = audit.loc[audit[outcome]].copy()
        subset["mortality_outcome"] = outcome
        outcome_rows.append(subset)
    outcome_audit = pd.concat(outcome_rows, ignore_index=True)
    outcome_audit["days_discharge_to_death"] = pd.to_numeric(
        outcome_audit.get("days_discharge_to_death"), errors="coerce"
    )
    outcome_audit["days_admission_to_death"] = pd.to_numeric(
        outcome_audit.get("days_admission_to_death"), errors="coerce"
    )

    keep_columns = [
        "mortality_outcome",
        "pair_id",
        *ID_COLUMNS,
        "admittime",
        "dischtime",
        "dod",
        "days_admission_to_death",
        "days_discharge_to_death",
        "combined_chief_complaint_group",
        "combined_chief_complaint_group_label",
        "pure_chief_complaint_group",
        "pure_chief_complaint_group_label",
        "primary_icd_version",
        "primary_icd_code",
        "primary_icd_title",
        "primary_is_psychiatric_icd",
        "primary_is_grey_zone_physical_icd",
        "n_total_icd_codes",
        "n_total_psychiatric_icd_codes",
        "n_total_grey_zone_icd_codes",
        "all_icd_codes",
        "all_icd_titles",
    ]
    outcome_audit = outcome_audit.loc[
        :, [column for column in keep_columns if column in outcome_audit.columns]
    ].sort_values(["mortality_outcome", "cohort", "primary_icd_title", "hadm_id"])
    return outcome_audit, primary_qc


def summarize_primary_icd(audit: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Count primary ICD diagnoses inside mortality outcome groups."""
    rows = []
    for keys, group in audit.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row_base = dict(zip(group_columns, keys, strict=True))
        counts = (
            group.groupby(
                [
                    "primary_icd_version",
                    "primary_icd_code",
                    "primary_icd_title",
                    "primary_is_psychiatric_icd",
                    "primary_is_grey_zone_physical_icd",
                ],
                dropna=False,
            )
            .agg(
                n_admissions=("hadm_id", "nunique"),
                n_subjects=("subject_id", "nunique"),
            )
            .reset_index()
        )
        total = group["hadm_id"].nunique()
        counts["pct_admissions_in_group"] = 100.0 * counts["n_admissions"] / total
        for _, count_row in counts.iterrows():
            rows.append({**row_base, **count_row.to_dict()})
    return pd.DataFrame(rows).sort_values(
        [*group_columns, "n_admissions", "primary_icd_title"],
        ascending=[True] * len(group_columns) + [False, True],
    )


def summarize_outcome_counts(audit: pd.DataFrame) -> pd.DataFrame:
    """Count mortality-outcome admissions with diagnosis coverage by cohort."""
    return (
        audit.groupby(["mortality_outcome", "cohort"], dropna=False)
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
            n_primary_icd_codes=("primary_icd_code", "count"),
            n_primary_psychiatric_icd=("primary_is_psychiatric_icd", "sum"),
            n_primary_grey_zone_icd=("primary_is_grey_zone_physical_icd", "sum"),
        )
        .reset_index()
        .sort_values(["mortality_outcome", "cohort"])
    )


def write_top_n(table: pd.DataFrame, path: Path, group_columns: list[str], n: int = 20) -> None:
    """Write top N rows per group from a primary ICD count table."""
    top = (
        table.groupby(group_columns, group_keys=False, dropna=False)
        .head(n)
        .reset_index(drop=True)
    )
    top.to_csv(path, index=False)


def main() -> None:
    """Run the mortality ICD audit and write CSV outputs."""
    mortality = load_mortality_dataset()
    diagnoses = load_diagnoses()
    audit, primary_qc = build_mortality_icd_audit(mortality, diagnoses)

    overall_primary = summarize_primary_icd(audit, ["mortality_outcome", "cohort"])
    combined_primary = summarize_primary_icd(
        audit.dropna(subset=["combined_chief_complaint_group"]),
        [
            "mortality_outcome",
            "combined_chief_complaint_group",
            "combined_chief_complaint_group_label",
            "cohort",
        ],
    )
    pure_primary = summarize_primary_icd(
        audit.dropna(subset=["pure_chief_complaint_group"]),
        [
            "mortality_outcome",
            "pure_chief_complaint_group",
            "pure_chief_complaint_group_label",
            "cohort",
        ],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_DIR / "mortality_outcome_icd_admission_audit.csv", index=False)
    primary_qc.to_csv(
        OUTPUT_DIR / "mortality_outcome_primary_icd_count_qc.csv",
        index=False,
    )
    summarize_outcome_counts(audit).to_csv(
        OUTPUT_DIR / "mortality_outcome_icd_coverage_summary.csv",
        index=False,
    )
    overall_primary.to_csv(
        OUTPUT_DIR / "mortality_outcome_primary_icd_counts_overall.csv",
        index=False,
    )
    combined_primary.to_csv(
        OUTPUT_DIR / "mortality_outcome_primary_icd_counts_combined_subgroups.csv",
        index=False,
    )
    pure_primary.to_csv(
        OUTPUT_DIR / "mortality_outcome_primary_icd_counts_pure_subgroups.csv",
        index=False,
    )
    write_top_n(
        overall_primary,
        OUTPUT_DIR / "mortality_outcome_top_primary_icd_overall.csv",
        ["mortality_outcome", "cohort"],
    )
    write_top_n(
        combined_primary,
        OUTPUT_DIR / "mortality_outcome_top_primary_icd_combined_subgroups.csv",
        [
            "mortality_outcome",
            "combined_chief_complaint_group",
            "combined_chief_complaint_group_label",
            "cohort",
        ],
    )
    write_top_n(
        pure_primary,
        OUTPUT_DIR / "mortality_outcome_top_primary_icd_pure_subgroups.csv",
        [
            "mortality_outcome",
            "pure_chief_complaint_group",
            "pure_chief_complaint_group_label",
            "cohort",
        ],
    )

    print(f"Saved mortality ICD audit to: {OUTPUT_DIR}")
    print("\n=== Mortality ICD coverage ===")
    print(summarize_outcome_counts(audit).to_string(index=False))
    print("\n=== Top primary ICD diagnoses overall ===")
    print(
        overall_primary.groupby(["mortality_outcome", "cohort"], group_keys=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
