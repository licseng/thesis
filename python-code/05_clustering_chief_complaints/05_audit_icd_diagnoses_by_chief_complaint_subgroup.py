"""Audit ICD diagnosis structure by chief-complaint subgroup.

This script checks whether the matched cohort's physical-primary-diagnosis
definition behaves as expected inside the two exclusive combined chief-complaint
groups from `01_describe_chief_complaint_subgroups.py`:

    - abdominal_pain_nausea_vomiting
    - chest_pain_shortness_of_breath

It reads `export_matched_cohort_diagnoses` directly from the DuckDB database or
from a matching export file through `_matched_cohort_characterization_common.py`.
Outputs are aggregate summaries plus an ID-level audit table with ICD codes and
titles. No note text or chief complaint text is written.
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


ASSIGNMENT_PATH = (
    SCRIPT_DIR
    / "analysis_output_chief_complaint_subgroup_balance_check"
    / "chief_complaint_subgroup_admission_assignments.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_chief_complaint_subgroup_icd_audit"

ID_COLUMNS = ["cohort", "subject_id", "hadm_id"]
GROUP_COLUMN = "exclusive_combined_group"
GROUP_LABELS = {
    "abdominal_pain_nausea_vomiting": "Abdominal pain / nausea / vomiting",
    "chest_pain_shortness_of_breath": "Chest pain / shortness of breath",
}


def load_combined_group_assignments() -> pd.DataFrame:
    """Load exclusive combined chief-complaint subgroup assignment rows."""
    if not ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            "Missing subgroup assignments. Run "
            "05_clustering_chief_complaints/01_describe_chief_complaint_subgroups.py first: "
            f"{ASSIGNMENT_PATH}"
        )
    assignments = pd.read_csv(ASSIGNMENT_PATH)
    missing = sorted(set(ID_COLUMNS + ["pair_id", GROUP_COLUMN]) - set(assignments.columns))
    if missing:
        raise ValueError(f"Subgroup assignment file is missing columns: {missing}")

    assignments = assignments.copy()
    assignments["cohort"] = assignments["cohort"].astype("string").str.strip()
    assignments["subject_id"] = pd.to_numeric(
        assignments["subject_id"], errors="raise"
    ).astype(int)
    assignments["hadm_id"] = pd.to_numeric(assignments["hadm_id"], errors="raise").astype(int)
    assignments[GROUP_COLUMN] = assignments[GROUP_COLUMN].astype("string").str.strip()
    assignments = assignments.loc[
        assignments[GROUP_COLUMN].isin(GROUP_LABELS)
    ].copy()
    assignments["exclusive_combined_group_label"] = assignments[GROUP_COLUMN].map(
        GROUP_LABELS
    )
    return assignments


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
        diagnoses[column] = pd.to_numeric(diagnoses[column], errors="coerce").fillna(0).astype(int)
    return diagnoses


def build_admission_icd_audit(
    assignments: pd.DataFrame,
    diagnoses: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse diagnosis rows to one audit row per subgroup admission."""
    group_columns = [
        "pair_id",
        *ID_COLUMNS,
        GROUP_COLUMN,
        "exclusive_combined_group_label",
    ]
    subgroup_admissions = assignments.loc[:, group_columns].drop_duplicates(ID_COLUMNS)
    subgroup_diagnoses = subgroup_admissions.merge(
        diagnoses,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_many",
    )

    primary_rows = subgroup_diagnoses.loc[
        subgroup_diagnoses["is_primary_diagnosis"].eq(1)
    ].copy()
    primary_counts = primary_rows.groupby(ID_COLUMNS).size()
    bad_primary = primary_counts.loc[primary_counts.ne(1)]
    if not bad_primary.empty:
        raise ValueError(
            "Expected exactly one primary diagnosis per subgroup admission, "
            f"but found {len(bad_primary)} admissions with non-one primary rows."
        )

    primary = primary_rows.loc[
        :,
        ID_COLUMNS
        + [
            "seq_num",
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
    primary = primary.drop(columns=["seq_num"])

    secondary = subgroup_diagnoses.loc[
        subgroup_diagnoses["is_primary_diagnosis"].ne(1)
    ].copy()
    secondary_summary = (
        secondary.groupby(ID_COLUMNS, as_index=False)
        .agg(
            n_secondary_icd_codes=("icd_code", "count"),
            n_secondary_psychiatric_icd_codes=("is_psychiatric_icd", "sum"),
            n_secondary_grey_zone_icd_codes=("is_grey_zone_physical_icd", "sum"),
        )
    )
    all_summary = (
        subgroup_diagnoses.groupby(ID_COLUMNS, as_index=False)
        .agg(
            n_total_icd_codes=("icd_code", "count"),
            n_total_psychiatric_icd_codes=("is_psychiatric_icd", "sum"),
            n_total_grey_zone_icd_codes=("is_grey_zone_physical_icd", "sum"),
        )
    )
    secondary_psych_titles = (
        secondary.loc[secondary["is_psychiatric_icd"].eq(1)]
        .sort_values(ID_COLUMNS + ["seq_num", "icd_code"])
        .groupby(ID_COLUMNS)
        .agg(
            secondary_psychiatric_icd_codes=(
                "icd_code",
                lambda values: " | ".join(map(str, values)),
            ),
            secondary_psychiatric_icd_titles=(
                "long_title",
                lambda values: " | ".join(map(str, values)),
            ),
        )
        .reset_index()
    )

    audit = subgroup_admissions.merge(primary, on=ID_COLUMNS, how="left", validate="one_to_one")
    audit = audit.merge(all_summary, on=ID_COLUMNS, how="left", validate="one_to_one")
    audit = audit.merge(secondary_summary, on=ID_COLUMNS, how="left", validate="one_to_one")
    audit = audit.merge(
        secondary_psych_titles,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    fill_zero_columns = [
        "n_total_icd_codes",
        "n_total_psychiatric_icd_codes",
        "n_total_grey_zone_icd_codes",
        "n_secondary_icd_codes",
        "n_secondary_psychiatric_icd_codes",
        "n_secondary_grey_zone_icd_codes",
    ]
    for column in fill_zero_columns:
        audit[column] = audit[column].fillna(0).astype(int)
    audit["has_secondary_psychiatric_icd"] = audit[
        "n_secondary_psychiatric_icd_codes"
    ].gt(0)
    audit["has_secondary_grey_zone_icd"] = audit[
        "n_secondary_grey_zone_icd_codes"
    ].gt(0)
    audit["has_any_psychiatric_icd"] = audit["n_total_psychiatric_icd_codes"].gt(0)
    audit["has_any_grey_zone_icd"] = audit["n_total_grey_zone_icd_codes"].gt(0)
    audit["secondary_psychiatric_icd_codes"] = audit[
        "secondary_psychiatric_icd_codes"
    ].fillna("")
    audit["secondary_psychiatric_icd_titles"] = audit[
        "secondary_psychiatric_icd_titles"
    ].fillna("")
    return audit.sort_values([GROUP_COLUMN, "cohort", "pair_id", "subject_id", "hadm_id"])


def summarize_binary_flags(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize ICD audit flags by subgroup/cohort."""
    flag_columns = [
        "primary_is_psychiatric_icd",
        "primary_is_grey_zone_physical_icd",
        "has_secondary_psychiatric_icd",
        "has_secondary_grey_zone_icd",
        "has_any_psychiatric_icd",
        "has_any_grey_zone_icd",
    ]
    rows = []
    for (group_name, group_label, cohort), group in audit.groupby(
        [GROUP_COLUMN, "exclusive_combined_group_label", "cohort"],
        dropna=False,
    ):
        for column in flag_columns:
            values = group[column].astype(bool)
            rows.append(
                {
                    GROUP_COLUMN: group_name,
                    "exclusive_combined_group_label": group_label,
                    "cohort": cohort,
                    "measure": column,
                    "n_admissions": len(group),
                    "n_positive": int(values.sum()),
                    "pct_positive": 100.0 * values.mean(),
                }
            )
    return pd.DataFrame(rows).sort_values([GROUP_COLUMN, "measure", "cohort"])


def summarize_icd_counts(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize total and secondary ICD-code counts by subgroup/cohort."""
    count_columns = [
        "n_total_icd_codes",
        "n_secondary_icd_codes",
        "n_secondary_psychiatric_icd_codes",
        "n_secondary_grey_zone_icd_codes",
    ]
    rows = []
    for (group_name, group_label, cohort), group in audit.groupby(
        [GROUP_COLUMN, "exclusive_combined_group_label", "cohort"],
        dropna=False,
    ):
        for column in count_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            rows.append(
                {
                    GROUP_COLUMN: group_name,
                    "exclusive_combined_group_label": group_label,
                    "cohort": cohort,
                    "measure": column,
                    "n_admissions": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(rows).sort_values([GROUP_COLUMN, "measure", "cohort"])


def summarize_primary_diagnoses(audit: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Write top primary ICD title/code distributions by subgroup/cohort."""
    rows = []
    for (group_name, group_label, cohort), group in audit.groupby(
        [GROUP_COLUMN, "exclusive_combined_group_label", "cohort"],
        dropna=False,
    ):
        denominator = group["hadm_id"].nunique()
        counts = (
            group.groupby(["primary_icd_version", "primary_icd_code", "primary_icd_title"], dropna=False)
            .agg(n_admissions=("hadm_id", "nunique"))
            .reset_index()
            .sort_values(["n_admissions", "primary_icd_code"], ascending=[False, True])
            .head(top_n)
        )
        counts[GROUP_COLUMN] = group_name
        counts["exclusive_combined_group_label"] = group_label
        counts["cohort"] = cohort
        counts["pct_within_subgroup_cohort"] = (
            100.0 * counts["n_admissions"] / denominator
        )
        rows.append(counts)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).loc[
        :,
        [
            GROUP_COLUMN,
            "exclusive_combined_group_label",
            "cohort",
            "primary_icd_version",
            "primary_icd_code",
            "primary_icd_title",
            "n_admissions",
            "pct_within_subgroup_cohort",
        ],
    ]


def summarize_secondary_psych_titles(audit: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Summarize secondary psychiatric ICD titles among admissions with any."""
    with_psych = audit.loc[audit["has_secondary_psychiatric_icd"]].copy()
    rows = []
    for (group_name, group_label, cohort), group in with_psych.groupby(
        [GROUP_COLUMN, "exclusive_combined_group_label", "cohort"],
        dropna=False,
    ):
        title_rows = []
        for titles in group["secondary_psychiatric_icd_titles"]:
            title_rows.extend(title for title in str(titles).split(" | ") if title)
        if not title_rows:
            continue
        denominator = len(group)
        counts = (
            pd.Series(title_rows, name="secondary_psychiatric_icd_title")
            .value_counts()
            .rename_axis("secondary_psychiatric_icd_title")
            .reset_index(name="n_code_occurrences")
            .head(top_n)
        )
        counts[GROUP_COLUMN] = group_name
        counts["exclusive_combined_group_label"] = group_label
        counts["cohort"] = cohort
        counts["n_admissions_with_secondary_psych_icd"] = denominator
        rows.append(counts)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).loc[
        :,
        [
            GROUP_COLUMN,
            "exclusive_combined_group_label",
            "cohort",
            "secondary_psychiatric_icd_title",
            "n_code_occurrences",
            "n_admissions_with_secondary_psych_icd",
        ],
    ]


def main() -> None:
    """Run ICD diagnosis audit for combined chief-complaint subgroups."""
    assignments = load_combined_group_assignments()
    diagnoses = load_diagnoses()
    audit = build_admission_icd_audit(assignments, diagnoses)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_DIR / "chief_complaint_subgroup_icd_admission_audit.csv", index=False)
    summarize_binary_flags(audit).to_csv(
        OUTPUT_DIR / "chief_complaint_subgroup_icd_flag_summary.csv",
        index=False,
    )
    summarize_icd_counts(audit).to_csv(
        OUTPUT_DIR / "chief_complaint_subgroup_icd_count_summary.csv",
        index=False,
    )
    summarize_primary_diagnoses(audit).to_csv(
        OUTPUT_DIR / "chief_complaint_subgroup_top_primary_icd_diagnoses.csv",
        index=False,
    )
    summarize_secondary_psych_titles(audit).to_csv(
        OUTPUT_DIR / "chief_complaint_subgroup_top_secondary_psych_icd_titles.csv",
        index=False,
    )

    print(f"Saved chief-complaint subgroup ICD audit outputs to: {OUTPUT_DIR}")
    print("\n=== ICD Flag Summary ===")
    print(summarize_binary_flags(audit).to_string(index=False))
    print("\n=== ICD Count Summary ===")
    print(summarize_icd_counts(audit).to_string(index=False))


if __name__ == "__main__":
    main()
