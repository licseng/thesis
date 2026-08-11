"""Analyze outcomes/utilization for pure chief-complaint subgroups.

This script uses the five granular chief-complaint subgroups from
`01_describe_chief_complaint_subgroups.py`:

    - abdominal pain
    - shortness of breath
    - chest pain
    - altered mental status
    - nausea vomiting

It keeps only "pure" subgroup admissions: admissions that matched exactly one of
those five groups. This gives a 5x2 analysis:

    pure chief-complaint subgroup x cohort

Outputs are aggregate summaries plus one ID-level analysis dataset. No raw chief
complaint text or discharge-note text is written by this script.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ASSIGNMENT_PATH = (
    SCRIPT_DIR
    / "analysis_output_chief_complaint_subgroup_balance_check"
    / "chief_complaint_subgroup_admission_assignments.csv"
)
OUTCOME_HELPER_PATH = SCRIPT_DIR / "03_analyze_chief_complaint_subgroup_outcomes.py"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_pure_chief_complaint_subgroup_outcomes"

ID_COLUMNS = ["cohort", "subject_id", "hadm_id"]
GROUP_COLUMN = "pure_chief_complaint_group"
GROUP_LABEL_COLUMN = "pure_chief_complaint_group_label"
SUBGROUPS = {
    "abdominal pain": "Abdominal pain",
    "shortness of breath": "Shortness of breath",
    "chest pain": "Chest pain",
    "altered mental status": "Altered mental status",
    "nausea vomiting": "Nausea / vomiting",
}
SUBGROUP_FLAG_COLUMNS = {
    "abdominal pain": "has_abdominal_pain",
    "shortness of breath": "has_shortness_of_breath",
    "chest pain": "has_chest_pain",
    "altered mental status": "has_altered_mental_status",
    "nausea vomiting": "has_nausea_vomiting",
}
CATEGORICAL_COLUMNS = [
    "admission_type",
    "admission_location",
    "discharge_location",
    "insurance",
    "race_group",
    "ethnicity_from_race",
    "language",
    "marital_status",
]
NUMERIC_SUMMARY_COLUMNS = [
    "age_at_admission",
    "elixhauser_score",
    "hospital_los_days",
    "ed_los_hours",
    "n_labevents_rows",
    "n_microbiologyevents_rows",
    "n_poe_rows",
    "n_poe_detail_rows",
]


def load_outcome_helper():
    """Load helper functions from the combined-subgroup outcome script."""
    spec = importlib.util.spec_from_file_location("chief_complaint_outcomes", OUTCOME_HELPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load outcome helper script: {OUTCOME_HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pure_subgroup_assignments() -> pd.DataFrame:
    """Load subgroup assignments and keep rows matching exactly one subgroup."""
    if not ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            "Missing subgroup assignments. Run "
            "04_clustering_chief_complaints/01_describe_chief_complaint_subgroups.py first: "
            f"{ASSIGNMENT_PATH}"
        )

    assignments = pd.read_csv(ASSIGNMENT_PATH)
    required_columns = {
        "pair_id",
        *ID_COLUMNS,
        "sex",
        "insurance_group",
        "age_at_admission",
        "elixhauser_score",
        "n_chief_complaint_subgroups_matched",
        *SUBGROUP_FLAG_COLUMNS.values(),
    }
    missing = sorted(required_columns - set(assignments.columns))
    if missing:
        raise ValueError(f"Subgroup assignment file is missing columns: {missing}")

    assignments = assignments.copy()
    assignments["cohort"] = assignments["cohort"].astype("string").str.strip()
    assignments["subject_id"] = pd.to_numeric(
        assignments["subject_id"], errors="raise"
    ).astype(int)
    assignments["hadm_id"] = pd.to_numeric(assignments["hadm_id"], errors="raise").astype(int)
    assignments["n_chief_complaint_subgroups_matched"] = pd.to_numeric(
        assignments["n_chief_complaint_subgroups_matched"], errors="raise"
    ).astype(int)

    pure = assignments.loc[
        assignments["n_chief_complaint_subgroups_matched"].eq(1)
    ].copy()
    for subgroup, flag_column in SUBGROUP_FLAG_COLUMNS.items():
        pure.loc[pure[flag_column].astype(bool), GROUP_COLUMN] = subgroup

    pure = pure.loc[pure[GROUP_COLUMN].isin(SUBGROUPS)].copy()
    pure[GROUP_LABEL_COLUMN] = pure[GROUP_COLUMN].map(SUBGROUPS)
    return pure


def build_analysis_dataset(
    assignments: pd.DataFrame,
    descriptors: pd.DataFrame,
    utilization_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Join pure subgroup assignments to descriptors and utilization counts."""
    descriptor_columns = [
        column
        for column in [
            *ID_COLUMNS,
            "matched_role",
            "gender",
            "admittime",
            "dischtime",
            "deathtime",
            "dod",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "race_group",
            "ethnicity_from_race",
            "marital_status",
            "hospital_expire_flag",
            "died_in_hospital",
            "has_deathtime",
            "has_dod",
            "hospital_los_days",
            "ed_los_hours",
        ]
        if column in descriptors.columns
    ]
    analysis = assignments.merge(
        descriptors.loc[:, descriptor_columns],
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    count_columns = [
        column
        for column in utilization_counts.columns
        if column in ID_COLUMNS or column.startswith("n_")
    ]
    analysis = analysis.merge(
        utilization_counts.loc[:, count_columns],
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    for column in [column for column in analysis.columns if column.startswith("n_")]:
        analysis[column] = analysis[column].fillna(0).astype(int)

    keep_columns = [
        column
        for column in [
            "pair_id",
            *ID_COLUMNS,
            GROUP_COLUMN,
            GROUP_LABEL_COLUMN,
            "sex",
            "gender",
            "insurance_group",
            "age_at_admission",
            "elixhauser_score",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "race_group",
            "ethnicity_from_race",
            "marital_status",
            "hospital_los_days",
            "ed_los_hours",
            "hospital_expire_flag",
            "died_in_hospital",
            "has_deathtime",
            "has_dod",
            "n_labevents_rows",
            "n_microbiologyevents_rows",
            "n_poe_rows",
            "n_poe_detail_rows",
        ]
        if column in analysis.columns
    ]
    return analysis.loc[:, keep_columns].sort_values(
        [GROUP_COLUMN, "cohort", "pair_id", "subject_id", "hadm_id"]
    )


def summarize_counts(analysis: pd.DataFrame) -> pd.DataFrame:
    """Count admissions and subjects in each pure subgroup/cohort cell."""
    return (
        analysis.groupby([GROUP_COLUMN, GROUP_LABEL_COLUMN, "cohort"])
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
            n_pairs=("pair_id", "nunique"),
        )
        .reset_index()
        .sort_values([GROUP_COLUMN, "cohort"])
    )


def summarize_numeric(analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric outcomes/utilization by pure subgroup and cohort."""
    available_columns = [
        column for column in NUMERIC_SUMMARY_COLUMNS if column in analysis.columns
    ]
    rows = []
    for (group_name, group_label, cohort), group in analysis.groupby(
        [GROUP_COLUMN, GROUP_LABEL_COLUMN, "cohort"],
        dropna=False,
    ):
        for column in available_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            rows.append(
                {
                    GROUP_COLUMN: group_name,
                    GROUP_LABEL_COLUMN: group_label,
                    "cohort": cohort,
                    "measure": column,
                    "n_admissions": len(group),
                    "n_nonmissing": len(values),
                    "mean": values.mean() if len(values) else pd.NA,
                    "sd": values.std(ddof=1) if len(values) > 1 else pd.NA,
                    "median": values.median() if len(values) else pd.NA,
                    "q1": values.quantile(0.25) if len(values) else pd.NA,
                    "q3": values.quantile(0.75) if len(values) else pd.NA,
                    "iqr": (
                        values.quantile(0.75) - values.quantile(0.25)
                        if len(values)
                        else pd.NA
                    ),
                    "min": values.min() if len(values) else pd.NA,
                    "max": values.max() if len(values) else pd.NA,
                    "n_with_any_positive_value": int(values.gt(0).sum())
                    if len(values)
                    else 0,
                    "pct_with_any_positive_value": 100.0 * values.gt(0).mean()
                    if len(values)
                    else pd.NA,
                }
            )
    return pd.DataFrame(rows).sort_values([GROUP_COLUMN, "measure", "cohort"])


def summarize_binary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize binary mortality/death indicators by pure subgroup and cohort."""
    binary_columns = [
        column
        for column in ["died_in_hospital", "has_deathtime", "has_dod"]
        if column in analysis.columns
    ]
    rows = []
    for (group_name, group_label, cohort), group in analysis.groupby(
        [GROUP_COLUMN, GROUP_LABEL_COLUMN, "cohort"],
        dropna=False,
    ):
        for column in binary_columns:
            values = group[column].dropna().astype(bool)
            rows.append(
                {
                    GROUP_COLUMN: group_name,
                    GROUP_LABEL_COLUMN: group_label,
                    "cohort": cohort,
                    "measure": column,
                    "n_admissions": len(group),
                    "n_nonmissing": len(values),
                    "n_positive": int(values.sum()),
                    "pct_positive": 100.0 * values.mean() if len(values) else pd.NA,
                }
            )
    return pd.DataFrame(rows).sort_values([GROUP_COLUMN, "measure", "cohort"])


def summarize_categorical(analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize categorical descriptors by pure subgroup and cohort."""
    available_columns = [
        column for column in CATEGORICAL_COLUMNS if column in analysis.columns
    ]
    rows = []
    denominators = (
        analysis.groupby([GROUP_COLUMN, GROUP_LABEL_COLUMN, "cohort"])["hadm_id"]
        .nunique()
        .to_dict()
    )
    for column in available_columns:
        values = analysis.loc[:, [GROUP_COLUMN, GROUP_LABEL_COLUMN, "cohort", "hadm_id", column]].copy()
        values[column] = values[column].fillna("missing").astype(str).str.strip()
        values.loc[values[column].eq(""), column] = "missing"
        counts = (
            values.groupby([GROUP_COLUMN, GROUP_LABEL_COLUMN, "cohort", column], as_index=False)[
                "hadm_id"
            ]
            .nunique()
            .rename(columns={column: "category", "hadm_id": "n_admissions"})
        )
        counts["measure"] = column
        counts["pct_within_subgroup_cohort"] = counts.apply(
            lambda row: 100.0
            * row["n_admissions"]
            / denominators.get(
                (row[GROUP_COLUMN], row[GROUP_LABEL_COLUMN], row["cohort"]),
                0,
            ),
            axis=1,
        )
        rows.append(counts)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).loc[
        :,
        [
            GROUP_COLUMN,
            GROUP_LABEL_COLUMN,
            "cohort",
            "measure",
            "category",
            "n_admissions",
            "pct_within_subgroup_cohort",
        ],
    ].sort_values(
        [GROUP_COLUMN, "measure", "cohort", "n_admissions"],
        ascending=[True, True, True, False],
    )


def build_pair_membership_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Report how often both admissions from a pair fall into each pure subgroup."""
    rows = []
    for group_name, group in analysis.groupby(GROUP_COLUMN):
        group_label = SUBGROUPS.get(group_name, group_name)
        pair_counts = group.groupby("pair_id")["cohort"].nunique()
        rows.append(
            {
                GROUP_COLUMN: group_name,
                GROUP_LABEL_COLUMN: group_label,
                "n_pairs_with_any_group_admission": int(pair_counts.size),
                "n_pairs_with_both_cohorts_in_group": int(pair_counts.eq(2).sum()),
                "pct_pairs_with_both_cohorts_in_group": 100.0
                * pair_counts.eq(2).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(GROUP_COLUMN)


def main() -> None:
    """Write pure chief-complaint subgroup outcome/utilization summaries."""
    outcome_helper = load_outcome_helper()
    assignments = load_pure_subgroup_assignments()
    descriptors = outcome_helper.load_descriptors()
    utilization_counts = outcome_helper.load_or_build_utilization_counts()
    analysis = build_analysis_dataset(assignments, descriptors, utilization_counts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(
        OUTPUT_DIR / "pure_chief_complaint_subgroup_outcome_analysis_dataset.csv",
        index=False,
    )
    summarize_counts(analysis).to_csv(
        OUTPUT_DIR / "pure_chief_complaint_subgroup_outcome_counts.csv",
        index=False,
    )
    summarize_numeric(analysis).to_csv(
        OUTPUT_DIR / "pure_chief_complaint_subgroup_numeric_outcome_summary.csv",
        index=False,
    )
    summarize_binary(analysis).to_csv(
        OUTPUT_DIR / "pure_chief_complaint_subgroup_binary_outcome_summary.csv",
        index=False,
    )
    summarize_categorical(analysis).to_csv(
        OUTPUT_DIR / "pure_chief_complaint_subgroup_categorical_summary.csv",
        index=False,
    )
    build_pair_membership_summary(analysis).to_csv(
        OUTPUT_DIR / "pure_chief_complaint_subgroup_pair_membership_summary.csv",
        index=False,
    )

    print(f"Saved pure chief-complaint subgroup outcome analysis to: {OUTPUT_DIR}")
    print("\n=== Pure 5x2 subgroup counts ===")
    print(summarize_counts(analysis).to_string(index=False))
    print("\n=== Pure subgroup mortality/death summary ===")
    binary_summary = summarize_binary(analysis)
    if binary_summary.empty:
        print("No binary death indicators were available.")
    else:
        print(binary_summary.to_string(index=False))


if __name__ == "__main__":
    main()
