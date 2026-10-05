"""Describe work-up and stay duration within five complete pure-CC pairs.

Run 01_describe_chief_complaint_subgroups.py first. The primary summaries retain
both members of an original matched pair with valid values for each outcome.
Missing/negative stay durations are excluded per outcome, without converting
missing event counts to zero. Event counts represent recorded database rows.
No inference or clinical free-text export is performed.
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
UTILIZATION_COUNTS_PATH = (
    COHORT_MATCHING_DIR
    / "analysis_output_matched_cohort_characterization_admission_level"
    / "matched_cohort_utilization_counts_by_admission.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_chief_complaint_subgroup_outcomes"

ID_COLUMNS = ["cohort", "subject_id", "hadm_id"]
GROUP_COLUMN = "exclusive_combined_group"
GROUP_LABELS = {
    "abdominal_pain_nausea_vomiting": "Abdominal pain / nausea / vomiting",
    "chest_pain_shortness_of_breath": "Chest pain / shortness of breath",
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


def load_subgroup_assignments() -> pd.DataFrame:
    """Load admission-level subgroup flags and keep exclusive subgroup rows."""
    if not ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            "Missing subgroup assignments. Run "
            "04_clustering_chief_complaints/01_describe_chief_complaint_subgroups.py first: "
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


def load_descriptors() -> pd.DataFrame:
    """Load descriptor rows and derive LOS/death fields used in summaries."""
    descriptors = common.add_derived_descriptor_columns(
        common.validate_id_columns(
            common.load_required_table("descriptors"),
            "descriptors",
        )
    )
    if descriptors.duplicated(ID_COLUMNS).any():
        duplicated = int(descriptors.duplicated(ID_COLUMNS, keep=False).sum())
        raise ValueError(
            f"Descriptor table has {duplicated} duplicated admission ID rows."
        )

    descriptors = descriptors.copy()
    if "admittime" in descriptors.columns and "dischtime" in descriptors.columns:
        descriptors["admittime"] = pd.to_datetime(descriptors["admittime"], errors="coerce")
        descriptors["dischtime"] = pd.to_datetime(descriptors["dischtime"], errors="coerce")
        descriptors["hospital_los_days"] = (
            descriptors["dischtime"] - descriptors["admittime"]
        ).dt.total_seconds() / 86400

    if "edregtime" in descriptors.columns and "edouttime" in descriptors.columns:
        descriptors["edregtime"] = pd.to_datetime(descriptors["edregtime"], errors="coerce")
        descriptors["edouttime"] = pd.to_datetime(descriptors["edouttime"], errors="coerce")
        descriptors["ed_los_hours"] = (
            descriptors["edouttime"] - descriptors["edregtime"]
        ).dt.total_seconds() / 3600

    if "deathtime" in descriptors.columns:
        descriptors["has_deathtime"] = pd.to_datetime(
            descriptors["deathtime"], errors="coerce"
        ).notna()
    if "dod" in descriptors.columns:
        descriptors["has_dod"] = pd.to_datetime(descriptors["dod"], errors="coerce").notna()
    if "hospital_expire_flag" in descriptors.columns:
        descriptors["hospital_expire_flag"] = pd.to_numeric(
            descriptors["hospital_expire_flag"], errors="coerce"
        )
        descriptors["died_in_hospital"] = descriptors["hospital_expire_flag"].eq(1)

    return descriptors


def load_or_build_utilization_counts() -> pd.DataFrame:
    """Load cached admission-level utilization counts, or rebuild them."""
    if UTILIZATION_COUNTS_PATH.exists():
        counts = pd.read_csv(UTILIZATION_COUNTS_PATH)
        return common.validate_id_columns(counts, "utilization_counts")

    matched_ids = common.load_expected_matched_ids()
    event_tables = {
        "labevents": common.load_optional_table("labevents"),
        "microbiologyevents": common.load_optional_table("microbiologyevents"),
        "poe": common.load_optional_table("poe"),
        "poe_detail": common.load_optional_table("poe_detail"),
    }
    return common.build_event_counts_by_admission(matched_ids, event_tables)


def build_analysis_dataset(
    assignments: pd.DataFrame,
    descriptors: pd.DataFrame,
    utilization_counts: pd.DataFrame,
) -> pd.DataFrame:
    """Join subgroup assignments to descriptors and utilization counts."""
    descriptor_columns = [
        column
        for column in [
            *ID_COLUMNS,
            "pair_id",
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
            "race",
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
        suffixes=("", "_descriptor"),
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

    if "age_at_admission" not in analysis.columns and "anchor_age" in descriptors.columns:
        age = descriptors.loc[:, ID_COLUMNS + ["anchor_age"]].rename(
            columns={"anchor_age": "age_at_admission"}
        )
        analysis = analysis.merge(age, on=ID_COLUMNS, how="left", validate="one_to_one")

    keep_columns = [
        column
        for column in [
            "pair_id",
            *ID_COLUMNS,
            GROUP_COLUMN,
            "exclusive_combined_group_label",
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


def summarize_numeric(analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric outcomes/utilization by subgroup and cohort."""
    available_columns = [
        column for column in NUMERIC_SUMMARY_COLUMNS if column in analysis.columns
    ]
    rows = []
    for (group_name, group_label, cohort), group in analysis.groupby(
        [GROUP_COLUMN, "exclusive_combined_group_label", "cohort"],
        dropna=False,
    ):
        for column in available_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            rows.append(
                {
                    GROUP_COLUMN: group_name,
                    "exclusive_combined_group_label": group_label,
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
    """Summarize binary mortality/death indicators by subgroup and cohort."""
    binary_columns = [
        column
        for column in ["died_in_hospital", "has_deathtime", "has_dod"]
        if column in analysis.columns
    ]
    rows = []
    for (group_name, group_label, cohort), group in analysis.groupby(
        [GROUP_COLUMN, "exclusive_combined_group_label", "cohort"],
        dropna=False,
    ):
        for column in binary_columns:
            values = group[column].dropna().astype(bool)
            rows.append(
                {
                    GROUP_COLUMN: group_name,
                    "exclusive_combined_group_label": group_label,
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
    """Summarize categorical descriptors by subgroup and cohort."""
    available_columns = [
        column for column in CATEGORICAL_COLUMNS if column in analysis.columns
    ]
    rows = []
    denominators = (
        analysis.groupby([GROUP_COLUMN, "exclusive_combined_group_label", "cohort"])[
            "hadm_id"
        ]
        .nunique()
        .to_dict()
    )
    for column in available_columns:
        values = analysis.loc[
            :, [GROUP_COLUMN, "exclusive_combined_group_label", "cohort", "hadm_id", column]
        ].copy()
        values[column] = values[column].fillna("missing").astype(str).str.strip()
        values.loc[values[column].eq(""), column] = "missing"
        counts = (
            values.groupby(
                [GROUP_COLUMN, "exclusive_combined_group_label", "cohort", column],
                as_index=False,
            )["hadm_id"]
            .nunique()
            .rename(columns={column: "category", "hadm_id": "n_admissions"})
        )
        counts["measure"] = column
        counts["pct_within_subgroup_cohort"] = counts.apply(
            lambda row: 100.0
            * row["n_admissions"]
            / denominators.get(
                (
                    row[GROUP_COLUMN],
                    row["exclusive_combined_group_label"],
                    row["cohort"],
                ),
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
            "exclusive_combined_group_label",
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


def summarize_counts(analysis: pd.DataFrame) -> pd.DataFrame:
    """Count admissions and subjects in each chief-complaint subgroup/cohort cell."""
    return (
        analysis.groupby([GROUP_COLUMN, "exclusive_combined_group_label", "cohort"])
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
            n_pairs=("pair_id", "nunique"),
        )
        .reset_index()
        .sort_values([GROUP_COLUMN, "cohort"])
    )


def build_pair_membership_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Report how often both admissions from a pair fall into each subgroup."""
    rows = []
    for group_name, group in analysis.groupby(GROUP_COLUMN):
        group_label = GROUP_LABELS.get(group_name, group_name)
        pair_counts = group.groupby("pair_id")["cohort"].nunique()
        rows.append(
            {
                GROUP_COLUMN: group_name,
                "exclusive_combined_group_label": group_label,
                "n_pairs_with_any_group_admission": int(pair_counts.size),
                "n_pairs_with_both_cohorts_in_group": int(pair_counts.eq(2).sum()),
                "pct_pairs_with_both_cohorts_in_group": 100.0
                * pair_counts.eq(2).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(GROUP_COLUMN)



# Current paired pure-subgroup analysis. Existing helper functions remain
# available to the optional all-pure-admission exploration script.
PAIRED_INPUT_DIR = SCRIPT_DIR / "analysis_output_complete_top_five_cc_pairs"
PAIRED_WORKUP_OUTPUT_DIR = SCRIPT_DIR / "analysis_output_complete_cc_pair_workup"
WORKUP_COUNT_COLUMNS = ["n_labevents_rows", "n_microbiologyevents_rows", "n_poe_rows", "n_poe_detail_rows"]
WORKUP_MEASURES = ["hospital_los_days", "ed_los_hours", *WORKUP_COUNT_COLUMNS]

def main():
    selected = pd.read_csv(PAIRED_INPUT_DIR / "complete_cc_pair_admissions.csv")
    descriptors = load_descriptors()
    counts = load_or_build_utilization_counts()
    if descriptors.duplicated(ID_COLUMNS).any() or counts.duplicated(ID_COLUMNS).any():
        raise ValueError("Duplicate descriptor/utilization admission keys")
    for label, table in [("descriptors", descriptors), ("counts", counts)]:
        check = selected[ID_COLUMNS].merge(table[ID_COLUMNS], on=ID_COLUMNS, how="left", indicator=True, validate="one_to_one")
        if not check._merge.eq("both").all():
            raise ValueError(f"Missing retained admission rows in {label}")
    data = selected.merge(descriptors[ID_COLUMNS + ["hospital_los_days", "ed_los_hours"]],
                          on=ID_COLUMNS, how="left", validate="one_to_one")
    data = data.merge(counts[ID_COLUMNS + WORKUP_COUNT_COLUMNS], on=ID_COLUMNS, how="left", validate="one_to_one")
    if not data.groupby("pair_id").size().eq(2).all():
        raise ValueError("Incomplete pair after joins")
    qc = []
    for measure in WORKUP_MEASURES:
        values = pd.to_numeric(data[measure], errors="raise")
        negative = values.lt(0)
        qc.append({"measure": measure, "n_missing_input": int(values.isna().sum()),
                   "n_negative_input": int(negative.sum())})
        if measure in WORKUP_COUNT_COLUMNS and (negative.any() or values.isna().any()):
            raise ValueError(f"Missing/negative recorded counts: {measure}")
        data[measure] = values.mask(negative)
    for count in WORKUP_COUNT_COLUMNS:
        data[count + "_per_hospital_day"] = data[count] / data.hospital_los_days.where(data.hospital_los_days.gt(0))
    measures = [*WORKUP_MEASURES, *(c + "_per_hospital_day" for c in WORKUP_COUNT_COLUMNS)]
    rows = []
    pair_rows = []
    for (group, cohort), frame in data.groupby(["pure_cc_group", "cohort"]):
        for measure in measures:
            group_data = data[data.pure_cc_group.eq(group)]
            valid_counts = group_data[group_data[measure].notna()].groupby("pair_id").size()
            complete_ids = valid_counts[valid_counts.eq(2)].index
            for sample in ["available_admissions", "complete_outcome_pairs"]:
                selected_frame = frame if sample == "available_admissions" else frame[frame.pair_id.isin(complete_ids)]
                values = selected_frame[measure].dropna()
                rows.append({"cc_group": group, "cohort": cohort, "measure": measure, "analysis_sample": sample,
                             "n_admissions": len(selected_frame), "n_nonmissing": len(values), "n_missing": selected_frame[measure].isna().sum(),
                             "mean": values.mean(), "sd": values.std(), "median": values.median(),
                             "q1": values.quantile(.25), "q3": values.quantile(.75),
                             "min": values.min(), "max": values.max(),
                             "n_zero": int(values.eq(0).sum()), "pct_zero": 100*values.eq(0).mean()})
    for group, frame in data.groupby("pure_cc_group"):
        for measure in measures:
            paired = frame.pivot(index="pair_id", columns="cohort", values=measure).dropna()
            difference = paired.MHC1_psychotic - paired.MHC0
            pair_rows.append({"cc_group": group, "measure": measure, "n_complete_outcome_pairs": len(paired),
                              "mean_paired_difference": difference.mean(), "median_paired_difference": difference.median(),
                              "q1_paired_difference": difference.quantile(.25), "q3_paired_difference": difference.quantile(.75)})
    PAIRED_WORKUP_OUTPUT_DIR.mkdir(exist_ok=True)
    data.to_csv(PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_dataset.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_paired_differences.csv", index=False)
    pd.DataFrame(qc).to_csv(PAIRED_WORKUP_OUTPUT_DIR / "complete_cc_pair_workup_qc.csv", index=False)
    print(summary[summary.measure.isin(WORKUP_MEASURES) & summary.analysis_sample.eq("complete_outcome_pairs")][["cc_group", "cohort", "measure", "n_nonmissing", "n_missing", "mean", "median", "q1", "q3", "pct_zero"]].to_string(index=False))
    print("\nInput quality checks:")
    print(pd.DataFrame(qc).to_string(index=False))



if __name__ == "__main__":
    main()
