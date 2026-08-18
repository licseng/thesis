"""Analyze mortality risk after matched admission.

The main analysis is admission-level:

1. Use all matched admissions.
2. Describe crude mortality percentages by cohort.
3. Fit age/Elixhauser-adjusted logistic models using all admissions and
   cluster-robust standard errors by matched pair.

The sensitivity analysis uses a subject-level matched-pair design:

1. For each unique MHH1_psychotic subject, keep that subject's earliest matched
   admission.
2. Keep the matched MHC0 admission from the same pair.
3. If an MHC0 subject appears in multiple selected pairs, keep only the earliest
   MHC0 admission and drop the other complete pairs.
4. Compare mortality outcomes between the two cohorts overall.
5. Repeat the comparison within the five pure chief-complaint subgroups, keeping
   only pairs where both admissions fall into the same pure subgroup.

Mortality is measured from the selected matched admission. The main one-year
outcome is based on `dod`, not just in-hospital death.

Outputs are aggregate summaries plus an ID-level audit table. No raw chief
complaint text or discharge-note text is written.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
COHORT_MATCHING_DIR = PROJECT_DIR / "02_cohort_matching"
CLUSTERING_DIR = PROJECT_DIR / "04_clustering_chief_complaints"
sys.path.insert(0, str(COHORT_MATCHING_DIR))

import _matched_cohort_characterization_common as common  # noqa: E402


ASSIGNMENT_PATH = (
    CLUSTERING_DIR
    / "analysis_output_chief_complaint_subgroup_balance_check"
    / "chief_complaint_subgroup_admission_assignments.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_subject_level_mortality_risk"

ID_COLUMNS = ["cohort", "subject_id", "hadm_id"]
MHH1_COHORT = "MHH1_psychotic"
MHC0_COHORT = "MHC0"
ONE_YEAR_DAYS = 365

PURE_SUBGROUPS = {
    "abdominal pain": "Abdominal pain",
    "shortness of breath": "Shortness of breath",
    "chest pain": "Chest pain",
    "altered mental status": "Altered mental status",
    "nausea vomiting": "Nausea / vomiting",
}
COMBINED_SUBGROUPS = {
    "abdominal_pain_nausea_vomiting": "Abdominal pain / nausea / vomiting",
    "chest_pain_shortness_of_breath": "Chest pain / shortness of breath",
}
SUBGROUP_FLAG_COLUMNS = {
    "abdominal pain": "has_abdominal_pain",
    "shortness of breath": "has_shortness_of_breath",
    "chest pain": "has_chest_pain",
    "altered mental status": "has_altered_mental_status",
    "nausea vomiting": "has_nausea_vomiting",
}
OUTCOME_COLUMNS = [
    "died_in_hospital",
    "death_within_1y_after_admission",
    "post_discharge_death_within_1y",
]
COVARIATE_COLUMNS = ["age_at_admission", "anchor_age", "elixhauser_score"]


def load_descriptors() -> pd.DataFrame:
    """Load matched admission descriptors and normalize date/death columns."""
    descriptors = common.add_derived_descriptor_columns(
        common.validate_id_columns(
            common.load_required_table("descriptors"),
            "descriptors",
        )
    )
    if descriptors.duplicated(ID_COLUMNS).any():
        duplicated = int(descriptors.duplicated(ID_COLUMNS, keep=False).sum())
        raise ValueError(f"Descriptor table has {duplicated} duplicated admission rows.")
    if "pair_id" not in descriptors.columns:
        raise ValueError("Descriptor table is missing pair_id.")

    descriptors = descriptors.copy()
    descriptors["pair_id"] = pd.to_numeric(
        descriptors["pair_id"], errors="raise"
    ).astype(int)
    for column in ["admittime", "dischtime", "deathtime", "dod"]:
        if column in descriptors.columns:
            descriptors[column] = pd.to_datetime(descriptors[column], errors="coerce")
        else:
            descriptors[column] = pd.NaT

    descriptors["hospital_expire_flag"] = pd.to_numeric(
        descriptors.get("hospital_expire_flag", 0), errors="coerce"
    ).fillna(0)
    descriptors["died_in_hospital"] = descriptors["hospital_expire_flag"].eq(1)
    descriptors["days_admission_to_death"] = (
        descriptors["dod"] - descriptors["admittime"]
    ).dt.total_seconds() / 86400
    descriptors["days_discharge_to_death"] = (
        descriptors["dod"] - descriptors["dischtime"]
    ).dt.total_seconds() / 86400
    descriptors.loc[
        descriptors["dod"].isna() | descriptors["dischtime"].isna(),
        "days_discharge_to_death",
    ] = pd.NA

    one_year_after_admit = descriptors["admittime"] + pd.Timedelta(days=ONE_YEAR_DAYS)
    one_year_after_discharge = descriptors["dischtime"] + pd.Timedelta(days=ONE_YEAR_DAYS)
    descriptors["death_within_1y_after_admission"] = (
        descriptors["dod"].notna()
        & descriptors["admittime"].notna()
        & descriptors["dod"].ge(descriptors["admittime"])
        & descriptors["dod"].le(one_year_after_admit)
    )
    descriptors["post_discharge_death_within_1y"] = (
        descriptors["dod"].notna()
        & descriptors["dischtime"].notna()
        & descriptors["dod"].gt(descriptors["dischtime"])
        & descriptors["dod"].le(one_year_after_discharge)
        & ~descriptors["died_in_hospital"]
    )
    return descriptors


def select_first_mhh1_subject_pairs(descriptors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep earliest matched MHH1 admission per MHH1 subject and its matched control."""
    mhh1 = descriptors.loc[descriptors["cohort"].eq(MHH1_COHORT)].copy()
    if mhh1.empty:
        raise ValueError(f"No {MHH1_COHORT} rows found in descriptor table.")

    selected_mhh1 = (
        mhh1.sort_values(["subject_id", "admittime", "pair_id", "hadm_id"])
        .drop_duplicates("subject_id", keep="first")
        .copy()
    )
    selected_pair_ids = set(selected_mhh1["pair_id"])
    selected = descriptors.loc[descriptors["pair_id"].isin(selected_pair_ids)].copy()

    pair_cohort_counts = (
        selected.groupby(["pair_id", "cohort"])["hadm_id"].nunique().unstack(fill_value=0)
    )
    complete_pair_ids = pair_cohort_counts.loc[
        pair_cohort_counts.get(MHH1_COHORT, 0).eq(1)
        & pair_cohort_counts.get(MHC0_COHORT, 0).eq(1)
    ].index
    incomplete_pairs = sorted(selected_pair_ids - set(complete_pair_ids))
    if incomplete_pairs:
        selected = selected.loc[selected["pair_id"].isin(complete_pair_ids)].copy()

    selection_summary = pd.DataFrame(
        [
            {
                "metric": "original_descriptor_rows",
                "value": len(descriptors),
            },
            {
                "metric": "original_pairs",
                "value": descriptors["pair_id"].nunique(),
            },
            {
                "metric": "original_mhh1_admissions",
                "value": len(mhh1),
            },
            {
                "metric": "unique_mhh1_subjects",
                "value": mhh1["subject_id"].nunique(),
            },
            {
                "metric": "selected_complete_pairs",
                "value": selected["pair_id"].nunique(),
            },
            {
                "metric": "selected_rows",
                "value": len(selected),
            },
            {
                "metric": "dropped_incomplete_selected_pairs",
                "value": len(incomplete_pairs),
            },
        ]
    )
    return selected.sort_values(["pair_id", "cohort"]), selection_summary


def restrict_to_unique_mhc0_subjects(
    selected: pd.DataFrame,
    selection_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop complete pairs until each MHC0 subject appears only once.

    The keep rule is deterministic and outcome-agnostic: among repeated MHC0
    subjects, keep the pair with the earliest MHC0 admission time.
    """
    mhc0 = selected.loc[selected["cohort"].eq(MHC0_COHORT)].copy()
    repeated_counts = mhc0.groupby("subject_id")["hadm_id"].nunique()
    repeated_counts = repeated_counts.loc[repeated_counts.gt(1)]

    kept_mhc0 = (
        mhc0.sort_values(["subject_id", "admittime", "pair_id", "hadm_id"])
        .drop_duplicates("subject_id", keep="first")
        .copy()
    )
    kept_pair_ids = set(kept_mhc0["pair_id"])
    dropped_pair_ids = sorted(set(selected["pair_id"]) - kept_pair_ids)
    restricted = selected.loc[selected["pair_id"].isin(kept_pair_ids)].copy()

    dropped_pairs = selected.loc[selected["pair_id"].isin(dropped_pair_ids)].copy()
    extra_rows = pd.DataFrame(
        [
            {
                "metric": "selected_pairs_before_unique_mhc0_restriction",
                "value": selected["pair_id"].nunique(),
            },
            {
                "metric": "repeated_mhc0_subjects_before_restriction",
                "value": len(repeated_counts),
            },
            {
                "metric": "extra_mhc0_admissions_before_restriction",
                "value": int((repeated_counts - 1).sum()),
            },
            {
                "metric": "dropped_pairs_for_repeated_mhc0_subjects",
                "value": len(dropped_pair_ids),
            },
            {
                "metric": "selected_pairs_after_unique_mhc0_restriction",
                "value": restricted["pair_id"].nunique(),
            },
            {
                "metric": "selected_rows_after_unique_mhc0_restriction",
                "value": len(restricted),
            },
            {
                "metric": "unique_mhh1_subjects_after_unique_mhc0_restriction",
                "value": restricted.loc[
                    restricted["cohort"].eq(MHH1_COHORT), "subject_id"
                ].nunique(),
            },
            {
                "metric": "unique_mhc0_subjects_after_unique_mhc0_restriction",
                "value": restricted.loc[
                    restricted["cohort"].eq(MHC0_COHORT), "subject_id"
                ].nunique(),
            },
        ]
    )
    selection_summary = pd.concat([selection_summary, extra_rows], ignore_index=True)
    return (
        restricted.sort_values(["pair_id", "cohort"]),
        selection_summary,
        dropped_pairs.sort_values(["pair_id", "cohort"]),
    )


def load_pure_subgroup_assignments() -> pd.DataFrame:
    """Load pure chief-complaint subgroup assignments without raw text."""
    if not ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            "Missing subgroup assignments. Run "
            "04_clustering_chief_complaints/01_describe_chief_complaint_subgroups.py first: "
            f"{ASSIGNMENT_PATH}"
        )

    assignments = pd.read_csv(ASSIGNMENT_PATH)
    required = {
        "pair_id",
        *ID_COLUMNS,
        "n_chief_complaint_subgroups_matched",
        *SUBGROUP_FLAG_COLUMNS.values(),
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Subgroup assignment file is missing columns: {missing}")

    assignments = assignments.copy()
    assignments["cohort"] = assignments["cohort"].astype("string").str.strip()
    assignments["subject_id"] = pd.to_numeric(
        assignments["subject_id"], errors="raise"
    ).astype(int)
    assignments["hadm_id"] = pd.to_numeric(assignments["hadm_id"], errors="raise").astype(int)
    assignments["pair_id"] = pd.to_numeric(assignments["pair_id"], errors="raise").astype(int)
    assignments["n_chief_complaint_subgroups_matched"] = pd.to_numeric(
        assignments["n_chief_complaint_subgroups_matched"], errors="raise"
    ).astype(int)

    pure = assignments.loc[
        assignments["n_chief_complaint_subgroups_matched"].eq(1)
    ].copy()
    pure["pure_chief_complaint_group"] = pd.NA
    for subgroup, flag_column in SUBGROUP_FLAG_COLUMNS.items():
        pure.loc[pure[flag_column].astype(bool), "pure_chief_complaint_group"] = subgroup

    pure = pure.loc[pure["pure_chief_complaint_group"].isin(PURE_SUBGROUPS)].copy()
    pure["pure_chief_complaint_group_label"] = pure["pure_chief_complaint_group"].map(
        PURE_SUBGROUPS
    )
    return pure.loc[
        :,
        [
            "pair_id",
            *ID_COLUMNS,
            "pure_chief_complaint_group",
            "pure_chief_complaint_group_label",
        ],
    ]


def load_combined_subgroup_assignments() -> pd.DataFrame:
    """Load the two hard-coded combined chief-complaint subgroup assignments."""
    if not ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            "Missing subgroup assignments. Run "
            "04_clustering_chief_complaints/01_describe_chief_complaint_subgroups.py first: "
            f"{ASSIGNMENT_PATH}"
        )

    assignments = pd.read_csv(
        ASSIGNMENT_PATH,
        usecols=["pair_id", *ID_COLUMNS, "exclusive_combined_group"],
    )
    missing = sorted(
        set(["pair_id", *ID_COLUMNS, "exclusive_combined_group"]) - set(assignments.columns)
    )
    if missing:
        raise ValueError(f"Subgroup assignment file is missing columns: {missing}")

    assignments = assignments.copy()
    assignments["cohort"] = assignments["cohort"].astype("string").str.strip()
    assignments["subject_id"] = pd.to_numeric(
        assignments["subject_id"], errors="raise"
    ).astype(int)
    assignments["hadm_id"] = pd.to_numeric(assignments["hadm_id"], errors="raise").astype(int)
    assignments["pair_id"] = pd.to_numeric(assignments["pair_id"], errors="raise").astype(int)
    assignments["combined_chief_complaint_group"] = (
        assignments["exclusive_combined_group"].astype("string").str.strip()
    )
    assignments = assignments.loc[
        assignments["combined_chief_complaint_group"].isin(COMBINED_SUBGROUPS)
    ].copy()
    assignments["combined_chief_complaint_group_label"] = assignments[
        "combined_chief_complaint_group"
    ].map(COMBINED_SUBGROUPS)
    return assignments.loc[
        :,
        [
            "pair_id",
            *ID_COLUMNS,
            "combined_chief_complaint_group",
            "combined_chief_complaint_group_label",
        ],
    ]


def load_matching_covariates() -> pd.DataFrame:
    """Load age/Elixhauser matching covariates for the selected admissions."""
    if not ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            "Missing subgroup assignments with matching covariates: "
            f"{ASSIGNMENT_PATH}"
        )

    covariates = pd.read_csv(
        ASSIGNMENT_PATH,
        usecols=[
            "cohort",
            "subject_id",
            "hadm_id",
            "age_at_admission",
            "elixhauser_score",
        ],
    )
    covariates = common.validate_id_columns(covariates, "matching_covariates")
    covariates["age_at_admission"] = pd.to_numeric(
        covariates["age_at_admission"], errors="coerce"
    )
    covariates["elixhauser_score"] = pd.to_numeric(
        covariates["elixhauser_score"], errors="coerce"
    )
    return covariates


def add_matching_covariates(selected: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """Attach matching covariates to selected admissions."""
    output = selected.drop(
        columns=[column for column in ["age_at_admission", "elixhauser_score"] if column in selected],
        errors="ignore",
    )
    return output.merge(
        covariates,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )


def add_pure_subgroup_columns(selected: pd.DataFrame, pure: pd.DataFrame) -> pd.DataFrame:
    """Attach pure subgroup labels to selected admissions."""
    return selected.merge(
        pure,
        on=["pair_id", *ID_COLUMNS],
        how="left",
        validate="one_to_one",
    )


def add_combined_subgroup_columns(selected: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    """Attach hard-coded combined subgroup labels to selected admissions."""
    return selected.merge(
        combined,
        on=["pair_id", *ID_COLUMNS],
        how="left",
        validate="one_to_one",
    )


def summarize_outcomes(analysis: pd.DataFrame, strata_columns: list[str] | None = None) -> pd.DataFrame:
    """Summarize mortality outcomes by cohort, optionally within strata."""
    strata_columns = strata_columns or []
    rows = []
    group_columns = [*strata_columns, "cohort"]
    groupby_keys = group_columns[0] if len(group_columns) == 1 else group_columns
    grouped = analysis.groupby(groupby_keys, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row_base = dict(zip(group_columns, keys, strict=True))
        for outcome in OUTCOME_COLUMNS:
            values = group[outcome].dropna().astype(bool)
            rows.append(
                {
                    **row_base,
                    "outcome": outcome,
                    "n_admissions": int(group["hadm_id"].nunique()),
                    "n_subjects": int(group["subject_id"].nunique()),
                    "n_pairs": int(group["pair_id"].nunique()),
                    "n_positive": int(values.sum()),
                    "pct_positive": 100.0 * values.mean() if len(values) else pd.NA,
                }
            )
    return pd.DataFrame(rows).sort_values([*strata_columns, "outcome", "cohort"])


def compare_cohorts(summary: pd.DataFrame, strata_columns: list[str] | None = None) -> pd.DataFrame:
    """Compute MHH1-MHC0 risk difference and risk ratio from summary rows."""
    strata_columns = strata_columns or []
    index_columns = [*strata_columns, "outcome"]
    cohort_values = summary.pivot_table(
        index=index_columns,
        columns="cohort",
        values=["n_admissions", "n_subjects", "n_pairs", "n_positive", "pct_positive"],
        aggfunc="first",
    )
    cohort_values.columns = [
        f"{measure}_{cohort}" for measure, cohort in cohort_values.columns.to_flat_index()
    ]
    comparison = cohort_values.reset_index()
    comparison["risk_difference_pct_points_mhh1_minus_mhc0"] = (
        comparison.get(f"pct_positive_{MHH1_COHORT}", pd.NA)
        - comparison.get(f"pct_positive_{MHC0_COHORT}", pd.NA)
    )
    comparison["risk_ratio_mhh1_vs_mhc0"] = (
        comparison.get(f"pct_positive_{MHH1_COHORT}", pd.NA)
        / comparison.get(f"pct_positive_{MHC0_COHORT}", pd.NA)
    )
    comparison.loc[
        comparison.get(f"pct_positive_{MHC0_COHORT}", pd.Series(dtype=float)).eq(0),
        "risk_ratio_mhh1_vs_mhc0",
    ] = pd.NA
    return comparison.sort_values(index_columns)


def summarize_time_to_death(
    analysis: pd.DataFrame,
    strata_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize days to death among selected admissions with deaths recorded."""
    strata_columns = strata_columns or []
    rows = []
    group_columns = [*strata_columns, "cohort"]
    groupby_keys = group_columns[0] if len(group_columns) == 1 else group_columns
    for keys, group in analysis.groupby(groupby_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row_base = dict(zip(group_columns, keys, strict=True))
        for column in ["days_admission_to_death", "days_discharge_to_death"]:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            values = values.loc[values.ge(0)]
            rows.append(
                {
                    **row_base,
                    "measure": column,
                    "n_deaths_with_nonnegative_time": len(values),
                    "mean_days": values.mean() if len(values) else pd.NA,
                    "median_days": values.median() if len(values) else pd.NA,
                    "q1_days": values.quantile(0.25) if len(values) else pd.NA,
                    "q3_days": values.quantile(0.75) if len(values) else pd.NA,
                    "min_days": values.min() if len(values) else pd.NA,
                    "max_days": values.max() if len(values) else pd.NA,
                }
            )
    return pd.DataFrame(rows).sort_values([*strata_columns, "measure", "cohort"])


def summarize_covariates(analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize age and Elixhauser balance in the mortality analysis set."""
    rows = []
    for cohort, group in analysis.groupby("cohort", dropna=False):
        for column in COVARIATE_COLUMNS:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            rows.append(
                {
                    "measure": column,
                    "cohort": cohort,
                    "n": len(values),
                    "mean": values.mean() if len(values) else pd.NA,
                    "sd": values.std(ddof=1) if len(values) > 1 else pd.NA,
                    "median": values.median() if len(values) else pd.NA,
                    "q1": values.quantile(0.25) if len(values) else pd.NA,
                    "q3": values.quantile(0.75) if len(values) else pd.NA,
                    "min": values.min() if len(values) else pd.NA,
                    "max": values.max() if len(values) else pd.NA,
                }
            )
    return pd.DataFrame(rows).sort_values(["measure", "cohort"])


def standardized_mean_difference(mhh1_values: pd.Series, mhc0_values: pd.Series) -> float | pd.NA:
    """Return MHH1-MHC0 standardized mean difference for one numeric covariate."""
    mhh1 = pd.to_numeric(mhh1_values, errors="coerce").dropna()
    mhc0 = pd.to_numeric(mhc0_values, errors="coerce").dropna()
    if len(mhh1) < 2 or len(mhc0) < 2:
        return pd.NA
    pooled_sd = ((mhh1.var(ddof=1) + mhc0.var(ddof=1)) / 2) ** 0.5
    if pooled_sd == 0:
        return pd.NA
    return (mhh1.mean() - mhc0.mean()) / pooled_sd


def summarize_covariate_smds(analysis: pd.DataFrame) -> pd.DataFrame:
    """Compute MHH1-MHC0 SMDs for numeric matching covariates."""
    rows = []
    mhh1 = analysis.loc[analysis["cohort"].eq(MHH1_COHORT)]
    mhc0 = analysis.loc[analysis["cohort"].eq(MHC0_COHORT)]
    for column in COVARIATE_COLUMNS:
        if column not in analysis.columns:
            continue
        rows.append(
            {
                "measure": column,
                "smd_mhh1_minus_mhc0": standardized_mean_difference(
                    mhh1[column],
                    mhc0[column],
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_covariate_bins(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count admissions by age and Elixhauser bins used for mortality QC."""
    age_counts = pd.DataFrame()
    if "age_at_admission" in analysis.columns:
        ages = pd.to_numeric(analysis["age_at_admission"], errors="coerce")
        age_bins = pd.cut(
            ages,
            bins=[18, 30, 40, 50, 60, 70, 80, 200],
            labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"],
            right=False,
        )
        age_counts = pd.crosstab(age_bins, analysis["cohort"]).reset_index()
        age_counts = age_counts.rename(columns={"age_at_admission": "age_bin"})

    elixhauser_counts = pd.DataFrame()
    if "elixhauser_score" in analysis.columns:
        scores = pd.to_numeric(analysis["elixhauser_score"], errors="coerce")
        elixhauser_bins = pd.cut(
            scores,
            bins=[-999, 0, 3, 6, 10, 999],
            labels=["<0", "0-2", "3-5", "6-9", "10+"],
            right=False,
        )
        elixhauser_counts = pd.crosstab(elixhauser_bins, analysis["cohort"]).reset_index()
        elixhauser_counts = elixhauser_counts.rename(
            columns={"elixhauser_score": "elixhauser_bin"}
        )
    return age_counts, elixhauser_counts


def prior_all_mimic_admission_group(n_prior: object) -> str:
    """Bucket real prior MIMIC admissions before the matched admission."""
    if pd.isna(n_prior):
        return "missing"
    n_prior = int(n_prior)
    if n_prior == 0:
        return "0_prior"
    if n_prior <= 2:
        return "1_2_prior"
    if n_prior <= 5:
        return "3_5_prior"
    return "6plus_prior"


def add_real_readmission_columns(analysis: pd.DataFrame) -> pd.DataFrame:
    """Add recent and full-MIMIC prior-admission variables for each admission."""
    required_columns = {
        "n_prior_all_admissions_for_subject",
        "n_prior_admissions_within_365d_for_subject",
    }
    missing = sorted(required_columns - set(analysis.columns))
    if missing:
        raise ValueError(
            "Missing prior-admission columns. Rerun the matched-cohort "
            "additional-info SQL export before mortality readmission adjustment. "
            f"Missing: {missing}"
        )
    output = analysis.copy()
    output["prior_all_mimic_admissions"] = pd.to_numeric(
        output["n_prior_all_admissions_for_subject"],
        errors="raise",
    )
    output["prior_admissions_365d"] = pd.to_numeric(
        output["n_prior_admissions_within_365d_for_subject"],
        errors="raise",
    )
    output["log1p_prior_all_mimic_admissions"] = np.log1p(
        output["prior_all_mimic_admissions"],
    )
    output["log1p_prior_admissions_365d"] = np.log1p(
        output["prior_admissions_365d"],
    )
    output["prior_all_mimic_admission_group"] = output[
        "prior_all_mimic_admissions"
    ].map(prior_all_mimic_admission_group)
    output["prior_all_mimic_1_2"] = output[
        "prior_all_mimic_admission_group"
    ].eq("1_2_prior").astype(int)
    output["prior_all_mimic_3_5"] = output[
        "prior_all_mimic_admission_group"
    ].eq("3_5_prior").astype(int)
    output["prior_all_mimic_6plus"] = output[
        "prior_all_mimic_admission_group"
    ].eq("6plus_prior").astype(int)
    return output.sort_values(["pair_id", "cohort"]).reset_index(drop=True)


def summarize_real_readmission_distribution(analysis: pd.DataFrame) -> pd.DataFrame:
    """Count admissions by full-MIMIC prior admission group and cohort."""
    denominators = analysis.groupby("cohort")["hadm_id"].nunique().to_dict()
    distribution = (
        analysis.groupby(["cohort", "prior_all_mimic_admission_group"], as_index=False)
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
        )
        .sort_values(["cohort", "prior_all_mimic_admission_group"])
    )
    distribution["pct_admissions_within_cohort"] = distribution.apply(
        lambda row: 100.0
        * row["n_admissions"]
        / denominators.get(row["cohort"], 0),
        axis=1,
    )
    return distribution


def summarize_recent_readmission_distribution(analysis: pd.DataFrame) -> pd.DataFrame:
    """Count admissions by number of prior admissions in the previous year."""
    denominators = analysis.groupby("cohort")["hadm_id"].nunique().to_dict()
    distribution = (
        analysis.groupby(["cohort", "prior_admissions_365d"], as_index=False)
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
        )
        .sort_values(["cohort", "prior_admissions_365d"])
    )
    distribution["pct_admissions_within_cohort"] = distribution.apply(
        lambda row: 100.0
        * row["n_admissions"]
        / denominators.get(row["cohort"], 0),
        axis=1,
    )
    return distribution


def fit_clustered_logistic_model(
    analysis: pd.DataFrame,
    outcome: str,
    model_name: str,
    stratum: str = "overall",
    extra_predictor_columns: list[str] | None = None,
    cluster_column: str = "pair_id",
) -> pd.DataFrame:
    """Fit logistic regression with cluster-robust SEs.

    Predictors are:
        intercept + MHH1 indicator + age at admission per 10y + Elixhauser per 5 points
        + optional extra predictors
    """
    extra_predictor_columns = extra_predictor_columns or []
    required_columns = [
        "cohort",
        "subject_id",
        cluster_column,
        outcome,
        "age_at_admission",
        "elixhauser_score",
        *extra_predictor_columns,
    ]
    model_data = analysis.loc[
        :,
        required_columns,
    ].copy()
    model_data[outcome] = model_data[outcome].astype(bool).astype(int)
    model_data["mhh1_psychotic"] = model_data["cohort"].eq(MHH1_COHORT).astype(float)
    model_data["age_at_admission_per_10y"] = (
        pd.to_numeric(model_data["age_at_admission"], errors="coerce") / 10.0
    )
    model_data["elixhauser_score_per_5pt"] = (
        pd.to_numeric(model_data["elixhauser_score"], errors="coerce") / 5.0
    )
    model_data["cluster_id"] = model_data[cluster_column].astype(str)
    model_data = model_data.dropna(
        subset=[
            outcome,
            "mhh1_psychotic",
            "age_at_admission_per_10y",
            "elixhauser_score_per_5pt",
            "cluster_id",
            *extra_predictor_columns,
        ]
    ).reset_index(drop=True)

    n_rows = len(model_data)
    n_events = int(model_data[outcome].sum())
    n_clusters = model_data["cluster_id"].nunique()
    coefficient_names = [
        "intercept",
        "mhh1_psychotic",
        "age_at_admission_per_10y",
        "elixhauser_score_per_5pt",
        *extra_predictor_columns,
    ]
    if n_rows == 0 or n_events == 0 or n_events == n_rows:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "stratum": stratum,
                    "outcome": outcome,
                    "term": term,
                    "n_admissions": n_rows,
                    "n_events": n_events,
                    "n_clusters": n_clusters,
                    "cluster_column": cluster_column,
                    "status": "not_fit_outcome_has_one_class",
                    "fit_method": "statsmodels_glm_binomial_cluster_robust",
                    "fit_converged": False,
                    "fit_message": "not_fit_outcome_has_one_class",
                    "fit_warnings": "",
                    "fit_iterations": pd.NA,
                }
                for term in coefficient_names
            ]
        )

    predictor_columns = [
        "mhh1_psychotic",
        "age_at_admission_per_10y",
        "elixhauser_score_per_5pt",
        *extra_predictor_columns,
    ]
    x = sm.add_constant(
        model_data.loc[:, predictor_columns].astype(float),
        has_constant="add",
    )
    x = x.rename(columns={"const": "intercept"})
    y = model_data[outcome].astype(float)
    coefficient_names = x.columns.tolist()

    captured_warnings = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": model_data["cluster_id"],
                    "use_correction": True,
                },
                maxiter=100,
            )
            captured_warnings = [
                f"{warning.category.__name__}: {warning.message}"
                for warning in caught
            ]
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "stratum": stratum,
                    "outcome": outcome,
                    "term": term,
                    "n_admissions": n_rows,
                    "n_events": n_events,
                    "n_clusters": n_clusters,
                    "cluster_column": cluster_column,
                    "status": "fit_failed",
                    "fit_method": "statsmodels_glm_binomial_cluster_robust",
                    "fit_converged": False,
                    "fit_message": repr(exc),
                    "fit_warnings": "",
                    "fit_iterations": pd.NA,
                }
                for term in coefficient_names
            ]
        )

    params = fit.params
    standard_errors = fit.bse
    conf_int = fit.conf_int(alpha=0.05)
    fit_history = getattr(fit, "fit_history", {}) or {}
    fit_converged = bool(getattr(fit, "converged", False))
    fit_warnings = " | ".join(captured_warnings)
    fit_message = "converged" if fit_converged else "not_converged"
    fit_iterations = fit_history.get("iteration", pd.NA)

    rows = []
    for term in coefficient_names:
        estimate = params.get(term, pd.NA)
        se = standard_errors.get(term, pd.NA)
        z_value = fit.tvalues.get(term, pd.NA)
        p_value = fit.pvalues.get(term, pd.NA)
        ci_low = conf_int.loc[term, 0] if term in conf_int.index else pd.NA
        ci_high = conf_int.loc[term, 1] if term in conf_int.index else pd.NA
        rows.append(
            {
                "model": model_name,
                "stratum": stratum,
                "outcome": outcome,
                "term": term,
                "n_admissions": n_rows,
                "n_events": n_events,
                "n_clusters": n_clusters,
                "cluster_column": cluster_column,
                "estimate_log_odds": estimate,
                "cluster_robust_se": se,
                "z": z_value,
                "p_value": p_value,
                "odds_ratio": math.exp(estimate),
                "odds_ratio_ci_low": math.exp(ci_low) if pd.notna(ci_low) else pd.NA,
                "odds_ratio_ci_high": math.exp(ci_high) if pd.notna(ci_high) else pd.NA,
                "status": "fit_converged" if fit_converged and not fit_warnings else "fit_warning",
                "fit_method": "statsmodels_glm_binomial_cluster_robust",
                "fit_converged": fit_converged,
                "fit_message": fit_message,
                "fit_warnings": fit_warnings,
                "fit_iterations": fit_iterations,
            }
        )
    return pd.DataFrame(rows)


def fit_admission_level_models(analysis: pd.DataFrame) -> pd.DataFrame:
    """Fit adjusted mortality models overall and within chief-complaint subgroups."""
    model_rows = []
    for outcome in OUTCOME_COLUMNS:
        model_rows.append(
            fit_clustered_logistic_model(
                analysis,
                outcome,
                "admission_level_age_elixhauser_adjusted",
                "overall",
            )
        )

    pure = analysis.dropna(subset=["pure_chief_complaint_group"]).copy()
    for (group_name, group_label), group in pure.groupby(
        ["pure_chief_complaint_group", "pure_chief_complaint_group_label"],
        dropna=False,
    ):
        for outcome in OUTCOME_COLUMNS:
            model_rows.append(
                fit_clustered_logistic_model(
                    group,
                    outcome,
                    "admission_level_age_elixhauser_adjusted_pure_subgroup",
                    f"{group_name}: {group_label}",
                )
            )
    return pd.concat(model_rows, ignore_index=True)


def fit_admission_level_readmission_adjusted_models(analysis: pd.DataFrame) -> pd.DataFrame:
    """Fit mortality models with recent and full-MIMIC prior-admission adjustments."""
    model_rows = []
    readmission_specs = [
        (
            "admission_level_age_elixhauser_prior365_adjusted",
            ["log1p_prior_admissions_365d"],
        ),
        (
            "admission_level_age_elixhauser_prior_all_mimic_adjusted",
            ["log1p_prior_all_mimic_admissions"],
        ),
    ]
    for model_name, extra_predictors in readmission_specs:
        for outcome in OUTCOME_COLUMNS:
            model_rows.append(
                fit_clustered_logistic_model(
                    analysis,
                    outcome,
                    model_name,
                    "overall",
                    extra_predictors,
                )
            )

        pure = analysis.dropna(subset=["pure_chief_complaint_group"]).copy()
        for (group_name, group_label), group in pure.groupby(
            ["pure_chief_complaint_group", "pure_chief_complaint_group_label"],
            dropna=False,
        ):
            for outcome in OUTCOME_COLUMNS:
                model_rows.append(
                    fit_clustered_logistic_model(
                        group,
                        outcome,
                        f"{model_name}_pure_subgroup",
                        f"{group_name}: {group_label}",
                        extra_predictors,
                    )
                )
    return pd.concat(model_rows, ignore_index=True)


def fit_admission_level_combined_models(analysis: pd.DataFrame) -> pd.DataFrame:
    """Fit adjusted mortality models within the two hard-coded subgroups."""
    model_rows = []
    combined = analysis.dropna(subset=["combined_chief_complaint_group"]).copy()
    for (group_name, group_label), group in combined.groupby(
        ["combined_chief_complaint_group", "combined_chief_complaint_group_label"],
        dropna=False,
    ):
        for outcome in OUTCOME_COLUMNS:
            model_rows.append(
                fit_clustered_logistic_model(
                    group,
                    outcome,
                    "admission_level_age_elixhauser_adjusted_combined_subgroup",
                    f"{group_name}: {group_label}",
                )
            )
    if not model_rows:
        return pd.DataFrame()
    return pd.concat(model_rows, ignore_index=True)


def fit_admission_level_combined_readmission_adjusted_models(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Fit combined-subgroup models with recent/full-MIMIC prior admissions."""
    model_rows = []
    readmission_specs = [
        (
            "admission_level_age_elixhauser_prior365_adjusted_combined_subgroup",
            ["log1p_prior_admissions_365d"],
        ),
        (
            "admission_level_age_elixhauser_prior_all_mimic_adjusted_combined_subgroup",
            ["log1p_prior_all_mimic_admissions"],
        ),
    ]
    combined = analysis.dropna(subset=["combined_chief_complaint_group"]).copy()
    for model_name, extra_predictors in readmission_specs:
        for (group_name, group_label), group in combined.groupby(
            ["combined_chief_complaint_group", "combined_chief_complaint_group_label"],
            dropna=False,
        ):
            for outcome in OUTCOME_COLUMNS:
                model_rows.append(
                    fit_clustered_logistic_model(
                        group,
                        outcome,
                        model_name,
                        f"{group_name}: {group_label}",
                        extra_predictors,
                    )
                )
    if not model_rows:
        return pd.DataFrame()
    return pd.concat(model_rows, ignore_index=True)


def build_cohort_term_model_comparison(
    age_elixhauser_models: pd.DataFrame,
    readmission_adjusted_models: pd.DataFrame,
    combined_age_elixhauser_models: pd.DataFrame,
    combined_readmission_adjusted_models: pd.DataFrame,
) -> pd.DataFrame:
    """Put cohort effect estimates before/after readmission adjustment side by side."""
    model_sets = [
        ("pure_or_overall", "age_elixhauser", age_elixhauser_models),
        (
            "pure_or_overall",
            "age_elixhauser_log1p_prior_admissions_365d",
            readmission_adjusted_models,
        ),
        (
            "pure_or_overall",
            "age_elixhauser_log1p_prior_all_mimic_admissions",
            readmission_adjusted_models,
        ),
        ("combined", "age_elixhauser", combined_age_elixhauser_models),
        (
            "combined",
            "age_elixhauser_log1p_prior_admissions_365d",
            combined_readmission_adjusted_models,
        ),
        (
            "combined",
            "age_elixhauser_log1p_prior_all_mimic_admissions",
            combined_readmission_adjusted_models,
        ),
    ]
    rows = []
    for subgroup_set, adjustment, model_table in model_sets:
        if model_table.empty:
            continue
        cohort_rows = model_table.loc[model_table["term"].eq("mhh1_psychotic")].copy()
        if cohort_rows.empty:
            continue
        if adjustment.endswith("prior_admissions_365d"):
            cohort_rows = cohort_rows.loc[
                cohort_rows["model"].str.contains("prior365", regex=False)
            ].copy()
        elif adjustment.endswith("prior_all_mimic_admissions"):
            cohort_rows = cohort_rows.loc[
                cohort_rows["model"].str.contains("prior_all_mimic", regex=False)
            ].copy()
        if cohort_rows.empty:
            continue
        cohort_rows["subgroup_set"] = subgroup_set
        cohort_rows["adjustment"] = adjustment
        rows.append(cohort_rows)
    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    value_columns = [
        "n_admissions",
        "n_events",
        "n_clusters",
        "odds_ratio",
        "odds_ratio_ci_low",
        "odds_ratio_ci_high",
        "p_value",
        "estimate_log_odds",
        "cluster_robust_se",
        "status",
        "fit_converged",
        "fit_message",
        "fit_warnings",
    ]
    comparison = combined.pivot_table(
        index=["subgroup_set", "stratum", "outcome"],
        columns="adjustment",
        values=value_columns,
        aggfunc="first",
    )
    comparison.columns = [
        f"{value}_{adjustment}" for value, adjustment in comparison.columns
    ]
    comparison = comparison.reset_index()

    base_or = "odds_ratio_age_elixhauser"
    readmit_or = "odds_ratio_age_elixhauser_log1p_prior_admissions_365d"
    if base_or in comparison.columns and readmit_or in comparison.columns:
        comparison["odds_ratio_difference_after_365d_readmission_adjustment"] = (
            comparison[readmit_or] - comparison[base_or]
        )
        comparison["odds_ratio_ratio_after_365d_readmission_adjustment"] = (
            comparison[readmit_or] / comparison[base_or]
        )

    outcome_order = {outcome: index for index, outcome in enumerate(OUTCOME_COLUMNS)}
    comparison["outcome_order"] = comparison["outcome"].map(outcome_order)
    comparison["stratum_order"] = comparison["stratum"].eq("overall").map(
        {True: 0, False: 1}
    )
    comparison = comparison.sort_values(
        ["subgroup_set", "stratum_order", "stratum", "outcome_order"]
    ).drop(columns=["stratum_order", "outcome_order"])
    return comparison


def main() -> None:
    """Run subject-level mortality analysis and write CSV outputs."""
    descriptors = load_descriptors()
    matching_covariates = load_matching_covariates()
    pure_assignments = load_pure_subgroup_assignments()
    combined_assignments = load_combined_subgroup_assignments()

    admission_level = add_matching_covariates(descriptors, matching_covariates)
    admission_level = add_pure_subgroup_columns(admission_level, pure_assignments)
    admission_level = add_combined_subgroup_columns(admission_level, combined_assignments)
    admission_level = add_real_readmission_columns(admission_level)
    admission_overall_summary = summarize_outcomes(admission_level)
    admission_overall_comparison = compare_cohorts(admission_overall_summary)
    admission_pure = admission_level.dropna(subset=["pure_chief_complaint_group"]).copy()
    admission_pure_summary = summarize_outcomes(
        admission_pure,
        ["pure_chief_complaint_group", "pure_chief_complaint_group_label"],
    )
    admission_pure_comparison = compare_cohorts(
        admission_pure_summary,
        ["pure_chief_complaint_group", "pure_chief_complaint_group_label"],
    )
    admission_combined = admission_level.dropna(
        subset=["combined_chief_complaint_group"]
    ).copy()
    admission_combined_summary = summarize_outcomes(
        admission_combined,
        ["combined_chief_complaint_group", "combined_chief_complaint_group_label"],
    )
    admission_combined_comparison = compare_cohorts(
        admission_combined_summary,
        ["combined_chief_complaint_group", "combined_chief_complaint_group_label"],
    )
    admission_adjusted_models = fit_admission_level_models(admission_level)
    admission_readmission_adjusted_models = (
        fit_admission_level_readmission_adjusted_models(admission_level)
    )
    admission_combined_adjusted_models = fit_admission_level_combined_models(
        admission_level
    )
    admission_combined_readmission_adjusted_models = (
        fit_admission_level_combined_readmission_adjusted_models(admission_level)
    )
    admission_cohort_term_model_comparison = build_cohort_term_model_comparison(
        admission_adjusted_models,
        admission_readmission_adjusted_models,
        admission_combined_adjusted_models,
        admission_combined_readmission_adjusted_models,
    )

    selected, selection_summary = select_first_mhh1_subject_pairs(descriptors)
    selected, selection_summary, dropped_repeated_mhc0_pairs = restrict_to_unique_mhc0_subjects(
        selected,
        selection_summary,
    )
    selected = add_matching_covariates(selected, matching_covariates)
    selected = add_pure_subgroup_columns(selected, pure_assignments)
    selected = add_combined_subgroup_columns(selected, combined_assignments)

    overall_summary = summarize_outcomes(selected)
    overall_comparison = compare_cohorts(overall_summary)
    pure_summary = summarize_outcomes(
        selected.dropna(subset=["pure_chief_complaint_group"]),
        ["pure_chief_complaint_group", "pure_chief_complaint_group_label"],
    )
    pure_comparison = compare_cohorts(
        pure_summary,
        ["pure_chief_complaint_group", "pure_chief_complaint_group_label"],
    )
    age_bin_counts, elixhauser_bin_counts = summarize_covariate_bins(selected)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    admission_level.to_csv(
        OUTPUT_DIR / "admission_level_mortality_all_matched_dataset.csv",
        index=False,
    )
    summarize_real_readmission_distribution(admission_level).to_csv(
        OUTPUT_DIR / "admission_level_prior_all_mimic_admission_distribution.csv",
        index=False,
    )
    summarize_recent_readmission_distribution(admission_level).to_csv(
        OUTPUT_DIR / "admission_level_prior_365d_admission_distribution.csv",
        index=False,
    )
    admission_overall_summary.to_csv(
        OUTPUT_DIR / "admission_level_mortality_overall_summary.csv",
        index=False,
    )
    admission_overall_comparison.to_csv(
        OUTPUT_DIR / "admission_level_mortality_overall_cohort_comparison.csv",
        index=False,
    )
    admission_pure_summary.to_csv(
        OUTPUT_DIR / "admission_level_mortality_pure_subgroup_summary.csv",
        index=False,
    )
    admission_pure_comparison.to_csv(
        OUTPUT_DIR / "admission_level_mortality_pure_subgroup_cohort_comparison.csv",
        index=False,
    )
    admission_combined_summary.to_csv(
        OUTPUT_DIR / "admission_level_mortality_combined_subgroup_summary.csv",
        index=False,
    )
    admission_combined_comparison.to_csv(
        OUTPUT_DIR / "admission_level_mortality_combined_subgroup_cohort_comparison.csv",
        index=False,
    )
    admission_adjusted_models.to_csv(
        OUTPUT_DIR / "admission_level_mortality_adjusted_logistic_models.csv",
        index=False,
    )
    admission_readmission_adjusted_models.to_csv(
        OUTPUT_DIR
        / "admission_level_mortality_readmission_adjusted_logistic_models.csv",
        index=False,
    )
    admission_combined_adjusted_models.to_csv(
        OUTPUT_DIR / "admission_level_mortality_combined_subgroup_adjusted_logistic_models.csv",
        index=False,
    )
    admission_combined_readmission_adjusted_models.to_csv(
        OUTPUT_DIR
        / "admission_level_mortality_combined_subgroup_readmission_adjusted_logistic_models.csv",
        index=False,
    )
    admission_cohort_term_model_comparison.to_csv(
        OUTPUT_DIR / "admission_level_mortality_cohort_term_model_comparison.csv",
        index=False,
    )
    selected.to_csv(
        OUTPUT_DIR / "subject_level_mortality_selected_pair_dataset.csv",
        index=False,
    )
    dropped_repeated_mhc0_pairs.to_csv(
        OUTPUT_DIR / "subject_level_mortality_dropped_repeated_mhc0_pairs.csv",
        index=False,
    )
    selection_summary.to_csv(
        OUTPUT_DIR / "subject_level_mortality_pair_selection_summary.csv",
        index=False,
    )
    summarize_covariates(selected).to_csv(
        OUTPUT_DIR / "subject_level_mortality_age_elixhauser_summary.csv",
        index=False,
    )
    summarize_covariate_smds(selected).to_csv(
        OUTPUT_DIR / "subject_level_mortality_age_elixhauser_smd.csv",
        index=False,
    )
    age_bin_counts.to_csv(
        OUTPUT_DIR / "subject_level_mortality_age_bin_counts.csv",
        index=False,
    )
    elixhauser_bin_counts.to_csv(
        OUTPUT_DIR / "subject_level_mortality_elixhauser_bin_counts.csv",
        index=False,
    )
    overall_summary.to_csv(
        OUTPUT_DIR / "subject_level_mortality_overall_summary.csv",
        index=False,
    )
    overall_comparison.to_csv(
        OUTPUT_DIR / "subject_level_mortality_overall_cohort_comparison.csv",
        index=False,
    )
    pure_summary.to_csv(
        OUTPUT_DIR / "subject_level_mortality_pure_subgroup_summary.csv",
        index=False,
    )
    pure_comparison.to_csv(
        OUTPUT_DIR / "subject_level_mortality_pure_subgroup_cohort_comparison.csv",
        index=False,
    )
    summarize_time_to_death(selected).to_csv(
        OUTPUT_DIR / "subject_level_mortality_overall_time_to_death_summary.csv",
        index=False,
    )
    summarize_time_to_death(
        selected.dropna(subset=["pure_chief_complaint_group"]),
        ["pure_chief_complaint_group", "pure_chief_complaint_group_label"],
    ).to_csv(
        OUTPUT_DIR / "subject_level_mortality_pure_subgroup_time_to_death_summary.csv",
        index=False,
    )

    print(f"Saved subject-level mortality risk analysis to: {OUTPUT_DIR}")
    print("\n=== Admission-level overall mortality comparison ===")
    print(admission_overall_comparison.to_string(index=False))
    print("\n=== Admission-level adjusted model, cohort term ===")
    cohort_model_rows = admission_adjusted_models.loc[
        admission_adjusted_models["term"].eq("mhh1_psychotic")
        & admission_adjusted_models["stratum"].eq("overall")
    ]
    print(cohort_model_rows.to_string(index=False))
    print("\n=== Admission-level prior admissions within 365d distribution ===")
    print(summarize_recent_readmission_distribution(admission_level).to_string(index=False))
    print("\n=== Admission-level full-MIMIC prior admission distribution ===")
    print(summarize_real_readmission_distribution(admission_level).to_string(index=False))
    print("\n=== Admission-level readmission-adjusted model, cohort term ===")
    readmission_cohort_model_rows = admission_readmission_adjusted_models.loc[
        admission_readmission_adjusted_models["term"].eq("mhh1_psychotic")
        & admission_readmission_adjusted_models["stratum"].eq("overall")
    ]
    print(readmission_cohort_model_rows.to_string(index=False))
    print("\n=== Admission-level combined subgroup mortality comparison ===")
    print(admission_combined_comparison.to_string(index=False))
    print("\n=== Admission-level combined subgroup adjusted models, cohort term ===")
    if admission_combined_adjusted_models.empty:
        print("No combined subgroup models were fit.")
    else:
        print(
            admission_combined_adjusted_models.loc[
                admission_combined_adjusted_models["term"].eq("mhh1_psychotic")
            ].to_string(index=False)
        )
    print("\n=== Admission-level combined subgroup readmission-adjusted models, cohort term ===")
    if admission_combined_readmission_adjusted_models.empty:
        print("No combined subgroup readmission-adjusted models were fit.")
    else:
        print(
            admission_combined_readmission_adjusted_models.loc[
                admission_combined_readmission_adjusted_models["term"].eq(
                    "mhh1_psychotic"
                )
            ].to_string(index=False)
        )
    print("\n=== Admission-level post-discharge mortality cohort term comparison ===")
    post_discharge_comparison = admission_cohort_term_model_comparison.loc[
        admission_cohort_term_model_comparison["outcome"].eq(
            "post_discharge_death_within_1y"
        )
    ]
    print(post_discharge_comparison.to_string(index=False))
    print("\n=== Pair selection ===")
    print(selection_summary.to_string(index=False))
    print("\n=== Overall mortality comparison ===")
    print(overall_comparison.to_string(index=False))
    print("\n=== Age/Elixhauser balance in mortality analysis set ===")
    print(summarize_covariates(selected).to_string(index=False))
    print("\n=== Age/Elixhauser SMDs ===")
    print(summarize_covariate_smds(selected).to_string(index=False))
    print("\n=== Pure chief-complaint subgroup mortality comparison ===")
    print(pure_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
