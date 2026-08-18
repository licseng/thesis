"""Model utilization differences adjusted for true prior admission history.

This analysis asks whether MHH1-vs-MHC0 differences in workup/utilization remain
after accounting for real prior MIMIC admissions, not just repeated admissions
inside the matched cohort.

Models use admission-level rows and cluster-robust standard errors by subject:

    log1p(utilization outcome) ~ cohort
    log1p(utilization outcome) ~ cohort + age + Elixhauser
    log1p(utilization outcome) ~ cohort + age + Elixhauser
                                + log1p(prior all-MIMIC admissions)

Outputs:
    analysis_output_utilization_readmission_adjusted/
        utilization_readmission_adjusted_model_dataset.csv
        utilization_readmission_adjusted_log_linear_models.csv
        utilization_readmission_adjusted_cohort_term_comparison.csv
        utilization_readmission_adjusted_outcome_summary.csv
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

COHORT_DIR_FOR_IMPORT = Path(__file__).resolve().parent.parent / "02_cohort_matching"
sys.path.insert(0, str(COHORT_DIR_FOR_IMPORT))

import _matched_cohort_characterization_common as common  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
COHORT_DIR = PROJECT_DIR / "02_cohort_matching"
CLUSTERING_DIR = PROJECT_DIR / "04_clustering_chief_complaints"
MATCHED_PAIRS_PATH = COHORT_DIR / "matched_cohort_output" / "matched_pairs.csv"
SUBGROUP_ASSIGNMENT_PATH = (
    CLUSTERING_DIR
    / "analysis_output_chief_complaint_subgroup_balance_check"
    / "chief_complaint_subgroup_admission_assignments.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_utilization_readmission_adjusted"
ID_COLUMNS = ["cohort", "subject_id", "hadm_id"]
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


def load_matching_covariates() -> pd.DataFrame:
    """Load age and Elixhauser covariates from the matched-pair output."""
    if not MATCHED_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing matched-pair output: {MATCHED_PAIRS_PATH}")
    pairs = pd.read_csv(MATCHED_PAIRS_PATH)

    mhh = pairs.loc[
        :,
        [
            "pair_id",
            "mhh_subject_id",
            "mhh_hadm_id",
            "mhh_age_at_admission",
            "mhh_elixhauser_score",
        ],
    ].rename(
        columns={
            "mhh_subject_id": "subject_id",
            "mhh_hadm_id": "hadm_id",
            "mhh_age_at_admission": "age_at_admission",
            "mhh_elixhauser_score": "elixhauser_score",
        }
    )
    mhh["cohort"] = "MHH1_psychotic"

    mhc0 = pairs.loc[
        :,
        [
            "pair_id",
            "mhc0_subject_id",
            "mhc0_hadm_id",
            "mhc0_age_at_admission",
            "mhc0_elixhauser_score",
        ],
    ].rename(
        columns={
            "mhc0_subject_id": "subject_id",
            "mhc0_hadm_id": "hadm_id",
            "mhc0_age_at_admission": "age_at_admission",
            "mhc0_elixhauser_score": "elixhauser_score",
        }
    )
    mhc0["cohort"] = "MHC0"

    covariates = pd.concat([mhh, mhc0], ignore_index=True)
    covariates = common.validate_id_columns(covariates, "matching_covariates")
    covariates["pair_id"] = pd.to_numeric(covariates["pair_id"], errors="raise").astype(int)
    covariates["age_at_admission"] = pd.to_numeric(
        covariates["age_at_admission"],
        errors="coerce",
    )
    covariates["elixhauser_score"] = pd.to_numeric(
        covariates["elixhauser_score"],
        errors="coerce",
    )
    return covariates.loc[
        :,
        ["pair_id", *ID_COLUMNS, "age_at_admission", "elixhauser_score"],
    ]


def load_or_build_utilization_counts(matched_ids: pd.DataFrame) -> pd.DataFrame:
    """Build event/order counts per admission from DBeaver exports."""
    event_tables = {
        "labevents": common.load_optional_table("labevents"),
        "microbiologyevents": common.load_optional_table("microbiologyevents"),
        "poe": common.load_optional_table("poe"),
        "poe_detail": common.load_optional_table("poe_detail"),
    }
    return common.build_event_counts_by_admission(matched_ids, event_tables)


def load_pure_subgroup_assignments() -> pd.DataFrame:
    """Load five pure chief-complaint subgroup assignments."""
    if not SUBGROUP_ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            "Missing chief complaint subgroup assignments. Run "
            "04_clustering_chief_complaints/01_describe_chief_complaint_subgroups.py first: "
            f"{SUBGROUP_ASSIGNMENT_PATH}"
        )

    assignments = pd.read_csv(SUBGROUP_ASSIGNMENT_PATH)
    required = {
        "pair_id",
        *ID_COLUMNS,
        "n_chief_complaint_subgroups_matched",
        *SUBGROUP_FLAG_COLUMNS.values(),
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Subgroup assignment file is missing columns: {missing}")

    assignments = common.validate_id_columns(assignments, "subgroup_assignments")
    assignments["pair_id"] = pd.to_numeric(
        assignments["pair_id"],
        errors="raise",
    ).astype(int)
    assignments["n_chief_complaint_subgroups_matched"] = pd.to_numeric(
        assignments["n_chief_complaint_subgroups_matched"],
        errors="raise",
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
        ["pair_id", *ID_COLUMNS, "pure_chief_complaint_group", "pure_chief_complaint_group_label"],
    ]


def load_combined_subgroup_assignments() -> pd.DataFrame:
    """Load the two exclusive combined chief-complaint subgroup assignments."""
    if not SUBGROUP_ASSIGNMENT_PATH.exists():
        raise FileNotFoundError(
            "Missing chief complaint subgroup assignments. Run "
            "04_clustering_chief_complaints/01_describe_chief_complaint_subgroups.py first: "
            f"{SUBGROUP_ASSIGNMENT_PATH}"
        )

    assignments = pd.read_csv(
        SUBGROUP_ASSIGNMENT_PATH,
        usecols=["pair_id", *ID_COLUMNS, "exclusive_combined_group"],
    )
    assignments = common.validate_id_columns(assignments, "combined_subgroup_assignments")
    assignments["pair_id"] = pd.to_numeric(
        assignments["pair_id"],
        errors="raise",
    ).astype(int)
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


def build_model_dataset() -> pd.DataFrame:
    """Assemble admission-level utilization outcomes and covariates."""
    matched_ids = common.load_expected_matched_ids()
    descriptors = common.validate_id_columns(
        common.load_required_table("descriptors"),
        "descriptors",
    )
    covariates = load_matching_covariates()
    utilization_counts = load_or_build_utilization_counts(matched_ids)
    pure_assignments = load_pure_subgroup_assignments()
    combined_assignments = load_combined_subgroup_assignments()

    required_descriptor_columns = {
        "cohort",
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "edregtime",
        "edouttime",
        "n_prior_all_admissions_for_subject",
    }
    missing = sorted(required_descriptor_columns - set(descriptors.columns))
    if missing:
        raise ValueError(
            "Descriptor table is missing utilization/readmission covariates. "
            "Rerun sql-scripts/06_save_tables/02_Additional_info_export_on_cohort.sql. "
            f"Missing: {missing}"
        )

    descriptor_subset = descriptors.loc[:, sorted(required_descriptor_columns)].copy()
    output = utilization_counts.merge(
        covariates,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    output = output.merge(
        descriptor_subset,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    output = output.merge(
        pure_assignments,
        on=["pair_id", *ID_COLUMNS],
        how="left",
        validate="one_to_one",
    )
    output = output.merge(
        combined_assignments,
        on=["pair_id", *ID_COLUMNS],
        how="left",
        validate="one_to_one",
    )

    for column in ["admittime", "dischtime", "edregtime", "edouttime"]:
        output[column] = pd.to_datetime(output[column], errors="coerce")
    output["hospital_los_days"] = (
        output["dischtime"] - output["admittime"]
    ).dt.total_seconds() / 86400.0
    output["ed_los_hours"] = (
        output["edouttime"] - output["edregtime"]
    ).dt.total_seconds() / 3600.0
    output.loc[output["hospital_los_days"].lt(0), "hospital_los_days"] = pd.NA
    output.loc[output["ed_los_hours"].lt(0), "ed_los_hours"] = pd.NA

    output["n_prior_all_admissions_for_subject"] = pd.to_numeric(
        output["n_prior_all_admissions_for_subject"],
        errors="raise",
    )
    output["log1p_prior_all_mimic_admissions"] = np.log1p(
        output["n_prior_all_admissions_for_subject"],
    )
    output["mhh1_psychotic"] = output["cohort"].eq("MHH1_psychotic").astype(float)
    output["age_at_admission_per_10y"] = output["age_at_admission"] / 10.0
    output["elixhauser_score_per_5pt"] = output["elixhauser_score"] / 5.0
    output["cluster_id"] = (
        output["cohort"].astype(str) + "_" + output["subject_id"].astype(str)
    )
    return output


def summarize_outcomes(model_data: pd.DataFrame, outcome_columns: list[str]) -> pd.DataFrame:
    """Summarize raw utilization outcomes by cohort."""
    rows = []
    for cohort, group in model_data.groupby("cohort"):
        for outcome in outcome_columns:
            values = pd.to_numeric(group[outcome], errors="coerce").dropna()
            rows.append(
                {
                    "cohort": cohort,
                    "outcome": outcome,
                    "n_admissions": len(values),
                    "n_subjects": group.loc[values.index, "subject_id"].nunique(),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "min": values.min(),
                    "max": values.max(),
                    "n_with_nonzero": int(values.gt(0).sum()),
                    "pct_with_nonzero": 100.0 * values.gt(0).mean(),
                }
            )
    return pd.DataFrame(rows).sort_values(["outcome", "cohort"])


def fit_log_linear_model(
    model_data: pd.DataFrame,
    outcome: str,
    model_name: str,
    predictor_columns: list[str],
    subgroup_set: str,
    stratum: str,
) -> pd.DataFrame:
    """Fit log1p(outcome) linear model with cluster-robust SEs by subject."""
    required_columns = [outcome, "cluster_id", *predictor_columns]
    data = model_data.loc[:, required_columns].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=required_columns).reset_index(drop=True)
    data = data.loc[data[outcome].ge(0)].reset_index(drop=True)

    coefficient_names = ["intercept", *predictor_columns]
    if data.empty:
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "subgroup_set": subgroup_set,
                    "stratum": stratum,
                    "outcome": outcome,
                    "term": term,
                    "status": "not_fit_no_complete_rows",
                }
                for term in coefficient_names
            ]
        )

    x = sm.add_constant(
        data.loc[:, predictor_columns].astype(float),
        has_constant="add",
    ).rename(columns={"const": "intercept"})
    y = np.log1p(data[outcome].astype(float))

    captured_warnings = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.OLS(y, x).fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": data["cluster_id"],
                    "use_correction": True,
                },
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
                    "subgroup_set": subgroup_set,
                    "stratum": stratum,
                    "outcome": outcome,
                    "term": term,
                    "n_admissions": len(data),
                    "n_clusters": data["cluster_id"].nunique(),
                    "status": "fit_failed",
                    "fit_message": repr(exc),
                }
                for term in coefficient_names
            ]
        )

    conf_int = fit.conf_int(alpha=0.05)
    rows = []
    for term in x.columns:
        estimate = fit.params.get(term, pd.NA)
        ci_low = conf_int.loc[term, 0] if term in conf_int.index else pd.NA
        ci_high = conf_int.loc[term, 1] if term in conf_int.index else pd.NA
        rows.append(
            {
                "model": model_name,
                "subgroup_set": subgroup_set,
                "stratum": stratum,
                "outcome": outcome,
                "term": term,
                "n_admissions": len(data),
                "n_clusters": data["cluster_id"].nunique(),
                "estimate_log1p_scale": estimate,
                "cluster_robust_se": fit.bse.get(term, pd.NA),
                "z": fit.tvalues.get(term, pd.NA),
                "p_value": fit.pvalues.get(term, pd.NA),
                "multiplicative_ratio": math.exp(estimate),
                "pct_difference": 100.0 * (math.exp(estimate) - 1.0),
                "multiplicative_ratio_ci_low": math.exp(ci_low)
                if ci_low is not pd.NA
                else pd.NA,
                "multiplicative_ratio_ci_high": math.exp(ci_high)
                if ci_high is not pd.NA
                else pd.NA,
                "status": "fit_converged" if not captured_warnings else "fit_warning",
                "fit_warnings": " | ".join(captured_warnings),
                "r_squared": fit.rsquared,
            }
        )
    return pd.DataFrame(rows)


def fit_all_models_for_stratum(
    model_data: pd.DataFrame,
    outcome_columns: list[str],
    subgroup_set: str,
    stratum: str,
) -> pd.DataFrame:
    """Fit all utilization models for all outcomes in one analysis stratum."""
    model_specs = [
        ("cohort_only", ["mhh1_psychotic"]),
        (
            "age_elixhauser",
            ["mhh1_psychotic", "age_at_admission_per_10y", "elixhauser_score_per_5pt"],
        ),
        (
            "age_elixhauser_log1p_prior_all_mimic_admissions",
            [
                "mhh1_psychotic",
                "age_at_admission_per_10y",
                "elixhauser_score_per_5pt",
                "log1p_prior_all_mimic_admissions",
            ],
        ),
    ]
    rows = []
    for outcome in outcome_columns:
        for model_name, predictors in model_specs:
            rows.append(
                fit_log_linear_model(
                    model_data,
                    outcome,
                    model_name,
                    predictors,
                    subgroup_set,
                    stratum,
                )
            )
    return pd.concat(rows, ignore_index=True)


def fit_all_models(model_data: pd.DataFrame, outcome_columns: list[str]) -> pd.DataFrame:
    """Fit utilization models overall and inside chief-complaint subgroups."""
    model_rows = [
        fit_all_models_for_stratum(model_data, outcome_columns, "overall", "overall")
    ]

    pure = model_data.dropna(subset=["pure_chief_complaint_group"]).copy()
    for (group_name, group_label), group in pure.groupby(
        ["pure_chief_complaint_group", "pure_chief_complaint_group_label"],
        dropna=False,
    ):
        model_rows.append(
            fit_all_models_for_stratum(
                group,
                outcome_columns,
                "pure_chief_complaint",
                f"{group_name}: {group_label}",
            )
        )

    combined = model_data.dropna(subset=["combined_chief_complaint_group"]).copy()
    for (group_name, group_label), group in combined.groupby(
        ["combined_chief_complaint_group", "combined_chief_complaint_group_label"],
        dropna=False,
    ):
        model_rows.append(
            fit_all_models_for_stratum(
                group,
                outcome_columns,
                "combined_chief_complaint",
                f"{group_name}: {group_label}",
            )
        )
    return pd.concat(model_rows, ignore_index=True)


def build_cohort_term_comparison(model_results: pd.DataFrame) -> pd.DataFrame:
    """Put the MHH1 coefficient from each model side by side."""
    cohort_rows = model_results.loc[model_results["term"].eq("mhh1_psychotic")].copy()
    value_columns = [
        "n_admissions",
        "n_clusters",
        "multiplicative_ratio",
        "multiplicative_ratio_ci_low",
        "multiplicative_ratio_ci_high",
        "pct_difference",
        "p_value",
        "estimate_log1p_scale",
        "cluster_robust_se",
        "status",
        "fit_warnings",
    ]
    comparison = cohort_rows.pivot_table(
        index=["subgroup_set", "stratum", "outcome"],
        columns="model",
        values=value_columns,
        aggfunc="first",
    )
    comparison.columns = [f"{value}_{model}" for value, model in comparison.columns]
    comparison = comparison.reset_index()

    base = "multiplicative_ratio_age_elixhauser"
    adjusted = "multiplicative_ratio_age_elixhauser_log1p_prior_all_mimic_admissions"
    if base in comparison.columns and adjusted in comparison.columns:
        comparison["ratio_difference_after_readmission_adjustment"] = (
            comparison[adjusted] - comparison[base]
        )
        comparison["ratio_ratio_after_readmission_adjustment"] = (
            comparison[adjusted] / comparison[base]
        )
    return comparison.sort_values(["subgroup_set", "stratum", "outcome"])


def main() -> None:
    """Run readmission-adjusted utilization models and write outputs."""
    model_data = build_model_dataset()
    outcome_columns = [
        "n_labevents_rows",
        "n_microbiologyevents_rows",
        "n_poe_rows",
        "n_poe_detail_rows",
        "hospital_los_days",
        "ed_los_hours",
    ]
    outcome_columns = [column for column in outcome_columns if column in model_data.columns]
    outcome_summary = summarize_outcomes(model_data, outcome_columns)
    model_results = fit_all_models(model_data, outcome_columns)
    cohort_term_comparison = build_cohort_term_comparison(model_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(
        OUTPUT_DIR / "utilization_readmission_adjusted_model_dataset.csv",
        index=False,
    )
    outcome_summary.to_csv(
        OUTPUT_DIR / "utilization_readmission_adjusted_outcome_summary.csv",
        index=False,
    )
    model_results.to_csv(
        OUTPUT_DIR / "utilization_readmission_adjusted_log_linear_models.csv",
        index=False,
    )
    cohort_term_comparison.to_csv(
        OUTPUT_DIR / "utilization_readmission_adjusted_cohort_term_comparison.csv",
        index=False,
    )

    print(f"Saved utilization readmission-adjusted analysis to: {OUTPUT_DIR}")
    print("\n=== Cohort Term Comparison ===")
    display_columns = [
        "subgroup_set",
        "stratum",
        "outcome",
        "multiplicative_ratio_age_elixhauser",
        "p_value_age_elixhauser",
        "multiplicative_ratio_age_elixhauser_log1p_prior_all_mimic_admissions",
        "p_value_age_elixhauser_log1p_prior_all_mimic_admissions",
        "ratio_difference_after_readmission_adjustment",
    ]
    display_columns = [
        column for column in display_columns if column in cohort_term_comparison.columns
    ]
    print(cohort_term_comparison.loc[:, display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
