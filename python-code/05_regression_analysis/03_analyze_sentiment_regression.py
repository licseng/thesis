"""Regression checks for sentiment differences between MHH1 and MHC0.

This script uses the completed sentiment-classifier output and the existing
admission-level regression covariate dataset. It estimates:

1. admission-level logistic models:
       any positive sentiment section vs none
       any negative sentiment section vs none
       any non-neutral sentiment section vs none

2. section-level logistic models:
       positive section vs all sections
       negative section vs all sections
       non-neutral section vs all sections
       positive vs negative among sentiment-bearing sections only

Section-level pooled models include section-name fixed effects. Section-specific
models are also written separately. Robust standard errors are clustered by
subject-level cluster_id from the shared regression covariate dataset.
"""

from __future__ import annotations

import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SENTIMENT_OUTPUT_DIR = (
    PROJECT_DIR / "07_sentiment_analysis" / "sentiment_classifier_output_prompt_A_all"
)
CHIEF_COMPLAINT_SUBGROUP_PATH = (
    PROJECT_DIR
    / "04_clustering_chief_complaints"
    / "analysis_output_chief_complaint_subgroup_balance_check"
    / "chief_complaint_subgroup_admission_assignments.csv"
)
COVARIATE_PATH = (
    SCRIPT_DIR
    / "analysis_output_utilization_readmission_adjusted"
    / "utilization_readmission_adjusted_model_dataset.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_sentiment_regression"

SECTION_RESULTS_PATH = SENTIMENT_OUTPUT_DIR / "sentiment_section_classifier_results.csv"
ADMISSION_RESULTS_PATH = SENTIMENT_OUTPUT_DIR / "sentiment_admission_summary.csv"

ID_COLUMNS = ["cohort", "subject_id", "hadm_id"]
SENTIMENT_BEARING_LABELS = {"positive", "negative"}
NON_NEUTRAL_LABELS = {"positive", "negative", "mixed"}

BASE_COVARIATE_COLUMNS = [
    "cohort",
    "subject_id",
    "hadm_id",
    "pair_id",
    "age_at_admission",
    "elixhauser_score",
    "n_prior_admissions_within_365d_for_subject",
    "n_prior_all_admissions_for_subject",
    "log1p_prior_admissions_365d",
    "log1p_prior_all_mimic_admissions",
    "mhh1_psychotic",
    "age_at_admission_per_10y",
    "elixhauser_score_per_5pt",
    "cluster_id",
]

MODEL_SPECS = {
    "cohort_only": ["mhh1_psychotic"],
    "age_elixhauser": [
        "mhh1_psychotic",
        "age_at_admission_per_10y",
        "elixhauser_score_per_5pt",
    ],
    "age_elixhauser_prior365": [
        "mhh1_psychotic",
        "age_at_admission_per_10y",
        "elixhauser_score_per_5pt",
        "log1p_prior_admissions_365d",
    ],
    "age_elixhauser_prior_all_mimic": [
        "mhh1_psychotic",
        "age_at_admission_per_10y",
        "elixhauser_score_per_5pt",
        "log1p_prior_all_mimic_admissions",
    ],
}

CHIEF_COMPLAINT_SUBGROUPS = {
    "has_abdominal_pain": "abdominal pain",
    "has_shortness_of_breath": "shortness of breath",
    "has_chest_pain": "chest pain",
    "has_altered_mental_status": "altered mental status",
    "has_nausea_vomiting": "nausea/vomiting",
    "has_abdominal_pain_nausea_vomiting": "combined abdominal/GI",
    "has_chest_pain_shortness_of_breath": "combined chest/SOB",
}

NEGATIVE_COUNT_MODEL_SPECS = {
    "age_elixhauser": [
        "mhh1_psychotic",
        "age_at_admission_per_10y",
        "elixhauser_score_per_5pt",
    ],
    "age_elixhauser_prior365": [
        "mhh1_psychotic",
        "age_at_admission_per_10y",
        "elixhauser_score_per_5pt",
        "log1p_prior_admissions_365d",
    ],
    "age_elixhauser_prior_all_mimic": [
        "mhh1_psychotic",
        "age_at_admission_per_10y",
        "elixhauser_score_per_5pt",
        "log1p_prior_all_mimic_admissions",
    ],
}


def normalize_id_columns(data: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Normalize cohort, subject_id, and hadm_id for reliable joins."""
    missing = [column for column in ID_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"{table_name} is missing ID column(s): {missing}")

    output = data.copy()
    output["cohort"] = output["cohort"].astype(str)
    output["subject_id"] = pd.to_numeric(output["subject_id"], errors="raise").astype(int)
    output["hadm_id"] = pd.to_numeric(output["hadm_id"], errors="raise").astype(int)
    return output


def load_covariates() -> pd.DataFrame:
    """Load the shared admission-level covariates used by regression scripts."""
    if not COVARIATE_PATH.exists():
        raise FileNotFoundError(
            "Missing regression covariate dataset. Run "
            "05_regression_analysis/01_analyze_utilization_adjusted_for_readmission.py first: "
            f"{COVARIATE_PATH}"
        )

    covariates = pd.read_csv(COVARIATE_PATH, usecols=BASE_COVARIATE_COLUMNS)
    covariates = normalize_id_columns(covariates, "covariates")
    if covariates.duplicated(ID_COLUMNS).any():
        duplicated = int(covariates.duplicated(ID_COLUMNS, keep=False).sum())
        raise ValueError(f"Covariate dataset has {duplicated} duplicated admission rows.")

    numeric_columns = [
        "pair_id",
        "age_at_admission",
        "elixhauser_score",
        "n_prior_admissions_within_365d_for_subject",
        "n_prior_all_admissions_for_subject",
        "log1p_prior_admissions_365d",
        "log1p_prior_all_mimic_admissions",
        "mhh1_psychotic",
        "age_at_admission_per_10y",
        "elixhauser_score_per_5pt",
    ]
    for column in numeric_columns:
        covariates[column] = pd.to_numeric(covariates[column], errors="coerce")
    covariates["cluster_id"] = covariates["cluster_id"].astype(str)
    return covariates


def load_sentiment_sections() -> pd.DataFrame:
    """Load section-level sentiment output and create binary outcomes."""
    if not SECTION_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing sentiment section output: {SECTION_RESULTS_PATH}")

    sections = pd.read_csv(SECTION_RESULTS_PATH)
    sections = normalize_id_columns(sections, "sentiment_section_results")
    sections["sentiment_label"] = (
        sections["sentiment_label"].astype(str).str.strip().str.lower()
    )
    sections["positive_vs_all"] = sections["sentiment_label"].eq("positive").astype(int)
    sections["negative_vs_all"] = sections["sentiment_label"].eq("negative").astype(int)
    sections["non_neutral_vs_all"] = (
        sections["sentiment_label"].isin(NON_NEUTRAL_LABELS).astype(int)
    )
    sections["positive_vs_negative"] = np.where(
        sections["sentiment_label"].isin(SENTIMENT_BEARING_LABELS),
        sections["sentiment_label"].eq("positive").astype(int),
        np.nan,
    )
    return sections


def load_sentiment_admissions() -> pd.DataFrame:
    """Load admission-level sentiment output and create binary outcomes."""
    if not ADMISSION_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing sentiment admission output: {ADMISSION_RESULTS_PATH}")

    admissions = pd.read_csv(ADMISSION_RESULTS_PATH)
    admissions = normalize_id_columns(admissions, "sentiment_admission_summary")
    for column in ["any_positive", "any_negative", "any_mixed"]:
        if column not in admissions.columns:
            raise ValueError(f"Admission summary is missing {column}.")
        admissions[column] = admissions[column].astype(bool).astype(int)
    admissions["any_non_neutral"] = (
        admissions[["any_positive", "any_negative", "any_mixed"]].max(axis=1).astype(int)
    )
    return admissions


def load_chief_complaint_subgroups() -> pd.DataFrame:
    """Load chief-complaint subgroup flags for admission-level subgroup models."""
    if not CHIEF_COMPLAINT_SUBGROUP_PATH.exists():
        raise FileNotFoundError(
            "Missing chief-complaint subgroup assignments. Run "
            "04_clustering_chief_complaints/01_describe_chief_complaint_subgroups.py first: "
            f"{CHIEF_COMPLAINT_SUBGROUP_PATH}"
        )

    subgroups = pd.read_csv(
        CHIEF_COMPLAINT_SUBGROUP_PATH,
        usecols=[*ID_COLUMNS, *CHIEF_COMPLAINT_SUBGROUPS.keys()],
    )
    subgroups = normalize_id_columns(subgroups, "chief_complaint_subgroups")
    if subgroups.duplicated(ID_COLUMNS).any():
        duplicated = int(subgroups.duplicated(ID_COLUMNS, keep=False).sum())
        raise ValueError(f"Subgroup table has {duplicated} duplicated admission rows.")

    for column in CHIEF_COMPLAINT_SUBGROUPS:
        subgroups[column] = subgroups[column].astype(bool)
    return subgroups


def attach_covariates(sentiment_data: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """Attach admission covariates to sentiment rows and check unmatched rows."""
    merged = sentiment_data.merge(
        covariates,
        on=ID_COLUMNS,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing = merged["_merge"].ne("both").sum()
    if missing:
        raise ValueError(f"{missing} sentiment rows could not be matched to covariates.")
    return merged.drop(columns=["_merge"])


def add_section_fixed_effects(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return section-name dummy variables for pooled section-level models."""
    section_dummies = pd.get_dummies(
        data["section_name"].astype(str),
        prefix="section",
        drop_first=True,
        dtype=float,
    )
    output = pd.concat(
        [data.reset_index(drop=True), section_dummies.reset_index(drop=True)],
        axis=1,
    )
    return output, section_dummies.columns.tolist()


def fit_logistic_model(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    model_level: str,
    stratum: str,
    model_spec: str,
    denominator: str,
    cluster_column: str = "cluster_id",
) -> pd.DataFrame:
    """Fit one clustered logistic model and return coefficient-level rows."""
    required_columns = [outcome, cluster_column, *predictors]
    model_data = data.loc[:, required_columns].copy()
    model_data[outcome] = pd.to_numeric(model_data[outcome], errors="coerce")
    for column in predictors:
        model_data[column] = pd.to_numeric(model_data[column], errors="coerce")
    model_data[cluster_column] = model_data[cluster_column].astype(str)
    model_data = model_data.dropna(subset=required_columns).reset_index(drop=True)

    coefficient_names = ["intercept", *predictors]
    n_rows = len(model_data)
    n_events = int(model_data[outcome].sum()) if n_rows else 0
    n_clusters = model_data[cluster_column].nunique() if n_rows else 0

    base_row = {
        "model_level": model_level,
        "stratum": stratum,
        "model_spec": model_spec,
        "denominator": denominator,
        "outcome": outcome,
        "n_rows": n_rows,
        "n_events": n_events,
        "event_rate_pct": 100.0 * n_events / n_rows if n_rows else pd.NA,
        "cluster_column": cluster_column,
        "n_clusters": n_clusters,
        "fit_method": "statsmodels_glm_binomial_cluster_robust",
    }
    if n_rows == 0 or n_events == 0 or n_events == n_rows:
        return pd.DataFrame(
            [
                {
                    **base_row,
                    "term": term,
                    "status": "not_fit_outcome_has_one_class",
                    "fit_converged": False,
                    "fit_message": "not_fit_outcome_has_one_class",
                    "fit_warnings": "",
                }
                for term in coefficient_names
            ]
        )

    x = sm.add_constant(
        model_data.loc[:, predictors].astype(float),
        has_constant="add",
    ).rename(columns={"const": "intercept"})
    y = model_data[outcome].astype(float)

    captured_warnings = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": model_data[cluster_column],
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
                    **base_row,
                    "term": term,
                    "status": "fit_failed",
                    "fit_converged": False,
                    "fit_message": repr(exc),
                    "fit_warnings": "",
                }
                for term in x.columns
            ]
        )

    conf_int = fit.conf_int(alpha=0.05)
    fit_warnings = " | ".join(captured_warnings)
    fit_converged = bool(getattr(fit, "converged", False))
    rows = []
    for term in x.columns:
        estimate = fit.params.get(term, pd.NA)
        ci_low = conf_int.loc[term, 0] if term in conf_int.index else pd.NA
        ci_high = conf_int.loc[term, 1] if term in conf_int.index else pd.NA
        rows.append(
            {
                **base_row,
                "term": term,
                "estimate_log_odds": estimate,
                "cluster_robust_se": fit.bse.get(term, pd.NA),
                "z": fit.tvalues.get(term, pd.NA),
                "p_value": fit.pvalues.get(term, pd.NA),
                "odds_ratio": math.exp(estimate) if pd.notna(estimate) else pd.NA,
                "odds_ratio_ci_low": math.exp(ci_low) if pd.notna(ci_low) else pd.NA,
                "odds_ratio_ci_high": math.exp(ci_high) if pd.notna(ci_high) else pd.NA,
                "status": "fit_converged" if fit_converged and not fit_warnings else "fit_warning",
                "fit_converged": fit_converged,
                "fit_message": "converged" if fit_converged else "not_converged",
                "fit_warnings": fit_warnings,
            }
        )
    return pd.DataFrame(rows)


def fit_poisson_count_model(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    model_level: str,
    stratum: str,
    model_spec: str,
    cluster_column: str = "cluster_id",
) -> pd.DataFrame:
    """Fit one clustered Poisson-log count model and return coefficient rows."""
    required_columns = [outcome, cluster_column, *predictors]
    model_data = data.loc[:, required_columns].copy()
    model_data[outcome] = pd.to_numeric(model_data[outcome], errors="coerce")
    for column in predictors:
        model_data[column] = pd.to_numeric(model_data[column], errors="coerce")
    model_data[cluster_column] = model_data[cluster_column].astype(str)
    model_data = model_data.dropna(subset=required_columns).reset_index(drop=True)
    model_data = model_data.loc[model_data[outcome].ge(0)].reset_index(drop=True)

    coefficient_names = ["intercept", *predictors]
    n_rows = len(model_data)
    total_events = int(model_data[outcome].sum()) if n_rows else 0
    n_clusters = model_data[cluster_column].nunique() if n_rows else 0
    mean_count = total_events / n_rows if n_rows else pd.NA

    base_row = {
        "model_level": model_level,
        "stratum": stratum,
        "model_spec": model_spec,
        "outcome": outcome,
        "n_rows": n_rows,
        "total_events": total_events,
        "mean_count": mean_count,
        "cluster_column": cluster_column,
        "n_clusters": n_clusters,
        "fit_method": "statsmodels_glm_poisson_log_cluster_robust",
    }
    if n_rows == 0 or total_events == 0:
        return pd.DataFrame(
            [
                {
                    **base_row,
                    "term": term,
                    "status": "not_fit_no_events",
                    "fit_converged": False,
                    "fit_message": "not_fit_no_events",
                    "fit_warnings": "",
                }
                for term in coefficient_names
            ]
        )

    x = sm.add_constant(
        model_data.loc[:, predictors].astype(float),
        has_constant="add",
    ).rename(columns={"const": "intercept"})
    y = model_data[outcome].astype(float)

    captured_warnings = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.GLM(
                y,
                x,
                family=sm.families.Poisson(link=sm.families.links.Log()),
            ).fit(
                cov_type="cluster",
                cov_kwds={
                    "groups": model_data[cluster_column],
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
                    **base_row,
                    "term": term,
                    "status": "fit_failed",
                    "fit_converged": False,
                    "fit_message": repr(exc),
                    "fit_warnings": "",
                }
                for term in x.columns
            ]
        )

    conf_int = fit.conf_int(alpha=0.05)
    fit_warnings = " | ".join(captured_warnings)
    fit_converged = bool(getattr(fit, "converged", False))
    rows = []
    for term in x.columns:
        estimate = fit.params.get(term, pd.NA)
        ci_low = conf_int.loc[term, 0] if term in conf_int.index else pd.NA
        ci_high = conf_int.loc[term, 1] if term in conf_int.index else pd.NA
        rows.append(
            {
                **base_row,
                "term": term,
                "estimate_log_rate": estimate,
                "cluster_robust_se": fit.bse.get(term, pd.NA),
                "z": fit.tvalues.get(term, pd.NA),
                "p_value": fit.pvalues.get(term, pd.NA),
                "rate_ratio": math.exp(estimate) if pd.notna(estimate) else pd.NA,
                "rate_ratio_ci_low": math.exp(ci_low) if pd.notna(ci_low) else pd.NA,
                "rate_ratio_ci_high": math.exp(ci_high) if pd.notna(ci_high) else pd.NA,
                "pct_difference": 100.0 * (math.exp(estimate) - 1.0)
                if pd.notna(estimate)
                else pd.NA,
                "status": "fit_converged" if fit_converged and not fit_warnings else "fit_warning",
                "fit_converged": fit_converged,
                "fit_message": "converged" if fit_converged else "not_converged",
                "fit_warnings": fit_warnings,
                "pearson_chi2": getattr(fit, "pearson_chi2", pd.NA),
                "deviance": getattr(fit, "deviance", pd.NA),
            }
        )
    return pd.DataFrame(rows)


def summarize_raw_admission_rates(admissions: pd.DataFrame) -> pd.DataFrame:
    """Summarize crude admission-level sentiment rates by cohort."""
    rows = []
    for outcome in ["any_positive", "any_negative", "any_non_neutral"]:
        grouped = (
            admissions.groupby("cohort")[outcome]
            .agg(n_admissions="size", n_events="sum", mean="mean")
            .reset_index()
        )
        grouped["summary_level"] = "admission"
        grouped["stratum"] = "overall"
        grouped["outcome"] = outcome
        grouped["event_rate_pct"] = grouped["mean"] * 100.0
        rows.append(grouped.drop(columns=["mean"]))
    return pd.concat(rows, ignore_index=True)


def summarize_raw_subgroup_admission_rates(
    admissions: pd.DataFrame,
    subgroups: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize crude admission-level sentiment rates inside complaint subgroups."""
    merged = admissions.merge(
        subgroups,
        on=ID_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for flag_column, subgroup in CHIEF_COMPLAINT_SUBGROUPS.items():
        subgroup_data = merged.loc[merged[flag_column]].copy()
        for outcome in ["any_positive", "any_negative", "any_non_neutral"]:
            grouped = (
                subgroup_data.groupby("cohort")[outcome]
                .agg(n_admissions="size", n_events="sum", mean="mean")
                .reset_index()
            )
            grouped["subgroup"] = subgroup
            grouped["flag_column"] = flag_column
            grouped["outcome"] = outcome
            grouped["event_rate_pct"] = grouped["mean"] * 100.0
            rows.append(grouped.drop(columns=["mean"]))
    return pd.concat(rows, ignore_index=True)


def summarize_raw_section_rates(sections: pd.DataFrame) -> pd.DataFrame:
    """Summarize crude section-level sentiment rates by cohort and section."""
    rows = []
    outcomes = [
        "positive_vs_all",
        "negative_vs_all",
        "non_neutral_vs_all",
        "positive_vs_negative",
    ]
    for outcome in outcomes:
        denominator = sections if outcome != "positive_vs_negative" else sections.dropna(subset=[outcome])
        strata = [("all_sections", denominator), *denominator.groupby("section_name")]
        for stratum, data in strata:
            grouped = (
                data.groupby("cohort")[outcome]
                .agg(n_rows="size", n_events="sum", mean="mean")
                .reset_index()
            )
            grouped["summary_level"] = "section"
            grouped["stratum"] = str(stratum)
            grouped["outcome"] = outcome
            grouped["event_rate_pct"] = grouped["mean"] * 100.0
            rows.append(grouped.drop(columns=["mean"]))
    return pd.concat(rows, ignore_index=True)


def fit_admission_models(admissions: pd.DataFrame) -> pd.DataFrame:
    """Fit admission-level sentiment logistic models."""
    rows = []
    for outcome in ["any_positive", "any_negative", "any_non_neutral"]:
        for spec_name, predictors in MODEL_SPECS.items():
            rows.append(
                fit_logistic_model(
                    data=admissions,
                    outcome=outcome,
                    predictors=predictors,
                    model_level="admission",
                    stratum="overall",
                    model_spec=spec_name,
                    denominator="all_admissions",
                )
            )
    return pd.concat(rows, ignore_index=True)


def fit_negative_section_count_models(admissions: pd.DataFrame) -> pd.DataFrame:
    """Fit admission-level count models for negative sentiment section burden."""
    rows = []
    for spec_name, predictors in NEGATIVE_COUNT_MODEL_SPECS.items():
        rows.append(
            fit_poisson_count_model(
                data=admissions,
                outcome="n_negative_sections",
                predictors=predictors,
                model_level="admission",
                stratum="overall",
                model_spec=spec_name,
            )
        )
    return pd.concat(rows, ignore_index=True)


def fit_subgroup_admission_models(
    admissions: pd.DataFrame,
    subgroups: pd.DataFrame,
) -> pd.DataFrame:
    """Fit admission-level sentiment models within chief-complaint subgroups."""
    merged = admissions.merge(
        subgroups,
        on=ID_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    rows = []
    outcomes = ["any_positive", "any_negative", "any_non_neutral"]
    for flag_column, subgroup in CHIEF_COMPLAINT_SUBGROUPS.items():
        subgroup_data = merged.loc[merged[flag_column]].copy()
        for outcome in outcomes:
            for spec_name, predictors in MODEL_SPECS.items():
                rows.append(
                    fit_logistic_model(
                        data=subgroup_data,
                        outcome=outcome,
                        predictors=predictors,
                        model_level="admission_chief_complaint_subgroup",
                        stratum=subgroup,
                        model_spec=spec_name,
                        denominator="subgroup_admissions",
                    )
                )
    return pd.concat(rows, ignore_index=True)


def summarize_negative_section_count_correlations(admissions: pd.DataFrame) -> pd.DataFrame:
    """Summarize correlations between negative burden and readmission history."""
    rows = []
    readmission_columns = [
        "n_prior_admissions_within_365d_for_subject",
        "n_prior_all_admissions_for_subject",
        "log1p_prior_admissions_365d",
        "log1p_prior_all_mimic_admissions",
    ]
    strata = [("overall", admissions), *admissions.groupby("cohort")]
    for stratum, data in strata:
        for column in readmission_columns:
            complete = data.loc[:, ["n_negative_sections", column]].dropna()
            rows.append(
                {
                    "stratum": str(stratum),
                    "readmission_variable": column,
                    "n_admissions": len(complete),
                    "n_negative_sections_total": int(complete["n_negative_sections"].sum()),
                    "mean_n_negative_sections": complete["n_negative_sections"].mean(),
                    "mean_readmission_variable": complete[column].mean(),
                    "spearman_correlation": complete["n_negative_sections"].corr(
                        complete[column],
                        method="spearman",
                    )
                    if len(complete) > 1
                    else pd.NA,
                    "pearson_correlation": complete["n_negative_sections"].corr(
                        complete[column],
                        method="pearson",
                    )
                    if len(complete) > 1
                    else pd.NA,
                }
            )
    return pd.DataFrame(rows)


def fit_section_models(sections: pd.DataFrame) -> pd.DataFrame:
    """Fit pooled and section-specific section-level sentiment models."""
    rows = []
    pooled, section_dummy_columns = add_section_fixed_effects(sections)
    outcomes = [
        ("positive_vs_all", "all_sections"),
        ("negative_vs_all", "all_sections"),
        ("non_neutral_vs_all", "all_sections"),
        ("positive_vs_negative", "positive_or_negative_sections"),
    ]

    for outcome, denominator in outcomes:
        pooled_data = pooled if outcome != "positive_vs_negative" else pooled.dropna(subset=[outcome])
        for spec_name, predictors in MODEL_SPECS.items():
            rows.append(
                fit_logistic_model(
                    data=pooled_data,
                    outcome=outcome,
                    predictors=[*predictors, *section_dummy_columns],
                    model_level="section_pooled_with_section_fe",
                    stratum="all_sections",
                    model_spec=spec_name,
                    denominator=denominator,
                )
            )

        for section_name, section_data in sections.groupby("section_name"):
            section_data = (
                section_data
                if outcome != "positive_vs_negative"
                else section_data.dropna(subset=[outcome])
            )
            for spec_name, predictors in MODEL_SPECS.items():
                rows.append(
                    fit_logistic_model(
                        data=section_data,
                        outcome=outcome,
                        predictors=predictors,
                        model_level="section_specific",
                        stratum=str(section_name),
                        model_spec=spec_name,
                        denominator=denominator,
                    )
                )
    return pd.concat(rows, ignore_index=True)


def extract_cohort_term_summary(model_results: pd.DataFrame) -> pd.DataFrame:
    """Keep only the MHH1-vs-MHC0 coefficient for easier reading."""
    return model_results.loc[
        model_results["term"].eq("mhh1_psychotic"),
        [
            "model_level",
            "stratum",
            "model_spec",
            "denominator",
            "outcome",
            "n_rows",
            "n_events",
            "event_rate_pct",
            "n_clusters",
            "odds_ratio",
            "odds_ratio_ci_low",
            "odds_ratio_ci_high",
            "p_value",
            "status",
            "fit_message",
            "fit_warnings",
        ],
    ].copy()


def main() -> None:
    """Run sentiment regression analysis and write CSV outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    covariates = load_covariates()
    sections = attach_covariates(load_sentiment_sections(), covariates)
    admissions = attach_covariates(load_sentiment_admissions(), covariates)
    subgroups = load_chief_complaint_subgroups()

    section_model_columns = [
        "classifier_row_id",
        *ID_COLUMNS,
        "pair_id",
        "cluster_id",
        "section_name",
        "section_word_count",
        "sentiment_label",
        "positive_vs_all",
        "negative_vs_all",
        "non_neutral_vs_all",
        "positive_vs_negative",
        "mhh1_psychotic",
        "age_at_admission",
        "elixhauser_score",
        "n_prior_admissions_within_365d_for_subject",
        "n_prior_all_admissions_for_subject",
        "log1p_prior_admissions_365d",
        "log1p_prior_all_mimic_admissions",
    ]
    admission_model_columns = [
        *ID_COLUMNS,
        "pair_id",
        "cluster_id",
        "n_sections_classified",
        "n_positive_sections",
        "n_negative_sections",
        "n_mixed_sections",
        "any_positive",
        "any_negative",
        "any_mixed",
        "any_non_neutral",
        "mhh1_psychotic",
        "age_at_admission",
        "elixhauser_score",
        "n_prior_admissions_within_365d_for_subject",
        "n_prior_all_admissions_for_subject",
        "log1p_prior_admissions_365d",
        "log1p_prior_all_mimic_admissions",
    ]
    sections.loc[:, section_model_columns].to_csv(
        OUTPUT_DIR / "sentiment_regression_section_model_dataset.csv",
        index=False,
    )
    admissions.loc[:, admission_model_columns].to_csv(
        OUTPUT_DIR / "sentiment_regression_admission_model_dataset.csv",
        index=False,
    )

    raw_summary = pd.concat(
        [
            summarize_raw_admission_rates(admissions),
            summarize_raw_section_rates(sections),
        ],
        ignore_index=True,
        sort=False,
    )
    raw_summary.to_csv(OUTPUT_DIR / "sentiment_regression_raw_summary.csv", index=False)
    subgroup_raw_summary = summarize_raw_subgroup_admission_rates(admissions, subgroups)
    subgroup_raw_summary.to_csv(
        OUTPUT_DIR / "sentiment_admission_chief_complaint_subgroup_raw_summary.csv",
        index=False,
    )

    model_results = pd.concat(
        [
            fit_admission_models(admissions),
            fit_section_models(sections),
        ],
        ignore_index=True,
    )
    model_results.to_csv(OUTPUT_DIR / "sentiment_regression_models.csv", index=False)

    cohort_summary = extract_cohort_term_summary(model_results)
    cohort_summary.to_csv(
        OUTPUT_DIR / "sentiment_regression_cohort_term_summary.csv",
        index=False,
    )

    count_models = fit_negative_section_count_models(admissions)
    count_models.to_csv(
        OUTPUT_DIR / "sentiment_negative_section_count_models.csv",
        index=False,
    )
    negative_correlations = summarize_negative_section_count_correlations(admissions)
    negative_correlations.to_csv(
        OUTPUT_DIR / "sentiment_negative_section_count_correlations.csv",
        index=False,
    )

    subgroup_models = fit_subgroup_admission_models(admissions, subgroups)
    subgroup_models.to_csv(
        OUTPUT_DIR / "sentiment_admission_chief_complaint_subgroup_models.csv",
        index=False,
    )
    extract_cohort_term_summary(subgroup_models).to_csv(
        OUTPUT_DIR / "sentiment_admission_chief_complaint_subgroup_cohort_term_summary.csv",
        index=False,
    )

    print(f"Wrote sentiment regression outputs to {OUTPUT_DIR}")
    print(f"Admission rows: {len(admissions):,}")
    print(f"Section rows: {len(sections):,}")
    print("Cohort-term rows:")
    print(
        cohort_summary.loc[
            :,
            [
                "model_level",
                "stratum",
                "model_spec",
                "outcome",
                "odds_ratio",
                "p_value",
                "status",
            ],
        ]
        .head(20)
        .to_string(index=False)
    )
    print("Negative-section count model cohort/readmission terms:")
    print(
        count_models.loc[
            count_models["term"].isin(
                [
                    "mhh1_psychotic",
                    "log1p_prior_admissions_365d",
                    "log1p_prior_all_mimic_admissions",
                ]
            ),
            [
                "model_spec",
                "term",
                "rate_ratio",
                "rate_ratio_ci_low",
                "rate_ratio_ci_high",
                "p_value",
                "status",
            ],
        ].to_string(index=False)
    )
    print("Chief-complaint subgroup admission-level cohort terms:")
    print(
        extract_cohort_term_summary(subgroup_models)
        .loc[
            lambda data: data["model_spec"].eq("age_elixhauser_prior_all_mimic"),
            [
                "stratum",
                "outcome",
                "n_rows",
                "n_events",
                "odds_ratio",
                "odds_ratio_ci_low",
                "odds_ratio_ci_high",
                "p_value",
                "status",
            ],
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
