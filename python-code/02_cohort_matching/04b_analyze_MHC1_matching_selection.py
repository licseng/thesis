"""Compare matched and unmatched MHC1-psychosis admissions.

This is a selection analysis, not another matching procedure. It determines
whether the MHC1 admissions retained in the matched analytic cohort differ from
the roughly 10% that could not be matched. Outputs are aggregate-only and do
not include chief-complaint or discharge-note text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MATCHING_VARIABLES_PATH = (
    SCRIPT_DIR
    / "02_matching_variables"
    / "matching_variable_tables_output"
    / "MHC1_psychotic_matching_variables.parquet"
)
MATCHED_PAIRS_PATH = SCRIPT_DIR / "matched_cohort_output" / "matched_pairs.parquet"
UNMATCHED_PATH = (
    SCRIPT_DIR / "matched_cohort_output" / "unmatched_MHC1_psychotic.parquet"
)
CONTEXT_PATH = (
    SCRIPT_DIR.parent
    / "01_discharge_note_preprocessing"
    / "01_discharge_note_parsing"
    / "parsed_chief_complaints"
    / "MHC1_psychotic_chief_complaints_from_discharge_notes.parquet"
)
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_MHC1_matching_selection"


def split_terms(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        term.strip().casefold()
        for term in str(value).split("|")
        if term.strip()
    }


def continuous_smd(matched: pd.Series, unmatched: pd.Series) -> float:
    matched = pd.to_numeric(matched, errors="coerce").dropna()
    unmatched = pd.to_numeric(unmatched, errors="coerce").dropna()
    pooled_sd = np.sqrt((matched.var(ddof=1) + unmatched.var(ddof=1)) / 2)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan
    return float((matched.mean() - unmatched.mean()) / pooled_sd)


def binary_smd(p_matched: float, p_unmatched: float) -> float:
    denominator = np.sqrt(
        (p_matched * (1 - p_matched) + p_unmatched * (1 - p_unmatched)) / 2
    )
    if denominator == 0:
        return 0.0 if p_matched == p_unmatched else np.nan
    return float((p_matched - p_unmatched) / denominator)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in ["age_at_admission", "elixhauser_score", "quickumls_term_count"]:
        groups = {
            status: pd.to_numeric(group[variable], errors="coerce").dropna()
            for status, group in df.groupby("match_status")
        }
        smd = continuous_smd(groups["matched"], groups["unmatched"])
        for status, values in groups.items():
            rows.append(
                {
                    "variable": variable,
                    "match_status": status,
                    "n_nonmissing": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "min": values.min(),
                    "max": values.max(),
                    "smd_matched_minus_unmatched": smd,
                }
            )
    return pd.DataFrame(rows)


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variable in ["sex", "insurance_group", "psychosis_context_version"]:
        values = df[variable].fillna("missing").astype(str)
        levels = sorted(values.unique())
        for level in levels:
            proportions: dict[str, float] = {}
            counts: dict[str, int] = {}
            for status in ["matched", "unmatched"]:
                status_mask = df["match_status"].eq(status)
                denominator = int(status_mask.sum())
                count = int((status_mask & values.eq(level)).sum())
                counts[status] = count
                proportions[status] = count / denominator if denominator else 0.0
            smd = binary_smd(proportions["matched"], proportions["unmatched"])
            for status in ["matched", "unmatched"]:
                rows.append(
                    {
                        "variable": variable,
                        "level": level,
                        "match_status": status,
                        "n_admissions": counts[status],
                        "pct_admissions": 100 * proportions[status],
                        "level_smd_matched_minus_unmatched": smd,
                    }
                )
    return pd.DataFrame(rows)


def term_comparison(df: pd.DataFrame) -> pd.DataFrame:
    term_rows: list[dict[str, object]] = []
    for row in df[["match_status", "hadm_id", "quickumls_term_list"]].itertuples(
        index=False
    ):
        for term in row.quickumls_term_list:
            term_rows.append(
                {
                    "match_status": row.match_status,
                    "hadm_id": row.hadm_id,
                    "quickumls_term": term,
                }
            )
    exploded = pd.DataFrame(term_rows).drop_duplicates()
    counts = (
        exploded.groupby(["quickumls_term", "match_status"])["hadm_id"]
        .nunique()
        .unstack(fill_value=0)
    )
    for status in ["matched", "unmatched"]:
        if status not in counts.columns:
            counts[status] = 0
    denominators = df.groupby("match_status")["hadm_id"].nunique().to_dict()
    output = counts.reset_index()
    output["matched_pct"] = 100 * output["matched"] / denominators["matched"]
    output["unmatched_pct"] = 100 * output["unmatched"] / denominators["unmatched"]
    output["percentage_point_difference"] = (
        output["matched_pct"] - output["unmatched_pct"]
    )
    output["absolute_percentage_point_difference"] = output[
        "percentage_point_difference"
    ].abs()
    output["overall_admissions"] = output["matched"] + output["unmatched"]
    return output.sort_values(
        ["absolute_percentage_point_difference", "overall_admissions"],
        ascending=[False, False],
    )


def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    per_subject_status = (
        df.groupby(["match_status", "subject_id"])["hadm_id"]
        .nunique()
        .rename("n_admissions")
        .reset_index()
    )
    rows = []
    for status, group in per_subject_status.groupby("match_status"):
        values = group["n_admissions"]
        rows.append(
            {
                "match_status": status,
                "n_subjects": group["subject_id"].nunique(),
                "n_admissions": int(values.sum()),
                "mean_admissions_per_subject": values.mean(),
                "median_admissions_per_subject": values.median(),
                "q1_admissions_per_subject": values.quantile(0.25),
                "q3_admissions_per_subject": values.quantile(0.75),
                "max_admissions_per_subject": values.max(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    variables = pd.read_parquet(MATCHING_VARIABLES_PATH)
    matched_pairs = pd.read_parquet(
        MATCHED_PAIRS_PATH,
        columns=["mhc1_subject_id", "mhc1_hadm_id"],
    ).rename(
        columns={"mhc1_subject_id": "subject_id", "mhc1_hadm_id": "hadm_id"}
    )
    unmatched = pd.read_parquet(
        UNMATCHED_PATH,
        columns=["subject_id", "hadm_id", "unmatched_reason"],
    )
    context = pd.read_parquet(
        CONTEXT_PATH,
        columns=["subject_id", "hadm_id", "psychosis_context_version"],
    )

    if matched_pairs["hadm_id"].duplicated().any():
        raise ValueError("Matched MHC1 admission IDs are not unique.")
    if unmatched["hadm_id"].duplicated().any():
        raise ValueError("Unmatched MHC1 admission IDs are not unique.")
    overlap = set(matched_pairs["hadm_id"]) & set(unmatched["hadm_id"])
    if overlap:
        raise ValueError("Matched and unmatched MHC1 admission sets overlap.")

    status = pd.concat(
        [
            matched_pairs.assign(match_status="matched", unmatched_reason=pd.NA),
            unmatched.assign(match_status="unmatched"),
        ],
        ignore_index=True,
    )
    if status["hadm_id"].nunique() != len(variables):
        raise ValueError("Matched plus unmatched admissions do not cover the MHC1 pool.")

    df = variables.merge(
        status,
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="one_to_one",
    ).merge(
        context,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )
    df["quickumls_term_list"] = df["quickumls_terms"].map(split_terms)
    df["quickumls_term_count"] = df["quickumls_term_list"].map(len)

    selection_summary = (
        df.groupby("match_status")
        .agg(n_admissions=("hadm_id", "nunique"), n_subjects=("subject_id", "nunique"))
        .reset_index()
    )
    selection_summary["pct_of_eligible_admissions"] = (
        100 * selection_summary["n_admissions"] / df["hadm_id"].nunique()
    )
    reason_summary = (
        df.loc[df["match_status"].eq("unmatched")]
        .groupby("unmatched_reason", dropna=False)
        .agg(n_admissions=("hadm_id", "nunique"), n_subjects=("subject_id", "nunique"))
        .reset_index()
    )
    reason_summary["pct_unmatched_admissions"] = (
        100 * reason_summary["n_admissions"] / reason_summary["n_admissions"].sum()
    )

    matched_subjects = set(df.loc[df["match_status"].eq("matched"), "subject_id"])
    unmatched_subjects = set(df.loc[df["match_status"].eq("unmatched"), "subject_id"])
    subject_overlap = pd.DataFrame(
        [
            {
                "n_subjects_with_matched_admission": len(matched_subjects),
                "n_subjects_with_unmatched_admission": len(unmatched_subjects),
                "n_subjects_in_both_sets": len(matched_subjects & unmatched_subjects),
                "n_subjects_only_unmatched": len(unmatched_subjects - matched_subjects),
            }
        ]
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    outputs = {
        "selection_summary.csv": selection_summary,
        "unmatched_reason_summary.csv": reason_summary,
        "numeric_summary.csv": numeric_summary(df),
        "categorical_summary.csv": categorical_summary(df),
        "quickumls_term_comparison.csv": term_comparison(df),
        "subject_admission_summary.csv": subject_summary(df),
        "subject_overlap_summary.csv": subject_overlap,
    }
    for filename, output in outputs.items():
        output.to_csv(OUTPUT_DIR / filename, index=False)

    print("=== MHC1 matching-selection analysis ===")
    print(selection_summary.to_string(index=False))
    print("\n=== Unmatched reasons ===")
    print(reason_summary.to_string(index=False))
    print("\n=== Numeric comparison ===")
    print(outputs["numeric_summary.csv"].to_string(index=False))
    print("\n=== Subject overlap ===")
    print(subject_overlap.to_string(index=False))
    print(f"\nSaved aggregate outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
