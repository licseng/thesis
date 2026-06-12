"""Analyze matched cohort outputs.

This script creates review tables from the matched cohort produced by
`03_match_chief_complaint_cohorts.py`. The main review file lists the matched
pairs with the lowest chief-complaint cosine similarity so they can be inspected
manually without mixing review artifacts into the core matching output folder.

Inputs:
    matched_cohort_output/matched_pairs.parquet
    matched_cohort_output/matching_summary.csv

Outputs:
    analysis_output_matched_cohort/match_quality_summary.csv
    analysis_output_matched_cohort/match_type_counts.csv
    analysis_output_matched_cohort/elixhauser_score_summary.csv
    analysis_output_matched_cohort/elixhauser_difference_counts.csv
    analysis_output_matched_cohort/elixhauser_score_pair_counts.csv
    analysis_output_matched_cohort/age_bin_distance_counts.csv
    analysis_output_matched_cohort/age_year_distance_summary.csv
    analysis_output_matched_cohort/age_year_distance_counts.csv
    analysis_output_matched_cohort/lowest_cosine_similarity_pairs_review.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
MATCHED_OUTPUT_DIR = SCRIPT_DIR / "matched_cohort_output"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_matched_cohort"

MATCHED_PAIRS_PATH = MATCHED_OUTPUT_DIR / "matched_pairs.parquet"
MATCHING_SUMMARY_PATH = MATCHED_OUTPUT_DIR / "matching_summary.csv"

# Number of low-similarity matched pairs to export for manual review.
LOWEST_COSINE_REVIEW_N = 100

# Columns expected in the matched-pairs output.
REQUIRED_COLUMNS = {
    "pair_id",
    "mhh_subject_id",
    "mhh_hadm_id",
    "mhh_chief_complaint_raw",
    "mhh_chief_complaint_normalized",
    "mhh_sex",
    "mhh_age_at_admission",
    "mhh_age_bin",
    "mhh_elixhauser_score",
    "mhc0_subject_id",
    "mhc0_hadm_id",
    "mhc0_chief_complaint_raw",
    "mhc0_chief_complaint_normalized",
    "mhc0_sex",
    "mhc0_age_at_admission",
    "mhc0_age_bin",
    "mhc0_elixhauser_score",
    "cosine_similarity",
    "embedding_distance",
    "abs_elixhauser_difference",
    "match_type",
    "candidate_pool_size",
}

# Local review columns. These include raw complaints because the file is meant
# for manual review; the script does not print those values to the terminal.
LOWEST_COSINE_REVIEW_COLUMNS = [
    "pair_id",
    "cosine_similarity",
    "embedding_distance",
    "match_type",
    "abs_elixhauser_difference",
    "quickumls_shared_term_count",
    "quickumls_jaccard",
    "used_quickumls_candidate_filter",
    "used_quickumls_filter_fallback",
    "candidate_pool_size",
    "mhh_subject_id",
    "mhh_hadm_id",
    "mhh_chief_complaint_raw",
    "mhh_quickumls_terms",
    "mhh_quickumls_extracted_text",
    "mhh_elixhauser_score",
    "mhc0_subject_id",
    "mhc0_hadm_id",
    "mhc0_chief_complaint_raw",
    "mhc0_quickumls_terms",
    "mhc0_quickumls_extracted_text",
    "mhc0_elixhauser_score",
]


# Load matched pairs and validate the expected schema.
def load_matched_pairs() -> pd.DataFrame:
    if not MATCHED_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing matched-pairs parquet: {MATCHED_PAIRS_PATH}")

    df = pd.read_parquet(MATCHED_PAIRS_PATH)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"{MATCHED_PAIRS_PATH.name} is missing required columns: {missing}"
        )
    return df


# Load the matching summary if present. The analysis can still run without it,
# because all review tables are based on `matched_pairs.parquet`.
def load_matching_summary() -> pd.DataFrame:
    if not MATCHING_SUMMARY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MATCHING_SUMMARY_PATH)


# Standardized mean difference for paired exposed/control numeric columns.
def standardized_mean_difference(exposed: pd.Series, control: pd.Series) -> float:
    exposed = pd.to_numeric(exposed, errors="coerce")
    control = pd.to_numeric(control, errors="coerce")
    pooled_sd = ((exposed.var(ddof=1) + control.var(ddof=1)) / 2) ** 0.5
    if pd.isna(pooled_sd) or pooled_sd == 0:
        return 0.0
    return float((exposed.mean() - control.mean()) / pooled_sd)


# Build a one-row summary of match quality and covariate balance.
def build_match_quality_summary(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    cosine_q1 = matched_pairs["cosine_similarity"].quantile(0.25)
    cosine_q3 = matched_pairs["cosine_similarity"].quantile(0.75)
    elix_q1 = matched_pairs["abs_elixhauser_difference"].quantile(0.25)
    elix_q3 = matched_pairs["abs_elixhauser_difference"].quantile(0.75)

    summary = {
        "n_matched_pairs": len(matched_pairs),
        "median_cosine_similarity": matched_pairs["cosine_similarity"].median(),
        "iqr_cosine_similarity": cosine_q3 - cosine_q1,
        "min_cosine_similarity": matched_pairs["cosine_similarity"].min(),
        "median_embedding_distance": matched_pairs["embedding_distance"].median(),
        "median_abs_elixhauser_difference": matched_pairs[
            "abs_elixhauser_difference"
        ].median(),
        "iqr_abs_elixhauser_difference": elix_q3 - elix_q1,
        "max_abs_elixhauser_difference": matched_pairs[
            "abs_elixhauser_difference"
        ].max(),
        "median_abs_age_difference": matched_pairs["abs_age_difference"].median(),
        "iqr_abs_age_difference": (
            matched_pairs["abs_age_difference"].quantile(0.75)
            - matched_pairs["abs_age_difference"].quantile(0.25)
        ),
        "max_abs_age_difference": matched_pairs["abs_age_difference"].max(),
        "mean_mhh_age": matched_pairs["mhh_age_at_admission"].mean(),
        "mean_mhc0_age": matched_pairs["mhc0_age_at_admission"].mean(),
        "smd_age": standardized_mean_difference(
            matched_pairs["mhh_age_at_admission"],
            matched_pairs["mhc0_age_at_admission"],
        ),
        "mean_mhh_elixhauser": matched_pairs["mhh_elixhauser_score"].mean(),
        "mean_mhc0_elixhauser": matched_pairs["mhc0_elixhauser_score"].mean(),
        "smd_elixhauser": standardized_mean_difference(
            matched_pairs["mhh_elixhauser_score"],
            matched_pairs["mhc0_elixhauser_score"],
        ),
        "n_same_sex_pairs": int(
            (matched_pairs["mhh_sex"] == matched_pairs["mhc0_sex"]).sum()
        ),
        "n_same_age_bin_pairs": int(
            (matched_pairs["mhh_age_bin"] == matched_pairs["mhc0_age_bin"]).sum()
        ),
    }
    return pd.DataFrame([summary])


# Count strict/relaxed match types.
def build_match_type_counts(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    counts = (
        matched_pairs.groupby("match_type")
        .agg(n_pairs=("pair_id", "size"))
        .reset_index()
        .sort_values(["n_pairs", "match_type"], ascending=[False, True])
    )
    counts["pct_pairs"] = 100.0 * counts["n_pairs"] / len(matched_pairs)
    return counts


# Summarize exposed/control Elixhauser distributions and balance after matching.
def build_elixhauser_score_summary(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort_label, column in [
        ("MHH_psychotic", "mhh_elixhauser_score"),
        ("only_MHC0", "mhc0_elixhauser_score"),
    ]:
        scores = pd.to_numeric(matched_pairs[column], errors="coerce")
        rows.append(
            {
                "cohort": cohort_label,
                "n_pairs": len(scores),
                "mean_elixhauser_score": scores.mean(),
                "sd_elixhauser_score": scores.std(ddof=1),
                "median_elixhauser_score": scores.median(),
                "q1_elixhauser_score": scores.quantile(0.25),
                "q3_elixhauser_score": scores.quantile(0.75),
                "min_elixhauser_score": scores.min(),
                "max_elixhauser_score": scores.max(),
                "n_score_zero": int(scores.eq(0).sum()),
                "pct_score_zero": 100.0 * scores.eq(0).mean(),
                "n_score_positive": int(scores.gt(0).sum()),
                "pct_score_positive": 100.0 * scores.gt(0).mean(),
                "n_score_10_or_higher": int(scores.ge(10).sum()),
                "pct_score_10_or_higher": 100.0 * scores.ge(10).mean(),
                "n_score_20_or_higher": int(scores.ge(20).sum()),
                "pct_score_20_or_higher": 100.0 * scores.ge(20).mean(),
            }
        )

    exposed = pd.to_numeric(matched_pairs["mhh_elixhauser_score"], errors="coerce")
    control = pd.to_numeric(matched_pairs["mhc0_elixhauser_score"], errors="coerce")
    rows.append(
        {
            "cohort": "balance_MHH_minus_MHC0",
            "n_pairs": len(matched_pairs),
            "mean_elixhauser_score": exposed.mean() - control.mean(),
            "sd_elixhauser_score": pd.NA,
            "median_elixhauser_score": (
                exposed.median() - control.median()
            ),
            "q1_elixhauser_score": pd.NA,
            "q3_elixhauser_score": pd.NA,
            "min_elixhauser_score": pd.NA,
            "max_elixhauser_score": pd.NA,
            "n_score_zero": pd.NA,
            "pct_score_zero": pd.NA,
            "n_score_positive": pd.NA,
            "pct_score_positive": pd.NA,
            "n_score_10_or_higher": pd.NA,
            "pct_score_10_or_higher": pd.NA,
            "n_score_20_or_higher": pd.NA,
            "pct_score_20_or_higher": pd.NA,
            "smd_elixhauser": standardized_mean_difference(exposed, control),
        }
    )
    return pd.DataFrame(rows)


# Count how closely matched pairs align on the Elixhauser score.
def build_elixhauser_difference_counts(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    bins = [-0.1, 0, 1, 2, 5, 10, 20, float("inf")]
    labels = ["0", "1", "2", "3-5", "6-10", "11-20", ">20"]
    counts = matched_pairs.copy()
    counts["abs_elixhauser_difference_bin"] = pd.cut(
        counts["abs_elixhauser_difference"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    counts = (
        counts.groupby("abs_elixhauser_difference_bin", observed=False)
        .agg(n_pairs=("pair_id", "size"))
        .reset_index()
    )
    counts["pct_pairs"] = 100.0 * counts["n_pairs"] / len(matched_pairs)
    return counts


# Cross-tab exposed/control Elixhauser scores to inspect exact score pairings.
def build_elixhauser_score_pair_counts(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    counts = (
        matched_pairs.groupby(["mhh_elixhauser_score", "mhc0_elixhauser_score"])
        .agg(n_pairs=("pair_id", "size"))
        .reset_index()
        .sort_values(
            ["n_pairs", "mhh_elixhauser_score", "mhc0_elixhauser_score"],
            ascending=[False, True, True],
        )
    )
    counts["pct_pairs"] = 100.0 * counts["n_pairs"] / len(matched_pairs)
    counts["elixhauser_score_difference"] = (
        counts["mhh_elixhauser_score"] - counts["mhc0_elixhauser_score"]
    )
    counts["abs_elixhauser_score_difference"] = counts[
        "elixhauser_score_difference"
    ].abs()
    return counts


# Count how often controls were selected from the same or neighboring age bins.
def build_age_bin_distance_counts(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    if "age_bin_distance" not in matched_pairs.columns:
        return pd.DataFrame()

    counts = (
        matched_pairs.groupby("age_bin_distance")
        .agg(n_pairs=("pair_id", "size"))
        .reset_index()
        .sort_values("age_bin_distance")
    )
    counts["pct_pairs"] = 100.0 * counts["n_pairs"] / len(matched_pairs)
    return counts


# Summarize actual exposed-control age distance in years.
def build_age_year_distance_summary(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    age_distance = matched_pairs["abs_age_difference"]
    return pd.DataFrame(
        [
            {
                "n_pairs": len(matched_pairs),
                "mean_abs_age_difference": age_distance.mean(),
                "sd_abs_age_difference": age_distance.std(ddof=1),
                "median_abs_age_difference": age_distance.median(),
                "q1_abs_age_difference": age_distance.quantile(0.25),
                "q3_abs_age_difference": age_distance.quantile(0.75),
                "min_abs_age_difference": age_distance.min(),
                "max_abs_age_difference": age_distance.max(),
                "n_exact_same_age": int(age_distance.eq(0).sum()),
                "pct_exact_same_age": 100.0 * age_distance.eq(0).mean(),
                "n_within_1_year": int(age_distance.le(1).sum()),
                "pct_within_1_year": 100.0 * age_distance.le(1).mean(),
                "n_within_2_years": int(age_distance.le(2).sum()),
                "pct_within_2_years": 100.0 * age_distance.le(2).mean(),
                "n_within_5_years": int(age_distance.le(5).sum()),
                "pct_within_5_years": 100.0 * age_distance.le(5).mean(),
                "n_within_10_years": int(age_distance.le(10).sum()),
                "pct_within_10_years": 100.0 * age_distance.le(10).mean(),
            }
        ]
    )


# Count matched pairs by actual age-distance bins in years.
def build_age_year_distance_counts(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    bins = [-0.1, 0, 1, 2, 5, 10, 20, float("inf")]
    labels = [
        "0",
        "1",
        "2",
        "3-5",
        "6-10",
        "11-20",
        ">20",
    ]
    counts = matched_pairs.copy()
    counts["abs_age_difference_bin"] = pd.cut(
        counts["abs_age_difference"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    counts = (
        counts.groupby("abs_age_difference_bin", observed=False)
        .agg(n_pairs=("pair_id", "size"))
        .reset_index()
    )
    counts["pct_pairs"] = 100.0 * counts["n_pairs"] / len(matched_pairs)
    return counts


# Export the lowest-cosine matched pairs for manual review.
def build_lowest_cosine_review(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in LOWEST_COSINE_REVIEW_COLUMNS
        if column in matched_pairs.columns
    ]
    return (
        matched_pairs.sort_values(
            ["cosine_similarity", "abs_elixhauser_difference", "pair_id"],
            ascending=[True, True, True],
        )
        .head(LOWEST_COSINE_REVIEW_N)
        .loc[:, columns]
        .copy()
    )


# Write all analysis outputs.
def write_outputs(
    match_quality_summary: pd.DataFrame,
    match_type_counts: pd.DataFrame,
    elixhauser_score_summary: pd.DataFrame,
    elixhauser_difference_counts: pd.DataFrame,
    elixhauser_score_pair_counts: pd.DataFrame,
    age_bin_distance_counts: pd.DataFrame,
    age_year_distance_summary: pd.DataFrame,
    age_year_distance_counts: pd.DataFrame,
    lowest_cosine_review: pd.DataFrame,
) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    outputs = {
        "match_quality_summary": OUTPUT_DIR / "match_quality_summary.csv",
        "match_type_counts": OUTPUT_DIR / "match_type_counts.csv",
        "elixhauser_score_summary": OUTPUT_DIR / "elixhauser_score_summary.csv",
        "elixhauser_difference_counts": OUTPUT_DIR
        / "elixhauser_difference_counts.csv",
        "elixhauser_score_pair_counts": OUTPUT_DIR
        / "elixhauser_score_pair_counts.csv",
        "age_bin_distance_counts": OUTPUT_DIR / "age_bin_distance_counts.csv",
        "age_year_distance_summary": OUTPUT_DIR / "age_year_distance_summary.csv",
        "age_year_distance_counts": OUTPUT_DIR / "age_year_distance_counts.csv",
        "lowest_cosine_review": OUTPUT_DIR / "lowest_cosine_similarity_pairs_review.csv",
    }
    match_quality_summary.to_csv(outputs["match_quality_summary"], index=False)
    match_type_counts.to_csv(outputs["match_type_counts"], index=False)
    elixhauser_score_summary.to_csv(
        outputs["elixhauser_score_summary"], index=False
    )
    elixhauser_difference_counts.to_csv(
        outputs["elixhauser_difference_counts"], index=False
    )
    elixhauser_score_pair_counts.to_csv(
        outputs["elixhauser_score_pair_counts"], index=False
    )
    age_bin_distance_counts.to_csv(outputs["age_bin_distance_counts"], index=False)
    age_year_distance_summary.to_csv(outputs["age_year_distance_summary"], index=False)
    age_year_distance_counts.to_csv(outputs["age_year_distance_counts"], index=False)
    lowest_cosine_review.to_csv(outputs["lowest_cosine_review"], index=False)
    return outputs


# Script entry point: summarize matched pairs and write local review artifacts.
def main() -> None:
    matched_pairs = load_matched_pairs()
    matching_summary = load_matching_summary()

    match_quality_summary = build_match_quality_summary(matched_pairs)
    match_type_counts = build_match_type_counts(matched_pairs)
    elixhauser_score_summary = build_elixhauser_score_summary(matched_pairs)
    elixhauser_difference_counts = build_elixhauser_difference_counts(matched_pairs)
    elixhauser_score_pair_counts = build_elixhauser_score_pair_counts(matched_pairs)
    age_bin_distance_counts = build_age_bin_distance_counts(matched_pairs)
    age_year_distance_summary = build_age_year_distance_summary(matched_pairs)
    age_year_distance_counts = build_age_year_distance_counts(matched_pairs)
    lowest_cosine_review = build_lowest_cosine_review(matched_pairs)

    outputs = write_outputs(
        match_quality_summary,
        match_type_counts,
        elixhauser_score_summary,
        elixhauser_difference_counts,
        elixhauser_score_pair_counts,
        age_bin_distance_counts,
        age_year_distance_summary,
        age_year_distance_counts,
        lowest_cosine_review,
    )

    print("\n=== Match Quality Summary ===")
    print(match_quality_summary.to_string(index=False))
    print("\n=== Age Distance In Years ===")
    print(age_year_distance_summary.to_string(index=False))
    print("\n=== Elixhauser Score Summary ===")
    print(elixhauser_score_summary.to_string(index=False))
    print("\n=== Elixhauser Absolute Difference Counts ===")
    print(elixhauser_difference_counts.to_string(index=False))
    if not matching_summary.empty:
        print("\n=== Original Matching Summary ===")
        print(matching_summary.to_string(index=False))
    print("\nSaved analysis outputs:")
    for path in outputs.values():
        print(f"  {path}")


# Allow the script to be run directly with:
#     python 04_analyze_matched_cohort.py
if __name__ == "__main__":
    main()
