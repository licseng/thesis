"""Summarize aggregate outputs from the psych-history classifier.

This script is for whole-run classifier outputs. It reads section-level
classifier results, admission-level summaries, and the keyword-prefilter
metadata used as the classifier input denominator.

It writes aggregate summaries only. It deliberately does not write raw note
text, evidence spans, reasons, patient IDs, or row-level classifier outputs.

Set PSYCH_HISTORY_SUMMARY_INPUT_DIR to summarize a different classifier output
folder. By default, the script uses the completed prompt B cluster output if it
is present.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_OUTPUT_BASE = (
    SCRIPT_DIR / "psych_history_classifier_cluster_outputs_psych_integrated"
)
DEFAULT_OUTPUT_DIR = (
    CLUSTER_OUTPUT_BASE / "psych_history_classifier_output_prompt_B_all"
)
FALLBACK_OUTPUT_DIR = SCRIPT_DIR / "psych_history_classifier_output_prompt_B"
CLASSIFIER_OUTPUT_DIR = Path(
    os.environ.get(
        "PSYCH_HISTORY_SUMMARY_INPUT_DIR",
        str(DEFAULT_OUTPUT_DIR if DEFAULT_OUTPUT_DIR.exists() else FALLBACK_OUTPUT_DIR),
    )
)
SUMMARY_OUTPUT_DIR = Path(
    os.environ.get(
        "PSYCH_HISTORY_SUMMARY_OUTPUT_DIR",
        str(CLASSIFIER_OUTPUT_DIR / "summary_output"),
    )
)

FILTER_SUMMARY_PATH = (
    SCRIPT_DIR / "psych_history_llm_input" / "filtered_psych_keyword_filter_summary.csv"
)
FILTER_METADATA_PATH = (
    SCRIPT_DIR / "psych_history_llm_input" / "filtered_psych_keyword_section_input_metadata.csv"
)
FILTER_ADMISSION_PATH = (
    SCRIPT_DIR / "psych_history_llm_input" / "filtered_psych_keyword_admission_summary.csv"
)

SECTION_RESULTS_PATH = CLASSIFIER_OUTPUT_DIR / "psych_history_section_classifier_results.csv"
ADMISSION_RESULTS_PATH = CLASSIFIER_OUTPUT_DIR / "psych_history_admission_summary.csv"


def safe_pct(numerator: int | float, denominator: int | float) -> float:
    """Return a percentage, with zero for empty denominators."""
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def load_classifier_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load final classifier outputs and fail if only checkpoints are present."""
    if not CLASSIFIER_OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Missing classifier output folder: {CLASSIFIER_OUTPUT_DIR}")
    if not SECTION_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing final section output: {SECTION_RESULTS_PATH}")
    if not ADMISSION_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing final admission output: {ADMISSION_RESULTS_PATH}")

    section_results = pd.read_csv(SECTION_RESULTS_PATH)
    admission_results = pd.read_csv(ADMISSION_RESULTS_PATH)
    required_section_columns = {
        "classifier_row_id",
        "cohort",
        "subject_id",
        "hadm_id",
        "section_name",
        "section_word_count",
        "n_psych_keyword_hits",
        "psych_keyword_groups",
        "psychiatric_context_label",
        "psychiatric_mention_type",
        "n_chunks",
        "n_positive_chunks",
        "json_recovered_from_partial_response",
    }
    missing = required_section_columns - set(section_results.columns)
    if missing:
        raise ValueError(
            "Section output is missing expected column(s): "
            + ", ".join(sorted(missing))
        )
    if "any_positive" not in admission_results.columns:
        raise ValueError("Admission output is missing expected column: any_positive")
    return section_results, admission_results


def load_filter_metadata() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load keyword-prefilter denominators used before LLM classification."""
    missing = [
        path
        for path in [FILTER_SUMMARY_PATH, FILTER_METADATA_PATH, FILTER_ADMISSION_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing keyword-prefilter file(s):\n"
            + "\n".join(str(path) for path in missing)
        )

    filter_summary = pd.read_csv(FILTER_SUMMARY_PATH)
    filter_metadata = pd.read_csv(FILTER_METADATA_PATH)
    filter_admissions = pd.read_csv(FILTER_ADMISSION_PATH)
    return filter_summary, filter_metadata, filter_admissions


def build_completeness_summary(
    section_results: pd.DataFrame,
    chunk_results_available: bool,
    filter_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Check whether final classifier output covers every filtered input row."""
    expected_ids = set(filter_metadata["classifier_row_id"].astype(int))
    observed_ids = set(section_results["classifier_row_id"].astype(int))
    missing_ids = expected_ids - observed_ids
    extra_ids = observed_ids - expected_ids
    return pd.DataFrame(
        [
            {
                "classifier_output_dir": str(CLASSIFIER_OUTPUT_DIR),
                "n_expected_filtered_sections": len(expected_ids),
                "n_final_section_rows": len(section_results),
                "n_unique_final_section_ids": section_results[
                    "classifier_row_id"
                ].nunique(),
                "n_missing_expected_section_ids": len(missing_ids),
                "n_extra_section_ids_not_in_input": len(extra_ids),
                "chunk_result_file_available": chunk_results_available,
            }
        ]
    )


def build_overall_summary(
    section_results: pd.DataFrame,
    admission_results: pd.DataFrame,
    filter_summary: pd.DataFrame,
    filter_admissions: pd.DataFrame,
) -> pd.DataFrame:
    """Build one-row whole-run classifier summary."""
    overall_filter = filter_summary.loc[
        filter_summary["section_name"].eq("any_selected_section")
    ]
    if overall_filter.empty:
        raise ValueError("Filter summary has no any_selected_section row.")
    overall_filter_row = overall_filter.iloc[0]

    n_filtered_admissions = int(
        overall_filter_row["n_admissions_with_keyword_positive_section"]
    )
    n_all_mhh1_admissions = int(overall_filter_row["n_admissions"])
    n_filtered_sections = int(overall_filter_row["n_keyword_positive_section_rows"])
    n_classified_sections = len(section_results)
    n_classified_admissions = len(admission_results)
    n_positive_sections = int(
        section_results["psychiatric_context_label"].eq("positive").sum()
    )
    n_negative_sections = int(
        section_results["psychiatric_context_label"].eq("negative").sum()
    )
    n_positive_admissions = int(admission_results["any_positive"].sum())
    n_negative_admissions = int((~admission_results["any_positive"].astype(bool)).sum())
    n_recovered_json_sections = int(
        section_results["json_recovered_from_partial_response"].fillna(False).sum()
    )

    return pd.DataFrame(
        [
            {
                "n_all_mhh1_admissions": n_all_mhh1_admissions,
                "n_keyword_prefilter_positive_admissions": n_filtered_admissions,
                "n_keyword_prefilter_positive_sections": n_filtered_sections,
                "n_classified_admissions": n_classified_admissions,
                "n_classified_sections": n_classified_sections,
                "n_positive_sections": n_positive_sections,
                "n_negative_sections": n_negative_sections,
                "pct_positive_sections_of_classified_sections": safe_pct(
                    n_positive_sections,
                    n_classified_sections,
                ),
                "n_positive_admissions": n_positive_admissions,
                "n_negative_admissions": n_negative_admissions,
                "pct_positive_admissions_of_classified_admissions": safe_pct(
                    n_positive_admissions,
                    n_classified_admissions,
                ),
                "pct_positive_admissions_of_keyword_prefilter_admissions": safe_pct(
                    n_positive_admissions,
                    n_filtered_admissions,
                ),
                "pct_positive_admissions_of_all_mhh1_admissions": safe_pct(
                    n_positive_admissions,
                    n_all_mhh1_admissions,
                ),
                "n_filter_admission_rows": len(filter_admissions),
                "n_json_recovered_sections": n_recovered_json_sections,
                "pct_json_recovered_sections": safe_pct(
                    n_recovered_json_sections,
                    n_classified_sections,
                ),
            }
        ]
    )


def build_section_summary(
    section_results: pd.DataFrame,
    filter_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize classifier labels by parsed discharge-note section."""
    classifier_summary = (
        section_results.assign(
            is_positive=section_results["psychiatric_context_label"].eq("positive"),
            is_json_recovered=section_results[
                "json_recovered_from_partial_response"
            ].fillna(False),
        )
        .groupby(["cohort", "section_name"], as_index=False)
        .agg(
            n_classified_sections=("classifier_row_id", "size"),
            n_positive_sections=("is_positive", "sum"),
            n_admissions_classified=("hadm_id", "nunique"),
            total_psych_keyword_hits=("n_psych_keyword_hits", "sum"),
            median_section_word_count=("section_word_count", "median"),
            mean_section_word_count=("section_word_count", "mean"),
            total_chunks=("n_chunks", "sum"),
            n_json_recovered_sections=("is_json_recovered", "sum"),
        )
    )
    classifier_summary["pct_positive_sections"] = classifier_summary.apply(
        lambda row: safe_pct(row["n_positive_sections"], row["n_classified_sections"]),
        axis=1,
    )
    classifier_summary["pct_json_recovered_sections"] = classifier_summary.apply(
        lambda row: safe_pct(
            row["n_json_recovered_sections"],
            row["n_classified_sections"],
        ),
        axis=1,
    )

    denominator_columns = [
        "cohort",
        "section_name",
        "n_admissions",
        "n_admissions_with_keyword_positive_section",
        "n_keyword_positive_section_rows",
        "pct_admissions_with_keyword_positive_section",
    ]
    denominators = filter_summary.loc[
        ~filter_summary["section_name"].eq("any_selected_section"),
        denominator_columns,
    ]
    summary = classifier_summary.merge(
        denominators,
        on=["cohort", "section_name"],
        how="left",
    )
    summary["pct_positive_sections_of_prefilter_sections"] = summary.apply(
        lambda row: safe_pct(
            row["n_positive_sections"],
            row["n_keyword_positive_section_rows"],
        ),
        axis=1,
    )
    return summary.sort_values(
        ["n_classified_sections", "section_name"],
        ascending=[False, True],
    )


def build_label_summary(section_results: pd.DataFrame) -> pd.DataFrame:
    """Count positive/negative section labels overall and by section."""
    return (
        section_results.groupby(
            ["cohort", "section_name", "psychiatric_context_label"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_sections"})
        .sort_values(["cohort", "section_name", "psychiatric_context_label"])
    )


def build_mention_type_summary(section_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize model-provided mention types without row-level examples."""
    return (
        section_results.groupby(
            [
                "cohort",
                "psychiatric_context_label",
                "psychiatric_mention_type",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_sections"})
        .sort_values(
            ["cohort", "psychiatric_context_label", "n_sections"],
            ascending=[True, True, False],
        )
    )


def build_keyword_group_summary(section_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize section positivity by keyword group membership."""
    rows = []
    for _, row in section_results.iterrows():
        groups = [
            item.strip()
            for item in str(row["psych_keyword_groups"]).split("|")
            if item.strip()
        ]
        for keyword_group in groups:
            rows.append(
                {
                    "cohort": row["cohort"],
                    "classifier_row_id": row["classifier_row_id"],
                    "keyword_group": keyword_group,
                    "is_positive": row["psychiatric_context_label"] == "positive",
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "cohort",
                "keyword_group",
                "n_sections_with_group",
                "n_positive_sections_with_group",
                "pct_positive_sections_with_group",
            ]
        )
    exploded = pd.DataFrame(rows).drop_duplicates(
        subset=["classifier_row_id", "keyword_group"]
    )
    summary = (
        exploded.groupby(["cohort", "keyword_group"], as_index=False)
        .agg(
            n_sections_with_group=("classifier_row_id", "size"),
            n_positive_sections_with_group=("is_positive", "sum"),
        )
        .sort_values("n_sections_with_group", ascending=False)
    )
    summary["pct_positive_sections_with_group"] = summary.apply(
        lambda row: safe_pct(
            row["n_positive_sections_with_group"],
            row["n_sections_with_group"],
        ),
        axis=1,
    )
    return summary


def build_admission_distribution(admission_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize classified sections and positives per admission."""
    numeric_columns = [
        "n_sections_classified",
        "n_positive_sections",
    ]
    return admission_results.loc[:, numeric_columns].describe().reset_index().rename(
        columns={"index": "statistic"}
    )


def main() -> None:
    """Write whole-run aggregate classifier summaries."""
    section_results, admission_results = load_classifier_outputs()
    filter_summary, filter_metadata, filter_admissions = load_filter_metadata()
    chunk_results_available = (
        CLASSIFIER_OUTPUT_DIR / "psych_history_section_chunk_classifier_results.csv"
    ).exists()

    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completeness_summary = build_completeness_summary(
        section_results,
        chunk_results_available,
        filter_metadata,
    )
    overall_summary = build_overall_summary(
        section_results,
        admission_results,
        filter_summary,
        filter_admissions,
    )
    section_summary = build_section_summary(section_results, filter_summary)
    label_summary = build_label_summary(section_results)
    mention_type_summary = build_mention_type_summary(section_results)
    keyword_group_summary = build_keyword_group_summary(section_results)
    admission_distribution = build_admission_distribution(admission_results)

    completeness_summary.to_csv(
        SUMMARY_OUTPUT_DIR / "psych_history_classifier_completeness_summary.csv",
        index=False,
    )
    overall_summary.to_csv(
        SUMMARY_OUTPUT_DIR / "psych_history_classifier_overall_summary.csv",
        index=False,
    )
    section_summary.to_csv(
        SUMMARY_OUTPUT_DIR / "psych_history_classifier_section_summary.csv",
        index=False,
    )
    label_summary.to_csv(
        SUMMARY_OUTPUT_DIR / "psych_history_classifier_section_label_summary.csv",
        index=False,
    )
    mention_type_summary.to_csv(
        SUMMARY_OUTPUT_DIR / "psych_history_classifier_mention_type_summary.csv",
        index=False,
    )
    keyword_group_summary.to_csv(
        SUMMARY_OUTPUT_DIR / "psych_history_classifier_keyword_group_summary.csv",
        index=False,
    )
    admission_distribution.to_csv(
        SUMMARY_OUTPUT_DIR / "psych_history_classifier_admission_distribution.csv",
        index=False,
    )

    print(f"Summarized classifier output from: {CLASSIFIER_OUTPUT_DIR}")
    print(f"Saved aggregate summaries to: {SUMMARY_OUTPUT_DIR}")
    print("\n=== Completeness ===")
    print(completeness_summary.to_string(index=False))
    print("\n=== Overall ===")
    print(overall_summary.to_string(index=False))


if __name__ == "__main__":
    main()
