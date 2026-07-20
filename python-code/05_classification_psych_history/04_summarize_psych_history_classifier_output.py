"""Summarize aggregate outputs from the psych-history section classifier.

This script reads the section-level and admission-level classifier outputs and
reports aggregate counts only. It does not print or write raw note text,
evidence spans, reasons, patient IDs, or row-level classifier results.

Outputs:
    psych_history_classifier_output/
        psych_history_classifier_overall_summary.csv
        psych_history_classifier_section_summary.csv
        psych_history_classifier_admission_summary_aggregate.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CLASSIFIER_OUTPUT_DIR = SCRIPT_DIR / "psych_history_classifier_output"
FILTER_SUMMARY_PATH = (
    SCRIPT_DIR / "psych_history_llm_input" / "filtered_psych_keyword_filter_summary.csv"
)

SECTION_RESULTS_PATH = CLASSIFIER_OUTPUT_DIR / "psych_history_section_classifier_results.csv"
ADMISSION_RESULTS_PATH = CLASSIFIER_OUTPUT_DIR / "psych_history_admission_summary.csv"


def load_classifier_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load section-level and admission-level classifier outputs."""
    if not SECTION_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing section classifier output: {SECTION_RESULTS_PATH}")
    if not ADMISSION_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing admission classifier output: {ADMISSION_RESULTS_PATH}")

    section_results = pd.read_csv(SECTION_RESULTS_PATH)
    admission_results = pd.read_csv(ADMISSION_RESULTS_PATH)
    return section_results, admission_results


def load_filter_denominators() -> dict[str, int]:
    """Load filtering denominators if available."""
    if not FILTER_SUMMARY_PATH.exists():
        return {}

    filter_summary = pd.read_csv(FILTER_SUMMARY_PATH)
    overall = filter_summary.loc[
        filter_summary["section_name"].eq("any_candidate_section")
    ]
    if overall.empty:
        return {}

    row = overall.iloc[0]
    return {
        "n_all_mhh1_admissions": int(row["n_admissions"]),
        "n_keyword_positive_admissions": int(
            row["n_admissions_with_keyword_positive_section"]
        ),
        "n_keyword_positive_sections": int(row["n_keyword_positive_section_rows"]),
    }


def build_overall_summary(
    section_results: pd.DataFrame,
    admission_results: pd.DataFrame,
    denominators: dict[str, int],
) -> pd.DataFrame:
    """Build one-row aggregate classifier summary."""
    n_sections = len(section_results)
    n_admissions = len(admission_results)
    n_all_admissions = denominators.get("n_all_mhh1_admissions", n_admissions)

    n_any_positive_sections = int(section_results["label"].eq("positive").sum())
    n_psychosis_sections = int(
        section_results["psychosis_related_context_label"].eq("positive").sum()
    )
    n_other_sections = int(
        section_results["other_psychiatric_context_label"].eq("positive").sum()
    )

    n_any_positive_admissions = int(admission_results["any_positive"].sum())
    n_psychosis_admissions = int(admission_results["any_psychosis_positive"].sum())
    n_other_admissions = int(admission_results["any_other_psychiatric_positive"].sum())

    return pd.DataFrame(
        [
            {
                "n_all_mhh1_admissions": n_all_admissions,
                "n_keyword_positive_admissions": denominators.get(
                    "n_keyword_positive_admissions",
                    n_admissions,
                ),
                "n_classified_admissions": n_admissions,
                "n_keyword_positive_sections": denominators.get(
                    "n_keyword_positive_sections",
                    n_sections,
                ),
                "n_classified_sections": n_sections,
                "n_any_positive_sections": n_any_positive_sections,
                "pct_any_positive_sections_of_keyword_sections": safe_pct(
                    n_any_positive_sections,
                    n_sections,
                ),
                "n_psychosis_positive_sections": n_psychosis_sections,
                "pct_psychosis_positive_sections_of_keyword_sections": safe_pct(
                    n_psychosis_sections,
                    n_sections,
                ),
                "n_other_psychiatric_positive_sections": n_other_sections,
                "pct_other_psychiatric_positive_sections_of_keyword_sections": safe_pct(
                    n_other_sections,
                    n_sections,
                ),
                "n_any_positive_admissions": n_any_positive_admissions,
                "pct_any_positive_admissions_of_keyword_admissions": safe_pct(
                    n_any_positive_admissions,
                    n_admissions,
                ),
                "pct_any_positive_admissions_of_all_mhh1": safe_pct(
                    n_any_positive_admissions,
                    n_all_admissions,
                ),
                "n_psychosis_positive_admissions": n_psychosis_admissions,
                "pct_psychosis_positive_admissions_of_keyword_admissions": safe_pct(
                    n_psychosis_admissions,
                    n_admissions,
                ),
                "pct_psychosis_positive_admissions_of_all_mhh1": safe_pct(
                    n_psychosis_admissions,
                    n_all_admissions,
                ),
                "n_other_psychiatric_positive_admissions": n_other_admissions,
                "pct_other_psychiatric_positive_admissions_of_keyword_admissions": safe_pct(
                    n_other_admissions,
                    n_admissions,
                ),
                "pct_other_psychiatric_positive_admissions_of_all_mhh1": safe_pct(
                    n_other_admissions,
                    n_all_admissions,
                ),
            }
        ]
    )


def build_section_summary(section_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize classifier labels by section."""
    rows = []
    for section_name, group in section_results.groupby("section_name"):
        n_sections = len(group)
        n_any_positive = int(group["label"].eq("positive").sum())
        n_psychosis_positive = int(
            group["psychosis_related_context_label"].eq("positive").sum()
        )
        n_other_positive = int(
            group["other_psychiatric_context_label"].eq("positive").sum()
        )
        rows.append(
            {
                "section_name": section_name,
                "n_keyword_positive_sections": n_sections,
                "n_any_positive": n_any_positive,
                "pct_any_positive_of_keyword_sections": safe_pct(
                    n_any_positive,
                    n_sections,
                ),
                "n_psychosis_positive": n_psychosis_positive,
                "pct_psychosis_positive_of_keyword_sections": safe_pct(
                    n_psychosis_positive,
                    n_sections,
                ),
                "n_other_psychiatric_positive": n_other_positive,
                "pct_other_psychiatric_positive_of_keyword_sections": safe_pct(
                    n_other_positive,
                    n_sections,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "n_keyword_positive_sections",
        ascending=False,
    )


def build_admission_aggregate(admission_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize classifier labels at admission level."""
    numeric_columns = [
        "n_sections_classified",
        "n_positive_sections",
        "n_psychosis_positive_sections",
        "n_other_psychiatric_positive_sections",
    ]
    summary = admission_results.loc[:, numeric_columns].describe().reset_index()
    summary = summary.rename(columns={"index": "statistic"})
    return summary


def safe_pct(numerator: int, denominator: int) -> float:
    """Return percentage and avoid division by zero."""
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def main() -> None:
    """Write aggregate classifier summaries."""
    section_results, admission_results = load_classifier_outputs()
    denominators = load_filter_denominators()

    overall_summary = build_overall_summary(
        section_results,
        admission_results,
        denominators,
    )
    section_summary = build_section_summary(section_results)
    admission_aggregate = build_admission_aggregate(admission_results)

    overall_summary.to_csv(
        CLASSIFIER_OUTPUT_DIR / "psych_history_classifier_overall_summary.csv",
        index=False,
    )
    section_summary.to_csv(
        CLASSIFIER_OUTPUT_DIR / "psych_history_classifier_section_summary.csv",
        index=False,
    )
    admission_aggregate.to_csv(
        CLASSIFIER_OUTPUT_DIR / "psych_history_classifier_admission_summary_aggregate.csv",
        index=False,
    )

    print("=== Overall Summary ===")
    print(overall_summary.to_string(index=False))
    print("\n=== Section Summary ===")
    print(section_summary.to_string(index=False))


if __name__ == "__main__":
    main()
