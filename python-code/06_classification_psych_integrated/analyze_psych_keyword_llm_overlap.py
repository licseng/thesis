"""Compare LLM psych-history labels with psych keyword section hits.

This script checks whether the local LLM classifier is only reproducing the
same section-level positives already found by the psychiatry keyword screen.

Inputs:
    psych_history_classifier_output/psych_history_section_classifier_results.csv
    ../04_discharge_note_text_analysis/analysis_output_psych_keyword_exploration/
        psych_keyword_section_hits.csv

Outputs:
    psych_keyword_llm_overlap_output/
        psych_keyword_llm_section_overlap.csv
        psych_keyword_llm_section_overlap_summary.csv
        psych_keyword_llm_admission_overlap.csv
        psych_keyword_llm_admission_overlap_summary.csv
        psych_keyword_llm_discordant_sections.csv
        psych_keyword_llm_keyword_hit_only_sections.csv
        psych_keyword_llm_keyword_hit_only_admissions.csv
        psych_keyword_llm_keyword_only_admissions_no_llm_positive.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
LLM_RESULTS_PATH = (
    SCRIPT_DIR
    / "psych_history_classifier_output"
    / "psych_history_section_classifier_results.csv"
)
KEYWORD_SECTION_HITS_PATH = (
    SCRIPT_DIR.parent
    / "04_discharge_note_text_analysis"
    / "analysis_output_psych_keyword_exploration"
    / "psych_keyword_section_hits.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "psych_keyword_llm_overlap_output"

SECTION_KEY_COLUMNS = ["cohort", "subject_id", "hadm_id", "note_id", "section_name"]
ADMISSION_KEY_COLUMNS = ["cohort", "subject_id", "hadm_id", "note_id"]


def load_llm_results() -> pd.DataFrame:
    """Load section-level LLM labels from the classifier output."""
    if not LLM_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Missing LLM classifier output: {LLM_RESULTS_PATH}")

    llm = pd.read_csv(LLM_RESULTS_PATH)
    missing = sorted(set(SECTION_KEY_COLUMNS + ["label"]) - set(llm.columns))
    if missing:
        raise ValueError(f"LLM output is missing required columns: {missing}")

    llm = llm.copy()
    llm["has_llm_positive"] = llm["label"].eq("positive")
    llm["has_llm_nonnegative"] = llm["label"].isin(["positive", "ambiguous"])
    return llm


def load_keyword_hits() -> pd.DataFrame:
    """Load section-level psych keyword hits and collapse to one row per section."""
    if not KEYWORD_SECTION_HITS_PATH.exists():
        raise FileNotFoundError(f"Missing psych keyword section hits: {KEYWORD_SECTION_HITS_PATH}")

    keyword_hits = pd.read_csv(KEYWORD_SECTION_HITS_PATH)
    missing = sorted(
        set(SECTION_KEY_COLUMNS + ["n_keyword_hits", "keyword_groups", "matched_terms"])
        - set(keyword_hits.columns)
    )
    if missing:
        raise ValueError(f"Keyword-hit output is missing required columns: {missing}")

    return (
        keyword_hits.groupby(SECTION_KEY_COLUMNS, as_index=False)
        .agg(
            n_keyword_hits=("n_keyword_hits", "sum"),
            n_keyword_groups=("n_keyword_groups", "max"),
            keyword_groups=("keyword_groups", join_unique_pipe_values),
            matched_terms=("matched_terms", join_unique_pipe_values),
        )
        .assign(has_keyword_hit=True)
    )


def join_unique_pipe_values(values: pd.Series) -> str:
    """Join pipe-delimited strings after removing duplicates."""
    terms = set()
    for value in values.dropna().astype(str):
        for term in value.split("|"):
            term = term.strip()
            if term:
                terms.add(term)
    return " | ".join(sorted(terms))


def categorize_overlap(row: pd.Series) -> str:
    """Return section-level overlap category for LLM positive vs keyword hit."""
    if row["has_llm_positive"] and row["has_keyword_hit"]:
        return "both_llm_positive_and_keyword_hit"
    if row["has_llm_positive"] and not row["has_keyword_hit"]:
        return "llm_positive_only"
    if not row["has_llm_positive"] and row["has_keyword_hit"]:
        return "keyword_hit_only"
    return "neither"


def build_section_overlap(llm: pd.DataFrame, keyword_hits: pd.DataFrame) -> pd.DataFrame:
    """Join LLM rows to keyword-hit rows at admission-note-section level."""
    overlap = llm.merge(
        keyword_hits,
        on=SECTION_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    overlap["has_keyword_hit"] = overlap["has_keyword_hit"].fillna(False).astype(bool)
    overlap["n_keyword_hits"] = overlap["n_keyword_hits"].fillna(0).astype(int)
    overlap["n_keyword_groups"] = overlap["n_keyword_groups"].fillna(0).astype(int)
    overlap["keyword_groups"] = overlap["keyword_groups"].fillna("")
    overlap["matched_terms"] = overlap["matched_terms"].fillna("")
    overlap["overlap_category"] = overlap.apply(categorize_overlap, axis=1)
    return overlap


def summarize_section_overlap(overlap: pd.DataFrame) -> pd.DataFrame:
    """Summarize section-level overlap overall and by section."""
    rows = []
    for section_name, group in [("overall", overlap), *overlap.groupby("section_name")]:
        n_llm_positive = int(group["has_llm_positive"].sum())
        n_keyword_hit = int(group["has_keyword_hit"].sum())
        n_both = int((group["has_llm_positive"] & group["has_keyword_hit"]).sum())
        n_llm_only = int((group["has_llm_positive"] & ~group["has_keyword_hit"]).sum())
        n_keyword_only = int((~group["has_llm_positive"] & group["has_keyword_hit"]).sum())
        rows.append(
            {
                "section_name": section_name,
                "n_classified_sections": len(group),
                "n_llm_positive": n_llm_positive,
                "n_keyword_hit": n_keyword_hit,
                "n_both_llm_positive_and_keyword_hit": n_both,
                "n_llm_positive_only": n_llm_only,
                "n_keyword_hit_only": n_keyword_only,
                "n_neither": int((~group["has_llm_positive"] & ~group["has_keyword_hit"]).sum()),
                "pct_llm_positives_with_keyword_hit": safe_pct(n_both, n_llm_positive),
                "pct_keyword_hits_llm_positive": safe_pct(n_both, n_keyword_hit),
            }
        )
    return pd.DataFrame(rows)


def build_admission_overlap(overlap: pd.DataFrame) -> pd.DataFrame:
    """Collapse section-level overlap to one row per admission/note."""
    return (
        overlap.groupby(ADMISSION_KEY_COLUMNS, as_index=False)
        .agg(
            n_sections_classified=("section_name", "size"),
            n_llm_positive_sections=("has_llm_positive", "sum"),
            n_keyword_hit_sections=("has_keyword_hit", "sum"),
            any_llm_positive=("has_llm_positive", "any"),
            any_keyword_hit=("has_keyword_hit", "any"),
        )
        .assign(
            both_any_positive_and_keyword=lambda df: (
                df["any_llm_positive"] & df["any_keyword_hit"]
            ),
            llm_positive_only=lambda df: df["any_llm_positive"] & ~df["any_keyword_hit"],
            keyword_hit_only=lambda df: ~df["any_llm_positive"] & df["any_keyword_hit"],
            neither=lambda df: ~df["any_llm_positive"] & ~df["any_keyword_hit"],
        )
    )


def summarize_admission_overlap(admission_overlap: pd.DataFrame) -> pd.DataFrame:
    """Summarize admission-level overlap."""
    n_admissions = len(admission_overlap)
    n_llm_positive = int(admission_overlap["any_llm_positive"].sum())
    n_keyword_hit = int(admission_overlap["any_keyword_hit"].sum())
    n_both = int(admission_overlap["both_any_positive_and_keyword"].sum())
    return pd.DataFrame(
        [
            {
                "n_admissions": n_admissions,
                "n_any_llm_positive": n_llm_positive,
                "n_any_keyword_hit": n_keyword_hit,
                "n_both_any_llm_positive_and_keyword_hit": n_both,
                "n_llm_positive_only": int(admission_overlap["llm_positive_only"].sum()),
                "n_keyword_hit_only": int(admission_overlap["keyword_hit_only"].sum()),
                "n_neither": int(admission_overlap["neither"].sum()),
                "pct_llm_positive_admissions_with_keyword_hit": safe_pct(
                    n_both,
                    n_llm_positive,
                ),
                "pct_keyword_hit_admissions_llm_positive": safe_pct(n_both, n_keyword_hit),
            }
        ]
    )


def safe_pct(numerator: int, denominator: int) -> float:
    """Return percentage and avoid division by zero."""
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def write_outputs(
    section_overlap: pd.DataFrame,
    section_summary: pd.DataFrame,
    admission_overlap: pd.DataFrame,
    admission_summary: pd.DataFrame,
) -> None:
    """Write overlap analysis outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    section_overlap.to_csv(OUTPUT_DIR / "psych_keyword_llm_section_overlap.csv", index=False)
    section_summary.to_csv(
        OUTPUT_DIR / "psych_keyword_llm_section_overlap_summary.csv",
        index=False,
    )
    admission_overlap.to_csv(OUTPUT_DIR / "psych_keyword_llm_admission_overlap.csv", index=False)
    admission_summary.to_csv(
        OUTPUT_DIR / "psych_keyword_llm_admission_overlap_summary.csv",
        index=False,
    )

    discordant = section_overlap.loc[
        section_overlap["overlap_category"].isin(["llm_positive_only", "keyword_hit_only"])
    ].copy()
    discordant.to_csv(OUTPUT_DIR / "psych_keyword_llm_discordant_sections.csv", index=False)

    keyword_hit_only_sections = section_overlap.loc[
        section_overlap["overlap_category"].eq("keyword_hit_only")
    ].copy()
    keyword_hit_only_sections.to_csv(
        OUTPUT_DIR / "psych_keyword_llm_keyword_hit_only_sections.csv",
        index=False,
    )

    keyword_hit_only_admissions = (
        keyword_hit_only_sections.groupby(ADMISSION_KEY_COLUMNS, as_index=False)
        .agg(
            n_keyword_hit_only_sections=("section_name", "size"),
            keyword_hit_only_sections=("section_name", join_unique_pipe_values),
            matched_terms=("matched_terms", join_unique_pipe_values),
            keyword_groups=("keyword_groups", join_unique_pipe_values),
        )
        .sort_values(["subject_id", "hadm_id"])
    )
    keyword_hit_only_admissions.to_csv(
        OUTPUT_DIR / "psych_keyword_llm_keyword_hit_only_admissions.csv",
        index=False,
    )

    pure_keyword_only_admissions = admission_overlap.loc[
        admission_overlap["keyword_hit_only"],
        ADMISSION_KEY_COLUMNS,
    ].merge(
        keyword_hit_only_admissions,
        on=ADMISSION_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    pure_keyword_only_admissions.to_csv(
        OUTPUT_DIR / "psych_keyword_llm_keyword_only_admissions_no_llm_positive.csv",
        index=False,
    )


def main() -> None:
    """Run LLM-vs-keyword overlap analysis."""
    llm = load_llm_results()
    keyword_hits = load_keyword_hits()
    section_overlap = build_section_overlap(llm, keyword_hits)
    section_summary = summarize_section_overlap(section_overlap)
    admission_overlap = build_admission_overlap(section_overlap)
    admission_summary = summarize_admission_overlap(admission_overlap)

    write_outputs(section_overlap, section_summary, admission_overlap, admission_summary)

    print(f"Loaded {len(llm)} LLM-classified sections.")
    print(f"Loaded {len(keyword_hits)} keyword-hit sections.")
    print(f"Saved overlap outputs to: {OUTPUT_DIR}")
    print("\n=== Section-Level Overlap ===")
    print(section_summary.to_string(index=False))
    print("\n=== Admission-Level Overlap ===")
    print(admission_summary.to_string(index=False))


if __name__ == "__main__":
    main()
