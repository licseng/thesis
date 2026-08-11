"""Summarize psych-keyword coverage in candidate LLM sections.

This script is an aggregate pre-filtering analysis for the WP2 LLM classifier.
It scans only the four sections currently selected for LLM classification and
counts whether each admission/section contains any psychiatry-related keyword
from the broad psych keyword vocabulary.

No raw note text or snippets are written. The admission-level output is an
ID-only filter table that can be used later to avoid sending every admission to
the local LLM.

Outputs:
    analysis_output_psych_keyword_llm_filter_coverage/
        psych_keyword_llm_candidate_section_coverage.csv
        psych_keyword_llm_candidate_admission_coverage.csv
        psych_keyword_llm_candidate_keyword_group_coverage.csv
        psych_keyword_llm_candidate_admission_filter.csv
        psych_keyword_llm_candidate_section_filter.csv
"""

from __future__ import annotations

import importlib.util
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PYTHON_DIR = SCRIPT_DIR.parent
PARSER_DIR = REPO_PYTHON_DIR / "01_discharge_note_preprocessing" / "01_discharge_note_parsing"
FULL_NOTE_DIR = PARSER_DIR / "full_discharge_note_sections"
PSYCH_KEYWORD_SCRIPT = SCRIPT_DIR / "02_explore_psych_keywords_by_section.py"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_psych_keyword_llm_filter_coverage"

LLM_CANDIDATE_SECTION_NAMES = [
    "present_illness",
    "brief_hospital_course",
    "problems",
    "discharge_diagnosis",
]

FULL_NOTE_FILES = [
    {
        "cohort": "MHH1_psychotic",
        "path": FULL_NOTE_DIR / "MHH1_psychotic_matched_full_discharge_note_sections.parquet",
    },
    {
        "cohort": "MHC0",
        "path": FULL_NOTE_DIR / "MHC0_matched_full_discharge_note_sections.parquet",
    },
]

METADATA_COLUMNS = ["cohort", "subject_id", "hadm_id", "note_id"]


def load_module(path: Path, module_name: str) -> Any:
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_keyword_patterns(
    keyword_patterns: dict[str, list[str]],
) -> dict[str, list[re.Pattern[str]]]:
    """Compile keyword regexes as case-insensitive patterns."""
    return {
        group: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
        for group, patterns in keyword_patterns.items()
    }


def load_full_note_outputs(section_columns: list[str]) -> pd.DataFrame:
    """Load matched full-note section outputs for both cohorts."""
    frames = []
    columns = ["subject_id", "hadm_id", "note_id"] + section_columns

    for file_config in FULL_NOTE_FILES:
        path = file_config["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing parsed full-note output: {path}")
        df = pd.read_parquet(path, columns=columns)
        df.insert(0, "cohort", file_config["cohort"])
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def find_keyword_hits(
    text: str,
    compiled_patterns: dict[str, list[re.Pattern[str]]],
) -> tuple[int, set[str], Counter[str]]:
    """Return total hits, matched keyword groups, and per-group hit counts."""
    total_hits = 0
    matched_groups = set()
    group_counts: Counter[str] = Counter()

    for keyword_group, patterns in compiled_patterns.items():
        for pattern in patterns:
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            n_matches = len(matches)
            total_hits += n_matches
            matched_groups.add(keyword_group)
            group_counts[keyword_group] += n_matches

    return total_hits, matched_groups, group_counts


def scan_candidate_sections(
    df: pd.DataFrame,
    compiled_patterns: dict[str, list[re.Pattern[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan the four candidate LLM sections without storing raw text."""
    section_rows = []
    group_counter_by_key: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    group_doc_counter_by_key: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        metadata = {column: row_dict[column] for column in METADATA_COLUMNS}

        for section_name in LLM_CANDIDATE_SECTION_NAMES:
            text = str(row_dict.get(section_name, "") or "").strip()
            has_section = bool(text)
            total_hits = 0
            matched_groups: set[str] = set()
            if has_section:
                total_hits, matched_groups, group_counts = find_keyword_hits(
                    text,
                    compiled_patterns,
                )
                for keyword_group, n_hits in group_counts.items():
                    group_counter_by_key[
                        (metadata["cohort"], section_name, keyword_group)
                    ][metadata["hadm_id"]] += n_hits
                for keyword_group in matched_groups:
                    group_doc_counter_by_key[
                        (metadata["cohort"], section_name, keyword_group)
                    ][metadata["hadm_id"]] += 1

            section_rows.append(
                {
                    **metadata,
                    "section_name": section_name,
                    "has_section": has_section,
                    "has_any_psych_keyword": total_hits > 0,
                    "n_psych_keyword_hits": total_hits,
                    "n_psych_keyword_groups": len(matched_groups),
                    "psych_keyword_groups": " | ".join(sorted(matched_groups)),
                }
            )

    group_rows = []
    for (cohort, section_name, keyword_group), admission_hit_counts in group_counter_by_key.items():
        group_rows.append(
            {
                "cohort": cohort,
                "section_name": section_name,
                "keyword_group": keyword_group,
                "n_admissions_with_keyword_group": len(
                    group_doc_counter_by_key[(cohort, section_name, keyword_group)]
                ),
                "total_keyword_group_hits": int(sum(admission_hit_counts.values())),
            }
        )

    return pd.DataFrame(section_rows), pd.DataFrame(group_rows)


def build_section_coverage(section_filter: pd.DataFrame) -> pd.DataFrame:
    """Summarize keyword coverage by cohort and candidate section."""
    summary = (
        section_filter.groupby(["cohort", "section_name"], as_index=False)
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_admissions_with_section=("has_section", "sum"),
            n_admissions_with_any_psych_keyword=("has_any_psych_keyword", "sum"),
            total_psych_keyword_hits=("n_psych_keyword_hits", "sum"),
            median_psych_keyword_hits_per_admission=("n_psych_keyword_hits", "median"),
            max_psych_keyword_hits_in_one_section=("n_psych_keyword_hits", "max"),
        )
    )
    summary["pct_all_admissions_with_section"] = (
        100.0 * summary["n_admissions_with_section"] / summary["n_admissions"]
    )
    summary["pct_all_admissions_with_any_psych_keyword"] = (
        100.0 * summary["n_admissions_with_any_psych_keyword"] / summary["n_admissions"]
    )
    summary["pct_present_sections_with_any_psych_keyword"] = (
        100.0
        * summary["n_admissions_with_any_psych_keyword"]
        / summary["n_admissions_with_section"].replace(0, pd.NA)
    ).fillna(0.0)
    return summary.sort_values(
        ["cohort", "n_admissions_with_any_psych_keyword"],
        ascending=[True, False],
    )


def build_admission_filter(section_filter: pd.DataFrame) -> pd.DataFrame:
    """Collapse section-level filter rows to one ID-only row per admission."""
    section_filter = section_filter.copy()
    section_filter["keyword_hit_section_name"] = section_filter["section_name"].where(
        section_filter["has_any_psych_keyword"],
        "",
    )
    return (
        section_filter.groupby(METADATA_COLUMNS, as_index=False)
        .agg(
            n_candidate_sections_present=("has_section", "sum"),
            n_candidate_sections_with_any_psych_keyword=("has_any_psych_keyword", "sum"),
            total_candidate_section_psych_keyword_hits=("n_psych_keyword_hits", "sum"),
            candidate_sections_with_any_psych_keyword=(
                "keyword_hit_section_name",
                join_nonempty_unique_values,
            ),
        )
        .assign(
            has_any_candidate_section_psych_keyword=lambda df: (
                df["n_candidate_sections_with_any_psych_keyword"] > 0
            )
        )
    )


def join_nonempty_unique_values(values: pd.Series) -> str:
    """Join non-empty unique string values."""
    return " | ".join(sorted({str(value) for value in values if str(value)}))


def build_admission_coverage(admission_filter: pd.DataFrame) -> pd.DataFrame:
    """Summarize admission-level coverage by cohort."""
    summary = (
        admission_filter.groupby("cohort", as_index=False)
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_admissions_with_any_candidate_section_psych_keyword=(
                "has_any_candidate_section_psych_keyword",
                "sum",
            ),
            mean_candidate_sections_with_any_psych_keyword=(
                "n_candidate_sections_with_any_psych_keyword",
                "mean",
            ),
            median_candidate_sections_with_any_psych_keyword=(
                "n_candidate_sections_with_any_psych_keyword",
                "median",
            ),
            total_candidate_section_psych_keyword_hits=(
                "total_candidate_section_psych_keyword_hits",
                "sum",
            ),
        )
    )
    summary["pct_admissions_with_any_candidate_section_psych_keyword"] = (
        100.0
        * summary["n_admissions_with_any_candidate_section_psych_keyword"]
        / summary["n_admissions"]
    )
    return summary


def add_group_percentages(
    keyword_group_coverage: pd.DataFrame,
    section_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Add section denominator percentages to keyword-group coverage."""
    if keyword_group_coverage.empty:
        return keyword_group_coverage
    denominators = section_coverage.loc[
        :,
        ["cohort", "section_name", "n_admissions", "n_admissions_with_section"],
    ]
    output = keyword_group_coverage.merge(
        denominators,
        on=["cohort", "section_name"],
        how="left",
        validate="many_to_one",
    )
    output["pct_all_admissions_with_keyword_group"] = (
        100.0 * output["n_admissions_with_keyword_group"] / output["n_admissions"]
    )
    output["pct_present_sections_with_keyword_group"] = (
        100.0
        * output["n_admissions_with_keyword_group"]
        / output["n_admissions_with_section"].replace(0, pd.NA)
    ).fillna(0.0)
    return output.sort_values(
        ["cohort", "section_name", "n_admissions_with_keyword_group"],
        ascending=[True, True, False],
    )


def write_outputs(
    section_coverage: pd.DataFrame,
    admission_coverage: pd.DataFrame,
    keyword_group_coverage: pd.DataFrame,
    admission_filter: pd.DataFrame,
    section_filter: pd.DataFrame,
) -> None:
    """Write aggregate coverage and ID-only filter tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    section_coverage.to_csv(
        OUTPUT_DIR / "psych_keyword_llm_candidate_section_coverage.csv",
        index=False,
    )
    admission_coverage.to_csv(
        OUTPUT_DIR / "psych_keyword_llm_candidate_admission_coverage.csv",
        index=False,
    )
    keyword_group_coverage.to_csv(
        OUTPUT_DIR / "psych_keyword_llm_candidate_keyword_group_coverage.csv",
        index=False,
    )
    admission_filter.to_csv(
        OUTPUT_DIR / "psych_keyword_llm_candidate_admission_filter.csv",
        index=False,
    )
    section_filter.to_csv(
        OUTPUT_DIR / "psych_keyword_llm_candidate_section_filter.csv",
        index=False,
    )


def main() -> None:
    """Run aggregate keyword-coverage analysis for candidate LLM sections."""
    psych_keywords = load_module(PSYCH_KEYWORD_SCRIPT, "psych_keyword_script")
    compiled_patterns = compile_keyword_patterns(psych_keywords.KEYWORD_PATTERNS)

    df = load_full_note_outputs(LLM_CANDIDATE_SECTION_NAMES)
    section_filter, keyword_group_coverage = scan_candidate_sections(df, compiled_patterns)
    section_coverage = build_section_coverage(section_filter)
    admission_filter = build_admission_filter(section_filter)
    admission_coverage = build_admission_coverage(admission_filter)
    keyword_group_coverage = add_group_percentages(keyword_group_coverage, section_coverage)

    write_outputs(
        section_coverage,
        admission_coverage,
        keyword_group_coverage,
        admission_filter,
        section_filter,
    )

    print(f"Scanned {len(admission_filter)} admissions.")
    print(f"Candidate sections: {', '.join(LLM_CANDIDATE_SECTION_NAMES)}")
    print(f"Saved aggregate/filter outputs to: {OUTPUT_DIR}")
    print("\n=== Admission Coverage ===")
    print(admission_coverage.to_string(index=False))
    print("\n=== Section Coverage ===")
    print(section_coverage.to_string(index=False))


if __name__ == "__main__":
    main()
