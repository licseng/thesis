"""Create keyword-filtered section inputs for the local LLM classifier.

This script prepares the LLM input subset for WP2. It keeps only section rows
with at least one psychiatry-related keyword hit. The section scope is
configurable with PSYCH_HISTORY_SECTION_SCOPE:
    - candidate4: the original four candidate sections
    - candidate6: the original four plus discharge instructions/pertinent results
    - all_keyword_sections: all selected psych-keyword exploration sections
    - all_parsed_sections: all parser sections except chief complaint
    - current_context_sections: all parser sections except the excluded
      history/social/family/chief-complaint/unsectioned/medical-history/
      discharge-medication sections

The keyword vocabulary is imported from
`04_discharge_note_text_analysis/02_explore_psych_keywords_by_section.py`, so it
uses all current psych-related keyword groups, including psychiatric
medications.

The parquet output contains section text for local LLM use. The CSV outputs are
aggregate or ID/metadata-only and do not include raw section text.

Outputs:
    psych_history_llm_input/
        filtered_psych_keyword_section_input.parquet
        filtered_psych_keyword_section_input_metadata.csv
        filtered_psych_keyword_admission_summary.csv
        filtered_psych_keyword_filter_summary.csv
"""

from __future__ import annotations

import importlib.util
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PYTHON_DIR = SCRIPT_DIR.parent
PARSER_DIR = REPO_PYTHON_DIR / "01_discharge_note_preprocessing" / "01_discharge_note_parsing"
FULL_NOTE_DIR = PARSER_DIR / "full_discharge_note_sections"
PSYCH_KEYWORD_SCRIPT = (
    REPO_PYTHON_DIR
    / "04_discharge_note_text_analysis"
    / "02_explore_psych_keywords_by_section.py"
)
OUTPUT_DIR = SCRIPT_DIR / "psych_history_llm_input"

SECTION_SCOPE = os.environ.get("PSYCH_HISTORY_SECTION_SCOPE", "candidate4").strip().lower()

CANDIDATE4_SECTION_NAMES = [
    "present_illness",
    "brief_hospital_course",
    "problems",
    "discharge_diagnosis",
]
CANDIDATE6_SECTION_NAMES = CANDIDATE4_SECTION_NAMES + [
    "discharge_instructions",
    "pertinent_results",
]
EXCLUDED_BACKGROUND_SECTION_NAMES = {
    "past_psychiatric_history",
    "social_history",
    "family_history",
    "chief_complaint",
    "unsectioned_text",
    "medical_history",
    "discharge_medications",
}

SECTION_SCOPE_OPTIONS = {
    "candidate4": CANDIDATE4_SECTION_NAMES,
    "candidate6": CANDIDATE6_SECTION_NAMES,
}

FULL_NOTE_FILES = [
    {
        "cohort": "MHH1_psychotic",
        "path": FULL_NOTE_DIR / "MHH1_psychotic_matched_full_discharge_note_sections.parquet",
    },
]

METADATA_COLUMNS = ["cohort", "subject_id", "hadm_id", "note_id", "charttime"]


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


def selected_section_names(psych_keywords: Any) -> list[str]:
    """Return section names for the configured prefiltering scope."""
    if SECTION_SCOPE == "all_keyword_sections":
        return list(psych_keywords.SELECTED_SECTION_NAMES)
    if SECTION_SCOPE == "all_parsed_sections":
        parser = load_module(psych_keywords.PARSER_PATH, "full_note_parser")
        return [
            section
            for section in parser.CANONICAL_SECTIONS
            if section != "chief_complaint"
        ]
    if SECTION_SCOPE == "current_context_sections":
        parser = load_module(psych_keywords.PARSER_PATH, "full_note_parser")
        return [
            section
            for section in parser.CANONICAL_SECTIONS
            if section not in EXCLUDED_BACKGROUND_SECTION_NAMES
        ]
    if SECTION_SCOPE not in SECTION_SCOPE_OPTIONS:
        raise ValueError(
            "PSYCH_HISTORY_SECTION_SCOPE must be one of: "
            f"{', '.join(sorted([*SECTION_SCOPE_OPTIONS, 'all_keyword_sections', 'all_parsed_sections', 'current_context_sections']))}"
        )
    return SECTION_SCOPE_OPTIONS[SECTION_SCOPE].copy()


def load_full_note_outputs(section_columns: list[str]) -> pd.DataFrame:
    """Load parsed full-note section outputs for configured cohorts."""
    frames = []
    columns = ["subject_id", "hadm_id", "note_id", "charttime"] + section_columns

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
) -> tuple[int, set[str], set[str], Counter[str]]:
    """Find keyword hits without storing text snippets."""
    total_hits = 0
    matched_groups = set()
    matched_terms = set()
    group_counts: Counter[str] = Counter()

    for keyword_group, patterns in compiled_patterns.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                term = re.sub(r"\s+", " ", match.group(0).lower()).strip()
                total_hits += 1
                matched_groups.add(keyword_group)
                matched_terms.add(term)
                group_counts[keyword_group] += 1

    return total_hits, matched_groups, matched_terms, group_counts


def build_filtered_section_input(
    df: pd.DataFrame,
    compiled_patterns: dict[str, list[re.Pattern[str]]],
    section_names: list[str],
) -> pd.DataFrame:
    """Return one row per keyword-positive selected section."""
    rows = []

    for source_row in df.itertuples(index=False):
        row_dict = source_row._asdict()
        metadata = {column: row_dict[column] for column in METADATA_COLUMNS}

        for section_name in section_names:
            section_text = str(row_dict.get(section_name, "") or "").strip()
            if not section_text:
                continue

            total_hits, matched_groups, matched_terms, group_counts = find_keyword_hits(
                section_text,
                compiled_patterns,
            )
            if total_hits == 0:
                continue

            rows.append(
                {
                    **metadata,
                    "section_name": section_name,
                    "section_word_count": len(section_text.split()),
                    "section_char_length": len(section_text),
                    "n_psych_keyword_hits": total_hits,
                    "n_psych_keyword_groups": len(matched_groups),
                    "psych_keyword_groups": " | ".join(sorted(matched_groups)),
                    "matched_terms": " | ".join(sorted(matched_terms)),
                    "psych_keyword_group_hit_counts": " | ".join(
                        f"{group}:{count}" for group, count in sorted(group_counts.items())
                    ),
                    "section_text": section_text,
                }
            )

    filtered = pd.DataFrame(rows)
    filtered.insert(0, "classifier_row_id", range(len(filtered)))
    return filtered


def build_admission_summary(filtered_sections: pd.DataFrame) -> pd.DataFrame:
    """Build one row per filtered admission without raw section text."""
    return (
        filtered_sections.groupby(["cohort", "subject_id", "hadm_id", "note_id"], as_index=False)
        .agg(
            n_filtered_sections=("section_name", "size"),
            filtered_sections=("section_name", join_unique_values),
            total_psych_keyword_hits=("n_psych_keyword_hits", "sum"),
            psych_keyword_groups=("psych_keyword_groups", join_pipe_values),
            matched_terms=("matched_terms", join_pipe_values),
        )
        .sort_values(["cohort", "subject_id", "hadm_id"])
    )


def build_filter_summary(
    filtered_sections: pd.DataFrame,
    source_notes: pd.DataFrame,
    section_names: list[str],
) -> pd.DataFrame:
    """Summarize filtered section coverage by cohort and section."""
    rows = []
    for cohort, cohort_notes in source_notes.groupby("cohort"):
        cohort_sections = filtered_sections.loc[filtered_sections["cohort"].eq(cohort)]
        n_admissions = int(cohort_notes["hadm_id"].nunique())
        rows.append(
            {
                "cohort": cohort,
                "section_name": "any_selected_section",
                "n_admissions": n_admissions,
                "n_admissions_with_keyword_positive_section": int(
                    cohort_sections["hadm_id"].nunique()
                ),
                "n_keyword_positive_section_rows": len(cohort_sections),
                "pct_admissions_with_keyword_positive_section": safe_pct(
                    int(cohort_sections["hadm_id"].nunique()),
                    n_admissions,
                ),
            }
        )
        for section_name in section_names:
            section_rows = cohort_sections.loc[cohort_sections["section_name"].eq(section_name)]
            rows.append(
                {
                    "cohort": cohort,
                    "section_name": section_name,
                    "n_admissions": n_admissions,
                    "n_admissions_with_keyword_positive_section": int(
                        section_rows["hadm_id"].nunique()
                    ),
                    "n_keyword_positive_section_rows": len(section_rows),
                    "pct_admissions_with_keyword_positive_section": safe_pct(
                        int(section_rows["hadm_id"].nunique()),
                        n_admissions,
                    ),
                }
            )
    return pd.DataFrame(rows)


def join_unique_values(values: pd.Series) -> str:
    """Join unique non-empty values."""
    return " | ".join(sorted({str(value) for value in values if str(value)}))


def join_pipe_values(values: pd.Series) -> str:
    """Join unique non-empty terms from pipe-delimited strings."""
    terms = set()
    for value in values.dropna().astype(str):
        for term in value.split("|"):
            term = term.strip()
            if term:
                terms.add(term)
    return " | ".join(sorted(terms))


def safe_pct(numerator: int, denominator: int) -> float:
    """Return percentage and avoid division by zero."""
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def write_outputs(
    filtered_sections: pd.DataFrame,
    admission_summary: pd.DataFrame,
    filter_summary: pd.DataFrame,
) -> None:
    """Write filtered LLM input and non-text summaries."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered_sections.to_parquet(
        OUTPUT_DIR / "filtered_psych_keyword_section_input.parquet",
        index=False,
    )

    metadata_columns = [
        column for column in filtered_sections.columns if column != "section_text"
    ]
    filtered_sections.loc[:, metadata_columns].to_csv(
        OUTPUT_DIR / "filtered_psych_keyword_section_input_metadata.csv",
        index=False,
    )
    admission_summary.to_csv(
        OUTPUT_DIR / "filtered_psych_keyword_admission_summary.csv",
        index=False,
    )
    filter_summary.to_csv(
        OUTPUT_DIR / "filtered_psych_keyword_filter_summary.csv",
        index=False,
    )


def main() -> None:
    """Create filtered section input for the local LLM classifier."""
    psych_keywords = load_module(PSYCH_KEYWORD_SCRIPT, "psych_keyword_script")
    compiled_patterns = compile_keyword_patterns(psych_keywords.KEYWORD_PATTERNS)
    section_names = selected_section_names(psych_keywords)

    source_notes = load_full_note_outputs(section_names)
    filtered_sections = build_filtered_section_input(
        source_notes,
        compiled_patterns,
        section_names,
    )
    admission_summary = build_admission_summary(filtered_sections)
    filter_summary = build_filter_summary(filtered_sections, source_notes, section_names)
    write_outputs(filtered_sections, admission_summary, filter_summary)

    print(f"Scanned {source_notes['hadm_id'].nunique()} admissions.")
    print(f"Section scope: {SECTION_SCOPE}")
    print(f"Selected sections: {', '.join(section_names)}")
    print(f"Filtered to {len(filtered_sections)} keyword-positive section rows.")
    print(f"Filtered admissions: {filtered_sections['hadm_id'].nunique()}")
    print(f"Saved LLM input to: {OUTPUT_DIR}")
    print("\n=== Filter Summary ===")
    print(filter_summary.to_string(index=False))


if __name__ == "__main__":
    main()
