"""Explore stigmatizing-language keyword hits across selected note sections.

This is an exploratory keyword screen for potentially stigmatizing language
(SL) in selected parsed matched discharge-note sections and writes local CSV
summaries for manual review.

The keyword list follows the provided SL table and additionally includes
`homeless`, as requested. This is a broad lexical screen, not a classifier.
Keyword hits should be interpreted manually in context.

Inputs:
    ../01_discharge_note_preprocessing/01_discharge_note_parsing/full_discharge_note_sections/
        MHH1_psychotic_matched_full_discharge_note_sections.parquet
        MHC0_matched_full_discharge_note_sections.parquet

Outputs:
    analysis_output_SL_keyword_exploration/
        SL_keyword_section_hits.csv
        SL_keyword_hit_snippets_for_review.csv
        SL_keyword_section_summary.csv
        SL_keyword_matches_by_section_summary.csv
        SL_keyword_group_by_section_summary.csv
        SL_keyword_top_terms_by_section.csv
        SL_keyword_top_terms_by_cohort.csv
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import pandas as pd


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PYTHON_DIR = SCRIPT_DIR.parent
PARSER_DIR = REPO_PYTHON_DIR / "01_discharge_note_preprocessing" / "01_discharge_note_parsing"
PARSER_PATH = PARSER_DIR / "02_parse_full_discharge_notes.py"
FULL_NOTE_DIR = PARSER_DIR / "full_discharge_note_sections"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_SL_keyword_exploration"

# Review settings
SNIPPET_WINDOW_CHARS = 140
MAX_SNIPPETS_PER_SECTION = 3
MAX_HIT_ROWS_FOR_REVIEW = 5000
SELECTED_SECTION_NAMES = [
    "problems",
    "review_of_systems",
    "brief_hospital_course",
    "pertinent_results",
    "physical_exam",
    "present_illness",
    "discharge_instructions",
    "discharge_disposition",
    "major_surgical_or_invasive_procedure",
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

# Keyword list from the SL table plus "homeless". Most terms use word-boundary
# matching. Multi-word phrases allow flexible whitespace.
KEYWORD_PATTERNS = {
    "stigmatizing_language": [
        r"\badherence\b",
        r"\bnonadherent\b",
        r"\bcompliance\b",
        r"\bunwilling\b",
        r"\babuse\b",
        r"\bbelligerent\b",
        r"\bdrug\s+seeking\b",
        r"\babuser\b",
        r"\bdifficult\s+patient\b",
        r"\brefused\b",
        r"\brefuses\b",
        r"\bnoncompliance\b",
        r"\bargumentative\b",
        r"\bcheat\b",
        r"\babuses\b",
        r"\bmalingering\b",
        r"\buser\b",
        r"\bsecondary\s+gain\b",
        r"\bin\s+denial\b",
        r"\brefuse\b",
        r"\bcompliant\b",
        r"\bsubstance\s+abuse\b",
        r"\bnonadherence\b",
        r"\bdegenerate\b",
        r"\bdrug\s+problem\b",
        r"\bcombative\b",
        r"\bfake\b",
        r"\bbeen\s+clean\b",
        r"\bnoncompliant\b",
        r"\baddicted\b",
        r"\bnarcotics\b",
        r"\bhabit\b",
        r"\badherent\b",
    ],
    "housing_insecurity": [
        r"\bhomeless\b",
        r"\bhomelessness\b",
    ],
}


def load_parser_module() -> Any:
    """Load canonical section names from the full discharge-note parser."""
    spec = importlib.util.spec_from_file_location("full_note_parser", PARSER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load parser module from {PARSER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_keyword_patterns() -> dict[str, list[re.Pattern[str]]]:
    """Compile keyword regexes as case-insensitive patterns."""
    return {
        group: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
        for group, patterns in KEYWORD_PATTERNS.items()
    }


def selected_section_columns(parser: Any) -> list[str]:
    """Return selected parser sections and fail clearly if a section changed."""
    available_sections = set(parser.CANONICAL_SECTIONS)
    missing = sorted(set(SELECTED_SECTION_NAMES) - available_sections)
    if missing:
        raise ValueError(f"Selected sections are not parsed: {', '.join(missing)}")
    return SELECTED_SECTION_NAMES.copy()


def load_full_note_outputs(section_columns: list[str]) -> pd.DataFrame:
    """Load matched full-note section outputs for both cohorts."""
    frames = []
    columns = ["cohort", "subject_id", "hadm_id", "note_id"] + section_columns

    for file_config in FULL_NOTE_FILES:
        path = file_config["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing parsed full-note output: {path}")
        df = pd.read_parquet(path)
        df["cohort"] = file_config["cohort"]
        frames.append(df[columns].copy())

    return pd.concat(frames, ignore_index=True)


def build_section_presence(df: pd.DataFrame, section_columns: list[str]) -> pd.DataFrame:
    """Count admissions where each selected section is present."""
    rows = []
    for cohort, cohort_df in df.groupby("cohort"):
        for section in section_columns:
            has_section = cohort_df[section].fillna("").astype(str).str.strip().ne("")
            rows.append(
                {
                    "cohort": cohort,
                    "section_name": section,
                    "n_admissions_with_section": int(has_section.sum()),
                }
            )
    return pd.DataFrame(rows)


def normalize_snippet(text: str, start: int, end: int) -> str:
    """Return a compact local-review snippet around a keyword hit."""
    snippet_start = max(0, start - SNIPPET_WINDOW_CHARS)
    snippet_end = min(len(text), end + SNIPPET_WINDOW_CHARS)
    snippet = text[snippet_start:snippet_end]
    return re.sub(r"\s+", " ", snippet).strip()


def find_keyword_hits(
    text: str,
    compiled_patterns: dict[str, list[re.Pattern[str]]],
) -> tuple[list[dict[str, str | int]], set[str], set[str]]:
    """Find keyword hits and return hit rows plus matched groups/terms."""
    hits = []
    matched_groups = set()
    matched_terms = set()

    for keyword_group, patterns in compiled_patterns.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                term = re.sub(r"\s+", " ", match.group(0).lower()).strip()
                matched_groups.add(keyword_group)
                matched_terms.add(term)
                hits.append(
                    {
                        "keyword_group": keyword_group,
                        "matched_term": term,
                        "match_start": match.start(),
                        "match_end": match.end(),
                        "snippet": normalize_snippet(text, match.start(), match.end()),
                    }
                )

    return hits, matched_groups, matched_terms


def scan_sections(
    df: pd.DataFrame,
    section_columns: list[str],
    compiled_patterns: dict[str, list[re.Pattern[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan all sections and return section-level and hit-level tables."""
    section_rows = []
    hit_rows = []

    for _, row in df.iterrows():
        metadata = {
            "cohort": row["cohort"],
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "note_id": row["note_id"],
        }
        for section in section_columns:
            text = str(row.get(section, "") or "").strip()
            if not text:
                continue

            hits, matched_groups, matched_terms = find_keyword_hits(text, compiled_patterns)
            if not hits:
                continue

            section_rows.append(
                {
                    **metadata,
                    "section_name": section,
                    "n_keyword_hits": len(hits),
                    "n_keyword_groups": len(matched_groups),
                    "keyword_groups": " | ".join(sorted(matched_groups)),
                    "matched_terms": " | ".join(sorted(matched_terms)),
                    "section_word_count": len(text.split()),
                    "section_char_length": len(text),
                }
            )

            for hit_index, hit in enumerate(hits):
                hit_rows.append(
                    {
                        **metadata,
                        "section_name": section,
                        "hit_index": hit_index,
                        **hit,
                    }
                )

    return pd.DataFrame(section_rows), pd.DataFrame(hit_rows)


def build_section_summary(
    section_hits: pd.DataFrame,
    n_notes_by_cohort: pd.Series,
    section_presence: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize keyword-hit coverage by cohort and section."""
    if section_hits.empty:
        return pd.DataFrame()

    summary = (
        section_hits.groupby(["cohort", "section_name"], as_index=False)
        .agg(
            n_admissions_with_any_keyword=("hadm_id", "nunique"),
            n_section_rows_with_any_keyword=("section_name", "size"),
            total_keyword_hits=("n_keyword_hits", "sum"),
            median_keyword_hits_per_hit_section=("n_keyword_hits", "median"),
            median_section_words=("section_word_count", "median"),
        )
        .sort_values(["cohort", "n_admissions_with_any_keyword"], ascending=[True, False])
    )
    summary = summary.merge(
        section_presence,
        on=["cohort", "section_name"],
        how="left",
        validate="one_to_one",
    )
    summary["n_admissions_in_cohort"] = summary["cohort"].map(n_notes_by_cohort).astype(int)
    summary["pct_section_present_with_keyword"] = (
        100.0
        * summary["n_admissions_with_any_keyword"]
        / summary["n_admissions_with_section"]
    )
    summary["pct_all_cohort_admissions_with_keyword_in_section"] = (
        100.0
        * summary["n_admissions_with_any_keyword"]
        / summary["n_admissions_in_cohort"]
    )
    return summary


def build_keyword_group_summary(
    section_hits: pd.DataFrame,
    hit_rows: pd.DataFrame,
    section_presence: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize keyword-group coverage and exact hits by cohort and section."""
    if section_hits.empty:
        return pd.DataFrame()

    exploded = section_hits.assign(
        keyword_group=section_hits["keyword_groups"].str.split(r"\s+\|\s+")
    ).explode("keyword_group")
    presence_summary = (
        exploded.groupby(["cohort", "section_name", "keyword_group"], as_index=False)
        .agg(
            n_admissions_with_term_group_in_section=("hadm_id", "nunique"),
            n_section_rows=("section_name", "size"),
        )
    )

    if hit_rows.empty:
        presence_summary["total_keyword_group_hits"] = 0
        return presence_summary.sort_values(
            ["cohort", "section_name", "n_admissions_with_term_group_in_section"],
            ascending=[True, True, False],
        )

    hit_summary = (
        hit_rows.groupby(["cohort", "section_name", "keyword_group"], as_index=False)
        .agg(total_keyword_group_hits=("matched_term", "size"))
    )

    return (
        presence_summary.merge(
            hit_summary,
            on=["cohort", "section_name", "keyword_group"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            section_presence,
            on=["cohort", "section_name"],
            how="left",
            validate="many_to_one",
        )
        .fillna({"total_keyword_group_hits": 0})
        .assign(
            total_keyword_group_hits=lambda df: df["total_keyword_group_hits"].astype(int),
            pct_section_present_with_keyword_group=lambda df: (
                100.0
                * df["n_admissions_with_term_group_in_section"]
                / df["n_admissions_with_section"]
            ),
        )
        .sort_values(
            ["cohort", "section_name", "n_admissions_with_term_group_in_section"],
            ascending=[True, True, False],
        )
    )


def build_section_keyword_match_summary(
    section_hits: pd.DataFrame,
    section_presence: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize any selected SL keyword match by cohort and section."""
    if section_hits.empty:
        return pd.DataFrame()

    summary = (
        section_hits.groupby(["cohort", "section_name"], as_index=False)
        .agg(
            n_admissions_with_any_selected_SL_keyword=("hadm_id", "nunique"),
            n_section_rows_with_any_selected_SL_keyword=("section_name", "size"),
            total_selected_SL_keyword_hits=("n_keyword_hits", "sum"),
            median_selected_SL_keyword_hits_per_hit_section=("n_keyword_hits", "median"),
        )
        .merge(
            section_presence,
            on=["cohort", "section_name"],
            how="left",
            validate="one_to_one",
        )
    )
    summary["pct_section_present_with_keyword"] = (
        100.0
        * summary["n_admissions_with_any_selected_SL_keyword"]
        / summary["n_admissions_with_section"]
    )
    return (
        summary
        .sort_values(
            ["cohort", "n_admissions_with_any_selected_SL_keyword"],
            ascending=[True, False],
        )
    )


def build_top_terms(hit_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize most frequent matched terms by cohort, section, and keyword group."""
    if hit_rows.empty:
        return pd.DataFrame()

    return (
        hit_rows.groupby(
            ["cohort", "section_name", "keyword_group", "matched_term"], as_index=False
        )
        .agg(
            n_hits=("matched_term", "size"),
            n_admissions_with_term_in_section=("hadm_id", "nunique"),
        )
        .sort_values(["cohort", "section_name", "n_hits"], ascending=[True, True, False])
    )


def build_top_terms_by_cohort(hit_rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize most frequent matched terms by cohort across selected sections."""
    if hit_rows.empty:
        return pd.DataFrame()

    return (
        hit_rows.groupby(["cohort", "keyword_group", "matched_term"], as_index=False)
        .agg(
            n_hits=("matched_term", "size"),
            n_admissions_with_term=("hadm_id", "nunique"),
            n_sections_with_term=("section_name", "nunique"),
        )
        .sort_values(["cohort", "n_hits"], ascending=[True, False])
    )


def write_outputs(
    section_hits: pd.DataFrame,
    hit_rows: pd.DataFrame,
    section_summary: pd.DataFrame,
    section_keyword_match_summary: pd.DataFrame,
    keyword_group_summary: pd.DataFrame,
    top_terms: pd.DataFrame,
    top_terms_by_cohort: pd.DataFrame,
) -> None:
    """Write local CSV outputs for exploratory review."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    section_hits.to_csv(OUTPUT_DIR / "SL_keyword_section_hits.csv", index=False)
    hit_rows.head(MAX_HIT_ROWS_FOR_REVIEW).to_csv(
        OUTPUT_DIR / "SL_keyword_hit_snippets_for_review.csv", index=False
    )
    section_summary.to_csv(OUTPUT_DIR / "SL_keyword_section_summary.csv", index=False)
    section_keyword_match_summary.to_csv(
        OUTPUT_DIR / "SL_keyword_matches_by_section_summary.csv",
        index=False,
    )
    keyword_group_summary.to_csv(
        OUTPUT_DIR / "SL_keyword_group_by_section_summary.csv", index=False
    )
    top_terms.to_csv(OUTPUT_DIR / "SL_keyword_top_terms_by_section.csv", index=False)
    top_terms_by_cohort.to_csv(
        OUTPUT_DIR / "SL_keyword_top_terms_by_cohort.csv",
        index=False,
    )


def main() -> None:
    """Run SL keyword exploration over selected parsed sections."""
    parser = load_parser_module()
    section_columns = selected_section_columns(parser)
    compiled_patterns = compile_keyword_patterns()

    df = load_full_note_outputs(section_columns)
    n_notes_by_cohort = df.groupby("cohort")["hadm_id"].nunique()
    section_presence = build_section_presence(df, section_columns)
    section_hits, hit_rows = scan_sections(df, section_columns, compiled_patterns)

    section_summary = build_section_summary(section_hits, n_notes_by_cohort, section_presence)
    section_keyword_match_summary = build_section_keyword_match_summary(
        section_hits,
        section_presence,
    )
    keyword_group_summary = build_keyword_group_summary(
        section_hits,
        hit_rows,
        section_presence,
    )
    top_terms = build_top_terms(hit_rows)
    top_terms_by_cohort = build_top_terms_by_cohort(hit_rows)
    write_outputs(
        section_hits,
        hit_rows,
        section_summary,
        section_keyword_match_summary,
        keyword_group_summary,
        top_terms,
        top_terms_by_cohort,
    )

    print(f"Scanned {len(df)} matched discharge notes across both cohorts.")
    print(f"Selected sections: {', '.join(section_columns)}")
    print(f"Found SL keyword hits in {len(section_hits)} note-section rows.")
    print(f"Saved SL keyword exploration outputs to: {OUTPUT_DIR}")
    if not section_summary.empty:
        print(section_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
