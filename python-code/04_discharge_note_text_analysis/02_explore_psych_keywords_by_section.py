"""Explore psychiatry-related keyword hits across selected discharge-note sections.

This is a pre-LLM exploration step for WP2. It scans every parsed section in the
matched discharge-note outputs and reports where psychiatry-related terms occur.
The goal is to understand which sections contain candidate psychiatric-history
language before deciding what should be sent to a local LLM classifier.

The main outputs are intentionally restricted to clinically relevant sections
and psych keyword groups selected for review. This keeps the CSVs smaller and
closer to the WP2 psychiatric-history documentation question.

Outputs are written locally to:
    analysis_output_psych_keyword_exploration/
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PYTHON_DIR = SCRIPT_DIR.parent
PARSER_DIR = REPO_PYTHON_DIR / "01_discharge_note_preprocessing" / "01_discharge_note_parsing"
PARSER_PATH = PARSER_DIR / "02_parse_full_discharge_notes.py"
FULL_NOTE_DIR = PARSER_DIR / "full_discharge_note_sections"
OUTPUT_DIR = (
    SCRIPT_DIR
    / "analysis_output_psych_keyword_exploration"
)

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
SELECTED_KEYWORD_GROUPS = [
    "psychosis_schizophrenia",
    "bipolar_mania",
    "psychiatry_general",
    "other_psych_conditions",
    "substance_use",
    "cognitive_behavioral",
]
NON_COGNITIVE_CHIEF_COMPLAINT_REVIEW_COLUMNS = [
    "cohort",
    "subject_id",
    "hadm_id",
    "note_id",
    "keyword_groups",
    "matched_terms",
    "chief_complaint",
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

# Broad exploratory vocabulary. This is deliberately wider than the final
# psychotic-history definition, because the purpose here is section discovery.
KEYWORD_PATTERNS = {
    "psychosis_schizophrenia": [
        r"\bschizophren\w*\b",
        r"\bschizoaffective\b",
        r"\bpsychosis\b",
        r"\bpsychotic\b",
        r"\bparanoid schizophrenia\b",
    ],
    "bipolar_mania": [
        r"\bbipolar\b",
        r"\bmania\b",
        r"\bmanic\b",
    ],
    "psychiatry_general": [
        r"\bpsychiatr\w*\b",
        r"\bmental health\b",
        r"\bbehavioral health\b",
        r"\bpsych history\b",
        r"\bpsych hx\b",
        r"\bpsychiatric history\b",
    ],
    "common_psych_meds": [
        r"\bhaloperidol\b",
        r"\bhaldol\b",
        r"\brisperidone\b",
        r"\brisperdal\b",
        r"\bolanzapine\b",
        r"\bzyprexa\b",
        r"\bquetiapine\b",
        r"\bseroquel\b",
        r"\bclozapine\b",
        r"\bclozaril\b",
        r"\baripiprazole\b",
        r"\babilify\b",
        r"\bziprasidone\b",
        r"\bgeodon\b",
        r"\bpaliperidone\b",
        r"\binvega\b",
        r"\bfluphenazine\b",
        r"\bperphenazine\b",
        r"\bchlorpromazine\b",
        r"\bthorazine\b",
        r"\blurasidone\b",
        r"\blatuda\b",
    ],
    "other_psych_conditions": [
        r"\bdepression\b",
        r"\bdepressive\b",
        r"\banxiety\b",
        r"\bptsd\b",
        r"\bpersonality disorder\b",
        r"\bsuicid\w*\b",
        r"\bself harm\b",
    ],
    "substance_use": [
        r"\balcohol use\b",
        r"\bsubstance use\b",
        r"\bsubstance abuse\b",
        r"\bcocaine\b",
        r"\bopioid\b",
        r"\bheroin\b",
        r"\bmarijuana\b",
        r"\bcannabis\b",
        r"\betoh\b",
    ],
    "cognitive_behavioral": [
        r"\bdelirium\b",
        r"\bdementia\b",
        r"\baltered mental status\b",
        r"\bagitation\b",
        r"\bconfusion\b",
        r"\bconfused\b",
        r"\bsomatization\b",
        r"\bsomatizing\b",
        r"\bsomatize\w*\b",
        r"\bsomatic symptom\w*\b",
        r"\bsomatic complaint\w*\b",
        r"\bsomatoform\b",
        r"\bpsychosomatic\b",
        r"\bconversion disorder\b",
        r"\bfunctional neurologic\w*\b",
        r"\bfunctional neurological disorder\b",
        r"\bmedically unexplained symptom\w*\b",
        r"\bhypochondriasis\b",
        r"\billness anxiety\b",
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


def compile_keyword_patterns(
    selected_groups: list[str] | None = None,
) -> dict[str, list[re.Pattern[str]]]:
    """Compile keyword regexes as case-insensitive patterns."""
    keyword_patterns = KEYWORD_PATTERNS
    if selected_groups is not None:
        missing = sorted(set(selected_groups) - set(KEYWORD_PATTERNS))
        if missing:
            raise ValueError(f"Unknown selected keyword groups: {', '.join(missing)}")
        keyword_patterns = {
            group: KEYWORD_PATTERNS[group]
            for group in selected_groups
        }

    return {
        group: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
        for group, patterns in keyword_patterns.items()
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
                term = match.group(0).lower()
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
    section_presence: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize broad keyword-group coverage by cohort and section."""
    if section_hits.empty:
        return pd.DataFrame()

    exploded = section_hits.assign(
        keyword_group=section_hits["keyword_groups"].str.split(r"\s+\|\s+")
    ).explode("keyword_group")
    summary = (
        exploded.groupby(["cohort", "section_name", "keyword_group"], as_index=False)
        .agg(
            n_admissions_with_term_group_in_section=("hadm_id", "nunique"),
            n_section_rows=("section_name", "size"),
        )
        .merge(
            section_presence,
            on=["cohort", "section_name"],
            how="left",
            validate="many_to_one",
        )
    )
    summary["pct_section_present_with_keyword_group"] = (
        100.0
        * summary["n_admissions_with_term_group_in_section"]
        / summary["n_admissions_with_section"]
    )
    return summary.sort_values(
        ["cohort", "section_name", "n_admissions_with_term_group_in_section"],
        ascending=[True, True, False],
    )


def build_section_keyword_match_summary(
    section_hits: pd.DataFrame,
    section_presence: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize any selected psych keyword match by cohort and section."""
    if section_hits.empty:
        return pd.DataFrame()

    summary = (
        section_hits.groupby(["cohort", "section_name"], as_index=False)
        .agg(
            n_admissions_with_any_selected_psych_keyword=("hadm_id", "nunique"),
            n_section_rows_with_any_selected_psych_keyword=("section_name", "size"),
            total_selected_psych_keyword_hits=("n_keyword_hits", "sum"),
            median_selected_psych_keyword_hits_per_hit_section=("n_keyword_hits", "median"),
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
        * summary["n_admissions_with_any_selected_psych_keyword"]
        / summary["n_admissions_with_section"]
    )
    return (
        summary
        .sort_values(
            ["cohort", "n_admissions_with_any_selected_psych_keyword"],
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


def build_non_cognitive_chief_complaint_review(
    section_hits: pd.DataFrame,
    full_notes: pd.DataFrame,
) -> pd.DataFrame:
    """Return chief-complaint keyword hits excluding cognitive/behavioral terms."""
    if section_hits.empty:
        return pd.DataFrame(columns=NON_COGNITIVE_CHIEF_COMPLAINT_REVIEW_COLUMNS)

    chief_hits = section_hits.loc[section_hits["section_name"] == "chief_complaint"].copy()
    if chief_hits.empty:
        return pd.DataFrame(columns=NON_COGNITIVE_CHIEF_COMPLAINT_REVIEW_COLUMNS)

    non_cognitive_mask = ~chief_hits["keyword_groups"].str.split(r"\s+\|\s+").map(
        lambda groups: set(groups).issubset({"cognitive_behavioral"})
    )
    chief_hits = chief_hits.loc[non_cognitive_mask].copy()
    if chief_hits.empty:
        return pd.DataFrame(columns=NON_COGNITIVE_CHIEF_COMPLAINT_REVIEW_COLUMNS)

    chief_text = full_notes.loc[
        :, ["cohort", "subject_id", "hadm_id", "note_id", "chief_complaint"]
    ].copy()
    review = chief_hits.merge(
        chief_text,
        on=["cohort", "subject_id", "hadm_id", "note_id"],
        how="left",
        validate="one_to_one",
    )
    return review.loc[:, NON_COGNITIVE_CHIEF_COMPLAINT_REVIEW_COLUMNS].sort_values(
        ["cohort", "subject_id", "hadm_id"]
    )


def write_outputs(
    section_hits: pd.DataFrame,
    hit_rows: pd.DataFrame,
    section_summary: pd.DataFrame,
    section_keyword_match_summary: pd.DataFrame,
    keyword_group_summary: pd.DataFrame,
    top_terms: pd.DataFrame,
    non_cognitive_chief_complaint_review: pd.DataFrame,
) -> None:
    """Write local CSV outputs for exploratory review."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    section_hits.to_csv(OUTPUT_DIR / "psych_keyword_section_hits.csv", index=False)
    hit_rows.head(MAX_HIT_ROWS_FOR_REVIEW).to_csv(
        OUTPUT_DIR / "psych_keyword_hit_snippets_for_review.csv", index=False
    )
    section_summary.to_csv(OUTPUT_DIR / "psych_keyword_section_summary.csv", index=False)
    section_keyword_match_summary.to_csv(
        OUTPUT_DIR / "psych_keyword_matches_by_section_summary.csv",
        index=False,
    )
    keyword_group_summary.to_csv(
        OUTPUT_DIR / "psych_keyword_group_by_section_summary.csv", index=False
    )
    top_terms.to_csv(OUTPUT_DIR / "psych_keyword_top_terms_by_section.csv", index=False)
    non_cognitive_chief_complaint_review.to_csv(
        OUTPUT_DIR / "non_cognitive_chief_complaint_keyword_hits_review.csv",
        index=False,
    )


def main() -> None:
    """Run psychiatry keyword exploration over selected parsed sections."""
    parser = load_parser_module()
    section_columns = selected_section_columns(parser)
    compiled_patterns = compile_keyword_patterns(SELECTED_KEYWORD_GROUPS)

    df = load_full_note_outputs(section_columns)
    n_notes_by_cohort = df.groupby("cohort")["hadm_id"].nunique()
    section_presence = build_section_presence(df, section_columns)
    section_hits, hit_rows = scan_sections(df, section_columns, compiled_patterns)

    section_summary = build_section_summary(section_hits, n_notes_by_cohort, section_presence)
    section_keyword_match_summary = build_section_keyword_match_summary(
        section_hits,
        section_presence,
    )
    keyword_group_summary = build_keyword_group_summary(section_hits, section_presence)
    top_terms = build_top_terms(hit_rows)
    non_cognitive_chief_complaint_review = build_non_cognitive_chief_complaint_review(
        section_hits,
        df,
    )
    write_outputs(
        section_hits,
        hit_rows,
        section_summary,
        section_keyword_match_summary,
        keyword_group_summary,
        top_terms,
        non_cognitive_chief_complaint_review,
    )

    print(f"Scanned {len(df)} matched discharge notes across both cohorts.")
    print(f"Selected sections: {', '.join(section_columns)}")
    print(f"Selected keyword groups: {', '.join(SELECTED_KEYWORD_GROUPS)}")
    print(f"Found keyword hits in {len(section_hits)} note-section rows.")
    print(f"Saved keyword exploration outputs to: {OUTPUT_DIR}")
    if not section_summary.empty:
        print(section_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
