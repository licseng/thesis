"""Summarize psych and stigmatizing-language keyword hits outside chief complaint.

This script scans the matched full-discharge-note section outputs and counts
keyword hits across the whole parsed note while excluding the `chief_complaint`
section. It reuses the keyword lists from:

    01_explore_psych_keywords_by_section.py
    01_explore_SL_keywords_by_section.py

The output is intentionally aggregate-only. It does not export raw note text or
snippets; use the section-level exploration scripts for local manual review.

Outputs:
    analysis_output_keyword_summary_excluding_chief_complaint/
        keyword_family_note_summary_excluding_chief_complaint.csv
        keyword_family_top_terms_excluding_chief_complaint.csv
        keyword_group_summary_excluding_chief_complaint.csv
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
PARSER_PATH = PARSER_DIR / "02_parse_full_discharge_notes.py"
FULL_NOTE_DIR = PARSER_DIR / "full_discharge_note_sections"
PSYCH_KEYWORD_SCRIPT = SCRIPT_DIR / "02_explore_psych_keywords_by_section.py"
SL_KEYWORD_SCRIPT = SCRIPT_DIR / "02_explore_SL_keywords_by_section.py"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_keyword_summary_excluding_chief_complaint"

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
    columns = ["cohort", "subject_id", "hadm_id", "note_id"] + section_columns

    for file_config in FULL_NOTE_FILES:
        path = file_config["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing parsed full-note output: {path}")
        df = pd.read_parquet(path)
        df["cohort"] = file_config["cohort"]
        frames.append(df[columns].copy())

    return pd.concat(frames, ignore_index=True)


def combined_note_text(row: pd.Series, section_columns: list[str]) -> str:
    """Concatenate all non-chief-complaint parsed sections for one note."""
    parts = []
    for section in section_columns:
        if section == "chief_complaint":
            continue
        text = str(row.get(section, "") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def find_keyword_hits(
    text: str,
    compiled_patterns: dict[str, list[re.Pattern[str]]],
) -> tuple[int, set[str], Counter[str], Counter[str]]:
    """Return hit count, keyword groups, term counts, and group hit counts."""
    total_hits = 0
    matched_groups = set()
    term_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()

    for keyword_group, patterns in compiled_patterns.items():
        for pattern in patterns:
            for match in pattern.finditer(text):
                term = re.sub(r"\s+", " ", match.group(0).lower()).strip()
                total_hits += 1
                matched_groups.add(keyword_group)
                term_counts[term] += 1
                group_counts[keyword_group] += 1

    return total_hits, matched_groups, term_counts, group_counts


def scan_keyword_family(
    df: pd.DataFrame,
    section_columns: list[str],
    keyword_family: str,
    compiled_patterns: dict[str, list[re.Pattern[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scan one keyword family and return note, term, and group summaries."""
    note_rows = []
    term_rows = []
    group_rows = []

    family_term_counts_by_cohort: dict[str, Counter[str]] = defaultdict(Counter)
    family_term_doc_counts_by_cohort: dict[str, Counter[str]] = defaultdict(Counter)
    group_counts_by_cohort: dict[str, Counter[str]] = defaultdict(Counter)
    group_doc_counts_by_cohort: dict[str, Counter[str]] = defaultdict(Counter)

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        cohort = row_dict["cohort"]
        text = combined_note_text(pd.Series(row_dict), section_columns)
        total_hits, matched_groups, term_counts, group_counts = find_keyword_hits(
            text,
            compiled_patterns,
        )

        note_rows.append(
            {
                "keyword_family": keyword_family,
                "cohort": cohort,
                "subject_id": row_dict["subject_id"],
                "hadm_id": row_dict["hadm_id"],
                "note_id": row_dict["note_id"],
                "n_keyword_hits": total_hits,
                "has_keyword_hit": total_hits > 0,
                "n_keyword_groups": len(matched_groups),
                "keyword_groups": " | ".join(sorted(matched_groups)),
            }
        )

        family_term_counts_by_cohort[cohort].update(term_counts)
        family_term_counts_by_cohort["overall"].update(term_counts)
        for term in term_counts:
            family_term_doc_counts_by_cohort[cohort][term] += 1
            family_term_doc_counts_by_cohort["overall"][term] += 1

        group_counts_by_cohort[cohort].update(group_counts)
        group_counts_by_cohort["overall"].update(group_counts)
        for group in matched_groups:
            group_doc_counts_by_cohort[cohort][group] += 1
            group_doc_counts_by_cohort["overall"][group] += 1

    note_df = pd.DataFrame(note_rows)

    for cohort, term_counts in family_term_counts_by_cohort.items():
        n_notes = len(note_df) if cohort == "overall" else int(note_df["cohort"].eq(cohort).sum())
        for term, n_hits in term_counts.most_common():
            n_notes_with_term = family_term_doc_counts_by_cohort[cohort][term]
            term_rows.append(
                {
                    "keyword_family": keyword_family,
                    "cohort": cohort,
                    "matched_term": term,
                    "n_hits": int(n_hits),
                    "n_notes_with_term": int(n_notes_with_term),
                    "pct_notes_with_term": 100.0 * n_notes_with_term / n_notes,
                }
            )

    for cohort, group_counts in group_counts_by_cohort.items():
        n_notes = len(note_df) if cohort == "overall" else int(note_df["cohort"].eq(cohort).sum())
        for keyword_group, n_hits in group_counts.most_common():
            n_notes_with_group = group_doc_counts_by_cohort[cohort][keyword_group]
            group_rows.append(
                {
                    "keyword_family": keyword_family,
                    "cohort": cohort,
                    "keyword_group": keyword_group,
                    "n_hits": int(n_hits),
                    "n_notes_with_group": int(n_notes_with_group),
                    "pct_notes_with_group": 100.0 * n_notes_with_group / n_notes,
                }
            )

    return note_df, pd.DataFrame(term_rows), pd.DataFrame(group_rows)


def build_note_summary(note_hits: pd.DataFrame) -> pd.DataFrame:
    """Summarize keyword-hit coverage by family and cohort."""
    return (
        note_hits.groupby(["keyword_family", "cohort"], as_index=False)
        .agg(
            n_notes=("hadm_id", "nunique"),
            n_notes_with_any_keyword=("has_keyword_hit", "sum"),
            total_keyword_hits=("n_keyword_hits", "sum"),
            mean_keyword_hits_per_note=("n_keyword_hits", "mean"),
            median_keyword_hits_per_note=("n_keyword_hits", "median"),
            max_keyword_hits_in_one_note=("n_keyword_hits", "max"),
        )
        .assign(
            pct_notes_with_any_keyword=lambda df: (
                100.0 * df["n_notes_with_any_keyword"] / df["n_notes"]
            )
        )
        .sort_values(["keyword_family", "cohort"])
    )


def write_outputs(
    note_summary: pd.DataFrame,
    top_terms: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> None:
    """Write aggregate outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    note_summary.to_csv(
        OUTPUT_DIR / "keyword_family_note_summary_excluding_chief_complaint.csv",
        index=False,
    )
    top_terms.to_csv(
        OUTPUT_DIR / "keyword_family_top_terms_excluding_chief_complaint.csv",
        index=False,
    )
    group_summary.to_csv(
        OUTPUT_DIR / "keyword_group_summary_excluding_chief_complaint.csv",
        index=False,
    )


def main() -> None:
    """Run aggregate psych and SL keyword summaries outside chief complaint."""
    parser = load_module(PARSER_PATH, "full_note_parser")
    psych_keywords = load_module(PSYCH_KEYWORD_SCRIPT, "psych_keyword_script")
    sl_keywords = load_module(SL_KEYWORD_SCRIPT, "sl_keyword_script")

    section_columns = list(parser.CANONICAL_SECTIONS) + ["unsectioned_text"]
    df = load_full_note_outputs(section_columns)

    all_note_hits = []
    all_top_terms = []
    all_group_summaries = []
    for keyword_family, keyword_patterns in [
        ("psych", psych_keywords.KEYWORD_PATTERNS),
        ("SL", sl_keywords.KEYWORD_PATTERNS),
    ]:
        note_hits, top_terms, group_summary = scan_keyword_family(
            df,
            section_columns,
            keyword_family,
            compile_keyword_patterns(keyword_patterns),
        )
        all_note_hits.append(note_hits)
        all_top_terms.append(top_terms)
        all_group_summaries.append(group_summary)

    note_hits = pd.concat(all_note_hits, ignore_index=True)
    top_terms = pd.concat(all_top_terms, ignore_index=True)
    group_summary = pd.concat(all_group_summaries, ignore_index=True)
    note_summary = build_note_summary(note_hits)

    write_outputs(note_summary, top_terms, group_summary)

    print(f"Scanned {len(df)} matched discharge notes.")
    print("Excluded section: chief_complaint")
    print(f"Saved aggregate keyword summaries to: {OUTPUT_DIR}")
    print("\n=== Keyword Family Note Summary ===")
    print(note_summary.to_string(index=False))
    print("\n=== Top Overall Terms ===")
    print(
        top_terms.loc[top_terms["cohort"].eq("overall")]
        .groupby("keyword_family", group_keys=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
