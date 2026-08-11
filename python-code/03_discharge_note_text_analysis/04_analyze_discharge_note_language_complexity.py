"""Analyze discharge-note language complexity by matched cohort.

This script computes transparent readability/complexity proxies for matched
MHH1_psychotic and MHC0 full discharge notes. It uses the parser's
`full_note_text` column, which preserves the original note text loaded from the
source table and includes chief complaint. It writes numeric metrics only; no
raw note text is written to output files.

Metrics are descriptive proxies. Clinical notes contain abbreviations, lists,
medication names, lab values, and deidentified placeholders, so school-grade
readability formulas should be interpreted cautiously.

The selected prose-section outputs use flattened parsed section text for:
present_illness, brief_hospital_course, and discharge_instructions. These are
intended for readability-style metrics, not line-format/list-structure metrics.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PYTHON_DIR = SCRIPT_DIR.parent
PARSER_DIR = REPO_PYTHON_DIR / "01_discharge_note_preprocessing" / "01_discharge_note_parsing"
FULL_NOTE_DIR = PARSER_DIR / "full_discharge_note_sections"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_language_complexity"

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

ID_COLUMNS = ["cohort", "subject_id", "hadm_id", "note_id", "charttime"]
SELECTED_PROSE_SECTION_COLUMNS = [
    "present_illness",
    "brief_hospital_course",
    "discharge_instructions",
]
SUMMARY_METRICS = [
    "n_characters",
    "n_whitespace_words",
    "n_tokens",
    "n_alpha_words",
    "n_sentences",
    "n_nonempty_lines",
    "mean_words_per_sentence",
    "mean_words_per_line",
    "pct_short_lines",
    "mean_characters_per_alpha_word",
    "pct_long_alpha_words",
    "pct_complex_alpha_words",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog_index",
    "smog_index",
    "automated_readability_index",
]

WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?")
ALPHA_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def clean_text(value: object) -> str:
    """Return text as a string, with nulls mapped to empty string."""
    if pd.isna(value):
        return ""
    return str(value)


def sentence_units(text: str) -> list[str]:
    """Split clinical text into rough sentence/list units."""
    stripped = text.strip()
    if not stripped:
        return []
    units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n{2,}", stripped)
        if unit.strip()
    ]
    if len(units) <= 1 and "\n" in stripped:
        units = [line.strip() for line in stripped.splitlines() if line.strip()]
    return units or [stripped]


def count_syllables(word: str) -> int:
    """Approximate English syllable count for readability formulas."""
    cleaned = re.sub(r"[^A-Za-z]", "", word).lower()
    if not cleaned:
        return 0
    groups = VOWEL_GROUP_RE.findall(cleaned)
    n_syllables = len(groups)
    if cleaned.endswith("e") and n_syllables > 1 and not cleaned.endswith(("le", "ye")):
        n_syllables -= 1
    return max(1, n_syllables)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return zero for empty denominators."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_complexity_metrics(text: str) -> dict[str, Any]:
    """Compute numeric note/section complexity metrics."""
    text = clean_text(text)
    whitespace_words = text.strip().split()
    tokens = WORD_RE.findall(text)
    alpha_words = ALPHA_WORD_RE.findall(text)
    sentences = sentence_units(text)
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_word_counts = [len(line.split()) for line in nonempty_lines]
    n_whitespace_words = len(whitespace_words)
    n_tokens = len(tokens)
    n_alpha_words = len(alpha_words)
    n_sentences = len(sentences)
    n_nonempty_lines = len(nonempty_lines)
    n_characters = len(text)
    alpha_characters = sum(len(re.sub(r"[^A-Za-z]", "", word)) for word in alpha_words)
    syllables = [count_syllables(word) for word in alpha_words]
    n_syllables = sum(syllables)
    n_long_alpha_words = sum(
        1 for word in alpha_words if len(re.sub(r"[^A-Za-z]", "", word)) >= 7
    )
    n_complex_alpha_words = sum(1 for syllable_count in syllables if syllable_count >= 3)

    words_per_sentence = safe_divide(n_tokens, n_sentences)
    words_per_line = safe_divide(n_whitespace_words, n_nonempty_lines)
    pct_short_lines = 100.0 * safe_divide(
        sum(1 for count in line_word_counts if count <= 5),
        n_nonempty_lines,
    )
    syllables_per_alpha_word = safe_divide(n_syllables, n_alpha_words)
    characters_per_alpha_word = safe_divide(alpha_characters, n_alpha_words)
    pct_long_alpha_words = 100.0 * safe_divide(n_long_alpha_words, n_alpha_words)
    pct_complex_alpha_words = 100.0 * safe_divide(n_complex_alpha_words, n_alpha_words)

    if n_tokens == 0 or n_alpha_words == 0 or n_sentences == 0:
        flesch = 0.0
        flesch_kincaid = 0.0
        gunning_fog = 0.0
        smog = 0.0
        ari = 0.0
    else:
        flesch = 206.835 - (1.015 * words_per_sentence) - (
            84.6 * syllables_per_alpha_word
        )
        flesch_kincaid = (0.39 * words_per_sentence) + (
            11.8 * syllables_per_alpha_word
        ) - 15.59
        gunning_fog = 0.4 * (words_per_sentence + pct_complex_alpha_words)
        smog = 1.043 * ((n_complex_alpha_words * 30.0 / n_sentences) ** 0.5) + 3.1291
        ari = (4.71 * characters_per_alpha_word) + (0.5 * words_per_sentence) - 21.43

    return {
        "n_characters": n_characters,
        "n_whitespace_words": n_whitespace_words,
        "n_tokens": n_tokens,
        "n_alpha_words": n_alpha_words,
        "n_sentences": n_sentences,
        "n_nonempty_lines": n_nonempty_lines,
        "n_syllables": n_syllables,
        "n_long_alpha_words": n_long_alpha_words,
        "n_complex_alpha_words": n_complex_alpha_words,
        "mean_words_per_sentence": words_per_sentence,
        "mean_words_per_line": words_per_line,
        "pct_short_lines": pct_short_lines,
        "mean_characters_per_alpha_word": characters_per_alpha_word,
        "mean_syllables_per_alpha_word": syllables_per_alpha_word,
        "pct_long_alpha_words": pct_long_alpha_words,
        "pct_complex_alpha_words": pct_complex_alpha_words,
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": flesch_kincaid,
        "gunning_fog_index": gunning_fog,
        "smog_index": smog,
        "automated_readability_index": ari,
    }


def load_notes() -> pd.DataFrame:
    """Load parsed matched discharge-note tables."""
    frames = []
    for config in FULL_NOTE_FILES:
        path = config["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing parsed discharge-note sections: {path}")
        columns = [
            column
            for column in [*ID_COLUMNS, "full_note_text", *SELECTED_PROSE_SECTION_COLUMNS]
            if column != "cohort"
        ]
        table = pd.read_parquet(path, columns=columns)
        table.insert(0, "cohort", config["cohort"])
        frames.append(table)
    notes = pd.concat(frames, ignore_index=True)
    notes["charttime"] = pd.to_datetime(notes["charttime"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    return notes


def build_note_level_metrics(notes: pd.DataFrame) -> pd.DataFrame:
    """Compute full-note complexity metrics, one row per admission/note."""
    rows = []
    for _, row in notes.iterrows():
        metrics = compute_complexity_metrics(row["full_note_text"])
        rows.append(
            {
                "cohort": row["cohort"],
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"],
                "note_id": row["note_id"],
                "charttime": row["charttime"],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_prose_section_metrics(notes: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for selected prose-like parsed sections."""
    rows = []
    for _, row in notes.iterrows():
        for section_name in SELECTED_PROSE_SECTION_COLUMNS:
            text = clean_text(row.get(section_name, ""))
            if not text.strip():
                continue
            metrics = compute_complexity_metrics(text)
            rows.append(
                {
                    "cohort": row["cohort"],
                    "subject_id": row["subject_id"],
                    "hadm_id": row["hadm_id"],
                    "note_id": row["note_id"],
                    "charttime": row["charttime"],
                    "section_name": section_name,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize complexity metrics for selected grouping columns."""
    rows = []
    groupby_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_values, group in df.groupby(groupby_key, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_columns, group_values, strict=True))
        row["n_rows"] = len(group)
        row["n_admissions"] = group["hadm_id"].nunique()
        row["n_subjects"] = group["subject_id"].nunique()
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_q1"] = values.quantile(0.25)
            row[f"{metric}_q3"] = values.quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def write_outputs(
    note_metrics: pd.DataFrame,
    note_summary: pd.DataFrame,
    prose_section_metrics: pd.DataFrame,
    prose_section_summary: pd.DataFrame,
) -> None:
    """Write complexity metric outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    note_metrics.to_csv(OUTPUT_DIR / "language_complexity_note_level_metrics.csv", index=False)
    note_summary.to_csv(OUTPUT_DIR / "language_complexity_note_summary.csv", index=False)
    prose_section_metrics.to_csv(
        OUTPUT_DIR / "language_complexity_prose_section_metrics.csv",
        index=False,
    )
    prose_section_summary.to_csv(
        OUTPUT_DIR / "language_complexity_prose_section_summary.csv",
        index=False,
    )


def main() -> None:
    """Run language-complexity analysis for matched discharge notes."""
    notes = load_notes()
    note_metrics = build_note_level_metrics(notes)
    prose_section_metrics = build_prose_section_metrics(notes)
    note_summary = summarize_metrics(note_metrics, ["cohort"])
    prose_section_summary = summarize_metrics(
        prose_section_metrics,
        ["cohort", "section_name"],
    )

    write_outputs(
        note_metrics,
        note_summary,
        prose_section_metrics,
        prose_section_summary,
    )

    print(f"Scanned {len(notes)} matched discharge notes.")
    print(f"Saved language-complexity outputs to: {OUTPUT_DIR}")
    print("\n=== Note-Level Summary ===")
    display_columns = [
        "cohort",
        "n_rows",
        "n_admissions",
        "n_whitespace_words_median",
        "mean_words_per_sentence_median",
        "mean_words_per_line_median",
        "pct_short_lines_median",
        "flesch_kincaid_grade_median",
        "gunning_fog_index_median",
    ]
    print(note_summary.loc[:, display_columns].to_string(index=False))
    print("\n=== Selected Prose-Section Summary ===")
    section_display_columns = [
        "cohort",
        "section_name",
        "n_rows",
        "n_admissions",
        "n_whitespace_words_median",
        "mean_words_per_sentence_median",
        "flesch_kincaid_grade_median",
        "gunning_fog_index_median",
    ]
    print(
        prose_section_summary.loc[:, section_display_columns].to_string(index=False)
    )


if __name__ == "__main__":
    main()
