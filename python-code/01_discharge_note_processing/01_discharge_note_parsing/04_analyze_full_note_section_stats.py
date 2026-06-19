"""Descriptive statistics for parsed matched discharge-note sections.

This script summarizes how often each parsed section is present and how long it
is in each matched cohort. It does not export note text.

Outputs:
    analysis_output_discharge_note_parsing/section_length_summary_by_group.csv
    analysis_output_discharge_note_parsing/section_length_comparison.csv
    analysis_output_discharge_note_parsing/note_level_structure_summary.csv
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PARSER_PATH = SCRIPT_DIR / "02_parse_full_discharge_notes.py"
FULL_NOTE_DIR = SCRIPT_DIR / "full_discharge_note_sections"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_discharge_note_parsing"

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


def load_parser_module() -> Any:
    """Load section names from the full-note parser."""
    spec = importlib.util.spec_from_file_location("full_note_parser", PARSER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load parser module from {PARSER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def word_count(series: pd.Series) -> pd.Series:
    """Count whitespace-delimited words after trimming each section."""
    return series.fillna("").astype(str).str.strip().map(
        lambda value: 0 if not value else len(re.findall(r"\S+", value))
    )


def summarize_numeric(values: pd.Series, prefix: str) -> dict[str, float]:
    """Return robust summary statistics for one numeric series."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_sd": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_median": float(values.median()),
        f"{prefix}_iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        f"{prefix}_p90": float(values.quantile(0.90)),
        f"{prefix}_max": float(values.max()),
    }


def load_full_note_outputs(section_columns: list[str]) -> list[tuple[str, pd.DataFrame]]:
    """Load parsed full-note outputs with only columns needed for analysis."""
    base_columns = ["subject_id", "hadm_id", "n_detected_sections", "full_note_text"]
    loaded = []
    for file_config in FULL_NOTE_FILES:
        path = file_config["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing parsed full-note output: {path}")
        columns = base_columns + section_columns
        loaded.append((file_config["cohort"], pd.read_parquet(path, columns=columns)))
    return loaded


def section_summary_for_group(
    df: pd.DataFrame,
    cohort: str,
    section_columns: list[str],
) -> pd.DataFrame:
    """Summarize section coverage and length for one cohort."""
    rows = []
    n_rows = len(df)

    for section in section_columns:
        text = df[section].fillna("").astype(str).str.strip()
        has_section = text.ne("")
        char_lengths = text.str.len()
        word_lengths = word_count(text)

        row = {
            "cohort": cohort,
            "section": section,
            "n_admissions": n_rows,
            "n_with_section": int(has_section.sum()),
            "pct_with_section": 100.0 * float(has_section.mean()) if n_rows else np.nan,
        }
        row.update(summarize_numeric(char_lengths[has_section], "chars_nonempty"))
        row.update(summarize_numeric(word_lengths[has_section], "words_nonempty"))
        row.update(summarize_numeric(char_lengths, "chars_all_rows"))
        row.update(summarize_numeric(word_lengths, "words_all_rows"))
        rows.append(row)

    return pd.DataFrame(rows)


def note_structure_summary_for_group(df: pd.DataFrame, cohort: str) -> dict[str, float]:
    """Summarize full note length and number of detected hard-coded sections."""
    full_text = df["full_note_text"].fillna("").astype(str)
    full_chars = full_text.str.len()
    full_words = word_count(full_text)
    row = {
        "cohort": cohort,
        "n_admissions": len(df),
    }
    row.update(summarize_numeric(df["n_detected_sections"], "detected_sections"))
    row.update(summarize_numeric(full_chars, "full_note_chars"))
    row.update(summarize_numeric(full_words, "full_note_words"))
    return row


def build_section_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Create a side-by-side MHH1-vs-MHC0 section comparison table."""
    mhh = summary[summary["cohort"] == "MHH1_psychotic"].set_index("section")
    mhc0 = summary[summary["cohort"] == "MHC0"].set_index("section")
    shared_sections = [section for section in mhh.index if section in mhc0.index]

    rows = []
    for section in shared_sections:
        mhh_row = mhh.loc[section]
        mhc0_row = mhc0.loc[section]
        rows.append(
            {
                "section": section,
                "mhh_n_with_section": int(mhh_row["n_with_section"]),
                "mhc0_n_with_section": int(mhc0_row["n_with_section"]),
                "mhh_pct_with_section": float(mhh_row["pct_with_section"]),
                "mhc0_pct_with_section": float(mhc0_row["pct_with_section"]),
                "pct_point_difference_mhh_minus_mhc0": float(
                    mhh_row["pct_with_section"] - mhc0_row["pct_with_section"]
                ),
                "mhh_median_words_nonempty": float(mhh_row["words_nonempty_median"]),
                "mhc0_median_words_nonempty": float(mhc0_row["words_nonempty_median"]),
                "median_word_difference_mhh_minus_mhc0": float(
                    mhh_row["words_nonempty_median"] - mhc0_row["words_nonempty_median"]
                ),
                "mhh_median_chars_nonempty": float(mhh_row["chars_nonempty_median"]),
                "mhc0_median_chars_nonempty": float(mhc0_row["chars_nonempty_median"]),
                "median_char_difference_mhh_minus_mhc0": float(
                    mhh_row["chars_nonempty_median"] - mhc0_row["chars_nonempty_median"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "pct_point_difference_mhh_minus_mhc0",
        key=lambda values: values.abs(),
        ascending=False,
    )


def main() -> None:
    """Write section-level descriptive statistics for the matched full notes."""
    parser = load_parser_module()
    section_columns = list(parser.CANONICAL_SECTIONS) + ["unsectioned_text"]

    group_summaries = []
    note_summaries = []
    for cohort, df in load_full_note_outputs(section_columns):
        group_summaries.append(section_summary_for_group(df, cohort, section_columns))
        note_summaries.append(note_structure_summary_for_group(df, cohort))

    section_summary = pd.concat(group_summaries, ignore_index=True)
    comparison = build_section_comparison(section_summary)
    note_summary = pd.DataFrame(note_summaries)

    OUTPUT_DIR.mkdir(exist_ok=True)
    section_summary.to_csv(
        OUTPUT_DIR / "section_length_summary_by_group.csv",
        index=False,
    )
    comparison.to_csv(
        OUTPUT_DIR / "section_length_comparison.csv",
        index=False,
    )
    note_summary.to_csv(
        OUTPUT_DIR / "note_level_structure_summary.csv",
        index=False,
    )

    print(f"Saved section descriptive statistics to: {OUTPUT_DIR}")
    print()
    print("Largest coverage differences:")
    print(comparison.head(10).to_string(index=False))
    print()
    print("Note-level structure:")
    print(note_summary.to_string(index=False))


if __name__ == "__main__":
    main()
