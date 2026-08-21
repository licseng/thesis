"""Prepare selected section input for the sentiment classifier.

This script builds the uploadable sentiment-analysis input table from:
    - the parsed full discharge-note section parquet files, and
    - optional all-section SL keyword exploration metadata.

It keeps every non-empty note-section row from the sections selected for
sentiment analysis. The saved parquet contains the full section text, plus
SL-keyword-hit metadata where available, so
`01_run_sentiment_section_classifier.py` can run directly on it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PYTHON_DIR = SCRIPT_DIR.parent
PARSER_DIR = REPO_PYTHON_DIR / "01_discharge_note_preprocessing" / "01_discharge_note_parsing"
FULL_NOTE_DIR = PARSER_DIR / "full_discharge_note_sections"
SL_OUTPUT_DIR = REPO_PYTHON_DIR / "03_discharge_note_text_analysis" / "analysis_output_SL_keyword_exploration"
SL_SECTION_HITS_PATH = SL_OUTPUT_DIR / "SL_keyword_section_hits.csv"
OUTPUT_DIR = SCRIPT_DIR / "sentiment_llm_input"

SELECTED_SECTION_NAMES = [
    "brief_hospital_course",
    "present_illness",
    "problems",
    "medical_history",
    "pertinent_results",
    "physical_exam",
    "discharge_instructions",
    "medication_admission",
    "discharge_medications",
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

ID_COLUMNS = ["cohort", "subject_id", "hadm_id", "note_id"]
NOTE_METADATA_COLUMNS = [
    "cohort",
    "subject_id",
    "hadm_id",
    "note_id",
    "charttime",
    "admittime",
    "sex",
    "age_at_admission",
]
SL_METADATA_COLUMNS = [
    "n_keyword_hits",
    "n_keyword_groups",
    "keyword_groups",
    "matched_terms",
    "keyword_hits_per_1000_words",
]


def validate_inputs() -> None:
    """Fail clearly if required upstream files have not been generated."""
    missing = []
    for file_config in FULL_NOTE_FILES:
        if not file_config["path"].exists():
            missing.append(file_config["path"])
    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n" + "\n".join(str(path) for path in missing)
        )


def load_sl_section_hits() -> pd.DataFrame:
    """Load optional SL-hit metadata for the selected sentiment sections."""
    if not SL_SECTION_HITS_PATH.exists():
        columns = ID_COLUMNS + ["section_name"] + SL_METADATA_COLUMNS
        return pd.DataFrame(columns=columns)

    hits = pd.read_csv(SL_SECTION_HITS_PATH)
    required = set(ID_COLUMNS + ["section_name"] + SL_METADATA_COLUMNS)
    missing = sorted(required - set(hits.columns))
    if missing:
        raise ValueError(f"SL section hits file is missing columns: {', '.join(missing)}")

    hits = hits.loc[hits["section_name"].isin(SELECTED_SECTION_NAMES)].copy()
    hits = hits.drop_duplicates(subset=ID_COLUMNS + ["section_name"])
    return hits


def load_section_text_long() -> pd.DataFrame:
    """Load parsed discharge-note sections and reshape selected sections to long form."""
    frames = []
    columns = NOTE_METADATA_COLUMNS + SELECTED_SECTION_NAMES
    for file_config in FULL_NOTE_FILES:
        df = pd.read_parquet(file_config["path"], columns=columns)
        df["cohort"] = file_config["cohort"]
        frames.append(df)

    notes = pd.concat(frames, ignore_index=True)
    long_df = notes.melt(
        id_vars=NOTE_METADATA_COLUMNS,
        value_vars=SELECTED_SECTION_NAMES,
        var_name="section_name",
        value_name="section_text",
    )
    long_df["section_text"] = long_df["section_text"].fillna("").astype(str).str.strip()
    long_df = long_df.loc[long_df["section_text"].ne("")].copy()
    return long_df


def build_sentiment_input() -> pd.DataFrame:
    """Join optional SL-hit metadata to all selected section text."""
    sl_hits = load_sl_section_hits()
    section_text = load_section_text_long()

    merged = section_text.merge(
        sl_hits,
        on=ID_COLUMNS + ["section_name"],
        how="left",
        validate="one_to_one",
    )
    merged["has_sl_keyword_hit"] = merged["n_keyword_hits"].notna()
    merged["n_keyword_hits"] = merged["n_keyword_hits"].fillna(0).astype(int)
    merged["n_keyword_groups"] = merged["n_keyword_groups"].fillna(0).astype(int)
    merged["keyword_hits_per_1000_words"] = (
        merged["keyword_hits_per_1000_words"].fillna(0).astype(float)
    )
    merged["keyword_groups"] = merged["keyword_groups"].fillna("")
    merged["matched_terms"] = merged["matched_terms"].fillna("")

    merged = merged.sort_values(["cohort", "subject_id", "hadm_id", "section_name"]).reset_index(drop=True)
    merged.insert(0, "sentiment_input_row_id", range(len(merged)))

    output_columns = [
        "sentiment_input_row_id",
        "cohort",
        "subject_id",
        "hadm_id",
        "note_id",
        "charttime",
        "admittime",
        "sex",
        "age_at_admission",
        "section_name",
        "section_text",
        "section_word_count",
        "section_char_length",
        "has_sl_keyword_hit",
        *SL_METADATA_COLUMNS,
    ]
    return merged.loc[:, output_columns]


def write_outputs(sentiment_input: pd.DataFrame) -> None:
    """Save classifier input and compact review summaries."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = OUTPUT_DIR / "sentiment_selected_section_input.parquet"
    metadata_path = OUTPUT_DIR / "sentiment_selected_section_input_metadata.csv"
    section_summary_path = OUTPUT_DIR / "sentiment_selected_section_input_section_summary.csv"
    cohort_summary_path = OUTPUT_DIR / "sentiment_selected_section_input_cohort_summary.csv"

    sentiment_input.to_parquet(parquet_path, index=False)
    sentiment_input.drop(columns=["section_text"]).to_csv(metadata_path, index=False)

    section_summary = (
        sentiment_input.groupby(["cohort", "section_name"], as_index=False)
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_section_rows=("section_name", "size"),
            n_section_rows_with_sl_keyword=("has_sl_keyword_hit", "sum"),
            total_sl_keyword_hits=("n_keyword_hits", "sum"),
            median_section_words=("section_word_count", "median"),
            mean_section_words=("section_word_count", "mean"),
        )
        .sort_values(["cohort", "n_section_rows"], ascending=[True, False])
    )
    section_summary.to_csv(section_summary_path, index=False)

    cohort_summary = (
        sentiment_input.groupby("cohort", as_index=False)
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_section_rows=("section_name", "size"),
            n_section_rows_with_sl_keyword=("has_sl_keyword_hit", "sum"),
            total_sl_keyword_hits=("n_keyword_hits", "sum"),
            median_sections_per_admission=("hadm_id", lambda x: x.value_counts().median()),
            max_sections_per_admission=("hadm_id", lambda x: x.value_counts().max()),
        )
        .sort_values("cohort")
    )
    cohort_summary.to_csv(cohort_summary_path, index=False)

    print(f"Saved sentiment input parquet: {parquet_path}")
    print(f"Saved metadata CSV: {metadata_path}")
    print(f"Saved section summary: {section_summary_path}")
    print(f"Saved cohort summary: {cohort_summary_path}")
    print("\n=== Cohort Summary ===")
    print(cohort_summary.to_string(index=False))
    print("\n=== Section Summary ===")
    print(section_summary.to_string(index=False))


def main() -> None:
    """Build and save the sentiment classifier input dataset."""
    validate_inputs()
    sentiment_input = build_sentiment_input()
    write_outputs(sentiment_input)


if __name__ == "__main__":
    main()
