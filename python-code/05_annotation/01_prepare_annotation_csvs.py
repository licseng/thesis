"""Prepare blinded human-annotation CSV files.

This script creates two annotator-facing CSV files:

1. Diagnostic-overshadowing annotation:
   - samples 100 MHH1 admissions from the keyword-prefiltered section input
   - one row per sampled candidate section
   - annotators fill psychiatric-context and diagnostic-overshadowing yes/no/unclear labels

2. Sentiment annotation:
   - samples 50 MHH1 and 50 MHC0 admissions from the sentiment input dataset
   - one row per selected sentiment section
   - annotators fill positive/negative/neutral/mixed label

The annotator CSVs are blinded to cohort, section name, and model output.
Private key CSVs are written alongside the workbooks so rows can be mapped back
to the source records after annotation.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CLASSIFICATION_DIR = PROJECT_DIR / "06_classification"
PSYCH_DIR = CLASSIFICATION_DIR / "01_classification_psych_integrated"
SENTIMENT_DIR = PROJECT_DIR / "07_sentiment_analysis"
CHIEF_COMPLAINT_DIR = (
    PROJECT_DIR
    / "01_discharge_note_preprocessing"
    / "02_chief_complaint"
    / "01_preprocessing"
    / "chief_complaint_final"
)
OUTPUT_DIR = SCRIPT_DIR / "annotation_input"

RANDOM_SEED = 20260911
N_DIAGNOSTIC_ADMISSIONS = 100
N_SENTIMENT_ADMISSIONS_PER_COHORT = 50

DIAGNOSTIC_FIRST_STAGE_INPUT_PATH = (
    PSYCH_DIR / "psych_history_llm_input" / "filtered_psych_keyword_section_input.parquet"
)
SENTIMENT_INPUT_PATH = (
    SENTIMENT_DIR / "sentiment_llm_input" / "sentiment_selected_section_input.parquet"
)
MHH1_CHIEF_COMPLAINT_PATH = (
    CHIEF_COMPLAINT_DIR / "MHH1_psychotic_chief_complaints_final.parquet"
)

MHH1_COHORT = "MHH1_psychotic"
MHC0_COHORT = "MHC0"
YES_NO_UNCLEAR = ["yes", "no", "unclear"]
SENTIMENT_LABELS = ["positive", "negative", "neutral", "mixed"]
ANNOTATION_TEXT_WRAP_WIDTH = 110


def require_file(path: Path) -> None:
    """Fail clearly if an upstream input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")


def normalize_id_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize ID columns for stable exports and joins."""
    output = data.copy()
    output["cohort"] = output["cohort"].astype(str)
    output["subject_id"] = pd.to_numeric(output["subject_id"], errors="raise").astype(int)
    output["hadm_id"] = pd.to_numeric(output["hadm_id"], errors="raise").astype(int)
    return output


def format_section_text_for_annotation(text: str) -> str:
    """Add line breaks inside long section text for easier spreadsheet review.

    CSV cannot store column widths or wrap settings, but quoted fields can
    contain newlines. Excel/Numbers/LibreOffice will keep these line breaks
    inside the cell when the CSV is opened.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = []
    for paragraph in text.split("\n"):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        paragraphs.append(
            textwrap.fill(
                paragraph,
                width=ANNOTATION_TEXT_WRAP_WIDTH,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n\n".join(paragraphs)


def sample_grouped_admissions(
    data: pd.DataFrame,
    n_admissions: int,
    *,
    seed: int,
    group_name: str,
) -> pd.DataFrame:
    """Sample admissions and keep all rows belonging to sampled admissions."""
    unique_hadm_ids = pd.Series(sorted(data["hadm_id"].dropna().unique()))
    if len(unique_hadm_ids) < n_admissions:
        raise ValueError(
            f"Cannot sample {n_admissions} admissions for {group_name}; "
            f"only {len(unique_hadm_ids)} available."
        )

    sampled_hadm_ids = unique_hadm_ids.sample(
        n=n_admissions,
        random_state=seed,
        replace=False,
    )
    return data.loc[data["hadm_id"].isin(sampled_hadm_ids)].copy()


def load_mhh1_chief_complaints() -> pd.DataFrame:
    """Load chief complaint context for MHH1 diagnostic annotation rows."""
    require_file(MHH1_CHIEF_COMPLAINT_PATH)
    chief = pd.read_parquet(
        MHH1_CHIEF_COMPLAINT_PATH,
        columns=["subject_id", "hadm_id", "chief_complaint_raw", "chief_complaint_normalized"],
    )
    chief = normalize_id_columns(chief.assign(cohort=MHH1_COHORT))
    chief["chief_complaint"] = (
        chief["chief_complaint_raw"]
        .fillna(chief["chief_complaint_normalized"])
        .fillna("")
        .astype(str)
        .str.strip()
    )
    chief = chief.drop_duplicates(subset=["cohort", "subject_id", "hadm_id"])
    return chief.loc[:, ["cohort", "subject_id", "hadm_id", "chief_complaint"]]


def build_diagnostic_annotation_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create annotator and key tables for diagnostic-overshadowing validation."""
    require_file(DIAGNOSTIC_FIRST_STAGE_INPUT_PATH)

    merged = pd.read_parquet(DIAGNOSTIC_FIRST_STAGE_INPUT_PATH)
    merged = normalize_id_columns(merged)
    chief_complaints = load_mhh1_chief_complaints()
    merged = merged.merge(
        chief_complaints,
        on=["cohort", "subject_id", "hadm_id"],
        how="left",
        validate="many_to_one",
    )
    merged["chief_complaint"] = merged["chief_complaint"].fillna("").astype(str).str.strip()
    merged["section_text"] = merged["section_text"].fillna("").astype(str).str.strip()
    merged = merged.loc[
        merged["cohort"].eq(MHH1_COHORT)
        & merged["section_text"].ne("")
    ].copy()

    sampled = sample_grouped_admissions(
        merged,
        N_DIAGNOSTIC_ADMISSIONS,
        seed=RANDOM_SEED,
        group_name="diagnostic-overshadowing annotation",
    )
    sampled = sampled.sample(frac=1, random_state=RANDOM_SEED + 1).reset_index(drop=True)
    sampled.insert(0, "annotation_row_id", range(1, len(sampled) + 1))

    annotator = pd.DataFrame(
        {
            "hadm_id": sampled["hadm_id"],
            "chief_complaint": sampled["chief_complaint"].map(
                format_section_text_for_annotation
            ),
            "section_name": sampled["section_name"],
            "section_text": sampled["section_text"].map(format_section_text_for_annotation),
            "stage1_psychiatric_context": "",
            "stage2_diagnostic_overshadowing": "",
        }
    )

    key_columns = [
        "annotation_row_id",
        "cohort",
        "subject_id",
        "hadm_id",
        "note_id",
        "charttime",
        "classifier_row_id",
        "chief_complaint",
        "section_name",
        "section_word_count",
        "section_char_length",
        "n_psych_keyword_hits",
        "psych_keyword_groups",
        "matched_terms",
        "psych_keyword_group_hit_counts",
    ]
    key = sampled.loc[:, [column for column in key_columns if column in sampled.columns]].copy()
    return annotator, key


def build_sentiment_annotation_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create annotator and key tables for sentiment validation."""
    require_file(SENTIMENT_INPUT_PATH)

    sentiment = pd.read_parquet(SENTIMENT_INPUT_PATH)
    sentiment = normalize_id_columns(sentiment)
    sentiment["section_text"] = sentiment["section_text"].fillna("").astype(str).str.strip()
    sentiment = sentiment.loc[sentiment["section_text"].ne("")].copy()

    mhh1 = sample_grouped_admissions(
        sentiment.loc[sentiment["cohort"].eq(MHH1_COHORT)].copy(),
        N_SENTIMENT_ADMISSIONS_PER_COHORT,
        seed=RANDOM_SEED,
        group_name="sentiment MHH1 annotation",
    )
    mhc0 = sample_grouped_admissions(
        sentiment.loc[sentiment["cohort"].eq(MHC0_COHORT)].copy(),
        N_SENTIMENT_ADMISSIONS_PER_COHORT,
        seed=RANDOM_SEED + 1,
        group_name="sentiment MHC0 annotation",
    )
    sampled = pd.concat([mhh1, mhc0], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=RANDOM_SEED + 2).reset_index(drop=True)
    sampled.insert(0, "annotation_row_id", range(1, len(sampled) + 1))

    annotator = pd.DataFrame(
        {
            "hadm_id": sampled["hadm_id"],
            "section_name": sampled["section_name"],
            "section_text": sampled["section_text"].map(format_section_text_for_annotation),
            "sentiment": "",
        }
    )

    key_columns = [
        "annotation_row_id",
        "cohort",
        "subject_id",
        "hadm_id",
        "note_id",
        "charttime",
        "admittime",
        "sex",
        "age_at_admission",
        "sentiment_input_row_id",
        "section_name",
        "section_word_count",
        "section_char_length",
        "has_sl_keyword_hit",
        "n_keyword_hits",
        "keyword_groups",
        "matched_terms",
    ]
    key = sampled.loc[:, [column for column in key_columns if column in sampled.columns]].copy()
    return annotator, key


def write_summary(
    diagnostic_key: pd.DataFrame,
    sentiment_key: pd.DataFrame,
    path: Path,
) -> None:
    """Write compact sampling summaries for audit/reproducibility."""
    rows = [
        {
            "workbook": "diagnostic_overshadowing",
            "n_admissions": diagnostic_key["hadm_id"].nunique(),
            "n_section_rows": len(diagnostic_key),
            "cohort_counts": diagnostic_key["cohort"].value_counts().to_dict(),
        },
        {
            "workbook": "sentiment",
            "n_admissions": sentiment_key["hadm_id"].nunique(),
            "n_section_rows": len(sentiment_key),
            "cohort_counts": sentiment_key["cohort"].value_counts().to_dict(),
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    """Build and save both human annotation CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    diagnostic_annotator, diagnostic_key = build_diagnostic_annotation_tables()
    sentiment_annotator, sentiment_key = build_sentiment_annotation_tables()

    diagnostic_csv = OUTPUT_DIR / "diagnostic_overshadowing_annotation_sample.csv"
    diagnostic_key_csv = OUTPUT_DIR / "diagnostic_overshadowing_annotation_sample_key.csv"
    sentiment_csv = OUTPUT_DIR / "sentiment_annotation_sample.csv"
    sentiment_key_csv = OUTPUT_DIR / "sentiment_annotation_sample_key.csv"
    summary_csv = OUTPUT_DIR / "annotation_sample_summary.csv"

    diagnostic_annotator.to_csv(diagnostic_csv, index=False)
    diagnostic_key.to_csv(diagnostic_key_csv, index=False)

    sentiment_annotator.to_csv(sentiment_csv, index=False)
    sentiment_key.to_csv(sentiment_key_csv, index=False)

    write_summary(diagnostic_key, sentiment_key, summary_csv)

    print(f"Saved diagnostic annotation CSV: {diagnostic_csv}")
    print(f"Allowed diagnostic labels: {', '.join(YES_NO_UNCLEAR)}")
    print(f"Saved diagnostic key: {diagnostic_key_csv}")
    print(f"Saved sentiment annotation CSV: {sentiment_csv}")
    print(f"Allowed sentiment labels: {', '.join(SENTIMENT_LABELS)}")
    print(f"Saved sentiment key: {sentiment_key_csv}")
    print(f"Saved summary: {summary_csv}")
    print("\n=== Diagnostic sample ===")
    print(
        diagnostic_key.groupby("cohort")
        .agg(n_admissions=("hadm_id", "nunique"), n_section_rows=("hadm_id", "size"))
        .to_string()
    )
    print("\n=== Sentiment sample ===")
    print(
        sentiment_key.groupby("cohort")
        .agg(n_admissions=("hadm_id", "nunique"), n_section_rows=("hadm_id", "size"))
        .to_string()
    )


if __name__ == "__main__":
    main()
