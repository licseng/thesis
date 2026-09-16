"""Prepare note-level diagnostic-overshadowing annotation and inspection CSVs.

This is a companion to 01_prepare_annotation_csvs.py. It keeps the previous
section-level CSVs intact and writes new files where each row is a full
discharge note or a targeted positive example from the second-stage classifier.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "annotation_input"

PSYCH_DIR = PROJECT_DIR / "06_classification" / "01_classification_psych_integrated"
DIAGNOSTIC_DIR = (
    PROJECT_DIR / "06_classification" / "02_classification_diag_overshadowing"
)
FULL_NOTE_DIR = (
    PROJECT_DIR
    / "01_discharge_note_preprocessing"
    / "01_discharge_note_parsing"
    / "full_discharge_note_sections"
)

PREFILTER_INPUT_PATH = (
    PSYCH_DIR / "psych_history_llm_input" / "filtered_psych_keyword_section_input.parquet"
)
FULL_MHH1_NOTES_PATH = (
    FULL_NOTE_DIR / "MHH1_psychotic_matched_full_discharge_note_sections.parquet"
)
SECOND_STAGE_RESULTS_PATH = (
    DIAGNOSTIC_DIR
    / "diagnostic_overshadowing_classifier_output"
    / "diagnostic_overshadowing_section_classifier_results.csv"
)

MHH1_COHORT = "MHH1_psychotic"
RANDOM_SEED = 20260916
N_PREFILTER_NOTES = 20
N_POSITIVE_NOTES = 15
ANNOTATION_TEXT_WRAP_WIDTH = 110


def require_file(path: Path) -> None:
    """Fail clearly if an upstream file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")


def format_text_for_csv_cell(text: str) -> str:
    """Wrap text with embedded line breaks for easier spreadsheet inspection."""
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


def normalize_ids(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize common ID columns for stable joins/exports."""
    output = table.copy()
    for column in ["subject_id", "hadm_id"]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="raise").astype(int)
    if "cohort" in output.columns:
        output["cohort"] = output["cohort"].astype(str)
    return output


def load_full_mhh1_notes() -> pd.DataFrame:
    """Load one full discharge note row per MHH1 admission."""
    require_file(FULL_MHH1_NOTES_PATH)
    notes = pd.read_parquet(
        FULL_MHH1_NOTES_PATH,
        columns=[
            "cohort",
            "subject_id",
            "hadm_id",
            "note_id",
            "charttime",
            "chief_complaint",
            "n_detected_sections",
            "detected_section_headings",
            "full_note_text",
        ],
    )
    notes = normalize_ids(notes)
    notes = notes.loc[
        notes["cohort"].eq(MHH1_COHORT)
        & notes["full_note_text"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    notes = notes.drop_duplicates(subset=["cohort", "subject_id", "hadm_id"])
    return notes


def load_prefilter_admission_ids() -> pd.DataFrame:
    """Load admissions with at least one first-stage keyword-prefilter section."""
    require_file(PREFILTER_INPUT_PATH)
    prefilter = pd.read_parquet(
        PREFILTER_INPUT_PATH,
        columns=[
            "cohort",
            "subject_id",
            "hadm_id",
            "n_psych_keyword_hits",
            "psych_keyword_groups",
            "matched_terms",
        ],
    )
    prefilter = normalize_ids(prefilter)
    prefilter = prefilter.loc[prefilter["cohort"].eq(MHH1_COHORT)].copy()
    grouped = (
        prefilter.groupby(["cohort", "subject_id", "hadm_id"], as_index=False)
        .agg(
            n_prefilter_section_rows=("hadm_id", "size"),
            total_psych_keyword_hits=("n_psych_keyword_hits", "sum"),
            psych_keyword_groups=(
                "psych_keyword_groups",
                lambda values: " | ".join(sorted(set(" | ".join(values.dropna()).split(" | ")))),
            ),
            matched_terms=(
                "matched_terms",
                lambda values: " | ".join(sorted(set(" | ".join(values.dropna()).split(" | ")))),
            ),
        )
    )
    grouped["psych_keyword_groups"] = grouped["psych_keyword_groups"].str.strip(" |")
    grouped["matched_terms"] = grouped["matched_terms"].str.strip(" |")
    return grouped


def load_second_stage_results() -> pd.DataFrame:
    """Load second-stage diagnostic-overshadowing section results."""
    require_file(SECOND_STAGE_RESULTS_PATH)
    require_file(PREFILTER_INPUT_PATH)
    results = pd.read_csv(SECOND_STAGE_RESULTS_PATH)
    results = normalize_ids(results)
    results["diagnostic_overshadowing_label"] = (
        results["diagnostic_overshadowing_label"].fillna("").astype(str).str.lower()
    )

    section_text = pd.read_parquet(
        PREFILTER_INPUT_PATH,
        columns=["classifier_row_id", "section_text"],
    ).rename(
        columns={
            "classifier_row_id": "first_llm_classifier_row_id",
            "section_text": "section_text",
        }
    )
    results = results.merge(
        section_text,
        on="first_llm_classifier_row_id",
        how="left",
        validate="many_to_one",
    )
    results["section_text"] = results["section_text"].fillna("").astype(str)
    return results


def summarize_positive_sections(results: pd.DataFrame) -> pd.DataFrame:
    """Create admission-level summaries of positive second-stage sections."""
    positives = results.loc[results["diagnostic_overshadowing_label"].eq("positive")].copy()
    if positives.empty:
        return pd.DataFrame(
            columns=[
                "cohort",
                "subject_id",
                "hadm_id",
                "n_positive_sections",
                "positive_section_names",
                "positive_evidence_spans",
                "positive_reasons",
            ]
        )

    return (
        positives.groupby(["cohort", "subject_id", "hadm_id"], as_index=False)
        .agg(
            n_positive_sections=("hadm_id", "size"),
            positive_section_names=(
                "section_name",
                lambda values: " | ".join(values.fillna("").astype(str)),
            ),
            positive_evidence_spans=(
                "evidence_span",
                lambda values: " || ".join(values.fillna("").astype(str)),
            ),
            positive_section_texts=(
                "section_text",
                lambda values: " || ".join(values.fillna("").astype(str)),
            ),
            positive_reasons=(
                "reason",
                lambda values: " || ".join(values.fillna("").astype(str)),
            ),
        )
    )


def build_prefilter_note_annotation_sample(notes: pd.DataFrame) -> pd.DataFrame:
    """Sample full discharge notes from admissions with keyword-prefilter hits."""
    prefilter = load_prefilter_admission_ids()
    eligible = prefilter.merge(
        notes,
        on=["cohort", "subject_id", "hadm_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(eligible) < N_PREFILTER_NOTES:
        raise ValueError(
            f"Cannot sample {N_PREFILTER_NOTES} prefilter notes; only {len(eligible)} available."
        )
    sampled = eligible.sample(n=N_PREFILTER_NOTES, random_state=RANDOM_SEED).copy()
    sampled = sampled.sample(frac=1, random_state=RANDOM_SEED + 1).reset_index(drop=True)
    sampled.insert(0, "annotation_row_id", range(1, len(sampled) + 1))

    return pd.DataFrame(
        {
            "annotation_row_id": sampled["annotation_row_id"],
            "hadm_id": sampled["hadm_id"],
            "chief_complaint": sampled["chief_complaint"].fillna("").astype(str),
            "full_discharge_note": sampled["full_note_text"].map(format_text_for_csv_cell),
            "stage1_psychiatric_context": "",
            "stage2_diagnostic_overshadowing": "",
        }
    )


def build_positive_note_examples(notes: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Sample full discharge notes with at least one positive second-stage section."""
    positive_summaries = summarize_positive_sections(results)
    positive_notes = positive_summaries.merge(
        notes,
        on=["cohort", "subject_id", "hadm_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(positive_notes) < N_POSITIVE_NOTES:
        raise ValueError(
            f"Cannot sample {N_POSITIVE_NOTES} positive notes; only {len(positive_notes)} available."
        )
    sampled = positive_notes.sample(n=N_POSITIVE_NOTES, random_state=RANDOM_SEED + 10).copy()
    sampled = sampled.sample(frac=1, random_state=RANDOM_SEED + 11).reset_index(drop=True)
    sampled.insert(0, "inspection_row_id", range(1, len(sampled) + 1))

    return pd.DataFrame(
        {
            "inspection_row_id": sampled["inspection_row_id"],
            "hadm_id": sampled["hadm_id"],
            "subject_id": sampled["subject_id"],
            "note_id": sampled["note_id"],
            "chief_complaint": sampled["chief_complaint"].fillna("").astype(str),
            "n_positive_sections": sampled["n_positive_sections"],
            "positive_section_names": sampled["positive_section_names"],
            "positive_section_texts": sampled["positive_section_texts"].map(
                format_text_for_csv_cell
            ),
            "positive_evidence_spans": sampled["positive_evidence_spans"].map(
                format_text_for_csv_cell
            ),
            "positive_reasons": sampled["positive_reasons"].map(format_text_for_csv_cell),
            "full_discharge_note": sampled["full_note_text"].map(format_text_for_csv_cell),
        }
    )


def build_positive_example_per_section(results: pd.DataFrame) -> pd.DataFrame:
    """Sample one positive second-stage row for each positive section_name."""
    positives = results.loc[results["diagnostic_overshadowing_label"].eq("positive")].copy()
    if positives.empty:
        raise ValueError("No positive second-stage classifier rows found.")

    sampled = (
        positives.sort_values(["section_name", "classifier_row_id"])
        .groupby("section_name", group_keys=False)
        .sample(n=1, random_state=RANDOM_SEED + 20)
        .sort_values("section_name")
        .reset_index(drop=True)
    )
    sampled.insert(0, "inspection_row_id", range(1, len(sampled) + 1))

    columns = [
        "inspection_row_id",
        "section_name",
        "hadm_id",
        "subject_id",
        "note_id",
        "diagnostic_overshadowing_label",
        "section_text",
        "psychiatric_context_present",
        "physical_or_medical_problem_present",
        "attribution_to_psychiatric_condition",
        "possible_missed_or_delayed_workup",
        "evidence_span",
        "reason",
        "matched_terms",
        "psych_keyword_groups",
        "first_llm_psychiatric_context_label",
        "first_llm_psychiatric_mention_type",
    ]
    output = sampled.loc[:, [column for column in columns if column in sampled.columns]].copy()
    for column in ["section_text", "evidence_span", "reason"]:
        if column in output.columns:
            output[column] = output[column].map(format_text_for_csv_cell)
    return output


def write_summary(
    prefilter_notes: pd.DataFrame,
    positive_notes: pd.DataFrame,
    positive_by_section: pd.DataFrame,
) -> None:
    """Write a compact audit summary for the generated note-level CSVs."""
    summary = pd.DataFrame(
        [
            {
                "file": "diagnostic_overshadowing_note_level_annotation_sample_v2.csv",
                "n_rows": len(prefilter_notes),
                "n_admissions": prefilter_notes["hadm_id"].nunique(),
            },
            {
                "file": "diagnostic_overshadowing_positive_note_examples_v2.csv",
                "n_rows": len(positive_notes),
                "n_admissions": positive_notes["hadm_id"].nunique(),
            },
            {
                "file": "diagnostic_overshadowing_positive_example_per_section_v2.csv",
                "n_rows": len(positive_by_section),
                "n_admissions": positive_by_section["hadm_id"].nunique(),
            },
        ]
    )
    summary.to_csv(
        OUTPUT_DIR / "diagnostic_overshadowing_note_level_annotation_summary_v2.csv",
        index=False,
    )


def main() -> None:
    """Build and save note-level diagnostic annotation/inspection CSVs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    notes = load_full_mhh1_notes()
    results = load_second_stage_results()

    prefilter_notes = build_prefilter_note_annotation_sample(notes)
    positive_notes = build_positive_note_examples(notes, results)
    positive_by_section = build_positive_example_per_section(results)

    prefilter_path = (
        OUTPUT_DIR / "diagnostic_overshadowing_note_level_annotation_sample_v2.csv"
    )
    positive_notes_path = (
        OUTPUT_DIR / "diagnostic_overshadowing_positive_note_examples_v2.csv"
    )
    positive_sections_path = (
        OUTPUT_DIR / "diagnostic_overshadowing_positive_example_per_section_v2.csv"
    )

    prefilter_notes.to_csv(prefilter_path, index=False)
    positive_notes.to_csv(positive_notes_path, index=False)
    positive_by_section.to_csv(positive_sections_path, index=False)
    write_summary(prefilter_notes, positive_notes, positive_by_section)

    print(f"Saved note-level annotation sample: {prefilter_path}")
    print(f"Saved positive full-note examples: {positive_notes_path}")
    print(f"Saved one positive example per section: {positive_sections_path}")
    print(
        "\nRows:",
        {
            "prefilter_note_rows": len(prefilter_notes),
            "positive_note_rows": len(positive_notes),
            "positive_section_example_rows": len(positive_by_section),
        },
    )
    print("\nPositive examples by section:")
    print(positive_by_section["section_name"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
