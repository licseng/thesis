"""Prepare all non-empty selected sections from BOTH full matched cohorts.

Sentiment eligibility does not depend on psychiatric or SL keyword hits.
Clinical text is written locally to the ignored input parquet, never printed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PYTHON_DIR = SCRIPT_DIR.parent
FULL_NOTE_DIR = (REPO_PYTHON_DIR / "01_discharge_note_preprocessing"
                 / "01_discharge_note_parsing" / "full_discharge_note_sections")
MATCHED_PAIRS_PATH = (REPO_PYTHON_DIR / "02_cohort_matching"
                      / "matched_cohort_output" / "matched_pairs.parquet")
OUTPUT_DIR = SCRIPT_DIR / "sentiment_llm_input"

# Retained from the earlier sentiment workflow; review before final inference.
SELECTED_SECTION_NAMES = [
    "brief_hospital_course", "present_illness", "problems", "medical_history",
    "pertinent_results", "physical_exam", "discharge_instructions",
    "medication_admission", "discharge_medications",
]
FULL_NOTE_FILES = [
    {"cohort": cohort, "path": FULL_NOTE_DIR / f"{cohort}_matched_full_discharge_note_sections.parquet"}
    for cohort in ["MHC1_psychotic", "MHC0"]
]
ID_COLUMNS = ["cohort", "subject_id", "hadm_id", "note_id"]
NOTE_METADATA_COLUMNS = [
    *ID_COLUMNS, "charttime", "admittime", "sex", "age_at_admission",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs() -> None:
    required_paths = [MATCHED_PAIRS_PATH, *(item["path"] for item in FULL_NOTE_FILES)]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n" + "\n".join(missing))


def load_matched_admissions() -> pd.DataFrame:
    columns = ["pair_id", "mhc1_subject_id", "mhc1_hadm_id", "mhc0_subject_id", "mhc0_hadm_id"]
    pairs = pd.read_parquet(MATCHED_PAIRS_PATH, columns=columns)
    if pairs.empty or pairs.isna().any().any() or pairs["pair_id"].duplicated().any():
        raise ValueError("Matched pairs must have non-null IDs and unique pair IDs.")
    frames = []
    for prefix, cohort in [("mhc1", "MHC1_psychotic"), ("mhc0", "MHC0")]:
        frame = pairs[["pair_id", f"{prefix}_subject_id", f"{prefix}_hadm_id"]].rename(
            columns={f"{prefix}_subject_id": "subject_id", f"{prefix}_hadm_id": "hadm_id"}
        )
        frames.append(frame.assign(cohort=cohort))
    admissions = pd.concat(frames, ignore_index=True)
    if admissions["hadm_id"].duplicated().any():
        raise ValueError("An admission appears in more than one matched position.")
    if admissions.groupby("subject_id")["cohort"].nunique().gt(1).any():
        raise ValueError("A patient appears in both cohorts.")
    return admissions


def build_sentiment_input() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return non-empty section rows and coverage of every matched admission."""
    matched = load_matched_admissions()
    frames = []
    for config in FULL_NOTE_FILES:
        notes = pd.read_parquet(config["path"], columns=NOTE_METADATA_COLUMNS + SELECTED_SECTION_NAMES)
        if notes[ID_COLUMNS].isna().any().any():
            raise ValueError(f"Null note identity in parsed {config['cohort']} input.")
        if not notes["cohort"].eq(config["cohort"]).all():
            raise ValueError("Parsed note cohort labels do not agree with their input file.")
        if notes["hadm_id"].duplicated().any():
            raise ValueError("Expected one parsed discharge note per matched admission.")
        expected = matched.loc[matched["cohort"].eq(config["cohort"])]
        identity = notes[["cohort", "subject_id", "hadm_id"]].merge(
            expected, on=["cohort", "subject_id", "hadm_id"], how="outer",
            validate="one_to_one", indicator=True,
        )
        if not identity["_merge"].eq("both").all():
            raise ValueError("Parsed note admissions differ from the current matched cohort; rerun parsing.")
        frames.append(notes.merge(expected, on=["cohort", "subject_id", "hadm_id"], validate="one_to_one"))

    notes = pd.concat(frames, ignore_index=True)
    sections = notes.melt(
        id_vars=["pair_id", *NOTE_METADATA_COLUMNS], value_vars=SELECTED_SECTION_NAMES,
        var_name="section_name", value_name="section_text",
    )
    sections["section_text"] = sections["section_text"].fillna("").astype(str).str.strip()
    sections = sections.loc[sections["section_text"].ne("")].copy()
    # Computed from the actual input, not from a keyword-hit table.
    sections["section_word_count"] = sections["section_text"].str.count(r"\S+")
    sections["section_char_length"] = sections["section_text"].str.len()
    sections = sections.sort_values(["cohort", "subject_id", "hadm_id", "note_id", "section_name"]).reset_index(drop=True)
    if sections.duplicated(ID_COLUMNS + ["section_name"]).any():
        raise ValueError("Duplicate sentiment section identities.")
    sections.insert(0, "sentiment_input_row_id", range(len(sections)))

    counts = sections.groupby(["cohort", "subject_id", "hadm_id"]).size().rename("n_nonempty_selected_sections").reset_index()
    coverage = matched.merge(counts, on=["cohort", "subject_id", "hadm_id"], how="left", validate="one_to_one")
    coverage["n_nonempty_selected_sections"] = coverage["n_nonempty_selected_sections"].fillna(0).astype(int)
    coverage["has_eligible_section"] = coverage["n_nonempty_selected_sections"].gt(0)
    return sections, coverage


def write_outputs(sections: pd.DataFrame, coverage: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = OUTPUT_DIR / "sentiment_selected_section_input.parquet"
    sections.to_parquet(input_path, index=False)
    sections.drop(columns="section_text").to_csv(OUTPUT_DIR / "sentiment_selected_section_input_metadata.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "sentiment_admission_input_coverage.csv", index=False)

    section_summary = sections.groupby(["cohort", "section_name"], as_index=False).agg(
        n_admissions=("hadm_id", "nunique"), n_section_rows=("section_name", "size"),
        median_section_words=("section_word_count", "median"), mean_section_words=("section_word_count", "mean"),
    )
    grid = pd.MultiIndex.from_product(
        [["MHC0", "MHC1_psychotic"], SELECTED_SECTION_NAMES], names=["cohort", "section_name"]
    ).to_frame(index=False)
    section_summary = grid.merge(section_summary, how="left", on=["cohort", "section_name"], validate="one_to_one")
    for column in ["n_admissions", "n_section_rows"]:
        section_summary[column] = section_summary[column].fillna(0).astype(int)
    section_summary.to_csv(OUTPUT_DIR / "sentiment_selected_section_input_section_summary.csv", index=False)
    cohort_summary = coverage.groupby("cohort", as_index=False).agg(
        n_matched_subjects=("subject_id", "nunique"), n_matched_admissions=("hadm_id", "size"),
        n_admissions_with_eligible_sections=("has_eligible_section", "sum"),
        n_section_rows=("n_nonempty_selected_sections", "sum"),
        median_sections_per_admission=("n_nonempty_selected_sections", "median"),
        max_sections_per_admission=("n_nonempty_selected_sections", "max"),
    )
    cohort_summary["n_admissions_without_eligible_sections"] = (
        cohort_summary["n_matched_admissions"] - cohort_summary["n_admissions_with_eligible_sections"]
    )
    cohort_summary.to_csv(OUTPUT_DIR / "sentiment_selected_section_input_cohort_summary.csv", index=False)
    manifest = {
        "selection": "all non-empty selected sections of both full matched cohorts; no keyword gate",
        "sections": SELECTED_SECTION_NAMES,
        "sources": [{"path": str(path), "sha256": sha256_file(path)} for path in
                    [MATCHED_PAIRS_PATH, *(config["path"] for config in FULL_NOTE_FILES)]],
        "input_sha256": sha256_file(input_path),
        "n_section_rows": len(sections), "n_matched_admissions": len(coverage),
    }
    (OUTPUT_DIR / "sentiment_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Sentiment input saved locally; no API request was made.")
    print(cohort_summary.to_string(index=False))
    print(section_summary.to_string(index=False))


def main() -> None:
    validate_inputs()
    write_outputs(*build_sentiment_input())


if __name__ == "__main__":
    main()
