"""Print one parsed discharge-note section for local manual review.

This is a manual inspection helper for checking a specific parsed section
without printing the full discharge note.

Inputs:
    ../01_discharge_note_preprocessing/01_discharge_note_parsing/
        full_discharge_note_sections/
            MHH1_psychotic_matched_full_discharge_note_sections.parquet
            MHC0_matched_full_discharge_note_sections.parquet

Usage:
    python print_discharge_note_section.py \
        --subject-id 123 \
        --hadm-id 456 \
        --section-name brief_hospital_course

If subject_id, hadm_id, or section_name is omitted, the script prompts for it
interactively.

Privacy note:
    This script prints parsed discharge-note text to the terminal. Use it only
    in your own local terminal when you intentionally want to inspect the
    requested section.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FULL_NOTE_SECTION_DIR = (
    SCRIPT_DIR
    / "01_discharge_note_preprocessing"
    / "01_discharge_note_parsing"
    / "full_discharge_note_sections"
)

SECTION_FILES = [
    {
        "cohort": "MHH1_psychotic",
        "path": FULL_NOTE_SECTION_DIR
        / "MHH1_psychotic_matched_full_discharge_note_sections.parquet",
    },
    {
        "cohort": "MHC0",
        "path": FULL_NOTE_SECTION_DIR / "MHC0_matched_full_discharge_note_sections.parquet",
    },
]

METADATA_COLUMNS = ["cohort", "subject_id", "hadm_id", "note_id", "charttime"]
PARSED_SECTION_COLUMNS = [
    "chief_complaint",
    "major_surgical_or_invasive_procedure",
    "present_illness",
    "medical_history",
    "past_psychiatric_history",
    "medication_admission",
    "allergies",
    "review_of_systems",
    "physical_exam",
    "family_history",
    "social_history",
    "problems",
    "pertinent_results",
    "brief_hospital_course",
    "discharge_medications",
    "discharge_disposition",
    "discharge_diagnosis",
    "discharge_condition",
    "discharge_instructions",
    "unsectioned_text",
]


def parse_args() -> ArgumentParser:
    """Parse optional identifiers and section name."""
    parser = ArgumentParser(description="Print one parsed discharge-note section.")
    parser.add_argument("--subject-id", type=int, help="MIMIC subject_id.")
    parser.add_argument("--hadm-id", type=int, help="MIMIC hadm_id.")
    parser.add_argument(
        "--section-name",
        help="Parsed section column to print, e.g. brief_hospital_course.",
    )
    parser.add_argument(
        "--cohort",
        choices=[file_config["cohort"] for file_config in SECTION_FILES],
        help="Optional cohort filter if the same IDs appear in multiple cohorts.",
    )
    parser.add_argument(
        "--list-sections",
        action="store_true",
        help="List available parsed section names and exit.",
    )
    return parser


def prompt_for_int(label: str) -> int:
    """Prompt until the user enters an integer ID."""
    while True:
        value = input(f"{label}: ").strip()
        try:
            return int(value)
        except ValueError:
            print(f"Please enter a numeric {label}.")


def prompt_for_section(section_names: list[str]) -> str:
    """Prompt until the user enters one of the available section names."""
    print("Available sections:")
    for section_name in section_names:
        print(f"  {section_name}")

    while True:
        value = input("section_name: ").strip()
        if value in section_names:
            return value
        print("Unknown section_name. Please enter one of the listed names.")


def existing_section_files(cohort: str | None = None) -> list[dict[str, Path | str]]:
    """Return configured parsed-section files, optionally filtered by cohort."""
    file_configs = SECTION_FILES
    if cohort is not None:
        file_configs = [
            file_config for file_config in SECTION_FILES if file_config["cohort"] == cohort
        ]

    missing = [str(file_config["path"]) for file_config in file_configs if not file_config["path"].exists()]
    if missing:
        raise FileNotFoundError("Missing parsed section file(s):\n" + "\n".join(missing))
    return file_configs


def available_section_names() -> list[str]:
    """Return parsed section columns available in the first configured file."""
    file_configs = existing_section_files()
    sample = pd.read_parquet(file_configs[0]["path"])
    return [column for column in PARSED_SECTION_COLUMNS if column in sample.columns]


def load_matching_rows(
    subject_id: int,
    hadm_id: int,
    section_name: str,
    cohort: str | None = None,
) -> pd.DataFrame:
    """Load matching admission rows from all configured parsed-section files."""
    rows = []
    for file_config in existing_section_files(cohort):
        columns = METADATA_COLUMNS + [section_name]
        df = pd.read_parquet(file_config["path"], columns=columns)
        df["cohort"] = file_config["cohort"]
        match = df.loc[
            df["subject_id"].eq(subject_id) & df["hadm_id"].eq(hadm_id),
            columns,
        ].copy()
        rows.append(match)

    if not rows:
        return pd.DataFrame(columns=METADATA_COLUMNS + [section_name])
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    """Collect identifiers and print only the requested parsed section."""
    parser = parse_args()
    args = parser.parse_args()

    section_names = available_section_names()
    if args.list_sections:
        for section_name in section_names:
            print(section_name)
        return

    subject_id = args.subject_id if args.subject_id is not None else prompt_for_int("subject_id")
    hadm_id = args.hadm_id if args.hadm_id is not None else prompt_for_int("hadm_id")
    section_name = args.section_name or prompt_for_section(section_names)
    if section_name not in section_names:
        raise ValueError(
            f"Unknown section_name={section_name!r}. "
            f"Use --list-sections to see available section names."
        )

    rows = load_matching_rows(subject_id, hadm_id, section_name, args.cohort)
    if rows.empty:
        print(f"No parsed note found for subject_id={subject_id}, hadm_id={hadm_id}.")
        return

    for index, row in rows.iterrows():
        if len(rows) > 1:
            print(f"\n--- Match {index + 1} of {len(rows)} ---")
        print(f"cohort: {row['cohort']}")
        print(f"subject_id: {row['subject_id']}")
        print(f"hadm_id: {row['hadm_id']}")
        print(f"note_id: {row['note_id']}")
        print(f"charttime: {row['charttime']}")
        print(f"section_name: {section_name}")

        section_text = str(row.get(section_name, "") or "").strip()
        if not section_text:
            print("\n[Section is empty.]")
        else:
            print("\n" + section_text)


if __name__ == "__main__":
    main()
