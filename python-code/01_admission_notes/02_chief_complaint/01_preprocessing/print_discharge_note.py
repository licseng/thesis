"""Print one raw discharge note from the local DuckDB database.

This is a manual inspection helper for cases where a parsed or preprocessed
chief complaint looks suspicious and the original discharge note needs to be
checked.

Inputs:
    Local DuckDB database at ../DataBase relative to the thesis code folder.
    Tables listed in `SOURCE_TABLES`.

Usage:
    python print_discharge_note.py --subject-id 123 --hadm-id 456

If either ID is omitted, the script prompts for it interactively.

Privacy note:
    This script prints raw discharge-note text to the terminal. Use it only in
    your own local terminal when you intentionally want to inspect the note.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import duckdb


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent

# Cohort export tables to search. The label is printed with the matching note so
# it is clear which cohort table supplied the row.
SOURCE_TABLES = {
    "MHH_psychotic": "export_MHH_psychotic",
    "MHC0": "export_only_MHC0",
}


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


# Safely quote SQL identifiers such as DuckDB table names.
def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


# Parse optional command-line IDs. Missing IDs are requested interactively.
def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Print a raw discharge note by subject_id and hadm_id.")
    parser.add_argument("--subject-id", type=int, help="MIMIC subject_id.")
    parser.add_argument("--hadm-id", type=int, help="MIMIC hadm_id.")
    return parser


# Prompt until the user enters an integer ID.
def prompt_for_int(label: str) -> int:
    while True:
        value = input(f"{label}: ").strip()
        try:
            return int(value)
        except ValueError:
            print(f"Please enter a numeric {label}.")


# Build a parameterized query that searches all configured cohort tables for the
# requested subject_id and hadm_id.
def raw_note_query() -> str:
    parts = []
    for group_name, table_name in SOURCE_TABLES.items():
        parts.append(
            f"""
            SELECT
                {sql_string(group_name)} AS mhc_group,
                subject_id,
                hadm_id,
                text
            FROM {quote_identifier(table_name)}
            WHERE subject_id = ?
              AND hadm_id = ?
              AND text IS NOT NULL
              AND trim(text) <> ''
            """
        )
    return "\nUNION ALL\n".join(parts)


# Locate the DuckDB database robustly from the current script location. The
# numbered pipeline is nested more deeply than the old scripts, so a fixed
# parent count is brittle.
def database_path() -> Path:
    for directory in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        candidate = directory / "DataBase"
        if candidate.exists():
            return candidate

    searched = "\n".join(str(directory / "DataBase") for directory in [SCRIPT_DIR, *SCRIPT_DIR.parents])
    raise FileNotFoundError(f"No DuckDB database found. Searched:\n{searched}")


# Script entry point: collect IDs, query the local database, and print any
# matching raw note text.
def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    subject_id = args.subject_id if args.subject_id is not None else prompt_for_int("subject_id")
    hadm_id = args.hadm_id if args.hadm_id is not None else prompt_for_int("hadm_id")

    con = duckdb.connect(str(database_path()), read_only=True)
    try:
        rows = con.execute(
            raw_note_query(),
            [subject_id, hadm_id] * len(SOURCE_TABLES),
        ).fetchall()
    finally:
        con.close()

    if not rows:
        print(f"No discharge note found for subject_id={subject_id}, hadm_id={hadm_id}.")
        return

    for index, (mhc_group, row_subject_id, row_hadm_id, note_text) in enumerate(rows, start=1):
        if len(rows) > 1:
            print(f"\n--- Match {index} of {len(rows)} ---")
        print(f"mhc_group: {mhc_group}")
        print(f"subject_id: {row_subject_id}")
        print(f"hadm_id: {row_hadm_id}")
        print("\n" + str(note_text))


if __name__ == "__main__":
    main()
