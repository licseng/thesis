from __future__ import annotations

from pathlib import Path

import duckdb

# parse_admission_notes.py
# ----------------------
# Summary:
#   Connects to the thesis DuckDB database and extracts structured admission note sections
#   from raw text using regex pattern matching. Parses key clinical sections for both
#   MHC0 and MHC1_sa groups.
#
#   Extracts sections:
#   - chief_complaint, present_illness, medical_history, medications_on_admission,
#     allergies, physical_exam, family_history, social_history
#
#   Outputs: Parquet files saved under `dataset_parsed_admission_notes/`


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DB_PATH = PROJECT_DIR.parent / "DataBase"
OUTPUT_DIR = SCRIPT_DIR / "parsed_admission_notes"

GROUP_TABLES = {
    "MHC0": "export_only_MHC0",
    "MHC1_sa": "export_only_MHC1_same_admission",
}

ADMISSION_SECTIONS = {
    "chief_complaint": "chief complaint:",
    "present_illness": "present illness:",
    "medical_history": "medical history:",
    "medication_adm": "medications on admission:",
    "allergies": "allergies:",
    "physical_exam": "physical exam:",
    "family_history": "family history:",
    "social_history": "social history:",
}


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def section_pattern(section_heading: str) -> str:
    return rf"(?i){section_heading}([\s\S]+?)\n\s*?\n[^(\\|\d|\.)]+?:"


def section_expr(section_heading: str) -> str:
    raw_section = (
        "trim("
        "replace("
        f"coalesce(regexp_extract(note_text, {sql_string(section_pattern(section_heading))}, 1), ''), "
        "'\n', "
        "' '"
        ")"
        ")"
    )
    return f"CASE WHEN starts_with({raw_section}, '[]') THEN '' ELSE {raw_section} END"


def create_admission_table(con: duckdb.DuckDBPyConnection, group_name: str, source_table: str) -> None:
    output_table = f"{group_name}_admission_notes"
    section_selects = ",\n                ".join(
        f"{section_expr(section_heading)} AS {quote_identifier(column_name)}"
        for column_name, section_heading in ADMISSION_SECTIONS.items()
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {quote_identifier(output_table)} AS
        WITH source_notes AS (
            SELECT
                {sql_string(group_name)} AS mhc_group,
                subject_id,
                hadm_id,
                regexp_replace(
                    coalesce(text, ''),
                    '(?i)___\\nFamily History:',
                    '___\\n\\nFamily History:'
                ) AS note_text
            FROM {quote_identifier(source_table)}
            WHERE subject_id IS NOT NULL
              AND text IS NOT NULL
              AND trim(text) <> ''
        ),
        extracted AS (
            SELECT
                mhc_group,
                subject_id,
                hadm_id,
                {section_selects}
            FROM source_notes
        ),
        filtered AS (
            SELECT *
            FROM extracted
            WHERE chief_complaint <> ''
               OR present_illness <> ''
               OR medical_history <> ''
        )
        SELECT
            mhc_group,
            subject_id,
            hadm_id,
            chief_complaint,
            present_illness,
            medical_history,
            medication_adm,
            allergies,
            physical_exam,
            family_history,
            social_history,
            'CHIEF COMPLAINT: ' || chief_complaint
                || '\\n\\nPRESENT ILLNESS: ' || present_illness
                || '\\n\\nMEDICAL HISTORY: ' || medical_history
                || '\\n\\nMEDICATION ON ADMISSION: ' || medication_adm
                || '\\n\\nALLERGIES: ' || allergies
                || '\\n\\nPHYSICAL EXAM: ' || physical_exam
                || '\\n\\nFAMILY HISTORY: ' || family_history
                || '\\n\\nSOCIAL HISTORY: ' || social_history
                AS text
        FROM filtered
        """
    )


def export_table(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    parquet_path = OUTPUT_DIR / f"{table_name}.parquet"
    csv_path = OUTPUT_DIR / f"{table_name}_sample.csv"

    con.execute(
        f"""
        COPY {quote_identifier(table_name)}
        TO {sql_string(str(parquet_path))}
        (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM {quote_identifier(table_name)}
            LIMIT 5000
        )
        TO {sql_string(str(csv_path))}
        (HEADER, DELIMITER ',')
        """
    )


def print_summary(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    summary = con.execute(
        f"""
        SELECT
            {sql_string(table_name)} AS table_name,
            count(*) AS n_rows,
            count(DISTINCT subject_id) AS n_subjects,
            count(DISTINCT hadm_id) AS n_admissions,
            sum(chief_complaint <> '') AS n_with_chief_complaint,
            sum(present_illness <> '') AS n_with_present_illness,
            sum(medical_history <> '') AS n_with_medical_history
        FROM {quote_identifier(table_name)}
        """
    ).fetchone()
    print(summary)


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    try:
        for group_name, source_table in GROUP_TABLES.items():
            output_table = f"{group_name}_admission_notes"
            print(f"Parsing {source_table} -> {output_table}", flush=True)
            create_admission_table(con, group_name, source_table)
            export_table(con, output_table)
            print_summary(con, output_table)
    finally:
        con.close()

    print(f"Saved parsed admission-note exports to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
