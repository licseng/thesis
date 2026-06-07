from __future__ import annotations

from pathlib import Path

import duckdb


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
THESIS_DIR = PROJECT_DIR.parent.parent
DB_PATH = THESIS_DIR / "DataBase"
OUTPUT_DIR = SCRIPT_DIR / "parsed_admission_notes"

EXPORTS = [
    {
        "source_table": "export_MHH_psychotic",
        "output_name": "MHH_psychotic_admission_notes",
    },
    {
        "source_table": "export_MHH_history_only",
        "output_name": "MHH_history_only_admission_notes",
    },
    {
        "source_table": "export_only_MHC0",
        "output_name": "MHC0_admission_notes",
    },
]

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


def table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    rows = con.execute(f"DESCRIBE {quote_identifier(table_name)}").fetchall()
    return [row[0] for row in rows]


def metadata_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    excluded_columns = {"subject_id", "hadm_id", "text"}
    return [column for column in table_columns(con, table_name) if column not in excluded_columns]


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


def parsed_notes_query(
    source_table: str,
    metadata_column_names: list[str],
) -> str:
    metadata_selects = ",\n            ".join(
        quote_identifier(column_name) for column_name in metadata_column_names
    )
    metadata_columns_sql = ",\n            ".join(
        quote_identifier(column_name) for column_name in metadata_column_names
    )
    metadata_final_sql = ",\n        ".join(
        quote_identifier(column_name) for column_name in metadata_column_names
    )
    section_selects = ",\n            ".join(
        f"{section_expr(section_heading)} AS {quote_identifier(column_name)}"
        for column_name, section_heading in ADMISSION_SECTIONS.items()
    )

    return f"""
    WITH source_notes AS (
        SELECT
            subject_id,
            hadm_id,
            {metadata_selects}
            {"," if metadata_selects else ""}
            regexp_replace(
                coalesce(text, ''),
                '(?i)___\\nFamily History:',
                '___\\n\\nFamily History:'
            ) AS note_text
        FROM {quote_identifier(source_table)}
        WHERE subject_id IS NOT NULL
          AND hadm_id IS NOT NULL
          AND text IS NOT NULL
          AND trim(text) <> ''
    ),
    extracted AS (
        SELECT
            subject_id,
            hadm_id,
            {metadata_columns_sql}
            {"," if metadata_columns_sql else ""}
            {section_selects}
        FROM source_notes
    )
    SELECT
        subject_id,
        hadm_id,
        {metadata_final_sql}
        {"," if metadata_final_sql else ""}
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
    FROM extracted
    """


def ensure_database_exists() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"No DuckDB database found at: {DB_PATH}")


def write_exports(con: duckdb.DuckDBPyConnection, query: str, output_name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    parquet_path = OUTPUT_DIR / f"{output_name}.parquet"
    sample_csv_path = OUTPUT_DIR / f"{output_name}_sample.csv"

    con.execute(
        f"""
        COPY ({query})
        TO {sql_string(str(parquet_path))}
        (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM ({query})
            LIMIT 5000
        )
        TO {sql_string(str(sample_csv_path))}
        (HEADER, DELIMITER ',')
        """
    )


def print_summary(con: duckdb.DuckDBPyConnection, query: str, output_name: str) -> None:
    summary = con.execute(
        f"""
        WITH parsed AS ({query})
        SELECT
            {sql_string(output_name)} AS output_name,
            count(*) AS n_rows,
            count(DISTINCT subject_id) AS n_subjects,
            count(DISTINCT hadm_id) AS n_admissions,
            sum(chief_complaint <> '') AS n_with_chief_complaint,
            sum(present_illness <> '') AS n_with_present_illness,
            sum(medical_history <> '') AS n_with_medical_history,
            sum(
                chief_complaint <> ''
                OR present_illness <> ''
                OR medical_history <> ''
            ) AS n_with_any_core_section
        FROM parsed
        """
    ).fetchdf()
    print(summary.to_string(index=False))


def main() -> None:
    ensure_database_exists()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        for export in EXPORTS:
            source_table = export["source_table"]
            output_name = export["output_name"]
            metadata_column_names = metadata_columns(con, source_table)
            query = parsed_notes_query(source_table, metadata_column_names)

            print(f"Parsing {source_table} from: {DB_PATH}", flush=True)
            write_exports(con, query, output_name)
            print_summary(con, query, output_name)
    finally:
        con.close()

    print(f"Saved parsed admission-note exports to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
