"""Parse chief complaints from MIMIC-IV discharge notes.

This script is the first text-processing step for the chief-complaint pipeline.
It reads cohort-specific discharge-note export tables from the local DuckDB
database and extracts only the chief complaint field.

The broader full-discharge-note parser is separate:
`02_parse_full_discharge_notes.py`.

Inputs:
    DuckDB tables listed in `EXPORTS`, currently:
        - export_MHH_psychotic
        - export_MHH_history_only, if present
        - export_only_MHC0

Outputs:
    parsed_chief_complaints/<output_name>.parquet
    parsed_chief_complaints/<output_name>_sample.csv

Important limitation:
    Extraction is rule-based. This intentionally preserves the same chief
    complaint extraction logic used by the earlier admission-section parser,
    but drops all non-chief-complaint output columns.
"""

from __future__ import annotations

from pathlib import Path
import re

import duckdb


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
THESIS_DIR = PROJECT_DIR.parent.parent.parent
DB_PATH = THESIS_DIR / "DataBase"
OUTPUT_DIR = SCRIPT_DIR / "parsed_chief_complaints"
SAMPLE_SIZE = 5000

# Cohort tables to parse
EXPORTS = [
    {
        "source_table": "export_MHH_psychotic",
        "output_name": "MHH_psychotic_chief_complaints_from_discharge_notes",
    },
    {
        "source_table": "export_MHH_history_only",
        "output_name": "MHH_history_only_chief_complaints_from_discharge_notes",
        "optional": True,
    },
    {
        "source_table": "export_only_MHC0",
        "output_name": "MHC0_chief_complaints_from_discharge_notes",
    },
]

# Same minimal section registry and chief-complaint boundary regex used by the
# earlier admission-section parser. Only chief_complaint is written downstream.
ADMISSION_SECTIONS = {
    "chief_complaint": "chief complaint:",
}

CHIEF_COMPLAINT_BOUNDARY_PATTERN = (
    r"(?:hpi|[A-Z][A-Za-z]{4,}(?:\s+(?:[A-Za-z]{2,}|_+)){0,8})\s*:"
)


def sql_string(value: str) -> str:
    """Safely quote a Python string as a DuckDB SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def quote_identifier(identifier: str) -> str:
    """Safely quote a SQL identifier such as a table or column name."""
    return '"' + identifier.replace('"', '""') + '"'


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Return True if a configured source table exists in DuckDB."""
    count = con.execute(
        """
        SELECT count(*)
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchone()[0]
    return count > 0


def table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    """Return the column names of a DuckDB table."""
    rows = con.execute(f"DESCRIBE {quote_identifier(table_name)}").fetchall()
    return [row[0] for row in rows]


def metadata_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    """Identify source-table columns that should be copied through."""
    excluded_columns = {"subject_id", "hadm_id", "text"}
    return [column for column in table_columns(con, table_name) if column not in excluded_columns]


def section_pattern(section_heading: str) -> str:
    """Build the regex used to extract chief complaint text."""
    start_pattern = re.escape(section_heading)
    return (
        rf"(?i)(?:{start_pattern})"
        rf"([\s\S]+?)(?:\n\s*(?:{CHIEF_COMPLAINT_BOUNDARY_PATTERN})|$)"
    )


def chief_complaint_expr() -> str:
    """Build the DuckDB SQL expression that extracts and cleans chief complaint."""
    raw_section = (
        "trim("
        "replace("
        f"coalesce(regexp_extract(note_text, {sql_string(section_pattern(ADMISSION_SECTIONS['chief_complaint']))}, 1), ''), "
        "'\n', "
        "' '"
        ")"
        ")"
    )
    return f"CASE WHEN starts_with({raw_section}, '[]') THEN '' ELSE {raw_section} END"


def parsed_chief_complaint_query(
    source_table: str,
    metadata_column_names: list[str],
) -> str:
    """Construct the DuckDB query that extracts chief complaints for one cohort."""
    metadata_selects = ",\n            ".join(
        quote_identifier(column_name) for column_name in metadata_column_names
    )
    metadata_final_sql = ",\n        ".join(
        quote_identifier(column_name) for column_name in metadata_column_names
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
    )
    SELECT
        subject_id,
        hadm_id,
        {metadata_final_sql}
        {"," if metadata_final_sql else ""}
        {chief_complaint_expr()} AS chief_complaint
    FROM source_notes
    """


def ensure_database_exists() -> None:
    """Fail early with a clear message if the expected DuckDB database is absent."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"No DuckDB database found at: {DB_PATH}")


def write_exports(con: duckdb.DuckDBPyConnection, query: str, output_name: str) -> None:
    """Write the parsed chief-complaint table and a local inspection sample."""
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
            LIMIT {SAMPLE_SIZE}
        )
        TO {sql_string(str(sample_csv_path))}
        (HEADER, DELIMITER ',')
        """
    )


def print_summary(con: duckdb.DuckDBPyConnection, query: str, output_name: str) -> None:
    """Print chief-complaint extraction counts without printing note text."""
    summary = con.execute(
        f"""
        WITH parsed AS ({query})
        SELECT
            {sql_string(output_name)} AS output_name,
            count(*) AS n_rows,
            count(DISTINCT subject_id) AS n_subjects,
            count(DISTINCT hadm_id) AS n_admissions,
            sum(chief_complaint <> '') AS n_with_chief_complaint
        FROM parsed
        """
    ).fetchdf()
    print(summary.to_string(index=False))


def main() -> None:
    """Parse configured source tables into chief-complaint-only outputs."""
    ensure_database_exists()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        for export in EXPORTS:
            source_table = export["source_table"]
            if export.get("optional") and not table_exists(con, source_table):
                print(f"Skipping optional missing table: {source_table}", flush=True)
                continue

            output_name = export["output_name"]
            metadata_column_names = metadata_columns(con, source_table)
            query = parsed_chief_complaint_query(source_table, metadata_column_names)

            print(f"Parsing chief complaints from {source_table}", flush=True)
            write_exports(con, query, output_name)
            print_summary(con, query, output_name)
    finally:
        con.close()

    print(f"Saved parsed chief-complaint outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
