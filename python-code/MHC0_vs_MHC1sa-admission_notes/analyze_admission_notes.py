from __future__ import annotations

from pathlib import Path

import duckdb

# analyze_admission_notes.py
# -------------------------
# Summary:
#   Connects to the thesis DuckDB database and computes admission note analysis statistics.
#   Generates summary statistics for both MHC1_sa and MHC0 groups.
#
#   Analyses include:
#   - Overall note length statistics (characters and words) for full and admission-only text.
#   - Admission section coverage (presence of chief_complaint, present_illness, medical_history).
#
#   Outputs are saved under `admission_note_analysis_output/`.

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DB_PATH = PROJECT_DIR.parent / "DataBase"
PARQUET_DIR = SCRIPT_DIR / "dataset_parsed_admission_notes"
OUTPUT_DIR = SCRIPT_DIR / "admission_note_analysis_output"

SOURCE_TABLES = {
    "MHC1_sa": "export_only_MHC1_same_admission",
    "MHC0": "export_only_MHC0",
}

ADMISSION_PARQUETS = {
    "MHC1_sa": PARQUET_DIR / "MHC1_sa_admission_notes.parquet",
    "MHC0": PARQUET_DIR / "MHC0_admission_notes.parquet",
}

CHIEF_COMPLAINT_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def write_csv(con: duckdb.DuckDBPyConnection, query: str, output_name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / output_name
    con.execute(
        f"""
        COPY ({query})
        TO {sql_string(str(output_path))}
        (HEADER, DELIMITER ',')
        """
    )


def print_query(con: duckdb.DuckDBPyConnection, title: str, query: str) -> None:
    print(f"\n=== {title} ===")
    print(con.execute(query).df().to_string(index=False))


def union_text_source(tables: dict[str, str], text_col: str = "text") -> str:
    parts = []
    for source_table, table_name in tables.items():
        parts.append(
            f"""
            SELECT
                {sql_string(source_table)} AS source_table,
                subject_id,
                hadm_id,
                trim(coalesce({quote_identifier(text_col)}, '')) AS text
            FROM {quote_identifier(table_name)}
            WHERE subject_id IS NOT NULL
              AND {quote_identifier(text_col)} IS NOT NULL
              AND trim({quote_identifier(text_col)}) <> ''
            """
        )
    return "\nUNION ALL\n".join(parts)


def length_summary_query(tables: dict[str, str], text_col: str = "text") -> str:
    source = union_text_source(tables, text_col=text_col)
    return f"""
    WITH notes AS ({source}),
    lengths AS (
        SELECT
            source_table,
            subject_id,
            hadm_id,
            length(text) AS n_chars,
            length(regexp_extract_all(text, '\\S+')) AS n_words
        FROM notes
    )
    SELECT
        source_table,
        count(DISTINCT hadm_id) AS n_admissions,
        count(DISTINCT subject_id) AS n_subjects,
        avg(n_chars) AS mean_chars,
        stddev_samp(n_chars) AS sd_chars,
        quantile_cont(n_chars, 0.25) AS q1_chars,
        median(n_chars) AS median_chars,
        quantile_cont(n_chars, 0.75) AS q3_chars,
        quantile_cont(n_chars, 0.75) - quantile_cont(n_chars, 0.25) AS iqr_chars,
        avg(n_words) AS mean_words,
        stddev_samp(n_words) AS sd_words,
        quantile_cont(n_words, 0.25) AS q1_words,
        median(n_words) AS median_words,
        quantile_cont(n_words, 0.75) AS q3_words,
        quantile_cont(n_words, 0.75) - quantile_cont(n_words, 0.25) AS iqr_words
    FROM lengths
    GROUP BY source_table
    ORDER BY source_table DESC
    """


def union_admission_source() -> str:
    parts = []
    for source_table, parquet_path in ADMISSION_PARQUETS.items():
        parts.append(
            f"""
            SELECT
                {sql_string(source_table)} AS source_table,
                subject_id,
                hadm_id,
                coalesce(chief_complaint, '') AS chief_complaint,
                coalesce(present_illness, '') AS present_illness,
                coalesce(medical_history, '') AS medical_history
            FROM read_parquet({sql_string(str(parquet_path))})
            """
        )
    return "\nUNION ALL\n".join(parts)


def admission_text_length_query() -> str:
    """Compute length statistics for parsed admission notes from parquet files."""
    parts = []
    for source_table, parquet_path in ADMISSION_PARQUETS.items():
        parts.append(
            f"""
            SELECT
                {sql_string(source_table)} AS source_table,
                subject_id,
                hadm_id,
                text,
                length(text) AS n_chars,
                length(regexp_extract_all(text, '\\S+')) AS n_words
            FROM read_parquet({sql_string(str(parquet_path))})
            WHERE text IS NOT NULL AND trim(text) <> ''
            """
        )
    union_sql = "\nUNION ALL\n".join(parts)
    return f"""
    WITH lengths AS ({union_sql})
    SELECT
        source_table,
        count(DISTINCT hadm_id) AS n_admissions,
        count(DISTINCT subject_id) AS n_subjects,
        avg(n_chars) AS mean_chars,
        stddev_samp(n_chars) AS sd_chars,
        quantile_cont(n_chars, 0.25) AS q1_chars,
        median(n_chars) AS median_chars,
        quantile_cont(n_chars, 0.75) AS q3_chars,
        quantile_cont(n_chars, 0.75) - quantile_cont(n_chars, 0.25) AS iqr_chars,
        avg(n_words) AS mean_words,
        stddev_samp(n_words) AS sd_words,
        quantile_cont(n_words, 0.25) AS q1_words,
        median(n_words) AS median_words,
        quantile_cont(n_words, 0.75) AS q3_words,
        quantile_cont(n_words, 0.75) - quantile_cont(n_words, 0.25) AS iqr_words
    FROM lengths
    GROUP BY source_table
    ORDER BY source_table DESC
    """


def section_coverage_query() -> str:
    source = union_admission_source()
    return f"""
    WITH notes AS ({source}),
    flags AS (
        SELECT
            *,
            chief_complaint <> '' AS has_chief_complaint,
            present_illness <> '' AS has_present_illness,
            medical_history <> '' AS has_medical_history
        FROM notes
    )
    SELECT
        source_table,
        count(DISTINCT hadm_id) AS n_admissions,
        count(DISTINCT subject_id) AS n_subjects,
        cast(sum(has_chief_complaint) AS BIGINT) AS n_with_chief_complaint,
        cast(sum(has_present_illness) AS BIGINT) AS n_with_present_illness,
        cast(sum(has_medical_history) AS BIGINT) AS n_with_medical_history,
        cast(sum(has_chief_complaint AND has_present_illness AND has_medical_history) AS BIGINT)
            AS n_with_all_three,
        100.0 * sum(has_chief_complaint) / count(DISTINCT hadm_id) AS pct_with_chief_complaint,
        100.0 * sum(has_present_illness) / count(DISTINCT hadm_id) AS pct_with_present_illness,
        100.0 * sum(has_medical_history) / count(DISTINCT hadm_id) AS pct_with_medical_history,
        100.0 * sum(has_chief_complaint AND has_present_illness AND has_medical_history)
            / count(DISTINCT hadm_id) AS pct_with_all_three
    FROM flags
    GROUP BY source_table
    ORDER BY source_table
    """




def main() -> None:
    queries = [
        (
            "Full note length sanity check",
            length_summary_query(SOURCE_TABLES),
            "full_note_length_summary.csv",
        ),
        (
            "Admission-only text length sanity check",
            admission_text_length_query(),
            "admission_note_length_summary.csv",
        ),
        (
            "Admission-section coverage",
            section_coverage_query(),
            "section_coverage_summary.csv",
        ),
    ]

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        for title, query, output_name in queries:
            print_query(con, title, query)
            write_csv(con, query, output_name)
    finally:
        con.close()

    print(f"\nSaved analysis CSVs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
