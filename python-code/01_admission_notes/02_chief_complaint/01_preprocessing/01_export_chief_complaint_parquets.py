"""Export minimal chief-complaint parquet files from parsed admission notes.

This script is a bridge between the discharge-note parsing step and the chief
complaint preprocessing step. The parsed admission-note parquet files contain
multiple extracted note sections and metadata columns, but the chief-complaint
preprocessing script only needs three columns:
    - subject_id
    - hadm_id
    - chief_complaint

Inputs:
    parsed_admission_notes/MHH_psychotic_admission_notes.parquet
    parsed_admission_notes/MHC0_admission_notes.parquet

Outputs:
    chief_complaint_parquets/MHH1_psychotic_chief_complaints.parquet
    chief_complaint_parquets/MHC0_chief_complaints.parquet

This script does not clean, normalize, embed, or classify chief complaints. It
only exports the minimal text field needed by the downstream chief-complaint NLP
pipeline.
"""

from __future__ import annotations
from pathlib import Path
import duckdb


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PARQUET_DIR = SCRIPT_DIR.parent.parent / "parsed_admission_notes"
OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_parquets"

# Input-to-output mapping
EXPORTS = {
    "MHH1_psychotic_chief_complaints.parquet": PARQUET_DIR / "MHH_psychotic_admission_notes.parquet",
    "MHC0_chief_complaints.parquet": PARQUET_DIR / "MHC0_admission_notes.parquet",
}

# Safely quote a Python string as a SQL string literal for DuckDB queries.
def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


# Build the SQL query that extracts the minimal chief-complaint fields from one
# parsed admission-note parquet file.
def chief_complaint_query(source_path: Path) -> str:
    return f"""
    SELECT
        subject_id,
        hadm_id,
        coalesce(chief_complaint, '') AS chief_complaint
    FROM read_parquet({sql_string(str(source_path))})
    """


# Write one chief-complaint parquet and print basic counts for quality control.
def write_parquet(con: duckdb.DuckDBPyConnection, output_name: str, source_path: Path) -> None:
    output_path = OUTPUT_DIR / output_name
    query = chief_complaint_query(source_path)
    con.execute(
        f"""
        COPY ({query})
        TO {sql_string(str(output_path))}
        (FORMAT PARQUET)
        """
    )

    summary = con.execute(
        f"""
        SELECT
            {sql_string(output_name)} AS output_name,
            count(*) AS n_rows,
            count(DISTINCT subject_id) AS n_subjects,
            count(DISTINCT hadm_id) AS n_admissions,
            sum(chief_complaint <> '') AS n_with_chief_complaint
        FROM ({query})
        """
    ).fetchdf()
    print(summary.to_string(index=False))


# Script entry point: create the output folder, export each configured cohort,
# print summaries, and close the DuckDB connection.
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    con = duckdb.connect()
    try:
        for output_name, source_path in EXPORTS.items():
            print(f"Writing {output_name}", flush=True)
            write_parquet(con, output_name, source_path)
    finally:
        con.close()

    print(f"Saved chief-complaint parquet exports to: {OUTPUT_DIR}")


# Allow the script to be run directly with:
#     python 01_export_chief_complaint_parquets.py
if __name__ == "__main__":
    main()
