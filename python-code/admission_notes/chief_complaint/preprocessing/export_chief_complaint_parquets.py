from __future__ import annotations

from pathlib import Path

import duckdb


SCRIPT_DIR = Path(__file__).resolve().parent
PARQUET_DIR = SCRIPT_DIR.parent.parent / "parsed_admission_notes"
OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_parquets"

EXPORTS = {
    "MHH1_psychotic_chief_complaints.parquet": PARQUET_DIR / "MHH_psychotic_admission_notes.parquet",
    "MHC0_chief_complaints.parquet": PARQUET_DIR / "MHC0_admission_notes.parquet",
}


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def chief_complaint_query(source_path: Path) -> str:
    return f"""
    SELECT
        subject_id,
        hadm_id,
        coalesce(chief_complaint, '') AS chief_complaint
    FROM read_parquet({sql_string(str(source_path))})
    """


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


if __name__ == "__main__":
    main()
