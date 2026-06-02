from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import duckdb

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DB_PATH = PROJECT_DIR.parent / "DataBase"
OUTPUT_DIR = SCRIPT_DIR / "mhc1_sa_icd_lookup_output"

PARQUET_PATH = SCRIPT_DIR / "dataset_parsed_admission_notes" / "MHC1_sa_admission_notes.parquet"
DIAGNOSES_TABLE = "diagnoses_icd"
ICD_LOOKUP_TABLE = "d_icd_diagnoses"


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_term_query(terms: list[str]) -> str:
    term_queries = []
    for term in terms:
        pattern = f"%{term.lower()}%"
        term_queries.append(
            f"""
            SELECT
                {sql_string(term)} AS search_term,
                e.subject_id,
                e.hadm_id,
                e.chief_complaint AS chief_complaint_text,
                d.icd_code,
                d.icd_version,
                dd.long_title AS icd_description
            FROM read_parquet({sql_string(str(PARQUET_PATH))}) AS e
            JOIN {DIAGNOSES_TABLE} AS d USING(subject_id, hadm_id)
            JOIN {ICD_LOOKUP_TABLE} AS dd
                ON dd.icd_code = d.icd_code
                AND dd.icd_version = d.icd_version
            WHERE d.seq_num = 1
              AND lower(coalesce(e.chief_complaint, '')) LIKE {sql_string(pattern)}
            """
        )
    return "\nUNION ALL\n".join(term_queries)


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


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="Find primary ICD codes for MHC1_sa admissions whose notes match given search terms."
    )
    parser.add_argument(
        "--terms",
        nargs="+",
        help="Search terms to match in MHC1_sa chief complaint text.",
    )
    parser.add_argument(
        "--terms-file",
        type=Path,
        help="Path to a newline-delimited file with search terms.",
    )
    parser.add_argument(
        "--output",
        default="mhc1_sa_primary_icd_by_term.csv",
        help="Output CSV file name under mhc1_sa_icd_lookup_output/.",
    )
    return parser


def load_terms(args: argparse.Namespace) -> list[str]:
    terms: list[str] = []
    if args.terms:
        terms.extend(args.terms)
    if args.terms_file:
        terms.extend(
            line.strip() for line in args.terms_file.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    return [term for term in terms if term.strip()]


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    terms = load_terms(args)

    if not terms:
        parser.error("At least one search term is required via --terms or --terms-file.")

    query = build_term_query(terms)

    con = duckdb.connect(str(DB_PATH))
    try:
        print("Running ICD lookup for MHC1_sa notes...")
        write_csv(con, query, args.output)
    finally:
        con.close()

    print(f"Saved ICD lookup results to: {OUTPUT_DIR / args.output}")


if __name__ == "__main__":
    main()
