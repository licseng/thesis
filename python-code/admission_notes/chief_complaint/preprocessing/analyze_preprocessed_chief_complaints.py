from __future__ import annotations

from pathlib import Path

import duckdb


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "chief_complaint_preprocessed"
OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_preprocessed_analysis_output"

INPUTS = {
    "MHH1_psychotic": INPUT_DIR / "MHH1_psychotic_chief_complaints_preprocessed.parquet",
    "MHC0": INPUT_DIR / "MHC0_chief_complaints_preprocessed.parquet",
}


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def union_preprocessed_source() -> str:
    parts = []
    for source_table, parquet_path in INPUTS.items():
        parts.append(
            f"""
            SELECT
                {sql_string(source_table)} AS source_table,
                subject_id,
                hadm_id,
                chief_complaint_raw,
                chief_complaint_normalized,
                coalesce(medspacy_entities_all, '') AS medspacy_entities_all,
                coalesce(physical_entities_negated, '') AS physical_entities_negated,
                coalesce(psych_substance_self_harm_entities_negated, '')
                    AS psych_substance_self_harm_entities_negated,
                has_affirmed_physical_entity,
                has_affirmed_psych_substance_self_harm_entity,
                has_chief_complaint
            FROM read_parquet({sql_string(str(parquet_path))})
            """
        )
    return "\nUNION ALL\n".join(parts)


def unmapped_summary_query() -> str:
    source = union_preprocessed_source()
    return f"""
    WITH complaints AS ({source})
    SELECT
        source_table,
        count(*) AS n_rows,
        count(DISTINCT subject_id) AS n_subjects,
        count(DISTINCT hadm_id) AS n_admissions,
        cast(sum(has_chief_complaint) AS BIGINT) AS n_with_chief_complaint,
        cast(sum(has_chief_complaint AND medspacy_entities_all = '') AS BIGINT)
            AS n_unmapped_chief_complaints,
        100.0 * sum(has_chief_complaint AND medspacy_entities_all = '')
            / nullif(sum(has_chief_complaint), 0) AS pct_unmapped_among_chief_complaints,
        cast(sum(has_chief_complaint AND has_affirmed_physical_entity) AS BIGINT)
            AS n_with_affirmed_physical_entity,
        100.0 * sum(has_chief_complaint AND has_affirmed_physical_entity)
            / nullif(sum(has_chief_complaint), 0) AS pct_with_affirmed_physical_entity,
        cast(
            sum(has_chief_complaint AND has_affirmed_psych_substance_self_harm_entity)
            AS BIGINT
        ) AS n_with_affirmed_psych_substance_self_harm_entity,
        100.0 * sum(has_chief_complaint AND has_affirmed_psych_substance_self_harm_entity)
            / nullif(sum(has_chief_complaint), 0)
            AS pct_with_affirmed_psych_substance_self_harm_entity,
        1.0 * sum(has_chief_complaint AND has_affirmed_psych_substance_self_harm_entity)
            / nullif(sum(has_chief_complaint AND has_affirmed_physical_entity), 0)
            AS ratio_psych_substance_self_harm_to_physical,
        cast(sum(has_chief_complaint AND physical_entities_negated <> '') AS BIGINT)
            AS n_with_possible_negated_physical_complaint,
        cast(
            sum(has_chief_complaint AND psych_substance_self_harm_entities_negated <> '')
            AS BIGINT
        ) AS n_with_possible_negated_psych_substance_self_harm_complaint,
        cast(
            sum(
                has_chief_complaint
                AND (
                    physical_entities_negated <> ''
                    OR psych_substance_self_harm_entities_negated <> ''
                )
            )
            AS BIGINT
        ) AS n_with_any_possible_negated_complaint,
        100.0 * sum(
            has_chief_complaint
            AND (
                physical_entities_negated <> ''
                OR psych_substance_self_harm_entities_negated <> ''
            )
        ) / nullif(sum(has_chief_complaint), 0)
            AS pct_with_any_possible_negated_complaint
    FROM complaints
    GROUP BY source_table
    ORDER BY source_table DESC
    """


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


def main() -> None:
    con = duckdb.connect()
    try:
        query = unmapped_summary_query()
        result = con.execute(query).fetchdf()
        print("\n=== Unmapped chief complaints ===")
        print(result.to_string(index=False))
        write_csv(con, query, "unmapped_chief_complaint_summary.csv")
    finally:
        con.close()

    print(f"\nSaved preprocessed chief-complaint analysis CSVs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
