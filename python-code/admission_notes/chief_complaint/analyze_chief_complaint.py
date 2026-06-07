from __future__ import annotations

from pathlib import Path

import duckdb

# analyze_chief_complaint.py
# -------------------------
# Summary:
#   Reads the parsed admission note parquet files and computes chief complaint
#   statistics for the MHH_psychotic, MHH_history_only, and MHC0 groups.
#
#   Outputs are saved under `chief_complaint_analysis_output/` and include:
#   - coverage of chief complaint presence by admission
#   - chief complaint length statistics
#   - most common chief complaint words and bigrams
#   - group-enriched chief complaint terms, saved separately by enrichment direction

SCRIPT_DIR = Path(__file__).resolve().parent
PARQUET_DIR = SCRIPT_DIR.parent / "parsed_admission_notes"
OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_analysis_output"

ADMISSION_PARQUETS = {
    "MHH_psychotic": PARQUET_DIR / "MHH_psychotic_admission_notes.parquet",
    "MHH_history_only": PARQUET_DIR / "MHH_history_only_admission_notes.parquet",
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


def union_chief_complaint_source() -> str:
    parts = []
    for source_table, parquet_path in ADMISSION_PARQUETS.items():
        parts.append(
            f"""
            SELECT
                {sql_string(source_table)} AS source_table,
                subject_id,
                hadm_id,
                coalesce(chief_complaint, '') AS chief_complaint
            FROM read_parquet({sql_string(str(parquet_path))})
            """
        )
    return "\nUNION ALL\n".join(parts)


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


def chief_complaint_coverage_query() -> str:
    source = union_chief_complaint_source()
    return f"""
    WITH notes AS ({source})
    SELECT
        source_table,
        count(DISTINCT hadm_id) AS n_admissions,
        count(DISTINCT subject_id) AS n_subjects,
        cast(sum(chief_complaint <> '') AS BIGINT) AS n_with_chief_complaint,
        100.0 * sum(chief_complaint <> '') / count(DISTINCT hadm_id) AS pct_with_chief_complaint
    FROM notes
    GROUP BY source_table
    ORDER BY source_table DESC
    """


def chief_complaint_length_query() -> str:
    source = union_chief_complaint_source()
    return f"""
    WITH notes AS ({source}),
    lengths AS (
        SELECT
            source_table,
            hadm_id,
            length(chief_complaint) AS chief_complaint_chars,
            length(regexp_extract_all(chief_complaint, '\\S+')) AS chief_complaint_words
        FROM notes
        WHERE chief_complaint <> ''
    )
    SELECT
        source_table,
        count(DISTINCT hadm_id) AS n_admissions,
        avg(chief_complaint_chars) AS mean_chief_complaint_chars,
        stddev_samp(chief_complaint_chars) AS sd_chief_complaint_chars,
        quantile_cont(chief_complaint_chars, 0.25) AS q1_chief_complaint_chars,
        median(chief_complaint_chars) AS median_chief_complaint_chars,
        quantile_cont(chief_complaint_chars, 0.75) AS q3_chief_complaint_chars,
        quantile_cont(chief_complaint_chars, 0.75)
            - quantile_cont(chief_complaint_chars, 0.25) AS iqr_chief_complaint_chars,
        avg(chief_complaint_words) AS mean_chief_complaint_words,
        stddev_samp(chief_complaint_words) AS sd_chief_complaint_words,
        quantile_cont(chief_complaint_words, 0.25) AS q1_chief_complaint_words,
        median(chief_complaint_words) AS median_chief_complaint_words,
        quantile_cont(chief_complaint_words, 0.75) AS q3_chief_complaint_words,
        quantile_cont(chief_complaint_words, 0.75)
            - quantile_cont(chief_complaint_words, 0.25) AS iqr_chief_complaint_words
    FROM lengths
    GROUP BY source_table
    ORDER BY source_table
    """


def chief_complaint_word_frequency_query(limit_per_group: int = 10) -> str:
    source = union_chief_complaint_source()
    stopwords = ", ".join(sql_string(word) for word in sorted(CHIEF_COMPLAINT_STOPWORDS))
    return f"""
    WITH notes AS ({source}),
    tokens AS (
        SELECT
            source_table,
            hadm_id,
            unnest(regexp_extract_all(lower(chief_complaint), '[a-z][a-z]+')) AS word
        FROM notes
        WHERE chief_complaint <> ''
    ),
    filtered_tokens AS (
        SELECT *
        FROM tokens
        WHERE word NOT IN ({stopwords})
          AND word <> '___'
    ),
    word_counts AS (
        SELECT
            source_table,
            word,
            count(*) AS n_occurrences,
            count(DISTINCT hadm_id) AS n_admissions_with_word
        FROM filtered_tokens
        GROUP BY source_table, word
    ),
    denominators AS (
        SELECT
            source_table,
            count(DISTINCT hadm_id) AS n_admissions_with_chief_complaint
        FROM notes
        WHERE chief_complaint <> ''
        GROUP BY source_table
    ),
    ranked AS (
        SELECT
            word_counts.source_table,
            row_number() OVER (
                PARTITION BY word_counts.source_table
                ORDER BY n_occurrences DESC, n_admissions_with_word DESC, word
            ) AS word_rank,
            word_counts.word,
            word_counts.n_occurrences,
            word_counts.n_admissions_with_word,
            denominators.n_admissions_with_chief_complaint,
            100.0 * word_counts.n_admissions_with_word
                / denominators.n_admissions_with_chief_complaint AS pct_admissions_with_word
        FROM word_counts
        JOIN denominators USING (source_table)
    )
    SELECT *
    FROM ranked
    WHERE word_rank <= {limit_per_group}
    ORDER BY source_table DESC, word_rank
    """


def chief_complaint_bigram_frequency_query(limit_per_group: int = 10) -> str:
    source = union_chief_complaint_source()
    stopwords = ", ".join(sql_string(word) for word in sorted(CHIEF_COMPLAINT_STOPWORDS))
    return f"""
    WITH notes AS ({source}),
    token_arrays AS (
        SELECT
            source_table,
            hadm_id,
            regexp_extract_all(lower(chief_complaint), '[a-z][a-z]+') AS tokens
        FROM notes
        WHERE chief_complaint <> ''
    ),
    bigrams AS (
        SELECT
            source_table,
            hadm_id,
            tokens[i] || ' ' || tokens[i + 1] AS bigram
        FROM token_arrays,
        generate_series(1, array_length(tokens) - 1) AS i(i)
    ),
    filtered_bigrams AS (
        SELECT *
        FROM bigrams
        WHERE split_part(bigram, ' ', 1) NOT IN ({stopwords})
          AND split_part(bigram, ' ', 2) NOT IN ({stopwords})
    ),
    bigram_counts AS (
        SELECT
            source_table,
            bigram,
            count(*) AS n_occurrences,
            count(DISTINCT hadm_id) AS n_admissions_with_bigram
        FROM filtered_bigrams
        GROUP BY source_table, bigram
    ),
    denominators AS (
        SELECT
            source_table,
            count(DISTINCT hadm_id) AS n_admissions_with_chief_complaint
        FROM notes
        GROUP BY source_table
    ),
    ranked AS (
        SELECT
            bigram_counts.source_table,
            row_number() OVER (
                PARTITION BY bigram_counts.source_table
                ORDER BY n_occurrences DESC, n_admissions_with_bigram DESC, bigram
            ) AS bigram_rank,
            bigram_counts.bigram,
            bigram_counts.n_occurrences,
            bigram_counts.n_admissions_with_bigram,
            denominators.n_admissions_with_chief_complaint,
            100.0 * bigram_counts.n_admissions_with_bigram
                / denominators.n_admissions_with_chief_complaint AS pct_admissions_with_bigram
        FROM bigram_counts
        JOIN denominators USING (source_table)
    )
    SELECT *
    FROM ranked
    WHERE bigram_rank <= {limit_per_group}
    ORDER BY source_table DESC, bigram_rank
    """


def chief_complaint_pairwise_enriched_terms_query(
    target_group: str,
    enriched_group: str,
    limit_per_group: int = 20,
    min_admissions_in_either_group: int = 10,
) -> str:
    if target_group not in {"MHH_psychotic", "MHH_history_only"}:
        raise ValueError("target_group must be either 'MHH_psychotic' or 'MHH_history_only'")
    if enriched_group not in {target_group, "MHC0"}:
        raise ValueError("enriched_group must be either target_group or 'MHC0'")

    target_slug = target_group.lower()

    source = union_chief_complaint_source()
    stopwords = ", ".join(sql_string(word) for word in sorted(CHIEF_COMPLAINT_STOPWORDS))
    return f"""
    WITH notes AS ({source}),
    tokens AS (
        SELECT
            source_table,
            hadm_id,
            unnest(regexp_extract_all(lower(chief_complaint), '[a-z][a-z]+')) AS word
        FROM notes
        WHERE chief_complaint <> ''
    ),
    filtered_tokens AS (
        SELECT *
        FROM tokens
        WHERE word NOT IN ({stopwords})
          AND word <> '___'
    ),
    word_counts AS (
        SELECT
            source_table,
            word,
            count(*) AS n_occurrences,
            count(DISTINCT hadm_id) AS n_admissions_with_word
        FROM filtered_tokens
        GROUP BY source_table, word
    ),
    denominators AS (
        SELECT
            source_table,
            count(DISTINCT hadm_id) AS n_admissions_with_chief_complaint
        FROM notes
        WHERE chief_complaint <> ''
        GROUP BY source_table
    ),
    metrics AS (
        SELECT
            word_counts.source_table,
            word_counts.word,
            word_counts.n_occurrences,
            word_counts.n_admissions_with_word,
            denominators.n_admissions_with_chief_complaint,
            100.0 * word_counts.n_admissions_with_word
                / denominators.n_admissions_with_chief_complaint AS pct_admissions_with_word
        FROM word_counts
        JOIN denominators USING (source_table)
    ),
    paired AS (
        SELECT
            target.word,
            target.n_admissions_with_word AS n_target,
            target.pct_admissions_with_word AS pct_target,
            mhc0.n_admissions_with_word AS n_mhc0,
            mhc0.pct_admissions_with_word AS pct_mhc0,
            target.pct_admissions_with_word / mhc0.pct_admissions_with_word
                AS ratio_target_over_mhc0,
            mhc0.pct_admissions_with_word / target.pct_admissions_with_word
                AS ratio_mhc0_over_target
        FROM metrics AS target
        JOIN metrics AS mhc0 ON target.word = mhc0.word
        WHERE target.source_table = {sql_string(target_group)}
          AND mhc0.source_table = 'MHC0'
          AND greatest(target.n_admissions_with_word, mhc0.n_admissions_with_word)
              >= {min_admissions_in_either_group}
    ),
    ranked AS (
        SELECT
            word,
            n_target AS n_{target_slug},
            pct_target AS pct_{target_slug},
            n_mhc0,
            pct_mhc0,
            ratio_target_over_mhc0 AS ratio_{target_slug}_over_mhc0,
            ratio_mhc0_over_target AS ratio_mhc0_over_{target_slug},
            CASE
                WHEN ratio_target_over_mhc0 >= ratio_mhc0_over_target
                    THEN {sql_string(target_group)}
                ELSE 'MHC0'
            END AS enriched_in,
            CASE
                WHEN ratio_target_over_mhc0 >= ratio_mhc0_over_target
                    THEN ratio_target_over_mhc0
                ELSE ratio_mhc0_over_target
            END AS enrichment_ratio
        FROM paired
    ),
    direction_ranked AS (
        SELECT
            row_number() OVER (
                PARTITION BY enriched_in
                ORDER BY enrichment_ratio DESC, word
            ) AS enrichment_rank,
            *
        FROM ranked
    )
    SELECT *
    FROM direction_ranked
    WHERE enriched_in = {sql_string(enriched_group)}
      AND enrichment_rank <= {limit_per_group}
    ORDER BY enrichment_rank
    """


def main() -> None:
    queries = [
        (
            "Chief complaint coverage",
            chief_complaint_coverage_query(),
            "chief_complaint_coverage.csv",
        ),
        (
            "Chief complaint length",
            chief_complaint_length_query(),
            "chief_complaint_length_summary.csv",
        ),
        (
            "Chief complaint top words",
            chief_complaint_word_frequency_query(),
            "chief_complaint_top_words.csv",
        ),
        (
            "Chief complaint top bigrams",
            chief_complaint_bigram_frequency_query(),
            "chief_complaint_top_bigrams.csv",
        ),
        (
            "Chief complaint terms enriched in MHH_psychotic",
            chief_complaint_pairwise_enriched_terms_query("MHH_psychotic", "MHH_psychotic"),
            "chief_complaint_terms_enriched_in_MHH_psychotic_vs_MHC0.csv",
        ),
        (
            "Chief complaint terms enriched in MHC0 vs MHH_psychotic",
            chief_complaint_pairwise_enriched_terms_query("MHH_psychotic", "MHC0"),
            "chief_complaint_terms_enriched_in_MHC0_vs_MHH_psychotic.csv",
        ),
        (
            "Chief complaint terms enriched in MHH_history_only",
            chief_complaint_pairwise_enriched_terms_query("MHH_history_only", "MHH_history_only"),
            "chief_complaint_terms_enriched_in_MHH_history_only_vs_MHC0.csv",
        ),
        (
            "Chief complaint terms enriched in MHC0 vs MHH_history_only",
            chief_complaint_pairwise_enriched_terms_query("MHH_history_only", "MHC0"),
            "chief_complaint_terms_enriched_in_MHC0_vs_MHH_history_only.csv",
        ),
    ]

    con = duckdb.connect()
    try:
        for title, query, output_name in queries:
            print(f"\n=== {title} ===")
            result = con.execute(query).df()
            print(result.to_string(index=False))
            write_csv(con, query, output_name)
    finally:
        con.close()

    print(f"\nSaved chief complaint analysis CSVs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
