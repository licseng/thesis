"""Create model-ready prediction datasets from patient-level partitions.

This script builds the text dataset for the clinical prediction model. It reads
the patient-level train/validation/test flags from DuckDB, joins raw discharge
note text, parses sections with the same parser used earlier in the project,
and saves only the sections used in the reference paper:

    - Chief Complaint
    - History of Present Illness / Present Illness
    - Medical History
    - Admission Medications
    - Allergies
    - Physical Exam
    - Family History
    - Social History

The primary label is prolonged length of stay > 7 days. A secondary 30-day
readmission label is also saved for planning.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_CODE_DIR = SCRIPT_DIR.parents[2]
THESIS_DIR = THESIS_CODE_DIR.parent
DB_PATH = Path(
    os.environ.get(
        "PREDICTION_DATASET_DB_PATH",
        str(THESIS_DIR / "DataBase"),
    )
)
PARSER_PATH = (
    THESIS_CODE_DIR
    / "python-code"
    / "01_discharge_note_preprocessing"
    / "01_discharge_note_parsing"
    / "02_parse_full_discharge_notes.py"
)
OUTPUT_DIR = SCRIPT_DIR / "prediction_model_dataset"
SUMMARY_OUTPUT_DIR = SCRIPT_DIR / "analysis_output_prediction_model_dataset"

CHUNK_SIZE = int(os.environ.get("PREDICTION_DATASET_CHUNK_SIZE", "5000"))
MAX_ROWS = os.environ.get("PREDICTION_DATASET_MAX_ROWS")
MAX_ROWS = int(MAX_ROWS) if MAX_ROWS else None
ONE_DAY_SECONDS = 24 * 60 * 60

PAPER_SECTION_TITLES = {
    "chief_complaint": "Chief Complaint",
    "present_illness": "History of Present Illness",
    "medical_history": "Medical History",
    "medication_admission": "Admission Medications",
    "allergies": "Allergies",
    "physical_exam": "Physical Exam",
    "family_history": "Family History",
    "social_history": "Social History",
}
PAPER_SECTIONS = list(PAPER_SECTION_TITLES)

DATASET_COLUMNS = [
    "dataset_split",
    "partition",
    "selected_for_general_test",
    "subject_id",
    "hadm_id",
    "note_id",
    "admittime",
    "dischtime",
    "deathtime",
    "hospital_expire_flag",
    "admission_type",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "race",
    "marital_status",
    "gender",
    "anchor_age",
    "dod",
    "is_mhh1_psychotic_admission",
    "is_mhc0_admission",
    "is_matched_mhh1_psychotic_admission",
    "is_matched_mhc0_admission",
    "is_mhh1_psychotic_subject",
    "is_mhc0_subject",
    "is_matched_mhh1_psychotic_subject",
    "is_matched_mhc0_subject",
    "hospital_los_days",
    "prolonged_los_gt_7d",
    "died_in_hospital",
    "eligible_for_30d_readmission",
    "next_hadm_id_after_discharge",
    "next_admittime_after_discharge",
    "days_to_next_admission_after_discharge",
    "readmission_within_30d",
    "model_text",
    "n_model_sections_present",
    "model_text_n_chars",
    "model_text_n_words",
    *PAPER_SECTIONS,
]

SPLIT_OUTPUTS = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "test_general": "test_general.parquet",
    "test_fairness_mhh1_mhc0": "test_fairness_mhh1_mhc0.parquet",
}


def quote_identifier(identifier: str) -> str:
    """Quote a DuckDB identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def load_parser_module() -> Any:
    """Load the existing full-note parser module from its file path."""
    if not PARSER_PATH.exists():
        raise FileNotFoundError(f"Missing parser script: {PARSER_PATH}")
    spec = importlib.util.spec_from_file_location("full_discharge_parser", PARSER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import parser from: {PARSER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_required_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Fail early if the partition SQL has not been rerun."""
    existing = set(con.execute("SHOW TABLES").fetchdf().iloc[:, 0].astype(str))
    required = {
        "eligible_prediction_admissions_with_partition",
        "admissions",
        "discharge",
    }
    missing = sorted(required - existing)
    if missing:
        raise FileNotFoundError(f"Missing required DuckDB tables: {missing}")

    columns = {
        row[0]
        for row in con.execute(
            "DESCRIBE eligible_prediction_admissions_with_partition"
        ).fetchall()
    }
    if "selected_for_general_test" not in columns:
        raise ValueError(
            "eligible_prediction_admissions_with_partition does not contain "
            "selected_for_general_test. Rerun the updated partition SQL first."
        )


def selected_rows_sql() -> str:
    """Return SQL for admissions that should be parsed for modeling/evaluation."""
    limit = f"LIMIT {MAX_ROWS}" if MAX_ROWS else ""
    return f"""
        SELECT
            e.*,
            dis.text AS full_note_text
        FROM eligible_prediction_admissions_with_partition e
        JOIN discharge dis
            ON e.subject_id = dis.subject_id
           AND e.hadm_id = dis.hadm_id
           AND e.note_id = dis.note_id
        WHERE e.partition IN ('train', 'validation')
           OR (
               e.partition = 'test_pool'
               AND (
                   e.selected_for_general_test = 1
                   OR e.is_matched_mhh1_psychotic_admission = 1
                   OR e.is_matched_mhc0_admission = 1
               )
           )
        ORDER BY
            CASE e.partition
                WHEN 'train' THEN 1
                WHEN 'validation' THEN 2
                WHEN 'test_pool' THEN 3
                ELSE 5
            END,
            e.subject_id,
            e.hadm_id,
            e.note_id
        {limit}
    """


def add_next_readmission_columns(
    con: duckdb.DuckDBPyConnection,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Find the next MIMIC admission after discharge for each selected admission."""
    if rows.empty:
        return rows
    selected = rows.loc[:, ["subject_id", "hadm_id", "dischtime"]].copy()
    selected["subject_id"] = pd.to_numeric(selected["subject_id"], errors="raise").astype(int)
    selected["hadm_id"] = pd.to_numeric(selected["hadm_id"], errors="raise").astype(int)
    selected["dischtime"] = pd.to_datetime(selected["dischtime"], errors="coerce")
    con.register("selected_chunk_for_readmission", selected)
    next_admissions = con.execute(
        """
        WITH candidate_next AS (
            SELECT
                current.subject_id,
                current.hadm_id,
                next_adm.hadm_id AS next_hadm_id_after_discharge,
                next_adm.admittime AS next_admittime_after_discharge,
                ROW_NUMBER() OVER (
                    PARTITION BY current.subject_id, current.hadm_id
                    ORDER BY next_adm.admittime, next_adm.hadm_id
                ) AS next_rank
            FROM selected_chunk_for_readmission current
            LEFT JOIN admissions next_adm
                ON current.subject_id = next_adm.subject_id
               AND current.hadm_id <> next_adm.hadm_id
               AND next_adm.admittime > current.dischtime
            WHERE current.dischtime IS NOT NULL
        )

        SELECT
            subject_id,
            hadm_id,
            next_hadm_id_after_discharge,
            next_admittime_after_discharge
        FROM candidate_next
        WHERE next_rank = 1
        """
    ).fetchdf()
    con.unregister("selected_chunk_for_readmission")

    output = rows.merge(
        next_admissions,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="many_to_one",
    )
    output["next_admittime_after_discharge"] = pd.to_datetime(
        output["next_admittime_after_discharge"],
        errors="coerce",
    )
    output["days_to_next_admission_after_discharge"] = (
        output["next_admittime_after_discharge"] - output["dischtime"]
    ).dt.total_seconds() / ONE_DAY_SECONDS
    return output


def assemble_model_text(parsed: dict[str, Any]) -> str:
    """Concatenate the paper sections into the model input text."""
    blocks = []
    for section in PAPER_SECTIONS:
        text = str(parsed.get(section, "") or "").strip()
        if text:
            blocks.append(f"{PAPER_SECTION_TITLES[section]}:\n{text}")
    return "\n\n".join(blocks)


def parse_and_label_chunk(
    chunk: pd.DataFrame,
    parser_module: Any,
    con: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Parse selected note sections and create prediction labels."""
    chunk = chunk.reset_index(drop=True).copy()
    datetime_columns = [
        "admittime",
        "dischtime",
        "deathtime",
        "dod",
        "discharge_note_charttime",
        "discharge_note_storetime",
    ]
    for column in datetime_columns:
        if column in chunk.columns:
            chunk[column] = pd.to_datetime(chunk[column], errors="coerce")

    parsed_rows = [
        parser_module.parse_sections(text)
        for text in chunk["full_note_text"].fillna("").astype(str)
    ]
    parsed_df = pd.DataFrame(parsed_rows).reset_index(drop=True)
    for section in PAPER_SECTIONS:
        if section in parsed_df.columns:
            chunk[section] = parsed_df[section].fillna("").astype(str)
        else:
            chunk[section] = ""

    chunk["model_text"] = [assemble_model_text(row) for row in parsed_rows]
    chunk["n_model_sections_present"] = (
        chunk[PAPER_SECTIONS].fillna("").astype(str).apply(lambda row: row.str.strip().ne("").sum(), axis=1)
    )
    chunk["model_text_n_chars"] = chunk["model_text"].str.len()
    chunk["model_text_n_words"] = chunk["model_text"].str.count(r"\S+")

    chunk["hospital_los_days"] = (
        chunk["dischtime"] - chunk["admittime"]
    ).dt.total_seconds() / ONE_DAY_SECONDS
    chunk.loc[chunk["hospital_los_days"].lt(0), "hospital_los_days"] = pd.NA
    chunk["prolonged_los_gt_7d"] = chunk["hospital_los_days"].gt(7)
    chunk["died_in_hospital"] = (
        pd.to_numeric(chunk["hospital_expire_flag"], errors="coerce").fillna(0).astype(int).eq(1)
    )

    chunk = add_next_readmission_columns(con, chunk)
    chunk["eligible_for_30d_readmission"] = (
        chunk["dischtime"].notna() & ~chunk["died_in_hospital"]
    )
    chunk["readmission_within_30d"] = (
        chunk["eligible_for_30d_readmission"]
        & pd.to_numeric(
            chunk["days_to_next_admission_after_discharge"],
            errors="coerce",
        ).between(0, 30, inclusive="both")
    )

    if "next_hadm_id_after_discharge" not in chunk.columns:
        chunk["next_hadm_id_after_discharge"] = pd.NA
    chunk["dataset_split"] = pd.NA
    for column in DATASET_COLUMNS:
        if column not in chunk.columns:
            chunk[column] = pd.NA
    return chunk.loc[:, DATASET_COLUMNS]


def write_parquet_chunk(
    writers: dict[str, pq.ParquetWriter],
    split: str,
    rows: pd.DataFrame,
) -> None:
    """Append rows for one dataset split to its parquet file."""
    if rows.empty:
        return
    table = pa.Table.from_pandas(rows, preserve_index=False)
    if split not in writers:
        writers[split] = pq.ParquetWriter(OUTPUT_DIR / SPLIT_OUTPUTS[split], table.schema)
    writers[split].write_table(table)


def dataset_split_views(parsed: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Return output views, allowing general-test/fairness-test overlap."""
    views = []
    split_masks = {
        "train": parsed["partition"].eq("train"),
        "validation": parsed["partition"].eq("validation"),
        "test_general": (
            parsed["partition"].eq("test_pool")
            & pd.to_numeric(
                parsed["selected_for_general_test"],
                errors="coerce",
            ).fillna(0).eq(1)
        ),
        "test_fairness_mhh1_mhc0": (
            parsed["partition"].eq("test_pool")
            & (
                pd.to_numeric(
                    parsed["is_matched_mhh1_psychotic_admission"],
                    errors="coerce",
                ).fillna(0).eq(1)
                | pd.to_numeric(
                    parsed["is_matched_mhc0_admission"],
                    errors="coerce",
                ).fillna(0).eq(1)
            )
        ),
    }
    for split, mask in split_masks.items():
        split_rows = parsed.loc[mask].copy()
        if not split_rows.empty:
            split_rows["dataset_split"] = split
            views.append((split, split_rows))
    return views


def summarize_dataset(table: pd.DataFrame) -> dict[str, Any]:
    """Summarize one model split or subgroup."""
    los_known = table["hospital_los_days"].notna()
    readmission_eligible = table["eligible_for_30d_readmission"].fillna(False)
    return {
        "n_rows": len(table),
        "n_subjects": table["subject_id"].nunique(),
        "n_admissions": table["hadm_id"].nunique(),
        "n_los_available": int(los_known.sum()),
        "n_prolonged_los_gt_7d": int(table.loc[los_known, "prolonged_los_gt_7d"].sum()),
        "pct_prolonged_los_gt_7d": 100 * table.loc[los_known, "prolonged_los_gt_7d"].mean()
        if los_known.any()
        else pd.NA,
        "n_eligible_for_30d_readmission": int(readmission_eligible.sum()),
        "n_readmission_within_30d": int(
            table.loc[readmission_eligible, "readmission_within_30d"].sum()
        ),
        "pct_readmission_within_30d": 100
        * table.loc[readmission_eligible, "readmission_within_30d"].mean()
        if readmission_eligible.any()
        else pd.NA,
        "median_model_text_words": table["model_text_n_words"].median(),
        "pct_with_any_model_text": 100 * table["model_text"].fillna("").astype(str).str.strip().ne("").mean(),
    }


def write_summary_outputs(all_rows: list[pd.DataFrame]) -> None:
    """Write QC summaries after all parquet files are created."""
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_rows, ignore_index=True)

    split_rows = []
    for split, group in combined.groupby("dataset_split", dropna=False):
        row = {"dataset_split": split}
        row.update(summarize_dataset(group))
        split_rows.append(row)
    split_summary = pd.DataFrame(split_rows).sort_values("dataset_split")
    split_summary.to_csv(SUMMARY_OUTPUT_DIR / "prediction_model_dataset_split_summary.csv", index=False)

    group_rows = []
    for split, split_group in combined.groupby("dataset_split", dropna=False):
        for group_name, flag in [
            ("MHH1_psychotic_full_cohort", "is_mhh1_psychotic_admission"),
            ("MHC0_full_cohort", "is_mhc0_admission"),
            ("MHH1_psychotic_matched_cohort", "is_matched_mhh1_psychotic_admission"),
            ("MHC0_matched_cohort", "is_matched_mhc0_admission"),
        ]:
            subgroup = split_group.loc[pd.to_numeric(split_group[flag], errors="coerce").fillna(0).eq(1)]
            row = {"dataset_split": split, "group_name": group_name}
            row.update(summarize_dataset(subgroup))
            group_rows.append(row)
    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(SUMMARY_OUTPUT_DIR / "prediction_model_dataset_group_summary.csv", index=False)

    section_rows = []
    for split, split_group in combined.groupby("dataset_split", dropna=False):
        for section in PAPER_SECTIONS:
            has_section = split_group[section].fillna("").astype(str).str.strip().ne("")
            section_rows.append(
                {
                    "dataset_split": split,
                    "section": section,
                    "n_rows": len(split_group),
                    "n_with_section": int(has_section.sum()),
                    "pct_with_section": 100 * has_section.mean(),
                }
            )
    section_summary = pd.DataFrame(section_rows)
    section_summary.to_csv(SUMMARY_OUTPUT_DIR / "prediction_model_dataset_section_coverage.csv", index=False)

    split_to_base_partition = {
        "train": "train",
        "validation": "validation",
        "test_general": "test_pool",
        "test_fairness_mhh1_mhc0": "test_pool",
    }
    combined["base_partition_for_qc"] = combined["dataset_split"].map(
        split_to_base_partition
    )
    subject_base_partitions = (
        combined.groupby("subject_id")["base_partition_for_qc"]
        .nunique()
        .reset_index(name="n_base_partitions")
    )
    subject_test_views = (
        combined.loc[
            combined["dataset_split"].isin(
                ["test_general", "test_fairness_mhh1_mhc0"]
            )
        ]
        .groupby("subject_id")["dataset_split"]
        .nunique()
        .reset_index(name="n_test_views")
    )
    qc = pd.DataFrame(
        [
            {
                "check": "n_subjects_crossing_train_validation_test_pool",
                "value": int(subject_base_partitions["n_base_partitions"].gt(1).sum()),
                "note": "Should be 0.",
            },
            {
                "check": "n_subjects_in_both_general_and_fairness_test_views",
                "value": int(subject_test_views["n_test_views"].gt(1).sum()),
                "note": "Expected to be nonzero because test views may intentionally overlap.",
            },
            {
                "check": "n_duplicate_subject_hadm_note_rows",
                "value": int(combined.duplicated(["subject_id", "hadm_id", "note_id", "dataset_split"]).sum()),
                "note": "Should be 0.",
            },
        ]
    )
    qc.to_csv(SUMMARY_OUTPUT_DIR / "prediction_model_dataset_qc.csv", index=False)

    print("\nDataset split summary:")
    print(split_summary.to_string(index=False))
    print("\nGroup summary:")
    print(group_summary.to_string(index=False))
    print("\nQC:")
    print(qc.to_string(index=False))


def main() -> None:
    """Build partitioned model datasets from raw discharge-note text."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.glob("*.parquet"):
        path.unlink()

    parser_module = load_parser_module()
    writers: dict[str, pq.ParquetWriter] = {}
    summary_chunks: list[pd.DataFrame] = []

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        ensure_required_tables(con)
        selected_query = selected_rows_sql()
        total_rows = con.execute(f"SELECT COUNT(*) FROM ({selected_query})").fetchone()[0]
        print(f"Rows selected for model-dataset creation: {total_rows:,}")

        selected = con.execute(selected_query).fetchdf()
        if len(selected) != total_rows:
            raise ValueError(
                f"Selected row count changed during fetch: expected {total_rows}, "
                f"got {len(selected)}"
            )
        batch_index = 0
        processed_rows = 0
        for start in range(0, len(selected), CHUNK_SIZE):
            chunk = selected.iloc[start : start + CHUNK_SIZE].copy()
            if chunk.empty:
                continue
            batch_index += 1
            parsed = parse_and_label_chunk(chunk, parser_module, con)
            processed_rows += len(parsed)
            for split, split_rows in dataset_split_views(parsed):
                write_parquet_chunk(writers, split, split_rows)
                summary_chunks.append(split_rows)
            print(
                f"Processed batch {batch_index}: {processed_rows:,}/{total_rows:,} rows",
                flush=True,
            )

    for writer in writers.values():
        writer.close()

    if not summary_chunks:
        raise ValueError("No rows were selected for dataset creation.")
    write_summary_outputs(summary_chunks)
    print(f"\nWrote parquet datasets to: {OUTPUT_DIR}")
    print(f"Wrote summaries to: {SUMMARY_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
