"""Parse full MIMIC-IV discharge notes into broad section maps.

This script is for later psychiatric-history detection and other full-note
inspection tasks. Unlike
`01_parse_chief_complaints_from_discharge_notes.py`, it is not limited to chief
complaints. It preserves the original full note text, detects many
discharge-note headings, stores a JSON section map, and writes named columns for
commonly used sections.

This parser is still rule-based. The key design choice is that it should not
drop text: text before the first hard-coded clinical heading is stored in
`unsectioned_text`, and the original `full_note_text` remains available in the
local output.

Inputs:
    - 02_cohort_matching/matched_cohort_output/matched_pairs.parquet
    - DuckDB source tables for the two matched cohorts:
        - export_MHH_psychotic
        - export_only_MHC0

Outputs:
    full_discharge_note_sections/<output_name>.parquet
    full_discharge_note_sections/<output_name>_sample.csv
    full_discharge_note_sections/<output_name>_section_summary.csv
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
THESIS_DIR = PROJECT_DIR.parent.parent.parent
DB_PATH = THESIS_DIR / "DataBase"
OUTPUT_DIR = SCRIPT_DIR / "full_discharge_note_sections"
MATCHED_PAIRS_PATH = (
    PROJECT_DIR.parent
    / "02_cohort_matching"
    / "matched_cohort_output"
    / "matched_pairs.parquet"
)
SAMPLE_SIZE = 5000

# Matched cohorts to parse. The matched-pairs parquet provides the admission IDs;
# the DuckDB table provides the full discharge-note text.
MATCHED_EXPORTS = [
    {
        "cohort": "MHH1_psychotic",
        "source_table": "export_MHH_psychotic",
        "subject_col": "mhh_subject_id",
        "hadm_col": "mhh_hadm_id",
        "output_name": "MHH1_psychotic_matched_full_discharge_note_sections",
    },
    {
        "cohort": "MHC0",
        "source_table": "export_only_MHC0",
        "subject_col": "mhc0_subject_id",
        "hadm_col": "mhc0_hadm_id",
        "output_name": "MHC0_matched_full_discharge_note_sections",
    },
]

# Named sections that become explicit output columns. Repeated headings are
# concatenated within the same canonical section.
CANONICAL_SECTIONS = [
    "chief_complaint",
    "major_surgical_or_invasive_procedure",
    "present_illness",
    "medical_history",
    "past_psychiatric_history",
    "medication_admission",
    "allergies",
    "review_of_systems",
    "physical_exam",
    "family_history",
    "social_history",
    "problems",
    "pertinent_results",
    "brief_hospital_course",
    "discharge_medications",
    "discharge_disposition",
    "discharge_diagnosis",
    "discharge_condition",
    "discharge_instructions",
]

# Heading aliases observed so far. Keys are normalized heading text without the
# colon; values are canonical output sections.
HEADING_ALIASES = {
    "cc": "chief_complaint",
    "chief complaint": "chief_complaint",
    "major surgical or invasive procedure": "major_surgical_or_invasive_procedure",
    "major surgical or invasive procedures": "major_surgical_or_invasive_procedure",
    "present illness": "present_illness",
    "history of present illness": "present_illness",
    "hpi": "present_illness",
    "medical history": "medical_history",
    "past medical history": "medical_history",
    "pmh": "medical_history",
    "past psychiatric history": "past_psychiatric_history",
    "psychiatric history": "past_psychiatric_history",
    "medications on admission": "medication_admission",
    "medication on admission": "medication_admission",
    "admission medications": "medication_admission",
    "allergies": "allergies",
    "review of systems": "review_of_systems",
    "ros": "review_of_systems",
    "admission physical": "physical_exam",
    "admission physical exam": "physical_exam",
    "physical exam": "physical_exam",
    "pe": "physical_exam",
    "physical examination": "physical_exam",
    "physical exam on discharge": "physical_exam",
    "discharge physical": "physical_exam",
    "discharge physical exam": "physical_exam",
    "family history": "family_history",
    "social history": "social_history",
    "personal and social history": "social_history",
    "problems": "problems",
    "pertinent results": "pertinent_results",
    "admission labs": "pertinent_results",
    "pertinent labs": "pertinent_results",
    "micro": "pertinent_results",
    "microbiology": "pertinent_results",
    "imaging": "pertinent_results",
    "imaging/other studies": "pertinent_results",
    "imaging and other studies": "pertinent_results",
    "labs on discharge": "pertinent_results",
    "discharge labs": "pertinent_results",
    "brief hospital course": "brief_hospital_course",
    "hospital course": "brief_hospital_course",
    "acute issues": "problems",
    "chronic issues": "problems",
    "plan/transitional issues": "problems",
    "transitional issues": "problems",
    "discharge medications": "discharge_medications",
    "discharge medication": "discharge_medications",
    "discharge disposition": "discharge_disposition",
    "discharge diagnosis": "discharge_diagnosis",
    "discharge diagnoses": "discharge_diagnosis",
    "primary diagnosis": "discharge_diagnosis",
    "primary diagnoses": "discharge_diagnosis",
    "secondary diagnosis": "discharge_diagnosis",
    "secondary diagnoses": "discharge_diagnosis",
    "discharge condition": "discharge_condition",
    "discharge instructions": "discharge_instructions",
    "followup instructions": "discharge_instructions",
    "follow-up instructions": "discharge_instructions",
}

# Hard-coded heading pattern. A line is treated as a section boundary only if
# the normalized heading is present in `HEADING_ALIASES`; colon-style clinical
# labels inside a section should not split the note into fake sections.
HEADING_LINE_RE = re.compile(
    r"^\s*(?P<heading>[A-Za-z][A-Za-z0-9/_&,\-() ]{1,90}|[A-Z]{2,12})\s*:\s*(?P<rest>.*)$"
)

# Keep the full-note parser's chief_complaint column aligned with the dedicated
# chief-complaint pipeline. That earlier pipeline extracts only text after
# "Chief Complaint:" and stops at the next likely section heading.
CHIEF_COMPLAINT_BOUNDARY_PATTERN = (
    r"(?:hpi|[A-Z][A-Za-z]{4,}(?:[ \t]+(?:[A-Za-z]{2,}|_+)){0,8})[ \t]*:"
)
CHIEF_COMPLAINT_RE = re.compile(
    rf"(?i)(?:{re.escape('chief complaint:')})"
    rf"([\s\S]+?)(?:\n\s*(?:{CHIEF_COMPLAINT_BOUNDARY_PATTERN})|$)"
)


def quote_identifier(identifier: str) -> str:
    """Safely quote a SQL identifier."""
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
    """Identify source columns that should be copied through as metadata."""
    excluded_columns = {"subject_id", "hadm_id", "text"}
    return [column for column in table_columns(con, table_name) if column not in excluded_columns]


def ensure_matched_pairs_exists() -> None:
    """Fail early if the matched cohort table is absent."""
    if not MATCHED_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing matched pairs parquet: {MATCHED_PAIRS_PATH}")


def load_matched_admission_ids(export_config: dict[str, str]) -> pd.DataFrame:
    """Load the matched admission IDs for one cohort side."""
    matched = pd.read_parquet(
        MATCHED_PAIRS_PATH,
        columns=[export_config["subject_col"], export_config["hadm_col"]],
    )
    ids = matched.rename(
        columns={
            export_config["subject_col"]: "subject_id",
            export_config["hadm_col"]: "hadm_id",
        }
    )
    return ids.dropna(subset=["subject_id", "hadm_id"]).drop_duplicates()


def normalize_heading(heading: str) -> str:
    """Normalize heading text for alias lookup."""
    heading = heading.strip().strip(":").lower()
    heading = re.sub(r"\s+", " ", heading)
    heading = heading.replace("___", "")
    return heading.strip()


def canonical_section_for_heading(heading: str) -> str:
    """Map a hard-coded raw heading to a canonical section."""
    normalized = normalize_heading(heading)
    return HEADING_ALIASES[normalized]


def is_heading_line(line: str) -> re.Match[str] | None:
    """Return a heading match only for hard-coded discharge-note headings."""
    match = HEADING_LINE_RE.match(line)
    if not match:
        return None

    heading = normalize_heading(match.group("heading"))
    if heading in HEADING_ALIASES:
        return match

    return None


def clean_section_text(text: str) -> str:
    """Normalize extracted section text and drop placeholder-only sections."""
    text = re.sub(r"(?<!\S)_+(?!\S)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    return text


def extract_pipeline_chief_complaint(note_text: str) -> str:
    """Extract chief complaint with the dedicated chief-complaint parser logic."""
    note_text = re.sub(
        r"(?i)___\nFamily History:",
        "___\n\nFamily History:",
        note_text or "",
    )
    match = CHIEF_COMPLAINT_RE.search(note_text)
    if not match:
        return ""
    chief_complaint = match.group(1).replace("\n", " ").strip()
    if chief_complaint.startswith("[]"):
        return ""
    return chief_complaint


def parse_sections(note_text: str) -> dict[str, Any]:
    """Parse one discharge note into canonical section columns plus JSON map."""
    note_text = note_text or ""
    lines = note_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    sections: list[dict[str, str]] = []
    current_heading = "unsectioned"
    current_canonical = "unsectioned"
    current_lines: list[str] = []

    def flush_current() -> None:
        text = clean_section_text("\n".join(current_lines))
        if text or current_heading != "unsectioned":
            sections.append(
                {
                    "heading": current_heading,
                    "normalized_heading": normalize_heading(current_heading),
                    "canonical_section": current_canonical,
                    "text": text,
                }
            )

    for line in lines:
        heading_match = is_heading_line(line)
        if heading_match:
            flush_current()
            current_heading = heading_match.group("heading").strip()
            current_canonical = canonical_section_for_heading(current_heading)
            rest = heading_match.group("rest").strip()
            current_lines = [rest] if rest else []
        else:
            current_lines.append(line)

    flush_current()

    section_texts: dict[str, list[str]] = defaultdict(list)
    for section in sections:
        canonical = section["canonical_section"]
        if canonical in CANONICAL_SECTIONS and section["text"]:
            section_texts[canonical].append(section["text"])

    parsed = {
        section: " ".join(section_texts.get(section, [])).strip()
        for section in CANONICAL_SECTIONS
    }
    parsed["chief_complaint"] = extract_pipeline_chief_complaint(note_text)
    parsed["unsectioned_text"] = " ".join(
        section["text"]
        for section in sections
        if section["canonical_section"] == "unsectioned" and section["text"]
    ).strip()
    parsed["all_detected_sections_json"] = json.dumps(
        sections, ensure_ascii=True, separators=(",", ":")
    )
    parsed["detected_section_headings"] = " | ".join(
        section["heading"] for section in sections if section["heading"] != "unsectioned"
    )
    parsed["n_detected_sections"] = sum(
        1 for section in sections if section["heading"] != "unsectioned"
    )
    parsed["full_note_text"] = note_text
    return parsed


def load_source_table(
    con: duckdb.DuckDBPyConnection,
    source_table: str,
    metadata_column_names: list[str],
    matched_ids: pd.DataFrame,
) -> pd.DataFrame:
    """Load full notes only for admissions present in the matched cohort."""
    con.register("matched_admission_filter", matched_ids)
    metadata_select = ", ".join(
        f"source.{quote_identifier(column)}" for column in metadata_column_names
    )
    metadata_sql = f", {metadata_select}" if metadata_select else ""
    return con.execute(
        f"""
        SELECT
            source.subject_id,
            source.hadm_id
            {metadata_sql},
            coalesce(source.text, '') AS text
        FROM {quote_identifier(source_table)} AS source
        INNER JOIN matched_admission_filter AS matched
            ON source.subject_id = matched.subject_id
           AND source.hadm_id = matched.hadm_id
        WHERE source.subject_id IS NOT NULL
          AND source.hadm_id IS NOT NULL
          AND source.text IS NOT NULL
          AND trim(source.text) <> ''
        """
    ).fetchdf()


def parse_source_table(
    con: duckdb.DuckDBPyConnection,
    export_config: dict[str, str],
) -> pd.DataFrame:
    """Load and parse matched admissions from one source table."""
    source_table = export_config["source_table"]
    output_name = export_config["output_name"]
    matched_ids = load_matched_admission_ids(export_config)
    metadata_column_names = metadata_columns(con, source_table)
    source_df = load_source_table(
        con=con,
        source_table=source_table,
        metadata_column_names=metadata_column_names,
        matched_ids=matched_ids,
    )
    parsed_records = [parse_sections(text) for text in source_df["text"]]
    parsed_df = pd.DataFrame(parsed_records)
    output_df = pd.concat(
        [
            source_df.drop(columns=["text"]).reset_index(drop=True),
            parsed_df.reset_index(drop=True),
        ],
        axis=1,
    )
    output_df.insert(0, "source_table", source_table)
    output_df.insert(1, "output_name", output_name)
    output_df.insert(2, "cohort", export_config["cohort"])
    return output_df


def write_outputs(parsed_df: pd.DataFrame, output_name: str) -> None:
    """Write one parsed output table, sample, and section-coverage summary."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    parquet_path = OUTPUT_DIR / f"{output_name}.parquet"
    sample_path = OUTPUT_DIR / f"{output_name}_sample.csv"
    summary_path = OUTPUT_DIR / f"{output_name}_section_summary.csv"

    parsed_df.to_parquet(parquet_path, index=False)
    parsed_df.head(SAMPLE_SIZE).to_csv(sample_path, index=False)

    summary_rows = []
    for section in CANONICAL_SECTIONS + ["unsectioned_text"]:
        has_section = parsed_df[section].fillna("").astype(str).str.strip().ne("")
        summary_rows.append(
            {
                "output_name": output_name,
                "section": section,
                "n_with_section": int(has_section.sum()),
                "pct_with_section": 100.0 * has_section.mean(),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print(
        f"{output_name}: rows={len(parsed_df)}, "
        f"median_detected_sections={parsed_df['n_detected_sections'].median()}"
    )
    print(summary_df.to_string(index=False))


def ensure_database_exists() -> None:
    """Fail early with a clear message if the DuckDB database file is absent."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"No DuckDB database found at: {DB_PATH}")


def main() -> None:
    """Parse full discharge-note sections for the already matched cohort only."""
    ensure_database_exists()
    ensure_matched_pairs_exists()
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        for export in MATCHED_EXPORTS:
            source_table = export["source_table"]
            if not table_exists(con, source_table):
                raise ValueError(f"Missing required DuckDB table: {source_table}")

            output_name = export["output_name"]
            print(
                f"Parsing matched full discharge-note sections from {source_table}",
                flush=True,
            )
            parsed_df = parse_source_table(con, export)
            write_outputs(parsed_df, output_name)
    finally:
        con.close()

    print(f"Saved full discharge-note section outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
