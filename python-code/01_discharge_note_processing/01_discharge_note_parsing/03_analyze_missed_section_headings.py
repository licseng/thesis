"""List potential discharge-note headings not hard-coded in the full parser.

This diagnostic script is for improving `02_parse_full_discharge_notes.py`.
It scans the already parsed matched full-note outputs and reports heading-like
strings only. It does not export note bodies.

Outputs:
    analysis_output_discharge_note_parsing/missed_section_heading_candidates.csv
    analysis_output_discharge_note_parsing/missed_section_heading_summary.csv
"""

from __future__ import annotations

import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FULL_NOTE_DIR = SCRIPT_DIR / "full_discharge_note_sections"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_discharge_note_parsing"

PARSER_PATH = SCRIPT_DIR / "02_parse_full_discharge_notes.py"
FULL_NOTE_FILES = [
    {
        "cohort": "MHH1_psychotic",
        "path": FULL_NOTE_DIR / "MHH1_psychotic_matched_full_discharge_note_sections.parquet",
    },
    {
        "cohort": "MHC0",
        "path": FULL_NOTE_DIR / "MHC0_matched_full_discharge_note_sections.parquet",
    },
]

NON_SECTION_HEADING_PATTERNS = [
    r"date of birth",
    r"sex",
    r"service",
    r"unit",
    r"phone",
    r"fax",
    r"provider",
    r"attending",
    r"dictated by",
    r"completed by",
    r"signed electronically",
    r"level of consciousness",
    r"mental status",
    r"temperature",
    r"heart rate",
    r"blood pressure",
    r"respiratory rate",
    r"oxygen saturation",
    r"\bsat\b",
    r"\bpulse\b",
    r"vitals?",
    r"\bdp right\b",
    r"\bdp left\b",
    r"\bpt right\b",
    r"\bpt left\b",
]

LIKELY_SECTION_TERMS = [
    "admission",
    "assessment",
    "course",
    "diagnos",
    "exam",
    "history",
    "hospital",
    "imaging",
    "impression",
    "instruction",
    "issue",
    "lab",
    "medication",
    "micro",
    "plan",
    "procedure",
    "problem",
    "result",
    "social",
    "study",
]


def load_parser_module() -> Any:
    """Load parser helpers so diagnostics use the same heading rules."""
    spec = importlib.util.spec_from_file_location("full_note_parser", PARSER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load parser module from {PARSER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_heading_for_output(heading: str, parser: Any) -> str:
    """Normalize heading text using the parser's normalization function."""
    return parser.normalize_heading(heading)


def safe_load_sections(value: str) -> list[dict[str, Any]]:
    """Load section JSON produced by the full-note parser."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def count_undetected_colon_heading_candidates(
    df: pd.DataFrame,
    cohort: str,
    parser: Any,
) -> list[dict[str, Any]]:
    """Count colon-ending heading candidates not detected by parser rules."""
    note_ids_by_heading: dict[str, set[tuple[Any, Any]]] = defaultdict(set)
    occurrence_counts: Counter[str] = Counter()

    for row in df.itertuples(index=False):
        note_id = (row.subject_id, row.hadm_id)
        note_text = row.full_note_text if isinstance(row.full_note_text, str) else ""
        for line in note_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            raw_match = parser.HEADING_LINE_RE.match(line)
            if raw_match:
                if parser.is_heading_line(line):
                    continue
                heading = normalize_heading_for_output(raw_match.group("heading"), parser)
            else:
                # Conservative fallback for short all-caps headings missed by the
                # parser's character whitelist.
                simple_match = re.match(r"^\s*([A-Z][A-Z0-9 /_&,\-()]{1,90})\s*:\s*", line)
                if not simple_match:
                    continue
                heading = normalize_heading_for_output(simple_match.group(1), parser)

            if not heading or heading in parser.HEADING_ALIASES:
                continue
            occurrence_counts[heading] += 1
            note_ids_by_heading[heading].add(note_id)

    return [
        {
            "cohort": cohort,
            "candidate_type": "colon_heading_candidate_not_hard_coded",
            "heading": heading,
            "n_occurrences": int(count),
            "n_notes": len(note_ids_by_heading[heading]),
            "current_parser_behavior": "left inside surrounding section text",
        }
        for heading, count in occurrence_counts.items()
    ]


def load_parsed_outputs() -> list[tuple[str, pd.DataFrame]]:
    """Load only columns needed for heading diagnostics."""
    loaded = []
    required_columns = [
        "subject_id",
        "hadm_id",
        "all_detected_sections_json",
        "full_note_text",
    ]
    for file_config in FULL_NOTE_FILES:
        path = file_config["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing parsed full-note output: {path}")
        loaded.append(
            (
                file_config["cohort"],
                pd.read_parquet(path, columns=required_columns),
            )
        )
    return loaded


def write_outputs(candidates: pd.DataFrame) -> None:
    """Write detailed and summarized heading-candidate CSVs."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    candidates = candidates.sort_values(
        ["candidate_type", "n_notes", "n_occurrences", "cohort", "heading"],
        ascending=[True, False, False, True, True],
    )
    candidates.to_csv(OUTPUT_DIR / "missed_section_heading_candidates.csv", index=False)

    likely = candidates[
        candidates["heading"].map(is_likely_section_heading)
        & (candidates["n_notes"] >= 10)
    ].copy()
    likely = likely.sort_values(
        ["n_notes", "n_occurrences", "cohort", "heading"],
        ascending=[False, False, True, True],
    )
    likely.to_csv(
        OUTPUT_DIR / "likely_missing_section_heading_candidates.csv",
        index=False,
    )

    summary = (
        candidates.groupby(["candidate_type"], as_index=False)
        .agg(
            n_unique_headings=("heading", "nunique"),
            total_occurrences=("n_occurrences", "sum"),
            max_notes_for_one_heading=("n_notes", "max"),
        )
        .sort_values("candidate_type")
    )
    summary.to_csv(OUTPUT_DIR / "missed_section_heading_summary.csv", index=False)

    print(f"Saved heading diagnostics to: {OUTPUT_DIR}")
    print(summary.to_string(index=False))
    print()
    print("Top candidates:")
    print(candidates.head(30).to_string(index=False))
    print()
    print("Top likely section-heading candidates:")
    print(likely.head(30).to_string(index=False))


def is_likely_section_heading(heading: str) -> bool:
    """Return True for heading candidates that look like real note sections."""
    heading = str(heading).strip().lower()
    if not heading:
        return False
    if any(re.search(pattern, heading) for pattern in NON_SECTION_HEADING_PATTERNS):
        return False
    return any(term in heading for term in LIKELY_SECTION_TERMS)


def main() -> None:
    """Build heading diagnostics from the matched full-note parser outputs."""
    parser = load_parser_module()
    rows: list[dict[str, Any]] = []
    for cohort, df in load_parsed_outputs():
        rows.extend(count_undetected_colon_heading_candidates(df, cohort, parser))

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        candidates = pd.DataFrame(
            columns=[
                "cohort",
                "candidate_type",
                "heading",
                "n_occurrences",
                "n_notes",
                "current_parser_behavior",
            ]
        )
    write_outputs(candidates)


if __name__ == "__main__":
    main()
