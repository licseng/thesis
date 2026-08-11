"""Analyze `confused` documentation in discharge-condition sections.

This script checks whether `confused` in the parsed discharge_condition section
is associated with discharge location, separately for matched MHH1_psychotic and
MHC0 admissions. It writes aggregate counts, percentages, risk differences, and
risk ratios only. It does not write raw note text.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PYTHON_DIR = SCRIPT_DIR.parent
PARSER_DIR = REPO_PYTHON_DIR / "01_discharge_note_preprocessing" / "01_discharge_note_parsing"
FULL_NOTE_DIR = PARSER_DIR / "full_discharge_note_sections"
COHORT_MATCHING_DIR = REPO_PYTHON_DIR / "02_cohort_matching"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_confused_discharge_condition"

sys.path.insert(0, str(COHORT_MATCHING_DIR.resolve()))
import _matched_cohort_characterization_common as cohort_characterization  # noqa: E402


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
ID_COLUMNS = ["cohort", "subject_id", "hadm_id"]
CONFUSED_RE = re.compile(r"\bconfused\b", flags=re.IGNORECASE)
MIN_STRATUM_DENOMINATOR = 30


def safe_pct(numerator: int | float, denominator: int | float) -> float:
    """Return a percentage, with zero for empty denominators."""
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a ratio, with NA for zero denominators."""
    if denominator == 0:
        return pd.NA
    return numerator / denominator


def load_discharge_condition_sections() -> pd.DataFrame:
    """Load parsed discharge-condition sections for both matched cohorts."""
    frames = []
    for file_config in FULL_NOTE_FILES:
        path = file_config["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing parsed discharge-note sections: {path}")
        frame = pd.read_parquet(
            path,
            columns=["subject_id", "hadm_id", "discharge_condition"],
        )
        frame.insert(0, "cohort", file_config["cohort"])
        frames.append(frame)

    sections = pd.concat(frames, ignore_index=True)
    sections["subject_id"] = pd.to_numeric(
        sections["subject_id"],
        errors="raise",
    ).astype(int)
    sections["hadm_id"] = pd.to_numeric(sections["hadm_id"], errors="raise").astype(int)
    sections["has_discharge_condition"] = (
        sections["discharge_condition"].notna()
        & sections["discharge_condition"].astype(str).str.strip().ne("")
    )
    sections["confused_in_discharge_condition"] = (
        sections["discharge_condition"]
        .fillna("")
        .astype(str)
        .str.contains(CONFUSED_RE, regex=True)
    )
    return sections.drop(columns=["discharge_condition"])


def load_discharge_locations() -> pd.DataFrame:
    """Load matched-cohort discharge locations from DBeaver descriptors."""
    descriptors = cohort_characterization.add_derived_descriptor_columns(
        cohort_characterization.validate_id_columns(
            cohort_characterization.load_required_table("descriptors"),
            "descriptors",
        )
    )
    required = set(ID_COLUMNS + ["discharge_location"])
    missing = sorted(required - set(descriptors.columns))
    if missing:
        raise ValueError(f"Descriptor table is missing required column(s): {missing}")

    locations = descriptors.loc[:, ID_COLUMNS + ["discharge_location"]].drop_duplicates()
    locations["discharge_location"] = (
        locations["discharge_location"]
        .fillna("missing")
        .astype(str)
        .str.strip()
        .replace({"": "missing"})
    )
    return locations


def collapse_discharge_location(value: str) -> str:
    """Collapse sparse discharge locations into interpretable care-setting groups."""
    location = str(value).strip().upper()
    if location == "SKILLED NURSING FACILITY":
        return "skilled_nursing_facility"
    if location == "HOME HEALTH CARE":
        return "home_health_care"
    if location == "HOME":
        return "home"
    if location in {"CHRONIC/LONG TERM ACUTE CARE", "REHAB", "OTHER FACILITY", "ACUTE HOSPITAL"}:
        return "other_facility_or_rehab"
    if location == "PSYCH FACILITY":
        return "psych_facility"
    if location in {"HOSPICE", "DIED"}:
        return "hospice_or_died"
    if location in {"MISSING", "NAN", "NONE"}:
        return "missing"
    return "other_or_rare"


def build_analysis_table() -> pd.DataFrame:
    """Join parsed discharge-condition flags to discharge location descriptors."""
    sections = load_discharge_condition_sections()
    locations = load_discharge_locations()
    analysis = locations.merge(
        sections,
        on=ID_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    analysis["has_discharge_condition"] = (
        analysis["has_discharge_condition"].fillna(False).astype(bool)
    )
    analysis["confused_in_discharge_condition"] = (
        analysis["confused_in_discharge_condition"].fillna(False).astype(bool)
    )
    analysis["discharge_location_group"] = analysis["discharge_location"].map(
        collapse_discharge_location
    )
    return analysis


def summarize_overall_by_cohort(analysis: pd.DataFrame) -> pd.DataFrame:
    """Summarize overall confused documentation by cohort."""
    return (
        analysis.groupby("cohort", as_index=False)
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_with_discharge_condition=("has_discharge_condition", "sum"),
            n_confused=("confused_in_discharge_condition", "sum"),
        )
        .assign(
            pct_with_discharge_condition=lambda df: df.apply(
                lambda row: safe_pct(row["n_with_discharge_condition"], row["n_admissions"]),
                axis=1,
            ),
            pct_confused_among_all_admissions=lambda df: df.apply(
                lambda row: safe_pct(row["n_confused"], row["n_admissions"]),
                axis=1,
            ),
            pct_confused_among_admissions_with_discharge_condition=lambda df: df.apply(
                lambda row: safe_pct(
                    row["n_confused"],
                    row["n_with_discharge_condition"],
                ),
                axis=1,
            ),
        )
    )


def summarize_by_location(
    analysis: pd.DataFrame,
    location_column: str,
) -> pd.DataFrame:
    """Summarize confused documentation by cohort and discharge-location stratum."""
    summary = (
        analysis.groupby(["cohort", location_column], as_index=False)
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_with_discharge_condition=("has_discharge_condition", "sum"),
            n_confused=("confused_in_discharge_condition", "sum"),
        )
    )
    summary["pct_confused"] = summary.apply(
        lambda row: safe_pct(row["n_confused"], row["n_admissions"]),
        axis=1,
    )
    summary["pct_confused_among_admissions_with_discharge_condition"] = summary.apply(
        lambda row: safe_pct(row["n_confused"], row["n_with_discharge_condition"]),
        axis=1,
    )
    return summary.sort_values(["cohort", "n_admissions"], ascending=[True, False])


def build_cohort_comparison(
    location_summary: pd.DataFrame,
    location_column: str,
) -> pd.DataFrame:
    """Compare MHH1 vs MHC0 confused risk within each discharge-location stratum."""
    pivot = location_summary.pivot(
        index=location_column,
        columns="cohort",
        values=["n_admissions", "n_confused", "pct_confused"],
    )
    pivot.columns = [f"{metric}_{cohort}" for metric, cohort in pivot.columns]
    pivot = pivot.reset_index()

    required_columns = [
        "n_admissions_MHH1_psychotic",
        "n_admissions_MHC0",
        "n_confused_MHH1_psychotic",
        "n_confused_MHC0",
        "pct_confused_MHH1_psychotic",
        "pct_confused_MHC0",
    ]
    for column in required_columns:
        if column not in pivot.columns:
            pivot[column] = 0
    pivot[required_columns] = pivot[required_columns].fillna(0)

    pivot["risk_difference_pct_points_MHH1_minus_MHC0"] = (
        pivot["pct_confused_MHH1_psychotic"] - pivot["pct_confused_MHC0"]
    )
    pivot["risk_ratio_MHH1_vs_MHC0"] = pivot.apply(
        lambda row: safe_ratio(
            row["pct_confused_MHH1_psychotic"],
            row["pct_confused_MHC0"],
        ),
        axis=1,
    )
    pivot["low_count_flag"] = (
        (pivot["n_admissions_MHH1_psychotic"] < MIN_STRATUM_DENOMINATOR)
        | (pivot["n_admissions_MHC0"] < MIN_STRATUM_DENOMINATOR)
    )
    return pivot.sort_values(
        ["low_count_flag", "n_admissions_MHH1_psychotic", "n_admissions_MHC0"],
        ascending=[True, False, False],
    )


def summarize_location_distribution_among_confused(
    analysis: pd.DataFrame,
    location_column: str,
) -> pd.DataFrame:
    """Summarize P(discharge location | confused) by cohort."""
    confused = analysis.loc[analysis["confused_in_discharge_condition"]].copy()
    summary = (
        confused.groupby(["cohort", location_column], as_index=False)
        .agg(n_confused_admissions=("hadm_id", "nunique"))
    )
    totals = (
        confused.groupby("cohort", as_index=False)
        .agg(total_confused_admissions=("hadm_id", "nunique"))
    )
    summary = summary.merge(totals, on="cohort", how="left")
    summary["pct_of_confused_admissions"] = summary.apply(
        lambda row: safe_pct(
            row["n_confused_admissions"],
            row["total_confused_admissions"],
        ),
        axis=1,
    )
    return summary.sort_values(["cohort", "n_confused_admissions"], ascending=[True, False])


def write_outputs(analysis: pd.DataFrame) -> None:
    """Write all aggregate confused/discharge-location analysis outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    exact_summary = summarize_by_location(analysis, "discharge_location")
    collapsed_summary = summarize_by_location(analysis, "discharge_location_group")

    outputs = {
        "confused_overall_by_cohort.csv": summarize_overall_by_cohort(analysis),
        "confused_by_discharge_location.csv": exact_summary,
        "confused_by_discharge_location_group.csv": collapsed_summary,
        "confused_discharge_location_cohort_comparison.csv": build_cohort_comparison(
            exact_summary,
            "discharge_location",
        ),
        "confused_discharge_location_group_cohort_comparison.csv": build_cohort_comparison(
            collapsed_summary,
            "discharge_location_group",
        ),
        "discharge_location_among_confused.csv": summarize_location_distribution_among_confused(
            analysis,
            "discharge_location",
        ),
        "discharge_location_group_among_confused.csv": summarize_location_distribution_among_confused(
            analysis,
            "discharge_location_group",
        ),
    }
    for filename, table in outputs.items():
        table.to_csv(OUTPUT_DIR / filename, index=False)


def main() -> None:
    """Run confused/discharge-location analyses."""
    analysis = build_analysis_table()
    write_outputs(analysis)
    overall = summarize_overall_by_cohort(analysis)
    comparison = build_cohort_comparison(
        summarize_by_location(analysis, "discharge_location_group"),
        "discharge_location_group",
    )

    print(f"Saved confused discharge-condition outputs to: {OUTPUT_DIR}")
    print("\n=== Overall by Cohort ===")
    print(overall.to_string(index=False))
    print("\n=== Collapsed Discharge-Location Comparison ===")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
