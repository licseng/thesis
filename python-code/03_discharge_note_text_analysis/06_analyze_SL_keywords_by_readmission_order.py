"""Analyze SL keyword hits by matched-admission order.

This script checks whether the MHH1-vs-MHC0 SL keyword difference is partly
explained by repeated admissions. It reuses the SL keyword list and selected
sections from ``02_explore_SL_keywords_by_section.py``, restricts to the current
matched cohort, and summarizes keyword-hit rates by admission order within each
subject.

Outputs:
    analysis_output_SL_keyword_readmission_order/
        SL_keyword_readmission_order_admission_summary.csv
        SL_keyword_readmission_order_overall_summary.csv
        SL_keyword_readmission_order_bucket_summary.csv
        SL_keyword_prior_admission_bucket_summary.csv
        SL_keyword_prior_admission_group_summary.csv
        SL_keyword_readmission_order_term_summary.csv
        SL_keyword_readmission_order_qc.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
COHORT_DIR = PROJECT_DIR / "02_cohort_matching"
MATCHED_IDS_PATH = COHORT_DIR / "matched_cohort_output" / "matched_admission_ids_for_dbeaver.csv"
FULL_NOTE_DIR = (
    PROJECT_DIR
    / "01_discharge_note_preprocessing"
    / "01_discharge_note_parsing"
    / "full_discharge_note_sections"
)
SL_SCRIPT_PATH = SCRIPT_DIR / "02_explore_SL_keywords_by_section.py"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_SL_keyword_readmission_order"

FULL_NOTE_FILES = [
    FULL_NOTE_DIR / "MHH1_psychotic_matched_full_discharge_note_sections.parquet",
    FULL_NOTE_DIR / "MHC0_matched_full_discharge_note_sections.parquet",
]


def load_sl_module() -> Any:
    """Load the existing SL keyword exploration script as a helper module."""
    spec = importlib.util.spec_from_file_location("sl_keyword_exploration", SL_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SL keyword script from {SL_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_matched_ids() -> pd.DataFrame:
    """Load the current matched admission IDs and standardize ID columns."""
    if not MATCHED_IDS_PATH.exists():
        raise FileNotFoundError(f"Missing current matched admission IDs: {MATCHED_IDS_PATH}")
    matched_ids = pd.read_csv(MATCHED_IDS_PATH)
    required = {"pair_id", "matched_role", "cohort", "subject_id", "hadm_id"}
    missing = sorted(required - set(matched_ids.columns))
    if missing:
        raise ValueError(f"Matched ID file is missing columns: {missing}")

    matched_ids = matched_ids.loc[:, ["pair_id", "matched_role", "cohort", "subject_id", "hadm_id"]].copy()
    matched_ids["cohort"] = matched_ids["cohort"].astype("string").str.strip()
    matched_ids["matched_role"] = matched_ids["matched_role"].astype("string").str.strip()
    matched_ids["subject_id"] = pd.to_numeric(matched_ids["subject_id"], errors="raise").astype(int)
    matched_ids["hadm_id"] = pd.to_numeric(matched_ids["hadm_id"], errors="raise").astype(int)
    matched_ids["pair_id"] = pd.to_numeric(matched_ids["pair_id"], errors="raise").astype(int)

    duplicate_keys = matched_ids.duplicated(["cohort", "subject_id", "hadm_id"], keep=False)
    if duplicate_keys.any():
        n_duplicate = int(duplicate_keys.sum())
        raise ValueError(f"Matched ID file has duplicated cohort/subject/hadm rows: {n_duplicate}")
    return matched_ids


def load_current_matched_notes(
    matched_ids: pd.DataFrame,
    section_columns: list[str],
) -> pd.DataFrame:
    """Load parsed note sections and keep only current matched admissions."""
    columns = ["cohort", "subject_id", "hadm_id", "admittime", "note_id"] + section_columns
    frames = []
    for path in FULL_NOTE_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing parsed full-note sections: {path}")
        note_df = pd.read_parquet(path, columns=columns)
        note_df["cohort"] = note_df["cohort"].astype("string").str.strip()
        note_df["subject_id"] = pd.to_numeric(note_df["subject_id"], errors="raise").astype(int)
        note_df["hadm_id"] = pd.to_numeric(note_df["hadm_id"], errors="raise").astype(int)
        frames.append(note_df)

    notes = pd.concat(frames, ignore_index=True)
    notes = notes.merge(
        matched_ids,
        on=["cohort", "subject_id", "hadm_id"],
        how="inner",
        validate="one_to_one",
    )
    notes["admittime"] = pd.to_datetime(notes["admittime"], errors="coerce")
    if notes["admittime"].isna().any():
        raise ValueError("Some current matched admissions have missing/unparseable admittime.")
    return notes


def add_admission_order(notes: pd.DataFrame) -> pd.DataFrame:
    """Add within-subject matched-admission order and coarse order buckets."""
    output = notes.sort_values(["cohort", "subject_id", "admittime", "hadm_id"]).copy()
    grouped = output.groupby(["cohort", "subject_id"], sort=False)
    output["matched_admission_order"] = grouped.cumcount() + 1
    output["n_matched_admissions_for_subject"] = grouped["hadm_id"].transform("size")
    output["n_prior_matched_admissions"] = output["matched_admission_order"] - 1
    output["prior_admission_bucket"] = output["n_prior_matched_admissions"].map(
        lambda n: "0_prior" if n == 0 else "1_prior" if n == 1 else "2plus_prior"
    )
    output["admission_order_bucket"] = output["matched_admission_order"].map(
        admission_order_bucket
    )
    return output


def admission_order_bucket(order: int) -> str:
    """Return a compact display bucket for admission order."""
    if order <= 5:
        return str(order)
    if order <= 10:
        return "6-10"
    return "11+"


def split_pipe_values(value: object) -> list[str]:
    """Split pipe-delimited keyword group/term strings."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def build_admission_keyword_summary(
    notes: pd.DataFrame,
    section_hits: pd.DataFrame,
    hit_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per current matched admission with SL keyword counts."""
    id_columns = [
        "cohort",
        "subject_id",
        "hadm_id",
        "pair_id",
        "matched_role",
        "admittime",
        "matched_admission_order",
        "n_matched_admissions_for_subject",
        "n_prior_matched_admissions",
        "prior_admission_bucket",
        "admission_order_bucket",
    ]
    admission_summary = notes.loc[:, id_columns].copy()

    if section_hits.empty:
        admission_summary["any_selected_SL_keyword"] = False
        admission_summary["n_sections_with_SL_keyword"] = 0
        admission_summary["total_SL_keyword_hits"] = 0
        admission_summary["keyword_groups"] = ""
        admission_summary["matched_terms"] = ""
        for keyword_group in ["adamant", "compliance", "other"]:
            admission_summary[f"has_{keyword_group}_keyword"] = False
        return admission_summary

    hit_summary = (
        section_hits.groupby(["cohort", "subject_id", "hadm_id"], as_index=False)
        .agg(
            n_sections_with_SL_keyword=("section_name", "nunique"),
            total_SL_keyword_hits=("n_keyword_hits", "sum"),
            keyword_groups=("keyword_groups", lambda s: " | ".join(sorted(set().union(*map(split_pipe_values, s))))),
            matched_terms=("matched_terms", lambda s: " | ".join(sorted(set().union(*map(split_pipe_values, s))))),
        )
    )
    admission_summary = admission_summary.merge(
        hit_summary,
        on=["cohort", "subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )
    admission_summary["any_selected_SL_keyword"] = admission_summary["n_sections_with_SL_keyword"].notna()
    admission_summary["n_sections_with_SL_keyword"] = (
        admission_summary["n_sections_with_SL_keyword"].fillna(0).astype(int)
    )
    admission_summary["total_SL_keyword_hits"] = (
        admission_summary["total_SL_keyword_hits"].fillna(0).astype(int)
    )
    admission_summary["keyword_groups"] = admission_summary["keyword_groups"].fillna("")
    admission_summary["matched_terms"] = admission_summary["matched_terms"].fillna("")

    keyword_groups = sorted(hit_rows["keyword_group"].dropna().astype(str).unique())
    admission_groups = (
        hit_rows.loc[:, ["cohort", "subject_id", "hadm_id", "keyword_group"]]
        .drop_duplicates()
        .assign(has_group=True)
        .pivot_table(
            index=["cohort", "subject_id", "hadm_id"],
            columns="keyword_group",
            values="has_group",
            aggfunc="any",
            fill_value=False,
        )
        .reset_index()
    )
    admission_summary = admission_summary.merge(
        admission_groups,
        on=["cohort", "subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )
    for keyword_group in keyword_groups:
        admission_summary[keyword_group] = admission_summary[keyword_group].fillna(False).astype(bool)
        admission_summary.rename(
            columns={keyword_group: f"has_{keyword_group}_keyword"},
            inplace=True,
        )
    return admission_summary


def summarize_by_group(
    admission_summary: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Summarize any SL keyword rate and hit burden by cohort/order grouping."""
    return (
        admission_summary.groupby(group_columns, as_index=False, dropna=False)
        .agg(
            n_admissions=("hadm_id", "nunique"),
            n_subjects=("subject_id", "nunique"),
            n_admissions_with_any_SL_keyword=("any_selected_SL_keyword", "sum"),
            pct_admissions_with_any_SL_keyword=("any_selected_SL_keyword", lambda s: 100.0 * s.mean()),
            total_SL_keyword_hits=("total_SL_keyword_hits", "sum"),
            mean_SL_keyword_hits_per_admission=("total_SL_keyword_hits", "mean"),
            median_SL_keyword_hits_per_admission=("total_SL_keyword_hits", "median"),
            mean_sections_with_SL_keyword_per_admission=("n_sections_with_SL_keyword", "mean"),
            median_sections_with_SL_keyword_per_admission=("n_sections_with_SL_keyword", "median"),
        )
        .sort_values(group_columns)
    )


def build_keyword_group_summary(
    admission_summary: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Summarize keyword-group presence by cohort/order grouping."""
    rows = []
    keyword_columns = [
        column
        for column in admission_summary.columns
        if column.startswith("has_") and column.endswith("_keyword")
    ]
    for keys, group_df in admission_summary.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys, strict=True))
        for column in keyword_columns:
            keyword_group = column.removeprefix("has_").removesuffix("_keyword")
            rows.append(
                {
                    **base,
                    "keyword_group": keyword_group,
                    "n_admissions": int(group_df["hadm_id"].nunique()),
                    "n_admissions_with_keyword_group": int(group_df[column].sum()),
                    "pct_admissions_with_keyword_group": 100.0 * float(group_df[column].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(group_columns + ["keyword_group"])


def build_term_summary(hit_rows: pd.DataFrame, notes_with_order: pd.DataFrame) -> pd.DataFrame:
    """Summarize matched SL terms by cohort and prior-admission bucket."""
    if hit_rows.empty:
        return pd.DataFrame()
    ordered_hits = hit_rows.merge(
        notes_with_order.loc[
            :,
            [
                "cohort",
                "subject_id",
                "hadm_id",
                "matched_admission_order",
                "prior_admission_bucket",
                "admission_order_bucket",
            ],
        ],
        on=["cohort", "subject_id", "hadm_id"],
        how="left",
        validate="many_to_one",
    )
    return (
        ordered_hits.groupby(
            ["cohort", "prior_admission_bucket", "keyword_group", "matched_term"],
            as_index=False,
        )
        .agg(
            n_hits=("matched_term", "size"),
            n_admissions_with_term=("hadm_id", "nunique"),
            n_subjects_with_term=("subject_id", "nunique"),
        )
        .sort_values(["cohort", "prior_admission_bucket", "n_hits"], ascending=[True, True, False])
    )


def build_qc(
    matched_ids: pd.DataFrame,
    notes: pd.DataFrame,
    section_hits: pd.DataFrame,
) -> pd.DataFrame:
    """Build basic coverage checks for the readmission-order analysis."""
    expected_keys = set(map(tuple, matched_ids[["cohort", "subject_id", "hadm_id"]].to_numpy()))
    note_keys = set(map(tuple, notes[["cohort", "subject_id", "hadm_id"]].to_numpy()))
    return pd.DataFrame(
        [
            {
                "metric": "current_matched_admissions",
                "value": len(expected_keys),
            },
            {
                "metric": "current_matched_admissions_with_parsed_note",
                "value": len(note_keys),
            },
            {
                "metric": "current_matched_admissions_missing_parsed_note",
                "value": len(expected_keys - note_keys),
            },
            {
                "metric": "note_section_rows_with_SL_keyword",
                "value": int(len(section_hits)),
            },
            {
                "metric": "admissions_with_any_SL_keyword",
                "value": int(section_hits["hadm_id"].nunique()) if not section_hits.empty else 0,
            },
        ]
    )


def write_outputs(
    admission_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    order_summary: pd.DataFrame,
    prior_summary: pd.DataFrame,
    prior_group_summary: pd.DataFrame,
    term_summary: pd.DataFrame,
    qc: pd.DataFrame,
) -> None:
    """Write analysis outputs as CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    admission_summary.to_csv(
        OUTPUT_DIR / "SL_keyword_readmission_order_admission_summary.csv",
        index=False,
    )
    overall_summary.to_csv(
        OUTPUT_DIR / "SL_keyword_readmission_order_overall_summary.csv",
        index=False,
    )
    order_summary.to_csv(
        OUTPUT_DIR / "SL_keyword_readmission_order_bucket_summary.csv",
        index=False,
    )
    prior_summary.to_csv(
        OUTPUT_DIR / "SL_keyword_prior_admission_bucket_summary.csv",
        index=False,
    )
    prior_group_summary.to_csv(
        OUTPUT_DIR / "SL_keyword_prior_admission_group_summary.csv",
        index=False,
    )
    term_summary.to_csv(
        OUTPUT_DIR / "SL_keyword_readmission_order_term_summary.csv",
        index=False,
    )
    qc.to_csv(OUTPUT_DIR / "SL_keyword_readmission_order_qc.csv", index=False)


def main() -> None:
    """Run the readmission-order SL keyword analysis."""
    sl_module = load_sl_module()
    parser = sl_module.load_parser_module()
    section_columns = sl_module.selected_section_columns(parser)
    compiled_patterns = sl_module.compile_keyword_patterns()

    matched_ids = load_matched_ids()
    notes = load_current_matched_notes(matched_ids, section_columns)
    notes_with_order = add_admission_order(notes)
    section_hits, hit_rows = sl_module.scan_sections(
        notes_with_order,
        section_columns,
        compiled_patterns,
    )
    admission_summary = build_admission_keyword_summary(notes_with_order, section_hits, hit_rows)
    overall_summary = summarize_by_group(admission_summary, ["cohort"])
    order_summary = summarize_by_group(
        admission_summary,
        ["cohort", "admission_order_bucket"],
    )
    prior_summary = summarize_by_group(
        admission_summary,
        ["cohort", "prior_admission_bucket"],
    )
    prior_group_summary = build_keyword_group_summary(
        admission_summary,
        ["cohort", "prior_admission_bucket"],
    )
    term_summary = build_term_summary(hit_rows, notes_with_order)
    qc = build_qc(matched_ids, notes_with_order, section_hits)

    write_outputs(
        admission_summary,
        overall_summary,
        order_summary,
        prior_summary,
        prior_group_summary,
        term_summary,
        qc,
    )

    print(f"Analyzed {len(admission_summary)} current matched admissions.")
    print(f"Selected sections: {', '.join(section_columns)}")
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(prior_summary.to_string(index=False))


if __name__ == "__main__":
    main()
