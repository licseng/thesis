"""Merge existing prompt-B classifier output with missing current-cohort rows.

The current matched cohort changed after the full prompt-B classifier run.
Most current rows are already covered by the older prompt-B output, but a small
set needed to be classified separately. This helper rebuilds a current-cohort
output folder by joining labels on subject_id + hadm_id + section_name and
keeping the current cohort metadata/classifier_row_id values.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_OUTPUT_BASE = SCRIPT_DIR / "psych_history_classifier_cluster_outputs_psych_integrated"

CURRENT_INPUT_PATH = (
    SCRIPT_DIR / "psych_history_llm_input" / "filtered_psych_keyword_section_input.parquet"
)
OLD_OUTPUT_DIR = CLUSTER_OUTPUT_BASE / "psych_history_classifier_output_prompt_B_all"
MISSING_OUTPUT_DIR = CLUSTER_OUTPUT_BASE / "psych_history_classifier_output_prompt_B_missing"
MERGED_OUTPUT_DIR = CLUSTER_OUTPUT_BASE / "psych_history_classifier_output_prompt_B_current_cohort_merged"

KEY_COLUMNS = ["subject_id", "hadm_id", "section_name"]
TEXT_COLUMNS = {"section_text"}
CLASSIFICATION_COLUMNS = [
    "psychiatric_context_label",
    "psychiatric_mention_type",
    "evidence_span",
    "reason",
    "n_chunks",
    "n_positive_chunks",
    "n_negative_chunks",
    "model_name",
    "json_recovered_from_partial_response",
]
CHUNK_CLASSIFICATION_COLUMNS = [
    "chunk_index",
    "n_chunks",
    "chunk_word_count",
    "psychiatric_context_label",
    "psychiatric_mention_type",
    "evidence_span",
    "reason",
    "model_name",
    "json_recovered_from_partial_response",
]


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file with a clear error message."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required parquet: {path}")
    return pd.read_parquet(path)


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize merge-key dtypes so old and current files join reliably."""
    df = df.copy()
    for column in ["subject_id", "hadm_id"]:
        df[column] = pd.to_numeric(df[column], errors="raise").astype("int64")
    df["section_name"] = df["section_name"].astype(str)
    return df


def key_set(df: pd.DataFrame) -> set[tuple[int, int, str]]:
    """Return section-level keys for coverage checks."""
    return set(map(tuple, df[KEY_COLUMNS].to_numpy()))


def prepare_current_metadata(current_input: pd.DataFrame) -> pd.DataFrame:
    """Keep current cohort metadata while excluding raw section text."""
    metadata_columns = [
        column for column in current_input.columns if column not in TEXT_COLUMNS
    ]
    return current_input[metadata_columns].copy()


def combine_section_labels(old_results: pd.DataFrame, missing_results: pd.DataFrame) -> pd.DataFrame:
    """Combine old and missing labels, preferring missing labels on duplicates."""
    required = set(KEY_COLUMNS + CLASSIFICATION_COLUMNS)
    for label, df in [("old", old_results), ("missing", missing_results)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{label} section output is missing column(s): "
                + ", ".join(sorted(missing))
            )

    old_results = old_results.copy()
    missing_results = missing_results.copy()
    old_results["_classification_source"] = "prompt_B_all"
    missing_results["_classification_source"] = "prompt_B_missing"
    combined = pd.concat([missing_results, old_results], ignore_index=True)
    combined = combined.drop_duplicates(KEY_COLUMNS, keep="first")
    return combined[KEY_COLUMNS + CLASSIFICATION_COLUMNS + ["_classification_source"]]


def build_section_results(
    current_metadata: pd.DataFrame,
    old_results: pd.DataFrame,
    missing_results: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Attach prompt-B labels to the current cohort section metadata."""
    combined_labels = combine_section_labels(old_results, missing_results)
    section_results = current_metadata.merge(
        combined_labels,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    missing_label_mask = section_results["psychiatric_context_label"].isna()
    if missing_label_mask.any():
        raise ValueError(
            "Merged section output is still missing classifier labels for "
            f"{int(missing_label_mask.sum())} current rows."
        )

    section_results = section_results.sort_values("classifier_row_id").reset_index(drop=True)
    coverage = {
        "current_input_rows": len(current_metadata),
        "old_output_rows": len(old_results),
        "missing_output_rows": len(missing_results),
        "merged_section_rows": len(section_results),
        "labels_from_old_output": int(
            section_results["_classification_source"].eq("prompt_B_all").sum()
        ),
        "labels_from_missing_output": int(
            section_results["_classification_source"].eq("prompt_B_missing").sum()
        ),
    }
    return section_results, coverage


def build_chunk_results(
    current_metadata: pd.DataFrame,
    old_chunks: pd.DataFrame,
    missing_chunks: pd.DataFrame,
) -> pd.DataFrame:
    """Filter chunk rows to the current cohort and remap classifier_row_id values."""
    required = set(KEY_COLUMNS + CHUNK_CLASSIFICATION_COLUMNS)
    for label, df in [("old", old_chunks), ("missing", missing_chunks)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{label} chunk output is missing column(s): "
                + ", ".join(sorted(missing))
            )

    old_chunks = old_chunks.copy()
    missing_chunks = missing_chunks.copy()
    old_chunks["_classification_source"] = "prompt_B_all"
    missing_chunks["_classification_source"] = "prompt_B_missing"
    combined_chunks = pd.concat([missing_chunks, old_chunks], ignore_index=True)

    current_chunk_metadata = current_metadata[
        [
            "classifier_row_id",
            "cohort",
            "subject_id",
            "hadm_id",
            "note_id",
            "charttime",
            "section_name",
            "n_psych_keyword_hits",
            "psych_keyword_groups",
            "matched_terms",
        ]
    ].copy()
    chunk_results = current_chunk_metadata.merge(
        combined_chunks[KEY_COLUMNS + CHUNK_CLASSIFICATION_COLUMNS + ["_classification_source"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_many",
    )
    missing_chunk_mask = chunk_results["psychiatric_context_label"].isna()
    if missing_chunk_mask.any():
        raise ValueError(
            "Merged chunk output is missing classifier labels for "
            f"{int(missing_chunk_mask.sum())} current section rows."
        )

    return chunk_results.sort_values(["classifier_row_id", "chunk_index"]).reset_index(drop=True)


def build_admission_summary(section_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate current-cohort section labels to admission-level positivity."""
    results = section_results.assign(
        is_positive=section_results["psychiatric_context_label"].eq("positive")
    )
    return (
        results.groupby(["cohort", "subject_id", "hadm_id"], as_index=False)
        .agg(
            n_sections_classified=("classifier_row_id", "size"),
            n_positive_sections=("is_positive", "sum"),
            n_negative_sections=("is_positive", lambda values: int((~values).sum())),
            any_positive=("is_positive", "any"),
        )
        .sort_values(["cohort", "subject_id", "hadm_id"])
    )


def build_label_summary(section_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize section labels and mention types."""
    return (
        section_results.groupby(
            [
                "cohort",
                "section_name",
                "psychiatric_context_label",
                "psychiatric_mention_type",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="n_sections")
        .sort_values(["cohort", "section_name", "psychiatric_context_label", "psychiatric_mention_type"])
    )


def stringify_problem_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid parquet dtype errors from mixed date/object columns."""
    df = df.copy()
    if "charttime" in df.columns:
        df["charttime"] = df["charttime"].astype(str)
    return df


def write_outputs(
    section_results: pd.DataFrame,
    chunk_results: pd.DataFrame,
    admission_summary: pd.DataFrame,
    label_summary: pd.DataFrame,
    coverage: dict[str, int],
) -> None:
    """Write merged classifier outputs in the same style as the classifier script."""
    MERGED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    section_results = stringify_problem_columns(section_results)
    chunk_results = stringify_problem_columns(chunk_results)
    admission_summary = stringify_problem_columns(admission_summary)

    section_results.to_parquet(
        MERGED_OUTPUT_DIR / "psych_history_section_classifier_results.parquet",
        index=False,
    )
    section_results.to_csv(
        MERGED_OUTPUT_DIR / "psych_history_section_classifier_results.csv",
        index=False,
    )
    chunk_results.to_parquet(
        MERGED_OUTPUT_DIR / "psych_history_section_chunk_classifier_results.parquet",
        index=False,
    )
    chunk_results.to_csv(
        MERGED_OUTPUT_DIR / "psych_history_section_chunk_classifier_results.csv",
        index=False,
    )
    with (MERGED_OUTPUT_DIR / "psych_history_section_classifier_results.jsonl").open(
        "w"
    ) as handle:
        for record in section_results.to_dict(orient="records"):
            handle.write(json.dumps(record, default=str) + "\n")

    admission_summary.to_csv(
        MERGED_OUTPUT_DIR / "psych_history_admission_summary.csv",
        index=False,
    )
    label_summary.to_csv(
        MERGED_OUTPUT_DIR / "psych_history_section_label_summary.csv",
        index=False,
    )
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "current_input_path": str(CURRENT_INPUT_PATH),
        "old_output_dir": str(OLD_OUTPUT_DIR),
        "missing_output_dir": str(MISSING_OUTPUT_DIR),
        "merged_output_dir": str(MERGED_OUTPUT_DIR),
        **coverage,
        "merged_chunk_rows": len(chunk_results),
        "merged_admissions": int(admission_summary["hadm_id"].nunique()),
        "positive_admissions": int(admission_summary["any_positive"].sum()),
    }
    with (MERGED_OUTPUT_DIR / "psych_history_run_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    """Run the current-cohort prompt-B merge."""
    current_input = normalize_keys(load_parquet(CURRENT_INPUT_PATH))
    old_results = normalize_keys(
        load_parquet(OLD_OUTPUT_DIR / "psych_history_section_classifier_results.parquet")
    )
    missing_results = normalize_keys(
        load_parquet(MISSING_OUTPUT_DIR / "psych_history_section_classifier_results.parquet")
    )
    old_chunks = normalize_keys(
        load_parquet(OLD_OUTPUT_DIR / "psych_history_section_chunk_classifier_results.parquet")
    )
    missing_chunks = normalize_keys(
        load_parquet(MISSING_OUTPUT_DIR / "psych_history_section_chunk_classifier_results.parquet")
    )

    current_metadata = prepare_current_metadata(current_input)
    current_keys = key_set(current_metadata)
    missing_needed = current_keys - key_set(old_results)
    missing_covered = missing_needed & key_set(missing_results)
    if missing_needed - missing_covered:
        raise ValueError(
            "Missing output does not cover all current rows absent from old output: "
            f"{len(missing_needed - missing_covered)} uncovered."
        )

    section_results, coverage = build_section_results(
        current_metadata=current_metadata,
        old_results=old_results,
        missing_results=missing_results,
    )
    chunk_results = build_chunk_results(current_metadata, old_chunks, missing_chunks)
    admission_summary = build_admission_summary(section_results)
    label_summary = build_label_summary(section_results)
    write_outputs(section_results, chunk_results, admission_summary, label_summary, coverage)

    print(f"Merged current-cohort prompt-B output saved to: {MERGED_OUTPUT_DIR}")
    print(f"Merged section rows: {len(section_results)}")
    print(f"Merged chunk rows: {len(chunk_results)}")
    print(f"Merged admissions: {admission_summary['hadm_id'].nunique()}")
    print(f"Positive admissions: {int(admission_summary['any_positive'].sum())}")
    print(f"Labels from missing output: {coverage['labels_from_missing_output']}")


if __name__ == "__main__":
    main()
