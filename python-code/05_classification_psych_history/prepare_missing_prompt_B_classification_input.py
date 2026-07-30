"""Prepare classifier input rows missing from the old prompt-B output.

The current matched cohort is a subset/change of the earlier MHH1 cohort, so
most prompt-B classifier outputs can be reused. This helper finds current
prefilter rows that are absent from the old prompt-B full output by matching on
subject_id + hadm_id + section_name, then writes a small parquet input for
classifying only those missing sections.

Inputs:
    psych_history_llm_input/filtered_psych_keyword_section_input.parquet
    psych_history_classifier_cluster_outputs_psych_integrated/
        psych_history_classifier_output_prompt_B_all/
        psych_history_section_classifier_results.parquet

Outputs:
    psych_history_llm_input_missing_prompt_B/
        current_cohort_rows_to_classify_prompt_B.parquet
    psych_history_llm_input_missing_prompt_B/
        current_cohort_rows_to_classify_prompt_B_metadata.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CURRENT_INPUT_PATH = (
    SCRIPT_DIR
    / "psych_history_llm_input"
    / "filtered_psych_keyword_section_input.parquet"
)
OLD_PROMPT_B_OUTPUT_PATH = (
    SCRIPT_DIR
    / "psych_history_classifier_cluster_outputs_psych_integrated"
    / "psych_history_classifier_output_prompt_B_all"
    / "psych_history_section_classifier_results.parquet"
)
OUTPUT_DIR = SCRIPT_DIR / "psych_history_llm_input_missing_prompt_B"
OUTPUT_PARQUET = OUTPUT_DIR / "current_cohort_rows_to_classify_prompt_B.parquet"
OUTPUT_METADATA_CSV = OUTPUT_DIR / "current_cohort_rows_to_classify_prompt_B_metadata.csv"

KEY_COLUMNS = ["subject_id", "hadm_id", "section_name"]
TEXT_COLUMNS = {"section_text"}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load current prefilter rows and old prompt-B classifier output."""
    if not CURRENT_INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing current input: {CURRENT_INPUT_PATH}")
    if not OLD_PROMPT_B_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Missing old prompt-B output: {OLD_PROMPT_B_OUTPUT_PATH}")

    current = pd.read_parquet(CURRENT_INPUT_PATH)
    old_output = pd.read_parquet(OLD_PROMPT_B_OUTPUT_PATH)

    for path, df in (
        (CURRENT_INPUT_PATH, current),
        (OLD_PROMPT_B_OUTPUT_PATH, old_output),
    ):
        missing = sorted(set(KEY_COLUMNS) - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing key columns: {missing}")

    return current, old_output


def find_missing_rows(current: pd.DataFrame, old_output: pd.DataFrame) -> pd.DataFrame:
    """Return current rows without an old prompt-B result for the same section."""
    old_keys = old_output.loc[:, KEY_COLUMNS].drop_duplicates()
    joined = current.merge(
        old_keys.assign(_already_classified=True),
        on=KEY_COLUMNS,
        how="left",
    )
    missing = joined.loc[~joined["_already_classified"].fillna(False)].copy()
    return missing.drop(columns=["_already_classified"])


def write_outputs(missing_rows: pd.DataFrame) -> None:
    """Write classifier parquet plus a metadata CSV without section text."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing_rows.to_parquet(OUTPUT_PARQUET, index=False)

    metadata_columns = [
        column for column in missing_rows.columns if column not in TEXT_COLUMNS
    ]
    missing_rows.loc[:, metadata_columns].to_csv(OUTPUT_METADATA_CSV, index=False)


def main() -> None:
    """Build the missing-section classifier input and print aggregate counts."""
    current, old_output = load_inputs()
    missing_rows = find_missing_rows(current, old_output)
    write_outputs(missing_rows)

    print(f"Current prefilter rows: {len(current):,}", flush=True)
    print(
        "Old prompt-B classified section keys: "
        f"{old_output[KEY_COLUMNS].drop_duplicates().shape[0]:,}",
        flush=True,
    )
    print(f"Missing current rows to classify: {len(missing_rows):,}", flush=True)
    print(
        "Missing current admissions: "
        f"{missing_rows['hadm_id'].nunique() if not missing_rows.empty else 0:,}",
        flush=True,
    )
    if not missing_rows.empty:
        print("\nMissing rows by section:", flush=True)
        print(missing_rows["section_name"].value_counts().to_string(), flush=True)

    print(f"\nSaved classifier input parquet to: {OUTPUT_PARQUET}", flush=True)
    print(f"Saved metadata CSV to: {OUTPUT_METADATA_CSV}", flush=True)


if __name__ == "__main__":
    main()
