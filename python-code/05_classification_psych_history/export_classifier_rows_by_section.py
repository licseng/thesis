"""Export classifier output rows for one discharge-note section.

This is a focused inspection helper for classifier outputs. By default it
exports all final classifier CSV rows for discharge_condition from the completed
prompt B run. Set PSYCH_HISTORY_EXPORT_SECTION_NAME to another section name,
such as discharge_disposition.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CLASSIFIER_OUTPUT_DIR = (
    SCRIPT_DIR
    / "psych_history_classifier_cluster_outputs_psych_integrated"
    / "psych_history_classifier_output_prompt_B_all"
)
CLASSIFIER_OUTPUT_DIR = Path(
    os.environ.get(
        "PSYCH_HISTORY_EXPORT_INPUT_DIR",
        str(DEFAULT_CLASSIFIER_OUTPUT_DIR),
    )
)
SECTION_NAME = os.environ.get(
    "PSYCH_HISTORY_EXPORT_SECTION_NAME",
    "discharge_condition",
)
OUTPUT_DIR = Path(
    os.environ.get(
        "PSYCH_HISTORY_EXPORT_OUTPUT_DIR",
        str(CLASSIFIER_OUTPUT_DIR / "section_row_exports"),
    )
)

INPUT_CSV = CLASSIFIER_OUTPUT_DIR / "psych_history_section_classifier_results.csv"


def main() -> None:
    """Write classifier rows for the selected section to a separate CSV."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing classifier results CSV: {INPUT_CSV}")

    results = pd.read_csv(INPUT_CSV)
    if "section_name" not in results.columns:
        raise ValueError("Classifier results CSV has no section_name column.")

    section_rows = results.loc[results["section_name"].eq(SECTION_NAME)].copy()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{SECTION_NAME}_classifier_rows.csv"
    section_rows.to_csv(output_path, index=False)

    print(f"Input: {INPUT_CSV}")
    print(f"Section: {SECTION_NAME}")
    print(f"Rows exported: {len(section_rows)}")
    print(f"Output: {output_path}")
    if "psychiatric_context_label" in section_rows.columns:
        print("\nLabel counts:")
        print(
            section_rows["psychiatric_context_label"]
            .value_counts(dropna=False)
            .to_string()
        )
    if "psychiatric_mention_type" in section_rows.columns:
        print("\nMention type counts:")
        print(
            section_rows["psychiatric_mention_type"]
            .value_counts(dropna=False)
            .to_string()
        )


if __name__ == "__main__":
    main()
