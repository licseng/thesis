"""Admission-level characterization of the matched cohorts.

This script summarizes admission-level descriptors and utilization for the
matched MHH1_psychotic and MHC0 admissions. It uses the DBeaver-created DuckDB
tables or file exports handled by `_matched_cohort_characterization_common.py`.
"""

from __future__ import annotations

import pandas as pd

import _matched_cohort_characterization_common as common


def main() -> None:
    """Write admission-level matched-cohort characterization outputs."""
    matched_ids = common.load_expected_matched_ids()
    descriptors = common.add_derived_descriptor_columns(
        common.validate_id_columns(
            common.load_required_table("descriptors"),
            "descriptors",
        )
    )
    event_tables = {
        "labevents": common.load_optional_table("labevents"),
        "microbiologyevents": common.load_optional_table("microbiologyevents"),
        "poe": common.load_optional_table("poe"),
        "poe_detail": common.load_optional_table("poe_detail"),
    }

    descriptor_completeness = common.build_descriptor_completeness(
        matched_ids,
        descriptors,
    )
    categorical_distribution = common.build_categorical_distribution(descriptors)
    categorical_balance = common.build_categorical_balance(categorical_distribution)
    utilization_counts = common.build_event_counts_by_admission(
        matched_ids,
        event_tables,
    )
    utilization_summary = common.build_utilization_summary(utilization_counts)
    optional_category_distribution = pd.concat(
        [
            common.build_optional_category_distribution(
                event_tables["poe"],
                "poe",
                ["order_type", "order_subtype", "transaction_type"],
            ),
            common.build_optional_category_distribution(
                event_tables["poe_detail"],
                "poe_detail",
                ["field_name", "field_value"],
            ),
            common.build_optional_category_distribution(
                event_tables["microbiologyevents"],
                "microbiologyevents",
                ["spec_type_desc", "test_name", "org_name"],
            ),
        ],
        ignore_index=True,
    )

    common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    descriptor_completeness.to_csv(
        common.OUTPUT_DIR / "matched_cohort_descriptor_completeness.csv",
        index=False,
    )
    categorical_distribution.to_csv(
        common.OUTPUT_DIR / "matched_cohort_categorical_distribution.csv",
        index=False,
    )
    categorical_balance.to_csv(
        common.OUTPUT_DIR / "matched_cohort_categorical_balance.csv",
        index=False,
    )
    utilization_counts.to_csv(
        common.OUTPUT_DIR / "matched_cohort_utilization_counts_by_admission.csv",
        index=False,
    )
    utilization_summary.to_csv(
        common.OUTPUT_DIR / "matched_cohort_utilization_summary.csv",
        index=False,
    )
    optional_category_distribution.to_csv(
        common.OUTPUT_DIR / "matched_cohort_optional_category_distribution.csv",
        index=False,
    )

    print(f"Saved admission-level characterization outputs to: {common.OUTPUT_DIR}")
    print("\n=== Descriptor Completeness ===")
    print(descriptor_completeness.to_string(index=False))
    print("\n=== Admission-Level Utilization Summary ===")
    print(utilization_summary.to_string(index=False))


if __name__ == "__main__":
    main()
