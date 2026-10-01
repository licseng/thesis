"""Admission-level characterization of the matched cohorts.

This script summarizes admission-level descriptors and utilization for the
matched MHC1_psychotic and MHC0 admissions. It uses the DBeaver-created DuckDB
tables or file exports handled by `_matched_cohort_characterization_common.py`.
"""

from __future__ import annotations

import pandas as pd

import _matched_cohort_characterization_common as common


def main() -> None:
    """Write admission-level matched-cohort characterization outputs."""
    output_dir = common.ADMISSION_LEVEL_OUTPUT_DIR
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
    psychosis_context = common.load_required_table("psychosis_context")
    diagnosis_history = common.load_required_table("subject_diagnosis_history")
    admission_history = common.load_required_table("subject_admission_history")

    descriptor_completeness = common.build_descriptor_completeness(
        matched_ids,
        descriptors,
    )
    numeric_demographic_summary = common.build_matched_numeric_demographic_summary()
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
    psychosis_context_summary = common.build_psychosis_context_summary(
        psychosis_context
    )
    (
        psychiatric_comorbidity_summary,
        psychiatric_comorbidity_context_summary,
        psychiatric_comorbidity_count_distribution,
    ) = common.build_matched_MHC1_comorbidity_outputs(
        psychosis_context,
        diagnosis_history,
    )
    future_readmission_summary = common.build_future_readmission_summary(
        matched_ids,
        descriptors,
        admission_history,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_completeness.to_csv(
        output_dir / "matched_cohort_descriptor_completeness.csv",
        index=False,
    )
    numeric_demographic_summary.to_csv(
        output_dir / "matched_cohort_numeric_demographic_summary.csv",
        index=False,
    )
    categorical_distribution.to_csv(
        output_dir / "matched_cohort_categorical_distribution.csv",
        index=False,
    )
    categorical_balance.to_csv(
        output_dir / "matched_cohort_categorical_balance.csv",
        index=False,
    )
    utilization_counts.to_csv(
        output_dir / "matched_cohort_utilization_counts_by_admission.csv",
        index=False,
    )
    utilization_summary.to_csv(
        output_dir / "matched_cohort_utilization_summary.csv",
        index=False,
    )
    optional_category_distribution.to_csv(
        output_dir / "matched_cohort_optional_category_distribution.csv",
        index=False,
    )
    psychosis_context_summary.to_csv(
        output_dir / "matched_MHC1_psychosis_context_summary.csv",
        index=False,
    )
    psychiatric_comorbidity_summary.to_csv(
        output_dir / "matched_MHC1_psychiatric_comorbidity_summary.csv",
        index=False,
    )
    psychiatric_comorbidity_context_summary.to_csv(
        output_dir / "matched_MHC1_psychiatric_comorbidity_context_summary.csv",
        index=False,
    )
    psychiatric_comorbidity_count_distribution.to_csv(
        output_dir
        / "matched_MHC1_psychiatric_comorbidity_count_distribution.csv",
        index=False,
    )
    future_readmission_summary.to_csv(
        output_dir / "matched_cohort_future_readmission_summary.csv",
        index=False,
    )

    print(f"Saved admission-level characterization outputs to: {output_dir}")
    print("\n=== Descriptor Completeness ===")
    print(descriptor_completeness.to_string(index=False))
    print("\n=== Numeric Demographics ===")
    print(numeric_demographic_summary.to_string(index=False))
    print("\n=== Admission-Level Utilization Summary ===")
    print(utilization_summary.to_string(index=False))
    print("\n=== Matched MHC1 Psychosis Context ===")
    print(psychosis_context_summary.to_string(index=False))
    print("\n=== Matched MHC1 Psychiatric Comorbidity ===")
    print(psychiatric_comorbidity_summary.to_string(index=False))
    print("\n=== Future Readmission Summary ===")
    print(future_readmission_summary.to_string(index=False))


if __name__ == "__main__":
    main()
