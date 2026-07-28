"""Subject-level characterization of the matched cohorts.

This script summarizes repeated admissions, subject-level descriptor categories,
and subject-level utilization for the matched MHH1_psychotic and MHC0 cohorts.
Admission-level categorical values that vary within a subject are collapsed to
`multiple_values`.
"""

from __future__ import annotations

import _matched_cohort_characterization_common as common


def main() -> None:
    """Write subject-level matched-cohort characterization outputs."""
    output_dir = common.SUBJECT_LEVEL_OUTPUT_DIR
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

    admissions_per_subject_summary = common.build_admissions_per_subject_summary(
        matched_ids,
    )
    subject_categorical_distribution = common.build_subject_categorical_distribution(
        descriptors,
    )
    subject_categorical_balance = common.build_subject_categorical_balance(
        subject_categorical_distribution,
    )
    utilization_counts = common.build_event_counts_by_admission(
        matched_ids,
        event_tables,
    )
    subject_utilization_counts = common.build_subject_utilization_counts(
        utilization_counts,
    )
    subject_utilization_summary = common.build_subject_utilization_summary(
        subject_utilization_counts,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    admissions_per_subject_summary.to_csv(
        output_dir / "matched_cohort_admissions_per_subject_summary.csv",
        index=False,
    )
    subject_categorical_distribution.to_csv(
        output_dir / "matched_cohort_subject_categorical_distribution.csv",
        index=False,
    )
    subject_categorical_balance.to_csv(
        output_dir / "matched_cohort_subject_categorical_balance.csv",
        index=False,
    )
    subject_utilization_counts.to_csv(
        output_dir / "matched_cohort_subject_utilization_counts.csv",
        index=False,
    )
    subject_utilization_summary.to_csv(
        output_dir / "matched_cohort_subject_utilization_summary.csv",
        index=False,
    )

    print(f"Saved subject-level characterization outputs to: {output_dir}")
    print("\n=== Admissions Per Subject Summary ===")
    print(admissions_per_subject_summary.to_string(index=False))
    print("\n=== Subject-Level Utilization Summary ===")
    print(subject_utilization_summary.to_string(index=False))


if __name__ == "__main__":
    main()
