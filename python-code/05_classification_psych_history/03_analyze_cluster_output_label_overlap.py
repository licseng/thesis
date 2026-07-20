"""Compare label overlap across downloaded psych-history cluster outputs.

This script compares section-level labels from two psych-history classifier
output folders. It is intended for prompt/run variance checks, for example:

    python 03_analyze_cluster_output_label_overlap.py --left 1a --right 1b
    python 03_analyze_cluster_output_label_overlap.py --left 1a --right oldest_prompt

Only section metadata, labels, and model audit fields are read. Matched terms
and section text are not loaded or exported. Evidence spans and reasons can
contain note-derived text, so treat the outputs as local audit files.

Outputs:
    overlap_analysis_output/<left>_vs_<right>/
        cluster_output_pair_overlap_summary.csv
        cluster_output_pair_label_crosstabs.csv
        cluster_output_pair_section_label_comparison.csv
        cluster_output_pair_discordant_sections.csv
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_OUTPUT_DIR = SCRIPT_DIR / "psych_history_classifier_cluster_outputs"

RESULTS_FILENAME = "psych_history_section_classifier_results.csv"
FIRST_N_SECTIONS = 70
DEFAULT_LEFT_OUTPUT = "1a"
DEFAULT_RIGHT_OUTPUT = "1b"

OUTPUT_FOLDERS = {
    "1a": CLUSTER_OUTPUT_DIR / "psych_history_classifier_output_1a",
    "1b": CLUSTER_OUTPUT_DIR / "psych_history_classifier_output_1b",
    "oldest_prompt": CLUSTER_OUTPUT_DIR
    / "psych_history_classifier_output-100test-oldest_prompt",
}

KEY_COLUMNS = [
    "classifier_row_id",
    "cohort",
    "subject_id",
    "hadm_id",
    "note_id",
    "section_name",
]

LABEL_COLUMNS = [
    "label",
    "psychosis_related_context_label",
    "other_psychiatric_context_label",
]

AUDIT_COLUMNS = [
    "evidence_span",
    "reason",
]

READ_COLUMNS = KEY_COLUMNS + LABEL_COLUMNS + AUDIT_COLUMNS


def parse_args() -> ArgumentParser:
    """Parse the one comparison to run."""
    parser = ArgumentParser(
        description="Compare labels from one pair of downloaded cluster outputs."
    )
    parser.add_argument(
        "--left",
        default=DEFAULT_LEFT_OUTPUT,
        choices=sorted(OUTPUT_FOLDERS),
        help="Left/primary cluster output label.",
    )
    parser.add_argument(
        "--right",
        default=DEFAULT_RIGHT_OUTPUT,
        choices=sorted(OUTPUT_FOLDERS),
        help="Right/comparison cluster output label.",
    )
    parser.add_argument(
        "--first-n-sections",
        type=int,
        default=FIRST_N_SECTIONS,
        help="Number of section rows to compare from each output CSV.",
    )
    return parser


def comparison_name(left_name: str, right_name: str) -> str:
    """Return a filesystem-safe comparison label."""
    return f"{left_name}_vs_{right_name}"


def output_dir(left_name: str, right_name: str) -> Path:
    """Return the pair-specific overlap output directory."""
    return CLUSTER_OUTPUT_DIR / "overlap_analysis_output" / comparison_name(
        left_name,
        right_name,
    )


def result_path(output_name: str) -> Path:
    """Return the section-level classifier CSV path for one downloaded output."""
    if output_name not in OUTPUT_FOLDERS:
        raise KeyError(f"Unknown output folder label: {output_name}")
    return OUTPUT_FOLDERS[output_name] / RESULTS_FILENAME


def load_first_sections(output_name: str, n_sections: int) -> pd.DataFrame:
    """Load the first n section-level rows without free-text audit columns."""
    path = result_path(output_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing classifier result CSV: {path}")

    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(READ_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = pd.read_csv(path, usecols=READ_COLUMNS).head(n_sections).copy()
    for column in LABEL_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip().str.lower()
    for column in AUDIT_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()
    return df


def compare_outputs(
    left_name: str,
    right_name: str,
    n_sections: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare two outputs on shared section keys and return summary tables."""
    left = load_first_sections(left_name, n_sections)
    right = load_first_sections(right_name, n_sections)

    merged = left.merge(
        right,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=(f"_{left_name}", f"_{right_name}"),
        indicator=True,
        validate="one_to_one",
    )

    shared = merged.loc[merged["_merge"].eq("both")].copy()
    comparison_rows = []
    crosstab_rows = []
    pair_name = comparison_name(left_name, right_name)

    summary = {
        "comparison": pair_name,
        "left_output": left_name,
        "right_output": right_name,
        "left_rows_limited_to": len(left),
        "right_rows_limited_to": len(right),
        "shared_sections": len(shared),
        "left_only_sections": int(merged["_merge"].eq("left_only").sum()),
        "right_only_sections": int(merged["_merge"].eq("right_only").sum()),
    }

    for label_column in LABEL_COLUMNS:
        left_column = f"{label_column}_{left_name}"
        right_column = f"{label_column}_{right_name}"
        agreement_column = f"{label_column}_agrees"
        if shared.empty:
            n_agree = 0
            pct_agree = 0.0
        else:
            shared[agreement_column] = shared[left_column].eq(shared[right_column])
            n_agree = int(shared[agreement_column].sum())
            pct_agree = 100.0 * n_agree / len(shared)

            crosstab = pd.crosstab(
                shared[left_column],
                shared[right_column],
                dropna=False,
            )
            for left_label, row in crosstab.iterrows():
                for right_label, n_sections_compared in row.items():
                    crosstab_rows.append(
                        {
                            "comparison": f"{left_name}_vs_{right_name}",
                            "label_column": label_column,
                            "left_label": left_label,
                            "right_label": right_label,
                            "n_sections": int(n_sections_compared),
                            "pct_shared_sections": (
                                100.0 * n_sections_compared / len(shared)
                                if len(shared)
                                else 0.0
                            ),
                        }
                    )

        summary[f"{label_column}_n_agree"] = n_agree
        summary[f"{label_column}_pct_agree"] = pct_agree

    if not shared.empty:
        comparison_columns = KEY_COLUMNS[:]
        for label_column in LABEL_COLUMNS:
            comparison_columns.extend(
                [
                    f"{label_column}_{left_name}",
                    f"{label_column}_{right_name}",
                    f"{label_column}_agrees",
                ]
            )
        for audit_column in AUDIT_COLUMNS:
            comparison_columns.extend(
                [
                    f"{audit_column}_{left_name}",
                    f"{audit_column}_{right_name}",
                ]
            )
        comparison_rows = shared.loc[:, comparison_columns].copy()
        comparison_rows.insert(0, "comparison", pair_name)
    else:
        comparison_rows = pd.DataFrame()

    return (
        pd.DataFrame([summary]),
        pd.DataFrame(crosstab_rows),
        comparison_rows,
    )


def build_discordant_sections(section_comparison: pd.DataFrame) -> pd.DataFrame:
    """Return shared sections where at least one compared label differs."""
    if section_comparison.empty:
        return section_comparison

    agreement_columns = [
        f"{label_column}_agrees"
        for label_column in LABEL_COLUMNS
        if f"{label_column}_agrees" in section_comparison.columns
    ]
    if not agreement_columns:
        return pd.DataFrame()

    discordant = section_comparison.loc[
        ~section_comparison.loc[:, agreement_columns].all(axis=1)
    ].copy()
    discordant["discordant_label_fields"] = discordant.apply(
        lambda row: " | ".join(
            label_column
            for label_column in LABEL_COLUMNS
            if f"{label_column}_agrees" in row.index
            and not bool(row[f"{label_column}_agrees"])
        ),
        axis=1,
    )
    return discordant


def write_outputs(
    destination: Path,
    summary: pd.DataFrame,
    crosstabs: pd.DataFrame,
    section_comparison: pd.DataFrame,
    discordant_sections: pd.DataFrame,
) -> None:
    """Write aggregate and label-only overlap outputs."""
    destination.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        destination / "cluster_output_pair_overlap_summary.csv",
        index=False,
    )
    crosstabs.to_csv(
        destination / "cluster_output_pair_label_crosstabs.csv",
        index=False,
    )
    section_comparison.to_csv(
        destination / "cluster_output_pair_section_label_comparison.csv",
        index=False,
    )
    discordant_sections.to_csv(
        destination / "cluster_output_pair_discordant_sections.csv",
        index=False,
    )


def main() -> None:
    """Run one pairwise comparison across downloaded cluster outputs."""
    parser = parse_args()
    args = parser.parse_args()

    summary_df, crosstab_df, section_comparison_df = compare_outputs(
        args.left,
        args.right,
        args.first_n_sections,
    )
    discordant_sections_df = build_discordant_sections(section_comparison_df)
    destination = output_dir(args.left, args.right)
    write_outputs(
        destination,
        summary_df,
        crosstab_df,
        section_comparison_df,
        discordant_sections_df,
    )

    print(
        f"Compared first {args.first_n_sections} sections for "
        f"{comparison_name(args.left, args.right)}."
    )
    print(f"Saved overlap outputs to: {destination}")
    print(f"Discordant shared sections: {len(discordant_sections_df)}")
    print("\n=== Pair Overlap Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
