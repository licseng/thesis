"""Compare psych-history classifier outputs from two prompt/run folders.

This script compares section-level labels and admission-level positive coverage
from two psych-history classifier output folders. For admission-level overlap,
an admission is positive if at least one compared section is positive.

The script reads classifier result CSVs only. It does not read section text.
Evidence spans and reasons can contain note-derived text, so treat discordance
outputs as local audit files.

Examples:
    python 03_analyze_cluster_output_label_overlap.py --left prompt_A --right prompt_B
    python 03_analyze_cluster_output_label_overlap.py --left-path path/to/A --right-path path/to/B

Outputs:
    <comparison root>/overlap_analysis_output/<left>_vs_<right>/
        prompt_output_pair_overlap_summary.csv
        prompt_output_pair_section_label_crosstabs.csv
        prompt_output_pair_section_label_comparison.csv
        prompt_output_pair_discordant_sections.csv
        prompt_output_pair_label_discordant_sections.csv
        prompt_output_pair_admission_label_comparison.csv
        prompt_output_pair_discordant_admissions.csv
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import re

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_FILENAME = "psych_history_section_classifier_results.csv"
FIRST_N_SECTIONS: int | None = None
DEFAULT_LEFT_OUTPUT = "prompt_A"
DEFAULT_RIGHT_OUTPUT = "prompt_B"

OUTPUT_FOLDERS = {
    "prompt_A": SCRIPT_DIR / "psych_history_classifier_output_prompt_A",
    "prompt_B": SCRIPT_DIR / "psych_history_classifier_output_prompt_B",
    "integrated_100": (
        SCRIPT_DIR
        / "psych_history_classifier_cluster_outputs_psych_integrated"
        / "psych_history_classifier_output"
    ),
    "mention_1a": (
        SCRIPT_DIR
        / "psych_history_classifier_cluster_outputs_psych_mention"
        / "psych_history_classifier_output_1a"
    ),
    "mention_1b": (
        SCRIPT_DIR
        / "psych_history_classifier_cluster_outputs_psych_mention"
        / "psych_history_classifier_output_1b"
    ),
    "oldest_prompt": (
        SCRIPT_DIR
        / "psych_history_classifier_cluster_outputs_psych_mention"
        / "psych_history_classifier_output-100test-oldest_prompt"
    ),
}

KEY_COLUMNS = [
    "classifier_row_id",
    "cohort",
    "subject_id",
    "hadm_id",
    "note_id",
    "section_name",
]
ADMISSION_KEY_COLUMNS = ["cohort", "subject_id", "hadm_id"]
AUDIT_COLUMNS = ["evidence_span", "reason"]
NEW_LABEL_COLUMNS = ["psychiatric_context_label", "psychiatric_mention_type"]
OLD_LABEL_COLUMNS = [
    "label",
    "psychosis_related_context_label",
    "other_psychiatric_context_label",
]


def parse_args() -> Namespace:
    """Parse one pairwise output comparison."""
    parser = ArgumentParser(
        description="Compare labels from two psych-history classifier output folders."
    )
    parser.add_argument(
        "--left",
        default=DEFAULT_LEFT_OUTPUT,
        choices=sorted(OUTPUT_FOLDERS),
        help="Named left output folder.",
    )
    parser.add_argument(
        "--right",
        default=DEFAULT_RIGHT_OUTPUT,
        choices=sorted(OUTPUT_FOLDERS),
        help="Named right output folder.",
    )
    parser.add_argument(
        "--left-path",
        type=Path,
        default=None,
        help="Explicit left output folder path. Overrides --left.",
    )
    parser.add_argument(
        "--right-path",
        type=Path,
        default=None,
        help="Explicit right output folder path. Overrides --right.",
    )
    parser.add_argument(
        "--left-name",
        default=None,
        help="Display name for --left-path.",
    )
    parser.add_argument(
        "--right-name",
        default=None,
        help="Display name for --right-path.",
    )
    parser.add_argument(
        "--first-n-sections",
        type=int,
        default=FIRST_N_SECTIONS,
        help="Optional number of section rows to compare from each output CSV.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    """Return a filesystem-safe comparison name component."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "output"


def comparison_name(left_name: str, right_name: str) -> str:
    """Return a filesystem-safe comparison label."""
    return f"{safe_name(left_name)}_vs_{safe_name(right_name)}"


def resolve_output(name: str, path: Path | None, explicit_name: str | None) -> tuple[str, Path]:
    """Resolve a named or explicit output folder."""
    if path is not None:
        resolved_path = path.expanduser().resolve()
        return explicit_name or resolved_path.name, resolved_path
    return name, OUTPUT_FOLDERS[name]


def comparison_output_dir(left_path: Path, right_path: Path, left_name: str, right_name: str) -> Path:
    """Return the comparison output directory."""
    common_parent = left_path.parent if left_path.parent == right_path.parent else SCRIPT_DIR
    return (
        common_parent
        / "overlap_analysis_output"
        / comparison_name(left_name, right_name)
    )


def result_path(output_path: Path) -> Path:
    """Return the section-level classifier CSV path for one output folder."""
    path = output_path / RESULTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing classifier result CSV: {path}")
    return path


def detect_label_columns(header: pd.Index) -> list[str]:
    """Return comparable label columns available in this output."""
    label_columns = [column for column in NEW_LABEL_COLUMNS if column in header]
    if label_columns:
        return label_columns
    return [column for column in OLD_LABEL_COLUMNS if column in header]


def load_sections(output_name: str, output_path: Path, n_sections: int | None) -> tuple[pd.DataFrame, list[str]]:
    """Load section-level rows without section text."""
    path = result_path(output_path)
    header = pd.read_csv(path, nrows=0).columns
    missing_keys = sorted(set(KEY_COLUMNS) - set(header))
    if missing_keys:
        raise ValueError(f"{path} is missing required key columns: {missing_keys}")

    label_columns = detect_label_columns(header)
    if not label_columns:
        raise ValueError(f"{path} has no recognized label columns.")

    read_columns = KEY_COLUMNS + label_columns + [
        column for column in AUDIT_COLUMNS if column in header
    ]
    df = pd.read_csv(path, usecols=read_columns)
    if n_sections is not None:
        df = df.head(n_sections).copy()
    else:
        df = df.copy()

    for column in label_columns:
        df[column] = df[column].fillna("").astype(str).str.strip().str.lower()
    for column in AUDIT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()

    df.insert(0, "source_output", output_name)
    return df, label_columns


def choose_primary_label(label_columns: list[str]) -> str:
    """Return the label column used for admission-level positivity."""
    if "psychiatric_context_label" in label_columns:
        return "psychiatric_context_label"
    if "label" in label_columns:
        return "label"
    return label_columns[0]


def common_label_columns(left_columns: list[str], right_columns: list[str]) -> list[str]:
    """Return label columns shared by both outputs."""
    return [column for column in left_columns if column in set(right_columns)]


def build_section_comparison(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    label_columns: list[str],
) -> pd.DataFrame:
    """Merge two section-level outputs on section keys."""
    merged = left.merge(
        right,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=(f"_{left_name}", f"_{right_name}"),
        indicator=True,
        validate="one_to_one",
    )
    shared = merged.loc[merged["_merge"].eq("both")].copy()
    if shared.empty:
        return shared

    for label_column in label_columns:
        shared[f"{label_column}_agrees"] = shared[f"{label_column}_{left_name}"].eq(
            shared[f"{label_column}_{right_name}"]
        )
    shared.insert(0, "comparison", comparison_name(left_name, right_name))
    return shared


def build_section_crosstabs(
    section_comparison: pd.DataFrame,
    left_name: str,
    right_name: str,
    label_columns: list[str],
) -> pd.DataFrame:
    """Build crosstabs for shared section-level labels."""
    if section_comparison.empty:
        return pd.DataFrame()

    rows = []
    for label_column in label_columns:
        left_column = f"{label_column}_{left_name}"
        right_column = f"{label_column}_{right_name}"
        crosstab = pd.crosstab(
            section_comparison[left_column],
            section_comparison[right_column],
            dropna=False,
        )
        for left_label, crosstab_row in crosstab.iterrows():
            for right_label, n_sections in crosstab_row.items():
                rows.append(
                    {
                        "comparison": comparison_name(left_name, right_name),
                        "label_column": label_column,
                        "left_label": left_label,
                        "right_label": right_label,
                        "n_sections": int(n_sections),
                        "pct_shared_sections": (
                            100.0 * n_sections / len(section_comparison)
                            if len(section_comparison)
                            else 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_discordant_sections(
    section_comparison: pd.DataFrame,
    label_columns: list[str],
) -> pd.DataFrame:
    """Return shared sections where at least one compared label differs."""
    if section_comparison.empty:
        return section_comparison

    agreement_columns = [f"{label_column}_agrees" for label_column in label_columns]
    discordant = section_comparison.loc[
        ~section_comparison.loc[:, agreement_columns].all(axis=1)
    ].copy()
    if discordant.empty:
        return discordant

    discordant["discordant_label_fields"] = discordant.apply(
        lambda row: " | ".join(
            label_column
            for label_column in label_columns
            if not bool(row[f"{label_column}_agrees"])
        ),
        axis=1,
    )
    return discordant


def build_primary_label_discordant_sections(
    section_comparison: pd.DataFrame,
    primary_label_column: str,
) -> pd.DataFrame:
    """Return shared sections where the primary positive/negative label differs."""
    agreement_column = f"{primary_label_column}_agrees"
    if section_comparison.empty or agreement_column not in section_comparison.columns:
        return pd.DataFrame()
    discordant = section_comparison.loc[~section_comparison[agreement_column]].copy()
    if not discordant.empty:
        discordant["discordant_label_fields"] = primary_label_column
    return discordant


def build_admission_labels(
    sections: pd.DataFrame,
    primary_label_column: str,
) -> pd.DataFrame:
    """Collapse section labels to admission labels using any-positive logic."""
    return (
        sections.assign(
            is_positive=sections[primary_label_column].eq("positive"),
        )
        .groupby(ADMISSION_KEY_COLUMNS, as_index=False)
        .agg(
            n_sections_classified=("section_name", "size"),
            n_positive_sections=("is_positive", "sum"),
            admission_positive=("is_positive", "any"),
        )
    )


def build_admission_comparison(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    left_primary_label: str,
    right_primary_label: str,
) -> pd.DataFrame:
    """Compare admission-level any-positive labels."""
    left_admissions = build_admission_labels(left, left_primary_label)
    right_admissions = build_admission_labels(right, right_primary_label)
    merged = left_admissions.merge(
        right_admissions,
        on=ADMISSION_KEY_COLUMNS,
        how="outer",
        suffixes=(f"_{left_name}", f"_{right_name}"),
        indicator=True,
        validate="one_to_one",
    )
    for column in [
        f"admission_positive_{left_name}",
        f"admission_positive_{right_name}",
    ]:
        merged[column] = merged[column].fillna(False).astype(bool)
    merged["admission_label_agrees"] = merged[
        f"admission_positive_{left_name}"
    ].eq(merged[f"admission_positive_{right_name}"])
    merged.insert(0, "comparison", comparison_name(left_name, right_name))
    return merged


def build_summary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    section_comparison: pd.DataFrame,
    admission_comparison: pd.DataFrame,
    left_name: str,
    right_name: str,
    label_columns: list[str],
) -> pd.DataFrame:
    """Build one-row aggregate overlap summary."""
    shared_sections = len(section_comparison)
    shared_admissions = int(admission_comparison["_merge"].eq("both").sum())
    summary = {
        "comparison": comparison_name(left_name, right_name),
        "left_output": left_name,
        "right_output": right_name,
        "left_section_rows": len(left),
        "right_section_rows": len(right),
        "shared_sections": shared_sections,
        "left_only_sections": int(
            left.merge(right[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator=True)[
                "_merge"
            ].eq("left_only").sum()
        ),
        "right_only_sections": int(
            right.merge(left[KEY_COLUMNS], on=KEY_COLUMNS, how="left", indicator=True)[
                "_merge"
            ].eq("left_only").sum()
        ),
        "left_admissions": int(left[ADMISSION_KEY_COLUMNS].drop_duplicates().shape[0]),
        "right_admissions": int(right[ADMISSION_KEY_COLUMNS].drop_duplicates().shape[0]),
        "shared_admissions": shared_admissions,
        "left_positive_admissions": int(
            admission_comparison[f"admission_positive_{left_name}"].sum()
        ),
        "right_positive_admissions": int(
            admission_comparison[f"admission_positive_{right_name}"].sum()
        ),
        "admission_label_n_agree": int(admission_comparison["admission_label_agrees"].sum()),
        "admission_label_pct_agree": (
            100.0
            * admission_comparison.loc[
                admission_comparison["_merge"].eq("both"),
                "admission_label_agrees",
            ].sum()
            / shared_admissions
            if shared_admissions
            else 0.0
        ),
    }

    for label_column in label_columns:
        agreement_column = f"{label_column}_agrees"
        n_agree = (
            int(section_comparison[agreement_column].sum())
            if agreement_column in section_comparison.columns
            else 0
        )
        summary[f"{label_column}_section_n_agree"] = n_agree
        summary[f"{label_column}_section_pct_agree"] = (
            100.0 * n_agree / shared_sections if shared_sections else 0.0
        )

    return pd.DataFrame([summary])


def write_outputs(
    destination: Path,
    summary: pd.DataFrame,
    section_crosstabs: pd.DataFrame,
    section_comparison: pd.DataFrame,
    discordant_sections: pd.DataFrame,
    primary_label_discordant_sections: pd.DataFrame,
    admission_comparison: pd.DataFrame,
    discordant_admissions: pd.DataFrame,
) -> None:
    """Write overlap outputs."""
    destination.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination / "prompt_output_pair_overlap_summary.csv", index=False)
    section_crosstabs.to_csv(
        destination / "prompt_output_pair_section_label_crosstabs.csv",
        index=False,
    )
    section_comparison.to_csv(
        destination / "prompt_output_pair_section_label_comparison.csv",
        index=False,
    )
    discordant_sections.to_csv(
        destination / "prompt_output_pair_discordant_sections.csv",
        index=False,
    )
    primary_label_discordant_sections.to_csv(
        destination / "prompt_output_pair_label_discordant_sections.csv",
        index=False,
    )
    admission_comparison.to_csv(
        destination / "prompt_output_pair_admission_label_comparison.csv",
        index=False,
    )
    discordant_admissions.to_csv(
        destination / "prompt_output_pair_discordant_admissions.csv",
        index=False,
    )


def main() -> None:
    """Run one pairwise comparison across classifier outputs."""
    args = parse_args()
    left_name, left_path = resolve_output(args.left, args.left_path, args.left_name)
    right_name, right_path = resolve_output(args.right, args.right_path, args.right_name)
    if safe_name(left_name) == safe_name(right_name):
        left_name = f"{left_name}_left"
        right_name = f"{right_name}_right"

    left, left_label_columns = load_sections(left_name, left_path, args.first_n_sections)
    right, right_label_columns = load_sections(right_name, right_path, args.first_n_sections)
    label_columns = common_label_columns(left_label_columns, right_label_columns)
    if not label_columns:
        raise ValueError(
            f"No shared label columns between {left_name} ({left_label_columns}) "
            f"and {right_name} ({right_label_columns})."
        )

    left_primary_label = choose_primary_label(left_label_columns)
    right_primary_label = choose_primary_label(right_label_columns)
    section_comparison = build_section_comparison(
        left,
        right,
        left_name,
        right_name,
        label_columns,
    )
    section_crosstabs = build_section_crosstabs(
        section_comparison,
        left_name,
        right_name,
        label_columns,
    )
    discordant_sections = build_discordant_sections(section_comparison, label_columns)
    primary_label_column = (
        "psychiatric_context_label"
        if "psychiatric_context_label" in label_columns
        else label_columns[0]
    )
    primary_label_discordant_sections = build_primary_label_discordant_sections(
        section_comparison,
        primary_label_column,
    )
    admission_comparison = build_admission_comparison(
        left,
        right,
        left_name,
        right_name,
        left_primary_label,
        right_primary_label,
    )
    discordant_admissions = admission_comparison.loc[
        admission_comparison["_merge"].eq("both")
        & ~admission_comparison["admission_label_agrees"]
    ].copy()
    summary = build_summary(
        left,
        right,
        section_comparison,
        admission_comparison,
        left_name,
        right_name,
        label_columns,
    )
    destination = comparison_output_dir(left_path, right_path, left_name, right_name)
    write_outputs(
        destination,
        summary,
        section_crosstabs,
        section_comparison,
        discordant_sections,
        primary_label_discordant_sections,
        admission_comparison,
        discordant_admissions,
    )

    n_sections = args.first_n_sections if args.first_n_sections is not None else "all"
    print(f"Compared {n_sections} sections for {comparison_name(left_name, right_name)}.")
    print(f"Saved overlap outputs to: {destination}")
    print(f"Discordant shared sections: {len(discordant_sections)}")
    print(f"Primary-label discordant shared sections: {len(primary_label_discordant_sections)}")
    print(f"Discordant shared admissions: {len(discordant_admissions)}")
    print("\n=== Pair Overlap Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
