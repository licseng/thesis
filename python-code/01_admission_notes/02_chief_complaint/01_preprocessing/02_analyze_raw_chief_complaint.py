"""Analyze raw chief-complaint parquet exports.

This script is a quality-control step before chief-complaint
preprocessing. 

Inputs:
    chief_complaint_parquets/MHH1_psychotic_chief_complaints.parquet
    chief_complaint_parquets/MHC0_chief_complaints.parquet

Outputs:
    analysis_output_chief_complaint_raw folder
"""

from __future__ import annotations
from pathlib import Path
import re
import pandas as pd


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "chief_complaint_parquets"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_chief_complaint_raw"

# Input parquet files produced by the chief-complaint export step.
INPUTS = {
    "MHH1_psychotic": INPUT_DIR / "MHH1_psychotic_chief_complaints.parquet",
    "MHC0": INPUT_DIR / "MHC0_chief_complaints.parquet",
}

# Analysis limits for local review CSVs.
TOP_N_RAW_COMPLAINTS = 100
TOP_N_WORDS = 100
TOP_N_BIGRAMS = 100

# Required minimal schema of the chief-complaint parquet exports.
REQUIRED_COLUMNS = {
    "subject_id",
    "hadm_id",
    "chief_complaint",
}

# Lightweight patterns for identifying values that are technically non-empty
# but not usable clinical chief complaints, such as "___" or "CC: ___".
PLACEHOLDER_RE = re.compile(r"\b_+\b|_+")
SPACING_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s/+-]")
TOKEN_RE = re.compile(r"[a-z][a-z]+")

MISSING_CHIEF_COMPLAINT_VALUES = {
    "",
    "none",
    "nan",
    "null",
    "na",
    "n/a",
    "unknown",
    "?",
}

PREFIX_PATTERNS = [
    re.compile(r"^\s*(?:cc|chief complaint)\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*(?:admit(?:ted)?\s+for|admission\s+for)\s+", re.IGNORECASE),
    re.compile(r"^\s*(?:s/p|status post)\s+", re.IGNORECASE),
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


# Load one raw chief-complaint parquet and validate that the expected columns
# are present before analysis.
def load_chief_complaints(path: Path, cohort: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing chief-complaint parquet for {cohort}: {path}")

    df = pd.read_parquet(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df.insert(0, "cohort", cohort)
    df["chief_complaint"] = df["chief_complaint"].fillna("").astype(str)
    return df


# Load all configured raw chief-complaint parquet files into one dataframe.
def load_all_chief_complaints() -> pd.DataFrame:
    frames = [load_chief_complaints(path, cohort) for cohort, path in INPUTS.items()]
    return pd.concat(frames, ignore_index=True)


# Remove common labels/prefixes that can make a placeholder look non-empty.
def strip_prefixes(value: str) -> str:
    for pattern in PREFIX_PATTERNS:
        value = pattern.sub("", value)
    return value


# Normalize only enough to identify non-informative placeholder values. This is
# not the full preprocessing normalization step.
def placeholder_screen_text(value: str) -> str:
    value = PLACEHOLDER_RE.sub(" ", value)
    value = strip_prefixes(value)
    value = value.lower()
    value = PUNCT_RE.sub(" ", value)
    value = SPACING_RE.sub(" ", value).strip()
    return value


# Return True when a raw chief complaint is empty or placeholder-like after a
# minimal cleanup screen.
def is_placeholder_like(value: str) -> bool:
    screened = placeholder_screen_text(value)
    return screened in MISSING_CHIEF_COMPLAINT_VALUES


# Add reusable raw-text quality-control columns used by all summaries.
def add_qc_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    stripped = output["chief_complaint"].str.strip()
    output["has_raw_chief_complaint"] = stripped.ne("")
    output["is_placeholder_like"] = output["chief_complaint"].map(is_placeholder_like)
    output["usable_for_preprocessing_screen"] = (
        output["has_raw_chief_complaint"] & ~output["is_placeholder_like"]
    )
    output["chief_complaint_chars"] = stripped.str.len()
    output["chief_complaint_words"] = stripped.map(lambda value: len(value.split()))
    return output


# Safely calculate percentages while avoiding division by zero.
def pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


# Summarize raw chief-complaint availability and placeholder-like values.
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, group in df.groupby("cohort", sort=True):
        total = len(group)
        raw_nonempty = int(group["has_raw_chief_complaint"].sum())
        placeholder_like = int(
            (group["has_raw_chief_complaint"] & group["is_placeholder_like"]).sum()
        )
        usable = int(group["usable_for_preprocessing_screen"].sum())

        rows.append(
            {
                "cohort": cohort,
                "n_rows": total,
                "n_subjects": group["subject_id"].nunique(),
                "n_admissions": group["hadm_id"].nunique(),
                "n_raw_nonempty_chief_complaint": raw_nonempty,
                "pct_raw_nonempty_chief_complaint": pct(raw_nonempty, total),
                "n_placeholder_like_chief_complaint": placeholder_like,
                "pct_placeholder_like_among_raw_nonempty": pct(
                    placeholder_like,
                    raw_nonempty,
                ),
                "n_usable_for_preprocessing_screen": usable,
                "pct_usable_for_preprocessing_screen": pct(usable, total),
            }
        )
    return pd.DataFrame(rows)


# Summarize raw chief-complaint character and word lengths among values that pass
# the placeholder screen.
def build_length_summary(df: pd.DataFrame) -> pd.DataFrame:
    usable = df.loc[df["usable_for_preprocessing_screen"]].copy()
    rows = []
    for cohort, group in usable.groupby("cohort", sort=True):
        rows.append(
            {
                "cohort": cohort,
                "n_usable_for_preprocessing_screen": len(group),
                "mean_chief_complaint_chars": group["chief_complaint_chars"].mean(),
                "median_chief_complaint_chars": group["chief_complaint_chars"].median(),
                "q1_chief_complaint_chars": group["chief_complaint_chars"].quantile(0.25),
                "q3_chief_complaint_chars": group["chief_complaint_chars"].quantile(0.75),
                "mean_chief_complaint_words": group["chief_complaint_words"].mean(),
                "median_chief_complaint_words": group["chief_complaint_words"].median(),
                "q1_chief_complaint_words": group["chief_complaint_words"].quantile(0.25),
                "q3_chief_complaint_words": group["chief_complaint_words"].quantile(0.75),
            }
        )
    return pd.DataFrame(rows)


# Count exact raw chief-complaint strings. This output is local-review material
# and is not printed to the terminal.
def build_common_raw_complaints(df: pd.DataFrame) -> pd.DataFrame:
    usable = df.loc[df["usable_for_preprocessing_screen"]].copy()
    rows = []
    for cohort, group in usable.groupby("cohort", sort=True):
        denomin = len(group)
        counts = (
            group.groupby("chief_complaint", dropna=False)
            .agg(
                n_occurrences=("hadm_id", "size"),
                n_admissions=("hadm_id", "nunique"),
            )
            .reset_index()
            .sort_values(["n_occurrences", "n_admissions", "chief_complaint"], ascending=[False, False, True])
            .head(TOP_N_RAW_COMPLAINTS)
        )
        counts.insert(0, "cohort", cohort)
        counts["pct_usable_admissions"] = counts["n_admissions"].map(
            lambda value: pct(value, denomin)
        )
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# Extract simple lowercase word tokens from a chief complaint.
def tokenize(value: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(value.lower())
        if token not in STOPWORDS and token != "___"
    ]


# Count common single-word tokens in usable raw chief complaints.
def build_common_words(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    usable = df.loc[df["usable_for_preprocessing_screen"]].copy()
    for row in usable.itertuples(index=False):
        for token in tokenize(row.chief_complaint):
            records.append(
                {
                    "cohort": row.cohort,
                    "hadm_id": row.hadm_id,
                    "word": token,
                }
            )
    if not records:
        return pd.DataFrame()

    token_df = pd.DataFrame(records)
    rows = []
    denominators = usable.groupby("cohort")["hadm_id"].nunique().to_dict()
    for cohort, group in token_df.groupby("cohort", sort=True):
        counts = (
            group.groupby("word")
            .agg(
                n_occurrences=("word", "size"),
                n_admissions_with_word=("hadm_id", "nunique"),
            )
            .reset_index()
            .sort_values(["n_occurrences", "n_admissions_with_word", "word"], ascending=[False, False, True])
            .head(TOP_N_WORDS)
        )
        counts.insert(0, "cohort", cohort)
        counts["pct_usable_admissions_with_word"] = counts[
            "n_admissions_with_word"
        ].map(lambda value: pct(value, denominators[cohort]))
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


# Count common adjacent token pairs in usable raw chief complaints.
def build_common_bigrams(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    usable = df.loc[df["usable_for_preprocessing_screen"]].copy()
    for row in usable.itertuples(index=False):
        tokens = tokenize(row.chief_complaint)
        for left, right in zip(tokens, tokens[1:]):
            records.append(
                {
                    "cohort": row.cohort,
                    "hadm_id": row.hadm_id,
                    "bigram": f"{left} {right}",
                }
            )
    if not records:
        return pd.DataFrame()

    bigram_df = pd.DataFrame(records)
    rows = []
    denominators = usable.groupby("cohort")["hadm_id"].nunique().to_dict()
    for cohort, group in bigram_df.groupby("cohort", sort=True):
        counts = (
            group.groupby("bigram")
            .agg(
                n_occurrences=("bigram", "size"),
                n_admissions_with_bigram=("hadm_id", "nunique"),
            )
            .reset_index()
            .sort_values(["n_occurrences", "n_admissions_with_bigram", "bigram"], ascending=[False, False, True])
            .head(TOP_N_BIGRAMS)
        )
        counts.insert(0, "cohort", cohort)
        counts["pct_usable_admissions_with_bigram"] = counts[
            "n_admissions_with_bigram"
        ].map(lambda value: pct(value, denominators[cohort]))
        rows.append(counts)
    return pd.concat(rows, ignore_index=True)


# Write raw placeholder-like rows to a local CSV so they can be inspected without
# printing raw values into the terminal.
def write_placeholder_like_rows(df: pd.DataFrame) -> Path:
    output_path = OUTPUT_DIR / "placeholder_like_chief_complaints.csv"
    columns = [
        "cohort",
        "subject_id",
        "hadm_id",
        "chief_complaint",
        "has_raw_chief_complaint",
        "is_placeholder_like",
    ]
    rows = df.loc[df["has_raw_chief_complaint"] & df["is_placeholder_like"], columns]
    rows.to_csv(output_path, index=False)
    return output_path


# Script entry point: load raw chief-complaint exports, build QC summaries, write
# local CSV outputs, and print aggregate summaries only.
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = add_qc_columns(load_all_chief_complaints())

    summary = build_summary(df)
    length_summary = build_length_summary(df)
    common_complaints = build_common_raw_complaints(df)
    common_words = build_common_words(df)
    common_bigrams = build_common_bigrams(df)

    summary_path = OUTPUT_DIR / "raw_chief_complaint_summary.csv"
    length_summary_path = OUTPUT_DIR / "raw_chief_complaint_length_summary.csv"
    common_complaints_path = OUTPUT_DIR / "common_raw_chief_complaints.csv"
    common_words_path = OUTPUT_DIR / "common_raw_words.csv"
    common_bigrams_path = OUTPUT_DIR / "common_raw_bigrams.csv"
    placeholder_path = write_placeholder_like_rows(df)

    summary.to_csv(summary_path, index=False)
    length_summary.to_csv(length_summary_path, index=False)
    common_complaints.to_csv(common_complaints_path, index=False)
    common_words.to_csv(common_words_path, index=False)
    common_bigrams.to_csv(common_bigrams_path, index=False)

    print("\n=== Raw chief complaint summary ===")
    print(summary.to_string(index=False))
    print("\n=== Raw chief complaint length summary ===")
    print(length_summary.to_string(index=False))
    print(f"\nSaved raw chief-complaint analysis CSVs to: {OUTPUT_DIR}")
    print(f"Saved placeholder-like rows to: {placeholder_path}")


# Allow the script to be run directly with:
#     python analyze_raw_chief_complaint.py
if __name__ == "__main__":
    main()
