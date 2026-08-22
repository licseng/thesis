"""Analyze discharge-note language complexity by matched cohort.

This script computes transparent readability/complexity proxies for matched
MHH1_psychotic and MHC0 full discharge notes. It uses the parser's
`full_note_text` column, which preserves the original note text loaded from the
source table and includes chief complaint. It writes numeric metrics only; no
raw note text is written to output files.

Metrics are descriptive proxies. Clinical notes contain abbreviations, lists,
medication names, lab values, and deidentified placeholders, so school-grade
readability formulas should be interpreted cautiously.

The selected prose-section outputs use flattened parsed section text for:
present_illness, brief_hospital_course, and discharge_instructions. These are
intended for readability-style metrics, not line-format/list-structure metrics.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_PYTHON_DIR = SCRIPT_DIR.parent
PARSER_DIR = REPO_PYTHON_DIR / "01_discharge_note_preprocessing" / "01_discharge_note_parsing"
FULL_NOTE_DIR = PARSER_DIR / "full_discharge_note_sections"
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_language_complexity"
REGRESSION_DATASET_PATH = (
    REPO_PYTHON_DIR
    / "05_regression_analysis"
    / "analysis_output_utilization_readmission_adjusted"
    / "utilization_readmission_adjusted_model_dataset.csv"
)

FULL_NOTE_FILES = [
    {
        "cohort": "MHH1_psychotic",
        "path": FULL_NOTE_DIR / "MHH1_psychotic_matched_full_discharge_note_sections.parquet",
    },
    {
        "cohort": "MHC0",
        "path": FULL_NOTE_DIR / "MHC0_matched_full_discharge_note_sections.parquet",
    },
]

ID_COLUMNS = ["cohort", "subject_id", "hadm_id", "note_id", "charttime"]
SELECTED_PROSE_SECTION_COLUMNS = [
    "present_illness",
    "brief_hospital_course",
    "discharge_instructions",
]
SUMMARY_METRICS = [
    "n_characters",
    "n_whitespace_words",
    "n_tokens",
    "n_alpha_words",
    "n_sentences",
    "n_nonempty_lines",
    "mean_words_per_sentence",
    "mean_words_per_line",
    "pct_short_lines",
    "mean_characters_per_alpha_word",
    "pct_long_alpha_words",
    "pct_complex_alpha_words",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog_index",
    "smog_index",
    "automated_readability_index",
]
REGRESSION_METRICS = [
    "mean_words_per_sentence",
    "mean_words_per_line",
    "pct_short_lines",
    "mean_characters_per_alpha_word",
    "pct_long_alpha_words",
    "pct_complex_alpha_words",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog_index",
    "smog_index",
    "automated_readability_index",
]

WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?")
ALPHA_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def clean_text(value: object) -> str:
    """Return text as a string, with nulls mapped to empty string."""
    if pd.isna(value):
        return ""
    return str(value)


def sentence_units(text: str) -> list[str]:
    """Split clinical text into rough sentence/list units."""
    stripped = text.strip()
    if not stripped:
        return []
    units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?])\s+|\n{2,}", stripped)
        if unit.strip()
    ]
    if len(units) <= 1 and "\n" in stripped:
        units = [line.strip() for line in stripped.splitlines() if line.strip()]
    return units or [stripped]


def count_syllables(word: str) -> int:
    """Approximate English syllable count for readability formulas."""
    cleaned = re.sub(r"[^A-Za-z]", "", word).lower()
    if not cleaned:
        return 0
    groups = VOWEL_GROUP_RE.findall(cleaned)
    n_syllables = len(groups)
    if cleaned.endswith("e") and n_syllables > 1 and not cleaned.endswith(("le", "ye")):
        n_syllables -= 1
    return max(1, n_syllables)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return zero for empty denominators."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_complexity_metrics(text: str) -> dict[str, Any]:
    """Compute numeric note/section complexity metrics."""
    text = clean_text(text)
    whitespace_words = text.strip().split()
    tokens = WORD_RE.findall(text)
    alpha_words = ALPHA_WORD_RE.findall(text)
    sentences = sentence_units(text)
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_word_counts = [len(line.split()) for line in nonempty_lines]
    n_whitespace_words = len(whitespace_words)
    n_tokens = len(tokens)
    n_alpha_words = len(alpha_words)
    n_sentences = len(sentences)
    n_nonempty_lines = len(nonempty_lines)
    n_characters = len(text)
    alpha_characters = sum(len(re.sub(r"[^A-Za-z]", "", word)) for word in alpha_words)
    syllables = [count_syllables(word) for word in alpha_words]
    n_syllables = sum(syllables)
    n_long_alpha_words = sum(
        1 for word in alpha_words if len(re.sub(r"[^A-Za-z]", "", word)) >= 7
    )
    n_complex_alpha_words = sum(1 for syllable_count in syllables if syllable_count >= 3)

    words_per_sentence = safe_divide(n_tokens, n_sentences)
    words_per_line = safe_divide(n_whitespace_words, n_nonempty_lines)
    pct_short_lines = 100.0 * safe_divide(
        sum(1 for count in line_word_counts if count <= 5),
        n_nonempty_lines,
    )
    syllables_per_alpha_word = safe_divide(n_syllables, n_alpha_words)
    characters_per_alpha_word = safe_divide(alpha_characters, n_alpha_words)
    pct_long_alpha_words = 100.0 * safe_divide(n_long_alpha_words, n_alpha_words)
    pct_complex_alpha_words = 100.0 * safe_divide(n_complex_alpha_words, n_alpha_words)

    if n_tokens == 0 or n_alpha_words == 0 or n_sentences == 0:
        flesch = 0.0
        flesch_kincaid = 0.0
        gunning_fog = 0.0
        smog = 0.0
        ari = 0.0
    else:
        flesch = 206.835 - (1.015 * words_per_sentence) - (
            84.6 * syllables_per_alpha_word
        )
        flesch_kincaid = (0.39 * words_per_sentence) + (
            11.8 * syllables_per_alpha_word
        ) - 15.59
        gunning_fog = 0.4 * (words_per_sentence + pct_complex_alpha_words)
        smog = 1.043 * ((n_complex_alpha_words * 30.0 / n_sentences) ** 0.5) + 3.1291
        ari = (4.71 * characters_per_alpha_word) + (0.5 * words_per_sentence) - 21.43

    return {
        "n_characters": n_characters,
        "n_whitespace_words": n_whitespace_words,
        "n_tokens": n_tokens,
        "n_alpha_words": n_alpha_words,
        "n_sentences": n_sentences,
        "n_nonempty_lines": n_nonempty_lines,
        "n_syllables": n_syllables,
        "n_long_alpha_words": n_long_alpha_words,
        "n_complex_alpha_words": n_complex_alpha_words,
        "mean_words_per_sentence": words_per_sentence,
        "mean_words_per_line": words_per_line,
        "pct_short_lines": pct_short_lines,
        "mean_characters_per_alpha_word": characters_per_alpha_word,
        "mean_syllables_per_alpha_word": syllables_per_alpha_word,
        "pct_long_alpha_words": pct_long_alpha_words,
        "pct_complex_alpha_words": pct_complex_alpha_words,
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": flesch_kincaid,
        "gunning_fog_index": gunning_fog,
        "smog_index": smog,
        "automated_readability_index": ari,
    }


def load_notes() -> pd.DataFrame:
    """Load parsed matched discharge-note tables."""
    frames = []
    for config in FULL_NOTE_FILES:
        path = config["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing parsed discharge-note sections: {path}")
        columns = [
            column
            for column in [*ID_COLUMNS, "full_note_text", *SELECTED_PROSE_SECTION_COLUMNS]
            if column != "cohort"
        ]
        table = pd.read_parquet(path, columns=columns)
        table.insert(0, "cohort", config["cohort"])
        frames.append(table)
    notes = pd.concat(frames, ignore_index=True)
    notes["charttime"] = pd.to_datetime(notes["charttime"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    return notes


def build_note_level_metrics(notes: pd.DataFrame) -> pd.DataFrame:
    """Compute full-note complexity metrics, one row per admission/note."""
    rows = []
    for _, row in notes.iterrows():
        metrics = compute_complexity_metrics(row["full_note_text"])
        rows.append(
            {
                "cohort": row["cohort"],
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"],
                "note_id": row["note_id"],
                "charttime": row["charttime"],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_prose_section_metrics(notes: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for selected prose-like parsed sections."""
    rows = []
    for _, row in notes.iterrows():
        for section_name in SELECTED_PROSE_SECTION_COLUMNS:
            text = clean_text(row.get(section_name, ""))
            if not text.strip():
                continue
            metrics = compute_complexity_metrics(text)
            rows.append(
                {
                    "cohort": row["cohort"],
                    "subject_id": row["subject_id"],
                    "hadm_id": row["hadm_id"],
                    "note_id": row["note_id"],
                    "charttime": row["charttime"],
                    "section_name": section_name,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarize complexity metrics for selected grouping columns."""
    rows = []
    groupby_key: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_values, group in df.groupby(groupby_key, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_columns, group_values, strict=True))
        row["n_rows"] = len(group)
        row["n_admissions"] = group["hadm_id"].nunique()
        row["n_subjects"] = group["subject_id"].nunique()
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_q1"] = values.quantile(0.25)
            row[f"{metric}_q3"] = values.quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def load_regression_covariates() -> pd.DataFrame:
    """Load matched-cohort covariates used by the regression-analysis scripts."""
    if not REGRESSION_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Missing regression covariate dataset. Run "
            "05_regression_analysis/01_analyze_utilization_adjusted_for_readmission.py first: "
            f"{REGRESSION_DATASET_PATH}"
        )

    columns = [
        "cohort",
        "subject_id",
        "hadm_id",
        "pair_id",
        "age_at_admission",
        "elixhauser_score",
        "n_prior_admissions_within_365d_for_subject",
        "n_prior_all_admissions_for_subject",
        "log1p_prior_admissions_365d",
        "log1p_prior_all_mimic_admissions",
        "cluster_id",
    ]
    covariates = pd.read_csv(REGRESSION_DATASET_PATH, usecols=columns)
    covariates["subject_id"] = covariates["subject_id"].astype(str)
    covariates["hadm_id"] = covariates["hadm_id"].astype(str)
    covariates["mhh1_psychotic"] = covariates["cohort"].eq("MHH1_psychotic").astype(int)
    covariates["age_at_admission_per_10y"] = (
        pd.to_numeric(covariates["age_at_admission"], errors="coerce") / 10.0
    )
    covariates["elixhauser_score_per_5pt"] = (
        pd.to_numeric(covariates["elixhauser_score"], errors="coerce") / 5.0
    )
    return covariates


def add_regression_covariates(metrics: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """Attach matched-cohort covariates to language-complexity metrics."""
    output = metrics.copy()
    output["subject_id"] = output["subject_id"].astype(str)
    output["hadm_id"] = output["hadm_id"].astype(str)
    return output.merge(
        covariates,
        on=["cohort", "subject_id", "hadm_id"],
        how="left",
        validate="many_to_one",
    )


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Return Benjamini-Hochberg adjusted p-values."""
    p_numeric = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_numeric.dropna()
    if valid.empty:
        return adjusted

    ordered = valid.sort_values()
    n_tests = len(ordered)
    ranks = np.arange(1, n_tests + 1, dtype=float)
    raw_adjusted = ordered.to_numpy() * n_tests / ranks
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted.loc[ordered.index] = np.minimum(monotone, 1.0)
    return adjusted


def fit_ols_cluster_robust(
    data: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    model_name: str,
    strata: dict[str, str],
) -> dict[str, Any] | None:
    """Fit one OLS model with subject-clustered robust standard errors."""
    required_columns = [outcome, *predictors, "cluster_id"]
    model_data = data.dropna(subset=required_columns).copy()
    if len(model_data) < 20 or model_data["mhh1_psychotic"].nunique() < 2:
        return None

    y = pd.to_numeric(model_data[outcome], errors="coerce")
    x = model_data.loc[:, predictors].apply(pd.to_numeric, errors="coerce")
    complete = y.notna() & x.notna().all(axis=1) & model_data["cluster_id"].notna()
    model_data = model_data.loc[complete].copy()
    y = y.loc[complete]
    x = sm.add_constant(x.loc[complete], has_constant="add")
    if len(model_data) < 20 or model_data["mhh1_psychotic"].nunique() < 2:
        return None

    fit = sm.OLS(y, x).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_data["cluster_id"]},
    )
    term = "mhh1_psychotic"
    ci_low, ci_high = fit.conf_int().loc[term].tolist()
    return {
        **strata,
        "outcome": outcome,
        "model": model_name,
        "n_rows": int(fit.nobs),
        "n_subject_clusters": int(model_data["cluster_id"].nunique()),
        "mhh1_coefficient": fit.params[term],
        "mhh1_ci_low": ci_low,
        "mhh1_ci_high": ci_high,
        "mhh1_p_value": fit.pvalues[term],
        "fit_method": "statsmodels_ols_cluster_robust_by_subject",
        "predictors": " + ".join(predictors),
    }


def fit_language_complexity_regressions(
    note_metrics: pd.DataFrame,
    prose_section_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit cohort-difference regressions for note and prose-section metrics."""
    covariates = load_regression_covariates()
    note_data = add_regression_covariates(note_metrics, covariates)
    section_data = add_regression_covariates(prose_section_metrics, covariates)

    model_specs = {
        "unadjusted": ["mhh1_psychotic"],
        "age_elixhauser": [
            "mhh1_psychotic",
            "age_at_admission_per_10y",
            "elixhauser_score_per_5pt",
        ],
        "age_elixhauser_prior365": [
            "mhh1_psychotic",
            "age_at_admission_per_10y",
            "elixhauser_score_per_5pt",
            "log1p_prior_admissions_365d",
        ],
    }

    note_rows = []
    section_rows = []
    for outcome in REGRESSION_METRICS:
        for model_name, predictors in model_specs.items():
            row = fit_ols_cluster_robust(
                note_data,
                outcome,
                predictors,
                model_name,
                {"analysis_level": "full_note"},
            )
            if row is not None:
                note_rows.append(row)

            for section_name, section_group in section_data.groupby("section_name"):
                section_row = fit_ols_cluster_robust(
                    section_group,
                    outcome,
                    predictors,
                    model_name,
                    {
                        "analysis_level": "prose_section",
                        "section_name": section_name,
                    },
                )
                if section_row is not None:
                    section_rows.append(section_row)

    note_results = pd.DataFrame(note_rows)
    section_results = pd.DataFrame(section_rows)
    for result in [note_results, section_results]:
        if not result.empty:
            result["mhh1_p_value_fdr_bh"] = result.groupby("model")[
                "mhh1_p_value"
            ].transform(benjamini_hochberg)

    return note_results, section_results


def write_outputs(
    note_metrics: pd.DataFrame,
    note_summary: pd.DataFrame,
    prose_section_metrics: pd.DataFrame,
    prose_section_summary: pd.DataFrame,
    note_regression_results: pd.DataFrame,
    section_regression_results: pd.DataFrame,
) -> None:
    """Write complexity metric outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    note_metrics.to_csv(OUTPUT_DIR / "language_complexity_note_level_metrics.csv", index=False)
    note_summary.to_csv(OUTPUT_DIR / "language_complexity_note_summary.csv", index=False)
    prose_section_metrics.to_csv(
        OUTPUT_DIR / "language_complexity_prose_section_metrics.csv",
        index=False,
    )
    prose_section_summary.to_csv(
        OUTPUT_DIR / "language_complexity_prose_section_summary.csv",
        index=False,
    )
    note_regression_results.to_csv(
        OUTPUT_DIR / "language_complexity_note_level_regression_results.csv",
        index=False,
    )
    section_regression_results.to_csv(
        OUTPUT_DIR / "language_complexity_prose_section_regression_results.csv",
        index=False,
    )


def main() -> None:
    """Run language-complexity analysis for matched discharge notes."""
    notes = load_notes()
    note_metrics = build_note_level_metrics(notes)
    prose_section_metrics = build_prose_section_metrics(notes)
    note_summary = summarize_metrics(note_metrics, ["cohort"])
    prose_section_summary = summarize_metrics(
        prose_section_metrics,
        ["cohort", "section_name"],
    )
    note_regression_results, section_regression_results = (
        fit_language_complexity_regressions(note_metrics, prose_section_metrics)
    )

    write_outputs(
        note_metrics,
        note_summary,
        prose_section_metrics,
        prose_section_summary,
        note_regression_results,
        section_regression_results,
    )

    print(f"Scanned {len(notes)} matched discharge notes.")
    print(f"Saved language-complexity outputs to: {OUTPUT_DIR}")
    print("\n=== Note-Level Summary ===")
    display_columns = [
        "cohort",
        "n_rows",
        "n_admissions",
        "n_whitespace_words_median",
        "mean_words_per_sentence_median",
        "mean_words_per_line_median",
        "pct_short_lines_median",
        "flesch_kincaid_grade_median",
        "gunning_fog_index_median",
    ]
    print(note_summary.loc[:, display_columns].to_string(index=False))
    print("\n=== Selected Prose-Section Summary ===")
    section_display_columns = [
        "cohort",
        "section_name",
        "n_rows",
        "n_admissions",
        "n_whitespace_words_median",
        "mean_words_per_sentence_median",
        "flesch_kincaid_grade_median",
        "gunning_fog_index_median",
    ]
    print(
        prose_section_summary.loc[:, section_display_columns].to_string(index=False)
    )
    print("\n=== Note-Level Regression Results: MHH1 term ===")
    if note_regression_results.empty:
        print("No note-level models were fit.")
    else:
        display_regression_columns = [
            "outcome",
            "model",
            "mhh1_coefficient",
            "mhh1_ci_low",
            "mhh1_ci_high",
            "mhh1_p_value",
            "mhh1_p_value_fdr_bh",
        ]
        print(note_regression_results.loc[:, display_regression_columns].to_string(index=False))


if __name__ == "__main__":
    main()
