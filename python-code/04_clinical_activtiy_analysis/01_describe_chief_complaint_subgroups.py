"""Describe CC subgroups and select complete same-complaint matched pairs.

This script is separate from clustering. It defines a small set of manually
selected chief-complaint symptom groups and counts how many matched admissions
in each cohort contain those symptoms. A subgroup hit can come from either
normalized chief-complaint text phrases or QuickUMLS terms.

The admission-assignment output keeps IDs and boolean subgroup flags for later
workup/outcome analyses. It does not write raw chief-complaint text.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MATCHED_PAIRS_PATH = (
    PROJECT_DIR / "02_cohort_matching" / "matched_cohort_output" / "matched_pairs.parquet"
)
OUTPUT_DIR = SCRIPT_DIR / "analysis_output_chief_complaint_subgroup_balance_check"

CHIEF_COMPLAINT_SUBGROUPS = {
    "abdominal pain": {
        "text_phrases": [
            "abdominal pain",
            "abd pain",
            "abdominal discomfort",
            "abd discomfort",
            "belly pain",
        ],
        "quickumls_terms": [
            "abdominal pain",
            "abdominal discomfort",
            "abdomen",
            "abdomen pain",
        ],
    },
    "shortness of breath": {
        "text_phrases": [
            "shortness of breath",
            "short of breath",
            "sob",
            "dyspnea",
            "difficulty breathing",
            "trouble breathing",
        ],
        "quickumls_terms": [
            "shortness of breath",
            "dyspnea",
            "breathing difficulty",
            "difficulty breathing",
            "sob",
        ],
    },
    "chest pain": {
        "text_phrases": [
            "chest pain",
            "cp",
            "chest discomfort",
            "chest pressure",
        ],
        "quickumls_terms": [
            "chest pain",
            "chest discomfort",
            "chest pressure",
        ],
    },
    "altered mental status": {
        "text_phrases": [
            "altered mental status",
            "mental status",
            "altered mental",
            "ams",
            "confusion",
            "confused",
        ],
        "quickumls_terms": [
            "altered mental status",
            "mental status",
            "confusion",
            "confused",
            "disorientation",
        ],
    },
    "nausea vomiting": {
        "text_phrases": [
            "nausea vomiting",
            "nausea and vomiting",
            "nausea",
            "vomiting",
            "emesis",
        ],
        "quickumls_terms": [
            "nausea",
            "vomiting",
            "emesis",
        ],
    },
}

COMBINED_CHIEF_COMPLAINT_GROUPS = {
    "abdominal_pain_nausea_vomiting": [
        "abdominal pain",
        "nausea vomiting",
    ],
    "chest_pain_shortness_of_breath": [
        "chest pain",
        "shortness of breath",
    ],
}

OVERLAP_GROUPS = {
    "fever": {
        "text_phrases": [
            "fever",
            "fevers",
            "febrile",
        ],
        "quickumls_terms": [
            "fever",
            "febrile",
        ],
    },
    "altered_mental_status": CHIEF_COMPLAINT_SUBGROUPS["altered mental status"],
}


def contains_token_phrase(text: object, phrase: str) -> bool:
    """Return True when phrase occurs as a contiguous token sequence in text."""
    text_tokens = str(text or "").lower().split()
    phrase_tokens = str(phrase or "").lower().split()
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return False
    width = len(phrase_tokens)
    return any(
        text_tokens[start : start + width] == phrase_tokens
        for start in range(0, len(text_tokens) - width + 1)
    )


def split_pipe_terms(value: object) -> set[str]:
    """Parse pipe-separated QuickUMLS term strings into lowercase terms."""
    if pd.isna(value):
        return set()
    return {
        term.strip().lower()
        for term in str(value).split("|")
        if term.strip()
    }


def has_quickumls_term(value: object, target_terms: list[str]) -> bool:
    """Return whether any target term appears in a QuickUMLS term string."""
    terms = split_pipe_terms(value)
    targets = {term.lower().strip() for term in target_terms if term.strip()}
    return bool(terms & targets)


def load_matched_pairs() -> pd.DataFrame:
    """Load current matched pairs with chief-complaint fields."""
    if not MATCHED_PAIRS_PATH.exists():
        raise FileNotFoundError(f"Missing matched pairs file: {MATCHED_PAIRS_PATH}")
    required_columns = [
        "pair_id",
        "mhc1_subject_id",
        "mhc1_hadm_id",
        "mhc1_chief_complaint_normalized",
        "mhc1_quickumls_terms",
        "mhc1_derived_quickumls_overlap_terms",
        "mhc1_sex",
        "mhc1_insurance_group",
        "mhc1_age_at_admission",
        "mhc1_elixhauser_score",
        "mhc0_subject_id",
        "mhc0_hadm_id",
        "mhc0_chief_complaint_normalized",
        "mhc0_quickumls_terms",
        "mhc0_derived_quickumls_overlap_terms",
        "mhc0_sex",
        "mhc0_insurance_group",
        "mhc0_age_at_admission",
        "mhc0_elixhauser_score",
    ]
    return pd.read_parquet(MATCHED_PAIRS_PATH, columns=required_columns)


def build_admission_level_complaints(matched_pairs: pd.DataFrame) -> pd.DataFrame:
    """Convert matched pairs to one row per admission with normalized complaint."""
    mhc1 = matched_pairs.loc[
        :,
        [
            "pair_id",
            "mhc1_subject_id",
            "mhc1_hadm_id",
            "mhc1_chief_complaint_normalized",
            "mhc1_quickumls_terms",
            "mhc1_derived_quickumls_overlap_terms",
            "mhc1_sex",
            "mhc1_insurance_group",
            "mhc1_age_at_admission",
            "mhc1_elixhauser_score",
        ],
    ].rename(
        columns={
            "mhc1_subject_id": "subject_id",
            "mhc1_hadm_id": "hadm_id",
            "mhc1_chief_complaint_normalized": "chief_complaint_normalized",
            "mhc1_quickumls_terms": "quickumls_terms",
            "mhc1_derived_quickumls_overlap_terms": "derived_quickumls_overlap_terms",
            "mhc1_sex": "sex",
            "mhc1_insurance_group": "insurance_group",
            "mhc1_age_at_admission": "age_at_admission",
            "mhc1_elixhauser_score": "elixhauser_score",
        }
    )
    mhc1["cohort"] = "MHC1_psychotic"

    mhc0 = matched_pairs.loc[
        :,
        [
            "pair_id",
            "mhc0_subject_id",
            "mhc0_hadm_id",
            "mhc0_chief_complaint_normalized",
            "mhc0_quickumls_terms",
            "mhc0_derived_quickumls_overlap_terms",
            "mhc0_sex",
            "mhc0_insurance_group",
            "mhc0_age_at_admission",
            "mhc0_elixhauser_score",
        ],
    ].rename(
        columns={
            "mhc0_subject_id": "subject_id",
            "mhc0_hadm_id": "hadm_id",
            "mhc0_chief_complaint_normalized": "chief_complaint_normalized",
            "mhc0_quickumls_terms": "quickumls_terms",
            "mhc0_derived_quickumls_overlap_terms": "derived_quickumls_overlap_terms",
            "mhc0_sex": "sex",
            "mhc0_insurance_group": "insurance_group",
            "mhc0_age_at_admission": "age_at_admission",
            "mhc0_elixhauser_score": "elixhauser_score",
        }
    )
    mhc0["cohort"] = "MHC0"

    admissions = pd.concat([mhc1, mhc0], ignore_index=True)
    admissions["chief_complaint_normalized"] = (
        admissions["chief_complaint_normalized"].fillna("").astype(str).str.strip()
    )
    admissions["subject_id"] = pd.to_numeric(
        admissions["subject_id"],
        errors="raise",
    ).astype(int)
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="raise").astype(int)
    return admissions.loc[
        :,
        [
            "pair_id",
            "cohort",
            "subject_id",
            "hadm_id",
            "chief_complaint_normalized",
            "quickumls_terms",
            "derived_quickumls_overlap_terms",
            "sex",
            "insurance_group",
            "age_at_admission",
            "elixhauser_score",
        ],
    ]


def subgroup_hit(row: pd.Series, config: dict[str, list[str]]) -> bool:
    """Return whether text or QuickUMLS terms match a symptom subgroup."""
    text_hit = any(
        contains_token_phrase(row["chief_complaint_normalized"], phrase)
        for phrase in config["text_phrases"]
    )
    quickumls_hit = (
        has_quickumls_term(row["quickumls_terms"], config["quickumls_terms"])
        or has_quickumls_term(
            row["derived_quickumls_overlap_terms"],
            config["quickumls_terms"],
        )
    )
    return bool(text_hit or quickumls_hit)


def add_subgroup_flags(admissions: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean flag per hard-coded chief-complaint subgroup."""
    flagged = admissions.copy()
    for subgroup, config in CHIEF_COMPLAINT_SUBGROUPS.items():
        column = f"has_{subgroup.replace(' ', '_')}"
        flagged[column] = flagged.apply(
            lambda row: subgroup_hit(row, config),
            axis=1,
        )

    for group_name, subgroup_names in COMBINED_CHIEF_COMPLAINT_GROUPS.items():
        source_columns = [
            f"has_{subgroup.replace(' ', '_')}" for subgroup in subgroup_names
        ]
        flagged[f"has_{group_name}"] = flagged[source_columns].any(axis=1)

    for group_name, config in OVERLAP_GROUPS.items():
        flagged[f"has_overlap_{group_name}"] = flagged.apply(
            lambda row: subgroup_hit(row, config),
            axis=1,
        )

    flag_columns = [
        f"has_{subgroup.replace(' ', '_')}" for subgroup in CHIEF_COMPLAINT_SUBGROUPS
    ]
    flagged["n_chief_complaint_subgroups_matched"] = flagged[flag_columns].sum(axis=1)
    flagged["matched_chief_complaint_subgroups"] = flagged.apply(
        lambda row: " | ".join(
            subgroup
            for subgroup in CHIEF_COMPLAINT_SUBGROUPS
            if row[f"has_{subgroup.replace(' ', '_')}"]
        ),
        axis=1,
    )
    return flagged


def build_combined_group_counts(flagged: pd.DataFrame) -> pd.DataFrame:
    """Count the supervisor-requested combined chief-complaint groups."""
    rows = []
    cohorts = ["MHC1_psychotic", "MHC0", "overall"]
    for group_name, subgroup_names in COMBINED_CHIEF_COMPLAINT_GROUPS.items():
        flag_column = f"has_{group_name}"
        source_columns = [
            f"has_{subgroup.replace(' ', '_')}" for subgroup in subgroup_names
        ]
        for cohort in cohorts:
            group = flagged if cohort == "overall" else flagged.loc[flagged["cohort"].eq(cohort)]
            n_admissions = len(group)
            n_with_group = int(group[flag_column].sum())
            n_pure_combined_group = int(
                (
                    group[flag_column]
                    & group[
                        [
                            f"has_{other_name}"
                            for other_name in COMBINED_CHIEF_COMPLAINT_GROUPS
                            if other_name != group_name
                        ]
                    ]
                    .any(axis=1)
                    .eq(False)
                ).sum()
            )
            rows.append(
                {
                    "combined_chief_complaint_group": group_name,
                    "source_subgroups": " | ".join(subgroup_names),
                    "cohort": cohort,
                    "n_admissions": n_admissions,
                    "n_admissions_with_group": n_with_group,
                    "pct_admissions_with_group": (
                        100.0 * n_with_group / n_admissions
                        if n_admissions
                        else 0.0
                    ),
                    "n_not_overlapping_other_combined_group": n_pure_combined_group,
                    "pct_not_overlapping_other_combined_group": (
                        100.0 * n_pure_combined_group / n_with_group
                        if n_with_group
                        else 0.0
                    ),
                    "n_with_fever": int((group[flag_column] & group["has_overlap_fever"]).sum()),
                    "pct_with_fever": (
                        100.0
                        * int((group[flag_column] & group["has_overlap_fever"]).sum())
                        / n_with_group
                        if n_with_group
                        else 0.0
                    ),
                    "n_with_altered_mental_status": int(
                        (
                            group[flag_column]
                            & group["has_overlap_altered_mental_status"]
                        ).sum()
                    ),
                    "pct_with_altered_mental_status": (
                        100.0
                        * int(
                            (
                                group[flag_column]
                                & group["has_overlap_altered_mental_status"]
                            ).sum()
                        )
                        / n_with_group
                        if n_with_group
                        else 0.0
                    ),
                    "n_with_both_fever_and_altered_mental_status": int(
                        (
                            group[flag_column]
                            & group["has_overlap_fever"]
                            & group["has_overlap_altered_mental_status"]
                        ).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_combined_group_pair_overlap(flagged: pd.DataFrame) -> pd.DataFrame:
    """Summarize overlap between the two combined chief-complaint groups."""
    group_names = list(COMBINED_CHIEF_COMPLAINT_GROUPS)
    first, second = group_names
    first_column = f"has_{first}"
    second_column = f"has_{second}"
    rows = []
    for cohort in ["MHC1_psychotic", "MHC0", "overall"]:
        group = flagged if cohort == "overall" else flagged.loc[flagged["cohort"].eq(cohort)]
        n_admissions = len(group)
        n_first = int(group[first_column].sum())
        n_second = int(group[second_column].sum())
        n_both = int((group[first_column] & group[second_column]).sum())
        n_either = int((group[first_column] | group[second_column]).sum())
        rows.append(
            {
                "cohort": cohort,
                "first_combined_group": first,
                "second_combined_group": second,
                "n_admissions": n_admissions,
                "n_first_group": n_first,
                "n_second_group": n_second,
                "n_both_groups": n_both,
                "n_either_group": n_either,
                "pct_first_group_overlapping_second": (
                    100.0 * n_both / n_first if n_first else 0.0
                ),
                "pct_second_group_overlapping_first": (
                    100.0 * n_both / n_second if n_second else 0.0
                ),
                "jaccard_overlap": n_both / n_either if n_either else 0.0,
            }
        )
    return pd.DataFrame(rows)


def add_exclusive_combined_group(flagged: pd.DataFrame) -> pd.DataFrame:
    """Assign mutually exclusive combined groups, dropping GI/chest-SOB overlaps."""
    flagged = flagged.copy()
    gi_column = "has_abdominal_pain_nausea_vomiting"
    chest_sob_column = "has_chest_pain_shortness_of_breath"
    flagged["has_both_combined_groups"] = flagged[gi_column] & flagged[chest_sob_column]
    flagged["exclusive_combined_group"] = pd.NA
    flagged.loc[
        flagged[gi_column] & ~flagged[chest_sob_column],
        "exclusive_combined_group",
    ] = "abdominal_pain_nausea_vomiting"
    flagged.loc[
        flagged[chest_sob_column] & ~flagged[gi_column],
        "exclusive_combined_group",
    ] = "chest_pain_shortness_of_breath"
    return flagged


def summarize_numeric(series: pd.Series) -> dict[str, float | int]:
    """Return descriptive statistics for one numeric series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "n_nonmissing": 0,
            "mean": pd.NA,
            "sd": pd.NA,
            "min": pd.NA,
            "p25": pd.NA,
            "median": pd.NA,
            "p75": pd.NA,
            "max": pd.NA,
        }
    return {
        "n_nonmissing": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)) if clean.shape[0] > 1 else 0.0,
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def build_exclusive_group_numeric_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    """Summarize age, Elixhauser, and QuickUMLS term counts for 2x2 groups."""
    analysis = flagged.loc[flagged["exclusive_combined_group"].notna()].copy()
    analysis["quickumls_term_count"] = analysis["quickumls_terms"].map(
        lambda value: len(split_pipe_terms(value))
    )
    analysis["derived_quickumls_term_count"] = analysis[
        "derived_quickumls_overlap_terms"
    ].map(lambda value: len(split_pipe_terms(value)))

    rows = []
    metrics = {
        "age_at_admission": "age_at_admission",
        "elixhauser_score": "elixhauser_score",
        "quickumls_term_count": "quickumls_term_count",
        "derived_quickumls_term_count": "derived_quickumls_term_count",
    }
    for (cohort, group_name), group in analysis.groupby(
        ["cohort", "exclusive_combined_group"]
    ):
        for metric_name, column in metrics.items():
            row = {
                "cohort": cohort,
                "exclusive_combined_group": group_name,
                "metric": metric_name,
                "n_admissions": len(group),
            }
            row.update(summarize_numeric(group[column]))
            rows.append(row)
    return pd.DataFrame(rows)


def build_exclusive_group_categorical_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    """Summarize sex and insurance group distributions for 2x2 groups."""
    analysis = flagged.loc[flagged["exclusive_combined_group"].notna()].copy()
    rows = []
    for variable in ["sex", "insurance_group"]:
        analysis[variable] = (
            analysis[variable].fillna("missing").astype(str).str.strip().replace({"": "missing"})
        )
        for (cohort, group_name), group in analysis.groupby(
            ["cohort", "exclusive_combined_group"]
        ):
            total = len(group)
            for category, n_category in group[variable].value_counts(dropna=False).items():
                rows.append(
                    {
                        "cohort": cohort,
                        "exclusive_combined_group": group_name,
                        "variable": variable,
                        "category": category,
                        "n_admissions": int(n_category),
                        "total_admissions": total,
                        "pct_admissions": (
                            100.0 * int(n_category) / total if total else 0.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_exclusive_group_top_quickumls_terms(flagged: pd.DataFrame) -> pd.DataFrame:
    """List top QuickUMLS terms for each mutually exclusive 2x2 group."""
    analysis = flagged.loc[flagged["exclusive_combined_group"].notna()].copy()
    rows = []
    for (cohort, group_name), group in analysis.groupby(
        ["cohort", "exclusive_combined_group"]
    ):
        term_counts: dict[str, int] = {}
        admission_counts: dict[str, int] = {}
        for terms in group["quickumls_terms"]:
            term_set = split_pipe_terms(terms)
            for term in term_set:
                term_counts[term] = term_counts.get(term, 0) + 1
                admission_counts[term] = admission_counts.get(term, 0) + 1
        top_terms = sorted(admission_counts.items(), key=lambda item: (-item[1], item[0]))[:25]
        for term, n_admissions_with_term in top_terms:
            rows.append(
                {
                    "cohort": cohort,
                    "exclusive_combined_group": group_name,
                    "quickumls_term": term,
                    "n_admissions_with_term": int(n_admissions_with_term),
                    "total_group_admissions": len(group),
                    "pct_group_admissions_with_term": (
                        100.0 * n_admissions_with_term / len(group)
                        if len(group)
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_exclusive_group_counts(flagged: pd.DataFrame) -> pd.DataFrame:
    """Count mutually exclusive combined groups by cohort."""
    analysis = flagged.loc[flagged["exclusive_combined_group"].notna()].copy()
    rows = []
    for cohort in ["MHC1_psychotic", "MHC0", "overall"]:
        group = analysis if cohort == "overall" else analysis.loc[analysis["cohort"].eq(cohort)]
        total_cohort = len(flagged) if cohort == "overall" else int(flagged["cohort"].eq(cohort).sum())
        for group_name, subgroup in group.groupby("exclusive_combined_group"):
            rows.append(
                {
                    "cohort": cohort,
                    "exclusive_combined_group": group_name,
                    "n_admissions": len(subgroup),
                    "total_cohort_admissions": total_cohort,
                    "pct_total_cohort_admissions": (
                        100.0 * len(subgroup) / total_cohort if total_cohort else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_subgroup_counts(flagged: pd.DataFrame) -> pd.DataFrame:
    """Count subgroup coverage by cohort and overall."""
    rows = []
    cohorts = ["MHC1_psychotic", "MHC0", "overall"]
    for subgroup, config in CHIEF_COMPLAINT_SUBGROUPS.items():
        flag_column = f"has_{subgroup.replace(' ', '_')}"
        for cohort in cohorts:
            if cohort == "overall":
                group = flagged
            else:
                group = flagged.loc[flagged["cohort"].eq(cohort)]
            n_admissions = len(group)
            n_with_subgroup = int(group[flag_column].sum())
            rows.append(
                {
                    "chief_complaint_subgroup": subgroup,
                    "text_phrases": " | ".join(config["text_phrases"]),
                    "quickumls_terms": " | ".join(config["quickumls_terms"]),
                    "cohort": cohort,
                    "n_admissions": n_admissions,
                    "n_admissions_with_subgroup": n_with_subgroup,
                    "pct_admissions_with_subgroup": (
                        100.0 * n_with_subgroup / n_admissions
                        if n_admissions
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_overlap_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    """Summarize overlap between the hard-coded subgroups."""
    return (
        flagged.groupby(["cohort", "n_chief_complaint_subgroups_matched"], as_index=False)
        .agg(n_admissions=("hadm_id", "size"))
        .sort_values(["cohort", "n_chief_complaint_subgroups_matched"])
    )


def write_outputs(
    flagged: pd.DataFrame,
    subgroup_counts: pd.DataFrame,
    combined_group_counts: pd.DataFrame,
) -> None:
    """Write subgroup count and assignment outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subgroup_counts.to_csv(
        OUTPUT_DIR / "chief_complaint_subgroup_counts.csv",
        index=False,
    )
    combined_group_counts.to_csv(
        OUTPUT_DIR / "chief_complaint_combined_group_counts.csv",
        index=False,
    )
    build_combined_group_pair_overlap(flagged).to_csv(
        OUTPUT_DIR / "chief_complaint_combined_group_pair_overlap.csv",
        index=False,
    )
    build_exclusive_group_counts(flagged).to_csv(
        OUTPUT_DIR / "chief_complaint_exclusive_combined_group_counts.csv",
        index=False,
    )
    build_exclusive_group_numeric_summary(flagged).to_csv(
        OUTPUT_DIR / "chief_complaint_exclusive_combined_group_numeric_summary.csv",
        index=False,
    )
    build_exclusive_group_categorical_summary(flagged).to_csv(
        OUTPUT_DIR / "chief_complaint_exclusive_combined_group_categorical_summary.csv",
        index=False,
    )
    build_exclusive_group_top_quickumls_terms(flagged).to_csv(
        OUTPUT_DIR / "chief_complaint_exclusive_combined_group_top_quickumls_terms.csv",
        index=False,
    )
    build_overlap_summary(flagged).to_csv(
        OUTPUT_DIR / "chief_complaint_subgroup_overlap_summary.csv",
        index=False,
    )
    assignment_columns = [
        column
        for column in flagged.columns
        if column
        not in {
            "chief_complaint_normalized",
            "quickumls_terms",
            "derived_quickumls_overlap_terms",
        }
    ]
    flagged.loc[:, assignment_columns].to_csv(
        OUTPUT_DIR / "chief_complaint_subgroup_admission_assignments.csv",
        index=False,
    )



# Current pipeline: explore seven symptom groups, then retain original pairs
# within the five selected pure groups. Older combined outputs above remain
# available for existing archival consumers.
EXPLORATION_OUTPUT_DIR = SCRIPT_DIR / "analysis_output_top_pure_chief_complaints"
PAIR_OUTPUT_DIR = SCRIPT_DIR / "analysis_output_complete_top_five_cc_pairs"
TOP_ASSIGNMENT_PATH = EXPLORATION_OUTPUT_DIR / "top_seven_cc_admission_assignments.csv"
CURRENT_PAIRS_CSV_PATH = PROJECT_DIR / "02_cohort_matching" / "matched_cohort_output" / "matched_pairs.csv"
PAIRED_GROUPS = ["shortness of breath", "altered mental status", "abdominal pain", "chest pain", "fall"]
PAIR_ADMISSION_KEYS = ["pair_id", "cohort", "subject_id", "hadm_id"]

def candidate_groups():
    groups = {name: {key: list(values) for key, values in config.items()}
              for name, config in CHIEF_COMPLAINT_SUBGROUPS.items()}
    # Broaden ranking beyond the five older, predefined groups. These are
    # common concepts observed in aggregate QuickUMLS frequency summaries.
    aliases = {
        "fall": ["fall", "falls", "falling", "mechanical fall"],
        "fever": ["fever", "fevers", "febrile"],
        "weakness": ["weakness", "generalized weakness"],
        "cough": ["cough", "coughing"],
        "hypotension": ["hypotension", "low blood pressure"],
        "diarrhea": ["diarrhea", "diarrhoea"],
        "syncope": ["syncope", "syncopal", "fainting"],
        "seizure": ["seizure", "seizures"],
        "headache": ["headache", "headaches"],
        "hypoxia": ["hypoxia", "hypoxemia"],
        "dizziness": ["dizziness", "dizzy", "lightheadedness"],
        "back pain": ["back pain", "low back pain"],
        "fatigue": ["fatigue", "tiredness"],
        "hyperglycemia": ["hyperglycemia", "high blood sugar"],
        "anemia": ["anemia", "anaemia"],
        "rectal bleeding": ["bright red blood per rectum", "brbpr", "rectal bleeding", "hematochezia"],
        "failure to thrive": ["failure to thrive", "ftt"],
        "leg swelling": ["leg swelling", "lower extremity swelling"],
    }
    for name, terms in aliases.items():
        groups[name] = {"text_phrases": terms, "quickumls_terms": terms}
    return groups


def explore_top_pure_groups():
    data = build_admission_level_complaints(load_matched_pairs())
    if data.duplicated(["cohort", "subject_id", "hadm_id"]).any():
        raise ValueError("Duplicate matched admission keys")
    groups = candidate_groups()
    flags = pd.DataFrame({name: data.apply(lambda row: subgroup_hit(row, config), axis=1)
                          for name, config in groups.items()}, index=data.index)
    rankings = []
    for cohort in ["MHC0", "MHC1_psychotic", "overall"]:
        mask = pd.Series(True, index=data.index) if cohort == "overall" else data.cohort.eq(cohort)
        table = flags.loc[mask].sum().rename_axis("complaint_group").reset_index(name="n_admissions")
        table = table.sort_values(["n_admissions", "complaint_group"], ascending=[False, True])
        table["rank"] = range(1, len(table) + 1)
        table["cohort"] = cohort
        table["pct_cohort_admissions"] = 100 * table.n_admissions / mask.sum()
        rankings.append(table)
    ranking = pd.concat(rankings, ignore_index=True)
    selected = ranking.loc[ranking.cohort.eq("overall") & ranking["rank"].le(7), "complaint_group"].tolist()
    n_matches = flags[selected].sum(axis=1)
    assignments = data.drop(columns=["chief_complaint_normalized", "quickumls_terms", "derived_quickumls_overlap_terms"]).copy()
    assignments["n_selected_groups_matched"] = n_matches
    assignments["selection_status"] = n_matches.map(lambda n: "outside_selected_groups" if n == 0 else "pure" if n == 1 else "multiple_selected_groups")
    assignments["pure_cc_group"] = flags[selected].idxmax(axis=1).where(n_matches.eq(1))
    for name in selected:
        assignments["has_" + name.replace(" ", "_")] = flags[name]
    pure = assignments[assignments.selection_status.eq("pure")]
    counts = []
    pairs = []
    demographics = []
    for name in selected:
        selected_data = pure[pure.pure_cc_group.eq(name)]
        pair_sizes = selected_data.groupby("pair_id").cohort.nunique()
        pairs.append({"complaint_group": name, "n_complete_same_group_pairs": int(pair_sizes.eq(2).sum()),
                      "n_one_sided_pairs": int(pair_sizes.eq(1).sum())})
        for cohort in ["MHC0", "MHC1_psychotic"]:
            group = selected_data[selected_data.cohort.eq(cohort)]
            before = int(flags.loc[data.cohort.eq(cohort), name].sum())
            counts.append({"complaint_group": name, "cohort": cohort, "n_before_purity_filter": before,
                           "n_pure_admissions": len(group), "n_subjects": group.subject_id.nunique(),
                           "n_removed_for_overlap": before-len(group),
                           "pct_cohort_admissions": 100*len(group)/int(data.cohort.eq(cohort).sum())})
            for variable in ["age_at_admission", "elixhauser_score"]:
                demographics.append({"complaint_group": name, "cohort": cohort, "variable": variable,
                                     **summarize_numeric(group[variable])})
    EXPLORATION_OUTPUT_DIR.mkdir(exist_ok=True)
    ranking.to_csv(EXPLORATION_OUTPUT_DIR / "candidate_group_ranking.csv", index=False)
    assignments.to_csv(EXPLORATION_OUTPUT_DIR / "top_seven_cc_admission_assignments.csv", index=False)
    pd.DataFrame(counts).to_csv(EXPLORATION_OUTPUT_DIR / "pure_cc_counts.csv", index=False)
    pd.DataFrame(pairs).to_csv(EXPLORATION_OUTPUT_DIR / "pure_cc_pair_coverage.csv", index=False)
    pd.DataFrame(demographics).to_csv(EXPLORATION_OUTPUT_DIR / "pure_cc_numeric_demographics.csv", index=False)
    selection = assignments.groupby(["cohort", "selection_status"]).size().reset_index(name="n_admissions")
    selection.to_csv(EXPLORATION_OUTPUT_DIR / "pure_cc_selection_summary.csv", index=False)
    definitions = [{"complaint_group": name, "text_phrases": " | ".join(config["text_phrases"]),
                    "quickumls_terms": " | ".join(config["quickumls_terms"]), "selected": name in selected}
                   for name, config in groups.items()]
    pd.DataFrame(definitions).to_csv(EXPLORATION_OUTPUT_DIR / "candidate_group_definitions.csv", index=False)
    print("Top seven grouped complaints before purity filtering:")
    print(ranking[ranking["rank"].le(7)].to_string(index=False))
    print("\nPure subgroup counts:")
    print(pd.DataFrame(counts).to_string(index=False))
    print("\nSelection:")
    print(selection.to_string(index=False))
    print("\nPair coverage:")
    print(pd.DataFrame(pairs).to_string(index=False))


def select_complete_top_five_pairs():
    assignments = pd.read_csv(TOP_ASSIGNMENT_PATH)
    original = pd.read_csv(CURRENT_PAIRS_CSV_PATH)
    expected = pd.concat([
        original[["pair_id", f"{prefix}_subject_id", f"{prefix}_hadm_id"]].rename(
            columns={f"{prefix}_subject_id": "subject_id", f"{prefix}_hadm_id": "hadm_id"}
        ).assign(cohort=cohort)
        for prefix, cohort in [("mhc0", "MHC0"), ("mhc1", "MHC1_psychotic")]
    ], ignore_index=True)
    if assignments.duplicated(PAIR_ADMISSION_KEYS).any() or len(assignments) != len(expected):
        raise ValueError("Assignments do not have one row per current matched admission")
    if set(map(tuple, assignments[PAIR_ADMISSION_KEYS].to_numpy())) != set(map(tuple, expected[PAIR_ADMISSION_KEYS].to_numpy())):
        raise ValueError("Assignment IDs or pair memberships differ from current matching")
    candidates = assignments.loc[
        assignments.selection_status.eq("pure") & assignments.pure_cc_group.isin(PAIRED_GROUPS)
    ].copy()
    if not candidates.n_selected_groups_matched.eq(1).all():
        raise ValueError("Non-pure admissions in candidate selection")
    pair_counts = candidates.groupby(["pure_cc_group", "pair_id"]).cohort.nunique()
    complete = pair_counts[pair_counts.eq(2)].reset_index()[["pure_cc_group", "pair_id"]]
    kept = candidates.merge(complete, on=["pure_cc_group", "pair_id"], how="inner", validate="many_to_one")
    if not kept.groupby("pair_id").size().eq(2).all():
        raise ValueError("Retained pairs do not each have exactly two admissions")
    if not kept.groupby("pair_id").pure_cc_group.nunique().eq(1).all():
        raise ValueError("Retained pair members differ in complaint group")
    rows = []
    for group in PAIRED_GROUPS:
        before = candidates[candidates.pure_cc_group.eq(group)]
        after = kept[kept.pure_cc_group.eq(group)]
        row = {"cc_group": group, "n_complete_pairs": after.pair_id.nunique()}
        for cohort, prefix in [("MHC0", "mhc0"), ("MHC1_psychotic", "mhc1")]:
            b = before[before.cohort.eq(cohort)]
            a = after[after.cohort.eq(cohort)]
            row.update({f"{prefix}_n_pure_before": len(b), f"{prefix}_n_retained": len(a),
                        f"{prefix}_n_removed": len(b)-len(a), f"{prefix}_n_subjects": a.subject_id.nunique(),
                        f"{prefix}_pct_retained": 100*len(a)/len(b) if len(b) else float("nan")})
        rows.append(row)
    pair_groups = kept[["pair_id", "pure_cc_group"]].drop_duplicates()
    pair_export = original.merge(pair_groups, on="pair_id", how="inner", validate="one_to_one")
    PAIR_OUTPUT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(PAIR_OUTPUT_DIR / "complete_cc_pair_selection_summary.csv", index=False)
    kept.sort_values(["pure_cc_group", "pair_id", "cohort"]).to_csv(PAIR_OUTPUT_DIR / "complete_cc_pair_admissions.csv", index=False)
    # Export only identifiers and subgroup labels, not the original CC text.
    pair_export[["pair_id", "pure_cc_group", "mhc0_subject_id", "mhc0_hadm_id", "mhc1_subject_id", "mhc1_hadm_id"]].to_csv(
        PAIR_OUTPUT_DIR / "complete_cc_pairs.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nRetained {len(pair_export):,} original matched pairs ({len(kept):,} admissions).")
    print("Purity remains defined relative to all seven selected complaint groups.")



def main() -> None:
    """Describe current CC groups and select the five complete paired subgroups."""
    matched_pairs = load_matched_pairs()
    admissions = build_admission_level_complaints(matched_pairs)
    flagged = add_exclusive_combined_group(add_subgroup_flags(admissions))
    write_outputs(flagged, build_subgroup_counts(flagged), build_combined_group_counts(flagged))
    explore_top_pure_groups()
    select_complete_top_five_pairs()


if __name__ == "__main__":
    main()
