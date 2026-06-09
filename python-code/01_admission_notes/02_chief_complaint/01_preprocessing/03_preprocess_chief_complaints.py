"""Preprocess extracted chief complaints for concept extraction and embedding.

This script is the main chief-complaint preprocessing step. It reads the minimal
chief-complaint parquet files created from parsed discharge notes, normalizes the
raw chief complaint text, creates lightweight audit tokens, flags foreground
psychiatric/substance/self-harm complaints with MedSpaCy rules, and
extracts UMLS concepts with a local QuickUMLS index.

Inputs:
    chief_complaint_parquets/MHH1_psychotic_chief_complaints.parquet
    chief_complaint_parquets/MHC0_chief_complaints.parquet

Outputs:
    chief_complaint_preprocessed/<group>_chief_complaints_preprocessed.parquet
    chief_complaint_preprocessing_samples/<group>_chief_complaint_preprocessing_sample.csv

Important limitations:
    Normalization is intentionally conservative and does not fully interpret
    clinical language. MedSpaCy rules are hand-written audit flags, not a complete
    symptom extractor. QuickUMLS matching depends on the local UMLS index,
    semantic-type filter, and matching threshold.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

import duckdb
import medspacy
import pandas as pd
import spacy
from medspacy.ner import TargetRule

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "chief_complaint_parquets"
OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_preprocessed"
SAMPLE_OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_preprocessing_samples"
SAMPLE_SIZE = 100 #size for the .csv sample of preprocessed complaints for manual inspection
SAMPLE_RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# QuickUMLS concept extraction
# -----------------------------------------------------------------------------
# QuickUMLS is a local UMLS dictionary/concept matcher. In this pipeline it is
# used as an additional audit/extraction layer on top of the normalized chief
# complaint text.

USE_QUICKUMLS = True
QUICKUMLS_INDEX_DIR = Path("QUICKUMLS_INDEX_DIR")
QUICKUMLS_THRESHOLD = 0.7 #how similar a phrase must be to a UMLS term to count as a match
QUICKUMLS_WINDOW = 5 #the maximum phrase length QuickUMLS considers when scanning text
QUICKUMLS_SIMILARITY_NAME = "jaccard" #jaccard compares token overlap
QUICKUMLS_BEST_MATCH = True #return only the best matching UMLS candidate
QUICKUMLS_IGNORE_SYNTAX = False

QUICKUMLS_ALLOWED_SEMTYPES = {
    # Symptoms/signs
    "T184",  # Sign or Symptom
    "T033",  # Finding

    # Diseases/disorders
    "T047",  # Disease or Syndrome
    "T046",  # Pathologic Function
    "T048",  # Mental or Behavioral Dysfunction
    "T191",  # Neoplastic Process

    # Injuries / abnormalities
    "T037",  # Injury or Poisoning
    "T190",  # Anatomical Abnormality

    # Anatomy
    "T023",  # Body Part, Organ, or Organ Component
    "T029",  # Body Location or Region
    "T030",  # Body Space or Junction

    # Functional concepts
    "T038"  # Biologic Function
    "T039"  # Physiologic Function
    "T040"  # Organism Function

    # Procedures (sometimes mentioned in chief complaints, e.g. "here for chemo")
    "T060",  # Diagnostic Procedure
    "T061",  # Therapeutic or Preventive Procedure
    "T059",  # Laboratory Procedure

    # Accident / mechanism-of-injury concepts (???) 
    "T051",  # Event
    "T052",  # Activity  # optional, inspect noise
}


# Input chief-complaint parquet files produced by the discharge-note parsing 
INPUTS = {
    "MHH1_psychotic": INPUT_DIR / "MHH1_psychotic_chief_complaints.parquet",
    "MHC0": INPUT_DIR / "MHC0_chief_complaints.parquet",
}

# Regexes and placeholder values used for lightweight text normalization.
PLACEHOLDER_RE = re.compile(r"\b_+\b|_+")
SPACING_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s/+-]")
TOKEN_RE = re.compile(r"[a-z][a-z0-9/+.-]*")
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

# Common prefixes that sometimes remain in the extracted chief complaint field.
PREFIX_PATTERNS = [
    re.compile(r"^\s*(?:cc|chief complaint)\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*(?:admit(?:ted)?\s+for|admission\s+for)\s+", re.IGNORECASE),
    re.compile(r"^\s*(?:s/p|status post)\s+", re.IGNORECASE),
]


# Small hand-written abbreviation map for frequent chief-complaint shorthand.
# This is not meant to be a full clinical abbreviation dictionary; it only covers
# forms that are common enough to affect complaint matching.
ABBREVIATIONS = {
    "abd": "abdominal",
    "ams": "altered mental status",
    "brbpr": "bright red blood per rectum",
    "cp": "chest pain",
    "cva": "stroke",
    "doe": "dyspnea on exertion",
    "etoh": "alcohol",
    "fx": "fracture",
    "gi": "gastrointestinal",
    "gi bleed": "gastrointestinal bleed",
    "gib": "gastrointestinal bleed",
    "ha": "headache",
    "lle": "left lower extremity",
    "llq": "left lower quadrant",
    "loc": "loss of consciousness",
    "lue": "left upper extremity",
    "luq": "left upper quadrant",
    "mva": "motor vehicle collision",
    "mvc": "motor vehicle collision",
    "n/v": "nausea vomiting",
    "n/v/d": "nausea vomiting diarrhea",
    "pna": "pneumonia",
    "rle": "right lower extremity",
    "rlq": "right lower quadrant",
    "rue": "right upper extremity",
    "ruq": "right upper quadrant",
    "sob": "shortness of breath",
    "uti": "urinary tract infection",
}


# MedSpaCy TargetMatcher vocabulary.
# These hand-written rules are independent of QuickUMLS. They are used only for
# transparent psychiatric, substance-related, and self-harm audit flags checking for contamination.
TARGET_RULES = {
    "anxiety": {
        "entity_type": "psych_substance_self_harm",
        "literals": ["anxiety", "panic attack", "panic"],
    },
    "depression": {
        "entity_type": "psych_substance_self_harm",
        "literals": ["depression", "depressed"],
    },
    "hallucinations": {
        "entity_type": "psych_substance_self_harm",
        "literals": [
            "hallucinations",
            "hallucination",
            "auditory hallucinations",
            "visual hallucinations",
            "hearing voices",
            "seeing things",
        ],
    },
    "homicidal_ideation": {
        "entity_type": "psych_substance_self_harm",
        "literals": ["homicidal ideation", "homicidal", "hi"],
    },
    "intoxication": {
        "entity_type": "psych_substance_self_harm",
        "literals": ["intoxication", "intoxicated", "alcohol intoxication", "etoh intoxication"],
    },
    "overdose_ingestion": {
        "entity_type": "psych_substance_self_harm",
        "literals": ["overdose", "od", "ingestion", "intentional ingestion"],
    },
    "psychosis": {
        "entity_type": "psych_substance_self_harm",
        "literals": ["psychosis", "psychotic", "paranoia", "paranoid", "delusions", "delusional"],
    },
    "substance_use": {
        "entity_type": "psych_substance_self_harm",
        "literals": [
            "alcohol",
            "etoh",
            "cocaine",
            "heroin",
            "opioid",
            "opiate",
            "substance use",
            "drug use",
            "withdrawal",
        ],
    },
    "suicidal_ideation": {
        "entity_type": "psych_substance_self_harm",
        "literals": ["suicidal ideation", "suicidal", "suicide", "si"],
    },
}


# SQL helper used when passing file paths to DuckDB COPY/read_parquet calls.
def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


# Remove repeated administrative prefixes from the beginning of a complaint.
def strip_prefixes(text: str) -> str:
    value = text
    changed = True
    while changed:
        changed = False
        for pattern in PREFIX_PATTERNS:
            new_value = pattern.sub("", value)
            if new_value != value:
                value = new_value
                changed = True
    return value



# Normalize one raw chief complaint string.
#
# This removes placeholders, strips common prefixes, lowercases text, expands a
# small set of abbreviations, removes most punctuation, and converts known
# missing-value placeholders to the empty string.
def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""

    value = str(text or "")
    value = PLACEHOLDER_RE.sub(" ", value)
    value = strip_prefixes(value)
    value = value.lower().replace("&", " and ")
    value = re.sub(r"\bs\.?\s*/\s*p\.?\b", " ", value)
    value = PUNCT_RE.sub(" ", value)
    value = SPACING_RE.sub(" ", value).strip()

    for short, expanded in sorted(ABBREVIATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        value = re.sub(rf"(?<!\w){re.escape(short)}(?!\w)", expanded, value)

    value = SPACING_RE.sub(" ", value).strip()
    if value in MISSING_CHIEF_COMPLAINT_VALUES:
        return ""
    return value



# Create a simple pipe-separated token audit column. This is for inspection only;
# it is not used as the main embedding representation.
def extract_tokens(text: str) -> str:
    return " | ".join(TOKEN_RE.findall(text))



# Build the local MedSpaCy pipeline used for rule-based entity extraction and
# negation detection.
def build_medspacy_pipeline() -> spacy.Language:
    # Use MedSpaCy's target matcher and ConText in a blank spaCy pipeline.
    # The sentencizer is required for ConText modifier scopes.
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("medspacy_target_matcher")
    nlp.add_pipe("medspacy_context")
    target_matcher = nlp.get_pipe("medspacy_target_matcher")

    rules = [
        TargetRule(
            literal=literal,
            category=concept_name,
            metadata={"entity_type": config["entity_type"]},
        )
        for concept_name, config in TARGET_RULES.items()
        for literal in config["literals"]
    ]
    target_matcher.add(rules)
    return nlp



# Format a list as a stable, deduplicated pipe-separated string.
def join_unique(values: list[str]) -> str:
    return " | ".join(sorted(set(values)))



# Build the local QuickUMLS matcher if enabled. The matcher uses the local UMLS
# index path.
def build_quickumls_matcher():
    if not USE_QUICKUMLS:
        print("QuickUMLS extraction disabled.", flush=True)
        return None

    if not QUICKUMLS_INDEX_DIR.exists():
        raise FileNotFoundError(
            "QuickUMLS is enabled, but the configured index directory does not exist: "
            f"{QUICKUMLS_INDEX_DIR}"
        )

    from quickumls import QuickUMLS

    print(f"Using QuickUMLS index: {QUICKUMLS_INDEX_DIR}", flush=True)
    return QuickUMLS(
        str(QUICKUMLS_INDEX_DIR),
        threshold=QUICKUMLS_THRESHOLD,
        window=QUICKUMLS_WINDOW,
        similarity_name=QUICKUMLS_SIMILARITY_NAME,
        accepted_semtypes=QUICKUMLS_ALLOWED_SEMTYPES,
    )



# Extract semantic type IDs from a QuickUMLS match object while being tolerant to
# small differences in QuickUMLS output key names.
def normalize_quickumls_semtypes(match: dict[str, Any]) -> list[str]:
    semtype_values = []
    for key, value in match.items():
        if "semtype" not in key.lower():
            continue
        if value is None:
            continue
        if isinstance(value, str):
            semtype_values.append(value)
        elif isinstance(value, (list, tuple, set)):
            semtype_values.extend(str(item) for item in value if item is not None)

    cleaned = []
    for semtype in semtype_values:
        value = semtype.strip()
        if value:
            cleaned.append(value)
    return sorted(set(cleaned))



# Keep only QuickUMLS matches whose semantic types are relevant to complaints,
# findings, diseases, injuries, mental/behavioral dysfunctions, or anatomy.
def is_allowed_quickumls_match(match: dict[str, Any]) -> bool:
    if not QUICKUMLS_ALLOWED_SEMTYPES:
        return True

    semtypes = normalize_quickumls_semtypes(match)
    return any(semtype in QUICKUMLS_ALLOWED_SEMTYPES for semtype in semtypes)



# QuickUMLS returns nested lists of candidate matches; flatten them into a simple
# list of dictionaries.
def flatten_quickumls_matches(matches: Any) -> list[dict[str, Any]]:
    flattened = []
    if matches is None:
        return flattened
    if isinstance(matches, dict):
        return [matches]
    if isinstance(matches, (list, tuple)):
        for item in matches:
            flattened.extend(flatten_quickumls_matches(item))
    return flattened



# Return the first non-missing value from a list of possible match-object keys.
def first_present(match: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in match and match[key] is not None:
            return match[key]
    return None



# Reduce a raw QuickUMLS match to a compact JSON-serializable structure. This
# avoids storing large raw matcher objects in the output parquet.
def simplify_quickumls_match(match: dict[str, Any]) -> dict[str, Any]:
    semtypes = normalize_quickumls_semtypes(match)
    similarity = first_present(match, ["similarity", "score", "sim"])
    start = first_present(match, ["start", "start_char", "begin"])
    end = first_present(match, ["end", "end_char", "stop"])
    return {
        "cui": first_present(match, ["cui", "CUI"]),
        "term": first_present(match, ["term", "ngram", "match", "matched_term"]),
        "preferred": first_present(match, ["preferred", "preferred_term", "term_preferred"]),
        "similarity": float(similarity) if similarity is not None else None,
        "semtypes": semtypes,
        "start": int(start) if start is not None else None,
        "end": int(end) if end is not None else None,
    }



# Return True for readable clinical strings and False for artifacts such as
# one-character or numeric-only UMLS labels.
def is_informative_quickumls_text(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if len(text) < 2:
        return False
    return any(char.isalpha() for char in text)



# Prefer the literal matched span for readable output. Keep the UMLS preferred
# term in `quickumls_matches_json` for audit, but do not let odd preferred labels
# such as "1" dominate the sample/extracted text columns.
def quickumls_display_term(match: dict[str, Any]) -> str:
    for key in ["term", "preferred"]:
        value = match.get(key)
        if is_informative_quickumls_text(value):
            return str(value).strip()
    return ""



# Preserve first occurrence order while removing empty and duplicate values.
def unique_nonempty(values: list[Any]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output



# Standard empty QuickUMLS output for blank complaints or no-match cases.
def empty_quickumls_row() -> dict[str, Any]:
    return {
        "quickumls_matches_json": "[]",
        "quickumls_cuis": "",
        "quickumls_terms": "",
        "quickumls_semtypes": "",
        "quickumls_extracted_text": "",
        "has_quickumls_match": False,
    }



# Run QuickUMLS over normalized chief complaints and return one output row per
# input complaint. 
def extract_quickumls_rows(matcher, texts: pd.Series) -> pd.DataFrame:
    if matcher is None:
        return pd.DataFrame([empty_quickumls_row() for _ in range(len(texts))])

    rows = []
    for i, text in enumerate(texts.fillna("").astype(str), start=1):
        if i % 1000 == 0:
            print(f"QuickUMLS processed {i:,} chief complaints", flush=True)

        if not text.strip():
            rows.append(empty_quickumls_row())
            continue

        raw_matches = matcher.match(
            text,
            best_match=QUICKUMLS_BEST_MATCH,
            ignore_syntax=QUICKUMLS_IGNORE_SYNTAX,
        )
        simplified_matches = []
        for match in flatten_quickumls_matches(raw_matches):
            if not is_allowed_quickumls_match(match):
                continue
            simplified = simplify_quickumls_match(match)
            if not simplified["cui"] and not simplified["term"] and not simplified["preferred"]:
                continue
            if not quickumls_display_term(simplified):
                continue
            simplified_matches.append(simplified)

        deduped_matches = []
        seen = set()
        for match in simplified_matches:
            key = (
                match["cui"],
                match["term"],
                match["preferred"],
                tuple(match["semtypes"]),
                match["start"],
                match["end"],
            )
            if key in seen:
                continue
            seen.add(key)
            deduped_matches.append(match)

        cuis = unique_nonempty([match["cui"] for match in deduped_matches])
        terms = unique_nonempty([quickumls_display_term(match) for match in deduped_matches])
        semtypes = unique_nonempty(
            [
                semtype
                for match in deduped_matches
                for semtype in match["semtypes"]
            ]
        )
        rows.append(
            {
                "quickumls_matches_json": json.dumps(
                    deduped_matches,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ),
                "quickumls_cuis": " | ".join(cuis),
                "quickumls_terms": " | ".join(terms),
                "quickumls_semtypes": " | ".join(semtypes),
                "quickumls_extracted_text": " | ".join(terms),
                "has_quickumls_match": bool(deduped_matches),
            }
        )

    return pd.DataFrame(rows)



# Run MedSpaCy TargetMatcher + ConText over normalized chief complaints and
# summarize affirmed/negated psychiatric, substance-related, and self-harm entities.
def extract_entity_rows_with_medspacy(nlp: spacy.Language, texts: pd.Series) -> pd.DataFrame:
    rows = []
    for doc in nlp.pipe(texts.fillna("").tolist(), batch_size=1000):
        all_entities = []
        affirmed_psych = []
        negated_psych = []

        for ent in doc.ents:
            entity_type = ent._.target_rule.metadata.get("entity_type", "")
            entity_text = f"{ent.text}=>{ent.label_}"
            is_negated = bool(ent._.is_negated)
            all_entities.append(
                f"{entity_text} ({entity_type}; {'negated' if is_negated else 'affirmed'})"
            )

            if entity_type == "psych_substance_self_harm":
                if is_negated:
                    negated_psych.append(entity_text)
                else:
                    affirmed_psych.append(entity_text)

        rows.append(
            {
                "medspacy_entities_all": join_unique(all_entities),
                "psych_substance_self_harm_entities_affirmed": join_unique(affirmed_psych),
                "psych_substance_self_harm_entities_negated": join_unique(negated_psych),
            }
        )

    return pd.DataFrame(rows)



# Apply the full preprocessing pipeline to one cohort dataframe.
#
# Output keeps the raw complaint, normalized complaint, token audit column,
# MedSpaCy audit columns, QuickUMLS audit columns, and boolean flags used for
# quality-control summaries.
def preprocess_frame(
    df: pd.DataFrame,
    group_name: str,
    nlp: spacy.Language,
    quickumls_matcher=None,
) -> pd.DataFrame:
    output = df.copy()
    output.insert(0, "source_table", group_name)
    output = output.rename(columns={"chief_complaint": "chief_complaint_raw"})
    output["chief_complaint_normalized"] = output["chief_complaint_raw"].map(normalize_text)
    output["chief_complaint_tokens"] = output["chief_complaint_normalized"].map(extract_tokens)

    entity_rows = extract_entity_rows_with_medspacy(nlp, output["chief_complaint_normalized"])
    quickumls_rows = extract_quickumls_rows(
        quickumls_matcher,
        output["chief_complaint_normalized"],
    )
    output = pd.concat(
        [
            output.reset_index(drop=True),
            entity_rows.reset_index(drop=True),
            quickumls_rows.reset_index(drop=True),
        ],
        axis=1,
    )
    output["has_chief_complaint"] = output["chief_complaint_normalized"] != ""
    output["has_affirmed_psych_substance_self_harm_entity"] = (
        output["psych_substance_self_harm_entities_affirmed"] != ""
    )
    output["has_any_affirmed_entity"] = output[
        "has_affirmed_psych_substance_self_harm_entity"
    ]
    return output



# Read the minimal chief-complaint input columns needed for preprocessing.
def read_parquet(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT subject_id, hadm_id, chief_complaint
        FROM read_parquet({sql_string(str(path))})
        """
    ).fetchdf()



# Write a preprocessed cohort dataframe to parquet via DuckDB.
def write_parquet(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, output_path: Path) -> None:
    con.register("preprocessed_chief_complaints", df)
    try:
        con.execute(
            f"""
            COPY preprocessed_chief_complaints
            TO {sql_string(str(output_path))}
            (FORMAT PARQUET)
            """
        )
    finally:
        con.unregister("preprocessed_chief_complaints")



# Write a small CSV sample for manual inspection of normalization, MedSpaCy, and
# QuickUMLS extraction quality.
def write_sample(df: pd.DataFrame, group_name: str) -> None:
    SAMPLE_OUTPUT_DIR.mkdir(exist_ok=True)
    sample_path = SAMPLE_OUTPUT_DIR / f"{group_name}_chief_complaint_preprocessing_sample.csv"
    sample_columns = [
        "sample_reason",
        "source_table",
        "subject_id",
        "hadm_id",
        "chief_complaint_raw",
        "chief_complaint_normalized",
        "chief_complaint_tokens",
        "psych_substance_self_harm_entities_affirmed",
        "psych_substance_self_harm_entities_negated",
        "medspacy_entities_all",
    ]
    quickumls_sample_columns = [
        "quickumls_terms",
        "quickumls_cuis",
        "quickumls_semtypes",
        "quickumls_extracted_text",
        "quickumls_matches_json",
        "has_quickumls_match",
    ]
    sample_columns.extend(
        column for column in quickumls_sample_columns if column in df.columns
    )

    candidates = df.loc[df["has_chief_complaint"]].copy()
    candidates["_psych_flag"] = (
        candidates["psych_substance_self_harm_entities_affirmed"].fillna("").ne("")
        | candidates["psych_substance_self_harm_entities_negated"].fillna("").ne("")
    )
    candidates["_quickumls_match"] = candidates.get(
        "has_quickumls_match",
        pd.Series(False, index=candidates.index),
    ).fillna(False)

    per_group = max(1, SAMPLE_SIZE // 3)
    sample_groups = [
        (
            "psych_substance_self_harm_flag",
            candidates.loc[candidates["_psych_flag"]],
        ),
        (
            "no_quickumls_match",
            candidates.loc[~candidates["_quickumls_match"]],
        ),
        (
            "ordinary_quickumls_match_without_psych_flag",
            candidates.loc[
                ~candidates["_psych_flag"] & candidates["_quickumls_match"]
            ],
        ),
    ]

    selected = []
    selected_indexes = set()
    for sample_reason, group in sample_groups:
        sampled = group.sample(
            n=min(per_group, len(group)),
            random_state=SAMPLE_RANDOM_SEED,
        ).copy()
        sampled["sample_reason"] = sample_reason
        selected.append(sampled)
        selected_indexes.update(sampled.index)

    remaining_n = SAMPLE_SIZE - sum(len(sampled) for sampled in selected)
    if remaining_n > 0:
        remaining = candidates.loc[~candidates.index.isin(selected_indexes)]
        fill = remaining.sample(
            n=min(remaining_n, len(remaining)),
            random_state=SAMPLE_RANDOM_SEED,
        ).copy()
        fill["sample_reason"] = "random_fill"
        selected.append(fill)

    sample = pd.concat(selected, ignore_index=True).head(SAMPLE_SIZE)
    sample = sample[sample_columns].copy()
    sample.to_csv(sample_path, index=False)



# Print lightweight quality-control counts after preprocessing one cohort.
def print_summary(df: pd.DataFrame, output_name: str) -> None:
    summary = {
        "output_name": output_name,
        "n_rows": len(df),
        "n_subjects": df["subject_id"].nunique(),
        "n_admissions": df["hadm_id"].nunique(),
        "n_with_chief_complaint": int(df["has_chief_complaint"].sum()),
        "n_with_affirmed_psych_substance_self_harm_entity": int(
            df["has_affirmed_psych_substance_self_harm_entity"].sum()
        ),
    }
    if "has_quickumls_match" in df.columns:
        n_with_quickumls_match = int(df["has_quickumls_match"].sum())
        summary["n_with_quickumls_match"] = n_with_quickumls_match
        summary["pct_with_quickumls_match"] = (
            100.0 * n_with_quickumls_match / len(df) if len(df) else 0.0
        )
    print(pd.DataFrame([summary]).to_string(index=False))



# Script entry point: build local NLP matchers, process each cohort input parquet,
# and write preprocessed parquet + inspection samples.
def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    SAMPLE_OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Using MedSpaCy {medspacy.__version__} target matcher + ConText", flush=True)
    nlp = build_medspacy_pipeline()
    quickumls_matcher = build_quickumls_matcher()

    con = duckdb.connect()
    try:
        for group_name, input_path in INPUTS.items():
            output_name = f"{group_name}_chief_complaints_preprocessed.parquet"
            output_path = OUTPUT_DIR / output_name

            print(f"Preprocessing {input_path.name}", flush=True)
            df = read_parquet(con, input_path)
            preprocessed = preprocess_frame(df, group_name, nlp, quickumls_matcher)
            write_parquet(con, preprocessed, output_path)
            write_sample(preprocessed, group_name)
            print_summary(
                preprocessed,
                output_name,
            )
    finally:
        con.close()

    print(f"Saved preprocessed chief-complaint parquets to: {OUTPUT_DIR}")
    print(f"Saved preprocessing samples to: {SAMPLE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
