from __future__ import annotations

import re
from pathlib import Path

import duckdb
import medspacy
import pandas as pd
import spacy
from medspacy.ner import TargetRule


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "chief_complaint_parquets"
OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_preprocessed"
SAMPLE_OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_preprocessing_samples"
SAMPLE_SIZE = 100

INPUTS = {
    "MHH1_psychotic": INPUT_DIR / "MHH1_psychotic_chief_complaints.parquet",
    "MHC0": INPUT_DIR / "MHC0_chief_complaints.parquet",
}

PLACEHOLDER_RE = re.compile(r"\b_+\b|_+")
SPACING_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s/+-]")
TOKEN_RE = re.compile(r"[a-z][a-z0-9/+.-]*")

PREFIX_PATTERNS = [
    re.compile(r"^\s*(?:cc|chief complaint)\s*:\s*", re.IGNORECASE),
    re.compile(r"^\s*(?:admit(?:ted)?\s+for|admission\s+for)\s+", re.IGNORECASE),
    re.compile(r"^\s*(?:s/p|status post)\s+", re.IGNORECASE),
]

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

# These rules are used as a synonym/near-synonym vocabulary, not as final
# unsupervised clusters. Each target should group surface forms that express the
# same presenting problem or a very close chief-complaint variant.
TARGET_RULES = {
    "abdominal_pain": {
        "entity_type": "physical_complaint",
        "literals": [
            "abdominal pain",
            "abd pain",
            "stomach pain",
            "belly pain",
            "epigastric pain",
            "right upper quadrant pain",
            "left upper quadrant pain",
            "right lower quadrant pain",
            "left lower quadrant pain",
            "ruq pain",
            "luq pain",
            "rlq pain",
            "llq pain",
        ],
    },
    "altered_mental_status": {
        "entity_type": "physical_complaint",
        "literals": [
            "altered mental status",
            "confusion",
            "confused",
            "encephalopathy",
            "lethargy",
            "lethargic",
            "somnolence",
            "somnolent",
            "unresponsive",
            "unresponsiveness",
        ],
    },
    "back_pain": {
        "entity_type": "physical_complaint",
        "literals": ["back pain", "low back pain", "lower back pain"],
    },
    "chest_pain": {
        "entity_type": "physical_complaint",
        "literals": [
            "chest pain",
            "chest pressure",
            "chest discomfort",
            "chest tightness",
            "chest heaviness",
            "angina",
        ],
    },
    "cough": {
        "entity_type": "physical_complaint",
        "literals": ["cough", "coughing", "hemoptysis", "coughing blood"],
    },
    "diarrhea": {
        "entity_type": "physical_complaint",
        "literals": ["diarrhea", "loose stools", "watery stools"],
    },
    "dizziness_vertigo": {
        "entity_type": "physical_complaint",
        "literals": ["dizziness", "dizzy", "vertigo", "room spinning"],
    },
    "dyspnea": {
        "entity_type": "physical_complaint",
        "literals": [
            "dyspnea",
            "shortness of breath",
            "difficulty breathing",
            "trouble breathing",
            "cannot breathe",
            "can't breathe",
            "cant breathe",
            "i cannot breathe",
            "i can't breathe",
            "i cant breathe",
            "dyspnea on exertion",
            "respiratory distress",
            "hypoxia",
            "hypoxemia",
        ],
    },
    "fall": {
        "entity_type": "physical_complaint",
        "literals": ["fall", "falls", "fell", "found down"],
    },
    "fever": {
        "entity_type": "physical_complaint",
        "literals": ["fever", "fevers", "febrile", "high fever"],
    },
    "gastrointestinal_bleed": {
        "entity_type": "physical_complaint",
        "literals": [
            "gastrointestinal bleed",
            "gastrointestinal bleeding",
            "gi bleed",
            "gib",
            "bright red blood per rectum",
            "brbpr",
            "rectal bleeding",
            "melena",
            "hematochezia",
            "hematemesis",
            "coffee ground emesis",
        ],
    },
    "headache": {
        "entity_type": "physical_complaint",
        "literals": ["headache", "headaches", "head pain", "migraine"],
    },
    "hypotension": {
        "entity_type": "physical_complaint",
        "literals": ["hypotension", "low blood pressure"],
    },
    "limb_pain": {
        "entity_type": "physical_complaint",
        "literals": [
            "arm pain",
            "leg pain",
            "hip pain",
            "knee pain",
            "ankle pain",
            "foot pain",
            "shoulder pain",
            "elbow pain",
            "wrist pain",
            "hand pain",
        ],
    },
    "limb_swelling": {
        "entity_type": "physical_complaint",
        "literals": [
            "leg swelling",
            "arm swelling",
            "foot swelling",
            "ankle swelling",
            "lower extremity swelling",
            "upper extremity swelling",
            "edema",
        ],
    },
    "loss_of_consciousness": {
        "entity_type": "physical_complaint",
        "literals": ["loss of consciousness", "lost consciousness", "loc"],
    },
    "motor_vehicle_collision": {
        "entity_type": "physical_complaint",
        "literals": [
            "motor vehicle collision",
            "motor vehicle crash",
            "motor vehicle accident",
            "mvc",
            "mva",
        ],
    },
    "nausea_vomiting": {
        "entity_type": "physical_complaint",
        "literals": [
            "nausea",
            "vomiting",
            "emesis",
            "nausea vomiting",
            "nausea and vomiting",
            "n/v",
        ],
    },
    "palpitations": {
        "entity_type": "physical_complaint",
        "literals": ["palpitations", "heart racing", "racing heart"],
    },
    "pneumonia": {
        "entity_type": "physical_complaint",
        "literals": ["pneumonia", "pna"],
    },
    "seizure": {
        "entity_type": "physical_complaint",
        "literals": ["seizure", "seizures", "sz"],
    },
    "stroke_like_symptoms": {
        "entity_type": "physical_complaint",
        "literals": [
            "stroke",
            "code stroke",
            "aphasia",
            "slurred speech",
            "facial droop",
            "facial weakness",
            "word finding difficulty",
        ],
    },
    "syncope": {
        "entity_type": "physical_complaint",
        "literals": ["syncope", "presyncope", "near syncope", "fainting", "passed out"],
    },
    "traumatic_injury": {
        "entity_type": "physical_complaint",
        "literals": ["fracture", "fx", "trauma", "injury", "laceration"],
    },
    "urinary_symptoms": {
        "entity_type": "physical_complaint",
        "literals": [
            "urinary tract infection",
            "uti",
            "dysuria",
            "hematuria",
            "urinary retention",
        ],
    },
    "weakness": {
        "entity_type": "physical_complaint",
        "literals": ["weakness", "generalized weakness", "fatigue", "malaise"],
    },
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


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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


def normalize_text(text: str) -> str:
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
    return value


def extract_tokens(text: str) -> str:
    return " | ".join(TOKEN_RE.findall(text))


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


def join_unique(values: list[str]) -> str:
    return " | ".join(sorted(set(values)))


def extract_entity_rows_with_medspacy(nlp: spacy.Language, texts: pd.Series) -> pd.DataFrame:
    rows = []
    for doc in nlp.pipe(texts.fillna("").tolist(), batch_size=1000):
        all_entities = []
        affirmed_physical = []
        negated_physical = []
        affirmed_psych = []
        negated_psych = []

        for ent in doc.ents:
            entity_type = ent._.target_rule.metadata.get("entity_type", "")
            entity_text = f"{ent.text}=>{ent.label_}"
            is_negated = bool(ent._.is_negated)
            all_entities.append(
                f"{entity_text} ({entity_type}; {'negated' if is_negated else 'affirmed'})"
            )

            if entity_type == "physical_complaint":
                if is_negated:
                    negated_physical.append(entity_text)
                else:
                    affirmed_physical.append(entity_text)
            elif entity_type == "psych_substance_self_harm":
                if is_negated:
                    negated_psych.append(entity_text)
                else:
                    affirmed_psych.append(entity_text)

        rows.append(
            {
                "medspacy_entities_all": join_unique(all_entities),
                "physical_entities_affirmed": join_unique(affirmed_physical),
                "physical_entities_negated": join_unique(negated_physical),
                "psych_substance_self_harm_entities_affirmed": join_unique(affirmed_psych),
                "psych_substance_self_harm_entities_negated": join_unique(negated_psych),
            }
        )

    return pd.DataFrame(rows)


def preprocess_frame(df: pd.DataFrame, group_name: str, nlp: spacy.Language) -> pd.DataFrame:
    output = df.copy()
    output.insert(0, "source_table", group_name)
    output = output.rename(columns={"chief_complaint": "chief_complaint_raw"})
    output["chief_complaint_normalized"] = output["chief_complaint_raw"].map(normalize_text)
    output["chief_complaint_tokens"] = output["chief_complaint_normalized"].map(extract_tokens)

    entity_rows = extract_entity_rows_with_medspacy(nlp, output["chief_complaint_normalized"])
    output = pd.concat([output.reset_index(drop=True), entity_rows], axis=1)
    output["has_chief_complaint"] = output["chief_complaint_normalized"] != ""
    output["has_affirmed_physical_entity"] = output["physical_entities_affirmed"] != ""
    output["has_affirmed_psych_substance_self_harm_entity"] = (
        output["psych_substance_self_harm_entities_affirmed"] != ""
    )
    output["has_any_affirmed_entity"] = (
        output["has_affirmed_physical_entity"]
        | output["has_affirmed_psych_substance_self_harm_entity"]
    )
    return output


def read_parquet(con: duckdb.DuckDBPyConnection, path: Path) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT subject_id, hadm_id, chief_complaint
        FROM read_parquet({sql_string(str(path))})
        """
    ).fetchdf()


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


def write_sample(df: pd.DataFrame, group_name: str) -> None:
    SAMPLE_OUTPUT_DIR.mkdir(exist_ok=True)
    sample_path = SAMPLE_OUTPUT_DIR / f"{group_name}_chief_complaint_preprocessing_sample.csv"
    sample_columns = [
        "source_table",
        "subject_id",
        "hadm_id",
        "chief_complaint_raw",
        "chief_complaint_normalized",
        "chief_complaint_tokens",
        "physical_entities_affirmed",
        "physical_entities_negated",
        "psych_substance_self_harm_entities_affirmed",
        "psych_substance_self_harm_entities_negated",
        "medspacy_entities_all",
    ]

    candidates = df.loc[df["has_chief_complaint"]].copy()
    sample = pd.concat(
        [
            candidates.loc[candidates["has_any_affirmed_entity"]],
            candidates.loc[~candidates["has_any_affirmed_entity"]],
        ],
        ignore_index=True,
    ).head(SAMPLE_SIZE)
    sample = sample[sample_columns].copy()
    sample.to_csv(sample_path, index=False)


def print_summary(df: pd.DataFrame, output_name: str) -> None:
    summary = {
        "output_name": output_name,
        "n_rows": len(df),
        "n_subjects": df["subject_id"].nunique(),
        "n_admissions": df["hadm_id"].nunique(),
        "n_with_chief_complaint": int(df["has_chief_complaint"].sum()),
        "n_with_affirmed_physical_entity": int(df["has_affirmed_physical_entity"].sum()),
        "n_with_affirmed_psych_substance_self_harm_entity": int(
            df["has_affirmed_psych_substance_self_harm_entity"].sum()
        ),
        "n_with_any_affirmed_entity": int(df["has_any_affirmed_entity"].sum()),
    }
    print(pd.DataFrame([summary]).to_string(index=False))


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    SAMPLE_OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Using MedSpaCy {medspacy.__version__} target matcher + ConText", flush=True)
    nlp = build_medspacy_pipeline()

    con = duckdb.connect()
    try:
        for group_name, input_path in INPUTS.items():
            output_name = f"{group_name}_chief_complaints_preprocessed.parquet"
            output_path = OUTPUT_DIR / output_name

            print(f"Preprocessing {input_path.name}", flush=True)
            df = read_parquet(con, input_path)
            preprocessed = preprocess_frame(df, group_name, nlp)
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
