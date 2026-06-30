"""Classify psychotic-disorder history mentions in parsed note sections.

This is WP2 of the thesis. The ICD cohort definition already establishes prior
psychotic disorder history for the exposed group. This script asks a different
question: whether that history is surfaced/documented in the current index
admission discharge note.

The classifier runs locally through Ollama. It calls the local Ollama HTTP
server only; it does not send note text to an external API.

Pilot behavior:
    `MAX_NOTES = 10` limits the number of discharge notes in the pilot. Set it
    to `None` to process every note in the input table.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    SCRIPT_DIR
    / "psych_history_llm_input"
    / "filtered_psych_keyword_section_input.parquet"
)
OUTPUT_DIR = SCRIPT_DIR / "psych_history_classifier_output"

# Ollama model settings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.environ.get("PSYCH_HISTORY_MODEL_NAME", "qwen3:4b")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("PSYCH_HISTORY_REQUEST_TIMEOUT_SECONDS", "180"))
MAX_NEW_TOKENS = int(os.environ.get("PSYCH_HISTORY_MAX_NEW_TOKENS", "384"))
CHUNK_WORDS = int(os.environ.get("PSYCH_HISTORY_CHUNK_WORDS", "600"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("PSYCH_HISTORY_CHUNK_OVERLAP_WORDS", "75"))
SLEEP_BETWEEN_REQUESTS_SECONDS = 0.1

# Pilot limit. Set to None to process every admission in the input table.
MAX_NOTES: int | None = 10
MAX_NOTES_OVERRIDE = os.environ.get("PSYCH_HISTORY_MAX_NOTES")
if MAX_NOTES_OVERRIDE:
    MAX_NOTES = None if MAX_NOTES_OVERRIDE.lower() == "none" else int(MAX_NOTES_OVERRIDE)

METADATA_COLUMNS = [
    "cohort",
    "subject_id",
    "hadm_id",
    "note_id",
    "charttime",
]

OLLAMA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "psychosis_related_context_label": {
            "type": "string",
            "enum": ["positive", "negative"],
        },
        "other_psychiatric_context_label": {
            "type": "string",
            "enum": ["positive", "negative"],
        },
        "disorder_type": {
            "type": "string",
            "enum": [
                "schizophrenia",
                "schizoaffective disorder",
                "psychotic disorder",
                "bipolar disorder with psychotic features",
                "other psychiatric",
                "substance use disorder",
                "none",
                "unclear",
            ],
        },
        "negated_only": {"type": "boolean"},
        "family_history_only": {"type": "boolean"},
        "patient_specific": {"type": "boolean"},
        "evidence_span": {"type": "string", "maxLength": 160},
        "reason": {"type": "string", "maxLength": 240},
    },
    "required": [
        "psychosis_related_context_label",
        "other_psychiatric_context_label",
        "disorder_type",
        "negated_only",
        "family_history_only",
        "patient_specific",
        "evidence_span",
        "reason",
    ],
}

SYSTEM_PROMPT = """You are a clinical text classification assistant.

Context:
The provided discharge-note section has already been selected by keyword matching
because it contains one or more psychiatry-related terms. Keyword matches can be
false positives. Your role is to decide whether the section truly contains
patient-specific psychiatric context.

Task:
Classify whether the provided discharge-note section mentions psychiatric context
for the patient.

Use two separate flags:
1. psychosis_related_context:
   Use "yes" if the section mentions psychosis-related context, such as schizophrenia,
   schizoaffective disorder, psychotic disorder, hallucinations, delusions, paranoia,
   or antipsychotic treatment clearly related to psychosis.
2. other_psychiatric_context:
   Use "yes" if the section mentions other psychiatric context that is not psychosis-related,
   such as anxiety, depression, bipolar disorder without psychotic features, PTSD,
   personality disorder, suicidal ideation, psychiatric admission, psychiatry consult,
   substance use disorder, withdrawal, behavioral framing, or psychiatric medication
   not clearly related to psychosis.

Do NOT count as positive:
- negated history or symptoms only, e.g. "no history of psychosis", "denies hallucinations"
- family history only
- psychiatric words referring only to someone other than the patient
- keyword matches that are not clinically meaningful psychiatric context

Use label values:
- positive
- negative

Return exactly this JSON schema without including any extra fields or comments.
Keep evidence_span short, preferably 15 words or fewer. 
Keep reason brief, one sentence maximum.
{

  "psychosis_related_context_label": "positive|negative",
  "other_psychiatric_context_label": "positive|negative",
  "disorder_type": "schizophrenia|schizoaffective disorder|psychotic disorder|bipolar disorder with psychotic features|other psychiatric|substance use disorder|none|unclear",
  "negated_only": true|false,
  "family_history_only": true|false,
  "patient_specific": true|false,
  "evidence_span": "",
  "reason": ""
}
"""


def validate_input_path() -> None:
    """Fail early if the configured parsed section table is absent."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input parquet: {INPUT_PATH}")


def validate_columns(df: pd.DataFrame) -> None:
    """Check that required metadata and section columns exist."""
    required = set(METADATA_COLUMNS + ["section_name", "section_text"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_PATH}: {missing}")


def load_section_rows() -> pd.DataFrame:
    """Load keyword-filtered section rows for LLM classification."""
    df = pd.read_parquet(INPUT_PATH)
    validate_columns(df)
    if MAX_NOTES is not None:
        admission_keys = df.loc[:, ["cohort", "subject_id", "hadm_id"]].drop_duplicates()
        selected_keys = admission_keys.head(MAX_NOTES)
        df = df.merge(
            selected_keys,
            on=["cohort", "subject_id", "hadm_id"],
            how="inner",
            validate="many_to_one",
        )

    df = df.copy().reset_index(drop=True)
    df["section_text"] = df["section_text"].fillna("").astype(str).str.strip()
    df = df.loc[df["section_text"].ne("")].copy()
    if "classifier_row_id" in df.columns:
        df = df.drop(columns=["classifier_row_id"])
    df.insert(0, "classifier_row_id", range(len(df)))
    return df


def split_text_into_chunks(text: str) -> list[str]:
    """Split full section text into overlapping word chunks for local LLM calls."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text]

    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP_WORDS)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + CHUNK_WORDS]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def build_messages(row: pd.Series, chunk_text: str, chunk_index: int, n_chunks: int) -> list[dict[str, str]]:
    """Build chat messages for one section chunk."""
    user_prompt = f"""/no_think

Section name: {row['section_name']}
Chunk: {chunk_index + 1} of {n_chunks}

Section chunk text:
{chunk_text}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def check_ollama_available() -> None:
    """Fail early if the local Ollama server/model is not reachable."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Return JSON only: {\"ok\": true}"}],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"num_predict": 16, "temperature": 0},
    }
    try:
        call_ollama(payload)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Start it with the Ollama app or `ollama serve`, "
            f"then make sure the model exists with `ollama pull {MODEL_NAME}`."
        ) from exc


def call_ollama(payload: dict[str, Any]) -> dict[str, Any]:
    """Send one request to the local Ollama chat endpoint."""
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {body}") from exc


def generate_response(messages: list[dict[str, str]]) -> str:
    """Generate a JSON response for one section with the local Ollama model."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "format": OLLAMA_RESPONSE_SCHEMA,
        "think": False,
        "options": {
            "num_predict": MAX_NEW_TOKENS,
            "temperature": 0,
        },
    }
    response = call_ollama(payload)
    message = response.get("message", {})
    return str(message.get("content", "")).strip()


def parse_json_response(response_text: str) -> dict[str, Any]:
    """Parse a model response as JSON, with a fallback for wrapped JSON text."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", response_text)
    if not match:
        raise ValueError(f"Model returned no JSON object: {response_text}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {response_text}") from exc


def normalize_model_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize optional or malformed model fields into stable output columns."""
    psychosis_label = str(result.get("psychosis_related_context_label", "")).strip().lower()
    other_label = str(result.get("other_psychiatric_context_label", "")).strip().lower()
    if psychosis_label not in {"positive", "negative"}:
        psychosis_label = "negative"
    if other_label not in {"positive", "negative"}:
        other_label = "negative"

    label = (
        "positive"
        if psychosis_label == "positive" or other_label == "positive"
        else "negative"
    )

    return {
        "psychosis_related_context_label": psychosis_label,
        "other_psychiatric_context_label": other_label,
        "label": label,
        "disorder_type": str(result.get("disorder_type", "unclear")).strip(),
        "negated_only": bool(result.get("negated_only", False)),
        "family_history_only": bool(result.get("family_history_only", False)),
        "patient_specific": bool(result.get("patient_specific", True)),
        "evidence_span": str(result.get("evidence_span", "")).strip(),
        "reason": str(result.get("reason", "")).strip(),
    }


def combine_chunk_results(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse chunk-level labels into one section-level label."""
    labels = [result["label"] for result in chunk_results]
    psychosis_labels = [result["psychosis_related_context_label"] for result in chunk_results]
    other_labels = [result["other_psychiatric_context_label"] for result in chunk_results]
    if "positive" in labels:
        chosen_label = "positive"
    else:
        chosen_label = "negative"

    chosen_result = next(
        result for result in chunk_results if result["label"] == chosen_label
    )
    return {
        **chosen_result,
        "psychosis_related_context_label": (
            "positive" if "positive" in psychosis_labels else "negative"
        ),
        "other_psychiatric_context_label": (
            "positive" if "positive" in other_labels else "negative"
        ),
        "label": chosen_label,
        "n_chunks": len(chunk_results),
        "n_positive_chunks": labels.count("positive"),
        "n_negative_chunks": labels.count("negative"),
        "n_psychosis_positive_chunks": psychosis_labels.count("positive"),
        "n_other_psychiatric_positive_chunks": other_labels.count("positive"),
    }


def classify_sections(
    section_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the local model over section chunks and return chunk/section labels."""
    output_rows = []
    chunk_output_rows = []
    total = len(section_rows)

    for index, row in section_rows.iterrows():
        if index == 0 or (index + 1) % 10 == 0 or (index + 1) == total:
            print(f"Classifying section {index + 1}/{total}", flush=True)

        chunks = split_text_into_chunks(row["section_text"])
        chunk_results = []
        for chunk_index, chunk_text in enumerate(chunks):
            print(
                f"  chunk {chunk_index + 1}/{len(chunks)} "
                f"for {row['section_name']}",
                flush=True,
            )
            messages = build_messages(row, chunk_text, chunk_index, len(chunks))
            response_text = generate_response(messages)
            raw_result = parse_json_response(response_text)
            normalized = normalize_model_result(raw_result)
            chunk_results.append(normalized)
            chunk_output_rows.append(
                {
                    "classifier_row_id": int(row["classifier_row_id"]),
                    "cohort": row["cohort"],
                    "subject_id": row["subject_id"],
                    "hadm_id": row["hadm_id"],
                    "note_id": row["note_id"],
                    "charttime": row["charttime"],
                    "section_name": row["section_name"],
                    "n_psych_keyword_hits": row.get("n_psych_keyword_hits", None),
                    "psych_keyword_groups": row.get("psych_keyword_groups", ""),
                    "matched_terms": row.get("matched_terms", ""),
                    "chunk_index": chunk_index,
                    "n_chunks": len(chunks),
                    "chunk_word_count": len(chunk_text.split()),
                    **normalized,
                    "model_name": MODEL_NAME,
                }
            )
            time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)

        section_result = combine_chunk_results(chunk_results)

        output_row = {
            "classifier_row_id": int(row["classifier_row_id"]),
            "cohort": row["cohort"],
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "note_id": row["note_id"],
            "charttime": row["charttime"],
            "section_name": row["section_name"],
            "section_char_length": len(row["section_text"]),
            "section_word_count": len(row["section_text"].split()),
            "n_psych_keyword_hits": row.get("n_psych_keyword_hits", None),
            "psych_keyword_groups": row.get("psych_keyword_groups", ""),
            "matched_terms": row.get("matched_terms", ""),
            **section_result,
            "model_name": MODEL_NAME,
        }
        output_rows.append(output_row)

    return pd.DataFrame(output_rows), pd.DataFrame(chunk_output_rows)


def write_outputs(results: pd.DataFrame, chunk_results: pd.DataFrame) -> None:
    """Write section-level classifier results and compact summaries."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUTPUT_DIR / "psych_history_section_classifier_results.parquet", index=False)
    results.to_csv(OUTPUT_DIR / "psych_history_section_classifier_results.csv", index=False)
    chunk_results.to_parquet(
        OUTPUT_DIR / "psych_history_section_chunk_classifier_results.parquet", index=False
    )
    chunk_results.to_csv(
        OUTPUT_DIR / "psych_history_section_chunk_classifier_results.csv", index=False
    )

    with (OUTPUT_DIR / "psych_history_section_classifier_results.jsonl").open("w") as handle:
        for row in results.to_dict(orient="records"):
            handle.write(json.dumps(row, default=str, ensure_ascii=True) + "\n")

    label_summary = (
        results.groupby(
            [
                "cohort",
                "section_name",
                "psychosis_related_context_label",
                "other_psychiatric_context_label",
                "label",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_sections"})
        .sort_values(
            [
                "cohort",
                "section_name",
                "psychosis_related_context_label",
                "other_psychiatric_context_label",
                "label",
            ]
        )
    )
    label_summary.to_csv(OUTPUT_DIR / "psych_history_section_label_summary.csv", index=False)

    admission_summary = (
        results.assign(
            is_positive=results["label"].eq("positive"),
            is_psychosis_positive=results["psychosis_related_context_label"].eq("positive"),
            is_other_psychiatric_positive=results[
                "other_psychiatric_context_label"
            ].eq("positive"),
        )
        .groupby(["cohort", "subject_id", "hadm_id"], as_index=False)
        .agg(
            n_sections_classified=("section_name", "size"),
            n_positive_sections=("is_positive", "sum"),
            n_psychosis_positive_sections=("is_psychosis_positive", "sum"),
            n_other_psychiatric_positive_sections=("is_other_psychiatric_positive", "sum"),
            any_positive=("is_positive", "any"),
            any_psychosis_positive=("is_psychosis_positive", "any"),
            any_other_psychiatric_positive=("is_other_psychiatric_positive", "any"),
        )
    )
    admission_summary.to_csv(OUTPUT_DIR / "psych_history_admission_summary.csv", index=False)

    print(f"Saved classifier outputs to: {OUTPUT_DIR}")
    print(label_summary.to_string(index=False))


def main() -> None:
    """Run the local Ollama classifier over parsed note sections."""
    validate_input_path()
    section_rows = load_section_rows()
    print(f"Loaded {len(section_rows)} non-empty section rows from: {INPUT_PATH}")
    print(f"Using local Ollama model: {MODEL_NAME}")
    print(f"Chunking sections at {CHUNK_WORDS} words with {CHUNK_OVERLAP_WORDS} word overlap")
    check_ollama_available()
    results, chunk_results = classify_sections(section_rows)
    write_outputs(results, chunk_results)


if __name__ == "__main__":
    main()
