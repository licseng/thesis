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
    SCRIPT_DIR.parent
    / "01_discharge_note_preprocessing"
    / "01_discharge_note_parsing"
    / "full_discharge_note_sections"
    / "MHH1_psychotic_matched_full_discharge_note_sections.parquet"
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

# Parsed section columns to classify. `full_note_text` is intentionally excluded
# because this script is section-level. These high-yield sections were selected
# from the psych keyword exploration to reduce local LLM runtime.
SECTION_COLUMNS = [
    "present_illness",
    "brief_hospital_course",
    "problems",
    "discharge_diagnosis",
]

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
        "mentions_psychotic_disorder_history": {
            "type": "string",
            "enum": ["yes", "no", "ambiguous"],
        },
        "label": {
            "type": "string",
            "enum": ["positive", "negative", "ambiguous"],
        },
        "disorder_type": {
            "type": "string",
            "enum": [
                "schizophrenia",
                "schizoaffective disorder",
                "psychotic disorder",
                "bipolar disorder with psychotic features",
                "other",
                "none",
                "unclear",
            ],
        },
        "temporality": {
            "type": "string",
            "enum": [
                "past_history",
                "existing_history",
                "current_admission_only",
                "family_history_only",
                "negated",
                "unclear",
                "none",
            ],
        },
        "negated": {"type": "boolean"},
        "evidence_span": {"type": "string", "maxLength": 160},
        "reason": {"type": "string", "maxLength": 240},
    },
    "required": [
        "mentions_psychotic_disorder_history",
        "label",
        "disorder_type",
        "temporality",
        "negated",
        "evidence_span",
        "reason",
    ],
}

SYSTEM_PROMPT = """You are a clinical text classification assistant.

Task:
Classify whether the provided discharge-note section mentions that the patient
has a prior or existing history of any psychiatric condition. That is the positive case.

Do NOT count as positive:
- negated history only, e.g. "no history of psychosis"
- family history only

Use label values:
- positive
- negative
- ambiguous

Use ambiguous when psychiatric history is hinted at but not explicit.

Return exactly this JSON schema without including any extra fields or comments.
Keep evidence_span short, preferably 15 words or fewer. If the label is
negative and there is no direct evidence phrase, use an empty evidence_span.
Keep reason brief, one sentence maximum.
{
  "mentions_psychotic_disorder_history": "yes|no|ambiguous",
  "label": "positive|negative|ambiguous",
  "disorder_type": "schizophrenia|schizoaffective disorder|psychotic disorder|bipolar disorder with psychotic features|other|none|unclear",
  "temporality": "past_history|existing_history|current_admission_only|family_history_only|negated|unclear|none",
  "negated": true,
}
"""


def validate_input_path() -> None:
    """Fail early if the configured parsed section table is absent."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input parquet: {INPUT_PATH}")


def validate_columns(df: pd.DataFrame) -> None:
    """Check that required metadata and section columns exist."""
    required = set(METADATA_COLUMNS + SECTION_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_PATH}: {missing}")


def load_section_rows() -> pd.DataFrame:
    """Load the parsed table and reshape non-empty sections to long format."""
    df = pd.read_parquet(INPUT_PATH)
    validate_columns(df)
    if MAX_NOTES is not None:
        df = df.head(MAX_NOTES).copy()

    rows = []
    for _, source_row in df.iterrows():
        metadata = {column: source_row[column] for column in METADATA_COLUMNS}
        for section in SECTION_COLUMNS:
            section_text = str(source_row.get(section, "") or "").strip()
            if section_text:
                rows.append(
                    {
                        **metadata,
                        "section_name": section,
                        "section_text": section_text,
                    }
                )

    long_df = pd.DataFrame(rows).reset_index(drop=True)
    long_df.insert(0, "classifier_row_id", range(len(long_df)))
    return long_df


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
    label = str(result.get("label", "")).strip().lower()
    mentions = str(result.get("mentions_psychotic_disorder_history", "")).strip().lower()

    if label not in {"positive", "negative", "ambiguous"}:
        label = "ambiguous"
    if mentions not in {"yes", "no", "ambiguous"}:
        mentions = {"positive": "yes", "negative": "no"}.get(label, "ambiguous")

    return {
        "mentions_psychotic_disorder_history": mentions,
        "label": label,
        "disorder_type": str(result.get("disorder_type", "unclear")).strip(),
        "temporality": str(result.get("temporality", "unclear")).strip(),
        "negated": bool(result.get("negated", False)),
        "evidence_span": str(result.get("evidence_span", "")).strip(),
        "reason": str(result.get("reason", "")).strip(),
    }


def combine_chunk_results(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse chunk-level labels into one section-level label."""
    labels = [result["label"] for result in chunk_results]
    if "positive" in labels:
        chosen_label = "positive"
    elif "ambiguous" in labels:
        chosen_label = "ambiguous"
    else:
        chosen_label = "negative"

    chosen_result = next(
        result for result in chunk_results if result["label"] == chosen_label
    )
    mentions = {"positive": "yes", "negative": "no"}.get(chosen_label, "ambiguous")
    return {
        **chosen_result,
        "mentions_psychotic_disorder_history": mentions,
        "label": chosen_label,
        "n_chunks": len(chunk_results),
        "n_positive_chunks": labels.count("positive"),
        "n_ambiguous_chunks": labels.count("ambiguous"),
        "n_negative_chunks": labels.count("negative"),
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
        results.groupby(["cohort", "section_name", "label"], as_index=False)
        .size()
        .rename(columns={"size": "n_sections"})
        .sort_values(["cohort", "section_name", "label"])
    )
    label_summary.to_csv(OUTPUT_DIR / "psych_history_section_label_summary.csv", index=False)

    admission_summary = (
        results.assign(is_positive=results["label"].eq("positive"))
        .groupby(["cohort", "subject_id", "hadm_id"], as_index=False)
        .agg(
            n_sections_classified=("section_name", "size"),
            n_positive_sections=("is_positive", "sum"),
            any_positive=("is_positive", "any"),
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
