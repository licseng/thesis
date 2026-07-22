"""Classify diagnostic-overshadowing evidence in first-LLM-positive sections.

This script is a separate local/cluster LLM runner for the diagnostic
overshadowing classification task. It uses a two-stage input:

1. The filtered section text used by the psych-history classifier:
   ../05_classification_psych_history/psych_history_llm_input/
       filtered_psych_keyword_section_input.parquet

2. The section-level psych-history LLM output:
   ../05_classification_psych_history/psych_history_classifier_output/
       psych_history_section_classifier_results.csv

Only sections positively labeled by the first LLM are sent to this diagnostic
overshadowing classifier. The LLM output is written to a separate folder so it
never overwrites the psych-history classifier output.

Before a real run, replace `SYSTEM_PROMPT` below or provide a prompt text file
with `DIAGNOSTIC_OVERSHADOWING_PROMPT_PATH`. The prompt file is useful because
it lets you add in-context examples without editing this script.

The model backend is configurable. Use the default `ollama` backend for local
testing, or set `DIAGNOSTIC_OVERSHADOWING_BACKEND=openai_compatible` for a
hosted chat-completions API.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PSYCH_HISTORY_DIR = SCRIPT_DIR.parent / "05_classification_psych_history"
INPUT_PATH = Path(
    os.environ.get(
        "DIAGNOSTIC_OVERSHADOWING_FIRST_LLM_INPUT_PATH",
        str(
            PSYCH_HISTORY_DIR
            / "psych_history_llm_input"
            / "filtered_psych_keyword_section_input.parquet"
        ),
    )
)
PSYCH_HISTORY_RESULTS_PATH = Path(
    os.environ.get(
        "DIAGNOSTIC_OVERSHADOWING_FIRST_LLM_RESULTS_PATH",
        str(
            PSYCH_HISTORY_DIR
            / "psych_history_classifier_output_prompt_B"
            / "psych_history_section_classifier_results.csv"
        ),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "DIAGNOSTIC_OVERSHADOWING_OUTPUT_DIR",
        str(SCRIPT_DIR / "diagnostic_overshadowing_classifier_output"),
    )
)

BACKEND = os.environ.get("DIAGNOSTIC_OVERSHADOWING_BACKEND", "ollama").lower()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
API_BASE_URL = os.environ.get(
    "DIAGNOSTIC_OVERSHADOWING_API_BASE_URL",
    "https://llm.mlcloud.uni-tuebingen.de/v1",
)
API_URL_OVERRIDE = os.environ.get("DIAGNOSTIC_OVERSHADOWING_API_URL", "")
API_KEY = (
    os.environ.get("DIAGNOSTIC_OVERSHADOWING_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)
MODEL_NAME = os.environ.get(
    "DIAGNOSTIC_OVERSHADOWING_MODEL_NAME",
    "Qwen/Qwen3.6-35B-A3B",
)
REQUEST_TIMEOUT_SECONDS = int(
    os.environ.get("DIAGNOSTIC_OVERSHADOWING_REQUEST_TIMEOUT_SECONDS", "240")
)
MAX_NEW_TOKENS = int(os.environ.get("DIAGNOSTIC_OVERSHADOWING_MAX_NEW_TOKENS", "512"))
API_JSON_MODE = os.environ.get("DIAGNOSTIC_OVERSHADOWING_API_JSON_MODE", "1").lower() not in {
    "0",
    "false",
    "no",
}
API_DISABLE_THINKING = os.environ.get(
    "DIAGNOSTIC_OVERSHADOWING_API_DISABLE_THINKING",
    "1",
).lower() not in {"0", "false", "no"}
CHUNK_WORDS = int(os.environ.get("DIAGNOSTIC_OVERSHADOWING_CHUNK_WORDS", "800"))
CHUNK_OVERLAP_WORDS = int(
    os.environ.get("DIAGNOSTIC_OVERSHADOWING_CHUNK_OVERLAP_WORDS", "100")
)
SLEEP_BETWEEN_REQUESTS_SECONDS = float(
    os.environ.get("DIAGNOSTIC_OVERSHADOWING_SLEEP_SECONDS", "0.1")
)
PROMPT_PATH = os.environ.get("DIAGNOSTIC_OVERSHADOWING_PROMPT_PATH")
PSYCH_HISTORY_POSITIVE_LABEL = os.environ.get(
    "DIAGNOSTIC_OVERSHADOWING_FIRST_LLM_POSITIVE_LABEL",
    "positive",
)

# Pilot limit. Set `DIAGNOSTIC_OVERSHADOWING_MAX_NOTES=none` for all filtered
# admissions, or set an integer for a smoke test.
MAX_NOTES: int | None = 10
MAX_NOTES_OVERRIDE = os.environ.get("DIAGNOSTIC_OVERSHADOWING_MAX_NOTES")
if MAX_NOTES_OVERRIDE:
    MAX_NOTES = None if MAX_NOTES_OVERRIDE.lower() == "none" else int(MAX_NOTES_OVERRIDE)

METADATA_COLUMNS = ["cohort", "subject_id", "hadm_id", "note_id", "charttime"]
FIRST_STAGE_SECTION_KEY_COLUMNS = METADATA_COLUMNS + ["section_name"]

OLLAMA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnostic_overshadowing_label": {
            "type": "string",
            "enum": ["positive", "negative", "unclear"],
        },
        "psychiatric_context_present": {"type": "boolean"},
        "physical_or_medical_problem_present": {"type": "boolean"},
        "attribution_to_psychiatric_condition": {"type": "boolean"},
        "possible_missed_or_delayed_workup": {"type": "boolean"},
        "evidence_span": {"type": "string", "maxLength": 180},
        "reason": {"type": "string", "maxLength": 320},
    },
    "required": [
        "diagnostic_overshadowing_label",
        "psychiatric_context_present",
        "physical_or_medical_problem_present",
        "attribution_to_psychiatric_condition",
        "possible_missed_or_delayed_workup",
        "evidence_span",
        "reason",
    ],
}

SYSTEM_PROMPT = """You are a clinical text classification assistant.

Task:
Classify whether the discharge-note section contains evidence consistent with
diagnostic overshadowing: a physical or medical symptom/problem appears to be
attributed to psychiatric illness, behavior, substance use, or mental status in
a way that could plausibly reduce, delay, or redirect medical evaluation.

Use labels:
- positive: clear evidence of possible diagnostic overshadowing
- negative: no evidence of diagnostic overshadowing
- unclear: insufficient or ambiguous evidence

Return exactly this JSON schema without extra fields or comments:
{
  "diagnostic_overshadowing_label": "positive|negative|unclear",
  "psychiatric_context_present": true|false,
  "physical_or_medical_problem_present": true|false,
  "attribution_to_psychiatric_condition": true|false,
  "possible_missed_or_delayed_workup": true|false,
  "evidence_span": "",
  "reason": ""
}

Keep evidence span and reason brief. Maximum 2 sentences each. 
"""


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as h:mm:ss."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def estimate_remaining(elapsed_seconds: float, completed: int, total: int) -> str:
    """Estimate remaining time from completed units."""
    if completed <= 0 or total <= completed:
        return "0:00" if total <= completed else "unknown"
    rate = completed / elapsed_seconds if elapsed_seconds > 0 else 0
    if rate <= 0:
        return "unknown"
    return format_duration((total - completed) / rate)


def get_system_prompt() -> str:
    """Load prompt text from a file when configured, otherwise use placeholder."""
    if not PROMPT_PATH:
        return SYSTEM_PROMPT

    path = Path(PROMPT_PATH).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()


def api_chat_completions_url() -> str:
    """Return the OpenAI-compatible chat-completions endpoint."""
    if API_URL_OVERRIDE:
        return API_URL_OVERRIDE
    return f"{API_BASE_URL.rstrip('/')}/chat/completions"


def validate_input_path() -> None:
    """Fail early if required first-stage inputs are absent."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input parquet: {INPUT_PATH}")
    if not PSYCH_HISTORY_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing psych-history classifier results: {PSYCH_HISTORY_RESULTS_PATH}"
        )


def validate_columns(df: pd.DataFrame) -> None:
    """Check that required metadata and section columns exist."""
    required = set(METADATA_COLUMNS + ["classifier_row_id", "section_name", "section_text"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_PATH}: {missing}")


def validate_psych_history_result_columns(df: pd.DataFrame) -> None:
    """Check that first-stage classifier output has required labels."""
    required = set(
        FIRST_STAGE_SECTION_KEY_COLUMNS
        + [
            "classifier_row_id",
            "psychiatric_context_label",
        ]
    )
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {PSYCH_HISTORY_RESULTS_PATH}: {missing}"
        )


def normalize_first_stage_section_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize section identity columns before merging first-stage outputs."""
    df = df.copy()
    for column in FIRST_STAGE_SECTION_KEY_COLUMNS:
        if column == "charttime":
            df[column] = pd.to_datetime(df[column], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
        elif column == "section_name":
            df[column] = df[column].fillna("").astype(str).str.strip()
    return df


def load_section_rows() -> pd.DataFrame:
    """Load first-LLM-positive section rows for diagnostic overshadowing."""
    section_text = pd.read_parquet(INPUT_PATH)
    validate_columns(section_text)
    section_text = normalize_first_stage_section_keys(section_text)

    first_stage_results = pd.read_csv(PSYCH_HISTORY_RESULTS_PATH)
    validate_psych_history_result_columns(first_stage_results)
    first_stage_results = normalize_first_stage_section_keys(first_stage_results)

    first_stage_label_columns = [
        "classifier_row_id",
        *FIRST_STAGE_SECTION_KEY_COLUMNS,
        "psychiatric_context_label",
        "psychiatric_mention_type",
        "n_chunks",
        "n_positive_chunks",
        "n_negative_chunks",
        "model_name",
    ]
    available_first_stage_columns = [
        column for column in first_stage_label_columns if column in first_stage_results.columns
    ]

    positive_results = first_stage_results.loc[
        first_stage_results["psychiatric_context_label"].astype(str).str.lower().eq(
            PSYCH_HISTORY_POSITIVE_LABEL.lower()
        ),
        available_first_stage_columns,
    ].copy()
    positive_results = positive_results.rename(
        columns={
            "classifier_row_id": "first_llm_classifier_row_id",
            "psychiatric_context_label": "first_llm_psychiatric_context_label",
            "psychiatric_mention_type": "first_llm_psychiatric_mention_type",
            "n_chunks": "first_llm_n_chunks",
            "n_positive_chunks": "first_llm_n_positive_chunks",
            "n_negative_chunks": "first_llm_n_negative_chunks",
            "model_name": "first_llm_model_name",
        }
    )

    df = section_text.merge(
        positive_results,
        on=FIRST_STAGE_SECTION_KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if positive_results.empty:
        raise ValueError(
            f"No first-stage rows with {PSYCH_HISTORY_POSITIVE_LABEL=} in "
            f"{PSYCH_HISTORY_RESULTS_PATH}"
        )
    if df.empty:
        raise ValueError(
            "No rows matched between first-stage section text input and "
            "first-stage positive results. Check that "
            "DIAGNOSTIC_OVERSHADOWING_FIRST_LLM_INPUT_PATH matches the input used "
            f"to create {PSYCH_HISTORY_RESULTS_PATH}."
        )

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
    df = df.rename(columns={"classifier_row_id": "source_input_classifier_row_id"})
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


def build_messages(
    row: pd.Series,
    chunk_text: str,
    chunk_index: int,
    n_chunks: int,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Build chat messages for one section chunk."""
    user_prompt = f"""/no_think

Section name: {row['section_name']}
Chunk: {chunk_index + 1} of {n_chunks}

Section chunk text:
{chunk_text}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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


def call_openai_compatible(payload: dict[str, Any]) -> dict[str, Any]:
    """Send one request to an OpenAI-compatible chat-completions endpoint."""
    if not API_KEY:
        raise RuntimeError("Set DIAGNOSTIC_OVERSHADOWING_API_KEY before API runs.")

    request = urllib.request.Request(
        api_chat_completions_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed with HTTP {exc.code}: {body}") from exc


def check_model_available() -> None:
    """Fail early if the configured model backend is not reachable."""
    if BACKEND == "openai_compatible":
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Return JSON only: {\"ok\": true}"}
            ],
            "temperature": 0,
            "max_tokens": 16,
        }
        if API_JSON_MODE:
            payload["response_format"] = {"type": "json_object"}
        if API_DISABLE_THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            call_openai_compatible(payload)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach the configured diagnostic-overshadowing API. "
                "Check DIAGNOSTIC_OVERSHADOWING_API_URL and API access."
            ) from exc
        return

    if BACKEND != "ollama":
        raise ValueError(
            "DIAGNOSTIC_OVERSHADOWING_BACKEND must be 'ollama' or "
            "'openai_compatible'."
        )

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


def generate_response(messages: list[dict[str, str]]) -> str:
    """Generate a JSON response for one section with the configured model."""
    if BACKEND == "openai_compatible":
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0,
            "max_tokens": MAX_NEW_TOKENS,
        }
        if API_JSON_MODE:
            payload["response_format"] = {"type": "json_object"}
        if API_DISABLE_THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        response = call_openai_compatible(payload)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"API response did not include choices: {response}")
        message = choices[0].get("message", {})
        return str(message.get("content", "")).strip()

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
    label = str(result.get("diagnostic_overshadowing_label", "")).strip().lower()
    if label not in {"positive", "negative", "unclear"}:
        label = "unclear"

    return {
        "diagnostic_overshadowing_label": label,
        "psychiatric_context_present": bool(
            result.get("psychiatric_context_present", False)
        ),
        "physical_or_medical_problem_present": bool(
            result.get("physical_or_medical_problem_present", False)
        ),
        "attribution_to_psychiatric_condition": bool(
            result.get("attribution_to_psychiatric_condition", False)
        ),
        "possible_missed_or_delayed_workup": bool(
            result.get("possible_missed_or_delayed_workup", False)
        ),
        "evidence_span": str(result.get("evidence_span", "")).strip(),
        "reason": str(result.get("reason", "")).strip(),
    }


def combine_chunk_results(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse chunk-level labels into one section-level label."""
    labels = [result["diagnostic_overshadowing_label"] for result in chunk_results]
    if "positive" in labels:
        chosen_label = "positive"
    elif "unclear" in labels:
        chosen_label = "unclear"
    else:
        chosen_label = "negative"

    chosen_result = next(
        result
        for result in chunk_results
        if result["diagnostic_overshadowing_label"] == chosen_label
    )
    return {
        **chosen_result,
        "diagnostic_overshadowing_label": chosen_label,
        "psychiatric_context_present": any(
            result["psychiatric_context_present"] for result in chunk_results
        ),
        "physical_or_medical_problem_present": any(
            result["physical_or_medical_problem_present"] for result in chunk_results
        ),
        "attribution_to_psychiatric_condition": any(
            result["attribution_to_psychiatric_condition"] for result in chunk_results
        ),
        "possible_missed_or_delayed_workup": any(
            result["possible_missed_or_delayed_workup"] for result in chunk_results
        ),
        "n_chunks": len(chunk_results),
        "n_positive_chunks": labels.count("positive"),
        "n_negative_chunks": labels.count("negative"),
        "n_unclear_chunks": labels.count("unclear"),
    }


def classify_sections(
    section_rows: pd.DataFrame,
    system_prompt: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the local model over section chunks and return chunk/section labels."""
    output_rows = []
    chunk_output_rows = []
    total = len(section_rows)
    section_start_time = time.monotonic()
    total_chunks = int(
        section_rows["section_text"].map(
            lambda text: len(split_text_into_chunks(str(text)))
        ).sum()
    )
    completed_chunks = 0

    for index, row in section_rows.iterrows():
        if index == 0 or (index + 1) % 10 == 0 or (index + 1) == total:
            completed_sections = index
            elapsed = time.monotonic() - section_start_time
            section_rate = 60.0 * completed_sections / elapsed if elapsed > 0 else 0.0
            print(
                f"Classifying section {index + 1}/{total} | "
                f"elapsed {format_duration(elapsed)} | "
                f"rate {section_rate:.2f} sections/min | "
                f"ETA {estimate_remaining(elapsed, completed_sections, total)} | "
                f"chunks {completed_chunks}/{total_chunks}",
                flush=True,
            )

        chunks = split_text_into_chunks(row["section_text"])
        chunk_results = []
        for chunk_index, chunk_text in enumerate(chunks):
            print(
                f"  chunk {chunk_index + 1}/{len(chunks)} for {row['section_name']}",
                flush=True,
            )
            messages = build_messages(
                row,
                chunk_text,
                chunk_index,
                len(chunks),
                system_prompt,
            )
            response_text = generate_response(messages)
            raw_result = parse_json_response(response_text)
            normalized = normalize_model_result(raw_result)
            chunk_results.append(normalized)
            chunk_output_rows.append(
                {
                    "classifier_row_id": int(row["classifier_row_id"]),
                    "first_llm_classifier_row_id": int(row["first_llm_classifier_row_id"]),
                    "cohort": row["cohort"],
                    "subject_id": row["subject_id"],
                    "hadm_id": row["hadm_id"],
                    "note_id": row["note_id"],
                    "charttime": row["charttime"],
                    "section_name": row["section_name"],
                    "n_psych_keyword_hits": row.get("n_psych_keyword_hits", None),
                    "psych_keyword_groups": row.get("psych_keyword_groups", ""),
                    "matched_terms": row.get("matched_terms", ""),
                    "first_llm_psychiatric_context_label": row.get(
                        "first_llm_psychiatric_context_label",
                        "",
                    ),
                    "first_llm_psychiatric_mention_type": row.get(
                        "first_llm_psychiatric_mention_type",
                        "",
                    ),
                    "chunk_index": chunk_index,
                    "n_chunks": len(chunks),
                    "chunk_word_count": len(chunk_text.split()),
                    **normalized,
                    "model_name": MODEL_NAME,
                }
            )
            time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)
            completed_chunks += 1

        section_result = combine_chunk_results(chunk_results)
        output_rows.append(
            {
                "classifier_row_id": int(row["classifier_row_id"]),
                "first_llm_classifier_row_id": int(row["first_llm_classifier_row_id"]),
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
                "first_llm_psychiatric_context_label": row.get(
                    "first_llm_psychiatric_context_label",
                    "",
                ),
                "first_llm_psychiatric_mention_type": row.get(
                    "first_llm_psychiatric_mention_type",
                    "",
                ),
                **section_result,
                "model_name": MODEL_NAME,
            }
        )

    elapsed = time.monotonic() - section_start_time
    print(
        f"Finished classifying {total} sections and {completed_chunks} chunks "
        f"in {format_duration(elapsed)} "
        f"({60.0 * total / elapsed:.2f} sections/min).",
        flush=True,
    )

    return pd.DataFrame(output_rows), pd.DataFrame(chunk_output_rows)


def write_outputs(
    results: pd.DataFrame,
    chunk_results: pd.DataFrame,
    run_metadata: dict[str, Any],
) -> None:
    """Write section-level classifier results and compact summaries."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_parquet(
        OUTPUT_DIR / "diagnostic_overshadowing_section_classifier_results.parquet",
        index=False,
    )
    results.to_csv(
        OUTPUT_DIR / "diagnostic_overshadowing_section_classifier_results.csv",
        index=False,
    )
    chunk_results.to_parquet(
        OUTPUT_DIR / "diagnostic_overshadowing_section_chunk_classifier_results.parquet",
        index=False,
    )
    chunk_results.to_csv(
        OUTPUT_DIR / "diagnostic_overshadowing_section_chunk_classifier_results.csv",
        index=False,
    )

    with (
        OUTPUT_DIR / "diagnostic_overshadowing_section_classifier_results.jsonl"
    ).open("w") as handle:
        for row in results.to_dict(orient="records"):
            handle.write(json.dumps(row, default=str, ensure_ascii=True) + "\n")

    with (OUTPUT_DIR / "diagnostic_overshadowing_run_metadata.json").open("w") as handle:
        json.dump(run_metadata, handle, indent=2, default=str)

    label_summary = (
        results.groupby(
            ["cohort", "section_name", "diagnostic_overshadowing_label"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_sections"})
        .sort_values(["cohort", "section_name", "diagnostic_overshadowing_label"])
    )
    label_summary.to_csv(
        OUTPUT_DIR / "diagnostic_overshadowing_section_label_summary.csv",
        index=False,
    )

    admission_summary = (
        results.assign(
            is_positive=results["diagnostic_overshadowing_label"].eq("positive"),
            is_unclear=results["diagnostic_overshadowing_label"].eq("unclear"),
        )
        .groupby(["cohort", "subject_id", "hadm_id"], as_index=False)
        .agg(
            n_sections_classified=("section_name", "size"),
            n_positive_sections=("is_positive", "sum"),
            n_unclear_sections=("is_unclear", "sum"),
            any_positive=("is_positive", "any"),
            any_unclear=("is_unclear", "any"),
        )
    )
    admission_summary.to_csv(
        OUTPUT_DIR / "diagnostic_overshadowing_admission_summary.csv",
        index=False,
    )

    print(f"Saved diagnostic-overshadowing classifier outputs to: {OUTPUT_DIR}")
    print(label_summary.to_string(index=False))


def main() -> None:
    """Run the local Ollama classifier over filtered note sections."""
    run_started_at = datetime.now().isoformat(timespec="seconds")
    run_start_time = time.monotonic()
    validate_input_path()
    section_rows = load_section_rows()
    system_prompt = get_system_prompt()

    print(
        f"Loaded {len(section_rows)} non-empty first-LLM-positive section rows "
        f"from: {INPUT_PATH}"
    )
    print(f"First-stage classifier results: {PSYCH_HISTORY_RESULTS_PATH}")
    print(f"First-stage positive label: {PSYCH_HISTORY_POSITIVE_LABEL}")
    print(f"Using diagnostic-overshadowing backend: {BACKEND}")
    print(f"Using diagnostic-overshadowing model: {MODEL_NAME}")
    if BACKEND == "openai_compatible":
        print(f"Using API URL: {api_chat_completions_url()}")
        print(f"Using API JSON mode: {API_JSON_MODE}")
        print(f"Using API disable thinking: {API_DISABLE_THINKING}")
    print(f"Chunking sections at {CHUNK_WORDS} words with {CHUNK_OVERLAP_WORDS} word overlap")
    print(f"Run started at: {run_started_at}")
    if PROMPT_PATH:
        print(f"Using diagnostic-overshadowing prompt file: {PROMPT_PATH}")
    else:
        print("Using placeholder diagnostic-overshadowing prompt embedded in script.")

    check_model_available()
    results, chunk_results = classify_sections(section_rows, system_prompt)
    elapsed_seconds = time.monotonic() - run_start_time
    run_metadata = {
        "run_started_at": run_started_at,
        "run_finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed": format_duration(elapsed_seconds),
        "backend": BACKEND,
        "model_name": MODEL_NAME,
        "ollama_url": OLLAMA_URL,
        "api_base_url": API_BASE_URL,
        "api_url": api_chat_completions_url(),
        "api_json_mode": API_JSON_MODE,
        "api_disable_thinking": API_DISABLE_THINKING,
        "input_path": str(INPUT_PATH),
        "psych_history_results_path": str(PSYCH_HISTORY_RESULTS_PATH),
        "psych_history_positive_label": PSYCH_HISTORY_POSITIVE_LABEL,
        "output_dir": str(OUTPUT_DIR),
        "prompt_path": PROMPT_PATH,
        "max_notes": MAX_NOTES,
        "chunk_words": CHUNK_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "n_section_rows": len(results),
        "n_chunk_rows": len(chunk_results),
        "n_admissions": int(
            results[["cohort", "subject_id", "hadm_id"]].drop_duplicates().shape[0]
        ),
    }
    write_outputs(results, chunk_results, run_metadata)
    print(f"Total runtime: {run_metadata['elapsed']}")


if __name__ == "__main__":
    main()
