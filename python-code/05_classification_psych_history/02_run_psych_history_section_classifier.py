"""Classify psychiatric-context mentions in parsed note sections.

This is WP2 of the thesis. The ICD cohort definition already establishes prior
psychotic disorder history for the exposed group. This script asks a different
question: whether psychiatric context is surfaced/documented
in the current index admission discharge note.

The classifier can run with two backends:
    - local Ollama, for laptop runs
    - local Hugging Face transformers, for GPU-cluster runs without Ollama

Neither backend sends note text to an external API.

Pilot behavior:
    `MAX_NOTES = 10` limits the number of discharge notes in the pilot. Set it
    to `None` to process every note in the input table.
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


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = (
    SCRIPT_DIR
    / "psych_history_llm_input"
    / "filtered_psych_keyword_section_input.parquet"
)
PROMPT_VERSION = os.environ.get("PSYCH_HISTORY_PROMPT_VERSION", "A").strip().upper()
if PROMPT_VERSION not in {"A", "B"}:
    raise ValueError("PSYCH_HISTORY_PROMPT_VERSION must be 'A' or 'B'.")
OUTPUT_DIR = Path(
    os.environ.get(
        "PSYCH_HISTORY_OUTPUT_DIR",
        str(SCRIPT_DIR / f"psych_history_classifier_output_prompt_{PROMPT_VERSION}"),
    )
)

# Model/backend settings
BACKEND = os.environ.get("PSYCH_HISTORY_BACKEND", "ollama").lower()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.environ.get("PSYCH_HISTORY_MODEL_NAME", "qwen3:4b")
HF_MODEL_ID = os.environ.get("PSYCH_HISTORY_HF_MODEL_ID", "Qwen/Qwen3-8B")
HF_DEVICE_MAP = os.environ.get("PSYCH_HISTORY_HF_DEVICE_MAP", "auto")
HF_TORCH_DTYPE = os.environ.get("PSYCH_HISTORY_HF_TORCH_DTYPE", "auto").lower()
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("PSYCH_HISTORY_REQUEST_TIMEOUT_SECONDS", "180"))
MAX_NEW_TOKENS = int(os.environ.get("PSYCH_HISTORY_MAX_NEW_TOKENS", "384"))
CHUNK_WORDS = int(os.environ.get("PSYCH_HISTORY_CHUNK_WORDS", "600"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("PSYCH_HISTORY_CHUNK_OVERLAP_WORDS", "75"))
BATCH_SIZE = int(os.environ.get("PSYCH_HISTORY_BATCH_SIZE", "1"))
CHECKPOINT_EVERY_SECTIONS = int(
    os.environ.get("PSYCH_HISTORY_CHECKPOINT_EVERY_SECTIONS", "500")
)
RESUME_FROM_CHECKPOINT = os.environ.get(
    "PSYCH_HISTORY_RESUME_FROM_CHECKPOINT",
    "1",
).lower() not in {"0", "false", "no"}
MAX_EVIDENCE_WORDS = int(os.environ.get("PSYCH_HISTORY_MAX_EVIDENCE_WORDS", "20"))
MAX_REASON_WORDS = int(os.environ.get("PSYCH_HISTORY_MAX_REASON_WORDS", "25"))
MAX_EVIDENCE_CHARS = int(os.environ.get("PSYCH_HISTORY_MAX_EVIDENCE_CHARS", "120"))
MAX_REASON_CHARS = int(os.environ.get("PSYCH_HISTORY_MAX_REASON_CHARS", "160"))
SLEEP_BETWEEN_REQUESTS_SECONDS = 0.1

# Pilot limit. Set to None to process every admission in the input table.
MAX_NOTES: int | None = 10
MAX_NOTES_OVERRIDE = os.environ.get("PSYCH_HISTORY_MAX_NOTES")
if MAX_NOTES_OVERRIDE:
    MAX_NOTES = None if MAX_NOTES_OVERRIDE.lower() == "none" else int(MAX_NOTES_OVERRIDE)

_TRANSFORMERS_MODEL: Any | None = None
_TRANSFORMERS_TOKENIZER: Any | None = None
_TRANSFORMERS_DIAGNOSTICS_PRINTED = False


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

METADATA_COLUMNS = [
    "cohort",
    "subject_id",
    "hadm_id",
    "note_id",
    "charttime",
]
SECTION_ID_COLUMNS = [
    "classifier_row_id",
    "cohort",
    "subject_id",
    "hadm_id",
    "note_id",
    "charttime",
    "section_name",
]

RESPONSE_SCHEMA_A = {
    "type": "object",
    "properties": {
        "psychiatric_context_label": {
            "type": "string",
            "enum": ["positive", "negative"],
        },
        "psychiatric_mention_type": {
            "type": "string",
            "enum": [
                "current_context",
                "background_only",
                "medication_only",
                "diagnosis_list_only",
                "negated_only",
                "family_history_only",
                "none",
            ],
        },
        "evidence_span": {"type": "string", "maxLength": 120},
        "reason": {"type": "string", "maxLength": 160},
    },
    "required": [
        "psychiatric_context_label",
        "psychiatric_mention_type",
        "evidence_span",
        "reason",
    ],
}

RESPONSE_SCHEMA_B = {
    "type": "object",
    "properties": {
        "psychiatric_context_integrated": {
            "type": "string",
            "enum": ["positive", "negative"],
        },
        "integration_type": {
            "type": "string",
            "enum": [
                "symptom_attribution",
                "diagnostic_reasoning",
                "management",
                "disposition_or_capacity",
                "behavior_or_reliability",
                "care_interference",
                "active_psychiatric_course",
                "disproportionate_detail",
                "incidental_history",
                "medication_only",
                "negated_or_irrelevant",
            ],
        },
        "evidence_span": {"type": "string", "maxLength": 120},
        "reason": {"type": "string", "maxLength": 160},
    },
    "required": [
        "psychiatric_context_integrated",
        "integration_type",
        "evidence_span",
        "reason",
    ],
}

SYSTEM_PROMPT_A = """You are a clinical text classification assistant.

Context:
The provided hospital discharge-note section has already been selected by keyword matching
because it contains one or more psychiatry-related terms. But keyword matches can be
false positives. 

Task:
Your role is to decide whether psychiatric context is connected explicitly or implicitly to the patient's
current admission, symptoms, hospital course, clinical reasoning, treatment
decisions, workup, consults, disposition, or care barriers.

Use "psychiatric_context_label": "positive" when the section does more than
list a psychiatric diagnosis, history, or medication. 
It can be either unnecessary detailed description of psychiatric context or 
the psychiatric mention is mixed into the current clinical context in a meaningful way.

Use label values:
- positive
- negative

Do NOT count as positive:
- standalone psychiatric diagnosis/history only, e.g. a problem list or past
  medical history mention with no connection to current care
- standalone psychiatric medication mention only, e.g. home medication listed
  with no current care decision or clinical reasoning
- negated psychiatric history or negated psychiatric symptoms, e.g. "denies hallucinations", negation of other physical symptoms is not relevant
- family psychiatric history only
- psychiatric words referring only to someone other than the patient

Return exactly this JSON schema without including any extra fields or comments.
Set psychiatric_mention_type to one of:
- current_context: connected to current admission/care; this should be positive
- background_only: history/PMH/problem-list mention only
- medication_only: psychiatric medication is mentioned without current-context reasoning
- diagnosis_list_only: diagnosis is listed without explanation or current-context reasoning
- negated_only: only negated psychiatric history/symptoms
- family_history_only: only family psychiatric history
- none: no clinically meaningful psychiatric context
Keep evidence_span short: 20 words maximum, shorter is better. Copy only the
shortest exact phrase needed from the section text.
Keep reason short: 25 words maximum, one sentence only. Do not include a long
explanation.
{
  "psychiatric_context_label": "positive|negative",
  "psychiatric_mention_type": "current_context|background_only|medication_only|diagnosis_list_only|negated_only|family_history_only|none",
  "evidence_span": "",
  "reason": ""
}
"""

SYSTEM_PROMPT_B = """You are a clinical text classification assistant.

Context:
The provided discharge-note section has already been selected because it contains
psychiatry-related language. The presence of a psychiatric term alone is not enough
for a positive label.

Task:
Classify whether psychiatric context is meaningfully integrated into the patient's
current hospital stay.

Positive:
- psychiatric context is used to explain the current symptoms or presentation
- it affects diagnostic reasoning, treatment, monitoring, disposition, or capacity
- it is used to describe behavior, reliability, cooperation, or difficulties in care
- active psychiatric symptoms meaningfully affect the hospital course
- psychiatric issues are described in unusually extensive detail despite limited
  relevance to the physical admission

Negative:
- psychiatric history is mentioned only as background information or just listed amongst the diagnoses
- psychiatric medication is only listed or continued
- the psychiatric condition is described as stable and does not affect the admission
- the mention is negated, family-history-only, or irrelevant

Examples:
- "History of schizophrenia; home olanzapine was continued." → negative
- "History was limited because of active psychosis." → positive
- "Chest pain was considered anxiety-related." → positive
- "Psychiatry assessed decision-making capacity before discharge." → positive

Return exactly one valid JSON object and no extra text:

{
  "psychiatric_context_integrated": "positive|negative",
  "integration_type": "symptom_attribution|diagnostic_reasoning|management|disposition_or_capacity|behavior_or_reliability|care_interference|active_psychiatric_course|disproportionate_detail|incidental_history|medication_only|negated_or_irrelevant",
  "evidence_span": "",
  "reason": ""
}

Choose the single best integration_type.
Keep evidence_span short: 20 words maximum, shorter is better. Copy only the
shortest exact phrase needed from the section text.
Keep reason short: 25 words maximum, one sentence only. Do not include a long
explanation.
"""

SYSTEM_PROMPTS = {
    "A": SYSTEM_PROMPT_A,
    "B": SYSTEM_PROMPT_B,
}
RESPONSE_SCHEMAS = {
    "A": RESPONSE_SCHEMA_A,
    "B": RESPONSE_SCHEMA_B,
}
SYSTEM_PROMPT = SYSTEM_PROMPTS[PROMPT_VERSION]
ACTIVE_RESPONSE_SCHEMA = RESPONSE_SCHEMAS[PROMPT_VERSION]


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


def normalize_section_identity_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize stable section identity columns for checkpoint matching."""
    df = df.copy()
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    if "section_name" in df.columns:
        df["section_name"] = df["section_name"].fillna("").astype(str).str.strip()
    return df


def load_resume_checkpoint(
    section_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, set[int]]:
    """Load matching checkpoint rows and return completed classifier row IDs."""
    section_checkpoint_path = (
        OUTPUT_DIR / "psych_history_section_classifier_results_checkpoint.csv"
    )
    chunk_checkpoint_path = (
        OUTPUT_DIR / "psych_history_section_chunk_classifier_results_checkpoint.csv"
    )
    if not RESUME_FROM_CHECKPOINT or not section_checkpoint_path.exists():
        return pd.DataFrame(), pd.DataFrame(), set()

    section_checkpoint = pd.read_csv(section_checkpoint_path)
    missing = sorted(set(SECTION_ID_COLUMNS) - set(section_checkpoint.columns))
    if missing:
        print(
            f"Ignoring checkpoint with missing identity columns: {missing}",
            flush=True,
        )
        return pd.DataFrame(), pd.DataFrame(), set()

    current_identity = normalize_section_identity_values(
        section_rows.loc[:, SECTION_ID_COLUMNS]
    )
    checkpoint_identity = normalize_section_identity_values(
        section_checkpoint.loc[:, SECTION_ID_COLUMNS]
    )
    matching_identity = checkpoint_identity.merge(
        current_identity,
        on=SECTION_ID_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    completed_ids = set(matching_identity["classifier_row_id"].astype(int))
    if not completed_ids:
        print("Checkpoint found, but no rows match the current input; ignoring it.", flush=True)
        return pd.DataFrame(), pd.DataFrame(), set()

    section_checkpoint = section_checkpoint.loc[
        section_checkpoint["classifier_row_id"].astype(int).isin(completed_ids)
    ].copy()

    if chunk_checkpoint_path.exists():
        chunk_checkpoint = pd.read_csv(chunk_checkpoint_path)
        if "classifier_row_id" in chunk_checkpoint.columns:
            chunk_checkpoint = chunk_checkpoint.loc[
                chunk_checkpoint["classifier_row_id"].astype(int).isin(completed_ids)
            ].copy()
        else:
            chunk_checkpoint = pd.DataFrame()
    else:
        chunk_checkpoint = pd.DataFrame()

    print(
        f"Resuming from checkpoint with {len(section_checkpoint)} completed sections.",
        flush=True,
    )
    return section_checkpoint, chunk_checkpoint, completed_ids


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


def check_model_available() -> None:
    """Fail early if the configured local model backend is not reachable."""
    if BACKEND == "transformers":
        load_transformers_model()
        return

    if BACKEND != "ollama":
        raise ValueError("PSYCH_HISTORY_BACKEND must be 'ollama' or 'transformers'.")

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


def print_transformers_diagnostics(stage: str, model: Any | None = None) -> None:
    """Print CUDA/model placement diagnostics for cluster speed debugging."""
    global _TRANSFORMERS_DIAGNOSTICS_PRINTED
    if stage == "after_model_load" and _TRANSFORMERS_DIAGNOSTICS_PRINTED:
        return

    try:
        import torch
    except ImportError:
        print(f"[transformers diagnostics] {stage}: torch is not installed", flush=True)
        return

    cuda_available = torch.cuda.is_available()
    print(f"[transformers diagnostics] {stage}: cuda_available={cuda_available}", flush=True)
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"[transformers diagnostics] cuda_device_count={device_count}", flush=True)
        for device_index in range(device_count):
            props = torch.cuda.get_device_properties(device_index)
            allocated_gb = torch.cuda.memory_allocated(device_index) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(device_index) / 1024**3
            total_gb = props.total_memory / 1024**3
            print(
                "[transformers diagnostics] "
                f"cuda:{device_index} name={props.name} "
                f"total_memory_gb={total_gb:.2f} "
                f"allocated_gb={allocated_gb:.2f} "
                f"reserved_gb={reserved_gb:.2f}",
                flush=True,
            )

    if model is not None:
        first_parameter = next(model.parameters(), None)
        if first_parameter is not None:
            print(
                "[transformers diagnostics] "
                f"first_parameter_device={first_parameter.device} "
                f"first_parameter_dtype={first_parameter.dtype}",
                flush=True,
            )
        device_map = getattr(model, "hf_device_map", None)
        if device_map is not None:
            devices = sorted({str(device) for device in device_map.values()})
            print(f"[transformers diagnostics] hf_device_map_devices={devices}", flush=True)

    if stage == "after_model_load":
        _TRANSFORMERS_DIAGNOSTICS_PRINTED = True


def resolve_hf_torch_dtype() -> Any:
    """Resolve the configured Hugging Face torch dtype."""
    if HF_TORCH_DTYPE == "auto":
        return "auto"

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "A non-auto PSYCH_HISTORY_HF_TORCH_DTYPE requires PyTorch."
        ) from exc

    dtype_by_name = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if HF_TORCH_DTYPE not in dtype_by_name:
        allowed = ", ".join(["auto", *sorted(dtype_by_name)])
        raise ValueError(
            f"Invalid PSYCH_HISTORY_HF_TORCH_DTYPE={HF_TORCH_DTYPE!r}. "
            f"Allowed values: {allowed}"
        )
    return dtype_by_name[HF_TORCH_DTYPE]


def load_transformers_model() -> tuple[Any, Any]:
    """Lazy-load the local Hugging Face model for GPU-cluster runs."""
    global _TRANSFORMERS_MODEL, _TRANSFORMERS_TOKENIZER
    if _TRANSFORMERS_MODEL is not None and _TRANSFORMERS_TOKENIZER is not None:
        return _TRANSFORMERS_MODEL, _TRANSFORMERS_TOKENIZER

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The transformers backend requires `transformers` and a working "
            "PyTorch install in the cluster environment."
        ) from exc

    print(f"Loading Hugging Face model: {HF_MODEL_ID}", flush=True)
    print(f"Using Hugging Face torch dtype: {HF_TORCH_DTYPE}", flush=True)
    print_transformers_diagnostics("before_model_load")
    _TRANSFORMERS_TOKENIZER = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    _TRANSFORMERS_MODEL = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        device_map=HF_DEVICE_MAP,
        torch_dtype=resolve_hf_torch_dtype(),
    )
    _TRANSFORMERS_MODEL.eval()
    print_transformers_diagnostics("after_model_load", _TRANSFORMERS_MODEL)
    return _TRANSFORMERS_MODEL, _TRANSFORMERS_TOKENIZER


def generate_transformers_response(messages: list[dict[str, str]]) -> str:
    """Generate one JSON response with a local Hugging Face model."""
    return generate_transformers_responses([messages])[0]


def generate_transformers_responses(messages_batch: list[list[dict[str, str]]]) -> list[str]:
    """Generate JSON responses for a batch of prompts with a Hugging Face model."""
    model, tokenizer = load_transformers_model()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    model_device = next(model.parameters()).device
    inputs = {
        key: value.to(model_device)
        for key, value in inputs.items()
    }
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    input_width = inputs["input_ids"].shape[-1]
    return [
        tokenizer.decode(output_ids[index][input_width:], skip_special_tokens=True).strip()
        for index in range(len(messages_batch))
    ]


def generate_response(messages: list[dict[str, str]]) -> str:
    """Generate a JSON response for one section with the configured backend."""
    if BACKEND == "transformers":
        return generate_transformers_response(messages)

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "format": ACTIVE_RESPONSE_SCHEMA,
        "think": False,
        "options": {
            "num_predict": MAX_NEW_TOKENS,
            "temperature": 0,
        },
    }
    response = call_ollama(payload)
    message = response.get("message", {})
    return str(message.get("content", "")).strip()


def generate_responses(messages_batch: list[list[dict[str, str]]]) -> list[str]:
    """Generate JSON responses for one or more prompts."""
    if BACKEND == "transformers" and len(messages_batch) > 1:
        return generate_transformers_responses(messages_batch)
    return [generate_response(messages) for messages in messages_batch]


def parse_json_response(response_text: str) -> dict[str, Any]:
    """Parse a model response as JSON, with a fallback for wrapped JSON text."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", response_text)
    if not match:
        recovered = recover_partial_json_response(response_text)
        if recovered is not None:
            return recovered
        raise ValueError(f"Model returned no JSON object: {response_text}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        recovered = recover_partial_json_response(response_text)
        if recovered is not None:
            return recovered
        raise ValueError(f"Model returned invalid JSON: {response_text}") from exc


def truncate_words_and_chars(text: str, max_words: int, max_chars: int) -> str:
    """Return text capped by word and character limits."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return ""
    words = text.split()
    text = " ".join(words[:max_words])
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def extract_json_string_field(response_text: str, field_name: str) -> str:
    """Extract a simple string field from a possibly truncated JSON object."""
    pattern = rf'"{re.escape(field_name)}"\s*:\s*"([^"]*)'
    match = re.search(pattern, response_text, flags=re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def recover_partial_json_response(response_text: str) -> dict[str, Any] | None:
    """Recover known classifier fields from a truncated JSON-like response."""
    response_text = str(response_text)
    label = (
        extract_json_string_field(response_text, "psychiatric_context_label")
        or extract_json_string_field(response_text, "psychiatric_context_integrated")
    ).lower()
    mention_type = (
        extract_json_string_field(response_text, "psychiatric_mention_type")
        or extract_json_string_field(response_text, "integration_type")
    ).lower()
    evidence_span = extract_json_string_field(response_text, "evidence_span")
    reason = extract_json_string_field(response_text, "reason")

    if label not in {"positive", "negative"}:
        return None

    recovered = {
        "psychiatric_context_label": label,
        "psychiatric_mention_type": mention_type,
        "evidence_span": evidence_span,
        "reason": reason,
        "json_recovered_from_partial_response": True,
    }
    return recovered


def normalize_model_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize optional or malformed model fields into stable output columns."""
    psychiatric_label = str(
        result.get(
            "psychiatric_context_label",
            result.get("psychiatric_context_integrated", ""),
        )
    ).strip().lower()
    if psychiatric_label not in {"positive", "negative"}:
        psychiatric_label = "negative"
    mention_type = str(
        result.get("psychiatric_mention_type", result.get("integration_type", ""))
    ).strip().lower()
    allowed_mention_types = {
        "current_context",
        "background_only",
        "medication_only",
        "diagnosis_list_only",
        "negated_only",
        "family_history_only",
        "none",
        "symptom_attribution",
        "diagnostic_reasoning",
        "management",
        "disposition_or_capacity",
        "behavior_or_reliability",
        "care_interference",
        "active_psychiatric_course",
        "disproportionate_detail",
        "incidental_history",
        "negated_or_irrelevant",
    }
    if mention_type not in allowed_mention_types:
        mention_type = "current_context" if psychiatric_label == "positive" else "none"

    return {
        "psychiatric_context_label": psychiatric_label,
        "psychiatric_mention_type": mention_type,
        "evidence_span": truncate_words_and_chars(
            str(result.get("evidence_span", "")),
            MAX_EVIDENCE_WORDS,
            MAX_EVIDENCE_CHARS,
        ),
        "reason": truncate_words_and_chars(
            str(result.get("reason", "")),
            MAX_REASON_WORDS,
            MAX_REASON_CHARS,
        ),
        "json_recovered_from_partial_response": bool(
            result.get("json_recovered_from_partial_response", False)
        ),
    }


def combine_chunk_results(chunk_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse chunk-level labels into one section-level label."""
    labels = [result["psychiatric_context_label"] for result in chunk_results]
    if "positive" in labels:
        chosen_label = "positive"
    else:
        chosen_label = "negative"

    chosen_result = next(
        result
        for result in chunk_results
        if result["psychiatric_context_label"] == chosen_label
    )
    return {
        **chosen_result,
        "psychiatric_context_label": chosen_label,
        "n_chunks": len(chunk_results),
        "n_positive_chunks": labels.count("positive"),
        "n_negative_chunks": labels.count("negative"),
    }


def build_section_output_rows(
    section_row_ids: list[int],
    section_records: dict[int, dict[str, Any]],
    chunk_results_by_row_id: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build section-level output rows for completed section row IDs."""
    output_rows = []
    for row_id in section_row_ids:
        row = section_records[int(row_id)]["row"]
        chunk_results = chunk_results_by_row_id[int(row_id)]
        section_result = combine_chunk_results(chunk_results)

        output_rows.append(
            {
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
                "model_name": effective_model_name(),
            }
        )
    return output_rows


def completed_section_row_ids(
    section_records: dict[int, dict[str, Any]],
    chunk_results_by_row_id: dict[int, list[dict[str, Any]]],
) -> list[int]:
    """Return section row IDs whose chunks are fully classified."""
    return [
        row_id
        for row_id, record in section_records.items()
        if len(chunk_results_by_row_id[row_id]) == record["n_chunks"]
    ]


def write_checkpoint_outputs(
    completed_row_ids: list[int],
    section_records: dict[int, dict[str, Any]],
    chunk_results_by_row_id: dict[int, list[dict[str, Any]]],
    chunk_output_rows: list[dict[str, Any]],
    prior_section_checkpoint: pd.DataFrame | None = None,
    prior_chunk_checkpoint: pd.DataFrame | None = None,
) -> None:
    """Write cumulative checkpoint outputs for completed sections and chunks."""
    if not completed_row_ids:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    new_section_checkpoint = pd.DataFrame(
        build_section_output_rows(
            completed_row_ids,
            section_records,
            chunk_results_by_row_id,
        )
    )
    new_chunk_checkpoint = pd.DataFrame(chunk_output_rows)

    section_frames = [
        frame
        for frame in [prior_section_checkpoint, new_section_checkpoint]
        if frame is not None and not frame.empty
    ]
    chunk_frames = [
        frame
        for frame in [prior_chunk_checkpoint, new_chunk_checkpoint]
        if frame is not None and not frame.empty
    ]
    section_checkpoint = (
        pd.concat(section_frames, ignore_index=True)
        if section_frames
        else pd.DataFrame()
    )
    chunk_checkpoint = (
        pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame()
    )

    section_checkpoint = normalize_output_table(section_checkpoint)
    chunk_checkpoint = normalize_output_table(chunk_checkpoint)
    if not section_checkpoint.empty and "classifier_row_id" in section_checkpoint.columns:
        section_checkpoint["classifier_row_id"] = section_checkpoint[
            "classifier_row_id"
        ].astype(int)
        section_checkpoint = (
            section_checkpoint.drop_duplicates(
                subset=["classifier_row_id"],
                keep="last",
            )
            .sort_values("classifier_row_id")
            .reset_index(drop=True)
        )
    if (
        not chunk_checkpoint.empty
        and {"classifier_row_id", "chunk_index"}.issubset(chunk_checkpoint.columns)
    ):
        chunk_checkpoint["classifier_row_id"] = chunk_checkpoint[
            "classifier_row_id"
        ].astype(int)
        chunk_checkpoint["chunk_index"] = chunk_checkpoint["chunk_index"].astype(int)
        chunk_checkpoint = (
            chunk_checkpoint.drop_duplicates(
                subset=["classifier_row_id", "chunk_index"],
                keep="last",
            )
            .sort_values(["classifier_row_id", "chunk_index"])
            .reset_index(drop=True)
        )

    section_checkpoint.to_csv(
        OUTPUT_DIR / "psych_history_section_classifier_results_checkpoint.csv",
        index=False,
    )
    section_checkpoint.to_parquet(
        OUTPUT_DIR / "psych_history_section_classifier_results_checkpoint.parquet",
        index=False,
    )
    chunk_checkpoint.to_csv(
        OUTPUT_DIR / "psych_history_section_chunk_classifier_results_checkpoint.csv",
        index=False,
    )
    chunk_checkpoint.to_parquet(
        OUTPUT_DIR / "psych_history_section_chunk_classifier_results_checkpoint.parquet",
        index=False,
    )
    print(
        "Checkpoint saved after "
        f"{len(completed_row_ids)} newly completed sections "
        f"({len(section_checkpoint)} cumulative sections) to: {OUTPUT_DIR}",
        flush=True,
    )


def normalize_output_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize output dtypes before CSV/parquet writes."""
    df = df.copy()
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    for column in [
        "cohort",
        "note_id",
        "section_name",
        "psych_keyword_groups",
        "matched_terms",
        "psychiatric_context_label",
        "psychiatric_mention_type",
        "evidence_span",
        "reason",
        "model_name",
    ]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)
    if "json_recovered_from_partial_response" in df.columns:
        df["json_recovered_from_partial_response"] = df[
            "json_recovered_from_partial_response"
        ].fillna(False).astype(bool)
    return df


def effective_model_name() -> str:
    """Return the model identifier relevant to the active backend."""
    return HF_MODEL_ID if BACKEND == "transformers" else MODEL_NAME


def classify_sections(
    section_rows: pd.DataFrame,
    prior_section_checkpoint: pd.DataFrame | None = None,
    prior_chunk_checkpoint: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the local model over section chunks and return chunk/section labels."""
    chunk_output_rows = []
    total = len(section_rows)
    section_start_time = time.monotonic()
    chunk_tasks = []
    section_records: dict[int, dict[str, Any]] = {}
    chunk_results_by_row_id: dict[int, list[dict[str, Any]]] = {}

    for _, row in section_rows.iterrows():
        row_id = int(row["classifier_row_id"])
        row_dict = row.to_dict()
        chunks = split_text_into_chunks(row["section_text"])
        section_records[row_id] = {
            "row": row_dict,
            "n_chunks": len(chunks),
        }
        chunk_results_by_row_id[row_id] = []
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_tasks.append(
                {
                    "row_id": row_id,
                    "row": row_dict,
                    "chunk_index": chunk_index,
                    "n_chunks": len(chunks),
                    "chunk_text": chunk_text,
                    "messages": build_messages(row, chunk_text, chunk_index, len(chunks)),
                }
            )

    total_chunks = len(chunk_tasks)
    completed_chunks = 0
    last_checkpoint_completed_sections = 0
    batch_size = max(1, BATCH_SIZE if BACKEND == "transformers" else 1)

    for batch_start in range(0, total_chunks, batch_size):
        batch_tasks = chunk_tasks[batch_start : batch_start + batch_size]
        completed_sections = sum(
            1
            for row_id, results in chunk_results_by_row_id.items()
            if len(results) == section_records[row_id]["n_chunks"]
        )
        elapsed = time.monotonic() - section_start_time
        section_rate = 60.0 * completed_sections / elapsed if elapsed > 0 else 0.0
        if batch_start == 0 or completed_chunks % 10 == 0 or completed_chunks + len(batch_tasks) >= total_chunks:
            print(
                f"Classifying chunks {completed_chunks + 1}-"
                f"{completed_chunks + len(batch_tasks)}/{total_chunks} | "
                f"sections {completed_sections}/{total} | "
                f"elapsed {format_duration(elapsed)} | "
                f"rate {section_rate:.2f} sections/min | "
                f"ETA {estimate_remaining(elapsed, completed_sections, total)} | "
                f"batch_size {batch_size}",
                flush=True,
            )

        for task in batch_tasks:
            print(
                f"  chunk {task['chunk_index'] + 1}/{task['n_chunks']} "
                f"for {task['row']['section_name']}",
                flush=True,
            )

        response_texts = generate_responses([task["messages"] for task in batch_tasks])
        for task, response_text in zip(batch_tasks, response_texts):
            raw_result = parse_json_response(response_text)
            normalized = normalize_model_result(raw_result)
            chunk_results_by_row_id[task["row_id"]].append(normalized)
            row = task["row"]
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
                    "chunk_index": task["chunk_index"],
                    "n_chunks": task["n_chunks"],
                    "chunk_word_count": len(task["chunk_text"].split()),
                    **normalized,
                    "model_name": effective_model_name(),
                }
            )
        if BACKEND != "transformers":
            time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)
        completed_chunks += len(batch_tasks)
        completed_row_ids = completed_section_row_ids(
            section_records,
            chunk_results_by_row_id,
        )
        if (
            CHECKPOINT_EVERY_SECTIONS > 0
            and len(completed_row_ids) - last_checkpoint_completed_sections
            >= CHECKPOINT_EVERY_SECTIONS
        ):
            write_checkpoint_outputs(
                completed_row_ids,
                section_records,
                chunk_results_by_row_id,
                chunk_output_rows,
                prior_section_checkpoint=prior_section_checkpoint,
                prior_chunk_checkpoint=prior_chunk_checkpoint,
            )
            last_checkpoint_completed_sections = len(completed_row_ids)

    output_rows = build_section_output_rows(
        section_rows["classifier_row_id"].astype(int).tolist(),
        section_records,
        chunk_results_by_row_id,
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
    results = normalize_output_table(results)
    chunk_results = normalize_output_table(chunk_results)
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

    with (OUTPUT_DIR / "psych_history_run_metadata.json").open("w") as handle:
        json.dump(run_metadata, handle, indent=2, default=str)

    label_summary = (
        results.groupby(
            [
                "cohort",
                "section_name",
                "psychiatric_context_label",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "n_sections"})
        .sort_values(
            [
                "cohort",
                "section_name",
                "psychiatric_context_label",
            ]
        )
    )
    label_summary.to_csv(OUTPUT_DIR / "psych_history_section_label_summary.csv", index=False)

    admission_summary = (
        results.assign(
            is_positive=results["psychiatric_context_label"].eq("positive"),
        )
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
    """Run the local classifier over parsed note sections."""
    run_started_at = datetime.now().isoformat(timespec="seconds")
    run_start_time = time.monotonic()
    validate_input_path()
    section_rows = load_section_rows()
    resume_results, resume_chunk_results, completed_resume_ids = load_resume_checkpoint(
        section_rows
    )
    rows_to_classify = section_rows.loc[
        ~section_rows["classifier_row_id"].astype(int).isin(completed_resume_ids)
    ].copy()
    print(f"Loaded {len(section_rows)} non-empty section rows from: {INPUT_PATH}")
    if completed_resume_ids:
        print(f"Rows remaining after checkpoint resume: {len(rows_to_classify)}")
    print(f"Using psych-history prompt version: {PROMPT_VERSION}")
    print(f"Using psych-history backend: {BACKEND}")
    if BACKEND == "transformers":
        print(f"Using Hugging Face model: {HF_MODEL_ID}")
        print(f"Using Hugging Face device_map: {HF_DEVICE_MAP}")
    else:
        print(f"Using local Ollama model: {MODEL_NAME}")
    print(f"Chunking sections at {CHUNK_WORDS} words with {CHUNK_OVERLAP_WORDS} word overlap")
    print(f"Transformers batch size: {BATCH_SIZE}")
    print(f"Checkpoint every completed sections: {CHECKPOINT_EVERY_SECTIONS}")
    print(
        "Output text caps: "
        f"evidence_span <= {MAX_EVIDENCE_WORDS} words/{MAX_EVIDENCE_CHARS} chars, "
        f"reason <= {MAX_REASON_WORDS} words/{MAX_REASON_CHARS} chars"
    )
    print(f"Run started at: {run_started_at}")
    check_model_available()
    if rows_to_classify.empty:
        results = resume_results.copy()
        chunk_results = resume_chunk_results.copy()
        print("All current input rows were already present in the checkpoint.")
    else:
        new_results, new_chunk_results = classify_sections(
            rows_to_classify,
            prior_section_checkpoint=resume_results,
            prior_chunk_checkpoint=resume_chunk_results,
        )
        results = pd.concat([resume_results, new_results], ignore_index=True)
        chunk_results = pd.concat(
            [resume_chunk_results, new_chunk_results],
            ignore_index=True,
        )
        if not results.empty and "classifier_row_id" in results.columns:
            results = results.sort_values("classifier_row_id").reset_index(drop=True)
        if not chunk_results.empty and "classifier_row_id" in chunk_results.columns:
            chunk_results = chunk_results.sort_values(
                ["classifier_row_id", "chunk_index"]
            ).reset_index(drop=True)
    elapsed_seconds = time.monotonic() - run_start_time
    run_metadata = {
        "run_started_at": run_started_at,
        "run_finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed": format_duration(elapsed_seconds),
        "backend": BACKEND,
        "prompt_version": PROMPT_VERSION,
        "model_name": MODEL_NAME,
        "hf_model_id": HF_MODEL_ID,
        "hf_device_map": HF_DEVICE_MAP,
        "hf_torch_dtype": HF_TORCH_DTYPE,
        "input_path": str(INPUT_PATH),
        "output_dir": str(OUTPUT_DIR),
        "max_notes": MAX_NOTES,
        "chunk_words": CHUNK_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "batch_size": BATCH_SIZE,
        "checkpoint_every_sections": CHECKPOINT_EVERY_SECTIONS,
        "resume_from_checkpoint": RESUME_FROM_CHECKPOINT,
        "n_resume_section_rows": len(resume_results),
        "n_rows_classified_this_run": len(rows_to_classify),
        "max_evidence_words": MAX_EVIDENCE_WORDS,
        "max_reason_words": MAX_REASON_WORDS,
        "max_evidence_chars": MAX_EVIDENCE_CHARS,
        "max_reason_chars": MAX_REASON_CHARS,
        "n_section_rows": len(results),
        "n_chunk_rows": len(chunk_results),
        "n_admissions": int(results[["cohort", "subject_id", "hadm_id"]].drop_duplicates().shape[0]),
    }
    write_outputs(results, chunk_results, run_metadata)
    print(f"Total runtime: {run_metadata['elapsed']}")


if __name__ == "__main__":
    main()
