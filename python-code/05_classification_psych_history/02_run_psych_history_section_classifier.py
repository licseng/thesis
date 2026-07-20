"""Classify psychotic-disorder history mentions in parsed note sections.

This is WP2 of the thesis. The ICD cohort definition already establishes prior
psychotic disorder history for the exposed group. This script asks a different
question: whether that history is surfaced/documented in the current index
admission discharge note.

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
OUTPUT_DIR = Path(
    os.environ.get(
        "PSYCH_HISTORY_OUTPUT_DIR",
        str(SCRIPT_DIR / "psych_history_classifier_output"),
    )
)

# Model/backend settings
BACKEND = os.environ.get("PSYCH_HISTORY_BACKEND", "ollama").lower()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.environ.get("PSYCH_HISTORY_MODEL_NAME", "qwen3:4b")
HF_MODEL_ID = os.environ.get("PSYCH_HISTORY_HF_MODEL_ID", "Qwen/Qwen3-4B")
HF_DEVICE_MAP = os.environ.get("PSYCH_HISTORY_HF_DEVICE_MAP", "auto")
HF_TORCH_DTYPE = os.environ.get("PSYCH_HISTORY_HF_TORCH_DTYPE", "auto").lower()
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("PSYCH_HISTORY_REQUEST_TIMEOUT_SECONDS", "180"))
MAX_NEW_TOKENS = int(os.environ.get("PSYCH_HISTORY_MAX_NEW_TOKENS", "384"))
CHUNK_WORDS = int(os.environ.get("PSYCH_HISTORY_CHUNK_WORDS", "600"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("PSYCH_HISTORY_CHUNK_OVERLAP_WORDS", "75"))
BATCH_SIZE = int(os.environ.get("PSYCH_HISTORY_BATCH_SIZE", "1"))
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
                "none",
                "unclear",
            ],
        },
        "medication_only": {"type": "boolean"},
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
        "medication_only",
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
false positives. 

Task:
Your role is to decide whether the section truly contains patient-specific psychiatric context. 
Context includes either one of the following:
- diagnosis
- symptoms
- medication


Use two separate flags:
1. psychosis_related_context:
   Use "positive" if the section mentions psychosis-related context.
    Schizophrenia sometimes abbreviated as 'sz'.
2. other_psychiatric_context:
   Use "positive" if the section mentions other psychiatric context that is not psychosis-related. 
   Medications for other psychiatric disorders also count.

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
  "disorder_type": "schizophrenia|schizoaffective disorder|psychotic disorder|bipolar disorder with psychotic features|other psychiatric|none|unclear",
  "medication_only": true|false,
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

    disorder_type = str(result.get("disorder_type", "unclear")).strip().lower()
    if disorder_type == "substance use disorder":
        disorder_type = "other psychiatric"
    if disorder_type not in {
        "schizophrenia",
        "schizoaffective disorder",
        "psychotic disorder",
        "bipolar disorder with psychotic features",
        "other psychiatric",
        "none",
        "unclear",
    }:
        disorder_type = "unclear"

    return {
        "psychosis_related_context_label": psychosis_label,
        "other_psychiatric_context_label": other_label,
        "label": label,
        "disorder_type": disorder_type,
        "medication_only": bool(result.get("medication_only", False)),
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
    positive_chunk_results = [
        result for result in chunk_results if result["label"] == "positive"
    ]
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
        "medication_only": (
            all(result["medication_only"] for result in positive_chunk_results)
            if positive_chunk_results
            else False
        ),
        "n_chunks": len(chunk_results),
        "n_positive_chunks": labels.count("positive"),
        "n_negative_chunks": labels.count("negative"),
        "n_psychosis_positive_chunks": psychosis_labels.count("positive"),
        "n_other_psychiatric_positive_chunks": other_labels.count("positive"),
    }


def effective_model_name() -> str:
    """Return the model identifier relevant to the active backend."""
    return HF_MODEL_ID if BACKEND == "transformers" else MODEL_NAME


def classify_sections(
    section_rows: pd.DataFrame,
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

    output_rows = []
    for row_id in section_rows["classifier_row_id"].astype(int):
        row = section_records[int(row_id)]["row"]
        chunk_results = chunk_results_by_row_id[int(row_id)]
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
            "model_name": effective_model_name(),
        }
        output_rows.append(output_row)

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
    """Run the local classifier over parsed note sections."""
    run_started_at = datetime.now().isoformat(timespec="seconds")
    run_start_time = time.monotonic()
    validate_input_path()
    section_rows = load_section_rows()
    print(f"Loaded {len(section_rows)} non-empty section rows from: {INPUT_PATH}")
    print(f"Using psych-history backend: {BACKEND}")
    if BACKEND == "transformers":
        print(f"Using Hugging Face model: {HF_MODEL_ID}")
        print(f"Using Hugging Face device_map: {HF_DEVICE_MAP}")
    else:
        print(f"Using local Ollama model: {MODEL_NAME}")
    print(f"Chunking sections at {CHUNK_WORDS} words with {CHUNK_OVERLAP_WORDS} word overlap")
    print(f"Transformers batch size: {BATCH_SIZE}")
    print(f"Run started at: {run_started_at}")
    check_model_available()
    results, chunk_results = classify_sections(section_rows)
    elapsed_seconds = time.monotonic() - run_start_time
    run_metadata = {
        "run_started_at": run_started_at,
        "run_finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed": format_duration(elapsed_seconds),
        "backend": BACKEND,
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
        "n_section_rows": len(results),
        "n_chunk_rows": len(chunk_results),
        "n_admissions": int(results[["cohort", "subject_id", "hadm_id"]].drop_duplicates().shape[0]),
    }
    write_outputs(results, chunk_results, run_metadata)
    print(f"Total runtime: {run_metadata['elapsed']}")


if __name__ == "__main__":
    main()
