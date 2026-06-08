from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parent / "preprocessing" / "chief_complaint_preprocessed"
OUTPUT_DIR = SCRIPT_DIR / "chief_complaint_embeddings"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
TEXT_COLUMN = "chief_complaint_normalized"
ROW_LIMIT = None

BATCH_SIZE = 8
MAX_LENGTH = 64

INPUTS = {
    #"MHH1_psychotic": INPUT_DIR / "MHH1_psychotic_chief_complaints_preprocessed.parquet",
    "MHC0": INPUT_DIR / "MHC0_chief_complaints_preprocessed.parquet",
}

BASE_METADATA_COLUMNS = [
    "source_table",
    "subject_id",
    "hadm_id",
    "chief_complaint_raw",
    "chief_complaint_normalized",
]

OPTIONAL_METADATA_COLUMNS = [
    "physical_entities_affirmed",
    "psych_substance_self_harm_entities_affirmed",
    "has_affirmed_physical_entity",
    "has_affirmed_psych_substance_self_harm_entity",
]


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_group(group_name: str, parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing input parquet for {group_name}: {parquet_path}")

    text_column = sql_identifier(TEXT_COLUMN)
    limit_clause = f"\nLIMIT {int(ROW_LIMIT)}" if ROW_LIMIT is not None else ""

    con = duckdb.connect()
    try:
        schema = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({sql_string(str(parquet_path))})"
        ).fetchdf()
        available_columns = set(schema["column_name"])

        required_columns = {"has_chief_complaint", *BASE_METADATA_COLUMNS}
        missing_columns = sorted(required_columns - available_columns)
        if missing_columns:
            raise ValueError(
                f"{parquet_path} is missing required columns: {missing_columns}"
            )

        if TEXT_COLUMN not in available_columns:
            raise ValueError(
                f"{TEXT_COLUMN!r} is not present in {parquet_path}. "
                f"Available columns: {sorted(available_columns)}"
            )

        query = f"""
            SELECT *
            FROM read_parquet({sql_string(str(parquet_path))})
            WHERE
                coalesce(has_chief_complaint, false)
                AND {text_column} IS NOT NULL
                AND trim(cast({text_column} AS VARCHAR)) <> ''
            ORDER BY subject_id, hadm_id
            {limit_clause}
        """
        df = con.execute(query).fetchdf()
    finally:
        con.close()

    metadata_columns = []
    for column in BASE_METADATA_COLUMNS + [TEXT_COLUMN] + OPTIONAL_METADATA_COLUMNS:
        if column in df.columns and column not in metadata_columns:
            metadata_columns.append(column)

    metadata = df.loc[:, metadata_columns].copy()
    metadata.insert(0, "embedding_row_id", np.arange(len(metadata), dtype=np.int64))
    return metadata


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    group_name: str,
) -> np.ndarray:
    embeddings = []
    n_texts = len(texts)

    for start in range(0, n_texts, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_texts)
        batch_texts = texts[start:end]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        embeddings.append(pooled.cpu().numpy().astype(np.float32))
        print(f"{group_name}: embedded rows {start + 1}-{end} of {n_texts}", flush=True)

    if not embeddings:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack(embeddings)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    device = get_device()
    print(f"Using device: {device}", flush=True)
    print(f"Loading model locally: {MODEL_NAME}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    for group_name, parquet_path in INPUTS.items():
        print(f"Loading {group_name}: {parquet_path}", flush=True)
        metadata = load_group(group_name, parquet_path)
        texts = metadata[TEXT_COLUMN].astype(str).tolist()

        embeddings = embed_texts(texts, tokenizer, model, device, group_name)
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"Metadata rows and embedding rows do not match for {group_name}: "
                f"{len(metadata)} metadata rows vs {embeddings.shape[0]} embeddings"
            )

        metadata_path = OUTPUT_DIR / f"{group_name}_chief_complaint_embedding_metadata.parquet"
        embeddings_path = OUTPUT_DIR / f"{group_name}_chief_complaint_embeddings.npy"

        metadata.to_parquet(metadata_path, index=False)
        np.save(embeddings_path, embeddings)

        print(f"Saved metadata: {metadata_path}", flush=True)
        print(f"Saved embeddings: {embeddings_path}", flush=True)


if __name__ == "__main__":
    main()
