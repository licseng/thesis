"""Run inference with the chunk-pooled Bio_ClinicalBERT LOS classifier.

This script loads the trained model produced by:

    02_train_bioclinicalbert_los_classifier.py

It runs prediction on the held-out parquet files created by:

    01_training_data_creation/03_create_prediction_model_dataset.py

Default evaluation files:
    - test_general.parquet
    - test_fairness_mhh1_mhc0.parquet

Outputs:
    - one prediction CSV per evaluation split
    - one metrics JSON per evaluation split
    - a combined metrics CSV
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "01_training_data_creation" / "prediction_model_dataset"
DEFAULT_TRAINING_OUTPUT_DIR = SCRIPT_DIR / "bioclinicalbert_los_classifier_output"
DEFAULT_MODEL_DIR = DEFAULT_TRAINING_OUTPUT_DIR / "best_model"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "bioclinicalbert_los_classifier_inference_output"
TRAINING_SCRIPT_PATH = SCRIPT_DIR / "02_train_bioclinicalbert_los_classifier.py"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR.parent / ".matplotlib"))


def env_int(name: str, default: int | None = None) -> int | None:
    """Read an optional integer environment variable."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    """Parse inference arguments, using environment variables as defaults."""
    parser = argparse.ArgumentParser(
        description="Run LOS classifier inference on held-out test parquet files."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("LOS_INFERENCE_MODEL_DIR", DEFAULT_MODEL_DIR)),
    )
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=Path(
            os.environ.get("LOS_TRAINING_OUTPUT_DIR", DEFAULT_TRAINING_OUTPUT_DIR)
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("LOS_INFERENCE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
    )
    parser.add_argument(
        "--test-general-path",
        type=Path,
        default=Path(
            os.environ.get(
                "LOS_TEST_GENERAL_PATH",
                DATASET_DIR / "test_general.parquet",
            )
        ),
    )
    parser.add_argument(
        "--test-fairness-path",
        type=Path,
        default=Path(
            os.environ.get(
                "LOS_TEST_FAIRNESS_PATH",
                DATASET_DIR / "test_fairness_mhh1_mhc0.parquet",
            )
        ),
    )
    parser.add_argument("--model-name", default=os.environ.get("LOS_MODEL_NAME"))
    parser.add_argument("--text-column", default=os.environ.get("LOS_TEXT_COLUMN"))
    parser.add_argument("--label-column", default=os.environ.get("LOS_LABEL_COLUMN"))
    parser.add_argument("--filter-column", default=os.environ.get("LOS_FILTER_COLUMN") or None)
    parser.add_argument("--max-length", type=int, default=env_int("LOS_MAX_LENGTH"))
    parser.add_argument("--max-chunks", type=int, default=env_int("LOS_MAX_CHUNKS"))
    parser.add_argument("--pooling-strategy", default=os.environ.get("LOS_POOLING_STRATEGY"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=env_int("LOS_INFERENCE_BATCH_SIZE", 16),
    )
    parser.add_argument("--max-rows", type=int, default=env_int("LOS_INFERENCE_MAX_ROWS"))
    parser.add_argument("--fp16", action="store_true", default=env_bool("LOS_INFERENCE_FP16", False))
    parser.add_argument("--bf16", action="store_true", default=env_bool("LOS_INFERENCE_BF16", False))
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("LOS_INFERENCE_THRESHOLD", "0.5")))
    return parser.parse_args()


def load_training_module() -> Any:
    """Load the training script so inference reuses the exact model classes."""
    if not TRAINING_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing training script: {TRAINING_SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("los_training_module", TRAINING_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import training script: {TRAINING_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["los_training_module"] = module
    spec.loader.exec_module(module)
    return module


def load_run_config(training_output_dir: Path) -> dict[str, Any]:
    """Load the training run configuration if available."""
    config_path = training_output_dir / "run_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_setting(
    args: argparse.Namespace,
    run_config: dict[str, Any],
    name: str,
    default: Any,
) -> Any:
    """Use CLI/env setting first, then run_config, then a hard-coded default."""
    value = getattr(args, name)
    if value is not None:
        return value
    return run_config.get(name, default)


def load_eval_split(
    path: Path,
    text_column: str,
    label_column: str,
    filter_column: str | None,
    max_rows: int | None,
) -> pd.DataFrame:
    """Load one evaluation parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation parquet: {path}")
    required_columns = ["subject_id", "hadm_id", text_column, label_column]
    if filter_column:
        required_columns.append(filter_column)
    table = pd.read_parquet(path)
    missing = sorted(set(required_columns) - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if filter_column:
        original_n = len(table)
        table = table.loc[coerce_bool_series(table[filter_column])].copy()
        print(
            f"Applied filter {filter_column} to {path.name}: "
            f"{len(table):,}/{original_n:,} rows kept",
            flush=True,
        )
    if max_rows is not None:
        table = table.head(max_rows).copy()
    table = table.loc[
        table[text_column].fillna("").astype(str).str.strip().ne("")
        & table[label_column].notna()
    ].copy()
    table[text_column] = table[text_column].fillna("").astype(str)
    table["labels"] = table[label_column].astype(bool).astype(int)
    return table


def coerce_bool_series(series: pd.Series) -> pd.Series:
    """Coerce a common boolean-like column to bool without treating missing as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def load_state_dict(model_dir: Path) -> dict[str, Any]:
    """Load model weights saved by Hugging Face Trainer."""
    import torch

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing trained model directory: {model_dir}")

    pytorch_path = model_dir / "pytorch_model.bin"
    safetensors_path = model_dir / "model.safetensors"
    if pytorch_path.exists():
        return torch.load(pytorch_path, map_location="cpu")
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu")

    candidates = sorted(model_dir.glob("*.bin")) + sorted(model_dir.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"No pytorch_model.bin or model.safetensors found in {model_dir}"
        )
    if candidates[0].suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(candidates[0]), device="cpu")
    return torch.load(candidates[0], map_location="cpu")


def strip_known_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove wrapper prefixes sometimes introduced by training utilities."""
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ["module.", "_orig_mod."]:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def build_model_and_tokenizer(
    training_module: Any,
    model_dir: Path,
    model_name: str,
    pooling_strategy: str,
    device: Any,
) -> tuple[Any, Any]:
    """Reconstruct the chunk classifier and load trained weights."""
    import torch
    from transformers import AutoTokenizer

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing trained model directory: {model_dir}")
    tokenizer_source = model_dir if (model_dir / "tokenizer_config.json").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = training_module.ChunkPooledBertClassifier(
        model_name,
        num_labels=2,
        pooling_strategy=pooling_strategy,
    )
    state_dict = strip_known_prefixes(load_state_dict(model_dir))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing model keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected model keys: {unexpected}")
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Compute binary classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics = {
        "n_rows": int(len(labels)),
        "n_positive": int(labels.sum()),
        "pct_positive": float(100 * labels.mean()) if len(labels) else float("nan"),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


def run_inference_for_split(
    split_name: str,
    table: pd.DataFrame,
    training_module: Any,
    model: Any,
    tokenizer: Any,
    text_column: str,
    max_length: int,
    max_chunks: int,
    batch_size: int,
    device: Any,
    threshold: float,
    fp16: bool,
    bf16: bool,
    output_dir: Path,
) -> dict[str, float]:
    """Run model inference for one split and write predictions/metrics."""
    import torch
    from scipy.special import softmax
    from torch.utils.data import DataLoader
    from transformers import default_data_collator

    dataset = training_module.ChunkedTextClassificationDataset(
        table[text_column].tolist(),
        table["labels"].astype(int).tolist(),
        tokenizer,
        max_length,
        max_chunks,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    logits = []
    labels = []
    autocast_dtype = torch.float16 if fp16 else torch.bfloat16
    use_autocast = device.type == "cuda" and (fp16 or bf16)
    for batch_index, batch in enumerate(loader, start=1):
        labels.append(batch.pop("labels").cpu().numpy())
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                outputs = model(**batch)
        logits.append(outputs.logits.detach().cpu().numpy())
        if batch_index % 50 == 0:
            print(f"{split_name}: processed {batch_index * batch_size:,}/{len(dataset):,}")

    logits_array = np.concatenate(logits, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    probabilities = softmax(logits_array, axis=1)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metadata_columns = [
        column
        for column in [
            "dataset_split",
            "partition",
            "selected_for_general_test",
            "subject_id",
            "hadm_id",
            "note_id",
            "admittime",
            "dischtime",
            "hospital_los_days",
            "prolonged_los_gt_7d",
            "eligible_for_30d_readmission",
            "readmission_within_30d",
            "days_to_next_admission_after_discharge",
            "is_mhh1_psychotic_admission",
            "is_mhc0_admission",
            "is_matched_mhh1_psychotic_admission",
            "is_matched_mhc0_admission",
            "is_mhh1_psychotic_subject",
            "is_mhc0_subject",
            "is_matched_mhh1_psychotic_subject",
            "is_matched_mhc0_subject",
            "model_text_n_words",
            "n_model_sections_present",
        ]
        if column in table.columns
    ]
    predictions_df = table.loc[:, metadata_columns].copy()
    predictions_df["true_label"] = labels_array
    predictions_df["predicted_probability"] = probabilities
    predictions_df["predicted_label"] = predictions
    predictions_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    metrics = compute_metrics(labels_array, probabilities, predictions)
    with (output_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main() -> None:
    """Run LOS model inference on held-out test splits."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_config = load_run_config(args.training_output_dir)
    model_name = resolve_setting(
        args,
        run_config,
        "model_name",
        "emilyalsentzer/Bio_ClinicalBERT",
    )
    text_column = resolve_setting(args, run_config, "text_column", "model_text")
    label_column = resolve_setting(args, run_config, "label_column", "prolonged_los_gt_7d")
    filter_column = resolve_setting(args, run_config, "filter_column", None)
    max_length = int(resolve_setting(args, run_config, "max_length", 512))
    max_chunks = int(resolve_setting(args, run_config, "max_chunks", 4))
    pooling_strategy = resolve_setting(args, run_config, "pooling_strategy", "mean_max")

    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Model dir: {args.model_dir}")
    print(f"Training config dir: {args.training_output_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Label column: {label_column}")
    print(f"Filter column: {filter_column or '(none)'}")
    print(f"Pooling: {pooling_strategy}; max_length={max_length}; max_chunks={max_chunks}")

    training_module = load_training_module()
    model, tokenizer = build_model_and_tokenizer(
        training_module,
        args.model_dir,
        model_name,
        pooling_strategy,
        device,
    )

    split_paths = {
        "test_general": args.test_general_path,
        "test_fairness_mhh1_mhc0": args.test_fairness_path,
    }
    metrics_rows = []
    for split_name, path in split_paths.items():
        table = load_eval_split(path, text_column, label_column, filter_column, args.max_rows)
        print(f"\nRunning inference for {split_name}: {len(table):,} rows")
        metrics = run_inference_for_split(
            split_name=split_name,
            table=table,
            training_module=training_module,
            model=model,
            tokenizer=tokenizer,
            text_column=text_column,
            max_length=max_length,
            max_chunks=max_chunks,
            batch_size=args.batch_size,
            device=device,
            threshold=args.threshold,
            fp16=args.fp16,
            bf16=args.bf16,
            output_dir=args.output_dir,
        )
        metrics_rows.append({"split": split_name, **metrics})
        print(json.dumps(metrics, indent=2))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(args.output_dir / "los_inference_metrics_summary.csv", index=False)
    with (args.output_dir / "inference_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_dir": str(args.model_dir),
                "training_output_dir": str(args.training_output_dir),
                "model_name": model_name,
                "text_column": text_column,
                "label_column": label_column,
                "filter_column": filter_column,
                "max_length": max_length,
                "max_chunks": max_chunks,
                "pooling_strategy": pooling_strategy,
                "batch_size": args.batch_size,
                "threshold": args.threshold,
                "fp16": args.fp16,
                "bf16": args.bf16,
            },
            handle,
            indent=2,
        )

    print(f"\nWrote inference outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
