"""Fine-tune Bio_ClinicalBERT for prolonged hospital length-of-stay prediction.

This script trains a chunk-based binary classifier using the model-ready parquet
files created by:

    01_training_data_creation/03_create_prediction_model_dataset.py

Primary task:
    Predict prolonged hospital length of stay, defined as LOS > 7 days.

Default inputs:
    01_training_data_creation/prediction_model_dataset/train.parquet
    01_training_data_creation/prediction_model_dataset/validation.parquet

Default model:
    emilyalsentzer/Bio_ClinicalBERT

Each admission note is split into up to LOS_MAX_CHUNKS chunks of LOS_MAX_LENGTH
tokens. Bio_ClinicalBERT encodes each chunk, chunk [CLS] representations are
pooled into one admission representation, and one classification head predicts
the admission-level LOS label. The default pooling strategy concatenates
element-wise mean and max pooled chunk representations.

The script is designed for cluster training, but it supports small local smoke
tests through TRAIN_MAX_ROWS and VALIDATION_MAX_ROWS.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "01_training_data_creation" / "prediction_model_dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "bioclinicalbert_los_classifier_output"
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR.parent / ".matplotlib"))


@dataclass
class RunConfig:
    """Training configuration loaded from CLI arguments and environment."""

    model_name: str
    train_path: Path
    validation_path: Path
    output_dir: Path
    text_column: str
    label_column: str
    max_length: int
    max_chunks: int
    pooling_strategy: str
    train_max_rows: int | None
    validation_max_rows: int | None
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    warmup_ratio: float
    seed: int
    fp16: bool
    bf16: bool
    gradient_accumulation_steps: int
    logging_steps: int
    save_total_limit: int
    use_class_weights: bool
    save_safetensors: bool


def env_int(name: str, default: int | None = None) -> int | None:
    """Read an optional integer environment variable."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    """Read a float environment variable."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> RunConfig:
    """Parse CLI arguments, using environment variables as defaults."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Bio_ClinicalBERT for LOS > 7 day prediction."
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get(
            "LOS_MODEL_NAME",
            "emilyalsentzer/Bio_ClinicalBERT",
        ),
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path(os.environ.get("LOS_TRAIN_PATH", DATASET_DIR / "train.parquet")),
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=Path(
            os.environ.get("LOS_VALIDATION_PATH", DATASET_DIR / "validation.parquet")
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("LOS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
    )
    parser.add_argument("--text-column", default=os.environ.get("LOS_TEXT_COLUMN", "model_text"))
    parser.add_argument(
        "--label-column",
        default=os.environ.get("LOS_LABEL_COLUMN", "prolonged_los_gt_7d"),
    )
    parser.add_argument("--max-length", type=int, default=env_int("LOS_MAX_LENGTH", 512))
    parser.add_argument("--max-chunks", type=int, default=env_int("LOS_MAX_CHUNKS", 4))
    parser.add_argument(
        "--pooling-strategy",
        choices=["mean", "max", "mean_max"],
        default=os.environ.get("LOS_POOLING_STRATEGY", "mean_max"),
    )
    parser.add_argument("--train-max-rows", type=int, default=env_int("TRAIN_MAX_ROWS"))
    parser.add_argument(
        "--validation-max-rows",
        type=int,
        default=env_int("VALIDATION_MAX_ROWS"),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=env_float("LOS_LEARNING_RATE", 2e-5),
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=env_float("LOS_NUM_TRAIN_EPOCHS", 3.0),
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=env_int("LOS_TRAIN_BATCH_SIZE", 16),
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=env_int("LOS_EVAL_BATCH_SIZE", 32),
    )
    parser.add_argument("--weight-decay", type=float, default=env_float("LOS_WEIGHT_DECAY", 0.01))
    parser.add_argument("--warmup-ratio", type=float, default=env_float("LOS_WARMUP_RATIO", 0.06))
    parser.add_argument("--seed", type=int, default=env_int("LOS_SEED", 20260903))
    parser.add_argument("--fp16", action="store_true", default=env_bool("LOS_FP16", False))
    parser.add_argument("--bf16", action="store_true", default=env_bool("LOS_BF16", False))
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=env_int("LOS_GRADIENT_ACCUMULATION_STEPS", 1),
    )
    parser.add_argument("--logging-steps", type=int, default=env_int("LOS_LOGGING_STEPS", 50))
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=env_int("LOS_SAVE_TOTAL_LIMIT", 2),
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=env_bool("LOS_USE_CLASS_WEIGHTS", False),
    )
    parser.add_argument(
        "--save-safetensors",
        action="store_true",
        default=env_bool("LOS_SAVE_SAFETENSORS", False),
    )
    args = parser.parse_args()
    return RunConfig(**vars(args))


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible fine-tuning runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_split(path: Path, text_column: str, label_column: str, max_rows: int | None) -> pd.DataFrame:
    """Load one parquet split and standardize text/label columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet split: {path}")
    required_columns = [
        "subject_id",
        "hadm_id",
        text_column,
        label_column,
    ]
    table = pd.read_parquet(path)
    missing = sorted(set(required_columns) - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if max_rows is not None:
        table = table.head(max_rows).copy()
    table = table.loc[
        table[text_column].fillna("").astype(str).str.strip().ne("")
        & table[label_column].notna()
    ].copy()
    table[text_column] = table[text_column].fillna("").astype(str)
    table["labels"] = table[label_column].astype(bool).astype(int)
    return table


class ChunkedTextClassificationDataset:
    """PyTorch dataset that converts each admission text into fixed chunk tensors."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int,
        max_chunks: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_chunks = max_chunks
        self.chunk_token_capacity = max_length - 2
        if self.chunk_token_capacity < 1:
            raise ValueError("max_length must allow room for special tokens.")
        if self.max_chunks < 1:
            raise ValueError("max_chunks must be >= 1.")
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        if self.cls_token_id is None or self.sep_token_id is None:
            raise ValueError("Tokenizer must define CLS and SEP token IDs.")
        if self.pad_token_id is None:
            self.pad_token_id = 0

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        token_ids = self.tokenizer(
            self.texts[index],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        chunks = [
            token_ids[start : start + self.chunk_token_capacity]
            for start in range(0, len(token_ids), self.chunk_token_capacity)
        ][: self.max_chunks]
        if not chunks:
            chunks = [[]]

        input_ids = []
        attention_masks = []
        token_type_ids = []
        chunk_attention_mask = []
        for chunk in chunks:
            ids = [self.cls_token_id] + chunk + [self.sep_token_id]
            attention = [1] * len(ids)
            padding = self.max_length - len(ids)
            ids = ids + [self.pad_token_id] * padding
            attention = attention + [0] * padding
            input_ids.append(ids)
            attention_masks.append(attention)
            token_type_ids.append([0] * self.max_length)
            chunk_attention_mask.append(1)

        while len(input_ids) < self.max_chunks:
            input_ids.append([self.pad_token_id] * self.max_length)
            attention_masks.append([0] * self.max_length)
            token_type_ids.append([0] * self.max_length)
            chunk_attention_mask.append(0)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "chunk_attention_mask": torch.tensor(chunk_attention_mask, dtype=torch.float32),
            "labels": torch.tensor(int(self.labels[index]), dtype=torch.long),
        }


class ChunkPoolingFactory:
    """Factory for chunk-pooling modules over `[batch, chunks, hidden]` tensors."""

    @staticmethod
    def build(strategy: str, hidden_size: int) -> Any:
        import torch

        class MeanChunkPooling(torch.nn.Module):
            """Element-wise mean over active chunk representations."""

            output_size = hidden_size

            def forward(
                self,
                chunk_embeddings: torch.Tensor,
                chunk_attention_mask: torch.Tensor,
            ) -> torch.Tensor:
                weights = chunk_attention_mask.unsqueeze(-1).to(chunk_embeddings.dtype)
                return (chunk_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

        class MaxChunkPooling(torch.nn.Module):
            """Element-wise maximum over active chunk representations."""

            output_size = hidden_size

            def forward(
                self,
                chunk_embeddings: torch.Tensor,
                chunk_attention_mask: torch.Tensor,
            ) -> torch.Tensor:
                inactive = chunk_attention_mask.eq(0).unsqueeze(-1)
                masked = chunk_embeddings.masked_fill(inactive, torch.finfo(chunk_embeddings.dtype).min)
                pooled = masked.max(dim=1).values
                return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

        class MeanMaxChunkPooling(torch.nn.Module):
            """Concatenate element-wise mean and max over active chunks."""

            output_size = hidden_size * 2

            def __init__(self) -> None:
                super().__init__()
                self.mean_pooling = MeanChunkPooling()
                self.max_pooling = MaxChunkPooling()

            def forward(
                self,
                chunk_embeddings: torch.Tensor,
                chunk_attention_mask: torch.Tensor,
            ) -> torch.Tensor:
                mean_pooled = self.mean_pooling(chunk_embeddings, chunk_attention_mask)
                max_pooled = self.max_pooling(chunk_embeddings, chunk_attention_mask)
                return torch.cat([mean_pooled, max_pooled], dim=-1)

        if strategy == "mean":
            return MeanChunkPooling()
        if strategy == "max":
            return MaxChunkPooling()
        if strategy == "mean_max":
            return MeanMaxChunkPooling()
        raise ValueError(f"Unsupported pooling strategy: {strategy}")


class ChunkPooledBertClassifier:
    """Bio_ClinicalBERT encoder with configurable pooling over admission chunks."""

    def __new__(
        cls,
        model_name: str,
        num_labels: int = 2,
        pooling_strategy: str = "mean_max",
        dropout: float | None = None,
    ) -> Any:
        import torch
        from transformers import AutoConfig, AutoModel
        from transformers.modeling_outputs import SequenceClassifierOutput

        class _ChunkPooledBertClassifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = AutoConfig.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    id2label={0: "LOS_LE_7_DAYS", 1: "LOS_GT_7_DAYS"},
                    label2id={"LOS_LE_7_DAYS": 0, "LOS_GT_7_DAYS": 1},
                )
                self.num_labels = num_labels
                self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
                dropout_prob = (
                    dropout
                    if dropout is not None
                    else getattr(self.config, "classifier_dropout", None)
                )
                if dropout_prob is None:
                    dropout_prob = getattr(self.config, "hidden_dropout_prob", 0.1)
                self.pooling_strategy = pooling_strategy
                self.pooling = ChunkPoolingFactory.build(
                    pooling_strategy,
                    self.config.hidden_size,
                )
                self.dropout = torch.nn.Dropout(dropout_prob)
                self.classifier = torch.nn.Linear(self.pooling.output_size, num_labels)

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                chunk_attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor | None = None,
                labels: torch.Tensor | None = None,
            ) -> Any:
                batch_size, max_chunks, seq_len = input_ids.shape
                flat_input_ids = input_ids.view(batch_size * max_chunks, seq_len)
                flat_attention_mask = attention_mask.view(batch_size * max_chunks, seq_len)
                flat_token_type_ids = (
                    token_type_ids.view(batch_size * max_chunks, seq_len)
                    if token_type_ids is not None
                    else None
                )

                encoder_kwargs = {
                    "input_ids": flat_input_ids,
                    "attention_mask": flat_attention_mask,
                }
                if flat_token_type_ids is not None:
                    encoder_kwargs["token_type_ids"] = flat_token_type_ids
                outputs = self.encoder(**encoder_kwargs)
                chunk_cls = outputs.last_hidden_state[:, 0, :].view(
                    batch_size,
                    max_chunks,
                    self.config.hidden_size,
                )

                pooled = self.pooling(chunk_cls, chunk_attention_mask)
                logits = self.classifier(self.dropout(pooled))

                loss = None
                if labels is not None:
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

                return SequenceClassifierOutput(loss=loss, logits=logits)

        return _ChunkPooledBertClassifier()


def summarize_split(table: pd.DataFrame, split_name: str) -> dict[str, Any]:
    """Summarize a train/validation table before training."""
    return {
        "split": split_name,
        "n_rows": len(table),
        "n_subjects": table["subject_id"].nunique() if "subject_id" in table else pd.NA,
        "n_admissions": table["hadm_id"].nunique() if "hadm_id" in table else pd.NA,
        "n_positive": int(table["labels"].sum()),
        "pct_positive": 100 * table["labels"].mean() if len(table) else pd.NA,
        "median_words": table["model_text_n_words"].median()
        if "model_text_n_words" in table
        else pd.NA,
    }


def make_compute_metrics() -> Any:
    """Create a Hugging Face Trainer metrics callback."""
    from scipy.special import softmax
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        probabilities = softmax(logits, axis=1)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "f1": f1_score(labels, predictions, zero_division=0),
            "auprc": average_precision_score(labels, probabilities),
        }
        try:
            metrics["auroc"] = roc_auc_score(labels, probabilities)
        except ValueError:
            metrics["auroc"] = float("nan")
        return {key: float(value) for key, value in metrics.items()}

    return compute_metrics


def training_arguments_kwargs(config: RunConfig) -> dict[str, Any]:
    """Build TrainingArguments kwargs across transformers versions."""
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "num_train_epochs": config.num_train_epochs,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "auroc",
        "greater_is_better": True,
        "save_total_limit": config.save_total_limit,
        "report_to": "none",
        "seed": config.seed,
        "data_seed": config.seed,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "save_safetensors" in signature.parameters:
        kwargs["save_safetensors"] = config.save_safetensors
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return kwargs


def trainer_tokenizer_kwargs(trainer_class: Any, tokenizer: Any) -> dict[str, Any]:
    """Return tokenizer/processing_class kwargs compatible with this transformers version."""
    signature = inspect.signature(trainer_class.__init__)
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in signature.parameters:
        return {"tokenizer": tokenizer}
    return {}


def build_trainer(
    config: RunConfig,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    validation_dataset: Any,
    train_table: pd.DataFrame,
) -> Any:
    """Build a Trainer, optionally with inverse-frequency class weights."""
    import torch
    from transformers import Trainer, TrainingArguments, default_data_collator

    training_args = TrainingArguments(**training_arguments_kwargs(config))
    base_trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": default_data_collator,
        "compute_metrics": make_compute_metrics(),
    }
    base_trainer_kwargs.update(trainer_tokenizer_kwargs(Trainer, tokenizer))

    if not config.use_class_weights:
        return Trainer(**base_trainer_kwargs)

    counts = train_table["labels"].value_counts().reindex([0, 1], fill_value=0)
    if (counts == 0).any():
        raise ValueError(
            "Cannot use class weights because one class is absent in the training split."
        )
    weights = len(train_table) / (2 * counts)
    class_weights = torch.tensor(weights.to_numpy(dtype="float32"), dtype=torch.float32)

    class WeightedTrainer(Trainer):
        """Trainer with weighted cross-entropy for imbalanced binary labels."""

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **kwargs: Any,
        ) -> Any:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
            loss = loss_fct(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    weighted_trainer_kwargs = dict(base_trainer_kwargs)
    weighted_trainer_kwargs.update(trainer_tokenizer_kwargs(WeightedTrainer, tokenizer))
    return WeightedTrainer(**weighted_trainer_kwargs)


def save_validation_predictions(
    trainer: Any,
    validation_dataset: Any,
    validation_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save validation probabilities and predictions for inspection."""
    from scipy.special import softmax

    prediction_output = trainer.predict(validation_dataset)
    probabilities = softmax(prediction_output.predictions, axis=1)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    output_columns = [
        column
        for column in [
            "subject_id",
            "hadm_id",
            "note_id",
            "hospital_los_days",
            "prolonged_los_gt_7d",
            "readmission_within_30d",
            "is_mhh1_psychotic_admission",
            "is_mhc0_admission",
            "is_matched_mhh1_psychotic_admission",
            "is_matched_mhc0_admission",
            "model_text_n_words",
        ]
        if column in validation_table.columns
    ]
    predictions_df = validation_table.loc[:, output_columns].copy()
    predictions_df["true_label"] = validation_table["labels"].to_numpy()
    predictions_df["predicted_probability"] = probabilities
    predictions_df["predicted_label"] = predictions
    predictions_df.to_csv(output_dir / "validation_predictions.csv", index=False)


def make_model_parameters_contiguous(model: Any) -> None:
    """Pack any non-contiguous trainable tensors before checkpoint saving."""
    import torch

    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.is_contiguous():
                parameter.data = parameter.data.contiguous()


def save_training_curves(trainer: Any, output_dir: Path) -> None:
    """Save Trainer log history as CSV and simple training/evaluation plots."""
    import matplotlib.pyplot as plt

    history = pd.DataFrame(trainer.state.log_history)
    if history.empty:
        return
    history.to_csv(output_dir / "training_log_history.csv", index=False)

    if "step" not in history.columns:
        return

    train_loss = history.loc[history["loss"].notna()] if "loss" in history else pd.DataFrame()
    eval_loss = (
        history.loc[history["eval_loss"].notna()]
        if "eval_loss" in history
        else pd.DataFrame()
    )
    if not train_loss.empty or not eval_loss.empty:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        if not train_loss.empty:
            ax.plot(train_loss["step"], train_loss["loss"], marker="o", label="train loss")
        if not eval_loss.empty:
            ax.plot(eval_loss["step"], eval_loss["eval_loss"], marker="o", label="validation loss")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Validation Loss")
        ax.legend()
        fig.savefig(output_dir / "training_validation_loss_curve.png", dpi=200)
        plt.close(fig)

    metric_columns = [
        column
        for column in [
            "eval_auroc",
            "eval_auprc",
            "eval_f1",
            "eval_recall",
            "eval_specificity",
            "eval_accuracy",
        ]
        if column in history.columns and history[column].notna().any()
    ]
    if metric_columns:
        eval_history = history.loc[history[metric_columns].notna().any(axis=1)]
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for column in metric_columns:
            ax.plot(eval_history["step"], eval_history[column], marker="o", label=column)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Metric")
        ax.set_ylim(0, 1)
        ax.set_title("Validation Metrics")
        ax.legend()
        fig.savefig(output_dir / "validation_metric_curves.png", dpi=200)
        plt.close(fig)


def main() -> None:
    """Run Bio_ClinicalBERT fine-tuning."""
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)

    from transformers import AutoTokenizer

    train_table = load_split(
        config.train_path,
        config.text_column,
        config.label_column,
        config.train_max_rows,
    )
    validation_table = load_split(
        config.validation_path,
        config.text_column,
        config.label_column,
        config.validation_max_rows,
    )
    split_summary = pd.DataFrame(
        [
            summarize_split(train_table, "train"),
            summarize_split(validation_table, "validation"),
        ]
    )
    split_summary.to_csv(config.output_dir / "training_split_summary.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = ChunkPooledBertClassifier(
        config.model_name,
        num_labels=2,
        pooling_strategy=config.pooling_strategy,
    )
    make_model_parameters_contiguous(model)

    train_dataset = ChunkedTextClassificationDataset(
        train_table[config.text_column].tolist(),
        train_table["labels"].astype(int).tolist(),
        tokenizer,
        config.max_length,
        config.max_chunks,
    )
    validation_dataset = ChunkedTextClassificationDataset(
        validation_table[config.text_column].tolist(),
        validation_table["labels"].astype(int).tolist(),
        tokenizer,
        config.max_length,
        config.max_chunks,
    )

    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
            handle,
            indent=2,
        )

    print("Training split summary:")
    print(split_summary.to_string(index=False))
    print(f"\nLoading model: {config.model_name}")
    print(f"Output directory: {config.output_dir}")

    trainer = build_trainer(
        config,
        model,
        tokenizer,
        train_dataset,
        validation_dataset,
        train_table,
    )
    train_result = trainer.train()
    make_model_parameters_contiguous(model)
    trainer.save_model(config.output_dir / "best_model")
    tokenizer.save_pretrained(config.output_dir / "best_model")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    validation_metrics = trainer.evaluate(validation_dataset)
    trainer.log_metrics("validation", validation_metrics)
    trainer.save_metrics("validation", validation_metrics)
    with (config.output_dir / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(validation_metrics, handle, indent=2)

    save_training_curves(trainer, config.output_dir)
    save_validation_predictions(
        trainer,
        validation_dataset,
        validation_table,
        config.output_dir,
    )
    print("\nValidation metrics:")
    print(json.dumps(validation_metrics, indent=2))


if __name__ == "__main__":
    main()
