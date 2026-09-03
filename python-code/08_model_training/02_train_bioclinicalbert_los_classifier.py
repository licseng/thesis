"""Fine-tune Bio_ClinicalBERT for prolonged hospital length-of-stay prediction.

This script trains a binary classifier using the model-ready parquet files
created by:

    01_training_data_creation/03_create_prediction_model_dataset.py

Primary task:
    Predict prolonged hospital length of stay, defined as LOS > 7 days.

Default inputs:
    01_training_data_creation/prediction_model_dataset/train.parquet
    01_training_data_creation/prediction_model_dataset/validation.parquet

Default model:
    emilyalsentzer/Bio_ClinicalBERT

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


class TextClassificationDataset:
    """Minimal PyTorch dataset that tokenizes clinical text on demand."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = int(self.labels[index])
        return encoded


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
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return kwargs


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
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    training_args = TrainingArguments(**training_arguments_kwargs(config))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if not config.use_class_weights:
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=make_compute_metrics(),
        )

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

    return WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(),
    )


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

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label={0: "LOS_LE_7_DAYS", 1: "LOS_GT_7_DAYS"},
        label2id={"LOS_LE_7_DAYS": 0, "LOS_GT_7_DAYS": 1},
    )

    train_dataset = TextClassificationDataset(
        train_table[config.text_column].tolist(),
        train_table["labels"].astype(int).tolist(),
        tokenizer,
        config.max_length,
    )
    validation_dataset = TextClassificationDataset(
        validation_table[config.text_column].tolist(),
        validation_table["labels"].astype(int).tolist(),
        tokenizer,
        config.max_length,
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
