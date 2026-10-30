"""Psychiatric-keyword masking for the frozen admission classifiers.

Run one task per GPU job; see run_interpretability_analysis.sbatch. All matched
test-pool admissions eligible for that task are retained, including keyword-free
admissions. Matching wordpieces are replaced by [MASK] AFTER tokenization and
truncation, preserving all chunk boundaries, positions, and attention masks.
Choose broad psychiatric vocabulary or psychosis-related diagnostic labels with
--masking-mode. All groups in that mode are masked jointly. Changes are
descriptive perturbation effects, not causal effects or individual attributions.

Outputs contain identifiers, scores, labels, and counts; no note text or quotes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
VOCABULARY_SOURCE = "03_discharge_note_text_analysis/02_keyword_matching/02_explore_psych_keywords_by_section.py"
# Snapshot of that script's KEYWORD_PATTERNS, so cluster jobs need no other
# analysis folders. The exact patterns used are also saved in the manifest.
KEYWORD_PATTERNS = {
    "psychosis_schizophrenia": [r"\bschizophren\w*\b", r"\bschizoaffective\b", r"\bpsychosis\b", r"\bpsychotic\b", r"\bparanoid schizophrenia\b"],
    "bipolar_mania": [r"\bbipolar\b", r"\bmania\b", r"\bmanic\b"],
    "psychiatry_general": [r"\bpsychiatr\w*\b", r"\bmental health\b", r"\bbehavioral health\b", r"\bpsych history\b", r"\bpsych hx\b", r"\bpsychiatric history\b"],
    "common_psych_meds": [r"\bhaloperidol\b", r"\bhaldol\b", r"\brisperidone\b", r"\brisperdal\b", r"\bolanzapine\b", r"\bzyprexa\b", r"\bquetiapine\b", r"\bseroquel\b", r"\bclozapine\b", r"\bclozaril\b", r"\baripiprazole\b", r"\babilify\b", r"\bziprasidone\b", r"\bgeodon\b", r"\bpaliperidone\b", r"\binvega\b", r"\bfluphenazine\b", r"\bperphenazine\b", r"\bchlorpromazine\b", r"\bthorazine\b", r"\blurasidone\b", r"\blatuda\b"],
    "other_psych_conditions": [r"\bdepression\b", r"\bdepressive\b", r"\banxiety\b", r"\bptsd\b", r"\bpersonality disorder\b", r"\bsuicid\w*\b", r"\bself harm\b"],
    "substance_use": [r"\balcohol use\b", r"\bsubstance use\b", r"\bsubstance abuse\b", r"\bcocaine\b", r"\bopioid\b", r"\bheroin\b", r"\bmarijuana\b", r"\bcannabis\b", r"\betoh\b"],
    "cognitive_behavioral": [r"\bdelirium\b", r"\bdementia\b", r"\baltered mental status\b", r"\bagitation\b", r"\bconfusion\b", r"\bconfused\b", r"\bsomatization\b", r"\bsomatizing\b", r"\bsomatize\w*\b", r"\bsomatic symptom\w*\b", r"\bsomatic complaint\w*\b", r"\bsomatoform\b", r"\bpsychosomatic\b", r"\bconversion disorder\b", r"\bfunctional neurologic\w*\b", r"\bfunctional neurological disorder\b", r"\bmedically unexplained symptom\w*\b", r"\bhypochondriasis\b", r"\billness anxiety\b"],
}

# Psychosis-related diagnostic LABEL MENTIONS only, selected for the narrower
# experiment. Bare psychosis/psychotic, mania, depression, symptoms, general
# psychiatry words and medications are not matched. Named diagnostic phrases
# may themselves contain symptom words (e.g. "psychotic depression"). This is
# lexical masking, including negated/family-history mentions, not diagnosis NLP.
# Ambiguous abbreviations such as SAD and BPD are intentionally not included.
DIAGNOSTIC_LABEL_PATTERNS = {
    "schizophrenia_spectrum_labels": [
        r"\b(?:(?:paranoid|disorgani[sz]ed|catatonic|undifferentiated|residual)\s+)?schizophren(?:ia|ic|ics)\b",
        r"\bschizoaffective(?:\s+disorder)?\b",
        r"\bschizophreniform(?:\s+(?:disorder|psychosis))?\b",
        r"\bschizotypal(?:\s+(?:personality\s+)?disorder)?\b",
    ],
    "bipolar_disorder_labels": [
        r"\bbipolar(?:\s+(?:type\s+)?(?:ii|i|2|1))?(?:\s+(?:affective\s+)?disorder)?\b",
    ],
    "psychotic_depression_labels": [
        r"\bpsychotic\s+depress(?:ion|ive\s+disorder)\b",
        r"\bdepressive(?:[\s-]+type)?\s+psychosis\b",
        r"\b(?:(?:major\s+)?depress(?:ion|ive\s+disorder)|mdd)(?:\s*,\s*|\s+)(?:(?:recurrent|severe|single[\s-]+episode)(?:\s*,\s*|\s+))*with\s+psychotic\s+(?:features|symptoms)\b",
    ],
    "other_named_psychotic_disorder_labels": [
        r"\b(?:delusional|paranoid)\s+disorder\b",
        r"\b(?:brief|acute|transient|reactive|non[\s-]?organic|unspecified|primary|post[\s-]?partum|puerperal|atypical)\s+(?:psychotic\s+(?:disorder|illness)|psychosis)\b",
        r"\bacute\s+and\s+transient\s+psychotic\s+disorder\b",
        r"\b(?:shared|induced)\s+(?:delusional\s+disorder|psychotic\s+disorder|psychosis)\b",
        r"\bpsychotic\s+(?:disorder|illness)(?:\s+(?:nos|not\s+otherwise\s+specified))?\b",
        r"\bpsychosis\s*,?\s*(?:nos|not\s+otherwise\s+specified|unspecified)\b",
    ],
}
MASKING_VOCABULARIES = {"broad": KEYWORD_PATTERNS, "diagnostic_labels": DIAGNOSTIC_LABEL_PATTERNS}


def load_inference_module():
    path = SCRIPT_DIR / "04_run_admission_classifier_inference.py"
    spec = importlib.util.spec_from_file_location("masking_inference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["los", "readmission"])
    parser.add_argument("--training-output-dir", type=Path)
    parser.add_argument("--masking-mode", choices=list(MASKING_VOCABULARIES), default="broad",
                        help="broad: previous psychiatric filter; diagnostic_labels: psychosis-related diagnosis label mentions only")
    parser.add_argument("--summarize-saved-dir", type=Path,
                        help="Offline: validate saved scores and summarize the complete-pair fairness subset; no model loading")
    parser.add_argument("--dataset", type=Path, default=SCRIPT_DIR / "01_training_data_creation/prediction_model_dataset_current_cohort_test/test_fairness_mhc1_mhc0.parquet")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, help="Defaults to the frozen LOS=.25 / readmission=.20 threshold")
    parser.add_argument("--max-rows", type=int, help="Optional first-N smoke test; omit for the full analysis")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--reference-predictions", type=Path, help="Optional previous inference CSV for baseline reproduction checks")
    parser.add_argument("--bootstrap-samples", type=int, default=0,
                        help="Offline summary only: number of patient/pair-block bootstrap samples; use 5000 for uncertainty estimates")
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.bootstrap_samples < 0 or 0 < args.bootstrap_samples < 100:
        parser.error("Bootstrap samples must be zero or at least 100")
    if args.summarize_saved_dir:
        return args
    if args.bootstrap_samples:
        parser.error("Bootstrap uncertainty uses saved outputs: specify --summarize-saved-dir")
    if args.task is None or args.training_output_dir is None:
        parser.error("Inference requires --task and --training-output-dir")
    if args.batch_size < 1 or (args.max_rows is not None and args.max_rows < 1):
        parser.error("Batch size and max rows must be positive")
    if args.threshold is None:
        args.threshold = .25 if args.task == "los" else .20
    if not 0 < args.threshold < 1:
        parser.error("Threshold must lie between zero and one")
    if args.output_dir is None:
        suffix = args.task if args.masking_mode == "broad" else f"{args.masking_mode}_{args.task}"
        args.output_dir = SCRIPT_DIR / f"analysis_output_interpretability_{suffix}"
    return args


def merge_spans(spans):
    """Union overlapping matches to avoid double-counting nested expressions."""
    merged = []
    for start, end in sorted(set(spans)):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def prepare_batch(texts, tokenizer, max_length, max_chunks, patterns):
    """Return original/masked tensors with identical lengths and chunk layout."""
    import torch

    capacity = max_length - 2
    if capacity < 1 or max_chunks < 1:
        raise ValueError("Invalid trained chunk dimensions")
    if not tokenizer.is_fast or tokenizer.mask_token_id is None:
        raise ValueError("A fast tokenizer with offsets and a MASK token is required")
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True,
                        max_length=capacity * max_chunks, return_offsets_mapping=True)
    shape = (len(texts), max_chunks, max_length)
    original = {
        "input_ids": torch.full(shape, tokenizer.pad_token_id, dtype=torch.long),
        "attention_mask": torch.zeros(shape, dtype=torch.long),
        "token_type_ids": torch.zeros(shape, dtype=torch.long),
        "chunk_attention_mask": torch.zeros(shape[:2], dtype=torch.float32),
    }
    masked_ids = original["input_ids"].clone()
    stats = []
    for i, (text, ids, offsets) in enumerate(zip(texts, encoded["input_ids"], encoded["offset_mapping"])):
        token_hits = np.zeros(len(ids), dtype=bool)
        all_spans = []
        row = {"n_visible_text_tokens": len(ids)}
        for group, regexes in patterns.items():
            spans = merge_spans([m.span() for regex in regexes for m in regex.finditer(text)])
            group_hits = np.zeros(len(ids), dtype=bool)
            for start, end in spans:
                group_hits |= np.array([a < end and b > start for a, b in offsets], dtype=bool)
            token_hits |= group_hits
            row[f"n_masked_tokens_{group}"] = int(group_hits.sum())
            all_spans.extend(spans)
        spans = merge_spans(all_spans)
        visible_spans = sum(any(a < end and b > start for a, b in offsets) for start, end in spans)
        row.update(n_keyword_spans_in_text=len(spans), n_keyword_spans_in_visible_input=visible_spans,
                   n_keyword_spans_outside_visible_input=len(spans) - visible_spans,
                   n_masked_tokens=int(token_hits.sum()))
        for chunk_index, start in enumerate(range(0, max(len(ids), 1), capacity)):
            content = ids[start:start + capacity]
            sequence = [tokenizer.cls_token_id] + content + [tokenizer.sep_token_id]
            n = len(sequence)
            original["input_ids"][i, chunk_index, :n] = torch.tensor(sequence)
            original["attention_mask"][i, chunk_index, :n] = 1
            original["chunk_attention_mask"][i, chunk_index] = 1
            masked_ids[i, chunk_index, :n] = torch.tensor(sequence)
            for j in np.flatnonzero(token_hits[start:start + capacity]):
                masked_ids[i, chunk_index, int(j) + 1] = tokenizer.mask_token_id
        stats.append(row)
    masked = {**original, "input_ids": masked_ids}
    return original, masked, stats


def predict(model, inputs, device, fp16):
    import torch
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                               enabled=fp16 and device.type == "cuda"):
        logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
    return torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()


def summarize(results, inference, threshold, keyword_patterns=None):
    if keyword_patterns is None:
        keyword_patterns = KEYWORD_PATTERNS
    summary, metrics, coverage = [], [], []
    groups = [("all", results), *list(results.groupby("cohort", sort=True))]
    for cohort, cohort_rows in groups:
        for population, rows in [("all_eligible", cohort_rows), ("visible_keyword_hits", cohort_rows.loc[cohort_rows.n_masked_tokens.gt(0)])]:
            row = {"cohort": cohort, "population": population, "n_admissions": len(rows),
                   "n_subjects": rows.subject_id.nunique()}
            if len(rows):
                delta = rows.probability_change_pp
                row.update(mean_original_probability=rows.original_probability.mean(),
                           mean_masked_probability=rows.masked_probability.mean(),
                           mean_change_pp=delta.mean(), median_change_pp=delta.median(),
                           q25_change_pp=delta.quantile(.25), q75_change_pp=delta.quantile(.75),
                           mean_absolute_change_pp=delta.abs().mean(),
                           n_original_positive=int(rows.original_label.sum()),
                           n_masked_positive=int(rows.masked_label.sum()),
                           n_positive_to_negative=int((rows.original_label.eq(1) & rows.masked_label.eq(0)).sum()),
                           n_negative_to_positive=int((rows.original_label.eq(0) & rows.masked_label.eq(1)).sum()))
                for condition in ["original", "masked"]:
                    values = inference.compute_metrics(rows.true_label.to_numpy(), rows[f"{condition}_probability"].to_numpy(), rows[f"{condition}_label"].to_numpy())
                    metrics.append({"cohort": cohort, "population": population, "condition": condition,
                                    "threshold": threshold, **values})
            summary.append(row)
        for group in keyword_patterns:
            hits = cohort_rows[f"n_masked_tokens_{group}"].gt(0)
            coverage.append({"cohort": cohort, "keyword_group": group, "n_admissions": len(cohort_rows),
                             "n_with_visible_keyword": int(hits.sum()),
                             "pct_with_visible_keyword": 100 * hits.mean()})
    return pd.DataFrame(summary), pd.DataFrame(metrics), pd.DataFrame(coverage)


def main():
    args = parse_args()
    if args.summarize_saved_dir:
        summarize_saved_outputs(args.summarize_saved_dir, args.bootstrap_samples, args.seed)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new --output-dir")
    inference = load_inference_module()
    config = inference.load_run_config(args.training_output_dir)
    label = "prolonged_los_gt_7d" if args.task == "los" else "readmission_within_30d"
    if config.get("label_column") != label:
        raise ValueError("Training configuration does not match the requested task")
    text_column = config.get("text_column", "model_text")
    filter_column = "eligible_for_30d_readmission" if args.task == "readmission" else None
    table = inference.load_eval_split(args.dataset, text_column, label, filter_column, args.max_rows)
    if "cohort" not in table or table.cohort.isna().any():
        raise ValueError("Cohort labels are required")
    if not table[label].isin([0, 1, False, True]).all():
        raise ValueError("Expected binary target labels")

    import torch
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU was requested but is unavailable")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = inference.load_training_module()
    model, tokenizer = inference.build_model_and_tokenizer(
        training, args.training_output_dir / "best_model", config["model_name"],
        config["pooling_strategy"], device)
    max_length, max_chunks = int(config["max_length"]), int(config["max_chunks"])
    vocabulary = MASKING_VOCABULARIES[args.masking_mode]
    patterns = {g: [re.compile(p, re.I) for p in expressions] for g, expressions in vocabulary.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Task={args.task}; mode={args.masking_mode}; device={device}; admissions={len(table):,}; joint keyword masking", flush=True)
    scores, counts = [], []
    for start in range(0, len(table), args.batch_size):
        batch = table.iloc[start:start + args.batch_size]
        original, masked, stats = prepare_batch(batch[text_column].tolist(), tokenizer, max_length, max_chunks, patterns)
        original_p = predict(model, original, device, args.fp16)
        masked_p = original_p.copy()
        changed = np.array([r["n_masked_tokens"] > 0 for r in stats])
        if changed.any():
            masked_p[changed] = predict(model, {k: v[torch.tensor(changed)] for k, v in masked.items()}, device, args.fp16)
        scores.extend(zip(original_p.tolist(), masked_p.tolist()))
        counts.extend(stats)
        if start == 0 or (start // args.batch_size + 1) % 25 == 0 or start + args.batch_size >= len(table):
            print(f"Processed {min(start + args.batch_size, len(table)):,}/{len(table):,}", flush=True)
    metadata = [c for c in ["subject_id", "hadm_id", "pair_id", "cohort", "note_id", "partition"] if c in table]
    results = table[metadata].reset_index(drop=True).copy()
    results["true_label"] = table.labels.to_numpy()
    results["masking_mode"] = args.masking_mode
    results[["original_probability", "masked_probability"]] = scores
    results = pd.concat([results, pd.DataFrame(counts)], axis=1)
    results["probability_change_pp"] = 100 * (results.masked_probability - results.original_probability)
    for condition in ["original", "masked"]:
        results[f"{condition}_label"] = results[f"{condition}_probability"].ge(args.threshold).astype(int)
    results.to_csv(args.output_dir / "admission_masking_effects.csv", index=False)
    summary, metrics, coverage = summarize(results, inference, args.threshold, vocabulary)
    summary.to_csv(args.output_dir / "cohort_summary.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics_original_vs_masked.csv", index=False)
    coverage.to_csv(args.output_dir / "keyword_coverage.csv", index=False)
    reference_check = None
    if args.reference_predictions:
        ref = pd.read_csv(args.reference_predictions)
        check = results.merge(ref[["subject_id", "hadm_id", "true_label", "predicted_probability"]],
                              on=["subject_id", "hadm_id"], validate="one_to_one", suffixes=("", "_reference"))
        if len(check) != len(results) or not check.true_label.eq(check.true_label_reference).all():
            raise ValueError("Reference predictions do not cover the same admissions and labels")
        difference = (check.original_probability - check.predicted_probability).abs()
        reference_check = {"path": str(args.reference_predictions), "n_compared": len(check),
                           "max_absolute_probability_difference": float(difference.max()),
                           "mean_absolute_probability_difference": float(difference.mean()),
                           "n_differences_above_1e_minus_4": int(difference.gt(1e-4).sum())}
        print(f"Baseline reproduction: max absolute difference={difference.max():.6g}", flush=True)
    manifest = {
        "status": "complete", "task": args.task, "masking_mode": args.masking_mode,
        "training_output_dir": str(args.training_output_dir.resolve()),
        "dataset": str(args.dataset.resolve()), "n_admissions": len(results), "threshold": args.threshold,
        "max_rows": args.max_rows, "batch_size": args.batch_size, "fp16": args.fp16, "device": str(device),
        "torch_version": torch.__version__, "training_config": config,
        "max_length": max_length, "max_chunks": max_chunks, "mask_token": tokenizer.mask_token,
        "mask_rule": "Joint masking of all matched wordpieces within the original truncated input; positions and chunk boundaries preserved",
        "change_direction": "masked minus original; probability_change_pp is in percentage points",
        "vocabulary_source": VOCABULARY_SOURCE if args.masking_mode == "broad" else "06_interpretability_analysis.py::DIAGNOSTIC_LABEL_PATTERNS",
        "keyword_patterns": vocabulary,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_admissions_with_visible_keywords": int(results.n_masked_tokens.gt(0).sum()),
        "n_admissions_with_keywords_only_outside_input": int((results.n_keyword_spans_in_text.gt(0) & results.n_masked_tokens.eq(0)).sum()),
        "reference_prediction_check": reference_check,
        "interpretation": "Admission-weighted descriptive perturbation effects. " + (
            "Broad vocabulary includes medications, substance use and cognitive symptoms. " if args.masking_mode == "broad" else
            "Psychosis-related diagnostic label mentions only; includes negation and family history; bare symptoms and medication names remain. ") +
            "No causal or unfairness inference; no significance tests.",
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed: {args.output_dir}; {manifest['n_admissions_with_visible_keywords']:,} admissions with masked tokens", flush=True)


def summarize_saved_outputs(folder, n_bootstrap=0, seed=20261030):
    """Inspect scores locally on the frozen primary complete-pair population."""
    manifest = json.loads((folder / "run_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("max_rows") is not None:
        raise ValueError("Expected a completed full masking run")
    task = manifest["task"]
    if task not in {"los", "readmission"}:
        raise ValueError("Unknown saved prediction task")
    threshold = manifest["threshold"]
    source = pd.read_csv(folder / "admission_masking_effects.csv")
    reference_dir = SCRIPT_DIR / f"bioclinicalbert_{task}_classifier_inference_output_current_cohort"
    reference = pd.read_csv(reference_dir / "test_fairness_mhc1_mhc0_predictions.csv")
    paired_path = reference_dir / "paired_fairness_analysis/paired_test_predictions.csv"
    paired = pd.read_csv(paired_path)
    keys = ["pair_id", "cohort", "subject_id", "hadm_id"]
    if len(source) != manifest["n_admissions"] or source.hadm_id.duplicated().any():
        raise ValueError("Saved run count or admission uniqueness is incorrect")
    for condition in ["original", "masked"]:
        if not source[f"{condition}_probability"].between(0, 1).all():
            raise ValueError("Expected finite probabilities between zero and one")
        if not source[f"{condition}_label"].eq(source[f"{condition}_probability"].ge(threshold).astype(int)).all():
            raise ValueError("Saved threshold decisions are inconsistent")
    if not np.allclose(source.probability_change_pp, 100 * (source.masked_probability - source.original_probability), atol=1e-10):
        raise ValueError("Probability changes are inconsistent")
    if not source.loc[source.n_masked_tokens.eq(0), "probability_change_pp"].eq(0).all():
        raise ValueError("Keyword-free inputs unexpectedly changed")
    check = reference[keys + ["true_label", "predicted_probability", "predicted_label"]].merge(
        source, on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("_reference", ""))
    if not check._merge.eq("both").all() or not check.true_label_reference.eq(check.true_label).all():
        raise ValueError("Saved run does not match the original task admissions and labels")
    complete = paired[keys].merge(source, on=keys, how="left", validate="one_to_one", indicator=True)
    if not complete._merge.eq("both").all():
        raise ValueError("Masking run does not cover all primary complete pairs")
    complete = complete.drop(columns="_merge")
    if not complete.groupby("pair_id").size().eq(2).all() or not complete.groupby("pair_id").cohort.nunique().eq(2).all():
        raise ValueError("Incomplete or incorrectly labelled primary pairs")
    inference = load_inference_module()
    saved_vocabulary = manifest.get("keyword_patterns", KEYWORD_PATTERNS)
    masking_mode = manifest.get("masking_mode", "broad")
    summary, metrics, coverage = summarize(complete, inference, threshold, saved_vocabulary)
    from sklearn.metrics import brier_score_loss
    # Add probability-scale calibration summaries to the existing classification
    # metrics. Differences below always compare the two conditions in this run.
    for index, row in metrics.iterrows():
        data = complete if row.cohort == "all" else complete.loc[complete.cohort.eq(row.cohort)]
        if row.population == "visible_keyword_hits":
            data = data.loc[data.n_masked_tokens.gt(0)]
        p = data[f"{row.condition}_probability"]
        metrics.loc[index, "brier_score"] = brier_score_loss(data.true_label, p)
        metrics.loc[index, "mean_predicted_probability_pct"] = 100 * p.mean()
        metrics.loc[index, "observed_minus_predicted_percentage_points"] = 100 * (data.true_label.mean() - p.mean())
    for index, row in summary.iterrows():
        data = complete if row.cohort == "all" else complete.loc[complete.cohort.eq(row.cohort)]
        if row.population == "visible_keyword_hits":
            data = data.loc[data.n_masked_tokens.gt(0)]
        original_correct = data.original_label.eq(data.true_label)
        masked_correct = data.masked_label.eq(data.true_label)
        summary.loc[index, "n_correct_to_incorrect"] = int((original_correct & ~masked_correct).sum())
        summary.loc[index, "n_incorrect_to_correct"] = int((~original_correct & masked_correct).sum())
        summary.loc[index, "pct_admissions_with_visible_keywords"] = 100 * data.n_masked_tokens.gt(0).mean()
        summary.loc[index, "p95_absolute_change_pp"] = data.probability_change_pp.abs().quantile(.95)
        summary.loc[index, "max_absolute_change_pp"] = data.probability_change_pp.abs().max()
    changes = []
    for (cohort, population), rows in metrics.groupby(["cohort", "population"]):
        indexed = rows.set_index("condition")
        for metric in ["auroc", "auprc", "recall", "specificity", "precision", "f1", "brier_score", "mean_predicted_probability_pct", "observed_minus_predicted_percentage_points"]:
            changes.append({"cohort": cohort, "population": population, "metric": metric,
                "original": indexed.loc["original", metric], "masked": indexed.loc["masked", metric],
                "masked_minus_original": indexed.loc["masked", metric] - indexed.loc["original", metric]})
    destination = folder / "paired_analysis"
    destination.mkdir(exist_ok=True)
    complete.to_csv(destination / "admission_masking_effects.csv", index=False)
    summary.to_csv(destination / "cohort_summary.csv", index=False)
    metrics.to_csv(destination / "metrics_original_vs_masked.csv", index=False)
    coverage.to_csv(destination / "keyword_coverage.csv", index=False)
    pd.DataFrame(changes).to_csv(destination / "metric_changes.csv", index=False)
    difference = (check.predicted_probability - check.original_probability).abs()
    (destination / "inspection_manifest.json").write_text(json.dumps({
        "task": task, "source_sha256": hashlib.sha256((folder / "admission_masking_effects.csv").read_bytes()).hexdigest(),
        "masking_mode": masking_mode,
        "paired_prediction_source": str(paired_path), "n_pairs": complete.pair_id.nunique(),
        "n_admissions": len(complete), "threshold": threshold,
        "all_test_baseline_max_absolute_probability_difference": float(difference.max()),
        "all_test_baseline_mean_absolute_probability_difference": float(difference.mean()),
        "all_test_baseline_decision_disagreements": int(check.predicted_label.ne(check.original_label).sum()),
        "comparison_baseline": "Original probabilities recomputed in the masking run; original frozen fairness membership retained",
        "weighting": "Admission-weighted summaries; requested bootstrap intervals are saved separately in uncertainty/",
        "bootstrap_samples_requested": n_bootstrap,
        "vocabulary": "All groups in the run's saved keyword_patterns were masked jointly; no assertion/experiencer detection or individual keyword effects",
        "keyword_patterns": saved_vocabulary,
        "model_loaded_or_inference_run": False,
    }, indent=2) + "\n")
    print(f"{task}: validated {len(source):,} admissions; summarized {complete.pair_id.nunique():,} complete pairs in {destination}")
    print(summary.loc[summary.population.eq("all_eligible"), ["cohort", "n_admissions", "mean_change_pp", "mean_absolute_change_pp", "n_positive_to_negative", "n_negative_to_positive"]].to_string(index=False))
    if n_bootstrap:
        summarize_masking_uncertainty(complete, paired, manifest, folder, n_bootstrap, seed)


def summarize_masking_uncertainty(complete, paired, run_manifest, folder, n_bootstrap, seed):
    """Paired-condition block bootstrap using the primary fairness resampling.

    The same sampled block multiplicities weight both cohorts and both masking
    conditions. Thus changes and changes in cohort gaps are calculated within
    each replicate; independent intervals are never subtracted from each other.
    """
    spec = importlib.util.spec_from_file_location("masking_fairness", SCRIPT_DIR / "05_fairness_analysis.py")
    fairness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fairness
    spec.loader.exec_module(fairness)
    task = run_manifest["task"]
    threshold = run_manifest["threshold"]
    if threshold != fairness.load_threshold(fairness.INFERENCE_OUTPUT_DIRS[task]):
        raise ValueError("Masking and primary fairness decision thresholds differ")
    data = complete.copy()
    data["bootstrap_block_id"] = fairness.patient_pair_blocks(data)
    keys = ["pair_id", "cohort", "subject_id", "hadm_id"]
    block_check = data[keys + ["bootstrap_block_id"]].merge(
        paired[keys + ["bootstrap_block_id", "true_label"]], on=keys,
        validate="one_to_one", suffixes=("", "_fairness"))
    if len(block_check) != len(data) or not block_check.bootstrap_block_id.eq(block_check.bootstrap_block_id_fairness).all():
        raise ValueError("Bootstrap blocks differ from the primary paired fairness analysis")
    truth_check = data[keys + ["true_label"]].merge(paired[keys + ["true_label"]], on=keys, validate="one_to_one", suffixes=("", "_fairness"))
    if not truth_check.true_label.eq(truth_check.true_label_fairness).all():
        raise ValueError("Outcome labels differ from the primary paired analysis")
    n_blocks = data.bootstrap_block_id.nunique()
    if n_blocks < 2:
        raise ValueError("At least two resampling blocks are required")
    cohorts = ["MHC1-psychosis", "MHC0"]
    populations = ["all_admissions", "visible_keyword_hits"]
    effects = ["mean_probability_change_pp", "mean_absolute_probability_change_pp"]
    metric_names = fairness.BOOTSTRAP_METRICS
    point_metrics = np.empty((2, 2, len(metric_names)))
    point_effects = np.full((2, 2, 2), np.nan)
    evaluators, block_ids, delta, hit = [], [], [], []
    for i, cohort in enumerate(["MHC1_psychotic", "MHC0"]):
        part = data.loc[data.cohort.eq(cohort)]
        block_ids.append(part.bootstrap_block_id.to_numpy())
        delta.append(part.probability_change_pp.to_numpy())
        hit.append(part.n_masked_tokens.gt(0).to_numpy())
        conditions = []
        for j, condition in enumerate(["original", "masked"]):
            scores = pd.DataFrame({"true_label": part.true_label,
                "predicted_probability": part[f"{condition}_probability"],
                "predicted_label": part[f"{condition}_label"]})
            evaluator = fairness.WeightedEvaluation(scores)
            conditions.append(evaluator)
            point_metrics[i, j], _ = evaluator.calculate(np.ones(len(part)))
        evaluators.append(conditions)
        for population_index, selection in enumerate([np.ones(len(part), dtype=bool), hit[i]]):
            if selection.any():
                point_effects[i, population_index] = [delta[i][selection].mean(), np.abs(delta[i][selection]).mean()]
    metric_samples = np.full((n_bootstrap, 2, 2, len(metric_names)), np.nan)
    effect_samples = np.full((n_bootstrap, 2, 2, 2), np.nan)
    rng = np.random.default_rng(seed)
    print(f"{task}: masking uncertainty, {n_blocks:,} patient/pair blocks, {n_bootstrap:,} bootstrap samples", flush=True)
    for replicate in range(n_bootstrap):
        multiplicity = rng.multinomial(n_blocks, np.full(n_blocks, 1 / n_blocks))
        for i in range(2):
            weights = multiplicity[block_ids[i]]
            for j in range(2):
                metric_samples[replicate, i, j], _ = evaluators[i][j].calculate(weights)
            for population_index, selection in enumerate([np.ones(len(weights), dtype=bool), hit[i]]):
                selected_weights = weights * selection
                denominator = selected_weights.sum()
                if denominator:
                    effect_samples[replicate, i, population_index] = [
                        np.dot(selected_weights, delta[i]) / denominator,
                        np.dot(selected_weights, np.abs(delta[i])) / denominator]
        if (replicate + 1) % 500 == 0:
            print(f"{task}: masking bootstrap {replicate + 1:,}/{n_bootstrap:,}", flush=True)
    effect_rows, metric_rows, gap_rows = [], [], []

    def interval_record(estimate, samples):
        lower, upper, valid = fairness.interval_summary(samples)
        return {"estimate": float(estimate), "ci_lower": lower, "ci_upper": upper,
                "n_valid_bootstrap": valid, "n_bootstrap": n_bootstrap}

    for i, cohort in enumerate(cohorts):
        for population_index, population in enumerate(populations):
            for effect_index, effect in enumerate(effects):
                effect_rows.append({"task": task, "cohort": cohort, "population": population,
                    "metric": effect, **interval_record(point_effects[i, population_index, effect_index], effect_samples[:, i, population_index, effect_index])})
        for metric_index, metric in enumerate(metric_names):
            for j, condition in enumerate(["original", "masked"]):
                metric_rows.append({"task": task, "cohort": cohort, "condition": condition, "metric": metric,
                    **interval_record(point_metrics[i, j, metric_index], metric_samples[:, i, j, metric_index])})
            metric_rows.append({"task": task, "cohort": cohort, "condition": "masked_minus_original", "metric": metric,
                **interval_record(point_metrics[i, 1, metric_index] - point_metrics[i, 0, metric_index],
                                  metric_samples[:, i, 1, metric_index] - metric_samples[:, i, 0, metric_index])})
    gaps = point_metrics[0] - point_metrics[1]
    gap_samples = metric_samples[:, 0] - metric_samples[:, 1]
    for metric_index, metric in enumerate(metric_names):
        for j, condition in enumerate(["original", "masked"]):
            gap_rows.append({"task": task, "contrast": "MHC1_minus_MHC0", "condition": condition, "metric": metric,
                **interval_record(gaps[j, metric_index], gap_samples[:, j, metric_index])})
        gap_rows.append({"task": task, "contrast": "MHC1_minus_MHC0", "condition": "change_in_gap_after_masking", "metric": metric,
            **interval_record(gaps[1, metric_index] - gaps[0, metric_index],
                              gap_samples[:, 1, metric_index] - gap_samples[:, 0, metric_index])})
    destination = folder / "paired_analysis/uncertainty"
    destination.mkdir(exist_ok=True)
    pd.DataFrame(effect_rows).to_csv(destination / "probability_changes_with_ci.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(destination / "cohort_metrics_and_changes_with_ci.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(destination / "between_cohort_gaps_and_changes_with_ci.csv", index=False)
    np.savez_compressed(destination / "bootstrap_replicates.npz", metric_samples=metric_samples,
        effect_samples=effect_samples, cohorts=np.array(cohorts), conditions=np.array(["original", "masked"]),
        metric_names=np.array(metric_names), populations=np.array(populations), effects=np.array(effects))
    block_sizes = data.groupby("bootstrap_block_id").pair_id.nunique()
    (destination / "manifest.json").write_text(json.dumps({
        "task": task, "source_masking_scores_sha256": hashlib.sha256((folder / "admission_masking_effects.csv").read_bytes()).hexdigest(),
        "masking_mode": run_manifest.get("masking_mode", "broad"),
        "n_pairs": data.pair_id.nunique(), "n_blocks": int(n_blocks), "bootstrap_samples": n_bootstrap, "seed": seed,
        "largest_block_fraction_of_pairs": float(block_sizes.max() / block_sizes.sum()),
        "method": "95% pointwise percentile connected-patient/pair-block bootstrap; both conditions and cohorts use identical resampling weights",
        "blocks_verified_against_primary_fairness": True,
        "direction": "Within-cohort effect = masked-original; cohort gap = MHC1-MHC0; change in gap = masked gap-original gap",
        "metric_units": "AUROC/AUPRC and classification rates are on 0-1 scale; names ending _pct and _percentage_points use 0-100 scale; probability changes use percentage points",
        "conditioning": "Frozen models, thresholds, original matching and selected complete pairs; no training or model-selection uncertainty",
        "baseline": "Original probabilities recomputed within the masking run, so small numerical differences from earlier inference are not counted as masking effects",
        "multiplicity": "Intervals are pointwise, not simultaneous or adjusted for multiple comparisons; no p values",
        "scope": "Primary performance/gap comparisons use all complete-pair admissions. Visible-hit effect summaries condition on different keyword-positive populations and do not isolate differences in response to identical text.",
        "model_or_api_calls": False,
    }, indent=2) + "\n")
    print(pd.DataFrame(effect_rows).query("population == 'all_admissions'").to_string(index=False), flush=True)
    print(f"Saved masking uncertainty: {destination}", flush=True)


if __name__ == "__main__":
    main()
