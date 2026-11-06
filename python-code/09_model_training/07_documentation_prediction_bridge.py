"""Exploratory documentation/error associations and targeted vs random masking.

Offline numeric outputs only; no note text, training, API requests or inference.
Primary population: original complete held-out pairs, separately for each task.
Random results are means of metrics calculated separately for each repeat,
never metrics calculated from averaged prediction probabilities.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent
KEY = ["pair_id", "cohort", "subject_id", "hadm_id"]
COHORTS = ["MHC0", "MHC1_psychotic"]
MODES = ["broad", "diagnostic_labels", "sl"]
KEYWORDS = PYTHON / "03_discharge_note_text_analysis/02_keyword_matching"
SL_PATH = KEYWORDS / "analysis_output_keyword_summary_excluding_chief_complaint/keyword_family_note_level_hits_excluding_chief_complaint.csv"
DO_PATH = PYTHON / ("06_classification/01_classification_diag_overshadowing/"
    "output_prompt_B_semantic_verification_gemma/semantic_judgement_summary/combined_labels_with_verification_status.csv")


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def single_run(pattern):
    found = sorted(HERE.glob(pattern + "/run_manifest.json"))
    if len(found) != 1:
        raise ValueError(f"Expected exactly one completed run for {pattern}, got {found}")
    return found[0].parent


def metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    d = p >= threshold
    return {"auroc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "auprc": average_precision_score(y, p) if y.sum() else np.nan,
            "f1": f1_score(y, d, zero_division=0),
            "precision": precision_score(y, d, zero_division=0),
            "sensitivity": d[y == 1].mean() if (y == 1).any() else np.nan,
            "false_positive_rate": d[y == 0].mean() if (y == 0).any() else np.nan,
            "accuracy": accuracy_score(y, d), "brier_score": brier_score_loss(y, p),
            "observed_minus_predicted_pp": 100 * (y.mean() - p.mean()),
            "mean_predicted_probability": p.mean(), "outcome_prevalence": y.mean()}


def load_run(folder, paired, task, threshold, baseline=None):
    manifest = json.loads((folder / "run_manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("max_rows") is not None:
        raise ValueError(f"Incomplete or smoke-test run: {folder}")
    if manifest["task"] != task or manifest["threshold"] != threshold:
        raise ValueError(f"Task/threshold mismatch: {folder}")
    data = pd.read_csv(folder / "admission_masking_effects.csv")
    if len(data) != manifest["n_admissions"] or data.hadm_id.duplicated().any():
        raise ValueError(f"Invalid source counts/IDs: {folder}")
    selected = paired[KEY].merge(data, on=KEY, how="left", validate="one_to_one", indicator=True)
    if not selected._merge.eq("both").all():
        raise ValueError(f"Missing complete-pair admissions: {folder}")
    selected = selected.drop(columns="_merge")
    if not np.array_equal(selected.true_label, paired.true_label):
        raise ValueError(f"Outcome mismatch: {folder}")
    for condition in ["original", "masked"]:
        p = selected[f"{condition}_probability"]
        if not p.between(0, 1).all() or not np.array_equal(selected[f"{condition}_label"], p.ge(threshold).astype(int)):
            raise ValueError(f"Invalid probabilities/threshold decisions: {folder}")
    if not np.allclose(selected.probability_change_pp,
                       100 * (selected.masked_probability - selected.original_probability), atol=1e-10):
        raise ValueError(f"Invalid probability deltas: {folder}")
    if not selected.loc[selected.n_masked_tokens.eq(0), "probability_change_pp"].eq(0).all():
        raise ValueError(f"Keyword-free admission changed: {folder}")
    if baseline is not None and not np.array_equal(selected.original_probability, baseline):
        raise ValueError(f"Same-run baselines differ between conditions: {folder}")
    return selected, manifest


def error_associations(task, paired, sl_run, sl_notes, do_labels):
    """Original frozen-test errors, stratified by actual outcome and cohort.

    Within each cohort there is only one member of each matched pair; patient
    clustering retains dependence from repeat admissions. These are exploratory
    models with raw p-values, not a confirmatory family of significance tests.
    """
    data = paired[KEY + ["true_label", "predicted_label", "predicted_probability"]].merge(
        sl_notes, on=["cohort", "subject_id", "hadm_id"], how="left", validate="one_to_one")
    if data.full_note_SL.isna().any() or data.note_words.le(0).any():
        raise ValueError("Missing full-note SL summary/length")
    data["visible_SL"] = sl_run.n_masked_tokens.gt(0).astype(int).to_numpy()
    data["visible_tokens"] = sl_run.n_visible_text_tokens.to_numpy()
    data = data.merge(do_labels, on=KEY, how="left", validate="one_to_one", indicator=True)
    data["DO_status"] = np.where(data._merge.eq("left_only"), "not_classified",
                                data.semantic_accepted_label.replace({"": "unresolved"}))
    data.loc[data.cohort.eq("MHC0"), "DO_status"] = "not_applicable"
    data = data.drop(columns=["_merge", "semantic_accepted_label"])
    data["DO_positive"] = data.DO_status.map({"positive": 1., "negative": 0.})
    data["error"] = data.predicted_label.ne(data.true_label).astype(int)
    data["error_type"] = np.select(
        [data.true_label.eq(0) & data.predicted_label.eq(0),
         data.true_label.eq(0) & data.predicted_label.eq(1),
         data.true_label.eq(1) & data.predicted_label.eq(0)], ["TN", "FP", "FN"], default="TP")
    data["task"] = task
    rows, fits, coverage = [], [], []
    for cohort in COHORTS:
        cohort_rows = data.loc[data.cohort.eq(cohort)]
        for status, group in cohort_rows.groupby("DO_status"):
            coverage.append({"task": task, "cohort": cohort, "DO_status": status,
                             "n_admissions": len(group), "n_subjects": group.subject_id.nunique()})
        for exposure in ["full_note_SL", "visible_SL", "DO_positive"]:
            if exposure == "DO_positive" and cohort == "MHC0":
                continue
            for outcome in [0, 1]:
                part = cohort_rows.loc[cohort_rows.true_label.eq(outcome)].dropna(subset=[exposure]).copy()
                error_type = "false_positive" if outcome == 0 else "false_negative"
                for value, group in part.groupby(exposure):
                    rows.append({"task": task, "cohort": cohort, "exposure": exposure,
                        "exposure_positive": int(value), "actual_outcome": outcome,
                        "error_type": error_type, "n_admissions": len(group),
                        "n_subjects": group.subject_id.nunique(), "n_errors": int(group.error.sum()),
                        "error_rate": group.error.mean()})
                length = "visible_tokens" if exposure == "visible_SL" else "note_words"
                part["log_length"] = np.log(part[length])
                for name, variables in [("unadjusted", [exposure]), ("length_adjusted", [exposure, "log_length"])]:
                    row = {"task": task, "cohort": cohort, "exposure": exposure,
                           "error_type": error_type, "model": name, "length_covariate": length,
                           "n_admissions": len(part), "n_subject_clusters": part.subject_id.nunique()}
                    if part[exposure].nunique() != 2 or part.error.nunique() != 2:
                        fits.append({**row, "status": "insufficient_variation"})
                        continue
                    try:
                        fit = sm.GLM(part.error, sm.add_constant(part[variables], has_constant="add"),
                            family=sm.families.Binomial()).fit(cov_type="cluster",
                            cov_kwds={"groups": part.subject_id, "use_correction": True})
                        low, high = fit.conf_int().loc[exposure]
                        fits.append({**row, "status": "converged" if fit.converged else "not_converged",
                            "coefficient": fit.params[exposure], "cluster_robust_se": fit.bse[exposure],
                            "odds_ratio": np.exp(fit.params[exposure]), "ci_lower": np.exp(low),
                            "ci_upper": np.exp(high), "p_value_unadjusted": fit.pvalues[exposure]})
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        fits.append({**row, "status": "fit_failed", "message": str(exc)})
    return data, pd.DataFrame(rows), pd.DataFrame(fits), pd.DataFrame(coverage)


def bootstrap_contrasts(paired, contributions, n_samples, seed, fairness):
    """Block-bootstrap targeted minus mean-random effects for linear metrics.

    Random draws are held fixed. Brier/errors are calculated for EACH repeat
    before averaging, preserving the interpretation of random-mask performance.
    AUROC/AP/F1 are reported descriptively per repeat, not approximated here.
    """
    blocks = fairness.patient_pair_blocks(paired)
    if not np.array_equal(blocks, paired.bootstrap_block_id):
        raise ValueError("Patient/pair blocks differ from the frozen fairness analysis")
    n_blocks = int(blocks.max()) + 1
    rng = np.random.default_rng(seed)
    # Reuse exactly the fairness procedure: n_blocks equal-probability draws.
    draws = np.zeros((n_samples, n_blocks), dtype=np.int32)
    for i in range(n_samples):
        draws[i] = np.bincount(rng.integers(0, n_blocks, size=n_blocks), minlength=n_blocks)
    rows = []
    for mode, metric, numerator, denominator in contributions:
        samples_by_cohort = {}
        estimates = {}
        for cohort in COHORTS:
            use = paired.cohort.eq(cohort).to_numpy()
            num = np.bincount(blocks, weights=numerator * use, minlength=n_blocks)
            den = np.bincount(blocks, weights=denominator * use, minlength=n_blocks)
            sample_den = draws @ den
            sample = np.divide(draws @ num, sample_den, out=np.full(n_samples, np.nan), where=sample_den > 0)
            estimate = num.sum() / den.sum()
            lo, hi, valid = fairness.interval_summary(sample)
            rows.append({"mode": mode, "cohort": cohort, "metric": metric, "target_minus_mean_random": estimate,
                         "ci_lower": lo, "ci_upper": hi, "valid_bootstraps": valid})
            samples_by_cohort[cohort] = sample
            estimates[cohort] = estimate
        gap = samples_by_cohort["MHC1_psychotic"] - samples_by_cohort["MHC0"]
        lo, hi, valid = fairness.interval_summary(gap)
        rows.append({"mode": mode, "cohort": "MHC1_minus_MHC0_gap", "metric": metric,
                     "target_minus_mean_random": estimates["MHC1_psychotic"] - estimates["MHC0"],
                     "ci_lower": lo, "ci_upper": hi, "valid_bootstraps": valid})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "analysis_output_documentation_prediction_bridge")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("Use at least 100 bootstrap samples")
    bundle = json.loads((args.bundle_dir / "bundle_manifest.json").read_text())
    if bundle.get("status") != "complete" or len(bundle["analyses"]) != 8:
        raise ValueError("Expected the complete eight-analysis bundle")
    sl_notes = pd.read_csv(SL_PATH, usecols=["keyword_family", "cohort", "subject_id", "hadm_id",
                                          "note_word_count_excluding_chief_complaint", "n_keyword_hits"])
    sl_notes = sl_notes.loc[sl_notes.keyword_family.eq("SL")].drop(columns="keyword_family")
    sl_notes = sl_notes.rename(columns={"note_word_count_excluding_chief_complaint": "note_words"})
    sl_notes["full_note_SL"] = sl_notes.n_keyword_hits.gt(0).astype(int)
    do_labels = pd.read_csv(DO_PATH, keep_default_na=False, usecols=KEY + ["semantic_accepted_label"])
    if not do_labels.semantic_accepted_label.isin(["positive", "negative", "unclear", ""]).all():
        raise ValueError("Unexpected overshadowing labels")
    fairness = module(HERE / "05_fairness_analysis.py")
    raw, effects, bridges, association_rows, model_rows, coverages, contrasts = [], [], [], [], [], [], []
    audit = {"bundle": str(args.bundle_dir), "tasks": {}}
    sources = {SL_PATH, DO_PATH, args.bundle_dir / "bundle_manifest.json"}
    for task, threshold in [("los", .25), ("readmission", .20)]:
        paired_path = HERE / f"bioclinicalbert_{task}_classifier_inference_output_current_cohort/paired_fairness_analysis/paired_test_predictions.csv"
        paired = pd.read_csv(paired_path)
        if paired.hadm_id.duplicated().any() or not paired.groupby("pair_id").size().eq(2).all() or not paired.groupby("pair_id").cohort.nunique().eq(2).all():
            raise ValueError("Invalid complete-pair membership")
        sources.add(paired_path)
        folders = {"sl": args.bundle_dir / task / "sl",
                   "broad": single_run(f"analysis_output_interpretability_{task}_*"),
                   "diagnostic_labels": single_run(f"analysis_output_interpretability_diagnostic_labels_{task}_*")}
        sl_run, sl_manifest = load_run(folders["sl"], paired, task, threshold)
        baseline = sl_run.original_probability.to_numpy()
        y = paired.true_label.to_numpy(dtype=int)
        old_delta = abs(paired.predicted_probability.to_numpy() - baseline)
        audit["tasks"][task] = {"n_pairs": int(paired.pair_id.nunique()), "n_admissions": len(paired),
            "old_fairness_baseline_max_probability_difference": float(old_delta.max()),
            "old_fairness_baseline_threshold_disagreements": int(np.sum(paired.predicted_label.to_numpy() != (baseline >= threshold)))}
        bridge, desc, models, coverage = error_associations(task, paired, sl_run, sl_notes, do_labels)
        bridges.append(bridge); association_rows.append(desc); model_rows.append(models); coverages.append(coverage)
        for cohort in COHORTS:
            use = paired.cohort.eq(cohort).to_numpy()
            raw.append({"task": task, "condition": "original", "repeat": 0, "cohort": cohort,
                        "n_admissions": int(use.sum()), **metrics(y[use], baseline[use], threshold)})
        contributions = []
        for mode in MODES:
            target, manifest = load_run(folders[mode], paired, task, threshold, baseline)
            if manifest["training_config"] != sl_manifest["training_config"]:
                raise ValueError("Checkpoint configuration mismatch")
            if mode == "sl" and any("agitat" in p for ps in manifest["keyword_patterns"].values() for p in ps):
                raise ValueError("SL mask still includes agitation/agitated")
            sources.update([folders[mode] / "run_manifest.json", folders[mode] / "admission_masking_effects.csv"])
            random_folders = sorted((args.bundle_dir / task / f"random_{mode}").glob("repeat_*"))
            if len(random_folders) != bundle["random_repeats"]:
                raise ValueError("Missing random repeats")
            random_probabilities, random_audits = [], []
            for repeat, folder in enumerate(random_folders, start=1):
                run, rm = load_run(folder, paired, task, threshold, baseline)
                if (rm.get("target_masking_mode") != mode or rm.get("random_repeat") != repeat
                        or rm["keyword_patterns"] != manifest["keyword_patterns"]
                        or rm.get("bundle_seed") != bundle["seed"]
                        or rm.get("random_repeats") != bundle["random_repeats"]):
                    raise ValueError("Random target vocabulary/repeat mismatch")
                if not np.array_equal(target.n_masked_tokens, run.n_masked_tokens):
                    raise ValueError("Random token budget differs from target")
                if not np.array_equal(target.n_visible_text_tokens, run.n_visible_text_tokens):
                    raise ValueError("Random run uses a different visible input length")
                if not run.n_random_tokens_outside_target_section.between(0, run.n_masked_tokens).all():
                    raise ValueError("Invalid random section-fallback counts")
                random_probabilities.append(run.masked_probability.to_numpy())
                random_audits.append(run[["n_random_span_fallbacks", "n_random_tokens_outside_target_section"]].to_numpy())
                sources.update([folder / "run_manifest.json", folder / "admission_masking_effects.csv"])
            random_p = np.asarray(random_probabilities)
            target_p = target.masked_probability.to_numpy()
            random_d, target_d = random_p >= threshold, target_p >= threshold
            for condition, repeats in [(mode, target_p[None, :]), (f"random_{mode}", random_p)]:
                for repeat, probability in enumerate(repeats, start=1):
                    for cohort in COHORTS:
                        use = paired.cohort.eq(cohort).to_numpy()
                        raw.append({"task": task, "condition": condition, "repeat": repeat, "cohort": cohort,
                                    "n_admissions": int(use.sum()), **metrics(y[use], probability[use], threshold)})
                        delta = 100 * (probability[use] - baseline[use])
                        correct_before = (baseline[use] >= threshold) == y[use]
                        correct_after = (probability[use] >= threshold) == y[use]
                        effects.append({"task": task, "condition": condition, "repeat": repeat, "cohort": cohort,
                            "n_visible_keyword_admissions": int(target.n_masked_tokens.to_numpy()[use].astype(bool).sum()),
                            "n_admissions": int(use.sum()), "mean_probability_change_pp": delta.mean(),
                            "mean_absolute_probability_change_pp": abs(delta).mean(),
                            "incorrect_to_correct": int((~correct_before & correct_after).sum()),
                            "correct_to_incorrect": int((correct_before & ~correct_after).sum()),
                            "mean_section_fallback_tokens": float(np.mean([a[use, 1].sum() for a in random_audits])) if condition.startswith("random_") else 0})
            # Exact mean-random metric contributions, not an ensemble prediction.
            contributions.extend([
                (mode, "mean_probability_change_pp", 100 * (target_p - random_p.mean(axis=0)), np.ones(len(y))),
                (mode, "mean_absolute_probability_change_pp", 100 * (abs(target_p - baseline) - abs(random_p - baseline).mean(axis=0)), np.ones(len(y))),
                (mode, "brier_score", (target_p - y) ** 2 - ((random_p - y) ** 2).mean(axis=0), np.ones(len(y))),
                (mode, "accuracy", (target_d == y).astype(float) - (random_d == y).mean(axis=0), np.ones(len(y))),
                (mode, "false_positive_rate", (target_d.astype(float) - random_d.mean(axis=0)) * (y == 0), (y == 0).astype(float)),
                (mode, "false_negative_rate", (random_d.mean(axis=0) - target_d.astype(float)) * (y == 1), (y == 1).astype(float))])
            for cohort in COHORTS:
                use = paired.cohort.eq(cohort).to_numpy()
                n = int(use.sum())
                target_errors = target_d[use] != y[use]
                baseline_errors = (baseline[use] >= threshold) != y[use]
                for exposure in ["all", "full_note_SL", "visible_SL", "DO_positive", "DO_negative"]:
                    if exposure == "all":
                        subgroup = np.ones(n, dtype=bool)
                    elif exposure == "DO_negative":
                        subgroup = bridge.loc[use, "DO_positive"].eq(0).to_numpy()
                    else:
                        subgroup = bridge.loc[use, exposure].eq(1).to_numpy()
                    if not subgroup.any():
                        continue
                    selected_indices = np.flatnonzero(use)[subgroup]
                    effects.append({"task": task, "condition": mode + "_subgroup", "repeat": 0, "cohort": cohort,
                        "subgroup": exposure, "n_admissions": int(subgroup.sum()),
                        "mean_probability_change_pp": 100 * (target_p[selected_indices] - baseline[selected_indices]).mean(),
                        "mean_absolute_probability_change_pp": 100 * abs(target_p[selected_indices] - baseline[selected_indices]).mean(),
                        "incorrect_to_correct": int((baseline_errors[subgroup] & ~target_errors[subgroup]).sum()),
                        "correct_to_incorrect": int((~baseline_errors[subgroup] & target_errors[subgroup]).sum())})
                    random_errors = random_d[:, selected_indices] != y[selected_indices]
                    effects.append({"task": task, "condition": f"random_{mode}_subgroup", "repeat": 0,
                        "cohort": cohort, "subgroup": exposure, "n_admissions": int(subgroup.sum()),
                        "summary": "mean over separately evaluated random repeats",
                        "mean_probability_change_pp": 100 * (random_p[:, selected_indices] - baseline[selected_indices]).mean(),
                        "mean_absolute_probability_change_pp": 100 * abs(random_p[:, selected_indices] - baseline[selected_indices]).mean(),
                        "incorrect_to_correct": float((baseline_errors[subgroup] & ~random_errors).sum(axis=1).mean()),
                        "correct_to_incorrect": float((~baseline_errors[subgroup] & random_errors).sum(axis=1).mean())})
        comparison = bootstrap_contrasts(paired, contributions, args.bootstrap_samples, args.seed, fairness)
        comparison.insert(0, "task", task); contrasts.append(comparison)
        print(f"{task}: validated {len(paired):,} admissions, all targeted masks and {bundle['random_repeats']} repeats per random control", flush=True)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(raw)
    raw.to_csv(out / "performance_each_run.csv", index=False)
    metric_cols = [c for c in raw if c not in ["task", "condition", "repeat", "cohort", "n_admissions"]]
    grouped = raw.groupby(["task", "condition", "cohort"], sort=False)
    averaged = grouped[metric_cols].mean().reset_index()
    averaged["n_runs"] = grouped.size().to_numpy()
    averaged["n_admissions"] = grouped.n_admissions.first().to_numpy()
    averaged.to_csv(out / "performance_summary.csv", index=False)
    grouped[metric_cols].agg(["min", "max", "std"]).to_csv(out / "performance_repeat_variation.csv")
    pd.DataFrame(effects).to_csv(out / "masking_effects_and_error_changes.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(out / "target_minus_random_with_ci.csv", index=False)
    pd.concat(bridges, ignore_index=True).to_csv(out / "test_admission_bridge.csv", index=False)
    pd.concat(association_rows, ignore_index=True).to_csv(out / "documentation_error_rates.csv", index=False)
    pd.concat(model_rows, ignore_index=True).to_csv(out / "documentation_error_associations.csv", index=False)
    pd.concat(coverages, ignore_index=True).to_csv(out / "overshadowing_test_coverage.csv", index=False)
    report = [f"Values are MHC0 / MHC1-psychosis. Random rows average {bundle['random_repeats']} separately evaluated masks; these are not confidence intervals.", ""]
    for task in ["los", "readmission"]:
        report.extend([task.upper(), "", "| Mask | AUROC | F1 | Brier |", "|---|---:|---:|---:|"])
        for condition in ["original", "broad", "random_broad", "diagnostic_labels", "random_diagnostic_labels", "sl", "random_sl"]:
            rows = averaged.loc[averaged.task.eq(task) & averaged.condition.eq(condition)].set_index("cohort")
            values = [" / ".join(f"{rows.loc[c, metric]:.4f}" for c in COHORTS) for metric in ["auroc", "f1", "brier_score"]]
            report.append("| " + " | ".join([condition] + values) + " |")
        report.append("")
    (out / "performance_tables.md").write_text("\n".join(report) + "\n")
    audit.update(status="complete", bootstrap_samples=args.bootstrap_samples, seed=args.seed,
        error_associations="Frozen original test predictions; actual-outcome strata; within-cohort patient-clustered logistic SEs; exploratory unadjusted p-values",
        masking_performance="Common recomputed original baseline; complete held-out pairs; frozen thresholds; random metrics averaged after separate evaluation",
        uncertainty="Patient/pair connected-component percentile bootstrap; target minus mean of fixed random repeats; pointwise 95% CI; no multiplicity correction",
        bootstrap_metrics="Linear metrics only; AUROC, AP and F1 remain descriptive in this script",
        exposure_scope="Full-note SL excludes chief complaint; visible SL uses only truncated prediction input; DO uses six classification sections",
        caution="Associations and masking do not establish causal effects, stigma, clinical irrelevance, or independent validation of LLM labels",
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)})
    (out / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("\n".join(report))
    print(f"Saved bridge analysis: {out}")


if __name__ == "__main__":
    main()
