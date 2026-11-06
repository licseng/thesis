"""Exploratory SL keyword association with accepted possible overshadowing labels.

SL uses existing numeric outputs. A separate conservative diagnosis-based
person-labelling screen (schizophrenic/psychotic forms) processes parsed text
locally without emitting text.
SL covers the full parsed discharge note
excluding chief complaint, with agitation/agitated already removed. Unclassified,
unresolved and unclear admissions are described but excluded from the positive
versus negative comparison. Logistic SEs cluster by patient because only MHC1
admissions enter: there is no second member of a matched pair in this analysis.
"""
from pathlib import Path
import hashlib
import json
import importlib.util
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
PYTHON = HERE.parent.parent
OUTPUT = HERE / "analysis_output_SL_by_overshadowing"
SL = HERE / "analysis_output_keyword_summary_excluding_chief_complaint/keyword_family_note_level_hits_excluding_chief_complaint.csv"
LABELS = PYTHON / ("06_classification/01_classification_diag_overshadowing/"
    "output_prompt_B_semantic_verification_gemma/semantic_judgement_summary/"
    "combined_labels_with_verification_status.csv")
MATCHED = PYTHON / "02_cohort_matching/matched_cohort_output/matched_admission_ids_for_dbeaver.csv"
KEY = ["cohort", "subject_id", "hadm_id"]
WORDS = "note_word_count_excluding_chief_complaint"

# These are surface-form indicators, not contextual stigma judgements. Singular
# nouns require a determiner and a conservative following boundary so that
# 'a schizophrenic episode/patient' is not counted as a noun label.
PERSON_LABEL_PATTERNS = {
    "diagnostic_noun": [
        r"\bschizophrenics\b",
        r"\b(?:a|an|the|this|that)\s+(?:(?:chronic|paranoid|disorganized|catatonic|residual|undifferentiated)\s+)*schizophrenic\b(?=\s*(?:$|[.,;:!?)]|(?:who|whose|with|without|and|or|is|was|has|had)\b))",
        r"(?<![\w-])(?<!anti\s)psychotics\b",
        r"\b(?:a|an|the|this|that)\s+(?:(?:chronic|paranoid|acutely|chronically)\s+)*psychotic\b(?=\s*(?:$|[.,;:!?)]|(?:who|whose|with|without|is|was|has|had)\b))",
    ],
    "person_adjective": [
        r"\bschizophrenic[\s-]+(?:patients?|persons?|people|individuals?|men|women|man|woman|males?|females?|gentlem[ae]n|lad(?:y|ies)|pt)\b",
        r"(?<![\w-])(?<!anti\s)psychotic[\s-]+(?:patients?|persons?|people|individuals?|men|women|man|woman|males?|females?|gentlem[ae]n|lad(?:y|ies)|pt)\b",
    ],
}


def person_label_counts(text):
    """Return counts only; no snippets, quotations or surrounding text."""
    return {name: sum(len(list(re.finditer(p, text, re.I))) for p in patterns)
            for name, patterns in PERSON_LABEL_PATTERNS.items()}


def analyze_person_labelling(joined):
    parser_path = PYTHON / "01_discharge_note_preprocessing/01_discharge_note_parsing/02_parse_full_discharge_notes.py"
    spec = importlib.util.spec_from_file_location("note_parser_for_person_labels", parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    sections = [s for s in list(parser.CANONICAL_SECTIONS) + ["unsectioned_text"]
                if s != "chief_complaint"]
    source = parser_path.parent / "full_discharge_note_sections/MHC1_psychotic_matched_full_discharge_note_sections.parquet"
    notes = pd.read_parquet(source, columns=["subject_id", "hadm_id"] + sections)
    if notes.hadm_id.duplicated().any():
        raise ValueError("Duplicate parsed admissions")
    rows = []
    for row in notes.itertuples(index=False):
        for section in sections:
            value = getattr(row, section)
            text = "" if pd.isna(value) else str(value)
            counts = person_label_counts(text)
            rows.append({"subject_id": row.subject_id, "hadm_id": row.hadm_id,
                         "section": section, **counts,
                         "all_schizophrenic_word_hits": len(re.findall(r"\bschizophrenics?\b", text, re.I)),
                         "all_psychotic_word_hits": len(re.findall(r"\bpsychotics?\b", text, re.I))})
    section_counts = pd.DataFrame(rows)
    counts = section_counts.groupby(["subject_id", "hadm_id"], as_index=False)[list(PERSON_LABEL_PATTERNS) + ["all_schizophrenic_word_hits", "all_psychotic_word_hits"]].sum()
    data = joined.merge(counts, on=["subject_id", "hadm_id"], how="outer",
                        validate="one_to_one", indicator=True)
    if not data._merge.eq("both").all():
        raise ValueError("Parsed admissions do not exactly match bridge cohort")
    data = data.drop(columns="_merge")
    data["any_person_labelling"] = data[list(PERSON_LABEL_PATTERNS)].sum(axis=1)
    summaries, models = [], []
    for category in list(PERSON_LABEL_PATTERNS) + ["any_person_labelling"]:
        for group, frame in data.groupby("classification_group"):
            summaries.append({"category": category, "classification_group": group,
                "n_admissions": len(frame), "n_subjects": frame.subject_id.nunique(),
                "n_positive": int(frame[category].gt(0).sum()),
                "pct_positive": 100 * frame[category].gt(0).mean(),
                "total_hits": int(frame[category].sum())})
        comparison = data.loc[data.classification_group.isin(["positive", "negative"])].copy()
        comparison["overshadowing_positive"] = comparison.classification_group.eq("positive").astype(int)
        comparison["log_note_words"] = np.log(comparison[WORDS])
        for name, variables in [("unadjusted", ["overshadowing_positive"]),
                                ("note_length_adjusted", ["overshadowing_positive", "log_note_words"])]:
            result = {"category": category, "model": name,
                "term": "overshadowing_positive", "n_admissions": len(comparison),
                "n_subject_clusters": comparison.subject_id.nunique()}
            cells = pd.crosstab(comparison.overshadowing_positive, comparison[category].gt(0)).reindex(index=[0, 1], columns=[False, True], fill_value=0)
            result["minimum_cell_count"] = int(cells.to_numpy().min())
            # A descriptive endpoint is preferable when only a handful of
            # admissions contain the construction. Ten is a conservative
            # reporting gate here, not a guarantee of adequate model precision.
            if int(cells[True].sum()) < 10:
                result["status"] = "not_fitted_fewer_than_10_positive_admissions"
            elif (cells.to_numpy() == 0).any():
                result["status"] = "not_fitted_zero_cell"
            else:
                x = sm.add_constant(comparison[variables], has_constant="add")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit = sm.GLM(comparison[category].gt(0).astype(int), x,
                        family=sm.families.Binomial()).fit(cov_type="cluster",
                        cov_kwds={"groups": comparison.subject_id, "use_correction": True})
                term = "overshadowing_positive"
                ci = fit.conf_int().loc[term]
                result.update(odds_ratio=float(np.exp(fit.params[term])),
                    OR_CI_lower=float(np.exp(ci.iloc[0])), OR_CI_upper=float(np.exp(ci.iloc[1])),
                    p_value=float(fit.pvalues[term]), converged=bool(fit.converged),
                    warning_types="|".join(sorted({w.category.__name__ for w in caught})),
                    status="ok" if fit.converged and np.isfinite(fit.bse).all()
                        and not any(not issubclass(w.category, (DeprecationWarning, FutureWarning)) for w in caught)
                        else "fit_warning")
            models.append(result)
    summary = pd.DataFrame(summaries)
    fits = pd.DataFrame(models)
    data.to_csv(OUTPUT / "admission_level_person_labelling_and_overshadowing.csv", index=False)
    section_counts.to_csv(OUTPUT / "person_labelling_section_counts.csv", index=False)
    summary.to_csv(OUTPUT / "person_labelling_by_overshadowing_summary.csv", index=False)
    fits.to_csv(OUTPUT / "person_labelling_by_overshadowing_logistic_models.csv", index=False)
    qc = data.groupby("classification_group").agg(
        n_admissions=("hadm_id", "size"),
        n_with_schizophrenic_word=("all_schizophrenic_word_hits", lambda s: int(s.gt(0).sum())),
        total_schizophrenic_word_hits=("all_schizophrenic_word_hits", "sum"),
        n_with_psychotic_word=("all_psychotic_word_hits", lambda s: int(s.gt(0).sum())),
        total_psychotic_word_hits=("all_psychotic_word_hits", "sum"),
        n_with_person_labelling=("any_person_labelling", lambda s: int(s.gt(0).sum())))
    qc.to_csv(OUTPUT / "person_labelling_lexical_coverage_qc.csv")
    manifest = {"status": "complete", "patterns": PERSON_LABEL_PATTERNS,
        "scope": "Schizophrenic and psychotic person-label forms; full parsed note excluding chief complaint. Symptom/episode descriptions and predicate states such as 'patient is psychotic' are excluded.",
        "limitations": "Conservative surface forms, not exhaustive syntactic parsing; no negation, experiencer or quotation attribution; not proof of stigma. Shared text may cue the DO classifier.",
        "models": "Presence of each category versus accepted positive/negative DO; unadjusted and log-note-length adjusted, patient-clustered SE; fits skipped for fewer than 10 positive admissions or zero cells. Ten is a descriptive reporting gate, not an adequacy guarantee.",
        "inference": "Post-hoc exploratory extension; up to six related models with unadjusted p-values and pointwise confidence intervals, no multiplicity correction",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [source, LABELS, MATCHED, Path(__file__)]}}
    (OUTPUT / "person_labelling_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.loc[summary.classification_group.isin(["positive", "negative"])].to_string(index=False))
    print(fits.to_string(index=False))
    print(qc.to_string())


def main():
    base = pd.read_csv(MATCHED, usecols=["pair_id"] + KEY)
    base = base.loc[base.cohort.eq("MHC1_psychotic")].copy()
    sl = pd.read_csv(SL, usecols=KEY + ["keyword_family", WORDS, "n_keyword_hits"])
    sl = sl.loc[sl.cohort.eq("MHC1_psychotic") & sl.keyword_family.eq("SL")].drop(columns="keyword_family")
    labels = pd.read_csv(LABELS, keep_default_na=False,
                         usecols=["pair_id"] + KEY + ["semantic_accepted_label"])
    for name, data in [("cohort", base), ("SL", sl), ("labels", labels)]:
        if data.hadm_id.duplicated().any():
            raise ValueError(f"Duplicate {name} admissions")
    if not labels.semantic_accepted_label.isin(["positive", "negative", "unclear", ""]).all():
        raise ValueError("Unexpected classification labels")
    joined = base.merge(sl, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("SL outputs do not exactly cover current matched MHC1 admissions")
    joined = joined.drop(columns="_merge").merge(labels, on=["pair_id"] + KEY,
        how="outer", validate="one_to_one", indicator=True)
    if joined._merge.eq("right_only").any():
        raise ValueError("Classifications are outside the matched MHC1 cohort")
    joined["classification_group"] = np.where(joined._merge.eq("left_only"), "not_classified",
        joined.semantic_accepted_label.replace({"": "unresolved"}))
    joined = joined.drop(columns="_merge")
    if joined[[WORDS, "n_keyword_hits"]].isna().any().any() or joined[WORDS].le(0).any():
        raise ValueError("Missing keyword data or nonpositive note word counts")
    joined["SL_keyword_positive"] = joined.n_keyword_hits.gt(0).astype(int)
    joined["SL_hits_per_1000_words"] = 1000 * joined.n_keyword_hits / joined[WORDS]
    summary = []
    for group, data in joined.groupby("classification_group", sort=True):
        summary.append({"classification_group": group, "n_admissions": len(data),
            "n_subjects": data.subject_id.nunique(),
            "n_SL_keyword_positive": int(data.SL_keyword_positive.sum()),
            "pct_SL_keyword_positive": 100 * data.SL_keyword_positive.mean(),
            "total_SL_hits": int(data.n_keyword_hits.sum()),
            "mean_SL_hits_per_admission": data.n_keyword_hits.mean(),
            "median_SL_hits_per_admission": data.n_keyword_hits.median(),
            "mean_note_words": data[WORDS].mean(), "median_note_words": data[WORDS].median(),
            "aggregate_SL_hits_per_1000_words": 1000 * data.n_keyword_hits.sum() / data[WORDS].sum(),
            "mean_SL_hits_per_1000_words": data.SL_hits_per_1000_words.mean()})
    summary = pd.DataFrame(summary)
    comparison = joined.loc[joined.classification_group.isin(["positive", "negative"])].copy()
    if comparison.classification_group.nunique() != 2:
        raise ValueError("Both positive and negative admissions are required")
    comparison["overshadowing_positive"] = comparison.classification_group.eq("positive").astype(int)
    comparison["log_note_words"] = np.log(comparison[WORDS])
    fits = []
    for name, variables in [("unadjusted", ["overshadowing_positive"]),
                            ("note_length_adjusted", ["overshadowing_positive", "log_note_words"])]:
        x = sm.add_constant(comparison[variables], has_constant="add")
        model = sm.GLM(comparison.SL_keyword_positive, x, family=sm.families.Binomial())
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": comparison.subject_id,
                        "use_correction": True})
        intervals = fit.conf_int()
        for term in x.columns:
            fits.append({"model": name, "term": term, "coefficient": fit.params[term],
                "cluster_robust_se": fit.bse[term], "odds_ratio": np.exp(fit.params[term]),
                "OR_CI_lower": np.exp(intervals.loc[term, 0]),
                "OR_CI_upper": np.exp(intervals.loc[term, 1]), "p_value": fit.pvalues[term],
                "n_admissions": len(comparison), "n_subject_clusters": comparison.subject_id.nunique(),
                "converged": bool(fit.converged)})
    fits = pd.DataFrame(fits)
    if not fits.converged.all() or not np.isfinite(fits[["coefficient", "cluster_robust_se"]]).all().all():
        raise RuntimeError("The exploratory logistic models did not produce finite converged fits")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    joined.to_csv(OUTPUT / "admission_level_SL_and_overshadowing.csv", index=False)
    summary.to_csv(OUTPUT / "SL_by_overshadowing_summary.csv", index=False)
    fits.to_csv(OUTPUT / "SL_by_overshadowing_logistic_models.csv", index=False)
    manifest = {"status": "complete", "design": "Exploratory association proposed after previous analyses",
        "SL_scope": "All parsed discharge note sections excluding chief complaint; agitation/agitated excluded",
        "DO_scope": "Accepted prompt B labels after quotation/semantic verification; six-section classification",
        "comparison": "Accepted positive versus negative possible diagnostic overshadowing within matched MHC1-psychosis",
        "excluded_from_comparison": "Unclassified, unresolved, unclear; none treated as negative",
        "outcome": "At least one candidate SL keyword", "effect": "Odds ratio for SL keyword presence, positive versus negative DO label",
        "models": "Unadjusted and adjusted for log note word count; patient-clustered SEs",
        "matching": "MHC0 pair partners do not enter this within-MHC1 analysis",
        "p_values": "Two exploratory association models; unadjusted p-values, no multiplicity correction",
        "interpretation": "Shared text and LLM cues may contribute; this is not independent validation, causality, or proof of stigmatizing context",
        "n_MHC1_admissions": len(joined), "n_comparison_admissions": len(comparison),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [SL, LABELS, MATCHED, HERE / "02_explore_SL_keywords_by_section.py"]}}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(fits.loc[fits.term.eq("overshadowing_positive")].to_string(index=False))
    print(f"Saved: {OUTPUT}")
    analyze_person_labelling(joined)


if __name__ == "__main__":
    main()
